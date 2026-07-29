"""
Cortex Resilience — circuit breaker, retry, fallback, graceful degradation.

The "fault tolerance + watchdog" layer of the agent microkernel.

Components:
    CircuitBreaker   Per-backend circuit breaker (closed → open → half-open)
    RetryPolicy      Configurable retry with exponential backoff
    Fallback         Tier waterfall when the target backend fails
    ResilienceLayer  Wraps everything into a single call interface

Design:
    - Each backend (identified by model or tier) gets its own circuit breaker
    - After N consecutive failures, the circuit opens (rejects calls immediately)
    - After a cooldown, it enters half-open (allows one probe request)
    - If the probe succeeds, circuit closes; if it fails, stays open
    - The resilience layer wraps Cortex.process() with retry + fallback + breaker
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, TypeVar

logger = logging.getLogger("cortex.resilience")

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Circuit Breaker
# ---------------------------------------------------------------------------

class CircuitState(str, Enum):
    CLOSED = "closed"         # normal — requests flow through
    OPEN = "open"             # tripped — requests rejected immediately
    HALF_OPEN = "half_open"   # probe — one request allowed to test recovery


@dataclass
class CircuitStats:
    """Per-circuit statistics."""
    total_calls: int = 0
    total_successes: int = 0
    total_failures: int = 0
    consecutive_failures: int = 0
    last_failure_time: float = 0.0
    last_success_time: float = 0.0
    times_opened: int = 0


class CircuitBreaker:
    """
    Per-backend circuit breaker.
    
    Args:
        name: identifier (e.g. model name or tier)
        failure_threshold: consecutive failures before opening
        recovery_timeout: seconds to wait before half-open probe
        success_threshold: successes in half-open before closing
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 3,
        recovery_timeout: float = 30.0,
        success_threshold: int = 1,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self.state = CircuitState.CLOSED
        self.stats = CircuitStats()
        self._half_open_successes = 0

    def can_execute(self) -> bool:
        """Check if a request is allowed through the circuit."""
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            # Check if recovery timeout has elapsed
            elapsed = time.monotonic() - self.stats.last_failure_time
            if elapsed >= self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                self._half_open_successes = 0
                logger.info("Circuit %s: OPEN → HALF_OPEN (probing)", self.name)
                return True
            return False

        if self.state == CircuitState.HALF_OPEN:
            # Allow one probe at a time
            return True

        return False

    def record_success(self) -> None:
        """Record a successful call."""
        self.stats.total_calls += 1
        self.stats.total_successes += 1
        self.stats.consecutive_failures = 0
        self.stats.last_success_time = time.monotonic()

        if self.state == CircuitState.HALF_OPEN:
            self._half_open_successes += 1
            if self._half_open_successes >= self.success_threshold:
                self.state = CircuitState.CLOSED
                logger.info("Circuit %s: HALF_OPEN → CLOSED (recovered)", self.name)

        elif self.state == CircuitState.OPEN:
            # Shouldn't happen, but reset
            self.state = CircuitState.CLOSED

    def record_failure(self, error: Optional[str] = None) -> None:
        """Record a failed call."""
        self.stats.total_calls += 1
        self.stats.total_failures += 1
        self.stats.consecutive_failures += 1
        self.stats.last_failure_time = time.monotonic()

        if self.state == CircuitState.HALF_OPEN:
            # Probe failed — back to open
            self.state = CircuitState.OPEN
            self.stats.times_opened += 1
            logger.warning(
                "Circuit %s: HALF_OPEN → OPEN (probe failed: %s)", self.name, error
            )

        elif self.state == CircuitState.CLOSED:
            if self.stats.consecutive_failures >= self.failure_threshold:
                self.state = CircuitState.OPEN
                self.stats.times_opened += 1
                logger.warning(
                    "Circuit %s: CLOSED → OPEN (%d consecutive failures)",
                    self.name, self.stats.consecutive_failures,
                )

    def reset(self) -> None:
        """Force reset the circuit to closed."""
        self.state = CircuitState.CLOSED
        self.stats.consecutive_failures = 0
        self._half_open_successes = 0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_threshold": self.failure_threshold,
            "recovery_timeout": self.recovery_timeout,
            "stats": {
                "total_calls": self.stats.total_calls,
                "total_successes": self.stats.total_successes,
                "total_failures": self.stats.total_failures,
                "consecutive_failures": self.stats.consecutive_failures,
                "times_opened": self.stats.times_opened,
            },
        }


# ---------------------------------------------------------------------------
# Retry Policy
# ---------------------------------------------------------------------------

@dataclass
class RetryPolicy:
    """Configurable retry with exponential backoff."""
    max_retries: int = 2
    base_delay: float = 0.5         # seconds
    max_delay: float = 10.0         # seconds
    exponential_base: float = 2.0
    retryable_exceptions: tuple = (Exception,)

    def delay_for_attempt(self, attempt: int) -> float:
        """Calculate delay for a given attempt number (0-indexed)."""
        delay = self.base_delay * (self.exponential_base ** attempt)
        return min(delay, self.max_delay)


# ---------------------------------------------------------------------------
# Resilience Layer
# ---------------------------------------------------------------------------

@dataclass
class ResilienceResult:
    """Result of a resilient call."""
    value: Any = None
    success: bool = False
    error: str = ""
    attempts: int = 0
    fallback_used: bool = False
    fallback_tier: str = ""
    circuit_state: str = ""
    total_ms: float = 0.0


class ResilienceLayer:
    """
    Wraps backend calls with circuit breaking, retry, and fallback.
    
    Usage:
        resilience = ResilienceLayer()
        result = await resilience.call(
            "qwen3:4b",
            lambda: cortex.process(messages, max_tokens=512),
            fallbacks=[
                ("qwen3:1.7b", lambda: cortex.process(messages, max_tokens=512)),
            ],
        )
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout: float = 30.0,
        retry_policy: Optional[RetryPolicy] = None,
    ):
        self._breakers: dict[str, CircuitBreaker] = {}
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._retry_policy = retry_policy or RetryPolicy()

    def _get_breaker(self, name: str) -> CircuitBreaker:
        """Get or create a circuit breaker for a backend."""
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(
                name=name,
                failure_threshold=self._failure_threshold,
                recovery_timeout=self._recovery_timeout,
            )
        return self._breakers[name]

    async def call(
        self,
        backend_name: str,
        fn: Callable,
        fallbacks: Optional[list[tuple[str, Callable]]] = None,
        retry_policy: Optional[RetryPolicy] = None,
    ) -> ResilienceResult:
        """
        Execute a call with full resilience stack:
          1. Check circuit breaker
          2. Execute with retry
          3. On failure, try fallbacks in order
        
        Args:
            backend_name: identifier for the primary backend
            fn: the primary callable (sync or async)
            fallbacks: list of (name, callable) fallback options
            retry_policy: override the default retry policy
        """
        t0 = time.monotonic()
        policy = retry_policy or self._retry_policy
        result = ResilienceResult()

        # --- Try primary ---
        primary_result = await self._try_with_retry(backend_name, fn, policy)
        result.attempts = primary_result.attempts

        if primary_result.success:
            result.value = primary_result.value
            result.success = True
            result.circuit_state = self._get_breaker(backend_name).state.value
            result.total_ms = (time.monotonic() - t0) * 1000
            return result

        # --- Primary failed, try fallbacks ---
        if fallbacks:
            for fb_name, fb_fn in fallbacks:
                fb_breaker = self._get_breaker(fb_name)
                if not fb_breaker.can_execute():
                    logger.info(
                        "Skipping fallback %s (circuit %s)",
                        fb_name, fb_breaker.state.value,
                    )
                    continue

                fb_result = await self._try_with_retry(
                    fb_name, fb_fn,
                    RetryPolicy(max_retries=1, base_delay=0.2),  # lighter retry for fallbacks
                )
                result.attempts += fb_result.attempts

                if fb_result.success:
                    result.value = fb_result.value
                    result.success = True
                    result.fallback_used = True
                    result.fallback_tier = fb_name
                    result.circuit_state = fb_breaker.state.value
                    result.total_ms = (time.monotonic() - t0) * 1000
                    logger.info(
                        "Fallback succeeded: %s → %s", backend_name, fb_name,
                    )
                    return result

        # --- All failed ---
        result.success = False
        result.error = primary_result.error
        result.circuit_state = self._get_breaker(backend_name).state.value
        result.total_ms = (time.monotonic() - t0) * 1000
        logger.error(
            "All backends failed for %s (%d attempts, %.1fms)",
            backend_name, result.attempts, result.total_ms,
        )
        return result

    async def _try_with_retry(
        self,
        name: str,
        fn: Callable,
        policy: RetryPolicy,
    ) -> ResilienceResult:
        """Try a callable with retry and circuit breaker."""
        breaker = self._get_breaker(name)
        result = ResilienceResult()

        for attempt in range(policy.max_retries + 1):
            result.attempts += 1

            if not breaker.can_execute():
                result.error = f"Circuit open for {name}"
                return result

            try:
                if asyncio.iscoroutinefunction(fn):
                    value = await fn()
                else:
                    loop = asyncio.get_event_loop()
                    value = await loop.run_in_executor(None, fn)

                breaker.record_success()
                result.value = value
                result.success = True
                return result

            except Exception as e:
                breaker.record_failure(str(e))
                result.error = str(e)
                logger.warning(
                    "Attempt %d/%d failed for %s: %s",
                    attempt + 1, policy.max_retries + 1, name, e,
                )

                if attempt < policy.max_retries:
                    delay = policy.delay_for_attempt(attempt)
                    await asyncio.sleep(delay)

        return result

    def call_sync(
        self,
        backend_name: str,
        fn: Callable,
        fallbacks: Optional[list[tuple[str, Callable]]] = None,
    ) -> ResilienceResult:
        """
        Synchronous version of call() for use in sync contexts.
        Uses circuit breaker + retry but no async.
        """
        t0 = time.monotonic()
        policy = self._retry_policy
        result = ResilienceResult()

        # --- Try primary ---
        primary_result = self._try_sync(backend_name, fn, policy)
        result.attempts = primary_result.attempts

        if primary_result.success:
            result.value = primary_result.value
            result.success = True
            result.circuit_state = self._get_breaker(backend_name).state.value
            result.total_ms = (time.monotonic() - t0) * 1000
            return result

        # --- Fallbacks ---
        if fallbacks:
            for fb_name, fb_fn in fallbacks:
                fb_breaker = self._get_breaker(fb_name)
                if not fb_breaker.can_execute():
                    continue

                fb_result = self._try_sync(
                    fb_name, fb_fn,
                    RetryPolicy(max_retries=1, base_delay=0.2),
                )
                result.attempts += fb_result.attempts

                if fb_result.success:
                    result.value = fb_result.value
                    result.success = True
                    result.fallback_used = True
                    result.fallback_tier = fb_name
                    result.circuit_state = fb_breaker.state.value
                    result.total_ms = (time.monotonic() - t0) * 1000
                    return result

        result.success = False
        result.error = primary_result.error
        result.circuit_state = self._get_breaker(backend_name).state.value
        result.total_ms = (time.monotonic() - t0) * 1000
        return result

    def _try_sync(
        self,
        name: str,
        fn: Callable,
        policy: RetryPolicy,
    ) -> ResilienceResult:
        """Synchronous try with retry and circuit breaker."""
        breaker = self._get_breaker(name)
        result = ResilienceResult()

        for attempt in range(policy.max_retries + 1):
            result.attempts += 1

            if not breaker.can_execute():
                result.error = f"Circuit open for {name}"
                return result

            try:
                value = fn()
                breaker.record_success()
                result.value = value
                result.success = True
                return result

            except Exception as e:
                breaker.record_failure(str(e))
                result.error = str(e)

                if attempt < policy.max_retries:
                    delay = policy.delay_for_attempt(attempt)
                    time.sleep(delay)

        return result

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def reset_circuit(self, name: str) -> bool:
        """Manually reset a circuit breaker."""
        if name in self._breakers:
            self._breakers[name].reset()
            logger.info("Circuit %s manually reset", name)
            return True
        return False

    def reset_all(self) -> int:
        """Reset all circuit breakers."""
        count = len(self._breakers)
        for b in self._breakers.values():
            b.reset()
        return count

    def status(self) -> dict:
        """Return all circuit breaker states."""
        return {
            "circuits": {
                name: breaker.to_dict()
                for name, breaker in sorted(self._breakers.items())
            },
            "total_circuits": len(self._breakers),
            "open_circuits": sum(
                1 for b in self._breakers.values()
                if b.state == CircuitState.OPEN
            ),
            "retry_policy": {
                "max_retries": self._retry_policy.max_retries,
                "base_delay": self._retry_policy.base_delay,
                "max_delay": self._retry_policy.max_delay,
            },
        }
