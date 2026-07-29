"""
AIMD Dispatcher — Adaptive parallelism for bulk API fetching.

Uses Additive Increase / Multiplicative Decrease to dynamically adjust
the number of in-flight requests based on server feedback.

Architecture:
  - One dispatcher thread owns all concurrency decisions
  - Workers are stateless: send request, report result
  - Epoch counter prevents stale responses from triggering extra reductions

Congestion signals (halve parallelism):
  - 429 Too Many Requests
  - 503 Service Unavailable
  - 502 Bad Gateway
  - Connection failures / timeouts

Non-congestion errors (don't reduce parallelism):
  - 500 Internal Server Error
  - 400 Bad Request
  - 401/403 Auth errors

Workers independently:
  - Respect Retry-After headers
  - Retry with exponential backoff
  - Skip already-completed items
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger("cortex.ckm.aimd_dispatch")


class ResponseType(Enum):
    SUCCESS = "success"
    CONGESTION = "congestion"      # 429, 503, 502, connection fail
    APP_ERROR = "app_error"        # 500, 400, etc.
    AUTH_ERROR = "auth_error"      # 401, 403


@dataclass
class RequestTelemetry:
    """Per-request telemetry from the OpenRouter API."""
    # Timing (seconds)
    time_to_first_token: float = 0.0   # TTFT: request sent → first byte back
    api_request_time: float = 0.0       # When request was sent (unix ts)
    api_response_time: float = 0.0      # When response completed (unix ts)
    api_response_total_s: float = 0.0   # Total round-trip time

    # Token usage (from OpenRouter usage block)
    tokens_in: int = 0                  # prompt_tokens
    tokens_out: int = 0                 # completion_tokens
    tokens_cached: int = 0             # prompt_tokens_details.cached_tokens
    tokens_reasoning: int = 0          # completion_tokens_details.reasoning_tokens

    # Cost (from OpenRouter)
    cost_usd: float = 0.0              # usage.cost

    # Model identity (self-declared by the API response)
    model_id: str = ""                 # The model that actually responded
    provider: str = ""                 # Upstream provider (e.g. "Alibaba", "Together")
    generation_id: str = ""            # OpenRouter generation ID

    def to_dict(self) -> dict:
        return {
            "ttft_s": self.time_to_first_token,
            "total_s": self.api_response_total_s,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "tokens_cached": self.tokens_cached,
            "tokens_reasoning": self.tokens_reasoning,
            "cost_usd": self.cost_usd,
            "model": self.model_id,
            "provider": self.provider,
            "generation_id": self.generation_id,
        }


@dataclass
class WorkResult:
    """What a worker reports back to the dispatcher."""
    task_id: str
    response_type: ResponseType
    status_code: Optional[int] = None
    response_body: Optional[str] = None
    retry_after_s: Optional[float] = None
    elapsed_s: float = 0.0
    epoch_at_dispatch: int = 0     # Epoch when this task was dispatched
    error: Optional[str] = None
    telemetry: Optional[RequestTelemetry] = None  # Rich per-request metrics


@dataclass
class DispatcherStats:
    """Live stats from the AIMD dispatcher."""
    current_parallelism: int = 4
    max_parallelism: int = 20
    min_parallelism: int = 1
    epoch: int = 0
    total_dispatched: int = 0
    total_completed: int = 0
    total_congestion: int = 0
    total_errors: int = 0
    total_retries: int = 0
    successes_since_last_increase: int = 0
    increase_threshold: int = 10   # Successes needed before +1
    start_time: float = 0.0

    @property
    def elapsed_s(self) -> float:
        return time.time() - self.start_time if self.start_time else 0.0

    @property
    def throughput_rps(self) -> float:
        elapsed = self.elapsed_s
        return self.total_completed / elapsed if elapsed > 0 else 0.0


class AIMDDispatcher:
    """
    Congestion-aware dispatcher using AIMD.

    Additive Increase: After `increase_threshold` consecutive successes,
    parallelism += 1 (up to max).

    Multiplicative Decrease: On congestion signal, parallelism = max(min, parallelism // 2).
    Epoch increments to ignore stale responses.
    """

    def __init__(
        self,
        initial_parallelism: int = 4,
        max_parallelism: int = 20,
        min_parallelism: int = 1,
        increase_threshold: int = 10,
        max_retries: int = 3,
        base_retry_delay: float = 1.0,
    ):
        self.stats = DispatcherStats(
            current_parallelism=initial_parallelism,
            max_parallelism=max_parallelism,
            min_parallelism=min_parallelism,
            increase_threshold=increase_threshold,
        )
        self.max_retries = max_retries
        self.base_retry_delay = base_retry_delay

        self._lock = threading.Lock()
        self._semaphore = threading.Semaphore(initial_parallelism)
        self._completed: dict[str, WorkResult] = {}
        self._in_flight: set[str] = set()
        self._callbacks: list[Callable] = []  # (event_type, data) callbacks

    @property
    def parallelism(self) -> int:
        with self._lock:
            return self.stats.current_parallelism

    def add_callback(self, cb: Callable[[str, Any], None]):
        """Add event callback: cb(event_type, data)."""
        self._callbacks.append(cb)

    def _emit(self, event: str, data: Any = None):
        for cb in self._callbacks:
            try:
                cb(event, data)
            except Exception:
                pass

    def report_result(self, result: WorkResult):
        """Worker reports back. Dispatcher adjusts parallelism."""
        with self._lock:
            # Ignore stale results from before a reduction
            if result.epoch_at_dispatch < self.stats.epoch:
                logger.debug(
                    f"Ignoring stale result for {result.task_id} "
                    f"(epoch {result.epoch_at_dispatch} < {self.stats.epoch})"
                )
                self._semaphore.release()
                return

            self.stats.total_completed += 1

            if result.response_type == ResponseType.SUCCESS:
                self.stats.successes_since_last_increase += 1
                self._completed[result.task_id] = result
                self._in_flight.discard(result.task_id)

                # Additive Increase
                if self.stats.successes_since_last_increase >= self.stats.increase_threshold:
                    if self.stats.current_parallelism < self.stats.max_parallelism:
                        self.stats.current_parallelism += 1
                        self._semaphore.release()  # Extra permit
                        logger.info(
                            f"AIMD ↑ parallelism={self.stats.current_parallelism} "
                            f"(after {self.stats.increase_threshold} successes)"
                        )
                        self._emit("parallelism_increase", self.stats.current_parallelism)
                    self.stats.successes_since_last_increase = 0

            elif result.response_type == ResponseType.CONGESTION:
                self.stats.total_congestion += 1
                self._in_flight.discard(result.task_id)

                # Multiplicative Decrease
                old = self.stats.current_parallelism
                self.stats.current_parallelism = max(
                    self.stats.min_parallelism,
                    self.stats.current_parallelism // 2
                )
                self.stats.epoch += 1
                self.stats.successes_since_last_increase = 0

                # Drain excess semaphore permits
                drained = old - self.stats.current_parallelism
                for _ in range(drained):
                    self._semaphore.acquire(blocking=False)

                logger.warning(
                    f"AIMD ↓ parallelism={self.stats.current_parallelism} "
                    f"(congestion: {result.status_code}, epoch={self.stats.epoch})"
                )
                self._emit("parallelism_decrease", {
                    "new": self.stats.current_parallelism,
                    "old": old,
                    "trigger": result.status_code,
                    "epoch": self.stats.epoch,
                })

            elif result.response_type == ResponseType.APP_ERROR:
                self.stats.total_errors += 1
                self._in_flight.discard(result.task_id)
                # Don't change parallelism for app errors

            elif result.response_type == ResponseType.AUTH_ERROR:
                self.stats.total_errors += 1
                self._in_flight.discard(result.task_id)
                logger.error(f"Auth error for {result.task_id}: {result.status_code}")

        self._semaphore.release()
        self._emit("result", result)

    def acquire_slot(self, task_id: str) -> int:
        """Block until a slot is available. Returns current epoch."""
        self._semaphore.acquire()
        with self._lock:
            self._in_flight.add(task_id)
            self.stats.total_dispatched += 1
            return self.stats.epoch

    def is_completed(self, task_id: str) -> bool:
        """Check if a task was already completed (skip logic)."""
        with self._lock:
            return task_id in self._completed

    def get_stats(self) -> dict:
        """Snapshot of current stats."""
        with self._lock:
            return {
                "parallelism": self.stats.current_parallelism,
                "epoch": self.stats.epoch,
                "dispatched": self.stats.total_dispatched,
                "completed": self.stats.total_completed,
                "congestion_events": self.stats.total_congestion,
                "errors": self.stats.total_errors,
                "in_flight": len(self._in_flight),
                "throughput_rps": self.stats.throughput_rps,
                "elapsed_s": self.stats.elapsed_s,
            }


# ---------------------------------------------------------------------------
# OpenRouter worker — stateless, reports back
# ---------------------------------------------------------------------------

CONGESTION_CODES = {429, 502, 503}


def classify_response(status_code: int) -> ResponseType:
    """Classify HTTP status into response type."""
    if 200 <= status_code < 300:
        return ResponseType.SUCCESS
    elif status_code in CONGESTION_CODES:
        return ResponseType.CONGESTION
    elif status_code in (401, 403):
        return ResponseType.AUTH_ERROR
    else:
        return ResponseType.APP_ERROR


def openrouter_worker(
    task_id: str,
    model_id: str,
    prompt: str,
    api_key: str,
    dispatcher: AIMDDispatcher,
    temperature: float = 0.2,
    max_tokens: int = 1024,
    max_retries: int = 3,
    base_delay: float = 1.0,
    on_stream: Optional[Callable] = None,  # (task_id, content, telemetry)
) -> Optional[str]:
    """
    Stateless worker: acquire slot, send request, report result.

    Handles retries with exponential backoff independently.
    Respects Retry-After headers.
    Skips already-completed tasks.
    """
    import httpx

    # Skip if already done
    if dispatcher.is_completed(task_id):
        return None

    # Acquire a parallelism slot (blocks if at capacity)
    epoch = dispatcher.acquire_slot(task_id)

    for attempt in range(max_retries + 1):
        t0 = time.time()
        t_request = t0  # Unix timestamp of request
        try:
            response = httpx.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/elevate-foundry/cortex",
                    "X-Title": "Cortex Tier Election",
                },
                json={
                    "model": model_id,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=60.0,
            )
            t_response = time.time()
            elapsed = t_response - t0
            status = response.status_code
            resp_type = classify_response(status)

            if resp_type == ResponseType.SUCCESS:
                data = response.json()
                content = data["choices"][0]["message"]["content"]

                # Extract telemetry from OpenRouter response
                usage = data.get("usage", {})
                prompt_details = usage.get("prompt_tokens_details", {})
                completion_details = usage.get("completion_tokens_details", {})

                telemetry = RequestTelemetry(
                    time_to_first_token=elapsed,  # Non-streaming: TTFT ≈ total
                    api_request_time=t_request,
                    api_response_time=t_response,
                    api_response_total_s=elapsed,
                    tokens_in=usage.get("prompt_tokens", 0),
                    tokens_out=usage.get("completion_tokens", 0),
                    tokens_cached=prompt_details.get("cached_tokens", 0),
                    tokens_reasoning=completion_details.get("reasoning_tokens", 0),
                    cost_usd=usage.get("cost", 0.0) or 0.0,
                    model_id=data.get("model", model_id),
                    provider=data.get("provider", ""),
                    generation_id=data.get("id", ""),
                )

                if on_stream:
                    on_stream(task_id, content, telemetry)

                dispatcher.report_result(WorkResult(
                    task_id=task_id,
                    response_type=ResponseType.SUCCESS,
                    status_code=status,
                    response_body=content,
                    elapsed_s=elapsed,
                    epoch_at_dispatch=epoch,
                    telemetry=telemetry,
                ))
                return content

            elif resp_type == ResponseType.CONGESTION:
                # Get Retry-After if present
                retry_after = None
                if "retry-after" in response.headers:
                    try:
                        retry_after = float(response.headers["retry-after"])
                    except ValueError:
                        retry_after = None

                if attempt < max_retries:
                    delay = retry_after or (base_delay * (2 ** attempt))
                    logger.info(
                        f"[{task_id}] Congestion {status}, retry in {delay:.1f}s "
                        f"(attempt {attempt+1}/{max_retries})"
                    )
                    # Report congestion to dispatcher (adjusts parallelism)
                    dispatcher.report_result(WorkResult(
                        task_id=task_id,
                        response_type=ResponseType.CONGESTION,
                        status_code=status,
                        retry_after_s=retry_after,
                        elapsed_s=elapsed,
                        epoch_at_dispatch=epoch,
                    ))
                    time.sleep(delay)
                    # Re-acquire slot for retry
                    epoch = dispatcher.acquire_slot(task_id)
                    dispatcher.stats.total_retries += 1
                    continue
                else:
                    dispatcher.report_result(WorkResult(
                        task_id=task_id,
                        response_type=ResponseType.CONGESTION,
                        status_code=status,
                        elapsed_s=elapsed,
                        epoch_at_dispatch=epoch,
                        error=f"Max retries exhausted (last: {status})",
                    ))
                    return None

            else:
                # App error or auth error — don't retry
                error_body = response.text[:500]
                dispatcher.report_result(WorkResult(
                    task_id=task_id,
                    response_type=resp_type,
                    status_code=status,
                    response_body=error_body,
                    elapsed_s=elapsed,
                    epoch_at_dispatch=epoch,
                    error=f"HTTP {status}: {error_body[:200]}",
                ))
                return None

        except (httpx.ConnectError, httpx.TimeoutException, httpx.ReadTimeout) as e:
            elapsed = time.time() - t0
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                logger.info(f"[{task_id}] Connection error: {e}, retry in {delay:.1f}s")
                dispatcher.report_result(WorkResult(
                    task_id=task_id,
                    response_type=ResponseType.CONGESTION,
                    status_code=0,
                    elapsed_s=elapsed,
                    epoch_at_dispatch=epoch,
                    error=str(e),
                ))
                time.sleep(delay)
                epoch = dispatcher.acquire_slot(task_id)
                dispatcher.stats.total_retries += 1
                continue
            else:
                dispatcher.report_result(WorkResult(
                    task_id=task_id,
                    response_type=ResponseType.CONGESTION,
                    status_code=0,
                    elapsed_s=elapsed,
                    epoch_at_dispatch=epoch,
                    error=f"Connection failed after {max_retries} retries: {e}",
                ))
                return None

        except Exception as e:
            elapsed = time.time() - t0
            dispatcher.report_result(WorkResult(
                task_id=task_id,
                response_type=ResponseType.APP_ERROR,
                status_code=0,
                elapsed_s=elapsed,
                epoch_at_dispatch=epoch,
                error=str(e),
            ))
            return None

    return None


# ---------------------------------------------------------------------------
# Streaming worker — true TTFT via SSE stream
# ---------------------------------------------------------------------------

def openrouter_stream_worker(
    task_id: str,
    model_id: str,
    prompt: str,
    api_key: str,
    dispatcher: AIMDDispatcher,
    temperature: float = 0.2,
    max_tokens: int = 1024,
    on_token: Optional[Callable] = None,     # (task_id, token_text, is_first)
    on_complete: Optional[Callable] = None,  # (task_id, full_content, telemetry)
    cancel_event: Optional[threading.Event] = None,  # set to abort early
) -> Optional[str]:
    """
    Streaming worker: uses stream=true for real TTFT measurement.

    Calls on_token for each chunk as it arrives. Returns full content.
    If cancel_event is set by another thread, aborts mid-stream.
    """
    import httpx

    if dispatcher.is_completed(task_id):
        return None

    epoch = dispatcher.acquire_slot(task_id)
    t0 = time.time()
    content_parts = []
    ttft = None

    try:
        with httpx.stream(
            "POST",
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/elevate-foundry/cortex",
                "X-Title": "Cortex Tier Election",
            },
            json={
                "model": model_id,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}],
                "stream": True,
            },
            timeout=60.0,
        ) as response:
            status = response.status_code
            resp_type = classify_response(status)

            if resp_type != ResponseType.SUCCESS:
                response.read()
                elapsed = time.time() - t0
                dispatcher.report_result(WorkResult(
                    task_id=task_id,
                    response_type=resp_type,
                    status_code=status,
                    elapsed_s=elapsed,
                    epoch_at_dispatch=epoch,
                    error=f"HTTP {status}",
                ))
                return None

            # Parse SSE stream
            for line in response.iter_lines():
                if cancel_event and cancel_event.is_set():
                    break

                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break

                try:
                    import json
                    chunk = json.loads(payload)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    token_text = delta.get("content", "")
                    if token_text:
                        if ttft is None:
                            ttft = time.time() - t0
                        content_parts.append(token_text)
                        if on_token:
                            on_token(task_id, token_text, ttft is not None and len(content_parts) == 1)
                except (ValueError, KeyError, IndexError):
                    continue

        t_done = time.time()
        elapsed = t_done - t0
        full_content = "".join(content_parts)

        telemetry = RequestTelemetry(
            time_to_first_token=ttft or elapsed,
            api_request_time=t0,
            api_response_time=t_done,
            api_response_total_s=elapsed,
            tokens_out=len(content_parts),  # approximate: 1 chunk ~ 1 token
            model_id=model_id,
        )

        if on_complete:
            on_complete(task_id, full_content, telemetry)

        dispatcher.report_result(WorkResult(
            task_id=task_id,
            response_type=ResponseType.SUCCESS,
            status_code=200,
            response_body=full_content,
            elapsed_s=elapsed,
            epoch_at_dispatch=epoch,
            telemetry=telemetry,
        ))
        return full_content

    except (httpx.ConnectError, httpx.TimeoutException, httpx.ReadTimeout) as e:
        elapsed = time.time() - t0
        dispatcher.report_result(WorkResult(
            task_id=task_id,
            response_type=ResponseType.CONGESTION,
            status_code=0,
            elapsed_s=elapsed,
            epoch_at_dispatch=epoch,
            error=str(e),
        ))
        return None
    except Exception as e:
        elapsed = time.time() - t0
        dispatcher.report_result(WorkResult(
            task_id=task_id,
            response_type=ResponseType.APP_ERROR,
            status_code=0,
            elapsed_s=elapsed,
            epoch_at_dispatch=epoch,
            error=str(e),
        ))
        return None


# ---------------------------------------------------------------------------
# Racing strategy — fire at N models, stream from first responder
# ---------------------------------------------------------------------------

def race_models_stream(
    prompt: str,
    model_ids: list[str],
    api_key: str,
    dispatcher: AIMDDispatcher,
    temperature: float = 0.2,
    max_tokens: int = 1024,
    on_token: Optional[Callable] = None,     # (model_id, token_text, is_first)
    on_winner: Optional[Callable] = None,    # (model_id, ttft)
    on_complete: Optional[Callable] = None,  # (model_id, full_content, telemetry)
) -> Optional[tuple[str, str]]:
    """
    Race multiple models in parallel, stream from the first to produce tokens.

    TTFT = min(TTFT_model_1, ..., TTFT_model_N)

    Fires all models simultaneously. The first model to emit a token "wins".
    Other models are cancelled. Returns (winner_model_id, full_content).

    This makes Cortex faster than any individual frontier model.
    """
    if not model_ids:
        return None

    # Shared state for the race
    winner_lock = threading.Lock()
    winner_id: list[Optional[str]] = [None]  # mutable container
    cancel_events: dict[str, threading.Event] = {}
    results: dict[str, dict] = {}
    race_start = time.time()

    def make_token_handler(mid: str) -> Callable:
        def handler(task_id: str, token_text: str, is_first: bool):
            if is_first:
                with winner_lock:
                    if winner_id[0] is None:
                        # This model won the race
                        winner_id[0] = mid
                        ttft = time.time() - race_start
                        if on_winner:
                            on_winner(mid, ttft)
                        # Cancel all other models
                        for other_mid, evt in cancel_events.items():
                            if other_mid != mid:
                                evt.set()
                    elif winner_id[0] != mid:
                        # Another model already won
                        cancel_events[mid].set()
                        return
            # Only forward tokens from the winner
            if winner_id[0] == mid and on_token:
                on_token(mid, token_text, is_first)
        return handler

    def make_complete_handler(mid: str) -> Callable:
        def handler(task_id: str, full_content: str, telemetry: RequestTelemetry):
            results[mid] = {"content": full_content, "telemetry": telemetry}
            if winner_id[0] == mid and on_complete:
                on_complete(mid, full_content, telemetry)
        return handler

    # Set up cancel events and launch all models simultaneously
    for mid in model_ids:
        cancel_events[mid] = threading.Event()

    threads: list[threading.Thread] = []
    for mid in model_ids:
        task = f"race_{mid}_{int(race_start)}"
        t = threading.Thread(
            target=openrouter_stream_worker,
            args=(task, mid, prompt, api_key, dispatcher),
            kwargs={
                "temperature": temperature,
                "max_tokens": max_tokens,
                "on_token": make_token_handler(mid),
                "on_complete": make_complete_handler(mid),
                "cancel_event": cancel_events[mid],
            },
            daemon=True,
        )
        threads.append(t)
        t.start()

    # Wait for all threads (winner completes, losers get cancelled quickly)
    for t in threads:
        t.join(timeout=90.0)

    # Return winner's result
    winner = winner_id[0]
    if winner and winner in results:
        return (winner, results[winner]["content"])
    elif winner:
        # Winner set but result not yet recorded — brief wait
        time.sleep(0.5)
        if winner in results:
            return (winner, results[winner]["content"])

    return None
