"""
Cortex — the unified inference orchestrator (PID 1 for AI).

Wires together:
  Router → ModelManager → Challenger → Swarm

This is the single entry point for processing a user request.
It handles the full escalation path:

  1. Router classifies the request → picks a tier
  2. Core model at that tier generates an answer
  3. If confidence is low → Challenger verifies with a different family
  4. If challenger disagrees → Swarm fans out to N models
  5. If swarm can't reach consensus → escalate to L7 (remote frontier)
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from .backend_adapter import (
    BackendAdapter,
    BackendPool,
    BackendType,
    CompletionRequest,
    CompletionResponse,
)
from .challenger import Challenger, ChallengeResult, AgreementLevel
from .model_manager import ModelManager, ManagerConfig, ModelState
from .router import RouteDecision, TaskCategory, route_heuristic, route_heuristic_messages, route_with_model
from .swarm import Swarm, SwarmResult, SwarmSize, AggregationMethod
from .tiers import Tier, TIER_SPECS, assess_tiers, max_feasible_tier
from .hardware_detect import SystemProfile, detect_system
from .memory import Memory

logger = logging.getLogger(__name__)


@dataclass
class CortexConfig:
    """Configuration for the Cortex orchestrator."""
    # Confidence thresholds
    challenge_threshold: float = 0.75   # below this → challenge
    swarm_threshold: float = 0.50       # below this → swarm
    large_swarm_threshold: float = 0.30 # below this → large swarm

    # Model manager config
    manager_config: Optional[ManagerConfig] = None

    # Whether to use L0 model for routing (vs heuristic)
    use_model_router: bool = True

    # Max escalation tier (capped by hardware)
    max_escalation: Tier = Tier.L7


@dataclass
class CortexResponse:
    """The full response from Cortex, including metadata about the process."""
    content: str
    tier_used: Tier
    model_used: str
    confidence: float
    route_decision: RouteDecision
    challenge_result: Optional[ChallengeResult] = None
    swarm_result: Optional[SwarmResult] = None
    escalation_path: list[str] = field(default_factory=list)
    total_ms: float = 0.0
    raw_response: Optional[dict] = None  # full raw backend response (for tool calls)

class Cortex:
    """
    The inference orchestrator — PID 1 for AI.
    
    Boot it, then call `process()` for each user request.
    It handles routing, generation, verification, and escalation.
    """

    def __init__(
        self,
        profile: Optional[SystemProfile] = None,
        config: Optional[CortexConfig] = None,
        memory: Optional["Memory"] = None,
    ):
        self.config = config or CortexConfig()
        self.profile = profile or detect_system()
        self.memory = memory  # optional — set externally (e.g. by daemon)

        # Initialize subsystems
        manager_config = self.config.manager_config or ManagerConfig()
        self.manager = ModelManager(self.profile, manager_config)
        self.challenger = Challenger(
            self.manager,
            escalation_threshold=self.config.swarm_threshold,
        )
        self.swarm = Swarm(self.manager)

        self._max_tier = max_feasible_tier(self.profile)
        self._booted = False
        self._pool: Optional[BackendPool] = None

    def boot(self) -> None:
        """
        Boot sequence: load always-hot models (L0, L1, L2).
        Must be called before process().
        """
        logger.info("=== Cortex Boot ===")
        self.manager.boot()

        # Discover cloud backends (OpenRouter, etc.) as fallback
        self._pool = BackendPool.auto_discover()
        if self._pool.available:
            logger.info("Cloud backends available: %s", self._pool.summary())
        else:
            logger.info("No cloud backends — local only mode")

        self._booted = True
        logger.info(f"Cortex ready. Max local tier: {self._max_tier.name}")

    def process(
        self,
        messages: list[dict],
        max_tokens: int = 512,
        tools: Optional[list[dict]] = None,
    ) -> CortexResponse:
        """
        Process a user request through the full pipeline.
        
        Args:
            messages: Chat messages [{"role": "user", "content": "..."}]
            max_tokens: Max tokens for generation
        """
        if not self._booted:
            self.boot()

        t0 = time.monotonic()
        escalation_path: list[str] = []

        # Thinking models (Qwen3) use tokens for reasoning before content.
        # Ensure a minimum budget so the model can produce visible output.
        gen_tokens = max(max_tokens, 256)

        # --- Step 1: Route (vision-aware) ---
        # Check for multimodal content first — routes to GPT-4o for vision
        route = route_heuristic_messages(messages, self._max_tier,
                    [a.tier for a in assess_tiers(self.profile) if a.feasible])

        # Extract text prompt for logging/downstream
        prompt = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                c = msg.get("content", "")
                prompt = c if isinstance(c, str) else "[multimodal]"
                break

        # If not vision, refine with model-based routing when available
        if route.category != TaskCategory.VISION:
            route = self._route(prompt)
        tier = route.tier
        escalation_path.append(f"route→{tier.name}(conf={route.confidence:.2f})")

        # --- Step 2: Generate with core model ---
        core_response = self._generate(messages, tier, gen_tokens, tools=tools)
        if core_response is None:
            # Escalate if core model failed
            tier, core_response = self._escalate_generate(messages, tier, max_tokens, tools=tools)
            escalation_path.append(f"escalate→{tier.name}")

        if core_response is None:
            return CortexResponse(
                content="I'm unable to process this request — no models available.",
                tier_used=tier,
                model_used="none",
                confidence=0.0,
                route_decision=route,
                escalation_path=escalation_path,
                total_ms=(time.monotonic() - t0) * 1000,
            )

        # --- Step 3: Decide if we need verification ---
        challenge_result = None
        swarm_result = None

        # Check if we should challenge
        effective_confidence = route.confidence
        should_challenge = effective_confidence < self.config.challenge_threshold

        if should_challenge:
            escalation_path.append("challenge")
            challenge_result = self.challenger.challenge(
                messages, core_response, tier, max_tokens
            )
            effective_confidence = challenge_result.confidence

            if challenge_result.agreement in (
                AgreementLevel.STRONG_AGREE,
                AgreementLevel.WEAK_AGREE,
            ):
                escalation_path.append(f"challenge_agree({challenge_result.agreement.value})")
            else:
                # --- Step 4: Disagreement → Swarm ---
                escalation_path.append(f"challenge_disagree({challenge_result.agreement.value})")

                if effective_confidence < self.config.large_swarm_threshold:
                    swarm_size = SwarmSize.LARGE
                    escalation_path.append("large_swarm")
                else:
                    swarm_size = SwarmSize.SMALL
                    escalation_path.append("swarm")

                swarm_result = self.swarm.query_sync(
                    messages, tier, size=swarm_size, max_tokens=max_tokens
                )
                effective_confidence = swarm_result.confidence

                if swarm_result.consensus_answer:
                    core_response = CompletionResponse(
                        content=swarm_result.consensus_answer,
                        model=f"swarm({swarm_result.num_models})",
                        backend=core_response.backend,
                        total_ms=swarm_result.total_ms,
                    )
                    escalation_path.append(
                        f"swarm_consensus(conf={swarm_result.confidence:.2f}, "
                        f"{swarm_result.num_agreeing_families}/{swarm_result.num_families} families)"
                    )

        total_ms = (time.monotonic() - t0) * 1000

        result = CortexResponse(
            content=core_response.content,
            tier_used=tier,
            model_used=core_response.model,
            confidence=effective_confidence,
            route_decision=route,
            challenge_result=challenge_result,
            swarm_result=swarm_result,
            escalation_path=escalation_path,
            total_ms=total_ms,
            raw_response=core_response.raw,
        )

        # Self-audit: log routing decision to Memory if available
        if self.memory is not None:
            try:
                self.memory.log_request(
                    thread_id="",
                    request_model="auto",
                    routed_tier=tier.name,
                    actual_model=core_response.model,
                    category=route.category.value,
                    confidence=effective_confidence,
                    tokens_prompt=0,
                    tokens_completion=core_response.tokens_generated,
                    latency_ms=total_ms,
                    ttft_ms=core_response.ttft_ms,
                    status_code=200,
                    escalation_path=escalation_path,
                )
            except Exception:
                pass  # never fail a request due to audit

        return result

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _route(self, prompt: str) -> RouteDecision:
        """Route a prompt to a tier."""
        assessments = assess_tiers(self.profile)
        available = [a.tier for a in assessments if a.feasible]

        if self.config.use_model_router:
            # Try to use the L0 model for routing
            l0_adapter = self.manager.get_adapter(Tier.L0)
            if l0_adapter is not None:
                def model_fn(routing_prompt: str) -> str:
                    req = CompletionRequest(
                        messages=[{"role": "user", "content": routing_prompt}],
                        max_tokens=100,
                        temperature=0.0,
                    )
                    resp = l0_adapter.complete_sync(req)
                    return resp.content

                return route_with_model(
                    prompt, model_fn,
                    max_tier=self._max_tier,
                    available_tiers=available,
                )

        # Fall back to heuristic
        return route_heuristic(
            prompt,
            max_tier=self._max_tier,
            available_tiers=available,
        )

    def _generate(
        self,
        messages: list[dict],
        tier: Tier,
        max_tokens: int,
        tools: Optional[list[dict]] = None,
    ) -> Optional[CompletionResponse]:
        """Generate a response using the core model at a tier."""
        adapter = self.manager.get_adapter(tier)
        if adapter is None:
            # Try to load any model in this tier on-demand
            from .tiers import get_models_for_tier
            models = get_models_for_tier(tier, self.profile)
            for m in models:
                try:
                    loaded = self.manager.load_model(tier, m)
                    if loaded and loaded.state == ModelState.READY:
                        adapter = loaded.adapter
                        break
                except Exception:
                    continue

        if adapter is None:
            # Final fallback: use cloud backend pool (OpenRouter)
            adapter = self._get_cloud_fallback(tier)
            if adapter is None:
                return None

        # For reflex tiers (L0-L2), suppress Qwen3 thinking mode to save
        # tokens and reduce latency — these are fast lookups, not reasoning.
        gen_messages = list(messages)
        if tier.value <= Tier.L2.value:
            if not any(m.get("role") == "system" for m in gen_messages):
                gen_messages.insert(0, {"role": "system", "content": "/no_think"})

        # Keep hot models (L0-L2) pinned in VRAM
        extra = {}
        if TIER_SPECS[tier].always_hot:
            extra["keep_alive"] = "24h"

        req = CompletionRequest(
            messages=gen_messages,
            max_tokens=max_tokens,
            temperature=0.0,
            extra=extra,
            tools=tools,
        )

        try:
            return adapter.complete_sync(req)
        except Exception as e:
            logger.error(f"Generation failed at {tier.name}: {e}")
            return None

    def _escalate_generate(
        self,
        messages: list[dict],
        current_tier: Tier,
        max_tokens: int,
        tools: Optional[list[dict]] = None,
    ) -> tuple[Tier, Optional[CompletionResponse]]:
        """Try higher tiers until one works."""
        for tier_val in range(current_tier + 1, min(self._max_tier + 1, Tier.L7 + 1)):
            tier = Tier(tier_val)
            resp = self._generate(messages, tier, max_tokens, tools=tools)
            if resp is not None:
                return tier, resp
        return current_tier, None

    # ------------------------------------------------------------------
    # Explicit model override (bypass routing)
    # ------------------------------------------------------------------

    def process_with_model(
        self,
        messages: list[dict],
        model: str,
        max_tokens: int = 512,
        tools: Optional[list[dict]] = None,
    ) -> CortexResponse:
        """
        Process a request targeting a specific model (bypass routing).

        Used when the user explicitly sets a model in the request.
        Routes directly to the appropriate backend for that model.
        """
        if not self._booted:
            self.boot()

        t0 = time.monotonic()
        gen_tokens = max(max_tokens, 256)

        # Determine which backend has this model
        adapter = None
        if self._pool:
            adapter = self._pool.get_backend_for_model(model)
        if adapter is None:
            # Try local
            adapter = self.manager.get_adapter_by_model(model)
        if adapter is None:
            # Last resort: try primary
            if self._pool and self._pool.primary:
                adapter = self._pool.primary

        if adapter is None:
            return CortexResponse(
                content=f"Model '{model}' is not available on any backend.",
                tier_used=Tier.L0,
                model_used="none",
                confidence=0.0,
                route_decision=route_heuristic("", max_tier=self._max_tier),
                total_ms=(time.monotonic() - t0) * 1000,
            )

        # Build request
        req = CompletionRequest(
            messages=messages,
            model=model,
            max_tokens=gen_tokens,
            temperature=0.0,
            tools=tools,
        )

        try:
            resp = adapter.complete_sync(req)
        except Exception as e:
            logger.error(f"Explicit model request failed ({model}): {e}")
            return CortexResponse(
                content=f"Backend error for model '{model}': {e}",
                tier_used=Tier.L7,
                model_used=model,
                confidence=0.0,
                route_decision=route_heuristic("", max_tier=self._max_tier),
                total_ms=(time.monotonic() - t0) * 1000,
            )

        total_ms = (time.monotonic() - t0) * 1000
        return CortexResponse(
            content=resp.content,
            tier_used=Tier.L7,  # explicit model = unclassified tier
            model_used=resp.model,
            confidence=1.0,
            route_decision=route_heuristic("", max_tier=self._max_tier),
            escalation_path=[f"explicit_model→{model}"],
            total_ms=total_ms,
            raw_response=resp.raw,
        )

    # ------------------------------------------------------------------
    # Cloud fallback
    # ------------------------------------------------------------------

    # Tier → recommended OpenRouter model
    _TIER_CLOUD_MODELS = {
        Tier.L0: "qwen/qwen3-1.7b",
        Tier.L1: "qwen/qwen3-1.7b",
        Tier.L2: "qwen/qwen3-4b",
        Tier.L3: "qwen/qwen3-8b",
        Tier.L4: "microsoft/phi-4",
        Tier.L5: "qwen/qwen3-32b",
        Tier.L6: "meta-llama/llama-3.3-70b-instruct",
        Tier.L7: "openai/gpt-4o",  # Frontier: vision + reasoning
    }

    # Designated vision model — GPT-4o is best-in-class for multimodal
    _VISION_MODEL = "openai/gpt-4o"

    def _get_cloud_fallback(self, tier: Tier) -> Optional[BackendAdapter]:
        """Get a cloud backend adapter configured for the appropriate tier."""
        if self._pool is None or not self._pool.available:
            return None

        or_backend = self._pool.get_backend(BackendType.OPENROUTER)
        if or_backend is None:
            return None

        # Set the model to match the requested tier
        model = self._TIER_CLOUD_MODELS.get(tier, "qwen/qwen3-8b")
        or_backend.default_model = model
        return or_backend

    # ------------------------------------------------------------------
    # Racing models — TTFT = min(all candidates)
    # ------------------------------------------------------------------

    # Models eligible for racing at L7 (diverse providers for best TTFT)
    _RACE_CANDIDATES = [
        "openai/gpt-4o-mini",
        "google/gemini-2.0-flash-001",
        "mistralai/mistral-small-24b-instruct-2501",
        "qwen/qwen3-coder",
        "deepseek/deepseek-chat-v3-0324",
    ]

    def race_cloud(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        on_token: Optional[Callable] = None,
        on_winner: Optional[Callable] = None,
        on_complete: Optional[Callable] = None,
        candidates: Optional[list[str]] = None,
    ) -> Optional[tuple[str, str]]:
        """
        Race multiple cloud models, stream from the fastest responder.

        TTFT = min(TTFT across all candidates).
        Cortex is faster than any single model because she races them all.

        Returns (winner_model_id, full_content) or None.
        """
        from .ckm.aimd_dispatch import AIMDDispatcher, race_models_stream

        if self._pool is None:
            return None
        or_backend = self._pool.get_backend(BackendType.OPENROUTER)
        if or_backend is None or not or_backend.api_key:
            return None

        dispatcher = AIMDDispatcher(
            initial_parallelism=len(candidates or self._RACE_CANDIDATES),
            max_parallelism=12,
        )

        return race_models_stream(
            prompt=prompt,
            model_ids=candidates or self._RACE_CANDIDATES,
            api_key=or_backend.api_key,
            dispatcher=dispatcher,
            temperature=temperature,
            max_tokens=max_tokens,
            on_token=on_token,
            on_winner=on_winner,
            on_complete=on_complete,
        )

    # ------------------------------------------------------------------
    # Streaming support
    # ------------------------------------------------------------------

    def resolve_backend(
        self,
        messages: list[dict],
    ) -> tuple[RouteDecision, Tier, "BackendAdapter", str]:
        """
        Route a request and return the adapter + model for streaming.
        
        Returns (route_decision, tier, adapter, model_tag).
        The caller can use the adapter directly for streaming calls.
        """
        if not self._booted:
            self.boot()

        prompt = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                prompt = msg.get("content", "")
                break

        route = self._route(prompt)
        tier = route.tier

        adapter = self.manager.get_adapter(tier)
        if adapter is None:
            from .tiers import get_models_for_tier
            models = get_models_for_tier(tier, self.profile)
            for m in models:
                try:
                    loaded = self.manager.load_model(tier, m)
                    if loaded and loaded.state == ModelState.READY:
                        adapter = loaded.adapter
                        break
                except Exception:
                    continue

        # Fall back to any available tier
        if adapter is None:
            for tier_val in range(self._max_tier.value, -1, -1):
                fallback_tier = Tier(tier_val)
                adapter = self.manager.get_adapter(fallback_tier)
                if adapter is not None:
                    tier = fallback_tier
                    break

        # Final fallback: cloud backend pool
        if adapter is None:
            adapter = self._get_cloud_fallback(tier)

        model_tag = adapter.default_model if adapter else ""
        return route, tier, adapter, model_tag

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> dict:
        """Return full Cortex status."""
        return {
            "booted": self._booted,
            "max_tier": self._max_tier.name,
            "config": {
                "challenge_threshold": self.config.challenge_threshold,
                "swarm_threshold": self.config.swarm_threshold,
                "large_swarm_threshold": self.config.large_swarm_threshold,
                "use_model_router": self.config.use_model_router,
            },
            "manager": self.manager.status(),
        }

    def __repr__(self) -> str:
        state = "booted" if self._booted else "not booted"
        return f"Cortex({state}, max={self._max_tier.name}, {self.manager})"
