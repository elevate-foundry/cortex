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
from typing import Any, Callable, Optional

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
from .tier_map import TierMap

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
    # Usage telemetry
    tokens_prompt: int = 0
    tokens_completion: int = 0
    tokens_cached: int = 0
    tokens_reasoning: int = 0
    cost_usd: float = 0.0
    provider: str = ""

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

        raw_max = max_feasible_tier(self.profile)
        # Memory-pressure guard: on machines with <= 16 GB unified RAM,
        # cap at L2 (4B model) to avoid thrashing. Hard problems use cloud
        # consensus race instead of loading huge local models.
        if self.profile.total_vram_mb <= 16384 and raw_max.value > Tier.L2.value:
            self._max_tier = Tier.L2
            logger.info(
                f"Memory guard: capping local tier at L2 "
                f"(machine has {self.profile.total_vram_mb}MB, "
                f"would need L4+ for {raw_max.name})"
            )
        else:
            self._max_tier = raw_max
        self._booted = False
        self._pool: Optional[BackendPool] = None
        self._tier_map: Optional[TierMap] = None

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

        # Load tier election results (model → tier mapping)
        self._tier_map = TierMap.load()
        if not self._tier_map.empty:
            logger.info("TierMap: %s", self._tier_map.summary())

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
            tokens_prompt=core_response.tokens_prompt,
            tokens_completion=core_response.tokens_completion,
            tokens_cached=core_response.tokens_cached,
            tokens_reasoning=core_response.tokens_reasoning,
            cost_usd=core_response.cost_usd,
            provider=core_response.provider,
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
        tool_choice: Any = None,
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
            tool_choice=tool_choice,
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
            tokens_prompt=resp.tokens_prompt,
            tokens_completion=resp.tokens_completion,
            tokens_cached=resp.tokens_cached,
            tokens_reasoning=resp.tokens_reasoning,
            cost_usd=resp.cost_usd,
            provider=resp.provider,
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

        # Use TierMap (election data) if available, else hardcoded defaults
        hardcoded = self._TIER_CLOUD_MODELS.get(tier, "qwen/qwen3-8b")
        if self._tier_map and not self._tier_map.empty:
            model = self._tier_map.cloud_model_for_tier(tier, fallback=hardcoded)
        else:
            model = hardcoded
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

    # Expanded pool by family — for adaptive selection
    _RACE_POOL = {
        # Tier 1: fastest/cheapest (always include)
        "fast": [
            "openai/gpt-4o-mini",
            "google/gemini-2.0-flash-001",
            "mistralai/mistral-small-24b-instruct-2501",
            "qwen/qwen3-coder",
            "deepseek/deepseek-chat-v3-0324",
        ],
        # Tier 2: strong reasoning (add for medium difficulty)
        "reasoning": [
            "anthropic/claude-3.5-haiku",
            "google/gemini-2.5-flash",
            "deepseek/deepseek-r1",
            "qwen/qwen3-235b-a22b",
            "nvidia/llama-3.1-nemotron-70b-instruct",
        ],
        # Tier 3: frontier (add for hard problems)
        "frontier": [
            "openai/gpt-4o",
            "anthropic/claude-sonnet-4",
            "google/gemini-2.5-pro",
            "deepseek/deepseek-r1",
            "meta-llama/llama-4-maverick",
        ],
        # Tier 4: specialists (add when domain-specific)
        "code": [
            "qwen/qwen3-coder",
            "deepseek/deepseek-chat-v3-0324",
            "openai/gpt-4o-mini",
            "anthropic/claude-sonnet-4",
            "mistralai/codestral-2501",
        ],
        "math": [
            "deepseek/deepseek-r1",
            "qwen/qwen3-235b-a22b",
            "openai/o4-mini",
            "google/gemini-2.5-pro",
        ],
    }

    def select_candidates(
        self,
        prompt: str,
        complexity: float = 0.0,
        category: str = "",
        max_candidates: int = 20,
    ) -> list[str]:
        """
        Adaptively select race candidates based on task difficulty.

        Strategy:
          - Complexity < 0.3 (easy):   3 fast models (diverse families)
          - Complexity 0.3-0.6 (med):  5-8 models (fast + reasoning)
          - Complexity 0.6-0.8 (hard): 10-15 models (fast + reasoning + frontier)
          - Complexity > 0.8 (critical): 15-20 (all tiers + specialists)

        Always ensures cross-family diversity (no more than 2 per provider).
        """
        from .router import _estimate_complexity, _categorize

        # Auto-detect if not provided
        if not complexity:
            complexity = _estimate_complexity(prompt)
        if not category:
            cat_result = _categorize(prompt)
            category = cat_result.name if hasattr(cat_result, 'name') else str(cat_result)

        candidates = []
        seen_families = {}

        def add_tier(pool_key: str, max_per_family: int = 2):
            for model in self._RACE_POOL.get(pool_key, []):
                family = model.split("/")[0]
                if seen_families.get(family, 0) >= max_per_family:
                    continue
                if model not in candidates:
                    candidates.append(model)
                    seen_families[family] = seen_families.get(family, 0) + 1

        # Always include fast tier
        add_tier("fast")

        # Add reasoning for medium+
        if complexity >= 0.3:
            add_tier("reasoning")

        # Add frontier for hard+
        if complexity >= 0.6:
            add_tier("frontier")

        # Add specialists based on category
        cat_lower = category.lower()
        if "code" in cat_lower or "debug" in cat_lower:
            add_tier("code")
        elif "math" in cat_lower or "analyze" in cat_lower:
            add_tier("math")

        # For critical tasks, add more from each tier
        if complexity >= 0.8:
            for pool_key in self._RACE_POOL:
                add_tier(pool_key, max_per_family=3)

        # Cap at max_candidates
        return candidates[:max_candidates]

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
    # Quality race — fire all models, collect ALL, pick the best
    # ------------------------------------------------------------------

    def race_quality(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.3,
        candidates: Optional[list[str]] = None,
        judge: Optional[Callable[[list[tuple[str, str]]], int]] = None,
    ) -> Optional[tuple[str, str, list[tuple[str, str]]]]:
        """
        Quality race with gossip consensus.

        Fire N models in parallel, wait for ALL responses, then run a
        gossip-style convergence protocol where models "vote" on the best
        answer by clustering agreement and propagating preferences.

        The gossip protocol ensures:
        - Responses are clustered by semantic agreement
        - Cross-family agreement gets bonus weight (independent validation)
        - The consensus is the answer that the most models converge on
        - Minority responses are preserved as alternatives

        Unlike race_cloud (TTFT-optimized), this maximizes answer QUALITY.
        Cost is O(N) API calls — use when getting the RIGHT answer matters
        more than speed (e.g., game decisions, important reasoning).

        Returns (winner_model, winner_response, all_responses) or None.
        """
        import threading
        import time as _time
        import httpx

        if self._pool is None:
            return None
        or_backend = self._pool.get_backend(BackendType.OPENROUTER)
        if or_backend is None or not or_backend.api_key:
            return None

        models = candidates or self.select_candidates(prompt)
        api_key = or_backend.api_key
        results: dict[str, str] = {}
        timings: dict[str, float] = {}
        lock = threading.Lock()

        def worker(model_id: str):
            t0 = _time.time()
            try:
                resp = httpx.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://github.com/elevate-foundry/cortex",
                    },
                    json={
                        "model": model_id,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                    },
                    timeout=90.0,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    content = data["choices"][0]["message"]["content"]
                    with lock:
                        results[model_id] = content
                        timings[model_id] = _time.time() - t0
            except Exception:
                pass

        # Phase 1: Fire all models in parallel
        threads = [threading.Thread(target=worker, args=(m,), daemon=True) for m in models]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=90.0)

        if not results:
            return None

        all_responses = [(mid, resp) for mid, resp in results.items() if resp]

        # Phase 2: Gossip consensus — cluster by agreement
        if judge:
            winner_idx = judge(all_responses)
        else:
            winner_idx = self._gossip_consensus(all_responses, timings)

        winner_model, winner_response = all_responses[winner_idx]
        return (winner_model, winner_response, all_responses)

    def _gossip_consensus(
        self,
        responses: list[tuple[str, str]],
        timings: dict[str, float],
    ) -> int:
        """
        Gossip-style consensus: each model's response is a "peer" that
        propagates its state. Peers cluster by semantic agreement.
        The largest cluster's highest-weighted member wins.

        Weighting:
        - Cross-family agreement bonus (independent validation)
        - Faster response = slight tiebreaker (model was more confident)
        - Longer substantive response = slight quality signal
        """
        from .challenger import compare_answers, AgreementLevel

        if len(responses) <= 1:
            return 0

        # Build agreement graph — each pair of responses gets a score
        n = len(responses)
        agreement_matrix = [[0.0] * n for _ in range(n)]

        for i in range(n):
            for j in range(i + 1, n):
                level, _, _ = compare_answers(responses[i][1], responses[j][1])
                score = {
                    AgreementLevel.STRONG_AGREE: 1.0,
                    AgreementLevel.WEAK_AGREE: 0.6,
                    AgreementLevel.DISAGREE: 0.0,
                    AgreementLevel.STRONG_DISAGREE: -0.3,
                }.get(level, 0.0)
                agreement_matrix[i][j] = score
                agreement_matrix[j][i] = score

        # Greedy clustering (same as swarm.py but on cloud responses)
        clusters: list[list[int]] = []
        assigned = [False] * n

        for i in range(n):
            if assigned[i]:
                continue
            cluster = [i]
            assigned[i] = True
            for j in range(i + 1, n):
                if assigned[j]:
                    continue
                # Join cluster if agrees with majority of existing members
                agree_count = sum(1 for k in cluster if agreement_matrix[j][k] >= 0.6)
                if agree_count > len(cluster) / 2:
                    cluster.append(j)
                    assigned[j] = True
            clusters.append(cluster)

        # Gossip round: propagate cluster preferences
        # Each cluster's weight = sum of individual member weights
        def member_weight(idx: int) -> float:
            mid, resp = responses[idx]
            w = 1.0
            # Cross-family bonus: different provider prefixes = independent
            family = mid.split("/")[0] if "/" in mid else mid
            # Substantive response bonus (log-scaled, caps at ~2x)
            import math
            w += min(1.0, math.log(max(len(resp), 1)) / 10)
            # Speed bonus (faster = more confident, slight effect)
            if mid in timings:
                speed_bonus = max(0, 1.0 - timings[mid] / 30.0) * 0.3
                w += speed_bonus
            return w

        cluster_weights = []
        for cluster in clusters:
            total_w = sum(member_weight(i) for i in cluster)
            # Cross-family bonus: count distinct providers in cluster
            families = set(responses[i][0].split("/")[0] for i in cluster)
            cross_family_bonus = (len(families) - 1) * 0.5
            cluster_weights.append(total_w + cross_family_bonus)

        # Winner = best member of heaviest cluster
        best_cluster_idx = max(range(len(clusters)), key=lambda i: cluster_weights[i])
        best_cluster = clusters[best_cluster_idx]

        # Within the winning cluster, pick the highest-weighted member
        winner_local = max(best_cluster, key=member_weight)
        return winner_local

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
