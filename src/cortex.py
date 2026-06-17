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
from typing import Optional

from .backend_adapter import (
    BackendAdapter,
    BackendType,
    CompletionRequest,
    CompletionResponse,
)
from .challenger import Challenger, ChallengeResult, AgreementLevel
from .model_manager import ModelManager, ManagerConfig, ModelState
from .router import RouteDecision, route_heuristic, route_with_model
from .swarm import Swarm, SwarmResult, SwarmSize, AggregationMethod
from .tiers import Tier, TIER_SPECS, assess_tiers, max_feasible_tier
from .hardware_detect import SystemProfile, detect_system

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
    ):
        self.config = config or CortexConfig()
        self.profile = profile or detect_system()

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

    def boot(self) -> None:
        """
        Boot sequence: load always-hot models (L0, L1, L2).
        Must be called before process().
        """
        logger.info("=== Cortex Boot ===")
        self.manager.boot()
        self._booted = True
        logger.info(f"Cortex ready. Max local tier: {self._max_tier.name}")

    def process(
        self,
        messages: list[dict],
        max_tokens: int = 512,
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

        # Extract the user prompt for routing
        prompt = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                prompt = msg.get("content", "")
                break

        # --- Step 1: Route ---
        route = self._route(prompt)
        tier = route.tier
        escalation_path.append(f"route→{tier.name}(conf={route.confidence:.2f})")

        # --- Step 2: Generate with core model ---
        core_response = self._generate(messages, tier, gen_tokens)
        if core_response is None:
            # Escalate if core model failed
            tier, core_response = self._escalate_generate(messages, tier, max_tokens)
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

        return CortexResponse(
            content=core_response.content,
            tier_used=tier,
            model_used=core_response.model,
            confidence=effective_confidence,
            route_decision=route,
            challenge_result=challenge_result,
            swarm_result=swarm_result,
            escalation_path=escalation_path,
            total_ms=total_ms,
        )

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
    ) -> Optional[CompletionResponse]:
        """Generate a response using the core model at a tier."""
        adapter = self.manager.get_adapter(tier)
        if adapter is None:
            # Try to load the model on-demand
            from .tiers import get_models_for_tier
            models = get_models_for_tier(tier, self.profile)
            if models:
                loaded = self.manager.load_model(tier, models[0])
                if loaded and loaded.state == ModelState.READY:
                    adapter = loaded.adapter

        if adapter is None:
            return None

        req = CompletionRequest(
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.0,
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
    ) -> tuple[Tier, Optional[CompletionResponse]]:
        """Try higher tiers until one works."""
        for tier_val in range(current_tier + 1, min(self._max_tier + 1, Tier.L7 + 1)):
            tier = Tier(tier_val)
            resp = self._generate(messages, tier, max_tokens)
            if resp is not None:
                return tier, resp
        return current_tier, None

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
