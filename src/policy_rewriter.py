"""
Cortex Policy Rewriter — self-modifying routing policy engine.

Observes routing_feedback, detects underperforming tiers/models,
and mutates TIER_SPECS and routing heuristics.  This is the
"metacognition" layer: Cortex learning from its own mistakes.

Design:
  - Feedback → accuracy analysis → mutation proposal → mutation audit → apply
  - Every mutation is recorded in policy_mutations table
  - Mutations are conservative: small deltas, high threshold
  - Rollback is possible via mutation history
"""

import json
import logging
from dataclasses import dataclass, asdict
from typing import Optional

from .tiers import Tier, TierSpec, TIER_SPECS
from .memory import Memory

logger = logging.getLogger("cortex.policy_rewriter")

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

MIN_SAMPLES = 10          # minimum feedback samples before mutation
ACCURACY_THRESHOLD = 0.85 # below this, we consider a tier underperforming
LATENCY_THRESHOLD_MS = 5000  # above this, bump latency-sensitive tasks down
MUTATION_MIN_DELTA = 0.05    # minimum confidence delta to trigger mutation


# ---------------------------------------------------------------------------
# Mutation types
# ---------------------------------------------------------------------------

@dataclass
class MutationProposal:
    """A proposed change to policy/tier configuration."""
    tier: str
    field: str
    old_value: str
    new_value: str
    reason: str
    confidence: float
    should_apply: bool = False


# ---------------------------------------------------------------------------
# Policy Rewriter
# ---------------------------------------------------------------------------

class PolicyRewriter:
    """
    Analyzes routing_feedback and proposes mutations to TIER_SPECS
    and routing heuristics.

    Usage (from daemon background task):
        rewriter = PolicyRewriter(memory)
        mutations = rewriter.analyze_and_mutate()
        for m in mutations:
            if m.should_apply:
                rewriter.apply_mutation(m)
    """

    def __init__(self, memory: Memory):
        self.memory = memory

    def analyze_and_mutate(self) -> list[MutationProposal]:
        """
        Main entry point.  Analyzes recent feedback and returns
        proposed mutations (some marked should_apply).
        """
        proposals: list[MutationProposal] = []

        # 1. Analyze per-tier accuracy
        tier_accuracy = self.memory.get_routing_accuracy(days=7)
        logger.debug("Policy rewriter: %s total feedback samples", tier_accuracy.get("total", 0))

        for tier_name, stats in tier_accuracy.get("by_tier", {}).items():
            total = stats.get("total", 0)
            if total < MIN_SAMPLES:
                continue
            acc = stats.get("predicted_accuracy", 1.0)
            if acc < ACCURACY_THRESHOLD:
                proposals.append(self._propose_tier_demotion(tier_name, acc, total))

        # 2. Analyze per-model accuracy
        model_accuracy = self.memory.get_routing_accuracy(days=7)
        for model_name, stats in model_accuracy.get("by_model", {}).items():
            total = stats.get("total", 0)
            if total < MIN_SAMPLES:
                continue
            acc = stats.get("predicted_accuracy", 1.0)
            if acc < ACCURACY_THRESHOLD:
                proposals.append(self._propose_model_penalty(model_name, acc, total))

        # 3. Auto-apply if confidence is high enough
        for p in proposals:
            if p.confidence > 0.7:
                p.should_apply = True
                self.apply_mutation(p)

        return proposals

    # ------------------------------------------------------------------
    # Mutation generators
    # ------------------------------------------------------------------

    def _propose_tier_demotion(
        self,
        tier_name: str,
        accuracy: float,
        total_samples: int,
    ) -> MutationProposal:
        """
        Propose demoting a tier: increase its minimum complexity threshold
        so fewer tasks route there.
        """
        tier_enum = getattr(Tier, tier_name, None)
        if tier_enum is None:
            return MutationProposal(
                tier=tier_name, field="ttft_target_ms",
                old_value="unknown", new_value="unknown",
                reason="Tier not found in enum", confidence=0.0,
            )

        spec = TIER_SPECS.get(tier_enum)
        if spec is None:
            return MutationProposal(
                tier=tier_name, field="ttft_target_ms",
                old_value="unknown", new_value="unknown",
                reason="TierSpec not found", confidence=0.0,
            )

        old_ttft = spec.ttft_target_ms
        # Penalize: increase TTFT target (makes router prefer lower tiers)
        new_ttft = int(old_ttft * 1.2)
        confidence = max(0.0, min(1.0, (ACCURACY_THRESHOLD - accuracy) / ACCURACY_THRESHOLD))

        return MutationProposal(
            tier=tier_name,
            field="ttft_target_ms",
            old_value=str(old_ttft),
            new_value=str(new_ttft),
            reason=(
                f"Tier {tier_name} predicted_accuracy={accuracy:.2f} "
                f"over {total_samples} samples.  Penalizing TTFT target "
                f"from {old_ttft} → {new_ttft}ms to reduce routing pressure."
            ),
            confidence=round(confidence, 3),
        )

    def _propose_model_penalty(
        self,
        model_name: str,
        accuracy: float,
        total_samples: int,
    ) -> MutationProposal:
        """
        Propose penalizing a specific model: add it to a temporary
        blocklist in policies.
        """
        confidence = max(0.0, min(1.0, (ACCURACY_THRESHOLD - accuracy) / ACCURACY_THRESHOLD))
        return MutationProposal(
            tier="global",
            field="model_penalty_list",
            old_value="",
            new_value=json.dumps({model_name: accuracy}),
            reason=(
                f"Model {model_name} predicted_accuracy={accuracy:.2f} "
                f"over {total_samples} samples.  Recording penalty."
            ),
            confidence=round(confidence, 3),
        )

    # ------------------------------------------------------------------
    # Mutation application
    # ------------------------------------------------------------------

    def apply_mutation(self, proposal: MutationProposal) -> bool:
        """Apply a mutation to TIER_SPECS and audit it."""
        if not proposal.should_apply:
            return False

        # Record in audit trail BEFORE applying
        self.memory.record_policy_mutation(
            tier=proposal.tier,
            field=proposal.field,
            old_value=proposal.old_value,
            new_value=proposal.new_value,
            reason=proposal.reason,
            confidence=proposal.confidence,
        )

        # Apply to in-memory TIER_SPECS
        tier_enum = getattr(Tier, proposal.tier, None)
        if tier_enum is not None and proposal.field == "ttft_target_ms":
            try:
                new_val = int(proposal.new_value)
                spec = TIER_SPECS.get(tier_enum)
                if spec:
                    # Use object.__setattr__ on the frozen dataclass
                    object.__setattr__(spec, "ttft_target_ms", new_val)
                    logger.info(
                        "POLICY MUTATION APPLIED: %s.%s = %s → %s (%s)",
                        proposal.tier, proposal.field,
                        proposal.old_value, proposal.new_value,
                        proposal.reason[:80],
                    )
                    return True
            except (ValueError, TypeError) as e:
                logger.warning("Failed to apply mutation: %s", e)

        logger.info("Mutation recorded (not applied to TIER_SPECS): %s", proposal.reason[:80])
        return False

    # ------------------------------------------------------------------
    # Heuristic: compute predicted_correct for a completed request
    # ------------------------------------------------------------------

    @staticmethod
    def compute_predicted_correct(
        tool_rounds: int = 0,
        tool_success: bool = False,
        latency_ms: float = 0.0,
        tier: str = "",
    ) -> int:
        """
        Heuristic: did this request likely succeed?
        Returns 1 (likely correct) or 0 (likely failed).
        """
        # Tool executed successfully → high confidence it was correct
        if tool_rounds > 0 and tool_success:
            return 1
        # Very fast response on low tier for simple tasks → likely ok
        if latency_ms < 1000 and tier in ("L0", "L1", "L2"):
            return 1
        # Very slow response → might have struggled
        if latency_ms > 10000:
            return 0
        # Default: assume ok
        return 1
