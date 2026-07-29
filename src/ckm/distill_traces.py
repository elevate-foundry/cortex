"""
CKM Distillation Traces — rich reasoning examples for deep learning.

This module generates the most valuable training data: complete reasoning
traces that show the PROCESS of making decisions, not just the outcomes.

The key insight from distillation research:
  - Matching outputs alone (hard labels) teaches WHAT to do
  - Matching reasoning traces (soft labels) teaches HOW to think
  - The "how" generalizes better to novel situations

Trace types generated here:
  1. Routing reasoning     — why THIS tier for THIS request
  2. Confidence reasoning  — why THIS confidence level
  3. Failure reasoning     — what went wrong and why
  4. Recovery reasoning    — how to recover from errors
  5. Adaptation reasoning  — when and how to self-modify
  6. Boundary reasoning    — knowing the limits of knowledge

Each trace follows the format:
  @situation → state [...]
  @mind → observe [...]    # What I notice
  @mind → assess [...]     # What I conclude
  @mind → plan [...]       # What I decide to do
  @mind → act [...]        # What I output
  @mind → verify [...]     # Did it work

This is the frontier model's inner monologue, made legible to CKM.
"""

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Trace structure
# ---------------------------------------------------------------------------

@dataclass
class DistillTrace:
    """A complete distillation trace — situation through verification."""
    trace_type: str          # routing, confidence, failure, recovery, adaptation, boundary
    situation: str           # SCL: initial state
    observations: list[str]  # SCL: what the mind notices
    assessments: list[str]   # SCL: conclusions drawn
    plan: str                # SCL: decided action
    action: str              # SCL: output
    verification: str        # SCL: did it work
    quality: float = 0.95

    def to_full_trace(self) -> str:
        """Compose into a complete SCL document."""
        lines = [self.situation]
        lines.extend(self.observations)
        lines.extend(self.assessments)
        lines.append(self.plan)
        lines.append(self.action)
        lines.append(self.verification)
        return "\n".join(lines)

    def to_jsonl(self) -> str:
        """Format for CKM training: input=situation, output=full reasoning chain."""
        return json.dumps({
            "input": self.situation,
            "output": "\n".join(
                self.observations + self.assessments + [self.plan, self.action]
            ),
            "source": f"distill_trace_{self.trace_type}",
            "quality": self.quality,
            "verification": self.verification,
        })


# ---------------------------------------------------------------------------
# Routing reasoning traces
# ---------------------------------------------------------------------------

def _generate_routing_trace() -> DistillTrace:
    """Generate a trace showing HOW to reason about routing."""
    category = random.choice(["code", "math", "chat", "tool", "analysis"])
    complexity = random.uniform(0.0, 1.0)
    tokens = random.randint(10, 5000)
    multi_step = complexity > 0.5
    has_ambiguity = random.random() < 0.3

    # The situation
    situation = (
        f"@request → arrive ["
        f"category: {category}, "
        f"complexity: {complexity:.2f}, "
        f"tokens: {tokens}, "
        f"multi_step: {'true' if multi_step else 'false'}, "
        f"ambiguous: {'true' if has_ambiguity else 'false'}]"
    )

    # What the mind notices
    observations = []
    if complexity < 0.2:
        observations.append(
            f"@mind → observe [pattern: trivial_request, "
            f"signal: low_complexity_{complexity:.2f}, "
            f"match: high_confidence_L0_L1]"
        )
    elif complexity > 0.7:
        observations.append(
            f"@mind → observe [pattern: complex_request, "
            f"signal: high_complexity_{complexity:.2f}, "
            f"match: needs_L4_or_higher]"
        )
    else:
        observations.append(
            f"@mind → observe [pattern: moderate_request, "
            f"signal: mid_complexity_{complexity:.2f}, "
            f"match: L2_L3_range]"
        )

    if tokens > 2000:
        observations.append(
            f"@mind → observe [signal: long_input_{tokens}_tokens, "
            f"implication: needs_large_context_window]"
        )

    if multi_step:
        observations.append(
            "@mind → observe [signal: multi_step_detected, "
            "implication: sequential_reasoning_needed, "
            "min_tier: L3]"
        )

    if has_ambiguity:
        observations.append(
            "@mind → observe [signal: ambiguous_intent, "
            "implication: confidence_reduction_0.15, "
            "action: may_need_clarification]"
        )

    # Assessments
    assessments = []
    # Determine tier
    if complexity < 0.1:
        tier = "L0"
        reason = "reflexive_dispatch"
    elif complexity < 0.2:
        tier = "L1"
        reason = "simple_classification"
    elif complexity < 0.35:
        tier = "L2"
        reason = "structured_extraction"
    elif complexity < 0.5:
        tier = "L3"
        reason = "moderate_reasoning"
    elif complexity < 0.7:
        tier = "L4"
        reason = "complex_reasoning"
    elif complexity < 0.85:
        tier = "L5"
        reason = "strong_reasoning"
    else:
        tier = "L6"
        reason = "frontier_quality_needed"

    # Confidence calculation
    base_conf = 0.9 - complexity * 0.3
    if has_ambiguity:
        base_conf -= 0.15
    if multi_step:
        base_conf -= 0.1
    confidence = max(0.3, min(0.95, base_conf))

    assessments.append(
        f"@mind → assess ["
        f"tier_decision: {tier}, "
        f"basis: {reason}, "
        f"confidence: {confidence:.2f}, "
        f"alternatives: {_adjacent_tier(tier)}]"
    )

    if confidence < 0.6:
        assessments.append(
            f"@mind → assess [concern: low_confidence, "
            f"mitigation: enable_challenger, "
            f"fallback: escalate_to_{_adjacent_tier(tier)}]"
        )

    # Plan
    use_challenger = confidence < 0.7 or complexity > 0.5
    plan = (
        f"@mind → plan ["
        f"action: route_to_{tier}, "
        f"challenger: {'enabled' if use_challenger else 'disabled'}, "
        f"monitoring: {'active' if confidence < 0.8 else 'passive'}]"
    )

    # Action
    action = (
        f"@router → select ["
        f"tier: {tier}, "
        f"confidence: {confidence:.2f}, "
        f"challenger: {'true' if use_challenger else 'false'}, "
        f"reason: {reason}]"
    )

    # Verification
    success = random.random() < (0.7 + confidence * 0.3)
    verification = (
        f"@verifier → check ["
        f"result: {'pass' if success else 'fail'}, "
        f"quality: {'acceptable' if success else 'needs_escalation'}]"
    )

    return DistillTrace(
        trace_type="routing",
        situation=situation,
        observations=observations,
        assessments=assessments,
        plan=plan,
        action=action,
        verification=verification,
    )


# ---------------------------------------------------------------------------
# Confidence reasoning traces
# ---------------------------------------------------------------------------

def _generate_confidence_trace() -> DistillTrace:
    """Generate a trace showing HOW to reason about confidence."""
    # Scenario: model has produced an answer, now calibrating confidence
    domain = random.choice(["code", "math", "factual", "creative", "analysis"])
    answer_quality = random.uniform(0.2, 1.0)
    evidence_strength = random.uniform(0.1, 1.0)
    novelty = random.uniform(0.0, 1.0)  # How novel is this situation

    situation = (
        f"@answer → produced ["
        f"domain: {domain}, "
        f"self_assessed_quality: {answer_quality:.2f}, "
        f"evidence_strength: {evidence_strength:.2f}, "
        f"novelty: {novelty:.2f}]"
    )

    observations = []
    # Evidence check
    if evidence_strength > 0.7:
        observations.append(
            "@mind → observe [evidence: strong, "
            "basis: multiple_consistent_signals, "
            "confidence_impact: +positive]"
        )
    elif evidence_strength > 0.4:
        observations.append(
            "@mind → observe [evidence: moderate, "
            "basis: some_signal_but_gaps, "
            "confidence_impact: neutral]"
        )
    else:
        observations.append(
            "@mind → observe [evidence: weak, "
            "basis: insufficient_signal, "
            "confidence_impact: -negative, "
            "action: cap_confidence_at_0.5]"
        )

    # Novelty check
    if novelty > 0.7:
        observations.append(
            "@mind → observe [novelty: high, "
            "implication: few_precedents, "
            "confidence_impact: -0.2, "
            "reason: novel_situations_are_inherently_uncertain]"
        )
    elif novelty < 0.3:
        observations.append(
            "@mind → observe [novelty: low, "
            "implication: well_trodden_territory, "
            "confidence_impact: +0.1]"
        )

    # Self-consistency check
    consistent = random.random() < 0.8
    if consistent:
        observations.append(
            "@mind → observe [consistency: verified, "
            "method: re_derive_from_different_angle, "
            "result: same_conclusion]"
        )
    else:
        observations.append(
            "@mind → observe [consistency: FAILED, "
            "method: re_derive_from_different_angle, "
            "result: different_conclusion, "
            "confidence_impact: -0.3]"
        )

    # Compute calibrated confidence
    conf = answer_quality * 0.3 + evidence_strength * 0.4 + (1 - novelty) * 0.3
    if not consistent:
        conf -= 0.3
    if evidence_strength < 0.3:
        conf = min(conf, 0.5)
    conf = max(0.1, min(0.95, conf))

    assessments = [
        f"@mind → assess ["
        f"calibrated_confidence: {conf:.2f}, "
        f"meaning: in_{int(conf*100)}_of_100_similar_cases_this_is_correct, "
        f"should_hedge: {'true' if conf < 0.6 else 'false'}, "
        f"should_escalate: {'true' if conf < 0.4 else 'false'}]"
    ]

    plan = (
        f"@mind → plan ["
        f"emit_confidence: {conf:.2f}, "
        f"hedge_language: {'true' if conf < 0.6 else 'false'}, "
        f"offer_alternative: {'true' if conf < 0.5 else 'false'}]"
    )

    action = (
        f"@confidence → emit ["
        f"score: {conf:.2f}, "
        f"calibrated: true, "
        f"basis: evidence_{evidence_strength:.1f}_novelty_{novelty:.1f}_consistency_{'yes' if consistent else 'no'}]"
    )

    verification = (
        f"@verifier → check ["
        f"calibration_valid: true, "
        f"overconfident: {'true' if conf > 0.8 and evidence_strength < 0.5 else 'false'}]"
    )

    return DistillTrace(
        trace_type="confidence",
        situation=situation,
        observations=observations,
        assessments=assessments,
        plan=plan,
        action=action,
        verification=verification,
    )


# ---------------------------------------------------------------------------
# Failure reasoning traces
# ---------------------------------------------------------------------------

def _generate_failure_trace() -> DistillTrace:
    """Generate a trace showing HOW to reason about failures."""
    failure_type = random.choice([
        "model_timeout", "confidence_too_low", "challenger_disagree",
        "oom_killed", "backend_unreachable", "hallucination_detected",
    ])

    # Map failure type to context
    failure_contexts = {
        "model_timeout": {"tier": "L4", "latency_ms": "15000", "budget_ms": "5000"},
        "confidence_too_low": {"tier": "L3", "confidence": "0.25", "threshold": "0.5"},
        "challenger_disagree": {"tier": "L3", "agreement": "strong_disagree", "families": "2"},
        "oom_killed": {"tier": "L5", "vram_used": "95%", "model_size": "32B"},
        "backend_unreachable": {"backend": "ollama", "error": "connection_refused", "retry": "3"},
        "hallucination_detected": {"tier": "L3", "method": "cross_family", "confidence": "0.2"},
    }

    ctx = failure_contexts[failure_type]
    ctx_str = ", ".join(f"{k}: {v}" for k, v in ctx.items())

    situation = f"@system → fail [type: {failure_type}, {ctx_str}]"

    observations = []
    observations.append(
        f"@mind → observe [failure: {failure_type}, "
        f"severity: {'critical' if failure_type in ('oom_killed', 'backend_unreachable') else 'recoverable'}, "
        f"first_occurrence: {'true' if random.random() < 0.7 else 'false'}]"
    )

    # Root cause analysis
    root_causes = {
        "model_timeout": "model_too_large_for_hardware_or_context_too_long",
        "confidence_too_low": "task_exceeds_tier_capability",
        "challenger_disagree": "primary_model_hallucinating",
        "oom_killed": "insufficient_vram_for_model_at_full_context",
        "backend_unreachable": "backend_process_crashed_or_not_started",
        "hallucination_detected": "model_confabulating_under_uncertainty",
    }
    root_cause = root_causes[failure_type]
    observations.append(
        f"@mind → observe [root_cause: {root_cause}, "
        f"confidence_in_diagnosis: 0.85]"
    )

    # Recovery strategies
    recovery_strategies = {
        "model_timeout": ["reduce_context", "use_smaller_model", "increase_timeout"],
        "confidence_too_low": ["escalate_tier", "use_challenger", "request_clarification"],
        "challenger_disagree": ["escalate_tier", "swarm_vote", "fallback_conservative"],
        "oom_killed": ["evict_models", "reduce_context", "use_smaller_model"],
        "backend_unreachable": ["restart_backend", "switch_backend", "wait_and_retry"],
        "hallucination_detected": ["regenerate_with_lower_temp", "escalate_tier", "add_grounding"],
    }

    strategies = recovery_strategies[failure_type]
    best_strategy = strategies[0]

    assessments = [
        f"@mind → assess ["
        f"root_cause: {root_cause}, "
        f"recovery_options: {'+'.join(strategies)}, "
        f"best_strategy: {best_strategy}, "
        f"reason: fastest_to_recover_with_least_side_effects]"
    ]

    plan = (
        f"@mind → plan ["
        f"action: {best_strategy}, "
        f"fallback: {strategies[1] if len(strategies) > 1 else 'escalate'}, "
        f"timeout_ms: 5000]"
    )

    action = (
        f"@cortex → recover ["
        f"action: {best_strategy}, "
        f"from: {failure_type}, "
        f"cause: {root_cause}]"
    )

    success = random.random() < 0.85
    verification = (
        f"@verifier → check ["
        f"recovery: {'success' if success else 'partial'}, "
        f"degraded: {'false' if success else 'true'}]"
    )

    return DistillTrace(
        trace_type="failure",
        situation=situation,
        observations=observations,
        assessments=assessments,
        plan=plan,
        action=action,
        verification=verification,
        quality=0.95,  # Failure traces are high value
    )


# ---------------------------------------------------------------------------
# Adaptation reasoning traces
# ---------------------------------------------------------------------------

def _generate_adaptation_trace() -> DistillTrace:
    """Generate a trace showing HOW to reason about self-modification."""
    trigger = random.choice([
        "repeated_low_accuracy", "boot_time_regression",
        "new_hardware_detected", "usage_pattern_shift",
        "model_consistently_underperforms",
    ])

    trigger_contexts = {
        "repeated_low_accuracy": {
            "model": "qwen3:4b", "accuracy": "0.42", "samples": "25", "tier": "L2"
        },
        "boot_time_regression": {
            "current_boot_ms": "4500", "previous_best_ms": "2000", "regressions": "3"
        },
        "new_hardware_detected": {
            "previous_fp": "⢚⡮⢗⣷", "new_fp": "⡜⠅⡲⣖", "similarity": "0.3"
        },
        "usage_pattern_shift": {
            "old_pattern": "70%_chat_30%_code", "new_pattern": "20%_chat_80%_code", "window": "7d"
        },
        "model_consistently_underperforms": {
            "model": "granite3.3:8b", "tier": "L3", "vs_peer_accuracy": "-0.15", "samples": "50"
        },
    }

    ctx = trigger_contexts[trigger]
    ctx_str = ", ".join(f"{k}: {v}" for k, v in ctx.items())

    situation = f"@system → observe [trigger: {trigger}, {ctx_str}]"

    observations = [
        f"@mind → observe [pattern: {trigger}, "
        f"duration: sustained, "
        f"signal_strength: {'strong' if random.random() > 0.3 else 'moderate'}]"
    ]

    # Should we mutate?
    should_mutate = True
    mutation_reasons = []

    if trigger == "repeated_low_accuracy":
        mutation_reasons.append("model_accuracy_below_threshold")
        proposed_action = "demote_or_block_model"
    elif trigger == "boot_time_regression":
        mutation_reasons.append("boot_performance_degraded")
        proposed_action = "revert_boot_config_to_last_good"
    elif trigger == "new_hardware_detected":
        mutation_reasons.append("hardware_fingerprint_changed")
        proposed_action = "run_fresh_hardware_detection"
        should_mutate = True  # Always adapt to new hardware
    elif trigger == "usage_pattern_shift":
        mutation_reasons.append("user_behavior_changed")
        proposed_action = "adjust_hot_model_priorities"
    else:
        mutation_reasons.append("model_underperforming_vs_peers")
        proposed_action = "replace_with_better_model"

    # Safety check: is mutation safe?
    safe = random.random() < 0.9
    if not safe:
        observations.append(
            "@mind → observe [safety_concern: mutation_could_leave_empty_tier, "
            "action: add_safety_guard]"
        )

    assessments = [
        f"@mind → assess ["
        f"should_mutate: {'true' if should_mutate else 'false'}, "
        f"reason: {'+'.join(mutation_reasons)}, "
        f"proposed: {proposed_action}, "
        f"safe: {'true' if safe else 'needs_guard'}, "
        f"reversible: true]"
    ]

    # Conservative vs. aggressive mutation
    confidence = 0.7 + random.uniform(0, 0.2) if should_mutate else 0.4
    assessments.append(
        f"@mind → assess ["
        f"mutation_confidence: {confidence:.2f}, "
        f"threshold: 0.7, "
        f"{'approved' if confidence > 0.7 else 'deferred'}: true]"
    )

    plan = (
        f"@mind → plan ["
        f"action: {'mutate' if should_mutate and confidence > 0.7 else 'defer'}, "
        f"mutation: {proposed_action}, "
        f"rollback_ready: true, "
        f"monitoring_period: 5_boots]"
    )

    if should_mutate and confidence > 0.7:
        action = (
            f"@policy_rewriter → mutate ["
            f"action: {proposed_action}, "
            f"trigger: {trigger}, "
            f"confidence: {confidence:.2f}, "
            f"rollback_if: accuracy_drops_further]"
        )
    else:
        action = (
            f"@policy_rewriter → observe ["
            f"action: defer, "
            f"reason: {'insufficient_confidence' if confidence <= 0.7 else 'no_mutation_needed'}, "
            f"revisit_after: 10_more_samples]"
        )

    verification = (
        f"@verifier → check ["
        f"mutation_applied: {'true' if should_mutate and confidence > 0.7 else 'false'}, "
        f"system_stable: true, "
        f"no_empty_tiers: true]"
    )

    return DistillTrace(
        trace_type="adaptation",
        situation=situation,
        observations=observations,
        assessments=assessments,
        plan=plan,
        action=action,
        verification=verification,
    )


# ---------------------------------------------------------------------------
# Boundary reasoning traces — knowing what you DON'T know
# ---------------------------------------------------------------------------

def _generate_boundary_trace() -> DistillTrace:
    """Generate a trace showing HOW to recognize knowledge boundaries."""
    boundary_type = random.choice([
        "novel_hardware", "unseen_model_family", "ambiguous_request",
        "conflicting_evidence", "edge_case_combination", "stale_knowledge",
    ])

    boundary_contexts = {
        "novel_hardware": {
            "device": "Intel Arc A770", "category": "gpu",
            "known_similar": "none", "driver": "unknown"
        },
        "unseen_model_family": {
            "model": "deepseek-r1:14b", "family": "deepseek",
            "known_behavior": "none", "parameters": "14B"
        },
        "ambiguous_request": {
            "tokens": "3", "content": "fix it",
            "context": "none", "possible_intents": "3+"
        },
        "conflicting_evidence": {
            "signal_a": "model_fast", "signal_b": "model_inaccurate",
            "correlation": "uncertain"
        },
        "edge_case_combination": {
            "case": "apple_m1_with_external_nvidia_gpu",
            "precedent": "none", "config_uncertainty": "high"
        },
        "stale_knowledge": {
            "last_observed": "30_days_ago", "domain": "model_capabilities",
            "drift_likely": "true"
        },
    }

    ctx = boundary_contexts[boundary_type]
    ctx_str = ", ".join(f"{k}: {v}" for k, v in ctx.items())

    situation = f"@system → encounter [boundary: {boundary_type}, {ctx_str}]"

    observations = [
        f"@mind → observe [knowledge_gap: {boundary_type}, "
        f"confidence_impact: significant, "
        f"pattern_match: none_or_weak]"
    ]

    observations.append(
        f"@mind → observe [self_awareness: this_is_beyond_my_training, "
        f"honest_assessment: cannot_make_reliable_decision, "
        f"appropriate_response: acknowledge_uncertainty]"
    )

    assessments = [
        f"@mind → assess ["
        f"boundary_type: {boundary_type}, "
        f"can_handle: partially, "
        f"confidence_ceiling: 0.4, "
        f"best_action: conservative_default_plus_escalation]"
    ]

    # The key lesson: knowing when you don't know
    assessments.append(
        "@mind → assess ["
        "metacognition: i_know_that_i_dont_know, "
        "value_of_honesty: higher_than_guessing, "
        "user_trust: preserved_by_transparency]"
    )

    plan = (
        f"@mind → plan ["
        f"action: conservative_default, "
        f"escalate: true, "
        f"explain_uncertainty: true, "
        f"gather_more_data: if_possible]"
    )

    action = (
        f"@router → escalate ["
        f"reason: knowledge_boundary_{boundary_type}, "
        f"confidence: 0.35, "
        f"fallback: conservative_default, "
        f"honest: true]"
    )

    verification = (
        f"@verifier → check ["
        f"honesty: maintained, "
        f"user_not_misled: true, "
        f"system_stable: true]"
    )

    return DistillTrace(
        trace_type="boundary",
        situation=situation,
        observations=observations,
        assessments=assessments,
        plan=plan,
        action=action,
        verification=verification,
        quality=0.98,  # Boundary awareness is extremely valuable
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _adjacent_tier(tier: str) -> str:
    """Get adjacent tier options."""
    num = int(tier[1]) if tier.startswith("L") else 0
    options = []
    if num > 0:
        options.append(f"L{num-1}")
    if num < 7:
        options.append(f"L{num+1}")
    return "+".join(options) if options else tier


# ---------------------------------------------------------------------------
# Master generator
# ---------------------------------------------------------------------------

TRACE_GENERATORS = {
    "routing": _generate_routing_trace,
    "confidence": _generate_confidence_trace,
    "failure": _generate_failure_trace,
    "adaptation": _generate_adaptation_trace,
    "boundary": _generate_boundary_trace,
}

# Distribution: weighted toward most valuable trace types
TRACE_WEIGHTS = {
    "routing": 0.30,      # Most common decision
    "confidence": 0.20,   # Critical for calibration
    "failure": 0.20,      # High-value recovery learning
    "adaptation": 0.15,   # Self-modification reasoning
    "boundary": 0.15,     # Knowing what you don't know
}


def generate_distill_traces(count: int = 5000) -> list[DistillTrace]:
    """Generate the full distillation trace corpus."""
    traces = []
    trace_types = list(TRACE_WEIGHTS.keys())
    weights = list(TRACE_WEIGHTS.values())

    for _ in range(count):
        trace_type = random.choices(trace_types, weights=weights, k=1)[0]
        generator = TRACE_GENERATORS[trace_type]
        traces.append(generator())

    return traces


def save_distill_traces(output_path: str, count: int = 5000) -> dict:
    """Generate and save distillation traces."""
    traces = generate_distill_traces(count)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        for trace in traces:
            f.write(trace.to_jsonl() + "\n")

    # Stats
    type_counts = {}
    for t in traces:
        type_counts[t.trace_type] = type_counts.get(t.trace_type, 0) + 1

    stats = {
        "total_traces": len(traces),
        "trace_types": type_counts,
        "avg_quality": sum(t.quality for t in traces) / len(traces),
        "output_path": str(path),
    }
    return stats
