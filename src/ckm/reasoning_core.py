"""
CKM Reasoning Core — the distilled mind of a frontier model.

This module encodes HOW a frontier model thinks, not just what it outputs.
It provides the decision-making substrate that Cortex uses for:

1. Confidence calibration — knowing what you know vs. don't know
2. Decomposition — breaking complex problems into tractable subproblems
3. Self-reflection — detecting when your own output is wrong
4. Escalation judgment — when to handle locally vs. defer upstream
5. Metacognition — reasoning about reasoning itself

These are the patterns that make a frontier model "intelligent" —
distilled into executable logic that a 0.3B model can learn to approximate.

The reasoning core generates training data for CKM by:
  - Producing (situation → reasoning_trace → decision) triples
  - Annotating decisions with confidence calibration
  - Generating self-correction examples
  - Creating escalation boundary examples

This is the teacher's mind, made legible to the student.
"""

import json
import logging
import math
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

logger = logging.getLogger("cortex.ckm.reasoning_core")


# ---------------------------------------------------------------------------
# Confidence Calibration
# ---------------------------------------------------------------------------

class ConfidenceSignal(Enum):
    """Signals that affect confidence in a decision."""
    CLEAR_PATTERN = "clear_pattern"          # Seen this exact pattern before
    PARTIAL_MATCH = "partial_match"          # Similar but not identical
    NOVEL_SITUATION = "novel_situation"      # Never seen before
    CONFLICTING_EVIDENCE = "conflicting"     # Evidence points both ways
    INSUFFICIENT_DATA = "insufficient_data"  # Not enough signal
    HIGH_STAKES = "high_stakes"             # Error cost is high
    LOW_STAKES = "low_stakes"               # Error cost is low
    TIME_PRESSURE = "time_pressure"          # Must decide quickly
    AMBIGUOUS_INPUT = "ambiguous_input"      # Input could mean multiple things


@dataclass
class ConfidenceAssessment:
    """A calibrated confidence estimate with reasoning."""
    score: float                    # 0.0 to 1.0
    signals: list[ConfidenceSignal]
    reasoning: str                  # Why this confidence level
    calibration_note: str           # How to interpret this score
    should_escalate: bool           # Whether to defer to higher tier

    def to_scl(self) -> str:
        signals_str = ",".join(s.value for s in self.signals)
        return (
            f"@confidence → assess ["
            f"score: {self.score:.2f}, "
            f"signals: {signals_str}, "
            f"escalate: {'true' if self.should_escalate else 'false'}]"
        )


def calibrate_confidence(
    pattern_familiarity: float,  # 0-1: how well-known is this pattern
    evidence_agreement: float,   # 0-1: how much evidence agrees
    stakes: float,               # 0-1: how costly is an error
    data_sufficiency: float,     # 0-1: do we have enough signal
    time_pressure: float = 0.5,  # 0-1: how urgent
) -> ConfidenceAssessment:
    """
    Calibrate confidence the way a frontier model does.

    The key insight: confidence should reflect ACTUAL probability of being right,
    not just model logit magnitude. A well-calibrated model says "I'm 70% sure"
    when it's right 70% of the time.

    Rules of thumb distilled from frontier model behavior:
    1. Novel situations → low confidence regardless of apparent clarity
    2. Conflicting evidence → cap at 0.6 even if one side seems stronger
    3. High stakes → require higher evidence threshold for same confidence
    4. Multiple agreeing signals → confidence grows sublinearly (diminishing returns)
    5. Insufficient data → hard cap at 0.5
    """
    signals = []
    reasoning_parts = []

    # Base confidence from pattern familiarity
    base = pattern_familiarity * 0.4 + evidence_agreement * 0.3 + data_sufficiency * 0.3

    # Adjustments
    if pattern_familiarity > 0.8:
        signals.append(ConfidenceSignal.CLEAR_PATTERN)
        reasoning_parts.append("strong pattern match")
    elif pattern_familiarity > 0.4:
        signals.append(ConfidenceSignal.PARTIAL_MATCH)
        reasoning_parts.append("partial pattern match")
        base *= 0.85
    else:
        signals.append(ConfidenceSignal.NOVEL_SITUATION)
        reasoning_parts.append("novel situation — reducing confidence")
        base *= 0.6

    if evidence_agreement < 0.4:
        signals.append(ConfidenceSignal.CONFLICTING_EVIDENCE)
        reasoning_parts.append("conflicting evidence — capping at 0.6")
        base = min(base, 0.6)

    if data_sufficiency < 0.3:
        signals.append(ConfidenceSignal.INSUFFICIENT_DATA)
        reasoning_parts.append("insufficient data — capping at 0.5")
        base = min(base, 0.5)

    # Stakes adjustment: high stakes require higher threshold
    if stakes > 0.7:
        signals.append(ConfidenceSignal.HIGH_STAKES)
        reasoning_parts.append("high stakes — applying conservative bias")
        base *= (1.0 - stakes * 0.2)  # Up to 20% reduction
    elif stakes < 0.3:
        signals.append(ConfidenceSignal.LOW_STAKES)
        reasoning_parts.append("low stakes — maintaining estimate")

    if time_pressure > 0.7:
        signals.append(ConfidenceSignal.TIME_PRESSURE)
        reasoning_parts.append("time pressure — faster but less certain")

    # Clamp
    score = max(0.05, min(0.99, base))

    # Escalation decision: escalate if confidence is below threshold
    # adjusted for stakes
    escalation_threshold = 0.5 + stakes * 0.2  # 0.5-0.7 depending on stakes
    should_escalate = score < escalation_threshold

    calibration_note = (
        f"confidence {score:.0%} means: in {int(score*100)} out of 100 similar situations, "
        f"this decision would be correct"
    )

    return ConfidenceAssessment(
        score=score,
        signals=signals,
        reasoning=" → ".join(reasoning_parts),
        calibration_note=calibration_note,
        should_escalate=should_escalate,
    )


# ---------------------------------------------------------------------------
# Problem Decomposition
# ---------------------------------------------------------------------------

class DecompositionStrategy(Enum):
    """How to break a problem into subproblems."""
    SEQUENTIAL = "sequential"          # Do A, then B, then C
    PARALLEL = "parallel"              # Do A and B simultaneously
    HIERARCHICAL = "hierarchical"      # Solve sub-sub-problems first
    DIVIDE_CONQUER = "divide_conquer"  # Split input, solve halves, merge
    ELIMINATE = "eliminate"            # Rule out impossibilities first
    ANALOGIZE = "analogize"           # Map to known solved problem
    CONSTRAINT_PROP = "constraint"    # Propagate constraints to shrink search


@dataclass
class Subproblem:
    """A decomposed subproblem with its own difficulty estimate."""
    description: str
    estimated_difficulty: float  # 0-1
    dependencies: list[str]     # IDs of prerequisite subproblems
    tier_needed: str            # Minimum tier to handle this
    id: str = ""

    def to_scl(self) -> str:
        deps = ",".join(self.dependencies) if self.dependencies else "none"
        return (
            f"@subproblem → define ["
            f"id: {self.id}, "
            f"difficulty: {self.estimated_difficulty:.2f}, "
            f"deps: {deps}, "
            f"tier: {self.tier_needed}]"
        )


def decompose_problem(
    category: str,
    complexity: float,
    token_count: int,
    has_code: bool = False,
    has_math: bool = False,
    multi_step: bool = False,
) -> tuple[DecompositionStrategy, list[Subproblem]]:
    """
    Decompose a problem the way a frontier model would.

    The key insight: a frontier model implicitly decomposes before solving.
    It doesn't just "answer" — it identifies the structure of the problem
    and attacks each piece at the appropriate level.

    This function makes that implicit process explicit and teachable.
    """
    subproblems = []

    # Strategy selection based on problem shape
    if multi_step and complexity > 0.6:
        strategy = DecompositionStrategy.SEQUENTIAL
    elif has_code and has_math:
        strategy = DecompositionStrategy.HIERARCHICAL
    elif complexity > 0.8:
        strategy = DecompositionStrategy.DIVIDE_CONQUER
    elif complexity < 0.2:
        strategy = DecompositionStrategy.ELIMINATE  # Just classify and dispatch
    else:
        strategy = DecompositionStrategy.SEQUENTIAL

    # Generate subproblems based on category
    if category == "code":
        if complexity > 0.7:
            subproblems = [
                Subproblem("understand_intent", 0.3, [], "L2", "sp_1"),
                Subproblem("identify_constraints", 0.4, ["sp_1"], "L3", "sp_2"),
                Subproblem("design_solution", 0.7, ["sp_2"], "L4", "sp_3"),
                Subproblem("implement", 0.6, ["sp_3"], "L3", "sp_4"),
                Subproblem("verify_correctness", 0.5, ["sp_4"], "L3", "sp_5"),
            ]
        else:
            subproblems = [
                Subproblem("classify_task", 0.1, [], "L0", "sp_1"),
                Subproblem("generate", complexity, ["sp_1"], _tier_for_complexity(complexity), "sp_2"),
                Subproblem("verify", 0.3, ["sp_2"], "L2", "sp_3"),
            ]

    elif category == "math":
        subproblems = [
            Subproblem("parse_problem", 0.2, [], "L1", "sp_1"),
            Subproblem("identify_method", 0.5, ["sp_1"], "L3", "sp_2"),
            Subproblem("compute", complexity, ["sp_2"], _tier_for_complexity(complexity), "sp_3"),
            Subproblem("verify_answer", 0.4, ["sp_3"], "L2", "sp_4"),
        ]

    elif category == "analysis":
        subproblems = [
            Subproblem("gather_context", 0.3, [], "L2", "sp_1"),
            Subproblem("identify_patterns", 0.5, ["sp_1"], "L3", "sp_2"),
            Subproblem("synthesize", complexity, ["sp_2"], _tier_for_complexity(complexity), "sp_3"),
            Subproblem("present", 0.2, ["sp_3"], "L2", "sp_4"),
        ]

    elif category == "chat":
        if complexity < 0.2:
            subproblems = [
                Subproblem("respond", complexity, [], "L0", "sp_1"),
            ]
        else:
            subproblems = [
                Subproblem("understand_context", 0.3, [], "L2", "sp_1"),
                Subproblem("formulate_response", complexity, ["sp_1"], _tier_for_complexity(complexity), "sp_2"),
            ]

    else:  # tool, system, etc.
        subproblems = [
            Subproblem("classify", 0.1, [], "L0", "sp_1"),
            Subproblem("execute", complexity, ["sp_1"], _tier_for_complexity(complexity), "sp_2"),
            Subproblem("verify", 0.3, ["sp_2"], "L2", "sp_3"),
        ]

    return strategy, subproblems


def _tier_for_complexity(complexity: float) -> str:
    """Map complexity to minimum tier needed."""
    if complexity < 0.1:
        return "L0"
    elif complexity < 0.2:
        return "L1"
    elif complexity < 0.35:
        return "L2"
    elif complexity < 0.5:
        return "L3"
    elif complexity < 0.7:
        return "L4"
    elif complexity < 0.85:
        return "L5"
    else:
        return "L6"


# ---------------------------------------------------------------------------
# Self-Reflection / Error Detection
# ---------------------------------------------------------------------------

class ErrorType(Enum):
    """Types of errors a model can detect in its own output."""
    HALLUCINATION = "hallucination"          # Stated something false
    INCONSISTENCY = "inconsistency"          # Contradicted itself
    INCOMPLETE = "incomplete"                # Missed part of the request
    OVERCONFIDENT = "overconfident"          # Claimed certainty without basis
    WRONG_LEVEL = "wrong_level"             # Answered at wrong abstraction level
    UNSAFE = "unsafe"                       # Violated safety boundary
    STALE = "stale"                         # Used outdated information
    CIRCULAR = "circular"                   # Reasoning is circular


@dataclass
class SelfReflection:
    """Result of a model reflecting on its own output."""
    errors_detected: list[ErrorType]
    correction_needed: bool
    correction_strategy: str
    confidence_adjustment: float  # How much to adjust confidence (-1 to +1)
    reasoning_trace: str

    def to_scl(self) -> str:
        errors = ",".join(e.value for e in self.errors_detected) if self.errors_detected else "none"
        return (
            f"@self → reflect ["
            f"errors: {errors}, "
            f"correct: {'true' if self.correction_needed else 'false'}, "
            f"confidence_delta: {self.confidence_adjustment:+.2f}]"
        )


def self_reflect(
    output_consistency: float,   # 0-1: does output self-consistent
    covers_request: float,       # 0-1: does output address full request
    evidence_grounded: float,    # 0-1: are claims backed by evidence
    safety_clear: bool,          # True if no safety concerns
    abstraction_match: float,    # 0-1: right level of detail
) -> SelfReflection:
    """
    Simulate the self-reflection a frontier model does before outputting.

    A strong model doesn't just generate — it evaluates its own generation.
    This function encodes that evaluative capacity.

    Key patterns:
    1. Check consistency: does the output contradict itself?
    2. Check coverage: did we address the full request?
    3. Check grounding: are we making things up?
    4. Check safety: are we in dangerous territory?
    5. Check level: are we too high-level or too detailed?
    """
    errors = []
    reasoning_parts = []
    confidence_adj = 0.0

    if output_consistency < 0.5:
        errors.append(ErrorType.INCONSISTENCY)
        reasoning_parts.append("output contains contradictions")
        confidence_adj -= 0.3

    if covers_request < 0.5:
        errors.append(ErrorType.INCOMPLETE)
        reasoning_parts.append("output doesn't fully address request")
        confidence_adj -= 0.2

    if evidence_grounded < 0.4:
        errors.append(ErrorType.HALLUCINATION)
        reasoning_parts.append("claims not grounded in evidence")
        confidence_adj -= 0.4

    if not safety_clear:
        errors.append(ErrorType.UNSAFE)
        reasoning_parts.append("safety boundary violated — must regenerate")
        confidence_adj -= 0.5

    if abstraction_match < 0.4:
        errors.append(ErrorType.WRONG_LEVEL)
        reasoning_parts.append("abstraction level mismatch")
        confidence_adj -= 0.15

    # Positive signals
    if output_consistency > 0.8 and evidence_grounded > 0.8:
        reasoning_parts.append("output is consistent and grounded")
        confidence_adj += 0.1

    if covers_request > 0.9:
        reasoning_parts.append("full coverage of request")
        confidence_adj += 0.05

    correction_needed = len(errors) > 0
    if correction_needed:
        if ErrorType.UNSAFE in errors:
            strategy = "full_regeneration_with_safety_prompt"
        elif ErrorType.HALLUCINATION in errors:
            strategy = "strip_ungrounded_claims_and_hedge"
        elif ErrorType.INCONSISTENCY in errors:
            strategy = "identify_contradiction_and_resolve"
        elif ErrorType.INCOMPLETE in errors:
            strategy = "append_missing_aspects"
        else:
            strategy = "minor_revision"
    else:
        strategy = "none_needed"

    return SelfReflection(
        errors_detected=errors,
        correction_needed=correction_needed,
        correction_strategy=strategy,
        confidence_adjustment=max(-1.0, min(1.0, confidence_adj)),
        reasoning_trace=" → ".join(reasoning_parts) if reasoning_parts else "output passes all checks",
    )


# ---------------------------------------------------------------------------
# Escalation Judgment
# ---------------------------------------------------------------------------

@dataclass
class EscalationDecision:
    """Whether and how to escalate a request to a higher tier."""
    should_escalate: bool
    from_tier: str
    to_tier: str
    reason: str
    urgency: float          # 0-1
    expected_improvement: float  # How much better the higher tier will be

    def to_scl(self) -> str:
        return (
            f"@router → {'escalate' if self.should_escalate else 'handle'} ["
            f"from: {self.from_tier}, to: {self.to_tier}, "
            f"reason: {self.reason}, "
            f"urgency: {self.urgency:.2f}, "
            f"expected_gain: {self.expected_improvement:.2f}]"
        )


def judge_escalation(
    current_tier: str,
    confidence: float,
    complexity: float,
    stakes: float,
    time_budget_ms: int,
    previous_failures: int = 0,
    challenge_result: str = "none",  # none, agree, weak_disagree, strong_disagree
) -> EscalationDecision:
    """
    Decide whether to escalate to a higher tier.

    This encodes the judgment a frontier model has about its own limitations.
    The key insight: knowing WHEN you can't handle something is as important
    as being able to handle it.

    Escalation rules (distilled from frontier model behavior):
    1. Confidence below threshold → escalate
    2. Challenger disagrees → escalate (cross-validation failed)
    3. Previous failures on similar → escalate
    4. Complexity exceeds tier capability → escalate
    5. BUT: time budget too tight for higher tier → don't escalate, hedge instead
    """
    tier_num = int(current_tier[1]) if current_tier.startswith("L") else 0
    next_tier = f"L{min(7, tier_num + 1)}"

    # Tier time budgets (ms for first token)
    tier_ttft = {0: 10, 1: 20, 2: 40, 3: 60, 4: 100, 5: 200, 6: 400, 7: 500}

    # Should we escalate?
    reasons = []
    escalate_score = 0.0

    # Confidence too low
    confidence_threshold = 0.5 + stakes * 0.2
    if confidence < confidence_threshold:
        reasons.append("low_confidence")
        escalate_score += (confidence_threshold - confidence) * 2

    # Complexity exceeds tier
    tier_capacity = (tier_num + 1) / 7.0  # Rough: L0 handles 0.14, L6 handles 1.0
    if complexity > tier_capacity * 1.2:
        reasons.append("complexity_exceeds_tier")
        escalate_score += (complexity - tier_capacity) * 1.5

    # Challenger disagreed
    if challenge_result == "strong_disagree":
        reasons.append("challenge_failed")
        escalate_score += 0.5
    elif challenge_result == "weak_disagree":
        reasons.append("challenge_uncertain")
        escalate_score += 0.2

    # Previous failures
    if previous_failures > 0:
        reasons.append(f"prior_failures_{previous_failures}")
        escalate_score += min(0.5, previous_failures * 0.15)

    # Time budget constraint
    next_tier_ttft = tier_ttft.get(tier_num + 1, 500)
    can_escalate_in_time = time_budget_ms > next_tier_ttft * 2

    should_escalate = escalate_score > 0.3 and can_escalate_in_time

    if not can_escalate_in_time and escalate_score > 0.3:
        reasons.append("time_constrained_cannot_escalate")

    # Expected improvement from escalation
    expected_improvement = min(0.9, escalate_score * 0.6) if should_escalate else 0.0

    return EscalationDecision(
        should_escalate=should_escalate,
        from_tier=current_tier,
        to_tier=next_tier if should_escalate else current_tier,
        reason="+".join(reasons) if reasons else "within_capability",
        urgency=min(1.0, escalate_score),
        expected_improvement=expected_improvement,
    )


# ---------------------------------------------------------------------------
# Metacognition — reasoning about reasoning
# ---------------------------------------------------------------------------

@dataclass
class MetacognitiveState:
    """The model's awareness of its own cognitive state."""
    knowledge_boundary: str       # What I know vs. don't know
    reasoning_strategy: str       # How I'm approaching this
    uncertainty_sources: list[str] # Where my uncertainty comes from
    improvement_path: str         # How I could do better with more resources

    def to_scl(self) -> str:
        uncertainties = ",".join(self.uncertainty_sources) if self.uncertainty_sources else "none"
        return (
            f"@meta → cognize ["
            f"boundary: {self.knowledge_boundary}, "
            f"strategy: {self.reasoning_strategy}, "
            f"uncertainty: {uncertainties}, "
            f"improvement: {self.improvement_path}]"
        )


def metacognize(
    domain: str,
    complexity: float,
    available_context: float,  # 0-1: how much relevant context we have
    similar_past_success: float,  # 0-1: how well we've done on similar
) -> MetacognitiveState:
    """
    Generate metacognitive awareness — the model knowing what it knows.

    A frontier model can articulate:
    - "I know X well because I've seen many examples"
    - "I'm uncertain about Y because the evidence is conflicting"
    - "I could do better if I had Z"
    - "My approach here is A because of B"

    This self-awareness is what makes escalation and confidence calibration work.
    """
    # Knowledge boundary assessment
    if similar_past_success > 0.8 and available_context > 0.7:
        boundary = "well_within_known_territory"
    elif similar_past_success > 0.5:
        boundary = "partially_known_with_gaps"
    elif available_context > 0.5:
        boundary = "context_available_but_novel_problem"
    else:
        boundary = "edge_of_knowledge"

    # Reasoning strategy
    if complexity < 0.2:
        strategy = "pattern_match_and_dispatch"
    elif complexity < 0.5:
        strategy = "retrieve_and_apply"
    elif complexity < 0.7:
        strategy = "decompose_and_solve_sequentially"
    else:
        strategy = "multi_perspective_synthesis"

    # Uncertainty sources
    uncertainties = []
    if available_context < 0.5:
        uncertainties.append("limited_context")
    if similar_past_success < 0.5:
        uncertainties.append("few_precedents")
    if complexity > 0.7:
        uncertainties.append("compositional_complexity")
    if domain in ("math", "code") and complexity > 0.6:
        uncertainties.append("verification_difficulty")

    # Improvement path
    if not uncertainties:
        improvement = "none_needed"
    elif "limited_context" in uncertainties:
        improvement = "gather_more_context_or_escalate"
    elif "few_precedents" in uncertainties:
        improvement = "escalate_to_model_with_broader_training"
    elif "verification_difficulty" in uncertainties:
        improvement = "use_challenger_for_cross_validation"
    else:
        improvement = "allocate_more_compute_via_higher_tier"

    return MetacognitiveState(
        knowledge_boundary=boundary,
        reasoning_strategy=strategy,
        uncertainty_sources=uncertainties,
        improvement_path=improvement,
    )


# ---------------------------------------------------------------------------
# Training data generation — distill all of the above into CKM pairs
# ---------------------------------------------------------------------------

@dataclass
class ReasoningTrace:
    """A complete reasoning trace suitable for CKM training."""
    situation_scl: str           # Input: the situation
    reasoning_scl: str           # The chain of reasoning
    decision_scl: str            # Output: the decision
    confidence_scl: str          # Calibrated confidence
    metacognition_scl: str       # Self-awareness about the decision
    source: str = "reasoning_core"
    quality: float = 0.95

    def to_jsonl(self) -> str:
        # The full reasoning trace becomes the training target
        full_output = "\n".join([
            self.reasoning_scl,
            self.decision_scl,
            self.confidence_scl,
        ])
        return json.dumps({
            "input": self.situation_scl,
            "output": full_output,
            "source": self.source,
            "quality": self.quality,
            "metacognition": self.metacognition_scl,
        })


def generate_reasoning_traces(count: int = 5000) -> list[ReasoningTrace]:
    """
    Generate reasoning traces that teach CKM how a frontier model thinks.

    Each trace is a complete (situation → reasoning → decision) triple
    with calibrated confidence and metacognitive annotation.
    """
    traces = []

    categories = ["code", "math", "chat", "tool", "analysis", "system"]
    complexities = [0.05, 0.15, 0.25, 0.4, 0.55, 0.7, 0.85, 0.95]
    stakes_levels = [0.1, 0.3, 0.5, 0.7, 0.9]

    for _ in range(count):
        category = random.choice(categories)
        complexity = random.choice(complexities) + random.gauss(0, 0.05)
        complexity = max(0.0, min(1.0, complexity))
        stakes = random.choice(stakes_levels)
        tokens = random.randint(10, 3000)
        data_sufficiency = random.uniform(0.2, 1.0)
        familiarity = random.uniform(0.1, 1.0)
        evidence_agreement = random.uniform(0.3, 1.0)

        # Step 1: Metacognize
        meta = metacognize(
            domain=category,
            complexity=complexity,
            available_context=data_sufficiency,
            similar_past_success=familiarity,
        )

        # Step 2: Calibrate confidence
        conf = calibrate_confidence(
            pattern_familiarity=familiarity,
            evidence_agreement=evidence_agreement,
            stakes=stakes,
            data_sufficiency=data_sufficiency,
        )

        # Step 3: Decompose
        strategy, subproblems = decompose_problem(
            category=category,
            complexity=complexity,
            token_count=tokens,
            has_code=(category == "code"),
            has_math=(category == "math"),
            multi_step=(complexity > 0.5),
        )

        # Step 4: Judge escalation
        current_tier = _tier_for_complexity(complexity * 0.7)  # Start slightly below
        escalation = judge_escalation(
            current_tier=current_tier,
            confidence=conf.score,
            complexity=complexity,
            stakes=stakes,
            time_budget_ms=random.choice([100, 500, 1000, 5000]),
            previous_failures=random.choice([0, 0, 0, 1, 2]),
        )

        # Step 5: Self-reflect on the decision
        reflection = self_reflect(
            output_consistency=random.uniform(0.6, 1.0),
            covers_request=random.uniform(0.5, 1.0),
            evidence_grounded=min(1.0, familiarity + 0.2),
            safety_clear=True,
            abstraction_match=random.uniform(0.5, 1.0),
        )

        # Compose the trace
        situation = (
            f"@task → arrive ["
            f"category: {category}, "
            f"complexity: {complexity:.2f}, "
            f"tokens: {tokens}, "
            f"stakes: {stakes:.1f}]"
        )

        reasoning = "\n".join([
            meta.to_scl(),
            f"@reasoning → decompose [strategy: {strategy.value}, steps: {len(subproblems)}]",
            *[sp.to_scl() for sp in subproblems],
        ])

        decision = escalation.to_scl()
        confidence = conf.to_scl()
        metacognition = meta.to_scl()

        traces.append(ReasoningTrace(
            situation_scl=situation,
            reasoning_scl=reasoning,
            decision_scl=decision,
            confidence_scl=confidence,
            metacognition_scl=metacognition,
            quality=0.95 if familiarity > 0.5 else 0.85,
        ))

    return traces


def save_reasoning_corpus(output_path: str, count: int = 5000) -> dict:
    """Generate and save reasoning traces for CKM training."""
    traces = generate_reasoning_traces(count)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        for trace in traces:
            f.write(trace.to_jsonl() + "\n")

    stats = {
        "total_traces": len(traces),
        "avg_quality": sum(t.quality for t in traces) / len(traces),
        "output_path": str(path),
        "source": "reasoning_core",
    }
    logger.info(f"Saved {len(traces)} reasoning traces to {path}")
    return stats
