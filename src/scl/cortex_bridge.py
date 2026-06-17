"""
SCL Cortex Bridge — converts Cortex runtime types ↔ SCL records.

This is the translation layer between Cortex's internal data structures
and the SCL semantic protocol. Every routing decision, challenge result,
swarm outcome, and full response can be expressed as SCL.

This enables:
  - Human-readable audit trails in SCL format
  - Machine-parseable routing logs
  - Braille fingerprinting of any Cortex operation
  - SCL-based coordination at scale
"""

from .types import Anchor, Relation, Scope, SCLRecord, SCLDocument
from ..router import RouteDecision
from ..challenger import ChallengeResult
from ..swarm import SwarmResult
from ..cortex import CortexResponse


def route_decision_to_scl(decision: RouteDecision) -> list[SCLRecord]:
    """Convert a RouteDecision into SCL records.

    Output:
      @task → classify [category: code, complexity: 0.45]
      @router → select [tier: L3, confidence: 0.82]
    """
    records = [
        SCLRecord(
            anchor=Anchor("task"),
            relation=Relation("classify"),
            scope=Scope({"category": decision.category.value}),
        ),
        SCLRecord(
            anchor=Anchor("router"),
            relation=Relation("select"),
            scope=Scope({
                "tier": decision.tier.name,
                "confidence": f"{decision.confidence:.2f}",
                "reason": decision.reason,
            }),
        ),
    ]

    if decision.escalation_hint is not None:
        records.append(SCLRecord(
            anchor=Anchor("router"),
            relation=Relation("escalate"),
            scope=Scope({"hint": decision.escalation_hint.name}),
        ))

    return records


def challenge_result_to_scl(result: ChallengeResult) -> list[SCLRecord]:
    """Convert a ChallengeResult into SCL records.

    Output:
      @core → answer [model: qwen3:8b]
      @challenger → answer [model: granite3.3:8b, family: granite]
      @agreement → evaluate [level: weak_agree, confidence: 0.65]
    """
    records = [
        SCLRecord(
            anchor=Anchor("core"),
            relation=Relation("answer"),
            scope=Scope({
                "model": result.original_model,
                "ms": f"{result.original_ms:.0f}",
            }),
        ),
        SCLRecord(
            anchor=Anchor("challenger"),
            relation=Relation("answer"),
            scope=Scope({
                "model": result.challenge_model,
                "family": result.challenge_family,
                "ms": f"{result.challenge_ms:.0f}",
            }),
        ),
        SCLRecord(
            anchor=Anchor("agreement"),
            relation=Relation("evaluate"),
            scope=Scope({
                "level": result.agreement.value,
                "confidence": f"{result.confidence:.2f}",
                "escalate": str(result.should_escalate).lower(),
            }),
        ),
    ]

    return records


def swarm_result_to_scl(result: SwarmResult) -> list[SCLRecord]:
    """Convert a SwarmResult into SCL records.

    Output:
      @swarm → query [size: small, models: 4, families: 3]
      @cluster_0 → agree [models: 3, families: 2, weight: 7.5]
      @cluster_1 → disagree [models: 1, families: 1, weight: 2.0]
      @consensus → select [cluster: 0, confidence: 0.78]
    """
    records = [
        SCLRecord(
            anchor=Anchor("swarm"),
            relation=Relation("query"),
            scope=Scope({
                "models": str(result.num_models),
                "families": str(result.num_families),
                "method": result.method.value,
                "ms": f"{result.total_ms:.0f}",
            }),
        ),
    ]

    # Emit each cluster
    for i, cluster in enumerate(result.clusters):
        families_in_cluster = len(set(v.family for v in cluster))
        total_weight = sum(v.weight for v in cluster)
        verb = "agree" if i == 0 else "disagree"

        records.append(SCLRecord(
            anchor=Anchor(f"cluster_{i}"),
            relation=Relation(verb),
            scope=Scope({
                "models": str(len(cluster)),
                "families": str(families_in_cluster),
                "weight": f"{total_weight:.1f}",
            }),
        ))

    # Consensus record
    records.append(SCLRecord(
        anchor=Anchor("consensus"),
        relation=Relation("select"),
        scope=Scope({
            "confidence": f"{result.confidence:.2f}",
            "agreement_ratio": f"{result.agreement_ratio:.2f}",
            "agreeing_families": str(result.num_agreeing_families),
        }),
    ))

    return records


def cortex_response_to_scl(response: CortexResponse) -> SCLDocument:
    """Full pipeline result → SCL document.

    Combines route decision, challenge result (if any), and swarm result
    (if any) into a single SCL document representing the complete
    processing pipeline.
    """
    records: list[SCLRecord] = []

    # Route decision
    records.extend(route_decision_to_scl(response.route_decision))

    # Core generation
    records.append(SCLRecord(
        anchor=Anchor("core"),
        relation=Relation("generate"),
        scope=Scope({
            "tier": response.tier_used.name,
            "model": response.model_used,
            "confidence": f"{response.confidence:.2f}",
        }),
    ))

    # Challenge (if it happened)
    if response.challenge_result is not None:
        records.extend(challenge_result_to_scl(response.challenge_result))

    # Swarm (if it happened)
    if response.swarm_result is not None:
        records.extend(swarm_result_to_scl(response.swarm_result))

    # Escalation path
    if response.escalation_path:
        records.append(SCLRecord(
            anchor=Anchor("pipeline"),
            relation=Relation("trace"),
            scope=Scope({
                "path": " → ".join(response.escalation_path),
                "total_ms": f"{response.total_ms:.0f}",
            }),
        ))

    return SCLDocument(
        records=records,
        metadata={"type": "cortex_response"},
    )
