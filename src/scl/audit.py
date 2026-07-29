"""
SCL Audit helpers — build SCL documents + fingerprints for Cortex requests.

Every route/generation emits an SCL document.  This module produces the
canonical text and Braille fingerprint so the daemon can persist them.
"""

from .cortex_bridge import cortex_response_to_scl, route_decision_to_scl
from .types import SCLDocument, SCLRecord, Anchor, Relation, Scope
from .emitter import emit_document
from ..braille.fingerprint import fingerprint_document
from ..cortex import CortexResponse
from ..router import RouteDecision


from ..tools import ToolCall, ToolResult


def build_scl_from_response(cortex_resp: CortexResponse) -> tuple[str, str]:
    """Full non-streaming pipeline result → (scl_text, fingerprint)."""
    doc = cortex_response_to_scl(cortex_resp)
    scl_text = emit_document(doc, compact=True)
    fp = fingerprint_document(doc, width=4)
    return scl_text, fp


def build_scl_from_autonomous_response(
    cortex_resp: CortexResponse,
    tool_rounds: list[tuple[list[ToolCall], list[ToolResult]]],
) -> tuple[str, str]:
    """Build full SCL document including route, generation, and all tool rounds."""
    import json
    doc = cortex_response_to_scl(cortex_resp)
    for tc_list, tr_list in tool_rounds:
        for tc in tc_list:
            doc.records.append(SCLRecord(
                anchor=Anchor("tool"),
                relation=Relation("call"),
                scope=Scope({
                    "name": tc.name,
                    "args": json.dumps(tc.arguments),
                }),
            ))
        for tr in tr_list:
            doc.records.append(SCLRecord(
                anchor=Anchor("tool"),
                relation=Relation("result"),
                scope=Scope({
                    "name": tr.name,
                    "success": str(tr.success).lower(),
                    "latency_ms": f"{tr.latency_ms:.0f}",
                }),
            ))
    scl_text = emit_document(doc, compact=True)
    fp = fingerprint_document(doc, width=4)
    return scl_text, fp


def build_scl_from_streaming_route(
    route: RouteDecision,
    model_tag: str,
    tier_name: str,
) -> tuple[str, str]:
    """Streaming route selection → (scl_text, fingerprint)."""
    records = route_decision_to_scl(route)
    records.append(
        SCLRecord(
            anchor=Anchor("core"),
            relation=Relation("generate"),
            scope=Scope({"tier": tier_name, "model": model_tag}),
        )
    )
    records.append(
        SCLRecord(
            anchor=Anchor("pipeline"),
            relation=Relation("trace"),
            scope=Scope({"path": "route→stream"}),
        )
    )
    doc = SCLDocument(records=records)
    scl_text = emit_document(doc, compact=True)
    fp = fingerprint_document(doc, width=4)
    return scl_text, fp
