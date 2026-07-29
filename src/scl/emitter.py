"""
SCL Emitter — SCL AST → text.

Converts SCLRecord and SCLDocument objects back to canonical text form.
Supports compact mode (one line per record) and pretty mode (aligned columns).
"""

from .types import SCLRecord, SCLDocument


def emit_record(record: SCLRecord, compact: bool = True) -> str:
    """Emit a single SCL record as text.

    Args:
        record: The SCLRecord to emit.
        compact: If True (default), single-line output.
                 If False, no difference for single records (pretty affects documents).

    Returns:
        Canonical SCL text: @anchor → verb [key: value, key: value]
    """
    return record.to_text()


def emit_document(
    doc: SCLDocument,
    compact: bool = True,
    align: bool = False,
    include_metadata: bool = False,
) -> str:
    """Emit a full SCL document as text.

    Args:
        doc: The SCLDocument to emit.
        compact: If True (default), one record per line, minimal whitespace.
        align: If True, align columns (anchor, arrow, verb, scope).
        include_metadata: If True, prepend metadata as comments.

    Returns:
        Multi-line SCL text.
    """
    lines: list[str] = []

    # Metadata as comments
    if include_metadata and doc.metadata:
        for k, v in doc.metadata.items():
            lines.append(f"# {k}: {v}")
        lines.append("")

    if not doc.records:
        return "\n".join(lines)

    if compact or not align:
        for record in doc.records:
            lines.append(emit_record(record))
    else:
        # Pretty mode: align columns
        lines.extend(_emit_aligned(doc.records))

    return "\n".join(lines)


def emit_summary(doc: SCLDocument) -> str:
    """Emit a one-line summary of a document.

    Format: {N} records: @anchor1, @anchor2, ...
    """
    if not doc.records:
        return "0 records"

    anchors = list(dict.fromkeys(r.anchor.name for r in doc.records))
    anchor_str = ", ".join(f"@{a}" for a in anchors[:5])
    if len(anchors) > 5:
        anchor_str += f", ... (+{len(anchors) - 5})"

    return f"{len(doc.records)} records: {anchor_str}"


def emit_table(doc: SCLDocument) -> str:
    """Emit records as a Markdown-style table.

    | Anchor | Verb | Scope |
    |--------|------|-------|
    | @router | select | model: qwen3:4b |
    """
    if not doc.records:
        return ""

    rows: list[tuple[str, str, str]] = []
    for r in doc.records:
        scope_str = ", ".join(
            k if v == "" else f"{k}: {v}"
            for k, v in r.scope.entries.items()
        )
        rows.append((r.anchor.to_text(), r.relation.verb, scope_str))

    # Column widths
    w_anchor = max(len("Anchor"), max(len(r[0]) for r in rows))
    w_verb = max(len("Verb"), max(len(r[1]) for r in rows))
    w_scope = max(len("Scope"), max(len(r[2]) for r in rows))

    lines: list[str] = []
    lines.append(
        f"| {'Anchor':<{w_anchor}} | {'Verb':<{w_verb}} | {'Scope':<{w_scope}} |"
    )
    lines.append(f"|{'-' * (w_anchor + 2)}|{'-' * (w_verb + 2)}|{'-' * (w_scope + 2)}|")
    for anchor, verb, scope in rows:
        lines.append(f"| {anchor:<{w_anchor}} | {verb:<{w_verb}} | {scope:<{w_scope}} |")

    return "\n".join(lines)


# --- Internal helpers ---

def _emit_aligned(records: list[SCLRecord]) -> list[str]:
    """Emit records with aligned columns."""
    if not records:
        return []

    # Compute column widths
    anchors = [r.anchor.to_text() for r in records]
    verbs = [r.relation.verb for r in records]

    w_anchor = max(len(a) for a in anchors)
    w_verb = max(len(v) for v in verbs)

    lines: list[str] = []
    for record, anchor_text, verb_text in zip(records, anchors, verbs):
        scope_text = record.scope.to_text()
        lines.append(
            f"{anchor_text:<{w_anchor}} → {verb_text:<{w_verb}} {scope_text}"
        )

    return lines
