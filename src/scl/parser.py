"""
SCL Parser — text → SCL AST.

Parses SCL records and documents from text input.
Supports both strict mode (raises on errors) and lenient mode (skips bad lines).

Grammar:
  record     := anchor relation scope
  anchor     := '@' IDENTIFIER
  relation   := '→' VERB  |  '->' VERB
  scope      := '[' entries ']'
  entries    := entry (',' entry)*
  entry      := KEY ':' VALUE  |  KEY (bare value)
  document   := record ('\\n' record)*

Examples:
  @router → select [model: qwen3:4b, confidence: 0.82]
  @agent_1 → own [tests, hardening, no_features]
  @cortex → status [modules: 20, runtime: working]
"""

from dataclasses import dataclass
from typing import Optional

from .types import Anchor, Relation, Scope, SCLRecord, SCLDocument


@dataclass
class ParseError:
    """A parse error with location info."""
    line: int
    column: int
    message: str
    source: str

    def __str__(self) -> str:
        return f"line {self.line}, col {self.column}: {self.message}"


@dataclass
class ParseResult:
    """Result of a parse attempt — either a record or an error."""
    record: Optional[SCLRecord] = None
    error: Optional[ParseError] = None

    @property
    def ok(self) -> bool:
        return self.record is not None


def parse_record(text: str, line_number: int = 1) -> ParseResult:
    """Parse a single SCL record from text.

    Args:
        text: A single line of SCL text.
        line_number: Line number for error reporting.

    Returns:
        ParseResult with either the parsed record or an error.

    Examples:
        >>> parse_record('@router → select [model: qwen3:4b]').record.to_text()
        '@router → select [model: qwen3:4b]'
    """
    text = text.strip()

    if not text:
        return ParseResult(error=ParseError(
            line=line_number, column=0,
            message="Empty input",
            source=text,
        ))

    # Skip comments
    if text.startswith("#") or text.startswith("//"):
        return ParseResult(error=ParseError(
            line=line_number, column=0,
            message="Comment line",
            source=text,
        ))

    # Must start with @
    if not text.startswith("@"):
        return ParseResult(error=ParseError(
            line=line_number, column=0,
            message=f"Expected '@', got {text[0]!r}",
            source=text,
        ))

    # Find arrow (→ or ->)
    arrow_pos, arrow_len = _find_arrow(text)
    if arrow_pos == -1:
        return ParseResult(error=ParseError(
            line=line_number, column=len(text),
            message="Expected '→' or '->'",
            source=text,
        ))

    # Extract anchor
    anchor_text = text[:arrow_pos].strip()
    try:
        anchor = Anchor.from_text(anchor_text)
    except ValueError as e:
        return ParseResult(error=ParseError(
            line=line_number, column=0,
            message=str(e),
            source=text,
        ))

    # Everything after arrow
    rest = text[arrow_pos + arrow_len:].strip()

    # Find scope brackets
    bracket_pos = rest.find("[")
    if bracket_pos == -1:
        return ParseResult(error=ParseError(
            line=line_number, column=arrow_pos + arrow_len,
            message="Expected '[' for scope",
            source=text,
        ))

    # Find matching closing bracket
    close_bracket = _find_matching_bracket(rest, bracket_pos)
    if close_bracket == -1:
        return ParseResult(error=ParseError(
            line=line_number, column=len(text),
            message="Unmatched '['",
            source=text,
        ))

    # Extract verb and scope
    verb_text = rest[:bracket_pos].strip()
    if not verb_text:
        return ParseResult(error=ParseError(
            line=line_number, column=arrow_pos + arrow_len,
            message="Expected verb after '→'",
            source=text,
        ))

    scope_text = rest[bracket_pos:close_bracket + 1]

    try:
        relation = Relation(verb=verb_text)
        scope = Scope.from_text(scope_text)
    except ValueError as e:
        return ParseResult(error=ParseError(
            line=line_number, column=bracket_pos,
            message=str(e),
            source=text,
        ))

    return ParseResult(record=SCLRecord(
        anchor=anchor,
        relation=relation,
        scope=scope,
    ))


def parse_document(text: str, strict: bool = False) -> SCLDocument:
    """Parse a multi-line SCL document.

    Args:
        text: Multi-line SCL text.
        strict: If True, raise ValueError on first parse error.
                If False (default), skip malformed lines.

    Returns:
        SCLDocument containing all successfully parsed records.

    Raises:
        ValueError: In strict mode, if any line fails to parse.
    """
    records: list[SCLRecord] = []
    errors: list[ParseError] = []

    for line_num, line in enumerate(text.splitlines(), start=1):
        line = line.strip()

        # Skip empty lines and comments
        if not line or line.startswith("#") or line.startswith("//"):
            continue

        # Skip lines that don't look like SCL records
        if not line.startswith("@"):
            continue

        result = parse_record(line, line_number=line_num)

        if result.ok:
            records.append(result.record)
        elif strict:
            raise ValueError(f"Parse error at {result.error}")
        else:
            errors.append(result.error)

    doc = SCLDocument(records=records)
    if errors:
        doc.metadata["parse_errors"] = str(len(errors))
    return doc


def parse_record_strict(text: str) -> SCLRecord:
    """Parse a single record, raising ValueError on failure.

    Convenience wrapper for strict parsing.
    """
    result = parse_record(text)
    if not result.ok:
        raise ValueError(f"Parse error: {result.error}")
    return result.record


# --- Internal helpers ---

def _find_arrow(text: str) -> tuple[int, int]:
    """Find the position and length of the arrow (→ or ->).

    Returns (position, length) or (-1, 0) if not found.
    """
    # Check for Unicode arrow first (preferred)
    pos = text.find("→")
    if pos != -1:
        return pos, len("→")

    # Fall back to ASCII arrow
    pos = text.find("->")
    if pos != -1:
        return pos, 2

    return -1, 0


def _find_matching_bracket(text: str, open_pos: int) -> int:
    """Find the matching ']' for a '[' at open_pos.

    Returns the index of ']' or -1 if unmatched.
    """
    depth = 0
    for i in range(open_pos, len(text)):
        if text[i] == "[":
            depth += 1
        elif text[i] == "]":
            depth -= 1
            if depth == 0:
                return i
    return -1
