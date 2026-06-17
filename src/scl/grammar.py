"""
SCL Formal Grammar — BNF specification for Semantic Compression Language.

This module defines the grammar as a parseable data structure and provides
validation utilities. The grammar itself is the specification; the parser
in parser.py implements it.
"""

# Formal grammar in BNF notation
GRAMMAR_BNF = r"""
    document    ::= (line NEWLINE)* line?
    line        ::= record | comment | BLANK
    record      ::= anchor SPACE relation SPACE scope
    anchor      ::= '@' IDENTIFIER
    relation    ::= ARROW SPACE VERB
    scope       ::= '[' entries ']'
    entries     ::= entry (',' SPACE? entry)*
    entry       ::= key ':' SPACE? value | key
    key         ::= TOKEN
    value       ::= TOKEN | QUOTED_STRING
    comment     ::= ('#' | '//') REST_OF_LINE
    
    ARROW       ::= '→' | '->'
    IDENTIFIER  ::= [a-zA-Z_][a-zA-Z0-9_.*]*
    VERB        ::= [a-zA-Z_][a-zA-Z0-9_]*
    TOKEN       ::= [^\[\],:\n]+
    QUOTED_STRING ::= '"' [^"]* '"'
    SPACE       ::= [ \t]+
    NEWLINE     ::= '\n' | '\r\n'
    BLANK       ::= SPACE? NEWLINE
    REST_OF_LINE ::= [^\n]*
"""

# Grammar as structured data for programmatic use
GRAMMAR_SPEC = {
    "primitives": {
        "@": {
            "name": "Anchor",
            "role": "Entity, subject, or noun",
            "pattern": r"@[a-zA-Z_][a-zA-Z0-9_.*]*",
            "examples": ["@router", "@agent_1", "@cortex", "@task"],
        },
        "→": {
            "name": "Relation",
            "role": "Verb, transition, causality",
            "pattern": r"→\s*[a-zA-Z_][a-zA-Z0-9_]*",
            "alternatives": ["->"],
            "examples": ["→ select", "→ classify", "→ own", "→ depend"],
        },
        "[]": {
            "name": "Scope",
            "role": "Bounded context frame with key-value entries",
            "pattern": r"\[.*?\]",
            "examples": [
                "[model: qwen3:4b, confidence: 0.82]",
                "[tests, hardening, no_features]",
                "[]",
            ],
        },
    },
    "composition": {
        "record": "anchor + relation + scope (one line)",
        "document": "multiple records (one per line)",
        "manifest": "multiple documents (multi-section)",
        "fingerprint": "manifest → SHA-256 → Braille (fixed-width hash)",
    },
    "canonical_form": "@anchor → verb [key: value, key: value]",
    "example": "@router → select [model: qwen3:4b, confidence: 0.82]",
}

# Valid relation verbs used in Cortex's SCL dialect
CORTEX_VERBS = {
    # System state
    "status", "has", "missing", "needs",
    # Agent coordination
    "own", "depend", "deliver", "mission",
    # Routing
    "classify", "select", "route", "escalate", "clamp",
    # Verification
    "answer", "challenge", "verify", "evaluate", "agree", "disagree",
    # Swarm
    "query", "vote", "consensus",
    # Data operations
    "define", "compose", "serialize", "implement", "create",
    # Coordination
    "require", "enforce", "prevent", "config", "note",
    # Braille
    "encode", "decode", "fingerprint", "hash",
    # Bridge
    "bridge", "translate", "emit", "parse",
}


def grammar_text() -> str:
    """Return the BNF grammar as a formatted string."""
    return GRAMMAR_BNF.strip()


def is_valid_verb(verb: str) -> bool:
    """Check if a verb is in the Cortex SCL vocabulary.

    Note: this is advisory, not enforced. SCL is extensible —
    any verb is syntactically valid, but this set is the
    standard vocabulary for Cortex operations.
    """
    return verb in CORTEX_VERBS
