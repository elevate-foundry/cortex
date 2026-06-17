"""
SCL — Semantic Compression Language.

The semantic control protocol for Cortex.
Grammar: @anchor → verb [key: value, key: value]

Modules:
  types       — Anchor, Relation, Scope, SCLRecord, SCLDocument
  parser      — text → SCL AST
  emitter     — SCL AST → text
  grammar     — formal BNF specification
  cortex_bridge — Cortex result types ↔ SCL
"""

from .types import Anchor, Relation, Scope, SCLRecord, SCLDocument

__all__ = ["Anchor", "Relation", "Scope", "SCLRecord", "SCLDocument"]
