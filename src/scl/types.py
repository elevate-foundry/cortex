"""
SCL core types — the state atoms for multi-agent coordination at any scale.

These dataclasses are the wire protocol. Everything else consumes them:
  - Parser reads text → these types
  - Emitter writes these types → text
  - Braille fingerprinter hashes these types → fixed-width Braille
  - Cortex bridge converts runtime types ↔ these types
  - At scale, gossip propagates these types between cluster-heads

Grammar:
  record     := anchor relation scope
  anchor     := '@' IDENTIFIER
  relation   := '→' VERB
  scope      := '[' entries ']'
  entries    := entry (',' entry)*
  entry      := KEY ':' VALUE
  document   := record ('\\n' record)*

Canonical form: @router → select [model: qwen3:4b, confidence: 0.82]
"""

import hashlib
import json
import struct
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Anchor:
    """@ — entity, subject, or noun. The thing being described.

    Examples: @router, @memory, @task, @agent_1, @swarm
    """

    name: str

    def to_text(self) -> str:
        return f"@{self.name}"

    @classmethod
    def from_text(cls, text: str) -> "Anchor":
        text = text.strip()
        if not text.startswith("@"):
            raise ValueError(f"Anchor must start with '@', got: {text!r}")
        name = text[1:].strip()
        if not name:
            raise ValueError("Anchor name cannot be empty")
        return cls(name=name)

    def to_bytes(self) -> bytes:
        return self.name.encode("utf-8")

    @classmethod
    def from_bytes(cls, data: bytes) -> "Anchor":
        return cls(name=data.decode("utf-8"))

    def to_dict(self) -> dict:
        return {"name": self.name}

    @classmethod
    def from_dict(cls, d: dict) -> "Anchor":
        return cls(name=d["name"])


@dataclass(frozen=True)
class Relation:
    """→ — verb, transition, causality. What the anchor does.

    Examples: → select, → escalate, → persist, → classify, → own
    """

    verb: str

    def to_text(self) -> str:
        return f"→ {self.verb}"

    @classmethod
    def from_text(cls, text: str) -> "Relation":
        text = text.strip()
        if text.startswith("→"):
            text = text[1:].strip()
        elif text.startswith("->"):
            text = text[2:].strip()
        if not text:
            raise ValueError("Relation verb cannot be empty")
        return cls(verb=text)

    def to_bytes(self) -> bytes:
        return self.verb.encode("utf-8")

    @classmethod
    def from_bytes(cls, data: bytes) -> "Relation":
        return cls(verb=data.decode("utf-8"))

    def to_dict(self) -> dict:
        return {"verb": self.verb}

    @classmethod
    def from_dict(cls, d: dict) -> "Relation":
        return cls(verb=d["verb"])


@dataclass(frozen=True)
class Scope:
    """[ ] — bounded context frame. Key-value attributes.

    Examples: [model: qwen3:4b, confidence: 0.82]
              [tier: L3, category: code]
              [tests, hardening, no_features]
    """

    entries: dict[str, str] = field(default_factory=dict)

    def to_text(self) -> str:
        if not self.entries:
            return "[]"
        parts = []
        for k, v in self.entries.items():
            if v == "":
                parts.append(k)
            else:
                parts.append(f"{k}: {v}")
        return "[" + ", ".join(parts) + "]"

    @classmethod
    def from_text(cls, text: str) -> "Scope":
        text = text.strip()
        if not text.startswith("[") or not text.endswith("]"):
            raise ValueError(f"Scope must be wrapped in [ ], got: {text!r}")
        inner = text[1:-1].strip()
        if not inner:
            return cls(entries={})

        entries: dict[str, str] = {}
        # Split on commas, but respect nested brackets
        parts = _split_scope_entries(inner)
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if ":" in part:
                key, _, value = part.partition(":")
                entries[key.strip()] = value.strip()
            else:
                # Bare value — use as key with empty value
                entries[part.strip()] = ""
        return cls(entries=entries)

    def to_bytes(self) -> bytes:
        return json.dumps(self.entries, separators=(",", ":")).encode("utf-8")

    @classmethod
    def from_bytes(cls, data: bytes) -> "Scope":
        return cls(entries=json.loads(data.decode("utf-8")))

    def to_dict(self) -> dict:
        return {"entries": dict(self.entries)}

    @classmethod
    def from_dict(cls, d: dict) -> "Scope":
        return cls(entries=d.get("entries", {}))

    def get(self, key: str, default: str = "") -> str:
        return self.entries.get(key, default)

    def __contains__(self, key: str) -> bool:
        return key in self.entries

    def __len__(self) -> int:
        return len(self.entries)


def _split_scope_entries(text: str) -> list[str]:
    """Split scope entries on commas, respecting nested brackets."""
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    for char in text:
        if char == "[":
            depth += 1
            current.append(char)
        elif char == "]":
            depth -= 1
            current.append(char)
        elif char == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(char)
    if current:
        parts.append("".join(current))
    return parts


@dataclass
class SCLRecord:
    """One SCL statement: @anchor → relation [scope]

    The fundamental unit of semantic state. Everything in the system —
    routing decisions, challenge results, agent heartbeats, audit entries —
    serializes to one or more SCLRecords.

    At scale, SCLRecords are:
      - Fingerprinted into fixed-width Braille hashes for gossip
      - Compared via Hamming distance for convergence checks
      - Weighted for consensus algorithms
    """

    anchor: Anchor
    relation: Relation
    scope: Scope
    timestamp_ms: int = 0
    weight: float = 1.0
    parent_id: Optional[str] = None

    def __post_init__(self):
        if self.timestamp_ms == 0:
            self.timestamp_ms = int(time.time() * 1000)

    def to_text(self) -> str:
        """Canonical text form: @router → select [model: qwen3:4b]"""
        return f"{self.anchor.to_text()} {self.relation.to_text()} {self.scope.to_text()}"

    @classmethod
    def from_text(cls, text: str) -> "SCLRecord":
        """Parse from canonical text form.

        Accepts:
          @router → select [model: qwen3:4b, confidence: 0.82]
          @router -> select [model: qwen3:4b]
        """
        text = text.strip()

        # Extract anchor
        if not text.startswith("@"):
            raise ValueError(f"SCLRecord must start with '@', got: {text!r}")

        # Find the relation marker (→ or ->)
        arrow_pos = text.find("→")
        arrow_len = 1
        if arrow_pos == -1:
            arrow_pos = text.find("->")
            arrow_len = 2
        if arrow_pos == -1:
            raise ValueError(f"SCLRecord must contain '→' or '->', got: {text!r}")

        anchor_text = text[:arrow_pos].strip()
        rest = text[arrow_pos + arrow_len :].strip()

        # Find the scope
        bracket_pos = rest.find("[")
        if bracket_pos == -1:
            raise ValueError(f"SCLRecord must contain '[scope]', got: {text!r}")

        verb_text = rest[:bracket_pos].strip()
        scope_text = rest[bracket_pos:]

        return cls(
            anchor=Anchor.from_text(anchor_text),
            relation=Relation(verb=verb_text),
            scope=Scope.from_text(scope_text),
        )

    def to_bytes(self) -> bytes:
        """Compact binary for Braille encoding and fingerprinting.

        Format: [anchor_len:2][anchor][verb_len:2][verb][scope_json]
        Fixed header allows efficient prefix parsing.
        """
        anchor_bytes = self.anchor.to_bytes()
        verb_bytes = self.relation.to_bytes()
        scope_bytes = self.scope.to_bytes()

        return (
            struct.pack(">H", len(anchor_bytes))
            + anchor_bytes
            + struct.pack(">H", len(verb_bytes))
            + verb_bytes
            + scope_bytes
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "SCLRecord":
        """Deserialize from compact binary."""
        offset = 0

        anchor_len = struct.unpack(">H", data[offset : offset + 2])[0]
        offset += 2
        anchor = Anchor.from_bytes(data[offset : offset + anchor_len])
        offset += anchor_len

        verb_len = struct.unpack(">H", data[offset : offset + 2])[0]
        offset += 2
        relation = Relation.from_bytes(data[offset : offset + verb_len])
        offset += verb_len

        scope = Scope.from_bytes(data[offset:])

        return cls(anchor=anchor, relation=relation, scope=scope)

    def to_dict(self) -> dict:
        """JSON-serializable dict."""
        d: dict = {
            "anchor": self.anchor.to_dict(),
            "relation": self.relation.to_dict(),
            "scope": self.scope.to_dict(),
            "timestamp_ms": self.timestamp_ms,
        }
        if self.weight != 1.0:
            d["weight"] = self.weight
        if self.parent_id is not None:
            d["parent_id"] = self.parent_id
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SCLRecord":
        return cls(
            anchor=Anchor.from_dict(d["anchor"]),
            relation=Relation.from_dict(d["relation"]),
            scope=Scope.from_dict(d["scope"]),
            timestamp_ms=d.get("timestamp_ms", 0),
            weight=d.get("weight", 1.0),
            parent_id=d.get("parent_id"),
        )

    def content_hash(self) -> str:
        """SHA-256 hash of the record's semantic content (excludes timestamp).

        Used by Braille fingerprinting for LSH-style similarity.
        """
        h = hashlib.sha256(self.to_bytes())
        return h.hexdigest()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SCLRecord):
            return NotImplemented
        return (
            self.anchor == other.anchor
            and self.relation == other.relation
            and self.scope == other.scope
        )

    def __hash__(self) -> int:
        return hash((self.anchor, self.relation, tuple(self.scope.entries.items())))

    def __repr__(self) -> str:
        return f"SCLRecord({self.to_text()})"


@dataclass
class SCLDocument:
    """An ordered collection of SCLRecords with metadata.

    Represents a complete SCL document — one per routing decision,
    one per challenge result, one per agent heartbeat.

    At scale:
      - Documents are fingerprinted as a unit for dedup
      - Documents are the unit of gossip propagation
      - Documents compose into manifests
    """

    records: list[SCLRecord] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)

    def to_text(self) -> str:
        """Multi-line text, one record per line."""
        return "\n".join(r.to_text() for r in self.records)

    @classmethod
    def from_text(cls, text: str) -> "SCLDocument":
        """Parse multiple records from multi-line text."""
        records: list[SCLRecord] = []
        for line in text.strip().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("//"):
                continue
            try:
                records.append(SCLRecord.from_text(line))
            except ValueError:
                continue  # lenient: skip malformed lines
        return cls(records=records)

    def to_bytes(self) -> bytes:
        """Concatenated binary of all records, length-prefixed."""
        parts: list[bytes] = []
        for r in self.records:
            rb = r.to_bytes()
            parts.append(struct.pack(">I", len(rb)) + rb)
        return b"".join(parts)

    def to_dict(self) -> dict:
        return {
            "records": [r.to_dict() for r in self.records],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SCLDocument":
        return cls(
            records=[SCLRecord.from_dict(r) for r in d.get("records", [])],
            metadata=d.get("metadata", {}),
        )

    def content_hash(self) -> str:
        """SHA-256 hash of the full document for dedup/fingerprinting."""
        h = hashlib.sha256(self.to_bytes())
        return h.hexdigest()

    def append(self, record: SCLRecord) -> None:
        self.records.append(record)

    def filter_by_anchor(self, name: str) -> "SCLDocument":
        """Return a sub-document containing only records with the given anchor."""
        return SCLDocument(
            records=[r for r in self.records if r.anchor.name == name],
            metadata=self.metadata,
        )

    def filter_by_verb(self, verb: str) -> "SCLDocument":
        """Return a sub-document containing only records with the given verb."""
        return SCLDocument(
            records=[r for r in self.records if r.relation.verb == verb],
            metadata=self.metadata,
        )

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self):
        return iter(self.records)

    def __getitem__(self, idx):
        return self.records[idx]

    def __repr__(self) -> str:
        return f"SCLDocument({len(self.records)} records)"
