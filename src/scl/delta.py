"""
Semantic State Deltas — Git for AI thoughts.

Instead of broadcasting full state, agents emit mutations:

    ΔS_t = S_t ⊖ S_{t-1}

Reconstruction is lossless:

    S_t = S_0 ⊕ Σ(ΔS_i, i=1..t)

This module implements:
  1. State snapshots (SCL scope dictionaries with vector clocks)
  2. Delta computation (diff two states → minimal mutation set)
  3. Delta application (apply a delta to a state → new state)
  4. Merge conflict resolution (CRDT, LWW, policy-weighted)
  5. Delta streams (ordered event log with time-travel replay)

Conflict resolution strategies:
  - LWW (Last-Writer-Wins): highest timestamp wins
  - PRIORITY: highest agent weight wins, timestamp breaks ties
  - UNION: set-valued keys merge (CRDT G-Set)
  - REJECT: conflict raises an error for manual resolution

Grammar extension:
  @agent → mutate [key: new_value, ...]     # delta emission
  @agent → snapshot [key: value, ...]        # full state
  @agent → rollback [to: t, reason: ...]     # time-travel
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .types import Anchor, Relation, Scope, SCLRecord, SCLDocument


# ---------------------------------------------------------------------------
# Vector clock — one entry per agent
# ---------------------------------------------------------------------------

class VectorClock:
    """Tracks causal ordering across agents.

    Each agent increments its own counter on every mutation.
    Comparison determines causal ordering or concurrency.
    """

    def __init__(self, clocks: Optional[dict[str, int]] = None):
        self._clocks: dict[str, int] = dict(clocks) if clocks else {}

    def tick(self, agent_id: str) -> int:
        """Increment this agent's clock. Returns new value."""
        self._clocks[agent_id] = self._clocks.get(agent_id, 0) + 1
        return self._clocks[agent_id]

    def get(self, agent_id: str) -> int:
        return self._clocks.get(agent_id, 0)

    def merge(self, other: "VectorClock") -> "VectorClock":
        """Merge two clocks (element-wise max)."""
        merged = dict(self._clocks)
        for agent_id, count in other._clocks.items():
            merged[agent_id] = max(merged.get(agent_id, 0), count)
        return VectorClock(merged)

    def __le__(self, other: "VectorClock") -> bool:
        """True if self happened-before or is concurrent with other."""
        for agent_id, count in self._clocks.items():
            if count > other._clocks.get(agent_id, 0):
                return False
        return True

    def __lt__(self, other: "VectorClock") -> bool:
        """True if self strictly happened-before other."""
        return self <= other and self._clocks != other._clocks

    def concurrent(self, other: "VectorClock") -> bool:
        """True if neither clock dominates the other."""
        return not (self <= other) and not (other <= self)

    def to_dict(self) -> dict[str, int]:
        return dict(self._clocks)

    @classmethod
    def from_dict(cls, d: dict[str, int]) -> "VectorClock":
        return cls(clocks=d)

    def __repr__(self) -> str:
        return f"VClock({self._clocks})"


# ---------------------------------------------------------------------------
# Conflict resolution
# ---------------------------------------------------------------------------

class MergeStrategy(str, Enum):
    LWW = "lww"             # Last-Writer-Wins (by timestamp)
    PRIORITY = "priority"   # Highest agent weight wins
    UNION = "union"         # Set-union for set-valued keys (CRDT G-Set)
    REJECT = "reject"       # Raise on conflict


@dataclass
class Conflict:
    """A detected merge conflict between two concurrent deltas."""
    key: str
    value_a: str
    value_b: str
    agent_a: str
    agent_b: str
    resolved_value: Optional[str] = None
    strategy_used: Optional[MergeStrategy] = None


# ---------------------------------------------------------------------------
# Delta — the minimal mutation
# ---------------------------------------------------------------------------

@dataclass
class Delta:
    """A semantic state mutation.

    Represents ΔS = S_t ⊖ S_{t-1}: only the keys that changed.

    Fields:
        agent_id: Who emitted this delta
        set_keys: Keys set to new values
        delete_keys: Keys removed from state
        timestamp_ms: When the delta was created
        seq: Sequence number from this agent's vector clock
        weight: Agent priority for conflict resolution
        parent_hash: Content hash of the state this delta was computed against
    """
    agent_id: str
    set_keys: dict[str, str] = field(default_factory=dict)
    delete_keys: set[str] = field(default_factory=set)
    timestamp_ms: int = 0
    seq: int = 0
    weight: float = 1.0
    parent_hash: str = ""

    def __post_init__(self):
        if self.timestamp_ms == 0:
            self.timestamp_ms = int(time.time() * 1000)

    def is_empty(self) -> bool:
        return not self.set_keys and not self.delete_keys

    def to_scl(self) -> SCLRecord:
        """Convert to SCL record: @agent → mutate [changed keys]."""
        entries = dict(self.set_keys)
        for k in self.delete_keys:
            entries[k] = "__deleted__"
        return SCLRecord(
            anchor=Anchor(self.agent_id),
            relation=Relation("mutate"),
            scope=Scope(entries=entries),
            timestamp_ms=self.timestamp_ms,
            weight=self.weight,
        )

    @classmethod
    def from_scl(cls, record: SCLRecord) -> "Delta":
        """Parse a delta from an SCL mutate record."""
        if record.relation.verb != "mutate":
            raise ValueError(f"Expected 'mutate' verb, got '{record.relation.verb}'")
        set_keys = {}
        delete_keys = set()
        for k, v in record.scope.entries.items():
            if v == "__deleted__":
                delete_keys.add(k)
            else:
                set_keys[k] = v
        return cls(
            agent_id=record.anchor.name,
            set_keys=set_keys,
            delete_keys=delete_keys,
            timestamp_ms=record.timestamp_ms,
            weight=record.weight,
        )

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "set_keys": self.set_keys,
            "delete_keys": sorted(self.delete_keys),
            "timestamp_ms": self.timestamp_ms,
            "seq": self.seq,
            "weight": self.weight,
            "parent_hash": self.parent_hash,
        }

    def __repr__(self) -> str:
        sets = ", ".join(f"{k}: {v}" for k, v in self.set_keys.items())
        dels = ", ".join(f"-{k}" for k in self.delete_keys)
        parts = [s for s in (sets, dels) if s]
        return f"Δ({self.agent_id}: {', '.join(parts)})"


# ---------------------------------------------------------------------------
# Semantic State — the full snapshot at time t
# ---------------------------------------------------------------------------

@dataclass
class SemanticState:
    """Complete semantic state of an agent at a point in time.

    S_t = {key: value, ...} with a vector clock for causal ordering.
    """
    entries: dict[str, str] = field(default_factory=dict)
    clock: VectorClock = field(default_factory=VectorClock)
    version: int = 0

    def get(self, key: str, default: str = "") -> str:
        return self.entries.get(key, default)

    def to_scl(self, agent_id: str) -> SCLRecord:
        """Snapshot as SCL: @agent → snapshot [all keys]."""
        return SCLRecord(
            anchor=Anchor(agent_id),
            relation=Relation("snapshot"),
            scope=Scope(entries=dict(self.entries)),
        )

    def content_hash(self) -> str:
        """Deterministic hash of state contents."""
        import hashlib
        import json
        payload = json.dumps(self.entries, sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()[:16]

    def copy(self) -> "SemanticState":
        return SemanticState(
            entries=dict(self.entries),
            clock=VectorClock(self.clock.to_dict()),
            version=self.version,
        )

    def __repr__(self) -> str:
        return f"State(v{self.version}, {len(self.entries)} keys, {self.clock})"


# ---------------------------------------------------------------------------
# Core operations: diff, apply, merge
# ---------------------------------------------------------------------------

def diff(old: SemanticState, new: SemanticState, agent_id: str) -> Delta:
    """Compute ΔS = new ⊖ old.

    Returns a Delta containing only the changed/deleted keys.
    """
    set_keys: dict[str, str] = {}
    delete_keys: set[str] = set()

    # Keys that changed or were added
    for k, v in new.entries.items():
        if k not in old.entries or old.entries[k] != v:
            set_keys[k] = v

    # Keys that were deleted
    for k in old.entries:
        if k not in new.entries:
            delete_keys.add(k)

    return Delta(
        agent_id=agent_id,
        set_keys=set_keys,
        delete_keys=delete_keys,
        parent_hash=old.content_hash(),
    )


def apply_delta(state: SemanticState, delta: Delta) -> SemanticState:
    """Apply a delta to a state: S_t = S_{t-1} ⊕ ΔS_t.

    Returns a new SemanticState (immutable operation).
    """
    new_entries = dict(state.entries)

    for k, v in delta.set_keys.items():
        new_entries[k] = v

    for k in delta.delete_keys:
        new_entries.pop(k, None)

    new_clock = state.clock.merge(VectorClock({delta.agent_id: delta.seq}))
    new_clock.tick(delta.agent_id)

    return SemanticState(
        entries=new_entries,
        clock=new_clock,
        version=state.version + 1,
    )


def merge_deltas(
    delta_a: Delta,
    delta_b: Delta,
    strategy: MergeStrategy = MergeStrategy.LWW,
) -> tuple[Delta, list[Conflict]]:
    """Merge two concurrent deltas into one, resolving conflicts.

    Returns (merged_delta, list_of_conflicts).

    Strategy determines how conflicting keys are resolved:
      - LWW: latest timestamp wins
      - PRIORITY: highest weight wins (timestamp breaks ties)
      - UNION: both values kept as comma-separated set
      - REJECT: raises ValueError on conflict
    """
    merged_set: dict[str, str] = {}
    merged_del: set[str] = set()
    conflicts: list[Conflict] = []

    # All keys touched by either delta
    all_set_keys = set(delta_a.set_keys) | set(delta_b.set_keys)
    all_del_keys = delta_a.delete_keys | delta_b.delete_keys

    for k in all_set_keys:
        in_a = k in delta_a.set_keys
        in_b = k in delta_b.set_keys

        if in_a and in_b:
            va = delta_a.set_keys[k]
            vb = delta_b.set_keys[k]

            if va == vb:
                merged_set[k] = va
                continue

            # Conflict!
            conflict = Conflict(
                key=k, value_a=va, value_b=vb,
                agent_a=delta_a.agent_id, agent_b=delta_b.agent_id,
            )

            if strategy == MergeStrategy.REJECT:
                conflicts.append(conflict)
                continue

            if strategy == MergeStrategy.LWW:
                winner = va if delta_a.timestamp_ms >= delta_b.timestamp_ms else vb
                conflict.resolved_value = winner
                conflict.strategy_used = MergeStrategy.LWW

            elif strategy == MergeStrategy.PRIORITY:
                if delta_a.weight > delta_b.weight:
                    winner = va
                elif delta_b.weight > delta_a.weight:
                    winner = vb
                else:
                    winner = va if delta_a.timestamp_ms >= delta_b.timestamp_ms else vb
                conflict.resolved_value = winner
                conflict.strategy_used = MergeStrategy.PRIORITY

            elif strategy == MergeStrategy.UNION:
                # CRDT G-Set: merge both values
                winner = ",".join(sorted(set(va.split(",") + vb.split(","))))
                conflict.resolved_value = winner
                conflict.strategy_used = MergeStrategy.UNION

            conflicts.append(conflict)
            merged_set[k] = conflict.resolved_value or va

        elif in_a:
            merged_set[k] = delta_a.set_keys[k]
        else:
            merged_set[k] = delta_b.set_keys[k]

    # For deletes: if one sets and other deletes same key, set wins (add-wins CRDT)
    for k in all_del_keys:
        if k not in merged_set:
            merged_del.add(k)

    merged = Delta(
        agent_id=f"{delta_a.agent_id}+{delta_b.agent_id}",
        set_keys=merged_set,
        delete_keys=merged_del,
        timestamp_ms=max(delta_a.timestamp_ms, delta_b.timestamp_ms),
        weight=max(delta_a.weight, delta_b.weight),
    )

    return merged, conflicts


# ---------------------------------------------------------------------------
# Delta Stream — the event-sourced log
# ---------------------------------------------------------------------------

class DeltaStream:
    """Ordered log of deltas — the semantic event source.

    Supports:
      - Append: add a delta to the stream
      - Replay: reconstruct state at any point in time
      - Rollback: rewind to a previous version
      - Compact: squash consecutive deltas into a checkpoint
    """

    def __init__(self, initial_state: Optional[SemanticState] = None):
        self._baseline = initial_state or SemanticState()
        self._deltas: list[Delta] = []
        self._checkpoints: dict[int, SemanticState] = {0: self._baseline.copy()}

    def append(self, delta: Delta) -> SemanticState:
        """Append a delta and return the new state."""
        current = self.current_state()
        new_state = apply_delta(current, delta)
        self._deltas.append(delta)
        return new_state

    def current_state(self) -> SemanticState:
        """Reconstruct current state: S_0 ⊕ Σ(ΔS_i)."""
        return self.state_at(len(self._deltas))

    def state_at(self, t: int) -> SemanticState:
        """Reconstruct state at time t.

        S_t = S_0 ⊕ Σ(ΔS_i, i=1..t)

        Uses checkpoints for efficiency when available.
        """
        if t < 0 or t > len(self._deltas):
            raise ValueError(f"t={t} out of range [0, {len(self._deltas)}]")

        # Find nearest checkpoint at or before t
        cp_t = max(k for k in self._checkpoints if k <= t)
        state = self._checkpoints[cp_t].copy()

        for i in range(cp_t, t):
            state = apply_delta(state, self._deltas[i])

        return state

    def rollback(self, to_t: int) -> SemanticState:
        """Roll back to time t, discarding subsequent deltas."""
        state = self.state_at(to_t)
        self._deltas = self._deltas[:to_t]
        # Purge checkpoints after rollback point
        self._checkpoints = {
            k: v for k, v in self._checkpoints.items() if k <= to_t
        }
        return state

    def checkpoint(self, at_t: Optional[int] = None) -> None:
        """Save a checkpoint at time t (default: current)."""
        t = at_t if at_t is not None else len(self._deltas)
        self._checkpoints[t] = self.state_at(t)

    def compact(self, from_t: int, to_t: int) -> Delta:
        """Squash deltas [from_t, to_t) into a single delta.

        The compacted delta, when applied to state_at(from_t),
        produces state_at(to_t).
        """
        s_from = self.state_at(from_t)
        s_to = self.state_at(to_t)
        return diff(s_from, s_to, agent_id="__compacted__")

    def to_scl_document(self) -> SCLDocument:
        """Export the entire delta stream as an SCL document."""
        records = [d.to_scl() for d in self._deltas]
        return SCLDocument(
            records=records,
            metadata={"type": "delta_stream", "length": str(len(self._deltas))},
        )

    @property
    def length(self) -> int:
        return len(self._deltas)

    @property
    def deltas(self) -> list[Delta]:
        return list(self._deltas)

    def __repr__(self) -> str:
        return (
            f"DeltaStream({len(self._deltas)} deltas, "
            f"{len(self._checkpoints)} checkpoints)"
        )
