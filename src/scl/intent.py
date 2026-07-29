"""
Semantic Intent & Reservation — optimistic concurrency for agent swarms.

Instead of 2PC (blocking, O(N) coordinator) or pessimistic locks (deadlock-prone),
agents declare intent before mutating shared anchors. This is optimistic, non-blocking,
and conflict-aware:

    @agent_04 → intent   [target: env.db.pool_size, action: mutate, ttl: 500]
    @agent_04 → mutate   [env.db.pool_size: 25]
    @agent_04 → release  [target: env.db.pool_size]

Design:
  - Intents are advisory, not blocking — they don't prevent others from writing
  - Intents create a "priority window" — if a conflict arises during the window,
    the agent that declared intent first gets priority (causal ordering via vclock)
  - Intents expire via TTL — no dangling locks, no deadlocks
  - Multiple agents can declare intent on the same anchor — this signals a
    potential conflict BEFORE it happens, enabling preemptive resolution
  - The gossip layer propagates intents alongside deltas

Conflict resolution hierarchy:
  1. If only one agent declared intent → that agent wins
  2. If multiple intents → authority weight decides
  3. If equal authority → vector clock ordering (first intent wins)
  4. If truly concurrent + equal → escalate to conductor

This is NOT 2PC. Comparison:
  - 2PC: "Everyone agree before I commit" (synchronous, blocking)
  - Semantic Intent: "I'm going to do X" (async, advisory, resolved after)
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .types import Anchor, Relation, Scope, SCLRecord
from .delta import VectorClock, Delta, SemanticState


# ---------------------------------------------------------------------------
# Intent lifecycle
# ---------------------------------------------------------------------------

class IntentStatus(str, Enum):
    DECLARED = "declared"       # Agent has announced intent
    ACTIVE = "active"           # Agent is executing the mutation
    COMPLETED = "completed"     # Mutation applied, intent fulfilled
    EXPIRED = "expired"         # TTL elapsed without completion
    CONFLICTED = "conflicted"   # Another agent's intent overlaps
    WITHDRAWN = "withdrawn"     # Agent voluntarily withdrew


class ConflictOutcome(str, Enum):
    FIRST_WINS = "first_wins"          # Earliest intent takes priority
    AUTHORITY_WINS = "authority_wins"  # Highest weight wins
    MERGE = "merge"                    # Both can proceed (non-conflicting values)
    ESCALATE = "escalate"              # Requires conductor decision


# ---------------------------------------------------------------------------
# Intent declaration
# ---------------------------------------------------------------------------

@dataclass
class Intent:
    """A semantic reservation — an agent's declared intention to mutate an anchor.

    Intents are lightweight (one per target key), non-blocking, and expire.
    They flow through gossip like any other SCL record.
    """

    agent_id: str
    target_anchor: str          # Which anchor will be mutated
    target_key: str             # Which key within the anchor
    intended_value: str = ""    # What the agent plans to set (optional, for preemption)
    action: str = "mutate"     # mutate | delete | create
    ttl_ms: int = 5000         # Time-to-live in milliseconds
    priority: float = 1.0      # Agent's authority weight
    status: IntentStatus = IntentStatus.DECLARED
    vclock: VectorClock = field(default_factory=VectorClock)
    declared_at_ms: int = 0
    completed_at_ms: int = 0

    def __post_init__(self):
        if self.declared_at_ms == 0:
            self.declared_at_ms = int(time.time() * 1000)

    @property
    def target(self) -> str:
        """Fully qualified target: anchor.key"""
        return f"{self.target_anchor}.{self.target_key}"

    @property
    def is_alive(self) -> bool:
        """True if intent hasn't expired or completed."""
        if self.status in (IntentStatus.COMPLETED, IntentStatus.EXPIRED, IntentStatus.WITHDRAWN):
            return False
        now = int(time.time() * 1000)
        return (now - self.declared_at_ms) < self.ttl_ms

    @property
    def remaining_ms(self) -> int:
        """Milliseconds until expiry."""
        elapsed = int(time.time() * 1000) - self.declared_at_ms
        return max(0, self.ttl_ms - elapsed)

    def expire_if_stale(self) -> bool:
        """Check and update status if TTL elapsed. Returns True if expired."""
        if self.status == IntentStatus.DECLARED and not self.is_alive:
            self.status = IntentStatus.EXPIRED
            return True
        return False

    def complete(self) -> None:
        """Mark intent as fulfilled."""
        self.status = IntentStatus.COMPLETED
        self.completed_at_ms = int(time.time() * 1000)

    def withdraw(self) -> None:
        """Agent voluntarily withdraws intent."""
        self.status = IntentStatus.WITHDRAWN

    def to_scl(self) -> SCLRecord:
        """Serialize as SCL for gossip propagation."""
        entries = {
            "target": self.target,
            "action": self.action,
            "ttl": str(self.ttl_ms),
            "status": self.status.value,
        }
        if self.intended_value:
            entries["value"] = self.intended_value
        return SCLRecord(
            anchor=Anchor(self.agent_id),
            relation=Relation("intent"),
            scope=Scope(entries=entries),
            timestamp_ms=self.declared_at_ms,
            weight=self.priority,
        )

    def to_delta(self) -> Delta:
        """Convert intent declaration to a delta for stream propagation."""
        return Delta(
            agent_id=self.agent_id,
            set_keys={
                f"__intent__.{self.target}": f"{self.action}:{self.intended_value}",
                f"__intent__.{self.target}.__status__": self.status.value,
                f"__intent__.{self.target}.__ttl__": str(self.ttl_ms),
            },
            weight=self.priority,
            timestamp_ms=self.declared_at_ms,
        )

    def __repr__(self) -> str:
        return (
            f"Intent({self.agent_id} → {self.target} "
            f"[{self.action}={self.intended_value!r}, "
            f"status={self.status.value}, "
            f"ttl={self.remaining_ms}ms])"
        )


# ---------------------------------------------------------------------------
# Intent Registry — tracks all active intents across the swarm
# ---------------------------------------------------------------------------

@dataclass
class ConflictReport:
    """Report of a detected intent conflict."""
    target: str
    intents: list[Intent]
    outcome: ConflictOutcome = ConflictOutcome.ESCALATE
    winner: Optional[Intent] = None
    reason: str = ""

    def to_scl(self) -> SCLRecord:
        agents = "+".join(i.agent_id for i in self.intents)
        return SCLRecord(
            anchor=Anchor("conflict"),
            relation=Relation("intent_collision"),
            scope=Scope(entries={
                "target": self.target,
                "agents": agents,
                "outcome": self.outcome.value,
                "winner": self.winner.agent_id if self.winner else "none",
                "reason": self.reason,
            }),
        )


class IntentRegistry:
    """Tracks all active intents across the swarm.

    Responsibilities:
      - Register new intents
      - Detect overlapping intents (potential conflicts) BEFORE they happen
      - Expire stale intents
      - Resolve intent priority when conflicts are detected
      - Provide "is this mutation safe?" queries

    This runs on each peer locally — intents propagate via gossip,
    so each peer has a (eventually consistent) view of all intents.
    """

    def __init__(self, conductor_id: str = "conductor"):
        self._intents: dict[str, list[Intent]] = {}  # target → [intents]
        self._conductor_id = conductor_id
        self._history: list[Intent] = []
        self._conflicts: list[ConflictReport] = []

    def declare(self, intent: Intent) -> Optional[ConflictReport]:
        """Register a new intent. Returns ConflictReport if overlap detected.

        The conflict report is advisory — it doesn't block the intent.
        The agent can choose to proceed, withdraw, or wait.
        """
        target = intent.target

        # Expire stale intents on this target
        self._expire_target(target)

        if target not in self._intents:
            self._intents[target] = []

        self._intents[target].append(intent)
        self._history.append(intent)

        # Check for overlapping alive intents from different agents
        alive = [i for i in self._intents[target]
                 if i.is_alive and i.agent_id != intent.agent_id]

        if alive:
            report = self._resolve_conflict(target, [intent] + alive)
            self._conflicts.append(report)
            return report

        return None

    def release(self, agent_id: str, target: str) -> None:
        """Release an intent (mark as completed)."""
        if target in self._intents:
            for intent in self._intents[target]:
                if intent.agent_id == agent_id and intent.is_alive:
                    intent.complete()

    def withdraw(self, agent_id: str, target: str) -> None:
        """Withdraw an intent without completing it."""
        if target in self._intents:
            for intent in self._intents[target]:
                if intent.agent_id == agent_id and intent.status == IntentStatus.DECLARED:
                    intent.withdraw()

    def is_safe(self, agent_id: str, target: str) -> bool:
        """Can this agent safely mutate this target?

        Safe if:
          - No other alive intents on this target, OR
          - This agent's intent has the highest priority among alive intents
        """
        self._expire_target(target)

        if target not in self._intents:
            return True

        alive_others = [
            i for i in self._intents[target]
            if i.is_alive and i.agent_id != agent_id
        ]

        if not alive_others:
            return True

        # Check if this agent has the highest priority
        my_intents = [
            i for i in self._intents[target]
            if i.agent_id == agent_id and i.is_alive
        ]

        if not my_intents:
            return False  # No intent declared — not safe

        my_priority = max(i.priority for i in my_intents)
        max_other_priority = max(i.priority for i in alive_others)

        return my_priority > max_other_priority

    def contention(self, target: str) -> int:
        """How many agents have alive intents on this target?"""
        self._expire_target(target)
        if target not in self._intents:
            return 0
        return len([i for i in self._intents[target] if i.is_alive])

    def active_intents(self, agent_id: Optional[str] = None) -> list[Intent]:
        """Get all alive intents, optionally filtered by agent."""
        self._expire_all()
        result = []
        for intents in self._intents.values():
            for intent in intents:
                if intent.is_alive:
                    if agent_id is None or intent.agent_id == agent_id:
                        result.append(intent)
        return result

    def conflicts(self) -> list[ConflictReport]:
        """Get all detected conflict reports."""
        return list(self._conflicts)

    def _resolve_conflict(self, target: str, intents: list[Intent]) -> ConflictReport:
        """Resolve a multi-intent conflict on a single target.

        Resolution hierarchy:
          1. If values are compatible (same intended_value) → MERGE
          2. First declared (by vclock/timestamp) → FIRST_WINS
          3. Highest authority → AUTHORITY_WINS
          4. Equal authority + concurrent → ESCALATE to conductor
        """
        # Check if values are compatible (same intended mutation)
        values = set(i.intended_value for i in intents if i.intended_value)
        if len(values) <= 1:
            return ConflictReport(
                target=target,
                intents=intents,
                outcome=ConflictOutcome.MERGE,
                winner=None,
                reason="compatible values — all agents intend the same mutation",
            )

        # Sort by priority descending, then timestamp ascending (earlier wins ties)
        ranked = sorted(intents, key=lambda i: (-i.priority, i.declared_at_ms))

        if ranked[0].priority > ranked[1].priority:
            winner = ranked[0]
            for i in intents:
                if i != winner:
                    i.status = IntentStatus.CONFLICTED
            return ConflictReport(
                target=target,
                intents=intents,
                outcome=ConflictOutcome.AUTHORITY_WINS,
                winner=winner,
                reason=f"authority {winner.priority} > {ranked[1].priority}",
            )

        # Equal priority — earliest timestamp wins
        if ranked[0].declared_at_ms < ranked[1].declared_at_ms:
            winner = ranked[0]
            for i in intents:
                if i != winner:
                    i.status = IntentStatus.CONFLICTED
            return ConflictReport(
                target=target,
                intents=intents,
                outcome=ConflictOutcome.FIRST_WINS,
                winner=winner,
                reason=f"declared first at t={winner.declared_at_ms}",
            )

        # Truly concurrent + equal authority → escalate
        return ConflictReport(
            target=target,
            intents=intents,
            outcome=ConflictOutcome.ESCALATE,
            winner=None,
            reason="concurrent intents with equal authority — requires conductor",
        )

    def _expire_target(self, target: str) -> None:
        """Expire stale intents on a specific target."""
        if target in self._intents:
            for intent in self._intents[target]:
                intent.expire_if_stale()

    def _expire_all(self) -> None:
        """Expire all stale intents."""
        for target in self._intents:
            self._expire_target(target)

    def to_scl_document(self) -> list[SCLRecord]:
        """Export all active intents as SCL records."""
        self._expire_all()
        return [i.to_scl() for i in self.active_intents()]

    @property
    def stats(self) -> dict:
        """Registry statistics."""
        self._expire_all()
        alive = sum(1 for intents in self._intents.values()
                    for i in intents if i.is_alive)
        return {
            "total_declared": len(self._history),
            "currently_alive": alive,
            "targets_with_contention": sum(
                1 for t in self._intents if self.contention(t) > 1
            ),
            "conflicts_detected": len(self._conflicts),
            "conflicts_escalated": sum(
                1 for c in self._conflicts if c.outcome == ConflictOutcome.ESCALATE
            ),
        }

    def __repr__(self) -> str:
        s = self.stats
        return (
            f"IntentRegistry({s['currently_alive']} alive, "
            f"{s['conflicts_detected']} conflicts)"
        )
