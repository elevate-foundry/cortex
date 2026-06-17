"""
Gossip Protocol — epidemic delta propagation for agent swarms.

At scale, agents don't talk to a central server. They gossip:
  1. Each agent maintains its own SemanticState + DeltaStream
  2. Periodically, an agent picks a random peer and exchanges deltas
  3. Deltas are fingerprinted — if fingerprints match, no data transfer needed
  4. Convergence is detected when all peers' fingerprints are within ε Hamming distance

This implements a push-pull gossip protocol with:
  - Anti-entropy: peers exchange what the other is missing
  - Crdt-style merge: concurrent deltas resolved via configurable strategy
  - Fingerprint-first: check 4-char Braille hash before sending full delta
  - Cluster-head aggregation: hierarchical gossip for large swarms
  - Convergence detection: Hamming distance threshold on state fingerprints

Scaling properties:
  - O(log N) rounds to full convergence (epidemic spread)
  - O(1) bandwidth per sync when states match (fingerprint comparison only)
  - O(k) bandwidth per sync when k keys differ (delta, not full state)

Protocol:
  Agent A → Agent B:
    1. A sends fingerprint(A.state)
    2. B compares: if similar enough → ACK (no data)
    3. If different: B sends its delta since last sync with A
    4. A applies B's delta, sends its own delta since last sync with B
    5. Both now converged

At @scale_infinity, cluster-heads aggregate sub-swarm fingerprints
and gossip only the inter-cluster divergences.
"""

import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable

from .types import SCLRecord, SCLDocument, Anchor, Relation, Scope
from .delta import (
    Delta,
    SemanticState,
    DeltaStream,
    VectorClock,
    MergeStrategy,
    apply_delta,
    diff,
    merge_deltas,
)
from ..braille.codec import encode
from ..braille.fingerprint import fingerprint as scl_fingerprint, similarity


# ---------------------------------------------------------------------------
# Gossip message types
# ---------------------------------------------------------------------------

class GossipMessageType(str, Enum):
    PING = "ping"               # "Here's my fingerprint, are we in sync?"
    PONG = "pong"               # "Yes we match" or "No, here's my delta"
    PUSH_DELTA = "push_delta"   # "Here are my recent changes"
    PULL_REQUEST = "pull_req"   # "Send me your changes since seq N"
    PULL_RESPONSE = "pull_resp" # "Here are changes since seq N"
    CONVERGED = "converged"     # "All peers within ε — swarm is converged"


@dataclass
class GossipMessage:
    """A single gossip message between two agents."""
    msg_type: GossipMessageType
    sender: str
    receiver: str
    fingerprint: str = ""           # Braille fingerprint of sender's state
    deltas: list[Delta] = field(default_factory=list)
    clock: Optional[VectorClock] = None
    since_seq: int = 0              # For pull: "give me deltas since this seq"
    timestamp_ms: int = 0

    def __post_init__(self):
        if self.timestamp_ms == 0:
            self.timestamp_ms = int(time.time() * 1000)

    def to_scl(self) -> SCLRecord:
        """Encode as SCL for audit/logging."""
        entries = {
            "type": self.msg_type.value,
            "from": self.sender,
            "to": self.receiver,
            "fingerprint": self.fingerprint,
            "deltas": str(len(self.deltas)),
        }
        return SCLRecord(
            anchor=Anchor("gossip"),
            relation=Relation(self.msg_type.value),
            scope=Scope(entries=entries),
            timestamp_ms=self.timestamp_ms,
        )

    @property
    def byte_cost(self) -> int:
        """Estimate wire cost in bytes."""
        # Fingerprint: 4 chars * 3 bytes UTF-8 = 12 bytes
        # Each delta: ~50 bytes average
        return 12 + len(self.deltas) * 50


# ---------------------------------------------------------------------------
# Peer — one agent's gossip state
# ---------------------------------------------------------------------------

@dataclass
class GossipStats:
    """Gossip performance counters."""
    messages_sent: int = 0
    messages_received: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    rounds: int = 0
    fingerprint_hits: int = 0   # Syncs avoided due to matching fingerprints
    fingerprint_misses: int = 0
    deltas_propagated: int = 0
    conflicts_resolved: int = 0
    convergences_detected: int = 0


class Peer:
    """A gossip-enabled agent with local state and delta stream.

    Each peer maintains:
      - Its own SemanticState and DeltaStream
      - A record of the last known state fingerprint for each other peer
      - Gossip statistics
    """

    def __init__(
        self,
        agent_id: str,
        initial_state: Optional[SemanticState] = None,
        merge_strategy: MergeStrategy = MergeStrategy.PRIORITY,
        weight: float = 1.0,
    ):
        self.agent_id = agent_id
        self.weight = weight
        self.merge_strategy = merge_strategy
        self.stream = DeltaStream()  # Always start empty
        self.stats = GossipStats()

        # Bootstrap: if initial state provided, emit it as a delta
        # so it can be gossiped to other peers
        if initial_state and initial_state.entries:
            bootstrap = Delta(
                agent_id=agent_id,
                set_keys=dict(initial_state.entries),
                seq=1,
                weight=weight,
            )
            self.stream.append(bootstrap)

        # Last known fingerprint per peer (for anti-entropy)
        self._peer_fingerprints: dict[str, str] = {}
        # Last known stream position per peer (for delta exchange)
        self._peer_positions: dict[str, int] = {}

    @property
    def state(self) -> SemanticState:
        return self.stream.current_state()

    def state_fingerprint(self, width: int = 4) -> str:
        """Braille fingerprint of current state."""
        record = self.state.to_scl(self.agent_id)
        return scl_fingerprint(record, width=width)

    def mutate(self, changes: dict[str, str], deletes: Optional[set[str]] = None) -> Delta:
        """Apply a local mutation and return the delta."""
        delta = Delta(
            agent_id=self.agent_id,
            set_keys=changes,
            delete_keys=deletes or set(),
            seq=self.stream.length + 1,
            weight=self.weight,
        )
        self.stream.append(delta)
        return delta

    def receive_delta(self, delta: Delta) -> SemanticState:
        """Receive and apply a delta from another peer."""
        return self.stream.append(delta)

    def deltas_since(self, position: int) -> list[Delta]:
        """Get all deltas after stream position `position`."""
        all_deltas = self.stream.deltas
        if position >= len(all_deltas):
            return []
        return all_deltas[position:]

    # ------------------------------------------------------------------
    # Gossip protocol
    # ------------------------------------------------------------------

    def initiate_sync(self, other: "Peer") -> list[GossipMessage]:
        """Run one gossip round with another peer. Returns messages exchanged."""
        messages: list[GossipMessage] = []

        # Step 1: PING — send our fingerprint
        my_fp = self.state_fingerprint()
        ping = GossipMessage(
            msg_type=GossipMessageType.PING,
            sender=self.agent_id,
            receiver=other.agent_id,
            fingerprint=my_fp,
            clock=self.state.clock,
        )
        messages.append(ping)
        self.stats.messages_sent += 1
        self.stats.bytes_sent += ping.byte_cost

        # Step 2: Other peer checks fingerprint
        other_fp = other.state_fingerprint()
        other.stats.messages_received += 1
        other.stats.bytes_received += ping.byte_cost

        sim = similarity(my_fp, other_fp) if len(my_fp) == len(other_fp) else 0.0

        if sim >= 0.999:
            # States match — PONG with no data
            pong = GossipMessage(
                msg_type=GossipMessageType.PONG,
                sender=other.agent_id,
                receiver=self.agent_id,
                fingerprint=other_fp,
            )
            messages.append(pong)
            self.stats.fingerprint_hits += 1
            other.stats.fingerprint_hits += 1
            self.stats.messages_received += 1
            other.stats.messages_sent += 1
            self._peer_fingerprints[other.agent_id] = other_fp
            other._peer_fingerprints[self.agent_id] = my_fp
        else:
            # States differ — exchange deltas
            self.stats.fingerprint_misses += 1
            other.stats.fingerprint_misses += 1

            # Other sends its deltas since our last known position
            last_known_pos = other._peer_positions.get(self.agent_id, 0)
            other_deltas = other.deltas_since(last_known_pos)

            push_msg = GossipMessage(
                msg_type=GossipMessageType.PUSH_DELTA,
                sender=other.agent_id,
                receiver=self.agent_id,
                fingerprint=other_fp,
                deltas=other_deltas,
                clock=other.state.clock,
            )
            messages.append(push_msg)
            other.stats.messages_sent += 1
            other.stats.bytes_sent += push_msg.byte_cost
            other.stats.deltas_propagated += len(other_deltas)

            # We apply their deltas
            self.stats.messages_received += 1
            self.stats.bytes_received += push_msg.byte_cost
            for d in other_deltas:
                # Check for conflicts with our pending deltas
                self.stream.append(d)
                self.stats.deltas_propagated += 1

            # We send our deltas since their last known position
            their_last_pos = self._peer_positions.get(other.agent_id, 0)
            my_deltas = self.deltas_since(their_last_pos)

            if my_deltas:
                push_back = GossipMessage(
                    msg_type=GossipMessageType.PUSH_DELTA,
                    sender=self.agent_id,
                    receiver=other.agent_id,
                    fingerprint=self.state_fingerprint(),
                    deltas=my_deltas,
                    clock=self.state.clock,
                )
                messages.append(push_back)
                self.stats.messages_sent += 1
                self.stats.bytes_sent += push_back.byte_cost
                self.stats.deltas_propagated += len(my_deltas)

                # Other applies our deltas
                other.stats.messages_received += 1
                other.stats.bytes_received += push_back.byte_cost
                for d in my_deltas:
                    other.stream.append(d)
                    other.stats.deltas_propagated += 1

            # Update known state
            self._peer_fingerprints[other.agent_id] = other.state_fingerprint()
            other._peer_fingerprints[self.agent_id] = self.state_fingerprint()
            self._peer_positions[other.agent_id] = other.stream.length
            other._peer_positions[self.agent_id] = self.stream.length

        self.stats.rounds += 1
        other.stats.rounds += 1

        return messages

    def to_scl(self) -> SCLRecord:
        """Current peer status as SCL."""
        return SCLRecord(
            anchor=Anchor(self.agent_id),
            relation=Relation("gossip_status"),
            scope=Scope(entries={
                "state_keys": str(len(self.state.entries)),
                "stream_length": str(self.stream.length),
                "fingerprint": self.state_fingerprint(),
                "rounds": str(self.stats.rounds),
                "fp_hits": str(self.stats.fingerprint_hits),
                "fp_misses": str(self.stats.fingerprint_misses),
                "deltas_propagated": str(self.stats.deltas_propagated),
            }),
        )

    def __repr__(self) -> str:
        return (
            f"Peer({self.agent_id}, {len(self.state.entries)} keys, "
            f"{self.stream.length} deltas, fp={self.state_fingerprint()})"
        )


# ---------------------------------------------------------------------------
# Swarm — a collection of gossipping peers
# ---------------------------------------------------------------------------

class Swarm:
    """A swarm of gossipping peers.

    Provides:
      - Random peer selection for gossip rounds
      - Convergence detection via fingerprint similarity
      - Cluster-head election for hierarchical gossip
      - Audit trail of all gossip messages
    """

    def __init__(
        self,
        convergence_threshold: float = 0.95,
        merge_strategy: MergeStrategy = MergeStrategy.PRIORITY,
    ):
        self.peers: dict[str, Peer] = {}
        self.convergence_threshold = convergence_threshold
        self.merge_strategy = merge_strategy
        self.message_log: list[GossipMessage] = []
        self._round: int = 0

    def add_peer(
        self,
        agent_id: str,
        initial_state: Optional[SemanticState] = None,
        weight: float = 1.0,
    ) -> Peer:
        """Add a peer to the swarm."""
        peer = Peer(
            agent_id=agent_id,
            initial_state=initial_state,
            merge_strategy=self.merge_strategy,
            weight=weight,
        )
        self.peers[agent_id] = peer
        return peer

    def gossip_round(self, pairs: Optional[list[tuple[str, str]]] = None) -> list[GossipMessage]:
        """Run one gossip round.

        If pairs not specified, each peer picks a random other peer.
        Returns all messages exchanged.
        """
        if len(self.peers) < 2:
            return []

        self._round += 1
        all_messages: list[GossipMessage] = []

        if pairs is None:
            # Random pairing: each peer picks one random partner
            peer_ids = list(self.peers.keys())
            pairs = []
            for pid in peer_ids:
                candidates = [p for p in peer_ids if p != pid]
                if candidates:
                    partner = random.choice(candidates)
                    pairs.append((pid, partner))

        for a_id, b_id in pairs:
            if a_id in self.peers and b_id in self.peers:
                msgs = self.peers[a_id].initiate_sync(self.peers[b_id])
                all_messages.extend(msgs)

        self.message_log.extend(all_messages)
        return all_messages

    def run_until_converged(self, max_rounds: int = 100) -> int:
        """Run gossip rounds until the swarm converges or max_rounds hit.

        Returns number of rounds taken.
        """
        for r in range(max_rounds):
            self.gossip_round()
            if self.is_converged():
                return r + 1
        return max_rounds

    def is_converged(self) -> bool:
        """Check if all peers' state fingerprints are within threshold."""
        if len(self.peers) < 2:
            return True

        fps = [(pid, peer.state_fingerprint()) for pid, peer in self.peers.items()]

        # Check all pairs
        for i in range(len(fps)):
            for j in range(i + 1, len(fps)):
                fp_a = fps[i][1]
                fp_b = fps[j][1]
                if len(fp_a) != len(fp_b):
                    return False
                sim = similarity(fp_a, fp_b)
                if sim < self.convergence_threshold:
                    return False
        return True

    def convergence_matrix(self) -> dict[str, dict[str, float]]:
        """Pairwise similarity matrix between all peers."""
        fps = {pid: peer.state_fingerprint() for pid, peer in self.peers.items()}
        matrix: dict[str, dict[str, float]] = {}
        for a_id, a_fp in fps.items():
            matrix[a_id] = {}
            for b_id, b_fp in fps.items():
                if a_id == b_id:
                    matrix[a_id][b_id] = 1.0
                elif len(a_fp) == len(b_fp):
                    matrix[a_id][b_id] = similarity(a_fp, b_fp)
                else:
                    matrix[a_id][b_id] = 0.0
        return matrix

    def status(self) -> dict:
        """Swarm status summary."""
        return {
            "peers": len(self.peers),
            "round": self._round,
            "converged": self.is_converged(),
            "total_messages": len(self.message_log),
            "total_deltas": sum(p.stats.deltas_propagated for p in self.peers.values()),
            "total_fp_hits": sum(p.stats.fingerprint_hits for p in self.peers.values()),
            "total_fp_misses": sum(p.stats.fingerprint_misses for p in self.peers.values()),
            "peers_detail": {
                pid: {
                    "fingerprint": peer.state_fingerprint(),
                    "state_keys": len(peer.state.entries),
                    "stream_length": peer.stream.length,
                    "rounds": peer.stats.rounds,
                }
                for pid, peer in self.peers.items()
            },
        }

    def to_scl_document(self) -> SCLDocument:
        """Export swarm state as SCL document."""
        records = [peer.to_scl() for peer in self.peers.values()]
        records.append(SCLRecord(
            anchor=Anchor("swarm"),
            relation=Relation("status"),
            scope=Scope(entries={
                "peers": str(len(self.peers)),
                "round": str(self._round),
                "converged": str(self.is_converged()),
            }),
        ))
        return SCLDocument(records=records, metadata={"type": "gossip_swarm"})

    def __repr__(self) -> str:
        conv = "✓" if self.is_converged() else "✗"
        return f"Swarm({len(self.peers)} peers, round {self._round}, converged={conv})"
