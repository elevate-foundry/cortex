"""
Cortex Audit Braid — Tamper-evident hash chain for consensus race logs.

A braid interleaves multiple independent hash strands into periodic "knots".
Each knot commits to:
  1. All prompt hashes since the last knot
  2. The previous knot hash (Merkle chain)
  3. A timestamp range
  4. The knot sequence number

Braiding gives:
  - Tamper detection: altering any past race breaks all subsequent knots
  - Concurrent audit: multiple races in a window are woven together
  - Compact proof: one knot hash proves N races existed in [t₀, t₁]
  - Verification: anyone with the braid file can validate integrity

Structure:
  strand: individual prompt_hash from a single race
  knot:   H(sorted(strands) ∥ prev_knot ∥ seq ∥ ts_start ∥ ts_end)
  braid:  ordered sequence of knots

SCL representation:
  @knot_0 → braid [strands: 3, prev: genesis, hash: a3f8..., ts: 2026-07-29T03:14:47]
  @knot_1 → braid [strands: 4, prev: a3f8..., hash: b2c1..., ts: 2026-07-29T03:15:02]

SOC 2 CC6.1 / CC7.2: Integrity monitoring and tamper evidence.
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("cortex.audit_braid")

GENESIS_HASH = "0" * 64  # The first knot's "prev" is all zeros


@dataclass
class Strand:
    """A single hash strand — one race's prompt hash."""
    prompt_hash: str
    race_ts: str
    winner_model: str
    n_responses: int


@dataclass
class Knot:
    """
    A braid knot — commits to N strands + previous knot.

    The knot_hash is: SHA-256(sorted_strand_hashes ∥ prev_knot ∥ seq ∥ ts_start ∥ ts_end)
    """
    seq: int
    strands: list[Strand]
    prev_knot: str
    knot_hash: str
    ts_start: str
    ts_end: str
    n_strands: int

    def to_dict(self) -> dict:
        return {
            "seq": self.seq,
            "knot_hash": self.knot_hash,
            "prev_knot": self.prev_knot,
            "ts_start": self.ts_start,
            "ts_end": self.ts_end,
            "n_strands": self.n_strands,
            "strands": [
                {
                    "prompt_hash": s.prompt_hash,
                    "race_ts": s.race_ts,
                    "winner": s.winner_model,
                    "n_responses": s.n_responses,
                }
                for s in self.strands
            ],
        }

    def to_scl(self) -> str:
        """Emit as SCL record."""
        strand_hashes = ",".join(s.prompt_hash[:8] for s in self.strands)
        return (
            f"@knot_{self.seq} → braid ["
            f"strands: {self.n_strands}, "
            f"prev: {self.prev_knot[:8]}, "
            f"hash: {self.knot_hash[:16]}, "
            f"ts: {self.ts_start[:19]}]"
        )


@dataclass
class AuditBraid:
    """
    The full braid — an append-only sequence of knots.

    Usage:
        braid = AuditBraid.load(path)       # Load existing or create new
        braid.add_strand(prompt_hash, ...)   # Add race hashes
        braid.maybe_tie_knot()               # Tie knot if threshold reached
        braid.save()                         # Persist

    Verification:
        braid = AuditBraid.load(path)
        assert braid.verify()                # Check entire chain integrity
    """

    knots: list[Knot] = field(default_factory=list)
    pending_strands: list[Strand] = field(default_factory=list)
    knot_interval: int = 5  # Tie a knot every N strands
    path: Optional[Path] = None

    @classmethod
    def load(cls, path: Path, knot_interval: int = 5) -> "AuditBraid":
        """Load braid from JSONL file, or create new if doesn't exist."""
        braid = cls(knot_interval=knot_interval, path=path)
        if path.exists():
            try:
                for line in path.read_text().strip().split("\n"):
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    if data.get("type") == "knot":
                        knot = Knot(
                            seq=data["seq"],
                            strands=[
                                Strand(
                                    prompt_hash=s["prompt_hash"],
                                    race_ts=s["race_ts"],
                                    winner_model=s["winner"],
                                    n_responses=s["n_responses"],
                                )
                                for s in data["strands"]
                            ],
                            prev_knot=data["prev_knot"],
                            knot_hash=data["knot_hash"],
                            ts_start=data["ts_start"],
                            ts_end=data["ts_end"],
                            n_strands=data["n_strands"],
                        )
                        braid.knots.append(knot)
                    elif data.get("type") == "strand":
                        braid.pending_strands.append(Strand(
                            prompt_hash=data["prompt_hash"],
                            race_ts=data["race_ts"],
                            winner_model=data["winner"],
                            n_responses=data["n_responses"],
                        ))
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Braid file corrupt, starting fresh: {e}")
                braid = cls(knot_interval=knot_interval, path=path)
        return braid

    def add_strand(
        self,
        prompt_hash: str,
        race_ts: str,
        winner_model: str,
        n_responses: int,
    ) -> None:
        """Add a new strand (race hash) to the pending set."""
        strand = Strand(
            prompt_hash=prompt_hash,
            race_ts=race_ts,
            winner_model=winner_model,
            n_responses=n_responses,
        )
        self.pending_strands.append(strand)

        # Persist strand immediately (crash recovery)
        if self.path:
            with open(self.path, "a") as f:
                f.write(json.dumps({
                    "type": "strand",
                    "prompt_hash": strand.prompt_hash,
                    "race_ts": strand.race_ts,
                    "winner": strand.winner_model,
                    "n_responses": strand.n_responses,
                }) + "\n")

    def maybe_tie_knot(self) -> Optional[Knot]:
        """Tie a knot if we have enough pending strands."""
        if len(self.pending_strands) >= self.knot_interval:
            return self.tie_knot()
        return None

    def tie_knot(self) -> Knot:
        """
        Tie a knot NOW — commit all pending strands.

        The knot hash is computed as:
            SHA-256(sorted_prompt_hashes ∥ prev_knot_hash ∥ seq ∥ ts_start ∥ ts_end)

        Sorting ensures the same set of strands always produces the same knot,
        regardless of arrival order.
        """
        if not self.pending_strands:
            raise ValueError("Cannot tie knot with no pending strands")

        seq = len(self.knots)
        prev_knot = self.knots[-1].knot_hash if self.knots else GENESIS_HASH

        # Sort strands by prompt_hash for deterministic ordering
        sorted_strands = sorted(self.pending_strands, key=lambda s: s.prompt_hash)

        ts_start = sorted_strands[0].race_ts
        ts_end = sorted_strands[-1].race_ts

        # Compute knot hash: H(strand_hashes ∥ prev ∥ seq ∥ ts_range)
        hasher = hashlib.sha256()
        for s in sorted_strands:
            hasher.update(s.prompt_hash.encode())
        hasher.update(prev_knot.encode())
        hasher.update(str(seq).encode())
        hasher.update(ts_start.encode())
        hasher.update(ts_end.encode())
        knot_hash = hasher.hexdigest()

        knot = Knot(
            seq=seq,
            strands=sorted_strands,
            prev_knot=prev_knot,
            knot_hash=knot_hash,
            ts_start=ts_start,
            ts_end=ts_end,
            n_strands=len(sorted_strands),
        )
        self.knots.append(knot)
        self.pending_strands = []

        # Persist knot (rewrite file to remove consumed strands)
        self._persist()

        logger.info(
            f"Braid knot #{seq}: {knot.n_strands} strands, "
            f"hash={knot_hash[:16]}..., prev={prev_knot[:8]}..."
        )

        return knot

    def verify(self) -> tuple[bool, Optional[str]]:
        """
        Verify entire braid chain integrity.

        Returns (is_valid, error_message).
        Checks:
          1. Each knot's hash matches recomputed hash from its strands
          2. Each knot's prev_knot matches the previous knot's hash
          3. Sequence numbers are monotonic
        """
        prev_hash = GENESIS_HASH

        for i, knot in enumerate(self.knots):
            # Check sequence
            if knot.seq != i:
                return False, f"Knot {i}: expected seq={i}, got seq={knot.seq}"

            # Check prev pointer
            if knot.prev_knot != prev_hash:
                return False, (
                    f"Knot {i}: prev_knot mismatch. "
                    f"Expected {prev_hash[:16]}, got {knot.prev_knot[:16]}"
                )

            # Recompute hash
            sorted_strands = sorted(knot.strands, key=lambda s: s.prompt_hash)
            hasher = hashlib.sha256()
            for s in sorted_strands:
                hasher.update(s.prompt_hash.encode())
            hasher.update(knot.prev_knot.encode())
            hasher.update(str(knot.seq).encode())
            hasher.update(knot.ts_start.encode())
            hasher.update(knot.ts_end.encode())
            expected_hash = hasher.hexdigest()

            if knot.knot_hash != expected_hash:
                return False, (
                    f"Knot {i}: hash mismatch (TAMPERED). "
                    f"Expected {expected_hash[:16]}, got {knot.knot_hash[:16]}"
                )

            prev_hash = knot.knot_hash

        return True, None

    def summary(self) -> str:
        """Human-readable braid summary."""
        total_strands = sum(k.n_strands for k in self.knots) + len(self.pending_strands)
        latest_hash = self.knots[-1].knot_hash[:16] if self.knots else GENESIS_HASH[:16]
        lines = [
            f"Audit Braid: {len(self.knots)} knots, {total_strands} total strands",
            f"  Latest knot: {latest_hash}...",
            f"  Pending: {len(self.pending_strands)} strands (next knot at {self.knot_interval})",
        ]
        valid, err = self.verify()
        lines.append(f"  Integrity: {'✓ VALID' if valid else f'✗ BROKEN: {err}'}")
        return "\n".join(lines)

    def to_scl(self) -> str:
        """Emit entire braid as SCL document."""
        lines = [f"@braid → status [knots: {len(self.knots)}, pending: {len(self.pending_strands)}]"]
        for knot in self.knots:
            lines.append(knot.to_scl())
        return "\n".join(lines)

    def _persist(self) -> None:
        """Rewrite braid file (knots + pending strands)."""
        if not self.path:
            return
        lines = []
        for knot in self.knots:
            entry = {"type": "knot", **knot.to_dict()}
            lines.append(json.dumps(entry))
        for strand in self.pending_strands:
            lines.append(json.dumps({
                "type": "strand",
                "prompt_hash": strand.prompt_hash,
                "race_ts": strand.race_ts,
                "winner": strand.winner_model,
                "n_responses": strand.n_responses,
            }))
        self.path.write_text("\n".join(lines) + "\n" if lines else "")
