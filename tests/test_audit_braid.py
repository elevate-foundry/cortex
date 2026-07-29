"""
Tests for the audit braid — tamper-evident hash chain for consensus races.

Tests:
  1. Strand accumulation and knot tying
  2. Hash determinism (same strands → same knot hash)
  3. Chain integrity verification
  4. Tamper detection (modify strand, modify knot, reorder)
  5. Persistence (save/load round-trip)
  6. Genesis handling (first knot prev = 0x00...)
  7. Multiple knots chain correctly
  8. SCL emission
"""

import copy
import json
import tempfile
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.audit_braid import AuditBraid, Knot, Strand, GENESIS_HASH


class TestStrandAccumulation:
    def test_add_strand_increments_pending(self):
        braid = AuditBraid(knot_interval=5)
        assert len(braid.pending_strands) == 0
        braid.add_strand("abc123", "2026-01-01T00:00:00", "model-a", 3)
        assert len(braid.pending_strands) == 1

    def test_knot_not_tied_before_threshold(self):
        braid = AuditBraid(knot_interval=5)
        for i in range(4):
            braid.add_strand(f"hash_{i}", f"2026-01-01T00:0{i}:00", "m", 2)
        assert braid.maybe_tie_knot() is None
        assert len(braid.knots) == 0

    def test_knot_tied_at_threshold(self):
        braid = AuditBraid(knot_interval=5)
        for i in range(5):
            braid.add_strand(f"hash_{i:04d}", f"2026-01-01T00:0{i}:00", f"model-{i}", 3)
        knot = braid.maybe_tie_knot()
        assert knot is not None
        assert knot.seq == 0
        assert knot.n_strands == 5
        assert len(braid.pending_strands) == 0

    def test_knot_tied_manually_with_fewer(self):
        braid = AuditBraid(knot_interval=10)
        braid.add_strand("h1", "2026-01-01T00:00:00", "m", 2)
        braid.add_strand("h2", "2026-01-01T00:01:00", "m", 3)
        knot = braid.tie_knot()
        assert knot.n_strands == 2


class TestHashDeterminism:
    def test_same_strands_same_hash(self):
        """Same set of strands should always produce the same knot hash."""
        def make_braid():
            b = AuditBraid(knot_interval=3)
            b.add_strand("aaa", "2026-01-01T00:00:00", "m1", 2)
            b.add_strand("bbb", "2026-01-01T00:01:00", "m2", 3)
            b.add_strand("ccc", "2026-01-01T00:02:00", "m3", 1)
            return b.tie_knot()

        k1 = make_braid()
        k2 = make_braid()
        assert k1.knot_hash == k2.knot_hash

    def test_order_independent(self):
        """Strands added in different order produce same knot (sorted internally)."""
        b1 = AuditBraid(knot_interval=3)
        b1.add_strand("aaa", "2026-01-01T00:00:00", "m1", 2)
        b1.add_strand("bbb", "2026-01-01T00:01:00", "m2", 3)
        b1.add_strand("ccc", "2026-01-01T00:02:00", "m3", 1)
        k1 = b1.tie_knot()

        b2 = AuditBraid(knot_interval=3)
        b2.add_strand("ccc", "2026-01-01T00:02:00", "m3", 1)
        b2.add_strand("aaa", "2026-01-01T00:00:00", "m1", 2)
        b2.add_strand("bbb", "2026-01-01T00:01:00", "m2", 3)
        k2 = b2.tie_knot()

        assert k1.knot_hash == k2.knot_hash

    def test_different_strands_different_hash(self):
        b1 = AuditBraid(knot_interval=2)
        b1.add_strand("aaa", "2026-01-01T00:00:00", "m1", 2)
        b1.add_strand("bbb", "2026-01-01T00:01:00", "m2", 3)
        k1 = b1.tie_knot()

        b2 = AuditBraid(knot_interval=2)
        b2.add_strand("aaa", "2026-01-01T00:00:00", "m1", 2)
        b2.add_strand("xxx", "2026-01-01T00:01:00", "m2", 3)
        k2 = b2.tie_knot()

        assert k1.knot_hash != k2.knot_hash


class TestChainIntegrity:
    def test_genesis_first_knot(self):
        braid = AuditBraid(knot_interval=2)
        braid.add_strand("h1", "t1", "m", 1)
        braid.add_strand("h2", "t2", "m", 1)
        knot = braid.tie_knot()
        assert knot.prev_knot == GENESIS_HASH

    def test_second_knot_links_to_first(self):
        braid = AuditBraid(knot_interval=2)
        braid.add_strand("h1", "t1", "m", 1)
        braid.add_strand("h2", "t2", "m", 1)
        k1 = braid.tie_knot()

        braid.add_strand("h3", "t3", "m", 1)
        braid.add_strand("h4", "t4", "m", 1)
        k2 = braid.tie_knot()

        assert k2.prev_knot == k1.knot_hash

    def test_verify_valid_chain(self):
        braid = AuditBraid(knot_interval=2)
        for i in range(6):
            braid.add_strand(f"h{i}", f"t{i}", "m", 1)
            braid.maybe_tie_knot()
        assert len(braid.knots) == 3
        valid, err = braid.verify()
        assert valid is True
        assert err is None

    def test_verify_empty_braid(self):
        braid = AuditBraid()
        valid, err = braid.verify()
        assert valid is True


class TestTamperDetection:
    def _build_braid(self):
        braid = AuditBraid(knot_interval=3)
        for i in range(9):
            braid.add_strand(f"hash_{i:04d}", f"2026-01-01T00:{i:02d}:00", f"m{i%3}", 2)
            braid.maybe_tie_knot()
        assert len(braid.knots) == 3
        return braid

    def test_tamper_strand_hash(self):
        braid = self._build_braid()
        braid.knots[0].strands[0].prompt_hash = "TAMPERED"
        valid, err = braid.verify()
        assert valid is False
        assert "TAMPERED" in err or "mismatch" in err

    def test_tamper_knot_hash(self):
        braid = self._build_braid()
        braid.knots[1].knot_hash = "deadbeef" * 8
        valid, err = braid.verify()
        assert valid is False
        assert "Knot 1" in err  # Knot 1's recomputed hash won't match

    def test_tamper_prev_pointer(self):
        braid = self._build_braid()
        braid.knots[1].prev_knot = "0" * 64
        valid, err = braid.verify()
        assert valid is False
        assert "prev_knot mismatch" in err

    def test_tamper_sequence_number(self):
        braid = self._build_braid()
        braid.knots[1].seq = 99
        valid, err = braid.verify()
        assert valid is False
        assert "expected seq=1" in err

    def test_delete_knot_detected(self):
        braid = self._build_braid()
        del braid.knots[1]  # Remove middle knot
        valid, err = braid.verify()
        assert valid is False  # seq or prev will mismatch


class TestPersistence:
    def test_save_load_roundtrip(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = Path(f.name)

        try:
            braid = AuditBraid(knot_interval=3, path=path)
            for i in range(6):
                braid.add_strand(f"h{i}", f"t{i}", f"m{i}", i + 1)
                braid.maybe_tie_knot()

            # Load from disk
            loaded = AuditBraid.load(path, knot_interval=3)
            assert len(loaded.knots) == 2
            assert loaded.knots[0].knot_hash == braid.knots[0].knot_hash
            assert loaded.knots[1].knot_hash == braid.knots[1].knot_hash
            assert len(loaded.pending_strands) == 0

            valid, err = loaded.verify()
            assert valid is True
        finally:
            path.unlink(missing_ok=True)

    def test_crash_recovery_pending_strands(self):
        """Pending strands are persisted and recovered after crash."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = Path(f.name)

        try:
            braid = AuditBraid(knot_interval=5, path=path)
            braid.add_strand("h1", "t1", "m1", 2)
            braid.add_strand("h2", "t2", "m2", 3)
            # Simulate crash — don't tie knot

            # Recover
            loaded = AuditBraid.load(path, knot_interval=5)
            assert len(loaded.pending_strands) == 2
            assert loaded.pending_strands[0].prompt_hash == "h1"
        finally:
            path.unlink(missing_ok=True)


class TestSCLEmission:
    def test_scl_format(self):
        braid = AuditBraid(knot_interval=2)
        braid.add_strand("abc123", "2026-01-01T00:00:00", "m1", 3)
        braid.add_strand("def456", "2026-01-01T00:01:00", "m2", 2)
        braid.tie_knot()

        scl = braid.to_scl()
        assert "@braid → status" in scl
        assert "@knot_0 → braid" in scl
        assert "strands: 2" in scl
        assert "prev: 00000000" in scl

    def test_knot_scl(self):
        braid = AuditBraid(knot_interval=2)
        braid.add_strand("aabbccdd", "2026-07-29T00:00:00", "gpt-4o", 5)
        braid.add_strand("11223344", "2026-07-29T00:01:00", "claude", 4)
        knot = braid.tie_knot()

        scl = knot.to_scl()
        assert "knot_0" in scl
        assert "strands: 2" in scl
        assert knot.knot_hash[:16] in scl
