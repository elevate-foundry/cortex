"""
Tests for SCL Semantic State Deltas — Git for AI thoughts.

Covers:
  1. Vector clocks — causal ordering, concurrency detection
  2. Diff (⊖) — computing minimal deltas between states
  3. Apply (⊕) — patching state with deltas
  4. Reconstruction — S_t = S_0 ⊕ Σ(ΔS_i)
  5. Merge conflicts — concurrent writes to same key
  6. Conflict resolution — LWW, PRIORITY, UNION, REJECT
  7. DeltaStream — event-sourced log with time-travel and rollback
  8. SCL round-trips — Delta ↔ SCLRecord serialization
  9. Braille integration — fingerprinting deltas
"""

import time
import pytest
from src.scl.delta import (
    VectorClock, Delta, SemanticState, DeltaStream,
    MergeStrategy, Conflict,
    diff, apply_delta, merge_deltas,
)


# ---------------------------------------------------------------------------
# Vector Clock
# ---------------------------------------------------------------------------

class TestVectorClock:

    def test_tick(self):
        vc = VectorClock()
        val = vc.tick("agent_a")
        assert val == 1
        assert vc.get("agent_a") == 1
        vc.tick("agent_a")
        assert vc.get("agent_a") == 2

    def test_merge(self):
        vc_a = VectorClock({"a": 3, "b": 1})
        vc_b = VectorClock({"a": 1, "b": 5, "c": 2})
        merged = vc_a.merge(vc_b)
        assert merged.get("a") == 3
        assert merged.get("b") == 5
        assert merged.get("c") == 2

    def test_dominates(self):
        """a < b means a happened-before b."""
        vc_a = VectorClock({"x": 1})
        vc_b = VectorClock({"x": 2})
        assert vc_a < vc_b
        assert not vc_b < vc_a

    def test_concurrent(self):
        """Neither dominates → concurrent."""
        vc_a = VectorClock({"a": 2, "b": 1})
        vc_b = VectorClock({"a": 1, "b": 2})
        assert vc_a.concurrent(vc_b)
        assert vc_b.concurrent(vc_a)

    def test_not_concurrent_when_equal(self):
        vc_a = VectorClock({"a": 1})
        vc_b = VectorClock({"a": 1})
        assert not vc_a.concurrent(vc_b)

    def test_round_trip_dict(self):
        vc = VectorClock({"a": 3, "b": 7})
        vc2 = VectorClock.from_dict(vc.to_dict())
        assert vc.to_dict() == vc2.to_dict()


# ---------------------------------------------------------------------------
# Diff (⊖)
# ---------------------------------------------------------------------------

class TestDiff:

    def test_no_changes(self):
        s = SemanticState(entries={"status": "idle"})
        d = diff(s, s, "agent_a")
        assert d.is_empty()

    def test_set_key(self):
        old = SemanticState(entries={"status": "idle"})
        new = SemanticState(entries={"status": "processing"})
        d = diff(old, new, "agent_a")
        assert d.set_keys == {"status": "processing"}
        assert not d.delete_keys

    def test_add_key(self):
        old = SemanticState(entries={"status": "idle"})
        new = SemanticState(entries={"status": "idle", "task": "build"})
        d = diff(old, new, "agent_a")
        assert d.set_keys == {"task": "build"}

    def test_delete_key(self):
        old = SemanticState(entries={"status": "idle", "task": "build"})
        new = SemanticState(entries={"status": "idle"})
        d = diff(old, new, "agent_a")
        assert "task" in d.delete_keys
        assert not d.set_keys

    def test_mixed_changes(self):
        old = SemanticState(entries={"a": "1", "b": "2", "c": "3"})
        new = SemanticState(entries={"a": "1", "b": "changed", "d": "new"})
        d = diff(old, new, "agent_x")
        assert d.set_keys == {"b": "changed", "d": "new"}
        assert "c" in d.delete_keys

    def test_parent_hash_set(self):
        old = SemanticState(entries={"x": "y"})
        new = SemanticState(entries={"x": "z"})
        d = diff(old, new, "a")
        assert d.parent_hash == old.content_hash()


# ---------------------------------------------------------------------------
# Apply (⊕)
# ---------------------------------------------------------------------------

class TestApply:

    def test_apply_set(self):
        state = SemanticState(entries={"status": "idle"})
        delta = Delta(agent_id="a", set_keys={"status": "processing"})
        new = apply_delta(state, delta)
        assert new.get("status") == "processing"
        # Original unchanged
        assert state.get("status") == "idle"

    def test_apply_delete(self):
        state = SemanticState(entries={"a": "1", "b": "2"})
        delta = Delta(agent_id="a", delete_keys={"b"})
        new = apply_delta(state, delta)
        assert new.get("a") == "1"
        assert new.get("b") == ""

    def test_apply_add(self):
        state = SemanticState(entries={"a": "1"})
        delta = Delta(agent_id="a", set_keys={"b": "2"})
        new = apply_delta(state, delta)
        assert new.get("a") == "1"
        assert new.get("b") == "2"

    def test_version_increments(self):
        state = SemanticState(entries={"x": "1"}, version=5)
        delta = Delta(agent_id="a", set_keys={"x": "2"})
        new = apply_delta(state, delta)
        assert new.version == 6

    def test_diff_then_apply_reconstructs(self):
        """The fundamental invariant: apply(old, diff(old, new)) == new."""
        old = SemanticState(entries={"a": "1", "b": "2", "c": "3"})
        new = SemanticState(entries={"a": "1", "b": "changed", "d": "new"})
        d = diff(old, new, "agent")
        reconstructed = apply_delta(old, d)
        assert reconstructed.entries == new.entries


# ---------------------------------------------------------------------------
# Reconstruction: S_t = S_0 ⊕ Σ(ΔS_i)
# ---------------------------------------------------------------------------

class TestReconstruction:

    def test_sequential_deltas(self):
        """Apply a chain of deltas and verify final state."""
        s0 = SemanticState(entries={"status": "boot"})
        d1 = Delta(agent_id="a", set_keys={"status": "routing", "tier": "L3"})
        d2 = Delta(agent_id="a", set_keys={"status": "generating", "model": "qwen3:8b"})
        d3 = Delta(agent_id="a", set_keys={"status": "done", "confidence": "0.82"}, delete_keys={"tier"})

        s1 = apply_delta(s0, d1)
        s2 = apply_delta(s1, d2)
        s3 = apply_delta(s2, d3)

        assert s3.get("status") == "done"
        assert s3.get("model") == "qwen3:8b"
        assert s3.get("confidence") == "0.82"
        assert s3.get("tier") == ""  # deleted

    def test_multi_agent_convergence(self):
        """Two agents applying non-conflicting deltas converge."""
        base = SemanticState(entries={"shared": "v0"})

        d_a = Delta(agent_id="a", set_keys={"task_a": "done"})
        d_b = Delta(agent_id="b", set_keys={"task_b": "done"})

        # Apply in order: a then b
        s_ab = apply_delta(apply_delta(base, d_a), d_b)
        # Apply in order: b then a
        s_ba = apply_delta(apply_delta(base, d_b), d_a)

        assert s_ab.entries == s_ba.entries  # commutative for non-conflicting


# ---------------------------------------------------------------------------
# Merge Conflicts
# ---------------------------------------------------------------------------

class TestMergeConflicts:

    def test_no_conflict(self):
        d_a = Delta(agent_id="a", set_keys={"x": "1"})
        d_b = Delta(agent_id="b", set_keys={"y": "2"})
        merged, conflicts = merge_deltas(d_a, d_b)
        assert not conflicts
        assert merged.set_keys == {"x": "1", "y": "2"}

    def test_same_value_no_conflict(self):
        """Both agents set same key to same value → no conflict."""
        d_a = Delta(agent_id="a", set_keys={"mode": "safe"})
        d_b = Delta(agent_id="b", set_keys={"mode": "safe"})
        merged, conflicts = merge_deltas(d_a, d_b)
        assert not conflicts
        assert merged.set_keys["mode"] == "safe"

    def test_conflict_detected(self):
        d_a = Delta(agent_id="a", set_keys={"mode": "safe"}, timestamp_ms=100)
        d_b = Delta(agent_id="b", set_keys={"mode": "aggressive"}, timestamp_ms=200)
        merged, conflicts = merge_deltas(d_a, d_b, strategy=MergeStrategy.REJECT)
        assert len(conflicts) == 1
        assert conflicts[0].key == "mode"
        assert conflicts[0].value_a == "safe"
        assert conflicts[0].value_b == "aggressive"

    def test_lww_resolution(self):
        """Last-Writer-Wins: latest timestamp wins."""
        d_a = Delta(agent_id="a", set_keys={"mode": "safe"}, timestamp_ms=100)
        d_b = Delta(agent_id="b", set_keys={"mode": "aggressive"}, timestamp_ms=200)
        merged, conflicts = merge_deltas(d_a, d_b, strategy=MergeStrategy.LWW)
        assert len(conflicts) == 1
        assert conflicts[0].resolved_value == "aggressive"
        assert merged.set_keys["mode"] == "aggressive"

    def test_priority_resolution(self):
        """Highest weight wins."""
        d_a = Delta(agent_id="conductor", set_keys={"mode": "safe"}, weight=10.0, timestamp_ms=100)
        d_b = Delta(agent_id="worker", set_keys={"mode": "aggressive"}, weight=1.0, timestamp_ms=200)
        merged, conflicts = merge_deltas(d_a, d_b, strategy=MergeStrategy.PRIORITY)
        assert len(conflicts) == 1
        assert conflicts[0].resolved_value == "safe"
        assert conflicts[0].strategy_used == MergeStrategy.PRIORITY

    def test_union_resolution(self):
        """CRDT G-Set: merge both values."""
        d_a = Delta(agent_id="a", set_keys={"tags": "fast"})
        d_b = Delta(agent_id="b", set_keys={"tags": "reliable"})
        merged, conflicts = merge_deltas(d_a, d_b, strategy=MergeStrategy.UNION)
        assert len(conflicts) == 1
        # Union should contain both values
        resolved = conflicts[0].resolved_value
        assert "fast" in resolved
        assert "reliable" in resolved

    def test_delete_vs_set(self):
        """If one agent sets and another deletes, set wins (add-wins CRDT)."""
        d_a = Delta(agent_id="a", set_keys={"key": "value"})
        d_b = Delta(agent_id="b", delete_keys={"key"})
        merged, conflicts = merge_deltas(d_a, d_b)
        assert "key" in merged.set_keys  # set wins

    def test_multiple_conflicts(self):
        d_a = Delta(agent_id="a", set_keys={"x": "1", "y": "2", "z": "same"}, timestamp_ms=100)
        d_b = Delta(agent_id="b", set_keys={"x": "A", "y": "B", "z": "same"}, timestamp_ms=200)
        merged, conflicts = merge_deltas(d_a, d_b, strategy=MergeStrategy.LWW)
        # x and y conflict, z doesn't
        assert len(conflicts) == 2
        assert merged.set_keys["z"] == "same"


# ---------------------------------------------------------------------------
# DeltaStream — event-sourced log with time-travel
# ---------------------------------------------------------------------------

class TestDeltaStream:

    def test_append_and_current(self):
        stream = DeltaStream()
        stream.append(Delta(agent_id="a", set_keys={"status": "boot"}))
        state = stream.current_state()
        assert state.get("status") == "boot"

    def test_sequential_appends(self):
        stream = DeltaStream()
        stream.append(Delta(agent_id="a", set_keys={"step": "1"}))
        stream.append(Delta(agent_id="a", set_keys={"step": "2"}))
        stream.append(Delta(agent_id="a", set_keys={"step": "3"}))
        assert stream.current_state().get("step") == "3"
        assert stream.length == 3

    def test_state_at_time_travel(self):
        """Reconstruct state at any historical point."""
        stream = DeltaStream()
        stream.append(Delta(agent_id="a", set_keys={"x": "1"}))
        stream.append(Delta(agent_id="a", set_keys={"x": "2", "y": "A"}))
        stream.append(Delta(agent_id="a", set_keys={"x": "3"}))

        s0 = stream.state_at(0)
        s1 = stream.state_at(1)
        s2 = stream.state_at(2)
        s3 = stream.state_at(3)

        assert s0.entries == {}
        assert s1.get("x") == "1"
        assert s2.get("x") == "2"
        assert s2.get("y") == "A"
        assert s3.get("x") == "3"
        assert s3.get("y") == "A"  # persisted from t=2

    def test_rollback(self):
        """Roll back to a previous state, discarding subsequent deltas."""
        stream = DeltaStream()
        stream.append(Delta(agent_id="a", set_keys={"status": "good"}))
        stream.append(Delta(agent_id="rogue", set_keys={"status": "corrupted"}))
        stream.append(Delta(agent_id="rogue", set_keys={"data": "bad"}))

        # Something went wrong at t=2 — roll back to t=1
        state = stream.rollback(1)
        assert state.get("status") == "good"
        assert state.get("data") == ""
        assert stream.length == 1

    def test_checkpoint_and_replay(self):
        stream = DeltaStream()
        for i in range(10):
            stream.append(Delta(agent_id="a", set_keys={"i": str(i)}))

        stream.checkpoint(at_t=5)

        # State at t=7 should use the t=5 checkpoint internally
        s7 = stream.state_at(7)
        assert s7.get("i") == "6"  # 0-indexed delta, value is i from the loop

    def test_compact(self):
        """Squash multiple deltas into a single equivalent delta."""
        stream = DeltaStream()
        stream.append(Delta(agent_id="a", set_keys={"a": "1", "b": "2"}))
        stream.append(Delta(agent_id="a", set_keys={"b": "3", "c": "4"}))
        stream.append(Delta(agent_id="a", delete_keys={"a"}))

        compacted = stream.compact(0, 3)

        # Apply compacted delta to baseline should equal state_at(3)
        expected = stream.state_at(3)
        result = apply_delta(stream.state_at(0), compacted)
        assert result.entries == expected.entries

    def test_out_of_range(self):
        stream = DeltaStream()
        with pytest.raises(ValueError):
            stream.state_at(1)
        with pytest.raises(ValueError):
            stream.state_at(-1)

    def test_to_scl_document(self):
        stream = DeltaStream()
        stream.append(Delta(agent_id="a", set_keys={"x": "1"}))
        stream.append(Delta(agent_id="b", set_keys={"y": "2"}))
        doc = stream.to_scl_document()
        assert len(doc) == 2
        assert doc.metadata["type"] == "delta_stream"


# ---------------------------------------------------------------------------
# SCL Round-trips
# ---------------------------------------------------------------------------

class TestSCLIntegration:

    def test_delta_to_scl_and_back(self):
        d = Delta(agent_id="agent_04", set_keys={"status": "idle", "queue": "job_42"})
        record = d.to_scl()
        assert record.anchor.name == "agent_04"
        assert record.relation.verb == "mutate"
        assert record.scope.get("status") == "idle"
        assert record.scope.get("queue") == "job_42"

        # Round-trip
        d2 = Delta.from_scl(record)
        assert d2.agent_id == "agent_04"
        assert d2.set_keys == {"status": "idle", "queue": "job_42"}

    def test_delete_round_trip(self):
        d = Delta(agent_id="a", delete_keys={"old_key"})
        record = d.to_scl()
        d2 = Delta.from_scl(record)
        assert "old_key" in d2.delete_keys

    def test_state_to_scl(self):
        s = SemanticState(entries={"status": "ready", "tier": "L3"})
        record = s.to_scl("agent_1")
        assert record.anchor.name == "agent_1"
        assert record.relation.verb == "snapshot"
        assert record.scope.get("status") == "ready"


# ---------------------------------------------------------------------------
# Braille Fingerprinting of Deltas
# ---------------------------------------------------------------------------

class TestDeltaFingerprinting:

    def test_delta_scl_fingerprint(self):
        """Deltas can be fingerprinted through their SCL representation."""
        from src.braille.fingerprint import fingerprint

        d = Delta(agent_id="a", set_keys={"status": "done"})
        record = d.to_scl()
        fp = fingerprint(record)
        assert len(fp) == 4  # 4 Braille chars

    def test_different_deltas_different_fingerprints(self):
        from src.braille.fingerprint import fingerprint

        d1 = Delta(agent_id="a", set_keys={"status": "done"})
        d2 = Delta(agent_id="a", set_keys={"status": "failed"})
        fp1 = fingerprint(d1.to_scl())
        fp2 = fingerprint(d2.to_scl())
        assert fp1 != fp2

    def test_stream_document_fingerprint(self):
        """Entire delta streams can be fingerprinted as documents."""
        from src.braille.fingerprint import fingerprint_document

        stream = DeltaStream()
        stream.append(Delta(agent_id="a", set_keys={"x": "1"}))
        stream.append(Delta(agent_id="b", set_keys={"y": "2"}))
        doc = stream.to_scl_document()
        fp = fingerprint_document(doc)
        assert len(fp) == 8


# ---------------------------------------------------------------------------
# Scale simulation: the microscopic token footprint
# ---------------------------------------------------------------------------

class TestScaleEfficiency:

    def test_delta_is_smaller_than_full_state(self):
        """A delta that changes 1 key in a 20-key state is tiny."""
        big_state = SemanticState(entries={f"key_{i}": f"value_{i}" for i in range(20)})
        new_state = big_state.copy()
        new_state.entries["key_5"] = "changed"

        d = diff(big_state, new_state, "agent")

        full_scl = big_state.to_scl("agent").to_text()
        delta_scl = d.to_scl().to_text()

        # Delta should be dramatically smaller
        assert len(delta_scl) < len(full_scl)
        # In fact, less than half
        assert len(delta_scl) < len(full_scl) // 2

    def test_empty_delta_for_unchanged_state(self):
        """No change → empty delta → zero network cost."""
        s = SemanticState(entries={"a": "1", "b": "2"})
        d = diff(s, s, "agent")
        assert d.is_empty()
