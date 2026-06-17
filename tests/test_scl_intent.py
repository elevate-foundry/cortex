"""
Tests for Semantic Intent & Reservation system.

Covers:
  1. Intent lifecycle: declare → active → complete/expire/withdraw
  2. TTL expiry: intents die automatically
  3. Conflict detection: overlapping intents on same target
  4. Resolution hierarchy: authority > first-declared > escalate
  5. Safety queries: "can I mutate this?"
  6. SCL serialization: intents as gossipable records
  7. Registry operations: contention, active intents, stats
"""

import time
import pytest
from src.scl.intent import (
    Intent,
    IntentStatus,
    IntentRegistry,
    ConflictOutcome,
    ConflictReport,
)


# ---------------------------------------------------------------------------
# Intent Lifecycle
# ---------------------------------------------------------------------------

class TestIntentLifecycle:

    def test_declare(self):
        i = Intent(
            agent_id="agent_04",
            target_anchor="env.db",
            target_key="pool_size",
            intended_value="25",
            action="mutate",
        )
        assert i.status == IntentStatus.DECLARED
        assert i.target == "env.db.pool_size"
        assert i.is_alive

    def test_complete(self):
        i = Intent(agent_id="a", target_anchor="x", target_key="y")
        i.complete()
        assert i.status == IntentStatus.COMPLETED
        assert not i.is_alive

    def test_withdraw(self):
        i = Intent(agent_id="a", target_anchor="x", target_key="y")
        i.withdraw()
        assert i.status == IntentStatus.WITHDRAWN
        assert not i.is_alive

    def test_ttl_expiry(self):
        i = Intent(
            agent_id="a", target_anchor="x", target_key="y",
            ttl_ms=1,  # 1ms TTL
            declared_at_ms=int(time.time() * 1000) - 100,  # declared 100ms ago
        )
        assert i.expire_if_stale()
        assert i.status == IntentStatus.EXPIRED

    def test_alive_with_remaining_ttl(self):
        i = Intent(
            agent_id="a", target_anchor="x", target_key="y",
            ttl_ms=60000,  # 60 seconds
        )
        assert i.is_alive
        assert i.remaining_ms > 0

    def test_target_construction(self):
        i = Intent(agent_id="a", target_anchor="env.db", target_key="latency")
        assert i.target == "env.db.latency"


# ---------------------------------------------------------------------------
# Intent Registry — Declaration & Detection
# ---------------------------------------------------------------------------

class TestIntentRegistry:

    def test_declare_no_conflict(self):
        reg = IntentRegistry()
        i = Intent(agent_id="a", target_anchor="x", target_key="y")
        report = reg.declare(i)
        assert report is None  # No conflict

    def test_same_agent_no_conflict(self):
        """Same agent declaring twice on same target doesn't conflict with itself."""
        reg = IntentRegistry()
        i1 = Intent(agent_id="a", target_anchor="x", target_key="y", ttl_ms=60000)
        i2 = Intent(agent_id="a", target_anchor="x", target_key="y", ttl_ms=60000)
        reg.declare(i1)
        report = reg.declare(i2)
        assert report is None

    def test_different_targets_no_conflict(self):
        reg = IntentRegistry()
        i1 = Intent(agent_id="a", target_anchor="x", target_key="y", ttl_ms=60000)
        i2 = Intent(agent_id="b", target_anchor="x", target_key="z", ttl_ms=60000)
        reg.declare(i1)
        report = reg.declare(i2)
        assert report is None

    def test_conflict_detected(self):
        """Two agents declare intent on same target → conflict."""
        reg = IntentRegistry()
        i1 = Intent(
            agent_id="a", target_anchor="env", target_key="mode",
            intended_value="safe", ttl_ms=60000, priority=1.0,
            declared_at_ms=int(time.time() * 1000),
        )
        i2 = Intent(
            agent_id="b", target_anchor="env", target_key="mode",
            intended_value="aggressive", ttl_ms=60000, priority=1.0,
            declared_at_ms=int(time.time() * 1000) + 10,
        )
        reg.declare(i1)
        report = reg.declare(i2)
        assert report is not None
        assert len(report.intents) == 2

    def test_compatible_values_merge(self):
        """Same intended value → no real conflict, just merge."""
        reg = IntentRegistry()
        i1 = Intent(
            agent_id="a", target_anchor="x", target_key="y",
            intended_value="same", ttl_ms=60000,
        )
        i2 = Intent(
            agent_id="b", target_anchor="x", target_key="y",
            intended_value="same", ttl_ms=60000,
        )
        reg.declare(i1)
        report = reg.declare(i2)
        assert report is not None
        assert report.outcome == ConflictOutcome.MERGE


# ---------------------------------------------------------------------------
# Conflict Resolution Hierarchy
# ---------------------------------------------------------------------------

class TestConflictResolution:

    def test_authority_wins(self):
        """Higher priority agent wins."""
        reg = IntentRegistry()
        i_worker = Intent(
            agent_id="worker", target_anchor="sys", target_key="mode",
            intended_value="aggressive", ttl_ms=60000, priority=1.0,
            declared_at_ms=int(time.time() * 1000),
        )
        i_conductor = Intent(
            agent_id="conductor", target_anchor="sys", target_key="mode",
            intended_value="safe", ttl_ms=60000, priority=10.0,
            declared_at_ms=int(time.time() * 1000) + 50,  # declared LATER
        )
        reg.declare(i_worker)
        report = reg.declare(i_conductor)
        assert report.outcome == ConflictOutcome.AUTHORITY_WINS
        assert report.winner.agent_id == "conductor"

    def test_first_declared_wins_on_equal_priority(self):
        """Equal priority → earliest timestamp wins."""
        reg = IntentRegistry()
        now = int(time.time() * 1000)
        i1 = Intent(
            agent_id="a", target_anchor="x", target_key="y",
            intended_value="val_a", ttl_ms=60000, priority=5.0,
            declared_at_ms=now,
        )
        i2 = Intent(
            agent_id="b", target_anchor="x", target_key="y",
            intended_value="val_b", ttl_ms=60000, priority=5.0,
            declared_at_ms=now + 100,  # 100ms later
        )
        reg.declare(i1)
        report = reg.declare(i2)
        assert report.outcome == ConflictOutcome.FIRST_WINS
        assert report.winner.agent_id == "a"

    def test_escalate_on_true_concurrency(self):
        """Same priority + same timestamp → escalate to conductor."""
        reg = IntentRegistry()
        now = int(time.time() * 1000)
        i1 = Intent(
            agent_id="a", target_anchor="x", target_key="y",
            intended_value="val_a", ttl_ms=60000, priority=5.0,
            declared_at_ms=now,
        )
        i2 = Intent(
            agent_id="b", target_anchor="x", target_key="y",
            intended_value="val_b", ttl_ms=60000, priority=5.0,
            declared_at_ms=now,  # exact same timestamp
        )
        reg.declare(i1)
        report = reg.declare(i2)
        assert report.outcome == ConflictOutcome.ESCALATE
        assert report.winner is None


# ---------------------------------------------------------------------------
# Safety Queries
# ---------------------------------------------------------------------------

class TestSafetyQueries:

    def test_safe_when_no_intents(self):
        reg = IntentRegistry()
        assert reg.is_safe("a", "x.y") is True

    def test_safe_when_only_own_intent(self):
        reg = IntentRegistry()
        i = Intent(agent_id="a", target_anchor="x", target_key="y", ttl_ms=60000)
        reg.declare(i)
        assert reg.is_safe("a", "x.y") is True

    def test_unsafe_when_higher_priority_intent_exists(self):
        reg = IntentRegistry()
        i_high = Intent(
            agent_id="conductor", target_anchor="x", target_key="y",
            ttl_ms=60000, priority=10.0,
        )
        i_low = Intent(
            agent_id="worker", target_anchor="x", target_key="y",
            ttl_ms=60000, priority=1.0,
        )
        reg.declare(i_high)
        reg.declare(i_low)
        assert reg.is_safe("worker", "x.y") is False
        assert reg.is_safe("conductor", "x.y") is True

    def test_safe_after_other_completes(self):
        reg = IntentRegistry()
        i1 = Intent(agent_id="a", target_anchor="x", target_key="y", ttl_ms=60000)
        i2 = Intent(agent_id="b", target_anchor="x", target_key="y", ttl_ms=60000)
        reg.declare(i1)
        reg.declare(i2)
        # a completes — now b should be safe
        reg.release("a", "x.y")
        assert reg.is_safe("b", "x.y") is True

    def test_unsafe_without_declaring_intent(self):
        """Agent that didn't declare intent can't safely mutate a contested target."""
        reg = IntentRegistry()
        i = Intent(agent_id="a", target_anchor="x", target_key="y", ttl_ms=60000)
        reg.declare(i)
        assert reg.is_safe("b", "x.y") is False  # b didn't declare

    def test_contention_count(self):
        reg = IntentRegistry()
        i1 = Intent(agent_id="a", target_anchor="x", target_key="y", ttl_ms=60000)
        i2 = Intent(agent_id="b", target_anchor="x", target_key="y", ttl_ms=60000)
        i3 = Intent(agent_id="c", target_anchor="x", target_key="y", ttl_ms=60000)
        reg.declare(i1)
        reg.declare(i2)
        reg.declare(i3)
        assert reg.contention("x.y") == 3


# ---------------------------------------------------------------------------
# SCL Serialization
# ---------------------------------------------------------------------------

class TestIntentSCL:

    def test_intent_to_scl(self):
        i = Intent(
            agent_id="agent_04",
            target_anchor="env.db",
            target_key="pool_size",
            intended_value="25",
            action="mutate",
            ttl_ms=500,
        )
        record = i.to_scl()
        assert record.anchor.name == "agent_04"
        assert record.relation.verb == "intent"
        assert record.scope.get("target") == "env.db.pool_size"
        assert record.scope.get("action") == "mutate"
        assert record.scope.get("value") == "25"

    def test_intent_to_delta(self):
        i = Intent(
            agent_id="a",
            target_anchor="x",
            target_key="y",
            intended_value="42",
            ttl_ms=1000,
        )
        d = i.to_delta()
        assert d.agent_id == "a"
        assert "__intent__.x.y" in d.set_keys

    def test_conflict_report_to_scl(self):
        i1 = Intent(agent_id="a", target_anchor="x", target_key="y", ttl_ms=60000)
        i2 = Intent(agent_id="b", target_anchor="x", target_key="y", ttl_ms=60000)
        report = ConflictReport(
            target="x.y",
            intents=[i1, i2],
            outcome=ConflictOutcome.AUTHORITY_WINS,
            winner=i1,
            reason="test",
        )
        record = report.to_scl()
        assert record.anchor.name == "conflict"
        assert record.relation.verb == "intent_collision"
        assert "a+b" in record.scope.get("agents")


# ---------------------------------------------------------------------------
# Registry Stats & Active Intents
# ---------------------------------------------------------------------------

class TestRegistryOperations:

    def test_stats(self):
        reg = IntentRegistry()
        i1 = Intent(agent_id="a", target_anchor="x", target_key="y", ttl_ms=60000)
        i2 = Intent(agent_id="b", target_anchor="z", target_key="w", ttl_ms=60000)
        reg.declare(i1)
        reg.declare(i2)
        s = reg.stats
        assert s["total_declared"] == 2
        assert s["currently_alive"] == 2
        assert s["conflicts_detected"] == 0

    def test_active_intents_filtered(self):
        reg = IntentRegistry()
        i1 = Intent(agent_id="a", target_anchor="x", target_key="y", ttl_ms=60000)
        i2 = Intent(agent_id="b", target_anchor="x", target_key="z", ttl_ms=60000)
        reg.declare(i1)
        reg.declare(i2)
        assert len(reg.active_intents(agent_id="a")) == 1
        assert len(reg.active_intents()) == 2

    def test_expired_intents_pruned(self):
        reg = IntentRegistry()
        i = Intent(
            agent_id="a", target_anchor="x", target_key="y",
            ttl_ms=1,
            declared_at_ms=int(time.time() * 1000) - 100,
        )
        reg.declare(i)
        assert reg.contention("x.y") == 0  # expired on query
