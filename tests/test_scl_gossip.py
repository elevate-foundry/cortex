"""
Test: Gossip convergence protocol — full 11-step verification.

Steps:
  1. Start Node A with N keys and M deltas.
  2. Start Node B empty.
  3. Run gossip until quiescence.
  4. Assert same materialized state.
  5. Assert same state fingerprint.
  6. Assert either same delta fingerprint or valid compacted-history proof.
  7. Restart both nodes (reconstruct from delta streams).
  8. Assert fingerprints survive reboot.
  9. Mutate Node B.
  10. Gossip back to Node A.
  11. Assert bidirectional convergence.
"""

import unittest

from src.scl.gossip import Peer, Swarm, GossipMessageType
from src.scl.delta import (
    Delta,
    DeltaStream,
    SemanticState,
    VectorClock,
    MergeStrategy,
    apply_delta,
    diff,
)
from src.braille.fingerprint import fingerprint as scl_fingerprint, similarity


class TestGossipConvergenceProtocol(unittest.TestCase):
    """Full 11-step gossip convergence proof."""

    # Test parameters
    N_KEYS = 10
    M_DELTAS = 5

    def _build_initial_state(self) -> SemanticState:
        """Build a state with N keys."""
        entries = {f"key_{i}": f"value_{i}" for i in range(self.N_KEYS)}
        return SemanticState(entries=entries)

    def _build_deltas(self, agent_id: str, base_state: SemanticState) -> list[Delta]:
        """Build M additional deltas on top of base_state."""
        deltas = []
        for i in range(self.M_DELTAS):
            d = Delta(
                agent_id=agent_id,
                set_keys={f"delta_key_{i}": f"delta_value_{i}"},
                seq=i + 1,
                weight=1.0,
            )
            deltas.append(d)
        return deltas

    def _reconstruct_peer(self, peer: Peer) -> Peer:
        """Simulate a 'reboot' by reconstructing a peer from its delta stream.

        This mimics restarting a node that persists its DeltaStream.
        The new peer replays all deltas from scratch.
        """
        new_peer = Peer(
            agent_id=peer.agent_id,
            initial_state=None,
            merge_strategy=peer.merge_strategy,
            weight=peer.weight,
        )
        for delta in peer.stream.deltas:
            new_peer.receive_delta(delta)
        return new_peer

    def test_full_gossip_convergence_protocol(self):
        """Execute the 11-step convergence verification."""

        # ==================================================================
        # Step 1: Start Node A with N keys and M deltas
        # ==================================================================
        initial_state_a = self._build_initial_state()
        node_a = Peer(
            agent_id="node_a",
            initial_state=initial_state_a,
            merge_strategy=MergeStrategy.LWW,
            weight=1.0,
        )
        # Apply M additional deltas to Node A
        additional_deltas = self._build_deltas("node_a", initial_state_a)
        for d in additional_deltas:
            node_a.receive_delta(d)

        # Verify Node A has the expected state: N original keys + M delta keys
        self.assertEqual(
            len(node_a.state.entries),
            self.N_KEYS + self.M_DELTAS,
            f"Node A should have {self.N_KEYS + self.M_DELTAS} keys, "
            f"got {len(node_a.state.entries)}",
        )
        # Node A stream: 1 bootstrap delta + M additional = M+1
        self.assertEqual(node_a.stream.length, 1 + self.M_DELTAS)

        # ==================================================================
        # Step 2: Start Node B empty
        # ==================================================================
        node_b = Peer(
            agent_id="node_b",
            initial_state=None,
            merge_strategy=MergeStrategy.LWW,
            weight=1.0,
        )
        self.assertEqual(len(node_b.state.entries), 0)
        self.assertEqual(node_b.stream.length, 0)

        # ==================================================================
        # Step 3: Run gossip until quiescence
        # ==================================================================
        swarm = Swarm(convergence_threshold=0.95, merge_strategy=MergeStrategy.LWW)
        swarm.peers["node_a"] = node_a
        swarm.peers["node_b"] = node_b

        rounds = swarm.run_until_converged(max_rounds=20)
        self.assertLess(rounds, 20, "Swarm did not converge within 20 rounds")
        self.assertTrue(swarm.is_converged(), "Swarm reports not converged")

        # ==================================================================
        # Step 4: Assert same materialized state
        # ==================================================================
        state_a = node_a.state
        state_b = node_b.state
        self.assertEqual(
            state_a.entries,
            state_b.entries,
            "Materialized states differ after gossip convergence",
        )
        self.assertEqual(len(state_a.entries), self.N_KEYS + self.M_DELTAS)

        # ==================================================================
        # Step 5: Assert same state fingerprint
        # ==================================================================
        fp_a = node_a.state_fingerprint()
        fp_b = node_b.state_fingerprint()
        self.assertEqual(
            fp_a, fp_b,
            f"State fingerprints differ: A={fp_a!r} vs B={fp_b!r}",
        )
        # Fingerprints should be 4 Braille characters
        self.assertEqual(len(fp_a), 4)

        # ==================================================================
        # Step 6: Assert either same delta fingerprint or valid compacted-history proof
        # ==================================================================
        # Delta streams may differ in ordering/representation, but the
        # compacted form (full diff from empty → current) must be equivalent.
        compact_a = node_a.stream.compact(0, node_a.stream.length)
        compact_b = node_b.stream.compact(0, node_b.stream.length)

        # The compacted delta applied to empty state should yield same result
        empty = SemanticState()
        result_a = apply_delta(empty, compact_a)
        result_b = apply_delta(empty, compact_b)
        self.assertEqual(
            result_a.entries,
            result_b.entries,
            "Compacted history proof failed — applying compacted deltas to "
            "empty state yields different results",
        )

        # Additionally check: content_hash match (stronger than fingerprint)
        self.assertEqual(state_a.content_hash(), state_b.content_hash())

        # ==================================================================
        # Step 7: Restart both nodes (reconstruct from delta streams)
        # ==================================================================
        node_a_rebooted = self._reconstruct_peer(node_a)
        node_b_rebooted = self._reconstruct_peer(node_b)

        # ==================================================================
        # Step 8: Assert fingerprints survive reboot
        # ==================================================================
        fp_a_reboot = node_a_rebooted.state_fingerprint()
        fp_b_reboot = node_b_rebooted.state_fingerprint()

        self.assertEqual(
            fp_a, fp_a_reboot,
            f"Node A fingerprint changed after reboot: {fp_a!r} → {fp_a_reboot!r}",
        )
        self.assertEqual(
            fp_b, fp_b_reboot,
            f"Node B fingerprint changed after reboot: {fp_b!r} → {fp_b_reboot!r}",
        )
        # Also verify materialized state survives reboot
        self.assertEqual(node_a_rebooted.state.entries, state_a.entries)
        self.assertEqual(node_b_rebooted.state.entries, state_b.entries)

        # ==================================================================
        # Step 9: Mutate Node B
        # ==================================================================
        node_b_rebooted.mutate({"new_key_from_b": "hello_from_b", "key_0": "overwritten_by_b"})

        # Verify B's state changed
        self.assertEqual(node_b_rebooted.state.entries["new_key_from_b"], "hello_from_b")
        self.assertEqual(node_b_rebooted.state.entries["key_0"], "overwritten_by_b")

        # Fingerprints should now differ
        fp_a_post = node_a_rebooted.state_fingerprint()
        fp_b_post = node_b_rebooted.state_fingerprint()
        self.assertNotEqual(
            fp_a_post, fp_b_post,
            "Fingerprints should differ after B mutates",
        )

        # ==================================================================
        # Step 10: Gossip back to Node A
        # ==================================================================
        swarm2 = Swarm(convergence_threshold=0.95, merge_strategy=MergeStrategy.LWW)
        swarm2.peers["node_a"] = node_a_rebooted
        swarm2.peers["node_b"] = node_b_rebooted

        rounds2 = swarm2.run_until_converged(max_rounds=20)
        self.assertLess(rounds2, 20, "Swarm did not re-converge within 20 rounds")
        self.assertTrue(swarm2.is_converged())

        # ==================================================================
        # Step 11: Assert bidirectional convergence
        # ==================================================================
        final_a = node_a_rebooted.state
        final_b = node_b_rebooted.state

        # Same materialized state
        self.assertEqual(
            final_a.entries,
            final_b.entries,
            "Bidirectional convergence failed — states differ after B→A gossip",
        )

        # Same fingerprint
        final_fp_a = node_a_rebooted.state_fingerprint()
        final_fp_b = node_b_rebooted.state_fingerprint()
        self.assertEqual(
            final_fp_a, final_fp_b,
            f"Bidirectional convergence failed — fingerprints: "
            f"A={final_fp_a!r} vs B={final_fp_b!r}",
        )

        # B's mutation propagated to A
        self.assertEqual(final_a.entries["new_key_from_b"], "hello_from_b")
        self.assertEqual(final_a.entries["key_0"], "overwritten_by_b")

        # Original keys still present
        for i in range(1, self.N_KEYS):
            self.assertIn(f"key_{i}", final_a.entries)
        for i in range(self.M_DELTAS):
            self.assertIn(f"delta_key_{i}", final_a.entries)

        # Content hash final agreement
        self.assertEqual(final_a.content_hash(), final_b.content_hash())

        # Convergence matrix should be all 1.0
        matrix = swarm2.convergence_matrix()
        for pid_a, row in matrix.items():
            for pid_b, sim_val in row.items():
                self.assertAlmostEqual(
                    sim_val, 1.0, places=2,
                    msg=f"Convergence matrix[{pid_a}][{pid_b}] = {sim_val}, expected 1.0",
                )


class TestGossipScaling(unittest.TestCase):
    """Verify O(log N) convergence scaling property."""

    def test_5_nodes_converge_quickly(self):
        """5 nodes should converge in O(log 5) ≈ 3 rounds."""
        swarm = Swarm(merge_strategy=MergeStrategy.LWW)

        # Only one node has initial state
        state = SemanticState(entries={f"k{i}": f"v{i}" for i in range(5)})
        swarm.add_peer("p0", initial_state=state, weight=1.0)
        for i in range(1, 5):
            swarm.add_peer(f"p{i}", weight=1.0)

        rounds = swarm.run_until_converged(max_rounds=20)
        self.assertTrue(swarm.is_converged())
        # O(log 5) ≈ 2-3, allow generous bound
        self.assertLessEqual(rounds, 5, f"5 nodes took {rounds} rounds (expected ≤ 5)")

    def test_10_nodes_converge(self):
        """10 nodes should converge in O(log 10) ≈ 4 rounds."""
        swarm = Swarm(merge_strategy=MergeStrategy.LWW)

        state = SemanticState(entries={f"k{i}": f"v{i}" for i in range(8)})
        swarm.add_peer("p0", initial_state=state, weight=1.0)
        for i in range(1, 10):
            swarm.add_peer(f"p{i}", weight=1.0)

        rounds = swarm.run_until_converged(max_rounds=30)
        self.assertTrue(swarm.is_converged())
        self.assertLessEqual(rounds, 8, f"10 nodes took {rounds} rounds (expected ≤ 8)")


class TestGossipEdgeCases(unittest.TestCase):
    """Edge cases for gossip protocol correctness."""

    def test_empty_to_empty_converges_immediately(self):
        """Two empty nodes should already be converged."""
        a = Peer("a")
        b = Peer("b")
        swarm = Swarm()
        swarm.peers["a"] = a
        swarm.peers["b"] = b
        self.assertTrue(swarm.is_converged())

    def test_concurrent_mutations_resolve(self):
        """Both nodes mutate concurrently — LWW resolves."""
        state = SemanticState(entries={"shared": "original"})
        a = Peer("a", initial_state=state, merge_strategy=MergeStrategy.LWW)
        b = Peer("b", initial_state=state, merge_strategy=MergeStrategy.LWW)

        # Concurrent mutations to same key
        a.mutate({"shared": "from_a"})
        b.mutate({"shared": "from_b"})

        swarm = Swarm(merge_strategy=MergeStrategy.LWW)
        swarm.peers["a"] = a
        swarm.peers["b"] = b

        rounds = swarm.run_until_converged(max_rounds=10)
        self.assertTrue(swarm.is_converged())

        # Both should have same value (LWW picks one)
        self.assertEqual(a.state.entries["shared"], b.state.entries["shared"])

    def test_delete_propagates(self):
        """A deletion on one node propagates to the other.

        Node A has the initial state, Node B starts empty and receives
        via gossip. Then A deletes a key — the deletion propagates.
        """
        state = SemanticState(entries={"keep": "yes", "remove": "this"})
        a = Peer("a", initial_state=state)
        b = Peer("b")  # B starts empty — receives state via gossip

        # First, sync so B has A's state
        swarm = Swarm()
        swarm.peers["a"] = a
        swarm.peers["b"] = b
        swarm.run_until_converged(max_rounds=5)
        self.assertTrue(swarm.is_converged())
        self.assertIn("remove", b.state.entries)

        # Now A deletes the key
        a.mutate({}, deletes={"remove"})
        self.assertNotIn("remove", a.state.entries)

        # Gossip the deletion
        rounds = swarm.run_until_converged(max_rounds=10)
        self.assertTrue(swarm.is_converged())
        self.assertNotIn("remove", b.state.entries)
        self.assertEqual(b.state.entries["keep"], "yes")

    def test_fingerprint_determinism(self):
        """Same state always produces same fingerprint."""
        state = SemanticState(entries={"x": "1", "y": "2"})
        a = Peer("a", initial_state=state)
        b = Peer("b", initial_state=state)

        # After gossip, fingerprints must match
        swarm = Swarm()
        swarm.peers["a"] = a
        swarm.peers["b"] = b
        swarm.run_until_converged(max_rounds=5)

        fp1 = a.state_fingerprint()
        fp2 = b.state_fingerprint()
        self.assertEqual(fp1, fp2)

        # Calling multiple times gives same result
        self.assertEqual(a.state_fingerprint(), fp1)
        self.assertEqual(b.state_fingerprint(), fp2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
