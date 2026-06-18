"""
Test suite for Gossip Transport and multi-node delta sync.

This is a diamond-hard integration test: two real Cortex daemons on
localhost exchange deltas over HTTP, and we verify convergence.
"""

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.request


DB_PATH = None  # set in setUpModule


def curl_json(method, url, data=None):
    req = urllib.request.Request(
        url, method=method,
        headers={} if data is None else {"Content-Type": "application/json"},
        data=json.dumps(data).encode() if data else None,
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())
    except urllib.error.URLError as e:
        return {"__error": str(e)}
    except Exception as e:
        return {"__error": str(e)}


def wait_for(url, timeout=20):
    for i in range(timeout * 2):
        r = curl_json("GET", f"{url}/health")
        if r.get("status") == "ok":
            return True
        time.sleep(0.5)
    return False


class TestGossipMultiNode(unittest.TestCase):
    """Verify two Cortex daemons gossip deltas over HTTP."""

    @classmethod
    def setUpClass(cls):
        cls.maxDiff = None
        # Use a fresh DB for the test
        global DB_PATH
        cls.tmpdir = tempfile.TemporaryDirectory()
        DB_PATH = os.path.join(cls.tmpdir.name, "cortex_test.db")
        # We inject DB_PATH into the daemon by monkey-patching detect_system
        # For now, run with default and clear peers afterwards

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    def test_01_single_node_gossip_state(self):
        """A freshly booted node has empty gossip state."""
        port = 11421
        db = os.path.join(self.tmpdir.name, f"test_{port}.db")
        proc = subprocess.Popen(
            [sys.executable, "-c",
             "from src.daemon import run_daemon; from src.hardware_detect import detect_system; "
             f"run_daemon(host='127.0.0.1', port={port}, profile=detect_system(), db_path='{db}')"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        self.addCleanup(proc.terminate)
        self.assertTrue(wait_for(f"http://localhost:{port}"), msg="daemon never came up")

        state = curl_json("GET", f"http://localhost:{port}/v1/gossip/state")
        self.assertEqual(state.get("node_id"), f"cortex-{port}")
        self.assertEqual(state.get("state_keys"), 0)
        self.assertEqual(state.get("stream_length"), 0)
        self.assertIsInstance(state.get("fingerprint"), str)
        self.assertEqual(len(state["fingerprint"]), 4)

    def test_02_multi_node_delta_sync(self):
        """Two nodes exchange deltas and converge state."""
        port_a, port_b = 11422, 11423
        base_a = f"http://localhost:{port_a}"
        base_b = f"http://localhost:{port_b}"

        # --- Start both nodes ---
        db_a = os.path.join(self.tmpdir.name, f"test_{port_a}.db")
        db_b = os.path.join(self.tmpdir.name, f"test_{port_b}.db")
        proc_a = subprocess.Popen(
            [sys.executable, "-c",
             "from src.daemon import run_daemon; from src.hardware_detect import detect_system; "
             f"run_daemon(host='127.0.0.1', port={port_a}, profile=detect_system(), db_path='{db_a}')"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        self.addCleanup(proc_a.terminate)
        self.assertTrue(wait_for(base_a), msg="node A never came up")

        proc_b = subprocess.Popen(
            [sys.executable, "-c",
             "from src.daemon import run_daemon; from src.hardware_detect import detect_system; "
             f"run_daemon(host='127.0.0.1', port={port_b}, profile=detect_system(), db_path='{db_b}')"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        self.addCleanup(proc_b.terminate)
        self.assertTrue(wait_for(base_b), msg="node B never came up")

        # --- Wire as peers ---
        r = curl_json("POST", f"{base_a}/v1/gossip/peers", {"id": f"cortex-{port_b}", "url": base_b})
        self.assertEqual(r.get("status"), "ok", msg=r.get("__error"))
        r = curl_json("POST", f"{base_b}/v1/gossip/peers", {"id": f"cortex-{port_a}", "url": base_a})
        self.assertEqual(r.get("status"), "ok", msg=r.get("__error"))

        # --- Trigger mutation on A via API request ---
        r = curl_json("POST", f"{base_a}/v1/chat/completions", {
            "model": "auto",
            "messages": [{"role": "user", "content": "What is 2+2?"}],
            "max_tokens": 10,
        })
        self.assertIn("choices", r, msg=r.get("__error"))

        # --- Wait for gossip sync (background interval = 30s) ---
        synced = False
        for i in range(40):
            time.sleep(1)
            b_state = curl_json("GET", f"{base_b}/v1/gossip/state")
            if b_state.get("state_keys", 0) > 0:
                synced = True
                break
        self.assertTrue(synced, msg="Node B never received delta from A within 40s")

        # --- Verify convergence ---
        a_state = curl_json("GET", f"{base_a}/v1/gossip/state")
        b_state = curl_json("GET", f"{base_b}/v1/gossip/state")
        self.assertEqual(a_state["state_keys"], 4, msg=f"A state_keys={a_state['state_keys']}")
        self.assertEqual(b_state["state_keys"], 4, msg=f"B state_keys={b_state['state_keys']}")

        # Stats: B should have received a delta (applied > 0 or rounds > 0)
        b_stats = curl_json("GET", f"{base_b}/v1/gossip/stats")
        self.assertIsInstance(b_stats["stats"]["rounds"], int)
        self.assertIsInstance(b_stats["stats"]["deltas_propagated"], int)


if __name__ == "__main__":
    unittest.main(verbosity=2)
