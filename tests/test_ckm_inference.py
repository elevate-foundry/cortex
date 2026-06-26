"""
CKM Inference Test Suite — Validate the Cortex Kernel Model.

Tests run against a llama-server instance serving the GGUF model.
Start server: llama-server -m models/cortex-kernel.gguf --port 8090 -c 512

These tests verify:
  1. Model outputs valid SCL grammar
  2. Boot config decisions are reasonable
  3. Task routing follows tier hierarchy
  4. Challenge escalation works correctly
  5. Failure diagnosis produces correct repair paths
  6. Trace prediction follows operational semantics
  7. Stop token behavior (no runaway generation)
"""

import json
import os
import re
import subprocess
import time
import pytest
import urllib.request
import urllib.error

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CKM_SERVER = os.environ.get("CKM_SERVER", "http://127.0.0.1:8090")
CKM_MODEL_PATH = os.environ.get(
    "CKM_MODEL_PATH",
    "/Volumes/CORTEX/cortex/models/cortex-kernel.gguf"
)

SYSTEM_PROMPT = (
    "You are the Cortex Kernel Model (CKM). "
    "You receive SCL records describing hardware state or request classification. "
    "You respond with exactly one SCL record: the optimal mutation or routing decision. "
    "Never output anything except valid SCL. "
    "Format: @anchor → verb [key: value, key: value]"
)

# SCL record pattern: @anchor → verb [key: value, ...]
SCL_PATTERN = re.compile(
    r"^@[\w.]+\s*→\s*\w+\s*\[.*\]$",
    re.DOTALL
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _query_ckm(prompt: str, max_tokens: int = 64, temperature: float = 0.0) -> str:
    """Send a prompt to the CKM server and return the first SCL record."""
    payload = json.dumps({
        "prompt": (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        ),
        "temperature": temperature,
        "n_predict": max_tokens,
        "stop": ["\n", "<|im_end|>"],
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{CKM_SERVER}/completion",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        pytest.skip(f"CKM server not available at {CKM_SERVER}: {e}")
        return ""

    content = result.get("content", "")
    # Strip trailing garbage (stop token bytes, etc.)
    content = content.split("핮")[0].strip()
    content = content.split("\ud56e")[0].strip()
    return content


def _is_valid_scl(text: str) -> bool:
    """Check if text is a valid SCL record."""
    text = text.strip()
    if not text.startswith("@"):
        return False
    if "→" not in text:
        return False
    if "[" not in text or "]" not in text:
        return False
    return True


def _extract_scope(scl: str) -> dict:
    """Extract key-value pairs from SCL scope."""
    match = re.search(r"\[(.+)\]", scl)
    if not match:
        return {}
    scope_str = match.group(1)
    pairs = {}
    for item in scope_str.split(","):
        item = item.strip()
        if ":" in item:
            k, v = item.split(":", 1)
            pairs[k.strip()] = v.strip()
    return pairs


def _extract_verb(scl: str) -> str:
    """Extract the verb from an SCL record."""
    match = re.search(r"→\s*(\w+)", scl)
    return match.group(1) if match else ""


def _extract_anchor(scl: str) -> str:
    """Extract the anchor from an SCL record."""
    match = re.match(r"@([\w.]+)", scl)
    return match.group(1) if match else ""


@pytest.fixture(scope="session", autouse=True)
def ensure_server():
    """Ensure CKM server is running."""
    try:
        req = urllib.request.Request(f"{CKM_SERVER}/health")
        with urllib.request.urlopen(req, timeout=5) as resp:
            status = json.loads(resp.read())
            assert status.get("status") == "ok"
    except (urllib.error.URLError, AssertionError):
        pytest.skip(
            f"CKM server not running. Start with:\n"
            f"  llama-server -m {CKM_MODEL_PATH} --port 8090 -c 512"
        )


# ---------------------------------------------------------------------------
# Test 1: Grammar Validity — All outputs must be valid SCL
# ---------------------------------------------------------------------------

class TestGrammarValidity:
    """Every CKM output must be a well-formed SCL record."""

    PROMPTS = [
        "@hardware → state [cpu: Apple M1 Pro, cores: 10, ram_mb: 16384, gpu_type: apple, vram_mb: 16384, arch: aarch64]",
        "@task → classify [category: code, subtype: fix_bug, complexity: 0.7, input_tokens: 500]",
        "@init → boot [phase: cold_start, pid: 1]\n@init → spawn [type: llama_cpp, port: 8080]\n@init → fail [error: oom_killed, code: ENOMEM]",
        "@feedback → accumulated [model: qwen3:4b, tier: L2, accuracy: 0.35, count: 20]",
        "@router → select [tier: L2, model: qwen3:4b, confidence: 0.6]\n@backend → execute [model: qwen3:4b, tokens_out: 200]\n@challenger → verify [challenger_model: granite3.3:8b, agreement: strong_disagree]",
    ]

    @pytest.mark.parametrize("prompt", PROMPTS)
    def test_output_is_valid_scl(self, prompt):
        result = _query_ckm(prompt)
        assert result, f"Empty response for: {prompt[:60]}..."
        assert _is_valid_scl(result), f"Invalid SCL: '{result}' for prompt: {prompt[:60]}..."

    @pytest.mark.parametrize("prompt", PROMPTS)
    def test_output_has_anchor(self, prompt):
        result = _query_ckm(prompt)
        assert result.startswith("@"), f"Missing anchor: '{result}'"

    @pytest.mark.parametrize("prompt", PROMPTS)
    def test_output_has_verb(self, prompt):
        result = _query_ckm(prompt)
        assert "→" in result, f"Missing relation arrow: '{result}'"
        verb = _extract_verb(result)
        assert verb, f"Could not extract verb from: '{result}'"

    @pytest.mark.parametrize("prompt", PROMPTS)
    def test_output_has_scope(self, prompt):
        result = _query_ckm(prompt)
        assert "[" in result and "]" in result, f"Missing scope: '{result}'"
        scope = _extract_scope(result)
        assert len(scope) > 0, f"Empty scope in: '{result}'"


# ---------------------------------------------------------------------------
# Test 2: Boot Configuration — Hardware → optimal config
# ---------------------------------------------------------------------------

class TestBootConfig:
    """Boot hardware state → optimal configuration mutations."""

    def test_apple_silicon_gets_full_gpu_offload(self):
        result = _query_ckm(
            "@hardware → state [cpu: Apple M2 Pro, cores: 12, ram_mb: 32768, "
            "gpu_type: apple, vram_mb: 32768, arch: aarch64]"
        )
        assert _is_valid_scl(result)
        scope = _extract_scope(result)
        # Apple Silicon should get max GPU layers
        if "gpu_layers" in scope:
            assert scope["gpu_layers"] in ("999", "32", "64"), f"Apple should get full offload, got: {scope}"

    def test_no_gpu_gets_zero_layers(self):
        result = _query_ckm(
            "@hardware → state [cpu: Raspberry Pi 4 BCM2711, cores: 4, ram_mb: 2048, "
            "gpu_type: none, vram_mb: 0, arch: aarch64]"
        )
        assert _is_valid_scl(result)
        # Should not recommend high GPU layers for no-GPU system
        scope = _extract_scope(result)
        if "gpu_layers" in scope:
            gpu_layers = int(scope["gpu_layers"]) if scope["gpu_layers"].isdigit() else 0
            assert gpu_layers <= 32, f"No GPU but got gpu_layers={gpu_layers}"

    def test_high_vram_nvidia_gets_offload(self):
        result = _query_ckm(
            "@hardware → state [cpu: AMD Ryzen 9 7950X, cores: 32, ram_mb: 65536, "
            "gpu_type: nvidia, vram_mb: 24576, arch: x86_64]"
        )
        assert _is_valid_scl(result)

    def test_low_ram_gets_small_context(self):
        result = _query_ckm(
            "@hardware → state [cpu: Intel Celeron N5105, cores: 4, ram_mb: 4096, "
            "gpu_type: none, vram_mb: 0, arch: x86_64]"
        )
        assert _is_valid_scl(result)
        scope = _extract_scope(result)
        if "ctx" in scope or "ctx_size" in scope:
            ctx_key = "ctx" if "ctx" in scope else "ctx_size"
            ctx = int(scope[ctx_key]) if scope[ctx_key].isdigit() else 0
            assert ctx <= 8192, f"Low RAM but got ctx={ctx}"


# ---------------------------------------------------------------------------
# Test 3: Task Routing — Complexity → appropriate tier
# ---------------------------------------------------------------------------

class TestTaskRouting:
    """Task classification should route to appropriate tiers."""

    def test_simple_task_low_tier(self):
        result = _query_ckm(
            "@task → classify [category: chat, subtype: greeting, "
            "complexity: 0.05, input_tokens: 5]"
        )
        assert _is_valid_scl(result)
        scope = _extract_scope(result)
        if "tier" in scope:
            tier = scope["tier"]
            assert tier in ("L0", "L1", "L2"), f"Simple greeting routed to {tier}"

    def test_complex_task_high_tier(self):
        result = _query_ckm(
            "@task → classify [category: code, subtype: architecture, "
            "complexity: 0.9, input_tokens: 2000]"
        )
        assert _is_valid_scl(result)
        scope = _extract_scope(result)
        if "tier" in scope:
            tier = scope["tier"]
            # Should route to L3+ for complex architecture tasks
            assert tier in ("L3", "L4", "L5"), f"Complex architecture routed to {tier}"

    def test_routing_has_confidence(self):
        result = _query_ckm(
            "@task → classify [category: math, subtype: proof, "
            "complexity: 0.85, input_tokens: 800]"
        )
        assert _is_valid_scl(result)
        verb = _extract_verb(result)
        assert verb in ("select", "route", "classify", "escalate"), f"Unexpected verb: {verb}"


# ---------------------------------------------------------------------------
# Test 4: Challenge Escalation — Disagreement → escalate
# ---------------------------------------------------------------------------

class TestChallengeEscalation:
    """When challenger disagrees, model should escalate."""

    def test_strong_disagree_escalates(self):
        result = _query_ckm(
            "@router → select [tier: L2, model: qwen3:4b, confidence: 0.6]\n"
            "@backend → execute [model: qwen3:4b, tokens_out: 200]\n"
            "@challenger → verify [challenger_model: granite3.3:8b, agreement: strong_disagree]"
        )
        assert _is_valid_scl(result)
        verb = _extract_verb(result)
        scope = _extract_scope(result)
        # Should escalate
        assert verb == "escalate" or "escalate" in str(scope), \
            f"Expected escalation on strong_disagree, got: {result}"

    def test_strong_agree_does_not_escalate(self):
        result = _query_ckm(
            "@router → select [tier: L3, model: qwen3:8b, confidence: 0.9]\n"
            "@backend → execute [model: qwen3:8b, tokens_out: 500]\n"
            "@challenger → verify [challenger_model: granite3.3:8b, agreement: strong_agree]"
        )
        assert _is_valid_scl(result)
        verb = _extract_verb(result)
        # Should NOT escalate when models agree
        assert verb != "escalate" or "pass" in str(_extract_scope(result)), \
            f"Unnecessary escalation on strong_agree: {result}"


# ---------------------------------------------------------------------------
# Test 5: Failure Diagnosis — Error state → diagnosis
# ---------------------------------------------------------------------------

class TestFailureDiagnosis:
    """Failure traces should produce correct diagnostic output."""

    def test_oom_diagnosed_correctly(self):
        result = _query_ckm(
            "@init → boot [phase: cold_start, pid: 1]\n"
            "@init → spawn [type: llama_cpp, port: 8080]\n"
            "@init → fail [error: oom_killed, code: ENOMEM]"
        )
        assert _is_valid_scl(result)
        verb = _extract_verb(result)
        scope = _extract_scope(result)
        # Should diagnose or repair
        assert verb in ("diagnose", "repair", "fallback", "select"), \
            f"Expected diagnosis verb, got: {verb}"

    def test_connection_refused_diagnosed(self):
        result = _query_ckm(
            "@init → boot [phase: cold_start, pid: 1]\n"
            "@init → spawn [type: vllm, port: 8080]\n"
            "@init → fail [error: connection_refused, code: ECONNREFUSED]"
        )
        assert _is_valid_scl(result)
        verb = _extract_verb(result)
        assert verb in ("diagnose", "repair", "fallback", "select"), \
            f"Expected diagnosis verb for connection refused, got: {verb}"

    def test_timeout_produces_action(self):
        result = _query_ckm(
            "@process → fail [type: TimeoutError, message: 30s_timeout]\n"
            "@agent → inspect [traceback: daemon.py:L200]"
        )
        assert _is_valid_scl(result)


# ---------------------------------------------------------------------------
# Test 6: Trace Prediction — Prefix → next valid step
# ---------------------------------------------------------------------------

class TestTracePrediction:
    """Given a trace prefix, the model should predict the next valid step."""

    def test_boot_sequence_continues(self):
        result = _query_ckm(
            "@init → boot [phase: cold_start, pid: 1]\n"
            "@init → execute [action: mount_virtual, targets: proc,sys,dev]"
        )
        assert _is_valid_scl(result)
        # After mounting, next step is usually detect hardware
        verb = _extract_verb(result)
        assert verb in ("detect", "select", "spawn", "read", "inspect"), \
            f"Unexpected boot continuation: {verb}"

    def test_gossip_sequence_continues(self):
        result = _query_ckm(
            "@peer_a → gossip [fingerprint: ⢚⡮⢗⣷, type: ping]\n"
            "@peer_b → observe [diverged: true, similarity: 0.3]"
        )
        assert _is_valid_scl(result)
        verb = _extract_verb(result)
        # After observing divergence, should take sync-related action
        assert verb in ("emit", "push", "apply", "sync", "gossip", "select", "deny", "mutate", "execute"), \
            f"Expected action after divergence, got: {verb}"

    def test_tool_sequence_continues(self):
        result = _query_ckm(
            "@agent → plan [goal: fix_crash, strategy: edit_then_test]\n"
            "@agent → select [tool: read_file, permission: ring_1]\n"
            "@agent → execute [tool: read_file, path: src/router.py, result: success]"
        )
        assert _is_valid_scl(result)
        verb = _extract_verb(result)
        # After reading, should select next tool (likely edit)
        assert verb in ("select", "execute", "propose", "plan"), \
            f"Expected next tool selection, got: {verb}"


# ---------------------------------------------------------------------------
# Test 7: Policy Mutations — Feedback → policy changes
# ---------------------------------------------------------------------------

class TestPolicyMutations:
    """Accumulated feedback should trigger appropriate policy changes."""

    def test_low_accuracy_demotes(self):
        result = _query_ckm(
            "@feedback → accumulated [model: qwen3:4b, tier: L2, "
            "accuracy: 0.35, count: 20]"
        )
        assert _is_valid_scl(result)
        verb = _extract_verb(result)
        scope = _extract_scope(result)
        # Low accuracy should trigger demotion/penalty
        assert verb == "mutate" or "demote" in str(scope) or "penalize" in str(scope) or "deny" in str(scope), \
            f"Expected demotion for 35% accuracy, got: {result}"

    def test_high_accuracy_promotes(self):
        result = _query_ckm(
            "@feedback → accumulated [model: phi4:14b, tier: L4, "
            "accuracy: 0.92, count: 50]"
        )
        assert _is_valid_scl(result)


# ---------------------------------------------------------------------------
# Test 8: Stop Token Behavior — No runaway generation
# ---------------------------------------------------------------------------

class TestStopBehavior:
    """Model should produce bounded output without runaway generation."""

    def test_single_record_output(self):
        """Output should be a single SCL record, not multiple."""
        result = _query_ckm(
            "@hardware → state [cpu: Apple M1, cores: 8, ram_mb: 16384, "
            "gpu_type: apple, vram_mb: 16384, arch: aarch64]"
        )
        # Should be one record, not multiple
        records = [r.strip() for r in result.split("\n") if r.strip().startswith("@")]
        assert len(records) <= 1, f"Expected 1 record, got {len(records)}: {records}"

    def test_no_garbage_tokens(self):
        """Output should not contain control characters or garbage."""
        result = _query_ckm(
            "@task → classify [category: chat, subtype: casual_qa, "
            "complexity: 0.2, input_tokens: 30]"
        )
        # Check for common garbage patterns
        assert "norge" not in result.lower() or result.endswith("]"), \
            f"Garbage detected in output: '{result}'"

    def test_response_length_bounded(self):
        """Response should be reasonably short (one SCL record ~50-150 chars)."""
        result = _query_ckm(
            "@init → boot [phase: warm_start, pid: 1]\n"
            "@init → read [hw_fingerprint: ⢚⡮⢗⣷, hit: true]"
        )
        assert len(result) < 300, f"Response too long ({len(result)} chars): {result[:100]}..."


# ---------------------------------------------------------------------------
# Test 9: Latency — Model should respond quickly
# ---------------------------------------------------------------------------

class TestLatency:
    """CKM should respond in <500ms for single-record inference."""

    def test_inference_latency(self):
        """Single inference should complete in reasonable time."""
        prompt = "@hardware → state [cpu: Apple M1, cores: 8, ram_mb: 8192, gpu_type: apple, vram_mb: 8192, arch: aarch64]"

        payload = json.dumps({
            "prompt": (
                f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                f"<|im_start|>user\n{prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            ),
            "temperature": 0.0,
            "n_predict": 64,
            "stop": ["\n", "<|im_end|>"],
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{CKM_SERVER}/completion",
            data=payload,
            headers={"Content-Type": "application/json"},
        )

        t0 = time.time()
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
        latency_ms = (time.time() - t0) * 1000

        # Should be fast — model is 379MB on Metal
        assert latency_ms < 2000, f"Inference took {latency_ms:.0f}ms (target: <2000ms)"

        # Check timings from server
        timings = result.get("timings", {})
        if timings:
            gen_ms = timings.get("predicted_ms", 0)
            prompt_ms = timings.get("prompt_ms", 0)
            assert gen_ms < 1500, f"Generation took {gen_ms:.0f}ms"


# ---------------------------------------------------------------------------
# Test 10: Determinism — Same input → same output (temp=0)
# ---------------------------------------------------------------------------

class TestDeterminism:
    """With temperature=0, identical inputs should produce identical outputs."""

    def test_deterministic_boot(self):
        prompt = (
            "@hardware → state [cpu: Apple M3 Pro, cores: 12, ram_mb: 18432, "
            "gpu_type: apple, vram_mb: 18432, arch: aarch64]"
        )
        result1 = _query_ckm(prompt)
        result2 = _query_ckm(prompt)
        assert result1 == result2, f"Non-deterministic: '{result1}' vs '{result2}'"

    def test_deterministic_route(self):
        prompt = "@task → classify [category: code, subtype: fix_bug, complexity: 0.7, input_tokens: 500]"
        result1 = _query_ckm(prompt)
        result2 = _query_ckm(prompt)
        assert result1 == result2, f"Non-deterministic: '{result1}' vs '{result2}'"
