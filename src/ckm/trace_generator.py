"""
CKM Trace Generator — Exhaustive typed state traces for training.

This generates the REAL training corpus for the Cortex Kernel Model.
Not flat key→value pairs. Full operational traces:

  English instruction
  ↓
  canonical SCL state trace
  ↓
  tool calls
  ↓
  observations
  ↓
  error states
  ↓
  repairs
  ↓
  verified final state

The model learns to predict valid state transitions, not generate prose.

Training task types:
  predict_next_state         — given trace prefix, predict next SCL record
  detect_invalid_transition  — identify which record violates grammar
  complete_missing_edge      — fill in dependency the trace needs
  compress_trace             — multi-record → single summary record
  expand_trace               — summary → full multi-record trace
  repair_failed_trace        — given error, propose fix records
  rank_next_action           — given state, rank possible actions
  select_tool                — given task, pick correct syscall
  diagnose_error             — given failure observation, identify cause
  propose_repair             — given diagnosis, emit repair records
"""

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..scl.ontology import (
    EntityType, RelationType, TransitionType, CKMTask,
    StateTrace,
)


# ---------------------------------------------------------------------------
# Trace templates — the operational patterns of the OS
# ---------------------------------------------------------------------------

@dataclass
class TraceTemplate:
    """A template for generating training traces."""
    name: str
    task_type: CKMTask
    description: str
    steps: list[dict]  # Each step: {agent, verb, target, scope_template}


# === BOOT TRACES ===

BOOT_TRACES = [
    TraceTemplate(
        name="cold_boot_hardware_detect",
        task_type=CKMTask.PREDICT_NEXT_STATE,
        description="Cold boot sequence: mount → detect → configure → start",
        steps=[
            {"agent": "init", "verb": "boot", "target": "system", "scope": {"phase": "cold_start", "pid": "1"}},
            {"agent": "init", "verb": "execute", "target": "filesystem", "scope": {"action": "mount_virtual", "targets": "proc,sys,dev"}},
            {"agent": "init", "verb": "detect", "target": "hardware", "scope": {"subsystem": "{subsystem}", "result": "{hw_result}"}},
            {"agent": "init", "verb": "select", "target": "config", "scope": {"threads": "{threads}", "gpu_layers": "{gpu_layers}", "ctx": "{ctx}"}},
            {"agent": "init", "verb": "spawn", "target": "backend", "scope": {"type": "{backend}", "port": "8080"}},
            {"agent": "init", "verb": "verify", "target": "backend", "scope": {"status": "healthy", "latency_ms": "{latency}"}},
            {"agent": "init", "verb": "spawn", "target": "daemon", "scope": {"port": "11411", "mode": "pid1"}},
            {"agent": "verifier", "verb": "check", "target": "system", "scope": {"result": "pass", "boot_ms": "{boot_ms}"}},
        ],
    ),
    TraceTemplate(
        name="warm_boot_cached",
        task_type=CKMTask.PREDICT_NEXT_STATE,
        description="Warm boot: cache hit → skip detection → fast start",
        steps=[
            {"agent": "init", "verb": "boot", "target": "system", "scope": {"phase": "warm_start", "pid": "1"}},
            {"agent": "init", "verb": "read", "target": "cache", "scope": {"hw_fingerprint": "{fp}", "hit": "true"}},
            {"agent": "init", "verb": "apply", "target": "config", "scope": {"source": "cache", "threads": "{threads}", "gpu_layers": "{gpu_layers}"}},
            {"agent": "init", "verb": "spawn", "target": "backend", "scope": {"type": "{backend}", "port": "8080", "preloaded": "true"}},
            {"agent": "verifier", "verb": "check", "target": "system", "scope": {"result": "pass", "boot_ms": "{boot_ms}", "speedup": "{speedup}x"}},
        ],
    ),
    TraceTemplate(
        name="boot_failure_recovery",
        task_type=CKMTask.REPAIR_FAILED_TRACE,
        description="Boot fails → diagnose → fallback → recover",
        steps=[
            {"agent": "init", "verb": "boot", "target": "system", "scope": {"phase": "cold_start"}},
            {"agent": "init", "verb": "spawn", "target": "backend", "scope": {"type": "{backend}", "port": "8080"}},
            {"agent": "init", "verb": "fail", "target": "backend", "scope": {"error": "{error}", "code": "{code}"}},
            {"agent": "init", "verb": "diagnose", "target": "error", "scope": {"cause": "{cause}", "category": "backend_unavailable"}},
            {"agent": "init", "verb": "repair", "target": "config", "scope": {"action": "fallback", "new_backend": "{fallback}"}},
            {"agent": "init", "verb": "spawn", "target": "backend", "scope": {"type": "{fallback}", "port": "8080"}},
            {"agent": "verifier", "verb": "check", "target": "system", "scope": {"result": "pass", "degraded": "true"}},
        ],
    ),
]

# === ROUTING TRACES ===

ROUTING_TRACES = [
    TraceTemplate(
        name="simple_route",
        task_type=CKMTask.PREDICT_NEXT_STATE,
        description="Request arrives → classify → route → respond",
        steps=[
            {"agent": "daemon", "verb": "receive", "target": "message", "scope": {"tokens": "{tokens}", "format": "{format}"}},
            {"agent": "router", "verb": "inspect", "target": "task", "scope": {"category": "{category}", "complexity": "{complexity}"}},
            {"agent": "router", "verb": "select", "target": "tier", "scope": {"tier": "{tier}", "model": "{model}", "confidence": "{confidence}"}},
            {"agent": "backend", "verb": "execute", "target": "model", "scope": {"model": "{model}", "tokens_out": "{tokens_out}", "latency_ms": "{latency}"}},
            {"agent": "daemon", "verb": "emit", "target": "message", "scope": {"status": "complete", "ttft_ms": "{ttft}"}},
        ],
    ),
    TraceTemplate(
        name="route_with_challenge",
        task_type=CKMTask.PREDICT_NEXT_STATE,
        description="Route → generate → challenge disagrees → escalate",
        steps=[
            {"agent": "router", "verb": "select", "target": "tier", "scope": {"tier": "{tier}", "model": "{model}"}},
            {"agent": "backend", "verb": "execute", "target": "model", "scope": {"model": "{model}", "tokens_out": "{tokens_out}"}},
            {"agent": "challenger", "verb": "verify", "target": "claim", "scope": {"challenger_model": "{challenger}", "agreement": "{agreement}"}},
            {"agent": "challenger", "verb": "detect", "target": "error", "scope": {"level": "disagree", "confidence": "{conf}"}},
            {"agent": "router", "verb": "escalate", "target": "tier", "scope": {"from": "{tier}", "to": "{higher_tier}", "reason": "challenge_disagree"}},
            {"agent": "backend", "verb": "execute", "target": "model", "scope": {"model": "{better_model}", "tokens_out": "{tokens_out2}"}},
            {"agent": "verifier", "verb": "check", "target": "claim", "scope": {"result": "pass", "method": "cross_family"}},
        ],
    ),
    TraceTemplate(
        name="route_swarm_consensus",
        task_type=CKMTask.PREDICT_NEXT_STATE,
        description="Challenger fails → swarm vote → consensus",
        steps=[
            {"agent": "challenger", "verb": "detect", "target": "error", "scope": {"level": "strong_disagree"}},
            {"agent": "swarm", "verb": "spawn", "target": "process", "scope": {"size": "{swarm_size}", "families": "{families}"}},
            {"agent": "swarm", "verb": "execute", "target": "model", "scope": {"model": "{m1}", "cluster": "0"}},
            {"agent": "swarm", "verb": "execute", "target": "model", "scope": {"model": "{m2}", "cluster": "0"}},
            {"agent": "swarm", "verb": "execute", "target": "model", "scope": {"model": "{m3}", "cluster": "1"}},
            {"agent": "swarm", "verb": "observe", "target": "graph", "scope": {"clusters": "2", "majority": "cluster_0", "weight": "{weight}"}},
            {"agent": "swarm", "verb": "select", "target": "claim", "scope": {"source": "cluster_0", "confidence": "{conf}"}},
        ],
    ),
]

# === DEBUGGING TRACES ===

DEBUG_TRACES = [
    TraceTemplate(
        name="inspect_dependency_graph",
        task_type=CKMTask.IDENTIFY_HIDDEN_COUPLING,
        description="Agent inspects repo → finds hidden coupling",
        steps=[
            {"agent": "agent", "verb": "inspect", "target": "module", "scope": {"path": "{path}"}},
            {"agent": "agent", "verb": "enumerate", "target": "symbol", "scope": {"count": "{count}", "types": "function,class"}},
            {"agent": "agent", "verb": "extract", "target": "dependency", "scope": {"edges": "{edges}", "imports": "{imports}"}},
            {"agent": "agent", "verb": "detect", "target": "edge", "scope": {"type": "implicit", "from": "{from_sym}", "to": "{to_sym}"}},
            {"agent": "agent", "verb": "claim", "target": "evidence", "scope": {"type": "hidden_coupling", "severity": "{severity}"}},
            {"agent": "verifier", "verb": "requires", "target": "proof", "scope": {"needs": "file_span,test_case"}},
        ],
    ),
    TraceTemplate(
        name="diagnose_crash",
        task_type=CKMTask.DIAGNOSE_ERROR,
        description="Process crashes → diagnose → identify root cause",
        steps=[
            {"agent": "process", "verb": "fail", "target": "error", "scope": {"type": "{error_type}", "message": "{message}"}},
            {"agent": "agent", "verb": "inspect", "target": "error", "scope": {"traceback": "{traceback_summary}"}},
            {"agent": "agent", "verb": "read", "target": "file", "scope": {"path": "{file}", "line": "{line}"}},
            {"agent": "agent", "verb": "detect", "target": "edge", "scope": {"cause": "{root_cause}", "type": "{cause_type}"}},
            {"agent": "agent", "verb": "claim", "target": "evidence", "scope": {"diagnosis": "{diagnosis}", "confidence": "{conf}"}},
        ],
    ),
    TraceTemplate(
        name="repair_and_verify",
        task_type=CKMTask.PROPOSE_REPAIR,
        description="Error diagnosed → propose repair → apply → verify",
        steps=[
            {"agent": "agent", "verb": "observe", "target": "error", "scope": {"diagnosis": "{diagnosis}", "cause": "{cause}"}},
            {"agent": "agent", "verb": "propose", "target": "repair", "scope": {"action": "{repair_action}", "target_file": "{file}"}},
            {"agent": "agent", "verb": "execute", "target": "tool", "scope": {"tool": "edit", "file": "{file}", "change": "{change}"}},
            {"agent": "agent", "verb": "execute", "target": "tool", "scope": {"tool": "test", "scope": "{test_scope}"}},
            {"agent": "verifier", "verb": "check", "target": "invariant", "scope": {"tests_pass": "true", "regression": "none"}},
            {"agent": "agent", "verb": "commit", "target": "checkpoint", "scope": {"message": "{commit_msg}"}},
        ],
    ),
]

# === GOSSIP / SYNC TRACES ===

GOSSIP_TRACES = [
    TraceTemplate(
        name="peer_sync_converge",
        task_type=CKMTask.PREDICT_NEXT_STATE,
        description="Two peers discover divergence → exchange deltas → converge",
        steps=[
            {"agent": "peer_a", "verb": "gossip", "target": "peer_b", "scope": {"fingerprint": "{fp_a}", "type": "ping"}},
            {"agent": "peer_b", "verb": "observe", "target": "peer_a", "scope": {"diverged": "true", "similarity": "{sim}"}},
            {"agent": "peer_b", "verb": "emit", "target": "peer_a", "scope": {"type": "push_delta", "deltas": "{delta_count}"}},
            {"agent": "peer_a", "verb": "apply", "target": "config", "scope": {"applied": "{applied}", "conflicts": "{conflicts}"}},
            {"agent": "peer_a", "verb": "emit", "target": "peer_b", "scope": {"type": "push_delta", "deltas": "{push_count}"}},
            {"agent": "verifier", "verb": "check", "target": "invariant", "scope": {"converged": "true", "fp_match": "true"}},
        ],
    ),
    TraceTemplate(
        name="sneakernet_import",
        task_type=CKMTask.PREDICT_NEXT_STATE,
        description="Offline device imports SCL from file → applies → gains knowledge",
        steps=[
            {"agent": "init", "verb": "boot", "target": "system", "scope": {"phase": "cold_start", "network": "none"}},
            {"agent": "init", "verb": "read", "target": "file", "scope": {"path": "/mnt/cortex/gossip/peer_a.scl", "records": "{records}"}},
            {"agent": "init", "verb": "apply", "target": "config", "scope": {"source": "sneakernet", "deltas": "{deltas}"}},
            {"agent": "init", "verb": "detect", "target": "hardware", "scope": {"match": "true", "cached_config": "available"}},
            {"agent": "verifier", "verb": "check", "target": "system", "scope": {"result": "pass", "source": "peer_a"}},
        ],
    ),
]

# === POLICY MUTATION TRACES ===

POLICY_TRACES = [
    TraceTemplate(
        name="feedback_driven_demotion",
        task_type=CKMTask.PREDICT_NEXT_STATE,
        description="Accumulated failures → policy rewriter demotes tier",
        steps=[
            {"agent": "feedback", "verb": "observe", "target": "model", "scope": {"model": "{model}", "accuracy": "{accuracy}", "count": "{count}"}},
            {"agent": "policy_rewriter", "verb": "inspect", "target": "evidence", "scope": {"threshold": "0.5", "below": "true"}},
            {"agent": "policy_rewriter", "verb": "propose", "target": "policy", "scope": {"action": "demote", "tier": "{tier}", "model": "{model}"}},
            {"agent": "policy_rewriter", "verb": "mutate", "target": "config", "scope": {"tier_{tier}_penalty": "{penalty}", "model_blocked": "{model}"}},
            {"agent": "verifier", "verb": "check", "target": "invariant", "scope": {"policy_valid": "true", "no_empty_tiers": "true"}},
        ],
    ),
    TraceTemplate(
        name="self_modification_rule",
        task_type=CKMTask.PREDICT_NEXT_STATE,
        description="RuleEngine detects pattern → rewrites own rules",
        steps=[
            {"agent": "rule_engine", "verb": "observe", "target": "task", "scope": {"pattern": "{pattern}", "frequency": "{freq}"}},
            {"agent": "rule_engine", "verb": "detect", "target": "invariant", "scope": {"name": "rule_efficiency", "violated": "true"}},
            {"agent": "rule_engine", "verb": "propose", "target": "repair", "scope": {"action": "rewrite_rule", "rule": "{rule_name}"}},
            {"agent": "rule_engine", "verb": "mutate", "target": "config", "scope": {"rule_{rule_name}_threshold": "{new_threshold}"}},
            {"agent": "verifier", "verb": "check", "target": "invariant", "scope": {"rule_fires_correctly": "true"}},
        ],
    ),
]

# === TOOL USE TRACES ===

TOOL_TRACES = [
    TraceTemplate(
        name="file_edit_with_verification",
        task_type=CKMTask.SELECT_TOOL,
        description="Agent selects edit tool → applies change → runs test",
        steps=[
            {"agent": "agent", "verb": "plan", "target": "task", "scope": {"goal": "{goal}", "strategy": "edit_then_test"}},
            {"agent": "agent", "verb": "select", "target": "tool", "scope": {"tool": "read_file", "permission": "ring_1"}},
            {"agent": "agent", "verb": "execute", "target": "tool", "scope": {"tool": "read_file", "path": "{path}", "result": "success"}},
            {"agent": "agent", "verb": "select", "target": "tool", "scope": {"tool": "edit_file", "permission": "ring_2"}},
            {"agent": "agent", "verb": "execute", "target": "tool", "scope": {"tool": "edit_file", "path": "{path}", "change": "{change}"}},
            {"agent": "agent", "verb": "select", "target": "tool", "scope": {"tool": "run_test", "permission": "ring_1"}},
            {"agent": "agent", "verb": "execute", "target": "tool", "scope": {"tool": "run_test", "scope": "{test}", "result": "{test_result}"}},
            {"agent": "verifier", "verb": "check", "target": "invariant", "scope": {"tests_pass": "{test_result}", "permission_valid": "true"}},
        ],
    ),
]

ALL_TEMPLATES = BOOT_TRACES + ROUTING_TRACES + DEBUG_TRACES + GOSSIP_TRACES + POLICY_TRACES + TOOL_TRACES


# ---------------------------------------------------------------------------
# Trace instantiation — fill templates with realistic values
# ---------------------------------------------------------------------------

# Value pools for template variables
VALUE_POOLS = {
    "subsystem": ["cpu", "gpu", "memory", "storage", "network"],
    "hw_result": ["apple_m1_pro_10c_16gb", "ryzen_9_32c_nvidia_24gb", "celeron_4c_4gb_nogpu", "jetson_orin_12c_8gb"],
    "threads": ["4", "8", "9", "12", "16", "24"],
    "gpu_layers": ["0", "16", "32", "999"],
    "ctx": ["2048", "4096", "8192", "16384"],
    "backend": ["llama_cpp", "ollama", "vllm"],
    "fallback": ["ollama", "llama_cpp"],
    "latency": ["50", "120", "250", "500", "1200"],
    "boot_ms": ["800", "1500", "2500", "4000", "6000"],
    "speedup": ["2.1", "3.5", "5.0"],
    "fp": ["⢚⡮⢗⣷", "⡜⠅⡲⣖", "⠘⣏⢁⡰", "⣯⠿⡇⢁"],
    "tokens": ["15", "50", "200", "500", "1500", "3000"],
    "tokens_out": ["50", "150", "500", "1000", "2000"],
    "format": ["openai", "ollama", "raw"],
    "category": ["code", "chat", "math", "tool", "analysis", "system"],
    "complexity": ["0.05", "0.2", "0.4", "0.6", "0.8", "0.95"],
    "tier": ["L0", "L1", "L2", "L3", "L4", "L5"],
    "higher_tier": ["L3", "L4", "L5"],
    "model": ["qwen3:4b", "qwen3:8b", "llama3.2:3b", "granite3.3:8b", "phi4:14b"],
    "better_model": ["qwen3:8b", "phi4:14b", "llama3.3:70b"],
    "challenger": ["granite3.3:8b", "gemma3:12b", "phi4:14b"],
    "confidence": ["0.35", "0.55", "0.72", "0.85", "0.93"],
    "conf": ["0.4", "0.6", "0.75", "0.88", "0.95"],
    "agreement": ["strong_agree", "weak_agree", "weak_disagree", "strong_disagree"],
    "swarm_size": ["3", "5", "7"],
    "families": ["2", "3", "4"],
    "m1": ["qwen3:8b", "llama3.2:3b"],
    "m2": ["granite3.3:8b", "gemma3:12b"],
    "m3": ["phi4:14b", "mistral:7b"],
    "weight": ["6.5", "8.0", "9.5"],
    "ttft": ["80", "150", "300", "600", "1200"],
    "error": ["connection_refused", "oom_killed", "timeout", "model_not_found", "gpu_error"],
    "error_type": ["ConnectionError", "MemoryError", "TimeoutError", "FileNotFoundError", "RuntimeError"],
    "code": ["ECONNREFUSED", "ENOMEM", "ETIMEDOUT", "ENOENT", "EPERM"],
    "cause": ["port_not_listening", "model_too_large", "backend_hung", "missing_dependency", "permission_denied"],
    "cause_type": ["resource", "config", "dependency", "permission", "hardware"],
    "message": ["Connection refused on port 8080", "Cannot allocate memory", "Request timed out after 30s"],
    "path": ["src/router.py", "src/daemon.py", "src/cortex.py", "src/model_manager.py", "src/tiers.py"],
    "file": ["src/router.py", "src/daemon.py", "src/gossip_transport.py", "src/boot_telemetry.py"],
    "count": ["12", "25", "48", "73", "150"],
    "edges": ["15", "32", "67", "120"],
    "imports": ["5", "12", "20", "35"],
    "from_sym": ["route_model", "ModelManager", "DaemonServer._handle", "Cortex.process"],
    "to_sym": ["config.MODEL_TIERS", "TIER_SPECS", "memory.get_thread", "policy.check_access"],
    "severity": ["low", "medium", "high", "critical"],
    "traceback_summary": ["router.py:L45_route_model", "daemon.py:L200_handle_request", "gossip.py:L150_sync"],
    "line": ["45", "120", "200", "350"],
    "root_cause": ["unhandled_none", "missing_import", "race_condition", "stale_cache"],
    "diagnosis": ["null_reference_on_optional_field", "import_removed_in_refactor", "concurrent_write_no_lock"],
    "repair_action": ["add_null_check", "restore_import", "add_lock", "add_retry", "catch_exception"],
    "change": ["add_guard_clause", "wrap_try_except", "add_timeout", "fix_import"],
    "test_scope": ["tests/test_router.py", "tests/test_daemon.py", "tests/"],
    "test_result": ["pass", "pass", "pass", "fail"],
    "commit_msg": ["fix: handle null peer", "fix: restore missing import", "fix: add connection timeout"],
    "sim": ["0.3", "0.5", "0.7", "0.85"],
    "delta_count": ["3", "5", "8", "12"],
    "applied": ["3", "5", "8"],
    "conflicts": ["0", "0", "1", "2"],
    "push_count": ["2", "4", "6"],
    "records": ["5", "10", "15", "20"],
    "deltas": ["3", "5", "8"],
    "accuracy": ["0.35", "0.42", "0.55", "0.75", "0.88", "0.95"],
    "penalty": ["0.3", "0.5", "0.7"],
    "pattern": ["high_complexity_misroute", "repeated_challenge_fail", "slow_ttft_pattern"],
    "freq": ["5", "10", "20", "50"],
    "rule_name": ["slow_boot_increase_threads", "low_accuracy_demote", "high_latency_escalate"],
    "new_threshold": ["3000", "5000", "0.6", "0.4"],
    "goal": ["fix_crash", "optimize_boot", "add_feature", "refactor_module"],
}


def instantiate_trace(template: TraceTemplate) -> list[str]:
    """Fill a trace template with random values from pools."""
    # Choose consistent values for this trace instance
    chosen = {}
    lines = []

    for step in template.steps:
        agent = step["agent"]
        verb = step["verb"]
        scope = {}

        for key, value_template in step["scope"].items():
            if value_template.startswith("{") and value_template.endswith("}"):
                var_name = value_template[1:-1]
                if var_name not in chosen:
                    pool = VALUE_POOLS.get(var_name, ["unknown"])
                    chosen[var_name] = random.choice(pool)
                scope[key] = chosen[var_name]
            else:
                scope[key] = value_template

        scope_str = ", ".join(f"{k}: {v}" for k, v in scope.items())
        lines.append(f"@{agent} → {verb} [{scope_str}]")

    return lines


# ---------------------------------------------------------------------------
# Training pair generation from traces
# ---------------------------------------------------------------------------

@dataclass
class TracePair:
    """A training pair derived from a state trace."""
    task_type: str          # CKMTask value
    input_scl: str          # Context (prefix of trace or error state)
    output_scl: str         # Target (next record, repair, etc.)
    full_trace: str         # Complete trace for reference
    template_name: str
    quality: float = 0.9

    def to_jsonl(self) -> str:
        return json.dumps({
            "input": self.input_scl,
            "output": self.output_scl,
            "source": f"trace_{self.task_type}",
            "quality": self.quality,
            "task_type": self.task_type,
            "template": self.template_name,
        })


def generate_predict_next_state(template: TraceTemplate, n: int = 10) -> list[TracePair]:
    """Generate 'predict next state' pairs from a trace.

    For each position in the trace, input = prefix, output = next record.
    This is the core training signal: given state history, predict transition.
    """
    pairs = []
    for _ in range(n):
        trace = instantiate_trace(template)
        full = "\n".join(trace)

        # For each step after the first, create a pair
        for i in range(1, len(trace)):
            prefix = "\n".join(trace[:i])
            target = trace[i]
            pairs.append(TracePair(
                task_type=CKMTask.PREDICT_NEXT_STATE.value,
                input_scl=prefix,
                output_scl=target,
                full_trace=full,
                template_name=template.name,
                quality=0.9,
            ))
    return pairs


def generate_compress_trace(template: TraceTemplate, n: int = 10) -> list[TracePair]:
    """Generate 'compress trace' pairs.

    Input: full multi-record trace
    Output: single summary record
    """
    pairs = []
    for _ in range(n):
        trace = instantiate_trace(template)
        full = "\n".join(trace)

        # Summary: take the final verifier check or last agent action
        last_record = trace[-1]
        # Create compressed summary from template metadata
        summary = f"@trace → summarize [name: {template.name}, steps: {len(trace)}, result: complete]"

        pairs.append(TracePair(
            task_type=CKMTask.COMPRESS_TRACE.value,
            input_scl=full,
            output_scl=summary,
            full_trace=full,
            template_name=template.name,
            quality=0.85,
        ))
    return pairs


def generate_repair_trace(template: TraceTemplate, n: int = 10) -> list[TracePair]:
    """Generate 'repair failed trace' pairs.

    Input: trace with failure
    Output: repair records
    """
    pairs = []

    # Only use templates that have a failure step
    has_fail = any(s["verb"] == "fail" for s in template.steps)
    if not has_fail:
        return pairs

    for _ in range(n):
        trace = instantiate_trace(template)

        # Find the failure point
        fail_idx = next(i for i, line in enumerate(trace) if "→ fail" in line)

        # Input: trace up to and including failure
        broken_trace = "\n".join(trace[:fail_idx + 1])

        # Output: the repair steps that follow
        repair_steps = "\n".join(trace[fail_idx + 1:])

        pairs.append(TracePair(
            task_type=CKMTask.REPAIR_FAILED_TRACE.value,
            input_scl=broken_trace,
            output_scl=repair_steps,
            full_trace="\n".join(trace),
            template_name=template.name,
            quality=0.95,  # Repair traces are high-value
        ))
    return pairs


def generate_detect_invalid(template: TraceTemplate, n: int = 10) -> list[TracePair]:
    """Generate 'detect invalid transition' pairs.

    Input: trace with one step replaced by an invalid transition
    Output: identification of the invalid step
    """
    pairs = []
    for _ in range(n):
        trace = instantiate_trace(template)

        # Pick a random step to corrupt
        if len(trace) < 3:
            continue
        corrupt_idx = random.randint(1, len(trace) - 2)

        # Generate an invalid transition (wrong verb for context)
        invalid_verbs = ["sleep", "kill", "rollback", "deny"]
        invalid_verb = random.choice(invalid_verbs)
        original = trace[corrupt_idx]
        # Replace the verb
        parts = original.split(" → ")
        if len(parts) >= 2:
            rest = parts[1].split(" [")
            corrupted = f"{parts[0]} → {invalid_verb} [{rest[1]}" if len(rest) > 1 else original
        else:
            continue

        corrupt_trace = trace.copy()
        corrupt_trace[corrupt_idx] = corrupted

        input_text = "\n".join(corrupt_trace)
        output_text = f"@verifier → detect [invalid_step: {corrupt_idx}, expected: {original.split(' → ')[1].split(' [')[0]}, got: {invalid_verb}]"

        pairs.append(TracePair(
            task_type=CKMTask.DETECT_INVALID_TRANSITION.value,
            input_scl=input_text,
            output_scl=output_text,
            full_trace="\n".join(trace),
            template_name=template.name,
            quality=0.9,
        ))
    return pairs


# ---------------------------------------------------------------------------
# Master generator
# ---------------------------------------------------------------------------

def generate_all_trace_pairs(
    instances_per_template: int = 20,
) -> list[TracePair]:
    """Generate the complete training corpus from all templates."""
    all_pairs = []

    for template in ALL_TEMPLATES:
        # Predict next state (main training signal)
        all_pairs.extend(generate_predict_next_state(template, instances_per_template))

        # Compress trace
        all_pairs.extend(generate_compress_trace(template, instances_per_template // 2))

        # Repair (only for failure templates)
        all_pairs.extend(generate_repair_trace(template, instances_per_template))

        # Detect invalid
        all_pairs.extend(generate_detect_invalid(template, instances_per_template // 2))

    random.shuffle(all_pairs)
    return all_pairs


def save_trace_corpus(output_path: str, instances_per_template: int = 20) -> dict:
    """Generate and save the full trace corpus."""
    pairs = generate_all_trace_pairs(instances_per_template)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        for pair in pairs:
            f.write(pair.to_jsonl() + "\n")

    # Stats
    task_counts = {}
    template_counts = {}
    for p in pairs:
        task_counts[p.task_type] = task_counts.get(p.task_type, 0) + 1
        template_counts[p.template_name] = template_counts.get(p.template_name, 0) + 1

    stats = {
        "total_pairs": len(pairs),
        "task_types": task_counts,
        "templates": template_counts,
        "avg_quality": sum(p.quality for p in pairs) / len(pairs) if pairs else 0,
        "output_path": str(path),
    }
    return stats
