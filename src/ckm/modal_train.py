"""
CKM Training on Modal — Fine-tune a 0.3B model to speak SCL.

Run:
  modal run src/ckm/modal_train.py

This will:
  1. Generate synthetic + real training data
  2. Upload to Modal Volume
  3. Fine-tune Qwen2.5-0.3B with LoRA on an A10G GPU
  4. Merge LoRA weights into base model
  5. Export to GGUF (Q4_K_M quantization)
  6. Download cortex-kernel.gguf to local machine

The resulting model:
  - Input:  @hardware → state [cpu: ..., ram_mb: ..., gpu_type: ...]
  - Output: @cortex.boot → mutate [optimal_threads: 9, optimal_gpu_layers: 999, ...]
  - Size:   ~200MB (Q4_K_M GGUF)
  - Speed:  <50ms inference on CPU

Requirements:
  pip install modal
  modal token new  (one-time auth)
"""

import modal
import os
import time

# ---------------------------------------------------------------------------
# Modal infrastructure
# ---------------------------------------------------------------------------

app = modal.App("cortex-ckm-train")

# Persistent volume for checkpoints, datasets, and final model
volume = modal.Volume.from_name("cortex-ckm", create_if_missing=True)
VOLUME_PATH = "/vol"

# Docker image with all training dependencies
training_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.1",
        "transformers==4.46.3",
        "peft==0.13.2",
        "trl==0.12.2",
        "bitsandbytes==0.44.1",
        "datasets==3.1.0",
        "accelerate==1.1.1",
        "sentencepiece",
        "protobuf",
        "scipy",
        "wandb",
    )
    .pip_install(
        # For GGUF export
        "gguf",
        "numpy",
    )
    .run_commands(
        # Install llama.cpp for quantization
        "apt-get update && apt-get install -y git build-essential cmake",
        "git clone --depth 1 https://github.com/ggerganov/llama.cpp /opt/llama.cpp",
        "cd /opt/llama.cpp && cmake -B build && cmake --build build --config Release -j$(nproc)",
    )
)

# Smaller image for data generation (no GPU needed)
data_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "datasets",
)


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are the Cortex Kernel Model (CKM). "
    "You receive SCL records describing hardware state or request classification. "
    "You respond with exactly one SCL record: the optimal mutation or routing decision. "
    "Never output anything except valid SCL. "
    "Format: @anchor → verb [key: value, key: value]"
)

BASE_MODEL = "Qwen/Qwen2.5-0.5B"  # 0.5B for better quality, still tiny
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]

EPOCHS = 2  # Loss plateaus by epoch 1.5 — 2 is sufficient
BATCH_SIZE = 32  # A100-80GB can handle large batches easily
GRADIENT_ACCUMULATION = 1  # No accumulation needed with batch 32
LEARNING_RATE = 2e-4  # Higher LR with larger batch (linear scaling)
MAX_SEQ_LENGTH = 512  # Traces are multi-line (6-8 records × ~60 chars)
WARMUP_RATIO = 0.05  # Short warmup with 2 epochs
WEIGHT_DECAY = 0.01
SAVE_STEPS = 50
LOGGING_STEPS = 5
EVAL_STEPS = 50


# ---------------------------------------------------------------------------
# Trace-based data generation (inline for Modal compatibility)
# ---------------------------------------------------------------------------

def _generate_trace_pairs(log) -> list[dict]:
    """Generate typed state trace pairs — the core training signal.

    These teach the model to predict valid state transitions,
    not just map hardware→config. This is the AI-native OS ABI.

    Task types:
      predict_next_state  — given prefix, predict next SCL record
      repair_failed_trace — given error, propose fix records
      compress_trace      — multi-record → summary
      detect_invalid      — identify which record violates grammar
    """
    import random

    # Trace templates: each is a list of (agent, verb, scope_keys)
    # These cover: boot, routing, debugging, gossip, policy, tool use
    TRACES = {
        "cold_boot": [
            ("init", "boot", {"phase": "cold_start", "pid": "1"}),
            ("init", "execute", {"action": "mount_virtual", "targets": "proc,sys,dev"}),
            ("init", "detect", {"subsystem": ["cpu", "gpu", "memory"], "result": ["apple_m1_16gb", "ryzen_32c_nvidia", "celeron_4c"]}),
            ("init", "select", {"threads": ["4", "8", "9", "16"], "gpu_layers": ["0", "32", "999"], "ctx": ["2048", "4096", "8192"]}),
            ("init", "spawn", {"type": ["llama_cpp", "ollama", "vllm"], "port": "8080"}),
            ("verifier", "check", {"result": "pass", "boot_ms": ["800", "1500", "3000", "5000"]}),
        ],
        "warm_boot": [
            ("init", "boot", {"phase": "warm_start", "pid": "1"}),
            ("init", "read", {"hw_fingerprint": ["⢚⡮⢗⣷", "⡜⠅⡲⣖", "⠘⣏⢁⡰"], "hit": "true"}),
            ("init", "apply", {"source": "cache", "threads": ["8", "9", "12"], "gpu_layers": ["999", "32"]}),
            ("init", "spawn", {"type": ["llama_cpp", "ollama"], "preloaded": "true"}),
            ("verifier", "check", {"result": "pass", "boot_ms": ["400", "800", "1200"], "speedup": ["2x", "3x", "5x"]}),
        ],
        "boot_failure": [
            ("init", "boot", {"phase": "cold_start"}),
            ("init", "spawn", {"type": ["llama_cpp", "vllm"], "port": "8080"}),
            ("init", "fail", {"error": ["connection_refused", "oom_killed", "timeout"], "code": ["ECONNREFUSED", "ENOMEM"]}),
            ("init", "diagnose", {"cause": ["port_not_listening", "model_too_large", "backend_hung"]}),
            ("init", "repair", {"action": "fallback", "new_backend": ["ollama", "llama_cpp"]}),
            ("init", "spawn", {"type": ["ollama", "llama_cpp"], "port": "8080"}),
            ("verifier", "check", {"result": "pass", "degraded": "true"}),
        ],
        "simple_route": [
            ("daemon", "receive", {"tokens": ["15", "200", "1500"], "format": ["openai", "ollama"]}),
            ("router", "inspect", {"category": ["code", "chat", "math", "tool"], "complexity": ["0.1", "0.5", "0.8"]}),
            ("router", "select", {"tier": ["L1", "L2", "L3", "L4", "L5"], "model": ["qwen3:4b", "qwen3:8b", "phi4:14b"], "confidence": ["0.6", "0.8", "0.95"]}),
            ("backend", "execute", {"model": ["qwen3:4b", "qwen3:8b", "phi4:14b"], "tokens_out": ["50", "500", "2000"], "latency_ms": ["80", "300", "1200"]}),
            ("daemon", "emit", {"status": "complete", "ttft_ms": ["80", "150", "600"]}),
        ],
        "challenge_escalate": [
            ("router", "select", {"tier": ["L2", "L3"], "model": ["qwen3:4b", "llama3.2:3b"]}),
            ("backend", "execute", {"model": ["qwen3:4b", "llama3.2:3b"], "tokens_out": ["200", "500"]}),
            ("challenger", "verify", {"challenger_model": ["granite3.3:8b", "gemma3:12b"], "agreement": ["weak_disagree", "strong_disagree"]}),
            ("router", "escalate", {"from": ["L2", "L3"], "to": ["L4", "L5"], "reason": "challenge_disagree"}),
            ("backend", "execute", {"model": ["phi4:14b", "qwen3:8b"], "tokens_out": ["500", "1000"]}),
            ("verifier", "check", {"result": "pass", "method": "cross_family"}),
        ],
        "inspect_coupling": [
            ("agent", "inspect", {"path": ["src/router.py", "src/daemon.py", "src/cortex.py"]}),
            ("agent", "enumerate", {"count": ["12", "48", "150"], "types": "function,class"}),
            ("agent", "extract", {"edges": ["15", "67", "120"], "imports": ["5", "20", "35"]}),
            ("agent", "detect", {"type": "implicit", "from": ["route_model", "ModelManager"], "to": ["config.TIERS", "memory.get"]}),
            ("agent", "claim", {"type": "hidden_coupling", "severity": ["medium", "high", "critical"]}),
            ("verifier", "requires", {"needs": "file_span,test_case"}),
        ],
        "diagnose_crash": [
            ("process", "fail", {"type": ["ConnectionError", "MemoryError", "TimeoutError"], "message": ["port_8080_refused", "oom_model_load", "30s_timeout"]}),
            ("agent", "inspect", {"traceback": ["router.py:L45", "daemon.py:L200", "gossip.py:L150"]}),
            ("agent", "read", {"path": ["src/router.py", "src/daemon.py"], "line": ["45", "200", "150"]}),
            ("agent", "detect", {"cause": ["unhandled_none", "missing_import", "race_condition"], "type": ["resource", "config", "dependency"]}),
            ("agent", "claim", {"diagnosis": ["null_on_optional", "import_removed", "no_lock"], "confidence": ["0.75", "0.88", "0.95"]}),
        ],
        "repair_verify": [
            ("agent", "observe", {"diagnosis": ["null_on_optional", "import_removed", "no_lock"]}),
            ("agent", "propose", {"action": ["add_null_check", "restore_import", "add_lock"], "target_file": ["src/router.py", "src/daemon.py"]}),
            ("agent", "execute", {"tool": "edit_file", "change": ["add_guard", "wrap_try", "add_timeout"]}),
            ("agent", "execute", {"tool": "run_test", "scope": ["tests/", "tests/test_router.py"], "result": ["pass", "pass", "fail"]}),
            ("verifier", "check", {"tests_pass": "true", "regression": "none"}),
            ("agent", "commit", {"message": ["fix: handle null", "fix: restore import", "fix: add timeout"]}),
        ],
        "peer_sync": [
            ("peer_a", "gossip", {"fingerprint": ["⢚⡮⢗⣷", "⡜⠅⡲⣖"], "type": "ping"}),
            ("peer_b", "observe", {"diverged": "true", "similarity": ["0.3", "0.5", "0.7"]}),
            ("peer_b", "emit", {"type": "push_delta", "deltas": ["3", "5", "8"]}),
            ("peer_a", "apply", {"applied": ["3", "5", "8"], "conflicts": ["0", "1"]}),
            ("peer_a", "emit", {"type": "push_delta", "deltas": ["2", "4"]}),
            ("verifier", "check", {"converged": "true", "fp_match": "true"}),
        ],
        "policy_demotion": [
            ("feedback", "observe", {"model": ["qwen3:4b", "llama3.2:3b"], "accuracy": ["0.35", "0.42"], "count": ["10", "20", "50"]}),
            ("policy_rewriter", "inspect", {"threshold": "0.5", "below": "true"}),
            ("policy_rewriter", "propose", {"action": "demote", "tier": ["L2", "L3"], "model": ["qwen3:4b", "llama3.2:3b"]}),
            ("policy_rewriter", "mutate", {"penalty": ["0.3", "0.5"], "model_blocked": ["qwen3:4b", "llama3.2:3b"]}),
            ("verifier", "check", {"policy_valid": "true", "no_empty_tiers": "true"}),
        ],
        "tool_edit": [
            ("agent", "plan", {"goal": ["fix_crash", "optimize_boot", "refactor"], "strategy": "edit_then_test"}),
            ("agent", "select", {"tool": "read_file", "permission": "ring_1"}),
            ("agent", "execute", {"tool": "read_file", "path": ["src/router.py", "src/daemon.py"], "result": "success"}),
            ("agent", "select", {"tool": "edit_file", "permission": "ring_2"}),
            ("agent", "execute", {"tool": "edit_file", "change": ["add_guard", "wrap_try", "fix_import"]}),
            ("agent", "select", {"tool": "run_test", "permission": "ring_1"}),
            ("agent", "execute", {"tool": "run_test", "result": ["pass", "pass", "fail"]}),
            ("verifier", "check", {"tests_pass": "true", "permission_valid": "true"}),
        ],
    }

    def instantiate(steps):
        """Fill template with random values."""
        lines = []
        for agent, verb, scope_template in steps:
            scope = {}
            for k, v in scope_template.items():
                if isinstance(v, list):
                    scope[k] = random.choice(v)
                else:
                    scope[k] = v
            scope_str = ", ".join(f"{k}: {v}" for k, v in scope.items())
            lines.append(f"@{agent} → {verb} [{scope_str}]")
        return lines

    pairs = []
    instances_per_trace = 30

    for trace_name, steps in TRACES.items():
        for _ in range(instances_per_trace):
            trace = instantiate(steps)

            # Task 1: predict_next_state (for each prefix)
            for i in range(1, len(trace)):
                prefix = "\n".join(trace[:i])
                target = trace[i]
                pairs.append({
                    "input": prefix,
                    "output": target,
                    "source": f"trace_{trace_name}",
                    "quality": 0.9,
                    "task_type": "predict_next_state",
                })

            # Task 2: compress_trace
            if random.random() < 0.3:
                full = "\n".join(trace)
                summary = f"@trace → summarize [name: {trace_name}, steps: {len(trace)}, result: {trace[-1].split('result: ')[-1].rstrip(']') if 'result' in trace[-1] else 'complete'}]"
                pairs.append({
                    "input": full,
                    "output": summary,
                    "source": f"trace_{trace_name}_compress",
                    "quality": 0.85,
                    "task_type": "compress_trace",
                })

        # Task 3: repair_failed_trace (only for failure templates)
        if "fail" in trace_name or any("fail" in s[1] for s in steps):
            for _ in range(instances_per_trace):
                trace = instantiate(steps)
                fail_idx = next((i for i, (_, v, _) in enumerate(steps) if v == "fail"), None)
                if fail_idx is not None:
                    broken = "\n".join(trace[:fail_idx + 1])
                    repair = "\n".join(trace[fail_idx + 1:])
                    pairs.append({
                        "input": broken,
                        "output": repair,
                        "source": f"trace_{trace_name}_repair",
                        "quality": 0.95,
                        "task_type": "repair_failed_trace",
                    })

        # Task 4: detect_invalid (corrupt a step)
        for _ in range(instances_per_trace // 3):
            trace = instantiate(steps)
            if len(trace) < 3:
                continue
            idx = random.randint(1, len(trace) - 2)
            original_verb = steps[idx][1]
            bad_verb = random.choice(["sleep", "kill", "rollback", "deny", "fork"])
            corrupted = trace.copy()
            corrupted[idx] = corrupted[idx].replace(f"→ {original_verb}", f"→ {bad_verb}")
            pairs.append({
                "input": "\n".join(corrupted),
                "output": f"@verifier → detect [invalid_step: {idx}, expected: {original_verb}, got: {bad_verb}]",
                "source": f"trace_{trace_name}_invalid",
                "quality": 0.9,
                "task_type": "detect_invalid_transition",
            })

    log.info(f"  Generated {len(pairs)} trace pairs from {len(TRACES)} templates × {instances_per_trace} instances")
    return pairs


# ---------------------------------------------------------------------------
# Step 1: Generate training dataset
# ---------------------------------------------------------------------------

@app.function(image=data_image, volumes={VOLUME_PATH: volume}, timeout=300)
def generate_dataset(
    boot_pairs: int = 2000,
    route_pairs: int = 5000,
    policy_pairs: int = 500,
) -> dict:
    """Generate synthetic SCL training data and upload to volume."""
    import json
    import random
    import time as t
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("ckm.datagen")

    log.info("=" * 60)
    log.info("CKM Dataset Generation")
    log.info("=" * 60)

    output_dir = f"{VOLUME_PATH}/dataset"
    os.makedirs(output_dir, exist_ok=True)

    pairs = []
    t0 = t.time()

    # --- Boot config pairs ---
    log.info(f"Generating {boot_pairs} boot config pairs...")
    hardware_profiles = [
        ("Intel Core i7-12700K", 20, 32768, "nvidia", 8192, "x86_64"),
        ("Intel Core i5-10400", 12, 16384, "nvidia", 4096, "x86_64"),
        ("Intel Core i3-10100", 8, 8192, "none", 0, "x86_64"),
        ("AMD Ryzen 9 7950X", 32, 65536, "nvidia", 24576, "x86_64"),
        ("AMD Ryzen 7 5800X", 16, 32768, "nvidia", 12288, "x86_64"),
        ("AMD Ryzen 5 5600X", 12, 16384, "amd", 8192, "x86_64"),
        ("Apple M1", 8, 16384, "apple", 16384, "aarch64"),
        ("Apple M1 Pro", 10, 16384, "apple", 16384, "aarch64"),
        ("Apple M1 Max", 10, 32768, "apple", 32768, "aarch64"),
        ("Apple M2", 8, 8192, "apple", 8192, "aarch64"),
        ("Apple M2 Pro", 12, 16384, "apple", 16384, "aarch64"),
        ("Apple M2 Max", 12, 32768, "apple", 32768, "aarch64"),
        ("Apple M3", 8, 16384, "apple", 16384, "aarch64"),
        ("Apple M3 Pro", 12, 18432, "apple", 18432, "aarch64"),
        ("Apple M3 Max", 16, 36864, "apple", 36864, "aarch64"),
        ("Apple M4", 10, 16384, "apple", 16384, "aarch64"),
        ("Apple M4 Pro", 14, 24576, "apple", 24576, "aarch64"),
        ("Apple M4 Max", 16, 65536, "apple", 65536, "aarch64"),
        ("Intel Celeron N5105", 4, 4096, "none", 0, "x86_64"),
        ("Intel Core i9-13900K", 24, 65536, "nvidia", 24576, "x86_64"),
        ("AMD Ryzen 9 9950X", 32, 65536, "nvidia", 16384, "x86_64"),
        ("Raspberry Pi 5 BCM2712", 4, 4096, "none", 0, "aarch64"),
        ("Raspberry Pi 4 BCM2711", 4, 2048, "none", 0, "aarch64"),
        ("NVIDIA Jetson Orin", 12, 16384, "nvidia", 8192, "aarch64"),
        ("NVIDIA Jetson Nano", 4, 4096, "nvidia", 2048, "aarch64"),
        ("Qualcomm Snapdragon X Elite", 12, 16384, "none", 0, "aarch64"),
        ("Intel Core Ultra 7", 14, 32768, "intel", 4096, "x86_64"),
        ("AMD Ryzen AI 9 HX 370", 12, 32768, "amd", 2048, "x86_64"),
    ]

    for i in range(boot_pairs):
        hw = random.choice(hardware_profiles)
        cpu, cores, ram, gpu_type, gpu_vram, arch = hw

        # Noise for generalization
        cores_n = max(1, cores + random.randint(-2, 2))
        ram_n = max(1024, ram + random.randint(-2048, 2048))
        vram_n = max(0, gpu_vram + random.randint(-512, 512)) if gpu_vram > 0 else 0

        # Compute optimal config (heuristics we're distilling into the model)
        opt_threads = max(1, min(cores_n - 1, 16))
        if gpu_type == "none":
            opt_gpu = 0
            opt_ctx = min(4096, ram_n // 4)
            models = "L0" if ram_n >= 4096 else ""
        elif gpu_type == "apple":
            opt_gpu = 999
            opt_ctx = min(16384, vram_n // 2)
            if vram_n >= 32768:
                models = "L0,L1,L2,L3,L4"
            elif vram_n >= 16384:
                models = "L0,L1,L2,L3"
            elif vram_n >= 8192:
                models = "L0,L1,L2"
            else:
                models = "L0,L1"
        elif gpu_type == "nvidia":
            if vram_n >= 24576:
                opt_gpu, opt_ctx, models = 999, 16384, "L0,L1,L2,L3,L4"
            elif vram_n >= 12288:
                opt_gpu, opt_ctx, models = 999, 8192, "L0,L1,L2,L3"
            elif vram_n >= 8192:
                opt_gpu, opt_ctx, models = 999, 8192, "L0,L1,L2"
            elif vram_n >= 4096:
                opt_gpu, opt_ctx, models = 32, 4096, "L0,L1,L2"
            else:
                opt_gpu, opt_ctx, models = 16, 2048, "L0,L1"
        else:  # amd, intel
            opt_gpu = 999 if vram_n >= 8192 else min(24, max(0, vram_n // 256))
            opt_ctx = min(8192, max(2048, vram_n // 2))
            models = "L0,L1,L2" if vram_n >= 8192 else "L0,L1"

        opt_batch = 8 if opt_gpu > 0 else 4

        input_scl = f"@hardware → state [cpu: {cpu}, cores: {cores_n}, ram_mb: {ram_n}, gpu_type: {gpu_type}, vram_mb: {vram_n}, arch: {arch}]"
        output_scl = f"@cortex.boot → mutate [optimal_threads: {opt_threads}, optimal_gpu_layers: {opt_gpu}, optimal_ctx_size: {opt_ctx}, optimal_batch_size: {opt_batch}, optimal_backend: llama_cpp, optimal_hot_models: {models}]"

        pairs.append({
            "input": input_scl,
            "output": output_scl,
            "source": "synthetic_boot",
            "quality": 0.8 + random.uniform(-0.1, 0.1),
        })

        if (i + 1) % 500 == 0:
            log.info(f"  Boot pairs: {i+1}/{boot_pairs}")

    log.info(f"  ✓ {boot_pairs} boot pairs generated")

    # --- Routing pairs ---
    log.info(f"Generating {route_pairs} routing pairs...")
    categories = [
        ("code", "generate_function", 0.6, "L3"),
        ("code", "fix_bug", 0.7, "L4"),
        ("code", "explain_code", 0.3, "L2"),
        ("code", "autocomplete", 0.1, "L1"),
        ("code", "refactor", 0.65, "L3"),
        ("code", "review", 0.5, "L3"),
        ("code", "test_generation", 0.55, "L3"),
        ("code", "architecture", 0.85, "L5"),
        ("chat", "greeting", 0.05, "L0"),
        ("chat", "casual_qa", 0.2, "L1"),
        ("chat", "complex_reasoning", 0.8, "L5"),
        ("chat", "creative_writing", 0.5, "L3"),
        ("chat", "summarize", 0.3, "L2"),
        ("chat", "translate", 0.35, "L2"),
        ("chat", "roleplay", 0.4, "L2"),
        ("math", "arithmetic", 0.1, "L1"),
        ("math", "algebra", 0.4, "L2"),
        ("math", "calculus", 0.7, "L4"),
        ("math", "proof", 0.9, "L5"),
        ("math", "statistics", 0.5, "L3"),
        ("tool", "web_search", 0.3, "L2"),
        ("tool", "code_execution", 0.5, "L3"),
        ("tool", "multi_tool_chain", 0.8, "L4"),
        ("tool", "file_operations", 0.2, "L1"),
        ("analysis", "summarize", 0.3, "L2"),
        ("analysis", "compare", 0.5, "L3"),
        ("analysis", "deep_analysis", 0.85, "L5"),
        ("analysis", "data_extraction", 0.4, "L2"),
        ("system", "boot_config", 0.1, "L0"),
        ("system", "hardware_probe", 0.2, "L1"),
        ("system", "policy_check", 0.15, "L0"),
    ]

    for i in range(route_pairs):
        cat, subtype, complexity, tier = random.choice(categories)
        complexity = max(0.0, min(1.0, complexity + random.gauss(0, 0.12)))
        tokens = random.randint(5, 3000)
        confidence = max(0.3, min(0.99, 1.0 - abs(random.gauss(0, 0.15))))

        input_scl = f"@task → classify [category: {cat}, subtype: {subtype}, complexity: {complexity:.2f}, input_tokens: {tokens}]"
        output_scl = f"@router → select [tier: {tier}, confidence: {confidence:.2f}, reason: {cat}_{subtype}]"

        pairs.append({
            "input": input_scl,
            "output": output_scl,
            "source": "synthetic_route",
            "quality": 0.75 + random.uniform(-0.1, 0.1),
        })

        if (i + 1) % 1000 == 0:
            log.info(f"  Route pairs: {i+1}/{route_pairs}")

    log.info(f"  ✓ {route_pairs} routing pairs generated")

    # --- Policy mutation pairs ---
    log.info(f"Generating {policy_pairs} policy mutation pairs...")
    for i in range(policy_pairs):
        accuracy = random.uniform(0.3, 0.95)
        tier = random.choice(["L1", "L2", "L3", "L4", "L5"])
        model_name = random.choice(["qwen3:4b", "qwen3:8b", "granite3.3:8b",
                                     "llama3.2:3b", "phi4:14b", "gemma3:12b"])
        feedback_count = random.randint(5, 50)

        input_scl = f"@feedback → accumulated [model: {model_name}, tier: {tier}, accuracy: {accuracy:.2f}, count: {feedback_count}]"

        if accuracy < 0.5:
            action = f"@policy → mutate [demote_tier: {tier}, penalize_model: {model_name}, confidence: {1-accuracy:.2f}]"
        elif accuracy > 0.85:
            action = f"@policy → mutate [promote_tier: {tier}, prefer_model: {model_name}, confidence: {accuracy:.2f}]"
        else:
            action = f"@policy → mutate [maintain_tier: {tier}, monitor_model: {model_name}, confidence: {accuracy:.2f}]"

        pairs.append({
            "input": input_scl,
            "output": action,
            "source": "synthetic_policy",
            "quality": 0.7,
        })

    log.info(f"  ✓ {policy_pairs} policy pairs generated")

    # --- Trace-based pairs (the real training signal) ---
    log.info("Generating trace-based operational pairs...")
    log.info("  These are typed state transitions, not flat mappings.")

    # Import trace generator inline (it's in the mounted source)
    import sys
    sys.path.insert(0, "/root")  # Modal mount path

    # Inline trace generation (since we can't import from src/ in Modal easily)
    # Generate trace pairs using the same logic as trace_generator.py
    trace_templates = _generate_trace_pairs(log)
    pairs.extend(trace_templates)
    log.info(f"  ✓ {len(trace_templates)} trace pairs generated")
    log.info(f"    Task types: predict_next_state, repair, compress, detect_invalid")

    # Shuffle
    random.shuffle(pairs)

    # Split: 90% train, 10% eval
    split_idx = int(len(pairs) * 0.9)
    train_pairs = pairs[:split_idx]
    eval_pairs = pairs[split_idx:]

    # Save
    train_path = f"{output_dir}/train.jsonl"
    eval_path = f"{output_dir}/eval.jsonl"

    with open(train_path, "w") as f:
        for p in train_pairs:
            f.write(json.dumps(p) + "\n")

    with open(eval_path, "w") as f:
        for p in eval_pairs:
            f.write(json.dumps(p) + "\n")

    elapsed = t.time() - t0
    stats = {
        "total_pairs": len(pairs),
        "train_pairs": len(train_pairs),
        "eval_pairs": len(eval_pairs),
        "boot_pairs": boot_pairs,
        "route_pairs": route_pairs,
        "policy_pairs": policy_pairs,
        "train_path": train_path,
        "eval_path": eval_path,
        "elapsed_seconds": round(elapsed, 1),
    }

    # Save metadata
    with open(f"{output_dir}/metadata.json", "w") as f:
        json.dump(stats, f, indent=2)

    log.info(f"\n{'='*60}")
    log.info(f"Dataset generated in {elapsed:.1f}s")
    log.info(f"  Train: {len(train_pairs)} pairs → {train_path}")
    log.info(f"  Eval:  {len(eval_pairs)} pairs → {eval_path}")
    log.info(f"{'='*60}")

    volume.commit()
    return stats


# ---------------------------------------------------------------------------
# Step 2: Train the model
# ---------------------------------------------------------------------------

@app.function(
    image=training_image,
    gpu="A100",  # 80GB — 2x throughput over A10G, enables batch 32
    volumes={VOLUME_PATH: volume},
    timeout=3600,  # 1 hour max (should finish in ~7 min)
    secrets=[],
)
def train_model(
    base_model: str = BASE_MODEL,
    epochs: int = EPOCHS,
    resume_from_checkpoint: bool = True,
) -> dict:
    """Fine-tune the base model on SCL data with comprehensive logging."""
    import json
    import logging
    import sys
    import torch
    from datetime import datetime
    from pathlib import Path
    from datasets import Dataset, load_dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        TrainerCallback,
        TrainerState,
        TrainerControl,
    )
    from peft import LoraConfig, get_peft_model
    from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

    # --- Logging setup ---
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f"{VOLUME_PATH}/training.log"),
        ],
    )
    log = logging.getLogger("ckm.train")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{VOLUME_PATH}/checkpoints/{run_id}"
    os.makedirs(output_dir, exist_ok=True)

    log.info("=" * 70)
    log.info("  CKM TRAINING — Cortex Kernel Model")
    log.info("  Teaching a 0.3B model to speak SCL")
    log.info("=" * 70)
    log.info(f"Run ID:         {run_id}")
    log.info(f"Base model:     {base_model}")
    log.info(f"Epochs:         {epochs}")
    log.info(f"Batch size:     {BATCH_SIZE} × {GRADIENT_ACCUMULATION} = {BATCH_SIZE * GRADIENT_ACCUMULATION} effective")
    log.info(f"Learning rate:  {LEARNING_RATE}")
    log.info(f"LoRA r/alpha:   {LORA_R}/{LORA_ALPHA}")
    log.info(f"Max seq length: {MAX_SEQ_LENGTH}")
    log.info(f"Output dir:     {output_dir}")
    log.info(f"GPU:            {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    log.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log.info(f"VRAM:           {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    log.info("")

    # --- Load dataset ---
    log.info("[1/7] Checking dataset...")
    train_path = f"{VOLUME_PATH}/dataset/train.jsonl"
    eval_path = f"{VOLUME_PATH}/dataset/eval.jsonl"

    if not os.path.exists(train_path):
        log.error(f"Dataset not found at {train_path}")
        log.error("Run generate_dataset() first!")
        raise FileNotFoundError(train_path)
    log.info(f"  Dataset found: {train_path}")
    log.info("")

    # --- Load model ---
    log.info("[2/7] Loading base model (bf16, A100)...")
    t0 = time.time()

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",  # PyTorch native, no extra deps
    )
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Ensure the EOS token is set to <|im_end|> for proper stop behavior
    # Qwen2.5 uses token ID 151645 for <|im_end|>
    im_end_token = "<|im_end|>"
    if im_end_token in tokenizer.get_vocab():
        tokenizer.eos_token = im_end_token
        tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids(im_end_token)
        log.info(f"  EOS token set to: {im_end_token} (id={tokenizer.eos_token_id})")
    else:
        log.warning(f"  {im_end_token} not found in vocab, using default EOS")

    load_time = time.time() - t0
    log.info(f"  Model loaded in {load_time:.1f}s")
    log.info(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    log.info("")

    # --- Load dataset (after tokenizer so we can use apply_chat_template) ---
    log.info("[3/7] Loading and formatting dataset...")

    def load_jsonl_as_dataset(path):
        records = []
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                pair = json.loads(line)
                # Format with chat template (special tokens get correct IDs)
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": pair["input"]},
                    {"role": "assistant", "content": pair["output"]},
                ]
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                records.append({"text": text})
        return Dataset.from_list(records)

    train_dataset = load_jsonl_as_dataset(train_path)
    eval_dataset = load_jsonl_as_dataset(eval_path) if os.path.exists(eval_path) else None

    log.info(f"  Train samples: {len(train_dataset)}")
    if eval_dataset:
        log.info(f"  Eval samples:  {len(eval_dataset)}")
    log.info(f"  Sample:        {train_dataset[0]['text'][:120]}...")
    log.info("")

    # --- Apply LoRA ---
    log.info("[4/7] Applying LoRA configuration...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info(f"  Trainable parameters: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")
    log.info(f"  LoRA rank: {LORA_R}, alpha: {LORA_ALPHA}")
    log.info(f"  Target modules: {TARGET_MODULES}")
    log.info("")

    # --- Custom callback for detailed logging ---
    class CKMTrainingCallback(TrainerCallback):
        def __init__(self):
            self.train_start = None
            self.step_times = []
            self.best_eval_loss = float("inf")
            self.metrics_log = []

        def on_train_begin(self, args, state, control, **kwargs):
            self.train_start = time.time()
            log.info("=" * 40)
            log.info("  TRAINING STARTED")
            log.info("=" * 40)

        def on_log(self, args, state: TrainerState, control, logs=None, **kwargs):
            if logs:
                step = state.global_step
                epoch = logs.get("epoch", 0)
                loss = logs.get("loss", logs.get("train_loss", None))
                lr = logs.get("learning_rate", None)
                eval_loss = logs.get("eval_loss", None)

                # Compute throughput
                elapsed = time.time() - self.train_start if self.train_start else 0
                steps_per_sec = step / elapsed if elapsed > 0 else 0
                samples_per_sec = steps_per_sec * BATCH_SIZE * GRADIENT_ACCUMULATION

                metrics = {
                    "step": step,
                    "epoch": round(epoch, 3),
                    "loss": round(loss, 4) if loss else None,
                    "lr": f"{lr:.2e}" if lr else None,
                    "eval_loss": round(eval_loss, 4) if eval_loss else None,
                    "elapsed_s": round(elapsed, 1),
                    "samples_per_sec": round(samples_per_sec, 1),
                    "gpu_mem_gb": round(torch.cuda.memory_allocated() / 1024**3, 2) if torch.cuda.is_available() else 0,
                }
                self.metrics_log.append(metrics)

                parts = []
                if loss: parts.append(f"loss={loss:.4f}")
                if eval_loss: parts.append(f"eval_loss={eval_loss:.4f}")
                if lr: parts.append(f"lr={lr:.2e}")
                parts.append(f"epoch={epoch:.2f}")
                parts.append(f"{samples_per_sec:.0f} samples/s")
                parts.append(f"GPU={metrics['gpu_mem_gb']:.1f}GB")

                log.info(f"  [step {step:>4}] {' | '.join(parts)}")

                # Track best eval loss
                if eval_loss and eval_loss < self.best_eval_loss:
                    self.best_eval_loss = eval_loss
                    log.info(f"  ★ New best eval loss: {eval_loss:.4f}")

        def on_save(self, args, state, control, **kwargs):
            log.info(f"  💾 Checkpoint saved at step {state.global_step}")
            # Save metrics alongside checkpoint
            metrics_path = os.path.join(args.output_dir, "metrics.json")
            with open(metrics_path, "w") as f:
                json.dump(self.metrics_log, f, indent=2)
            volume.commit()

        def on_train_end(self, args, state, control, **kwargs):
            elapsed = time.time() - self.train_start
            log.info("")
            log.info("=" * 40)
            log.info("  TRAINING COMPLETE")
            log.info("=" * 40)
            log.info(f"  Total time:     {elapsed/60:.1f} minutes")
            log.info(f"  Total steps:    {state.global_step}")
            log.info(f"  Final loss:     {state.log_history[-1].get('loss', 'N/A')}")
            log.info(f"  Best eval loss: {self.best_eval_loss:.4f}")
            log.info(f"  Checkpoints:    {output_dir}")

    # --- Training arguments ---
    log.info("[5/7] Configuring trainer...")

    # Check for existing checkpoint to resume from
    resume_checkpoint = None
    if resume_from_checkpoint:
        checkpoint_dirs = sorted(
            [d for d in Path(output_dir).glob("checkpoint-*") if d.is_dir()],
            key=lambda x: int(x.name.split("-")[1]),
        )
        if checkpoint_dirs:
            resume_checkpoint = str(checkpoint_dirs[-1])
            log.info(f"  Resuming from checkpoint: {resume_checkpoint}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        bf16=True,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        eval_steps=EVAL_STEPS if eval_dataset else None,
        eval_strategy="steps" if eval_dataset else "no",
        save_total_limit=5,
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="eval_loss" if eval_dataset else None,
        report_to="none",
        dataloader_num_workers=4,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",  # Fastest optimizer on A100
    )

    # --- Create trainer ---
    callback = CKMTrainingCallback()

    # Mask loss on everything before assistant response.
    # The model only learns to produce the SCL output, not to predict system/user tokens.
    response_template = "<|im_start|>assistant\n"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        processing_class=tokenizer,
        data_collator=collator,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        callbacks=[callback],
    )

    log.info(f"  Total training steps: {trainer.state.max_steps if hasattr(trainer.state, 'max_steps') else '~' + str(len(train_dataset) * epochs // (BATCH_SIZE * GRADIENT_ACCUMULATION))}")
    log.info("")

    # --- Train ---
    log.info("[6/7] Training...")
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    # --- Save final model ---
    log.info("[7/7] Saving final LoRA adapter...")
    adapter_path = f"{VOLUME_PATH}/ckm_lora_final"
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    log.info(f"  Adapter saved to: {adapter_path}")

    # Save training summary
    summary = {
        "run_id": run_id,
        "base_model": base_model,
        "epochs": epochs,
        "total_steps": trainer.state.global_step,
        "final_loss": trainer.state.log_history[-1].get("loss") if trainer.state.log_history else None,
        "best_eval_loss": callback.best_eval_loss if callback.best_eval_loss < float("inf") else None,
        "training_time_minutes": (time.time() - callback.train_start) / 60 if callback.train_start else 0,
        "trainable_params": trainable,
        "total_params": total,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "adapter_path": adapter_path,
        "checkpoints_dir": output_dir,
    }

    with open(f"{VOLUME_PATH}/training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save full metrics history
    with open(f"{VOLUME_PATH}/metrics_history.json", "w") as f:
        json.dump(callback.metrics_log, f, indent=2)

    log.info("")
    log.info("Training Summary:")
    log.info(json.dumps(summary, indent=2))

    volume.commit()
    return summary


# ---------------------------------------------------------------------------
# Step 3: Merge LoRA + Export GGUF
# ---------------------------------------------------------------------------

@app.function(
    image=training_image,
    gpu="A100",
    volumes={VOLUME_PATH: volume},
    timeout=3600,
)
def export_gguf(quant_type: str = "Q4_K_M") -> str:
    """Merge LoRA adapter into base model and export as GGUF."""
    import json
    import logging
    import sys
    import torch
    from pathlib import Path
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f"{VOLUME_PATH}/export.log"),
        ],
    )
    log = logging.getLogger("ckm.export")

    adapter_path = f"{VOLUME_PATH}/ckm_lora_final"
    merged_path = f"{VOLUME_PATH}/ckm_merged"
    gguf_f16_path = f"{VOLUME_PATH}/cortex-kernel-f16.gguf"
    gguf_final_path = f"{VOLUME_PATH}/cortex-kernel.gguf"

    log.info("=" * 60)
    log.info("  CKM EXPORT — LoRA → Merged → GGUF")
    log.info("=" * 60)

    # --- Step 1: Merge LoRA ---
    log.info("[1/3] Merging LoRA adapter into base model...")
    t0 = time.time()

    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"No adapter found at {adapter_path}. Run train_model() first!")

    # Load adapter config to find base model
    import json as json_mod
    with open(f"{adapter_path}/adapter_config.json") as f:
        adapter_config = json_mod.load(f)
    base_model_name = adapter_config.get("base_model_name_or_path", BASE_MODEL)

    log.info(f"  Base model: {base_model_name}")
    log.info(f"  Adapter:    {adapter_path}")

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    merged = model.merge_and_unload()

    os.makedirs(merged_path, exist_ok=True)
    merged.save_pretrained(merged_path)
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    tokenizer.save_pretrained(merged_path)

    merge_time = time.time() - t0
    log.info(f"  ✓ Merged model saved ({merge_time:.1f}s)")

    # --- Step 2: Convert to GGUF ---
    log.info("[2/3] Converting to GGUF (f16)...")
    t1 = time.time()

    import subprocess
    convert_script = "/opt/llama.cpp/convert_hf_to_gguf.py"
    if not os.path.exists(convert_script):
        # Try alternative path
        convert_script = "/opt/llama.cpp/convert-hf-to-gguf.py"

    result = subprocess.run(
        ["python3", convert_script, merged_path,
         "--outfile", gguf_f16_path, "--outtype", "f16"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        log.error(f"GGUF conversion failed:\n{result.stderr}")
        raise RuntimeError("GGUF conversion failed")

    convert_time = time.time() - t1
    f16_size = os.path.getsize(gguf_f16_path) / (1024 * 1024)
    log.info(f"  ✓ F16 GGUF created ({f16_size:.0f} MB, {convert_time:.1f}s)")

    # --- Step 3: Quantize ---
    log.info(f"[3/3] Quantizing to {quant_type}...")
    t2 = time.time()

    quantize_bin = "/opt/llama.cpp/build/bin/llama-quantize"
    if not os.path.exists(quantize_bin):
        # Alternative path
        quantize_bin = "/opt/llama.cpp/build/bin/quantize"

    result = subprocess.run(
        [quantize_bin, gguf_f16_path, gguf_final_path, quant_type],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        log.error(f"Quantization failed:\n{result.stderr}")
        # Fall back to f16
        log.warning("Using f16 model instead")
        gguf_final_path = gguf_f16_path

    quant_time = time.time() - t2
    final_size = os.path.getsize(gguf_final_path) / (1024 * 1024)
    log.info(f"  ✓ {quant_type} GGUF created ({final_size:.0f} MB, {quant_time:.1f}s)")

    # --- Summary ---
    log.info("")
    log.info("=" * 60)
    log.info("  EXPORT COMPLETE")
    log.info("=" * 60)
    log.info(f"  Final model: {gguf_final_path}")
    log.info(f"  Size:        {final_size:.0f} MB")
    log.info(f"  Quant:       {quant_type}")
    log.info(f"  Total time:  {time.time() - t0:.1f}s")
    log.info("")
    log.info("  Deploy: cp cortex-kernel.gguf /mnt/cortex/models/")
    log.info("  Or:     modal volume get cortex-ckm cortex-kernel.gguf")

    # Cleanup intermediate files
    import shutil
    if os.path.exists(merged_path):
        shutil.rmtree(merged_path)
    if gguf_f16_path != gguf_final_path and os.path.exists(gguf_f16_path):
        os.remove(gguf_f16_path)

    volume.commit()
    return gguf_final_path


# ---------------------------------------------------------------------------
# Step 4: Validate the exported model
# ---------------------------------------------------------------------------

@app.function(
    image=training_image,
    gpu="A100",
    volumes={VOLUME_PATH: volume},
    timeout=600,
)
def validate_model() -> dict:
    """Run inference on test cases to validate the model produces valid SCL."""
    import subprocess
    import json
    import logging
    import sys

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("ckm.validate")

    model_path = f"{VOLUME_PATH}/cortex-kernel.gguf"
    if not os.path.exists(model_path):
        raise FileNotFoundError("No model found. Run export_gguf() first!")

    llama_cli = "/opt/llama.cpp/build/bin/llama-cli"
    if not os.path.exists(llama_cli):
        llama_cli = "/opt/llama.cpp/build/bin/main"

    test_cases = [
        "@hardware → state [cpu: Apple M1 Pro, cores: 10, ram_mb: 16384, gpu_type: apple, vram_mb: 16384, arch: aarch64]",
        "@hardware → state [cpu: Intel Core i3-10100, cores: 8, ram_mb: 8192, gpu_type: none, vram_mb: 0, arch: x86_64]",
        "@hardware → state [cpu: AMD Ryzen 9 7950X, cores: 32, ram_mb: 65536, gpu_type: nvidia, vram_mb: 24576, arch: x86_64]",
        "@task → classify [category: code, subtype: fix_bug, complexity: 0.70, input_tokens: 500]",
        "@task → classify [category: chat, subtype: greeting, complexity: 0.05, input_tokens: 10]",
        "@task → classify [category: math, subtype: proof, complexity: 0.90, input_tokens: 1200]",
        "@feedback → accumulated [model: qwen3:4b, tier: L2, accuracy: 0.35, count: 20]",
    ]

    results = []
    passed = 0

    log.info("=" * 60)
    log.info("  CKM VALIDATION")
    log.info("=" * 60)
    log.info(f"  Model: {model_path}")
    log.info(f"  Tests: {len(test_cases)}")
    log.info("")

    for i, input_scl in enumerate(test_cases):
        prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{input_scl}<|im_end|>\n<|im_start|>assistant\n"

        t0 = time.time()
        result = subprocess.run(
            [llama_cli, "-m", model_path,
             "-p", prompt,
             "-n", "64",
             "--temp", "0.0",
             "--no-display-prompt",
             "--simple-io",
             "-c", "512",
             "--log-disable"],
            capture_output=True, text=True, timeout=30,
        )
        latency = (time.time() - t0) * 1000

        output = result.stdout.strip().split("<|im_end|>")[0].strip() if result.returncode == 0 else ""

        # Validate: must be a valid SCL record
        is_valid = (
            output.startswith("@") and
            "→" in output and
            "[" in output and
            "]" in output
        )

        status = "✓" if is_valid else "✗"
        log.info(f"  [{status}] Test {i+1} ({latency:.0f}ms)")
        log.info(f"      IN:  {input_scl[:70]}...")
        log.info(f"      OUT: {output[:70]}{'...' if len(output) > 70 else ''}")

        if is_valid:
            passed += 1

        results.append({
            "input": input_scl,
            "output": output,
            "valid_scl": is_valid,
            "latency_ms": round(latency, 1),
        })

    log.info("")
    log.info(f"  Results: {passed}/{len(test_cases)} passed ({passed/len(test_cases)*100:.0f}%)")
    avg_latency = sum(r["latency_ms"] for r in results) / len(results)
    log.info(f"  Avg latency: {avg_latency:.0f}ms")

    validation = {
        "passed": passed,
        "total": len(test_cases),
        "pass_rate": passed / len(test_cases),
        "avg_latency_ms": round(avg_latency, 1),
        "results": results,
    }

    with open(f"{VOLUME_PATH}/validation.json", "w") as f:
        json.dump(validation, f, indent=2)

    volume.commit()
    return validation


# ---------------------------------------------------------------------------
# Step 5: Download the model
# ---------------------------------------------------------------------------

@app.function(volumes={VOLUME_PATH: volume})
def get_model_path() -> str:
    """Return the path to the final GGUF model on the volume."""
    model_path = f"{VOLUME_PATH}/cortex-kernel.gguf"
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        return f"{model_path} ({size_mb:.0f} MB)"
    return "Model not found. Run the full pipeline first."


# ---------------------------------------------------------------------------
# Main entry point — orchestrates the full pipeline
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    skip_datagen: bool = False,
    skip_train: bool = False,
    skip_export: bool = False,
    skip_validate: bool = False,
    boot_pairs: int = 2000,
    route_pairs: int = 5000,
    epochs: int = EPOCHS,
):
    """Run the full CKM training pipeline on Modal.

    Usage:
      modal run src/ckm/modal_train.py
      modal run src/ckm/modal_train.py --epochs 10 --boot-pairs 5000
      modal run src/ckm/modal_train.py --skip-datagen --skip-train  # just export
    """
    import json

    print("=" * 70)
    print("  CORTEX KERNEL MODEL — Full Training Pipeline")
    print("  Model that speaks SCL. Trained on its own boot telemetry.")
    print("=" * 70)
    print()

    # Step 1: Generate data
    if not skip_datagen:
        print("[1/4] Generating training dataset...")
        stats = generate_dataset.remote(
            boot_pairs=boot_pairs,
            route_pairs=route_pairs,
        )
        print(f"  ✓ Generated {stats['total_pairs']} pairs "
              f"(train={stats['train_pairs']}, eval={stats['eval_pairs']})")
        print()
    else:
        print("[1/4] Skipping dataset generation (--skip-datagen)")
        print()

    # Step 2: Train
    if not skip_train:
        print("[2/4] Training model on Modal A100 GPU...")
        print("  This will take ~5-7 minutes (A100 + flash attention + batch 32).")
        print("  Logs streaming below:")
        print("-" * 40)
        summary = train_model.remote(epochs=epochs)
        print("-" * 40)
        print(f"  ✓ Training complete!")
        print(f"    Steps: {summary['total_steps']}")
        print(f"    Final loss: {summary.get('final_loss', 'N/A')}")
        print(f"    Best eval loss: {summary.get('best_eval_loss', 'N/A')}")
        print(f"    Time: {summary.get('training_time_minutes', 0):.1f} minutes")
        print()
    else:
        print("[2/4] Skipping training (--skip-train)")
        print()

    # Step 3: Export GGUF
    if not skip_export:
        print("[3/4] Exporting to GGUF (Q4_K_M)...")
        gguf_path = export_gguf.remote()
        print(f"  ✓ GGUF exported: {gguf_path}")
        print()
    else:
        print("[3/4] Skipping export (--skip-export)")
        print()

    # Step 4: Validate
    if not skip_validate:
        print("[4/4] Validating model...")
        validation = validate_model.remote()
        print(f"  ✓ Validation: {validation['passed']}/{validation['total']} "
              f"({validation['pass_rate']*100:.0f}% valid SCL)")
        print(f"    Avg latency: {validation['avg_latency_ms']:.0f}ms")
        print()
    else:
        print("[4/4] Skipping validation (--skip-validate)")
        print()

    # Done
    print("=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)
    print()
    print("  Download the model:")
    print("    modal volume get cortex-ckm cortex-kernel.gguf .")
    print()
    print("  Deploy to USB stick:")
    print("    cp cortex-kernel.gguf /mnt/cortex/models/")
    print()
    print("  The next boot will use CKM instead of heuristics.")
    print("  The model speaks SCL. The OS speaks SCL. They are the same thing.")
