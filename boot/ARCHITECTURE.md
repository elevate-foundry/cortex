# Cortex v3 — The Model IS the Kernel

> Not "Linux that runs AI." The AI that runs Linux.

## The Problem With v2

```
v2 architecture:
  kernel → init.sh → cortex-init.py → llama.cpp → qwen/tinyllama
                                        ↑
                                    commodity runtime
                                    commodity weights
                                    = nothing novel
```

The "intelligence" is just prompt routing to off-the-shelf models.
Any hobbyist can replicate this in a weekend.

## v3 Vision: Three Pillars

### Pillar 1: Specialized Cortex Models

The L0 model isn't a general chatbot — it's a **trained kernel scheduler**.

```
Traditional OS kernel:
  syscall → lookup table → handler function → return

Cortex kernel:
  request → L0 inference → {route, config, action} → execute
```

**What this means concretely:**

- **CortexRouter-0.6B**: Fine-tuned on (prompt, hardware_profile) → (tier, model, params)
  - Training data: millions of routing decisions + their outcomes
  - Beats heuristic routing by 30%+ on accuracy
  - Runs in <5ms on any CPU (0.6B, int4)

- **CortexOS-1.7B**: Fine-tuned on OS-level tasks
  - Input: hardware state + user intent
  - Output: shell commands, kernel params, service configs
  - Training data: sysadmin tasks, hardware manuals, Linux source
  - This model configures the system it's running on

- **CortexProbe-0.3B**: Ultra-tiny diagnostic model
  - Runs BEFORE the main model loads
  - Classifies hardware from /proc and /sys in 1 inference pass
  - Outputs: optimal thread count, GPU layers, context size, quantization level
  - Replaces 500 lines of if/else hardware detection

**Dataset creation:**
- Synthetic: Generate (hardware_profile, request) → best_routing_decision
- Live: Log every routing decision in deployed Cortex instances
- Distillation: Use GPT-4/Claude to label ideal routes, distill into 0.6B

### Pillar 2: Hardware-Native Inference

Skip the runtime entirely. The model weights map directly to hardware.

```
Current path (slow):
  GGUF file → llama.cpp → BLAS library → CUDA/Metal → GPU

Cortex path (fast):
  Cortex binary blob → mmap → direct GPU kernel dispatch → token
```

**Concrete implementation:**

1. **Boot-stage micro-engine** (~50KB binary, no dependencies)
   - Statically linked, position-independent
   - Only implements: attention, RoPE, RMSNorm, SiLU, matmul
   - Architecture-specific builds:
     - x86_64: AVX-512 / AVX2 / SSE4 (runtime detection)
     - aarch64: NEON / SVE / SVE2
     - NVIDIA: pre-compiled PTX for sm_75+ (Turing+)
     - Apple: pre-compiled Metal shader archive

2. **Weight format: Cortex Tensor Format (.ctf)**
   - Weights pre-arranged for zero-copy mmap inference
   - Memory layout matches hardware's preferred access pattern
   - No deserialization — `mmap()` and go
   - Contains embedded compute graph (no runtime graph construction)
   - Header includes hardware affinity hints

3. **Boot-to-token pipeline:**
   ```
   t=0.0s  kernel loads initramfs
   t=0.1s  init.sh mmap's L0 weights from ESP partition
   t=0.2s  micro-engine runs first inference (hardware probe)
   t=0.5s  system configured based on probe output
   t=0.8s  full daemon ready, first token generated
   ```
   Target: **<1 second from power-on to first inference token**

4. **GPU fast-path (zero-copy):**
   - On NVIDIA: model weights in VRAM via BAR1 mapping
   - On Apple: model in unified memory, no copy needed
   - On CPU: mmap with huge pages, NUMA-aware placement

### Pillar 3: Self-Modifying OS

Each boot, Cortex observes what happened and mutates its own configuration.

```
Boot N:
  - Hardware: RTX 4090 detected, 24GB VRAM
  - Observation: L2 model fit entirely in VRAM
  - Observation: GPU layers=999 worked, batch_size=8 was optimal
  - Observation: Network on eth0, DHCP took 1.2s
  - Action: Write optimized config to CORTEX partition

Boot N+1:
  - Skip hardware detection (fingerprint matches)
  - Load pre-computed optimal config
  - Boot 0.4s faster

Boot N+10:
  - Pattern: user always sends code requests
  - Mutation: promote code-specialized model to L1 hot slot
  - Mutation: increase context window (user sends long files)
  - Mutation: disable challenger (user accepts first answer 98% of time)
```

**Implementation:**

1. **Boot Telemetry Log** (append-only, on CORTEX partition)
   ```json
   {"boot_id": "abc123", "hardware_fp": "a1b2c3", "boot_ms": 4200,
    "gpu_detected": "RTX 4090", "vram_used_mb": 8400,
    "models_loaded": ["L0", "L1", "L2"], "first_token_ms": 890}
   ```

2. **Boot Optimizer** (runs as background task after daemon is ready)
   - Reads last N boot logs
   - Proposes mutations to `cortex.toml`:
     - Thread count, GPU layers, batch size, context window
     - Model selection (promote/demote)
     - Service topology (skip unused services)
   - Writes `.cortex-boot-cache` for next boot

3. **Self-Modifying Policy (already exists in v2!)**
   - `policy_rewriter.py` already observes routing outcomes
   - Extend to observe boot-time decisions
   - The USB stick literally gets smarter with each boot

4. **Hardware Fingerprinting**
   - SHA256(cpu_model + ram_size + gpu_pci_id + disk_serial)
   - Cache optimal config per fingerprint
   - Same stick on different machines = different optimal configs
   - Plug into laptop → one config. Plug into server → different config.

---

## What Makes This Actually Novel

| Existing | Cortex v3 |
|----------|-----------|
| OS runs AI as an app | **AI IS the OS** — inference is the scheduler |
| Generic model weights | **Purpose-trained kernel models** — routing IS inference |
| llama.cpp/vLLM runtime | **Hardware-native micro-engine** — zero overhead |
| Static boot config | **Self-modifying** — each boot learns from the last |
| One hardware target | **Hardware-adaptive** — same stick, different optimal paths |

Nobody has done: **a purpose-trained transformer model that serves as the process scheduler, hardware configurator, and routing engine of a self-modifying operating system that gets faster every time you boot it.**

That's not "Linux + llama.cpp." That's a new kind of OS.

---

## Implementation Priority

```
Phase 1: Micro-Engine (2-3 weeks)
  - 50KB statically-linked inference binary
  - Supports 0.3B-0.6B models only (tiny, fast)
  - x86_64 AVX2 + aarch64 NEON
  - Custom weight format (.ctf) with mmap
  - Target: first token in <200ms on cold boot

Phase 2: CortexProbe-0.3B (1-2 weeks)
  - Train on (hwinfo_text → optimal_config_json) pairs
  - Generate training data from 1000+ hardware profiles
  - Distill from GPT-4 labeling of optimal configurations
  - Deploy as the FIRST inference at boot (before anything else loads)

Phase 3: Boot Telemetry + Self-Modification (1 week)
  - Log every boot decision
  - Background optimizer proposes mutations
  - Hardware fingerprint → cached optimal config
  - Measure: boot time reduction over 10 boots

Phase 4: CortexRouter-0.6B (2-3 weeks)
  - Fine-tune on routing decision dataset
  - Replace heuristic router entirely
  - A/B test: heuristic vs trained router
  - Measure: routing accuracy, TTFT, user satisfaction

Phase 5: GPU Kernels (ongoing)
  - Pre-compiled CUDA PTX for common NVIDIA GPUs
  - Metal shader archive for Apple Silicon
  - Vulkan compute shaders for AMD + Intel
  - Target: match llama.cpp perf with zero setup
```

---

## The Tagline

> **Cortex: The first operating system where the kernel is a neural network.**

Not "runs AI." IS AI.
