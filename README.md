# Cortex

**AI as PID 1.** An AI-native operating system kernel where the inference engine isn't an app — it's `init`.

Cortex treats LLM inference the way a traditional OS treats process scheduling: it's the always-running, kernel-adjacent service that mediates between the user and everything else. It detects your hardware, builds a tiered model hierarchy (L0–L7), and routes every request to the smallest model that can handle it — escalating only when confidence is low.

## Architecture

| Traditional OS | Cortex |
|----------------|--------|
| `init` / `systemd` (PID 1) | **L0 Router** — always running, first thing up, last thing down |
| Kernel syscall interface | **L1 Agent** — thin wrapper over tool calls, always hot |
| Shell / window manager | **L2 Agent** — always-hot local agent, user-facing |
| User-space daemons | **L3–L6** — loaded on demand, handle real work |
| Remote API calls | **L7** — escalation to frontier when local fails |

## Quick Start

```bash
# Detect your hardware and show what Cortex can run
python -m src detect

# Output as JSON
python -m src detect --json

# Show which model tiers fit on your system
python -m src tiers

# Route a prompt to the appropriate tier
python -m src route "refactor the database layer to use connection pooling"

# Launch the optimal inference server
python -m src serve

# Benchmark TTFT against a running server
python -m src benchmark --url http://localhost:8000 --n 10

# Simulate different hardware profiles
python -m src detect --simulate linux-h100
python -m src tiers --simulate mac-m4-ultra
python -m src simulate-all
```

## What It Does

1. **Hardware Detection** — Scans CPU, GPU (NVIDIA/AMD/Apple Silicon), RAM, OS, architecture
2. **Tier Assessment** — Maps your hardware budget to an 8-tier model hierarchy (L0–L7)
3. **Intelligent Routing** — L0 classifies every request and routes to the smallest capable tier
4. **Backend Selection** — Picks the optimal engine (vLLM, llama.cpp, Ollama) with GGUF as the universal format
5. **Cross-Family Verification** — Challenge models from different families (Qwen, Llama, Gemma, Granite, Phi, OLMo) increase confidence through disagreement detection
6. **TTFT Optimization** — Prefix caching, flash attention, quantization, right-sized models

## Tier System

| Tier | Size | OS Role | Always Hot? | TTFT Target |
|------|------|---------|-------------|-------------|
| L0 | 0.5–1B | Reflex / Router | Yes | ~10ms |
| L1 | 1–2B | Tiny Syscall Agent | Yes | ~20ms |
| L2 | 3–4B | Always-Hot Local Agent | Yes | ~40ms |
| L3 | 7–8B | Primary Local OS Agent | No | ~60ms |
| L4 | 12–14B | Heavy Local Reasoner | No | ~100ms |
| L5 | 30–32B | Local Frontier-ish | No | ~200ms |
| L6 | 64–70B | Workstation Model | No | ~400ms |
| L7 | Frontier | Remote Escalation | No | ~500ms |

## Confidence & Swarm Strategy

Cross-family agreement increases confidence. If models from different families converge on the same answer, trust is higher than N copies of the same family agreeing.

| Difficulty | Strategy | Description |
|------------|----------|-------------|
| Easy | Single model | One core-ladder model answers; fast, cheap |
| Medium | Verify | Core model answers, one challenge model from a different family confirms |
| Hard | Swarm | Fan out to 3–5 models across multiple families, majority-vote |
| Hardest | Large swarm | All available families + sizes, weighted consensus |

## Supported Platforms

| Platform | GPU | Best Backend | Expected TTFT |
|----------|-----|-------------|---------------|
| Linux | NVIDIA A100/H100 | vLLM (AWQ) | 20–50ms |
| Linux | NVIDIA RTX 4090 | vLLM / llama.cpp | 30–80ms |
| macOS | Apple M1–M4 | llama.cpp (Metal) | 50–150ms |
| Linux | AMD RX 7900 | llama.cpp (ROCm) | 40–100ms |
| Any | CPU only | llama.cpp (AVX2/NEON) | 200–2000ms |

## Key Design Principles

- **GGUF-first** — Same model file runs on CUDA, Metal, ROCm, and CPU. One format everywhere.
- **Start small** — Route to the smallest model likely to handle the task.
- **Escalate on uncertainty** — Low confidence or cross-family disagreement triggers escalation.
- **Always-resident core** — L0–L2 stay in memory like kernel modules. They never swap out.
- **Graceful degradation** — If hardware only supports L0–L2, the system still functions.

## Requirements

- Python 3.10+
- At least one inference backend installed (Ollama is easiest to start with)

## License

MIT
