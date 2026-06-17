# Cortex: AI as PID 1

**An AI-native operating system kernel where the inference engine isn't an app — it's `init`.**

---

## Abstract

Modern AI-assisted computing treats large language models as user-space applications: launched on demand, isolated from the system, and discarded after use. This is analogous to the pre-Unix era, where every program had to implement its own I/O routines. Cortex proposes a different architecture: the LLM inference engine as PID 1 — the always-running, kernel-adjacent process that mediates between the user and everything else.

Cortex detects available hardware, constructs a tiered model hierarchy (L0–L7), and routes every request to the smallest model capable of handling it. When confidence is low, it escalates — first to a cross-family challenger model, then to a multi-model swarm, and finally to a remote frontier API. The result is an inference layer that behaves like an operating system scheduler: resource-aware, always available, and self-correcting.

---

## 1. Introduction

### 1.1 The Problem

Today's LLM deployment follows a client-server model inherited from web applications. A user opens an app, the app calls an API, a large model processes the request, and the response returns. This architecture has three structural inefficiencies:

1. **Overprovisioning.** Most requests are routed to the largest available model regardless of complexity. A yes/no classification consumes the same resources as a multi-step coding task.

2. **Cold starts.** Models are loaded on demand and evicted under memory pressure. The resulting latency — often seconds — makes LLMs unsuitable as system-level services.

3. **Single points of failure.** A request goes to one model. If that model hallucinates, there is no second opinion. Confidence is assumed, never measured.

### 1.2 The Thesis

An operating system doesn't ask which process scheduler to start. It doesn't load the memory manager on demand. These are kernel services — always running, always available, mediating access to hardware on behalf of user-space programs.

Cortex applies this principle to inference. The L0 router is PID 1: the first process up, the last one down. It classifies every inbound request, selects the appropriate tier, and supervises execution. Tiers L0–L2 remain resident in memory like kernel modules. Higher tiers load on demand like user-space daemons. Remote frontier models (L7) are the network — a fallback when local resources are insufficient.

---

## 2. Architecture

### 2.1 Tier System

Cortex organizes models into eight tiers mapped to operating system concepts:

| Tier | Parameters | OS Analogy | Role | Resident | TTFT Target |
|------|-----------|------------|------|----------|-------------|
| L0 | 0.5–1B | `init` / `systemd` | Router, classifier, reflex | Always | ~10ms |
| L1 | 1–2B | Syscall interface | Tool calls, summarization | Always | ~20ms |
| L2 | 3–4B | Shell / window manager | File ops, drafts, structured extraction | Always | ~40ms |
| L3 | 7–8B | User-space daemon | Coding, multi-step tasks | On demand | ~60ms |
| L4 | 12–14B | Heavy daemon | Debugging, planning, complex reasoning | On demand | ~100ms |
| L5 | 30–32B | Workstation service | Strong local reasoning | On demand | ~200ms |
| L6 | 64–70B | Full workstation | Frontier-adjacent local quality | On demand | ~400ms |
| L7 | Frontier | Network / remote API | Escalation when local fails | On demand | ~500ms |

**Always-resident tiers (L0–L2)** stay loaded in VRAM at all times. Their combined footprint is approximately 4.5 GB at 4-bit quantization — small enough to coexist with higher tiers on any modern GPU or Apple Silicon machine. They never swap out, ensuring sub-50ms time-to-first-token for the majority of requests.

**On-demand tiers (L3–L6)** load when needed and are evicted under memory pressure, managed by the Model Manager in the same way `systemd` manages service lifecycles.

**L7** is not a local model. It is a passthrough to remote frontier APIs (OpenAI, Anthropic) — the "network" in the OS analogy. It activates only when local models cannot reach sufficient confidence.

### 2.2 The Core Ladder

Cortex uses Qwen3 as its primary model family across all local tiers:

| Tier | Model |
|------|-------|
| L0 | Qwen3-0.6B |
| L1 | Qwen3-1.7B |
| L2 | Qwen3-4B |
| L3 | Qwen3-8B |
| L4 | Qwen3-14B |
| L5 | Qwen3-30B-A3B / Qwen3-32B |
| L6 | Qwen2.5-72B |

A single model family provides a consistent "core ladder" — predictable behavior, shared tokenization, and uniform instruction formatting. The choice of Qwen3 is pragmatic: it offers the widest range of sizes (0.6B to 32B) from a single architecture, with strong multilingual performance and native tool-calling support.

### 2.3 GGUF-First Format Strategy

Cortex standardizes on GGUF (llama.cpp's quantized model format) as the universal model file format. A single GGUF file runs identically on:

- **NVIDIA CUDA** (Linux, Windows)
- **AMD ROCm** (Linux)
- **Apple Metal** (macOS)
- **CPU** (AVX2, NEON — any platform)

This eliminates the format fragmentation problem. There is no need for separate model files per platform. For NVIDIA users running vLLM, Cortex optionally uses AWQ-quantized models for better prefix caching performance, but falls back to GGUF universally.

---

## 3. Routing

### 3.1 The L0 Router

Every request passes through L0 before any work begins. The router performs two functions:

1. **Task classification.** Categorizes the request into one of ten categories: `classify`, `tool_call`, `multi_tool`, `code`, `debug`, `plan`, `analyze`, `generate`, `safety`, or `unknown`.

2. **Tier selection.** Maps the category and estimated complexity to the minimum viable tier.

When the L0 model (Qwen3-0.6B) is loaded, routing uses the model itself — a 0.6B classifier that reads the prompt and outputs a JSON object specifying tier, category, and confidence. When no L0 model is available, Cortex falls back to rule-based heuristics using regex pattern matching and complexity scoring.

### 3.2 Complexity Estimation

The heuristic router estimates complexity on a 0–1 scale using:

- **Input length** — longer prompts generally require larger context windows
- **Pattern matching** — code keywords, planning language, tool verbs, safety indicators
- **Multi-step indicators** — conjunctions like "then", "after that", "next"
- **Tool count** — number of distinct tool-call verbs in the prompt

Each category also has a minimum tier floor. For example, `code` requires at least L3; `safety` requires at least L4. The router takes the maximum of the category floor and the complexity-derived tier.

### 3.3 Tier Clamping

If the selected tier exceeds what the hardware can run, the router clamps to the maximum feasible tier and attaches an **escalation hint** pointing to L7. Confidence is reduced proportionally to the gap between the ideal and actual tier.

---

## 4. Confidence Verification

### 4.1 The Problem with Single-Model Inference

A single model's output has no external check. It may be fluent but wrong. Traditional approaches address this with temperature sampling or self-consistency (querying the same model multiple times), but these offer limited value: copies of the same model share the same blind spots.

### 4.2 Cross-Family Challenge

Cortex maintains a **challenge model catalog** — models from families other than the core ladder (Llama, Gemma, Granite, Phi, OLMo, SmolLM). These models were trained on different data, with different architectures and different optimization objectives. When they agree with the core model, the probability of a shared hallucination is substantially lower than agreement among copies of the same family.

The challenge process:

1. The core model (Qwen) generates an answer.
2. If routing confidence is below 0.75, the Challenger loads a model from a different family at the same tier.
3. The challenger model answers the same prompt independently.
4. Answers are compared using normalized text overlap, yes/no detection, and conclusion extraction.
5. Agreement is classified as: `strong_agree`, `weak_agree`, `ambiguous`, `disagree`, or `strong_disagree`.

| Agreement | Confidence | Action |
|-----------|-----------|--------|
| Strong agree | 0.85–0.95 | Return answer |
| Weak agree | 0.65 | Return answer |
| Ambiguous | 0.45 | Consider escalation |
| Disagree | 0.25 | Escalate to swarm |
| Strong disagree | 0.10 | Escalate to swarm |

### 4.3 Challenge Model Catalog

Each tier has 2–3 challenge models from different families:

| Tier | Challenge Families |
|------|-------------------|
| L0 | Llama 3.2, OLMo 2, Gemma 3 |
| L1 | Llama 3.2, Granite 3.3, SmolLM3 |
| L2 | Gemma 3, Phi-4 mini |
| L3 | Granite 3.3, Llama 3.1 |
| L4 | Gemma 3, Phi-4 |
| L5 | Granite 3.3, OLMo 2 |
| L6 | Llama 3.3 |

---

## 5. Swarm Consensus

### 5.1 When Challenge Fails

If the challenger disagrees with the core model — or if initial confidence is very low — Cortex escalates to a **swarm**: a parallel fan-out to multiple models across multiple families.

| Difficulty | Swarm Size | Strategy |
|-----------|-----------|----------|
| Hard | 3–5 models | Small swarm, weighted vote |
| Hardest | All available | Large swarm, weighted consensus |

### 5.2 Clustering and Voting

Swarm responses are grouped into **agreement clusters** using pairwise comparison (the same text-overlap algorithm used by the Challenger). Two responses land in the same cluster if they exhibit strong or weak agreement.

**Weighted voting** determines the winner. Each model's vote is weighted by its parameter count:

| Parameters | Weight |
|-----------|--------|
| ≤1B | 1.0 |
| ≤4B | 1.5 |
| ≤8B | 2.0 |
| ≤14B | 2.5 |
| ≤32B | 3.0 |
| ≤70B | 4.0 |

The winning cluster's highest-weighted model provides the final answer. Confidence is computed as:

```
confidence = agreement_ratio + family_bonus - size_penalty
```

Where:
- **agreement_ratio** = fraction of total weight in the winning cluster
- **family_bonus** = 0.05 per agreeing family (capped at 0.15)
- **size_penalty** = penalty for small swarms (diminishes with more models)

### 5.3 Cross-Family Signal

The key insight is that **cross-family agreement is a stronger signal than within-family agreement**. Three models from three different families converging on the same answer provides higher confidence than five copies of the same family agreeing — because independent training pipelines, data mixtures, and architectures make correlated failures less likely.

---

## 6. Escalation Pipeline

The full processing pipeline follows a deterministic escalation path:

```
User Request
    │
    ▼
┌─────────┐
│  Router  │   L0 classifies → selects tier
│  (L0)    │
└────┬────┘
     │
     ▼
┌──────────────┐
│  Core Model   │   Qwen at selected tier generates answer
│  (Qwen3)      │
└──────┬───────┘
       │
  confidence < 0.75?
       │
  ┌────┴────┐
  │ No      │ Yes
  ▼         ▼
Return  ┌────────────┐
        │ Challenger  │   Different family verifies
        └──────┬─────┘
               │
          agree?
          ┌────┴────┐
          │ Yes     │ No
          ▼         ▼
       Return   ┌────────┐
                │ Swarm   │   3–5+ models, weighted vote
                └────┬───┘
                     │
               consensus?
               ┌────┴────┐
               │ Yes     │ No
               ▼         ▼
            Return   Escalate → L7 (remote frontier)
```

Each step is logged to the audit trail with the full escalation path, enabling post-hoc analysis of routing quality.

---

## 7. System Layer

### 7.1 Hardware Detection

On boot, Cortex scans the host system:

- **CPU**: model, core count, ISA features (AVX2, AVX-512, NEON, AMX)
- **GPU**: NVIDIA (via `nvidia-smi`), AMD (via `rocm-smi`), Apple Silicon (via `sysctl`)
- **Memory**: total and available RAM; for Apple Silicon, unified memory with 75% GPU allocation
- **Backends**: which inference engines are installed (Ollama, llama.cpp, vLLM, MLX, TensorRT-LLM, ExLlamaV2)

The result is a `SystemProfile` — a structured snapshot of the machine's capabilities. Tier feasibility is assessed against this profile: each tier's VRAM requirement is compared against the available budget (total VRAM minus a 20% OS reserve for discrete GPUs, or 75% of unified memory for Apple Silicon).

### 7.2 Model Manager

The Model Manager is the `systemd` of Cortex. It:

- **Boots** always-hot models (L0–L2) at startup
- **Loads** on-demand models when a tier is first requested
- **Evicts** models under VRAM pressure using LRU policy
- **Health-checks** loaded models and restarts failed backends
- **Tracks** state per model: `loading`, `ready`, `failed`, `evicted`

### 7.3 Backend Adapters

Cortex supports multiple inference backends through a unified adapter interface:

| Backend | Format | Platforms | Best For |
|---------|--------|-----------|----------|
| Ollama | GGUF | All | Easiest setup, automatic model management |
| llama.cpp | GGUF | All | Direct control, lowest overhead |
| vLLM | AWQ/GPTQ | Linux + NVIDIA | Prefix caching, high throughput |
| OpenAI API | API | Any (remote) | L7 frontier escalation |

The adapter interface exposes synchronous and asynchronous completion methods, abstracting away backend differences. Backend selection is automatic based on the detected hardware profile.

### 7.4 Persistent Memory

All state that must survive a restart is stored in SQLite (`~/.cortex/cortex.db`) with WAL mode for concurrent reads:

- **Threads**: conversation state, message history, per-thread metadata
- **Audit log**: every request with routing decision, tier, model, confidence, latency, and escalation path (append-only)
- **KV cache index**: which prompt prefixes are cached, for prefix-cache reuse across requests
- **Policies**: runtime configuration and per-app overrides
- **Usage stats**: daily aggregates by tier and model

### 7.5 Policy Engine

Policies are scoped hierarchically: `thread > app > global`. They control:

- Maximum tier allowed per app
- Cloud (L7) escalation permissions
- Rate limits
- Token budgets
- Tool access restrictions

---

## 8. API Compatibility

### 8.1 The Daemon

Cortex runs as a local HTTP daemon on `localhost:11411`. Any application that supports `OPENAI_BASE_URL` can use it:

```bash
export OPENAI_BASE_URL=http://localhost:11411/v1
export OPENAI_API_KEY=local
```

### 8.2 Multi-Format Support

The daemon accepts three API formats and translates them to a unified internal representation:

| Format | Endpoint | Status |
|--------|----------|--------|
| OpenAI Chat Completions | `POST /v1/chat/completions` | Native (pass-through) |
| OpenAI Responses API | `POST /v1/responses` | Translated |
| Anthropic Messages API | `POST /v1/messages` | Translated |

Translation is handled by the API Adapter, which normalizes inbound requests to Chat Completions format and converts responses back to the client's expected format. Tool calling is passed through natively. Multimodal content (images, audio) is text-extracted for text-only models.

### 8.3 Compatible Clients

The following tools work with Cortex out of the box:

- **IDE assistants**: Cursor, VS Code Copilot, Cline, Continue, aider
- **SDKs**: OpenAI Python SDK, Anthropic Python SDK
- **Frameworks**: LangChain, LlamaIndex
- **UIs**: Open WebUI

---

## 9. Performance Characteristics

### 9.1 TTFT Optimization

Time-to-first-token is the critical latency metric. Cortex optimizes TTFT through:

1. **Right-sizing**: routing to the smallest capable model eliminates unnecessary computation
2. **Always-resident tiers**: L0–L2 are never cold-started
3. **Prefix caching**: common prompt prefixes are cached and reused (tracked in the KV cache index)
4. **Quantization**: 4-bit quantization (Q4_K_M) reduces memory footprint and speeds inference
5. **Thinking suppression**: for reflex tiers (L0–L2), Qwen3's thinking mode is suppressed via `/no_think` to avoid wasting tokens on internal reasoning

### 9.2 Platform-Specific Performance

| Platform | GPU | Backend | Expected TTFT |
|----------|-----|---------|---------------|
| Linux | NVIDIA H100 | vLLM (AWQ) | 20–50ms |
| Linux | NVIDIA RTX 4090 | vLLM / llama.cpp | 30–80ms |
| macOS | Apple M1–M4 | llama.cpp (Metal) | 50–150ms |
| Linux | AMD RX 7900 | llama.cpp (ROCm) | 40–100ms |
| Any | CPU only | llama.cpp (AVX2/NEON) | 200–2000ms |

### 9.3 Graceful Degradation

Cortex adapts to available hardware:

- **192 GB Apple Silicon**: all tiers L0–L6 fit concurrently
- **24 GB NVIDIA GPU**: L0–L4 comfortably, L5 with eviction
- **8 GB RAM, no GPU**: L0–L2 only, with L7 as fallback
- **No backends installed**: graceful failure with clear diagnostics

The system always functions at whatever level the hardware supports, even if that is only L0–L2 — analogous to a minimal `initramfs` boot.

---

## 10. Design Principles

1. **Start small.** Route to the smallest model likely to handle the task. Escalate only on evidence of insufficient capability.

2. **Measure confidence, don't assume it.** Cross-family verification provides an empirical signal that single-model inference lacks.

3. **One format everywhere.** GGUF runs on every platform. Eliminate format fragmentation.

4. **Always available.** The core inference tiers (L0–L2) are kernel modules, not applications. They are loaded at boot and never evicted.

5. **Degrade gracefully.** If hardware supports only L0–L2, the system still works. L7 is the network fallback.

6. **Audit everything.** Every request, routing decision, escalation, and latency measurement is logged for post-hoc analysis and continuous improvement.

---

## 11. Future Work

- **LLM-as-judge aggregation**: replace text-overlap comparison in the swarm with a small judge model for more nuanced agreement detection
- **Persistent credibility scoring**: track which models consistently agree with eventual consensus and weight their votes accordingly over time
- **Embedding-based semantic memory**: vector search over conversation history for long-term context recall
- **Adaptive routing**: use audit log data to fine-tune the L0 router based on observed outcomes
- **Multi-GPU scheduling**: distribute concurrent tiers across multiple GPUs with VRAM-aware placement
- **Hardware-specific kernel tuning**: automatic selection of flash attention, paged attention, and speculative decoding based on detected GPU architecture

---

## 12. Conclusion

Cortex reframes LLM inference as an operating system problem. Instead of treating models as isolated applications, it treats them as a tiered hierarchy of kernel services — always available, resource-managed, and self-verifying. The L0 router acts as PID 1, the challenge system provides cross-family confidence checks, and the swarm offers consensus for the hardest problems. The result is an inference layer that is faster (by routing to the smallest capable model), more reliable (by measuring confidence empirically), and universally deployable (by standardizing on GGUF).

The question is not whether AI should be integrated into the operating system. It is whether the operating system should be redesigned around AI. Cortex is an argument that it should.

---

*Cortex is open source under the MIT License.*
*Repository: github.com/elevate-foundry/braille*
