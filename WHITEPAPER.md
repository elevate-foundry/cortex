# Cortex: AI as PID 1

**An agent microkernel where the inference engine isn't an app — it's `init`.**

---

## Abstract

Modern AI-assisted computing treats large language models as user-space applications: launched on demand, isolated from the system, and discarded after use. This is analogous to the pre-Unix era, where every program had to implement its own I/O routines. Cortex proposes a different architecture: the LLM inference engine as PID 1 — the always-running, kernel-adjacent process that mediates between the user and everything else.

Cortex detects available hardware, constructs a tiered model hierarchy (L0–L7), and routes every request to the smallest model capable of handling it. When confidence is low, it escalates — first to a cross-family challenger model, then to a multi-model swarm, and finally to a remote frontier API. Any model the user installs — regardless of family, quantization, or size — is automatically discovered, tier-classified, and made available as a challenger or swarm participant. The result is an inference layer that behaves like an operating system scheduler: resource-aware, always available, self-correcting, and model-agnostic.

Beyond routing, Cortex implements the full agent microkernel: a tool registry with permission rings, a policy engine for per-app access control, a resilient execution layer with circuit breakers and fallback, persistent memory for conversations and audit trails, and Semantic Compression Language (SCL) as its native control protocol. Every request is a syscall. Every model is a device driver. Every tool is a capability. Every policy is a permission boundary.

---

## 1. Introduction

### 1.1 The Problem

Today's LLM deployment follows a client-server model inherited from web applications. A user opens an app, the app calls an API, a large model processes the request, and the response returns. This architecture has four structural inefficiencies:

1. **Overprovisioning.** Most requests are routed to the largest available model regardless of complexity. A yes/no classification consumes the same resources as a multi-step coding task.

2. **Cold starts.** Models are loaded on demand and evicted under memory pressure. The resulting latency — often seconds — makes LLMs unsuitable as system-level services.

3. **Single points of failure.** A request goes to one model. If that model hallucinates, there is no second opinion. Confidence is assumed, never measured.

4. **No permission model.** Applications and tools operate with unconstrained access. There is no kernel-level enforcement of what a model can do, what data it can access, or what actions it can take.

### 1.2 The Thesis

An operating system doesn't ask which process scheduler to start. It doesn't load the memory manager on demand. These are kernel services — always running, always available, mediating access to hardware on behalf of user-space programs.

Cortex applies this principle to inference. The L0 router is PID 1: the first process up, the last one down. It classifies every inbound request, selects the appropriate tier, and supervises execution. Tiers L0–L2 remain resident in memory like kernel modules. Higher tiers load on demand like user-space daemons. Remote frontier models (L7) are the network — a fallback when local resources are insufficient.

### 1.3 The Agent Microkernel

Beyond the inference scheduler, Cortex implements six kernel subsystems:

| Subsystem | OS Analogy | Module |
|-----------|-----------|--------|
| **Tier Routing** | Process scheduler | `router.py` |
| **Model Manager** | `systemd` | `model_manager.py` |
| **Persistent Memory** | Filesystem + `/proc` | `memory.py` |
| **Tool Registry** | Device driver interface + capabilities | `tools.py` |
| **Policy Engine** | SELinux / AppArmor | `policy.py` |
| **Resilience Layer** | Watchdog + circuit breakers | `resilience.py` |

Together these form a complete agent microkernel: any application can make inference syscalls, invoke tools, and operate within enforced permission boundaries — all through a single localhost endpoint.

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

### 2.3 Infinity Models: Dynamic Discovery

The core ladder is the *default*. It is not the limit. Cortex dynamically discovers every model installed in Ollama (or any compatible backend) and automatically classifies it:

1. **Parameter-based tier assignment.** A model's parameter count maps it to a tier: Phi-4 (14B) → L4, Llama 3.3 (70B) → L6, DeepSeek-R1 (7B) → L3.

2. **Family extraction.** The model's training family (llama, gemma, granite, phi, deepseek, mistral, olmo, smollm, etc.) is normalized and tracked.

3. **Automatic challenge enrollment.** Any discovered model from a *non-core* family at any tier becomes available as a challenger. You don't curate challengers — you pull models and they become challengers.

4. **Swarm pool expansion.** The swarm draws from all available models at a tier: core + curated challengers + discovered models. More models installed = wider cross-family coverage = higher confidence ceilings.

The model census is queryable at runtime via `GET /v1/models/census` and refreshable via `POST /v1/models/discover`. A system with 50 Ollama models has 50 potential participants in confidence verification — the challenge and swarm pools scale to infinity.

```
@cortex → discover [source: ollama, auto_classify: true, auto_enroll: true]
@discovery → result [models: 47, families: 9, tiers_covered: L0-L6]
@phi4:14b → classify [tier: L4, family: phi, role: challenger]
@deepseek-r1:7b → classify [tier: L3, family: deepseek, role: challenger]
@llama3.3:70b → classify [tier: L6, family: llama, role: challenger]
```

### 2.4 GGUF-First Format Strategy

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

If the selected tier exceeds what the hardware can run, the router clamps to the maximum feasible tier and attaches an **escalation hint** pointing to L7. Confidence is reduced proportionally to the gap between the ideal and actual tier. The policy engine may further clamp the tier: an app-specific `max_tier: L2` policy constrains that application to L0–L2 regardless of task complexity.

---

## 4. Confidence Verification

### 4.1 The Problem with Single-Model Inference

A single model's output has no external check. It may be fluent but wrong. Traditional approaches address this with temperature sampling or self-consistency (querying the same model multiple times), but these offer limited value: copies of the same model share the same blind spots.

### 4.2 Cross-Family Challenge

Cortex maintains a challenge model pool — models from families other than the core ladder (Llama, Gemma, Granite, Phi, OLMo, SmolLM, DeepSeek, Mistral, and any other discovered family). These models were trained on different data, with different architectures and different optimization objectives. When they agree with the core model, the probability of a shared hallucination is substantially lower than agreement among copies of the same family.

The challenge process:

1. The core model (Qwen) generates an answer.
2. If routing confidence is below 0.75, the Challenger selects a model from a different family at the same tier.
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

### 4.3 Challenge Model Sources

Challenge models come from two sources:

1. **Curated catalog.** Known-good models per tier, tested for compatibility:

| Tier | Curated Challenge Families |
|------|--------------------------|
| L0 | Llama 3.2, OLMo 2, Gemma 3 |
| L1 | Llama 3.2, Granite 3.3, SmolLM3 |
| L2 | Gemma 3, Phi-4 mini |
| L3 | Granite 3.3, Llama 3.1 |
| L4 | Gemma 3, Phi-4 |
| L5 | Granite 3.3, OLMo 2 |
| L6 | Llama 3.3 |

2. **Discovered models.** Any model installed in Ollama from a non-Qwen family is automatically added to the challenge pool at its corresponding tier. Running `ollama pull deepseek-r1:7b` instantly adds a DeepSeek challenger at L3.

---

## 5. Swarm Consensus

### 5.1 When Challenge Fails

If the challenger disagrees with the core model — or if initial confidence is very low — Cortex escalates to a **swarm**: a parallel fan-out to multiple models across multiple families.

| Difficulty | Swarm Size | Strategy |
|-----------|-----------|----------|
| Hard | 3–5 models | Small swarm, weighted vote |
| Hardest | All available | Large swarm, weighted consensus |

With infinity models, the "all available" swarm can include every installed model at a tier — core, curated challengers, and discovered models. A machine with 8 models at L3 produces an 8-way swarm with potentially 5+ independent families.

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

The key insight is that **cross-family agreement is a stronger signal than within-family agreement**. Three models from three different families converging on the same answer provides higher confidence than five copies of the same family agreeing — because independent training pipelines, data mixtures, and architectures make correlated failures less likely. The infinity model architecture maximizes this signal by automatically enrolling every installed model into the cross-family verification pipeline.

---

## 6. Escalation Pipeline

The full processing pipeline follows a deterministic escalation path:

```
User Request
    │
    ▼
┌─────────────┐
│ Policy Check │   Rate limit, permissions, tier cap
└──────┬──────┘
       │
  denied? ──→ 403 Forbidden (reason)
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
        │ Challenger  │   Different family verifies (curated + discovered)
        └──────┬─────┘
               │
          agree?
          ┌────┴────┐
          │ Yes     │ No
          ▼         ▼
       Return   ┌────────┐
                │ Swarm   │   All available models, weighted consensus
                └────┬───┘
                     │
               consensus?
               ┌────┴────┐
               │ Yes     │ No
               ▼         ▼
            Return   Escalate → L7 (remote frontier)
```

Each step is logged to the audit trail with the full escalation path. The resilience layer wraps each backend call with circuit breakers and retry logic; if a backend fails, the system automatically falls back to the next available model.

---

## 7. Tool Registry

### 7.1 The Device Driver Model

In a traditional OS, device drivers provide a controlled interface to hardware. Applications don't access the disk directly — they call `read()` and `write()` through the kernel, which enforces permissions.

Cortex applies the same principle to tool calling. When a model invokes a tool (function calling), the request passes through the Tool Registry, which:

1. **Validates** the tool exists and is enabled
2. **Checks permissions** against the privilege ring
3. **Enforces rate limits** per tool
4. **Executes** the tool in a controlled context
5. **Audits** every invocation

### 7.2 Permission Rings

Tools are classified into five privilege rings, from safest to most dangerous:

| Ring | Level | Description | Examples |
|------|-------|-------------|----------|
| 0 | `READ_ONLY` | No side effects. Info retrieval only. | `web_search`, `get_time`, `read_file` |
| 1 | `DRAFT` | Creates artifacts, doesn't execute them. | `code_interpreter` (sandbox), staged file writes |
| 2 | `EXECUTE` | Local side effects. Modifies local state. | `shell`, `write_file`, `git_commit` |
| 3 | `EXTERNAL` | External side effects. Network mutations. | `send_email`, `post_to_api`, `deploy` |
| 4 | `DANGEROUS` | Destructive / irreversible. | `delete_database`, `rm -rf`, `transfer_funds` |

Each request context has a `max_ring` (set by the policy engine). A tool call is allowed only if `tool.ring ≤ context.max_ring`. By default, contexts allow Ring 1 (DRAFT) — models can retrieve information and draft artifacts but cannot execute commands or make external calls without explicit policy escalation.

### 7.3 Built-in Tools

Cortex ships with six built-in tools:

| Tool | Ring | Description |
|------|------|-------------|
| `get_time` | 0 | Get current date/time |
| `calculate` | 0 | Evaluate math expressions |
| `read_file` | 0 | Read a file's contents |
| `web_search` | 0 | Search the web (pluggable provider) |
| `write_file` | 2 | Write content to a file |
| `shell` | 2 | Execute a shell command |

Additional tools can be registered at runtime via the API, and MCP (Model Context Protocol) server tools can be proxied through the registry.

---

## 8. Policy Engine

### 8.1 Hierarchical Scoping

Policies are scoped with a three-level hierarchy where the most specific scope wins:

```
thread:{id}  >  app:{id}  >  global
```

A policy set at the thread level overrides the same policy at the app level, which overrides the global default. This allows fine-grained control: a coding assistant app can operate at Ring 2 while a search-only app is constrained to Ring 0.

### 8.2 Policy Types

| Policy | Type | Default | Description |
|--------|------|---------|-------------|
| `max_tier` | string | `"L7"` | Maximum tier to route to |
| `max_ring` | int | `1` | Maximum tool permission ring |
| `cloud_allowed` | bool | `true` | Allow L7 cloud escalation |
| `local_only` | bool | `false` | Never escalate to cloud |
| `rate_limit` | int | `0` | Max requests per minute (0 = unlimited) |
| `max_tokens` | int | `8192` | Token budget cap per request |
| `blocked_tools` | list | `[]` | Tool names that are blocked |
| `allowed_models` | list | `[]` | Whitelist of model names (empty = all) |
| `require_approval` | bool | `false` | Require human approval for tool execution |
| `audit_level` | string | `"full"` | Logging level: `full`, `minimal`, `none` |

### 8.3 Enforcement

Every request passes through the policy engine before reaching the router. The engine:

1. Resolves the effective policy for each key (thread → app → global fallback)
2. Checks rate limits using a sliding-window counter
3. Validates the requested model against `allowed_models`
4. Blocks cloud escalation if `local_only` is set
5. Caps `max_tokens` to the policy maximum
6. Returns a `PolicyDecision` with allowed/denied status, effective values, and warnings

Denied requests receive a `403 Forbidden` response with a machine-readable reason.

---

## 9. Resilience

### 9.1 Circuit Breakers

Each backend (model or tier) gets its own circuit breaker with three states:

```
CLOSED ─── normal, requests flow through
   │
   ├── N consecutive failures
   ▼
OPEN ───── requests rejected immediately (fail-fast)
   │
   ├── after recovery_timeout
   ▼
HALF_OPEN ── one probe request allowed
   │
   ├── probe succeeds → CLOSED
   └── probe fails → OPEN
```

Default thresholds: 3 consecutive failures to open, 30-second recovery timeout, 1 successful probe to close.

### 9.2 Retry with Backoff

Failed calls are retried with exponential backoff:

- **Max retries**: 2 (3 total attempts)
- **Base delay**: 0.5s
- **Exponential factor**: 2× per attempt
- **Max delay**: 10s

Fallback calls use lighter retry (1 retry, 0.2s delay) to minimize cascading latency.

### 9.3 Tier Waterfall

When the primary backend for a request fails and its circuit is open, the resilience layer automatically tries fallback backends in order:

1. Same tier, different model (e.g., core → challenge model)
2. Adjacent lower tier (e.g., L4 → L3)
3. L7 remote frontier (if cloud_allowed)

The system never returns an error to the user if *any* backend is available. This mirrors how a well-configured OS handles disk failure: fall back to another disk, then to network storage, before reporting an error.

---

## 10. Persistent Memory

### 10.1 Storage Layer

All state that must survive a restart is stored in SQLite (`~/.cortex/cortex.db`) with WAL mode for concurrent reads:

- **Threads**: conversation state, message history, per-thread metadata
- **Audit log**: every request with routing decision, tier, model, confidence, latency, and escalation path (append-only)
- **KV cache index**: which prompt prefixes are cached, for prefix-cache reuse across requests
- **Policies**: runtime configuration and per-app overrides
- **Usage stats**: daily aggregates by tier and model

### 10.2 Context Management

Threads maintain conversation history. When a thread's context approaches the model's context window limit, Cortex trims messages from the beginning while preserving the system prompt and recent exchanges. Token counts are tracked per message for precise trimming.

---

## 11. System Layer

### 11.1 Hardware Detection

On boot, Cortex scans the host system:

- **CPU**: model, core count, ISA features (AVX2, AVX-512, NEON, AMX)
- **GPU**: NVIDIA (via `nvidia-smi`), AMD (via `rocm-smi`), Apple Silicon (via `sysctl`)
- **Memory**: total and available RAM; for Apple Silicon, unified memory with 75% GPU allocation
- **Backends**: which inference engines are installed (Ollama, llama.cpp, vLLM)

The result is a `SystemProfile` — a structured snapshot of the machine's capabilities. Tier feasibility is assessed against this profile: each tier's VRAM requirement is compared against the available budget (total VRAM minus a 20% OS reserve for discrete GPUs, or 75% of unified memory for Apple Silicon).

### 11.2 Model Manager

The Model Manager is the `systemd` of Cortex. It:

- **Boots** always-hot models (L0–L2) at startup
- **Discovers** all installed models and classifies them by tier and family
- **Loads** on-demand models when a tier is first requested
- **Evicts** models under VRAM pressure using LRU policy
- **Health-checks** loaded models and restarts failed backends
- **Tracks** state per model: `loading`, `ready`, `failed`, `evicted`

### 11.3 Backend Adapters

Cortex supports multiple inference backends through a unified adapter interface:

| Backend | Format | Platforms | Best For |
|---------|--------|-----------|----------|
| Ollama | GGUF | All | Easiest setup, automatic model management |
| llama.cpp | GGUF | All | Direct control, lowest overhead |
| vLLM | AWQ/GPTQ | Linux + NVIDIA | Prefix caching, high throughput |
| OpenAI API | API | Any (remote) | L7 frontier escalation |

The adapter interface exposes synchronous and asynchronous completion methods, abstracting away backend differences. Backend selection is automatic based on the detected hardware profile.

---

## 12. API Surface

### 12.1 The Daemon

Cortex runs as a local HTTP daemon on `localhost:11411`. Any application that supports `OPENAI_BASE_URL` can use it:

```bash
export OPENAI_BASE_URL=http://localhost:11411/v1
export OPENAI_API_KEY=local
```

### 12.2 Multi-Format Support

The daemon accepts three API formats and translates them to a unified internal representation:

| Format | Endpoint | Status |
|--------|----------|--------|
| OpenAI Chat Completions | `POST /v1/chat/completions` | Native (pass-through) |
| OpenAI Responses API | `POST /v1/responses` | Translated |
| Anthropic Messages API | `POST /v1/messages` | Translated |

### 12.3 Kernel Endpoints

Beyond inference, the daemon exposes the full kernel API:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Inference (OpenAI-compatible) |
| `/v1/models` | GET | List loaded models |
| `/v1/models/census` | GET | Full model census (core + challenge + discovered) |
| `/v1/models/discover` | POST | Re-scan for new models |
| `/v1/tools` | GET | List registered tools with permission rings |
| `/v1/policies` | GET/POST | Read/write policies |
| `/v1/threads` | GET | List conversation threads |
| `/v1/threads/{id}/messages` | GET | Get thread messages |
| `/v1/usage` | GET | Usage statistics |
| `/v1/audit` | GET | Audit log |
| `/v1/resilience` | GET | Circuit breaker status |
| `/v1/resilience/reset` | POST | Reset circuit breakers |
| `/health` | GET | Daemon health check |
| `/status` | GET | Full system status |

### 12.4 Compatible Clients

The following tools work with Cortex out of the box:

- **IDE assistants**: Cursor, VS Code Copilot, Cline, Continue, aider
- **SDKs**: OpenAI Python SDK, Anthropic Python SDK
- **Frameworks**: LangChain, LlamaIndex
- **UIs**: Open WebUI

---

## 13. Semantic Compression Language (SCL)

### 13.1 The Control Protocol

SCL is a compact, machine-parseable notation for expressing the semantic content of inference decisions, model coordination, and system state. It serves as the "assembly language" of the Cortex kernel — readable by both humans and machines, and dramatically more token-efficient than natural language or JSON.

### 13.2 Grammar

```
record     := anchor relation scope
anchor     := '@' IDENTIFIER
relation   := '→' VERB
scope      := '[' entries ']'
entries    := entry (',' entry)*
entry      := KEY ':' VALUE
document   := record ('\n' record)*
```

**Primitives:**
- `@` — **Anchor.** Entity, subject, or noun.
- `→` — **Relation.** Verb, transition, causality.
- `[ ]` — **Scope.** Bounded context frame. Key-value attributes.

### 13.3 Core Types

SCL is implemented as five frozen dataclasses in `src/scl/types.py`:

| Type | Description | Serialization |
|------|-------------|---------------|
| `Anchor` | `@name` — the entity being described | `.to_text()`, `.to_bytes()`, `.to_dict()` |
| `Relation` | `→ verb` — the action or transition | `.to_text()`, `.to_bytes()`, `.to_dict()` |
| `Scope` | `[k: v, ...]` — bounded context frame | `.to_text()`, `.to_bytes()` (JSON), `.to_dict()` |
| `SCLRecord` | One complete statement: anchor + relation + scope | All formats + `.content_hash()` |
| `SCLDocument` | Ordered collection of records with metadata | All formats + `.filter_by_anchor()`, `.filter_by_verb()` |

Every type supports four serialization round-trips: text ↔ type ↔ bytes ↔ dict. The binary format uses length-prefixed fields (`struct.pack(">H")`) for efficient prefix parsing. Content hashes (SHA-256) exclude timestamps, making them stable across time for deduplication.

`SCLRecord` carries two fields beyond the grammar:
- **`weight: float`** — used by swarm consensus to weight votes
- **`parent_id: Optional[str]`** — DAG linkage for gossip chains

### 13.4 Examples

A routing decision:
```scl
@task → classify [category: code, complexity: 0.45]
@router → select [tier: L3, confidence: 0.82]
```

A challenge result:
```scl
@core → answer [model: qwen3:8b, family: qwen]
@challenger → answer [model: granite3.3:8b, family: granite]
@agreement → evaluate [level: weak_agree, confidence: 0.65]
```

A policy enforcement:
```scl
@policy → deny [app: untrusted, reason: cloud_not_allowed, scope: global]
```

### 13.5 Semantic State Deltas

Instead of broadcasting full state, agents emit mutations — the semantic equivalent of `git diff`:

$$\Delta S_t = S_t \ominus S_{t-1}$$

Reconstruction is lossless:

$$S_t = S_0 \oplus \sum_{i=1}^{t} \Delta S_i$$

In SCL, a delta is simply a `mutate` record containing only the changed keys:

```scl
@agent_04 → mutate [status: idle, queue: job_42]
```

If a key isn't mentioned, the receiving agents assume the previous value persists. A full state snapshot (verbose, for bootstrapping new agents) uses the `snapshot` verb:

```scl
@agent_04 → snapshot [status: processing, target: database, memory: optimal, queue: empty]
```

**Delta streams** are the semantic event log. The stream supports:

- **Time-travel replay**: reconstruct any agent's state at any point by replaying deltas from the nearest checkpoint
- **Rollback**: rewind to $t=98$, isolate the rogue agent, patch, and replay
- **Compaction**: squash a range of deltas into a single checkpoint delta

**Merge conflict resolution** handles concurrent mutations from multiple agents using three strategies:

| Strategy | Mechanism | When to Use |
|----------|-----------|-------------|
| **LWW** (Last-Writer-Wins) | Latest timestamp wins | General purpose, low contention |
| **PRIORITY** | Highest agent weight wins, timestamp breaks ties | Supervisor overrides worker |
| **UNION** (CRDT G-Set) | Set-valued keys merge | Additive operations (model pools, capability sets) |

Vector clocks track causal ordering. When two deltas are concurrent (neither's vector clock dominates), the merge strategy resolves the conflict automatically. The `weight` field on SCLRecord doubles as the agent's priority in conflict resolution.

This architecture gives three superpowers at scale:

1. **Microscopic token footprint.** A delta might be 20 characters. Agents sync thousands of times per minute without hitting context-window bottlenecks.
2. **Time-travel debugging.** When a swarm of 10,000 agents hits a hallucination loop at $t=100$, roll back the delta stream to isolate the rogue agent.
3. **Event-sourced consensus.** Agents don't maintain a global state database. They append deltas to a shared log. Consensus = processing the same log in the same order.

### 13.6 Applications

- **Audit logs** in SCL are 3–5× more compact than JSON while remaining human-readable
- **Model manifests** are expressed as SCL documents and fingerprinted for deduplication
- **Agent coordination** uses SCL to describe task assignments, file ownership, and interface contracts (see `AGENTS.md`)
- **Braille encoding** of SCL records provides fixed-width fingerprints for content-addressing
- **Swarm votes** are SCLRecords with `weight` fields, enabling weighted consensus over structured data
- **State deltas** enable Git-like synchronization between agents at any scale
- **Gossip protocol** provides epidemic delta propagation across agent swarms

### 13.7 Gossip Protocol

At scale, agents don't talk to a central server. They gossip: each agent periodically picks a random peer and exchanges deltas. The protocol is fingerprint-first — a 4-character Braille hash is compared before any data is sent.

**Push-pull anti-entropy:**

```
Agent A → Agent B:  PING [fingerprint: ⢍⠱⡈⠦]
Agent B → Agent A:  fingerprints match? → ACK (0 bytes transferred)
                    fingerprints differ? → PUSH_DELTA [changed keys only]
Agent A → Agent B:  PUSH_DELTA [our changes they haven't seen]
```

Content-keyed deduplication ensures each unique delta is applied exactly once, regardless of how many peers relay it.

**Convergence properties** (measured):

| Swarm Size | Rounds to Converge | Total Deltas | Epidemic Factor |
|------------|-------------------|--------------|----------------|
| 5 agents | 1 | 18 | O(1) |
| 50 agents | 4 | ~200 | O(log N) |
| N agents | O(log N) | O(N) | Epidemic |

Convergence is detected by comparing content hashes — O(N) against a reference, not O(N²) pairwise. Once converged, subsequent gossip rounds cost zero data transfer (fingerprint hits).

**Swarm-level API** (`src/scl/gossip.py`):

| Component | Purpose |
|-----------|--------|
| `Peer` | One gossip-enabled agent with state, delta stream, and dedup |
| `Swarm` | Collection of peers with round management and convergence detection |
| `GossipMessage` | Typed message (PING, PONG, PUSH_DELTA, CONVERGED) |
| `convergence_matrix()` | Pairwise Braille fingerprint similarity for diagnostics |
| `to_scl_document()` | Export full swarm state as auditable SCL |

---

## 14. Braille Encoding Layer

### 14.1 The Codec

The 256 Unicode Braille characters (U+2800 to U+28FF) map bijectively to byte values 0x00–0xFF. Each character's codepoint offset from U+2800 *is* the byte value. This gives a natural, lossless encoding with fixed density: 1 byte = 1 Braille character.

```
0x00 → ⠀ (blank)    0x41 ('A') → ⡁    0xFF → ⣿ (all 8 dots raised)
```

The codec (`src/braille/codec.py`) provides:

| Function | Description |
|----------|-------------|
| `encode(bytes) → str` | Bytes to Braille string |
| `decode(str) → bytes` | Braille string back to bytes |
| `encode_hex(str) → str` | Hex string to Braille |
| `encode_int(n, width) → str` | Integer to fixed-width Braille |
| `decode_int(str) → int` | Braille back to integer |
| `is_braille(char) → bool` | Check if character is in U+2800–U+28FF |

Token efficiency: most LLM tokenizers treat Braille characters as single tokens or small multi-byte sequences, making this encoding competitive with or better than hex/base64 for transmitting binary data through language models.

### 14.2 Fingerprinting

The fingerprint module (`src/braille/fingerprint.py`) maps SCL records to fixed-width Braille hashes:

1. Serialize record to canonical bytes via `record.to_bytes()`
2. SHA-256 hash
3. Truncate to `width` bytes (default: 4 bytes = 32 bits)
4. Encode as Braille

Result: a 4-character Braille string that uniquely identifies the semantic content of any SCL record.

```python
fingerprint(SCLRecord(Anchor('router'), Relation('select'),
            Scope({'model': 'qwen3:4b'})))  # → '⢍⠱⡈⠦'
```

**Convergence checking** uses Hamming distance between fingerprints:

| Similarity | Interpretation | Action |
|------------|---------------|--------|
| > 0.9 | Agents agree | No action |
| 0.5–0.9 | Partial agreement | Verification needed |
| < 0.5 | Disagreement | Trigger challenger/swarm |

At scale, agents broadcast fingerprints (4 chars) instead of full state. Cluster-heads aggregate fingerprints for sub-swarms. This reduces gossip bandwidth by orders of magnitude.

### 14.3 Manifests

The manifest module (`src/braille/manifest.py`) encodes system and routing state as fixed-width Braille strings:

**Routing signature** (4 chars = 32 bits):
```
[tier:1][category:1][confidence:1][flags:1]
```

Every routing decision compresses to exactly 4 Braille characters — a complete audit entry in 12 UTF-8 bytes.

**System manifest** (8 chars = 64 bits):
```
[os:1][arch:1][accel:1][vram_gb:1][ram_gb:1][cores:1][max_tier:1][backends:1]
```

An entire hardware profile in 8 characters. Two machines can be compared by visual inspection of their manifests.

**Tier manifest** (6 chars = 48 bits):
```
[tier:1][param_min:1][param_max:1][vram_req:1][always_hot:1][feasible:1]
```

All manifests round-trip through `encode` / `decode` functions, providing both compact storage and human-readable Braille visualization.

---

## 15. Performance Characteristics

### 15.1 TTFT Optimization

Time-to-first-token is the critical latency metric. Cortex optimizes TTFT through:

1. **Right-sizing**: routing to the smallest capable model eliminates unnecessary computation
2. **Always-resident tiers**: L0–L2 are never cold-started
3. **Prefix caching**: common prompt prefixes are cached and reused (tracked in the KV cache index)
4. **Quantization**: 4-bit quantization (Q4_K_M) reduces memory footprint and speeds inference
5. **Thinking suppression**: for reflex tiers (L0–L2), Qwen3's thinking mode is suppressed via `/no_think` to avoid wasting tokens on internal reasoning

### 15.2 Platform-Specific Performance

| Platform | GPU | Backend | Expected TTFT |
|----------|-----|---------|---------------|
| Linux | NVIDIA H100 | vLLM (AWQ) | 20–50ms |
| Linux | NVIDIA RTX 4090 | vLLM / llama.cpp | 30–80ms |
| macOS | Apple M1–M4 | llama.cpp (Metal) | 50–150ms |
| Linux | AMD RX 7900 | llama.cpp (ROCm) | 40–100ms |
| Any | CPU only | llama.cpp (AVX2/NEON) | 200–2000ms |

### 15.3 Graceful Degradation

Cortex adapts to available hardware:

- **192 GB Apple Silicon**: all tiers L0–L6 fit concurrently
- **24 GB NVIDIA GPU**: L0–L4 comfortably, L5 with eviction
- **8 GB RAM, no GPU**: L0–L2 only, with L7 as fallback
- **No backends installed**: graceful failure with clear diagnostics

The system always functions at whatever level the hardware supports, even if that is only L0–L2 — analogous to a minimal `initramfs` boot.

---

## 16. Design Principles

1. **Start small.** Route to the smallest model likely to handle the task. Escalate only on evidence of insufficient capability.

2. **Measure confidence, don't assume it.** Cross-family verification provides an empirical signal that single-model inference lacks.

3. **One format everywhere.** GGUF runs on every platform. Eliminate format fragmentation.

4. **Always available.** The core inference tiers (L0–L2) are kernel modules, not applications. They are loaded at boot and never evicted.

5. **Degrade gracefully.** If hardware supports only L0–L2, the system still works. L7 is the network fallback.

6. **Audit everything.** Every request, routing decision, escalation, and latency measurement is logged for post-hoc analysis and continuous improvement.

7. **Infinity models.** Don't curate — discover. Every model the user installs becomes a participant in the confidence verification pipeline. The kernel adapts to what's available.

8. **Least privilege.** Tool access follows permission rings. No model gets more capability than its context requires. The default is DRAFT (Ring 1): models can observe and compose, but not execute.

9. **Policy over code.** Runtime behavior is controlled by policies, not hardcoded rules. Per-app, per-thread scoping allows different applications to operate under different constraints without code changes.

---

## 17. Future Work

- **LLM-as-judge aggregation**: replace text-overlap comparison in the swarm with a small judge model for more nuanced agreement detection
- **Persistent credibility scoring**: track which models consistently agree with eventual consensus and weight their votes accordingly over time
- **Embedding-based semantic memory**: vector search over conversation history for long-term context recall
- **Adaptive routing**: use audit log data to fine-tune the L0 router based on observed outcomes
- **Multi-GPU scheduling**: distribute concurrent tiers across multiple GPUs with VRAM-aware placement
- **Hardware-specific kernel tuning**: automatic selection of flash attention, paged attention, and speculative decoding based on detected GPU architecture
- **SCL native audit format**: wire SCL documents and Braille routing signatures into the audit log pipeline, replacing JSON entries
- **BFI integration**: bridge the Braille Infinity Token format (from `semantic-compression-language`) with SCLRecords, enabling emotional and morphological channels on SCL state atoms
- **MCP tool proxy**: bridge Model Context Protocol servers through the tool registry with automatic permission classification
- **Self-hosting coordination**: use Cortex itself to coordinate parallel agent work, with SCL manifests per task and Braille fingerprints for deduplication
- **Delta-aware swarm consensus**: integrate gossip convergence detection into the existing challenger/swarm pipeline so model deliberation uses delta streams natively
- **Semantic merge CRDTs**: extend the UNION merge strategy with full CRDT semantics (OR-Set, LWW-Map) for richer conflict-free replication across agent clusters

---

## 18. Conclusion

Cortex reframes LLM inference as an operating system problem. Instead of treating models as isolated applications, it treats them as a tiered hierarchy of kernel services — always available, resource-managed, and self-verifying. The L0 router acts as PID 1, the challenge system provides cross-family confidence checks, and the swarm offers consensus for the hardest problems.

The infinity model architecture means the system grows stronger as more models are installed — every new model is a new voice in the cross-family verification choir. The tool registry and policy engine provide the permission model that multi-agent systems have lacked: fine-grained, scoped, auditable control over what AI can do. The resilience layer ensures the system never fails silently — it retries, falls back, and degrades gracefully.

SCL and Braille encoding provide the kernel's native data format: structured, hashable, gossip-ready state atoms that compress routing decisions into 4-character fingerprints and hardware profiles into 8-character manifests. Semantic state deltas — the `git diff` for AI thoughts — mean agents broadcast only their mutations, not their entire world-view. With vector clocks for causal ordering, CRDT-style merge resolution for concurrent writes, and delta streams for time-travel debugging, the architecture scales to an infinity of agents the same way Git scales to an infinity of developers: through minimal, composable, conflict-resolvable patches to shared semantic state.

The result is an inference layer that is faster (by routing to the smallest capable model), more reliable (by measuring confidence empirically across model families), more secure (by enforcing permission rings and policies), and universally deployable (by standardizing on GGUF and discovering whatever models are available).

The question is not whether AI should be integrated into the operating system. It is whether the operating system should be redesigned around AI. Cortex is an argument that it should.

---

*Cortex is open source under the MIT License.*
*Repository: github.com/elevate-foundry/cortex*
