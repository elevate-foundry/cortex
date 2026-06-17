# Cortex — Internal Design Context

## Architecture: AI as PID 1

In Cortex, the inference engine is not an app — it's **init/systemd-equivalent**, a kernel-adjacent service that mediates between the user and everything else.

### Analogy to traditional OS

| Traditional OS | Cortex |
|----------------|--------------|
| `init` / `systemd` (PID 1) | **L0 router** — always running, first thing up, last thing down |
| Kernel syscall interface | **L1 agent** — thin wrapper over tool calls, always hot |
| Shell / window manager | **L2 agent** — always-hot local agent, user-facing |
| User-space daemons | **L3–L5** — loaded on demand, handle real work |
| Remote API calls | **L7** — escalation to frontier when local fails |

### Design implications

- **Boot sequence**: L0 loads first (< 500MB), gates everything. No user request executes without L0 classifying it.
- **Always-resident**: L0–L2 stay in memory like kernel modules. They never swap out.
- **Process supervision**: L0 monitors L1–L6 the way systemd monitors services — restarts, health checks, timeout kills.
- **Capability hierarchy**: Higher tiers require explicit escalation, like privilege elevation (`sudo`). L0 decides if the task warrants the cost.
- **Resource accounting**: VRAM budget is the new memory management. The tier system is the scheduler — deciding what's loaded, what's evicted, what runs concurrently.
- **Graceful degradation**: If hardware can only support L0–L2, the system still functions (like a minimal initramfs). L7 is the network fallback.

## Core Ladder (Qwen3 family)

| Size | Model |
|------|-------|
| 0.6B | Qwen3 |
| 1.7B | Qwen3 |
| 4B   | Qwen3 |
| 8B   | Qwen3 |
| 14B  | Qwen3 |
| 30B  | Qwen3-30B-A3B |
| 32B  | Qwen3-32B |

## Challenge Models

| Size | Candidates |
|------|------------|
| 1B     | Llama 3.2 / OLMo 2 / Gemma 3 |
| 3B     | Llama 3.2 / Granite / SmolLM3 / Phi-class |
| 4B     | Qwen3 / Gemma / Phi |
| 8B     | Qwen3 / Granite / Llama |
| 12-14B | Qwen3 / Gemma / Phi |
| 30-32B | Qwen3 / Granite / OLMo |
| 70B    | Llama-class or Qwen-class if hardware allows |

## Confidence & Swarm Strategy

Cross-family agreement increases confidence — if models from different families (Qwen, Llama, Gemma, Granite, Phi, OLMo) converge on the same answer, trust is higher than N copies of the same family agreeing.

| Difficulty | Strategy | Description |
|------------|----------|-------------|
| Easy       | Single model | One core-ladder model answers; fast, cheap |
| Medium     | Verify   | Core model answers, one challenge model from a different family confirms |
| Hard       | Swarm    | Fan out to 3-5 models across multiple families, majority-vote or judge |
| Hardest    | Large swarm | Fan out to many models across all available families + sizes, weighted consensus |

### Routing logic

1. **Start small** — Route to the smallest core-ladder model likely to handle the task.
2. **Confidence check** — If the model's confidence is low or the task is flagged as hard, escalate.
3. **Cross-family challenge** — Query a challenge model from a *different* family at a similar size.
4. **Agreement → done** — If they agree, return with high confidence.
5. **Disagreement → swarm** — Fan out to more families. More disagreement → larger swarm.
6. **Hardest problems** — Large swarm across all available sizes and families; use weighted voting (larger models get more weight) or an LLM-as-judge step.

## Implementation Modules

| Module | File | Role |
|--------|------|------|
| Hardware Detection | `hardware_detect.py` | Detect CPU, GPU, RAM, backends |
| Tier System | `tiers.py` | L0–L7 specs, model catalogs (core + challenge), feasibility |
| Router | `router.py` | Classify requests → pick tier (heuristic or L0 model) |
| Backend Adapter | `backend_adapter.py` | Unified interface over Ollama / llama.cpp / vLLM / OpenAI API |
| API Adapter | `api_adapter.py` | Translate Responses API / Anthropic / multimodal → Chat Completions |
| Model Manager | `model_manager.py` | Systemd for models: boot, load, evict, health-check, VRAM budget |
| Challenger | `challenger.py` | Cross-family verification: compare core vs challenge model answers |
| Swarm | `swarm.py` | Fan-out to N models, cluster votes, weighted consensus |
| Cortex | `cortex.py` | Top-level orchestrator: Router → Manager → Challenger → Swarm |
| Backend Selector | `backend_selector.py` | Pick optimal backend + config for a system profile |
| Daemon | `daemon.py` | HTTP proxy server: accepts all API formats, routes through Cortex |

## API Compatibility Layer

Apps using OpenAI-compatible Chat Completions work natively. Apps using newer or provider-specific APIs need translation. The `api_adapter.py` module handles this.

### Compatibility matrix

| API Format | Endpoint | Status | Strategy |
|------------|----------|--------|----------|
| OpenAI Chat Completions | `POST /v1/chat/completions` | Native | Pass-through |
| OpenAI Responses API | `POST /v1/responses` | Translated | `input[]` → `messages[]`, `output[]` → `choices[]` |
| Anthropic Messages | `POST /v1/messages` | Translated | `content[]` blocks → text, `system` string → system message |
| Tool calling | (any endpoint) | Pass-through | `tools[]` forwarded to backend |
| Multimodal (vision) | (any endpoint) | Text-extracted | Images described as `[image: ...]` for text-only models |
| Multimodal (audio) | (any endpoint) | Text-extracted | Audio described as `[audio content]` |
| Built-in tools | `POST /v1/responses` | Noted/stripped | `web_search`, `code_interpreter`, `computer_use` logged but not executed locally |
| Computer-use | (any endpoint) | Stripped | Not supported locally, noted in logs |
| MCP tool refs | `POST /v1/responses` | Stripped | MCP server references logged |

### Translation flow

```
Client Request (any format)
    │
    ▼
┌─────────────────┐
│  API Adapter     │  normalize_request()
│  (api_adapter.py)│  detect format from path + body
└────────┬────────┘
         │  NormalizedRequest (messages[] + tools[])
         ▼
┌─────────────────┐
│  Cortex Pipeline │  route → generate → challenge → swarm
└────────┬────────┘
         │  CompletionResponse
         ▼
┌─────────────────┐
│  API Adapter     │  format_response()
│  (api_adapter.py)│  convert back to client's format
└────────┬────────┘
         │
         ▼
Client Response (matching format)
```

### What works out of the box

- **Cursor, VS Code, Cline, aider**: Use Chat Completions → native
- **OpenAI Python SDK (responses)**: Uses Responses API → translated
- **Anthropic Python SDK**: Uses Messages API → translated
- **LangChain, LlamaIndex**: Use Chat Completions → native
- **Open WebUI, Continue**: Use Chat Completions → native

### Pipeline flow

```
User Request
    │
    ▼
┌─────────┐   classify    ┌──────────┐
│  Router  │──────────────▶│ Tier Lx  │
│  (L0)    │               └────┬─────┘
└─────────┘                     │
                                ▼
                    ┌───────────────────┐
                    │ Core Model (Qwen3)│
                    │   generates answer│
                    └────────┬──────────┘
                             │
                    confidence < 0.75?
                      ┌──────┴──────┐
                      │ no          │ yes
                      ▼             ▼
                   Return    ┌────────────┐
                             │ Challenger  │
                             │ (diff family)│
                             └──────┬─────┘
                                    │
                              agree?
                         ┌──────┴──────┐
                         │ yes         │ no
                         ▼             ▼
                      Return    ┌──────────┐
                                │  Swarm   │
                                │ (3-5+ models)│
                                └──────┬───┘
                                       │
                                 consensus?
                            ┌──────┴──────┐
                            │ yes         │ no
                            ▼             ▼
                         Return    Escalate to L7
```
