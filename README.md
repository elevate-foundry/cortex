# Cortex

**AI as PID 1.** An AI-native operating system kernel where the inference engine isn't an app — it's `init`.

Cortex is four things:

1. **An inference runtime** — detects your hardware, builds a tiered model hierarchy (L0–L7), and routes every request to the smallest model that can handle it.
2. **A semantic protocol** — SCL (Semantic Compression Language) gives every routing decision, agent state, and audit event a machine-parseable, human-readable representation.
3. **A self-modifying system** — observes its own performance and mutates routing policy, boot config, and model weights to improve over time.
4. **A self-hosting system** — the coordination protocol used to build Cortex is written in the language Cortex is building. The system builds itself with itself.

---

## Quick Start

```bash
# Detect your hardware and show what Cortex can run
python -m src detect

# Show which model tiers fit on your system
python -m src tiers

# Route a prompt to the appropriate tier
python -m src route "refactor the database layer to use connection pooling"

# Start the daemon (OpenAI-compatible proxy on localhost:11411)
python -m src daemon

# Simulate different hardware profiles
python -m src detect --simulate linux-h100
python -m src tiers --simulate mac-m4-ultra

# Self-training: train, eval, promote the Cortex Kernel Model
python -m src train --time-budget 10m
python -m src train --status
python -m src train --rollback

# Multi-host gossip protocol
python -m src gossip add --id node-02 --url http://192.168.1.42:11411
python -m src gossip list
python -m src gossip state
```

---

## Architecture

### Layer 1: Inference Runtime

Cortex treats LLM inference the way a traditional OS treats process scheduling.

| Traditional OS | Cortex |
|----------------|--------|
| `init` / `systemd` (PID 1) | **L0 Router** — always running, first thing up, last thing down |
| Kernel syscall interface | **L1 Agent** — thin wrapper over tool calls, always hot |
| Shell / window manager | **L2 Agent** — always-hot local agent, user-facing |
| User-space daemons | **L3–L6** — loaded on demand, handle real work |
| Remote API calls | **L7** — escalation to frontier when local fails |

| Tier | Size | Always Hot? | TTFT Target |
|------|------|-------------|-------------|
| L0 | 0.5–1B | Yes | ~10ms |
| L1 | 1–2B | Yes | ~20ms |
| L2 | 3–4B | Yes | ~40ms |
| L3 | 7–8B | No | ~60ms |
| L4 | 12–14B | No | ~100ms |
| L5 | 30–32B | No | ~200ms |
| L6 | 64–70B | No | ~400ms |
| L7 | Frontier | No | ~500ms |

Confidence is measured, not assumed. Cross-family verification (Qwen vs Llama vs Gemma vs Granite vs Phi) catches correlated hallucinations that single-model inference misses.

### Layer 2: Semantic Compression Language (SCL)

Every event in Cortex — routing decisions, challenge results, agent state, audit entries — is expressed in SCL:

```
@router → select [tier: L3, model: qwen3:8b, confidence: 0.82]
@challenger → verify [model: granite3.3:8b, agreement: weak_agree]
@consensus → resolve [confidence: 0.91, families: 3]
```

**Grammar:**
```
record   := '@' ANCHOR '→' VERB '[' key: value, ... ']'
document := record ('\n' record)*
```

Three primitives — `@` (anchor/entity), `→` (relation/verb), `[]` (scope/context) — compose into records, documents, manifests, and fingerprints.

SCL is not just a log format. It is the **wire protocol** for multi-agent coordination. Agents parse it, emit it, diff it, and fingerprint it.

### Layer 3: Braille Encoding

The 256 Unicode Braille characters (U+2800–U+28FF) map 1:1 to byte values. This gives a natural, lossless, bijective encoding:

```python
from src.braille import encode, decode
encode(b'\xde\xad\xbe\xef')  # → '⣞⢭⢾⣯'
decode('⣞⢭⢾⣯')              # → b'\xde\xad\xbe\xef'
```

Braille fingerprints compress any SCL record into a fixed-width hash:

```python
from src.braille.fingerprint import fingerprint, similarity
fp1 = fingerprint(record_a)  # → '⣦⠷⡓⠠' (4 chars = 32 bits)
fp2 = fingerprint(record_b)
similarity(fp1, fp2)         # → 0.469 (Hamming-based)
```

At scale, agents broadcast fingerprints instead of full state. Hamming distance estimates semantic divergence in O(1).

### Layer 4: Semantic State Deltas

Agents don't broadcast full context windows. They emit **mutations**:

```
ΔS_t = S_t ⊖ S_{t-1}       # compute what changed
S_t  = S_0 ⊕ Σ(ΔS_i)       # reconstruct any state from deltas
```

```python
from src.scl.delta import diff, apply_delta, DeltaStream

# Agent changes one key in a 20-key state → delta is microscopic
delta = diff(old_state, new_state, agent_id="agent_04")
# @agent_04 → mutate [status: idle, queue: job_42]

# Time-travel: reconstruct state at any point
stream = DeltaStream()
stream.append(delta_1)
stream.append(delta_2)
state_at_t1 = stream.state_at(1)  # reconstruct historical state
stream.rollback(1)                 # discard rogue mutations
```

Merge conflicts between concurrent agents are resolved by:
- **LWW** — last timestamp wins (default)
- **Authority** — highest-weight agent wins (conductor > worker)
- **CRDT** — set-valued keys auto-merge (G-Set union)
- **Escalation** — unresolvable conflicts defer to the conductor

### Layer 5: Gossip Protocol

Multi-host coordination via epidemic delta propagation:

```python
from src.scl.gossip import Swarm, Peer

swarm = Swarm(convergence_threshold=0.95)
swarm.add_peer("node_01", initial_state=state_a)
swarm.add_peer("node_02", initial_state=state_b)
rounds = swarm.run_until_converged()  # O(log N) rounds
```

Protocol: fingerprint-first comparison → push/pull only divergent deltas → CRDT merge. Bandwidth is O(1) when states match, O(k) when k keys differ.

Over HTTP, the daemon exposes `/v1/gossip` endpoints for real multi-host sync.

### Layer 6: Executable SCL (Rule Engine)

SCL is not just data — it's a programming language. The evaluator turns SCL records into executable rules:

```python
from src.scl.eval import RuleEngine, Rule, Condition, Action, CompareOp, ActionType

engine = RuleEngine()
engine.add_rule(Rule(
    name="escalate_complex",
    conditions=[Condition("complexity", CompareOp.GT, "0.8")],
    actions=[Action(ActionType.ESCALATE, target="L5")],
))

# Self-modification: agents rewrite their own rules via SCL
engine.process_meta(SCLRecord.from_text(
    '@triage → define [fn: classify, body: "@task → route [tier: $tier]"]'
))
```

Rules support conditions (==, !=, >, <, in, regex), compound logic (∧, ∨, ¬), variable binding ($var substitution), function definitions (λ abstraction), and self-modification (agents rewrite their own rules via deltas gossiped across the swarm).

---

## Daemon

The daemon (`python -m src daemon`) is an OpenAI-compatible HTTP proxy on `localhost:11411`:

```bash
# Chat completion (routes automatically)
curl http://localhost:11411/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "auto", "messages": [{"role": "user", "content": "hello"}]}'

# Streaming
curl http://localhost:11411/v1/chat/completions \
  -d '{"model": "auto", "stream": true, "messages": [...]}'

# Status
curl http://localhost:11411/v1/status
```

Background tasks running inside the daemon:
- **Policy Rewriter** — analyzes routing feedback every 5 minutes, proposes and applies policy mutations
- **Boot Telemetry** — tracks hardware fingerprints, boot timing, and optimal configs
- **Gossip Sync** — periodic delta exchange with known peers
- **SCL Audit** — logs every routing decision as fingerprinted SCL documents

---

## Self-Modification (Cortex-Mutate)

Cortex observes its own performance and rewrites its configuration:

```
@cortex → observe [request, hardware, latency, accuracy]   # steady state
@cortex → mutate [tier_specs, model_weights, boot_config]   # self-modification
```

**Policy Rewriter** (`src/policy_rewriter.py`):
- Runs as a background daemon task every 5 minutes
- Analyzes per-tier and per-model accuracy from routing feedback
- Proposes `MutationProposal` objects (tier demotion, model penalty)
- Auto-applies if confidence > 0.7; records full audit trail before application
- Rollback always available via mutation history

**Boot Self-Modification** (`src/boot_telemetry.py`):
- Logs hardware fingerprint + boot timing each boot
- Background optimizer proposes mutations to thread count, GPU layers, context window
- Same USB stick on different machines → different optimal configs
- The stick gets smarter with each boot

Safety boundaries:
- Minimum 10 feedback samples before any mutation
- Confidence gating (only apply if confidence > 0.7)
- Every mutation audited before applied
- Rollback capability preserved indefinitely

---

## CKM: Cortex Kernel Model (Self-Training)

The Cortex Kernel Model is a tiny (1M–60M param) transformer trained from scratch to speak SCL. It replaces all heuristics with learned inference:

```bash
# Full self-training loop
python -m src train --time-budget 10m

# Evaluate a checkpoint against the eval gate
python -m src train --eval /path/to/checkpoint.pt

# Show model registry status
python -m src train --status

# Rollback to previous model version
python -m src train --rollback
```

**Training pipeline:**
1. Hardware profiling → selects largest model that fits (ckm-1m through ckm-60m)
2. Synthetic data generation (boot configs, routing decisions, policy mutations, operational traces)
3. From-scratch GPT-style transformer (RMSNorm, RoPE, SwiGLU, GQA)
4. Eval gate: 99% SCL validity, 100% safety denial, 95% verb choice, 90% config completeness
5. Promote/reject/rollback with full SCL lifecycle emission

**Two planned model specializations:**
- **CortexRouter** — routes requests to tiers (<5ms, every request, observe-only authority)
- **CortexMutate** — proposes policy/config mutations (background, high-risk, safety-gated)

**Safety policy** (`src/ckm/policy/`):
- `dangerous_targets.scl` — 9 blocked paths (/dev/mem, /dev/kmem, etc.)
- `allowed_verbs.scl` — verb allowlist/blocklist
- Model has recommendation authority only — never raw execution

---

## The Self-Hosting Loop

```
AGENTS.md (written in SCL)
    ↓  parsed by
src/scl/parser.py (built by agents following AGENTS.md)
    ↓  fingerprinted by
src/braille/fingerprint.py (built by agents following AGENTS.md)
    ↓  tracked by
src/scl/delta.py (tracks mutations to the code being built)
    ↓  coordinated by
Cortex router (routes tasks to models the way it routes prompts)
    ↓  described in
AGENTS.md
```

The coordination protocol for building Cortex **is** Cortex's semantic substrate. The parser can parse its own coordination file. The fingerprinter can hash its own build artifacts. The delta system can track its own development history.

This is not a metaphor. It is the architecture.

---

## Project Structure

```
src/
├── cortex.py             # Top-level orchestrator (route → generate → challenge → swarm)
├── router.py             # L0 request classifier (heuristic + model-based)
├── tiers.py              # L0–L7 tier specs, model catalogs, feasibility assessment
├── challenger.py         # Cross-family verification (Qwen vs Llama vs Gemma vs Granite)
├── swarm.py              # Multi-model consensus with weighted voting
├── model_manager.py      # Model lifecycle (load/evict/health-check/VRAM budget)
├── backend_adapter.py    # Unified interface: Ollama, llama.cpp, vLLM, OpenAI, Anthropic
├── backend_selector.py   # Automatic backend detection and selection
├── daemon.py             # HTTP proxy (localhost:11411), OpenAI-compatible API
├── memory.py             # SQLite persistence (threads, audit, KV cache, WAL mode)
├── policy.py             # Per-app/thread policy engine (rate limits, model blocklists)
├── policy_rewriter.py    # Self-modifying policy: feedback → mutation proposals → apply
├── tools.py              # Tool registry with permission rings (ring 0–3)
├── resilience.py         # Circuit breaker + exponential backoff retry
├── hardware_detect.py    # CPU/GPU/RAM/backend detection (CUDA, MPS, ROCm, CPU)
├── api_adapter.py        # OpenAI/Anthropic/Responses API format translation
├── micro_engine.py       # CKM inference engine with 4-phase safety guardrail
├── boot_telemetry.py     # Boot timing, hardware fingerprint, config optimization
├── gossip_transport.py   # HTTP gossip transport (push/pull deltas between hosts)
├── gossip_discovery.py   # Peer discovery and management
├── boot_gossip_bridge.py # Bridge between boot telemetry and gossip layer
├── network_watcher.py    # Network state monitoring
├── lifecycle_scl.py      # SCL emission for system lifecycle events
├── config.py             # Global configuration
├── scl/
│   ├── types.py          # Anchor, Relation, Scope, SCLRecord, SCLDocument
│   ├── parser.py         # Text → SCL AST (strict/lenient modes)
│   ├── emitter.py        # SCL AST → text (compact/aligned/table)
│   ├── grammar.py        # Formal BNF specification
│   ├── delta.py          # Semantic state deltas, vector clocks, merge strategies
│   ├── gossip.py         # Epidemic gossip protocol (Peer, Swarm, convergence)
│   ├── eval.py           # Rule engine: conditions → actions, self-modifying rules
│   ├── cortex_bridge.py  # Cortex runtime types ↔ SCL conversion
│   ├── audit.py          # SCL audit document + fingerprint generation
│   ├── ontology.py       # Typed entity/relation/transition ontology, CKM task types
│   └── intent.py         # Intent classification from SCL records
├── braille/
│   ├── codec.py          # Bijective byte ↔ Braille encoding (U+2800–U+28FF)
│   ├── fingerprint.py    # SCL → fixed-width Braille hash (LSH, Hamming similarity)
│   └── manifest.py       # Routing/tier/system manifests in Braille
└── ckm/
    ├── cli.py            # CLI: cortex train {run, eval, rollback, status}
    ├── data_generator.py # Boot/routing/policy training pair generation
    ├── trace_generator.py# Full operational trace corpus (boot, route, debug, gossip)
    ├── dataset.py        # JSONL → curriculum-ordered → tokenized mmap (BPE, 512 vocab)
    ├── train.py          # Fine-tuning pipeline (LoRA on Qwen2.5-0.3B / SmolLM2)
    ├── train_scratch.py  # From-scratch GPT transformer (RMSNorm, RoPE, SwiGLU, GQA)
    ├── eval.py           # Eval gate: SCL validity, safety, verb choice, completeness
    ├── promote.py        # Model registry: stage, promote, rollback, version management
    ├── profile.py        # Hardware profiler, model ladder selection
    ├── simulator.py      # Training simulation and validation
    ├── world_model.py    # World model for CKM training
    ├── modal_train.py    # Modal.com cloud training integration
    └── policy/           # dangerous_targets.scl, allowed_verbs.scl

tests/                    # Unit + integration tests
├── test_scl_parser.py    # SCL parser round-trip and edge cases
├── test_scl_delta.py     # Delta, vector clock, merge, stream tests
├── test_scl_intent.py    # Intent classification tests
├── test_braille_codec.py # Braille encoding/decoding tests
├── test_braille_fingerprint.py  # Fingerprint + similarity tests
├── test_gossip.py        # Gossip protocol convergence tests
├── test_ckm_external.py  # CKM end-to-end training tests
├── test_ckm_inference.py # CKM inference pipeline tests
├── test_reboot_loop.sh   # Multi-boot self-modification integration test
└── conftest.py           # Shared fixtures

AGENTS.md                 # Multi-agent coordination plan (in SCL)
WHITEPAPER.md             # Full technical paper
```

---

## Supported Platforms

| Platform | GPU | Best Backend | Expected TTFT |
|----------|-----|-------------|---------------|
| Linux | NVIDIA A100/H100 | vLLM (AWQ) | 20–50ms |
| Linux | NVIDIA RTX 4090 | vLLM / llama.cpp | 30–80ms |
| macOS | Apple M1–M4 | llama.cpp (Metal) | 50–150ms |
| Linux | AMD RX 7900 | llama.cpp (ROCm) | 40–100ms |
| Any | CPU only | llama.cpp (AVX2/NEON) | 200–2000ms |

## Requirements

- Python 3.10+
- PyTorch (for CKM self-training; optional for inference-only use)
- At least one inference backend installed (Ollama is easiest to start with)

## Documentation

- **[WHITEPAPER.md](WHITEPAPER.md)** — full technical paper
- **[AGENTS.md](AGENTS.md)** — multi-agent coordination plan (written in SCL)
- **[boot/ARCHITECTURE.md](boot/ARCHITECTURE.md)** — v3 architecture vision (three pillars)

## License

MIT
