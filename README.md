# Cortex

**AI as PID 1.** An AI-native operating system kernel where the inference engine isn't an app — it's `init`.

Cortex is three things:

1. **An inference runtime** — detects your hardware, builds a tiered model hierarchy (L0–L7), and routes every request to the smallest model that can handle it.
2. **A semantic protocol** — SCL (Semantic Compression Language) gives every routing decision, agent state, and audit event a machine-parseable, human-readable representation.
3. **A self-hosting system** — the coordination protocol used to build Cortex is written in the language Cortex is building. The system builds itself with itself.

---

## Quick Start

```bash
# Detect your hardware and show what Cortex can run
python -m src detect

# Show which model tiers fit on your system
python -m src tiers

# Route a prompt to the appropriate tier
python -m src route "refactor the database layer to use connection pooling"

# Launch the optimal inference server
python -m src serve

# Simulate different hardware profiles
python -m src detect --simulate linux-h100
python -m src tiers --simulate mac-m4-ultra
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
├── cortex.py           # Top-level orchestrator (PID 1)
├── router.py           # L0 request classifier
├── tiers.py            # L0–L7 tier specs and model catalogs
├── challenger.py       # Cross-family verification
├── swarm.py            # Multi-model consensus
├── model_manager.py    # Model lifecycle (load/evict/health-check)
├── backend_adapter.py  # Unified interface: Ollama, llama.cpp, vLLM
├── daemon.py           # HTTP proxy (localhost:11411)
├── memory.py           # SQLite persistence (threads, audit, KV cache)
├── policy.py           # Per-app/thread policy engine
├── tools.py            # Tool registry with permission rings
├── resilience.py       # Circuit breaker + retry
├── hardware_detect.py  # CPU/GPU/RAM/backend detection
├── api_adapter.py      # OpenAI/Anthropic/Responses API translation
├── scl/
│   ├── types.py        # Anchor, Relation, Scope, SCLRecord, SCLDocument
│   ├── parser.py       # Text → SCL AST (strict/lenient)
│   ├── emitter.py      # SCL AST → text (compact/aligned/table)
│   ├── grammar.py      # Formal BNF specification
│   ├── delta.py        # Semantic state deltas, vector clocks, merge
│   └── cortex_bridge.py # Cortex types ↔ SCL conversion
└── braille/
    ├── codec.py        # Bijective byte ↔ Braille encoding
    ├── fingerprint.py  # SCL → fixed-width Braille hash (LSH)
    └── manifest.py     # Routing/tier/system manifests in Braille

tests/                  # 108 tests across 4 test files
AGENTS.md               # Multi-agent coordination plan (in SCL)
WHITEPAPER.md           # Full technical paper
```

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
- At least one inference backend installed (Ollama is easiest to start with)

## Documentation

- **[WHITEPAPER.md](WHITEPAPER.md)** — full technical paper
- **[AGENTS.md](AGENTS.md)** — multi-agent coordination plan (written in SCL)

## License

MIT
