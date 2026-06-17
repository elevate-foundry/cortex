# Cortex + SCL/Braille — Parallel Agent Work Plan

## Current State

Cortex is a working AI runtime with 20 source files (~280KB):

```
DONE: hardware_detect, tiers, router, backend_adapter, backend_selector,
      model_manager, challenger, swarm, cortex, daemon, api_adapter,
      memory, policy, tools, resilience, config, main
MISSING: tests, SCL, Braille, CLI polish, docs
```

No test suite exists. No SCL grammar. No Braille encoding. The runtime works but is unverified and has no semantic control layer.

---

## Architecture of the Work

Three agents. Zero file conflicts. Each agent owns a directory or set of files that no other agent touches. Agents communicate through **interface contracts** defined upfront.

```
Agent 1 (FOUNDATION)    — tests + hardening of existing Cortex runtime
Agent 2 (SCL)           — semantic compression language: grammar, parser, emitter
Agent 3 (BRAILLE)       — compact encoding layer: codec, fingerprints, manifests
```

### Dependency graph

```
Agent 1 ──────────────────────────────────────────── (independent)
Agent 2 ──────────────────────────────────────────── (independent)
Agent 3 ── depends on Agent 2's SCL data model ──── (starts day 1, blocks on
                                                      SCL types by end of phase 1)
```

Agent 3 can start immediately on the Braille codec (pure encoding, no SCL dependency), then integrates with SCL types once Agent 2 publishes them.

---

## File Ownership (Strict — No Conflicts)

### Agent 1 owns:
```
tests/                          # new directory, all test files
tests/test_router.py
tests/test_tiers.py
tests/test_challenger.py
tests/test_swarm.py
tests/test_cortex.py
tests/test_memory.py
tests/test_policy.py
tests/test_tools.py
tests/test_resilience.py
tests/test_api_adapter.py
tests/test_backend_adapter.py
tests/test_hardware_detect.py
tests/test_integration.py
tests/conftest.py               # shared fixtures
```

### Agent 2 owns:
```
src/scl/                        # new directory
src/scl/__init__.py
src/scl/types.py                # SCL data model (Record, Anchor, Relation, Scope)
src/scl/parser.py               # text → SCL AST
src/scl/emitter.py              # SCL AST → text
src/scl/cortex_bridge.py        # RouteDecision/ChallengeResult/SwarmResult ↔ SCL
src/scl/grammar.py              # formal grammar definition
tests/test_scl_parser.py
tests/test_scl_emitter.py
tests/test_scl_bridge.py
```

### Agent 3 owns:
```
src/braille/                    # new directory
src/braille/__init__.py
src/braille/codec.py            # encode/decode bytes ↔ Braille Unicode
src/braille/fingerprint.py      # SCL Record → fixed-width Braille hash
src/braille/manifest.py         # model/tier manifests in Braille notation
src/braille/tokenizer_bench.py  # empirical token efficiency measurements
tests/test_braille_codec.py
tests/test_braille_fingerprint.py
tests/test_braille_manifest.py
```

---

## Interface Contracts (Defined Before Work Starts)

These are the types that agents agree on before any code is written. Agent 2 publishes the SCL types; Agent 3 consumes them.

### Contract 1: SCL Record (Agent 2 publishes, Agent 3 consumes)

```python
# src/scl/types.py — Agent 2 defines this

@dataclass
class Anchor:
    """@ — entity/subject"""
    name: str                    # e.g. "router", "memory", "task"
    
@dataclass  
class Relation:
    """→ — transition/mapping/causality"""
    verb: str                    # e.g. "select", "escalate", "persist"

@dataclass
class Scope:
    """[ ] — bounded context frame"""
    entries: dict[str, str]      # e.g. {"model": "qwen3:4b", "confidence": "0.82"}

@dataclass
class SCLRecord:
    """One SCL statement: @anchor → relation [scope]"""
    anchor: Anchor
    relation: Relation
    scope: Scope
    timestamp_ms: int = 0
    
    def to_text(self) -> str:
        """Canonical text form: @router → select [model: qwen3:4b]"""
        ...
    
    @classmethod
    def from_text(cls, text: str) -> "SCLRecord":
        """Parse from canonical text form"""
        ...
    
    def to_bytes(self) -> bytes:
        """Compact binary for Braille encoding"""
        ...
```

### Contract 2: Cortex → SCL Bridge (Agent 2 publishes, Agent 1 can use in integration tests)

```python
# src/scl/cortex_bridge.py — Agent 2 defines this

def route_decision_to_scl(decision: RouteDecision) -> list[SCLRecord]:
    """Convert a RouteDecision into SCL records."""
    ...

def challenge_result_to_scl(result: ChallengeResult) -> list[SCLRecord]:
    """Convert a ChallengeResult into SCL records."""
    ...

def swarm_result_to_scl(result: SwarmResult) -> list[SCLRecord]:
    """Convert a SwarmResult into SCL records."""
    ...

def cortex_response_to_scl(response: CortexResponse) -> list[SCLRecord]:
    """Full pipeline result → SCL document."""
    ...
```

### Contract 3: SCL → Braille (Agent 3 consumes SCL types)

```python
# src/braille/fingerprint.py — Agent 3 defines this

def fingerprint(record: SCLRecord, width: int = 4) -> str:
    """
    SCL record → fixed-width Braille fingerprint.
    Returns a string of `width` Braille Unicode characters.
    """
    ...

def fingerprint_batch(records: list[SCLRecord]) -> str:
    """Multiple SCL records → single Braille fingerprint string."""
    ...
```

---

## Agent 1: FOUNDATION

**Mission:** Make the existing Cortex runtime trustworthy. Write comprehensive tests. Fix bugs found during testing. Do not add features.

### Phase 1 (Days 1–2): Core unit tests

Create `tests/conftest.py` with shared fixtures:
- Simulated `SystemProfile` objects (reuse `SIMULATED_PROFILES` from `hardware_detect.py`)
- Mock `BackendAdapter` that returns canned `CompletionResponse` objects
- In-memory `Memory` instance (`:memory:` SQLite)
- Deterministic `CortexConfig` for reproducible tests

Write unit tests for pure functions (no I/O):
- `test_router.py` — `_estimate_complexity()`, `_categorize()`, `route_heuristic()`, `route_with_model()` with mock model_fn. Test every `TaskCategory`. Test tier clamping. Test confidence scoring.
- `test_tiers.py` — `assess_tiers()` across all simulated profiles. `max_feasible_tier()`. `get_models_for_tier()`. `get_challenge_models()`. `concurrent_vram_budget()`. `_parse_param_size()`. `_params_to_tier()`.
- `test_challenger.py` — `compare_answers()` with known agree/disagree pairs. `_normalize()`. `_token_overlap()`. `_detect_yes_no()`. Test each `AgreementLevel` threshold.
- `test_swarm.py` — `_model_weight()`. `_cluster_votes()` with synthetic votes. `_majority_vote()` vs `_weighted_vote()`. Confidence formula edge cases.

### Phase 2 (Days 2–3): System tests

- `test_memory.py` — Full CRUD on threads, messages, audit log, KV cache, policies, usage stats. Test WAL mode. Test `get_context_window()` trimming. Test `prune_audit()`.
- `test_policy.py` — Every policy type: `max_tier`, `cloud_allowed`, `local_only`, `rate_limit`, `max_tokens`, `blocked_tools`, `allowed_models`. Test hierarchy (thread > app > global). Test `_RateLimiter`.
- `test_tools.py` — `ToolRegistry` registration, permission rings, `execute()` with ring violations, rate limiting, `parse_tool_calls()`, `results_to_messages()`.
- `test_resilience.py` — `CircuitBreaker` state transitions (closed → open → half-open → closed). `RetryPolicy.delay_for_attempt()`. `ResilienceLayer` with injected failures.
- `test_api_adapter.py` — `normalize_request()` for all three API formats. `format_response()` back to each format. Round-trip tests. Multimodal text extraction.
- `test_hardware_detect.py` — Test with simulated profiles. Verify VRAM budget calculations. Test `_run()` error handling.

### Phase 3 (Days 3–4): Integration + bug fixes

- `test_cortex.py` — End-to-end `Cortex.process()` with mock adapters. Verify the full escalation path: route → generate → challenge → swarm. Test `boot()`. Test `_escalate_generate()`. Test `resolve_backend()`.
- `test_integration.py` — `DaemonServer` with mock Cortex: test each HTTP endpoint, verify JSON responses, test API format translation end-to-end.
- Fix any bugs discovered. Do NOT change public interfaces without coordinating with Agents 2/3.
- Add `pytest.ini` or `pyproject.toml` test config.

### Deliverables
- `tests/` directory with 12+ test files, 200+ test cases
- All tests pass with `pytest tests/ -v`
- Bug fix PRs for anything found
- Coverage report

---

## Agent 2: SCL (Semantic Compression Language)

**Mission:** Design and implement SCL as the semantic control protocol for Cortex. Make it parseable, emittable, and bridged to existing Cortex types.

### Phase 1 (Days 1–2): Grammar + types + parser

Define the formal grammar in `src/scl/grammar.py`:

```
record     := anchor relation scope
anchor     := '@' IDENTIFIER
relation   := '→' VERB
scope      := '[' entries ']'
entries    := entry (',' entry)*
entry      := KEY ':' VALUE
document   := record ('\n' record)*
```

Implement in `src/scl/types.py`:
- `Anchor`, `Relation`, `Scope`, `SCLRecord` dataclasses (per contract above)
- `SCLDocument` — ordered list of `SCLRecord` with metadata
- `to_text()`, `from_text()`, `to_bytes()`, `from_bytes()` on each type
- `to_dict()` / `from_dict()` for JSON serialization

Implement in `src/scl/parser.py`:
- `parse_record(text: str) -> SCLRecord`
- `parse_document(text: str) -> SCLDocument`
- Error handling with line/column info
- Lenient mode (skip malformed lines) vs strict mode

Implement in `src/scl/emitter.py`:
- `emit_record(record: SCLRecord) -> str`
- `emit_document(doc: SCLDocument) -> str`
- Compact mode (one line per record) vs pretty mode (aligned columns)

Write tests:
- `tests/test_scl_parser.py` — round-trip parse/emit, edge cases, error handling
- `tests/test_scl_emitter.py` — formatting, compact vs pretty

### Phase 2 (Days 2–3): Cortex bridge

Implement `src/scl/cortex_bridge.py`:

```python
# Route decision → SCL
route_decision_to_scl(RouteDecision) -> list[SCLRecord]
# Example output:
#   @task → classify [category: code, complexity: 0.45]
#   @router → select [tier: L3, confidence: 0.82]

# Challenge result → SCL  
challenge_result_to_scl(ChallengeResult) -> list[SCLRecord]
# Example output:
#   @core → answer [model: qwen3:8b, family: qwen]
#   @challenger → answer [model: granite3.3:8b, family: granite]
#   @agreement → evaluate [level: weak_agree, confidence: 0.65]

# Swarm result → SCL
swarm_result_to_scl(SwarmResult) -> list[SCLRecord]
# Example output:
#   @swarm → query [size: small, models: 4, families: 3]
#   @cluster_0 → agree [models: 3, families: 2, weight: 7.5]
#   @cluster_1 → disagree [models: 1, families: 1, weight: 2.0]
#   @consensus → select [cluster: 0, confidence: 0.78]

# Full response → SCL document
cortex_response_to_scl(CortexResponse) -> SCLDocument
```

Write `tests/test_scl_bridge.py`:
- Create real `RouteDecision`, `ChallengeResult`, `SwarmResult`, `CortexResponse` objects
- Convert to SCL, verify structure
- Round-trip: Cortex type → SCL → text → parse → verify fields match

### Phase 3 (Days 3–4): Audit log integration

Add SCL output to the memory/audit layer (coordinate with Agent 1 on test expectations):

- `scl_from_audit_entry(AuditEntry) -> SCLDocument` — convert audit log entries to SCL
- `scl_summary(entries: list[AuditEntry]) -> SCLDocument` — summarize N audit entries into a compact SCL document
- CLI command: `python -m src scl-audit --last 10` — dump last 10 audit entries as SCL

### Deliverables
- `src/scl/` package with 5 modules
- Formal grammar spec
- Parser + emitter with round-trip fidelity
- Bridge to all Cortex result types
- 3 test files, 100+ test cases

---

## Agent 3: BRAILLE (Compact Encoding Layer)

**Mission:** Build the Braille encoding layer — codec, fingerprints, and manifests. Empirically validate token efficiency.

### Phase 1 (Days 1–2): Core codec

Implement `src/braille/codec.py`:

```python
# The 256 Unicode Braille characters (U+2800 to U+28FF) map 1:1 to byte values 0-255.
# Each character's Unicode codepoint offset from U+2800 IS the byte value.
# This is a natural, lossless, bijective encoding.

def encode(data: bytes) -> str:
    """Encode arbitrary bytes as Braille Unicode string."""
    # byte 0x00 → ⠀ (U+2800), byte 0xFF → ⣿ (U+28FF)
    ...

def decode(braille: str) -> bytes:
    """Decode Braille Unicode string back to bytes."""
    ...

def encode_hex(hex_str: str) -> str:
    """Encode a hex string as Braille."""
    ...

def encode_int(n: int, width: int = 4) -> str:
    """Encode an integer as fixed-width Braille."""
    ...
```

Key properties to verify:
- **Bijective**: every byte sequence has exactly one Braille representation
- **Lossless**: `decode(encode(x)) == x` for all `x`
- **Fixed density**: 1 byte = 1 Braille character (3 UTF-8 bytes)

Write `tests/test_braille_codec.py`:
- Round-trip tests for all 256 byte values
- Random byte strings of various lengths
- Edge cases: empty input, single byte, max-length
- Verify Unicode codepoint math

Implement `src/braille/tokenizer_bench.py`:

```python
def bench_tokenizer(text: str, braille: str, tokenizer_name: str = "Qwen/Qwen3-0.6B"):
    """
    Compare token counts:
      - Original text as tokens
      - Braille-encoded as tokens
      - JSON-encoded as tokens
      - Hex-encoded as tokens
    
    Prints a comparison table.
    """
    ...
```

This is the empirical validation that the white paper needs. Run against:
- Qwen3 tokenizer
- Llama tokenizer
- GPT-4 tokenizer (tiktoken)

Report: tokens per byte for each encoding × tokenizer combination.

### Phase 2 (Days 2–3): Fingerprints

Implement `src/braille/fingerprint.py`:

```python
import hashlib
from ..scl.types import SCLRecord, SCLDocument  # Agent 2's types

def fingerprint(record: SCLRecord, width: int = 4) -> str:
    """
    SCL record → fixed-width Braille fingerprint.
    
    Process:
      1. Serialize record to canonical bytes (record.to_bytes())
      2. SHA-256 hash
      3. Take first `width` bytes of hash
      4. Encode as Braille
    
    Result: `width` Braille characters (e.g. 4 chars = 32 bits of hash)
    """
    ...

def fingerprint_document(doc: SCLDocument, width: int = 8) -> str:
    """Full SCL document → single Braille fingerprint."""
    ...

def fingerprint_match(fp1: str, fp2: str) -> bool:
    """Compare two fingerprints for equality."""
    return fp1 == fp2

def similarity(fp1: str, fp2: str) -> float:
    """
    Hamming-distance-based similarity between two fingerprints.
    Returns 0.0 (completely different) to 1.0 (identical).
    """
    ...
```

Write `tests/test_braille_fingerprint.py`:
- Deterministic: same input → same fingerprint
- Collision resistance: different inputs → different fingerprints (probabilistic)
- Width parameter works correctly
- Similarity metric: identical = 1.0, random ≈ 0.5

### Phase 3 (Days 3–4): Manifests + routing signatures

Implement `src/braille/manifest.py`:

```python
def tier_manifest(tier: Tier, profile: SystemProfile) -> str:
    """
    Encode a tier's capability manifest as a Braille string.
    
    Encodes: tier level, param range, VRAM, capabilities, hot status.
    Useful for compact model cards and routing tables.
    """
    ...

def routing_signature(decision: RouteDecision) -> str:
    """
    Encode a routing decision as a short Braille signature.
    
    Format: [tier 1 char][category 1 char][confidence 1 char][flags 1 char]
    Total: 4 Braille characters = compact routing fingerprint.
    """
    ...

def system_manifest(profile: SystemProfile) -> str:
    """
    Encode a full system profile as a Braille manifest.
    
    Compact representation of: OS, arch, GPU, VRAM, backends, max tier.
    """
    ...

def decode_routing_signature(sig: str) -> dict:
    """Decode a routing signature back to human-readable dict."""
    ...
```

Write `tests/test_braille_manifest.py`:
- Round-trip routing signatures
- Manifests for all simulated profiles
- Verify decode matches original values

### Deliverables
- `src/braille/` package with 4 modules
- Empirical tokenizer benchmark report
- Fingerprint system with collision testing
- Manifest/routing signature codec
- 3 test files, 80+ test cases

---

## Coordination Protocol

### Daily sync points

Each agent publishes a one-line status to `STATUS.md` at the end of each work session:

```
## Agent 1 (FOUNDATION)
Phase 1 complete. 87 tests passing. Bug found in router confidence calc. Fix pending.

## Agent 2 (SCL)
Phase 1 complete. Grammar + parser + emitter working. types.py published — Agent 3 can import.

## Agent 3 (BRAILLE)
Phase 1 complete. Codec done. Tokenizer bench: Braille = 2.8 tokens/byte on Qwen3 vs 2.1 for hex. Waiting on SCL types.
```

### Merge order

```
1. Agent 2 merges src/scl/types.py FIRST (unblocks Agent 3)
2. Agent 1 merges tests/ (independent, can merge anytime)
3. Agent 2 merges remaining src/scl/ 
4. Agent 3 merges src/braille/
5. Integration PR: wire SCL into Cortex audit log + daemon
```

### Conflict avoidance rules

1. **Never edit a file you don't own.** If you need a change in another agent's file, open an issue.
2. **Imports only at boundaries.** Agent 3 imports from `src.scl.types` only. Agent 1 imports from `src.*` (existing) only.
3. **No changes to existing `src/*.py` signatures** without all-agent agreement. Internal implementation changes are fine.
4. **Tests go in `tests/` only.** Test file naming: `test_{module}.py` for Agent 1, `test_scl_{module}.py` for Agent 2, `test_braille_{module}.py` for Agent 3.

---

## Scaling to N Agents

The architecture above scales by subdivision:

### At 6 agents:
```
Agent 1a: tests for router, tiers, hardware_detect
Agent 1b: tests for memory, policy, tools, resilience
Agent 2a: SCL grammar + parser + emitter
Agent 2b: SCL cortex_bridge + audit integration
Agent 3a: Braille codec + tokenizer bench
Agent 3b: Braille fingerprint + manifest
```

### At 12 agents:
```
Agent 1a–1d: one test file per agent (4 modules each)
Agent 2a–2d: grammar, parser, emitter, bridge (one each)
Agent 3a–3d: codec, fingerprint, manifest, bench (one each)
```

### At 100 agents:
The unit of work becomes a **single function or single test case**. At this scale you need:

1. **A task queue** — each task is: "implement function X in file Y with signature Z, test with inputs A→B"
2. **An SCL manifest per task** — `@task → implement [fn: _cluster_votes, file: swarm.py, inputs: 3, outputs: 1]`
3. **Braille fingerprints for dedup** — hash each task to detect redundant work
4. **Cortex itself as the coordinator** — L0 routes tasks to agents the way it routes prompts to models

This is where SCL/Braille becomes self-hosting: the coordination protocol for building Cortex IS Cortex's semantic substrate.

---

## Success Criteria

### Week 1
- [ ] All existing Cortex modules have unit tests (Agent 1)
- [ ] SCL grammar is defined and parser round-trips (Agent 2)
- [ ] Braille codec is bijective and benchmarked (Agent 3)
- [ ] Tokenizer efficiency report published (Agent 3)

### Week 2
- [ ] SCL bridge converts all Cortex result types (Agent 2)
- [ ] Braille fingerprints work on SCL records (Agent 3)
- [ ] Integration: `python -m src route "..." --scl` outputs SCL (Agent 2)
- [ ] Integration: routing signatures appear in audit log as Braille (Agent 3)

### Week 3
- [ ] SCL is the native format for Cortex audit records
- [ ] Braille fingerprints in daemon /status endpoint
- [ ] All 3 agents' code merged, all tests green
- [ ] Updated white paper with SCL/Braille sections
