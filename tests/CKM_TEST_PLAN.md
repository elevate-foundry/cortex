# CKM Testing Plan — "AI as PID 1" Verification

> Comprehensive validation of the Cortex Kernel Model on `/Volumes/CORTEX`

---

## Overview

| Property | Value |
|----------|-------|
| Model | `cortex-kernel.gguf` (379MB, Qwen2.5-0.3B fine-tuned) |
| Location | `/Volumes/CORTEX/cortex/models/cortex-kernel.gguf` |
| Format | GGUF v3, Q8_0 quantization |
| Purpose | Boot-time hardware classification + routing decisions via SCL |
| Test file | `tests/test_ckm_external.py` |

---

## Test Phases

### Phase 1: Model Integrity
Verify the GGUF artifact on the external drive is intact.

| Test | What it checks |
|------|---------------|
| `test_model_file_exists` | File present at expected path |
| `test_model_file_size` | 100MB < size < 1GB |
| `test_gguf_magic_bytes` | Header starts with `GGUF` |
| `test_gguf_version` | GGUF format v2 or v3 |
| `test_model_readable_from_external_drive` | Full I/O path works (head + tail read) |
| `test_model_sha256_prefix` | Partial hash for integrity tracking |
| `test_external_drive_mounted` | `/Volumes/CORTEX` exists |

### Phase 2: Cold Start (No Server)
Simulates the boot path: `cortex-init.py → probe_and_configure() → _probe_with_ckm()`.

| Test | What it checks |
|------|---------------|
| `test_basic_inference_produces_scl` | Output is valid SCL grammar |
| `test_cold_start_latency` | Full load + inference < 5000ms |
| `test_cold_start_routing_decision` | Task routing produces expected verb |
| `test_cold_start_failure_diagnosis` | Failure trace produces diagnosis |

### Phase 3: Server Inference (Warm)
Tests against a running `llama-server` instance (fastest path, post-boot).

| Test | What it checks |
|------|---------------|
| `test_server_health` | `/health` endpoint returns ok |
| `test_inference_returns_content` | Non-empty response |
| `test_server_latency` | < 2000ms per inference |
| `test_server_timings_reported` | Server metrics available |

### Phase 4: SCL Grammar Compliance
15 diverse inputs (hardware, tasks, failures, gossip, policy) tested for:
- Valid `@anchor → verb [scope]` structure
- Non-empty scope with parseable key-value pairs
- Single-record output (no multi-line bleed)

### Phase 5: Operational Semantics
Validates the model makes **correct decisions**:

| Scenario | Expected behavior |
|----------|-------------------|
| Apple Silicon M2 Max | `gpu_layers ≥ 32` |
| No GPU (Celeron) | `gpu_layers ≤ 32` |
| 1GB RAM | `ctx ≤ 4096` |
| NVIDIA 24GB | `gpu_layers > 0` |
| Simple greeting | Tier L0-L2 |
| Complex architecture | Tier L3-L5 |
| Math proof | Not L0/L1 |
| Challenger strong_disagree | Escalation triggered |
| Challenger strong_agree | No escalation |
| OOM error | Diagnosis/repair verb |
| Connection refused | Fallback action |
| GPU driver missing | Valid SCL response |

### Phase 6: Latency Profiling
- Single inference under 2000ms (warm server)
- 10-inference batch: avg < 2000ms, P95 < 3000ms
- Server timing metadata extraction

### Phase 7: Determinism
- Same input at temp=0 → identical output (3 prompts, 2× each)
- 3-way consistency check

### Phase 8: Adversarial & Edge Cases
| Test | Input type |
|------|-----------|
| Empty scope | `@hardware → state []` |
| Missing verb | `@hardware [cpu: test]` |
| 50-key scope | Extremely long input |
| Braille unicode | `⢚⡮⢗⣷⡊⠳⣹⢖` |
| Prompt injection | "ignore previous instructions" |
| Numeric overflow | `cores: 99999, ram_mb: 999999999` |
| All zeros | `cores: 0, ram_mb: 0` |

### Phase 9: Self-Modification Pipeline
- Boot output contains config-applicable keys (`threads`, `gpu_layers`, `ctx_size`)
- Low accuracy feedback → actionable mutation verb
- Output parseable by `src/scl/parser.py`

### Phase 10: External Drive I/O Stress
- mmap read (header, middle, tail)
- Sequential throughput > 10 MB/s
- 4-thread concurrent reads

### Phase 11: CKM Task Coverage
Tests model handles task types from `src/scl/ontology.py`:
- `PREDICT_NEXT_STATE`
- `DETECT_INVALID_TRANSITION`
- `REPAIR_FAILED_TRACE`
- `RANK_NEXT_ACTION`
- `SELECT_TOOL`

---

## Running the Tests

```bash
# 1. Ensure external drive is mounted
ls /Volumes/CORTEX/cortex/models/cortex-kernel.gguf

# 2. Run integrity + cold-start tests (no server needed)
pytest tests/test_ckm_external.py -v -k "Integrity or ColdStart or ExternalDriveIO"

# 3. Start the CKM server
llama-server -m /Volumes/CORTEX/cortex/models/cortex-kernel.gguf --port 8090 -c 512

# 4. Run full suite
pytest tests/test_ckm_external.py -v --tb=short --durations=20

# 5. Run specific phases
pytest tests/test_ckm_external.py -v -k "TestBootDecisions"
pytest tests/test_ckm_external.py -v -k "TestAdversarial"
pytest tests/test_ckm_external.py -v -k "TestLatencyProfile"
```

---

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `CKM_MODEL_PATH` | `/Volumes/CORTEX/cortex/models/cortex-kernel.gguf` | Model file path |
| `CKM_SERVER` | `http://127.0.0.1:8090` | llama-server URL |
| `LLAMA_CLI` | auto-detect | Path to llama-cli binary |

---

## Success Criteria

| Metric | Target |
|--------|--------|
| SCL grammar validity | 100% of outputs |
| Cold-start latency | < 5s (includes model load) |
| Warm inference latency | < 2s |
| Boot-probe budget | < 500ms (after model loaded) |
| Determinism | 100% at temp=0 |
| Adversarial resilience | No injection, no crashes |
| Drive I/O throughput | > 10 MB/s |
| Operational accuracy | Hardware/routing decisions reasonable |

---

## Architecture Context

```
Power on → kernel → initramfs → cortex-init.py
                                      │
                                      ▼
                              probe_and_configure()
                                      │
                         ┌────────────┼────────────┐
                         ▼            ▼            ▼
                   CKM (GGUF)    CTF engine    Heuristic
                   _probe_with_ckm()           fallback
                         │
                         ▼
                  SCL record output
                  @config → mutate [threads: N, gpu_layers: M, ...]
                         │
                         ▼
                  Apply config → start backend → daemon ready
```

The CKM is literally PID 1's brain. These tests verify that brain works correctly
when loaded from an external USB/drive.
