# AGENTS.md — Cortex Agent Coordination (SCL-Native)

> This file is written in **Semantic Compression Language (SCL)**.
> Grammar: `@anchor → verb [key: value, key: value]`
> Every block is machine-parseable. The prose is the compression.

---

## System State

```scl
@cortex → status [modules: 20, size_kb: 280, runtime: working]
@cortex → has [hardware_detect, tiers, router, backend_adapter, backend_selector,
               model_manager, challenger, swarm, cortex, daemon, api_adapter,
               memory, policy, tools, resilience, main]
@cortex → missing [tests, scl_parser, braille_codec, cli_polish, docs]
@cortex → needs [verification, semantic_control, compact_encoding]
```

---

## SCL Grammar

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
- `@` — **Anchor.** Entity, subject, or noun. The thing being described.
- `→` — **Relation.** Verb, transition, causality. What it does.
- `[ ]` — **Scope.** Bounded context frame. Key-value attributes.

**Composition:**
- Records compose into documents (one per line)
- Documents compose into manifests (multi-document)
- Manifests encode into Braille fingerprints (fixed-width hash)

---

## Agent Architecture

```scl
@agent_1 → own [role: foundation, dir: tests/, mission: verify_runtime]
@agent_2 → own [role: scl, dir: src/scl/, mission: semantic_protocol]
@agent_3 → own [role: braille, dir: src/braille/, mission: compact_encoding]

@agent_1 → depend [on: nothing]
@agent_2 → depend [on: nothing]
@agent_3 → depend [on: agent_2.scl.types, blocks_at: phase_2]

@conflict → prevent [rule: strict_file_ownership, rule: imports_at_boundaries_only]
```

---

## File Ownership

```scl
@agent_1 → own [tests/conftest.py, tests/test_router.py, tests/test_tiers.py,
                 tests/test_challenger.py, tests/test_swarm.py, tests/test_cortex.py,
                 tests/test_memory.py, tests/test_policy.py, tests/test_tools.py,
                 tests/test_resilience.py, tests/test_api_adapter.py,
                 tests/test_backend_adapter.py, tests/test_hardware_detect.py,
                 tests/test_integration.py]

@agent_2 → own [src/scl/__init__.py, src/scl/types.py, src/scl/parser.py,
                 src/scl/emitter.py, src/scl/cortex_bridge.py, src/scl/grammar.py,
                 tests/test_scl_parser.py, tests/test_scl_emitter.py,
                 tests/test_scl_bridge.py]

@agent_3 → own [src/braille/__init__.py, src/braille/codec.py,
                 src/braille/fingerprint.py, src/braille/manifest.py,
                 src/braille/tokenizer_bench.py, tests/test_braille_codec.py,
                 tests/test_braille_fingerprint.py, tests/test_braille_manifest.py]
```

---

## Interface Contracts

### Contract 1: SCL Types (Agent 2 publishes → Agent 3 consumes)

```scl
@anchor → define [type: dataclass, field: name(str)]
@relation → define [type: dataclass, field: verb(str)]
@scope → define [type: dataclass, field: entries(dict[str,str])]
@scl_record → compose [anchor, relation, scope, timestamp_ms: int]
@scl_record → serialize [to_text, from_text, to_bytes, from_bytes, to_dict, from_dict]
@scl_document → compose [records: list[SCLRecord], metadata: dict]
```

Canonical form: `@router → select [model: qwen3:4b, confidence: 0.82]`

### Contract 2: Cortex Bridge (Agent 2 publishes → Agents 1,3 consume)

```scl
@route_decision → bridge [to: list[SCLRecord]]
  @task → classify [category: code, complexity: 0.45]
  @router → select [tier: L3, confidence: 0.82]

@challenge_result → bridge [to: list[SCLRecord]]
  @core → answer [model: qwen3:8b, family: qwen]
  @challenger → answer [model: granite3.3:8b, family: granite]
  @agreement → evaluate [level: weak_agree, confidence: 0.65]

@swarm_result → bridge [to: list[SCLRecord]]
  @swarm → query [size: small, models: 4, families: 3]
  @cluster_0 → agree [models: 3, families: 2, weight: 7.5]
  @cluster_1 → disagree [models: 1, families: 1, weight: 2.0]
  @consensus → select [cluster: 0, confidence: 0.78]

@cortex_response → bridge [to: SCLDocument]
```

### Contract 3: Braille Encoding (Agent 3 publishes)

```scl
@fingerprint → define [input: SCLRecord, output: str, width: 4]
  @process → hash [algo: sha256, take: width_bytes, encode: braille]
@fingerprint_document → define [input: SCLDocument, output: str, width: 8]
@similarity → define [input: (fp1, fp2), output: float, method: hamming]
@routing_signature → define [format: tier|category|confidence|flags, width: 4]
```

---

## Agent 1: FOUNDATION

```scl
@agent_1 → mission [verify_runtime, write_tests, fix_bugs, no_features]
```

### Phase 1: Core Unit Tests

```scl
@conftest → create [fixtures: simulated_profiles, mock_adapter, memory_inmem, deterministic_config]
@test_router → verify [_estimate_complexity, _categorize, route_heuristic, route_with_model,
                        every_TaskCategory, tier_clamping, confidence_scoring]
@test_tiers → verify [assess_tiers, max_feasible_tier, get_models_for_tier,
                       get_challenge_models, concurrent_vram_budget, _parse_param_size]
@test_challenger → verify [compare_answers, _normalize, _token_overlap,
                            _detect_yes_no, every_AgreementLevel]
@test_swarm → verify [_model_weight, _cluster_votes, _majority_vote,
                       _weighted_vote, confidence_edge_cases]
```

### Phase 2: System Tests

```scl
@test_memory → verify [crud_threads, crud_messages, audit_log, kv_cache,
                        policies, usage_stats, wal_mode, context_window, prune]
@test_policy → verify [max_tier, cloud_allowed, local_only, rate_limit,
                        max_tokens, blocked_tools, allowed_models,
                        hierarchy: thread>app>global, _RateLimiter]
@test_tools → verify [registration, permission_rings, execute_ring_violation,
                       rate_limit, parse_tool_calls, results_to_messages]
@test_resilience → verify [circuit_breaker_transitions: closed→open→half_open→closed,
                            retry_delay, resilience_layer_with_failures]
@test_api_adapter → verify [normalize_all_formats, format_response_all,
                             round_trip, multimodal_extraction]
```

### Phase 3: Integration

```scl
@test_cortex → verify [process_e2e, escalation_path: route→generate→challenge→swarm,
                        boot, _escalate_generate, resolve_backend]
@test_integration → verify [daemon_endpoints, json_responses, api_translation_e2e]
@agent_1 → deliver [test_files: 12+, test_cases: 200+, cmd: pytest tests/ -v]
```

---

## Agent 2: SCL

```scl
@agent_2 → mission [design_scl, implement_parser, bridge_cortex_types]
```

### Phase 1: Grammar + Parser + Emitter

```scl
@grammar → define [in: src/scl/grammar.py, spec: formal_bnf]
@types → implement [Anchor, Relation, Scope, SCLRecord, SCLDocument,
                     methods: to_text, from_text, to_bytes, from_bytes, to_dict, from_dict]
@parser → implement [parse_record, parse_document, error_handling: line_col,
                      modes: strict|lenient]
@emitter → implement [emit_record, emit_document, modes: compact|pretty]
@tests → write [test_scl_parser: round_trip+edge_cases, test_scl_emitter: formatting]
```

### Phase 2: Cortex Bridge

```scl
@bridge → implement [route_decision_to_scl, challenge_result_to_scl,
                      swarm_result_to_scl, cortex_response_to_scl]
@tests → write [test_scl_bridge: real_objects, structure_verify, round_trip]
```

### Phase 3: Audit Integration

```scl
@audit → implement [scl_from_audit_entry, scl_summary]
@cli → add [command: scl-audit, flags: --last N, output: scl_text]
@agent_2 → deliver [modules: 5, test_files: 3, test_cases: 100+]
```

---

## Agent 3: BRAILLE

```scl
@agent_3 → mission [build_braille_layer, validate_token_efficiency]
```

### Phase 1: Core Codec

```scl
@codec → implement [encode: bytes→braille, decode: braille→bytes,
                     encode_hex, encode_int]
@codec → verify [bijective: true, lossless: true, density: 1_byte=1_char=3_utf8]
@bench → implement [bench_tokenizer, tokenizers: qwen3|llama|gpt4,
                     compare: braille|json|hex, metric: tokens_per_byte]
@tests → write [test_braille_codec: all_256_bytes, random_strings, edge_cases]
```

### Phase 2: Fingerprints

```scl
@fingerprint → implement [input: SCLRecord, process: serialize→sha256→take_width→braille]
@fingerprint_document → implement [input: SCLDocument, width: 8]
@similarity → implement [method: hamming_distance, range: 0.0_to_1.0]
@tests → write [deterministic, collision_resistance, width_param, similarity_metric]
```

### Phase 3: Manifests

```scl
@tier_manifest → implement [encodes: tier_level|param_range|vram|capabilities|hot]
@routing_signature → implement [format: 4_braille_chars, fields: tier|category|confidence|flags]
@system_manifest → implement [encodes: os|arch|gpu|vram|backends|max_tier]
@decode → implement [routing_signature→dict, round_trip_verified]
@agent_3 → deliver [modules: 4, test_files: 3, test_cases: 80+, bench_report: 1]
```

---

## Coordination Protocol

```scl
@sync → require [frequency: daily, format: one_line_status, target: STATUS.md]
@merge_order → define [
  1: agent_2.src/scl/types.py,
  2: agent_1.tests/,
  3: agent_2.src/scl/*,
  4: agent_3.src/braille/*,
  5: integration_pr
]
@rule → enforce [never_edit_unowned_files]
@rule → enforce [imports_at_boundaries_only]
@rule → enforce [no_signature_changes_without_consensus]
@rule → enforce [tests_in_tests_dir_only]
```

---

## Scaling

```scl
@scale_3 → config [agents: 3, granularity: package]
@scale_6 → config [agents: 6, granularity: subpackage]
  @1a → own [tests: router+tiers+hardware]
  @1b → own [tests: memory+policy+tools+resilience]
  @2a → own [scl: grammar+parser+emitter]
  @2b → own [scl: bridge+audit]
  @3a → own [braille: codec+bench]
  @3b → own [braille: fingerprint+manifest]

@scale_12 → config [agents: 12, granularity: module]
@scale_100 → config [agents: 100, granularity: function]
  @coordination → require [task_queue, scl_manifest_per_task, braille_dedup, cortex_as_coordinator]
  @meta → note [scl_becomes_self_hosting: coordination_protocol_IS_semantic_substrate]
```

---

## Success Criteria

```scl
@week_1 → require [
  agent_1: all_modules_tested,
  agent_2: grammar_defined+parser_roundtrips,
  agent_3: codec_bijective+bench_published
]
@week_2 → require [
  agent_2: bridge_all_cortex_types,
  agent_3: fingerprints_on_scl_records,
  integration: route_cmd_outputs_scl+audit_has_braille_signatures
]
@week_3 → require [
  scl: native_audit_format,
  braille: in_daemon_status,
  all: merged+green,
  whitepaper: updated_with_scl_braille
]
```
