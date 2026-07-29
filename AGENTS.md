# AGENTS.md — Cortex Agent Coordination (SCL-Native)

> This file is written in **Semantic Compression Language (SCL)**.
> Grammar: `@anchor → verb [key: value, key: value]`
> Every block is machine-parseable. The prose is the compression.

---

## System State

```scl
@cortex → status [modules: 28, size_kb: 380, runtime: working]
@cortex → has [hardware_detect, tiers, router, backend_adapter, backend_selector,
               model_manager, challenger, swarm, cortex, daemon, api_adapter,
               memory, policy, tools, resilience, main,
               scl_types, scl_parser, scl_emitter, scl_grammar, scl_bridge, scl_delta, scl_gossip, scl_eval,
               braille_codec, braille_fingerprint, braille_manifest]
@cortex → mutate [missing: [cli_polish, docs, tokenizer_bench, audit_scl_format]]
@cortex → mutate [needs: [audit_integration, daemon_braille_status, cluster_head_election]]
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

**Deltas:**
- `@agent → mutate [key: new_value]` — only changed keys, receivers assume previous state persists
- `@agent → snapshot [key: value, ...]` — full state, for bootstrapping new agents
- `@agent → rollback [to: t, reason: ...]` — time-travel to previous version
- Reconstruction: $S_t = S_0 \oplus \sum_{i=1}^{t} \Delta S_i$

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
                 src/scl/delta.py, src/scl/gossip.py, src/scl/eval.py,
                 tests/test_scl_parser.py, tests/test_scl_emitter.py,
                 tests/test_scl_bridge.py, tests/test_scl_delta.py,
                 tests/test_scl_gossip.py, tests/test_scl_eval.py]

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

### Contract 4: Semantic State Deltas (Agent 2 publishes → All consume)

```scl
@vector_clock → define [type: dict[str,int], ops: tick, merge, concurrent]
@delta → define [agent_id: str, set_keys: dict, delete_keys: set,
                  timestamp_ms: int, seq: int, weight: float, parent_hash: str]
@delta → serialize [to_scl: @agent → mutate [...], from_scl: parse_mutate_record]
@semantic_state → define [entries: dict[str,str], clock: VectorClock, version: int]
@diff → define [input: (old_state, new_state), output: Delta]
@apply_delta → define [input: (state, delta), output: new_state]
@merge_deltas → define [input: (delta_a, delta_b, strategy), output: (merged, conflicts)]
  @strategy → enum [lww: last_writer_wins, priority: weight_wins, union: crdt_gset, reject: error]
@delta_stream → define [ops: append, current_state, state_at, rollback, checkpoint, compact,
                         to_scl_document]
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

### Phase 4: Semantic State Deltas ✓

```scl
@delta → implement [VectorClock, Delta, SemanticState, diff, apply_delta,
                     merge_deltas, DeltaStream]
@delta → mutate [status: complete, tests: 43_passing, file: src/scl/delta.py]
@merge_strategies → implement [lww: timestamp_wins, priority: weight_wins,
                                union: crdt_gset, reject: raise_error]
@delta_stream → implement [append, state_at, rollback, checkpoint, compact,
                            to_scl_document]
@tests → write [test_scl_delta: vector_clock+diff+apply+merge+stream+fingerprint+scale,
                 test_cases: 43]
```

### Phase 5: Gossip Protocol ✓

```scl
@gossip → implement [Peer, Swarm, GossipMessage, GossipMessageType, GossipStats]
@gossip → mutate [status: complete, file: src/scl/gossip.py]
@peer → implement [mutate, receive_delta, initiate_sync, state_fingerprint,
                    dedup: content_keyed, bootstrap: initial_state_as_delta]
@swarm → implement [add_peer, gossip_round, run_until_converged, is_converged,
                     convergence_matrix, to_scl_document]
@convergence → measure [5_agents: 1_round, 50_agents: 4_rounds, scaling: O_log_N]
@protocol → define [ping: fingerprint_compare, push_delta: changed_keys_only,
                     dedup: content_keyed_seen_set, anti_entropy: push_pull]
```

### Phase 6: Executable SCL — Evaluator ✓

```scl
@eval → implement [Condition, CompoundCondition, Action, Rule, SCLFunction, RuleEngine]
@eval → mutate [status: complete, file: src/scl/eval.py]
@conditions → implement [ops: == != > >= < <= in not_in ~ exists,
                          logic: ∧_and ∨_or ¬_not]
@actions → implement [emit, mutate, escalate, call, chain, log, suppress]
@functions → implement [define: λ_abstraction, invoke: $var_substitution,
                         versioned: auto_increment, self_modify: via_delta]
@meta → implement [process_meta: when+define+undefine+enable+disable,
                    self_modifying: agents_rewrite_own_rules_via_scl]
@rules → serialize [to_scl: @name → when [...], from_scl: parse_when_record]
@layer_stack → define [
  L0: types.py_structure,
  L1: delta.py_mutation,
  L2: gossip.py_propagation,
  L3: eval.py_execution,
  L4: agents_rewrite_L3_via_L1_deltas_gossipped_by_L2
]
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
@sync → mutate [format: delta_scl, frequency: per_commit, target: delta_stream]
@sync → note [agents_emit_deltas_not_full_state, receivers_apply_in_order]
@merge_order → define [
  1: agent_2.src/scl/types.py,           ✓ complete
  2: agent_1.tests/,                      ✓ complete
  3: agent_2.src/scl/*,                   ✓ complete (parser, emitter, grammar, bridge, delta)
  4: agent_3.src/braille/*,               ✓ complete (codec, fingerprint, manifest)
  5: integration_pr                       → next
]
@merge_conflict → resolve [strategy: priority, fallback: lww,
                            supervisor_weight: 5.0, worker_weight: 1.0]
@rule → enforce [never_edit_unowned_files]
@rule → enforce [imports_at_boundaries_only]
@rule → enforce [no_signature_changes_without_consensus]
@rule → enforce [tests_in_tests_dir_only]
@rule → enforce [deltas_over_snapshots_for_sync]
```

---

## Scaling

```scl
@scale_3 → config [agents: 3, granularity: package]
@scale_6 → config [agents: 6, granularity: subpackage]
  @1a → own [tests: router+tiers+hardware]
  @1b → own [tests: memory+policy+tools+resilience]
  @2a → own [scl: grammar+parser+emitter+delta]
  @2b → own [scl: bridge+audit]
  @3a → own [braille: codec+bench]
  @3b → own [braille: fingerprint+manifest]

@scale_12 → config [agents: 12, granularity: module]
@scale_100 → config [agents: 100, granularity: function]
  @coordination → require [delta_stream, scl_manifest_per_task, braille_dedup, cortex_as_coordinator]
  @gossip → require [broadcast: fingerprinted_deltas, protocol: crdt_merge,
                      convergence: hamming_threshold, conflict: priority_weighted]
  @meta → note [scl_becomes_self_hosting: coordination_protocol_IS_semantic_substrate]
  @meta → note [at_scale_100: agents_sync_via_delta_streams_not_full_state]

@scale_infinity → config [agents: unbounded, granularity: semantic_atom]
  @sync → require [protocol: delta_gossip, payload: braille_fingerprinted_mutations]
  @convergence → check [method: hamming_distance_on_fingerprints, threshold: 0.9]
  @conflict → resolve [concurrent: vector_clock, strategy: priority|lww|union]
  @time_travel → enable [rollback: delta_stream_replay, debug: isolate_rogue_agent]
```

---

## Success Criteria

```scl
@week_1 → mutate [status: complete]
  @agent_1 → deliver [all_modules_tested: ✓]
  @agent_2 → deliver [grammar_defined: ✓, parser_roundtrips: ✓, types: ✓]
  @agent_3 → deliver [codec_bijective: ✓]

@week_2 → mutate [status: complete]
  @agent_2 → deliver [bridge_all_cortex_types: ✓, delta_layer: ✓, tests_43_passing: ✓]
  @agent_3 → deliver [fingerprints_on_scl_records: ✓, manifests: ✓]
  @whitepaper → deliver [updated_with_scl_braille_deltas: ✓]

@week_3 → require [
  scl: native_audit_format,
  braille: in_daemon_status,
  gossip: ✓_epidemic_convergence_in_O_log_N,
  all: merged+green,
  integration_pr: complete
]
```
