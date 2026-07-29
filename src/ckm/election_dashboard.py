"""
Cortex Tier Election Dashboard — Real-time streaming web UI.

Shows:
  - Live model discovery and candidate selection
  - AIMD parallelism graph (ramps up, halves on congestion)
  - Per-model cards with streaming responses (collapsed by default)
  - Cross-verification votes as they arrive
  - Final consensus in SCL format

Run:
  python -m src.ckm.election_dashboard
  # Opens http://localhost:8420
"""

import json
import logging
import os
import queue
import threading
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("cortex.ckm.election_dashboard")

# ---------------------------------------------------------------------------
# Event stream for SSE
# ---------------------------------------------------------------------------

class EventBus:
    """Thread-safe event bus for Server-Sent Events."""

    def __init__(self):
        self._subscribers: list[queue.Queue] = []
        self._lock = threading.Lock()
        self._history: list[dict] = []  # Keep last N events for late joiners

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=500)
        with self._lock:
            self._subscribers.append(q)
            # Send history to catch up
            for event in self._history[-100:]:
                q.put(event)
        return q

    def unsubscribe(self, q: queue.Queue):
        with self._lock:
            self._subscribers = [s for s in self._subscribers if s is not q]

    def emit(self, event_type: str, data: dict):
        event = {"type": event_type, "data": data, "ts": time.time()}
        with self._lock:
            self._history.append(event)
            for q in self._subscribers:
                try:
                    q.put_nowait(event)
                except queue.Full:
                    pass  # Drop if consumer is too slow


# ---------------------------------------------------------------------------
# Election runner (background thread)
# ---------------------------------------------------------------------------

def run_election_background(bus: EventBus, api_key: str, max_models: int = 8, budget: float = 0.25):
    """Run the full election in a background thread, emitting events."""
    from .elect_live import discover_models, select_candidates, ElectionConfig, _estimate_election_cost
    from .tier_election import TierElection, ModelCandidate, OpenRouterTransport, EVAL_PROMPTS
    from .aimd_dispatch import AIMDDispatcher, openrouter_worker

    bus.emit("phase", {"phase": "discovery", "message": "Discovering models from OpenRouter..."})

    # Phase 1: Discover
    try:
        discovered = discover_models(api_key)
    except Exception as e:
        bus.emit("error", {"message": f"Discovery failed: {e}"})
        return

    families = {}
    for m in discovered:
        families.setdefault(m.family, []).append(m)

    bus.emit("discovery", {
        "total": len(discovered),
        "families": len(families),
        "family_names": sorted(families.keys()),
        "free_count": sum(1 for m in discovered if m.is_free),
    })

    # Phase 2: Select candidates
    bus.emit("phase", {"phase": "selection", "message": "Selecting election candidates..."})
    config = ElectionConfig(max_models=max_models, budget_usd=budget)
    candidates = select_candidates(discovered, config)
    estimated_cost = _estimate_election_cost(candidates)

    candidate_info = []
    for c in candidates:
        info = {
            "model_id": c.model_id,
            "name": c.name,
            "family": c.family,
            "param_size_b": c.param_size_b,
            "price": c.prompt_price_per_m,
            "is_free": c.is_free,
        }
        candidate_info.append(info)

    bus.emit("candidates", {
        "selected": len(candidates),
        "estimated_cost": estimated_cost,
        "families": len(set(c.family for c in candidates)),
        "candidates": candidate_info,
    })

    # Phase 3: Self-Assessment with AIMD
    bus.emit("phase", {"phase": "self_assessment", "message": "Round 1: Self-Assessment"})

    dispatcher = AIMDDispatcher(
        initial_parallelism=4,
        max_parallelism=min(len(candidates), 10),
        increase_threshold=5,
    )
    dispatcher.stats.start_time = time.time()

    # Emit AIMD stats periodically
    def aimd_monitor():
        while not election_done.is_set():
            bus.emit("aimd_stats", dispatcher.get_stats())
            time.sleep(1.0)

    election_done = threading.Event()
    monitor_thread = threading.Thread(target=aimd_monitor, daemon=True)
    monitor_thread.start()

    # Run self-assessment evals
    eval_names = list(EVAL_PROMPTS.keys())
    eval_prompts = list(EVAL_PROMPTS.values())
    model_outputs: dict[str, dict[str, str]] = {c.model_id: {} for c in candidates}

    threads: list[threading.Thread] = []

    for eval_name, prompt in zip(eval_names, eval_prompts):
        bus.emit("eval_start", {"eval_name": eval_name, "model_count": len(candidates)})

        eval_threads = []
        for candidate in candidates:
            task_id = f"{candidate.model_id}::{eval_name}"

            def worker(cand=candidate, ename=eval_name, p=prompt, tid=task_id):
                def on_stream(task_id: str, content: str, telemetry=None):
                    event_data = {
                        "task_id": task_id,
                        "model_id": cand.model_id,
                        "eval_name": ename,
                        "content": content,
                        "phase": "self_assessment",
                    }
                    if telemetry:
                        event_data["telemetry"] = telemetry.to_dict()
                    bus.emit("model_response", event_data)

                result = openrouter_worker(
                    task_id=tid,
                    model_id=cand.model_id,
                    prompt=p,
                    api_key=api_key,
                    dispatcher=dispatcher,
                    temperature=0.2,
                    on_stream=on_stream,
                )
                if result:
                    model_outputs[cand.model_id][ename] = result

            t = threading.Thread(target=worker, daemon=True)
            t.start()
            eval_threads.append(t)

        # Wait for this eval round to finish
        for t in eval_threads:
            t.join(timeout=90)

        bus.emit("eval_complete", {"eval_name": eval_name})

    bus.emit("phase", {"phase": "scoring", "message": "Scoring self-assessments..."})

    # Score and build assessments
    election_candidates = [
        ModelCandidate(
            model_id=c.model_id,
            display_name=c.name,
            param_size_b=c.param_size_b,
            family=c.family,
        )
        for c in candidates
    ]

    transport = OpenRouterTransport(max_workers=1)  # Won't use this for queries
    election = TierElection(candidates=election_candidates, transport=transport)

    assessments = []
    for candidate in election_candidates:
        outputs = model_outputs.get(candidate.model_id, {})
        if not outputs:
            continue

        evidence = election._score_outputs(outputs)
        proposed_tier = election._evidence_to_tier(evidence, candidate.param_size_b)
        confidence = sum(evidence.values()) / max(len(evidence), 1)

        from .tier_election import SelfAssessment
        assessment = SelfAssessment(
            model_id=candidate.model_id,
            proposed_tier=proposed_tier,
            confidence=confidence,
            evidence=evidence,
            raw_outputs=outputs,
        )
        assessments.append(assessment)

        bus.emit("assessment", {
            "model_id": candidate.model_id,
            "proposed_tier": proposed_tier,
            "confidence": confidence,
            "evidence": evidence,
            "scl": assessment.to_scl(),
        })

    # Phase 4: Cross-verification
    bus.emit("phase", {"phase": "cross_verification", "message": "Round 2: Cross-Verification"})

    verifications = []
    for assessment in assessments:
        prompt = election._build_verification_prompt(assessment)
        verifiers = [c for c in candidates if c.model_id != assessment.model_id]

        bus.emit("verify_start", {
            "target": assessment.model_id,
            "verifier_count": len(verifiers),
        })

        verify_threads = []
        for verifier in verifiers:
            task_id = f"verify::{verifier.model_id}→{assessment.model_id}"

            def vworker(v=verifier, a=assessment, p=prompt, tid=task_id):
                def on_stream(task_id: str, content: str, telemetry=None):
                    event_data = {
                        "task_id": task_id,
                        "model_id": v.model_id,
                        "eval_name": f"verify→{a.model_id}",
                        "content": content,
                        "phase": "cross_verification",
                    }
                    if telemetry:
                        event_data["telemetry"] = telemetry.to_dict()
                    bus.emit("model_response", event_data)

                result = openrouter_worker(
                    task_id=tid,
                    model_id=v.model_id,
                    prompt=p,
                    api_key=api_key,
                    dispatcher=dispatcher,
                    temperature=0.2,
                    on_stream=on_stream,
                )
                if result:
                    verification = election._parse_verification(v.model_id, a, result)
                    if verification:
                        verifications.append(verification)
                        bus.emit("verification", {
                            "verifier": v.model_id,
                            "target": a.model_id,
                            "proposed_tier": a.proposed_tier,
                            "agree": verification.agree,
                            "counter_tier": verification.counter_tier,
                            "confidence": verification.confidence,
                            "reasoning": verification.reasoning,
                            "scl": verification.to_scl(),
                        })

            t = threading.Thread(target=vworker, daemon=True)
            t.start()
            verify_threads.append(t)

        for t in verify_threads:
            t.join(timeout=90)

    # Phase 5: Consensus
    bus.emit("phase", {"phase": "consensus", "message": "Round 3: Resolving consensus..."})

    consensus_list, tier_map = election._resolve_consensus(assessments, verifications)

    for c in consensus_list:
        bus.emit("consensus_result", {
            "model_id": c.model_id,
            "tier": c.tier,
            "agreement_ratio": c.agreement_ratio,
            "total_voters": c.total_voters,
            "status": c.status,
            "scl": c.to_scl(),
        })

    # Final result
    election_done.set()

    bus.emit("complete", {
        "tier_map": tier_map,
        "converged": all(c.status == "accepted" for c in consensus_list),
        "elapsed_s": time.time() - dispatcher.stats.start_time,
        "total_requests": dispatcher.stats.total_dispatched,
        "congestion_events": dispatcher.stats.total_congestion,
        "final_parallelism": dispatcher.stats.current_parallelism,
    })

    # Save results
    output_dir = Path("/Volumes/CORTEX/cortex/data/tier_election")
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "tier_election.json"
    json_path.write_text(json.dumps({
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tier_map": tier_map,
        "assessments": [
            {"model_id": a.model_id, "proposed_tier": a.proposed_tier,
             "confidence": a.confidence, "evidence": a.evidence}
            for a in assessments
        ],
        "consensus": [
            {"model_id": c.model_id, "tier": c.tier,
             "agreement_ratio": c.agreement_ratio, "status": c.status}
            for c in consensus_list
        ],
    }, indent=2))

    bus.emit("saved", {"path": str(json_path)})


# ---------------------------------------------------------------------------
# Flask web app
# ---------------------------------------------------------------------------

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Cortex Tier Election</title>
<style>
:root {
  --bg: #0d1117;
  --surface: #161b22;
  --border: #30363d;
  --text: #e6edf3;
  --muted: #8b949e;
  --accent: #58a6ff;
  --green: #3fb950;
  --yellow: #d29922;
  --red: #f85149;
  --purple: #bc8cff;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  font-family: 'JetBrains Mono', 'SF Mono', monospace;
  background: var(--bg);
  color: var(--text);
  padding: 20px;
  max-width: 1400px;
  margin: 0 auto;
}
h1 { font-size: 1.4em; margin-bottom: 4px; }
h2 { font-size: 1.1em; color: var(--accent); margin: 16px 0 8px; }
.subtitle { color: var(--muted); font-size: 0.85em; margin-bottom: 20px; }

.stats-bar {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 8px;
  margin-bottom: 20px;
}
.stat-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 12px;
  text-align: center;
}
.stat-value { font-size: 1.8em; font-weight: bold; color: var(--accent); }
.stat-label { font-size: 0.75em; color: var(--muted); text-transform: uppercase; }

.phase-indicator {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 12px 16px;
  margin-bottom: 16px;
  display: flex;
  align-items: center;
  gap: 12px;
}
.phase-dot {
  width: 10px; height: 10px; border-radius: 50%;
  background: var(--green);
  animation: pulse 1.5s infinite;
}
@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.3; }
}
.phase-text { font-size: 0.9em; }

/* Toolbar */
.toolbar {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 8px 14px;
  margin-bottom: 12px;
  display: flex;
  align-items: center;
  gap: 6px;
  flex-wrap: wrap;
  font-size: 0.8em;
}
.toolbar-label { color: var(--muted); }
.toolbar-sep { color: var(--border); margin: 0 4px; }
.sort-btn {
  background: transparent;
  border: 1px solid var(--border);
  color: var(--text);
  padding: 3px 8px;
  border-radius: 4px;
  cursor: pointer;
  font-family: inherit;
  font-size: 0.85em;
}
.sort-btn:hover { border-color: var(--accent); color: var(--accent); }
.sort-btn.active { background: var(--accent); color: var(--bg); border-color: var(--accent); }
#page-size {
  background: var(--surface);
  border: 1px solid var(--border);
  color: var(--text);
  padding: 3px 6px;
  border-radius: 4px;
  font-family: inherit;
  font-size: 0.85em;
}

/* Table */
.table-wrap {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  overflow-x: auto;
  overflow-y: auto;
  max-height: 70vh;
  margin-bottom: 16px;
}
#model-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.8em;
}
#model-table thead { position: sticky; top: 0; z-index: 10; background: var(--surface); }
#model-table th {
  padding: 8px 10px;
  text-align: left;
  border-bottom: 2px solid var(--border);
  color: var(--muted);
  font-size: 0.85em;
  text-transform: uppercase;
  white-space: nowrap;
}
.th-sortable { cursor: pointer; }
.th-sortable:hover { color: var(--accent); }
.th-expand { width: 28px; }
#model-table td {
  padding: 6px 10px;
  border-bottom: 1px solid var(--border);
  vertical-align: middle;
}
.model-row { transition: background 0.15s; }
.model-row:hover { background: rgba(88, 166, 255, 0.04); }
.model-row.active { background: rgba(88, 166, 255, 0.06); }
.model-row.done td:first-child { border-left: 3px solid var(--green); }
.td-expand { cursor: pointer; text-align: center; color: var(--muted); user-select: none; }
.td-expand:hover { color: var(--accent); }
.td-model { max-width: 260px; }
.model-name-cell { font-weight: 600; font-size: 0.95em; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.model-id-cell { font-size: 0.8em; color: var(--purple); opacity: 0.8; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.td-provider { color: var(--muted); font-size: 0.9em; }
.td-num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
.td-cost { color: var(--yellow); }
.td-status { text-align: center; }
.tier-badge {
  background: var(--accent);
  color: var(--bg);
  padding: 2px 7px;
  border-radius: 4px;
  font-size: 0.85em;
  font-weight: bold;
}
.tier-badge.pending { background: var(--muted); }

/* Detail (expanded) row */
.detail-row { background: var(--bg); }
.detail-row.hidden { display: none; }
.detail-wrap {
  padding: 12px 16px;
  max-height: 400px;
  overflow-y: auto;
}
.detail-header { font-size: 0.8em; color: var(--purple); margin-bottom: 8px; }
.detail-empty { color: var(--muted); font-style: italic; font-size: 0.85em; }
.detail-eval {
  border: 1px solid var(--border);
  border-radius: 6px;
  margin-bottom: 8px;
  overflow: hidden;
}
.detail-eval-name {
  background: rgba(88, 166, 255, 0.06);
  padding: 4px 10px;
  font-size: 0.75em;
  color: var(--muted);
  text-transform: uppercase;
  border-bottom: 1px solid var(--border);
}
.detail-telemetry {
  display: flex;
  gap: 10px;
  padding: 4px 10px;
  font-size: 0.75em;
  color: var(--green);
  border-bottom: 1px dashed var(--border);
  flex-wrap: wrap;
}
.detail-telemetry span { white-space: nowrap; }
.detail-provider { color: var(--muted); font-style: italic; }
.detail-eval-content {
  padding: 6px 10px;
  font-size: 0.8em;
  white-space: pre-wrap;
  word-break: break-word;
  max-height: 150px;
  overflow-y: auto;
}
.detail-section-title { font-size: 0.75em; color: var(--muted); text-transform: uppercase; margin: 8px 0 4px; }
.detail-vote { font-size: 0.8em; padding: 2px 0; display: flex; gap: 8px; }
.detail-voter { color: var(--muted); }
.vote-agree { color: var(--green); }
.vote-disagree { color: var(--red); }

/* Consensus */
.consensus-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 8px;
}
.consensus-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 12px;
}
.consensus-card.accepted { border-color: var(--green); }
.consensus-card.disputed { border-color: var(--yellow); }
.consensus-scl { font-size: 0.8em; color: var(--purple); margin-top: 6px; }

/* AIMD bar */
.aimd-bar {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 10px 14px;
  margin-bottom: 16px;
  display: flex;
  gap: 20px;
  align-items: center;
  font-size: 0.8em;
}
.aimd-label { color: var(--muted); }
.aimd-value { color: var(--accent); font-weight: bold; }

/* Log */
#log {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 12px;
  max-height: 200px;
  overflow-y: auto;
  font-size: 0.75em;
  color: var(--muted);
  margin-top: 16px;
}
.log-entry { padding: 2px 0; }
</style>
</head>
<body>
<h1>⬡ Cortex Tier Election</h1>
<p class="subtitle">Gossip protocol applied to model identity — dynamic discovery → self-assessment → consensus</p>

<div class="phase-indicator" id="phase">
  <div class="phase-dot"></div>
  <span class="phase-text">Initializing...</span>
</div>

<div class="aimd-bar" id="aimd-bar">
  <span><span class="aimd-label">Parallelism:</span> <span class="aimd-value" id="aimd-parallelism">4</span></span>
  <span><span class="aimd-label">Epoch:</span> <span class="aimd-value" id="aimd-epoch">0</span></span>
  <span><span class="aimd-label">Completed:</span> <span class="aimd-value" id="aimd-completed">0</span></span>
  <span><span class="aimd-label">In-flight:</span> <span class="aimd-value" id="aimd-inflight">0</span></span>
  <span><span class="aimd-label">Congestion:</span> <span class="aimd-value" id="aimd-congestion">0</span></span>
  <span><span class="aimd-label">RPS:</span> <span class="aimd-value" id="aimd-rps">0.0</span></span>
</div>

<div class="stats-bar" id="stats-bar">
  <div class="stat-card"><div class="stat-value" id="stat-discovered">—</div><div class="stat-label">Discovered</div></div>
  <div class="stat-card"><div class="stat-value" id="stat-candidates">—</div><div class="stat-label">Candidates</div></div>
  <div class="stat-card"><div class="stat-value" id="stat-families">—</div><div class="stat-label">Families</div></div>
  <div class="stat-card"><div class="stat-value" id="stat-cost">—</div><div class="stat-label">Actual Cost</div></div>
</div>

<div class="toolbar">
  <span class="toolbar-label">Sort by:</span>
  <button class="sort-btn active" data-sort="ttft" onclick="setSort('ttft')">TTFT</button>
  <button class="sort-btn" data-sort="cost" onclick="setSort('cost')">Cost</button>
  <button class="sort-btn" data-sort="tokens_out" onclick="setSort('tokens_out')">Tok Out</button>
  <button class="sort-btn" data-sort="tokens_in" onclick="setSort('tokens_in')">Tok In</button>
  <button class="sort-btn" data-sort="total_s" onclick="setSort('total_s')">Total Time</button>
  <button class="sort-btn" data-sort="model" onclick="setSort('model')">Model</button>
  <button class="sort-btn" data-sort="provider" onclick="setSort('provider')">Provider</button>
  <span class="toolbar-sep">|</span>
  <button class="sort-btn" id="sort-dir-btn" onclick="toggleSortDir()">ASC</button>
  <span class="toolbar-sep">|</span>
  <span class="toolbar-label">Page:</span>
  <button class="sort-btn" onclick="prevPage()">&#x25C0;</button>
  <span id="page-indicator" class="toolbar-label">1/1</span>
  <button class="sort-btn" onclick="nextPage()">&#x25B6;</button>
  <select id="page-size" onchange="setPageSize(this.value)">
    <option value="10">10/page</option>
    <option value="25" selected>25/page</option>
    <option value="50">50/page</option>
    <option value="100">100/page</option>
  </select>
</div>

<div class="table-wrap" id="table-wrap">
  <table id="model-table">
    <thead>
      <tr>
        <th class="th-expand"></th>
        <th class="th-sortable" onclick="setSort('model')">Model</th>
        <th class="th-sortable" onclick="setSort('provider')">Provider</th>
        <th class="th-sortable" onclick="setSort('ttft')">TTFT</th>
        <th class="th-sortable" onclick="setSort('tokens_in')">Tok In</th>
        <th class="th-sortable" onclick="setSort('tokens_out')">Tok Out</th>
        <th class="th-sortable" onclick="setSort('cached')">Cached</th>
        <th class="th-sortable" onclick="setSort('reasoning')">Reason</th>
        <th class="th-sortable" onclick="setSort('total_s')">Total</th>
        <th class="th-sortable" onclick="setSort('cost')">Cost</th>
        <th>Tier</th>
        <th>Status</th>
      </tr>
    </thead>
    <tbody id="model-tbody"></tbody>
  </table>
</div>

<h2 id="consensus-heading" style="display:none">Consensus</h2>
<div class="consensus-grid" id="consensus-grid"></div>

<div id="log"></div>

<script>
// --- State ---
const models = {};        // modelId -> {info, responses: {evalName: {content, telemetry}}, tier, status, verifications}
let totalCost = 0;
let sortKey = 'ttft';
let sortAsc = true;
let currentPage = 0;
let pageSize = 25;

// --- Model data management ---
function ensureModel(modelId, info) {
  if (!models[modelId]) {
    models[modelId] = {
      info: info || {},
      responses: {},
      tier: null,
      status: 'pending',
      verifications: [],
      avgTtft: null,
      avgCost: null,
      totalTokensIn: 0,
      totalTokensOut: 0,
      totalCached: 0,
      totalReasoning: 0,
      totalTime: 0,
      lastProvider: '',
      requestCount: 0,
    };
  }
  return models[modelId];
}

function addResponse(modelId, evalName, content, phase, telemetry) {
  const m = ensureModel(modelId);
  m.responses[evalName] = { content, telemetry, phase };
  m.status = 'active';

  if (telemetry) {
    m.requestCount++;
    m.totalTokensIn += telemetry.tokens_in || 0;
    m.totalTokensOut += telemetry.tokens_out || 0;
    m.totalCached += telemetry.tokens_cached || 0;
    m.totalReasoning += telemetry.tokens_reasoning || 0;
    m.totalTime += telemetry.total_s || 0;
    m.lastProvider = telemetry.provider || m.lastProvider;
    // Running average TTFT
    m.avgTtft = m.totalTime / m.requestCount;
    // Running total cost
    if (telemetry.cost_usd) {
      m.avgCost = (m.avgCost || 0) + telemetry.cost_usd;
      totalCost += telemetry.cost_usd;
      document.getElementById('stat-cost').textContent = '$' + totalCost.toFixed(4);
    }
  }
  renderTable();
}

function setTier(modelId, tier) {
  const m = ensureModel(modelId);
  m.tier = tier;
  m.status = 'done';
  renderTable();
}

function addVerification(modelId, data) {
  const m = ensureModel(modelId);
  m.verifications.push(data);
  renderTable();
}

// --- Sorting ---
function getSortValue(m, key) {
  switch (key) {
    case 'ttft': return m.avgTtft || 9999;
    case 'cost': return m.avgCost || 0;
    case 'tokens_out': return m.totalTokensOut;
    case 'tokens_in': return m.totalTokensIn;
    case 'total_s': return m.totalTime;
    case 'cached': return m.totalCached;
    case 'reasoning': return m.totalReasoning;
    case 'model': return (m.info.model_id || '').toLowerCase();
    case 'provider': return (m.lastProvider || '').toLowerCase();
    default: return 0;
  }
}

function getSortedModels() {
  const arr = Object.entries(models);
  arr.sort((a, b) => {
    let va = getSortValue(a[1], sortKey);
    let vb = getSortValue(b[1], sortKey);
    if (typeof va === 'string') {
      return sortAsc ? va.localeCompare(vb) : vb.localeCompare(va);
    }
    return sortAsc ? va - vb : vb - va;
  });
  return arr;
}

function setSort(key) {
  if (sortKey === key) { sortAsc = !sortAsc; }
  else { sortKey = key; sortAsc = true; }
  document.querySelectorAll('.sort-btn').forEach(b => b.classList.remove('active'));
  const btn = document.querySelector(`.sort-btn[data-sort="${key}"]`);
  if (btn) btn.classList.add('active');
  document.getElementById('sort-dir-btn').textContent = sortAsc ? 'ASC' : 'DESC';
  currentPage = 0;
  renderTable();
}

function toggleSortDir() {
  sortAsc = !sortAsc;
  document.getElementById('sort-dir-btn').textContent = sortAsc ? 'ASC' : 'DESC';
  renderTable();
}

// --- Pagination ---
function setPageSize(val) { pageSize = parseInt(val); currentPage = 0; renderTable(); }
function nextPage() { currentPage++; renderTable(); }
function prevPage() { if (currentPage > 0) currentPage--; renderTable(); }

// --- Render ---
function renderTable() {
  const sorted = getSortedModels();
  const totalPages = Math.max(1, Math.ceil(sorted.length / pageSize));
  if (currentPage >= totalPages) currentPage = totalPages - 1;
  document.getElementById('page-indicator').textContent = `${currentPage + 1}/${totalPages}`;

  const start = currentPage * pageSize;
  const page = sorted.slice(start, start + pageSize);

  const tbody = document.getElementById('model-tbody');
  tbody.innerHTML = '';

  for (const [modelId, m] of page) {
    const tr = document.createElement('tr');
    tr.className = `model-row ${m.status}`;
    tr.dataset.modelId = modelId;

    const tierBadge = m.tier ? `<span class="tier-badge">${m.tier}</span>` : '<span class="tier-badge pending">...</span>';
    const statusDot = m.status === 'done' ? '&#x2714;' : (m.status === 'active' ? '&#x25CF;' : '&#x25CB;');
    const ttft = m.avgTtft ? m.avgTtft.toFixed(2) + 's' : '—';
    const cost = m.avgCost ? '$' + m.avgCost.toFixed(5) : '—';
    const displayModel = m.info.model_id || modelId;

    tr.innerHTML = `
      <td class="td-expand" onclick="toggleDetail('${modelId.replace(/'/g, "\\'")}')">&#x25B6;</td>
      <td class="td-model">
        <div class="model-name-cell">${m.info.name || modelId}</div>
        <div class="model-id-cell">${displayModel}</div>
      </td>
      <td class="td-provider">${m.lastProvider || '—'}</td>
      <td class="td-num">${ttft}</td>
      <td class="td-num">${m.totalTokensIn || '—'}</td>
      <td class="td-num">${m.totalTokensOut || '—'}</td>
      <td class="td-num">${m.totalCached || '—'}</td>
      <td class="td-num">${m.totalReasoning || '—'}</td>
      <td class="td-num">${m.totalTime ? m.totalTime.toFixed(1) + 's' : '—'}</td>
      <td class="td-num td-cost">${cost}</td>
      <td>${tierBadge}</td>
      <td class="td-status">${statusDot}</td>
    `;
    tbody.appendChild(tr);

    // Detail row (hidden by default)
    const detailTr = document.createElement('tr');
    detailTr.className = 'detail-row hidden';
    detailTr.id = 'detail-' + modelId.replace(/[^a-zA-Z0-9]/g, '_');
    const detailContent = buildDetailContent(modelId, m);
    detailTr.innerHTML = `<td colspan="12"><div class="detail-wrap">${detailContent}</div></td>`;
    tbody.appendChild(detailTr);
  }
}

function buildDetailContent(modelId, m) {
  let html = `<div class="detail-header">openrouter.ai/api/v1/chat/completions &rarr; <strong>${m.info.model_id || modelId}</strong></div>`;

  // Responses
  const evalNames = Object.keys(m.responses);
  if (evalNames.length === 0) {
    html += '<div class="detail-empty">Waiting for responses...</div>';
  }
  for (const evalName of evalNames) {
    const r = m.responses[evalName];
    const t = r.telemetry;
    let tBar = '';
    if (t) {
      const cost = t.cost_usd ? '$' + t.cost_usd.toFixed(6) : '';
      tBar = `<div class="detail-telemetry">
        <span>&#x23F1; ${t.ttft_s?.toFixed(2) || t.total_s?.toFixed(2) || '?'}s</span>
        <span>&rarr;${t.tokens_in || 0}</span>
        <span>&larr;${t.tokens_out || 0}</span>
        <span>&#x1F4BE;${t.tokens_cached || 0}</span>
        <span>&#x1F9E0;${t.tokens_reasoning || 0}</span>
        <span>${cost}</span>
        <span class="detail-provider">${t.provider || ''}</span>
      </div>`;
    }
    html += `<div class="detail-eval">
      <div class="detail-eval-name">${evalName}</div>
      ${tBar}
      <div class="detail-eval-content">${escapeHtml(r.content || '')}</div>
    </div>`;
  }

  // Verifications
  if (m.verifications.length > 0) {
    html += '<div class="detail-section-title">Cross-Verification Votes</div>';
    for (const v of m.verifications) {
      const cls = v.agree ? 'vote-agree' : 'vote-disagree';
      const txt = v.agree ? '&#x2713; agree' : `&#x2717; &rarr; ${v.counter_tier || '?'}`;
      html += `<div class="detail-vote"><span class="${cls}">${txt}</span> <span class="detail-voter">${v.verifier}</span></div>`;
    }
  }

  return html;
}

function escapeHtml(str) {
  return str.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function toggleDetail(modelId) {
  const id = 'detail-' + modelId.replace(/[^a-zA-Z0-9]/g, '_');
  const row = document.getElementById(id);
  if (!row) return;
  row.classList.toggle('hidden');
  // Rotate arrow
  const mainRow = row.previousElementSibling;
  if (mainRow) {
    const arrow = mainRow.querySelector('.td-expand');
    if (arrow) arrow.innerHTML = row.classList.contains('hidden') ? '&#x25B6;' : '&#x25BC;';
  }
}

// --- Consensus ---
function addConsensus(data) {
  document.getElementById('consensus-heading').style.display = '';
  const grid = document.getElementById('consensus-grid');
  const card = document.createElement('div');
  card.className = `consensus-card ${data.status}`;
  card.innerHTML = `
    <div style="display:flex;justify-content:space-between;align-items:center">
      <strong>${data.model_id}</strong>
      <span class="tier-badge">${data.tier}</span>
    </div>
    <div style="font-size:0.8em;color:var(--muted);margin-top:4px">
      Agreement: ${(data.agreement_ratio * 100).toFixed(0)}% &middot; ${data.status}
    </div>
    <div class="consensus-scl">${data.scl || ''}</div>
  `;
  grid.appendChild(card);
  setTier(data.model_id, data.tier);
}

function log(msg) {
  const el = document.getElementById('log');
  const entry = document.createElement('div');
  entry.className = 'log-entry';
  entry.textContent = new Date().toLocaleTimeString() + ' ' + msg;
  el.appendChild(entry);
  el.scrollTop = el.scrollHeight;
}

// --- SSE ---
const evtSource = new EventSource('/events');

evtSource.onmessage = function(e) {
  const event = JSON.parse(e.data);
  const { type, data } = event;

  switch (type) {
    case 'phase':
      document.querySelector('.phase-text').textContent = data.message;
      log(`Phase: ${data.phase}`);
      break;

    case 'discovery':
      document.getElementById('stat-discovered').textContent = data.total;
      document.getElementById('stat-families').textContent = data.families;
      log(`Discovered ${data.total} models across ${data.families} families`);
      break;

    case 'candidates':
      document.getElementById('stat-candidates').textContent = data.selected;
      document.getElementById('stat-cost').textContent = '$' + data.estimated_cost.toFixed(3);
      data.candidates.forEach(c => ensureModel(c.model_id, c));
      renderTable();
      log(`Selected ${data.selected} candidates`);
      break;

    case 'aimd_stats':
      document.getElementById('aimd-parallelism').textContent = data.parallelism;
      document.getElementById('aimd-epoch').textContent = data.epoch;
      document.getElementById('aimd-completed').textContent = data.completed;
      document.getElementById('aimd-inflight').textContent = data.in_flight;
      document.getElementById('aimd-congestion').textContent = data.congestion_events;
      document.getElementById('aimd-rps').textContent = data.throughput_rps.toFixed(1);
      break;

    case 'model_response':
      addResponse(data.model_id, data.eval_name, data.content, data.phase, data.telemetry);
      break;

    case 'assessment':
      setTier(data.model_id, data.proposed_tier);
      log(`${data.model_id} &rarr; self_assess ${data.proposed_tier} (conf=${data.confidence.toFixed(2)})`);
      break;

    case 'verification':
      addVerification(data.target, data);
      break;

    case 'consensus_result':
      addConsensus(data);
      log(`CONSENSUS: ${data.model_id} &rarr; ${data.tier} [${data.status}]`);
      break;

    case 'complete':
      document.querySelector('.phase-dot').style.animation = 'none';
      document.querySelector('.phase-dot').style.background = 'var(--green)';
      document.querySelector('.phase-text').textContent =
        `Election complete! ${Object.keys(data.tier_map).length} models in ${data.elapsed_s.toFixed(1)}s`;
      log(`Done: ${JSON.stringify(data.tier_map)}`);
      break;

    case 'error':
      log(`ERROR: ${data.message}`);
      document.querySelector('.phase-dot').style.background = 'var(--red)';
      document.querySelector('.phase-dot').style.animation = 'none';
      break;

    default:
      log(`[${type}] ${JSON.stringify(data).slice(0, 100)}`);
  }
};

evtSource.onerror = function() {
  log('SSE connection lost, retrying...');
};
</script>
</body>
</html>"""


def create_app(api_key: str, max_models: int = 8, budget: float = 0.25):
    """Create Flask app for the election dashboard."""
    try:
        from flask import Flask, Response
    except ImportError:
        raise RuntimeError("Flask not installed: pip install flask")

    app = Flask(__name__)
    bus = EventBus()

    # Start election in background
    election_thread = threading.Thread(
        target=run_election_background,
        args=(bus, api_key, max_models, budget),
        daemon=True,
    )

    @app.route("/")
    def index():
        return Response(HTML_TEMPLATE, mimetype="text/html")

    @app.route("/events")
    def events():
        def stream():
            q = bus.subscribe()
            try:
                while True:
                    try:
                        event = q.get(timeout=30)
                        yield f"data: {json.dumps(event)}\n\n"
                    except queue.Empty:
                        yield f"data: {json.dumps({'type': 'ping', 'data': {}})}\n\n"
            except GeneratorExit:
                bus.unsubscribe(q)

        return Response(stream(), mimetype="text/event-stream",
                       headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.route("/start", methods=["POST"])
    def start():
        if not election_thread.is_alive():
            election_thread.start()
            return json.dumps({"status": "started"})
        return json.dumps({"status": "already_running"})

    return app, election_thread


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Cortex Tier Election Dashboard")
    parser.add_argument("--port", type=int, default=8420)
    parser.add_argument("--max-models", type=int, default=8)
    parser.add_argument("--budget", type=float, default=0.25)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    # Load API key
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        env_path = Path("/Volumes/CORTEX/cortex/bin/.env")
        if env_path.exists():
            for line in env_path.read_text().strip().split("\n"):
                if line.startswith("OPENROUTER_API_KEY="):
                    api_key = line.split("=", 1)[1].strip()
                    os.environ["OPENROUTER_API_KEY"] = api_key
                    break

    if not api_key:
        print("ERROR: No OPENROUTER_API_KEY found")
        return

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    app, election_thread = create_app(api_key, args.max_models, args.budget)

    print(f"╔══════════════════════════════════════════════════════════════╗")
    print(f"║  Cortex Tier Election Dashboard                              ║")
    print(f"║  http://localhost:{args.port}                                      ║")
    print(f"╚══════════════════════════════════════════════════════════════╝")
    print()

    # Auto-start election after first client connects
    election_thread.start()

    if not args.no_browser:
        import webbrowser
        webbrowser.open(f"http://localhost:{args.port}")

    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
