"""
CKM Training Data Generator.

Converts SCL DeltaStreams into training pairs for the 0.3B kernel model.

Training format:
  Input:  SCL context (hardware state + recent deltas + request)
  Output: SCL response (optimal mutation or routing decision)

Sources of training data:
  1. Boot telemetry logs (hardware → config mapping)
  2. Inference routing decisions (request → tier mapping)
  3. Policy rewriter mutations (feedback → policy changes)
  4. Gossip convergence outcomes (multi-agent agreement)

The model learns to:
  - Map hardware descriptions → optimal inference config
  - Route requests → appropriate tier/model
  - Predict when to escalate (challenger/swarm)
  - Propose policy mutations from feedback patterns
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Iterator

from ..scl.types import SCLRecord, SCLDocument, Anchor, Relation, Scope
from ..scl.delta import Delta, SemanticState, DeltaStream
from ..scl.emitter import emit_record, emit_document
from ..scl.parser import parse_record, parse_document
from ..braille.fingerprint import fingerprint as scl_fingerprint


# ---------------------------------------------------------------------------
# Training pair format
# ---------------------------------------------------------------------------

@dataclass
class TrainingPair:
    """One input→output pair for CKM training.

    Both input and output are SCL text. The model learns:
      given this SCL context → produce this SCL response.
    """
    input_scl: str           # Context: hardware + history + request
    output_scl: str          # Target: optimal mutation/decision
    source: str = ""         # boot, route, policy, gossip
    quality: float = 1.0     # Weight (higher = more reliable training signal)
    timestamp_ms: int = 0

    def to_jsonl(self) -> str:
        """Serialize as JSONL for training."""
        return json.dumps({
            "input": self.input_scl,
            "output": self.output_scl,
            "source": self.source,
            "quality": self.quality,
        })

    @classmethod
    def from_jsonl(cls, line: str) -> "TrainingPair":
        d = json.loads(line)
        return cls(
            input_scl=d["input"],
            output_scl=d["output"],
            source=d.get("source", ""),
            quality=d.get("quality", 1.0),
        )


# ---------------------------------------------------------------------------
# Boot telemetry → training pairs
# ---------------------------------------------------------------------------

def generate_boot_pairs(delta_stream: DeltaStream) -> list[TrainingPair]:
    """Convert a boot DeltaStream into training pairs.

    For each boot, the training pair is:
      Input:  hardware state at boot time (SCL snapshot)
      Output: the optimal config that was computed (SCL mutate)

    This teaches the model: "given THIS hardware, output THAT config."
    """
    pairs = []
    state = SemanticState()

    for i, delta in enumerate(delta_stream.deltas):
        # Build progressive state
        from ..scl.delta import apply_delta
        new_state = apply_delta(state, delta)

        # If this delta contains optimization results, it's a training target
        if "optimal_threads" in delta.set_keys or "optimal_gpu_layers" in delta.set_keys:
            # Input: the hardware state BEFORE optimization
            hw_keys = {k: v for k, v in state.entries.items()
                       if k.startswith("last_") or k in ("boot_count",)}
            input_record = SCLRecord(
                anchor=Anchor("hardware"),
                relation=Relation("state"),
                scope=Scope(entries=hw_keys),
            )

            # Output: the optimization delta
            output_record = delta.to_scl()

            pair = TrainingPair(
                input_scl=emit_record(input_record),
                output_scl=emit_record(output_record),
                source="boot",
                quality=_compute_boot_quality(new_state),
                timestamp_ms=delta.timestamp_ms,
            )
            pairs.append(pair)

        # If this delta is a boot event, pair it with the NEXT optimization
        if "last_boot_ms" in delta.set_keys and i + 1 < len(delta_stream.deltas):
            next_delta = delta_stream.deltas[i + 1]
            if "optimal_threads" in next_delta.set_keys:
                # Input: full boot event
                input_record = delta.to_scl()
                # Output: resulting optimization
                output_record = next_delta.to_scl()
                pair = TrainingPair(
                    input_scl=emit_record(input_record),
                    output_scl=emit_record(output_record),
                    source="boot_chain",
                    quality=_compute_boot_quality(new_state),
                    timestamp_ms=delta.timestamp_ms,
                )
                pairs.append(pair)

        state = new_state

    return pairs


def _compute_boot_quality(state: SemanticState) -> float:
    """Quality score for a boot training pair.

    Higher boot_count = more signal = higher quality.
    Lower boot time = better outcome = higher quality.
    """
    boot_count = int(state.entries.get("boot_count", "1"))
    best_ms = float(state.entries.get("best_boot_ms", "5000"))

    # More boots = more signal
    count_score = min(1.0, boot_count / 10)
    # Faster boot = better training signal
    speed_score = max(0.1, 1.0 - best_ms / 10000)

    return round((count_score + speed_score) / 2, 3)


# ---------------------------------------------------------------------------
# Routing decisions → training pairs
# ---------------------------------------------------------------------------

def generate_routing_pairs(audit_records: list[SCLRecord]) -> list[TrainingPair]:
    """Convert routing audit trail into training pairs.

    For each route decision, the training pair is:
      Input:  request classification (category, complexity, tokens)
      Output: routing decision (tier, model, confidence)

    This teaches the model: "given THIS request shape, route to THAT tier."
    """
    pairs = []

    for i, record in enumerate(audit_records):
        if record.relation.verb == "classify":
            # This is the input — find the corresponding route decision
            if i + 1 < len(audit_records) and audit_records[i + 1].relation.verb == "select":
                route_record = audit_records[i + 1]
                pair = TrainingPair(
                    input_scl=emit_record(record),
                    output_scl=emit_record(route_record),
                    source="route",
                    quality=float(route_record.scope.entries.get("confidence", "0.5")),
                    timestamp_ms=record.timestamp_ms,
                )
                pairs.append(pair)

    return pairs


# ---------------------------------------------------------------------------
# Policy mutations → training pairs
# ---------------------------------------------------------------------------

def generate_policy_pairs(
    mutation_records: list[SCLRecord],
    feedback_records: list[SCLRecord],
) -> list[TrainingPair]:
    """Convert policy rewriter actions into training pairs.

    Input:  accumulated feedback pattern (accuracy stats)
    Output: policy mutation (tier penalty, model blocklist)

    This teaches the model: "given THIS feedback pattern, mutate THAT policy."
    """
    pairs = []

    for mutation in mutation_records:
        if mutation.relation.verb != "mutate":
            continue

        # Find related feedback that triggered this mutation
        mutation_time = mutation.timestamp_ms
        relevant_feedback = [
            f for f in feedback_records
            if abs(f.timestamp_ms - mutation_time) < 300_000  # within 5 min
        ]

        if relevant_feedback:
            # Combine feedback into input context
            feedback_entries = {}
            for f in relevant_feedback[-5:]:  # last 5 feedback items
                feedback_entries.update(f.scope.entries)

            input_record = SCLRecord(
                anchor=Anchor("feedback"),
                relation=Relation("accumulated"),
                scope=Scope(entries=feedback_entries),
            )
            pair = TrainingPair(
                input_scl=emit_record(input_record),
                output_scl=emit_record(mutation),
                source="policy",
                quality=float(mutation.scope.entries.get("confidence", "0.5")),
                timestamp_ms=mutation_time,
            )
            pairs.append(pair)

    return pairs


# ---------------------------------------------------------------------------
# Synthetic data generation (bootstrapping)
# ---------------------------------------------------------------------------

def generate_synthetic_boot_data(count: int = 1000) -> list[TrainingPair]:
    """Generate synthetic training pairs for bootstrapping.

    Since real boot telemetry takes time to accumulate, we generate
    synthetic hardware→config mappings based on known heuristics.
    Once the model is trained on these, it replaces the heuristics.

    This is the "distillation" step: heuristics → model.
    """
    import random
    pairs = []

    # Hardware configurations we know about
    hardware_profiles = [
        # (cpu, cores, ram_mb, gpu_type, gpu_vram_mb, arch)
        ("Intel Core i7-12700K", 20, 32768, "nvidia", 8192, "x86_64"),
        ("Intel Core i5-10400", 12, 16384, "nvidia", 4096, "x86_64"),
        ("Intel Core i3-10100", 8, 8192, "none", 0, "x86_64"),
        ("AMD Ryzen 9 7950X", 32, 65536, "nvidia", 24576, "x86_64"),
        ("AMD Ryzen 7 5800X", 16, 32768, "nvidia", 12288, "x86_64"),
        ("AMD Ryzen 5 5600X", 12, 16384, "amd", 8192, "x86_64"),
        ("Apple M1", 8, 16384, "apple", 16384, "aarch64"),
        ("Apple M1 Pro", 10, 16384, "apple", 16384, "aarch64"),
        ("Apple M1 Max", 10, 32768, "apple", 32768, "aarch64"),
        ("Apple M2", 8, 8192, "apple", 8192, "aarch64"),
        ("Apple M2 Pro", 12, 16384, "apple", 16384, "aarch64"),
        ("Apple M2 Max", 12, 32768, "apple", 32768, "aarch64"),
        ("Apple M3", 8, 8192, "apple", 8192, "aarch64"),
        ("Apple M3 Pro", 12, 18432, "apple", 18432, "aarch64"),
        ("Apple M3 Max", 16, 36864, "apple", 36864, "aarch64"),
        ("Apple M4", 10, 16384, "apple", 16384, "aarch64"),
        ("Qualcomm Snapdragon 8cx", 8, 8192, "none", 0, "aarch64"),
        ("Intel Celeron N5105", 4, 4096, "none", 0, "x86_64"),
        ("Raspberry Pi 5 BCM2712", 4, 4096, "none", 0, "aarch64"),
        ("Raspberry Pi 4 BCM2711", 4, 2048, "none", 0, "aarch64"),
        ("NVIDIA Jetson Orin", 12, 16384, "nvidia", 8192, "aarch64"),
        ("NVIDIA Jetson Nano", 4, 4096, "nvidia", 2048, "aarch64"),
    ]

    for _ in range(count):
        hw = random.choice(hardware_profiles)
        cpu, cores, ram, gpu_type, gpu_vram, arch = hw

        # Add noise to make training more robust
        cores_noise = max(1, cores + random.randint(-1, 1))
        ram_noise = ram + random.randint(-1024, 1024)

        # Compute optimal config (THE HEURISTICS we're distilling)
        opt_threads = max(1, min(cores_noise - 1, 16))
        if gpu_type == "none":
            opt_gpu_layers = 0
            opt_ctx = min(4096, ram // 4)
            hot_models = ["L0"] if ram >= 4096 else []
        elif gpu_type == "apple":
            opt_gpu_layers = 999  # Metal: all layers
            opt_ctx = min(16384, gpu_vram // 2)
            if gpu_vram >= 32768:
                hot_models = ["L0", "L1", "L2", "L3", "L4"]
            elif gpu_vram >= 16384:
                hot_models = ["L0", "L1", "L2", "L3"]
            elif gpu_vram >= 8192:
                hot_models = ["L0", "L1", "L2"]
            else:
                hot_models = ["L0", "L1"]
        elif gpu_type == "nvidia":
            if gpu_vram >= 24576:
                opt_gpu_layers = 999
                opt_ctx = 16384
                hot_models = ["L0", "L1", "L2", "L3", "L4"]
            elif gpu_vram >= 8192:
                opt_gpu_layers = 999
                opt_ctx = 8192
                hot_models = ["L0", "L1", "L2", "L3"]
            elif gpu_vram >= 4096:
                opt_gpu_layers = 32
                opt_ctx = 4096
                hot_models = ["L0", "L1", "L2"]
            else:
                opt_gpu_layers = 16
                opt_ctx = 2048
                hot_models = ["L0", "L1"]
        else:  # amd
            opt_gpu_layers = 999 if gpu_vram >= 8192 else 24
            opt_ctx = min(8192, gpu_vram // 2)
            hot_models = ["L0", "L1", "L2"] if gpu_vram >= 8192 else ["L0", "L1"]

        opt_backend = "llama_cpp" if gpu_type != "none" or ram >= 8192 else "llama_cpp"
        opt_batch = 8 if opt_gpu_layers > 0 else 4

        # Construct SCL training pair
        input_record = SCLRecord(
            anchor=Anchor("hardware"),
            relation=Relation("state"),
            scope=Scope(entries={
                "cpu": cpu,
                "cores": str(cores_noise),
                "ram_mb": str(ram_noise),
                "gpu_type": gpu_type,
                "vram_mb": str(gpu_vram),
                "arch": arch,
            }),
        )

        output_record = SCLRecord(
            anchor=Anchor("cortex.boot"),
            relation=Relation("mutate"),
            scope=Scope(entries={
                "optimal_threads": str(opt_threads),
                "optimal_gpu_layers": str(opt_gpu_layers),
                "optimal_ctx_size": str(opt_ctx),
                "optimal_batch_size": str(opt_batch),
                "optimal_backend": opt_backend,
                "optimal_hot_models": ",".join(hot_models),
            }),
        )

        pair = TrainingPair(
            input_scl=emit_record(input_record),
            output_scl=emit_record(output_record),
            source="synthetic_boot",
            quality=0.7,  # Synthetic is lower quality than real telemetry
            timestamp_ms=int(time.time() * 1000),
        )
        pairs.append(pair)

    return pairs


def generate_synthetic_routing_data(count: int = 2000) -> list[TrainingPair]:
    """Generate synthetic routing training pairs.

    Maps request characteristics → tier selection.
    """
    import random
    pairs = []

    categories = [
        ("code", "generate_function", 0.6, "L3"),
        ("code", "fix_bug", 0.7, "L4"),
        ("code", "explain_code", 0.3, "L2"),
        ("code", "autocomplete", 0.1, "L1"),
        ("chat", "greeting", 0.05, "L0"),
        ("chat", "casual_qa", 0.2, "L1"),
        ("chat", "complex_reasoning", 0.8, "L5"),
        ("chat", "creative_writing", 0.5, "L3"),
        ("math", "arithmetic", 0.1, "L1"),
        ("math", "algebra", 0.4, "L2"),
        ("math", "proof", 0.9, "L5"),
        ("tool", "web_search", 0.3, "L2"),
        ("tool", "code_execution", 0.5, "L3"),
        ("tool", "multi_tool_chain", 0.8, "L4"),
        ("analysis", "summarize", 0.3, "L2"),
        ("analysis", "compare", 0.5, "L3"),
        ("analysis", "deep_analysis", 0.85, "L5"),
    ]

    for _ in range(count):
        cat, subtype, complexity, tier = random.choice(categories)
        # Add noise
        complexity = max(0.0, min(1.0, complexity + random.gauss(0, 0.1)))
        tokens = random.randint(10, 2000)
        confidence = max(0.3, min(0.99, 1.0 - abs(random.gauss(0, 0.15))))

        input_record = SCLRecord(
            anchor=Anchor("task"),
            relation=Relation("classify"),
            scope=Scope(entries={
                "category": cat,
                "subtype": subtype,
                "complexity": f"{complexity:.2f}",
                "input_tokens": str(tokens),
            }),
        )

        output_record = SCLRecord(
            anchor=Anchor("router"),
            relation=Relation("select"),
            scope=Scope(entries={
                "tier": tier,
                "confidence": f"{confidence:.2f}",
                "reason": f"{cat}_{subtype}_complexity_{complexity:.1f}",
            }),
        )

        pair = TrainingPair(
            input_scl=emit_record(input_record),
            output_scl=emit_record(output_record),
            source="synthetic_route",
            quality=0.7,
        )
        pairs.append(pair)

    return pairs


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

class CKMDataset:
    """
    Aggregates training data from all sources into a unified dataset.

    The dataset is a JSONL file where each line is:
      {"input": "<SCL context>", "output": "<SCL response>", "source": "...", "quality": 0.X}

    This is the fine-tuning data for the 0.3B base model.
    """

    def __init__(self, output_dir: str = "/mnt/cortex/var/lib/ckm"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pairs: list[TrainingPair] = []

    def add_boot_telemetry(self, delta_stream: DeltaStream) -> int:
        """Add real boot telemetry data."""
        new_pairs = generate_boot_pairs(delta_stream)
        self.pairs.extend(new_pairs)
        return len(new_pairs)

    def add_routing_audit(self, records: list[SCLRecord]) -> int:
        """Add real routing audit data."""
        new_pairs = generate_routing_pairs(records)
        self.pairs.extend(new_pairs)
        return len(new_pairs)

    def add_policy_mutations(
        self, mutations: list[SCLRecord], feedback: list[SCLRecord]
    ) -> int:
        """Add real policy mutation data."""
        new_pairs = generate_policy_pairs(mutations, feedback)
        self.pairs.extend(new_pairs)
        return len(new_pairs)

    def add_synthetic(self, boot_count: int = 1000, route_count: int = 2000) -> int:
        """Bootstrap with synthetic data."""
        boot_pairs = generate_synthetic_boot_data(boot_count)
        route_pairs = generate_synthetic_routing_data(route_count)
        self.pairs.extend(boot_pairs)
        self.pairs.extend(route_pairs)
        return len(boot_pairs) + len(route_pairs)

    def save(self, filename: str = "ckm_training.jsonl") -> Path:
        """Save dataset as JSONL."""
        path = self.output_dir / filename
        with open(path, "w") as f:
            for pair in self.pairs:
                f.write(pair.to_jsonl() + "\n")
        return path

    def stats(self) -> dict:
        """Dataset statistics."""
        sources = {}
        for p in self.pairs:
            sources[p.source] = sources.get(p.source, 0) + 1
        avg_quality = sum(p.quality for p in self.pairs) / len(self.pairs) if self.pairs else 0
        return {
            "total_pairs": len(self.pairs),
            "sources": sources,
            "avg_quality": round(avg_quality, 3),
            "output_dir": str(self.output_dir),
        }
