"""
Boot Telemetry & Self-Modification Engine.

Built on the SCL stack we already have:
  - Boot state IS an SCL SemanticState (src/scl/delta.py)
  - Each boot emits a Delta (mutation to the boot state)
  - Hardware fingerprints use Braille codec (src/braille/fingerprint.py)
  - Self-modification uses the RuleEngine (src/scl/eval.py)
  - Multi-stick sync uses Gossip (src/scl/gossip.py)

The USB stick gets smarter every time you plug it in.

Architecture:
  Boot N:
    1. Load boot SemanticState from CORTEX partition
    2. Compute Braille fingerprint of current state
    3. Check if hardware fingerprint matches cached optimal config IN the state
    4. If match → apply cached config (fast path)
    5. If miss → full detection, emit Delta, RuleEngine evaluates
    6. After daemon ready → optimizer emits mutations as Deltas
    7. State persisted as SCL document

  Boot N+1:
    - Same hardware → Braille fingerprint match → skip detection
    - Multiple sticks on same LAN → gossip boot state improvements

Data lives on CORTEX partition as SCL:
  /mnt/cortex/var/lib/boot_state.scl     (SemanticState as SCL document)
  /mnt/cortex/var/lib/boot_deltas.scl    (DeltaStream as SCL records)
"""

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .scl.types import SCLRecord, SCLDocument, Anchor, Relation, Scope
from .scl.delta import Delta, SemanticState, DeltaStream, VectorClock, apply_delta, diff
from .scl.emitter import emit_document, emit_record
from .scl.parser import parse_document, parse_record
from .scl.eval import RuleEngine, Rule, Condition, Action, ActionType, CompareOp
from .braille.fingerprint import fingerprint as scl_fingerprint, similarity as fp_similarity
from .braille.codec import encode as braille_encode


# ---------------------------------------------------------------------------
# Boot Agent — the "self" that mutates across boots
# ---------------------------------------------------------------------------

BOOT_AGENT_ID = "cortex.boot"


# ---------------------------------------------------------------------------
# Hardware Fingerprinting (via Braille)
# ---------------------------------------------------------------------------

@dataclass
class HardwareFingerprint:
    """Unique identifier for a hardware configuration.

    Uses Braille encoding (same as gossip peer fingerprints).
    """
    cpu_model: str = ""
    cpu_cores: int = 0
    ram_mb: int = 0
    gpu_type: str = "none"       # nvidia, amd, intel, apple, none
    gpu_name: str = ""
    gpu_vram_mb: int = 0
    arch: str = ""               # x86_64, aarch64
    disk_serial: str = ""

    @property
    def fingerprint(self) -> str:
        """Braille fingerprint of hardware identity (4 chars = 32 bits).

        Uses the same Braille codec as SCL record fingerprints.
        """
        record = SCLRecord(
            anchor=Anchor("hardware"),
            relation=Relation("identity"),
            scope=Scope(entries={
                "cpu": self.cpu_model,
                "cores": str(self.cpu_cores),
                "ram_mb": str(self.ram_mb),
                "gpu": self.gpu_type,
                "gpu_name": self.gpu_name,
                "vram_mb": str(self.gpu_vram_mb),
                "arch": self.arch,
            }),
        )
        return scl_fingerprint(record, width=4)

    def to_scl(self) -> SCLRecord:
        """Hardware as SCL record."""
        return SCLRecord(
            anchor=Anchor("hardware"),
            relation=Relation("detect"),
            scope=Scope(entries={
                "cpu": self.cpu_model,
                "cores": str(self.cpu_cores),
                "ram_mb": str(self.ram_mb),
                "gpu_type": self.gpu_type,
                "gpu_name": self.gpu_name,
                "vram_mb": str(self.gpu_vram_mb),
                "arch": self.arch,
            }),
            timestamp_ms=int(time.time() * 1000),
        )

    def to_dict(self) -> dict:
        return {
            "cpu_model": self.cpu_model,
            "cpu_cores": self.cpu_cores,
            "ram_mb": self.ram_mb,
            "gpu_type": self.gpu_type,
            "gpu_name": self.gpu_name,
            "gpu_vram_mb": self.gpu_vram_mb,
            "arch": self.arch,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "HardwareFingerprint":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def detect_fingerprint() -> HardwareFingerprint:
    """Detect current hardware and produce a Braille fingerprint."""
    import platform

    fp = HardwareFingerprint()
    fp.arch = os.uname().machine
    fp.cpu_cores = os.cpu_count() or 0

    if platform.system() == "Darwin":
        import subprocess
        try:
            r = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                               capture_output=True, text=True, timeout=3)
            if r.returncode == 0:
                fp.cpu_model = r.stdout.strip()
        except Exception:
            pass
        try:
            r = subprocess.run(["sysctl", "-n", "hw.memsize"],
                               capture_output=True, text=True, timeout=3)
            if r.returncode == 0:
                fp.ram_mb = int(r.stdout.strip()) // (1024 * 1024)
        except Exception:
            pass
        if fp.arch in ("arm64", "aarch64"):
            fp.gpu_type = "apple"
    else:
        # Linux
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        fp.cpu_model = line.split(":")[1].strip()
                        break
        except Exception:
            pass
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if "MemTotal" in line:
                        fp.ram_mb = int(line.split()[1]) // 1024
                        break
        except Exception:
            pass
        if Path("/dev/nvidia0").exists():
            fp.gpu_type = "nvidia"
        elif Path("/dev/dri/renderD128").exists():
            fp.gpu_type = "amd"

    return fp


# ---------------------------------------------------------------------------
# Boot Record — emitted as SCL Delta
# ---------------------------------------------------------------------------

@dataclass
class BootRecord:
    """A single boot event, represented as an SCL Delta."""
    boot_id: str = ""
    timestamp_ms: int = 0
    hardware_fp: str = ""
    hardware: dict = field(default_factory=dict)
    boot_total_ms: float = 0.0
    time_to_first_token_ms: float = 0.0
    mount_ms: float = 0.0
    gpu_detect_ms: float = 0.0
    network_ms: float = 0.0
    backend_start_ms: float = 0.0
    backend_type: str = ""
    models_loaded: list = field(default_factory=list)
    gpu_layers: int = 0
    thread_count: int = 0
    context_size: int = 0
    batch_size: int = 0
    first_request_tier: str = ""
    first_request_latency_ms: float = 0.0
    errors: list = field(default_factory=list)
    used_cache: bool = False

    def to_delta(self) -> Delta:
        """Convert boot record to an SCL Delta (mutation of boot state)."""
        return Delta(
            agent_id=BOOT_AGENT_ID,
            set_keys={
                "last_boot_id": self.boot_id,
                "last_boot_ms": str(int(self.boot_total_ms)),
                "last_ttft_ms": str(int(self.time_to_first_token_ms)),
                "last_backend": self.backend_type,
                "last_gpu_layers": str(self.gpu_layers),
                "last_thread_count": str(self.thread_count),
                "last_context_size": str(self.context_size),
                "last_models": ",".join(self.models_loaded),
                "last_hw_fp": self.hardware_fp,
                "boot_count": "increment",  # handled specially by apply
                "last_errors": ",".join(self.errors) if self.errors else "",
                "used_cache": str(self.used_cache).lower(),
            },
            timestamp_ms=self.timestamp_ms or int(time.time() * 1000),
            weight=1.0,
        )

    def to_scl(self) -> SCLRecord:
        """Boot event as SCL record for audit trail."""
        return SCLRecord(
            anchor=Anchor(BOOT_AGENT_ID),
            relation=Relation("boot"),
            scope=Scope(entries={
                "id": self.boot_id,
                "ms": str(int(self.boot_total_ms)),
                "ttft": str(int(self.time_to_first_token_ms)),
                "backend": self.backend_type,
                "gpu_layers": str(self.gpu_layers),
                "threads": str(self.thread_count),
                "models": ",".join(self.models_loaded),
                "hw_fp": self.hardware_fp,
                "cached": str(self.used_cache).lower(),
            }),
            timestamp_ms=self.timestamp_ms,
        )


# ---------------------------------------------------------------------------
# Optimal Configuration (derived from SemanticState)
# ---------------------------------------------------------------------------

@dataclass
class OptimalConfig:
    """Cached optimal config — extracted from boot SemanticState."""
    hardware_fp: str = ""
    created_at_ms: int = 0
    updated_at_ms: int = 0
    boot_count: int = 0
    backend_type: str = "llama_cpp"
    thread_count: int = 4
    gpu_layers: int = 0
    context_size: int = 4096
    batch_size: int = 8
    flash_attn: bool = True
    mmap: bool = True
    hot_models: list = field(default_factory=list)
    cold_models: list = field(default_factory=list)
    skip_getty: bool = False
    skip_sshd: bool = False
    skip_network: bool = False
    skip_gpu_detect: bool = False
    avg_boot_ms: float = 0.0
    avg_ttft_ms: float = 0.0
    best_boot_ms: float = 0.0

    @classmethod
    def from_semantic_state(cls, state: SemanticState) -> "OptimalConfig":
        """Extract OptimalConfig from an SCL SemanticState."""
        e = state.entries
        return cls(
            hardware_fp=e.get("hardware_fp", ""),
            boot_count=int(e.get("boot_count", "0")),
            backend_type=e.get("optimal_backend", "llama_cpp"),
            thread_count=int(e.get("optimal_threads", "4")),
            gpu_layers=int(e.get("optimal_gpu_layers", "0")),
            context_size=int(e.get("optimal_ctx_size", "4096")),
            batch_size=int(e.get("optimal_batch_size", "8")),
            hot_models=e.get("optimal_hot_models", "L0").split(","),
            skip_gpu_detect=e.get("skip_gpu_detect", "false") == "true",
            skip_network=e.get("skip_network", "false") == "true",
            avg_boot_ms=float(e.get("avg_boot_ms", "0")),
            avg_ttft_ms=float(e.get("avg_ttft_ms", "0")),
            best_boot_ms=float(e.get("best_boot_ms", "0")),
            updated_at_ms=int(e.get("updated_at_ms", "0")),
        )

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


# ---------------------------------------------------------------------------
# Boot Telemetry (backed by SCL DeltaStream + RuleEngine)
# ---------------------------------------------------------------------------

class BootTelemetry:
    """
    Manages boot state as SCL SemanticState with Delta mutations.

    This is the self-modifying OS core:
      - Boot state persists as SCL document on CORTEX partition
      - Each boot emits a Delta (mutation)
      - RuleEngine evaluates boot events → optimization actions
      - Braille fingerprints detect hardware matches
      - DeltaStream enables time-travel to any previous boot config

    The same gossip protocol that syncs inference state between nodes
    can sync boot optimizations between multiple Cortex sticks.
    """

    def __init__(self, data_dir: str = "/mnt/cortex/var/lib"):
        self.data_dir = Path(data_dir)
        self.state_path = self.data_dir / "boot_state.scl"
        self.deltas_path = self.data_dir / "boot_deltas.scl"

        # Core SCL infrastructure
        self.stream = DeltaStream()
        self.rules = RuleEngine()
        self._setup_boot_rules()

        # Load existing state from disk
        self._load_state()

    def _setup_boot_rules(self) -> None:
        """Install SCL rules for boot self-modification.

        These rules evaluate boot events and emit optimization mutations.
        The RuleEngine is the SAME one used for inference routing — same
        infrastructure, different rules.
        """
        # Rule: if boot_total_ms > 5000, escalate thread count
        self.rules.add_rule(Rule(
            name="slow_boot_increase_threads",
            conditions=[Condition("ms", CompareOp.GT, "5000")],
            actions=[Action(ActionType.MUTATE,
                           params={"optimal_threads": "increase"})],
            priority=10,
        ))

        # Rule: if 5 consecutive boots with same GPU, skip detection
        self.rules.add_rule(Rule(
            name="stable_gpu_skip_detect",
            conditions=[Condition("boot_count", CompareOp.GTE, "5"),
                        Condition("gpu_stable", CompareOp.EQ, "true")],
            actions=[Action(ActionType.MUTATE,
                           params={"skip_gpu_detect": "true"})],
            priority=8,
        ))

        # Rule: if used_cache=true and boot was fast, reinforce config
        self.rules.add_rule(Rule(
            name="reinforce_fast_cached_boot",
            conditions=[Condition("cached", CompareOp.EQ, "true"),
                        Condition("ms", CompareOp.LT, "3000")],
            actions=[Action(ActionType.LOG,
                           params={"message": "boot_config_reinforced"})],
            priority=5,
        ))

    def _load_state(self) -> None:
        """Load boot state from SCL document on disk."""
        if self.state_path.exists():
            try:
                text = self.state_path.read_text()
                doc = parse_document(text)
                # Reconstruct SemanticState from snapshot record
                for record in doc.records:
                    if record.relation.verb == "snapshot":
                        # Full state restore
                        for k, v in record.scope.entries.items():
                            self.stream.append(Delta(
                                agent_id=BOOT_AGENT_ID,
                                set_keys={k: v},
                                seq=self.stream.length + 1,
                            ))
                    elif record.relation.verb == "mutate":
                        # Apply delta
                        delta = Delta.from_scl(record)
                        self.stream.append(delta)
            except Exception:
                pass  # Fresh state on parse failure

        # Also load delta history
        if self.deltas_path.exists():
            try:
                text = self.deltas_path.read_text()
                doc = parse_document(text)
                for record in doc.records:
                    if record.relation.verb == "mutate":
                        delta = Delta.from_scl(record)
                        self.stream.append(delta)
            except Exception:
                pass

    def _save_state(self) -> None:
        """Persist current state as SCL document."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        state = self.stream.current_state()
        doc = SCLDocument(records=[state.to_scl(BOOT_AGENT_ID)])
        self.state_path.write_text(emit_document(doc))

    def _save_delta(self, delta: Delta) -> None:
        """Append a delta to the on-disk delta log."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        with open(self.deltas_path, "a") as f:
            f.write(emit_record(delta.to_scl()) + "\n")

    @property
    def state(self) -> SemanticState:
        """Current boot state."""
        return self.stream.current_state()

    @property
    def state_fingerprint(self) -> str:
        """Braille fingerprint of current boot state."""
        record = self.state.to_scl(BOOT_AGENT_ID)
        return scl_fingerprint(record, width=4)

    def log_boot(self, record: BootRecord) -> None:
        """Record a boot event as a Delta in the stream."""
        delta = record.to_delta()
        delta.seq = self.stream.length + 1

        # Handle boot_count increment
        current_count = int(self.state.entries.get("boot_count", "0"))
        delta.set_keys["boot_count"] = str(current_count + 1)

        # Apply to stream
        self.stream.append(delta)
        self._save_delta(delta)
        self._save_state()

        # Evaluate boot rules (self-modification trigger)
        boot_scl = record.to_scl()
        self.rules.evaluate(boot_scl)

    def get_cached_config(self, hardware_fp: str) -> Optional[OptimalConfig]:
        """Get optimal config from SemanticState if hardware matches.

        Uses Braille fingerprint similarity (same as gossip convergence check).
        """
        state = self.state
        cached_fp = state.entries.get("hardware_fp", "")
        if not cached_fp:
            return None

        # Compare using Braille similarity (same function as gossip peer sync)
        if len(hardware_fp) == len(cached_fp):
            sim = fp_similarity(hardware_fp, cached_fp)
            if sim >= 0.999:  # exact match
                return OptimalConfig.from_semantic_state(state)

        # Fallback: string equality (for non-Braille fingerprints)
        if cached_fp == hardware_fp:
            return OptimalConfig.from_semantic_state(state)

        return None

    def optimize(self, hardware_fp: str) -> Optional[OptimalConfig]:
        """
        Emit optimization Delta based on boot history.

        This IS the self-modifying OS:
          1. Analyze DeltaStream (all past boot decisions)
          2. Compute optimal config
          3. Emit Delta (mutation to boot state)
          4. Next boot reads mutated state → faster

        The Delta uses the same infrastructure as gossip deltas,
        so if two sticks gossip, boot optimizations propagate.
        """
        state = self.state
        boot_count = int(state.entries.get("boot_count", "0"))
        if boot_count < 1:
            return None

        # Analyze recent deltas for optimization
        optimal_mutations: dict[str, str] = {}

        # Use best observed values
        last_boot_ms = float(state.entries.get("last_boot_ms", "0"))
        last_threads = state.entries.get("last_thread_count", "4")
        last_gpu = state.entries.get("last_gpu_layers", "0")
        last_ctx = state.entries.get("last_context_size", "4096")
        last_backend = state.entries.get("last_backend", "llama_cpp")
        last_models = state.entries.get("last_models", "L0")

        optimal_mutations["hardware_fp"] = hardware_fp
        optimal_mutations["optimal_backend"] = last_backend
        optimal_mutations["optimal_threads"] = last_threads
        optimal_mutations["optimal_gpu_layers"] = last_gpu
        optimal_mutations["optimal_ctx_size"] = last_ctx
        optimal_mutations["optimal_hot_models"] = last_models
        optimal_mutations["updated_at_ms"] = str(int(time.time() * 1000))

        # Track performance
        best_ms = state.entries.get("best_boot_ms", str(last_boot_ms))
        if last_boot_ms > 0 and last_boot_ms < float(best_ms):
            optimal_mutations["best_boot_ms"] = str(int(last_boot_ms))
        else:
            optimal_mutations["best_boot_ms"] = best_ms

        # Compute running average
        avg = float(state.entries.get("avg_boot_ms", "0"))
        if avg > 0:
            optimal_mutations["avg_boot_ms"] = str(int((avg + last_boot_ms) / 2))
        else:
            optimal_mutations["avg_boot_ms"] = str(int(last_boot_ms))

        # GPU stability check (for skip_gpu_detect rule)
        last_gpu_type = state.entries.get("last_gpu_type", "")
        if last_gpu_type and last_gpu_type == state.entries.get("prev_gpu_type", last_gpu_type):
            optimal_mutations["gpu_stable"] = "true"
        optimal_mutations["prev_gpu_type"] = state.entries.get("last_gpu_type", "none")

        # Emit optimization delta
        opt_delta = Delta(
            agent_id=BOOT_AGENT_ID,
            set_keys=optimal_mutations,
            seq=self.stream.length + 1,
            weight=2.0,  # Higher weight → wins in gossip merge conflicts
            timestamp_ms=int(time.time() * 1000),
        )
        self.stream.append(opt_delta)
        self._save_delta(opt_delta)
        self._save_state()

        # Run rules against updated state
        state_record = self.state.to_scl(BOOT_AGENT_ID)
        results = self.rules.evaluate(state_record)

        # Apply any MUTATE actions from rules
        for result in results:
            if result.action.action_type == ActionType.MUTATE and result.mutations:
                rule_delta = Delta(
                    agent_id=BOOT_AGENT_ID,
                    set_keys=result.mutations,
                    seq=self.stream.length + 1,
                )
                self.stream.append(rule_delta)
                self._save_delta(rule_delta)

        self._save_state()
        return OptimalConfig.from_semantic_state(self.state)

    def get_improvement_report(self, hardware_fp: str) -> dict:
        """Report improvement using SCL state (not raw JSON logs)."""
        state = self.state
        boot_count = int(state.entries.get("boot_count", "0"))
        if boot_count < 2:
            return {"status": "insufficient_data", "boots": boot_count}

        avg_ms = float(state.entries.get("avg_boot_ms", "0"))
        best_ms = float(state.entries.get("best_boot_ms", "0"))
        last_ms = float(state.entries.get("last_boot_ms", "0"))

        return {
            "status": "ok",
            "boots": boot_count,
            "avg_boot_ms": avg_ms,
            "best_boot_ms": best_ms,
            "last_boot_ms": last_ms,
            "state_fingerprint": self.state_fingerprint,
            "stream_length": self.stream.length,
            "rules_fired": self.rules._fire_count,
            "improvement_pct": round((1 - best_ms / avg_ms) * 100, 1) if avg_ms > 0 else 0,
        }

    def to_scl_document(self) -> SCLDocument:
        """Export full boot telemetry as SCL document (for gossip/audit)."""
        records = []
        # Current state snapshot
        records.append(self.state.to_scl(BOOT_AGENT_ID))
        # All deltas
        for delta in self.stream.deltas:
            records.append(delta.to_scl())
        return SCLDocument(records=records, metadata={"type": "boot_telemetry"})
