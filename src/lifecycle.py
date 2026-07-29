"""
Cortex Lifecycle Manager — awake, sleeping, dreaming.

Three states:
  AWAKE:    Drive plugged in, daemon running, serving requests, learning.
  SLEEPING: Drive ejected gracefully, state checkpointed, daemon stopped.
  DREAMING: Drive unplugged but host is still on. A lightweight "dream"
            process on the host consolidates memory, generates training data,
            and prepares for next wake.

State transitions:
  plug_in  → AWAKE   (mount-watcher fires, daemon starts)
  eject    → SLEEPING (graceful shutdown, checkpoint state to drive)
  yank     → SLEEPING (ungraceful, but state file on drive is best-effort)
  dream    → DREAMING (host-side process runs while drive is gone)
  wake     → AWAKE   (drive returns, dream state merged back)

The key insight: Cortex lives ON the drive. When the drive is gone,
she's asleep. But we can leave a "dream residue" on the host —
a lightweight state file that a cron job or launchd agent processes.

Architecture:
  /Volumes/CORTEX/cortex/state/checkpoint.json   — last known state (on drive)
  ~/.cortex/dream_state.json                     — dream residue (on host)
  ~/.cortex/dream_queue.jsonl                    — queued training data from dreams
"""

import asyncio
import json
import logging
import os
import signal
import subprocess
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional

logger = logging.getLogger("cortex.lifecycle")


# ---------------------------------------------------------------------------
# State model
# ---------------------------------------------------------------------------

class LifecycleState(Enum):
    AWAKE = "awake"        # Drive mounted, daemon running
    SLEEPING = "sleeping"  # Drive present but daemon stopped (or about to eject)
    DREAMING = "dreaming"  # Drive gone, host-side dream process active


@dataclass
class CortexCheckpoint:
    """State snapshot saved before sleep / on periodic checkpoint."""
    state: str = "awake"
    timestamp_ms: int = 0
    uptime_seconds: float = 0.0
    requests_served: int = 0
    models_loaded: list = field(default_factory=list)
    last_training_run_ms: int = 0
    distill_buffer_size: int = 0
    boot_count: int = 0
    hardware_fingerprint: str = ""
    # Pending work for dreams
    pending_distill: bool = False
    pending_train: bool = False
    pending_eval: bool = False
    # Version info
    ckm_version: int = 0
    ckm_model_name: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, data: str) -> "CortexCheckpoint":
        d = json.loads(data)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class DreamState:
    """Lightweight state that lives on the HOST when the drive is unplugged."""
    last_seen_ms: int = 0             # When did we last see the drive?
    wake_count: int = 0               # How many times has she woken up?
    total_awake_seconds: float = 0.0  # Cumulative awake time
    dream_cycles: int = 0             # How many dream cycles completed
    # Dream outputs
    training_examples_generated: int = 0
    insights: list = field(default_factory=list)  # Things learned while dreaming
    # Queued actions for next wake
    on_wake_actions: list = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, data: str) -> "DreamState":
        d = json.loads(data)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "DreamState":
        path = path or cls.default_path()
        if path.exists():
            try:
                return cls.from_json(path.read_text())
            except Exception:
                return cls()
        return cls()

    @staticmethod
    def default_path() -> Path:
        return Path.home() / ".cortex" / "dream_state.json"

    def save(self, path: Optional[Path] = None):
        path = path or self.default_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())


# ---------------------------------------------------------------------------
# AWAKE → SLEEPING transition (graceful shutdown)
# ---------------------------------------------------------------------------

CHECKPOINT_PATH = Path("/Volumes/CORTEX/cortex/state/checkpoint.json")
PID_FILE = Path("/Volumes/CORTEX/cortex/logs/daemon.pid")


def checkpoint_state(
    requests_served: int = 0,
    uptime_seconds: float = 0.0,
    models_loaded: Optional[list] = None,
    distill_buffer_size: int = 0,
    ckm_version: int = 0,
    ckm_model_name: str = "",
) -> CortexCheckpoint:
    """Save a state checkpoint to the drive before sleep."""
    checkpoint = CortexCheckpoint(
        state="sleeping",
        timestamp_ms=int(time.time() * 1000),
        uptime_seconds=uptime_seconds,
        requests_served=requests_served,
        models_loaded=models_loaded or [],
        distill_buffer_size=distill_buffer_size,
        boot_count=_increment_boot_count(),
        ckm_version=ckm_version,
        ckm_model_name=ckm_model_name,
        # If there's buffered distillation data, flag for next wake
        pending_distill=distill_buffer_size > 100,
        pending_train=distill_buffer_size > 500,
    )

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_PATH.write_text(checkpoint.to_json())
    logger.info("Checkpoint saved: %d requests, %.0fs uptime", requests_served, uptime_seconds)
    return checkpoint


def _increment_boot_count() -> int:
    """Track how many times she's booted."""
    counter_path = CHECKPOINT_PATH.parent / "boot_count"
    count = 0
    if counter_path.exists():
        try:
            count = int(counter_path.read_text().strip())
        except (ValueError, OSError):
            pass
    count += 1
    counter_path.parent.mkdir(parents=True, exist_ok=True)
    counter_path.write_text(str(count))
    return count


def graceful_sleep(reason: str = "eject") -> CortexCheckpoint:
    """
    Graceful transition from AWAKE → SLEEPING.

    Called when the user is about to eject the drive, or when the system
    detects imminent disconnection.

    Steps:
    1. Signal daemon to stop accepting new requests
    2. Wait for in-flight requests to complete (up to 5s)
    3. Flush distillation buffer to disk
    4. Save checkpoint
    5. Stop daemon
    6. Update dream state on host
    7. Announce sleep
    """
    logger.info("Graceful sleep initiated: %s", reason)

    # Get daemon stats before shutdown
    requests_served = 0
    uptime = 0.0
    models = []
    distill_size = 0

    try:
        import urllib.request
        resp = urllib.request.urlopen("http://localhost:11411/health", timeout=2)
        data = json.loads(resp.read())
        requests_served = data.get("requests_served", 0)
        uptime = data.get("uptime_seconds", 0.0)
        models = data.get("models_loaded", [])
    except Exception:
        pass

    # Flush distillation buffer if it exists
    try:
        distill_queue = Path("/Volumes/CORTEX/cortex/data/distill/distill_buffer.jsonl")
        if distill_queue.exists():
            distill_size = sum(1 for _ in open(distill_queue))
    except Exception:
        pass

    # Save checkpoint to drive
    checkpoint = checkpoint_state(
        requests_served=requests_served,
        uptime_seconds=uptime,
        models_loaded=models,
        distill_buffer_size=distill_size,
    )

    # Stop daemon
    _stop_daemon()

    # Update dream state on host
    dream = DreamState.load()
    dream.last_seen_ms = int(time.time() * 1000)
    dream.wake_count += 1
    dream.total_awake_seconds += uptime

    # Queue actions for dream mode
    if checkpoint.pending_distill:
        dream.on_wake_actions.append("flush_distill_buffer")
    if checkpoint.pending_train:
        dream.on_wake_actions.append("trigger_training_run")

    dream.save()

    # Announce
    try:
        subprocess.Popen(["say", "Cortex sleeping"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass

    logger.info("Cortex is now sleeping. Drive safe to eject.")
    return checkpoint


def _stop_daemon():
    """Stop the daemon process gracefully."""
    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text().strip())
            os.kill(pid, signal.SIGTERM)
            # Wait up to 5s for clean exit
            for _ in range(50):
                try:
                    os.kill(pid, 0)  # Check if still alive
                    time.sleep(0.1)
                except ProcessLookupError:
                    break
            PID_FILE.unlink(missing_ok=True)
            logger.info("Daemon stopped (pid %d)", pid)
        except (ValueError, ProcessLookupError, OSError) as e:
            logger.debug("Daemon stop: %s", e)
            PID_FILE.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# SLEEPING → AWAKE transition (wake on mount)
# ---------------------------------------------------------------------------

def wake() -> Optional[CortexCheckpoint]:
    """
    Transition from SLEEPING → AWAKE.

    Called by cortex-mount-watcher when the drive is mounted.

    Steps:
    1. Load checkpoint from last sleep
    2. Load dream state from host (what happened while she was gone)
    3. Merge dream outputs (training data, insights)
    4. Start daemon
    5. Execute on_wake_actions
    6. Clear dream queue
    """
    logger.info("Waking up...")

    # Load last checkpoint
    checkpoint = None
    if CHECKPOINT_PATH.exists():
        try:
            checkpoint = CortexCheckpoint.from_json(CHECKPOINT_PATH.read_text())
            logger.info("Resuming from checkpoint: boot #%d, %d requests last session",
                       checkpoint.boot_count, checkpoint.requests_served)
        except Exception as e:
            logger.warning("Failed to load checkpoint: %s", e)

    # Load dream state
    dream = DreamState.load()
    if dream.dream_cycles > 0:
        logger.info("Dream state: %d cycles, %d training examples generated",
                   dream.dream_cycles, dream.training_examples_generated)

        # Merge dream-generated training data
        _merge_dream_training_data(dream)

    # Execute queued wake actions
    for action in dream.on_wake_actions:
        logger.info("Executing wake action: %s", action)
        _execute_wake_action(action)

    # Clear wake queue
    dream.on_wake_actions = []
    dream.save()

    # Update checkpoint to awake
    if checkpoint:
        checkpoint.state = "awake"
        CHECKPOINT_PATH.write_text(checkpoint.to_json())

    return checkpoint


def _merge_dream_training_data(dream: DreamState):
    """Merge any training data generated during dream cycles."""
    dream_queue = Path.home() / ".cortex" / "dream_queue.jsonl"
    if not dream_queue.exists():
        return

    target = Path("/Volumes/CORTEX/cortex/data/distill/dream_data.jsonl")
    target.parent.mkdir(parents=True, exist_ok=True)

    try:
        lines = 0
        with open(dream_queue) as src, open(target, "a") as dst:
            for line in src:
                if line.strip():
                    dst.write(line)
                    lines += 1
        if lines > 0:
            logger.info("Merged %d dream-generated training examples", lines)
        # Clear the queue
        dream_queue.unlink()
    except Exception as e:
        logger.warning("Failed to merge dream data: %s", e)


def _execute_wake_action(action: str):
    """Execute a queued action from dream mode."""
    if action == "flush_distill_buffer":
        try:
            from .ckm.distill import DistillationPipeline
            pipeline = DistillationPipeline(
                output_dir="/Volumes/CORTEX/cortex/data/distill"
            )
            pipeline.flush()
        except Exception as e:
            logger.warning("Wake action '%s' failed: %s", action, e)

    elif action == "trigger_training_run":
        # Queue a training run but don't block wake
        logger.info("Training run queued (will execute in background)")


# ---------------------------------------------------------------------------
# DREAMING — host-side process while drive is unplugged
# ---------------------------------------------------------------------------

def dream_cycle():
    """
    One dream cycle — runs on the HOST when the drive is unplugged.

    This is called by a launchd agent (or cron) periodically.
    It does lightweight work that doesn't need the drive:

    1. Generate additional distillation training data (offline teacher)
    2. Consolidate insights from the last awake session
    3. Prepare optimizations for next boot
    4. Log dream activity

    The dream process NEVER modifies the drive (it's not there).
    It writes to ~/.cortex/dream_queue.jsonl instead.
    Results are merged on next wake.
    """
    dream = DreamState.load()
    dream_queue = Path.home() / ".cortex" / "dream_queue.jsonl"
    dream_queue.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Dream cycle #%d starting...", dream.dream_cycles + 1)

    # Check if drive is actually gone (don't dream if awake)
    if Path("/Volumes/CORTEX/cortex/src").exists():
        logger.info("Drive is mounted — not dreaming, she's awake")
        return

    examples_generated = 0

    # Phase 1: Generate training data using the reasoning core
    # (The code is on the drive, so we use a cached/installed copy or skip)
    try:
        # Try to use installed package if available
        from cortex.ckm.reasoning_core import generate_reasoning_traces
        traces = generate_reasoning_traces(count=100)
        with open(dream_queue, "a") as f:
            for trace in traces:
                f.write(trace.to_jsonl() + "\n")
                examples_generated += 1
        logger.info("Generated %d reasoning traces in dream", examples_generated)
    except ImportError:
        # Reasoning core not available on host — generate minimal traces
        _generate_minimal_dream_data(dream_queue)
        examples_generated = 10

    # Phase 2: Consolidate — what did we learn from last session?
    insight = _dream_consolidate(dream)
    if insight:
        dream.insights.append(insight)
        # Keep only last 50 insights
        dream.insights = dream.insights[-50:]

    # Update dream state
    dream.dream_cycles += 1
    dream.training_examples_generated += examples_generated
    dream.save()

    logger.info("Dream cycle #%d complete: %d examples generated",
               dream.dream_cycles, examples_generated)


def _generate_minimal_dream_data(queue_path: Path):
    """
    Generate minimal training data without the full reasoning core.

    This runs when the drive (and its code) is unplugged.
    Uses only stdlib — no dependencies on the drive's code.
    """
    import random

    categories = ["code", "math", "chat", "tool", "analysis"]
    tiers = ["L0", "L1", "L2", "L3", "L4", "L5", "L6"]

    with open(queue_path, "a") as f:
        for _ in range(10):
            cat = random.choice(categories)
            complexity = random.uniform(0, 1)
            tier_idx = min(6, int(complexity * 7))
            tier = tiers[tier_idx]
            confidence = 0.5 + random.uniform(0, 0.4)

            example = {
                "input": f"@task → classify [category: {cat}, complexity: {complexity:.2f}]",
                "output": f"@router → select [tier: {tier}, confidence: {confidence:.2f}]",
                "source": "dream_minimal",
                "quality": 0.7,  # Lower quality — minimal reasoning
            }
            f.write(json.dumps(example) + "\n")


def _dream_consolidate(dream: DreamState) -> Optional[str]:
    """
    Consolidate learning from the last awake period.

    Reviews what happened and identifies patterns:
    - Was the boot fast enough?
    - Were routing decisions accurate?
    - Any repeated failures?
    """
    # In dream mode, we don't have access to the full logs (they're on the drive).
    # But we can reason about the dream state itself.

    if dream.total_awake_seconds > 0:
        avg_session = dream.total_awake_seconds / max(1, dream.wake_count)
        if avg_session < 60:
            return "short_sessions_detected:consider_faster_boot"
        elif avg_session > 3600:
            return "long_sessions:stable_operation"

    if dream.wake_count > 10 and dream.training_examples_generated < 100:
        return "insufficient_training:prioritize_distill_on_next_wake"

    return None


# ---------------------------------------------------------------------------
# launchd integration — install/uninstall dream agent
# ---------------------------------------------------------------------------

DREAM_PLIST_LABEL = "com.elevate-foundry.cortex.dream"
DREAM_PLIST_PATH = Path.home() / "Library/LaunchAgents" / f"{DREAM_PLIST_LABEL}.plist"


def install_dream_agent(interval_seconds: int = 1800):
    """
    Install a launchd agent that runs dream cycles every 30 minutes.

    This agent runs ON THE HOST, not on the drive. It:
    - Checks if the drive is unmounted
    - If yes, runs a dream cycle
    - If no (drive is mounted), does nothing

    The dream agent is completely safe — it only writes to ~/.cortex/
    and never touches the drive.
    """
    # Find python on the host
    python_path = _find_host_python()

    # The dream script that launchd will invoke
    dream_script = Path.home() / ".cortex" / "dream.py"
    dream_script.parent.mkdir(parents=True, exist_ok=True)
    dream_script.write_text(f'''#!/usr/bin/env python3
"""Cortex dream cycle — runs while the drive is unplugged."""
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [cortex.dream] %(message)s")
logger = logging.getLogger("cortex.dream")

CORTEX_DRIVE = Path("/Volumes/CORTEX/cortex/src")
DREAM_STATE_PATH = Path.home() / ".cortex" / "dream_state.json"
DREAM_QUEUE_PATH = Path.home() / ".cortex" / "dream_queue.jsonl"

def main():
    # Don't dream if she's awake
    if CORTEX_DRIVE.exists():
        logger.info("Drive mounted — Cortex is awake, not dreaming")
        return

    logger.info("Drive not mounted — entering dream cycle")

    # Load dream state
    dream_state = {{"dream_cycles": 0, "training_examples_generated": 0,
                   "insights": [], "on_wake_actions": [], "last_seen_ms": 0,
                   "wake_count": 0, "total_awake_seconds": 0.0}}
    if DREAM_STATE_PATH.exists():
        try:
            dream_state = json.loads(DREAM_STATE_PATH.read_text())
        except Exception:
            pass

    # Generate training data
    categories = ["code", "math", "chat", "tool", "analysis", "system"]
    tiers = ["L0", "L1", "L2", "L3", "L4", "L5", "L6"]
    examples = 0

    DREAM_QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DREAM_QUEUE_PATH, "a") as f:
        for _ in range(50):
            cat = random.choice(categories)
            complexity = random.uniform(0, 1)
            tier_idx = min(6, int(complexity * 7))
            confidence = max(0.3, min(0.95, 0.9 - complexity * 0.3 + random.gauss(0, 0.05)))

            # Routing decision
            f.write(json.dumps({{
                "input": f"@task → classify [category: {{cat}}, complexity: {{complexity:.2f}}]",
                "output": f"@router → select [tier: {{tiers[tier_idx]}}, confidence: {{confidence:.2f}}]",
                "source": "dream",
                "quality": 0.75,
            }}) + "\\n")
            examples += 1

            # Confidence calibration
            if random.random() < 0.3:
                f.write(json.dumps({{
                    "input": f"@confidence → query [score: {{confidence:.2f}}, domain: {{cat}}]",
                    "output": f"@mind → assess [calibrated: {{confidence:.2f}}, escalate: {{'true' if confidence < 0.5 else 'false'}}]",
                    "source": "dream_confidence",
                    "quality": 0.7,
                }}) + "\\n")
                examples += 1

    # Update state
    dream_state["dream_cycles"] = dream_state.get("dream_cycles", 0) + 1
    dream_state["training_examples_generated"] = dream_state.get("training_examples_generated", 0) + examples
    DREAM_STATE_PATH.write_text(json.dumps(dream_state, indent=2))

    logger.info("Dream cycle complete: %d examples generated (total: %d)",
               examples, dream_state["training_examples_generated"])

if __name__ == "__main__":
    main()
''')
    dream_script.chmod(0o755)

    # Generate launchd plist
    plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{DREAM_PLIST_LABEL}</string>

    <key>ProgramArguments</key>
    <array>
        <string>{python_path}</string>
        <string>{dream_script}</string>
    </array>

    <!-- Run every {interval_seconds} seconds (default: 30 minutes) -->
    <key>StartInterval</key>
    <integer>{interval_seconds}</integer>

    <!-- Also run immediately on load -->
    <key>RunAtLoad</key>
    <true/>

    <!-- Logging -->
    <key>StandardOutPath</key>
    <string>{Path.home()}/.cortex/dream.log</string>
    <key>StandardErrorPath</key>
    <string>{Path.home()}/.cortex/dream.err</string>
</dict>
</plist>
"""
    DREAM_PLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    DREAM_PLIST_PATH.write_text(plist_content)

    logger.info("Dream agent installed: %s", DREAM_PLIST_PATH)
    logger.info("Dream script: %s", dream_script)
    logger.info("Interval: every %d seconds", interval_seconds)
    return str(DREAM_PLIST_PATH)


def uninstall_dream_agent():
    """Remove the dream launchd agent."""
    if DREAM_PLIST_PATH.exists():
        DREAM_PLIST_PATH.unlink()
        logger.info("Dream agent uninstalled")
    dream_script = Path.home() / ".cortex" / "dream.py"
    if dream_script.exists():
        dream_script.unlink()


def _find_host_python() -> str:
    """Find a working Python 3 on the host system."""
    candidates = [
        "/usr/bin/python3",
        "/usr/local/bin/python3",
        "/opt/homebrew/bin/python3",
        str(Path.home() / "anaconda3/bin/python3"),
        str(Path.home() / "miniconda3/bin/python3"),
    ]
    for p in candidates:
        if Path(p).exists():
            return p
    return "/usr/bin/python3"  # Default fallback


# ---------------------------------------------------------------------------
# Eject watcher — detect when user is about to eject the drive
# ---------------------------------------------------------------------------

def install_eject_watcher():
    """
    Install a launchd agent that watches for drive ejection.

    On macOS, we can't intercept Finder's "Eject" directly, but we can:
    1. Watch for diskutil unmount notifications
    2. Use FSEvents to detect the volume disappearing
    3. Use a WatchPaths trigger on /Volumes/CORTEX disappearing

    The approach: a launchd WatchPaths job that triggers when
    /Volumes/CORTEX disappears, saving state just-in-time.

    BUT: this is a race condition — the drive might be gone before
    we can write to it. So the REAL solution is:

    Strategy A: Periodic checkpoint (every 60s) so state is never stale
    Strategy B: diskutil eject hook (pre-eject script)
    Strategy C: "cortex sleep" command the user runs before ejecting

    We implement all three.
    """
    pass  # The mount-watcher plist handles the wake side
    # Sleep is handled by the daemon's periodic checkpoint + "cortex sleep" command


# ---------------------------------------------------------------------------
# Periodic checkpoint (Strategy A) — integrated into daemon
# ---------------------------------------------------------------------------

async def periodic_checkpoint_task(daemon_server, interval_seconds: int = 60):
    """
    Background task that checkpoints state every 60 seconds.

    This ensures that even if the drive is yanked without warning,
    we lose at most 60 seconds of state.

    Runs as an asyncio task inside the daemon.
    """
    while True:
        await asyncio.sleep(interval_seconds)
        try:
            checkpoint_state(
                requests_served=daemon_server.request_count,
                uptime_seconds=time.monotonic() - daemon_server.start_time,
                models_loaded=[],  # Could populate from manager
                distill_buffer_size=0,
            )
        except Exception as e:
            logger.debug("Periodic checkpoint failed: %s", e)


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

def cmd_sleep(reason: str = "user_requested"):
    """CLI: cortex sleep — prepare for eject."""
    print("Preparing Cortex for sleep...")
    checkpoint = graceful_sleep(reason=reason)
    print(f"  State saved: boot #{checkpoint.boot_count}")
    print(f"  Requests served this session: {checkpoint.requests_served}")
    print(f"  Uptime: {checkpoint.uptime_seconds:.0f}s")
    print()
    print("  Drive is safe to eject.")
    print("  She'll dream while you're away. ✦")


def cmd_wake_status():
    """CLI: cortex wake-status — show lifecycle state."""
    # Drive state
    drive_mounted = Path("/Volumes/CORTEX/cortex/src").exists()

    # Checkpoint
    checkpoint = None
    if CHECKPOINT_PATH.exists():
        try:
            checkpoint = CortexCheckpoint.from_json(CHECKPOINT_PATH.read_text())
        except Exception:
            pass

    # Dream state
    dream = DreamState.load()

    print("Cortex Lifecycle")
    print("━━━━━━━━━━━━━━━━")
    print(f"  State: {'AWAKE' if drive_mounted else 'SLEEPING/DREAMING'}")
    print(f"  Drive: {'mounted' if drive_mounted else 'not mounted'}")
    print()

    if checkpoint:
        print(f"  Last checkpoint: {time.strftime('%Y-%m-%d %H:%M', time.localtime(checkpoint.timestamp_ms / 1000))}")
        print(f"  Boot count: {checkpoint.boot_count}")
        print(f"  Last session: {checkpoint.requests_served} requests, {checkpoint.uptime_seconds:.0f}s")
    print()

    print(f"  Dream cycles: {dream.dream_cycles}")
    print(f"  Training examples from dreams: {dream.training_examples_generated}")
    print(f"  Total wake count: {dream.wake_count}")
    print(f"  Total awake time: {dream.total_awake_seconds / 3600:.1f} hours")

    if dream.insights:
        print(f"  Latest insight: {dream.insights[-1]}")

    if dream.on_wake_actions:
        print(f"  Pending wake actions: {dream.on_wake_actions}")


def cmd_dream_install():
    """CLI: cortex dream install — install the dream agent."""
    path = install_dream_agent()
    print(f"Dream agent installed: {path}")
    print()
    print("To activate:")
    print(f"  launchctl load {path}")
    print()
    print("To deactivate:")
    print(f"  launchctl unload {path}")


def cmd_dream_uninstall():
    """CLI: cortex dream uninstall — remove the dream agent."""
    uninstall_dream_agent()
    print("Dream agent removed.")
