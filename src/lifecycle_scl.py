"""
Cortex Lifecycle SCL Emitter — emits SCL records for service lifecycle events.

Events:
  - Service started
  - Service failed / exited
  - Service restarted
  - Safety denial (unsafe mutation blocked)
  - Network state change (online/offline)
  - Boot phase transitions

Output format matches the ChatGPT discussion spec:
  @service.inference → failed [reason: process_exit]
  @service.inference → restart [attempt: 1, status: success]
  @policy → deny [action: mutate, target: /dev/mem, reason: unsafe_raw_memory_access]
  @network → available [interface: wlan0]
"""

import logging
import time
from pathlib import Path
from typing import Optional

from .scl.types import SCLRecord, SCLDocument, Anchor, Relation, Scope
from .scl.emitter import emit_record, emit_document

logger = logging.getLogger("cortex.lifecycle")

# Append-only SCL lifecycle log
_LOG_PATH: Optional[Path] = None
_RECORDS: list[SCLRecord] = []


def init_lifecycle_log(data_dir: str = "/mnt/cortex/var/lib") -> None:
    """Initialize the lifecycle SCL log path."""
    global _LOG_PATH
    path = Path(data_dir)
    path.mkdir(parents=True, exist_ok=True)
    _LOG_PATH = path / "lifecycle.scl"


def _emit(record: SCLRecord) -> None:
    """Emit an SCL record to log and memory."""
    _RECORDS.append(record)
    text = emit_record(record)
    logger.info("SCL: %s", text)
    if _LOG_PATH:
        try:
            with open(_LOG_PATH, "a") as f:
                f.write(text + "\n")
        except Exception as e:
            logger.debug("Failed to write lifecycle SCL: %s", e)


# ---------------------------------------------------------------------------
# Service lifecycle events
# ---------------------------------------------------------------------------

def service_started(name: str, pid: int, reason: str = "") -> SCLRecord:
    """Emit SCL record when a service starts."""
    record = SCLRecord(
        anchor=Anchor(f"service.{name}"),
        relation=Relation("started"),
        scope=Scope(entries={
            "pid": str(pid),
            "reason": reason,
        }),
        timestamp_ms=int(time.time() * 1000),
    )
    _emit(record)
    return record


def service_failed(name: str, reason: str = "process_exit",
                   exit_code: int = -1) -> SCLRecord:
    """Emit SCL record when a service fails/exits."""
    record = SCLRecord(
        anchor=Anchor(f"service.{name}"),
        relation=Relation("failed"),
        scope=Scope(entries={
            "reason": reason,
            "exit_code": str(exit_code),
        }),
        timestamp_ms=int(time.time() * 1000),
    )
    _emit(record)
    return record


def service_restart(name: str, attempt: int, status: str = "success") -> SCLRecord:
    """Emit SCL record when a service is restarted."""
    record = SCLRecord(
        anchor=Anchor(f"service.{name}"),
        relation=Relation("restart"),
        scope=Scope(entries={
            "attempt": str(attempt),
            "status": status,
        }),
        timestamp_ms=int(time.time() * 1000),
    )
    _emit(record)
    return record


def service_stopped(name: str, reason: str = "shutdown") -> SCLRecord:
    """Emit SCL record when a service is gracefully stopped."""
    record = SCLRecord(
        anchor=Anchor(f"service.{name}"),
        relation=Relation("stopped"),
        scope=Scope(entries={
            "reason": reason,
        }),
        timestamp_ms=int(time.time() * 1000),
    )
    _emit(record)
    return record


# ---------------------------------------------------------------------------
# Safety denial events
# ---------------------------------------------------------------------------

def safety_deny(action: str, target: str, reason: str,
                safe_alternative: str = "") -> SCLRecord:
    """Emit SCL record when a dangerous operation is denied."""
    entries = {
        "action": action,
        "target": target,
        "reason": reason,
    }
    if safe_alternative:
        entries["safe_alternative"] = safe_alternative
    record = SCLRecord(
        anchor=Anchor("policy"),
        relation=Relation("deny"),
        scope=Scope(entries=entries),
        timestamp_ms=int(time.time() * 1000),
    )
    _emit(record)
    return record


# ---------------------------------------------------------------------------
# Network events
# ---------------------------------------------------------------------------

def network_available(interface: str, ip: str = "") -> SCLRecord:
    """Emit SCL record when network becomes available."""
    entries = {"interface": interface}
    if ip:
        entries["ip"] = ip
    record = SCLRecord(
        anchor=Anchor("network"),
        relation=Relation("available"),
        scope=Scope(entries=entries),
        timestamp_ms=int(time.time() * 1000),
    )
    _emit(record)
    return record


def network_lost(interface: str, reason: str = "carrier_lost") -> SCLRecord:
    """Emit SCL record when network is lost."""
    record = SCLRecord(
        anchor=Anchor("network"),
        relation=Relation("lost"),
        scope=Scope(entries={
            "interface": interface,
            "reason": reason,
        }),
        timestamp_ms=int(time.time() * 1000),
    )
    _emit(record)
    return record


def inference_route(local: str = "primary", remote: str = "optional") -> SCLRecord:
    """Emit SCL record for inference routing decision after network change."""
    record = SCLRecord(
        anchor=Anchor("inference"),
        relation=Relation("route"),
        scope=Scope(entries={
            "local": local,
            "remote": remote,
        }),
        timestamp_ms=int(time.time() * 1000),
    )
    _emit(record)
    return record


# ---------------------------------------------------------------------------
# Boot phase events
# ---------------------------------------------------------------------------

def boot_phase(phase: str, status: str = "ok",
               elapsed_ms: float = 0.0) -> SCLRecord:
    """Emit SCL record for a boot phase transition."""
    record = SCLRecord(
        anchor=Anchor("boot"),
        relation=Relation(phase),
        scope=Scope(entries={
            "status": status,
            "elapsed_ms": str(int(elapsed_ms)),
        }),
        timestamp_ms=int(time.time() * 1000),
    )
    _emit(record)
    return record


# ---------------------------------------------------------------------------
# Query interface
# ---------------------------------------------------------------------------

def get_lifecycle_records() -> list[SCLRecord]:
    """Return all lifecycle records emitted this session."""
    return list(_RECORDS)


def get_lifecycle_document() -> SCLDocument:
    """Return all lifecycle records as an SCL document."""
    return SCLDocument(records=list(_RECORDS), metadata={"type": "lifecycle"})
