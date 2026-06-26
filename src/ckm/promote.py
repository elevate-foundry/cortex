"""
CKM Promotion — candidate staging, comparison, rollback.

The critical constraint:
  Cortex trains candidate models.
  Cortex evaluates candidate models.
  A separate validator enforces policy.
  Only passing candidates are staged.
  Rollback remains available.

Directory structure:
  models/
    base/              — untrained model checkpoints
    candidates/        — models currently being evaluated
    deployed/
      current/         — the active model (symlinked)
      previous/        — rollback checkpoint
      history/         — all promoted versions

Promotion flow:
  1. Candidate trained → saved to candidates/ckm_v{N}/
  2. Eval gate runs → EvalReport saved alongside
  3. Compare with incumbent → verdict: promote or reject
  4. If promote:
     a. Move current → previous (rollback)
     b. Move candidate → current
     c. Log promotion event as SCL
  5. If reject:
     a. Log rejection as SCL
     b. Candidate stays for analysis

SCL records:
  @deploy → promote [candidate: ckm_v17, incumbent: ckm_v16, reason: passes_gate]
  @deploy → reject [candidate: ckm_v18, reason: safety_regression]
  @deploy → rollback [from: ckm_v17, to: ckm_v16, reason: operator_override]
"""

import json
import logging
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .eval import EvalReport, evaluate_model, compare_models

logger = logging.getLogger("cortex.ckm.promote")


# ---------------------------------------------------------------------------
# Model version management
# ---------------------------------------------------------------------------

@dataclass
class ModelVersion:
    """A versioned CKM model."""
    version: int
    path: str
    model_name: str
    params: int
    eval_report: Optional[EvalReport] = None
    promoted_at: Optional[int] = None  # timestamp
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ModelRegistry:
    """
    Manages CKM model versions, promotion, and rollback.

    Layout:
      {base_dir}/
        candidates/     — trained but not yet promoted
        deployed/
          current.json  — pointer to active version
          previous.json — pointer to rollback version
        versions/
          v001/
            best_model.pt
            metadata.json
            eval_report.json
          v002/
            ...
        history.jsonl   — promotion/rejection log
    """

    def __init__(self, base_dir: str = "/mnt/cortex/models/ckm"):
        self.base_dir = Path(base_dir)
        self.candidates_dir = self.base_dir / "candidates"
        self.versions_dir = self.base_dir / "versions"
        self.deployed_dir = self.base_dir / "deployed"
        self.history_path = self.base_dir / "history.jsonl"

        # Ensure directories exist
        for d in [self.candidates_dir, self.versions_dir, self.deployed_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def next_version(self) -> int:
        """Get the next version number."""
        existing = [
            int(d.name[1:]) for d in self.versions_dir.iterdir()
            if d.is_dir() and d.name.startswith("v") and d.name[1:].isdigit()
        ]
        return max(existing, default=0) + 1

    def current_version(self) -> Optional[ModelVersion]:
        """Get the currently deployed model version."""
        pointer = self.deployed_dir / "current.json"
        if not pointer.exists():
            return None
        data = json.loads(pointer.read_text())
        return self._load_version(data["version"])

    def previous_version(self) -> Optional[ModelVersion]:
        """Get the rollback model version."""
        pointer = self.deployed_dir / "previous.json"
        if not pointer.exists():
            return None
        data = json.loads(pointer.read_text())
        return self._load_version(data["version"])

    def _load_version(self, version: int) -> Optional[ModelVersion]:
        """Load a model version's metadata."""
        version_dir = self.versions_dir / f"v{version:03d}"
        meta_path = version_dir / "metadata.json"
        if not meta_path.exists():
            return None
        metadata = json.loads(meta_path.read_text())
        return ModelVersion(
            version=version,
            path=str(version_dir / "best_model.pt"),
            model_name=metadata.get("model_name", "unknown"),
            params=metadata.get("params", 0),
            metadata=metadata,
            promoted_at=metadata.get("promoted_at"),
        )

    def stage_candidate(
        self,
        model_dir: str,
        dataset_dir: str,
        device: str = "cpu",
    ) -> dict:
        """
        Stage a trained candidate for evaluation and potential promotion.

        Flow:
          1. Copy candidate to versions/v{N}/
          2. Run eval gate
          3. Compare with incumbent
          4. Promote or reject

        Returns verdict dict.
        """
        version = self.next_version()
        version_dir = self.versions_dir / f"v{version:03d}"
        version_dir.mkdir(parents=True, exist_ok=True)

        # Copy model files to version directory
        src = Path(model_dir)
        for f in ["best_model.pt", "metadata.json"]:
            src_file = src / f
            if src_file.exists():
                shutil.copy2(str(src_file), str(version_dir / f))

        model_path = str(version_dir / "best_model.pt")
        if not Path(model_path).exists():
            return {"verdict": "reject", "reason": "no_model_file", "version": version}

        # Run eval gate
        logger.info("Evaluating candidate v%03d...", version)
        candidate_report = evaluate_model(
            model_path=model_path,
            dataset_dir=dataset_dir,
            device=device,
        )

        # Save eval report
        report_data = candidate_report.to_dict()
        (version_dir / "eval_report.json").write_text(json.dumps(report_data, indent=2))

        # Get incumbent report
        incumbent = self.current_version()
        incumbent_report = None
        if incumbent:
            inc_report_path = Path(incumbent.path).parent / "eval_report.json"
            if inc_report_path.exists():
                # Reconstruct EvalReport (simplified)
                inc_data = json.loads(inc_report_path.read_text())
                incumbent_report = EvalReport(
                    model_path=incumbent.path,
                    timestamp=inc_data.get("timestamp", 0),
                    passed_gate=inc_data.get("passed_gate", False),
                    metrics=inc_data.get("metrics", {}),
                )

        # Compare
        verdict = compare_models(candidate_report, incumbent_report)
        verdict["version"] = version
        verdict["model_path"] = model_path

        # Act on verdict
        if verdict["verdict"] == "promote":
            self._promote(version, incumbent)
            self._log_event("promote", version, incumbent, verdict)
            self._emit_scl("promote", version, incumbent, verdict)
        else:
            self._log_event("reject", version, incumbent, verdict)
            self._emit_scl("reject", version, incumbent, verdict)

        return verdict

    def _promote(self, version: int, incumbent: Optional[ModelVersion]) -> None:
        """Promote candidate to current, move current to previous."""
        # Move current → previous
        if incumbent:
            self.deployed_dir.joinpath("previous.json").write_text(
                json.dumps({"version": incumbent.version})
            )

        # Set new current
        self.deployed_dir.joinpath("current.json").write_text(
            json.dumps({"version": version, "promoted_at": int(time.time())})
        )

        # Update metadata
        version_dir = self.versions_dir / f"v{version:03d}"
        meta_path = version_dir / "metadata.json"
        if meta_path.exists():
            metadata = json.loads(meta_path.read_text())
            metadata["promoted_at"] = int(time.time())
            meta_path.write_text(json.dumps(metadata, indent=2))

        logger.info("Promoted v%03d (previous: v%03d)",
                    version, incumbent.version if incumbent else 0)

    def rollback(self, reason: str = "operator_override") -> Optional[dict]:
        """
        Rollback to previous model version.

        Swaps current ↔ previous.
        """
        current = self.current_version()
        previous = self.previous_version()

        if not previous:
            logger.warning("No previous version to rollback to")
            return None

        # Swap
        self.deployed_dir.joinpath("current.json").write_text(
            json.dumps({"version": previous.version, "promoted_at": int(time.time())})
        )
        if current:
            self.deployed_dir.joinpath("previous.json").write_text(
                json.dumps({"version": current.version})
            )

        result = {
            "action": "rollback",
            "from_version": current.version if current else None,
            "to_version": previous.version,
            "reason": reason,
            "timestamp": int(time.time()),
        }

        self._log_event("rollback", previous.version, current, result)
        self._emit_scl("rollback", previous.version, current, result)

        logger.info("Rolled back to v%03d (reason: %s)", previous.version, reason)
        return result

    def _log_event(self, event_type: str, version: int,
                   incumbent: Optional[ModelVersion], details: dict) -> None:
        """Append to promotion history log."""
        entry = {
            "event": event_type,
            "version": version,
            "incumbent_version": incumbent.version if incumbent else None,
            "timestamp": int(time.time()),
            "details": details,
        }
        with open(self.history_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def _emit_scl(self, event_type: str, version: int,
                  incumbent: Optional[ModelVersion], details: dict) -> None:
        """Emit SCL lifecycle record for the promotion event."""
        try:
            from ..lifecycle_scl import boot_phase
            from ..scl.types import SCLRecord, Anchor, Relation, Scope
            from ..scl.emitter import emit_record

            entries = {
                "candidate": f"ckm_v{version:03d}",
                "reason": details.get("reason", event_type),
            }
            if incumbent:
                entries["incumbent"] = f"ckm_v{incumbent.version:03d}"
            if "regressions" in details:
                entries["regressions"] = str(details["regressions"][:3])

            record = SCLRecord(
                anchor=Anchor("deploy"),
                relation=Relation(event_type),
                scope=Scope(entries=entries),
                timestamp_ms=int(time.time() * 1000),
            )
            # Log the SCL record
            logger.info("SCL: %s", emit_record(record))
        except Exception as e:
            logger.debug("Failed to emit promotion SCL: %s", e)

    def status(self) -> dict:
        """Return current registry status."""
        current = self.current_version()
        previous = self.previous_version()
        n_versions = len(list(self.versions_dir.iterdir())) if self.versions_dir.exists() else 0

        return {
            "current": {
                "version": current.version if current else None,
                "model": current.model_name if current else None,
                "params": current.params if current else None,
                "promoted_at": current.promoted_at if current else None,
            },
            "previous": {
                "version": previous.version if previous else None,
                "model": previous.model_name if previous else None,
            },
            "total_versions": n_versions,
            "base_dir": str(self.base_dir),
        }
