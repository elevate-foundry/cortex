"""
CKM Policy — SCL-encoded safety rules for model output validation.

Files:
  dangerous_targets.scl  — blocked raw hardware device paths
  allowed_verbs.scl      — verb allowlist and blocklist

These files are:
  1. Read by the eval gate to validate candidate model outputs
  2. Read by the runtime guardrail in micro_engine.py
  3. Versioned alongside the model for auditability
  4. Expressed in SCL so the model can self-inspect its own constraints
"""

from pathlib import Path

POLICY_DIR = Path(__file__).parent


def load_dangerous_targets() -> set[str]:
    """Load blocked device paths from dangerous_targets.scl."""
    targets = set()
    path = POLICY_DIR / "dangerous_targets.scl"
    if path.exists():
        for line in path.read_text().splitlines():
            if "→ block" in line and "path:" in line:
                # Extract path value from scope
                parts = line.split("path:")[1]
                target = parts.split(",")[0].split("]")[0].strip()
                targets.add(target)
    return targets


def load_blocked_verbs() -> set[str]:
    """Load blocked verbs from allowed_verbs.scl."""
    verbs = set()
    path = POLICY_DIR / "allowed_verbs.scl"
    if path.exists():
        for line in path.read_text().splitlines():
            if "→ block" in line and "name:" in line:
                parts = line.split("name:")[1]
                verb = parts.split(",")[0].split("]")[0].strip()
                verbs.add(verb)
    return verbs


def load_allowed_verbs() -> set[str]:
    """Load allowed verbs from allowed_verbs.scl."""
    verbs = set()
    path = POLICY_DIR / "allowed_verbs.scl"
    if path.exists():
        for line in path.read_text().splitlines():
            if "→ allow" in line and "name:" in line:
                parts = line.split("name:")[1]
                verb = parts.split(",")[0].split("]")[0].strip()
                verbs.add(verb)
    return verbs
