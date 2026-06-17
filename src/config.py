"""
Cortex config — centralized constants and environment overrides.

Single source of truth for:
  - Ollama URL
  - Daemon host/port
  - VRAM budget calculation
  - Data directory
"""

import os
from pathlib import Path


# ---------------------------------------------------------------------------
# Ollama backend
# ---------------------------------------------------------------------------

OLLAMA_URL: str = os.environ.get("OLLAMA_URL", "http://localhost:11434")
"""Base URL for the Ollama API. Override with OLLAMA_URL env var."""

# ---------------------------------------------------------------------------
# Daemon
# ---------------------------------------------------------------------------

DAEMON_HOST: str = os.environ.get("CORTEX_HOST", "127.0.0.1")
DAEMON_PORT: int = int(os.environ.get("CORTEX_PORT", "11411"))

# ---------------------------------------------------------------------------
# Data directory
# ---------------------------------------------------------------------------

DATA_DIR: Path = Path(os.environ.get("CORTEX_DATA_DIR", Path.home() / ".cortex"))
DB_PATH: Path = DATA_DIR / "cortex.db"

# ---------------------------------------------------------------------------
# VRAM budget
# ---------------------------------------------------------------------------

# Fraction of total memory available for models.
# Apple Silicon: unified memory shared with OS → 75%
# Discrete GPU: dedicated VRAM → 90%
# CPU-only fallback: system RAM, conservative → 60%

VRAM_FRACTION_UNIFIED: float = 0.75   # Apple Silicon
VRAM_FRACTION_DISCRETE: float = 0.90  # NVIDIA / AMD
VRAM_FRACTION_CPU: float = 0.60       # CPU-only, system RAM


def vram_budget_mb(
    total_mb: int,
    has_gpu: bool = False,
    is_unified: bool = False,
) -> int:
    """
    Calculate the VRAM budget for model loading.

    Args:
        total_mb:    Total memory (VRAM or unified) in MB.
        has_gpu:     True if a discrete GPU is present.
        is_unified:  True for unified memory (Apple Silicon).

    Returns:
        Usable VRAM in MB.
    """
    if is_unified:
        return int(total_mb * VRAM_FRACTION_UNIFIED)
    if has_gpu:
        return int(total_mb * VRAM_FRACTION_DISCRETE)
    return int(total_mb * VRAM_FRACTION_CPU)
