"""
CKM Training Profiler — inspect hardware → choose training profile.

The model ladder:
  L0: rules only (no training needed)
  L1: tiny classifier / intent router     (1M params)
  L2: small SCL decoder                   (5M params)
  L3: CKM boot-config model              (15M-30M params)
  L4: local LLM adapter / LoRA           (requires HF ecosystem)
  L5: remote teacher distillation         (optional, network required)

For "any machine," L1-L3 are the important layers.

This module:
  1. Probes available hardware (CPU cores, RAM, GPU, VRAM)
  2. Selects the largest training job that fits within the time budget
  3. Picks model size, batch size, precision, and device
  4. Returns a TrainingProfile used by train.py

Example:
  | Machine                 | Training target                            |
  | ----------------------- | ------------------------------------------ |
  | Low-end CPU laptop      | Tiny intent classifier + SCL grammar model |
  | 16-32GB RAM CPU machine | Small CKM/SCL model                       |
  | 64GB RAM machine        | Larger CKM model, more evals              |
  | 8GB VRAM GPU            | Small transformer from scratch or LoRA     |
  | Larger GPU              | Distillation + boot-policy model           |
"""

import os
import platform
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("cortex.ckm.profile")


# ---------------------------------------------------------------------------
# Model size ladder
# ---------------------------------------------------------------------------

@dataclass
class ModelSpec:
    """Specification for a CKM model size."""
    name: str
    params: int               # Total parameters
    n_layers: int
    d_model: int
    n_heads: int
    d_ff: int                 # Feed-forward inner dimension
    vocab_size: int
    max_seq_len: int
    estimated_ram_mb: float   # RAM needed for training (rough)
    estimated_vram_mb: float  # VRAM needed for training (rough)
    training_minutes_cpu: float   # Rough estimate on 8-core CPU
    training_minutes_gpu: float   # Rough estimate on 8GB GPU


MODEL_LADDER = {
    "ckm-1m": ModelSpec(
        name="ckm-1m",
        params=1_000_000,
        n_layers=4,
        d_model=128,
        n_heads=4,
        d_ff=512,
        vocab_size=512,
        max_seq_len=256,
        estimated_ram_mb=512,
        estimated_vram_mb=256,
        training_minutes_cpu=2,
        training_minutes_gpu=0.5,
    ),
    "ckm-5m": ModelSpec(
        name="ckm-5m",
        params=5_000_000,
        n_layers=6,
        d_model=256,
        n_heads=8,
        d_ff=1024,
        vocab_size=512,
        max_seq_len=256,
        estimated_ram_mb=1024,
        estimated_vram_mb=512,
        training_minutes_cpu=8,
        training_minutes_gpu=2,
    ),
    "ckm-15m": ModelSpec(
        name="ckm-15m",
        params=15_000_000,
        n_layers=8,
        d_model=384,
        n_heads=8,
        d_ff=1536,
        vocab_size=512,
        max_seq_len=256,
        estimated_ram_mb=2048,
        estimated_vram_mb=1024,
        training_minutes_cpu=20,
        training_minutes_gpu=5,
    ),
    "ckm-30m": ModelSpec(
        name="ckm-30m",
        params=30_000_000,
        n_layers=12,
        d_model=512,
        n_heads=8,
        d_ff=2048,
        vocab_size=512,
        max_seq_len=256,
        estimated_ram_mb=4096,
        estimated_vram_mb=2048,
        training_minutes_cpu=45,
        training_minutes_gpu=10,
    ),
    "ckm-60m": ModelSpec(
        name="ckm-60m",
        params=60_000_000,
        n_layers=16,
        d_model=640,
        n_heads=10,
        d_ff=2560,
        vocab_size=512,
        max_seq_len=256,
        estimated_ram_mb=8192,
        estimated_vram_mb=4096,
        training_minutes_cpu=90,
        training_minutes_gpu=20,
    ),
}


# ---------------------------------------------------------------------------
# Hardware detection (lightweight, no deps)
# ---------------------------------------------------------------------------

@dataclass
class HardwareInfo:
    """Detected hardware for training profile selection."""
    cpu_cores: int
    ram_mb: int
    gpu_type: str              # "cuda", "mps", "rocm", "none"
    gpu_name: str
    vram_mb: int
    arch: str                  # x86_64, aarch64

    @property
    def has_gpu(self) -> bool:
        return self.gpu_type != "none" and self.vram_mb > 0


def detect_hardware() -> HardwareInfo:
    """Detect available hardware for training."""
    cpu_cores = os.cpu_count() or 4
    arch = platform.machine()

    # RAM
    ram_mb = 4096  # default
    try:
        if platform.system() == "Darwin":
            import subprocess
            r = subprocess.run(["sysctl", "-n", "hw.memsize"],
                             capture_output=True, text=True, timeout=5)
            if r.returncode == 0:
                ram_mb = int(r.stdout.strip()) // (1024 * 1024)
        elif platform.system() == "Linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        ram_mb = int(line.split()[1]) // 1024
                        break
    except Exception:
        pass

    # GPU detection
    gpu_type = "none"
    gpu_name = "none"
    vram_mb = 0

    # Try CUDA
    try:
        import subprocess
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0 and r.stdout.strip():
            parts = r.stdout.strip().split(",")
            gpu_name = parts[0].strip()
            vram_mb = int(parts[1].strip())
            gpu_type = "cuda"
    except Exception:
        pass

    # Try Apple Metal (MPS)
    if gpu_type == "none" and platform.system() == "Darwin":
        if arch == "arm64" or arch == "aarch64":
            gpu_type = "mps"
            gpu_name = "Apple Silicon"
            # On Apple Silicon, VRAM = shared with system RAM
            vram_mb = ram_mb  # unified memory

    # Try ROCm
    if gpu_type == "none":
        try:
            import subprocess
            r = subprocess.run(
                ["rocm-smi", "--showmeminfo", "vram"],
                capture_output=True, text=True, timeout=5,
            )
            if r.returncode == 0 and "Total" in r.stdout:
                for line in r.stdout.splitlines():
                    if "Total" in line:
                        vram_mb = int(line.split()[-2]) // (1024 * 1024)
                        gpu_type = "rocm"
                        gpu_name = "AMD GPU"
                        break
        except Exception:
            pass

    return HardwareInfo(
        cpu_cores=cpu_cores,
        ram_mb=ram_mb,
        gpu_type=gpu_type,
        gpu_name=gpu_name,
        vram_mb=vram_mb,
        arch=arch,
    )


# ---------------------------------------------------------------------------
# Training profile
# ---------------------------------------------------------------------------

@dataclass
class TrainingProfile:
    """Complete training configuration selected by the profiler."""
    model_spec: ModelSpec
    device: str                # "cuda", "mps", "cpu"
    precision: str             # "fp32", "fp16", "bf16"
    batch_size: int
    gradient_accumulation: int
    learning_rate: float
    max_epochs: int
    time_budget_minutes: float
    num_workers: int           # data loading workers
    compile_model: bool        # torch.compile for speed
    early_stop_patience: int   # checkpoints without improvement

    def to_dict(self) -> dict:
        return {
            "model": self.model_spec.name,
            "params": self.model_spec.params,
            "device": self.device,
            "precision": self.precision,
            "batch_size": self.batch_size,
            "gradient_accumulation": self.gradient_accumulation,
            "learning_rate": self.learning_rate,
            "max_epochs": self.max_epochs,
            "time_budget_minutes": self.time_budget_minutes,
            "num_workers": self.num_workers,
            "compile_model": self.compile_model,
            "early_stop_patience": self.early_stop_patience,
        }


def select_training_profile(
    hardware: Optional[HardwareInfo] = None,
    time_budget_minutes: float = 10.0,
    target_model: Optional[str] = None,
) -> TrainingProfile:
    """
    Select the best training profile for the available hardware and time budget.

    Strategy:
      1. Detect hardware if not provided
      2. Filter model ladder by what fits in RAM/VRAM
      3. Filter by what can complete within time budget
      4. Select the largest feasible model
      5. Optimize batch size and precision for throughput
    """
    if hardware is None:
        hardware = detect_hardware()

    logger.info(
        "Hardware: %d cores, %dMB RAM, %s (%s, %dMB VRAM)",
        hardware.cpu_cores, hardware.ram_mb,
        hardware.gpu_name, hardware.gpu_type, hardware.vram_mb,
    )

    # If user specified a target, use it directly
    if target_model and target_model in MODEL_LADDER:
        spec = MODEL_LADDER[target_model]
    else:
        # Select largest feasible model
        spec = _select_model_for_hardware(hardware, time_budget_minutes)

    # Select device
    device = "cpu"
    if hardware.gpu_type == "cuda":
        device = "cuda"
    elif hardware.gpu_type == "mps":
        device = "mps"

    # Select precision
    if device == "cuda":
        precision = "fp16"
    elif device == "mps":
        precision = "fp16"
    else:
        precision = "fp32"

    # Select batch size (maximize without OOM)
    if device == "cpu":
        # Limited by RAM
        available_mb = hardware.ram_mb * 0.6  # leave headroom
        batch_size = max(1, min(16, int(available_mb / spec.estimated_ram_mb * 4)))
    else:
        # Limited by VRAM
        available_mb = hardware.vram_mb * 0.7
        batch_size = max(1, min(32, int(available_mb / spec.estimated_vram_mb * 8)))

    # Gradient accumulation to simulate larger effective batch
    effective_batch = 32
    gradient_accumulation = max(1, effective_batch // batch_size)

    # Learning rate (scale with effective batch size)
    base_lr = 3e-4
    lr = base_lr * (batch_size * gradient_accumulation / 32) ** 0.5

    # Max epochs (limited by time budget)
    if device != "cpu":
        est_minutes_per_epoch = spec.training_minutes_gpu
    else:
        est_minutes_per_epoch = spec.training_minutes_cpu
    max_epochs = max(1, int(time_budget_minutes / est_minutes_per_epoch))

    # Workers
    num_workers = min(4, hardware.cpu_cores // 2)

    # Compile (only if torch >= 2.0 and CUDA)
    compile_model = device == "cuda"

    profile = TrainingProfile(
        model_spec=spec,
        device=device,
        precision=precision,
        batch_size=batch_size,
        gradient_accumulation=gradient_accumulation,
        learning_rate=lr,
        max_epochs=max_epochs,
        time_budget_minutes=time_budget_minutes,
        num_workers=num_workers,
        compile_model=compile_model,
        early_stop_patience=3,
    )

    logger.info("Selected profile: %s", profile.to_dict())
    return profile


def _select_model_for_hardware(hardware: HardwareInfo, time_budget: float) -> ModelSpec:
    """Pick the largest model that fits in hardware and time budget."""
    candidates = []

    for name, spec in MODEL_LADDER.items():
        # Check RAM fit
        if spec.estimated_ram_mb > hardware.ram_mb * 0.7:
            continue

        # Check VRAM fit (if using GPU)
        if hardware.has_gpu and spec.estimated_vram_mb > hardware.vram_mb * 0.7:
            # Can still run on CPU
            time_est = spec.training_minutes_cpu
        elif hardware.has_gpu:
            time_est = spec.training_minutes_gpu
        else:
            time_est = spec.training_minutes_cpu

        # Check time budget (at least 1 epoch must fit)
        if time_est > time_budget:
            continue

        candidates.append((spec.params, spec))

    if not candidates:
        # Fallback to smallest model
        return MODEL_LADDER["ckm-1m"]

    # Return largest that fits
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]
