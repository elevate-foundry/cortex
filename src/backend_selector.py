"""
Cortex backend selector — maps a SystemProfile to the optimal inference
configuration for minimizing TTFT (Time to First Token).

Cross-platform priority (llama.cpp-first):
  1. NVIDIA GPU + vLLM (AWQ)         — best TTFT at scale, prefix caching
  2. Any GPU/CPU + llama.cpp (GGUF)  — universal, works everywhere
  3. Any + Ollama                    — wraps llama.cpp, easiest setup
  
llama.cpp is the universal backend:
  - Linux + NVIDIA CUDA
  - Linux + AMD ROCm
  - macOS + Apple Metal
  - Any OS + CPU (AVX2/NEON)
  
Same GGUF model files, same API, same behavior everywhere.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .hardware_detect import (
    AcceleratorType,
    SystemProfile,
    BackendAvailability,
)
from .tiers import (
    Tier,
    TierModel,
    max_feasible_tier,
    get_models_for_tier,
)


class InferenceBackend(str, Enum):
    VLLM = "vllm"
    LLAMA_CPP = "llama_cpp"
    OLLAMA = "ollama"


class QuantFormat(str, Enum):
    NONE = "none"
    AWQ = "awq"
    GGUF_Q4_K_M = "Q4_K_M"
    GGUF_Q5_K_M = "Q5_K_M"
    GGUF_Q8_0 = "Q8_0"
    GGUF_F16 = "F16"


@dataclass
class ModelRecommendation:
    """A recommended model + quantization for the detected hardware."""
    model_id: str
    quant: QuantFormat
    max_context: int
    estimated_vram_mb: int
    reason: str


@dataclass
class InferenceConfig:
    """Complete inference configuration for minimum TTFT."""
    backend: InferenceBackend
    model: ModelRecommendation
    tier: Tier = Tier.L3
    # Backend-specific settings
    tensor_parallel: int = 1
    gpu_layers: int = -1  # -1 = all layers on GPU (llama.cpp)
    kv_cache_dtype: str = "auto"
    prefix_caching: bool = True  # Critical for TTFT
    continuous_batching: bool = True
    max_batch_size: int = 1  # Start with 1 for lowest TTFT
    threads: int = 0  # 0 = auto
    flash_attention: bool = True
    extra_args: dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"=== Inference Configuration (TTFT-Optimized) ===",
            f"Backend:         {self.backend.value}",
            f"Tier:            {self.tier.name}",
            f"Model:           {self.model.model_id}",
            f"Quantization:    {self.model.quant.value}",
            f"Max Context:     {self.model.max_context:,}",
            f"Est. VRAM:       {self.model.estimated_vram_mb:,} MB",
            f"Prefix Caching:  {self.prefix_caching}",
            f"Flash Attention: {self.flash_attention}",
            f"Tensor Parallel: {self.tensor_parallel}",
            f"Reason:          {self.model.reason}",
        ]
        if self.extra_args:
            lines.append(f"Extra Args:      {self.extra_args}")
        return "\n".join(lines)

    def to_launch_command(self) -> list[str]:
        """Generate the shell command to launch the inference server."""
        if self.backend == InferenceBackend.VLLM:
            return self._vllm_cmd()
        elif self.backend == InferenceBackend.LLAMA_CPP:
            return self._llama_cpp_cmd()
        elif self.backend == InferenceBackend.OLLAMA:
            return self._ollama_cmd()
        else:
            return ["echo", f"Launch command not implemented for {self.backend.value}"]

    def _vllm_cmd(self) -> list[str]:
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model.model_id,
            "--max-model-len", str(self.model.max_context),
            "--dtype", "auto",
            "--port", "8000",
        ]
        if self.prefix_caching:
            cmd.append("--enable-prefix-caching")
        if self.tensor_parallel > 1:
            cmd.extend(["--tensor-parallel-size", str(self.tensor_parallel)])
        if self.kv_cache_dtype != "auto":
            cmd.extend(["--kv-cache-dtype", self.kv_cache_dtype])
        return cmd

    def _llama_cpp_cmd(self) -> list[str]:
        cmd = [
            "llama-server",
            "-m", self.model.model_id,
            "--port", "8000",
            "-c", str(self.model.max_context),
            "--n-gpu-layers", str(self.gpu_layers),
        ]
        if self.threads > 0:
            cmd.extend(["-t", str(self.threads)])
        if self.flash_attention:
            cmd.append("-fa")
        if self.continuous_batching:
            cmd.append("-cb")
        return cmd

    def _ollama_cmd(self) -> list[str]:
        return ["ollama", "serve"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_backend(backends: list[BackendAvailability], name: str) -> bool:
    return any(b.name == name and b.available for b in backends)


def _tier_model_to_recommendation(
    tm: TierModel,
    tier: Tier,
    reason: str,
) -> ModelRecommendation:
    """Convert a TierModel to a ModelRecommendation."""
    quant_map = {
        "Q4_K_M": QuantFormat.GGUF_Q4_K_M,
        "Q5_K_M": QuantFormat.GGUF_Q5_K_M,
        "Q8_0": QuantFormat.GGUF_Q8_0,
        "F16": QuantFormat.GGUF_F16,
        "awq": QuantFormat.AWQ,
        "api": QuantFormat.NONE,
    }
    return ModelRecommendation(
        model_id=tm.model_id,
        quant=quant_map.get(tm.quant, QuantFormat.GGUF_Q4_K_M),
        max_context=tm.context_default,
        estimated_vram_mb=tm.vram_mb,
        reason=reason,
    )


# ---------------------------------------------------------------------------
# Main selector
# ---------------------------------------------------------------------------

def select_backend(
    profile: SystemProfile,
    tier: Optional[Tier] = None,
) -> InferenceConfig:
    """
    Given a system profile, select the optimal backend + model + config
    for minimum TTFT.
    
    Uses GGUF as the universal model format (works on Linux/Mac/Windows,
    CUDA/ROCm/Metal/CPU). Falls back to AWQ only for NVIDIA+vLLM.
    
    Args:
        profile: Detected system profile
        tier: Optional specific tier to configure. If None, uses max feasible.
    """
    accel = profile.primary_accelerator
    backends = profile.backends

    # Determine which tier to serve
    if tier is None:
        tier = max_feasible_tier(profile)

    # Get candidate models from the universal catalog
    models = get_models_for_tier(tier, profile)
    if not models:
        # Absolute fallback
        return InferenceConfig(
            backend=InferenceBackend.OLLAMA,
            model=ModelRecommendation(
                model_id="qwen2.5:3b",
                quant=QuantFormat.GGUF_Q4_K_M,
                max_context=4096,
                estimated_vram_mb=2400,
                reason="No models in catalog. Install Ollama or llama.cpp.",
            ),
            tier=Tier.L2,
            extra_args={"install_hint": "curl -fsSL https://ollama.com/install.sh | sh"},
        )

    # Pick the first fitting model
    tm = models[0]
    reason = f"{tier.name} model — GGUF runs on any platform (CUDA/Metal/ROCm/CPU)"

    # --- NVIDIA + vLLM path (optional AWQ upgrade) ---
    if (accel == AcceleratorType.NVIDIA_CUDA
            and _has_backend(backends, "vLLM")
            and tm.format == "awq"):
        model = _tier_model_to_recommendation(tm, tier, f"{tier.name} AWQ model via vLLM (prefix caching enabled)")
        tp = len(profile.gpus) if len(profile.gpus) > 1 else 1
        return InferenceConfig(
            backend=InferenceBackend.VLLM,
            model=model,
            tier=tier,
            tensor_parallel=tp,
            prefix_caching=True,
            continuous_batching=True,
            flash_attention=True,
            kv_cache_dtype="fp8" if any(
                g.compute_capability >= "8.9" for g in profile.gpus
            ) else "auto",
        )

    # --- Universal llama.cpp path (GGUF) ---
    # Find the GGUF model (skip AWQ entries)
    gguf_model = None
    for m in models:
        if m.format == "gguf":
            gguf_model = m
            break

    if gguf_model is None:
        gguf_model = tm  # use whatever we have

    model = _tier_model_to_recommendation(gguf_model, tier, reason)

    # Determine GPU layer offloading
    has_gpu = accel in (
        AcceleratorType.NVIDIA_CUDA,
        AcceleratorType.APPLE_METAL,
        AcceleratorType.AMD_ROCM,
    )
    gpu_layers = -1 if has_gpu else 0

    if _has_backend(backends, "llama.cpp"):
        return InferenceConfig(
            backend=InferenceBackend.LLAMA_CPP,
            model=model,
            tier=tier,
            gpu_layers=gpu_layers,
            flash_attention=has_gpu,
            continuous_batching=True,
            threads=profile.cpu.physical_cores,
        )

    # --- Ollama fallback (wraps llama.cpp) ---
    if _has_backend(backends, "Ollama"):
        # Use ollama tag if available
        ollama_id = gguf_model.ollama_tag or f"qwen2.5:{tier.name.lower()}"
        model.model_id = ollama_id
        model.reason = f"{tier.name} via Ollama (wraps llama.cpp)"
        return InferenceConfig(
            backend=InferenceBackend.OLLAMA,
            model=model,
            tier=tier,
        )

    # --- Nothing installed ---
    model.reason = "No inference backend detected. Install llama.cpp or Ollama."
    return InferenceConfig(
        backend=InferenceBackend.OLLAMA,
        model=model,
        tier=tier,
        extra_args={"install_hint": "curl -fsSL https://ollama.com/install.sh | sh"},
    )
