"""
Cortex tier system — OS-level inference hierarchy.

L0 (0.5-1B)   — reflex/router: classify intent, select mode, detect danger
L1 (1-2B)     — tiny syscall agent: narrow tool calls, summarization
L2 (3-4B)     — always-hot local agent: 10-25 tools, file ops, drafts
L3 (7-8B)     — primary local OS agent: 20-50 tools, coding, multi-step
L4 (12-14B)   — heavy local reasoner: debugging, planning, code repair
L5 (30-32B)   — local frontier-ish: serious hardware, strong reasoning
L6 (64-70B)   — workstation model: strong local assistant
L7 (frontier) — remote escalation: when local confidence fails
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

from .hardware_detect import AcceleratorType, SystemProfile


class Tier(IntEnum):
    L0 = 0  # reflex / router
    L1 = 1  # tiny syscall agent
    L2 = 2  # always-hot local agent
    L3 = 3  # primary local OS agent
    L4 = 4  # heavy local reasoner
    L5 = 5  # local frontier-ish
    L6 = 6  # workstation model
    L7 = 7  # remote frontier (escalation)


@dataclass
class TierSpec:
    tier: Tier
    label: str
    os_role: str
    param_range: str           # e.g. "0.5B-1B"
    param_min_b: float         # billions, lower bound
    param_max_b: float         # billions, upper bound
    capabilities: list[str]
    vram_min_mb: int           # minimum VRAM to run at 4-bit quant
    ttft_target_ms: int        # target TTFT in ms
    always_hot: bool = False   # should this tier stay loaded in memory?
    local: bool = True         # runs locally (False = L7 remote)


# ---------------------------------------------------------------------------
# Tier definitions
# ---------------------------------------------------------------------------

TIER_SPECS: dict[Tier, TierSpec] = {
    Tier.L0: TierSpec(
        tier=Tier.L0,
        label="L0 — Reflex/Router",
        os_role="reflex/router",
        param_range="0.5B–1B",
        param_min_b=0.5,
        param_max_b=1.0,
        capabilities=[
            "classify intent",
            "select mode/tier",
            "detect danger/safety",
            "simple JSON extraction",
            "yes/no decisions",
        ],
        vram_min_mb=500,
        ttft_target_ms=10,
        always_hot=True,
        local=True,
    ),
    Tier.L1: TierSpec(
        tier=Tier.L1,
        label="L1 — Tiny Syscall Agent",
        os_role="tiny syscall agent",
        param_range="1B–2B",
        param_min_b=1.0,
        param_max_b=2.0,
        capabilities=[
            "narrow tool calls",
            "shell-safe commands",
            "summarization",
            "local memory lookup",
            "simple reformatting",
        ],
        vram_min_mb=1000,
        ttft_target_ms=20,
        always_hot=True,
        local=True,
    ),
    Tier.L2: TierSpec(
        tier=Tier.L2,
        label="L2 — Always-Hot Local Agent",
        os_role="always-hot local agent",
        param_range="3B–4B",
        param_min_b=3.0,
        param_max_b=4.0,
        capabilities=[
            "10–25 clean tools",
            "file operations",
            "simple code edits",
            "email/calendar drafts",
            "structured extraction",
        ],
        vram_min_mb=2000,
        ttft_target_ms=40,
        always_hot=True,
        local=True,
    ),
    Tier.L3: TierSpec(
        tier=Tier.L3,
        label="L3 — Primary Local OS Agent",
        os_role="primary local OS agent",
        param_range="7B–8B",
        param_min_b=7.0,
        param_max_b=8.0,
        capabilities=[
            "20–50 tools with validation",
            "coding",
            "recovery from errors",
            "multi-step tasks",
            "context-aware routing",
        ],
        vram_min_mb=4500,
        ttft_target_ms=60,
        always_hot=False,
        local=True,
    ),
    Tier.L4: TierSpec(
        tier=Tier.L4,
        label="L4 — Heavy Local Reasoner",
        os_role="heavy local reasoner",
        param_range="12B–14B",
        param_min_b=12.0,
        param_max_b=14.0,
        capabilities=[
            "harder debugging",
            "planning",
            "more reliable code/tool repair",
            "complex multi-step reasoning",
            "document analysis",
        ],
        vram_min_mb=8000,
        ttft_target_ms=100,
        always_hot=False,
        local=True,
    ),
    Tier.L5: TierSpec(
        tier=Tier.L5,
        label="L5 — Local Frontier-ish",
        os_role="local frontier-ish",
        param_range="30B–32B",
        param_min_b=30.0,
        param_max_b=32.0,
        capabilities=[
            "best local reasoning",
            "complex analysis",
            "long-form generation",
            "multi-domain expertise",
            "nuanced judgment",
        ],
        vram_min_mb=18000,
        ttft_target_ms=200,
        always_hot=False,
        local=True,
    ),
    Tier.L6: TierSpec(
        tier=Tier.L6,
        label="L6 — Workstation Model",
        os_role="workstation model",
        param_range="64B–70B",
        param_min_b=64.0,
        param_max_b=70.0,
        capabilities=[
            "strong local assistant",
            "frontier-adjacent quality",
            "complex code generation",
            "deep analysis",
            "multi-turn complex tasks",
        ],
        vram_min_mb=40000,
        ttft_target_ms=400,
        always_hot=False,
        local=True,
    ),
    Tier.L7: TierSpec(
        tier=Tier.L7,
        label="L7 — Remote Frontier",
        os_role="escalation only",
        param_range="remote frontier",
        param_min_b=0,
        param_max_b=0,
        capabilities=[
            "when local confidence/evals fail",
            "highest quality reasoning",
            "novel/ambiguous tasks",
            "safety-critical decisions",
            "frontier-only capabilities",
        ],
        vram_min_mb=0,
        ttft_target_ms=500,
        always_hot=False,
        local=False,
    ),
}


# ---------------------------------------------------------------------------
# Model catalog — concrete model IDs per tier × accelerator
# ---------------------------------------------------------------------------

@dataclass
class TierModel:
    """A concrete model assigned to a tier for a specific accelerator."""
    model_id: str
    quant: str              # "Q4_K_M", "Q5_K_M", "Q8_0", "awq", "api", etc.
    format: str             # "gguf", "awq", "api"
    vram_mb: int            # estimated VRAM at this quant
    context_default: int    # default max context length
    family: str = "qwen"    # model family: qwen, llama, gemma, granite, phi, olmo, smollm
    ollama_tag: str = ""    # ollama model tag if applicable


# ---------------------------------------------------------------------------
# UNIVERSAL MODEL CATALOG — GGUF-first, cross-platform
#
# GGUF (llama.cpp) is the primary format because it runs identically on:
#   - Linux + NVIDIA CUDA
#   - Linux + AMD ROCm
#   - macOS + Apple Metal
#   - Any OS + CPU (AVX2/NEON)
#
# Same model file, same API, same behavior everywhere.
# NVIDIA+vLLM users can optionally use AWQ for better prefix caching.
# ---------------------------------------------------------------------------

# Key: Tier → list of models (preference order, universal)
UNIVERSAL_CATALOG: dict[Tier, list[TierModel]] = {
    # =====================================================================
    # L0 — Reflex/Router (0.5B-1B)  →  Qwen3-0.6B
    # =====================================================================
    Tier.L0: [
        TierModel("unsloth/Qwen3-0.6B-GGUF:Q4_K_M", "Q4_K_M", "gguf", 500, 4096,
                  family="qwen", ollama_tag="qwen3:0.6b"),
    ],

    # =====================================================================
    # L1 — Tiny Syscall Agent (1B-2B)  →  Qwen3-1.7B
    # =====================================================================
    Tier.L1: [
        TierModel("unsloth/Qwen3-1.7B-GGUF:Q4_K_M", "Q4_K_M", "gguf", 1200, 4096,
                  family="qwen", ollama_tag="qwen3:1.7b"),
    ],

    # =====================================================================
    # L2 — Always-Hot Local Agent (3B-4B)  →  Qwen3-4B
    # =====================================================================
    Tier.L2: [
        TierModel("unsloth/Qwen3-4B-GGUF:Q4_K_M", "Q4_K_M", "gguf", 2800, 8192,
                  family="qwen", ollama_tag="qwen3:4b"),
    ],

    # =====================================================================
    # L3 — Primary Local OS Agent (7B-8B)  →  Qwen3-8B
    # =====================================================================
    Tier.L3: [
        TierModel("unsloth/Qwen3-8B-GGUF:Q4_K_M", "Q4_K_M", "gguf", 5000, 8192,
                  family="qwen", ollama_tag="qwen3:8b"),
    ],

    # =====================================================================
    # L4 — Heavy Local Reasoner (12B-14B)  →  Qwen3-14B
    # =====================================================================
    Tier.L4: [
        TierModel("unsloth/Qwen3-14B-GGUF:Q4_K_M", "Q4_K_M", "gguf", 9000, 8192,
                  family="qwen", ollama_tag="qwen3:14b"),
    ],

    # =====================================================================
    # L5 — Local Frontier-ish (30B-32B)  →  Qwen3-30B-A3B (MoE) + Qwen3-32B
    # =====================================================================
    Tier.L5: [
        TierModel("unsloth/Qwen3-30B-A3B-GGUF:Q4_K_M", "Q4_K_M", "gguf", 18000, 8192,
                  family="qwen", ollama_tag="qwen3:30b-a3b"),
        TierModel("unsloth/Qwen3-32B-GGUF:Q4_K_M", "Q4_K_M", "gguf", 19000, 8192,
                  family="qwen", ollama_tag="qwen3:32b"),
    ],

    # =====================================================================
    # L6 — Workstation Model (64B-70B)
    # =====================================================================
    Tier.L6: [
        TierModel("bartowski/Qwen2.5-72B-Instruct-GGUF:Q4_K_M", "Q4_K_M", "gguf", 42000, 8192,
                  family="qwen", ollama_tag="qwen2.5:72b"),
    ],

    # =====================================================================
    # L7 — Remote Frontier (API-based)
    # =====================================================================
    Tier.L7: [
        TierModel("openai/gpt-4o", "api", "api", 0, 128000, family="openai"),
        TierModel("anthropic/claude-sonnet-4-20250514", "api", "api", 0, 200000, family="anthropic"),
    ],
}

# ---------------------------------------------------------------------------
# CHALLENGE MODEL CATALOG — cross-family models for confidence verification
#
# Different model families (Qwen, Llama, Gemma, Granite, Phi, OLMo, SmolLM)
# trained on different data with different architectures. When models from
# multiple families agree on an answer, confidence is much higher than
# N copies of the same family agreeing.
#
# Key: Tier → list of challenge models from non-Qwen families
# ---------------------------------------------------------------------------

CHALLENGE_CATALOG: dict[Tier, list[TierModel]] = {
    # L0 — 1B challenge models
    Tier.L0: [
        TierModel("bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M", "Q4_K_M", "gguf", 800, 4096,
                  family="llama", ollama_tag="llama3.2:1b"),
        TierModel("allenai/OLMo-2-0425-1B-Instruct-GGUF:Q4_K_M", "Q4_K_M", "gguf", 800, 4096,
                  family="olmo"),
        TierModel("bartowski/gemma-3-1b-it-GGUF:Q4_K_M", "Q4_K_M", "gguf", 800, 4096,
                  family="gemma", ollama_tag="gemma3:1b"),
    ],

    # L1 — 1-3B challenge models
    Tier.L1: [
        TierModel("bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M", "Q4_K_M", "gguf", 2000, 4096,
                  family="llama", ollama_tag="llama3.2:3b"),
        TierModel("bartowski/granite-3.3-2b-instruct-GGUF:Q4_K_M", "Q4_K_M", "gguf", 1500, 4096,
                  family="granite", ollama_tag="granite3.3:2b"),
        TierModel("bartowski/SmolLM3-3B-Instruct-GGUF:Q4_K_M", "Q4_K_M", "gguf", 2000, 8192,
                  family="smollm"),
    ],

    # L2 — 3-4B challenge models
    Tier.L2: [
        TierModel("bartowski/gemma-3-4b-it-GGUF:Q4_K_M", "Q4_K_M", "gguf", 2800, 8192,
                  family="gemma", ollama_tag="gemma3:4b"),
        TierModel("bartowski/Phi-4-mini-instruct-GGUF:Q4_K_M", "Q4_K_M", "gguf", 2500, 8192,
                  family="phi", ollama_tag="phi4-mini"),
    ],

    # L3 — 7-8B challenge models
    Tier.L3: [
        TierModel("bartowski/granite-3.3-8b-instruct-GGUF:Q4_K_M", "Q4_K_M", "gguf", 5000, 8192,
                  family="granite", ollama_tag="granite3.3:8b"),
        TierModel("bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M", "Q4_K_M", "gguf", 5000, 8192,
                  family="llama", ollama_tag="llama3.1:8b"),
    ],

    # L4 — 12-14B challenge models
    Tier.L4: [
        TierModel("bartowski/gemma-3-12b-it-GGUF:Q4_K_M", "Q4_K_M", "gguf", 8000, 8192,
                  family="gemma", ollama_tag="gemma3:12b"),
        TierModel("bartowski/Phi-4-14b-instruct-GGUF:Q4_K_M", "Q4_K_M", "gguf", 9000, 8192,
                  family="phi", ollama_tag="phi4:14b"),
    ],

    # L5 — 30-32B challenge models
    Tier.L5: [
        TierModel("bartowski/granite-3.3-8b-instruct-GGUF:Q4_K_M", "Q4_K_M", "gguf", 5000, 8192,
                  family="granite", ollama_tag="granite3.3:8b"),
        TierModel("allenai/OLMo-2-0425-32B-Instruct-GGUF:Q4_K_M", "Q4_K_M", "gguf", 19000, 8192,
                  family="olmo"),
    ],

    # L6 — 70B challenge models
    Tier.L6: [
        TierModel("bartowski/Meta-Llama-3.3-70B-Instruct-GGUF:Q4_K_M", "Q4_K_M", "gguf", 42000, 8192,
                  family="llama", ollama_tag="llama3.3:70b"),
    ],
}

# Optional NVIDIA-only upgrades: AWQ models for vLLM prefix caching.
# These are tried first when NVIDIA+vLLM is detected; fall back to GGUF.
NVIDIA_VLLM_OVERRIDES: dict[Tier, list[TierModel]] = {
    Tier.L0: [
        TierModel("Qwen/Qwen2.5-0.5B-Instruct-AWQ", "awq", "awq", 400, 4096),
    ],
    Tier.L1: [
        TierModel("Qwen/Qwen2.5-1.5B-Instruct-AWQ", "awq", "awq", 1000, 4096),
    ],
    Tier.L2: [
        TierModel("Qwen/Qwen2.5-3B-Instruct-AWQ", "awq", "awq", 2000, 8192),
    ],
    Tier.L3: [
        TierModel("Qwen/Qwen2.5-7B-Instruct-AWQ", "awq", "awq", 4500, 8192),
    ],
    Tier.L4: [
        TierModel("Qwen/Qwen2.5-14B-Instruct-AWQ", "awq", "awq", 9000, 16384),
    ],
    Tier.L5: [
        TierModel("Qwen/Qwen2.5-32B-Instruct-AWQ", "awq", "awq", 18000, 16384),
    ],
    Tier.L6: [
        TierModel("Qwen/Qwen2.5-72B-Instruct-AWQ", "awq", "awq", 42000, 8192),
    ],
}


def _parse_param_size(param_str: str) -> float:
    """Parse Ollama's parameter_size string to billions. E.g. '7B' → 7.0, '751.63M' → 0.75."""
    if not param_str:
        return 0.0
    param_str = param_str.strip().upper()
    try:
        if param_str.endswith("B"):
            return float(param_str[:-1])
        elif param_str.endswith("M"):
            return float(param_str[:-1]) / 1000.0
        elif param_str.endswith("K"):
            return float(param_str[:-1]) / 1_000_000.0
        else:
            return float(param_str)
    except ValueError:
        return 0.0


def _params_to_tier(params_b: float) -> Optional[Tier]:
    """Map a parameter count (in billions) to the appropriate tier."""
    if params_b <= 0:
        return None
    if params_b <= 1.0:
        return Tier.L0
    if params_b <= 2.5:
        return Tier.L1
    if params_b <= 5.0:
        return Tier.L2
    if params_b <= 10.0:
        return Tier.L3
    if params_b <= 16.0:
        return Tier.L4
    if params_b <= 35.0:
        return Tier.L5
    if params_b <= 80.0:
        return Tier.L6
    return None  # too large for any local tier


def _estimate_vram_mb(size_bytes: int, params_b: float) -> int:
    """Estimate VRAM needed from file size or param count."""
    if size_bytes > 0:
        # File size is a good proxy — model + overhead ~= 1.1x file size
        return int(size_bytes / 1e6 * 1.1)
    # Rough estimate: Q4 ≈ 0.6 GB per billion params
    return int(params_b * 600)


def _normalize_family(ollama_family: str) -> str:
    """Normalize Ollama family names to Cortex family names."""
    f = ollama_family.lower().strip()
    # Map variations to canonical family names
    family_map = {
        "qwen3": "qwen",
        "qwen35": "qwen",
        "qwen2": "qwen",
        "llama": "llama",
        "gemma": "gemma",
        "gemma3": "gemma",
        "phi3": "phi",
        "granite": "granite",
        "nomic-bert": "nomic",
    }
    return family_map.get(f, f)


# Patterns to skip — embedding models, custom fine-tunes that aren't general-purpose
_SKIP_PATTERNS = {"nomic-embed", "embed"}


def discover_ollama_models(
    ollama_url: str = "http://localhost:11434",
) -> dict[Tier, list[TierModel]]:
    """
    Query Ollama's /api/tags endpoint and map installed models to tiers.

    Returns a dict of Tier → list of TierModel for every model installed
    in Ollama that can be assigned to a tier.

    This discovers models that aren't in the hardcoded catalogs —
    your local llama3.3:70b, phi4, deepseek-r1, gemma3:4b, etc.
    """
    import urllib.request

    try:
        req = urllib.request.Request(f"{ollama_url}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            import json
            data = json.loads(resp.read())
    except Exception:
        return {}

    discovered: dict[Tier, list[TierModel]] = {}
    seen_tags: set[str] = set()

    # Collect all ollama tags already in the hardcoded catalogs
    for catalog in (UNIVERSAL_CATALOG, CHALLENGE_CATALOG):
        for models in catalog.values():
            for m in models:
                if m.ollama_tag:
                    seen_tags.add(m.ollama_tag)

    for model_info in data.get("models", []):
        tag = model_info.get("name", "")
        details = model_info.get("details", {})
        param_str = details.get("parameter_size", "")
        family_raw = details.get("family", "")
        quant = details.get("quantization_level", "Q4_K_M")
        fmt = details.get("format", "gguf")
        size_bytes = model_info.get("size", 0)

        # Skip if already in hardcoded catalog
        if tag in seen_tags:
            continue

        # Skip embedding models
        if any(skip in tag.lower() for skip in _SKIP_PATTERNS):
            continue

        params_b = _parse_param_size(param_str)
        tier = _params_to_tier(params_b)
        if tier is None:
            continue

        family = _normalize_family(family_raw)
        vram_mb = _estimate_vram_mb(size_bytes, params_b)

        # Default context based on tier
        ctx_defaults = {
            Tier.L0: 4096, Tier.L1: 4096, Tier.L2: 8192,
            Tier.L3: 8192, Tier.L4: 8192, Tier.L5: 8192,
            Tier.L6: 8192,
        }

        tm = TierModel(
            model_id=tag,
            quant=quant,
            format=fmt,
            vram_mb=vram_mb,
            context_default=ctx_defaults.get(tier, 8192),
            family=family,
            ollama_tag=tag,
        )

        if tier not in discovered:
            discovered[tier] = []
        discovered[tier].append(tm)

    return discovered


# Module-level cache for discovered models
_ollama_discovered: Optional[dict[Tier, list[TierModel]]] = None


def _get_discovered() -> dict[Tier, list[TierModel]]:
    """Lazy-load Ollama model discovery (cached)."""
    global _ollama_discovered
    if _ollama_discovered is None:
        _ollama_discovered = discover_ollama_models()
    return _ollama_discovered


def refresh_ollama_discovery():
    """Force re-scan of Ollama models (call after pulling/removing models)."""
    global _ollama_discovered
    _ollama_discovered = None


def get_models_for_tier(
    tier: Tier,
    profile: SystemProfile,
) -> list[TierModel]:
    """
    Get candidate models for a tier, in preference order.
    
    Strategy:
      1. If NVIDIA + vLLM available → try AWQ overrides first, then GGUF
      2. Otherwise → GGUF (works on Metal, CUDA, ROCm, CPU)
      3. L7 → always API
      4. Append any additional Ollama-discovered models not in the catalog
    """
    from .hardware_detect import AcceleratorType

    models: list[TierModel] = []

    # Check if we should try NVIDIA vLLM overrides
    has_nvidia = profile.primary_accelerator == AcceleratorType.NVIDIA_CUDA
    has_vllm = any(b.name == "vLLM" and b.available for b in profile.backends)

    if has_nvidia and has_vllm and tier in NVIDIA_VLLM_OVERRIDES:
        models.extend(NVIDIA_VLLM_OVERRIDES[tier])

    # Always include universal GGUF models as fallback
    if tier in UNIVERSAL_CATALOG:
        models.extend(UNIVERSAL_CATALOG[tier])

    # Append discovered Ollama models not already in catalog
    discovered = _get_discovered()
    if tier in discovered:
        models.extend(discovered[tier])

    return models


def get_challenge_models(
    tier: Tier,
    exclude_family: str = "qwen",
) -> list[TierModel]:
    """
    Get challenge models for a tier, optionally excluding a family.
    
    Used by the challenger/swarm to pick models from different families
    than the core model that produced the initial answer.
    """
    if tier not in CHALLENGE_CATALOG:
        return []

    if not exclude_family:
        return list(CHALLENGE_CATALOG[tier])

    return [m for m in CHALLENGE_CATALOG[tier] if m.family != exclude_family]


def get_all_models_for_tier(
    tier: Tier,
    profile: SystemProfile,
) -> list[TierModel]:
    """
    Get ALL models for a tier — core + challenge + discovered.
    Used by the swarm for maximum cross-family coverage.
    """
    core = get_models_for_tier(tier, profile)
    challenge = CHALLENGE_CATALOG.get(tier, [])

    # Deduplicate by ollama_tag
    seen = {m.ollama_tag for m in core if m.ollama_tag}
    extra = [m for m in challenge if not m.ollama_tag or m.ollama_tag not in seen]

    return core + extra


def print_discovered_models() -> str:
    """Show all Ollama-discovered models mapped to tiers."""
    discovered = discover_ollama_models()
    if not discovered:
        return "No additional Ollama models discovered (is Ollama running?)"

    lines = [
        "=== Discovered Ollama Models ===",
        "",
        f"{'Model':<50s} {'Family':<12s} {'Params':>8s} {'VRAM MB':>8s} {'Tier':<5s} {'Quant':<10s}",
        "─" * 95,
    ]

    total = 0
    for tier in sorted(discovered.keys()):
        for m in discovered[tier]:
            params_b = _parse_param_size(
                # Reverse-estimate from VRAM if needed
                f"{m.vram_mb / 600:.1f}B" if m.vram_mb > 0 else "?"
            )
            lines.append(
                f"{m.ollama_tag or m.model_id:<50s} {m.family:<12s} "
                f"{'?':>8s} {m.vram_mb:>8,} {tier.name:<5s} {m.quant:<10s}"
            )
            total += 1

    lines.append("")
    lines.append(f"Total discovered: {total} (not in hardcoded catalog)")

    # Also show catalog models
    lines.append("")
    lines.append("=== Catalog Models (hardcoded) ===")
    lines.append("")
    for catalog_name, catalog in [("Core", UNIVERSAL_CATALOG), ("Challenge", CHALLENGE_CATALOG)]:
        for tier in sorted(catalog.keys()):
            for m in catalog[tier]:
                installed = "✓" if m.ollama_tag else " "
                lines.append(
                    f"  {installed} {tier.name} [{catalog_name:9s}] "
                    f"{m.ollama_tag or m.model_id:<40s} {m.family:<10s} {m.vram_mb:>6,} MB"
                )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tier feasibility — what can this system actually run?
# ---------------------------------------------------------------------------

@dataclass
class TierFeasibility:
    tier: Tier
    spec: TierSpec
    feasible: bool
    model: Optional[TierModel]
    reason: str
    headroom_mb: int = 0      # VRAM headroom after loading model


def assess_tiers(profile: SystemProfile) -> list[TierFeasibility]:
    """
    Given a system profile, determine which tiers are feasible
    and which concrete model to use for each.
    
    Uses the universal GGUF catalog (works on all platforms) with
    optional NVIDIA+vLLM AWQ overrides when available.
    """
    accel = profile.primary_accelerator
    # For Apple Silicon, unified memory means GPU can use most of RAM
    if accel == AcceleratorType.APPLE_METAL:
        available_vram = int(profile.memory.total_mb * 0.75)
    elif profile.gpus:
        available_vram = profile.total_vram_mb
    else:
        # CPU-only: use ~60% of RAM
        available_vram = int(profile.memory.total_mb * 0.60)

    results: list[TierFeasibility] = []

    for tier in Tier:
        spec = TIER_SPECS[tier]

        # L7 is always feasible (it's remote)
        if tier == Tier.L7:
            models = get_models_for_tier(tier, profile)
            results.append(TierFeasibility(
                tier=tier,
                spec=spec,
                feasible=True,
                model=models[0] if models else None,
                reason="Remote API — always available (requires API key)",
            ))
            continue

        # Get models for this tier (universal GGUF + optional AWQ)
        models = get_models_for_tier(tier, profile)

        if not models:
            results.append(TierFeasibility(
                tier=tier,
                spec=spec,
                feasible=False,
                model=None,
                reason="No model available in catalog",
            ))
            continue

        # Find the first model that fits
        best_model = None
        for m in models:
            if m.vram_mb <= available_vram:
                best_model = m
                break

        if best_model:
            headroom = available_vram - best_model.vram_mb
            fmt_note = f" [{best_model.format.upper()}]" if best_model.format != "gguf" else ""
            results.append(TierFeasibility(
                tier=tier,
                spec=spec,
                feasible=True,
                model=best_model,
                headroom_mb=headroom,
                reason=f"Fits in {available_vram:,}MB with {headroom:,}MB headroom{fmt_note}",
            ))
        else:
            results.append(TierFeasibility(
                tier=tier,
                spec=spec,
                feasible=False,
                model=models[0],
                reason=f"Needs {models[0].vram_mb:,}MB but only {available_vram:,}MB available",
            ))

    return results


def max_feasible_tier(profile: SystemProfile) -> Tier:
    """Return the highest local tier this system can run."""
    assessments = assess_tiers(profile)
    max_tier = Tier.L0
    for a in assessments:
        if a.feasible and a.spec.local and a.tier > max_tier:
            max_tier = a.tier
    return max_tier


def hot_tiers(profile: SystemProfile) -> list[TierFeasibility]:
    """
    Return the tiers that should be kept always-loaded in memory.
    
    Strategy: load all always_hot tiers (L0, L1, L2) if they fit,
    plus the highest feasible tier for escalation.
    """
    assessments = assess_tiers(profile)
    hot: list[TierFeasibility] = []

    # Always-hot tiers
    for a in assessments:
        if a.feasible and a.spec.always_hot:
            hot.append(a)

    return hot


def concurrent_vram_budget(profile: SystemProfile) -> dict:
    """
    Calculate how many tiers can be loaded concurrently.
    
    Returns a dict with:
      - max_concurrent_tiers: list of tiers that can be loaded simultaneously
      - total_vram_used: sum of their VRAM
      - remaining_vram: leftover for KV cache / context
    """
    accel = profile.primary_accelerator
    if accel == AcceleratorType.APPLE_METAL:
        total = int(profile.memory.total_mb * 0.75)
    elif profile.gpus:
        total = profile.total_vram_mb
    else:
        total = int(profile.memory.total_mb * 0.60)

    assessments = assess_tiers(profile)
    feasible = [a for a in assessments if a.feasible and a.spec.local and a.model]

    # Greedy: load always-hot first, then highest tiers
    always_hot = [a for a in feasible if a.spec.always_hot]
    on_demand = sorted(
        [a for a in feasible if not a.spec.always_hot],
        key=lambda a: a.tier,
        reverse=True,
    )

    loaded: list[TierFeasibility] = []
    used = 0

    for a in always_hot:
        if used + a.model.vram_mb <= total:
            loaded.append(a)
            used += a.model.vram_mb

    for a in on_demand:
        if used + a.model.vram_mb <= total:
            loaded.append(a)
            used += a.model.vram_mb

    return {
        "max_concurrent_tiers": [a.tier.name for a in loaded],
        "total_vram_used_mb": used,
        "remaining_vram_mb": total - used,
        "details": [
            {
                "tier": a.tier.name,
                "model": a.model.model_id,
                "vram_mb": a.model.vram_mb,
                "always_hot": a.spec.always_hot,
            }
            for a in loaded
        ],
    }


def print_tier_report(profile: SystemProfile) -> str:
    """Generate a human-readable tier feasibility report."""
    assessments = assess_tiers(profile)
    budget = concurrent_vram_budget(profile)
    max_tier = max_feasible_tier(profile)

    lines = [
        "=== Tier Feasibility Report ===",
        "",
    ]

    for a in assessments:
        status = "✓" if a.feasible else "✗"
        hot_marker = " [HOT]" if a.spec.always_hot and a.feasible else ""
        model_str = a.model.model_id if a.model else "N/A"
        lines.append(
            f"  {status} {a.spec.label}{hot_marker}"
        )
        lines.append(
            f"    Model:  {model_str}"
        )
        lines.append(
            f"    TTFT:   ~{a.spec.ttft_target_ms}ms target"
        )
        lines.append(
            f"    Status: {a.reason}"
        )
        lines.append("")

    lines.append(f"Max local tier: {max_tier.name}")
    lines.append(f"")
    lines.append(f"=== Concurrent Loading Budget ===")
    lines.append(f"Loadable tiers: {', '.join(budget['max_concurrent_tiers'])}")
    lines.append(f"VRAM used:      {budget['total_vram_used_mb']:,} MB")
    lines.append(f"VRAM remaining: {budget['remaining_vram_mb']:,} MB (for KV cache)")

    return "\n".join(lines)
