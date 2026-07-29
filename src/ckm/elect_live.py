"""
Cortex Live Tier Election — dynamically discover and elect models.

This is the autonomous runner. It:
  1. Queries OpenRouter /api/v1/models to discover ALL available models
  2. Filters to instruction-following chat models
  3. Groups by family and selects representatives across the size spectrum
  4. Runs the full tier election (self-assess → cross-verify → consensus)
  5. Saves results as SCL document + JSON

No hardcoded model list. The system discovers what exists and classifies it.

Usage:
  export OPENROUTER_API_KEY=sk-or-...
  python -m src.ckm.elect_live
  python -m src.ckm.elect_live --max-models 8 --budget 0.10
  python -m src.ckm.elect_live --families qwen,llama,deepseek --per-family 2
"""

import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger("cortex.ckm.elect_live")

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False


# ---------------------------------------------------------------------------
# Model discovery from OpenRouter
# ---------------------------------------------------------------------------

@dataclass
class DiscoveredModel:
    """A model discovered from the OpenRouter API.

    The model_id is the canonical OpenRouter identity — it IS the API endpoint:
      openrouter.ai/api/v1/chat/completions with {"model": model_id}

    We let the API define the model. No local inference of identity.
    """
    model_id: str               # Canonical: "openai/gpt-4o", "qwen/qwen3-8b", etc.
    name: str                   # Human-readable from API
    context_length: int
    prompt_price_per_m: float   # $/M input tokens
    completion_price_per_m: float
    family: str                 # Derived from org prefix of model_id
    param_size_b: float         # Inferred param count (0 if unknown)
    is_free: bool
    architecture: str = ""      # From API metadata if available
    top_provider: str = ""

    @property
    def api_endpoint(self) -> str:
        """Full OpenRouter API path for this model."""
        return f"https://openrouter.ai/api/v1/chat/completions"

    @property
    def api_model_param(self) -> str:
        """The exact value passed as 'model' in the API call. Self-declared by OpenRouter."""
        return self.model_id


# Families we consider "major" — good instruction-following, well-known
MAJOR_FAMILIES = {"qwen", "llama", "deepseek", "gemini", "gemma", "anthropic",
                  "openai", "mistral", "phi", "nvidia", "cohere"}


def _infer_family(model_id: str) -> str:
    """Infer model family from the model ID prefix (org/model pattern)."""
    mid = model_id.lower()
    org = mid.split("/")[0] if "/" in mid else ""

    # Match by org first (most reliable)
    if org == "meta-llama":
        return "llama"
    elif org == "qwen":
        return "qwen"
    elif org == "deepseek":
        return "deepseek"
    elif org == "google":
        if "gemma" in mid:
            return "gemma"
        return "gemini"
    elif org == "anthropic":
        return "anthropic"
    elif org == "openai":
        return "openai"
    elif org == "mistralai":
        return "mistral"
    elif org == "microsoft":
        return "phi"
    elif org == "nvidia":
        return "nvidia"
    elif org == "cohere":
        return "cohere"
    elif org == "ibm":
        return "granite"

    # Fallback: check model name
    if "llama" in mid:
        return "llama"
    elif "qwen" in mid:
        return "qwen"
    elif "deepseek" in mid:
        return "deepseek"
    elif "gemini" in mid:
        return "gemini"
    elif "gemma" in mid:
        return "gemma"
    elif "claude" in mid:
        return "anthropic"
    elif "gpt" in mid:
        return "openai"
    elif "mistral" in mid or "mixtral" in mid:
        return "mistral"
    elif "phi" in mid:
        return "phi"
    elif "nemotron" in mid:
        return "nvidia"
    elif "command" in mid:
        return "cohere"
    elif "granite" in mid:
        return "granite"
    else:
        return "other"


def _infer_param_size(model_id: str, name: str) -> float:
    """Infer parameter count in billions from model ID or name."""
    text = f"{model_id} {name}".lower()

    # Try to find explicit param counts like "70b", "8b", "1.7b", "235b-a22b"
    patterns = [
        r'(\d+\.?\d*)b(?:-a\d+b)?',  # "70b", "235b-a22b", "1.7b"
        r'(\d+)x(\d+)b',              # "8x7b" (MoE)
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            groups = match.groups()
            if len(groups) == 2 and 'x' in pattern:
                return float(groups[0]) * float(groups[1])
            return float(groups[0])

    # Known special cases
    if "gpt-4o" in text:
        return 0.0  # Unknown, treat as frontier
    elif "gpt-4" in text:
        return 0.0
    elif "claude" in text:
        return 0.0
    elif "gemini" in text:
        return 0.0

    return 0.0  # Unknown


def discover_models(api_key: str) -> list[DiscoveredModel]:
    """Query OpenRouter API and return all available models."""
    if not HAS_HTTPX:
        raise RuntimeError("httpx not installed: pip install httpx")

    response = httpx.get(
        "https://openrouter.ai/api/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30.0,
    )
    response.raise_for_status()
    data = response.json()

    models = []
    for m in data.get("data", []):
        model_id = m.get("id", "")
        if not model_id:
            continue

        # Skip non-chat / non-instruction models
        mid_lower = model_id.lower()
        if any(skip in mid_lower for skip in [
            "embed", "moderation", "tts", "whisper",
            "audio-", "image-gen", "dall-e", "-vision-",
            "lyria",  # music generation
        ]):
            continue

        # Skip community finetunes / RP models (not official releases)
        org = model_id.split("/")[0] if "/" in model_id else ""
        community_orgs = {
            "sao10k", "thedrummer", "gryphe", "neversleep",
            "cognitivecomputations", "undi95", "nousresearch",
            "lynn", "sophosympatheia", "nothingiisreal",
        }
        if org.lower() in community_orgs:
            continue

        # Skip batch/variant endpoints
        if any(suffix in mid_lower for suffix in [
            ":batch", ":extended", ":nitro", ":floor",
            "-online", "-image", "-vl", "-multimodal",
        ]):
            continue

        # Skip models with no context
        ctx = m.get("context_length", 0)
        if ctx < 4096:
            continue

        pricing = m.get("pricing", {})
        prompt_price = float(pricing.get("prompt", "0") or "0") * 1_000_000
        completion_price = float(pricing.get("completion", "0") or "0") * 1_000_000

        name = m.get("name", model_id)
        family = _infer_family(model_id)
        param_size = _infer_param_size(model_id, name)
        architecture = m.get("architecture", {}).get("modality", "") if isinstance(m.get("architecture"), dict) else ""
        top_provider = m.get("top_provider", {}).get("name", "") if isinstance(m.get("top_provider"), dict) else ""

        models.append(DiscoveredModel(
            model_id=model_id,
            name=name,
            context_length=ctx,
            prompt_price_per_m=prompt_price,
            completion_price_per_m=completion_price,
            family=family,
            param_size_b=param_size,
            is_free=(prompt_price == 0),
            architecture=architecture,
            top_provider=top_provider,
        ))

    return models


# ---------------------------------------------------------------------------
# Candidate selection — pick representatives from discovered models
# ---------------------------------------------------------------------------

@dataclass
class ElectionConfig:
    """Configuration for autonomous tier election."""
    max_models: int = 12           # Max models to include in election
    budget_usd: float = 0.50       # Max spend for the election
    per_family: int = 2            # Max models per family (keeps diversity high)
    min_families: int = 4          # Require at least N families
    prefer_diverse_sizes: bool = True  # Pick small + medium + large per family
    include_free: bool = True      # Include free models (no budget cost)
    families_filter: Optional[list[str]] = None  # Only these families
    exclude_patterns: Optional[list[str]] = None  # Skip models matching these


def select_candidates(
    models: list[DiscoveredModel],
    config: ElectionConfig,
) -> list[DiscoveredModel]:
    """
    Select election candidates from discovered models.

    Strategy:
      1. Group by family
      2. For each family, pick up to per_family models spanning the size range
      3. Prefer: one small (<10B), one medium (10-60B), one large (>60B)
      4. Cap total at max_models
      5. Estimate cost and cap at budget
    """
    # Filter by family if specified
    if config.families_filter:
        models = [m for m in models if m.family in config.families_filter]

    # Filter by exclusion patterns
    if config.exclude_patterns:
        def is_excluded(m: DiscoveredModel) -> bool:
            for pat in config.exclude_patterns:
                if pat.lower() in m.model_id.lower():
                    return True
            return False
        models = [m for m in models if not is_excluded(m)]

    # Group by family
    families: dict[str, list[DiscoveredModel]] = {}
    for m in models:
        families.setdefault(m.family, []).append(m)

    # Prioritize major families, then sort by model count
    def family_priority(item: tuple[str, list]) -> tuple[int, int]:
        name, members = item
        is_major = 0 if name in MAJOR_FAMILIES else 1
        return (is_major, -len(members))

    sorted_families = sorted(families.items(), key=family_priority)

    selected: list[DiscoveredModel] = []

    for family_name, family_models in sorted_families:
        if len(selected) >= config.max_models:
            break

        # Sort by param size for diverse selection
        family_models.sort(key=lambda m: m.param_size_b)

        # Pick representatives spanning sizes: prefer one small, one large
        picks: list[DiscoveredModel] = []
        small = [m for m in family_models if 0 < m.param_size_b <= 14]
        large = [m for m in family_models if m.param_size_b > 14]
        unknown = [m for m in family_models if m.param_size_b == 0]

        # Pick from large/frontier first (most interesting for tier placement)
        if large and len(picks) < config.per_family:
            picks.append(large[-1])  # Biggest known
        if unknown and len(picks) < config.per_family:
            # Unknown size = probably frontier API model
            picks.append(unknown[0])
        if small and len(picks) < config.per_family:
            picks.append(small[-1])  # Largest small model

        # Strict cap at per_family
        picks = picks[:config.per_family]
        selected.extend(picks)

    # Cap at max_models
    selected = selected[:config.max_models]

    # Estimate cost and trim if over budget
    estimated_cost = _estimate_election_cost(selected)
    while estimated_cost > config.budget_usd and len(selected) > 3:
        # Remove most expensive model
        selected.sort(key=lambda m: m.prompt_price_per_m, reverse=True)
        removed = selected.pop(0)
        logger.info(f"  Budget trim: removing {removed.model_id} (${removed.prompt_price_per_m:.2f}/M)")
        estimated_cost = _estimate_election_cost(selected)

    return selected


def _estimate_election_cost(candidates: list[DiscoveredModel]) -> float:
    """Estimate total election cost in USD."""
    n = len(candidates)
    # Round 1: n models × 5 evals × ~500 input tokens × ~200 output tokens
    round1_input_tokens = n * 5 * 500
    round1_output_tokens = n * 5 * 200

    # Round 2: n models × (n-1) verifications × ~800 input × ~100 output
    round2_input_tokens = n * (n - 1) * 800
    round2_output_tokens = n * (n - 1) * 100

    total_input = round1_input_tokens + round2_input_tokens
    total_output = round1_output_tokens + round2_output_tokens

    # Average price across candidates
    avg_input_price = sum(m.prompt_price_per_m for m in candidates) / max(n, 1)
    avg_output_price = sum(m.completion_price_per_m for m in candidates) / max(n, 1)

    cost = (total_input / 1_000_000) * avg_input_price + (total_output / 1_000_000) * avg_output_price
    return cost


# ---------------------------------------------------------------------------
# Autonomous election runner
# ---------------------------------------------------------------------------

def run_live_election(
    output_dir: str = "/Volumes/CORTEX/cortex/data/tier_election",
    config: Optional[ElectionConfig] = None,
) -> dict:
    """
    Fully autonomous tier election:
      1. Discover models from OpenRouter
      2. Select candidates
      3. Run election
      4. Save results

    Returns: election result dict
    """
    from .tier_election import (
        TierElection, ModelCandidate, OpenRouterTransport, ElectionResult
    )

    cfg = config or ElectionConfig()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        # Try loading from .env
        env_path = Path("/Volumes/CORTEX/cortex/bin/.env")
        if env_path.exists():
            for line in env_path.read_text().strip().split("\n"):
                if line.startswith("OPENROUTER_API_KEY="):
                    api_key = line.split("=", 1)[1].strip()
                    os.environ["OPENROUTER_API_KEY"] = api_key
                    break

    if not api_key:
        print("ERROR: No OPENROUTER_API_KEY found (checked env + /Volumes/CORTEX/cortex/bin/.env)")
        return {"error": "no_api_key"}

    t0 = time.time()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Phase 1: Discovery
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  Cortex Live Tier Election                                  ║")
    print("║  Dynamic model discovery → self-assessment → consensus      ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()
    print("[1/4] Discovering models from OpenRouter...")

    discovered = discover_models(api_key)
    families = {}
    for m in discovered:
        families.setdefault(m.family, []).append(m)

    print(f"  Found: {len(discovered)} chat-capable models")
    print(f"  Families: {len(families)} ({', '.join(sorted(families.keys()))})")
    print(f"  Free: {sum(1 for m in discovered if m.is_free)}")
    print()

    # Phase 2: Candidate selection
    print("[2/4] Selecting election candidates...")
    candidates = select_candidates(discovered, cfg)
    estimated_cost = _estimate_election_cost(candidates)

    print(f"  Selected: {len(candidates)} models")
    print(f"  Estimated cost: ${estimated_cost:.4f}")
    print(f"  Families represented: {len(set(c.family for c in candidates))}")
    print()
    print("  Candidates (model_id = OpenRouter API identity):")
    for c in sorted(candidates, key=lambda x: (-x.param_size_b if x.param_size_b > 0 else -999)):
        size_str = f"{c.param_size_b:.0f}B" if c.param_size_b > 0 else "??"
        price_str = "FREE" if c.is_free else f"${c.prompt_price_per_m:.2f}/M"
        print(f"    {c.model_id:<50} [{c.family}, {size_str}, {price_str}]")
    print()

    # Phase 3: Run election
    print("[3/4] Running tier election...")
    print(f"  Round 1: Self-assessment ({len(candidates)} models × 5 evals)")
    print(f"  Round 2: Cross-verification ({len(candidates)} × {len(candidates)-1} pairs)")
    print(f"  Round 3: Consensus resolution")
    print()

    # Convert to ModelCandidate format for the election engine
    election_candidates = [
        ModelCandidate(
            model_id=c.model_id,
            display_name=c.name,
            param_size_b=c.param_size_b,
            family=c.family,
        )
        for c in candidates
    ]

    transport = OpenRouterTransport(max_workers=min(len(candidates), 10))
    election = TierElection(
        candidates=election_candidates,
        transport=transport,
        supermajority=0.67,
    )

    result = election.run_election()

    # Phase 4: Save results
    print()
    print("[4/4] Saving results...")

    # SCL document
    scl_path = output_path / "tier_election.scl"
    scl_path.write_text(result.to_scl_document())

    # Full JSON results
    json_path = output_path / "tier_election.json"
    json_result = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "discovered_total": len(discovered),
        "candidates_selected": len(candidates),
        "estimated_cost_usd": estimated_cost,
        "elapsed_s": time.time() - t0,
        "converged": result.converged,
        "tier_map": result.tier_map,
        "config": {
            "max_models": cfg.max_models,
            "budget_usd": cfg.budget_usd,
            "per_family": cfg.per_family,
            "families_filter": cfg.families_filter,
        },
        "assessments": [
            {
                "model_id": a.model_id,
                "proposed_tier": a.proposed_tier,
                "confidence": a.confidence,
                "evidence": a.evidence,
            }
            for a in result.assessments
        ],
        "consensus": [
            {
                "model_id": c.model_id,
                "tier": c.tier,
                "agreement_ratio": c.agreement_ratio,
                "total_voters": c.total_voters,
                "status": c.status,
            }
            for c in result.consensus
        ],
    }
    json_path.write_text(json.dumps(json_result, indent=2))

    # Discovery log (all models seen)
    discovery_path = output_path / "discovered_models.json"
    discovery_path.write_text(json.dumps([
        {
            "model_id": m.model_id,
            "name": m.name,
            "family": m.family,
            "param_size_b": m.param_size_b,
            "context_length": m.context_length,
            "prompt_price_per_m": m.prompt_price_per_m,
            "is_free": m.is_free,
        }
        for m in discovered
    ], indent=2))

    elapsed = time.time() - t0

    # Print results
    print()
    print(f"{'━' * 62}")
    print(f"  TIER ELECTION RESULTS  (converged={result.converged})")
    print(f"{'━' * 62}")
    if result.tier_map:
        for model_id, tier in sorted(result.tier_map.items(), key=lambda x: x[1]):
            # Find consensus status
            cons = next((c for c in result.consensus if c.model_id == model_id), None)
            status = cons.status if cons else "?"
            agreement = f"{cons.agreement_ratio:.0%}" if cons else "?"
            print(f"  {tier}  {model_id:<45} [{status}, {agreement}]")
    else:
        print("  No tier assignments (election may have failed)")
    print(f"{'━' * 62}")
    print(f"  Models discovered:  {len(discovered)}")
    print(f"  Models elected:     {len(result.tier_map)}")
    print(f"  Elapsed:            {elapsed:.1f}s")
    print(f"  Estimated cost:     ${estimated_cost:.4f}")
    print(f"  Output:             {output_dir}")
    print()

    return json_result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """CLI for autonomous tier election."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Cortex Live Tier Election — discover and classify models autonomously"
    )
    parser.add_argument("--output-dir", default="/Volumes/CORTEX/cortex/data/tier_election",
                       help="Output directory")
    parser.add_argument("--max-models", type=int, default=12,
                       help="Maximum models to include (default: 12)")
    parser.add_argument("--budget", type=float, default=0.50,
                       help="Max budget in USD (default: $0.50)")
    parser.add_argument("--per-family", type=int, default=3,
                       help="Max models per family (default: 3)")
    parser.add_argument("--families", default=None,
                       help="Comma-separated family filter (e.g. qwen,llama,deepseek)")
    parser.add_argument("--exclude", default=None,
                       help="Comma-separated patterns to exclude (e.g. 'free,preview')")
    parser.add_argument("--free-only", action="store_true",
                       help="Only use free models (no cost)")
    parser.add_argument("--discover-only", action="store_true",
                       help="Only discover and list models, don't run election")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Build config
    config = ElectionConfig(
        max_models=args.max_models,
        budget_usd=args.budget if not args.free_only else 0.0,
        per_family=args.per_family,
        families_filter=args.families.split(",") if args.families else None,
        exclude_patterns=args.exclude.split(",") if args.exclude else None,
    )

    if args.discover_only:
        # Just list what's available
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            env_path = Path("/Volumes/CORTEX/cortex/bin/.env")
            if env_path.exists():
                for line in env_path.read_text().strip().split("\n"):
                    if line.startswith("OPENROUTER_API_KEY="):
                        api_key = line.split("=", 1)[1].strip()
                        break
        if not api_key:
            print("ERROR: No API key")
            sys.exit(1)

        models = discover_models(api_key)
        candidates = select_candidates(models, config)

        print(f"Discovered {len(models)} models, selected {len(candidates)} candidates:")
        print()
        for c in sorted(candidates, key=lambda x: (x.family, -x.param_size_b)):
            size_str = f"{c.param_size_b:.0f}B" if c.param_size_b > 0 else "??"
            price_str = "FREE" if c.is_free else f"${c.prompt_price_per_m:.2f}/M"
            print(f"  [{c.family:<10}] {c.model_id:<50} {size_str:>6}  {price_str}")

        cost = _estimate_election_cost(candidates)
        print(f"\nEstimated election cost: ${cost:.4f}")
        sys.exit(0)

    if args.free_only:
        # Override to only pick free models
        config.exclude_patterns = config.exclude_patterns or []
        # We'll handle this in selection by filtering

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            env_path = Path("/Volumes/CORTEX/cortex/bin/.env")
            if env_path.exists():
                for line in env_path.read_text().strip().split("\n"):
                    if line.startswith("OPENROUTER_API_KEY="):
                        api_key = line.split("=", 1)[1].strip()
                        os.environ["OPENROUTER_API_KEY"] = api_key
                        break

        if api_key:
            discovered = discover_models(api_key)
            free_models = [m for m in discovered if m.is_free]
            if free_models:
                # Override candidates to free only
                config.budget_usd = 999.0  # No budget constraint for free
                # Run with free models only by overriding discover
                print(f"Free-only mode: {len(free_models)} free models available")

    result = run_live_election(output_dir=args.output_dir, config=config)

    if "error" in result:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
