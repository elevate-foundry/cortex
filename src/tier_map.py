"""
Cortex TierMap — loads tier election results and provides model recommendations.

The TierMap bridges the tier election protocol (which discovers model capabilities)
with the routing system (which needs to pick the right model for each task tier).

Usage:
    from .tier_map import TierMap
    tm = TierMap.load()          # loads latest election result
    model = tm.best_for(Tier.L4) # → "openai/gpt-5.6-luna-pro"
    tier = tm.tier_of("qwen/qwen3-coder")  # → Tier.L5
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .tiers import Tier

logger = logging.getLogger("cortex.tier_map")

# Default locations for election data
ELECTION_PATHS = [
    Path("/Volumes/CORTEX/cortex/data/tier_election/tier_election.json"),
    Path("data/tier_election/tier_election.json"),
]


@dataclass
class ModelAssessment:
    """Assessment of a single model from the election."""
    model_id: str
    tier: Tier
    confidence: float
    agreement_ratio: float = 0.0
    status: str = ""  # "confirmed", "disputed", "abstained"
    evidence: dict = field(default_factory=dict)


@dataclass
class TierMap:
    """
    Maps model IDs to validated tiers based on election results.
    
    Provides:
    - tier_of(model_id) → Tier: what tier a model belongs to
    - best_for(tier) → model_id: recommended model for a tier
    - models_at(tier) → list[str]: all models validated at a tier
    """
    
    # model_id → Tier
    _model_to_tier: dict[str, Tier] = field(default_factory=dict)
    # Tier → list of model_ids (ordered by confidence)
    _tier_to_models: dict[Tier, list[str]] = field(default_factory=dict)
    # Full assessment data
    _assessments: dict[str, ModelAssessment] = field(default_factory=dict)
    # Metadata
    timestamp: str = ""
    source_path: str = ""

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "TierMap":
        """
        Load the latest tier election results.
        
        Tries the provided path, then default locations.
        Returns an empty TierMap if no election data is found.
        """
        paths_to_try = [path] if path else ELECTION_PATHS
        
        for p in paths_to_try:
            if p is None:
                continue
            if p.exists():
                try:
                    return cls._from_file(p)
                except Exception as e:
                    logger.warning("Failed to load TierMap from %s: %s", p, e)
        
        logger.info("No tier election data found — using empty TierMap")
        return cls()

    @classmethod
    def _from_file(cls, path: Path) -> "TierMap":
        """Parse election JSON into a TierMap."""
        data = json.loads(path.read_text())
        
        tm = cls(
            timestamp=data.get("timestamp", ""),
            source_path=str(path),
        )
        
        # Load tier_map (the consensus result)
        tier_map_raw = data.get("tier_map", {})
        for model_id, tier_str in tier_map_raw.items():
            try:
                tier = Tier[tier_str]
                tm._model_to_tier[model_id] = tier
                if tier not in tm._tier_to_models:
                    tm._tier_to_models[tier] = []
                tm._tier_to_models[tier].append(model_id)
            except KeyError:
                logger.debug("Unknown tier %s for model %s", tier_str, model_id)

        # Load assessments for confidence data
        for assessment in data.get("assessments", []):
            model_id = assessment.get("model_id", "")
            proposed_str = assessment.get("proposed_tier", "")
            try:
                proposed_tier = Tier[proposed_str]
            except KeyError:
                proposed_tier = Tier.L0
            
            tm._assessments[model_id] = ModelAssessment(
                model_id=model_id,
                tier=tm._model_to_tier.get(model_id, proposed_tier),
                confidence=assessment.get("confidence", 0.0),
                evidence=assessment.get("evidence", {}),
            )
        
        # Load consensus for agreement data
        for consensus in data.get("consensus", []):
            model_id = consensus.get("model_id", "")
            if model_id in tm._assessments:
                tm._assessments[model_id].agreement_ratio = consensus.get("agreement_ratio", 0.0)
                tm._assessments[model_id].status = consensus.get("status", "")

        # Sort tier_to_models by confidence (highest first)
        for tier in tm._tier_to_models:
            tm._tier_to_models[tier].sort(
                key=lambda m: tm._assessments.get(m, ModelAssessment(m, tier, 0.0)).confidence,
                reverse=True,
            )
        
        logger.info(
            "TierMap loaded: %d models, %d tiers (from %s)",
            len(tm._model_to_tier), len(tm._tier_to_models), path.name,
        )
        return tm

    def tier_of(self, model_id: str) -> Optional[Tier]:
        """Get the validated tier for a model, or None if unknown."""
        return self._model_to_tier.get(model_id)

    def best_for(self, tier: Tier) -> Optional[str]:
        """Get the best (highest confidence) model for a tier."""
        models = self._tier_to_models.get(tier, [])
        return models[0] if models else None

    def models_at(self, tier: Tier) -> list[str]:
        """Get all models validated at a given tier."""
        return self._tier_to_models.get(tier, [])

    def all_models(self) -> dict[str, Tier]:
        """Get the full model→tier mapping."""
        return dict(self._model_to_tier)

    def assessment(self, model_id: str) -> Optional[ModelAssessment]:
        """Get full assessment data for a model."""
        return self._assessments.get(model_id)

    @property
    def empty(self) -> bool:
        """Whether this TierMap has any data."""
        return len(self._model_to_tier) == 0

    def summary(self) -> str:
        """Human-readable summary of the tier map."""
        if self.empty:
            return "TierMap: empty (no election data)"
        lines = [f"TierMap ({len(self._model_to_tier)} models, {self.timestamp}):"]
        for tier in sorted(self._tier_to_models.keys(), key=lambda t: t.value):
            models = self._tier_to_models[tier]
            model_strs = [f"{m}({self._assessments.get(m, ModelAssessment(m, tier, 0.0)).confidence:.0%})" for m in models]
            lines.append(f"  {tier.name}: {', '.join(model_strs)}")
        return "\n".join(lines)

    def cloud_model_for_tier(self, tier: Tier, fallback: str = "") -> str:
        """
        Get the best cloud model for a given tier.
        
        Uses election data when available, falls back to the provided default.
        Prioritizes confirmed models over disputed/abstained.
        """
        models = self._tier_to_models.get(tier, [])
        
        # Prefer confirmed models
        for model_id in models:
            assessment = self._assessments.get(model_id)
            if assessment and assessment.status == "confirmed":
                return model_id
        
        # Accept any model at this tier
        if models:
            return models[0]
        
        # Try adjacent tiers (prefer one tier down for safety)
        if tier.value > 0:
            lower = Tier(tier.value - 1)
            lower_models = self._tier_to_models.get(lower, [])
            if lower_models:
                return lower_models[0]
        
        return fallback
