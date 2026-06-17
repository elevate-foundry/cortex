"""
Model Lifecycle Manager — the systemd for models.

Manages the lifecycle of inference models: loading, unloading, health checks,
and VRAM-aware scheduling. Treats L0-L2 as always-hot services and L3+ as
on-demand daemons that are loaded/evicted based on need and available memory.

Responsibilities:
  - Boot sequence: load L0 first, then L1, L2
  - On-demand loading: load higher tiers when the router requests them
  - VRAM accounting: track what's loaded, enforce budget
  - Eviction: unload lowest-priority on-demand tier when VRAM is tight
  - Health monitoring: periodic checks, restart on failure
  - Adapter registry: map loaded models to BackendAdapter instances
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .backend_adapter import BackendAdapter, BackendType, ollama_adapter
from .tiers import (
    Tier,
    TierModel,
    TierSpec,
    TIER_SPECS,
    UNIVERSAL_CATALOG,
    CHALLENGE_CATALOG,
    get_models_for_tier,
    get_challenge_models,
    assess_tiers,
    concurrent_vram_budget,
)
from .config import OLLAMA_URL, vram_budget_mb
from .hardware_detect import SystemProfile

logger = logging.getLogger(__name__)


class ModelState(str, Enum):
    UNLOADED = "unloaded"
    LOADING = "loading"
    READY = "ready"
    FAILED = "failed"
    EVICTING = "evicting"


@dataclass
class LoadedModel:
    """Tracks a model that's currently loaded or being managed."""
    tier: Tier
    model: TierModel
    adapter: BackendAdapter
    state: ModelState = ModelState.UNLOADED
    loaded_at: float = 0.0
    last_used: float = 0.0
    last_health_check: float = 0.0
    request_count: int = 0
    is_challenge: bool = False      # True if this is a challenge model


@dataclass
class ManagerConfig:
    """Configuration for the model manager."""
    ollama_url: str = OLLAMA_URL
    llama_cpp_url: str = "http://localhost:8080"
    vllm_url: str = "http://localhost:8000"
    preferred_backend: BackendType = BackendType.OLLAMA
    health_check_interval_s: float = 30.0
    eviction_grace_period_s: float = 60.0   # don't evict if used recently
    auto_pull: bool = True                   # auto-pull models via Ollama


class ModelManager:
    """
    Manages the full model lifecycle.
    
    Like systemd: starts services (models), monitors them, restarts on failure,
    and manages resource budgets (VRAM instead of memory/CPU).
    """

    def __init__(
        self,
        profile: SystemProfile,
        config: Optional[ManagerConfig] = None,
    ):
        self.profile = profile
        self.config = config or ManagerConfig()
        self._loaded: dict[str, LoadedModel] = {}   # key: "{tier}:{model_id}"
        self._vram_used_mb: int = 0
        self._vram_budget_mb: int = self._compute_vram_budget()

    def _compute_vram_budget(self) -> int:
        """Compute total VRAM budget based on hardware."""
        from .hardware_detect import AcceleratorType
        is_unified = self.profile.primary_accelerator == AcceleratorType.APPLE_METAL
        has_gpu = bool(self.profile.gpus)
        total = self.profile.total_vram_mb if has_gpu and not is_unified else self.profile.memory.total_mb
        return vram_budget_mb(total, has_gpu=has_gpu, is_unified=is_unified)

    def _model_key(self, tier: Tier, model: TierModel) -> str:
        return f"{tier.name}:{model.model_id}"

    def _make_adapter(self, model: TierModel) -> BackendAdapter:
        """Create a BackendAdapter for a model based on its format."""
        tag = model.ollama_tag or model.model_id

        if model.format == "api":
            return BackendAdapter(
                backend=BackendType.OPENAI_API,
                base_url="https://api.openai.com",
                default_model=model.model_id,
            )

        if self.config.preferred_backend == BackendType.OLLAMA:
            return ollama_adapter(model=tag, base_url=self.config.ollama_url)

        if self.config.preferred_backend == BackendType.LLAMA_CPP:
            return BackendAdapter(
                backend=BackendType.LLAMA_CPP,
                base_url=self.config.llama_cpp_url,
                default_model=model.model_id,
            )

        if self.config.preferred_backend == BackendType.VLLM:
            return BackendAdapter(
                backend=BackendType.VLLM,
                base_url=self.config.vllm_url,
                default_model=model.model_id,
            )

        # Default to Ollama
        return ollama_adapter(model=tag, base_url=self.config.ollama_url)

    # ------------------------------------------------------------------
    # Boot sequence
    # ------------------------------------------------------------------

    def boot(self) -> list[LoadedModel]:
        """
        Execute the boot sequence: load all always-hot tiers (L0, L1, L2).
        Returns list of successfully loaded models.
        """
        logger.info("=== Model Manager: Boot Sequence ===")
        booted: list[LoadedModel] = []

        for tier in (Tier.L0, Tier.L1, Tier.L2):
            spec = TIER_SPECS.get(tier)
            if not spec or not spec.always_hot:
                continue

            models = get_models_for_tier(tier, self.profile)
            if not models:
                logger.warning(f"No models available for {tier.name}, skipping")
                continue

            model = models[0]
            loaded = self.load_model(tier, model)
            if loaded and loaded.state == ModelState.READY:
                booted.append(loaded)

        logger.info(
            f"Boot complete: {len(booted)} models loaded, "
            f"{self._vram_used_mb}MB / {self._vram_budget_mb}MB VRAM"
        )
        return booted

    # ------------------------------------------------------------------
    # Load / Unload
    # ------------------------------------------------------------------

    def load_model(
        self,
        tier: Tier,
        model: TierModel,
        is_challenge: bool = False,
    ) -> Optional[LoadedModel]:
        """
        Load a model. Handles VRAM budget enforcement and eviction.
        Returns the LoadedModel on success, None on failure.
        """
        key = self._model_key(tier, model)

        # Already loaded?
        if key in self._loaded and self._loaded[key].state == ModelState.READY:
            self._loaded[key].last_used = time.monotonic()
            return self._loaded[key]

        # Check VRAM budget
        if model.vram_mb > 0 and (self._vram_used_mb + model.vram_mb) > self._vram_budget_mb:
            freed = self._evict_for_space(model.vram_mb)
            if not freed:
                logger.error(
                    f"Cannot load {model.model_id} ({model.vram_mb}MB): "
                    f"insufficient VRAM ({self._vram_used_mb}/{self._vram_budget_mb}MB)"
                )
                return None

        # Create adapter
        adapter = self._make_adapter(model)

        loaded = LoadedModel(
            tier=tier,
            model=model,
            adapter=adapter,
            state=ModelState.LOADING,
            loaded_at=time.monotonic(),
            last_used=time.monotonic(),
            is_challenge=is_challenge,
        )
        self._loaded[key] = loaded

        # For Ollama: pull the model if auto_pull is enabled
        if (self.config.auto_pull
                and self.config.preferred_backend == BackendType.OLLAMA
                and model.format == "gguf"):
            tag = model.ollama_tag or model.model_id
            logger.info(f"Pulling {tag} via Ollama...")
            if not adapter.pull_model_sync(tag):
                logger.warning(f"Failed to pull {tag}, may already be available")

        # Verify the backend is reachable
        if adapter.health_check():
            loaded.state = ModelState.READY
            loaded.last_health_check = time.monotonic()
            self._vram_used_mb += model.vram_mb
            logger.info(
                f"Loaded {tier.name}: {model.model_id} "
                f"({model.vram_mb}MB, {self._vram_used_mb}/{self._vram_budget_mb}MB total)"
            )
        else:
            loaded.state = ModelState.FAILED
            logger.error(f"Health check failed for {model.model_id} — backend not reachable")

        return loaded

    def unload_model(self, tier: Tier, model: TierModel) -> bool:
        """Unload a model and free its VRAM."""
        key = self._model_key(tier, model)
        loaded = self._loaded.get(key)
        if not loaded:
            return False

        loaded.state = ModelState.EVICTING
        self._vram_used_mb = max(0, self._vram_used_mb - model.vram_mb)
        del self._loaded[key]
        logger.info(
            f"Unloaded {tier.name}: {model.model_id} "
            f"(freed {model.vram_mb}MB, {self._vram_used_mb}/{self._vram_budget_mb}MB)"
        )
        return True

    def _evict_for_space(self, needed_mb: int) -> bool:
        """
        Evict on-demand models (not always-hot) to free space.
        Returns True if enough space was freed.
        """
        now = time.monotonic()
        grace = self.config.eviction_grace_period_s

        # Candidates: not always-hot, not recently used, sorted by last_used (oldest first)
        candidates = [
            lm for lm in self._loaded.values()
            if (lm.state == ModelState.READY
                and not TIER_SPECS[lm.tier].always_hot
                and (now - lm.last_used) > grace)
        ]
        candidates.sort(key=lambda lm: lm.last_used)

        freed = 0
        for lm in candidates:
            if (self._vram_used_mb + needed_mb - freed) <= self._vram_budget_mb:
                return True
            logger.info(f"Evicting {lm.tier.name}: {lm.model.model_id} (idle {now - lm.last_used:.0f}s)")
            self.unload_model(lm.tier, lm.model)
            freed += lm.model.vram_mb

        return (self._vram_used_mb + needed_mb) <= self._vram_budget_mb

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get_adapter(self, tier: Tier) -> Optional[BackendAdapter]:
        """Get the adapter for a loaded core model at a tier."""
        for key, lm in self._loaded.items():
            if lm.tier == tier and lm.state == ModelState.READY and not lm.is_challenge:
                lm.last_used = time.monotonic()
                lm.request_count += 1
                return lm.adapter
        return None

    def get_challenge_adapter(
        self,
        tier: Tier,
        exclude_family: str = "qwen",
    ) -> Optional[BackendAdapter]:
        """
        Get an adapter for a challenge model at a tier.
        Loads one on-demand if not already loaded.
        """
        # Check if a challenge model is already loaded at this tier
        for key, lm in self._loaded.items():
            if (lm.tier == tier
                    and lm.state == ModelState.READY
                    and lm.is_challenge
                    and lm.model.family != exclude_family):
                lm.last_used = time.monotonic()
                lm.request_count += 1
                return lm.adapter

        # Load one on-demand
        challengers = get_challenge_models(tier, exclude_family)
        if not challengers:
            return None

        # Pick the first one that fits
        for model in challengers:
            loaded = self.load_model(tier, model, is_challenge=True)
            if loaded and loaded.state == ModelState.READY:
                return loaded.adapter

        return None

    def get_all_adapters_for_tier(
        self,
        tier: Tier,
    ) -> list[tuple[TierModel, BackendAdapter]]:
        """Get all loaded adapters (core + challenge) for a tier."""
        results = []
        for key, lm in self._loaded.items():
            if lm.tier == tier and lm.state == ModelState.READY:
                lm.last_used = time.monotonic()
                results.append((lm.model, lm.adapter))
        return results

    # ------------------------------------------------------------------
    # Health monitoring
    # ------------------------------------------------------------------

    def health_check_all(self) -> dict[str, bool]:
        """Run health checks on all loaded models."""
        results = {}
        for key, lm in list(self._loaded.items()):
            if lm.state != ModelState.READY:
                continue

            healthy = lm.adapter.health_check()
            results[key] = healthy

            if healthy:
                lm.last_health_check = time.monotonic()
            else:
                logger.warning(f"Health check FAILED: {key}")
                lm.state = ModelState.FAILED

        return results

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> dict:
        """Return current manager status."""
        models = []
        for key, lm in self._loaded.items():
            models.append({
                "key": key,
                "tier": lm.tier.name,
                "model_id": lm.model.model_id,
                "family": lm.model.family,
                "state": lm.state.value,
                "vram_mb": lm.model.vram_mb,
                "is_challenge": lm.is_challenge,
                "request_count": lm.request_count,
                "uptime_s": round(time.monotonic() - lm.loaded_at, 1) if lm.loaded_at else 0,
            })

        return {
            "vram_used_mb": self._vram_used_mb,
            "vram_budget_mb": self._vram_budget_mb,
            "vram_remaining_mb": self._vram_budget_mb - self._vram_used_mb,
            "models_loaded": len([m for m in models if m["state"] == "ready"]),
            "models": models,
        }

    def __repr__(self) -> str:
        ready = sum(1 for lm in self._loaded.values() if lm.state == ModelState.READY)
        return (
            f"ModelManager({ready} models ready, "
            f"{self._vram_used_mb}/{self._vram_budget_mb}MB VRAM)"
        )
