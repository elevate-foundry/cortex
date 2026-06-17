"""
Swarm — cross-family fan-out for hard and hardest problems.

When the challenger detects disagreement, the swarm fans out the same
prompt to multiple models across different families and aggregates their
answers via majority vote or LLM-as-judge.

Difficulty tiers:
  - Hard:    3-5 models across multiple families → majority vote
  - Hardest: all available models across all families → weighted consensus

Weighting:
  - Larger models get more weight (they're generally more capable)
  - Cross-family agreement gets a bonus (independent validation)
  - Models that frequently agree with the eventual consensus get
    a credibility boost over time (future: persistent scoring)
"""

import asyncio
import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .backend_adapter import (
    BackendAdapter,
    CompletionRequest,
    CompletionResponse,
)
from .challenger import compare_answers, AgreementLevel, _normalize
from .model_manager import ModelManager
from .tiers import (
    Tier,
    TierModel,
    TIER_SPECS,
    get_challenge_models,
    get_all_models_for_tier,
)

logger = logging.getLogger(__name__)


class SwarmSize(str, Enum):
    SMALL = "small"     # 3-5 models (Hard)
    LARGE = "large"     # all available (Hardest)


class AggregationMethod(str, Enum):
    MAJORITY_VOTE = "majority_vote"
    WEIGHTED_VOTE = "weighted_vote"
    LLM_JUDGE = "llm_judge"


@dataclass
class SwarmVote:
    """A single model's response in the swarm."""
    model_id: str
    family: str
    content: str
    weight: float           # based on model size
    response_ms: float
    tier: Tier
    cluster_id: int = -1    # assigned during clustering


@dataclass
class SwarmResult:
    """Aggregated result from a swarm query."""
    consensus_answer: str
    confidence: float               # 0.0 - 1.0
    agreement_ratio: float          # fraction of models in the majority cluster
    num_models: int
    num_families: int               # distinct families that participated
    num_agreeing_families: int      # distinct families in the majority cluster
    votes: list[SwarmVote]
    clusters: list[list[SwarmVote]] # grouped by agreement
    method: AggregationMethod
    total_ms: float                 # wall-clock time for entire swarm
    reason: str = ""


def _model_weight(tier: Tier, model: TierModel) -> float:
    """
    Compute a weight for a model based on its size.
    Larger models get more weight in voting.
    """
    spec = TIER_SPECS.get(tier)
    if not spec:
        return 1.0

    # Weight by parameter count midpoint (in billions)
    param_mid = (spec.param_min_b + spec.param_max_b) / 2
    if param_mid <= 1:
        return 1.0
    elif param_mid <= 4:
        return 1.5
    elif param_mid <= 8:
        return 2.0
    elif param_mid <= 14:
        return 2.5
    elif param_mid <= 32:
        return 3.0
    elif param_mid <= 70:
        return 4.0
    else:
        return 5.0


def _cluster_votes(votes: list[SwarmVote]) -> list[list[SwarmVote]]:
    """
    Group votes into agreement clusters.
    Two votes are in the same cluster if they have STRONG_AGREE or WEAK_AGREE.
    Simple greedy clustering: assign each vote to the first matching cluster.
    """
    clusters: list[list[SwarmVote]] = []

    for vote in votes:
        placed = False
        for cluster in clusters:
            # Compare against the first vote in the cluster (representative)
            agreement, _, _ = compare_answers(cluster[0].content, vote.content)
            if agreement in (AgreementLevel.STRONG_AGREE, AgreementLevel.WEAK_AGREE):
                vote.cluster_id = cluster[0].cluster_id
                cluster.append(vote)
                placed = True
                break

        if not placed:
            vote.cluster_id = len(clusters)
            clusters.append([vote])

    return clusters


def _majority_vote(clusters: list[list[SwarmVote]]) -> tuple[list[SwarmVote], float]:
    """
    Simple majority vote: the largest cluster wins.
    Returns (winning_cluster, agreement_ratio).
    """
    if not clusters:
        return [], 0.0

    total = sum(len(c) for c in clusters)
    clusters.sort(key=len, reverse=True)
    winner = clusters[0]
    ratio = len(winner) / total if total > 0 else 0.0
    return winner, ratio


def _weighted_vote(clusters: list[list[SwarmVote]]) -> tuple[list[SwarmVote], float]:
    """
    Weighted vote: sum weights per cluster, highest total weight wins.
    Returns (winning_cluster, weighted_agreement_ratio).
    """
    if not clusters:
        return [], 0.0

    total_weight = sum(v.weight for c in clusters for v in c)
    cluster_weights = [(c, sum(v.weight for v in c)) for c in clusters]
    cluster_weights.sort(key=lambda x: x[1], reverse=True)

    winner, winner_weight = cluster_weights[0]
    ratio = winner_weight / total_weight if total_weight > 0 else 0.0
    return winner, ratio


class Swarm:
    """
    Orchestrates swarm queries across multiple models and families.
    """

    def __init__(
        self,
        manager: ModelManager,
        default_method: AggregationMethod = AggregationMethod.WEIGHTED_VOTE,
    ):
        self.manager = manager
        self.default_method = default_method

    def query_sync(
        self,
        messages: list[dict],
        tier: Tier,
        size: SwarmSize = SwarmSize.SMALL,
        max_tokens: int = 512,
        method: Optional[AggregationMethod] = None,
    ) -> SwarmResult:
        """
        Synchronous swarm query. Queries models sequentially.
        Use `query` for async parallel execution.
        """
        method = method or self.default_method
        t0 = time.monotonic()

        # Collect adapters
        adapters = self._collect_adapters(tier, size)

        if not adapters:
            return SwarmResult(
                consensus_answer="",
                confidence=0.0,
                agreement_ratio=0.0,
                num_models=0,
                num_families=0,
                num_agreeing_families=0,
                votes=[],
                clusters=[],
                method=method,
                total_ms=0.0,
                reason="No models available for swarm",
            )

        # Query all models
        votes: list[SwarmVote] = []
        for model, adapter in adapters:
            req = CompletionRequest(
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.0,
            )
            try:
                resp = adapter.complete_sync(req)
                weight = _model_weight(tier, model)
                votes.append(SwarmVote(
                    model_id=model.model_id,
                    family=model.family,
                    content=resp.content,
                    weight=weight,
                    response_ms=resp.total_ms,
                    tier=tier,
                ))
            except Exception as e:
                logger.warning(f"Swarm member {model.model_id} failed: {e}")

        return self._aggregate(votes, method, time.monotonic() - t0)

    async def query(
        self,
        messages: list[dict],
        tier: Tier,
        size: SwarmSize = SwarmSize.SMALL,
        max_tokens: int = 512,
        method: Optional[AggregationMethod] = None,
    ) -> SwarmResult:
        """
        Async swarm query. Queries all models in parallel.
        """
        method = method or self.default_method
        t0 = time.monotonic()

        adapters = self._collect_adapters(tier, size)

        if not adapters:
            return SwarmResult(
                consensus_answer="",
                confidence=0.0,
                agreement_ratio=0.0,
                num_models=0,
                num_families=0,
                num_agreeing_families=0,
                votes=[],
                clusters=[],
                method=method,
                total_ms=0.0,
                reason="No models available for swarm",
            )

        # Fan out queries in parallel
        async def _query_one(model: TierModel, adapter: BackendAdapter) -> Optional[SwarmVote]:
            req = CompletionRequest(
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.0,
            )
            try:
                resp = await adapter.complete(req)
                weight = _model_weight(tier, model)
                return SwarmVote(
                    model_id=model.model_id,
                    family=model.family,
                    content=resp.content,
                    weight=weight,
                    response_ms=resp.total_ms,
                    tier=tier,
                )
            except Exception as e:
                logger.warning(f"Swarm member {model.model_id} failed: {e}")
                return None

        tasks = [_query_one(model, adapter) for model, adapter in adapters]
        results = await asyncio.gather(*tasks)
        votes = [v for v in results if v is not None]

        return self._aggregate(votes, method, time.monotonic() - t0)

    def _collect_adapters(
        self,
        tier: Tier,
        size: SwarmSize,
    ) -> list[tuple[TierModel, BackendAdapter]]:
        """
        Collect model/adapter pairs for the swarm.
        
        SMALL: core model + up to 4 challenge models (3-5 total)
        LARGE: everything available at this tier
        """
        adapters: list[tuple[TierModel, BackendAdapter]] = []

        # Always include the core model if loaded
        core_adapters = self.manager.get_all_adapters_for_tier(tier)
        adapters.extend(core_adapters)

        # Load challenge models
        challengers = get_challenge_models(tier, exclude_family="qwen")

        if size == SwarmSize.SMALL:
            # Pick up to 4 challenge models from different families
            seen_families = {m.family for m, _ in adapters}
            for model in challengers:
                if len(adapters) >= 5:
                    break
                if model.family in seen_families:
                    continue
                loaded = self.manager.load_model(tier, model, is_challenge=True)
                if loaded and loaded.adapter:
                    adapters.append((model, loaded.adapter))
                    seen_families.add(model.family)

        elif size == SwarmSize.LARGE:
            # Load ALL available challenge models
            for model in challengers:
                loaded = self.manager.load_model(tier, model, is_challenge=True)
                if loaded and loaded.adapter:
                    adapters.append((model, loaded.adapter))

        logger.info(
            f"Swarm [{tier.name}] {size.value}: "
            f"{len(adapters)} models from "
            f"{len(set(m.family for m, _ in adapters))} families"
        )

        return adapters

    def _aggregate(
        self,
        votes: list[SwarmVote],
        method: AggregationMethod,
        elapsed_s: float,
    ) -> SwarmResult:
        """Aggregate votes into a consensus."""
        total_ms = elapsed_s * 1000

        if not votes:
            return SwarmResult(
                consensus_answer="",
                confidence=0.0,
                agreement_ratio=0.0,
                num_models=0,
                num_families=0,
                num_agreeing_families=0,
                votes=[],
                clusters=[],
                method=method,
                total_ms=total_ms,
                reason="No votes collected",
            )

        all_families = set(v.family for v in votes)

        # Cluster votes by agreement
        clusters = _cluster_votes(votes)

        # Select winner based on method
        if method == AggregationMethod.MAJORITY_VOTE:
            winner, ratio = _majority_vote(clusters)
        else:  # WEIGHTED_VOTE or LLM_JUDGE (LLM_JUDGE falls back to weighted for now)
            winner, ratio = _weighted_vote(clusters)

        if not winner:
            return SwarmResult(
                consensus_answer="",
                confidence=0.0,
                agreement_ratio=0.0,
                num_models=len(votes),
                num_families=len(all_families),
                num_agreeing_families=0,
                votes=votes,
                clusters=clusters,
                method=method,
                total_ms=total_ms,
                reason="No consensus reached",
            )

        # Pick the answer from the highest-weighted model in the winning cluster
        winner.sort(key=lambda v: v.weight, reverse=True)
        consensus_answer = winner[0].content

        agreeing_families = set(v.family for v in winner)

        # Confidence formula:
        #   base = agreement_ratio (what fraction of weight agrees)
        #   bonus for cross-family agreement
        #   penalty for small swarm
        base_confidence = ratio
        family_bonus = min(0.15, len(agreeing_families) * 0.05)
        size_penalty = max(0, 0.1 - len(votes) * 0.02)  # small swarms are less confident
        confidence = min(1.0, base_confidence + family_bonus - size_penalty)

        reason_parts = [
            f"{len(winner)}/{len(votes)} models agree",
            f"{len(agreeing_families)}/{len(all_families)} families agree",
            f"ratio={ratio:.2f}",
        ]

        result = SwarmResult(
            consensus_answer=consensus_answer,
            confidence=confidence,
            agreement_ratio=ratio,
            num_models=len(votes),
            num_families=len(all_families),
            num_agreeing_families=len(agreeing_families),
            votes=votes,
            clusters=clusters,
            method=method,
            total_ms=total_ms,
            reason=", ".join(reason_parts),
        )

        logger.info(
            f"Swarm result: confidence={confidence:.2f}, "
            f"{len(winner)}/{len(votes)} models agree across "
            f"{len(agreeing_families)} families ({total_ms:.0f}ms)"
        )

        return result
