"""
Challenger — cross-family confidence verification.

Given an answer from the core model, queries a challenge model from a
different family and compares outputs. If they agree, confidence increases.
If they disagree, signals the need for escalation (swarm).

This implements the "Medium" difficulty strategy from context.md:
  Core model answers → challenge model from a different family confirms.

Comparison strategies:
  1. Exact match (for structured outputs like JSON, yes/no)
  2. Semantic similarity (normalized text overlap)
  3. LLM-as-judge (use the core model to judge if answers agree)
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .backend_adapter import BackendAdapter, CompletionRequest, CompletionResponse
from .model_manager import ModelManager
from .tiers import Tier, TierModel

logger = logging.getLogger(__name__)


class AgreementLevel(str, Enum):
    STRONG_AGREE = "strong_agree"       # identical or near-identical answers
    WEAK_AGREE = "weak_agree"           # same conclusion, different phrasing
    AMBIGUOUS = "ambiguous"             # can't determine agreement
    DISAGREE = "disagree"               # different conclusions
    STRONG_DISAGREE = "strong_disagree" # contradictory answers


@dataclass
class ChallengeResult:
    """Result of challenging a core model's answer."""
    original_answer: str
    challenge_answer: str
    original_model: str
    challenge_model: str
    challenge_family: str
    agreement: AgreementLevel
    confidence: float                   # 0.0 - 1.0 composite confidence
    should_escalate: bool               # True if swarm is needed
    original_ms: float = 0.0
    challenge_ms: float = 0.0
    reason: str = ""


def _normalize(text: str) -> str:
    """Normalize text for comparison: lowercase, strip whitespace, collapse spaces."""
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    # Strip common prefixes like "Sure, ", "Of course, ", etc.
    text = re.sub(r'^(sure|of course|certainly|absolutely|yes|no)[,!.]?\s*', '', text)
    return text


def _extract_conclusion(text: str) -> str:
    """
    Extract the core conclusion from an answer.
    Tries to get the first substantive sentence.
    """
    text = text.strip()
    # If it's a short answer, use it all
    if len(text) < 200:
        return _normalize(text)

    # Try to find a concluding statement
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if sentences:
        return _normalize(sentences[0])
    return _normalize(text[:200])


def _token_overlap(a: str, b: str) -> float:
    """Compute Jaccard similarity between word sets."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


def _detect_yes_no(text: str) -> Optional[bool]:
    """Try to extract a yes/no answer."""
    text = text.strip().lower()
    if re.match(r'^(yes|true|correct|right|affirmative)\b', text):
        return True
    if re.match(r'^(no|false|incorrect|wrong|negative)\b', text):
        return False
    return None


def compare_answers(
    original: str,
    challenge: str,
) -> tuple[AgreementLevel, float, str]:
    """
    Compare two model outputs and determine agreement level.
    
    Returns (agreement_level, confidence_score, reason).
    """
    norm_orig = _normalize(original)
    norm_chal = _normalize(challenge)

    # Exact match after normalization
    if norm_orig == norm_chal:
        return AgreementLevel.STRONG_AGREE, 0.95, "exact match after normalization"

    # Yes/No agreement
    yn_orig = _detect_yes_no(original)
    yn_chal = _detect_yes_no(challenge)
    if yn_orig is not None and yn_chal is not None:
        if yn_orig == yn_chal:
            return AgreementLevel.STRONG_AGREE, 0.90, "yes/no agreement"
        else:
            return AgreementLevel.STRONG_DISAGREE, 0.15, "yes/no contradiction"

    # Token overlap
    overlap = _token_overlap(norm_orig, norm_chal)

    # Extract conclusions and compare
    conc_orig = _extract_conclusion(original)
    conc_chal = _extract_conclusion(challenge)
    conc_overlap = _token_overlap(conc_orig, conc_chal)

    # Weighted score: conclusion similarity matters more
    combined = conc_overlap * 0.6 + overlap * 0.4

    if combined > 0.7:
        return AgreementLevel.STRONG_AGREE, 0.85, f"high token overlap ({combined:.2f})"
    elif combined > 0.45:
        return AgreementLevel.WEAK_AGREE, 0.65, f"moderate token overlap ({combined:.2f})"
    elif combined > 0.25:
        return AgreementLevel.AMBIGUOUS, 0.45, f"low token overlap ({combined:.2f})"
    elif combined > 0.1:
        return AgreementLevel.DISAGREE, 0.25, f"very low overlap ({combined:.2f})"
    else:
        return AgreementLevel.STRONG_DISAGREE, 0.10, f"near-zero overlap ({combined:.2f})"


class Challenger:
    """
    Orchestrates the challenge process.
    
    Uses the ModelManager to get/load challenge models and compares
    their output against the core model's answer.
    """

    def __init__(
        self,
        manager: ModelManager,
        escalation_threshold: float = 0.5,
    ):
        self.manager = manager
        self.escalation_threshold = escalation_threshold

    def challenge(
        self,
        messages: list[dict],
        original_response: CompletionResponse,
        tier: Tier,
        max_tokens: int = 512,
    ) -> ChallengeResult:
        """
        Challenge a core model's answer with a model from a different family.
        
        Args:
            messages: Original conversation messages
            original_response: The core model's response
            tier: The tier to challenge at
            max_tokens: Max tokens for the challenge model
        """
        original_family = "qwen"  # Core ladder is always Qwen

        # Get a challenge model adapter
        adapter = self.manager.get_challenge_adapter(tier, exclude_family=original_family)

        if adapter is None:
            logger.warning(f"No challenge model available for {tier.name}")
            return ChallengeResult(
                original_answer=original_response.content,
                challenge_answer="",
                original_model=original_response.model,
                challenge_model="unavailable",
                challenge_family="none",
                agreement=AgreementLevel.AMBIGUOUS,
                confidence=0.5,
                should_escalate=False,
                original_ms=original_response.total_ms,
                reason="No challenge model available",
            )

        # Query the challenge model
        req = CompletionRequest(
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.0,
        )

        try:
            challenge_resp = adapter.complete_sync(req)
        except Exception as e:
            logger.error(f"Challenge model failed: {e}")
            return ChallengeResult(
                original_answer=original_response.content,
                challenge_answer="",
                original_model=original_response.model,
                challenge_model="error",
                challenge_family="none",
                agreement=AgreementLevel.AMBIGUOUS,
                confidence=0.5,
                should_escalate=False,
                original_ms=original_response.total_ms,
                reason=f"Challenge model error: {e}",
            )

        # Compare answers
        agreement, confidence, reason = compare_answers(
            original_response.content,
            challenge_resp.content,
        )

        should_escalate = confidence < self.escalation_threshold

        # Determine challenge model family from loaded models
        challenge_family = "unknown"
        for key, lm in self.manager._loaded.items():
            if lm.adapter is adapter:
                challenge_family = lm.model.family
                break

        result = ChallengeResult(
            original_answer=original_response.content,
            challenge_answer=challenge_resp.content,
            original_model=original_response.model,
            challenge_model=challenge_resp.model,
            challenge_family=challenge_family,
            agreement=agreement,
            confidence=confidence,
            should_escalate=should_escalate,
            original_ms=original_response.total_ms,
            challenge_ms=challenge_resp.total_ms,
            reason=reason,
        )

        logger.info(
            f"Challenge [{tier.name}]: {agreement.value} "
            f"(confidence={confidence:.2f}, escalate={should_escalate}) — {reason}"
        )

        return result
