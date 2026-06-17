"""
Cortex L0 Router — classifies incoming requests and routes to the appropriate tier.

The router itself runs as the L0 model (0.5B-1B) for minimal latency.
When no L0 model is loaded, falls back to rule-based heuristics.

Routing signals:
  - Task complexity (tool count, step count)
  - Required capabilities (coding, planning, safety)
  - Input length (longer inputs may need larger context)
  - Confidence threshold (escalate if model is uncertain)
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .tiers import Tier, TIER_SPECS


class TaskCategory(str, Enum):
    CLASSIFY = "classify"         # intent classification, yes/no
    TOOL_CALL = "tool_call"       # execute a single tool
    MULTI_TOOL = "multi_tool"     # execute multiple tools in sequence
    CODE = "code"                 # write or edit code
    DEBUG = "debug"               # debug/fix code
    PLAN = "plan"                 # multi-step planning
    ANALYZE = "analyze"           # deep analysis / reasoning
    GENERATE = "generate"         # long-form content generation
    SAFETY = "safety"             # safety-critical decision
    UNKNOWN = "unknown"


@dataclass
class RouteDecision:
    """The output of the router: which tier to use and why."""
    tier: Tier
    category: TaskCategory
    confidence: float       # 0.0 - 1.0
    reason: str
    escalation_hint: Optional[Tier] = None  # if confidence is low, suggest this tier


# ---------------------------------------------------------------------------
# Heuristic patterns for rule-based routing
# ---------------------------------------------------------------------------

# Keywords/patterns that suggest higher complexity
_CODE_PATTERNS = re.compile(
    r'\b(code|function|class|implement|refactor|bug|error|debug|fix|compile|test|unittest)\b',
    re.IGNORECASE,
)
_PLAN_PATTERNS = re.compile(
    r'\b(plan|strategy|design|architect|step.by.step|multi.?step|workflow|pipeline)\b',
    re.IGNORECASE,
)
_TOOL_PATTERNS = re.compile(
    r'\b(run|execute|call|open|create|delete|move|copy|search|find|list|send|fetch|api)\b',
    re.IGNORECASE,
)
_ANALYZE_PATTERNS = re.compile(
    r'\b(analyze|explain|compare|evaluate|assess|review|critique|reason|think|consider)\b',
    re.IGNORECASE,
)
_SIMPLE_PATTERNS = re.compile(
    r'\b(yes|no|true|false|classify|categorize|which|select|pick|choose|is it|does it)\b',
    re.IGNORECASE,
)
_SAFETY_PATTERNS = re.compile(
    r'\b(danger|unsafe|security|vulnerability|exploit|injection|sudo|rm -rf|drop table|password)\b',
    re.IGNORECASE,
)


def _count_tools_requested(prompt: str) -> int:
    """Estimate how many distinct tool calls a prompt needs."""
    tool_verbs = re.findall(
        r'\b(run|execute|call|open|create|delete|move|copy|search|find|send|fetch|read|write)\b',
        prompt,
        re.IGNORECASE,
    )
    return len(set(v.lower() for v in tool_verbs))


def _estimate_complexity(prompt: str) -> float:
    """
    Estimate task complexity on a 0-1 scale.
    Used to select the appropriate tier.
    """
    score = 0.0
    length = len(prompt)

    # Length-based scoring
    if length < 50:
        score += 0.0
    elif length < 200:
        score += 0.1
    elif length < 500:
        score += 0.2
    elif length < 2000:
        score += 0.3
    else:
        score += 0.4

    # Pattern-based scoring
    if _CODE_PATTERNS.search(prompt):
        score += 0.25
    if _PLAN_PATTERNS.search(prompt):
        score += 0.2
    if _ANALYZE_PATTERNS.search(prompt):
        score += 0.15
    if _TOOL_PATTERNS.search(prompt):
        score += 0.1

    # Multi-step indicators
    steps = len(re.findall(r'\b(then|after that|next|finally|also|and then)\b', prompt, re.IGNORECASE))
    score += min(steps * 0.05, 0.2)

    # Tool count
    tools = _count_tools_requested(prompt)
    score += min(tools * 0.05, 0.15)

    return min(score, 1.0)


def _categorize(prompt: str) -> TaskCategory:
    """Classify the task category from the prompt."""
    if _SAFETY_PATTERNS.search(prompt):
        return TaskCategory.SAFETY

    prompt_lower = prompt.lower().strip()

    # Very short / yes-no
    if len(prompt_lower) < 30 and _SIMPLE_PATTERNS.search(prompt):
        return TaskCategory.CLASSIFY

    if _CODE_PATTERNS.search(prompt) and _PLAN_PATTERNS.search(prompt):
        return TaskCategory.DEBUG  # code + planning = debug/complex
    if _CODE_PATTERNS.search(prompt):
        return TaskCategory.CODE
    if _PLAN_PATTERNS.search(prompt):
        return TaskCategory.PLAN
    if _ANALYZE_PATTERNS.search(prompt):
        return TaskCategory.ANALYZE

    tools = _count_tools_requested(prompt)
    if tools > 2:
        return TaskCategory.MULTI_TOOL
    if tools > 0:
        return TaskCategory.TOOL_CALL

    if len(prompt) > 500:
        return TaskCategory.GENERATE

    return TaskCategory.UNKNOWN


# Category → minimum tier mapping
_CATEGORY_MIN_TIER: dict[TaskCategory, Tier] = {
    TaskCategory.CLASSIFY:   Tier.L0,
    TaskCategory.TOOL_CALL:  Tier.L1,
    TaskCategory.MULTI_TOOL: Tier.L2,
    TaskCategory.CODE:       Tier.L3,
    TaskCategory.GENERATE:   Tier.L2,
    TaskCategory.DEBUG:      Tier.L4,
    TaskCategory.PLAN:       Tier.L3,
    TaskCategory.ANALYZE:    Tier.L3,
    TaskCategory.SAFETY:     Tier.L4,
    TaskCategory.UNKNOWN:    Tier.L2,
}


def route_heuristic(
    prompt: str,
    max_tier: Tier = Tier.L6,
    available_tiers: Optional[list[Tier]] = None,
) -> RouteDecision:
    """
    Rule-based routing (used when no L0 model is loaded).
    
    Args:
        prompt: The user's input
        max_tier: Highest tier available on this system
        available_tiers: Specific tiers that are loaded/available
    """
    category = _categorize(prompt)
    complexity = _estimate_complexity(prompt)
    min_tier = _CATEGORY_MIN_TIER.get(category, Tier.L2)

    # Map complexity score to tier
    if complexity < 0.1:
        complexity_tier = Tier.L0
    elif complexity < 0.2:
        complexity_tier = Tier.L1
    elif complexity < 0.3:
        complexity_tier = Tier.L2
    elif complexity < 0.45:
        complexity_tier = Tier.L3
    elif complexity < 0.6:
        complexity_tier = Tier.L4
    elif complexity < 0.75:
        complexity_tier = Tier.L5
    else:
        complexity_tier = Tier.L6

    # Take the higher of category-based and complexity-based
    target_tier = Tier(max(min_tier, complexity_tier))

    # Clamp to available tiers
    if available_tiers:
        feasible = [t for t in available_tiers if t >= target_tier]
        if feasible:
            target_tier = min(feasible)  # smallest feasible tier >= target
        else:
            target_tier = max(available_tiers)  # best we can do

    # Clamp to max_tier (with L7 escalation hint if needed)
    escalation = None
    if target_tier > max_tier:
        escalation = target_tier
        target_tier = max_tier

    # Confidence: higher when the tier matches naturally, lower when clamped
    confidence = 1.0 - (abs(target_tier - max(min_tier, complexity_tier)) * 0.15)
    confidence = max(0.3, min(1.0, confidence))

    reason_parts = [
        f"category={category.value}",
        f"complexity={complexity:.2f}",
        f"min_tier={min_tier.name}",
    ]
    if escalation:
        reason_parts.append(f"wanted={escalation.name} (clamped to {target_tier.name})")

    return RouteDecision(
        tier=target_tier,
        category=category,
        confidence=confidence,
        reason=", ".join(reason_parts),
        escalation_hint=Tier.L7 if escalation else None,
    )


# ---------------------------------------------------------------------------
# L0 model-based routing (when a tiny model is loaded)
# ---------------------------------------------------------------------------

_L0_ROUTING_PROMPT = """You are a request router. Classify the user's request and respond with ONLY a JSON object:
{{"tier": "L0"|"L1"|"L2"|"L3"|"L4"|"L5"|"L6"|"L7", "category": "<category>", "confidence": 0.0-1.0}}

Tier guide:
- L0: simple yes/no, classification, intent detection
- L1: single tool call, summarization, memory lookup  
- L2: multiple simple tools, file ops, drafts
- L3: coding, 20+ tools, multi-step tasks
- L4: debugging, planning, complex reasoning
- L5: hard reasoning on strong hardware
- L6: workstation-level, complex multi-turn
- L7: frontier-only, beyond local capability

Categories: classify, tool_call, multi_tool, code, debug, plan, analyze, generate, safety

User request: {prompt}"""


def route_with_model(
    prompt: str,
    model_fn,
    max_tier: Tier = Tier.L6,
    available_tiers: Optional[list[Tier]] = None,
) -> RouteDecision:
    """
    Route using an L0 model for classification.
    
    Args:
        prompt: The user's input
        model_fn: Callable that takes a prompt string and returns model output string
        max_tier: Highest tier available on this system
        available_tiers: Specific tiers that are loaded/available
    """
    import json

    routing_prompt = _L0_ROUTING_PROMPT.format(prompt=prompt[:500])

    try:
        output = model_fn(routing_prompt)
        # Extract JSON from model output
        json_match = re.search(r'\{[^}]+\}', output)
        if json_match:
            data = json.loads(json_match.group())
            tier_str = data.get("tier", "L2")
            tier = Tier[tier_str] if tier_str in Tier.__members__ else Tier.L2
            category_str = data.get("category", "unknown")
            try:
                category = TaskCategory(category_str)
            except ValueError:
                category = TaskCategory.UNKNOWN
            confidence = float(data.get("confidence", 0.5))

            # Clamp to available tiers
            if available_tiers and tier not in available_tiers:
                feasible = [t for t in available_tiers if t >= tier]
                tier = min(feasible) if feasible else max(available_tiers)

            escalation = None
            if tier > max_tier:
                escalation = Tier.L7
                tier = max_tier

            return RouteDecision(
                tier=tier,
                category=category,
                confidence=confidence,
                reason=f"L0 model classification",
                escalation_hint=escalation,
            )
    except Exception:
        pass

    # Fallback to heuristic if model fails
    return route_heuristic(prompt, max_tier, available_tiers)
