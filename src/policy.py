"""
Cortex Policy Engine — enforces rules at request time.

The "security policy / capabilities model" of the agent microkernel.
Policies are stored in Memory (SQLite) and checked on every request.

Policy hierarchy (most specific wins):
    thread:{id}  >  app:{id}  >  global

Supported policies:
    cloud_allowed       bool     Whether cloud escalation (L7) is allowed
    max_tier            str      Maximum tier to route to (e.g. "L4")
    max_ring            int      Maximum tool permission ring (0-4)
    rate_limit          int      Max requests per minute
    max_tokens          int      Cap on max_tokens per request
    require_approval    bool     Require human approval for tool execution
    blocked_tools       list     Tool names that are blocked
    allowed_models      list     Whitelist of model names (empty = all)
    local_only          bool     Never escalate to cloud
    audit_level         str      "full" | "minimal" | "none"

Usage:
    engine = PolicyEngine(memory)
    decision = engine.check(request_context)
    if decision.denied:
        return 403, decision.reason
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from .tiers import Tier

logger = logging.getLogger("cortex.policy")


# ---------------------------------------------------------------------------
# Policy decision
# ---------------------------------------------------------------------------

@dataclass
class PolicyContext:
    """Context for a policy check — describes the incoming request."""
    app_id: str = ""
    thread_id: str = ""
    requested_model: str = ""
    requested_tier: Optional[Tier] = None
    max_tokens: int = 512
    has_tools: bool = False
    tool_names: list[str] = field(default_factory=list)
    is_cloud: bool = False
    client_ip: str = ""


@dataclass
class PolicyDecision:
    """Result of a policy check."""
    allowed: bool = True
    denied: bool = False
    reason: str = ""
    # Adjusted values (may differ from request)
    effective_max_tier: Optional[Tier] = None
    effective_max_tokens: int = 0
    effective_max_ring: int = 1      # default: DRAFT
    cloud_allowed: bool = True
    blocked_tools: list[str] = field(default_factory=list)
    audit_level: str = "full"
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Rate limiter (in-memory, per-scope)
# ---------------------------------------------------------------------------

class _RateLimiter:
    """Simple sliding-window rate limiter."""

    def __init__(self):
        self._windows: dict[str, list[float]] = {}

    def check(self, key: str, limit: int, window_seconds: int = 60) -> bool:
        """Returns True if the request is within the rate limit."""
        if limit <= 0:
            return True
        now = time.monotonic()
        timestamps = self._windows.setdefault(key, [])
        # Prune expired
        timestamps[:] = [t for t in timestamps if now - t < window_seconds]
        if len(timestamps) >= limit:
            return False
        timestamps.append(now)
        return True

    def remaining(self, key: str, limit: int, window_seconds: int = 60) -> int:
        """How many requests remain in the window."""
        if limit <= 0:
            return 999
        now = time.monotonic()
        timestamps = self._windows.get(key, [])
        active = [t for t in timestamps if now - t < window_seconds]
        return max(0, limit - len(active))


# ---------------------------------------------------------------------------
# Policy Engine
# ---------------------------------------------------------------------------

class PolicyEngine:
    """
    Enforces policies on every request.
    
    Usage:
        engine = PolicyEngine(memory)
        decision = engine.check(context)
    """

    # Default policies (used when nothing is set in Memory)
    DEFAULTS = {
        "cloud_allowed": True,
        "max_tier": "L7",
        "max_ring": 1,           # DRAFT by default
        "rate_limit": 0,         # unlimited
        "max_tokens": 8192,
        "require_approval": False,
        "blocked_tools": [],
        "allowed_models": [],
        "local_only": False,
        "audit_level": "full",
    }

    def __init__(self, memory):
        """
        Args:
            memory: A Memory instance for reading policies.
        """
        self._memory = memory
        self._rate_limiter = _RateLimiter()

    def _get(self, key: str, app_id: str = "", thread_id: str = "") -> Any:
        """Get effective policy value with hierarchy fallback."""
        val = self._memory.get_effective_policy(key, app_id=app_id, thread_id=thread_id)
        if val is not None:
            return val
        return self.DEFAULTS.get(key)

    def check(self, ctx: PolicyContext) -> PolicyDecision:
        """
        Check all policies against the request context.
        Returns a PolicyDecision with allow/deny + adjusted values.
        """
        decision = PolicyDecision()
        app_id = ctx.app_id
        thread_id = ctx.thread_id

        # --- max_tier ---
        max_tier_str = self._get("max_tier", app_id, thread_id)
        try:
            effective_max_tier = Tier[max_tier_str]
        except (KeyError, TypeError):
            effective_max_tier = Tier.L7
        decision.effective_max_tier = effective_max_tier

        if ctx.requested_tier is not None and ctx.requested_tier.value > effective_max_tier.value:
            decision.warnings.append(
                f"Requested tier {ctx.requested_tier.name} exceeds max allowed "
                f"{effective_max_tier.name}, capping"
            )

        # --- cloud_allowed / local_only ---
        cloud_allowed = self._get("cloud_allowed", app_id, thread_id)
        local_only = self._get("local_only", app_id, thread_id)
        if local_only:
            cloud_allowed = False
        decision.cloud_allowed = bool(cloud_allowed)

        if ctx.is_cloud and not decision.cloud_allowed:
            decision.allowed = False
            decision.denied = True
            decision.reason = "Cloud escalation is not allowed (local_only=true)"
            return decision

        # --- max_tokens ---
        policy_max_tokens = self._get("max_tokens", app_id, thread_id)
        decision.effective_max_tokens = min(ctx.max_tokens, int(policy_max_tokens or 8192))

        # --- max_ring (tool permissions) ---
        max_ring = self._get("max_ring", app_id, thread_id)
        decision.effective_max_ring = int(max_ring if max_ring is not None else 1)

        # --- blocked_tools ---
        blocked = self._get("blocked_tools", app_id, thread_id) or []
        decision.blocked_tools = list(blocked)
        for tool_name in ctx.tool_names:
            if tool_name in blocked:
                decision.warnings.append(f"Tool '{tool_name}' is blocked by policy")

        # --- allowed_models ---
        allowed_models = self._get("allowed_models", app_id, thread_id) or []
        if allowed_models and ctx.requested_model:
            if ctx.requested_model not in allowed_models and ctx.requested_model not in ("auto", "default", "cortex"):
                decision.allowed = False
                decision.denied = True
                decision.reason = (
                    f"Model '{ctx.requested_model}' is not in allowed_models: {allowed_models}"
                )
                return decision

        # --- rate_limit ---
        rate_limit = int(self._get("rate_limit", app_id, thread_id) or 0)
        if rate_limit > 0:
            # Rate limit by app_id or by global
            rate_key = f"app:{app_id}" if app_id else "global"
            if not self._rate_limiter.check(rate_key, rate_limit):
                remaining = self._rate_limiter.remaining(rate_key, rate_limit)
                decision.allowed = False
                decision.denied = True
                decision.reason = (
                    f"Rate limit exceeded: {rate_limit}/min "
                    f"(scope={rate_key}, remaining={remaining})"
                )
                return decision

        # --- audit_level ---
        decision.audit_level = str(self._get("audit_level", app_id, thread_id) or "full")

        return decision

    def status(self) -> dict:
        """Return current policy engine status."""
        # Read all policies from memory
        rows = self._memory._conn.execute(
            "SELECT key, scope, value FROM policies ORDER BY scope, key"
        ).fetchall()
        policies = [
            {"key": r["key"], "scope": r["scope"], "value": r["value"]}
            for r in rows
        ]
        return {
            "defaults": self.DEFAULTS,
            "active_policies": policies,
            "rate_limiter_scopes": list(self._rate_limiter._windows.keys()),
        }
