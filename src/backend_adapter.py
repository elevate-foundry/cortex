"""
Cortex backend adapter — one interface over Ollama, llama.cpp, vLLM, and remote APIs.

Every backend speaks OpenAI-compatible chat completions. This adapter normalizes
the differences (URL paths, auth, streaming format) so the rest of the system
can call `adapter.complete(messages)` without caring what's underneath.

This is the foundation for:
  - Model lifecycle manager (load/unload models)
  - Challenger (query a different-family model)
  - Swarm (fan out to N models concurrently)
"""

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator, Optional

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

try:
    import urllib.request
    import urllib.error
except ImportError:
    pass

from .config import OLLAMA_URL
from .tiers import Tier, TierModel


class BackendType(str, Enum):
    OLLAMA = "ollama"
    LLAMA_CPP = "llama_cpp"
    VLLM = "vllm"
    OPENAI_API = "openai_api"      # OpenAI, Anthropic via OpenAI-compat, etc.
    OPENROUTER = "openrouter"      # OpenRouter — multi-model proxy
    MODAL = "modal"                # Modal (Salus profile) — cloud GPU


@dataclass
class CompletionRequest:
    """Unified completion request."""
    messages: list[dict]           # [{"role": "user", "content": "..."}]
    model: str = ""                # model identifier (backend-specific)
    max_tokens: int = 512
    temperature: float = 0.0
    stream: bool = False
    stop: Optional[list[str]] = None
    tools: Optional[list[dict]] = None  # OpenAI function-calling format
    extra: dict = field(default_factory=dict)


@dataclass
class CompletionResponse:
    """Unified completion response."""
    content: str                   # the generated text
    model: str                     # which model actually responded
    backend: BackendType
    ttft_ms: float = 0.0           # time to first token
    total_ms: float = 0.0          # total generation time
    tokens_generated: int = 0
    finish_reason: str = ""
    raw: Optional[dict] = None     # raw response for debugging
    # Token usage (extracted from API response)
    tokens_prompt: int = 0         # input tokens
    tokens_completion: int = 0     # output tokens
    tokens_cached: int = 0         # prompt cache hits
    tokens_reasoning: int = 0      # internal reasoning tokens (R1, o3, etc.)
    # Cost (from OpenRouter / provider)
    cost_usd: float = 0.0          # actual cost for this request
    provider: str = ""             # upstream provider (e.g. "Together", "Fireworks")


@dataclass
class ModelStatus:
    """Status of a model on a backend."""
    model_id: str
    loaded: bool
    backend: BackendType
    vram_mb: int = 0
    size_mb: int = 0               # model file size
    details: dict = field(default_factory=dict)


class BackendAdapter:
    """
    Unified interface to a single inference backend.
    
    Each adapter instance targets one backend at one URL.
    The model lifecycle manager creates multiple adapters
    to manage multiple backends/models concurrently.
    """

    def __init__(
        self,
        backend: BackendType,
        base_url: str = "",
        api_key: str = "",
        default_model: str = "",
        timeout_s: float = 60.0,
    ):
        self.backend = backend
        self.base_url = (base_url or OLLAMA_URL).rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.default_model = default_model
        self.timeout_s = timeout_s

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        if self.backend == BackendType.OPENROUTER:
            h["HTTP-Referer"] = "https://github.com/elevate-foundry/cortex"
            h["X-Title"] = "Cortex"
        return h

    def _completions_url(self) -> str:
        # All backends speak OpenAI-compatible /v1/chat/completions
        return f"{self.base_url}/v1/chat/completions"

    def _models_url(self) -> str:
        if self.backend == BackendType.OLLAMA:
            return f"{self.base_url}/api/tags"
        return f"{self.base_url}/v1/models"

    # ------------------------------------------------------------------
    # Synchronous interface (for simple use cases)
    # ------------------------------------------------------------------

    def complete_sync(self, req: CompletionRequest) -> CompletionResponse:
        """
        Blocking completion call. Works without aiohttp.
        """
        model = req.model or self.default_model
        t0 = time.monotonic()

        payload = {
            "model": model,
            "messages": req.messages,
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
            "stream": False,
        }
        if req.stop:
            payload["stop"] = req.stop
        if req.tools:
            payload["tools"] = req.tools
        if req.extra.get("keep_alive"):
            payload["keep_alive"] = req.extra["keep_alive"]

        url = self._completions_url()
        data = json.dumps(payload).encode()
        http_req = urllib.request.Request(
            url,
            data=data,
            headers=self._headers(),
            method="POST",
        )

        try:
            with urllib.request.urlopen(http_req, timeout=self.timeout_s) as resp:
                body = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            error_body = e.read().decode() if e.fp else ""
            raise RuntimeError(
                f"Backend {self.backend.value} returned {e.code}: {error_body}"
            ) from e
        except urllib.error.URLError as e:
            raise ConnectionError(
                f"Cannot reach {self.backend.value} at {url}: {e.reason}"
            ) from e

        total_ms = (time.monotonic() - t0) * 1000

        # Parse response — unified OpenAI-compatible format (all backends)
        choice = body.get("choices", [{}])[0]
        msg = choice.get("message", {})
        content = msg.get("content") or ""
        if not content.strip():
            reasoning = msg.get("reasoning") or msg.get("thinking") or ""
            if reasoning:
                content = reasoning
            else:
                refusal = msg.get("refusal") or ""
                if refusal:
                    content = f"Refusal: {refusal}"
        resp_model = body.get("model", model)
        finish = msg.get("finish_reason", choice.get("finish_reason", "stop"))

        # Extract usage telemetry
        usage = body.get("usage", {})
        tokens_prompt = usage.get("prompt_tokens", 0) or 0
        tokens_completion = usage.get("completion_tokens", 0) or 0
        tokens_cached = 0
        tokens_reasoning = 0
        prompt_details = usage.get("prompt_tokens_details", {})
        if prompt_details:
            tokens_cached = prompt_details.get("cached_tokens", 0) or 0
        completion_details = usage.get("completion_tokens_details", {})
        if completion_details:
            tokens_reasoning = completion_details.get("reasoning_tokens", 0) or 0
        cost_usd = float(usage.get("cost", 0) or 0)
        provider = body.get("provider", "")

        return CompletionResponse(
            content=content,
            model=resp_model,
            backend=self.backend,
            total_ms=total_ms,
            tokens_generated=tokens_completion,
            finish_reason=finish,
            raw=body,
            tokens_prompt=tokens_prompt,
            tokens_completion=tokens_completion,
            tokens_cached=tokens_cached,
            tokens_reasoning=tokens_reasoning,
            cost_usd=cost_usd,
            provider=provider,
        )

    # ------------------------------------------------------------------
    # Async interface (for swarm / concurrent use)
    # ------------------------------------------------------------------

    async def complete(self, req: CompletionRequest) -> CompletionResponse:
        """
        Async completion call. Requires aiohttp.
        Falls back to sync in a thread if aiohttp is unavailable.
        """
        if not HAS_AIOHTTP:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.complete_sync, req)

        model = req.model or self.default_model
        t0 = time.monotonic()

        payload = {
            "model": model,
            "messages": req.messages,
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
            "stream": False,
        }
        if req.stop:
            payload["stop"] = req.stop
        if req.tools:
            payload["tools"] = req.tools

        url = self._completions_url()
        timeout = aiohttp.ClientTimeout(total=self.timeout_s)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                url,
                json=payload,
                headers=self._headers(),
            ) as resp:
                if resp.status != 200:
                    error_body = await resp.text()
                    raise RuntimeError(
                        f"Backend {self.backend.value} returned {resp.status}: {error_body}"
                    )
                body = await resp.json()

        total_ms = (time.monotonic() - t0) * 1000

        choice = body.get("choices", [{}])[0]
        msg = choice.get("message", {})
        content = msg.get("content") or ""
        if not content.strip():
            reasoning = msg.get("reasoning") or msg.get("thinking") or ""
            if reasoning:
                content = reasoning
            else:
                refusal = msg.get("refusal") or ""
                if refusal:
                    content = f"Refusal: {refusal}"
        resp_model = body.get("model", model)
        finish = msg.get("finish_reason", choice.get("finish_reason", "stop"))

        # Extract usage telemetry
        usage = body.get("usage", {})
        tokens_prompt = usage.get("prompt_tokens", 0) or 0
        tokens_completion = usage.get("completion_tokens", 0) or 0
        tokens_cached = 0
        tokens_reasoning = 0
        prompt_details = usage.get("prompt_tokens_details", {})
        if prompt_details:
            tokens_cached = prompt_details.get("cached_tokens", 0) or 0
        completion_details = usage.get("completion_tokens_details", {})
        if completion_details:
            tokens_reasoning = completion_details.get("reasoning_tokens", 0) or 0
        cost_usd = float(usage.get("cost", 0) or 0)
        provider = body.get("provider", "")

        return CompletionResponse(
            content=content,
            model=resp_model,
            backend=self.backend,
            total_ms=total_ms,
            tokens_generated=tokens_completion,
            finish_reason=finish,
            raw=body,
            tokens_prompt=tokens_prompt,
            tokens_completion=tokens_completion,
            tokens_cached=tokens_cached,
            tokens_reasoning=tokens_reasoning,
            cost_usd=cost_usd,
            provider=provider,
        )

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------

    def list_models_sync(self) -> list[ModelStatus]:
        """List models available/loaded on this backend."""
        url = self._models_url()
        http_req = urllib.request.Request(url, headers=self._headers())

        try:
            with urllib.request.urlopen(http_req, timeout=10) as resp:
                body = json.loads(resp.read())
        except (urllib.error.URLError, urllib.error.HTTPError):
            return []

        models: list[ModelStatus] = []

        if self.backend == BackendType.OLLAMA:
            for m in body.get("models", []):
                models.append(ModelStatus(
                    model_id=m.get("name", ""),
                    loaded=True,  # listed = downloaded
                    backend=self.backend,
                    size_mb=m.get("size", 0) // (1024 * 1024),
                    details=m.get("details", {}),
                ))
        else:
            for m in body.get("data", []):
                models.append(ModelStatus(
                    model_id=m.get("id", ""),
                    loaded=True,
                    backend=self.backend,
                ))

        return models

    def pull_model_sync(self, model_id: str) -> bool:
        """
        Pull/download a model. Only supported on Ollama.
        Returns True on success.
        """
        if self.backend != BackendType.OLLAMA:
            return False

        url = f"{self.base_url}/api/pull"
        payload = json.dumps({"name": model_id, "stream": False}).encode()
        http_req = urllib.request.Request(
            url,
            data=payload,
            headers=self._headers(),
            method="POST",
        )

        try:
            with urllib.request.urlopen(http_req, timeout=600) as resp:
                resp.read()
            return True
        except (urllib.error.URLError, urllib.error.HTTPError):
            return False

    def health_check(self) -> bool:
        """Check if the backend is reachable."""
        try:
            if self.backend == BackendType.OLLAMA:
                url = self.base_url
            else:
                url = f"{self.base_url}/v1/models"

            http_req = urllib.request.Request(url, headers=self._headers())
            with urllib.request.urlopen(http_req, timeout=5) as resp:
                return resp.status == 200
        except (urllib.error.URLError, urllib.error.HTTPError, OSError):
            return False

    def __repr__(self) -> str:
        return f"BackendAdapter({self.backend.value}, {self.base_url}, model={self.default_model!r})"


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def ollama_adapter(
    model: str = "",
    base_url: str = "",
) -> BackendAdapter:
    """Create an adapter for a local Ollama instance."""
    return BackendAdapter(
        backend=BackendType.OLLAMA,
        base_url=base_url,
        default_model=model,
    )


def llama_cpp_adapter(
    model: str = "",
    base_url: str = "http://localhost:8080",
) -> BackendAdapter:
    """Create an adapter for a llama.cpp server."""
    return BackendAdapter(
        backend=BackendType.LLAMA_CPP,
        base_url=base_url,
        default_model=model,
    )


def vllm_adapter(
    model: str = "",
    base_url: str = "http://localhost:8000",
) -> BackendAdapter:
    """Create an adapter for a vLLM server."""
    return BackendAdapter(
        backend=BackendType.VLLM,
        base_url=base_url,
        default_model=model,
    )


def openai_adapter(
    model: str = "gpt-4o",
    api_key: str = "",
) -> BackendAdapter:
    """Create an adapter for the OpenAI API."""
    return BackendAdapter(
        backend=BackendType.OPENAI_API,
        base_url="https://api.openai.com",
        api_key=api_key or os.environ.get("OPENAI_API_KEY", ""),
        default_model=model,
    )


def openrouter_adapter(
    model: str = "qwen/qwen3-8b",
    api_key: str = "",
) -> BackendAdapter:
    """Create an adapter for OpenRouter (multi-model cloud proxy)."""
    key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        # Try loading from bin/.env on the drive
        from pathlib import Path
        env_path = Path("/Volumes/CORTEX/cortex/bin/.env")
        if env_path.exists():
            for line in env_path.read_text().strip().split("\n"):
                if line.startswith("OPENROUTER_API_KEY="):
                    key = line.split("=", 1)[1].strip()
                    break
    return BackendAdapter(
        backend=BackendType.OPENROUTER,
        base_url="https://openrouter.ai/api",
        api_key=key,
        default_model=model,
    )


# ---------------------------------------------------------------------------
# Multi-backend orchestrator
# ---------------------------------------------------------------------------

class BackendPool:
    """
    Manages multiple backends with automatic discovery and fallback.

    Priority order:
      1. Ollama (local, free, fast)
      2. llama.cpp (local, free, fast)
      3. vLLM (local or remote, fast)
      4. OpenRouter (cloud, cheap, wide model coverage)
      5. OpenAI/Anthropic (cloud, expensive, last resort)

    Usage:
      pool = BackendPool.auto_discover()
      response = await pool.complete(request)
    """

    def __init__(self):
        self.backends: list[BackendAdapter] = []
        self._primary: Optional[BackendAdapter] = None

    @classmethod
    def auto_discover(cls) -> "BackendPool":
        """
        Probe for available backends and create a pool.

        Checks:
          - Ollama at localhost:11434 (default)
          - llama.cpp at localhost:8080
          - vLLM at localhost:8000
          - OpenRouter if OPENROUTER_API_KEY is set
          - OpenAI if OPENAI_API_KEY is set
        """
        import logging
        logger = logging.getLogger("cortex.backend_pool")

        pool = cls()

        # Ollama
        ollama = ollama_adapter()
        if ollama.health_check():
            pool.backends.append(ollama)
            logger.info("Backend discovered: Ollama at %s", ollama.base_url)

        # llama.cpp
        lcpp = llama_cpp_adapter()
        if lcpp.health_check():
            pool.backends.append(lcpp)
            logger.info("Backend discovered: llama.cpp at %s", lcpp.base_url)

        # vLLM
        vllm = vllm_adapter()
        if vllm.health_check():
            pool.backends.append(vllm)
            logger.info("Backend discovered: vLLM at %s", vllm.base_url)

        # OpenRouter (always available if key exists)
        or_adapter = openrouter_adapter()
        if or_adapter.api_key:
            pool.backends.append(or_adapter)
            logger.info("Backend discovered: OpenRouter (API key present)")

        # OpenAI (if key exists)
        oai_key = os.environ.get("OPENAI_API_KEY", "")
        if oai_key and oai_key != "local":
            pool.backends.append(openai_adapter(api_key=oai_key))
            logger.info("Backend discovered: OpenAI API")

        if pool.backends:
            pool._primary = pool.backends[0]
            logger.info("Primary backend: %s", pool._primary.backend.value)
        else:
            logger.warning("No backends discovered!")

        return pool

    @property
    def primary(self) -> Optional[BackendAdapter]:
        return self._primary

    @property
    def available(self) -> bool:
        return len(self.backends) > 0

    def get_backend(self, backend_type: BackendType) -> Optional[BackendAdapter]:
        """Get a specific backend by type."""
        for b in self.backends:
            if b.backend == backend_type:
                return b
        return None

    def get_backend_for_model(self, model: str) -> Optional[BackendAdapter]:
        """
        Find the best backend for a given model.

        Logic:
          - If model looks like an OpenRouter ID (org/name), use OpenRouter
          - If model is a local name (no slash, or ollama tag format), use Ollama
          - Otherwise use primary
        """
        if "/" in model and not model.startswith("http"):
            # Looks like org/model format (OpenRouter, HuggingFace)
            or_backend = self.get_backend(BackendType.OPENROUTER)
            if or_backend:
                return or_backend
        # Local model name — try Ollama first
        ollama_backend = self.get_backend(BackendType.OLLAMA)
        if ollama_backend:
            return ollama_backend
        return self._primary

    async def complete(
        self, req: CompletionRequest, preferred_backend: Optional[BackendType] = None
    ) -> CompletionResponse:
        """
        Complete a request using the best available backend.

        Tries preferred_backend first, then falls back through the pool.
        """
        # Pick starting backend
        if preferred_backend:
            backend = self.get_backend(preferred_backend)
            if backend:
                try:
                    return await backend.complete(req)
                except Exception:
                    pass  # Fall through to others

        # Route by model name
        if req.model:
            backend = self.get_backend_for_model(req.model)
            if backend:
                try:
                    return await backend.complete(req)
                except Exception:
                    pass

        # Fallback chain
        last_error = None
        for backend in self.backends:
            try:
                return await backend.complete(req)
            except Exception as e:
                last_error = e
                continue

        raise ConnectionError(
            f"All backends failed. Last error: {last_error}"
        )

    async def complete_streaming(
        self, req: CompletionRequest, preferred_backend: Optional[BackendType] = None
    ) -> AsyncIterator[str]:
        """
        Stream a completion response as SSE chunks.

        Yields: SSE-formatted data lines ("data: {...}\n\n")
        """
        # Force streaming in request
        req.stream = True

        # Pick backend
        backend = None
        if preferred_backend:
            backend = self.get_backend(preferred_backend)
        if not backend and req.model:
            backend = self.get_backend_for_model(req.model)
        if not backend:
            backend = self._primary
        if not backend:
            raise ConnectionError("No backends available")

        # Stream via aiohttp
        if not HAS_AIOHTTP:
            raise RuntimeError("aiohttp required for streaming")

        model = req.model or backend.default_model
        payload = {
            "model": model,
            "messages": req.messages,
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
            "stream": True,
        }
        if req.stop:
            payload["stop"] = req.stop
        if req.tools:
            payload["tools"] = req.tools

        headers = {"Content-Type": "application/json"}
        if backend.api_key:
            headers["Authorization"] = f"Bearer {backend.api_key}"
        # OpenRouter extras
        if backend.backend == BackendType.OPENROUTER:
            headers["HTTP-Referer"] = "https://github.com/elevate-foundry/cortex"
            headers["X-Title"] = "Cortex"

        url = backend._completions_url()
        timeout = aiohttp.ClientTimeout(total=backend.timeout_s)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload, headers=headers) as resp:
                if resp.status != 200:
                    error_body = await resp.text()
                    raise RuntimeError(
                        f"Backend {backend.backend.value} returned {resp.status}: {error_body}"
                    )
                async for line in resp.content:
                    decoded = line.decode("utf-8", errors="replace").strip()
                    if decoded.startswith("data: "):
                        yield decoded + "\n\n"
                    elif decoded == "data: [DONE]":
                        yield "data: [DONE]\n\n"
                        break

    def summary(self) -> dict:
        """Return a summary of available backends."""
        return {
            "backends": [
                {
                    "type": b.backend.value,
                    "url": b.base_url,
                    "model": b.default_model,
                    "is_primary": b is self._primary,
                }
                for b in self.backends
            ],
            "primary": self._primary.backend.value if self._primary else None,
            "count": len(self.backends),
        }
