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

from .tiers import Tier, TierModel


class BackendType(str, Enum):
    OLLAMA = "ollama"
    LLAMA_CPP = "llama_cpp"
    VLLM = "vllm"
    OPENAI_API = "openai_api"      # OpenAI, Anthropic via OpenAI-compat, etc.


@dataclass
class CompletionRequest:
    """Unified completion request."""
    messages: list[dict]           # [{"role": "user", "content": "..."}]
    model: str = ""                # model identifier (backend-specific)
    max_tokens: int = 512
    temperature: float = 0.0
    stream: bool = False
    stop: Optional[list[str]] = None
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
        base_url: str = "http://localhost:11434",
        api_key: str = "",
        default_model: str = "",
        timeout_s: float = 60.0,
    ):
        self.backend = backend
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.default_model = default_model
        self.timeout_s = timeout_s

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _completions_url(self) -> str:
        if self.backend == BackendType.OLLAMA:
            return f"{self.base_url}/api/chat"
        # vLLM, llama.cpp, OpenAI all use /v1/chat/completions
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

        if self.backend == BackendType.OLLAMA:
            payload = {
                "model": model,
                "messages": req.messages,
                "stream": False,
                "options": {
                    "temperature": req.temperature,
                    "num_predict": req.max_tokens,
                },
            }
            if req.stop:
                payload["options"]["stop"] = req.stop
        else:
            payload = {
                "model": model,
                "messages": req.messages,
                "max_tokens": req.max_tokens,
                "temperature": req.temperature,
                "stream": False,
            }
            if req.stop:
                payload["stop"] = req.stop

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

        # Parse response — Ollama vs OpenAI format
        if self.backend == BackendType.OLLAMA:
            content = body.get("message", {}).get("content", "")
            resp_model = body.get("model", model)
            finish = "stop"
        else:
            choice = body.get("choices", [{}])[0]
            content = choice.get("message", {}).get("content", "")
            resp_model = body.get("model", model)
            finish = choice.get("finish_reason", "")

        return CompletionResponse(
            content=content,
            model=resp_model,
            backend=self.backend,
            total_ms=total_ms,
            finish_reason=finish,
            raw=body,
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

        if self.backend == BackendType.OLLAMA:
            payload = {
                "model": model,
                "messages": req.messages,
                "stream": False,
                "options": {
                    "temperature": req.temperature,
                    "num_predict": req.max_tokens,
                },
            }
            if req.stop:
                payload["options"]["stop"] = req.stop
        else:
            payload = {
                "model": model,
                "messages": req.messages,
                "max_tokens": req.max_tokens,
                "temperature": req.temperature,
                "stream": False,
            }
            if req.stop:
                payload["stop"] = req.stop

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

        if self.backend == BackendType.OLLAMA:
            content = body.get("message", {}).get("content", "")
            resp_model = body.get("model", model)
            finish = "stop"
        else:
            choice = body.get("choices", [{}])[0]
            content = choice.get("message", {}).get("content", "")
            resp_model = body.get("model", model)
            finish = choice.get("finish_reason", "")

        return CompletionResponse(
            content=content,
            model=resp_model,
            backend=self.backend,
            total_ms=total_ms,
            finish_reason=finish,
            raw=body,
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
    base_url: str = "http://localhost:11434",
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
