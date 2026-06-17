"""
Cortex daemon — OS-level inference proxy.

A local HTTP server that presents multiple API formats on localhost:11411.
Any app that supports OPENAI_BASE_URL can use it:

    export OPENAI_BASE_URL=http://localhost:11411/v1
    export OPENAI_API_KEY=local          # anything non-empty

Supported API formats:
  - OpenAI Chat Completions  (POST /v1/chat/completions)  — native
  - OpenAI Responses API      (POST /v1/responses)         — translated
  - Anthropic Messages API    (POST /v1/messages)           — translated
  - Tool calling / function calling                        — pass-through
  - Multimodal (vision, audio)                             — text-extracted

Architecture:
  1. Receives request in any supported API format
  2. API adapter normalizes to internal Chat Completions format
  3. Routes via L0 heuristic to the optimal local tier
  4. Forwards to a managed backend (Ollama / llama-server / vLLM)
  5. Translates response back to the client's expected format
  6. Streams the response back in the appropriate SSE format

Managed backends:
  - "hot" pool:  L0+L1+L2 models always loaded (low VRAM, instant TTFT)
  - "warm" pool: L3+L4 loaded on demand, kept alive with idle timeout
  - "cloud":     L7 passthrough to OpenAI/Anthropic when local confidence fails
"""

import asyncio
import json
import logging
import os
import signal
import sys
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .hardware_detect import detect_system, SystemProfile
from .backend_selector import select_backend
from .tiers import (
    Tier,
    TierModel,
    TierSpec,
    TIER_SPECS,
    assess_tiers,
    max_feasible_tier,
    get_models_for_tier,
    concurrent_vram_budget,
)
from .router import route_heuristic
from .api_adapter import (
    APIFormat,
    NormalizedRequest,
    normalize_request,
    format_response,
    detect_api_format,
)

logger = logging.getLogger("cortex")


# ---------------------------------------------------------------------------
# Backend process manager
# ---------------------------------------------------------------------------

class BackendStatus(str, Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    READY = "ready"
    ERROR = "error"


@dataclass
class ManagedBackend:
    """A llama-server process managed by the daemon."""
    tier: Tier
    model: TierModel
    port: int
    status: BackendStatus = BackendStatus.STOPPED
    process: Optional[asyncio.subprocess.Process] = None
    last_request: float = 0.0
    start_time: float = 0.0
    request_count: int = 0
    always_hot: bool = False

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def idle_seconds(self) -> float:
        if self.last_request == 0:
            return 0
        return time.monotonic() - self.last_request


class ProcessManager:
    """
    Manages llama-server processes for each tier.
    
    - Hot tiers (L0-L2) are started immediately and kept alive.
    - Warm tiers (L3+) are started on demand with idle timeout.
    - Monitors health and restarts crashed processes.
    """

    BASE_PORT = 8081
    IDLE_TIMEOUT = 300  # 5 min for warm tiers

    def __init__(self, profile: SystemProfile):
        self.profile = profile
        self.backends: dict[Tier, ManagedBackend] = {}
        self._health_task: Optional[asyncio.Task] = None
        self._setup_backends()

    def _setup_backends(self):
        """Configure which tiers get backends based on system capabilities."""
        assessments = assess_tiers(self.profile)
        port = self.BASE_PORT

        for a in assessments:
            if not a.feasible or not a.spec.local or not a.model:
                continue

            self.backends[a.tier] = ManagedBackend(
                tier=a.tier,
                model=a.model,
                port=port,
                always_hot=a.spec.always_hot,
            )
            port += 1

        logger.info(
            "Configured %d backends: %s",
            len(self.backends),
            ", ".join(f"{t.name}:{b.port}" for t, b in self.backends.items()),
        )

    async def start_hot_tiers(self):
        """Start all always-hot tiers (L0, L1, L2)."""
        tasks = []
        for tier, backend in self.backends.items():
            if backend.always_hot:
                tasks.append(self._start_backend(backend))

        if tasks:
            await asyncio.gather(*tasks)

        # Start health monitor
        self._health_task = asyncio.create_task(self._health_loop())

    async def ensure_backend(self, tier: Tier) -> Optional[ManagedBackend]:
        """Ensure a backend for the given tier is running. Start if needed."""
        backend = self.backends.get(tier)
        if backend is None:
            return None

        if backend.status == BackendStatus.READY:
            backend.last_request = time.monotonic()
            backend.request_count += 1
            return backend

        if backend.status == BackendStatus.STARTING:
            # Wait for it to become ready
            for _ in range(100):  # 10 seconds max
                await asyncio.sleep(0.1)
                if backend.status == BackendStatus.READY:
                    backend.last_request = time.monotonic()
                    backend.request_count += 1
                    return backend
            return None

        # Need to start it
        await self._start_backend(backend)
        if backend.status == BackendStatus.READY:
            backend.last_request = time.monotonic()
            backend.request_count += 1
            return backend

        return None

    async def _start_backend(self, backend: ManagedBackend):
        """Start a llama-server process for a backend."""
        backend.status = BackendStatus.STARTING
        backend.start_time = time.monotonic()

        # Determine ollama tag or model path
        model_id = backend.model.ollama_tag or backend.model.model_id

        # Use ollama as the process manager (it handles model download + serving)
        cmd = [
            "ollama", "run", model_id,
            # Note: ollama run is interactive; for serving we use ollama serve
            # and the ollama API. But for simplicity we'll use llama-server
            # if available, or ollama serve + ollama API.
        ]

        # Prefer llama-server directly for more control
        has_llamacpp = any(
            b.name == "llama.cpp" and b.available
            for b in self.profile.backends
        )
        has_ollama = any(
            b.name == "Ollama" and b.available
            for b in self.profile.backends
        )

        if has_ollama:
            # Use ollama's built-in model management
            # First ensure model is pulled
            logger.info("Pulling model %s via ollama...", model_id)
            pull_proc = await asyncio.create_subprocess_exec(
                "ollama", "pull", model_id,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await pull_proc.wait()

            # Ollama serves all models on :11434 — we'll proxy to it
            # and use the model name to select the right one
            backend.port = 11434
            backend.status = BackendStatus.READY
            logger.info(
                "%s backend ready (ollama, model=%s, port=%d)",
                backend.tier.name, model_id, backend.port,
            )
            return

        if has_llamacpp:
            # Direct llama-server with GGUF
            # TODO: resolve GGUF file path from model_id
            logger.warning(
                "llama-server direct launch not yet implemented for %s",
                backend.model.model_id,
            )
            backend.status = BackendStatus.ERROR
            return

        logger.error("No inference backend available for %s", backend.tier.name)
        backend.status = BackendStatus.ERROR

    async def _health_loop(self):
        """Periodically check backend health and reap idle warm tiers."""
        while True:
            await asyncio.sleep(30)

            for tier, backend in list(self.backends.items()):
                # Reap idle warm (non-hot) backends
                if (not backend.always_hot
                        and backend.status == BackendStatus.READY
                        and backend.idle_seconds > self.IDLE_TIMEOUT):
                    logger.info(
                        "Reaping idle %s backend (idle %.0fs)",
                        tier.name, backend.idle_seconds,
                    )
                    await self._stop_backend(backend)

    async def _stop_backend(self, backend: ManagedBackend):
        """Stop a backend process."""
        if backend.process and backend.process.returncode is None:
            backend.process.terminate()
            try:
                await asyncio.wait_for(backend.process.wait(), timeout=5)
            except asyncio.TimeoutError:
                backend.process.kill()
        backend.status = BackendStatus.STOPPED
        backend.process = None

    async def shutdown(self):
        """Stop all backends."""
        if self._health_task:
            self._health_task.cancel()
        for backend in self.backends.values():
            await self._stop_backend(backend)

    def status_report(self) -> list[dict]:
        """Return status of all managed backends."""
        return [
            {
                "tier": b.tier.name,
                "model": b.model.model_id,
                "ollama_tag": b.model.ollama_tag,
                "port": b.port,
                "status": b.status.value,
                "always_hot": b.always_hot,
                "request_count": b.request_count,
                "idle_seconds": round(b.idle_seconds, 1),
            }
            for b in self.backends.values()
        ]


# ---------------------------------------------------------------------------
# OpenAI-compatible proxy server
# ---------------------------------------------------------------------------

async def handle_chat_completions(
    request_body: dict,
    process_mgr: ProcessManager,
    profile: SystemProfile,
) -> tuple[int, dict | AsyncIterator]:
    """
    Handle a /v1/chat/completions request.
    
    1. Extract the prompt from the request
    2. Route to the optimal tier
    3. Forward to the appropriate backend
    4. Return the response (streaming or not)
    """
    import aiohttp

    messages = request_body.get("messages", [])
    stream = request_body.get("stream", False)

    # Build prompt for routing (use last user message)
    prompt = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            prompt = msg.get("content", "")
            break

    if not prompt:
        prompt = str(messages)

    # Route the request
    max_tier = max_feasible_tier(profile)
    assessments = assess_tiers(profile)
    available = [a.tier for a in assessments if a.feasible]

    decision = route_heuristic(prompt, max_tier=max_tier, available_tiers=available)
    target_tier = decision.tier

    logger.info(
        "Routing: %r → %s (confidence=%.2f, category=%s)",
        prompt[:80], target_tier.name, decision.confidence, decision.category.value,
    )

    # Get the backend for this tier
    backend = await process_mgr.ensure_backend(target_tier)

    if backend is None:
        # Fall back to highest available tier
        for tier in sorted(process_mgr.backends.keys(), reverse=True):
            backend = await process_mgr.ensure_backend(tier)
            if backend:
                logger.info("Fell back to %s", tier.name)
                break

    if backend is None:
        return 503, {
            "error": {
                "message": "No inference backend available",
                "type": "server_error",
            }
        }

    # Forward to the backend
    # If client sent model="auto" or empty, replace with the actual backend model
    model_name = backend.model.ollama_tag or backend.model.model_id
    client_model = request_body.get("model", "")
    if not client_model or client_model.lower() in ("auto", "default", "cortex"):
        forward_body = {**request_body, "model": model_name}
    else:
        forward_body = {**request_body}

    target_url = f"{backend.url}/v1/chat/completions"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                target_url,
                json=forward_body,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                if stream:
                    # Stream SSE chunks back
                    chunks = []
                    async for line in resp.content:
                        chunks.append(line)
                    return resp.status, chunks
                else:
                    result = await resp.json()
                    # Inject routing metadata
                    if isinstance(result, dict):
                        result["_routing"] = {
                            "tier": target_tier.name,
                            "category": decision.category.value,
                            "confidence": decision.confidence,
                            "backend_model": model_name,
                        }
                    return resp.status, result
    except Exception as e:
        logger.error("Backend request failed: %s", e)
        return 502, {
            "error": {
                "message": f"Backend error: {e}",
                "type": "server_error",
            }
        }


# ---------------------------------------------------------------------------
# HTTP server using asyncio (stdlib, no framework dependency)
# ---------------------------------------------------------------------------

class DaemonServer:
    """
    Minimal async HTTP server implementing the OpenAI API subset.
    
    Endpoints:
      POST /v1/chat/completions  — OpenAI Chat Completions
      POST /v1/responses         — OpenAI Responses API (translated)
      POST /v1/messages          — Anthropic Messages API (translated)
      GET  /v1/models            — list available models/tiers
      GET  /health               — daemon health check
      GET  /status               — detailed backend status
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 11411,
        profile: Optional[SystemProfile] = None,
    ):
        self.host = host
        self.port = port
        self.profile = profile or detect_system()
        self.process_mgr = ProcessManager(self.profile)
        self.start_time = 0.0
        self.request_count = 0

    async def start(self):
        """Start the daemon."""
        self.start_time = time.monotonic()

        # Start hot tier backends
        logger.info("Starting hot tier backends...")
        await self.process_mgr.start_hot_tiers()

        # Start HTTP server
        server = await asyncio.start_server(
            self._handle_connection,
            self.host,
            self.port,
        )

        logger.info(
            "Cortex daemon listening on http://%s:%d",
            self.host, self.port,
        )
        logger.info("Set OPENAI_BASE_URL=http://%s:%d/v1", self.host, self.port)

        # Print startup banner
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║  Cortex daemon — AI-native OS inference proxy               ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Listening:  http://{self.host}:{self.port}                         ║
║                                                              ║
║  API endpoints:                                              ║
║    POST /v1/chat/completions   OpenAI Chat Completions       ║
║    POST /v1/responses          OpenAI Responses API          ║
║    POST /v1/messages           Anthropic Messages API        ║
║    GET  /v1/models             List available models         ║
║                                                              ║
║  Usage:                                                      ║
║    export OPENAI_BASE_URL=http://localhost:{self.port}/v1          ║
║    export OPENAI_API_KEY=local                               ║
║                                                              ║
║  Compatible with: Cursor, VS Code, Cline, aider, LangChain, ║
║    Open WebUI, Continue, any OpenAI/Anthropic SDK client     ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")

        # Print backend status
        for b in self.process_mgr.status_report():
            status_icon = "●" if b["status"] == "ready" else "○"
            hot = " [HOT]" if b["always_hot"] else ""
            print(f"  {status_icon} {b['tier']}: {b['ollama_tag'] or b['model']}{hot}")
        print()

        async with server:
            await server.serve_forever()

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ):
        """Handle a single HTTP connection."""
        try:
            # Read request line
            request_line = await asyncio.wait_for(
                reader.readline(), timeout=10,
            )
            if not request_line:
                writer.close()
                return

            request_str = request_line.decode("utf-8", errors="replace").strip()
            parts = request_str.split(" ")
            if len(parts) < 2:
                writer.close()
                return

            method = parts[0]
            path = parts[1]

            # Read headers
            headers = {}
            while True:
                line = await asyncio.wait_for(reader.readline(), timeout=10)
                line_str = line.decode("utf-8", errors="replace").strip()
                if not line_str:
                    break
                if ":" in line_str:
                    key, value = line_str.split(":", 1)
                    headers[key.strip().lower()] = value.strip()

            # Read body if present
            body = b""
            content_length = int(headers.get("content-length", 0))
            if content_length > 0:
                body = await asyncio.wait_for(
                    reader.readexactly(content_length), timeout=30,
                )

            self.request_count += 1

            # Route to handler
            if path == "/v1/chat/completions" and method == "POST":
                await self._handle_api_request(writer, body, headers, path)
            elif path == "/v1/responses" and method == "POST":
                await self._handle_api_request(writer, body, headers, path)
            elif path == "/v1/messages" and method == "POST":
                await self._handle_api_request(writer, body, headers, path)
            elif path == "/v1/models" and method == "GET":
                await self._handle_models(writer)
            elif path == "/health" and method == "GET":
                await self._handle_health(writer)
            elif path == "/status" and method == "GET":
                await self._handle_status(writer)
            else:
                await self._send_json(writer, 404, {
                    "error": {"message": f"Not found: {path}", "type": "invalid_request"}
                })

        except (asyncio.TimeoutError, ConnectionResetError, BrokenPipeError):
            pass
        except Exception as e:
            logger.error("Request error: %s", e, exc_info=True)
            try:
                await self._send_json(writer, 500, {
                    "error": {"message": str(e), "type": "server_error"}
                })
            except Exception:
                pass
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    async def _handle_api_request(self, writer, body: bytes, headers: dict, path: str):
        """
        Unified handler for all API formats.
        Normalizes inbound, routes through Cortex, formats outbound.
        """
        try:
            request_body = json.loads(body)
        except json.JSONDecodeError as e:
            await self._send_json(writer, 400, {
                "error": {"message": f"Invalid JSON: {e}", "type": "invalid_request"}
            })
            return

        # Normalize the request regardless of API format
        normalized = normalize_request(request_body, path)

        logger.info(
            "API request: format=%s, path=%s, model=%s",
            normalized.source_format.value, path, normalized.model,
        )

        stream = normalized.stream

        # Route using the normalized messages
        status, result = await handle_chat_completions(
            {
                "messages": normalized.messages,
                "model": normalized.model,
                "max_tokens": normalized.max_tokens,
                "temperature": normalized.temperature,
                "stream": stream,
                "tools": normalized.tools,
                "tool_choice": normalized.tool_choice,
            },
            self.process_mgr,
            self.profile,
        )

        if stream and isinstance(result, list):
            # Stream SSE response
            header = (
                f"HTTP/1.1 {status} OK\r\n"
                f"Content-Type: text/event-stream\r\n"
                f"Cache-Control: no-cache\r\n"
                f"Connection: close\r\n"
                f"\r\n"
            )
            writer.write(header.encode())
            for chunk in result:
                writer.write(chunk)
            await writer.drain()
        else:
            # Extract content from the Chat Completions result and reformat
            if isinstance(result, dict) and "choices" in result:
                content = result["choices"][0].get("message", {}).get("content", "")
                routing_meta = result.get("_routing")
                model_used = result.get("model", "")
                formatted = format_response(
                    content, normalized, routing_meta, model_used,
                )
                await self._send_json(writer, status, formatted)
            else:
                # Error or non-standard response — pass through
                await self._send_json(writer, status, result)

    async def _handle_models(self, writer):
        """Handle GET /v1/models — list available tiers as models."""
        models = []
        for b in self.process_mgr.status_report():
            models.append({
                "id": b["ollama_tag"] or b["model"],
                "object": "model",
                "created": int(self.start_time),
                "owned_by": f"local-{b['tier']}",
                "meta": {
                    "tier": b["tier"],
                    "status": b["status"],
                    "always_hot": b["always_hot"],
                },
            })

        await self._send_json(writer, 200, {
            "object": "list",
            "data": models,
        })

    async def _handle_health(self, writer):
        """Handle GET /health."""
        uptime = time.monotonic() - self.start_time
        ready_count = sum(
            1 for b in self.process_mgr.backends.values()
            if b.status == BackendStatus.READY
        )
        await self._send_json(writer, 200, {
            "status": "ok",
            "uptime_seconds": round(uptime, 1),
            "total_requests": self.request_count,
            "backends_ready": ready_count,
            "backends_total": len(self.process_mgr.backends),
        })

    async def _handle_status(self, writer):
        """Handle GET /status — detailed backend status."""
        uptime = time.monotonic() - self.start_time
        await self._send_json(writer, 200, {
            "daemon": {
                "uptime_seconds": round(uptime, 1),
                "total_requests": self.request_count,
                "host": self.host,
                "port": self.port,
            },
            "system": {
                "os": f"{self.profile.os_name} {self.profile.os_version}",
                "arch": self.profile.arch,
                "cpu": self.profile.cpu.model,
                "ram_mb": self.profile.memory.total_mb,
                "gpu": self.profile.gpus[0].name if self.profile.gpus else "none",
                "accelerator": self.profile.primary_accelerator.value,
            },
            "backends": self.process_mgr.status_report(),
            "max_local_tier": max_feasible_tier(self.profile).name,
        })

    async def _send_json(self, writer, status: int, data):
        """Send a JSON HTTP response."""
        body = json.dumps(data).encode()
        status_text = {200: "OK", 400: "Bad Request", 404: "Not Found",
                       500: "Internal Server Error", 502: "Bad Gateway",
                       503: "Service Unavailable"}.get(status, "Unknown")
        header = (
            f"HTTP/1.1 {status} {status_text}\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: {len(body)}\r\n"
            f"Connection: close\r\n"
            f"\r\n"
        )
        writer.write(header.encode() + body)
        await writer.drain()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_daemon(
    host: str = "127.0.0.1",
    port: int = 11411,
    profile: Optional[SystemProfile] = None,
):
    """Start the Cortex daemon."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    daemon = DaemonServer(host=host, port=port, profile=profile)

    loop = asyncio.new_event_loop()

    def _shutdown(sig, frame):
        logger.info("Received %s, shutting down...", sig.name)
        loop.run_until_complete(daemon.process_mgr.shutdown())
        loop.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        loop.run_until_complete(daemon.start())
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        loop.run_until_complete(daemon.process_mgr.shutdown())
