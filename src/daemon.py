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
import signal
import sys
import time
from typing import Optional

from .hardware_detect import detect_system, SystemProfile
from .tiers import (
    Tier,
    TIER_SPECS,
    max_feasible_tier,
    get_models_for_tier,
    model_census,
    refresh_ollama_discovery,
)
from .api_adapter import (
    normalize_request,
    format_response,
)
from .cortex import Cortex, CortexResponse
from .memory import Memory
from .tools import ToolRegistry, ToolCall, PermissionRing
from .policy import PolicyEngine, PolicyContext
from .resilience import ResilienceLayer

logger = logging.getLogger("cortex")


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
        self.memory = Memory()
        self.cortex = Cortex(profile=self.profile, memory=self.memory)
        self.tools = ToolRegistry()
        self.policy = PolicyEngine(self.memory)
        self.resilience = ResilienceLayer()
        self.start_time = 0.0
        self.request_count = 0

    async def start(self):
        """Start the daemon."""
        self.start_time = time.monotonic()

        # Boot the Cortex orchestrator (loads L0-L2 models via ModelManager)
        logger.info("Booting Cortex orchestrator...")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.cortex.boot)

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

        # Print loaded model status from Cortex's ModelManager
        mgr_status = self.cortex.manager.status()
        for m in mgr_status.get("models", []):
            icon = "●" if m["state"] == "ready" else "○"
            hot = " [HOT]" if not m["is_challenge"] and m["tier"] in ("L0","L1","L2") else ""
            print(f"  {icon} {m['tier']}: {m['model_id']}{hot}")
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
            elif path == "/v1/threads" and method == "GET":
                await self._handle_threads(writer, headers)
            elif path.startswith("/v1/threads/") and method == "GET":
                thread_id = path.split("/v1/threads/")[1].split("/")[0]
                if "/messages" in path:
                    await self._handle_thread_messages(writer, thread_id)
                else:
                    await self._handle_thread_detail(writer, thread_id)
            elif path == "/v1/usage" and method == "GET":
                await self._handle_usage(writer, headers)
            elif path == "/v1/audit" and method == "GET":
                await self._handle_audit(writer, headers)
            elif path == "/v1/memory/stats" and method == "GET":
                await self._handle_memory_stats(writer)
            elif path == "/v1/policies" and method == "POST":
                await self._handle_set_policy(writer, body)
            elif path == "/v1/policies" and method == "GET":
                await self._handle_get_policies(writer, headers)
            elif path == "/v1/models/census" and method == "GET":
                await self._handle_model_census(writer)
            elif path == "/v1/models/discover" and method == "POST":
                await self._handle_model_discover(writer)
            elif path == "/v1/tools" and method == "GET":
                await self._handle_tools_list(writer)
            elif path == "/v1/resilience" and method == "GET":
                await self._handle_resilience_status(writer)
            elif path == "/v1/resilience/reset" and method == "POST":
                await self._handle_resilience_reset(writer, body)
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
        Persists conversation history and audit log via Memory.
        """
        t0 = time.monotonic()
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

        # --- Memory: resolve thread ---
        thread_id = request_body.get("thread_id", request_body.get("metadata", {}).get("thread_id", ""))
        app_id = headers.get("x-app-id", request_body.get("metadata", {}).get("app_id", ""))

        # --- Policy check ---
        tool_names = [t.get("function", {}).get("name", "") for t in (normalized.tools or []) if t.get("type") == "function"]
        policy_ctx = PolicyContext(
            app_id=app_id,
            thread_id=thread_id,
            requested_model=normalized.model,
            max_tokens=normalized.max_tokens,
            has_tools=bool(normalized.tools),
            tool_names=tool_names,
        )
        policy_decision = self.policy.check(policy_ctx)
        if policy_decision.denied:
            logger.warning("Policy denied: %s", policy_decision.reason)
            await self._send_json(writer, 403, {
                "error": {"message": policy_decision.reason, "type": "policy_denied"}
            })
            return

        # Apply policy adjustments
        if policy_decision.effective_max_tokens < normalized.max_tokens:
            normalized.max_tokens = policy_decision.effective_max_tokens

        if thread_id:
            thread = self.memory.get_or_create_thread(
                thread_id, app_id=app_id, model_hint=normalized.model,
            )
        else:
            # Create an ephemeral thread for audit tracking
            thread = self.memory.create_thread(
                app_id=app_id, model_hint=normalized.model,
            )
            thread_id = thread.id

        # Persist inbound user message
        user_content = ""
        for msg in reversed(normalized.messages):
            if msg.get("role") == "user":
                user_content = msg.get("content", "")
                break
        if user_content:
            self.memory.add_message(thread_id, "user", user_content)

        # --- If thread has history, inject it into context ---
        if request_body.get("thread_id") or request_body.get("metadata", {}).get("thread_id"):
            ctx = self.memory.get_context_window(thread_id, max_tokens=normalized.max_tokens * 4)
            if len(ctx) > len(normalized.messages):
                normalized.messages = ctx

        stream = normalized.stream
        response_content = ""
        routed_tier = ""
        actual_model = ""
        category = ""
        confidence = 0.0
        tokens_prompt = 0
        tokens_completion = 0
        status_code = 200
        escalation_path: list[str] = []
        error_msg = ""

        if stream:
            # Streaming: resolve backend via Cortex, then stream via aiohttp
            try:
                import aiohttp

                loop = asyncio.get_event_loop()
                route, tier, adapter, model_tag = await loop.run_in_executor(
                    None,
                    lambda: self.cortex.resolve_backend(normalized.messages),
                )
                routed_tier = tier.name
                actual_model = model_tag
                category = route.category.value
                confidence = route.confidence

                if adapter is None:
                    await self._send_json(writer, 503, {
                        "error": {"message": "No backend available", "type": "server_error"}
                    })
                    return

                # Build streaming request to the Ollama backend
                forward_body = {
                    "model": model_tag,
                    "messages": normalized.messages,
                    "max_tokens": max(normalized.max_tokens, 256),
                    "temperature": normalized.temperature,
                    "stream": True,
                }
                if normalized.tools:
                    forward_body["tools"] = normalized.tools
                if TIER_SPECS[tier].always_hot:
                    forward_body["keep_alive"] = "24h"

                target_url = f"{adapter.base_url}/v1/chat/completions"

                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        target_url, json=forward_body,
                        timeout=aiohttp.ClientTimeout(total=120),
                    ) as resp:
                        header = (
                            f"HTTP/1.1 {resp.status} OK\r\n"
                            f"Content-Type: text/event-stream\r\n"
                            f"Cache-Control: no-cache\r\n"
                            f"Connection: close\r\n"
                            f"\r\n"
                        )
                        writer.write(header.encode())
                        async for chunk in resp.content:
                            writer.write(chunk)
                        await writer.drain()

            except Exception as e:
                logger.error("Streaming error: %s", e, exc_info=True)
                error_msg = str(e)
                await self._send_json(writer, 502, {
                    "error": {"message": f"Streaming error: {e}", "type": "server_error"}
                })
        else:
            # Non-streaming: full Cortex pipeline (route → generate → challenge → swarm)
            try:
                loop = asyncio.get_event_loop()
                cortex_resp: CortexResponse = await loop.run_in_executor(
                    None,
                    lambda: self.cortex.process(
                        normalized.messages,
                        max_tokens=normalized.max_tokens,
                    ),
                )

                response_content = cortex_resp.content
                routed_tier = cortex_resp.tier_used.name
                actual_model = cortex_resp.model_used
                category = cortex_resp.route_decision.category.value
                confidence = cortex_resp.confidence
                escalation_path = cortex_resp.escalation_path

                routing_meta = {
                    "tier": routed_tier,
                    "category": category,
                    "confidence": confidence,
                    "backend_model": actual_model,
                    "escalation_path": escalation_path,
                    "total_ms": round(cortex_resp.total_ms, 1),
                    "thread_id": thread_id,
                }

                formatted = format_response(
                    cortex_resp.content,
                    normalized,
                    routing_meta,
                    cortex_resp.model_used,
                    cortex_resp.total_ms,
                )
                await self._send_json(writer, 200, formatted)

            except Exception as e:
                logger.error("Cortex processing error: %s", e, exc_info=True)
                error_msg = str(e)
                status_code = 502
                await self._send_json(writer, 502, {
                    "error": {"message": f"Processing error: {e}", "type": "server_error"}
                })

        # --- Memory: persist response + audit ---
        latency_ms = (time.monotonic() - t0) * 1000

        if response_content:
            self.memory.add_message(
                thread_id, "assistant", response_content,
                model=actual_model, tier=routed_tier,
                tokens_prompt=tokens_prompt, tokens_completion=tokens_completion,
                latency_ms=latency_ms,
            )

        self.memory.log_request(
            thread_id=thread_id,
            request_model=normalized.model,
            routed_tier=routed_tier,
            actual_model=actual_model,
            category=category,
            confidence=confidence,
            tokens_prompt=tokens_prompt,
            tokens_completion=tokens_completion,
            latency_ms=latency_ms,
            ttft_ms=latency_ms,  # approximate; real TTFT needs streaming instrumentation
            status_code=status_code,
            app_id=app_id,
            escalation_path=escalation_path,
            error=error_msg,
        )

    async def _handle_models(self, writer):
        """Handle GET /v1/models — list ALL available models (loaded + discovered)."""
        models = []
        seen_ids = set()

        # Add models loaded in Cortex's ModelManager
        mgr_status = self.cortex.manager.status()
        for m in mgr_status.get("models", []):
            model_id = m["model_id"]
            if model_id not in seen_ids:
                models.append({
                    "id": model_id,
                    "object": "model",
                    "created": int(self.start_time),
                    "owned_by": f"local-{m['tier']}",
                    "meta": {
                        "tier": m["tier"],
                        "state": m["state"],
                        "family": m.get("family", ""),
                        "vram_mb": m.get("vram_mb", 0),
                    },
                })
                seen_ids.add(model_id)

        # Add all discovered Ollama models not already listed
        for tier in Tier:
            tier_models = get_models_for_tier(tier, self.profile)
            for m in tier_models:
                model_id = m.ollama_tag or m.model_id
                if model_id not in seen_ids:
                    models.append({
                        "id": model_id,
                        "object": "model",
                        "created": int(self.start_time),
                        "owned_by": f"local-{tier.name}",
                        "meta": {
                            "tier": tier.name,
                            "family": m.family,
                            "vram_mb": m.vram_mb,
                            "format": m.format,
                        },
                    })
                    seen_ids.add(model_id)

        await self._send_json(writer, 200, {
            "object": "list",
            "data": models,
        })

    async def _handle_health(self, writer):
        """Handle GET /health."""
        uptime = time.monotonic() - self.start_time
        mgr_status = self.cortex.manager.status()
        ready_count = sum(
            1 for m in mgr_status.get("models", [])
            if m["state"] == "ready"
        )
        await self._send_json(writer, 200, {
            "status": "ok",
            "uptime_seconds": round(uptime, 1),
            "total_requests": self.request_count,
            "models_ready": ready_count,
            "models_loaded": mgr_status.get("models_loaded", 0),
        })

    async def _handle_status(self, writer):
        """Handle GET /status — detailed Cortex + system status."""
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
            "cortex": self.cortex.status(),
            "max_local_tier": max_feasible_tier(self.profile).name,
        })

    # ------------------------------------------------------------------
    # Memory endpoints
    # ------------------------------------------------------------------

    async def _handle_threads(self, writer, headers: dict):
        """GET /v1/threads — list conversation threads."""
        app_id = headers.get("x-app-id", "")
        threads = self.memory.list_threads(app_id=app_id or None, limit=50)
        await self._send_json(writer, 200, {
            "object": "list",
            "data": [
                {
                    "id": t.id,
                    "title": t.title,
                    "app_id": t.app_id,
                    "model_hint": t.model_hint,
                    "message_count": t.message_count,
                    "total_tokens": t.total_tokens,
                    "created_at": t.created_at,
                    "updated_at": t.updated_at,
                }
                for t in threads
            ],
        })

    async def _handle_thread_detail(self, writer, thread_id: str):
        """GET /v1/threads/{id} — get thread with recent messages."""
        thread = self.memory.get_thread(thread_id)
        if thread is None:
            await self._send_json(writer, 404, {
                "error": {"message": f"Thread not found: {thread_id}", "type": "not_found"}
            })
            return
        messages = self.memory.get_messages(thread_id, limit=20)
        await self._send_json(writer, 200, {
            "id": thread.id,
            "title": thread.title,
            "app_id": thread.app_id,
            "model_hint": thread.model_hint,
            "message_count": thread.message_count,
            "total_tokens": thread.total_tokens,
            "created_at": thread.created_at,
            "updated_at": thread.updated_at,
            "messages": [
                {
                    "id": m.id,
                    "role": m.role,
                    "content": m.content[:500],
                    "model": m.model,
                    "tier": m.tier,
                    "latency_ms": m.latency_ms,
                    "created_at": m.created_at,
                }
                for m in messages
            ],
        })

    async def _handle_thread_messages(self, writer, thread_id: str):
        """GET /v1/threads/{id}/messages — full message history."""
        messages = self.memory.get_messages(thread_id)
        await self._send_json(writer, 200, {
            "object": "list",
            "thread_id": thread_id,
            "data": [
                {
                    "id": m.id,
                    "role": m.role,
                    "content": m.content,
                    "model": m.model,
                    "tier": m.tier,
                    "tokens_prompt": m.tokens_prompt,
                    "tokens_completion": m.tokens_completion,
                    "latency_ms": m.latency_ms,
                    "created_at": m.created_at,
                }
                for m in messages
            ],
        })

    async def _handle_usage(self, writer, headers: dict):
        """GET /v1/usage — usage stats for the last 7 days."""
        usage = self.memory.get_usage_summary(days=7)
        await self._send_json(writer, 200, usage)

    async def _handle_audit(self, writer, headers: dict):
        """GET /v1/audit — recent audit log entries."""
        entries = self.memory.get_audit_log(limit=50)
        await self._send_json(writer, 200, {
            "object": "list",
            "data": [
                {
                    "id": e.id,
                    "thread_id": e.thread_id,
                    "request_model": e.request_model,
                    "routed_tier": e.routed_tier,
                    "actual_model": e.actual_model,
                    "category": e.category,
                    "confidence": e.confidence,
                    "tokens_prompt": e.tokens_prompt,
                    "tokens_completion": e.tokens_completion,
                    "latency_ms": round(e.latency_ms, 1),
                    "ttft_ms": round(e.ttft_ms, 1),
                    "status_code": e.status_code,
                    "app_id": e.app_id,
                    "error": e.error,
                    "created_at": e.created_at,
                }
                for e in entries
            ],
        })

    async def _handle_memory_stats(self, writer):
        """GET /v1/memory/stats — database statistics."""
        stats = self.memory.db_stats()
        await self._send_json(writer, 200, stats)

    async def _handle_set_policy(self, writer, body: bytes):
        """POST /v1/policies — set a policy/config value."""
        try:
            data = json.loads(body)
        except json.JSONDecodeError as e:
            await self._send_json(writer, 400, {
                "error": {"message": f"Invalid JSON: {e}", "type": "invalid_request"}
            })
            return
        key = data.get("key", "")
        value = data.get("value")
        scope = data.get("scope", "global")
        if not key:
            await self._send_json(writer, 400, {
                "error": {"message": "Missing 'key' field", "type": "invalid_request"}
            })
            return
        self.memory.set_policy(key, value, scope=scope)
        await self._send_json(writer, 200, {"status": "ok", "key": key, "scope": scope})

    async def _handle_get_policies(self, writer, headers: dict):
        """GET /v1/policies — list all policies."""
        rows = self.memory._conn.execute(
            "SELECT key, scope, value, updated_at FROM policies ORDER BY scope, key"
        ).fetchall()
        await self._send_json(writer, 200, {
            "object": "list",
            "data": [
                {
                    "key": r["key"],
                    "scope": r["scope"],
                    "value": json.loads(r["value"]),
                    "updated_at": r["updated_at"],
                }
                for r in rows
            ],
        })

    # ------------------------------------------------------------------
    # Tools & Resilience endpoints
    # ------------------------------------------------------------------

    async def _handle_model_census(self, writer):
        """GET /v1/models/census — full census of all available models."""
        census = model_census(self.profile)
        await self._send_json(writer, 200, census)

    async def _handle_model_discover(self, writer):
        """POST /v1/models/discover — re-scan Ollama for new models."""
        refresh_ollama_discovery()
        census = model_census(self.profile)
        await self._send_json(writer, 200, {
            "status": "ok",
            "message": f"Discovered {census['total_unique_models']} models across {census['total_families']} families",
            "census": census,
        })

    async def _handle_tools_list(self, writer):
        """GET /v1/tools — list all registered tools."""
        await self._send_json(writer, 200, self.tools.status())

    async def _handle_resilience_status(self, writer):
        """GET /v1/resilience — circuit breaker status."""
        await self._send_json(writer, 200, self.resilience.status())

    async def _handle_resilience_reset(self, writer, body: bytes):
        """POST /v1/resilience/reset — reset circuit breakers."""
        try:
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            data = {}
        name = data.get("circuit", "")
        if name:
            ok = self.resilience.reset_circuit(name)
            await self._send_json(writer, 200, {"status": "ok" if ok else "not_found", "circuit": name})
        else:
            count = self.resilience.reset_all()
            await self._send_json(writer, 200, {"status": "ok", "circuits_reset": count})

    async def _send_json(self, writer, status: int, data):
        """Send a JSON HTTP response."""
        body = json.dumps(data).encode()
        status_text = {200: "OK", 400: "Bad Request", 403: "Forbidden",
                       404: "Not Found", 500: "Internal Server Error",
                       502: "Bad Gateway", 503: "Service Unavailable"}.get(status, "Unknown")
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
        sig_name = signal.Signals(sig).name if isinstance(sig, int) else sig.name
        logger.info("Received %s, shutting down...", sig_name)
        daemon.memory.close()
        loop.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        loop.run_until_complete(daemon.start())
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        daemon.memory.close()
