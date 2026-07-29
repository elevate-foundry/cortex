"""
Cortex Tool Registry — function calling broker with permission model.

This is the "device driver + permission system" layer of the agent microkernel.
Tools are capabilities that models can invoke. The registry:

  - Registers tools (built-in, user-defined, MCP server tools)
  - Validates tool calls against permission levels
  - Executes tool calls in a controlled sandbox
  - Audits every tool invocation
  - Enforces per-app and per-thread permission scopes

Permission model (privilege rings):

  Ring 0 — READ_ONLY:    No side effects. Info retrieval only.
                         Examples: web_search, get_weather, lookup
  Ring 1 — DRAFT:        Creates artifacts but doesn't execute them.
                         Examples: code_interpreter (sandbox), write_file (staged)
  Ring 2 — EXECUTE:      Local side effects. Modifies local state.
                         Examples: run_command, edit_file, git_commit
  Ring 3 — EXTERNAL:     External side effects. Network mutations.
                         Examples: send_email, post_to_api, deploy
  Ring 4 — DANGEROUS:    Destructive / irreversible.
                         Examples: delete_database, rm_rf, transfer_funds

Each tool declares its ring. Each request context has a max_ring.
A tool call is allowed only if tool.ring <= context.max_ring.
"""

import asyncio
import json
import logging
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Coroutine, Optional, Union

logger = logging.getLogger("cortex.tools")


# ---------------------------------------------------------------------------
# Permission rings
# ---------------------------------------------------------------------------

class PermissionRing(IntEnum):
    """Privilege rings, lower = safer."""
    READ_ONLY = 0
    DRAFT = 1
    EXECUTE = 2
    EXTERNAL = 3
    DANGEROUS = 4


# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------

@dataclass
class ToolDef:
    """
    A registered tool definition.
    
    Maps to OpenAI function calling format but adds:
      - permission ring
      - execution handler
      - rate limit
      - audit flag
    """
    name: str
    description: str
    parameters: dict                          # JSON Schema
    ring: PermissionRing = PermissionRing.READ_ONLY
    handler: Optional[Callable] = None        # sync or async callable
    enabled: bool = True
    rate_limit: int = 0                       # max calls per minute (0=unlimited)
    requires_approval: bool = False           # require user confirmation before executing
    tags: list[str] = field(default_factory=list)  # e.g. ["builtin", "mcp", "user"]

    def to_openai_tool(self) -> dict:
        """Convert to OpenAI function-calling tool format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_registry_dict(self) -> dict:
        """Full metadata for the /v1/tools endpoint."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "ring": self.ring.name,
            "ring_level": self.ring.value,
            "enabled": self.enabled,
            "rate_limit": self.rate_limit,
            "requires_approval": self.requires_approval,
            "tags": self.tags,
        }


@dataclass
class ToolCall:
    """A single tool invocation."""
    id: str
    name: str
    arguments: dict
    ring: PermissionRing = PermissionRing.READ_ONLY


@dataclass
class ToolResult:
    """Result of executing a tool call."""
    tool_call_id: str
    name: str
    output: str
    success: bool = True
    error: str = ""
    latency_ms: float = 0.0
    ring: PermissionRing = PermissionRing.READ_ONLY
    blocked: bool = False
    block_reason: str = ""


# ---------------------------------------------------------------------------
# Built-in tool handlers
# ---------------------------------------------------------------------------

async def _handler_get_time(**kwargs) -> str:
    """Get current date/time."""
    import datetime
    fmt = kwargs.get("format", "%Y-%m-%d %H:%M:%S")
    return datetime.datetime.now().strftime(fmt)


async def _handler_calculate(**kwargs) -> str:
    """Evaluate a math expression safely."""
    expr = kwargs.get("expression", "")
    # Only allow safe math operations
    allowed = set("0123456789+-*/.() eE")
    if not all(c in allowed for c in expr):
        return f"Error: unsafe expression '{expr}'"
    try:
        result = eval(expr, {"__builtins__": {}}, {})  # noqa: S307
        return str(result)
    except Exception as e:
        return f"Error: {e}"


async def _handler_shell(**kwargs) -> str:
    """Execute a shell command (Ring 2: EXECUTE)."""
    cmd = kwargs.get("command", "")
    timeout = min(kwargs.get("timeout", 30), 60)
    try:
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=timeout,
        )
        output = stdout.decode(errors="replace")
        if stderr:
            output += "\nSTDERR:\n" + stderr.decode(errors="replace")
        return output[:10000]  # cap output
    except asyncio.TimeoutError:
        return f"Error: command timed out after {timeout}s"
    except Exception as e:
        return f"Error: {e}"


async def _handler_read_file(**kwargs) -> str:
    """Read a file's contents (Ring 0: READ_ONLY)."""
    import os
    path = kwargs.get("path", "")
    if not os.path.isfile(path):
        return f"Error: file not found: {path}"
    try:
        with open(path, "r") as f:
            content = f.read(50000)  # cap at 50k chars
        return content
    except Exception as e:
        return f"Error: {e}"


async def _handler_write_file(**kwargs) -> str:
    """Write content to a file (Ring 2: EXECUTE)."""
    path = kwargs.get("path", "")
    content = kwargs.get("content", "")
    try:
        with open(path, "w") as f:
            f.write(content)
        return f"Wrote {len(content)} chars to {path}"
    except Exception as e:
        return f"Error: {e}"


async def _handler_web_search(**kwargs) -> str:
    """Stub for web search (Ring 0: READ_ONLY). Override with real implementation."""
    query = kwargs.get("query", "")
    return f"[web_search stub] No results for: {query}. Connect a search provider."


# ---------------------------------------------------------------------------
# Tool Registry
# ---------------------------------------------------------------------------

class ToolRegistry:
    """
    Central tool registry for the Cortex kernel.
    
    Usage:
        registry = ToolRegistry()
        registry.register(tool_def)
        result = await registry.execute(tool_call, max_ring=PermissionRing.DRAFT)
    """

    def __init__(self):
        self._tools: dict[str, ToolDef] = {}
        self._call_counts: dict[str, list[float]] = {}  # name -> [timestamps]
        self._register_builtins()

    def _register_builtins(self):
        """Register built-in tools."""
        builtins = [
            ToolDef(
                name="get_time",
                description="Get the current date and time",
                parameters={
                    "type": "object",
                    "properties": {
                        "format": {
                            "type": "string",
                            "description": "strftime format string",
                            "default": "%Y-%m-%d %H:%M:%S",
                        },
                    },
                },
                ring=PermissionRing.READ_ONLY,
                handler=_handler_get_time,
                tags=["builtin"],
            ),
            ToolDef(
                name="calculate",
                description="Evaluate a mathematical expression",
                parameters={
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Math expression to evaluate (e.g. '2+2', '3.14*r**2')",
                        },
                    },
                    "required": ["expression"],
                },
                ring=PermissionRing.READ_ONLY,
                handler=_handler_calculate,
                tags=["builtin"],
            ),
            ToolDef(
                name="read_file",
                description="Read the contents of a file",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Absolute file path"},
                    },
                    "required": ["path"],
                },
                ring=PermissionRing.READ_ONLY,
                handler=_handler_read_file,
                tags=["builtin"],
            ),
            ToolDef(
                name="write_file",
                description="Write content to a file",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Absolute file path"},
                        "content": {"type": "string", "description": "Content to write"},
                    },
                    "required": ["path", "content"],
                },
                ring=PermissionRing.EXECUTE,
                handler=_handler_write_file,
                tags=["builtin"],
            ),
            ToolDef(
                name="shell",
                description="Execute a shell command",
                parameters={
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Shell command to execute"},
                        "timeout": {"type": "integer", "description": "Timeout in seconds (max 60)", "default": 30},
                    },
                    "required": ["command"],
                },
                ring=PermissionRing.EXECUTE,
                handler=_handler_shell,
                tags=["builtin"],
            ),
            ToolDef(
                name="web_search",
                description="Search the web for information",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                    },
                    "required": ["query"],
                },
                ring=PermissionRing.READ_ONLY,
                handler=_handler_web_search,
                tags=["builtin"],
            ),
        ]
        for t in builtins:
            self._tools[t.name] = t

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, tool: ToolDef) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool
        logger.info("Tool registered: %s (ring=%s)", tool.name, tool.ring.name)

    def unregister(self, name: str) -> bool:
        """Unregister a tool."""
        if name in self._tools:
            del self._tools[name]
            return True
        return False

    def get(self, name: str) -> Optional[ToolDef]:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(
        self,
        max_ring: Optional[PermissionRing] = None,
        tags: Optional[list[str]] = None,
        enabled_only: bool = True,
    ) -> list[ToolDef]:
        """List tools, optionally filtered by ring and tags."""
        result = []
        for t in self._tools.values():
            if enabled_only and not t.enabled:
                continue
            if max_ring is not None and t.ring > max_ring:
                continue
            if tags and not any(tag in t.tags for tag in tags):
                continue
            result.append(t)
        return sorted(result, key=lambda t: (t.ring, t.name))

    def get_openai_tools(
        self, max_ring: Optional[PermissionRing] = None
    ) -> list[dict]:
        """Get tools in OpenAI function-calling format, filtered by permission."""
        return [t.to_openai_tool() for t in self.list_tools(max_ring=max_ring)]

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def execute(
        self,
        call: ToolCall,
        max_ring: PermissionRing = PermissionRing.DRAFT,
    ) -> ToolResult:
        """
        Execute a tool call with permission checking.
        
        Args:
            call: The tool call to execute
            max_ring: Maximum allowed permission ring for this context
        """
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                output="",
                success=False,
                error=f"Unknown tool: {call.name}",
            )

        if not tool.enabled:
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                output="",
                success=False,
                error=f"Tool '{call.name}' is disabled",
                blocked=True,
                block_reason="disabled",
            )

        # Permission check
        if tool.ring > max_ring:
            logger.warning(
                "Tool call BLOCKED: %s requires ring %s, context allows %s",
                call.name, tool.ring.name, max_ring.name,
            )
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                output="",
                success=False,
                error=f"Permission denied: '{call.name}' requires {tool.ring.name} "
                      f"(ring {tool.ring.value}), but context only allows "
                      f"{max_ring.name} (ring {max_ring.value})",
                ring=tool.ring,
                blocked=True,
                block_reason=f"ring:{tool.ring.name}>{max_ring.name}",
            )

        # Rate limit check
        if tool.rate_limit > 0:
            now = time.monotonic()
            timestamps = self._call_counts.setdefault(call.name, [])
            # Prune old timestamps (older than 60s)
            timestamps[:] = [t for t in timestamps if now - t < 60]
            if len(timestamps) >= tool.rate_limit:
                return ToolResult(
                    tool_call_id=call.id,
                    name=call.name,
                    output="",
                    success=False,
                    error=f"Rate limit exceeded: {tool.rate_limit}/min for '{call.name}'",
                    blocked=True,
                    block_reason="rate_limit",
                )
            timestamps.append(now)

        # Execute
        if tool.handler is None:
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                output="",
                success=False,
                error=f"Tool '{call.name}' has no handler",
            )

        t0 = time.monotonic()
        try:
            if asyncio.iscoroutinefunction(tool.handler):
                output = await tool.handler(**call.arguments)
            else:
                loop = asyncio.get_event_loop()
                output = await loop.run_in_executor(
                    None, lambda: tool.handler(**call.arguments)
                )
            latency = (time.monotonic() - t0) * 1000

            logger.info(
                "Tool executed: %s (ring=%s, %.1fms)",
                call.name, tool.ring.name, latency,
            )
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                output=str(output),
                success=True,
                latency_ms=latency,
                ring=tool.ring,
            )
        except Exception as e:
            latency = (time.monotonic() - t0) * 1000
            logger.error("Tool execution failed: %s: %s", call.name, e)
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                output="",
                success=False,
                error=str(e),
                latency_ms=latency,
                ring=tool.ring,
            )

    async def execute_batch(
        self,
        calls: list[ToolCall],
        max_ring: PermissionRing = PermissionRing.DRAFT,
        parallel: bool = True,
    ) -> list[ToolResult]:
        """Execute multiple tool calls, optionally in parallel."""
        if parallel:
            tasks = [self.execute(c, max_ring) for c in calls]
            return await asyncio.gather(*tasks)
        else:
            results = []
            for c in calls:
                results.append(await self.execute(c, max_ring))
            return results

    # ------------------------------------------------------------------
    # Helpers for parsing model tool call responses
    # ------------------------------------------------------------------

    @staticmethod
    def parse_tool_calls(message: dict) -> list[ToolCall]:
        """Parse tool calls from an OpenAI-format assistant message."""
        calls = []
        for tc in message.get("tool_calls", []):
            fn = tc.get("function", {})
            args = fn.get("arguments", "{}")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"_raw": args}
            calls.append(ToolCall(
                id=tc.get("id", str(uuid.uuid4())),
                name=fn.get("name", ""),
                arguments=args,
            ))
        return calls

    @staticmethod
    def results_to_messages(results: list[ToolResult]) -> list[dict]:
        """Convert tool results back to OpenAI chat messages for the next turn."""
        return [
            {
                "role": "tool",
                "tool_call_id": r.tool_call_id,
                "content": r.output if r.success else f"Error: {r.error}",
            }
            for r in results
        ]

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> dict:
        return {
            "total_tools": len(self._tools),
            "enabled": sum(1 for t in self._tools.values() if t.enabled),
            "by_ring": {
                ring.name: sum(1 for t in self._tools.values() if t.ring == ring)
                for ring in PermissionRing
            },
            "tools": [t.to_registry_dict() for t in self.list_tools(enabled_only=False)],
        }
