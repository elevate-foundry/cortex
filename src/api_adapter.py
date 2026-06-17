"""
Cortex API adapter layer — translate diverse client API formats into
Cortex's internal CompletionRequest/CompletionResponse.

Problem:
  Apps using OpenAI-compatible Chat Completions "just work" through the proxy.
  Apps using newer/richer APIs need translation:

  1. OpenAI Responses API (POST /v1/responses)
     - input[] items instead of messages[]
     - output[] with output_text, tool calls, reasoning
     - built-in tool definitions (web_search, code_interpreter, etc.)
     - response.output_text shorthand

  2. Tool calling / function calling
     - tools[] with type=function, function={name, description, parameters}
     - tool_choice: "auto" | "required" | "none" | {type, function}
     - parallel_tool_calls
     - Streaming: tool call deltas across chunks

  3. Anthropic Messages API (POST /v1/messages)
     - content[] blocks instead of single string
     - tool_use / tool_result content blocks
     - system as top-level string (not a message)
     - different stop_reason values

  4. Multimodal (vision, audio)
     - content as array: [{type: "text", text: "..."}, {type: "image_url", ...}]
     - image_url with base64 or URL
     - audio content parts

  5. Computer-use / MCP
     - computer_use_preview tool type
     - MCP server tool references

Strategy:
  - Normalize everything INTO Chat Completions messages[] + tools[]
  - Route through Cortex normally
  - Translate the response BACK to the client's expected format
"""

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class APIFormat(str, Enum):
    CHAT_COMPLETIONS = "chat_completions"   # POST /v1/chat/completions
    RESPONSES = "responses"                  # POST /v1/responses
    ANTHROPIC = "anthropic"                  # POST /v1/messages (Anthropic format)


# ---------------------------------------------------------------------------
# Normalized internal representation
# ---------------------------------------------------------------------------

@dataclass
class NormalizedRequest:
    """
    API-agnostic request that Cortex can process.
    Everything gets normalized to this before routing.
    """
    messages: list[dict]                      # OpenAI chat format
    model: str = ""
    max_tokens: int = 512
    temperature: float = 0.0
    stream: bool = False
    stop: Optional[list[str]] = None
    tools: Optional[list[dict]] = None        # OpenAI function-calling format
    tool_choice: Any = None                   # "auto" | "required" | "none" | dict
    source_format: APIFormat = APIFormat.CHAT_COMPLETIONS
    source_request: Optional[dict] = None     # original request for reference
    extra: dict = field(default_factory=dict)  # pass-through fields


# ===================================================================
# INBOUND: Client request → NormalizedRequest
# ===================================================================

def normalize_request(body: dict, path: str = "") -> NormalizedRequest:
    """
    Detect the API format from the request body/path and normalize it.
    """
    if path.endswith("/responses"):
        return _normalize_responses_api(body)
    if path.endswith("/messages"):
        return _normalize_anthropic(body)
    # Default: Chat Completions (already normalized)
    return _normalize_chat_completions(body)


def _normalize_chat_completions(body: dict) -> NormalizedRequest:
    """
    OpenAI Chat Completions — already the native format.
    Just extract tool-calling fields and multimodal content.
    """
    messages = body.get("messages", [])
    # Normalize multimodal content blocks to text for local models
    normalized_msgs = [_normalize_message_content(m) for m in messages]

    return NormalizedRequest(
        messages=normalized_msgs,
        model=body.get("model", ""),
        max_tokens=body.get("max_tokens", 512),
        temperature=body.get("temperature", 0.0),
        stream=body.get("stream", False),
        stop=body.get("stop"),
        tools=body.get("tools"),
        tool_choice=body.get("tool_choice"),
        source_format=APIFormat.CHAT_COMPLETIONS,
        source_request=body,
    )


def _normalize_responses_api(body: dict) -> NormalizedRequest:
    """
    OpenAI Responses API → Chat Completions messages.

    Responses API shape:
      {
        "model": "...",
        "input": "string" | [
          {"type": "message", "role": "user", "content": "..."},
          {"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": "..."},
            {"type": "input_image", "image_url": "..."},
          ]},
        ],
        "instructions": "system prompt",
        "tools": [{"type": "web_search"}, {"type": "function", ...}],
        "temperature": 0.7,
        "max_output_tokens": 1024,
      }
    """
    messages: list[dict] = []

    # System prompt from "instructions"
    instructions = body.get("instructions")
    if instructions:
        messages.append({"role": "system", "content": instructions})

    # Parse input
    raw_input = body.get("input", "")
    if isinstance(raw_input, str):
        messages.append({"role": "user", "content": raw_input})
    elif isinstance(raw_input, list):
        for item in raw_input:
            if isinstance(item, str):
                messages.append({"role": "user", "content": item})
            elif isinstance(item, dict):
                msg = _responses_item_to_message(item)
                if msg:
                    messages.append(msg)

    # Translate tools — filter out built-in tools the local model can't use
    tools, builtin_tools = _translate_responses_tools(body.get("tools", []))

    return NormalizedRequest(
        messages=messages,
        model=body.get("model", ""),
        max_tokens=body.get("max_output_tokens", 512),
        temperature=body.get("temperature", 0.0),
        stream=body.get("stream", False),
        tools=tools if tools else None,
        source_format=APIFormat.RESPONSES,
        source_request=body,
        extra={"builtin_tools": builtin_tools},
    )


def _normalize_anthropic(body: dict) -> NormalizedRequest:
    """
    Anthropic Messages API → Chat Completions messages.

    Anthropic shape:
      {
        "model": "...",
        "system": "system prompt",
        "messages": [
          {"role": "user", "content": "string" | [
            {"type": "text", "text": "..."},
            {"type": "image", "source": {"type": "base64", ...}},
            {"type": "tool_use", "id": "...", "name": "...", "input": {...}},
            {"type": "tool_result", "tool_use_id": "...", "content": "..."},
          ]},
        ],
        "max_tokens": 1024,
        "tools": [{"name": "...", "description": "...", "input_schema": {...}}],
      }
    """
    messages: list[dict] = []

    # System prompt
    system = body.get("system")
    if system:
        messages.append({"role": "system", "content": system})

    # Translate messages
    for msg in body.get("messages", []):
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if isinstance(content, str):
            messages.append({"role": role, "content": content})
        elif isinstance(content, list):
            # Anthropic content blocks → flatten to text + tool calls
            text_parts = []
            tool_calls = []
            tool_results = []

            for block in content:
                btype = block.get("type", "")
                if btype == "text":
                    text_parts.append(block.get("text", ""))
                elif btype == "image":
                    # Convert Anthropic image to OpenAI image_url format
                    source = block.get("source", {})
                    if source.get("type") == "base64":
                        media = source.get("media_type", "image/png")
                        data = source.get("data", "")
                        text_parts.append(f"[image: {media}, {len(data)} bytes base64]")
                    elif source.get("type") == "url":
                        text_parts.append(f"[image: {source.get('url', '')}]")
                elif btype == "tool_use":
                    tool_calls.append({
                        "id": block.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                        "type": "function",
                        "function": {
                            "name": block.get("name", ""),
                            "arguments": json.dumps(block.get("input", {})),
                        },
                    })
                elif btype == "tool_result":
                    tool_results.append({
                        "role": "tool",
                        "tool_call_id": block.get("tool_use_id", ""),
                        "content": block.get("content", ""),
                    })

            if role == "assistant" and tool_calls:
                messages.append({
                    "role": "assistant",
                    "content": "\n".join(text_parts) if text_parts else None,
                    "tool_calls": tool_calls,
                })
            elif tool_results:
                for tr in tool_results:
                    messages.append(tr)
            else:
                messages.append({
                    "role": role,
                    "content": "\n".join(text_parts) if text_parts else "",
                })

    # Translate Anthropic tools → OpenAI tools
    tools = None
    anthropic_tools = body.get("tools", [])
    if anthropic_tools:
        tools = []
        for tool in anthropic_tools:
            tools.append({
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {}),
                },
            })

    return NormalizedRequest(
        messages=messages,
        model=body.get("model", ""),
        max_tokens=body.get("max_tokens", 512),
        temperature=body.get("temperature", 0.0),
        stream=body.get("stream", False),
        tools=tools,
        source_format=APIFormat.ANTHROPIC,
        source_request=body,
    )


# ===================================================================
# OUTBOUND: Cortex response → client format
# ===================================================================

def format_response(
    content: str,
    normalized: NormalizedRequest,
    routing_meta: Optional[dict] = None,
    model_used: str = "",
    total_ms: float = 0.0,
) -> dict:
    """
    Convert Cortex's response back to the client's expected format.
    """
    if normalized.source_format == APIFormat.RESPONSES:
        return _format_responses_api(content, normalized, routing_meta, model_used)
    if normalized.source_format == APIFormat.ANTHROPIC:
        return _format_anthropic(content, normalized, routing_meta, model_used)
    # Default: Chat Completions
    return _format_chat_completions(content, normalized, routing_meta, model_used)


def _format_chat_completions(
    content: str,
    normalized: NormalizedRequest,
    routing_meta: Optional[dict] = None,
    model_used: str = "",
) -> dict:
    """Standard OpenAI Chat Completions response."""
    resp = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_used or normalized.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }
    if routing_meta:
        resp["_routing"] = routing_meta
    return resp


def _format_responses_api(
    content: str,
    normalized: NormalizedRequest,
    routing_meta: Optional[dict] = None,
    model_used: str = "",
) -> dict:
    """
    OpenAI Responses API response format.

    Response shape:
      {
        "id": "resp_...",
        "object": "response",
        "status": "completed",
        "output": [
          {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "..."}]
          }
        ],
        "output_text": "...",  # convenience shorthand
        "model": "...",
        "usage": {...}
      }
    """
    resp_id = f"resp_{uuid.uuid4().hex[:16]}"
    msg_id = f"msg_{uuid.uuid4().hex[:16]}"

    output_message = {
        "type": "message",
        "id": msg_id,
        "role": "assistant",
        "status": "completed",
        "content": [
            {
                "type": "output_text",
                "text": content,
            }
        ],
    }

    resp = {
        "id": resp_id,
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "model": model_used or normalized.model,
        "output": [output_message],
        "output_text": content,
        "usage": {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        },
    }
    if routing_meta:
        resp["_routing"] = routing_meta
    return resp


def _format_anthropic(
    content: str,
    normalized: NormalizedRequest,
    routing_meta: Optional[dict] = None,
    model_used: str = "",
) -> dict:
    """
    Anthropic Messages API response format.

    Response shape:
      {
        "id": "msg_...",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "..."}],
        "model": "...",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 0, "output_tokens": 0}
      }
    """
    resp = {
        "id": f"msg_{uuid.uuid4().hex[:16]}",
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": content,
            }
        ],
        "model": model_used or normalized.model,
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": 0,
            "output_tokens": 0,
        },
    }
    if routing_meta:
        resp["_routing"] = routing_meta
    return resp


# ===================================================================
# Helpers
# ===================================================================

def _normalize_message_content(msg: dict) -> dict:
    """
    Handle multimodal content in a message.
    If content is an array of parts, extract text and describe non-text parts.
    Local models typically can't process images — describe them instead.
    """
    content = msg.get("content")
    if content is None or isinstance(content, str):
        return msg

    if isinstance(content, list):
        text_parts = []
        for part in content:
            ptype = part.get("type", "")
            if ptype == "text":
                text_parts.append(part.get("text", ""))
            elif ptype == "input_text":
                text_parts.append(part.get("text", ""))
            elif ptype == "image_url":
                url = part.get("image_url", {}).get("url", "")
                if url.startswith("data:"):
                    text_parts.append("[image: base64-encoded image provided]")
                else:
                    text_parts.append(f"[image: {url}]")
            elif ptype == "input_image":
                text_parts.append(f"[image: {part.get('image_url', 'provided')}]")
            elif ptype == "input_audio":
                text_parts.append("[audio content provided]")
            elif ptype == "input_file":
                text_parts.append(f"[file: {part.get('filename', 'provided')}]")
            else:
                text_parts.append(f"[{ptype} content]")

        return {**msg, "content": "\n".join(text_parts)}

    return msg


def _responses_item_to_message(item: dict) -> Optional[dict]:
    """Convert a Responses API input item to a Chat Completions message."""
    itype = item.get("type", "")

    if itype == "message":
        role = item.get("role", "user")
        content = item.get("content", "")
        # Content can be string or array of content parts
        if isinstance(content, list):
            text_parts = []
            for part in content:
                ptype = part.get("type", "")
                if ptype in ("input_text", "text"):
                    text_parts.append(part.get("text", ""))
                elif ptype in ("input_image", "image_url"):
                    text_parts.append(f"[image: {part.get('image_url', 'provided')}]")
                elif ptype == "input_audio":
                    text_parts.append("[audio content]")
                elif ptype == "input_file":
                    text_parts.append(f"[file: {part.get('filename', 'provided')}]")
                else:
                    text_parts.append(f"[{ptype}]")
            content = "\n".join(text_parts)
        return {"role": role, "content": content}

    if itype == "item_reference":
        # Reference to a previous response item — can't resolve locally
        return {"role": "user", "content": f"[reference to item: {item.get('id', '')}]"}

    return None


def _translate_responses_tools(
    tools: list[dict],
) -> tuple[Optional[list[dict]], list[str]]:
    """
    Translate Responses API tools to OpenAI function-calling format.

    Responses API supports built-in tools that local models can't run:
      - web_search / web_search_preview
      - code_interpreter
      - computer_use_preview
      - mcp (Model Context Protocol servers)
      - file_search

    These get noted but stripped. Custom function tools pass through.

    Returns: (openai_tools, list_of_builtin_tool_names)
    """
    openai_tools: list[dict] = []
    builtin_names: list[str] = []

    for tool in tools:
        ttype = tool.get("type", "")

        if ttype == "function":
            # Custom function tool — pass through
            openai_tools.append({
                "type": "function",
                "function": tool.get("function", tool),
            })

        elif ttype in ("web_search", "web_search_preview"):
            builtin_names.append("web_search")

        elif ttype == "code_interpreter":
            builtin_names.append("code_interpreter")

        elif ttype in ("computer_use", "computer_use_preview"):
            builtin_names.append("computer_use")

        elif ttype == "mcp":
            builtin_names.append(f"mcp:{tool.get('server_label', 'unknown')}")

        elif ttype == "file_search":
            builtin_names.append("file_search")

        else:
            builtin_names.append(f"unknown:{ttype}")

    return openai_tools if openai_tools else None, builtin_names


# ===================================================================
# API format detection
# ===================================================================

def detect_api_format(body: dict, path: str = "") -> APIFormat:
    """
    Detect which API format a request is using.
    Used by the daemon to route to the correct handler.
    """
    # Path-based detection
    if "/responses" in path:
        return APIFormat.RESPONSES
    if "/messages" in path:
        return APIFormat.ANTHROPIC

    # Content-based heuristics
    if "input" in body and "messages" not in body:
        return APIFormat.RESPONSES
    if "system" in body and isinstance(body.get("system"), str) and "messages" in body:
        # Anthropic uses system as a top-level string
        # OpenAI uses system as a message role
        msgs = body.get("messages", [])
        if msgs and isinstance(msgs[0].get("content"), list):
            return APIFormat.ANTHROPIC

    return APIFormat.CHAT_COMPLETIONS
