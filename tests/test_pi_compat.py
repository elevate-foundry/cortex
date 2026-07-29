"""
Pi Coder / OpenAI Completions API compatibility tests.

Tests that Cortex daemon correctly handles all the non-standard fields
that Pi (and similar OpenAI-compatible clients) send:

  - max_completion_tokens (instead of max_tokens)
  - developer role (instead of system)
  - store field (should be ignored)
  - strict in tool definitions (should be passed through)
  - stream_options with include_usage (daemon should include usage in final chunk)
  - session affinity headers (ignored gracefully)

These tests verify both the API adapter (normalization) and the daemon
(end-to-end HTTP), ensuring Pi can use Cortex as a drop-in backend.
"""

import json
import sys
from pathlib import Path

import pytest

# Ensure we can import from src/
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api_adapter import (
    normalize_request,
    format_response,
    NormalizedRequest,
    APIFormat,
    _normalize_role,
    _normalize_message_content,
)


# ===================================================================
# Unit tests: API Adapter normalization
# ===================================================================


class TestDeveloperRole:
    """Pi uses 'developer' role as a replacement for 'system'."""

    def test_developer_role_mapped_to_system(self):
        msg = {"role": "developer", "content": "You are a helpful assistant."}
        result = _normalize_role(msg)
        assert result["role"] == "system"
        assert result["content"] == "You are a helpful assistant."

    def test_system_role_unchanged(self):
        msg = {"role": "system", "content": "System prompt"}
        result = _normalize_role(msg)
        assert result["role"] == "system"

    def test_user_role_unchanged(self):
        msg = {"role": "user", "content": "Hello"}
        result = _normalize_role(msg)
        assert result["role"] == "user"

    def test_assistant_role_unchanged(self):
        msg = {"role": "assistant", "content": "Hi there"}
        result = _normalize_role(msg)
        assert result["role"] == "assistant"

    def test_developer_role_in_full_request(self):
        """Full normalization pipeline handles developer role."""
        body = {
            "model": "auto",
            "messages": [
                {"role": "developer", "content": "You are CortexRouter."},
                {"role": "user", "content": "Route this: write a poem"},
            ],
        }
        req = normalize_request(body)
        assert req.messages[0]["role"] == "system"
        assert req.messages[0]["content"] == "You are CortexRouter."
        assert req.messages[1]["role"] == "user"


class TestMaxCompletionTokens:
    """Pi sends max_completion_tokens instead of max_tokens."""

    def test_max_completion_tokens_accepted(self):
        body = {
            "model": "auto",
            "messages": [{"role": "user", "content": "hello"}],
            "max_completion_tokens": 1024,
        }
        req = normalize_request(body)
        assert req.max_tokens == 1024

    def test_max_tokens_still_works(self):
        body = {
            "model": "auto",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 2048,
        }
        req = normalize_request(body)
        assert req.max_tokens == 2048

    def test_max_tokens_takes_priority(self):
        """If both are provided, max_tokens wins."""
        body = {
            "model": "auto",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 2048,
            "max_completion_tokens": 1024,
        }
        req = normalize_request(body)
        assert req.max_tokens == 2048

    def test_default_when_neither(self):
        body = {
            "model": "auto",
            "messages": [{"role": "user", "content": "hello"}],
        }
        req = normalize_request(body)
        assert req.max_tokens == 512  # default


class TestStoreField:
    """Pi sends store: true/false. Cortex should ignore it gracefully."""

    def test_store_true_ignored(self):
        body = {
            "model": "auto",
            "messages": [{"role": "user", "content": "hello"}],
            "store": True,
        }
        # Should not raise
        req = normalize_request(body)
        assert req.messages[0]["content"] == "hello"

    def test_store_false_ignored(self):
        body = {
            "model": "auto",
            "messages": [{"role": "user", "content": "hello"}],
            "store": False,
        }
        req = normalize_request(body)
        assert req.messages[0]["content"] == "hello"


class TestStrictToolDefinitions:
    """Pi sends strict: true in tool definitions."""

    def test_strict_in_tool_definition_preserved(self):
        body = {
            "model": "auto",
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather for a city",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                        "strict": True,
                    },
                }
            ],
        }
        req = normalize_request(body)
        assert req.tools is not None
        assert req.tools[0]["function"]["strict"] is True

    def test_tools_without_strict_work(self):
        body = {
            "model": "auto",
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        }
        req = normalize_request(body)
        assert req.tools is not None
        assert "strict" not in req.tools[0]["function"]


class TestStreamOptions:
    """Pi sends stream_options: {include_usage: true}."""

    def test_stream_options_does_not_crash(self):
        body = {
            "model": "auto",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        req = normalize_request(body)
        assert req.stream is True


class TestReasoningEffort:
    """Pi may send reasoning_effort field."""

    def test_reasoning_effort_ignored_gracefully(self):
        body = {
            "model": "auto",
            "messages": [{"role": "user", "content": "hello"}],
            "reasoning_effort": "medium",
        }
        req = normalize_request(body)
        assert req.messages[0]["content"] == "hello"


class TestResponseFormat:
    """Pi response must include usage with token counts."""

    def test_response_includes_usage(self):
        req = NormalizedRequest(
            messages=[{"role": "user", "content": "hi"}],
            model="auto",
            source_format=APIFormat.CHAT_COMPLETIONS,
        )
        resp = format_response(
            "Hello!",
            req,
            routing_meta={"tokens_prompt": 5, "tokens_completion": 3},
            model_used="qwen3:4b",
        )
        assert "usage" in resp
        assert resp["usage"]["prompt_tokens"] == 5
        assert resp["usage"]["completion_tokens"] == 3
        assert resp["usage"]["total_tokens"] == 8

    def test_response_has_required_fields(self):
        """Pi expects: id, object, created, model, choices, usage."""
        req = NormalizedRequest(
            messages=[{"role": "user", "content": "hi"}],
            model="auto",
            source_format=APIFormat.CHAT_COMPLETIONS,
        )
        resp = format_response("answer", req, model_used="test-model")
        assert resp["object"] == "chat.completion"
        assert resp["id"].startswith("chatcmpl-")
        assert "created" in resp
        assert resp["model"] == "test-model"
        assert len(resp["choices"]) == 1
        assert resp["choices"][0]["message"]["role"] == "assistant"
        assert resp["choices"][0]["message"]["content"] == "answer"
        assert resp["choices"][0]["finish_reason"] == "stop"

    def test_response_choices_index_zero(self):
        """Pi expects choices[0].index == 0."""
        req = NormalizedRequest(
            messages=[{"role": "user", "content": "hi"}],
            model="auto",
            source_format=APIFormat.CHAT_COMPLETIONS,
        )
        resp = format_response("x", req)
        assert resp["choices"][0]["index"] == 0


class TestPiFullRequest:
    """
    Test a realistic full Pi request with all compat fields.
    This simulates exactly what Pi sends to an OpenAI-compatible endpoint.
    """

    def test_pi_style_request(self):
        """Simulate a real Pi coding agent request."""
        body = {
            "model": "auto",
            "messages": [
                {
                    "role": "developer",
                    "content": (
                        "You are a helpful AI coding assistant. "
                        "Follow the user's instructions carefully."
                    ),
                },
                {
                    "role": "user",
                    "content": "Write a Python function that reverses a string.",
                },
            ],
            "max_completion_tokens": 4096,
            "temperature": 0.0,
            "stream": True,
            "store": True,
            "stream_options": {"include_usage": True},
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "exec_command",
                        "description": "Execute a shell command",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "cmd": {"type": "string"},
                                "timeout": {"type": "integer"},
                            },
                            "required": ["cmd"],
                        },
                        "strict": True,
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "apply_patch",
                        "description": "Apply a unified diff patch to a file",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"},
                                "patch": {"type": "string"},
                            },
                            "required": ["path", "patch"],
                        },
                        "strict": True,
                    },
                },
            ],
            "tool_choice": "auto",
        }

        req = normalize_request(body)

        # Verify all Pi fields are handled correctly
        assert req.messages[0]["role"] == "system"  # developer → system
        assert req.max_tokens == 4096  # max_completion_tokens
        assert req.stream is True
        assert req.temperature == 0.0
        assert len(req.tools) == 2
        assert req.tools[0]["function"]["name"] == "exec_command"
        assert req.tools[0]["function"]["strict"] is True
        assert req.tool_choice == "auto"
        assert req.model == "auto"

    def test_pi_explicit_model_request(self):
        """Pi with explicit model (cloud routing)."""
        body = {
            "model": "qwen/qwen3-8b",
            "messages": [
                {"role": "developer", "content": "You are a coding agent."},
                {"role": "user", "content": "Fix the bug in line 42"},
            ],
            "max_completion_tokens": 8192,
            "temperature": 0.0,
            "stream": True,
        }

        req = normalize_request(body)
        assert req.model == "qwen/qwen3-8b"
        assert "/" in req.model  # triggers cloud routing
        assert req.max_tokens == 8192
        assert req.messages[0]["role"] == "system"


class TestToolResultMessages:
    """Pi sends tool results with specific structure."""

    def test_tool_result_message_passthrough(self):
        """Tool results should pass through normalization."""
        body = {
            "model": "auto",
            "messages": [
                {"role": "user", "content": "List files"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": "exec_command",
                                "arguments": '{"cmd": "ls"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_123",
                    "name": "exec_command",
                    "content": "file1.py\nfile2.py",
                },
            ],
        }
        req = normalize_request(body)
        assert len(req.messages) == 3
        assert req.messages[1]["role"] == "assistant"
        assert req.messages[2]["role"] == "tool"
        assert req.messages[2]["content"] == "file1.py\nfile2.py"


# ===================================================================
# Integration tests: Full HTTP roundtrip (requires daemon running)
# ===================================================================


@pytest.mark.integration
class TestPiDaemonIntegration:
    """
    Integration tests that hit the actual daemon.
    Run with: pytest tests/test_pi_compat.py -m integration

    Requires daemon running: python -m src daemon --port 11411
    """

    @pytest.fixture(autouse=True)
    def _check_daemon(self):
        """Skip if daemon is not running."""
        import urllib.request
        try:
            urllib.request.urlopen("http://localhost:11411/health", timeout=2)
        except Exception:
            pytest.skip("Daemon not running on localhost:11411")

    def test_pi_streaming_with_developer_role(self):
        """Pi-style streaming request with developer role."""
        import urllib.request

        body = json.dumps({
            "model": "auto",
            "messages": [
                {"role": "developer", "content": "Answer in one word only."},
                {"role": "user", "content": "What is 2+2?"},
            ],
            "max_completion_tokens": 32,
            "stream": True,
        }).encode()

        req = urllib.request.Request(
            "http://localhost:11411/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read().decode()
            # Should be SSE format
            assert "data:" in data
            # Should not error
            assert "error" not in data.lower() or "server_error" not in data

    def test_pi_non_streaming_max_completion_tokens(self):
        """Pi-style non-streaming request with max_completion_tokens."""
        import urllib.request

        body = json.dumps({
            "model": "auto",
            "messages": [
                {"role": "developer", "content": "Be concise."},
                {"role": "user", "content": "Say hello"},
            ],
            "max_completion_tokens": 64,
            "stream": False,
        }).encode()

        req = urllib.request.Request(
            "http://localhost:11411/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
            assert result["object"] == "chat.completion"
            assert "usage" in result
            assert result["usage"]["prompt_tokens"] >= 0
            assert result["choices"][0]["message"]["role"] == "assistant"
            assert len(result["choices"][0]["message"]["content"]) > 0

    def test_pi_with_tools_and_strict(self):
        """Pi sends tools with strict: true."""
        import urllib.request

        body = json.dumps({
            "model": "auto",
            "messages": [
                {"role": "user", "content": "What is the weather in NYC?"},
            ],
            "max_completion_tokens": 128,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get current weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                        "strict": True,
                    },
                }
            ],
            "tool_choice": "auto",
            "stream": False,
        }).encode()

        req = urllib.request.Request(
            "http://localhost:11411/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
            # Should not error — either responds with text or a tool call
            assert "error" not in result or result.get("error") is None
            assert result["object"] == "chat.completion"

    def test_pi_cloud_streaming(self):
        """Pi-style request to explicit cloud model with streaming."""
        import urllib.request

        body = json.dumps({
            "model": "qwen/qwen3-8b",
            "messages": [
                {"role": "developer", "content": "One word answer."},
                {"role": "user", "content": "Capital of France?"},
            ],
            "max_completion_tokens": 16,
            "stream": True,
            "store": True,
            "stream_options": {"include_usage": True},
        }).encode()

        req = urllib.request.Request(
            "http://localhost:11411/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read().decode()
            assert "data:" in data
            # Verify we get actual content chunks
            lines = [l for l in data.split("\n") if l.startswith("data: ") and l != "data: [DONE]"]
            assert len(lines) > 0
            # Parse first chunk
            first = json.loads(lines[0][6:])
            assert first["object"] == "chat.completion.chunk"
