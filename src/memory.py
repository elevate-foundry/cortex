"""
Cortex Memory — persistent storage layer for the AI kernel.

This is the "filesystem + memory manager" of the agent microkernel.
Everything that needs to survive a restart lives here:

  - Conversations:  thread state, message history, metadata
  - Audit log:      every syscall (request/response/routing/latency/cost)
  - KV cache index: which prefixes are cached, for which models
  - Config/policy:  runtime settings, per-app overrides
  - Embeddings:     vector memory for semantic recall (future)

Storage: SQLite (single file, zero-config, ACID, WAL mode for concurrent reads).
Default location: ~/.cortex/cortex.db

Design principles:
  - Every table has created_at/updated_at timestamps (INTEGER, unix epoch ms)
  - Conversations are identified by thread_id (UUID or user-provided)
  - Audit log is append-only, never deleted
  - All text fields are UTF-8, JSON stored as TEXT
  - WAL mode for concurrent reads from multiple daemon instances
"""

import hashlib
import json
import logging
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional, Any, Iterator

logger = logging.getLogger("cortex.memory")


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def default_data_dir() -> Path:
    """~/.cortex/ — the kernel's persistent state directory."""
    return Path(os.environ.get("CORTEX_DATA_DIR", Path.home() / ".cortex"))


def default_db_path() -> Path:
    return default_data_dir() / "cortex.db"


# ---------------------------------------------------------------------------
# Clock
# ---------------------------------------------------------------------------

def _now_ms() -> int:
    """Current time as integer milliseconds since epoch."""
    return int(time.time() * 1000)


def _uuid() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Message:
    """A single message in a conversation."""
    id: str
    thread_id: str
    role: str          # system, user, assistant, tool
    content: str
    model: str = ""
    tier: str = ""
    tokens_prompt: int = 0
    tokens_completion: int = 0
    latency_ms: float = 0.0
    metadata: dict = field(default_factory=dict)
    created_at: int = 0

    def to_chat_dict(self) -> dict:
        """Convert to OpenAI chat message format."""
        d = {"role": self.role, "content": self.content}
        if self.role == "tool" and self.metadata.get("tool_call_id"):
            d["tool_call_id"] = self.metadata["tool_call_id"]
        return d


@dataclass
class Thread:
    """A conversation thread."""
    id: str
    title: str = ""
    app_id: str = ""         # which app created this thread
    system_prompt: str = ""
    model_hint: str = ""     # client-requested model, or "auto"
    message_count: int = 0
    total_tokens: int = 0
    metadata: dict = field(default_factory=dict)
    created_at: int = 0
    updated_at: int = 0


@dataclass
class AuditEntry:
    """An audit log entry — one per API syscall."""
    id: str
    thread_id: str
    request_model: str       # what the client asked for
    routed_tier: str         # what the router picked
    actual_model: str        # what actually ran
    category: str            # route category (code, chat, classify, etc.)
    confidence: float
    tokens_prompt: int
    tokens_completion: int
    latency_ms: float
    ttft_ms: float           # time to first token
    status_code: int
    client_ip: str = ""
    app_id: str = ""
    escalation_path: str = ""  # JSON list of escalation steps
    error: str = ""
    created_at: int = 0


@dataclass
class KVCacheEntry:
    """Tracks which prompt prefixes are cached in which backends."""
    id: str
    model: str
    prefix_hash: str         # SHA-256 of the prompt prefix
    prefix_tokens: int
    backend: str             # ollama, llama_cpp, vllm
    created_at: int = 0
    last_hit: int = 0
    hit_count: int = 0


@dataclass
class PolicyRule:
    """A policy/config rule."""
    key: str                 # e.g. "cloud_allowed", "max_tier", "rate_limit"
    value: str               # JSON-encoded value
    scope: str = "global"    # "global", "app:{id}", "thread:{id}"
    created_at: int = 0
    updated_at: int = 0


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

SCHEMA_VERSION = 1

SCHEMA_SQL = """
-- Cortex kernel persistent state
-- Schema version: 1

CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Conversations
CREATE TABLE IF NOT EXISTS threads (
    id            TEXT PRIMARY KEY,
    title         TEXT NOT NULL DEFAULT '',
    app_id        TEXT NOT NULL DEFAULT '',
    system_prompt TEXT NOT NULL DEFAULT '',
    model_hint    TEXT NOT NULL DEFAULT 'auto',
    message_count INTEGER NOT NULL DEFAULT 0,
    total_tokens  INTEGER NOT NULL DEFAULT 0,
    metadata      TEXT NOT NULL DEFAULT '{}',
    created_at    INTEGER NOT NULL,
    updated_at    INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_threads_app    ON threads(app_id);
CREATE INDEX IF NOT EXISTS idx_threads_updated ON threads(updated_at);

CREATE TABLE IF NOT EXISTS messages (
    id                TEXT PRIMARY KEY,
    thread_id         TEXT NOT NULL REFERENCES threads(id),
    role              TEXT NOT NULL,
    content           TEXT NOT NULL,
    model             TEXT NOT NULL DEFAULT '',
    tier              TEXT NOT NULL DEFAULT '',
    tokens_prompt     INTEGER NOT NULL DEFAULT 0,
    tokens_completion INTEGER NOT NULL DEFAULT 0,
    latency_ms        REAL NOT NULL DEFAULT 0,
    metadata          TEXT NOT NULL DEFAULT '{}',
    created_at        INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_thread ON messages(thread_id, created_at);

-- Audit log (append-only)
CREATE TABLE IF NOT EXISTS audit_log (
    id                TEXT PRIMARY KEY,
    thread_id         TEXT NOT NULL DEFAULT '',
    request_model     TEXT NOT NULL DEFAULT '',
    routed_tier       TEXT NOT NULL DEFAULT '',
    actual_model      TEXT NOT NULL DEFAULT '',
    category          TEXT NOT NULL DEFAULT '',
    confidence        REAL NOT NULL DEFAULT 0,
    tokens_prompt     INTEGER NOT NULL DEFAULT 0,
    tokens_completion INTEGER NOT NULL DEFAULT 0,
    latency_ms        REAL NOT NULL DEFAULT 0,
    ttft_ms           REAL NOT NULL DEFAULT 0,
    status_code       INTEGER NOT NULL DEFAULT 200,
    client_ip         TEXT NOT NULL DEFAULT '',
    app_id            TEXT NOT NULL DEFAULT '',
    escalation_path   TEXT NOT NULL DEFAULT '[]',
    error             TEXT NOT NULL DEFAULT '',
    created_at        INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audit_time     ON audit_log(created_at);
CREATE INDEX IF NOT EXISTS idx_audit_thread   ON audit_log(thread_id);
CREATE INDEX IF NOT EXISTS idx_audit_tier     ON audit_log(routed_tier);

-- KV cache prefix index
CREATE TABLE IF NOT EXISTS kv_cache_index (
    id            TEXT PRIMARY KEY,
    model         TEXT NOT NULL,
    prefix_hash   TEXT NOT NULL,
    prefix_tokens INTEGER NOT NULL DEFAULT 0,
    backend       TEXT NOT NULL DEFAULT '',
    created_at    INTEGER NOT NULL,
    last_hit      INTEGER NOT NULL DEFAULT 0,
    hit_count     INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_kv_model_hash ON kv_cache_index(model, prefix_hash);

-- Policy / config store
CREATE TABLE IF NOT EXISTS policies (
    key        TEXT NOT NULL,
    scope      TEXT NOT NULL DEFAULT 'global',
    value      TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    PRIMARY KEY (key, scope)
);

-- Usage stats (daily aggregates)
CREATE TABLE IF NOT EXISTS usage_daily (
    date_str          TEXT NOT NULL,     -- YYYY-MM-DD
    tier              TEXT NOT NULL,
    model             TEXT NOT NULL DEFAULT '',
    request_count     INTEGER NOT NULL DEFAULT 0,
    tokens_prompt     INTEGER NOT NULL DEFAULT 0,
    tokens_completion INTEGER NOT NULL DEFAULT 0,
    total_latency_ms  REAL NOT NULL DEFAULT 0,
    error_count       INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (date_str, tier, model)
);
"""


# ---------------------------------------------------------------------------
# Memory store
# ---------------------------------------------------------------------------

class Memory:
    """
    The Cortex persistent memory layer.
    
    Usage:
        mem = Memory()          # opens/creates ~/.cortex/cortex.db
        mem = Memory(":memory:")  # in-memory for tests
    """

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = str(default_db_path())

        self._db_path = db_path

        # Ensure directory exists
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA busy_timeout=5000")

        self._init_schema()
        logger.info("Memory initialized: %s", db_path)

    def _init_schema(self):
        """Create tables if they don't exist."""
        self._conn.executescript(SCHEMA_SQL)
        # Set schema version
        self._conn.execute(
            "INSERT OR REPLACE INTO schema_meta (key, value) VALUES (?, ?)",
            ("schema_version", str(SCHEMA_VERSION)),
        )
        self._conn.commit()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Cursor]:
        """Transaction context manager."""
        cursor = self._conn.cursor()
        try:
            yield cursor
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    # ------------------------------------------------------------------
    # Threads
    # ------------------------------------------------------------------

    def create_thread(
        self,
        thread_id: Optional[str] = None,
        title: str = "",
        app_id: str = "",
        system_prompt: str = "",
        model_hint: str = "auto",
        metadata: Optional[dict] = None,
    ) -> Thread:
        """Create a new conversation thread."""
        now = _now_ms()
        thread = Thread(
            id=thread_id or _uuid(),
            title=title,
            app_id=app_id,
            system_prompt=system_prompt,
            model_hint=model_hint,
            metadata=metadata or {},
            created_at=now,
            updated_at=now,
        )
        with self._tx() as c:
            c.execute(
                """INSERT INTO threads 
                   (id, title, app_id, system_prompt, model_hint, metadata, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (thread.id, thread.title, thread.app_id, thread.system_prompt,
                 thread.model_hint, json.dumps(thread.metadata),
                 thread.created_at, thread.updated_at),
            )
        return thread

    def get_thread(self, thread_id: str) -> Optional[Thread]:
        """Get a thread by ID."""
        row = self._conn.execute(
            "SELECT * FROM threads WHERE id = ?", (thread_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_thread(row)

    def get_or_create_thread(
        self, thread_id: str, **kwargs
    ) -> Thread:
        """Get a thread or create it if it doesn't exist."""
        thread = self.get_thread(thread_id)
        if thread is not None:
            return thread
        return self.create_thread(thread_id=thread_id, **kwargs)

    def list_threads(
        self,
        app_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Thread]:
        """List threads, most recently updated first."""
        if app_id:
            rows = self._conn.execute(
                "SELECT * FROM threads WHERE app_id = ? ORDER BY updated_at DESC LIMIT ? OFFSET ?",
                (app_id, limit, offset),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM threads ORDER BY updated_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [self._row_to_thread(r) for r in rows]

    def delete_thread(self, thread_id: str) -> bool:
        """Delete a thread and all its messages."""
        with self._tx() as c:
            c.execute("DELETE FROM messages WHERE thread_id = ?", (thread_id,))
            c.execute("DELETE FROM threads WHERE id = ?", (thread_id,))
            return c.rowcount > 0

    def _row_to_thread(self, row: sqlite3.Row) -> Thread:
        return Thread(
            id=row["id"],
            title=row["title"],
            app_id=row["app_id"],
            system_prompt=row["system_prompt"],
            model_hint=row["model_hint"],
            message_count=row["message_count"],
            total_tokens=row["total_tokens"],
            metadata=json.loads(row["metadata"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    # ------------------------------------------------------------------
    # Messages
    # ------------------------------------------------------------------

    def add_message(
        self,
        thread_id: str,
        role: str,
        content: str,
        model: str = "",
        tier: str = "",
        tokens_prompt: int = 0,
        tokens_completion: int = 0,
        latency_ms: float = 0.0,
        metadata: Optional[dict] = None,
    ) -> Message:
        """Add a message to a thread."""
        now = _now_ms()
        msg = Message(
            id=_uuid(),
            thread_id=thread_id,
            role=role,
            content=content,
            model=model,
            tier=tier,
            tokens_prompt=tokens_prompt,
            tokens_completion=tokens_completion,
            latency_ms=latency_ms,
            metadata=metadata or {},
            created_at=now,
        )
        total_toks = tokens_prompt + tokens_completion
        with self._tx() as c:
            c.execute(
                """INSERT INTO messages 
                   (id, thread_id, role, content, model, tier, 
                    tokens_prompt, tokens_completion, latency_ms, metadata, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (msg.id, msg.thread_id, msg.role, msg.content, msg.model,
                 msg.tier, msg.tokens_prompt, msg.tokens_completion,
                 msg.latency_ms, json.dumps(msg.metadata), msg.created_at),
            )
            # Update thread counters
            c.execute(
                """UPDATE threads 
                   SET message_count = message_count + 1,
                       total_tokens = total_tokens + ?,
                       updated_at = ?
                   WHERE id = ?""",
                (total_toks, now, thread_id),
            )
        return msg

    def get_messages(
        self,
        thread_id: str,
        limit: Optional[int] = None,
        since_ms: Optional[int] = None,
    ) -> list[Message]:
        """Get messages for a thread, in chronological order."""
        query = "SELECT * FROM messages WHERE thread_id = ?"
        params: list[Any] = [thread_id]

        if since_ms is not None:
            query += " AND created_at > ?"
            params.append(since_ms)

        query += " ORDER BY created_at ASC"

        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)

        rows = self._conn.execute(query, params).fetchall()
        return [self._row_to_message(r) for r in rows]

    def get_context_window(
        self,
        thread_id: str,
        max_messages: int = 50,
        max_tokens: int = 8192,
        include_system: bool = True,
    ) -> list[dict]:
        """
        Build a context window from thread history.
        
        Returns messages in OpenAI chat format, trimmed to fit
        within token/message budget. Always includes system prompt
        and the most recent messages.
        """
        thread = self.get_thread(thread_id)
        if thread is None:
            return []

        messages = self.get_messages(thread_id)

        # Start with system prompt if present
        result: list[dict] = []
        if include_system and thread.system_prompt:
            result.append({"role": "system", "content": thread.system_prompt})

        # Take the most recent messages that fit
        # Simple heuristic: ~4 chars per token
        token_budget = max_tokens
        if result:
            token_budget -= len(result[0]["content"]) // 4

        recent: list[dict] = []
        token_count = 0
        for msg in reversed(messages[-max_messages:]):
            msg_tokens = (len(msg.content) // 4) + 4  # rough estimate
            if token_count + msg_tokens > token_budget:
                break
            recent.insert(0, msg.to_chat_dict())
            token_count += msg_tokens

        result.extend(recent)
        return result

    def _row_to_message(self, row: sqlite3.Row) -> Message:
        return Message(
            id=row["id"],
            thread_id=row["thread_id"],
            role=row["role"],
            content=row["content"],
            model=row["model"],
            tier=row["tier"],
            tokens_prompt=row["tokens_prompt"],
            tokens_completion=row["tokens_completion"],
            latency_ms=row["latency_ms"],
            metadata=json.loads(row["metadata"]),
            created_at=row["created_at"],
        )

    # ------------------------------------------------------------------
    # Audit log
    # ------------------------------------------------------------------

    def log_request(
        self,
        thread_id: str = "",
        request_model: str = "",
        routed_tier: str = "",
        actual_model: str = "",
        category: str = "",
        confidence: float = 0.0,
        tokens_prompt: int = 0,
        tokens_completion: int = 0,
        latency_ms: float = 0.0,
        ttft_ms: float = 0.0,
        status_code: int = 200,
        client_ip: str = "",
        app_id: str = "",
        escalation_path: Optional[list[str]] = None,
        error: str = "",
    ) -> AuditEntry:
        """Log an API request to the audit log."""
        now = _now_ms()
        entry = AuditEntry(
            id=_uuid(),
            thread_id=thread_id,
            request_model=request_model,
            routed_tier=routed_tier,
            actual_model=actual_model,
            category=category,
            confidence=confidence,
            tokens_prompt=tokens_prompt,
            tokens_completion=tokens_completion,
            latency_ms=latency_ms,
            ttft_ms=ttft_ms,
            status_code=status_code,
            client_ip=client_ip,
            app_id=app_id,
            escalation_path=json.dumps(escalation_path or []),
            error=error,
            created_at=now,
        )
        with self._tx() as c:
            c.execute(
                """INSERT INTO audit_log
                   (id, thread_id, request_model, routed_tier, actual_model,
                    category, confidence, tokens_prompt, tokens_completion,
                    latency_ms, ttft_ms, status_code, client_ip, app_id,
                    escalation_path, error, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (entry.id, entry.thread_id, entry.request_model,
                 entry.routed_tier, entry.actual_model, entry.category,
                 entry.confidence, entry.tokens_prompt, entry.tokens_completion,
                 entry.latency_ms, entry.ttft_ms, entry.status_code,
                 entry.client_ip, entry.app_id, entry.escalation_path,
                 entry.error, entry.created_at),
            )
            # Update daily usage
            date_str = time.strftime("%Y-%m-%d")
            err_inc = 1 if error else 0
            c.execute(
                """INSERT INTO usage_daily 
                   (date_str, tier, model, request_count, tokens_prompt, 
                    tokens_completion, total_latency_ms, error_count)
                   VALUES (?, ?, ?, 1, ?, ?, ?, ?)
                   ON CONFLICT(date_str, tier, model) DO UPDATE SET
                    request_count = request_count + 1,
                    tokens_prompt = tokens_prompt + excluded.tokens_prompt,
                    tokens_completion = tokens_completion + excluded.tokens_completion,
                    total_latency_ms = total_latency_ms + excluded.total_latency_ms,
                    error_count = error_count + excluded.error_count""",
                (date_str, routed_tier, actual_model,
                 tokens_prompt, tokens_completion, latency_ms, err_inc),
            )
        return entry

    def get_audit_log(
        self,
        thread_id: Optional[str] = None,
        tier: Optional[str] = None,
        limit: int = 100,
        since_ms: Optional[int] = None,
    ) -> list[AuditEntry]:
        """Query the audit log."""
        query = "SELECT * FROM audit_log WHERE 1=1"
        params: list[Any] = []

        if thread_id:
            query += " AND thread_id = ?"
            params.append(thread_id)
        if tier:
            query += " AND routed_tier = ?"
            params.append(tier)
        if since_ms:
            query += " AND created_at > ?"
            params.append(since_ms)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        rows = self._conn.execute(query, params).fetchall()
        return [self._row_to_audit(r) for r in rows]

    def _row_to_audit(self, row: sqlite3.Row) -> AuditEntry:
        return AuditEntry(
            id=row["id"],
            thread_id=row["thread_id"],
            request_model=row["request_model"],
            routed_tier=row["routed_tier"],
            actual_model=row["actual_model"],
            category=row["category"],
            confidence=row["confidence"],
            tokens_prompt=row["tokens_prompt"],
            tokens_completion=row["tokens_completion"],
            latency_ms=row["latency_ms"],
            ttft_ms=row["ttft_ms"],
            status_code=row["status_code"],
            client_ip=row["client_ip"],
            app_id=row["app_id"],
            escalation_path=row["escalation_path"],
            error=row["error"],
            created_at=row["created_at"],
        )

    # ------------------------------------------------------------------
    # KV cache index
    # ------------------------------------------------------------------

    def register_kv_prefix(
        self,
        model: str,
        prefix_text: str,
        prefix_tokens: int,
        backend: str = "",
    ) -> KVCacheEntry:
        """Register a KV cache prefix for future reuse."""
        prefix_hash = hashlib.sha256(prefix_text.encode()).hexdigest()[:32]
        now = _now_ms()

        # Upsert — if this exact prefix exists, just bump hit count
        with self._tx() as c:
            existing = c.execute(
                "SELECT id FROM kv_cache_index WHERE model = ? AND prefix_hash = ?",
                (model, prefix_hash),
            ).fetchone()

            if existing:
                c.execute(
                    """UPDATE kv_cache_index 
                       SET last_hit = ?, hit_count = hit_count + 1
                       WHERE id = ?""",
                    (now, existing["id"]),
                )
                return KVCacheEntry(
                    id=existing["id"], model=model, prefix_hash=prefix_hash,
                    prefix_tokens=prefix_tokens, backend=backend,
                    created_at=now, last_hit=now,
                )

            entry = KVCacheEntry(
                id=_uuid(), model=model, prefix_hash=prefix_hash,
                prefix_tokens=prefix_tokens, backend=backend,
                created_at=now, last_hit=now, hit_count=1,
            )
            c.execute(
                """INSERT INTO kv_cache_index
                   (id, model, prefix_hash, prefix_tokens, backend, created_at, last_hit, hit_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (entry.id, entry.model, entry.prefix_hash, entry.prefix_tokens,
                 entry.backend, entry.created_at, entry.last_hit, entry.hit_count),
            )
        return entry

    def find_kv_prefix(self, model: str, prefix_text: str) -> Optional[KVCacheEntry]:
        """Check if a KV cache prefix exists for reuse."""
        prefix_hash = hashlib.sha256(prefix_text.encode()).hexdigest()[:32]
        row = self._conn.execute(
            "SELECT * FROM kv_cache_index WHERE model = ? AND prefix_hash = ?",
            (model, prefix_hash),
        ).fetchone()
        if row is None:
            return None
        return KVCacheEntry(
            id=row["id"], model=row["model"], prefix_hash=row["prefix_hash"],
            prefix_tokens=row["prefix_tokens"], backend=row["backend"],
            created_at=row["created_at"], last_hit=row["last_hit"],
            hit_count=row["hit_count"],
        )

    # ------------------------------------------------------------------
    # Policy / config store
    # ------------------------------------------------------------------

    def set_policy(self, key: str, value: Any, scope: str = "global") -> None:
        """Set a policy/config value."""
        now = _now_ms()
        with self._tx() as c:
            c.execute(
                """INSERT INTO policies (key, scope, value, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(key, scope) DO UPDATE SET
                    value = excluded.value,
                    updated_at = excluded.updated_at""",
                (key, scope, json.dumps(value), now, now),
            )

    def get_policy(self, key: str, scope: str = "global", default: Any = None) -> Any:
        """Get a policy/config value."""
        row = self._conn.execute(
            "SELECT value FROM policies WHERE key = ? AND scope = ?",
            (key, scope),
        ).fetchone()
        if row is None:
            return default
        return json.loads(row["value"])

    def get_effective_policy(self, key: str, app_id: str = "", thread_id: str = "") -> Any:
        """
        Get the effective policy, checking scopes in order:
          thread:{id} > app:{id} > global
        """
        if thread_id:
            val = self.get_policy(key, scope=f"thread:{thread_id}")
            if val is not None:
                return val
        if app_id:
            val = self.get_policy(key, scope=f"app:{app_id}")
            if val is not None:
                return val
        return self.get_policy(key, scope="global")

    # ------------------------------------------------------------------
    # Usage stats
    # ------------------------------------------------------------------

    def get_usage_summary(
        self,
        days: int = 7,
    ) -> dict:
        """Get usage summary for the last N days."""
        rows = self._conn.execute(
            """SELECT 
                 date_str,
                 SUM(request_count) as requests,
                 SUM(tokens_prompt) as prompt_tokens,
                 SUM(tokens_completion) as completion_tokens,
                 SUM(total_latency_ms) as total_latency,
                 SUM(error_count) as errors
               FROM usage_daily
               WHERE date_str >= date('now', ?)
               GROUP BY date_str
               ORDER BY date_str DESC""",
            (f"-{days} days",),
        ).fetchall()

        daily = [
            {
                "date": r["date_str"],
                "requests": r["requests"],
                "prompt_tokens": r["prompt_tokens"],
                "completion_tokens": r["completion_tokens"],
                "avg_latency_ms": round(r["total_latency"] / max(r["requests"], 1), 1),
                "errors": r["errors"],
            }
            for r in rows
        ]

        # Tier breakdown
        tier_rows = self._conn.execute(
            """SELECT
                 tier,
                 SUM(request_count) as requests,
                 SUM(tokens_prompt + tokens_completion) as total_tokens
               FROM usage_daily
               WHERE date_str >= date('now', ?)
               GROUP BY tier
               ORDER BY requests DESC""",
            (f"-{days} days",),
        ).fetchall()

        by_tier = {r["tier"]: {"requests": r["requests"], "tokens": r["total_tokens"]}
                   for r in tier_rows}

        # Totals
        total_requests = sum(d["requests"] for d in daily)
        total_tokens = sum(d["prompt_tokens"] + d["completion_tokens"] for d in daily)

        return {
            "period_days": days,
            "total_requests": total_requests,
            "total_tokens": total_tokens,
            "daily": daily,
            "by_tier": by_tier,
        }

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def vacuum(self) -> None:
        """Compact the database."""
        self._conn.execute("VACUUM")

    def prune_audit(self, keep_days: int = 30) -> int:
        """Delete audit entries older than N days."""
        cutoff_ms = _now_ms() - (keep_days * 86400 * 1000)
        with self._tx() as c:
            c.execute("DELETE FROM audit_log WHERE created_at < ?", (cutoff_ms,))
            return c.rowcount

    def prune_kv_cache(self, keep_days: int = 7) -> int:
        """Delete stale KV cache entries."""
        cutoff_ms = _now_ms() - (keep_days * 86400 * 1000)
        with self._tx() as c:
            c.execute("DELETE FROM kv_cache_index WHERE last_hit < ?", (cutoff_ms,))
            return c.rowcount

    def db_stats(self) -> dict:
        """Database size and table counts."""
        counts = {}
        for table in ["threads", "messages", "audit_log", "kv_cache_index", "policies", "usage_daily"]:
            row = self._conn.execute(f"SELECT COUNT(*) as cnt FROM {table}").fetchone()
            counts[table] = row["cnt"]

        db_size = 0
        if self._db_path != ":memory:":
            try:
                db_size = os.path.getsize(self._db_path)
            except OSError:
                pass

        return {
            "db_path": self._db_path,
            "db_size_bytes": db_size,
            "db_size_mb": round(db_size / (1024 * 1024), 2),
            "tables": counts,
        }

    def close(self):
        """Close the database connection."""
        self._conn.close()

    def __repr__(self) -> str:
        stats = self.db_stats()
        return f"Memory({self._db_path}, {stats['db_size_mb']}MB, {stats['tables']})"
