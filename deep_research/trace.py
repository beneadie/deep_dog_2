"""Structured, content-rich per-run trace for the Deep Dog engine.

The product-level ``deep_research.events`` channel is deliberately content-free
(counts, names, URLs) so host UIs and databases never depend on model reasoning
or provider payloads. This module is the complementary, opt-in channel: full
prompts, model thinking, tool calls and source rationale, each record carrying a
stable UUID so production logs can be correlated and streamed.

Design:
  - ``TraceRecord`` — one structured log line (full UUID + monotonic ``seq``).
  - ``deliver_trace`` — deliver a record to a sync or async sink.
  - ``TraceCollector`` — in-memory sink for tests and simple hosts.
  - ``serialize_message`` / ``serialize_messages`` — LangChain message -> dict.
  - ``redact`` / ``truncate`` — applied by the Observer before a record is
    buffered, so secrets and oversized payloads never leave the process.

Trace is in-memory only: records are retained on the per-run Observer for the
duration of the run and optionally streamed through ``RuntimeOptions.trace_sink``.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Protocol


def new_trace_id() -> str:
    """A full UUID4 hex identifier for a single trace record."""
    return uuid.uuid4().hex


@dataclass
class TraceRecord:
    """A single structured, content-bearing trace record."""

    kind: str
    run_id: str = ""
    event_id: str = field(default_factory=new_trace_id)
    seq: int = 0
    timestamp: float = field(default_factory=time.time)
    phase: Optional[str] = None
    agent: Optional[str] = None
    platform: Optional[str] = None
    iteration: Optional[int] = None
    parent_id: Optional[str] = None
    content: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "run_id": self.run_id,
            "event_id": self.event_id,
            "seq": self.seq,
            "timestamp": self.timestamp,
            "phase": self.phase,
            "agent": self.agent,
            "platform": self.platform,
            "iteration": self.iteration,
            "parent_id": self.parent_id,
            "content": self.content,
        }


class TraceSink(Protocol):
    def emit(self, record: TraceRecord) -> Any:  # pragma: no cover - protocol
        ...


async def deliver_trace(sink: Any, record: TraceRecord) -> None:
    """Deliver one trace record to a sink that may be sync or async."""
    if sink is None:
        return
    if isinstance(sink, TraceCollector):
        await sink.emit(record)
        return
    emit = getattr(sink, "emit", None)
    if emit is not None:
        result = emit(record)
        if hasattr(result, "__await__"):
            await result
        return
    if callable(sink):
        result = sink(record)
        if hasattr(result, "__await__"):
            await result
        return


class TraceCollector:
    """In-memory trace sink. Useful for tests and simple hosts."""

    def __init__(self) -> None:
        self.records: list[TraceRecord] = []

    async def emit(self, record: TraceRecord) -> None:
        self.records.append(record)

    def kinds(self) -> list[str]:
        return [r.kind for r in self.records]


# ── Content helpers ─────────────────────────────────────────────────────

def _to_text(content: Any) -> str:
    """Best-effort stringification of a message content (str | blocks | other)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                parts.append(text if isinstance(text, str) else str(block))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    if content is None:
        return ""
    return str(content)


def serialize_message(message: Any) -> dict:
    """Serialize a LangChain message (or plain dict) to a JSON-safe dict."""
    if isinstance(message, dict):
        return dict(message)
    tool_calls = getattr(message, "tool_calls", None) or []
    serialized_calls = []
    for tc in tool_calls:
        if isinstance(tc, dict):
            serialized_calls.append({"name": tc.get("name", ""), "args": tc.get("args", {})})
    return {
        "type": getattr(message, "type", type(message).__name__),
        "role": getattr(message, "type", None),
        "name": getattr(message, "name", None),
        "content": _to_text(getattr(message, "content", message)),
        "tool_calls": serialized_calls,
    }


def serialize_messages(messages: Iterable[Any]) -> list[dict]:
    return [serialize_message(m) for m in (messages or [])]


def redact(value: Any, secrets: Iterable[str]) -> Any:
    """Recursively replace any occurrence of a secret value with [REDACTED].

    ``secrets`` should contain only non-trivial values; empty/short strings are
    ignored by the caller. Works on strings, lists and dicts (values and keys).
    """
    secret_list = [s for s in secrets if s]
    if not secret_list:
        return value
    if isinstance(value, str):
        for s in secret_list:
            if s in value:
                value = value.replace(s, "[REDACTED]")
        return value
    if isinstance(value, dict):
        return {
            redact(k, secret_list): redact(v, secret_list)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact(v, secret_list) for v in value]
    return value


def truncate(value: Any, limit: Optional[int]) -> Any:
    """Recursively truncate strings longer than ``limit`` chars.

    ``limit`` of ``None`` or ``<= 0`` means no truncation.
    """
    if not limit or limit <= 0:
        return value
    if isinstance(value, str):
        if len(value) > limit:
            return value[:limit] + f"... [truncated {len(value) - limit} chars]"
        return value
    if isinstance(value, dict):
        return {k: truncate(v, limit) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [truncate(v, limit) for v in value]
    return value
