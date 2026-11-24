"""Per-run usage accounting for LLM tokens and billable search tool calls.

Every model call records a small, content-free usage entry into the active
``RuntimeContext``. Search tools record one entry per provider call. Entries
are aggregated and exposed through ``ResearchResult.usage``, structured
events, the console, and an optional ``usage.json`` log file. No dollar
amounts are computed here: the tracked metrics (tokens, search calls, cache
hits) are the quantities that drive spend.

Token metadata is normalized from the LangChain shape (``usage_metadata``),
the OpenAI-compatible shape (``response_metadata["token_usage"]``), and
DeepSeek's cache-hit/cache-miss fields. Missing metadata is tolerated so
scripted fakes and unusual providers never break a run.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Optional

from deep_research import events as _events

TOKEN_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_creation_tokens",
    "cache_miss_tokens",
    "reasoning_tokens",
    "total_tokens",
)


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _to_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _detail_value(details: Any, *suffixes: str) -> int:
    """Read a detail key, tolerating service-tier prefixes (e.g. priority_)."""
    if not isinstance(details, dict):
        return 0
    for key, value in details.items():
        for suffix in suffixes:
            if key == suffix or key.endswith("_" + suffix):
                number = _to_int(value)
                if number:
                    return number
    return 0


def empty_bucket() -> dict:
    """Zeroed accumulator for one role (or the whole run)."""
    bucket = {"calls": 0, "calls_with_usage": 0}
    bucket.update({name: 0 for name in TOKEN_FIELDS})
    return bucket


def extract_usage(response: Any) -> dict:
    """Normalize token counts from a chat-model response.

    Returns a dict of TOKEN_FIELDS plus ``usage_available``, ``model`` and
    ``provider``. ``usage_available`` is False when the response carried no
    token metadata. Cache-miss and reasoning counts are breakdown fields
    inside input/output respectively; they are never added to total_tokens.
    """
    metadata = _get(response, "usage_metadata", None)
    response_metadata = _get(response, "response_metadata", None) or {}
    if not isinstance(response_metadata, dict):
        response_metadata = {}
    token_usage = (
        response_metadata.get("token_usage")
        or response_metadata.get("usage")
        or _get(response, "token_usage", None)
        or {}
    )

    input_details = _get(metadata, "input_token_details", None) or {}
    output_details = _get(metadata, "output_token_details", None) or {}
    prompt_details = _get(token_usage, "prompt_tokens_details", None) or {}
    completion_details = _get(token_usage, "completion_tokens_details", None) or {}

    input_tokens = (
        _to_int(_get(metadata, "input_tokens"))
        or _to_int(_get(token_usage, "prompt_tokens"))
        or _to_int(_get(token_usage, "input_tokens"))
    )
    output_tokens = (
        _to_int(_get(metadata, "output_tokens"))
        or _to_int(_get(token_usage, "completion_tokens"))
        or _to_int(_get(token_usage, "output_tokens"))
    )
    cache_read = (
        _detail_value(input_details, "cache_read")
        or _to_int(_get(prompt_details, "cached_tokens"))
        or _to_int(_get(token_usage, "prompt_cache_hit_tokens"))
        or _to_int(_get(token_usage, "cache_read_input_tokens"))
    )
    cache_creation = (
        _detail_value(input_details, "cache_creation")
        or _to_int(_get(prompt_details, "cache_write_tokens"))
        or _to_int(_get(token_usage, "cache_creation_input_tokens"))
        or _to_int(_get(token_usage, "cache_write_input_tokens"))
    )
    cache_miss = _to_int(_get(token_usage, "prompt_cache_miss_tokens"))
    reasoning = (
        _detail_value(output_details, "reasoning")
        or _to_int(_get(completion_details, "reasoning_tokens"))
    )
    total = _to_int(_get(metadata, "total_tokens")) or _to_int(_get(token_usage, "total_tokens"))
    if not total:
        total = input_tokens + output_tokens

    model = (
        response_metadata.get("model_name")
        or response_metadata.get("model")
        or response_metadata.get("model_id")
        or _get(_get(response, "additional_kwargs", None), "model_name", None)
    )
    provider = response_metadata.get("model_provider") or _get(response, "model_provider", None)

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_tokens": cache_read,
        "cache_creation_tokens": cache_creation,
        "cache_miss_tokens": cache_miss,
        "reasoning_tokens": reasoning,
        "total_tokens": total,
        "usage_available": bool(metadata) or bool(token_usage),
        "model": model,
        "provider": provider,
    }


def _configured_model(runtime: Any, role: str) -> Optional[str]:
    """Best-effort model name from the run config when a provider omits it."""
    cfg = getattr(runtime, "config", None)
    if cfg is None:
        return None
    if role == "final_report_writer":
        chain = getattr(cfg, "final_report_model_fallback_chain", None) \
            or getattr(cfg, "supervisor_model_fallback_chain", None)
    elif role in ("scope_brief", "draft_report"):
        chain = getattr(cfg, "draft_report_model_fallback_chain", None)
    elif role in ("subagent", "subagent_report_writer"):
        chain = getattr(cfg, "subagent_model_fallback_chain", None)
    else:
        chain = getattr(cfg, "supervisor_model_fallback_chain", None)
    return next(iter(chain), None) if chain else None


class UsageTracker:
    """Thread-safe accumulator of per-call LLM token usage for one run."""

    def __init__(self):
        self._lock = threading.Lock()
        self._records: list[dict] = []

    def record(
        self,
        response: Any,
        *,
        role: str,
        agent: Optional[str] = None,
        platform: Optional[str] = None,
        iteration: Optional[int] = None,
        model: Optional[str] = None,
        model_fallback: Optional[str] = None,
    ) -> dict:
        usage = extract_usage(response)
        resolved_model = model or usage["model"]
        model_inferred = resolved_model is None
        if model_inferred:
            resolved_model = model_fallback or _configured_model(_current_runtime(), role)
        entry = {
            "timestamp": time.time(),
            "role": role,
            "agent": agent,
            "platform": platform,
            "iteration": iteration,
            "model": resolved_model,
            "model_inferred": bool(model_inferred and resolved_model),
            "provider": usage["provider"],
            "usage_available": usage["usage_available"],
        }
        entry.update({name: usage[name] for name in TOKEN_FIELDS})
        with self._lock:
            self._records.append(entry)
        return entry

    def records(self) -> list[dict]:
        with self._lock:
            return [dict(entry) for entry in self._records]

    def snapshot(self) -> dict:
        """Return calls, run totals, and totals keyed by role."""
        with self._lock:
            records = [dict(entry) for entry in self._records]

        total = empty_bucket()
        by_role: dict[str, dict] = {}
        for entry in records:
            role = entry.get("role") or "unknown"
            bucket = by_role.setdefault(role, empty_bucket())
            for target in (total, bucket):
                target["calls"] += 1
                if entry.get("usage_available"):
                    target["calls_with_usage"] += 1
                for name in TOKEN_FIELDS:
                    target[name] += _to_int(entry.get(name))
        return {"calls": len(records), "total": total, "by_role": by_role}


class ToolUsageTracker:
    """Thread-safe accumulator of billable search-tool calls for one run."""

    def __init__(self):
        self._lock = threading.Lock()
        self._records: list[dict] = []

    def record(
        self,
        provider: str,
        tool: str,
        *,
        agent: Optional[str] = None,
        platform: Optional[str] = None,
        query: Optional[str] = None,
    ) -> dict:
        entry = {
            "timestamp": time.time(),
            "provider": provider,
            "tool": tool,
            "agent": agent,
            "platform": platform,
            "query": query[:200] if query else None,
        }
        with self._lock:
            self._records.append(entry)
        return entry

    def records(self) -> list[dict]:
        with self._lock:
            return [dict(entry) for entry in self._records]

    def snapshot(self) -> dict:
        with self._lock:
            records = [dict(entry) for entry in self._records]
        by_provider: dict[str, int] = {}
        by_tool: dict[str, int] = {}
        for entry in records:
            provider = entry.get("provider") or "unknown"
            tool = entry.get("tool") or "unknown"
            by_provider[provider] = by_provider.get(provider, 0) + 1
            by_tool[tool] = by_tool.get(tool, 0) + 1
        return {"total": len(records), "by_provider": by_provider, "by_tool": by_tool}


def _current_runtime() -> Any:
    from deep_research.runtime import get_runtime

    return get_runtime()


def record_llm_usage(
    response: Any,
    *,
    role: str,
    agent: Optional[str] = None,
    platform: Optional[str] = None,
    iteration: Optional[int] = None,
    model: Optional[str] = None,
) -> dict:
    """Record one LLM response and emit a content-free ``llm_usage`` event."""
    runtime = _current_runtime()
    entry = runtime.usage.record(
        response,
        role=role,
        agent=agent,
        platform=platform,
        iteration=iteration,
        model=model,
    )
    runtime.observer.emit(
        _events.LLM_USAGE,
        phase="usage",
        agent=agent or role,
        platform=platform,
        iteration=iteration,
        role=role,
        model=entry["model"],
        model_inferred=entry["model_inferred"],
        provider=entry["provider"],
        usage_available=entry["usage_available"],
        **{name: entry[name] for name in TOKEN_FIELDS},
    )
    return entry


def record_tool_call(
    provider: str,
    tool: str,
    *,
    agent: Optional[str] = None,
    platform: Optional[str] = None,
    query: Optional[str] = None,
) -> Optional[dict]:
    """Record one billable search call and emit a ``tool_usage`` event."""
    runtime = _current_runtime()
    tracker = getattr(runtime, "tool_usage", None)
    if tracker is None:
        return None
    entry = tracker.record(provider, tool, agent=agent, platform=platform, query=query)
    runtime.observer.emit(
        _events.TOOL_USAGE,
        phase="usage",
        agent=agent or provider,
        platform=platform,
        provider=provider,
        tool=tool,
    )
    return entry


def format_usage_summary(snapshot: dict, tool_snapshot: Optional[dict] = None) -> str:
    """Render a compact, multi-line token/search summary for the console."""
    total = snapshot.get("total", empty_bucket())
    lines = [
        "Token usage: "
        f"input {total['input_tokens']:,} (cache read {total['cache_read_tokens']:,}, "
        f"cache write {total['cache_creation_tokens']:,}), "
        f"output {total['output_tokens']:,}, total {total['total_tokens']:,} "
        f"across {total['calls']} LLM calls"
    ]
    for role, bucket in sorted(snapshot.get("by_role", {}).items()):
        lines.append(
            f"  {role}: {bucket['total_tokens']:,} tokens "
            f"({bucket['input_tokens']:,} in / {bucket['output_tokens']:,} out, "
            f"{bucket['cache_read_tokens']:,} cache read, {bucket['calls']} calls)"
        )
    if tool_snapshot:
        lines.append(
            "Search calls: "
            + str(tool_snapshot.get("total", 0))
            + " ("
            + ", ".join(
                f"{name} {count}" for name, count in sorted(
                    tool_snapshot.get("by_provider", {}).items())
            )
            + ")"
        )
    return "\n".join(lines)


__all__ = [
    "TOKEN_FIELDS",
    "empty_bucket",
    "extract_usage",
    "UsageTracker",
    "ToolUsageTracker",
    "record_llm_usage",
    "record_tool_call",
    "format_usage_summary",
]
