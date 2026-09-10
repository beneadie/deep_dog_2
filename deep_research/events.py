"""Product-level structured events for the Deep Dog research engine.

Events are semantic, product-level notifications — NOT raw model reasoning,
NOT chain-of-thought, NOT provider payloads, and never secrets. Each event
carries a stable ``type`` plus safe metadata a host application can persist
and render (counts, titles, URLs, statuses, phase, agent/platform, iteration).

Sinks may be synchronous or asynchronous. Event *production* is synchronous
(the graph nodes are sync function bodies inside async nodes), so events are
buffered per-run and flushed by the runtime driver between graph steps. See
``deep_research.observer.Observer``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, List, Protocol

# ── Event types ──────────────────────────────────────────────────────────
# Product-level lifecycle events. Keep names stable: the host application
# persists these; renaming is a breaking change for persisted event streams.

RUN_STARTED = "run_started"
CONFIG_VALIDATED = "config_validated"
SCOPE_STARTED = "scope_started"
SCOPE_COMPLETED = "scope_completed"
DRAFT_STARTED = "draft_started"
DRAFT_COMPLETED = "draft_completed"
SUPERVISOR_ITERATION = "supervisor_iteration"
DELEGATION_STARTED = "delegation_started"
SUBAGENT_STARTED = "subagent_started"
SUBAGENT_COMPLETED = "subagent_completed"
SUBAGENT_FAILED = "subagent_failed"
SOURCE_FOUND = "source_found"
SOURCE_READ = "source_read"
SOURCE_SAVED = "source_saved"
REPORT_STARTED = "report_started"
CITATIONS_VALIDATED = "citations_validated"
SUBTopic_GENERATION_STARTED = "subtopic_generation_started"
SUBTopic_GENERATION_COMPLETED = "subtopic_generation_completed"
RUN_COMPLETED = "run_completed"
RUN_FAILED = "run_failed"
RUN_CANCELLED = "run_cancelled"
RUN_TIMED_OUT = "run_timed_out"

# Optional trace-lite events (not chain-of-thought): coarse phase markers.
PHASE_STARTED = "phase_started"
PHASE_COMPLETED = "phase_completed"

EVENT_TYPES = frozenset({
    RUN_STARTED, CONFIG_VALIDATED, SCOPE_STARTED, SCOPE_COMPLETED, DRAFT_STARTED, DRAFT_COMPLETED,
    SUPERVISOR_ITERATION, DELEGATION_STARTED, SUBAGENT_STARTED,
    SUBAGENT_COMPLETED, SUBAGENT_FAILED, SOURCE_FOUND, SOURCE_READ,
    SOURCE_SAVED, REPORT_STARTED, CITATIONS_VALIDATED,
    SUBTopic_GENERATION_STARTED, SUBTopic_GENERATION_COMPLETED,
    RUN_COMPLETED, RUN_FAILED, RUN_CANCELLED, RUN_TIMED_OUT,
    PHASE_STARTED, PHASE_COMPLETED,
})

# Event types that must NEVER carry content that could include hidden
# reasoning or provider internals. (Reference set — the observer enforces a
# metadata allowlist per event type.)
_SENSITIVE_FREE = EVENT_TYPES


@dataclass
class ResearchEvent:
    """A single product-level research event."""

    type: str
    run_id: str | None = None
    timestamp: float = field(default_factory=time.time)
    phase: str | None = None
    agent: str | None = None
    platform: str | None = None
    iteration: int | None = None
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "phase": self.phase,
            "agent": self.agent,
            "platform": self.platform,
            "iteration": self.iteration,
            "payload": self.payload,
        }


# ── Sink protocol ────────────────────────────────────────────────────────
# A sink is anything with an (async or sync) ``emit(event)`` method, or a
# plain callable taking a single ResearchEvent.


class EventSink(Protocol):
    def emit(self, event: ResearchEvent) -> Any:  # pragma: no cover - protocol
        ...


async def deliver_event(sink: Any, event: ResearchEvent) -> None:
    """Deliver one event to a sink that may be sync or async."""
    if sink is None:
        return
    if isinstance(sink, EventCollector):
        await sink.emit(event)
        return
    emit = getattr(sink, "emit", None)
    if emit is not None:
        result = emit(event)
        if hasattr(result, "__await__"):
            await result
        return
    if callable(sink):
        result = sink(event)
        if hasattr(result, "__await__"):
            await result
        return


class EventCollector:
    """In-memory sink that records every event. Useful for tests and simple hosts."""

    def __init__(self) -> None:
        self.events: list[ResearchEvent] = []

    async def emit(self, event: ResearchEvent) -> None:
        self.events.append(event)

    def types(self) -> list[str]:
        return [e.type for e in self.events]
