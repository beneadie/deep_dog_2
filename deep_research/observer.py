"""Per-run observability state for the Deep Dog engine.

The legacy ``deep_research/observability.py`` used process-global state (one
log folder, one source counter, one research trace), which is unsafe when many
runs execute concurrently in one API process. This module replaces that global
state with a per-run ``Observer``.

Each ``Observer`` owns:
  - run ID
  - sequence counters (sub-agent id, source id)
  - an in-memory source registry
  - research-trace entries
  - an event queue (flushed by the runtime driver to the host event sink)
  - an optional artifact sink (host storage for reports/traces/source bundles)
  - an optional local file output path (CLI logs)

``Observer.emit`` is synchronous (called from graph node bodies); events are
buffered and flushed between graph steps by the integration driver, which
adapts sync or async sinks.
"""

from __future__ import annotations

import json
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from deep_research.events import ResearchEvent, deliver_event
from deep_research.run_config import RunConfig


@dataclass
class Artifact:
    """A named output artifact produced by a run."""

    name: str
    data: Any
    mime: str = "text/markdown"


# An artifact sink stores named artifacts. May be sync or async.
class ArtifactSink:
    def store(self, name: str, data: Any) -> Any:  # pragma: no cover - protocol
        ...


async def deliver_artifact(sink: Any, name: str, data: Any) -> None:
    if sink is None:
        return
    store = getattr(sink, "store", None)
    if store is None and callable(sink):
        store = sink
    if store is None:
        return
    result = store(name, data)
    if hasattr(result, "__await__"):
        await result


class Observer:
    """Isolated, per-run observability + event state."""

    def __init__(
        self,
        run_id: str,
        config: RunConfig,
        event_sink: Any = None,
        artifact_sink: Any = None,
        log_folder: Optional[Path] = None,
        console_enabled: bool = True,
    ):
        self.run_id = run_id
        self.config = config
        self.event_sink = event_sink
        self.artifact_sink = artifact_sink
        self.log_folder: Optional[Path] = None
        self.console_enabled = console_enabled

        # counters / registries
        self._lock = threading.Lock()
        self._sub_agent_counter = 0
        self._source_counter = 0
        self._sub_agent_start_times: dict[int, float] = {}
        self.sources: dict[int, dict] = {}
        self._trace: list[dict] = []
        self.conductor_turns: list[dict] = []
        self.subagent_logs: list[dict] = []

        # event queue (flushed by the runtime driver)
        self._pending: deque[ResearchEvent] = deque()

        if log_folder is not None:
            self.set_log_folder(log_folder)

    # ── File output configuration ──────────────────────────────────────
    def set_log_folder(self, folder: Path) -> Path:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "sub_agents").mkdir(exist_ok=True)
        self.log_folder = folder
        return folder

    def init_run_folder(self, output_dir: Path, timestamp: str) -> Path:
        """Create the classic <output_dir>/research_<timestamp> log folder."""
        run_folder = Path(output_dir) / f"research_{timestamp}"
        self._sub_agent_counter = 0
        self._source_counter = 0
        return self.set_log_folder(run_folder)

    # ── Events ─────────────────────────────────────────────────────────
    def emit(
        self,
        type_: str,
        *,
        phase: Optional[str] = None,
        agent: Optional[str] = None,
        platform: Optional[str] = None,
        iteration: Optional[int] = None,
        **payload: Any,
    ) -> ResearchEvent:
        event = ResearchEvent(
            type=type_,
            run_id=self.run_id,
            phase=phase,
            agent=agent,
            platform=platform,
            iteration=iteration,
            payload=dict(payload),
        )
        with self._lock:
            self._pending.append(event)
        return event

    async def flush_events(self) -> None:
        """Deliver buffered events to the sink (sync or async)."""
        while True:
            with self._lock:
                if not self._pending:
                    return
                event = self._pending.popleft()
            await deliver_event(self.event_sink, event)

    async def put_artifact(self, name: str, data: Any) -> None:
        await deliver_artifact(self.artifact_sink, name, data)

    # ── Sequence counters ──────────────────────────────────────────────
    def next_subagent_id(self) -> int:
        with self._lock:
            self._sub_agent_counter += 1
            return self._sub_agent_counter

    def mark_subagent_start(self, agent_id: int) -> None:
        import time
        with self._lock:
            self._sub_agent_start_times[agent_id] = time.time()

    def mark_subagent_end(self, agent_id: int) -> float:
        import time
        with self._lock:
            start = self._sub_agent_start_times.pop(agent_id, None)
        return (time.time() - start) if start is not None else 0.0

    def reset_console(self) -> None:
        with self._lock:
            self._sub_agent_counter = 0
            self._sub_agent_start_times = {}

    # ── Source logging ─────────────────────────────────────────────────
    def log_source(self, tool_name: str, link: str, content: str) -> int:
        """Record a source. Returns the assigned source id, or -1 when the
        source log is disabled (mirrors the legacy behaviour)."""
        if not self.config.enable_source_log:
            return -1
        with self._lock:
            self._source_counter += 1
            source_id = self._source_counter
            entry = {
                "id": source_id,
                "tool": tool_name,
                "link": link,
                "content": content,
                "timestamp": datetime.now().isoformat(),
            }
            self.sources[source_id] = entry
        if self.config.log_mode in ("file", "both") and self.log_folder is not None:
            try:
                sources_file = self.log_folder / "sources.jsonl"
                with open(sources_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            except OSError:  # pragma: no cover - best-effort file logging
                pass
        return source_id

    def aggregate_sources(self) -> dict:
        with self._lock:
            return {
                sid: {
                    "tool": e["tool"],
                    "link": e["link"],
                    "content": e["content"],
                }
                for sid, e in self.sources.items()
            }

    # ── Conductor / sub-agent logs (file + in-memory) ──────────────────
    def log_conductor_turn(self, system_prompt: str, messages: list, response: Any,
                           elapsed_minutes: float, iteration: int) -> None:
        response_content = ""
        tool_calls = []
        if hasattr(response, "content"):
            response_content = str(response.content)
        if hasattr(response, "tool_calls"):
            tool_calls = [
                {"name": tc.get("name", ""), "args": tc.get("args", {})}
                for tc in response.tool_calls
            ]
        turn_data = {
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "elapsed_minutes": round(elapsed_minutes, 2),
            "system_prompt": system_prompt,
            "messages_count": len(messages),
            "response": response_content[:2000] + ("..." if len(response_content) > 2000 else ""),
            "tool_calls": tool_calls,
        }
        self.conductor_turns.append(turn_data)
        if self.config.log_mode in ("file", "both") and self.log_folder is not None:
            try:
                log_file = self.log_folder / "conductor_log.json"
                log_data = {"agent_type": "conductor", "turns": list(self.conductor_turns)}
                with open(log_file, "w", encoding="utf-8") as f:
                    json.dump(log_data, f, indent=2, ensure_ascii=False)
            except OSError:  # pragma: no cover
                pass

    def log_sub_agent(self, research_topic: str, system_prompt: str, compressed_research: str,
                      agent_type: str = "research_agent", search_queries: list = None,
                      agent_number: Optional[int] = None) -> None:
        with self._lock:
            self._sub_agent_counter += 1
            number = agent_number if agent_number is not None else self._sub_agent_counter
        log_data = {
            "agent_type": agent_type,
            "agent_number": number,
            "timestamp": datetime.now().isoformat(),
            "research_topic": research_topic,
            "search_queries": search_queries or [],
            "system_prompt": system_prompt,
            "compressed_research": (compressed_research or "")[:5000]
            + ("..." if compressed_research and len(compressed_research) > 5000 else ""),
        }
        self.subagent_logs.append(log_data)
        if self.config.log_mode not in ("file", "both") or self.log_folder is None:
            return
        try:
            log_file = self.log_folder / "sub_agents" / f"sub_agent_{number:03d}.json"
            with open(log_file, "w", encoding="utf-8") as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
            if self.config.save_subagent_reports_to_file:
                md_file = self.log_folder / "sub_agents" / f"sub_agent_{number:03d}.md"
                md_content = (
                    f"# {agent_type} — sub-agent {number:03d}\n\n"
                    f"**Topic:** {research_topic}\n\n"
                    f"**Queries:** {', '.join(search_queries or [])}\n\n"
                    "---\n\n"
                    f"{compressed_research}"
                )
                with open(md_file, "w", encoding="utf-8") as f:
                    f.write(md_content)
        except OSError:  # pragma: no cover
            pass

    # ── Research trace ─────────────────────────────────────────────────
    def log_trace_delegation(self, research_topic: str) -> int:
        """Log a supervisor delegation. Returns the loop index to update later."""
        if not self.config.enable_research_trace:
            return -1
        with self._lock:
            self._trace.append({
                "loop_number": len(self._trace) + 1,
                "timestamp": datetime.now().isoformat(),
                "research_topic": research_topic,
                "findings": None,
                "supervisor_reaction": None,
            })
            return len(self._trace) - 1

    def log_trace_findings(self, loop_index: int, findings: str) -> None:
        if not self.config.enable_research_trace or loop_index < 0:
            return
        with self._lock:
            if 0 <= loop_index < len(self._trace):
                self._trace[loop_index]["findings"] = findings

    def log_trace_supervisor_reaction(self, reaction: str) -> None:
        if not self.config.enable_research_trace:
            return
        with self._lock:
            for loop in reversed(self._trace):
                if loop.get("findings") is not None and loop.get("supervisor_reaction") is None:
                    loop["supervisor_reaction"] = reaction
                    break

    def get_research_trace(self) -> list:
        with self._lock:
            return list(self._trace)

    def clear_trace(self) -> None:
        with self._lock:
            self._trace = []
