"""Stable public integration entry point for the Deep Dog engine.

The research_agent_api product imports exactly one entry point from Deep Dog:

    from deep_research.integration import run_research, RunConfig, RuntimeOptions

``run_research`` executes the existing public research graph with a fully
per-run configuration, per-run provider credentials, isolated observability,
structured events, cooperative cancellation, and an optional
external LangGraph checkpointer — without writing files and without mutating
``os.environ``.

Internal graph objects (``research_agent_full``, ``multi_agent_supervisor``,
``agents.base``, ``config``) are implementation details and are NOT imported
by the host application.
"""

from __future__ import annotations

import asyncio
import logging
import random
import string
import time
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from deep_research import events as _events
from deep_research.cancellation import (
    CancellationToken,
    RunCancelledError,
    as_cancellation_checker,
)
from deep_research.events import EventCollector
from deep_research.models import Credentials, ModelFactory
from deep_research.observer import Observer
from deep_research.run_config import RunConfig
from deep_research.runtime import RuntimeContext, runtime_scope
from deep_research.trace import TraceCollector, TraceRecord

_log = logging.getLogger(__name__)


class RunStatus(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    PARTIAL = "partial"


@dataclass
class CredentialCheck:
    """Pre-flight credential report for a (config, credentials) pair.

    ``missing_required`` lists env-var names that will degrade or fail the run
    (model chain entries, search provider keys, reddit creds when reddit is
    enabled). ``optional_missing`` lists nice-to-have keys whose absence only
    removes optional capabilities (perplexity, pubmed/sec contact email).
    """

    missing_required: list = field(default_factory=list)
    optional_missing: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.missing_required

    def to_dict(self) -> dict:
        return {
            "missing_required": list(self.missing_required),
            "optional_missing": list(self.optional_missing),
        }


def validate_credentials(config: Optional[RunConfig] = None,
                         credentials: Optional[Credentials] = None) -> CredentialCheck:
    """Report which provider keys a run would use and which are missing.

    Pure pre-flight check: reads nothing from the environment, mutates
    nothing, and never raises. Hosts call it before ``run_research`` to
    reject bad requests early; ``run_research`` calls it once and surfaces
    the result as a ``config_validated`` event + ``run_metadata`` entry.

    Model chains are checked per-entry (a chain is usable if ANY entry has
    its key; entries without keys are reported only when the whole chain is
    unusable, matching the engine's degradation semantics).
    """
    cfg = (config or RunConfig()).finalize()
    creds = credentials if credentials is not None else Credentials.from_env()

    missing_required: list[str] = []
    optional_missing: list[str] = []

    def _missing(names: list) -> bool:
        return not any(creds.has(n) for n in names)

    # ── Model chains (per role) ────────────────────────────────────────
    from deep_research.config import model_required_keys
    roles = (
        ("supervisor", cfg.supervisor_model_fallback_chain,
         cfg.resolved_role_route("supervisor")),
        ("draft_report", cfg.draft_report_model_fallback_chain,
         cfg.resolved_role_route("draft")),
        ("subagent", cfg.subagent_model_fallback_chain,
         cfg.resolved_role_route("subagent")),
    )
    for role, chain, route in roles:
        usable = any(
            not _missing(model_required_keys(name, route)) or
            not model_required_keys(name, route)
            for name in (chain or [])
        )
        if not usable:
            missing_required.extend(
                model_required_keys(chain[0], route) if chain
                else [f"<no model in {role} chain>"])

    # ── Web search engine keys ─────────────────────────────────────────
    engine = (cfg.web_search_engine or "").lower()
    if engine in ("tavily", "both") and _missing(["TAVILY_API_KEY"]):
        missing_required.append("TAVILY_API_KEY")
    if engine in ("exa", "both") and _missing(["EXA_API_KEY"]):
        missing_required.append("EXA_API_KEY")

    # ── Enabled-agent platform keys ────────────────────────────────────
    enabled = set(cfg.enabled_agents or [])
    if "ResearchReddit" in enabled:
        reddit_keys = ["REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET",
                       "REDDIT_USERNAME", "REDDIT_PASSWORD"]
        if any(creds.has(k) is False for k in reddit_keys):
            missing_required.extend(
                k for k in reddit_keys if not creds.has(k))
    if "ResearchPubMed" in enabled or "ResearchSEC" in enabled:
        if not creds.has("PUBMED_EMAIL") and not creds.has("SEC_EDGAR_CONTACT_EMAIL"):
            optional_missing.append("PUBMED_EMAIL")

    # ── Optional capabilities ──────────────────────────────────────────
    if not creds.has("PERPLEXITY_API_KEY"):
        optional_missing.append("PERPLEXITY_API_KEY")

    return CredentialCheck(
        missing_required=sorted(set(missing_required)),
        optional_missing=optional_missing,
    )


@dataclass
class RuntimeOptions:
    """Host-supplied runtime hooks for one run (all optional).

    - ``event_sink``:  sync/async sink for ResearchEvent objects.
    - ``artifact_sink``: object with ``store(name, data)`` (sync or async).
    - ``cancellation``: CancellationToken, ``() -> bool``, or object with
      ``is_cancelled()``.
    - ``checkpointer``: external LangGraph checkpointer (e.g. Postgres saver).
    - ``run_id`` / ``thread_id``: supplied by the host (thread_id must be
      stable across resume).
    - ``output_dir``: when set (CLI), a run log folder is created there and
      local file behaviour follows the config's ``output_mode``.
    """

    run_id: Optional[str] = None
    thread_id: Optional[str] = None
    event_sink: Any = None
    trace_sink: Any = None
    artifact_sink: Any = None
    cancellation: Any = None
    checkpointer: Any = None
    output_dir: Optional[Path] = None
    console_enabled: bool = True


@dataclass
class ResearchResult:
    """Structured result of a research run (never parsed from files/console)."""

    status: str = RunStatus.COMPLETED.value
    run_id: str = ""
    thread_id: str = ""
    final_report: str = ""
    research_brief: str = ""
    draft_report: str = ""
    notes: list = field(default_factory=list)
    source_registry: list = field(default_factory=list)
    curated_sources: list = field(default_factory=list)
    sources: dict = field(default_factory=dict)          # observer source registry
    trace: list = field(default_factory=list)             # research trace when enabled
    secondary_reports: list = field(default_factory=list)
    run_metadata: dict = field(default_factory=dict)
    usage: dict = field(default_factory=dict)
    failure: Optional[str] = None
    logs: list = field(default_factory=list)   # structured trace records (dicts)

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "run_id": self.run_id,
            "thread_id": self.thread_id,
            "final_report": self.final_report,
            "research_brief": self.research_brief,
            "draft_report": self.draft_report,
            "notes": self.notes,
            "source_registry": self.source_registry,
            "curated_sources": self.curated_sources,
            "sources": self.sources,
            "trace": self.trace,
            "secondary_reports": self.secondary_reports,
            "run_metadata": self.run_metadata,
            "usage": self.usage,
            "failure": self.failure,
            "logs": self.logs,
        }


def _generate_id(prefix: str) -> str:
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{suffix}"


# Interval (seconds) for the background flusher that streams buffered events and
# trace records while the graph is inside a long-running nested node (e.g. the
# supervisor subgraph awaiting sub-agents). Without it, records only flush when
# the top-level graph yields, which can be minutes later — or never, on a crash.
_FLUSH_INTERVAL_SECONDS = 0.2


async def _periodic_flush(observer: Observer, interval: float = _FLUSH_INTERVAL_SECONDS) -> None:
    """Flush the observer's buffers on a fixed cadence until cancelled.

    A failing host sink must not take down the run: the error is logged and the
    cadence continues. A persistent failure still surfaces through the normal
    end-of-run flush path.
    """
    while True:
        await asyncio.sleep(interval)
        try:
            await observer.flush_events()
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - background liveness must not crash the run
            _log.exception("periodic trace/event flush failed")


def _materialize(result: dict, observer: Observer, trace_enabled: bool) -> dict:
    """Extract stable, JSON-safe result fields from the final graph state."""
    final_report = result.get("final_report") or result.get("draft_report") or ""
    return {
        "final_report": final_report,
        "research_brief": result.get("research_brief", ""),
        "draft_report": result.get("draft_report", ""),
        "notes": list(result.get("notes") or []),
        "source_registry": list(result.get("source_registry") or []),
        "curated_sources": list(result.get("curated_sources") or []),
        "secondary_reports": list(result.get("secondary_reports") or []),
        "research_iterations": int(result.get("research_iterations") or 0),
        "aborted": bool(result.get("aborted")),
        "abort_reason": result.get("abort_reason") or None,
        "sources": observer.aggregate_sources(),
        "trace": observer.get_research_trace() if trace_enabled else [],
    }


async def run_research(
    prompt: str,
    config: Optional[RunConfig] = None,
    runtime: Optional[RuntimeOptions] = None,
    credentials: Optional[Credentials] = None,
) -> ResearchResult:
    """Run the Deep Dog research graph once, returning a structured result.

    Args:
        prompt: the research prompt/question.
        config: resolved per-run configuration. ``None`` = env defaults.
        runtime: host hooks (events, cancellation, checkpointer, ids, files).
        credentials: per-run provider credentials (host-supplied). ``None``
            falls back to the process environment.

    Returns:
        ``ResearchResult`` with status completed/failed/cancelled/timed_out.
    """
    started_monotonic = time.monotonic()
    options = runtime or RuntimeOptions()
    cfg = (config or RunConfig()).finalize()

    run_id = options.run_id or _generate_id("run")
    thread_id = options.thread_id or run_id
    creds = credentials if credentials is not None else Credentials.from_env()

    # Observer: events + optional file log folder (CLI).
    observer = Observer(
        run_id=run_id,
        config=cfg,
        event_sink=options.event_sink,
        trace_sink=options.trace_sink,
        artifact_sink=options.artifact_sink,
        console_enabled=options.console_enabled,
        credentials=creds,
    )
    if options.output_dir is not None and cfg.log_mode in ("file", "both"):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        observer.init_run_folder(Path(options.output_dir), timestamp)

    checkpointer = options.checkpointer if options.checkpointer is not None else InMemorySaver()

    context = RuntimeContext(
        run_id=run_id,
        thread_id=thread_id,
        config=cfg,
        credentials=creds,
        models=ModelFactory(cfg, creds),
        observer=observer,
        cancellation=options.cancellation,
        checkpointer=checkpointer,
        console_enabled=options.console_enabled,
        start_monotonic=started_monotonic,
        host=options,
    )
    cancellation_check = as_cancellation_checker(options.cancellation)

    result = ResearchResult(status=RunStatus.COMPLETED.value, run_id=run_id, thread_id=thread_id)

    # Compile the full graph with the run's checkpointer.
    from deep_research.research_agent_full import build_research_agent
    graph = build_research_agent(checkpointer=checkpointer)

    graph_config = {
        "configurable": {"thread_id": thread_id, "recursion_limit": 200},
    }
    inputs = {
        "messages": [HumanMessage(content=prompt)],
        "start_time": time.time(),
        "config_snapshot": cfg.to_dict(),
    }

    with runtime_scope(context):
        observer.emit(_events.RUN_STARTED, phase="run", agent="runner",
                      prompt=prompt[:2000], run_profile=cfg.profile)
        observer.emit_trace("run_started", phase="run", agent="runner",
                            prompt=prompt, profile=cfg.profile)
        # Pre-flight credential report (names only, never values).
        cred_check = validate_credentials(cfg, creds)
        observer.emit(_events.CONFIG_VALIDATED, phase="run", agent="runner",
                      missing_required=cred_check.missing_required,
                      optional_missing=cred_check.optional_missing)
        await observer.flush_events()

        state_holder: list[dict] = []

        async def _drive() -> None:
            """Stream graph values and cooperate with explicit host cancellation.

            Research timing is handled by the supervisor's transition to final
            writing. Never cancel the graph because that window has elapsed.
            """
            async for chunk in graph.astream(inputs, config=graph_config, stream_mode="values"):
                if isinstance(chunk, dict) and chunk:
                    state_holder[:] = [dict(chunk)]
                await observer.flush_events()
                if cancellation_check():
                    raise RunCancelledError("run cancelled by host")

        # Background flusher: streams records while nested nodes block the
        # top-level driver (supervisor subgraph, awaited sub-agents, etc.).
        flusher = asyncio.create_task(_periodic_flush(observer))
        try:
            await _drive()
        except RunCancelledError as e:
            await observer.flush_events()
            observer.emit(_events.RUN_CANCELLED, phase="run", agent="runner",
                          reason=str(e), elapsed_seconds=round(time.monotonic() - started_monotonic, 2))
            await observer.flush_events()
            result.status = RunStatus.CANCELLED.value
            result.failure = str(e) or "cancelled"
        except Exception as e:  # noqa: BLE001 - surface as failed result
            await observer.flush_events()
            observer.emit(_events.RUN_FAILED, phase="run", agent="runner",
                          error=type(e).__name__,
                          elapsed_seconds=round(time.monotonic() - started_monotonic, 2))
            await observer.flush_events()
            result.status = RunStatus.FAILED.value
            result.failure = f"{type(e).__name__}: {e}"
        finally:
            flusher.cancel()
            with suppress(asyncio.CancelledError):
                await flusher

        final_state: dict = state_holder[-1] if state_holder else {}

        # Build structured output from whatever final/partial state we have.
        mat = _materialize(final_state, observer, trace_enabled=bool(cfg.enable_research_trace))
        if result.status in (RunStatus.FAILED.value, RunStatus.TIMED_OUT.value, RunStatus.CANCELLED.value):
            if mat["final_report"] or mat["notes"] or mat["curated_sources"]:
                # We salvaged partial output.
                result.status = RunStatus.PARTIAL.value
        result.final_report = mat["final_report"]
        result.research_brief = mat["research_brief"]
        result.draft_report = mat["draft_report"]
        result.notes = mat["notes"]
        result.source_registry = mat["source_registry"]
        result.curated_sources = mat["curated_sources"]
        result.secondary_reports = mat["secondary_reports"]
        result.sources = mat["sources"]
        result.trace = mat["trace"]
        result.run_metadata = {
            "profile": cfg.profile,
            "prompt_version": cfg.prompt_version,
            "output_mode": cfg.output_mode,
            "enabled_agents": list(cfg.enabled_agents),
            "model_profile": {
                "supervisor_chain": list(cfg.supervisor_model_fallback_chain),
                "subagent_chain": list(cfg.subagent_model_fallback_chain),
                "draft_chain": list(cfg.draft_report_model_fallback_chain),
            },
            "research_iterations": mat["research_iterations"],
            "aborted": mat["aborted"],
            "abort_reason": mat["abort_reason"],
            "credential_check": cred_check.to_dict(),
            "elapsed_seconds": round(time.monotonic() - started_monotonic, 2),
        }
        result.usage = {
            "sources_found_logged": len(result.sources),
            "source_registry_entries": len(result.source_registry),
            "curated_sources": len(result.curated_sources),
            "subagent_log_entries": len(observer.subagent_logs),
            "trace_entries": len(result.trace),
        }

        # Terminal trace record (full report/brief/draft content).
        observer.emit_trace(
            "run_completed", phase="run", agent="runner",
            status=result.status,
            final_report=result.final_report,
            research_brief=result.research_brief,
            draft_report=result.draft_report,
            elapsed_seconds=round(time.monotonic() - started_monotonic, 2),
        )

        # Fail/cancel paths already emitted their specific terminal events.
        if result.status == RunStatus.COMPLETED.value:
            observer.emit(_events.RUN_COMPLETED, phase="run", agent="runner",
                          elapsed_seconds=round(time.monotonic() - started_monotonic, 2),
                          report_chars=len(result.final_report))
        await observer.flush_events()

        result.logs = observer.get_trace_records()

        # Deliver artifacts to the host artifact sink when provided.
        if observer.artifact_sink is not None:
            from deep_research.observer import deliver_artifact
            await deliver_artifact(observer.artifact_sink, "final_report", result.final_report)
            if result.research_brief:
                await deliver_artifact(observer.artifact_sink, "research_brief", result.research_brief)
            if result.trace:
                await deliver_artifact(observer.artifact_sink, "research_trace", result.trace)

    return result


# Convenience re-exports so the host imports from one module.
__all__ = [
    "run_research",
    "RunConfig",
    "Credentials",
    "RuntimeOptions",
    "ResearchResult",
    "RunStatus",
    "CredentialCheck",
    "validate_credentials",
    "CancellationToken",
    "EventCollector",
    "TraceRecord",
    "TraceCollector",
]
