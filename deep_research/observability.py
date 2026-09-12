"""
Observability Module for Deep Research Agent (runtime-aware shim).

Historically this module kept process-global state (one log folder, source
counter, research trace) — unsafe for concurrent runs. State now lives in a
per-run ``deep_research.observer.Observer`` obtained from the active runtime
context (``deep_research.runtime.get_runtime()``). These functions remain as
backwards-compatible shims so existing call sites (platform agent engine,
search utilities, supervisor) keep working unchanged: when a per-run runtime
is active they route to that run's observer; otherwise they route to the
default environment runtime (legacy standalone behaviour).

The CLI may continue to write local files by configuring the observer's log
folder (``init_run_folder``); API runs use the in-memory observer/event sink
instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from deep_research.runtime import get_runtime


def init_run_folder(output_dir: Path, timestamp: str) -> Path:
    """Initialize the logging folder for a research run (per-run observer)."""
    return get_runtime().observer.init_run_folder(output_dir, timestamp)


def get_log_folder() -> Optional[Path]:
    """Get the current run's log folder."""
    return get_runtime().observer.log_folder


def log_conductor_turn(
    system_prompt: str,
    messages: list,
    response: Any,
    elapsed_minutes: float,
    iteration: int,
) -> None:
    """Log a single turn of the Conductor agent."""
    get_runtime().observer.log_conductor_turn(
        system_prompt, messages, response, elapsed_minutes, iteration
    )


def log_sub_agent(
    research_topic: str,
    system_prompt: str,
    compressed_research: str,
    agent_type: str = "research_agent",
    search_queries: list = None,
    discovery: bool = False,
    output_mode: str = None,
) -> None:
    """Log a sub-agent's full lifecycle."""
    get_runtime().observer.log_sub_agent(
        research_topic=research_topic,
        system_prompt=system_prompt,
        compressed_research=compressed_research,
        agent_type=agent_type,
        search_queries=search_queries,
        discovery=discovery,
        output_mode=output_mode,
    )


def log_source(tool_name: str, link: str, content: str) -> int:
    """Log a source (URL + content) discovered during research."""
    return get_runtime().observer.log_source(tool_name, link, content)


def aggregate_sources() -> dict:
    """Aggregate all sources from the current run into a dictionary."""
    return get_runtime().observer.aggregate_sources()


# ===== RESEARCH TRACE LOGGING =====


def clear_research_trace() -> None:
    """Clear the research trace for a new run."""
    get_runtime().observer.clear_trace()


def log_trace_delegation(research_topic: str) -> int:
    """Log when the supervisor delegates a research task to a subagent."""
    return get_runtime().observer.log_trace_delegation(research_topic)


def log_trace_findings(loop_index: int, findings: str) -> None:
    """Log the findings returned by a subagent."""
    get_runtime().observer.log_trace_findings(loop_index, findings)


def log_trace_supervisor_reaction(reaction: str) -> None:
    """Log the supervisor's reaction (think_tool) after receiving findings."""
    get_runtime().observer.log_trace_supervisor_reaction(reaction)


def get_research_trace() -> list:
    """Get the accumulated research trace for the current run."""
    return get_runtime().observer.get_research_trace()
