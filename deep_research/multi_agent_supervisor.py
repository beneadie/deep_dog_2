
"""Multi-agent supervisor for coordinating research across multiple specialized agents.

This module implements a supervisor pattern where:
1. A supervisor agent coordinates research activities and delegates tasks
2. Multiple researcher agents work on specific sub-topics independently
3. Results are aggregated and compressed for final reporting

The supervisor uses parallel research execution to improve efficiency while
maintaining isolated context windows for each research topic.
"""

import asyncio
import logging

from typing_extensions import Literal

logger = logging.getLogger(__name__)

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    BaseMessage,
    SystemMessage,
    ToolMessage,
    filter_messages
)
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

from deep_research.agents import AGENT_REGISTRY
from deep_research.state_multi_agent_supervisor import (
    SupervisorState,
    ResearchComplete,
    AGENT_TOOL_SCHEMAS,
    AGENT_DESCRIPTIONS,
)
from deep_research.utils import get_today_str, extract_text_from_response
from deep_research.citation_utils import remap_codes, build_final_registry, finalize_citations
from deep_research.agents.shared.tools import think_tool
from deep_research.config import looks_like_refusal
from deep_research.runtime import get_runtime
from deep_research.cancellation import RunCancelledError
from deep_research import events as _events
import time
from deep_research.observability import log_conductor_turn, log_sub_agent, log_trace_delegation, log_trace_findings, log_trace_supervisor_reaction
from deep_research.console_logger import Colors
from deep_research import console_logger

# ── Per-run configuration helpers ──────────────────────────────────────
# The supervisor's tool set and timing/iteration limits are now derived from
# the active runtime's RunConfig + prompt library rather than import-time
# globals, so concurrent API runs never share supervisor settings.


def _active_supervisor_tools():
    """Build the supervisor tool list for the active runtime's config."""
    runtime = get_runtime()
    cfg = runtime.config
    tools = [ResearchComplete, think_tool]
    for name in cfg.enabled_agents:
        if name in AGENT_TOOL_SCHEMAS:
            tools.append(AGENT_TOOL_SCHEMAS[name])
    return tools


def _subagent_tool_names():
    """Names of registry-backed sub-agent tools enabled for this run."""
    return {n for n in get_runtime().config.enabled_agents if n in AGENT_REGISTRY}


def _supervisor_model():
    """Cached per-run supervisor model bound to the run's supervisor tools."""
    runtime = get_runtime()
    model = getattr(runtime, "_supervisor_model", None)
    if model is None:
        model = runtime.models.supervisor(tools=_active_supervisor_tools(), max_tokens=32000)
        setattr(runtime, "_supervisor_model", model)
    return model


def _final_report_writer_model():
    runtime = get_runtime()
    model = getattr(runtime, "_final_report_writer_model", None)
    if model is None:
        model = runtime.models.supervisor(max_tokens=40000)
        setattr(runtime, "_final_report_writer_model", model)
    return model


def _emit(type_: str, *, phase: str | None = None, agent: str | None = None,
          platform: str | None = None, iteration: int | None = None, **payload) -> None:
    get_runtime().observer.emit(type_, phase=phase, agent=agent, platform=platform,
                                iteration=iteration, **payload)


async def _run_subagent_with_progress(agent_graph, agent_input: dict, *,
                                      timeout: float, agent_id: int,
                                      tool_name: str, iteration: int):
    """Observe each task as it finishes; leave batch result ordering unchanged."""
    discovery = bool(agent_input.get("discovery", False))
    metadata = dict(
        phase="subagent", agent=tool_name, platform=tool_name,
        iteration=iteration, agent_id=agent_id,
        research_topic=agent_input.get("research_topic", ""),
        discovery=discovery, mode="discovery" if discovery else "research",
    )
    try:
        result = await asyncio.wait_for(agent_graph.ainvoke(agent_input), timeout=timeout)
    except (Exception, asyncio.CancelledError) as exc:
        status = "cancelled" if isinstance(exc, asyncio.CancelledError) else (
            "timed out" if isinstance(exc, asyncio.TimeoutError) else "failed")
        elapsed = console_logger.log_sub_agent_failed(
            agent_id, discovery=discovery, status=status)
        _emit(_events.SUBAGENT_FAILED, **metadata, error=type(exc).__name__,
              elapsed_seconds=elapsed)
        raise

    counts = dict(
        search_count=int(result.get("search_count", 0)),
        read_count=int(result.get("read_count", 0)),
        selected_count=len(result.get("saved_articles") or {}),
    )
    log_complete = (console_logger.log_discovery_complete if discovery
                    else console_logger.log_sub_agent_complete)
    elapsed = log_complete(agent_id, **counts)
    _emit(_events.SUBAGENT_COMPLETED, **metadata, **counts,
          source_count=len(result.get("source_registry") or []),
          elapsed_seconds=elapsed)
    return result


def get_notes_from_tool_calls(messages: list[BaseMessage]) -> list[str]:
    """Extract research notes from ToolMessage objects in supervisor message history.

    This function retrieves the compressed research findings that sub-agents
    return as ToolMessage content. When the supervisor delegates research to
    sub-agents via research tool calls, each sub-agent returns its
    compressed findings as the content of a ToolMessage. This function
    extracts all such ToolMessage content to compile the final research notes.

    Args:
        messages: List of messages from supervisor's conversation history

    Returns:
        List of research note strings extracted from ToolMessage objects
    """
    return [
        tool_msg.content
        for tool_msg in filter_messages(messages, include_types="tool")
        if getattr(tool_msg, "name", None) != "think_tool"
    ]


def _denoise_verdict_continue(messages: list[BaseMessage], current_tool_calls: list = None) -> bool:
    """True if the denoise reflection carries VERDICT: CONTINUE_RESEARCH.

    The denoise that accompanies ResearchComplete is the one emitted in the SAME
    turn (a think_tool tool_call in the current AIMessage, not yet a ToolMessage
    when the exit gate runs). Inspect that reflection first; fall back to the most
    recent think_tool ToolMessage in history only if there is no think_tool call
    in the current turn.
    """
    if current_tool_calls:
        for tc in current_tool_calls:
            if tc["name"] == "think_tool":
                reflection = str(tc.get("args", {}).get("reflection", "") or "")
                return "VERDICT: CONTINUE_RESEARCH" in reflection.upper()
    for m in reversed(messages):
        if isinstance(m, ToolMessage) and getattr(m, "name", None) == "think_tool":
            content = str(getattr(m, "content", "") or "")
            return "VERDICT: CONTINUE_RESEARCH" in content.upper()
    return False


def _classify_error(e: BaseException) -> Literal["fatal", "retryable", "bug"]:
    """Classify an exception into a recovery policy bucket.

    - fatal:     retrying cannot succeed (auth/billing/quota).
    - retryable: transient — retrying the same call is worthwhile (rate limit,
                 server error, timeout, connection).
    - bug:       unexpected internal error (KeyError, AttributeError, etc.).
    """
    text = str(e)
    lowered = text.lower()
    if any(marker in text for marker in ("RESOURCE_EXHAUSTED", "insufficient_quota", "Invalid API key", "authentication", "Forbidden")):
        return "fatal"
    if any(marker in lowered for marker in ("402", "403", "401", "payment required", "permission denied")):
        return "fatal"
    if any(marker in text for marker in ("429", "500", "502", "503", "504", "Internal Server Error", "Bad Gateway", "Service Unavailable", "Gateway Timeout")):
        return "retryable"
    if isinstance(e, (asyncio.TimeoutError, TimeoutError, ConnectionError, OSError)):
        return "retryable"
    return "bug"


def _has_research_findings(state: dict, all_curated_sources: list, new_registry_entries: list) -> bool:
    """Whether at least one iteration gathered findings of task value.

    Any non-empty curated source (with full text / relevance) or registry entry
    counts — either from prior turns (accumulated in state) or this turn.
    """
    return bool(
        all_curated_sources
        or new_registry_entries
        or state.get("curated_sources")
        or state.get("source_registry")
    )


def _should_salvage(
    state: dict,
    all_curated_sources: list,
    new_registry_entries: list,
    elapsed_minutes: float,
) -> bool:
    """Whether a stalled run should salvage a final report instead of aborting.

    Only when we're far enough through the research window AND at least one
    iteration produced findings of task value; otherwise abort (no fabrication
    from zero context).
    """
    has_findings = _has_research_findings(state, all_curated_sources, new_registry_entries)
    max_minutes = _supervisor_limits()["max_minutes"]
    elapsed_fraction = (
        elapsed_minutes / max_minutes if max_minutes > 0 else 0.0
    )
    return has_findings and elapsed_fraction >= _supervisor_limits()["salvage_fraction"]


def _complete_unanswered_tool_calls(decision_message, tool_messages: list, error_content: str) -> None:
    """Ensure every tool call in the decision message has a ToolMessage.

    When an exception interrupts tool execution mid-turn, some tool calls may
    not have results yet; the next supervisor call would be rejected by the API
    (assistant tool_calls must be followed by matching tool messages). Fill the
    gaps with error messages so the loop stays API-valid.
    """
    answered = {tm.tool_call_id for tm in tool_messages}
    for tc in (getattr(decision_message, "tool_calls", None) or []):
        if tc.get("id") and tc["id"] not in answered:
            tool_messages.append(
                ToolMessage(
                    content=error_content,
                    name=tc.get("name", ""),
                    tool_call_id=tc["id"],
                )
            )


# Ensure async compatibility for Jupyter environments
try:
    import nest_asyncio
    # Only apply if running in Jupyter/IPython environment
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            nest_asyncio.apply()
    except ImportError:
        pass  # Not in Jupyter, no need for nest_asyncio
except ImportError:
    pass  # nest_asyncio not available, proceed without it


# ===== SUPERVISOR NODES =====

def _supervisor_limits():
    """Active run's supervisor iteration/concurrency limits + timing config."""
    cfg = get_runtime().config
    return {
        "max_researcher_iterations": int(cfg.supervisor_max_iterations),
        "max_concurrent_researchers": int(cfg.supervisor_max_concurrent_research),
        "max_concurrent_discovery": int(cfg.supervisor_max_concurrent_discovery),
        "min_minutes": float(cfg.research_time_min_minutes),
        "max_minutes": float(cfg.research_time_max_minutes),
        "strict_minutes": float(cfg.strict_timeout_minutes),
        "salvage_fraction": float(cfg.findings_salvage_time_fraction),
        "subagent_timeout_seconds": int(cfg.subagent_timeout_seconds),
        "supervisor_timeout_seconds": int(cfg.supervisor_timeout_seconds),
    }

def _build_subagent_tools_block(enabled: list[str]) -> str:
    """Render the `<Available Tools>` sub-agent list from the active config.

    Only agents present in ENABLED_AGENTS (and with a known description) are
    described, so the supervisor prompt never advertises a tool it cannot call.
    """
    lines = ["**Research Sub-Agents** (launch independent deep-research sub-agents):"]
    for i, name in enumerate([n for n in enabled if n in AGENT_DESCRIPTIONS], 1):
        lines.append(f"{i}. **{name}**: {AGENT_DESCRIPTIONS[name]}")
    return "\n".join(lines)


def _build_supervisor_system_message(state: SupervisorState) -> str:
    """Byte-stable supervisor system prompt, shared by the supervisor and write nodes.

    Keeping this identical across turns (and across the final report write) is
    what lets the provider's prompt cache reuse the research-history prefix.
    The active prompt set and the timing/iteration limits are read from the
    per-run runtime config, so concurrent runs with different durations/model
    profiles get their own prompt text.
    """
    runtime = get_runtime()
    cfg = runtime.config
    limits = _supervisor_limits()
    lead_prompt = runtime.prompts.get(
        "lead_researcher_with_multiple_steps_diffusion_double_check_prompt", "")
    example_report = runtime.prompts.get("example_report", "")
    format_kwargs = dict(
        date=get_today_str(),
        max_concurrent_research_units=limits["max_concurrent_researchers"],
        max_concurrent_discovery_units=limits["max_concurrent_discovery"],
        max_researcher_iterations=limits["max_researcher_iterations"],
        min_research_time_minutes=int(limits["min_minutes"]),
        max_research_time_minutes=int(limits["max_minutes"]),
    )
    # Only pass example_report if the active prompt expects it.
    if "{example_report}" in lead_prompt:
        format_kwargs["example_report"] = example_report
    # Only pass target_language if the prompt expects it
    if "{target_language}" in lead_prompt:
        format_kwargs["target_language"] = state.get("target_language") or cfg.target_language_fallback
    # Only pass the enabled sub-agent list if the prompt expects it
    if "{available_subagents}" in lead_prompt:
        format_kwargs["available_subagents"] = _build_subagent_tools_block(cfg.enabled_agents)
    return lead_prompt.format(**format_kwargs)

async def supervisor(state: SupervisorState) -> Command[Literal["supervisor_tools"]]:
    """Coordinate research activities.

    Analyzes the research brief and current progress to decide:
    - What research topics need investigation
    - Whether to conduct parallel research
    - When research is complete

    Args:
        state: Current supervisor state with messages and research progress

    Returns:
        Command to proceed to supervisor_tools node with updated state
    """
    runtime = get_runtime()
    runtime.check_cancelled()
    limits = _supervisor_limits()
    supervisor_messages = state.get("supervisor_messages", [])

    # Prepare system message with current date and constraints. The elapsed-time
    # note is NOT injected here: supervisor_tools folds it into the last tool
    # message so this conversation stays append-only and the provider's prompt
    # cache can reuse the accumulated history prefix across turns.
    import time
    start_time = state.get("start_time", 0.0)
    elapsed_minutes = 0.0
    if start_time > 0:
        elapsed_minutes = (time.time() - start_time) / 60.0

    system_message = _build_supervisor_system_message(state)

    messages = [SystemMessage(content=system_message)] + supervisor_messages

    # Make decision about next research steps (bounded — retry retryable errors
    # once, then end gracefully). Fatal errors (quota/auth) are raised so the
    # graph can abort loudly instead of fabricating a report.
    response = None
    for attempt in (1, 2):
        try:
            response = await asyncio.wait_for(
                _supervisor_model().ainvoke(messages),
                timeout=limits["supervisor_timeout_seconds"],
            )
            break
        except Exception as e:
            kind = _classify_error(e)
            if kind == "fatal":
                raise
            logger.warning(
                "Supervisor decision failed (attempt %d, %s): %s",
                attempt, kind, e,
            )

    # Chinese moderation: a refusal-style decision (no tool calls + very short
    # output that reads like a content-filter rejection) would end the run
    # prematurely. Retry once with a vague continuation nudge appended.
    if (
        runtime.config.chinese_moderation
        and response is not None
        and not response.tool_calls
        and looks_like_refusal(response.content)
    ):
        logger.warning("Supervisor decision looks like a content-filter refusal — retrying with a continuation nudge")
        print(f"\n{Colors.RED}[CONTENT MODERATION] Supervisor decision was refused — retrying with a continuation nudge.{Colors.RESET}")
        retry_messages = messages + [
            HumanMessage(
                content="Continue the research. Produce a complete, factual response in the target language."
            )
        ]
        try:
            response = await asyncio.wait_for(
                _supervisor_model().ainvoke(retry_messages),
                timeout=limits["supervisor_timeout_seconds"],
            )
        except Exception as e:
            kind = _classify_error(e)
            if kind != "fatal":
                logger.warning("Supervisor retry failed: %s", e)
                print(f"{Colors.RED}[CONTENT MODERATION] Supervisor retry after refusal failed: {e}{Colors.RESET}")

    if response is None:
        response = AIMessage(
            content="(supervisor decision failed twice; ending research gracefully)",
            tool_calls=[],
        )

    # Console logging for real-time visibility
    iteration = state.get("research_iterations", 0) + 1
    console_logger.log_supervisor_start(iteration, elapsed_minutes)
    if response.tool_calls:
        console_logger.log_supervisor_tool_calls(response.tool_calls)

    # Structured event: supervisor decided its next step (names only — never
    # reflection/reasoning content).
    _emit(
        _events.SUPERVISOR_ITERATION,
        phase="supervisor",
        agent="supervisor",
        iteration=iteration,
        tool_calls=[tc.get("name") for tc in (response.tool_calls or [])],
        elapsed_minutes=round(elapsed_minutes, 2),
        remaining_minutes=round(runtime.remaining_minutes(), 2),
    )

    # Log this conductor turn for observability
    log_conductor_turn(
        system_prompt=system_message,
        messages=supervisor_messages,
        response=response,
        elapsed_minutes=elapsed_minutes,
        iteration=iteration
    )

    return Command(
        goto="supervisor_tools",
        update={
            "supervisor_messages": [response],
            "research_iterations": state.get("research_iterations", 0) + 1
        }
    )

async def supervisor_tools(state: SupervisorState) -> Command[Literal["supervisor", "write_final_report", "__end__"]]:
    """Execute supervisor decisions - either conduct research or end the process.

    Handles:
    - Executing think_tool calls for strategic reflection
    - Launching parallel research agents for different topics
    - Aggregating research results
    - Determining when research is complete

    Args:
        state: Current supervisor state with messages and iteration count

    Returns:
        Command to continue supervision, end process, or handle errors
    """
    runtime = get_runtime()
    runtime.check_cancelled()
    limits = _supervisor_limits()
    cfg = runtime.config
    max_researcher_iterations = limits["max_researcher_iterations"]
    subagent_timeout = limits["subagent_timeout_seconds"]
    supervisor_messages = state.get("supervisor_messages", [])
    research_iterations = state.get("research_iterations", 0)
    target_language = state.get("target_language") or cfg.target_language_fallback
    most_recent_message = supervisor_messages[-1]
    _SUBAGENT_TOOL_NAMES = _subagent_tool_names()

    # Initialize variables for return pattern
    tool_messages = []
    all_curated_sources = []
    # Seed the global registry + dedup map from prior turns so sources are not
    # re-coded across multiple supervisor research rounds. `source_registry`
    # uses operator.add, so we must return ONLY the new entries this turn.
    global_registry = list(state.get("source_registry", []) or [])
    new_registry_entries = []
    global_code_by_identifier = {
        (entry.get("identifier") or entry.get("url") or "").strip(): entry.get("code") or ""
        for entry in global_registry
        if (entry.get("identifier") or entry.get("url") or "").strip()
        and (entry.get("code") or "").strip()
    }
    draft_report = state.get("draft_report", "")
    next_step = "supervisor"  # Default next step
    should_end = False
    aborted = False  # Circuit breaker flag
    abort_reason = ""  # Human-readable reason for an abort
    turn_failed = False  # Set when this iteration ended in a supervisor-level failure
    consecutive_failures = state.get("consecutive_failures", 0)

    # Check exit criteria first
    exceeded_iterations = research_iterations >= max_researcher_iterations
    no_tool_calls = not most_recent_message.tool_calls
    research_complete = any(
        tool_call["name"] == "ResearchComplete"
        for tool_call in most_recent_message.tool_calls
    )

    # Check if there are research calls in the same turn
    has_research_calls = any(
        tool_call["name"] in _SUBAGENT_TOOL_NAMES
        for tool_call in most_recent_message.tool_calls
    )

    # Match the original engine: elapsed research time routes to writing.
    start_time = state.get("start_time", 0.0)
    elapsed_minutes = 0.0
    if start_time > 0:
        elapsed_minutes = (time.time() - start_time) / 60.0

    exceeded_time = elapsed_minutes >= limits["strict_minutes"]

    # Denoise gate: while the research window is still open, a denoise verdict of
    # CONTINUE_RESEARCH overrides a premature ResearchComplete so the loop keeps
    # closing gaps (the active prompt's denoise protocol).
    denoise_forces_continue = (
        research_complete
        and elapsed_minutes < limits["max_minutes"]
        and _denoise_verdict_continue(supervisor_messages, most_recent_message.tool_calls)
    )

    # Exit if: exceeded limits OR no calls OR (ResearchComplete WITHOUT research
    # calls and WITHOUT a denoise verdict forcing more research)
    if exceeded_iterations or no_tool_calls or exceeded_time or (
        research_complete and not has_research_calls and not denoise_forces_continue
    ):
        if exceeded_time:
             print(f"\n{Colors.YELLOW}Research window reached ({limits['strict_minutes']} minutes); moving to final report writing.{Colors.RESET}")
        should_end = True
        next_step = "write_final_report"

    else:
        # This includes: (1) normal tool calls, (2) ResearchComplete WITH research
        # calls (salvage case), (3) ResearchComplete overridden by a denoise verdict.
        if research_complete and (has_research_calls or denoise_forces_continue):
            if denoise_forces_continue:
                print(f"\n{Colors.YELLOW}[Warning] ResearchComplete ignored: denoise verdict is CONTINUE_RESEARCH and the research window is still open - continuing research.{Colors.RESET}")
                rc_reason = (
                    "ResearchComplete ignored: your most recent denoise verdict was "
                    "VERDICT: CONTINUE_RESEARCH and the research window has not elapsed; "
                    "continuing research to close the remaining gaps."
                )
            else:
                # Incorrect tool call combination - ignore ResearchComplete and continue
                print(f"\n{Colors.YELLOW}[Warning] ResearchComplete called with research tools - ignoring ResearchComplete, continuing research.{Colors.RESET}")
                rc_reason = "ResearchComplete ignored: research tools were called in the same decision; continuing research."
            # The decision message still lists ResearchComplete among its tool_calls;
            # answer it with a ToolMessage so the history stays API-valid (an
            # orphaned tool_call_id makes the next model call 400).
            for tc in most_recent_message.tool_calls:
                if tc["name"] == "ResearchComplete":
                    tool_messages.append(
                        ToolMessage(
                            content=rc_reason,
                            name="ResearchComplete",
                            tool_call_id=tc["id"],
                        )
                    )

        # Execute ALL tool calls before deciding next step
        try:
            # Separate tool calls by type
            think_tool_calls = [
                tool_call for tool_call in most_recent_message.tool_calls
                if tool_call["name"] == "think_tool"
            ]

            # All sub-agent tool calls (ResearchWeb, ResearchGeneral,
            # ResearchReddit, etc.) handled uniformly.
            subagent_calls = [
                tool_call for tool_call in most_recent_message.tool_calls
                if tool_call["name"] in _SUBAGENT_TOOL_NAMES
            ]

            # Handle think_tool calls (unchanged — supervisor-level)
            for tool_call in think_tool_calls:
                observation = await think_tool.ainvoke(tool_call["args"])
                if runtime.config.enable_research_trace:
                    log_trace_supervisor_reaction(tool_call["args"].get("reflection", ""))
                runtime.observer.emit_trace(
                    "supervisor_thinking",
                    phase="supervisor",
                    agent="supervisor",
                    iteration=research_iterations,
                    purpose=tool_call["args"].get("purpose", "denoise"),
                    reflection=tool_call["args"].get("reflection", ""),
                )
                console_logger.log_supervisor_thinking(
                    tool_call["args"].get("purpose", "denoise"),
                    tool_call["args"].get("reflection", ""),
                )
                tool_messages.append(
                    ToolMessage(
                        content=observation,
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"]
                    )
                )

            # ── Unified sub-agent spawn loop ─────────────────────────────
            if subagent_calls:
                runtime.check_cancelled()
                agent_ids = []
                coros = []
                for tc in subagent_calls:
                    topic = tc["args"].get("research_topic", "")
                    discovery = tc["args"].get("discovery", False)
                    if discovery:
                        agent_ids.append(console_logger.log_discovery_start(topic))
                    else:
                        agent_ids.append(console_logger.log_sub_agent_start(topic))
                    if runtime.config.enable_research_trace:
                        log_trace_delegation(topic)

                    runtime.observer.emit_trace(
                        "delegation",
                        phase="supervisor",
                        agent=tc["name"],
                        platform=tc["name"],
                        iteration=research_iterations,
                        research_topic=topic,
                        discovery=bool(discovery),
                        mode="discovery" if discovery else "research",
                    )

                    # Structured events: delegation + subagent start.
                    _emit(
                        _events.DELEGATION_STARTED,
                        phase="supervisor",
                        agent=tc["name"],
                        platform=tc["name"],
                        iteration=research_iterations,
                        research_topic=topic,
                        discovery=bool(discovery),
                    )
                    _emit(
                        _events.SUBAGENT_STARTED,
                        phase="subagent",
                        agent=tc["name"],
                        platform=tc["name"],
                        iteration=research_iterations,
                        agent_id=agent_ids[-1],
                        research_topic=topic,
                        discovery=bool(discovery),
                    )

                    agent_graph = AGENT_REGISTRY[tc["name"]]
                    agent_input = {
                        "researcher_messages": [
                            HumanMessage(content=topic)
                        ],
                        "research_topic": topic,
                        "target_language": target_language,
                        "discovery": discovery,
                        "console_agent_id": agent_ids[-1],
                        "chinese_supervisor_international_subagent": runtime.config.chinese_supervisor_international_subagent,
                    }
                    # Supervisor may override the sub-agent's total read budget
                    # (deeper or narrower research). None = use the config default.
                    max_total_reads = tc["args"].get("max_total_reads")
                    if max_total_reads is not None:
                        agent_input["max_total_reads"] = max_total_reads
                    coros.append(
                        _run_subagent_with_progress(
                            agent_graph, agent_input,
                            timeout=subagent_timeout + 30,
                            agent_id=agent_ids[-1], tool_name=tc["name"],
                            iteration=research_iterations,
                        )
                    )

                all_results = await asyncio.gather(*coros, return_exceptions=True)

                for i, (tc, result) in enumerate(zip(subagent_calls, all_results)):
                    topic = tc["args"].get("research_topic", "")
                    is_discovery = bool(tc["args"].get("discovery", False))
                    if isinstance(result, BaseException):
                        logger.warning(f"Subagent failed for: {topic} - {result}")
                        compressed = f"Subagent timed out: {topic}"
                        sub_registry = []
                    else:
                        compressed = result.get("compressed_research", "Error")
                        sub_registry = result.get("source_registry", []) or []

                    # Remap this sub-report's local citation codes ([S2#3]) to
                    # globally-unique codes ([A{agent_id}-S2#3]) and merge its
                    # registry entries into the global registry, deduping by
                    # identifier/URL so two agents reading the same source map
                    # to the same global code.
                    code_remap = {}
                    if sub_registry and isinstance(result, dict):
                        agent_id = agent_ids[i] if i < len(agent_ids) else 0
                        for entry in sub_registry:
                            local_code = (entry.get("code") or "").strip()
                            identifier = (entry.get("identifier") or entry.get("url") or "").strip()
                            if not identifier or not local_code:
                                continue
                            if identifier not in global_code_by_identifier:
                                global_code = f"A{agent_id}-{local_code}"
                                entry = {**entry, "code": global_code, "agent": agent_id}
                                global_registry.append(entry)
                                new_registry_entries.append(entry)
                                global_code_by_identifier[identifier] = global_code
                            code_remap[local_code] = global_code_by_identifier[identifier]
                    if code_remap:
                        compressed = remap_codes(compressed, code_remap)

                    tool_messages.append(
                        ToolMessage(
                            content=compressed,
                            name=tc["name"],
                            tool_call_id=tc["id"],
                        )
                    )
                    search_queries = (
                        result.get("search_queries", [])
                        if not isinstance(result, BaseException) else []
                    )
                    log_sub_agent(
                        research_topic=topic,
                        system_prompt="(full prompt emitted as a 'subagent_prompt' trace record)",
                        compressed_research=compressed,
                        agent_type=tc["name"],
                        search_queries=search_queries,
                        discovery=is_discovery,
                        output_mode=(result.get("output_mode")
                                     if not isinstance(result, BaseException) else None),
                    )

                    if not isinstance(result, BaseException):
                        sub_curated = result.get("curated_sources", [])
                        if sub_curated:
                            all_curated_sources.extend(sub_curated)

            runtime.check_cancelled()

            # ── Circuit breaker ──────────────────────────────────────────
            total_agents = len(subagent_calls)
            failed_agents = sum(
                1 for r in all_results if isinstance(r, BaseException)
            ) if subagent_calls else 0
            if total_agents > 0 and failed_agents == total_agents:
                # Check if failures are quota/rate-limit errors (not just timeouts)
                sample_error = str(all_results[0])
                is_quota_error = "RESOURCE_EXHAUSTED" in sample_error or "429" in sample_error
                if is_quota_error:
                    print(f"\n{Colors.RED}[CIRCUIT BREAKER] All {total_agents} subagents failed with API quota errors. "
                          f"Aborting research — check your API billing/quota.{Colors.RESET}")
                    logger.error(f"Circuit breaker triggered: all {total_agents} subagents hit quota limits. Aborting.")
                    should_end = True
                    aborted = True
                    abort_reason = f"All {total_agents} subagents failed with API quota errors: {sample_error}"
                    next_step = END
                else:
                    print(f"\n{Colors.YELLOW}[WARNING] All {total_agents} subagents failed. "
                          f"Returning errors to supervisor for retry decision.{Colors.RESET}")

            # Safety net: guarantee every tool_call in the decision message has a
            # matching ToolMessage before returning. Normally a no-op (all calls
            # are answered above); guards against any unexecuted/unknown tool call
            # leaving an orphaned tool_call_id that would 400 the next model call.
            _complete_unanswered_tool_calls(
                most_recent_message,
                tool_messages,
                "Tool execution did not produce a result this turn (call skipped).",
            )


        except Exception as e:
            if isinstance(e, RunCancelledError):
                raise
            kind = _classify_error(e)
            if kind == "fatal":
                logger.error("Fatal error during research — aborting: %s", e)
                aborted = True
                abort_reason = f"Fatal error during research: {e}"
                next_step = END
                should_end = True
            else:
                # Supervisor-level failure. Retry once per iteration: the first
                # failure re-does the iteration; two consecutive failures stop.
                consecutive_failures += 1
                if consecutive_failures >= 2:
                    if _should_salvage(state, all_curated_sources, new_registry_entries, elapsed_minutes):
                        logger.warning(
                            "Research stalled after %d consecutive failures — salvaging final report: %s",
                            consecutive_failures, e,
                        )
                        should_end = True
                        next_step = "write_final_report"
                    else:
                        logger.error(
                            "Research stalled after %d consecutive failures — aborting: %s",
                            consecutive_failures, e,
                        )
                        aborted = True
                        abort_reason = (
                            f"Research stalled after {consecutive_failures} consecutive failures "
                            f"with insufficient findings: {e}"
                        )
                        next_step = END
                        should_end = True
                else:
                    # First failure this streak — re-do the iteration. Ensure
                    # every decision tool call is answered so the loop stays
                    # API-valid, then hand back to the supervisor.
                    logger.warning(
                        "Supervisor iteration failed (1 consecutive) — re-doing: %s", e,
                    )
                    _complete_unanswered_tool_calls(most_recent_message, tool_messages, f"Tool execution failed: {e}")
                    next_step = "supervisor"
                    turn_failed = True

    # Fold the elapsed-time progress note into the LAST tool message so the next
    # supervisor call sees it as the tail of an append-only conversation
    # (cache-friendly) instead of a fresh transient user message. The supervisor
    # node no longer injects a per-turn HumanMessage for this.
    runtime.check_cancelled()
    if tool_messages:
        last = tool_messages[-1]
        tool_messages[-1] = ToolMessage(
            content=last.content + f"\n\nCURRENT PROGRESS: You have been researching for {elapsed_minutes:.1f} minutes.",
            tool_call_id=last.tool_call_id,
            name=last.name,
        )

    # Single return point with appropriate state updates
    if should_end:
        console_logger.log_research_complete()
        update = {
            "notes": get_notes_from_tool_calls(supervisor_messages),
            "research_brief": state.get("research_brief", ""),
            "draft_report": draft_report,
            "curated_sources": all_curated_sources,
            "source_registry": new_registry_entries,
        }
        if aborted:
            update["aborted"] = True
            update["abort_reason"] = abort_reason
        else:
            update["consecutive_failures"] = 0
        return Command(
            goto=next_step,
            update=update
        )
    else:
        return Command(
            goto=next_step,
            update={
                "supervisor_messages": tool_messages,
                "curated_sources": all_curated_sources,
                "source_registry": new_registry_entries,
                "consecutive_failures": 0 if not turn_failed else consecutive_failures,
            }
        )


# ===== GRAPH CONSTRUCTION =====

async def write_final_report(state: SupervisorState) -> dict:
    """Write the final report inside the supervisor subgraph (cache-friendly).

    Reuses the supervisor's byte-identical system prompt and the exact
    accumulated conversation, so the provider's prompt cache serves the entire
    research-history prefix; only the injected write instruction (registry +
    curated full text + writing rules) and the generation are paid fresh.
    Citations are finalized deterministically in code (codes -> [N]).
    """
    if state.get("aborted"):
        print("\n--- [NODE: write_final_report] ---")
        print("[Final Report] SKIPPED — research was aborted by circuit breaker.")
        return {"final_report": ""}

    runtime = get_runtime()
    runtime.check_cancelled()
    cfg = runtime.config
    print("\n--- [NODE: write_final_report] ---")
    curated = state.get("curated_sources", []) or []
    registry = state.get("source_registry", []) or []
    final_registry = build_final_registry(curated, registry)
    _emit(_events.REPORT_STARTED, phase="report", agent="final_report_writer",
          source_count=len(final_registry), report_kind="final")

    registry_lines = []
    for entry in final_registry:
        code = entry.get("code") or ""
        title = entry.get("title") or "Untitled"
        url = entry.get("url") or ""
        ref = entry.get("ref") or ""
        line = f"[{code}] {title} ({url})"
        if ref:
            line += f"\n    Relevance: {ref}"
        registry_lines.append(line)
    registry_block = "\n".join(registry_lines) if registry_lines else "(no sources available)"

    full_text_parts = []
    for entry in final_registry:
        full_text = (entry.get("full_text") or "").strip()
        if not full_text:
            continue
        full_text_parts.append(
            f"[{entry.get('code')}] {entry.get('title') or 'Untitled'} ({entry.get('url') or ''})\n{full_text}"
        )
    curated_full_text = (
        "\n\n---\n\n".join(full_text_parts)
        if full_text_parts
        else "(No additional source text — the sources are already covered in the research findings above.)"
    )

    system_message = _build_supervisor_system_message(state)
    final_report_write_prompt = runtime.prompts.get("final_report_write_prompt", "")
    write_instruction = final_report_write_prompt.format(
        source_registry_block=registry_block,
        curated_full_text=curated_full_text,
        target_language=state.get("target_language") or cfg.target_language_fallback,
        date=get_today_str(),
    )

    # The supervisor's trailing decision message (e.g. ResearchComplete, or a
    # tool call that never got a tool result) is an AIMessage
    # with tool_calls. The API rejects an assistant tool_calls message that
    # isn't followed by matching tool messages, so drop that trailing decision
    # before assembling the write conversation.
    history = list(state.get("supervisor_messages", []))
    if history and isinstance(history[-1], AIMessage) and getattr(history[-1], "tool_calls", None):
        history = history[:-1]

    messages = [SystemMessage(content=system_message)] + history
    messages.append(HumanMessage(content=write_instruction))

    response = await _final_report_writer_model().ainvoke(messages)
    raw_report = extract_text_from_response(response.content)

    # Chinese moderation: a content-filter refusal would otherwise be written
    # verbatim as the report. Retry once with a vague continuation nudge.
    if cfg.chinese_moderation and looks_like_refusal(raw_report):
        logger.warning("Final report looks like a content-filter refusal — retrying with a continuation nudge")
        print(f"\n{Colors.RED}[CONTENT MODERATION] Final report was refused — retrying with a continuation nudge.{Colors.RESET}")
        retry_messages = messages + [
            HumanMessage(
                content="Continue the research. Produce a complete, factual response in the target language."
            )
        ]
        try:
            retry_response = await _final_report_writer_model().ainvoke(retry_messages)
            retry_raw = extract_text_from_response(retry_response.content)
            if retry_raw.strip() and not looks_like_refusal(retry_raw):
                raw_report = retry_raw
        except Exception as e:
            logger.warning("Final report retry failed: %s", e)
            print(f"{Colors.RED}[CONTENT MODERATION] Final report retry after refusal failed: {e}{Colors.RESET}")

    final_report = finalize_citations(raw_report, final_registry, renumber=True)

    _emit(_events.CITATIONS_VALIDATED, phase="report", agent="final_report_writer",
          report_kind="final", chars=len(final_report), sources=len(final_registry))

    print(f"[Final Report] Written: {len(final_report)} chars (citations finalized in code)")
    return {"final_report": final_report}


# Build supervisor graph
supervisor_builder = StateGraph(SupervisorState)
supervisor_builder.add_node("supervisor", supervisor)
supervisor_builder.add_node("supervisor_tools", supervisor_tools)
supervisor_builder.add_node("write_final_report", write_final_report)
supervisor_builder.add_edge(START, "supervisor")
supervisor_agent = supervisor_builder.compile()

