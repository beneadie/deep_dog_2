"""Platform agent engine — one LangGraph for every platform research agent.

This is the shared sub-agent loop for all platforms: reddit, pubmed,
sec_edgar, arxiv, substack, web, and the all-tools general agent. The
PLATFORMS dict maps each agent_type to its tool list, prompt guidance, and
save-tool identifier mapping — the graph itself is platform-agnostic.

Key design points:
- Each invocation gets its OWN ResearcherState, so curation dicts
  (articles_read / saved_articles / findings_log) never cross-contaminate
  between concurrent runs.
- tool_node intercepts read_*, save_*, list_saved, and log_finding calls to
  do curation bookkeeping into the per-invocation state. Tools themselves
  are stateless.
- compress_research synthesizes from the curated saved_articles dict first
  (never from raw noisy conversation), per the design philosophy.
- Model selection uses the sub-agent model-name fallback chain
  (get_subagent_model → SUBAGENT_MODEL_FALLBACK_CHAIN), overridable per
  invocation via the `model_chain` state key.
"""

import asyncio
import logging
import logging
import time
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from deep_research.config import (
    DEFAULT_MAX_TOTAL_READS,
    GENERAL_AGENT_PLATFORMS, WEB_SEARCH_ENGINE,
    contains_sensitive,
)
from deep_research.time_utils import get_today_str
from deep_research.console_logger import Colors
from deep_research.observability import log_source
from deep_research.runtime import get_runtime
from deep_research.secrets import get_secret
from deep_research import events as _events
from deep_research.platform_prompts import (
    BASE_AGENT_PROMPT,
    DISCOVERY_AGENT_PROMPT,
    REPORT_MODE_INSTRUCTIONS,
    FINISH_INSTRUCTIONS,
    REDDIT_TOOL_GUIDANCE, PUBMED_TOOL_GUIDANCE, SEC_EDGAR_TOOL_GUIDANCE,
    ARXIV_TOOL_GUIDANCE, SUBSTACK_TOOL_GUIDANCE,
    WEB_TOOL_GUIDANCE, build_general_guidance,
    CURATION_TOOL_GUIDANCE,
    CURATION_GATHERING, FULL_CONTEXT_GATHERING_SOURCES, FULL_CONTEXT_GATHERING_REPORT,
    DISCOVERY_FULL_CONTEXT_GATHERING_REPORT,
    FULL_CONTEXT_SOURCES_TOOLS, FULL_CONTEXT_REPORT_TOOLS,
    CURATION_STRATEGY, DISCOVERY_STRATEGY, FULL_CONTEXT_STRATEGY,
    DISCOVERY_FULL_CONTEXT_STRATEGY,
)
from deep_research.state_research import ResearcherOutputState, ResearcherState
from deep_research.agents.shared.tools import (
    fetch_urls,
    _fetch_url_content,
    list_saved, log_finding,
    set_target_language, finish_research, batch_save_selected,
)
from deep_research.agents.reddit.tools import (
    search_term_in_subreddit, get_subreddit_posts, get_reddit_posts,
    search_subreddits, check_user_profile,
    _fetch_reddit_post,
)
from deep_research.agents.pubmed.tools import (
    search_pubmed, read_pubmed_articles,
    _fetch_pubmed_article,
)
from deep_research.agents.sec_edgar.tools import (
    lookup_company, get_company_profile, search_sec_filings, get_latest_filing,
    read_filings, get_financials, compare_companies, get_insider_transactions,
    _fetch_filing,
)
from deep_research.agents.arxiv.tools import (
    search_arxiv, read_arxiv_articles,
    _fetch_arxiv_article,
)
from deep_research.agents.substack.tools import (
    search_substack, read_substack_articles, check_author_profile,
    _fetch_substack_article,
)
from deep_research.agents.web.tools import (
    tavily_search, exa_deep_search,
)

logger = logging.getLogger(__name__)

# ── Tool lists (per platform) ──────────────────────────────────────────

_base_tools = [fetch_urls, set_target_language, finish_research, batch_save_selected]

# Tool names that count as "searches" for depth-cap enforcement
SEARCH_TOOL_NAMES = {
    "search_pubmed", "search_arxiv", "search_term_in_subreddit",
    "get_subreddit_posts", "search_subreddits", "search_sec_filings",
    "search_substack",
    "tavily_search", "exa_deep_search",
}

# Search tools whose results get an [S#] handle for index-based reads.
SEARCH_INDEXABLE_TOOLS = {
    "search_pubmed", "search_arxiv", "search_term_in_subreddit",
    "get_subreddit_posts", "search_sec_filings", "search_substack",
    "tavily_search", "exa_deep_search",
}

_reddit_tools = _base_tools + [
    search_term_in_subreddit, get_subreddit_posts, get_reddit_posts,
    search_subreddits, check_user_profile,
    list_saved, log_finding,
]

_pubmed_tools = _base_tools + [
    search_pubmed, read_pubmed_articles, list_saved, log_finding,
]

_sec_edgar_tools = _base_tools + [
    lookup_company, get_company_profile, search_sec_filings, get_latest_filing,
    read_filings, get_financials, compare_companies, get_insider_transactions,
    list_saved, log_finding,
]

_arxiv_tools = _base_tools + [
    search_arxiv, read_arxiv_articles, list_saved, log_finding,
]

_substack_tools = _base_tools + [
    search_substack, read_substack_articles, check_author_profile,
    list_saved, log_finding,
]

_web_search_tools = []
_warned = []
if WEB_SEARCH_ENGINE in ("tavily", "both"):
    if get_secret("TAVILY_API_KEY"):
        _web_search_tools.append(tavily_search)
    else:
        _warned.append("TAVILY_API_KEY")
if WEB_SEARCH_ENGINE in ("exa", "both"):
    if get_secret("EXA_API_KEY"):
        _web_search_tools.append(exa_deep_search)
    else:
        _warned.append("EXA_API_KEY")
if _warned:
    logging.getLogger(__name__).warning(
        "WEB_SEARCH_ENGINE=%r but %s not set in .env — those search tools are "
        "not available to the web/general agents.",
        WEB_SEARCH_ENGINE, " and ".join(_warned),
    )
elif WEB_SEARCH_ENGINE not in ("tavily", "exa", "both"):
    logging.getLogger(__name__).warning(
        "Invalid WEB_SEARCH_ENGINE=%r (expected 'tavily', 'exa', or 'both'). "
        "No web search tools exposed.", WEB_SEARCH_ENGINE,
    )

_web_tools = _base_tools + _web_search_tools + [
    list_saved, log_finding,
]

# Platform tool sets, keyed by PLATFORMS key. Used to build the general agent's
# tool set from GENERAL_AGENT_PLATFORMS (empty = all platforms).
_PLATFORM_TOOL_SETS = {
    "reddit": _reddit_tools,
    "pubmed": _pubmed_tools,
    "sec_edgar": _sec_edgar_tools,
    "arxiv": _arxiv_tools,
    "substack": _substack_tools,
    "web": _web_tools,
}


def _build_general_tools(keys: list[str]) -> list:
    """Build the general-agent tool set from the given platform keys.

    Unknown keys are skipped with a warning. An empty key list returns every
    platform's tools (the historical behavior).
    """
    _logger = logging.getLogger(__name__)
    chosen = []
    for key in keys or list(_PLATFORM_TOOL_SETS):
        if key not in _PLATFORM_TOOL_SETS:
            _logger.warning("GENERAL_AGENT_PLATFORMS: unknown platform %r skipped", key)
            continue
        chosen.extend(_PLATFORM_TOOL_SETS[key])
    return list({t.name: t for t in chosen}.values())


# General-purpose agent: every tool across the configured platforms
# (deduplicated by name). GENERAL_AGENT_PLATFORMS filters which platforms.
_general_keys = GENERAL_AGENT_PLATFORMS or list(_PLATFORM_TOOL_SETS)
_general_tools = _build_general_tools(_general_keys)


# ── Per-run platform tool resolution ─────────────────────────────────────
# The module-level tool sets above are built from environment defaults at
# import (kept for legacy/standalone use). Under an active runtime, tool sets
# are resolved from the run's config + credentials so two concurrent API runs
# with different search providers/keys never share tools.


def _runtime_search_tools():
    """Web search tools enabled for the active runtime's config + credentials."""
    rt = get_runtime()
    engine = (rt.config.web_search_engine or "").lower()
    tools = []
    if engine in ("tavily", "both") and rt.credentials.has("TAVILY_API_KEY"):
        tools.append(tavily_search)
    if engine in ("exa", "both") and rt.credentials.has("EXA_API_KEY"):
        tools.append(exa_deep_search)
    return tools


def _dedupe_tools(tools) -> list:
    return list({t.name: t for t in tools}.values())


def _runtime_platform_tools(rt, agent_type: str) -> list:
    """Resolve the platform's tool list for the active runtime."""
    _web_static = [t for t in _PLATFORM_TOOL_SETS.get("web", [])
                   if t.name not in {"tavily_search", "exa_deep_search"}]
    if agent_type == "web":
        return _dedupe_tools(list(_web_static) + _runtime_search_tools())
    if agent_type == "general":
        keys = rt.config.general_agent_platforms or list(_PLATFORM_TOOL_SETS)
        chosen = []
        for key in keys:
            if key not in _PLATFORM_TOOL_SETS:
                continue
            if key == "web":
                chosen.extend(_web_static)
            else:
                chosen.extend(_PLATFORM_TOOL_SETS[key])
        chosen = [t for t in chosen if t.name not in {"tavily_search", "exa_deep_search"}]
        chosen.extend(_runtime_search_tools())
        return _dedupe_tools(chosen)
    return list(PLATFORMS.get(agent_type, PLATFORMS["reddit"]).get("tools", []))


def _emit_event(type_: str, *, phase: str | None = None, agent: str | None = None,
                platform: str | None = None, iteration: int | None = None, **payload) -> None:
    try:
        get_runtime().observer.emit(type_, phase=phase, agent=agent, platform=platform,
                                    iteration=iteration, **payload)
    except Exception:  # pragma: no cover - events must never break research
        pass


# ── Platform registry ───────────────────────────────────────────────────

PLATFORMS: dict[str, dict] = {
    "reddit": {
        "tools": _reddit_tools,
        "guidance": REDDIT_TOOL_GUIDANCE,
        "label": "reddit",
    },
    "pubmed": {
        "tools": _pubmed_tools,
        "guidance": PUBMED_TOOL_GUIDANCE,
        "label": "pubmed",
    },
    "sec_edgar": {
        "tools": _sec_edgar_tools,
        "guidance": SEC_EDGAR_TOOL_GUIDANCE,
        "label": "sec_edgar",
    },
    "arxiv": {
        "tools": _arxiv_tools,
        "guidance": ARXIV_TOOL_GUIDANCE,
        "label": "arxiv",
    },
    "substack": {
        "tools": _substack_tools,
        "guidance": SUBSTACK_TOOL_GUIDANCE,
        "label": "substack",
    },
    "web": {
        "tools": _web_tools,
        "guidance": WEB_TOOL_GUIDANCE,
        "label": "web",
    },
    "general": {
        "tools": _general_tools,
        "guidance": build_general_guidance(_general_keys),
        "label": "general",
    },
}

# Batch read tools → spec used by tool_node to expand items, resolve ids,
# fetch each item, and record articles_read. `fetch_urls` takes a bare URL
# list instead of {index, ref} items (handled specially in tool_node).
BATCH_READ_MAP: dict[str, dict] = {
    "get_reddit_posts": {
        "id_key": "url", "fetch_fn": _fetch_reddit_post,
        "extra": ["include_comments"],
    },
    "read_pubmed_articles": {
        "id_key": "pmid", "fetch_fn": _fetch_pubmed_article, "extra": [],
    },
    "read_filings": {
        "id_key": "filing_url", "fetch_fn": _fetch_filing, "extra": ["sections"],
    },
    "read_arxiv_articles": {
        "id_key": "arxiv_id", "fetch_fn": _fetch_arxiv_article, "extra": [],
    },
    "read_substack_articles": {
        "id_key": "url", "fetch_fn": _fetch_substack_article, "extra": [],
    },
    "fetch_urls": {
        "id_key": "url", "fetch_fn": _fetch_url_content, "extra": [],
        "cap": 3,  # secondary link-chasing tool — 3 URLs per iteration
    },
}

# Where log_finding stores its key/value in tool args
LOG_FINDING_TOOL = "log_finding"


def get_platform(agent_type: str) -> dict:
    return PLATFORMS.get(agent_type, PLATFORMS["reddit"])


def list_platforms() -> list[str]:
    return list(PLATFORMS.keys())


# ── Helpers (module-level for testability) ─────────────────────────────

def _find_in_search_results(search_results: dict, identifier: str) -> tuple[str, int, str] | None:
    """Find (ref, 1-based index, title) for an identifier in search_results."""
    for ref, entry in search_results.items():
        for i, item in enumerate(entry.get("items", []), 1):
            if str(item.get("id", "")).strip() == identifier:
                return (ref, i, str(item.get("title", "")))
    return None


def _lookup_search_full_text(search_results: dict, identifier: str) -> tuple[str, str] | None:
    """Return (title, full_text) already captured by a web search, if any.

    Search tools retain the full text returned by the provider (Exa `text` /
    Tavily `raw_content`) on the registered [S#] items. fetch_urls serves this
    cached text back instead of re-fetching the URL via Tavily, so pages the
    agent already saw in search results are not fetched twice.
    """
    for entry in (search_results or {}).values():
        for item in entry.get("items", []):
            if str(item.get("id", "")).strip() == identifier:
                full = (item.get("full_text") or "").strip()
                if full:
                    return (str(item.get("title", "")), full)
    return None


def _find_ref_tag(search_results: dict, identifier: str) -> str | None:
    """Find the [S# #i] tag string for an identifier, or None."""
    found = _find_in_search_results(search_results, identifier)
    if found:
        return f"{found[0]} #{found[1]}"
    return None


def _find_title(search_results: dict, identifier: str) -> str:
    """Look up the title for an identifier in search_results."""
    found = _find_in_search_results(search_results, identifier)
    return found[2] if found else ""


def _build_source_registry(search_results: dict, articles_read: dict) -> list[dict]:
    """Ordered registry of every source read/seen, for citation resolution.

    Order is deterministic: search_results in insertion order (S1#1, S1#2, ...),
    then any articles_read identifier not already present. Each entry is
    {identifier, url, title, ref, code} — title + URL only, no content. `code`
    is the stable citation token the writer cites inline (S1#2 for search hits,
    R1/R2/... for read-only leftovers); the supervisor prefixes it with the
    agent id to make it globally unique.
    """
    registry: list[dict] = []
    seen: set[str] = set()
    for ref, entry in (search_results or {}).items():
        for i, item in enumerate(entry.get("items", []), 1):
            identifier = str(item.get("id", "")).strip()
            if not identifier or identifier in seen:
                continue
            seen.add(identifier)
            registry.append({
                "identifier": identifier,
                "url": str(item.get("url") or identifier).strip(),
                "title": str(item.get("title", "")).strip(),
                "ref": f"{ref} #{i}",
                "code": f"{ref}#{i}",
            })
    leftover_idx = 0
    for identifier in articles_read or {}:
        identifier = str(identifier).strip()
        if not identifier or identifier in seen:
            continue
        seen.add(identifier)
        leftover_idx += 1
        registry.append({
            "identifier": identifier,
            "url": identifier,
            "title": "",
            "ref": "",
            "code": f"R{leftover_idx}",
        })
    return registry


def _build_read_refs(search_results: dict, articles_read: dict, limit: int = 25) -> list[str]:
    """Reverse-map read items to their [S# #i] tags + titles.

    Works even for URL-based reads (where the agent typed a URL instead of
    using index+ref), since we match by the canonical id stored in
    search_results items.
    """
    refs = []
    for identifier in articles_read:
        found = _find_in_search_results(search_results, identifier)
        if found:
            ref, idx, title = found
            label = title[:80] if title else identifier[:80]
            refs.append(f"[{ref} #{idx}] {label}")
        if len(refs) >= limit:
            break
    return refs


# ── Output mode → gather strategy + deliverable ────────────────────────
# One enum replaces the old (output_type, report_mode) pair. `_mode_key`
# maps each mode back to the internal gather-strategy key that the prompts,
# tool restriction, and finalize steps key off.

OUTPUT_MODE_CONFIG: dict[str, dict] = {
    "sources":        {"gather": "curation",      "deliverable": "sources"},
    "report":         {"gather": "curation",      "deliverable": "report"},
    "sources_inline": {"gather": "full_context",  "deliverable": "sources"},
    "report_inline":  {"gather": "full_context",  "deliverable": "report"},
}


def _mode_key(output_mode: str) -> str:
    """Map an output_mode to the internal gather-strategy key.

    "curation" is the save-as-you-go strategy used by both sources and report
    modes; the full_context variants produce their deliverable in the agent's
    own final turn.
    """
    cfg = OUTPUT_MODE_CONFIG.get(output_mode, OUTPUT_MODE_CONFIG["sources"])
    if cfg["gather"] == "full_context":
        return "full_context_sources" if cfg["deliverable"] == "sources" else "full_context_report"
    return "curation"


def _delivers_report(output_mode: str) -> bool:
    return output_mode in ("report", "report_inline")


def _tools_for_turn(plat_tools: list, mode_key: str, is_final: bool) -> list:
    """Return the tool subset for this turn.

    On the final iteration of full_context mode, restrict to only the tools
    needed for the deliverable so the agent MUST select or write, not read more.
    """
    if not is_final or mode_key == "curation":
        return plat_tools
    if mode_key == "full_context_sources":
        return [t for t in plat_tools
                if t.name in {"batch_save_selected", "finish_research", "set_target_language"}]
    if mode_key == "full_context_report":
        # No tools at all on the final turn — the agent MUST write the report
        # as a plain text message (there's nothing else it can call).
        return []
    return plat_tools


# Tool calls that are session housekeeping, not research work. Rounds whose
# tool calls are ALL housekeeping don't consume the research iteration budget.
_HOUSEKEEPING_TOOLS = {"set_target_language", "finish_research"}


def _iteration_count(messages: list) -> int:
    """Count research iterations = LLM rounds that made at least one real
    (non-housekeeping) tool call.

    A round dedicated to set_target_language (or an all-housekeeping round) is
    overhead and must not eat into the max_iterations budget.
    """
    count = 0
    for m in messages:
        calls = getattr(m, "tool_calls", None) or []
        names = []
        for tc in calls:
            if isinstance(tc, dict):
                names.append(tc.get("name", ""))
            else:
                names.append(getattr(tc, "name", ""))
        if calls and not all(n in _HOUSEKEEPING_TOOLS for n in names):
            count += 1
    return count


def _compute_turn_plan(state: dict) -> dict:
    """Compute this turn's budget counts, caps, nudges, and status text.

    Shared by llm_call (banner + tool restriction) and tool_node (which folds
    the status text into the last ToolMessage it returns). Because the status
    is persisted as part of the tool observation instead of being injected as a
    fresh trailing user message, the conversation stays append-only and the
    provider's prompt cache can reuse the history prefix across turns.
    """
    output_mode = state.get("output_mode", "sources")
    mode_key = _mode_key(output_mode)
    current_iteration = _iteration_count(state.get("researcher_messages", []))
    max_iter = state.get("max_iterations", 10)
    max_reads = state.get("max_reads", 8)
    max_total_reads = state.get("max_total_reads", DEFAULT_MAX_TOTAL_READS)
    max_saves = state.get("max_saves", 15)
    max_searches = state.get("max_searches", 8)
    n_saved = len(state.get("saved_articles", {}))
    n_read = len(state.get("articles_read", {}))
    n_searches = state.get("search_count", 0)
    iters_remaining = max_iter - current_iteration
    is_final = iters_remaining <= 1

    # Forced save round (curation): budget exhausted but read items never saved.
    # Mirrors the grant condition in route_after_tools.
    forced_save = (
        mode_key == "curation"
        and current_iteration >= max_iter
        and not state.get("final_save_round")
        and n_saved < n_read
    )

    caps = []
    nudges = []
    if n_searches >= max_searches:
        caps.append("searches MAXED")
    if n_saved >= max_saves:
        caps.append("saves MAXED")
    if n_read >= max_total_reads:
        caps.append("reads MAXED")
    if mode_key == "curation":
        if n_saved == 0 and n_read > 0 and iters_remaining == 1:
            nudges.append(f"FINAL TURN — you read {n_read} items but saved 0. "
                          "Read AND call batch_save_selected in this SAME turn "
                          "(reads execute before saves).")
    elif mode_key == "full_context_sources":
        if n_saved == 0 and iters_remaining == 1:
            nudges.append("FINAL ITERATION — STOP reading. Call batch_save_selected NOW.")
        elif iters_remaining <= 2 and n_saved == 0:
            nudges.append("next turn is your last — prepare to call batch_save_selected")
    elif mode_key == "full_context_report":
        if iters_remaining == 1:
            nudges.append("FINAL ITERATION — STOP reading. Write your COMPLETE research report "
                          "as your final message NOW: answer the research question with the "
                          "best material you read, citing each claim inline with its exact "
                          "code from the SOURCE REGISTRY.")
        elif iters_remaining <= 2:
            nudges.append("next turn is your last — prepare to write your report as your final message")
    if forced_save:
        nudges.append(f"FINAL SAVE ROUND — you read {n_read} items but saved only "
                      f"{n_saved}. Call batch_save_selected NOW to save the useful "
                      "ones (your reads are already complete).")

    read_refs = _build_read_refs(
        state.get("search_results", {}), state.get("articles_read", {}))

    target_language = state.get("target_language", "English")
    lang_set = bool(target_language and target_language.strip().lower() != "auto")

    status_lines = [
        "SESSION STATUS:",
        f"  Target language: {target_language}",
        f"  Iteration: {current_iteration + 1}/{max_iter} ({iters_remaining} remaining)",
        f"  Searches: {n_searches}/{max_searches} | Reads: {n_read}/{max_total_reads} | "
        f"Read budget/iter: {max_reads} | Saved: {n_saved}/{max_saves}",
    ]
    if not lang_set:
        status_lines.append("  (Language not set yet — call set_target_language in your first tool round, alongside your searches)")
    if caps:
        status_lines.append("  Caps: " + "; ".join(caps))
    if nudges:
        status_lines.append("  ⚠ " + "; ".join(nudges))
    if mode_key == "full_context_report" and is_final:
        status_lines.append("  AVAILABLE TOOLS: none — your final message IS the report.")
    if read_refs:
        status_lines.append("  READ ITEMS (use these refs for batch_save_selected):")
        for r in read_refs:
            status_lines.append(f"    {r}")
    if mode_key == "full_context_report":
        registry = _build_source_registry(
            state.get("search_results", {}), state.get("articles_read", {}))
        if registry:
            status_lines.append("  SOURCE REGISTRY (cite each source by its exact [code] inline, do NOT reproduce URLs):")
            for entry in registry:
                title = entry["title"] or entry["identifier"]
                status_lines.append(f"    [{entry['code']}] {title} ({entry['url']})")

    return {
        "mode_key": mode_key,
        "current_iteration": current_iteration,
        "max_iter": max_iter,
        "iters_remaining": iters_remaining,
        "is_final": is_final,
        "n_saved": n_saved,
        "n_read": n_read,
        "n_searches": n_searches,
        "max_reads": max_reads,
        "max_total_reads": max_total_reads,
        "max_saves": max_saves,
        "max_searches": max_searches,
        "forced_save": forced_save,
        "caps": caps,
        "nudges": nudges,
        "status_text": "\n".join(status_lines),
    }


# ── Graph nodes ────────────────────────────────────────────────────────

async def llm_call(state: ResearcherState):
    """Invoke the LLM with the platform-specific tool set + prompt.

    This is the 'the model speaks' node: it sends the conversation to the
    LLM, which either requests tool calls (routing to tool_node) or decides
    it's done (routing to the finalize/compress step).
    """
    agent_type = state.get("agent_type", "reddit")
    plat = get_platform(agent_type)
    _rt = get_runtime()
    plat_tools = _runtime_platform_tools(_rt, agent_type)

    # ── Mode-specific prompt pieces ──
    output_mode = state.get("output_mode", "sources")
    discovery = state.get("discovery", False)
    mode_key = _mode_key(output_mode)
    report_mode_instruction = REPORT_MODE_INSTRUCTIONS.get(mode_key, "")
    finish_block = FINISH_INSTRUCTIONS.get(mode_key, {})
    finish_instruction = finish_block.get("task", "")
    finish_strategy_line = finish_block.get("strategy", "")

    if mode_key == "curation":
        gathering_instruction = CURATION_GATHERING
        mode_tools_section = CURATION_TOOL_GUIDANCE
        mode_strategy = DISCOVERY_STRATEGY if discovery else CURATION_STRATEGY
    elif mode_key == "full_context_sources":
        gathering_instruction = FULL_CONTEXT_GATHERING_SOURCES
        mode_tools_section = FULL_CONTEXT_SOURCES_TOOLS
        mode_strategy = DISCOVERY_FULL_CONTEXT_STRATEGY if discovery else FULL_CONTEXT_STRATEGY
    else:
        gathering_instruction = (
            DISCOVERY_FULL_CONTEXT_GATHERING_REPORT
            if discovery else FULL_CONTEXT_GATHERING_REPORT
        )
        mode_tools_section = FULL_CONTEXT_REPORT_TOOLS
        mode_strategy = DISCOVERY_FULL_CONTEXT_STRATEGY if discovery else FULL_CONTEXT_STRATEGY

    # ── Live budget counts + per-turn status plan ──
    # _compute_turn_plan is shared with tool_node, which folds the status text
    # into the last ToolMessage so the conversation stays append-only (the
    # provider's prompt cache can then reuse the history prefix across turns).
    plan = _compute_turn_plan(state)
    current_iteration = plan["current_iteration"]
    max_iter = plan["max_iter"]
    n_saved = plan["n_saved"]
    n_read = plan["n_read"]
    n_searches = plan["n_searches"]
    caps = plan["caps"]
    nudges = plan["nudges"]

    # ── Final-turn tool restriction ──
    if plan["forced_save"]:
        tools = [t for t in plat_tools
                 if t.name in {"batch_save_selected", "list_saved", "finish_research"}]
    else:
        tools = _tools_for_turn(plat_tools, mode_key, plan["is_final"])
        # Hide tools whose budget is exhausted so the model can't waste a turn
        # on them (e.g. searching again once the search cap is reached).
        if plan["n_searches"] >= plan["max_searches"]:
            tools = [t for t in tools if t.name not in SEARCH_TOOL_NAMES]
        if plan["n_read"] >= plan["max_total_reads"]:
            tools = [t for t in tools if t.name not in BATCH_READ_MAP]
        if plan["n_saved"] >= plan["max_saves"]:
            tools = [t for t in tools if t.name != "batch_save_selected"]

    target_language = state.get("target_language", "English")
    lang_set = bool(target_language and target_language.strip().lower() != "auto")

    # ── Conditional target-language guidance ──
    if lang_set:
        target_language_section = (
            f"TARGET_LANGUAGE: {target_language}\n\n"
            "CRITICAL OUTPUT LANGUAGE RULES:\n"
            "- Write ALL of your written outputs in TARGET_LANGUAGE.\n"
            "- Do NOT switch output language except unavoidable proper nouns, source titles, and direct quotes.\n"
            "- You may search in local languages when useful, but your written output must remain in TARGET_LANGUAGE.\n"
            "- The target language is set for this session and must NOT be changed."
        )
        set_language_tool_section = ""
        tools = [t for t in tools if t.name != "set_target_language"]
    else:
        target_language_section = (
            "The target language has NOT been set for this session.\n"
            "In your FIRST tool round, call **set_target_language** ALONGSIDE your "
            "other tool calls (e.g. your first searches) — it must NOT be given its own "
            "iteration. Do not start researching without setting it in that same first "
            "round. Once set, all of your written output must be in that language and "
            "must not change."
        )
        set_language_tool_section = (
            "2. **set_target_language**: Set the target language for this session. "
            "Call it in your very first tool round, in the SAME batch as your first "
            "searches — never as a standalone iteration."
        )

    model = _rt.models.subagent(tools=tools, chain=state.get("model_chain"))

    # ── Static system prompt (byte-stable across iterations for caching) ──
    prompt_template = DISCOVERY_AGENT_PROMPT if discovery else BASE_AGENT_PROMPT
    system_text = prompt_template.format(
        agent_type=plat["label"],
        date=get_today_str(),
        report_mode_instruction=report_mode_instruction,
        gathering_instruction=gathering_instruction,
        finish_instruction=finish_instruction,
        mode_tools_section=mode_tools_section,
        platform_section=plat["guidance"],
        mode_strategy=mode_strategy,
        finish_strategy_line=finish_strategy_line,
        target_language_section=target_language_section,
        set_language_tool_section=set_language_tool_section,
    )

    # Mixed-team moderation: a Chinese supervisor delegates to this
    # international sub-agent — keep subject references neutral so the
    # supervisor's next (Chinese-model) call isn't rejected.
    if _rt.config.chinese_supervisor_international_subagent and state.get("chinese_supervisor_international_subagent"):
        system_text += (
            "\n\n<Content note>Your findings are read by a supervisor whose "
            "provider restricts certain subjects. Keep descriptions factual "
            "and neutral, and avoid naming specific restricted subjects."
            "</Content note>\n"
        )

    messages = [SystemMessage(content=system_text)] + list(state["researcher_messages"])

    # ── Banner (caps and nudges rendered once each) ──
    cap_str = f"  caps: {', '.join(caps)}" if caps else ""
    warn_str = "  ⚠ " + "; ".join(nudges) if nudges else ""
    mode_tag = f"{Colors.YELLOW}{Colors.BOLD}DISCOVERY MODE{Colors.RESET}  " if discovery else ""
    if _rt.console_enabled:
        print(f"\n{'─' * 55}")
        print(f"  {mode_tag}{plat['label']} agent  |  iter {current_iteration + 1}/{max_iter}"
              f"  |  saved: {n_saved}  |  read: {n_read}  |  searches: {n_searches}{cap_str}{warn_str}")
        print(f"{'─' * 55}")

    try:
        response = await asyncio.wait_for(model.ainvoke(messages), timeout=_rt.config.llm_timeout)
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return {
            "researcher_messages": [AIMessage(
                content=f"I encountered an error ({e}). Here is what I have gathered so far.",
                tool_calls=[],
            )]
        }

    updates = {
        "researcher_messages": [AIMessage(
            content=response.content if isinstance(response.content, str) else "",
            tool_calls=response.tool_calls if hasattr(response, "tool_calls") else [],
            additional_kwargs={"reasoning_content": response.additional_kwargs.get("reasoning_content")},
        )]
    }
    if plan["forced_save"]:
        updates["final_save_round"] = True
    return updates


async def tool_node(state: ResearcherState):
    """Execute tool calls in parallel. Intercepts read/save/log/list for curation.

    All curation writes go into THIS invocation's state dicts — never any
    module globals — so concurrent agents stay fully isolated.
    Tools run concurrently via asyncio.gather, gated by a semaphore.
    """
    agent_type = state.get("agent_type", "reddit")
    plat = get_platform(agent_type)
    _rt = get_runtime()
    _rt.check_cancelled()
    tools_by_name = {t.name: t for t in _runtime_platform_tools(_rt, agent_type)}

    tool_calls = state["researcher_messages"][-1].tool_calls
    articles_read = dict(state.get("articles_read", {}))
    saved_articles = dict(state.get("saved_articles", {}))
    findings_log = dict(state.get("findings_log", {}))
    target_language = state.get("target_language", "English")
    search_count = state.get("search_count", 0)
    search_results = dict(state.get("search_results", {}))

    max_reads = state.get("max_reads", 8)          # per-iteration read budget
    max_total_reads = state.get("max_total_reads", DEFAULT_MAX_TOTAL_READS)  # total run budget
    max_saves = state.get("max_saves", 15)
    max_searches = state.get("max_searches", 8)
    max_concurrency = state.get("max_concurrency", 4)

    semaphore = asyncio.Semaphore(max_concurrency)

    # Count current-batch search calls for cap checking
    batch_search_count = sum(
        1 for tc in tool_calls
        if tc.get("name", "") in SEARCH_TOOL_NAMES
    )

    # Per-iteration read budget (max_reads is per-iteration, not a workflow
    # total). The total run budget (max_total_reads) further caps reads across
    # all iterations. Pre-assign how many items each batch read call may read
    # so parallel batch reads can't collectively exceed either budget.
    read_budget_by_tc: dict[str, int] = {}
    remaining_total = max(0, max_total_reads - len(articles_read))
    _read_budget = min(max_reads, remaining_total)
    for _tc in tool_calls:
        _tname = _tc.get("name", _tc.get("function", {}).get("name", ""))
        if _tname in BATCH_READ_MAP:
            _spec = BATCH_READ_MAP[_tname]
            _cap = _spec.get("cap") or max_reads
            _args = _tc.get("args", _tc.get("function", {}).get("arguments", {})) or {}
            if _tname == "fetch_urls":
                _n = len(_args.get("urls", []) or [])
            else:
                _n = len(_args.get("items", []) or [])
            _allowed = max(0, min(_n, _read_budget, _cap))
            read_budget_by_tc[_tc.get("id")] = _allowed
            _read_budget -= _allowed

    def _save_by_ref(ref: str, index: int, reason: str) -> str:
        """Resolve a [S#] ref + 1-based index to a canonical id and save it.

        Returns a status string for the batch_save_selected summary.
        """
        nonlocal search_results
        if not ref:
            return "Error: missing ref"
        entry = search_results.get(ref)
        if not entry:
            return f"Error: unknown ref '{ref}'"
        items = entry.get("items", [])
        if index < 1 or index > len(items):
            return f"Error: index {index} out of range (1-{len(items)})"
        identifier = str(items[index - 1].get("id", "")).strip()
        if not identifier:
            return "Error: item has no id"
        if identifier not in articles_read:
            return "You must read this item first before saving it. Use the platform's read tool, then try again."
        if identifier in saved_articles:
            return f"[ALREADY SAVED] {identifier}"
        if len(saved_articles) >= max_saves:
            return f"Error: save cap reached ({max_saves})"
        item = items[index - 1]
        saved_articles[identifier] = {
            "url": str(item.get("url", "")).strip() or identifier,
            "reason": reason,
            "content": articles_read.get(identifier, {}).get("content", ""),
            "title": str(item.get("title", "")) if items else "",
        }
        logger.info(f"Saved item: {identifier} ({len(saved_articles)} total)")
        log_source(
            tool_name=agent_type,
            link=str(item.get("url", "")).strip() or identifier,
            content=str(item.get("title", "") or identifier),
        )
        _emit_event(_events.SOURCE_SAVED, phase="subagent", agent=agent_type,
                    platform=agent_type,
                    url=str(item.get("url", "")).strip() or identifier,
                    title=str(item.get("title", "")) or identifier)
        return f"Saved {identifier} — {reason[:80]}"

    async def run_one(tc: dict) -> tuple[dict, str, dict]:
        """Execute one tool call with semaphore, error handling, and depth caps."""
        async with semaphore:
            t_name = tc.get("name", tc.get("function", {}).get("name", ""))
            t_args = tc.get("args", tc.get("function", {}).get("arguments", {}))
            tool_fn = tools_by_name.get(t_name)
            t_start = time.perf_counter()

            extra_state = {}

            try:
                # ── Intercept: finish_research (signal, not executed) ──
                if t_name == "finish_research":
                    summary = str(t_args.get("summary", "")).strip()
                    content = f"Research finalized. {summary[:500]}" if summary else "Research finalized."
                    elapsed = time.perf_counter() - t_start
                    print(f"    → {t_name}(...) in {elapsed:.1f}s")
                    return (tc, content, extra_state)

                # ── Intercept: set_target_language ──
                if t_name == "set_target_language":
                    lang = str(t_args.get("language", "")).strip()
                    if lang:
                        extra_state["target_language"] = lang
                        content = f"Target language set to: {lang}"
                    else:
                        content = "Please provide a language."
                    elapsed = time.perf_counter() - t_start
                    print(f"    → {t_name}(...) in {elapsed:.1f}s")
                    return (tc, content, extra_state)

                # ── Intercept: log_finding ──
                if t_name == LOG_FINDING_TOOL:
                    key = str(t_args.get("key", "")).strip()
                    value = str(t_args.get("value", "")).strip()
                    if not key or not value:
                        content = "Please provide both key and value for the finding."
                    else:
                        findings_log[key] = value
                        content = f"Logged finding [{len(findings_log)}]: {key} = {value[:120]}"
                    elapsed = time.perf_counter() - t_start
                    print(f"    → {t_name}(...) in {elapsed:.1f}s")
                    return (tc, content, extra_state)

                # ── Intercept: list_saved (read from state) ──
                if t_name == "list_saved":
                    parts = []
                    if saved_articles:
                        parts.append(f"## Saved Items ({len(saved_articles)} total)\n")
                        for idx, (id_, item) in enumerate(sorted(saved_articles.items()), 1):
                            title = item.get("title") or id_
                            parts.append(f"{idx}. **{title}**")
                            parts.append(f"   URL: {item.get('url', id_)}")
                            parts.append(f"   Reason: {item.get('reason', '')}")
                            parts.append("")
                    else:
                        parts.append("No items saved yet. Use your save tool after reading.")
                    if findings_log:
                        parts.append(f"## Logged Findings ({len(findings_log)} total)")
                        for k, v in findings_log.items():
                            parts.append(f"  **{k}**: {v}")
                    content = "\n".join(parts)
                    elapsed = time.perf_counter() - t_start
                    print(f"    → {t_name}(...) in {elapsed:.1f}s")
                    return (tc, content, extra_state)

                # ── Intercept: batch_save_selected (end-of-run selection) ──
                if t_name == "batch_save_selected":
                    items = t_args.get("items", []) or []
                    if not isinstance(items, list):
                        items = []
                    parts = []
                    new_count = 0
                    for entry in items:
                        if not isinstance(entry, dict):
                            parts.append(f"- invalid entry: {entry}")
                            continue
                        ref = str(entry.get("ref", "")).strip()
                        index = int(entry.get("index", 0) or 0)
                        reason = str(entry.get("reason", "")).strip()
                        entry_result = _save_by_ref(ref, index, reason)
                        parts.append(f"- [{ref}:{index}] {entry_result}")
                        if entry_result.startswith("Saved"):
                            new_count += 1
                    content = f"batch_save_selected: {new_count} saved.\n" + "\n".join(parts)
                    elapsed = time.perf_counter() - t_start
                    print(f"    → {t_name}(...) in {elapsed:.1f}s")
                    return (tc, content, extra_state)

                # ── Intercept: batch read tools (record articles_read) ──
                if t_name in BATCH_READ_MAP:
                    spec = BATCH_READ_MAP[t_name]
                    id_key = spec["id_key"]
                    fetch_fn = spec["fetch_fn"]
                    allowed = read_budget_by_tc.get(tc.get("id"), max_reads)

                    if t_name == "fetch_urls":
                        all_entries = [{"url": str(u)} for u in (t_args.get("urls", []) or [])]
                    else:
                        all_entries = [e for e in (t_args.get("items", []) or [])
                                       if isinstance(e, dict)]
                    requested = len(all_entries)
                    entries = all_entries[:allowed]

                    if not entries:
                        if requested and remaining_total <= 0:
                            content = (f"Read cap reached ({max_total_reads} total items). "
                                       f"Use batch_save_selected on what you have, then finalize.")
                        else:
                            content = (f"No valid items passed to {t_name}. Pass "
                                       "items=[{\"ref\": \"S1\", \"index\": 2}] or a direct id.")
                        elapsed = time.perf_counter() - t_start
                        print(f"    → {t_name}(...) in {elapsed:.1f}s")
                        return (tc, content, extra_state)

                    # Resolve each entry to a canonical identifier.
                    resolved = []
                    errors = []
                    for e in entries:
                        index = int(e.get("index", 0) or 0)
                        ref = str(e.get("ref", "") or "").strip()
                        if index and not ref:
                            errors.append("index requires ref")
                            continue
                        if index:
                            entry = search_results.get(ref)
                            if not entry:
                                errors.append(f"unknown ref '{ref}'")
                                continue
                            items_lst = entry.get("items", [])
                            if index < 1 or index > len(items_lst):
                                errors.append(f"index {index} out of range for '{ref}'")
                                continue
                            identifier = str(items_lst[index - 1].get("id", "")).strip()
                        else:
                            identifier = str(e.get(id_key, "")).strip()
                        if not identifier:
                            errors.append("item has no id")
                            continue
                        resolved.append(identifier)

                    if not resolved:
                        content = "No readable items: " + "; ".join(errors[:5])
                        elapsed = time.perf_counter() - t_start
                        print(f"    → {t_name}(...) in {elapsed:.1f}s")
                        return (tc, content, extra_state)

                    async def _read_one(identifier: str):
                        # Serve the full text already captured by a web search
                        # instead of re-fetching the URL (avoid double fetch).
                        if t_name == "fetch_urls":
                            cached = _lookup_search_full_text(search_results, identifier)
                            if cached is not None:
                                return cached
                        if t_name == "get_reddit_posts":
                            return await fetch_fn(
                                identifier,
                                include_comments=bool(t_args.get("include_comments", True)))
                        if t_name == "read_filings":
                            return await fetch_fn(identifier, sections=t_args.get("sections"))
                        return await fetch_fn(identifier)

                    fetched = await asyncio.gather(*[_read_one(i) for i in resolved])

                    # fetch_urls returns (title, body). Register the fetched URLs
                    # as indexable items so external link-chasing sources can be
                    # saved later via batch_save_selected by [S#] ref.
                    handle = _handles_by_tc.get(tc.get("id"))
                    if handle and t_name == "fetch_urls":
                        items = []
                        for identifier, result in zip(resolved, fetched):
                            title = result[0] if isinstance(result, tuple) else identifier
                            items.append({"id": identifier, "title": title, "url": identifier})
                        search_results[handle] = {"tool": t_name, "items": items}

                    parts = []
                    skipped = 0
                    for identifier, result in zip(resolved, fetched):
                        body = result[1] if isinstance(result, tuple) else result
                        # Chinese moderation: don't let sensitive article bodies
                        # enter the model's context at all.
                        if _rt.config.chinese_moderation and contains_sensitive(body):
                            skipped += 1
                            continue
                        articles_read[identifier] = {"content": body}
                        tag = _find_ref_tag(search_results, identifier)
                        header = f"[{tag}] {identifier}" if tag else identifier
                        parts.append(f"=== {header} ===\n\n{body}")
                    if skipped:
                        parts.append(f"(source skipped: {skipped} item(s) filtered)")
                    if errors:
                        parts.append("(Skipped: " + "; ".join(errors[:5]) + ")")
                    if requested > allowed:
                        parts.append(f"(Read cap: {allowed} items this iteration "
                                     f"— {requested - allowed} not read. Use batch_save_selected on what you have.)")
                    content = "\n\n".join(parts)
                    _emit_event(_events.SOURCE_READ, phase="subagent", agent=agent_type,
                                platform=agent_type, tool=t_name, count=len(resolved))
                    elapsed = time.perf_counter() - t_start
                    print(f"    → {t_name}({len(resolved)} items) in {elapsed:.1f}s")
                    return (tc, content, extra_state)

                # ── Normal tool call (search, etc.) ──
                if tool_fn is not None:
                    # Cap check for search tools (state count + pre-computed batch cap)
                    if t_name in SEARCH_TOOL_NAMES and (
                            search_count >= max_searches or tc.get("id") in capped_search_ids):
                        content = f"Search cap reached ({max_searches}). Read and save what you have, then finalize."
                        elapsed = time.perf_counter() - t_start
                        print(f"    → {t_name}(...) in {elapsed:.1f}s [CAP]")
                        return (tc, content, extra_state)

                    result = await tool_fn.ainvoke(t_args)
                    if t_name in SEARCH_TOOL_NAMES:
                        extra_state["_search_increment"] = 1

                    # Indexable search tools return {"display", "items"} — tag with
                    # a handle, prefix the display, and register for index reads.
                    if t_name in SEARCH_INDEXABLE_TOOLS and isinstance(result, dict):
                        handle = _handles_by_tc.get(tc.get("id"), "")
                        items = result.get("items", [])
                        display = result.get("display", "")
                        # Chinese moderation: drop any source whose title/URL
                        # matches a sensitive pattern before the model sees it.
                        if _rt.config.chinese_moderation and items:
                            kept = [
                                it for it in items
                                if not contains_sensitive(
                                    f"{it.get('title', '')} {it.get('url', '')}"
                                )
                            ]
                            if len(kept) < len(items):
                                dropped = len(items) - len(kept)
                                items = kept
                                if kept:
                                    display = (
                                        "## Search results (some sources filtered)\n\n"
                                        + "\n".join(
                                            f"- **{it.get('title', '')}**\n  URL: {it.get('url', '')}"
                                            for it in kept
                                        )
                                        + f"\n\n_({dropped} source(s) filtered)_"
                                    )
                                else:
                                    display = (
                                        "## Search results\n\n"
                                        f"(all {dropped} results filtered out)"
                                    )
                        if handle:
                            search_results[handle] = {"tool": t_name, "items": items}
                            content = f"[{handle}] {display}"
                            _emit_event(_events.SOURCE_FOUND, phase="subagent", agent=agent_type,
                                        platform=agent_type, tool=t_name, count=len(items))
                        else:
                            content = display
                    else:
                        content = str(result)
                    elapsed = time.perf_counter() - t_start
                    print(f"    → {t_name}(...) in {elapsed:.1f}s")
                    return (tc, content, extra_state)

                # ── Unknown tool ──
                elapsed = time.perf_counter() - t_start
                print(f"    → {t_name}(...) in {elapsed:.1f}s")
                return (tc, f"Unknown tool: {t_name}", extra_state)

            except Exception as e:
                elapsed = time.perf_counter() - t_start
                logger.warning(f"Tool {t_name} failed: {e}")
                print(f"    → {t_name}(...) in {elapsed:.1f}s [ERROR: {type(e).__name__}]")
                return (tc, f"Error executing {t_name}: {type(e).__name__}: {str(e)[:200]}", extra_state)

    # Phase-ordered execution: reads/searches first (parallel), then saves (parallel).
    # Saves depend on articles_read being populated by reads in the same batch.
    SAVE_TOOL_NAMES = {"batch_save_selected"}
    phase_0 = []  # reads, searches, fetches, session tools
    phase_1 = []  # saves

    for tc in tool_calls:
        t_name = tc.get("name", tc.get("function", {}).get("name", ""))
        if t_name in SAVE_TOOL_NAMES:
            phase_1.append(tc)
        else:
            phase_0.append(tc)

    # Pre-assign [S#] handles to indexable search calls BEFORE parallel execution,
    # so concurrent searches get deterministic, collision-free handles. fetch_urls
    # is included so external link-chasing URLs can also be saved by [S#] ref.
    _handles_by_tc = {}
    _next_handle = len(search_results) + 1
    for tc in phase_0:
        t_name = tc.get("name", tc.get("function", {}).get("name", ""))
        if t_name in SEARCH_INDEXABLE_TOOLS or t_name == "fetch_urls":
            _handles_by_tc[tc.get("id")] = f"S{_next_handle}"
            _next_handle += 1

    # Pre-compute batch search budget so concurrent searches in ONE batch can't
    # all pass the cap check against the same stale search_count (the bug that
    # let searches run past max_searches). Excess search calls get capped.
    batch_search_calls = [tc for tc in phase_0
                          if tc.get("name", "") in SEARCH_TOOL_NAMES]
    search_budget = max(0, max_searches - search_count)
    capped_search_ids = {tc.get("id") for tc in batch_search_calls[search_budget:]}

    all_results = []
    if phase_0:
        all_results.extend(await asyncio.gather(*[run_one(tc) for tc in phase_0]))
    if phase_1:
        all_results.extend(await asyncio.gather(*[run_one(tc) for tc in phase_1]))
    results = all_results

    # Apply results sequentially (deterministic ordering for state)
    observations = []
    search_increment = 0
    for tc, content, extra in results:
        observations.append(ToolMessage(content=content, tool_call_id=tc["id"]))
        if "_search_increment" in extra:
            search_increment += extra["_search_increment"]
        if "target_language" in extra:
            target_language = extra["target_language"]

    # ── Next-turn status folded into the last observation ──
    # Persisting the status here (instead of injecting a fresh trailing user
    # message in llm_call) keeps the conversation append-only, so the provider's
    # prompt cache serves the accumulated history and only this turn's delta is
    # paid. The plan is computed from the POST-tool state so the counts shown
    # are exactly what the next llm_call will see.
    post_state = dict(state)
    post_state["articles_read"] = articles_read
    post_state["saved_articles"] = saved_articles
    post_state["search_count"] = search_count + search_increment
    post_state["search_results"] = search_results
    post_state["target_language"] = target_language
    status_text = _compute_turn_plan(post_state)["status_text"]
    if observations and status_text:
        observations[-1] = ToolMessage(
            content=observations[-1].content + "\n\n" + status_text,
            tool_call_id=observations[-1].tool_call_id,
        )

    return {
        "researcher_messages": observations,
        "articles_read": articles_read,
        "saved_articles": saved_articles,
        "findings_log": findings_log,
        "search_count": search_count + search_increment,
        "target_language": target_language,
        "search_results": search_results,
    }


def should_continue(state: ResearcherState) -> Literal["tool_node", "compress_research"]:
    """Route: LLM requested tools → tool_node; otherwise research is done.

    Also routes to compress_research when finish_research was called (early
    exit signal). The max_iterations cap is enforced AFTER tool_node runs, so
    final-turn tool calls (e.g. batch_save_selected) always get processed.
    """
    messages = state["researcher_messages"]
    last = messages[-1]

    if hasattr(last, "tool_calls") and last.tool_calls:
        for tc in last.tool_calls:
            if tc.get("name") == "finish_research":
                logger.info("Agent called finish_research. Finalizing.")
                return "compress_research"
        return "tool_node"

    return "compress_research"


def route_after_tools(state: ResearcherState) -> Literal["llm_call", "compress_research"]:
    """After tool execution, check the iteration cap.

    Runs after tool_node so the last turn's tool calls (e.g.
    batch_save_selected) are always executed before finalizing.
    """
    messages = state["researcher_messages"]
    current_iteration = _iteration_count(messages)
    max_iter = state.get("max_iterations", 10)
    if current_iteration >= max_iter:
        if (
            _mode_key(state.get("output_mode", "sources")) == "curation"
            and not state.get("final_save_round")
            and len(state.get("saved_articles", {})) < len(state.get("articles_read", {}))
        ):
            logger.info("Granting final save round: read-but-unsaved items remain")
            return "llm_call"
        logger.info(f"Max iterations ({max_iter}) reached. Finalizing.")
        return "compress_research"
    return "llm_call"


def _fallback_select_reads(articles_read: dict, search_results: dict, limit: int = 10) -> dict:
    """Auto-select the most recently read items when batch_save_selected was never called.

    Fallback only — fired when the agent failed to deliver its selection.
    """
    saved = {}
    for ident, rec in list(articles_read.items())[-limit:]:
        saved[ident] = {
            "url": ident,
            "reason": "auto-selected (agent did not call batch_save_selected)",
            "title": _find_title(search_results, ident) or ident,
            "content": str(rec.get("content", "")),
        }
    return saved


async def _fallback_report_from_reads(state: ResearcherState, articles_read: dict) -> str:
    """Synthesize a report from read content when the agent never wrote one.

    Fallback only — fired when the agent exhausted its budget without writing
    the report as its final message.
    """
    from deep_research.subagent_report_write import generate_report
    search_results = state.get("search_results", {})
    saved = {}
    for ident, rec in list(articles_read.items())[-10:]:
        saved[ident] = {
            "url": ident,
            "reason": "read during research (fallback: agent did not write a report)",
            "title": _find_title(search_results, ident) or ident,
            "content": str(rec.get("content", "")),
        }
    plat = get_platform(state.get("agent_type", "reddit"))
    return await generate_report(
        saved_articles=saved,
        topic=state.get("research_topic", ""),
        platform_label=plat.get("label", state.get("agent_type", "reddit")),
        model_chain=state.get("model_chain"),
    )


def _build_full_context_findings(state: ResearcherState) -> str:
    """Compact source digest for the full-context report citation repair pass.

    The full-context agent writes the report inline (it has no curated
    saved_articles dict), so the repair pass needs a fallback source list to
    recover/verify URLs. Built from search_results items plus articles_read ids.
    """
    sources: dict[str, str] = {}
    for entry in (state.get("search_results") or {}).values():
        for item in entry.get("items", []):
            url = str(item.get("url") or item.get("id") or "").strip()
            if url and url not in sources:
                sources[url] = str(item.get("title") or url)
    for ident in (state.get("articles_read") or {}):
        ident = str(ident).strip()
        if ident and ident not in sources:
            sources[ident] = ident
    if not sources:
        return ""
    parts = ["# Sources"]
    for i, (url, title) in enumerate(sources.items(), 1):
        parts.append(f"[{i}] {title} ({url})")
    return "\n".join(parts)


async def compress_research(state: ResearcherState) -> dict:
    """Finalize the run. Synthesize from the curated saved_articles dict.

    This is the 'synthesize from the dictionary, not the noisy conversation'
    principle: the compressed summary is built primarily from saved items
    (with their reasons), falling back to raw messages only when nothing
    was saved.
    """
    saved = state.get("saved_articles", {})
    findings = state.get("findings_log", {})
    messages = state.get("researcher_messages", [])
    output_mode = state.get("output_mode", "sources")

    # In report_inline mode, the agent's final message IS the report.
    # Only capture the last AIMessage with NO tool_calls and non-empty content
    # (a tool-call message like finish_research has empty content).
    report = ""
    if output_mode == "report_inline":
        for m in reversed(list(messages)):
            if not isinstance(m, AIMessage):
                continue
            if getattr(m, "tool_calls", None):
                # Last message was a tool call (e.g. finish_research) — try to
                # recover the report from its summary argument.
                for tc in m.tool_calls:
                    if isinstance(tc, dict) and tc.get("name") == "finish_research":
                        summary = (tc.get("args") or {}).get("summary", "") or ""
                        if summary:
                            report = summary
                            break
                if report:
                    break
                continue
            if (m.content or "").strip():
                report = m.content
                break

    # Full-context reports are written inline by the agent. Rebuild ## Sources
    # deterministically in code from the source registry (URLs never come from
    # the model) — same citation flow as the curation report writer.
    #
    # The registry is only meaningful for report_inline, whose inline [code]
    # citations index into it. The curation `report` mode writes against a
    # DIFFERENT registry (sorted saved_articles, in subagent_report_write), so
    # it must NOT emit this registry — otherwise the supervisor would renumber
    # its citations against the wrong order.
    source_registry = (
        _build_source_registry(
            state.get("search_results", {}), state.get("articles_read", {}))
        if output_mode == "report_inline" else []
    )
    if report and output_mode == "report_inline":
        from deep_research.citation_utils import finalize_citations
        report = finalize_citations(report, source_registry)

    # In report modes, the report is written by the report_write node
    # (output_mode "report"), or is the agent's final message (report_inline).
    # Only ONE synthesis per run.
    if _delivers_report(output_mode):
        if report:
            logger.info("Compress step: captured the agent's final report message")
            compressed = report
        elif output_mode == "report":
            # Backstop: the agent never saved anything → auto-select the most
            # recently read items so the report write always has real sources.
            if not saved and state.get("articles_read"):
                logger.warning("Fallback: agent did not save any sources — "
                               "auto-selecting most recently read items")
                saved = _fallback_select_reads(
                    state["articles_read"], state.get("search_results", {}))
            # report_write overwrites compressed_research with the synthesized
            # report in a follow-up node, so nothing to emit here.
            compressed = "" if saved else (
                "(No sources were found during this session — report skipped.)")
            logger.info("Compress step: skipped in 'report' mode "
                        "(report_write handles the synthesis)")
        elif state.get("articles_read"):
            logger.warning("Fallback: agent did not write a report — "
                           "auto-synthesizing from read content")
            report = await _fallback_report_from_reads(state, state["articles_read"])
            compressed = report
        else:
            logger.warning("No report was written and nothing was read — report is empty")
            compressed = "(No report was written and nothing was read during this session.)"
    else:
        # ── Sources modes (sources / sources_inline): markdown list (no LLM) ──
        include_text = state.get("include_article_text", False)
        n_read = len(state.get("articles_read", {}))

        # Fallback: the agent never called batch_save_selected → auto-select
        # the most recently read items so the output is never empty.
        if not saved and state.get("articles_read"):
            logger.warning("Fallback: agent did not call batch_save_selected — "
                           "auto-selecting most recently read items")
            saved = _fallback_select_reads(
                state["articles_read"], state.get("search_results", {}))

        date_str = get_today_str()
        plat_label = state.get("agent_type", "unknown")
        lines = [
            f"# Sources: {state.get('research_topic', '')}",
            f"_{len(saved)} selected of {n_read} read · {plat_label} · {date_str}_",
            "",
        ]
        if not saved:
            lines.append(
                f"No sources were selected "
                f"(agent read {n_read} items but never called batch_save_selected).")
        else:
            for i, (ident, item) in enumerate(saved.items(), 1):
                title = item.get("title") or ident
                url = item.get("url", ident)
                reason = item.get("reason", "")
                lines.append(f"## {i}. {title}")
                lines.append(f"- URL: {url}")
                lines.append(f"- Why selected: {reason}")
                if include_text:
                    full_content = item.get("content", "")
                    if full_content:
                        lines.append("")
                        lines.append("### Full text")
                        lines.append(full_content)
                lines.append("")
        compressed = "\n".join(lines)
        logger.info(f"Compress step: formatted {len(saved)} sources "
                    f"(include_text={include_text})")

    iteration_count = _iteration_count(messages)

    return {
        "compressed_research": compressed,
        "saved_articles": saved,
        "findings_log": findings,
        "iteration_count": iteration_count,
        "source_registry": source_registry,
    }


async def report_write(state: ResearcherState) -> dict:
    """Write the markdown report from the full curated sources.

    Runs in curation + report mode (the agent saved sources as it went, and a
    dedicated LLM call turns them into the report). In full_context + report,
    the agent's final message is the report — captured by compress_research, so
    this node is not reached.
    """
    from deep_research.subagent_report_write import generate_report

    plat = get_platform(state.get("agent_type", "reddit"))
    n_sources = len(state.get("saved_articles", {}))
    if n_sources == 0:
        msg = ("No sources were found during this session — report skipped. "
               "Nothing was read or saved that could be synthesized.")
        logger.warning(msg)
        return {"compressed_research": msg}
    _w_start = time.perf_counter()
    logger.info(f"Report write: generating report from {n_sources} curated sources...")
    report_text = await generate_report(
        saved_articles=state.get("saved_articles", {}),
        topic=state.get("research_topic", ""),
        platform_label=plat.get("label", state.get("agent_type", "reddit")),
        model_chain=state.get("model_chain"),
    )
    logger.info(f"Report write: done in {time.perf_counter() - _w_start:.1f}s ({len(report_text)} chars)")
    return {"compressed_research": report_text}


def route_after_compress(state: ResearcherState) -> str:
    """Route to the report writer only in "report" mode WITH sources.

    If nothing was saved (and nothing could be auto-selected because nothing
    was read), skip the report writer entirely — no point synthesizing an empty
    source set, and it avoids a bogus report failing the citation gate.
    """
    if (
        state.get("output_mode") == "report"
        and len(state.get("saved_articles", {})) > 0
    ):
        return "report_write"
    return END


# ── Graph construction ─────────────────────────────────────────────────

_research_agent_cache = None


def _build_agent():
    global _research_agent_cache
    if _research_agent_cache is not None:
        return _research_agent_cache

    builder = StateGraph(ResearcherState, output_schema=ResearcherOutputState)
    builder.add_node("llm_call", llm_call)
    builder.add_node("tool_node", tool_node)
    builder.add_node("compress_research", compress_research)
    builder.add_node("report_write", report_write)
    builder.add_edge(START, "llm_call")
    builder.add_conditional_edges("llm_call", should_continue, {
        "tool_node": "tool_node",
        "compress_research": "compress_research",
    })
    builder.add_conditional_edges("tool_node", route_after_tools, {
        "llm_call": "llm_call",
        "compress_research": "compress_research",
    })
    builder.add_conditional_edges("compress_research", route_after_compress, {
        "report_write": "report_write",
        END: END,
    })
    builder.add_edge("report_write", END)

    _research_agent_cache = builder.compile()
    return _research_agent_cache


researcher_agent = _build_agent()
