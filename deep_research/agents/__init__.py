"""Agent Registry — maps supervisor tool-call names to platform agent graphs.

The platform engine (deep_research.agents.base) is ONE compiled graph that
serves every platform: the platform, discovery mode, provider, and depth caps
are read from the invocation state at runtime. Each registry entry is a thin
adapter that injects those per-agent defaults before invoking the shared
graph.

Adding a new platform agent:
1. Create agents/{platform}/tools.py with @tool functions
2. Add the platform entry to PLATFORMS in agents/base.py
3. Add one entry to AGENT_REGISTRY below

The supervisor spawn loop (multi_agent_supervisor.py) reads this registry
at runtime — no spawn code changes needed per new agent.
"""

import asyncio

from langchain_core.messages import HumanMessage

from deep_research.agents.base import researcher_agent, _compute_turn_plan
from deep_research.config import (
    DEFAULT_MAX_TOTAL_READS,
    subagent_recursion_limit,
)
from deep_research.runtime import get_runtime

# ── Default depth caps (supervisor-launched runs) ──────────────────────
# Single source of truth: deep_research.config.SUBAGENT_MAX_* constants,
# shared with run_platform.py so both entry points agree.

# Platform key → production curated-source source_type label
_SOURCE_TYPE_MAP = {
    "reddit": "reddit_post",
    "pubmed": "pubmed_article",
    "sec_edgar": "sec_filing",
    "arxiv": "arxiv_paper",
    "substack": "substack",
    "web": "web",
    "general": "web",
}


def _saved_to_curated(saved_articles: dict, agent_type: str) -> list[dict]:
    """Convert the engine's saved_articles dict to the pipeline's curated_sources list.

    saved_articles: {identifier: {"url", "reason", "content", "title"}}
    curated_sources: [{url, title, full_text, reason, source_type}]
    """
    source_type = _SOURCE_TYPE_MAP.get(agent_type, "web")
    curated = []
    for identifier, item in saved_articles.items():
        curated.append({
            "url": item.get("url") or identifier,
            "title": item.get("title", "") or "",
            "full_text": item.get("content", "") or "",
            "reason": item.get("reason", "") or "",
            "source_type": source_type,
        })
    return curated


class _ConfiguredAgent:
    """Thin adapter: injects per-agent defaults into the shared engine graph.

    The supervisor spawn loop calls .ainvoke({researcher_messages, ...}) with
    only a handful of keys. This adapter fills in the platform, discovery
    flag, provider, api keys, output mode, and depth caps that the engine
    needs, and passes a recursion_limit large enough for the iteration cap.
    """

    def __init__(self, *, agent_type: str, discovery: bool = False,
                 model_chain: list | None = None):
        self._agent_type = agent_type
        self._discovery = discovery
        # Per-agent model chain override (from SUBAGENT_MODEL_CHAIN_BY_AGENT);
        # None = use the shared SUBAGENT_MODEL_FALLBACK_CHAIN.
        self._model_chain = model_chain
        self._graph = researcher_agent

    async def ainvoke(self, input: dict, config: dict | None = None):
        cfg = get_runtime().config
        # Per-agent model chain override: run config takes precedence over the
        # import-time default (SUBAGENT_MODEL_CHAIN_BY_AGENT).
        override_chain = cfg.subagent_model_chain_by_agent.get(self._agent_type) \
            if cfg.subagent_model_chain_by_agent else None
        model_chain = override_chain or self._model_chain
        defaults = {
            "agent_type": self._agent_type,
            "discovery": self._discovery,
            "output_mode": cfg.discovery_output_mode if self._discovery else cfg.subagent_output_mode,
            "include_article_text": False,
            "max_iterations": int(cfg.subagent_max_iterations),
            "max_reads": int(cfg.subagent_max_reads),
            "max_total_reads": int(cfg.subagent_max_total_reads),
            "max_saves": int(cfg.subagent_max_saves),
            "max_searches": int(cfg.subagent_max_searches),
            "max_concurrency": int(cfg.subagent_max_concurrency),
        }
        if model_chain:
            defaults["model_chain"] = list(model_chain)
        merged = {**defaults, **input}

        # Fold the FIRST-turn session status into the topic message. Subsequent
        # turns have it folded into the last ToolMessage by tool_node, so the
        # whole conversation is append-only and the prompt cache can reuse the
        # history prefix from turn 2 onward.
        msgs = list(merged.get("researcher_messages") or [])
        if msgs:
            first = msgs[0]
            plan = _compute_turn_plan(merged)
            content = (first.content if hasattr(first, "content") else str(first)) + "\n\n" + plan["status_text"]
            msgs[0] = HumanMessage(content=content)
            merged["researcher_messages"] = msgs

        # recursion_limit must exceed max_iterations * 2 (LLM turn + tool round)
        recursion_limit = subagent_recursion_limit(int(merged.get("max_iterations", 5)))
        cfg = dict(config or {})
        cfg.setdefault("recursion_limit", recursion_limit)

        result = await self._graph.ainvoke(merged, cfg)

        # Convert the engine's saved_articles dict to the pipeline's
        # curated_sources list so the supervisor/final-report flow is unchanged.
        if isinstance(result, dict):
            saved = result.get("saved_articles") or {}
            result["curated_sources"] = _saved_to_curated(saved, self._agent_type)
            source_type = _SOURCE_TYPE_MAP.get(self._agent_type, "web")
            for entry in result.get("source_registry") or []:
                if isinstance(entry, dict):
                    entry.setdefault("source_type", source_type)
        return result


# ── Registry ────────────────────────────────────────────────────────────

AGENT_REGISTRY = {
    # One entry per platform. `discovery` defaults to False — the supervisor
    # overrides it per call via the `discovery` argument on each tool.
    # Per-agent model chains come from the run config's
    # SUBAGENT_MODEL_CHAIN_BY_AGENT equivalent (resolved per run in
    # _ConfiguredAgent.ainvoke); the shared fallback chain is used otherwise.
    "ResearchWeb":       _ConfiguredAgent(agent_type="web", discovery=False, model_chain=None),
    "ResearchGeneral":   _ConfiguredAgent(agent_type="general", discovery=False, model_chain=None),
    "ResearchReddit":    _ConfiguredAgent(agent_type="reddit", discovery=False, model_chain=None),
    "ResearchSubstack":  _ConfiguredAgent(agent_type="substack", discovery=False, model_chain=None),
    "ResearchPubMed":    _ConfiguredAgent(agent_type="pubmed", discovery=False, model_chain=None),
    "ResearchArxiv":     _ConfiguredAgent(agent_type="arxiv", discovery=False, model_chain=None),
    "ResearchSEC":       _ConfiguredAgent(agent_type="sec_edgar", discovery=False, model_chain=None),
}
