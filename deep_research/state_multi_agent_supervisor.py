
"""
State Definitions for Multi-Agent Research Supervisor

This module defines the state objects and tools used for the multi-agent
research supervisor workflow, including coordination state and research tools.
"""

import operator
from typing_extensions import Annotated, TypedDict, Sequence, Optional

from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

class SupervisorState(TypedDict):
    """
    State for the multi-agent research supervisor.

    Manages coordination between supervisor and research agents, tracking
    research progress and accumulating findings from multiple sub-agents.
    """

    # Messages exchanged with supervisor for coordination and decision-making
    supervisor_messages: Annotated[Sequence[BaseMessage], add_messages]
    # Detailed research brief that guides the overall research direction
    research_brief: str
    # Processed and structured notes ready for final report generation
    notes: Annotated[list[str], operator.add] = []
    # Counter tracking the number of research iterations performed
    research_iterations: int = 0
    # Curation: sources the sub-agents selected with reasons and full text
    curated_sources: Annotated[list[dict], operator.add] = []
    # Source registries emitted by sub-agents (title+URL only) for citation resolution
    source_registry: Annotated[list[dict], operator.add] = []
    # Draft report
    draft_report: str
    # Start time of the research process for time-conscious agents
    start_time: float = 0.0
    # Canonical target language for delegated subagent outputs
    target_language: str
    # Set to True when circuit breaker aborts research (e.g., API quota exhausted)
    aborted: bool = False
    # Human-readable reason for an abort (surfaces a truthful failure message)
    abort_reason: str = ""
    # Consecutive supervisor-level failed iterations (retry-once each; >=2 stops)
    consecutive_failures: int = 0
    # Final report written by the write_final_report node (maps up to AgentState)
    final_report: str = ""

# ── Platform-specific sub-agent tool schemas ──────────────────────────
# One schema per platform. All take research_topic plus a `discovery` flag
# that switches the sub-agent between focused research and broad exploratory
# discovery. The platform, provider, output_mode, and caps are injected by the
# registry adapter (output_mode is config-driven, not per-call).

@tool
class ResearchWeb(BaseModel):
    """Delegate to the web search sub-agent (Tavily, Exa, Google + fetch_urls). Best for general research topics."""
    research_topic: str = Field(
        description="The topic to research on the web. Should be a single topic, described in high detail (at least a paragraph).",
    )
    discovery: bool = Field(
        default=False,
        description="Set True for DISCOVERY mode (broad, exploratory sweep for leads and novel angles). False (default) is focused deep-dive research.",
    )
    max_total_reads: Optional[int] = Field(
        default=None,
        description="Optional override for this sub-agent's total read budget (default 25). Set higher to read more deeply, lower for a narrow check.",
    )


@tool
class ResearchComplete(BaseModel):
    """Tool for indicating that the research process is complete."""
    pass


@tool
class ResearchGeneral(BaseModel):
    """Delegate to the cross-platform general sub-agent (all tools: web + Reddit + Substack + PubMed + arXiv + SEC). Best when the best platform is unknown."""
    research_topic: str = Field(
        description="The topic to research across platforms. Should be a single topic, described in high detail (at least a paragraph).",
    )
    discovery: bool = Field(
        default=False,
        description="Set True for DISCOVERY mode (broad, exploratory sweep for leads and novel angles). False (default) is focused deep-dive research.",
    )
    max_total_reads: Optional[int] = Field(
        default=None,
        description="Optional override for this sub-agent's total read budget (default 25). Set higher to read more deeply, lower for a narrow check.",
    )


@tool
class ResearchReddit(BaseModel):
    """Delegate to the Reddit sub-agent for community sentiment, forum discussion, and grassroots takes."""
    research_topic: str = Field(
        description="The topic to research on Reddit. Should be a single topic, described in high detail (at least a paragraph).",
    )
    discovery: bool = Field(
        default=False,
        description="Set True for DISCOVERY mode (broad, exploratory sweep for leads and novel angles). False (default) is focused deep-dive research.",
    )
    max_total_reads: Optional[int] = Field(
        default=None,
        description="Optional override for this sub-agent's total read budget (default 25). Set higher to read more deeply, lower for a narrow check.",
    )


@tool
class ResearchSubstack(BaseModel):
    """Delegate to the Substack sub-agent for expert newsletter analysis and long-form opinion pieces."""
    research_topic: str = Field(
        description="The topic to research on Substack. Should be a single topic, described in high detail (at least a paragraph).",
    )
    discovery: bool = Field(
        default=False,
        description="Set True for DISCOVERY mode (broad, exploratory sweep for leads and novel angles). False (default) is focused deep-dive research.",
    )
    max_total_reads: Optional[int] = Field(
        default=None,
        description="Optional override for this sub-agent's total read budget (default 25). Set higher to read more deeply, lower for a narrow check.",
    )


@tool
class ResearchPubMed(BaseModel):
    """Delegate to the PubMed sub-agent for biomedical literature (drug efficacy, clinical trials, medical evidence)."""
    research_topic: str = Field(
        description="The topic to research on PubMed. Should be a single topic, described in high detail (at least a paragraph).",
    )
    discovery: bool = Field(
        default=False,
        description="Set True for DISCOVERY mode (broad, exploratory sweep for leads and novel angles). False (default) is focused deep-dive research.",
    )
    max_total_reads: Optional[int] = Field(
        default=None,
        description="Optional override for this sub-agent's total read budget (default 25). Set higher to read more deeply, lower for a narrow check.",
    )


@tool
class ResearchArxiv(BaseModel):
    """Delegate to the arXiv sub-agent for academic preprints (CS, physics, math, bio)."""
    research_topic: str = Field(
        description="The topic to research on arXiv. Should be a single topic, described in high detail (at least a paragraph).",
    )
    discovery: bool = Field(
        default=False,
        description="Set True for DISCOVERY mode (broad, exploratory sweep for leads and novel angles). False (default) is focused deep-dive research.",
    )
    max_total_reads: Optional[int] = Field(
        default=None,
        description="Optional override for this sub-agent's total read budget (default 25). Set higher to read more deeply, lower for a narrow check.",
    )


@tool
class ResearchSEC(BaseModel):
    """Delegate to the SEC EDGAR sub-agent for company filings, financials, and insider transactions."""
    research_topic: str = Field(
        description="The topic to research on SEC filings. Should be a single topic, described in high detail (at least a paragraph).",
    )
    discovery: bool = Field(
        default=False,
        description="Set True for DISCOVERY mode (broad, exploratory sweep for leads and novel angles). False (default) is focused deep-dive research.",
    )
    max_total_reads: Optional[int] = Field(
        default=None,
        description="Optional override for this sub-agent's total read budget (default 25). Set higher to read more deeply, lower for a narrow check.",
    )


# ── Name → schema map (single source of truth) ────────────────────────
# Keys must match AGENT_REGISTRY keys in deep_research/agents/__init__.py.
# The supervisor builds its tool list from ENABLED_AGENTS (config.py) using
# this map, so adding a sub-agent only needs a new entry here + in the registry.

AGENT_TOOL_SCHEMAS = {
    "ResearchWeb": ResearchWeb,
    "ResearchGeneral": ResearchGeneral,
    "ResearchReddit": ResearchReddit,
    "ResearchSubstack": ResearchSubstack,
    "ResearchPubMed": ResearchPubMed,
    "ResearchArxiv": ResearchArxiv,
    "ResearchSEC": ResearchSEC,
}

# ── Name → description map (for the supervisor prompt's <Available Tools>) ──
# Keys must match AGENT_REGISTRY / AGENT_TOOL_SCHEMAS keys. Only agents present
# in config.ENABLED_AGENTS are described to the supervisor, so the prompt never
# advertises a tool the supervisor cannot call.

AGENT_DESCRIPTIONS = {
    "ResearchWeb": "Web search sub-agent (Tavily, Exa, Google + fetch_urls). Best for most general research topics.",
    "ResearchGeneral": "Cross-platform sub-agent with ALL tools (web + Reddit + Substack + PubMed + arXiv + SEC). Best when you don't know which platform to target — the sub-agent decides.",
    "ResearchReddit": "Reddit-only sub-agent for community sentiment, forum discussions, and grassroots takes.",
    "ResearchSubstack": "Substack-only sub-agent for expert newsletter analysis and long-form opinion pieces.",
    "ResearchPubMed": "PubMed-only sub-agent for biomedical literature (drug efficacy, clinical trials, medical evidence).",
    "ResearchArxiv": "arXiv-only sub-agent for academic preprints (CS, physics, math, bio).",
    "ResearchSEC": "SEC EDGAR sub-agent for company filings, financials, insider transactions, and regulatory documents.",
}
