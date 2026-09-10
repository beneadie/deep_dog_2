
"""
State Definitions and Pydantic Schemas for Research Agent

This module defines the state objects and structured schemas used for
the research agent workflow, including researcher state management and output schemas.

Merged schema (CP4): ResearcherState/ResearcherOutputState now hold the union
of the production pipeline's fields (search_queries, curated_sources,
start_time, tool_call_iterations) and the platform agent engine's fields
(output/report modes, depth caps, curation dicts).

Semantics:
- agent_type now means the PLATFORM (e.g. "reddit", "pubmed", "web").
  The old production meaning ("research" / "discovery") moved to the
  `discovery` boolean flag.
- Each engine invocation gets its own private ResearcherState instance, so
  curation dicts (articles_read, saved_articles, findings_log) never
  cross-contaminate between concurrent runs.
"""

import operator
from typing_extensions import TypedDict, Annotated, List, Sequence
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

# ===== STATE DEFINITIONS =====

class ResearcherState(TypedDict):
    """
    State for the research agent containing message history and research metadata.

    This state tracks the researcher's conversation, iteration count for limiting
    tool calls, the research topic being investigated, compressed findings,
    and raw research notes for detailed analysis.
    """
    # Conversation + tool loop
    researcher_messages: Annotated[Sequence[BaseMessage], add_messages]
    tool_call_iterations: int
    research_topic: str
    compressed_research: str
    search_queries: Annotated[List[str], operator.add]
    curated_sources: Annotated[List[dict], operator.add]
    start_time: float
    target_language: str

    # Platform dispatch + model config
    agent_type: str          # platform key, e.g. "reddit" | "pubmed" | "web"
    discovery: bool = False  # discovery vs research mode (was production's agent_type meaning)

    # Optional per-invocation model-name fallback chain override for this
    # sub-agent (e.g. ["gemini-3-flash-preview"] or
    # ["deepseek-baba-singapore"]). Absent/empty → SUBAGENT_MODEL_FALLBACK_CHAIN.
    model_chain: list = None

    # True when a Chinese supervisor delegated to this (international) sub-agent;
    # appends one vague "keep it neutral" line to the sub-agent prompt.
    chinese_supervisor_international_subagent: bool = False

    # Output mode: "sources" | "report" | "sources_inline" | "report_inline"
    output_mode: str

    # Whether sources output includes full article text (forced True for report)
    include_article_text: bool

    # ── Depth / concurrency caps (from CLI, with defaults) ──
    max_iterations: int    # -m, default 10
    max_reads: int         # --max-reads, default 8, PER-ITERATION budget
    max_total_reads: int   # --max-total-reads, default 25, TOTAL run budget
    max_saves: int         # --max-saves, default 15
    max_searches: int      # --max-searches, default 8
    max_concurrency: int   # --concurrency, default 4

    # ── Curation fields (per-run, private — no reducer) ──
    articles_read: dict      # identifier -> {"content": str}
    saved_articles: dict     # identifier -> {"url", "reason", "content", "title"}
    findings_log: dict       # key -> value (cross-post observations)
    search_count: int        # total search tool calls so far
    search_results: dict     # handle ("S1") -> {"tool": str, "items": [{"id", "title", ...}]}

    # Whether the one-time final save round (curation gather mode, iteration
    # budget exhausted with read-but-unsaved items) has already been granted.
    # Persists in state so it fires exactly once.
    final_save_round: bool = False

class ResearcherOutputState(TypedDict):
    """
    Output state for the research agent containing final research results.

    This represents the final output of the research process with compressed
    research findings, curated sources, and all raw notes from the research process.
    """
    compressed_research: str
    curated_sources: Annotated[List[dict], operator.add]
    researcher_messages: Annotated[Sequence[BaseMessage], add_messages]
    search_queries: Annotated[List[str], operator.add]
    saved_articles: dict     # the curated source dict (per-agent)
    findings_log: dict       # cross-item observations
    iteration_count: int
    compressed_research: str  # the deliverable (sources list or report)
    source_registry: list[dict]  # ordered [{identifier, url, title, source_type, ref}]

# ===== STRUCTURED OUTPUT SCHEMAS =====

class ClarifyWithUser(BaseModel):
    """Schema for user clarification decisions during scoping phase."""
    need_clarification: bool = Field(
        description="Whether the user needs to be asked a clarifying question.",
    )
    question: str = Field(
        description="A question to ask the user to clarify the report scope",
    )
    verification: str = Field(
        description="Verify message that we will start research after the user has provided the necessary information.",
    )

class ResearchQuestion(BaseModel):
    """Schema for research brief generation."""
    research_brief: str = Field(
        description="A research question that will be used to guide the research.",
    )
