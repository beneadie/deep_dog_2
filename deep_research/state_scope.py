
"""State Definitions and Pydantic Schemas for Research Scoping.

This defines the state objects and structured schemas used for
the research agent scoping workflow, including researcher state management and output schemas.
"""

import operator
from typing_extensions import Optional, Annotated, List, Sequence, Dict

from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

# ===== STATE DEFINITIONS =====

class AgentInputState(MessagesState):
    """Input state for the full agent - only contains messages from user input."""
    start_time: float = 0.0
    target_language: Optional[str] = None
    # Serialized RunConfig snapshot, persisted in the checkpoint so a resumed
    # run can reconstruct its runtime configuration (no clients/secrets here).
    config_snapshot: Optional[dict] = None

class AgentState(MessagesState):
    """
    Main state for the full multi-agent research system.

    Extends MessagesState with additional fields for research coordination.
    Note: Some fields are duplicated across different state classes for proper
    state management between subgraphs and the main workflow.
    """

    # Research brief generated from user conversation history
    research_brief: Optional[str]
    # Messages exchanged with the supervisor agent for coordination
    supervisor_messages: Annotated[Sequence[BaseMessage], add_messages]
    # Processed and structured notes ready for report generation
    notes: Annotated[list[str], operator.add] = []
    # Curated sources: full-text sources selected by sub-agents with quality reasons
    curated_sources: Annotated[list[dict], operator.add] = []
    # Source registries emitted by sub-agents (title+URL only) for citation resolution
    source_registry: Annotated[list[dict], operator.add] = []
    # Draft research report
    draft_report: str
    # Final formatted research report
    final_report: str
    # Start time of the research process
    start_time: float = 0.0
    # Canonical target language for all generated outputs (e.g., "English", "Chinese")
    target_language: Optional[str] = None
    # Language the user's message is literally written in (model-detected;
    # target_language is derived from it so the two can never diverge)
    input_language: Optional[str] = None
    # Set to True when circuit breaker aborts research (e.g., API quota exhausted)
    aborted: bool = False
    # Human-readable reason for an abort (surfaces a truthful failure message)
    abort_reason: str = ""
    # Consecutive supervisor-level failed iterations (retry-once each; >=2 stops)
    consecutive_failures: int = 0
    # Serialized RunConfig snapshot (persisted through the checkpointer)
    config_snapshot: Optional[dict] = None
    # Secondary subtopic reports: list of dicts {"title": str, "content": str}
    secondary_reports: Annotated[list[dict], operator.add] = []
    # Pending subtopic briefs from evaluation (used to pass to generation node)
    pending_subtopic_briefs: list[dict] = []

# ===== STRUCTURED OUTPUT SCHEMAS =====

class ClarifyWithUser(BaseModel):
    """Schema for user clarification decision and questions."""

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
    """Schema for structured research brief generation."""

    research_brief: str = Field(
        description="A research question that will be used to guide the research.",
    )
    input_language: Optional[str] = Field(
        default=None,
        description="The language the user's message is LITERALLY written in, determined only by reading the actual script of <Messages> (e.g., 'English', '中文').",
    )
    target_language: str = Field(
        description="The language that the report and the research brief should be written in. MUST be identical to input_language: write the report in the same language as the user's question.",
    )

class DraftReport(BaseModel):
    """Schema for structured draft report generation."""

    draft_report: str = Field(
        description="A draft report that will be used to guide the research.",
    )

# ===== SUBTOPIC EVALUATION TOOLS =====

@tool
class GenerateSubtopicReport(BaseModel):
    """Tool for requesting a subtopic report to be generated.

    Call this for each distinct topic that warrants a detailed supplementary report.
    You can call this multiple times for different topics.
    """
    title: str = Field(
        description="Title for the Subtopic Report (e.g., 'Detailed Analysis of Stock XYZ').",
    )
    generation_brief: str = Field(
        description="Instructions for generating this report. Describe what information to extract from the research notes.",
    )

@tool
class EndSubtopicEvaluation(BaseModel):
    """Tool for indicating that subtopic evaluation is complete.

    Call this when you have finished evaluating and either:
    - Have already called GenerateSubtopicReport for all needed topics, OR
    - Determined that no subtopic reports are necessary
    """
    pass

