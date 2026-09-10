"""Shared fakes for the Deep Dog test suite.

No test in this suite calls a real LLM/search provider. Model construction and
invocation are faked so individual stages and graph wiring can be exercised
cheaply and deterministically.
"""

from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.messages import AIMessage

from deep_research.models import Credentials
from deep_research.run_config import RunConfig

FINAL_TEXT = "# Final Report\n\nA verified body of findings with citations preserved in code."
DRAFT_TEXT = "DRAFT REPORT TEXT"


def _fill_schema(schema) -> Any:
    """Instantiate a pydantic structured-output schema with canned values."""
    vals: dict[str, Any] = {}
    for name, field_ in getattr(schema, "model_fields", {}).items():
        if not field_.is_required():
            continue
        ann = str(field_.annotation)
        if name == "research_brief":
            vals[name] = "RESEARCH BRIEF: analyse the test question."
        elif name in ("input_language", "target_language"):
            vals[name] = "English"
        elif "bool" in ann:
            vals[name] = False
        elif "int" in ann:
            vals[name] = 0
        elif "float" in ann:
            vals[name] = 0.0
        else:
            vals[name] = "x"
    return schema(**vals)


class FakeLLM:
    """A scripted fake chat model that never touches the network."""

    def __init__(self, kind: str):
        self.kind = kind
        self.calls = 0
        self._schema = None

    def bind_tools(self, tools):
        return self

    def with_fallbacks(self, fallbacks, **kwargs):
        return self

    def with_structured_output(self, schema, **kwargs):
        # Mirrors real models: returns a NEW bound runnable, never mutating the
        # cached base model (the cached draft model is reused by other nodes).
        bound = FakeLLM(self.kind)
        bound._schema = schema
        return bound

    async def ainvoke(self, messages):
        self.calls += 1
        return self._respond(messages)

    def invoke(self, messages):
        self.calls += 1
        if self._schema is not None:
            return _fill_schema(self._schema)
        return AIMessage(content="ok")

    def _respond(self, messages):
        if self._schema is not None:
            return _fill_schema(self._schema)
        if self.kind == "supervisor":
            return AIMessage(
                content="supervisor decision: research complete",
                tool_calls=[{"id": "tc-research-complete", "name": "ResearchComplete", "args": {}}],
            )
        if self.kind == "writer":
            return AIMessage(content=FINAL_TEXT)
        if self.kind == "draft":
            return AIMessage(content=DRAFT_TEXT)
        return AIMessage(content="subagent deliverable")


class FakeModelFactory:
    """Drop-in replacement for ModelFactory that returns FakeLLM instances."""

    def __init__(self, config: RunConfig, credentials: Credentials | None = None):
        self.config = config
        self.credentials = credentials or Credentials()

    def supervisor(self, tools=None, *, max_tokens=None, temperature=0.0, chain=None,
                   reasoning_effort=None):
        kind = "writer" if max_tokens == 40000 else "supervisor"
        return FakeLLM(kind)

    def draft_report(self, *, max_tokens=None, temperature=0.0, chain=None, reasoning_effort=None):
        return FakeLLM("draft")

    def subagent(self, tools=None, *, max_tokens=None, temperature=0.0, chain=None,
                 reasoning_effort=None):
        return FakeLLM("subagent")

    def get(self, model_name, *, temperature=0.0, max_tokens=None, reasoning_effort=None,
            route_via_openrouter=None):
        return FakeLLM("subagent")


def api_config(**overrides) -> RunConfig:
    """A run config safe for API-style tests (no file output, fast)."""
    base = dict(
        output_mode="none",
        log_mode="none",
        save_report_to_file=False,
        save_subagent_reports_to_file=False,
        enable_source_log=False,
        enable_subtopic_generation=False,
        enable_research_trace=False,
        disable_model_fallback=True,
    )
    base.update(overrides)
    return RunConfig(**base).finalize()
