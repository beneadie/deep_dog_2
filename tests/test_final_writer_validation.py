"""Exercise final-writing recovery through the real graph without provider calls."""

import asyncio

import pytest
from langchain_core.messages import AIMessage

import deep_research.integration as integration
from conftest import DRAFT_TEXT, FINAL_TEXT, FakeModelFactory, api_config


@pytest.mark.parametrize("bad_output", [
    "", "   ", DRAFT_TEXT, f"\n{DRAFT_TEXT}\n",
    "<CitationPlanList>Report body</CitationPlanList>",
    DRAFT_TEXT + "\n\n## Sources\n[1] Example",
])
@pytest.mark.parametrize("recovers", [True, False])
def test_invalid_final_report_retries_once(monkeypatch, bad_output, recovers):
    calls = []

    class Writer:
        async def ainvoke(self, messages):
            calls.append(messages)
            return AIMessage(content=FINAL_TEXT if recovers and len(calls) == 2 else bad_output)

    class Models(FakeModelFactory):
        def supervisor(self, tools=None, **kwargs):
            if kwargs.get("max_tokens") == 40000:
                return Writer()
            return super().supervisor(tools=tools, **kwargs)

    monkeypatch.setattr(integration, "ModelFactory", Models)
    result = asyncio.run(integration.run_research(
        "Test question", config=api_config(),
        runtime=integration.RuntimeOptions(console_enabled=False),
    ))
    assert len(calls) == 2
    assert result.draft_report == DRAFT_TEXT
    if recovers:
        assert result.status == "completed"
        assert result.final_report == FINAL_TEXT
        assert result.failure is None
    else:
        assert result.status == "partial"
        assert result.final_report == ""
        assert "Final report generation failed after two attempts" in result.failure


def test_valid_final_report_needs_no_retry(monkeypatch):
    calls = []

    class Models(FakeModelFactory):
        def supervisor(self, tools=None, **kwargs):
            model = super().supervisor(tools=tools, **kwargs)
            if kwargs.get("max_tokens") == 40000:
                original = model.ainvoke

                async def invoke(messages):
                    calls.append(messages)
                    return await original(messages)

                model.ainvoke = invoke
            return model

    monkeypatch.setattr(integration, "ModelFactory", Models)
    result = asyncio.run(integration.run_research(
        "Test question", config=api_config(),
        runtime=integration.RuntimeOptions(console_enabled=False),
    ))
    assert result.status == "completed"
    assert result.final_report == FINAL_TEXT
    assert len(calls) == 1
