"""Offline integration tests using fake models.

These exercise the real graph wiring and the run_research driver (events,
cancellation, deadline, checkpointer, file behaviour) end-to-end with NO LLM
or provider calls.
"""

import asyncio
import json
from pathlib import Path

import pytest

import deep_research.integration as integration
from deep_research.cancellation import CancellationToken
from deep_research.events import EventCollector
from deep_research.models import Credentials
from conftest import FakeModelFactory, api_config


@pytest.fixture(autouse=True)
def _fake_models(monkeypatch):
    """Replace the real ModelFactory with offline fakes for every integration test."""
    monkeypatch.setattr(integration, "ModelFactory", FakeModelFactory)


def test_completed_run_emits_ordered_events_and_no_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    collector = EventCollector()
    creds = Credentials({"DEEPSEEK_API_KEY": "sk-super-secret-xyz"})

    async def run():
        result = await integration.run_research(
            "What is the latest on LLM evaluation?",
            config=api_config(),
            runtime=integration.RuntimeOptions(
                run_id="run-a", thread_id="thread-a", event_sink=collector),
            credentials=creds,
        )
        return result

    result = asyncio.run(run())

    assert result.status == "completed"
    assert result.final_report
    assert result.research_brief
    assert result.run_id == "run-a"
    assert result.thread_id == "thread-a"

    types = collector.types()
    assert types[0] == "run_started"
    assert types[-1] == "run_completed"
    for expected in ("scope_started", "draft_started", "supervisor_iteration",
                     "report_started", "citations_validated"):
        assert expected in types

    # API mode must not write files (no report md, no sources json, no logs).
    assert list(tmp_path.iterdir()) == []

    # No secret ever appears in serialized results or events.
    blob = json.dumps(result.to_dict(), default=str)
    assert "sk-super-secret-xyz" not in blob
    for e in collector.events:
        assert "sk-super-secret-xyz" not in json.dumps(e.to_dict(), default=str)
    # Config snapshot holds no credentials.
    assert "api_key" not in json.dumps(result.run_metadata, default=str)


def test_external_checkpointer_is_used():
    from langgraph.checkpoint.memory import InMemorySaver
    saver = InMemorySaver()
    collector = EventCollector()

    async def run():
        return await integration.run_research(
            "test prompt",
            config=api_config(),
            runtime=integration.RuntimeOptions(
                thread_id="thread-cp", checkpointer=saver, event_sink=collector),
        )

    result = asyncio.run(run())
    assert result.status == "completed"
    checkpoints = list(saver.list({"configurable": {"thread_id": "thread-cp"}}))
    assert len(checkpoints) >= 1


def test_cancellation_between_iterations():
    token = CancellationToken()
    token.cancel("api stop")
    collector = EventCollector()

    async def run():
        return await integration.run_research(
            "test prompt",
            config=api_config(),
            runtime=integration.RuntimeOptions(
                cancellation=token, event_sink=collector),
        )

    result = asyncio.run(run())
    assert result.status in ("cancelled", "partial")
    assert "run_cancelled" in collector.types()


def test_deadline_enforcement():
    collector = EventCollector()

    async def run():
        return await integration.run_research(
            "test prompt",
            config=api_config(max_duration_minutes=30.0),
            runtime=integration.RuntimeOptions(
                deadline_override_seconds=0.0, event_sink=collector),
        )

    result = asyncio.run(run())
    assert result.status == "timed_out"
    assert "run_timed_out" in collector.types()
    assert result.failure


def test_cli_mode_still_writes_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    collector = EventCollector()
    out = tmp_path / "outputs"
    cfg = api_config(output_mode="file", log_mode="file", save_report_to_file=True,
                     enable_source_log=True)

    async def run():
        return await integration.run_research(
            "test prompt",
            config=cfg,
            runtime=integration.RuntimeOptions(
                output_dir=out, event_sink=collector, console_enabled=False),
        )

    result = asyncio.run(run())
    assert result.status == "completed"
    finals = list(tmp_path.glob("Final_Report_*.md"))
    assert finals, "CLI (file) mode must still write the final report file"
    run_folders = list((out).glob("research_*"))
    assert run_folders, "CLI log folder should be created"
