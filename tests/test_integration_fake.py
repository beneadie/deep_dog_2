"""Offline integration tests using fake models.

These exercise the real graph wiring and the run_research driver (events,
cancellation, research timing, checkpointer, file behaviour) end-to-end with NO LLM
or provider calls.
"""

import asyncio
import json
import time
from pathlib import Path

import pytest

import deep_research.integration as integration
from deep_research.cancellation import CancellationToken
from deep_research.events import EventCollector
from deep_research.models import Credentials
from conftest import FINAL_TEXT, FakeLLM, FakeModelFactory, api_config


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


def test_minimal_api_prints_progress_and_returns_text_without_files(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    result = asyncio.run(integration.run_research("test prompt"))
    assert result.status == "completed", result.failure
    assert result.final_report
    assert list(tmp_path.iterdir()) == []
    output = capsys.readouterr().out
    assert "Research started" in output
    assert "Creating research brief" in output
    assert "Creating initial draft" in output
    assert "Writing final report" in output
    assert "Research completed" in output


def test_output_mode_none_prevents_automatic_report_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = asyncio.run(integration.run_research(
        "test prompt", config=api_config(output_mode="none", save_report_to_file=True)))
    assert result.status == "completed", result.failure
    assert result.final_report
    assert list(tmp_path.iterdir()) == []


def test_supervisor_continues_after_thinking_without_refinement(monkeypatch):
    import deep_research.multi_agent_supervisor as supervisor
    from langchain_core.messages import AIMessage

    calls = []

    class ScriptedSupervisor:
        async def ainvoke(self, messages):
            calls.append(messages)
            if len(calls) == 1:
                return AIMessage(content="Plan research", tool_calls=[{
                    "id": "think-first", "name": "think_tool",
                    "args": {"reflection": "Assess the research question."},
                }])
            return AIMessage(content="Done", tool_calls=[{
                "id": "complete", "name": "ResearchComplete", "args": {},
            }])

    monkeypatch.setattr(supervisor, "_supervisor_model", ScriptedSupervisor)
    result = asyncio.run(integration.run_research("test prompt", config=api_config()))
    assert result.status == "completed", result.failure
    assert len(calls) >= 2
    assert any(getattr(message, "tool_call_id", None) == "think-first"
               for message in calls[1])
    assert result.final_report


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


@pytest.mark.parametrize("phase", ["draft", "writer"])
def test_elapsed_research_window_does_not_cancel_pending_work(monkeypatch, phase):
    """Cross the former run deadline during a model call, then finish normally."""
    from deep_research.runtime import RuntimeContext, get_runtime

    collector = EventCollector()

    async def run():
        started = asyncio.Event()
        release = asyncio.Event()

        class WaitingLLM(FakeLLM):
            async def ainvoke(self, messages):
                started.set()
                await release.wait()
                assert get_runtime().remaining_minutes() == 0
                return await super().ainvoke(messages)

        class Factory(FakeModelFactory):
            def supervisor(self, *args, **kwargs):
                model = super().supervisor(*args, **kwargs)
                return WaitingLLM(model.kind) if model.kind == phase else model

            def draft_report(self, *args, **kwargs):
                model = super().draft_report(*args, **kwargs)
                return WaitingLLM(model.kind) if model.kind == phase else model

        def near_end_context(**kwargs):
            # Represent time already spent in the run, with 20ms left before
            # the old whole-run cutoff. Do not alter asyncio's actual clock.
            kwargs["start_monotonic"] = (
                time.monotonic() - kwargs["config"].strict_timeout_minutes * 60 + 0.02
            )
            return RuntimeContext(**kwargs)

        monkeypatch.setattr(integration, "ModelFactory", Factory)
        monkeypatch.setattr(integration, "RuntimeContext", near_end_context)
        task = asyncio.create_task(integration.run_research(
            "test prompt", config=api_config(),
            runtime=integration.RuntimeOptions(event_sink=collector),
        ))
        try:
            await asyncio.wait_for(started.wait(), timeout=5)
            await asyncio.sleep(0.05)
            assert not task.done(), "Research time must not cancel pending model work"
            release.set()
            return await asyncio.wait_for(task, timeout=5)
        finally:
            release.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    result = asyncio.run(run())
    assert result.status == "completed", result.failure
    assert result.final_report == FINAL_TEXT
    assert result.final_report != result.draft_report
    assert result.failure is None
    assert "citations_validated" in collector.types()
    assert "run_timed_out" not in collector.types()
    assert collector.types()[-1] == "run_completed"


@pytest.mark.parametrize("elapsed_minutes", [11, 30])
def test_research_time_routes_to_final_writer_like_original(monkeypatch, elapsed_minutes):
    import deep_research.multi_agent_supervisor as supervisor
    from langchain_core.messages import AIMessage
    from types import SimpleNamespace
    from deep_research.runtime import RuntimeContext, runtime_scope

    cfg = api_config(research_time_max_minutes=10)
    runtime = RuntimeContext(
        run_id="timing", thread_id="timing", config=cfg,
        models=FakeModelFactory(cfg), console_enabled=False,
    )
    monkeypatch.setattr(supervisor, "time", SimpleNamespace(
        time=lambda: 1000 + elapsed_minutes * 60,
    ))
    state = {
        "start_time": 1000,
        "research_iterations": 1,
        "supervisor_messages": [AIMessage(content="More research", tool_calls=[{
            "id": "extra-research", "name": "ResearchWeb",
            "args": {"research_topic": "An extra research task"},
        }])],
    }
    with runtime_scope(runtime):
        command = asyncio.run(supervisor.supervisor_tools(state))
    # Even a request for more research must route to writing after max + 1.
    assert command.goto == "write_final_report"
    assert not command.update.get("aborted")


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
