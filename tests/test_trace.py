"""Structured trace logging: identifiers, toggle, redaction, truncation, streaming.

Covers:
- TraceRecord event_id (full UUID) + monotonic seq
- logging_enabled master toggle
- log_truncation config (default = no truncation)
- secret redaction from record content
- streaming to a trace_sink + in-memory retention on the Observer
- end-to-end run_research producing trace records and result.logs
- LOGGING_ENABLED / LOG_TRUNCATION env parsing (subprocess: import time)
"""

import asyncio
import json
import os
import subprocess
import sys
from contextlib import suppress
from pathlib import Path

import pytest

import deep_research.integration as integration
from deep_research.models import Credentials
from deep_research.observer import Observer
from deep_research.run_config import RunConfig
from deep_research.trace import TraceCollector, TraceRecord
from conftest import FakeLLM, FakeModelFactory, api_config

ROOT = Path(__file__).resolve().parents[1]
SECRET = "sk-super-secret-xyz-123456"


def _observer(collector=None, credentials=None, **cfg_overrides) -> Observer:
    return Observer(
        run_id="run-trace",
        config=RunConfig(**cfg_overrides),
        trace_sink=collector,
        credentials=credentials,
    )


# ── Record identity ─────────────────────────────────────────────────────

def test_emit_trace_assigns_full_uuid_and_sequence():
    obs = _observer()
    a = obs.emit_trace("kind_a", phase="p", agent="x")
    b = obs.emit_trace("kind_b", phase="p", agent="x")

    assert a is not None and b is not None
    assert a.event_id != b.event_id
    assert len(a.event_id) == 32  # uuid4 hex
    assert (a.seq, b.seq) == (1, 2)
    assert a.run_id == "run-trace"
    assert a.kind == "kind_a"


def test_logging_disabled_emits_nothing():
    obs = _observer(logging_enabled=False)
    assert obs.emit_trace("nope", value="x") is None
    assert obs.get_trace_records() == []


# ── Redaction ───────────────────────────────────────────────────────────

def test_secret_values_are_redacted():
    creds = Credentials({"DEEPSEEK_API_KEY": SECRET})
    obs = _observer(credentials=creds)
    rec = obs.emit_trace("prompt", system_prompt=f"use key {SECRET} now",
                         nested={"k": [f"prefix-{SECRET}"]})

    blob = json.dumps(rec.to_dict())
    assert SECRET not in blob
    assert "[REDACTED]" in blob
    assert "prefix-[REDACTED]" in rec.content["nested"]["k"][0]


def test_short_values_are_not_redacted():
    # Values under 6 chars are ignored so ordinary text is not mangled.
    obs = _observer(credentials=Credentials({"X": "abc"}))
    rec = obs.emit_trace("prompt", text="abcabc")
    assert rec.content["text"] == "abcabc"


# ── Truncation ──────────────────────────────────────────────────────────

def test_no_truncation_by_default():
    obs = _observer()
    long_text = "x" * 50_000
    rec = obs.emit_trace("prompt", text=long_text)
    assert rec.content["text"] == long_text


def test_log_truncation_applied_when_configured():
    obs = _observer(log_truncation=10)
    rec = obs.emit_trace("prompt", text="y" * 100)
    assert rec.content["text"].startswith("y" * 10)
    assert "truncated" in rec.content["text"]


# ── Streaming + retention ───────────────────────────────────────────────

def test_trace_records_stream_to_sink_and_are_retained():
    collector = TraceCollector()
    obs = _observer(collector)
    obs.emit_trace("a", value=1)
    obs.emit_trace("b", value=2)
    assert collector.records == []          # buffered until flush
    asyncio.run(obs.flush_events())
    assert [r.kind for r in collector.records] == ["a", "b"]

    retained = obs.get_trace_records()
    assert [r["kind"] for r in retained] == ["a", "b"]
    assert all(r["event_id"] for r in retained)


# ── Periodic flushing (nested-node liveness) ────────────────────────────

def test_periodic_flusher_delivers_without_explicit_flush():
    """Records must reach the sink while the graph is inside a blocking node."""
    collector = TraceCollector()
    obs = _observer(collector)

    async def run():
        task = asyncio.create_task(integration._periodic_flush(obs, interval=0.01))
        obs.emit_trace("mid_run", value=1)
        await asyncio.sleep(0.06)          # no explicit flush_events() call
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

    asyncio.run(run())
    assert [r.kind for r in collector.records] == ["mid_run"]


def test_concurrent_flushes_deliver_each_record_exactly_once():
    collector = TraceCollector()
    obs = _observer(collector)
    for i in range(100):
        obs.emit_trace("r", i=i)

    async def run():
        await asyncio.gather(obs.flush_events(), obs.flush_events(), obs.flush_events())

    asyncio.run(run())
    assert len(collector.records) == 100
    assert sorted(r.content["i"] for r in collector.records) == list(range(100))


# ── End-to-end ──────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _fake_models(monkeypatch):
    monkeypatch.setattr(integration, "ModelFactory", FakeModelFactory)


def test_run_research_emits_trace_records():
    collector = TraceCollector()
    creds = Credentials({"DEEPSEEK_API_KEY": SECRET})

    async def run():
        return await integration.run_research(
            "What is the latest on LLM evaluation?",
            config=api_config(),
            runtime=integration.RuntimeOptions(
                run_id="run-e2e", thread_id="t-e2e",
                event_sink=None, trace_sink=collector),
            credentials=creds,
        )

    result = asyncio.run(run())

    kinds = [r.kind for r in collector.records]
    for expected in ("run_started", "research_brief", "draft_report",
                     "supervisor_turn", "run_completed"):
        assert expected in kinds, f"missing trace kind: {expected}"

    assert result.logs, "result.logs must carry the trace records"
    assert all(r["run_id"] == "run-e2e" for r in result.logs)
    assert all(r["event_id"] for r in result.logs)

    # Full content is captured (no truncation by default).
    brief = next(r for r in collector.records if r.kind == "research_brief")
    assert brief.content["research_brief"]
    final = next(r for r in collector.records if r.kind == "run_completed")
    assert final.content["final_report"]

    # Secrets never leak into the trace stream or the result.
    assert SECRET not in json.dumps([r.to_dict() for r in collector.records], default=str)
    assert SECRET not in json.dumps(result.to_dict(), default=str)


def test_subagent_capture_points(monkeypatch):
    """A delegated sub-agent emits its full prompt, response and findings."""
    from langchain_core.messages import AIMessage

    class ScriptedSupervisor(FakeLLM):
        def __init__(self):
            super().__init__("supervisor")
            self.n = 0

        def bind_tools(self, tools):
            return self

        async def ainvoke(self, messages):
            self.n += 1
            if self.n == 1:
                return AIMessage(content="delegate", tool_calls=[{
                    "id": "tc-web-1", "name": "ResearchWeb",
                    "args": {"research_topic": "quaternion basics", "discovery": False}}])
            return AIMessage(content="done", tool_calls=[{
                "id": "tc-done", "name": "ResearchComplete", "args": {}}])

    class ScriptedFactory(FakeModelFactory):
        def __init__(self, config, credentials=None):
            super().__init__(config, credentials)
            self._sup = ScriptedSupervisor()

        def supervisor(self, tools=None, *, max_tokens=None, temperature=0.0,
                       chain=None, reasoning_effort=None):
            return self._sup if max_tokens != 40000 else FakeLLM("writer")

    monkeypatch.setattr(integration, "ModelFactory", ScriptedFactory)
    collector = TraceCollector()

    async def run():
        return await integration.run_research(
            "What is a quaternion?",
            config=api_config(),
            runtime=integration.RuntimeOptions(
                run_id="run-sub", trace_sink=collector, console_enabled=False),
        )

    asyncio.run(run())
    kinds = set(collector.kinds())
    for expected in ("delegation", "subagent_prompt", "subagent_response",
                     "subagent_findings"):
        assert expected in kinds, f"missing trace kind: {expected}"

    prompt = next(r for r in collector.records if r.kind == "subagent_prompt")
    assert prompt.content["system_prompt"]
    assert prompt.content["messages"]
    assert prompt.content["tools"]


def test_run_research_trace_disabled_by_config():
    collector = TraceCollector()

    async def run():
        return await integration.run_research(
            "test prompt",
            config=api_config(logging_enabled=False),
            runtime=integration.RuntimeOptions(trace_sink=collector),
        )

    result = asyncio.run(run())
    assert collector.records == []
    assert result.logs == []


# ── Env parsing (import-time) ───────────────────────────────────────────

def test_logging_env_vars_parsed():
    env = {k: v for k, v in os.environ.items()
           if k not in ("LOGGING_ENABLED", "LOG_TRUNCATION")}
    env.update({"LOGGING_ENABLED": "false", "LOG_TRUNCATION": "123"})
    code = ("from deep_research.run_config import RunConfig; "
            "c = RunConfig(); print(c.logging_enabled); print(c.log_truncation)")
    r = subprocess.run([sys.executable, "-c", code], env=env, cwd=ROOT,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["False", "123"]
