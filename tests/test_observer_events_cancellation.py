"""Observer, events, cancellation, deadline, and concurrency-isolation tests.

All tests are offline: no LLM or search provider is called.
"""

import asyncio

import pytest

from deep_research.cancellation import (
    CancellationToken,
    Deadline,
    RunCancelledError,
    as_cancellation_checker,
)
from deep_research.events import EventCollector, ResearchEvent, deliver_event
from deep_research.models import Credentials
from deep_research.observer import Observer
from deep_research.run_config import RunConfig
from deep_research.runtime import RuntimeContext, get_runtime, runtime_scope
from conftest import api_config


def _observer(run_id="r1", sink=None, **cfg_overrides):
    return Observer(run_id=run_id, config=api_config(**cfg_overrides), event_sink=sink)


def test_cancellation_token_and_checker():
    token = CancellationToken()
    checker = as_cancellation_checker(token)
    assert checker() is False
    token.cancel("user pressed stop")
    assert checker() is True
    with pytest.raises(RunCancelledError):
        token.raise_if_cancelled()

    # plain callables also work
    assert as_cancellation_checker(lambda: True)() is True


def test_deadline_expiry():
    d = Deadline.after(max_seconds=0.0)
    assert d.expired()
    assert d.remaining_seconds() == 0.0
    d2 = Deadline.after(max_seconds=60)
    assert not d2.expired()
    assert d2.remaining_seconds() > 0


def test_event_collector_and_sync_delivery():
    collector = EventCollector()
    ev = ResearchEvent(type="run_started", run_id="r1", payload={"phase": "run"})
    asyncio.run(deliver_event(collector, ev))
    assert collector.types() == ["run_started"]


def test_event_ordering_and_payloads():
    collector = EventCollector()
    obs = _observer(run_id="r1", sink=collector, enable_source_log=True)
    obs.emit("run_started", phase="run", agent="runner")
    obs.emit("source_found", phase="subagent", platform="web", count=3)
    obs.emit("source_saved", phase="subagent", platform="web",
             url="https://example.com/a", title="A")
    obs.emit("run_completed", phase="run", report_chars=100)

    asyncio.run(obs.flush_events())
    types = collector.types()
    assert types == ["run_started", "source_found", "source_saved", "run_completed"]
    saved = collector.events[2]
    assert saved.payload["url"] == "https://example.com/a"
    assert saved.run_id == "r1"
    assert all(e.timestamp > 0 for e in collector.events)


def test_observer_source_counters_isolated_per_observer():
    o1 = _observer(run_id="a", enable_source_log=True)
    o2 = _observer(run_id="b", enable_source_log=True)
    a = o1.log_source("tavily", "https://a", "content-a")
    b1 = o2.log_source("tavily", "https://b", "content-b")
    b2 = o2.log_source("exa", "https://c", "content-c")
    assert (a, b1, b2) == (1, 1, 2)
    assert set(o1.sources) == {1}
    assert set(o2.sources) == {1, 2}


def test_observer_no_files_in_api_mode(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    obs = _observer(enable_source_log=True, log_mode="none")
    obs.log_source("tavily", "https://a", "c")
    obs.log_sub_agent("topic", "sys", "deliverable")
    assert list(tmp_path.iterdir()) == []


def test_observer_writes_files_in_cli_mode(tmp_path):
    obs = _observer(enable_source_log=True, log_mode="file",
                    save_subagent_reports_to_file=True)
    obs.init_run_folder(tmp_path, "2026-01-01_00-00-00")
    obs.log_source("tavily", "https://a", "c")
    obs.log_sub_agent("topic", "sys", "deliverable", agent_type="ResearchWeb")
    assert (tmp_path / "research_2026-01-01_00-00-00" / "sources.jsonl").exists()
    assert (tmp_path / "research_2026-01-01_00-00-00" / "sub_agents" / "sub_agent_001.json").exists()


def test_two_concurrent_runtimes_isolated():
    """Two concurrent runs must not share model/config/observer state."""

    async def run_one(profile: str, chain: list, collector: EventCollector) -> dict:
        cfg = api_config(profile=profile, subagent_model_fallback_chain=chain,
                         enable_source_log=True)
        creds = Credentials({"DEEPSEEK_API_KEY": "sk-test-not-real"})
        ctx = RuntimeContext(
            run_id=profile,
            thread_id=profile,
            config=cfg,
            credentials=creds,
            models=None,
            observer=Observer(run_id=profile, config=cfg, event_sink=collector),
            console_enabled=False,
        )
        with runtime_scope(ctx):
            model = ctx.models.subagent()
            ctx.observer.emit("subagent_started", phase="subagent",
                              platform="web", model_profile=profile)
            await ctx.observer.flush_events()
            for i in range(3):
                ctx.observer.log_source("tavily", f"https://{profile}/{i}", "c")
            # Each context sees its own config while the scope is active.
            assert get_runtime().config.profile == profile
        return {"profile": profile, "model": model, "sources": len(ctx.observer.sources)}

    async def main():
        c1, c2 = EventCollector(), EventCollector()
        t1 = asyncio.create_task(run_one("muse", ["deepseek-v4-pro"], c1))
        t2 = asyncio.create_task(run_one("deepseek", ["deepseek-v4-flash"], c2))
        r1, r2 = await asyncio.gather(t1, t2)
        return r1, r2, c1, c2

    r1, r2, c1, c2 = asyncio.run(main())
    assert r1["profile"] == "muse"
    assert r2["profile"] == "deepseek"
    assert r1["model"] is not r2["model"]
    assert r1["sources"] == 3 and r2["sources"] == 3
    # Events are fully isolated per run.
    assert {e.run_id for e in c1.events} == {"muse"}
    assert {e.run_id for e in c2.events} == {"deepseek"}
    # Sources never cross streams.
    assert all("muse" in e.payload["model_profile"] for e in c1.events)
    assert all("deepseek" in e.payload["model_profile"] for e in c2.events)
