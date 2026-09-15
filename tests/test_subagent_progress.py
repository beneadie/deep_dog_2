"""Console presentation and per-task completion metrics; no provider calls."""

import asyncio
from copy import deepcopy
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from conftest import FakeModelFactory, api_config
from deep_research.agents import base
from deep_research.console_logger import log_sub_agent_start
from deep_research.events import EventCollector
from deep_research.models import Credentials
from deep_research.multi_agent_supervisor import _run_subagent_with_progress
from deep_research.observer import Observer
from deep_research.runtime import RuntimeContext, runtime_scope


def context(collector=None):
    cfg = api_config()
    return RuntimeContext(
        run_id="progress", thread_id="progress", config=cfg,
        credentials=Credentials(), models=FakeModelFactory(cfg),
        observer=Observer("progress", cfg, event_sink=collector),
    )


def test_search_query_from_args_handles_known_keys():
    assert base._search_query_from_args({"query": "DRAM prices"}) == "DRAM prices"
    assert base._search_query_from_args({"search_term": "HBM"}) == "HBM"
    assert base._search_query_from_args({"subreddit": "stocks"}) == "stocks"
    assert base._search_query_from_args(None) == ""
    assert base._search_query_from_args({"limit": 5}) == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("iterations", [0, 5])
async def test_banner_is_zero_based_without_changing_prompt_or_state(iterations, capsys):
    messages = [HumanMessage(content="test topic")]
    messages += [AIMessage(content="", tool_calls=[{
        "id": f"search-{i}", "name": "exa_deep_search", "args": {"query": "test"},
    }]) for i in range(iterations)]
    state = {
        "agent_type": "web", "console_agent_id": 7,
        "researcher_messages": messages, "max_iterations": 5,
        "output_mode": "sources", "target_language": "English",
        "articles_read": {"https://example.com": {"content": "evidence"}},
    }
    before = deepcopy(state)
    with runtime_scope(context()):
        await base.llm_call(state)
    output = capsys.readouterr().out
    assert f"web agent #7  |  iter {iterations}/5" in output
    assert "Call batch_save_selected NOW" not in output
    # The model's original one-based instructions and actual caps stay intact.
    assert f"Iteration: {iterations + 1}/5" in base._compute_turn_plan(state)["status_text"]
    assert state == before


@pytest.mark.asyncio
async def test_real_agent_output_exports_actual_counts():
    reads = {f"https://example.com/{i}": {"content": "evidence"} for i in range(12)}
    saved = {url: {"url": url, "title": "Evidence", "reason": "Relevant"}
             for url in list(reads)[:10]}
    with runtime_scope(context()):
        result = await base.researcher_agent.ainvoke({
            "agent_type": "web", "research_topic": "test", "output_mode": "sources",
            "target_language": "English", "max_iterations": 5,
            "researcher_messages": [HumanMessage(content="test")],
            "search_count": 3, "articles_read": reads, "saved_articles": saved,
        })
    assert result["search_count"] == 3
    assert result["read_count"] == 12
    assert len(result["saved_articles"]) == 10


@pytest.mark.asyncio
async def test_completion_prints_before_slower_sibling_and_keeps_result_order(capsys):
    collector = EventCollector()
    ctx = context(collector)
    release_slow = asyncio.Event()
    slow_result = {"search_count": 3, "read_count": 12, "saved_articles": {str(i): {} for i in range(10)}}
    fast_result = {"search_count": 2, "read_count": 4, "saved_articles": {"one": {}}}

    async def slow(_):
        await release_slow.wait()
        return slow_result

    async def fast(_):
        return fast_result

    with runtime_scope(ctx):
        slow_id = log_sub_agent_start("Slow task")
        fast_id = log_sub_agent_start("Fast task")
        capsys.readouterr()
        slow_task = asyncio.create_task(_run_subagent_with_progress(
            SimpleNamespace(ainvoke=slow), {}, timeout=5,
            agent_id=slow_id, tool_name="ResearchWeb", iteration=1))
        fast_task = asyncio.create_task(_run_subagent_with_progress(
            SimpleNamespace(ainvoke=fast), {"discovery": True}, timeout=5,
            agent_id=fast_id, tool_name="ResearchWeb", iteration=1))
        batch = asyncio.gather(slow_task, fast_task, return_exceptions=True)
        try:
            await asyncio.wait_for(fast_task, timeout=2)
            assert not slow_task.done()
            output = capsys.readouterr().out
            assert f"Discovery agent #{fast_id} complete" in output
            assert "2 searches | 4 reads | 1 selected" in output
            assert f"Sub-agent #{slow_id} complete" not in output
            assert fast_id not in ctx.observer._sub_agent_start_times
            assert slow_id in ctx.observer._sub_agent_start_times
        finally:
            release_slow.set()
            results = await batch
        assert results == [slow_result, fast_result]
        assert "3 searches | 12 reads | 10 selected" in capsys.readouterr().out
        await ctx.observer.flush_events()

    completed = [e for e in collector.events if e.type == "subagent_completed"]
    assert [e.payload["agent_id"] for e in completed] == [fast_id, slow_id]
    assert completed[0].payload["search_count"] == 2
    assert completed[1].payload["read_count"] == 12
    assert completed[1].payload["selected_count"] == 10


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [ValueError("failure"), asyncio.TimeoutError(), asyncio.CancelledError()])
async def test_failed_agent_never_prints_success_or_fabricated_counts(error, capsys):
    collector = EventCollector()
    ctx = context(collector)

    async def fail(_):
        raise error

    with runtime_scope(ctx):
        agent_id = log_sub_agent_start("Failure test")
        capsys.readouterr()
        with pytest.raises(type(error)):
            await _run_subagent_with_progress(
                SimpleNamespace(ainvoke=fail), {}, timeout=5,
                agent_id=agent_id, tool_name="ResearchWeb", iteration=1)
        await ctx.observer.flush_events()
    output = capsys.readouterr().out
    assert "complete" not in output
    assert "0 searches" not in output
    assert [e.type for e in collector.events] == ["subagent_failed"]


@pytest.mark.asyncio
async def test_source_events_remain_available_without_console_spam(capsys):
    collector = EventCollector()
    ctx = context(collector)
    for i in range(10):
        ctx.observer.emit("source_saved", source_id=i)
    assert capsys.readouterr().out == ""
    await ctx.observer.flush_events()
    assert len(collector.events) == 10
