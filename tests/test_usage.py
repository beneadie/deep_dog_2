"""Offline token-usage accounting tests (no LLM or search provider calls)."""

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor

from langchain_core.messages import AIMessage

from deep_research.events import EventCollector, LLM_USAGE, TOOL_USAGE
from deep_research.models import Credentials
from deep_research.observer import Observer
from deep_research.run_config import RunConfig
from deep_research.runtime import RuntimeContext, runtime_scope
from deep_research.usage import (
    ToolUsageTracker,
    UsageTracker,
    extract_usage,
    format_usage_summary,
    record_llm_usage,
    record_tool_call,
)
from conftest import api_config, FakeModelFactory
from deep_research.research_agent_scope import write_research_brief


def _fake_response(**kwargs):
    return AIMessage(content="ok", **kwargs)


def test_extract_usage_langchain_shape_with_cache_details():
    response = _fake_response(
        usage_metadata={
            "input_tokens": 1000,
            "output_tokens": 200,
            "total_tokens": 1200,
            "input_token_details": {"cache_read": 800, "cache_creation": 50},
        },
        response_metadata={"model_name": "deepseek-flash"},
    )
    usage = extract_usage(response)
    assert usage["input_tokens"] == 1000
    assert usage["output_tokens"] == 200
    assert usage["cache_read_tokens"] == 800
    assert usage["cache_creation_tokens"] == 50
    assert usage["total_tokens"] == 1200
    assert usage["usage_available"] is True
    assert usage["model"] == "deepseek-flash"


def test_extract_usage_openai_shape():
    response = _fake_response(
        response_metadata={
            "token_usage": {
                "prompt_tokens": 300,
                "completion_tokens": 70,
                "total_tokens": 370,
                "prompt_tokens_details": {"cached_tokens": 240},
            }
        }
    )
    usage = extract_usage(response)
    assert usage["input_tokens"] == 300
    assert usage["output_tokens"] == 70
    assert usage["cache_read_tokens"] == 240
    assert usage["total_tokens"] == 370
    assert usage["usage_available"] is True


def test_extract_usage_missing_metadata_is_zero_and_safe():
    usage = extract_usage(_fake_response())
    assert usage["usage_available"] is False
    assert usage["total_tokens"] == 0
    assert usage["model"] is None


def test_tracker_aggregates_by_role_and_is_thread_safe():
    tracker = UsageTracker()
    response = _fake_response(
        usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    )

    def record(role):
        for _ in range(20):
            tracker.record(response, role=role)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(record, ["supervisor", "subagent"]))

    snapshot = tracker.snapshot()
    assert snapshot["calls"] == 40
    assert snapshot["total"]["input_tokens"] == 400
    assert snapshot["total"]["output_tokens"] == 200
    assert snapshot["total"]["total_tokens"] == 600
    assert snapshot["total"]["calls_with_usage"] == 40
    assert set(snapshot["by_role"]) == {"supervisor", "subagent"}
    assert snapshot["by_role"]["subagent"]["calls"] == 20


def test_record_llm_usage_emits_event_and_stores_record():
    sink = EventCollector()
    config = RunConfig().finalize()
    observer = Observer(run_id="r1", config=config, event_sink=sink, console_enabled=False)
    context = RuntimeContext(
        run_id="r1", thread_id="r1", config=config,
        credentials=Credentials(), models=FakeModelFactory(config),
        observer=observer, console_enabled=False,
    )
    response = _fake_response(
        usage_metadata={"input_tokens": 7, "output_tokens": 3, "total_tokens": 10}
    )

    with runtime_scope(context):
        record_llm_usage(response, role="supervisor", agent="supervisor", iteration=2)
        snapshot = context.usage.snapshot()
        asyncio.run(observer.flush_events())

    assert snapshot["calls"] == 1
    assert snapshot["total"]["total_tokens"] == 10
    assert snapshot["by_role"]["supervisor"]["calls"] == 1
    assert len(context.usage.records()) == 1
    assert sink.types() == [LLM_USAGE]
    assert sink.events[0].payload["total_tokens"] == 10


def test_format_usage_summary_includes_roles_and_cache():
    snapshot = UsageTracker().snapshot()
    snapshot["total"].update({
        "calls": 3, "input_tokens": 100, "output_tokens": 50,
        "cache_read_tokens": 60, "total_tokens": 150,
    })
    snapshot["by_role"]["supervisor"] = dict(snapshot["total"])
    text = format_usage_summary(snapshot)
    assert "cache read 60" in text
    assert "supervisor" in text
    assert "150 tokens" in text


def test_observer_writes_usage_report_when_file_logging_enabled(tmp_path):
    config = api_config(log_mode="file", logging_enabled=True)
    observer = Observer(run_id="r1", config=config, console_enabled=False)
    observer.init_run_folder(tmp_path, "2026-01-01_00-00-00")
    observer.write_usage_report({"calls": 1, "records": [{"role": "supervisor"}]})

    usage_file = observer.log_folder / "usage.json"
    assert usage_file.exists()
    assert json.loads(usage_file.read_text(encoding="utf-8"))["calls"] == 1


def test_extract_usage_deepseek_cache_and_reasoning():
    response = _fake_response(
        response_metadata={
            "model_provider": "deepseek",
            "model_name": "deepseek-flash",
            "token_usage": {
                "prompt_tokens": 1200,
                "completion_tokens": 95,
                "total_tokens": 1295,
                "prompt_cache_hit_tokens": 900,
                "prompt_cache_miss_tokens": 300,
                "completion_tokens_details": {"reasoning_tokens": 60},
            },
        }
    )
    usage = extract_usage(response)
    assert usage["provider"] == "deepseek"
    assert usage["input_tokens"] == 1200
    assert usage["cache_read_tokens"] == 900
    assert usage["cache_miss_tokens"] == 300
    assert usage["reasoning_tokens"] == 60
    assert usage["total_tokens"] == 1295


def test_extract_usage_tier_prefixed_and_raw_cache_write_keys():
    tiered = _fake_response(
        usage_metadata={
            "input_tokens": 100, "output_tokens": 10, "total_tokens": 110,
            "input_token_details": {"priority_cache_read": 40, "flex_cache_creation": 10},
        }
    )
    usage = extract_usage(tiered)
    assert usage["cache_read_tokens"] == 40
    assert usage["cache_creation_tokens"] == 10

    raw = _fake_response(
        response_metadata={
            "token_usage": {
                "prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110,
                "prompt_tokens_details": {"cache_write_tokens": 25},
            }
        }
    )
    assert extract_usage(raw)["cache_creation_tokens"] == 25


def test_tool_usage_tracker_and_event():
    sink = EventCollector()
    config = RunConfig().finalize()
    observer = Observer(run_id="r1", config=config, event_sink=sink, console_enabled=False)
    context = RuntimeContext(
        run_id="r1", thread_id="r1", config=config,
        credentials=Credentials(), models=FakeModelFactory(config),
        observer=observer, console_enabled=False,
    )
    with runtime_scope(context):
        record_tool_call("tavily", "tavily_search", query="batteries")
        record_tool_call("tavily", "tavily_search", query="storage")
        record_tool_call("exa", "exa_deep_search", query="grid")
        record_tool_call("perplexity", "perplexity_search", query="news")
        snapshot = context.tool_usage.snapshot()
        asyncio.run(observer.flush_events())

    assert snapshot["total"] == 4
    assert snapshot["by_provider"] == {"tavily": 2, "exa": 1, "perplexity": 1}
    assert snapshot["by_tool"]["tavily_search"] == 2
    assert len(context.tool_usage.records()) == 4
    assert sink.types() == [TOOL_USAGE] * 4


def test_model_fallback_is_flagged_as_inferred():
    config = RunConfig().finalize()
    context = RuntimeContext(
        run_id="r1", thread_id="r1", config=config,
        credentials=Credentials(), models=FakeModelFactory(config),
        observer=Observer(run_id="r1", config=config, console_enabled=False),
        console_enabled=False,
    )
    with runtime_scope(context):
        entry = record_llm_usage(_fake_response(), role="supervisor")

    assert entry["usage_available"] is False
    assert entry["model_inferred"] is True
    assert entry["model"]


def test_research_brief_records_usage_from_raw_response():
    config = RunConfig().finalize()
    context = RuntimeContext(
        run_id="r1", thread_id="r1", config=config,
        credentials=Credentials(), models=FakeModelFactory(config),
        observer=Observer(run_id="r1", config=config, console_enabled=False),
        console_enabled=False,
    )
    with runtime_scope(context):
        command = write_research_brief({"messages": []})
        snapshot = context.usage.snapshot()

    assert snapshot["by_role"]["scope_brief"]["total_tokens"] == 15
    assert snapshot["by_role"]["scope_brief"]["calls_with_usage"] == 1
    assert command.update["research_brief"]


def test_format_usage_summary_includes_search_calls():
    snapshot = UsageTracker().snapshot()
    tool_snapshot = ToolUsageTracker().snapshot()
    tool_snapshot["total"] = 3
    tool_snapshot["by_provider"] = {"exa": 2, "tavily": 1}
    text = format_usage_summary(snapshot, tool_snapshot)
    assert "Search calls: 3" in text
    assert "exa 2" in text
