"""Run from a source checkout after `python -m pip install -e .`.

Edit CONFIG below to choose models, agents, search and research budgets.
"""

import argparse
import asyncio
from pathlib import Path

from dotenv import load_dotenv

# Load deployment defaults before importing the engine. Existing environment
# variables take precedence over the checkout's .env file.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from deep_research.integration import (  # noqa: E402
    Credentials, RunConfig, RuntimeOptions, run_research, validate_credentials,
)

CONFIG = RunConfig(
    # A one-item chain selects one model. Add names to enable fallback options.
    supervisor_model_fallback_chain=["deepseek-v4-flash"],
    subagent_model_fallback_chain=["deepseek-v4-flash"],
    draft_report_model_fallback_chain=["deepseek-v4-flash"],
    subagent_model_chain_by_agent={},
    disable_model_fallback=True,
    supervisor_route_via_openrouter=False,  # Native DeepSeek API.
    subagent_route_via_openrouter=False,
    draft_route_via_openrouter=False,
    enabled_agents=["ResearchWeb"],
    web_search_engine="exa",  # "exa", "tavily", or "both".
    research_time_min_minutes=2,
    research_time_max_minutes=10,
    supervisor_max_iterations=20,
    subagent_max_iterations=5,  # Keep >= 4: search, select, read, save.
    subagent_max_searches=3,  # Per delegated agent, not the entire run.
    subagent_max_reads=10,
    subagent_max_total_reads=25,
    subagent_max_saves=10,
    subagent_max_concurrency=3,
    subagent_output_mode="sources",  # Save-as-you-go curation.
    discovery_output_mode="report_inline",  # Write from the agent's context.
    output_mode="none",  # This example saves the returned report itself.
    log_mode="none",
    save_report_to_file=False,
    save_subagent_reports_to_file=False,
    enable_source_log=False,
    enable_subtopic_generation=False,
    enable_research_trace=False,
    logging_enabled=False,  # Set True for rich result.logs / trace_sink data.
)


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", help="Research question, in quotes")
    parser.add_argument("--output", type=Path, default=Path("outputs/report.md"))
    args = parser.parse_args()

    config = CONFIG.finalize()
    credentials = Credentials.from_env()
    check = validate_credentials(config, credentials)
    if not check.ok:
        parser.error("Missing credentials: " + ", ".join(check.missing_required))

    result = await run_research(
        args.prompt, config=config, credentials=credentials,
        runtime=RuntimeOptions(console_enabled=True),
    )
    print(f"Status: {result.status}")
    if result.failure:
        print(result.failure)
    if result.final_report:
        # Keep interrupted output separate from a completed report.
        output = args.output
        if result.status != "completed":
            output = output.with_name(output.stem + ".partial" + output.suffix)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(result.final_report, encoding="utf-8")
        print(f"Report: {output.resolve()}")
    return 0 if result.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
