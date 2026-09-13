import asyncio
from pathlib import Path

from dotenv import load_dotenv

from deep_research.integration import RunConfig, run_research  # noqa: E402

load_dotenv()


CONFIG = RunConfig(
    supervisor_model_fallback_chain=["deepseek-v4-flash"],
    subagent_model_fallback_chain=["deepseek-v4-flash"],
    research_time_max_minutes=10,
)


async def main():
    result = await run_research("What are the main benefits and limitations of sodium-ion batteries for home energy storage?", config=CONFIG)
    print(f"Status: {result.status}")
    if result.failure:
        print(result.failure)
    print(result.final_report)  # Markdown-formatted string; no file is created.

    # Optional: save a completed report yourself.
    # if result.status == "completed":
    #     Path("report.md").write_text(result.final_report, encoding="utf-8")

    return 0 if result.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

