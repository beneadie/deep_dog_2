"""Run with: python scripts/simple_research.py"""

import asyncio
from pathlib import Path

from dotenv import load_dotenv

# Put DEEPSEEK_API_KEY and EXA_API_KEY in the repository's .env file.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from deep_research.integration import run_research  # noqa: E402

# Change this to your research question. All engine settings use defaults.
QUESTION = "What are the main benefits and limitations of sodium-ion batteries for home energy storage?"


async def main():
    result = await run_research(QUESTION)
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
