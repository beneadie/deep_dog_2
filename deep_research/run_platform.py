"""Direct single-platform runner.

Run one platform research agent standalone (also the programmatic
benchmark/launch contract):

    python -m deep_research.run_platform --agent reddit --topic "NVIDIA earnings Q4" \
        --provider nvidia-30b -m 10 --output-mode report

    python -m deep_research.run_platform --agent pubmed --topic "GLP1 efficacy" \
        --provider deepseek -m 8 --output-mode sources

--output-mode report        → save-as-you-go, then isolated LLM writes the report
--output-mode report_inline → agent reads everything and writes the report inline
--output-mode sources       → curated source list (save-as-you-go)
--output-mode sources_inline→ curated source list (read-all, batch-select at end)
"""

import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if __name__ == "__main__" and not __package__:
    __package__ = Path(__file__).resolve().parent.name
load_dotenv()

from langchain_core.messages import HumanMessage

from deep_research.config import (
    DEFAULT_PLATFORM,
    DEFAULT_MAX_TOTAL_READS,
    SUBAGENT_MAX_CONCURRENCY,
    SUBAGENT_MAX_ITERATIONS,
    SUBAGENT_MAX_READS,
    SUBAGENT_MAX_SAVES,
    SUBAGENT_MAX_SEARCHES,
    SUBAGENT_MODEL_FALLBACK_CHAIN,
    subagent_recursion_limit,
)
from deep_research.agents.base import PLATFORMS, list_platforms, researcher_agent

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


async def run_platform(
    agent_type: str = "reddit",
    topic: str = "",
    model_chain: list | None = None,
    max_iterations: int = SUBAGENT_MAX_ITERATIONS,
    max_reads: int = SUBAGENT_MAX_READS,
    max_total_reads: int = DEFAULT_MAX_TOTAL_READS,
    max_saves: int = SUBAGENT_MAX_SAVES,
    max_searches: int = SUBAGENT_MAX_SEARCHES,
    max_concurrency: int = SUBAGENT_MAX_CONCURRENCY,
    target_language: str = "auto",
    output_mode: str = "report",
    include_article_text: bool = False,
    save_md: bool = False,
    output_dir: Path | None = None,
) -> dict:
    """Run a single platform research agent.

    Returns:
        {"compressed_research", "saved_articles", "iteration_count",
         "articles_read", "search_results"}
    """
    plat = PLATFORMS.get(agent_type, PLATFORMS[DEFAULT_PLATFORM])
    t_start = time.perf_counter()

    recursion_limit = subagent_recursion_limit(max_iterations)

    # Report modes mandate full text in the sources note (the report writer
    # needs it for the final synthesis)
    if output_mode in ("report", "report_inline"):
        include_article_text = True

    result = await researcher_agent.ainvoke({
        "researcher_messages": [HumanMessage(content=f"Research this topic: {topic}")],
        "research_topic": topic,
        "agent_type": agent_type,
        "discovery": False,
        "model_chain": model_chain or list(SUBAGENT_MODEL_FALLBACK_CHAIN),
        "target_language": target_language,
        "output_mode": output_mode,
        "include_article_text": include_article_text,
        "articles_read": {},
        "saved_articles": {},
        "findings_log": {},
        "search_count": 0,
        "search_results": {},
        "max_iterations": max_iterations,
        "max_reads": max_reads,
        "max_total_reads": max_total_reads,
        "max_saves": max_saves,
        "max_searches": max_searches,
        "max_concurrency": max_concurrency,
    }, {"recursion_limit": recursion_limit})

    t_total = round(time.perf_counter() - t_start, 1)
    logger.info(f"Agent run complete: {t_total}s total")

    saved_articles = result.get("saved_articles", {})
    deliverable = result.get("compressed_research", "") or ""
    if save_md and deliverable:
        from deep_research.subagent_report_write import _safe_title
        kind = "Report" if output_mode in ("report", "report_inline") else "Sources"
        filename = f"{plat.get('label', agent_type).title()}_{kind}_{_safe_title(topic)}.md"
        out_path = (output_dir or Path.cwd()) / filename
        out_path.write_text(deliverable, encoding="utf-8")
        logger.info(f"Saved: {out_path}")
    elif save_md and not deliverable:
        logger.warning("Output was empty — no .md file written")

    return {
        "compressed_research": result.get("compressed_research", ""),
        "saved_articles": saved_articles,
        "iteration_count": result.get("iteration_count", 0),
        "articles_read": result.get("articles_read", {}),
        "search_results": result.get("search_results", {}),
    }


# CLI short name → model-name chain (resolved by get_model). These are plain
# model names, so gemini and the Alibaba-Singapore deepseek are just another
# chain entry. `--provider` picks one of these; default = SUBAGENT_MODEL_FALLBACK_CHAIN.
_CLI_MODEL_CHAIN = {
    "mimo": ["mimo-v2.5-pro"],
    "deepseek": ["deepseek-v4-flash"],
    "deepseek-baba-singapore": ["deepseek-baba-singapore"],
    "gemini": ["gemini-3-flash-preview"],
    "meta": ["meta/muse-spark-1.2"],
    "openai": ["gpt-5.2"],
    "nvidia-lightning": ["nvidia/nemotron-3.5-lightning"],
    "nvidia-120b": ["nvidia/nemotron-3-super-120b-a12b"],
    "nvidia-30b": ["nvidia/nemotron-3-nano-30b-a3b"],
    "gpt-5.6-luna": ["gpt-5.6-luna"],
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Platform Research Agent")
    parser.add_argument("--agent", "-a", type=str, default=DEFAULT_PLATFORM,
                        choices=list_platforms(),
                        help=f"Platform agent to run: {list_platforms()}")
    parser.add_argument("--topic", "-t", type=str, required=True, help="Research topic")
    parser.add_argument("--provider", "-p", type=str, default=None,
                        choices=sorted(_CLI_MODEL_CHAIN),
                        help="Model chain to use (maps to a model-name chain, e.g. 'gemini', "
                             "'deepseek-baba-singapore'). Default: SUBAGENT_MODEL_FALLBACK_CHAIN")
    parser.add_argument("--max-iterations", "-m", type=int, default=SUBAGENT_MAX_ITERATIONS,
                        help=f"Max tool-call rounds (default: {SUBAGENT_MAX_ITERATIONS})")
    parser.add_argument("--max-reads", type=int, default=SUBAGENT_MAX_READS,
                        help=f"Max items to read per iteration (default: {SUBAGENT_MAX_READS})")
    parser.add_argument("--max-total-reads", type=int, default=DEFAULT_MAX_TOTAL_READS,
                        help=f"Max items to read across the whole run (default: {DEFAULT_MAX_TOTAL_READS})")
    parser.add_argument("--max-saves", type=int, default=SUBAGENT_MAX_SAVES,
                        help=f"Max items to save per run (default: {SUBAGENT_MAX_SAVES})")
    parser.add_argument("--max-searches", type=int, default=SUBAGENT_MAX_SEARCHES,
                        help=f"Max search tool calls per run (default: {SUBAGENT_MAX_SEARCHES})")
    parser.add_argument("--concurrency", type=int, default=SUBAGENT_MAX_CONCURRENCY,
                        help=f"Max parallel tool calls per iteration (default: {SUBAGENT_MAX_CONCURRENCY})")
    parser.add_argument("--language", type=str, default="auto",
                        help="Target language ('auto' = agent detects, or specify e.g. 'English')")
    parser.add_argument("--output-mode", "-o", type=str, default="report",
                        choices=["sources", "report", "sources_inline", "report_inline"],
                        help="'report' = isolated report writer (default); 'report_inline' = agent writes "
                             "report inline; 'sources' = curated list (save-as-you-go); "
                             "'sources_inline' = curated list (read-all, batch-select at end)")
    parser.add_argument("--include-article-text", action="store_true", default=False,
                        help="Include full article text in the sources note (forced True for report modes)")
    parser.add_argument("--save-md", action="store_true", default=False,
                        help="Write the output to a .md file (default: print to terminal only)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to write report files (default: cwd)")
    return parser


async def async_main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    print(f"\n{'=' * 60}")
    print(f"  Platform: {args.agent}  |  Provider: {args.provider or 'sub-agent chain'}")
    print(f"  Topic:    {args.topic}")
    print(f"  Iterations: {args.max_iterations}  |  Output mode: {args.output_mode}")
    print(f"  Reads: {args.max_reads}/iter  |  Total reads: {args.max_total_reads}  |  Saves: {args.max_saves}  |  "
          f"Searches: {args.max_searches}  |  Concurrency: {args.concurrency}")
    print(f"  Language: {args.language}  |  Include text: {args.include_article_text}  |  Save .md: {args.save_md}")
    print(f"{'=' * 60}\n")

    model_chain = _CLI_MODEL_CHAIN.get(args.provider) if args.provider else None
    result = await run_platform(
        agent_type=args.agent,
        topic=args.topic,
        model_chain=model_chain,
        max_iterations=args.max_iterations,
        max_reads=args.max_reads,
        max_total_reads=args.max_total_reads,
        max_saves=args.max_saves,
        max_searches=args.max_searches,
        max_concurrency=args.concurrency,
        target_language=args.language,
        output_mode=args.output_mode,
        include_article_text=args.include_article_text,
        save_md=args.save_md,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )

    print("\n" + "=" * 60)
    if args.output_mode in ("sources", "sources_inline"):
        print(f"  SOURCES SELECTED ({len(result['saved_articles'])} items)")
    else:
        print("  RESEARCH SUMMARY")
    print("=" * 60)
    print(result["compressed_research"][:5000])

    if result["saved_articles"] and args.output_mode in ("sources", "sources_inline"):
        print(f"\n{'=' * 60}")
        print(f"  SAVED ITEMS ({len(result['saved_articles'])} total)")
        print("=" * 60)
        for identifier, item in result["saved_articles"].items():
            print(f"\n  [{item.get('title', identifier)[:80]}]")
            print(f"    URL: {item.get('url', identifier)}")
            print(f"    Reason: {item.get('reason', '')[:200]}")

    if result["compressed_research"] and args.output_mode in ("report", "report_inline"):
        print(f"\n{'=' * 60}")
        print(f"  REPORT GENERATED ({len(result['compressed_research'])} chars)")
        print("=" * 60)
        print(result["compressed_research"])


if __name__ == "__main__":
    asyncio.run(async_main())
