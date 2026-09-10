#!/usr/bin/env python3

"""

Deep Research Agent - Server Runner

A standalone script for running the Deep Research agent on a server.

Loads environment variables, runs research, and saves outputs with citations.

Usage (with uv):

    uv run python run_research.py --prompt "Your research question here"

    uv run python run_research.py --prompt-file input.txt

    uv run python run_research.py --prompt-file input.txt --output-dir my_outputs

Usage (without uv, if dependencies are installed):

    python run_research.py --prompt "Your research question here"

The CLI now executes through the same stable engine entry point the API uses
(``deep_research.integration.run_research``) but with file output enabled, so
CLI behaviour (Markdown/JSON/trace files) is preserved while the underlying
engine is concurrency-safe and per-run configurable.
"""

import asyncio
import argparse
import json
import logging
import random
import string
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

# Load environment variables before importing anything else
from dotenv import load_dotenv

# Find and load .env file from project root
env_path = Path(__file__).resolve().parent / '.env'
load_dotenv(dotenv_path=env_path)

from langchain_core.messages import HumanMessage  # noqa: E402

from deep_research.config import ENABLE_RESEARCH_TRACE, OUTPUT_MODE, get_supervisor_model  # noqa: E402
from deep_research.integration import (  # noqa: E402
    RuntimeOptions,
    RunStatus,
    run_research as engine_run_research,
)
from deep_research.run_config import RunConfig  # noqa: E402
from deep_research.runtime import PromptLibrary  # noqa: E402
from deep_research.utils import extract_text_from_response  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def _save_output_to_db(output_data: dict) -> None:
    """
    Placeholder for database insert logic for research outputs.

    Implement your DB insert here when ready. The output_data dict contains:
        - thread_id: Unique identifier for this research session
        - timestamp: When the research was run
        - prompt: Original user query
        - research_brief: The generated research plan
        - final_report: The main report content
        - notes: List of raw research notes
    """
    # TODO: Implement database insert logic
    pass


async def run_research(prompt: str, output_dir: Path, thread_id: str = None,
                       clean_output: bool = False) -> Dict[str, Any]:
    """
    Run the Deep Research agent and save the output to a file.

    Args:
        prompt: The research prompt/question
        output_dir: Directory to save the output file
        thread_id: Optional thread ID for the research session
        clean_output: If True, write only the final report (no header/brief)

    Returns:
        Dictionary with:
            - output_file: Path to the saved output file
            - final_report: The raw final report content
            - sources: List of source dictionaries
            - trace_content: The research trace document (if enabled)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate timestamp for filename and thread (random letters avoid
    # collisions when multiple agents start in the same second).
    random_suffix = ''.join(random.choices(string.ascii_lowercase, k=3))
    timestamp = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{random_suffix}"
    if thread_id is None:
        thread_id = timestamp
    output_file = output_dir / f"research_{timestamp}.md"

    logger.info(f"Starting research with thread ID: {thread_id}")
    logger.info(f"Output will be saved to: {output_file}")

    cfg = RunConfig().finalize()
    result = await engine_run_research(
        prompt,
        config=cfg,
        runtime=RuntimeOptions(
            run_id=thread_id,
            thread_id=thread_id,
            output_dir=output_dir,
            console_enabled=True,
        ),
    )

    if result.status in (RunStatus.FAILED.value, RunStatus.CANCELLED.value) and not result.final_report:
        error = result.failure or result.status
        error_file = output_dir / f"error_{timestamp}.txt"
        error_file.write_text(
            f"Error occurred at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            f"Prompt:\n{prompt}\n\n"
            f"Status: {result.status}\nError:\n{error}\n",
            encoding="utf-8",
        )
        logger.error(f"Research failed ({result.status}): {error}")
        raise RuntimeError(f"Research failed ({result.status}): {error}")

    final_report = result.final_report or result.draft_report or ""
    research_brief = result.research_brief or ""
    sources = result.sources if result.sources else {}

    output_data = {
        "thread_id": thread_id,
        "timestamp": datetime.now().isoformat(),
        "prompt": prompt,
        "research_brief": research_brief,
        "final_report": final_report,
        "status": result.status,
    }

    # Write to file if enabled (CLI default output_mode is "file")
    if OUTPUT_MODE in ("file", "both"):
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            if not clean_output:
                f.write("# Deep Research Report\n\n")
                f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write("---\n\n")
                f.write("## Research Prompt\n\n")
                f.write(f"{prompt}\n\n")
                f.write("---\n\n")
                if research_brief:
                    f.write("## Research Brief\n\n")
                    f.write(f"{research_brief}\n\n")
                    f.write("---\n\n")
            f.write("## Final Report\n\n")
            f.write(f"{final_report}\n\n")
        logger.info(f"Report saved to file: {output_file}")

    if OUTPUT_MODE in ("file", "both") and sources:
        sources_json_file = output_dir / f"research_data_{timestamp}.json"
        research_data = {
            "thread_id": thread_id,
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "sources": sources,
            "final_report": final_report,
        }
        sources_json_file.parent.mkdir(parents=True, exist_ok=True)
        with open(sources_json_file, "w", encoding="utf-8") as f:
            json.dump(research_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Sources saved to: {sources_json_file} ({len(sources)} sources)")

    if OUTPUT_MODE in ("db", "both"):
        _save_output_to_db(output_data)
        logger.info("Report saved to database")

    logger.info("Research complete!")

    # Generate research trace content (only if enabled in config/env).
    trace_content = None
    trace_file = None
    trace_data = result.trace if ENABLE_RESEARCH_TRACE else []

    if trace_data:
        interaction_log = ""
        for loop in trace_data:
            interaction_log += f"""

--- Loop {loop['loop_number']} (at {loop['timestamp']}) ---

SUPERVISOR DELEGATED RESEARCH TOPIC:

{loop['research_topic']}

SUBAGENT RETURNED FINDINGS:

{loop['findings'][:] if loop.get('findings') else 'No findings returned'}

SUPERVISOR REACTION:

{loop.get('supervisor_reaction', 'No explicit reaction captured')}

"""
        try:
            logger.info("Generating research trace document...")
            prompts = PromptLibrary(cfg.prompt_version)
            trace_prompt_tmpl = prompts.get("research_trace_compression_prompt", "")
            if not trace_prompt_tmpl:
                raise ValueError("research_trace_compression_prompt unavailable")
            trace_prompt = trace_prompt_tmpl.format(
                research_brief=research_brief,
                interaction_log=interaction_log,
            )
            trace_model = get_supervisor_model(max_tokens=16000)
            trace_response = await trace_model.ainvoke([HumanMessage(content=trace_prompt)])
            trace_content = extract_text_from_response(trace_response.content)
        except Exception as e:
            logger.warning(f"Failed to generate research trace: {e}")
            trace_content = "# Research Process Trace (Raw)\n\n"
            trace_content += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            trace_content += f"**Research Brief:** {research_brief}\n\n"
            trace_content += "---\n\n"
            trace_content += interaction_log

        if OUTPUT_MODE in ("file", "both") and trace_content:
            trace_file = output_dir / f"trace_{timestamp}.md"
            trace_file.parent.mkdir(parents=True, exist_ok=True)
            with open(trace_file, "w", encoding="utf-8") as f:
                f.write(trace_content)
            logger.info(f"Research trace saved to: {trace_file}")

    output_data["trace_content"] = trace_content
    if OUTPUT_MODE in ("db", "both"):
        _save_output_to_db(output_data)
        logger.info("Report saved to database")

    return {
        "output_file": str(output_file),
        "final_report": final_report,
        "sources": sources,
        "trace_content": trace_content,
    }


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Run the Deep Research Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_research.py --prompt "What are the latest developments in AI safety?"

    python run_research.py --prompt-file research_question.txt

    python run_research.py --prompt-file input.txt --output-dir results

        """,
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--prompt", "-p",
        type=str,
        help="Research prompt/question to investigate",
    )
    input_group.add_argument(
        "--prompt-file", "-f",
        type=str,
        help="Path to a file containing the research prompt",
    )

    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="outputs",
        help="Directory to save output files (default: outputs)",
    )
    parser.add_argument(
        "--thread-id", "-t",
        type=str,
        default=None,
        help="Thread ID for the research session (default: auto-generated timestamp)",
    )

    args = parser.parse_args()

    if args.prompt:
        prompt = args.prompt
    else:
        prompt_path = Path(args.prompt_file)
        if not prompt_path.exists():
            logger.error(f"Prompt file not found: {prompt_path}")
            sys.exit(1)
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt = f.read().strip()

    if not prompt:
        logger.error("Empty prompt provided")
        sys.exit(1)

    logger.info(f"Prompt length: {len(prompt)} characters")

    output_dir = Path(args.output_dir)

    try:
        result = asyncio.run(run_research(prompt, output_dir, args.thread_id))
        print(f"\n✅ Research complete! Output saved to: {result['output_file']}")
    except KeyboardInterrupt:
        logger.info("Research interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Research failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
