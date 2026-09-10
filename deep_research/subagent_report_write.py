"""Report generation — synthesize a markdown report from curated sources.

Used when output_mode is "report". The report is built ONLY from the
saved_articles dict (the curated evidence base), never from raw tool
output — per the design philosophy.

Citation flow: the writer sees a numbered registry of sources and emits only
inline [N] citations; `finalize_citations` rebuilds the ## Sources section in
code from that registry, so the model never reproduces URLs.
"""

import asyncio
import logging
import re

from langchain_core.messages import HumanMessage

from deep_research.config import LLM_TIMEOUT
from deep_research.runtime import get_runtime
from deep_research.platform_prompts import REPORT_GENERATION_PROMPT
from deep_research.citation_utils import finalize_citations

logger = logging.getLogger(__name__)


def _build_source_registry(saved_articles: dict) -> list[dict]:
    """Ordered registry of curated sources; list order defines the [N] ids
    shown to the writer and used to rebuild the Sources section in code."""
    registry = []
    for identifier, item in sorted(saved_articles.items()):
        registry.append({
            "identifier": identifier,
            "title": item.get("title") or "",
            "url": str(item.get("url") or identifier),
            "reason": item.get("reason") or "",
            "content": item.get("content") or item.get("preview", ""),
        })
    return registry


def _format_registry(registry: list[dict]) -> str:
    """Render the registry into the prompt's <Curated Sources> block.

    Each source is numbered [N] — the stable id the writer must cite inline.
    The full content is included so the writer can quote from it directly.
    """
    if not registry:
        return "(No sources were saved during this research session.)"

    parts = []
    for idx, entry in enumerate(registry, 1):
        header = f"[{idx}] {entry['title']} ({entry['url']})" if entry["title"] else f"[{idx}] {entry['url']}"
        parts.append(header)
        if entry["reason"]:
            parts.append(f"- **Reason saved**: {entry['reason']}")
        if entry["content"]:
            parts.append(f"- **Full content**:")
            parts.append(entry["content"])
        parts.append("")
    return "\n".join(parts)


def _safe_title(title: str) -> str:
    safe = re.sub(r"[^\w\s-]", "", title).strip().replace(" ", "_")[:80]
    return safe or "research"


async def generate_report(
    saved_articles: dict,
    topic: str,
    platform_label: str = "research",
    model_chain: list = None,
) -> str:
    """Synthesize a markdown research report from the curated sources.

    Returns the report text (no file IO — writing happens at the runner boundary).
    """
    if not saved_articles:
        msg = "No sources were saved during this session — report skipped."
        logger.warning(msg)
        return msg

    title = f"{platform_label.title()} Research: {topic[:60]}"
    registry = _build_source_registry(saved_articles)
    findings_text = _format_registry(registry)
    prompt = REPORT_GENERATION_PROMPT.format(
        title=title,
        topic=topic,
        saved_items_text=findings_text,
    )

    _rt = get_runtime()
    model = _rt.models.subagent(max_tokens=8000, chain=model_chain)

    logger.info(f"Generating report ({len(saved_articles)} sources)...")
    try:
        response = await asyncio.wait_for(
            model.ainvoke([HumanMessage(content=prompt)]),
            timeout=_rt.config.llm_timeout,
        )
        report_text = response.content if isinstance(response.content, str) else str(response.content)
    except Exception as e:
        logger.warning(f"Report generation failed ({type(e).__name__}): {e}")
        return _format_registry(registry)

    # Deterministic Sources rebuild — URLs mapped in code from the registry.
    return finalize_citations(report_text, registry)
