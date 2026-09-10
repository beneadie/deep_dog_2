"""Shared tools: reflection, web fetching, curation, and session tools.

These tools are shared across all platform agents. The stub tools
(fetch_urls, list_saved, log_finding, batch_save_selected) are intercepted
by the platform agent engine's tool_node, which does the real state writes —
the tools themselves are stateless and safe for concurrent invocations.
"""

import asyncio
import logging
import os
import re
import requests
from typing import Literal

import httpx
from bs4 import BeautifulSoup
from langchain_core.tools import tool

from deep_research.config import FETCH_URL_MAX_CHARS
from deep_research.secrets import get_secret

logger = logging.getLogger(__name__)

HTTP_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 "
    "Safari/537.36"
)


# ═════════════════════════════════════════════════════════════════════════
#  THINK TOOL
# ═════════════════════════════════════════════════════════════════════════

@tool(parse_docstring=True)
async def think_tool(
    purpose: Literal["denoise", "plan", "assess", "complete"] = "denoise",
    reflection: str = "",
) -> str:
    """Tool for strategic reflection on research progress and decision-making.

    You can use this tool after searches to analyze results and plan next steps systematically.
    This creates a deliberate pause in the research workflow for quality decision-making.

    Purpose guides the reflection:
    - denoise: (default) produce the denoise report — (a) mark every draft section /
      research angle as [COVERED], [PARTIAL], [UNSUPPORTED], or [CONTRADICTED];
      (b) list the concrete residual shortcomings (see <Gap Taxonomy>); (c) give the
      1-3 next research topics that target the top shortcomings; (d) end with a single
      verdict line: "VERDICT: CONTINUE_RESEARCH", "VERDICT: READY_TO_CONCLUDE", or
      "VERDICT: TIME_LIMIT".
    - plan: plan the next delegation(s) against the current gaps.
    - assess: review findings so far (what changed, what is still missing).
    - complete: final check before ResearchComplete — confirm every material shortcoming
      is closed or ruled out, or that the time limit is reached.

    When to use:
    - After receiving search results: What key information did I find?
    - Before deciding next steps: Do I have enough to answer comprehensively?
    - When assessing research gaps: What specific information am I still missing?
    - Before concluding research: Can I provide a complete answer now?

    Args:
        purpose: Which kind of reflection this is (denoise, plan, assess, or complete)
        reflection: Your detailed reflection following the purpose's template

    Returns:
        Confirmation that reflection was recorded for decision-making
    """
    return f"Reflection recorded: {reflection}"


# ═════════════════════════════════════════════════════════════════════════
#  PERPLEXITY SEARCH  (underlying engine for Substack search; never bound
#  raw to platform agents — wrapped inside search_substack)
# ═════════════════════════════════════════════════════════════════════════

PERPLEXITY_RECENCY = {"hour", "day", "week", "month", "year"}


@tool(parse_docstring=True)
async def perplexity_search(
    query: str,
    recency: str = "month",
    domain_filter: str = "",
) -> str:
    """Search the web via the Perplexity API for realtime, up-to-date information.

    Supports an optional domain filter (e.g. 'substack.com') to restrict
    results to a single site. Use this for current events, market data,
    breaking news, and anything time-sensitive.

    Args:
        query: Natural-language search query
        recency: Time window — 'hour', 'day', 'week', 'month', 'year'
        domain_filter: Optional domain to restrict results to (e.g. 'substack.com')
    """
    api_key = get_secret("PERPLEXITY_API_KEY")
    if not api_key:
        return "Error: PERPLEXITY_API_KEY not set in .env"
    if recency not in PERPLEXITY_RECENCY:
        recency = "month"

    payload = {"query": query, "search_recency_filter": recency}
    if domain_filter:
        payload["search_domain_filter"] = [domain_filter]

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    label = domain_filter.split(".")[0].title() if domain_filter else "Perplexity"
    logger.info(f"{label} search: '{query[:80]}' (recency={recency}, domain={domain_filter or 'all'})")

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post("https://api.perplexity.ai/search", json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        return f"Perplexity search error: {e}"

    results = data.get("results", []) if isinstance(data, dict) else []
    if not results:
        return f"No results for: '{query}'"

    lines = [f"Found {len(results)} results for '{query}':", ""]
    for i, r in enumerate(results[:10], 1):
        title = r.get("title", "Untitled")
        url = r.get("url", "")
        snippet = r.get("snippet", "")[:250]
        date = r.get("date", "")
        lines.append(f"{i}. **{title}**")
        if date:
            lines.append(f"   Date: {date}")
        lines.append(f"   URL: {url}")
        lines.append(f"   {snippet}")
        lines.append("")

    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════
#  GENERIC URL FETCH  (base tool — available to all platforms)
# ═════════════════════════════════════════════════════════════════════════

def _scrape_generic(html: str) -> dict:
    """Extract title + author + date + body from arbitrary HTML."""
    soup = BeautifulSoup(html, "html.parser")

    title = ""
    og_title = soup.find("meta", attrs={"property": "og:title"})
    if og_title:
        title = og_title["content"].strip()
    if not title:
        h1 = soup.find("h1")
        if h1:
            title = h1.get_text().strip()
    if not title:
        title_el = soup.find("title")
        if title_el:
            title = title_el.get_text().strip()
    title = title[:200]

    content = ""
    for sel in ["div.available-content", "div.body", "article", "div#body",
                "div#fulltext", "main", "div.content", "div.article-body",
                "div.article-content"]:
        container = soup.select_one(sel)
        if container and len(container.get_text(strip=True)) > 500:
            paragraphs = container.find_all(["p", "h1", "h2", "h3", "li"])
            content = "\n\n".join(p.get_text().strip() for p in paragraphs if p.get_text().strip())
            if len(content) > 500:
                break

    if not content:
        paragraphs = soup.find_all("p")
        content = "\n\n".join(p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 30)

    author_meta = soup.find("meta", attrs={"name": "author"})
    author = author_meta["content"].strip() if author_meta else ""
    date_meta = soup.find("meta", attrs={"property": "article:published_time"})
    date = date_meta["content"].strip() if date_meta else ""

    return {"title": title, "author": author, "date": date, "content": content}


@tool(parse_docstring=True)
async def fetch_urls(urls: list) -> str:
    """Fetch and extract the full content of one or more external URLs.

    Use to follow links found via other tools — a Reddit post linking to a news
    article, a Substack article citing a report, a search result pointing at a
    primary source. Returns title, author, date, and body text for each URL.

    - Each fetched URL gets an [S#] handle so you can save it later with
      batch_save_selected(items=[{"ref": "S#", "index": N, "reason": "..."}]).
    - Budgeted at ~3 URLs per iteration — chase only links that are clearly
      central to the research (official docs, reports, primary articles).
    - Use for PRIMARY sources you can't read via your platform's own read tool,
      not for pages you can already read directly.

    Args:
        urls: List of full https URLs to fetch
    """
    return ""  # state write handled by tool_node


async def _fetch_url_content(url: str) -> tuple[str, str]:
    """Fetch and extract content from a single URL (via Tavily).

    Returns (title, body): the title is used to register an [S#] index entry
    so external URLs can be saved by ref; the body is the model-facing text.
    """
    logger.info(f"fetch_urls: {url}")
    api_key = get_secret("TAVILY_API_KEY")
    if not api_key:
        return url, "Error: fetch_urls requires TAVILY_API_KEY in .env"

    try:
        from tavily import TavilyClient
    except ImportError:
        return url, "Error: tavily-python not installed. Run: pip install tavily-python"

    try:
        client = TavilyClient(api_key=api_key)
        response = await asyncio.to_thread(
            client.extract, urls=[url], include_images=False, extract_depth="advanced")
    except Exception as e:
        return url, f"Failed to fetch {url}: {e}"

    results = response.get("results", []) if isinstance(response, dict) else []
    failed = response.get("failed_results", []) if isinstance(response, dict) else []
    if not results:
        reason = failed[0].get("error", "unknown") if failed else "no content extracted"
        return url, f"Failed to extract content from {url}: {reason}"

    item = results[0]
    title = item.get("title", "") or url
    content = item.get("raw_content", "") or item.get("text", "") or ""
    author = item.get("author", "") or ""
    date = item.get("published_date", "") or ""

    text = content[:FETCH_URL_MAX_CHARS]
    if len(content) > FETCH_URL_MAX_CHARS:
        text += "\n\n[content truncated]"
    author_line = f"**Author:** {author}\n" if author else ""
    date_line = f"**Date:** {date}\n" if date else ""

    return title, f"# {title}\n{author_line}{date_line}**URL:** {url}\n\n{text}"


# ═════════════════════════════════════════════════════════════════════════
#  CURATION — shared (read-only; state read in tool_node)
# ═════════════════════════════════════════════════════════════════════════

@tool(parse_docstring=True)
async def list_saved() -> str:
    """List everything you've saved so far (titles, URLs, and save reasons).

    Call periodically to review your collection, spot gaps or duplicates, and
    confirm coverage; and before finishing to make sure you've curated the right
    items. In curation mode, review this before your final save round so you
    don't re-save items you already have.
    """
    return ""  # state read handled by tool_node


@tool(parse_docstring=True)
async def log_finding(key: str, value: str) -> str:
    """Record a freeform cross-item observation or statistic for the final report.

    Use for findings that span MULTIPLE items and aren't tied to a single
    source — e.g. key='sentiment', value='~70% of 20 posts bullish on X'.
    These are surfaced alongside your saved items, so use them to capture your
    synthesis as you go rather than only at the very end.

    Args:
        key: Short label for the finding (e.g. 'sentiment_trend')
        value: The finding content
    """
    return ""  # state write handled by tool_node


# ═════════════════════════════════════════════════════════════════════════
#  SESSION TOOLS  (intercepted by tool_node for state writes)
# ═════════════════════════════════════════════════════════════════════════

@tool(parse_docstring=True)
async def set_target_language(language: str) -> str:
    """Set the target language for all written output this session.

    Call this in your FIRST tool round, in the SAME batch as your first searches
    (never as a standalone iteration). Choose the language most appropriate for
    the topic and audience. Once set, write ALL output in that language and do
    not switch it mid-session.

    Args:
        language: Language name (e.g. 'English', 'Spanish', 'German')
    """
    return f"Target language set to: {language}"


@tool(parse_docstring=True)
async def finish_research(summary: str = "") -> str:
    """Signal that research is complete and exit early.

    Call this to end your run before the system forces it. Use when:
    - You've gathered enough evidence and saved your items.
    - Search tools are returning poor/irrelevant results and continuing
      would waste budget.
    - You've hit your depth limits (the system enforces this automatically,
      but you can exit early if you're done).

    Summary requirement depends on your output mode:
    - Curation modes (sources, sources_inline, report): summary is optional --
      your saved items are the deliverable.
    - Inline report mode (report_inline): include a summary -- it becomes your
      report if you haven't written one as your final message.

    Args:
        summary: Brief summary of findings. Optional for curation modes,
                 recommended for report_inline.
    """
    return f"Research finalized. {summary[:500]}" if summary else "Research finalized."


@tool(parse_docstring=True)
async def batch_save_selected(items: list) -> str:
    """Save one or more read items to your curated collection (your final output).

    Reference each item by its [S#] ref + 1-based index from a search result,
    e.g. batch_save_selected(items=[{"ref": "S1", "index": 2, "reason": "why it matters"}).
    You may save a single item or many in one call.

    - You MUST read each item first (via the platform read tool or fetch_urls).
    - Give each a SPECIFIC reason — why it matters for the research, not just
      "relevant" (e.g. "quantifies the 8x request cut with cancellation data").
    - In curation mode, read broadly first and save your BEST items near the END
      (you may read and save in the same turn — reads run before saves). Don't
      save everything you read; curate for quality and coverage.
    - Saves are capped — choose the highest-value items.

    Args:
        items: List of objects, each {"ref": "S1", "index": 2, "reason": "why it matters"}
    """
    return ""  # state write handled by tool_node


# ═════════════════════════════════════════════════════════════════════════
#  URL TEXT FETCH  (kept for potential reuse — not currently wired to any tool)
# ═════════════════════════════════════════════════════════════════════════

async def _fetch_url_text(url: str, timeout: int = 15) -> str:
    """Fetch text content from a URL, stripping HTML if needed.

    Tries Tavily extract first if TAVILY_API_KEY is set, otherwise uses
    direct HTTP GET with basic HTML tag removal.
    """
    tavily_key = get_secret("TAVILY_API_KEY") or ""
    if tavily_key:
        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=tavily_key)
            extract_result = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: client.extract(urls=[url]),
            )
            if extract_result and extract_result.get("results"):
                for r in extract_result["results"]:
                    if r.get("url") == url:
                        return r.get("raw_content", "") or ""
            if extract_result.get("failed_results"):
                logger.debug(f"Tavily extract failed for {url}, falling back to HTTP GET")
        except Exception as e:
            logger.debug(f"Tavily extract error for {url}: {e}")

    # Fallback: direct HTTP GET with HTML tag stripping
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 DeepResearchAgent/1.0"
    }
    loop = asyncio.get_running_loop()
    response = await loop.run_in_executor(
        None,
        lambda: requests.get(url, headers=headers, timeout=timeout),
    )
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "")
    text = response.text
    if "text/html" in content_type or text.strip().startswith("<"):
        text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text)
        text = text.strip()
    return text
