"""arXiv tools (Atom API): search and batch-read preprints.

The batch read tool (read_arxiv_articles) is a stub intercepted by the
platform agent engine's tool_node.
"""

import asyncio
import logging
import time
from typing import Literal

import requests
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

_BLUE = "\033[94m"
_RESET = "\033[0m"

ARXIV_API_BASE = "https://export.arxiv.org/api/query"
ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}


def _arxiv_get(params: dict) -> str:
    """GET the arXiv API with rate-limit retry/backoff. Returns XML text."""
    for attempt in range(3):
        try:
            resp = requests.get(ARXIV_API_BASE, params=params, timeout=30)
            if resp.status_code in (429, 503):
                wait = 3 * (attempt + 1)
                logger.warning(f"arXiv rate limited ({resp.status_code}), retrying in {wait}s")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.text
        except Exception:
            if attempt < 2:
                time.sleep(3)
                continue
            raise
    raise RuntimeError("arXiv API request failed after retries")


def _arxiv_search(query: str, max_results: int = 20, sort_by: str = "relevance",
                  category: str = "", year_from: int = 0, year_to: int = 0) -> list:
    """arXiv Atom API search — returns list of article dicts."""
    import xml.etree.ElementTree as ET
    max_results = min(max(1, max_results), 200)
    params = {"search_query": query, "start": 0, "max_results": max_results,
              "sortBy": sort_by, "sortOrder": "descending"}
    xml_text = _arxiv_get(params)

    root = ET.fromstring(xml_text)
    results = []
    for entry in root.findall("atom:entry", ARXIV_NS):
        id_el = entry.find("atom:id", ARXIV_NS)
        arxiv_id = id_el.text.split("/abs/")[-1] if id_el is not None and id_el.text else ""
        title_el = entry.find("atom:title", ARXIV_NS)
        title = " ".join(title_el.text.strip().split()) if title_el is not None and title_el.text else "Untitled"
        summary_el = entry.find("atom:summary", ARXIV_NS)
        summary = (summary_el.text or "").strip()[:400] if summary_el is not None else ""
        published_el = entry.find("atom:published", ARXIV_NS)
        published = (published_el.text or "")[:10] if published_el is not None else ""
        authors = [a.find("atom:name", ARXIV_NS).text for a in entry.findall("atom:author", ARXIV_NS)
                   if a.find("atom:name", ARXIV_NS) is not None]
        cat_el = entry.find("atom:category", ARXIV_NS)
        cat = cat_el.get("term", "") if cat_el is not None else ""

        year = int(published[:4]) if published else 0
        if year_from and year < year_from:
            continue
        if year_to and year > year_to:
            continue

        results.append({
            "arxiv_id": arxiv_id, "title": title[:300], "summary": summary,
            "authors": authors[:10], "published": published, "category": cat,
            "url": f"https://arxiv.org/abs/{arxiv_id}",
        })
    return results[:max_results]


def _arxiv_fetch_details(arxiv_id: str) -> dict:
    """Fetch full metadata + abstract for a single arXiv ID."""
    import xml.etree.ElementTree as ET
    params = {"id_list": arxiv_id, "max_results": 1}
    xml_text = _arxiv_get(params)

    root = ET.fromstring(xml_text)
    entry = root.find("atom:entry", ARXIV_NS)
    if entry is None:
        return {"arxiv_id": arxiv_id, "title": "Not found", "abstract": ""}

    title_el = entry.find("atom:title", ARXIV_NS)
    title = " ".join(title_el.text.strip().split()) if title_el is not None else "Untitled"
    summary_el = entry.find("atom:summary", ARXIV_NS)
    abstract = (summary_el.text or "").strip() if summary_el is not None else ""
    published_el = entry.find("atom:published", ARXIV_NS)
    published = (published_el.text or "")[:10] if published_el is not None else ""
    authors = [a.find("atom:name", ARXIV_NS).text for a in entry.findall("atom:author", ARXIV_NS)
               if a.find("atom:name", ARXIV_NS) is not None]
    cat_el = entry.find("atom:category", ARXIV_NS)
    category = cat_el.get("term", "") if cat_el is not None else ""

    return {"arxiv_id": arxiv_id, "title": title, "abstract": abstract,
            "authors": authors[:15], "published": published, "category": category}


# ── Tools ──────────────────────────────────────────────────────────────

@tool(parse_docstring=True)
async def search_arxiv(
    query: str,
    max_results: int = 20,
    sort_by: Literal["relevance", "lastUpdatedDate", "submittedDate"] = "relevance",
    category: str = "",
    year_from: int = 0,
    year_to: int = 0,
) -> str:
    """Search arXiv for academic preprints (CS, physics, math, biology, etc.).

    This is your PRIMARY discovery tool for preprints. Results carry an [S#]
    handle — read papers with read_arxiv_articles(items=[{"ref": "S1",
    "index": N}]) and save them with batch_save_selected.

    Use it strategically:
    - Run MULTIPLE searches in parallel (one call per angle or category), then
      read the most promising hits before searching again.
    - category narrows the field — e.g. 'cs.AI', 'cs.LG', 'q-bio.BM', 'physics'.
    - sort_by='submittedDate' or 'lastUpdatedDate' for recency; 'relevance' for
      best match. Use year_from/year_to to bound the time window.
    - Boolean syntax works: 'diffusion AND protein', 'attention NOT transformer'.
    - Each call counts toward your SEARCH budget; don't re-search once capped.

    Args:
        query: Search query (supports arXiv boolean syntax, e.g. 'diffusion AND protein')
        max_results: Max results (1-100)
        sort_by: Sort by 'relevance', 'lastUpdatedDate', or 'submittedDate'
        category: Optional category filter (e.g. 'cs.AI', 'physics', 'q-bio.BM')
        year_from: Filter from year (0 = any)
        year_to: Filter to year (0 = any)
    """
    max_results = min(max(1, max_results), 100)
    logger.info(f"{_BLUE}Search arXiv{_RESET}: '{query[:80]}' (max={max_results})")

    try:
        results = await asyncio.to_thread(_arxiv_search, query, max_results, sort_by, category, year_from, year_to)
    except Exception as e:
        return f"arXiv search error: {e}"

    if not results:
        return {"display": f"No arXiv results for: '{query}'", "items": []}

    items = []
    lines = [f"Found {len(results)} arXiv results for '{query}':", ""]
    for i, r in enumerate(results, 1):
        authors_str = ", ".join(r["authors"][:3])
        et_al = " et al." if len(r.get("authors", [])) > 3 else ""
        title = r['title'][:120]
        arxiv_id = r['arxiv_id']
        lines.append(f"{i}. **{title}**")
        lines.append(f"   Authors: {authors_str}{et_al}")
        lines.append(f"   arXiv ID: {arxiv_id}  |  Date: {r['published']}  |  URL: {r['url']}")
        if r.get("summary"):
            lines.append(f"   Abstract: {r['summary'][:250]}...")
        lines.append("")
        items.append({"id": arxiv_id, "title": title, "url": r['url']})
    lines.append("Use read_arxiv_articles(items=[{\"index\": N, \"ref\": \"S#\"}]) to read a paper, then save it with batch_save_selected.")
    return {"display": "\n".join(lines), "items": items}


@tool(parse_docstring=True)
async def read_arxiv_articles(items: list) -> str:
    """Read one or more arXiv papers in a single call.

    Pass each paper by its [S#] ref + 1-based index from a search_arxiv output,
    e.g. read_arxiv_articles(items=[{"ref": "S1", "index": 2}, ...]). Returns
    title, authors, abstract, category, and date. Read up to 8 papers per call.

    Prioritize the most relevant/impactful preprints and read immediately after
    each search batch before searching again. Note arXiv is NOT peer-reviewed —
    treat strong claims with appropriate skepticism and note that in your
    synthesis. In curation mode, save the best papers near the END with
    batch_save_selected (read + save can happen in the same turn).

    Args:
        items: List of objects, each {"ref": "S1", "index": 2}.
    """
    return ""  # state write handled by tool_node


async def _fetch_arxiv_article(arxiv_id: str) -> str:
    """Fetch metadata + abstract for a single arXiv paper by ID."""
    logger.info(f"{_BLUE}Read arXiv article{_RESET}: {arxiv_id}")
    try:
        details = await asyncio.to_thread(_arxiv_fetch_details, arxiv_id)
    except Exception as e:
        return f"Error reading arXiv article {arxiv_id}: {e}"

    lines = [
        f"# {details['title']}",
        f"**Authors:** {', '.join(details['authors'][:15])}",
        f"**arXiv ID:** {arxiv_id}  |  **Date:** {details['published']}",
        f"**Category:** {details.get('category', 'N/A')}",
        f"**URL:** https://arxiv.org/abs/{arxiv_id}",
        "",
        "## Abstract",
        details.get("abstract", "No abstract available"),
        "",
        "Use batch_save_selected(items=[{\"ref\": \"S#\", \"index\": N, \"reason\": \"...\"}]) to curate this paper.",
    ]
    return "\n".join(lines)
