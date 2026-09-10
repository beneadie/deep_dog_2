"""Substack tools: search (Perplexity-backed), batch-read, author credibility.

Search uses Perplexity with a substack.com domain filter as the underlying
engine (never bound raw to this agent). check_author_profile scores a
publication's credibility before saving.

The batch read tool (read_substack_articles) is a stub intercepted by the
platform agent engine's tool_node.
"""

import asyncio
import logging
import re
from typing import Literal

import httpx
from bs4 import BeautifulSoup
from langchain_core.tools import tool

from deep_research.agents.shared.tools import (
    HTTP_USER_AGENT,
    _scrape_generic,
    perplexity_search,
)

logger = logging.getLogger(__name__)

_ORANGE = "\033[38;5;208m"
_RESET = "\033[0m"

VALID_SUBSTACK_RECENCY = {"hour", "day", "week", "month", "year"}


def _extract_substack_items(text: str) -> list[dict]:
    """Parse article URLs out of perplexity_search output for index resolution."""
    items = []
    for line in text.splitlines():
        if not line.startswith("   URL: "):
            continue
        url = line[len("   URL: "):].strip()
        if url:
            items.append({"id": url, "title": url, "url": url})
    return items


@tool(parse_docstring=True)
async def search_substack(
    search_term: str,
    recency: Literal["hour", "day", "week", "month", "year"] = "month",
) -> str:
    """Search Substack newsletters (Perplexity-backed, domain-filtered to substack.com).

    This is your PRIMARY discovery tool for expert opinion and long-form
    analysis on Substack. Results carry an [S#] handle — read articles with
    read_substack_articles(items=[{"ref": "S1", "index": N}]) and save them with
    batch_save_selected.

    Use it strategically:
    - Use simple, specific terms: company names, product names, person names
      (e.g. 'NVIDIA', 'semaglutide', 'Peter Thiel') rather than long questions.
    - Run MULTIPLE searches in parallel for different angles/names, then read
      the most promising hits before searching again.
    - recency: 'month' (default) for current takes; 'year' for broader coverage;
      'week'/'day' for breaking developments.
    - Each call counts toward your SEARCH budget; don't re-search once capped.
    - Before saving a publication, run check_author_profile to filter for
      credible authors.

    Args:
        search_term: Simple search term (e.g. 'NVIDIA', 'Peter Thiel')
        recency: Time filter — 'hour', 'day', 'week', 'month', 'year'
    """
    if recency not in VALID_SUBSTACK_RECENCY:
        recency = "month"
    raw = await perplexity_search.ainvoke({
        "query": search_term,
        "recency": recency,
        "domain_filter": "substack.com",
    })
    if isinstance(raw, str):
        if raw.startswith("Error") or raw.startswith("No results"):
            return {"display": raw, "items": []}
        return {"display": raw, "items": _extract_substack_items(raw)}
    return raw


@tool(parse_docstring=True)
async def read_substack_articles(items: list) -> str:
    """Read one or more Substack articles in a single call.

    Pass each article by its [S#] ref + 1-based index from a search_substack
    output, e.g. read_substack_articles(items=[{"ref": "S1", "index": 2}, ...]).
    Returns the full article body. Read up to 8 articles per call.

    Read the most promising hits immediately after each search batch before
    searching again. In curation mode, save the best articles near the END with
    batch_save_selected (read + save can happen in the same turn). Consider
    check_author_profile before treating an article as authoritative.

    Args:
        items: List of objects, each {"ref": "S1", "index": 2}.
    """
    return ""  # state write handled by tool_node


async def _fetch_substack_article(url: str) -> str:
    """Fetch the full content of a single Substack article."""
    logger.info(f"{_ORANGE}Read Substack article{_RESET}: {url[:80]}...")
    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            resp = await client.get(url, headers={"User-Agent": HTTP_USER_AGENT})
            resp.raise_for_status()
        article = await asyncio.to_thread(_scrape_generic, resp.text)
    except Exception as e:
        return f"Error reading Substack article: {e}"

    content = article["content"]
    return (
        f"# {article['title']}\n"
        f"**Author:** {article['author'] or 'Unknown'}  |  **Date:** {article['date'] or 'Unknown'}\n"
        f"**URL:** {url}\n\n{content}\n\n"
        "Save this article with batch_save_selected(items=[{\"ref\": \"S#\", \"index\": N, \"reason\": \"...\"}]) — use the ref+index from the header above."
    )


@tool(parse_docstring=True)
async def check_author_profile(article_url: str) -> str:
    """Check a Substack publication's credibility via its /about page.

    Returns publication name, author, bio, subscriber count, social links, and
    a quality score (HIGH/MEDIUM/LOW/MINIMAL). Call BEFORE saving to filter for
    credible sources — a large subscriber base and established author are much
    stronger than an anonymous, brand-new newsletter. Does NOT count toward
    your search or read budget.

    Args:
        article_url: Any article URL from the publication (domain is extracted)
    """
    # Normalize Substack's share wrapper form: open.substack.com/pub/<pub>/... → <pub>.substack.com
    open_wrapper = re.search(r"open\.substack\.com/pub/([^/]+)", article_url)
    if open_wrapper:
        domain = open_wrapper.group(1)
        about_url = f"https://{domain}.substack.com/about"
        logger.info(f"{_ORANGE}Check Substack author{_RESET}: {domain}.substack.com")
        return await _fetch_substack_about(about_url, domain)

    domain_match = re.search(r"(https?://)?([^/]*(?:\.substack\.com|substack\.com))", article_url)
    if not domain_match:
        return f"Error: Could not extract Substack domain from URL: {article_url}"
    domain = domain_match.group(2).rstrip(".")
    if domain == "substack.com":
        # A bare substack.com/@user/note/... URL — this is a Note, not a newsletter
        # article. Notes belong to a user, not a publication, so there's no
        # /about page to profile.
        user_match = re.search(r"substack\.com/@([^/]+)", article_url)
        if user_match:
            return (f"**Note by @{user_match.group(1)}** — this is a Substack Note, "
                    f"not a newsletter article. Notes have no publication profile page. "
                    f"QUALITY: MINIMAL (0/5)")
        return f"Error: Not a valid Substack publication URL: {article_url}"
    about_url = f"https://{domain}/about"
    logger.info(f"{_ORANGE}Check Substack author{_RESET}: {domain}")
    return await _fetch_substack_about(about_url, domain)


async def _fetch_substack_about(about_url: str, domain_label: str) -> str:
    """Fetch and score a Substack publication's /about page."""
    try:
        async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
            resp = await client.get(about_url, headers={"User-Agent": HTTP_USER_AGENT})
            resp.raise_for_status()
        soup = await asyncio.to_thread(BeautifulSoup, resp.text, "html.parser")
    except Exception as e:
        return f"Error fetching author profile: {e}"

    pub_name = ""
    for sel in ["h1", "h2"]:
        el = soup.find(sel)
        if el and el.get_text().strip():
            pub_name = " ".join(el.get_text().strip().split())[:120]
            break
    if not pub_name:
        og = soup.find("meta", attrs={"property": "og:title"})
        pub_name = og["content"].strip() if og else domain_label

    author_found = ""
    author_meta = soup.find("meta", attrs={"name": "author"})
    if author_meta:
        author_found = author_meta["content"]
    else:
        text_sample = soup.get_text(" ", strip=True)[:500]
        by_matches = re.findall(r"by\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)", text_sample)
        author_found = by_matches[0] if by_matches else domain_label.split(".substack")[0]

    desc_meta = soup.find("meta", attrs={"name": "description"}) or soup.find("meta", attrs={"property": "og:description"})
    bio = desc_meta["content"].strip() if desc_meta else ""

    subs_count = 0
    text = soup.get_text(" ", strip=True)
    for pat in [r"(\d[\d,]*)\s*subscribers", r"(\d[\d,]*k?)\s*(?:total\s*)?subscriber",
                r"subscriber\w*\s*(\d[\d,]*k?)", r"(\d[\d,]*)\s*(?:people|readers)\s+subscribe"]:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            val = m.group(1).replace(",", "").lower()
            subs_count = int(float(val.replace("k", "")) * 1000) if "k" in val else int(val)
            break

    socials = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        for plat in ["twitter.com", "x.com", "linkedin.com", "github.com", "youtube.com"]:
            if plat in href:
                socials.add(plat)

    signals = [
        bool(pub_name),
        bool(author_found and author_found != "Unknown"),
        bool(bio and len(bio) > 100),
        bool(subs_count > 0),
        bool(len(socials) >= 1),
    ]
    score = sum(signals)
    level = "HIGH" if score >= 4 else ("MEDIUM" if score >= 2 else ("LOW" if score == 1 else "MINIMAL"))

    return (
        f"**{pub_name}**  |  Author: {author_found or 'Unknown'}  |  QUALITY: {level} ({score}/5)\n"
        f"  Subscribers: {subs_count or 'unknown'}  |  Social: {', '.join(sorted(socials)) or 'none'}\n"
        f"  Bio: {bio[:200]}{'...' if len(bio) > 200 else ''}\n"
        f"  URL: {about_url}"
    )
