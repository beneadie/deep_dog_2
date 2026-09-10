

"""Research Utilities and Tools.

This module provides search and content processing utilities for the research agent,
including web search capabilities and content summarization tools.
"""

from pathlib import Path
from typing_extensions import Annotated, List, Literal, Optional
import aiohttp
import asyncio

from langchain_core.tools import tool, InjectedToolArg
from tavily import TavilyClient

from deep_research.config import EXA_SEARCH_MAX_CHARS
from deep_research.observability import log_source
from deep_research.secrets import get_secret
from deep_research.time_utils import get_today_str  # re-export (canonical def lives in time_utils.py)

# ===== UTILITY FUNCTIONS =====

import logging
logger = logging.getLogger(__name__)

# ANSI color codes for tool-specific log highlighting
def extract_text_from_response(content) -> str:
    """
    Safely extract text from a model response, handling different provider formats.

    Different LLM providers return content in different formats:
    - OpenAI: Returns plain string
    - Gemini: May return list of dicts with 'type' and 'text' keys
    - Anthropic: May return list of content blocks

    This function normalizes all formats to a plain string, logging warnings
    for unexpected formats but never discarding data.

    Args:
        content: The .content attribute from a model response

    Returns:
        A plain string with the extracted text content
    """
    # Case 1: Already a string - most common, return as-is
    if isinstance(content, str):
        return content

    # Case 2: None or empty
    if content is None:
        logger.warning("Model returned None content")
        return ""

    # Case 3: List (Gemini structured content, Anthropic content blocks)
    if isinstance(content, list):
        logger.info(f"Model returned list content with {len(content)} items - extracting text")
        extracted_parts = []
        for i, item in enumerate(content):
            if isinstance(item, str):
                extracted_parts.append(item)
            elif isinstance(item, dict):
                # Look for standard text keys used by various providers
                text = item.get('text') or item.get('content') or item.get('value') or item.get('message')
                if text:
                    extracted_parts.append(str(text))
                else:
                    # Unknown dict structure - log and stringify
                    keys = list(item.keys())
                    logger.warning(f"Unknown dict structure in response item {i}, keys: {keys}")
                    # Try to extract meaningful content, skip metadata like 'signature', 'extras'
                    meaningful_keys = [k for k in keys if k not in ('extras', 'signature', 'type', 'metadata')]
                    if meaningful_keys:
                        extracted_parts.append(str({k: item[k] for k in meaningful_keys}))
            else:
                # Unknown type - stringify it
                logger.warning(f"Unexpected type in response list item {i}: {type(item).__name__}")
                extracted_parts.append(str(item))
        return "\n".join(extracted_parts)

    # Case 4: Dictionary (single structured response)
    if isinstance(content, dict):
        logger.info(f"Model returned dict content with keys: {list(content.keys())}")
        text = content.get('text') or content.get('content') or content.get('value') or content.get('message')
        if text:
            return str(text)
        # Fallback: stringify without metadata
        meaningful = {k: v for k, v in content.items() if k not in ('extras', 'signature', 'type', 'metadata')}
        logger.warning(f"Could not find text key in dict, using fallback stringification")
        return str(meaningful) if meaningful else str(content)

    # Case 5: Unknown type - log warning and stringify to preserve data
    logger.warning(f"Unexpected response content type: {type(content).__name__}. Preserving as string.")
    return str(content)


def get_current_dir() -> Path:
    """Get the current directory of the module.

    This function is compatible with Jupyter notebooks and regular Python scripts.

    Returns:
        Path object representing the current directory
    """
    try:
        return Path(__file__).resolve().parent
    except NameError:  # __file__ is not defined
        return Path.cwd()

# ===== CONFIGURATION =====

tavily_client = TavilyClient()
MAX_CONTEXT_LENGTH = 250000
TAVILY_TIMEOUT = 30.0  # 30 seconds for Tavily API calls
EXA_TIMEOUT = 35.0

# ===== SEARCH FUNCTIONS =====

async def tavily_search_multiple(
    search_queries: List[str],
    max_results: int = 3,
    topic: Literal["general", "news", "finance"] = "general",
    days: Optional[int] = None,
    include_raw_content: bool = True,
) -> List[dict]:
    """Perform search using Tavily API for multiple queries.

    Args:
        search_queries: List of search queries to execute
        max_results: Maximum number of results per query
        topic: Topic filter for search results
        days: Limit search to the last N days (for recency). this number can be as big as you want it to be.
        include_raw_content: Whether to include raw webpage content

    Returns:
        List of search result dictionaries
    """
    import time

    # Execute searches sequentially with timeout + retry protection
    search_docs = []
    for query in search_queries:
        search_start = time.time()
        max_retries = 3
        result = None
        for attempt in range(max_retries):
            try:
                # Tavily SDK call is sync, so run in a thread and enforce a hard timeout.
                # Build the client per call so per-run credentials are honoured.
                _key = get_secret("TAVILY_API_KEY")
                _client = TavilyClient(api_key=_key) if _key else tavily_client
                result = await asyncio.wait_for(
                    asyncio.to_thread(
                        _client.search,
                        query,
                        max_results=max_results,
                        include_raw_content=include_raw_content,
                        topic=topic,
                        days=days,
                    ),
                    timeout=TAVILY_TIMEOUT,
                )
                break
            except asyncio.TimeoutError:
                if attempt < max_retries - 1:
                    backoff = 2 * (attempt + 1)
                    logger.warning(
                        f"Tavily search timeout for '{query}' after {TAVILY_TIMEOUT:.1f}s "
                        f"(attempt {attempt + 1}/{max_retries}). Retrying in {backoff}s..."
                    )
                    await asyncio.sleep(backoff)
                else:
                    search_elapsed = time.time() - search_start
                    logger.error(
                        f"Tavily search timed out for '{query}' after {search_elapsed:.2f}s "
                        f"across {max_retries} attempts"
                    )
            except Exception as e:
                if attempt < max_retries - 1:
                    backoff = 2 * (attempt + 1)
                    logger.warning(
                        f"Tavily search failed for '{query}' on attempt {attempt + 1}/{max_retries}: {e}. "
                        f"Retrying in {backoff}s..."
                    )
                    await asyncio.sleep(backoff)
                else:
                    search_elapsed = time.time() - search_start
                    logger.error(
                        f"Tavily search failed for '{query}' after {search_elapsed:.2f}s "
                        f"across {max_retries} attempts: {e}"
                    )

        if result is None:
            # Return empty result to avoid breaking the pipeline.
            search_docs.append({'results': []})
        else:
            search_elapsed = time.time() - search_start
            logger.debug(f"Tavily search for '{query}' completed in {search_elapsed:.2f}s")
            search_docs.append(result)

    return search_docs


def _normalize_search_item(item: dict, provider: str) -> dict:
    """Normalize provider result formats into a shared internal schema."""
    url = item.get("url") or item.get("link") or item.get("source") or ""
    title = item.get("title") or item.get("name") or "Untitled"
    published_date = item.get("published_date") or item.get("date") or item.get("published") or ""

    if provider == "exa":
        # Exa returns the extracted page text in `text`. Full text = `text`;
        # preview = truncated full text.
        raw_content = item.get("text") or item.get("content") or ""
    else:
        # Tavily returns a short `content` snippet plus `raw_content` (full text
        # when include_raw_content=True). Full text = `raw_content`; preview =
        # truncated full text (Tavily has no summary field).
        raw_content = item.get("raw_content") or item.get("text") or item.get("content") or ""

    if raw_content and len(raw_content) > MAX_CONTEXT_LENGTH:
        raw_content = raw_content[:MAX_CONTEXT_LENGTH]

    return {
        "url": url,
        "title": title,
        "raw_content": raw_content,
        "content": raw_content[:1000],
        "published_date": published_date,
        "provider": provider,
    }


async def _exa_search_multiple(search_queries: List[str], max_results: int = 10) -> List[dict]:
    """Execute Exa deep search requests for multiple queries."""
    import os

    api_key = get_secret("EXA_API_KEY")
    if not api_key:
        raise ValueError("EXA_API_KEY not found in environment variables.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    responses = []
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=EXA_TIMEOUT)) as session:
        for query in search_queries:
            payload = {
                "query": query,
                "num_results": max_results,
                "contents": {
                    "text": {
                        "max_characters": EXA_SEARCH_MAX_CHARS,
                    }
                },
            }
            try:
                async with session.post("https://api.exa.ai/search", json=payload, headers=headers) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.warning(f"Exa search error {response.status} for '{query}': {error_text}")
                        responses.append({"results": []})
                        continue
                    data = await response.json()
                    raw_results = data.get("results", []) if isinstance(data, dict) else []
                    normalized = [_normalize_search_item(item, provider="exa") for item in raw_results]
                    responses.append({"results": [r for r in normalized if r.get("url")]})
            except Exception as e:
                logger.warning(f"Exa search failed for '{query}': {e}")
                responses.append({"results": []})

    return responses


def deduplicate_search_results(search_results: List[dict]) -> dict:
    """Deduplicate search results by URL to avoid processing duplicate content.

    Args:
        search_results: List of search result dictionaries

    Returns:
        Dictionary mapping URLs to unique results
    """
    unique_results = {}

    for response in search_results:
        for result in response['results']:
            url = result['url']
            if url not in unique_results:
                unique_results[url] = result

    return unique_results

async def process_search_results(unique_results: dict, tool_name: str = "tavily") -> dict:
    """Build the source list for search results without an LLM summarization pass.

    Returns each search engine result with its own snippet/content so the
    agent can judge relevance and read full text via read_*/fetch_urls when
    it wants depth. Falls back to a truncated slice of raw_content when the
    engine returned no snippet.

    Args:
        unique_results: Dictionary of unique search results

    Returns:
        Dictionary mapping URLs to {title, content} entries
    """
    processed_results = {}

    for url, result in unique_results.items():
        content = (result.get("content") or "").strip()
        full_text = (result.get("raw_content") or "").strip()
        if not content:
            content = (full_text or "")[:MAX_CONTEXT_LENGTH]

        log_source(tool_name=tool_name, link=url, content=content)

        processed_results[url] = {
            'title': result['title'],
            'content': content,
            'full_text': full_text,
        }

    return processed_results

def format_search_output(summarized_results: dict) -> str:
    """Format search results into a well-structured string output.

    Args:
        summarized_results: Dictionary of processed search results

    Returns:
        Formatted string of search results with clear source separation
    """
    if not summarized_results:
        return "No valid search results found. Please try different search queries or use a different search API."

    formatted_output = "Search results: \n\n"

    for i, (url, result) in enumerate(summarized_results.items(), 1):
        formatted_output += f"\n\n--- SOURCE {i}: {result['title']} ---\n"
        formatted_output += f"URL: {url}\n\n"
        formatted_output += f"SUMMARY:\n{result['content']}\n\n"
        formatted_output += "-" * 80 + "\n"

    return formatted_output

# no-op to trigger a refinement of the draft report in the supervisor subgraph.
# The actual refinement is done in the multi_agent_supervisor.supervisor_tools.refine_draft_model tool,
# which is called directly by the supervisor model against the cached conversation prefix.
# This stub keeps the tool registered so the supervisor model can still select it.
@tool(parse_docstring=True)
async def refine_draft_report(research_brief: Annotated[str, InjectedToolArg],
                        findings: Annotated[str, InjectedToolArg],
                        draft_report: Annotated[str, InjectedToolArg],
                        target_language: Annotated[str, InjectedToolArg]):
    """Refine draft report

    Synthesizes all research findings into a comprehensive draft report

    Args:
        research_brief: user's research request
        findings: collected research findings for the user request
        draft_report: draft report based on the findings and user request

    Returns:
        refined draft report
    """

    # Intentional no-op: the supervisor subgraph executes refine_draft_report
    # directly against the cached conversation prefix (refine_draft_model in
    # multi_agent_supervisor.supervisor_tools). This schema-only stub keeps the
    # tool registered so the supervisor model can still select it.
    return ""
