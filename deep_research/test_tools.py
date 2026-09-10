"""Per-tool smoke tests for the deep research platform agents.

Tests each platform's tools with one canned invocation each (no LLM calls,
no full agent loops). Chained tools derive input from the previous tool's
real output so we exercise genuine data flow.

Platform tests hit the NEW production tool modules (deep_research.agents.*).
Engine-interception tests (curation, index resolution, routing, caps,
fallbacks) run against the shared platform engine in deep_research/agents/base.py.

Usage:
    python test_tools.py --agent reddit          # test one platform's tools
    python test_tools.py --agent meta            # Meta Model API routing + live probe
    python test_tools.py --agent all             # test every platform
    python test_tools.py --agent sec_edgar --tool get_financials
"""

import argparse
import asyncio
import logging
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s")

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = Path(__file__).resolve().parent.name

# ── Platform tools (new production modules) ────────────────────────────
from deep_research.agents.shared.tools import (        # noqa: E402
    _fetch_url_content,
    perplexity_search,
)
from deep_research.agents.reddit.tools import (        # noqa: E402
    search_term_in_subreddit,
    get_subreddit_posts,
    search_subreddits,
    _fetch_reddit_post,
)
from deep_research.agents.pubmed.tools import (        # noqa: E402
    search_pubmed,
    _fetch_pubmed_article,
)
from deep_research.agents.sec_edgar.tools import (     # noqa: E402
    lookup_company,
    get_company_profile,
    search_sec_filings,
    get_latest_filing,
    get_financials,
    get_insider_transactions,
    compare_companies,
    _fetch_filing,
)
from deep_research.agents.arxiv.tools import (         # noqa: E402
    search_arxiv,
    _fetch_arxiv_article,
)
from deep_research.agents.substack.tools import (      # noqa: E402
    search_substack,
    check_author_profile,
    _fetch_substack_article,
)

PASS_MARK = "PASS"
FAIL_MARK = "FAIL"

TOOL_TIMEOUT = 30


def _display(result) -> str:
    """Unwrap a search tool's {'display', 'items'} dict to its display string."""
    if isinstance(result, dict):
        return str(result.get("display", ""))
    return str(result)


def _extract_url(text: str) -> str:
    """Pull the first real http(s) URL out of tool output."""
    for line in _display(text).splitlines():
        m = re.search(r"(https?://\S+)", line)
        if m:
            return m.group(1).rstrip(".,;)")
    return ""


async def run_tool(name: str, coro, results: list) -> None:
    """Run a single tool with a hard timeout, record PASS/FAIL."""
    start = time.perf_counter()
    try:
        out = await asyncio.wait_for(coro, timeout=TOOL_TIMEOUT)
        elapsed = time.perf_counter() - start
        text = _display(out)
        ok = bool(text) and not text.lower().startswith(("error", "failed", "no "))
        status = PASS_MARK if ok else FAIL_MARK
        results.append((name, status, f"{elapsed:.1f}s", text[:90]))
    except asyncio.TimeoutError:
        elapsed = time.perf_counter() - start
        results.append((name, FAIL_MARK, f"{elapsed:.1f}s", f"TIMEOUT after {TOOL_TIMEOUT}s"))
    except Exception as e:
        elapsed = time.perf_counter() - start
        results.append((name, FAIL_MARK, f"{elapsed:.1f}s", f"{type(e).__name__}: {str(e)[:80]}"))


# ── Base tools ──────────────────────────────────────────────────────────

async def test_base(results: list) -> None:
    # _fetch_url_content is the engine behind fetch_urls. Returns (title, body).
    async def _call():
        _, body = await _fetch_url_content("https://example.com")
        return body
    await run_tool("fetch_urls", _call(), results)


# ── Reddit ──────────────────────────────────────────────────────────────

async def test_reddit(results: list) -> None:
    await run_tool("search_term_in_subreddit",
                   search_term_in_subreddit.ainvoke(
                       {"query": "bloom energy", "subreddit": "stocks", "limit": 5,
                        "sort": "top", "time_filter": "year"}), results)
    await run_tool("get_subreddit_posts",
                   get_subreddit_posts.ainvoke({"subreddit": "stocks", "listing": "hot", "limit": 5}),
                   results)
    await run_tool("search_subreddits",
                   search_subreddits.ainvoke({"query": "investing", "limit": 3}), results)

    # Chain: search → grab a real post URL → read it
    search_out = await search_term_in_subreddit.ainvoke(
        {"query": "bloom energy", "subreddit": "stocks", "limit": 3,
         "sort": "top", "time_filter": "year"})
    post_url = _extract_url(search_out)

    if post_url:
        await run_tool("get_reddit_posts", _fetch_reddit_post(post_url), results)
        await run_tool("get_reddit_posts(no_comments)",
                       _fetch_reddit_post(post_url, include_comments=False), results)
    else:
        results.append(("get_reddit_posts", FAIL_MARK, "-", "No post URL found from search"))


# ── PubMed ──────────────────────────────────────────────────────────────

async def test_pubmed(results: list) -> None:
    await run_tool("search_pubmed",
                   search_pubmed.ainvoke({"query": "GLP-1 efficacy", "max_results": 3}), results)

    search_out = await search_pubmed.ainvoke({"query": "GLP-1 efficacy", "max_results": 3})
    pmid = ""
    for line in _display(search_out).splitlines():
        if "PMID:" in line:
            pmid = line.split("PMID:")[1].strip().split()[0]
            break

    if pmid:
        await run_tool("read_pubmed_articles", _fetch_pubmed_article(pmid), results)
    else:
        results.append(("read_pubmed_articles", FAIL_MARK, "-", "No PMID found from search"))


# ── SEC EDGAR ───────────────────────────────────────────────────────────

async def test_sec_edgar(results: list) -> None:
    await run_tool("lookup_company", lookup_company.ainvoke({"query": "NVIDIA"}), results)

    lookup_out = await lookup_company.ainvoke({"query": "NVIDIA"})
    cik = ""
    for line in str(lookup_out).splitlines():
        if "CIK:" in line:
            cik = line.split("CIK:")[1].strip().split()[0]
            break

    if cik:
        await run_tool("get_company_profile",
                       get_company_profile.ainvoke({"cik": cik, "filing_count": 5}), results)
        await run_tool("search_sec_filings",
                       search_sec_filings.ainvoke(
                           {"query": "risk factors", "cik": cik, "form_types": ["10-K"], "limit": 3}),
                       results)
        await run_tool("get_latest_filing",
                       get_latest_filing.ainvoke({"cik": cik, "form_type": "10-K"}), results)
        await run_tool("get_financials",
                       get_financials.ainvoke({"cik": cik, "metric_bundle": "key_metrics"}), results)
        await run_tool("get_insider_transactions",
                       get_insider_transactions.ainvoke({"cik": cik, "days_back": 90}), results)

        filings_out = await search_sec_filings.ainvoke(
            {"query": "risk factors", "cik": cik, "form_types": ["10-K"], "limit": 3})
        filing_url = _extract_url(filings_out)
        if filing_url:
            await run_tool("read_filings", _fetch_filing(filing_url), results)
        else:
            results.append(("read_filings", FAIL_MARK, "-", "No filing URL found from search"))
    else:
        results.append(("get_company_profile", FAIL_MARK, "-", "No CIK found from lookup"))
        results.append(("search_sec_filings", FAIL_MARK, "-", "No CIK found from lookup"))
        results.append(("get_latest_filing", FAIL_MARK, "-", "No CIK found from lookup"))
        results.append(("get_financials", FAIL_MARK, "-", "No CIK found from lookup"))
        results.append(("get_insider_transactions", FAIL_MARK, "-", "No CIK found from lookup"))
        results.append(("read_filings", FAIL_MARK, "-", "No CIK found from lookup"))

    await run_tool("compare_companies",
                   compare_companies.ainvoke({"companies": ["NVDA", "AMD"], "metric_bundle": "key_metrics"}),
                   results)


# ── arXiv ───────────────────────────────────────────────────────────────

async def test_arxiv(results: list) -> None:
    await run_tool("search_arxiv",
                   search_arxiv.ainvoke({"query": "diffusion model protein", "max_results": 3}), results)

    search_out = await search_arxiv.ainvoke({"query": "diffusion model protein", "max_results": 3})
    arxiv_id = ""
    for line in _display(search_out).splitlines():
        if "arXiv ID:" in line:
            arxiv_id = line.split("arXiv ID:")[1].strip().split()[0]
            break

    if arxiv_id:
        await run_tool("read_arxiv_articles", _fetch_arxiv_article(arxiv_id), results)
    else:
        results.append(("read_arxiv_articles", FAIL_MARK, "-", "No arxiv_id found from search"))


# ── Substack ────────────────────────────────────────────────────────────

async def test_substack(results: list) -> None:
    # perplexity_search is the engine behind search_substack (domain-limited)
    await run_tool("perplexity_search",
                   perplexity_search.ainvoke(
                       {"query": "NVIDIA", "recency": "month", "domain_filter": "substack.com"}),
                   results)
    await run_tool("search_substack",
                   search_substack.ainvoke({"search_term": "NVIDIA", "recency": "month"}), results)

    search_out = await search_substack.ainvoke({"search_term": "NVIDIA", "recency": "month"})
    article_url = _extract_url(search_out)

    if article_url:
        await run_tool("read_substack_articles", _fetch_substack_article(article_url), results)
        await run_tool("check_author_profile", check_author_profile.ainvoke({"article_url": article_url}), results)
    else:
        results.append(("read_substack_articles", FAIL_MARK, "-", "No article URL found from search"))
        results.append(("check_author_profile", FAIL_MARK, "-", "No article URL found from search"))


# ── Engine-interception tests (run against the prototype engine until CP5) ──
# NOTE: re-point `ra` to deep_research.agents.base once CP5 installs the engine.

async def test_curation(results: list) -> None:
    from deep_research.agents import base as ra
    from langchain_core.messages import AIMessage

    search_results = {"S1": {"tool": "search_arxiv", "items": [
        {"id": "2301.07041", "title": "Test Paper", "url": "https://arxiv.org/abs/2301.07041"},
    ]}}

    state = {
        "researcher_messages": [AIMessage(content="", tool_calls=[
            {"name": "read_arxiv_articles",
             "args": {"items": [{"index": 1, "ref": "S1"}]}, "id": "tc1"},
            {"name": "batch_save_selected",
             "args": {"items": [{"ref": "S1", "index": 1, "reason": "Key baseline"}]}, "id": "tc2"},
            {"name": "log_finding", "args": {"key": "method", "value": "diffusion"}, "id": "tc3"},
            {"name": "list_saved", "args": {}, "id": "tc4"},
        ])],
        "articles_read": {}, "saved_articles": {}, "findings_log": {},
        "search_results": search_results,
        "agent_type": "arxiv", "search_count": 0,
        "target_language": "English",
        "max_reads": 8, "max_saves": 15, "max_searches": 8, "max_concurrency": 4,
    }
    out = await ra.tool_node(state)
    saved = out.get("saved_articles", {})
    findings = out.get("findings_log", {})
    msgs = [str(m.content) for m in out["researcher_messages"]]
    ok = ("2301.07041" in saved and findings.get("method") == "diffusion"
          and any("Saved" in m for m in msgs) and any("Logged finding" in m for m in msgs))
    results.append(("curation_interception", PASS_MARK if ok else FAIL_MARK, "-",
                    f"saved={len(saved)}, findings={len(findings)}"))

    # read-before-save guard
    state2 = {
        "researcher_messages": [AIMessage(content="", tool_calls=[
            {"name": "batch_save_selected",
             "args": {"items": [{"ref": "S1", "index": 1, "reason": "nope"}]}, "id": "tc5"}])],
        "articles_read": {}, "saved_articles": {}, "findings_log": {},
        "search_results": search_results,
        "agent_type": "arxiv", "search_count": 0,
        "target_language": "English",
        "max_reads": 20, "max_saves": 15, "max_searches": 8, "max_concurrency": 5,
    }
    out2 = await ra.tool_node(state2)
    guard = str(out2["researcher_messages"][0].content)
    guard_ok = "must read this item first" in guard and not out2.get("saved_articles")
    results.append(("read_before_save_guard", PASS_MARK if guard_ok else FAIL_MARK, "-", guard[:70]))


async def test_index_resolution(results: list) -> None:
    """Search → capture [S#] handle → read by index+ref through tool_node."""
    from deep_research.agents import base as ra
    from langchain_core.messages import AIMessage

    state = {
        "researcher_messages": [AIMessage(content="", tool_calls=[
            {"name": "search_arxiv", "args": {"query": "diffusion model protein", "max_results": 3}, "id": "tc_search"},
        ])],
        "articles_read": {}, "saved_articles": {}, "findings_log": {},
        "search_count": 0, "search_results": {},
        "agent_type": "arxiv", "target_language": "English",
        "max_reads": 20, "max_saves": 15, "max_searches": 8, "max_concurrency": 5,
    }
    out = await ra.tool_node(state)
    msgs = out["researcher_messages"]
    search_results = out.get("search_results", {})

    handle = next(iter(search_results), "")
    has_handle = bool(handle) and search_results[handle]["items"]
    prefix_ok = any(f"[{handle}]" in str(m.content) for m in msgs)
    results.append(("index_search_handle", PASS_MARK if (has_handle and prefix_ok) else FAIL_MARK, "-",
                    f"handle={handle or 'NONE'}, items={len(search_results.get(handle, {}).get('items', []))}"))

    if not has_handle:
        return

    arxiv_id = search_results[handle]["items"][0]["id"]
    state2 = {
        "researcher_messages": [AIMessage(content="", tool_calls=[
            {"name": "read_arxiv_articles",
             "args": {"items": [{"index": 1, "ref": handle}]}, "id": "tc_read"},
        ])],
        "articles_read": {}, "saved_articles": {}, "findings_log": {},
        "search_count": 1, "search_results": search_results,
        "agent_type": "arxiv", "target_language": "English",
        "max_reads": 8, "max_saves": 15, "max_searches": 8, "max_concurrency": 4,
    }
    out2 = await ra.tool_node(state2)
    read_ok = (arxiv_id in out2.get("articles_read", {})
               and any("diffusion" in str(m.content).lower() or arxiv_id in str(m.content)
                       for m in out2["researcher_messages"]))
    results.append(("index_read_resolution", PASS_MARK if read_ok else FAIL_MARK, "-",
                    f"resolved to {arxiv_id}"))

    # index without ref → graceful error
    state3 = {
        "researcher_messages": [AIMessage(content="", tool_calls=[
            {"name": "read_arxiv_articles",
             "args": {"items": [{"index": 1}]}, "id": "tc_err"}])],
        "articles_read": {}, "saved_articles": {}, "findings_log": {},
        "search_count": 1, "search_results": search_results,
        "agent_type": "arxiv", "target_language": "English",
        "max_reads": 8, "max_saves": 15, "max_searches": 8, "max_concurrency": 4,
    }
    out3 = await ra.tool_node(state3)
    err_ok = "index requires ref" in str(out3["researcher_messages"][0].content)
    results.append(("index_without_ref_error", PASS_MARK if err_ok else FAIL_MARK, "-",
                    str(out3["researcher_messages"][0].content)[:60]))


async def test_batch_save(results: list) -> None:
    """batch_save_selected resolves ref+index and saves via tool_node."""
    from deep_research.agents import base as ra
    from langchain_core.messages import AIMessage

    base = {
        "articles_read": {}, "saved_articles": {}, "findings_log": {},
        "search_count": 0, "search_results": {},
        "agent_type": "arxiv", "target_language": "English",
        "max_reads": 20, "max_saves": 15, "max_searches": 8, "max_concurrency": 5,
    }

    # Search to populate search_results
    s = dict(base, researcher_messages=[AIMessage(content="", tool_calls=[
        {"name": "search_arxiv", "args": {"query": "diffusion model protein", "max_results": 3}, "id": "tc_s"}])])
    out = await ra.tool_node(s)
    handle = next(iter(out["search_results"]), "")
    if not handle:
        results.append(("batch_save_resolve", FAIL_MARK, "-", "no search handle"))
        return
    arxiv_id = out["search_results"][handle]["items"][0]["id"]

    # Read it first (required before save)
    s2 = dict(base, search_count=1, search_results=out["search_results"],
              researcher_messages=[AIMessage(content="", tool_calls=[
                  {"name": "read_arxiv_articles",
                   "args": {"items": [{"index": 1, "ref": handle}]}, "id": "tc_r"}])])
    out2 = await ra.tool_node(s2)

    # batch_save_selected
    s3 = dict(base, search_count=1, search_results=out["search_results"],
              articles_read=out2["articles_read"],
              researcher_messages=[AIMessage(content="", tool_calls=[
                  {"name": "batch_save_selected",
                   "args": {"items": [{"ref": handle, "index": 1, "reason": "key baseline"}]},
                   "id": "tc_b"}])])
    out3 = await ra.tool_node(s3)
    saved = out3["saved_articles"]
    ok = arxiv_id in saved and saved[arxiv_id]["reason"] == "key baseline"
    results.append(("batch_save_resolve", PASS_MARK if ok else FAIL_MARK, "-",
                    f"saved={arxiv_id}" if ok else "not saved"))

    # batch_save with out-of-range index → graceful per-entry error
    s4 = dict(base, search_count=1, search_results=out["search_results"],
              articles_read=out2["articles_read"],
              researcher_messages=[AIMessage(content="", tool_calls=[
                  {"name": "batch_save_selected",
                   "args": {"items": [{"ref": handle, "index": 99, "reason": "bad"}]},
                   "id": "tc_b2"}])])
    out4 = await ra.tool_node(s4)
    err_ok = "out of range" in str(out4["researcher_messages"][0].content)
    results.append(("batch_save_bad_index", PASS_MARK if err_ok else FAIL_MARK, "-",
                    str(out4["researcher_messages"][0].content)[:60]))


async def test_full_context_report(results: list) -> None:
    """compress_research captures the agent's final message as report in full_context+report."""
    from deep_research.agents import base as ra
    from langchain_core.messages import AIMessage, HumanMessage

    final_report = (
        "# Bloom Energy Sentiment\n\n"
        "Market is divided [1].\n\n"
        "## Sources\n\n[1] Example (https://example.com)\n"
    )
    state = {
        "researcher_messages": [
            HumanMessage(content="Research topic"),
            AIMessage(content="[S1] Found 3 results...", tool_calls=[]),
            AIMessage(content=final_report, tool_calls=[]),
        ],
        "saved_articles": {}, "findings_log": {},
        "research_topic": "Bloom sentiment",
        "model_chain": ["deepseek-v4-flash"],
        "output_mode": "report_inline",
        "max_reads": 20, "max_saves": 15, "max_searches": 8, "max_concurrency": 5,
    }
    out = await ra.compress_research(state)
    ok = "Bloom Energy Sentiment" in out.get("compressed_research", "")
    results.append(("full_context_report_capture", PASS_MARK if ok else FAIL_MARK, "-",
                    "captured" if ok else "missing"))


async def test_final_iteration_routing(results: list) -> None:
    """should_continue sends final-turn tool calls to tool_node; route_after_tools
    enforces the cap AFTER tool execution so batch_save_selected is processed."""
    from deep_research.agents import base as ra
    from langchain_core.messages import AIMessage

    base = {
        "saved_articles": {}, "articles_read": {}, "findings_log": {},
        "search_results": {}, "search_count": 0, "max_iterations": 2,
        "max_reads": 20, "max_saves": 15, "max_searches": 8, "max_concurrency": 5,
    }

    # At the cap, batch_save_selected must route to tool_node (not compress).
    msgs_at_cap = [
        AIMessage(content="", tool_calls=[{"name": "search_arxiv", "args": {"query": "q", "max_results": 1}, "id": "a"}]),
        AIMessage(content="", tool_calls=[{"name": "batch_save_selected", "args": {"items": []}, "id": "b"}]),
    ]
    route = ra.should_continue(dict(base, researcher_messages=msgs_at_cap))
    results.append(("final_iter_should_continue", PASS_MARK if route == "tool_node" else FAIL_MARK, "-",
                    f"route={route}"))

    # After tool_node executes, the cap triggers compress_research.
    post_tools = msgs_at_cap + [AIMessage(content="batch_save_selected: 1 saved.", tool_calls=[])]
    route2 = ra.route_after_tools(dict(base, researcher_messages=post_tools))
    results.append(("final_iter_after_tools", PASS_MARK if route2 == "compress_research" else FAIL_MARK, "-",
                    f"route={route2}"))

    # Under the cap, route_after_tools keeps the loop going.
    under_cap = [AIMessage(content="", tool_calls=[{"name": "search_arxiv", "args": {"query": "q", "max_results": 1}, "id": "c"}])]
    route3 = ra.route_after_tools(dict(base, researcher_messages=under_cap))
    results.append(("under_cap_loop", PASS_MARK if route3 == "llm_call" else FAIL_MARK, "-",
                    f"route={route3}"))


async def test_sources_output(results: list) -> None:
    """compress_research formats sources as markdown list (no LLM) in sources mode."""
    from deep_research.agents import base as ra

    saved = {
        "https://reddit.com/r/stocks/post1": {
            "url": "https://reddit.com/r/stocks/post1",
            "reason": "bullish sentiment thread",
            "content": "Post body text here...",
            "title": "Is Bloom Energy gonna keep blooming",
        },
        "https://reddit.com/r/wsb/post2": {
            "url": "https://reddit.com/r/wsb/post2",
            "reason": "bearish counterpoint",
            "content": "Short thesis discussion...",
            "title": "BE short report analysis",
        },
    }
    articles_read = {k: {"content": v["content"]} for k, v in saved.items()}

    # Without include_article_text → compact (URL + reason, no full text)
    state = {
        "saved_articles": saved, "articles_read": articles_read,
        "findings_log": {}, "researcher_messages": [],
        "research_topic": "Bloom Energy", "agent_type": "reddit",
        "model_chain": ["deepseek-v4-flash"],
        "output_mode": "sources_inline",
        "include_article_text": False,
        "max_reads": 20, "max_saves": 15, "max_searches": 8, "max_concurrency": 5,
    }
    out = await ra.compress_research(state)
    compact = out["compressed_research"]
    ok_compact = ("## 1." in compact and "Why selected:" in compact
                  and "Full text" not in compact)
    results.append(("sources_compact", PASS_MARK if ok_compact else FAIL_MARK, "-",
                    f"has_titles={('Is Bloom' in compact)}"))

    # With include_article_text → full content embedded
    state["include_article_text"] = True
    out2 = await ra.compress_research(state)
    full = out2["compressed_research"]
    ok_full = ("### Full text" in full and "Post body text here" in full)
    results.append(("sources_full_text", PASS_MARK if ok_full else FAIL_MARK, "-",
                    f"has_full={('Full text' in full)}"))

    # Empty saved AND nothing read → clear message
    state["saved_articles"] = {}
    state["articles_read"] = {}
    state["include_article_text"] = False
    out3 = await ra.compress_research(state)
    empty_ok = "No sources were selected" in out3["compressed_research"]
    results.append(("sources_empty", PASS_MARK if empty_ok else FAIL_MARK, "-",
                    out3["compressed_research"][:60]))


async def test_helpers(results: list) -> None:
    """Unit-test engine, Reddit formatting, and module-level helpers."""
    from deep_research.agents import base as ra
    from deep_research.agents.shared.tools import batch_save_selected, finish_research, set_target_language
    from deep_research.agents.reddit.tools import (
        _format_reddit_comments,
        _format_reddit_posts,
        _post_date_and_age,
    )

    search_results = {
        "S1": {"tool": "search_arxiv", "items": [
            {"id": "2401.00001", "title": "Paper A"},
            {"id": "2401.00002", "title": "Paper B"},
        ]},
        "S2": {"tool": "search_arxiv", "items": [
            {"id": "2401.00003", "title": "Paper C"},
        ]},
    }
    articles_read = {"2401.00001": {}, "2401.00003": {}}

    # _find_ref_tag
    tag = ra._find_ref_tag(search_results, "2401.00001")
    results.append(("find_ref_tag", PASS_MARK if tag == "S1 #1" else FAIL_MARK, "-",
                    f"tag={tag}"))

    # _build_read_refs — reverse-maps URL-reads to [S# #i]
    refs = ra._build_read_refs(search_results, articles_read)
    ok_refs = len(refs) == 2 and "S1 #1" in refs[0] and "Paper A" in refs[0]
    results.append(("build_read_refs", PASS_MARK if ok_refs else FAIL_MARK, "-",
                    f"refs={refs}"))

    # _tools_for_turn — final turn full_context_sources restricts to save tools
    all_tools = [batch_save_selected, finish_research, set_target_language]
    restricted = ra._tools_for_turn(all_tools + [ra.fetch_urls], "full_context_sources", True)
    names = {t.name for t in restricted}
    ok_tools = "fetch_urls" not in names and "batch_save_selected" in names
    results.append(("final_tools_restricted", PASS_MARK if ok_tools else FAIL_MARK, "-",
                    f"names={names}"))

    # Non-final turn → all tools
    unrestricted = ra._tools_for_turn(all_tools + [ra.fetch_urls], "full_context_sources", False)
    ok_unres = len(unrestricted) == 4
    results.append(("nonfinal_tools_full", PASS_MARK if ok_unres else FAIL_MARK, "-",
                    f"count={len(unrestricted)}"))

    # Reddit listings retain both an absolute date and relative age.
    date, age = _post_date_and_age(time.time() - 2 * 86400)
    listing = _format_reddit_posts([{
        "date": date, "age": age, "score": 7, "comments": 3,
        "title": "Example post", "url": "https://reddit.com/r/test/example",
    }], "Found 1 post:")
    ok_reddit_age = date in listing and "2 days ago" in listing and "Age" in listing
    results.append(("reddit_post_date_age", PASS_MARK if ok_reddit_age else FAIL_MARK, "-",
                    f"date={date}, age={age}"))

    # Nested Reddit replies retain the author they directly answer.
    comments = [{
        "kind": "t1",
        "data": {
            "author": "alice", "body": "Root comment", "score": 4,
            "replies": {"data": {"children": [{
                "kind": "t1",
                "data": {
                    "author": "bob", "body": "First reply", "score": 2,
                    "replies": {"data": {"children": [{
                        "kind": "t1",
                        "data": {"author": "carol", "body": "Second reply", "score": 1},
                    }]}}
                },
            }]}}
        },
    }]
    comment_text = "\n".join(_format_reddit_comments(comments))
    ok_attribution = (
        "u/bob" in comment_text and "replying to u/alice" in comment_text
        and "u/carol" in comment_text and "replying to u/bob" in comment_text
    )
    results.append(("reddit_reply_attribution", PASS_MARK if ok_attribution else FAIL_MARK, "-",
                    "nested parent authors preserved"))

    # subagent_output_mode — discovery defaults to full-context report_inline,
    # research to the configured sources default; both overridable per call.
    from deep_research.config import subagent_output_mode, SUBAGENT_OUTPUT_MODE, DISCOVERY_OUTPUT_MODE
    ok_disc = subagent_output_mode(discovery=True) == "report_inline"
    ok_res = subagent_output_mode(discovery=False) == SUBAGENT_OUTPUT_MODE
    results.append(("discovery_default_report_inline", PASS_MARK if ok_disc else FAIL_MARK, "-",
                    f"discovery->{subagent_output_mode(True)}"))
    results.append(("research_default_sources", PASS_MARK if ok_res else FAIL_MARK, "-",
                    f"research->{subagent_output_mode(False)}"))
    results.append(("discovery_knob_override", PASS_MARK if DISCOVERY_OUTPUT_MODE else FAIL_MARK, "-",
                    f"DISCOVERY_OUTPUT_MODE={DISCOVERY_OUTPUT_MODE}"))

    # _build_general_tools — GENERAL_AGENT_PLATFORMS filters the general tool set.
    filtered = ra._build_general_tools(["reddit", "pubmed"])
    fnames = {t.name for t in filtered}
    ok_filt = (
        "search_term_in_subreddit" in fnames and "search_pubmed" in fnames
        and not {"search_sec_filings", "search_arxiv", "search_substack",
                 "tavily_search", "exa_deep_search"} & fnames
    )
    results.append(("general_tools_filtered", PASS_MARK if ok_filt else FAIL_MARK, "-",
                    f"names={sorted(n for n in fnames if 'search' in n)}"))
    all_tools = ra._build_general_tools([])
    anames = {t.name for t in all_tools}
    ok_all = {"search_sec_filings", "search_arxiv", "search_substack"} <= anames
    results.append(("general_tools_all_default", PASS_MARK if ok_all else FAIL_MARK, "-",
                    f"count={len(all_tools)}"))
    unknown_ok = len(ra._build_general_tools(["reddit", "bogus_platform"])) > 0
    results.append(("general_tools_unknown_skipped", PASS_MARK if unknown_ok else FAIL_MARK, "-",
                    "unknown key skipped"))


async def test_reasoning_passthrough(results: list) -> None:
    """DeepSeek thinking-mode reasoning_content survives the message round-trip.

    ChatDeepSeek stores the trace in additional_kwargs; the engine must re-emit
    it on follow-up tool-call turns or DeepSeek returns HTTP 400. Regression
    guard for the `getattr(response, "reasoning_content", None)` bug (which
    always returned None) in agents/base.py llm_call.
    """
    import langchain_openai.chat_models.base as lc_base  # noqa: F401
    import deep_research.config  # noqa: F401  (applies the passthrough patch)
    from langchain_core.messages import AIMessage

    # Simulate a ChatDeepSeek response: reasoning in additional_kwargs + tool call.
    response = AIMessage(
        content="",
        tool_calls=[{"name": "tavily_search", "args": {"query": "x"},
                     "id": "call_1", "type": "tool_call"}],
        additional_kwargs={"reasoning_content": "thinking about the search..."},
    )

    # Mirror the llm_call reconstruction; reads the trace from additional_kwargs.
    rebuilt = AIMessage(
        content=response.content if isinstance(response.content, str) else "",
        tool_calls=response.tool_calls if hasattr(response, "tool_calls") else [],
        additional_kwargs={"reasoning_content": response.additional_kwargs.get("reasoning_content")},
    )

    outgoing = lc_base._convert_message_to_dict(rebuilt)
    ok = outgoing.get("reasoning_content") == "thinking about the search..."
    results.append(("reasoning_passthrough", PASS_MARK if ok else FAIL_MARK, "-",
                    f"reasoning={outgoing.get('reasoning_content')!r}"))


async def test_batch_read_cap(results: list) -> None:
    """Batch reads are capped per iteration (max_reads), not per workflow."""
    from deep_research.agents import base as ra
    from langchain_core.messages import AIMessage

    # Two items requested, max_reads=1 per iteration → only the first is read
    # and a cap note is returned.
    state = {
        "researcher_messages": [AIMessage(content="", tool_calls=[
            {"name": "read_arxiv_articles",
             "args": {"items": [{"arxiv_id": "2301.07041"}, {"arxiv_id": "2301.07042"}]},
             "id": "tc_a"}])],
        "articles_read": {}, "saved_articles": {}, "findings_log": {},
        "search_results": {}, "search_count": 0,
        "agent_type": "arxiv", "target_language": "English",
        "max_reads": 1, "max_saves": 15, "max_searches": 8, "max_concurrency": 4,
    }
    out = await ra.tool_node(state)
    n_read = len(out.get("articles_read", {}))
    text = str(out["researcher_messages"][0].content)
    cap_ok = n_read == 1 and "Read cap" in text
    results.append(("batch_read_per_iter_cap", PASS_MARK if cap_ok else FAIL_MARK, "-",
                    f"read={n_read}"))

    # A second iteration has a fresh budget — a new item can be read.
    state2 = {
        "researcher_messages": [AIMessage(content="", tool_calls=[
            {"name": "read_arxiv_articles",
             "args": {"items": [{"arxiv_id": "2301.07042"}]},
             "id": "tc_b"}])],
        "articles_read": out["articles_read"], "saved_articles": {}, "findings_log": {},
        "search_results": {}, "search_count": 0,
        "agent_type": "arxiv", "target_language": "English",
        "max_reads": 1, "max_saves": 15, "max_searches": 8, "max_concurrency": 4,
    }
    out2 = await ra.tool_node(state2)
    n_read2 = len(out2.get("articles_read", {}))
    results.append(("batch_read_fresh_budget", PASS_MARK if n_read2 == 2 else FAIL_MARK, "-",
                    f"read={n_read2}"))

    # fetch_urls has its own tighter cap of 3 per iteration.
    state3 = {
        "researcher_messages": [AIMessage(content="", tool_calls=[
            {"name": "fetch_urls",
             "args": {"urls": [f"http://example.com/u{i}" for i in range(5)]},
             "id": "tc_u"}])],
        "articles_read": {}, "saved_articles": {}, "findings_log": {},
        "search_results": {}, "search_count": 0,
        "agent_type": "reddit", "target_language": "English",
        "max_reads": 8, "max_saves": 15, "max_searches": 8, "max_concurrency": 4,
    }
    out3 = await ra.tool_node(state3)
    n3 = len(out3.get("articles_read", {}))
    text3 = str(out3["researcher_messages"][0].content)
    urlcap_ok = n3 == 3 and "Read cap" in text3
    results.append(("fetch_urls_cap", PASS_MARK if urlcap_ok else FAIL_MARK, "-",
                    f"read={n3}"))


async def test_iteration_counting(results: list) -> None:
    """Housekeeping-only rounds (set_target_language) don't consume research
    iterations; a round mixing language + search counts as one research round."""
    from langchain_core.messages import AIMessage
    from deep_research.agents.base import _iteration_count

    lang_only = [AIMessage(content="", tool_calls=[
        {"name": "set_target_language", "args": {"language": "English"}, "id": "t1"}])]
    c = _iteration_count(lang_only)
    results.append(("iter_lang_only", PASS_MARK if c == 0 else FAIL_MARK, "-",
                    f"count={c}"))

    lang_plus_search = [AIMessage(content="", tool_calls=[
        {"name": "set_target_language", "args": {"language": "English"}, "id": "t1"},
        {"name": "search_term_in_subreddit", "args": {"query": "x"}, "id": "t2"}])]
    c = _iteration_count(lang_plus_search)
    results.append(("iter_lang_plus_search", PASS_MARK if c == 1 else FAIL_MARK, "-",
                    f"count={c}"))

    sequence = lang_only + [
        AIMessage(content="", tool_calls=[
            {"name": "search_term_in_subreddit", "args": {"query": "y"}, "id": "t3"}])]
    c = _iteration_count(sequence)
    results.append(("iter_lang_then_search", PASS_MARK if c == 1 else FAIL_MARK, "-",
                    f"count={c}"))

    finish_only = [AIMessage(content="", tool_calls=[
        {"name": "finish_research", "args": {"summary": "done"}, "id": "t4"}])]
    c = _iteration_count(finish_only)
    results.append(("iter_finish_only", PASS_MARK if c == 0 else FAIL_MARK, "-",
                    f"count={c}"))

    # Forced save round: at the cap with read-but-unsaved items in curation
    # mode, route to one more llm_call; after the round it must finalize.
    from deep_research.agents.base import route_after_tools
    state = {
        "researcher_messages": lang_plus_search,  # _iteration_count == 1
        "max_iterations": 1,
        "output_mode": "report",
        "saved_articles": {},
        "articles_read": {"https://reddit.com/r/x/a": {"content": "c"}},
    }
    r1 = route_after_tools(state)
    results.append(("save_round_granted", PASS_MARK if r1 == "llm_call" else FAIL_MARK, "-",
                    f"route={r1}"))

    state["final_save_round"] = True
    r2 = route_after_tools(state)
    results.append(("save_round_once", PASS_MARK if r2 == "compress_research" else FAIL_MARK, "-",
                    f"route={r2}"))

    state2 = dict(state, final_save_round=False,
                  saved_articles={"https://reddit.com/r/x/a": {"url": "u"}})
    r3 = route_after_tools(state2)
    results.append(("save_round_not_needed", PASS_MARK if r3 == "compress_research" else FAIL_MARK, "-",
                    f"route={r3}"))

    # Schema regression guard: final_save_round must be a DECLARED ResearcherState
    # channel, otherwise LangGraph silently drops the flag and the save round is
    # granted forever (infinite loop beyond the iteration cap).
    from deep_research.state_research import ResearcherState
    ok_schema = "final_save_round" in ResearcherState.__annotations__
    results.append(("save_round_schema_declared", PASS_MARK if ok_schema else FAIL_MARK, "-",
                    f"keys={'final_save_round' if ok_schema else 'MISSING'}"))


async def test_fallbacks(results: list) -> None:
    """Fallbacks fire only when the agent fails to deliver its output."""
    from deep_research.agents import base as ra
    from deep_research.agents.reddit.tools import get_reddit_posts
    from deep_research.agents.shared.tools import finish_research

    articles_read = {
        "https://reddit.com/r/stocks/a": {"content": "post A content"},
        "https://reddit.com/r/stocks/b": {"content": "post B content"},
    }
    search_results = {
        "S1": {"tool": "search_term_in_subreddit", "items": [
            {"id": "https://reddit.com/r/stocks/a", "title": "Post A"},
        ]},
    }

    # _fallback_select_reads picks most-recent reads with the auto reason.
    picked = ra._fallback_select_reads(articles_read, search_results)
    ok_pick = len(picked) == 2 and all(
        "auto-selected" in v["reason"] for v in picked.values())
    results.append(("fallback_select_reads", PASS_MARK if ok_pick else FAIL_MARK, "-",
                    f"count={len(picked)}"))

    # compress_research sources fallback: nothing saved + items read → auto-select.
    state = {
        "saved_articles": {}, "articles_read": articles_read,
        "findings_log": {}, "researcher_messages": [],
        "research_topic": "Bloom", "agent_type": "reddit",
        "model_chain": ["deepseek-v4-flash"],
        "output_mode": "sources_inline",
        "include_article_text": False, "search_results": search_results,
        "max_reads": 8, "max_saves": 15, "max_searches": 8, "max_concurrency": 4,
    }
    out = await ra.compress_research(state)
    saved = out.get("saved_articles", {})
    ok_src = len(saved) == 2 and "auto-selected" in out["compressed_research"]
    results.append(("fallback_sources_autoselect", PASS_MARK if ok_src else FAIL_MARK, "-",
                    f"saved={len(saved)}"))

    # _tools_for_turn: full_context_report final turn → NO tools at all.
    no_tools = ra._tools_for_turn([get_reddit_posts, finish_research], "full_context_report", True)
    ok_nt = no_tools == []
    results.append(("fallback_report_no_tools", PASS_MARK if ok_nt else FAIL_MARK, "-",
                    f"tools={[t.name for t in no_tools]}"))


async def test_citations(results: list) -> None:
    """Citation validator + repair/retry pipeline (citation_utils)."""
    from deep_research.citation_utils import citations_match_sources, ensure_report_citations

    valid = (
        "<CitationPlanList>\n[1] A (https://a.com)\n[2] B (https://b.com)\n</CitationPlanList>\n\n"
        "# Report\n\nClaim one [1]. Claim two [2].\n\n"
        "## Sources\n\n[1] A (https://a.com)\n[2] B (https://b.com)\n"
    )
    results.append(("cit_valid", PASS_MARK if citations_match_sources(valid) else FAIL_MARK, "-", ""))

    missing = (
        "# Report\n\nClaim three [3].\n\n"
        "## Sources\n\n[1] A (https://a.com)\n[2] B (https://b.com)\n"
    )
    results.append(("cit_missing_id", PASS_MARK if not citations_match_sources(missing) else FAIL_MARK, "-", ""))

    noncontig = (
        "# Report\n\nClaim one [1].\n\n## Sources\n\n[1] A (https://a.com)\n[3] C (https://c.com)\n"
    )
    results.append(("cit_noncontiguous", PASS_MARK if not citations_match_sources(noncontig) else FAIL_MARK, "-", ""))

    no_sources = "# Report\n\nClaim one [1].\n"
    results.append(("cit_no_sources", PASS_MARK if not citations_match_sources(no_sources) else FAIL_MARK, "-", ""))

    class _Resp:
        def __init__(self, c): self.content = c
    class _FakeModel:
        def __init__(self, text): self._text = text
        async def ainvoke(self, messages, **kw): return _Resp(self._text)

    repaired = (
        "# Report\n\nClaim one [1]. Claim two [2].\n\n"
        "## Sources\n\n[1] A (https://a.com)\n[2] B (https://b.com)\n"
    )
    out = await ensure_report_citations(missing, "", _FakeModel(repaired))
    ok_repair = "[1]" in out and "<CitationPlanList>" not in out
    results.append(("cit_repair_accepted", PASS_MARK if ok_repair else FAIL_MARK, "-", out[:60]))

    out2 = await ensure_report_citations(missing, "", _FakeModel(missing))
    results.append(("cit_repair_fallback", PASS_MARK if out2 == missing.strip() else FAIL_MARK, "-", out2[:60]))

    stripped = await ensure_report_citations(valid, "", _FakeModel("SHOULD NOT BE CALLED"))
    ok_strip = "<CitationPlanList>" not in stripped and "[1]" in stripped
    results.append(("cit_plan_stripped", PASS_MARK if ok_strip else FAIL_MARK, "-", stripped[:60]))

    # Large-findings truncation path — exercises the sync extractor in the
    # repair flow (regression: a stray `await` crashed this branch).
    from deep_research.citation_utils import llm_repair_citations

    big_findings = (
        "# Note 1\n\nContent.\n\n## Sources\n\n[1] A (https://a.com)\n"
        + "x" * 40000
    )
    big_out = await llm_repair_citations(missing, big_findings, _FakeModel(repaired), timeout=5)
    ok_big = "[1]" in big_out and "<CitationPlanList>" not in big_out
    results.append(("cit_large_findings", PASS_MARK if ok_big else FAIL_MARK, "-", big_out[:60]))

    # ── Registry-based flow (sub-agent report writer) ──────────────────
    from deep_research.citation_utils import extract_cited_ids, finalize_citations

    registry = [
        {"title": "Alpha", "url": "https://a.com"},
        {"title": "Beta", "url": "https://b.com"},
        {"title": "Gamma", "url": "https://c.com"},
    ]

    out = finalize_citations("# Report\n\nClaim two [2] and one [1].\n", registry)
    ok_src = (
        "## Sources" in out
        and "[1] Alpha (https://a.com)" in out
        and "[2] Beta (https://b.com)" in out
        and "https://c.com" not in out  # uncited source excluded
        and "https://b.com" in out
    )
    results.append(("cit_registry_build", PASS_MARK if ok_src else FAIL_MARK, "-", out[:80]))

    out2 = finalize_citations("# Report\n\nClaim [9].\n", registry)
    results.append(("cit_out_of_range", PASS_MARK if "## Sources" not in out2 else FAIL_MARK, "-", out2[:80]))

    out3 = finalize_citations(
        "<CitationPlanList>\n[1] Alpha (https://a.com)\n</CitationPlanList>\n\n"
        "# Report\n\nClaim [1].\n\n## Sources\n\n[1] wrong (https://wrong.com)\n",
        registry,
    )
    ok3 = "https://wrong.com" not in out3 and "https://a.com" in out3
    results.append(("cit_writer_sources_replaced", PASS_MARK if ok3 else FAIL_MARK, "-", out3[:80]))

    ids = extract_cited_ids("# R\n\nA [3] and [1][2] and [1]\n")
    results.append(("cit_extract_dedup", PASS_MARK if ids == [3, 1, 2] else FAIL_MARK, "-", f"ids={ids}"))

    # ── renumber_citations (global id mapping at supervisor boundary) ──
    from deep_research.citation_utils import renumber_citations

    r1 = renumber_citations("Claim [1] and [2].", {1: 3, 2: 4})
    ok_r1 = r1 == "Claim [3] and [4]."
    results.append(("renum_basic", PASS_MARK if ok_r1 else FAIL_MARK, "-", r1))

    r2 = renumber_citations("Claim [1][2].", {1: 5, 2: 6})
    ok_r2 = r2 == "Claim [5][6]."
    results.append(("renum_multi", PASS_MARK if ok_r2 else FAIL_MARK, "-", r2))

    r3 = renumber_citations("Claim [1] and [9].", {1: 7})
    ok_r3 = r3 == "Claim [7] and [9]."  # unmapped [9] untouched
    results.append(("renum_unmapped", PASS_MARK if ok_r3 else FAIL_MARK, "-", r3))

    r4 = renumber_citations("Claim [1].", {})
    ok_r4 = r4 == "Claim [1]."
    results.append(("renum_empty_map", PASS_MARK if ok_r4 else FAIL_MARK, "-", r4))

    # ── _build_source_registry (ordered, dedup, URL-read fallback) ─────
    from deep_research.agents.base import _build_source_registry

    sr = {"S1": {"tool": "search", "items": [
        {"id": "a", "url": "https://a.com", "title": "A"},
        {"id": "b", "url": "https://b.com", "title": "B"},
    ]}, "S2": {"tool": "search", "items": [
        {"id": "a", "url": "https://a.com", "title": "A"},  # duplicate id
        {"id": "c", "url": "https://c.com", "title": "C"},
    ]}}
    ar = {"a": {"content": "x"}, "d": {"content": "y"}}  # "d" not in search_results
    reg = _build_source_registry(sr, ar)
    ok_reg = (
        [e["identifier"] for e in reg] == ["a", "b", "c", "d"]
        and reg[0]["ref"] == "S1 #1"
        and reg[1]["ref"] == "S1 #2"
        and reg[2]["ref"] == "S2 #2"
        and reg[3]["url"] == "d"  # URL-read fallback
        and [e["code"] for e in reg] == ["S1#1", "S1#2", "S2#2", "R1"]
    )
    results.append(("registry_order_dedup", PASS_MARK if ok_reg else FAIL_MARK, "-",
                    f"ids={[e['identifier'] for e in reg]}"))

    # ── Code-based citation flow ────────────────────────────────────────
    from deep_research.citation_utils import extract_cited_codes, remap_codes

    codes = extract_cited_codes("# R\n\nA [S1#2] and [A4-S2#3][S1#2]\n")
    ok_codes = codes == ["S1#2", "A4-S2#3"]
    results.append(("codes_extract_dedup", PASS_MARK if ok_codes else FAIL_MARK, "-", f"codes={codes}"))

    c1 = remap_codes("Claim [S1#2].", {"S1#2": "A4-S1#2"})
    ok_c1 = c1 == "Claim [A4-S1#2]."
    results.append(("codes_local_to_global", PASS_MARK if ok_c1 else FAIL_MARK, "-", c1))

    c2 = remap_codes("Claim [A4-S1#2].", {"A4-S1#2": 1})
    ok_c2 = c2 == "Claim [1]."
    results.append(("codes_to_number", PASS_MARK if ok_c2 else FAIL_MARK, "-", c2))

    c3 = remap_codes("Claim [S9#9].", {"S1#1": "A1-S1#1"})
    ok_c3 = c3 == "Claim [S9#9]."  # unmapped code untouched
    results.append(("codes_unmapped", PASS_MARK if ok_c3 else FAIL_MARK, "-", c3))

    # Code-keyed registry: rebuild Sources keyed by code (renumber=False)
    code_registry = [
        {"code": "S1#1", "title": "Alpha", "url": "https://a.com"},
        {"code": "S1#2", "title": "Beta", "url": "https://b.com"},
        {"code": "S2#1", "title": "Gamma", "url": "https://c.com"},
    ]
    out_codes = finalize_citations("# Report\n\nClaim [S1#2] and [S1#1].\n", code_registry)
    ok_codes_fin = (
        "## Sources" in out_codes
        and "Claim [S1#2] and [S1#1]." in out_codes  # body codes kept
        and "[S1#1] Alpha (https://a.com)" in out_codes
        and "[S1#2] Beta (https://b.com)" in out_codes
        and "https://c.com" not in out_codes  # uncited source excluded
    )
    results.append(("codes_finalize_keep", PASS_MARK if ok_codes_fin else FAIL_MARK, "-", out_codes[:80]))

    # Code-keyed registry with renumber=True: codes -> contiguous [N]
    out_renum = finalize_citations("# Report\n\nClaim [S1#2] and [S1#1].\n", code_registry, renumber=True)
    ok_renum = (
        "Claim [1] and [2]." in out_renum  # first-appearance order
        and "[1] Beta (https://b.com)" in out_renum
        and "[2] Alpha (https://a.com)" in out_renum
    )
    results.append(("codes_finalize_renumber", PASS_MARK if ok_renum else FAIL_MARK, "-", out_renum[:80]))

    # Out-of-registry code: dropped from Sources, body unchanged
    out_oor = finalize_citations("# Report\n\nClaim [S9#9].\n", code_registry)
    ok_oor = "## Sources" not in out_oor and "Claim [S9#9]." in out_oor
    results.append(("codes_out_of_registry", PASS_MARK if ok_oor else FAIL_MARK, "-", out_oor[:80]))

    # ── build_final_registry (merge, dedup, curated C# codes) ──────────
    from deep_research.citation_utils import build_final_registry

    curated_in = [
        {"url": "https://b.com", "title": "Beta", "reason": "dup of registry", "full_text": "beta body"},
        {"url": "https://d.com", "title": "Delta", "reason": "curated only", "full_text": "delta body"},
    ]
    registry_in = [
        {"identifier": "a", "url": "https://a.com", "title": "Alpha", "code": "A4-S1#1", "ref": "S1 #1"},
        {"identifier": "b", "url": "https://b.com", "title": "Beta", "code": "A4-S1#2", "ref": "S1 #2"},
    ]
    merged = build_final_registry(curated_in, registry_in)
    ok_bfr = (
        [e["code"] for e in merged] == ["A4-S1#1", "A4-S1#2", "C1"]
        and merged[2]["url"] == "https://d.com"
        and merged[2]["full_text"] == "delta body"
        and merged[2]["ref"] == "curated only"
    )
    results.append(("build_final_registry", PASS_MARK if ok_bfr else FAIL_MARK, "-",
                    f"codes={[e['code'] for e in merged]}"))

    # Curated codes matched by the code token
    mixed_codes = extract_cited_codes("Claim [A4-S1#1] and [C1] and [A4-S1#2].\n")
    ok_mixed_codes = mixed_codes == ["A4-S1#1", "C1", "A4-S1#2"]
    results.append(("codes_curated_token", PASS_MARK if ok_mixed_codes else FAIL_MARK, "-",
                    f"codes={mixed_codes}"))

    # Mixed global + curated registry: renumber to contiguous [N]
    out_mixed = finalize_citations(
        "# Report\n\nClaim [A4-S1#1] and [C1].\n",
        merged,
        renumber=True,
    )
    ok_mixed = (
        "Claim [1] and [2]." in out_mixed
        and "[1] Alpha (https://a.com)" in out_mixed
        and "[2] Delta (https://d.com)" in out_mixed
        and "https://b.com" not in out_mixed  # uncited curated dup excluded
    )
    results.append(("codes_finalize_mixed", PASS_MARK if ok_mixed else FAIL_MARK, "-", out_mixed[:80]))


async def test_draft_model_config(results: list) -> None:
    """DRAFT_REPORT_MODEL chain resolution + draft writer runnable."""
    import subprocess
    import sys

    from deep_research import config

    # Default: uses its own dedicated chain for the research brief and initial draft.
    ok_default = (
        bool(config.DRAFT_REPORT_MODEL)
        and config.DRAFT_REPORT_MODEL_FALLBACK_CHAIN[0] == config.DRAFT_REPORT_MODEL
        and config.DRAFT_REPORT_MODEL_FALLBACK_CHAIN != config.SUBAGENT_MODEL_FALLBACK_CHAIN
    )
    results.append(("draft_model_dedicated_chain", PASS_MARK if ok_default else FAIL_MARK, "-",
                    str(config.DRAFT_REPORT_MODEL_FALLBACK_CHAIN)))

    # get_draft_report_model returns a runnable
    from deep_research.config import get_draft_report_model
    m = get_draft_report_model(max_tokens=32000)
    ok_run = hasattr(m, "ainvoke")
    results.append(("draft_model_runnable", PASS_MARK if ok_run else FAIL_MARK, "-", type(m).__name__))

    # Env override resolves to its own chain (fresh interpreter to re-read config)
    code = (
        "from dotenv import load_dotenv; load_dotenv();"
        "import deep_research.config as c;"
        "print(c.DRAFT_REPORT_MODEL_FALLBACK_CHAIN)"
    )
    env = dict(os.environ)
    env["DRAFT_REPORT_MODEL"] = "nvidia/nemotron-3.5-lightning"
    repo_root = str(Path(__file__).resolve().parents[1])
    r = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, env=env, cwd=repo_root,
    )
    ok_override = "nvidia/nemotron-3.5-lightning" in r.stdout
    results.append(("draft_model_env_override", PASS_MARK if ok_override else FAIL_MARK, "-",
                    r.stdout.strip() or r.stderr.strip()))


async def test_supervisor_prompt(results: list) -> None:
    """Supervisor prompt .format() renders for every PROMPT_VERSION (no KeyError)."""
    import subprocess
    import sys

    code = (
        "from dotenv import load_dotenv; load_dotenv();"
        "from deep_research.prompts import lead_researcher_with_multiple_steps_diffusion_double_check_prompt as p;"
        "kwargs = dict(date='2026-08-27', max_concurrent_research_units=3, max_concurrent_discovery_units=2,"
        " max_researcher_iterations=10, min_research_time_minutes=5, max_research_time_minutes=15,"
        " example_report='x', target_language='en', available_subagents='_list_');"
        "rendered = p.format(**kwargs);"
        "leftover = [s for s in ('{{claim}}', '{{X}}', '{{subtrack}}') if s in rendered];"
        "print('OK' if not leftover else ('LEFTOVER ' + ','.join(leftover)))"
    )
    repo_root = str(Path(__file__).resolve().parents[1])
    for version in ("OPEN", "LEGACY"):
        env = dict(os.environ)
        env["PROMPT_VERSION"] = version
        r = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, env=env, cwd=repo_root,
        )
        ok = r.returncode == 0 and r.stdout.strip() == "OK"
        results.append((f"supervisor_prompt_format_{version}", PASS_MARK if ok else FAIL_MARK, "-",
                        r.stdout.strip() or r.stderr.strip()))


async def test_language_detection(results: list) -> None:
    """ResearchQuestion schema exposes input_language; transform prompt enforces detection."""
    from deep_research.state_scope import ResearchQuestion

    ok_schema = "input_language" in ResearchQuestion.model_fields
    results.append(("detection_schema_input_language", PASS_MARK if ok_schema else FAIL_MARK, "-",
                    list(ResearchQuestion.model_fields.keys())))

    import subprocess
    import sys

    code = (
        "from dotenv import load_dotenv; load_dotenv();"
        "from deep_research.prompts import transform_messages_into_research_topic_human_msg_prompt as p;"
        "rendered = p.format(messages='<English user message>', date='2026-08-27');"
        "checks = ["
        "  '<Language Detection' in rendered,"
        "  'input_language must name that language exactly' in rendered,"
        "  'IDENTICAL to input_language' in rendered,"
        "  '{{messages}}' not in rendered,"
        "];"
        "print('OK' if all(checks) else 'MISSING')"
    )
    repo_root = str(Path(__file__).resolve().parents[1])
    for version in ("OPEN", "LEGACY"):
        env = dict(os.environ)
        env["PROMPT_VERSION"] = version
        r = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, env=env, cwd=repo_root,
        )
        ok = r.returncode == 0 and r.stdout.strip() == "OK"
        results.append((f"detection_prompt_{version}", PASS_MARK if ok else FAIL_MARK, "-",
                        r.stdout.strip() or r.stderr.strip()))


async def test_subagent_config(results: list) -> None:
    """SUBAGENT_MODEL / SUBAGENT_MODEL_FALLBACK_CHAIN + per-agent override."""
    import subprocess
    import sys

    from deep_research import config

    # SUBAGENT_MODEL is the primary of the default chain.
    ok_primary = (
        config.SUBAGENT_MODEL_FALLBACK_CHAIN[0] == config.SUBAGENT_MODEL
    )
    results.append(("subagent_primary_first", PASS_MARK if ok_primary else FAIL_MARK, "-",
                    str(config.SUBAGENT_MODEL_FALLBACK_CHAIN)))

    # get_subagent_model returns a runnable.
    from deep_research.config import get_subagent_model
    m = get_subagent_model(max_tokens=32000)
    ok_run = hasattr(m, "ainvoke")
    results.append(("subagent_model_runnable", PASS_MARK if ok_run else FAIL_MARK, "-", type(m).__name__))

    # Env override resolves its own chain (fresh interpreter).
    code = (
        "from dotenv import load_dotenv; load_dotenv();"
        "import deep_research.config as c;"
        "print(c.SUBAGENT_MODEL, '|', c.SUBAGENT_MODEL_FALLBACK_CHAIN)"
    )
    env = dict(os.environ)
    env["SUBAGENT_MODEL"] = "deepseek-v4-flash"
    env["SUBAGENT_MODEL_FALLBACK_CHAIN"] = "deepseek-v4-flash,deepseek-v4-pro"
    repo_root = str(Path(__file__).resolve().parents[1])
    r = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, env=env, cwd=repo_root,
    )
    out = r.stdout.strip()
    ok_override = "deepseek-v4-flash | ['deepseek-v4-flash', 'deepseek-v4-pro']" in out
    results.append(("subagent_model_env_override", PASS_MARK if ok_override else FAIL_MARK, "-", out))

    # Per-agent override map injects model_chain into the registry adapter.
    from deep_research.agents import AGENT_REGISTRY
    agent = AGENT_REGISTRY["ResearchWeb"]
    ok_none = agent._model_chain is None
    results.append(("subagent_default_no_override", PASS_MARK if ok_none else FAIL_MARK, "-",
                    str(getattr(agent, "_model_chain", None))))

    # Registry adapters accept a per-agent chain (fresh interpreter with the map set).
    code2 = (
        "from dotenv import load_dotenv; load_dotenv();"
        "import deep_research.config as c;"
        "c.SUBAGENT_MODEL_CHAIN_BY_AGENT['ResearchReddit'] = ['deepseek-v4-pro'];"
        "import deep_research.agents as a;"
        "print(a.AGENT_REGISTRY['ResearchReddit']._model_chain, '|', a.AGENT_REGISTRY['ResearchWeb']._model_chain)"
    )
    r2 = subprocess.run(
        [sys.executable, "-c", code2],
        capture_output=True, text=True, env=env, cwd=repo_root,
    )
    out2 = r2.stdout.strip()
    ok_per_agent = "['deepseek-v4-pro'] | None" in out2
    results.append(("subagent_per_agent_chain", PASS_MARK if ok_per_agent else FAIL_MARK, "-", out2))

    # Missing API key for one chain entry → skipped with warning (no import crash).
    code3 = (
        "import os;"
        "[os.environ.pop(k, None) for k in ['DEEPSEEK_API_KEY','DEEPSEEK_KEY']];"
        "from deep_research.config import get_subagent_model;"
        "m = get_subagent_model();"
        "print(type(m).__name__, '|', getattr(m, 'model_name', '?'))"
    )
    env = dict(os.environ)
    env["SUBAGENT_MODEL_FALLBACK_CHAIN"] = "nvidia/nemotron-3.5-lightning,deepseek-v4-flash"
    env["OPENROUTER_API_KEY"] = "test-openrouter"
    r3 = subprocess.run(
        [sys.executable, "-c", code3],
        capture_output=True, text=True, env=env, cwd=repo_root,
    )
    ok_degrade = r3.returncode == 0 and "nvidia/nemotron-3.5-lightning" in r3.stdout
    results.append(("subagent_missing_key_degrades", PASS_MARK if ok_degrade else FAIL_MARK, "-",
                    (r3.stdout.strip() or r3.stderr.strip())[:120]))

    # No usable model at all → clear ValueError (not an import crash).
    code4 = (
        "import os;"
        "[os.environ.pop(k, None) for k in ['DEEPSEEK_API_KEY','DEEPSEEK_KEY',"
        "'OPENROUTER_API_KEY','MIMO_API_KEY','META_API_KEY','GEMINI_API_KEY',"
        "'GOOGLE_API_KEY','OPENAI_API_KEY','ZHIPUAI_API_KEY']];"
        "import deep_research.config as c;"
        "print(c._model_key_available('deepseek-v4-flash', False))"
    )
    env4 = dict(os.environ)
    r4 = subprocess.run(
        [sys.executable, "-c", code4],
        capture_output=True, text=True, env=env4, cwd=repo_root,
    )
    ok_missing = r4.returncode == 0 and r4.stdout.strip() == "False"
    results.append(("subagent_missing_key_detected", PASS_MARK if ok_missing else FAIL_MARK, "-",
                    (r4.stdout.strip() or r4.stderr.strip())[:120]))


async def test_routing_config(results: list) -> None:
    """Per-role OpenRouter routing + draft reasoning-effort knob."""
    import subprocess

    from deep_research import config
    repo_root = str(Path(__file__).resolve().parents[1])

    # 1. _resolve_route_flag precedence: explicit arg > role flag > global
    ok_explicit = config._resolve_route_flag("", explicit=True) is True
    ok_inherit = config._resolve_route_flag("") == config.ROUTE_VIA_OPENROUTER
    ok_role_off = config._resolve_route_flag("false") is False
    results.append(("route_explicit_wins", PASS_MARK if ok_explicit else FAIL_MARK, "-", ""))
    results.append(("route_inherits_global", PASS_MARK if ok_inherit else FAIL_MARK, "-", str(config.ROUTE_VIA_OPENROUTER)))
    results.append(("route_role_false", PASS_MARK if ok_role_off else FAIL_MARK, "-", ""))

    # 2. Role flags override the global default independently (fresh interpreter)
    code = (
        "from dotenv import load_dotenv; load_dotenv();"
        "import deep_research.config as c;"
        "m = c.get_supervisor_model(max_tokens=32000);"
        "d = c.get_draft_report_model(max_tokens=32000);"
        "print(getattr(m,'openai_api_base',None) or 'native', '|', getattr(d,'openai_api_base',None) or 'native')"
    )
    env = dict(os.environ)
    env.update({
        "ROUTE_VIA_OPENROUTER": "true",
        "SUPERVISOR_ROUTE_VIA_OPENROUTER": "false",
        "DRAFT_ROUTE_VIA_OPENROUTER": "true",
        "DRAFT_REPORT_REASONING_EFFORT": "",
    })
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=repo_root)
    out = r.stdout.strip()
    ok_roles = "native" in out and "https://openrouter.ai/api/v1" in out
    results.append(("route_role_overrides", PASS_MARK if ok_roles else FAIL_MARK, "-", out))

    # 3. Draft reasoning effort attaches reasoning.effort when routed via OpenRouter
    code2 = (
        "from dotenv import load_dotenv; load_dotenv();"
        "import deep_research.config as c;"
        "d = c.get_draft_report_model(max_tokens=32000);"
        "print(getattr(d,'openai_api_base',None), '|', getattr(d,'extra_body',None))"
    )
    env2 = dict(os.environ)
    env2.update({"DRAFT_ROUTE_VIA_OPENROUTER": "true", "DRAFT_REPORT_REASONING_EFFORT": "medium"})
    r2 = subprocess.run([sys.executable, "-c", code2], capture_output=True, text=True, env=env2, cwd=repo_root)
    out2 = r2.stdout.strip()
    ok_effort = "https://openrouter.ai/api/v1" in out2 and "'effort': 'medium'" in out2
    results.append(("draft_effort_medium", PASS_MARK if ok_effort else FAIL_MARK, "-", out2))

    # 4. Invalid effort value raises a clear ValueError
    try:
        config.get_model("deepseek-v4-flash", reasoning_effort="turbo")
        ok_invalid = False
    except ValueError:
        ok_invalid = True
    results.append(("effort_invalid_raises", PASS_MARK if ok_invalid else FAIL_MARK, "-", ""))

    # 5. Effort set + native routing on a native DeepSeek draft → warning + native model
    code3 = (
        "from dotenv import load_dotenv; load_dotenv();"
        "import deep_research.config as c;"
        "d = c.get_draft_report_model(max_tokens=32000);"
        "print(type(d).__name__, '|', getattr(d,'openai_api_base',None))"
    )
    env3 = dict(os.environ)
    env3["ROUTE_VIA_OPENROUTER"] = ""
    env3["DRAFT_ROUTE_VIA_OPENROUTER"] = ""
    env3["DRAFT_REPORT_REASONING_EFFORT"] = "medium"
    env3["DRAFT_REPORT_MODEL"] = "deepseek-v4-flash"
    r3 = subprocess.run([sys.executable, "-c", code3], capture_output=True, text=True, env=env3, cwd=repo_root)
    out3 = r3.stdout.strip()
    ok_native = out3.startswith("ChatDeepSeek") and "reasoning_effort ignored" in r3.stderr
    results.append(("effort_native_warns", PASS_MARK if ok_native else FAIL_MARK, "-", out3))

    # 6. The draft chain is independent from the platform sub-agent chain.
    code4 = (
        "from dotenv import load_dotenv; load_dotenv();"
        "import deep_research.config as c;"
        "print(c.DRAFT_REPORT_MODEL_FALLBACK_CHAIN, '|', c.SUBAGENT_MODEL_FALLBACK_CHAIN)"
    )
    env4 = dict(os.environ)
    env4["ROUTE_VIA_OPENROUTER"] = ""
    env4["DRAFT_ROUTE_VIA_OPENROUTER"] = ""
    env4["DRAFT_REPORT_REASONING_EFFORT"] = ""
    env4["DRAFT_REPORT_MODEL"] = "nvidia/nemotron-3.5-lightning"
    env4.pop("DRAFT_REPORT_MODEL_FALLBACK_CHAIN", None)
    r4 = subprocess.run([sys.executable, "-c", code4], capture_output=True, text=True, env=env4, cwd=repo_root)
    out4 = r4.stdout.strip()
    parts4 = out4.split("|")
    ok_default = (
        len(parts4) == 2
        and "nvidia/nemotron-3.5-lightning" in parts4[0]
        and parts4[0].strip() != parts4[1].strip()
    )
    results.append(("draft_chain_independent", PASS_MARK if ok_default else FAIL_MARK, "-", out4))


async def test_meta(results: list) -> None:
    """Meta Model API (Muse Spark): routing asserts + live connectivity probe.

    Offline checks always run. Live probes hit api.meta.ai only when a real
    META_API_KEY is present, and use raw httpx so the exact HTTP status and
    body are surfaced — endpoint-vs-model-id 404s are reported distinctly.
    """
    import httpx
    from deep_research import config as cfg
    from deep_research.config import (
        META_BASE_URL,
        get_model,
    )
    META_MODEL = "muse-spark-1.2"

    # ── Offline routing ────────────────────────────────────────────────
    try:
        m = get_model(META_MODEL)
        ok_provider = "api.meta.ai" in str(m.openai_api_base)
        results.append(("meta_get_model_native", PASS_MARK if ok_provider else FAIL_MARK, "-",
                        f"model={getattr(m, 'model_name', '?')} base={getattr(m, 'openai_api_base', '?')}"))
    except Exception as e:
        results.append(("meta_get_model_native", FAIL_MARK, "-",
                        f"{type(e).__name__}: {str(e)[:80]}"))

    results.append(("meta_key_available", PASS_MARK if os.getenv("META_API_KEY") else FAIL_MARK, "-",
                    "META_API_KEY " + ("present" if os.getenv("META_API_KEY") else "MISSING")))

    n = get_model("nvidia/nemotron-3.5-lightning")
    extra = getattr(n, "extra_body", None)
    ok_pin = "openrouter" in str(n.openai_api_base) and extra == {"provider": {"order": ["CoreWeave"]}}
    results.append(("meta_search_nemotron_coreweave", PASS_MARK if ok_pin else FAIL_MARK, "-",
                    f"extra={extra}"))

    mu = get_model("muse-spark-1.2-contributor")
    ok_native = "api.meta.ai" in str(mu.openai_api_base)
    results.append(("meta_contributor_native", PASS_MARK if ok_native else FAIL_MARK, "-",
                    f"base={getattr(mu, 'openai_api_base', '?')}"))

    mo = get_model("meta/muse-spark-1.2")
    ok_or = "openrouter" in str(mo.openai_api_base)
    results.append(("meta_get_model_openrouter", PASS_MARK if ok_or else FAIL_MARK, "-",
                    f"base={getattr(mo, 'openai_api_base', '?')}"))

    # ── ROUTE_VIA_OPENROUTER toggle (offline) ─────────────────────────
    _prev = cfg.ROUTE_VIA_OPENROUTER
    try:
        cfg.ROUTE_VIA_OPENROUTER = True
        for name, expect_slug in (
            ("deepseek-v4-flash", "deepseek/deepseek-v4-flash"),
            ("mimo-v2.5-pro", "xiaomi/mimo-v2.5-pro"),
            (f"meta/{META_MODEL}", f"meta/{META_MODEL}"),
        ):
            m = get_model(name)
            ok = "openrouter" in str(m.openai_api_base) and m.model_name == expect_slug
            results.append((f"toggle_{name}", PASS_MARK if ok else FAIL_MARK, "-",
                            f"model={getattr(m, 'model_name', '?')} base={getattr(m, 'openai_api_base', '?')}"))
        m = get_model("deepseek-v4-pro")
        ok_sup = "openrouter" in str(m.openai_api_base) and m.model_name == "deepseek/deepseek-v4-pro"
        results.append(("toggle_supervisor", PASS_MARK if ok_sup else FAIL_MARK, "-",
                        f"model={getattr(m, 'model_name', '?')}"))
        _prev_or = os.environ.get("OPENROUTER_API_KEY")
        os.environ["OPENROUTER_API_KEY"] = "test-key"
        m = get_model("deepseek-v4-flash")
        results.append(("toggle_key_available", PASS_MARK if "openrouter" in str(m.openai_api_base) else FAIL_MARK, "-",
                        "openrouter key suffices"))
        if _prev_or is None:
            os.environ.pop("OPENROUTER_API_KEY", None)
        else:
            os.environ["OPENROUTER_API_KEY"] = _prev_or
    finally:
        cfg.ROUTE_VIA_OPENROUTER = _prev

    # ── Live probes (only with a real key) ─────────────────────────────
    key = os.environ.get("META_API_KEY", "")
    if not key:
        results.append(("meta_live", PASS_MARK, "-", "skipped: META_API_KEY not set"))
        return

    timeout = httpx.Timeout(30.0)
    headers = {"Authorization": f"Bearer {key}"}

    async def _probe(label: str, method: str, path: str, payload: dict | None = None):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                if method == "GET":
                    r = await client.get(f"{META_BASE_URL}{path}", headers=headers)
                else:
                    r = await client.post(f"{META_BASE_URL}{path}", headers=headers, json=payload)
            body = r.text[:200].replace("\n", " ")
            ok = r.status_code == 200
            results.append((label, PASS_MARK if ok else FAIL_MARK,
                            f"{r.elapsed.total_seconds():.1f}s",
                            f"HTTP {r.status_code}: {body}"))
            return r.status_code, r.text
        except Exception as e:
            results.append((label, FAIL_MARK, "-", f"{type(e).__name__}: {str(e)[:120]}"))
            return None, str(e)

    # 1) Models list — definitive: does the endpoint exist and what ids are valid?
    await _probe("meta_models_list", "GET", "/models")

    # 2) OpenAI-compatible chat completion with the configured model.
    status2, _ = await _probe(
        "meta_chat_completions",
        "POST", "/chat/completions",
        {"model": META_MODEL,
         "messages": [{"role": "user", "content": "Reply with the single word: ok"}]},
    )

    # 3) If the OpenAI surface failed, probe the Anthropic messages surface.
    if status2 not in (200, None):
        await _probe(
            "meta_anthropic_messages",
            "POST", "/messages",
            {"model": META_MODEL,
             "messages": [{"role": "user", "content": "Reply with the single word: ok"}]},
        )


PLATFORM_TESTS = {
    "base": test_base,
    "reddit": test_reddit,
    "pubmed": test_pubmed,
    "sec_edgar": test_sec_edgar,
    "arxiv": test_arxiv,
    "substack": test_substack,
    "curation": test_curation,
    "index_resolution": test_index_resolution,
    "batch_save": test_batch_save,
    "full_context_report": test_full_context_report,
    "final_iteration_routing": test_final_iteration_routing,
    "sources_output": test_sources_output,
    "helpers": test_helpers,
    "reasoning": test_reasoning_passthrough,
    "batch_read_cap": test_batch_read_cap,
    "fallbacks": test_fallbacks,
    "citations": test_citations,
    "draft_model_config": test_draft_model_config,
    "language_detection": test_language_detection,
    "supervisor_prompt": test_supervisor_prompt,
    "subagent_config": test_subagent_config,
    "routing_config": test_routing_config,
    "meta": test_meta,
    "iteration_counting": test_iteration_counting,
}


async def main(agent: str) -> None:
    tests_to_run = PLATFORM_TESTS if agent == "all" else {
        agent: PLATFORM_TESTS[agent],
        "curation": test_curation,
    }

    results = []
    for name, test_fn in tests_to_run.items():
        print(f"\n── Testing: {name} ──")
        try:
            await test_fn(results)
        except Exception as e:
            results.append((f"{name}_block", FAIL_MARK, "-", f"{type(e).__name__}: {str(e)[:80]}"))

    print(f"\n\n{'=' * 60}")
    print(f"  RESULTS ({len(results)} checks)")
    print("=" * 60)
    passed = sum(1 for r in results if r[1] == PASS_MARK)
    for name, status, elapsed, detail in results:
        print(f"  [{status}] {name:<28s} {elapsed:<6s} {detail}")
    print(f"\n  {passed}/{len(results)} passed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Per-tool smoke tests")
    parser.add_argument("--agent", "-a", type=str, default="all",
                        choices=list(PLATFORM_TESTS.keys()) + ["all"],
                        help="Platform to test (or 'all')")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    asyncio.run(main(args.agent))
