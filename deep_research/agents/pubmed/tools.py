"""PubMed tools (NCBI E-utilities): search, batch-read, full-text cascade.

Search and metadata via NCBI esearch/efetch/elink. Full text attempted
through a cascade: DOI scrape → PMC open-access → Unpaywall → Tavily.

The batch read tool (read_pubmed_articles) is a stub intercepted by the
platform agent engine's tool_node.
"""

import asyncio
import logging
import time
from typing import Literal

import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool

from deep_research.secrets import get_secret

logger = logging.getLogger(__name__)

_BLUE = "\033[94m"
_RESET = "\033[0m"

NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
NCBI_TIMEOUT = 15
NCBI_TOOL_NAME = "KiyosiDeepResearch"


def _get_ncbi_email() -> str:
    return get_secret("PUBMED_EMAIL") or "user@example.com"


def _ncbi_search(query: str, max_results: int = 20, sort: str = "relevance",
                 year_from: int = 0, year_to: int = 0, database: str = "pubmed") -> dict:
    """NCBI esearch — returns {'pmids': [...], 'total_count': N}."""
    params = {
        "db": database, "term": query, "retmax": min(max_results, 100),
        "retmode": "json", "sort": "pub_date" if sort == "date" else sort,
        "email": _get_ncbi_email(), "tool": NCBI_TOOL_NAME,
    }
    if year_from:
        params["mindate"] = str(year_from)
        params["datetype"] = "pdat"
    if year_to:
        params["maxdate"] = str(year_to)
        params["datetype"] = "pdat"

    for attempt in range(3):
        try:
            resp = requests.get(f"{NCBI_BASE}/esearch.fcgi", params=params, timeout=NCBI_TIMEOUT)
            if resp.status_code in (429, 503):
                time.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            data = resp.json()
            idlist = data.get("esearchresult", {}).get("idlist", [])
            count = int(data.get("esearchresult", {}).get("count", 0))
            return {"pmids": [str(i) for i in idlist], "total_count": count}
        except Exception:
            if attempt < 2:
                time.sleep(1)
                continue
            raise
    return {"pmids": [], "total_count": 0}


def _ncbi_fetch_details(pmid: str) -> dict | None:
    """NCBI efetch — returns full article metadata for a single PMID."""
    import xml.etree.ElementTree as ET
    params = {"db": "pubmed", "id": pmid, "retmode": "xml",
              "email": _get_ncbi_email(), "tool": NCBI_TOOL_NAME}
    try:
        resp = requests.get(f"{NCBI_BASE}/efetch.fcgi", params=params, timeout=NCBI_TIMEOUT)
        resp.raise_for_status()
    except Exception:
        return None
    root = ET.fromstring(resp.text)
    article = root.find(".//PubmedArticle")
    if article is None:
        return {"pmid": pmid, "title": "Not found", "abstract": "", "doi": "", "doi_url": ""}

    med = article.find(".//MedlineCitation")
    art = med.find(".//Article") if med is not None else None
    title_el = art.find(".//ArticleTitle") if art is not None else None
    title = "".join(title_el.itertext()).strip() if title_el is not None else "Untitled"

    abstract_parts = []
    abs_el = art.find(".//Abstract") if art is not None else None
    if abs_el is not None:
        for at in abs_el.findall("AbstractText"):
            label = at.get("Label", "")
            text = "".join(at.itertext()).strip()
            abstract_parts.append(f"{label}: {text}" if label else text)
    abstract = "\n".join(abstract_parts)

    authors = []
    author_list = art.find(".//AuthorList") if art is not None else None
    if author_list is not None:
        for au in author_list.findall("Author"):
            ln = au.findtext("LastName", "")
            fn = au.findtext("ForeName", "")
            if ln:
                authors.append(f"{ln} {fn}".strip())

    journal_el = art.find(".//Journal/Title") if art is not None else None
    journal = journal_el.text.strip() if journal_el is not None and journal_el.text else ""
    year_el = art.find(".//Journal/JournalIssue/PubDate/Year") if art is not None else None
    year = year_el.text.strip() if year_el is not None and year_el.text else ""

    doi_el = art.find(".//ArticleId[@IdType='doi']") if art is not None else None
    doi = doi_el.text.strip() if doi_el is not None and doi_el.text else ""

    return {
        "pmid": pmid, "title": title[:500], "abstract": abstract[:5000],
        "authors": authors[:15], "journal": journal, "year": year, "doi": doi,
        "doi_url": f"https://doi.org/{doi}" if doi else "",
        "pubmed_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
    }


def _ncbi_fetch_bulk(pmids: list) -> list:
    """NCBI efetch bulk — lighter metadata for multiple PMIDs."""
    import xml.etree.ElementTree as ET
    params = {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml",
              "email": _get_ncbi_email(), "tool": NCBI_TOOL_NAME}
    resp = requests.get(f"{NCBI_BASE}/efetch.fcgi", params=params, timeout=NCBI_TIMEOUT * 2)
    resp.raise_for_status()
    root = ET.fromstring(resp.text)
    results = []
    for article in root.findall(".//PubmedArticle"):
        med = article.find(".//MedlineCitation")
        art = med.find(".//Article") if med is not None else None
        title_el = art.find(".//ArticleTitle") if art is not None else None
        title = "".join(title_el.itertext()).strip() if title_el is not None else "Untitled"
        abstract = ""
        abs_el = art.find(".//Abstract") if art is not None else None
        if abs_el is not None:
            abstract = " ".join("".join(at.itertext()).strip() for at in abs_el.findall("AbstractText"))
        pmid_el = med.find(".//PMID") if med is not None else None
        pmid_val = pmid_el.text.strip() if pmid_el is not None else ""
        results.append({"pmid": pmid_val, "title": title[:300], "abstract": abstract[:1000]})
    return results


# ── Full-text cascade ──────────────────────────────────────────────────

def _scrape_html(url: str) -> str | None:
    """Scrape full text from a URL using BS4 + publisher-specific selectors."""
    try:
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (compatible; AcademicResearchBot/1.0)"}, timeout=15)
        resp.raise_for_status()
    except Exception:
        return None

    soup = BeautifulSoup(resp.text, "html.parser")
    selectors = ["div#pmc", "div.article-content", "div.c-article-body", "div.Body",
                 "div.article-section__content.enhanced", "article", "div#body", "div#fulltext"]
    for sel in selectors:
        container = soup.select_one(sel)
        if container and len(container.get_text(strip=True)) > 500:
            return container.get_text(separator="\n", strip=True)
    paragraphs = soup.find_all("p")
    text = "\n".join(p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 30)
    if len(text) > 500:
        return text
    return None


def _try_pmc(pmid: str) -> str | None:
    """Check if PMID has a PMC open-access version and fetch full text."""
    params = {"dbfrom": "pubmed", "db": "pmc", "id": pmid, "retmode": "json",
              "email": _get_ncbi_email(), "tool": NCBI_TOOL_NAME}
    try:
        resp = requests.get(f"{NCBI_BASE}/elink.fcgi", params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        linksets = data.get("linksets", [])
        if not linksets or not linksets[0].get("linksetdbs"):
            return None
        pmc_ids = []
        for linkdb in linksets[0]["linksetdbs"]:
            if linkdb.get("linkname") == "pubmed_pmc":
                pmc_ids.extend(linkdb.get("links", []))
        if not pmc_ids:
            return None
        pmcid = str(pmc_ids[0])
        return _scrape_html(f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/")
    except Exception:
        return None


def _try_unpaywall(doi: str) -> str | None:
    """Use Unpaywall API to find an open-access URL, then scrape it."""
    if not doi:
        return None
    try:
        resp = requests.get(f"https://api.unpaywall.org/v2/{doi}",
                            params={"email": _get_ncbi_email()}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return None
    if not data.get("is_oa"):
        return None
    best = data.get("best_oa_location") or {}
    oa_url = best.get("url_for_pdf") or best.get("url") or data.get("oa_locations", [{}])[0].get("url")
    if not oa_url:
        return None
    return _scrape_html(oa_url)


def _try_tavily(url: str) -> str | None:
    """Use Tavily extract to retrieve full text from a resolved URL (fallback)."""
    api_key = get_secret("TAVILY_API_KEY")
    if not api_key:
        return None
    try:
        from tavily import TavilyClient
    except ImportError:
        return None
    try:
        client = TavilyClient(api_key=api_key)
        response = client.extract(urls=[url], include_images=False, extract_depth="advanced")
    except Exception:
        return None
    results = response.get("results", []) if isinstance(response, dict) else []
    for item in results:
        text = item.get("text", "")
        if text and len(text) > 500:
            return text
    return None


def _resolve_doi(doi_url: str) -> str | None:
    """Follow DOI redirect to get the actual publisher URL."""
    try:
        resp = requests.head(doi_url, allow_redirects=True, timeout=10)
        if resp.status_code < 400:
            return str(resp.url)
        return None
    except Exception:
        return None


def _scrape_full_text(details: dict) -> str | None:
    """Full-text cascade: DOI scrape → PMC → Unpaywall → Tavily."""
    doi_url = details.get("doi_url")
    pmid = details.get("pmid")
    doi = details.get("doi")

    if doi_url:
        result = _scrape_html(doi_url)
        if result:
            return result
    if pmid:
        result = _try_pmc(pmid)
        if result:
            return result
    if doi:
        result = _try_unpaywall(doi)
        if result:
            return result
    if doi_url:
        resolved_url = _resolve_doi(doi_url)
        result = _try_tavily(resolved_url or doi_url)
        if result:
            return result
    return None


# ── Tools ──────────────────────────────────────────────────────────────

@tool(parse_docstring=True)
async def search_pubmed(
    query: str,
    max_results: int = 20,
    sort: Literal["relevance", "date"] = "relevance",
    year_from: int = 0,
    year_to: int = 0,
    database: Literal["pubmed", "pmc"] = "pubmed",
    publication_type: str = "",
    mesh_term: str = "",
    substance: str = "",
    affiliation: str = "",
    language: str = "",
    journal: str = "",
    grant: str = "",
    cited_by: str = "",
    related_to: str = "",
) -> str:
    """Search PubMed or PMC for biomedical literature using NCBI E-utilities.

    This is your PRIMARY discovery tool for biomedical evidence. Results carry
    an [S#] handle — read papers with read_pubmed_articles(items=[{"ref": "S1",
    "index": N}]) and save them with batch_save_selected.

    Use it strategically:
    - Run MULTIPLE searches in parallel (one call per angle) to cover the topic
      broadly, then read the most promising hits before searching again.
    - Prefer EVIDENCE over opinion: set publication_type='Meta-Analysis' or
      'Review' (or 'Randomized Controlled Trial') to surface the strongest
      study designs, especially for clinical questions.
    - Use year_from/year_to to bound recency; sort='date' for the newest
      first, sort='relevance' for best match.
    - Boolean syntax works: 'glp-1 AND obesity', 'cancer[Title] AND 2024[pdat]'.
    - cited_by / related_to switch to citation-graph discovery (overrides the
      query) — useful for finding follow-up or foundational work on a known PMID.
    - Each call counts toward your SEARCH budget; don't re-search once capped.

    Args:
        query: Search query (supports Boolean AND/OR/NOT and [field] qualifiers)
        max_results: Number of results (1-100)
        sort: Sort by 'relevance' (default) or 'date' (newest first)
        year_from: Filter from year (0 = any)
        year_to: Filter to year (0 = any)
        database: 'pubmed' (default) or 'pmc'
        publication_type: Filter by publication type (e.g. 'Meta-Analysis', 'Review')
        mesh_term: Filter by MeSH term
        substance: Filter by substance name
        affiliation: Filter by author affiliation
        language: Filter by language (e.g. 'eng')
        journal: Filter by journal name
        grant: Filter by grant number
        cited_by: PMID to find articles that cite it (overrides query)
        related_to: PMID to find articles related to it (overrides query)
    """
    max_results = min(max(1, max_results), 50)
    logger.info(f"{_BLUE}Search PubMed{_RESET}: '{query[:80]}' (sort={sort}, max={max_results})")

    qualifiers = []
    if publication_type:
        qualifiers.append(f"{publication_type}[pt]")
    if mesh_term:
        qualifiers.append(f"{mesh_term}[MeSH Terms]")
    if substance:
        qualifiers.append(f"{substance}[Substance Name]")
    if affiliation:
        qualifiers.append(f"{affiliation}[Affiliation]")
    if language:
        qualifiers.append(f"{language}[Language]")
    if journal:
        qualifiers.append(f"{journal}[Journal]")
    if grant:
        qualifiers.append(f"{grant}[Grant]")
    if qualifiers:
        query = f"({query}) AND ({' AND '.join(qualifiers)})"

    try:
        if cited_by or related_to:
            pmid_arg = cited_by or related_to
            link_name = "pubmed_pubmed_citedin" if cited_by else "pubmed_pubmed"
            params = {"dbfrom": "pubmed", "db": "pubmed", "id": pmid_arg,
                      "linkname": link_name, "retmode": "json",
                      "email": _get_ncbi_email(), "tool": NCBI_TOOL_NAME}
            resp = await asyncio.to_thread(requests.get, f"{NCBI_BASE}/elink.fcgi",
                                           params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            pmids = []
            for linkset in data.get("linksets", []):
                for linkdb in linkset.get("linksetdbs", []):
                    pmids.extend(linkdb.get("links", []))
            search_result = {"pmids": [str(p) for p in pmids[:max_results]], "total_count": len(pmids)}
        else:
            search_result = await asyncio.to_thread(
                _ncbi_search, query, max_results, sort, year_from, year_to, database)
    except Exception as e:
        return f"PubMed search error: {e}"

    pmids = search_result["pmids"][:max_results]
    if not pmids:
        return {"display": f"No PubMed results for: '{query}' (total: {search_result['total_count']})", "items": []}

    try:
        previews = await asyncio.to_thread(_ncbi_fetch_bulk, pmids)
    except Exception as e:
        return {"display": f"Error fetching PubMed previews: {e}", "items": []}

    items = []
    lines = [f"Found {len(previews)} PubMed results for '{query}':", ""]
    for i, p in enumerate(previews, 1):
        title = p['title'][:120]
        pmid = p['pmid']
        lines.append(f"{i}. **{title}**")
        lines.append(f"   PMID: {pmid}  |  PubMed: https://pubmed.ncbi.nlm.nih.gov/{pmid}/")
        if p.get("abstract"):
            lines.append(f"   Abstract: {p['abstract'][:300]}...")
        lines.append("")
        items.append({"id": pmid, "title": title, "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"})
    lines.append("Use read_pubmed_articles(items=[{\"index\": N, \"ref\": \"S#\"}]) to read a paper, then save it with batch_save_selected.")
    return {"display": "\n".join(lines), "items": items}


@tool(parse_docstring=True)
async def read_pubmed_articles(items: list) -> str:
    """Read one or more PubMed papers in a single call.

    Pass each paper by its [S#] ref + 1-based index from a search_pubmed
    output, e.g. read_pubmed_articles(items=[{"ref": "S1", "index": 2}, ...]).
    Returns title, authors, journal, year, abstract, and full text when an
    open-access version can be resolved. Read up to 8 papers per call.

    Prioritize the strongest evidence you found (meta-analyses, RCTs, large
    reviews) and read immediately after each search batch before searching
    again. In curation mode, save the best papers near the END with
    batch_save_selected (read + save can happen in the same turn).

    Args:
        items: List of objects, each {"ref": "S1", "index": 2}.
    """
    return ""  # state write handled by tool_node


async def _fetch_pubmed_article(pmid: str) -> str:
    """Fetch metadata + abstract for a single PubMed paper by PMID."""
    logger.info(f"{_BLUE}Read PubMed article{_RESET}: {pmid}")
    try:
        details = await asyncio.to_thread(_ncbi_fetch_details, pmid)
    except Exception as e:
        return f"Error reading PMID {pmid}: {e}"

    if not details or not details.get("title"):
        return f"No article found for PMID {pmid}."

    authors_str = ", ".join(details.get("authors", [])[:10])
    lines = [
        f"# {details['title']}",
        f"**Authors:** {authors_str}",
        f"**Journal:** {details.get('journal', '?')}  |  **Year:** {details.get('year', '?')}",
        f"**PMID:** {pmid}  |  **URL:** {details.get('pubmed_url', '')}",
    ]
    if details.get("doi"):
        lines.append(f"**DOI:** {details['doi']}  |  {details['doi_url']}")
    lines.append("")
    lines.append("## Abstract")
    lines.append(details.get("abstract", "No abstract available"))
    lines.append("")

    full_text = await asyncio.to_thread(_scrape_full_text, details)
    if full_text:
        lines.append("## Full Text")
        lines.append(full_text)
        lines.append("")

    lines.append("Use batch_save_selected(items=[{\"ref\": \"S#\", \"index\": N, \"reason\": \"...\"}]) to curate this paper.")
    return "\n".join(lines)
