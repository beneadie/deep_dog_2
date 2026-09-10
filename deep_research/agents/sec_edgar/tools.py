"""SEC EDGAR tools: company lookup, filings search/read, XBRL financials.

Uses the public SEC data APIs (company_tickers.json, submissions, XBRL
companyfacts, EFTS full-text search, and the legacy browse-edgar Atom feed).

The batch read tool (read_filings) is a stub intercepted by the platform
agent engine's tool_node.
"""

import asyncio
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Literal

import httpx
from langchain_core.tools import tool

from deep_research.secrets import get_secret

logger = logging.getLogger(__name__)

_GREEN = "\033[92m"
_RESET = "\033[0m"


def _contact_email() -> str:
    """SEC contact email, resolved per-call so per-run credentials apply."""
    return (get_secret("SEC_EDGAR_CONTACT_EMAIL")
            or get_secret("PUBMED_EMAIL")
            or "user@example.com")


def _sec_user_agent() -> str:
    return f"SECEdgarAgent/1.0 ({_contact_email()})"


def _sec_headers() -> dict:
    return {"User-Agent": _sec_user_agent(), "Accept-Encoding": "gzip, deflate"}


def _efts_headers() -> dict:
    return {"User-Agent": _sec_user_agent(), "Referer": "https://www.sec.gov/edgar/search/",
            "Accept-Encoding": "gzip, deflate"}


SEC_TIMEOUT = 20

TICKER_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_BASE = "https://data.sec.gov/submissions"
XBRL_BASE = "https://data.sec.gov/api/xbrl"
EFTS_URL = "https://efts.sec.gov/LATEST/search-index"
ARCHIVE_BASE = "https://www.sec.gov/Archives/edgar/data"

_company_ticker_cache = None

XBRL_METRIC_BUNDLES: dict = {
    "key_metrics": {
        "label": "Key Metrics",
        "concepts": {
            "Revenue": "RevenueFromContractWithCustomerExcludingAssessedTax",
            "Cost of Revenue": "CostOfRevenue",
            "Gross Profit": "GrossProfit",
            "Operating Income": "OperatingIncomeLoss",
            "Net Income": "NetIncomeLoss",
            "EPS (Basic)": "EarningsPerShareBasic",
            "EPS (Diluted)": "EarningsPerShareDiluted",
            "Total Assets": "Assets",
            "Total Liabilities": "Liabilities",
            "Operating Cash Flow": "NetCashProvidedByUsedInOperatingActivities",
        },
    },
    "profitability": {
        "label": "Profitability Ratios",
        "concepts": {
            "Revenue": "RevenueFromContractWithCustomerExcludingAssessedTax",
            "Gross Profit": "GrossProfit",
            "Operating Income": "OperatingIncomeLoss",
            "Net Income": "NetIncomeLoss",
            "Total Assets": "Assets",
            "Stockholders Equity": "StockholdersEquity",
        },
    },
    "balance_sheet": {
        "label": "Balance Sheet",
        "concepts": {
            "Cash & Equivalents": "CashAndCashEquivalentsAtCarryingValue",
            "Accounts Receivable": "AccountsReceivableNetCurrent",
            "Inventory": "InventoryNet",
            "Property & Equipment": "PropertyPlantAndEquipmentNet",
            "Goodwill": "Goodwill",
            "Total Assets": "Assets",
            "Long-Term Debt": "LongTermDebtNoncurrent",
            "Total Liabilities": "Liabilities",
            "Stockholders Equity": "StockholdersEquity",
        },
    },
    "cash_flow": {
        "label": "Cash Flow",
        "concepts": {
            "Operating Cash Flow": "NetCashProvidedByUsedInOperatingActivities",
            "Capital Expenditures": "PaymentsToAcquirePropertyPlantAndEquipment",
            "Dividends Paid": "PaymentsOfDividends",
            "Stock Buybacks": "PaymentsForRepurchaseOfCommonStock",
            "Financing Cash Flow": "NetCashProvidedByUsedInFinancingActivities",
            "Free Cash Flow": "_calc_fcf",
        },
    },
    "liquidity": {
        "label": "Liquidity & Solvency",
        "concepts": {
            "Current Assets": "AssetsCurrent",
            "Current Liabilities": "LiabilitiesCurrent",
            "Total Assets": "Assets",
            "Total Liabilities": "Liabilities",
            "Stockholders Equity": "StockholdersEquity",
        },
    },
}

FORM_TYPE_LABELS = {
    "10-K": "Annual Report", "10-Q": "Quarterly Report", "8-K": "Current Report",
    "4": "Insider Transaction", "13D": "Beneficial Ownership", "13G": "Beneficial Ownership",
    "S-1": "Registration Statement", "20-F": "Foreign Private Issuer Annual",
    "6-K": "Foreign Private Issuer Report", "DEF 14A": "Proxy Statement",
    "SD": "Conflict Minerals Report",
}


# ── Low-level fetch helpers ────────────────────────────────────────────

async def _load_ticker_map() -> dict:
    global _company_ticker_cache
    if _company_ticker_cache is not None:
        return _company_ticker_cache
    async with httpx.AsyncClient(timeout=SEC_TIMEOUT) as client:
        r = await client.get(TICKER_URL, headers=_sec_headers())
        r.raise_for_status()
        _company_ticker_cache = r.json()
    return _company_ticker_cache


def _format_cik(cik: str | int) -> str:
    return str(cik).zfill(10)


async def _fetch_json(url: str, headers: dict | None = None) -> dict | None:
    for attempt in range(3):
        try:
            async with httpx.AsyncClient(timeout=SEC_TIMEOUT) as client:
                r = await client.get(url, headers=headers or _sec_headers())
                if r.status_code in (429, 503):
                    wait = 2 ** attempt
                    logger.warning(f"SEC rate limited ({r.status_code}), retrying in {wait}s")
                    await asyncio.sleep(wait)
                    continue
                r.raise_for_status()
                return r.json()
        except Exception:
            if attempt < 2:
                await asyncio.sleep(1)
                continue
            return None
    return None


async def _fetch_text(url: str) -> str | None:
    for attempt in range(3):
        try:
            async with httpx.AsyncClient(timeout=SEC_TIMEOUT) as client:
                r = await client.get(url, headers=_sec_headers())
                if r.status_code in (429, 503):
                    await asyncio.sleep(2 ** attempt)
                    continue
                r.raise_for_status()
                return r.text
        except Exception:
            if attempt < 2:
                await asyncio.sleep(1)
                continue
            return None
    return None


def _search_ticker_map(ticker_map: dict, query: str) -> list:
    results = []
    q = query.strip().lower()
    for entry in ticker_map.values():
        ticker = entry.get("ticker", "").lower()
        title = entry.get("title", "").lower()
        cik_str = str(entry.get("cik_str", ""))
        if q == ticker or q in title or q == cik_str:
            results.append({"cik": cik_str.zfill(10), "ticker": entry.get("ticker", ""),
                            "name": entry.get("title", "")})
    if not results:
        for entry in ticker_map.values():
            ticker = entry.get("ticker", "").lower()
            title = entry.get("title", "").lower()
            if q in ticker or q.split()[-1] in title.split():
                results.append({"cik": str(entry.get("cik_str", "")).zfill(10),
                                "ticker": entry.get("ticker", ""), "name": entry.get("title", "")})
    return results[:10]


def _extract_filing_sections(text: str, sections: list[str]) -> dict:
    items = [s.strip().upper() for s in sections]
    item_regex = re.compile(
        r"(ITEM\s+(\d+[A-Za-z]?(?:\.\d+)?)\.?\s*(?:-+|–+|—+)?\s*.*?\n)(.*?)(?=(?:ITEM\s+\d|\Z))",
        re.DOTALL | re.IGNORECASE,
    )
    matches = item_regex.findall(text)
    matched = {}
    for full_header, num, body in matches:
        num_clean = num.upper().strip()
        if num_clean in items:
            matched[num_clean] = f"{full_header.strip()}\n{body.strip()}"
    if not matched:
        for item in items:
            pattern = re.compile(
                rf"ITEM\s+{re.escape(item)}\.?\s*(?:-+|–+|—+)?\s*(.*?)(?=(?:ITEM\s+\d|\Z))",
                re.DOTALL | re.IGNORECASE,
            )
            m = pattern.search(text)
            if m:
                matched[item] = f"ITEM {item}\n{m.group(1).strip()}"
    return matched


# ── XBRL financials ────────────────────────────────────────────────────

async def _fetch_xbrl_facts(cik: str) -> dict | None:
    url = f"{XBRL_BASE}/companyfacts/CIK{_format_cik(cik)}.json"
    return await _fetch_json(url)


def _extract_concept_values(facts_data: dict, concept: str, years: list[int], quarter: int = 0) -> dict:
    us_gaap = facts_data.get("facts", {}).get("us-gaap", {})
    concept_data = us_gaap.get(concept)
    if not concept_data:
        return {"label": concept, "values": {}}
    found = {}
    for unit_name, entries in concept_data.get("units", {}).items():
        for entry in entries:
            fy = entry.get("fy")
            fp = entry.get("fp", "")
            if years and fy not in years:
                continue
            if quarter > 0 and quarter <= 4:
                if fp != f"Q{quarter}":
                    continue
            elif quarter == 0 and fp != "FY":
                continue
            existing = found.get(fy)
            if existing is None or entry.get("end", "") > existing.get("end", ""):
                found[fy] = {"val": entry.get("val"), "end": entry.get("end"),
                             "fp": fp, "unit": unit_name}
    return {"label": concept_data.get("label", concept), "values": found}


def _render_financials_table(cik: str, name: str, bundle_data: dict, bundle_label: str, years: list[int]) -> str:
    if not years:
        years = sorted({int(k) for metric in bundle_data.values() for k in metric.get("values", {}).keys()})
    if not years:
        return "No financial data available."
    years_sorted = sorted(years)
    header = f"{'Metric':<40s}" + "".join(f"{y:<16s}" for y in map(str, years_sorted))
    lines = [f"## {bundle_label} — {name} (CIK: {cik})", "", header, "-" * len(header)]
    for metric_name, values_dict in bundle_data.items():
        vals = values_dict.get("values", {})
        row = f"{metric_name:<40s}"
        for y in years_sorted:
            entry = vals.get(y)
            if entry and isinstance(entry.get("val"), (int, float)):
                row += f"${entry['val']:>12,.0f}  "
            elif entry:
                row += f"{str(entry['val']):>15s} "
            else:
                row += f"{'N/A':>15s} "
        lines.append(row)
    lines.extend(["", "*Values in USD. Annual (FY) data shown.*"])
    return "\n".join(lines)


def _render_comparison_table(companies_data: list, metric_names: list[str], year: int) -> str:
    header = f"{'Metric':<40s}" + "".join(f"{c['name'][:20]:<22s}" for c in companies_data)
    lines = [f"## Peer Comparison — FY {year}", "", header, "-" * len(header)]
    for metric in metric_names:
        row = f"{metric:<40s}"
        for c in companies_data:
            entry = c["bundle_data"].get(metric, {}).get("values", {}).get(year)
            if entry and isinstance(entry.get("val"), (int, float)):
                row += f"${entry['val']:>14,.0f}  "
            elif entry:
                row += f"{str(entry['val']):>15s} "
            else:
                row += f"{'N/A':>15s} "
        lines.append(row)
    lines.extend(["", "*Values in USD.*"])
    return "\n".join(lines)


# ── Company lookup / profile ───────────────────────────────────────────

@tool(parse_docstring=True)
async def lookup_company(query: str) -> str:
    """Resolve a company name or ticker symbol to its SEC CIK and metadata.

    Use this as your FIRST tool when researching a company — you need the CIK
    for nearly every other SEC tool (profile, filings, financials, insider
    trades). Returns name, ticker, CIK, SIC industry, and exchange.

    Args:
        query: Company name or ticker symbol (e.g. 'Apple', 'AAPL', 'NVDA')
    """
    ticker_map = await _load_ticker_map()
    results = _search_ticker_map(ticker_map, query)
    if not results:
        return f"No company found matching '{query}'."

    lines = [f"Found {len(results)} match(es) for '{query}':", ""]
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r['name']}  |  Ticker: {r['ticker']}  |  CIK: {r['cik']}")
        profile = await _fetch_json(f"{SUBMISSIONS_BASE}/CIK{r['cik']}.json")
        if profile:
            extra = []
            sic = profile.get("sicDescription", "")
            exchange = ", ".join(profile.get("exchanges", []))
            fy_end = profile.get("fiscalYearEnd", "")
            if sic:
                extra.append(f"SIC: {sic}")
            if exchange:
                extra.append(f"Exchange: {exchange}")
            if fy_end:
                extra.append(f"Fiscal year end: {fy_end[:2]}/{fy_end[2:]}")
            if extra:
                lines.append(f"   {' | '.join(extra)}")
        lines.append("")
    lines.append("Use get_company_profile(CIK) for full details and filing history.")
    return "\n".join(lines)


@tool(parse_docstring=True)
async def get_company_profile(cik: str, include_financials: bool = False,
                              filing_count: int = 20) -> str:
    """Get a company's profile, metadata, and recent SEC filing history.

    Use after lookup_company (which gives you the CIK) to understand who the
    company is and what it has filed recently. Set include_financials=True to
    also pull a quick financial snapshot in the same call.

    Args:
        cik: CIK number (e.g. '320193' or '0000320193')
        include_financials: If True, also fetch a quick financial snapshot
        filing_count: Number of recent filings to list (max 100)
    """
    cik_padded = _format_cik(cik)
    cik_clean = str(int(cik_padded))

    data = await _fetch_json(f"{SUBMISSIONS_BASE}/CIK{cik_padded}.json")
    if not data:
        return f"Could not retrieve profile for CIK {cik}."

    name = data.get("name", "Unknown")
    tickers = data.get("tickers", [])
    exchanges = data.get("exchanges", [])
    fy_end = data.get("fiscalYearEnd", "")
    address = data.get("address", {})
    alt_names = data.get("formerNames", []) or []

    lines = [
        f"# {name}",
        f"**CIK:** {cik_padded}",
        f"**Tickers:** {', '.join(tickers) if tickers else 'N/A'}",
        f"**Exchanges:** {', '.join(exchanges) if exchanges else 'N/A'}",
        f"**SIC:** {data.get('sicDescription', '')}",
        f"**Fiscal Year End:** {fy_end[:2]}/{fy_end[2:]}" if fy_end else "",
    ]
    if address:
        addr_parts = [address.get(k, "") for k in ("street1", "street2", "city", "stateOrCountry", "zipCode") if address.get(k)]
        if addr_parts:
            lines.append(f"**Address:** {', '.join(addr_parts)}")
    if alt_names:
        former = "; ".join(f"{a.get('name','')} (until {a.get('date','')})" for a in alt_names[:3])
        lines.append(f"**Former Names:** {former}")

    lines.append("")
    filings = data.get("filings", {}).get("recent", {})
    form_list = filings.get("form", [])
    date_list = filings.get("filingDate", [])
    acc_list = filings.get("accessionNumber", [])

    count = min(filing_count, len(form_list))
    lines.append(f"### Recent Filings (showing up to {count})\n")
    lines.append(f"{'Date':<14s} {'Form':<45s} Filing URL")
    lines.append("-" * 100)
    for i in range(count):
        form = form_list[i] if i < len(form_list) else "?"
        date = date_list[i] if i < len(date_list) else "?"
        acc = acc_list[i] if i < len(acc_list) else ""
        label = FORM_TYPE_LABELS.get(form, "")
        disp = f"{form} {label}"[:45] if label else form[:45]
        filing_url = f"https://www.sec.gov/Archives/edgar/data/{cik_clean}/{acc.replace('-','')}/{acc}-index.htm"
        lines.append(f"{date:<14s} {disp:<45s} {filing_url}")
    lines.append(f"\n{count} filings listed.")

    if include_financials:
        lines.append("\n### Quick Financial Snapshot\n")
        facts = await _fetch_xbrl_facts(cik_padded)
        if facts:
            for label, concept in [("Revenue", "RevenueFromContractWithCustomerExcludingAssessedTax"),
                                   ("Net Income", "NetIncomeLoss"),
                                   ("EPS (Basic)", "EarningsPerShareBasic")]:
                kv = _extract_concept_values(facts, concept, [], 0)
                vals = kv.get("values", {})
                if vals:
                    sorted_years = sorted(vals.keys(), reverse=True)[:2]
                    parts = [f"{label}:"]
                    for y in sorted_years:
                        entry = vals[y]
                        if isinstance(entry.get("val"), (int, float)):
                            parts.append(f"FY{y}: ${entry['val']:,.0f}")
                    lines.append("  " + " | ".join(parts))
        else:
            lines.append("  (XBRL financial data not available)")

    return "\n".join(lines)


# ── Filings search / read ──────────────────────────────────────────────

@tool(parse_docstring=True)
async def search_sec_filings(
    query: str,
    form_types: list[str] | None = None,
    cik: str = "",
    sic: str = "",
    items: list[str] | None = None,
    date_from: str = "",
    date_to: str = "",
    exclude_form_types: list[str] | None = None,
    limit: int = 20,
    sort: Literal["relevance", "date"] = "relevance",
) -> str:
    """Search SEC EDGAR filings by full-text (Elasticsearch API).

    This is your PRIMARY discovery tool for filings. Results carry an [S#]
    handle — read filings with read_filings(items=[{"ref": "S1", "index": N}])
    and save them with batch_save_selected.

    Use it strategically:
    - ALWAYS pass the CIK for company-specific research (from lookup_company) —
      full-text search without it returns noise from thousands of filers.
    - form_types narrows to what matters: ['10-K'] (annual), ['10-Q'] (quarterly),
      ['8-K'] (material events), ['4'] (insider trades).
    - items narrows 8-K disclosures, e.g. ['2.02'] for earnings, ['1.01'] for
      material agreements.
    - Use date_from/date_to for a specific period; sort='date' for newest first.
    - Boolean syntax: AND/OR/NOT and "phrase matching" in query.
    - Each call counts toward your SEARCH budget; don't re-search once capped.

    Args:
        query: Search text. Supports AND, OR, NOT and "phrase matches"
        form_types: Form types to include, e.g. ['10-K', '10-Q', '8-K']
        cik: Restrict to a specific company CIK
        sic: Restrict to SIC code / industry
        items: 8-K item numbers, e.g. ['1.01'] for material agreements
        date_from: Start date (YYYY-MM-DD)
        date_to: End date (YYYY-MM-DD)
        exclude_form_types: Form types to exclude
        limit: Max results (1-100)
        sort: 'relevance' or 'date'
    """
    limit = min(max(1, limit), 100)
    params: dict = {"q": query}
    if form_types:
        params["forms"] = ",".join(form_types)
    if cik:
        params["ciks"] = _format_cik(str(int(_format_cik(cik))))
    if date_from or date_to:
        params["dateRange"] = "custom"
    if date_from:
        params["startdt"] = date_from
    if date_to:
        params["enddt"] = date_to
    if items:
        params["items"] = ",".join(items)
    if sic:
        params["sics"] = sic
    if exclude_form_types:
        params["notForms"] = ",".join(exclude_form_types)
    if sort == "date":
        params["order"] = "desc"

    logger.info(f"{_GREEN}Search SEC filings{_RESET}: '{query[:80]}' (limit={limit}, cik={cik or 'any'})")

    try:
        async with httpx.AsyncClient(timeout=SEC_TIMEOUT) as client:
            r = await client.get(EFTS_URL, params=params, headers=_efts_headers())
            r.raise_for_status()
            data = r.json()
    except Exception as e:
        return f"SEC search error: {e}"

    hits = data.get("hits", {}).get("hits", [])
    if not hits:
        return {"display": f"No SEC filings found for: '{query}'", "items": []}

    # EFTS API returns snake_case fields; build the filing URL from adsh + cik.
    def _filing_url(s: dict) -> str:
        adsh = s.get("adsh", "")
        cik_list = s.get("ciks", [])
        if not adsh:
            return s.get("root_filing_url", "") or ""
        cik = str(cik_list[0]).zfill(10) if cik_list else ""
        return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{adsh.replace('-', '')}/{adsh}-index.htm"

    items_out = []
    lines = [f"Found {len(hits)} SEC filings for '{query}':", ""]
    for i, h in enumerate(hits[:limit], 1):
        s = h.get("_source", {})
        display_names = s.get("display_names") or []
        name = display_names[0] if display_names else (s.get("entity_name") or s.get("entityName") or "?")
        form = s.get("form") or (s.get("root_forms") or ["?"])[0]
        filed = s.get("file_date", "?")
        url = _filing_url(s)
        lines.append(f"{i}. **{name}**")
        lines.append(f"   Form: {form}  |  Filed: {filed}")
        lines.append(f"   URL: {url}")
        description = s.get("file_description", "") or s.get("description", "")
        if description:
            lines.append(f"   {description[:200]}")
        lines.append("")
        items_out.append({"id": url, "title": f"{name} {form}", "url": url})
    lines.append("Use read_filings(items=[{\"index\": N, \"ref\": \"S#\"}]) to read a filing, then save it with batch_save_selected.")
    return {"display": "\n".join(lines), "items": items_out}


@tool(parse_docstring=True)
async def get_latest_filing(
    cik: str,
    form_type: str,
    include_amendments: bool = False,
    extracts: list[str] | None = None,
) -> str:
    """Get the most recent filing of a given type for a company.

    Use when you want the LATEST annual/quarterly report or a specific event
    without searching. `extracts` pulls just the sections you care about instead
    of the full text — much more efficient for large filings.

    For 10-K: extracts=['1A']=Risk Factors, ['7']=MD&A, ['8']=Financials.
    For 8-K:  extracts=['1.01']=Material Agreements, ['2.02']=Earnings.
    For 10-Q: same section numbering as 10-K.

    Args:
        cik: CIK of the company
        form_type: Form type (e.g. '10-K', '10-Q', '8-K', '4', '13D')
        include_amendments: Include amended filings (/A)
        extracts: Item/section numbers to extract. Empty returns the full filing text.
    """
    cik_padded = _format_cik(cik)
    cik_clean = str(int(cik_padded))

    params = {"action": "getcompany", "CIK": cik_clean, "type": form_type,
              "owner": "include", "count": "10", "output": "atom"}

    try:
        async with httpx.AsyncClient(timeout=SEC_TIMEOUT) as client:
            r = await client.get("https://www.sec.gov/cgi-bin/browse-edgar", params=params, headers=_sec_headers())
            r.raise_for_status()
            xml_text = r.text
    except Exception as e:
        return f"Failed to fetch latest {form_type}: {e}"

    entries = re.findall(r"<entry>(.*?)</entry>", xml_text, re.DOTALL)
    if not entries:
        return f"No {form_type} filings found for CIK {cik}."

    entry = entries[0]
    filing_date = (re.search(r"<filing-date>([^<]+)", entry) or [None, "?"])[1]
    acc_num = (re.search(r"<accession-number>([^<]+)", entry) or [None, "?"])[1]
    form_name = (re.search(r"<form-name>([^<]+)", entry) or [None, form_type])[1]
    filing_href = (re.search(r"<filing-href>([^<]+)", entry) or [None, ""])[1]
    size = (re.search(r"<size>([^<]+)", entry) or [None, "?"])[1]

    lines = [f"## Latest {form_type} — CIK {cik_padded}", f"**Filing Date:** {filing_date}",
             f"**Description:** {form_name}", f"**Size:** {size}",
             f"**Accession:** {acc_num}", f"**Link:** {filing_href}", ""]

    acc_clean_dir = acc_num.replace("-", "")
    txt_url = f"https://www.sec.gov/Archives/edgar/data/{cik_clean}/{acc_clean_dir}/{acc_num}.txt"

    txt = await _fetch_text(txt_url)
    if not txt:
        lines.append("(Could not retrieve filing text.)")
        return "\n".join(lines)

    if extracts:
        sections_text = _extract_filing_sections(txt, extracts)
        if sections_text:
            lines.append(f"--- Extracted Sections: {', '.join(extracts)} ---\n")
            for sec_id in extracts:
                sec_clean = sec_id.upper().strip()
                content = sections_text.get(sec_clean, f"(Section ITEM {sec_clean} not found)")
                lines.append(f"### ITEM {sec_clean}")
                lines.append(content)
                lines.append("")
        else:
            lines.append("Could not find the requested sections. Returning filing summary:")
            lines.append(txt)
    else:
        words = txt.split()
        lines.append(f"--- Filing ({len(words)} words) ---\n")
        lines.append(txt)

    return "\n".join(lines)


@tool(parse_docstring=True)
async def read_filings(items: list, sections: list[str] | None = None) -> str:
    """Read one or more SEC filings in a single call.

    Pass each filing by its [S#] ref + 1-based index from a search output, e.g.
    read_filings(items=[{"ref": "S1", "index": 2}, ...]). Read up to 8 filings
    per call. `sections` extracts only the specified items — far more efficient
    than the full text for large 10-Ks.

    For 10-K/10-Q: sections=['1A', '7', '8'] = Risk Factors, MD&A, Financials.
    Read the most promising filings immediately after each search batch before
    searching again. In curation mode, save the best filings near the END with
    batch_save_selected (read + save can happen in the same turn).

    Args:
        items: List of objects, each {"ref": "S1", "index": 2}.
        sections: Item numbers to extract (applied to all). Empty returns the full filing text.
    """
    return ""  # state write handled by tool_node


async def _fetch_filing(filing_url: str, sections: list[str] | None = None) -> str:
    """Fetch the text of a single SEC filing, optionally extracting sections."""
    adsh_match = re.search(r"/(\d{10}-\d{2}-\d{6})", filing_url)
    cik_match = re.search(r"/data/(\d+)", filing_url)
    if not adsh_match or not cik_match:
        return "Could not parse CIK or accession number from URL."

    cik_clean = cik_match.group(1)
    adsh = adsh_match.group(1)
    txt_url = f"https://www.sec.gov/Archives/edgar/data/{cik_clean}/{adsh.replace('-','')}/{adsh}.txt"

    txt = await _fetch_text(txt_url)
    if not txt:
        return f"Could not retrieve filing text."

    words = len(txt.split())
    logger.info(f"{_GREEN}Read SEC filing{_RESET}: {filing_url[:60]}... ({words} words)")

    if not sections:
        return f"--- Filing ({words} total words) ---\n\n{txt}"

    extracted = _extract_filing_sections(txt, sections)
    if not extracted:
        return f"Could not find sections {sections} in filing.\n\n{txt}"

    lines = [f"--- Filing ({words} words) ---", f"--- Extracted: {', '.join(sections)} ---\n"]
    for sec_id in sections:
        sec_clean = sec_id.upper().strip()
        content = extracted.get(sec_clean, f"(Section {sec_clean} not found)")
        lines.append(f"### ITEM {sec_clean}\n")
        lines.append(content)
        lines.append("")
    return "\n".join(lines)


# ── Financials ─────────────────────────────────────────────────────────

@tool(parse_docstring=True)
async def get_financials(
    cik: str,
    metric_bundle: Literal["key_metrics", "profitability", "balance_sheet", "cash_flow", "liquidity"] = "key_metrics",
    years: list[int] | None = None,
    quarter: int = 0,
) -> str:
    """Get structured financial statement data from XBRL filings.

    Use for quantitative analysis — revenue, margins, cash flow, balance-sheet
    health — without parsing raw filings. Choose the bundle that matches your
    question:

      key_metrics — Revenue, Gross Profit, Op Income, Net Income, EPS, Assets, Liabilities, OCF
      profitability — Revenue, Gross/Operating/Net margins, ROA, ROE
      balance_sheet — Cash, Receivables, Inventory, PP&E, Debt, Equity
      cash_flow — OCF, CapEx, FCF, Dividends, Buybacks
      liquidity — Current ratio, Debt/Equity, Working Capital

    Use quarter=1..4 for quarterly data, 0 for annual. `years` picks fiscal years
    (empty = last 2 available). Does NOT count toward your read budget.

    Args:
        cik: CIK number
        metric_bundle: Which metric set to return
        years: Fiscal years (e.g. [2023, 2024, 2025]). Empty = last 2 available.
        quarter: 0 for annual, 1-4 for quarterly data
    """
    cik_padded = _format_cik(cik)
    facts = await _fetch_xbrl_facts(cik_padded)
    if not facts:
        return f"No XBRL financial data available for CIK {cik}."

    company_name = facts.get("entityName", f"CIK {cik}")
    bundle_def = XBRL_METRIC_BUNDLES.get(metric_bundle)
    if not bundle_def:
        return f"Unknown bundle '{metric_bundle}'. Choose: key_metrics, profitability, balance_sheet, cash_flow, liquidity."

    bundle_data = {}
    available_years: set = set()
    for metric_name, concept in bundle_def["concepts"].items():
        if concept == "_calc_fcf":
            ocf_data = _extract_concept_values(facts, "NetCashProvidedByUsedInOperatingActivities", years or [], quarter)
            capex_data = _extract_concept_values(facts, "PaymentsToAcquirePropertyPlantAndEquipment", years or [], quarter)
            fcf_vals = {}
            for fy, entry in ocf_data["values"].items():
                capex_entry = capex_data["values"].get(fy)
                if capex_entry and isinstance(entry["val"], (int, float)) and isinstance(capex_entry["val"], (int, float)):
                    fcf_vals[fy] = {"val": entry["val"] + capex_entry["val"], "end": entry["end"], "fp": entry["fp"], "unit": entry["unit"]}
                    available_years.add(int(fy))
                elif isinstance(entry["val"], (int, float)):
                    fcf_vals[fy] = entry
                    available_years.add(int(fy))
            bundle_data[metric_name] = {"label": "Free Cash Flow (calculated)", "values": fcf_vals}
            continue

        result = _extract_concept_values(facts, concept, years or [], quarter)
        bundle_data[metric_name] = result
        for fy in result.get("values", {}).keys():
            available_years.add(int(fy))

    target_years = years or sorted(available_years, reverse=True)[:2]
    table = _render_financials_table(cik_padded, company_name, bundle_data, bundle_def["label"], target_years)

    if metric_bundle == "profitability" and bundle_data.get("Revenue") and bundle_data.get("Net Income"):
        table += "\n\n### Calculated Ratios\n"
        for fy in sorted(target_years):
            rev = bundle_data["Revenue"]["values"].get(fy)
            ni = bundle_data["Net Income"]["values"].get(fy)
            oi = bundle_data.get("Operating Income", {}).get("values", {}).get(fy)
            assets = bundle_data.get("Total Assets", {}).get("values", {}).get(fy)
            eq = bundle_data.get("Stockholders Equity", {}).get("values", {}).get(fy)
            parts = [f"**FY {fy}:**"]
            if rev and ni and rev["val"]:
                parts.append(f"Net Margin: {(ni['val'] / rev['val']) * 100:.1f}%")
            if rev and oi and rev["val"]:
                parts.append(f"Operating Margin: {(oi['val'] / rev['val']) * 100:.1f}%")
            if rev and bundle_data.get("Gross Profit", {}).get("values", {}).get(fy):
                parts.append(f"Gross Margin: {(bundle_data['Gross Profit']['values'][fy]['val'] / rev['val']) * 100:.1f}%")
            if ni and assets and assets["val"]:
                parts.append(f"ROA: {(ni['val'] / assets['val']) * 100:.1f}%")
            if ni and eq and eq["val"]:
                parts.append(f"ROE: {(ni['val'] / eq['val']) * 100:.1f}%")
            table += " | ".join(parts) + "\n"

    if metric_bundle == "liquidity" and bundle_data.get("Current Assets") and bundle_data.get("Current Liabilities"):
        table += "\n\n### Calculated Ratios\n"
        for fy in sorted(target_years):
            ca = bundle_data["Current Assets"]["values"].get(fy)
            cl = bundle_data["Current Liabilities"]["values"].get(fy)
            tl = bundle_data.get("Total Liabilities", {}).get("values", {}).get(fy)
            eq = bundle_data.get("Stockholders Equity", {}).get("values", {}).get(fy)
            parts = [f"**FY {fy}:**"]
            if ca and cl and cl["val"]:
                parts.append(f"Current Ratio: {ca['val'] / cl['val']:.2f}")
                parts.append(f"Working Capital: ${ca['val'] - cl['val']:,.0f}")
            if tl and eq and eq["val"]:
                parts.append(f"Debt/Equity: {tl['val'] / eq['val']:.2f}")
            table += " | ".join(parts) + "\n"

    return table


@tool(parse_docstring=True)
async def compare_companies(
    companies: list[str],
    metric_bundle: Literal["key_metrics", "profitability", "balance_sheet", "cash_flow", "liquidity"] = "key_metrics",
    year: int = 0,
) -> str:
    """Compare financial metrics side-by-side across 2-5 companies.

    Use for relative analysis — e.g. margins or cash flow across competitors in
    the same sector. Resolves each name/ticker to a CIK automatically (no
    lookup_company needed first). `year` = 0 uses the most recent available.

    Args:
        companies: List of company names or tickers, e.g. ['AAPL', 'MSFT', 'GOOGL']
        metric_bundle: Same bundles as get_financials
        year: Fiscal year. 0 = most recent available.
    """
    if len(companies) < 2:
        return "Need at least 2 companies to compare."
    if len(companies) > 5:
        companies = companies[:5]

    ticker_map = await _load_ticker_map()
    cik_list = []
    for c in companies:
        results = _search_ticker_map(ticker_map, c)
        if not results:
            return f"Could not find company '{c}'."
        cik_list.append(results[0])

    bundle_def = XBRL_METRIC_BUNDLES.get(metric_bundle)
    if not bundle_def:
        return f"Unknown bundle '{metric_bundle}'."

    companies_data = []
    all_years: set = set()
    for entry in cik_list:
        facts = await _fetch_xbrl_facts(entry["cik"])
        if not facts:
            continue
        bundle_data = {}
        for metric_name, concept in bundle_def["concepts"].items():
            if concept == "_calc_fcf":
                continue
            result = _extract_concept_values(facts, concept, [], 0)
            bundle_data[metric_name] = result
            for fy in result.get("values", {}).keys():
                all_years.add(int(fy))
        companies_data.append({"name": entry["name"], "ticker": entry["ticker"], "bundle_data": bundle_data})

    if not companies_data:
        return "Could not retrieve financial data for any company."

    target_year = year if year > 0 else (max(all_years) if all_years else 0)
    if target_year == 0:
        return "No financial data available."

    metric_names = [k for k in bundle_def["concepts"] if bundle_def["concepts"][k] != "_calc_fcf"]
    return _render_comparison_table(companies_data, metric_names, target_year)


@tool(parse_docstring=True)
async def get_insider_transactions(
    cik: str,
    transaction_type: Literal["all", "buy", "sell", "grant"] = "all",
    days_back: int = 365,
    min_value: float = 0,
    limit: int = 50,
) -> str:
    """Get insider trading activity from Form 4 filings.

    Use to gauge insider sentiment — heavy insider buying is a bullish signal,
    heavy selling can be a warning sign (though sales are common for many
    legitimate reasons). transaction_type='buy'/'sell' filters the direction;
    days_back sets the lookback window.

    Args:
        cik: CIK of the company
        transaction_type: 'all', 'buy', 'sell', or 'grant'
        days_back: Lookback window in days
        min_value: Minimum transaction value filter
        limit: Maximum results
    """
    cik_padded = _format_cik(cik)
    cik_clean = str(int(cik_padded))

    date_from = datetime.now(timezone.utc) - timedelta(days=days_back)
    date_from_str = date_from.strftime("%Y-%m-%d")

    params = {"action": "getcompany", "CIK": cik_clean, "type": "4",
              "dateb": "", "owner": "include", "count": "100", "output": "atom"}

    try:
        async with httpx.AsyncClient(timeout=SEC_TIMEOUT) as client:
            r = await client.get("https://www.sec.gov/cgi-bin/browse-edgar", params=params, headers=_sec_headers())
            r.raise_for_status()
            xml_text = r.text
    except Exception as e:
        return f"Failed to fetch insider transactions: {e}"

    entries = re.findall(r"<entry>(.*?)</entry>", xml_text, re.DOTALL)
    if not entries:
        return "No Form 4 filings found."

    lines = [f"## Insider Transactions — CIK {cik_padded}", f"**Period:** {date_from_str} to present", ""]
    count = 0
    for entry in entries:
        if count >= limit:
            break
        filing_date = (re.search(r"<filing-date>([^<]+)", entry) or [None, ""])[1]
        acc_num = (re.search(r"<accession-number>([^<]+)", entry) or [None, ""])[1]
        if not filing_date or filing_date < date_from_str:
            continue
        acc_clean = acc_num.replace("-", "")
        link = f"https://www.sec.gov/Archives/edgar/data/{cik_clean}/{acc_clean}/{acc_clean}-index.htm"
        lines.append(f"{filing_date:<14s} Form 4 | Acc: {acc_num}")
        lines.append(f"{'':<14s} Link: {link}")
        lines.append("")
        count += 1

    if count == 0:
        return f"No Form 4 filings found in the last {days_back} days."
    lines.append(f"{count} Form 4 filings found. Use read_filings on individual URLs for details.")
    return "\n".join(lines)
