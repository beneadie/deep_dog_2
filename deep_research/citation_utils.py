"""Citation reliability helpers — shared by the supervisor and sub-agent report stages.

The pipeline is deterministic + one optional LLM repair pass (with retries):

1. The report writer is prompted to emit a <CitationPlanList> registry at the
   top, numeric inline citations ([1], [2], [1][3]) in the body, and a
   ## Sources list that mirrors the plan exactly.
2. `citations_match_sources` validates the result with pure regex (no LLM):
   every inline id must exist in ## Sources, and Sources ids must be contiguous 1..N.
3. `ensure_report_citations` runs validation; on failure it invokes
   `llm_repair_citations` (up to `max_repairs` times) and only accepts a repair
   that is both valid and preserves most of the original body. Otherwise it
   returns the original untouched. This mirrors the retry-style safety of a
   graph conditional edge without needing LangGraph.

All functions here are model-agnostic: `llm_repair_citations` and
`ensure_report_citations` take the chat model as a parameter so any caller
(supervisor writer, sub-agent report writer, or the in-agent full-context
report path) can reuse them.

The sub-agent report path uses a newer, deterministic flow instead:
`finalize_citations(report, registry)` — the writer cites stable inline codes
([S2#3] locally, [A4-S2#3] globally) against a code-keyed registry and the
## Sources section is rebuilt in code, so URLs never come from the model.
`remap_codes` rewrites those codes at each pipeline boundary (local → global,
then codes → contiguous [N] at the final report). `build_final_registry`
merges the supervisor's global registry with curated sources (assigned fresh
C1, C2, ... codes) for the final-report writer.
"""

import asyncio
import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage

from deep_research.utils import extract_text_from_response

logger = logging.getLogger(__name__)

REPAIR_TIMEOUT = 420.0  # seconds — headroom for concurrent tasks


def _strip_citation_plan_list(report_text: str) -> str:
    """Remove optional CitationPlanList scaffolding block from report text."""
    return re.sub(
        r"(?is)<CitationPlanList>.*?</CitationPlanList>\s*",
        "",
        report_text,
    ).strip()


def _split_report_sections(report_text: str) -> tuple[str | None, str | None]:
    """Split report into body and sources block using the ##/### Sources header."""
    normalized_report = _strip_citation_plan_list(report_text)
    sources_header_match = re.search(r"(?im)^###{0,1}\s+Sources\s*$", normalized_report)
    if not sources_header_match:
        return None, None

    body = normalized_report[:sources_header_match.start()].rstrip()
    raw_sources_block = normalized_report[sources_header_match.end():]
    return body, raw_sources_block


def citations_match_sources(report_text: str) -> bool:
    """Validate inline numeric citations against the sources section.

    True only when: every inline [N] citation in the body exists in the
    ## Sources list, AND the Sources ids are contiguous 1..N.
    """
    body, raw_sources_block = _split_report_sections(report_text)
    if body is None or raw_sources_block is None:
        return False

    body_ids = [int(m.group(1)) for m in re.finditer(r"\[(\d{1,3})\]", body)]
    source_ids = []
    for line in raw_sources_block.splitlines():
        match = re.match(r"^\[(\d+)\]\s*(.*)$", line.strip())
        if not match:
            continue
        if re.search(r"https?://\S+", match.group(2)):
            source_ids.append(int(match.group(1)))

    if not body_ids or not source_ids:
        return False

    body_set = set(body_ids)
    source_set = set(source_ids)
    return body_set.issubset(source_set) and source_set == set(range(1, len(source_set) + 1))


# ── Registry-based citations (sub-agent report path) ─────────────────────
# The report writer only emits inline [N] citations against a numbered
# registry of sources provided in the prompt. `finalize_citations` strips any
# plan block and any writer-produced Sources section, then rebuilds
# ## Sources in code from the registry — so URLs never have to be reproduced
# by the model and the Sources section is always internally consistent.


def extract_cited_ids(report_text: str) -> list[int]:
    """Inline citation numbers [N] in body order, deduplicated."""
    ids = [int(m.group(1)) for m in re.finditer(r"\[(\d{1,3})\]", report_text)]
    seen: set[int] = set()
    out: list[int] = []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def renumber_citations(report_text: str, id_remap: dict) -> str:
    """Rewrite inline [N] citations to new ids per `id_remap`.

    Single-pass regex callback over `[N]`; replaces [N] → [id_remap[N]] when the
    key exists, otherwise leaves the citation untouched. Multi-citations like
    [1][3] are handled naturally. Pure and deterministic.
    """
    if not id_remap:
        return report_text

    def _repl(m):
        old = int(m.group(1))
        new = id_remap.get(old)
        return f"[{new}]" if new is not None else m.group(0)

    return re.sub(r"\[(\d{1,3})\]", _repl, report_text)


# ── Code-based citation tokens (source registry codes) ─────────────────
# A source's citation code is either a local code (S2#3 / R4) used inside a
# sub-agent, or a global code (A4-S2#3 / A4-R4) once the supervisor has stamped
# the agent id. One regex matches both, so a single extract/remap helper works
# at every pipeline stage: sub-agent report, supervisor remap, final renumber.
_CODE_TOKEN = r"(?:A\d+-)?(?:S\d+#\d+|R\d+)|C\d+"
_CODE_CITE_RE = re.compile(rf"\[({_CODE_TOKEN})\]")


def extract_cited_codes(report_text: str) -> list[str]:
    """Inline citation codes ([S2#3] / [A4-S2#3]) in body order, deduplicated."""
    out: list[str] = []
    seen: set[str] = set()
    for m in _CODE_CITE_RE.finditer(report_text):
        code = m.group(1)
        if code not in seen:
            seen.add(code)
            out.append(code)
    return out


def remap_codes(report_text: str, code_map: dict) -> str:
    """Rewrite inline [code] citations per `code_map` (code -> new label).

    Single-pass regex callback over code tokens; replaces [code] -> [code_map[code]]
    when the key exists, otherwise leaves the citation untouched (caller logs the
    miss). Values may be strings (local -> global codes) or ints (codes -> [N]).
    Pure and deterministic.
    """
    if not code_map:
        return report_text

    def _repl(m):
        code = m.group(1)
        new = code_map.get(code)
        return f"[{new}]" if new is not None else m.group(0)

    return _CODE_CITE_RE.sub(_repl, report_text)


def _extract_plan_ids(report_text: str) -> list[int]:
    """Extract [N] ids from a <CitationPlanList> block, if present."""
    plan_match = re.search(r"(?is)<CitationPlanList>.*?</CitationPlanList>", report_text)
    if not plan_match:
        return []
    plan_block = plan_match.group(0)
    return [int(m.group(1)) for m in re.finditer(r"\[(\d{1,3})\]", plan_block)]


def finalize_citations(report_text: str, registry: list[dict], renumber: bool = False) -> str:
    """Deterministically rebuild a report's ## Sources section.

    Two modes, auto-selected from the registry entries:

    - Numeric (legacy): entries carry no `code`; the writer cited [N] positions
      (index + 1 into `registry`).
    - Code-based: entries carry `code` (S2#3 / A4-S2#3); the writer cited those
      codes inline. When `renumber` is True, codes are rewritten to contiguous
      [1..N] in first-appearance order (used at the final-report boundary); when
      False the body keeps its codes and Sources is keyed by code (used for
      sub-agent reports, which the supervisor remaps to global codes later).

    Any <CitationPlanList> block and any writer-produced Sources section are
    stripped. Out-of-registry citations are dropped from Sources (body left
    intact); if no in-range citation exists the cleaned body is returned
    unchanged (never worse than the writer produced).
    """
    body, _ = _split_report_sections(report_text)
    if body is None:
        body = _strip_citation_plan_list(report_text)

    # Validate CitationPlanList matches inline citations (if present — numeric)
    plan_ids = _extract_plan_ids(report_text)
    if plan_ids:
        cited = extract_cited_ids(body)
        plan_set = set(plan_ids)
        cited_set = set(cited)
        in_plan_not_cited = plan_set - cited_set
        cited_not_in_plan = cited_set - plan_set
        if in_plan_not_cited:
            logger.warning(
                "finalize_citations: sources in plan but not cited inline: %s",
                sorted(in_plan_not_cited),
            )
        if cited_not_in_plan:
            logger.warning(
                "finalize_citations: sources cited inline but not in plan: %s",
                sorted(cited_not_in_plan),
            )

    if registry and registry[0].get("code"):
        return _finalize_code_citations(body, registry, renumber)
    return _finalize_numeric_citations(body, registry)


def _finalize_numeric_citations(body: str, registry: list[dict]) -> str:
    """Rebuild ## Sources for the legacy positional [N] citation flow."""
    cited = extract_cited_ids(body)
    n = len(registry)
    valid = sorted(i for i in cited if 1 <= i <= n)
    out_of_range = [i for i in cited if i not in valid]
    if out_of_range:
        logger.warning("finalize_citations: out-of-range citations %s — dropped from Sources",
                       out_of_range)

    if not valid:
        logger.warning("finalize_citations: no in-range inline citations — leaving body unchanged")
        return body

    lines = [body, "", "## Sources", ""]
    for i in valid:
        entry = registry[i - 1]
        title = entry.get("title") or ""
        url = entry.get("url") or entry.get("identifier") or ""
        lines.append(f"[{i}] {title} ({url})" if title else f"[{i}] {url}")
    return "\n".join(lines)


def _finalize_code_citations(body: str, registry: list[dict], renumber: bool) -> str:
    """Rebuild ## Sources for the code-based citation flow.

    Valid codes are those cited inline AND present in the registry (first-
    appearance order). With `renumber`, codes are remapped to contiguous [1..N]
    in that order; otherwise the body keeps its codes and Sources is keyed by
    code.
    """
    by_code = {str(e.get("code")): e for e in registry if e.get("code")}
    cited = extract_cited_codes(body)
    valid = [c for c in cited if c in by_code]
    out_of_registry = [c for c in cited if c not in valid]
    if out_of_registry:
        logger.warning("finalize_citations: codes not in registry %s — dropped from Sources",
                       out_of_registry)

    if not valid:
        logger.warning("finalize_citations: no in-registry inline codes — leaving body unchanged")
        return body

    lines = [body, "", "## Sources", ""]
    if renumber:
        code_to_num = {c: i for i, c in enumerate(valid, 1)}
        lines[0] = remap_codes(body, code_to_num)
        for c in valid:
            entry = by_code[c]
            title = entry.get("title") or ""
            url = entry.get("url") or entry.get("identifier") or ""
            lines.append(f"[{code_to_num[c]}] {title} ({url})" if title else f"[{code_to_num[c]}] {url}")
    else:
        for c in valid:
            entry = by_code[c]
            title = entry.get("title") or ""
            url = entry.get("url") or entry.get("identifier") or ""
            lines.append(f"[{c}] {title} ({url})" if title else f"[{c}] {url}")
    return "\n".join(lines)


def build_final_registry(curated_sources: list[dict], source_registry: list[dict]) -> list[dict]:
    """Merge the supervisor's global registry with curated sources for the final report.

    Order matters: `source_registry` entries come first and keep their existing
    global codes (A{agent_id}-S2#3 / A{agent_id}-R#). `curated_sources` entries
    (sources-mode full text, which the supervisor never saw) are appended and
    assigned fresh C1, C2, ... codes. Entries are deduplicated by URL (fallback
    identifier), so a source present in both lists appears once.

    Returns normalized entries: {code, url, title, source_type, ref, full_text}.
    """
    merged: list[dict] = []
    seen: set[str] = set()

    def _key(entry: dict) -> str:
        return (entry.get("url") or entry.get("identifier") or "").strip().lower()

    for entry in source_registry:
        key = _key(entry)
        if not key:
            continue
        code = (entry.get("code") or "").strip()
        if not code:
            logger.warning(
                "build_final_registry: source_registry entry without code skipped: %s", key
            )
            continue
        if key in seen:
            continue
        seen.add(key)
        merged.append({
            "code": code,
            "url": entry.get("url") or entry.get("identifier") or "",
            "title": entry.get("title") or "Untitled",
            "source_type": entry.get("source_type") or "web",
            "ref": entry.get("ref") or "",
            "full_text": entry.get("full_text") or "",
        })

    curated_idx = 1
    for entry in curated_sources:
        key = _key(entry)
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        merged.append({
            "code": f"C{curated_idx}",
            "url": entry.get("url") or entry.get("identifier") or "",
            "title": entry.get("title") or "Untitled",
            "source_type": entry.get("source_type") or "web",
            "ref": entry.get("reason") or "",
            "full_text": entry.get("full_text") or "",
        })
        curated_idx += 1

    return merged


def _body_length_sane(original_body: str | None, repaired_text: str) -> bool:
    """Reject a repair that gutted the report body (< 60% of the original)."""
    repaired_body, _ = _split_report_sections(repaired_text)
    return (
        bool(repaired_body)
        and bool(original_body)
        and len(repaired_body) >= int(0.6 * len(original_body))
    )


def _extract_sources_from_findings(findings_text: str, max_chars: int) -> str:
    """Extract just the Sources sections from findings to reduce context size."""
    # Pattern matches ### Sources Used or ## Sources sections
    pattern = r"(?im)^#{2,3}\s+Sources.*?(?=\n#{2,3}\s|\Z)"
    sources_sections = re.findall(pattern, findings_text, re.DOTALL)

    if sources_sections:
        combined = "\n\n".join(sources_sections)
        if len(combined) <= max_chars:
            return combined
        return combined[:max_chars] + "\n... [truncated]"

    return findings_text[:max_chars] + "\n... [truncated]"


async def llm_repair_citations(
    report_text: str,
    findings_text: str,
    model,
    timeout: float = REPAIR_TIMEOUT,
) -> str:
    """Single LLM repair pass for citation/source consistency issues.

    Takes the chat model explicitly so any caller can reuse it. Returns "" on
    timeout/error (the caller decides whether to accept or fall back).
    """
    # Truncate findings to a reasonable size — only need source URLs, not full content
    max_findings_chars = 35000  # ~8-10K tokens, leaves room for report
    if len(findings_text) > max_findings_chars:
        logger.info(
            "Citation repair: truncating findings from %d to %d chars",
            len(findings_text), max_findings_chars,
        )
        findings_text = _extract_sources_from_findings(findings_text, max_findings_chars)

    system_prompt = (
        "You are a citation repair engine. "
        "Fix only inline numeric citations, the optional <CitationPlanList>, and the ##/### Sources list."
    )
    human_prompt = f"""
Repair citation numbering consistency in this markdown report.

Rules:
1) Keep original prose, structure, and language unchanged as much as possible.
2) Renumber inline citations to contiguous [1], [2], ...
3) Ensure every inline citation appears in ## Sources (or ### Sources).
4) Keep ## Sources or ### Sources at the end of the report.
5) If <CitationPlanList> exists, ensure it mirrors ##/### Sources exactly (same IDs and entries).
6) Unused sources are allowed in Sources/CitationPlanList, but numbering must remain contiguous in Sources.
7) Use Findings Context only to recover or verify missing/incorrect source entries; do not rewrite argumentation.
8) Output ONLY the cleaned markdown report.

<Findings Context>
{findings_text}
</Findings Context>

<Report>
{report_text}
</Report>
"""

    try:
        response = await asyncio.wait_for(
            model.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt),
            ]),
            timeout=timeout,
        )
        return extract_text_from_response(response.content)
    except asyncio.TimeoutError:
        logger.warning("Citation repair: timeout after %.0fs", timeout)
        return ""
    except Exception as e:  # noqa: BLE001
        logger.warning("Citation repair: error %s: %s", type(e).__name__, e)
        return ""


async def ensure_report_citations(
    report_text: str,
    findings_text: str,
    model,
    max_repairs: int = 2,
    timeout: float = REPAIR_TIMEOUT,
) -> str:
    """Validate a report's citations; repair (with retries) if invalid.

    Returns the cleaned report (CitationPlanList scaffolding removed). On
    failure it retries up to `max_repairs` times, accepting a repair only when
    it validates AND preserves >= 60% of the original body. Never returns
    something worse than the original: falls back to the untouched report.
    """
    if not report_text.strip():
        return report_text

    if citations_match_sources(report_text):
        logger.info("Citation validation passed")
        return _strip_citation_plan_list(report_text)

    logger.info("Citation validation failed — invoking repair (up to %d attempts)", max_repairs)
    original_body, _ = _split_report_sections(report_text)

    current = report_text
    for attempt in range(1, max_repairs + 1):
        repaired = await llm_repair_citations(current, findings_text, model, timeout=timeout)
        if not repaired.strip():
            logger.info("Citation repair attempt %d returned empty", attempt)
            break
        if citations_match_sources(repaired) and _body_length_sane(original_body, repaired):
            logger.info("Citation repair accepted (attempt %d)", attempt)
            return _strip_citation_plan_list(repaired)
        logger.info("Citation repair attempt %d rejected — retrying", attempt)
        current = repaired

    logger.info("Citation repair rejected — keeping original report")
    return _strip_citation_plan_list(report_text)
