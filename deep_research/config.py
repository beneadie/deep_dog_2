"""
Model Configuration for Deep Research Agent.

This module centralizes model selection for the whole codebase.

There is ONE way to pick a model: a model NAME resolved by get_model().
Each role (supervisor, initial draft, sub-agents) is configured with a PRIMARY
model plus an optional FALLBACK CHAIN (a comma-separated list of model names).
Routing (OpenRouter vs native API) is controlled by the ROUTE_VIA_OPENROUTER
flags in the Routing section below.

Examples of model names accepted by get_model():
    - OpenAI: "gpt-5", "gpt-5-mini"
    - Google Gemini: "gemini-2.5-pro", "gemini-3-flash-preview", "gemini-3-pro-preview"
    - DeepSeek: "deepseek-v4-pro", "deepseek-v4-flash"
    - OpenRouter Nemotron: "nvidia/nemotron-3.5-lightning"
"""

# ═══════════════════════════════════════════════════════════════════════════
# AVAILABLE MODELS — Quick reference for swapping during testing.
#
# Model names below work in SUBAGENT_MODEL, SUPERVISOR_MODEL,
# DRAFT_REPORT_MODEL, their *_FALLBACK_CHAIN lists, and get_model().
# ═══════════════════════════════════════════════════════════════════════════
#
# DEEPSEEK (native API: api.deepseek.com) — env: DEEPSEEK_API_KEY
#   deepseek-v4-flash          fast, cost-effective, tool calling
#   deepseek-v4-pro            strongest reasoning, slower
#
# MIMO (native API: api.xiaomimimo.com) — env: MIMO_API_KEY
#   mimo-v2.5-pro              flagship, strong agentic/coding (1M context)
#   mimo-v2.5                  lighter, ~half cost (native, both hosted)
#
# META MUSE (native API: api.meta.ai/v1) — env: META_API_KEY
#   muse-spark-1.2             reasoning model for agentic tasks (1M ctx)
#   muse-spark-1.2-contributor same weights, cheaper, Meta trains on data
#   muse-spark-1.1             previous generation
#   muse-glimmer-30b           open-weight, distilled (OpenRouter only)
#
# GOOGLE GEMINI (via google_genai) — env: GEMINI_API_KEY / GOOGLE_API_KEY
#   gemini-3-flash-preview     fast, cost-effective
#   gemini-3-pro-preview       stronger reasoning
#   gemini-3.7-flash           newest flash
#   gemini-2.5-pro             previous gen pro
#
# OPENAI — env: OPENAI_API_KEY
#   gpt-5.2                    current default for OpenAI provider
#   gpt-5.6-luna               strong reasoning
#   gpt-5 / gpt-5-mini         other available models
#
# NVIDIA (via OpenRouter) — env: OPENROUTER_API_KEY
#   nvidia/nemotron-3.5-lightning         cheap search/filter model
#   nvidia/nemotron-3-super-120b-a12b     120B
#   nvidia/nemotron-3-nano-30b-a3b        30B lightweight
#
# Z.AI GLM — env: ZHIPUAI_API_KEY
#   glm-5.3                    latest, reasoning always on
#   glm-5.2                    previous generation
#
# OPENROUTER (single key covers all) — env: OPENROUTER_API_KEY
#   Set ROUTE_VIA_OPENROUTER=true to route deepseek/mimo/meta through OR.
#   Any slug from openrouter.ai/models works directly:
#     "deepseek/deepseek-v4-pro", "xiaomi/mimo-v2.5", "meta/muse-spark-1.2"
#   DeepSeek V4 Flash via Alibaba (Singapore): "deepseek-baba-singapore"
#
# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION VARIABLES — All knobs with their valid options.
# ═══════════════════════════════════════════════════════════════════════════
#
# ── Model selection ─────────────────────────────────────────────────────
# SUBAGENT_MODEL             Primary sub-agent model (any model name above)
#                            Default: "deepseek-v4-flash"
#                            Env: SUBAGENT_MODEL
# SUBAGENT_MODEL_FALLBACK_CHAIN   Comma-separated fallback list.
#                            Default: [SUBAGENT_MODEL, "deepseek-v4-flash"]
#                            Env: SUBAGENT_MODEL_FALLBACK_CHAIN
# SUBAGENT_MODEL_CHAIN_BY_AGENT   Per-agent override map, keyed by
#                            AGENT_REGISTRY key (e.g. "ResearchReddit").
#                            Empty (default) = shared SUBAGENT_MODEL_FALLBACK_CHAIN.
# SUPERVISOR_MODEL           Supervisor model (any model name above)
#                            Default: "deepseek-v4-flash"
#                            Env: SUPERVISOR_MODEL
# SUPERVISOR_MODEL_FALLBACK_CHAIN   Comma-separated fallback list.
#                            Default: [SUPERVISOR_MODEL, "deepseek-v4-pro"]
#                            Env: SUPERVISOR_MODEL_FALLBACK_CHAIN
# DRAFT_REPORT_MODEL         Research brief and initial draft model (any model
#                            name above).
#                            Default: "deepseek-v4-flash"
#                            Env: DRAFT_REPORT_MODEL
# DRAFT_REPORT_MODEL_FALLBACK_CHAIN   Comma-separated fallback list.
#                            Default: [DRAFT_REPORT_MODEL, "deepseek-v4-flash",
#                            "deepseek-v4-pro"]
#                            Env: DRAFT_REPORT_MODEL_FALLBACK_CHAIN
#
# ── Routing ─────────────────────────────────────────────────────────────
# ROUTE_VIA_OPENROUTER       Shared default: route deepseek/mimo/meta through
#                            OpenRouter for ALL roles unless a role flag below
#                            overrides it.
#                            Default: false
#                            Env: ROUTE_VIA_OPENROUTER=true|1|yes|on
# SUBAGENT_ROUTE_VIA_OPENROUTER
#                            Platform research agents.
#                            Unset = inherit ROUTE_VIA_OPENROUTER.
# SUPERVISOR_ROUTE_VIA_OPENROUTER
#                            Supervisor + final report write
#                            (cache-tied; they always share this flag).
#                            Unset = inherit ROUTE_VIA_OPENROUTER.
# DRAFT_ROUTE_VIA_OPENROUTER Research brief + initial draft report nodes
#                            (cold passes, independent).
#                            Unset = inherit ROUTE_VIA_OPENROUTER.
# DRAFT_REPORT_REASONING_EFFORT
#                            Reasoning effort for the research brief and initial
#                            draft nodes:
#                            "" / "low" / "medium" / "high". Only honored when the
#                            draft is routed via OpenRouter (native DeepSeek is
#                            enabled/disabled only).
# DISABLE_MODEL_FALLBACK     Disable model fallback chains; failures raise
#                            instead of switching models.
#                            Default: false
#                            Env: DISABLE_MODEL_FALLBACK=true|1|yes|on
#
# ── Prompt version ──────────────────────────────────────────────────────
# PROMPT_VERSION             Supervisor prompt set.
#                            "OPEN"             — current open-ended prompt
#                            Default: "OPEN"
#
# ── Research timing ─────────────────────────────────────────────────────
# RESEARCH_TIME_MIN_MINUTES  Minimum research time before ResearchComplete.
#                            Default: 5
# RESEARCH_TIME_MAX_MINUTES  Target max research time.
#                            Default: 10
# RESEARCH_STRICT_TIMEOUT_MINUTES  Hard stop (auto-computed: max + 1).
#                            Default: 11.0
# SUBAGENT_TIMEOUT_SECONDS   Per-subagent timeout.
#                            Default: 600 (10 min)
# SUPERVISOR_MAX_ITERATIONS  Supervisor loop iteration cap.
#                            Default: 30
# SUBAGENT_MAX_ITERATIONS    Tool-call rounds per delegated sub-agent.
#                            Default: 10
#
# ── Depth caps ──────────────────────────────────────────────────────────
# DEFAULT_MAX_TOTAL_READS    Total read budget per sub-agent run.
#                            Default: 25
# SUBAGENT_MAX_READS         Max items read per iteration.
#                            Default: 8
# SUBAGENT_MAX_SAVES         Max items saved per run.
#                            Default: 15
# SUBAGENT_MAX_SEARCHES      Max search tool calls per run.
#                            Default: 8
# SUBAGENT_MAX_CONCURRENCY   Max parallel tool calls per iteration.
#                            Default: 4
# DEFAULT_MAX_TOKENS         Max tokens for writer models (None = unlimited).
#                            Default: None
# The sub-agent depth caps (iterations/reads/saves/searches/concurrency) are
# shared by the supervisor registry and run_platform.py, so the same agent
# behaves identically regardless of entry point.
#
# ── Sub-agent output ────────────────────────────────────────────────────
# SUBAGENT_OUTPUT_MODE       Deliverable for research sub-agents.
#                            "sources"        — save-as-you-go, curated list
#                            "report"         — save-as-you-go, mini-report
#                            "sources_inline" — read-all, curated list
#                            "report_inline"  — read-all, agent writes report
#                            Default: "sources"
# DISCOVERY_OUTPUT_MODE      Deliverable for discovery sub-agents.
#                            Same options as above.
#                            Default: "report_inline"
#
# ── Agent filtering ─────────────────────────────────────────────────────
# ENABLED_AGENTS             Which sub-agents the supervisor may call.
#                            Comma-separated env var, e.g.
#                            ENABLED_AGENTS=ResearchWeb,ResearchReddit
#                            Unset/empty = WEB agent only (deployment default).
#                            Per-run override: RunConfig(enabled_agents=[...]).
#                            Order defines tool-schema binding order.
# GENERAL_AGENT_PLATFORMS    Platforms the general agent may use.
#                            Comma-separated env var, e.g.
#                            GENERAL_AGENT_PLATFORMS=web,reddit,pubmed
#                            Unset/empty = all platforms.
#                            Per-run override: RunConfig(general_agent_platforms=[...])
#
# ── Thinking / reasoning ───────────────────────────────────────────────
# THINKING_MODE              Reasoning enablement for all providers.
#                            "auto" — provider default + explicit param where needed
#                            "on"   — always pass explicit enable param
#                            "off"  — never force thinking (providers still reason
#                                     by default if they do natively)
#                            Default: "auto"
#
# ── Web search ──────────────────────────────────────────────────────────
# WEB_SEARCH_ENGINE          Which search tools to expose.
#                            "tavily" — Tavily only
#                            "exa"    — Exa only
#                            "both"   — both Tavily and Exa
#                            Default: "exa"
#                            Env: WEB_SEARCH_ENGINE
#
# ── Output / logging ───────────────────────────────────────────────────
# OUTPUT_MODE                Where reports go.
#                            "file" — Markdown files
#                            "db"   — structured database
#                            "both" — both
#                            Default: "file"
# LOG_MODE                   Where logs go.
#                            "file" | "db" | "both"
#                            Default: "file"
# SAVE_REPORT_TO_FILE        Write final/subtopic reports to disk.
#                            Default: False (return report text to the caller)
# SAVE_SUBAGENT_REPORTS_TO_FILE  Also write each sub-agent deliverable as a
#                            full-length .md (sub_agents/sub_agent_XXX.md).
#                            Default: False
#                            Env: SAVE_SUBAGENT_REPORTS_TO_FILE
# ENABLE_SOURCE_LOG          Log every saved source to sources.jsonl /
#                            research_data_*.json (web search + all platform
#                            sub-agents). Off disables source logging entirely;
#                            the report's ## Sources section is unaffected.
#                            Default: True
#                            Env: ENABLE_SOURCE_LOG
# ENABLE_SUBTOPIC_GENERATION Run subtopic evaluation after final report.
#                            Default: False
# ENABLE_RESEARCH_TRACE      Supervisor-subagent trace logging.
#                            Default: False
#
# ── Per-call timeouts ──────────────────────────────────────────────────
# LLM_TIMEOUT                LLM call timeout (seconds).
#                            Default: 180
# TOOL_TIMEOUT               Tool call timeout (seconds).
#                            Default: 120
#
# ── Environment variables (.env) ────────────────────────────────────────
# Required:
#   DEEPSEEK_API_KEY (or DEEPSEEK_KEY)   DeepSeek native API
#   MIMO_API_KEY                         MiMo native API
#   META_API_KEY                         Meta Muse native API
#   GEMINI_API_KEY (or GOOGLE_API_KEY)   Google Gemini
#   TAVILY_API_KEY                       Tavily web search
#   OPENROUTER_API_KEY                   OpenRouter (for nvidia, baba, fallback)
#
# Optional:
#   OPENAI_API_KEY                       OpenAI
#   ZHIPUAI_API_KEY                      Z.AI GLM
#   EXA_API_KEY                          Exa search
#   PERPLEXITY_KEY                       Substack/Perplexity search
#   PUBMED_EMAIL                         PubMed E-utilities contact
#   SEC_EDGAR_CONTACT_EMAIL              SEC EDGAR contact
#
# ═══════════════════════════════════════════════════════════════════════════

import contextvars
import json
import os
import re
from typing import Optional
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage

# ===== CONFIGURATION =====

# Supervisor model can be configured independently from subagents.
# Defaults to DeepSeek V4 Flash unless overridden
# via environment variable.
SUPERVISOR_MODEL = os.environ.get("SUPERVISOR_MODEL", "deepseek-v4-flash")

# OPEN             = lean, most open-ended; minimal draft, supervisor fills in the information
# Env: PROMPT_VERSION (only OPEN is supported)
PROMPT_VERSION = os.environ.get("PROMPT_VERSION", "OPEN")

# Research time window (in minutes) - controls the expected task duration
RESEARCH_TIME_MIN_MINUTES = 5
RESEARCH_TIME_MAX_MINUTES = 15
RESEARCH_STRICT_TIMEOUT_MINUTES = RESEARCH_TIME_MAX_MINUTES + 1.0  # Hard stop limit

# Salvage threshold: if the supervisor hits consecutive failures, route to the
# final report node only when at least this fraction of the research window has
# elapsed AND some findings of task value were gathered; otherwise abort.
FINDINGS_SALVAGE_TIME_FRACTION = 0.6

# Maximum number of supervisor research iterations (tool calls)
SUPERVISOR_MAX_ITERATIONS = 30

# Maximum tool-call rounds per sub-agent (research platform agent). This is
# the iteration ceiling for a single delegated research run.
SUBAGENT_MAX_ITERATIONS = int(os.environ.get("SUBAGENT_MAX_ITERATIONS", "5"))

# Shared depth-cap defaults for sub-agents, used by BOTH the supervisor-launched
# registry (_ConfiguredAgent) and the standalone runner (run_platform.py) so the
# same agent behaves identically regardless of entry point.
SUBAGENT_MAX_READS = int(os.environ.get("SUBAGENT_MAX_READS", "10"))
SUBAGENT_MAX_SAVES = int(os.environ.get("SUBAGENT_MAX_SAVES", "10"))
SUBAGENT_MAX_SEARCHES = int(os.environ.get("SUBAGENT_MAX_SEARCHES", "3"))
SUBAGENT_MAX_CONCURRENCY = int(os.environ.get("SUBAGENT_MAX_CONCURRENCY", "3"))

# Soft (prompt-level) caps on how many research / discovery sub-agents the
# supervisor may launch per iteration. These are injected into the supervisor
# system prompt via {max_concurrent_research_units} /
# {max_concurrent_discovery_units}; they guide the model but are not enforced
# by the spawn loop. Env: SUPERVISOR_MAX_CONCURRENT_RESEARCH,
# SUPERVISOR_MAX_CONCURRENT_DISCOVERY
SUPERVISOR_MAX_CONCURRENT_RESEARCH = int(os.environ.get("SUPERVISOR_MAX_CONCURRENT_RESEARCH", "3"))
SUPERVISOR_MAX_CONCURRENT_DISCOVERY = int(os.environ.get("SUPERVISOR_MAX_CONCURRENT_DISCOVERY", "2"))


def subagent_recursion_limit(max_iterations: int) -> int:
    """Recursion limit for a sub-agent graph given its iteration cap.

    Each iteration is up to 2 graph nodes (LLM call + tool round), so the limit
    must exceed max_iterations * 2. Single source of truth so the supervisor
    registry and the standalone runner agree.
    """
    return max_iterations * 2 + 7

# Per-subagent time limit (seconds) - graceful stop; agent finishes current iteration then compresses
SUBAGENT_TIMEOUT_SECONDS = 600  # 10 minutes

# Per-supervisor-decision time limit (seconds). Bounds the supervisor's model
# call so a stalled provider can't hang research.
SUPERVISOR_TIMEOUT_SECONDS = int(os.environ.get("SUPERVISOR_TIMEOUT_SECONDS", "420"))


# Canonical fallback language used when the upstream language classifier fails.
TARGET_LANGUAGE_FALLBACK = "English"

# Default total read budget per sub-agent run — a hard cap on how many items the
# agent may READ across the whole run, to prevent over-reading. The supervisor
# can override this per sub-agent when a deeper or narrower search is warranted.
DEFAULT_MAX_TOTAL_READS = 25

# Optional: Set max tokens for writer models (None = no limit)
DEFAULT_MAX_TOKENS = None  # Previously was 32000-40000, now unlimited

# Controls whether reports (final and subtopic) are saved to disk as files.
# Env: SAVE_REPORT_TO_FILE (0/false/no/off disables file writes)
SAVE_REPORT_TO_FILE: bool = os.environ.get(
    "SAVE_REPORT_TO_FILE", "false"
).strip().lower() not in {"0", "false", "no", "off"}

# When ON, each sub-agent's deliverable (curated list, mini-report, or inline
# report) is ALSO written as a full-length .md file next to its JSON log entry,
# under <run folder>/sub_agents/sub_agent_XXX.md — handy for reviewing what each
# delegated sub-agent produced. The JSON log keeps its truncated copy regardless.
# Env: SAVE_SUBAGENT_REPORTS_TO_FILE
SAVE_SUBAGENT_REPORTS_TO_FILE: bool = os.environ.get(
    "SAVE_SUBAGENT_REPORTS_TO_FILE", ""
).strip().lower() in {"1", "true", "yes", "on"}

# When ON, every saved source (from web search AND all platform sub-agents) is
# logged to sources.jsonl / research_data_*.json. Turning this off disables the
# source log entirely; the report's ## Sources section is built from the
# registry and is unaffected.
# Env: ENABLE_SOURCE_LOG
ENABLE_SOURCE_LOG: bool = os.environ.get(
    "ENABLE_SOURCE_LOG", ""
).strip().lower() not in {"0", "false", "no", "off"}

# Controls whether the subtopic evaluation and generation workflow runs after the final report
ENABLE_SUBTOPIC_GENERATION: bool = False  # Set to False to skip subtopic reports entirely

# Default deliverable for sub-agents launched by the supervisor. One of:
#   "sources"        — save-as-you-go → curated sources list
#   "report"         — save-as-you-go → isolated report_write LLM call (mini-report)
#   "sources_inline" — read-all-then-select → curated sources list
#   "report_inline"  — read-all → agent writes the report as its final message
# Env: SUBAGENT_OUTPUT_MODE
SUBAGENT_OUTPUT_MODE: str = os.environ.get("SUBAGENT_OUTPUT_MODE", "sources")

# Default deliverable for DISCOVERY-mode sub-agents. Discovery agents default to
# the full-context "read everything, write the discovery report inline" flow
# (report_inline) rather than save-as-you-go curation. Overridable per deployment
# — e.g. "sources" restores the old curation-style discovery behavior, "report"
# uses the isolated report-writer call. Research agents always use
# SUBAGENT_OUTPUT_MODE.
# Env: DISCOVERY_OUTPUT_MODE
DISCOVERY_OUTPUT_MODE: str = os.environ.get("DISCOVERY_OUTPUT_MODE", "report_inline")

# Controls whether supervisor-subagent research trace logging/compression is enabled
# Set DISABLE_RESEARCH_TRACE=1/true/yes/on to disable trace generation
ENABLE_RESEARCH_TRACE: bool = False #os.environ.get("DISABLE_RESEARCH_TRACE", "").strip().lower() not in {
#    "1",
#    "true",
#    "yes",
#    "on",
#}

# Supervisor-specific fallback chain (model NAMES, not providers — routing is
# resolved by get_model()). If unset, defaults to [SUPERVISOR_MODEL, deepseek-v4-pro].
# Example env value: "deepseek-v4-flash,deepseek-v4-pro"
_supervisor_chain_env = os.environ.get("SUPERVISOR_MODEL_FALLBACK_CHAIN", "").strip()
if _supervisor_chain_env:
    SUPERVISOR_MODEL_FALLBACK_CHAIN = [
        m.strip() for m in _supervisor_chain_env.split(",") if m.strip()
    ]
else:
    SUPERVISOR_MODEL_FALLBACK_CHAIN = [SUPERVISOR_MODEL, "deepseek-v4-pro"]

# Sub-agent model chain (research platform agents), as
# model NAMES resolved by get_model(). SUBAGENT_MODEL is the primary; the chain
# is the ordered fallback list. Override the whole chain with
# SUBAGENT_MODEL_FALLBACK_CHAIN (comma-separated env value).
# Example env value: "nvidia/nemotron-3.5-lightning,deepseek-v4-flash"
SUBAGENT_MODEL = os.environ.get("SUBAGENT_MODEL", "deepseek-v4-flash")
_subagent_chain_env = os.environ.get("SUBAGENT_MODEL_FALLBACK_CHAIN", "").strip()
if _subagent_chain_env:
    SUBAGENT_MODEL_FALLBACK_CHAIN = [
        m.strip() for m in _subagent_chain_env.split(",") if m.strip()
    ]
else:
    SUBAGENT_MODEL_FALLBACK_CHAIN = [SUBAGENT_MODEL, "deepseek-v4-flash"]

# Draft-report model for the research brief and the initial, cold,
# non-cacheable draft pass in research_agent_scope.py. These passes do not use
# the sub-agent chain, so a slow/expensive supervisor does NOT slow them. They
# are scaffolding that must be LONG, not perfect, so a fast model is fine.
# Override with DRAFT_REPORT_MODEL to point both passes at any model.
# Example env value: "nvidia/nemotron-3.5-lightning"
DRAFT_REPORT_MODEL = os.environ.get("DRAFT_REPORT_MODEL", "deepseek-v4-flash").strip()
_draft_chain_env = os.environ.get("DRAFT_REPORT_MODEL_FALLBACK_CHAIN", "").strip()
if _draft_chain_env:
    DRAFT_REPORT_MODEL_FALLBACK_CHAIN = [
        m.strip() for m in _draft_chain_env.split(",") if m.strip()
    ]
elif DRAFT_REPORT_MODEL:
    DRAFT_REPORT_MODEL_FALLBACK_CHAIN = [DRAFT_REPORT_MODEL, "deepseek-v4-flash", "deepseek-v4-pro"]
else:
    DRAFT_REPORT_MODEL_FALLBACK_CHAIN = SUBAGENT_MODEL_FALLBACK_CHAIN

# Per-agent model chain override, keyed by AGENT_REGISTRY key (e.g.
# "ResearchWeb", "ResearchReddit"). Leave empty to use the shared
# SUBAGENT_MODEL_FALLBACK_CHAIN for every sub-agent. Set to give a specific
# sub-agent its own model/fallback chain:
#   SUBAGENT_MODEL_CHAIN_BY_AGENT = {"ResearchReddit": ["deepseek-v4-flash"]}
SUBAGENT_MODEL_CHAIN_BY_AGENT: dict[str, list[str]] = {}

# Controls the destination of the research outputs
# Options: "file" (standard Markdown), "db" (structured database), "both"
OUTPUT_MODE = "file"

# Controls where logs are stored
# Options: "file" (default), "db", "both"
LOG_MODE = "file" #os.environ.get("LOG_MODE", "file")

# Structured, content-rich per-run trace (deep_research/trace.py). This is the
# opt-in "read the full prompts / thinking / tool calls / source rationale"
# channel; it is independent of the content-free product events.
#   LOGGING_ENABLED   Master toggle for trace records. Default ON everywhere.
#                     Env: LOGGING_ENABLED=true|1|yes|on (0/false/no/off disables)
#   LOG_TRUNCATION    Max characters per string in a trace record. Default None
#                     = NO truncation. Env: LOG_TRUNCATION=<int>
LOGGING_ENABLED: bool = os.environ.get(
    "LOGGING_ENABLED", "true"
).strip().lower() not in {"0", "false", "no", "off"}

_log_truncation_env = os.environ.get("LOG_TRUNCATION", "").strip()
LOG_TRUNCATION: Optional[int] = (
    int(_log_truncation_env) if _log_truncation_env.isdigit() else None
)


# ===== PLATFORM AGENTS =====
# Sub-agents the supervisor may call, as AGENT_REGISTRY keys (see
# deep_research/agents/__init__.py).
#
# Configured via the ENABLED_AGENTS environment variable as a comma-separated
# list of tool-schema names, e.g.:
#     ENABLED_AGENTS=ResearchWeb,ResearchReddit,ResearchSubstack
#
# When the env var is unset or empty, the deployment default is the WEB agent
# only (paired with the default WEB_SEARCH_ENGINE="exa"). Per-run selection is
# also possible via RunConfig.enabled_agents (see deep_research/run_config.py).
#
# Order matters: the order below (or in the env var) is the order the tool
# schemas are bound to the supervisor model.
DEFAULT_PLATFORM = "reddit"  # Default platform for standalone run_platform.py

# All known agent tool-schema names (single source of truth for validation).
KNOWN_AGENT_NAMES: tuple = (
    "ResearchWeb",
    "ResearchGeneral",
    "ResearchReddit",
    "ResearchSubstack",
    "ResearchPubMed",
    "ResearchArxiv",
    "ResearchSEC",
)

_ENABLED_AGENTS_ENV = os.environ.get("ENABLED_AGENTS", "")
ENABLED_AGENTS = (
    [n.strip() for n in _ENABLED_AGENTS_ENV.split(",") if n.strip()]
    or ["ResearchWeb"]
)

# Platforms the ResearchGeneral / "general" platform agent may use, as keys of
# PLATFORMS in deep_research/agents/base.py. Empty list = ALL platforms
# (default). Configured via the GENERAL_AGENT_PLATFORMS env variable as a
# comma-separated list of platform keys, e.g.
#     GENERAL_AGENT_PLATFORMS=web,reddit,pubmed
KNOWN_PLATFORM_KEYS: tuple = (
    "web", "general", "reddit", "substack", "pubmed", "arxiv", "sec_edgar",
)
_GENERAL_AGENT_PLATFORMS_ENV = os.environ.get("GENERAL_AGENT_PLATFORMS", "")
GENERAL_AGENT_PLATFORMS: list[str] = [
    n.strip() for n in _GENERAL_AGENT_PLATFORMS_ENV.split(",") if n.strip()
]

# Per-call timeouts used by the platform agent engine and report writer
LLM_TIMEOUT = 180
TOOL_TIMEOUT = 120

# Base URLs used by get_model() for the native (non-OpenRouter) providers
MIMO_BASE_URL = "https://api.xiaomimimo.com/v1"
META_BASE_URL = "https://api.meta.ai/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# DeepSeek V4 Flash hosted on OpenRouter via Alibaba Cloud International (Singapore)
BABA_MODEL_ID = "deepseek/deepseek-v4-flash"
BABA_PROVIDER_SLUG = "alibaba"

# ── ROUTING: OpenRouter vs direct API ──────────────────────────────────
# ROUTE_VIA_OPENROUTER is the SHARED DEFAULT for every role. Each role also has
# an explicit override flag below; when a per-role flag is left UNSET, that role
# inherits this global default. Values: "true"/"1"/"yes"/"on" route via
# OpenRouter; "false"/"0"/"no"/"off" force the direct API.
#
#   ROUTE_VIA_OPENROUTER                shared default (global toggle)
#   SUBAGENT_ROUTE_VIA_OPENROUTER       platform research agents
#   SUPERVISOR_ROUTE_VIA_OPENROUTER     supervisor decision +
#                                       final report write. These three are CACHE-TIED:
#                                       they must share the same provider/model so the
#                                       prompt-cache prefix hits, so they always use the
#                                       supervisor flag together.
#   DRAFT_ROUTE_VIA_OPENROUTER          the research brief and initial draft report nodes.
#                                       These passes are cold/non-cacheable, so they may
#                                       route independently (e.g. to OpenRouter for a
#                                       reasoning-effort draft).
#
# Example:
#   ROUTE_VIA_OPENROUTER=true                 (everyone routes via OpenRouter)
#   SUPERVISOR_ROUTE_VIA_OPENROUTER=false     (supervisor/write stay native)
#   DRAFT_ROUTE_VIA_OPENROUTER=true           (draft goes via OpenRouter)
ROUTE_VIA_OPENROUTER = os.environ.get("ROUTE_VIA_OPENROUTER", "").strip().lower() in {"1", "true", "yes", "on"}
SUBAGENT_ROUTE_VIA_OPENROUTER = os.environ.get("SUBAGENT_ROUTE_VIA_OPENROUTER", "").strip().lower()
SUPERVISOR_ROUTE_VIA_OPENROUTER = os.environ.get("SUPERVISOR_ROUTE_VIA_OPENROUTER", "").strip().lower()
DRAFT_ROUTE_VIA_OPENROUTER = os.environ.get("DRAFT_ROUTE_VIA_OPENROUTER", "").strip().lower()

# Reasoning effort for the research brief and initial draft report nodes.
#                            "" = provider default.
#   "" / "low" / "medium" / "high"
# NOTE: effort tiers are an OpenRouter "reasoning.effort" convention. Native DeepSeek
# only supports thinking enabled/disabled, so effort is only honored when the brief and
# draft are routed via OpenRouter (DRAFT_ROUTE_VIA_OPENROUTER=true). Set it together
# with DRAFT_REPORT_MODEL (draft model) and DRAFT_REPORT_MODEL_FALLBACK_CHAIN (draft chain).
DRAFT_REPORT_REASONING_EFFORT = os.environ.get("DRAFT_REPORT_REASONING_EFFORT", "").strip().lower()

# ── Chinese content moderation guard ───────────────────────────────────
# Set CHINESE_MODERATION=true when the model chain uses a DIRECT Chinese API
# (e.g. native DeepSeek / MiMo), whose content filter rejects requests whose
# input history contains certain terms. When on:
#   - sources whose text matches SENSITIVE_CONTENT_PATTERNS are dropped before
#     the model ever sees them,
#   - suspiciously short / refusal-style supervisor output triggers a retry
#     with a vague continuation nudge instead of ending the run.
# deepseek-baba-singapore and other OpenRouter-hosted models are NOT moderated
# by the same filter — leave this off for them.
CHINESE_MODERATION = os.environ.get("CHINESE_MODERATION", "").strip().lower() in {"1", "true", "yes", "on"}

# Mixed-team case: a Chinese supervisor delegating to international sub-agents.
# Set CHINESE_SUPERVISOR_INTERNATIONAL_SUBAGENT=true to append one vague line to
# sub-agent prompts telling them to keep subject references neutral, so the
# supervisor's next (Chinese-model) call isn't rejected by terms the sub-agent
# surfaced.
CHINESE_SUPERVISOR_INTERNATIONAL_SUBAGENT = os.environ.get("CHINESE_SUPERVISOR_INTERNATIONAL_SUBAGENT", "").strip().lower() in {"1", "true", "yes", "on"}

# ← EDIT ME: regex terms that must never enter model context. One pattern per
# entry. Empty list = filter disabled.
SENSITIVE_CONTENT_PATTERNS: list[str] = [
    # Xi Jinping / senior CCP leadership
    r"习近平",
    r"習近平",
    r"\bXi\s+Jinping\b",

    # Mao
    r"毛泽东",
    r"毛澤東",
    r"\bMao\s+Zedong\b",

    # Tiananmen / June 4
    r"天安门事件",
    r"天安門事件",
    r"六四事件",
    r"六四",
    r"Tiananmen\s+(?:Square|massacre|incident|crackdown)",
    r"June\s+Fourth",
    r"Tank\s+Man",

    # CCP itself
    r"中国共产党",
    r"中國共產黨",
    r"\bCCP\b",
    r"Chinese\s+Communist\s+Party",

    # Falun Gong
    r"法轮功",
    r"法輪功",
    r"\bFalun\s+Gong\b",

    # Dalai Lama
    r"达赖喇嘛",
    r"達賴喇嘛",
    r"\bDalai\s+Lama\b",

    # Xinjiang/Uyghur specifically in political contexts
    r"新疆人权",
    r"新疆人權",
    r"\bUyghur\s+(?:genocide|persecution|camps)\b",
    r"\bUighur\s+(?:genocide|persecution|camps)\b",

    # Taiwan independence
    r"台湾独立",
    r"臺灣獨立",
    r"台独",
    r"臺獨",
    r"\bTaiwan\s+independence\b",

    # Hong Kong political movements
    r"反送中",
    r"雨伞运动",
    r"雨傘運動",
    r"Umbrella\s+Movement",
    r"占中",
    r"佔中",
    r"香港独立",
    r"香港獨立",

    # Chinese political movements
    r"白纸运动",
    r"白紙運動",
    r"White\s+Paper\s+Movement",
    r"茉莉花革命",
    r"Jasmine\s+Revolution",
    r"零八宪章",
    r"零八憲章",
    r"Charter\s+08",
]

_sensitive_re = None


def contains_sensitive(text: str) -> bool:
    """True if any SENSITIVE_CONTENT_PATTERNS matches text (single regex pass)."""
    global _sensitive_re
    if not SENSITIVE_CONTENT_PATTERNS:
        return False
    if _sensitive_re is None:
        _sensitive_re = re.compile("|".join(SENSITIVE_CONTENT_PATTERNS), re.IGNORECASE)
    return bool(_sensitive_re.search(text))


# Loose refusal signatures (EN + CN) for the short-output supervisor retry.
_REFUSAL_MARKERS = (
    "rejected because", "considered high risk", "content filter", "content policy",
    "cannot", "i can't", "i can’t", "unable to", "refuse", "敏感", "无法", "拒绝",
)


def looks_like_refusal(text: str) -> bool:
    """Lenient check: very short output that reads like a content-filter refusal."""
    if not text or len(text.strip()) > 200:
        return False
    lowered = text.lower()
    return any(marker in lowered for marker in _REFUSAL_MARKERS)

_TRUE_FLAG_VALUES = {"1", "true", "yes", "on"}
_FALSE_FLAG_VALUES = {"0", "false", "no", "off"}


def _resolve_route_flag(role_flag: str, explicit: bool | None = None) -> bool:
    """Resolve a role's OpenRouter routing to a bool.

    Precedence: explicit per-call arg > per-role env flag > global
    ROUTE_VIA_OPENROUTER default. The model factories resolve their own role's
    flag and pass the bool into get_model().
    """
    if explicit is not None:
        return explicit
    if role_flag in _TRUE_FLAG_VALUES:
        return True
    if role_flag in _FALSE_FLAG_VALUES:
        return False
    return ROUTE_VIA_OPENROUTER

# When ON, disable model fallback chains entirely: every get_*_model factory
# returns the bare primary model, so a failure raises instead of silently
# switching to another model. Set DISABLE_MODEL_FALLBACK=1/true/yes/on.
DISABLE_MODEL_FALLBACK = os.environ.get("DISABLE_MODEL_FALLBACK", "yes").strip().lower() in {"1", "true", "yes", "on"}

# Which web search engine to expose to the web/general platform agents.
# "tavily" = only Tavily; "exa" = only Exa (default); "both" = expose both.
WEB_SEARCH_ENGINE = os.environ.get("WEB_SEARCH_ENGINE", "exa").strip().lower()

# Max characters of extracted page text Exa returns per search result. This
# text is retained as the article's full text and served back on fetch_urls
# (so the same page is not re-fetched via Tavily). Bundled into the /search
# base price regardless of size.
EXA_SEARCH_MAX_CHARS = int(os.environ.get("EXA_SEARCH_MAX_CHARS", "6000"))

# Max characters of page text returned by fetch_urls via Tavily extract.
# Tavily extract returns the full page with no size bound, which could blow
# up the sub-agent context. Truncate to this cap before entering the context.
# Env: FETCH_URL_MAX_CHARS
FETCH_URL_MAX_CHARS = int(os.environ.get("FETCH_URL_MAX_CHARS", "10000"))

# Model name → OpenRouter slug when ROUTE_VIA_OPENROUTER is ON for that role.
# meta models are handled separately (slug is "meta/<model-name>").
_OPENROUTER_MODEL_MAP_BY_NAME = {
    "deepseek-v4-pro": "deepseek/deepseek-v4-pro",
    "deepseek-v4-flash": "deepseek/deepseek-v4-flash",
    "mimo-v2.5": "xiaomi/mimo-v2.5",
    "mimo-v2.5-pro": "xiaomi/mimo-v2.5-pro",
}

# ── Per-run credential resolver ──────────────────────────────────────────
# Model clients are created from a per-run RuntimeContext/ModelFactory. The
# factory sets this ContextVar to a ``credentials.get`` resolver for the
# duration of each model construction, so key lookups honor per-run
# credentials without ever mutating the process environment. When no factory
# is active (legacy/CLI), lookups fall back to ``os.getenv``.
_cred_resolver_var: contextvars.ContextVar = contextvars.ContextVar(
    "deep_research_cred_resolver", default=None
)


def _active_cred_resolver():
    return _cred_resolver_var.get()


class credential_scope:
    """Context manager installing a per-run credential resolver.

    Usage (inside a ModelFactory):

        with credential_scope(credentials):
            model = get_supervisor_model(...)
    """

    def __init__(self, credentials=None):
        self._credentials = credentials
        self._token = None

    def __enter__(self):
        resolver = None
        if self._credentials is not None:
            getter = getattr(self._credentials, "get", None)
            if callable(getter):
                resolver = getter
        self._token = _cred_resolver_var.set(resolver)
        return self

    def __exit__(self, *exc):
        _cred_resolver_var.reset(self._token)
        return False


def model_required_keys(model_name: str, route_via_openrouter: bool = False) -> list:
    """Env-var names whose presence makes ``model_name`` usable.

    Mirrors get_model()'s provider detection (including OpenRouter routing).
    Returns [] for providers that need no key or are unknown (unknown providers
    are left to get_model() to raise its own mapping error).
    Alternative spellings are listed so validation can accept any of them
    (e.g. DEEPSEEK_API_KEY or DEEPSEEK_KEY).
    """
    model_lower = model_name.lower()
    if route_via_openrouter:
        slug = _OPENROUTER_MODEL_MAP_BY_NAME.get(model_lower)
        if slug is None and "muse" in model_lower and not model_name.startswith("meta/"):
            slug = f"meta/{model_name}"
        if slug is not None:
            return ["OPENROUTER_API_KEY"]
    if "glm" in model_lower:
        return ["ZHIPUAI_API_KEY"]
    if "nemotron" in model_lower or model_name.startswith("nvidia/"):
        return ["OPENROUTER_API_KEY"]
    if model_lower == "deepseek-baba-singapore":
        return ["OPENROUTER_API_KEY"]
    if "deepseek" in model_lower:
        return ["DEEPSEEK_API_KEY", "DEEPSEEK_KEY"]
    if "mimo" in model_lower:
        return ["MIMO_API_KEY"]
    if model_name.startswith("meta/"):
        return ["OPENROUTER_API_KEY"]
    if "muse" in model_lower:
        return ["META_API_KEY"]
    if "gemini" in model_lower:
        return ["GEMINI_API_KEY", "GOOGLE_API_KEY"]
    if "gpt" in model_lower or "o1" in model_lower or "o3" in model_lower:
        return ["OPENAI_API_KEY"]
    return []


def _model_key_available(model_name: str, route_via_openrouter: bool, resolver=None) -> bool:
    """True if the model's required API key is present.

    Mirrors get_model()'s provider detection (including OpenRouter routing) so
    the model-chain factories can skip a model whose key is missing WITHOUT
    crashing at import time. Returns True for providers that need no key.
    ``resolver`` (optional) supplies keys per-run; falls back to the process
    environment.
    """
    if resolver is None:
        resolver = _active_cred_resolver()

    def _has(env_name: str) -> bool:
        if resolver is not None:
            val = resolver(env_name)
            if val:
                return True
        return bool(os.getenv(env_name))

    alternatives = model_required_keys(model_name, route_via_openrouter)
    if not alternatives:
        # Unknown provider: let get_model() raise its own mapping error rather
        # than silently skipping the model here.
        return True
    return any(_has(env_name) for env_name in alternatives)


def _filter_usable_chain(chain: list, route_via_openrouter: bool, role: str, resolver=None) -> list:
    """Drop chain entries whose API key is missing, warning for each.

    Used by the model factories so a misconfigured chain (missing key for one
    entry) degrades to the next usable model instead of crashing at import.
    ``resolver`` (optional) supplies keys per-run. Returns the filtered chain;
    the caller raises if nothing remains.
    """
    usable = []
    for model_name in chain:
        if _model_key_available(model_name, route_via_openrouter, resolver=resolver):
            usable.append(model_name)
        else:
            _fallback_logger.warning(
                f"Skipping {model_name} in {role} model chain: required API key "
                "is not set in .env. Falling back to the next usable model."
            )
    return usable


def _require_api_key(env_var: str, model_name: str, *alt_env_vars: str, resolver=None) -> str:
    """Raise a clear ValueError if a model's API key is missing.

    ChatOpenAI/ChatDeepSeek raise a cryptic pydantic error when api_key is
    None; this surfaces an actionable message naming the env var(s) instead.
    ``resolver`` (optional) is a callable ``env_var -> str|None`` consulted
    before ``os.getenv`` so a per-run Credentials object can supply keys
    without touching the process environment.
    """
    if resolver is None:
        resolver = _active_cred_resolver()
    for var in (env_var, *alt_env_vars):
        value = None
        if resolver is not None:
            value = resolver(var)
        if not value:
            value = os.getenv(var)
        if value:
            return value
    alts = f" (or {', '.join(alt_env_vars)})" if alt_env_vars else ""
    raise ValueError(
        f"{env_var}{alts} not found in .env for model {model_name!r}. "
        f"Add the key or set the model chain to a provider you have a key for."
    )


# ── THINKING MODE ─────────────────────────────────────────────────────
# Controls explicit reasoning-enablement for every model in the pipeline.
#   "auto" — rely on the provider's default thinking (all configured models
#            reason by default, confirmed by spike_thinking.py probes); fall
#            back to an explicit enable param where a provider needs one.
#   "on"   — always pass an explicit enable param with a proven format.
#   "off"  — never pass an explicit thinking param (providers that reason by
#            default still do; this only stops us from forcing it).
# Captured reasoning is round-tripped to the API regardless of this flag.
THINKING_MODE = "auto"

# Per-provider thinking enablement, keyed by the tags used in get_model.
# auto_param is used when THINKING_MODE="auto" (None means
# the provider reasons by default and needs no explicit param), on_param when
# THINKING_MODE="on". Providers absent from this map get no explicit param but
# still benefit from reasoning capture (e.g. gemini is handled separately).
_THINKING_CONFIG = {
    "deepseek": {
        "auto_param": None,  # DeepSeek thinking mode is ON by default (effort high);
                             # the explicit thinking:enabled param is redundant. Forced
                             # tool_choice (with_structured_output function_calling) is
                             # rejected in thinking mode regardless of this param, so the
                             # scope node uses json_mode (json_object) instead.
        "on_param": {"thinking": {"type": "enabled"}},
    },
    "deepseek-openrouter": {
        "auto_param": None,
        "on_param": {"reasoning": {"effort": "medium"}},
    },
    "mimo": {
        "auto_param": None,
        "on_param": {"thinking": {"type": "enabled"}},
    },
    "mimo-openrouter": {
        "auto_param": None,
        "on_param": {"reasoning": {"effort": "medium"}},
    },
    "meta-openrouter": {
        "auto_param": None,
        "on_param": {"reasoning": {"effort": "medium"}},
    },
}


def _thinking_params(provider_key: str, mode: str = None) -> dict | None:
    """Return the extra_body params that enable thinking for a provider, or None."""
    mode = THINKING_MODE if mode is None else mode
    if mode == "off":
        return None
    cfg = _THINKING_CONFIG.get(provider_key) or {}
    return cfg.get("on_param" if mode == "on" else "auto_param")


def subagent_output_mode(discovery: bool = False) -> str:
    """Resolve the default output_mode for a sub-agent based on its mode.

    Discovery agents default to the full-context report-inline flow
    (DISCOVERY_OUTPUT_MODE); research agents use SUBAGENT_OUTPUT_MODE. A caller
    may still override the resolved value per call.
    """
    return DISCOVERY_OUTPUT_MODE if discovery else SUBAGENT_OUTPUT_MODE


# ── DeepSeek reasoning_content passthrough ─────────────────────────────
# langchain_openai doesn't serialize reasoning_content back to the API.
# This patch preserves DeepSeek's thinking tokens across message round-trips.
# REQUIRED for thinking-mode DeepSeek models; applied at import time.

import langchain_openai.chat_models.base as _lc_base

if not getattr(_lc_base._convert_message_to_dict, "_reasoning_patched", False):
    _original_convert_msg_to_dict = _lc_base._convert_message_to_dict

    def _patched_convert_message_to_dict(message, api="chat/completions"):
        msg_dict = _original_convert_msg_to_dict(message, api)
        if isinstance(message, AIMessage):
            reasoning = message.additional_kwargs.get("reasoning_content")
            if reasoning:
                msg_dict["reasoning_content"] = reasoning
        return msg_dict

    _patched_convert_message_to_dict._reasoning_patched = True
    _lc_base._convert_message_to_dict = _patched_convert_message_to_dict


# Generic reasoning CAPTURE for OpenAI-compatible providers. langchain_openai
# drops vendor reasoning fields (reasoning_content / reasoning / reasoning_details)
# from responses. Mirror ChatDeepSeek's own capture so every ChatOpenAI-based
# model (mimo, meta, OpenRouter, nemotron, glm) round-trips reasoning.
# ChatDeepSeek's override calls super() into BaseChatOpenAI, so it is covered
# by this patch too (the double-set is idempotent).
def _extract_reasoning_from_message(message) -> str | None:
    def _first(obj) -> str | None:
        for key in ("reasoning_content", "reasoning", "reasoning_details"):
            val = obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)
            if val:
                if isinstance(val, str):
                    return val
                return json.dumps(val, ensure_ascii=False)[:20000]
        return None

    found = _first(message)
    if found is None:
        extra = getattr(message, "model_extra", None)
        if isinstance(extra, dict):
            found = _first(extra)
    return found


if not getattr(_lc_base.BaseChatOpenAI._create_chat_result, "_reasoning_capture_patched", False):
    _original_create_chat_result = _lc_base.BaseChatOpenAI._create_chat_result

    def _patched_create_chat_result(self, response, generation_info=None):
        rtn = _original_create_chat_result(self, response, generation_info)
        try:
            choices = getattr(response, "choices", None)
            if choices and rtn.generations:
                reasoning = _extract_reasoning_from_message(choices[0].message)
                if reasoning:
                    rtn.generations[0].message.additional_kwargs["reasoning_content"] = reasoning
        except Exception:
            pass
        return rtn

    _patched_create_chat_result._reasoning_capture_patched = True
    _lc_base.BaseChatOpenAI._create_chat_result = _patched_create_chat_result


# ===== MODEL-NAME FACTORY =====

def get_model(model_name: str = SUBAGENT_MODEL, temperature: float = 0, max_tokens: int = None,
              route_via_openrouter: bool = None, reasoning_effort: str = None,
              thinking_mode: str = None):
    """
    Get a chat model instance based on the model name.

    Automatically detects the provider based on the model name and returns
    the appropriate LangChain chat model instance.

    Args:
        model_name: Name of the model (e.g., "gpt-5.2", "gemini-3-flash-preview")
        temperature: Temperature for generation (default: 0)
        max_tokens: Maximum tokens for generation (default: None, uses model default)
        route_via_openrouter: Force OpenRouter routing for this call (True/False).
            None (default) falls back to ROUTE_VIA_OPENROUTER / the caller's
            role-specific resolution.
        reasoning_effort: "low" / "medium" / "high" reasoning effort. Only honored
            when the call is routed via OpenRouter (native DeepSeek only supports
            thinking enabled/disabled). Default None = provider default.
        thinking_mode: Optional per-run override for THINKING_MODE
            ("auto"/"on"/"off"). None uses the module default.

    Returns:
        LangChain chat model instance

    Raises:
        ValueError: If the model name cannot be mapped to a known provider,
            or reasoning_effort is not a valid value.
    """
    model_lower = model_name.lower()
    _route = ROUTE_VIA_OPENROUTER if route_via_openrouter is None else bool(route_via_openrouter)
    _eff_mode = THINKING_MODE if thinking_mode is None else thinking_mode

    def _tp(provider_key: str) -> dict | None:
        return _thinking_params(provider_key, mode=_eff_mode)

    # Validate reasoning_effort up front so a typo fails fast with a clear message.
    _effort_body = None
    if reasoning_effort is not None:
        reasoning_effort = reasoning_effort.strip().lower()
        if reasoning_effort not in ("low", "medium", "high"):
            raise ValueError(
                f"Invalid reasoning_effort: {reasoning_effort!r}. Use 'low', 'medium', or 'high'."
            )
        _effort_body = {"reasoning": {"effort": reasoning_effort}}

    # ROUTE_VIA_OPENROUTER: send known direct-API models through OpenRouter so a
    # single OPENROUTER_API_KEY covers the supervisor chain, writers, and search.
    if _route:
        slug = _OPENROUTER_MODEL_MAP_BY_NAME.get(model_lower)
        if slug is None and "muse" in model_lower and not model_name.startswith("meta/"):
            slug = f"meta/{model_name}"
        if slug is not None:
            api_key = _require_api_key("OPENROUTER_API_KEY", model_name)
            kwargs = {
                "model": slug,
                "api_key": api_key,
                "base_url": "https://openrouter.ai/api/v1",
                "temperature": temperature,
            }
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            if "deepseek" in model_lower:
                provider_key = "deepseek-openrouter"
            elif "mimo" in model_lower:
                provider_key = "mimo-openrouter"
            elif "muse" in model_lower or model_name.startswith("meta/"):
                provider_key = "meta-openrouter"
            else:
                provider_key = None
            extra_body = {}
            if _effort_body:
                extra_body.update(_effort_body)
            else:
                thinking = _tp(provider_key) if provider_key else None
                if thinking:
                    extra_body.update(thinking)
            if extra_body:
                kwargs["extra_body"] = extra_body
            return ChatOpenAI(**kwargs)

    # Z.AI GLM models (OpenAI-compatible API)
    if "glm" in model_lower:
        kwargs = {
            "model": model_name,
            "api_key": _require_api_key("ZHIPUAI_API_KEY", model_name),
            "base_url": "https://api.z.ai/api/paas/v4/",
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        return ChatOpenAI(**kwargs)

    # OpenRouter-hosted Nemotron models (OpenAI-compatible API)
    if "nemotron" in model_lower or model_name.startswith("nvidia/"):
        extra_body = {"provider": {"order": ["CoreWeave"]}}
        if _effort_body:
            extra_body = {**extra_body, **_effort_body}
        kwargs = {
            "model": model_name,
            "api_key": _require_api_key("OPENROUTER_API_KEY", model_name),
            "base_url": "https://openrouter.ai/api/v1",
            "temperature": temperature,
            # Pin OpenRouter routing to CoreWeave (cheap, low-latency inference
            # for the search model; see nvidia/nemotron-3.5-lightning providers).
            "extra_body": extra_body,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        return ChatOpenAI(**kwargs)

    # DeepSeek V4 Flash hosted on OpenRouter via Alibaba Cloud (Singapore).
    # Kept as a model NAME in chains so routing pins are just chain entries.
    if model_lower == "deepseek-baba-singapore":
        _require_api_key("OPENROUTER_API_KEY", model_name)
        extra_body = {"provider": {"order": [BABA_PROVIDER_SLUG]}}
        if _effort_body:
            extra_body = {**extra_body, **_effort_body}
        else:
            thinking = _tp("deepseek-openrouter")
            if thinking:
                extra_body = {**extra_body, **thinking}
        kwargs = {
            "model": BABA_MODEL_ID,
            "api_key": _require_api_key("OPENROUTER_API_KEY", model_name),
            "base_url": OPENROUTER_BASE_URL,
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        kwargs["extra_body"] = extra_body
        return ChatOpenAI(**kwargs)

    # DeepSeek models (OpenAI-compatible API)
    if "deepseek" in model_lower:
        from langchain_deepseek import ChatDeepSeek
        if _effort_body:
            _fallback_logger.warning(
                "reasoning_effort ignored for %s: the native DeepSeek API only supports "
                "thinking enabled/disabled. Route via OpenRouter (e.g. "
                "DRAFT_ROUTE_VIA_OPENROUTER=true) to honor effort levels.", model_name
            )
        kwargs = {
            "model": model_name,
            "api_key": _require_api_key("DEEPSEEK_API_KEY", model_name, "DEEPSEEK_KEY"),
            "api_base": "https://api.deepseek.com/v1",
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        thinking = _tp("deepseek")
        if thinking:
            kwargs["extra_body"] = thinking
        return ChatDeepSeek(**kwargs)

    # MiMo models (native API: api.xiaomimimo.com). Hosts both
    # "mimo-v2.5-pro" (flagship) and "mimo-v2.5" (lighter, ~half cost).
    # To route MiMo through OpenRouter instead, use ROUTE_VIA_OPENROUTER
    # (slug xiaomi/mimo-v2.5-pro / xiaomi/mimo-v2.5).
    if "mimo" in model_lower:
        if _effort_body:
            _fallback_logger.warning(
                "reasoning_effort ignored for %s: the native MiMo API does not support "
                "effort levels. Route via OpenRouter to honor them.", model_name
            )
        kwargs = {
            "model": model_lower if model_lower in ("mimo-v2.5", "mimo-v2.5-pro") else "mimo-v2.5-pro",
            "api_key": _require_api_key("MIMO_API_KEY", model_name),
            "base_url": MIMO_BASE_URL,
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        thinking = _tp("mimo")
        if thinking:
            kwargs["extra_body"] = thinking
        return ChatOpenAI(**kwargs)

    # Meta Muse models on OpenRouter (OpenAI-compatible API). Slugs look like
    # "meta/muse-spark-1.2" or "meta/muse-glimmer-30b".
    if model_name.startswith("meta/"):
        extra_body = {}
        if _effort_body:
            extra_body.update(_effort_body)
        kwargs = {
            "model": model_name,
            "api_key": _require_api_key("OPENROUTER_API_KEY", model_name),
            "base_url": "https://openrouter.ai/api/v1",
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if extra_body:
            kwargs["extra_body"] = extra_body
        return ChatOpenAI(**kwargs)

    # Meta Muse models on the native Meta Model API (OpenAI-compatible).
    # Model ids look like "muse-spark-1.2-contributor".
    if "muse" in model_lower:
        if _effort_body:
            _fallback_logger.warning(
                "reasoning_effort ignored for %s: the native Meta Model API does not "
                "support effort levels. Route via OpenRouter to honor them.", model_name
            )
        kwargs = {
            "model": model_name,
            "api_key": _require_api_key("META_API_KEY", model_name),
            "base_url": META_BASE_URL,
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        return ChatOpenAI(**kwargs)

    # Google Gemini models
    if "gemini" in model_lower:
        # Use google_genai provider prefix for init_chat_model
        if not model_name.startswith("google_genai:"):
            model_name = f"google_genai:{model_name}"
        kwargs = {
            "temperature": temperature,
            "max_retries": 1,  # Set to 1 to disable SDK retries (0 defaults to 5). This allows LangChain fallback to trigger.
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        # Enable Gemini thinking via thinking_config in the generation config.
        # Untested (no working GEMINI_API_KEY available during development).
        if _eff_mode != "off":
            kwargs["generation_config"] = {"thinking_config": {"thinking_budget": 1}}
        return init_chat_model(model_name, **kwargs)

    # OpenAI models (default fallback)
    if "gpt" in model_lower or "o1" in model_lower or "o3" in model_lower:
        if not model_name.startswith("openai:"):
            model_name = f"openai:{model_name}"
        kwargs = {"temperature": temperature}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        return init_chat_model(model_name, **kwargs)

    # If we can't determine the provider, raise an error
    raise ValueError(
        f"Cannot determine provider for model: {model_name}. "
        f"Supported providers: OpenAI (gpt-*), Google (gemini-*), "
        f"Z.AI GLM (glm-*), DeepSeek (deepseek-*), MiMo (mimo-*), "
        f"Meta Muse (muse-* / meta/*), OpenRouter Nemotron (nemotron-* / nvidia/*)"
    )


# ===== RESILIENT INVOCATION =====

import logging

_fallback_logger = logging.getLogger("model_fallback")


def _fallback_exceptions() -> tuple:
    """Exception types that trigger model fallback."""
    try:
        from google.genai.errors import ClientError, ServerError, APIError
        return (ClientError, ServerError, APIError, Exception)
    except Exception:
        return (Exception,)


def get_supervisor_model(tools: list = None, max_tokens: int = None, temperature: float = 0,
                         chain: list = None, route_via_openrouter: bool = None,
                         reasoning_effort: str = None, thinking_mode: str = None,
                         disable_fallback: bool = None):
    """Get a resilient model runnable for the supervisor role only.

    Uses SUPERVISOR_MODEL_FALLBACK_CHAIN (defaults to deepseek-v4-flash →
    deepseek-v4-pro) with LangChain's .with_fallbacks() mechanism. Pass `chain`
    to override the chain per invocation (e.g. the draft report writer).

    Routing resolves via SUPERVISOR_ROUTE_VIA_OPENROUTER (inheriting
    ROUTE_VIA_OPENROUTER) unless an explicit `route_via_openrouter` is given.
    NOTE: supervisor and the final report write all build
    through this factory, so they always share the same routing/provider —
    required for the prompt-cache prefix to hit.
    """
    chain = chain or SUPERVISOR_MODEL_FALLBACK_CHAIN
    if not chain:
        raise ValueError("Supervisor model chain cannot be empty")

    _route = _resolve_route_flag(SUPERVISOR_ROUTE_VIA_OPENROUTER, route_via_openrouter)
    chain = _filter_usable_chain(chain, _route, "supervisor")
    if not chain:
        raise ValueError(
            "Supervisor model chain is unusable: no model has its required API "
            "key set in .env. Check SUPERVISOR_MODEL_FALLBACK_CHAIN and the "
            "OPENROUTER_API_KEY / DEEPSEEK_API_KEY env vars."
        )
    _disable = DISABLE_MODEL_FALLBACK if disable_fallback is None else bool(disable_fallback)
    primary_model = get_model(chain[0], temperature=temperature, max_tokens=max_tokens,
                              route_via_openrouter=_route, reasoning_effort=reasoning_effort,
                              thinking_mode=thinking_mode)
    if tools:
        primary_model = primary_model.bind_tools(tools)

    if _disable or len(chain) < 2:
        return primary_model

    fallbacks = []
    for model_name in chain[1:]:
        fb = get_model(model_name, temperature=temperature, max_tokens=max_tokens,
                       route_via_openrouter=_route, reasoning_effort=reasoning_effort,
                       thinking_mode=thinking_mode)
        if tools:
            fb = fb.bind_tools(tools)
        fallbacks.append(fb)

    if not fallbacks:
        return primary_model

    _fallback_logger.info(f"Created supervisor model chain: {chain} (route_via_openrouter={_route})")
    return primary_model.with_fallbacks(
        fallbacks,
        exceptions_to_handle=_fallback_exceptions(),
    )


def get_draft_report_model(max_tokens: int = None, temperature: float = 0,
                           route_via_openrouter: bool = None,
                           reasoning_effort: str = None, thinking_mode: str = None,
                           disable_fallback: bool = None,
                           chain: list = None):
    """Get the resilient model chain for the research brief and initial draft.

    Uses DRAFT_REPORT_MODEL_FALLBACK_CHAIN. The research brief and initial draft
    are cold, non-cacheable passes, so DRAFT_REPORT_MODEL lets deployments point
    both at any model without touching the supervisor or platform agents.

    Routing resolves via DRAFT_ROUTE_VIA_OPENROUTER (inheriting
    ROUTE_VIA_OPENROUTER). Reasoning effort defaults to
    DRAFT_REPORT_REASONING_EFFORT and is only honored when routed via OpenRouter.
    """
    if reasoning_effort is None:
        reasoning_effort = DRAFT_REPORT_REASONING_EFFORT or None
    _route = _resolve_route_flag(DRAFT_ROUTE_VIA_OPENROUTER, route_via_openrouter)
    return get_supervisor_model(
        max_tokens=max_tokens,
        temperature=temperature,
        chain=chain or DRAFT_REPORT_MODEL_FALLBACK_CHAIN,
        route_via_openrouter=_route,
        reasoning_effort=reasoning_effort,
        thinking_mode=thinking_mode,
        disable_fallback=disable_fallback,
    )


def get_subagent_model(
    tools: list = None,
    max_tokens: int = None,
    temperature: float = 0,
    chain: list = None,
    route_via_openrouter: bool = None,
    thinking_mode: str = None,
    disable_fallback: bool = None,
):
    """Get a resilient model runnable for the sub-agent role only.

    Research platform sub-agents use SUBAGENT_MODEL_FALLBACK_CHAIN (defaults to
    deepseek-v4-flash) with LangChain's
    .with_fallbacks() mechanism. Pass `chain` to override the
    chain per invocation (e.g. run_platform.py selecting gemini or the
    Alibaba-Singapore deepseek by model name). Routing resolves via
    SUBAGENT_ROUTE_VIA_OPENROUTER (inheriting ROUTE_VIA_OPENROUTER).
    """
    chain = chain or SUBAGENT_MODEL_FALLBACK_CHAIN
    if not chain:
        raise ValueError("Sub-agent model chain cannot be empty")

    _route = _resolve_route_flag(SUBAGENT_ROUTE_VIA_OPENROUTER, route_via_openrouter)
    chain = _filter_usable_chain(chain, _route, "sub-agent")
    if not chain:
        raise ValueError(
            "Sub-agent model chain is unusable: no model has its required API "
            "key set in .env. Check SUBAGENT_MODEL_FALLBACK_CHAIN and the "
            "OPENROUTER_API_KEY / DEEPSEEK_API_KEY env vars."
        )
    _disable = DISABLE_MODEL_FALLBACK if disable_fallback is None else bool(disable_fallback)
    primary_model = get_model(chain[0], temperature=temperature, max_tokens=max_tokens,
                              route_via_openrouter=_route, thinking_mode=thinking_mode)
    if tools:
        primary_model = primary_model.bind_tools(tools)

    if _disable or len(chain) < 2:
        return primary_model

    fallbacks = []
    for model_name in chain[1:]:
        fb = get_model(model_name, temperature=temperature, max_tokens=max_tokens,
                       route_via_openrouter=_route, thinking_mode=thinking_mode)
        if tools:
            fb = fb.bind_tools(tools)
        fallbacks.append(fb)

    if not fallbacks:
        return primary_model

    _fallback_logger.info(f"Created sub-agent model chain: {chain}")
    return primary_model.with_fallbacks(
        fallbacks,
        exceptions_to_handle=_fallback_exceptions(),
    )
