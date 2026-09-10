"""Per-run configuration for the Deep Dog research engine.

The standalone CLI has long read every knob from environment variables at
module import time. For a concurrent API product that is unsafe: one process
must host many runs with different models, timing and depth limits without
mutating ``os.environ``.

``RunConfig`` is a resolved, per-run configuration. Resolution precedence:

    environment/deployment defaults   (config.py module constants, set from
                                       env at import time)
        ->  RunConfig (constructed by the host; omitted fields fall back to
            the env defaults above)
        ->  apply_overrides()/request overrides
        ->  hard server safety caps   (applied by ``finalize()``)

The engine never modifies ``os.environ`` for a request. ``RunConfig`` is a
plain dataclass of JSON-serializable values, so a snapshot can be embedded in
checkpointed graph state and used to reconstruct the runtime on resume.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields, replace
from typing import Any, Optional

from deep_research import config as _cfg

# ── Hard server safety caps ──────────────────────────────────────────────
# These bound any request value regardless of profile/request overrides.
HARD_MAX_DURATION_MINUTES = 240.0   # 4 hours absolute ceiling for one run
HARD_MIN_DURATION_MINUTES = 0.25    # 15s floor (below that is a config bug)
HARD_MAX_SUPERVISOR_ITERATIONS = 500
HARD_MAX_SUBAGENT_ITERATIONS = 60
HARD_MAX_SUBAGENT_READS = 200
HARD_MAX_SUBAGENT_SAVES = 200
HARD_MAX_SUBAGENT_SEARCHES = 100
HARD_MAX_SUBAGENT_CONCURRENCY = 12
HARD_MAX_SUBAGENT_TOTAL_READS = 500
OUTPUT_MODES = ("file", "db", "both", "none")


def _clamp(value: float, lo: float, hi: float, name: str) -> float:
    if value < lo or value > hi:
        return max(lo, min(hi, value))
    return value


def _env_role_flag(value: Optional[str]) -> Optional[bool]:
    """Parse a role OpenRouter env flag into True/False, or None to inherit."""
    s = (value or "").strip().lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off"):
        return False
    return None


@dataclass
class RunConfig:
    """Fully-resolved per-run configuration (JSON-serializable snapshot)."""

    # ── Identity / labelling ───────────────────────────────────────────
    profile: Optional[str] = None          # human label, e.g. "muse-spark-1.2"
    prompt_version: str = _cfg.PROMPT_VERSION          # "OPEN" | "LEGACY"
    output_mode: str = _cfg.OUTPUT_MODE     # "file" | "db" | "both" | "none"
    log_mode: str = _cfg.LOG_MODE           # "file" | "db" | "both"
    language: Optional[str] = None          # optional seed target language

    # ── Model selection (role chains) ──────────────────────────────────
    supervisor_model_fallback_chain: list = field(
        default_factory=lambda: list(_cfg.SUPERVISOR_MODEL_FALLBACK_CHAIN))
    subagent_model_fallback_chain: list = field(
        default_factory=lambda: list(_cfg.SUBAGENT_MODEL_FALLBACK_CHAIN))
    draft_report_model_fallback_chain: list = field(
        default_factory=lambda: list(_cfg.DRAFT_REPORT_MODEL_FALLBACK_CHAIN))
    subagent_model_chain_by_agent: dict = field(
        default_factory=lambda: dict(_cfg.SUBAGENT_MODEL_CHAIN_BY_AGENT))
    disable_model_fallback: bool = _cfg.DISABLE_MODEL_FALLBACK

    # ── Routing (OpenRouter vs native) ─────────────────────────────────
    route_via_openrouter: bool = _cfg.ROUTE_VIA_OPENROUTER
    supervisor_route_via_openrouter: Optional[bool] = _env_role_flag(
        _cfg.SUPERVISOR_ROUTE_VIA_OPENROUTER)
    subagent_route_via_openrouter: Optional[bool] = _env_role_flag(
        _cfg.SUBAGENT_ROUTE_VIA_OPENROUTER)
    draft_route_via_openrouter: Optional[bool] = _env_role_flag(
        _cfg.DRAFT_ROUTE_VIA_OPENROUTER)
    draft_report_reasoning_effort: str = _cfg.DRAFT_REPORT_REASONING_EFFORT
    thinking_mode: str = _cfg.THINKING_MODE   # "auto" | "on" | "off"

    # ── Research timing (minutes) ──────────────────────────────────────
    research_time_min_minutes: float = _cfg.RESEARCH_TIME_MIN_MINUTES
    research_time_max_minutes: float = _cfg.RESEARCH_TIME_MAX_MINUTES
    # Optional explicit hard deadline for the run. When unset the engine uses
    # research_time_max_minutes + 1 (historical CLI behaviour). Clamped to
    # HARD_MAX_DURATION_MINUTES by finalize().
    max_duration_minutes: Optional[float] = None
    findings_salvage_time_fraction: float = _cfg.FINDINGS_SALVAGE_TIME_FRACTION

    # ── Supervisor limits ──────────────────────────────────────────────
    supervisor_max_iterations: int = _cfg.SUPERVISOR_MAX_ITERATIONS
    supervisor_timeout_seconds: int = _cfg.SUPERVISOR_TIMEOUT_SECONDS
    refine_timeout_seconds: int = _cfg.REFINE_TIMEOUT_SECONDS
    supervisor_max_concurrent_research: int = _cfg.SUPERVISOR_MAX_CONCURRENT_RESEARCH
    supervisor_max_concurrent_discovery: int = _cfg.SUPERVISOR_MAX_CONCURRENT_DISCOVERY

    # ── Sub-agent depth caps ───────────────────────────────────────────
    subagent_max_iterations: int = _cfg.SUBAGENT_MAX_ITERATIONS
    subagent_max_reads: int = _cfg.SUBAGENT_MAX_READS
    subagent_max_total_reads: int = _cfg.DEFAULT_MAX_TOTAL_READS
    subagent_max_saves: int = _cfg.SUBAGENT_MAX_SAVES
    subagent_max_searches: int = _cfg.SUBAGENT_MAX_SEARCHES
    subagent_max_concurrency: int = _cfg.SUBAGENT_MAX_CONCURRENCY
    subagent_timeout_seconds: int = _cfg.SUBAGENT_TIMEOUT_SECONDS

    # ── Agent filtering ────────────────────────────────────────────────
    enabled_agents: list = field(default_factory=lambda: list(_cfg.ENABLED_AGENTS))
    general_agent_platforms: list = field(
        default_factory=lambda: list(_cfg.GENERAL_AGENT_PLATFORMS))

    # ── Web search ─────────────────────────────────────────────────────
    web_search_engine: str = _cfg.WEB_SEARCH_ENGINE          # "tavily"|"exa"|"both"
    exa_search_max_chars: int = _cfg.EXA_SEARCH_MAX_CHARS
    fetch_url_max_chars: int = _cfg.FETCH_URL_MAX_CHARS

    # ── Output behaviour ───────────────────────────────────────────────
    save_report_to_file: bool = _cfg.SAVE_REPORT_TO_FILE
    save_subagent_reports_to_file: bool = _cfg.SAVE_SUBAGENT_REPORTS_TO_FILE
    enable_source_log: bool = _cfg.ENABLE_SOURCE_LOG
    enable_subtopic_generation: bool = _cfg.ENABLE_SUBTOPIC_GENERATION
    enable_research_trace: bool = _cfg.ENABLE_RESEARCH_TRACE
    subagent_output_mode: str = _cfg.SUBAGENT_OUTPUT_MODE    # "sources" | ...
    discovery_output_mode: str = _cfg.DISCOVERY_OUTPUT_MODE  # "report_inline" | ...

    # ── Chinese content moderation guard ──────────────────────────────
    chinese_moderation: bool = _cfg.CHINESE_MODERATION
    chinese_supervisor_international_subagent: bool = _cfg.CHINESE_SUPERVISOR_INTERNATIONAL_SUBAGENT
    sensitive_content_patterns: list = field(
        default_factory=lambda: list(_cfg.SENSITIVE_CONTENT_PATTERNS))

    # ── Per-call timeouts ──────────────────────────────────────────────
    llm_timeout: int = _cfg.LLM_TIMEOUT
    tool_timeout: int = _cfg.TOOL_TIMEOUT

    # ── Misc ───────────────────────────────────────────────────────────
    target_language_fallback: str = _cfg.TARGET_LANGUAGE_FALLBACK

    # ───────────────────────────────────────────────────────────────────
    def resolved_role_route(self, role: str) -> bool:
        """Resolve a role's OpenRouter routing to a bool using this config."""
        value = getattr(self, f"{role}_route_via_openrouter", None)
        if value is None:
            return bool(self.route_via_openrouter)
        return bool(value)

    @property
    def strict_timeout_minutes(self) -> float:
        """Hard deadline for the run, in minutes.

        Explicit ``max_duration_minutes`` wins; otherwise the historical CLI
        behaviour (research_time_max + 1) is preserved.
        """
        if self.max_duration_minutes is not None:
            return float(self.max_duration_minutes)
        return float(self.research_time_max_minutes) + 1.0

    @property
    def output_mode_none(self) -> bool:
        return self.output_mode == "none"

    def files_enabled(self) -> bool:
        return self.output_mode in ("file", "both")

    def sources_log_enabled(self) -> bool:
        return bool(self.enable_source_log) and self.log_mode in ("file", "both")

    def finalize(self, *, in_place: bool = False) -> "RunConfig":
        """Apply hard server safety caps. Returns a clamped copy by default."""
        def norm(value: Any, lo: Any, hi: Any) -> Any:
            return max(lo, min(hi, value))

        kw: dict[str, Any] = {}
        if self.max_duration_minutes is not None:
            kw["max_duration_minutes"] = _clamp(
                float(self.max_duration_minutes),
                HARD_MIN_DURATION_MINUTES, HARD_MAX_DURATION_MINUTES,
                "max_duration_minutes")
        if not (0.5 <= float(self.research_time_min_minutes) <= HARD_MAX_DURATION_MINUTES):
            kw["research_time_min_minutes"] = _clamp(
                float(self.research_time_min_minutes),
                0.5, HARD_MAX_DURATION_MINUTES, "research_time_min_minutes")
        if not (0.5 <= float(self.research_time_max_minutes) <= HARD_MAX_DURATION_MINUTES):
            kw["research_time_max_minutes"] = _clamp(
                float(self.research_time_max_minutes),
                0.5, HARD_MAX_DURATION_MINUTES, "research_time_max_minutes")
        kw["supervisor_max_iterations"] = norm(
            self.supervisor_max_iterations, 1, HARD_MAX_SUPERVISOR_ITERATIONS)
        kw["subagent_max_iterations"] = norm(
            self.subagent_max_iterations, 1, HARD_MAX_SUBAGENT_ITERATIONS)
        kw["subagent_max_reads"] = norm(self.subagent_max_reads, 1, HARD_MAX_SUBAGENT_READS)
        kw["subagent_max_total_reads"] = norm(
            self.subagent_max_total_reads, 1, HARD_MAX_SUBAGENT_TOTAL_READS)
        kw["subagent_max_saves"] = norm(self.subagent_max_saves, 1, HARD_MAX_SUBAGENT_SAVES)
        kw["subagent_max_searches"] = norm(self.subagent_max_searches, 1, HARD_MAX_SUBAGENT_SEARCHES)
        kw["subagent_max_concurrency"] = norm(
            self.subagent_max_concurrency, 1, HARD_MAX_SUBAGENT_CONCURRENCY)
        if self.output_mode not in OUTPUT_MODES:
            kw["output_mode"] = "file"

        # ── Agent filtering validation ──────────────────────────────────
        # Unknown names are dropped with a warning (lenient, like model
        # chains); a fully-empty selection is a config error.
        known_agents = set(_cfg.KNOWN_AGENT_NAMES)
        enabled = [a for a in (self.enabled_agents or []) if a in known_agents]
        dropped = [a for a in (self.enabled_agents or []) if a not in known_agents]
        if dropped:
            import logging
            logging.getLogger(__name__).warning(
                "Dropping unknown enabled_agents entries (not in "
                "config.KNOWN_AGENT_NAMES): %s", dropped)
        if not enabled:
            raise ValueError(
                "enabled_agents resolves to an empty selection. Provide at "
                f"least one of: {', '.join(_cfg.KNOWN_AGENT_NAMES)}")
        if enabled != list(self.enabled_agents or []):
            kw["enabled_agents"] = enabled

        known_platforms = set(_cfg.KNOWN_PLATFORM_KEYS)
        platforms = [p for p in (self.general_agent_platforms or [])
                     if p in known_platforms]
        dropped_plat = [p for p in (self.general_agent_platforms or [])
                        if p not in known_platforms]
        if dropped_plat:
            import logging
            logging.getLogger(__name__).warning(
                "Dropping unknown general_agent_platforms entries (not in "
                "config.KNOWN_PLATFORM_KEYS): %s", dropped_plat)
        if platforms != list(self.general_agent_platforms or []):
            kw["general_agent_platforms"] = platforms

        return replace(self, **kw)

    # ── Serialization ──────────────────────────────────────────────────
    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunConfig":
        allowed = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in allowed})

    @classmethod
    def from_env(cls, overrides: Optional[dict[str, Any]] = None) -> "RunConfig":
        """Build a RunConfig from environment/deployment defaults.

        ``overrides`` may set any RunConfig field on top of the env defaults.
        The result is NOT finalized (hard caps applied separately so callers
        can inspect the request before clamping).
        """
        cfg = cls()
        if overrides:
            return replace(cfg, **{
                k: v for k, v in overrides.items()
                if k in {f.name for f in fields(cls)}
            })
        return cfg
