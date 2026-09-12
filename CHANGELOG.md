# Changelog

## Unreleased

- Restore the original engine's research timing: the supervisor routes to
  final writing after the research window plus one minute, and writing and
  citation processing can finish beyond that window. Remove the package's
  whole-run cancellation timer, `RunConfig.max_duration_minutes`, and
  `RuntimeOptions.deadline_override_seconds`. Configure the research window
  with `research_time_min_minutes` / `research_time_max_minutes` instead.
  Explicit host cancellation and existing individual request/subagent timeouts
  remain available.

## 2.0.1 — product release (concurrency-safe engine)

### Agent selection (breaking default change)
- `ENABLED_AGENTS` is now configurable via env var (comma-separated tool-schema
  names, e.g. `ENABLED_AGENTS=ResearchWeb,ResearchReddit`). **Default when
  unset: the WEB agent only** (previously the source array selected
  Web/Reddit/Substack — deployments relying on that must set the env var or
  pass `RunConfig(enabled_agents=[...])`). Per-request selection is supported
  via `RunConfig.enabled_agents` (order = tool-schema binding order).
- `GENERAL_AGENT_PLATFORMS` is now env-configurable the same way (unset = all
  platforms). Per-run via `RunConfig.general_agent_platforms`.
- `RunConfig.finalize()` validates both lists: unknown names are dropped with
  a warning; an empty agent selection raises `ValueError`.
- `config.KNOWN_AGENT_NAMES` / `config.KNOWN_PLATFORM_KEYS` are the single
  source of truth for validation.

### Credentials
- All platform tools now resolve provider keys run-credentials-first
  (`deep_research.secrets.get_secret`), falling back to the process
  environment. Previously reddit (4 vars), pubmed (email + tavily fallback),
  and sec-edgar (contact email / User-Agent) read `os.getenv` directly, so
  per-user keys for those platforms were ignored; the SEC User-Agent was also
  frozen at import time and is now resolved per call.
- New pre-flight check: `validate_credentials(config, credentials) ->
  CredentialCheck` reports `missing_required` / `optional_missing` env-var
  names (never values). `run_research()` calls it once per run and surfaces
  the result as a `config_validated` event plus
  `ResearchResult.run_metadata["credential_check"]`. The engine stays lenient
  (degradation semantics unchanged); hosts may reject bad requests early.
- `config.model_required_keys(model_name, route)` exposes the model→key
  mapping used by chain filtering and validation.

### Structured trace logging (new)
- New content-rich trace channel (`deep_research/trace.py`), independent of the
  content-free product events. Every record carries a full UUID4 `event_id` plus
  a monotonic `seq`, `run_id`, `phase`, `agent`, `platform`, `iteration` and an
  optional `parent_id`, so production logs can be correlated and streamed.
- Captures: run prompt, research brief, draft report, full sub-agent system
  prompts + message history, sub-agent model thinking (`reasoning_content`),
  tool calls with args, source-save rationale, supervisor turns/thinking,
  delegation and sub-agent findings, and the final report.
- Toggle + truncation are per-run config:
  `RunConfig.logging_enabled` (env `LOGGING_ENABLED`, default **on**) and
  `RunConfig.log_truncation` (env `LOG_TRUNCATION`, default **None = no
  truncation**).
- Streaming via `RuntimeOptions.trace_sink` (sync/async); records are also
  retained in memory for the run and returned as `ResearchResult.logs`.
  `TraceCollector` is provided for tests/simple hosts.
- Secret safety: any host-supplied credential value (length ≥ 6) is scrubbed to
  `[REDACTED]` from every record before it is stored or streamed.
- Trace field naming: assistant text is `response` and model chain-of-thought is
  `thinking` on both `supervisor_turn` and `subagent_response` (previously the
  sub-agent used `content`/`reasoning`, colliding with the record's own
  `content` payload). Sub-agent response text now also normalizes block-style
  provider content via `extract_text_from_response` instead of dropping it.
- Liveness fix: a background flusher drains events/trace every ~200 ms for the
  duration of a run, so records stream **while** nested nodes block the
  top-level graph driver (the supervisor subgraph and awaited sub-agents).
  Previously records emitted inside those nodes were buffered until the
  subgraph returned — appearing only at the end, or lost entirely on a crash.
  Flushes are serialized (`Observer._flush_lock`) so the driver and the flusher
  cannot interleave.

### Package version / compatibility
- Distribution: `deep-dog-2==2.0.1` (import package remains `deep_research`).
- Compatible API adapter: `research_agent_api` adapter version `>=1.0.0` must
  call only `deep_research.integration.run_research` and its typed inputs.
- Python `>=3.11`. LangChain/LangGraph versions unchanged from `requirements.txt`.

### What changed (engine, no rewrite)
- The research graph, agents, supervisor, citation system, and file layout are
  unchanged in structure. Prompt *text* was not rewritten; only the
  supervisor timing literals were converted to `{min_research_time_minutes}` /
  `{max_research_time_minutes}` placeholders so each run's time budget reaches
  the prompt (runtime correctness).
- Model clients are no longer built at module import time. Each run builds its
  own supervisor / draft-report / sub-agent clients from its own config +
  credentials (`deep_research.models.ModelFactory`, `RunConfig`,
  `Credentials`). Importing any module no longer requires API keys.
- Observability moved from process globals to a per-run `Observer`
  (`deep_research.observer`); the legacy `observability.py` functions remain as
  backward-compatible shims.
- Console logger is runtime-gated; API runs suppress console chatter and the
  host receives product-level `ResearchEvent`s instead.

### New public surface (stable, typed)
- `deep_research.integration.run_research(prompt, config, runtime, credentials)`
  returns `ResearchResult` (status: completed / failed / cancelled / timed_out /
  partial). No FastAPI schemas are imported by the engine.
- `RunConfig` (per-run: model chains, routing, timing, iteration/depth caps,
  enabled agents, search engine, prompt version, output mode, language,
  subtopics, trace) with env-defaults -> request overrides -> hard safety caps.
- `RuntimeOptions` (event sink, artifact sink, cancellation, external LangGraph
  checkpointer, run/thread id, output dir).
- Structured events: run/scope/draft/supervisor/subagent/source/report/citation
  lifecycle events — never chain-of-thought or secrets.
- Cancellation is cooperative (token or `() -> bool`), checked between steps.
- Research time limits guide the supervisor to final writing; they do not
  cancel the run (restored in the unreleased fix above).

### Migration / checkpoint compatibility
- `config_snapshot` is stored in checkpointed graph state (no clients, no
  secrets) so a resumed run can rebuild its runtime configuration.
- In-memory checkpointer remains the CLI default; the host injects a durable
  checkpointer per run. Thread id is host-supplied and stable across resume.
- No existing environment variables changed meaning; module constants in
  `config.py` remain the environment/deployment defaults for the CLI.

### Quality / cost / latency notes
- No prompt or default-model changes: quality is unchanged for identical
  configs. Per-run model/time isolation means an API run can exceed the old
  15-minute CLI window only when the host requests it (hard ceiling 240 min).

### Developer / public-private workflow
- `origin` = private product repo; `upstream` = public Deep Dog 2 repo.
- Improvements suitable for the public repo should be upstreamed; product-only
  changes (this integration layer, events, packaging, product versioning)
  remain private.
- Sync procedure: fetch `upstream`, review, deliberately merge/cherry-pick, run
  `python -m pytest tests/`, then tag a new private release here.
- Release tags are pinned by the API; do not install from a moving `main`.
