# Deep Dog 2 — Deep Research Agent

A multi-agent deep research system built with [LangGraph](https://github.com/langchain-ai/langgraph) and [LangChain](https://github.com/langchain-ai/langchain). A supervisor agent decomposes a research question into sub-tasks and delegates to specialized platform sub-agents (Web, Reddit, Substack) that search, read, and synthesize sources into a cited final report.

This project builds on Deep Dog 1, which was built upon [ThinkDepth Deep Research](https://github.com/thinkdepthai/Deep_Research) by Paichun Lin. See [LICENSE](LICENSE) for the applicable attribution and license notices.

Pipeline: `clarify_with_user` → `write_research_brief` → `write_draft_report` → `supervisor_subgraph` (parallel sub-agents) → `final_report_generation` — see `deep_research/research_agent_full.py:235` and `deep_research/research_agent_scope.py`.

## About the Project

Deep Dog 2 extends the original Deep Dog draft-first research architecture with a more explicit separation of model roles, platform-specific research agents, shared source registries, and citation validation. It is designed for people who want to trade off research quality, latency, model cost, and provider reliability without changing the graph itself.

The main design goals are:

- **Role-specific models** — use different models for the supervisor, initial draft, and platform sub-agents.
- **Reflection and delegation** — the supervisor plans research, delegates distinct questions, evaluates findings, and decides whether more work is needed.
- **Platform specialization** — Web, Reddit, and Substack agents are enabled by default, with General, PubMed, Arxiv, and SEC agents available for extension.
- **Evidence-first reports** — findings are collected into a source registry, then final inline citations and the `## Sources` section are validated before the report is returned.
- **Operational control** — time limits, iteration caps, search budgets, fallback chains, and output modes can be tuned for local experiments or more economical deployments.

## Features

- **Supervisor + sub-agent architecture** — configurable platform agents at `deep_research/config.py:449` (`ResearchWeb`, `ResearchReddit`, `ResearchSubstack`; `ResearchGeneral`, `ResearchPubMed`, `ResearchArxiv`, `ResearchSEC` available but disabled by default)
- **Multiple search providers** — Tavily and/or Exa (`WEB_SEARCH_ENGINE` at `deep_research/config.py:658`)
- **Provider-agnostic models** — DeepSeek, MiMo, Meta Muse, Gemini, OpenAI, GLM, and OpenRouter-hosted models via a single `get_model()` factory (`deep_research/config.py:883`)
- **Model fallback chains** — per-role fallback lists (`SUBAGENT_MODEL_FALLBACK_CHAIN`, `SUPERVISOR_MODEL_FALLBACK_CHAIN`, `DRAFT_REPORT_MODEL_FALLBACK_CHAIN`)
- **Cited reports** — markdown report + `research_data_*.json` sources file + optional research trace
- **LangGraph execution** — recursion limit, timeouts, and observability logging

## How It Works

Deep Dog 2 retains the draft-first, iterative refinement idea from Deep Dog 1, while making reflection, delegation, and source handling explicit:

```text
User question
     |
     v
clarify_with_user → write_research_brief → write_draft_report
                                               |
                                               v
                                    supervisor research loop
                              ┌───────────────┼────────────────┐
                              │               │                │
                           reflect        delegate       refine draft
                         (think_tool)   (parallel agents)  (when useful)
                                              |
                                              v
                               Web / Reddit / Substack / ...
                                              |
                                              v
                                     findings + sources
                                              |
                                              └── repeat until complete
                                                        |
                                                        v
                              final report → citation validation → output
```

1. **Scope the question.** The input is converted into a structured research brief.
2. **Create a scaffold.** An initial draft establishes a useful report structure before live research begins. It is not treated as evidence.
3. **Reflect and delegate.** The supervisor uses internal reflection to identify gaps and delegates focused, non-overlapping research tasks to platform agents.
4. **Research in parallel.** Agents search, read, save, and compress findings using the tools available for their platform. Their results are returned with source metadata and citations.
5. **Evaluate and refine.** The supervisor reviews the findings, may request another round, and can refine the draft as the evidence improves.
6. **Finalize the report.** The final writer combines the brief, draft, and findings. Citation checks validate the relationship between inline citations and the final sources list.

Supervisor reflection is used for planning and control; it is not copied into the final research report. Optional subtopic evaluation and parallel subtopic reports can run after the main report when enabled in `deep_research/config.py`.

The default prompt family is `OPEN`. `LEGACY` is retained as a compatibility option for the older iterative draft-refinement behavior; select it with `PROMPT_VERSION` when comparing prompt strategies.

## Official Benchmark Results

At the time of publication, Deep Dog 2 ranked **5th overall** and **1st among open-source research agents** on the [DeepResearch Bench](https://huggingface.co/spaces/muset-ai/DeepResearch-Bench-Leaderboard) benchmark. These results were achieved with a relatively economical configuration: a maximum of 3 Exa searches per sub-agent, 15 minutes of research time, 20 total iterations, and DeepSeek V4 Pro and DeepSeek V4 Flash as the supervisor and sub-agent base models.

| Metric | Score |
|---|---:|
| Overall | **0.5432** |
| Comprehensiveness | 0.5468 |
| Insight | 0.5532 |
| Instruction Following | 0.5426 |
| Readability | 0.5105 |

For a detailed explanation of the reflection and delegation methods used, see the engineering article [Deep Dog 2: How Reflection and Structured Delegation Improve Supervisor–Subagent Research Systems](https://beneadie01.substack.com/p/deep-dog-2-how-reflection-delegation).

The benchmark configuration is a reproducibility profile rather than a promise about the current defaults. In particular, the benchmark used a 20-iteration supervisor cap, while the current code default is 30; see [Configuration](#configuration) when reproducing the result.

## Requirements

- **Python** 3.10+ (tested with 3.14.4)
- **Package manager:** `pip` or [`uv`](https://docs.astral.sh/uv/)
- **API keys:** at least one LLM provider + one search provider (see Environment)

## Installation

### 1. Clone

```bash
git clone https://github.com/anomalyco/Deep-Dog-2.git
cd "Deep Dog 2"
```

### 2. Install dependencies

**With uv (recommended):**

```bash
uv sync
# or directly:
uv pip install -r requirements.txt
```

**With pip + venv:**

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> **Note on `requirements.txt` encoding:** the file is stored as UTF-16 LE with BOM. If `pip install -r requirements.txt` fails with a decoding error, convert first:
>
> ```bash
> iconv -f UTF-16 -t UTF-8 requirements.txt | pip install -r /dev/stdin
> ```

## Environment Setup

Create a `.env` file in the project root (next to `run_research.py`). `run_research.py:75` loads it via `python-dotenv`. There is no committed `.env.example`; create the file manually and keep secrets out of version control.

### Required variables

| Variable | Description |
|---|---|
| `DEEPSEEK_API_KEY` or `DEEPSEEK_KEY` | DeepSeek native API (`api.deepseek.com`) |
| `OPENROUTER_API_KEY` | OpenRouter — required if any model routes via OpenRouter (e.g. `nvidia/nemotron-3.5-lightning`, `deepseek-baba-singapore`) |
| `TAVILY_API_KEY` | Tavily search — required if `WEB_SEARCH_ENGINE=tavily` or `both` |
| `EXA_API_KEY` | Exa search — required if `WEB_SEARCH_ENGINE=exa` (default) or `both` |

You only need keys for the providers you actually use. The model factories at `deep_research/config.py:682` skip chain entries whose key is missing and fall back to the next model.

### Optional / commonly used

| Variable | Description |
|---|---|
| `GEMINI_API_KEY` / `GOOGLE_API_KEY` | Google Gemini |
| `OPENAI_API_KEY` | OpenAI |
| `MIMO_API_KEY` | MiMo native API |
| `META_API_KEY` | Meta Muse native API |
| `ZHIPUAI_API_KEY` | Z.AI GLM |
| `PUBMED_EMAIL` | PubMed E-utilities contact |
| `SEC_EDGAR_CONTACT_EMAIL` | SEC EDGAR contact |
| `PERPLEXITY_KEY` | Substack/Perplexity search |

### Example `.env`

```ini
# LLM
DEEPSEEK_API_KEY=sk-...
OPENROUTER_API_KEY=sk-or-...

# Search (at least one)
EXA_API_KEY=...
TAVILY_API_KEY=...

# Optional overrides
# SUPERVISOR_MODEL=deepseek-v4-pro
# SUBAGENT_MODEL=nvidia/nemotron-3.5-lightning
# WEB_SEARCH_ENGINE=exa
# DISABLE_MODEL_FALLBACK=false  # enable configured fallback chains
```

## Usage

All commands assume you are in the project root (where `run_research.py` lives).

### Quick start

```bash
# with uv
uv run python run_research.py --prompt "What are the latest developments in AI safety?"

# without uv (venv activated)
python run_research.py --prompt "What are the latest developments in AI safety?"
```

### From a prompt file

```bash
uv run python run_research.py --prompt-file input.txt
uv run python run_research.py --prompt-file input.txt --output-dir my_outputs
```

### CLI reference

```
python run_research.py --help
```

| Flag | Description | Default |
|---|---|---|
| `--prompt`, `-p` | Research question as a string (mutually exclusive with `--prompt-file`) | — |
| `--prompt-file`, `-f` | Path to a file containing the research prompt | — |
| `--output-dir`, `-o` | Directory for output files | `outputs` |
| `--thread-id`, `-t` | Thread ID for the session | auto-generated `YYYY-MM-DD_HH-MM-SS_xxx` |

Defined at `run_research.py:615`.

### Programmatic API

```python
import asyncio
from pathlib import Path
from run_research import run_research

result = asyncio.run(run_research(
    prompt="What are the latest developments in quantum error correction?",
    output_dir=Path("outputs"),
))

print(result["output_file"])   # Path to research_*.md
print(result["final_report"])  # str
print(result["sources"])       # list[dict]
print(result["trace_content"]) # str | None
```

Return dict documented at `run_research.py:167`.

### Standalone platform runner

Run a single platform sub-agent without the supervisor — useful for testing/benchmarking. See `deep_research/run_platform.py:1`.

```bash
python -m deep_research.run_platform --agent reddit --topic "NVIDIA earnings Q4" --output-mode report
python -m deep_research.run_platform --agent pubmed --topic "GLP-1 efficacy" --output-mode sources

# Options
# --agent: reddit | web | substack | pubmed | arxiv | sec | general
# --output-mode: sources | report | sources_inline | report_inline
# --provider / -m: model iterations, reads, saves, etc.
```

## Output

Each run creates timestamped files in `--output-dir` (default `outputs/`, gitignored at `.gitignore:14`):

```
outputs/
  research_2026-09-01_10-30-00_abc.md      # final report (markdown)
  research_data_2026-09-01_10-30-00_abc.json  # structured sources + metadata
  trace_2026-09-01_10-30-00_abc.md         # supervisor ↔ sub-agent trace (if ENABLE_RESEARCH_TRACE)
  error_2026-09-01_10-30-00_abc.txt        # error dump on failure
```

Report structure (`run_research.py:333`): header → Research Prompt → Research Brief → Final Report (with `## Sources` citations). Sources JSON contains `thread_id`, `timestamp`, `prompt`, `sources`, `final_report`.

The research trace is disabled by default. To generate the supervisor ↔ sub-agent process trace, set `ENABLE_RESEARCH_TRACE = True` in `deep_research/config.py`. To save full-length sub-agent deliverables in addition to the source logs, set `SAVE_SUBAGENT_REPORTS_TO_FILE=true` in `.env`.

## Configuration

The complete configuration lives in `deep_research/config.py`. Start with the settings below; most users do not need to change anything else. Model and provider choices are normally set in `.env`, while several structural limits remain direct constants in `config.py`.

### Model roles and provider routing

| Setting | Default | What it controls |
|---|---|---|
| `SUPERVISOR_MODEL` | `deepseek-v4-pro` | Planning, reflection, draft refinement, and final report writing |
| `SUBAGENT_MODEL` | `deepseek-v4-flash` | Web, Reddit, Substack, and other platform agents |
| `DRAFT_REPORT_MODEL` | `nvidia/nemotron-3.5-lightning` | Research brief and initial draft; this default requires `OPENROUTER_API_KEY` |
| `*_MODEL_FALLBACK_CHAIN` | Role-dependent | Comma-separated fallback models for each role |
| `DISABLE_MODEL_FALLBACK` | `true` in the current code | Set to `false` to allow the configured fallback chains to activate after a model failure |
| `ROUTE_VIA_OPENROUTER` | `false` | Routes supported DeepSeek, MiMo, and Meta models through OpenRouter; role-specific routing flags can override it |
| `DRAFT_REPORT_REASONING_EFFORT` | empty | `low`, `medium`, or `high` when the draft role is routed through OpenRouter |

See the model catalog at the top of `deep_research/config.py` for supported DeepSeek, MiMo, Muse, Gemini, OpenAI, Nemotron, GLM, and OpenRouter names.

### Research budget

| Setting | Default | Where to change it | What it controls |
|---|---|---|---|
| `RESEARCH_TIME_MIN_MINUTES` | `5` | `config.py` | Minimum research window before completion is accepted |
| `RESEARCH_TIME_MAX_MINUTES` | `15` | `config.py` | Target maximum research window; the hard stop is derived from it |
| `SUPERVISOR_MAX_ITERATIONS` | `30` | `config.py` | Maximum supervisor research-loop iterations |
| `SUBAGENT_MAX_ITERATIONS` | `5` | `.env` | Tool-call rounds for each delegated sub-agent |
| `SUBAGENT_MAX_SEARCHES` | `3` | `.env` | Search-call budget for each sub-agent run; a useful first lever for cost control |
| `SUBAGENT_MAX_READS`, `SUBAGENT_MAX_SAVES`, `SUBAGENT_MAX_CONCURRENCY` | `10`, `10`, `3` | `.env` | Per-agent depth and parallel tool-call caps |
| `DEFAULT_MAX_TOTAL_READS` | `25` | `config.py` | Total number of items a sub-agent may read across its run |

For the published benchmark profile, use a 15-minute research window, a 20-iteration supervisor cap, and at most 3 Exa searches per sub-agent. The benchmark cap therefore differs from the current code default for `SUPERVISOR_MAX_ITERATIONS`.

### Agent, search, and output behavior

| Setting | Default | What it controls |
|---|---|---|
| `ENABLED_AGENTS` | Web, Reddit, Substack | Platform agents the supervisor may call; edit the list in `config.py` to add General, PubMed, Arxiv, or SEC |
| `WEB_SEARCH_ENGINE` | `exa` | Search backend: `exa`, `tavily`, or `both`; set the matching API key in `.env` |
| `PROMPT_VERSION` | `OPEN` | Prompt family: `OPEN` or the retained `LEGACY` mode |
| `SUBAGENT_OUTPUT_MODE` | `sources` | Whether research agents return curated sources or mini-reports; advanced setting |
| `DISCOVERY_OUTPUT_MODE` | `report_inline` | Deliverable format for discovery-mode agents; advanced setting |
| `OUTPUT_MODE` | `file` | Supported production path is timestamped Markdown/JSON files; database persistence is currently an extension point |
| `SAVE_REPORT_TO_FILE` | `true` | Write final and optional subtopic reports to disk |
| `ENABLE_SOURCE_LOG` | `true` | Write collected sources to JSONL/JSON metadata files |

When tuning for a new provider or deployment, change model roles and `WEB_SEARCH_ENGINE` first. Only then adjust time, iteration, search, and output-mode settings; changing the entire configuration at once makes benchmark and quality comparisons difficult.

## Project Structure

```
Deep Dog 2/
├── run_research.py              # Main entry point (CLI + programmatic API)
├── requirements.txt             # Pinned dependencies (UTF-16 LE)
├── deep_research/
│   ├── config.py                # All configuration & model factory (get_model)
│   ├── research_agent_full.py   # Full LangGraph workflow (deep_researcher_builder)
│   ├── research_agent_scope.py  # Scoping: clarify → brief → draft
│   ├── multi_agent_supervisor.py # Supervisor agent & delegation logic
│   ├── run_platform.py          # Standalone single-platform runner
│   ├── agents/                  # Platform agents (web, reddit, substack, ...)
│   ├── prompts*.py              # Prompt bundles (open, legacy)
│   ├── state_*.py               # Graph state schemas
│   ├── observability.py         # Run folder & source aggregation
│   ├── utils.py                 # Helpers (extract_text, date, etc.)
│   └── citation_utils.py        # Citation formatting
└── outputs/                     # Generated reports (gitignored)
```

## Extending Deep Dog 2

Platform agents use a shared research engine. To add a new platform:

1. Create `deep_research/agents/<platform>/tools.py` with its LangChain tools.
2. Add the platform to `PLATFORMS` in `deep_research/agents/base.py`.
3. Add an adapter entry to `AGENT_REGISTRY` in `deep_research/agents/__init__.py`.
4. Add the registry key to `ENABLED_AGENTS` in `deep_research/config.py` when it should be available to the supervisor.

The supervisor spawn loop reads the registry at runtime, so adding a platform does not require a new branch in the orchestration logic. The existing General, PubMed, Arxiv, and SEC agents provide examples of optional platform integrations.

## Limitations and Current Status

- Live research depends on the availability, limits, and pricing of the configured LLM and search providers.
- More research time, iterations, reads, and model reasoning generally improve coverage but increase latency and cost.
- The default enabled platform agents are Web, Reddit, and Substack; domain-specific agents must be enabled explicitly.
- File output is the supported path. `OUTPUT_MODE=db` and `OUTPUT_MODE=both` remain extension points in the current runner rather than a complete database persistence implementation.
- Benchmark rankings and scores are snapshots from the time of publication and may change as the benchmark leaderboard changes.

## Acknowledgements

Deep Dog 2 builds on [ThinkDepth Deep Research](https://github.com/thinkdepthai/Deep_Research) by Paichun Lin. See [LICENSE](LICENSE) for the applicable attribution and license notices.

## Troubleshooting

- **`ValueError: OPENROUTER_API_KEY not found`** — the active model chain requires that key. Set it in `.env` or switch the chain to a provider you have a key for (e.g. `SUBAGENT_MODEL=deepseek-v4-flash` with `DEEPSEEK_API_KEY`).
- **`pip install -r requirements.txt` fails** — file is UTF-16 LE; use the `iconv` workaround above or re-save as UTF-8.
- **`ModuleNotFoundError: deep_research`** — run from the project root, not from inside `deep_research/`. `run_research.py:61` expects to be at the root.
- **Research hangs** — check `SUBAGENT_TIMEOUT_SECONDS` (600s) and `SUPERVISOR_TIMEOUT_SECONDS` (420s) at `config.py:311`; a stalled provider will be bounded. Inspect `outputs/error_*.txt`.
- **Chinese content filter rejections** — if using native DeepSeek/MiMo with sensitive topics, set `CHINESE_MODERATION=true` (`config.py:523`).

## License

MIT — see [LICENSE](LICENSE).
