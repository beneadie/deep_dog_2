#!/usr/bin/env python3
"""Repeatable trace smoke test for the Deep Dog engine.

Offline (default): runs the real research graph with fake models — no network,
no API keys, no cost. Use it to confirm the structured trace channel emits
identified records (UUID + seq) with full content.

Live (--live): runs a real research job against the provider credentials in
``.env``. This spends credits and can take up to the configured budget.

Trace records are printed to stdout **live as they are emitted** (full content:
prompts, thinking, tool calls, source rationale) and optionally appended to a
JSONL file incrementally, so you can watch the run in the terminal and tail the
file at the same time.

The size of each record's content is governed by the engine's log truncation
config: ``--truncation N`` caps every string at N chars; the default is **no
truncation** (full prompts/thinking).

Usage:
    python scripts/trace_smoke.py
    python scripts/trace_smoke.py --live --prompt "What changed in Python 3.14?"
    python scripts/trace_smoke.py --live --profile nemotron-subagents \
        --prompt "..." --max-minutes 10 --output-dir outputs \
        --trace-file outputs/live_trace.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Allow running as `python scripts/trace_smoke.py` from anywhere in the repo.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# Model profiles for test runs — edit here instead of setting env vars.
# Each profile maps to RunConfig model-chain fields (single-element = no fallback).
PROFILES: dict[str, dict] = {
    "default": {},  # engine defaults (supervisor deepseek-v4-pro, subagents deepseek-v4-flash)
    "nemotron-subagents": {
        "supervisor_model_fallback_chain": ["deepseek-v4-flash"],
        "subagent_model_fallback_chain": ["nvidia/nemotron-3.5-lightning"],
        "draft_report_model_fallback_chain": ["nvidia/nemotron-3.5-lightning"],
    },
    "all-deepseek": {
        "supervisor_model_fallback_chain": ["deepseek-v4-pro"],
        "subagent_model_fallback_chain": ["deepseek-v4-flash"],
        "draft_report_model_fallback_chain": ["deepseek-v4-flash"],
    },
}


class LiveTraceSink:
    """Async trace sink that prints every record immediately and appends JSONL.

    Content is printed in full; the engine already applied ``log_truncation``
    before the record reaches this sink.
    """

    def __init__(self, path: Path | None = None):
        self.count = 0
        self._fh = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = path.open("w", encoding="utf-8")

    async def emit(self, record) -> None:
        self.count += 1
        ts = datetime.fromtimestamp(record.timestamp).strftime("%H:%M:%S")
        who = record.agent or record.phase or ""
        # Make discovery vs research unmistakable for sub-agent records.
        mode = record.content.get("mode")
        tag = ""
        if mode and (record.kind.startswith("subagent") or record.kind == "delegation"):
            tag = f"  <<{str(mode).upper()}>>"
        print(f"\n[{ts}] {record.seq:03d} {record.kind}  {record.event_id}  {who}{tag}",
              flush=True)
        if record.content:
            print(json.dumps(record.content, indent=2, ensure_ascii=False, default=str),
                  flush=True)
        if self._fh is not None:
            self._fh.write(json.dumps(record.to_dict(), ensure_ascii=False, default=str) + "\n")
            self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--live", action="store_true",
                   help="run against real providers (spends credits); default is offline fakes")
    p.add_argument("--prompt", default="What changed in Python 3.14?")
    p.add_argument("--agents", default="ResearchWeb",
                   help="comma-separated agent names, e.g. ResearchWeb,ResearchArxiv")
    p.add_argument("--max-minutes", type=float, default=15.0,
                   help="research time budget in minutes (live mode)")
    p.add_argument("--output-dir", default=None,
                   help="when set, the engine writes its normal run artifacts here")
    p.add_argument("--trace-file", default=None,
                   help="append trace records as JSONL to this path (incremental, tail-able)")
    p.add_argument("--truncation", type=int, default=None,
                   help="max chars per trace string (default: no truncation = full prompts/thinking)")
    p.add_argument("--no-logging", action="store_true",
                   help="disable trace logging (should produce zero records)")
    p.add_argument("--console", action="store_true",
                   help="also enable engine console output (emoji logs)")
    p.add_argument("--profile", default="default", choices=sorted(PROFILES),
                   help="named model profile from PROFILES (default: %(default)s)")
    p.add_argument("--supervisor-model", default=None,
                   help="override supervisor model (comma-separated = fallback chain)")
    p.add_argument("--subagent-model", default=None,
                   help="override sub-agent model (comma-separated = fallback chain)")
    p.add_argument("--draft-model", default=None,
                   help="override draft/brief model (comma-separated = fallback chain)")
    return p.parse_args()


def _agents(raw: str) -> list:
    return [a.strip() for a in raw.split(",") if a.strip()]


def _chain(raw: str) -> list:
    return [m.strip() for m in raw.split(",") if m.strip()]


def _model_overrides(args: argparse.Namespace) -> dict:
    overrides = dict(PROFILES.get(args.profile, {}))
    if args.supervisor_model:
        overrides["supervisor_model_fallback_chain"] = _chain(args.supervisor_model)
    if args.subagent_model:
        overrides["subagent_model_fallback_chain"] = _chain(args.subagent_model)
    if args.draft_model:
        overrides["draft_report_model_fallback_chain"] = _chain(args.draft_model)
    return overrides


def _build_config(args: argparse.Namespace):
    agents = _agents(args.agents)
    overrides = _model_overrides(args)
    if args.live:
        from deep_research.integration import RunConfig
        return RunConfig(
            enabled_agents=agents,
            research_time_max_minutes=args.max_minutes,
            logging_enabled=not args.no_logging,
            log_truncation=args.truncation,
            **overrides,
        )
    # Offline: reuse the test-suite fakes so no provider is ever called.
    sys.path.insert(0, str(ROOT / "tests"))
    import deep_research.integration as integration
    from conftest import FakeModelFactory, api_config
    integration.ModelFactory = FakeModelFactory
    return api_config(
        enabled_agents=agents,
        logging_enabled=not args.no_logging,
        log_truncation=args.truncation,
        **overrides,
    )


async def _run(args: argparse.Namespace):
    if args.live:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")  # MUST precede the engine import

    from deep_research.integration import RuntimeOptions, run_research

    sink = LiveTraceSink(Path(args.trace_file) if args.trace_file else None)
    runtime = RuntimeOptions(
        run_id="trace-smoke",
        output_dir=Path(args.output_dir) if args.output_dir else None,
        trace_sink=sink,
        console_enabled=args.console,
    )
    result = await run_research(
        args.prompt,
        config=_build_config(args),
        runtime=runtime,
        credentials=None,  # falls back to .env / process environment
    )
    return result, sink


def main() -> int:
    args = _parse_args()
    mode = "LIVE (real providers)" if args.live else "OFFLINE (fake models)"
    models = _model_overrides(args)
    print(f"[trace_smoke] mode: {mode}")
    print(f"[trace_smoke] prompt: {args.prompt}")
    print(f"[trace_smoke] agents: {_agents(args.agents)}")
    print(f"[trace_smoke] profile: {args.profile}")
    print(f"[trace_smoke] truncation: {args.truncation or 'none (full content)'}")
    for role in ("supervisor", "subagent", "draft_report"):
        chain = models.get(f"{role}_model_fallback_chain")
        if chain:
            print(f"[trace_smoke] {role:<12} model: {chain}")
    print("[trace_smoke] streaming trace below...\n", flush=True)

    result, sink = asyncio.run(_run(args))
    sink.close()

    print(f"\n[trace_smoke] status: {result.status}")
    print(f"[trace_smoke] report chars: {len(result.final_report)}"
          f" | brief: {bool(result.research_brief)}"
          f" | records: {sink.count}")

    if args.trace_file:
        print(f"[trace_smoke] trace -> {args.trace_file}")

    if result.status not in ("completed", "partial"):
        print(f"[trace_smoke] failure: {result.failure}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
