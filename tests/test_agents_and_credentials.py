"""Agent selection config, credential resolution, and pre-flight validation.

Covers:
- ENABLED_AGENTS / GENERAL_AGENT_PLATFORMS env parsing (subprocess: import time)
- RunConfig.finalize() lenient validation (unknown names dropped, empty raises)
- validate_credentials() across representative config shapes
- Per-run credential resolution for reddit / sec_edgar / pubmed (get_secret)
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from deep_research.integration import CredentialCheck, validate_credentials
from deep_research.models import Credentials
from deep_research.run_config import RunConfig

ROOT = Path(__file__).resolve().parents[1]


# ── Env-var parsing (import-time behavior) ──────────────────────────────

def _run_config_snippet(env_extra: dict) -> str:
    env = {
        k: v for k, v in os.environ.items()
        if k not in ("ENABLED_AGENTS", "GENERAL_AGENT_PLATFORMS")
    }
    env.update(env_extra)
    code = (
        "from deep_research import config as c; "
        "print(repr(c.ENABLED_AGENTS)); print(repr(c.GENERAL_AGENT_PLATFORMS))"
    )
    r = subprocess.run([sys.executable, "-c", code], env=env, cwd=ROOT,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    lines = r.stdout.strip().splitlines()
    return eval(lines[0]), eval(lines[1])


def test_enabled_agents_default_is_web_only():
    agents, platforms = _run_config_snippet({})
    assert agents == ["ResearchWeb"]
    assert platforms == []


def test_enabled_agents_env_var_parsed():
    agents, platforms = _run_config_snippet({
        "ENABLED_AGENTS": "ResearchWeb , ResearchPubMed,ResearchSEC",
        "GENERAL_AGENT_PLATFORMS": "web, reddit",
    })
    assert agents == ["ResearchWeb", "ResearchPubMed", "ResearchSEC"]
    assert platforms == ["web", "reddit"]


def test_enabled_agents_empty_env_falls_back_to_default():
    agents, _ = _run_config_snippet({"ENABLED_AGENTS": "  , "})
    assert agents == ["ResearchWeb"]


# ── RunConfig.finalize() validation ─────────────────────────────────────

def test_finalize_drops_unknown_agents_preserving_order():
    cfg = RunConfig(
        enabled_agents=["ResearchReddit", "Bogus", "ResearchWeb"],
    ).finalize()
    assert cfg.enabled_agents == ["ResearchReddit", "ResearchWeb"]


def test_finalize_rejects_empty_agent_selection():
    with pytest.raises(ValueError, match="empty selection"):
        RunConfig(enabled_agents=["Nope"]).finalize()


def test_finalize_drops_unknown_general_platforms():
    cfg = RunConfig(general_agent_platforms=["web", "bogus", "pubmed"]).finalize()
    assert cfg.general_agent_platforms == ["web", "pubmed"]


def test_finalize_known_platform_keys_exist():
    from deep_research import config as _cfg
    for name in ("ResearchWeb", "ResearchGeneral", "ResearchReddit",
                 "ResearchSubstack", "ResearchPubMed", "ResearchArxiv",
                 "ResearchSEC"):
        assert name in _cfg.KNOWN_AGENT_NAMES


# ── validate_credentials ────────────────────────────────────────────────

_DS = Credentials({"DEEPSEEK_API_KEY": "x", "EXA_API_KEY": "y"})


def test_validate_missing_everything_reports_required_keys():
    check = validate_credentials()  # default config, empty credentials
    assert not check.ok
    assert "EXA_API_KEY" in check.missing_required
    # deepseek accepts either spelling; at least one must be reported
    assert "DEEPSEEK_API_KEY" in check.missing_required or \
        "DEEPSEEK_KEY" in check.missing_required
    assert "PERPLEXITY_API_KEY" in check.optional_missing


def test_validate_satisfied_config_is_ok():
    check = validate_credentials(credentials=_DS)
    assert check.ok
    assert check.missing_required == []
    assert check.optional_missing == ["PERPLEXITY_API_KEY"]


def test_validate_chain_degrades_without_false_positive():
    # First draft-chain entry (nemotron) needs OPENROUTER_API_KEY, but the
    # deepseek fallback is usable — the chain must NOT be reported missing.
    cfg = RunConfig(
        draft_report_model_fallback_chain=[
            "nvidia/nemotron-3.5-lightning", "deepseek-v4-flash"],
        web_search_engine="exa",
    )
    check = validate_credentials(config=cfg, credentials=_DS)
    assert check.ok
    assert "OPENROUTER_API_KEY" not in check.missing_required


def test_validate_dead_chain_reports_primary_keys():
    cfg = RunConfig(supervisor_model_fallback_chain=["muse-spark-1.2"],
                    subagent_model_fallback_chain=["muse-spark-1.2"],
                    draft_report_model_fallback_chain=["muse-spark-1.2"],
                    web_search_engine="exa")
    check = validate_credentials(config=cfg, credentials=_DS)
    assert not check.ok
    assert "META_API_KEY" in check.missing_required


def test_validate_reddit_enabled_requires_reddit_creds():
    cfg = RunConfig(enabled_agents=["ResearchWeb", "ResearchReddit"])
    check = validate_credentials(config=cfg, credentials=_DS)
    for key in ("REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET",
                "REDDIT_USERNAME", "REDDIT_PASSWORD"):
        assert key in check.missing_required
    # With all four supplied the run validates clean.
    full = Credentials({
        "DEEPSEEK_API_KEY": "x", "EXA_API_KEY": "y",
        "REDDIT_CLIENT_ID": "a", "REDDIT_CLIENT_SECRET": "b",
        "REDDIT_USERNAME": "c", "REDDIT_PASSWORD": "d",
    })
    assert validate_credentials(config=cfg, credentials=full).ok


def test_validate_openrouter_routing_reports_openrouter_key():
    cfg = RunConfig(route_via_openrouter=True,
                    supervisor_model_fallback_chain=["muse-spark-1.2"],
                    subagent_model_fallback_chain=["deepseek-v4-flash"],
                    draft_report_model_fallback_chain=["deepseek-v4-flash"],
                    web_search_engine="exa")
    check = validate_credentials(config=cfg, credentials=_DS)
    assert not check.ok
    assert "OPENROUTER_API_KEY" in check.missing_required


def test_validate_engine_tavily_requires_tavily_key():
    cfg = RunConfig(web_search_engine="tavily")
    check = validate_credentials(config=cfg, credentials=_DS)
    assert "TAVILY_API_KEY" in check.missing_required
    assert "EXA_API_KEY" not in check.missing_required


def test_validate_pubmed_email_is_optional():
    cfg = RunConfig(enabled_agents=["ResearchWeb", "ResearchPubMed"])
    check = validate_credentials(config=cfg, credentials=_DS)
    assert check.ok
    assert "PUBMED_EMAIL" in check.optional_missing


def test_credential_check_helper():
    check = CredentialCheck(missing_required=["A"], optional_missing=["B"])
    assert check.ok is False
    assert check.to_dict() == {"missing_required": ["A"], "optional_missing": ["B"]}
    assert CredentialCheck().ok is True


# ── Per-run credential resolution (get_secret) ──────────────────────────

def test_get_secret_prefers_runtime_credentials_over_env(monkeypatch):
    from deep_research.observer import Observer
    from deep_research.runtime import RuntimeContext, runtime_scope
    from deep_research.secrets import get_secret

    monkeypatch.setenv("REDDIT_CLIENT_ID", "env-value")
    cfg = RunConfig(enabled_agents=["ResearchWeb"]).finalize()
    ctx = RuntimeContext(
        run_id="r", thread_id="t", config=cfg,
        credentials=Credentials({"REDDIT_CLIENT_ID": "run-value"}),
    )
    with runtime_scope(ctx):
        assert get_secret("REDDIT_CLIENT_ID") == "run-value"
    # Outside the run scope the env fallback applies again.
    assert get_secret("REDDIT_CLIENT_ID") == "env-value"


def test_reddit_token_uses_runtime_credentials(monkeypatch):
    from deep_research.agents.reddit import tools as reddit_tools
    from deep_research.observer import Observer
    from deep_research.runtime import RuntimeContext, runtime_scope

    cfg = RunConfig(enabled_agents=["ResearchWeb"]).finalize()
    creds = Credentials({
        "REDDIT_CLIENT_ID": "cid", "REDDIT_CLIENT_SECRET": "csec",
        "REDDIT_USERNAME": "u", "REDDIT_PASSWORD": "p",
    })

    captured = {}

    class _Resp:
        status_code = 200
        def json(self):
            return {"access_token": "tok", "expires_in": 3600}
        def raise_for_status(self):
            return None

    def _fake_post(url, auth=None, data=None, headers=None, timeout=None):
        captured["auth"] = auth
        captured["data"] = data
        return _Resp()

    monkeypatch.setattr(reddit_tools.requests, "post", _fake_post)
    reddit_tools._reddit_token_cache.update({"token": None, "expires_at": 0})

    ctx = RuntimeContext(run_id="r", thread_id="t", config=cfg, credentials=creds)
    with runtime_scope(ctx):
        token = reddit_tools._get_reddit_token()

    assert token == "tok"
    from requests.auth import HTTPBasicAuth
    assert isinstance(captured["auth"], HTTPBasicAuth)
    assert captured["data"]["username"] == "u"
    assert captured["data"]["password"] == "p"


def test_sec_user_agent_resolves_runtime_credentials(monkeypatch):
    from deep_research.agents.sec_edgar import tools as sec_tools
    from deep_research.observer import Observer
    from deep_research.runtime import RuntimeContext, runtime_scope

    cfg = RunConfig(enabled_agents=["ResearchWeb"]).finalize()
    ctx = RuntimeContext(
        run_id="r", thread_id="t", config=cfg,
        credentials=Credentials({"SEC_EDGAR_CONTACT_EMAIL": "analyst@firm.com"}),
    )
    with runtime_scope(ctx):
        ua = sec_tools._sec_user_agent()
    assert "analyst@firm.com" in ua
    assert ua.startswith("SECEdgarAgent/1.0")
    # Headers helper builds from the same value.
    with runtime_scope(ctx):
        headers = sec_tools._sec_headers()
    assert headers["User-Agent"] == ua
