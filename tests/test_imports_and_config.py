"""Package installation/import sanity, RunConfig, and per-run model isolation."""

import importlib
import sys
import types

import pytest

from deep_research import _version
from deep_research.models import Credentials, ModelFactory
from deep_research.run_config import RunConfig


def test_package_version():
    assert _version.__version__ == "2.0.1"


def test_pyproject_exists_and_names_package():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    pyproject = (root / "pyproject.toml").read_text()
    assert "deep-dog-engine" in pyproject
    assert "include = [\"deep_research*\"]" in pyproject


def test_import_package_and_integration_entry_point():
    import deep_research  # noqa: F401
    import deep_research.integration as integration
    assert hasattr(integration, "run_research")
    assert integration.RunConfig is RunConfig
    assert integration.Credentials is Credentials
    assert "run_research" in integration.__all__


def test_import_all_subpackages_and_cli():
    # Ensure every public/agent module and the CLI import cleanly WITHOUT
    # requiring any provider API key (no import-time model construction).
    mods = [
        "deep_research.config",
        "deep_research.run_config",
        "deep_research.models",
        "deep_research.observer",
        "deep_research.runtime",
        "deep_research.events",
        "deep_research.cancellation",
        "deep_research.observability",
        "deep_research.research_agent_scope",
        "deep_research.research_agent_full",
        "deep_research.multi_agent_supervisor",
        "deep_research.agents",
        "deep_research.agents.web",
        "deep_research.agents.reddit",
        "deep_research.agents.substack",
        "deep_research.agents.pubmed",
        "deep_research.agents.arxiv",
        "deep_research.agents.sec_edgar",
        "deep_research.agents.shared",
    ]
    for name in mods:
        mod = importlib.import_module(name)
        assert isinstance(mod, types.ModuleType)

    # Standalone CLI imports (no keys required at import time).
    import run_research  # noqa: F401


def test_run_config_env_defaults_and_serialization():
    cfg = RunConfig().finalize()
    d = cfg.to_dict()
    assert d["output_mode"] == cfg.output_mode
    assert d["prompt_version"] == cfg.prompt_version

    roundtrip = RunConfig.from_dict(d)
    assert roundtrip.supervisor_model_fallback_chain == cfg.supervisor_model_fallback_chain
    assert roundtrip.enabled_agents == cfg.enabled_agents


def test_run_config_overrides_and_hard_caps():
    cfg = RunConfig(
        profile="muse-spark-1.2",
        research_time_min_minutes=2,
        research_time_max_minutes=10,
        max_duration_minutes=30.0,
        supervisor_max_iterations=99999,
        subagent_max_iterations=99999,
        output_mode="none",
    ).finalize()
    assert cfg.profile == "muse-spark-1.2"
    assert cfg.strict_timeout_minutes == 30.0
    # Hard caps clamp runaway values.
    assert cfg.supervisor_max_iterations <= 500
    assert cfg.subagent_max_iterations <= 60


def test_model_factory_per_run_isolation():
    creds = Credentials({"DEEPSEEK_API_KEY": "sk-test-not-real"})
    cfg_a = RunConfig(
        profile="A",
        subagent_model_fallback_chain=["deepseek-v4-pro"],
        disable_model_fallback=True,
        route_via_openrouter=False,
    )
    cfg_b = RunConfig(
        profile="B",
        subagent_model_fallback_chain=["deepseek-v4-flash"],
        disable_model_fallback=True,
        route_via_openrouter=False,
    )
    mf_a = ModelFactory(cfg_a, creds)
    mf_b = ModelFactory(cfg_b, creds)

    model_a1 = mf_a.subagent()
    model_a2 = mf_a.subagent()  # cached per run
    model_b = mf_b.subagent()

    assert model_a1 is model_a2
    assert model_a1 is not model_b
    assert model_a1.__class__ is model_b.__class__  # same provider family


def test_model_factory_cached_per_run():
    creds = Credentials({"DEEPSEEK_API_KEY": "sk-test-not-real"})
    cfg = RunConfig(subagent_model_fallback_chain=["deepseek-v4-flash"],
                    disable_model_fallback=True)
    mf = ModelFactory(cfg, creds)
    first = mf.supervisor()
    second = mf.supervisor()
    assert first is second
