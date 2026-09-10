"""Per-run model construction.

Historically every model client was built from environment variables at module
import time (``config.get_supervisor_model()`` etc.). That shared one global
supervisor/writer/sub-agent client across all runs. For a concurrent API that
is unsafe: one user's Muse Spark run and another's DeepSeek run must not share
a client or switch models by mutating ``os.environ``.

``ModelFactory`` is bound to a single ``RunConfig`` + ``Credentials`` and
produces isolated model clients per run. It wraps the legacy, credentials-aware
builders in ``deep_research.config`` and scopes credential resolution via
``config.credential_scope`` so keys are taken from the run's ``Credentials``
(never the process environment).

Results are cached per ``(role, chain, tool set, max_tokens, temperature,
reasoning_effort, route)`` so a run never rebuilds the supervisor client on
every iteration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from deep_research import config as _cfg
from deep_research.run_config import RunConfig


@dataclass
class Credentials:
    """Provider credentials supplied by the host application for one run.

    ``keys`` maps environment-var names (e.g. ``"DEEPSEEK_API_KEY"``,
    ``"OPENROUTER_API_KEY"``, ``"TAVILY_API_KEY"``) to secret values.
    ``.get()`` falls back to the process environment, so a run that does not
    override a provider still works in deployments that set env vars.
    """

    keys: dict[str, str] = field(default_factory=dict)

    def get(self, name: str) -> Optional[str]:
        value = self.keys.get(name)
        if value:
            return value
        return os.getenv(name)

    def has(self, name: str) -> bool:
        return bool(self.get(name))

    @classmethod
    def from_env(cls) -> "Credentials":
        """Build Credentials from the current process environment (CLI)."""
        return cls(keys={})


class ModelFactory:
    """Builds isolated, per-run LangChain model clients."""

    def __init__(self, config: RunConfig, credentials: Optional[Credentials] = None):
        self.config = config
        self.credentials = credentials or Credentials(keys={})
        self._cache: dict[tuple, Any] = {}

    # ── public role builders ───────────────────────────────────────────
    def supervisor(self, tools: Optional[list] = None, *, max_tokens: Optional[int] = 32000,
                   temperature: float = 0, chain: Optional[list] = None,
                   reasoning_effort: Optional[str] = None):
        chain = list(chain or self.config.supervisor_model_fallback_chain)
        route = self.config.resolved_role_route("supervisor")
        return self._cached("supervisor", tools, max_tokens, temperature,
                            tuple(chain), reasoning_effort, route)

    def draft_report(self, *, max_tokens: Optional[int] = 32000, temperature: float = 0,
                     chain: Optional[list] = None, reasoning_effort: Optional[str] = None):
        chain = list(chain or self.config.draft_report_model_fallback_chain)
        route = self.config.resolved_role_route("draft")
        return self._cached("draft_report", None, max_tokens, temperature,
                            tuple(chain), reasoning_effort or None, route)

    def subagent(self, tools: Optional[list] = None, *, max_tokens: Optional[int] = None,
                 temperature: float = 0, chain: Optional[list] = None,
                 reasoning_effort: Optional[str] = None):
        chain = list(chain or self.config.subagent_model_fallback_chain)
        route = self.config.resolved_role_route("subagent")
        return self._cached("subagent", tools, max_tokens, temperature,
                            tuple(chain), reasoning_effort, route)

    def get(self, model_name: str, *, temperature: float = 0, max_tokens: Optional[int] = None,
            reasoning_effort: Optional[str] = None, route_via_openrouter: Optional[bool] = None):
        """Single model by name (route honours this run's config by default)."""
        route = bool(self.config.route_via_openrouter) if route_via_openrouter is None else bool(route_via_openrouter)
        with _cfg.credential_scope(self.credentials):
            return _cfg.get_model(
                model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                route_via_openrouter=route,
                reasoning_effort=reasoning_effort,
                thinking_mode=self.config.thinking_mode,
            )

    # ── internals ──────────────────────────────────────────────────────
    def _tool_key(self, tools) -> tuple:
        if not tools:
            return ()
        names = []
        for t in tools:
            n = getattr(t, "name", None)
            names.append(n if n is not None else str(t))
        return tuple(sorted(names))

    def _cached(self, kind: str, tools, max_tokens, temperature, chain: tuple,
                reasoning_effort, route: bool):
        key = (kind, self._tool_key(tools), max_tokens, temperature, chain,
               reasoning_effort, route)
        model = self._cache.get(key)
        if model is None:
            model = self._build(kind, tools=tools, max_tokens=max_tokens,
                                temperature=temperature, chain=chain,
                                reasoning_effort=reasoning_effort, route=route)
            self._cache[key] = model
        return model

    def _build(self, kind: str, *, tools, max_tokens, temperature, chain,
               reasoning_effort, route):
        common = dict(
            max_tokens=max_tokens,
            temperature=temperature,
            thinking_mode=self.config.thinking_mode,
            disable_fallback=self.config.disable_model_fallback,
        )
        with _cfg.credential_scope(self.credentials):
            if kind == "supervisor":
                return _cfg.get_supervisor_model(
                    tools=tools, chain=list(chain),
                    route_via_openrouter=route,
                    reasoning_effort=reasoning_effort,
                    **common)
            if kind == "draft_report":
                return _cfg.get_draft_report_model(
                    chain=list(chain), route_via_openrouter=route,
                    reasoning_effort=reasoning_effort, **common)
            if kind == "subagent":
                return _cfg.get_subagent_model(
                    tools=tools, chain=list(chain),
                    route_via_openrouter=route, **common)
            raise ValueError(f"unknown model kind: {kind}")
