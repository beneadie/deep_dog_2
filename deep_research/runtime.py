"""Per-run runtime context for the Deep Dog engine.

A ``RuntimeContext`` bundles everything a research run needs that used to live
in process globals: resolved config, per-run credentials + model factory,
per-run observer (events/sources/trace/counters), cancellation token,
plus an optional external LangGraph checkpointer.

The context is installed in a ``contextvars.ContextVar`` by the integration
driver for the duration of a run. Graph nodes read ``get_runtime()`` to obtain
their isolated models/config/observer. asyncio child tasks (``asyncio.gather``
sub-agent batches) inherit the current context automatically, so two concurrent
runs never share model, config, trace, or source state.

When no context is installed (legacy direct graph use, e.g. ``run_platform.py``
or the CLI), ``get_runtime()`` returns a memoized default context built from
environment defaults, preserving the old standalone behaviour.
"""

from __future__ import annotations

import contextvars
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Optional

from deep_research.cancellation import (
    CancellationToken,
    RunCancelledError,
    as_cancellation_checker,
)
from deep_research.models import Credentials, ModelFactory
from deep_research.observer import Observer
from deep_research.run_config import RunConfig

# ── Prompt library ──────────────────────────────────────────────────────


class PromptLibrary:
    """Access the OPEN prompt bundle; reject retired prompt versions."""

    def __init__(self, version: str):
        self.version = (version or "OPEN").upper()
        if self.version != "OPEN":
            raise ValueError("Only the OPEN prompt version is supported.")

    @property
    def module(self) -> Any:
        from deep_research import prompts_open
        return prompts_open

    def get(self, name: str, default: Any = None) -> Any:
        return getattr(self.module, name, default)


# ── Runtime context ─────────────────────────────────────────────────────

class _Missing:
    pass


_MISSING = _Missing()


@dataclass
class RuntimeContext:
    """Everything a single research run needs, isolated per run."""

    run_id: str
    thread_id: str
    config: RunConfig
    credentials: Credentials = field(default_factory=Credentials.from_env)
    models: Optional[ModelFactory] = None
    observer: Optional[Observer] = None
    cancellation: Any = None            # CancellationToken | ()->bool | None
    checkpointer: Any = None            # external LangGraph checkpointer | None
    console_enabled: bool = True        # CLI prints on; API off
    start_monotonic: float = field(default_factory=time.monotonic)
    host: Any = None                    # opaque host-provided runtime object

    def __post_init__(self):
        if self.models is None:
            self.models = ModelFactory(self.config, self.credentials)
        if self.observer is None:
            self.observer = Observer(
                run_id=self.run_id,
                config=self.config,
                event_sink=None,
                artifact_sink=None,
                console_enabled=self.console_enabled,
            )

    @property
    def prompts(self) -> PromptLibrary:
        lib = getattr(self, "_prompts", None)
        if lib is None:
            lib = PromptLibrary(self.config.prompt_version)
            object.__setattr__(self, "_prompts", lib)
        return lib

    def remaining_minutes(self) -> float:
        """Advisory research time remaining for telemetry, never a run cutoff."""
        elapsed = (time.monotonic() - self.start_monotonic) / 60.0
        return max(0.0, self.config.strict_timeout_minutes - elapsed)

    def check_cancelled(self) -> None:
        """Raise RunCancelledError if the cooperative cancellation flag is set."""
        checker = getattr(self, "_cancel_checker", None)
        if checker is None:
            checker = as_cancellation_checker(self.cancellation)
            object.__setattr__(self, "_cancel_checker", checker)
        if checker():
            raise RunCancelledError("run cancelled")

    def raise_if_cancelled(self) -> None:
        self.check_cancelled()

    # ── Serialization (config snapshot) ────────────────────────────────
    def config_snapshot(self) -> dict:
        return self.config.to_dict()

    @classmethod
    def from_snapshot(cls, snapshot: dict, **overrides) -> "RuntimeContext":
        """Rebuild a runtime context from a config snapshot (checkpoint resume)."""
        config = RunConfig.from_dict(snapshot).finalize()
        return cls(
            run_id=overrides.pop("run_id", "resume"),
            thread_id=overrides.pop("thread_id", "resume"),
            config=config,
            **overrides,
        )


# ── ContextVar plumbing ─────────────────────────────────────────────────
_runtime_var: contextvars.ContextVar = contextvars.ContextVar(
    "deep_research_runtime", default=None
)

_default_runtime: Optional[RuntimeContext] = None
_default_runtime_lock = threading.Lock()


def get_runtime() -> RuntimeContext:
    """Return the active run context, or the default env-based one."""
    ctx = _runtime_var.get()
    if ctx is not None:
        return ctx
    global _default_runtime
    if _default_runtime is None:
        with _default_runtime_lock:
            if _default_runtime is None:
                config = RunConfig().finalize()
                _default_runtime = RuntimeContext(
                    run_id="default",
                    thread_id="default",
                    config=config,
                    credentials=Credentials.from_env(),
                    console_enabled=True,
                )
    return _default_runtime


def set_runtime(runtime: RuntimeContext) -> None:
    _runtime_var.set(runtime)


def reset_runtime(token: contextvars.Token) -> None:
    _runtime_var.reset(token)


@contextmanager
def runtime_scope(runtime: RuntimeContext):
    """Install a runtime for the duration of a block (asyncio-safe)."""
    token = _runtime_var.set(runtime)
    try:
        yield runtime
    finally:
        _runtime_var.reset(token)
