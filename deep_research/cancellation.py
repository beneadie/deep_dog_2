"""Cancellation and deadline primitives for concurrent research runs.

Cancellation is cooperative: a currently executing HTTP/model/tool request is
allowed to finish; the runtime checks the token between scoping steps,
supervisor iterations, sub-agent batches, tool batches, report generation,
and citation finalization, and the integration driver checks it between graph
steps. Deadline enforcement is similar — nodes and the driver observe the
remaining budget and stop/salvage safely when it is exhausted.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


class RunCancelledError(Exception):
    """Raised when a research run is cancelled between cooperative steps."""


class RunDeadlineExceededError(Exception):
    """Raised when a research run exceeds its wall-clock deadline."""


@dataclass
class CancellationToken:
    """A mutable cooperative cancellation flag.

    The host application can flip ``.cancelled`` (or call ``.cancel()``) from
    any thread; the run observes it between cooperative steps.
    """

    cancelled: bool = False
    reason: str | None = None

    def cancel(self, reason: str | None = None) -> None:
        self.cancelled = True
        if reason is not None:
            self.reason = reason

    def is_cancelled(self) -> bool:
        return bool(self.cancelled)

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise RunCancelledError(self.reason or "cancelled")


CancellationChecker = Callable[[], bool]
"""A host-supplied callable returning True when the run should stop."""


def as_cancellation_checker(token_or_checker: Any) -> Callable[[], bool]:
    """Normalize a token/checker into a plain ``() -> bool`` checker."""
    if token_or_checker is None:
        return lambda: False
    if isinstance(token_or_checker, CancellationToken):
        return token_or_checker.is_cancelled
    if callable(token_or_checker):
        return token_or_checker
    # duck-typed token with is_cancelled()
    check = getattr(token_or_checker, "is_cancelled", None)
    if callable(check):
        return check
    raise TypeError(
        "cancellation must be a CancellationToken, a ()->bool callable, or an "
        "object with is_cancelled()"
    )


@dataclass
class Deadline:
    """Wall-clock deadline for a research run (monotonic clock)."""

    start_monotonic: float = field(default_factory=time.monotonic)
    max_seconds: float | None = None

    @classmethod
    def after(cls, max_seconds: float | None) -> "Deadline":
        return cls(start_monotonic=time.monotonic(), max_seconds=max_seconds)

    @property
    def deadline_monotonic(self) -> float | None:
        if self.max_seconds is None:
            return None
        return self.start_monotonic + self.max_seconds

    def remaining_seconds(self) -> float | None:
        if self.max_seconds is None:
            return None
        return max(0.0, self.deadline_monotonic - time.monotonic())

    def expired(self) -> bool:
        if self.max_seconds is None:
            return False
        return time.monotonic() >= self.deadline_monotonic

    def remaining_minutes(self) -> float:
        rem = self.remaining_seconds()
        if rem is None:
            return float("inf")
        return rem / 60.0
