"""Cooperative cancellation for concurrent research runs.

Cancellation is cooperative: a currently executing HTTP/model/tool request is
allowed to finish; the runtime checks the token between scoping steps,
supervisor iterations, sub-agent batches, tool batches, report generation,
and citation finalization, and the integration driver checks it between graph
steps. Research time limits route the supervisor to final writing; they do
not cancel the run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


class RunCancelledError(Exception):
    """Raised when a research run is cancelled between cooperative steps."""


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
