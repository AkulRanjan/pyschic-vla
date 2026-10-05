"""BudgetGuard: hard USD cap on Jev spend (SPEC.md §5.4). Owner: Person 2 (Jagriti)."""

from __future__ import annotations


class BudgetExceeded(RuntimeError):
    """Raised when a call would push spend over ``budget.max_usd``. Stops the whole run."""


class BudgetGuard:
    def __init__(self, max_usd: float, usd_per_mtok: float = 0.042) -> None:
        self.max_usd = max_usd
        self.usd_per_mtok = usd_per_mtok
        self.spent_usd = 0.0

    def check(self, estimated_tokens: int) -> None:
        raise NotImplementedError("Person 2 (Jagriti): reserve before await (concurrency-safe)")

    def record(self, actual_tokens: int, estimated_tokens: int) -> None:
        raise NotImplementedError("Person 2 (Jagriti)")
