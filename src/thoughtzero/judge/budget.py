"""BudgetGuard: hard USD cap on Jev spend (SPEC.md §5.4). Owner: Person 2 (Jagriti).

``check``/``record`` are deliberately sync (no ``await`` inside): a coroutine
can't be preempted mid-call, so reserving in ``check`` before the real HTTP
``await`` and settling in ``record`` after it is already concurrency-safe
with no lock needed. ``check`` must still run *before* the await and
``record`` *after* it, or N concurrent calls can jointly overshoot.

A process-wide instance is exposed via ``current_guard`` (parallels
``accounting.current_ledger``): unset by default, so judges and tests don't
need to configure one unless the run actually wants a hard cap.
"""

from __future__ import annotations


class BudgetExceeded(RuntimeError):
    """Raised when a call would push spend over ``budget.max_usd``. Stops the whole run."""


class BudgetGuard:
    def __init__(self, max_usd: float, usd_per_mtok: float = 0.042) -> None:
        self.max_usd = max_usd
        self.usd_per_mtok = usd_per_mtok
        self.spent_usd = 0.0
        self._reserved_usd = 0.0

    def _usd(self, tokens: int) -> float:
        return tokens * self.usd_per_mtok / 1e6

    def check(self, estimated_tokens: int) -> None:
        estimate = self._usd(estimated_tokens)
        if self.spent_usd + self._reserved_usd + estimate > self.max_usd:
            raise BudgetExceeded(
                f"budget exceeded: spent=${self.spent_usd:.4f} reserved=${self._reserved_usd:.4f} "
                f"+estimate=${estimate:.4f} > max_usd=${self.max_usd:.4f}"
            )
        self._reserved_usd += estimate

    def record(self, actual_tokens: int, estimated_tokens: int) -> None:
        self._reserved_usd = max(0.0, self._reserved_usd - self._usd(estimated_tokens))
        self.spent_usd += self._usd(actual_tokens)


current_guard: BudgetGuard | None = None


def configure(max_usd: float, usd_per_mtok: float = 0.042) -> BudgetGuard:
    global current_guard
    current_guard = BudgetGuard(max_usd, usd_per_mtok)
    return current_guard


def estimate_run_cost(n_calls: int, avg_tokens: float, usd_per_mtok: float = 0.042) -> float:
    return n_calls * avg_tokens * usd_per_mtok / 1e6


class SpendCap:
    """Hard USD cap on hosted-Gemma spend, charged after each call (a call already made has
    been paid for, so the cap can be overshot by at most one call)."""

    def __init__(self, max_usd: float) -> None:
        self.max_usd = max_usd
        self.spent_usd = 0.0

    def charge(self, usd: float) -> None:
        self.spent_usd += usd
        if self.spent_usd > self.max_usd:
            raise BudgetExceeded(
                f"Gemma budget exceeded: spent=${self.spent_usd:.4f} > max=${self.max_usd:.4f}"
            )


gemma_cap: SpendCap | None = None


def configure_gemma(max_usd: float) -> SpendCap:
    global gemma_cap
    gemma_cap = SpendCap(max_usd)
    return gemma_cap


def charge_gemma(usd: float) -> None:
    """Called by the hosted generator after every request; no-op without a cap."""
    if gemma_cap is not None and usd > 0:
        gemma_cap.charge(usd)
