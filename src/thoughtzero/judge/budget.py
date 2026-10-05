"""Budget guard for Jev spend (spec A4.4)."""

import asyncio
import json
import os
import time
from pathlib import Path

PRICE_USD_PER_1M_INPUT_TOKENS = 0.042


def usd(input_tokens: int) -> float:
    return input_tokens * PRICE_USD_PER_1M_INPUT_TOKENS / 1e6


class BudgetExceeded(Exception):
    pass


class BudgetGuard:
    """Reserve-then-settle so concurrent `await`s can't all pass `check`
    at once and jointly overshoot `max_usd`.
    """

    def __init__(self, max_usd: float, spend_log: str = "results/jev_spend.jsonl") -> None:
        self.max_usd = max_usd
        self.spent = 0.0
        self._reserved = 0.0
        self._lock = asyncio.Lock()
        self.spend_log = spend_log

    async def check(self, estimated_tokens: int) -> float:
        estimate = usd(estimated_tokens)
        async with self._lock:
            if self.spent + self._reserved + estimate > self.max_usd:
                raise BudgetExceeded(
                    f"budget exceeded: spent={self.spent:.4f} reserved={self._reserved:.4f} "
                    f"estimate={estimate:.4f} max_usd={self.max_usd:.4f}"
                )
            self._reserved += estimate
        return estimate

    async def record(
        self, reserved_estimate: float, actual_tokens: int, cache_hit: bool = False
    ) -> float:
        actual = 0.0 if cache_hit else usd(actual_tokens)
        async with self._lock:
            self._reserved = max(0.0, self._reserved - reserved_estimate)
            self.spent += actual
        self._append_spend_log(actual_tokens, actual, cache_hit)
        return actual

    def _append_spend_log(self, tokens: int, usd_spent: float, cache_hit: bool) -> None:
        person = os.environ.get("TZ_PERSON", "unknown")
        run_id = os.environ.get("TZ_RUN_ID", "local")
        row = {
            "timestamp": time.time(),
            "person": person,
            "tokens": tokens,
            "usd": usd_spent,
            "cache_hit": cache_hit,
            "run_id": run_id,
        }
        path = Path(self.spend_log)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(json.dumps(row) + "\n")


def estimate_run_cost(n_calls: int, avg_tokens: float) -> float:
    return usd(int(n_calls * avg_tokens))


def print_estimate_and_confirm(n_calls: int, avg_tokens: float, yes: bool) -> None:
    estimate = estimate_run_cost(n_calls, avg_tokens)
    print(
        f"Estimated cost: ${estimate:.4f} for {n_calls} calls @ {avg_tokens:.0f} avg input tokens"
    )
    if estimate > 1.0 and not yes:
        raise SystemExit("Estimated cost exceeds $1 — rerun with --yes to proceed (spec §5.4).")
