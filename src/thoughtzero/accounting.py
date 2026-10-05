"""Per-problem token / cost / time ledger (SPEC.md §5.4, team/<Name>.md §B4.7).

Owner: Person 4 (Harjas). Shared contract: changing ``Ledger`` fields needs a PR tagged to all.

Usage::

    with ledger_scope() as ledger:       # eval/runner.py, once per problem
        await method.solve(problem)       # generator/judge call record_* inside
    ledger.gemma_completion_tokens       # primary compute axis

Tasks created inside the scope inherit the ContextVar and all mutate the same ``Ledger``
(asyncio is single-threaded, so no locking is needed). Do not record from worker threads.
``record_*`` is a no-op when no ledger is active, so unit tests need no setup.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class Ledger:
    gemma_prompt_tokens: int = 0
    gemma_completion_tokens: int = 0
    gemma_calls: int = 0
    gemma_usd: float = 0.0  # hosted Gemma only (generator.usd_per_mtok_*); 0 if self-hosted
    jev_calls: int = 0
    jev_cache_hits: int = 0
    jev_input_tokens: int = 0
    jev_usd: float = 0.0
    wall_time_s: float = 0.0
    expansions: int = 0
    max_depth: int = 0
    terminal_leaves: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


current_ledger: ContextVar[Ledger | None] = ContextVar("current_ledger", default=None)


def record_gemma(prompt_tokens: int, completion_tokens: int, usd: float = 0.0) -> None:
    ledger = current_ledger.get()
    if ledger is None:
        return
    ledger.gemma_calls += 1
    ledger.gemma_prompt_tokens += prompt_tokens
    ledger.gemma_completion_tokens += completion_tokens
    ledger.gemma_usd += usd


def record_jev(input_tokens: int, usd: float, cache_hit: bool) -> None:
    ledger = current_ledger.get()
    if ledger is None:
        return
    if cache_hit:
        ledger.jev_cache_hits += 1
        return
    ledger.jev_calls += 1
    ledger.jev_input_tokens += input_tokens
    ledger.jev_usd += usd


def record_search(expansions: int = 0, depth: int = 0, terminal_leaves: int = 0) -> None:
    """Search diagnostics; ``depth`` updates the running maximum."""
    ledger = current_ledger.get()
    if ledger is None:
        return
    ledger.expansions += expansions
    ledger.terminal_leaves += terminal_leaves
    ledger.max_depth = max(ledger.max_depth, depth)


@contextmanager
def ledger_scope() -> Iterator[Ledger]:
    ledger = Ledger()
    token = current_ledger.set(ledger)
    start = time.perf_counter()
    try:
        yield ledger
    finally:
        ledger.wall_time_s = time.perf_counter() - start
        current_ledger.reset(token)
