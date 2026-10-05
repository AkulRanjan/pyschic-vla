"""Ledger / accounting (spec B4, item 7). Local copy, owned jointly by
Person 1 and Person 4 on the real team repo; co-authored here just enough
for judge/ to call record_jev. Replace once reconciled.
"""

from contextvars import ContextVar
from dataclasses import dataclass


@dataclass
class Ledger:
    gemma_prompt_tokens: int = 0
    gemma_completion_tokens: int = 0
    gemma_calls: int = 0
    jev_calls: int = 0
    jev_cache_hits: int = 0
    jev_input_tokens: int = 0
    jev_usd: float = 0.0
    wall_time_s: float = 0.0
    expansions: int = 0
    max_depth: int = 0
    terminal_leaves: int = 0


current_ledger: ContextVar[Ledger] = ContextVar("current_ledger")


def record_gemma(prompt_tokens: int, completion_tokens: int) -> None:
    ledger = current_ledger.get()
    ledger.gemma_prompt_tokens += prompt_tokens
    ledger.gemma_completion_tokens += completion_tokens
    ledger.gemma_calls += 1


def record_jev(input_tokens: int, usd: float, cache_hit: bool) -> None:
    ledger = current_ledger.get()
    ledger.jev_calls += 1
    ledger.jev_input_tokens += input_tokens
    ledger.jev_usd += usd
    if cache_hit:
        ledger.jev_cache_hits += 1
