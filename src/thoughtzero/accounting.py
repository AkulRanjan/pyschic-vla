"""Per-problem token / cost / time ledger (team file §B4 item 7).

STAND-IN: owned by Person 4 (written into Person 1's scaffold PR). This copy
mirrors §B4 so the generator can record usage. Replace with the scaffold's
version; do not edit here.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass


@dataclass
class Ledger:
    gemma_prompt_tokens: int = 0
    gemma_completion_tokens: int = 0  # PRIMARY compute axis
    gemma_calls: int = 0
    jev_calls: int = 0
    jev_cache_hits: int = 0
    jev_input_tokens: int = 0
    jev_usd: float = 0.0
    wall_time_s: float = 0.0
    expansions: int = 0
    max_depth: int = 0
    terminal_leaves: int = 0


# The runner sets a fresh Ledger per problem; the default only catches stray calls.
current_ledger: ContextVar[Ledger] = ContextVar("current_ledger", default=Ledger())  # noqa: B039


def record_gemma(prompt_tokens: int, completion_tokens: int) -> None:
    led = current_ledger.get()
    led.gemma_prompt_tokens += prompt_tokens
    led.gemma_completion_tokens += completion_tokens
    led.gemma_calls += 1


def record_jev(input_tokens: int, usd: float, cache_hit: bool) -> None:
    led = current_ledger.get()
    led.jev_calls += 1
    led.jev_input_tokens += input_tokens
    led.jev_usd += 0.0 if cache_hit else usd
    if cache_hit:
        led.jev_cache_hits += 1
