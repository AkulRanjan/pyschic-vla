"""GemmaSelfJudge: the generator model as its own judge (baseline B4, SPEC.md §5.3).

The same questions as ``JevJudge`` (``prompts.SOUND_VARIANTS``, ``NEXT_INSTRUCTION``,
``FINAL_INSTRUCTION`` over the same state text), posed to Gemma as chat prompts. Each answer
is read from the **first token's logprobs**, one greedy request with ``max_tokens=1``:

- value / ``step_sound`` / ``final_correct``: ``p(Yes) / (p(Yes) + p(No))``, summing token
  variants (``Yes``, `` yes``, ...);
- priors: candidates are lettered A, B, C, ...; each letter's probability, renormalised.
  A letter missing from the returned top-k gets a small epsilon (``base.renormalize``).

``prior_and_value`` makes the two requests in parallel. The root (no steps) gets
``judge.root_value``, as with Jev. A route that returns no logprobs at all is an error
(the answer would be a guess).
"""

from __future__ import annotations

import asyncio
import logging
import math
from typing import Protocol

from thoughtzero.config import JudgeCfg
from thoughtzero.judge.base import renormalize
from thoughtzero.llm.prompts import (
    FINAL_INSTRUCTION,
    LETTERS,
    SOUND_VARIANTS,
    self_judge_choice_messages,
    self_judge_yes_no_messages,
)

log = logging.getLogger(__name__)

TOP_K = 20


class LogprobSource(Protocol):
    """What the self-judge needs from a generator (both generator classes implement it)."""

    async def first_token_logprobs(
        self, messages: list[dict[str, str]], top_k: int = 20
    ) -> dict[str, float]: ...


class NoLogprobsError(RuntimeError):
    """The model route returned no logprobs, so the self-judge can't read an answer."""


def _clean(token: str) -> str:
    """``" Yes"`` -> ``"yes"``; ``"B."`` -> ``"b"`` (SentencePiece ``▁`` counts as a space)."""
    return token.replace("▁", " ").strip().strip(".:)*").lower()


def yes_probability(logprobs: dict[str, float]) -> float | None:
    """``p(yes) / (p(yes) + p(no))`` over the returned tokens; None if neither appears."""
    yes = sum(math.exp(lp) for tok, lp in logprobs.items() if _clean(tok) == "yes")
    no = sum(math.exp(lp) for tok, lp in logprobs.items() if _clean(tok) == "no")
    if yes + no == 0:
        return None
    return yes / (yes + no)


def letter_probabilities(logprobs: dict[str, float], n: int) -> list[float | None]:
    """Probability of each of the first ``n`` letters; None for a letter not returned."""
    out: list[float | None] = []
    for letter in LETTERS[:n].lower():
        p = sum(math.exp(lp) for tok, lp in logprobs.items() if _clean(tok) == letter)
        out.append(p if p > 0 else None)
    return out


class GemmaSelfJudge:
    def __init__(self, cfg: JudgeCfg, source: LogprobSource) -> None:
        self.cfg = cfg
        self.source = source
        self.sound_instruction = SOUND_VARIANTS[cfg.sound_variant]

    async def _logprobs(self, messages: list[dict[str, str]]) -> dict[str, float]:
        lps = await self.source.first_token_logprobs(messages, TOP_K)
        if not lps:
            raise NoLogprobsError("the generator route returned no logprobs")
        return lps

    async def _yes(self, messages: list[dict[str, str]]) -> float:
        lps = await self._logprobs(messages)
        p = yes_probability(lps)
        if p is None:
            log.warning("self-judge: neither Yes nor No in the top tokens %s; using 0.5", list(lps))
            return 0.5
        return p

    async def priors(self, problem: str, steps: list[str], candidates: list[str]) -> list[float]:
        if len(candidates) == 1:
            return [1.0]
        lps = await self._logprobs(self_judge_choice_messages(problem, steps, candidates))
        return renormalize(letter_probabilities(lps, len(candidates)))

    async def prior_and_value(
        self, problem: str, steps: list[str], candidates: list[str]
    ) -> tuple[list[float], float]:
        priors, value = await asyncio.gather(
            self.priors(problem, steps, candidates), self.step_sound(problem, steps)
        )
        return priors, value

    async def prior_only(
        self, problem: str, steps: list[str], candidates: list[str]
    ) -> list[float]:
        """For ``HybridJudge`` (prior from here, value from another judge)."""
        return await self.priors(problem, steps, candidates)

    async def value_only(self, problem: str, steps: list[str]) -> float:
        return await self.step_sound(problem, steps)

    async def step_sound(self, problem: str, steps: list[str]) -> float:
        if not steps:
            return self.cfg.root_value
        return await self._yes(self_judge_yes_no_messages(problem, steps, self.sound_instruction))

    async def final_correct(self, problem: str, steps: list[str]) -> float:
        messages = self_judge_yes_no_messages(problem, steps, FINAL_INSTRUCTION, final=True)
        return await self._yes(messages)
