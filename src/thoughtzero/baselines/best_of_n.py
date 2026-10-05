"""B3 best-of-N picked by judge.final_correct (SPEC.md §8.1). Owner: Person 3 (Akul).

Implements ``eval.runner.Method``.

Reuses B2's stored samples when given them (``stored``), so it costs no GPU time, only one
judge call per sample. Scores are computed once for all ``N_max`` samples; the pick for any
``N`` is the argmax over the first ``N`` (ties to the earliest sample), which mirrors B2's
prefix subsampling.

Jev calls cost money: call ``estimate_usd`` and confirm before large runs.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable, Mapping, Sequence

from thoughtzero.baselines.self_consistency import (
    DEFAULT_TEMPERATURE,
    SCSamples,
    sample_self_consistency,
)
from thoughtzero.eval.runner import MethodResult
from thoughtzero.llm.prompts import split_steps
from thoughtzero.types import Generator, Judge, Problem

JEV_USD_PER_INPUT_TOKEN = 0.042 / 1e6  # SPEC.md §3


def estimate_usd(problems: Sequence[Problem], samples: Sequence[SCSamples]) -> float:
    """Rough Jev cost of scoring every sample (input tokens ~ chars / 4, as in SPEC.md §5.3)."""
    chars = sum(
        len(p.question) * len(s) + sum(len(sol) for sol in s.solutions)
        for p, s in zip(problems, samples, strict=True)
    )
    return chars / 4 * JEV_USD_PER_INPUT_TOKEN


async def score_samples(judge: Judge, problem: Problem, samples: SCSamples) -> list[float]:
    """``judge.final_correct`` for every stored sample, concurrently.

    The judge applies its own concurrency limit, cache and budget guard.
    """
    return list(
        await asyncio.gather(
            *(judge.final_correct(problem.question, split_steps(s)) for s in samples.solutions)
        )
    )


def pick_best(scores: Sequence[float], n: int) -> int:
    """Index of the highest score among the first ``n``; ties to the earliest."""
    if not 1 <= n <= len(scores):
        raise ValueError(f"n={n} outside 1..{len(scores)}")
    return max(range(n), key=lambda i: (scores[i], -i))


def curve(
    samples: SCSamples, scores: Sequence[float], ns: Iterable[int]
) -> dict[int, tuple[str | None, int]]:
    """``{N: (picked answer, completion tokens)}`` for each N."""
    return {n: (samples.answers[pick_best(scores, n)], samples.tokens(n)) for n in ns}


class BestOfN:
    """B3. Picks the best of the first ``n`` samples by ``judge.final_correct``.

    ``stored`` maps problem id -> B2 samples (e.g. ``SCSamples.from_raw`` over B2's JSONL);
    problems not in it are sampled fresh with ``generator`` (needs ``count_tokens``).
    All ``N_max`` samples are scored, so ``raw["scores"]`` supports a post-hoc curve.
    """

    name = "best_of_n"

    def __init__(
        self,
        judge: Judge,
        n: int,
        generator: Generator | None = None,
        count_tokens: Callable[[str], int] | None = None,
        stored: Mapping[str, SCSamples] | None = None,
        n_max: int | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> None:
        if stored is None and (generator is None or count_tokens is None):
            raise ValueError("need either stored samples or a generator and count_tokens")
        self.judge = judge
        self.n = n
        self.n_max = n_max or n
        self.generator = generator
        self.count_tokens = count_tokens
        self.stored = stored or {}
        self.temperature = temperature

    async def _samples(self, problem: Problem) -> SCSamples:
        if problem.id in self.stored:
            return self.stored[problem.id]
        if self.generator is None or self.count_tokens is None:
            raise KeyError(f"no stored samples for {problem.id} and no generator to sample")
        return await sample_self_consistency(
            self.generator, problem, self.n_max, self.count_tokens, self.temperature
        )

    async def solve(self, problem: Problem) -> MethodResult:
        samples = await self._samples(problem)
        if len(samples) < self.n:
            raise ValueError(f"{problem.id}: {len(samples)} stored samples < n={self.n}")
        scores = await score_samples(self.judge, problem, samples)
        best = pick_best(scores, self.n)
        raw = {
            "n": self.n,
            "picked": best,
            "scores": scores,
            "completion_tokens_at_n": samples.tokens(self.n),
            "reused_samples": problem.id in self.stored,
            **samples.to_raw(),
        }
        return MethodResult(answer=samples.answers[best], raw=raw)
