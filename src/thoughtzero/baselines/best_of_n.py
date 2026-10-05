"""B3: best-of-N, picked by the judge's ``final_correct`` (spec §8.1, team file §A5.2).

Reuses B2's stored samples, so it costs no GPU time, only one judge call per
sample. Scores are computed once for all ``N_max`` samples; the pick for any
``N`` is the argmax over the first ``N`` (ties to the earliest sample), which
mirrors B2's prefix subsampling.

Jev calls cost money: call ``estimate_usd`` and confirm before large runs.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Sequence

from thoughtzero.baselines.common import BaselineOutput
from thoughtzero.baselines.self_consistency import SCSamples
from thoughtzero.llm.prompts import split_steps
from thoughtzero.types import Judge, Problem

JEV_USD_PER_INPUT_TOKEN = 0.042 / 1e6  # spec §3


def estimate_usd(problems: Sequence[Problem], samples: Sequence[SCSamples]) -> float:
    """Rough Jev cost of scoring every sample (input tokens ~ chars / 4, as in spec §5.3)."""
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


def output(samples: SCSamples, scores: Sequence[float], n: int) -> BaselineOutput:
    best = pick_best(scores, n)
    return BaselineOutput(
        answer=samples.answers[best],
        solutions=samples.solutions[:n],
        extra={
            "n": n,
            "picked": best,
            "scores": list(scores[:n]),
            "completion_tokens": samples.tokens(n),
        },
    )


def curve(
    samples: SCSamples, scores: Sequence[float], ns: Iterable[int]
) -> dict[int, tuple[str | None, int]]:
    """``{N: (picked answer, completion tokens)}`` for each N."""
    return {n: (samples.answers[pick_best(scores, n)], samples.tokens(n)) for n in ns}


async def run_best_of_n(
    judge: Judge, problem: Problem, samples: SCSamples, n: int | None = None
) -> BaselineOutput:
    """Score the stored samples and pick the best of the first ``n`` (default: all)."""
    scores = await score_samples(judge, problem, samples)
    return output(samples, scores, n or len(samples))
