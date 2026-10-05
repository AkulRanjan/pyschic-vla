"""B2: self-consistency, the primary baseline (spec §8.1, team file §A5.2).

Sample ``N_max`` solutions once and store them all, with per-sample token
counts. The vote for any ``N <= N_max`` is then computed post hoc from the
first ``N`` samples (prefix subsampling), so one run gives the whole
accuracy-vs-compute curve, and N can be matched to ThoughtZero's tokens later.

Voting rules:
- answers are grouped by ``grading.normalize_answer`` (the same grouping as
  ``value_vote``); ``None`` and empty answers are ignored;
- ties go to the group whose first occurrence came earliest;
- the returned answer is that group's first raw answer.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

from thoughtzero.baselines.common import BaselineOutput
from thoughtzero.data.grading import extract_answer, normalize_answer
from thoughtzero.llm.prompts import step_header
from thoughtzero.types import Generator, Problem

DEFAULT_TEMPERATURE = 0.7


def majority_vote(answers: Sequence[str | None]) -> str | None:
    """Most common answer by normalized form; ties to the earliest first occurrence."""
    counts: dict[str, int] = {}
    first: dict[str, int] = {}
    for i, a in enumerate(answers):
        key = normalize_answer(a)
        if not key:
            continue
        counts[key] = counts.get(key, 0) + 1
        first.setdefault(key, i)
    if not counts:
        return None
    winner = max(counts, key=lambda k: (counts[k], -first[k]))
    return answers[first[winner]]


@dataclass
class SCSamples:
    """Stored samples for one problem, in sampling order."""

    solutions: list[str]
    answers: list[str | None]
    completion_tokens: list[int]  # per sample
    temperature: float = DEFAULT_TEMPERATURE
    vote_cache: dict[int, str | None] = field(default_factory=dict, repr=False)

    def __len__(self) -> int:
        return len(self.solutions)

    def vote(self, n: int) -> str | None:
        """Majority vote over the first ``n`` samples."""
        if not 1 <= n <= len(self):
            raise ValueError(f"n={n} outside 1..{len(self)}")
        if n not in self.vote_cache:
            self.vote_cache[n] = majority_vote(self.answers[:n])
        return self.vote_cache[n]

    def tokens(self, n: int) -> int:
        """Completion tokens spent by the first ``n`` samples (the compute axis)."""
        return sum(self.completion_tokens[:n])

    def output(self, n: int) -> BaselineOutput:
        return BaselineOutput(
            answer=self.vote(n),
            solutions=self.solutions[:n],
            extra={
                "n": n,
                "completion_tokens": self.tokens(n),
                "answers": self.answers[:n],
                "temperature": self.temperature,
            },
        )


def curve(samples: SCSamples, ns: Iterable[int]) -> dict[int, tuple[str | None, int]]:
    """``{N: (voted answer, completion tokens)}`` for each N, from one stored run."""
    return {n: (samples.vote(n), samples.tokens(n)) for n in ns}


async def sample_self_consistency(
    generator: Generator,
    problem: Problem,
    n_max: int,
    count_tokens: Callable[[str], int],
    temperature: float = DEFAULT_TEMPERATURE,
    batch_size: int = 16,
) -> SCSamples:
    """Draw ``n_max`` solutions in batched ``n=batch_size`` requests (run concurrently).

    ``count_tokens`` should be the generator tokenizer's counter, so per-sample
    counts match what the generator recorded in the ledger. The ``"Step 1:"``
    prefix is part of the prompt, not the completion, so it isn't counted.
    """
    sizes = [min(batch_size, n_max - i) for i in range(0, n_max, batch_size)]
    batches = await asyncio.gather(
        *(generator.sample_solutions(problem.question, n=s, temperature=temperature) for s in sizes)
    )
    solutions = [s for batch in batches for s in batch]
    prefix = step_header(1)
    return SCSamples(
        solutions=solutions,
        answers=[extract_answer(s) for s in solutions],
        completion_tokens=[count_tokens(s.removeprefix(prefix)) for s in solutions],
        temperature=temperature,
    )


async def run_self_consistency(
    generator: Generator,
    problem: Problem,
    n: int,
    count_tokens: Callable[[str], int],
    temperature: float = DEFAULT_TEMPERATURE,
) -> BaselineOutput:
    """Single-N convenience wrapper. For sweeps, sample once and use ``curve``."""
    samples = await sample_self_consistency(generator, problem, n, count_tokens, temperature)
    return samples.output(n)
