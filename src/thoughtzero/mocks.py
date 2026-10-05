"""Deterministic offline mocks + a toy task where greedy-by-prior fails but search succeeds.

Owner: Person 1 (Prakhar). Everyone uses these for tests and for developing without a GPU or
API key. Keep them deterministic: randomness is derived from sha256 of the state, never from
Python's per-process salted ``hash()``.

Toy task: start at 0; each step adds 1, 2 or 5; reach exactly 13.
- The judge's prior always favours "+5" (misleading): greedy goes 5 -> 10 -> 15 and is wrong.
- Value is 0.9 while the running total is <= 13, else 0.1.
- ``final_correct`` is 0.95 if the total is exactly 13, else 0.05.
Search should find e.g. 5 -> 10 -> 12 -> 13.
"""

from __future__ import annotations

import asyncio
import hashlib
import random
import re
from collections import Counter

from thoughtzero.accounting import record_gemma, record_jev
from thoughtzero.types import GenOut, Problem

TARGET = 13
INCREMENTS = (5, 2, 1)
PRIOR_WEIGHTS = {5: 0.6, 2: 0.25, 1: 0.15}

TOY_QUESTION = (
    "Start at 0. Each step adds 1, 2 or 5 to the running total. "
    f"Reach exactly {TARGET} and put the final total in \\boxed{{}}."
)
TOY = Problem(id="toy/reach13", question=TOY_QUESTION, answer=str(TARGET), source="toy")

_TOTAL_RE = re.compile(r"->\s*(-?\d+)")
_INC_RE = re.compile(r"add\s+(\d+)", re.IGNORECASE)
_BOX_RE = re.compile(r"\\boxed\{([^{}]*)\}")


class MockFailure(RuntimeError):
    """Injected failure (``fail_rate``) for testing error paths."""


def running_total(steps: list[str]) -> int:
    for step in reversed(steps):
        m = _TOTAL_RE.search(step)
        if m:
            return int(m.group(1))
    return 0


def toy_extract_answer(text: str) -> str | None:
    found = _BOX_RE.findall(text)
    return found[-1].strip() if found else None


def toy_step(total: int, inc: int) -> str:
    new = total + inc
    if new >= TARGET:
        return f"add {inc} -> {new}. \\boxed{{{new}}}"
    return f"add {inc} -> {new}"


def _is_final(step: str) -> bool:
    return "\\boxed{" in step


def _stable_rng(*parts: object) -> random.Random:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def _approx_tokens(*texts: str) -> int:
    return max(1, sum(len(t) for t in texts) // 4)


class _Faulty:
    def __init__(self, seed: int, latency_s: float, fail_rate: float) -> None:
        self.seed = seed
        self.latency_s = latency_s
        self.fail_rate = fail_rate
        self.calls: Counter[str] = Counter()
        self._fail_rng = random.Random(seed)

    async def _tick(self, name: str) -> None:
        self.calls[name] += 1
        if self.latency_s > 0:
            await asyncio.sleep(self.latency_s)
        if self.fail_rate > 0 and self._fail_rng.random() < self.fail_rate:
            raise MockFailure(f"injected failure in {name}")


class MockGenerator(_Faulty):
    """Implements ``types.Generator`` for the toy task."""

    def __init__(self, seed: int = 0, latency_s: float = 0.0, fail_rate: float = 0.0) -> None:
        super().__init__(seed, latency_s, fail_rate)

    async def propose(self, problem: str, steps: list[str], k: int) -> list[GenOut]:
        await self._tick("propose")
        rng = _stable_rng(self.seed, "propose", problem, *steps)
        if k <= len(INCREMENTS):
            incs = rng.sample(INCREMENTS, k)
        else:
            incs = list(INCREMENTS) + rng.choices(INCREMENTS, k=k - len(INCREMENTS))
            rng.shuffle(incs)
        total = running_total(steps)
        prompt_tokens = _approx_tokens(problem, *steps)
        outs = [GenOut(toy_step(total, i), prompt_tokens, 8) for i in incs]
        record_gemma(prompt_tokens, 8 * len(outs))
        return outs

    def _rollout(self, steps: list[str], rng: random.Random, temperature: float) -> list[str]:
        total = running_total(steps)
        new_steps: list[str] = []
        while not (new_steps and _is_final(new_steps[-1])) and len(steps) + len(new_steps) < 50:
            inc = INCREMENTS[0] if temperature == 0 else rng.choice(INCREMENTS)
            new_steps.append(toy_step(total, inc))
            total += inc
        return new_steps

    async def complete(self, problem: str, steps: list[str], temperature: float = 0.0) -> list[str]:
        await self._tick("complete")
        rng = _stable_rng(self.seed, "complete", temperature, problem, *steps)
        new_steps = self._rollout(steps, rng, temperature)
        record_gemma(_approx_tokens(problem, *steps), 8 * len(new_steps))
        return new_steps

    async def sample_completions(
        self, problem: str, steps: list[str], n: int, temperature: float
    ) -> list[list[str]]:
        await self._tick("sample_completions")
        out = []
        for i in range(n):
            rng = _stable_rng(self.seed, "sample", i, temperature, problem, *steps)
            out.append(self._rollout(steps, rng, temperature))
        record_gemma(_approx_tokens(problem, *steps), 8 * sum(len(c) for c in out))
        return out

    async def sample_solutions(self, problem: str, n: int, temperature: float) -> list[str]:
        completions = await self.sample_completions(problem, [], n, temperature)
        return [
            "".join(f"Step {i}: {s}\n\n" for i, s in enumerate(steps, start=1)).rstrip()
            for steps in completions
        ]


class MockJudge(_Faulty):
    """Implements ``types.Judge`` for the toy task (misleading prior, informative value)."""

    def __init__(self, seed: int = 0, latency_s: float = 0.0, fail_rate: float = 0.0) -> None:
        super().__init__(seed, latency_s, fail_rate)

    @staticmethod
    def _record(problem: str, steps: list[str], extra: list[str] | None = None) -> None:
        tokens = _approx_tokens(problem, *steps, *(extra or []))
        record_jev(tokens, tokens * 0.042 / 1e6, cache_hit=False)

    @staticmethod
    def _value(steps: list[str]) -> float:
        return 0.9 if running_total(steps) <= TARGET else 0.1

    async def prior_and_value(
        self, problem: str, steps: list[str], candidates: list[str]
    ) -> tuple[list[float], float]:
        await self._tick("prior_and_value")
        weights = []
        for cand in candidates:
            m = _INC_RE.search(cand)
            weights.append(PRIOR_WEIGHTS.get(int(m.group(1)), 0.1) if m else 0.1)
        total = sum(weights)
        self._record(problem, steps, candidates)
        return [w / total for w in weights], self._value(steps)

    async def final_correct(self, problem: str, steps: list[str]) -> float:
        await self._tick("final_correct")
        self._record(problem, steps)
        return 0.95 if running_total(steps) == TARGET else 0.05

    async def step_sound(self, problem: str, steps: list[str]) -> float:
        await self._tick("step_sound")
        self._record(problem, steps)
        return self._value(steps)


async def greedy_by_prior(
    problem: str, generator: MockGenerator, judge: MockJudge, k: int = 3, max_depth: int = 20
) -> str | None:
    """Follow the highest-prior candidate at every step (what search must beat)."""
    steps: list[str] = []
    while len(steps) < max_depth:
        cands = await generator.propose(problem, steps, k)
        priors, _ = await judge.prior_and_value(problem, steps, [c.text for c in cands])
        best = cands[max(range(len(cands)), key=lambda i: priors[i])].text
        steps.append(best)
        if _is_final(best):
            return toy_extract_answer(best)
    return None
