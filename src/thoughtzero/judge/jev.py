"""JevJudge: request building, parsing, prior_mode, truncation (spec A4.2).

Talks to the local open-jev stand-in today (via JevClient(transport="local_stub")).
Every call goes cache -> budget -> client, and records to the ledger.
"""

import logging

from thoughtzero import accounting
from thoughtzero.judge.base import fill_missing_options, renormalize
from thoughtzero.judge.budget import BudgetGuard
from thoughtzero.judge.cache import CacheEntry, DiskCache, cache_key
from thoughtzero.judge.client import JevClient
from thoughtzero.llm.prompts import (
    CANDIDATE_SOUND_INSTRUCTION,
    FINAL_CORRECT_INSTRUCTION,
    NEXT_STEP_INSTRUCTION,
    PROMPT_VERSION,
    SOUND_INSTRUCTION,
    judge_state,
    judge_state_final,
)

logger = logging.getLogger(__name__)

MAX_STATE_CHARS = 28_000 * 4  # ~28K tokens at chars/4


class MissingDistributionError(Exception):
    """Raised when a `choice` response only reports the top pick, not a
    full distribution. Caller should switch prior_mode to per_candidate_noul.
    """


def _truncate_state(problem: str, steps: list[str]) -> tuple[list[str], bool]:
    """Keep problem + step 1 + the last N steps that fit; replace the
    middle with a placeholder. Returns (possibly-truncated steps, truncated?).
    """
    full = judge_state(problem, steps)
    if len(full) <= MAX_STATE_CHARS or len(steps) <= 2:
        return steps, False

    kept_tail: list[str] = []
    budget = MAX_STATE_CHARS - len(problem) - len(steps[0]) - 200
    for s in reversed(steps[1:]):
        if budget - len(s) < 0:
            break
        kept_tail.insert(0, s)
        budget -= len(s)

    omitted = len(steps) - 1 - len(kept_tail)
    if omitted <= 0:
        return steps, False
    truncated = [steps[0], f"[... {omitted} steps omitted ...]", *kept_tail]
    logger.warning("judge state truncated: omitted %d of %d steps", omitted, len(steps))
    return truncated, True


class JevJudge:
    def __init__(
        self,
        client: JevClient | None = None,
        cache: DiskCache | None = None,
        budget: BudgetGuard | None = None,
        prior_mode: str = "choice",
        root_value: float = 0.5,
        model_name: str = "open-jev-deberta-v3-large",
    ) -> None:
        self.client = client or JevClient()
        self.cache = cache or DiskCache()
        self.budget = budget
        self.prior_mode = prior_mode
        self.root_value = root_value
        self.model_name = model_name
        self.truncation_count = 0

    async def _ask(self, state: str, questions: list[dict]) -> list[dict]:
        key = cache_key(state, questions, self.model_name, PROMPT_VERSION)
        cached = self.cache.get(key)
        if cached is not None:
            accounting.record_jev(cached.input_tokens, 0.0, cache_hit=True)
            return cached.raw_response

        estimate = 0.0
        if self.budget is not None:
            chars = len(state) + sum(len(str(q)) for q in questions)
            estimate = await self.budget.check(max(1, chars // 4))

        response = await self.client.ask(state, questions)

        usd_spent = 0.0
        if self.budget is not None:
            usd_spent = await self.budget.record(estimate, response.input_tokens)

        self.cache.set(
            key,
            CacheEntry(
                raw_response=response.answers,
                parsed=None,
                input_tokens=response.input_tokens,
                latency_ms=response.latency_ms,
                created_at=0.0,
            ),
        )
        accounting.record_jev(response.input_tokens, usd_spent, cache_hit=False)
        return response.answers

    async def prior_and_value(
        self, problem: str, steps: list[str], candidates: list[str]
    ) -> tuple[list[float], float]:
        steps, truncated = _truncate_state(problem, steps)
        if truncated:
            self.truncation_count += 1
        state = judge_state(problem, steps)

        if not steps:
            value = self.root_value
            questions: list[dict] = []
        else:
            questions = [{"id": "sound", "type": "noul", "instructions": SOUND_INSTRUCTION}]

        option_keys = [f"c{i}" for i in range(len(candidates))]

        if self.prior_mode == "choice":
            questions.append(
                {
                    "id": "next",
                    "type": "choice",
                    "instructions": NEXT_STEP_INSTRUCTION,
                    "options": dict(zip(option_keys, candidates)),
                }
            )
            answers = await self._ask(state, questions)
            priors = self._parse_choice(answers, option_keys)
        elif self.prior_mode == "per_candidate_noul":
            for i, cand in enumerate(candidates):
                questions.append(
                    {
                        "id": f"cand_{i}",
                        "type": "noul",
                        "instructions": CANDIDATE_SOUND_INSTRUCTION.format(candidate=cand),
                    }
                )
            answers = await self._ask(state, questions)
            priors = self._parse_per_candidate_noul(answers, option_keys, offset=1 if steps else 0)
        else:
            raise ValueError(f"unknown prior_mode: {self.prior_mode!r}")

        if steps:
            value = self._parse_sound(answers[0])

        return [priors[k] for k in option_keys], value

    async def prior_only(
        self, problem: str, steps: list[str], candidates: list[str]
    ) -> list[float]:
        """Like prior_and_value, but never asks `sound` — for HybridJudge's
        prior-only ablation, so Jev isn't charged for a value it won't use.
        """
        steps, truncated = _truncate_state(problem, steps)
        if truncated:
            self.truncation_count += 1
        state = judge_state(problem, steps)
        option_keys = [f"c{i}" for i in range(len(candidates))]

        if self.prior_mode == "choice":
            questions = [
                {
                    "id": "next",
                    "type": "choice",
                    "instructions": NEXT_STEP_INSTRUCTION,
                    "options": dict(zip(option_keys, candidates)),
                }
            ]
            answers = await self._ask(state, questions)
            priors = self._parse_choice(answers, option_keys)
        elif self.prior_mode == "per_candidate_noul":
            questions = [
                {
                    "id": f"cand_{i}",
                    "type": "noul",
                    "instructions": CANDIDATE_SOUND_INSTRUCTION.format(candidate=cand),
                }
                for i, cand in enumerate(candidates)
            ]
            answers = await self._ask(state, questions)
            priors = self._parse_per_candidate_noul(answers, option_keys, offset=0)
        else:
            raise ValueError(f"unknown prior_mode: {self.prior_mode!r}")

        return [priors[k] for k in option_keys]

    async def value_only(self, problem: str, steps: list[str]) -> float:
        """Alias for step_sound: HybridJudge's value-only path has no
        candidates either way, so there's nothing extra to skip here."""
        return await self.step_sound(problem, steps)

    async def step_sound(self, problem: str, steps: list[str]) -> float:
        if not steps:
            return self.root_value
        steps, truncated = _truncate_state(problem, steps)
        if truncated:
            self.truncation_count += 1
        state = judge_state(problem, steps)
        answers = await self._ask(
            state, [{"id": "sound", "type": "noul", "instructions": SOUND_INSTRUCTION}]
        )
        return self._parse_sound(answers[0])

    async def final_correct(self, problem: str, steps: list[str]) -> float:
        steps, truncated = _truncate_state(problem, steps)
        if truncated:
            self.truncation_count += 1
        state = judge_state_final(problem, steps)
        answers = await self._ask(
            state, [{"id": "final", "type": "noul", "instructions": FINAL_CORRECT_INSTRUCTION}]
        )
        return self._parse_sound(answers[0])

    @staticmethod
    def _parse_sound(answer: dict) -> float:
        if "noul" not in answer:
            raise ValueError(f"expected a noul answer, got: {answer}")
        return float(answer["noul"])

    @staticmethod
    def _parse_choice(answers: list[dict], option_keys: list[str]) -> dict[str, float]:
        choice_answer = next((a for a in answers if "probabilities" in a or "choice" in a), None)
        if choice_answer is None:
            raise ValueError(f"no choice answer found in {answers}")
        probs = choice_answer.get("probabilities")
        if probs is None:
            raise MissingDistributionError(
                "response only reports the top pick, not a full distribution; "
                "switch judge.prior_mode to 'per_candidate_noul'"
            )
        return fill_missing_options(probs, option_keys)

    @staticmethod
    def _parse_per_candidate_noul(
        answers: list[dict], option_keys: list[str], offset: int
    ) -> dict[str, float]:
        cand_answers = answers[offset:]
        probs = {k: float(a.get("noul", 0.0)) for k, a in zip(option_keys, cand_answers)}
        return (
            renormalize(probs) if sum(probs.values()) > 0 else fill_missing_options({}, option_keys)
        )
