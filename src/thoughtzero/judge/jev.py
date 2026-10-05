"""JevJudge: one request per expansion (SPEC.md §5.3). Owner: Person 2 (Jagriti).

Every request goes cache -> budget -> client and calls ``accounting.record_jev``.
"""

from __future__ import annotations

import hashlib
import logging

from thoughtzero import accounting
from thoughtzero.config import JudgeCfg
from thoughtzero.judge import budget as budget_module
from thoughtzero.judge.base import renormalize
from thoughtzero.judge.cache import DiskCache, cache_key
from thoughtzero.judge.client import JevClient
from thoughtzero.llm.prompts import (
    CANDIDATE_INSTRUCTION_TEMPLATE,
    FINAL_INSTRUCTION,
    NEXT_INSTRUCTION,
    PROMPT_VERSION,
    SOUND_VARIANTS,
    judge_state,
    judge_state_final,
)

logger = logging.getLogger(__name__)


class MissingDistributionError(Exception):
    """A ``choice`` response reported only the top pick, not a full
    distribution. Switch ``judge.prior_mode`` to ``per_candidate_noul``."""


def _truncate_steps(
    problem: str, steps: list[str], max_state_tokens: int
) -> tuple[list[str], bool]:
    """Keep the problem, step 1, and as many trailing steps as fit; replace
    the omitted middle with a placeholder (SPEC.md §5.3). Tokens estimated
    as chars/4.
    """
    max_chars = max_state_tokens * 4
    if len(judge_state(problem, steps)) <= max_chars or len(steps) <= 2:
        return steps, False

    kept_tail: list[str] = []
    budget_chars = max_chars - len(problem) - len(steps[0]) - 200
    for s in reversed(steps[1:]):
        if budget_chars - len(s) < 0:
            break
        kept_tail.insert(0, s)
        budget_chars -= len(s)

    omitted = len(steps) - 1 - len(kept_tail)
    if omitted <= 0:
        return steps, False
    truncated = [steps[0], f"[... {omitted} steps omitted ...]", *kept_tail]
    logger.warning("judge state truncated: omitted %d of %d steps", omitted, len(steps))
    return truncated, True


def _shuffle_permutation(state: str, n: int) -> list[int]:
    """A deterministic permutation of range(n), seeded by sha256(state)."""
    digest = hashlib.sha256(state.encode("utf-8")).hexdigest()
    seed = int(digest[:16], 16)
    perm = list(range(n))
    # Fisher-Yates with a simple LCG seeded from the digest, so this has no
    # dependency on `random`'s state or Python's salted `hash()`.
    rng_state = seed
    for i in range(n - 1, 0, -1):
        rng_state = (rng_state * 6364136223846793005 + 1) & ((1 << 64) - 1)
        j = rng_state % (i + 1)
        perm[i], perm[j] = perm[j], perm[i]
    return perm


class JevJudge:
    def __init__(self, cfg: JudgeCfg) -> None:
        self.cfg = cfg
        # IndexError (not caught by build_judge) if out of range: a config bug, fail loudly
        self.sound_instruction = SOUND_VARIANTS[cfg.sound_variant]
        self.client = JevClient(cfg)
        self.cache = DiskCache()
        self.truncation_count = 0
        self._last_shuffle_seed = ""

    async def _ask(self, state: str, questions: dict[str, dict]) -> dict[str, dict]:
        key = cache_key(state, questions, self.cfg.jev_model, PROMPT_VERSION)
        cached = self.cache.get(key)
        if cached is not None:
            accounting.record_jev(cached["input_tokens"], 0.0, cache_hit=True)
            return cached["answers"]

        chars = len(state) + sum(len(str(q)) for q in questions.values())
        estimated_tokens = max(1, chars // 4)

        if budget_module.current_guard is not None:
            budget_module.current_guard.check(estimated_tokens)

        try:
            answers = await self.client.ask(state, questions)
        except Exception:
            # release the reservation on failure (no spend), or it leaks
            # and eventually causes false "budget exceeded" errors.
            if budget_module.current_guard is not None:
                budget_module.current_guard.record(0, estimated_tokens)
            raise

        if budget_module.current_guard is not None:
            budget_module.current_guard.record(estimated_tokens, estimated_tokens)

        self.cache.set(key, {"answers": answers, "input_tokens": estimated_tokens})
        accounting.record_jev(estimated_tokens, self._usd(estimated_tokens), cache_hit=False)
        return answers

    def _usd(self, tokens: int) -> float:
        return tokens * self.cfg.usd_per_mtok / 1e6

    def _option_keys(self, n: int) -> list[str]:
        return [f"c{i}" for i in range(n)]

    def _build_prior_questions(self, candidates: list[str]) -> dict[str, dict]:
        option_keys = self._option_keys(len(candidates))
        order = (
            _shuffle_permutation(" ".join(candidates), len(candidates))
            if self.cfg.shuffle_options
            else list(range(len(candidates)))
        )

        if self.cfg.prior_mode == "choice":
            options = {option_keys[pos]: candidates[orig] for pos, orig in enumerate(order)}
            return {
                "next": {"type": "choice", "instructions": NEXT_INSTRUCTION, "options": options}
            }
        if self.cfg.prior_mode == "per_candidate_noul":
            return {
                f"cand_{pos}": {
                    "type": "noul",
                    "instructions": CANDIDATE_INSTRUCTION_TEMPLATE.format(
                        candidate=candidates[orig]
                    ),
                }
                for pos, orig in enumerate(order)
            }
        raise ValueError(f"unknown prior_mode: {self.cfg.prior_mode!r}")

    def _parse_priors(self, answers: dict[str, dict], n: int) -> list[float]:
        option_keys = self._option_keys(n)

        if self.cfg.prior_mode == "choice":
            next_answer = answers["next"]
            probs = next_answer.get("probabilities")
            if probs is None:
                raise MissingDistributionError(
                    "response only reports the top pick, not a full distribution; "
                    "switch judge.prior_mode to 'per_candidate_noul'"
                )
            by_position = [probs.get(k) for k in option_keys]
        else:
            by_position = [answers[f"cand_{i}"].get("noul") for i in range(n)]

        priors_by_position = renormalize(by_position)

        if self.cfg.shuffle_options:
            perm = _shuffle_permutation(self._last_shuffle_seed, n)
            priors_by_original_index = [0.0] * n
            for pos, orig in enumerate(perm):
                priors_by_original_index[orig] = priors_by_position[pos]
            return priors_by_original_index
        return priors_by_position

    async def prior_and_value(
        self, problem: str, steps: list[str], candidates: list[str]
    ) -> tuple[list[float], float]:
        steps, truncated = _truncate_steps(problem, steps, self.cfg.max_state_tokens)
        if truncated:
            self.truncation_count += 1

        if self.cfg.shuffle_options:
            self._last_shuffle_seed = " ".join(candidates)

        if not steps:
            questions = self._build_prior_questions(candidates)
            answers = await self._ask(judge_state(problem, steps), questions)
            return self._parse_priors(answers, len(candidates)), self.cfg.root_value

        state = judge_state(problem, steps)
        questions = {"sound": {"type": "noul", "instructions": self.sound_instruction}}
        questions.update(self._build_prior_questions(candidates))
        answers = await self._ask(state, questions)
        value = float(answers["sound"]["noul"])
        return self._parse_priors(answers, len(candidates)), value

    async def prior_only(
        self, problem: str, steps: list[str], candidates: list[str]
    ) -> list[float]:
        """Like prior_and_value, but never asks `sound` (HybridJudge's
        prior-only ablation, so Jev isn't charged for a discarded value)."""
        steps, truncated = _truncate_steps(problem, steps, self.cfg.max_state_tokens)
        if truncated:
            self.truncation_count += 1
        if self.cfg.shuffle_options:
            self._last_shuffle_seed = " ".join(candidates)
        state = judge_state(problem, steps)
        questions = self._build_prior_questions(candidates)
        answers = await self._ask(state, questions)
        return self._parse_priors(answers, len(candidates))

    async def value_only(self, problem: str, steps: list[str]) -> float:
        return await self.step_sound(problem, steps)

    async def step_sound(self, problem: str, steps: list[str]) -> float:
        if not steps:
            return self.cfg.root_value
        steps, truncated = _truncate_steps(problem, steps, self.cfg.max_state_tokens)
        if truncated:
            self.truncation_count += 1
        state = judge_state(problem, steps)
        answers = await self._ask(
            state, {"sound": {"type": "noul", "instructions": self.sound_instruction}}
        )
        return float(answers["sound"]["noul"])

    async def final_correct(self, problem: str, steps: list[str]) -> float:
        steps, truncated = _truncate_steps(problem, steps, self.cfg.max_state_tokens)
        if truncated:
            self.truncation_count += 1
        state = judge_state_final(problem, steps)
        answers = await self._ask(
            state, {"final": {"type": "noul", "instructions": FINAL_INSTRUCTION}}
        )
        return float(answers["final"]["noul"])
