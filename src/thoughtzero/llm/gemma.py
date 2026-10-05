"""Gemma generator over an OpenAI-compatible completions endpoint (spec §5.2, team file §A4.4).

Uses the raw **completions** endpoint, not chat: the generator continues a
partially written assistant turn, so the chat template is applied client-side
(``prompts.generator_prompt``). Works with SGLang, vLLM or Ollama.

Token accounting: completion tokens are the paper's primary compute axis.
With ``n > 1`` a server's usage block is aggregated across choices, so each
choice's completion tokens are counted with the HF tokenizer, and the prompt
is counted once per request. Every request calls ``accounting.record_gemma``.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import openai
from openai import AsyncOpenAI
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from thoughtzero import accounting
from thoughtzero.config import Config, GeneratorCfg
from thoughtzero.data.grading import has_final_answer
from thoughtzero.llm.prompts import (
    generator_prompt,
    split_steps,
    step_header,
    truncate_at_next_step,
)
from thoughtzero.llm.tokenize import ChatTokenizer, HFChatTokenizer
from thoughtzero.types import GenOut

log = logging.getLogger(__name__)

# Markup that would mean thinking mode leaked into the output (spec §5.2).
# Gemma 4 wraps thoughts in "<|channel>thought ... <channel|>" and turns
# thinking on with "<|think|>" (from its chat template; docs/verified_apis.md).
# "<think>" covers other model families served by mistake.
THINKING_MARKERS: tuple[str, ...] = ("<|channel>", "<channel|>", "<|think|>", "<think>")


@dataclass
class _Sampled:
    texts: list[str]  # raw choice texts, in choice-index order
    completion_tokens: list[int]
    prompt_tokens: int


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(
        exc,
        openai.APIConnectionError  # includes APITimeoutError
        | openai.RateLimitError
        | openai.InternalServerError,
    ):
        return True
    return isinstance(exc, openai.APIStatusError) and exc.status_code >= 500


def _derive_seed(base: int, prompt: str, attempt: int) -> int:
    """A per-request seed that depends only on the prompt, not on call order.

    Async simulations run in nondeterministic order, so a counter-based seed
    would not reproduce. Hashing the prompt does. ``attempt`` changes the seed
    when a request is retried because it produced nothing.
    """
    h = hashlib.sha256(f"{base}:{attempt}:{prompt}".encode()).digest()
    return int.from_bytes(h[:4], "big") & 0x7FFFFFFF


class OpenAICompatibleGenerator:
    """Implements the ``Generator`` protocol (``thoughtzero.types``).

    ``cfg`` is the ``generator`` config section. Settings that ``GeneratorCfg``
    doesn't have (yet) are keyword arguments; :meth:`from_config` fills them
    from the full ``Config`` (``seed``, ``search.max_depth``).

    - ``tokenizer``: defaults to ``HFChatTokenizer(tokenizer_name or cfg.tokenizer or cfg.model)``.
      Pass ``tokenizer_name`` when ``cfg.model`` isn't an HF ID (e.g. an Ollama
      tag) or is a quantized checkpoint.
    - ``seed``: base for per-request seeds (None disables seeding).
    - ``max_depth``: full solutions are capped at this many steps (spec §6.2).
    - ``use_system_role``: Gemma 4's template has a system turn (verified).
    - ``solution_stop``: stop strings for full solutions; by default they end at EOS.
    - The API key comes from the environment (``GEMMA_API_KEY``), never the config.
    """

    def __init__(
        self,
        cfg: GeneratorCfg,
        tokenizer: ChatTokenizer | None = None,
        client: AsyncOpenAI | None = None,
        *,
        tokenizer_name: str | None = None,
        seed: int | None = 0,
        max_depth: int = 20,
        use_system_role: bool = True,
        solution_stop: Sequence[str] = (),
        max_attempts: int = 5,
        request_timeout_s: float = 300.0,
    ) -> None:
        self.cfg = cfg
        self.tokenizer: ChatTokenizer = tokenizer or HFChatTokenizer(
            tokenizer_name or cfg.tokenizer or cfg.model
        )
        self.seed = seed
        self.max_depth = max_depth
        self.use_system_role = use_system_role
        self.solution_stop = list(solution_stop)
        self.max_attempts = max_attempts
        self.client = client or AsyncOpenAI(
            base_url=cfg.base_url,
            api_key=os.environ.get("GEMMA_API_KEY") or "EMPTY",
            max_retries=0,  # tenacity handles retries
            timeout=request_timeout_s,
        )
        self._sem = asyncio.Semaphore(cfg.max_concurrency)

    @classmethod
    def from_config(
        cls,
        config: Config,
        tokenizer: ChatTokenizer | None = None,
        **kwargs: Any,
    ) -> OpenAICompatibleGenerator:
        """Build from the full config: ``generator`` section, ``seed``, ``search.max_depth``."""
        kwargs.setdefault("seed", config.seed)
        kwargs.setdefault("max_depth", config.search.max_depth)
        return cls(config.generator, tokenizer, **kwargs)

    # ------------------------------------------------------------------ helpers

    def count_tokens(self, text: str) -> int:
        return self.tokenizer.count_tokens(text)

    def _prompt(self, problem: str, steps: Sequence[str]) -> str:
        return generator_prompt(problem, steps, self.tokenizer, self.use_system_role)

    async def _create(self, **kwargs: Any) -> Any:
        async with self._sem:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self.max_attempts),
                wait=wait_exponential(multiplier=0.5, max=20),
                retry=retry_if_exception(_is_retryable),
                reraise=True,
            ):
                with attempt:
                    return await self.client.completions.create(**kwargs)
        raise AssertionError("unreachable")  # pragma: no cover

    async def _sample(
        self,
        prompt: str,
        n: int,
        temperature: float,
        max_tokens: int,
        stop: list[str],
        attempt: int = 0,
    ) -> _Sampled:
        kwargs: dict[str, Any] = {
            "model": self.cfg.model,
            "prompt": prompt,
            "n": n,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if temperature > 0:
            kwargs["top_p"] = self.cfg.top_p
        if stop:
            kwargs["stop"] = stop
        if self.seed is not None:
            kwargs["seed"] = _derive_seed(self.seed, prompt, attempt)

        resp = await self._create(**kwargs)
        choices = sorted(resp.choices, key=lambda c: c.index)
        texts = [c.text or "" for c in choices]
        completion_tokens = [self.tokenizer.count_tokens(t) for t in texts]
        prompt_tokens = self.tokenizer.count_tokens(prompt)
        accounting.record_gemma(prompt_tokens, sum(completion_tokens))

        for t in texts:
            if any(mark in t for mark in THINKING_MARKERS):
                log.warning("Thinking-mode markup in generator output; check the chat template")
                break
        return _Sampled(texts, completion_tokens, prompt_tokens)

    def _new_steps(self, n_existing: int, continuation: str) -> list[str]:
        """Steps in a continuation of ``Step {n+1}:``, cut after the first final answer.

        Empty steps are dropped. Capped so the full solution never exceeds
        ``max_depth`` steps.
        """
        first = n_existing + 1
        new = split_steps(step_header(first) + continuation, first_number=first)
        new = [s for s in new if s]
        for i, s in enumerate(new):
            if has_final_answer(s):
                new = new[: i + 1]
                break
        return new[: max(0, self.max_depth - n_existing)]

    # ---------------------------------------------------------------- protocol

    async def propose(self, problem: str, steps: list[str], k: int) -> list[GenOut]:
        """K candidate next steps, from one ``n=k`` request.

        Empty outputs are dropped; if all are empty, retries once (with a new
        seed) and returns whatever that gives.
        """
        prompt = self._prompt(problem, steps)
        outs: list[GenOut] = []
        for attempt in range(2):
            s = await self._sample(
                prompt,
                n=k,
                temperature=self.cfg.temperature,
                max_tokens=self.cfg.max_step_tokens,
                stop=self.cfg.stop,
                attempt=attempt,
            )
            outs = [
                GenOut(truncate_at_next_step(t), s.prompt_tokens, c)
                for t, c in zip(s.texts, s.completion_tokens, strict=True)
            ]
            outs = [o for o in outs if o.text]
            if outs:
                return outs
            log.warning("All %d proposals empty at depth %d (attempt %d)", k, len(steps), attempt)
        return outs

    async def complete(self, problem: str, steps: list[str], temperature: float = 0.0) -> list[str]:
        """Continue to the end of the solution in one request; returns only the NEW steps."""
        return (await self.sample_completions(problem, steps, n=1, temperature=temperature))[0]

    async def sample_completions(
        self, problem: str, steps: list[str], n: int, temperature: float
    ) -> list[list[str]]:
        """``n`` independent completions of the prefix in one request; NEW steps only."""
        if len(steps) >= self.max_depth:
            return [[] for _ in range(n)]
        s = await self._sample(
            self._prompt(problem, steps),
            n=n,
            temperature=temperature,
            max_tokens=self.cfg.max_solution_tokens,
            stop=self.solution_stop,
        )
        return [self._new_steps(len(steps), t) for t in s.texts]

    async def sample_solutions(self, problem: str, n: int, temperature: float) -> list[str]:
        """``n`` full raw solution texts, each starting with ``"Step 1:"``.

        Use ``prompts.split_steps`` on the result to get steps.
        """
        s = await self._sample(
            self._prompt(problem, []),
            n=n,
            temperature=temperature,
            max_tokens=self.cfg.max_solution_tokens,
            stop=self.solution_stop,
        )
        return [step_header(1) + t for t in s.texts]

    # ----------------------------------------------------------- for the judge

    async def raw_completion(
        self,
        prompt: str,
        max_tokens: int = 1,
        logprobs: int | None = None,
        temperature: float = 0.0,
    ) -> dict[str, object]:
        """Raw completions call (with logprobs) for Person 2 (Jagriti)'s GemmaSelfJudge.

        ``prompt`` is sent as-is (no chat template). Returns::

            {"text": str,
             "top_logprobs": list[dict[str, float]],  # one {token: logprob} per output token;
                                                      # empty unless ``logprobs`` is set
             "prompt_tokens": int, "completion_tokens": int}

        The server caps ``logprobs`` (vLLM: ``--max-logprobs``, default 20).
        """
        kwargs: dict[str, Any] = {
            "model": self.cfg.model,
            "prompt": prompt,
            "n": 1,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if logprobs is not None:
            kwargs["logprobs"] = logprobs
        resp = await self._create(**kwargs)
        choice = resp.choices[0]
        text = choice.text or ""
        top: list[dict[str, float]] = []
        if choice.logprobs is not None and choice.logprobs.top_logprobs:
            top = [dict(d or {}) for d in choice.logprobs.top_logprobs]
        prompt_tokens = self.tokenizer.count_tokens(prompt)
        completion_tokens = (
            len(choice.logprobs.tokens)
            if choice.logprobs is not None and choice.logprobs.tokens
            else self.tokenizer.count_tokens(text)
        )
        accounting.record_gemma(prompt_tokens, completion_tokens)
        return {
            "text": text,
            "top_logprobs": top,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }
