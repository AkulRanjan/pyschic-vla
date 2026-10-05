"""Gemma generator over an OpenAI-compatible **chat** endpoint (hosted APIs, e.g. OpenRouter).

``OpenAICompatibleGenerator`` (``llm/gemma.py``) continues a partial assistant turn on the
raw completions endpoint, which needs a self-hosted server. Hosted APIs don't allow that:
checked on OpenRouter's Gemma 4 (2026-10-05), a raw prompt or an assistant prefill is
answered from "Step 1" again, ``n > 1`` returns one choice, and seeds aren't honoured
(docs/verified_apis.md, G11). So this generator:

- puts the steps so far in the user turn and asks for **only the next step** (``propose``)
  or **the rest of the solution** (``complete`` / ``sample_*``)
  (``prompts.generator_chat_messages``);
- makes ``k`` (or ``n``) parallel single-choice requests instead of one ``n=k`` request;
- strips the "Step n:" header the model writes, and tolerates a reply that restarts the
  numbering at "Step 1".

Token accounting uses the usage the API reports (falls back to the tokenizer). The tokenizer
is also the compute-axis counter the baselines use (``eval.methods.token_counter``).
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Sequence
from typing import Any

from openai import AsyncOpenAI
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_exponential

from thoughtzero import accounting
from thoughtzero.config import Config, GeneratorCfg
from thoughtzero.data.grading import has_final_answer
from thoughtzero.llm.gemma import THINKING_MARKERS, _derive_seed, _is_retryable
from thoughtzero.llm.prompts import generator_chat_messages, split_steps, step_header
from thoughtzero.llm.tokenize import ChatTokenizer, HFChatTokenizer
from thoughtzero.types import GenOut

log = logging.getLogger(__name__)


# Hosted endpoints whose own key variable is used when GEMMA_API_KEY isn't set.
HOSTED_KEYS = {
    "openrouter.ai": "OPENROUTER_API_KEY",
    # Gemini API's OpenAI-compatible endpoint (Gemma 4 26B-A4B / 31B; AI Studio key)
    "generativelanguage.googleapis.com": "GEMINI_API_KEY",
}


def api_key_for(base_url: str) -> str:
    """``GEMMA_API_KEY``, else the hosted endpoint's own key (``HOSTED_KEYS``)."""
    key = os.environ.get("GEMMA_API_KEY") or ""
    for host, env in HOSTED_KEYS.items():
        if not key and host in base_url:
            key = os.environ.get(env) or ""
    return key or "EMPTY"


def steps_from_reply(text: str, first_number: int) -> list[str]:
    """The steps in a reply that should start at ``Step {first_number}:``.

    If the model restarted the numbering at "Step 1" (it sometimes ignores the request),
    the steps before ``first_number`` are dropped. Text without any header is one step.
    """
    if first_number > 1 and text.lstrip().startswith(step_header(1)):  # restarted at Step 1
        steps = split_steps(text, first_number=1)[first_number - 1 :]
    else:
        steps = split_steps(text, first_number=first_number)
    return [s for s in steps if s]


class ChatGenerator:
    """Implements the ``Generator`` protocol (``thoughtzero.types``) over chat completions."""

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
        self.max_attempts = max_attempts
        self.client = client or AsyncOpenAI(
            base_url=cfg.base_url,
            api_key=api_key_for(cfg.base_url),
            max_retries=0,  # tenacity handles retries
            timeout=request_timeout_s,
        )
        self._sem = asyncio.Semaphore(cfg.max_concurrency)

    @classmethod
    def from_config(
        cls, config: Config, tokenizer: ChatTokenizer | None = None, **kwargs: Any
    ) -> ChatGenerator:
        kwargs.setdefault("seed", config.seed)
        kwargs.setdefault("max_depth", config.search.max_depth)
        return cls(config.generator, tokenizer, **kwargs)

    def count_tokens(self, text: str) -> int:
        return self.tokenizer.count_tokens(text)

    async def _chat(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        stop: Sequence[str],
        sample: int,
    ) -> str:
        """One single-choice request; records Gemma tokens; returns the reply text."""
        kwargs: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if temperature > 0:
            kwargs["top_p"] = self.cfg.top_p
        if stop:
            kwargs["stop"] = list(stop)
        if self.seed is not None:  # passed for reproducibility where the route honours it
            kwargs["seed"] = _derive_seed(self.seed, repr(messages), sample)
        async with self._sem:
            resp: Any = None
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self.max_attempts),
                wait=wait_exponential(multiplier=0.5, max=20),
                retry=retry_if_exception(_is_retryable),
                reraise=True,
            ):
                with attempt:
                    resp = await self.client.chat.completions.create(**kwargs)
        text = (resp.choices[0].message.content or "") if resp.choices else ""
        usage = resp.usage
        if usage is not None:
            accounting.record_gemma(usage.prompt_tokens, usage.completion_tokens)
        else:
            prompt_text = "\n".join(m["content"] for m in messages)
            accounting.record_gemma(self.count_tokens(prompt_text), self.count_tokens(text))
        if any(mark in text for mark in THINKING_MARKERS):
            log.warning("Thinking-mode markup in generator output; check the model / route")
        return text

    def _messages(self, problem: str, steps: Sequence[str], next_step_only: bool) -> Any:
        return generator_chat_messages(
            problem, steps, next_step_only=next_step_only, use_system_role=self.use_system_role
        )

    def _new_steps(self, n_existing: int, reply: str) -> list[str]:
        """Steps continuing ``n_existing`` ones, cut after the first final answer and capped
        at ``max_depth`` steps in total."""
        new = steps_from_reply(reply, n_existing + 1)
        for i, s in enumerate(new):
            if has_final_answer(s):
                new = new[: i + 1]
                break
        return new[: max(0, self.max_depth - n_existing)]

    # ---------------------------------------------------------------- protocol

    async def propose(self, problem: str, steps: list[str], k: int) -> list[GenOut]:
        """``k`` candidate next steps from ``k`` parallel requests; empty ones are dropped.

        If all ``k`` are empty, tries once more with new seeds.
        """
        messages = self._messages(problem, steps, next_step_only=True)
        first = len(steps) + 1
        outs: list[GenOut] = []
        for attempt in range(2):
            replies = await asyncio.gather(
                *(
                    self._chat(
                        messages,
                        self.cfg.temperature,
                        self.cfg.max_step_tokens,
                        self.cfg.stop,
                        sample=attempt * k + i,
                    )
                    for i in range(k)
                )
            )
            prompt_tokens = self.count_tokens("\n".join(m["content"] for m in messages))
            outs = []
            for reply in replies:
                parsed = steps_from_reply(reply, first)
                text = parsed[0] if parsed else ""
                if text:
                    outs.append(GenOut(text, prompt_tokens, self.count_tokens(reply)))
            if outs:
                return outs
            log.warning("All %d proposals empty at depth %d (attempt %d)", k, len(steps), attempt)
        return outs

    async def complete(self, problem: str, steps: list[str], temperature: float = 0.0) -> list[str]:
        """The rest of the solution in one request; returns only the NEW steps."""
        return (await self.sample_completions(problem, steps, n=1, temperature=temperature))[0]

    async def sample_completions(
        self, problem: str, steps: list[str], n: int, temperature: float
    ) -> list[list[str]]:
        """``n`` independent completions of the prefix (parallel requests); NEW steps only."""
        if len(steps) >= self.max_depth:
            return [[] for _ in range(n)]
        messages = self._messages(problem, steps, next_step_only=False)
        replies = await asyncio.gather(
            *(
                self._chat(messages, temperature, self.cfg.max_solution_tokens, (), sample=i)
                for i in range(n)
            )
        )
        return [self._new_steps(len(steps), r) for r in replies]

    async def sample_solutions(self, problem: str, n: int, temperature: float) -> list[str]:
        """``n`` full raw solution texts, each starting with ``"Step 1:"``."""
        messages = self._messages(problem, [], next_step_only=False)
        replies = await asyncio.gather(
            *(
                self._chat(messages, temperature, self.cfg.max_solution_tokens, (), sample=i)
                for i in range(n)
            )
        )
        header = step_header(1)
        return [r if r.lstrip().startswith(header) else f"{header} {r.lstrip()}" for r in replies]
