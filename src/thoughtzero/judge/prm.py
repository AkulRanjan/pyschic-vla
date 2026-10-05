"""PRMJudge: a process reward model scores every step (baseline B5, SPEC.md §5.3).

Uses ``Qwen/Qwen2.5-Math-PRM-7B`` (``judge.prm_model``) exactly as its model card shows
(https://huggingface.co/Qwen/Qwen2.5-Math-PRM-7B, checked 2026-10-05): the steps are the
assistant turn, joined and ended by the ``<extra_0>`` token; at each ``<extra_0>`` the model
outputs two-way logits, and softmax[:, 1] is that step's probability of being correct.

- value / ``step_sound``: the **minimum** step score ("are all steps so far correct?" fails
  if any step does); the root (no steps) gets ``judge.root_value``.
- ``final_correct``: the same, over the finished solution.
- priors: uniform (the PRM has none); pair it with another judge's priors through a hybrid
  judge (``prior_from: self``, ``value_from: prm``).

The model is loaded lazily, once per process, and needs the ``[prm]`` extra (torch) and a
GPU with ~16 GB for the bf16 weights. Forward passes run one at a time in a worker thread.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any

from thoughtzero.config import JudgeCfg

log = logging.getLogger(__name__)

STEP_SEPARATOR = "<extra_0>"
SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."

# (problem, steps) -> one probability-correct per step
Scorer = Callable[[str, list[str]], list[float]]


def prm_messages(problem: str, steps: list[str]) -> list[dict[str, str]]:
    """The chat the PRM scores (model card format)."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem.strip()},
        {
            "role": "assistant",
            "content": STEP_SEPARATOR.join(s.strip() for s in steps) + STEP_SEPARATOR,
        },
    ]


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


class _HFScorer:
    """Loads the PRM on first use and scores step lists (model card code, batch size 1)."""

    _cache: dict[str, Any] = {}

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def _load(self) -> tuple[Any, Any]:
        if self.model_name not in _HFScorer._cache:
            import torch
            from transformers import AutoModel, AutoTokenizer

            log.info("loading PRM %s (bf16)", self.model_name)
            tok = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            model = AutoModel.from_pretrained(
                self.model_name,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            ).eval()
            _HFScorer._cache[self.model_name] = (tok, model)
        return _HFScorer._cache[self.model_name]

    def __call__(self, problem: str, steps: list[str]) -> list[float]:
        import torch
        import torch.nn.functional as F

        tok, model = self._load()
        text = tok.apply_chat_template(
            prm_messages(problem, steps), tokenize=False, add_generation_prompt=False
        )
        input_ids = tok.encode(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            logits = model(input_ids=input_ids)[0]  # (1, seq_len, 2)
        sep_id = tok.encode(STEP_SEPARATOR)[0]
        mask = (input_ids == sep_id)[0]
        probs = F.softmax(logits[0][mask].float(), dim=-1)[:, 1]
        return [float(p) for p in probs.cpu().tolist()]


class PRMJudge:
    def __init__(self, cfg: JudgeCfg, scorer: Scorer | None = None) -> None:
        self.cfg = cfg
        self.scorer: Scorer = scorer or _HFScorer(cfg.prm_model)
        self._lock = asyncio.Lock()  # one forward pass at a time on the GPU

    async def _min_score(self, problem: str, steps: list[str]) -> float:
        if not steps:
            return self.cfg.root_value
        async with self._lock:
            scores = await asyncio.to_thread(self.scorer, problem, steps)
        if len(scores) != len(steps):
            raise ValueError(f"PRM returned {len(scores)} scores for {len(steps)} steps")
        return min(scores)

    async def prior_and_value(
        self, problem: str, steps: list[str], candidates: list[str]
    ) -> tuple[list[float], float]:
        value = await self._min_score(problem, steps)
        return [1.0 / len(candidates)] * len(candidates), value

    async def value_only(self, problem: str, steps: list[str]) -> float:
        return await self._min_score(problem, steps)

    async def step_sound(self, problem: str, steps: list[str]) -> float:
        return await self._min_score(problem, steps)

    async def final_correct(self, problem: str, steps: list[str]) -> float:
        return await self._min_score(problem, steps)
