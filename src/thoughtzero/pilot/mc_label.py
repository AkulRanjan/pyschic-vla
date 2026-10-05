"""Pilot step 3 (spec §7.4): Monte Carlo soundness labels per prefix.

For each prefix, sample n completions at temperature T and grade each final
answer: soft_label = n_correct / n, hard_label = soft_label > 0.

Edge case: if the prefix already contains a final answer (always true for the
last prefix of a trace that boxed an answer), no sampling is done; the label is
just whether that answer is correct. A final prefix without any answer is
labelled 0 (the trace ended without answering).

This is the GPU-heavy step (~12,800 completions): run sharded and resumable.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from thoughtzero.accounting import ledger_scope
from thoughtzero.eval.runner import in_shard
from thoughtzero.eval.toolkit import Toolkit, real_toolkit
from thoughtzero.pilot.common import StageStats, load_rows, resumable_map
from thoughtzero.pilot.traces import prefix_steps
from thoughtzero.types import Generator, Problem

LABELS_FILE = "labels.jsonl"


def answer_in(steps: Sequence[str], tools: Toolkit | None = None) -> str | None:
    """The final answer already stated in ``steps`` (None if there is none)."""
    tk = tools or real_toolkit()
    return tk.extract(tk.format(list(steps))) if steps else None


def label_from_completions(
    prefix: Sequence[str],
    completions: Sequence[Sequence[str]],
    gold: str,
    tools: Toolkit | None = None,
) -> dict[str, Any]:
    """Grade completions (new steps only, per contract) continuing ``prefix``."""
    tk = tools or real_toolkit()
    answers = [answer_in(list(prefix) + list(c), tk) for c in completions]
    n = len(answers)
    k = sum(bool(tk.grade(a, gold)) for a in answers)
    soft = k / n if n else 0.0
    return {
        "n_samples": n,
        "n_correct": k,
        "soft_label": soft,
        "hard_label": soft > 0,
        "label_source": "mc",
        "answers": answers,
    }


def label_terminal(answer: str | None, gold: str, tools: Toolkit | None = None) -> dict[str, Any]:
    ok = bool((tools or real_toolkit()).grade(answer, gold))
    return {
        "n_samples": 0,
        "n_correct": int(ok),
        "soft_label": float(ok),
        "hard_label": ok,
        "label_source": "terminal",
        "answers": [answer],
    }


async def label_prefixes(
    prefixes: Sequence[dict[str, Any]],
    traces_by_id: dict[str, dict[str, Any]],
    problems_by_id: dict[str, Problem],
    generator: Generator,
    out_dir: str | Path,
    *,
    n_samples: int = 8,
    temperature: float = 0.7,
    concurrency: int = 8,
    shard: tuple[int, int] | None = None,
    tools: Toolkit | None = None,
) -> StageStats:
    tk = tools or real_toolkit()

    async def work(px: dict[str, Any]) -> dict[str, Any]:
        p = problems_by_id[px["problem_id"]]
        steps = prefix_steps(px, traces_by_id)
        with ledger_scope() as led:
            ans = answer_in(steps, tk)
            if ans is not None or px["is_final"]:
                lab = label_terminal(ans, p.answer, tk)
            else:
                comps = await generator.sample_completions(
                    p.question, steps, n=n_samples, temperature=temperature
                )
                lab = label_from_completions(steps, comps, p.answer, tk)
        return {
            **px,
            **lab,
            "gemma_completion_tokens": led.gemma_completion_tokens,
            "wall_time_s": led.wall_time_s,
        }

    mine = [px for px in prefixes if in_shard(px["problem_id"], shard)]
    return await resumable_map(
        mine,
        lambda px: px["prefix_id"],
        work,
        Path(out_dir) / LABELS_FILE,
        key_field="prefix_id",
        concurrency=concurrency,
    )


def load_labels(out_dir: str | Path) -> list[dict[str, Any]]:
    return [r for r in load_rows(Path(out_dir) / LABELS_FILE, "prefix_id") if not r.get("error")]
