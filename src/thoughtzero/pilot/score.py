"""Pilot step 4-5 (spec §7.4): judge scores per prefix, one JSONL per judge (and variant).

Judges only ever receive the question text and the prefix steps; the label and
gold answer stay in this module's caller. All Jev calls go through Person 2's
cache, so re-running a stage after a crash costs nothing for finished calls.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from thoughtzero.accounting import ledger_scope
from thoughtzero.eval.runner import in_shard
from thoughtzero.pilot.common import StageStats, load_rows, resumable_map
from thoughtzero.pilot.traces import prefix_steps
from thoughtzero.types import Judge, Problem


def scores_path(out_dir: str | Path, judge_name: str, variant: str | None = None) -> Path:
    stem = f"scores_{judge_name}" + (f"__{variant}" if variant else "")
    return Path(out_dir) / f"{stem}.jsonl"


def sensitivity_subset(problem_ids: Sequence[str], n: int = 50, seed: int = 0) -> list[str]:
    """Deterministic n-problem subset of the pilot for prompt-sensitivity scoring."""
    ids = sorted(set(problem_ids))
    if len(ids) <= n:
        return ids
    rng = np.random.default_rng(seed)
    return sorted(ids[i] for i in rng.choice(len(ids), size=n, replace=False))


async def score_prefixes(
    prefixes: Sequence[dict[str, Any]],
    traces_by_id: dict[str, dict[str, Any]],
    problems_by_id: dict[str, Problem],
    judge: Judge,
    judge_name: str,
    out_dir: str | Path,
    *,
    variant: str | None = None,
    concurrency: int = 8,
    shard: tuple[int, int] | None = None,
) -> StageStats:
    async def work(px: dict[str, Any]) -> dict[str, Any]:
        q = problems_by_id[px["problem_id"]].question
        steps = prefix_steps(px, traces_by_id)
        with ledger_scope() as led:
            t0 = time.perf_counter()
            s = float(await judge.step_sound(q, steps))
            lat = time.perf_counter() - t0
        if not 0.0 <= s <= 1.0:
            raise ValueError(f"judge returned {s}, outside [0, 1]")
        return {
            "prefix_id": px["prefix_id"],
            "problem_id": px["problem_id"],
            "depth": px["depth"],
            "judge": judge_name,
            "variant": variant,
            "score": s,
            "latency_s": lat,
            "jev_calls": led.jev_calls,
            "jev_cache_hits": led.jev_cache_hits,
            "jev_usd": led.jev_usd,
            "jev_input_tokens": led.jev_input_tokens,
            "gemma_completion_tokens": led.gemma_completion_tokens,
        }

    mine = [px for px in prefixes if in_shard(px["problem_id"], shard)]
    return await resumable_map(
        mine,
        lambda px: px["prefix_id"],
        work,
        scores_path(out_dir, judge_name, variant),
        key_field="prefix_id",
        concurrency=concurrency,
    )


def load_scores(
    out_dir: str | Path, judge_name: str, variant: str | None = None
) -> list[dict[str, Any]]:
    rows = load_rows(scores_path(out_dir, judge_name, variant), "prefix_id")
    return [r for r in rows if not r.get("error")]
