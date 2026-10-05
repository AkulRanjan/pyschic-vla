"""Pilot steps 1-2 (spec §7.2-7.3): one sampled trace per problem, then prefixes.

Trace row: {problem_id, source, level, subject, steps, final_answer, trace_correct, n_steps, text}
Prefix:    {prefix_id, problem_id, depth, rel_depth, n_steps, is_final}
           (its steps are trace.steps[:depth])
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from thoughtzero.accounting import ledger_scope
from thoughtzero.eval.runner import in_shard
from thoughtzero.eval.toolkit import Toolkit, real_toolkit
from thoughtzero.pilot.common import StageStats, load_rows, resumable_map
from thoughtzero.types import Generator, Problem

TRACES_FILE = "traces.jsonl"


async def generate_traces(
    problems: Sequence[Problem],
    generator: Generator,
    out_dir: str | Path,
    *,
    temperature: float = 0.7,
    concurrency: int = 8,
    shard: tuple[int, int] | None = None,
    tools: Toolkit | None = None,
) -> StageStats:
    """Sample one solution per problem (resumable; sharded by problem id)."""
    tk = tools or real_toolkit()

    async def work(p: Problem) -> dict[str, Any]:
        with ledger_scope() as led:
            (text,) = await generator.sample_solutions(p.question, n=1, temperature=temperature)
        steps = tk.split(text)
        ans = tk.extract(text)
        return {
            "problem_id": p.id,
            "source": p.source,
            "level": p.level,
            "subject": p.subject,
            "steps": steps,
            "n_steps": len(steps),
            "final_answer": ans,
            "trace_correct": bool(tk.grade(ans, p.answer)),
            "text": text,
            "gemma_completion_tokens": led.gemma_completion_tokens,
        }

    mine = [p for p in problems if in_shard(p.id, shard)]
    return await resumable_map(
        mine,
        lambda p: p.id,
        work,
        Path(out_dir) / TRACES_FILE,
        key_field="problem_id",
        concurrency=concurrency,
    )


def load_traces(out_dir: str | Path) -> list[dict[str, Any]]:
    rows = load_rows(Path(out_dir) / TRACES_FILE, "problem_id")
    return sorted((r for r in rows if not r.get("error")), key=lambda r: r["problem_id"])


def select_prefix_lengths(n_steps: int, cap: int = 8) -> list[int]:
    """Up to `cap` prefix lengths evenly spread over 1..n_steps, always including n_steps."""
    if n_steps <= 0:
        return []
    if n_steps <= cap:
        return list(range(1, n_steps + 1))
    raw = np.linspace(1, n_steps, cap).round().astype(int)
    return sorted({int(x) for x in raw})


def make_prefixes(traces: Sequence[dict[str, Any]], cap: int = 8) -> list[dict[str, Any]]:
    out = []
    for t in traces:
        n = t["n_steps"]
        for d in select_prefix_lengths(n, cap):
            out.append(
                {
                    "prefix_id": f"{t['problem_id']}#{d}",
                    "problem_id": t["problem_id"],
                    "depth": d,
                    "rel_depth": d / n,
                    "n_steps": n,
                    "is_final": d == n,
                }
            )
    return out


def prefix_steps(prefix: dict[str, Any], traces_by_id: dict[str, dict[str, Any]]) -> list[str]:
    return list(traces_by_id[prefix["problem_id"]]["steps"][: prefix["depth"]])


def trace_stats(traces: Sequence[dict[str, Any]]) -> dict[str, float]:
    if not traces:
        return {"n": 0}
    return {
        "n": len(traces),
        "accuracy": float(np.mean([t["trace_correct"] for t in traces])),
        "frac_no_answer": float(np.mean([t["final_answer"] is None for t in traces])),
        "mean_steps": float(np.mean([t["n_steps"] for t in traces])),
        "frac_zero_steps": float(np.mean([t["n_steps"] == 0 for t in traces])),
        "frac_one_step": float(np.mean([t["n_steps"] == 1 for t in traces])),
    }


def sanity_markdown(traces: Sequence[dict[str, Any]], n: int = 20, seed: int = 0) -> str:
    """Day-8 check: are steps sensible, do answers parse, is accuracy ~40-70%?"""
    st = trace_stats(traces)
    lines = ["# Trace sanity check", "", f"Stats: `{st}`", ""]
    if st.get("n") and not 0.4 <= st["accuracy"] <= 0.7:
        lines += [
            (
                f"**WARNING:** trace accuracy {st['accuracy']:.0%} is outside the expected "
                "40-70% for a 4B model on MATH. Tell the team."
            ),
            "",
        ]
    rng = np.random.default_rng(seed)
    k = min(n, len(traces))
    pick: list[int] = rng.choice(len(traces), size=k, replace=False).tolist() if traces else []
    for i in sorted(pick):
        t = traces[i]
        lines += [
            (
                f"## {t['problem_id']}: {t['n_steps']} steps, answer `{t['final_answer']}`, "
                f"correct={t['trace_correct']}"
            ),
            "",
        ]
        lines += [f"{k}. {s[:300]}" for k, s in enumerate(t["steps"], 1)] + [""]
    return "\n".join(lines)
