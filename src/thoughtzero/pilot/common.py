"""Resumable, sharded, append-and-flush async map used by every pilot stage."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from thoughtzero.eval.runner import append_jsonl, read_jsonl
from thoughtzero.judge.budget import BudgetExceeded

log = logging.getLogger(__name__)
T = TypeVar("T")


def load_rows(path: str | Path, key: str) -> list[dict[str, Any]]:
    """Rows keyed by `key`: a success beats an error row; otherwise the last wins."""
    best: dict[str, dict[str, Any]] = {}
    for r in read_jsonl(Path(path)):
        old = best.get(r[key])
        if old is None or not (r.get("error") and not old.get("error")):
            best[r[key]] = r
    return list(best.values())


@dataclass
class StageStats:
    n_items: int
    n_skipped: int = 0
    n_run: int = 0
    n_errors: int = 0
    stopped_reason: str | None = None


async def resumable_map(
    items: Sequence[T],
    key: Callable[[T], str],
    work: Callable[[T], Awaitable[dict[str, Any]]],
    out_path: str | Path,
    *,
    key_field: str,
    concurrency: int = 8,
) -> StageStats:
    """Run `work(item)` for items whose key has no successful row yet.

    Each result row is appended and flushed immediately. An exception writes an
    error row `{key_field: ..., "error": ...}` and the stage continues;
    `BudgetExceeded` stops the stage (in-flight items finish).
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {r[key_field] for r in load_rows(out, key_field) if not r.get("error")}
    todo = [it for it in items if key(it) not in done]
    stats = StageStats(len(items), n_skipped=len(items) - len(todo))
    queue: asyncio.Queue[T] = asyncio.Queue()
    for it in todo:
        queue.put_nowait(it)
    stop = asyncio.Event()

    async def worker() -> None:
        while not stop.is_set():
            try:
                it = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            t0 = time.perf_counter()
            try:
                row = await work(it)
            except BudgetExceeded as e:
                stats.stopped_reason = stats.stopped_reason or f"budget_exceeded: {e}"
                stop.set()
                return
            except Exception as e:  # noqa: BLE001
                row = {
                    key_field: key(it),
                    "error": f"{type(e).__name__}: {e}",
                    "latency_s": time.perf_counter() - t0,
                }
                stats.n_errors += 1
            append_jsonl(out, row)
            stats.n_run += 1

    await asyncio.gather(*(worker() for _ in range(max(1, min(concurrency, len(todo))))))
    log.info("%s: %s", out.name, stats)
    return stats
