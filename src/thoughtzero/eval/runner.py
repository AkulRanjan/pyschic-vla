"""Runs any Method over a dataset and writes JSONL (SPEC.md §8.5, §13.7). Owner: Person 4 (Harjas).

Layout of a run directory `results/<run_id>/`:
    config.yaml        resolved config (written at start; no secrets live in the config)
    git_commit.txt     `git rev-parse HEAD`, "-dirty" if uncommitted changes
    per_problem.jsonl  one line per (problem, method), appended + flushed as each finishes
    summary.csv        per-method aggregates, recomputed from the JSONL at the end

Guarantees:
- Methods never see the ground truth: `solve()` receives a copy of the Problem
  with `answer=""`. Grading happens here, after `solve()` returns.
- Resume: (problem_id, method_id) pairs already in the JSONL are skipped.
- Error isolation: an exception writes a flagged error row and the run goes on.
  `BudgetExceeded` is the exception: it stops the whole run cleanly.
- Every problem runs in its own `ledger_scope()`, so ledgers never mix.
"""

from __future__ import annotations

import asyncio
import csv
import dataclasses
import datetime as _dt
import hashlib
import json
import logging
import shutil
import subprocess
import traceback
import warnings
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from thoughtzero.accounting import Ledger, ledger_scope
from thoughtzero.config import Config, config_hash, dump_resolved
from thoughtzero.data.grading import is_equivalent
from thoughtzero.eval.metrics import accuracy, bootstrap_ci
from thoughtzero.judge.budget import BudgetExceeded
from thoughtzero.llm.prompts import PROMPT_VERSION
from thoughtzero.types import Problem

log = logging.getLogger(__name__)

ROWS_FILE = "per_problem.jsonl"
SUMMARY_FILE = "summary.csv"
BOOTSTRAP_N = 1000  # SPEC.md §8.3

# Settings that change how a run executes but not what it measures. They are reset before
# hashing for `protocol_hash`, so shards / resumed sessions with e.g. a different concurrency
# can still be merged.
_RUN_ONLY_EVAL_FIELDS = ("concurrency", "run_id", "out_dir")


# --------------------------------------------------------------------------- method protocol


@dataclass
class MethodResult:
    answer: str | None
    raw: dict[str, Any] = field(default_factory=dict)  # method-specific extras (tree, votes, ...)


@runtime_checkable
class Method(Protocol):
    name: str

    async def solve(self, problem: Problem) -> MethodResult: ...


def method_params(method: Method) -> dict[str, Any]:
    """Optional compute-level parameters a method exposes (e.g. n_simulations, n)."""
    return dict(getattr(method, "params", {}) or {})


def method_id(method: Method) -> str:
    """Unique key per method *and* compute level, e.g. 'tz[n_simulations=16]'."""
    p = method_params(method)
    if not p:
        return method.name
    return f"{method.name}[{','.join(f'{k}={v}' for k, v in sorted(p.items()))}]"


# --------------------------------------------------------------------------- JSONL io


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    """Append one row and flush to disk immediately (survives a killed session)."""
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str, ensure_ascii=False) + "\n")
        f.flush()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read rows, skipping a truncated/corrupt line (e.g. a kill mid-write) with a warning."""
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                warnings.warn(
                    f"{path}:{i}: skipping unparseable line", RuntimeWarning, stacklevel=2
                )
    return rows


def row_key(row: dict[str, Any]) -> tuple[str, str]:
    return (row["problem_id"], row.get("method_id") or row["method"])


def dedupe_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per (problem_id, method_id): a success beats an error; otherwise the last wins."""
    best: dict[tuple[str, str], dict[str, Any]] = {}
    for r in rows:
        k = row_key(r)
        old = best.get(k)
        if old is None or not (r.get("error") and not old.get("error")):
            best[k] = r
    return list(best.values())


# --------------------------------------------------------------------------- provenance


def git_commit(cwd: str | Path | None = None) -> str:
    """HEAD commit hash, suffixed '-dirty' if there are uncommitted changes."""
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return head + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def protocol_hash(cfg: Config) -> str:
    """`config_hash` with run-only eval settings reset to defaults (see _RUN_ONLY_EVAL_FIELDS)."""
    defaults = type(cfg.eval)()
    reset = {f: getattr(defaults, f) for f in _RUN_ONLY_EVAL_FIELDS}
    return config_hash(cfg.model_copy(update={"eval": cfg.eval.model_copy(update=reset)}))


@dataclass
class RunContext:
    run_dir: Path
    run_id: str
    config_hash: str
    protocol_hash: str
    git_commit: str
    seed: int


def prepare_run_dir(run_dir: str | Path, cfg: Config) -> RunContext:
    """Create the run dir and write config.yaml + git_commit.txt (spec §13.7).

    On resume, a config.yaml with a different hash is kept as config.<n>.yaml and
    a warning is printed: mixed configs inside one run are allowed (e.g. a
    concurrency change) but must be visible.
    """
    d = Path(run_dir)
    d.mkdir(parents=True, exist_ok=True)
    h = config_hash(cfg)
    cfg_path = d / "config.yaml"
    if cfg_path.exists():
        old_hash_file = d / "config_hash.txt"
        old = old_hash_file.read_text().strip() if old_hash_file.exists() else None
        if old and old != h:
            n = len(list(d.glob("config.*.yaml")))
            shutil.copy(cfg_path, d / f"config.{n}.yaml")
            warnings.warn(
                f"{d}: resuming with a different config (hash {old[:8]} -> {h[:8]})",
                RuntimeWarning,
                stacklevel=2,
            )
    dump_resolved(cfg, cfg_path)
    (d / "config_hash.txt").write_text(h + "\n")
    commit = git_commit()
    (d / "git_commit.txt").write_text(commit + "\n")
    return RunContext(d, d.name, h, protocol_hash(cfg), commit, cfg.seed)


# --------------------------------------------------------------------------- sharding


def parse_shard(spec: str | None) -> tuple[int, int] | None:
    """'i/n' -> (i, n), 0-based. None passes through."""
    if not spec:
        return None
    i, _, n = spec.partition("/")
    si, sn = int(i), int(n)
    if not (sn >= 1 and 0 <= si < sn):
        raise ValueError(f"bad shard {spec!r}: need 0 <= i < n")
    return si, sn


def shard_of(problem_id: str, n: int) -> int:
    return int(hashlib.sha256(problem_id.encode()).hexdigest(), 16) % n


def in_shard(problem_id: str, shard: tuple[int, int] | None) -> bool:
    return shard is None or shard_of(problem_id, shard[1]) == shard[0]


def shard_problems(problems: Sequence[Problem], shard: tuple[int, int] | None) -> list[Problem]:
    """Stable split by hashed problem id (independent of list order)."""
    if shard is None:
        return list(problems)
    return [p for p in problems if in_shard(p.id, shard)]


# --------------------------------------------------------------------------- running


def redact(problem: Problem) -> Problem:
    """The copy of a problem a method is allowed to see: no ground truth."""
    return dataclasses.replace(problem, answer="")


@dataclass
class RunSummary:
    run_dir: Path
    method: str
    method_id: str
    n_problems: int  # in this shard
    n_skipped: int  # already present (resume)
    n_run: int  # rows written now
    n_errors: int  # error rows written now
    stopped_reason: str | None = None  # e.g. "budget_exceeded: ..."

    @property
    def stopped(self) -> bool:
        return self.stopped_reason is not None


def _now() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")


def _base_row(
    ctx: RunContext, problem: Problem, method: Method, shard: tuple[int, int] | None
) -> dict[str, Any]:
    return {
        "run_id": ctx.run_id,
        "problem_id": problem.id,
        "source": problem.source,
        "level": problem.level,
        "subject": problem.subject,
        "method": method.name,
        "method_id": method_id(method),
        "params": method_params(method),
        "gold": problem.answer,
        "config_hash": ctx.config_hash,
        "protocol_hash": ctx.protocol_hash,
        "prompt_version": PROMPT_VERSION,
        "git_commit": ctx.git_commit,
        "seed": ctx.seed,
        "shard": f"{shard[0]}/{shard[1]}" if shard else None,
    }


async def run_method(
    method: Method,
    problems: Sequence[Problem],
    cfg: Config,
    ctx: RunContext,
    *,
    shard: tuple[int, int] | None = None,
    retry_errors: bool = False,
    timeout_s: float | None = None,
    grader: Callable[[str | None, str], bool] = is_equivalent,
) -> RunSummary:
    """Run `method` over `problems` (this shard), appending one JSONL row per problem."""
    path = ctx.run_dir / ROWS_FILE
    mid = method_id(method)
    existing = dedupe_rows(read_jsonl(path))
    done = {
        r["problem_id"]
        for r in existing
        if (r.get("method_id") or r["method"]) == mid and not (retry_errors and r.get("error"))
    }
    mine = shard_problems(problems, shard)
    todo = [p for p in mine if p.id not in done]
    summary = RunSummary(ctx.run_dir, method.name, mid, len(mine), len(mine) - len(todo), 0, 0)
    log.info(
        "%s: %d problems, %d already done, %d to run", mid, len(mine), summary.n_skipped, len(todo)
    )

    queue: asyncio.Queue[Problem] = asyncio.Queue()
    for p in todo:
        queue.put_nowait(p)
    stop = asyncio.Event()

    async def solve_one(p: Problem) -> None:
        row = _base_row(ctx, p, method, shard)
        led: Ledger
        with ledger_scope() as led:
            try:
                coro = method.solve(redact(p))
                res = await (asyncio.wait_for(coro, timeout_s) if timeout_s else coro)
                err = None
            except BudgetExceeded:
                raise
            except Exception as e:  # noqa: BLE001 - isolate any per-problem failure
                res, err = None, e
        if err is None and res is not None:
            try:
                correct = bool(grader(res.answer, p.answer))
                grade_err = None
            except Exception as e:  # noqa: BLE001
                correct, grade_err = False, f"grader: {type(e).__name__}: {e}"
            row.update(
                answer=res.answer,
                correct=correct,
                error=grade_err,
                flagged=grade_err is not None,
                **led.to_dict(),
                timestamp=_now(),
                raw=res.raw,
            )
        else:
            assert err is not None
            row.update(
                answer=None,
                correct=False,
                error=f"{type(err).__name__}: {err}",
                flagged=True,
                **led.to_dict(),
                timestamp=_now(),
                raw={"traceback": "".join(traceback.format_exception(err))[-4000:]},
            )
        append_jsonl(path, row)
        summary.n_run += 1
        summary.n_errors += int(row["error"] is not None)

    async def worker() -> None:
        while not stop.is_set():
            try:
                p = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                await solve_one(p)
            except BudgetExceeded as e:
                if summary.stopped_reason is None:
                    summary.stopped_reason = f"budget_exceeded: {e}"
                stop.set()

    n_workers = max(1, min(cfg.eval.concurrency, len(todo)))
    await asyncio.gather(*(worker() for _ in range(n_workers)))
    if summary.stopped:
        log.error("%s stopped: %s", mid, summary.stopped_reason)
    return summary


async def run_methods(
    methods: Sequence[Method],
    problems: Sequence[Problem],
    cfg: Config,
    ctx: RunContext,
    *,
    shard: tuple[int, int] | None = None,
    retry_errors: bool = False,
    timeout_s: float | None = None,
    grader: Callable[[str | None, str], bool] = is_equivalent,
) -> list[RunSummary]:
    """Run methods one after another; a budget stop ends the whole run. Writes summary.csv."""
    ids = [p.id for p in problems]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate problem ids in the problem set")
    out = []
    for m in methods:
        s = await run_method(
            m,
            problems,
            cfg,
            ctx,
            shard=shard,
            retry_errors=retry_errors,
            timeout_s=timeout_s,
            grader=grader,
        )
        out.append(s)
        if s.stopped:
            break
    write_summary(ctx.run_dir, seed=cfg.seed)
    return out


# --------------------------------------------------------------------------- summary


SUMMARY_COLUMNS = [
    "method_id",
    "method",
    "params",
    "n_problems",
    "n_errors",
    "accuracy",
    "ci_lo",
    "ci_hi",
    "mean_completion_tokens",
    "median_completion_tokens",
    "mean_prompt_tokens",
    "jev_calls_total",
    "jev_calls_mean",
    "jev_usd_total",
    "jev_usd_mean",
    "wall_time_s_total",
    "wall_time_s_mean",
]


def summarize(
    rows: Sequence[dict[str, Any]], n_boot: int = BOOTSTRAP_N, seed: int = 0
) -> list[dict[str, Any]]:
    """Per-method aggregates. Error rows count as incorrect and as compute spent."""
    by: dict[str, list[dict[str, Any]]] = {}
    for r in dedupe_rows(rows):
        by.setdefault(r.get("method_id") or r["method"], []).append(r)
    out = []
    for mid in sorted(by):
        rs = by[mid]
        corr = [bool(r["correct"]) for r in rs]
        ct = np.array([r.get("gemma_completion_tokens", 0) for r in rs], dtype=float)
        pt = np.array([r.get("gemma_prompt_tokens", 0) for r in rs], dtype=float)
        jc = np.array([r.get("jev_calls", 0) for r in rs], dtype=float)
        ju = np.array([r.get("jev_usd", 0.0) for r in rs], dtype=float)
        wt = np.array([r.get("wall_time_s", 0.0) for r in rs], dtype=float)
        lo, hi = bootstrap_ci(corr, n=n_boot, seed=seed)
        out.append(
            {
                "method_id": mid,
                "method": rs[0]["method"],
                "params": json.dumps(rs[0].get("params") or {}, sort_keys=True),
                "n_problems": len(rs),
                "n_errors": sum(1 for r in rs if r.get("error")),
                "accuracy": accuracy(corr),
                "ci_lo": lo,
                "ci_hi": hi,
                "mean_completion_tokens": float(ct.mean()),
                "median_completion_tokens": float(np.median(ct)),
                "mean_prompt_tokens": float(pt.mean()),
                "jev_calls_total": int(jc.sum()),
                "jev_calls_mean": float(jc.mean()),
                "jev_usd_total": float(ju.sum()),
                "jev_usd_mean": float(ju.mean()),
                "wall_time_s_total": float(wt.sum()),
                "wall_time_s_mean": float(wt.mean()),
            }
        )
    return out


def write_summary(
    run_dir: str | Path, n_boot: int = BOOTSTRAP_N, seed: int = 0
) -> list[dict[str, Any]]:
    d = Path(run_dir)
    rows = summarize(read_jsonl(d / ROWS_FILE), n_boot=n_boot, seed=seed)
    with (d / SUMMARY_FILE).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return rows


def read_summary(run_dir: str | Path) -> list[dict[str, Any]]:
    with (Path(run_dir) / SUMMARY_FILE).open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


# --------------------------------------------------------------------------- merging shards / runs


def merge_runs(
    src_dirs: Sequence[str | Path],
    dst_dir: str | Path,
    *,
    allow_mixed_config: bool = False,
    n_boot: int = BOOTSTRAP_N,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Concatenate shard (or per-method) runs into `dst_dir`, de-duplicated.

    Within one method_id every row must share a protocol_hash, unless
    `allow_mixed_config`; different methods may come from different configs.
    """
    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    commits = []
    for s in map(Path, src_dirs):
        rows.extend(read_jsonl(s / ROWS_FILE))
        gc = s / "git_commit.txt"
        commits.append(f"{s.name}: {gc.read_text().strip() if gc.exists() else 'unknown'}")
        for cfg_file in s.glob("config*.yaml"):
            shutil.copy(cfg_file, dst / f"{s.name}.{cfg_file.name}")
    merged = dedupe_rows(rows)
    if not allow_mixed_config:
        hashes: dict[str, set[str]] = {}
        for r in merged:
            hashes.setdefault(r.get("method_id") or r["method"], set()).add(
                r.get("protocol_hash") or r.get("config_hash", "")
            )
        bad = {m: h for m, h in hashes.items() if len(h) > 1}
        if bad:
            raise ValueError(
                f"mixed protocol hashes within a method: {sorted(bad)}; "
                "pass allow_mixed_config=True if intended"
            )
    merged.sort(key=lambda r: (r.get("method_id") or r["method"], r["problem_id"]))
    with (dst / ROWS_FILE).open("w", encoding="utf-8") as f:
        for r in merged:
            f.write(json.dumps(r, default=str, ensure_ascii=False) + "\n")
    (dst / "git_commit.txt").write_text("\n".join(commits) + "\n")
    return write_summary(dst, n_boot=n_boot, seed=seed)


# --------------------------------------------------------------------------- cost gate


class CostConfirmationRequired(RuntimeError):
    pass


# Rough Jev input tokens per call (problem + steps so far + candidates + instructions).
# Deliberately pessimistic; replace with Person 2's estimator once it exists.
JEV_TOKENS_PER_CALL = 2000


def estimate_jev_usd(cfg: Config, n_problems: int, methods: Sequence[Method]) -> float:
    """Upper-bound-ish Jev USD for running `methods` on `n_problems` (ignores the cache).

    Only methods with ``uses_jev = True`` count. Search makes at most one Jev call per
    simulation (an expansion or a terminal evaluation); best-of-N one per sample.
    """
    per_token = cfg.judge.usd_per_mtok / 1e6
    calls = 0
    for m in methods:
        if not getattr(m, "uses_jev", False):
            continue
        p = method_params(m)
        per_problem = int(p["n_simulations"]) if "n_simulations" in p else int(p.get("n", 1))
        calls += n_problems * per_problem
    return calls * JEV_TOKENS_PER_CALL * per_token


def check_cost(estimate_usd: float, yes: bool, threshold_usd: float = 1.0) -> None:
    """Print the estimate; refuse above the threshold unless --yes (team rule B8)."""
    print(f"Estimated Jev cost: ${estimate_usd:.4f} (confirmation threshold ${threshold_usd:.2f})")
    if estimate_usd > threshold_usd and not yes:
        raise CostConfirmationRequired(
            f"estimated ${estimate_usd:.2f} > ${threshold_usd:.2f}: "
            "ask the team, then re-run with --yes"
        )
