import asyncio
import csv
import json

import pytest

from thoughtzero.accounting import record_gemma, record_jev
from thoughtzero.config import load_config
from thoughtzero.eval import analysis
from thoughtzero.eval.cli import mock_problems
from thoughtzero.eval.methods import build_method, load_stored_samples, parse_method_spec
from thoughtzero.eval.runner import (
    ROWS_FILE,
    CostConfirmationRequired,
    MethodResult,
    check_cost,
    estimate_jev_usd,
    merge_runs,
    parse_shard,
    prepare_run_dir,
    protocol_hash,
    read_jsonl,
    read_summary,
    run_methods,
    shard_problems,
)
from thoughtzero.eval.toolkit import mock_toolkit
from thoughtzero.judge.budget import BudgetExceeded
from thoughtzero.mocks import MockGenerator
from thoughtzero.types import Problem

TOOLS = mock_toolkit()


def problems(n=10):
    return [Problem(f"t/{i:03d}", f"q{i}", str(i), "math500", level=1 + i % 5) for i in range(n)]


class Fake:
    """Answers correctly on even problem ids; records 10 tokens per problem."""

    def __init__(self, name="fake", fail_on=(), budget_on=None, params=None):
        self.name = name
        self.calls = []
        self.fail_on = set(fail_on)
        self.budget_on = budget_on
        self.params = params or {}

    async def solve(self, p):
        assert p.answer == "", "method must not see the gold answer"
        self.calls.append(p.id)
        await asyncio.sleep(0)
        record_gemma(5, 10)
        record_jev(100, 0.001, cache_hit=False)
        if p.id in self.fail_on:
            raise ValueError("boom")
        if self.budget_on is not None and len(self.calls) >= self.budget_on:
            raise BudgetExceeded("over budget")
        i = int(p.id.split("/")[1])
        return MethodResult(str(i) if i % 2 == 0 else "wrong", {"i": i})


def cfg(*overrides):
    return load_config(None, list(overrides))


def run(methods, probs, run_dir, c=None, **kw):
    c = c or cfg()
    ctx = prepare_run_dir(run_dir, c)
    kw.setdefault("grader", TOOLS.grade)  # data.grading is Person 3's (stub until it lands)
    return asyncio.run(run_methods(methods, probs, c, ctx, **kw))


def test_rows_grading_provenance(tmp_path):
    (s,) = run([Fake()], problems(6), tmp_path)
    rows = read_jsonl(tmp_path / ROWS_FILE)
    assert len(rows) == 6 and s.n_run == 6 and s.n_errors == 0
    for r in rows:
        i = int(r["problem_id"].split("/")[1])
        assert r["correct"] == (i % 2 == 0)
        assert r["gold"] == str(i)
        assert r["gemma_completion_tokens"] == 10 and r["jev_calls"] == 1
        for k in (
            "config_hash",
            "protocol_hash",
            "prompt_version",
            "git_commit",
            "seed",
            "timestamp",
            "raw",
            "level",
        ):
            assert k in r
    assert (tmp_path / "config.yaml").exists() and (tmp_path / "git_commit.txt").exists()


def test_resume_skips_finished(tmp_path):
    probs = problems(8)
    run([Fake()], probs[:5], tmp_path)
    m = Fake()
    (s,) = run([m], probs, tmp_path)
    assert sorted(m.calls) == [p.id for p in probs[5:]]
    assert s.n_skipped == 5
    assert len(read_jsonl(tmp_path / ROWS_FILE)) == 8


def test_error_row_and_run_continues(tmp_path):
    m = Fake(fail_on={"t/002"})
    (s,) = run([m], problems(5), tmp_path)
    rows = {r["problem_id"]: r for r in read_jsonl(tmp_path / ROWS_FILE)}
    assert len(rows) == 5 and s.n_errors == 1
    err = rows["t/002"]
    assert err["correct"] is False and err["flagged"] is True and "boom" in err["error"]
    assert err["gemma_completion_tokens"] == 10  # compute spent before the error is still counted
    # resume does not retry errors by default, --retry-errors does
    m2 = Fake()
    run([m2], problems(5), tmp_path)
    assert m2.calls == []
    m3 = Fake()
    run([m3], problems(5), tmp_path, retry_errors=True)
    assert m3.calls == ["t/002"]
    summ = {r["method_id"]: r for r in read_summary(tmp_path)}
    assert int(summ["fake"]["n_errors"]) == 0  # the success superseded the error row


def test_budget_exceeded_stops_whole_run(tmp_path):
    m1 = Fake("a", budget_on=3)
    m2 = Fake("b")
    out = run([m1, m2], problems(10), tmp_path, c=cfg("eval.concurrency=1"))
    assert len(out) == 1 and out[0].stopped and "budget" in out[0].stopped_reason
    rows = read_jsonl(tmp_path / ROWS_FILE)
    assert len(rows) == 2  # the 3rd problem raised: no row, it will be redone on resume
    assert m2.calls == []


def test_timeout_is_an_error_row(tmp_path):
    class Slow(Fake):
        async def solve(self, p):
            await asyncio.sleep(5)

    (s,) = run([Slow("slow")], problems(2), tmp_path, timeout_s=0.05)
    assert s.n_errors == 2
    assert all("Timeout" in r["error"] for r in read_jsonl(tmp_path / ROWS_FILE))


def test_shard_merge_reproduces_unsharded(tmp_path):
    probs = problems(23)
    run([Fake()], probs, tmp_path / "full")
    shards = []
    for i in range(3):
        d = tmp_path / f"s{i}"
        run([Fake()], probs, d, shard=(i, 3))
        shards.append(d)
    sizes = [len(read_jsonl(d / ROWS_FILE)) for d in shards]
    assert sum(sizes) == 23 and all(s > 0 for s in sizes)
    merge_runs(shards + [shards[0]], tmp_path / "merged")  # duplicate source: de-duplicated

    def key(d):
        return sorted(
            (r["problem_id"], r["answer"], r["correct"]) for r in read_jsonl(d / ROWS_FILE)
        )

    assert key(tmp_path / "merged") == key(tmp_path / "full")
    a, b = read_summary(tmp_path / "merged")[0], read_summary(tmp_path / "full")[0]
    for k in ("n_problems", "accuracy", "ci_lo", "ci_hi", "mean_completion_tokens"):
        assert a[k] == b[k]


def test_shards_partition_and_parse():
    probs = problems(50)
    parts = [shard_problems(probs, (i, 4)) for i in range(4)]
    ids = [p.id for part in parts for p in part]
    assert sorted(ids) == sorted(p.id for p in probs)
    assert shard_problems(list(reversed(probs)), (1, 4))[::-1] == parts[1]  # order-independent
    assert parse_shard("2/3") == (2, 3) and parse_shard(None) is None
    with pytest.raises(ValueError):
        parse_shard("3/3")


def test_merge_refuses_mixed_protocol(tmp_path):
    run([Fake()], problems(4), tmp_path / "a")
    run([Fake()], problems(4)[2:], tmp_path / "b", c=cfg("search.k=6"))
    with pytest.raises(ValueError, match="mixed protocol"):
        merge_runs([tmp_path / "a", tmp_path / "b"], tmp_path / "m")
    merge_runs([tmp_path / "a", tmp_path / "b"], tmp_path / "m", allow_mixed_config=True)


def test_run_only_settings_do_not_block_merge(tmp_path):
    a, b = cfg(), cfg("eval.concurrency=16", "eval.run_id=x", "eval.out_dir=elsewhere")
    assert protocol_hash(a) == protocol_hash(b)
    assert protocol_hash(a) != protocol_hash(cfg("search.n_simulations=8"))
    run([Fake()], problems(4), tmp_path / "a", c=a)
    run([Fake()], problems(6), tmp_path / "b", c=b)
    rows = merge_runs([tmp_path / "a", tmp_path / "b"], tmp_path / "m")
    assert rows[0]["n_problems"] == 6


def test_summary_matches_jsonl(tmp_path):
    run([Fake("a"), Fake("b", params={"n": 4}, fail_on={"t/001"})], problems(9), tmp_path)
    rows = read_jsonl(tmp_path / ROWS_FILE)
    with (tmp_path / "summary.csv").open() as f:
        summ = {r["method_id"]: r for r in csv.DictReader(f)}
    assert set(summ) == {"a", "b[n=4]"}
    for mid, s in summ.items():
        rs = [r for r in rows if r["method_id"] == mid]
        assert int(s["n_problems"]) == len(rs)
        assert int(s["n_errors"]) == sum(1 for r in rs if r["error"])
        assert float(s["accuracy"]) == pytest.approx(sum(r["correct"] for r in rs) / len(rs))
        assert float(s["mean_completion_tokens"]) == pytest.approx(10)
        assert float(s["jev_usd_total"]) == pytest.approx(0.001 * len(rs))
        assert float(s["ci_lo"]) <= float(s["accuracy"]) <= float(s["ci_hi"])
    assert json.loads(summ["b[n=4]"]["params"]) == {"n": 4}


def test_truncated_line_is_tolerated(tmp_path):
    run([Fake()], problems(3), tmp_path)
    with (tmp_path / ROWS_FILE).open("a") as f:
        f.write('{"problem_id": "t/00')  # killed mid-write
    with pytest.warns(RuntimeWarning):
        rows = read_jsonl(tmp_path / ROWS_FILE)
    assert len(rows) == 3


def test_duplicate_problem_ids_rejected(tmp_path):
    p = problems(2)
    with pytest.raises(ValueError, match="duplicate"):
        run([Fake()], p + p[:1], tmp_path)


def test_cost_gate(capsys):
    check_cost(0.5, yes=False)
    check_cost(3.0, yes=True)
    with pytest.raises(CostConfirmationRequired):
        check_cost(1.01, yes=False)
    assert "Estimated Jev cost" in capsys.readouterr().out


def test_config_overrides():
    c = load_config(
        None, ["eval.methods=[tz, sc]", "eval.concurrency=2", "search.n_simulations=16"]
    )
    assert c.eval.methods == ["tz", "sc"] and c.eval.concurrency == 2
    assert c.search.n_simulations == 16


def test_method_specs():
    assert parse_method_spec("tz") == ("tz", None, None)
    assert parse_method_spec("sc:8") == ("sc", 8, None)
    assert parse_method_spec("sc:8/64") == ("sc", 8, 64)
    with pytest.raises(ValueError):
        parse_method_spec("sc:8/4")
    with pytest.raises(KeyError):
        parse_method_spec("nope")
    with pytest.raises(ValueError):
        parse_method_spec("sc")
    with pytest.raises(ValueError):
        parse_method_spec("tz:16")


def test_tz_through_runner_on_mocks(tmp_path):
    c = cfg("search.k=3", "search.n_simulations=32")
    m = build_method("tz", c, MockGenerator(), mock=True, tools=TOOLS)
    (s,) = run([m], mock_problems(3), tmp_path, c=c)
    rows = read_jsonl(tmp_path / ROWS_FILE)
    assert s.n_errors == 0 and len(rows) == 3
    assert all(r["method_id"] == "tz[n_simulations=32]" for r in rows)
    assert all(r["correct"] for r in rows)  # 32 sims beat the misleading prior on the toy
    assert all(r["jev_calls"] > 0 and r["gemma_completion_tokens"] > 0 for r in rows)
    assert {"stats", "tree", "answer_steps"} <= set(rows[0]["raw"])
    diag = analysis.diagnostics_table(rows)[0]
    assert diag["n_with_stats"] == 3 and diag["mean_expansions"] > 0


def test_baselines_through_runner_and_sample_curves(tmp_path):
    c = cfg()
    probs = mock_problems(3)
    cot = build_method("cot", c, MockGenerator(), mock=True)
    sc = build_method("sc:2/4", c, MockGenerator(), mock=True)
    run([cot, sc], probs, tmp_path / "b2", c=c)
    rows = read_jsonl(tmp_path / "b2" / ROWS_FILE)
    assert {r["method_id"] for r in rows} == {"cot", "sc[n=2]"}
    assert not any(r["error"] for r in rows)
    for r in (r for r in rows if r["method"] == "sc"):
        # the ledger paid for all 4 samples; the compute axis counts only the 2 that voted
        assert r["compute_tokens"] == r["raw"]["completion_tokens_at_n"]
        assert 0 < r["compute_tokens"] < r["gemma_completion_tokens"]

    # B3 reranks B2's stored samples: Jev only, no new Gemma tokens
    stored = load_stored_samples(tmp_path / "b2")
    assert set(stored) == {p.id for p in probs}
    bon = build_method("bon:2/4", c, MockGenerator(), mock=True, stored_samples=stored)
    run([bon], probs, tmp_path / "b3", c=c)
    b3 = read_jsonl(tmp_path / "b3" / ROWS_FILE)
    assert all(r["raw"]["reused_samples"] and r["gemma_completion_tokens"] == 0 for r in b3)
    assert all(r["compute_tokens"] > 0 and r["jev_calls"] == 4 for r in b3)

    curves = analysis.expand_sample_curves(rows + b3, [1, 2, 4, 8], TOOLS.grade)
    ids = {r["method_id"] for r in curves}
    assert ids == {"cot", "sc[n=1]", "sc[n=2]", "sc[n=4]", "bon[n=1]", "bon[n=2]", "bon[n=4]"}
    by = analysis.rows_by_method(curves)
    tok = [sum(r["compute_tokens"] for r in by[f"sc[n={n}]"]) for n in (1, 2, 4)]
    assert tok[0] < tok[1] < tok[2]  # cost grows with N; n=8 > 4 stored is skipped
    assert all(r["derived_from"] == "bon[n=2]" for r in by["bon[n=4]"])


def test_jev_estimate():
    c = cfg("search.n_simulations=16")
    tz = build_method("tz", c, MockGenerator(), mock=True)
    self_tz = build_method("tz_self", c, MockGenerator(), mock=True)
    usd = estimate_jev_usd(c, 100, [tz, self_tz])
    assert usd == pytest.approx(100 * 16 * 2000 * 0.042e-6)  # the self-judge costs no Jev


def test_analysis_paired_and_id_check(tmp_path):
    probs = problems(20)
    run(
        [Fake("tz", params={"n_simulations": 8}), Fake("sc", params={"n": 4})],
        probs,
        tmp_path / "x",
    )
    rows = analysis.load_runs([tmp_path / "x"])
    table = analysis.main_table(rows)
    tz = next(r for r in table if r["method"] == "tz")
    assert tz["vs"] == "sc[n=4]" and tz["diff"] == 0.0
    assert analysis.tz_tokens_per_nsim(rows) == {8: 10.0}
    assert analysis.by_level(rows)
    md = analysis.main_table_markdown(table)
    assert "Δ vs B2" in md
    bad = [r for r in rows if not (r["method"] == "sc" and r["problem_id"] == "t/000")]
    with pytest.raises(analysis.ProblemSetMismatch):
        analysis.main_table(bad)
    audit = analysis.sanity_audit(bad)
    assert audit["ids_identical"] is False
    assert len(audit["manual_sample"]) == 20


def test_aime_split():
    rows = [
        {"problem_id": "aime2023/1", "source": "aime", "method": "tz", "correct": True},
        {"problem_id": "aime2025/1", "source": "aime", "method": "tz", "correct": False},
    ]
    out = {r["group"]: r for r in analysis.aime_split(rows, cutoff_year=2024)}
    assert out["pre-cutoff"]["accuracy"] == 1.0 and out["post-cutoff"]["accuracy"] == 0.0
