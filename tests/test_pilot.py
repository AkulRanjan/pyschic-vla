import asyncio
import json
import math

import numpy as np
import pytest

from thoughtzero.eval.cli import mock_problems
from thoughtzero.eval.toolkit import mock_toolkit
from thoughtzero.mocks import MockGenerator, MockJudge
from thoughtzero.pilot import mc_label, report, score, traces
from thoughtzero.pilot.report import decide

TOOLS = mock_toolkit()  # data.grading / llm.prompts are Person 3's (stubs until they land)

# --------------------------------------------------------------------------- prefixes


@pytest.mark.parametrize("n", [1, 2, 5, 8, 9, 13, 20, 57])
def test_prefix_lengths(n):
    ls = traces.select_prefix_lengths(n, cap=8)
    assert len(ls) <= 8 and len(ls) == len(set(ls))
    assert ls == sorted(ls) and ls[0] == 1 and ls[-1] == n
    if n <= 8:
        assert ls == list(range(1, n + 1))
    else:
        gaps = np.diff(ls)
        assert gaps.max() - gaps.min() <= 1  # evenly spread


def test_prefix_lengths_empty():
    assert traces.select_prefix_lengths(0) == []


def test_make_prefixes():
    tr = [
        {"problem_id": "a", "n_steps": 3, "steps": ["x", "y", "z"]},
        {"problem_id": "b", "n_steps": 0, "steps": []},
    ]
    px = traces.make_prefixes(tr)
    assert [p["prefix_id"] for p in px] == ["a#1", "a#2", "a#3"]
    assert px[-1]["is_final"] and not px[0]["is_final"]
    assert px[1]["rel_depth"] == pytest.approx(2 / 3)
    assert traces.prefix_steps(px[1], {"a": tr[0]}) == ["x", "y"]


# --------------------------------------------------------------------------- labels


def test_label_from_completions():
    comps = [["so \\boxed{4}"], ["\\boxed{5}"], [], ["\\boxed{4}"]]
    lab = mc_label.label_from_completions(["2+2"], comps, "4", TOOLS)
    assert lab["n_samples"] == 4 and lab["n_correct"] == 2
    assert lab["soft_label"] == 0.5 and lab["hard_label"] is True
    assert lab["answers"] == ["4", "5", None, "4"]
    none = mc_label.label_from_completions(["x"], [["\\boxed{1}"]] * 8, "4", TOOLS)
    assert none["soft_label"] == 0 and none["hard_label"] is False


def test_label_terminal():
    assert mc_label.label_terminal("4", "4", TOOLS)["soft_label"] == 1.0
    lab = mc_label.label_terminal(None, "4", TOOLS)
    assert (
        lab["hard_label"] is False and lab["label_source"] == "terminal" and lab["n_samples"] == 0
    )


class CountingGen(MockGenerator):
    def __init__(self):
        super().__init__()
        self.n_completion_calls = 0
        self.n_solution_calls = 0

    async def sample_completions(self, problem, steps, n, temperature):
        self.n_completion_calls += 1
        assert n == 8
        return await super().sample_completions(problem, steps, n, temperature)

    async def sample_solutions(self, problem, n, temperature):
        # MockGenerator.sample_solutions routes through sample_completions; bypass our counter
        self.n_solution_calls += 1
        comps = await MockGenerator.sample_completions(self, problem, [], n, temperature)
        return [TOOLS.format(c).rstrip() for c in comps]


class SpyJudge(MockJudge):
    def __init__(self):
        super().__init__()
        self.seen = []

    async def step_sound(self, problem, steps):
        assert isinstance(problem, str) and isinstance(steps, list)
        self.seen.append((problem, tuple(steps)))
        return await super().step_sound(problem, steps)


def test_pipeline_end_to_end_on_mocks(tmp_path):
    probs = mock_problems(6)
    pb = {p.id: p for p in probs}
    gen = CountingGen()

    async def go():
        await traces.generate_traces(probs, gen, tmp_path, tools=TOOLS)
        tl = traces.load_traces(tmp_path)
        tr = {t["problem_id"]: t for t in tl}
        px = traces.make_prefixes(tl)
        await mc_label.label_prefixes(px, tr, pb, gen, tmp_path, tools=TOOLS)
        judge = SpyJudge()
        await score.score_prefixes(px, tr, pb, judge, "jev", tmp_path)
        return tl, px, judge

    tl, px, judge = asyncio.run(go())
    assert len(tl) == 6 and gen.n_solution_calls == 6
    labels = mc_label.load_labels(tmp_path)
    assert len(labels) == len(px)
    # final prefixes contain the boxed answer: labelled without sampling
    finals = [lab for lab in labels if lab["is_final"]]
    assert finals and all(lab["label_source"] == "terminal" for lab in finals)
    assert gen.n_completion_calls == len(px) - len(finals)
    for lab in labels:
        assert lab["soft_label"] == pytest.approx(lab["n_correct"] / max(lab["n_samples"], 1))
        assert lab["hard_label"] == (lab["soft_label"] > 0)
    assert len(judge.seen) == len(px)
    questions = {p.question for p in probs}
    assert all(q in questions for q, _ in judge.seen)  # the judge sees question + prefix only

    # resume: re-running stages does no new work
    n_before = (gen.n_solution_calls, gen.n_completion_calls)
    asyncio.run(traces.generate_traces(probs, gen, tmp_path, tools=TOOLS))
    tr = {t["problem_id"]: t for t in traces.load_traces(tmp_path)}
    asyncio.run(mc_label.label_prefixes(px, tr, pb, gen, tmp_path, tools=TOOLS))
    assert (gen.n_solution_calls, gen.n_completion_calls) == n_before

    md, dec = report.build_report(tmp_path, judges=["jev"], n_boot=50)
    assert dec.verdict in {"GO", "PARTIAL", "NO-GO", "UNAVAILABLE"}
    assert (tmp_path / "report.md").exists() and (
        tmp_path / "figures" / "reliability_jev.png"
    ).exists()
    assert "INCOMPLETE" not in md


def test_sharded_labels_cover_everything(tmp_path):
    probs = mock_problems(9)
    pb = {p.id: p for p in probs}
    gen = MockGenerator()
    asyncio.run(traces.generate_traces(probs, gen, tmp_path, tools=TOOLS))
    tl = traces.load_traces(tmp_path)
    tr = {t["problem_id"]: t for t in tl}
    px = traces.make_prefixes(tl)
    for i in range(3):
        asyncio.run(mc_label.label_prefixes(px, tr, pb, gen, tmp_path, shard=(i, 3), tools=TOOLS))
    assert {lab["prefix_id"] for lab in mc_label.load_labels(tmp_path)} == {
        p["prefix_id"] for p in px
    }


def test_sensitivity_subset_deterministic():
    ids = [f"p{i}" for i in range(200)]
    a = score.sensitivity_subset(ids, 50, seed=0)
    assert len(a) == 50 and a == score.sensitivity_subset(list(reversed(ids)), 50, seed=0)
    assert score.sensitivity_subset(ids[:10], 50) == sorted(ids[:10])


# --------------------------------------------------------------------------- decision & report


def test_decide_thresholds():
    assert decide(0.80, (0.78, 0.83)).verdict == "GO"
    assert decide(0.75, (0.76, 0.80)).verdict == "GO"
    assert decide(0.70, (0.67, 0.72)).verdict == "PARTIAL"
    assert decide(0.65, (0.66, 0.70)).verdict == "PARTIAL"
    assert decide(0.60, (0.55, 0.64)).verdict == "NO-GO"
    assert decide(math.nan).verdict == "UNAVAILABLE"
    assert report.GO_THRESHOLD == 0.75 and report.PARTIAL_THRESHOLD == 0.65  # goalposts are fixed


def test_decide_straddle():
    d = decide(0.77, (0.70, 0.84))
    assert d.verdict == "GO" and d.straddles == (0.75,)
    assert "straddles" in d.markdown() and "CP2" in d.markdown()
    assert decide(0.70, (0.60, 0.80)).straddles == (0.65, 0.75)
    assert decide(0.90, (0.85, 0.95)).straddles == ()


def _write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def synthetic_pilot(tmp_path, n_pos_high: int, n: int = 10):
    """n problems x 2 prefixes: depth 1 positive, depth 2 negative.

    Negatives all score 0.5; `n_pos_high` positives score 0.9, the rest 0.1, so
    AUROC == n_pos_high / n exactly.
    """
    tr, labs, sc = [], [], []
    for k in range(n):
        pid = f"syn/{k:02d}"
        tr.append(
            {
                "problem_id": pid,
                "steps": ["a", "b"],
                "n_steps": 2,
                "final_answer": "1",
                "trace_correct": False,
            }
        )
        for depth, hard, s in ((1, True, 0.9 if k < n_pos_high else 0.1), (2, False, 0.5)):
            px = f"{pid}#{depth}"
            labs.append(
                {
                    "prefix_id": px,
                    "problem_id": pid,
                    "depth": depth,
                    "rel_depth": depth / 2,
                    "n_steps": 2,
                    "is_final": depth == 2,
                    "n_samples": 8,
                    "n_correct": 4 * hard,
                    "soft_label": 0.5 * hard,
                    "hard_label": hard,
                    "label_source": "mc",
                }
            )
            sc.append(
                {
                    "prefix_id": px,
                    "problem_id": pid,
                    "depth": depth,
                    "judge": "jev",
                    "score": s,
                    "latency_s": 0.2 + 0.01 * k,
                    "jev_calls": 1,
                    "jev_cache_hits": 0,
                    "jev_usd": 1e-5,
                }
            )
    _write(tmp_path / "traces.jsonl", tr)
    _write(tmp_path / "labels.jsonl", labs)
    _write(tmp_path / "scores_jev.jsonl", sc)


@pytest.mark.parametrize(
    "n_high,auc,verdict", [(8, 0.80, "GO"), (7, 0.70, "PARTIAL"), (6, 0.60, "NO-GO")]
)
def test_report_decision_on_synthetic(tmp_path, n_high, auc, verdict):
    synthetic_pilot(tmp_path, n_high)
    md, dec = report.build_report(tmp_path, judges=["jev", "self"], n_boot=200)
    assert dec.auroc == pytest.approx(auc)
    assert dec.verdict == verdict
    assert f"**DECISION: {verdict}**" in md
    assert md.index("Decision rule") < md.index("**DECISION:")  # thresholds printed before numbers
    assert "| self | 0 | 0 | not scored" in md
    assert "By depth" in md and "| 1-2 | 20 | 10 |" in md
    assert "$0.0002" in md  # 20 calls x $1e-5
    assert (tmp_path / "figures" / "jev_latency.png").exists()


def test_report_flags_incomplete_labels(tmp_path):
    synthetic_pilot(tmp_path, 8)
    labs = (tmp_path / "labels.jsonl").read_text().splitlines()
    (tmp_path / "labels.jsonl").write_text("\n".join(labs[:10]) + "\n")
    md, _ = report.build_report(tmp_path, judges=["jev"], n_boot=50)
    assert "INCOMPLETE DATA" in md and "10 of 20" in md


def test_report_warns_on_skewed_labels():
    ctx = {
        "decision": decide(math.nan),
        "judges": {},
        "label_stats": {
            "n": 100,
            "frac_hard_pos": 0.97,
            "mean_soft": 0.8,
            "frac_terminal": 0.1,
            "soft_hist": {"0": 3, "1": 97},
        },
    }
    assert "AUROC is noisy" in report.render_report(ctx)
