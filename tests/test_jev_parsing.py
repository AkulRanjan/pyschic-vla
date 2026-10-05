"""Offline parsing tests (spec A4.6) — no network, no model load.
Exercises JevJudge's parsing against real fixtures recorded in
tests/fixtures/jev/ (see docs/verified_apis.md), through the local_stub
transport's actual request/response shape.
"""

import json
from pathlib import Path

import pytest

from thoughtzero.config import JudgeCfg
from thoughtzero.judge.base import renormalize
from thoughtzero.judge.jev import JevJudge, MissingDistributionError

FIXTURES = Path(__file__).parent / "fixtures" / "jev"


def load(name: str):
    return json.loads((FIXTURES / f"{name}.json").read_text())


def make_judge(**overrides) -> JevJudge:
    return JevJudge(JudgeCfg(transport="local_stub", **overrides))


def test_parse_sound_from_fixture():
    answers = load("noul_basic")
    assert float(answers["sound"]["noul"]) == pytest.approx(0.5071545243263245)


def test_parse_choice_sums_to_one_from_fixture():
    judge = make_judge()
    answers = load("choice_4_options")
    priors = judge._parse_priors(answers, 4)
    assert sum(priors) == pytest.approx(1.0, abs=1e-6)
    assert len(priors) == 4


def test_parse_choice_missing_option_is_eps_filled_and_renormalized():
    judge = make_judge()
    answers = {"next": {"choice": "c0", "probabilities": {"c0": 0.9, "c1": 0.1}}}
    priors = judge._parse_priors(answers, 3)
    assert sum(priors) == pytest.approx(1.0, abs=1e-6)
    assert priors[2] < priors[0]


def test_parse_choice_top_pick_only_raises_clear_error():
    judge = make_judge()
    answers = {"next": {"choice": "c0"}}  # no "probabilities" key
    with pytest.raises(MissingDistributionError, match="prior_mode"):
        judge._parse_priors(answers, 2)


def test_parse_per_candidate_noul():
    judge = make_judge(prior_mode="per_candidate_noul")
    answers = {"cand_0": {"noul": 0.3}, "cand_1": {"noul": 0.1}, "cand_2": {"noul": 0.5}}
    priors = judge._parse_priors(answers, 3)
    assert sum(priors) == pytest.approx(1.0, abs=1e-6)
    assert priors[2] > priors[1]


def test_parse_per_candidate_noul_all_zero_falls_back_to_uniform():
    judge = make_judge(prior_mode="per_candidate_noul")
    answers = {"cand_0": {"noul": 0.0}, "cand_1": {"noul": 0.0}}
    priors = judge._parse_priors(answers, 2)
    assert priors == [0.5, 0.5]


def test_malformed_response_missing_key_gives_clear_error():
    judge = make_judge()
    with pytest.raises(KeyError):
        judge._parse_priors({}, 2)


def test_renormalize_uniform_on_zero_total():
    assert renormalize([0.0, 0.0]) == [0.5, 0.5]


def test_renormalize_all_missing_warns_and_uniform(caplog):
    result = renormalize([None, None, None])
    assert result == [pytest.approx(1 / 3)] * 3
    assert "uniform priors" in caplog.text


def test_renormalize_eps_fills_single_missing():
    result = renormalize([0.9, None])
    assert sum(result) == pytest.approx(1.0, abs=1e-6)
    assert result[1] < result[0]
