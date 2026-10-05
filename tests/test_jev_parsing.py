"""Offline parsing tests (spec A4.6) — no network, no model load.
Exercises JevJudge's parsing against the real fixtures recorded in
tests/fixtures/jev/ (see docs/verified_apis.md).
"""

import json
from pathlib import Path

import pytest

from thoughtzero.judge.base import fill_missing_options, renormalize
from thoughtzero.judge.jev import JevJudge, MissingDistributionError

FIXTURES = Path(__file__).parent / "fixtures" / "jev"


def load(name: str):
    return json.loads((FIXTURES / f"{name}.json").read_text())


def test_parse_sound_from_fixture():
    answers = load("noul_basic")
    assert JevJudge._parse_sound(answers[0]) == pytest.approx(0.8237845301628113)


def test_parse_choice_sums_to_one_from_fixture():
    answers = load("choice_4_options")
    priors = JevJudge._parse_choice(answers, ["c0", "c1", "c2", "c3"])
    assert sum(priors.values()) == pytest.approx(1.0, abs=1e-6)
    assert set(priors) == {"c0", "c1", "c2", "c3"}


def test_parse_choice_missing_option_is_eps_filled_and_renormalized():
    answers = [{"choice": "c0", "probabilities": {"c0": 0.9, "c1": 0.1}}]
    priors = JevJudge._parse_choice(answers, ["c0", "c1", "c2"])
    assert sum(priors.values()) == pytest.approx(1.0, abs=1e-6)
    assert priors["c2"] < priors["c0"]


def test_parse_choice_top_pick_only_raises_clear_error():
    answers = [{"choice": "c0"}]  # no "probabilities" key
    with pytest.raises(MissingDistributionError, match="prior_mode"):
        JevJudge._parse_choice(answers, ["c0", "c1"])


def test_parse_per_candidate_noul():
    answers = [{"noul": 0.3}, {"noul": 0.1}, {"noul": 0.5}]
    priors = JevJudge._parse_per_candidate_noul(answers, ["c0", "c1", "c2"], offset=0)
    assert sum(priors.values()) == pytest.approx(1.0, abs=1e-6)
    assert priors["c2"] > priors["c1"]


def test_parse_per_candidate_noul_all_zero_falls_back_to_uniform():
    answers = [{"noul": 0.0}, {"noul": 0.0}]
    priors = JevJudge._parse_per_candidate_noul(answers, ["c0", "c1"], offset=0)
    assert priors == {"c0": 0.5, "c1": 0.5}


def test_malformed_response_gives_clear_error():
    with pytest.raises(ValueError, match="expected a noul answer"):
        JevJudge._parse_sound({"not_noul": 1.0})


def test_renormalize_uniform_on_zero_total():
    assert renormalize({"a": 0.0, "b": 0.0}) == {"a": 0.5, "b": 0.5}


def test_fill_missing_options_all_missing_warns_and_uniform(caplog):
    result = fill_missing_options({}, ["a", "b"])
    assert result == {"a": 0.5, "b": 0.5}
    assert "uniform priors" in caplog.text
