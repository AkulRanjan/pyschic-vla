from pathlib import Path

import pytest

from thoughtzero.config import (
    Config,
    ConfigError,
    apply_variant,
    config_hash,
    dump_resolved,
    load_config,
)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMMA_BASE_URL", raising=False)
    monkeypatch.delenv("GEMMA_MODEL", raising=False)
    monkeypatch.setattr("thoughtzero.config.load_dotenv", lambda: None)


def test_defaults_without_file() -> None:
    cfg = load_config()
    assert cfg.search.c_puct == 1.5
    assert cfg.search.k == 4
    assert cfg.generator.stop == ["\n\nStep", "\n\n\n"]


def test_default_yaml_loads_without_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # ignore the developer's .env: this checks the defaults written in default.yaml
    for var in ("GEMMA_BASE_URL", "GEMMA_MODEL", "JEV_TRANSPORT"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("thoughtzero.config.load_dotenv", lambda *a, **k: False)
    cfg = load_config(CONFIGS / "default.yaml")
    assert cfg.generator.base_url == "http://localhost:8000/v1"
    assert cfg.judge.transport == "openrouter" and cfg.judge.jev_model is None
    from thoughtzero.judge.client import resolve_route

    assert resolve_route(cfg.judge)[1] == "typesafe/jev-1.13"  # the route pins jev-1.13


def test_env_interpolation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMMA_BASE_URL", "http://gpu:9000/v1")
    cfg = load_config(CONFIGS / "default.yaml")
    assert cfg.generator.base_url == "http://gpu:9000/v1"


def test_missing_env_without_default_errors(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text("generator:\n  base_url: ${THOUGHTZERO_SURELY_UNSET}\n")
    with pytest.raises(ConfigError, match="THOUGHTZERO_SURELY_UNSET"):
        load_config(p)


def test_overrides_are_typed() -> None:
    cfg = load_config(
        CONFIGS / "default.yaml",
        [
            "search.k=6",
            "search.c_puct=3.0",
            "eval.methods=[tz, sc]",
            "search.root_dirichlet_alpha=null",
        ],
    )
    assert cfg.search.k == 6
    assert cfg.search.c_puct == 3.0
    assert cfg.eval.methods == ["tz", "sc"]
    assert cfg.search.root_dirichlet_alpha is None


def test_unknown_key_rejected() -> None:
    with pytest.raises(ValueError):
        load_config(None, ["search.cpuct=1.0"])


def test_bad_override_syntax() -> None:
    with pytest.raises(ConfigError):
        load_config(None, ["search.k"])


def test_base_merge() -> None:
    cfg = load_config(CONFIGS / "pilot.yaml")
    assert cfg.pilot.m_completions == 8
    assert cfg.budget.max_usd == 1.0  # overridden
    assert cfg.search.c_puct == 1.5  # inherited from default.yaml


def test_all_shipped_configs_load() -> None:
    for path in CONFIGS.glob("*.yaml"):
        load_config(path)


def test_variants_apply() -> None:
    cfg = load_config(CONFIGS / "ablations.yaml")
    k2 = apply_variant(cfg, "k2")
    assert k2.search.k == 2 and cfg.search.k == 4
    with pytest.raises(ConfigError):
        apply_variant(cfg, "nope")


def test_hash_stable_and_sensitive() -> None:
    a, b = load_config(), load_config()
    assert config_hash(a) == config_hash(b)
    assert config_hash(a) != config_hash(load_config(None, ["search.k=5"]))


def test_sections_are_not_shared_between_instances() -> None:
    a, b = Config(), Config()
    a.generator.stop.append("X")
    assert b.generator.stop == ["\n\nStep", "\n\n\n"]


def test_dump_roundtrip(tmp_path: Path) -> None:
    cfg = load_config(None, ["search.k=3"])
    out = tmp_path / "run" / "config.yaml"
    dump_resolved(cfg, out)
    assert config_hash(load_config(out)) == config_hash(cfg)


def test_gemma_tokenizer_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("thoughtzero.config.load_dotenv", lambda *a, **k: False)
    monkeypatch.delenv("GEMMA_TOKENIZER", raising=False)
    assert not load_config(CONFIGS / "default.yaml").generator.tokenizer  # unset: use model
    monkeypatch.setenv("GEMMA_TOKENIZER", "google/gemma-4-E4B-it")
    assert load_config(CONFIGS / "default.yaml").generator.tokenizer == "google/gemma-4-E4B-it"
