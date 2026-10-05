"""Typed config: YAML + ``${ENV}`` interpolation + ``--set a.b=c`` overrides.

Owner: Person 1 (Prakhar). Shared file: add fields to your own section by PR tagged to all.

YAML features:
- ``${VAR}`` is replaced from the environment (after ``load_dotenv``); missing → error.
- ``${VAR:-default}`` falls back to ``default`` when VAR is unset or empty.
- ``base: other.yaml`` deep-merges this file on top of ``other.yaml`` (path relative to file).

API keys never live in the config; they are read from the environment by the module that
needs them. ``config_hash`` therefore hashes the whole resolved config.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GeneratorCfg(_Section):
    # "completions": continue a partial assistant turn on a self-hosted server (vLLM / SGLang;
    # llm/gemma.py). "chat": hosted chat APIs such as OpenRouter (llm/chat_generator.py).
    api: Literal["completions", "chat"] = "completions"
    base_url: str = "http://localhost:8000/v1"
    model: str = "google/gemma-4-E4B-it"
    # HF tokenizer id when ``model`` isn't one (a quantized checkpoint or an Ollama tag);
    # empty/None: use ``model``
    tokenizer: str | None = None
    temperature: float = 0.9
    top_p: float = 0.95
    max_step_tokens: int = 256
    max_solution_tokens: int = 2048
    stop: list[str] = ["\n\nStep", "\n\n\n"]
    max_concurrency: int = 32
    # OpenRouter provider routing (sent as "provider" with generator.api=chat on OpenRouter).
    # 13 providers serve Gemma 4 26B at bf16 / fp8 / unknown precision, and not all support
    # logprobs, seed or stop: keep every call on full-precision providers that honour every
    # parameter sent (docs/verified_apis.md G12).
    provider: dict[str, Any] = {"quantizations": ["bf16"], "require_parameters": True}


class JudgeCfg(_Section):
    kind: Literal["jev", "self", "prm", "uniform", "hybrid"] = "jev"
    # Route to Jev (judge/client.py PROVIDERS; docs/verified_apis.md). direct / openrouter /
    # zen / vercel serve TypeSafe's jev-1.13; bocha is Bocha's own Jev-compatible model, and
    # local_stub is open-jev, a local open-source stand-in. Neither of the last two is
    # TypeSafe's model: development only, opt in explicitly.
    transport: Literal["direct", "openrouter", "zen", "vercel", "bocha", "local_stub"] = (
        "openrouter"
    )
    jev_model: str | None = None  # None: the route's default (pinned jev-1.13 where possible)
    jev_url: str | None = None  # full POST URL, overriding the route's default
    prior_mode: Literal["choice", "per_candidate_noul"] = "choice"
    # index into prompts.SOUND_VARIANTS (pilot prompt-sensitivity check, SPEC.md §7)
    sound_variant: int = Field(default=0, ge=0)
    max_concurrency: int = 16
    shuffle_options: bool = False
    root_value: float = 0.5
    usd_per_mtok: float = 0.042
    max_state_tokens: int = 28_000
    # hybrid only: which judges supply prior / value
    prior_from: str | None = None
    value_from: str | None = None
    prm_model: str = "Qwen/Qwen2.5-Math-PRM-7B"


class SearchCfg(_Section):
    n_simulations: int = 32
    k: int = 4
    c_puct: float = 1.5
    fpu_value: float = 0.0
    parallel_sims: int = 8
    virtual_loss: int = 1
    max_depth: int = 20
    extract_mode: Literal["most_visited", "value_vote"] = "most_visited"
    root_dirichlet_alpha: float | None = None
    dedupe_jaccard: float | None = 0.9


class BudgetCfg(_Section):
    max_usd: float = 5.0
    confirm_above_usd: float = 1.0


class EvalCfg(_Section):
    dataset: str = "math500"
    split: str = "test"
    n_problems: int | None = None
    subset_seed: int = 0
    methods: list[str] = ["tz"]
    concurrency: int = 4
    run_id: str | None = None
    out_dir: str = "results"


class PilotCfg(_Section):
    n_problems: int = 200
    # SPEC.md §7 uses MATH-500 ("test"); B11.1 proposes "train" to avoid leakage (team decides)
    source_split: str = "test"
    max_prefixes: int = 8
    m_completions: int = 8
    temperature: float = 0.7
    sensitivity_subset: int = 50


class Config(_Section):
    generator: GeneratorCfg = Field(default_factory=GeneratorCfg)
    judge: JudgeCfg = Field(default_factory=JudgeCfg)
    search: SearchCfg = Field(default_factory=SearchCfg)
    budget: BudgetCfg = Field(default_factory=BudgetCfg)
    eval: EvalCfg = Field(default_factory=EvalCfg)
    pilot: PilotCfg = Field(default_factory=PilotCfg)
    seed: int = 0
    # named override sets, e.g. {"k2": ["search.k=2"]}; used by ablation configs
    variants: dict[str, list[str]] = {}


class ConfigError(ValueError):
    pass


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _interpolate(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _interpolate(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate(v) for v in value]
    if not isinstance(value, str):
        return value

    def sub(m: re.Match[str]) -> str:
        name, default = m.group(1), m.group(2)
        env = os.environ.get(name)
        if env:
            return env
        if default is not None:
            return default
        raise ConfigError(f"environment variable {name} is not set (see .env.example)")

    return _ENV_PATTERN.sub(sub, value)


def _deep_merge(base: dict[str, Any], top: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, val in top.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def _read_yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    base = raw.pop("base", None)
    if base is not None:
        raw = _deep_merge(_read_yaml(path.parent / base), raw)
    return raw


def _apply_override(raw: dict[str, Any], override: str) -> None:
    if "=" not in override:
        raise ConfigError(f"override must look like a.b.c=value, got {override!r}")
    dotted, text = override.split("=", 1)
    keys = dotted.strip().split(".")
    node = raw
    for key in keys[:-1]:
        node = node.setdefault(key, {})
        if not isinstance(node, dict):
            raise ConfigError(f"cannot set {dotted}: {key} is not a section")
    node[keys[-1]] = yaml.safe_load(text)


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    """Load ``path`` (or pure defaults when None), interpolate env vars, apply overrides."""
    load_dotenv()
    raw = _read_yaml(Path(path)) if path is not None else {}
    for ov in overrides or []:
        _apply_override(raw, ov)
    return Config.model_validate(_interpolate(raw))


def apply_variant(cfg: Config, name: str) -> Config:
    if name not in cfg.variants:
        raise ConfigError(f"unknown variant {name!r}; known: {sorted(cfg.variants)}")
    raw = cfg.model_dump()
    for ov in cfg.variants[name]:
        _apply_override(raw, ov)
    return Config.model_validate(raw)


def config_hash(cfg: Config) -> str:
    canonical = json.dumps(cfg.model_dump(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def dump_resolved(cfg: Config, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(yaml.safe_dump(cfg.model_dump(), sort_keys=False), encoding="utf-8")


def add_config_args(parser: argparse.ArgumentParser, default: str | None = None) -> None:
    """Standard ``--config`` / ``--set`` flags for every script."""
    parser.add_argument("--config", default=default, help="YAML config file")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config value, e.g. --set search.k=6 (repeatable)",
    )


def config_from_args(args: argparse.Namespace) -> Config:
    return load_config(args.config, args.overrides)
