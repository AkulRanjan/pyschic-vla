"""Registry: method spec -> ``eval.runner.Method`` (SPEC.md §8.1). Owner: Person 4 (Harjas).

A method spec in ``eval.methods`` is ``name`` or ``name:N``:

| spec        | what                                       | compute knob                 |
|-------------|--------------------------------------------|------------------------------|
| ``tz``      | MCTS with the configured judge (``judge``) | ``search.n_simulations``     |
| ``tz_self`` | B4: MCTS with GemmaSelfJudge               | ``search.n_simulations``     |
| ``tz_prm``  | B5: MCTS, self-judge prior + PRM value     | ``search.n_simulations``     |
| ``cot``     | B1: greedy chain-of-thought                | -                            |
| ``sc:N``    | B2: self-consistency over N samples        | N                            |
| ``bon:N``   | B3: best-of-N by ``judge.final_correct``   | N                            |
| ``c31b``    | C: greedy CoT on the configured generator  | run it with the 31B endpoint |

Ablations (prior-only, value-only, k, c_puct, ...) are ``tz`` with a config variant
(``configs/ablations.yaml`` + ``--variant``), not separate names.

``params`` label points on the accuracy-vs-compute plot and keep compute levels apart inside one
run folder (``method_id`` = ``name[param=value]``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from thoughtzero.config import Config, JudgeCfg
from thoughtzero.eval.runner import Method, MethodResult
from thoughtzero.eval.toolkit import Toolkit
from thoughtzero.types import Generator, Judge, Problem


class MethodUnavailable(RuntimeError):
    """The owner's code for this method has not landed (or its constructor differs)."""


TZ_JUDGES: dict[str, dict[str, Any]] = {
    "tz": {},  # whatever cfg.judge says (default: jev); ablation variants edit cfg.judge
    "tz_self": {"kind": "self"},
    "tz_prm": {"kind": "hybrid", "prior_from": "self", "value_from": "prm"},
}
SAMPLED = {"sc", "bon"}  # require ":N"
KNOWN = [*TZ_JUDGES, "cot", "sc", "bon", "c31b"]


def parse_method_spec(spec: str) -> tuple[str, int | None]:
    """``"sc:8"`` -> ``("sc", 8)``; ``"tz"`` -> ``("tz", None)``."""
    name, _, n = spec.strip().partition(":")
    if name not in KNOWN:
        raise KeyError(f"unknown method {name!r}; known: {KNOWN}")
    if name in SAMPLED and not n:
        raise ValueError(f"{name} needs a sample count, e.g. '{name}:8'")
    if n and name not in SAMPLED:
        raise ValueError(f"{name} takes no ':N' (its compute is set in the config)")
    return name, (int(n) if n else None)


def judge_uses_jev(jcfg: JudgeCfg) -> bool:
    return jcfg.kind == "jev" or (
        jcfg.kind == "hybrid" and "jev" in (jcfg.prior_from, jcfg.value_from)
    )


def build_generator(cfg: Config, *, mock: bool = False) -> Generator:
    if mock:
        from thoughtzero.mocks import MockGenerator

        return MockGenerator(cfg.seed)
    from thoughtzero.llm.gemma import OpenAICompatibleGenerator

    return OpenAICompatibleGenerator(cfg.generator)


def build_judge(jcfg: JudgeCfg, *, seed: int = 0, mock: bool = False) -> Judge:
    """Person 2's ``make_judge`` (or the mock). Raises MethodUnavailable while it is a stub."""
    if mock:
        from thoughtzero.mocks import MockJudge

        return MockJudge(seed)
    from thoughtzero.judge.factory import make_judge

    try:
        return make_judge(jcfg)
    except NotImplementedError as e:
        raise MethodUnavailable(f"judge kind={jcfg.kind!r} not available yet: {e}") from e


# --------------------------------------------------------------------------- ThoughtZero


@dataclass
class TZMethod:
    """Wraps Person 1's ``search.mcts.search``. Sees only ``problem.question``."""

    generator: Generator
    judge: Judge
    cfg: Config
    tools: Toolkit | None = None  # None: search's defaults (data.grading)
    name: str = "tz"
    uses_jev: bool = True
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.params = {"n_simulations": self.cfg.search.n_simulations}

    async def solve(self, problem: Problem) -> MethodResult:
        from thoughtzero.search.mcts import search

        res = await search(
            problem.question,
            self.generator,
            self.judge,
            self.cfg.search,
            seed=self.cfg.seed,
            extract=self.tools.extract if self.tools else None,
            normalize=self.tools.normalize if self.tools else None,
        )
        return MethodResult(
            res.answer,
            {"answer_steps": res.answer_steps, "stats": res.stats, "tree": res.tree_dump},
        )


# --------------------------------------------------------------------------- registry


def _baseline(module: str, cls_name: str, **kwargs: Any) -> Method:
    """Instantiate Person 3's baseline class ``cls(generator=..., cfg=..., [judge=..., n=...])``."""
    import importlib

    cls: Any = getattr(importlib.import_module(module), cls_name)
    try:
        return cls(**kwargs)
    except TypeError as e:
        raise MethodUnavailable(
            f"{module}.{cls_name} does not accept {sorted(kwargs)} yet (Person 3): {e}"
        ) from e


def build_method(
    spec: str,
    cfg: Config,
    generator: Generator,
    *,
    mock: bool = False,
    tools: Toolkit | None = None,
) -> Method:
    """Build one method from its spec. ``tools`` overrides extraction (mock runs, tests)."""
    name, n = parse_method_spec(spec)
    m: Any
    if name in TZ_JUDGES:
        jcfg = cfg.judge.model_copy(update=TZ_JUDGES[name])
        judge = build_judge(jcfg, seed=cfg.seed, mock=mock)
        m = TZMethod(generator, judge, cfg, tools, name=name, uses_jev=judge_uses_jev(jcfg))
        return m
    if name in ("cot", "c31b"):
        m = _baseline("thoughtzero.baselines.cot", "ChainOfThought", generator=generator, cfg=cfg)
        m.params, m.uses_jev = {}, False
    elif name == "sc":
        m = _baseline(
            "thoughtzero.baselines.self_consistency",
            "SelfConsistency",
            generator=generator,
            cfg=cfg,
            n=n,
        )
        m.params, m.uses_jev = {"n": n}, False
    else:  # bon
        judge = build_judge(cfg.judge, seed=cfg.seed, mock=mock)
        m = _baseline(
            "thoughtzero.baselines.best_of_n",
            "BestOfN",
            generator=generator,
            judge=judge,
            cfg=cfg,
            n=n,
        )
        m.params, m.uses_jev = {"n": n}, judge_uses_jev(cfg.judge)
    m.name = name  # the registry name is the canonical name in results
    return m


def build_methods(
    specs: list[str],
    cfg: Config,
    generator: Generator,
    *,
    mock: bool = False,
    tools: Toolkit | None = None,
) -> list[Method]:
    return [build_method(s, cfg, generator, mock=mock, tools=tools) for s in specs]
