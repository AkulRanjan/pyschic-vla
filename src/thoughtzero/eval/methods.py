"""Registry: method spec -> ``eval.runner.Method`` (SPEC.md §8.1). Owner: Person 4 (Harjas).

A method spec in ``eval.methods`` is ``name``, ``name:N`` or ``name:N/M``:

| spec        | what                                       | compute knob                 |
|-------------|--------------------------------------------|------------------------------|
| ``tz``      | MCTS with the configured judge (``judge``) | ``search.n_simulations``     |
| ``tz_self`` | B4: MCTS with GemmaSelfJudge               | ``search.n_simulations``     |
| ``tz_prm``  | B5: MCTS, self-judge prior + PRM value     | ``search.n_simulations``     |
| ``cot``     | B1: greedy chain-of-thought                | -                            |
| ``sc:N``    | B2: self-consistency over N samples        | N                            |
| ``sc:N/M``  | B2, voting N of M stored samples           | N (M kept for post-hoc curve)|
| ``bon:N``   | B3: best-of-N by ``judge.final_correct``   | N                            |
| ``c31b``    | C: greedy CoT on the configured generator  | run it with the 31B endpoint |

Ablations (prior-only, value-only, k, c_puct, ...) are ``tz`` with a config variant
(``configs/ablations.yaml`` + ``--variant``), not separate names.

``params`` label points on the accuracy-vs-compute plot and keep compute levels apart inside one
run folder (``method_id`` = ``name[param=value]``). With ``M > N``, ``raw`` keeps all M samples,
so ``eval.analysis.expand_sample_curves`` can evaluate every N' <= M without new GPU time.
B3 can reuse B2's stored samples (``stored_samples``) instead of sampling again.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
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


def parse_method_spec(spec: str) -> tuple[str, int | None, int | None]:
    """``"sc:8/64"`` -> ``("sc", 8, 64)``; ``"sc:8"`` -> ``("sc", 8, None)``; ``"tz"`` ->
    ``("tz", None, None)``."""
    name, _, rest = spec.strip().partition(":")
    if name not in KNOWN:
        raise KeyError(f"unknown method {name!r}; known: {KNOWN}")
    if name in SAMPLED and not rest:
        raise ValueError(f"{name} needs a sample count, e.g. '{name}:8'")
    if rest and name not in SAMPLED:
        raise ValueError(f"{name} takes no ':N' (its compute is set in the config)")
    if not rest:
        return name, None, None
    n_text, _, m_text = rest.partition("/")
    n, n_max = int(n_text), (int(m_text) if m_text else None)
    if n < 1 or (n_max is not None and n_max < n):
        raise ValueError(f"bad sample counts in {spec!r}: need 1 <= N <= M")
    return name, n, n_max


def judge_uses_jev(jcfg: JudgeCfg) -> bool:
    return jcfg.kind == "jev" or (
        jcfg.kind == "hybrid" and "jev" in (jcfg.prior_from, jcfg.value_from)
    )


def build_generator(cfg: Config, *, mock: bool = False) -> Generator:
    if mock:
        from thoughtzero.mocks import MockGenerator

        return MockGenerator(cfg.seed)
    from thoughtzero.llm.factory import make_generator

    return make_generator(cfg)


def judge_needs_generator(jcfg: JudgeCfg) -> bool:
    return jcfg.kind == "self" or (
        jcfg.kind == "hybrid" and "self" in (jcfg.prior_from, jcfg.value_from)
    )


def build_judge(
    jcfg: JudgeCfg, *, seed: int = 0, mock: bool = False, generator: Generator | None = None
) -> Judge:
    """Person 2's ``make_judge`` (or the mock). Raises MethodUnavailable while it is a stub."""
    if mock:
        from thoughtzero.mocks import MockJudge

        return MockJudge(seed)
    from thoughtzero.judge.factory import make_judge

    try:
        return make_judge(jcfg, generator)
    except (NotImplementedError, ValueError, TypeError) as e:  # stub / unknown kind / old API
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


def token_counter(generator: Generator, *, mock: bool = False) -> Callable[[str], int]:
    """The generator's own tokenizer (the paper's compute axis); approximate only for mocks."""
    tok = getattr(generator, "tokenizer", None)
    if tok is not None:
        return tok.count_tokens
    if mock:
        from thoughtzero.llm.tokenize import ApproxTokenizer

        return ApproxTokenizer().count_tokens
    raise MethodUnavailable("generator has no tokenizer to count completion tokens")


def where_c_ran(cfg: Config) -> str:
    """SPEC.md §8.1: record where C ran (model and endpoint; never a key)."""
    return f"{cfg.generator.model} @ {cfg.generator.base_url}"


def build_method(
    spec: str,
    cfg: Config,
    generator: Generator,
    *,
    mock: bool = False,
    tools: Toolkit | None = None,
    stored_samples: Mapping[str, Any] | None = None,
) -> Method:
    """Build one method from its spec.

    ``tools`` overrides answer extraction (mock runs, tests). ``stored_samples`` maps problem
    id -> ``SCSamples`` from a B2 run, so B3 reranks them instead of sampling again.
    """
    name, n, n_max = parse_method_spec(spec)
    m: Any
    if name in TZ_JUDGES:
        jcfg = cfg.judge.model_copy(update=TZ_JUDGES[name])
        judge = build_judge(jcfg, seed=cfg.seed, mock=mock, generator=generator)
        return TZMethod(generator, judge, cfg, tools, name=name, uses_jev=judge_uses_jev(jcfg))

    from thoughtzero.baselines.best_of_n import BestOfN
    from thoughtzero.baselines.cot import ChainOfThought
    from thoughtzero.baselines.self_consistency import SelfConsistency

    if name == "cot":
        m = ChainOfThought(generator, name="cot")
        m.params, m.uses_jev = {}, False
    elif name == "c31b":
        m = ChainOfThought(generator, where=where_c_ran(cfg), name="c31b")
        m.params, m.uses_jev = {}, False
    elif name == "sc":
        assert n is not None
        m = SelfConsistency(generator, n, token_counter(generator, mock=mock), n_max=n_max)
        m.params, m.uses_jev = {"n": n}, False
    else:  # bon
        assert n is not None
        judge = build_judge(cfg.judge, seed=cfg.seed, mock=mock, generator=generator)
        m = BestOfN(
            judge,
            n,
            generator=generator,
            count_tokens=token_counter(generator, mock=mock),
            stored=stored_samples,
            n_max=n_max,
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
    stored_samples: Mapping[str, Any] | None = None,
) -> list[Method]:
    return [
        build_method(s, cfg, generator, mock=mock, tools=tools, stored_samples=stored_samples)
        for s in specs
    ]


def load_stored_samples(run_dir: str, method: str = "sc") -> dict[str, Any]:
    """problem id -> ``SCSamples`` from a finished B2 run folder (for B3 reuse)."""
    from pathlib import Path

    from thoughtzero.baselines.self_consistency import SCSamples
    from thoughtzero.eval.runner import ROWS_FILE, dedupe_rows, read_jsonl

    out: dict[str, Any] = {}
    for r in dedupe_rows(read_jsonl(Path(run_dir) / ROWS_FILE)):
        if r["method"] == method and not r.get("error") and "answers" in (r.get("raw") or {}):
            out[r["problem_id"]] = SCSamples.from_raw(r["raw"])
    return out
