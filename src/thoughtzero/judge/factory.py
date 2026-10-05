"""Build any judge from config. Owner: Person 2 (Jagriti)."""

from __future__ import annotations

from thoughtzero.config import JudgeCfg
from thoughtzero.judge.hybrid import HybridJudge
from thoughtzero.judge.jev import JevJudge
from thoughtzero.judge.prm import PRMJudge
from thoughtzero.judge.self_judge import GemmaSelfJudge
from thoughtzero.judge.uniform import ConstantValueJudge, UniformJudge
from thoughtzero.types import Judge


def make_judge(cfg: JudgeCfg) -> Judge:
    if cfg.kind == "jev":
        return JevJudge(cfg)
    if cfg.kind == "self":
        return GemmaSelfJudge(cfg)
    if cfg.kind == "prm":
        return PRMJudge(cfg)
    if cfg.kind == "uniform":
        return UniformJudge()
    if cfg.kind == "hybrid":
        if cfg.prior_from is None or cfg.value_from is None:
            raise ValueError("hybrid judge needs judge.prior_from and judge.value_from set")
        prior_judge = make_judge(cfg.model_copy(update={"kind": cfg.prior_from}))
        value_judge = make_judge(cfg.model_copy(update={"kind": cfg.value_from}))
        return HybridJudge(prior_from=prior_judge, value_from=value_judge)
    raise ValueError(f"unknown judge kind: {cfg.kind!r}")


def make_constant_value_judge(value: float = 0.5) -> ConstantValueJudge:
    """Not reachable from config (there's no 'constant' kind), but used
    directly by ablations: prior-only = HybridJudge(jev, constant(0.5))."""
    return ConstantValueJudge(value)
