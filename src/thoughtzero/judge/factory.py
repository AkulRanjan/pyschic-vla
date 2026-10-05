"""Build any judge from config. Owner: Person 2 (Jagriti).

``make_judge`` raises ``NotImplementedError`` for judges that aren't built yet, so
``eval/methods.py::build_judge`` reports them as unavailable and skips them instead of
erroring on every call.
"""

from __future__ import annotations

from thoughtzero.config import JudgeCfg
from thoughtzero.judge.hybrid import HybridJudge
from thoughtzero.judge.jev import JevJudge
from thoughtzero.judge.uniform import ConstantValueJudge, UniformJudge
from thoughtzero.types import Judge

# Hybrid-only sub-judge (configs/ablations.yaml: prior_only uses value_from=constant).
# Not a top-level `judge.kind`.
CONSTANT_VALUE = 0.5


def make_judge(cfg: JudgeCfg) -> Judge:
    if cfg.kind == "jev":
        return JevJudge(cfg)
    if cfg.kind == "self":
        raise NotImplementedError("GemmaSelfJudge (B4) is not implemented yet (team file A5.2)")
    if cfg.kind == "prm":
        raise NotImplementedError("PRMJudge (B5) is not implemented yet (team file A5.3)")
    if cfg.kind == "uniform":
        return UniformJudge()
    if cfg.kind == "hybrid":
        if cfg.prior_from is None or cfg.value_from is None:
            raise ValueError("hybrid judge needs judge.prior_from and judge.value_from set")
        return HybridJudge(
            prior_from=_sub_judge(cfg, cfg.prior_from),
            value_from=_sub_judge(cfg, cfg.value_from),
        )
    raise ValueError(f"unknown judge kind: {cfg.kind!r}")


def _sub_judge(cfg: JudgeCfg, name: str) -> Judge:
    if name == "constant":
        return ConstantValueJudge(CONSTANT_VALUE)
    if name == "hybrid":
        raise ValueError("a hybrid judge's prior_from/value_from cannot itself be 'hybrid'")
    return make_judge(cfg.model_copy(update={"kind": name}))
