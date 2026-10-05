"""Build any judge from config. Owner: Person 2 (Jagriti).

``make_judge`` raises ``NotImplementedError`` for judges that aren't built yet, so
``eval/methods.py::build_judge`` reports them as unavailable and skips them instead of
erroring on every call.
"""

from __future__ import annotations

from typing import Any

from thoughtzero.config import JudgeCfg
from thoughtzero.judge.hybrid import HybridJudge
from thoughtzero.judge.jev import JevJudge
from thoughtzero.judge.uniform import ConstantValueJudge, UniformJudge
from thoughtzero.types import Judge

# Hybrid-only sub-judge (configs/ablations.yaml: prior_only uses value_from=constant).
# Not a top-level `judge.kind`.
CONSTANT_VALUE = 0.5


def make_judge(cfg: JudgeCfg, generator: Any = None) -> Judge:
    """``generator`` is needed only by ``kind="self"`` (also as a hybrid sub-judge): any
    generator with ``first_token_logprobs`` (both ``llm`` generator classes)."""
    if cfg.kind == "jev":
        return JevJudge(cfg)
    if cfg.kind == "self":
        if generator is None or not hasattr(generator, "first_token_logprobs"):
            raise NotImplementedError("the self judge needs a generator with logprobs")
        from thoughtzero.judge.self_judge import GemmaSelfJudge

        return GemmaSelfJudge(cfg, generator)
    if cfg.kind == "prm":
        # needs a local GPU for Qwen2.5-Math-PRM-7B (PLAN.md D7)
        raise NotImplementedError("PRMJudge (B5) is not implemented (PLAN.md D7)")
    if cfg.kind == "uniform":
        return UniformJudge()
    if cfg.kind == "hybrid":
        if cfg.prior_from is None or cfg.value_from is None:
            raise ValueError("hybrid judge needs judge.prior_from and judge.value_from set")
        return HybridJudge(
            prior_from=_sub_judge(cfg, cfg.prior_from, generator),
            value_from=_sub_judge(cfg, cfg.value_from, generator),
        )
    raise ValueError(f"unknown judge kind: {cfg.kind!r}")


def _sub_judge(cfg: JudgeCfg, name: str, generator: Any = None) -> Judge:
    if name == "constant":
        return ConstantValueJudge(CONSTANT_VALUE)
    if name == "hybrid":
        raise ValueError("a hybrid judge's prior_from/value_from cannot itself be 'hybrid'")
    return make_judge(cfg.model_copy(update={"kind": name}), generator)
