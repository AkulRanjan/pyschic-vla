"""make_judge(cfg) -> Judge (spec A4.5).

`cfg` is a plain dict here (no real pydantic config.py locally yet — see
docs/verified_apis.md). Expected shape:
    {"kind": "jev" | "uniform" | "constant" | "hybrid",
     "prior_mode": "choice" | "per_candidate_noul",   # jev only
     "root_value": 0.5,                                 # jev only
     "value": 0.5,                                       # constant only
     "prior_from": {...}, "value_from": {...}}           # hybrid only
"""

from typing import Any

from thoughtzero.judge.hybrid import HybridJudge
from thoughtzero.judge.jev import JevJudge
from thoughtzero.judge.uniform import ConstantValueJudge, UniformJudge
from thoughtzero.types import Judge


def make_judge(cfg: dict[str, Any]) -> Judge:
    kind = cfg["kind"]
    if kind == "jev":
        return JevJudge(
            prior_mode=cfg.get("prior_mode", "choice"),
            root_value=cfg.get("root_value", 0.5),
        )
    if kind == "uniform":
        return UniformJudge()
    if kind == "constant":
        return ConstantValueJudge(value=cfg.get("value", 0.5))
    if kind == "hybrid":
        return HybridJudge(
            prior_from=make_judge(cfg["prior_from"]),
            value_from=make_judge(cfg["value_from"]),
        )
    raise ValueError(f"unknown judge kind: {kind!r}")
