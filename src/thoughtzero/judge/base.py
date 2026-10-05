"""Judge protocol re-export + shared parsing helpers."""

import logging

from thoughtzero.types import Judge

logger = logging.getLogger(__name__)

EPS = 1e-3

__all__ = ["Judge", "fill_missing_options", "renormalize"]


def renormalize(probs: dict[str, float]) -> dict[str, float]:
    total = sum(probs.values())
    if total <= 0:
        n = len(probs)
        return {k: 1.0 / n for k in probs}
    return {k: v / total for k, v in probs.items()}


def fill_missing_options(probs: dict[str, float], option_keys: list[str]) -> dict[str, float]:
    """Fill any option missing from `probs` with EPS, then renormalize.
    If all options are missing, fall back to uniform and warn (spec A4.2).
    """
    present = {k: probs.get(k, EPS) for k in option_keys}
    if not any(k in probs for k in option_keys):
        logger.warning(
            "judge returned no recognizable options out of %s; using uniform priors", option_keys
        )
        n = len(option_keys)
        return {k: 1.0 / n for k in option_keys}
    return renormalize(present)
