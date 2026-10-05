"""Judge helpers.

Owner: Person 2 (Jagriti). The ``Judge`` protocol itself lives in ``thoughtzero.types``."""

from __future__ import annotations

import logging

from thoughtzero.types import Judge

logger = logging.getLogger(__name__)

__all__ = ["Judge", "renormalize"]


def renormalize(probs: list[float | None], eps: float = 1e-3) -> list[float]:
    """Fill missing entries with ``eps`` and rescale to sum to 1 (uniform if all missing)."""
    if all(p is None for p in probs):
        logger.warning(
            "judge returned no recognizable options out of %d; using uniform priors", len(probs)
        )
        n = len(probs)
        return [1.0 / n] * n
    filled = [p if p is not None else eps for p in probs]
    total = sum(filled)
    if total <= 0:
        n = len(filled)
        return [1.0 / n] * n
    return [p / total for p in filled]
