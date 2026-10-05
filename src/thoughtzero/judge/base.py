"""Judge helpers.

Owner: Person 2 (Jagriti). The ``Judge`` protocol itself lives in ``thoughtzero.types``."""

from __future__ import annotations

from thoughtzero.types import Judge

__all__ = ["Judge", "renormalize"]


def renormalize(probs: list[float | None], eps: float = 1e-3) -> list[float]:
    """Fill missing entries with ``eps`` and rescale to sum to 1 (uniform if all missing)."""
    raise NotImplementedError("Person 2 (Jagriti)")
