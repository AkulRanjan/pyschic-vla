"""Candidate step normalization + merging (SPEC.md §6.3). Owner: Person 1 (Prakhar).

The search calls the judge on the UNIQUE candidates only (fewer judge tokens), so each unique
candidate gets its own prior directly. If a judge is ever called on all k originals instead,
``merge_priors`` sums the originals' priors into their unique slots, as §6.3 describes.
"""

from __future__ import annotations

import re

from thoughtzero.types import GenOut

_STEP_PREFIX = re.compile(r"^\s*step\s*\d+\s*[:.)]\s*", re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Lowercase, drop a repeated ``Step n:`` prefix, collapse whitespace, trim end punctuation."""
    text = _STEP_PREFIX.sub("", text.lower())
    return _WHITESPACE.sub(" ", text).strip().rstrip(".;,").strip()


def jaccard(a: str, b: str) -> float:
    """Token Jaccard similarity on whitespace tokens (1.0 for two empty strings)."""
    ta, tb = set(a.split()), set(b.split())
    if not ta and not tb:
        return 1.0
    return len(ta & tb) / len(ta | tb)


def dedupe(cands: list[GenOut], jaccard_min: float | None = 0.9) -> tuple[list[GenOut], list[int]]:
    """Unique candidates (first occurrence kept) + mapping original index -> unique index.

    Empty candidates map to -1 (dropped). With ``jaccard_min`` set, a candidate whose
    normalized token Jaccard with an earlier unique one is >= ``jaccard_min`` is merged into it.
    """
    unique: list[GenOut] = []
    unique_norm: list[str] = []
    mapping: list[int] = []
    for cand in cands:
        norm = normalize(cand.text)
        if not norm:
            mapping.append(-1)
            continue
        match = next(
            (
                j
                for j, other in enumerate(unique_norm)
                if norm == other
                or (jaccard_min is not None and jaccard(norm, other) >= jaccard_min)
            ),
            None,
        )
        if match is None:
            mapping.append(len(unique))
            unique.append(cand)
            unique_norm.append(norm)
        else:
            mapping.append(match)
    return unique, mapping


def merge_priors(priors: list[float], mapping: list[int], n_unique: int) -> list[float]:
    """Sum per-original priors into unique slots, renormalized (dropped originals ignored)."""
    if len(priors) != len(mapping):
        raise ValueError("priors and mapping must have the same length")
    merged = [0.0] * n_unique
    for p, j in zip(priors, mapping, strict=True):
        if j >= 0:
            merged[j] += p
    total = sum(merged)
    if total <= 0:
        return [1.0 / n_unique] * n_unique if n_unique else []
    return [m / total for m in merged]
