"""sha256-keyed disk cache for Jev responses (SPEC.md §5.4). Owner: Person 2 (Jagriti).

Planned CLI: ``python -m thoughtzero.judge.cache export|import <file.jsonl>``.
"""

from __future__ import annotations

from typing import Any


class DiskCache:
    def __init__(self, directory: str) -> None:
        self.directory = directory

    def get(self, key: str) -> dict[str, Any] | None:
        raise NotImplementedError("Person 2 (Jagriti)")

    def set(self, key: str, value: dict[str, Any]) -> None:
        raise NotImplementedError("Person 2 (Jagriti)")
