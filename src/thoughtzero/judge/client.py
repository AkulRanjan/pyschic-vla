"""JevClient: transport (direct HTTP / OpenRouter / SDK), retries, semaphore.

Owner: Person 2 (Jagriti)."""

from __future__ import annotations

from typing import Any

from thoughtzero.config import JudgeCfg


class JevClient:
    def __init__(self, cfg: JudgeCfg) -> None:
        self.cfg = cfg

    async def ask(self, state: str, questions: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError("Person 2 (Jagriti): field names per docs/verified_apis.md")
