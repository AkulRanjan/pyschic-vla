"""Gemma generator over an OpenAI-compatible COMPLETIONS endpoint (SPEC.md §5.2).

Owner: Person 3 (Akul). Implements ``types.Generator``.
Must call ``accounting.record_gemma`` once per request.
"""

from __future__ import annotations

from thoughtzero.config import GeneratorCfg
from thoughtzero.types import GenOut


class OpenAICompatibleGenerator:
    def __init__(self, cfg: GeneratorCfg) -> None:
        self.cfg = cfg

    async def propose(self, problem: str, steps: list[str], k: int) -> list[GenOut]:
        raise NotImplementedError("Person 3 (Akul)")

    async def complete(self, problem: str, steps: list[str], temperature: float = 0.0) -> list[str]:
        raise NotImplementedError("Person 3 (Akul)")

    async def sample_completions(
        self, problem: str, steps: list[str], n: int, temperature: float
    ) -> list[list[str]]:
        raise NotImplementedError("Person 3 (Akul)")

    async def sample_solutions(self, problem: str, n: int, temperature: float) -> list[str]:
        raise NotImplementedError("Person 3 (Akul)")

    async def raw_completion(
        self, prompt: str, max_tokens: int = 1, logprobs: int | None = None
    ) -> dict[str, object]:
        """Raw completions call (with logprobs) for Person 2 (Jagriti)'s GemmaSelfJudge."""
        raise NotImplementedError("Person 3 (Akul)")
