"""Transport layer for Jev-shaped judges (spec A4.1).

Three transports, selected by `judge.transport`:
  - "local_stub": the self-hosted open-jev stand-in (com-kotobalabs/open-jev-deberta-v3-large),
    used while real TypeSafe access is pending. In-process, no network, no key.
  - "direct" / "openrouter": real TypeSafe Jev. NOT YET IMPLEMENTED — the
    waitlist/key are still pending (see docs/verified_apis.md). Calling them
    raises NotImplementedError with a pointer to that doc, rather than
    guessing field names (spec §13.5).
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Any

MODEL_NAME = "open-jev-deberta-v3-large"


@dataclass
class JevResponse:
    answers: list[dict[str, Any]]
    latency_ms: float
    input_tokens: int
    model: str = MODEL_NAME
    cache_hit: bool = False


def _estimate_tokens(state: str, questions: list[dict[str, Any]]) -> int:
    chars = len(state) + sum(len(str(q)) for q in questions)
    return max(1, chars // 4)


class _LocalStubBackend:
    """Lazily loads the stand-in model once per process, runs the (sync,
    torch) forward pass in a thread so callers stay async.
    """

    def __init__(self) -> None:
        self._model = None
        self._lock = asyncio.Lock()

    async def _get_model(self):
        if self._model is None:
            async with self._lock:
                if self._model is None:
                    from typed_decisions.open_jev import OpenJev

                    self._model = OpenJev.from_pretrained(
                        "com-kotobalabs/open-jev-deberta-v3-large"
                    )
        return self._model

    async def ask(self, state: str, questions: list[dict[str, Any]]) -> JevResponse:
        model = await self._get_model()
        t0 = time.monotonic()
        answers = await asyncio.to_thread(model.decide, state, questions)
        latency_ms = (time.monotonic() - t0) * 1000
        return JevResponse(
            answers=answers,
            latency_ms=latency_ms,
            input_tokens=_estimate_tokens(state, questions),
        )


class JevClient:
    """`async def ask(state, questions) -> JevResponse`, per spec A4.1.

    `questions` is a list of {"id", "type", "instructions", "options"?}
    dicts (the local_stub library's shape). Concurrency is capped by an
    internal semaphore; the local_stub backend doesn't need retries (no
    network), so tenacity-based retry/backoff is added only once the real
    "direct"/"openrouter" transports are implemented.
    """

    def __init__(self, transport: str = "local_stub", max_concurrency: int = 16) -> None:
        self.transport = transport
        self._semaphore = asyncio.Semaphore(max_concurrency)
        if transport == "local_stub":
            self._backend: _LocalStubBackend | None = _LocalStubBackend()
        else:
            self._backend = None

    async def ask(self, state: str, questions: list[dict[str, Any]]) -> JevResponse:
        if self.transport != "local_stub":
            raise NotImplementedError(
                f"transport={self.transport!r} is not implemented yet: real TypeSafe "
                "access (waitlist / OpenRouter key) is still pending. See "
                "docs/verified_apis.md -> 'Jev (TypeSafe AI) - direct/OpenRouter access'. "
                "Use transport='local_stub' for now."
            )
        async with self._semaphore:
            assert self._backend is not None
            return await self._backend.ask(state, questions)
