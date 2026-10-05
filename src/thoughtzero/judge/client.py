"""JevClient: transport (direct HTTP / OpenRouter / SDK / local stand-in), retries, semaphore.

Owner: Person 2 (Jagriti).

``direct``/``openrouter``/``sdk`` talk to real TypeSafe Jev and are not
implemented yet: the waitlist/key are still pending, and the exact request
field names are an open VERIFY item (SPEC.md §13.5 — no guessing). See
docs/verified_apis.md. ``local_stub`` is fully implemented: it runs
com-kotobalabs/open-jev-deberta-v3-large (Apache-2.0) in-process, so
JevJudge can be developed and tested for real while that access is pending.

The retry/semaphore machinery for the HTTP transports (the part that's
NOT TypeSafe-specific) is implemented and tested now via
``request_with_retry``, so it's ready to use once the real payload shape
is confirmed.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_random_exponential

from thoughtzero.config import JudgeCfg

RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class JevHTTPError(Exception):
    def __init__(self, status_code: int, body: str) -> None:
        super().__init__(f"Jev request failed with {status_code}: {body[:500]}")
        self.status_code = status_code
        self.body = body


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.TimeoutException):
        return True
    if isinstance(exc, JevHTTPError):
        return exc.status_code in RETRYABLE_STATUS
    return False


async def request_with_retry(
    client: httpx.AsyncClient, method: str, url: str, **kwargs: Any
) -> httpx.Response:
    """Generic retry wrapper: exponential backoff with jitter on 429/5xx/timeouts,
    max 5 attempts; fails fast (no retry) on other 4xx — a bad request never
    succeeds by retrying it.
    """

    @retry(
        retry=retry_if_exception(_is_retryable),
        wait=wait_random_exponential(multiplier=0.5, max=10),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    async def _do() -> httpx.Response:
        resp = await client.request(method, url, **kwargs)
        if resp.status_code >= 400:
            raise JevHTTPError(resp.status_code, resp.text)
        return resp

    return await _do()


class _LocalStubBackend:
    """Lazily loads the stand-in model once per process; runs the (sync,
    torch) forward pass off the event loop via asyncio.to_thread.
    """

    _model: Any = None
    _load_lock = asyncio.Lock()

    async def decide(self, state: str, questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if _LocalStubBackend._model is None:
            async with _LocalStubBackend._load_lock:
                if _LocalStubBackend._model is None:
                    from typed_decisions.open_jev import OpenJev

                    _LocalStubBackend._model = OpenJev.from_pretrained(
                        "com-kotobalabs/open-jev-deberta-v3-large"
                    )
        return await asyncio.to_thread(_LocalStubBackend._model.decide, state, questions)


_local_backend = _LocalStubBackend()


class JevClient:
    def __init__(self, cfg: JudgeCfg) -> None:
        # Fail at construction, not on the first call: eval/methods.py::build_judge turns a
        # NotImplementedError from make_judge into "method unavailable, skipped".
        if cfg.transport != "local_stub":
            raise NotImplementedError(
                f"judge.transport={cfg.transport!r} is not implemented yet: real TypeSafe "
                "access (waitlist / OpenRouter key) is still pending, and request field "
                "names are unverified (SPEC.md §13.5). See docs/verified_apis.md -> "
                "'Jev / TypeSafe'. For development only: --set judge.transport=local_stub "
                "(the open-jev stand-in, not TypeSafe's model)."
            )
        self.cfg = cfg
        self._semaphore = asyncio.Semaphore(cfg.max_concurrency)
        self.latencies_ms: list[float] = []

    async def ask(self, state: str, questions: dict[str, Any]) -> dict[str, Any]:
        async with self._semaphore:
            t0 = time.monotonic()
            try:
                return await self._ask_local_stub(state, questions)
            finally:
                self.latencies_ms.append((time.monotonic() - t0) * 1000)

    async def _ask_local_stub(self, state: str, questions: dict[str, Any]) -> dict[str, Any]:
        ids = list(questions.keys())
        question_list = [{"id": qid, **spec} for qid, spec in questions.items()]
        answers = await _local_backend.decide(state, question_list)
        return dict(zip(ids, answers, strict=True))
