"""JevClient: one Jev "System One" request per call, over HTTP or the local stand-in.

Wire protocol (official API reference, https://docs.typesafe.ai/api.md; also what the
jev-chat/jev-chat-jarvis client sends; see docs/verified_apis.md)::

    POST <route URL>            Authorization: Bearer <key>     Content-Type: application/json
    {"model": "...", "state": "<text>", "questions": {"<id>": {"type": "noul" | "choice",
        "instructions": "...", "criteria": {...}}}}
    -> {"model": "jev-1.13.0", "answers": {"<id>": {...}}, "usage": {"input_tokens": n, ...}}

Choice options go in ``criteria`` (a map option -> description, 2..255 options); the
answer carries ``probabilities`` for every option. A noul answer is ``{"noul": p}``.

Routes (``judge.transport``) differ only in URL, key and model name. The key is read from
the environment (``.env``), never from config, and never logged.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from typing import Any

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_random_exponential

from thoughtzero.config import JudgeCfg

RETRYABLE_STATUS = {429, 500, 502, 503, 504, 529}  # 529 = TypeSafe "overloaded"
REQUEST_TIMEOUT_S = 60.0


@dataclass(frozen=True)
class Provider:
    url: str
    key_env: str | None  # None: no API key (a local server)
    model: str
    headers: tuple[tuple[str, str], ...] = ()
    is_typesafe_model: bool = True  # False: a different model behind the same protocol
    free: bool = False  # True: no per-token price (local), so cost and budget are 0
    max_state_tokens: int | None = (
        None  # the route's context limit, if below judge.max_state_tokens
    )


PROVIDERS: dict[str, Provider] = {
    "direct": Provider("https://api.typesafe.ai/v1/systemone", "TYPESAFE_API_KEY", "jev-1.13.0"),
    "openrouter": Provider(
        "https://openrouter.ai/api/alpha/decisions",
        "OPENROUTER_API_KEY",
        "typesafe/jev-1.13",
        headers=(("X-Title", "ThoughtZero"),),
    ),
    "zen": Provider("https://opencode.ai/zen/v1/systemone", "OPENCODE_ZEN_API_KEY", "jev-1.13"),
    "vercel": Provider(
        "https://ai-gateway.vercel.sh/typesafe/v1/systemone",
        "AI_GATEWAY_API_KEY",
        "typesafe-ai/jev",
    ),
    "bocha": Provider(
        "https://jev.bocha.cn/v1/systemone",
        "BOCHA_API_KEY",
        "bocha-jev-v1",
        is_typesafe_model=False,
    ),
    # jevos (github.com/feder-cr/jev, MIT): an open-source model that speaks Jev's wire format,
    # served locally on the CPU by `jev serve`. Not TypeSafe's model (results/DECISIONS.md).
    # 8,192-token context: keep the state well under it.
    "jevos": Provider(
        "http://127.0.0.1:8017/v1/systemone",
        None,
        "jevos-v4",
        is_typesafe_model=False,
        free=True,
        max_state_tokens=6_000,
    ),
}
LOCAL_STUB_MODEL = "open-jev-deberta-v3-large"
FREE_ROUTES = {"local_stub"} | {name for name, p in PROVIDERS.items() if p.free}


def judge_usd_per_mtok(cfg: JudgeCfg) -> float:
    """The judge's price per million input tokens: 0 on a free (local) route."""
    return 0.0 if cfg.transport in FREE_ROUTES else cfg.usd_per_mtok


def judge_max_state_tokens(cfg: JudgeCfg) -> int:
    """``judge.max_state_tokens``, lowered to the route's context limit if it has one."""
    provider = PROVIDERS.get(cfg.transport)
    if provider is not None and provider.max_state_tokens is not None:
        return min(cfg.max_state_tokens, provider.max_state_tokens)
    return cfg.max_state_tokens


class JevConfigError(ValueError):
    """The route can't be used as configured (e.g. its API key is not set)."""


class JevHTTPError(Exception):
    def __init__(self, status_code: int, body: str) -> None:
        super().__init__(f"Jev request failed with {status_code}: {body[:500]}")
        self.status_code = status_code
        self.body = body


@dataclass
class JevReply:
    answers: dict[str, Any]
    input_tokens: int | None  # as reported by the API; None if the route doesn't report it
    model: str | None = None  # the versioned model that answered, if reported


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.TransportError):  # timeouts, connection resets
        return True
    if isinstance(exc, JevHTTPError):
        return exc.status_code in RETRYABLE_STATUS
    return False


async def request_with_retry(
    client: httpx.AsyncClient, method: str, url: str, **kwargs: Any
) -> httpx.Response:
    """Exponential backoff with jitter on 429/529/5xx and transport errors, max 5 attempts;
    fails fast on other 4xx (401 bad key, 422 malformed request): retrying can't fix them.
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


def resolve_route(cfg: JudgeCfg) -> tuple[str, str]:
    """(POST URL, model name) for ``cfg``; config overrides win over the route's defaults."""
    if cfg.transport == "local_stub":
        return "local", cfg.jev_model or LOCAL_STUB_MODEL
    provider = PROVIDERS[cfg.transport]
    return cfg.jev_url or provider.url, cfg.jev_model or provider.model


class _LocalStubBackend:
    """Lazily loads the open-jev stand-in once per process; runs the (sync, torch) forward
    pass off the event loop via asyncio.to_thread.
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
        self.cfg = cfg
        self.url, self.model = resolve_route(cfg)
        self._key = ""
        self._headers: dict[str, str] = {}
        if cfg.transport != "local_stub":
            provider = PROVIDERS[cfg.transport]
            self._key = (os.environ.get(provider.key_env) or "").strip() if provider.key_env else ""
            if provider.key_env and not self._key:
                # build_judge turns this into "method unavailable", with this message
                raise JevConfigError(
                    f"judge.transport={cfg.transport!r} needs {provider.key_env} in .env "
                    "(see .env.example and docs/verified_apis.md)"
                )
            self._headers = dict(provider.headers)
        self._semaphore = asyncio.Semaphore(cfg.max_concurrency)
        self._http: httpx.AsyncClient | None = None
        self.latencies_ms: list[float] = []

    async def ask(self, state: str, questions: dict[str, Any]) -> JevReply:
        async with self._semaphore:
            t0 = time.monotonic()
            try:
                if self.cfg.transport == "local_stub":
                    return await self._ask_local_stub(state, questions)
                return await self._ask_http(state, questions)
            finally:
                self.latencies_ms.append((time.monotonic() - t0) * 1000)

    async def _ask_http(self, state: str, questions: dict[str, Any]) -> JevReply:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=REQUEST_TIMEOUT_S)
        resp = await request_with_retry(
            self._http,
            "POST",
            self.url,
            json={"model": self.model, "state": state, "questions": questions},
            headers={
                **({"Authorization": f"Bearer {self._key}"} if self._key else {}),
                **self._headers,
            },
        )
        body = resp.json()
        answers = body.get("answers")
        if not isinstance(answers, dict):
            raise JevHTTPError(resp.status_code, f"response has no 'answers' map: {resp.text}")
        usage = body.get("usage") or {}
        tokens = usage.get("input_tokens")
        return JevReply(answers, int(tokens) if tokens is not None else None, body.get("model"))

    async def _ask_local_stub(self, state: str, questions: dict[str, Any]) -> JevReply:
        # the stand-in takes a list of questions, with choice options under "options"
        question_list = []
        for qid, spec in questions.items():
            q = {"id": qid, **spec}
            if spec.get("type") == "choice":
                q["options"] = q.pop("criteria")
            question_list.append(q)
        answers = await _local_backend.decide(state, question_list)
        return JevReply(dict(zip(questions, answers, strict=True)), None, self.model)

    async def aclose(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None
