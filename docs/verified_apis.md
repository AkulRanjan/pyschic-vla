# Verified API facts

Every **VERIFY** item from `SPEC.md` gets resolved here, with a link to the official source and the date it was checked. Code may depend only on what's written here or seen in a recorded fixture (SPEC.md §13.5).

Status values: ⬜ open · ✅ verified · ⚠️ verified, with a surprise (explain it).

---

## Jev / TypeSafe (owner: Person 2, Jagriti)

**Decision (2026-10-05):** the TypeSafe direct-API waitlist and OpenRouter
access are both still pending, so Week 1 (`JevJudge`, cache, budget) was
built and tested against an **open-source stand-in**,
`com-kotobalabs/open-jev-deberta-v3-large` (Apache-2.0, DeBERTa-v3-large,
434M params), self-hosted locally (`judge.transport = "local_stub"` — added
to `JudgeCfg`'s `transport` literal, flagged below since `config.py` is
shared and needs team sign-off). It is **not** TypeSafe's Jev; J1–J11 below
are resolved **for the stand-in only**, and must be re-verified once real
access lands. Don't assume they transfer.

HF-Space hosting was tried first (`hugging-apps/open-jev-deberta-v3-large-demo`,
duplicated from `com-kotobalabs/...`) and abandoned: duplicating it requires
an **HF PRO subscription** even on free `cpu-basic` hardware (`402 Payment
Required`) — a real money decision, not pursued without the team. Self-hosting
locally (CPU, no GPU needed) sidesteps that.

| # | Question | Answer | Source (link) | Checked | Status |
|---|---|---|---|---|---|
| J1 | Does `choice` return a probability for **every** option, or only the top pick? (**critical**) | **Yes, for the stand-in**: full `probabilities` dict over every option, plus the argmax `choice` and its `confidence`. Real TypeSafe Jev: still open. | `tests/fixtures/jev/choice_4_options.json` | 2026-10-05 | ⚠️ (stand-in only) |
| J2 | Exact request fields: `model`, `state`, `questions`, question `type`, `instructions`, options field name | For the stand-in (our own `JevClient.ask(state, questions: dict[id, {type, instructions, options?}])` contract, not TypeSafe's wire format): `type` ∈ `noul`\|`choice`\|`score`, `instructions` (str), `options` (dict, choice/score only). Real TypeSafe field names: still open — no guessing (spec §13.5). | `src/thoughtzero/judge/client.py`, `scripts/probe_open_jev.py` | 2026-10-05 | ⚠️ (stand-in only) |
| J3 | Exact response shape: where answers and probabilities live for `noul` and `choice` | Stand-in: `{question_id: {"noul": p}}` or `{question_id: {"choice": key, "probabilities": {...}, "confidence": float}}`. Real TypeSafe: still open. | `tests/fixtures/jev/*.json` | 2026-10-05 | ⚠️ (stand-in only) |
| J4 | Is usage (input tokens) reported? | Stand-in: no (it's a local forward pass). `JevJudge` always estimates `chars/4` itself rather than depending on a reported `usage` field, so this doesn't block anything either way. Real TypeSafe: still open. | — | 2026-10-05 | ⬜ |
| J5 | Can `jev-1.13.0` be pinned on each route (direct, OpenRouter, Cloudflare, Vercel)? | Not applicable to the stand-in (no versioned API). Real TypeSafe: still open. | — | — | ⬜ |
| J6 | Does OpenRouter take the typed `questions` request, or a chat-style wrapper? | Still open — access pending. | — | — | ⬜ |
| J7 | Rate limits (RPM / TPM / concurrency) | Not applicable to the stand-in (local, no rate limit; `JevClient`'s semaphore still caps `judge.max_concurrency` to protect CPU, tested in `test_jev_client.py`). Real TypeSafe: still open. | — | — | ⬜ |
| J8 | Max options per `choice` (spec: 255); context window (spec: 32K) | Stand-in enforces **2..255** options (`ValueError` below 2, not silently `{option: 1.0}`), 512-token context but **only the first 256 tokens of the state are read** (truncates from the end, not matching our own middle-truncation strategy — `JevJudge._truncate_steps` still runs first regardless). Real TypeSafe: still open. | `tests/fixtures/jev/choice_1_option_rejected.json`; model card | 2026-10-05 | ⚠️ (stand-in only, and the stand-in's own truncation differs from spec §5.3) |
| J9 | `typesafe-sdk-python` API (sync and async); do we use it or raw HTTP? | Still open — no key yet to try it. | — | — | ⬜ |
| J10 | Measured latency p50 / p95 (~20 calls) | Stand-in: sub-second per call on CPU (informal only; not yet measured p50/p95 over 20 calls). Real TypeSafe: still open. | — | — | ⬜ |
| J11 | Terms of service: is publishing benchmarks or stress-testing allowed? | Not applicable to the stand-in (Apache-2.0, self-hosted). Real TypeSafe ToS: still open — not yet read, since we don't have an account. | — | — | ⬜ |

**Fixtures recorded in `tests/fixtures/jev/`** (all from the stand-in, via `scripts/probe_open_jev.py`):
`noul_basic.json`, `choice_4_options.json`, `choice_and_noul_together.json`, `choice_1_option_rejected.json` (model rejects a single-option `choice`), `error_bad_question_type.json` (malformed question → `KeyError`, not a typed validation error).

---

## Gemma 4 and serving (owner: Person 3, Akul)

| # | Question | Answer | Source (link) | Checked | Status |
|---|---|---|---|---|---|
| G1 | HF model IDs (E4B-it, 31B-it) and licence-acceptance steps | `google/gemma-4-E2B-it`, `google/gemma-4-E4B-it`, `google/gemma-4-26B-A4B-it` and `google/gemma-4-31B-it` all exist and are **not gated** (`"gated": false`), so there's no licence click-through and no HF token is needed. Licence: Apache 2.0. | HF API `huggingface.co/api/models/<id>`; [model card](https://ai.google.dev/gemma/docs/core/model_card_4) | 2026-10-05 | ✅ |
| G2 | Chat template special tokens; is a **system** role supported? | Turns are `<\|turn>{role}\n…<turn\|>\n`, and the generation prompt is `<\|turn>model\n`. The template starts with `<bos>`. **The system role is native** (`<\|turn>system\n…`), so we use it (`use_system_role=True`). **Surprise:** the HF tokenizer does **not** add BOS on `encode`, so the template's `<bos>` must stay in the prompt text for vLLM/SGLang (`HFChatTokenizer(strip_bos=False)`, the default). For a server that adds its own BOS, set `strip_bos=True`. | `chat_template.jinja` and `tokenizer_config.json` in the E4B repo; [model card](https://ai.google.dev/gemma/docs/core/model_card_4); `tests/test_tokenize_network.py` | 2026-10-05 | ⚠️ |
| G3 | How to **disable thinking mode** | It's **off by default**. Thinking is on only if `<\|think\|>` starts the system turn, which the template inserts when `enable_thinking=True` (default false). Thoughts appear as `<\|channel>thought\n…<channel\|>`. We pass `enable_thinking=False` explicitly, and `llm/gemma.py` warns if `<\|channel>`, `<channel\|>` or `<\|think\|>` appear in outputs. | `chat_template.jinja` (~lines 186–195); [model card](https://ai.google.dev/gemma/docs/core/model_card_4) | 2026-10-05 | ✅ |
| G4 | E4B memory at fp16 / 8-bit / 4-bit; does fp16 work on a T4 (no bf16)? | 4.5B effective / **8B params with embeddings**. The bf16 weights are **16.0 GB**, which doesn't fit a 16 GB T4. The official 4-bit QAT checkpoint `google/gemma-4-E4B-it-qat-w4a16-ct` is 11.5 GB on disk (vLLM quotes ~9.8 GB in memory). The native dtype is bf16. **fp16 correctness on a T4 (sm_75) is untested**; vLLM's recipe lists 1× 24 GB+ in bf16. | HF file sizes (`X-Linked-Size`); `config.json`; [vLLM Gemma 4 recipe](https://docs.vllm.ai/projects/recipes/en/latest/Google/Gemma4.html) | 2026-10-05 | ⚠️ |
| G5 | vLLM / SGLang versions supporting Gemma 4; prefix-caching flags | vLLM: the recipe says install the latest (`uv pip install -U vllm --pre`); model pages cite **≥ 0.19.1**. Prefix caching is on by default in vLLM V1; we also pass `--enable-prefix-caching`. SGLang: **not checked yet**. | [vLLM recipe](https://docs.vllm.ai/projects/recipes/en/latest/Google/Gemma4.html), [recipes.vllm.ai](https://recipes.vllm.ai/Google/gemma-4-E4B-it) | 2026-10-05 | ⚠️ |
| G6 | Completions endpoint: `n`, `stop`, `seed`, `logprobs`, `top_logprobs` (and its max)? | vLLM's `/v1/completions` accepts all of them. `logprobs=k` returns the top-k per token, capped by `--max-logprobs` (default 20). **Still has to be confirmed on our running server.** | vLLM OpenAI-compatible server docs | — | ⬜ |
| G7 | Ollama tags and raw-completion support with `n` | The tag `gemma4:e4b` exists (~9.6 GB, quantized). `/v1/completions` exists, but Ollama applies its **own** template by default, and `n>1` isn't confirmed. Use it for smoke tests only, never for measurements. | [Ollama guide](https://gemma4all.com/blog/run-gemma-4-with-ollama), [nothink GGUF](https://huggingface.co/tawatchai/gemma4-e4b-nothink-gguf) | 2026-10-05 | ⚠️ |
| G8 | Where the 31B ceiling (baseline C) can run, and at what price | bf16 needs ~62 GB. The QAT `google/gemma-4-31B-it-qat-w4a16-ct` checkpoint (23.3 GB) fits one 48 GB GPU, or possibly 32 GB. A hosted provider and its price are **still open**. | HF file sizes | 2026-10-05 | ⬜ |
| G9 | Gemma 4 training-data cutoff (for the AIME contamination split) | **January 2025** (we use 2025-01-31 in `datasets.GEMMA4_TRAINING_CUTOFF`). So AIME 2025 and AIME 2026 are post-cutoff, and AIME 2024 is pre-cutoff. | [model card](https://ai.google.dev/gemma/docs/core/model_card_4) | 2026-10-05 | ✅ |
| G10 | Measured throughput (tokens/s), `n=4` short steps with a shared prefix, on our GPU | Not measured yet (no GPU run). | — | — | ⬜ |

Generation defaults (`generation_config.json`): temperature 1.0, top_k 64, top_p 0.95, eos ids `[1, 106, 50]` (includes `<turn|>`). We override with SPEC.md §5.2 values. Full solutions end at EOS without a stop string.

---

## Datasets and other models (owners noted)

| # | Question | Answer | Source (link) | Checked | Status |
|---|---|---|---|---|---|
| D1 | `HuggingFaceH4/MATH-500` ID and field names (Akul) | Split `test`, 500 rows. Fields: `problem, solution, answer, subject, level (int), unique_id` (e.g. `test/precalculus/807.json`, which gives our id `math500/test/precalculus/807`). | `datasets-server.huggingface.co/first-rows` | 2026-10-05 | ✅ |
| D2 | MATH train-split dataset ID (Akul) | `EleutherAI/hendrycks_math`: 7 configs (one per subject), split `train`, 7,500 rows. Fields: `problem, level ("Level 5"), type, solution`. **Surprise:** there's no `answer` field; we extract it from the solution's last `\boxed{}` and skip rows without one. It's disjoint from MATH-500, which is drawn from the MATH test split. | `datasets-server.huggingface.co/{splits,first-rows,size}` | 2026-10-05 | ⚠️ |
| D3 | AIME dataset IDs for recent years (Akul) | 2024: `HuggingFaceH4/aime_2024` (`id, problem, solution, answer (str), url, year`). 2025: `MathArena/aime_2025` (`problem_idx, problem, answer (int), problem_type`). 2026: `MathArena/aime_2026` (`problem_idx, answer (int), problem`). Each has 30 rows, split `train`. MathArena's `problem_idx` 1–15 = AIME I, 16–30 = AIME II. The exam dates in `datasets.AIME_SOURCES` are from memory; check them against the MAA calendar (they don't change the pre/post-cutoff split). | `datasets-server.huggingface.co/first-rows` | 2026-10-05 | ⚠️ |
| D4 | `Qwen/Qwen2.5-Math-PRM-7B` ID, step-separator format, how to read scores (Jagriti) | | | | ⬜ |
| D5 | Kaggle GPU quota and session limits (Akul) | Not checked yet. | | | ⬜ |

### Grader validation (Akul)
- `scripts/check_grader_math500.py`: `extract_answer(solution)` vs `answer` on all of MATH-500 → **500/500 agreement** (2026-10-05).
- False-positive check (each answer graded against the *next* problem's gold): 3/500 matched, and all 3 are genuinely equal (`5` vs `x=5`, `7` vs `7`, `3` vs `3`).
- `math-verify` 0.9 times out via `signal.alarm`, which only works on the main thread and not at all on Windows. We disable its timeouts and enforce our own 5 s limit with a daemon thread (`grading._math_verify_with_timeout`).
