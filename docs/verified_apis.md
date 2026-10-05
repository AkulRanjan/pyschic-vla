# Verified APIs

Every entry here must cite a link to official docs, or a real fixture file in
`tests/fixtures/jev/` that proves the behavior. No guessed field names (spec §13.5).

## Jev (TypeSafe AI) — direct/OpenRouter access

**Access status**
- [ ] TypeSafe direct-API waitlist joined (`https://api.typesafe.ai/v1/systemone`) — date requested: _____
- [ ] OpenRouter key obtained, `typesafe/jev-1.13` availability confirmed
- [ ] Cloudflare Workers AI backup route (`typesafe/jev`) checked
- [ ] Vercel AI Gateway backup route (`typesafe-ai/jev`) checked
- [ ] ToS read (TypeSafe + gateway) — benchmarking/publishing allowed? note below

Decision (2026-10-05): deprioritized while the waitlist is pending. Developing
against the open-source stand-in below instead so Week 1 work isn't blocked.
Revisit this section once TypeSafe/OpenRouter access actually comes through —
**do not assume the stand-in's answers below transfer to real Jev** without
re-verifying against a real response.

**ToS notes**
> (summarize what's allowed re: publishing benchmark results / stress-testing, with link)

**VERIFY checklist (real TypeSafe Jev — still open)**

| Question | Answer | Source / link |
|---|---|---|
| Does `choice` return a probability for every option, or only the top pick? | | |
| Exact request field names (`state`, `questions`, question `type`, `instructions`, options field name, `model`) | | |
| Exact response shape (answer + probabilities location, `noul` yes/no representation) | | |
| Does the response report `usage` (input tokens)? | | |
| Is the model pinnable to `jev-1.13.0` on each route? | | |
| Does OpenRouter use the same typed `questions` request, or a chat-style wrapper? | | |
| Rate limits (RPM / TPM), concurrency limits | | |
| Max options in a `choice` (spec says 255) and max context (32K) | | |
| `typesafe-sdk-python` API quality (sync/async) vs raw HTTP | | |
| Latency p50 / p95 over ~20 calls | | |

## open-jev (self-hosted stand-in, used for Week 1 development)

**What it is:** `com-kotobalabs/open-jev-deberta-v3-large` — an independent,
Apache-2.0, open-weights reimplementation of Jev's `noul`/`choice`/`score`
decision interface, built on a DeBERTa-v3-large (434M param) encoder. Not
TypeSafe's model; used only as a free, local, same-shaped substitute while
real Jev access is pending. [Model card](https://huggingface.co/com-kotobalabs/open-jev-deberta-v3-large).

**Setup actually used (2026-10-05):**
- Tried duplicating the HF Space `hugging-apps/open-jev-deberta-v3-large-demo`
  (itself duplicated from `com-kotobalabs/...`) to run it hosted — blocked:
  HF now requires a **PRO subscription** to duplicate even onto free
  `cpu-basic` hardware (`402 Payment Required`). Not pursued since that's a
  real money decision outside this budget.
- Instead, self-hosted locally: `pip install git+https://github.com/kotoba-lang/typed-decisions`
  plus `torch transformers safetensors huggingface_hub sentencepiece protobuf`,
  then `OpenJev.from_pretrained("com-kotobalabs/open-jev-deberta-v3-large")`.
  Runs on CPU on a Mac, no GPU needed. ~1.7GB of weights, downloaded once to
  the HF cache.

**Findings, verified against real local responses (fixtures in `tests/fixtures/jev/`):**

| Question | Answer | Fixture |
|---|---|---|
| Does `choice` return a probability for every option? | **Yes** — full `probabilities` dict over every option key, plus `choice` (argmax) and `confidence` (= the winning probability). | `choice_4_options.json`, `choice_and_noul_together.json` |
| Request shape (this library, not raw HTTP) | `model.decide(state: str, questions: list[dict])`. Each question dict: `id`, `type` (`"noul"` \| `"choice"` \| `"score"`), `instructions`, and for `choice`/`score` an `options` dict mapping option-key → option text. | `scripts/probe_open_jev.py` |
| Response shape | A **list**, one entry per question, in the same order as the request (not keyed by question id). `noul` → `{"noul": p_yes}`. `choice` → `{"choice": <winning key>, "probabilities": {...}, "confidence": float}`. | `noul_basic.json`, `choice_4_options.json` |
| `choice` option count bounds | Enforced **2..255** options — a single-option `choice` raises `ValueError("choice takes 2..255 options")`, it does not silently return `{that option: 1.0}`. | `choice_1_option_rejected.json` |
| Malformed question (bad `type`, missing `options`) | Raises `KeyError('options')` rather than a typed validation error — the library doesn't validate `type` up front, it just tries to read the fields the branch needs. | `error_bad_question_type.json` |
| Context window | 512 tokens, but **only the first 256 tokens of the state are read** per the model card — longer states are silently truncated from the end, not the middle. **This differs from the spec's truncation strategy (§5.3, keep-problem-and-last-N-steps)** — `JevJudge`'s own truncation logic still has to run before calling this, we can't rely on the model to do it sensibly. | model card |
| Usage/token accounting | Not reported by this library — it's a local forward pass, no `usage` field. Real Jev's `usage` field is still an open VERIFY item above. | — |
| Latency | Single CPU forward pass, sub-second per call observed informally; not yet measured p50/p95 over 20 calls. | — |
| License | Apache-2.0 (package + weights). | model card |

**Caveat:** this is a *stand-in*, not TypeSafe's Jev. Field names, response
shape, truncation behavior, and calibration are the open model's own design
and may not match real Jev at all. Keep `judge.transport` configurable
(`direct` \| `openrouter` \| `local_stub`) so swapping in real Jev later is a
config change, not a rewrite. The CRITICAL VERIFY item (full probability
distribution from `choice`) is answered **for this stand-in only** — still
needs re-confirming against real Jev before CP1.

## Gemma (owned by Person 3 — reference only)

(Person 3 fills this in; do not edit.)
