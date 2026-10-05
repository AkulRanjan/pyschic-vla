# Verified API facts

Every **VERIFY** item from `SPEC.md` gets resolved here, with a link to the official source and the date it was checked. Code may depend only on what's written here or seen in a recorded fixture (SPEC.md §13.5).

Status values: ⬜ open · ✅ verified · ⚠️ verified, with a surprise (explain it).

---

## Jev / TypeSafe (owner: Person 2, Jagriti)

| # | Question | Answer | Source (link) | Checked | Status |
|---|---|---|---|---|---|
| J1 | Does `choice` return a probability for **every** option, or only the top pick? (**critical**) | | | | ⬜ |
| J2 | Exact request fields: `model`, `state`, `questions`, question `type`, `instructions`, options field name | | | | ⬜ |
| J3 | Exact response shape: where answers and probabilities live for `noul` and `choice` | | | | ⬜ |
| J4 | Is usage (input tokens) reported? | | | | ⬜ |
| J5 | Can `jev-1.13.0` be pinned on each route (direct, OpenRouter, Cloudflare, Vercel)? | | | | ⬜ |
| J6 | Does OpenRouter take the typed `questions` request, or a chat-style wrapper? | | | | ⬜ |
| J7 | Rate limits (RPM / TPM / concurrency) | | | | ⬜ |
| J8 | Max options per `choice` (spec: 255); context window (spec: 32K) | | | | ⬜ |
| J9 | `typesafe-sdk-python` API (sync and async); do we use it or raw HTTP? | | | | ⬜ |
| J10 | Measured latency p50 / p95 (~20 calls) | | | | ⬜ |
| J11 | Terms of service: is publishing benchmarks or stress-testing allowed? | | | | ⬜ |

Fixtures recorded in `tests/fixtures/jev/`: _(list them here)_

---

## Gemma 4 and serving (owner: Person 3, Akul)

| # | Question | Answer | Source (link) | Checked | Status |
|---|---|---|---|---|---|
| G1 | HF model IDs (E4B-it, 31B-it) and licence-acceptance steps | | | | ⬜ |
| G2 | Chat template special tokens; is a **system** role supported? | | | | ⬜ |
| G3 | How to **disable thinking mode** | | | | ⬜ |
| G4 | E4B memory at fp16 / 8-bit / 4-bit; does fp16 work on a T4 (no bf16)? | | | | ⬜ |
| G5 | vLLM / SGLang versions supporting Gemma 4; prefix-caching flags | | | | ⬜ |
| G6 | Completions endpoint: `n`, `stop`, `seed`, `logprobs`, `top_logprobs` (and its max)? | | | | ⬜ |
| G7 | Ollama tags and raw-completion support with `n` | | | | ⬜ |
| G8 | Where the 31B ceiling (baseline C) can run, and at what price | | | | ⬜ |
| G9 | Gemma 4 training-data cutoff (for the AIME contamination split) | | | | ⬜ |
| G10 | Measured throughput (tokens/s), `n=4` short steps with a shared prefix, on our GPU | | | | ⬜ |

---

## Datasets and other models (owners noted)

| # | Question | Answer | Source (link) | Checked | Status |
|---|---|---|---|---|---|
| D1 | `HuggingFaceH4/MATH-500` ID and field names (Akul) | | | | ⬜ |
| D2 | MATH train-split dataset ID (Akul) | | | | ⬜ |
| D3 | AIME dataset IDs for recent years (Akul) | | | | ⬜ |
| D4 | `Qwen/Qwen2.5-Math-PRM-7B` ID, step-separator format, how to read scores (Jagriti) | | | | ⬜ |
| D5 | Kaggle GPU quota and session limits (Akul) | | | | ⬜ |
