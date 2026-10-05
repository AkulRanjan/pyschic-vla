# Verified APIs

Each item records what was checked, how, and when. **Status:** ✅ verified · ⚠️ partly verified / needs a GPU check · ❌ open.

## Jev (TypeSafe AI)

_Owner: Person 2. To be filled in._

## Gemma / serving

_Owner: Person 3 (Akul). Last checked 2026-10-05._

| # | Question | Answer | Status | Source |
|---|---|---|---|---|
| 1 | HF model IDs; licence acceptance | `google/gemma-4-E2B-it`, `google/gemma-4-E4B-it`, `google/gemma-4-26B-A4B-it`, `google/gemma-4-31B-it` all exist. **Not gated** (`"gated": false`), so no licence click-through or HF token is needed to download. Licence: Apache 2.0. | ✅ | HF API `huggingface.co/api/models/<id>`; [model card](https://ai.google.dev/gemma/docs/core/model_card_4) |
| 2 | Chat template; system role | Turns are `<\|turn>{role}\n…<turn\|>\n`, and the generation prompt is `<\|turn>model\n`. The template starts with `<bos>`. The **system role is supported natively** (`<\|turn>system\n…`), so we use it (`use_system_role=True`). | ✅ | `chat_template.jinja` in the E4B repo; [model card](https://ai.google.dev/gemma/docs/core/model_card_4): "native support for the `system` role" |
| 3 | Disabling thinking | Thinking is **off by default**. It's on only if `<\|think\|>` appears at the start of the system turn. The template inserts it when `enable_thinking=True`, which defaults to false. Thoughts appear as `<\|channel>thought\n…<channel\|>`. We pass `enable_thinking=False` explicitly, and `gemma.py` warns if `<\|channel>` / `<channel\|>` / `<\|think\|>` show up in outputs. | ✅ | `chat_template.jinja` (lines ~186–195); model card; `tests/test_tokenize_network.py` |
| 4 | **BOS handling** (not in the original list; important) | The rendered template already contains `<bos>`, and the Gemma 4 HF tokenizer does **not** add BOS on `encode`. So the BOS must stay in the prompt text (`HFChatTokenizer(strip_bos=False)`, the default). If a server adds its own BOS (possibly Ollama/llama.cpp), set `strip_bos=True` to avoid a double BOS. | ✅ (HF); ⚠️ per server | `tests/test_tokenize_network.py` |
| 5 | Memory; fp16 on T4 | E4B is 4.5B effective / **8B params with embeddings**. bf16 weights are **16.0 GB** (`model.safetensors`), which won't fit a 16 GB T4. Official QAT 4-bit checkpoint `google/gemma-4-E4B-it-qat-w4a16-ct`: 11.5 GB on disk (vLLM quotes ~9.8 GB in memory). Native dtype is bf16, which the T4 lacks. **Whether vLLM runs the W4A16 checkpoint correctly in fp16 on a T4 (sm_75) is untested** (possible fp16 overflow). vLLM's recipe lists 1× 24 GB+ GPU in bf16 for E4B. | ⚠️ | HF file sizes (`X-Linked-Size`); `config.json` (`dtype: bfloat16`); [vLLM Gemma 4 recipe](https://docs.vllm.ai/projects/recipes/en/latest/Google/Gemma4.html) |
| 6 | vLLM / SGLang version; prefix caching | vLLM: the official recipe recommends the latest (`uv pip install -U vllm --pre`), and model pages cite **≥ 0.19.1**. Prefix caching is on by default in vLLM V1; pass `--enable-prefix-caching` explicitly anyway. SGLang: not yet checked. | ⚠️ | [vLLM recipe](https://docs.vllm.ai/projects/recipes/en/latest/Google/Gemma4.html), [recipes.vllm.ai](https://recipes.vllm.ai/Google/gemma-4-E4B-it) |
| 7 | Completions endpoint: `n`, `stop`, `seed`, `logprobs`, `top_logprobs` | vLLM's `/v1/completions` accepts all of them. `logprobs=k` returns the top-k per token, and the server caps it with `--max-logprobs` (default 20). This must be **checked on the running server** with `scripts/check_server.py` (TODO). | ⚠️ | vLLM OpenAI-compatible server docs |
| 8 | Ollama | The tag `gemma4:e4b` exists (~9.6 GB, quantized). Ollama has `/v1/completions`, but by default it applies its **own** template, and `n>1` is not known to be supported. Use Ollama for smoke tests only, never for measurements. | ⚠️ | [Ollama guide](https://gemma4all.com/blog/run-gemma-4-with-ollama), [HF nothink GGUF](https://huggingface.co/tawatchai/gemma4-e4b-nothink-gguf) |
| 9 | Where the 31B (C) runs | bf16 needs ~62 GB. The QAT W4A16 checkpoint `google/gemma-4-31B-it-qat-w4a16-ct` (23.3 GB) fits a single 48 GB GPU, or maybe a 32 GB one. The hosted-provider choice and price are still **open**. | ❌ | HF file sizes |
| 10 | Datasets | **MATH-500**: `HuggingFaceH4/MATH-500`, split `test`, 500 rows. Fields: `problem, solution, answer, subject, level(int), unique_id` (e.g. `test/precalculus/807.json`). **MATH train**: `EleutherAI/hendrycks_math`, 7 configs (one per subject), split `train`, 7,500 rows. Fields: `problem, level("Level 5"), type, solution`. There is **no `answer` field**: it's extracted from `solution`'s last `\boxed{}`. **AIME**: `HuggingFaceH4/aime_2024` (`id, problem, solution, answer(str), url, year`); `MathArena/aime_2025` (`problem_idx, problem, answer(int), problem_type`); `MathArena/aime_2026` (`problem_idx, answer(int), problem`). All have 30 rows, split `train`. | ✅ | `datasets-server.huggingface.co/{splits,first-rows,size}` |
| 11 | Gemma 4 training cutoff | **January 2025.** So AIME 2025 (Feb 2025) and AIME 2026 are post-cutoff, and AIME 2024 is pre-cutoff. | ✅ | [model card](https://ai.google.dev/gemma/docs/core/model_card_4) |
| 12 | AIME exam dates | 2024: I = Jan 31, II = Feb 7. 2025: I = Feb 6, II = Feb 12. 2026: I = Feb 5, II = Feb 11. These only affect the pre/post-cutoff flag, which doesn't change for any year given a January 2025 cutoff. | ⚠️ (from memory; check the MAA calendar) | — |
| 13 | Generation defaults | `generation_config.json`: temperature 1.0, top_k 64, top_p 0.95, eos ids `[1, 106, 50]` (`<eos>`, `<turn\|>`, …). We override with spec §5.2 values (0.9 / 0.95). The server stops at `<turn\|>`, so full solutions end at EOS without a stop string. | ✅ | `generation_config.json` |

### Grader validation
- `scripts/check_grader_math500.py`: `extract_answer(solution)` vs `answer` on all of MATH-500 → **500/500 agreement** (2026-10-05).
- A false-positive check (each answer graded against the *next* problem's gold) gave 3/500 matches, and all 3 are genuinely equal values (`5` vs `x=5`, `7` vs `7`, `3` vs `3`).
- `math-verify` 0.9.0 uses `signal.alarm` for timeouts, which only works on the main thread and not at all on Windows. We disable its timeouts and enforce our own 5 s limit with a daemon thread (`grading._math_verify_with_timeout`).
