# Person 3 (Akul): Generator, data, baselines and GPU

> **Read this whole file, then `SPEC.md` §1–§5 (especially §3 Gemma, §5.2), §8 and §11–§13.** Part A is your personal work plan. Part B (at the bottom) is shared team information, identical in everyone's file.

---

# PART A — Your work

## A1. Your mission

You own **everything that produces text and everything that decides whether an answer is right**:
- the Gemma generator client and the prompt templates;
- dataset loading;
- answer grading;
- the non-search baselines (B1 chain-of-thought, B2 self-consistency, B3 best-of-N, C the 31B ceiling);
- the **GPU environment** (Kaggle notebooks running an SGLang/vLLM server).

Two of your pieces are used by *everyone*:
- **`grading.py`.** If grading is wrong, every accuracy number in the project is wrong.
- **The step format** (`format_steps` / `split_steps` plus the generator prompt). If steps are badly delimited, the search breaks.

Get those two right and well-tested first.

## A2. What you own

```
src/thoughtzero/llm/gemma.py          # OpenAICompatibleGenerator (Generator protocol)
src/thoughtzero/llm/prompts.py        # FILE OWNER: PROMPT_VERSION, generator prompts, format/split helpers
                                      #   (Person 2 edits only the JUDGE section)
src/thoughtzero/llm/tokenize.py       # HF tokenizer wrapper: chat template, token counting
src/thoughtzero/data/datasets.py      # MATH-500, MATH train, AIME loaders; subsets
src/thoughtzero/data/grading.py       # extract_answer, normalize_answer, is_equivalent
src/thoughtzero/baselines/cot.py              # B1 (and C with a different endpoint)
src/thoughtzero/baselines/self_consistency.py # B2
src/thoughtzero/baselines/best_of_n.py        # B3 (uses Person 2's judge.final_correct)
notebooks/kaggle_server.ipynb         # template: start server + run any script inside Kaggle
docs/verified_apis.md → "Gemma / serving" section
docs/gpu_setup.md                     # how to run on Kaggle / rented GPU, throughput numbers
tests/test_grading.py, test_prompts.py, test_gemma_client.py, test_datasets.py, test_baselines.py
```

**You do NOT own:** `search/` (1), `judge/` (2), `eval/` / `pilot/` (4).

## A3. Days 1–2: VERIFY the Gemma items and get a GPU server running

You don't need the scaffold for this. Start on day 1.

### A3.1 Resolve the Gemma VERIFY items
Write each answer, with links, in `docs/verified_apis.md` → **Gemma / serving**:

| Question | Why it matters |
|---|---|
| Exact HF model IDs (`google/gemma-4-E4B-it`, `google/gemma-4-31B-it`?) and licence acceptance steps | Loading |
| The **chat template**: exact special tokens, and **whether a system role is supported** (if not, fold the instructions into the user turn) | Prompt construction |
| **How to disable thinking mode** for Gemma 4 (a template flag, a special token, or a separate model?) | Spec §5.2: we use non-thinking mode |
| Memory needed for E4B at fp16 / 8-bit / 4-bit, and **does it run correctly in fp16 on a T4** (no bf16 on T4; some Gemma versions overflow in fp16) | Whether we can use Kaggle at all |
| Which **vLLM / SGLang version** supports Gemma 4; flags for prefix caching (`--enable-prefix-caching` in vLLM; RadixAttention is SGLang's default) | Throughput |
| Does the server's **completions** endpoint support `n`, `stop`, `seed`, `logprobs` and `top_logprobs` (and the max `top_logprobs`)? | Generator and Person 2's self-judge |
| Ollama tags `gemma4:e4b`, `gemma4:31b`, and whether Ollama's OpenAI-compatible API supports raw completions with `n` | Local dev |
| Where the **31B** can run for the ceiling (C): a hosted endpoint (which provider, price) or a rented GPU | Baseline C |
| HF dataset IDs and field names: `HuggingFaceH4/MATH-500`, the MATH train split, and recent AIME sets (2025/2026) | Data loading |

### A3.2 GPU setup (`notebooks/kaggle_server.ipynb`, `docs/gpu_setup.md`)
- Each teammate has their own Kaggle account (B8). Make **one notebook template** that anyone can copy:
  1. Install vLLM or SGLang (pin versions).
  2. Log in to HF with a Kaggle *secret* (never in plain text).
  3. Start the server in the background on `localhost:8000` with prefix caching, on GPU 0. Leave GPU 1 free for Person 2's PRM.
  4. Wait until it's healthy.
  5. `pip install git+https://github.com/<org>/thoughtzero@<branch>`, write a `.env` from Kaggle secrets, and run `python scripts/<script>.py ...`.
  6. Copy `results/` to the notebook output.
- Measure and record **throughput**: tokens/s for `n=4` short-step proposals with a shared prefix, and for full solutions. Everyone uses these numbers for planning.
- Local development: Ollama `gemma4:e4b` (or E2B if a laptop can't hold E4B) so others can test against a real model on their own machines.

## A4. Week 1 (days 3–7): prompts, grading, data and generator

### A4.1 `prompts.py` (deliver by day 4, because everyone imports it)
- `PROMPT_VERSION = "g1"`. Bump it on any prompt text change; Person 2 bumps it too when editing the judge section.
- `format_steps(steps) -> str` gives `"Step 1: ...\n\nStep 2: ...\n\n"`. `split_steps(text) -> list[str]` is the inverse, and it strips the `Step n:` prefixes.
- **Convention:** steps are stored **without** the prefix (B4). Unit-test that `split_steps(format_steps(x)) == x` for awkward inputs: steps containing newlines, LaTeX, the word "Step", empty steps.
- `generator_prompt(problem, steps, tokenizer) -> str` per spec §5.2:
  - apply the chat template to `[system?, user: problem + format instructions]` with `add_generation_prompt=True`;
  - append `format_steps(steps)`;
  - append `f"Step {len(steps)+1}:"`.
- Format instructions: write one numbered step at a time, keep each step short (one idea or calculation), and put the final answer in `\boxed{}` in the last step. Include a short worked example of the format in the instructions; it noticeably improves format adherence. **Tune the wording on the MATH *train* split only** (B11.1).
- Leave a clearly marked `# === JUDGE SECTION (owner: Person 2) ===` block for Person 2.

### A4.2 `grading.py` (deliver by day 4, the most important correctness code you write)
- `extract_answer(text) -> str | None`:
  - takes the **last** `\boxed{...}`, handling **nested braces** (`\boxed{\frac{1}{2}}`) by brace counting, not regex;
  - also handles `\boxed {}`, `\fbox{}`, and a fallback `Final answer: X`;
  - returns `None` if nothing is found.
- `normalize_answer(ans) -> str`, a canonical string used for grouping in `value_vote` and self-consistency:
  - strip `$`, `\left`/`\right`, `\!`, spaces;
  - `\dfrac`/`\tfrac` → `\frac`;
  - strip trailing `.` and text units (`\text{ cm}`), `^\circ`, a leading `x=`;
  - unify `0.5` / `.5`, and remove thousands commas (`1,000`).
- `is_equivalent(pred, gold) -> bool`:
  - `math-verify` first, with a **timeout** (it uses sympy, which can hang on pathological input; run it in a thread or subprocess with a ~5 s limit);
  - fall back to a normalized string comparison;
  - `None` is never equivalent.
- `test_grading.py`: **at least 30 hand-written cases** (spec §8.2), including:
  - fractions vs decimals; `\frac12`; `\sqrt{2}/2` vs `\frac{\sqrt2}{2}`;
  - π expressions; degrees; units; percentages;
  - intervals `[1,3)`; unordered sets `{1,2}` vs `{2,1}`; multiple answers;
  - text answers ("Monday"); negative numbers; scientific notation;
  - matrices; nested boxes; two boxes (take the last); no box.
- Also run the grader on **MATH-500's own reference solutions** vs their `answer` field. Agreement should be about 100%; investigate every mismatch. This is a cheap, strong check.

### A4.3 `datasets.py` (deliver by day 5)
- `load_math500() -> list[Problem]` (fields to VERIFY: `problem`, `solution`, `answer`, `subject`, `level`, `unique_id`), `load_math_train()`, and `load_aime(year)`.
- Stable `Problem.id` values that never change between runs.
- Deterministic subsets, so everyone uses the **same** problems:
  - `dev_subset(n=50, seed=0)` from MATH-500, for CP3;
  - `pilot_subset(n=200, seed=0)`, stratified by `level`. Draw it **from the MATH train split** if the team accepts B11.1, otherwise from MATH-500;
  - `ablation_subset(n=200, seed=1)` from MATH-500, for week 4.
- Cache the downloaded data locally (`datasets` handles this). Tests use a tiny bundled JSON sample, not the network.
- AIME: tag each problem with its release date. Mark problems released **after Gemma 4's training cutoff** (VERIFY the cutoff date) as `post_cutoff=True`, so they're reported separately (contamination check, spec §8.2).

### A4.4 `gemma.py`: OpenAICompatibleGenerator (deliver by day 6)
- Uses `openai.AsyncOpenAI(base_url=GEMMA_BASE_URL, api_key="EMPTY" or env)` and the **completions** endpoint (`client.completions.create`), **not chat**. We continue a partially written assistant turn (spec §5.2).
- `propose(problem, steps, k)`:
  - one request with `n=k`, `temperature=0.9`, `top_p=0.95`, `stop=cfg.stop` (`["\n\nStep", "\n\n\n"]`), `max_tokens=max_step_tokens`, and `seed` if supported;
  - strip each output and **drop empty ones**; if all are empty, retry once, then return what you have;
  - truncate any accidental `Step n+1:` that slipped through.
- **Token accounting:**
  - with `n=k`, the usage block is aggregated across choices, so count each choice's completion tokens with the HF tokenizer (or logprobs), and count the prompt tokens **once per request**;
  - call `accounting.record_gemma(prompt, completion)` for every request;
  - the completion-token count is the **primary compute axis** for the whole paper, so get it exactly right and test it.
- `complete(problem, steps, temperature=0.0) -> list[str]`: continue to the end in one request, without the step stop sequence but with a larger `max_tokens`, stopping after `\boxed{...}` or at max depth. Split the result into steps and return only the **new** steps.
- `sample_completions(problem, steps, n, temperature) -> list[list[str]]` (B4 addition): same as `complete`, but `n` samples in one request. The pilot uses it for Monte Carlo labels.
- `sample_solutions(problem, n, temperature) -> list[str]`: full solutions from an empty prefix, used by B2 and the pilot traces.
- Also expose `raw_completion(prompt, max_tokens=1, logprobs=k)` for Person 2's self-judge.
- Retry on connection errors and 5xx (tenacity), and use a semaphore for client-side concurrency (config `generator.max_concurrency`, default 32).
- `tests/test_gemma_client.py`: use `respx` to mock the server; check the request body (completions, `n`, `stop`, prompt ends with `Step 3:`), empty-output handling, and token accounting. No real server in tests.

### A4.5 CP1 support (day 7)
Have a server running and a URL or notebook ready for Person 1's real smoke test, plus a short "how to run against the GPU" note in `docs/gpu_setup.md`.

## A5. Days 8–14: pilot compute and baselines

### A5.1 Pilot compute (Person 4 owns the pilot; you provide the GPU)
- The pilot is mostly a **Gemma** workload:
  - 200 traces;
  - ~1,600 prefixes × 8 completions = **~12,800 completions**.
- Estimate GPU-hours from your throughput numbers, and plan which Kaggle account runs which shard. Split by problem ID; the runner resumes.
- Sanity-check the traces: the fraction with a `\boxed{}` answer, average steps per solution, and average step length. If steps are huge (whole solutions in one "step"), fix the prompt **before** labelling, because labels from bad step splits are worthless.

### A5.2 Baselines (spec §8.1), each implementing Person 4's `Method` protocol
- **B1 `cot.py`**: greedy chain-of-thought (`temperature=0`), one full solution, then `extract_answer`.
- **B2 `self_consistency.py`**: the **primary baseline**. N samples at `temperature=0.7`, then a majority vote over `normalize_answer`, ignoring `None`. Ties go to the answer whose first occurrence came earliest; document this.
  - Sweep N so that **mean completion tokens per problem match ThoughtZero's** at each `n_simulations` (Person 1 sends the numbers around day 16).
  - Implement it so N can be chosen post hoc: sample `N_max` once, store all samples, and compute the vote for every N ≤ N_max by **prefix subsampling**. One run then gives the whole curve. Store the per-sample token counts too.
- **B3 `best_of_n.py`**: the same N samples (reuse B2's stored samples to save GPU time) picked by `judge.final_correct` (Jev). It costs Jev calls, so estimate first.
- **C**: Gemma 4 31B greedy chain-of-thought (and self-consistency if the budget allows), the same code as B1 with a different `base_url`/`model`. **Record where it ran** (provider, quantization) in the results; spec §8.1 requires this.
- `tests/test_baselines.py`: mock generator; voting logic, tie-breaking, `None` handling, and prefix subsampling.

### A5.3 CP3 (day 14)
B1, B2 and B3 run on the dev-50 subset through Person 4's runner and write comparable JSONL.

## A6. Week 3 (days 15–21): main baseline runs

- Run B1, B2 (N sweep covering TZ's token range), B3 and C on **full MATH-500 and AIME**.
- Take on extra GPU shards for the others if your quota allows. Keep a shared GPU-hours log.
- Check that compute matching is honest: for each TZ point there's a B2 point with mean completion tokens within ±10%.

## A7. Week 4 (days 22–28): ablations and write-up

- Your ablations:
  - **step granularity**: default delimiter vs `max_step_tokens=96` with a "shorter steps" prompt variant (bump `PROMPT_VERSION`);
  - **AIME contamination**: pre-cutoff vs post-cutoff results reported separately.
- Write these report sections:
  - **Setup**: models, serving, quantization, hardware, sampling parameters, datasets, grading;
  - **Baselines**;
  - a **Reproduction** subsection on GPU setup (from `docs/gpu_setup.md`).

## A8. Pitfalls specific to you

| Pitfall | Avoid by |
|---|---|
| Using the chat endpoint (can't continue a partial assistant turn) | Completions endpoint plus a manually applied chat template |
| Thinking mode left on, so steps get mixed with hidden reasoning | VERIFY how to disable it; assert no thinking tokens in outputs |
| Wrong token counts with `n>1` | Count per choice with the tokenizer; test it |
| Regex `\boxed{` extraction breaking on nested braces | Brace counting |
| `math-verify` hanging | Timeout wrapper |
| Steps that are whole solutions | Check stats on 20 traces before the pilot; tune the prompt on the train split |
| Prompt tuning on test problems | Tune only on MATH train (B11.1) |
| fp16 overflow / NaNs on a T4 | VERIFY; try 8-bit/AWQ, or rent a GPU with bf16 (A10G/L4/3090/4090) |
| Kaggle session dies mid-run | Resumable runner, small shards, copy outputs often |
| HF token in a notebook | Kaggle secrets only; clear outputs before sharing |

## A9. Your checklist

- [ ] Day 1: VERIFY research started; HF licence accepted; Kaggle notebook boots vLLM/SGLang with E4B
- [ ] Day 2: Gemma section of `verified_apis.md` complete; throughput measured; scaffold PR reviewed
- [ ] Day 4: `prompts.py` (format/split/generator prompt) and `grading.py` with ≥ 30 tests merged
- [ ] Day 5: `datasets.py` with dev/pilot/ablation subsets merged; grader checked against MATH-500 references
- [ ] Day 6: `OpenAICompatibleGenerator` merged; `raw_completion` with logprobs delivered to Person 2
- [ ] Day 7: CP1, server and notebook ready for the smoke test
- [ ] Day 10: pilot GPU shards run; trace sanity stats shared; CP2
- [ ] Day 14: B1, B2, B3 and C working on dev-50; CP3
- [ ] Day 21: full baseline runs with compute matching checked; CP4
- [ ] Day 28: step-granularity and contamination ablations; Setup, Baselines and Reproduction sections; CP5


---

# PART B — Team-wide information

> This part is **identical in all four person files**. If you change it, change it in all four and tell everyone.

## B1. The project in 60 seconds

**ThoughtZero** runs AlphaZero-style tree search (MCTS with PUCT) over the *reasoning steps* of a math solution.

- **Generator:** Gemma 4 E4B, a small local model. At each tree node it proposes K candidate next steps.
- **Judge:** Jev, TypeSafe AI's calibrated decision API. One call per expansion returns:
  - a **prior** over the K candidates (a `choice` question), which replaces AlphaZero's policy network;
  - a **value** for the current state, "are all steps so far correct?" (a `noul` yes/no probability), which replaces AlphaZero's value network.
- **No training.** The claim is that an off-the-shelf calibrated judge can be both policy and value, so a ~4B model with search approaches a much larger model at matched compute.
- **Primary baseline:** self-consistency (majority vote over N samples) at an equal number of Gemma completion tokens.
- **Go/no-go gate:** a cheap pilot first checks whether Jev can actually tell good steps from bad (AUROC ≥ 0.75 → GO).

The full spec is `SPEC.md` at the repo root. Section numbers like "§6.2" refer to it. **Read §1–§6 before writing any code.**

## B2. Team and ownership map

| # | Role | Owns (edit freely) | Name |
|---|---|---|---|
| **1** | Search lead and integrator | `search/`, `config.py`, `types.py`, `mocks.py`, `scripts/smoke_test.py`, `scripts/view_tree.py`, CI, `CLAUDE.md`, merging to `main` | Prakhar |
| **2** | Judge | `judge/` (all files), `tests/fixtures/jev/`, the Jev section of `docs/verified_apis.md`, the **judge section** of `llm/prompts.py` | Jagriti |
| **3** | Generator, data and GPU | `llm/gemma.py`, `llm/prompts.py` (file owner), `data/`, `baselines/`, `notebooks/kaggle_*.ipynb`, the Gemma section of `docs/verified_apis.md` | Akul |
| **4** | Evaluation and pilot | `eval/`, `pilot/`, `accounting.py`, `scripts/run_pilot.py`, `scripts/run_experiment.py`, `scripts/make_plots.py`, `results/REPORT.md` assembly | Harjas |

**Rule:** you may *read* anything. To change a file you don't own, open a PR and tag the owner, and wait for their approval.

**Shared files that need a PR tagged to all four:** `src/thoughtzero/types.py`, `src/thoughtzero/config.py`, `src/thoughtzero/accounting.py` (the `Ledger` fields), `configs/default.yaml`, `pyproject.toml`, `SPEC.md`.

## B3. Timeline and checkpoints

Day numbers count from the day the repo is created.

| When | What | Gate / checkpoint |
|---|---|---|
| **Days 1–2** | Person 1 builds the scaffold and the interface contract (B4). Everyone else does their "Day 1–2" prep work (API access, VERIFY research, pure functions). | **CP0:** the scaffold PR is reviewed by all 4 and merged. Nobody merges feature code before CP0. |
| **Days 3–7** | Week 1: four parallel streams, each built against mocks. | **CP1 (day 7):** Person 1 runs `smoke_test.py` against real Gemma and Jev. ≥ 1 of 3 easy problems solved, ≤ 20 Jev calls, ≤ $0.01. `docs/verified_apis.md` complete. This is the spec's Phase 0 definition of done. |
| **Days 8–10** | The pilot runs (owner: Person 4, with help from 2 and 3). In parallel, Person 1 builds async MCTS on mocks. | **CP2 (day 10): GO / PARTIAL / NO-GO meeting.** All 4 attend. No real experiments start before this. |
| **Days 11–14** | Week 2: async MCTS, alternative judges, baselines, runner. | **CP3 (day 14):** a 50-problem MATH-500 run with every method, comparable JSONL output, and inspectable tree dumps. This is the spec's Phase 2 and Phase 3 definition of done. |
| **Days 15–21** | Week 3: main experiments, split by method. | **CP4 (day 21):** the accuracy-vs-compute plot and summary table exist, and Jev spend is under budget. |
| **Days 22–28** | Week 4: ablations, report, README. | **CP5 (day 28):** `results/REPORT.md` and the README are done; tag `v0.1.0`. |
| Later | Phase 6 (expert iteration), only if time allows. | — |

**Weekly sync:** every Friday, 30–45 min. Each person demos what merged, says what's blocked, and states their cost and GPU-hours spent. Person 1 writes `results/PHASE_<n>_NOTES.md` afterwards.

**Daily async stand-up** (one message in the group chat): *done yesterday / doing today / blocked on*.

## B4. The shared contract (interfaces)

Person 1 writes these on days 1–2 in `src/thoughtzero/types.py` (and `accounting.py`, co-written with Person 4). **Everyone codes against exactly these signatures.** Changing one needs a PR tagged to all four.

```python
# src/thoughtzero/types.py
from dataclasses import dataclass, field
from typing import Protocol

@dataclass(frozen=True)
class Problem:
    id: str                 # stable, e.g. "math500/test/precalculus/807"
    question: str
    answer: str             # ground truth; ONLY used for grading, never inside search
    source: str             # "math500" | "aime2025" | ...
    level: int | None = None    # MATH difficulty 1-5 (for stratification)
    subject: str | None = None

@dataclass
class GenOut:
    text: str               # step text, stripped, WITHOUT the "Step n:" prefix
    prompt_tokens: int
    completion_tokens: int

class Generator(Protocol):
    async def propose(self, problem: str, steps: list[str], k: int) -> list[GenOut]: ...
    async def complete(self, problem: str, steps: list[str], temperature: float = 0.0) -> list[str]: ...
    async def sample_solutions(self, problem: str, n: int, temperature: float) -> list[str]: ...
    # ADDITION to the spec (needed by the pilot's Monte Carlo labels, one batched n=m request):
    async def sample_completions(self, problem: str, steps: list[str], n: int,
                                 temperature: float) -> list[list[str]]: ...

class Judge(Protocol):
    async def prior_and_value(self, problem: str, steps: list[str],
                              candidates: list[str]) -> tuple[list[float], float]: ...
    async def final_correct(self, problem: str, steps: list[str]) -> float: ...
    async def step_sound(self, problem: str, steps: list[str]) -> float: ...
```

**Conventions everyone must follow**

1. **A step is stored as plain text without the `Step n:` prefix**, stripped of leading and trailing whitespace. Numbering is added only when formatting, by `prompts.format_steps(steps)`. Person 1's dedupe, Person 2's judge state and Person 3's generator prompt all depend on this.
2. **`complete` and `sample_completions` return only the *new* steps**, never the given prefix.
3. **`sample_solutions` returns full raw solution texts.** Use `prompts.split_steps(text)` to turn them into steps.
4. **Probabilities** are Python floats in [0, 1]. Priors returned by a judge have `len == len(candidates)` and sum to 1 (±1e-6).
5. **Everything that does I/O is `async`.** Pure logic (PUCT, dedupe, metrics, grading) is sync and side-effect free.
6. **Answers:** `grading.extract_answer(text) -> str | None` and `grading.is_equivalent(pred, gold) -> bool` (Person 3). Nobody writes their own answer parsing.
7. **Accounting (ADDITION to the spec):** `complete` and `sample_solutions` return strings, so token counts can't travel in return values. Instead, `accounting.py` exposes a `ContextVar`:
   ```python
   # src/thoughtzero/accounting.py
   @dataclass
   class Ledger:
       gemma_prompt_tokens: int = 0
       gemma_completion_tokens: int = 0     # PRIMARY compute axis
       gemma_calls: int = 0
       jev_calls: int = 0
       jev_cache_hits: int = 0
       jev_input_tokens: int = 0
       jev_usd: float = 0.0
       wall_time_s: float = 0.0
       expansions: int = 0
       max_depth: int = 0
       terminal_leaves: int = 0

   current_ledger: ContextVar[Ledger]           # set by eval/runner.py per problem
   def record_gemma(prompt_tokens: int, completion_tokens: int) -> None: ...
   def record_jev(input_tokens: int, usd: float, cache_hit: bool) -> None: ...
   ```
   The generator calls `record_gemma` and the judge calls `record_jev` on every request. The runner sets a fresh `Ledger` per problem, and asyncio tasks inherit it automatically. **Cache hits record `usd=0`.**
8. **Config:** one pydantic tree (`config.py`) with sections `generator`, `judge`, `search`, `budget`, `eval`, `pilot`, plus `seed`. Add fields to your own section by PR. Configs load from YAML, `${ENV_VAR}` placeholders are interpolated, and CLI overrides look like `--set search.k=6`. `config_hash()` is the sha256 of the resolved config **with secrets removed**.
9. **Prompts** live only in `llm/prompts.py` (file owner: Person 3; the judge section is edited by Person 2). Every prompt change bumps `PROMPT_VERSION`, which goes into cache keys and logs.

## B5. Dependency map: who needs what from whom

| Needed by | What | From | By |
|---|---|---|---|
| Everyone | `types.py`, `config.py`, `MockGenerator`, `MockJudge`, CI | 1 | Day 2 (CP0) |
| Everyone | `Ledger` and `current_ledger` | 4 (in 1's scaffold PR) | Day 2 |
| 1, 2, 4 | `prompts.format_steps`, `split_steps`; `grading.extract_answer`, `is_equivalent` | 3 | Day 4 |
| 2 (self-judge), 4 (pilot) | A running Gemma server URL, or a notebook template, plus confirmed logprobs support | 3 | Day 6 |
| 1 (smoke test), 4 (pilot) | A working `JevJudge` with cache and budget | 2 | Day 6 |
| 4 (pilot) | Recorded Jev fixtures and 3 phrasings of the `sound` question | 2 | Day 7 |
| 4 (pilot) | The stratified 200-problem pilot set and the 50-problem dev set (functions in `datasets.py`) | 3 | Day 5 |
| 2, 3 (baselines) | `eval/runner.py` and the `Method` protocol | 4 | Day 9 |
| 3 (B2 matching) | TZ mean completion tokens per `n_simulations` | 1 and 4 | Day 16 |
| 4 (report) | Everyone's section text and their final JSONL | all | Day 26 |

**If you're blocked, say so in the chat the same day.** Never sit blocked. Build against the mock and keep going.

## B6. Git workflow

**One-time setup**
```bash
git clone <repo-url> thoughtzero && cd thoughtzero
git config core.autocrlf input        # Windows users: avoids CRLF diffs
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env                  # fill in YOUR OWN keys; never commit .env
pre-commit install                    # if configured: runs ruff on commit
```

**Branches:** `main` is protected: no direct pushes, CI must be green, and every PR needs 1 approving review.
- Branch names: `p<N>/<area>-<short-desc>`, for example `p1/search-puct`, `p2/judge-jev-parsing`, `p3/data-math500`, `p4/eval-metrics`.
- Keep branches **short-lived** (1–3 days). Prefer small PRs (< 400 changed lines).
- Before opening a PR: `git fetch && git rebase origin/main`, then `ruff check . && ruff format --check . && pytest`.

**Commits:** imperative mood with an area prefix, e.g. `search: implement PUCT selection`, `judge: parse choice distribution`.

**PR template** (Person 1 adds it to `.github/pull_request_template.md`):
- What / why
- Which spec section(s) it implements
- Tests added
- Does it touch a shared file? (If yes, tag all 4.)
- Does it change a prompt? (If yes, `PROMPT_VERSION` bumped?)
- Cost incurred (USD, GPU-hours)

**Reviewers:** 1 ↔ 4 review each other, and 2 ↔ 3 review each other, by default. Anyone can review anything.

**Never commit:** `.env`, API keys, `results/`, cache directories, model weights, notebook outputs containing keys, or anything larger than ~1 MB. Small JSON test fixtures *are* committed.

## B7. Environment

- Python **3.11+**. Dependencies are in spec §12; the package is installed editable.
- `.env` keys: `TYPESAFE_API_KEY`, `OPENROUTER_API_KEY`, `GEMMA_BASE_URL`, `GEMMA_MODEL`. **Each person uses their own keys.** Never paste a key in chat, a PR, a notebook, or a log.
- `pytest` must pass with **no network and no keys**. Tests that need the network are marked `@pytest.mark.network` and skipped in CI.
- Code style: type hints everywhere, `ruff` clean, `mypy src` clean (or explain any `# type: ignore`).

## B8. Money and compute rules

**Jev**
- Price: $0.042 per 1M input tokens. The total project budget is **$5.00** (spec default).
- Suggested split: $1.00 per person plus a $1.00 reserve held by Person 1.
- Set `budget.max_usd` in your own runs to your remaining share.
- **Ask Person 1 (and therefore the whole team) before any run estimated above $1.** Scripts refuse without `--yes` above $1.
- **Cache sharing:** the Jev cache is a local SQLite DB. Never put it in a synced folder (Drive or Dropbox) that two people write to at once, because that corrupts it.
  - Use `python -m thoughtzero.judge.cache export|import <file.jsonl>` (Person 2 builds this) to merge caches.
  - Before a big run, import the latest team cache so you don't re-pay for calls.
- Report your spend at every Friday sync.

**GPU (Kaggle)**
- Each Kaggle account gets roughly **30 GPU-hours per week** (T4 ×2), with 12-hour session limits. Four accounts give ~120 h/week. **VERIFY** current quotas.
- Run experiments **inside the Kaggle notebook** that hosts the Gemma server, using `localhost`: `pip install git+<repo>` and then run the script. Don't expose the server publicly.
- Every long run must be **resumable**. The runner skips problem IDs already in `per_problem.jsonl`, so a killed 12-hour session loses nothing.
- Download results at the end of each session (Kaggle "Output" or a private dataset).
- Ask the team before downloading weights larger than 5 GB (spec §13).
- Log GPU-hours per person in the weekly notes.

## B9. Working with Claude Code

Each of you can use Claude Code in the repo. Start every session with:

> "Read `SPEC.md`, `CLAUDE.md` and `team/PERSON_<N>.md`. I am Person N. Only edit files I own (§B2); for anything else, tell me what change to request from its owner."

Rules from spec §13 apply to everyone and to Claude:
- no paid API calls in tests;
- ask before runs costing more than $1, weights over 5 GB, or protocol changes;
- never print or log keys;
- resolve VERIFY items from official docs and record them in `docs/verified_apis.md` with links;
- every results folder includes the resolved config and the git commit hash.

## B10. Definition of done for any PR

- [ ] Implements a named piece of the spec (section referenced in the PR).
- [ ] Unit tests added; `pytest` green offline.
- [ ] `ruff` and `mypy` clean.
- [ ] Only touches your owned files, or the owner approved.
- [ ] No keys, no results, no large files.
- [ ] If it touches a prompt: `PROMPT_VERSION` bumped.
- [ ] If it touches an interface or the config schema: all 4 tagged, and the mocks updated.
- [ ] Docstrings on public functions; any surprising decision explained in a comment.

## B11. Open decisions (Person 1 raises these at CP0 and CP2; the team decides)

1. **Test leakage in the pilot.** The spec's pilot uses 200 MATH-500 problems, and the main evaluation is also on MATH-500. Tuning judge prompts on pilot problems then leaks into the test results. **Proposal:** draw the pilot (and all prompt tuning) from the **MATH train split** (Hendrycks MATH), and keep MATH-500 untouched until Phase 4. *This changes the protocol, so the team must agree.*
2. **Interface additions** (B4): `Generator.sample_completions` and the `Ledger`/`ContextVar` accounting. Confirm at CP0.
3. **Phase overlap.** The spec says not to start Phase 2 until the pilot is done. We overlap them: MCTS is built on mocks during the pilot, because it's needed under every pilot outcome. The real gate (no real experiments before CP2) is kept.
4. **Budget.** The rough estimate for the full plan is $2–4 of Jev spend (Person 2 refines it after CP1). If the estimate exceeds $5, decide at CP2 whether to raise the cap or shrink the sweeps.
5. **Option position bias.** LLM judges often favour the first option. Person 2 adds `judge.shuffle_options` (default `false`, per spec) and measures its effect in the pilot. The team decides at CP2 whether to turn it on.
