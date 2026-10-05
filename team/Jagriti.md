# Person 2 (Jagriti): Judge (Jev and alternative judges)

> **Read this whole file, then `SPEC.md` §1–§5 (especially §3 and §5.3–§5.4), §7 and §11–§13.** Part A is your personal work plan. Part B (at the bottom) is shared team information, identical in everyone's file.

---

# PART A — Your work

## A1. Your mission

The whole research claim rests on one question: **can Jev judge reasoning steps?** You own every judge:
- the Jev client;
- the disk cache and the budget guard that make reruns free and overruns impossible;
- the alternative judges that baselines and ablations need (Gemma self-judge, PRM, uniform/constant, hybrid).

You also own the **Jev API facts**. The spec has several **VERIFY** items about Jev, and one of them is critical: *does `choice` return a full probability distribution?* Nobody may guess field names (spec §13.5), so your day-1 research unblocks the team.

## A2. What you own

```
src/thoughtzero/judge/base.py        # Judge protocol re-export + helpers (renormalize, eps-fill)
src/thoughtzero/judge/client.py      # JevClient: transport (direct HTTP / OpenRouter / SDK)
src/thoughtzero/judge/jev.py         # JevJudge: request building, parsing, prior_mode, truncation
src/thoughtzero/judge/cache.py       # DiskCache + export/import CLI
src/thoughtzero/judge/budget.py      # BudgetGuard, BudgetExceeded, cost estimation helpers
src/thoughtzero/judge/uniform.py     # UniformJudge, ConstantValueJudge
src/thoughtzero/judge/hybrid.py      # HybridJudge(prior_from=..., value_from=...)
src/thoughtzero/judge/self_judge.py  # GemmaSelfJudge (B4)
src/thoughtzero/judge/prm.py         # PRMJudge (B5, optional)
src/thoughtzero/judge/factory.py     # make_judge(cfg) -> Judge
llm/prompts.py → the "JUDGE" section only (state formatting, question texts, self-judge prompts)
tests/test_jev_parsing.py, test_cache_budget.py, test_jev_client.py, test_judges_misc.py
tests/fixtures/jev/*.json           # REAL recorded responses (keys stripped)
docs/verified_apis.md → "Jev" section
```

**You do NOT own:** `search/` (1), `llm/gemma.py` / `data/` / `baselines/` (3), `eval/` / `pilot/` (4). `prompts.py` belongs to Person 3; you edit only the judge section, and every change bumps `PROMPT_VERSION`.

## A3. Days 1–2: get access and resolve the VERIFY items (start immediately)

You don't need the scaffold for this. **Start on day 1, hour 1.**

### A3.1 Access (some of it can take days)
1. Join the **TypeSafe direct-API waitlist** (`https://api.typesafe.ai/v1/systemone`) today. It may take a while.
2. Meanwhile, get an **OpenRouter** key with a few dollars of credit and check that `typesafe/jev-1.13` is available. Also look at the Cloudflare Workers AI (`typesafe/jev`) and Vercel AI Gateway (`typesafe-ai/jev`) routes as backups.
3. **Read the Terms of Service** of TypeSafe and of whichever gateway you use. In particular, check whether publishing benchmark results or stress-testing is allowed (spec §3). Summarize this in `docs/verified_apis.md` and flag anything worrying to the team.

### A3.2 Resolve every Jev VERIFY item and write it down
Write each answer, with links, in `docs/verified_apis.md` → **Jev**:

| Question | Why it matters |
|---|---|
| Does `choice` return a **probability for every option**, or only the top pick? | **Critical.** It decides `prior_mode` (`choice` vs `per_candidate_noul`) |
| Exact request field names: `state`, `questions`, question `type`, `instructions`, options field (`choices`? `options`?), `model` | We may not guess (spec §13.5) |
| Exact response shape: where each question's answer and probabilities live; how `noul` yes/no probabilities appear | Parsing |
| Does the response report **usage** (input tokens)? | Accounting, otherwise estimate `chars / 4` |
| Is the model pinnable to `jev-1.13.0` on *each* route (direct and OpenRouter)? | Reproducibility |
| Does OpenRouter use the same typed `questions` request, or a chat-style wrapper? | Transport design |
| Rate limits (RPM / TPM), concurrency limits | Semaphore default, backoff |
| Max options in a `choice` (spec says 255) and max context (32K) | Truncation |
| The `typesafe-sdk-python` API (sync and async): is it cleaner than raw HTTP? | Client choice |
| Latency, measured: p50 / p95 over ~20 calls | Pilot report, planning |

### A3.3 Record real fixtures (costs fractions of a cent)
Make ~10 real calls with a throwaway script and save the raw responses, **with headers and keys stripped**, to `tests/fixtures/jev/`:
- `noul_basic.json`
- `choice_4_options.json`
- `choice_and_noul_together.json` (our actual per-expansion shape)
- `choice_1_option.json`
- an error response: a 4xx with a bad field name
- a 429, if you can trigger one politely; otherwise hand-write one and label it `synthetic_429.json`

These fixtures let all tests run offline forever.

## A4. Week 1 (days 3–7): JevJudge, cache and budget

### A4.1 `client.py`: transport
- `class JevClient` with `async def ask(state: str, questions: dict) -> JevResponse`. Backends: `direct`, `openrouter` (and `sdk` if it's cleaner), selected by `judge.transport` in the config.
- Uses `httpx.AsyncClient` (one instance, reused, with a timeout of ~30 s). Reads the key from the environment only (`TYPESAFE_API_KEY` or `OPENROUTER_API_KEY`). **Never log headers.**
- Retries via `tenacity`: exponential backoff with jitter on **429 and 5xx and timeouts**, max 5 attempts. **Fail fast on other 4xx**, since a bad request will never succeed.
- An `asyncio.Semaphore(judge.max_concurrency)` (default 16).
- Measures latency per call (for the pilot report).

### A4.2 `jev.py`: JevJudge (spec §5.3)
- **State text** comes from `prompts.judge_state(problem, steps)`:
  ```
  PROBLEM:
  <problem>

  SOLUTION SO FAR:
  Step 1: ...
  Step 2: ...
  ```
  Use `prompts.format_steps` (Person 3) for numbering. Steps are stored without the prefix (B4 convention).
- **`prior_and_value(problem, steps, candidates)`**: **one request** with two questions:
  - `sound` (`noul`): "Is every step in SOLUTION SO FAR mathematically correct and logically valid? Ignore whether the solution is finished."
  - `next` (`choice`): "Which candidate next step is most likely to lead to a correct final answer?", with options `c0..c{K-1}` mapped to the candidate text.
  - Parse: `V = P(yes)` for `sound`, and the priors are the probabilities for `c0..c{K-1}`. **Fill any missing option with ε (1e-3) and renormalize.** If all are missing, use uniform priors and log a warning.
  - **Empty prefix (the root):** there are no steps yet, so `sound` is trivially "yes". Either skip the `sound` question and return `V = 0.5`, or a neutral constant. Decide, document it, and make it configurable (`judge.root_value`).
- **`prior_mode: per_candidate_noul`** (fallback): K `noul` questions in the **same request**, `cand_i`: "Does the following step correctly continue the solution so far? STEP: <candidate i>", plus `sound`. Priors = yes-probabilities normalized. Implement it even if `choice` works, because it's an ablation (spec §8.4).
- **`final_correct(problem, steps)`**: one `noul`, "Is the final answer of this solution correct?", over the full solution (state heading "SOLUTION:").
- **`step_sound(problem, steps)`**: the `sound` question alone. The pilot uses it.
- **Long states:** estimate tokens as `chars / 4`. If the estimate exceeds 28K, keep the problem, step 1 and the last N steps that fit, replace the middle with `[... k steps omitted ...]`, and **log a warning** (spec §5.3). Count truncations in stats.
- **Option shuffling (optional, B11.5):** if `judge.shuffle_options` is set, permute candidates with a deterministic permutation seeded by `hash(state)`, then un-permute the priors. Default is off.
- Every request goes **cache → budget → client** and calls `accounting.record_jev(...)`.

### A4.3 `cache.py`: DiskCache (spec §5.4)
- Key: `sha256(canonical_json({"request": state_and_questions, "model": jev_model, "prompt_version": PROMPT_VERSION}))`. **Leave the transport out of the key**, so a direct-API hit and an OpenRouter hit are interchangeable if the model version is identical. Canonical JSON means `sort_keys=True, separators=(",", ":"), ensure_ascii=False`.
- Value: `{raw_response, parsed, input_tokens, latency_ms, created_at}`.
- Backend: `diskcache.Cache` (SQLite) at `JEV_CACHE_DIR` (default `.cache/jev`).
- **CLI:** `python -m thoughtzero.judge.cache export out.jsonl` and `... import in.jsonl` (import skips keys that already exist). This is how the team shares paid calls (B8).
- Cache hits cost $0 and record `cache_hit=True`.

### A4.4 `budget.py`: BudgetGuard
- `usd(input_tokens) = input_tokens * 0.042 / 1e6`. Keep the price in config, not hard-coded twice.
- `check(estimated_tokens)` runs **before** each call: raise `BudgetExceeded` if `spent + estimate > max_usd`. `record(actual_tokens)` runs after.
- Safe under asyncio concurrency. Reserve the estimate before the `await`, and settle after; otherwise 16 concurrent calls can all pass the check at once.
- `estimate_run_cost(n_calls, avg_tokens) -> float` plus a helper that scripts use to print a pre-run estimate. **Refuse without `--yes` if the estimate is over $1** (spec §5.4).
- Also append every paid call to `results/jev_spend.jsonl` (timestamp, person, tokens, usd, run_id), so the team can sum real spend. Read the person from the env var `TZ_PERSON`.

### A4.5 Simple judges
- `UniformJudge`: uniform priors and a constant value of 0.5 for everything.
- `ConstantValueJudge(value=0.5)`: used in combination via the hybrid judge.
- **`HybridJudge(prior_from: Judge, value_from: Judge)`.** The spec doesn't name this, but several things need it:
  - ablation *prior only*: Jev prior + constant value;
  - ablation *value only*: uniform prior + Jev value;
  - the **PARTIAL / NO-GO** outcome: Jev prior + PRM value;
  - **B5**: self-judge prior + PRM value.

  Run the two sub-calls concurrently. Make sure Jev isn't charged for a `sound` question whose answer gets thrown away; add a flag to `JevJudge` for prior-only and value-only requests.
- `factory.make_judge(cfg)` builds any of these from the config (`judge.kind: jev | self | prm | uniform | hybrid`).

### A4.6 Tests (week 1), all offline
- `test_jev_parsing.py`, against every fixture: priors sum to 1; ε-fill when options are missing; uniform plus a warning when all are missing; the top-pick-only response triggers a clear error that says to switch `prior_mode`; `per_candidate_noul` parsing; malformed JSON gives a clear error.
- `test_jev_client.py` with `respx`: a 429 then 200 succeeds after a retry; 500 ×5 raises; a 400 is **not** retried; the semaphore caps concurrency.
- `test_cache_budget.py`: the second identical call is a cache hit with no HTTP and $0; a `PROMPT_VERSION` change gives a cache miss; the budget raises at the cap; 16 concurrent calls don't overshoot; export/import round-trips.
- Truncation test: a fake 40K-token state gets truncated, keeps the problem and the last steps, and warns.

### A4.7 Deliver by day 6–7 (CP1)
A working `JevJudge` that Person 1 can use in `smoke_test.py`, a complete Jev section in `verified_apis.md`, and the fixtures committed.

## A5. Days 8–14: pilot support, self-judge and PRM

### A5.1 Pilot support (Person 4 runs the pilot)
- Give Person 4 **3 phrasings** of the `sound` instruction for the prompt-sensitivity check (spec §7). Keep the first one identical to the main phrasing. Put all three in `prompts.py` as `SOUND_VARIANTS`.
- Give Person 4 a **pilot cost estimate**: about 200 traces × ≤ 8 prefixes ≈ 1,600 `step_sound` calls × average tokens, plus 3 phrasings × the 50-problem subset. It should be well under $1, but compute it.
- If `shuffle_options` is in play, measure position bias: on ~100 expansions, compare the prior on the same candidate in position 0 vs position K−1.

### A5.2 `self_judge.py`: GemmaSelfJudge (baseline B4)
- The same questions as Jev, but posed to Gemma as completion prompts (judge section of `prompts.py`). Read the answer from **next-token logprobs** on the completions endpoint (`logprobs` / `top_logprobs`). Person 3 confirms the server supports this and gives you a raw `completions` helper with logprobs.
- **Value:** the prompt ends in `... Answer (Yes or No):`, and `P(yes) = softmax over {Yes-variants, No-variants}`.
  - Sum the probability mass over token variants: `"Yes"`, `" Yes"`, `"yes"`, `" yes"`, and the same for No.
  - If neither appears in `top_logprobs`, return 0.5 and count it.
- **Priors:** list the candidates as `A) ... B) ...` and end with `Best next step (letter):`. Read the logprobs of `A..` (with space variants) and renormalize over the K letters. Missing letters get ε.
- One forward pass per question: `max_tokens=1`, `temperature=0`. Check whether the server caps `top_logprobs` (e.g. at 20) and record that in `verified_apis.md`.
- Gemma calls made by the self-judge count toward **Gemma prompt tokens** in the ledger (`record_gemma`), not Jev.

### A5.3 `prm.py`: PRMJudge (B5, optional)
- Model: `Qwen/Qwen2.5-Math-PRM-7B`. **VERIFY** the ID, the step-separator token format, and how to read per-step scores from the model card, then record them in `verified_apis.md`.
- Value only. `step_sound` and `prior_and_value` return the PRM's score for the **last step**, or the minimum over steps; choose one and document it. `final_correct` returns the score of the final step.
- Memory: a 7B model in fp16 is ~15 GB, so it doesn't fit next to Gemma on one T4. On Kaggle T4 ×2, put **Gemma on GPU 0 and the PRM on GPU 1**, or load the PRM in 4-bit. Coordinate with Person 3. If it doesn't fit, **skip it and say so in the report** (spec §5.3).
- Batch PRM calls (several states per forward pass) behind an async queue, so they don't run one at a time.
- Keep `torch` imports inside this module (optional extra `[prm]`), so the package imports without torch.

### A5.4 CP2 (day 10)
Bring the Jev-specific numbers to the decision meeting: AUROC per phrasing, cost, latency p50/p95, position bias (if measured), and any parsing anomalies.

## A6. Week 3 (days 15–21): judge-side experiments

- Run **B4** (TZ with GemmaSelfJudge) and **B5** (TZ with PRM value + self-judge prior) on MATH-500 at the same `n_simulations` sweep as Person 1. Use Person 4's runner.
- If CP2 was **PARTIAL** or **NO-GO**, you also own making the hybrid the main configuration (Jev prior + PRM value).
- Report **actual** Jev spend versus the estimate, and update the budget numbers for week 4.
- Optional variant from spec §11: **per-depth temperature calibration** of Jev values, fitted on pilot data (train split only) and reported as a separate variant, never silently.

## A7. Week 4 (days 22–28): ablations and write-up

- Your ablations (spec §8.4): **prior only** (Jev prior + constant 0.5), **value only** (uniform prior + Jev value), and **`prior_mode` choice vs per-candidate noul**. Use the same fixed subset as Person 1's ablations.
- Write these report sections:
  - **Judges** (how each judge works, exact question texts, `PROMPT_VERSION`);
  - **Jev API notes** (ToS, versions, latency, cost);
  - your ablation results.
- Make sure `verified_apis.md` is complete and every link works.

## A8. Pitfalls specific to you

| Pitfall | Avoid by |
|---|---|
| Guessing field names | Only use what's documented or seen in a real fixture; cite it in `verified_apis.md` |
| Budget race under concurrency | Reserve the estimate before `await`, settle after |
| Paying twice for the same call | Cache before budget before HTTP; share caches via export/import |
| Cache key missing the prompt or model version | Include `PROMPT_VERSION` and `jev_model` in the key; test that a change gives a miss |
| Retrying on 400 forever | Only retry 429, 5xx and timeouts |
| Keys in logs, fixtures or notebooks | Strip headers before saving fixtures; grep for `sk-`/`key` before committing |
| Position bias in `choice` | `shuffle_options` flag, measured in the pilot |
| Self-judge "Yes" vs " Yes" token mismatch | Sum over token variants; count the misses |
| Root-node value being meaningless | A configurable `judge.root_value`, documented |
| Priors not summing to 1 | Always renormalize; assert in tests |

## A9. Your checklist

- [ ] Day 1: waitlist joined; OpenRouter key; ToS read; first real call made
- [ ] Day 2: all Jev VERIFY items answered in `verified_apis.md`; fixtures recorded; scaffold PR reviewed
- [ ] Day 5: client, cache, budget and their tests merged
- [ ] Day 6: `JevJudge` (both prior modes), uniform/constant/hybrid, factory merged
- [ ] Day 7: CP1, smoke test passes with your judge; `SOUND_VARIANTS` ready
- [ ] Day 10: CP2 with pilot cost estimate, latency and Jev numbers presented
- [ ] Day 14: GemmaSelfJudge merged; PRMJudge merged, or skipped with the reason written; CP3
- [ ] Day 21: B4 and B5 runs done; actual vs estimated spend reported; CP4
- [ ] Day 28: ablations; Judges and API-notes sections written; CP5


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
    id: str  # stable, e.g. "math500/test/precalculus/807"
    question: str
    answer: str  # ground truth; ONLY used for grading, never inside search
    source: str  # "math500" | "aime2025" | ...
    level: int | None = None  # MATH difficulty 1-5 (for stratification)
    subject: str | None = None


@dataclass
class GenOut:
    text: str  # step text, stripped, WITHOUT the "Step n:" prefix
    prompt_tokens: int
    completion_tokens: int


class Generator(Protocol):
    async def propose(self, problem: str, steps: list[str], k: int) -> list[GenOut]: ...
    async def complete(
        self, problem: str, steps: list[str], temperature: float = 0.0
    ) -> list[str]: ...
    async def sample_solutions(self, problem: str, n: int, temperature: float) -> list[str]: ...
    # ADDITION to the spec (needed by the pilot's Monte Carlo labels, one batched n=m request):
    async def sample_completions(
        self, problem: str, steps: list[str], n: int, temperature: float
    ) -> list[list[str]]: ...


class Judge(Protocol):
    async def prior_and_value(
        self, problem: str, steps: list[str], candidates: list[str]
    ) -> tuple[list[float], float]: ...
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
       gemma_completion_tokens: int = 0  # PRIMARY compute axis
       gemma_calls: int = 0
       jev_calls: int = 0
       jev_cache_hits: int = 0
       jev_input_tokens: int = 0
       jev_usd: float = 0.0
       wall_time_s: float = 0.0
       expansions: int = 0
       max_depth: int = 0
       terminal_leaves: int = 0


   current_ledger: ContextVar[Ledger]  # set by eval/runner.py per problem


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

> "Read `SPEC.md`, `CLAUDE.md` and `team/<YourName>.md`. I am Person N. Only edit files I own (§B2); for anything else, tell me what change to request from its owner."

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
