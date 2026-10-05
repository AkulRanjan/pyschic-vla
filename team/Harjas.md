# Person 4 (Harjas): Evaluation and pilot

> **Read this whole file, then `SPEC.md` §1–§5, §7 (pilot), §8 (experiments), §9 (metrics) and §11–§13.** Part A is your personal work plan. Part B (at the bottom) is shared team information, identical in everyone's file.

---

# PART A — Your work

## A1. Your mission

You own **measurement**. That means:
- the **go/no-go pilot** (spec §7), the first real result of the project, which decides whether Jev is used at all;
- the **experiment runner** every method runs through;
- the **metrics** and **plots**;
- the **accounting ledger** that produces our primary compute axis;
- the **final report**.

If your numbers are wrong, the project's conclusions are wrong. Favour correctness, determinism and resumability over cleverness.

## A2. What you own

```
src/thoughtzero/accounting.py         # Ledger, current_ledger ContextVar (shared contract, B4)
src/thoughtzero/eval/runner.py        # Method protocol, run_method over a dataset, JSONL writer, resume
src/thoughtzero/eval/methods.py       # registry: method name -> Method (TZ, B1..B5, C, ablations)
src/thoughtzero/eval/metrics.py       # accuracy+CI, AUROC, Brier, ECE, reliability, per-depth, paired tests
src/thoughtzero/eval/plots.py         # accuracy-vs-compute, reliability diagrams
src/thoughtzero/pilot/traces.py
src/thoughtzero/pilot/mc_label.py
src/thoughtzero/pilot/score.py
src/thoughtzero/pilot/report.py
scripts/run_pilot.py, scripts/run_experiment.py, scripts/make_plots.py
configs/pilot.yaml, configs/exp_main.yaml, configs/ablations.yaml
results/REPORT.md (assembly), README.md (final polish in week 4)
tests/test_metrics.py, test_runner.py, test_pilot.py, test_accounting.py
```

**You do NOT own:** `search/` (1), `judge/` (2), `llm/` / `data/` / `baselines/` (3).

## A3. Days 1–2: accounting and metrics (no dependencies, start now)

### A3.1 `accounting.py`: write it on day 1 and give it to Person 1 for the scaffold PR
- The `Ledger` dataclass, exactly as in B4.
- `current_ledger: ContextVar[Ledger | None]` (default `None`).
- `record_gemma(...)` / `record_jev(...)` are **no-ops when no ledger is set**, so unit tests don't need one.
- `@contextmanager ledger_scope() -> Ledger` sets a fresh ledger, times the wall clock, and resets the ContextVar on exit.
- Make it **safe under asyncio**: tasks created inside the scope inherit the ContextVar, and they all mutate the *same* `Ledger` object, which is fine because asyncio is single-threaded. **Don't** use it from threads. If Person 3's `math-verify` timeout uses threads, that code never records anything.
- `test_accounting.py`:
  - nested tasks accumulate into one ledger;
  - two concurrent problems in two scopes don't mix;
  - recording with no ledger is a no-op.

### A3.2 `metrics.py`: pure functions, all tested against known values
- `accuracy(correct: list[bool]) -> float`, and `bootstrap_ci(values, stat=np.mean, n=1000, alpha=0.05, seed=0) -> (lo, hi)` (spec §8.3: 1,000 resamples, 95%).
- **`paired_bootstrap_diff(correct_a, correct_b, ...)`**: CI on the accuracy *difference* between two methods **on the same problems**. Overlapping individual CIs are a weak test; this is the right one for "TZ vs self-consistency".
- `auroc(scores, labels)` via scikit-learn. Return `nan` (with a warning) if only one class is present, which happens in small depth buckets.
- `brier(probs, soft_labels)` gives the mean of (p − y)². It works with soft labels.
- `ece(probs, soft_labels, n_bins=10)` with equal-width bins: Σ (n_b / N) · |mean(p) − mean(y)| over non-empty bins.
- `reliability_data(probs, labels, n_bins=10) -> list[{bin_lo, bin_hi, mean_pred, mean_label, count}]`.
- `by_depth(records, buckets=[(1,2), (3,5), (6,9), (10, inf)], fn)` applies any metric per depth bucket (spec §7.5).
- Search diagnostics (spec §9):
  - mean expansions;
  - max depth;
  - fraction of most-visited paths ending terminal;
  - `most_visited` vs `value_vote` agreement.
- `test_metrics.py`:
  - AUROC of perfect, inverted and random scores → 1, 0, ≈0.5;
  - ECE of a perfectly calibrated synthetic set ≈ 0;
  - hand-computed Brier;
  - bootstrap CI contains the mean and is deterministic with a seed;
  - the single-class AUROC case.

## A4. Week 1 (days 3–7): runner and pilot pipeline on mocks

### A4.1 `runner.py`: one runner for every method
```python
class Method(Protocol):
    name: str

    async def solve(self, problem: Problem) -> MethodResult: ...


@dataclass
class MethodResult:
    answer: str | None
    raw: dict  # method-specific extras (e.g. tree_dump, votes, samples)
```
`run_method(method, problems, cfg) -> RunSummary`:
- Concurrency across problems: `eval.concurrency` (default 4). Each problem runs inside its own `ledger_scope()`.
- Grading uses Person 3's `is_equivalent(answer, problem.answer)`. **Grading happens only here, after the method returns.** Methods never see `problem.answer`.
- Writes **one JSONL line per (problem, method) immediately** after it finishes (append and flush), with:
  - `problem_id`, `source`, `level`, `subject`, `method`, `answer`, `gold`, `correct`;
  - every `Ledger` field;
  - `config_hash`, `PROMPT_VERSION`, `git_commit`, `seed`, `timestamp`;
  - `raw` (for TZ: the compact tree dump from Person 1).
- **Resume:** on start, read the existing `per_problem.jsonl` and skip `(problem_id, method)` pairs already present. Essential for 12-hour Kaggle sessions.
- **Error isolation:** an exception on one problem writes an error row (`"error": "..."`, `correct: false`, flagged) and the run continues. `BudgetExceeded` is the exception: it **stops the whole run** cleanly.
- Writes `results/<run_id>/config.yaml` (via Person 1's `dump_resolved`) and `git_commit.txt` (`git rev-parse HEAD`, plus a `-dirty` flag if there are uncommitted changes) at start. This is spec §13.7.
- At the end, writes `summary.csv`: per method, accuracy with CI, mean and median completion tokens, Jev calls and USD, wall time, N problems, N errors.
- Supports **sharding**, `--shard i/n`, so several Kaggle accounts can split one run, plus a `merge_runs` utility that concatenates shards and de-duplicates.
- Prints a **pre-run cost estimate** and requires `--yes` above $1 (using Person 2's estimator).

### A4.2 `methods.py`: registry
It maps config names to Method objects: `tz` (Person 1's `search`, wrapped), `cot`, `sc`, `bon`, `c31b` (Person 3's baselines), and the judge variants (`tz_self`, `tz_prm`, `tz_hybrid`, `tz_prior_only`, `tz_value_only`) built through Person 2's `make_judge`. Use stubs until their code lands.

### A4.3 `scripts/run_experiment.py`
```
python scripts/run_experiment.py --config configs/exp_main.yaml \
    --set eval.methods=[tz] --set search.n_simulations=16 --run-id tz16_math500 \
    [--shard 0/3] [--yes]
```

### A4.4 Pilot pipeline (spec §7), built and tested on mocks this week
1. **`traces.py`**: for each pilot problem (Person 3's `pilot_subset(200)`), call `generator.sample_solutions(problem, n=1, temperature=0.7)`, split it into steps with `prompts.split_steps`, and store `{problem_id, steps, final_answer, trace_correct}`.
2. **Prefixes:** for each trace, take `steps[:i]` for `i = 1..len`. **Cap at 8 per trace, chosen evenly across depth** (`np.linspace(1, len, 8).round()` de-duplicated). Store `depth = i` and `rel_depth = i / len`.
3. **`mc_label.py`**: for each prefix, `generator.sample_completions(problem, prefix, n=8, temperature=0.7)`, grade each completion's final answer, and set:
   - `soft_label = correct / 8`;
   - `hard_label = soft_label > 0`.

   Note the edge case where the prefix *already contains* the final answer (the last prefix of a trace). There the label is just whether that answer is correct, and no sampling is needed. **This is the most GPU-heavy step** (~12,800 completions). Run it sharded and resumable, and coordinate GPU time with Person 3.
4. **`score.py`**: for each prefix, run `judge.step_sound(problem, prefix)` for **JevJudge**, **GemmaSelfJudge**, and **PRMJudge** if available. Record the score, latency and cost. Everything goes through Person 2's cache. Store each judge's scores in a separate JSONL so they can be re-run independently.
5. **Prompt sensitivity:** on a 50-problem subset of the pilot, score with all 3 `SOUND_VARIANTS` (from Person 2) and report AUROC per variant.
6. **`report.py`** writes `results/pilot/report.md` plus PNGs:
   - per judge: AUROC vs `hard_label`, Brier and ECE vs `soft_label`, with bootstrap CIs (**resample by problem, not by prefix**, because prefixes of the same trace are correlated);
   - a reliability diagram per judge;
   - AUROC and ECE **by depth bucket** (1–2, 3–5, 6–9, 10+), with counts per bucket;
   - a prompt-sensitivity table;
   - Jev cost (total USD, per call) and a latency histogram (p50/p95);
   - label statistics: the fraction of prefixes with `hard_label=1`, and the soft-label distribution. If almost every prefix is positive, AUROC is noisy; report it;
   - **the decision, printed in bold** per spec §7.6:
     - **GO** if Jev AUROC ≥ 0.75;
     - **PARTIAL** if 0.65 ≤ AUROC < 0.75;
     - **NO-GO** if AUROC < 0.65.

     Also print the CI. If the CI straddles a threshold, say so explicitly; the team decides at CP2.
7. **`scripts/run_pilot.py`** runs the steps with `--stage traces|label|score|report|all`, so each can be re-run alone.

### A4.5 Tests (week 1)
- `test_runner.py`, with mocks:
  - resume skips finished problems;
  - an error row is written and the run continues;
  - `BudgetExceeded` stops the run;
  - sharding plus merge reproduces the unsharded run;
  - `summary.csv` matches the JSONL.
- `test_pilot.py`, with mocks:
  - prefix sampling gives ≤ 8 prefixes, evenly spread and including the last;
  - label math is right;
  - the report renders from a tiny synthetic dataset and prints the right decision for AUROCs of 0.80, 0.70 and 0.60.

## A5. Days 8–10: run the pilot (you drive it; CP2 is day 10)

- **Day 8:**
  - traces (GPU, with Person 3);
  - sanity checks on 20 traces: are steps sensible, do answers parse, and is the trace accuracy plausible for a 4B model on MATH (roughly 40–70%)? Tell the team if it's wildly off.
- **Days 8–9:** Monte Carlo labels (GPU-heavy, sharded across accounts).
- **Day 9:** scoring with Jev (cheap; confirm Person 2's estimate first), self-judge (GPU), and PRM (GPU 1, if available).
- **Day 10:** the report. Send `results/pilot/report.md` to the team **before** the CP2 meeting. Present it at the meeting. Person 1 records the decision.
- If the labelling runs late, report on what you have, but say so, and don't change the decision thresholds after looking at the numbers.

## A6. Days 11–14: runner hardening and CP3

- Integrate the real methods as they land: TZ (Person 1), B1–B3 and C (Person 3), B4 and B5 (Person 2).
- CP3: **every method runs on dev-50** and writes comparable JSONL in one `run_id` folder; `summary.csv` covers all of them.
- `plots.py`:
  - **accuracy vs mean completion tokens per problem** (log x), one line per method with 95% bootstrap CI error bars, with TZ points labelled by `n_simulations` and B2 by N, and C as a horizontal dashed line (its tokens noted);
  - reliability diagrams (pilot);
  - an optional accuracy-by-difficulty-level plot (spec §11: "analyze where search helps by difficulty").
- `scripts/make_plots.py --runs results/<id1> results/<id2> ...` merges runs.

## A7. Week 3 (days 15–21): main experiments (CP4)

- Coordinate the run matrix (spec §8.3) in a shared table: each run has an owner, a Kaggle account, a shard, a status and its cost.
  - Person 1 runs TZ; Person 2 runs B4/B5; Person 3 runs B1–B3 and C.
  - You own the merge, the plots and the tables.
- Compute the **TZ mean completion tokens per `n_simulations`** as soon as the TZ runs land, and send them to Person 3 for B2 matching (with Person 1).
- Produce the **main results table**:
  - one row per method and compute level;
  - columns: accuracy [CI], mean completion tokens, wall time, Jev USD;
  - the **paired bootstrap difference vs B2 at matched compute**.
- Report MATH-500 by difficulty level, and AIME split into pre- and post-cutoff.
- Run a sanity audit:
  - the error-row count per method is small;
  - the same problem IDs appear across methods;
  - the grader agrees with a manual check on 20 random rows.

## A8. Week 4 (days 22–28): report and README (CP5)

- Assemble `results/REPORT.md`. You own the structure and the results; each person writes their own sections (B5, day 26):
  1. Summary: claim, headline result, pilot decision
  2. Method (Person 1)
  3. Judges (Person 2)
  4. Setup and baselines (Person 3)
  5. Pilot results (you)
  6. Main results: plot, table, paired tests (you)
  7. Ablations (each owner)
  8. Limitations: be honest. If TZ doesn't beat self-consistency, say so and analyse where it helps (spec §11).
  9. Reproduction commands: exact, copy-pasteable, tested on a fresh clone with Person 1
- Polish `README.md` for the open-source release: what it is, the headline plot, install, quickstart with `smoke_test.py --mock`, how to reproduce, citation, licence.
- Your ablation: the **spec §9 search diagnostics** across all TZ runs, and `extract_mode` agreement analysis (shared with Person 1).

## A9. Pitfalls specific to you

| Pitfall | Avoid by |
|---|---|
| Bootstrapping over prefixes (correlated) | Resample by **problem** |
| Overlapping CIs used as a significance test | Paired bootstrap on the difference |
| Moving the GO/NO-GO goalposts after seeing the numbers | Thresholds fixed in code and printed before the numbers |
| Losing a 12-hour run | Append-and-flush JSONL, resume, sharding |
| Mismatched problem sets across methods | One subset function; assert identical ID sets before comparing |
| Ledger mixing between concurrent problems | `ledger_scope()` per problem; test it |
| Single-class depth buckets crashing AUROC | Return `nan` and show counts |
| Results without provenance | `config.yaml`, `git_commit.txt`, `PROMPT_VERSION` in every row |
| Ground truth passed into a method | Runner grades after `solve()`; methods get the question only |

## A10. Your checklist

- [ ] Day 1: `accounting.py` written and handed to Person 1 for the scaffold; `metrics.py` started
- [ ] Day 2: scaffold PR reviewed; `metrics.py` and its tests merged
- [ ] Day 5: `runner.py` (resume, shard, errors, summary) merged with tests
- [ ] Day 7: pilot pipeline end-to-end on mocks; decision logic tested; CP1
- [ ] Day 9: traces and Monte Carlo labels done; all judges scored
- [ ] Day 10: `results/pilot/report.md` sent before the meeting; CP2 decision
- [ ] Day 14: all methods on dev-50 via the runner; plots working; CP3
- [ ] Day 21: main table, accuracy-vs-compute plot, paired tests; CP4
- [ ] Day 28: `REPORT.md` assembled; README polished; reproduction verified; CP5


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
