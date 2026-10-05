# Person 1 (Prakhar): Search lead and integrator

> **Read this whole file, then `SPEC.md` §1–§6 and §10–§13.** Part A is your personal work plan. Part B (at the bottom) is shared team information, identical in everyone's file.

---

# PART A — Your work

## A1. Your mission

You own the **heart of the method**: the MCTS engine that searches over reasoning steps (spec §4, §6). You are also the **integrator**. You create the repo and the shared contract everyone codes against, keep `main` green, run the end-to-end smoke tests at each checkpoint, and chair the GO / PARTIAL / NO-GO decision.

Your days 1–2 are on the **critical path**: the other three can't merge code until your scaffold lands. Do it first, do it fast, and keep it small and correct.

## A2. What you own

```
SPEC.md                         # rename thoughtzero-spec.md → SPEC.md
CLAUDE.md                       # working rules from spec §13
README.md                       # skeleton now; Person 4 polishes it in week 4
pyproject.toml, .gitignore, .gitattributes, .env.example, .pre-commit-config.yaml
.github/workflows/ci.yml, .github/pull_request_template.md
configs/default.yaml            # shared: changes by PR to all
src/thoughtzero/__init__.py
src/thoughtzero/types.py        # THE CONTRACT (Part B4)
src/thoughtzero/config.py
src/thoughtzero/mocks.py        # MockGenerator, MockJudge, toy task
src/thoughtzero/logging_setup.py
src/thoughtzero/search/node.py
src/thoughtzero/search/mcts.py
src/thoughtzero/search/extract.py
src/thoughtzero/search/dedupe.py
src/thoughtzero/search/puct.py  # pure selection math (optional split from mcts.py)
scripts/smoke_test.py
scripts/view_tree.py            # pretty-print a tree dump (nice to have, very useful)
tests/test_puct.py, test_mcts_toy.py, test_virtual_loss.py, test_extract.py,
tests/test_dedupe.py, test_config.py
team/{Prakhar,Jagriti,Akul,Harjas}.md             # these files
```

**You do NOT own:** `judge/` (Person 2), `llm/` and `data/` and `baselines/` (Person 3), `eval/` and `pilot/` and `accounting.py` (Person 4). Request changes there via PR or chat.

## A3. Days 1–2: scaffold and contract (critical path)

Goal: **CP0**, a merged scaffold that everyone can branch from. Aim to open the PR by the end of day 1 and merge it on day 2.

### A3.1 Repo
1. Create a GitHub repo (private for now) and invite persons 2–4 as collaborators.
2. Protect `main`: require a PR, 1 approval, and passing CI. No force-push.
3. `git mv thoughtzero-spec.md SPEC.md`. Add `team/{Prakhar,Jagriti,Akul,Harjas}.md`.
4. Create the full folder layout from spec §5.1, with **empty modules containing stubs** (functions that raise `NotImplementedError` with a docstring naming the owner). Everyone can then see where their code goes, and imports already resolve.
5. `.gitignore`: `results/`, `.env`, `.venv/`, `__pycache__/`, `*.egg-info`, `.cache/`, `jev_cache/`, `*.sqlite`, `.ipynb_checkpoints/`, `wandb/`, model weight files.
6. `.gitattributes`: `* text=auto eol=lf`. The team is partly on Windows; this prevents whole-file diffs.

### A3.2 `pyproject.toml`
- Package `thoughtzero` with the `src/` layout, `requires-python = ">=3.11"`.
- Core deps from spec §12: `httpx`, `openai`, `pydantic>=2`, `pyyaml`, `python-dotenv`, `tenacity`, `diskcache`, `transformers`, `datasets`, `math-verify`, `numpy`, `scikit-learn`, `matplotlib`, `pandas`.
- Extras:
  - `dev`: `pytest`, `pytest-asyncio`, `respx`, `ruff`, `mypy`, `pre-commit`
  - `prm`: `torch`, `accelerate`
  - `ft`: `peft`
- `[tool.pytest.ini_options]`: `asyncio_mode = "auto"`, `markers = ["network: needs network/keys"]`, `addopts = "-m 'not network'"`.
- `[tool.ruff]`: line-length 100, target py311, rules `E,F,I,UP,B,SIM,ASYNC`.
- `[tool.mypy]`: `strict = false`, `disallow_untyped_defs = true` for `src/`.

### A3.3 CI (`.github/workflows/ci.yml`)
Ubuntu, Python 3.11. Steps: `pip install -e ".[dev]"`, `ruff check .`, `ruff format --check .`, `mypy src`, `pytest`. Cache pip. **No secrets in CI.** Tests must never need them.

### A3.4 `types.py`
Write exactly the contract in Part B4. Add docstrings that state the conventions (the step-prefix rule, priors summing to 1, ground truth never used in search).

### A3.5 `accounting.py`
Co-write this with Person 4 on day 1: they own it, but it ships in your scaffold PR. It holds the `Ledger` dataclass, the `current_ledger` ContextVar, `record_gemma`, `record_jev`, and a `ledger_scope()` context manager the runner uses. If no ledger is set, `record_*` should be a **no-op**, not a crash, so unit tests don't need one.

### A3.6 `config.py`
- Pydantic v2 models: `GeneratorCfg`, `JudgeCfg`, `SearchCfg`, `BudgetCfg`, `EvalCfg`, `PilotCfg`, `Config`. Fields and defaults come from spec §6.4. `EvalCfg` holds `dataset`, `split`, `n_problems`, `subset_seed`, `methods`, `concurrency`, `run_id`, `out_dir`. `PilotCfg` holds `n_problems=200`, `max_prefixes=8`, `m_completions=8`, `temperature=0.7`.
- `load_config(path, overrides: list[str]) -> Config`:
  - load YAML;
  - interpolate `${VAR}` from the environment (after `load_dotenv()`), and raise a clear error if a required variable is missing;
  - apply `--set a.b.c=value` overrides (parse the value with `yaml.safe_load` so numbers and lists work);
  - validate.
- `config_hash(cfg) -> str`: sha256 of canonical JSON (sorted keys) of `cfg.model_dump()`, with any field whose name contains `key`/`token`/`secret` removed.
- `dump_resolved(cfg, path)`: writes `config.yaml` for a results folder, with secrets redacted.
- `test_config.py` covers env interpolation, overrides, hash stability, and redaction.

### A3.7 `mocks.py`: the toy task (important, everyone uses it)
Design a toy problem where **greedy-by-prior fails but search succeeds**. Example:

- **Problem:** "Start at 0. Reach exactly 13. Each step adds 1, 2 or 5." The correct final answer is `13`.
- **`MockGenerator.propose(problem, steps, k)`** returns k candidate steps like `"add 5 -> 10"`, deterministic from `hash((seed, tuple(steps)))`.
  - A step that lands on 13 writes `"add 1 -> 13. \boxed{13}"` (terminal, correct).
  - A step that overshoots writes `"add 5 -> 15. \boxed{15}"` (terminal, wrong).
- **`MockJudge`:**
  - **Priors are deliberately misleading:** the highest prior always goes to "+5". Greedy-by-prior goes 5 → 10 → 15 and is wrong.
  - **Value** is 0.9 while the running total is ≤ 13, and 0.1 otherwise.
  - `final_correct` returns 0.95 if the total is exactly 13, else 0.05.
  - The search should find a path such as 5 → 10 → 12 → 13.
- Constructor options: `latency_s` (an `asyncio.sleep`, for virtual-loss tests), `fail_rate` (raises sometimes, for error-path tests), and call counters.
- Both mocks call `record_gemma` / `record_jev` with fake token counts, so accounting gets exercised.
- Ship a `ToyProblem` with a known answer, and a helper `greedy_by_prior(...)` that tests can compare against.

### A3.8 `CLAUDE.md`
Contains the 9 rules of spec §13, plus "read `team/<YourName>.md` for your role", the ownership rule (B2), and the step-prefix convention.

### A3.9 Scaffold PR checklist
- [ ] `pytest` passes (only config and mock sanity tests at this point)
- [ ] CI green
- [ ] All 4 people approved: they've read `types.py` and agree with the signatures
- [ ] Open decisions B11.1–B11.3 raised in the PR description

## A4. Week 1 (days 3–7): sequential MCTS on mocks

Build §6 **sequentially first**, with no concurrency. Correctness before speed.

### A4.1 `node.py`
`Node` per spec §6.1, plus:
- `depth` (`len(steps)`), `parent` (weakref or plain; plain is fine), `id` (an incrementing int, for dumps);
- `Q` with FPU: `fpu_value` when `N + virtual_loss == 0`;
- with virtual loss: **`Q_eff = W / (N + vl_count)`** and **`N_eff = N + vl_count`**. Virtual visits count as value 0, as §6.2 says.

### A4.2 PUCT (`puct.py`, pure functions)
`select_child(node, c_puct, fpu_value) -> Node`, with
`score = Q_eff(child) + c_puct * child.prior * sqrt(N_parent_eff) / (1 + N_eff(child))`.

Pitfalls:
- **Single-agent: never negate Q.** AlphaZero flips signs between players; we have no opponent. Copying chess code is the #1 bug here.
- Use `sqrt(max(1, N_parent_eff))`. With a parent N of 0, all exploration terms are 0 and the prior is ignored.
- Deterministic tie-breaking: highest prior wins, then lowest index. Tests depend on this.
- Terminal children are selectable. Revisiting one re-backs-up its **cached** value without calling the judge again.

### A4.3 `dedupe.py` (spec §6.3)
- `normalize(text)`: lowercase, collapse whitespace, strip any `Step \d+:` prefix the model may have repeated, and strip trailing punctuation and whitespace.
- `dedupe(cands: list[GenOut], jaccard: float | None = 0.9) -> tuple[list[GenOut], list[int]]` returns the unique candidates **and** a mapping from each original index to its unique index.
- Priors from the judge are over the **unique** list (fewer judge tokens). If you instead call the judge on all k and merge, sum the priors per group. Pick one approach and document it; **calling on unique candidates is preferred.** Spec §6.3 says "sum their priors after the judge call", which refers to near-duplicate groups. Clarify this with Person 2.
- Token Jaccard on whitespace tokens of the normalized text. Make it O(k²); k ≤ 6, so that's fine.
- If one candidate is left, the judge is still called, so the node gets a value.
- Drop empty candidates (Person 3 should already have dropped them, but be defensive).

### A4.4 `mcts.py` (sequential version)
`async def search(problem: str, generator, judge, cfg: SearchCfg) -> SearchResult`, where `SearchResult` holds `answer: str | None`, `answer_steps: list[str]`, `root: Node`, `tree_dump: dict` and `stats: dict`.

- Loop `n_simulations` times: select to a leaf → expand or evaluate → backup.
- Expansion: `propose` → `dedupe` → `judge.prior_and_value(problem, node.steps, [c.text ...])` → `node.value = v` → children via `make_child`.
- `make_child`: terminal if `"\\boxed{" in text` or `"final answer" in text.lower()`, or if `depth + 1 >= max_depth`. If terminal, set `final_answer = grading.extract_answer(...)` (Person 3's function; use a local stub until it lands).
- Terminal leaf: `if node.value is None: node.value = await judge.final_correct(problem, node.steps)`.
- **Backup:** `W += v`, `N += 1` for **every node from root to leaf, inclusive**.
- Update the ledger: `expansions`, `max_depth`, `terminal_leaves`.
- Optional root Dirichlet noise (`root_dirichlet_alpha`, ε = 0.25), applied once when the root is expanded. Seed it from `cfg.seed` plus a hash of the problem.

### A4.5 `extract.py` (spec §4)
- `most_visited(root, generator, problem)`: follow the child with max `N` (ties: higher Q, then higher prior) until terminal. If the path ends non-terminal, `await generator.complete(problem, steps, temperature=0)`, append, and extract the answer.
- `value_vote(root)`: collect **all terminal nodes** (DFS), skip `final_answer is None`, group by `grading.normalize_answer(ans)`, and score each group as `sum(node.N * node.value)`. The highest sum wins. Ties: more total N, then the lexicographically smaller normalized answer.
- Both return `(answer, path_steps, diagnostics)`. Also report whether the two modes **agree**; spec §9 wants that metric.

### A4.6 Tree dump
`dump_tree(root) -> dict`, compact and JSON-serialisable: `{id, step (truncated to 300 chars), N, Q, prior, value, terminal, final_answer, children:[...]}`. Person 4 writes it into the JSONL. `scripts/view_tree.py <jsonl> --problem <id>` prints an indented tree sorted by N.

### A4.7 Tests (week 1)
- `test_puct.py`: hand-computed scores for a 3-child node; FPU behaviour; the `max(1, N)` rule; tie-breaking; no sign flipping.
- `test_extract.py`: hand-built trees for both modes, including a `None` answer, ties, and a path ending non-terminal (mock `complete`).
- `test_dedupe.py`: exact duplicates, prefix variants, Jaccard 0.9 boundary, index mapping, single survivor.
- `test_mcts_toy.py`: **the key test.** On the toy task, `greedy_by_prior` gets the wrong answer, while `search(n_simulations=64)` gets the right answer, for seeds 0–4. Also check that `sum(child.N) == root.N - 1` after search (the root's own first visit), or whatever invariant your backup implies; write it down and test it.

### A4.8 CP1 (day 7): you run the real smoke test
`scripts/smoke_test.py [--mock]`: 3 easy problems end-to-end.
- With `--mock`, it uses the toy task.
- Without it, it uses real Gemma (Person 3's server) and Jev (Person 2). It prints answers, correctness, number of Jev calls, USD and time, and asserts **≤ 20 Jev calls and ≤ $0.01** (use `n_simulations=4`, `k=3`).

Pass all three criteria, then write `results/PHASE_0_NOTES.md`.

## A5. Days 8–14: async MCTS, pilot support, first real run

### A5.1 Async with virtual loss (spec §6.2)
- `parallel_sims` simulations in flight: `asyncio.TaskGroup` plus an `asyncio.Semaphore(parallel_sims)`. Launch until `n_simulations` have **completed**.
- During selection, add `virtual_loss` to every node on the path. **Always remove it in a `finally:`**, so an exception never leaves virtual loss stuck.
- Expansion in flight: `node.expanding = asyncio.get_running_loop().create_future()`. Others reaching the same leaf `await node.expanding`, then use `node.value`.
  - **On failure:** `set_exception`, then reset `node.expanding = None` so a later simulation can retry. Also make the awaiting simulations handle the exception (they abort without backup and log it).
  - Retrieve the future's exception if nobody awaited it, to avoid "Future exception was never retrieved".
- Selection and backup are synchronous (no `await` inside them), so they're atomic on the event loop. Keep it that way; it's why we don't need locks.
- Determinism: with `parallel_sims=1` the async version must reproduce the sequential one exactly. Test this.

### A5.2 `test_virtual_loss.py`
- With `MockJudge(latency_s=0.05)` and `parallel_sims=4`, the first 4 concurrent simulations below the root go to **different** children (k ≥ 4).
- After search, all `virtual_loss == 0`.
- With `fail_rate=0.2`, the search completes, no virtual loss is leaked, and failed simulations are counted in stats.
- Two simulations racing to the same leaf trigger **one** expansion (count `propose` calls).

### A5.3 Pilot week support
Person 4 runs the pilot; you help unblock them. Prepare the **CP2 meeting**: an agenda, the decision rule from spec §7.6, and the open decisions B11.1, B11.4 and B11.5.

### A5.4 After CP2: the first real 50-problem run
- Use the dev-50 subset (Person 3's `datasets.dev_subset(50)`), `n_simulations=16`, `k=4`, with the judge chosen by the CP2 decision.
- Inspect 10 trees with `view_tree.py`. Look for these failure modes and write them down:
  - steps too long or too short;
  - duplicate siblings that dedupe misses;
  - the search never reaching terminals;
  - priors that are near-uniform (Jev not discriminating);
  - values all ≈ 1.0 (overconfident).
- Share the findings at the Friday sync. **Tuning `c_puct`, `k` or the prompts on test data is leakage.** Tune on the train split, or not at all.

## A6. Week 3 (days 15–21): main ThoughtZero runs

- Run TZ on full MATH-500 with `n_simulations ∈ {8, 16, 32, 64}`, plus AIME. Use the method configs in `configs/exp_main.yaml` (Person 4 owns the runner; you own the TZ section).
- Before each run, compute a cost estimate (expansions × average state tokens × price), get team approval if it's over $1, and pass `--yes`.
- Split the runs across Kaggle accounts if needed. All runs are resumable.
- As soon as the n=8/16 runs finish, **give Person 3 the mean completion tokens per problem**, so they can set N for self-consistency (B2) to match.
- Watch the search diagnostics: mean expansions, max depth, the fraction of most-visited paths ending terminal, and `most_visited`/`value_vote` agreement.

## A7. Week 4 (days 22–28): ablations and write-up

- Your ablations (spec §8.4): `k ∈ {2, 4, 6}`, `c_puct ∈ {0.5, 1.5, 3.0}`, `extract_mode`. To stay inside budget, run them on a **fixed subset** (e.g. 200 problems) at one `n_simulations`; agree the subset with Person 4.
- Write the **Method** section of `REPORT.md` and your ablation paragraphs.
- Final integration: a fresh clone, `pip install -e .`, `pytest`, `smoke_test.py --mock` and the reproduction commands from the report. Tag `v0.1.0`.

## A8. Integrator duties (all month)

- Review or merge PRs within 24 h. Keep `main` green; revert rather than leave it broken.
- Own the shared files. Any interface change: update `types.py`, the mocks and the tests in the same PR, and tag everyone.
- Every Friday: run `smoke_test.py --mock` on `main`, collect cost and GPU-hours per person, and write `results/PHASE_<n>_NOTES.md` (what was built, verified, open issues, cost) per spec §13.9.
- Track the team Jev budget (B8) and hold the $1 reserve.
- Chair CP2 and record the decision in `results/pilot/DECISION.md`.

## A9. Pitfalls specific to you

| Pitfall | Avoid by |
|---|---|
| Negating Q (two-player habit) | Single-agent: Q is always "probability this is good"; test it |
| Forgetting root in backup | Backup over the full `path` including root |
| Virtual loss leaked after an exception | `try/finally` around every simulation |
| Terminal node judged again on each visit | Cache `node.value`; count judge calls in tests |
| An infinite loop when the whole tree is terminal | Count *completed* simulations; revisiting terminals still counts as a simulation |
| Ground truth leaking into search | `search()` never receives `Problem.answer`; pass only the question string |
| Non-determinism making tests flaky | Seeded mocks, deterministic ties, `parallel_sims=1` reference test |
| Scaffold PR too big to review | Stubs only; the logic lands later in small PRs |

## A10. Your checklist

- [ ] Day 1: repo, protection, layout, pyproject, CI, `types.py`, `config.py`, mocks, CLAUDE.md, PR opened
- [ ] Day 2: CP0, scaffold merged with 4 approvals
- [ ] Day 5: PUCT, dedupe, extract and node merged with tests
- [ ] Day 6: sequential MCTS and `test_mcts_toy` passing
- [ ] Day 7: CP1 real smoke test passes; `PHASE_0_NOTES.md`
- [ ] Day 10: async and virtual loss merged; CP2 chaired; `DECISION.md` written
- [ ] Day 14: 50-problem real run done; trees inspected; CP3
- [ ] Day 21: TZ sweep on MATH-500 and AIME done; token counts given to Person 3; CP4
- [ ] Day 28: ablations, Method section, fresh-clone check, `v0.1.0` tag; CP5


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
