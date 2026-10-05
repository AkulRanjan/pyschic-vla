# ThoughtZero — Training-free AlphaZero-style search over reasoning steps

> **Amended 2026-10-05:** the generator is Gemma 4 **26B-A4B** via OpenRouter's chat API (not self-hosted E4B on the completions endpoint), and baseline C is Gemma 4 31B on OpenRouter. Where this spec says E4B, raw completions, `n=k` or prefix caching, read `results/DECISIONS.md`. Other amendments are logged there too.

> **Build spec for Claude Code.** Place this file at the repo root as `SPEC.md`. Work phase by phase (§10). Do not start a phase until the previous phase's *Definition of done* is met. Items marked **VERIFY** are facts that must be checked against current docs before code depends on them.

---

## 1. One-paragraph summary

ThoughtZero runs Monte Carlo Tree Search (PUCT, as in AlphaZero) over the *reasoning steps* of a math solution. A small local model, **Gemma 4 E4B**, acts as the generator: at each node it proposes K candidate next steps. **Jev** (TypeSafe AI's typed-decision model) replaces both of AlphaZero's trained networks, in **one API call per expansion**:

- **Policy prior.** A Jev `choice` over the K candidates gives P(s, a) for each child.
- **Value.** A Jev `noul` ("is every step so far correct?") gives V(s) in [0, 1].

No model is trained. The research claim is:

> An off-the-shelf calibrated System-1 decision model can serve as both the policy prior and the value function for reasoning tree search. This lets a ~4B model approach the accuracy of a much larger model at matched or lower compute.

---

## 2. Goals and non-goals

### Goals
1. **Pilot (go/no-go).** Measure whether Jev can judge step-level correctness of Gemma reasoning traces (AUROC, ECE, calibration by depth).
2. A clean, async, well-tested MCTS implementation over reasoning steps.
3. Rigorous evaluation at **matched compute** against strong baselines, especially self-consistency.
4. Reproducible results: pinned model versions, cached judge calls, seeded sampling, and JSONL logs of every tree.
5. An open-source, pip-installable package with a clear README.

### Non-goals (for now)
- Training any model. Expert iteration is a stretch goal (§10, Phase 6).
- Domains other than math with verifiable final answers.
- A UI. CLI and scripts only.
- Squeezing out maximum throughput beyond what §8 asks for.

---

## 3. Key facts and constraints

### Jev (TypeSafe AI)

| Fact | Value | Status |
|---|---|---|
| Output | Typed decisions with calibrated probabilities: `noul` (yes/no probability), `choice` (distribution over options), `score` (ordinal rubric) | Known |
| Input | Text or JSON `state`, plus a map of typed `questions`, answered in parallel in one call | Known |
| Context window | 32,000 tokens | Known |
| Max options in a `choice` | 255 | Known |
| Latency | ~50–100 ms per call | Known (community-reported) |
| Price | $0.042 per 1M input tokens; output free; no free tier | Known |
| Direct endpoint | `POST https://api.typesafe.ai/v1/systemone` (waitlist) | Known |
| Alternative access | OpenRouter (`typesafe/jev-1.13`, beta), Cloudflare Workers AI (`typesafe/jev`), Vercel AI Gateway (`typesafe-ai/jev`) | Known |
| Version pinning | Pin `jev-1.13.0` for reproducibility; `jev-latest` floats | Known |
| Python SDK | `typesafe-sdk-python` exists (sync + async) | **VERIFY** the exact API |
| Exact field names for `choice` options and the response shape | e.g. `choices` vs `options`, where probabilities live | **VERIFY** |
| Whether `choice` returns the full distribution or only the top pick | Required for the prior | **VERIFY** (critical) |
| Rate limits | Unknown | **VERIFY** |
| Terms of service | Check before publishing anything that stress-tests or benchmarks the API | **VERIFY** |

**Hard constraint:** Jev is text-only and remote. It must never sit in a per-token loop. Here it is called once per tree expansion, plus once per new terminal node.

### Gemma 4

| Fact | Value | Status |
|---|---|---|
| Sizes | E2B, E4B, 12B, 26B A4B (MoE, ~4B active), 31B dense | Known |
| License | Apache 2.0 | Known |
| Context window | 128K (E2B/E4B); 256K (26B/31B) | Known |
| Speculative decoding | All sizes ship a multi-token-prediction draft model | Known |
| Hugging Face IDs | e.g. `google/gemma-4-E4B-it`, `google/gemma-4-31B-it` | **VERIFY** |
| Ollama tags | `gemma4:e4b`, `gemma4:31b` | Known |
| Memory on a 16 GB T4 | E4B likely needs 8-bit or 4-bit weights; T4 has no bf16/fp8 | **VERIFY** |

### Hardware assumption
- Development: a laptop, with a mocked or remote generator.
- Experiments: Kaggle (T4 ×2) or a rented 24 GB GPU.
- The code must run with whatever OpenAI-compatible server is available (§5.2).

---

## 4. Problem formulation

| MCTS concept | ThoughtZero meaning |
|---|---|
| State `s` | The problem text plus the ordered list of reasoning steps so far |
| Action `a` | One next reasoning step: a sentence to a short paragraph, ending at a step delimiter |
| Transition | Deterministic: `s' = s + [a]` |
| Terminal | A step containing `\boxed{...}` or `Final answer:`, **or** depth ≥ `max_depth` |
| Prior `P(s, a)` | Jev `choice` distribution over the K candidate children of `s` |
| Value `V(s)` | Non-terminal: Jev `noul` P("all steps so far are correct"). Terminal: Jev `noul` P("the final answer is correct") |
| Rollouts | **None** (AlphaZero-style). The value estimate replaces rollouts |
| Ground truth | Used **only for evaluation**, never inside the search |

### Selection rule (PUCT)

```
a* = argmax_a  Q(s,a) + c_puct · P(s,a) · sqrt(N(s)) / (1 + N(s,a))
```

- `Q(s,a) = W(s,a) / N(s,a)`. Use `0.0` when `N(s,a) == 0`; make this configurable as `fpu_value` (first-play urgency).
- `c_puct` defaults to 1.5.

### Backup
Add the leaf value `v` to `W` and increment `N` for every node on the path from the root to the leaf.

### Answer extraction (implement both, select via config)
1. **`most_visited`.** From the root, repeatedly move to the child with the highest `N` until reaching a terminal node.
   - If the path ends at a non-terminal node, complete it greedily with Gemma at temperature 0.
2. **`value_vote`.** Collect every terminal leaf, group by normalized final answer, and sum `N × value` per group. Return the group with the largest sum.

---

## 5. Architecture

```
            ┌──────────────────────────────────────────────┐
            │                 MCTS engine                  │
            │   select (PUCT) → expand → evaluate → backup │
            └──────┬──────────────────────────┬────────────┘
                   │ propose(prefix, k)       │ prior_and_value(...) / final_correct(...)
          ┌────────▼────────┐        ┌────────▼─────────────────────┐
          │   Generator     │        │   Judge (pluggable)          │
          │ Gemma 4 E4B via │        │ JevJudge (main)              │
          │ OpenAI-compat.  │        │ GemmaSelfJudge (baseline B4) │
          │ server (SGLang/ │        │ PRMJudge (baseline B5)       │
          │ vLLM/Ollama)    │        │ UniformJudge (ablations)     │
          └─────────────────┘        └──────────┬───────────────────┘
                                                │
                                   ┌────────────▼───────────┐
                                   │ DiskCache + BudgetGuard│
                                   │ + Accounting           │
                                   └────────────────────────┘
```

### 5.1 Repository layout

```
thoughtzero/
├── SPEC.md                      # this file
├── CLAUDE.md                    # short pointer to SPEC.md and working rules (§13)
├── README.md
├── pyproject.toml               # Python 3.11+, deps in §12
├── .env.example                 # TYPESAFE_API_KEY=, OPENROUTER_API_KEY=, GEMMA_BASE_URL=, GEMMA_MODEL=
├── configs/
│   ├── default.yaml
│   ├── pilot.yaml
│   ├── exp_main.yaml
│   └── ablations.yaml
├── src/thoughtzero/
│   ├── config.py                # pydantic models; load YAML + env overrides
│   ├── accounting.py            # per-problem token, cost and time ledger
│   ├── data/
│   │   ├── datasets.py          # MATH-500, AIME loaders → Problem(id, question, answer, source)
│   │   └── grading.py           # answer extraction and equivalence (math-verify)
│   ├── llm/
│   │   ├── gemma.py             # Generator protocol + OpenAICompatibleGenerator
│   │   └── prompts.py           # all prompt templates (generator + judge state formatting)
│   ├── judge/
│   │   ├── base.py              # Judge protocol
│   │   ├── jev.py               # JevJudge (HTTP or SDK), retries, response parsing
│   │   ├── self_judge.py        # GemmaSelfJudge via logprobs
│   │   ├── prm.py               # PRMJudge (optional, GPU)
│   │   ├── uniform.py           # UniformJudge / ConstantValueJudge
│   │   ├── cache.py             # sha256-keyed disk cache (sqlite or diskcache)
│   │   └── budget.py            # BudgetGuard: hard USD cap, raises BudgetExceeded
│   ├── search/
│   │   ├── node.py              # Node dataclass
│   │   ├── mcts.py              # async MCTS with virtual loss
│   │   ├── extract.py           # most_visited / value_vote
│   │   └── dedupe.py            # candidate step normalization + merging
│   ├── baselines/
│   │   ├── cot.py               # B1
│   │   ├── self_consistency.py  # B2
│   │   └── best_of_n.py         # B3
│   ├── pilot/
│   │   ├── traces.py            # generate full solutions, split into steps
│   │   ├── mc_label.py          # Math-Shepherd-style Monte Carlo prefix labels
│   │   ├── score.py             # run Judge.step_sound on each prefix
│   │   └── report.py            # AUROC, ECE, Brier, by-depth calibration, go/no-go
│   └── eval/
│       ├── runner.py            # runs any method over a dataset, writes JSONL
│       ├── metrics.py           # accuracy + bootstrap CI, ECE, AUROC, Brier
│       └── plots.py             # accuracy-vs-compute curves, reliability diagrams
├── scripts/
│   ├── run_pilot.py
│   ├── run_experiment.py
│   ├── make_plots.py
│   └── smoke_test.py            # 3 problems end-to-end, real or mocked backends
├── tests/
│   ├── test_puct.py
│   ├── test_mcts_toy.py         # MCTS solves a toy problem with mock generator/judge
│   ├── test_virtual_loss.py
│   ├── test_extract.py
│   ├── test_dedupe.py
│   ├── test_jev_parsing.py      # parse recorded fixture responses
│   ├── test_cache_budget.py
│   └── test_grading.py
└── results/                     # gitignored: JSONL logs, trees, plots, reports
```

### 5.2 Generator interface

```python
class Generator(Protocol):
    async def propose(self, problem: str, steps: list[str], k: int) -> list[GenOut]: ...
    async def complete(self, problem: str, steps: list[str], temperature: float = 0.0) -> list[str]: ...
    async def sample_solutions(self, problem: str, n: int, temperature: float) -> list[str]: ...

@dataclass
class GenOut:
    text: str
    prompt_tokens: int
    completion_tokens: int
```

**`OpenAICompatibleGenerator`** talks to SGLang, vLLM or Ollama through the OpenAI-compatible **completions** endpoint, not chat. Chat is ruled out because the generator must *continue a partial assistant message*.

**Prompt construction.**
- Apply Gemma's chat template (via the HF tokenizer) to `[system?, user: problem + format instructions]`.
- Leave the assistant turn open, then append the existing steps formatted as `Step 1: ...\n\nStep 2: ...\n\n` and the prefix `Step {n+1}:`.
- **VERIFY** the Gemma 4 chat template, and whether a system role is supported.

**Sampling.**
- `n=k`, `temperature=0.9`, `top_p=0.95`.
- `stop=["\n\nStep", "\n\n\n"]`; make this configurable.
- `max_tokens=max_step_tokens` (default 256).

**Thinking mode.** Use the non-thinking (standard instruct) mode. Thinking mode blurs step boundaries. **VERIFY** how to disable it for Gemma 4.

**Prefix caching.** This is the main efficiency lever, because siblings share their whole prefix.
- Prefer SGLang (RadixAttention).
- Otherwise use vLLM with `--enable-prefix-caching`.
- Ollama is allowed for local development only.

**Generator prompt.** Instruct the model to:
- write one numbered step at a time;
- keep each step short;
- put the final answer in `\boxed{}` in the last step.

### 5.3 Judge interface

```python
class Judge(Protocol):
    async def prior_and_value(
        self, problem: str, steps: list[str], candidates: list[str]
    ) -> tuple[list[float], float]: ...          # priors (sum to 1, len == len(candidates)), V(s)

    async def final_correct(self, problem: str, steps: list[str]) -> float: ...   # terminal value
    async def step_sound(self, problem: str, steps: list[str]) -> float: ...      # used by the pilot
```

#### JevJudge — one request per expansion

**State text** (built in `prompts.py`):

```
PROBLEM:
<problem>

SOLUTION SO FAR:
Step 1: ...
Step 2: ...
```

**Questions:**

| key | type | instructions (draft — tune in Phase 1) |
|---|---|---|
| `sound` | `noul` | "Is every step in SOLUTION SO FAR mathematically correct and logically valid? Ignore whether the solution is finished." |
| `next` | `choice` | "Which candidate next step is most likely to lead to a correct final answer?" Options: `c0..c{K-1}` → candidate text |

**Request shape** (field names to **VERIFY**):

```json
{
  "model": "jev-1.13.0",
  "state": "<state text>",
  "questions": {
    "sound": {"type": "noul", "instructions": "..."},
    "next":  {"type": "choice", "instructions": "...",
              "choices": {"c0": "<candidate 0>", "c1": "<candidate 1>"}}
  }
}
```

**Parsing.**
- `V(s)` = the probability of "yes" for `sound`.
- `priors` = the probability for each of `c0..c{K-1}` from `next`. Renormalize to sum to 1. If any are missing, fill with a small epsilon before renormalizing.

**Fallback if `choice` returns only a top pick and no distribution.** Ask K parallel `noul` questions instead, one per candidate ("Does candidate i correctly continue the solution?"), and normalize the yes-probabilities into priors. Implement this as `prior_mode: choice | per_candidate_noul`.

**Terminal call (`final_correct`).** A `noul`: "Is the final answer of this solution correct?" over the full solution.

**Robustness and accounting.**
- Retry with exponential backoff on HTTP 429/5xx (max 5 attempts).
- Use an `asyncio.Semaphore` to limit concurrency (default 16).
- Every request goes through `DiskCache` and then `BudgetGuard`.
- Record input tokens: use API-reported usage if available, otherwise estimate as `len(chars) / 4`.

**Long states.** If the state exceeds ~28K tokens, truncate the middle steps and keep the problem plus the last few steps. Log a warning.

#### Other judges
- **GemmaSelfJudge (B4).** The same questions, posed to Gemma as prompts.
  - Read the next-token logprobs for `Yes`/`No` to get a value.
  - Read the logprobs for candidate letters `A..` to get priors.
  - Use one forward pass each, via `logprobs` / `top_logprobs` on the completions endpoint. **VERIFY** server support.
- **PRMJudge (B5, optional).** An open process reward model, e.g. `Qwen/Qwen2.5-Math-PRM-7B` (**VERIFY** ID and usage).
  - It supplies the value only. Priors are uniform, or come from GemmaSelfJudge.
  - Skip this baseline if GPU memory doesn't allow it, and record that in the report.
- **UniformJudge.** Uniform priors and a constant value of 0.5. Used for ablations and tests.

### 5.4 Cache, budget and accounting

**`DiskCache`**
- Key: `sha256(canonical_json(request including model version))`.
- Value: the raw response plus parsed fields.
- Reruns must cost $0.

**`BudgetGuard`**
- Tracks cumulative estimated USD across the run: `input_tokens × 0.042 / 1e6`.
- Raises `BudgetExceeded` when the run exceeds `budget.max_usd` (default **$5.00**).
- Prints a pre-run estimate for each script and requires `--yes` above $1.

**`Accounting`.** For each problem and method, record:
- Gemma prompt and completion tokens (completion tokens are the primary compute axis);
- the number of Jev calls and Jev input tokens;
- Jev USD;
- wall-clock time;
- the number of expansions, maximum depth, and number of terminal leaves.

---

## 6. MCTS details

### 6.1 Node

```python
@dataclass
class Node:
    steps: list[str]
    prior: float = 1.0
    N: int = 0
    W: float = 0.0
    value: float | None = None          # judge value of this node's state, cached
    terminal: bool = False
    final_answer: str | None = None     # parsed for terminal nodes
    children: list["Node"] = field(default_factory=list)
    expanding: asyncio.Future | None = None   # set while an expansion is in flight
    virtual_loss: int = 0

    @property
    def Q(self) -> float: ...
```

### 6.2 Algorithm (async, with virtual loss)

```
search(problem, generator, judge, cfg):
    root = Node(steps=[])
    run cfg.n_simulations simulations, with at most cfg.parallel_sims in flight (TaskGroup + Semaphore)
    return extract(root, cfg.extract_mode), tree, accounting

simulate(root):
    path = [root]; node = root
    while node.children and not node.terminal:
        node = select(node)                  # PUCT using N + virtual_loss
        node.virtual_loss += cfg.vl          # applied to every node on the path
        path.append(node)

    if node.terminal:
        if node.value is None: node.value = await judge.final_correct(...)
        v = node.value
    elif node.expanding is not None:          # another sim is expanding this leaf
        await node.expanding; v = node.value
    else:
        node.expanding = Future()
        cands = dedupe(await generator.propose(problem, node.steps, k=cfg.k))
        priors, v = await judge.prior_and_value(problem, node.steps, [c.text for c in cands])
        node.value = v
        node.children = [make_child(node, c, p) for c, p in zip(cands, priors)]
        node.expanding.set_result(None)

    backup(path, v); remove virtual loss along path
```

**Virtual loss.** While a simulation is in flight, PUCT treats each node on its path as if it had `N + vl` visits and contributed a value of 0 for those extra visits. This pushes concurrent simulations onto different branches. Default `vl = 1`.

**`make_child`**
- Mark the child `terminal` if its step contains `\boxed{` or `Final answer`, or if `depth + 1 >= max_depth`.
- For terminal children, parse `final_answer` with `grading.extract_answer`.

**Depth cap.** A node created at `max_depth` (default 20) without an answer is terminal with `final_answer = None`. Its value still comes from `final_correct`.

**Root.** The root is expanded on the first simulation. Optional Dirichlet noise on root priors is a config flag (`root_dirichlet_alpha`), off by default.

### 6.3 Candidate dedupe (`dedupe.py`)
- Normalize each candidate: lowercase, collapse whitespace, strip the `Step n:` prefix.
- Merge candidates that are exact duplicates after normalization.
- Optionally merge near-duplicates with token Jaccard similarity ≥ `0.9`.
- When merging, sum their priors after the judge call. Implement this by mapping candidates to unique indices before calling the judge.
- If dedupe leaves a single candidate, still call the judge, so the node gets a value.

### 6.4 Defaults (`configs/default.yaml`)

```yaml
generator:
  base_url: ${GEMMA_BASE_URL}
  model: ${GEMMA_MODEL}          # e.g. google/gemma-4-E4B-it
  temperature: 0.9
  top_p: 0.95
  max_step_tokens: 256
  stop: ["\n\nStep", "\n\n\n"]
judge:
  kind: jev                     # jev | self | prm | uniform
  jev_model: jev-1.13.0
  prior_mode: choice            # choice | per_candidate_noul
  max_concurrency: 16
search:
  n_simulations: 32
  k: 4
  c_puct: 1.5
  fpu_value: 0.0
  parallel_sims: 8
  virtual_loss: 1
  max_depth: 20
  extract_mode: most_visited    # most_visited | value_vote
  root_dirichlet_alpha: null
budget:
  max_usd: 5.0
seed: 0
```

---

## 7. Phase 1 pilot: can Jev judge reasoning steps? (go/no-go)

**Purpose:** kill or confirm the core assumption for under $1, before building the search.

1. **Traces.** Take 200 MATH-500 problems, stratified across difficulty levels. For each, sample 1 solution from Gemma E4B at temperature 0.7 in the step format. Split it into steps.
2. **Prefixes.** For each trace, take every prefix `steps[:i]` for `i = 1..len`. Cap at 8 prefixes per trace, sampled evenly across depth.
3. **Monte Carlo labels (Math-Shepherd-style).** From each prefix, sample `m = 8` completions with Gemma at temperature 0.7 and grade the final answers.
   - `soft_label` = the fraction of completions that are correct.
   - `hard_label` = `soft_label > 0`.
4. **Scores.** Run `judge.step_sound(problem, prefix)` for:
   - JevJudge;
   - GemmaSelfJudge;
   - PRMJudge, if available.
5. **Report** (`results/pilot/report.md` plus plots):
   - AUROC of each score against `hard_label`;
   - Brier score and ECE (10 bins) against `soft_label`;
   - a reliability diagram;
   - AUROC and ECE **bucketed by prefix depth** (1–2, 3–5, 6–9, 10+);
   - Jev's cost and latency distribution.
6. **Decision rule.** Print this in the report.
   - **GO:** Jev AUROC ≥ 0.75 → Jev is used for both prior and value.
   - **PARTIAL:** 0.65 ≤ AUROC < 0.75 → continue, but make PRMJudge or GemmaSelfJudge a first-class value option and add a "Jev prior + PRM value" hybrid.
   - **NO-GO:** AUROC < 0.65 → use the PRM for value and keep Jev only for the prior. The pilot itself is written up as a negative result.

Also log a prompt-sensitivity check: compute AUROC for 3 phrasings of the `sound` instruction on a 50-problem subset.

---

## 8. Experiments

### 8.1 Methods

| ID | Method | Notes |
|---|---|---|
| B1 | E4B chain-of-thought, greedy | Floor |
| B2 | E4B self-consistency (majority vote over N samples) | **Primary baseline.** Choose N so that completion tokens ≈ ThoughtZero's mean per problem |
| B3 | E4B best-of-N, picked by Jev `final_correct` | Value of the tree versus plain reranking |
| B4 | ThoughtZero with GemmaSelfJudge | Value of Jev specifically |
| B5 | ThoughtZero with PRMJudge for value (priors from self-judge) | Jev versus a trained PRM. Optional |
| TZ | ThoughtZero with JevJudge | Main method |
| C | Gemma 4 31B chain-of-thought, greedy (and self-consistency if budget allows) | Ceiling / headline comparison. Can use a hosted endpoint; record where it ran |

### 8.2 Datasets
- **MATH-500** (`HuggingFaceH4/MATH-500` — **VERIFY**). This is the main benchmark.
- **AIME:** the most recent years available (**VERIFY** dataset IDs).
  - Prefer problems released after Gemma 4's training cutoff, and report them separately as a contamination check.
- **Grading:** use `math-verify` for answer equivalence, with a regex fallback for `\boxed{}` extraction. Unit-test it on 30 hand-written cases.

### 8.3 Compute matching and curves
- The primary compute axis is **total Gemma completion tokens per problem**.
- Secondary axes are wall-clock time and total cost. Jev cost is small, but report it.
- For TZ, sweep `n_simulations ∈ {8, 16, 32, 64}`. For B2, sweep N to cover the same token range.
- Plot accuracy against mean completion tokens, with 95% bootstrap confidence intervals (1,000 resamples).

### 8.4 Ablations (`configs/ablations.yaml`)
- **Prior only:** Jev prior with a constant value of 0.5.
- **Value only:** uniform prior with Jev value.
- `k ∈ {2, 4, 6}`; `c_puct ∈ {0.5, 1.5, 3.0}`.
- `extract_mode`: `most_visited` versus `value_vote`.
- `prior_mode`: `choice` versus `per_candidate_noul`.
- Step granularity: the default delimiter versus forcing shorter steps (`max_step_tokens=96`).

### 8.5 Outputs
- `results/<run_id>/per_problem.jsonl`. One line per (problem, method) with:
  - answer, correctness, the accounting fields, and config hash;
  - for TZ, a compact tree dump (steps, N, Q, prior, value per node).
- `results/<run_id>/summary.csv` and the plots.
- `results/<run_id>/config.yaml`: the fully resolved config.

---

## 9. Metrics (`eval/metrics.py`)
- Accuracy, with a bootstrap confidence interval.
- AUROC (scikit-learn).
- Brier score.
- ECE with 10 equal-width bins.
- Reliability-diagram data.
- Per-depth versions of the above, for the pilot.
- Search diagnostics:
  - mean number of expansions;
  - maximum depth;
  - fraction of problems where the most-visited path reached a terminal node;
  - answer agreement between `most_visited` and `value_vote`.

---

## 10. Phased plan with definitions of done

### Phase 0 — Scaffold (day 1–2)
- `pyproject.toml`, config loading, `.env.example`, logging.
- `OpenAICompatibleGenerator`, `JevJudge` (with cache, budget, retries), dataset loaders, grading.
- Mocks: `MockGenerator` and `MockJudge`, which are deterministic and driven by a toy task.

**Done when:**
- `pytest` passes with no network.
- `scripts/smoke_test.py --mock` runs end to end.
- `scripts/smoke_test.py` against real backends solves at least 1 of 3 easy problems, makes ≤ 20 Jev calls, and spends ≤ $0.01.
- The **VERIFY** items for the Jev and Gemma APIs are resolved and documented in `docs/verified_apis.md`.

### Phase 1 — Pilot (day 3–5)
- Implement §7.

**Done when:** `results/pilot/report.md` exists, with all the metrics and the GO / PARTIAL / NO-GO decision. **Stop and report the decision to the user before Phase 2.**

### Phase 2 — MCTS core (week 2)
- Implement §6, first sequentially, then async with virtual loss.

**Done when:**
- Unit tests cover PUCT math, backup, virtual-loss bookkeeping, dedupe, and both extract modes.
- `test_mcts_toy.py` shows the search finds the correct answer on a toy problem where greedy fails, using a mock judge.
- A 50-problem MATH-500 run completes, and its tree dumps are inspectable.

### Phase 3 — Baselines (week 3, first half)
- Implement B1–B5 and C behind the same `runner.py` interface.

**Done when:** every method runs on 50 problems and writes comparable JSONL.

### Phase 4 — Main experiments (week 3)
- Run §8.3 on the full MATH-500 and on AIME.

**Done when:**
- The accuracy-vs-compute plot and summary table exist.
- Total Jev spend is under budget.

### Phase 5 — Ablations and write-up (week 4)
- Run §8.4.
- Produce `results/REPORT.md` with the method, results, ablations, a limitations section, and the reproduction commands.
- Polish the README for the open-source release.

### Phase 6 — Stretch: expert iteration
- Collect ThoughtZero's correct (by ground truth) search paths on the *train* split.
- LoRA-fine-tune Gemma E4B on them (Unsloth or PEFT).
- Re-run TZ with the fine-tuned generator. Repeat for 2–3 iterations.

**Done when:** accuracy-vs-compute improves across iterations on the held-out test split.

---

## 11. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Jev can't judge step correctness (low AUROC) | The Phase 1 pilot catches this early; fall back to PRM value (§7, decision rule) |
| `choice` returns no distribution | `prior_mode: per_candidate_noul` fallback |
| Generator latency dominates | Prefix caching (SGLang/vLLM), parallel simulations, batched `n=k` sampling |
| TZ doesn't beat self-consistency at matched compute | Report it honestly. Analyze where search helps (by difficulty level); ablate step granularity |
| Jev overconfident on deep traces | Per-depth calibration report. Optional per-depth temperature calibration fitted on pilot data (report it as a variant) |
| Contamination | Report post-cutoff AIME problems separately |
| State exceeds 32K context | Middle-step truncation with a logged warning (§5.3) |
| Cost overrun | BudgetGuard hard cap, disk cache, pre-run estimate, `--yes` gate |
| API schema drift | Pin `jev-1.13.0`; keep recorded fixture responses in tests |

---

## 12. Dependencies

- `python >= 3.11`
- Core: `httpx`, `openai`, `pydantic>=2`, `pyyaml`, `python-dotenv`, `tenacity`, `diskcache`
- Models and data: `transformers` (for the tokenizer and chat template only), `datasets`, `math-verify`
- Metrics and plots: `numpy`, `scikit-learn`, `matplotlib`, `pandas`
- Development: `pytest`, `pytest-asyncio`, `ruff`, `mypy`
- Optional: `typesafe-sdk-python` (if it proves cleaner than raw HTTP), `sglang` or `vllm` (installed on the GPU box, not as a package dependency), `torch` (for the PRM baseline), `peft` / `unsloth` (for Phase 6)

---

## 13. Working rules for Claude Code

Put these in `CLAUDE.md`:

1. Read `SPEC.md` first. Work one phase at a time and check its Definition of done before moving on.
2. **Never call paid APIs in tests.** Tests use mocks and recorded fixtures only.
3. **Ask the user before:**
   - any run with an estimated cost above $1;
   - downloading model weights larger than 5 GB;
   - changing the experimental protocol in §7–§8.
4. Never hardcode, log, or print API keys. Read them from the environment.
5. Resolve each **VERIFY** item by reading the official docs, and record the findings in `docs/verified_apis.md` with links. Don't guess field names.
6. Keep all prompt text in `llm/prompts.py`. Every prompt change bumps `PROMPT_VERSION`, which is included in cache keys and logs.
7. Every experiment script writes a resolved config and a git commit hash into its results folder.
8. Code style: type hints throughout, `ruff` clean, small pure functions in `search/`, side effects only in `llm/`, `judge/` and `eval/runner.py`.
9. After each phase, write a short `results/PHASE_<n>_NOTES.md` covering what was built, what was verified, open issues, and the cost spent.

---

## 14. Related work (for the write-up)

- **Tree of Thoughts** (Yao et al., 2023) — search over LLM "thoughts" with LLM self-evaluation.
- **RAP** (Hao et al., 2023) — MCTS with the LLM as a world model.
- **TS-LLM** (Feng et al., 2023) — AlphaZero-like tree search for LLMs, with a learned value.
- **Math-Shepherd** (Wang et al., 2023) — Monte Carlo process supervision; the source of the pilot's labeling method.
- **rStar-Math** (Microsoft, 2025) — MCTS with small models and a *trained* process reward model.
- **ThoughtZero's difference:** it is training-free. A single off-the-shelf calibrated decision model provides **both** the policy prior and the value function, in one call per expansion.
