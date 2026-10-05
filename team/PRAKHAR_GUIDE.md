# Prakhar's guide: your work, phase by phase, plus a glossary

> A companion to `Prakhar.md`. That file says **what** to deliver; this one shows **how** to do it, with code sketches and a worked example. The glossary is at the end. Terms marked ★ are core to your role.

---

## Part 1: Your work in 9 phases

| Phase | Days | What | Ends with |
|---|---|---|---|
| **P0** | Day 1 (morning) | Accounts, repo and local environment | Empty repo cloned by all 4 |
| **P1** | Days 1–2 | Scaffold and shared contract | **CP0**: scaffold merged |
| **P2** | Days 3–5 | Search building blocks (Node, PUCT, dedupe, extract) | Unit tests green |
| **P3** | Days 5–6 | Sequential MCTS and the toy test | Search beats greedy on the toy task |
| **P4** | Day 7 | First real end-to-end run | **CP1**: real smoke test passes |
| **P5** | Days 8–10 | Async MCTS with virtual loss; chair the pilot meeting | **CP2**: GO / PARTIAL / NO-GO |
| **P6** | Days 11–14 | First real 50-problem run; inspect trees | **CP3** |
| **P7** | Days 15–21 | Main ThoughtZero experiments | **CP4** |
| **P8** | Days 22–28 | Ablations, write-up, release | **CP5**: `v0.1.0` |
| Ongoing | Every day | Integrator duties (reviews, keeping `main` green, notes) | — |

---

### P0: Accounts, repo, environment (day 1, ~2 h)

**Goal:** everyone can clone an empty, protected repo.

1. Create a **private** GitHub repo `thoughtzero` and invite Jagriti, Akul and Harjas.
2. Settings → Branches → add a rule for `main`: require a PR, 1 approval and status checks; block force-pushes.
3. Locally:
   ```bash
   cd P:/LateVLA
   git init -b main
   git mv thoughtzero-spec.md SPEC.md   # or plain mv before the first commit
   git remote add origin <url>
   ```
4. Create the Python env: `python -m venv .venv`, then `.venv\Scripts\activate` (Windows).
5. Post in the group chat: the repo link, "read your `<name>.md`", and "Jagriti, join the Jev waitlist today".

---

### P1: Scaffold and shared contract (days 1–2), the critical path

**Goal:** a single PR that gives everyone the folders, the interfaces, the config, the mocks and CI. Nobody can merge until it lands, so **keep it to stubs and contracts, with no real logic.**

**Order of work** (each step is a small commit on branch `p1/scaffold`):

1. **Folder layout** from spec §5.1. Every module exists with a docstring and stubs:
   ```python
   # src/thoughtzero/judge/jev.py
   """JevJudge. Owner: Person 2 (Jagriti). See SPEC.md §5.3."""


   class JevJudge:
       async def prior_and_value(self, problem, steps, candidates):
           raise NotImplementedError("Person 2")
   ```
2. **`pyproject.toml`**, `.gitignore`, `.gitattributes` (`* text=auto eol=lf`) and `.env.example`. The details are in `Prakhar.md` §A3.2.
3. **`types.py`**: copy the contract from Part B4 of `Prakhar.md` verbatim.
4. **`accounting.py`**: take Harjas's version, or write the stub with him.
5. **`config.py`**: pydantic models, YAML loading, `${ENV}` interpolation, `--set` overrides, `config_hash`. Sketch:
   ```python
   class SearchCfg(BaseModel):
       n_simulations: int = 32
       k: int = 4
       c_puct: float = 1.5
       fpu_value: float = 0.0
       parallel_sims: int = 8
       virtual_loss: int = 1
       max_depth: int = 20
       extract_mode: Literal["most_visited", "value_vote"] = "most_visited"
       root_dirichlet_alpha: float | None = None


   def load_config(path: str, overrides: list[str] = []) -> Config:
       load_dotenv()
       raw = yaml.safe_load(Path(path).read_text())
       raw = _interpolate_env(raw)  # "${GEMMA_BASE_URL}" -> value
       for ov in overrides:  # "search.k=6"
           key, val = ov.split("=", 1)
           _set_nested(raw, key.split("."), yaml.safe_load(val))
       return Config.model_validate(raw)
   ```
6. **`mocks.py`**: the "reach 13 by adding 1, 2 or 5" toy task. See P3 for why it matters.
7. **CI**: `.github/workflows/ci.yml` running ruff, mypy and pytest.
8. **`CLAUDE.md`**, the PR template, and the `team/` files.
9. Open the PR and tag all 3. In the description, list the open decisions B11.1–B11.3 and ask for a yes/no on each.

**Done when:** CI is green, 3 approvals, merged. Then announce in the group: "CP0 done, branch from `main`."

---

### P2: Search building blocks (days 3–5)

Four small PRs, each with its tests. These are **pure functions** (no network), so they're quick to test.

#### 2a. `Node`
```python
@dataclass(eq=False)
class Node:
    steps: list[str]
    prior: float = 1.0
    N: int = 0
    W: float = 0.0
    value: float | None = None
    terminal: bool = False
    final_answer: str | None = None
    children: list["Node"] = field(default_factory=list)
    expanding: asyncio.Future | None = None
    virtual_loss: int = 0

    @property
    def n_eff(self) -> int:  # visits incl. in-flight simulations
        return self.N + self.virtual_loss

    def q_eff(self, fpu: float) -> float:
        if self.n_eff == 0:
            return fpu
        return self.W / self.n_eff  # virtual visits count as value 0
```

#### 2b. PUCT selection
```python
def puct_score(parent: Node, child: Node, c_puct: float, fpu: float) -> float:
    u = c_puct * child.prior * math.sqrt(max(1, parent.n_eff)) / (1 + child.n_eff)
    return child.q_eff(fpu) + u


def select_child(parent: Node, c_puct: float, fpu: float) -> Node:
    # tie-break: higher prior, then lower index (deterministic)
    return max(
        enumerate(parent.children),
        key=lambda ic: (puct_score(parent, ic[1], c_puct, fpu), ic[1].prior, -ic[0]),
    )[1]
```

**Worked example** (c_puct = 1.5, fpu = 0). The root has been expanded once: `N(root)=1`. It has children A, B, C with priors 0.5, 0.3, 0.2, all unvisited.

| Child | Q | Exploration term U = 1.5 · P · √1 / (1+0) | Score |
|---|---|---|---|
| A | 0 | 0.75 | **0.75** ← selected |
| B | 0 | 0.45 | 0.45 |
| C | 0 | 0.30 | 0.30 |

A gets expanded, and the judge says V(A) = 0.1 (a bad step). After backup, A has N=1, W=0.1 and the root has N=2. The next simulation computes:

| Child | Q | U = 1.5 · P · √2 / (1+N) | Score |
|---|---|---|---|
| A | 0.10 | 1.5·0.5·1.414/2 = 0.53 | 0.63 |
| B | 0 | 1.5·0.3·1.414/1 = 0.636 | **0.636** ← selected |
| C | 0 | 0.42 | 0.42 |

**The judge's low value pulled the search away from A**, even though A had the best prior. If V(A) had been 0.9, A would score 1.43 and the search would go deeper into A. That's the whole method in one example: **the prior decides where to look first; the value decides where to keep looking.**

#### 2c. Dedupe
```python
STEP_PREFIX = re.compile(r"^\s*step\s*\d+\s*:\s*", re.I)


def normalize(t: str) -> str:
    t = STEP_PREFIX.sub("", t.lower())
    return re.sub(r"\s+", " ", t).strip(" .")


def dedupe(cands, jaccard=0.9):
    uniq, mapping = [], []
    for c in cands:
        n = normalize(c.text)
        if not n:
            mapping.append(-1)
            continue
        for j, u in enumerate(uniq):
            if n == normalize(u.text) or (jaccard and _jac(n, normalize(u.text)) >= jaccard):
                mapping.append(j)
                break
        else:
            mapping.append(len(uniq))
            uniq.append(c)
    return uniq, mapping
```

#### 2d. Answer extraction
- `most_visited`: walk `max(children, key=N)` until terminal. If the walk stops at a non-terminal node, call `generator.complete(...)` greedily.
- `value_vote`: collect terminal leaves, group them by `normalize_answer`, and score each group as `Σ N × value`.

**Tests:** `test_puct.py` (the worked example above makes a perfect test case), `test_dedupe.py`, `test_extract.py`.

---

### P3: Sequential MCTS and the toy test (days 5–6)

**The loop.** One simulation is select → expand or evaluate → backup:
```python
async def simulate(root, problem, gen, judge, cfg):
    path, node = [root], root
    while node.children and not node.terminal:  # 1. SELECT
        node = select_child(node, cfg.c_puct, cfg.fpu_value)
        path.append(node)

    if node.terminal:  # 2a. EVALUATE terminal
        if node.value is None:
            node.value = await judge.final_correct(problem, node.steps)
        v = node.value
    else:  # 2b. EXPAND leaf
        outs = await gen.propose(problem, node.steps, cfg.k)
        cands, _ = dedupe(outs)
        priors, v = await judge.prior_and_value(problem, node.steps, [c.text for c in cands])
        node.value = v
        node.children = [make_child(node, c.text, p, cfg) for c, p in zip(cands, priors)]

    for n in path:  # 3. BACKUP (root included)
        n.N += 1
        n.W += v
```

**The toy test proves the search works:**
```python
async def test_search_beats_greedy():
    gen, judge = MockGenerator(seed=0), MockJudge()
    assert await greedy_by_prior(TOY, gen, judge) != "13"  # +5,+5,+5 = 15 → wrong
    res = await search(TOY, gen, judge, SearchCfg(n_simulations=64, k=3))
    assert res.answer == "13"
```

**Invariant to assert after every search:** `root.N == n_simulations` and `sum(c.N for c in root.children) == root.N - 1`. The first simulation expands the root and backs up only to the root itself.

---

### P4: First real end-to-end run (day 7), CP1

Needed from teammates: Akul's Gemma server URL plus `prompts`/`grading`, and Jagriti's `JevJudge`.

```bash
python scripts/smoke_test.py --mock                 # must always pass
python scripts/smoke_test.py --n-sims 4 --k 3       # real: Gemma + Jev
```

The script prints, per problem: the answer, whether it's correct, the number of Jev calls, the USD spent and the time taken. It **asserts ≤ 20 Jev calls and ≤ $0.01**. Pass condition: ≥ 1 of 3 easy problems correct.

Then write `results/PHASE_0_NOTES.md`: what was built, the VERIFY outcomes, open issues and cost.

---

### P5: Async MCTS with virtual loss (days 8–10), plus chairing CP2

**Why async:** each expansion waits on the network (Gemma ~1 s, Jev ~0.1 s). Running 8 simulations at once hides that wait.

**Why virtual loss:** without it, all 8 concurrent simulations pick the *same* best child and do duplicate work. Virtual loss temporarily makes a path look visited and worth 0, so the next simulation picks a different branch.

```python
async def simulate_async(root, ctx):
    path, node = [root], root
    root.virtual_loss += ctx.vl
    try:
        while node.children and not node.terminal:
            node = select_child(node, ctx.c_puct, ctx.fpu)   # uses N + virtual_loss
            node.virtual_loss += ctx.vl
            path.append(node)

        if node.terminal:
            if node.value is None:
                node.value = await ctx.judge.final_correct(ctx.problem, node.steps)
        elif node.expanding is not None:                     # someone else is expanding it
            await node.expanding
        else:
            node.expanding = asyncio.get_running_loop().create_future()
            try:
                await expand(node, ctx)                      # propose + judge + children
                node.expanding.set_result(None)
            except Exception as e:
                node.expanding.set_exception(e)
                node.expanding = None                        # allow a retry later
                raise
        backup(path, node.value)
    finally:
        for n in path:                                       # ALWAYS undo virtual loss
            n.virtual_loss -= ctx.vl

async def search(...):
    sem = asyncio.Semaphore(cfg.parallel_sims)
    async def one():
        async with sem:
            await simulate_async(root, ctx)
    async with asyncio.TaskGroup() as tg:
        for _ in range(cfg.n_simulations):
            tg.create_task(one())
```

Two notes on this sketch:
- `TaskGroup` cancels everything if one task raises. Wrap `one()` in a try/except that logs and counts failures, so one bad Jev call doesn't kill the whole search.
- Selection and backup contain **no `await`**, so they run atomically on the event loop. That's why no locks are needed.

**Tests:** see `Prakhar.md` §A5.2. They check that concurrent simulations spread out, that no virtual loss leaks, that failures are survived, and that a leaf two simulations race to gets exactly one expansion.

**CP2 meeting (day 10):** Harjas presents the pilot report.

- **Agenda:**
  1. Jev AUROC and its confidence interval.
  2. Calibration by depth.
  3. Prompt sensitivity.
  4. Cost.
  5. Decision.
  6. Open decisions B11.1, B11.4 and B11.5.
- **Decision rule:**
  - AUROC ≥ 0.75: **GO**.
  - 0.65 to 0.75: **PARTIAL** (hybrid judge).
  - Below 0.65: **NO-GO** (PRM value, Jev prior only).
- **Afterwards:** record the decision in `results/pilot/DECISION.md`.

---

### P6: First real 50-problem run (days 11–14), CP3

```bash
python scripts/run_experiment.py --config configs/exp_main.yaml \
  --set eval.methods=[tz] --set eval.n_problems=50 --set search.n_simulations=16 \
  --run-id tz16_dev50
python scripts/view_tree.py results/tz16_dev50/per_problem.jsonl --problem <id>
```

Read about 10 trees and write down what you see. Here's what each symptom means:

| Symptom | Likely cause | Who to tell |
|---|---|---|
| Steps are whole solutions | Generator prompt or stop sequence | Akul |
| Many identical siblings | Temperature too low, or dedupe too weak | Akul / you |
| Priors all ≈ 1/k | Jev not discriminating | Jagriti |
| Values all ≈ 0.95 | Overconfident judge | Jagriti / Harjas (calibration) |
| Most-visited path never terminal | Too few simulations, or the depth cap is too low | You |

---

### P7: Main ThoughtZero experiments (days 15–21), CP4

1. Estimate the cost per run: `problems × expansions × avg_state_tokens × $0.042/1M`, plus terminal calls.
2. Run `n_simulations ∈ {8, 16, 32, 64}` on MATH-500, then AIME. Shard across Kaggle accounts with `--shard i/n`.
3. **As soon as the n=8 and n=16 runs finish**, send Akul the mean completion tokens per problem, so he can match self-consistency's N.
4. Track the search diagnostics: expansions, max depth, the fraction of most-visited paths that end terminal, and `most_visited` vs `value_vote` agreement.

---

### P8: Ablations, write-up, release (days 22–28), CP5

- Ablations on the shared 200-problem ablation subset, at one `n_simulations`: `k ∈ {2, 4, 6}`, `c_puct ∈ {0.5, 1.5, 3.0}`, and `extract_mode`.
- Write the **Method** section of `REPORT.md` (the worked PUCT example above is a good figure).
- Do a fresh clone in a new folder, then run `pip install -e ".[dev]"`, `pytest`, `smoke_test.py --mock` and the reproduction commands. Tag `v0.1.0`.

---

### Ongoing: integrator routine

| When | Do |
|---|---|
| Daily | Review PRs within 24 h; post your own stand-up; unblock people |
| Every merge | If `main` breaks, revert first and debug after |
| Friday | Run `smoke_test.py --mock` on `main`; collect cost and GPU-hours; write `PHASE_<n>_NOTES.md` |
| Any interface change | Update `types.py`, the mocks and the tests in one PR, and tag all 3 |

---

## Part 2: Glossary

### A. Tree search and reinforcement learning ★ (your core)

| Term | Meaning |
|---|---|
| ★ **MCTS (Monte Carlo Tree Search)** | Builds a search tree gradually. It repeatedly picks a promising path, extends it by one level, scores it, and passes the score back up. Effort concentrates on good branches. |
| ★ **AlphaZero** | DeepMind's game-playing system: MCTS guided by a trained *policy* network and *value* network, with no rollouts. ThoughtZero copies the search and replaces both networks with Jev. |
| ★ **State (s)** | Where you are: the problem plus the reasoning steps written so far. |
| ★ **Action (a)** | One move: here, one next reasoning step. |
| ★ **Node / edge** | A node is a state. An edge, or child, is the action that leads to the next state. |
| ★ **Root / leaf / terminal** | The root is the empty solution. A leaf is a node not yet expanded. A terminal node is a finished solution (contains `\boxed{}`) or one at the depth cap. |
| ★ **Policy / prior P(s,a)** | How promising each candidate action looks *before* exploring it. From Jev's `choice`. |
| ★ **Value V(s)** | An estimate of how good a state is (here, the probability that the steps so far are correct). From Jev's `noul`. |
| ★ **N, W, Q** | N is the visit count, W the sum of backed-up values, and Q = W/N the average value. |
| ★ **Simulation** | One pass of select → expand/evaluate → backup. `n_simulations` is the search budget. |
| ★ **Selection** | Walking down from the root, picking the child with the best PUCT score at each level. |
| ★ **Expansion** | Creating the children of a leaf: Gemma proposes K steps, and Jev gives the priors and the leaf's value. |
| ★ **Backup (backpropagation)** | Adding the leaf's value to W and 1 to N for every node on the path. *Not* the same as neural-network backprop. |
| ★ **Rollout / playout** | Classic MCTS plays to the end randomly to score a leaf. AlphaZero, and we, use a value estimate instead. |
| ★ **PUCT** | The selection formula `Q + c_puct · P · √N_parent / (1 + N_child)`. It balances what's known to be good (Q) against what looks promising and is under-explored (the second term). |
| **UCB / UCT** | The older, prior-free ancestors of PUCT. |
| ★ **Exploration vs exploitation** | Trying new branches vs deepening known-good ones. `c_puct` sets the balance: higher means more exploration. |
| ★ **c_puct** | The exploration constant (default 1.5). |
| ★ **Branching factor (k)** | How many candidate steps are proposed per expansion (default 4). |
| ★ **FPU (first-play urgency)** | The Q value assumed for an unvisited child (default 0). Low FPU means unvisited children must earn attention through their prior. |
| ★ **Virtual loss** | Temporarily treating in-flight paths as visited with value 0, so parallel simulations spread out. |
| **Dirichlet noise** | Random noise added to the root priors to force exploration (optional, off by default). |
| ★ **Most-visited extraction** | The final answer comes from following the most-visited child at each level. Visits are a robust signal. |
| ★ **Value vote** | The final answer is the one whose terminal leaves have the most Σ N·value. |
| **Depth cap (max_depth)** | The maximum number of steps; a node at the cap becomes terminal. |
| **Single-agent vs two-player** | Chess MCTS flips the value's sign each turn. We have no opponent, so **never flip signs**. |
| **Expert iteration** | Train the generator on the search's own good outputs, then search again (Phase 6, stretch goal). |

### B. LLMs and serving

| Term | Meaning |
|---|---|
| ★ **Generator** | The model that writes candidate steps (Gemma 4 E4B). |
| **Gemma 4 E4B** | A Google open model; "E4B" means about 4B *effective* parameters. Apache-2.0 licensed. |
| ★ **Token** | The unit of text a model reads and writes (about ¾ of a word). |
| ★ **Prompt tokens / completion tokens** | Tokens in and tokens out. **Completion tokens are our primary compute axis.** |
| ★ **Temperature** | Sampling randomness. 0 is greedy and deterministic; 0.9 is diverse. We want diverse candidates. |
| **Top-p (nucleus sampling)** | Samples only from the smallest set of tokens covering probability p (0.95). |
| ★ **n (samples per request)** | Asks for k completions of the same prompt in one call. Cheap, thanks to prefix caching. |
| ★ **Stop sequence** | A string that ends generation (`"\n\nStep"`). This is how we cut output into single steps. |
| ★ **Chat template** | The model-specific special tokens that wrap user and assistant turns. |
| ★ **Completions vs chat endpoint** | The completions endpoint takes raw text, so we can *continue* a half-written answer. Chat can't, which is why we use completions. |
| **Thinking mode** | Hidden chain-of-thought some models produce. We disable it because it blurs step boundaries. |
| ★ **KV cache / prefix caching** | Reusing computation for a shared prompt prefix. Siblings share their entire history, so this makes search affordable. |
| **RadixAttention** | SGLang's automatic prefix cache, built on a radix tree. |
| **vLLM / SGLang / Ollama** | Model-serving engines. SGLang or vLLM for experiments; Ollama for local development. |
| ★ **OpenAI-compatible API** | A server that mimics OpenAI's HTTP API, so one client works with any engine. |
| **Logprobs / top_logprobs** | The model's log-probabilities for the next tokens. Used by the self-judge baseline. |
| **Quantization (fp16, bf16, 8-bit, 4-bit, AWQ)** | Storing weights at lower precision to save memory. A T4 GPU lacks bf16. |
| **Context window** | The maximum tokens a model can read (Gemma E4B 128K; Jev 32K). |
| **MoE (mixture of experts)** | Only some sub-networks run per token (Gemma 26B A4B has about 4B active). |
| **Speculative decoding / MTP draft** | A small draft model guesses tokens that the big model verifies, which is faster. |
| **LoRA** | Cheap fine-tuning via small low-rank adapter matrices (Phase 6). |

### C. Reasoning methods and related work

| Term | Meaning |
|---|---|
| ★ **CoT (chain of thought)** | The model writes step-by-step reasoning before answering (baseline B1). |
| ★ **Greedy decoding** | Always picking the most likely token (temperature 0). |
| ★ **Self-consistency** | Sample N solutions and majority-vote the answer. **Our main rival (B2).** |
| ★ **Best-of-N** | Sample N solutions and let a judge pick one (B3). |
| ★ **Test-time compute** | Spending more inference compute (sampling or search) instead of using a bigger model. The theme of this project. |
| ★ **Matched compute** | Comparing methods at the same token budget. The only fair comparison. |
| ★ **PRM (process reward model)** | A trained model that scores each *step*. Compare **ORM**, which scores only the final answer. |
| ★ **LLM-as-judge / verifier** | Using a model to grade reasoning. Jev is our judge. |
| **Self-judge** | The generator grading its own work (baseline B4). |
| **Math-Shepherd labels** | A step's quality is the fraction of sampled completions from that step that reach the right answer. Used in the pilot. |
| **Tree of Thoughts, RAP, TS-LLM, rStar-Math** | Prior work on searching over LLM reasoning (spec §14). Ours differs by being training-free. |
| **System 1 / System 2** | Fast intuitive judgement vs slow deliberate reasoning. Jev is pitched as a fast "System 1" decider. |
| **Contamination** | Benchmark problems that leaked into training data. We check with post-cutoff AIME problems. |

### D. Jev-specific

| Term | Meaning |
|---|---|
| ★ **Jev** | TypeSafe AI's model that answers typed questions with calibrated probabilities. |
| ★ **`noul`** | A yes/no question type; returns P(yes). It gives our value. |
| ★ **`choice`** | Pick-one-of-N; ideally returns a probability per option. It gives our priors. |
| **`score`** | An ordinal rubric rating (unused for now). |
| **`state` / `questions`** | Jev's input: the context text, plus a map of questions answered in parallel in one call. |
| ★ **`prior_mode`** | `choice` (one question) or `per_candidate_noul` (K yes/no questions), the fallback if `choice` returns no distribution. |
| **Version pinning** | Using `jev-1.13.0`, not `jev-latest`, so results are reproducible. |
| ★ **Calibrated probability** | When it says 70%, it's right about 70% of the time. Essential for a value function. |

### E. Evaluation and statistics

| Term | Meaning |
|---|---|
| ★ **Pilot / go-no-go** | A cheap early experiment that decides whether the main idea is worth building. |
| ★ **AUROC** | The probability that a random correct prefix scores higher than a random incorrect one. 0.5 is chance, 1.0 is perfect. Our go threshold is 0.75. |
| ★ **Calibration** | Whether predicted probabilities match actual frequencies. |
| **ECE (expected calibration error)** | The average gap between predicted and actual frequency across 10 probability bins. Lower is better. |
| **Brier score** | The mean squared error of probabilities against outcomes. Lower is better. |
| **Reliability diagram** | A plot of predicted vs actual frequency. A perfect judge sits on the diagonal. |
| **Soft / hard label** | Soft is the fraction of correct completions (e.g. 0.375); hard is "any correct?" (yes/no). |
| ★ **Bootstrap CI** | A confidence interval made by resampling problems with replacement 1,000 times. |
| ★ **Paired bootstrap** | A CI on the *difference* between two methods on the same problems. The right significance test. |
| ★ **Ablation** | Removing or varying one component to measure its contribution. |
| ★ **Train/test leakage** | Tuning on the problems you report results on, which inflates scores. Tune on the train split. |
| **Stratified sampling** | Sampling evenly across difficulty levels. |
| **MATH-500 / AIME** | Benchmarks: 500 competition math problems (levels 1–5), and hard olympiad-qualifier problems. |
| **`\boxed{}` / math-verify** | Where the final answer goes, and the library that checks whether two math answers are equivalent. |

### F. Engineering (you'll touch all of these)

| Term | Meaning |
|---|---|
| ★ **asyncio / coroutine / event loop** | Python's single-threaded concurrency: `async def` functions pause at `await` while waiting on I/O. |
| ★ **Future** | A placeholder for a result that arrives later. Used so other simulations can wait on an expansion in flight. |
| ★ **TaskGroup** | Runs several async tasks together and waits for all of them (Python 3.11+). |
| ★ **Semaphore** | Limits how many tasks run at once (e.g. 8 simulations, 16 Jev calls). |
| ★ **ContextVar** | A per-task variable; how each problem gets its own cost ledger. |
| ★ **Protocol** | Python structural typing: anything with the right methods counts as a `Judge`. That's how mocks swap in. |
| ★ **dataclass / pydantic** | Lightweight data containers / validated config models. |
| ★ **Mock / fixture** | Fake objects (MockJudge), and recorded real responses, used so tests run offline and for free. |
| **pytest-asyncio / respx** | Testing async code / faking HTTP servers in tests. |
| **Exponential backoff / HTTP 429** | Retrying with growing waits / the "too many requests" rate-limit error. |
| **Disk cache / sha256 key** | Stores every paid response, keyed by a hash of the request, so reruns cost $0. |
| **BudgetGuard** | A hard stop when spending hits the cap. |
| **JSONL** | One JSON object per line; append-friendly, so runs can resume. |
| **Resumable run / sharding** | Skip already-finished problems / split a run across machines. |
| ★ **CI (continuous integration)** | GitHub Actions running lint and tests on every PR. |
| ★ **Branch protection / PR / review / rebase** | The rules that keep `main` working, plus the standard GitHub workflow. |
| **ruff / mypy** | The linter and formatter / the static type checker. |
| **Editable install (`pip install -e .`)** | Code changes take effect without reinstalling. |
| **`src/` layout** | The package lives in `src/thoughtzero/`, which avoids accidental imports from the repo root. |
| **`.env`** | A local, git-ignored file holding API keys. |

---

### What to learn first (in this order)

1. **MCTS and PUCT.** Work through the example in P2 by hand, then read a short AlphaZero MCTS explainer.
2. **asyncio:** `async`/`await`, `Future`, `TaskGroup`, `Semaphore`. You need these for P5.
3. **pydantic v2 and pytest-asyncio:** you need these for P1–P3.
4. **Self-consistency, PRMs and AUROC:** enough to judge results at CP2 and CP4.
