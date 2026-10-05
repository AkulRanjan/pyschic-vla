# ThoughtZero: solo plan

Status as of **2026-10-05**. From now on the whole project has one owner (Prakhar). This file
replaces the four-person split in `team/*.md` (kept for background). Phase numbers in
parentheses refer to `SPEC.md` §10.

---

## 1. Where the project stands

All code paths run end to end **offline** (mock generator and judge): `pytest` (362 tests),
`smoke_test.py --mock`, `run_experiment.py --mock` with every method, and
`run_pilot.py --stage all --mock`, including the prompt-sensitivity stage.

| Area | Status | What's missing |
|---|---|---|
| Scaffold: config, types, accounting, mocks, CI | ✅ done | — |
| Search: PUCT, dedupe, extract, MCTS (sequential + async with virtual loss), tree dumps, `view_tree.py` | ✅ done | Tuning on real data |
| Generator: `OpenAICompatibleGenerator`, Gemma prompts, tokenizer | ✅ done, offline-tested | Never run against a real Gemma server |
| Data and grading: MATH-500, MATH train, AIME 2024–26, `math-verify` grading | ✅ done (grader checked 500/500 on MATH-500) | — |
| Baselines: CoT (B1), self-consistency (B2), best-of-N (B3), 31B ceiling (C) | ✅ done | Real runs |
| Judge: `JevJudge` (cache, budget cap, prior modes, truncation, shuffle), uniform, constant, hybrid, factory | ✅ done against the **open-jev stand-in** | **Real TypeSafe Jev transport** (blocked on access) |
| Judge: `GemmaSelfJudge` (B4), `PRMJudge` (B5) | ❌ stubs | Phase S3 |
| Eval: runner (resumable, shardable), metrics, plots, analysis | ✅ done | Real runs |
| Pilot: traces, Monte Carlo labels, judge scores, sensitivity, report with GO / PARTIAL / NO-GO | ✅ done | Real run |
| GPU serving: `notebooks/kaggle_server.ipynb`, `docs/gpu_setup.md` | ✅ written | Never launched; throughput unmeasured |
| `results/REPORT.md` | skeleton | Filled in S7 |

**In short: the software is built; no real experiment has run yet.** Two external things
block every real result: **Jev access** and **a running Gemma server**.

---

## 2. Decisions only you can make

Each of these changes the experimental protocol or the spend, so settle them before the phase
that needs them. Write each decision down in `results/DECISIONS.md` with its date.

| # | Decision | Needed by | Recommendation |
|---|---|---|---|
| D1 | **Pilot split:** MATH-500 (`test`, as the spec says) or MATH **train** (`pilot.source_split=train`)? Tuning prompts on MATH-500 leaks into the main results. | S2 | **Train.** Keep MATH-500 untouched until S6. |
| D2 | **If real Jev access doesn't come:** run the study on the open-jev stand-in (then it's a study of open-jev, not Jev), or wait? | S1 | Give the waitlist a deadline (e.g. one week). Meanwhile use the stand-in **for development only**. |
| D3 | **Jev budget.** The default cap is `budget.max_usd=5`. Upper-bound estimate for everything (runner's 2,000 tokens/call, no cache): TZ sweep on MATH-500 ~$5.0, AIME ~$0.9, best-of-N ~$2.7, ablations ~$4.3, pilot ~$0.2, so ~$13. Real cost should be well below that: states are shorter than 2,000 tokens, and the seeded generator makes the n=8/16/32 trees largely repeat the n=64 tree's expansions, which then hit the cache. | S6 | Measure tokens/call in S4, re-estimate, then raise the cap or trim the sweep. |
| D4 | **`parallel_sims`** (default 8). On the toy task, 32 simulations with 8 in parallel loses accuracy with `most_visited`; parallelism trades quality for wall-clock time. | S4 | Measure on the 50-problem dev run (1 vs 4 vs 8) and fix it before S6. |
| D5 | **Spec §6.2 deviation (already merged):** a simulation that waits on a leaf being expanded keeps descending, instead of re-backing-up that leaf. | now | Keep; record it in the Method section of the report. |
| D6 | **Option shuffling** (`judge.shuffle_options`): Jev may favour the first option. | S2 | Measure position bias in the pilot; turn it on if the bias is real. |
| D7 | **PRM baseline (B5):** `Qwen/Qwen2.5-Math-PRM-7B` is ~15 GB in bf16 (above the 5 GB "ask first" line, and too big for one T4 next to Gemma). | S3 | Optional per spec. Skip it unless a bigger GPU is available, and say so in the report. |

---

## 3. Phases

The order matters: S0 starts the slow external processes, and S3 is offline work to do
**while waiting** for them.

### S0. Unblock access (day 1, ~2 hours, then waiting)

- **Jev:** join the TypeSafe direct-API waitlist (`https://api.typesafe.ai/v1/systemone`) and
  try OpenRouter (`typesafe/jev-1.13`, beta). Read both terms of service (J11: are benchmarks
  publishable?).
- **GPU:** check the Kaggle GPU quota (D5 in `docs/verified_apis.md`); have a fallback
  (a rented 24 GB GPU) priced.
- **Hugging Face token** in `.env` (Gemma 4 isn't gated, but rate limits are friendlier).
- Put keys in `.env` only (see `.env.example`); never in code or logs.

**Done when:** requests are submitted, and you know which GPU you'll use.

### S3 first. Alternative judges (days 1–4, offline; spec Phase 3 needs them)

Do this while S0 is pending; it needs no keys.

1. **`GemmaSelfJudge` (B4)** in `judge/self_judge.py`.
   - Use `OpenAICompatibleGenerator.raw_completion(prompt, max_tokens=1, logprobs=20)`.
   - Value: the same `sound` / `final` questions as a chat prompt that ends in "Answer Yes or
     No:"; value = `p(Yes) / (p(Yes) + p(No))` from the top logprobs (summing token variants
     such as `Yes` / `▁Yes`).
   - Priors: list the candidates as `A.`, `B.`, …; prior = softmax over the letters'
     logprobs; a letter missing from the top 20 gets a small epsilon.
   - Prompts go in the judge section of `llm/prompts.py`; bump `PROMPT_VERSION`.
   - `make_judge(kind="self")` needs a generator: build one from the generator config.
   - Tests with a fake `raw_completion` (letters missing, Yes/No variants, all missing).
2. **`PRMJudge` (B5)**: only if D7 says yes. Lazy `torch` import, `[prm]` extra, value only
   (priors uniform or from the self-judge). First resolve row D4 of `docs/verified_apis.md`
   (model ID, step separator, how to read the scores).
3. `make_judge` only receives `JudgeCfg`, so give it the generator config (or a generator)
   for `kind="self"`, and update `eval/methods.py::build_judge` to pass it.
4. Remove the matching "not implemented" cases from `judge/factory.py`; the runner and pilot
   pick the judges up automatically (`tz_self`, `tz_prm`, pilot `--judges self,prm`).

**Done when:** `make_judge` builds `self` (and `prm`, or a documented skip), with unit tests;
`run_experiment.py --mock` still passes.

### S1. Real backends and the first real smoke test (spec Phase 0)

1. **Gemma server:** launch `notebooks/kaggle_server.ipynb` (vLLM, 4-bit QAT checkpoint,
   prefix caching; see `docs/gpu_setup.md`). Set `GEMMA_BASE_URL` / `GEMMA_MODEL`.
   - Measure throughput for `n=4` short steps (G10) and check `logprobs` on the endpoint (G6),
     which `GemmaSelfJudge` needs.
2. **Real Jev transport**, once you have access:
   - Resolve J1–J11 in `docs/verified_apis.md` **from the official docs and real responses**;
     no guessed field names (CLAUDE.md rule 4). J1 (does `choice` return a full
     distribution?) is critical: if not, use `prior_mode=per_candidate_noul`.
   - Record real responses as fixtures in `tests/fixtures/jev/` (the stand-in fixtures stay,
     labelled as such).
   - Implement the `direct` / `openrouter` branch of `judge/client.py` on top of the existing
     `request_with_retry` (retries and semaphore are already tested); map the real response
     into the `{question_id: {"noul": p}}` / `{"probabilities": {...}}` shape `JevJudge` parses.
3. Smoke test: `python scripts/smoke_test.py --n-sims 4 --k 3`.

**Done when** (spec Phase 0): ≥ 1 of 3 easy problems solved, ≤ 20 Jev calls, ≤ $0.01, and
the Jev and Gemma VERIFY rows are ✅. Write `results/PHASE_0_NOTES.md`.

### S2. Pilot: can Jev judge reasoning steps? (spec Phase 1, the go/no-go gate)

```bash
python scripts/run_pilot.py --stage traces                    # GPU: 200 greedy solutions
python scripts/run_pilot.py --stage label --shard 0/3         # GPU, heavy: shard over sessions
python scripts/run_pilot.py --stage merge --from <shard dirs>
python scripts/run_pilot.py --stage score --judges jev,self   # Jev ~$0.1-0.2; self needs GPU
python scripts/run_pilot.py --stage sensitivity               # Jev x 3 SOUND_VARIANTS on 50 problems
python scripts/run_pilot.py --stage report                    # -> results/pilot/report.md
```

- Settle D1 first (`--set pilot.source_split=train`).
- Labels are the GPU cost: 200 problems x up to 8 prefixes x 8 completions, so up to 12,800
  full solutions. Run it in shards across sessions; every stage resumes.
- Check D6 (position bias) in the report.

**Done when:** `results/pilot/report.md` exists with AUROC and its CI, calibration by depth,
prompt sensitivity, cost, and the decision:
**GO** (AUROC ≥ 0.75) / **PARTIAL** (0.65–0.75: hybrid judge) / **NO-GO** (< 0.65: PRM or
self value, Jev prior only). **Stop here and record the decision** in
`results/pilot/DECISION.md`: it sets the judge config for everything after.

### S4. First real 50-problem run and tuning (spec Phase 2 definition of done)

- `python scripts/run_experiment.py --subset dev --set "eval.methods=[tz]" --set search.n_simulations=16 --run-id tz16_dev`
  (judge per the S2 decision).
- Inspect ~10 trees: `python scripts/view_tree.py results/tz16_dev --problem <id> --min-visits 2`.
  Look for steps that are too long or too short, duplicate siblings that dedupe misses, searches
  that never reach a terminal, near-uniform priors, and values all ≈ 1.
- Decide D4 (`parallel_sims` 1 vs 4 vs 8) and measure real Jev tokens/call for D3.
- **Tune only on the train split**, never on MATH-500 or AIME.

**Done when:** a 50-problem run completes with inspectable trees, and D3/D4 are decided.

### S5. Baselines on 50 problems (spec Phase 3)

```bash
python scripts/run_experiment.py --subset dev --set "eval.methods=[cot,'sc:64',tz_self]" --run-id base_dev
python scripts/run_experiment.py --subset dev --set "eval.methods=['bon:64']" --reuse-samples results/base_dev --run-id bon_dev
```

C (31B ceiling) needs a 31B endpoint (`--set generator.model=... --set generator.base_url=...`):
4-bit QAT is ~23 GB, so a 32–48 GB GPU, or a hosted provider (price it first).

**Done when:** every method writes comparable JSONL on the same 50 problems, and
`make_plots.py` draws them on one accuracy-vs-compute plot.

### S6. Main experiments (spec Phase 4)

- TZ on full MATH-500 at `n_simulations` ∈ {8, 16, 32, 64}, and on AIME 2024/2025/2026.
- B2 self-consistency once at 64 samples (every N ≤ 64 comes from `make_plots.py --sample-ns`),
  B3 by reusing those samples, B1, B4, and C.
- Match self-consistency's N to TZ's mean completion tokens per problem (spec §8.3).
- Before each run: read the cost estimate it prints; anything above $1 needs `--yes` (and a
  deliberate decision). Shard long runs with `--shard i/n` and `--merge` them.

**Done when:** the accuracy-vs-compute plot and summary table exist for MATH-500 and AIME,
and total Jev spend is under the cap. Write `results/PHASE_4_NOTES.md`.

### S7. Ablations, report and release (spec Phase 5)

- Ablations on the fixed 200-problem subset at one `n_simulations`
  (`configs/ablations.yaml`): `k2`, `k6`, `cpuct_0.5`, `cpuct_3.0`, `value_vote`,
  `prior_only`, `value_only`, `per_candidate_noul`, `short_steps`:
  `python scripts/run_experiment.py --config configs/ablations.yaml --subset ablation --variant <name> --run-id abl_<name>`
- Fill in `results/REPORT.md`: method (including D5), results, ablations, limitations
  (stand-in vs real Jev if relevant, skipped baselines), and reproduction commands.
- Polish the README; fresh-clone check (`pip install -e ".[dev]"`, `pytest`,
  `smoke_test.py --mock`); tag `v0.1.0`.

**Done when:** `REPORT.md` is complete and a fresh clone reproduces the offline checks.

### S8. Stretch: expert iteration (spec Phase 6)

Collect TZ's correct search paths on the MATH **train** split, LoRA-fine-tune Gemma E4B on
them (`[ft]` extra), and re-run TZ; repeat 2–3 times. Done when accuracy-vs-compute improves
on the held-out test split.

---

## 4. Suggested schedule (one person)

| Days | Work | Notes |
|---|---|---|
| 1 | S0, then start S3 | Access requests take time; start them first |
| 2–4 | S3 (self-judge, optional PRM) | Offline |
| 4–6 | S1 (Gemma server; Jev once access arrives) | If Jev is still pending, decide D2 |
| 6–9 | S2 pilot | GPU-heavy labels; run shards back to back |
| 10–12 | S4 | Decide D3, D4 |
| 12–14 | S5 | |
| 15–21 | S6 | Most of the GPU time and Jev spend |
| 22–28 | S7 | |
| later | S8 | Only if time allows |

## 5. Habits that keep it reproducible

- Every run writes its resolved config and git commit into its results folder (already
  automatic); commit before real runs so the hash means something.
- Never call paid APIs in tests; tests use mocks and recorded fixtures only.
- Prompt text lives only in `llm/prompts.py`; every change bumps `PROMPT_VERSION`.
- After each phase, write `results/PHASE_<n>_NOTES.md`: what was built, what was verified,
  open issues, cost.
