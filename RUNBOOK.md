# ThoughtZero runbook: from API keys to the final report

Everything is built and tested offline (`pytest`, `smoke_test.py --mock`). What's left is
**running** it, which needs API access and some money. This file says what to run, in which
order, what each step costs, and where to stop and decide. Status and decisions live in
`PLAN.md` and `results/DECISIONS.md`.

Every run is **resumable**: re-running a command skips finished rows and retries failed ones
(rate limits, out-of-credit errors, network drops). Nothing is lost if a run stops part-way.

---

## 1. One-time setup

```bash
git clone https://github.com/AkulRanjan/pyschic-vla.git && cd pyschic-vla
python -m venv .venv && .venv\Scripts\activate         # Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env                                    # then fill in .env (below)
pytest                                                  # offline, no keys: must pass
python scripts/smoke_test.py --mock                     # offline: prints "MCTS ... 13 -> OK"
```

### Choose how Gemma (the generator) is served

| Route | `.env` | Cost | Speed | Notes |
|---|---|---|---|---|
| **OpenRouter** (recommended) | `GEMMA_BASE_URL=https://openrouter.ai/api/v1`, `GEMMA_MODEL=google/gemma-4-26b-a4b-it`, `GEMMA_MAX_RPM=0`, prices as in `.env.example` | per token (table in §2) | fast | Needs purchased credit. Supports logprobs, so the Gemma self-judge (B4) works |
| **Gemini API free tier** | `GEMMA_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/`, `GEMMA_MODEL=gemma-4-26b-a4b-it`, `GEMINI_API_KEY=...`, `GEMMA_MAX_RPM=28`, prices `0` | $0 | **30 requests/minute** | Fine for the pilot (~7 h) and dev runs; far too slow for the main sweep (days). No logprobs: B4 can't run there |
| Self-hosted vLLM (L4/A100) | `GEMMA_API=completions`, see `docs/gpu_setup.md` | GPU rental | fast | Can serve the original Gemma 4 **E4B**; a T4 can't |

Keep **one route for all results**: the pilot's labels, the search and the baselines should
come from the same Gemma (`results/DECISIONS.md`).

### The judge: jevos (default) or Jev

**Default since 2026-10-05: jevos** (`results/DECISIONS.md`), free and local. Download and start
it once per session (Windows PowerShell; Linux/macOS use the `.tar.gz` build):

```powershell
mkdir P:\jevos; cd P:\jevos
gh release download jevos-v4 --repo feder-cr/jev --pattern "jev-windows-x64.zip" --pattern "jevos-v4-openvino-int8.zip" --pattern "SHA256SUMS.txt"
Get-FileHash jev-windows-x64.zip, jevos-v4-openvino-int8.zip -Algorithm SHA256   # compare with SHA256SUMS.txt
Expand-Archive jev-windows-x64.zip -DestinationPath .
cd jev; Expand-Archive ..\jevos-v4-openvino-int8.zip -DestinationPath .        # creates model.\jev.exe serve                                                                # keep this window open
```

`.env`: `JEV_TRANSPORT=jevos` (no key). Check it: `python scripts/probe_jev.py --transport jevos`.

**To use TypeSafe's Jev instead** (the original study):

`OPENROUTER_API_KEY=...`, `JEV_TRANSPORT=openrouter`. Jev is cheap (~$0.04 per 1,000 calls on
short states), but the account needs **purchased** credit; the free allowance runs out fast
(HTTP 402 "Insufficient credits"). Check a key costs nothing:

```bash
python scripts/probe_jev.py          # one call, ~$0.00002; prints the answers and checks
```

### Spending safety

- `budget.max_usd` (default $5) hard-caps **Jev**; `budget.max_gemma_usd` (default $10)
  hard-caps **hosted Gemma** (needs the prices in `.env`). A run that hits a cap stops cleanly.
- Every script prints a cost estimate first and refuses anything above $1 without `--yes`.
- Also set a credit limit on the OpenRouter key itself (openrouter.ai → Keys).

---

## 2. Order of runs and cost (OpenRouter prices, rounded up)

| # | Step (PLAN.md) | Command(s) | Gemma | Jev | Stop and decide? |
|---|---|---|---|---|---|
| 0 | Smoke test (S1, done once) | `python scripts/smoke_test.py --n-sims 4 --k 3` | <$0.01 | <$0.01 | must print PASS |
| 1 | **Pilot (S2)** | §3 | ~$3–4 | ~$0.25 | **yes: GO / PARTIAL / NO-GO** |
| 2 | Dev run, 50 problems (S4) | §4 | ~$0.5 | ~$0.05 | tune on the train split only |
| 3 | Baselines on dev (S5) | §5 | ~$2 | ~$0.1 | — |
| 4 | Main runs (S6) | §6 | ~$50–60 | ~$2–3 | — |
| 5 | Ablations (S7) | §7 | ~$25–35 | ~$1 | — |
| 6 | Plots, report, release (S7) | §8 | — | — | — |

**Full plan ≈ $90–100**, mostly Gemma in steps 4–5. **A smaller study** (pilot, dev runs, and
MATH-500 at `n_simulations=16` only, with B1/B2/C) costs **≈ $15–20**. See §6 for the cuts.

---

## 3. Pilot: can Jev judge reasoning steps? (S2, the go/no-go gate)

`configs/pilot.yaml`: 200 **level-5 MATH train** problems, 8 prefixes each, 8 completions per
prefix (decisions D1, D13). Partial outputs from earlier attempts may exist in `results/pilot`;
if they came from a different Gemma route, move that folder aside first.

```bash
python scripts/run_pilot.py --stage traces                  # 200 solutions
python scripts/run_pilot.py --stage label --yes             # ~11,000 continuations: the cost
python scripts/run_pilot.py --stage score --judges jev,self --yes   # self only on OpenRouter
python scripts/run_pilot.py --stage sensitivity --yes       # Jev x 3 phrasings, 50 problems
python scripts/run_pilot.py --stage report                  # -> results/pilot/report.md
```

`report.md` ends with the decision: **GO** (Jev AUROC ≥ 0.75), **PARTIAL** (0.65–0.75: hybrid
judge), **NO-GO** (< 0.65: Jev prior only, value from the self-judge or a PRM). Record it in
`results/DECISIONS.md` and set `judge.*` in `configs/default.yaml` accordingly **before** S4.
Also decide D6 (turn `judge.shuffle_options` on if the report shows first-option bias).

---

## 4. First real search run (S4)

```bash
python scripts/run_experiment.py --subset dev --set "eval.methods=[tz]" \
    --set search.n_simulations=16 --run-id tz16_dev
python scripts/view_tree.py results/tz16_dev                       # list problems
python scripts/view_tree.py results/tz16_dev --problem <id> --min-visits 2
```

Look at ~10 trees for: steps too long/short, near-duplicate siblings, searches that never
reach an answer, near-uniform priors, values all ≈ 1. Decide `parallel_sims` (D4: try 1, 4, 8)
and `search.prior_floor` (D9: try 0 vs 0.1). **Tune on the train split only**, never on
MATH-500 or AIME:

```bash
python scripts/run_experiment.py --set eval.dataset=math_train --set eval.n_problems=50     --set "eval.methods=[tz]" --set search.parallel_sims=4 --run-id tune_ps4
```

## 5. Baselines on the dev set (S5)

```bash
python scripts/run_experiment.py --subset dev --set "eval.methods=[cot,'sc:64',tz_self]" --run-id base_dev
python scripts/run_experiment.py --subset dev --set "eval.methods=['bon:64']" \
    --reuse-samples results/base_dev --run-id bon_dev                 # reranks B2's samples
python scripts/run_experiment.py --config configs/c31b.yaml --subset dev --run-id c31b_dev
python scripts/make_plots.py --runs results/tz16_dev results/base_dev results/bon_dev \
    results/c31b_dev --sample-ns 1,2,4,8,16,32,64 --out results/figures_dev
```

B5 (PRM judge, `tz_prm`) needs `pip install -e ".[prm]"` and a ~16 GB GPU (D7); skip it otherwise.

## 6. Main runs (S6)

```bash
for n in 8 16 32 64; do
  python scripts/run_experiment.py --config configs/exp_main.yaml --set "eval.methods=[tz]" \
      --set search.n_simulations=$n --run-id tz${n}_math500 --yes
done
python scripts/run_experiment.py --config configs/exp_main.yaml --set "eval.methods=[cot,'sc:64']" --run-id base_math500 --yes
python scripts/run_experiment.py --config configs/exp_main.yaml --set "eval.methods=['bon:64']" \
    --reuse-samples results/base_math500 --run-id bon_math500 --yes
python scripts/run_experiment.py --config configs/c31b.yaml --run-id c31b_math500 --yes
# AIME: repeat with --set eval.dataset=aime2024 (and aime2025, aime2026)
```

Long runs can be split: `--shard 0/3`, `--shard 1/3`, ... then
`python scripts/run_experiment.py --merge results/a results/b --run-id merged`.
Match self-consistency's N to ThoughtZero's tokens per problem with `make_plots.py
--sample-ns` (one B2 run at 64 samples covers every N ≤ 64).

**To cut cost:** drop `n_simulations=64` (≈ half of the TZ cost), run AIME at one
`n_simulations`, or use `--set eval.n_problems=200`.

## 7. Ablations (S7)

```bash
for v in k2 k6 cpuct_0.5 cpuct_3.0 value_vote prior_only value_only per_candidate_noul short_steps; do
  python scripts/run_experiment.py --config configs/ablations.yaml --subset ablation \
      --variant $v --run-id abl_$v --yes
done
```

200 fixed problems at `n_simulations=32`. Each variant costs ≈ $3–4 of Gemma; pick the ones
that matter most if the budget is tight.

## 8. Plots, report, release

```bash
python scripts/make_plots.py --runs results/tz*_math500 results/base_math500 results/bon_math500 \
    results/c31b_math500 --sample-ns 1,2,4,8,16,32,64 --out results/figures
```

Fill in `results/REPORT.md` §1, §5–§8 from the generated files (never type numbers by hand),
then do a fresh-clone check (§1 commands) and tag `v0.1.0`.

---

## Demonstration (for presentations)

```bash
python scripts/demo.py --mock --html demo.html     # offline, no keys: always works
python scripts/demo.py --html demo.html            # real Gemma + judge, from .env (~2–5 min)
python scripts/demo.py --problem "What is 3^4 - 2^5?" --gold 49 --html demo.html
```

It narrates the search live: each expansion's candidate next steps with the judge's prior as a
bar, the judge's value of the state, each finished solution with the judge's verdict. Then it
prints the chosen solution, the answer (✓/✗), the search tree and a no-search baseline on the
same problem, and `--html` writes an interactive page of the whole tree (collapsible nodes,
the chosen path in green) to open in a browser.

**Suggested walkthrough (5 minutes):** run the mock demo first. The toy judge's prior always
favours "add 5", so plain greedy ends at 15 (wrong), while the search follows the judge's
*values* and finds 13. Open `demo.html` and follow the green path. Then run the real demo on
a maths problem to show the same machinery with Gemma and the judge.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `402 Insufficient credits` | OpenRouter account has no purchased credit | add credit; re-run (resumes) |
| `429 ... RESOURCE_EXHAUSTED ... limit: 30` | Gemini free tier rate limit | set `GEMMA_MAX_RPM=28`; re-run |
| `REFUSING: estimated $x > $1.00` | cost guard | check the estimate, then add `--yes` |
| `budget exceeded` | `budget.max_usd` / `max_gemma_usd` reached | raise it in the config if intended |
| `judge kind='self' not available` | route has no logprobs (Gemini) or no generator | use OpenRouter for B4 |
| `<thought>` text in outputs | thinking mode on | Gemini route sets it off automatically; check the route |
| `Gemma server not reachable` | wrong `GEMMA_BASE_URL` or server down | fix `.env` |
