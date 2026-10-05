# ThoughtZero: results report

> Assembled by Person 4 (Harjas). Each section owner writes their own section by **day 26** (B5).
> Numbers in sections 5–7 are pasted from generated artifacts. Never type them by hand:
> `results/pilot/report.md`, `results/figures/main_table.md`, `by_level.md`, `aime_split.md`, `diagnostics.md`.
> Provenance for every number: the run folder's `config.yaml`, `git_commit.txt`, and `prompt_version` per row.

## 1. Summary *(Person 4)*

- **Claim:** an off-the-shelf calibrated judge (Jev) can act as both policy (prior over candidate steps) and value for MCTS over reasoning steps, with no training, so a ~4B generator with search approaches a much larger model at matched compute.
- **Pilot decision (CP2):** _GO / PARTIAL / NO-GO_, Jev AUROC _x.xx [lo, hi]_.
- **Headline result:** _TZ at n_simulations=… vs B2 self-consistency at matched completion tokens: Δ = … pp [CI], p = …_

## 2. Method *(Person 1)*

_PUCT selection, expansion with K Gemma candidates, one Jev call per expansion (prior + value), backup, answer extraction (`extract_mode`)._

## 3. Judges *(Person 2)*

_JevJudge (choice prior + noul value), GemmaSelfJudge, PRMJudge, hybrid; caching, budget, option shuffling._

## 4. Setup and baselines *(Person 3)*

_Gemma 4 E4B serving, datasets (MATH-500, AIME; pilot split), grading (`math-verify`), B1 CoT, B2 self-consistency, B3 best-of-N, C large model; compute matching of B2 to TZ._

## 5. Pilot results *(Person 4)*

_Paste from `results/pilot/report.md`: decision, per-judge AUROC/Brier/ECE with problem-level bootstrap CIs, reliability diagrams, by-depth table, prompt sensitivity, label statistics, Jev cost/latency._

## 6. Main results *(Person 4)*

![Accuracy vs compute](figures/accuracy_vs_compute_math500.png)

_Main table from `figures/main_table.md`: accuracy [95% CI], mean completion tokens, wall time, Jev USD, paired bootstrap Δ vs B2 at matched compute._

_MATH-500 by difficulty level (`figures/accuracy_by_level.png`, `by_level.md`); AIME pre-/post-cutoff (`aime_split.md`)._

_Sanity audit (`figures/audit.md`): error rows per method, identical problem-ID sets, manual grader check on 20 random rows._

## 7. Ablations *(each owner)*

- Judge variants: self / PRM / hybrid / prior-only / value-only *(Person 2, Person 1)*
- Search diagnostics (spec §9): mean expansions, max depth, fraction of most-visited paths ending terminal, `most_visited` vs `value_vote` agreement *(Person 4, from `diagnostics.md`)*
- `extract_mode` agreement *(Person 4 + Person 1)*

## 8. Limitations *(all; Person 4 edits)*

_Be honest. If TZ does not beat self-consistency at matched compute, say so, and analyse where it helps (by difficulty, by depth, by judge quality; spec §11)._

## 9. Reproduction *(Person 4 + Person 1; tested on a fresh clone)*

```bash
git clone https://github.com/AkulRanjan/pyschic-vla.git && cd pyschic-vla
pip install -e ".[dev]"
pytest                                                    # offline, no keys
python scripts/smoke_test.py --mock
python scripts/run_pilot.py --config configs/pilot.yaml --stage all
python scripts/run_experiment.py --config configs/exp_main.yaml --set "eval.methods=[tz]" \
    --set search.n_simulations=16 --run-id tz16_math500
python scripts/make_plots.py --runs results/<run_ids...> --out results/figures
```
