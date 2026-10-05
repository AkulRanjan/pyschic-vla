# ThoughtZero: results report

> **Status (2026-10-05): method, judges and setup are written; no real experiment has run
> yet** (no API credit). Sections marked _[pending]_ are filled from generated artifacts after
> the runs in `RUNBOOK.md`; never type numbers by hand. Provenance for every number: the run
> folder's `config.yaml`, `git_commit.txt`, and `prompt_version` per row.

## 1. Summary

- **Question:** can an off-the-shelf, calibrated judge (TypeSafe's **Jev**) act as both the
  policy (a prior over candidate next steps) and the value (is the solution so far sound?) of
  an AlphaZero-style search over reasoning steps, with no training, so that a small generator
  with search beats sampling-and-voting at the same compute?
- **Generator:** Gemma 4 **26B-A4B** (a mixture-of-experts model, ~4B active parameters) via a
  hosted API. The original design used the 4B Gemma 4 E4B, which no hosted API serves
  (`results/DECISIONS.md`). The claim is therefore about a 26B MoE model, not a 4B dense one.
- **Judge:** **jevos-v4**, an open-source model that speaks Jev's protocol and runs locally,
  replacing TypeSafe's Jev (`results/DECISIONS.md`). Every judge result below is about jevos;
  the question becomes whether a *free, local* judge can guide the search.
- **Pilot decision:** _[pending: GO / PARTIAL / NO-GO, Jev AUROC x.xx [lo, hi]]_
- **Headline result:** _[pending: ThoughtZero at n_simulations=… vs self-consistency at matched
  completion tokens: Δ = … pp [CI], p = …]_

## 2. Method

**Search space.** A state is the problem plus the reasoning steps so far; an action is one
next step (a sentence to a short paragraph). A step containing `\boxed{…}` or "final answer"
ends the solution, as does reaching `max_depth` = 20 steps.

**One simulation** (`search/mcts.py`): select a leaf by PUCT, expand it or evaluate it, back up.

- **Selection (PUCT, single-agent):** `Q(s,a) + c_puct · P(s,a) · √N(s) / (1 + N(s,a))`, with
  `c_puct` = 1.5, first-play urgency 0 for unvisited children, and `√max(1, N)` so priors
  matter from the first visit. Values are never negated (there is no opponent). Ties go to the
  higher prior, then the lower index.
- **Expansion:** the generator proposes `k` = 4 candidate next steps; exact and near duplicates
  (token Jaccard ≥ 0.9) are merged; **one judge call** returns the priors over the unique
  candidates and the value V(s) of the current state. With `search.prior_floor` = f, priors
  become (1 − f)·P + f/k so that a candidate the judge zeroes can still be explored (off by
  default; compared in the ablations).
- **Terminal nodes:** valued once by the judge's "is the final answer correct?" and cached.
- **Backup:** `N += 1`, `W += v` on every node from the root to the leaf.
- **Answer extraction:** `most_visited` follows the most-visited child from the root (and
  completes greedily if the path ends early); `value_vote` groups all terminal answers and
  picks the answer with the largest `Σ N·value`. Both are logged, with their agreement.
- **Concurrency:** `parallel_sims` simulations run at once with virtual loss (each in-flight
  path counts as extra visits worth 0, steering concurrent simulations apart). A simulation
  that reaches a leaf another one is expanding waits, then keeps descending, instead of
  re-backing-up the same leaf (a documented deviation from the spec; with 8 simulations in
  flight the spec version wasted 28 of 64 simulations on the toy task).
- **Ground truth never enters the search**: the runner passes only the question text.

## 3. Judges

All judges answer the same three questions over the same state text (`llm/prompts.py`,
judge section): *is every step so far correct?* (value), *which candidate next step is most
likely to lead to a correct answer?* (priors), *is the final answer correct?* (terminal value).

- **Jev** (`judge/jev.py`), the subject of the study: one request per expansion with a `noul`
  question (value = P(yes)) and a `choice` question (priors = the probability of each
  candidate). Protocol from the official API reference; cached on disk keyed by the exact
  request, the model and the prompt version, so reruns are free; a hard USD cap; optional
  option shuffling against first-option bias; long states truncated in the middle. Real
  responses are rounded to two decimals, with exact zeros for options Jev rules out.
- **Gemma self-judge** (B4, `judge/self_judge.py`): the generator model answers the same
  questions; values from P(Yes)/(P(Yes)+P(No)) and priors from the letters A, B, C, … of the
  first output token's log-probabilities.
- **PRM** (B5, `judge/prm.py`): `Qwen/Qwen2.5-Math-PRM-7B` scores every step; the value is the
  lowest step score; priors are uniform (or come from another judge through a hybrid).
- **Hybrid**: priors from one judge, values from another; used for the PARTIAL/NO-GO
  fallbacks and the prior-only / value-only ablations.

## 4. Setup and baselines

- **Generator:** Gemma 4 26B-A4B over a chat API (OpenRouter, restricted to bf16 providers that
  honour every parameter; or Google's Gemini API). Hosted APIs can't continue a partial
  answer, so each candidate step comes from a separate request asking for "only the next step"
  (`llm/chat_generator.py`); temperature 0.9, top-p 0.95; solutions up to 4,096 tokens. Seeds
  aren't honoured on these routes, so generations are not bit-reproducible.
- **Data:** MATH-500 (main), AIME 2024–2026 (with a pre/post training-cutoff split), MATH train
  (pilot and all tuning: no tuning touches test data).
- **Grading:** answer extraction from `\boxed{}` plus `math-verify` equivalence with a timeout
  (500/500 agreement with the reference answers on MATH-500).
- **Baselines:** B1 chain of thought (one greedy solution); B2 self-consistency (majority vote
  over N samples, N matched to ThoughtZero's completion tokens); B3 best-of-N (B2's samples
  reranked by the judge); B4 ThoughtZero with the Gemma self-judge; B5 ThoughtZero with PRM
  values; C the large model (Gemma 4 31B, one greedy solution).
- **Compute axis:** generator completion tokens per problem; Jev cost reported separately.

## 5. Pilot results _[pending]_

_Paste from `results/pilot/report.md`: decision, per-judge AUROC / Brier / ECE with
problem-level bootstrap CIs, reliability diagrams, by-depth table, prompt sensitivity, label
statistics, Jev cost and latency. The pilot covers level-5 MATH train problems only (D13)._

## 6. Main results _[pending]_

![Accuracy vs compute](figures/accuracy_vs_compute_math500.png)

_Main table from `figures/main_table.md`: accuracy [95% CI], mean completion tokens, wall time,
Jev USD, paired bootstrap Δ vs B2 at matched compute. By level (`by_level.md`); AIME pre/post
cutoff (`aime_split.md`); sanity audit (`audit.md`)._

## 7. Ablations _[pending]_

_k ∈ {2, 6}; c_puct ∈ {0.5, 3.0}; value_vote extraction; prior-only / value-only judges;
per-candidate noul priors; short steps; prior floor; search diagnostics (`diagnostics.md`)._

## 8. Limitations

Known before any result:

- **The generator is a 26B MoE model**, not the 4B model the method was designed for; a
  stronger generator leaves less for search to fix.
- **Jev is weak at arithmetic by its maker's account** ("not a calculator"); in early checks
  both Jev and the self-judge rated an arithmetic slip as correct.
- **Low branching diversity:** at temperature 0.9 the 26B model's candidate next steps were
  often rephrasings of one step.
- **Not bit-reproducible:** hosted routes ignore seeds; disk caches make reruns free but not
  identical once anything upstream changes.
- **The pilot covers hard (level-5) training problems only.**

_[Add after the runs: if ThoughtZero doesn't beat self-consistency at matched compute, say so
and analyse where it helps (by difficulty, by depth, by judge quality; spec §11).]_

## 9. Reproduction

`RUNBOOK.md` has every command in order with costs. Offline checks (no keys):

```bash
git clone https://github.com/AkulRanjan/pyschic-vla.git && cd pyschic-vla
pip install -e ".[dev]"
pytest
python scripts/smoke_test.py --mock
```
