# Eval and pilot: what Person 4's code needs from each owner

Owner: Person 4 (Harjas). This lists every assumption `eval/`, `pilot/` and the
`scripts/run_*.py` / `make_plots.py` scripts make about code owned by someone else. If yours
differs, tell Harjas: each assumption sits in one place (file named below), so adapting is quick.

## Person 1 (Prakhar): search and scaffold

Already compatible; nothing blocking.
- TZ is wrapped in `eval/methods.py::TZMethod`. It calls
  `search.mcts.search(question, generator, judge, cfg.search, seed=cfg.seed, extract=, normalize=)`
  and stores `answer_steps`, `stats` and `tree` (the `tree_dump`) in `raw`.
- The spec §9 diagnostics read `stats["expansions"]`, `stats["max_depth"]`,
  `stats["most_visited_reached_terminal"]` and `stats["modes_agree"]`. Please keep those keys.
- Ledger: `search()` calls `record_search(terminal_leaves=1)` only, so the ledger's `expansions`
  and `max_depth` stay 0. The diagnostics use `stats`, so this isn't blocking, but calling
  `record_search(expansions=1, depth=len(node.steps))` in `_expand` would fill the ledger too.
- `.gitignore` ignores the whole `results/` folder, but `results/REPORT.md` must be committed
  (A8). Suggest `results/*` plus `!results/REPORT.md`. Until then it is force-added.

## Person 2 (Jagriti): judges

**Status (fetched 2026-10-05): not on `main` yet.** `jev-judge-reasoning` was built on a local
copy of the scaffold, so before it can merge it needs rebasing onto `main`:
- it re-creates `types.py`, `accounting.py` and `llm/prompts.py` (they exist on `main`; keep
  `main`'s and add only the judge section of `prompts.py`);
- `make_judge` takes a `dict`; on `main` it takes `JudgeCfg` (`judge.kind`, `prior_from`, ...);
- kinds `self` and `prm` are missing (needed for B4, B5 and the pilot);
- `SOUND_VARIANTS` has 3 phrasings, but `step_sound` always uses `SOUND_INSTRUCTION`.

What eval and the pilot use:
- `judge.factory.make_judge(cfg: JudgeCfg) -> Judge`. Judge variants are
  `cfg.judge.model_copy(update=...)`:
  - `tz`: `cfg.judge` unchanged (ablations edit it through `configs/ablations.yaml` variants)
  - `tz_self` (B4): `kind="self"`
  - `tz_prm` (B5): `kind="hybrid", prior_from="self", value_from="prm"`
  - pilot scoring: `kind` = `jev`, `self`, `prm`

  Until it lands, those methods and pilot judges report "not available yet" and are skipped.
- **Request: prompt-sensitivity knob.** The pilot scores the `sound` question with each of
  `prompts.SOUND_VARIANTS` (3 phrasings, §7) by setting `judge.sound_variant = i`. Please add
  `sound_variant: int = 0` to `JudgeCfg` (shared file, PR to all four) and use
  `SOUND_VARIANTS[cfg.sound_variant]` in `step_sound`. Until then the `sensitivity` stage is
  skipped with a message.
- `BudgetExceeded` must propagate out of judge calls (don't swallow it); the runner stops the
  whole run on it.
- `record_jev(input_tokens, usd, cache_hit)` on every request, `usd=0` on cache hits.
- Cost gate: `eval/runner.py::estimate_jev_usd` is a rough stand-in (2,000 tokens per call,
  one call per simulation). Your `budget.estimate_run_cost(n_calls, avg_tokens)` can replace it
  once merged.

## Person 3 (Akul): generator, data and baselines

**Status: on `main` and integrated.** Nothing blocking.
- Generator: `llm.gemma.OpenAICompatibleGenerator(cfg.generator)`; its `tokenizer.count_tokens`
  is passed to B2/B3 (`eval.methods.token_counter`).
- Grading and steps go through `eval/toolkit.py::real_toolkit()` (`extract_answer`,
  `normalize_answer`, `is_equivalent`, `split_steps`, `format_steps`). Checked on mock traces.
- Datasets: `load_math500`, `load_aime(year)`, `dev_subset`, `ablation_subset`,
  `pilot_subset(n, seed, split=)`.
- Baselines as built by the registry:
  - `cot`: `ChainOfThought(generator, name="cot")`
  - `c31b`: `ChainOfThought(generator, where="<model> @ <base_url>", name="c31b")`. Run it with
    `--set generator.model=... --set generator.base_url=...` pointing at the 31B endpoint.
  - `sc:N/M`: `SelfConsistency(generator, N, count_tokens, n_max=M)`
  - `bon:N/M`: `BestOfN(judge, N, generator=, count_tokens=, stored=, n_max=M)`; with
    `--reuse-samples <B2 run dir>`, it reranks B2's samples instead of sampling again.
- Compute axis: rows use `raw["completion_tokens_at_n"]` when present (`compute_tokens`), so
  B2 voting N of M samples, and B3 on reused samples, are charged for N samples, not M (or 0).
  `make_plots.py --sample-ns 1,2,4,...` derives every N <= M from one B2/B3 run.

## Runner guarantees (for everyone)

- `solve()` receives the problem with `answer=""`. Grading happens only in the runner.
- One JSONL row per (problem, method_id) is appended and flushed as soon as it finishes. Resume
  skips finished pairs. An exception becomes a flagged error row. `BudgetExceeded` stops the run.
- `method_id = name[param=value,...]`, e.g. `tz[n_simulations=16]`, `sc[n=8]`.
- Rows carry `config_hash` and `protocol_hash`. The latter ignores run-only settings
  (`eval.concurrency`, `eval.run_id`, `eval.out_dir`), so shards and resumed sessions with a
  different concurrency still merge.
