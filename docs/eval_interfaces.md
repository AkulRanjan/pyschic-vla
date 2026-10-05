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

- `judge.factory.make_judge(cfg: JudgeCfg) -> Judge` is used as-is. Judge variants are
  `cfg.judge.model_copy(update=...)`:
  - `tz`: `cfg.judge` unchanged (ablations edit it through `configs/ablations.yaml` variants)
  - `tz_self` (B4): `kind="self"`
  - `tz_prm` (B5): `kind="hybrid", prior_from="self", value_from="prm"`
  - pilot scoring: `kind` = `jev`, `self`, `prm`
- **Request: prompt-sensitivity knob.** The pilot scores the `sound` question with each of
  `prompts.SOUND_VARIANTS` (3 phrasings, §7). The runner sets `judge.sound_variant = i`
  (index into `SOUND_VARIANTS`). Please add `sound_variant: int = 0` to `JudgeCfg` (shared file,
  PR to all four) and use it in `step_sound`. Until then the `sensitivity` stage is skipped with
  a message.
- `BudgetExceeded` must propagate out of judge calls (don't swallow it); the runner stops the
  whole run on it.
- `record_jev(input_tokens, usd, cache_hit)` on every request, `usd=0` on cache hits.
- Cost gate: `eval/runner.py::estimate_jev_usd` is a rough stand-in (2,000 tokens per call,
  one call per simulation). If you build a proper estimator, it replaces that one function.

## Person 3 (Akul): generator, data and baselines

- Generator: `llm.gemma.OpenAICompatibleGenerator(cfg.generator)`, calling `record_gemma` on
  every request from the event loop (not from threads).
- Grading and steps: `data.grading.extract_answer`, `normalize_answer`, `is_equivalent` and
  `llm.prompts.split_steps`, `format_steps`, all through `eval/toolkit.py::real_toolkit()`.
- Datasets: `load_math500()`, `load_aime(year)`, `dev_subset(n, seed)`,
  `ablation_subset(n, seed)`, `pilot_subset(n, seed, split=)`. Ids must be stable.
- **Request: baseline constructors.** The registry builds:
  - `ChainOfThought(generator=..., cfg=...)` for `cot` and `c31b`
  - `SelfConsistency(generator=..., cfg=..., n=N)` for `sc:N`
  - `BestOfN(generator=..., judge=..., cfg=..., n=N)` for `bon:N`

  `N` comes from the method spec (`eval.methods: ["sc:8"]`), because `EvalCfg` has no field for
  it. C (`c31b`) is `ChainOfThought` run with `generator.base_url` / `generator.model` pointed at
  the 31B endpoint.

## Runner guarantees (for everyone)

- `solve()` receives the problem with `answer=""`. Grading happens only in the runner.
- One JSONL row per (problem, method_id) is appended and flushed as soon as it finishes. Resume
  skips finished pairs. An exception becomes a flagged error row. `BudgetExceeded` stops the run.
- `method_id = name[param=value,...]`, e.g. `tz[n_simulations=16]`, `sc[n=8]`.
- Rows carry `config_hash` and `protocol_hash`. The latter ignores run-only settings
  (`eval.concurrency`, `eval.run_id`, `eval.out_dir`), so shards and resumed sessions with a
  different concurrency still merge.
