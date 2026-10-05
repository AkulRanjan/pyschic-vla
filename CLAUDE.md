# CLAUDE.md

ThoughtZero: training-free AlphaZero-style MCTS over math reasoning steps. Gemma 4 (26B-A4B via a hosted API; see `results/DECISIONS.md`) proposes steps; a Jev-protocol judge (jevos by default, see `results/DECISIONS.md`) gives priors and values. The full spec is in `SPEC.md`.

## Start of every session
1. Read `SPEC.md`, this file, `PLAN.md` (status, remaining phases, open decisions) and `RUNBOOK.md` (how to run everything, with costs).
2. **Prakhar now owns the whole project alone** (since 2026-10-05). The per-person ownership rules in `team/*.md` no longer apply; those files are kept as background on how each module was designed (Akul: generator/data, Jagriti: judge, Harjas: eval/pilot).
3. Work one phase at a time. Check the phase's definition of done before moving on.

## Rules (SPEC.md §13)
1. **Never call paid APIs in tests.** Tests use mocks (`thoughtzero.mocks`) and recorded fixtures only. Tests that need the network are marked `@pytest.mark.network` and skipped by default.
2. **Ask the user before:**
   - any run with an estimated cost above $1;
   - downloading weights larger than 5 GB;
   - changing the experimental protocol in SPEC.md §7–§8.
3. Never hardcode, log or print API keys. Read them from the environment (`.env`, see `.env.example`).
4. Resolve **VERIFY** items from official docs and record them, with links, in `docs/verified_apis.md`. Don't guess field names.
5. All prompt text lives in `src/thoughtzero/llm/prompts.py`. Every prompt change bumps `PROMPT_VERSION`.
6. Every experiment script writes the resolved config and the git commit hash into its results folder.
7. Code style:
   - type hints throughout; `ruff` and `mypy` clean;
   - small pure functions in `search/`;
   - side effects only in `llm/`, `judge/` and `eval/runner.py`.
8. After each phase, write `results/PHASE_<n>_NOTES.md`: what was built, what was verified, open issues, cost.

## Conventions (team/<Name>.md §B4)
- **Steps are stored without the `Step n:` prefix**, stripped. `prompts.format_steps` adds numbering.
- `complete` / `sample_completions` return only the **new** steps.
- Judge priors sum to 1; values are in [0, 1].
- `search()` never receives the ground-truth answer.
- Token and cost accounting goes through `thoughtzero.accounting` (`record_gemma`, `record_jev`) via a per-problem ContextVar ledger.
- Interface files (`types.py`, `config.py`, `accounting.py`, `configs/default.yaml`) are the contract every module codes against: change them deliberately, and update the mocks and tests in the same commit.

## Commands
```bash
pip install -e ".[dev]"
ruff check . && ruff format --check . && mypy && pytest
python scripts/smoke_test.py --mock
```
