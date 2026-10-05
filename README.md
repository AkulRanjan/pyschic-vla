# ThoughtZero

Training-free, AlphaZero-style tree search over the reasoning steps of a math solution.

- **Gemma 4** proposes candidate next steps (26B-A4B through a hosted API; the original design
  used the 4B E4B, see `results/DECISIONS.md`).
- A **Jev-protocol judge** supplies both the **policy prior** (which step to try) and the
  **value** (is the solution so far correct?), in one call per expansion: by default
  **jevos**, a free open-source model that runs locally, or TypeSafe's **Jev** via OpenRouter.
- No model is trained. The question is whether search guided by an off-the-shelf judge beats
  sampling-and-voting at the same compute.

> **Status (2026-10-05): code complete, not yet run at scale.** Everything is built and tested
> offline, and a real end-to-end smoke test passed (3/3). The experiments need API credit:
> `RUNBOOK.md` lists every run in order with its cost.

## Quickstart (offline, no keys)

```bash
git clone https://github.com/AkulRanjan/pyschic-vla.git && cd pyschic-vla
python -m venv .venv              # needs Python 3.11+
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest                            # ~420 tests, no network
python scripts/smoke_test.py --mock
python scripts/run_experiment.py --mock --run-id mock_check     # every method on a toy task
python scripts/run_pilot.py --stage all --mock                  # the whole pilot on a toy task
```

## Running for real

Copy `.env.example` to `.env`, add an OpenRouter key (for Jev, and optionally Gemma) or a
Gemini API key (free Gemma, 30 requests/minute), then follow **`RUNBOOK.md`**: pilot →
dev runs → baselines → main runs → ablations → report. Every run prints a cost estimate,
refuses anything over $1 without `--yes`, stops at a hard budget cap, and resumes where it
stopped.

```bash
python scripts/probe_jev.py                          # check the Jev key (~$0.00002)
python scripts/smoke_test.py --n-sims 4 --k 3        # 3 easy problems end to end (< $0.01)
python scripts/view_tree.py results/<run> --problem <id>   # inspect a search tree
```

## Demonstration

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

## Layout

| Path | What |
|---|---|
| `src/thoughtzero/search/` | MCTS: PUCT, async simulations with virtual loss, dedupe, answer extraction, tree view |
| `src/thoughtzero/judge/` | Jev (HTTP, every route), Gemma self-judge, PRM, uniform/constant/hybrid; cache and budget caps |
| `src/thoughtzero/llm/` | Generators (hosted chat API; self-hosted completions) and every prompt |
| `src/thoughtzero/data/` | MATH-500, MATH train, AIME loaders; answer grading |
| `src/thoughtzero/baselines/` | Chain of thought, self-consistency, best-of-N |
| `src/thoughtzero/pilot/` | The go/no-go pilot: can Jev judge reasoning steps? |
| `src/thoughtzero/eval/` | Resumable, shardable runner; metrics; plots |
| `configs/` | `default`, `pilot`, `exp_main`, `c31b`, `ablations` |
| `scripts/` | `demo`, `run_pilot`, `run_experiment`, `make_plots`, `smoke_test`, `probe_jev`, `view_tree`, `kaggle_run` |
| `notebooks/` | Self-hosting Gemma on Colab (L4/A100) or Kaggle |

## Documents

| File | What |
|---|---|
| `SPEC.md` | The original design (amended: see the note at its top) |
| `PLAN.md` | Status, remaining phases, open decisions |
| `RUNBOOK.md` | How to run everything, in order, with costs |
| `results/DECISIONS.md` | Every protocol and design change, with reasons |
| `results/REPORT.md` | The write-up (method written; results pending) |
| `docs/verified_apis.md` | API facts checked against official docs and real calls |

## License

MIT (see `LICENSE`). Gemma model weights are under their own licence.
