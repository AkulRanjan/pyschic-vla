# ThoughtZero

Training-free, AlphaZero-style tree search over the reasoning steps of a math solution.

- **Gemma 4 E4B**, a small local model, proposes candidate next steps.
- **Jev**, a calibrated decision API, supplies both the **policy prior** (which step to try) and the **value** (is the solution so far correct?), in one call per expansion.
- No model is trained.

> Work in progress. See `SPEC.md` for the full design and `team/` for who is building what.

## Quickstart (development)

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env              # fill in your own keys
pytest                            # offline, no keys needed
python scripts/smoke_test.py --mock
```

## Layout

| Path | What |
|---|---|
| `src/thoughtzero/types.py` | Shared interfaces: `Problem`, `GenOut`, `Generator`, `Judge` |
| `src/thoughtzero/config.py` | Typed YAML config with env interpolation and `--set` overrides |
| `src/thoughtzero/search/` | MCTS (PUCT, virtual loss, dedupe, answer extraction) |
| `src/thoughtzero/judge/` | Jev judge, cache, budget guard, baseline judges |
| `src/thoughtzero/llm/` | Gemma generator and all prompt templates |
| `src/thoughtzero/data/` | Dataset loaders and answer grading |
| `src/thoughtzero/baselines/` | Chain-of-thought, self-consistency, best-of-N |
| `src/thoughtzero/pilot/` | The go/no-go pilot: can Jev judge reasoning steps? |
| `src/thoughtzero/eval/` | Runner, metrics, plots |
| `configs/` | `default`, `pilot`, `exp_main` and `ablations` configs |

## License

Apache-2.0
