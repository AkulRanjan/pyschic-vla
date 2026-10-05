# Phase 0 notes (SPEC.md §10 Phase 0; PLAN.md S0–S1)

Date: 2026-10-05.

## Built

- Scaffold, search (sequential and async MCTS with virtual loss), mocks, generator, data
  and grading, baselines, eval runner, pilot (earlier PRs).
- Judge layer merged; real Jev over HTTP for every TypeSafe route (`judge/client.py`).
- Chat-API generator for hosted Gemma (`llm/chat_generator.py`, `generator.api=chat`).
- Real `scripts/smoke_test.py`; `scripts/probe_jev.py`; `scripts/view_tree.py`.
- GPU tooling, not used for now: `notebooks/colab_gemma.ipynb`, `scripts/kaggle_run.py`.

## Verified

- **Jev**: protocol from the official API reference, checked with real calls over
  OpenRouter (`typesafe/jev-1.13-20260917`): every choice option gets a probability, usage
  is reported, p50 latency 406 ms, ~430 input tokens per call on short states.
  `docs/verified_apis.md` J1–J12.
- **Gemma 4 26B-A4B on OpenRouter**: chat completions work with an explicit next-step
  prompt; raw completions, assistant prefill, `n` and `seed` don't. Chat logprobs work
  (top-5). `docs/verified_apis.md` G11.
- **Real smoke test passed** (`smoke_test.py --n-sims 4 --k 3`, Gemma and Jev both on
  OpenRouter): 3/3 easy problems solved, 11 Jev calls (limit 20), Jev $0.00019 (limit
  $0.01), ~1,400 Gemma completion tokens, ~10 s per problem.

## Open issues

- Jev rated an arithmetic slip as sound (0.97) in one worked example: the pilot measures how
  often this happens (PLAN.md D8).
- Jev priors are rounded with exact zeros, so PUCT never explores those children (D9).
- Gemma 26B's next-step samples are often rephrasings of one step (2–5 distinct out of 6
  at temperature 0.9): little real branching on easy steps (D11).
- Generator outputs aren't reproducible (OpenRouter ignores `seed`): reruns differ, and
  the Jev cache only helps for identical states.
- Baselines B4 (self-judge) and B5 (PRM) aren't built (S3).

## Cost

Jev: ~$0.001 in total (probe, latency check, smoke test). Gemma on OpenRouter: < $0.002 in
total (probes and smoke test). GPU: none (one Kaggle check job, free quota).
