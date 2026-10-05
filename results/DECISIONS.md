# Decisions

Protocol and design decisions, newest first. Open decisions are listed in `PLAN.md` §2.

## 2026-10-05 — Judge: jevos instead of TypeSafe's Jev

**Decided by:** Prakhar ("swap it for jev"), after being told the consequences.

**What:** the judge is **jevos-v4** ([feder-cr/jev](https://github.com/feder-cr/jev), MIT), an
open-source model that speaks Jev's wire format and runs locally on the CPU (`jev serve`,
`http://127.0.0.1:8017/v1/systemone`), instead of TypeSafe's jev-1.13 through OpenRouter
(`judge.transport: jevos`, the new default). Same questions, same parsing.

**Why:** Jev needs purchased API credit; jevos is free, unlimited and fast (25–110 ms per call).

**Consequences:**
- **The study is now about jevos, not Jev.** jevos is a ~0.6 GB model trained on policy and rule
  decisions (its README's benchmarks); nothing shows it can check maths, and real Jev already
  missed an arithmetic slip. The report, pilot decision and every number must name jevos.
- Context is 8,192 tokens: states are capped at 6,000 tokens on this route.
- Cost and budget for the judge are 0 on this route; the cache keeps jevos answers separate.
- Running jevos means running a third-party pre-built binary (`jev.exe`) and model, which the
  owner accepted. Verify the downloads against the release's `SHA256SUMS.txt`.
- The TypeSafe routes (`openrouter`, `direct`, ...) still work: set `JEV_TRANSPORT`.

## 2026-10-05 — Full solutions may be 4096 tokens (all experiments)

`generator.max_solution_tokens` 2048 -> 4096 in `configs/default.yaml`. On level-5 problems 25%
of Gemma 26B's solutions hit 2048 tokens and were cut off without an answer, which would count
as wrong for every method that writes full solutions (CoT, self-consistency, best-of-N, the
pilot, and ThoughtZero's greedy completion).

## 2026-10-05 — Pilot on hard MATH train problems (D1, D13)

**Decided by:** Prakhar ("go with next phase", accepting the recommendations).

- The pilot draws from the **MATH train split**, not MATH-500 (SPEC.md §7 said MATH-500), so
  no judge-prompt tuning touches test data.
- **Level 5 only**: with Gemma 26B, 5 random train problems gave 40 prefixes all labelled
  correct (AUROC undefined); on 20 level-5 problems it solved 65%. The report must state
  that the pilot covers hard problems only.
- `generator.max_solution_tokens = 4096` for the pilot: 25% of level-5 traces hit the 2048
  limit and were cut off without an answer.
- `search.prior_floor` (PLAN D9) exists, default 0 (spec behaviour); it doesn't affect the
  pilot (no search) and gets compared in S4.

## 2026-10-05 — Generator: Gemma 4 26B-A4B on OpenRouter (replaces self-hosted E4B)

**Decided by:** Prakhar.

**What:** the generator is `google/gemma-4-26b-a4b-it` served by OpenRouter
(`generator.api=chat`), not Gemma 4 E4B on a self-hosted GPU. Baseline C (the large-model
ceiling) becomes `google/gemma-4-31b-it`, also on OpenRouter.

**Why:** the laptop GPU (6 GB) can't run E4B; Colab sessions (12 h max, a new tunnel each
time) are awkward for long runs; OpenRouter doesn't host E4B. 26B-A4B is a mixture-of-experts
model with ~4B parameters active per token, the closest hosted match.

**Consequences:**
- The research claim changes from "a ~4B model" to "a 26B MoE model with ~4B active
  parameters". The report must say so.
- OpenRouter can't continue a partial assistant turn, ignores `n`, and ignores `seed`
  (docs/verified_apis.md, G11). The chat generator (`llm/chat_generator.py`) asks for the
  next step explicitly and makes `k` parallel requests. That deviates from SPEC.md §5.2
  (raw completions, `n=k`, prefix caching), and generator outputs aren't bit-reproducible.
- Cost is per token: $0.09/M input, $0.30/M output (2026-10-05). Each of the `k` proposals
  re-sends the prompt, so the prompt cost is paid `k` times.
- The self-hosted path (`generator.api=completions`, `notebooks/colab_gemma.ipynb`) still
  works, if E4B is wanted later.

## 2026-10-05 — Spec §6.2 deviation: waiting simulations keep descending

A simulation that reaches a leaf another simulation is expanding waits, then continues
down from it, instead of re-backing-up the leaf's value. On the toy task with 8 parallel
simulations, the spec version spent 28 of 64 simulations re-visiting the first-expanded
branch and answered wrong. `parallel_sims=1` is unaffected. (`search/mcts.py`)
