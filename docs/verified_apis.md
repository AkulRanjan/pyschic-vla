# Verified APIs

Every entry here must cite a link to official docs, or a real fixture file in
`tests/fixtures/jev/` that proves the behavior. No guessed field names (spec §13.5).

## Jev (TypeSafe AI)

**Access status**
- [ ] TypeSafe direct-API waitlist joined (`https://api.typesafe.ai/v1/systemone`) — date requested: _____
- [ ] OpenRouter key obtained, `typesafe/jev-1.13` availability confirmed
- [ ] Cloudflare Workers AI backup route (`typesafe/jev`) checked
- [ ] Vercel AI Gateway backup route (`typesafe-ai/jev`) checked
- [ ] ToS read (TypeSafe + gateway) — benchmarking/publishing allowed? note below

**ToS notes**
> (summarize what's allowed re: publishing benchmark results / stress-testing, with link)

**VERIFY checklist**

| Question | Answer | Source / link |
|---|---|---|
| Does `choice` return a probability for every option, or only the top pick? | | |
| Exact request field names (`state`, `questions`, question `type`, `instructions`, options field name, `model`) | | |
| Exact response shape (answer + probabilities location, `noul` yes/no representation) | | |
| Does the response report `usage` (input tokens)? | | |
| Is the model pinnable to `jev-1.13.0` on each route? | | |
| Does OpenRouter use the same typed `questions` request, or a chat-style wrapper? | | |
| Rate limits (RPM / TPM), concurrency limits | | |
| Max options in a `choice` (spec says 255) and max context (32K) | | |
| `typesafe-sdk-python` API quality (sync/async) vs raw HTTP | | |
| Latency p50 / p95 over ~20 calls | | |

## Gemma (owned by Person 3 — reference only)

(Person 3 fills this in; do not edit.)
