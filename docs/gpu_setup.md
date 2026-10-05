# GPU setup

_Status: **draft. Nothing has been run on a GPU yet; no throughput numbers.**_

## Colab (the default GPU route)

`notebooks/colab_gemma.ipynb`: open it from GitHub in Colab, pick a GPU runtime, Run all.
It picks the weights for the GPU it gets (table below), starts vLLM with a random password,
checks a greedy answer and logprobs, and measures n=4 throughput. Then either:

- **tunnel** (default): a cloudflared quick tunnel gives a private HTTPS URL. Paste the
  printed `GEMMA_BASE_URL`, `GEMMA_MODEL`, `GEMMA_TOKENIZER` and `GEMMA_API_KEY` into your
  local `.env`; every script then runs locally against Colab's GPU, and the Jev key stays
  on your machine. The URL changes every session.
- **in_colab**: run a script inside Colab, with the Jev key from a Colab secret.

The free tier gives a T4 (16 GB, no bf16), so the notebook tries the 4-bit QAT checkpoint
first, then bitsandbytes 4-bit. Colab Pro's L4 / A100 serve the bf16 weights. Sessions end
when idle (~90 min) and after at most 12 h (free tier): long runs must resume (the runner
does), and a new session means new `.env` lines.

`scripts/kaggle_run.py` + `kaggle/gpu_check/` (Kaggle's T4 x2, run from the command line)
remain as an alternative.

## Which weights on which GPU

| GPU | E4B option | Notes |
|---|---|---|
| T4 (Colab free / Kaggle; 16 GB, fp16 only) | `google/gemma-4-E4B-it-qat-w4a16-ct` (11.5 GB) | bf16 (16 GB) doesn't fit. The 4-bit QAT checkpoint should, but **fp16 correctness on a T4 is unverified**. Before any real run, check outputs for NaNs or garbage on 20 MATH train problems. |
| 24 GB+ with bf16 (Colab Pro L4 / A100; rented A10G, 3090, 4090) | `google/gemma-4-E4B-it` (bf16) | The configuration vLLM's recipe supports. Prefer this for the main experiments. |
| Laptop | Ollama `gemma4:e4b` | Smoke tests only. Ollama applies its own template and may not support `n>1`. |

⚠️ CLAUDE.md: ask before downloading weights larger than 5 GB. Every option above is larger (approved for Colab/Kaggle GPU runs, 2026-10-05).

## vLLM server (on GPU 0, leaving GPU 1 for Person 2's PRM)

```bash
pip install -U "vllm>=0.19.1"
CUDA_VISIBLE_DEVICES=0 vllm serve google/gemma-4-E4B-it \
  --max-model-len 8192 \
  --enable-prefix-caching \
  --max-logprobs 20 \
  --host 127.0.0.1 --port 8000 &
# On a T4: swap in the -qat-w4a16-ct checkpoint and add --dtype float16
until curl -s localhost:8000/v1/models >/dev/null; do sleep 5; done
```

`.env` for the scripts:

```
GEMMA_BASE_URL=http://127.0.0.1:8000/v1
GEMMA_MODEL=google/gemma-4-E4B-it
```

`GEMMA_MODEL` must be the model name the server reports at `/v1/models`. The tokenizer must be the matching HF ID; the QAT checkpoint uses the same tokenizer.

## Running against the server

```python
from thoughtzero.config import load_config
from thoughtzero.llm.gemma import OpenAICompatibleGenerator

cfg = load_config("configs/default.yaml")  # generator.base_url / model come from .env
# tokenizer_name is needed when generator.model is a quantized checkpoint or an Ollama tag
gen = OpenAICompatibleGenerator.from_config(cfg, tokenizer_name="google/gemma-4-E4B-it")
```

## Throughput (TODO: measure on day 1–2)

| Setup | Workload | tokens/s | Notes |
|---|---|---|---|
| — | `propose`, n=4, shared prefix | — | |
| — | full solutions, n=16 | — | |

## Kaggle

Use `notebooks/kaggle_server.ipynb`. Keys come from Kaggle *secrets* only. Clear outputs before sharing the notebook.
