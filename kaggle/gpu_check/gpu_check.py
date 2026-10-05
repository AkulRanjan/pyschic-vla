"""Kaggle GPU check: can vLLM serve Gemma 4 E4B here, and does our generator work against it?

Runs as a Kaggle script kernel (``python scripts/kaggle_run.py gpu_check``). No API keys are
needed: nothing here calls Jev. Writes ``gpu_check.json`` (and vLLM log tails) to
/kaggle/working, which ``kaggle_run.py`` downloads.

Tries serving setups in order and stops at the first that works:
  A. 4-bit QAT checkpoint, fp16, one GPU      (docs/gpu_setup.md's T4 plan)
  B. full-precision checkpoint in fp16, tensor-parallel over two GPUs
For the working setup it records: sanity of greedy outputs (no NaN / garbage), logprobs
support (verified_apis G6), throughput for n=4 short steps (G10), and our own
``OpenAICompatibleGenerator`` proposing steps and completing 3 easy problems.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import traceback
import urllib.request
from pathlib import Path

OUT = Path("/kaggle/working")
REPO = "https://github.com/AkulRanjan/pyschic-vla.git"
TOKENIZER = "google/gemma-4-E4B-it"
SETUPS = [
    {
        "name": "A_qat_w4a16_fp16_1gpu",
        "model": "google/gemma-4-E4B-it-qat-w4a16-ct",
        "args": ["--dtype", "float16"],
        "env": {"CUDA_VISIBLE_DEVICES": "0"},
    },
    {
        "name": "B_full_fp16_tp2",
        "model": "google/gemma-4-E4B-it",
        "args": ["--dtype", "float16", "--tensor-parallel-size", "2"],
        "env": {},
    },
]
PROBLEMS = [
    ("What is 15% of 80?", "12"),
    ("Solve for x: 3x + 7 = 22.", "5"),
    ("What is the sum of the interior angles of a pentagon, in degrees?", "540"),
]
report: dict = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "setups": []}


def save() -> None:
    (OUT / "gpu_check.json").write_text(json.dumps(report, indent=2, default=str))


def sh(cmd: str, timeout: int = 1800) -> str:
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return (r.stdout + r.stderr)[-4000:]


def post(path: str, body: dict, timeout: int = 300) -> dict:
    req = urllib.request.Request(
        f"http://127.0.0.1:8000/v1/{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def looks_sane(text: str) -> bool:
    printable = sum(c.isprintable() or c.isspace() for c in text) / max(1, len(text))
    return bool(text.strip()) and printable > 0.95 and "nan" not in text.lower()[:20]


def wait_for_server(proc: subprocess.Popen, log: Path, limit_s: int = 1500) -> bool:
    for _ in range(limit_s // 5):
        try:
            urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=2)
            return True
        except Exception:
            if proc.poll() is not None:
                return False
            time.sleep(5)
    return False


def check_setup(setup: dict) -> dict:
    res: dict = {"name": setup["name"], "model": setup["model"], "ok": False}
    log = OUT / f"vllm_{setup['name']}.log"
    cmd = [
        "vllm", "serve", setup["model"], *setup["args"],
        "--max-model-len", "8192", "--enable-prefix-caching", "--max-logprobs", "20",
        "--host", "127.0.0.1", "--port", "8000",
    ]  # fmt: skip
    t0 = time.time()
    with log.open("w") as f:
        proc = subprocess.Popen(
            cmd, env={**os.environ, **setup["env"]}, stdout=f, stderr=subprocess.STDOUT
        )
    try:
        if not wait_for_server(proc, log):
            res["error"] = "server did not start"
            res["log_tail"] = log.read_text(errors="replace")[-3000:]
            return res
        res["startup_s"] = round(time.time() - t0)

        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(TOKENIZER)
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": "What is 2+3? Put the final answer in \\boxed{}."}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        out = post(
            "completions",
            {"model": setup["model"], "prompt": prompt, "max_tokens": 64, "temperature": 0,
             "logprobs": 20},
        )  # fmt: skip
        choice = out["choices"][0]
        res["greedy_text"] = choice["text"]
        res["sane"] = looks_sane(choice["text"]) and "5" in choice["text"]
        lp = choice.get("logprobs") or {}
        res["logprobs_top_k"] = len((lp.get("top_logprobs") or [{}])[0] or {})

        # throughput: n=4 short steps sharing one prompt (verified_apis G10)
        t1 = time.time()
        out = post(
            "completions",
            {"model": setup["model"], "prompt": prompt, "max_tokens": 128, "temperature": 0.9,
             "top_p": 0.95, "n": 4},
        )  # fmt: skip
        dt = time.time() - t1
        res["n4_completion_tokens"] = out["usage"]["completion_tokens"]
        res["n4_seconds"] = round(dt, 2)
        res["n4_tokens_per_s"] = round(out["usage"]["completion_tokens"] / dt, 1)
        res["ok"] = bool(res["sane"])
        if res["ok"]:
            res["generator"] = check_generator(setup["model"])
    except Exception:
        res["error"] = traceback.format_exc()[-3000:]
        res["log_tail"] = log.read_text(errors="replace")[-3000:]
    finally:
        if not res["ok"]:
            proc.terminate()
            proc.wait(timeout=60)
    res["server"] = proc
    return res


def check_generator(model: str) -> dict:
    """Our OpenAICompatibleGenerator against the server: propose + greedy completion."""
    sh(f"git clone -q --depth 1 {REPO} repo", 300)
    # --no-deps: keep vLLM's pinned torch/transformers; add only what vLLM doesn't bring
    sh("pip install -q --no-deps -e repo", 900)
    sh("pip install -q diskcache math-verify python-dotenv tenacity", 900)
    os.environ.update(GEMMA_BASE_URL="http://127.0.0.1:8000/v1", GEMMA_MODEL=model)
    code = f"""
import asyncio, json, time
from thoughtzero.config import Config, GeneratorCfg
from thoughtzero.llm.gemma import OpenAICompatibleGenerator
from thoughtzero.data.grading import extract_answer, is_equivalent
PROBLEMS = {PROBLEMS!r}
async def main():
    cfg = Config(generator=GeneratorCfg(base_url="http://127.0.0.1:8000/v1", model={model!r}))
    gen = OpenAICompatibleGenerator.from_config(cfg, tokenizer_name={TOKENIZER!r})
    out = []
    for q, gold in PROBLEMS:
        t = time.time()
        props = await gen.propose(q, [], 4)
        steps = await gen.complete(q, [], temperature=0.0)
        ans = extract_answer(steps[-1]) if steps else None
        out.append({{"q": q, "proposals": [p.text for p in props], "greedy_steps": steps,
                    "answer": ans, "correct": is_equivalent(ans, gold),
                    "seconds": round(time.time() - t, 1)}})
    print("RESULT" + json.dumps(out))
asyncio.run(main())
"""
    log = sh(f"cd repo && python -c {json.dumps(code)}", 1200)
    for line in log.splitlines():
        if line.startswith("RESULT"):
            return {"ok": True, "problems": json.loads(line[len("RESULT") :])}
    return {"ok": False, "log_tail": log}


def main() -> None:
    report["nvidia_smi"] = sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
    report["pip_vllm"] = sh('pip install -q -U "vllm>=0.19.1" 2>&1 | tail -5; pip show vllm torch')
    save()
    for setup in SETUPS:
        res = check_setup(setup)
        server = res.pop("server")
        report["setups"].append(res)
        save()
        if res["ok"]:
            server.terminate()
            break
    report["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save()
    print(json.dumps(report, indent=2, default=str)[:20000])


if __name__ == "__main__":
    main()
