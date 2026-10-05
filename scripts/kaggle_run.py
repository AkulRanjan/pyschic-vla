"""Run a job from ``kaggle/<job>/`` on a Kaggle GPU and download its outputs.

    python scripts/kaggle_run.py gpu_check                  # push, wait, download
    python scripts/kaggle_run.py gpu_check --no-wait        # push only
    python scripts/kaggle_run.py gpu_check --fetch          # download the last run's outputs
    python scripts/kaggle_run.py gpu_check --machine NvidiaL4

The job folder holds ``<job>.py`` (a Kaggle script kernel). This script writes the
``kernel-metadata.json`` (private, GPU, internet on) into a temporary copy, pushes it as
``<your user>/thoughtzero-<job>``, polls until it finishes, and downloads /kaggle/working
to ``results/kaggle/<job>/``. Uses KAGGLE_API_TOKEN from ``.env``; never prints it.

Kaggle *secrets* (e.g. the Jev key) can't be attached through the API: attach them once
in the notebook's web UI (Add-ons -> Secrets) and they stay attached for later pushes.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
TERMINAL = {"complete", "error", "cancelacknowledged", "cancelrequested"}


class KaggleError(RuntimeError):
    pass


def kaggle(*args: str, attempts: int = 5) -> str:
    """Run the Kaggle CLI. Retries with backoff: the API intermittently rejects calls
    (reported as "Authentication required") when they come in quick succession."""
    exe = Path(sys.executable).with_name("kaggle.exe" if os.name == "nt" else "kaggle")
    # Use only KAGGLE_API_TOKEN: an older ~/.kaggle/kaggle.json key takes precedence over it
    # and can be read-only ("Authentication required" on push). An empty config dir hides it.
    config_dir = ROOT / ".cache" / "kaggle"
    config_dir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "KAGGLE_CONFIG_DIR": str(config_dir)}
    out = ""
    for attempt in range(attempts):
        r = subprocess.run([str(exe), *args], capture_output=True, text=True, env=env)
        out = (r.stdout + r.stderr).strip()
        if r.returncode == 0:
            return out
        if attempt < attempts - 1:
            time.sleep(min(30 * 2**attempt, 300))
    raise KaggleError(f"kaggle {' '.join(args[:2])} failed: {out[-1500:]}")


def username() -> str:
    if os.environ.get("KAGGLE_USERNAME"):
        return os.environ["KAGGLE_USERNAME"]
    for line in kaggle("config", "view").splitlines():
        if "username:" in line:
            return line.split(":", 1)[1].strip()
    raise SystemExit("could not read the Kaggle username: set KAGGLE_USERNAME in .env")


def status(ref: str) -> str:
    """e.g. 'has status "KernelWorkerStatus.COMPLETE"' -> 'complete'."""
    text = kaggle("kernels", "status", ref)
    return text.rsplit(" ", 1)[-1].strip('"').rsplit(".", 1)[-1].lower()


def fetch(ref: str, job: str) -> Path:
    dest = ROOT / "results" / "kaggle" / job
    dest.mkdir(parents=True, exist_ok=True)
    kaggle("kernels", "output", ref, "-p", str(dest), "--force")
    return dest


def main() -> int:
    load_dotenv(ROOT / ".env")
    if not os.environ.get("KAGGLE_API_TOKEN"):
        raise SystemExit("KAGGLE_API_TOKEN is not set in .env")
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("job", help="folder under kaggle/, e.g. gpu_check")
    parser.add_argument(
        "--machine", default="NvidiaTeslaT4", help="NvidiaTeslaT4 (T4 x2) | NvidiaL4"
    )
    parser.add_argument("--no-wait", action="store_true")
    parser.add_argument("--fetch", action="store_true", help="only download the last outputs")
    parser.add_argument("--attach", action="store_true", help="wait for the running job, no push")
    parser.add_argument("--timeout-min", type=int, default=120, help="stop waiting after this")
    args = parser.parse_args()

    job_dir = ROOT / "kaggle" / args.job
    if not (job_dir / f"{args.job}.py").exists():
        raise SystemExit(f"no kaggle/{args.job}/{args.job}.py")
    slug = f"thoughtzero-{args.job.replace('_', '-')}"
    ref = f"{username()}/{slug}"

    if args.fetch:
        print(f"outputs -> {fetch(ref, args.job)}")
        return 0
    if not args.attach:
        push(ref, slug, args.job, job_dir, args.machine)
    print(f"https://www.kaggle.com/code/{ref}")
    if args.no_wait:
        return 0

    t0 = time.time()
    state = "queued"
    while time.time() - t0 < args.timeout_min * 60:
        time.sleep(60)
        try:
            state = status(ref)
        except KaggleError as e:  # a transient rejection: keep waiting
            print(f"status check failed, will retry: {str(e)[:120]}", flush=True)
            continue
        print(f"[{(time.time() - t0) / 60:5.1f} min] {state}", flush=True)
        if state in TERMINAL:
            break
    dest = fetch(ref, args.job)
    print(f"final status: {state}; outputs -> {dest}")
    return 0 if state == "complete" else 1


def push(ref: str, slug: str, job: str, job_dir: Path, machine: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copytree(job_dir, tmp, dirs_exist_ok=True)
        meta = {
            "id": ref,
            "title": slug,
            "code_file": f"{job}.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "true",
            "enable_internet": "true",
            "machine_shape": machine,
            "dataset_sources": [],
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": [],
        }
        (Path(tmp) / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
        print(kaggle("kernels", "push", "-p", tmp))


if __name__ == "__main__":
    raise SystemExit(main())
