#!/usr/bin/env python3
"""
make_jobs.py — emit QUEUED job specs for the P2 pipeline.

Two job kinds per checkpoint:
  1. `gen`  (gpu_required=true)  — k=8 GSM8K sampling from the pre-RL checkpoint;
                                   produces samples/<ckpt>.json (input to run_diag).
  2. `diag` (gpu_required=false) — run_diag.py on that samples file -> results/.

These are SPECS ONLY.  This script does not run anything on the GPU; it writes
JSON job files (with a created-stamp) into queue/.  A serial runner (mirroring
edit-harness/queue/run_all.sh) executes them later, inside the patched `dl-rl`
clone for GPU jobs.  The GRPO training step is intentionally NOT queued here —
it depends on the dl-rl env being built + patched (SETUP.md) and is gated on the
pre-RL diagnostic clearing its kill-gate first.

Usage:
    python make_jobs.py                 # queue gen+diag for all 7 checkpoints
    python make_jobs.py --k 8 --n-problems 200 --seed 0
    python make_jobs.py --only Qwen2.5-1.5B
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

from grpo_config import CHECKPOINT_PANEL

HERE = os.path.dirname(os.path.abspath(__file__))
QUEUE = os.path.join(HERE, "queue")
SAMPLES_DIR = os.path.join(HERE, "samples")
RESULTS_DIR = os.path.join(HERE, "results")


def _stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ckpt_id(path: str) -> str:
    return os.path.basename(path.rstrip("/"))


def gen_job(ckpt_path: str, k: int, n_problems: int, seed: int,
            max_new_tokens: int) -> dict:
    cid = _ckpt_id(ckpt_path)
    samples_out = os.path.join("branches/p2_prerl_diag/samples", f"{cid}.json")
    return {
        "job_kind": "gen",
        "job_id": f"p2_gen_{cid}",
        "gpu_required": True,
        "created": _stamp(),
        "checkpoint_id": cid,
        "model_path": ckpt_path,
        "env": "dl-rl",   # patched clone; DO NOT run gen in shared dl
        "_note": (
            "k-sample GSM8K generation from the PRE-RL checkpoint (no training). "
            "Greedy=False; sample k independent CoT traces per problem, grade the "
            "final \\boxed{} against GSM8K gold, record {text,len,correct}."
        ),
        "spec": {
            "dataset": "openai/gsm8k",
            "dataset_config": "main",
            "split": "test",
            "n_problems": n_problems,
            "k": k,
            "temperature": 0.9,
            "top_p": 1.0,
            "max_new_tokens": max_new_tokens,
            "len_unit": "generated_tokens",   # `len` field = # generated tokens
            "grader": "boxed_gsm8k",
            "seed": seed,
            "out": samples_out,
        },
        # exact command the serial runner will execute (inside dl-rl):
        "cmd": (
            "env -u ALL_PROXY -u all_proxy conda run -n dl-rl python3 "
            "branches/p2_prerl_diag/sample_ckpt.py "
            f"--model {ckpt_path} --dataset openai/gsm8k --config main --split test "
            f"--n-problems {n_problems} --k {k} --temperature 0.9 --top-p 1.0 "
            f"--max-new-tokens {max_new_tokens} --seed {seed} --out {samples_out}"
        ),
        "_cmd_note": (
            "sample_ckpt.py is the GPU sampler (NOT in this branch yet — it is the "
            "queued work). It must import transformers/torch and therefore only "
            "runs in dl-rl. Its output JSON is the input contract of run_diag.py."
        ),
    }


def diag_job(ckpt_path: str, n_boot: int, seed: int) -> dict:
    cid = _ckpt_id(ckpt_path)
    samples_in = os.path.join("branches/p2_prerl_diag/samples", f"{cid}.json")
    return {
        "job_kind": "diag",
        "job_id": f"p2_diag_{cid}",
        "gpu_required": False,
        "created": _stamp(),
        "checkpoint_id": cid,
        "depends_on": f"p2_gen_{cid}",
        "env": "dl",   # numpy-only diagnostic; safe in shared dl
        "_note": "CPU diagnostic on the generated samples; numpy/stdlib only.",
        "cmd": (
            "conda run -n dl python3 branches/p2_prerl_diag/run_diag.py "
            f"{samples_in} --id {cid} --n-boot {n_boot} --seed {seed}"
        ),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=8, help="samples per problem")
    ap.add_argument("--n-problems", type=int, default=200)
    ap.add_argument("--max-new-tokens", type=int, default=640)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", default=None,
                    help="restrict to one checkpoint id (e.g. Qwen2.5-1.5B)")
    ap.add_argument("--queue-dir", default=QUEUE)
    args = ap.parse_args(argv)

    os.makedirs(args.queue_dir, exist_ok=True)
    panel = CHECKPOINT_PANEL
    if args.only:
        panel = [p for p in panel if _ckpt_id(p) == args.only]
        if not panel:
            ap.error(f"--only {args.only} matched no checkpoint in the panel")

    written = []
    for ckpt in panel:
        for job in (
            gen_job(ckpt, args.k, args.n_problems, args.seed, args.max_new_tokens),
            diag_job(ckpt, args.n_boot, args.seed),
        ):
            path = os.path.join(args.queue_dir, job["job_id"] + ".json")
            with open(path, "w") as fh:
                json.dump(job, fh, indent=2)
            written.append(path)
            gpu = "GPU" if job["gpu_required"] else "cpu"
            print(f"[make_jobs] queued [{gpu}] {job['job_id']} -> {path}")

    print(f"[make_jobs] wrote {len(written)} job spec(s) to {args.queue_dir}")
    print("[make_jobs] created-stamp:", _stamp())
    print("[make_jobs] NOTE: gen jobs need the dl-rl clone (SETUP.md); "
          "nothing was executed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
