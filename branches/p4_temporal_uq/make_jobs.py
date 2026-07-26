"""make_jobs.py — emit fission-engine queue job specs for the P4 pilot.

Generates one per-dataset config (configs/ett_<name>.json) from the pilot
template and one matching queue job spec (fission-engine/queue/<id>.json) per
ETT dataset. Every job is marked ``gpu_required: true`` — the committee sweep
needs Ollama on the GPU — so a CPU-only worker will skip them.

The queue job schema mirrors edit-harness/queue conventions (a JSON spec whose
``cmd`` a serial runner executes, moving finished specs to done/ and failures to
failed/), extended with ``gpu_required`` and a ``created`` stamp.

Usage
-----
    python3 make_jobs.py 20260630_2359            # created-stamp is required
    python3 make_jobs.py 20260630_2359 --datasets ETTh1 ETTm1
    python3 make_jobs.py 20260630_2359 --backend mock   # for a no-GPU rehearsal

The created-stamp becomes part of each job id and result filename, so repeated
emissions never collide.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(HERE, "configs")
QUEUE_DIR = os.path.join(HERE, "fission-engine", "queue")
PILOT_CONFIG = os.path.join(CONFIG_DIR, "ett_pilot.json")

DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]


def load_pilot() -> Dict:
    with open(PILOT_CONFIG) as fh:
        return json.load(fh)


def make_config(pilot: Dict, dataset: str, stamp: str, backend: str) -> Dict:
    cfg = json.loads(json.dumps(pilot))  # deep copy
    cfg["id"] = f"p4_{dataset.lower()}_{stamp}"
    cfg["backend"] = backend
    cfg.setdefault("data", {})
    cfg["data"]["csv"] = f"data/{dataset}.csv"
    cfg["_derived_from"] = "configs/ett_pilot.json"
    cfg["_dataset"] = dataset
    return cfg


def make_job(cfg: Dict, stamp: str, backend: str) -> Dict:
    run_id = cfg["id"]
    config_rel = f"configs/{run_id}.json"
    result_rel = f"results/{run_id}.json"
    gpu_required = backend == "ollama"
    return {
        "id": run_id,
        "created": stamp,
        "gpu_required": gpu_required,
        "env": "dl",
        "cwd": HERE,
        "config": config_rel,
        "cmd": f"conda run -n dl python3 run_committee.py {config_rel}",
        "expect_result": result_rel,
        "dataset": cfg.get("_dataset"),
        "backend": backend,
        "notes": "P4 cross-architecture committee disagreement UQ on ETT; "
                 "needs Ollama on GPU. Kill-gate: cross-arch Spearman(disagreement,"
                 "|error|) > temperature-resampling null (bootstrap CI of delta > 0).",
    }


def emit(stamp: str, datasets: List[str], backend: str) -> List[str]:
    pilot = load_pilot()
    os.makedirs(CONFIG_DIR, exist_ok=True)
    os.makedirs(QUEUE_DIR, exist_ok=True)
    written: List[str] = []
    for ds in datasets:
        cfg = make_config(pilot, ds, stamp, backend)
        cfg_path = os.path.join(CONFIG_DIR, f"{cfg['id']}.json")
        with open(cfg_path, "w") as fh:
            json.dump(cfg, fh, indent=2)
        written.append(cfg_path)

        job = make_job(cfg, stamp, backend)
        job_path = os.path.join(QUEUE_DIR, f"{cfg['id']}.json")
        with open(job_path, "w") as fh:
            json.dump(job, fh, indent=2)
        written.append(job_path)
        print(f"[make_jobs] {ds}: config -> {os.path.relpath(cfg_path, HERE)} | "
              f"job -> {os.path.relpath(job_path, HERE)} (gpu_required={job['gpu_required']})")
    return written


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description="emit P4 fission-engine queue job specs")
    ap.add_argument("stamp", help="created-stamp (e.g. 20260630_2359), used in ids/results")
    ap.add_argument("--datasets", nargs="+", default=DATASETS,
                    help=f"ETT datasets to emit (default: {DATASETS})")
    ap.add_argument("--backend", choices=["ollama", "mock"], default="ollama",
                    help="backend baked into jobs (ollama=gpu_required)")
    args = ap.parse_args(argv)

    written = emit(args.stamp, args.datasets, args.backend)
    print(f"[make_jobs] emitted {len(written)} files for stamp={args.stamp} "
          f"backend={args.backend}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
