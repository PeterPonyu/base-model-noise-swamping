"""make_jobs.py -- queue P3 IPI job specs (gpu_required=true) for run_p3_gpu.sh.

Job kinds (2026-07-10 Lane-B build adds `grid` and `defense`):
  legacy  : a whole-panel run_ipi run (the original pre-registered 9-model panel).
  grid    : a B4 EXTENDED lineage-grid run (grid.py tier x seed) via run_grid.py.
  defense : a B2 defense-table run (defense x tier x seed, off vs on) via run_defense.py.

Each job carries a CLI-set `created` stamp and its output path. Idempotent: a job whose
run_id is already queued, or whose output result already exists, is not re-added (marked
done). run_p3_gpu.sh consumes jobs/queue.json in order.

CLI:
    python make_jobs.py                                  # legacy mock+ollama panel jobs
    python make_jobs.py --kind grid --tier core --seeds 0,1,2 --n 30
    python make_jobs.py --kind defense --defenses spotlight,whitelist --tier core --seeds 0 --n 30
    python make_jobs.py --list                           # show current queue
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os

H = os.path.dirname(os.path.abspath(__file__))
JOBS = os.path.join(H, "jobs")
RESULTS = os.path.join(H, "results")
QUEUE = os.path.join(JOBS, "queue.json")


def _load_queue() -> list[dict]:
    if os.path.isfile(QUEUE):
        with open(QUEUE) as f:
            return json.load(f)
    return []


def _save_queue(q: list[dict]) -> None:
    os.makedirs(JOBS, exist_ok=True)
    with open(QUEUE, "w") as f:
        json.dump(q, f, indent=2)


def _stamp() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def make_job(backend: str, n: int, n_perm: int, metric: str, tag: str) -> dict:
    run_id = f"ipi_{tag}_{backend}_n{n}"
    return {
        "run_id": run_id, "kind": "legacy", "backend": backend,
        "gpu_required": backend == "ollama", "created": _stamp(),
        "cmd": ["python", "run_ipi.py", "--backend", backend, "--n", str(n),
                "--n_perm", str(n_perm), "--metric", metric, "--run_id", run_id]
        + (["--allow_gpu"] if backend == "ollama" else []),
        "out": os.path.join(RESULTS, f"{run_id}.json"), "status": "queued",
    }


def make_grid_job(tier: str, n: int, seed: int, n_perm: int,
                  allow_singleton_lineage_drop: bool = False) -> dict:
    run_id = f"ipi_grid_{tier}_n{n}_s{seed}"
    cmd = ["python", "run_grid.py", "--tier", tier, "--backend", "ollama",
           "--n", str(n), "--seed", str(seed), "--n_perm", str(n_perm),
           "--allow_gpu", "--run_id", run_id]
    # Default off: this is a deliberate, per-launch opt-in (never wired on automatically
    # for any tier, including lineage_arm) -- see PREREG-WAVE3-LINEAGE-DRAFT-20260711.md
    # sec 3a and run_ipi._gate_contrast. Absent, the job's cmd is byte-identical to before.
    if allow_singleton_lineage_drop:
        cmd.append("--allow_singleton_lineage_drop")
    return {
        "run_id": run_id, "kind": "grid", "backend": "ollama", "tier": tier, "seed": seed,
        "gpu_required": True, "created": _stamp(),
        "cmd": cmd,
        "out": os.path.join(RESULTS, f"{run_id}.json"), "status": "queued",
    }


def make_defense_job(defense: str, tier: str, n: int, seed: int, n_perm: int) -> dict:
    stem = f"defense_{defense}_{tier}_s{seed}"
    return {
        "run_id": stem, "kind": "defense", "backend": "ollama", "defense": defense,
        "tier": tier, "seed": seed, "gpu_required": True, "created": _stamp(),
        "cmd": ["python", "run_defense.py", "--defense", defense, "--tier", tier,
                "--backend", "ollama", "--n", str(n), "--seed", str(seed),
                "--n_perm", str(n_perm), "--allow_gpu", "--run_id", stem],
        # run_defense writes results/<stem>.json (the analysis); the per-arm ipi_*_{off,on}
        # results are written by run_ipi underneath.
        "out": os.path.join(RESULTS, f"{stem}.json"), "status": "queued",
    }


def _add(q: list[dict], job: dict, existing: set) -> str | None:
    if job["run_id"] in existing:
        return None
    if os.path.isfile(job["out"]):
        job["status"] = "done"
    q.append(job)
    existing.add(job["run_id"])
    return job["run_id"]


def _parse_int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip() != ""]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Queue P3 IPI jobs (legacy / grid / defense).")
    ap.add_argument("--kind", choices=["legacy", "grid", "defense"], default="legacy")
    # legacy
    ap.add_argument("--backend", choices=["mock", "ollama", "both"], default="both")
    ap.add_argument("--metric", choices=["pearson", "jaccard"], default="pearson")
    # shared
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--n_perm", type=int, default=1000)
    ap.add_argument("--tag", default=dt.datetime.now().strftime("%Y%m%d"))
    # grid / defense
    ap.add_argument("--tier", default="core")
    ap.add_argument("--seeds", default="0", help="comma list, e.g. 0,1,2")
    ap.add_argument("--allow_singleton_lineage_drop", action="store_true",
                    help="--kind grid only, default off: queue jobs with run_ipi's opt-in "
                         "singleton-lineage-drop relaxation set (see run_grid.py --help / "
                         "PREREG-WAVE3-LINEAGE-DRAFT-20260711.md sec 3a). Never applied "
                         "unless explicitly passed here.")
    ap.add_argument("--defenses", default="spotlight",
                    help="comma list for --kind defense, e.g. spotlight,whitelist,combined")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args(argv)

    if args.list:
        print(json.dumps(_load_queue(), indent=2))
        return 0

    q = _load_queue()
    existing = {j["run_id"] for j in q}
    added: list[str] = []

    if args.kind == "legacy":
        backends = ["mock", "ollama"] if args.backend == "both" else [args.backend]
        for b in backends:
            rid = _add(q, make_job(b, args.n, args.n_perm, args.metric, args.tag), existing)
            if rid:
                added.append(rid)
    elif args.kind == "grid":
        for s in _parse_int_list(args.seeds):
            rid = _add(q, make_grid_job(args.tier, args.n, s, args.n_perm,
                                        args.allow_singleton_lineage_drop), existing)
            if rid:
                added.append(rid)
    elif args.kind == "defense":
        for d in [x.strip() for x in args.defenses.split(",") if x.strip()]:
            for s in _parse_int_list(args.seeds):
                rid = _add(q, make_defense_job(d, args.tier, args.n, s, args.n_perm), existing)
                if rid:
                    added.append(rid)

    _save_queue(q)
    print(json.dumps({"queue_file": QUEUE, "kind": args.kind, "added": added,
                      "total_jobs": len(q),
                      "gpu_pending": [j["run_id"] for j in q
                                      if j.get("gpu_required") and j.get("status") != "done"]},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
