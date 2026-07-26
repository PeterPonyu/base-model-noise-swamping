#!/usr/bin/env python3
"""
run_diag.py — entrypoint: consume a pre-generated per-checkpoint samples JSON,
compute the overthinking length-bias diagnostic, write results/<id>.json.

This is the CPU half of P2.  It does NOT generate samples and does NOT touch the
GPU / trl / unsloth.  The k=8 GSM8K sampling per checkpoint that PRODUCES the
input JSON is a QUEUED GPU job — its spec/command lives in make_jobs.py and is
run only inside the `dl-rl` clone (see SETUP.md), never from here.

Input JSON shape (produced by the queued sampler):
    {"checkpoint": "Qwen2.5-1.5B",
     "problems": [
        {"problem": "gsm8k/train/17",
         "samples": [ {"text": "...", "len": 214, "correct": false}, ... x8 ]},
        ...]}

Usage:
    python run_diag.py samples/Qwen2.5-1.5B.json
    python run_diag.py samples/Qwen2.5-1.5B.json --id qwen15b --n-boot 5000
    # optional cross-checkpoint step once >=3 result files + a gap file exist:
    python run_diag.py --cross results/*.json --post-gap post_grpo_gap.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from datetime import datetime, timezone

import diagnostic as diag

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "results")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_one(
    samples_path: str,
    checkpoint_id: str | None,
    out_dir: str,
    n_boot: int,
    ci: float,
    seed: int,
) -> str:
    problems = diag.load_samples(samples_path)
    # checkpoint id: CLI override > "checkpoint" key in payload > filename stem
    cid = checkpoint_id
    if cid is None:
        try:
            with open(samples_path) as fh:
                head = json.load(fh)
            cid = head.get("checkpoint") if isinstance(head, dict) else None
        except Exception:
            cid = None
    if cid is None:
        cid = os.path.splitext(os.path.basename(samples_path))[0]

    record = diag.aggregate_checkpoint(
        problems, checkpoint_id=cid, n_boot=n_boot, ci=ci, seed=seed
    )
    record["_meta"] = {
        "input": os.path.abspath(samples_path),
        "generated_at": _now(),
        "tool": "run_diag.py",
        "note": "CPU diagnostic; sample generation is a separate queued GPU job.",
    }
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{cid}.json")
    with open(out_path, "w") as fh:
        json.dump(record, fh, indent=2)
    dw = record["D_within"]
    dp = record["D_pooled"]
    print(f"[run_diag] {cid}: "
          f"D_pooled={dp['point']:.3f} "
          f"[{dp['ci_lo']:.3f},{dp['ci_hi']:.3f}]  "
          f"D_within={dw['point']:.3f} "
          f"[{dw['ci_lo']:.3f},{dw['ci_hi']:.3f}]  "
          f"length_bias={record['length_bias_flag']}")
    print(f"[run_diag] wrote {out_path}")
    return out_path


def run_cross(result_globs: list[str], post_gap_path: str, out_dir: str,
              seed: int) -> str:
    """Cross-checkpoint payoff: correlate per-checkpoint pre-RL D_within against
    the measured post-GRPO overthinking gap."""
    paths: list[str] = []
    for g in result_globs:
        paths.extend(sorted(glob.glob(g)))
    pre_rl_D: dict[str, float] = {}
    for p in paths:
        with open(p) as fh:
            rec = json.load(fh)
        cid = rec.get("checkpoint_id", os.path.splitext(os.path.basename(p))[0])
        pre_rl_D[cid] = rec["D_within"]["point"]
    with open(post_gap_path) as fh:
        post_gap = json.load(fh)  # {checkpoint_id: gap}

    out = diag.cross_checkpoint_spearman(pre_rl_D, post_gap, seed=seed)
    out["_meta"] = {"generated_at": _now(), "n_result_files": len(paths)}
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "cross_checkpoint_spearman.json")
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[run_diag] cross-checkpoint: rho={out['rho']} "
          f"p_perm={out['p_perm']} n={out['n']}")
    if "warning" in out:
        print(f"[run_diag] WARNING: {out['warning']}")
    print(f"[run_diag] wrote {out_path}")
    return out_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("samples", nargs="?", help="per-checkpoint samples JSON")
    ap.add_argument("--id", default=None, help="checkpoint id override")
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--ci", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cross", nargs="+", default=None,
                    help="glob(s) of results/*.json for the cross-checkpoint step")
    ap.add_argument("--post-gap", default=None,
                    help="JSON {checkpoint_id: post_grpo_overthinking_gap}")
    args = ap.parse_args(argv)

    if args.cross is not None:
        if not args.post_gap:
            ap.error("--cross requires --post-gap")
        run_cross(args.cross, args.post_gap, args.out_dir, args.seed)
        return 0

    if not args.samples:
        ap.error("provide a samples JSON (or use --cross ... --post-gap ...)")
    run_one(args.samples, args.id, args.out_dir, args.n_boot, args.ci, args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
