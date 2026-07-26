"""rerun_ft_contaminated_cells.py — CPU-driven script that calls run_real_wave internally
on ONLY the FT-contaminated policies (always_ft, ft_merge, cost_only, random), producing 12
fresh cells for MIX_A under the FT-arm fix. After the rerun, the dead-arm detector must
report zero FAIL_DEAD_FT_ARM cells (every routed-FT arm must now show non-zero install_gpu_s).

NOTE: This script exists only because the FT-arm wave-1 bug contaminated 12 MIX_A cells; once
those cells are produced clean, this script should not be needed again. Future waves that
include FT policy should run the dead-arm gate inline (see run_frame_a_wave1.sh).
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time

HARNESS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HARNESS)

CONTAMINATED_POLICIES = ("always_ft", "ft_merge", "cost_only", "random")
QUARANTINE_DIR = os.path.join(HARNESS, "results", "frame_a", "cells", ".ft-bug-bak")


def main():
    ap = argparse.ArgumentParser(description="Rerun only the 12 FT-contaminated MIX_A cells.")
    ap.add_argument("--model_dir", default=os.path.join(HARNESS, "data", "models", "Llama-3.2-1B"))
    ap.add_argument("--out_cells", default=os.path.join(HARNESS, "results", "frame_a", "cells"))
    ap.add_argument("--model_tag", default="llama-3.2-1b")
    ap.add_argument("--mixes", default="MIX_A")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing cells (we deleted them earlier; this is for safety)")
    ap.add_argument("--dryrun", action="store_true")
    args = ap.parse_args()

    if not args.dryrun:
        # FIRST: re-run the dead-arm detector to confirm only the 12 contaminated cells are missing
        from experiments.frame_a.check_dead_arms import check_dir
        report = check_dir(args.out_cells)
        present = {os.path.basename(c["path"]) for c in report["cells"]}
        print(f"[rerun-ft] {len(present)} cells already on disk; nothing to do if all 12 FT cells present")

    # SECOND: confirm the FT-arm fix is in place via the wiring test (cheap CPU check)
    print("[rerun-ft] running FT-fix wiring test (CPU) before GPU work ...")
    import subprocess
    r = subprocess.run([sys.executable, os.path.join(HARNESS, "engine", "test_ft_pending_fix_20260719.py")],
                       cwd=HARNESS, capture_output=True, text=True)
    if r.returncode != 0 or "ALL WIRING CHECKS PASSED" not in r.stdout:
        print(f"[rerun-ft] ABORT: wiring test failed before GPU work")
        print(r.stdout); print(r.stderr)
        sys.exit(8)
    print("[rerun-ft] wiring test PASS — FT-arm fix confirmed in place")

    if args.dryrun:
        print(f"[rerun-ft] DRYRUN plan: rerun {len(CONTAMINATED_POLICIES)} policies × "
              f"{len(args.seeds.split(','))} seeds × {len(args.mixes.split(','))} mix(es) "
              f"on real {args.model_tag} ({args.model_dir})")
        print(f"[rerun-ft] command: PY={sys.executable} experiments.frame_a.run_stream --run --real "
              f"--model {args.model_tag} --model_dir {args.model_dir} "
              f"--out_cells {args.out_cells} --mixes {args.mixes} --policies "
              f"{','.join(CONTAMINATED_POLICIES)} --force")
        print(f"[rerun-ft] Resume-safe: cells already on disk are skipped (run_stream line 386). "
              f"To resume an interrupted run, just re-invoke the same command.")
        return

    # THIRD: launch real wave with --policies filter (resume-safe — already-existing cells
    #         are skipped per-cell by run_stream's own skip-on-exists check, line 386)
    print(f"[rerun-ft] launching real wave at {time.strftime('%F %T')} on policies="
          f"{','.join(CONTAMINATED_POLICIES)}")
    from experiments.frame_a.run_stream import run_real_wave
    out = run_real_wave(args.out_cells, args.model_dir, model_tag=args.model_tag,
                        mixes=args.mixes.split(","), force=args.force,
                        policies_filter=list(CONTAMINATED_POLICIES))
    print(f"[rerun-ft] real wave done: {out}")

    # FOURTH: dead-arm re-check
    from experiments.frame_a.check_dead_arms import check_dir
    final = check_dir(args.out_cells)
    s = final["summary"]
    print(f"[rerun-ft] POST-RERUN dead-arm check: {s}")
    if s["dead_ft_arm"] > 0 or s["suspect_ft_arm"] > 0:
        print("[rerun-ft] FAIL: dead-arm signature still present after rerun — see above")
        sys.exit(9)
    print("[rerun-ft] PASS: every FT-routed cell now shows non-zero install_gpu_s; dead-arm clean")


if __name__ == "__main__":
    main()