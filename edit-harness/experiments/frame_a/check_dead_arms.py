"""check_dead_arms.py — Frame-A dead-arm detector.

The 2026-07-19 FT-arm bug was a silent no-op: routes flowed into FT, the gate never fired,
the merge never ran, and Q landed at the 0.3*A_loc floor. This script scans per-cell
JSON outputs and FLAGS any cell where:

  (a) the policy routed >= K=50 updates to FT (the merge interval), BUT
  (b) cost.install_gpu_s == 0.0 for the FT arm (i.e. no merge ever ran),

which is the exact signature of the original bug. The aggregate verdict step REFUSES
to publish MIX_A as INCOMPLETE→clean until every cell passes this gate.

CPU-only. Reads cell_llama-3.2-1b_real_MIX_A_<policy>_s<seed>.json from results/frame_a/cells/.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

HARNESS = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT = os.path.join(HARNESS, "..", "..", "results", "frame_a", "cells")
FT_MERGE_INTERVAL_K = 50   # must match C.FT_MERGE_INTERVAL_K; kept literal for check_dead_arms.py standalone
DEAD_ARM_Q_FLOOR = 0.300   # the exact value A_cum=0 + A_loc=1 + A_upd=0 lands on; if you change the
                            # composite weights in metrics this number will move — the check
                            # still works because the FT signature is "0 merges happened".


def _load_cell(path: str) -> Dict[str, Any]:
    try:
        return json.load(open(path))
    except Exception as e:
        return {"_load_error": str(e)}


def _ft_routing(cell: Dict[str, Any]) -> int:
    """How many updates this cell routed to the FT arm."""
    return int(cell.get("routing", {}).get("arm_counts", {}).get("ft", 0))


def _ft_cost(cell: Dict[str, Any]) -> float:
    """Total GPU-seconds the FT arm spent installing (real GPU path)."""
    routing = cell.get("routing", {})
    return float(routing.get("install_gpu_s_by_arm", {}).get("ft", 0.0))


def _ft_quality(cell: Dict[str, Any]) -> float:
    """Cell-level Q from the scored output."""
    return float(cell.get("quality", {}).get("Q", 1.0))


def check_one(path: str) -> Dict[str, Any]:
    cell = _load_cell(path)
    if "_load_error" in cell:
        return {"path": path, "status": "LOAD_ERROR", "reason": cell["_load_error"]}

    name = os.path.basename(path).replace("cell_llama-3.2-1b_real_MIX_A_", "").replace(".json", "")
    policy, seed = name.rsplit("_s", 1)
    seed = int(seed)

    ft_n = _ft_routing(cell)
    ft_cost = _ft_cost(cell)
    q = _ft_quality(cell)

    issues = []
    status = "PASS"

    # (a) Policy routes >= K updates to FT but cost is 0 → exact signature of the original bug
    if ft_n >= FT_MERGE_INTERVAL_K and ft_cost == 0.0:
        issues.append(f"DEAD_FT_ARM: routed {ft_n} updates to FT but install_gpu_s=0.0 (no merge ran)")
        status = "FAIL_DEAD_FT_ARM"
    elif ft_n > 0 and ft_cost == 0.0:
        issues.append(f"SUSPECT_FT_ARM: routed {ft_n} (< K={FT_MERGE_INTERVAL_K}) updates to FT but cost=0.0 — should have at least one merge after K updates, not before")
        status = "FAIL_SUSPECT_FT_ARM"

    # (b) A Q at the 0.300 floor with non-zero routing to FT is also a red flag (could be
    #     a different dead-arm signature: routed, merged, but the merge did nothing)
    if q == DEAD_ARM_Q_FLOOR and ft_n > 0:
        issues.append(f"DEAD_FT_Q_FLOOR: routed {ft_n} updates to FT but cell Q={q} (the 0.300 floor) — same signature as the original silent no-op")
        status = "FAIL_DEAD_FT_ARM"

    return {
        "path": path, "policy": policy, "seed": seed,
        "ft_routed": ft_n, "ft_install_gpu_s": ft_cost, "Q": q,
        "status": status, "issues": issues,
    }


def check_dir(root: str = DEFAULT_ROOT) -> Dict[str, Any]:
    """Scan every cell_llama-3.2-1b_real_MIX_A_*.json in root. Returns aggregate + per-cell list."""
    if not os.path.isdir(root):
        return {"root": root, "error": f"directory not found", "cells": [], "summary": {}}

    per_cell: List[Dict[str, Any]] = []
    for name in sorted(os.listdir(root)):
        if not name.startswith("cell_llama-3.2-1b_real_MIX_A_") or not name.endswith(".json"):
            continue
        # skip the *.INVALID- quarantined ones (those are known-bad from earlier contamination)
        if ".INVALID" in name:
            continue
        per_cell.append(check_one(os.path.join(root, name)))

    summary = {
        "n_cells_scanned": len(per_cell),
        "n_fail": sum(1 for c in per_cell if not c["status"].startswith("PASS")),
        "n_pass": sum(1 for c in per_cell if c["status"] == "PASS"),
        "dead_ft_arm": sum(1 for c in per_cell if c["status"] == "FAIL_DEAD_FT_ARM"),
        "suspect_ft_arm": sum(1 for c in per_cell if c["status"] == "FAIL_SUSPECT_FT_ARM"),
        "load_errors": sum(1 for c in per_cell if c["status"] == "LOAD_ERROR"),
    }
    return {"root": root, "cells": per_cell, "summary": summary}


def selftest():
    """Construct synthetic good and bad cells, verify the detector flags exactly the right ones."""
    import tempfile
    # A passing cell: routed >K to FT AND paid GPU cost (≥1 merge ran).
    good = {"routing": {"arm_counts": {"ft": 60, "edit": 30}, "install_gpu_s_by_arm": {"ft": 4.5}},
            "quality": {"Q": 0.83}}
    # A dead-FT cell: routed >>K to FT but ZERO GPU cost — the original 2026-07-19 bug signature.
    bad_dead = {"routing": {"arm_counts": {"ft": 60}, "install_gpu_s_by_arm": {"ft": 0.0}},
                "quality": {"Q": 0.300}}
    # A dead-FT-by-floor cell: routed >>K AND paid cost but cell Q still at the 0.300 floor.
    bad_floor = {"routing": {"arm_counts": {"ft": 60}, "install_gpu_s_by_arm": {"ft": 12.0}},
                 "quality": {"Q": 0.300}}
    # A SUSPECT cell: routed 46 (< K=50) to FT with no cost — flagged as suspicious but not the
    # same severity as a true dead-FT arm.
    suspect = {"routing": {"arm_counts": {"ft": 46, "edit": 100}, "install_gpu_s_by_arm": {"ft": 0.0}},
               "quality": {"Q": 0.78}}
    with tempfile.TemporaryDirectory() as td:
        open(os.path.join(td, "cell_llama-3.2-1b_real_MIX_A_always_ft_s0.json"), "w").write(json.dumps(good))
        open(os.path.join(td, "cell_llama-3.2-1b_real_MIX_A_always_ft_s1.json"), "w").write(json.dumps(bad_dead))
        open(os.path.join(td, "cell_llama-3.2-1b_real_MIX_A_always_ft_s2.json"), "w").write(json.dumps(bad_floor))
        open(os.path.join(td, "cell_llama-3.2-1b_real_MIX_A_cost_only_s0.json"), "w").write(json.dumps(suspect))
        report = check_dir(td)
        s = report["summary"]
        assert s["n_cells_scanned"] == 4
        assert s["n_pass"] == 1, f"expected 1 pass, got {s['n_pass']}"
        assert s["dead_ft_arm"] == 2, f"expected 2 dead-ft, got {s['dead_ft_arm']}"
        assert s["suspect_ft_arm"] == 1, f"expected 1 suspect, got {s['suspect_ft_arm']}"
    print(f"[dead-arm selftest] PASS: {s}")


def main():
    ap = argparse.ArgumentParser(description="Frame-A dead-arm detector.")
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--refuse_verdict_if_fail", action="store_true",
                    help="Exit non-zero (and print a refusal) if any cell fails the dead-arm check.")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    report = check_dir(args.root)
    s = report["summary"]
    print(f"[dead-arm] scanned {s['n_cells_scanned']} cells in {report['root']}")
    print(f"[dead-arm] {s['n_pass']} PASS | {s['dead_ft_arm']} DEAD_FT_ARM | "
          f"{s['suspect_ft_arm']} SUSPECT_FT_ARM | {s['load_errors']} LOAD_ERROR")
    if s["n_fail"] > 0:
        print("\n[dead-arm] FAILED CELLS:")
        for c in report["cells"]:
            if not c["status"].startswith("PASS"):
                print(f"  - {c['policy']}_s{c['seed']}: {c['status']}")
                for issue in c["issues"]:
                    print(f"      {issue}")

    if args.refuse_verdict_if_fail and s["n_fail"] > 0:
        print("\n[dead-arm] REFUSING verdict: dead-arm signature detected — re-run contaminated cells first")
        sys.exit(2)


if __name__ == "__main__":
    main()