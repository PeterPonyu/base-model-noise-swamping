"""reanalyze_cleaned.py — CPU re-analysis of saved P4 runs through the
plausibility-guarded calibration pipeline.

The raw decoded forecasts were never persisted (run_committee.py discards the
per-model matrix), so corrupted windows CANNOT be repaired offline — but they
CAN be dropped honestly. This script re-runs calibration_report (with the new
plausible_mask guard) on the per-window scalars saved in each result JSON and
writes a cleaned side-by-side, answering: does the disagreement->error null
survive once the finite-but-astronomical decode artifacts are removed?

Pure CPU / numpy+scipy on saved JSONs. NO network, NO Ollama.

Usage:  python reanalyze_cleaned.py [--out results/p4_cleaned_reanalysis.json]
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from calibrate import calibration_report

RUNS = sorted(glob.glob(os.path.join(os.path.dirname(__file__) or ".", "results", "p4_*.json")))


def arr(per_window, key):
    return np.asarray([w.get(key, float("nan")) for w in per_window], dtype=np.float64)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results/p4_cleaned_reanalysis.json")
    ap.add_argument("--rel_cap", type=float, default=1e4)
    args = ap.parse_args()

    out = {"statistic": "calibration_report on saved per-window scalars after plausible_mask "
                        f"(rel_cap={args.rel_cap:g}); raw decodes not on disk, corrupt windows "
                        "dropped not repaired",
           "runs": {}}
    for path in RUNS:
        base = os.path.basename(path)
        if "cleaned_reanalysis" in base:
            continue
        d = json.load(open(path))
        pw = d.get("per_window")
        if not pw:
            continue
        cross, resamp, err = (arr(pw, "cross_disagreement"),
                              arr(pw, "resampling_disagreement"),
                              arr(pw, "signed_error"))
        rep = calibration_report(cross, resamp, err, rel_cap=args.rel_cap)
        orig = d.get("calibration", d.get("report", {}))
        cmp_ = rep["comparison"]
        out["runs"][base] = {
            "backend": d.get("config", {}).get("backend", d.get("backend", "?")),
            "dataset": d.get("config", {}).get("dataset", d.get("dataset", "?")),
            "n_windows_total": rep["n_windows_total"],
            "n_dropped_implausible": rep["n_windows_dropped_implausible"],
            "cleaned": {
                "spearman_cross_rho": rep["spearman_cross"]["rho"],
                "spearman_cross_p": rep["spearman_cross"]["pvalue"],
                "spearman_resampling_rho": rep["spearman_resampling"]["rho"],
                "picp_090": rep["picp_width_cross"]["picp"],
                "mean_width": rep["picp_width_cross"]["mean_width"],
                "crps": rep["crps_cross"]["crps"],
                "delta_point": cmp_["delta_point"],
                "delta_ci": [cmp_["delta_ci"]["lo95"], cmp_["delta_ci"]["hi95"]],
                "gate_pass": cmp_["gate_pass"],
            },
            "original_gate_pass": (orig.get("comparison", {}) or {}).get("gate_pass"),
        }
        r = out["runs"][base]
        print(f"{base}: backend={r['backend']} dropped={r['n_dropped_implausible']}/"
              f"{r['n_windows_total']} rho={r['cleaned']['spearman_cross_rho']:.3f} "
              f"(p={r['cleaned']['spearman_cross_p']:.3f}) PICP={r['cleaned']['picp_090']:.3f} "
              f"CRPS={r['cleaned']['crps']:.4g} gate={r['cleaned']['gate_pass']}")

    json.dump(out, open(args.out, "w"), indent=2)
    print(f"[reanalyze] wrote {args.out}")


if __name__ == "__main__":
    main()
