"""cp_kg0_bootstrap.py (kg0) — CP-Edit KG-0 CPU bootstrap kill-gate.

Split-conformal, 4 scores, pooled 3 seeds x ~200 edits per ROME layer. Evaluates
the KG-0 kill-gate thresholds MECHANICALLY (booleans in the JSON).

  (a) bootstrap-mean coverage within MC null band at every layer x score;
  (b) SxC >=10% tighter than marginal at L8,L10,L12 (L14 exempt);
  (c) full ordering SxC<keycos<NG<marginal in >=75% of splits at L8-L12.

CPU only. 0 GPU, 0 downloads. Writes results/cpedit/CP_KG0_bootstrap.json.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cp_edit import io, conformal

OUT = os.path.abspath(os.path.join(io.HERE, "..", "..", "results", "cpedit", "CP_KG0_bootstrap.json"))


def _hash(arr):
    return hashlib.sha256(np.ascontiguousarray(arr, dtype=np.float64).tobytes()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--n_outer", type=int, default=2000)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    t0 = time.time()

    layers = {}
    for L in io.LAYERS:
        d = io.load_layer("rome", L)
        seeds_ncal = [int((d["seed_labels"] == i).sum()) for i in np.unique(d["seed_labels"])]
        bs = conformal.bootstrap_cp(d["y"], d["scores"], d["seed_labels"],
                                    io.SCORE_ORDER, io.NORMALIZED, B=args.B)
        nb = conformal.mc_null_band(bs["n_cal"], bs["n_test"], seeds_ncal,
                                    B=args.B, n_outer=args.n_outer)
        # per-score cell: coverage in band?
        per_score = {}
        cov_in_band_all = True
        for s in io.SCORE_ORDER:
            ps = dict(bs["per_score"][s])
            cov = ps["mean_coverage"]
            in_band = bool(nb["band_lo"] <= cov <= nb["band_hi"])
            cov_in_band_all = cov_in_band_all and in_band
            ps["coverage_in_mc_band"] = in_band
            ps["cov_arr_hash"] = _hash(bs["_arrays"]["cov"][s])
            ps["width_arr_hash"] = _hash(bs["_arrays"]["wid"][s])
            per_score[s] = ps
        sxc_tighter = per_score["SxC"]["pct_tighter_than_marginal"]
        layers[str(L)] = {
            "layer": L, "n_cal": bs["n_cal"], "n_test": bs["n_test"],
            "n_pooled": int(d["y"].shape[0]),
            "mc_null_band": nb,
            "ordering_fraction": bs["ordering_fraction"],
            "sxc_pct_tighter_than_marginal": sxc_tighter,
            "per_score": per_score,
            "book": d["book"],
            # per-layer gate components
            "gate_a_all_cov_in_band": cov_in_band_all,
            "gate_b_sxc_tighter_ge_10pct": bool(sxc_tighter >= 0.10),
            "gate_c_ordering_ge_75pct": bool(bs["ordering_fraction"] >= 0.75),
        }

    # ---- assemble KG-0 kill-gate verdict (mechanical) ----
    kg_layers_bc = ["8", "10", "12"]  # L14 exempt from (b); ordering (c) at L8-L12
    fail_a = [f"L{L}:{s}" for L in layers for s in io.SCORE_ORDER
              if not layers[L]["per_score"][s]["coverage_in_mc_band"]]
    fail_b = [L for L in kg_layers_bc if not layers[L]["gate_b_sxc_tighter_ge_10pct"]]
    fail_c = [L for L in kg_layers_bc if not layers[L]["gate_c_ordering_ge_75pct"]]
    killed = bool(fail_a or fail_b or fail_c)
    verdict = {
        "gate_a_coverage_in_band": {"pass": not fail_a, "failing_cells": fail_a},
        "gate_b_sxc_ge_10pct_tighter_L8_L10_L12": {"pass": not fail_b, "failing_layers": fail_b},
        "gate_c_ordering_ge_75pct_L8_L10_L12": {"pass": not fail_c, "failing_layers": fail_c},
        "KG0_VERDICT": ("KILL — CP-Edit direction killed" if killed
                        else "PASS — KG-0 clears; authorizes E1/E2/E5"),
        "killed": killed,
    }
    out = {
        "experiment": "CP-Edit KG-0 CPU bootstrap kill-gate (ROME, split-conformal, 4 scores)",
        "rng_seed": conformal.RNG_SEED, "B": args.B, "n_outer": args.n_outer,
        "target_coverage": 0.90,
        "score_order": list(io.SCORE_ORDER),
        "layers": layers,
        "verdict": verdict,
        "runtime_s": round(time.time() - t0, 1),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tmp = args.out + ".tmp"; json.dump(out, open(tmp, "w"), indent=2); os.replace(tmp, args.out)
    print(json.dumps(verdict, indent=2))
    print(f"[kg0] wrote {args.out}  ({out['runtime_s']}s)")


if __name__ == "__main__":
    main()
