"""e2_mondrian.py — CP-Edit E2 Mondrian conditional-coverage audit (coarse strata).

Coarse strata only (audit fix): 4 layers x SxC terciles (~100 cal/tercile). Plus
the pre-registered failure locus: L14 stratified by NORM-GROWTH terciles.

  Step 1: pooled per-layer calibration; conditional coverage in each SxC tercile.
  Step 2: L14 x NG terciles; pooled coverage in the top-NG tercile (prereg: UNDER-covers).
  Step 3: Mondrian recalibration within each stratum; coverage restored to ~0.90.

Deviations reported in pp with MC bands for n~100 test strata. ROME editor.
CPU only. 0 GPU, 0 downloads. Writes results/cpedit/CP_E2_mondrian_audit.json.
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cp_edit import io, conformal

OUT = os.path.abspath(os.path.join(io.HERE, "..", "..", "results", "cpedit", "CP_E2_mondrian_audit.json"))


_BAND_CACHE = {}  # keyed by (seeds_ncal tuple, B, n_outer): band depends only on split geometry


def strat_block(editor, L, strat_name, B, n_outer_band=400):
    """AUDIT FIX 2026-07-01: the MC null band is now built with the SAME procedure
    as the gated statistic — conformal.mc_null_band_mondrian simulates
    mondrian_bootstrap itself (B-split bootstrap-mean pooled-tercile coverage,
    pooled-cal q_hat, per-split terciles from an independent null stratifier).
    The earlier band (mc_null_band with B=1, n_cal = the tercile's ~N/6 points)
    simulated a single-split statistic and was several times too wide."""
    d = io.load_layer(editor, L)
    strat = d["scores"][strat_name]
    md = conformal.mondrian_bootstrap(d["y"], strat, d["seed_labels"], n_terciles=3, B=B)
    seeds_ncal = [int((d["seed_labels"] == s).sum()) for s in np.unique(d["seed_labels"])]
    key = (tuple(seeds_ncal), B, n_outer_band)
    if key not in _BAND_CACHE:
        _BAND_CACHE[key] = conformal.mc_null_band_mondrian(
            seeds_ncal, n_terciles=3, B=B, n_outer=n_outer_band)
    nb = _BAND_CACHE[key]
    bands = [{"tercile": pt["tercile"], "band_lo": pt["band_lo"],
              "band_hi": pt["band_hi"], "E_cov": pt["null_mean"]}
             for pt in nb["per_tercile"]]
    md["mc_band_meta"] = {k: nb[k] for k in
                          ("n_outer", "B", "band_level", "statistic", "note_qhat_clamp")}
    return md, bands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=1000)
    args = ap.parse_args()
    t0 = time.time()

    # --- Block 1: SxC-tercile partition for all 4 layers ---
    sxc_block = {}
    any_ge_5pp = False
    for L in io.LAYERS:
        md, bands = strat_block("rome", L, "SxC", args.B)
        for t, b in zip(md["terciles"], bands):
            t["mc_band_lo"] = b["band_lo"]; t["mc_band_hi"] = b["band_hi"]
            t["pooled_under_covers_vs_band"] = (
                bool(b["band_lo"] is not None and t["pooled_coverage"] < b["band_lo"]))
            t["pooled_over_covers_vs_band"] = (
                bool(b["band_hi"] is not None and t["pooled_coverage"] > b["band_hi"]))
            if abs(t["pooled_dev_pp"]) >= 5.0:
                any_ge_5pp = True
        sxc_block[str(L)] = md

    # --- Block 2: L14 x NG-tercile partition (pre-registered failure locus) ---
    md_ng, bands_ng = strat_block("rome", 14, "NG", args.B)
    for t, b in zip(md_ng["terciles"], bands_ng):
        t["mc_band_lo"] = b["band_lo"]; t["mc_band_hi"] = b["band_hi"]
        t["pooled_under_covers_vs_band"] = (
            bool(b["band_lo"] is not None and t["pooled_coverage"] < b["band_lo"]))
        t["pooled_over_covers_vs_band"] = (
            bool(b["band_hi"] is not None and t["pooled_coverage"] > b["band_hi"]))
    top_ng = md_ng["terciles"][2]  # tercile 2 = top NG
    prereg_pass = bool(top_ng["pooled_under_covers_vs_band"])
    # also NG-tercile deviations count toward the arm's 5pp survival
    for t in md_ng["terciles"]:
        if abs(t["pooled_dev_pp"]) >= 5.0:
            any_ge_5pp = True

    # --- E2 arm verdict ---
    arm_dies = bool(not any_ge_5pp)  # dies if ALL pooled deviations <5pp
    out = {
        "experiment": "CP-Edit E2 Mondrian conditional-coverage audit (coarse strata, ROME)",
        "rng_seed": conformal.RNG_SEED, "B": args.B, "target_coverage": 0.90,
        "band_fix_2026_07_01": (
            "The previously shipped MC bands were computed for the WRONG statistic: a "
            "single-split coverage (B=1) with a Mondrian-sized calibration set (~N/6), "
            "while the gated number is a B-split bootstrap-MEAN coverage under POOLED "
            "calibration (~N/2). Those bands (e.g. [0.7765, 0.9894] for the L14 top-NG "
            "stratum) were several times too wide, so real under-/over-coverage was "
            "silently missed and the prereg confirmation cleared its band by only 0.005 "
            "for the wrong reason. Bands here are recomputed with the matched procedure "
            "(conformal.mc_null_band_mondrian). All pooled_under/over_covers_vs_band "
            "flags are re-derived against the corrected bands."),
        "ng_stratifier_timing_note": (
            "The NG stratifier (norm_growth = ||delta_W|| at the edit layer) is NOT a "
            "pre-edit quantity: it requires the edit to be computed (post-edit weights; "
            "probe-outcome-free and closed-form predictable for rank-one ROME). The L14 "
            "x NG-tercile Mondrian arm therefore ships under the 'probe-outcome-free / "
            "computable at edit time' banner, not 'pre-edit-only'."),
        "note_marginal_nonconformity": ("conditional coverage measured with the marginal "
                                        "nonconformity (r=y, U=q_hat) so strata coverage is a clean "
                                        "statement about the certificate itself."),
        "block1_SxC_terciles": sxc_block,
        "block2_L14_NG_terciles": md_ng,
        "preregistered_L14_highNG": {
            "stratum": "L14 x top-NG-tercile",
            "pooled_coverage": top_ng["pooled_coverage"],
            "pooled_dev_pp": top_ng["pooled_dev_pp"],
            "mc_band": [top_ng.get("mc_band_lo"), top_ng.get("mc_band_hi")],
            "mondrian_coverage": top_ng["mondrian_coverage"],
            "mondrian_dev_pp": top_ng["mondrian_dev_pp"],
            "PREREG_PREDICTION": "pooled calibration UNDER-covers in L14 x high-NG",
            "prereg_pass": prereg_pass,
            "verbatim_result": ("CONFIRMED: pooled under-covers below MC band" if prereg_pass
                                else "MISS: pooled coverage NOT below MC band in L14 x high-NG "
                                     "(reported verbatim per prereg; does not kill arm)"),
        },
        "e2_arm_verdict": {
            "any_pooled_deviation_ge_5pp": any_ge_5pp,
            "arm_dies_all_dev_below_5pp": arm_dies,
            "verdict": ("MONDRIAN ARM CUT — pooled calibration already conditionally fine (<5pp)"
                        if arm_dies else
                        "MONDRIAN ARM SURVIVES — >=5pp conditional deviation exists to repair"),
        },
        "runtime_s": round(time.time() - t0, 1),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tmp = OUT + ".tmp"; json.dump(out, open(tmp, "w"), indent=2); os.replace(tmp, OUT)
    print(json.dumps({"prereg_L14_highNG": out["preregistered_L14_highNG"],
                      "arm": out["e2_arm_verdict"]}, indent=2))
    print(f"[e2] wrote {OUT}  ({out['runtime_s']}s)")


if __name__ == "__main__":
    main()
