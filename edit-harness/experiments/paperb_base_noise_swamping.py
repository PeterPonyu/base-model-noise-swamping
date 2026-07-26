#!/usr/bin/env python3
"""Base-noise-swamping analysis for Paper B (quantization survival).

POST HOC / EXPLORATORY. None of the quantities here were preregistered. They are
reported as a mechanism account for a preregistered gate that FAILED: the rank-survival
threshold at 4-bit full-model does not hold at Llama-3.2-3B or Qwen-2.5-1.5B, and this
module characterises what replaces the signal when it fails.

Three quantities, all computed from the 27 per-cell raw NPZ files:

  (1) noise-to-signal ratio (NSR) = base_quant_noise_mean_abs / mean|D_fp32|.
      Numerator: mean absolute damage under the UNEDITED-quantized baseline arm, i.e.
      damage the quantizer injects with no edit installed. Denominator: mean absolute
      edit-attributable damage in FP32. NSR is an arm x cell property.

  (2) The additive-base-noise model  D_quant[i,j] ~= D_fp32[i,j] + base[j].
      Tested by decomposing the residual R = D_quant - D_fp32 into its probe-level
      (column) component and the remainder:
          colshare = var(mean_i R[i,j]) / var(R)          in [0,1]
          r_base   = Pearson( mean_i R[i,j] , base[j] )
      colshare near 1 with r_base near 1 means the quantizer's effect on measured damage
      is an edit-INDEPENDENT per-probe offset, not a distortion of the edit itself.

  (3) The consequence for the two rank estimands. A per-probe offset cancels exactly in a
      within-probe Spearman (probe identity held fixed) but shifts every probe column
      relative to the others in the flattened-grid Spearman. So the additive model
      predicts that flat rank collapses with NSR while within-probe rank is comparatively
      spared -- which is the observed dissociation.

Interpretation caveat, stated in the manuscript as well: under an additive-noise model the
attenuation of a rank correlation with NSR is expected analytically (classical attenuation
by measurement error). The substantive finding is not that attenuation occurs but that the
edit-independent additive model DESCRIBES these data this well, which is what licenses the
deployment rule (shrink the injected base noise -- quantize the edited layer only, or use
INT8 -- and the diagnostic is preserved, because the edit's own damage signal is untouched).

Usage:  python paperb_base_noise_swamping.py [--out PATH]
CPU only; runtime ~1 min (reads 27 x 25 MB NPZ).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from datetime import datetime, timezone

import numpy as np
from scipy.stats import spearmanr

REPO = pathlib.Path(__file__).resolve().parents[2]
RESULTS = REPO / "edit-harness" / "results" / "quant_survival"
AGG = RESULTS / "aggregate" / "quant_survival_repair_v1.json"

MODELS = {"llama1b": 12, "llama3b": 24, "qwen15b": 21}
EDITORS = ["rome", "memit", "alpha"]
ARMS = ["nf4dq_edited_layer", "nf4dq_full_model", "int8_edited_layer", "int8_full_model"]
SEEDS = [0, 1, 2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        type=str,
        default=str(RESULTS / "aggregate" / "base_noise_swamping_20260726.json"),
    )
    args = ap.parse_args()

    with AGG.open() as f:
        agg = json.load(f)
    canon = {(c["slug"], c["editor"]): c for c in agg["cells"]}

    per_cell = {}
    for slug, layer in MODELS.items():
        for editor in EDITORS:
            sig, resid = [], {arm: {"colshare": [], "r_base": []} for arm in ARMS}
            for s in SEEDS:
                path = RESULTS / f"{slug}_{editor}_L{layer}_s{s}" / "QS_phase1_raw.npz"
                with np.load(path) as z:
                    dfp = z["damage_fp32"]
                    sig.append(float(np.abs(dfp).mean()))
                    for arm in ARMS:
                        r = z[f"damage__{arm}"] - dfp
                        base = z[f"base__{arm}"]
                        colmean = r.mean(axis=0)
                        resid[arm]["colshare"].append(
                            float(colmean.var() / r.var()) if r.var() > 0 else np.nan
                        )
                        resid[arm]["r_base"].append(
                            float(np.corrcoef(colmean, base)[0, 1])
                            if base.std() > 0 and colmean.std() > 0
                            else np.nan
                        )
            mean_abs_dfp32 = float(np.mean(sig))
            for arm in ARMS:
                a = canon[(slug, editor)]["arms"][arm]
                noise = float(a["base_quant_noise_mean_abs"])
                per_cell[f"{slug}__{editor}__{arm}"] = {
                    "slug": slug,
                    "editor": editor,
                    "arm": arm,
                    "base_quant_noise_mean_abs": noise,
                    "mean_abs_damage_fp32": mean_abs_dfp32,
                    "noise_to_signal_ratio": noise / mean_abs_dfp32,
                    "flat_rank_survival": float(a["flat_rank"]["point"]),
                    "within_probe_rank_survival": float(a["within_probe_rank"]["point"]),
                    "conditional_survival": float(
                        a["conditional_survival_given_fp32_worked"]["point"]
                    ),
                    "resid_probe_level_share": float(np.nanmean(resid[arm]["colshare"])),
                    "resid_colmean_vs_base_pearson": float(np.nanmean(resid[arm]["r_base"])),
                }

    V = list(per_cell.values())

    def sp(x, y):
        rho, p = spearmanr([v[x] for v in V], [v[y] for v in V])
        return {"rho": float(rho), "p": float(p), "n": len(V)}

    nf4fm = [v for v in V if v["arm"] == "nf4dq_full_model"]
    rho_nf4fm, p_nf4fm = spearmanr(
        [v["noise_to_signal_ratio"] for v in nf4fm], [v["flat_rank_survival"] for v in nf4fm]
    )

    payload = {
        "module": "paperb_base_noise_swamping",
        "version": "1.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "POST HOC / EXPLORATORY -- not preregistered",
        "source_artefact": str(AGG.relative_to(REPO)),
        "definitions": {
            "noise_to_signal_ratio": (
                "base_quant_noise_mean_abs (unedited-quantized baseline damage) divided by "
                "mean|D_fp32| (edit-attributable damage in FP32)"
            ),
            "resid_probe_level_share": (
                "var(column means of D_quant - D_fp32) / var(D_quant - D_fp32); the fraction "
                "of the quantization residual that is an edit-independent per-probe offset"
            ),
            "resid_colmean_vs_base_pearson": (
                "Pearson r between the residual's per-probe column mean and the independently "
                "measured unedited-quantized baseline damage base[j]"
            ),
        },
        "correlations_over_36_arm_cells": {
            "nsr_vs_flat_rank": sp("noise_to_signal_ratio", "flat_rank_survival"),
            "nsr_vs_within_probe_rank": sp("noise_to_signal_ratio", "within_probe_rank_survival"),
            "nsr_vs_conditional_survival": sp("noise_to_signal_ratio", "conditional_survival"),
            "raw_noise_vs_flat_rank": sp("base_quant_noise_mean_abs", "flat_rank_survival"),
            "raw_noise_vs_conditional_survival": sp(
                "base_quant_noise_mean_abs", "conditional_survival"
            ),
        },
        "nf4_full_model_only": {
            "n": len(nf4fm),
            "nsr_vs_flat_rank": {"rho": float(rho_nf4fm), "p": float(p_nf4fm)},
        },
        "additive_model_fit_by_arm": {
            arm: {
                "resid_probe_level_share_mean": float(
                    np.mean([v["resid_probe_level_share"] for v in V if v["arm"] == arm])
                ),
                "resid_colmean_vs_base_pearson_mean": float(
                    np.mean([v["resid_colmean_vs_base_pearson"] for v in V if v["arm"] == arm])
                ),
            }
            for arm in ARMS
        },
        "per_cell": per_cell,
        "numpy_version": np.__version__,
    }

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(payload, f, indent=2)

    c = payload["correlations_over_36_arm_cells"]
    print("rho(NSR, flat rank)        = %+.4f  p=%.2e  n=%d" % (
        c["nsr_vs_flat_rank"]["rho"], c["nsr_vs_flat_rank"]["p"], c["nsr_vs_flat_rank"]["n"]))
    print("rho(NSR, within-probe)     = %+.4f  p=%.2e" % (
        c["nsr_vs_within_probe_rank"]["rho"], c["nsr_vs_within_probe_rank"]["p"]))
    print("rho(raw noise, flat rank)  = %+.4f  p=%.2e" % (
        c["raw_noise_vs_flat_rank"]["rho"], c["raw_noise_vs_flat_rank"]["p"]))
    print("rho(raw noise, cond surv)  = %+.4f  p=%.2e" % (
        c["raw_noise_vs_conditional_survival"]["rho"],
        c["raw_noise_vs_conditional_survival"]["p"]))
    print("NF4 full-model only: rho(NSR, flat) = %+.4f  p=%.2e" % (rho_nf4fm, p_nf4fm))
    for arm, d in payload["additive_model_fit_by_arm"].items():
        print("  %-22s probe-level share %.3f   r vs base %.3f" % (
            arm, d["resid_probe_level_share_mean"], d["resid_colmean_vs_base_pearson_mean"]))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
