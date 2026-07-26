"""gradsim_baseline.py — G2 GradSim damage predictor (CPU-only, no GPU/no torch).

Reviewer baseline for the key-geometry headline: "isn't the ROME collateral just a
first-order GRADIENT-SIMILARITY effect, i.e. edits that push harder (large update)
onto probe-aligned directions?" We test that competitor on the SAME clean
within-probe partialled Spearman metric used for key-cosine.

WHY THIS IS CPU-ONLY (and faithful):
  A per-parameter GradSim = cos(∂L_edit/∂W , ∂L_probe/∂W) would need backprop → GPU.
  But for ROME the edit is a CLOSED-FORM rank-one update to down_proj:
        ΔW = (v − W k_e) k_eᵀ / c .
  Its first-order effect on probe j (whose pre-edit key is k_pj) is
        ΔW k_pj = (v − W k_e) · (k_e·k_pj) / c ,
  whose magnitude ∝ ‖v − W k_e‖ · (k_e·k_pj) = S_i · ‖k_e‖‖k_pj‖·COS[i,j].
  So the gradient/influence-similarity predictor factorizes into
        GRAD[i,j] = g_i · COS[i,j] ,
  the edit-strength scalar g_i (S = residual_norm ‖v−Wk‖, or the ENCORE norm-growth
  ‖ΔW‖) times the pre-edit key cosine. Every term is ALREADY in the killgate .npz —
  no model, no gradients, no GPU. This is the exact rank-one surrogate of GradSim
  that governs ROME collateral; we report it as the G2 baseline.

METRIC DISCIPLINE (project rule): the primary statistic is the SIGNED within-probe
partialled Spearman(predictor, damage) (holds probe identity fixed, correlates DOWN
each column across edits) — NOT AUROC. We always also report per-config MEAN signed
damage. Sign convention: higher GRAD ⇒ more predicted damage, matching COS.

Usage:
  python experiments/gradsim_baseline.py \
      results/matrices/gate_llama1b_rome_cf_L8_s*.npz \
      --metric logit --known --edit_ok --variant both \
      --out results/G2_gradsim_L8.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

# Reuse the AUDITED within-probe machinery so G2 is scored on the SAME metric as
# the key-cosine headline (analyze_matrices.py is the G1 gate analyzer).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_matrices import spearman, within_probe_rhos  # noqa: E402


def _apply_masks(d, COS, D, known, edit_ok_filter):
    """Replicate analyze_matrices.analyze_one masking EXACTLY.

    Also returns the row/col boolean masks so per-edit scalars (S, norm_growth)
    can be filtered identically to the [N,M] matrices.
    """
    rows = np.ones(COS.shape[0], bool)
    cols = np.ones(COS.shape[1], bool)
    if edit_ok_filter and "edit_ok" in d.files:
        rows = d["edit_ok"].astype(float) > 0.5
    if known and "pre_p" in d.files:
        c = d["pre_p"].astype(float) > 0.05
        if c.sum() >= 5:   # analyze_matrices only applies the known filter if >=5 survive
            cols = c
    return rows, cols, COS[np.ix_(rows, cols)], D[np.ix_(rows, cols)]


def _wp(pred, D):
    """Signed within-probe partialled Spearman summary for a predictor matrix."""
    rhos = within_probe_rhos(pred, D)                 # NaN for degenerate/constant columns
    n_nan = int(np.sum(~np.isfinite(rhos)))
    return {
        "within_probe_mean": (None if np.all(np.isnan(rhos))
                              else round(float(np.nanmean(rhos)), 4)),
        "within_probe_frac_positive": (None if np.all(np.isnan(rhos))
                                       else round(float(np.nanmean(rhos > 0)), 3)),
        "nan_column_count": n_nan,
        "n_columns": int(pred.shape[1]),
    }


def analyze_one(npz_path, metric, known, edit_ok_filter, variants):
    d = np.load(npz_path)
    COS = d["COS"].astype(float)                                              # [N,M]
    D = (d["damage_logit"] if metric == "logit" else d["damage_prob"]).astype(float)
    rows, cols, COSm, Dm = _apply_masks(d, COS, D, known, edit_ok_filter)

    out = {
        "npz": os.path.basename(npz_path),
        "shape": [int(COSm.shape[0]), int(COSm.shape[1])],
        "mean_signed_damage": round(float(np.nanmean(Dm)), 5),   # project rule: always report
        "keycos": _wp(COSm, Dm),                                 # reference headline predictor
        "gradsim": {},
    }

    # per-edit edit-strength scalars, masked to the surviving rows
    scalars = {}
    if "resid_norm" in d.files:
        s = d["resid_norm"].astype(float)
        if np.any(np.isfinite(s)):
            scalars["resid"] = s[rows]        # S = ‖v−Wk‖ (NaN for ft editor → skipped)
    if "norm_growth" in d.files:
        scalars["normgrowth"] = d["norm_growth"].astype(float)[rows]  # ‖ΔW‖ (ENCORE)

    for v in variants:
        if v not in scalars or not np.any(np.isfinite(scalars[v])):
            out["gradsim"][v] = {"note": f"predictor '{v}' unavailable (all-NaN/missing) — skipped"}
            continue
        g = scalars[v][:, None]                       # [n_rows,1]
        GRAD = g * COSm                               # first-order rank-one influence surrogate
        rep = _wp(GRAD, Dm)
        # does key-cosine beat this GradSim variant on the within-probe metric?
        km = out["keycos"]["within_probe_mean"]
        gm = rep["within_probe_mean"]
        rep["keycos_beats_gradsim"] = (None if (km is None or gm is None)
                                       else bool(abs(km) > abs(gm)))
        out["gradsim"][v] = rep
    return out


def _agg(per_seed, key_path):
    """mean±std across seeds of a nested within_probe_mean, skipping None."""
    vals = []
    for r in per_seed:
        node = r
        ok = True
        for k in key_path:
            if isinstance(node, dict) and k in node:
                node = node[k]
            else:
                ok = False
                break
        if ok and isinstance(node, (int, float)):
            vals.append(float(node))
    if not vals:
        return {"mean": None, "std": None, "n_seeds": 0}
    return {"mean": round(float(np.mean(vals)), 4),
            "std": round(float(np.std(vals)), 4), "n_seeds": len(vals)}


def main():
    ap = argparse.ArgumentParser(description="G2 GradSim (first-order ROME influence) baseline")
    ap.add_argument("npz", nargs="+", help="killgate .npz (globs ok); each expanded path = one seed")
    ap.add_argument("--metric", choices=["logit", "prob"], default="logit")
    ap.add_argument("--known", action="store_true", help="filter to probes base model knows (pre_p>0.05)")
    ap.add_argument("--edit_ok", action="store_true", help="drop failed edits (edit_ok==0)")
    ap.add_argument("--variant", choices=["resid", "normgrowth", "both"], default="both",
                    help="edit-strength scalar g_i: resid=‖v−Wk‖ (S), normgrowth=‖ΔW‖ (ENCORE)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    variants = ["resid", "normgrowth"] if args.variant == "both" else [args.variant]
    paths = sorted({p for pat in args.npz for p in glob.glob(pat)})
    if not paths:
        raise SystemExit("no .npz matched")
    per_seed = [analyze_one(p, args.metric, args.known, args.edit_ok, variants) for p in paths]

    agg = {
        "n_seeds": len(paths),
        "metric": args.metric,
        "filters": {"known": args.known, "edit_ok": args.edit_ok},
        "keycos_within_probe": _agg(per_seed, ["keycos", "within_probe_mean"]),
        "gradsim_within_probe": {v: _agg(per_seed, ["gradsim", v, "within_probe_mean"])
                                 for v in variants},
        "mean_signed_damage_across_seeds": _agg(per_seed, ["mean_signed_damage"]),
    }
    # headline verdict: does key-cosine beat every available GradSim variant?
    km = agg["keycos_within_probe"]["mean"]
    beats = []
    for v in variants:
        gm = agg["gradsim_within_probe"][v]["mean"]
        if km is not None and gm is not None:
            beats.append(abs(km) > abs(gm))
    agg["VERDICT"] = ("UNDETERMINED" if not beats else
                      ("key-cosine BEATS GradSim (headline survives G2 baseline)"
                       if all(beats) else
                       "GradSim MATCHES/BEATS key-cosine on at least one variant — reframe"))
    res = {"aggregate": agg, "per_seed": per_seed}
    print(json.dumps(res, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(res, open(args.out, "w"), indent=2)
        print(f"[gradsim] wrote {args.out}")


if __name__ == "__main__":
    main()
