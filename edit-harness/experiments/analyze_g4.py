"""analyze_g4.py — the AlphaEdit causal test (G4).

Compares a matched ROME run and AlphaEdit run (same model/layer/seed => identical
edits, probes, and COS matrix) by binning edit-probe pairs into COS quartiles and
measuring mean collateral damage under each editor.

Causal prediction of the key-geometry mechanism:
  AlphaEdit projects the edit off the preserved-key subspace, so it should
  PROTECT high-cosine probes disproportionately. I.e. the damage REDUCTION
  (ROME − AlphaEdit) should be concentrated in the top-cosine quartile and small
  in the bottom quartile. A large reduction_ratio (top/bottom) is causal evidence
  that the ROME damage was driven by key-cosine alignment (not a third variable).

Usage:
  python analyze_g4.py --rome results/matrices/gate_..._rome_..._L8_s0.npz \
                       --alpha results/matrices/g4_..._alpha_..._L8_s0.npz \
                       --known --edit_ok --out results/G4_L8.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rome", required=True)
    ap.add_argument("--alpha", required=True)
    ap.add_argument("--known", action="store_true")
    ap.add_argument("--edit_ok", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dr = np.load(args.rome)
    da = np.load(args.alpha)
    COS = dr["COS"].astype(float)                     # base-model geometry, identical across editors
    Dr = dr["damage_logit"].astype(float)
    Da = da["damage_logit"].astype(float)
    if not (COS.shape == Dr.shape == Da.shape):
        raise SystemExit(f"shape mismatch {COS.shape}/{Dr.shape}/{Da.shape} — "
                         "rome and alpha must share model/layer/seed")
    # SHARED masks so ROME and AlphaEdit are compared on the SAME edit-probe pairs.
    # edit_ok is editor-specific -> require BOTH editors to have succeeded on that edit.
    row = np.ones(COS.shape[0], bool)
    if args.edit_ok and "edit_ok" in dr.files and "edit_ok" in da.files:
        row = (dr["edit_ok"].astype(float) > 0.5) & (da["edit_ok"].astype(float) > 0.5)
    col = np.ones(COS.shape[1], bool)
    if args.known and "pre_p" in dr.files:            # pre_p is base-model -> identical across editors
        c = dr["pre_p"].astype(float) > 0.05
        if c.sum() >= 5:
            col = c
    cos = COS[row][:, col].reshape(-1)
    dmg_r = Dr[row][:, col].reshape(-1)
    dmg_a = Da[row][:, col].reshape(-1)
    if cos.size < 20:
        raise SystemExit("too few shared pairs after filtering")
    qs = np.quantile(cos, [0.25, 0.5, 0.75])
    bins = np.digitize(cos, qs)  # 0..3 low->high cosine
    rows = []
    for q in range(4):
        m = bins == q
        rome_d = float(dmg_r[m].mean())
        alpha_d = float(dmg_a[m].mean())
        rows.append({
            "cosine_quartile": ["Q1(low)", "Q2", "Q3", "Q4(high)"][q],
            "n_pairs": int(m.sum()),
            "mean_cos": round(float(cos[m].mean()), 4),
            "rome_mean_damage": round(rome_d, 5),
            "alpha_mean_damage": round(alpha_d, 5),
            "damage_reduction": round(rome_d - alpha_d, 5),
            "protection_ratio": round(rome_d / alpha_d, 3) if abs(alpha_d) > 1e-6 else None,
        })
    red_top = rows[3]["damage_reduction"]
    red_bot = rows[0]["damage_reduction"]
    ratio = round(red_top / red_bot, 3) if abs(red_bot) > 1e-6 else None
    verdict = (
        "PASS — AlphaEdit protection concentrated in high-cosine probes (causal support)"
        if (red_top > 0 and (red_bot <= 0 or (ratio is not None and ratio >= 2.0)))
        else "WEAK — protection not cosine-concentrated; geometry account not causally supported"
    )
    res = {
        "rome": os.path.basename(args.rome), "alpha": os.path.basename(args.alpha),
        "filters": {"known": args.known, "edit_ok": args.edit_ok},
        "by_cosine_quartile": rows,
        "reduction_top_vs_bottom_ratio": ratio,
        "VERDICT": verdict,
    }
    print(json.dumps(res, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(res, open(args.out, "w"), indent=2)
        print(f"[g4] wrote {args.out}")


if __name__ == "__main__":
    main()
