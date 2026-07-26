#!/usr/bin/env python3
"""DESCRIPTIVE gain-law table (2026-07-15). Frozen protocol:
docs/plans/PREDICTIONS-GAIN-WAVE-2026-07-15.md.

Per RG bundle (auto-discovered results/merging/*_RG with rg_meta.json):
  rel_dose = (d_a . R_a) / ||R_a||^2   (dimensionless received cross-talk)
  gain     = median over positive-dose obs (g<=20, all seeds) of |drop| / rel_dose
  regime   = frac(drop < 0) over the same observations
plus per-bundle Spearman(rel_dose, drop) and n. Ordering test across bundles:
Spearman(gain, frac_drop_neg) — the frozen prediction is <= -0.7.

Formulas imported from merging_m0.py; nothing re-derived.
"""
import argparse
import glob
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from merging_m0 import _spearman, cross_term, load_rg, SCHEMA_VERSION

HARNESS = os.path.dirname(HERE)


def bundle_gain(rg_dir, gmax=20):
    per_seed, meas, meta = load_rg(rg_dir)
    obs_seed = meas["obs_seed"].astype(int); obs_g = meas["obs_g"].astype(int)
    obs_group = meas["obs_group"].astype(int); obs_edit = meas["obs_edit"].astype(int)
    obs_lp = meas["obs_logit_post"].astype(float)
    mem = defaultdict(list)
    for s, g, gr, ed in zip(meas["mem_seed"].astype(int), meas["mem_g"].astype(int),
                            meas["mem_group"].astype(int), meas["mem_edit"].astype(int)):
        mem[(s, g, gr)].append(ed)
    RD, DR = [], []
    for s in [int(x) for x in meta["seeds"]]:
        v = per_seed[s]
        K = v["K"].astype(float); R = v["R"].astype(float)
        denom = v["denom"].astype(float); ls = v["logit_solo"].astype(float)
        Rn2 = np.sum(R * R, axis=1)
        sel = np.where((obs_seed == s) & (obs_g <= gmax))[0]
        for idx in sel:
            a = int(obs_edit[idx]); g = int(obs_g[idx]); gr = int(obs_group[idx])
            others = [b for b in mem[(s, g, gr)] if b != a]
            if not others:
                continue
            d = np.zeros(R.shape[1])
            for b in others:
                d += cross_term(R[b], K[b], K[a], denom[b])
            RD.append(float(np.dot(d, R[a])) / (Rn2[a] + 1e-12))
            DR.append(float(ls[a] - obs_lp[idx]))
    RD = np.array(RD); DR = np.array(DR)
    pos = RD > 0
    gain = float(np.median(np.abs(DR[pos]) / RD[pos])) if pos.sum() else None
    return {
        "model": meta.get("model"), "layer": meta.get("layer"),
        "n_obs": int(RD.size), "n_pos_dose": int(pos.sum()),
        "gain_median_absdrop_per_dose": round(gain, 4) if gain is not None else None,
        "frac_drop_negative": round(float(np.mean(DR[pos] < 0)), 4) if pos.sum() else None,
        "rho_reldose_drop": round(float(_spearman(RD[pos], DR[pos])), 4) if pos.sum() else None,
        "median_rel_dose": round(float(np.median(RD[pos])), 4) if pos.sum() else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(
        HARNESS, "results", "merging", "RG_gain_law_20260715.json"))
    args = ap.parse_args()
    mg = os.path.join(HARNESS, "results", "merging")
    rows = {}
    for meta_path in sorted(glob.glob(os.path.join(mg, "*_RG", "rg_meta.json"))):
        rg_dir = os.path.dirname(meta_path)
        name = os.path.basename(rg_dir)
        print(f"[{time.strftime('%H:%M:%S')}] {name} ...", flush=True)
        try:
            rows[name] = bundle_gain(rg_dir)
        except Exception as e:  # a partially-written bundle mid-wave must not kill the table
            rows[name] = {"status": f"ERROR: {e}"}
    ok = {k: v for k, v in rows.items() if v.get("gain_median_absdrop_per_dose") is not None}
    gains = [v["gain_median_absdrop_per_dose"] for v in ok.values()]
    fracs = [v["frac_drop_negative"] for v in ok.values()]
    ordering = (round(float(_spearman(np.array(gains), np.array(fracs))), 4)
                if len(ok) >= 4 else None)
    report = {
        "experiment": "RG_gain_law",
        "status": "DESCRIPTIVE_NON_PREREGISTERED",
        "protocol": "docs/plans/PREDICTIONS-GAIN-WAVE-2026-07-15.md (frozen pre-launch)",
        "schema_version": SCHEMA_VERSION,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "bundles": rows,
        "ordering_test": {
            "spearman_gain_vs_fracdropneg": ordering,
            "n_bundles": len(ok),
            "frozen_prediction": "<= -0.7",
        },
    }
    tmp = args.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(report, f, indent=2)
    os.replace(tmp, args.out)
    print(f"wrote {args.out}")
    for k, v in sorted(ok.items(), key=lambda kv: -kv[1]["gain_median_absdrop_per_dose"]):
        print(f"{k:<26} gain={v['gain_median_absdrop_per_dose']:>9.3f}  "
              f"frac_drop_neg={v['frac_drop_negative']:.3f}  "
              f"rho(dose,drop)={v['rho_reldose_drop']:+.3f}")
    print(f"ordering Spearman(gain, frac_neg) = {ordering}  (prediction <= -0.7)")


if __name__ == "__main__":
    main()
