#!/usr/bin/env python3
"""DESCRIPTIVE gain-law rows for editor-general federation bundles (schema editors.v1).

Companion to rg_gain_law.py (ROME/merging_m0 bundles). The editors.v1 bundles store
per-observation obs_dose and obs_drop directly in rg_measurements.npz (added in the
07-16 review round), so gain is computable WITHOUT re-deriving cross-terms:
  gain   = median over positive-dose obs (g<=20 pooled, all seeds) of |drop| / dose
  regime = frac(drop < 0) over the same observations
  rho    = Spearman(dose, drop) over the same observations
Used for PREREG-FED-EDITORS-2026-07-16.md prediction (c): the gain-vs-constructive
ordering extends across editor-varied cells. Nothing here alters any frozen verdict.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from merging_m0 import _spearman, SCHEMA_VERSION

HARNESS = os.path.dirname(HERE)


def bundle_gain(rg_dir, gmax=20):
    with open(os.path.join(rg_dir, "rg_meta.json")) as f:
        meta = json.load(f)
    meas = dict(np.load(os.path.join(rg_dir, "rg_measurements.npz")))
    for k in ("obs_dose", "obs_drop", "obs_g"):
        if k not in meas:
            return {"status": f"MISSING_FIELD:{k} (schema {meta.get('schema_version')})"}
    sel = meas["obs_g"].astype(int) <= gmax
    dose = meas["obs_dose"].astype(float)[sel]
    drop = meas["obs_drop"].astype(float)[sel]
    pos = dose > 0
    if not pos.sum():
        return {"status": "NO_POSITIVE_DOSE"}
    return {
        "model": meta.get("model"), "editor": meta.get("editor"),
        "dataset": meta.get("dataset"), "layers": meta.get("edited_layers", meta.get("layer")),
        "schema_version": meta.get("schema_version"),
        "n_obs": int(sel.sum()), "n_pos_dose": int(pos.sum()),
        "gain_median_absdrop_per_dose": round(float(np.median(np.abs(drop[pos]) / dose[pos])), 4),
        "frac_drop_negative": round(float(np.mean(drop[pos] < 0)), 4),
        "rho_dose_drop": round(float(_spearman(dose[pos], drop[pos])), 4),
        "median_rel_dose": round(float(np.median(dose[pos])), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(
        HARNESS, "results", "merging_editors", "RG_gain_law_editors_20260716.json"))
    args = ap.parse_args()
    mg = os.path.join(HARNESS, "results", "merging_editors")
    rows = {}
    for meta_path in sorted(glob.glob(os.path.join(mg, "*_RG", "rg_meta.json"))):
        rg_dir = os.path.dirname(meta_path)
        name = os.path.basename(rg_dir)
        print(f"[{time.strftime('%H:%M:%S')}] {name} ...", flush=True)
        try:
            rows[name] = bundle_gain(rg_dir)
        except Exception as e:
            rows[name] = {"status": f"ERROR: {e}"}
    report = {
        "experiment": "RG_gain_law_editors",
        "status": "DESCRIPTIVE (prereg prediction (c) read-out; frozen doc "
                  "PREREG-FED-EDITORS-2026-07-16.md)",
        "schema_version": SCHEMA_VERSION,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "bundles": rows,
    }
    tmp = args.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(report, f, indent=2)
    os.replace(tmp, args.out)
    print(f"wrote {args.out}")
    ok = {k: v for k, v in rows.items() if "gain_median_absdrop_per_dose" in v}
    for k, v in sorted(ok.items(), key=lambda kv: -kv[1]["gain_median_absdrop_per_dose"]):
        print(f"{k:<40} gain={v['gain_median_absdrop_per_dose']:>9.3f}  "
              f"frac_neg={v['frac_drop_negative']:.3f}  rho={v['rho_dose_drop']:+.3f}")


if __name__ == "__main__":
    main()
