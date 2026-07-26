#!/usr/bin/env python3
"""Consolidate the per-cell RG operating-curve tables into ONE map-evidence artifact
for the D2 paper's dose-response and gate-evidence figures.

Reads every results/merging/RG_operating_curve_table*.json (one per model x layer cell)
plus results/merging/RG_gain_law_20260715.json (for the canonical cell list, gain and
regime), and emits per cell x group size:
  - median_abs_drop_logit per seed and the across-seed median (dose-response figure)
  - partial_rho_geom per seed, mean and min/max (gate-evidence figure)
  - the qualification flags (non_negligible, saturated, c2_coherent) per seed
plus per-cell totals (n_obs summed over sub-cells) so the paper can state the true
observation count of the map. CPU-only, no model access; pure re-keying of frozen JSONs
(inputs listed in the output's provenance block).
"""
import glob
import json
import os
import re

import numpy as np

OUT = "results/merging/RG_map_evidence_20260716.json"

# canonical cell list + gain/regime from the gain-law artifact
gl = json.load(open("results/merging/RG_gain_law_20260715.json"))
bundles = gl["bundles"]


def cell_key_from_bundle_name(name: str) -> str:
    return name  # gain_law keys are '<model>_L<layer>_RG'-style dir names


# map opcurve files to gain-law bundles via (model tail, layer)
op_files = sorted(glob.glob("results/merging/RG_operating_curve_table*.json")
                  + glob.glob("results/merging/*_RG/RG_operating_curve_table.json"))
ops = {}
for p in op_files:
    t = json.load(open(p))
    model_tail = os.path.basename(str(t["model"]).rstrip("/"))
    ops[(model_tail, int(t["layer"]))] = (p, t)

out_cells = {}
missing = []
total_obs = 0
n_subcells = 0
for bname, b in bundles.items():
    model_tail = os.path.basename(str(b["model"]).rstrip("/"))
    m = re.search(r"_L(\d+)", bname)
    layer = int(m.group(1)) if m else None
    hit = ops.get((model_tail, layer))
    if hit is None:
        missing.append(bname)
        continue
    path, t = hit
    cells = t["cells"]
    per_g = {}
    for g in (2, 3, 5, 10, 20):
        seeds = [cells.get(f"g{g}_s{s}") for s in (0, 1, 2)]
        seeds = [c for c in seeds if c]
        if not seeds:
            continue
        n_subcells += len(seeds)
        total_obs += sum(int(c["n_obs"]) for c in seeds)
        med = [float(c["median_abs_drop_logit"]) for c in seeds]
        par = [float(c["partial_rho_geom"]) for c in seeds]
        per_g[str(g)] = {
            "median_abs_drop_per_seed": med,
            "median_abs_drop_med3": float(np.median(med)),
            "partial_rho_per_seed": par,
            "partial_rho_mean": float(np.mean(par)),
            "partial_rho_min": float(min(par)),
            "partial_rho_max": float(max(par)),
            "non_negligible": [bool(c["non_negligible"]) for c in seeds],
            "saturated": [bool(c["saturated"]) for c in seeds],
            "c2_coherent": [bool(c["c2_coherent"]) for c in seeds],
        }
    out_cells[bname] = {
        "model": b["model"], "layer": layer,
        "gain": float(b["gain_median_absdrop_per_dose"]),
        "frac_negative": float(b["frac_drop_negative"]),
        # regime banding follows the paper's display convention: the gain cut at 8
        # (see main.tex "Threshold, gradedness, and robustness"); the OUTCOME midpoint
        # (frac >= 0.5) is a separate, graded quantity carried in frac_negative.
        "regime": "low-gain" if b["gain_median_absdrop_per_dose"] < 8 else "high-gain",
        "source_table": path,
        "per_g": per_g,
    }

out = {
    "experiment": "RG_map_evidence",
    "created": "2026-07-16",
    "provenance": {"gain_law": "results/merging/RG_gain_law_20260715.json",
                   "opcurve_files": op_files},
    "n_cells": len(out_cells),
    "n_subcells": n_subcells,
    "total_merge_observations": total_obs,
    "missing_opcurve_for": missing,
    "cells": out_cells,
}
json.dump(out, open(OUT, "w"), indent=1)
print(f"cells={len(out_cells)} subcells={n_subcells} total_obs={total_obs} missing={missing}")
print(f"-> {OUT}")
