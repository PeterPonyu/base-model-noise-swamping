#!/usr/bin/env python3
"""Consolidate the per-cell RG operating-curve tables into ONE map-evidence artifact
for the D2 paper's dose-response and gate-evidence figures.

Reads every results/merging/RG_operating_curve_table*.json (one per model x layer cell)
plus a gain-law artifact (for the canonical cell list, gain and regime), and emits per
cell x group size:
  - median_abs_drop_logit per seed and the across-seed median (dose-response figure)
  - partial_rho_geom per seed, mean and min/max (gate-evidence figure)
  - the qualification flags (non_negligible, saturated, c2_coherent) per seed
plus per-cell totals (n_obs summed over sub-cells) so the paper can state the true
observation count of the map. CPU-only, no model access; pure re-keying of frozen JSONs
(inputs listed in the output's provenance block).

2026-08-01 (deposit self-containment repair): inputs/outputs are now explicit CLI
arguments and duplicate (model, layer) operating tables are resolved DETERMINISTICALLY
instead of by glob order. The pre-Phi-refix run was

  python3 experiments/rg_map_evidence_consolidate.py \
      --gain_law results/merging/RG_gain_law_20260715.json \
      --out      results/merging/RG_map_evidence_20260716.json

and the tokenizer-refixed run (current canonical artifact) is the DEFAULT:

  python3 experiments/rg_map_evidence_consolidate.py
  # --gain_law results/merging/RG_gain_law_MERGED_REFIX20260730.json
  # --out      results/merging/RG_map_evidence_REFIX20260801.json

Rename map (Phi-3.5 tokenizer collision, findings-PHI35-TOKENIZER-COLLISION-2026-07-30):
  RG_gain_law_20260715.json                -> RG_gain_law_MERGED_REFIX20260730.json
  RG_operating_curve_table_phi35_L16.json  -> ..._phi35_L16_REFIX20260730.json
  RG_operating_curve_table_phi35_L24.json  -> ..._phi35_L24_REFIX20260730.json
  RG_map_evidence_20260716.json            -> RG_map_evidence_REFIX20260801.json

Source precedence when several tables describe the same (model, layer):
REFIX-tagged > bundle-local (<cell>_RG/RG_operating_curve_table.json) > flat legacy
name. A tie at the top rank whose contents disagree is a hard error, so a stale
duplicate can never be silently picked up again.
"""
import argparse
import glob
import hashlib
import json
import os
import re

import numpy as np

DEF_RESULTS = "results/merging"
DEF_GAIN_LAW = "RG_gain_law_MERGED_REFIX20260730.json"
DEF_OUT = "RG_map_evidence_REFIX20260801.json"

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--results_dir", default=DEF_RESULTS,
                help=f"directory holding the RG artifacts (default: {DEF_RESULTS})")
ap.add_argument("--gain_law", default=None,
                help=f"gain-law artifact (default: <results_dir>/{DEF_GAIN_LAW})")
ap.add_argument("--out", default=None,
                help=f"output artifact (default: <results_dir>/{DEF_OUT})")
args = ap.parse_args()

RES = args.results_dir
GAIN_LAW = args.gain_law or os.path.join(RES, DEF_GAIN_LAW)
OUT = args.out or os.path.join(RES, DEF_OUT)

# canonical cell list + gain/regime from the gain-law artifact
gl = json.load(open(GAIN_LAW))
bundles = gl["bundles"]


def cell_key_from_bundle_name(name: str) -> str:
    return name  # gain_law keys are '<model>_L<layer>_RG'-style dir names


def _rank(path: str) -> int:
    """Source precedence: REFIX-tagged > bundle-local > flat legacy name."""
    if "REFIX" in os.path.basename(path):
        return 2
    if os.path.basename(path) == "RG_operating_curve_table.json":
        return 1
    return 0


def _digest(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def _rec(path: str) -> str:
    """Record paths as `results/merging/<rel>` so the artifact is byte-identical no
    matter which absolute or relative --results_dir produced it."""
    return os.path.join("results", "merging", os.path.relpath(path, RES))


# map opcurve files to gain-law bundles via (model tail, layer)
op_files = sorted(glob.glob(os.path.join(RES, "RG_operating_curve_table*.json"))
                  + glob.glob(os.path.join(RES, "*_RG", "RG_operating_curve_table.json")))
candidates = {}
for p in op_files:
    t = json.load(open(p))
    model_tail = os.path.basename(str(t["model"]).rstrip("/"))
    candidates.setdefault((model_tail, int(t["layer"])), []).append((p, t))

ops = {}
superseded = []
for key, cands in candidates.items():
    top = max(_rank(p) for p, _ in cands)
    best = [(p, t) for p, t in cands if _rank(p) == top]
    if len({_digest(p) for p, _ in best}) > 1:
        raise SystemExit(
            f"ambiguous operating tables for {key}: "
            f"{[p for p, _ in best]} share the top precedence rank but differ")
    ops[key] = best[0]
    superseded += [_rec(p) for p, _ in cands if _rank(p) < top]

used_files = sorted(_rec(p) for p, _ in ops.values())
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
        "source_table": _rec(path),
        "per_g": per_g,
    }

out = {
    "experiment": "RG_map_evidence",
    "created": "2026-08-01",
    "provenance": {"gain_law": _rec(GAIN_LAW),
                   "opcurve_files": used_files,
                   "superseded_duplicates_ignored": sorted(superseded),
                   "source_precedence": "REFIX > bundle-local > flat legacy name",
                   "note": ("Phi-3.5 rows come from the tokenizer-refixed operating "
                            "tables (findings-PHI35-TOKENIZER-COLLISION-2026-07-30); "
                            "supersedes RG_map_evidence_20260716.json.")},
    "n_cells": len(out_cells),
    "n_subcells": n_subcells,
    "total_merge_observations": total_obs,
    "missing_opcurve_for": missing,
    "cells": out_cells,
}
json.dump(out, open(OUT, "w"), indent=1)
print(f"cells={len(out_cells)} subcells={n_subcells} total_obs={total_obs} missing={missing}")
print(f"-> {OUT}")
