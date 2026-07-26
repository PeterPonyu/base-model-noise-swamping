#!/usr/bin/env python3
"""enhance_aggregates.py — canonical aggregate tables for the 2026-07-09 enhancement
round. IDEMPOTENT + PARTIAL-SAFE: computes from whatever rows are on disk and stamps
per-table coverage, so it is re-run (not hand-edited) as the remaining GPU rows land
(NeoX L11 s1/s2 + alphaHO + DOC; Pythia battery; ripple alpha s2). Figures and prose
must read THESE jsons, never per-row files (the make_figures.R provenance convention).

Outputs (results/):
  GLUE_BRIDGE_summary.json   — per-row esr/damage/flips/NG + the 3 estimands per row:
                               within-example cosine rho, edit-level NG->|margin dmg|,
                               edit-level NG->flip-rate; plus the rome-vs-alpha contrast
  RIPPLE_depth_profile.json  — per-layer 3-seed pooled rho_ripple / rho_unrelated
  NEOX20B_law_table.json     — per-layer/seed within-probe cosine + norm-growth rho
                               (edit_ok x known-probe filtered, the G1 estimand)
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys

import numpy as np
from scipy.stats import spearmanr

HARNESS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HARNESS, "experiments"))
from analyze_matrices import within_probe_rhos  # noqa: E402

R = lambda p: os.path.join(HARNESS, "results", p)


def glue_bridge() -> dict:
    rows = []
    for jp in sorted(glob.glob(R("glue_bridge/gb_*.json"))):
        base = os.path.splitext(os.path.basename(jp))[0]
        j = json.load(open(jp))
        npz = R(f"matrices/{base}.npz")
        row = {
            "row": base, "editor": j["editor"], "layer": j["layer"], "seed": j["seed"],
            "esr": j["edit_success_rate"], "pre_accuracy": j["pre_accuracy"],
            "cos_within_example_rho_filtered":
                j["rho"]["margin"]["within_example_mean_editok_precorrect"],
        }
        if os.path.exists(npz):
            d = np.load(npz)
            ok = d["edit_ok"] > 0
            pc = d["pre_correct"] > 0
            dm = np.abs(d["damage_margin"][ok][:, pc])
            fl = d["damage_flip"][ok][:, pc]
            ng = d["norm_growth"][ok]
            r_m, p_m = spearmanr(ng, dm.mean(axis=1))
            r_f, p_f = spearmanr(ng, fl.mean(axis=1))
            row.update({
                "mean_abs_margin_damage": round(float(dm.mean()), 4),
                "flip_rate": round(float(fl.mean()), 5),
                "mean_norm_growth": round(float(ng.mean()), 3),
                "ng_to_margin_damage_rho": round(float(r_m), 4),
                "ng_to_margin_damage_p": float(f"{p_m:.2g}"),
                "ng_to_flip_rho": round(float(r_f), 4),
                "ng_to_flip_p": float(f"{p_f:.2g}"),
                "n_edit_ok": int(ok.sum()),
            })
        rows.append(row)

    def agg(ed, layer, key):
        v = [r[key] for r in rows
             if r["editor"] == ed and r["layer"] == layer and key in r]
        return (round(float(np.mean(v)), 4), len(v)) if v else (None, 0)

    contrast = {}
    for key in ("mean_abs_margin_damage", "flip_rate", "mean_norm_growth"):
        rm, n_rm = agg("rome", 12, key)
        al, n_al = agg("alpha", 12, key)
        contrast[key] = {"rome_L12_mean": rm, "alpha_L12_mean": al,
                         "n_seeds": [n_rm, n_al],
                         "ratio_rome_over_alpha":
                             (round(rm / al, 2) if rm and al else None)}
    return {"rows": rows, "rome_vs_alpha_L12": contrast,
            "coverage": f"{len(rows)}/8 planned science rows (driver's '9 done' includes the smoke)",
            "estimand_note": ("cos rho = within-example Spearman across edits, edit_ok x "
                              "pre-correct filtered; NG rhos = edit-level Spearman of "
                              "norm_growth vs per-edit mean damage / flip rate")}


def ripple() -> dict:
    layers: dict = {}
    for jp in sorted(glob.glob(R("ripple_llama1b_rome_popular_L*_s*.json"))):
        m = re.search(r"_L(\d+)_s(\d+)\.json$", jp)
        if not m:
            continue
        j = json.load(open(jp))
        e = layers.setdefault(int(m.group(1)), {"rho_ripple": [], "rho_unrelated": [], "esr": []})
        e["rho_ripple"].append(j["within_probe_rho_logit"]["ripple"])
        e["rho_unrelated"].append(j["within_probe_rho_logit"]["unrelated"])
        e["esr"].append(j["edit_success_rate"])
    prof = {f"L{L}": {
        "n_seeds": len(v["rho_ripple"]),
        "rho_ripple_mean": round(float(np.mean(v["rho_ripple"])), 4),
        "rho_ripple_range": [round(min(v["rho_ripple"]), 3), round(max(v["rho_ripple"]), 3)],
        "rho_unrelated_mean": round(float(np.mean(v["rho_unrelated"])), 4),
        "rho_unrelated_range": [round(min(v["rho_unrelated"]), 3), round(max(v["rho_unrelated"]), 3)],
        "esr_mean": round(float(np.mean(v["esr"])), 3),
    } for L, v in sorted(layers.items())}
    alpha = sorted(os.path.basename(p) for p in glob.glob(R("ripple_llama1b_alpha_popular_L12_s*.json")))
    return {"profile": prof, "alpha_L12_rows_present": alpha,
            "note": ("07-09 extension supersedes the L12-only read: ripple rho ~0.45-0.49 "
                     "at L8/L10/L14 with an L12 DIP; ripple exceeds unrelated at L14. "
                     "Quote the full profile, never L12 alone.")}


def neox() -> dict:
    rows = []
    for np_path in sorted(glob.glob(R("matrices/gate_neox20b_rome_cf_L*_s*.npz"))):
        m = re.search(r"_L(\d+)_s(\d+)\.npz$", np_path)
        if not m:
            continue
        d = np.load(np_path)
        C = d["COS"].astype(float)
        D = d["damage_logit"].astype(float)
        ok = d["edit_ok"].astype(float) > 0
        kn = d["pre_p"].astype(float) > 0.05
        ng = np.repeat(d["norm_growth"].astype(float)[:, None], D.shape[1], axis=1)
        rows.append({
            "layer": int(m.group(1)), "seed": int(m.group(2)),
            "esr": round(float(d["edit_ok"].mean()), 3),
            "frac_known": round(float(kn.mean()), 3),
            "n_edit_ok": int(ok.sum()),
            "rho_cos": round(float(np.nanmean(within_probe_rhos(C[ok][:, kn], D[ok][:, kn]))), 4),
            "rho_norm_growth": round(float(np.nanmean(within_probe_rhos(ng[ok][:, kn], D[ok][:, kn]))), 4),
        })
    rows.sort(key=lambda r: (r["layer"], r["seed"]))
    return {"rows": rows, "lr": 0.5, "coverage": f"{len(rows)} science rows on disk "
            "(grid = L11/L16/L22 x s0-2 + DOC L28/L33 s0; alphaHO separate)",
            "verdict_note": ("BOTH channels dead across the editable band (cosine "
                             "+0.03..+0.08; NG flat except a seed-UNSTABLE L22 trace "
                             "0.162/0.187/0.086). lr-0.1 rows are quarantined "
                             ".DEAD-LR01 — never mix.")}


def main() -> None:
    out = {"GLUE_BRIDGE_summary.json": glue_bridge(),
           "RIPPLE_depth_profile.json": ripple(),
           "NEOX20B_law_table.json": neox()}
    for name, obj in out.items():
        with open(R(name), "w") as f:
            json.dump(obj, f, indent=1)
        cov = obj.get("coverage", "")
        print(f"[aggregates] wrote results/{name} {cov}")


if __name__ == "__main__":
    main()
