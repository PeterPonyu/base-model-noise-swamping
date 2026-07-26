"""magnitude_table.py — the CANONICAL per-family MAGNITUDE-law table.

AUTHORING PASS (2026-07-04). A separate reviewer/verifier pass gates any paper
use of these numbers; do NOT cite this file's output as verified.

WHY THIS FILE EXISTS
--------------------
The paper (fig F7) claims "the magnitude law transfers 4/5 families", but that
number lived only in prose / peek records — no JSON stored it. This script
produces the canonical table so the claim has a reproducible artifact.

DEFINITION
----------
The MAGNITUDE law is the within-probe Spearman correlation between the
*absolute* pre-edit key-cosine and the *absolute* collateral damage:

        within-probe rho( |COS| , |damage| )         [magnitude]

as opposed to the SIGNED law measured by analyze_matrices.py:

        within-probe rho(  COS  ,  damage  )          [signed]

Everything else is IDENTICAL to analyze_matrices.py: the tie-averaged /
partialled-by-probe-column Spearman machinery, the --known (pre_p>0.05) and
--edit_ok (edit_ok>0.5) filters, the probe-level (legacy) and STRICT edit-level
permutation nulls, and the aggregate-across-seeds JSON shape. This script
IMPORTS those functions from analyze_matrices so the two tables cannot drift.

The only change is: feed |COS| and |damage| into the same estimators.

USAGE
-----
  # canonical run (all families, G1 filter convention):
  python experiments/magnitude_table.py --known --edit_ok \
      --out results/C1_magnitude_table.json

  # a single family / custom glob:
  python experiments/magnitude_table.py --known --edit_ok \
      --family llama1b_L12=results/matrices/gate_llama1b_rome_cf_L12_s*.npz
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

# Import the EXACT estimators/nulls from the signed analyzer so the magnitude
# table shares one implementation with the signed table (no drift).
from analyze_matrices import (  # noqa: E402
    RNG_SEED,
    boot_ci,
    edit_cluster_boot_ci,
    editlevel_permutation_null,
    per_edit_rhos,
    spearman,
    within_probe_permutation_null,
    within_probe_rhos,
)

# Canonical family -> glob spec. Layer per family is the campaign's chosen
# ROME edit layer (the one the cross-family magnitude claim is quoted at).
DEFAULT_FAMILIES = {
    "llama1b_L8":  "results/matrices/gate_llama1b_rome_cf_L8_s*.npz",
    "llama1b_L10": "results/matrices/gate_llama1b_rome_cf_L10_s*.npz",
    "llama1b_L12": "results/matrices/gate_llama1b_rome_cf_L12_s*.npz",
    "llama1b_L14": "results/matrices/gate_llama1b_rome_cf_L14_s*.npz",
    "qwen05b_L12": "results/matrices/gate_qwen05b_rome_cf_L12_s*.npz",
    "qwen15b_L14": "results/matrices/gate_qwen15b_rome_cf_L14_s*.npz",
    "qwen3b_L18":  "results/matrices/gate_qwen3b_rome_cf_L18_s*.npz",
    "phi35_L16":   "results/matrices/gate_phi35_rome_cf_L16_s*.npz",
    "gemma2b_L13": "results/matrices/gate_gemma2b_rome_cf_L13_s*.npz",
}

# Remembered 07-02 peek values (magnitude law), for the honest cross-check.
# These are the "magnitude law transfers 4/5 families" numbers from prose.
REMEMBERED = {
    "llama1b_L12": 0.613,
    "qwen05b_L12": 0.320,
    "qwen15b_L14": 0.412,
    "qwen3b_L18":  0.411,
    "phi35_L16":   0.362,
    "gemma2b_L13": 0.111,
}


def analyze_one_magnitude(npz_path, metric, known, edit_ok_filter, n_perm=1000):
    """MAGNITUDE analog of analyze_matrices.analyze_one: same filters, same
    estimators, but correlate |COS| against |damage|."""
    d = np.load(npz_path)
    COS = d["COS"].astype(float)                                   # [N,M]
    D = (d["damage_logit"] if metric == "logit" else d["damage_prob"]).astype(float)
    # filters (identical to the signed analyzer)
    if edit_ok_filter and "edit_ok" in d:
        rows = d["edit_ok"].astype(float) > 0.5
        COS, D = COS[rows], D[rows]
    if known and "pre_p" in d:
        cols = d["pre_p"].astype(float) > 0.05
        if cols.sum() >= 5:
            COS, D = COS[:, cols], D[:, cols]

    # THE MAGNITUDE TRANSFORM: absolute value on both axes, then the identical
    # within-probe / per-edit / null machinery.
    A = np.abs(COS)
    B = np.abs(D)

    wp = within_probe_rhos(A, B)
    pe = per_edit_rhos(A, B)
    wp_mean = float(np.nanmean(wp))
    wp_perm_p = within_probe_permutation_null(A, B, wp_mean)   # legacy per-column null
    edit_perm_p, null_mean, null_std = editlevel_permutation_null(A, B, wp_mean, n_perm=n_perm)
    cl_lo, cl_hi = edit_cluster_boot_ci(A, B)
    return {
        "npz": os.path.basename(npz_path),
        "shape": [int(A.shape[0]), int(A.shape[1])],
        "flat_spearman_abs_INFLATED": round(spearman(A.reshape(-1), B.reshape(-1)), 4),
        "within_probe_mean_abs": round(wp_mean, 4),
        "within_probe_ci95": [round(x, 4) for x in boot_ci(wp)],
        "within_probe_ci95_editcluster": [round(x, 4) for x in (cl_lo, cl_hi)],
        "within_probe_frac_positive": round(float(np.nanmean(wp > 0)), 3),
        "per_edit_mean_abs": round(float(np.nanmean(pe)), 4),
        "per_edit_ci95": [round(x, 4) for x in boot_ci(pe)],
        "within_probe_perm_p": round(wp_perm_p, 4),                # legacy (anti-conservative)
        "within_probe_perm_p_editlevel": round(edit_perm_p, 4),    # STRICT null (primary)
        "editlevel_null_mean": round(null_mean, 4),
        "editlevel_null_std": round(null_std, 4),
        "editlevel_z": (round((wp_mean - null_mean) / null_std, 2)
                        if null_std > 1e-9 else None),
    }


def verdict(within_mean, perm_p):
    """Same PASS/BORDERLINE/DEAD gate as the signed analyzer, applied to the
    magnitude within-probe mean. 'transfers' == PASS here."""
    if not np.isfinite(within_mean):
        return "UNDETERMINED"
    if within_mean >= 0.15 and perm_p < 0.05:
        return "PASS — magnitude law present (edit-specific |geometry|->|damage|)"
    if within_mean >= 0.10:
        return "BORDERLINE — weak magnitude signal; needs more edits/probes or reframe"
    return "DEAD — within-probe |rho| < 0.10; magnitude law does NOT transfer here"


def analyze_family(name, pattern, metric, known, edit_ok_filter, n_perm):
    paths = sorted(glob.glob(pattern))
    if not paths:
        return {
            "family": name, "name": name, "pattern": pattern, "n_seeds": 0,
            "within_probe_mean_across_seeds": None, "within_probe_std_across_seeds": None,
            "MISSING": True, "per_seed": [],
        }
    per_seed = [analyze_one_magnitude(p, metric, known, edit_ok_filter, n_perm=n_perm)
                for p in paths]
    wp = np.array([r["within_probe_mean_abs"] for r in per_seed], float)
    pe = np.array([r["per_edit_mean_abs"] for r in per_seed], float)
    ppe = np.array([r["within_probe_perm_p_editlevel"] for r in per_seed], float)
    mean_across = float(np.nanmean(wp))
    std_across = float(np.nanstd(wp))
    out = {
        "family": name,
        # `name` + `within_probe_mean_across_seeds`(+ std) are the figures-hook
        # contract keys (make_figures.py F7 panel). They alias the explicit
        # magnitude fields below; for this table the within-probe mean IS the
        # |cos|-vs-|damage| magnitude mean.
        "name": name,
        "within_probe_mean_across_seeds": round(mean_across, 4),
        "within_probe_std_across_seeds": round(std_across, 4),
        "pattern": pattern,
        "n_seeds": len(paths),
        "seeds": [r["npz"] for r in per_seed],
        "within_probe_mean_abs_across_seeds": round(mean_across, 4),
        "within_probe_std_abs_across_seeds": round(std_across, 4),
        "per_edit_mean_abs_across_seeds": round(float(np.nanmean(pe)), 4),
        "max_within_probe_perm_p_editlevel": round(float(np.nanmax(ppe)), 4),
        "VERDICT": verdict(mean_across, float(np.nanmax(ppe))),
        "per_seed": per_seed,
    }
    if name in REMEMBERED:
        rem = REMEMBERED[name]
        out["remembered_0702"] = rem
        out["delta_vs_remembered"] = round(mean_across - rem, 4)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", action="append", default=None,
                    metavar="NAME=GLOB",
                    help="override/add a family as NAME=glob; repeatable. "
                         "If given, ONLY these families are analyzed.")
    ap.add_argument("--metric", choices=["logit", "prob"], default="logit")
    ap.add_argument("--known", action="store_true",
                    help="filter to probes the base model knows (pre_p>0.05)")
    ap.add_argument("--edit_ok", action="store_true",
                    help="drop failed edits (edit_ok==0)")
    ap.add_argument("--n_perm", type=int, default=1000,
                    help="permutations for the strict edit-level null (default 1000)")
    ap.add_argument("--out", default="results/C1_magnitude_table.json")
    args = ap.parse_args()

    if args.family:
        families = {}
        for spec in args.family:
            if "=" not in spec:
                raise SystemExit(f"--family expects NAME=GLOB, got {spec!r}")
            k, v = spec.split("=", 1)
            families[k] = v
    else:
        families = DEFAULT_FAMILIES

    fam_results = [analyze_family(name, pat, args.metric, args.known,
                                  args.edit_ok, args.n_perm)
                   for name, pat in families.items()]

    # Compact headline table: family -> mean±std (magnitude within-probe rho).
    headline = {}
    for f in fam_results:
        if f.get("MISSING"):
            headline[f["family"]] = "MISSING (0 npz)"
        else:
            headline[f["family"]] = {
                "mean_abs": f["within_probe_mean_abs_across_seeds"],
                "std_abs": f["within_probe_std_abs_across_seeds"],
                "n_seeds": f["n_seeds"],
                "verdict": f["VERDICT"].split(" — ")[0],
            }

    n_transfer = sum(1 for f in fam_results
                     if not f.get("MISSING") and f["VERDICT"].startswith("PASS"))
    n_fam = sum(1 for f in fam_results if not f.get("MISSING"))
    res = {
        "definition": "within-probe Spearman(|key-cos|, |damage|), tie-averaged, "
                       "partialled by holding probe identity fixed down each column",
        "metric": args.metric,
        "filters": {"known": args.known, "edit_ok": args.edit_ok},
        "n_perm_editlevel": args.n_perm,
        "families_transfer_PASS": f"{n_transfer}/{n_fam}",
        "headline": headline,
        "families": fam_results,
    }
    print(json.dumps(res, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(res, open(args.out, "w"), indent=2)
        print(f"[magnitude] wrote {args.out}")


if __name__ == "__main__":
    main()
