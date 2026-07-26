#!/usr/bin/env python3
"""Held-out-split and fixed-g re-analyses of the gain ordering (referee demands M1/M3),
plus per-cell bootstrap CIs for gain and constructive fraction (R1.R7).

M1 (shared-data coupling): estimate gain on a DISJOINT half of each cell's merge
observations and compute the constructive fraction on the other half, then correlate
across the 22 cells. If the ordering is a re-description of shared drops, it dies here;
if gain is a real cell-level property, it survives (split-half noise only).

M3 (pooled-g conflation): recompute the ordering with the outcome restricted to g=2
observations only (the regime-cleanest group size), gain still pooled g<=20 and,
separately, gain also restricted to g=2.

Per-observation dose is recomputed exactly as rg_gain_law.py defines it:
dose_a = (d_a . r_a) / ||r_a||^2 with d_a = sum_{b in group, b!=a} cross_term(...).
Pooled positive-dose observations only, matching the frozen convention.

CPU-only; reads frozen bundle npz files; writes results/merging/RG_gain_holdout_20260716.json.
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from merging_m0 import cross_term  # noqa: E402
from scipy import stats  # noqa: E402

GSIZES = (2, 3, 5, 10, 20)
SEEDS = (0, 1, 2)


def per_obs_dose_drop(bundle: Path):
    """All pooled (dose, drop, g) observations for a bundle, positive-dose filtered."""
    z = np.load(bundle / "rg_measurements.npz")
    out_dose, out_drop, out_g = [], [], []
    for s in SEEDS:
        v = np.load(bundle / f"rg_seed{s}_vectors.npz")
        K, R, denom = v["K"], v["R"], v["denom"]
        key_norm, logit_solo = v["key_norm"], v["logit_solo"]
        del key_norm  # dose does not use it; keep load explicit
        for g in GSIZES:
            msel = (z["mem_seed"] == s) & (z["mem_g"] == g)
            members = {}
            for grp, ed in zip(z["mem_group"][msel], z["mem_edit"][msel]):
                members.setdefault(int(grp), []).append(int(ed))
            osel = (z["obs_seed"] == s) & (z["obs_g"] == g)
            for a, grp, lp in zip(z["obs_edit"][osel], z["obs_group"][osel],
                                  z["obs_logit_post"][osel]):
                a, grp = int(a), int(grp)
                others = [b for b in members[grp] if b != a]
                if not others:
                    continue
                d_a = np.zeros(R.shape[1])
                for b in others:
                    d_a += cross_term(R[b], K[b], K[a], denom[b])
                r_a = R[a]
                dose = float(np.dot(d_a, r_a) / (np.dot(r_a, r_a) + 1e-12))
                if dose <= 0:
                    continue
                out_dose.append(dose)
                out_drop.append(float(logit_solo[a] - lp))
                out_g.append(g)
    return np.array(out_dose), np.array(out_drop), np.array(out_g)


def gain_of(dose, drop):
    return float(np.median(np.abs(drop) / dose))


def frac_of(drop):
    return float((drop < 0).mean())


def main():
    gl = json.load(open("results/merging/RG_gain_law_20260715.json"))
    bundles = {n: b for n, b in gl["bundles"].items()}
    rng = np.random.default_rng(20260716)

    rows = {}
    for name in sorted(bundles):
        bdir = Path("results/merging") / name
        dose, drop, g = per_obs_dose_drop(bdir)
        # sanity: full-data gain must reproduce the frozen artifact
        gain_full = gain_of(dose, drop)
        ref = bundles[name]["gain_median_absdrop_per_dose"]
        if abs(gain_full - ref) / (abs(ref) + 1e-12) > 0.02:
            raise AssertionError(f"{name}: recomputed gain {gain_full:.4f} != frozen {ref:.4f}")
        # split-half: random disjoint halves of pooled observations
        idx = rng.permutation(len(dose))
        A, B = idx[: len(idx) // 2], idx[len(idx) // 2:]
        # fixed-g outcomes
        g2 = g == 2
        # per-cell bootstrap CIs (2000 reps) for gain and frac on full data
        bg, bf = [], []
        for _ in range(2000):
            i = rng.integers(0, len(dose), len(dose))
            bg.append(gain_of(dose[i], drop[i]))
            bf.append(frac_of(drop[i]))
        rows[name] = {
            "gain_full": gain_full,
            "frac_full": frac_of(drop),
            "gain_halfA": gain_of(dose[A], drop[A]),
            "frac_halfB": frac_of(drop[B]),
            "gain_g2": gain_of(dose[g2], drop[g2]),
            "frac_g2": frac_of(drop[g2]),
            "gain_ci95": [float(np.percentile(bg, 2.5)), float(np.percentile(bg, 97.5))],
            "frac_ci95": [float(np.percentile(bf, 2.5)), float(np.percentile(bf, 97.5))],
            "n_obs_posdose": int(len(dose)),
        }
        print(f"{name:32s} gain {gain_full:7.2f} (A {rows[name]['gain_halfA']:7.2f}) "
              f"frac {rows[name]['frac_full']:.3f} (B {rows[name]['frac_halfB']:.3f}) "
              f"g2 {rows[name]['gain_g2']:7.2f}/{rows[name]['frac_g2']:.3f}", flush=True)

    names = sorted(rows)
    def spear(x, y):
        r, p = stats.spearmanr(x, y)
        return {"rho": float(r), "p": float(p)}

    orderings = {
        "full_pooled (frozen headline)": spear([rows[n]["gain_full"] for n in names],
                                               [rows[n]["frac_full"] for n in names]),
        "heldout_split (gain half A vs frac half B)": spear(
            [rows[n]["gain_halfA"] for n in names], [rows[n]["frac_halfB"] for n in names]),
        "fixed_g2_outcome (gain pooled vs frac at g=2)": spear(
            [rows[n]["gain_full"] for n in names], [rows[n]["frac_g2"] for n in names]),
        "fixed_g2_both (gain at g=2 vs frac at g=2)": spear(
            [rows[n]["gain_g2"] for n in names], [rows[n]["frac_g2"] for n in names]),
    }
    out = {
        "experiment": "RG_gain_holdout",
        "created": "2026-07-16",
        "provenance": {"gain_law": "results/merging/RG_gain_law_20260715.json",
                       "bundles": [str(Path('results/merging') / n) for n in names],
                       "split": "random disjoint halves of pooled positive-dose obs, seed 20260716"},
        "orderings": orderings,
        "cells": rows,
    }
    Path("results/merging/RG_gain_holdout_20260716.json").write_text(json.dumps(out, indent=1))
    for k, v in orderings.items():
        print(f"{k}: rho={v['rho']:+.4f} p={v['p']:.2e}")


if __name__ == "__main__":
    main()
