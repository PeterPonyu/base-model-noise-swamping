#!/usr/bin/env python3
"""Retrospective admission-benefit table for the D2 federation paper ([TODO-BENEFIT]).

SPEC (FROZEN in submissions/d2-federation/canonical.md §6 BEFORE this implementation —
implement exactly, no metric shopping):
  - per (cell, g): admission = bottom-q I_cos observations at FIXED group composition
    (retrospective: groups as measured; the rule selects which member edits to accept).
  - primary metric = mean signed drop among admitted edits vs the same-budget RANDOM
    baseline (expected mean of a uniform q-subset = the overall mean; exact, no MC).
  - baselines: random (overall mean) and magnitude-only (bottom-q by I_mag).
  - report per-regime aggregates with SEEDS SEPARATED; no threshold fitting.
Regime label comes from the measured gain (RG_gain_law_20260715.json); the high/low
split uses the pre-existing bimodal gap (cut at gain=8: observed clusters 12.5+ vs 3.6-).

benefit = mean_drop(random) - mean_drop(admitted): positive = damage avoided per
admitted edit (destructive regimes); in constructive regimes a NEGATIVE benefit under
bottom-q admission is expected (low-I_cos edits forgo the boost) and is reported as-is.

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
from merging_m0 import load_rg, SCHEMA_VERSION

HARNESS = os.path.dirname(HERE)
BUDGETS = (0.25, 0.50)
GSMALL = (2, 3, 5)          # small-g (regime-defining) aggregate
GAIN_CUT = 8.0              # bimodal gap: high-gain >= 8, low-gain < 8 (observed 12.5 vs 3.6)


def per_cell(v, obs_edit, obs_group, obs_lp, members):
    K = v["K"].astype(float); S = v["S"].astype(float)
    key_norm = v["key_norm"].astype(float)
    logit_solo = v["logit_solo"].astype(float)
    I_cos, I_mag, drop = [], [], []
    for a, g, lp in zip(obs_edit, obs_group, obs_lp):
        a, g = int(a), int(g)
        others = [b for b in members[g] if b != a]
        if not others:
            continue
        ob = np.array(others)
        coses = np.array([abs(float(np.dot(K[b], K[a]))) /
                          (key_norm[b] * key_norm[a] + 1e-12) for b in others])
        w = S[ob] * key_norm[a]
        I_cos.append(float(np.sum(w * coses)))
        I_mag.append(float(np.sum(w)))
        drop.append(float(logit_solo[a] - lp))
    return np.array(I_cos), np.array(I_mag), np.array(drop)


def admit_stats(I, drop, q):
    n = drop.size
    k = max(1, int(np.floor(q * n)))
    idx = np.argsort(I, kind="stable")[:k]
    return float(np.mean(drop[idx])), k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(
        HARNESS, "results", "merging", "RG_admission_benefit_20260715.json"))
    args = ap.parse_args()
    mg = os.path.join(HARNESS, "results", "merging")

    with open(os.path.join(mg, "RG_gain_law_20260715.json")) as f:
        gain_tbl = json.load(f)["bundles"]

    rows = {}
    for meta_path in sorted(glob.glob(os.path.join(mg, "*_RG", "rg_meta.json"))):
        rg_dir = os.path.dirname(meta_path)
        name = os.path.basename(rg_dir)
        gain = gain_tbl.get(name, {}).get("gain_median_absdrop_per_dose")
        if gain is None:
            rows[name] = {"status": "NO_GAIN_ROW"}
            continue
        regime = "high_gain" if gain >= GAIN_CUT else "low_gain"
        per_seed, meas, meta = load_rg(rg_dir)
        obs_seed = meas["obs_seed"].astype(int); obs_g = meas["obs_g"].astype(int)
        obs_group = meas["obs_group"].astype(int); obs_edit = meas["obs_edit"].astype(int)
        obs_lp = meas["obs_logit_post"].astype(float)
        mem = defaultdict(list)
        for s, g, gr, ed in zip(meas["mem_seed"].astype(int), meas["mem_g"].astype(int),
                                meas["mem_group"].astype(int), meas["mem_edit"].astype(int)):
            mem[(s, g, gr)].append(ed)
        cells = {}
        for s in [int(x) for x in meta["seeds"]]:
            for g in [int(x) for x in meta["group_sizes"] if int(x) <= 20]:
                sel = np.where((obs_seed == s) & (obs_g == g))[0]
                if sel.size == 0:
                    continue
                members = {int(gr): mem[(s, g, int(gr))]
                           for gr in set(obs_group[sel].tolist())}
                I_cos, I_mag, drop = per_cell(per_seed[s], obs_edit[sel],
                                              obs_group[sel], obs_lp[sel], members)
                if drop.size < 20:
                    continue
                rand_mean = float(np.mean(drop))
                ent = {"n_obs": int(drop.size), "mean_drop_random": round(rand_mean, 4)}
                for q in BUDGETS:
                    gm, k = admit_stats(I_cos, drop, q)
                    mm, _ = admit_stats(I_mag, drop, q)
                    ent[f"q{int(q*100)}"] = {
                        "k_admitted": k,
                        "mean_drop_geometry": round(gm, 4),
                        "mean_drop_magnitude": round(mm, 4),
                        "benefit_geometry_vs_random": round(rand_mean - gm, 4),
                        "benefit_magnitude_vs_random": round(rand_mean - mm, 4),
                    }
                cells[f"g{g}_s{s}"] = ent
        rows[name] = {"gain": gain, "regime": regime, "cells": cells}

    # per-regime small-g aggregates, seeds separated then averaged (spec: seeds separated)
    agg = {}
    for regime in ("high_gain", "low_gain"):
        for q in BUDGETS:
            qk = f"q{int(q*100)}"
            per_seed_vals = defaultdict(list)   # seed -> benefits across cells/g
            mag_vals = defaultdict(list)
            for name, r in rows.items():
                if r.get("regime") != regime:
                    continue
                for key, ent in r["cells"].items():
                    g = int(key.split("_")[0][1:]); s = key.split("_s")[1]
                    if g not in GSMALL:
                        continue
                    per_seed_vals[s].append(ent[qk]["benefit_geometry_vs_random"])
                    mag_vals[s].append(ent[qk]["benefit_magnitude_vs_random"])
            agg[f"{regime}_{qk}"] = {
                "benefit_geometry_by_seed": {s: round(float(np.mean(v)), 4)
                                             for s, v in sorted(per_seed_vals.items())},
                "benefit_magnitude_by_seed": {s: round(float(np.mean(v)), 4)
                                              for s, v in sorted(mag_vals.items())},
                "benefit_geometry_mean": round(float(np.mean(
                    [x for v in per_seed_vals.values() for x in v])), 4) if per_seed_vals else None,
                "benefit_magnitude_mean": round(float(np.mean(
                    [x for v in mag_vals.values() for x in v])), 4) if mag_vals else None,
                "n_cell_g_points": sum(len(v) for v in per_seed_vals.values()),
            }

    report = {
        "experiment": "RG_admission_benefit",
        "status": "RETROSPECTIVE (spec frozen in submissions/d2-federation/canonical.md §6 "
                  "before implementation)",
        "schema_version": SCHEMA_VERSION,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "budgets": list(BUDGETS), "small_g": list(GSMALL), "gain_cut": GAIN_CUT,
        "bundles": rows, "regime_aggregates_small_g": agg,
    }
    tmp = args.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(report, f, indent=2)
    os.replace(tmp, args.out)
    print(f"wrote {args.out}\n")
    for k, v in agg.items():
        print(f"{k:<22} geometry {v['benefit_geometry_mean']:+.4f}  "
              f"magnitude {v['benefit_magnitude_mean']:+.4f}  "
              f"by-seed geo {v['benefit_geometry_by_seed']}")


if __name__ == "__main__":
    main()
