#!/usr/bin/env python3
"""DESCRIPTIVE, NON-PREREGISTERED signed re-analysis of the RG merging bundles.

Context (2026-07-15): the frozen RG gate scores partial_rho(I_cos, drop | I_mag)
against a POSITIVE threshold, and I_cos is built from |cos(k_b, k_a)|. Qwen2.5-14B
L36 produced strongly NEGATIVE partials at g=2/3/5 (all 3 seeds) — structurally
invisible to the gate. This script characterizes that inversion; it does NOT alter
any frozen verdict (Qwen-14B stays INCONCLUSIVE under the prereg rule).

Per (bundle, g, seed) cell it reports, alongside the canonical |cos| metric
(recomputed and CHECKED against the on-disk tables):
  - I_cos_signed  = ||k_a|| * sum_b S_b * cos(k_b, k_a)      (signed cosine)
  - I_cos_pos/neg = ||k_a|| * sum_b S_b * max(+/-cos, 0)     (one-sided splits)
  - raw + partial (| I_mag) Spearman of each variant vs drop
  - drop sign structure: mean drop, frac(drop < 0) (merge HELPS the target logit)
  - exact cross-term norm rho (the CPU x-check, for reference)

All formulas are imported from merging_m0.py — nothing is re-derived here.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from collections import defaultdict

from merging_m0 import (_spearman, partial_spearman_multi, cross_term, load_rg,
                        SCHEMA_VERSION)

HARNESS = os.path.dirname(HERE)


def cell_stats(v, obs_edit, obs_group, obs_logit_post, members):
    K = v["K"].astype(float); R = v["R"].astype(float)
    denom = v["denom"].astype(float); S = v["S"].astype(float)
    key_norm = v["key_norm"].astype(float)
    logit_solo = v["logit_solo"].astype(float)

    I_abs, I_sgn, I_pos, I_neg, I_mag, normdelta, drop = [], [], [], [], [], [], []
    for a, g, lp in zip(obs_edit, obs_group, obs_logit_post):
        a, g = int(a), int(g)
        others = [b for b in members[g] if b != a]
        if others:
            ob = np.array(others)
            coses = np.array([float(np.dot(K[b], K[a])) /
                              (key_norm[b] * key_norm[a] + 1e-12) for b in others])
            w = S[ob] * key_norm[a]
            I_abs.append(float(np.sum(w * np.abs(coses))))
            I_sgn.append(float(np.sum(w * coses)))
            I_pos.append(float(np.sum(w * np.clip(coses, 0, None))))
            I_neg.append(float(np.sum(w * np.clip(-coses, 0, None))))
            I_mag.append(float(np.sum(w)))
            d_a = np.zeros(R.shape[1])
            for b in others:
                d_a += cross_term(R[b], K[b], K[a], denom[b])
            normdelta.append(float(np.linalg.norm(d_a)))
        else:
            for L in (I_abs, I_sgn, I_pos, I_neg, I_mag, normdelta):
                L.append(0.0)
        drop.append(float(logit_solo[a] - lp))

    I_abs = np.array(I_abs); I_sgn = np.array(I_sgn); I_pos = np.array(I_pos)
    I_neg = np.array(I_neg); I_mag = np.array(I_mag); drop = np.array(drop)
    normdelta = np.array(normdelta)

    def rnd(x):
        return round(float(x), 4) if (x is not None and np.isfinite(x)) else None

    out = {"n_obs": int(drop.size),
           "mean_drop_logit": rnd(np.mean(drop)) if drop.size else None,
           "median_abs_drop_logit": rnd(np.median(np.abs(drop))) if drop.size else None,
           "frac_drop_negative": rnd(np.mean(drop < 0)) if drop.size else None}
    for tag, x in (("abscos", I_abs), ("signedcos", I_sgn),
                   ("poscos", I_pos), ("negcos", I_neg)):
        out[f"rho_{tag}"] = rnd(_spearman(x, drop))
        out[f"partial_{tag}_given_mag"] = rnd(partial_spearman_multi(x, drop, [I_mag]))
    out["rho_abscos_normdelta_xcheck"] = rnd(_spearman(I_abs, normdelta))
    return out


def analyze_bundle(rg_dir, table_path=None):
    per_seed, meas, meta = load_rg(rg_dir)
    seeds = [int(s) for s in meta["seeds"]]
    gsizes = [int(g) for g in meta["group_sizes"]]
    obs_seed = meas["obs_seed"].astype(int); obs_g = meas["obs_g"].astype(int)
    obs_group = meas["obs_group"].astype(int); obs_edit = meas["obs_edit"].astype(int)
    obs_lp = meas["obs_logit_post"].astype(float)
    mem_seed = meas["mem_seed"].astype(int); mem_g = meas["mem_g"].astype(int)
    mem_group = meas["mem_group"].astype(int); mem_edit = meas["mem_edit"].astype(int)

    members_all = defaultdict(list)
    for s, g, gr, ed in zip(mem_seed, mem_g, mem_group, mem_edit):
        members_all[(int(s), int(g), int(gr))].append(int(ed))

    # canonical table for the reproduction check (loader/metric validation)
    canon = None
    if table_path and os.path.exists(table_path):
        with open(table_path) as f:
            canon = json.load(f)["cells"]

    cells, checks = {}, []
    for s in seeds:
        for g in gsizes:
            sel = np.where((obs_seed == s) & (obs_g == g))[0]
            if sel.size == 0:
                continue
            members = {int(gr): members_all[(s, g, int(gr))]
                       for gr in set(obs_group[sel].tolist())}
            st = cell_stats(per_seed[s], obs_edit[sel], obs_group[sel],
                            obs_lp[sel], members)
            key = f"g{g}_s{s}"
            cells[key] = st
            if canon and key in canon and canon[key].get("rho_I_cos_drop") is not None:
                ok = abs(st["rho_abscos"] - canon[key]["rho_I_cos_drop"]) < 5e-4
                checks.append((key, ok, st["rho_abscos"], canon[key]["rho_I_cos_drop"]))
    n_bad = sum(1 for _, ok, _, _ in checks if not ok)
    return {
        "rg_dir": rg_dir,
        "model": meta.get("model"), "layer": meta.get("layer"),
        "seeds": seeds, "group_sizes": gsizes,
        "reproduction_check": {
            "n_checked": len(checks), "n_mismatch": n_bad,
            "status": ("PASS" if checks and n_bad == 0 else
                       "FAIL" if n_bad else "NO_CANONICAL_TABLE"),
            "mismatches": [c for c in checks if not c[1]][:5],
        },
        "cells": cells,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(
        HARNESS, "results", "merging", "RG_signed_reanalysis_20260715.json"))
    args = ap.parse_args()

    mg = os.path.join(HARNESS, "results", "merging")
    bundles = [
        ("Llama-3.2-1B_L8",  os.path.join(mg, "Llama-3.2-1B_L8_RG"),
         os.path.join(mg, "RG_operating_curve_table_L8.json")),
        ("Llama-3.2-1B_L12", os.path.join(mg, "Llama-3.2-1B_L12_RG"),
         os.path.join(mg, "RG_operating_curve_table.json")),
        ("Llama-3.2-1B_L14", os.path.join(mg, "Llama-3.2-1B_L14_RG"),
         os.path.join(mg, "RG_operating_curve_table_L14.json")),
        ("Qwen2.5-1.5B_L14", os.path.join(mg, "Qwen2.5-1.5B_L14_RG"),
         os.path.join(mg, "RG_operating_curve_table_qwen15b_L14.json")),
        ("Qwen2.5-1.5B_L21", os.path.join(mg, "Qwen2.5-1.5B_L21_RG"),
         os.path.join(mg, "RG_operating_curve_table_qwen15b_L21.json")),
        ("Qwen2.5-1.5B_L24", os.path.join(mg, "Qwen2.5-1.5B_L24_RG"),
         os.path.join(mg, "RG_operating_curve_table_qwen15b_L24.json")),
        ("Mistral-7B-v0.3_L24", os.path.join(mg, "Mistral-7B-v0.3_L24_RG"),
         os.path.join(mg, "Mistral-7B-v0.3_L24_RG", "RG_operating_curve_table.json")),
        ("Qwen2.5-7B_L21", os.path.join(mg, "Qwen2.5-7B_L21_RG"),
         os.path.join(mg, "Qwen2.5-7B_L21_RG", "RG_operating_curve_table.json")),
        ("Llama-3.1-8B_L24", os.path.join(mg, "Llama-3.1-8B_L24_RG"),
         os.path.join(mg, "Llama-3.1-8B_L24_RG", "RG_operating_curve_table.json")),
        ("Qwen2.5-14B_L36", os.path.join(mg, "Qwen2.5-14B_L36_RG"),
         os.path.join(mg, "RG_operating_curve_table_qwen25_14b_L36.json")),
        ("gemma-2-2b_L19", os.path.join(mg, "gemma-2-2b_L19_RG"),
         os.path.join(mg, "RG_operating_curve_table_gemma2b_L19.json")),
        ("Llama-3.2-3B_L21", os.path.join(mg, "Llama-3.2-3B_L21_RG"),
         os.path.join(mg, "RG_operating_curve_table_llama3b_L21.json")),
        ("Qwen2.5-3B_L27", os.path.join(mg, "Qwen2.5-3B_L27_RG"),
         os.path.join(mg, "RG_operating_curve_table_qwen3b_L27.json")),
        ("Phi-3.5-mini_L24", os.path.join(mg, "Phi-3.5-mini_L24_RG"),
         os.path.join(mg, "RG_operating_curve_table_phi35_L24.json")),
        ("Phi-3.5-mini_L16", os.path.join(mg, "Phi-3.5-mini_L16_RG"),
         os.path.join(mg, "RG_operating_curve_table_phi35_L16.json")),
        ("Qwen2.5-3B_L18", os.path.join(mg, "Qwen2.5-3B_L18_RG"),
         os.path.join(mg, "RG_operating_curve_table_qwen3b_L18.json")),
        ("gemma-2-2b_L13", os.path.join(mg, "gemma-2-2b_L13_RG"),
         os.path.join(mg, "RG_operating_curve_table_gemma2b_L13.json")),
        ("gpt2-xl_L36", os.path.join(mg, "gpt2-xl_L36_RG"),
         os.path.join(mg, "RG_operating_curve_table_gpt2xl_L36.json")),
        ("gpt2-xl_L24", os.path.join(mg, "gpt2-xl_L24_RG"),
         os.path.join(mg, "RG_operating_curve_table_gpt2xl_L24.json")),
    ]
    report = {
        "experiment": "RG_signed_reanalysis",
        "status": "DESCRIPTIVE_NON_PREREGISTERED",
        "note": ("Signed/one-sided cosine decomposition of the RG merging bundles. "
                 "Does not alter any frozen gate verdict; canonical |cos| rho is "
                 "recomputed and checked against the on-disk tables per cell."),
        "schema_version": SCHEMA_VERSION,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "bundles": {},
    }
    for name, rg_dir, table in bundles:
        if not os.path.isdir(rg_dir):
            report["bundles"][name] = {"status": "MISSING_LOCALLY"}
            continue
        print(f"[{time.strftime('%H:%M:%S')}] analyzing {name} ...", flush=True)
        report["bundles"][name] = analyze_bundle(rg_dir, table)
        rc = report["bundles"][name]["reproduction_check"]
        print(f"  reproduction_check: {rc['status']} "
              f"({rc['n_checked']} cells, {rc['n_mismatch']} mismatch)", flush=True)

    tmp = args.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(report, f, indent=2)
    os.replace(tmp, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
