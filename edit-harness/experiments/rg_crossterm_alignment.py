#!/usr/bin/env python3
"""DESCRIPTIVE, NON-PREREGISTERED follow-up to rg_signed_reanalysis.py (2026-07-15).

Mechanism probe for the Qwen-14B constructive-merging flip: for each merge observation
(edit a in group), compute the exact received cross-term d_a = sum_b cross_term(b -> a)
and its alignment with the edit's OWN residual direction R_a:

  cos_align = cos(d_a, R_a)
  proj      = (d_a . R_a) / ||R_a||   (signed logit-relevant component)

Hypothesis (from findings-RG-SIGNED-REANALYSIS-2026-07-15.md, fact 2): the merged model
delivers W k_a ~ own-value + d_a; if d_a aligns with R_a the cross-talk REINFORCES the
edit (drop < 0), if anti-aligned it degrades it. Prediction: cos_align mass is positive
at Qwen-14B, ~0/negative elsewhere, and drop anti-correlates with proj.

Formulas imported from merging_m0.py; nothing re-derived.
"""
import argparse
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


def analyze_bundle(rg_dir, gmax=20):
    per_seed, meas, meta = load_rg(rg_dir)
    seeds = [int(s) for s in meta["seeds"]]
    gsizes = [int(g) for g in meta["group_sizes"] if int(g) <= gmax]
    obs_seed = meas["obs_seed"].astype(int); obs_g = meas["obs_g"].astype(int)
    obs_group = meas["obs_group"].astype(int); obs_edit = meas["obs_edit"].astype(int)
    obs_lp = meas["obs_logit_post"].astype(float)
    mem = defaultdict(list)
    for s, g, gr, ed in zip(meas["mem_seed"].astype(int), meas["mem_g"].astype(int),
                            meas["mem_group"].astype(int), meas["mem_edit"].astype(int)):
        mem[(s, g, gr)].append(ed)

    def rnd(x):
        return round(float(x), 4) if (x is not None and np.isfinite(x)) else None

    cells = {}
    for s in seeds:
        v = per_seed[s]
        K = v["K"].astype(float); R = v["R"].astype(float)
        denom = v["denom"].astype(float)
        logit_solo = v["logit_solo"].astype(float)
        Rn = np.linalg.norm(R, axis=1)
        for g in gsizes:
            sel = np.where((obs_seed == s) & (obs_g == g))[0]
            if sel.size == 0:
                continue
            cos_align, proj, drop = [], [], []
            for idx in sel:
                a = int(obs_edit[idx]); gr = int(obs_group[idx])
                others = [b for b in mem[(s, g, gr)] if b != a]
                if not others:
                    continue
                d_a = np.zeros(R.shape[1])
                for b in others:
                    d_a += cross_term(R[b], K[b], K[a], denom[b])
                nd = float(np.linalg.norm(d_a))
                dot = float(np.dot(d_a, R[a]))
                cos_align.append(dot / (nd * Rn[a] + 1e-12))
                proj.append(dot / (Rn[a] + 1e-12))
                drop.append(float(logit_solo[a] - obs_lp[idx]))
            ca = np.array(cos_align); pj = np.array(proj); dr = np.array(drop)
            cells[f"g{g}_s{s}"] = {
                "n_obs": int(dr.size),
                "mean_cos_align": rnd(np.mean(ca)),
                "median_cos_align": rnd(np.median(ca)),
                "frac_cos_align_pos": rnd(np.mean(ca > 0)),
                "rho_cos_align_drop": rnd(_spearman(ca, dr)),
                "rho_proj_drop": rnd(_spearman(pj, dr)),
                "frac_drop_negative": rnd(np.mean(dr < 0)),
            }
    return {"rg_dir": rg_dir, "model": meta.get("model"), "layer": meta.get("layer"),
            "seeds": seeds, "group_sizes": gsizes, "cells": cells}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(
        HARNESS, "results", "merging", "RG_crossterm_alignment_20260715.json"))
    args = ap.parse_args()
    mg = os.path.join(HARNESS, "results", "merging")
    bundles = [("Qwen2.5-14B_L36", "Qwen2.5-14B_L36_RG"),
               ("Mistral-7B-v0.3_L24", "Mistral-7B-v0.3_L24_RG"),
               ("Llama-3.2-1B_L12", "Llama-3.2-1B_L12_RG"),
               ("Qwen2.5-1.5B_L21", "Qwen2.5-1.5B_L21_RG"),
               ("Qwen2.5-1.5B_L24", "Qwen2.5-1.5B_L24_RG"),
               ("Phi-3.5-mini_L24", "Phi-3.5-mini_L24_RG"),
               ("Qwen2.5-3B_L27", "Qwen2.5-3B_L27_RG"),
               ("gemma-2-2b_L19", "gemma-2-2b_L19_RG"),
               ("Llama-3.2-3B_L21", "Llama-3.2-3B_L21_RG"),
               ("Qwen2.5-7B_L21", "Qwen2.5-7B_L21_RG"),
               ("Llama-3.1-8B_L24", "Llama-3.1-8B_L24_RG"),
               ("Phi-3.5-mini_L16", "Phi-3.5-mini_L16_RG"),
               ("Qwen2.5-3B_L18", "Qwen2.5-3B_L18_RG"),
               ("gemma-2-2b_L13", "gemma-2-2b_L13_RG"),
               ("gpt2-xl_L36", "gpt2-xl_L36_RG"),
               ("gpt2-xl_L24", "gpt2-xl_L24_RG"),
               ("Mistral-Nemo-Base-2407_L30", "Mistral-Nemo-Base-2407_L30_RG"),
               ("gemma-2-9b_L31", "gemma-2-9b_L31_RG"),
               ("gpt-neox-20b_L33", "gpt-neox-20b_L33_RG")]
    report = {"experiment": "RG_crossterm_alignment",
              "status": "DESCRIPTIVE_NON_PREREGISTERED",
              "note": ("Alignment of the exact received cross-term with the edit's own "
                       "residual direction; mechanism probe for the 14B constructive flip."),
              "schema_version": SCHEMA_VERSION,
              "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "bundles": {}}
    for name, sub in bundles:
        p = os.path.join(mg, sub)
        if not os.path.isdir(p):
            report["bundles"][name] = {"status": "MISSING_LOCALLY"}
            continue
        print(f"[{time.strftime('%H:%M:%S')}] {name} ...", flush=True)
        report["bundles"][name] = analyze_bundle(p)
    tmp = args.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(report, f, indent=2)
    os.replace(tmp, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
