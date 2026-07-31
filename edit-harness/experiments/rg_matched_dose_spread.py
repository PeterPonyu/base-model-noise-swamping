#!/usr/bin/env python3
"""RG matched-dose-spread table (deposit artifact, H9 of PLAN-GAP-CLOSURE-MASTER-2026-07-31).

Per-cell median response-per-dose (|drop| / rel_dose) inside the dose band common to
all protocol cells, plus the max pairwise matched-band response ratio. Convention
reverse-engineered from RG_matched_dose_spread_20260716.json and verified to 6 s.f.
against every non-Phi cell (2026-07-31):

  common band      = [max over cells of q10(positive rel_dose),
                      min over cells of q90(positive rel_dose)]
  pairwise band    = same q10/q90 rule applied to the pair
  response         = median(|drop| / rel_dose) over in-band observations
  eligibility      = >= 30 in-band observations per cell (per pair)

2026-07-31 regeneration: the two Phi-3.5 bundles are the TOKENIZER-REFIXED ones
(findings-PHI35-TOKENIZER-COLLISION-2026-07-30); every other bundle is byte-identical
to 2026-07-16. The 22-cell protocol set is frozen (the Llama-2-13b L30 boundary cell
arrived later and is reported as an addendum field only).

CPU-only. Usage: python3 experiments/rg_matched_dose_spread.py [--tag REFIX20260731]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HARNESS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HARNESS / "experiments"))
from rg_gain_law import load_rg, cross_term  # noqa: E402

PROTOCOL_22 = [  # frozen 2026-07-16 set (order = old artifact's sorted keys)
    "Llama-3.1-8B_L24_RG", "Llama-3.2-1B_L12_RG", "Llama-3.2-1B_L14_RG", "Llama-3.2-1B_L8_RG",
    "Llama-3.2-3B_L21_RG", "Mistral-7B-v0.3_L24_RG", "Mistral-Nemo-Base-2407_L30_RG",
    "Phi-3.5-mini_L16_RG", "Phi-3.5-mini_L24_RG",
    "Qwen2.5-1.5B_L14_RG", "Qwen2.5-1.5B_L21_RG", "Qwen2.5-1.5B_L24_RG", "Qwen2.5-14B_L36_RG",
    "Qwen2.5-3B_L18_RG", "Qwen2.5-3B_L27_RG", "Qwen2.5-7B_L21_RG",
    "gemma-2-2b_L13_RG", "gemma-2-2b_L19_RG", "gemma-2-9b_L31_RG",
    "gpt-neox-20b_L33_RG", "gpt2-xl_L24_RG", "gpt2-xl_L36_RG",
]
ADDENDUM_CELLS = ["Llama-2-13b-hf_L30_RG"]  # post-freeze boundary replication; never in the 22


def rd_dr(rg_dir: str, gmax: int = 20):
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
    return np.array(RD), np.array(DR)


def response(RD, DR, lo, hi):
    m = (RD >= lo) & (RD <= hi)
    if m.sum() < 30:
        return None, int(m.sum())
    return float(np.median(np.abs(DR[m]) / RD[m])), int(m.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="REFIX20260731")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or str(HARNESS / "results" / "merging" / f"RG_matched_dose_spread_{args.tag}.json")

    cache = {}
    for c in PROTOCOL_22 + ADDENDUM_CELLS:
        d = HARNESS / "results" / "merging" / c
        if not d.is_dir():
            raise SystemExit(f"missing bundle {d}")
        cache[c] = rd_dr(str(d))

    q10 = {c: float(np.percentile(RD[RD > 0], 10)) for c, (RD, _) in cache.items()}
    q90 = {c: float(np.percentile(RD[RD > 0], 90)) for c, (RD, _) in cache.items()}
    band = [max(q10[c] for c in PROTOCOL_22), min(q90[c] for c in PROTOCOL_22)]
    band_cells = {"lo": max(PROTOCOL_22, key=lambda c: q10[c]),
                  "hi": min(PROTOCOL_22, key=lambda c: q90[c])}

    per_cell, n_in_band = {}, {}
    for c in PROTOCOL_22:
        r, n = response(*cache[c], *band)
        if r is None:
            raise SystemExit(f"{c}: only {n} obs in common band — protocol broken")
        per_cell[c], n_in_band[c] = r, n
    vals = np.array(list(per_cell.values()))
    spread = float(vals.max() / vals.min())

    best = (0.0, None)
    for i, a in enumerate(PROTOCOL_22):
        for b in PROTOCOL_22[i + 1:]:
            lo, hi = max(q10[a], q10[b]), min(q90[a], q90[b])
            if lo >= hi:
                continue
            ra, na = response(*cache[a], lo, hi)
            rb, nb = response(*cache[b], lo, hi)
            if ra is None or rb is None:
                continue
            r = max(ra, rb) / min(ra, rb)
            if r > best[0]:
                best = (r, (a, b, lo, hi, na, nb))
    _, (pa, pb, plo, phi, pna, pnb) = best

    addendum = {}
    for c in ADDENDUM_CELLS:
        r, n = response(*cache[c], *band)
        addendum[c] = {"response_in_common_band": r, "n_in_band": n}

    doc = {
        "experiment": "RG_matched_dose_spread",
        "tag": args.tag,
        "convention": "common band = [max q10, min q90] of positive rel_dose across protocol cells; "
                      "response = median(|drop|/rel_dose) in-band; >=30 obs eligibility",
        "protocol_cells": 22,
        "phi_bundles": "tokenizer-REFIXED (2026-07-30); all other bundles byte-identical to 2026-07-16",
        "common_band": band,
        "common_band_endpoint_cells": band_cells,
        "per_cell_response_in_band": per_cell,
        "n_obs_in_band": n_in_band,
        "spread_max_over_min": spread,
        "spread_argmax": max(per_cell, key=per_cell.get),
        "spread_argmin": min(per_cell, key=per_cell.get),
        "max_pairwise_matched_dose_ratio": float(best[0]),
        "max_pair": {"cells": [pa, pb], "band": [float(plo), float(phi)], "n_obs": [pna, pnb]},
        "addendum_post_freeze_cells": addendum,
        "supersedes": "RG_matched_dose_spread_20260716.json (pre-Phi-refix)",
    }
    with open(out, "w") as f:
        json.dump(doc, f, indent=1)
    print(f"wrote {out}")
    print(f"common band [{band[0]:.5f}, {band[1]:.5f}] (lo: {band_cells['lo']}, hi: {band_cells['hi']})")
    print(f"spread {spread:.1f}x (max {vals.max():.3f} {doc['spread_argmax']}, "
          f"min {vals.min():.4f} {doc['spread_argmin']})")
    print(f"max pairwise {best[0]:.2f}x {pa} vs {pb}")
    for c in PROTOCOL_22:
        if "Phi" in c:
            print(f"  {c}: {per_cell[c]:.4f} (n={n_in_band[c]})")


if __name__ == "__main__":
    main()
