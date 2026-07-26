"""aggregate_g4_causal.py — C4 AlphaEdit causal aggregation (all layers, all seeds).

The causal question: does AlphaEdit REMOVE collateral damage in proportion to the
pre-edit key-cosine? For matched (edit, probe) pairs (ROME and AlphaEdit at the SAME
model/layer/seed share an identical COS matrix, edit bank, and probe bank), we form

    damage_removed = rome_damage - alpha_damage

and ask whether the removal concentrates on high-cosine probes. The SIGNED
within-probe partialled Spearman(key-cos, damage_removed) is the primary statistic
(NOT AUROC — a probe-marginal artifact). We also report the per-key-cosine QUARTILE
mean damage-removed (pooled across seeds) and the mean SIGNED damage for ROME and
AlphaEdit separately.

Key scientific line: at L14 (the norm-growth-dominant regime) does AlphaEdit still
track cosine? (Handled gracefully — layers with no matched AlphaEdit matrices are
skipped and noted, never crash.)

CPU-only. numpy on existing .npz. No GPU / torch / downloads.

Usage (verbatim from run_deep_until1900.sh):
  python experiments/aggregate_g4_causal.py \
    --rome_glob 'results/matrices/gate_llama1b_rome_cf_L{L}_s*.npz' \
    --alpha_glob 'results/matrices/g4_llama1b_alpha_cf_L{L}_s*.npz' \
    --layers 8 10 12 14 --known --edit_ok \
    --out results/C4_causal_table.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    # reuse the project's canonical signed within-probe metric
    from analyze_matrices import spearman, within_probe_rhos  # noqa: E402
except Exception:  # pragma: no cover - fallback replica if import path breaks
    def spearman(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        m = np.isfinite(a) & np.isfinite(b)
        a, b = a[m], b[m]
        if a.size < 3:
            return np.nan
        ar = a.argsort().argsort().astype(float)
        br = b.argsort().argsort().astype(float)
        if ar.std() == 0 or br.std() == 0:
            return np.nan
        return float(np.corrcoef(ar, br)[0, 1])

    def within_probe_rhos(COS, D):
        return np.array([spearman(COS[:, j], D[:, j]) for j in range(COS.shape[1])])


SEED_RE = re.compile(r"_s(\d+)\.npz$")


def seed_of(path):
    m = SEED_RE.search(os.path.basename(path))
    return int(m.group(1)) if m else None


def seed_map(pattern):
    """glob -> {seed: path} (last one wins if duplicate seeds)."""
    out = {}
    for p in sorted(glob.glob(pattern)):
        s = seed_of(p)
        if s is not None:
            out[s] = p
    return out


def proj_source_of(da):
    """Projector source recorded in the AlphaEdit npz ('probes'|'holdout'|'generic').
    Older npz without the key predate the disjoint-projector fix -> 'probes' (the
    by-construction default that was in force then)."""
    if "alpha_proj_source" in da.files:
        return str(da["alpha_proj_source"])
    return "probes"


def masked_pair(dr, da, known, edit_ok):
    """Return (cos, dmg_rome, dmg_alpha) both flattened AND 2D-masked views.

    ROME/AlphaEdit at matched model/layer/seed share COS, edits, probes. edit_ok is
    editor-specific, so we require BOTH editors to have succeeded on an edit (shared
    row mask). pre_p is base-model geometry -> identical across editors (shared cols).
    Returns None on shape mismatch or too few pairs.
    """
    COS = dr["COS"].astype(float)
    Dr = dr["damage_logit"].astype(float)
    Da = da["damage_logit"].astype(float)
    if not (COS.shape == Dr.shape == Da.shape):
        return None
    row = np.ones(COS.shape[0], bool)
    if edit_ok and "edit_ok" in dr.files and "edit_ok" in da.files:
        row = (dr["edit_ok"].astype(float) > 0.5) & (da["edit_ok"].astype(float) > 0.5)
    col = np.ones(COS.shape[1], bool)
    if known and "pre_p" in dr.files:
        c = dr["pre_p"].astype(float) > 0.05
        if c.sum() >= 5:
            col = c
    C2 = COS[row][:, col]
    R2 = Dr[row][:, col]
    A2 = Da[row][:, col]
    if C2.size < 20:
        return None
    return C2, R2, A2


def analyze_layer(rome_glob, alpha_glob, L, known, edit_ok, proj_source=None):
    rmap = seed_map(rome_glob.replace("{L}", str(L)))
    amap = seed_map(alpha_glob.replace("{L}", str(L)))
    seeds = sorted(set(rmap) & set(amap))
    if not seeds:
        return None, f"L{L}: no matched ROME/AlphaEdit seeds (rome={sorted(rmap)}, alpha={sorted(amap)}) — skipped"

    cos_pool, rem_pool, rome_pool, alpha_pool = [], [], [], []
    wp_removed_per_seed = []
    used = []
    proj_srcs = set()
    for s in seeds:
        dr = np.load(rmap[s])
        da = np.load(amap[s])
        src = proj_source_of(da)
        if proj_source is not None and src != proj_source:
            continue  # honest-aggregation filter: keep only the requested projector source
        proj_srcs.add(src)
        got = masked_pair(dr, da, known, edit_ok)
        if got is None:
            continue
        C2, R2, A2 = got
        removed2 = R2 - A2
        cos_pool.append(C2.reshape(-1))
        rem_pool.append(removed2.reshape(-1))
        rome_pool.append(R2.reshape(-1))
        alpha_pool.append(A2.reshape(-1))
        # signed within-probe Spearman(key-cos, damage_removed), per seed (probes differ
        # across seeds -> average per-seed within-probe means, do NOT stack columns).
        wp_removed_per_seed.append(float(np.nanmean(within_probe_rhos(C2, removed2))))
        used.append(s)

    if not used:
        return None, f"L{L}: matched seeds present but no usable pairs after filtering — skipped"

    cos = np.concatenate(cos_pool)
    rem = np.concatenate(rem_pool)
    rome = np.concatenate(rome_pool)
    alpha = np.concatenate(alpha_pool)

    # per-key-cosine QUARTILE mean damage-removed (pooled across seeds)
    qs = np.quantile(cos, [0.25, 0.5, 0.75])
    bins = np.digitize(cos, qs)  # 0..3 low->high cosine
    quartile_means = []
    for q in range(4):
        m = bins == q
        quartile_means.append({
            "cosine_quartile": ["Q1(low)", "Q2", "Q3", "Q4(high)"][q],
            "n_pairs": int(m.sum()),
            "mean_cos": round(float(cos[m].mean()), 4) if m.any() else None,
            "mean_damage_removed": round(float(rem[m].mean()), 5) if m.any() else None,
            "mean_damage_rome": round(float(rome[m].mean()), 5) if m.any() else None,
            "mean_damage_alpha": round(float(alpha[m].mean()), 5) if m.any() else None,
        })

    top = quartile_means[3]["mean_damage_removed"]
    bot = quartile_means[0]["mean_damage_removed"]
    removed_top_vs_bottom_ratio = (round(top / bot, 3)
                                   if (top is not None and bot is not None and abs(bot) > 1e-6)
                                   else None)

    res = {
        "layer": L,
        "n_pairs": int(cos.size),
        "seeds_used": used,
        "proj_sources": sorted(proj_srcs),
        "within_probe_spearman": round(float(np.nanmean(wp_removed_per_seed)), 4),
        "within_probe_spearman_per_seed": [round(x, 4) for x in wp_removed_per_seed],
        "mean_damage_rome": round(float(rome.mean()), 5),
        "mean_damage_alpha": round(float(alpha.mean()), 5),
        "mean_damage_removed": round(float(rem.mean()), 5),
        "quartile_means": quartile_means,
        "removed_top_vs_bottom_ratio": removed_top_vs_bottom_ratio,
    }
    return res, None


def main():
    ap = argparse.ArgumentParser(description="C4 AlphaEdit causal aggregation across layers/seeds.")
    ap.add_argument("--rome_glob", required=True, help="glob with {L} placeholder for the ROME npz")
    ap.add_argument("--alpha_glob", required=True, help="glob with {L} placeholder for the AlphaEdit npz")
    ap.add_argument("--layers", type=int, nargs="+", required=True)
    ap.add_argument("--known", action="store_true", help="restrict to probes the base model knows (pre_p>0.05)")
    ap.add_argument("--edit_ok", action="store_true", help="restrict to edits BOTH editors succeeded on")
    ap.add_argument("--proj_source", choices=["probes", "holdout", "generic"], default=None,
                    help="keep only AlphaEdit npz whose projector was fit on this source. "
                         "'holdout'/'generic' = the honest causal test (projector disjoint from "
                         "the measured probes); default = all sources pooled.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    layers = {}
    notes = []
    for L in args.layers:
        res, note = analyze_layer(args.rome_glob, args.alpha_glob, L, args.known, args.edit_ok,
                                  proj_source=args.proj_source)
        if note:
            notes.append(note)
        if res is not None:
            layers[str(L)] = res

    out = {
        "metric": "damage_logit",
        "statistic": "signed within-probe partialled Spearman(key-cos, damage_removed); quartile mean damage-removed",
        "filters": {"known": args.known, "edit_ok": args.edit_ok, "proj_source": args.proj_source},
        "rome_glob": args.rome_glob,
        "alpha_glob": args.alpha_glob,
        "layers": layers,
        "layers_skipped": [L for L in args.layers if str(L) not in layers],
        "notes": notes,
    }
    print(json.dumps(out, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=2)
        print(f"[c4] wrote {args.out}")


if __name__ == "__main__":
    main()
