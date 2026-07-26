"""route_per_edit_sweep.py — D3 geometry-gated routing at PER-EDIT granularity.

Complements geometry_router.py (which routes whole model/layer configs): here each
EDIT is routed by its own pre-edit geometry signal s_i = mean_j |COS[i,j]| over the
known-probe columns, and the routing threshold is swept so the artifact is a
tunable damage-vs-compute CURVE rather than a single operating point. PURE CPU /
numpy on the saved gate/g4 matrices — NO torch, NO model load, NO network.

Per matched (model,dataset,layer,seed) ROME<->AlphaEdit pair, for each threshold
quantile q (t = quantile_q of s over the shared edit rows):
    route edit i -> AlphaEdit if s_i >= t else vanilla ROME (projection skipped).
Reported per point:
  - frac_alpha            fraction of edits paying the projection compute
  - routed_mean_damage    mean signed damage_logit under the routed policy
  - capture_frac          (always_rome - routed) / (always_rome - always_alpha):
                          the fraction of AlphaEdit's total achievable damage
                          reduction captured while only routing frac_alpha of edits.
                          capture_frac > frac_alpha  <=>  geometry targets the
                          projection at the edits that need it (the D3 claim).
  - routed_edit_success   efficacy under the routed policy (edit_ok of the editor
                          each edit was routed to, over ALL edits, no edit_ok filter)
  - bootstrap CI (edit-level resample) on routed_mean_damage
Damage stats use the analyze_g4 shared-mask convention (both editors' edit_ok when
--edit_ok, base-model pre_p>0.05 known columns). Metric = mean SIGNED damage_logit,
never AUROC.

PROVENANCE CAVEAT (carried into the output JSON): all g4 alpha matrices currently
on disk use the probes-fit projector (alpha_proj_source='probes'); per the C4
circularity fix the curve must be re-derived from the E6 holdout/generic matrices
before it is used as a primary paper number.

Usage:
  python route_per_edit_sweep.py \
      --gate_glob 'results/matrices/gate_llama1b_rome_cf_L*_s*.npz' \
      --alpha_glob 'results/matrices/g4_llama1b_alpha_cf_L*_s*.npz' \
      --known --edit_ok --out results/D3_routing_per_edit_sweep.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from geometry_router import GATE_RE, ALPHA_RE, parse_name, masks, r4

Q_GRID = [round(q, 2) for q in np.arange(0.0, 1.0001, 0.05)]


def row_means(A, cmask):
    """Per-edit mean over the known-probe columns, finite-safe."""
    sub = A[:, cmask]
    with np.errstate(invalid="ignore"):
        m = np.nanmean(np.where(np.isfinite(sub), sub, np.nan), axis=1)
    return m


def bootstrap_ci(vals, n_boot=1000, seed=0):
    vals = vals[np.isfinite(vals)]
    if vals.size < 5:
        return None, None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, vals.size, size=(n_boot, vals.size))
    means = vals[idx].mean(axis=1)
    return r4(np.percentile(means, 2.5)), r4(np.percentile(means, 97.5))


def sweep_config(dg, da, known, edit_ok):
    rmask, cmask = masks(dg, known, edit_ok)
    srow = rmask.copy()
    if edit_ok and "edit_ok" in da.files:
        srow = srow & (da["edit_ok"].astype(float) > 0.5)

    s = row_means(np.abs(dg["COS"].astype(float)), cmask)            # pre-edit signal
    d_rome = row_means(dg["damage_logit"].astype(float), cmask)      # per-edit damage
    d_alpha = row_means(da["damage_logit"].astype(float), cmask)

    s_sh, dr_sh, da_sh = s[srow], d_rome[srow], d_alpha[srow]
    ok_r = dg["edit_ok"].astype(float) if "edit_ok" in dg.files else np.ones_like(s)
    ok_a = da["edit_ok"].astype(float) if "edit_ok" in da.files else np.ones_like(s)

    always_rome, always_alpha = float(np.nanmean(dr_sh)), float(np.nanmean(da_sh))
    span = always_rome - always_alpha
    points = []
    for q in Q_GRID:
        t = float(np.quantile(s_sh, q)) if q < 1.0 else float(np.max(s_sh)) + 1e-9
        to_alpha = s_sh >= t
        routed = np.where(to_alpha, da_sh, dr_sh)
        routed_mean = float(np.nanmean(routed))
        lo, hi = bootstrap_ci(routed)
        # efficacy over ALL edits (no edit_ok filter — that's the thing being measured)
        to_alpha_all = s >= t
        succ = float(np.nanmean(np.where(to_alpha_all, ok_a, ok_r)))
        points.append({
            "q": q, "threshold": r4(t),
            "frac_alpha": r4(float(to_alpha.mean())),
            "routed_mean_damage": r4(routed_mean),
            "routed_damage_ci95": [lo, hi],
            "capture_frac": r4((always_rome - routed_mean) / span) if abs(span) > 1e-9 else None,
            "routed_edit_success": r4(succ),
        })
    return {
        "n_edits_shared": int(srow.sum()), "n_probes_masked": int(cmask.sum()),
        "always_rome_mean_damage": r4(always_rome),
        "always_alpha_mean_damage": r4(always_alpha),
        "per_edit_signal_spearman_vs_rome_damage": r4(_spearman(s_sh, dr_sh)),
        "curve": points,
    }


def _spearman(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if a.size < 5:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gate_glob", required=True)
    ap.add_argument("--alpha_glob", required=True)
    ap.add_argument("--known", action="store_true")
    ap.add_argument("--edit_ok", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    alpha_index = {}
    for p in sorted(set(glob.glob(args.alpha_glob))):
        key = parse_name(p, ALPHA_RE)
        if key is not None:
            alpha_index[key] = p

    by_config, proj_sources = [], set()
    for gp in sorted(set(glob.glob(args.gate_glob))):
        key = parse_name(gp, GATE_RE)
        if key is None or key not in alpha_index:
            continue
        model, dataset, layer, seed = key
        dg, da = np.load(gp), np.load(alpha_index[key])
        if dg["COS"].shape != da["damage_logit"].shape:
            continue
        if "alpha_proj_source" in da.files:
            proj_sources.add(str(da["alpha_proj_source"]))
        row = {"config": f"{model}_{dataset}_L{layer}_s{seed}",
               "model": model, "dataset": dataset, "layer": layer, "seed": seed}
        row.update(sweep_config(dg, da, args.known, args.edit_ok))
        by_config.append(row)

    if not by_config:
        raise SystemExit("no matched gate/alpha pairs")

    # per-layer cross-seed mean curve (grids share Q_GRID so points align by q)
    layers = sorted({r["layer"] for r in by_config})
    per_layer = {}
    for L in layers:
        rows = [r for r in by_config if r["layer"] == L]
        curve = []
        for i, q in enumerate(Q_GRID):
            pts = [r["curve"][i] for r in rows]
            curve.append({
                "q": q,
                "frac_alpha": r4(np.mean([p["frac_alpha"] for p in pts])),
                "routed_mean_damage": r4(np.mean([p["routed_mean_damage"] for p in pts])),
                "capture_frac": r4(np.mean([p["capture_frac"] for p in pts if p["capture_frac"] is not None])),
                "routed_edit_success": r4(np.mean([p["routed_edit_success"] for p in pts])),
            })
        per_layer[str(L)] = {
            "seeds": sorted(r["seed"] for r in rows),
            "always_rome_mean_damage": r4(np.mean([r["always_rome_mean_damage"] for r in rows])),
            "always_alpha_mean_damage": r4(np.mean([r["always_alpha_mean_damage"] for r in rows])),
            "per_edit_signal_spearman_vs_rome_damage": r4(np.mean(
                [r["per_edit_signal_spearman_vs_rome_damage"] for r in rows])),
            "mean_curve": curve,
        }

    res = {
        "statistic": "per-edit geometry-gated routing sweep; routed edit -> AlphaEdit iff "
                     "row-mean|COS| >= quantile-q threshold; damage = mean signed damage_logit",
        "filters": {"known": args.known, "edit_ok": args.edit_ok},
        "alpha_proj_source": sorted(proj_sources),
        "provenance_caveat": "probes-fit projector (circular for primary claims); re-derive "
                             "from E6 holdout/generic matrices before publication use",
        "q_grid": Q_GRID,
        "per_layer": per_layer,
        "by_config": by_config,
    }
    print(json.dumps({"per_layer": {k: {kk: vv for kk, vv in v.items() if kk != "mean_curve"}
                                    for k, v in per_layer.items()}}, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(res, open(args.out, "w"), indent=2)
        print(f"[route-sweep] wrote {args.out}")


if __name__ == "__main__":
    main()
