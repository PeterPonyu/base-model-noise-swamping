"""analyze_matrices.py — the G1 GATE analyzer.

Reads killgate `--save_matrices` .npz files and retires the two confounds that
make the flattened key-geometry result indefensible:

  (i)  NON-INDEPENDENCE: the flat Spearman over N*M pairs is really ~N edits x M
       reused probes, so its p-value is meaningless. We instead report
       WITHIN-PROBE and PER-EDIT correlations, whose units are independent.
  (ii) PROBE-MARGINAL LEAKAGE: AUROC/flat-rho are partly driven by "which probe
       is intrinsically fragile" (column structure). The WITHIN-PROBE Spearman
       holds probe identity fixed (correlates COS vs damage DOWN each column,
       across edits), so any surviving signal is edit-specific pairwise geometry.
       A COLUMN-PERMUTATION NULL breaks the pairing while keeping both marginals.

GATE (per the analysis plan):
  within-probe mean rho >= 0.15  AND  permutation p < 0.05  -> headline SURVIVES
  0.10 <= within-probe rho < 0.15                           -> BORDERLINE
  within-probe rho < 0.10                                   -> headline DEAD
                                                              (pivot to Qwen-null
                                                               / FT-overcollateral story)

Also runs the same for the norm-growth baseline (head-to-head) and reports the
Qwen-vs-Llama mechanism number (mean residual norm ‖v-Wk‖) if present.

Usage:
  python analyze_matrices.py results/matrices/gate_llama1b_rome_cf_L8_s*.npz \
      --metric logit --known --edit_ok --out results/gate_L8.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

RNG_SEED = 12345  # fixed so the permutation null is reproducible


def _midrank(x):
    """Tie-averaged ranks (proper Spearman ranks). argsort().argsort() assigns
     arbitrary distinct ranks to ties, biasing rho when damage has exact 0s /
    saturated logits; average-rank is the textbook-correct treatment."""
    order = x.argsort(kind="mergesort")
    ranks = np.empty(len(x), float)
    ranks[order] = np.arange(1, len(x) + 1, dtype=float)
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt)); np.add.at(sums, inv, ranks)
    return (sums / cnt)[inv]


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if a.size < 3:
        return np.nan
    ar, br = _midrank(a), _midrank(b)
    if ar.std() == 0 or br.std() == 0:
        return np.nan
    return float(np.corrcoef(ar, br)[0, 1])


def within_probe_rhos(COS, D):
    """Spearman(COS[:,j], D[:,j]) down each probe column (across edits)."""
    return np.array([spearman(COS[:, j], D[:, j]) for j in range(COS.shape[1])])


def per_edit_rhos(COS, D):
    """Spearman(COS[i,:], D[i,:]) across probes for each edit."""
    return np.array([spearman(COS[i, :], D[i, :]) for i in range(COS.shape[0])])


def boot_ci(vals, n_boot=2000, seed=RNG_SEED):
    """Bootstrap 95% CI of the mean of a vector (resampling its entries)."""
    vals = vals[np.isfinite(vals)]
    if vals.size < 3:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = [rng.choice(vals, vals.size, replace=True).mean() for _ in range(n_boot)]
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def flat_permutation_null(COS, D, n_perm=1000, seed=RNG_SEED):
    """Column-permutation null for the FLAT Spearman: shuffle probe columns of D
    (keeps both marginals). NOTE: this leaves the per-column within-probe rhos
    UNCHANGED, so it tests only the flat/marginal correlation — NOT the GATE's
    within-probe mean. Returns (obs, p_value)."""
    obs = spearman(COS.reshape(-1), D.reshape(-1))
    rng = np.random.default_rng(seed)
    M = COS.shape[1]
    ge = 0
    for _ in range(n_perm):
        perm = rng.permutation(M)
        null = spearman(COS.reshape(-1), D[:, perm].reshape(-1))
        if abs(null) >= abs(obs):
            ge += 1
    return obs, (ge + 1) / (n_perm + 1)


def within_probe_permutation_null(COS, D, obs_mean, n_perm=300, seed=RNG_SEED):
    """Null for the WITHIN-PROBE mean (the GATE statistic): independently shuffle
    the edit ordering DOWN each probe column, breaking the COS↔damage pairing
    within every probe while preserving each column's damage marginal. Recompute
    the within-probe mean under the null. Returns the empirical p-value.

    CAVEAT (why the edit-level null below exists): per-column independent shuffles
    also destroy the *cross-column* structure of a single edit (a high-norm edit
    that damages MANY probes together). That extra destruction shrinks the null
    variance, making this test anti-conservative. It is retained for continuity;
    the edit-level (single-row-permutation) null is the defensible one."""
    rng = np.random.default_rng(seed)
    M = COS.shape[1]
    ge = 0
    for _ in range(n_perm):
        Dp = D.copy()
        for j in range(M):
            rng.shuffle(Dp[:, j])
        null_mean = float(np.nanmean(within_probe_rhos(COS, Dp)))
        if abs(null_mean) >= abs(obs_mean):
            ge += 1
    return (ge + 1) / (n_perm + 1)


def editlevel_permutation_null(COS, D, obs_mean, n_perm=1000, seed=RNG_SEED):
    """STRICT exchangeable null. The independent sampling unit is the EDIT (a row).
    Apply ONE permutation of the edit order to ALL columns of D at once: this keeps
    every edit's whole damage profile intact (row structure preserved) and every
    probe's damage marginal intact (column contents preserved), and breaks ONLY the
    COS↔damage alignment. Recompute the within-probe mean under each permutation.
    This is the correct null for 'does key-cosine predict damage beyond edit-level
    and probe-level marginals'. Returns (p_value, null_mean, null_std)."""
    rng = np.random.default_rng(seed)
    N = COS.shape[0]
    ge = 0
    null_means = np.empty(n_perm)
    for t in range(n_perm):
        perm = rng.permutation(N)
        nm = float(np.nanmean(within_probe_rhos(COS, D[perm, :])))
        null_means[t] = nm
        if abs(nm) >= abs(obs_mean):
            ge += 1
    return (ge + 1) / (n_perm + 1), float(np.nanmean(null_means)), float(np.nanstd(null_means))


def edit_cluster_boot_ci(COS, D, n_boot=1000, seed=RNG_SEED):
    """Bootstrap 95% CI of the within-probe mean rho by resampling EDITS (rows)
    with replacement — the correct cluster unit. The plain boot_ci (resampling the
    per-column rho vector) treats columns as independent and understates uncertainty
    because the same edits populate every column. Returns (lo, hi)."""
    N = COS.shape[0]
    if N < 5:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for t in range(n_boot):
        idx = rng.integers(0, N, N)
        means[t] = float(np.nanmean(within_probe_rhos(COS[idx, :], D[idx, :])))
    return (float(np.nanpercentile(means, 2.5)), float(np.nanpercentile(means, 97.5)))


def analyze_one(npz_path, metric, known, edit_ok_filter, n_perm=1000):
    d = np.load(npz_path)
    COS = d["COS"].astype(float)                                   # [N,M]
    D = (d["damage_logit"] if metric == "logit" else d["damage_prob"]).astype(float)
    ng = d["norm_growth"].astype(float)                            # [N]
    NG = np.repeat(ng[:, None], COS.shape[1], axis=1)              # [N,M] broadcast
    # filters
    if edit_ok_filter and "edit_ok" in d:
        rows = d["edit_ok"].astype(float) > 0.5
        COS, D, NG = COS[rows], D[rows], NG[rows]
    if known and "pre_p" in d:
        cols = d["pre_p"].astype(float) > 0.05
        if cols.sum() >= 5:
            COS, D, NG = COS[:, cols], D[:, cols], NG[:, cols]
    wp = within_probe_rhos(COS, D)
    pe = per_edit_rhos(COS, D)
    # FIX 2026-07-01 (audit): was within_probe_rhos(COS, NG) = corr(key-cos, norm-growth),
    # a collinearity diagnostic mislabeled as the norm-growth baseline. The true head-to-head
    # baseline is corr(norm-growth, damage) down each probe column.
    wp_ng = within_probe_rhos(NG, D) if np.nanstd(ng) > 0 else np.array([np.nan])
    wp_mean = float(np.nanmean(wp))
    _, flat_p = flat_permutation_null(COS, D)
    wp_perm_p = within_probe_permutation_null(COS, D, wp_mean)   # legacy per-column null
    # STRICT edit-level null (defensible; edits are the exchangeable unit) + cluster CI.
    edit_perm_p, null_mean, null_std = editlevel_permutation_null(COS, D, wp_mean, n_perm=n_perm)
    cl_lo, cl_hi = edit_cluster_boot_ci(COS, D)
    resid = d["resid_norm"].astype(float) if "resid_norm" in d else np.array([np.nan])
    return {
        "npz": os.path.basename(npz_path),
        "shape": [int(COS.shape[0]), int(COS.shape[1])],
        "flat_spearman_INFLATED": round(spearman(COS.reshape(-1), D.reshape(-1)), 4),
        "within_probe_mean": round(wp_mean, 4),
        "within_probe_ci95": [round(x, 4) for x in boot_ci(wp)],
        "within_probe_ci95_editcluster": [round(x, 4) for x in (cl_lo, cl_hi)],
        "within_probe_frac_positive": round(float(np.nanmean(wp > 0)), 3),
        "per_edit_mean": round(float(np.nanmean(pe)), 4),
        "per_edit_ci95": [round(x, 4) for x in boot_ci(pe)],
        "within_probe_perm_p": round(wp_perm_p, 4),   # legacy per-column null (anti-conservative)
        "within_probe_perm_p_editlevel": round(edit_perm_p, 4),  # STRICT null (primary)
        "editlevel_null_mean": round(null_mean, 4),
        "editlevel_null_std": round(null_std, 4),
        "editlevel_z": (round((wp_mean - null_mean) / null_std, 2)
                        if null_std > 1e-9 else None),
        "flat_perm_p": round(flat_p, 4),               # tests the flat correlation (marginal-driven)
        "within_probe_mean_normgrowth": round(float(np.nanmean(wp_ng)), 4),
        "mean_residual_norm(S)": (None if np.all(np.isnan(resid))
                                  else round(float(np.nanmean(resid)), 4)),
    }


def verdict(within_mean, perm_p):
    if not np.isfinite(within_mean):
        return "UNDETERMINED"
    if within_mean >= 0.15 and perm_p < 0.05:
        return "PASS — headline survives (edit-specific pairwise geometry, not probe-marginal)"
    if within_mean >= 0.10:
        return "BORDERLINE — weak within-probe signal; needs more edits/probes or reframe"
    return "DEAD — within-probe rho < 0.10; demote headline, pivot to Qwen-null / FT-overcollateral"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="+", help="one or more killgate .npz (globs ok); each = one seed")
    ap.add_argument("--metric", choices=["logit", "prob"], default="logit")
    ap.add_argument("--known", action="store_true", help="filter to probes the base model knows (pre_p>0.05)")
    ap.add_argument("--edit_ok", action="store_true", help="drop failed edits (edit_ok==0)")
    ap.add_argument("--n_perm", type=int, default=1000,
                    help="permutations for the strict edit-level null (default 1000)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    paths = sorted({p for pat in args.npz for p in glob.glob(pat)})
    if not paths:
        raise SystemExit("no .npz matched")
    per_seed = [analyze_one(p, args.metric, args.known, args.edit_ok, n_perm=args.n_perm)
                for p in paths]

    wp = np.array([r["within_probe_mean"] for r in per_seed], float)
    pe = np.array([r["per_edit_mean"] for r in per_seed], float)
    pp = np.array([r["within_probe_perm_p"] for r in per_seed], float)
    ppe = np.array([r["within_probe_perm_p_editlevel"] for r in per_seed], float)
    agg = {
        "n_seeds": len(paths),
        "metric": args.metric,
        "filters": {"known": args.known, "edit_ok": args.edit_ok},
        "within_probe_mean_across_seeds": round(float(np.nanmean(wp)), 4),
        "within_probe_std_across_seeds": round(float(np.nanstd(wp)), 4),
        "per_edit_mean_across_seeds": round(float(np.nanmean(pe)), 4),
        "max_within_probe_perm_p": round(float(np.nanmax(pp)), 4),
        "max_within_probe_perm_p_editlevel": round(float(np.nanmax(ppe)), 4),
        "VERDICT": verdict(float(np.nanmean(wp)), float(np.nanmax(ppe))),
    }
    res = {"aggregate": agg, "per_seed": per_seed}
    print(json.dumps(res, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(res, open(args.out, "w"), indent=2)
        print(f"[analyze] wrote {args.out}")


if __name__ == "__main__":
    main()
