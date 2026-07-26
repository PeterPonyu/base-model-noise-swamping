"""cp_edit/conformal.py — split-conformal upper bounds + bootstrap harness.

Split-conformal (one-sided upper bound on per-edit signed damage y_i):
  Nonconformity (frozen in CP_EDIT_PREREG.json):
    * normalized predictors (SxC, keycos, NG): r_i = y_i / max(p_i, EPS)
      -> upper bound U_i = q_hat * max(p_i, EPS)
    * marginal (constant p_i=1): r_i = y_i ; U_i = q_hat  (constant width)
  q_hat = k-th smallest calibration score, k = ceil((n_cal+1) * (1 - alpha)).
  Coverage(test) = mean_i [ y_i <= U_i ];  Width W = mean_i U_i.

Splits are SEED-STRATIFIED (no request-level leakage — seeds edit disjoint
requests, verified against killgate loader). Each replicate partitions each
seed's edits ~50/50 without replacement.

RNG: base seed RNG_SEED = 12345 (LAB NON-NEGOTIABLE: "RNG_SEED=12345 for
permutation nulls"; also used as the bootstrap base). This overrides the spec's
suggested 20260701 per the binding lab rule; recorded verbatim in the prereg.

CPU only, numpy only.
"""
from __future__ import annotations

import numpy as np

RNG_SEED = 12345
EPS = 1e-8
ALPHA = 0.10  # target 0.90 coverage


def q_hat_from_cal(r_cal, alpha=ALPHA):
    """k-th smallest calibration score, k = ceil((n_cal+1)*(1-alpha))."""
    n = r_cal.shape[0]
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    k = min(max(k, 1), n)           # clamp: if k>n, use max (finite-sample: no bound)
    return float(np.sort(r_cal)[k - 1]), k


def split_cp(y_cal, p_cal, y_test, p_test, normalized, alpha=ALPHA):
    """One split-CP fit. Returns (coverage, width, q_hat, k)."""
    if normalized:
        r_cal = y_cal / np.maximum(p_cal, EPS)
    else:
        r_cal = y_cal.copy()
    qh, k = q_hat_from_cal(r_cal, alpha)
    if normalized:
        U = qh * np.maximum(p_test, EPS)
    else:
        U = np.full_like(y_test, qh)
    cov = float(np.mean(y_test <= U))
    width = float(np.mean(U))
    return cov, width, qh, k


def stratified_split(seed_labels, rng):
    """Seed-stratified ~50/50 cal/test partition without replacement.
    Returns (cal_idx, test_idx) into the pooled arrays."""
    cal, test = [], []
    for s in np.unique(seed_labels):
        idx = np.where(seed_labels == s)[0]
        perm = rng.permutation(idx)
        ncal = len(idx) // 2
        cal.append(perm[:ncal])
        test.append(perm[ncal:])
    return np.concatenate(cal), np.concatenate(test)


def bootstrap_cp(y, scores, seed_labels, score_order, normalized_map,
                 B=1000, seed=RNG_SEED, alpha=ALPHA):
    """B bootstrap replicates. Each replicate: one seed-stratified split; per score
    record coverage, width, q_hat. Also the strict full ordering indicator
    W[order[0]] < W[order[1]] < ... per replicate.

    Returns dict: per-score arrays cov[B], width[B], qhat[B]; ordering_frac (float);
    ncal/ntest (fixed), plus mean/SD summaries.
    """
    rng = np.random.default_rng(seed)
    cov = {s: np.empty(B) for s in score_order}
    wid = {s: np.empty(B) for s in score_order}
    qha = {s: np.empty(B) for s in score_order}
    order_hits = 0
    ncal = ntest = None
    for b in range(B):
        cal, test = stratified_split(seed_labels, rng)
        if ncal is None:
            ncal, ntest = len(cal), len(test)
        y_cal, y_test = y[cal], y[test]
        wvals = {}
        for s in score_order:
            p_cal = scores[s][cal]
            p_test = scores[s][test]
            c, w, qh, _ = split_cp(y_cal, p_cal, y_test, p_test, normalized_map[s], alpha)
            cov[s][b] = c; wid[s][b] = w; qha[s][b] = qh
            wvals[s] = w
        # strict full ordering across the ordered score list
        widths_in_order = [wvals[s] for s in score_order]
        if all(widths_in_order[i] < widths_in_order[i + 1]
               for i in range(len(widths_in_order) - 1)):
            order_hits += 1
    out = {"B": B, "n_cal": int(ncal), "n_test": int(ntest),
           "ordering_fraction": order_hits / B,
           "per_score": {}}
    for s in score_order:
        out["per_score"][s] = {
            "mean_coverage": float(np.mean(cov[s])),
            "sd_coverage": float(np.std(cov[s])),
            "mean_width": float(np.mean(wid[s])),
            "sd_width": float(np.std(wid[s])),
            "mean_qhat": float(np.mean(qha[s])),
        }
    # %-tighter-than-marginal for each score (using bootstrap-mean widths)
    wm = out["per_score"]["marginal"]["mean_width"]
    for s in score_order:
        ws = out["per_score"][s]["mean_width"]
        out["per_score"][s]["pct_tighter_than_marginal"] = (
            float((wm - ws) / wm) if abs(wm) > 1e-12 else float("nan"))
    out["_arrays"] = {"cov": cov, "wid": wid, "qha": qha}  # for hashing / downstream
    return out


def mondrian_bootstrap(y, strat_score, seed_labels, other_scores_unused=None,
                       n_terciles=3, B=1000, seed=RNG_SEED, alpha=ALPHA):
    """Mondrian conditional-coverage audit for ONE stratification variable.

    Per replicate: seed-stratified split; tercile edges computed on the CAL half's
    strat_score; assign both cal and test edits to terciles by those edges.
      * POOLED calibrate: single q_hat on all cal (marginal-style r=y), measure
        coverage inside each test tercile.
      * MONDRIAN: recompute q_hat within each stratum's cal (k=ceil((n_cal_str+1)*.9)),
        measure coverage inside the matching test tercile.
    We use the MARGINAL nonconformity (r=y, U=q_hat) here so strata coverage is a
    clean conditional-coverage statement about the certificate itself.

    Returns per-tercile bootstrap-mean pooled & Mondrian coverage + deviations (pp).
    """
    rng = np.random.default_rng(seed)
    T = n_terciles
    pooled_cov = [[] for _ in range(T)]
    mond_cov = [[] for _ in range(T)]
    ncal_str = [[] for _ in range(T)]
    ntest_str = [[] for _ in range(T)]
    for b in range(B):
        cal, test = stratified_split(seed_labels, rng)
        y_cal, y_test = y[cal], y[test]
        sc_cal, sc_test = strat_score[cal], strat_score[test]
        # tercile edges from cal half
        edges = np.quantile(sc_cal, [1/3, 2/3])
        cal_bin = np.digitize(sc_cal, edges)   # 0,1,2
        test_bin = np.digitize(sc_test, edges)
        # pooled q_hat (marginal r = y)
        qh_pool, _ = q_hat_from_cal(y_cal, alpha)
        for t in range(T):
            tt = test_bin == t
            if tt.sum() > 0:
                pooled_cov[t].append(float(np.mean(y_test[tt] <= qh_pool)))
                ntest_str[t].append(int(tt.sum()))
            # Mondrian: q_hat within this stratum's cal
            ct = cal_bin == t
            if ct.sum() >= 3 and tt.sum() > 0:
                qh_m, _ = q_hat_from_cal(y_cal[ct], alpha)
                mond_cov[t].append(float(np.mean(y_test[tt] <= qh_m)))
                ncal_str[t].append(int(ct.sum()))
    res = []
    for t in range(T):
        pc = float(np.mean(pooled_cov[t])) if pooled_cov[t] else float("nan")
        mc = float(np.mean(mond_cov[t])) if mond_cov[t] else float("nan")
        res.append({
            "tercile": t,  # 0=low,1=mid,2=high
            "n_cal_mean": float(np.mean(ncal_str[t])) if ncal_str[t] else float("nan"),
            "n_test_mean": float(np.mean(ntest_str[t])) if ntest_str[t] else float("nan"),
            "pooled_coverage": pc,
            "mondrian_coverage": mc,
            "pooled_dev_pp": (pc - 0.90) * 100.0 if np.isfinite(pc) else float("nan"),
            "mondrian_dev_pp": (mc - 0.90) * 100.0 if np.isfinite(mc) else float("nan"),
        })
    return {"B": B, "terciles": res}


def mc_null_band_mondrian(seeds_ncal, n_terciles=3, B=1000, n_outer=400,
                          seed=RNG_SEED, alpha=ALPHA, band=0.99):
    """Matched-procedure MC null band for the BOOTSTRAP-MEAN per-tercile POOLED
    coverage produced by mondrian_bootstrap (AUDIT FIX 2026-07-01).

    The statistic E2 gates is a B-split bootstrap MEAN of pooled-tercile
    coverage, where q_hat is calibrated on the FULL cal half (~N/2) and tercile
    edges come from the cal half's stratifier. The band previously shipped
    (conformal.mc_null_band with B=1 and n_cal = the tercile's ~N/6 points)
    simulated a DIFFERENT, far noisier statistic — a single-split coverage with
    a Mondrian-sized calibration set — and was several times too wide, silently
    missing real conditional-coverage failures. Here the null is simulated with
    the EXACT mondrian_bootstrap machinery: iid uniform y (split-CP coverage is
    rank-based / distribution-free) and an INDEPENDENT uniform stratifier
    (exchangeable null: strat carries no information about y), the same
    seed-block structure and the same B. Percentile bands over n_outer outer
    draws, per tercile.
    """
    rng = np.random.default_rng(seed + 1555)
    seed_labels = np.concatenate([np.full(int(n), i, dtype=int)
                                  for i, n in enumerate(seeds_ncal)])
    outer = np.full((n_outer, n_terciles), np.nan)
    for o in range(n_outer):
        y = rng.random(seed_labels.shape[0])
        strat = rng.random(seed_labels.shape[0])   # independent of y under the null
        md = mondrian_bootstrap(y, strat, seed_labels, n_terciles=n_terciles,
                                B=B, seed=int(rng.integers(2 ** 31)), alpha=alpha)
        for t in range(n_terciles):
            outer[o, t] = md["terciles"][t]["pooled_coverage"]
    lo_q = (1 - band) / 2 * 100
    hi_q = (1 + band) / 2 * 100
    per_tercile = []
    for t in range(n_terciles):
        col = outer[:, t][np.isfinite(outer[:, t])]
        per_tercile.append({"tercile": t,
                            "band_lo": float(np.percentile(col, lo_q)),
                            "band_hi": float(np.percentile(col, hi_q)),
                            "null_mean": float(np.mean(col))})
    return {
        "per_tercile": per_tercile, "n_outer": int(n_outer), "B": int(B),
        "band_level": band,
        "statistic": ("bootstrap-mean pooled-tercile coverage, pooled-cal q_hat, "
                      "per-split tercile assignment from an independent null "
                      "stratifier (matched to mondrian_bootstrap exactly)"),
        "note_qhat_clamp": (
            "q_hat_from_cal clamps k=min(k,n): for n_cal<9 the 0.90 split-CP "
            "order statistic k=ceil((n+1)*0.9) exceeds n and the 'no finite "
            "bound' case silently becomes the sample max. The pooled q_hat here "
            "uses ~N/2 cal points (>>9) so the clamp never fires for the pooled "
            "statistic; the Mondrian arm's ct.sum()>=3 guard permits it in "
            "principle for tiny strata — flagged as a caveat."),
    }


def mc_null_band(n_cal, n_test, seeds_ncal, B=1000, n_outer=2000, seed=RNG_SEED,
                 alpha=ALPHA, band=0.99):
    """MC-CI null band for the BOOTSTRAP-MEAN coverage under iid-exchangeable scores.

    Split-CP coverage is distribution-free (rank-based), so we simulate with
    uniform iid scores. Each outer draw: fresh pool of (n_cal+n_test) uniforms per
    the seed structure; B bootstrap seed-stratified splits; bootstrap-mean coverage.
    n_outer outer draws -> band percentiles. E[cov] ~= k/(n_cal+1).

    seeds_ncal: list of per-seed edit counts (to reproduce the exact stratified split).
    Returns dict with theoretical E[cov], band lo/hi, and the realized mean.
    """
    rng = np.random.default_rng(seed + 777)
    seed_blocks = []  # (start, n, ncal) per seed
    start = 0
    for n in seeds_ncal:
        seed_blocks.append((start, int(n), int(n) // 2))
        start += int(n)
    N = start
    k_total = int(np.ceil((n_cal + 1) * (1.0 - alpha)))
    e_cov = min(k_total, n_cal) / (n_cal + 1)
    outer_means = np.empty(n_outer)
    for o in range(n_outer):
        u = rng.random(N)  # iid uniform "scores" = marginal nonconformity y
        # vectorized B stratified splits: (B, N) random keys, argsort within each seed block
        keys = rng.random((B, N))
        cal_mask = np.zeros((B, N), bool)
        for (st, n, ncal) in seed_blocks:
            blk = keys[:, st:st + n]
            order = np.argsort(blk, axis=1)          # (B, n)
            calsel = order[:, :ncal]                  # first ncal -> calibration
            rows = np.repeat(np.arange(B), ncal)
            cols = st + calsel.reshape(-1)
            cal_mask[rows, cols] = True
        covs = np.empty(B)
        uu = np.broadcast_to(u, (B, N))
        # q_hat per replicate = k-th smallest of the calibration scores
        for b in range(B):
            rc = uu[b][cal_mask[b]]
            qh, _ = q_hat_from_cal(rc, alpha)
            rt = uu[b][~cal_mask[b]]
            covs[b] = np.mean(rt <= qh)
        outer_means[o] = covs.mean()
    lo = float(np.percentile(outer_means, (1 - band) / 2 * 100))
    hi = float(np.percentile(outer_means, (1 + band) / 2 * 100))
    return {
        "n_cal": int(n_cal), "n_test": int(n_test), "k": int(k_total),
        "E_cov_theoretical": float(e_cov),
        "band_level": band, "n_outer": n_outer, "B": B,
        "band_lo": lo, "band_hi": hi,
        "null_mean_coverage": float(outer_means.mean()),
    }
