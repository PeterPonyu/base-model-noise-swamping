"""analyze_sequential.py — the SEQUENTIAL-COLLAPSE analyzer.

Consumes killgate_keygeom.py `--no_restore` (SEQ mode) .npz streams and answers
the direction's key unknown (H1): does S-weighted key-overlap interference from
LATER edits predict whether an EARLIER edit's fact survives (or when it dies)?

npz fields consumed (schema: killgate_keygeom.py --save_matrices, no_restore=1):
  prior_eff/prior_pnew/prior_ptrue  [n_rechecks, n_edits]  NaN=unchecked (j >= recheck_at[t])
  recheck_at                        [n_rechecks]           absolute edit-count checkpoints
  GRAM_pre                          [N, N]                 pre-sequence key cosine Gram
  resid_norm                        [N]                    ROME edit-strength S (NaN->0 for ft)
  edit_ok                           [N]                    per-edit efficacy at insertion time
  cum_interference_pre              [N]                    PRIOR-direction interference (not H1;
                                                             H1 needs the FORWARD direction, recomputed
                                                             here from GRAM_pre+resid_norm as source of truth)

Design (per the analysis plan):
  (a) Survival curves + per-edit death labels/timing — see `compute_survival`. TWO death
      labels are exposed: `died_strict` (efficacious at the edit's FIRST valid recheck AND
      failed by the FINAL recheck — the "died" the spec asks for) and `died_simple` (crude:
      failed at the FINAL recheck regardless of earlier state — conflates true overwrite-death
      with "never worked to begin with"; kept for contrast per spec). The original `died`/
      `death_time` fields (ANY-recheck-failure, i.e. first observed drop below 0.5 among an
      at-risk edit's valid rechecks, whether or not it later recovers) are also kept — a third,
      richer variant — since nothing downstream depends on removing them.
  (b) H1 predictors (per edit i, looking FORWARD to later edits j>i):
        interference_sw    = sum_{j>i} S_j * |cos(k_i,k_j)|      (S-weighted; the spec predictor)
        interference_plain = sum_{j>i} |cos(k_i,k_j)|            (unweighted control)
        interference_max   = max_{j>i} |cos(k_i,k_j)|            (max-single-overlap control)
        cum_interference_pre_field                                (contrast only — this npz field
                                                                    looks BACKWARD, j<i; included
                                                                    per spec (d), not read as an
                                                                    overwrite-death predictor)
      each raw AND position-partialled (rank-residualized on insertion index i) vs
      died_strict, died_simple, and (legacy) died/death_time.
  (c) Position-fragility: Spearman(position, died) + a permutation null (RNG_SEED=12345).
  (d) H2 (toxicity) — does an edit's OWN required correction strength (resid_norm, S_t)
      inflate with accumulated PRIOR interference (the npz's `cum_interference_pre`, the
      correct BACKWARD direction for this question), beyond edit-index and norm-growth
      nulls? See `h2_toxicity`.
  (e) H3 (ex-ante collapse) — can the recheck at which population survival first drops
      below 0.5 be flagged from BASE-MODEL Gram geometry alone (mean/quantile |cos| of
      GRAM_pre), a 0-GPU toxicity signal computable before any edit is applied? See
      `h3_collapse_onset`.
  (f) Cross-stream aggregation over a glob: per-stream stats, a POOLED re-analysis over the
      concatenation of all streams' per-edit vectors, AND an explicit cross-ordering
      consistency check (does each H1 predictor's position-partialled sign/magnitude hold
      across independently-ordered streams — the actual evidence for a position-vs-geometry
      decoupling, since a single stream can't separate "geometry causes death" from "this
      particular ordering causes death"). See `cross_ordering_consistency`.

Usage:
  python experiments/seq/analyze_sequential.py "results/matrices/seq_llama1b_nr_L12_s*.npz" \
      --out results/seq_H1.json [--n_perm 1000]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

# Reuse the AUDITED rank machinery from the G1 gate analyzer (task convention: "reuse
# analyze_matrices._midrank if importable") instead of re-deriving it — analyze_sequential.py
# lives one level below experiments/, so add that parent dir to sys.path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analyze_matrices import _midrank, spearman  # noqa: E402

RNG_SEED = 12345  # fixed so every permutation null is reproducible (matches analyze_matrices.py)


# ---------------------------------------------------------------- rank stats (partial Spearman)
def _rank_residual(x, z):
    """Residual of rank(x) after linearly regressing on rank(z) (+intercept)."""
    rx, rz = _midrank(x), _midrank(z)
    A = np.vstack([rz, np.ones_like(rz)]).T
    coef, *_ = np.linalg.lstsq(A, rx, rcond=None)
    return rx - A @ coef


def partial_spearman(x, y, z):
    """Partial Spearman of x vs y, controlling for z: rank-residualize BOTH x and y
    on z (linear regression on ranks), then Pearson the residuals. Standard
    rank-based partial-correlation estimator; z may be binary/continuous."""
    x, y, z = np.asarray(x, float), np.asarray(y, float), np.asarray(z, float)
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[m], y[m], z[m]
    if x.size < 5:
        return float("nan")
    rx = _rank_residual(x, z)
    ry = _rank_residual(y, z)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def perm_null_p(x, y, obs, n_perm=1000, seed=RNG_SEED):
    """Permutation null for Spearman(x, y): shuffle y, recompute, empirical p-value."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 3 or not np.isfinite(obs):
        return float("nan")
    ge = 0
    for _ in range(n_perm):
        yp = rng.permutation(y)
        s = spearman(x, yp)
        if np.isfinite(s) and abs(s) >= abs(obs):
            ge += 1
    return (ge + 1) / (n_perm + 1)


# ---------------------------------------------------------------- (a) survival + death timing
def compute_survival(edit_ok, recheck_at, prior_eff):
    """Per-edit at-risk/checked/died/death_time + the per-recheck survival curve.

    at_risk[i]    = edit i succeeded when it was FIRST installed (edit_ok[i]==1);
                    only at-risk edits can meaningfully "die" later.
    checked[i]    = at least one recheck happened after i was installed
                    (i.e. exists t with recheck_at[t] > i).
    died[i]       = 1.0 if, among i's valid rechecks (in increasing time order), prior_eff
                    ever reads 0 (a fact that WAS installed successfully got overwritten);
                    0.0 if checked but never observed to fail; NaN if not at_risk or not checked.
                    (ANY-recheck-failure variant — richest, but doesn't distinguish "died then
                    stayed dead" from "flickered once then recovered".)
    death_time[i] = the recheck_at value of the FIRST recheck at which i failed (NaN unless died==1).
    first_check_val[i] = efficacy (0/1) at i's FIRST valid recheck (NaN if never checked).
    final_val[i]       = efficacy (0/1) at the LAST recheck (covers all edits by construction;
                          NaN only if prior_eff's last row is somehow NaN for i).
    died_strict[i]      = 1.0 iff first_check_val==1 AND final_val==0 (efficacious early, dead
                          by the end — the spec's primary "died" label); 0.0 iff first_check_val==1
                          AND final_val==1 (survived to the end); NaN if first_check_val==0
                          ("stillborn" — never alive at a recheck, so it cannot "die"; EXCLUDED).
    died_simple[i]      = 1.0 iff final_val==0, else 0.0 — the crude label the spec also asks
                          for. WARNING: conflates stillborn edits with genuine overwrite-deaths;
                          prefer died_strict for H1, use died_simple only for contrast.
    """
    N = len(edit_ok)
    at_risk = np.asarray(edit_ok, float) > 0.5
    died = np.full(N, np.nan)
    death_time = np.full(N, np.nan)
    first_check_val = np.full(N, np.nan)
    final_val = np.full(N, np.nan)
    n_checks = np.zeros(N, dtype=int)
    order = np.argsort(recheck_at)  # rechecks are already chronological, but be defensive
    for i in range(N):
        valid_ts = [t for t in order if recheck_at[t] > i]
        if valid_ts:
            fv = prior_eff[valid_ts[0], i]
            first_check_val[i] = fv if not np.isnan(fv) else np.nan
        lv = prior_eff[order[-1], i]
        final_val[i] = lv if not np.isnan(lv) else np.nan
        if not at_risk[i]:
            continue
        if not valid_ts:
            continue
        n_checks[i] = len(valid_ts)
        d, dt = 0.0, np.nan
        for t in valid_ts:
            v = prior_eff[t, i]
            if np.isnan(v):
                continue
            if v < 0.5:
                d, dt = 1.0, float(recheck_at[t])
                break
        died[i] = d
        death_time[i] = dt
    checked = n_checks > 0

    stillborn = first_check_val == 0
    died_strict = np.where(np.isnan(first_check_val), np.nan,
                            np.where(stillborn, np.nan,
                                     np.where(final_val == 0, 1.0, 0.0)))
    died_simple = np.where(np.isnan(final_val), np.nan, np.where(final_val == 0, 1.0, 0.0))

    survival_curve = []
    for t in order:
        col = prior_eff[t]
        pos_mask = np.arange(N) < recheck_at[t]
        mask = at_risk & pos_mask & ~np.isnan(col)
        n = int(mask.sum())
        rate = float(np.nanmean(col[mask])) if n else float("nan")
        survival_curve.append({
            "recheck_at": int(recheck_at[t]), "n_at_risk": n, "survival_rate": rate,
        })
    return {
        "at_risk": at_risk, "checked": checked, "died": died, "death_time": death_time,
        "first_check_val": first_check_val, "final_val": final_val,
        "died_strict": died_strict, "died_simple": died_simple,
        "n_checks": n_checks, "survival_curve": survival_curve,
    }


# ---------------------------------------------------------------- (b) forward interference (H1)
def forward_interference(GRAM_pre, resid_norm):
    """Per-edit i, looking FORWARD to all later edits j>i:
        interference_sw    = sum_j S_j * |cos(k_i,k_j)|   (S-weighted; spec predictor)
        interference_plain = sum_j |cos(k_i,k_j)|          (unweighted sum control)
        interference_max   = max_j |cos(k_i,k_j)|          (max-single-overlap control)
    NOTE: this is the MIRROR of the npz's `cum_interference_pre` (which looks BACKWARD,
    j<i) — recomputed here from GRAM_pre + resid_norm (source of truth) since the
    forward-looking direction is what predicts whether an EARLIER edit gets overwritten
    by a later, key-overlapping, high-strength edit."""
    N = GRAM_pre.shape[0]
    S = np.nan_to_num(np.asarray(resid_norm, float), nan=0.0)
    absG = np.abs(np.asarray(GRAM_pre, float))
    # NaN-init (not zeros): the last edit has no LATER edits, so its interference is
    # UNDEFINED, not "measured zero" — a hardcoded 0.0 would wrongly leak a degenerate
    # data point into every downstream correlation.
    sw = np.full(N, np.nan); plain = np.full(N, np.nan); mx = np.full(N, np.nan)
    for i in range(N - 1):
        fut = absG[i, i + 1:]
        Sfut = S[i + 1:]
        sw[i] = float((Sfut * fut).sum())
        plain[i] = float(fut.sum())
        mx[i] = float(fut.max())
    return {"interference_sw": sw, "interference_plain": plain, "interference_max": mx}


def _corr_pair(pred, outcome, pos, min_n=8):
    """Raw + position-partialled Spearman(pred, outcome | pos) for one predictor/outcome
    pair, with an honest note when there aren't enough eligible (finite) edits."""
    pred, outcome, pos = np.asarray(pred, float), np.asarray(outcome, float), np.asarray(pos, float)
    m = np.isfinite(pred) & np.isfinite(outcome) & np.isfinite(pos)
    n = int(m.sum())
    if n < min_n:
        return {"n": n, "note": f"too few eligible edits (<{min_n}) for a rank correlation"}
    raw = spearman(pred[m], outcome[m])
    part = partial_spearman(pred[m], outcome[m], pos[m])
    return {
        "n": n,
        "raw_spearman": round(raw, 4) if np.isfinite(raw) else None,
        "position_partialled_spearman": round(part, 4) if np.isfinite(part) else None,
    }


# ---------------------------------------------------------------- (d) H2: toxicity
def h2_toxicity(d, resid_norm, norm_growth):
    """H2: does an edit's OWN required correction strength (S_t = resid_norm[t]) inflate
    with accumulated PRIOR interference (the npz's `cum_interference_pre`, the correct
    BACKWARD direction: sum_{j<t} S_j*|cos(k_j,k_t)| — interference fed INTO edit t by
    edits that came before it), beyond edit-index and norm-growth nulls?

    Also reports the same regression against per-edit mean probe damage (row-mean of
    `damage_logit`, where measured) as a complementary "how much collateral did THIS edit
    itself cause" proxy, when that field is present and probe rows were actually measured
    (--probe_stride==1, or this is the final edit)."""
    if "cum_interference_pre" not in d.files:
        return {"note": "cum_interference_pre not present in this npz — H2 skipped"}
    CI = d["cum_interference_pre"].astype(float)
    N = CI.shape[0]
    index = np.arange(N, dtype=float)
    m = np.isfinite(resid_norm) & np.isfinite(CI)
    out = {
        "n": int(m.sum()),
        "S_vs_cum_interference_pre": {
            "raw_spearman": (round(spearman(CI[m], resid_norm[m]), 4)
                            if m.sum() >= 8 and np.isfinite(spearman(CI[m], resid_norm[m])) else None),
            "index_partialled": (round(partial_spearman(CI[m], resid_norm[m], index[m]), 4)
                                 if m.sum() >= 8 else None),
            "normgrowth_partialled": (round(partial_spearman(CI[m], resid_norm[m], norm_growth[m]), 4)
                                      if m.sum() >= 8 else None),
        },
    }
    if "damage_logit" in d.files:
        row_mean = np.nanmean(d["damage_logit"].astype(float), axis=1)
        mrow = np.isfinite(row_mean) & np.isfinite(CI)
        if int(mrow.sum()) >= 8:
            out["meandamage_vs_cum_interference_pre"] = {
                "n": int(mrow.sum()),
                "raw_spearman": round(spearman(CI[mrow], row_mean[mrow]), 4),
                "index_partialled": round(partial_spearman(CI[mrow], row_mean[mrow], index[mrow]), 4),
                "normgrowth_partialled": round(
                    partial_spearman(CI[mrow], row_mean[mrow], norm_growth[mrow]), 4),
            }
        else:
            out["meandamage_vs_cum_interference_pre"] = {"note": "too few measured probe rows"}
    out["note"] = ("cum_interference_pre is the BACKWARD/INCOMING direction (sum_{j<t} "
                  "S_j*|cos(k_j,k_t)|) — the correct one for 'does accumulated PRIOR "
                  "interference inflate this edit's own required correction / collateral'; "
                  "this is the mirror image of H1's FORWARD/OUTGOING interference_sw.")
    return out


# ---------------------------------------------------------------- (e) H3: ex-ante collapse
def h3_collapse_onset(GRAM_pre, recheck_at, prior_eff):
    """H3: can collapse onset (the recheck at which population-mean survival first drops
    below 0.5, among edits checked so far) be flagged from BASE-MODEL Gram geometry ALONE
    — a 0-GPU toxicity signal computable before any edit is even applied?"""
    N = GRAM_pre.shape[0]
    order = np.argsort(recheck_at)
    survival = []
    for t in order:
        col = prior_eff[t]
        pos_mask = np.arange(N) < recheck_at[t]
        mask = pos_mask & ~np.isnan(col)
        survival.append(float(np.nanmean(col[mask])) if mask.sum() else float("nan"))
    survival = np.asarray(survival)
    below = np.where(np.isfinite(survival) & (survival < 0.5))[0]
    onset_idx = int(below[0]) if below.size else None
    onset_edit_count = int(recheck_at[order][onset_idx]) if onset_idx is not None else None
    absG = np.abs(np.asarray(GRAM_pre, float))
    iu = np.triu_indices(N, k=1)
    offdiag = absG[iu]
    return {
        "survival_curve": [round(float(x), 3) if np.isfinite(x) else None for x in survival],
        "recheck_at_sorted": recheck_at[order].tolist(),
        "collapse_onset_recheck_idx": onset_idx,
        "collapse_onset_edit_count": onset_edit_count,
        "base_gram_geometry": {
            "mean_offdiag_abs_cos": round(float(offdiag.mean()), 4),
            "median_offdiag_abs_cos": round(float(np.median(offdiag)), 4),
            "p90_offdiag_abs_cos": round(float(np.quantile(offdiag, 0.9)), 4),
        },
        "note": ("single-stream scalar pair — a defensible onset-vs-geometry TREND needs "
                "several independent streams (>=4-5); with 2-3 streams this is a raw-pairs "
                "listing at the multi-stream aggregation level, not a tested correlation."),
    }


# ---------------------------------------------------------------- per-stream analysis
def analyze_one(npz_path, n_perm=1000, seed=RNG_SEED):
    d = np.load(npz_path)
    if "seq_no_restore" not in d.files or int(d["seq_no_restore"]) != 1:
        return {"npz": os.path.basename(npz_path), "note": "not a SEQ (--no_restore) npz — skipped"}
    need = {"prior_eff", "recheck_at", "GRAM_pre", "edit_ok", "resid_norm"}
    missing = need - set(d.files)
    if missing:
        return {"npz": os.path.basename(npz_path), "note": f"missing fields {missing} — skipped"}

    edit_ok = d["edit_ok"].astype(float)
    recheck_at = d["recheck_at"].astype(int)
    prior_eff = d["prior_eff"].astype(float)
    GRAM_pre = d["GRAM_pre"].astype(float)
    resid_norm = d["resid_norm"].astype(float)
    norm_growth = (d["norm_growth"].astype(float) if "norm_growth" in d.files
                  else np.full(len(edit_ok), np.nan))
    N = len(edit_ok)
    position = np.arange(N, dtype=float)

    surv = compute_survival(edit_ok, recheck_at, prior_eff)
    interf = forward_interference(GRAM_pre, resid_norm)
    if "cum_interference_pre" in d.files:
        # (d) contrast-only candidate: BACKWARD/INCOMING direction, opposite of H1's question
        # (does edit i get overwritten by LATER edits j>i) — kept for completeness per spec,
        # never read as evidence for/against the overwrite hypothesis.
        interf = dict(interf)
        interf["cum_interference_pre_field_CONTRAST_ONLY"] = d["cum_interference_pre"].astype(float)

    pop = surv["at_risk"] & surv["checked"]           # population with a defined died/death_time
    died = surv["died"]
    death_time = surv["death_time"]
    died_strict = surv["died_strict"]
    died_simple = surv["died_simple"]

    h1 = {}
    for name, pred in interf.items():
        pred_pop = pred[pop]; died_pop = died[pop]; pos_pop = position[pop]
        died_mask = died_pop > 0.5
        dt_sub = death_time[pop][died_mask]
        pred_sub = pred_pop[died_mask]
        pos_sub = pos_pop[died_mask]
        h1[name] = {
            # legacy ("died" = ANY-recheck-failure) — kept for continuity with the earlier version
            "vs_died_any_legacy": _corr_pair(pred_pop, died_pop, pos_pop),
            "vs_death_time_amongdied_any_legacy": _corr_pair(pred_sub, dt_sub, pos_sub, min_n=5),
            # spec's primary/simple labels (whole-stream population, not just at_risk&checked)
            "vs_died_strict": _corr_pair(pred, died_strict, position),
            "vs_died_simple": _corr_pair(pred, died_simple, position),
        }

    # (c) position-fragility: does insertion order ALONE predict death?
    pos_pop = position[pop]; died_pop = died[pop]
    rho_pos_died = spearman(pos_pop, died_pop)
    p_pos_died = perm_null_p(pos_pop, died_pop, rho_pos_died, n_perm=n_perm, seed=seed)
    position_fragility = {
        "n": int(pop.sum()),
        "spearman_position_died": round(rho_pos_died, 4) if np.isfinite(rho_pos_died) else None,
        "perm_p": round(p_pos_died, 4) if np.isfinite(p_pos_died) else None,
        "n_perm": n_perm,
    }

    h2 = h2_toxicity(d, resid_norm, norm_growth)
    h3 = h3_collapse_onset(GRAM_pre, recheck_at, prior_eff)

    return {
        "npz": os.path.basename(npz_path),
        "n_edits": int(N),
        "n_rechecks": int(len(recheck_at)),
        "recheck_at": recheck_at.tolist(),
        "n_at_risk": int(surv["at_risk"].sum()),
        "n_checked": int(surv["checked"].sum()),
        "n_in_population": int(pop.sum()),
        "died_rate": (round(float(np.nanmean(died[pop])), 4) if pop.sum() else None),
        "n_stillborn": int(np.nansum(surv["first_check_val"] == 0)),
        "n_died_strict": int(np.nansum(died_strict == 1.0)),
        "n_died_simple": int(np.nansum(died_simple == 1.0)),
        "survival_curve": surv["survival_curve"],
        "H1": h1,
        "position_fragility": position_fragility,
        "H2_toxicity": h2,
        "H3_collapse_onset": h3,
        # raw per-edit arrays kept for pooled re-analysis (not for direct JSON consumption below)
        "_raw": {
            "position": position, "pop": pop, "died": died, "death_time": death_time,
            "died_strict": died_strict, "died_simple": died_simple,
            **interf,
        },
    }


def _pool(streams):
    """Concatenate per-edit arrays across streams (each stream's position/interference
    values are stream-local, which is correct: 'insertion index' and 'forward interference'
    are both defined within a single sequential run)."""
    keys = ["position", "died", "death_time", "died_strict", "died_simple",
           "interference_sw", "interference_plain", "interference_max"]
    pooled = {k: [] for k in keys}
    pop_all = []
    contrast_present = all("cum_interference_pre_field_CONTRAST_ONLY" in (s.get("_raw") or {})
                           for s in streams) and len(streams) > 0
    if contrast_present:
        pooled["cum_interference_pre_field_CONTRAST_ONLY"] = []
    for s in streams:
        raw = s.get("_raw")
        if raw is None:
            continue
        pop = raw["pop"]
        pop_all.append(pop)
        for k in pooled:
            pooled[k].append(raw[k])
    if not pooled["position"]:
        return None
    for k in pooled:
        pooled[k] = np.concatenate(pooled[k])
    pop_all = np.concatenate(pop_all)
    return pooled, pop_all


def pooled_analysis(streams, n_perm=1000, seed=RNG_SEED):
    out = _pool(streams)
    if out is None:
        return None
    pooled, pop = out
    position, died, death_time = pooled["position"], pooled["died"], pooled["death_time"]
    died_strict, died_simple = pooled["died_strict"], pooled["died_simple"]
    pop_pos, pop_died = position[pop], died[pop]

    predictor_names = [k for k in pooled if k not in
                      ("position", "died", "death_time", "died_strict", "died_simple")]
    h1 = {}
    for name in predictor_names:
        pred = pooled[name]
        pred_pop = pred[pop]
        died_mask = pop_died > 0.5
        dt_sub = death_time[pop][died_mask]
        pred_sub = pred_pop[died_mask]
        pos_sub = pop_pos[died_mask]
        h1[name] = {
            "vs_died_any_legacy": _corr_pair(pred_pop, pop_died, pop_pos),
            "vs_death_time_amongdied_any_legacy": _corr_pair(pred_sub, dt_sub, pos_sub, min_n=5),
            "vs_died_strict": _corr_pair(pred, died_strict, position),
            "vs_died_simple": _corr_pair(pred, died_simple, position),
        }

    rho_pos_died = spearman(pop_pos, pop_died)
    p_pos_died = perm_null_p(pop_pos, pop_died, rho_pos_died, n_perm=n_perm, seed=seed)
    return {
        "n_streams_pooled": int(len(streams)),
        "n_in_population": int(pop.sum()),
        "died_rate": round(float(np.nanmean(pop_died)), 4) if pop.sum() else None,
        "H1": h1,
        "position_fragility": {
            "n": int(pop.sum()),
            "spearman_position_died": round(rho_pos_died, 4) if np.isfinite(rho_pos_died) else None,
            "perm_p": round(p_pos_died, 4) if np.isfinite(p_pos_died) else None,
            "n_perm": n_perm,
        },
        "note": ("this pooled view CONCATENATES all streams' per-edit vectors into one "
                "regression — a simple joint model, not a mixed/hierarchical one; it does NOT "
                "by itself demonstrate cross-ordering consistency (a strong single stream could "
                "dominate). See 'cross_ordering_consistency' for the per-stream sign/magnitude "
                "comparison, which is the actual decoupling evidence."),
    }


def cross_ordering_consistency(streams):
    """THE cross-ordering check: does each H1 predictor's POSITION-PARTIALLED sign/magnitude
    hold across independently-ordered streams? A single stream can never distinguish
    'geometry causes death' from 'this particular ordering causes death' — this is the only
    part of the analysis that can."""
    if len(streams) < 2:
        return {"note": "single stream — no cross-ordering check possible (need >=2 orderings)"}
    predictor_names = list(streams[0]["H1"].keys())
    out = {}
    for name in predictor_names:
        row = {}
        for label in ("vs_died_strict", "vs_died_simple", "vs_died_any_legacy"):
            vals = [s["H1"].get(name, {}).get(label, {}).get("position_partialled_spearman")
                    for s in streams]
            finite = [v for v in vals if isinstance(v, (int, float))]
            same_sign = (len(finite) >= 2 and
                        (all(v > 0 for v in finite) or all(v < 0 for v in finite)))
            row[label] = {
                "per_stream_values": vals,
                "same_sign_across_streams": same_sign if len(finite) >= 2 else None,
                "spread": round(max(finite) - min(finite), 4) if len(finite) >= 2 else None,
            }
        out[name] = row
    return out


def h3_cross_stream(streams):
    pairs = [(s["H3_collapse_onset"]["collapse_onset_edit_count"],
             s["H3_collapse_onset"]["base_gram_geometry"]["mean_offdiag_abs_cos"])
            for s in streams]
    return {
        "onset_geometry_pairs(edit_count, mean_offdiag_abs_cos)": pairs,
        "note": (f"n_streams={len(streams)} — a Spearman/trend over onset-vs-geometry needs "
                "several independent streams (>=4-5) to say anything; with 2-3 streams this is "
                "a raw-pairs listing only, not a tested correlation."),
    }


def verdicts(streams):
    """Honest, non-gating verdict strings. No pass/fail threshold is preregistered for a
    single-/few-stream n=50-edit pilot run (contrast analyze_matrices.py's G1 gate, which
    has N*M pair structure and a real permutation-null p-value) — this reports what was
    observed, flags the power limitation, and defers to cross-stream sign-consistency as
    the only defensible corroboration."""
    def _h1_verdict(label, name="interference_sw"):
        vals = [s["H1"].get(name, {}).get(label, {}).get("position_partialled_spearman")
                for s in streams]
        finite = [v for v in vals if isinstance(v, (int, float))]
        if not finite:
            return "UNDETERMINED (predictor degenerate / too few eligible edits on every stream)"
        if len(finite) == 1:
            return f"SINGLE-STREAM ONLY, rho={finite[0]:.3f} — no cross-ordering corroboration; do not trust"
        same_sign = all(v > 0 for v in finite) or all(v < 0 for v in finite)
        spread = max(finite) - min(finite)
        tag = ("SIGN-CONSISTENT" if same_sign else
              "SIGN-FLIPS ACROSS STREAMS (likely ordering/position artifact, not geometry)")
        return f"{tag} across {len(finite)} streams; rhos={[round(v, 3) for v in finite]}, spread={spread:.3f}"

    return {
        "H1_primary(interference_sw, vs_died_strict)": _h1_verdict("vs_died_strict"),
        "H1_primary(interference_sw, vs_died_simple)": _h1_verdict("vs_died_simple"),
        "H2_toxicity": ("see per_stream[*].H2_toxicity.S_vs_cum_interference_pre — no "
                       "cross-stream aggregation implemented beyond raw per-stream numbers "
                       "(H2 is a within-stream regression, not a cross-ordering claim)"),
        "H3_collapse_onset": ("see pooled.H3_cross_stream — raw onset/geometry pairs only; "
                              "not enough streams for a tested trend"),
        "global_power_verdict": (
            f"{len(streams)} stream(s) x n=50 edits: this is a PILOT. Single-digit stream "
            "counts cannot support a permutation-null-style gate the way the restore-mode "
            "G1 analyzer (analyze_matrices.py) can with its N*M within-probe structure. "
            "Sign-consistency across streams is suggestive; magnitude should not be quoted "
            "as a stable effect size until >=4-5 independent orderings are run."
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz_glob", nargs="+", help="one or more SEQ (--no_restore) killgate .npz (globs ok)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n_perm", type=int, default=1000)
    args = ap.parse_args()

    paths = sorted({p for pat in args.npz_glob for p in glob.glob(pat)})
    if not paths:
        raise SystemExit("[analyze_sequential] no .npz matched")

    per_stream_full = [analyze_one(p, n_perm=args.n_perm) for p in paths]
    valid = [s for s in per_stream_full if "_raw" in s]
    skipped = [s for s in per_stream_full if "_raw" not in s]

    pooled = pooled_analysis(valid, n_perm=args.n_perm) if valid else None
    cross_ordering = cross_ordering_consistency(valid)
    h3_cross = h3_cross_stream(valid) if valid else None
    verdict = verdicts(valid) if valid else None

    # strip the internal _raw arrays before JSON serialization
    per_stream = []
    for s in per_stream_full:
        s = dict(s)
        s.pop("_raw", None)
        per_stream.append(s)

    res = {
        "n_streams": len(paths),
        "n_streams_valid": len(valid),
        "n_streams_skipped": len(skipped),
        "per_stream": per_stream,
        "pooled": pooled,
        "cross_ordering_consistency": cross_ordering,
        "H3_cross_stream": h3_cross,
        "power_note_global": (
            "2-3 seed/ordering streams of n=50 sequential edits each is a PILOT, not a "
            "powered study. Per-stream H1/H2 rhos are noisy point estimates; the 'pooled' "
            "view is a simple concatenated regression (not a joint/mixed model) and carries "
            "NO computed p-value beyond position_fragility's own permutation null — do not "
            "quote one for H1/H2. Read consistent sign across streams (cross_ordering_"
            "consistency) as the primary evidence unit."
        ),
        "VERDICTS": verdict,
    }
    print(json.dumps(res, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        tmp = args.out + ".tmp"
        with open(tmp, "w") as f:
            json.dump(res, f, indent=2)
        os.replace(tmp, args.out)  # atomic
        print(f"[analyze_sequential] wrote {args.out}")


if __name__ == "__main__":
    main()
