"""d3_benefit_predictor.py — D3 benefit-MAGNITUDE predictor (CPU-only, numpy-only).

Reframed D3 (07-02): NOT a router (it was degenerate 12/12 always-AlphaEdit). The
sanctioned framing is MAGNITUDE prediction — a deployable pre-edit gate answering
"how much collateral will AlphaEdit save me on THIS edit?". The realized benefit of
choosing AlphaEdit over ROME is the causally-removed collateral damage

    damage_removed = rome_damage - alpha_damage        (matched (edit, probe) pairs)

and B6 established that pre-edit key-geometry (key-cosine to the reference/probe bank)
predicts it. This script packages geometry -> predicted damage_removed and evaluates
it with honest generalization (leave-one-seed-out, leave-one-layer-out).

Conventions are MIRRORED from experiments/aggregate_g4_causal.py + analyze_matrices.py
(same canonical `spearman`/`within_probe_rhos`, same masked_pair row/col masks, same
per-seed-averaged within-probe pooling). NO ML libraries; the "learned" predictor is a
closed-form ordinary-least-squares linear combination of transparent geometric
features — the point is deployability, not fit quality.

Predictors evaluated
  raw_keycos : prediction = signed key-cosine itself (rank-based; zero parameters).
               This is the maximally-deployable gate — available from the edit key
               alone, before the ROME update is even solved.
  ols_combo  : closed-form OLS of damage_removed on [key-cos, S(=norm_growth),
               S*|key-cos|] (+ intercept), features z-scored on the TRAIN split only.
               S needs the solved ROME update, so this gate costs one update solve.

Evaluation (all rank-based, so scale/intercept are irrelevant)
  * within-probe Spearman(predicted, realized) — the canonical B6 statistic, per layer.
  * leave-one-seed-out (LOSO) and leave-one-layer-out (LOLO) held-out within-probe rho.
  * predicted-quartile -> realized mean damage_removed calibration tables.
  * decile screening: top-decile recall of high-benefit pairs (pair-level) and of
    high-benefit edits (per-edit deployable-gate view).

Data (read-only; canonical C4 "probes"-projector cells, dated <= 2026-07-10):
  results/matrices/gate_llama1b_rome_cf_L{L}_s{0,1,2}.npz
  results/matrices/g4_llama1b_alpha_cf_L{L}_s{0,1,2}.npz
Output: results/D3_benefit_predictor_eval.json
"""
from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # edit-harness/
RESULTS = os.path.join(ROOT, "results")
MATRICES = os.path.join(RESULTS, "matrices")

LAYERS = [8, 10, 12, 14]
SEEDS = [0, 1, 2]
ROME_TMPL = "gate_llama1b_rome_cf_L{L}_s{s}.npz"
ALPHA_TMPL = "g4_llama1b_alpha_cf_L{L}_s{s}.npz"  # "probes"-projector cells = the C4 table
FEATURES = ["key_cos", "S", "S_x_absC"]
REPRO_TOL = 0.02
RNG_SEED = 12345


# ----------------------------------------------------------------------------- #
# Canonical rank statistics — copied VERBATIM from analyze_matrices.py so the    #
# reproduction check matches the C4 table to full precision.                     #
# ----------------------------------------------------------------------------- #
def _midrank(x):
    order = x.argsort(kind="mergesort")
    ranks = np.empty(len(x), float)
    ranks[order] = np.arange(1, len(x) + 1, dtype=float)
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt))
    np.add.at(sums, inv, ranks)
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


def within_probe_mean(pred2d, real2d):
    """Mean over probe columns of Spearman(prediction, realized) — the B6 statistic
    applied to (prediction, realized) instead of (key-cos, realized)."""
    return float(np.nanmean(within_probe_rhos(pred2d, real2d)))


# ----------------------------------------------------------------------------- #
# Loading + masking — MIRRORS aggregate_g4_causal.masked_pair exactly.           #
# ----------------------------------------------------------------------------- #
def load_cell(L, s):
    """Return the masked (edit x probe) views + per-edit S for one layer/seed cell,
    or None if a file is missing / too few pairs. Row mask = both editors succeeded;
    col mask = base model knows the probe (pre_p > 0.05). Identical to masked_pair."""
    rp = os.path.join(MATRICES, ROME_TMPL.format(L=L, s=s))
    ap = os.path.join(MATRICES, ALPHA_TMPL.format(L=L, s=s))
    if not (os.path.exists(rp) and os.path.exists(ap)):
        return None
    dr = np.load(rp)
    da = np.load(ap)
    COS = dr["COS"].astype(float)
    Dr = dr["damage_logit"].astype(float)
    Da = da["damage_logit"].astype(float)
    if not (COS.shape == Dr.shape == Da.shape):
        return None
    row = np.ones(COS.shape[0], bool)
    if "edit_ok" in dr.files and "edit_ok" in da.files:
        row = (dr["edit_ok"].astype(float) > 0.5) & (da["edit_ok"].astype(float) > 0.5)
    col = np.ones(COS.shape[1], bool)
    if "pre_p" in dr.files:
        c = dr["pre_p"].astype(float) > 0.05
        if c.sum() >= 5:
            col = c
    cos2d = COS[row][:, col]
    if cos2d.size < 20:
        return None
    rome2d = Dr[row][:, col]
    alpha2d = Da[row][:, col]
    removed2d = rome2d - alpha2d
    # S = per-edit norm_growth (the Eq.2 S surrogate), broadcast across probe columns.
    S = dr["norm_growth"].astype(float)[row]
    S2d = np.repeat(S[:, None], cos2d.shape[1], axis=1)
    return {
        "L": L, "s": s,
        "cos": cos2d, "S": S2d, "rome": rome2d, "alpha": alpha2d, "removed": removed2d,
        "n_edits": cos2d.shape[0], "n_probes": cos2d.shape[1], "n_pairs": cos2d.size,
    }


def feat_stack(cell):
    """(key_cos, S, S*|key_cos|) as 2D arrays, same order as FEATURES."""
    cos, S = cell["cos"], cell["S"]
    return [cos, S, S * np.abs(cos)]


def feat_flat(cell, idx):
    """(n_pairs, len(idx)) design matrix over feature columns `idx`."""
    return np.stack([a.reshape(-1) for a in feat_stack(cell)], axis=1)[:, idx]


# ----------------------------------------------------------------------------- #
# Closed-form OLS on z-scored features (train-split statistics only).            #
# `idx` selects which of FEATURES to use, enabling the feature-ablation runs.    #
# ----------------------------------------------------------------------------- #
def fit_ols(cells, idx=None):
    if idx is None:
        idx = list(range(len(FEATURES)))
    X = np.concatenate([feat_flat(c, idx) for c in cells], axis=0)
    y = np.concatenate([c["removed"].reshape(-1) for c in cells], axis=0)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-12] = 1.0
    Xs = (X - mu) / sd
    Xd = np.concatenate([Xs, np.ones((Xs.shape[0], 1))], axis=1)
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    return {"mu": mu, "sd": sd, "beta": beta, "idx": idx}


def predict_cell(model, cell):
    """Apply the fitted linear model -> predicted damage_removed (edit x probe).
    Uses only the feature columns the model was fit on (model['idx'])."""
    mu, sd, beta, idx = model["mu"], model["sd"], model["beta"], model["idx"]
    stack = feat_stack(cell)
    pred = np.zeros_like(cell["cos"])
    for j, k in enumerate(idx):
        pred = pred + beta[j] * (stack[k] - mu[j]) / sd[j]
    return pred + beta[-1]


def raw_predict_cell(cell):
    """The zero-parameter deployable gate: prediction = signed key-cosine."""
    return cell["cos"]


# ----------------------------------------------------------------------------- #
# Calibration + screening (pooled per layer).                                    #
# ----------------------------------------------------------------------------- #
def quartile_calibration(pred_flat, real_flat):
    """Predicted-quartile -> realized mean damage_removed (mirrors aggregate's
    np.quantile + np.digitize quartile binning, but bins on the PREDICTION)."""
    qs = np.quantile(pred_flat, [0.25, 0.5, 0.75])
    bins = np.digitize(pred_flat, qs)
    rows = []
    for q in range(4):
        m = bins == q
        rows.append({
            "pred_quartile": ["Q1(low)", "Q2", "Q3", "Q4(high)"][q],
            "n_pairs": int(m.sum()),
            "mean_pred": round(float(pred_flat[m].mean()), 5) if m.any() else None,
            "realized_mean_damage_removed": round(float(real_flat[m].mean()), 5) if m.any() else None,
        })
    return rows


def topk_recall(pred_flat, real_flat, frac=0.10):
    """Recall of the true top-`frac` high-benefit units by screening the top-`frac`
    predicted units. |pred_top & true_top| / |true_top| (== precision at equal cut)."""
    n = pred_flat.size
    k = max(1, int(round(frac * n)))
    true_top = set(np.argsort(real_flat)[-k:].tolist())
    pred_top = set(np.argsort(pred_flat)[-k:].tolist())
    hit = len(true_top & pred_top)
    return {"frac": frac, "n_units": int(n), "k": int(k),
            "recall": round(hit / len(true_top), 4) if true_top else None,
            "chance": round(k / n, 4)}


def per_edit_gate(cells, predict_fn):
    """Deployable per-edit magnitude gate: aggregate per (edit) predicted vs realized
    total benefit (mean damage_removed over the known-probe bank), Spearman across
    edits within each cell then averaged, plus top-decile edit recall (pooled)."""
    rhos, pred_all, real_all = [], [], []
    for c in cells:
        pred2d = predict_fn(c)
        pe_pred = pred2d.mean(axis=1)      # per-edit predicted benefit
        pe_real = c["removed"].mean(axis=1)  # per-edit realized benefit
        rhos.append(spearman(pe_pred, pe_real))
        pred_all.append(pe_pred)
        real_all.append(pe_real)
    pred_all = np.concatenate(pred_all)
    real_all = np.concatenate(real_all)
    return {
        "per_edit_within_cell_spearman_mean": round(float(np.nanmean(rhos)), 4),
        "per_edit_within_cell_spearman_per_cell": [round(float(x), 4) for x in rhos],
        "top_decile_edit_recall": topk_recall(pred_all, real_all, 0.10),
        "n_edits_pooled": int(real_all.size),
    }


# ----------------------------------------------------------------------------- #
def main():
    # ---- load all 12 cells --------------------------------------------------- #
    cells = {}
    missing = []
    for L in LAYERS:
        for s in SEEDS:
            c = load_cell(L, s)
            if c is None:
                missing.append(f"L{L}_s{s}")
            else:
                cells[(L, s)] = c
    if missing:
        print(f"[warn] missing/too-small cells: {missing}")
    by_layer = {L: [cells[(L, s)] for s in SEEDS if (L, s) in cells] for L in LAYERS}

    # ---- reproduction check: within-probe rho(key-cos, damage_removed) ------- #
    canon_path = os.path.join(RESULTS, "C4_causal_table.json")
    canon = json.load(open(canon_path))["layers"]
    repro = {}
    max_abs_diff = 0.0
    for L in LAYERS:
        per_seed = [within_probe_mean(c["cos"], c["removed"]) for c in by_layer[L]]
        recomputed = float(np.mean(per_seed))
        canonical = float(canon[str(L)]["within_probe_spearman"])
        diff = abs(recomputed - canonical)
        max_abs_diff = max(max_abs_diff, diff)
        repro[str(L)] = {
            "canonical_within_probe_rho": round(canonical, 4),
            "recomputed_within_probe_rho": round(recomputed, 4),
            "recomputed_per_seed": [round(x, 4) for x in per_seed],
            "abs_diff": round(diff, 4),
        }
    repro_passed = max_abs_diff <= REPRO_TOL
    repro_summary = {"per_layer": repro, "max_abs_diff": round(max_abs_diff, 4),
                     "tolerance": REPRO_TOL, "passed": bool(repro_passed)}

    print("\n=== REPRODUCTION CHECK: within-probe rho(key-cos, damage_removed) ===")
    print(f"{'layer':>6} {'canonical':>10} {'recomputed':>11} {'abs_diff':>9}")
    for L in LAYERS:
        r = repro[str(L)]
        print(f"{L:>6} {r['canonical_within_probe_rho']:>10.4f} "
              f"{r['recomputed_within_probe_rho']:>11.4f} {r['abs_diff']:>9.4f}")
    print(f"max_abs_diff={max_abs_diff:.4f}  tol={REPRO_TOL}  -> "
          f"{'PASS' if repro_passed else 'FAIL'}")

    if not repro_passed:
        # Honesty gate: do not report predictor numbers on a schema we cannot trust.
        out = {"reproduction_check": repro_summary,
               "ABORTED": "reproduction failed (>0.02) — schema misread suspected; "
                          "predictor evaluation withheld."}
        json.dump(out, open(os.path.join(RESULTS, "D3_benefit_predictor_eval.json"), "w"), indent=2)
        print("\n[ABORT] reproduction failed — see JSON; predictor numbers withheld.")
        return

    # ---- predictor definitions ----------------------------------------------- #
    # Two variants, both evaluated below + written to the JSON:
    #   raw_keycos -> raw_predict_cell (zero-param, signed key-cosine)
    #   ols_combo  -> ols_pred_full (full-fit) for calibration/screening; refit per
    #                 split via fit_ols/predict_cell for LOSO / LOLO.
    # Full-data OLS fit (for reported coefficients + in-sample calibration/screening).
    all_cells = [c for L in LAYERS for c in by_layer[L]]
    full_model = fit_ols(all_cells)
    coeffs_full = {FEATURES[i]: round(float(full_model["beta"][i]), 5) for i in range(len(FEATURES))}
    coeffs_full["intercept"] = round(float(full_model["beta"][-1]), 5)

    def ols_pred_full(c):
        return predict_cell(full_model, c)

    # ---- in-sample within-probe rho of predicted vs realized (per layer) ----- #
    insample = {"raw_keycos": {}, "ols_combo": {}}
    for L in LAYERS:
        insample["raw_keycos"][str(L)] = round(
            float(np.mean([within_probe_mean(raw_predict_cell(c), c["removed"]) for c in by_layer[L]])), 4)
        insample["ols_combo"][str(L)] = round(
            float(np.mean([within_probe_mean(ols_pred_full(c), c["removed"]) for c in by_layer[L]])), 4)

    # ---- LOSO: fit on 2 seeds, evaluate held-out seed (per layer) ------------ #
    loso = {"raw_keycos": {}, "ols_combo": {}}
    for L in LAYERS:
        raw_folds, ols_folds = [], []
        for s in SEEDS:
            if (L, s) not in cells:
                continue
            test = cells[(L, s)]
            train = [cells[(L, ss)] for ss in SEEDS if ss != s and (L, ss) in cells]
            raw_folds.append(within_probe_mean(raw_predict_cell(test), test["removed"]))
            if train:
                m = fit_ols(train)
                ols_folds.append(within_probe_mean(predict_cell(m, test), test["removed"]))
        loso["raw_keycos"][str(L)] = {
            "mean": round(float(np.mean(raw_folds)), 4),
            "per_fold": [round(x, 4) for x in raw_folds]}
        loso["ols_combo"][str(L)] = {
            "mean": round(float(np.mean(ols_folds)), 4),
            "per_fold": [round(x, 4) for x in ols_folds]}

    # ---- LOLO: fit on the other 3 layers, evaluate held-out layer ------------ #
    lolo = {"raw_keycos": {}, "ols_combo": {}}
    for L in LAYERS:
        test_cells = by_layer[L]
        train_cells = [c for LL in LAYERS if LL != L for c in by_layer[LL]]
        raw_rho = float(np.mean([within_probe_mean(raw_predict_cell(c), c["removed"]) for c in test_cells]))
        m = fit_ols(train_cells)
        ols_rho = float(np.mean([within_probe_mean(predict_cell(m, c), c["removed"]) for c in test_cells]))
        lolo["raw_keycos"][str(L)] = round(raw_rho, 4)
        lolo["ols_combo"][str(L)] = {
            "held_out_within_probe_rho": round(ols_rho, 4),
            "train_coefficients": {FEATURES[i]: round(float(m["beta"][i]), 5) for i in range(len(FEATURES))},
        }

    # ---- calibration + screening (pooled per layer, full-fit predictions) ---- #
    calibration = {"raw_keycos": {}, "ols_combo": {}}
    screening = {"raw_keycos": {}, "ols_combo": {}}
    for L in LAYERS:
        real = np.concatenate([c["removed"].reshape(-1) for c in by_layer[L]])
        raw_pred = np.concatenate([raw_predict_cell(c).reshape(-1) for c in by_layer[L]])
        ols_pred = np.concatenate([ols_pred_full(c).reshape(-1) for c in by_layer[L]])
        calibration["raw_keycos"][str(L)] = quartile_calibration(raw_pred, real)
        calibration["ols_combo"][str(L)] = quartile_calibration(ols_pred, real)
        screening["raw_keycos"][str(L)] = topk_recall(raw_pred, real, 0.10)
        screening["ols_combo"][str(L)] = topk_recall(ols_pred, real, 0.10)

    # ---- deployable per-edit magnitude gate ---------------------------------- #
    # raw = held-out-clean (zero parameters). ols uses the FULL-DATA fit and is
    # therefore IN-SAMPLE (the evaluated edits are in the fit) — labelled as such.
    per_edit = {
        "note": ("raw_keycos_heldout_clean is the clean held-out-style headline "
                 "(zero-param). ols_combo_in_sample_full_fit is fit on ALL data "
                 "including the evaluated edits -> IN-SAMPLE, not a held-out number."),
        "headline_held_out_clean": "raw_keycos",
        "raw_keycos_heldout_clean": {},
        "ols_combo_in_sample_full_fit": {},
    }
    for L in LAYERS:
        per_edit["raw_keycos_heldout_clean"][str(L)] = per_edit_gate(by_layer[L], raw_predict_cell)
        per_edit["ols_combo_in_sample_full_fit"][str(L)] = per_edit_gate(by_layer[L], ols_pred_full)

    # ---- feature-ablation LOSO per layer: geometry vs norm-growth-magnitude --- #
    # Decomposes the ols_combo LOSO gain. key_cos_only reproduces the raw within-
    # probe rho (single standardized feature -> identical ranks). The point is L14:
    # the gain there is carried by S(=norm-growth), not key-geometry or its
    # interaction (the interaction is inert / fractionally negative).
    ablation_sets = {
        "key_cos_only": [0],
        "S_only": [1],
        "key_cos+S": [0, 1],
        "full(key_cos+S+interaction)": [0, 1, 2],
    }
    feature_ablation = {}
    for L in LAYERS:
        rows = {}
        for name, idx in ablation_sets.items():
            folds = []
            for s in SEEDS:
                if (L, s) not in cells:
                    continue
                test = cells[(L, s)]
                train = [cells[(L, ss)] for ss in SEEDS if ss != s and (L, ss) in cells]
                m = fit_ols(train, idx)
                folds.append(within_probe_mean(predict_cell(m, test), test["removed"]))
            rows[name] = {"loso_within_probe_rho": round(float(np.mean(folds)), 4),
                          "per_fold": [round(x, 4) for x in folds]}
        feature_ablation[str(L)] = rows
    feature_ablation["interpretation"] = (
        "At L14 the ols_combo gain (0.302->0.536 LOSO) is a NORM-GROWTH-MAGNITUDE "
        "effect: S_only already reaches ~0.494 while key_cos_only stays at 0.302 and "
        "adding the S*|cos| interaction does not improve on key_cos+S (inert). This is "
        "the documented L14 regime transition (norm-growth overtakes key-cosine); key "
        "geometry does NOT predict better at L14. At L8/L10/L12 key-cos is the primary "
        "carrier and S adds little to nothing.")

    # ---- screening lift summary (scoped ranges, not a pooled single range) ---- #
    def _lift(recall, chance):
        return round(recall / chance, 2) if (recall is not None and chance) else None
    pair_lifts, edit_lifts = [], []
    screening_lift = {"pair_level": {}, "per_edit": {}}
    for L in LAYERS:
        sL = str(L)
        pr = {"raw": _lift(screening["raw_keycos"][sL]["recall"], screening["raw_keycos"][sL]["chance"]),
              "ols": _lift(screening["ols_combo"][sL]["recall"], screening["ols_combo"][sL]["chance"])}
        er = {"raw": _lift(per_edit["raw_keycos_heldout_clean"][sL]["top_decile_edit_recall"]["recall"],
                           per_edit["raw_keycos_heldout_clean"][sL]["top_decile_edit_recall"]["chance"]),
              "ols": _lift(per_edit["ols_combo_in_sample_full_fit"][sL]["top_decile_edit_recall"]["recall"],
                           per_edit["ols_combo_in_sample_full_fit"][sL]["top_decile_edit_recall"]["chance"])}
        screening_lift["pair_level"][sL] = pr
        screening_lift["per_edit"][sL] = er
        pair_lifts += [pr["raw"], pr["ols"]]
        edit_lifts += [er["raw"], er["ols"]]
    screening_lift["pair_level_range_x"] = [min(pair_lifts), max(pair_lifts)]
    screening_lift["per_edit_range_x"] = [min(edit_lifts), max(edit_lifts)]
    screening_lift["note"] = (
        "Lift = recall / chance(=0.10). Ranges are scoped by view, NOT pooled into one "
        "range. Per-edit raw L14 is the floor at ~1.4x (marginally above chance).")

    # ---- assemble + write ---------------------------------------------------- #
    out = {
        "artifact": "D3 benefit-magnitude predictor (geometry -> AlphaEdit collateral-damage removed)",
        "framing": ("Deployable pre-edit MAGNITUDE gate: predict how much collateral "
                    "damage AlphaEdit removes vs ROME on an edit. NOT a router "
                    "(the AlphaEdit>ROME choice is degenerate 12/12); this predicts "
                    "the benefit MAGNITUDE."),
        "metric": "damage_logit; realized benefit = rome_damage - alpha_damage on matched (edit,probe) pairs",
        "statistic": "signed within-probe Spearman(prediction, realized); rank-based, mirrors aggregate_g4_causal",
        "data": {
            "rome_glob": f"results/matrices/{ROME_TMPL}",
            "alpha_glob": f"results/matrices/{ALPHA_TMPL}",
            "projector_source": "probes (the canonical C4 table cells)",
            "layers": LAYERS, "seeds": SEEDS,
            "cells_loaded": [f"L{L}_s{s}" for (L, s) in sorted(cells)],
            "cells_missing": missing,
            "n_pairs_per_layer": {str(L): int(sum(c["n_pairs"] for c in by_layer[L])) for L in LAYERS},
            "n_edits_per_cell": {f"L{L}_s{s}": cells[(L, s)]["n_edits"] for (L, s) in sorted(cells)},
        },
        "reproduction_check": repro_summary,
        "predictors": {
            "raw_keycos": "prediction = signed key-cosine (zero parameters; deployable from the edit key alone)",
            "ols_combo": {
                "features": FEATURES,
                "note": "closed-form OLS on z-scored features (full-data fit shown; splits refit)",
                "coefficients_full_fit": coeffs_full,
            },
        },
        "evaluation": {
            "in_sample_within_probe_rho": insample,
            "leave_one_seed_out": loso,
            "leave_one_layer_out": lolo,
            "quartile_calibration_pred_to_realized": calibration,
            "decile_screening_pair_level": screening,
            "decile_screening_lift_scoped": screening_lift,
            "feature_ablation_loso": feature_ablation,
            "per_edit_magnitude_gate": per_edit,
        },
        "caveats": [
            "L14 ATTRIBUTION: the ols_combo L14 gain (0.302->0.536 LOSO) is a "
            "norm-growth-MAGNITUDE effect; key-geometry and the S*|cos| interaction "
            "are inert there (ablation: key_cos_only 0.302, S_only 0.494, key_cos+S "
            "0.536, full 0.536). Consistent with the documented L14 regime transition "
            "(norm-growth overtakes key-cosine). Do NOT read this as geometry predicting "
            "better at L14 — it does not.",
            "OLS gain is claimable only at L12 (+0.065 LOSO) and L14 (+0.234, and that "
            "is a norm-growth effect); the L8 (+0.007) and L10 (+0.018) ols gains over "
            "raw key-cos are within seed noise.",
            "Per-edit ols_combo rhos (0.459/0.630/0.784/0.433) are IN-SAMPLE (full-data "
            "fit); the clean held-out-style per-edit headline is raw_keycos "
            "(0.446/0.601/0.725/0.309).",
            "Single edit family (ROME collateral, AlphaEdit as the repair editor); "
            "does not cover MEMIT/FT/GRACE editors.",
            "CounterFact only — not validated on zsRE / MQuAKE / RippleEdits benefit.",
            "Llama-3.2-1B only — cross-arch geometry law does NOT transfer off Llama "
            "(Qwen L14 sign-inverts), so these predictor coefficients are Llama-specific.",
            "AlphaEdit projector = 'probes' source (the C4 table). The disjoint-projector "
            "holdout/generic causal cells (E6) are NOT used here; benefit magnitudes under "
            "a disjoint projector may differ.",
            "S (=norm_growth) requires the solved ROME update, so ols_combo costs one "
            "update solve; raw_keycos needs only the edit key.",
            "LOLO transfers a single coefficient vector across layers whose damage scales "
            "differ ~3x; rank-based eval tolerates this but absolute calibration does not "
            "(refit per deployment layer).",
            "Realized benefit is measured, not predicted-at-fit-time out-of-model; the honest "
            "generalization tests are LOSO / LOLO (held-out data), not in-sample rho.",
            "Per-edit gate aggregates by MEAN over the known-probe bank; a deployment with a "
            "different probe/reference bank will see different magnitudes.",
        ],
    }
    out_path = os.path.join(RESULTS, "D3_benefit_predictor_eval.json")
    json.dump(out, open(out_path, "w"), indent=2)

    # ---- compact human-readable table ---------------------------------------- #
    print("\n=== PREDICTOR: held-out within-probe rho(prediction, damage_removed) ===")
    print(f"{'layer':>6} | {'raw insamp':>10} {'raw LOSO':>9} {'raw LOLO':>9} "
          f"| {'ols insamp':>10} {'ols LOSO':>9} {'ols LOLO':>9}")
    for L in LAYERS:
        sL = str(L)
        print(f"{L:>6} | {insample['raw_keycos'][sL]:>10.4f} "
              f"{loso['raw_keycos'][sL]['mean']:>9.4f} {lolo['raw_keycos'][sL]:>9.4f} "
              f"| {insample['ols_combo'][sL]:>10.4f} "
              f"{loso['ols_combo'][sL]['mean']:>9.4f} "
              f"{lolo['ols_combo'][sL]['held_out_within_probe_rho']:>9.4f}")

    print(f"\nols_combo full-fit coefficients (z-scored): {coeffs_full}")

    print("\n=== QUARTILE CALIBRATION (raw_keycos): pred-quartile -> realized mean damage_removed ===")
    print(f"{'layer':>6} | {'Q1(low)':>9} {'Q2':>9} {'Q3':>9} {'Q4(high)':>9}")
    for L in LAYERS:
        rows = calibration["raw_keycos"][str(L)]
        vals = [r["realized_mean_damage_removed"] for r in rows]
        print(f"{L:>6} | " + " ".join(f"{v:>9.3f}" for v in vals))

    print("\n=== L14-focus FEATURE ABLATION (LOSO within-probe rho) — gain is norm-growth, not geometry ===")
    print(f"{'layer':>6} | {'keycos_only':>12} {'S_only':>8} {'keycos+S':>9} {'full':>8}")
    for L in LAYERS:
        r = feature_ablation[str(L)]
        print(f"{L:>6} | {r['key_cos_only']['loso_within_probe_rho']:>12.4f} "
              f"{r['S_only']['loso_within_probe_rho']:>8.4f} "
              f"{r['key_cos+S']['loso_within_probe_rho']:>9.4f} "
              f"{r['full(key_cos+S+interaction)']['loso_within_probe_rho']:>8.4f}")

    print("\n=== DECILE SCREENING: top-10% recall (chance 0.10) + lift x; per-edit ols = IN-SAMPLE ===")
    print(f"{'layer':>6} | {'raw pair':>9} {'ols pair':>9} {'raw edit':>9} {'olsIS edit':>10} "
          f"| {'pair x':>7} {'edit x':>7}")
    for L in LAYERS:
        sL = str(L)
        re_ = per_edit["raw_keycos_heldout_clean"][sL]["top_decile_edit_recall"]["recall"]
        oe_ = per_edit["ols_combo_in_sample_full_fit"][sL]["top_decile_edit_recall"]["recall"]
        print(f"{L:>6} | {screening['raw_keycos'][sL]['recall']:>9.3f} "
              f"{screening['ols_combo'][sL]['recall']:>9.3f} {re_:>9.3f} {oe_:>10.3f} "
              f"| {screening_lift['pair_level'][sL]['raw']:>7.2f} "
              f"{screening_lift['per_edit'][sL]['raw']:>7.2f}")
    print(f"scoped lift ranges: pair-level {screening_lift['pair_level_range_x']}x  "
          f"per-edit {screening_lift['per_edit_range_x']}x  "
          f"(per-edit raw L14 ~1.4x = near chance)")

    print("\n=== PER-EDIT MAGNITUDE GATE: Spearman(pred, realized) across edits ===")
    print(f"{'layer':>6} | {'raw(hoclean)':>13} {'ols(in-samp)':>13}")
    for L in LAYERS:
        sL = str(L)
        print(f"{L:>6} | {per_edit['raw_keycos_heldout_clean'][sL]['per_edit_within_cell_spearman_mean']:>13.4f} "
              f"{per_edit['ols_combo_in_sample_full_fit'][sL]['per_edit_within_cell_spearman_mean']:>13.4f}")

    print(f"\n[d3] wrote {out_path}")


if __name__ == "__main__":
    main()
