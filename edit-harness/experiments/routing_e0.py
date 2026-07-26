"""routing_e0.py — D4 / T3.2 weight-vs-memory ROUTING gate (E0, CPU-only, numpy-only).

The RRDA-line question, re-scoped to our edge: for each incoming edit, decide whether to
apply it in the WEIGHTS (a locate-then-edit weight editor — ROME by default, AlphaEdit as a
secondary skyline) or store it in an external MEMORY (GRACE, a codebook whose ΔW ≡ 0, so it
inflicts EXACTLY zero collateral damage but consumes a capacity/retrieval slot). Weight edits
are free at inference but damage-prone; memory edits are damage-free but capacity-limited.
The D3 result — pre-edit key-geometry ranks which edits a weight editor will most damage
(per-edit held-out Spearman(raw key-cos, damage) = 0.725 at L12) — is exactly the signal a
router needs to send the predicted-worst edits to memory.

PRIMARY OBJECTIVE (ONE; see docs/plans/ANALYSIS-D4-ROUTING-E0-20260714.md — RETROSPECTIVE
analysis plan, NOT a pre-registration, since the EGL cells pre-existed and the outcome was
known at authoring time; the gate is a retrospective screen and a confirmatory claim needs E1):
    Capacity-constrained memory routing. A memory budget admits at most a fraction f_mem of
    edits to GRACE; the rest take the weight editor. Minimize total residual collateral damage
        D(policy) = sum_{i NOT routed to GRACE} weight_damage(i)          (GRACE contributes 0)
    equivalently MAXIMIZE the damage AVOIDED by offloading edits to memory
        reduction(policy) = sum_{i routed to GRACE} (weight_damage(i) - grace_damage(i)).
    Per-edit damage = mean SIGNED damage_logit over the base-known probe columns on the shared
    edit_ok rows (identical masking to d3_benefit_predictor / route_per_edit_sweep). Metric is
    signed damage_logit — never AUROC (the banned probe-marginal artifact).

    Oracle-efficiency  eta(policy) = reduction(policy) / reduction(oracle at the same budget k),
    where oracle routes the top-k edits by ACTUAL weight damage. Regret = 1 - eta.

POLICIES (all pre-registered): oracle (upper bound), geometry_raw (route top-k by the
zero-parameter signed key-cos signal — the deployable D3 headline), geometry_ols (route by the
d3 OLS combo predicted damage; IN-SAMPLE reference), random (route a random k-subset; analytic
expectation f_mem * total plus an empirical resample band), always_weight (k=0 corner) and the
unbudgeted always_grace / always_alpha SKYLINES.

FEATURE SET (== the D3 predictor's features, reused, not duplicated):
  key_cos (signed key-cosine to the probe bank, per-edit = row-mean over known probes; the
  zero-parameter primary router signal), S (= norm_growth; needs the solved weight update),
  and the S*|key_cos| interaction. This module imports d3_benefit_predictor.{spearman,
  feat_stack,fit_ols,predict_cell,FEATURES} so the router shares the exact predictor.

BINDING WORDING CAVEAT (from memory d3-benefit-predictor-result-20260712): the L14 predictor
lift (0.302->0.536) is NORM-GROWTH-carried; key GEOMETRY is inert there. The PRIMARY gate is
therefore L12 ONLY (the geometry-valid layer). L14 may only be run as a DESCRIPTIVE secondary
whose any routing gain is attributed to norm-growth magnitude, never to key geometry.

DATA-PROVENANCE / PRE-REGISTRATION HONESTY: the per-editor damage cells this reads (the EGL
5-editor grid, produced by run_u6.sh / run_revins.sh) PRE-EXIST — the per-editor damage
distributions were known at B6 submission. What is NEW and unseen here is the ROUTING
accounting (eta, regret, the primary gate). The gate thresholds are frozen in the prereg from
D3's INDEPENDENT held-out per-edit rank correlation (0.725 at L12), not from any routing
computation. See the prereg §0 honesty note.

Data (read-only; CPU): results/matrices/egl_llama1b_{rome,alpha,grace}_cf_L{L}_s{0,1,2}.npz
Output: results/analysis/D4_routing_e0.json
Self-test: `python routing_e0.py --selftest` proves the reduction/oracle/regret accounting on
synthetic fixtures with hand-computed exact values (planted oracle > random > always-weight,
anti-correlated predictor < random, nonzero-grace subtraction, scalarized-cost threshold).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

# Reuse the D3 predictor's statistics + feature machinery rather than re-implementing it.
import d3_benefit_predictor as d3

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # edit-harness/
RESULTS = os.path.join(ROOT, "results")
MATRICES = os.path.join(RESULTS, "matrices")
ANALYSIS = os.path.join(RESULTS, "analysis")

# Pre-registered constants (frozen; see the prereg).
LAYERS_PRIMARY = [12]
SEEDS = [0, 1, 2]
F_MEM_PRIMARY = 0.20
F_MEM_SWEEP = [0.05, 0.10, 0.20, 0.30, 0.50]
CMEM_GRID = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]  # scalarized-cost secondary
N_RANDOM = 500
RNG_SEED = 12345
ROME_TMPL = "egl_llama1b_rome_cf_L{L}_s{s}.npz"
ALPHA_TMPL = "egl_llama1b_alpha_cf_L{L}_s{s}.npz"
GRACE_TMPL = "egl_llama1b_grace_cf_L{L}_s{s}.npz"
GRACE_ZERO_TOL = 1e-5  # GRACE dW==0 => damage must be identically 0; larger aborts (schema distrust)

# Primary-gate thresholds (frozen).
GATE_ETA_MEAN = 0.70      # mean-across-seeds oracle-efficiency of the headline (raw) router
GATE_ETA_PERSEED = 0.60   # per-seed floor, required in >= GATE_PERSEED_MIN seeds
GATE_PERSEED_MIN = 2
GATE_MARGIN = 0.25        # mean eta(geometry) - eta(random)
KILL_ETA_MEAN = 0.50
GREY_MARGIN_LO = 0.10


def r4(x):
    return None if x is None else round(float(x), 4)


# --------------------------------------------------------------------------- #
# Routing accounting — the load-bearing math, kept in pure 1-D helpers so the  #
# self-test can plant exact values without touching any npz.                   #
# --------------------------------------------------------------------------- #
def budget_k(n, f_mem):
    """Number of edits admitted to memory at budget fraction f_mem (>=1 if any)."""
    return max(1, int(round(f_mem * n))) if f_mem > 0 else 0


def policy_reduction(delta, order_desc, k):
    """Damage AVOIDED by routing the first k edits of `order_desc` to memory.
    delta[i] = weight_damage(i) - grace_damage(i); reduction = sum of delta over the routed
    set (the actual delta, regardless of the order that selected them)."""
    if k <= 0:
        return 0.0
    return float(np.asarray(delta, float)[np.asarray(order_desc)[:k]].sum())


def oracle_order(delta):
    """Route the largest-delta edits first (maximizes reduction at every budget)."""
    return np.argsort(-np.asarray(delta, float), kind="mergesort")


def signal_order(signal):
    """Route the highest-predicted-damage edits first."""
    return np.argsort(-np.asarray(signal, float), kind="mergesort")


def eta(reduction, oracle_reduction):
    """Oracle-efficiency; None when the oracle itself removes no damage (undefined regret)."""
    if oracle_reduction is None or abs(oracle_reduction) < 1e-12:
        return None
    return reduction / oracle_reduction


def random_expected_reduction(delta, k):
    """Analytic E[reduction] of a uniform random k-subset = (k/n) * sum(delta)."""
    delta = np.asarray(delta, float)
    n = delta.size
    return 0.0 if n == 0 else float(k) / n * float(delta.sum())


def random_empirical(delta, k, n_boot=N_RANDOM, seed=RNG_SEED):
    """Empirical mean and 95% band of reduction over random k-subsets (for a CI on random)."""
    delta = np.asarray(delta, float)
    n = delta.size
    if k <= 0 or n == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    reds = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.choice(n, size=k, replace=False)
        reds[b] = delta[idx].sum()
    return float(reds.mean()), float(np.percentile(reds, 2.5)), float(np.percentile(reds, 97.5))


def scalarized_cost(weight_dmg, grace_dmg, route_to_grace, c_mem):
    """cost = residual weight damage (edits kept in weights) + c_mem * (#edits in memory) +
    grace damage of the memory-routed edits. route_to_grace is a boolean per-edit mask."""
    weight_dmg = np.asarray(weight_dmg, float)
    grace_dmg = np.asarray(grace_dmg, float)
    m = np.asarray(route_to_grace, bool)
    return float(weight_dmg[~m].sum() + grace_dmg[m].sum() + c_mem * int(m.sum()))


# --------------------------------------------------------------------------- #
# Loading — aligned rome/alpha/grace cells; reuses d3's masking convention.    #
# --------------------------------------------------------------------------- #
def _load_npz(tmpl, L, s):
    p = os.path.join(MATRICES, tmpl.format(L=L, s=s))
    return (np.load(p), p) if os.path.exists(p) else (None, p)


def load_routing_cell(L, s, weight_editor="rome"):
    """Return the per-edit routing quantities for one (layer, seed), or None if unavailable.

    Row mask  = weight-editor edit_ok & GRACE edit_ok (an edit must be applicable by whichever
                editor it is routed to; both arms must have succeeded to be comparable).
    Col mask  = base model knows the probe (weight-editor pre_p > 0.05), >=5-or-all fallback —
                identical to d3.load_cell / route_per_edit_sweep.
    The returned dict is d3-compatible (keys `cos`, `S`, `removed`=weight damage) so d3.fit_ols
    / d3.predict_cell can be reused verbatim for the OLS router, PLUS the per-edit routing
    arrays w_dmg / g_dmg / a_dmg (signed mean damage over the known-probe bank)."""
    wt_tmpl = ROME_TMPL if weight_editor == "rome" else ALPHA_TMPL
    dw, wpath = _load_npz(wt_tmpl, L, s)
    dg, gpath = _load_npz(GRACE_TMPL, L, s)
    da, apath = _load_npz(ALPHA_TMPL, L, s)  # alpha always loaded for the secondary skyline
    if dw is None or dg is None:
        return None
    COS = dw["COS"].astype(float)
    Dw = dw["damage_logit"].astype(float)
    Dg = dg["damage_logit"].astype(float)
    Da = da["damage_logit"].astype(float) if da is not None else np.zeros_like(Dw)
    if not (COS.shape == Dw.shape == Dg.shape):
        return None
    row = np.ones(COS.shape[0], bool)
    if "edit_ok" in dw.files:
        row &= dw["edit_ok"].astype(float) > 0.5
    if "edit_ok" in dg.files:
        row &= dg["edit_ok"].astype(float) > 0.5
    col = np.ones(COS.shape[1], bool)
    if "pre_p" in dw.files:
        c = dw["pre_p"].astype(float) > 0.05
        if c.sum() >= 5:
            col = c
    cos2d = COS[row][:, col]
    if cos2d.size < 20:
        return None
    w2d = Dw[row][:, col]
    g2d = Dg[row][:, col]
    a2d = Da[row][:, col]
    S = dw["norm_growth"].astype(float)[row]
    S2d = np.repeat(S[:, None], cos2d.shape[1], axis=1)
    return {
        "L": L, "s": s, "weight_editor": weight_editor,
        "cos": cos2d, "S": S2d, "removed": w2d,   # d3-compatible: predict weight damage
        # per-edit routing scalars (signed mean over the known-probe bank)
        "w_dmg": w2d.mean(axis=1),
        "g_dmg": g2d.mean(axis=1),
        "a_dmg": a2d.mean(axis=1),
        "grace_max_abs": float(np.max(np.abs(g2d))),
        "n_edits": cos2d.shape[0], "n_probes": cos2d.shape[1],
        "paths": {"weight": wpath, "grace": gpath, "alpha": apath},
    }


def per_edit_signals(cell, ols_model=None):
    """Router signals per edit: raw = mean signed key-cos over known probes (d3 zero-param
    headline); ols = d3 linear predictor of weight damage, mean over probes (in-sample)."""
    raw = cell["cos"].mean(axis=1)
    ols = None
    if ols_model is not None:
        ols = d3.predict_cell(ols_model, cell).mean(axis=1)
    return raw, ols


# --------------------------------------------------------------------------- #
# One cell -> all policy reductions + eta at a given budget fraction.          #
# --------------------------------------------------------------------------- #
def evaluate_cell(cell, f_mem, ols_model=None, rng_seed=RNG_SEED):
    delta = cell["w_dmg"] - cell["g_dmg"]              # damage removed by routing edit->GRACE
    n = delta.size
    k = budget_k(n, f_mem)
    o_order = oracle_order(delta)
    red_oracle = policy_reduction(delta, o_order, k)
    raw, ols = per_edit_signals(cell, ols_model)
    red_raw = policy_reduction(delta, signal_order(raw), k)
    red_ols = policy_reduction(delta, signal_order(ols), k) if ols is not None else None
    red_rand_e = random_expected_reduction(delta, k)
    red_rand_m, rlo, rhi = random_empirical(delta, k, seed=rng_seed + cell["s"])
    total = float(delta.sum())
    return {
        "n_edits": int(n), "k_routed": int(k), "f_mem": f_mem,
        "total_delta": r4(total),
        "reduction": {"oracle": r4(red_oracle), "geometry_raw": r4(red_raw),
                      "geometry_ols": r4(red_ols), "random_expected": r4(red_rand_e),
                      "random_empirical_mean": r4(red_rand_m),
                      "random_empirical_ci95": [r4(rlo), r4(rhi)],
                      "always_grace_unbudgeted": r4(total)},
        "eta": {"oracle": r4(eta(red_oracle, red_oracle)),
                "geometry_raw": r4(eta(red_raw, red_oracle)),
                "geometry_ols": r4(eta(red_ols, red_oracle)) if red_ols is not None else None,
                "random_expected": r4(eta(red_rand_e, red_oracle))},
        "per_edit_signal_spearman_vs_weight_damage": {
            "raw": r4(d3.spearman(raw, cell["w_dmg"])),
            "ols": r4(d3.spearman(ols, cell["w_dmg"])) if ols is not None else None},
    }


def scalarized_curve(cell, ols_model=None):
    """Secondary: for each c_mem, oracle routes edit->GRACE iff (w-g) > c_mem; geometry
    thresholds the raw signal at the matched routed-count. Reports the corner skylines too."""
    w, g, a = cell["w_dmg"], cell["g_dmg"], cell["a_dmg"]
    delta = w - g
    raw, _ = per_edit_signals(cell, ols_model)
    rows = []
    for cm in CMEM_GRID:
        m_oracle = delta > cm
        korf = int(m_oracle.sum())
        # geometry: route the top-korf edits by raw signal (matched count to the oracle)
        m_geom = np.zeros_like(m_oracle)
        if korf > 0:
            m_geom[signal_order(raw)[:korf]] = True
        rows.append({
            "c_mem": cm,
            "oracle_cost": r4(scalarized_cost(w, g, m_oracle, cm)),
            "geometry_raw_cost": r4(scalarized_cost(w, g, m_geom, cm)),
            "random_cost_expected": r4(w.sum() - (korf / w.size) * delta.sum()
                                       + cm * korf + (korf / w.size) * g.sum())
            if w.size else None,
            "n_routed_to_grace": korf,
        })
    return {
        "always_rome_cost_ref": r4(float(w.sum())),      # c_mem-independent weight-only corner
        "always_grace_cost_at_cmem": {str(cm): r4(scalarized_cost(w, g, np.ones_like(w, bool), cm))
                                       for cm in CMEM_GRID},
        "always_alpha_cost_ref": r4(float(a.sum())),     # projector-based weight skyline
        "curve": rows,
    }


# --------------------------------------------------------------------------- #
def _verdict(mean_eta_raw, perseed_pass, margin, beats_random_count):
    if mean_eta_raw is None:
        return "INCONCLUSIVE"
    if mean_eta_raw < KILL_ETA_MEAN or beats_random_count < GATE_PERSEED_MIN:
        return "KILL"
    if (mean_eta_raw >= GATE_ETA_MEAN and perseed_pass >= GATE_PERSEED_MIN
            and margin is not None and margin >= GATE_MARGIN
            and beats_random_count >= GATE_PERSEED_MIN):
        return "PASS"
    return "GREY"


def run_analysis(layers, seeds, weight_editor, f_mem, out_path):
    # ---- load cells --------------------------------------------------------- #
    cells, missing = {}, []
    for L in layers:
        for s in seeds:
            c = load_routing_cell(L, s, weight_editor)
            (cells.__setitem__((L, s), c) if c is not None else missing.append(f"L{L}_s{s}"))
    if not cells:
        out = {"ABORTED": "no routing cells loadable", "cells_missing": missing}
        os.makedirs(ANALYSIS, exist_ok=True)
        json.dump(out, open(out_path, "w"), indent=2)
        print("[routing] ABORT: no cells"); return out

    # ---- HARD-FAIL: the primary-gate layer must have EVERY requested seed. A silent
    # 2-seed gate would let a "2/3 seeds" rule pass on a degenerate 2-seed sample. -- #
    primary_L = layers[0]
    primary_seeds_loaded = [s for s in seeds if (primary_L, s) in cells]
    if set(primary_seeds_loaded) != set(seeds):
        out = {"ABORTED": f"primary layer L{primary_L} missing seed(s) "
                          f"{sorted(set(seeds) - set(primary_seeds_loaded))}; the per-seed gate "
                          f"requires ALL {len(seeds)} requested seeds loaded (no silent short gate).",
               "cells_missing": missing,
               "cells_loaded": [f"L{L}_s{s}" for (L, s) in sorted(cells)]}
        os.makedirs(ANALYSIS, exist_ok=True)
        json.dump(out, open(out_path, "w"), indent=2)
        print(f"[routing] ABORT: primary L{primary_L} not all seeds loaded "
              f"({len(primary_seeds_loaded)}/{len(seeds)})"); return out

    # ---- honesty gate: GRACE damage MUST be identically 0 ------------------- #
    gmax = max(c["grace_max_abs"] for c in cells.values())
    if gmax > GRACE_ZERO_TOL:
        out = {"ABORTED": f"GRACE damage not ~0 (max|.|={gmax:.3g} > {GRACE_ZERO_TOL}); "
                          "routing premise (dW==0 => zero collateral) violated; schema distrust.",
               "cells_loaded": [f"L{L}_s{s}" for (L, s) in sorted(cells)]}
        os.makedirs(ANALYSIS, exist_ok=True)
        json.dump(out, open(out_path, "w"), indent=2)
        print(f"[routing] ABORT: GRACE damage max|.|={gmax:.3g}"); return out

    # ---- OLS router: fit d3 predictor of weight damage (in-sample reference) - #
    ols_model = d3.fit_ols([cells[k] for k in sorted(cells)])

    # ---- PRIMARY gate: per-seed eta at f_mem on the primary layer(s) -------- #
    per_seed = {}
    for s in seeds:
        if (primary_L, s) in cells:
            per_seed[str(s)] = evaluate_cell(cells[(primary_L, s)], f_mem, ols_model)
    etas_raw = [v["eta"]["geometry_raw"] for v in per_seed.values() if v["eta"]["geometry_raw"] is not None]
    etas_rand = [v["eta"]["random_expected"] for v in per_seed.values() if v["eta"]["random_expected"] is not None]
    mean_eta_raw = float(np.mean(etas_raw)) if etas_raw else None
    mean_eta_rand = float(np.mean(etas_rand)) if etas_rand else None
    perseed_pass = sum(1 for e in etas_raw if e >= GATE_ETA_PERSEED)
    beats_random_count = sum(1 for v in per_seed.values()
                             if v["eta"]["geometry_raw"] is not None
                             and v["eta"]["random_expected"] is not None
                             and v["eta"]["geometry_raw"] > v["eta"]["random_expected"])
    margin = (mean_eta_raw - mean_eta_rand) if (mean_eta_raw is not None and mean_eta_rand is not None) else None
    verdict = _verdict(mean_eta_raw, perseed_pass, margin, beats_random_count)

    # ---- SECONDARY: budget sweep + scalarized cost (primary layer) ---------- #
    budget_sweep = {}
    for f in F_MEM_SWEEP:
        rows = {str(s): evaluate_cell(cells[(primary_L, s)], f, ols_model)
                for s in seeds if (primary_L, s) in cells}
        e_raw = [r["eta"]["geometry_raw"] for r in rows.values() if r["eta"]["geometry_raw"] is not None]
        e_rand = [r["eta"]["random_expected"] for r in rows.values() if r["eta"]["random_expected"] is not None]
        budget_sweep[str(f)] = {
            "mean_eta_geometry_raw": r4(np.mean(e_raw)) if e_raw else None,
            "mean_eta_random_expected": r4(np.mean(e_rand)) if e_rand else None,
            "per_seed": rows}
    scalar = {str(s): scalarized_curve(cells[(primary_L, s)], ols_model)
              for s in seeds if (primary_L, s) in cells}

    # ---- SECONDARY: L14 (or any non-primary layer) DESCRIPTIVE only --------- #
    descriptive_layers = {}
    for L in layers[1:]:
        rows = {str(s): evaluate_cell(cells[(L, s)], f_mem, ols_model)
                for s in seeds if (L, s) in cells}
        if rows:
            descriptive_layers[str(L)] = {
                "NOTE": ("DESCRIPTIVE ONLY. Any routing gain here is attributed to norm-growth "
                         "MAGNITUDE, not key geometry (L14 key-geometry is inert; the D3 lift is "
                         "norm-growth-carried). Not part of the primary gate."),
                "per_seed": rows}

    out = {
        "artifact": "D4/T3.2 weight-vs-memory routing E0 (ROME/AlphaEdit weight vs GRACE memory)",
        "analysis_plan": "docs/plans/ANALYSIS-D4-ROUTING-E0-20260714.md",
        "status_note": ("RETROSPECTIVE analysis on PRE-EXISTING EGL cells (outcome known at "
                        "authoring time) — NOT a pre-registration. The gate below is a "
                        "retrospective screen; a confirmatory claim needs E1 (new data)."),
        "objective": ("capacity-constrained memory routing: minimize total residual collateral "
                      "damage at memory budget f_mem; metric = signed damage_logit; "
                      "oracle-efficiency eta = reduction/reduction(oracle)"),
        "weight_editor": weight_editor,
        "primary_layer": primary_L, "seeds": seeds, "f_mem_primary": f_mem,
        "feature_set": d3.FEATURES,
        "router_signals": {
            "headline": "geometry_raw = mean signed key-cos over known probes (zero-parameter, "
                        "deployable from the edit key alone; the D3 held-out headline)",
            "reference": "geometry_ols = d3 OLS(key_cos, S, S*|cos|) predicting weight damage "
                         "(IN-SAMPLE full-fit; upper reference, not a held-out number)"},
        "data": {
            "rome_glob": f"results/matrices/{ROME_TMPL}",
            "alpha_glob": f"results/matrices/{ALPHA_TMPL}",
            "grace_glob": f"results/matrices/{GRACE_TMPL}",
            "cells_loaded": [f"L{L}_s{s}" for (L, s) in sorted(cells)],
            "cells_missing": missing,
            "grace_max_abs_damage": r4(gmax),
            "provenance_note": ("EGL 5-editor cells PRE-EXIST (run_u6/run_revins); the per-editor "
                                "damage was known at authoring time. NEW here = the routing "
                                "accounting. This router reads the egl_ cells; D3's per-edit rank "
                                "corr (~0.72 @ L12) was measured on the gate_/g4_ cells — the two "
                                "cell families agree to ~0.72, so D3 is a consistency cross-check, "
                                "NOT the source of the thresholds.")},
        "primary_gate": {
            "thresholds_note": ("conventional round-number go/no-go thresholds (not derived from "
                                "any statistic); a retrospective screen, not a pre-registered gate"),
            "thresholds": {"eta_mean>=": GATE_ETA_MEAN, "eta_perseed>=": GATE_ETA_PERSEED,
                           "perseed_min_seeds": GATE_PERSEED_MIN, "margin_vs_random>=": GATE_MARGIN,
                           "kill_if_eta_mean<": KILL_ETA_MEAN,
                           "kill_if_beats_random_seeds<": GATE_PERSEED_MIN},
            "mean_eta_geometry_raw": r4(mean_eta_raw),
            "mean_eta_random_expected": r4(mean_eta_rand),
            "margin_geometry_minus_random": r4(margin),
            "seeds_eta_raw>=perseed_floor": perseed_pass,
            "seeds_geometry_beats_random": beats_random_count,
            "VERDICT": verdict,
            "VERDICT_INTERPRETATION": {
                "PASS": "retrospective gate met — a confirmatory claim requires E1 (new data)",
                "GREY": "retrospective gate partially met — weak router; fold into D3/B6 remark",
                "KILL": "retrospective gate not met — router adds nothing over a blind budget",
                "INCONCLUSIVE": "eta undefined (oracle removed no damage)",
            }.get(verdict, verdict),
            "per_seed": per_seed},
        # MINOR-7: sequential-editing stress is a structured status, not just a caveat string.
        "secondary_sequential_stress": {
            "status": "NOT_RUN",
            "reason": ("edits-accumulate stress needs fresh sequential GPU runs; not part of this "
                       "CPU retrospective. H1 geometry-attribution was NULL for sequential "
                       "(findings-SEQ-ANALYSIS) -> descriptive-only when run."),
            "promoted_to": "E1"},
        "secondary_budget_sweep": budget_sweep,
        "secondary_scalarized_cost": {
            "c_mem_grid": CMEM_GRID,
            "note": ("cost = residual weight damage + c_mem*(#edits in memory) + grace damage of "
                     "memory-routed edits. always_rome / always_grace / always_alpha are the "
                     "corner/skyline points. always_alpha needs a GLOBAL null-space projector "
                     "(preserved-knowledge covariance) — a heavyweight corpus-dependent "
                     "prerequisite the per-edit router does not assume; reported as a SKYLINE."),
            "per_seed": scalar},
        "secondary_descriptive_layers": descriptive_layers,
        "caveats": [
            "L14 (and any non-L12 layer) is DESCRIPTIVE ONLY: its routing signal is norm-growth "
            "MAGNITUDE, not key geometry (D3 L14 lift is norm-growth-carried; geometry inert). "
            "The primary geometry-routing claim is L12 only.",
            "geometry_ols is IN-SAMPLE (full-data fit incl. the evaluated edits); the clean "
            "zero-parameter held-out headline is geometry_raw.",
            "always_alpha SKYLINE requires AlphaEdit's global null-space projector; the router's "
            "premise is the online / no-preserved-knowledge-statistics setting where the "
            "per-edit choice is ROME-vs-GRACE. Do not read always_alpha as the baseline the "
            "router must beat.",
            "GRACE damage==0 is by construction (dW==0 codebook) and confirmed empirically "
            "(honesty gate above); its real cost is retrieval/capacity + no generalization/"
            "ripple — modeled here as the memory budget f_mem / c_mem, not as damage.",
            "Llama-3.2-1B / CounterFact only; the cross-arch geometry law does NOT transfer off "
            "Llama (Qwen L14 sign-inverts), so the router coefficients are Llama-specific.",
            "Sequential-editing stress (edits accumulate) is tracked as the structured "
            "secondary_sequential_stress status field (NOT_RUN here; promoted to E1) — "
            "descriptive-only when run (H1 geometry-attribution NULL, findings-SEQ-ANALYSIS).",
            "RAG memory arm is OUT of E0 scope (needs corpus downloads); registered as E1.",
            "RETROSPECTIVE: data pre-exists and the outcome was known at authoring time (see "
            "status_note) — this is an analysis plan, not a pre-registration. The validity "
            "defense rests on the router being a DETERMINISTIC, ZERO-PARAMETER rule (raw key-cos "
            "ranking; no fitting, no free knobs, nothing to overfit), NOT on prior registration. "
            "A confirmatory claim requires E1 (new data). D3's ~0.72 rank corr (gate_/g4_ cells) "
            "is a consistency cross-check for the egl_-cell router, not the source of the gate.",
        ],
    }
    os.makedirs(ANALYSIS, exist_ok=True)
    json.dump(out, open(out_path, "w"), indent=2)

    # ---- compact human-readable table --------------------------------------- #
    print(f"\n=== D4 ROUTING E0 (weight={weight_editor}, L{primary_L}, f_mem={f_mem}) ===")
    print(f"{'seed':>4} | {'eta_oracle':>10} {'eta_raw':>8} {'eta_ols':>8} {'eta_rand':>9} "
          f"| {'raw>rand':>8}")
    for s in seeds:
        v = per_seed.get(str(s))
        if v is None:
            continue
        e = v["eta"]
        beats = (e["geometry_raw"] is not None and e["random_expected"] is not None
                 and e["geometry_raw"] > e["random_expected"])
        # None-safe formatting for EVERY eta column (any policy can be None when the oracle
        # removes no damage in a cell) — not just geometry_ols.
        def _f(x, w):
            return f"{x:>{w}.3f}" if x is not None else f"{'NA':>{w}}"
        print(f"{s:>4} | {_f(e['oracle'], 10)} {_f(e['geometry_raw'], 8)} "
              f"{_f(e['geometry_ols'], 8)} {_f(e['random_expected'], 9)} | {str(beats):>8}")
    print(f"mean_eta_raw={r4(mean_eta_raw)}  mean_eta_random={r4(mean_eta_rand)}  "
          f"margin={r4(margin)}  perseed_pass={perseed_pass}/3  beats_random={beats_random_count}/3")
    print(f"VERDICT: {verdict}")
    print(f"[routing] wrote {out_path}")
    return out


# --------------------------------------------------------------------------- #
# Self-test: prove the reduction/oracle/regret accounting on synthetic data.   #
# --------------------------------------------------------------------------- #
def selftest():
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
        ok = ok and cond

    # Fixture 1: planted PERFECT predictor, all-positive distinct damage.
    #   w = [1,2,3,4,5,6,7,8,9,10], grace=0, f=0.30 -> k=3.
    #   oracle routes {10,9,8}=27; random E = 0.3*55 = 16.5; always_weight reduction=0.
    #   perfect predictor p=w -> geometry routes {10,9,8}=27 -> eta_geom=1.0.
    w = np.arange(1.0, 11.0)
    g = np.zeros_like(w)
    delta = w - g
    k = budget_k(w.size, 0.30)
    check("budget_k(10,0.30)==3", k == 3, f"k={k}")
    red_oracle = policy_reduction(delta, oracle_order(delta), k)
    check("oracle reduction == 27", abs(red_oracle - 27.0) < 1e-9, f"{red_oracle}")
    red_perfect = policy_reduction(delta, signal_order(w), k)
    check("perfect-predictor eta == 1.0", abs(eta(red_perfect, red_oracle) - 1.0) < 1e-12)
    re_rand = random_expected_reduction(delta, k)
    check("random E[reduction] == 16.5", abs(re_rand - 16.5) < 1e-9, f"{re_rand}")
    check("ordering oracle > random > always_weight",
          red_oracle > re_rand > 0.0, f"{red_oracle} > {re_rand} > 0")
    check("eta_random == 16.5/27", abs(eta(re_rand, red_oracle) - 16.5 / 27.0) < 1e-12)

    # Fixture 2: ANTI-correlated predictor p=-w -> routes {1,2,3}=6 (the bottom-k).
    #   eta_geom = 6/27 < eta_random(16.5/27); recovered exactly.
    red_anti = policy_reduction(delta, signal_order(-w), k)
    check("anti-corr reduction == 6", abs(red_anti - 6.0) < 1e-9, f"{red_anti}")
    check("eta_anti == 6/27", abs(eta(red_anti, red_oracle) - 6.0 / 27.0) < 1e-12)
    check("eta_anti < eta_random", eta(red_anti, red_oracle) < eta(re_rand, red_oracle))

    # Fixture 3: NONZERO grace damage subtracts exactly. grace=0.5 each -> delta=w-0.5.
    #   oracle {9.5,8.5,7.5}=25.5; accounting must subtract grace.
    g3 = np.full_like(w, 0.5)
    d3delta = w - g3
    red_o3 = policy_reduction(d3delta, oracle_order(d3delta), k)
    check("nonzero-grace oracle reduction == 25.5", abs(red_o3 - 25.5) < 1e-9, f"{red_o3}")

    # Fixture 4: scalarized-cost threshold accounting.
    #   w=[0,1,2,3,4], grace=0, c_mem=1.5 -> route iff w>1.5 -> {2,3,4} to memory.
    #   cost = kept-weight(0+1) + grace(0) + 1.5*3 = 1 + 4.5 = 5.5.
    w4 = np.arange(0.0, 5.0)
    g4 = np.zeros_like(w4)
    mask = (w4 - g4) > 1.5
    c = scalarized_cost(w4, g4, mask, 1.5)
    check("scalarized cost == 5.5", abs(c - 5.5) < 1e-9, f"{c}")
    check("scalarized routes 3 edits", int(mask.sum()) == 3, f"{int(mask.sum())}")
    # always-in-weights cost == sum(w); always-in-memory cost == c_mem*n + grace.
    check("always-weight cost == 10", abs(scalarized_cost(w4, g4, np.zeros_like(w4, bool), 1.5) - 10.0) < 1e-9)
    check("always-memory cost == 7.5", abs(scalarized_cost(w4, g4, np.ones_like(w4, bool), 1.5) - 7.5) < 1e-9)

    # Fixture 5: eta undefined when oracle removes nothing (all delta <= 0).
    check("eta None when oracle reduction ~0", eta(0.0, 0.0) is None)

    print("\nALL CHECKS PASSED" if ok else "\nSELF-TEST FAILED")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true",
                    help="run synthetic-fixture accounting checks and exit")
    ap.add_argument("--layers", default="12",
                    help="comma list; the FIRST is the primary-gate layer (L12). Others are "
                         "DESCRIPTIVE-only secondaries (e.g. '12,14').")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--weight_editor", choices=["rome", "alpha"], default="rome",
                    help="the weight arm; rome is the damage-prone primary, alpha the secondary")
    ap.add_argument("--f_mem", type=float, default=F_MEM_PRIMARY)
    ap.add_argument("--out", default=os.path.join(ANALYSIS, "D4_routing_e0.json"))
    args = ap.parse_args()

    if args.selftest:
        import sys
        sys.exit(0 if selftest() else 1)

    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    run_analysis(layers, seeds, args.weight_editor, args.f_mem, args.out)


if __name__ == "__main__":
    main()
