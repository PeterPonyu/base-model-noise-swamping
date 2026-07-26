"""merging_editors.py — editor-general RG federation (ROME / MEMIT / AlphaEdit).

Extends experiments/merging_m0.py's ROME-only edit-federation RG operating curve to
the two other native editors in this harness (editors/memit.py, editors/alphaedit.py)
so we can test whether the two-regime federation law is EDITOR-GENERAL. merging_m0.py
is NOT modified — this module reuses its analysis primitives (Spearman / partial
Spearman / thresholds / pass-rule / atomic I-O / tiled grouping) verbatim and only
adds the multi-layer, effective-key generalisation the non-ROME editors require.

------------------------------------------------------------------ the generalisation
Every native editor here applies, per edited layer l, a RANK-ONE weight delta

        ΔW^l = outer(r^l, kk^l) / denom^l                                    (†)

where kk^l is the "effective column key" that the delta's column space is spanned by,
and denom^l the editor's own scalar divisor. The three editors differ ONLY in
(kk^l, denom^l, and how many layers l are touched):

  * ROME  (editors/rome_native.py:236)  — ONE layer L:
        kk = k                (the raw down_proj-input key at the subject token)
        denom = k·k (+1e-8)
        r = v − W k
    -> (†) is exactly merging_m0.rank_one_delta; this file's ROME path REPRODUCES
       merging_m0 (the equivalence anchor, asserted in --selftest).

  * AlphaEdit (editors/alphaedit.py:144) — ONE layer L, null-space projected:
        kk = P k              (P = I − U_r U_rᵀ projects the key off the preserved-key
                               subspace)
        denom = k·(P k) (+1e-8)
        r = v − W k
    The delta's column key is the PROJECTED key P k, but the vector the OTHER edits'
    keys arrive on is still the raw k. So the cross-term is asymmetric (see below).

  * MEMIT (editors/memit.py:232) — a SPAN of layers ending at the z-layer:
        kk^l = C_l^{-1} k_l   (whitened key; = k_l when cov is identity)
        denom^l = k_l·(C_l^{-1} k_l) (+1e-8)
        r^l = shortfall_l / (n−i)   (keys RECAPTURED under the partially-edited model,
                                     MEMIT's own recompute-per-layer convention)

------------------------------------------------------------------ closed-form cross-talk
The perturbation edit b's layer-l delta imposes on edit a's layer-l key is EXACT and
needs no dense ΔW (generalises merging_m0.cross_term, which is the kk=k special case):

        ΔW_b^l @ k_a^l = r_b^l (kk_b^l · k_a^l) / denom_b^l                   (‡)

RECEIVED CROSS-TALK aggregated across the edited layers (dim = d_out = hidden):

        d_a = Σ_l Σ_{b≠a in group}  ΔW_b^l @ k_a^l                            (§)

This is the first-order shift the merge imposes on edit a's z-layer output residual —
the same first-order approximation merging_m0 makes (the measured `drop` is the
ground-truth nonlinear effect; d_a / I_cos are its CPU-computable predictors).

GENERALISED STRENGTH and the screening statistics (reduce to merging_m0's for ROME):

        S_b^l = ||r_b^l|| · ||kk_b^l|| / denom_b^l
        ||ΔW_b^l @ k_a^l|| = S_b^l · ||k_a^l|| · |cos(kk_b^l, k_a^l)|          (exact)

        I_cos(a) = Σ_l ||k_a^l|| Σ_b S_b^l |cos(kk_b^l, k_a^l)|
        I_mag(a) = Σ_l ||k_a^l|| Σ_b S_b^l                                    (cosine=1 bound)

For ROME (one layer, kk=k, denom=k·k) S_b = ||r_b||/||k_b|| and I_cos/I_mag collapse to
merging_m0's definitions exactly — this is why the two are directly comparable.

DOSE (the merging-law "received dose", corresponding-layer pairing choice documented
here so the reviewer can see it): the signed projection of the received cross-talk onto
edit a's OWN aggregate intended-output direction

        R_a^agg = Σ_l r_a^l                 (sum of a's per-layer residual vectors)
        dose(a) = (d_a · R_a^agg) / ||R_a^agg||^2

For ROME this is single-layer (R_a^agg = r_a), matching merging_m0's single-layer
geometry so the numbers stay comparable; for MEMIT it sums a's residual contributions
across its edited layers (the natural multi-layer analogue — d_a is already the
across-layer sum, so pairing it with the across-layer own-residual keeps both sides on
the same z-layer residual axis). `gain` (drop vs dose) is left to downstream analysis;
this module stores drop, dose, I_cos, I_mag, ||d_a|| and R_a^agg so gain is computable
without re-running the model.

------------------------------------------------------------------ what is reused vs new
Reused (imported) from experiments/merging_m0.py — one audited implementation:
  _spearman, partial_spearman_multi, _default_thresholds, PRE_REG_PASS_RULE,
  _write_table, _savez_atomic, _tiled_groups, _model_tag, DEF_* thresholds, and the
  GPU helper _load_edit_model. The measurement bookkeeping (tiled groups × seeds ×
  group-sizes, obs_/mem_ columns) and the RG pass-rule/verdict shape are identical.
New here: the multi-layer per-edit schema (K/KK/R are [N, L, d]), the effective-key
cross-term (‡)/(§), _regime_stat_ml, analyze_rg_ml, per-editor solo capture, and the
multi-layer federation measurement.

------------------------------------------------------------------ RG bundle schema (editors.v1)
Per-seed vectors npz: layers[L], K/KK[N,L,d_in], R[N,L,d_out], denom/S/key_norm/resid_norm[N,L],
target_tok/logit_solo/argmax_ok_solo/recon_rel_err[N]. Measurements npz mirrors merging_m0's RG
columns (obs_/mem_ seed/g/group/edit, obs_logit_post, obs_argmax_ok_post) PLUS obs_drop and
obs_dose (per-observation drop = logit_solo−merged_logit and the received dose), so absolute GAIN
(drop vs dose) is computable downstream with NO model. schema_version bumped to editors.v1, so
merging_m0's ROME-only tools error loudly rather than silently mis-reading the multi-layer arrays.
Bundle dir + table names carry the dataset tag: <model>_<editor>_<dataset>_L<layer>_RG so cf and
zsre cells at the same (model,editor,layer) coexist.

------------------------------------------------------------------ CPU-only validation
--selftest (no GPU) asserts:
  (a) generalised cross-term (‡)/(§) == brute-force dense-ΔW @ k_a to fp64, with
      effective keys kk≠k AND multiple layers (the non-trivial regime);
  (b) federation additivity: (Σ_b ΔW_b) @ k_a == Σ_b (ΔW_b @ k_a);
  (c) ROME-EQUIVALENCE ANCHOR: on a single-layer (kk=k) synthetic fixture,
      _regime_stat_ml reproduces merging_m0._regime_stat's I_cos/I_mag/||d_a||/rho/
      partial fields to fp64 — this file's ROME path == merging_m0;
  (d) end-to-end RG pass/kill on synthetic multi-layer bundles through analyze_rg_ml;
  (e) ΔW-FIDELITY: on a tiny random-weight Llama, one edit per editor through the REAL
      editors/{rome_native,alphaedit,memit}.apply_edit, asserting the re-derived ΔW (stored
      factors) matches to Frobenius rel-err < 1e-4 per layer — replaces the near-tautological
      recon_rel_err as the fidelity check (best-effort; needs a local tokenizer, else SKIP).
--smoke closes the same ΔW-fidelity loop on the REAL model (wired into the driver as a first-run
gate) — the hard launch gate that always runs on the real weights.

Standing workspace rules honored: value-opt stays fp32 (the editors keep their own
.float() casts; this file never overrides model dtype); the reported statistic is
signed Spearman, never AUROC; PID-only process control (driver side).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

HARNESS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HARNESS)

# reuse merging_m0's audited numpy primitives + thresholds + pass rule (no reimpl)
from experiments.merging_m0 import (  # noqa: E402
    _spearman, partial_spearman_multi, _default_thresholds, PRE_REG_PASS_RULE,
    _write_table, _savez_atomic, _tiled_groups, _model_tag,
    DEF_PARTIAL_MIN, DEF_RHO_MIN, DEF_NEG_LOGIT, DEF_NEG_ARGMAX, DEF_SAT_ARGMAX,
    DEF_DRHO_MIN,
)

SCHEMA_VERSION = "editors.v1"
EDITORS = ("rome", "memit", "alpha")


# ============================================================ effective-key cross-talk
def cross_term_eff(r_b, kk_b, k_a, denom_b):
    """Exact ΔW_b @ k_a for the rank-one delta ΔW_b = outer(r_b, kk_b)/denom_b (eq ‡),
    without materialising ΔW_b. kk_b is the effective COLUMN key (k for ROME, P k for
    AlphaEdit, C^{-1}k for MEMIT); k_a is edit a's RAW receiver key. Reduces to
    merging_m0.cross_term when kk_b == k_b."""
    r_b = np.asarray(r_b, float)
    return r_b * (float(np.dot(kk_b, k_a)) / float(denom_b))


def dense_delta_eff(r_b, kk_b, denom_b):
    """The dense ΔW_b = outer(r_b, kk_b)/denom_b, [d_out, d_in]. Brute-force reference for
    the self-test only; production never materialises this."""
    return np.outer(np.asarray(r_b, float), np.asarray(kk_b, float)) / float(denom_b)


def _obs_quantities(K, KK, R, denom, S, key_norm, a, others):
    """The per-observation screening quantities for edit `a` merged with `others`, aggregated
    over edited layers (eqs §): (I_cos, I_mag, d_a, ||d_a||, dose). One implementation shared by
    _regime_stat_ml (analysis) and _obs_dose_drop (bundle storage) so the two can never drift."""
    d_out = R.shape[2]
    L = K.shape[1]
    ic = im = 0.0
    d_a = np.zeros(d_out)
    if others:
        for li in range(L):
            ka = K[a, li]
            kn = float(key_norm[a, li])
            for b in others:
                kkb = KK[b, li]
                denomb = float(denom[b, li])
                # |cos(kk_b, k_a)| — effective column key vs the raw receiver key
                cab = abs(float(np.dot(kkb, ka)) /
                          (float(np.linalg.norm(kkb)) * kn + 1e-12))
                ic += kn * float(S[b, li]) * cab
                im += kn * float(S[b, li])
                d_a += cross_term_eff(R[b, li], kkb, ka, denomb)
    nd = float(np.linalg.norm(d_a))
    Ra = R[a].sum(axis=0)                       # [d_out] = Σ_l r_a^l (aggregate own residual)
    ra2 = float(np.dot(Ra, Ra))
    dose = float(np.dot(d_a, Ra) / ra2) if ra2 > 0 else 0.0
    return ic, im, d_a, nd, dose


# ============================================================ multi-layer per-regime stats
def _regime_stat_ml(vecs, obs_edit, obs_group, obs_logit_post, obs_argmax_ok_post, members, t):
    """Per-regime stats over the merge observations, multi-layer + effective-key.

    vecs = (K, KK, R, denom, S, key_norm, logit_solo, argmax_ok_solo) with
      K, KK: [N, L, d_in]   R: [N, L, d_out]   denom, S, key_norm: [N, L]
      logit_solo, argmax_ok_solo: [N].
    members: {group_id -> [edit indices]}.

    Returns the SAME dict schema as merging_m0._regime_stat (so analyze_rg_ml / the RG
    verdict logic are shared), PLUS dose fields. Screening quantities per observation
    (edit a in group g) aggregate over edited layers l per eqs (§) I_cos/I_mag/dose."""
    K, KK, R, denom, S, key_norm, logit_solo, argmax_ok_solo = vecs
    L = K.shape[1]

    I_cos, I_mag, normdelta, drop, dose = [], [], [], [], []
    kn_a, S_a, argmax_loss, worked = [], [], [], []
    for a, g, lp, ap in zip(obs_edit, obs_group, obs_logit_post, obs_argmax_ok_post):
        a, g = int(a), int(g)
        others = [b for b in members[g] if b != a]
        ic, im, _d_a, nd, ds = _obs_quantities(K, KK, R, denom, S, key_norm, a, others)
        I_cos.append(ic); I_mag.append(im); normdelta.append(nd); dose.append(ds)
        drop.append(float(logit_solo[a] - lp))
        kn_a.append(float(np.linalg.norm(K[a].reshape(-1))))   # aggregate own key scale
        S_a.append(float(S[a].sum()))
        ws = argmax_ok_solo[a] > 0.5
        worked.append(bool(ws)); argmax_loss.append(bool(ws and ap < 0.5))

    I_cos = np.array(I_cos); I_mag = np.array(I_mag); normdelta = np.array(normdelta)
    drop = np.array(drop); dose = np.array(dose)
    kn_a = np.array(kn_a); S_a = np.array(S_a)
    argmax_loss = np.array(argmax_loss, bool); worked = np.array(worked, bool)

    n_worked = int(worked.sum())
    # |drop| median (sign cannot hide real perturbation — matches merging_m0's convention)
    med_abs_drop = float(np.median(np.abs(drop))) if drop.size else float("nan")
    argmax_loss_rate = (float(argmax_loss.sum() / n_worked) if n_worked > 0 else float("nan"))
    rho_cos = _spearman(I_cos, drop)
    rho_mag = _spearman(I_mag, drop)
    drho = (float(rho_cos - rho_mag) if np.isfinite(rho_cos) and np.isfinite(rho_mag)
            else float("nan"))
    partial_geom = partial_spearman_multi(I_cos, drop, [I_mag])
    partial_ownmag = partial_spearman_multi(I_cos, drop, [kn_a, S_a])
    xcheck = _spearman(I_cos, normdelta)
    rho_dose = _spearman(dose, drop)

    non_negligible = bool((np.isfinite(med_abs_drop) and med_abs_drop >= t["neg_logit"]) or
                          (np.isfinite(argmax_loss_rate) and argmax_loss_rate >= t["neg_argmax"]))
    saturated = bool(np.isfinite(argmax_loss_rate) and argmax_loss_rate > t["sat_argmax"])
    c2_coherent = bool(np.isfinite(rho_cos) and rho_cos >= t["rho_min"])
    c3_eligible = bool(non_negligible and (not saturated) and c2_coherent)
    collapses_ownmag = not bool(np.isfinite(partial_ownmag) and partial_ownmag >= t["partial_min"])

    def rnd(x):
        return round(float(x), 4) if (x is not None and np.isfinite(x)) else None
    return {
        "n_obs": int(len(drop)),
        "n_groups": int(len(members)),
        "n_worked_solo": n_worked,
        "n_layers": int(L),
        "median_abs_drop_logit": rnd(med_abs_drop),
        "mean_drop_logit": (round(float(np.mean(drop)), 5) if drop.size else None),
        "argmax_loss_rate": rnd(argmax_loss_rate),
        "rho_I_cos_drop": rnd(rho_cos),
        "rho_I_magonly_drop": rnd(rho_mag),
        "partial_rho_geom": rnd(partial_geom),
        "partial_rho_geom_ownmag": rnd(partial_ownmag),
        "delta_rho_cos_minus_mag": rnd(drho),
        "rho_I_cos_normdelta_cpu_xcheck": rnd(xcheck),
        "rho_dose_drop": rnd(rho_dose),
        "mean_abs_dose": (round(float(np.mean(np.abs(dose))), 5) if dose.size else None),
        "non_negligible": non_negligible,
        "saturated": saturated,
        "c2_coherent": c2_coherent,
        "c3_eligible": c3_eligible,
        "collapses_under_ownmag_partial": collapses_ownmag,
    }


# ============================================================ RG operating-curve analysis (CPU)
def analyze_rg_ml(per_seed_vectors, measurements, meta, thr=None):
    """CPU operating-curve analysis for the editor-general bundle. Same pre-registered pass
    rule + verdict shape as merging_m0.analyze_rg, computed via _regime_stat_ml. Standalone-
    rerunnable from a saved bundle (no model, no GPU)."""
    t = _default_thresholds()
    if thr:
        t.update(thr)
    seeds = [int(s) for s in meta["seeds"]]
    gsizes = [int(g) for g in meta["group_sizes"]]
    obs_seed = measurements["obs_seed"].astype(int); obs_g = measurements["obs_g"].astype(int)
    obs_group = measurements["obs_group"].astype(int); obs_edit = measurements["obs_edit"].astype(int)
    obs_lp = measurements["obs_logit_post"].astype(float)
    obs_ap = measurements["obs_argmax_ok_post"].astype(float)
    mem_seed = measurements["mem_seed"].astype(int); mem_g = measurements["mem_g"].astype(int)
    mem_group = measurements["mem_group"].astype(int); mem_edit = measurements["mem_edit"].astype(int)

    members_all = defaultdict(list)
    for s, g, gr, ed in zip(mem_seed, mem_g, mem_group, mem_edit):
        members_all[(int(s), int(g), int(gr))].append(int(ed))

    cells = {}
    for s in seeds:
        v = per_seed_vectors[s]
        vecs = (v["K"].astype(float), v["KK"].astype(float), v["R"].astype(float),
                v["denom"].astype(float), v["S"].astype(float), v["key_norm"].astype(float),
                v["logit_solo"].astype(float), v["argmax_ok_solo"].astype(float))
        for g in gsizes:
            sel = np.where((obs_seed == s) & (obs_g == g))[0]
            if sel.size == 0:
                continue
            members = {int(gr): members_all[(s, g, int(gr))] for gr in set(obs_group[sel].tolist())}
            cells[f"g{g}_s{s}"] = _regime_stat_ml(
                vecs, obs_edit[sel], obs_group[sel], obs_lp[sel], obs_ap[sel], members, t)

    def _cell(g, s):
        return cells.get(f"g{g}_s{s}")

    def _geom_pass(st):
        p = st["partial_rho_geom"]
        return bool(st["c3_eligible"] and p is not None and p >= t["partial_min"]
                    and not st["collapses_under_ownmag_partial"])

    per_g = {}
    for g in gsizes:
        sc = {s: _cell(g, s) for s in seeds if _cell(g, s) is not None}
        pass_seeds = [s for s, st in sc.items() if _geom_pass(st)]
        per_g[str(g)] = {
            "n_seeds_measured": len(sc),
            "n_seeds_eligible": sum(1 for st in sc.values() if st["c3_eligible"]),
            "n_seeds_pass_geometry": len(pass_seeds),
            "pass_seeds": pass_seeds,
            "partial_by_seed": {str(s): st["partial_rho_geom"] for s, st in sc.items()},
            "eligible_by_seed": {str(s): st["c3_eligible"] for s, st in sc.items()},
            "argmax_loss_by_seed": {str(s): st["argmax_loss_rate"] for s, st in sc.items()},
            "rho_dose_by_seed": {str(s): st["rho_dose_drop"] for s, st in sc.items()},
            "qualifies": len(pass_seeds) >= 2,
        }
    qualifying = [g for g in gsizes if per_g[str(g)]["qualifies"]]

    elig = [(g, s) for g in gsizes for s in seeds
            if _cell(g, s) is not None and _cell(g, s)["c3_eligible"]]
    any_geom_pass = any(_geom_pass(_cell(g, s)) for (g, s) in elig)
    all_collapse = bool(elig) and all(_cell(g, s)["collapses_under_ownmag_partial"] for (g, s) in elig)

    if not elig:
        overall = "INCONCLUSIVE"
    elif len(qualifying) >= 2:
        overall = "PASS"
    elif (not any_geom_pass) or all_collapse:
        overall = "KILL"
    else:
        overall = "MIXED"

    boundary = {}
    for s in seeds:
        nonsat = [g for g in gsizes if _cell(g, s) is not None and not _cell(g, s)["saturated"]]
        boundary[str(s)] = (max(nonsat) if nonsat else None)
    bvals = [b for b in boundary.values() if b is not None]

    table = {
        "experiment": "merging_law_RG_operating_curve_editors",
        "schema_version": SCHEMA_VERSION,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": meta.get("model"), "layer": meta.get("layer"), "editor": meta.get("editor"),
        "edited_layers": meta.get("edited_layers"),
        "dataset": meta.get("dataset", "counterfact"),
        "seeds": seeds, "group_sizes": gsizes, "n_edits": meta.get("n_edits"),
        "thresholds": t,
        "pass_rule": PRE_REG_PASS_RULE,
        "geometry_metric": "partial_rho(I_cos, drop | I_mag)  [own-mag guard: rho(...|||k_a||,S_a)]",
        "generalisation_note": ("editor-general: per-layer rank-one ΔW^l=outer(r^l,kk^l)/denom^l; "
                                "kk=k(ROME)/Pk(AlphaEdit)/C^-1 k(MEMIT); cross-term "
                                "ΔW_b^l@k_a^l=r_b^l(kk_b^l·k_a^l)/denom_b^l aggregated over layers; "
                                "ROME path == merging_m0 (selftest anchor)"),
        "cells": cells,
        "per_g_summary": per_g,
        "scoped_federation_boundary": {
            "per_seed_largest_nonsaturated_g": boundary,
            "conservative_min_across_seeds": (min(bvals) if bvals else None),
            "note": ("largest group size g where damage stays partial (argmax_loss < sat_argmax) "
                     "i.e. measurable/gradated — the scoped-federation limit"),
        },
        "verdict": {
            "overall": overall,
            "qualifying_group_sizes": qualifying,
            "n_qualifying_group_sizes": len(qualifying),
            "n_testable_cells": len(elig),
            "any_geometry_pass": any_geom_pass,
            "all_collapse_under_ownmag": all_collapse,
            "pass_rule": PRE_REG_PASS_RULE,
            "interpretation": {
                "PASS": ("geometry (partial rho) predicts merge interference at >= 2 group sizes "
                         "across >= 2 seeds for this editor -> the federation law is editor-general "
                         "at this cell"),
                "KILL": ("geometry partial < partial_min everywhere testable (or collapses under "
                         "own-magnitude partialling) -> the law does not carry to this editor"),
                "MIXED": ("geometry passes somewhere but not at >= 2 group sizes x >= 2 seeds -> "
                          "under-powered / scope narrower"),
                "INCONCLUSIVE": ("no non-negligible, non-saturated, c2-coherent cell -> the curve "
                                 "cannot test geometry (all saturated or all negligible)"),
            }[overall],
        },
    }
    return table


def print_rg_table(table):
    v = table["verdict"]
    print("\n=== MERGING LAW RG OPERATING CURVE (editor-general) ===", flush=True)
    print(f"model={table['model']} editor={table['editor']} layer={table['layer']} "
          f"edited_layers={table.get('edited_layers')} seeds={table['seeds']} "
          f"group_sizes={table['group_sizes']}", flush=True)
    for g in table["group_sizes"]:
        s = table["per_g_summary"][str(g)]
        print(f"  g={g:>3}: partial_by_seed={s['partial_by_seed']} "
              f"eligible={s['n_seeds_eligible']}/{s['n_seeds_measured']} "
              f"pass_geometry={s['n_seeds_pass_geometry']} rho_dose={s['rho_dose_by_seed']} "
              f"qualifies={s['qualifies']}", flush=True)
    b = table["scoped_federation_boundary"]
    print(f"  scoped-federation boundary (largest non-saturated g): "
          f"per_seed={b['per_seed_largest_nonsaturated_g']} "
          f"conservative={b['conservative_min_across_seeds']}", flush=True)
    print(f"  VERDICT overall={v['overall']} qualifying_g={v['qualifying_group_sizes']} "
          f"testable_cells={v['n_testable_cells']}", flush=True)
    print(f"  -> {v['interpretation']}", flush=True)


# ============================================================ per-obs dose/drop (bundle storage)
def _obs_dose_drop(per_seed_vectors, measurements):
    """Per-observation (drop, dose) aligned to the measurements obs_* order, so absolute GAIN
    (drop vs dose) is computable downstream WITHOUT re-running the model. Same cross-talk math
    as _regime_stat_ml (both call _obs_quantities). Stored into the RG measurements npz as
    obs_drop / obs_dose (schema editors.v1). drop = logit_solo[a] − merged_logit[a]."""
    obs_seed = measurements["obs_seed"].astype(int); obs_g = measurements["obs_g"].astype(int)
    obs_group = measurements["obs_group"].astype(int); obs_edit = measurements["obs_edit"].astype(int)
    obs_lp = measurements["obs_logit_post"].astype(float)
    mem_seed = measurements["mem_seed"].astype(int); mem_g = measurements["mem_g"].astype(int)
    mem_group = measurements["mem_group"].astype(int); mem_edit = measurements["mem_edit"].astype(int)
    members_all = defaultdict(list)
    for s, g, gr, ed in zip(mem_seed, mem_g, mem_group, mem_edit):
        members_all[(int(s), int(g), int(gr))].append(int(ed))
    drop = np.zeros(len(obs_edit), np.float64)
    dose = np.zeros(len(obs_edit), np.float64)
    for i in range(len(obs_edit)):
        s, g, gr, a = int(obs_seed[i]), int(obs_g[i]), int(obs_group[i]), int(obs_edit[i])
        v = per_seed_vectors[s]
        others = [b for b in members_all[(s, g, gr)] if b != a]
        _ic, _im, _d_a, _nd, ds = _obs_quantities(
            v["K"].astype(float), v["KK"].astype(float), v["R"].astype(float),
            v["denom"].astype(float), v["S"].astype(float), v["key_norm"].astype(float), a, others)
        drop[i] = float(v["logit_solo"][a]) - float(obs_lp[i])
        dose[i] = ds
    return drop.astype(np.float32), dose.astype(np.float32)


# ============================================================ schema I/O
def save_rg(rg_dir, per_seed_vectors, measurements, meta):
    os.makedirs(rg_dir, exist_ok=True)
    for s, v in per_seed_vectors.items():
        _savez_atomic(os.path.join(rg_dir, f"rg_seed{int(s)}_vectors.npz"), v)
    _savez_atomic(os.path.join(rg_dir, "rg_measurements.npz"), measurements)
    tmp = os.path.join(rg_dir, "rg_meta.json.tmp")
    with open(tmp, "w") as f:
        json.dump(meta, f, indent=2)
    os.replace(tmp, os.path.join(rg_dir, "rg_meta.json"))


def load_rg(rg_dir):
    with open(os.path.join(rg_dir, "rg_meta.json")) as f:
        meta = json.load(f)
    seeds = [int(s) for s in meta["seeds"]]
    per_seed = {s: dict(np.load(os.path.join(rg_dir, f"rg_seed{s}_vectors.npz"))) for s in seeds}
    meas = dict(np.load(os.path.join(rg_dir, "rg_measurements.npz")))
    return per_seed, meas, meta


def _bundle_complete(rg_dir):
    """True iff rg_dir holds a FULLY-written, valid RG bundle: meta + measurements + a table with a
    recognised verdict. A partial (crashed/timed-out) bundle — e.g. rg_meta.json written by save_rg
    BEFORE the table — reads False so a re-run is allowed (MAJOR-1 fix: the refuse guard must never
    wedge a cell on a half-written bundle; the driver's mv-to-.INVALID + re-run path depends on it)."""
    try:
        if not (os.path.isfile(os.path.join(rg_dir, "rg_meta.json")) and
                os.path.isfile(os.path.join(rg_dir, "rg_measurements.npz")) and
                os.path.isfile(os.path.join(rg_dir, "RG_editors_table.json"))):
            return False
        with open(os.path.join(rg_dir, "RG_editors_table.json")) as f:
            d = json.load(f)
        return d.get("verdict", {}).get("overall") in ("PASS", "KILL", "MIXED", "INCONCLUSIVE")
    except Exception:
        return False


# ============================================================ dataset loaders (reuse harness parse)
def load_edits(dataset, path, n_edits, seed=0, n_holdout=0):
    """Return (edits, holdout) as lists of {subject,prompt,target_new,target_true}.

    dataset='cf': byte-identical edit ORDER to merging_m0.load_counterfact (same
    json.load -> default_rng(seed).shuffle -> requested_rewrite parse -> first-n), then a
    disjoint holdout slice for the AlphaEdit preserved-key / MEMIT covariance banks.
    dataset='zsre': the harness zsRE editing schema (src/subject/alt/pred), reusing
    killgate_keygeom.load_zsre's field mapping."""
    data = json.load(open(path))
    rng = np.random.default_rng(seed)
    rng.shuffle(data)
    recs = []
    if dataset == "cf":
        for d in data:
            rr = d.get("requested_rewrite", d)
            try:
                subj = rr["subject"]
                prompt = rr["prompt"].format(subj) if "{}" in rr["prompt"] else rr["prompt"]
                tnew = rr["target_new"]["str"] if isinstance(rr["target_new"], dict) else rr["target_new"]
                ttrue = rr["target_true"]["str"] if isinstance(rr["target_true"], dict) else rr["target_true"]
            except Exception:
                continue
            recs.append({"subject": subj, "prompt": prompt, "target_new": tnew, "target_true": ttrue})
            if len(recs) >= n_edits + n_holdout:
                break
    elif dataset == "zsre":
        for d in data:
            s, p, alt, pred = d.get("subject"), d.get("src"), d.get("alt"), d.get("pred")
            if not (s and p and alt and pred):
                continue
            recs.append({"subject": s, "prompt": p, "target_new": alt, "target_true": pred})
            if len(recs) >= n_edits + n_holdout:
                break
    else:
        raise ValueError(f"unknown dataset {dataset!r} (expected 'cf' or 'zsre')")
    return recs[:n_edits], recs[n_edits:n_edits + n_holdout]


# ============================================================ phase-1 GPU: per-editor solo capture
def _empty_vectors(N, L, d_in, d_out):
    return dict(
        layers=np.zeros(L, np.int64),
        K=np.zeros((N, L, d_in), np.float32), KK=np.zeros((N, L, d_in), np.float32),
        R=np.zeros((N, L, d_out), np.float32),
        denom=np.zeros((N, L), np.float64), key_norm=np.zeros((N, L), np.float32),
        S=np.zeros((N, L), np.float32), resid_norm=np.zeros((N, L), np.float32),
        target_tok=np.zeros(N, np.int64), logit_solo=np.full(N, np.nan, np.float32),
        argmax_ok_solo=np.zeros(N, np.float32), recon_rel_err=np.zeros(N, np.float32),
    )


def _finalize_S(vec):
    """Generalised strength S^l = ||r^l|| ||kk^l|| / denom^l and key_norm^l = ||k^l||."""
    K, KK, R, denom = vec["K"], vec["KK"], vec["R"], vec["denom"]
    vec["key_norm"] = np.linalg.norm(K, axis=2).astype(np.float32)
    kk_norm = np.linalg.norm(KK, axis=2)
    vec["resid_norm"] = np.linalg.norm(R, axis=2).astype(np.float32)
    vec["S"] = (vec["resid_norm"] * kk_norm / (denom + 1e-30)).astype(np.float32)
    return vec


# ============================================================ MEMIT true-covariance (R-D, 2026-07-16)
# --memit_cov wiki: estimate C_l on an EXTERNAL text corpus (wikitext) rather than this cell's own
# holdout/edit prompts ('generic'). editors/memit.py's estimate_layer_covariances is reused
# UNCHANGED — only the prompt SOURCE and an on-disk cache are new here, so the ΔW-fidelity gate
# (_fidelity_check_editor, below) is comparing against the SAME real editors/memit.py apply_edit
# install as the existing 'generic' arm: an honest anchor, not a decomposition-level fallback.
COV_CACHE_DIR = os.path.join(HARNESS, "results", "merging_editors", "cov_cache")


def _cov_cache_path(model_path, z_layer, source):
    """results/merging_editors/cov_cache/<model_tag>_L<z_layer>_<source>.npz. `source` in the
    filename (not just <model>_L<layer>.npz as the literal spec phrasing) is load-bearing: MEMIT
    covariance differs by PROMPT SOURCE at a fixed (model, layer) — 'identity' (no cache; ROME-
    style), 'generic' (this cell's own holdout bank) and 'wiki'/'cf_fallback' (an external, cell-
    independent corpus) are three different C_l matrices. Folding them into one filename would
    silently serve a generic-cov cache to a wiki-cov request (or vice versa) on a cache hit."""
    return os.path.join(COV_CACHE_DIR, f"{_model_tag(model_path)}_L{int(z_layer)}_{source}.npz")


def _save_cov_cache(path, cov, layers, source, n_target_tokens):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arrs = {"layers": np.array(sorted(int(l) for l in layers), np.int64),
            "n_target_tokens": np.array(int(n_target_tokens))}
    for l in layers:
        l = int(l)
        arrs[f"chol_{l}"] = cov[l]["chol"].detach().cpu().numpy().astype(np.float32)
        arrs[f"n_tokens_{l}"] = np.array(int(cov[l]["n_tokens"]))
        arrs[f"reg_used_{l}"] = np.array(float(cov[l]["reg_used"]))
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **arrs)
    os.replace(tmp, path)
    print(f"[fed] MEMIT true-cov cache written -> {path} (source={source}, layers={list(layers)})",
          flush=True)


def _load_cov_cache(path, layers):
    """Returns the cached {layer: {chol, n_tokens, reg_used}} dict, or None on any mismatch
    (missing file, corrupt npz, or a DIFFERENT layer span than requested — a stale cache from a
    prior --memit_span/--memit_layers choice must never be silently reused)."""
    import torch
    if not os.path.isfile(path):
        return None
    try:
        d = np.load(path)
        cached_layers = sorted(int(x) for x in d["layers"])
        want_layers = sorted(int(l) for l in layers)
        if cached_layers != want_layers:
            return None
        cov = {}
        for l in want_layers:
            cov[l] = {"chol": torch.from_numpy(np.asarray(d[f"chol_{l}"], np.float32)).float(),
                      "n_tokens": int(d[f"n_tokens_{l}"]), "reg_used": float(d[f"reg_used_{l}"])}
        return cov
    except Exception as ex:
        print(f"[fed] MEMIT true-cov cache at {path} unreadable ({ex}) — rebuilding", flush=True)
        return None


def _wiki_corpus_candidates():
    """Local wikitext corpus locations this harness would recognise, if ever downloaded (ask-first
    standing policy — see CLAUDE.md). None present as of 2026-07-16 (verified: `ls data/` has no
    wiki* entry), so every wave currently falls through to the CF-fallback below; this list exists
    so a future `hf download`/manual drop needs no code change."""
    names = ("wikitext-103-raw-v1", "wikitext-103", "wikitext-2-raw-v1", "wikitext.txt", "wikitext")
    return [os.path.join(HARNESS, "data", n) for n in names]


def _load_wiki_or_fallback_prompts(cf_data_path, n_prompts=4000, seed=999):
    """Prefer a local wikitext corpus under data/ (paragraphs >= 8 words as one "prompt" each, for
    estimate_layer_covariances's per-prompt forward loop); FALL BACK to a broad, CELL-INDEPENDENT
    sample of raw CounterFact prompts (drawn from the full data file with a FIXED seed, distinct
    from any --seed used for edit selection — never just this cell's own --n_edits/--n_holdout
    slice, which would make the "external corpus" claim circular). Returns (prompts, source) where
    source in {'wiki', 'cf_fallback'} — the caller must record this so the paper/prereg text never
    says "MEMIT-style (wiki)" when the fallback actually ran."""
    for path in _wiki_corpus_candidates():
        if not os.path.exists(path):
            continue
        try:
            texts = []
            if os.path.isdir(path):
                for fn in sorted(os.listdir(path))[:8]:
                    fp = os.path.join(path, fn)
                    if os.path.isfile(fp):
                        with open(fp, encoding="utf-8", errors="ignore") as f:
                            texts.append(f.read())
            else:
                with open(path, encoding="utf-8", errors="ignore") as f:
                    texts.append(f.read())
            chunks = [para.strip() for t in texts for para in t.split("\n")
                     if len(para.split()) >= 8]
            if chunks:
                rng = np.random.default_rng(seed)
                rng.shuffle(chunks)
                return chunks[:n_prompts], "wiki"
        except Exception as ex:
            print(f"[fed] wikitext candidate {path} unusable ({ex}) — trying next", flush=True)
    data = json.load(open(cf_data_path))
    rng = np.random.default_rng(seed)   # fixed, --seed-independent: a stable external-ish corpus
    idx = rng.permutation(len(data))[:min(n_prompts, len(data))]
    prompts = []
    for i in idx:
        d = data[int(i)]
        rr = d.get("requested_rewrite", d)
        try:
            subj = rr["subject"]
            prompt = rr["prompt"].format(subj) if "{}" in rr["prompt"] else rr["prompt"]
        except Exception:
            continue
        prompts.append(prompt)
    return prompts, "cf_fallback"


def _get_or_build_memit_cov(args, model, tok, layers, device, holdout, edits):
    """Dispatch --memit_cov in {generic, wiki} to the right prompt source + on-disk cache. Both
    arms call the SAME editors/memit.estimate_layer_covariances (no reimplementation); 'wiki' adds
    only the corpus swap + caching. Ridge/invertibility (eps*trace/d) is handled INSIDE
    estimate_layer_covariances (`A = C_hat + reg_used*mean_diag*I`, mean_diag == trace(C_hat)/d;
    already matches the spec, so nothing new is added here)."""
    from editors.memit import estimate_layer_covariances
    if args.memit_cov == "generic":
        source = "generic"
        cache_path = _cov_cache_path(args.model, max(layers), source)
        cached = _load_cov_cache(cache_path, layers)
        if cached is not None:
            print(f"[fed] MEMIT cov cache hit: {cache_path}", flush=True)
            return cached, source
        cov_prompts = [h["prompt"] for h in holdout] or [e["prompt"] for e in edits]
        cov = estimate_layer_covariances(model, tok, cov_prompts, layers, device,
                                         max_tokens=args.cov_max_tokens)
        _save_cov_cache(cache_path, cov, layers, source, args.cov_max_tokens)
        return cov, source
    # wiki (or its documented CF-fallback)
    prompts, source = _load_wiki_or_fallback_prompts(args.data, n_prompts=4000, seed=999)
    cache_path = _cov_cache_path(args.model, max(layers), source)
    cached = _load_cov_cache(cache_path, layers)
    if cached is not None:
        print(f"[fed] MEMIT true-cov cache hit: {cache_path} (source={source})", flush=True)
        return cached, source
    cov = estimate_layer_covariances(model, tok, prompts, layers, device,
                                     max_tokens=args.cov_max_tokens)
    _save_cov_cache(cache_path, cov, layers, source, args.cov_max_tokens)
    print(f"[fed] MEMIT true-cov built: {len(prompts)} prompts, source={source} "
          f"(--memit_cov=wiki requested; 'wiki' means a real wikitext corpus was found under "
          f"data/, 'cf_fallback' means it was NOT and a cell-independent CounterFact prompt "
          f"sample was used instead — see _wiki_corpus_candidates)", flush=True)
    return cov, source


def _editor_context(editor, model, tok, layer, device, edits, holdout, args, quiet=False):
    """Resolve the editor's edited-layer span, AlphaEdit null-space projector, and MEMIT layer
    covariances ONCE per (model, editor, seed). Returns (layers, projector_or_None, cov_or_None).
    Shared by _compute_solo_editor (the RG solo loop) and _fidelity_check_editor so both install
    the identical operator."""
    import torch
    from editors.rome_native import _capture_key, find_subject_last_token_index  # noqa: E402
    cov = None
    if editor == "memit":
        from editors.memit import parse_memit_layers
        layers = parse_memit_layers(args.memit_layers, layer, model.config.num_hidden_layers,
                                    span=args.memit_span)
        cov_label = "identity"
        if args.memit_cov in ("generic", "wiki"):
            cov, cov_label = _get_or_build_memit_cov(args, model, tok, layers, device, holdout, edits)
        if not quiet:
            print(f"[fed] MEMIT layers={layers} cov={cov_label}", flush=True)
        # MAJOR-1 (revwave review): the RESOLVED source (e.g. cf_fallback when no wikitext
        # exists) must reach the persisted meta, not just stdout — stash it on args so the
        # meta writer can persist it without changing every callsite's tuple shape.
        args._memit_cov_resolved = cov_label
        return layers, None, cov
    projector = None
    if editor == "alpha":
        from editors.alphaedit import build_null_projector
        pk_prompts = holdout or edits
        rows = []
        for h in pk_prompts:
            idx = find_subject_last_token_index(tok, h["prompt"], h.get("subject"))
            rows.append(_capture_key(model, tok, layer, h["prompt"], idx, device).float())
        projector = build_null_projector(torch.stack(rows), keep_ratio=args.keep_ratio)
        if not quiet:
            print(f"[fed] AlphaEdit projector built from {len(rows)} preserved keys "
                  f"(keep_ratio={args.keep_ratio}, "
                  f"rank_removed={int(round(projector.shape[0]-float(torch.trace(projector))))})",
                  flush=True)
    return [layer], projector, cov


def _install_and_capture_one(editor, model, tok, layer, layers, projector, cov, device, edit,
                             steps, lr, v_wd):
    """Install ONE edit with `editor`, APPLYING each layer's ΔW in place (the CALLER restores),
    and return the per-layer factor tuples [(k, kk, r, denom), ...] that define the applied
    rank-one deltas ΔW^l = outer(r,kk)/denom. This is the single audited install used by BOTH the
    solo RG loop and the ΔW-fidelity check, so the stored (K,KK,R,denom) always reconstruct the
    exact weight delta the editor writes."""
    import torch
    from editors.rome_native import _capture_key, find_subject_last_token_index, _optimise_value
    idx = find_subject_last_token_index(tok, edit["prompt"], edit.get("subject"))
    factors = []
    if editor in ("rome", "alpha"):
        k = _capture_key(model, tok, layer, edit["prompt"], idx, device).float()
        v, _v0, _hist = _optimise_value(model, tok, layer, edit["prompt"], idx,
                                        edit["target_new"], device, steps, lr, v_wd)
        v = v.float()
        W = model.model.layers[layer].mlp.down_proj.weight
        r = (v - (W.detach().float() @ k))
        kk = (projector.to(k.device) @ k) if (editor == "alpha" and projector is not None) else k
        denom = float((k @ kk).item()) + 1e-8
        delta = torch.outer(r, kk) / denom
        with torch.no_grad():
            W.add_(delta.to(W.dtype))
        factors.append((k.detach().cpu().numpy(), kk.detach().cpu().numpy(),
                        r.detach().cpu().numpy(), denom))
    else:  # memit — mirror editors/memit.apply_edit's per-layer install loop
        from editors.memit import _solve_cov, _hidden_at
        v, v0, _hist = _optimise_value(model, tok, layer, edit["prompt"], idx,
                                       edit["target_new"], device, steps, lr, v_wd)
        dz = (v.float() - v0.float())
        h_base, _ = _hidden_at(model, tok, edit["prompt"], layer, idx, device)
        z_target = h_base + dz
        n = len(layers)
        for li, l in enumerate(layers):
            h_cur, k_l = _hidden_at(model, tok, edit["prompt"], layer, idx, device, capture_layer=l)
            r_l = (z_target - h_cur) / float(n - li)
            Ck = _solve_cov(cov.get(l) if cov is not None else None, k_l)
            denom = float((k_l @ Ck).item()) + 1e-8
            delta = torch.outer(r_l, Ck) / denom
            with torch.no_grad():
                model.model.layers[l].mlp.down_proj.weight.add_(delta.to(
                    model.model.layers[l].mlp.down_proj.weight.dtype))
            factors.append((k_l.detach().cpu().numpy(), Ck.detach().cpu().numpy(),
                            r_l.detach().cpu().numpy(), denom))
    return factors


def _compute_solo_editor(editor, model, tok, layer, device, edits, holdout, args, t0):
    """Install each edit SOLO with `editor`, capture its per-layer effective-key decomposition
    (K, KK, R, denom), and its solo target-logit; RESTORE all touched weights after each edit so
    every edit sees base weights. Returns (vectors dict, {layer: W}, {layer: W_base}, layers).

    Per-edit install is delegated to _install_and_capture_one (also used by the ΔW-fidelity
    check), so the stored factors are exactly the ones defining the applied ΔW."""
    import torch
    from metrics import next_token_logits, first_target_token_id  # noqa: E402
    N = len(edits)
    steps, lr, v_wd = args.steps, args.lr, 1e-3
    layers, projector, cov = _editor_context(editor, model, tok, layer, device, edits, holdout, args)
    L = len(layers)

    d_out = int(model.model.layers[layers[0]].mlp.down_proj.weight.shape[0])
    d_in = int(model.model.layers[layers[0]].mlp.down_proj.weight.shape[1])
    vec = _empty_vectors(N, L, d_in, d_out)
    vec["layers"] = np.array(layers, np.int64)

    Ws = {l: model.model.layers[l].mlp.down_proj.weight for l in layers}
    W_base = {l: Ws[l].detach().clone() for l in layers}

    def restore_all():
        with torch.no_grad():
            for l in layers:
                Ws[l].copy_(W_base[l])

    for i, e in enumerate(edits):
        tgt = first_target_token_id(tok, e["target_new"])
        factors = _install_and_capture_one(editor, model, tok, layer, layers, projector, cov,
                                           device, e, steps, lr, v_wd)
        max_rre = 0.0
        for li, (k, kk, r, denom) in enumerate(factors):
            vec["K"][i, li] = k
            vec["KK"][i, li] = kk
            vec["R"][i, li] = r
            vec["denom"][i, li] = denom
            # residual-only sanity (NOT the fidelity check — see _fidelity_check_editor):
            # (outer(r,kk)/denom) @ k ≈ r
            rre = float(np.linalg.norm((np.outer(r, kk) / denom) @ k - r)) / (
                float(np.linalg.norm(r)) + 1e-30)
            max_rre = max(max_rre, rre)
        vec["recon_rel_err"][i] = max_rre

        logits = next_token_logits(model, tok, e["prompt"], device)
        lg = float(logits[tgt].item()); am = int(logits.argmax().item())
        vec["target_tok"][i] = tgt
        vec["logit_solo"][i] = lg
        vec["argmax_ok_solo"][i] = 1.0 if am == tgt else 0.0
        restore_all()
        if (i + 1) % 20 == 0:
            print(f"[fed] {editor} edit {i+1}/{N}  {time.time()-t0:.1f}s", flush=True)

    restore_all()
    for l in layers:
        assert torch.allclose(Ws[l], W_base[l]), f"[fed] solo-loop restore FAILED at layer {l}"
    _finalize_S(vec)
    print(f"[fed] {editor} edits done; esr={float(vec['argmax_ok_solo'].mean()):.3f} "
          f"max_recon_rel_err={float(vec['recon_rel_err'].max()):.2e}  {time.time()-t0:.1f}s",
          flush=True)
    return vec, Ws, W_base, layers


def _fidelity_check_editor(editor, model, tok, layer, device, edits, holdout, args, n_check=1,
                           tol=1e-4):
    """ΔW-FIDELITY GATE (MINOR-1): assert the stored factors reconstruct the SAME weight delta
    the REAL editor installs. For each of the first n_check edits: (1) call the real
    editors/{rome_native,alphaedit,memit}.apply_edit and read the ACTUAL per-layer ΔW = W_after −
    W_base; (2) re-derive ΔW from _install_and_capture_one's factors (outer(r,kk)/denom); assert
    Frobenius rel-err < tol per layer. Replaces the near-tautological recon_rel_err as the fidelity
    check. Returns (worst_relerr, per_layer_report). Restores all weights."""
    import torch
    layers, projector, cov = _editor_context(editor, model, tok, layer, device, edits, holdout,
                                             args, quiet=True)
    steps, lr, v_wd = args.steps, args.lr, 1e-3
    Ws = {l: model.model.layers[l].mlp.down_proj.weight for l in layers}
    W_base = {l: Ws[l].detach().clone() for l in layers}

    def restore():
        with torch.no_grad():
            for l in layers:
                Ws[l].copy_(W_base[l])

    worst = 0.0
    report = []
    for e in edits[:n_check]:
        # (1) REAL editor delta
        if editor == "rome":
            from editors import rome_native as ED
            cfg = {"layer": layer, "steps": steps, "lr": lr, "v_weight_decay": v_wd}
        elif editor == "alpha":
            from editors import alphaedit as ED
            cfg = {"layer": layer, "steps": steps, "lr": lr, "v_weight_decay": v_wd,
                   "projector": projector}
        else:
            from editors import memit as ED
            cfg = {"layers": layers, "z_layer": layer, "steps": steps, "lr": lr,
                   "v_weight_decay": v_wd, "cov": cov}
        ED.apply_edit(model, tok, e, cfg, device)
        actual = {l: (Ws[l].detach() - W_base[l]).float().cpu().numpy() for l in layers}
        restore()
        # (2) my re-derived delta from the captured factors
        factors = _install_and_capture_one(editor, model, tok, layer, layers, projector, cov,
                                           device, e, steps, lr, v_wd)
        restore()
        mine = {layers[li]: (np.outer(r, kk) / denom) for li, (k, kk, r, denom) in enumerate(factors)}
        for l in layers:
            num = float(np.linalg.norm(actual[l] - mine[l]))
            den = float(np.linalg.norm(actual[l])) + 1e-30
            rel = num / den
            worst = max(worst, rel)
            report.append((editor, int(l), rel))
            assert rel < tol, (f"[fidelity] {editor} layer {l}: Frobenius rel-err {rel:.3e} >= {tol} "
                               f"(re-derived ΔW != real editor ΔW)")
    restore()
    for l in layers:
        assert torch.allclose(Ws[l], W_base[l]), f"[fidelity] restore FAILED at layer {l}"
    return worst, report


def _measure_merged_groups_ml(model, tok, device, Ws, W_base, layers, vec, edits, groups):
    """Federation measurement: for each group, form the merged per-layer ΔW = Σ_members
    outer(R,KK)/denom, add to every edited layer's W, measure each member's post-merge
    target-token logit + argmax, restore. Returns (gid, edit, logit_post, argmax_ok_post)."""
    import torch
    from metrics import next_token_logits  # noqa: E402
    KK = vec["KK"]; R = vec["R"]; denom = vec["denom"]; target_tok = vec["target_tok"]
    # torch factors per layer for the fast merged delta R_g^T @ (KK/denom)_g
    Rt = {}; KKsc = {}
    for li, l in enumerate(layers):
        Rt[l] = torch.tensor(R[:, li, :], device=device, dtype=torch.float32)
        KKsc[l] = torch.tensor(KK[:, li, :] / denom[:, li][:, None], device=device, dtype=torch.float32)

    G_id, G_edit, G_lp, G_ap = [], [], [], []
    for gid, group in enumerate(groups):
        gi = torch.tensor(group, device=device, dtype=torch.long)
        with torch.no_grad():
            for l in layers:
                merged = Rt[l].index_select(0, gi).t() @ KKsc[l].index_select(0, gi)  # [d_out, d_in]
                Ws[l].add_(merged.to(Ws[l].dtype))
        for a in group:
            logits = next_token_logits(model, tok, edits[a]["prompt"], device)
            lg = float(logits[int(target_tok[a])].item()); am = int(logits.argmax().item())
            G_id.append(gid); G_edit.append(a); G_lp.append(lg)
            G_ap.append(1.0 if am == int(target_tok[a]) else 0.0)
        with torch.no_grad():
            for l in layers:
                Ws[l].copy_(W_base[l])
    with torch.no_grad():
        for l in layers:
            Ws[l].copy_(W_base[l])
    for l in layers:
        assert torch.allclose(Ws[l], W_base[l]), f"[fed] merge-measure restore FAILED at layer {l}"
    return G_id, G_edit, G_lp, G_ap


def _cov_variant_suffix(editor, args):
    """Bundle/table namespace suffix for --editor memit at a non-identity --memit_cov.

    WHY (R-D, 2026-07-16): the bundle DIRECTORY is derived solely from
    f"{tag}_{editor}_{dataset}_L{layer}_RG" — it does NOT depend on --memit_cov. The prereg's
    PRIMARY arm (identity) already has landed bundles on disk (e.g.
    results/merging_editors/Llama-3.2-1B_memit_cf_L12_RG/, generated 2026-07-16 before this
    change). Running --memit_cov wiki at the SAME (model, editor, dataset, layer) would either
    silently CLOBBER that identity-cov bundle (if --no_refuse_clobber, the driver's own mode) or
    silently REFUSE and exit 0 without running the true-cov cell at all (if the module's own
    refuse-clobber guard sees the existing valid identity-cov table and treats it as "already
    done") — either way a real footgun for a wiki-cov wave at the same cell. Appending the cov
    source (never for 'identity', so every existing/default invocation is byte-identical) gives
    generic/wiki their own directory. Symmetric fix in run_merging_editors.sh's own RG_DIR
    re-derivation (used only for its preflight refuse-check + post-run validate) — see the
    driver's MODEL_BASENAME/RG_DIR lines."""
    if editor == "memit" and getattr(args, "memit_cov", "identity") != "identity":
        return f"_{args.memit_cov}"
    return ""


def run_phase_rg_editor(args):
    """RG operating curve (GPU) for --editor over group sizes x seeds at one z-layer."""
    import torch
    import transformers as _tf
    from experiments.merging_m0 import _load_edit_model
    t0 = time.time()
    device = args.device
    tag = _model_tag(args.model)
    dir_tag = tag + _cov_variant_suffix(args.editor, args)
    # MAJOR-1 / MINOR-3: refuse only a COMPLETE valid bundle, and do it BEFORE the model load when
    # the layer is explicit (no model needed to resolve rg_dir). The driver passes
    # --no_refuse_clobber (its own preflight owns the valid-bundle refuse); this module guard is a
    # backstop for direct invocation that never wedges a half-written bundle.
    if args.refuse_clobber and str(args.layer) != "auto":
        pre_dir = os.path.join(args.out_dir,
                               f"{dir_tag}_{args.editor}_{args.dataset}_L{int(args.layer)}_RG")
        if _bundle_complete(pre_dir):
            raise SystemExit(f"[fed] refuse-clobber: complete valid bundle already at {pre_dir} "
                             f"(pass --no_refuse_clobber to overwrite)")
    model, tok, layer, _nL = _load_edit_model(args.model, args.layer, device)
    seeds = [int(x) for x in str(args.rg_seeds).split(",") if x != ""]
    gsizes = [int(x) for x in str(args.rg_group_sizes).split(",") if x != ""]
    rg_dir = os.path.join(args.out_dir, f"{dir_tag}_{args.editor}_{args.dataset}_L{layer}_RG")

    if args.refuse_clobber and _bundle_complete(rg_dir):
        raise SystemExit(f"[fed] refuse-clobber: complete valid bundle already at {rg_dir} "
                         f"(pass --no_refuse_clobber to overwrite)")

    per_seed_vectors = {}
    edited_layers = None
    obs_seed, obs_g, obs_group, obs_edit, obs_lp, obs_ap = [], [], [], [], [], []
    mem_seed, mem_g, mem_group, mem_edit = [], [], [], []
    for s in seeds:
        edits, holdout = load_edits(args.dataset, args.data, args.n_edits, s, args.n_holdout)
        N = len(edits)
        print(f"[fed] seed {s}: {N} edits + {len(holdout)} holdout ({args.dataset})", flush=True)
        vec, Ws, W_base, layers = _compute_solo_editor(
            args.editor, model, tok, layer, device, edits, holdout, args, t0)
        edited_layers = layers
        per_seed_vectors[s] = vec
        for g in gsizes:
            groups = _tiled_groups(N, g, s)
            gid, ed, lp, ap = _measure_merged_groups_ml(
                model, tok, device, Ws, W_base, layers, vec, edits, groups)
            for gg, e2, l2, a2 in zip(gid, ed, lp, ap):
                obs_seed.append(s); obs_g.append(g); obs_group.append(gg); obs_edit.append(e2)
                obs_lp.append(l2); obs_ap.append(a2)
                mem_seed.append(s); mem_g.append(g); mem_group.append(gg); mem_edit.append(e2)
            print(f"[fed] seed {s} g={g}: {len(groups)} groups measured  {time.time()-t0:.1f}s",
                  flush=True)

    measurements = dict(
        obs_seed=np.array(obs_seed, np.int32), obs_g=np.array(obs_g, np.int32),
        obs_group=np.array(obs_group, np.int32), obs_edit=np.array(obs_edit, np.int32),
        obs_logit_post=np.array(obs_lp, np.float32), obs_argmax_ok_post=np.array(obs_ap, np.float32),
        mem_seed=np.array(mem_seed, np.int32), mem_g=np.array(mem_g, np.int32),
        mem_group=np.array(mem_group, np.int32), mem_edit=np.array(mem_edit, np.int32),
    )
    # MINOR-2: store per-observation drop + dose so downstream absolute-GAIN analysis needs no model.
    obs_drop, obs_dose = _obs_dose_drop(per_seed_vectors, measurements)
    measurements["obs_drop"] = obs_drop
    measurements["obs_dose"] = obs_dose
    meta = dict(
        schema_version=SCHEMA_VERSION, experiment="RG_editors", model=args.model, model_tag=tag,
        editor=args.editor, edited_layers=[int(l) for l in (edited_layers or [layer])],
        dataset=args.dataset, layer=layer, seeds=seeds, group_sizes=gsizes, n_edits=args.n_edits,
        steps=args.steps, lr=args.lr, keep_ratio=args.keep_ratio, memit_cov=args.memit_cov,
        memit_cov_resolved=getattr(args, "_memit_cov_resolved", "identity"),
        device=device,
        esr_by_seed={str(s): round(float(per_seed_vectors[s]["argmax_ok_solo"].mean()), 4) for s in seeds},
        max_recon_rel_err_by_seed={str(s): float(per_seed_vectors[s]["recon_rel_err"].max())
                                   for s in seeds},
        torch=torch.__version__, transformers=_tf.__version__, numpy=np.__version__,
    )
    save_rg(rg_dir, per_seed_vectors, measurements, meta)
    print(f"[fed] saved RG bundle -> {rg_dir}", flush=True)
    table = analyze_rg_ml(per_seed_vectors, measurements, meta)
    out = args.table_out or os.path.join(
        args.out_dir, f"RG_editors_table_{dir_tag}_{args.editor}_{args.dataset}_L{layer}.json")
    _write_table(table, out)
    _write_table(table, os.path.join(rg_dir, "RG_editors_table.json"))
    print_rg_table(table)
    print(f"[fed] wrote {out}  total {time.time()-t0:.1f}s", flush=True)


# ============================================================ smoke: real-model ΔW fidelity gate
def run_smoke(args):
    """--smoke (MINOR-1 real-model closure): load the real model, run n_check edits per the chosen
    --editor through the REAL editor apply_edit, and assert the re-derived ΔW (stored factors)
    matches to Frobenius rel-err < 1e-4 per layer. Wired into run_merging_editors.sh as a first-run
    gate. Runs on --device (cuda in production; cpu here for CPU-only build validation)."""
    from experiments.merging_m0 import _load_edit_model
    device = args.device
    model, tok, layer, _nL = _load_edit_model(args.model, args.layer, device)
    edits, holdout = load_edits(args.dataset, args.data, max(args.smoke_n, 4), args.seed,
                                args.n_holdout)
    worst, report = _fidelity_check_editor(args.editor, model, tok, layer, device, edits, holdout,
                                           args, n_check=args.smoke_n, tol=args.fidelity_tol)
    for ed, l, rel in report:
        print(f"[smoke] {ed} L{l}: Frobenius rel-err = {rel:.3e}", flush=True)
    print(f"[smoke] {args.editor} ΔW-FIDELITY PASS — worst rel-err {worst:.3e} < {args.fidelity_tol} "
          f"(re-derived factors reconstruct the real editor's ΔW)", flush=True)
    return worst


def _tiny_vocab_size(tok):
    """Embedding size for the tiny fixture that COVERS every id `tok` can emit — tokenizer-agnostic.
    tok.vocab_size reports only the BASE vocab and EXCLUDES added/special tokens: e.g. Llama-3's BOS
    id 128000 == vocab_size 128000 (out of range) while len(tok)==128256 includes it. Size to
    max(len(tok), vocab_size) + margin so BOS/special ids never index past the embedding (the box
    IndexError). Qwen adds no BOS so it fit by luck; this makes the fixture robust to any tokenizer."""
    return max(int(len(tok)), int(getattr(tok, "vocab_size", 0) or 0)) + 16


def _build_tiny_llama(tok):
    """A TINY random-weight LlamaForCausalLM (hidden 32, 4 layers) whose embedding covers `tok`'s
    full id space — for the CPU ΔW-fidelity self-test. Native Llama arch so
    model.model.layers[l].mlp.down_proj exists without arch normalisation; random weights are fine
    (fidelity is a weight-delta identity, not a prediction test)."""
    import torch
    from transformers import LlamaForCausalLM, LlamaConfig
    vocab = _tiny_vocab_size(tok)
    cfg = LlamaConfig(vocab_size=vocab, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=4,
                      max_position_embeddings=64, tie_word_embeddings=True)
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg).to("cpu").float().eval()
    return model


def _find_local_tokenizer(prefer=None):
    """Discover a local tokenizer dir (offline) for the tiny-model fidelity self-test. Returns a
    path or None (then the self-test SKIPs the tiny check — the driver's real-model --smoke is the
    hard gate). `prefer` overrides the search order (used by the large-vocab regression to force a
    Llama tokenizer)."""
    roots = [os.path.join(HARNESS, "data", "models")]
    prefer = prefer or ["Qwen2.5-0.5B", "Llama-3.2-1B", "Qwen2.5-1.5B", "gpt2-xl"]
    for root in roots:
        if not os.path.isdir(root):
            continue
        names = os.listdir(root)
        for p in prefer:
            if p in names and os.path.isfile(os.path.join(root, p, "config.json")):
                d = os.path.join(root, p)
                if any(os.path.isfile(os.path.join(d, f))
                       for f in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json")):
                    return d
    return None


def _largevocab_regression():
    """Regression for the box IndexError (E2 editor cells aborted at selftest): FORCE a Llama
    tokenizer — whose BOS id (128000) equals tok.vocab_size and so was NOT covered when the fixture
    sized its embedding from vocab_size — and confirm the tiny fixture now covers it: encode a
    prompt (prepends BOS) and run a forward pass with no IndexError at embed_tokens."""
    lla = _find_local_tokenizer(prefer=["Llama-3.2-1B", "Llama-3.1-8B", "Llama-3.2-1B-Instruct"])
    if lla is None:
        print("[selftest] (f) large-vocab regression: SKIP (no Llama tokenizer under data/models)",
              flush=True)
        return None
    import torch
    from transformers import AutoTokenizer
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    tok = AutoTokenizer.from_pretrained(lla)
    ids = tok("Paris is the capital of", return_tensors="pt")["input_ids"]
    max_id = int(ids.max().item()); vsz = _tiny_vocab_size(tok)
    assert max_id < vsz, (f"tiny fixture vocab {vsz} does not cover max token id {max_id} "
                          f"(len={len(tok)}, vocab_size={getattr(tok,'vocab_size',None)})")
    model = _build_tiny_llama(tok)
    with torch.no_grad():
        model(ids)   # embed_tokens would IndexError here if the fixture were mis-sized (the box bug)
    print(f"[selftest]   large-vocab regression OK ({os.path.basename(lla)}: max_id={max_id} "
          f"< tiny_vocab={vsz}, len={len(tok)}, vocab_size={getattr(tok,'vocab_size',None)}; "
          f"forward ran clean)", flush=True)
    return True


def _tiny_fidelity_selftest():
    """CPU ΔW-fidelity check on a tiny random-weight model (MINOR-1). Best-effort: needs a local
    tokenizer; SKIPs loudly if none is found (the driver's real-model --smoke still closes the loop).
    Runs one edit per editor through the REAL apply_edit and asserts re-derived ΔW matches."""
    tokdir = _find_local_tokenizer()
    if tokdir is None:
        print("[selftest] (e) ΔW-fidelity: SKIP (no local tokenizer under data/models; the "
              "driver's real-model --smoke gate is the hard closure)", flush=True)
        return None
    try:
        from transformers import AutoTokenizer
    except Exception as ex:
        print(f"[selftest] (e) ΔW-fidelity: SKIP (transformers unavailable: {ex})", flush=True)
        return None
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    tok = AutoTokenizer.from_pretrained(tokdir)
    model = _build_tiny_llama(tok)

    class _A:  # minimal args holder
        steps, lr, keep_ratio = 3, 0.1, 0.99
        memit_layers, memit_span, memit_cov, cov_max_tokens = "auto", 3, "identity", 2000
    a = _A()
    edits = [{"subject": "Paris", "prompt": "Paris is the capital of", "target_new": "Spain",
              "target_true": "France"},
             {"subject": "Rome", "prompt": "Rome is the capital of", "target_new": "Egypt",
              "target_true": "Italy"}]
    holdout = [{"subject": "Berlin", "prompt": "Berlin is the capital of", "target_new": "Peru",
                "target_true": "Germany"}]
    z_layer = 2  # tiny model has 4 layers; MEMIT auto span 3 -> layers [0,1,2], max==z ✓
    worst = 0.0
    for ed in EDITORS:
        w, rep = _fidelity_check_editor(ed, model, tok, z_layer, "cpu", edits, holdout, a,
                                        n_check=1, tol=1e-4)
        worst = max(worst, w)
        print(f"[selftest]   ΔW-fidelity {ed}: worst rel-err {w:.3e} "
              f"(layers {[l for _e, l, _r in rep]}) — PASS", flush=True)
    print(f"[selftest]   ΔW-FIDELITY OK (tiny random Llama, tokenizer {os.path.basename(tokdir)}) "
          f"worst {worst:.3e} < 1e-4", flush=True)
    return worst


# ============================================================ self-test (CPU, no model)
def _assert_cross_term_bruteforce(rng, trials=100, N=6, L=3, d_in=13, d_out=9, tol=1e-9):
    """(a) generalised cross-term (‡)/(§) vs brute-force dense ΔW @ k_a, with kk≠k and L>1."""
    worst = 0.0
    for _ in range(trials):
        K = rng.standard_normal((N, L, d_in))
        KK = rng.standard_normal((N, L, d_in))      # effective keys DIFFER from receiver keys
        R = rng.standard_normal((N, L, d_out))
        denom = rng.uniform(0.5, 3.0, (N, L))
        a = int(rng.integers(N))
        others = [b for b in range(N) if b != a]
        # closed form (§): d_a = Σ_l Σ_b r_b^l (kk_b^l·k_a^l)/denom_b^l
        d_closed = np.zeros(d_out)
        for li in range(L):
            for b in others:
                d_closed += cross_term_eff(R[b, li], KK[b, li], K[a, li], denom[b, li])
        # brute force: build each dense ΔW_b^l, sum, apply to k_a^l, sum over layers
        d_brute = np.zeros(d_out)
        for li in range(L):
            merged = np.zeros((d_out, d_in))
            for b in others:
                merged += dense_delta_eff(R[b, li], KK[b, li], denom[b, li])
            d_brute += merged @ K[a, li]
        err = float(np.max(np.abs(d_closed - d_brute)))
        worst = max(worst, err)
        assert np.allclose(d_closed, d_brute, rtol=tol, atol=1e-10), f"cross-term broke: {err:.3e}"
    return worst


def _assert_federation_additivity(rng, trials=50, N=5, d_in=11, d_out=7, tol=1e-10):
    """(b) (Σ_b ΔW_b) @ k_a == Σ_b (ΔW_b @ k_a) — weight-space federation additivity."""
    worst = 0.0
    for _ in range(trials):
        R = rng.standard_normal((N, d_out)); KK = rng.standard_normal((N, d_in))
        denom = rng.uniform(0.5, 2.0, N); k_a = rng.standard_normal(d_in)
        merged = sum(dense_delta_eff(R[b], KK[b], denom[b]) for b in range(N))
        lhs = merged @ k_a
        rhs = sum(cross_term_eff(R[b], KK[b], k_a, denom[b]) for b in range(N))
        err = float(np.max(np.abs(lhs - rhs)))
        worst = max(worst, err)
        assert np.allclose(lhs, rhs, rtol=tol, atol=1e-12), f"additivity broke: {err:.3e}"
    return worst


def _single_layer_fixture(rng, N=40, d_in=16, d_out=8, group_size=20, seed_off=0):
    """A single-layer (L=1, kk=k) ROME-shaped fixture in BOTH schemas (flat for merging_m0,
    [N,1,·] for this module) + one tiled group, with synthetic I_cos-driven drops. Used by the
    equivalence anchor: both analyses must return identical I_cos/I_mag/||d_a||/rho/partial."""
    from experiments.merging_m0 import _crowded_keys
    K = _crowded_keys(rng, N, d_in).astype(np.float64)
    R = rng.standard_normal((N, d_out)).astype(np.float64)
    key_norm = np.linalg.norm(K, axis=1)
    resid_norm = np.linalg.norm(R, axis=1)
    denom = key_norm ** 2 + 1e-8
    S = resid_norm / (key_norm + 1e-12)           # merging_m0's S (fed identically to both)
    logit_solo = (5.0 + rng.standard_normal(N)).astype(np.float64)
    argmax_ok_solo = np.ones(N, np.float64)

    groups = _tiled_groups(N, group_size, seed_off)
    # build I_cos-driven drops so rho/partial are well-defined and non-degenerate
    rows = []
    for gid, group in enumerate(groups):
        for a in group:
            others = [b for b in group if b != a]
            ic = float(key_norm[a] * np.sum([S[b] * abs(float(K[b] @ K[a]) /
                       (key_norm[b] * key_norm[a] + 1e-12)) for b in others]))
            rows.append((gid, a, ic))
    ic_arr = np.array([r[2] for r in rows])
    sig = (ic_arr - ic_arr.mean()) / (ic_arr.std() + 1e-12)
    drops = 0.7 * sig + 0.05 * rng.standard_normal(len(rows))

    obs_edit, obs_group, obs_lp, obs_ap = [], [], [], []
    members = defaultdict(list)
    for (gid, a, _ic), dr in zip(rows, drops):
        obs_edit.append(a); obs_group.append(gid)
        obs_lp.append(float(logit_solo[a] - dr)); obs_ap.append(1.0)
        members[gid].append(a)
    flat = dict(K=K, R=R, denom=denom, S=S, key_norm=key_norm,
                logit_solo=logit_solo, argmax_ok_solo=argmax_ok_solo)
    ml = dict(K=K[:, None, :], KK=K[:, None, :], R=R[:, None, :], denom=denom[:, None],
              S=S[:, None], key_norm=key_norm[:, None],
              logit_solo=logit_solo, argmax_ok_solo=argmax_ok_solo)
    obs = (np.array(obs_edit), np.array(obs_group), np.array(obs_lp), np.array(obs_ap),
           {int(g): v for g, v in members.items()})
    return flat, ml, obs


def _assert_rome_equivalence(rng):
    """(c) ROME-equivalence anchor: _regime_stat_ml on the L=1 view reproduces
    merging_m0._regime_stat on the flat view to fp64 for every shared numeric field."""
    from experiments.merging_m0 import _regime_stat as m0_stat
    t = _default_thresholds()
    flat, ml, (oe, og, olp, oap, members) = _single_layer_fixture(rng)
    m0 = m0_stat((flat["K"], flat["R"], flat["denom"], flat["S"], flat["key_norm"],
                  flat["logit_solo"], flat["argmax_ok_solo"]),
                 oe, og, olp, oap, members, t)
    me = _regime_stat_ml((ml["K"], ml["KK"], ml["R"], ml["denom"], ml["S"], ml["key_norm"],
                          ml["logit_solo"], ml["argmax_ok_solo"]),
                         oe, og, olp, oap, members, t)
    shared = ["rho_I_cos_drop", "rho_I_magonly_drop", "partial_rho_geom",
              "partial_rho_geom_ownmag", "delta_rho_cos_minus_mag",
              "rho_I_cos_normdelta_cpu_xcheck", "median_abs_drop_logit", "mean_drop_logit",
              "argmax_loss_rate", "non_negligible", "saturated", "c3_eligible"]
    worst = 0.0
    for f in shared:
        a, b = m0.get(f), me.get(f)
        if isinstance(a, bool) or a is None or b is None:
            assert a == b, f"[selftest] ROME-equiv field {f}: m0={a!r} ml={b!r}"
        else:
            e = abs(float(a) - float(b)); worst = max(worst, e)
            assert e < 1e-9, f"[selftest] ROME-equiv field {f}: m0={a} ml={b} (|Δ|={e:.2e})"
    return worst, m0, me


def _make_rg_ml_fixture(mode, seeds=(0, 1, 2), gsizes=(2, 3, 5, 10, 20, 50), N=60, L=3,
                        d_in=16, d_out=8):
    """Synthetic multi-layer RG bundle. mode='pass' drives drop by I_cos (geometry beyond
    magnitude) at small non-saturated g -> PASS; mode='kill' drives drop by I_mag with collinear
    effective keys -> partial ~0 everywhere -> KILL. Large g saturated to exercise the boundary."""
    from experiments.merging_m0 import _crowded_keys
    per_seed = {}
    O = defaultdict(list)
    for s in seeds:
        srng = np.random.default_rng(7000 + s + (0 if mode == "pass" else 500))
        K = np.zeros((N, L, d_in)); KK = np.zeros((N, L, d_in)); R = np.zeros((N, L, d_out))
        denom = np.zeros((N, L))
        for li in range(L):
            if mode == "pass":
                Kl = _crowded_keys(srng, N, d_in).astype(np.float64)
                KKl = Kl.copy()
            else:
                shared = srng.standard_normal(d_in); shared /= (np.linalg.norm(shared) + 1e-12)
                Kl = (shared[None, :] * srng.uniform(0.7, 1.4, N)[:, None]).astype(np.float64)
                KKl = Kl.copy()                       # collinear -> |cos|==1 -> partial degenerates
            Rl = srng.standard_normal((N, d_out))
            K[:, li, :] = Kl; KK[:, li, :] = KKl; R[:, li, :] = Rl
            denom[:, li] = np.linalg.norm(Kl, axis=1) ** 2 + 1e-8
        key_norm = np.linalg.norm(K, axis=2)
        kk_norm = np.linalg.norm(KK, axis=2)
        Sarr = np.linalg.norm(R, axis=2) * kk_norm / (denom + 1e-30)
        logit_solo = 5.0 + srng.standard_normal(N)
        per_seed[s] = dict(
            K=K.astype(np.float32), KK=KK.astype(np.float32), R=R.astype(np.float32),
            denom=denom.astype(np.float64), key_norm=key_norm.astype(np.float32),
            S=Sarr.astype(np.float32), resid_norm=np.linalg.norm(R, axis=2).astype(np.float32),
            target_tok=np.arange(N, dtype=np.int64),
            logit_solo=logit_solo.astype(np.float32), argmax_ok_solo=np.ones(N, np.float32),
            recon_rel_err=np.zeros(N, np.float32), layers=np.arange(L, dtype=np.int64))
        for g in gsizes:
            groups = _tiled_groups(N, g, s)
            if not groups:
                continue
            rows = []
            for gid, group in enumerate(groups):
                for a in group:
                    others = [b for b in group if b != a]
                    ic = im = 0.0
                    for li in range(L):
                        for b in others:
                            cab = abs(float(KK[b, li] @ K[a, li]) /
                                      (np.linalg.norm(KK[b, li]) * key_norm[a, li] + 1e-12))
                            ic += key_norm[a, li] * Sarr[b, li] * cab
                            im += key_norm[a, li] * Sarr[b, li]
                    rows.append((gid, a, ic, im))
            Ic = np.array([r[2] for r in rows]); Im = np.array([r[3] for r in rows])
            sig = (Ic if mode == "pass" else Im)
            sig = (sig - sig.mean()) / (sig.std() + 1e-12)
            drops = 0.7 * sig + 0.05 * srng.standard_normal(len(rows))
            target = 0.2 if g <= 20 else 0.9
            am = np.ones(len(rows)); kf = int(round(target * len(rows)))
            if kf > 0:
                am[np.argsort(-drops)[:kf]] = 0.0
            for (gid, a, _ic, _im), dr, ao in zip(rows, drops, am):
                O["obs_seed"].append(s); O["obs_g"].append(g); O["obs_group"].append(gid)
                O["obs_edit"].append(a); O["obs_logit_post"].append(float(logit_solo[a] - dr))
                O["obs_argmax_ok_post"].append(float(ao))
                O["mem_seed"].append(s); O["mem_g"].append(g); O["mem_group"].append(gid)
                O["mem_edit"].append(a)
    measurements = dict(
        obs_seed=np.array(O["obs_seed"], np.int32), obs_g=np.array(O["obs_g"], np.int32),
        obs_group=np.array(O["obs_group"], np.int32), obs_edit=np.array(O["obs_edit"], np.int32),
        obs_logit_post=np.array(O["obs_logit_post"], np.float32),
        obs_argmax_ok_post=np.array(O["obs_argmax_ok_post"], np.float32),
        mem_seed=np.array(O["mem_seed"], np.int32), mem_g=np.array(O["mem_g"], np.int32),
        mem_group=np.array(O["mem_group"], np.int32), mem_edit=np.array(O["mem_edit"], np.int32))
    meta = dict(schema_version=SCHEMA_VERSION, experiment="RG_editors",
                model=f"SYNTHETIC_RG_{mode}", model_tag=f"synthetic_rg_{mode}", editor="synthetic",
                edited_layers=list(range(L)), dataset="synthetic", layer=-1,
                seeds=list(seeds), group_sizes=list(gsizes), n_edits=N)
    return per_seed, measurements, meta


def selftest(selftest_dir):
    os.makedirs(selftest_dir, exist_ok=True)
    rng = np.random.default_rng(20260716)

    print("[selftest] (a) generalised cross-term (kk≠k, multi-layer) vs brute-force dense ΔW ...",
          flush=True)
    w = _assert_cross_term_bruteforce(rng)
    print(f"[selftest]   CROSS-TERM OK — worst |closed − brute| over 100 trials = {w:.3e}", flush=True)

    print("[selftest] (b) federation additivity (Σ ΔW_b)@k_a == Σ (ΔW_b@k_a) ...", flush=True)
    w = _assert_federation_additivity(rng)
    print(f"[selftest]   ADDITIVITY OK — worst residual over 50 trials = {w:.3e}", flush=True)

    print("[selftest] (c) ROME-equivalence anchor: _regime_stat_ml (L=1) == merging_m0._regime_stat ...",
          flush=True)
    w, m0, me = _assert_rome_equivalence(rng)
    print(f"[selftest]   ROME-EQUIV OK — worst |Δ| over shared fields = {w:.3e}", flush=True)
    print(f"[selftest]     m0 partial_rho_geom={m0['partial_rho_geom']}  "
          f"ml partial_rho_geom={me['partial_rho_geom']}  (identical)", flush=True)

    print("[selftest] (d) end-to-end RG on synthetic multi-layer bundles ...", flush=True)
    for mode, expect in [("pass", "PASS"), ("kill", "KILL")]:
        per_seed, meas, meta = _make_rg_ml_fixture(mode)
        rg_dir = os.path.join(selftest_dir, f"rg_{mode}")
        save_rg(rg_dir, per_seed, meas, meta)                # disk round-trip
        ps2, me2, mt2 = load_rg(rg_dir)
        rgt = analyze_rg_ml(ps2, me2, mt2)
        _write_table(rgt, os.path.join(rg_dir, "RG_editors_table.json"))
        print_rg_table(rgt)
        got = rgt["verdict"]["overall"]
        assert got == expect, f"[selftest] RG {mode}: overall {got!r} != expected {expect!r}"
        if mode == "pass":
            assert rgt["verdict"]["n_qualifying_group_sizes"] >= 2, \
                "[selftest] RG pass: need >= 2 qualifying group sizes"
            assert rgt["scoped_federation_boundary"]["conservative_min_across_seeds"] is not None, \
                "[selftest] RG pass: scoped-federation boundary should be recorded"
        if mode == "kill":
            assert not rgt["verdict"]["any_geometry_pass"], \
                "[selftest] RG kill: no cell should pass geometry"
        print(f"[selftest]   RG {mode}: verdict={got} "
              f"qualifying_g={rgt['verdict']['qualifying_group_sizes']} — as expected", flush=True)

    print("[selftest] (e) ΔW-fidelity: re-derived factors vs REAL editor apply_edit (tiny model) ...",
          flush=True)
    fid = _tiny_fidelity_selftest()

    print("[selftest] (f) large-vocab regression: tiny fixture survives a Llama tokenizer "
          "(BOS id == vocab_size) ...", flush=True)
    _largevocab_regression()

    tail = "" if fid is not None else " [ΔW-fidelity SKIPPED — no local tokenizer; --smoke is the gate]"
    print("\n[selftest] ALL CHECKS PASSED (cross-term brute-force + federation additivity + "
          "ROME-equivalence anchor + RG pass/kill + ΔW-fidelity, disk round-trips)" + tail, flush=True)
    return True


# ============================================================ CLI
def main():
    ap = argparse.ArgumentParser(description="Editor-general RG federation (rome/memit/alpha).")
    ap.add_argument("--selftest", action="store_true",
                    help="CPU self-test: cross-term/additivity/ROME-equivalence + RG pass/kill "
                         "on synthetic fixtures (no model, no GPU). Writes only under --selftest_dir.")
    ap.add_argument("--selftest_dir",
                    default=os.path.join(HARNESS, "results", "merging_editors", "selftest"))
    ap.add_argument("--rg", action="store_true",
                    help="RG operating curve (GPU): group sizes x seeds at one z-layer.")
    ap.add_argument("--smoke", action="store_true",
                    help="ΔW-fidelity gate (loads the real model): assert re-derived ΔW matches the "
                         "REAL editor apply_edit to Frobenius rel-err < --fidelity_tol per layer.")
    ap.add_argument("--smoke_n", type=int, default=2, help="edits to check in --smoke mode")
    ap.add_argument("--fidelity_tol", type=float, default=1e-4)
    ap.add_argument("--rg_phase2_dir", default=None,
                    help="STANDALONE CPU reanalysis of a saved RG bundle dir — no model, no GPU.")
    ap.add_argument("--editor", choices=EDITORS, default="rome")
    ap.add_argument("--dataset", choices=["cf", "zsre"], default="cf")
    ap.add_argument("--rg_seeds", default="0,1,2")
    ap.add_argument("--rg_group_sizes", default="2,3,5,10,20")
    # phase-1 (GPU) args
    ap.add_argument("--model", default=os.path.join(HARNESS, "data", "models", "Llama-3.2-1B"))
    ap.add_argument("--data", default=os.path.join(HARNESS, "data", "counterfact.json"))
    ap.add_argument("--n_edits", type=int, default=200)
    ap.add_argument("--n_holdout", type=int, default=50,
                    help="disjoint holdout pool for the AlphaEdit preserved-key / MEMIT cov bank")
    ap.add_argument("--layer", default="12", help="z-layer (int) or 'auto' for n_layers//2")
    ap.add_argument("--seed", type=int, default=0, help="edit-selection seed for --smoke")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    # editor-specific
    ap.add_argument("--keep_ratio", type=float, default=0.99, help="AlphaEdit preserved-energy ratio")
    ap.add_argument("--memit_layers", default="auto", help="MEMIT layer span: 'auto' or comma ints")
    ap.add_argument("--memit_span", type=int, default=4, help="MEMIT auto span (layers up to z)")
    ap.add_argument("--memit_cov", choices=["identity", "generic", "wiki"], default="identity",
                    help="MEMIT covariance source (identity = ROME-style; generic = this cell's "
                         "own holdout bank; wiki = external wikitext corpus if present under "
                         "data/, else a cell-independent CounterFact prompt sample [logged as "
                         "'cf_fallback'] — see _load_wiki_or_fallback_prompts). generic/wiki are "
                         "cached under results/merging_editors/cov_cache/.")
    ap.add_argument("--cov_max_tokens", type=int, default=20000)
    # output
    ap.add_argument("--out_dir", default=os.path.join(HARNESS, "results", "merging_editors"))
    ap.add_argument("--table_out", default=None)
    ap.add_argument("--refuse_clobber", dest="refuse_clobber", action="store_true", default=True,
                    help="refuse to overwrite an existing RG bundle (default on)")
    ap.add_argument("--no_refuse_clobber", dest="refuse_clobber", action="store_false")
    # thresholds
    ap.add_argument("--neg_logit", type=float, default=DEF_NEG_LOGIT)
    ap.add_argument("--neg_argmax", type=float, default=DEF_NEG_ARGMAX)
    ap.add_argument("--rho_min", type=float, default=DEF_RHO_MIN)
    ap.add_argument("--drho_min", type=float, default=DEF_DRHO_MIN)
    ap.add_argument("--partial_min", type=float, default=DEF_PARTIAL_MIN)
    ap.add_argument("--sat_argmax", type=float, default=DEF_SAT_ARGMAX)
    args = ap.parse_args()

    thr = dict(neg_logit=args.neg_logit, neg_argmax=args.neg_argmax, rho_min=args.rho_min,
               drho_min=args.drho_min, partial_min=args.partial_min, sat_argmax=args.sat_argmax)

    if args.selftest:
        selftest(args.selftest_dir)
        return
    if args.smoke:
        run_smoke(args)
        return
    if args.rg_phase2_dir:
        per_seed, meas, meta = load_rg(args.rg_phase2_dir)
        table = analyze_rg_ml(per_seed, meas, meta, thr)
        out = args.table_out or os.path.join(args.rg_phase2_dir, "RG_editors_table.json")
        _write_table(table, out)
        print_rg_table(table)
        print(f"[fed] RG reanalysis wrote {out}", flush=True)
        return
    if args.rg:
        run_phase_rg_editor(args)
        return
    ap.error("nothing to do: pass --selftest, --rg, or --rg_phase2_dir")


if __name__ == "__main__":
    main()
