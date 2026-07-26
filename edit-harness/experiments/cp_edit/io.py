"""cp_edit/io.py — npz loader + masks + 4 probe-outcome-free score functions.

Shared library for the CP-Edit CPU analysis stack (KG-0 / E1 / E2 / E5).
CPU only. numpy on existing .npz. 0 GPU, 0 downloads.

LAB CONVENTIONS honored (see mechanism_sc_table.py / analyze_matrices.py):
  * masked(d, known, edit_ok): edit_ok row mask (edit_ok>0.5), known col mask
    (pre_p>0.05, applied only if >=5 probes survive else keep all). Same order as
    mechanism_sc_table.masked so per-edit numbers are commensurate with C1/C4.
  * S factor: per mechanism_sc_table.py, S is effectively resid_norm = ||v-Wk||
    (the S-NUMERATOR, NOT divided by ||k||) when the npz stores only 'resid_norm'.
    The 24 gate/g4 npz store 'resid_norm' (no 'S' array), so S_i = resid_norm[i].
  * C factor: |cos| (absolute cosine), matching mechanism_sc_table.absC.
  * damage: signed damage_logit[i,j] = pre_l[j] - post_l (positive = damaged).
    Project rule: SIGNED metric, NEVER AUROC.

PER-EDIT REDUCTION (the CP-Edit convention, distinct from the within-probe
headline pipeline): CP certifies a per-edit scalar target
    y_i = mean_j damage_logit[i,j]   over the seed's masked probe columns  (SIGNED).
The lab's headline pipeline never collapses probes to a per-edit scalar; CP does,
and this is stated verbatim in the prereg. Four PROBE-OUTCOME-FREE predictors per
edit (computable at edit time, before deployment — terminology corrected
2026-07-01: NOT all are strictly pre-edit; 'pre-edit-only' is reserved for keycos):
    SxC_i    = resid_norm[i] * mean_j|COS[i,j]|   [EDIT-COMPUTATION-TIME: S=||v-Wk||
                                                   arises during the ROME value-opt]
    keycos_i = mean_j|COS[i,j]|                   [strictly PRE-EDIT]
    NG_i     = norm_growth[i] = ||delta_W||       [REQUIRES THE EDIT TO BE COMPUTED:
                                                   post-edit weights, probe-outcome-free]
    marginal = 1  (constant)
No predictor reads any post-edit probe outcome (damage_*, post-edit logit/prob).

EXCHANGEABILITY: verified against killgate_keygeom.load_counterfact — each seed
does default_rng(seed).shuffle(data) BEFORE slicing recs[:n_edits], so seeds
s0/s1/s2 edit DIFFERENT 200 requests (and different probe banks). Pooling 600
distinct-request edits therefore does NOT leak request identity across a random
split. Splits are still SEED-STRATIFIED (each seed's probe bank differs; y_i is
precomputed per edit against its own seed's probes). group_key = (seed, edit_idx)
is recorded for auditability; no request-level grouping is required.
"""
from __future__ import annotations

import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MATRIX_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "results", "matrices"))

ROME_FMT = "gate_llama1b_rome_cf_L{L}_s{s}.npz"
ALPHA_FMT = "g4_llama1b_alpha_cf_L{L}_s{s}.npz"
LAYERS = (8, 10, 12, 14)
SEEDS = (0, 1, 2)


def masked(d, known=True, edit_ok=True):
    """Replica of mechanism_sc_table.masked: returns (COS2, D2, S_row, NG_row, row_mask, col_mask).

    COS2, D2 are [n_rows, n_cols]; S_row/NG_row are per-edit aligned to kept rows."""
    COS = d["COS"].astype(float)
    D = d["damage_logit"].astype(float)
    S_full = d["S"].astype(float) if "S" in d.files else (
        d["resid_norm"].astype(float) if "resid_norm" in d.files else None)
    NG_full = d["norm_growth"].astype(float) if "norm_growth" in d.files else None

    row = np.ones(COS.shape[0], bool)
    if edit_ok and "edit_ok" in d.files:
        row = d["edit_ok"].astype(float) > 0.5
    col = np.ones(COS.shape[1], bool)
    if known and "pre_p" in d.files:
        c = d["pre_p"].astype(float) > 0.05
        if c.sum() >= 5:
            col = c
    COS2 = COS[row][:, col]
    D2 = D[row][:, col]
    S_row = None if S_full is None else S_full[row]
    NG_row = None if NG_full is None else NG_full[row]
    return COS2, D2, S_row, NG_row, row, col


def per_edit_scores(COS2, D2, S_row, NG_row):
    """Collapse masked [n,m] matrices to per-edit scalars (the CP-Edit convention).

    Returns dict of per-edit arrays: y (signed mean damage), keycos, SxC, NG,
    marginal (ones), plus finite-count bookkeeping.
    """
    absC = np.abs(COS2)                       # [n,m]  |cos|
    keycos = np.nanmean(absC, axis=1)          # [n]    mean_j |cos|
    y = np.nanmean(D2, axis=1)                 # [n]    signed mean damage
    sxc = (S_row * keycos) if S_row is not None else np.full_like(keycos, np.nan)
    ng = NG_row.copy() if NG_row is not None else np.full_like(keycos, np.nan)
    marginal = np.ones_like(keycos)
    return {
        "y": y, "keycos": keycos, "SxC": sxc, "NG": ng, "marginal": marginal,
    }


def load_layer(editor, layer, matrix_dir=MATRIX_DIR, known=True, edit_ok=True,
               seeds=SEEDS):
    """Pool the 3 seeds of a (editor, layer) into per-edit CP arrays.

    Returns a dict with pooled per-edit arrays (y, keycos, SxC, NG, marginal),
    seed_labels, group_keys, and full bookkeeping (mask counts, NaN counts).
    """
    fmt = ROME_FMT if editor == "rome" else ALPHA_FMT
    ys, kcs, scs, ngs, mgs, seed_lab, grp = [], [], [], [], [], [], []
    book = {"editor": editor, "layer": int(layer), "per_seed": [],
            "nan_y_count": 0, "nan_keycos_count": 0, "nan_SxC_count": 0,
            "nan_NG_count": 0, "n_edits_pooled": 0}
    for s in seeds:
        path = os.path.join(matrix_dir, fmt.format(L=layer, s=s))
        d = np.load(path)
        COS2, D2, S_row, NG_row, row, col = masked(d, known, edit_ok)
        sc = per_edit_scores(COS2, D2, S_row, NG_row)
        n = sc["y"].shape[0]
        # count non-finite per-edit values (never silently dropped)
        nan_y = int(np.sum(~np.isfinite(sc["y"])))
        nan_kc = int(np.sum(~np.isfinite(sc["keycos"])))
        nan_sc = int(np.sum(~np.isfinite(sc["SxC"])))
        nan_ng = int(np.sum(~np.isfinite(sc["NG"])))
        book["nan_y_count"] += nan_y
        book["nan_keycos_count"] += nan_kc
        book["nan_SxC_count"] += nan_sc
        book["nan_NG_count"] += nan_ng
        book["per_seed"].append({
            "seed": int(s), "npz": os.path.basename(path),
            "n_edits_kept": int(n), "n_edits_raw": int(row.shape[0]),
            "n_edits_dropped_edit_ok": int(row.shape[0] - int(row.sum())),
            "n_probes_kept": int(col.sum()), "n_probes_raw": int(col.shape[0]),
            "nan_y": nan_y, "nan_keycos": nan_kc, "nan_SxC": nan_sc, "nan_NG": nan_ng,
        })
        ys.append(sc["y"]); kcs.append(sc["keycos"]); scs.append(sc["SxC"])
        ngs.append(sc["NG"]); mgs.append(sc["marginal"])
        seed_lab.append(np.full(n, s, dtype=int))
        grp.append(np.array([(s, i) for i in range(n)], dtype=object))
    out = {
        "editor": editor, "layer": int(layer),
        "y": np.concatenate(ys),
        "scores": {
            "SxC": np.concatenate(scs),
            "keycos": np.concatenate(kcs),
            "NG": np.concatenate(ngs),
            "marginal": np.concatenate(mgs),
        },
        "seed_labels": np.concatenate(seed_lab),
        "group_keys": np.concatenate(grp),
        "book": book,
    }
    book["n_edits_pooled"] = int(out["y"].shape[0])
    return out


SCORE_ORDER = ("SxC", "keycos", "NG", "marginal")
NORMALIZED = {"SxC": True, "keycos": True, "NG": True, "marginal": False}
