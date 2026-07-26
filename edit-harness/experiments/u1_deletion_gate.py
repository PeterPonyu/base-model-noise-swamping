"""u1_deletion_gate.py — U1-E0 prereg gate + mandatory receipt scorer (CPU, numpy).

Scores a DELETION killgate npz against its matched INSERTION npz.

PREREG (frozen before the GPU run; RNG_SEED=12345):
  PRIMARY: mean within-probe Spearman of DOUBLE-CENTERED SxC (SxC[i,j] =
  resid_norm[i] * COS[i,j]) vs DOUBLE-CENTERED damage_logit, known probes
  (pre_p > 0.05). NULL: strict edit-level row-permutation (n_perm=1000).
  KILL: dc_rho < 0.15 OR perm_p >= 0.05.
  FLAG_DEGENERATE: var(S_del)/var(S_ins) < 0.1 AND the effect passes only
  without double-centering (nondc_rho >= 0.15 & nondc_p < 0.05 while dc fails)
  -> shared-refusal-token degeneracy; single pre-authorized fallback = one
  eos-variant re-run; a second FLAG or KILL is terminal.
  MANDATORY RECEIPT regardless of outcome: var(S_del)/var(S_ins) with ns/means.

STATISTICAL NOTES
  * Row-permutation COMMUTES with double-centering (column means + grand mean
    are invariant under row permutation; row means travel with rows), so
    editlevel_permutation_null is fed the ALREADY double-centered matrices.
  * NG-/S-partialling uses the corrected SYMMETRIC rank-residualization:
    midrank BOTH sides, residualize rank(pred) AND rank(D) on rank(Z), Pearson
    of residuals — per cp_edit/e5_confounds.py:125-133 (_rank_residual_on_ng),
    whose module also documents (lines ~138-145) a previously-shipped WRONG
    ASYMMETRIC residualization (transform applied to one side only). Do not
    regress to that. (cp_edit is NOT imported — it drags io/conformal deps.)
  * S-alone baseline arm: within_probe_rhos(broadcast(S), D) — quantitative
    diagnosis of S-degeneracy (if S collapses, S-alone -> NaN/0 and SxC -> ~c*COS,
    visible in the JSON, complementing the var-ratio receipt).
  * Signed within-probe Spearman ONLY. AUROC banned.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_matrices import (  # noqa: E402
    _midrank, spearman, within_probe_rhos, editlevel_permutation_null,
    edit_cluster_boot_ci, RNG_SEED,
)


def _double_center(M):
    """== cp_edit/e5_confounds.py:87-90 (reimplemented; do NOT import cp_edit)."""
    r = M.mean(axis=1, keepdims=True); c = M.mean(axis=0, keepdims=True); g = M.mean()
    return M - r - c + g


def _rank_residual(x, z):
    """Residual of midrank(x) linearly regressed on midrank(z) (centered)."""
    rx, rz = _midrank(x), _midrank(z)
    rx = rx - rx.mean(); rz = rz - rz.mean()
    beta = float(rx @ rz / (rz @ rz)) if (rz @ rz) > 0 else 0.0
    return rx - beta * rz


def partial_within_probe(PRED, D, z_row):
    """Per column j: SYMMETRIC rank-residualize PRED[:,j] and D[:,j] on z_row,
    Pearson of residuals. Returns [M] (NaN where degenerate)."""
    M = PRED.shape[1]
    out = np.full(M, np.nan)
    zfin = np.isfinite(z_row)
    for j in range(M):
        m = np.isfinite(PRED[:, j]) & np.isfinite(D[:, j]) & zfin
        if m.sum() < 5:
            continue
        rp = _rank_residual(PRED[m, j], z_row[m])
        rd = _rank_residual(D[m, j], z_row[m])
        if rp.std() == 0 or rd.std() == 0:
            continue
        out[j] = float(np.corrcoef(rp, rd)[0, 1])
    return out


def _mask(d, known, edit_ok_filter, metric):
    COS = d["COS"].astype(float)
    D = (d["damage_logit"] if metric == "logit" else d["damage_prob"]).astype(float)
    S = d["resid_norm"].astype(float) if "resid_norm" in d.files else np.full(COS.shape[0], np.nan)
    NG = d["norm_growth"].astype(float)
    row = np.ones(COS.shape[0], bool)
    if edit_ok_filter and "edit_ok" in d.files:
        row = d["edit_ok"].astype(float) > 0.5
    col = np.ones(COS.shape[1], bool)
    if known and "pre_p" in d.files:
        c = d["pre_p"].astype(float) > 0.05
        if c.sum() >= 5:
            col = c
    return COS[row][:, col], D[row][:, col], S[row], NG[row]


def _wp_summary(rhos):
    fin = np.isfinite(rhos)
    return {"mean": (None if not fin.any() else round(float(np.nanmean(rhos)), 4)),
            "nan_cols": int((~fin).sum()), "n_cols": int(rhos.size)}


def gate_stats(COS, D, S, NG, n_perm):
    """Full battery for one (filter-arm) view. SxC primary, double-centered."""
    SxC = S[:, None] * COS
    s_ok = np.isfinite(S).all() and np.nanstd(S) > 0
    # double-centered primary (row-perm commutes with DC -> feed DC'd matrices to the null)
    SxC_dc = _double_center(np.nan_to_num(SxC, nan=np.nanmean(SxC) if np.isfinite(SxC).any() else 0.0)) \
        if s_ok else None
    D_dc = _double_center(D)
    out = {}
    if s_ok:
        dc_rhos = within_probe_rhos(SxC_dc, D_dc)
        dc_rho = float(np.nanmean(dc_rhos))
        dc_p, dc_null_mean, dc_null_std = editlevel_permutation_null(
            SxC_dc, D_dc, dc_rho, n_perm=n_perm, seed=RNG_SEED)
        nondc_rhos = within_probe_rhos(SxC, D)
        nondc_rho = float(np.nanmean(nondc_rhos))
        nondc_p, _, _ = editlevel_permutation_null(SxC, D, nondc_rho, n_perm=n_perm, seed=RNG_SEED)
        cl_lo, cl_hi = edit_cluster_boot_ci(SxC_dc, D_dc)
        out.update({
            "dc_rho": round(dc_rho, 4), "dc_perm_p": round(dc_p, 4),
            "dc_null_mean": round(dc_null_mean, 4), "dc_null_std": round(dc_null_std, 4),
            "dc_nan_cols": int((~np.isfinite(dc_rhos)).sum()),
            "nondc_rho": round(nondc_rho, 4), "nondc_perm_p": round(nondc_p, 4),
            "dc_ci95_editcluster": [round(cl_lo, 4), round(cl_hi, 4)],
            "sxc_ng_partialled": _wp_summary(partial_within_probe(SxC, D, NG)),
            "sxc_s_partialled": _wp_summary(partial_within_probe(SxC, D, S)),
        })
    else:
        out.update({"dc_rho": None, "dc_perm_p": None, "nondc_rho": None, "nondc_perm_p": None,
                    "note": "S (resid_norm) unavailable/degenerate — SxC arms skipped"})
    # confound / baseline arms (reported, non-gating)
    out["raw_cos"] = _wp_summary(within_probe_rhos(COS, D))
    out["ng_baseline"] = _wp_summary(within_probe_rhos(np.repeat(NG[:, None], COS.shape[1], 1), D))
    out["s_alone"] = _wp_summary(within_probe_rhos(np.repeat(S[:, None], COS.shape[1], 1), D))
    return out


def variance_receipt(S_del, S_ins):
    fd, fi = S_del[np.isfinite(S_del)], S_ins[np.isfinite(S_ins)]
    vd = float(np.var(fd)) if fd.size else float("nan")
    vi = float(np.var(fi)) if fi.size else float("nan")
    return {"var_del": vd, "var_ins": vi,
            "var_ratio": (vd / vi if vi and np.isfinite(vd) and vi > 0 else None),
            "n_del": int(fd.size), "n_ins": int(fi.size),
            "mean_del": (float(fd.mean()) if fd.size else None),
            "mean_ins": (float(fi.mean()) if fi.size else None)}


def prereg_verdict(dc_rho, dc_p, var_ratio, nondc_rho, nondc_p):
    if dc_rho is None or not np.isfinite(dc_rho):
        return "UNDETERMINED — SxC arm unavailable"
    dc_pass = (dc_rho >= 0.15) and (dc_p is not None and dc_p < 0.05)
    nondc_pass = (nondc_rho is not None and nondc_rho >= 0.15
                  and nondc_p is not None and nondc_p < 0.05)
    degenerate = (var_ratio is not None and var_ratio < 0.1 and nondc_pass and not dc_pass)
    if degenerate:
        return "FLAG_DEGENERATE — shared-token degeneracy; single pre-authorized eos fallback"
    if not dc_pass:
        return "KILL — dc_rho < 0.15 or perm_p >= 0.05; U1 dead at this cell (no resurrection without new prereg)"
    return "PASS — authorizes ONLY U1-E1 transplant head-to-head + QuantEdit E5 deletion unlock"


def main():
    ap = argparse.ArgumentParser(description="U1-E0 deletion gate scorer")
    ap.add_argument("--del_npz", required=True)
    ap.add_argument("--ins_npz", required=True, help="matched insertion npz (variance receipt)")
    ap.add_argument("--metric", choices=["logit", "prob"], default="logit")
    ap.add_argument("--n_perm", type=int, default=1000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dd = np.load(args.del_npz)
    di = np.load(args.ins_npz)
    assert dd["COS"].shape[1] == di["COS"].shape[1], "del/ins npz probe count mismatch"
    if "edit_mode" in dd.files:
        assert str(dd["edit_mode"]) == "delete", f"--del_npz edit_mode={dd['edit_mode']}"
    if "edit_mode" in di.files:  # old gate_* npz lack the field: treated as rewrite
        assert str(di["edit_mode"]) == "rewrite", f"--ins_npz edit_mode={di['edit_mode']}"

    arms = {}
    for known in (True, False):
        for eok in (False, True):   # PRIMARY = known + edit_ok-UNfiltered
            COS, D, S, NG = _mask(dd, known, eok, args.metric)
            arms[f"known={known}|edit_ok={eok}"] = gate_stats(COS, D, S, NG, args.n_perm)

    primary = arms["known=True|edit_ok=False"]
    S_del = dd["resid_norm"].astype(float)
    S_ins = di["resid_norm"].astype(float)
    receipt = variance_receipt(S_del, S_ins)

    ptrue_pre = dd["edit_ptrue_pre"].astype(float) if "edit_ptrue_pre" in dd.files else None
    supp = float(dd["edit_ok"].astype(float).mean()) if "edit_ok" in dd.files else None
    supp_known = None
    if ptrue_pre is not None and "edit_ok" in dd.files:
        m = ptrue_pre > 0.05  # 2x criterion is noise when the base model never knew the fact
        supp_known = (float(dd["edit_ok"].astype(float)[m].mean()) if m.any() else None)

    res = {
        "del_npz": os.path.basename(args.del_npz), "ins_npz": os.path.basename(args.ins_npz),
        "metric": args.metric, "n_perm": args.n_perm, "rng_seed": RNG_SEED,
        "primary_arm": "known=True|edit_ok=False (DC SxC vs DC damage, edit-level perm null)",
        "arms": arms,
        "variance_receipt": receipt,
        "suppression_rate": supp,
        "suppression_rate_ptrue_pre_gt_0.05": supp_known,
        "delete_variant": (str(dd["delete_variant"]) if "delete_variant" in dd.files else None),
        "VERDICT": prereg_verdict(primary.get("dc_rho"), primary.get("dc_perm_p"),
                                  receipt.get("var_ratio"),
                                  primary.get("nondc_rho"), primary.get("nondc_perm_p")),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    tmp = args.out + ".tmp"
    json.dump(res, open(tmp, "w"), indent=2)
    os.replace(tmp, args.out)
    print(json.dumps({"VERDICT": res["VERDICT"], "primary": primary,
                      "variance_receipt": receipt}, indent=2))
    print(f"[u1-gate] wrote {args.out}")


if __name__ == "__main__":
    main()
