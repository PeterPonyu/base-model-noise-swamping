"""analyze_sequential.py — the SEQUENTIAL (no-restore) arm analyzer.

Reads killgate `--no_restore --recheck_every K --save_matrices` .npz files and
adjudicates the sequential-editing claims that killgate itself defers here (its
SEQ_MODE json literally says "gate lives in experiments/.../analyze_sequential.py").

WHAT THE npz HOLDS (grounded in killgate_keygeom.py savez, no_restore branch):
  recheck_at   [C]      edit-count checkpoints (i+1), e.g. [10,20,30,40,50]
  prior_eff    [C,N]    prior_eff[c,jj] = efficacy.success (1/0) of edit jj re-run at
                        checkpoint c; NaN for jj >= recheck_at[c] (not yet applied).
                        The FINAL row (last checkpoint) has every edit present.
  prior_pnew   [C,N]    P(target_new) for edit jj at checkpoint c (continuous survival)
  prior_ptrue  [C,N]    P(target_true) for edit jj at checkpoint c
  GRAM_pre     [N,N]    pre-SEQUENCE key cosines among the N edit keys (diag 1). This is
                        the edit<->edit geometry; GRAM_pre[jj,s] = cos(k_jj, k_s).
  resid_norm   [N]      per-edit ROME edit strength S = ||v - W k|| (the S factor)
  key_norm     [N]      ||k_edit|| (mechanical / H2 null lever)
  edit_ok      [N]      immediate post-edit efficacy (did edit jj install at all)
  norm_growth  [N]      ||ΔW|| per edit

THE THREE CLAIMS (each stated with its threshold BEFORE the numbers are read):
  (A) COLLAPSE      — fraction of installed edits still successful decays to ~10-14% @50.
                      Descriptive survival curve, per stream.
  (B) POSITION-FRAGILITY — later-applied edits survive more (fewer subsequent overwriters):
                      Spearman(stream position, survival) with an edit-level permutation null.
  (C) H1 GEOMETRY ATTRIBUTION — does edit<->edit key geometry explain WHICH edits are
                      overwritten BEYOND mere stream position? For a prior edit jj the
                      overwrite EXPOSURE is its coupling to the edits applied AFTER it:
                        C_fwd[jj]  = sum_{s>jj} |GRAM_pre[jj,s]|            (key-cosine only)
                        SC_fwd[jj] = sum_{s>jj} S[s]*|GRAM_pre[jj,s]|       (the S×C mechanism)
                      Both C_fwd and SC_fwd are MECHANICALLY confounded with position: an
                      early edit simply has more subsequent terms to sum. H1 therefore lives
                      in the POSITION-PARTIALLED Spearman, with a Freedman-Lane residual
                      permutation null, per stream AND pooled. The reverse partial (position
                      | geometry) is reported so the confound direction is explicit.

House rules honored (see analyze_matrices.py): SIGNED Spearman with tie-averaged ranks
(NEVER AUROC); edit is the exchangeable unit for every null; thresholds are pre-registered
in `verdict()` and NOT tuned to the data.

Outcome sign convention: the H1 statistic uses OVERWRITTEN = 1 - survived, so that "more
geometry exposure -> more damage" reads as a POSITIVE rho (matching the collateral-damage
framing and the verdict's rho>0 gate). Survival-framed numbers are also reported.

Usage:
  python analyze_sequential.py results/matrices/seq_llama1b_nr_L12_s0.npz \
      results/matrices/seq_llama1b_nr_L12_s1.npz --out results/SEQ_analysis_L12.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

RNG_SEED = 12345  # fixed so every permutation null is reproducible (matches analyze_matrices)


# ----------------------------------------------------------------------------- ranks / Spearman
def _midrank(x):
    """Tie-averaged ranks (textbook-correct Spearman ranks). Binary survival outcomes
    are almost all ties, so average-rank is mandatory here, not cosmetic."""
    x = np.asarray(x, float)
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


def _residualize(y_rank, z_rank):
    """Least-squares residual of y_rank on [1, z_rank] (rank-space partialling)."""
    Z = np.column_stack([np.ones_like(z_rank), z_rank])
    beta, *_ = np.linalg.lstsq(Z, y_rank, rcond=None)
    return y_rank - Z @ beta


def partial_spearman(x, y, z):
    """Signed partial Spearman rho of x,y controlling z: correlate the rank-space
    residuals of x-on-z and y-on-z. Returns (rho, rx_res, ry_res) so the caller can
    reuse the residuals for a Freedman-Lane permutation null."""
    x, y, z = np.asarray(x, float), np.asarray(y, float), np.asarray(z, float)
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[m], y[m], z[m]
    if x.size < 4:
        return np.nan, None, None
    rx, ry, rz = _midrank(x), _midrank(y), _midrank(z)
    if rx.std() == 0 or ry.std() == 0 or rz.std() == 0:
        return np.nan, None, None
    rx_res, ry_res = _residualize(rx, rz), _residualize(ry, rz)
    if rx_res.std() == 0 or ry_res.std() == 0:
        return np.nan, None, None
    return float(np.corrcoef(rx_res, ry_res)[0, 1]), rx_res, ry_res


# ----------------------------------------------------------------------------- permutation nulls
def perm_null_spearman(x, y, obs, n_perm=2000, seed=RNG_SEED):
    """Edit-level null for a raw Spearman: the edit (a vector entry) is the exchangeable
    unit, so permute y across edits and recompute. Returns (p, null_mean, null_std)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 4 or not np.isfinite(obs):
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    nulls = np.empty(n_perm)
    ge = 0
    for t in range(n_perm):
        nulls[t] = spearman(x, rng.permutation(y))
        if abs(nulls[t]) >= abs(obs):
            ge += 1
    return (ge + 1) / (n_perm + 1), float(np.nanmean(nulls)), float(np.nanstd(nulls))


def perm_null_partial(rx_res, ry_res, obs, n_perm=2000, seed=RNG_SEED):
    """Freedman-Lane residual permutation null for a partial correlation: with the
    confound already residualized out of both variables, the y-residuals are exchangeable
    under H0 (x ⟂ y | z). Permute ry_res, recompute Pearson corr with fixed rx_res."""
    if rx_res is None or ry_res is None or not np.isfinite(obs):
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    nulls = np.empty(n_perm)
    ge = 0
    for t in range(n_perm):
        nulls[t] = float(np.corrcoef(rx_res, rng.permutation(ry_res))[0, 1])
        if abs(nulls[t]) >= abs(obs):
            ge += 1
    return (ge + 1) / (n_perm + 1), float(np.nanmean(nulls)), float(np.nanstd(nulls))


# ----------------------------------------------------------------------------- geometry from GRAM
def forward_exposure(GRAM, S):
    """Per-edit overwrite exposure to SUBSEQUENT edits (the edits that can overwrite it).
    Returns dict of [N] vectors:
      C_fwd_sum   = sum_{s>jj} |G[jj,s]|                 (pure key-cosine coupling)
      SC_fwd_sum  = sum_{s>jj} S[s]*|G[jj,s]|            (S×C: strength-weighted coupling)
      *_mean      = the same divided by the # of subsequent edits (count-normalized; the
                    marginal-robust variant, see double_centering note). NaN for the last
                    edit (no subsequent edits).
      n_sub       = # subsequent edits = N-1-jj (== the raw stream-position confound)
    """
    N = GRAM.shape[0]
    A = np.abs(GRAM).astype(float)
    Ssafe = np.nan_to_num(np.asarray(S, float), nan=0.0)
    C_sum = np.zeros(N); SC_sum = np.zeros(N); n_sub = np.zeros(N)
    for jj in range(N):
        sl = slice(jj + 1, N)
        C_sum[jj] = A[jj, sl].sum()
        SC_sum[jj] = (Ssafe[sl] * A[jj, sl]).sum()
        n_sub[jj] = N - 1 - jj
    with np.errstate(invalid="ignore", divide="ignore"):
        C_mean = np.where(n_sub > 0, C_sum / n_sub, np.nan)
        SC_mean = np.where(n_sub > 0, SC_sum / n_sub, np.nan)
    return {"C_fwd_sum": C_sum, "SC_fwd_sum": SC_sum,
            "C_fwd_mean": C_mean, "SC_fwd_mean": SC_mean, "n_sub": n_sub}


# ----------------------------------------------------------------------------- per-stream analysis
def _r4(x):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), 4)


def survival_curve(pe, recheck_at):
    """(A) Fraction of already-applied edits still successful at each checkpoint.
    At checkpoint c only edits jj < recheck_at[c] are present (rest NaN)."""
    curve = []
    for c in range(pe.shape[0]):
        row = pe[c]
        present = np.isfinite(row)
        n_present = int(present.sum())
        n_succ = float(np.nansum(row))
        curve.append({
            "checkpoint_nedits": int(recheck_at[c]),
            "n_present": n_present,
            "n_survived": int(round(n_succ)),
            "frac_survived": _r4(n_succ / n_present if n_present else np.nan),
        })
    return curve


def stream_h1(geom_vec, overwritten, position, n_perm, seed, label):
    """Raw + position-partialled Spearman(geometry, overwritten), with nulls, plus the
    reverse partial (position | geometry). `overwritten` = 1 - survived so rho>0 means
    'more geometry exposure -> more damage'."""
    raw = spearman(geom_vec, overwritten)
    raw_p, raw_nm, raw_ns = perm_null_spearman(geom_vec, overwritten, raw, n_perm, seed)
    par, rg_res, ry_res = partial_spearman(geom_vec, overwritten, position)
    par_p, par_nm, par_ns = perm_null_partial(rg_res, ry_res, par, n_perm, seed)
    # reverse partial: does position still predict overwrite once geometry is held fixed?
    rev, rp_res, ryr_res = partial_spearman(position, overwritten, geom_vec)
    rev_p, _, _ = perm_null_partial(rp_res, ryr_res, rev, n_perm, seed)
    return {
        "label": label,
        "raw_rho": _r4(raw), "raw_perm_p": _r4(raw_p),
        "raw_null_mean": _r4(raw_nm), "raw_null_std": _r4(raw_ns),
        "partial_rho_ctrl_position": _r4(par), "partial_perm_p": _r4(par_p),
        "partial_null_mean": _r4(par_nm), "partial_null_std": _r4(par_ns),
        "reverse_partial_position_ctrl_geometry": _r4(rev), "reverse_partial_perm_p": _r4(rev_p),
    }


def analyze_stream(npz_path, n_perm, seed, edit_ok_filter):
    d = np.load(npz_path, allow_pickle=True)
    for req in ("prior_eff", "recheck_at", "GRAM_pre", "resid_norm"):
        if req not in d.files:
            raise SystemExit(f"[seq] {os.path.basename(npz_path)} missing '{req}' — not a "
                             f"no_restore+recheck npz; cannot run H1.")
    pe = d["prior_eff"].astype(float)              # [C,N]
    pn = d["prior_pnew"].astype(float) if "prior_pnew" in d.files else None
    recheck_at = d["recheck_at"].astype(int)       # [C]
    GRAM = d["GRAM_pre"].astype(float)             # [N,N]
    S = d["resid_norm"].astype(float)              # [N]  ROME edit strength
    edit_ok = d["edit_ok"].astype(float) if "edit_ok" in d.files else None
    N = pe.shape[1]

    # ---- (A) collapse / survival curve ----
    curve = survival_curve(pe, recheck_at)

    # ---- final-checkpoint per-edit outcome (all N edits present) ----
    final = pe[-1]                                  # [N] survival at last checkpoint
    survived = final.astype(float)
    overwritten = 1.0 - survived                    # H1 outcome (rho>0 == more damage)
    position = np.arange(N, dtype=float)            # stream order (0 = first applied)
    pnew_final = pn[-1].astype(float) if pn is not None else None  # continuous survival proxy

    # ---- edit-ok mask (only edits that actually installed can be "overwritten") ----
    if edit_ok_filter and edit_ok is not None:
        keep = edit_ok > 0.5
    else:
        keep = np.ones(N, dtype=bool)

    # ---- (B) position-fragility: Spearman(position, survival) ----
    pf_rho = spearman(position[keep], survived[keep])
    pf_p, pf_nm, pf_ns = perm_null_spearman(position[keep], survived[keep], pf_rho, n_perm, seed)

    # ---- geometry (C) ----
    geo = forward_exposure(GRAM, S)

    def h1_block(geom_key, outcome, outcome_name):
        return {
            "outcome": outcome_name,
            "n_edits": int(keep.sum()),
            "SC_fwd_sum": stream_h1(geo[geom_key][keep], outcome[keep], position[keep],
                                    n_perm, seed, "S×C forward exposure (sum)"),
            "C_fwd_sum": stream_h1(geo["C_fwd_sum"][keep], outcome[keep], position[keep],
                                   n_perm, seed, "key-cosine forward coupling (sum)"),
            # double-centering analog (see note): count-normalized geometry drops the
            # position-count marginal that inflates the sum-based raw rho. NaN on the last
            # edit (n_sub=0) is dropped by the finite-mask inside the stats.
            "SC_fwd_mean_countnorm": stream_h1(geo["SC_fwd_mean"][keep], outcome[keep],
                                               position[keep], n_perm, seed,
                                               "S×C forward exposure (count-normalized)"),
        }

    h1 = {"binary_overwritten": h1_block("SC_fwd_sum", overwritten,
                                         "overwritten = 1 - survived (binary)")}
    if pnew_final is not None:
        # continuous outcome: lower P(new) at the end == more overwritten -> use (1 - p_new)
        cont = 1.0 - pnew_final
        h1["continuous_1_minus_pnew"] = h1_block("SC_fwd_sum", cont,
                                                 "1 - P(target_new) at final checkpoint")

    return {
        "npz": os.path.basename(npz_path),
        "n_edits": int(N),
        "recheck_at": [int(x) for x in recheck_at],
        "edit_ok_filter": bool(edit_ok_filter),
        "edit_success_rate": _r4(float(edit_ok.mean())) if edit_ok is not None else None,
        "survival_curve": curve,
        "final_survival_frac": _r4(float(np.nanmean(survived[keep]))),
        "position_fragility": {
            "rho_position_vs_survival": _r4(pf_rho),
            "perm_p": _r4(pf_p), "null_mean": _r4(pf_nm), "null_std": _r4(pf_ns),
            "note": "rho>0 == later-applied edits survive more (fewer subsequent overwriters)",
        },
        "H1_geometry": h1,
        # carry the per-edit vectors so the pooled analysis can concatenate streams
        "_vecs": {"survived": survived, "overwritten": overwritten, "position": position,
                  "pnew_final": pnew_final, "keep": keep,
                  "SC_fwd_sum": geo["SC_fwd_sum"], "C_fwd_sum": geo["C_fwd_sum"],
                  "SC_fwd_mean": geo["SC_fwd_mean"]},
    }


def analyze_pooled(streams, n_perm, seed):
    """Pool the per-edit (geometry, position, outcome) triples across streams — each edit
    is an independent unit; the two streams are different orderings, so this doubles n
    without reusing any pair. Position is per-stream 0..N-1 (ties across streams handled by
    midrank in the partial)."""
    def cat(key):
        return np.concatenate([s["_vecs"][key] for s in streams])
    keep = cat("keep").astype(bool)
    position = cat("position"); survived = cat("survived"); overwritten = cat("overwritten")
    SC = cat("SC_fwd_sum"); C = cat("C_fwd_sum"); SCm = cat("SC_fwd_mean")

    pf_rho = spearman(position[keep], survived[keep])
    pf_p, pf_nm, pf_ns = perm_null_spearman(position[keep], survived[keep], pf_rho, n_perm, seed)
    return {
        "n_streams": len(streams),
        "n_edits_pooled": int(keep.sum()),
        "final_survival_frac": _r4(float(np.nanmean(survived[keep]))),
        "position_fragility": {
            "rho_position_vs_survival": _r4(pf_rho),
            "perm_p": _r4(pf_p), "null_mean": _r4(pf_nm), "null_std": _r4(pf_ns),
        },
        "H1_geometry": {
            "outcome": "overwritten = 1 - survived (binary)",
            "n_edits": int(keep.sum()),
            "SC_fwd_sum": stream_h1(SC[keep], overwritten[keep], position[keep],
                                    n_perm, seed, "S×C forward exposure (sum)"),
            "C_fwd_sum": stream_h1(C[keep], overwritten[keep], position[keep],
                                   n_perm, seed, "key-cosine forward coupling (sum)"),
            "SC_fwd_mean_countnorm": stream_h1(SCm[keep], overwritten[keep], position[keep],
                                               n_perm, seed, "S×C forward exposure (count-normalized)"),
        },
    }


# ----------------------------------------------------------------------------- pre-registered verdict
def verdict(per_stream, pooled):
    """Thresholds fixed BEFORE the numbers are read (do NOT tune to data):

      H1 SUPPORTED  iff the position-partialled S×C rho is > 0 AND perm-p < 0.05 in BOTH
                    streams (the primary binary-overwritten outcome, SC_fwd_sum block).
      H1 REFUTED    iff the partialled rho is <= 0 in both streams (sign against the
                    hypothesis) — geometry adds nothing beyond position.
      H1 UNSETTLED  otherwise (mixed sign, or right sign but not both significant) — the
                    r4 informal read. A sequential geometry-attribution claim MUST NOT enter
                    the paper under UNSETTLED or REFUTED.
    """
    def part(s):
        b = s["H1_geometry"]["binary_overwritten"]["SC_fwd_sum"]
        return b["partial_rho_ctrl_position"], b["partial_perm_p"]
    rhos, ps = zip(*[part(s) for s in per_stream])
    both_pos_sig = all((r is not None and r > 0) and (p is not None and p < 0.05)
                       for r, p in zip(rhos, ps))
    both_nonpos = all((r is not None and r <= 0) for r in rhos)
    if both_pos_sig:
        status = "SUPPORTED"
    elif both_nonpos:
        status = "REFUTED"
    else:
        status = "UNSETTLED"
    pooled_b = pooled["H1_geometry"]["SC_fwd_sum"]
    return {
        "criterion": "position-partialled S×C rho > 0 AND perm-p < 0.05 in BOTH streams",
        "per_stream_partial_rho": [r for r in rhos],
        "per_stream_partial_perm_p": [p for p in ps],
        "pooled_partial_rho": pooled_b["partial_rho_ctrl_position"],
        "pooled_partial_perm_p": pooled_b["partial_perm_p"],
        "H1_STATUS": status,
        "paper_guidance": ("A sequential geometry-attribution claim is admissible ONLY under "
                           "SUPPORTED. Current status governs what may be written."),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz", nargs="+",
                    help="one or more no_restore+recheck killgate .npz (globs ok); each = one stream")
    ap.add_argument("--edit_ok", action="store_true",
                    help="restrict H1 to edits that actually installed (edit_ok>0.5); only an "
                         "installed edit can be 'overwritten'")
    ap.add_argument("--n_perm", type=int, default=2000,
                    help="permutations for every edit-level / residual null (default 2000)")
    ap.add_argument("--seed", type=int, default=RNG_SEED)
    ap.add_argument("--out", default="results/SEQ_analysis_L12.json")
    args = ap.parse_args()

    paths = sorted({p for pat in args.npz for p in glob.glob(pat)})
    if not paths:
        raise SystemExit("no .npz matched")

    per_stream = [analyze_stream(p, args.n_perm, args.seed, args.edit_ok) for p in paths]
    pooled = analyze_pooled(per_stream, args.n_perm, args.seed)
    vd = verdict(per_stream, pooled)

    # strip the internal vectors before serializing
    clean = []
    for s in per_stream:
        s = dict(s); s.pop("_vecs", None); clean.append(s)

    res = {
        "double_centering_note": (
            "analyze_matrices.py double-centering removes row+column marginals from the "
            "[edit×probe] COS/damage matrix. Here the H1 outcome is a PER-EDIT scalar "
            "(one final-survival value per edit, aggregated over all subsequent overwriters), "
            "so there is no 2-D (edit,probe) pairing to double-center. The faithful analog — "
            "removing the marginal (subsequent-edit COUNT == stream position) that inflates the "
            "sum-based raw rho — is (i) the position-partialled Spearman [primary] and "
            "(ii) the count-normalized geometry variant 'SC_fwd_mean_countnorm'."),
        "geometry_definition": (
            "For prior edit jj: overwrite exposure to SUBSEQUENT edits. "
            "C_fwd_sum=Σ_{s>jj}|GRAM_pre[jj,s]|; SC_fwd_sum=Σ_{s>jj} resid_norm[s]·|GRAM_pre[jj,s]|. "
            "Outcome overwritten=1-survived so rho>0 == geometry predicts damage."),
        "n_perm": args.n_perm,
        "seed": args.seed,
        "edit_ok_filter": bool(args.edit_ok),
        "per_stream": clean,
        "pooled": pooled,
        "verdict": vd,
    }
    print(json.dumps(res, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(res, open(args.out, "w"), indent=2)
        print(f"[seq] wrote {args.out}")


if __name__ == "__main__":
    main()
