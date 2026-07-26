"""analyze_aniso.py — anisotropy / whitening profile of ROME edit-key banks.

=============================================================================
AUTHORING PASS. A separate hostile review gates any paper claim built on this.
Nothing here is a headline; it feeds the Discussion ("why the L14 regime
transition / why gemma is geometry-blind") and must be read as descriptive.
=============================================================================

WHAT THIS READS
  The rank-one `--save_vectors` banks produced by experiments/killgate_keygeom.py
  (schema grounded below). Each npz carries, among other fields:
    K       [N, d_in]  float32  base-model EDIT keys: the down_proj INPUT at the
                                 edit layer L, captured at the subject's LAST
                                 token position (the fact-retrieval key), RAW /
                                 un-normalised. d_in = intermediate_size.
    knorm   [N]        float32  row norms of K (== ||k|| per edit).
    layer, model, editor, dataset, seed, n_edits, vectors_valid  (provenance).
  (killgate_keygeom.py:363-376 defines the key; :814/:823 save K/knorm.)

  These banks do NOT contain probe keys, a probe cosine matrix, or any collateral
  -damage matrix. K_probe IS computed in killgate (line 376) but is deliberately
  not persisted, and GRAM_pre is stored only in SEQ mode. Consequences, enforced
  as hard scope limits below:
    * NO matched-probe-pair arm (the original task's item-1 probe arm) — the data
      to compute it does not exist in any current bank. Recorded as unavailable.
    * This analysis describes KEY-SPACE geometry ONLY. It cannot be joined to
      collateral damage here, so it cannot by itself explain the damage regime
      transition — it can only characterise the key distribution that a damage
      analysis would then have to reference.

WHAT IT COMPUTES, per bank (edit keys only):
  1. Anisotropy profile of K: participation ratio (uncentered second-moment AND
     centered covariance), spectral decay (cumulative-variance thresholds +
     log-log slope), and mean/std of pairwise edit-key cosine — each against an
     isotropic-Gaussian (norm-matched) baseline and a column-permutation baseline.
  2. Whitening test: shrinkage-regularised whitening inside the POPULATED PCA
     subspace (n<d makes full d-space whitening ill-posed — see constraints),
     recomputing the edit-edit cosine structure to see whether it sharpens or
     dissolves. Reported across a shrinkage grid so the ill-posedness is visible.
  3. Cross-model contrast: pass two npz paths; the two profiles are emitted side
     by side, descriptively, with the permutation/Gaussian bands as uncertainty.

  Emits ONE JSON (--out) with every number, per-array provenance, and an honest
  `interpretation_constraints` block listing what this CANNOT distinguish.

Conventions match experiments/analyze_matrices.py: fixed RNG seed, permutation
p-values as (ge+1)/(n_perm+1), 4-dp rounding, numpy/scipy only, no AUROC.

Usage:
  python experiments/analyze_aniso.py BANK.npz [BANK2.npz] --out results/ANISO.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

RNG_SEED = 12345  # fixed so every baseline / permutation band is reproducible


# --------------------------------------------------------------------------- IO
def load_bank(npz_path):
    """Load an edit-key bank; return (K[float64 N,d], meta dict). Fails loudly if
    the required raw-key array K is absent — no proxy is substituted."""
    d = np.load(npz_path, allow_pickle=False)
    if "K" not in d.files:
        raise SystemExit(
            f"[aniso] {npz_path}: no raw key matrix 'K' in this npz "
            f"(fields={list(d.files)}). This analysis requires raw edit keys; "
            f"scalar-only dumps (e.g. results/aniso_*_L14) are NOT usable — no proxy.")
    K = np.asarray(d["K"], dtype=np.float64)
    if K.ndim != 2:
        raise SystemExit(f"[aniso] {npz_path}: K has ndim={K.ndim}, expected 2 [N,d].")

    def _scalar(name, default=None):
        return d[name].item() if name in d.files else default

    meta = {
        "npz": npz_path,
        "model": _scalar("model", "unknown"),
        "editor": _scalar("editor", "unknown"),
        "dataset": _scalar("dataset", "unknown"),
        "layer": (int(d["layer"].item()) if "layer" in d.files else None),
        "seed": (int(d["seed"].item()) if "seed" in d.files else None),
        "N": int(K.shape[0]),
        "d_in": int(K.shape[1]),
        "vectors_valid": (int(d["vectors_valid"].item()) if "vectors_valid" in d.files else None),
        "n_nonfinite_rows": int((~np.isfinite(K).all(axis=1)).sum()),
    }
    finite = np.isfinite(K).all(axis=1)
    if not finite.all():
        K = K[finite]
        meta["N_after_finite_filter"] = int(K.shape[0])
    return K, meta


# ------------------------------------------------------------------- primitives
def _gram_eigs(X):
    """Non-negative eigenvalues of the N x N Gram X X^T, descending. These are the
    NONZERO eigenvalues of the d x d (second-moment or covariance) matrix — same
    spectrum, cheaper for n << d. Only min(N-1?, N) are meaningfully populated."""
    G = X @ X.T
    w = np.linalg.eigvalsh(G)          # ascending, may carry tiny negatives (numerical)
    w = np.clip(w[::-1], 0.0, None)    # descending, floored at 0
    return w


def participation_ratio(eigs):
    """PR = (sum lambda)^2 / sum(lambda^2). Effective # of directions carrying the
    variance/energy. Scale-invariant. Low => anisotropic (few directions dominate)."""
    s1 = float(eigs.sum())
    s2 = float((eigs ** 2).sum())
    return (s1 * s1 / s2) if s2 > 0 else float("nan")


def spectral_summary(eigs):
    """Cumulative-variance thresholds + log-log decay slope of the eigenspectrum."""
    tot = float(eigs.sum())
    out = {"top1_frac": float(eigs[0] / tot) if tot > 0 else float("nan"),
           "top5_frac": float(eigs[:5].sum() / tot) if tot > 0 else float("nan")}
    if tot > 0:
        cum = np.cumsum(eigs) / tot
        for thr in (0.5, 0.9, 0.99):
            out[f"n_comp_{int(thr*100)}pct"] = int(np.searchsorted(cum, thr) + 1)
    # log-log slope over the populated (positive) eigenvalues
    pos = eigs[eigs > eigs[0] * 1e-12]
    if pos.size >= 3:
        x = np.log(np.arange(1, pos.size + 1)); y = np.log(pos)
        out["loglog_slope"] = float(np.polyfit(x, y, 1)[0])
        out["n_positive_eigs"] = int(pos.size)
    else:
        out["loglog_slope"] = float("nan"); out["n_positive_eigs"] = int(pos.size)
    return out


def pairwise_cos_stats(X):
    """Mean / std of the off-diagonal (i<j) cosine among the rows of X. High mean =>
    keys concentrated in a cone (directional anisotropy); std = discriminability."""
    n = X.shape[0]
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    C = Xn @ Xn.T
    iu = np.triu_indices(n, k=1)
    off = C[iu]
    return {"mean_cos": float(off.mean()), "std_cos": float(off.std()),
            "mean_abs_cos": float(np.abs(off).mean()),
            "n_pairs": int(off.size)}


# ----------------------------------------------------------------------- nulls
def isotropic_gaussian_band(K, n_rep, rng):
    """Norm-matched isotropic null: draw N iid N(0,I_d) rows, rescale each to a
    (shuffled) empirical key norm. This is the 'no anisotropy at all, finite N,d'
    floor for PR and pairwise cosine. Implemented as a random rotation of an
    isotropic cloud, i.e. the rotation-style baseline the task asked for."""
    N, d = K.shape
    norms = np.linalg.norm(K, axis=1)
    prs, mcs = [], []
    for _ in range(n_rep):
        G = rng.standard_normal((N, d))
        G *= (rng.permutation(norms) / (np.linalg.norm(G, axis=1) + 1e-12))[:, None]
        prs.append(participation_ratio(_gram_eigs(G - G.mean(0))))
        mcs.append(pairwise_cos_stats(G)["mean_cos"])
    return _band(prs), _band(mcs)


def column_permutation_band(K, n_rep, rng):
    """Column-permutation null: independently shuffle each feature column across
    rows. Destroys cross-feature (off-axis / rotational) covariance while PRESERVING
    every column's marginal AND the mean vector. So: for PR it isolates genuinely
    rotational anisotropy from mere per-neuron variance heterogeneity; for mean
    cosine it is NOT informative (the mean direction survives) and is reported only
    for completeness. Centered PR is used (mean removed)."""
    N, d = K.shape
    prs, mcs = [], []
    for _ in range(n_rep):
        P = np.empty_like(K)
        for j in range(d):
            P[:, j] = K[rng.permutation(N), j]
        prs.append(participation_ratio(_gram_eigs(P - P.mean(0))))
        mcs.append(pairwise_cos_stats(P)["mean_cos"])
    return _band(prs), _band(mcs)


def _band(vals):
    a = np.asarray(vals, float)
    return {"mean": round(float(a.mean()), 4), "std": round(float(a.std()), 4),
            "p05": round(float(np.percentile(a, 5)), 4),
            "p95": round(float(np.percentile(a, 95)), 4),
            "n_rep": int(a.size)}


# ------------------------------------------------------------------- whitening
def whiten_subspace(K, var_keep, shrink_grid):
    """Shrinkage-regularised whitening INSIDE the populated PCA subspace.

    Rationale (see interpretation_constraints): with n<<d the d x d covariance has
    rank <= N-1, so ~ (d-N+1) directions have zero empirical variance and full
    d-space whitening is ill-posed (it would just divide by the shrinkage prior in
    unconstrained directions). We therefore whiten only within the top-r data
    subspace (r chosen by cumulative variance `var_keep`), scaling each retained
    principal component by 1/sqrt(var_j + shrink). `shrink = alpha * mean(var_kept)`.
    Everything lives in the N x r score space (no d x d matrix is formed).

    Returns, per alpha in `shrink_grid`, the pairwise-cosine stats of the whitened
    keys, so the shrinkage dependence is visible rather than hidden."""
    Xc = K - K.mean(0)
    # eigh of the centered Gram gives PC directions (P) and eigenvalues (=S^2).
    G = Xc @ Xc.T
    w, P = np.linalg.eigh(G)           # ascending
    w = w[::-1]; P = P[:, ::-1]
    w = np.clip(w, 0.0, None)
    scores = P * np.sqrt(w)[None, :]   # [N, N] projections onto principal axes (P@Sigma)
    var = w / max(1, (K.shape[0] - 1))  # per-component variance
    tot = var.sum()
    r = int(np.searchsorted(np.cumsum(var) / tot, var_keep) + 1) if tot > 0 else 1
    r = int(min(r, np.count_nonzero(var > var[0] * 1e-12)))
    r = max(r, 1)
    Sc = scores[:, :r]; var_r = var[:r]
    out = {"rank_kept": r, "var_kept_frac": round(float(var[:r].sum() / tot), 4),
           "by_shrinkage": {}}
    for alpha in shrink_grid:
        shrink = float(alpha) * float(var_r.mean())
        W = Sc / np.sqrt(var_r + shrink)[None, :]   # whitened scores
        st = pairwise_cos_stats(W)
        out["by_shrinkage"][f"alpha_{alpha}"] = {
            "shrink_abs": round(shrink, 6),
            "mean_cos": round(st["mean_cos"], 4),
            "std_cos": round(st["std_cos"], 4),
            "mean_abs_cos": round(st["mean_abs_cos"], 4)}
    return out


# ---------------------------------------------------------------- per-bank driver
def analyze_one(npz_path, args, rng):
    K, meta = load_bank(npz_path)
    N = K.shape[0]

    eig_unc = _gram_eigs(K)                 # uncentered second-moment spectrum
    eig_cen = _gram_eigs(K - K.mean(0))     # centered covariance spectrum
    pr_unc = participation_ratio(eig_unc)
    pr_cen = participation_ratio(eig_cen)
    cos = pairwise_cos_stats(K)

    iso_pr, iso_mc = isotropic_gaussian_band(K, args.n_rep, rng)
    col_pr, col_mc = column_permutation_band(K, args.n_rep, rng)

    whiten = whiten_subspace(K, args.var_keep, args.shrink_grid)

    return {
        "provenance": meta,
        "key_norm": {"mean": round(float(np.linalg.norm(K, axis=1).mean()), 4),
                     "std": round(float(np.linalg.norm(K, axis=1).std()), 4)},
        "anisotropy": {
            "participation_ratio_uncentered": round(pr_unc, 4),
            "participation_ratio_centered": round(pr_cen, 4),
            "pr_uncentered_frac_of_N": round(pr_unc / N, 4),
            "pr_centered_frac_of_Nminus1": round(pr_cen / max(1, N - 1), 4),
            "spectrum_uncentered": {k: (round(v, 4) if isinstance(v, float) else v)
                                    for k, v in spectral_summary(eig_unc).items()},
            "spectrum_centered": {k: (round(v, 4) if isinstance(v, float) else v)
                                  for k, v in spectral_summary(eig_cen).items()},
            "mean_pairwise_cos": round(cos["mean_cos"], 4),
            "std_pairwise_cos": round(cos["std_cos"], 4),
            "mean_abs_pairwise_cos": round(cos["mean_abs_cos"], 4),
            "n_pairs": cos["n_pairs"],
            "baselines": {
                "isotropic_gaussian_norm_matched": {
                    "pr_centered": iso_pr, "mean_cos": iso_mc,
                    "note": "no-anisotropy floor (finite N,d). Empirical PR << this "
                            "band and mean_cos >> this band => real anisotropy."},
                "column_permutation": {
                    "pr_centered": col_pr, "mean_cos_UNINFORMATIVE": col_mc,
                    "note": "destroys off-axis covariance, PRESERVES per-neuron "
                            "variance + mean vector; PR gap vs empirical = rotational "
                            "(non-axis-aligned) anisotropy. mean_cos here is not a "
                            "valid null (mean direction survives the shuffle)."}}},
        "whitening_subspace": whiten,
    }


def cross_model_contrast(profiles):
    """Descriptive side-by-side of two banks. No story is fit — just the deltas a
    reviewer would compute, with the baseline bands left in for uncertainty."""
    a, b = profiles
    ta = f'{a["provenance"]["model"]}@L{a["provenance"]["layer"]}'
    tb = f'{b["provenance"]["model"]}@L{b["provenance"]["layer"]}'

    def _wa(p):  # whitened mean_cos at the middle shrinkage alpha
        bs = p["whitening_subspace"]["by_shrinkage"]
        mid = sorted(bs.keys())[len(bs) // 2]
        return bs[mid]["mean_cos"], mid

    wa_a, ka = _wa(a); wa_b, kb = _wa(b)
    return {
        "pair": [ta, tb],
        "participation_ratio_uncentered": {ta: a["anisotropy"]["participation_ratio_uncentered"],
                                            tb: b["anisotropy"]["participation_ratio_uncentered"]},
        "mean_pairwise_cos": {ta: a["anisotropy"]["mean_pairwise_cos"],
                              tb: b["anisotropy"]["mean_pairwise_cos"]},
        "std_pairwise_cos": {ta: a["anisotropy"]["std_pairwise_cos"],
                             tb: b["anisotropy"]["std_pairwise_cos"]},
        "top1_frac_uncentered": {ta: a["anisotropy"]["spectrum_uncentered"]["top1_frac"],
                                 tb: b["anisotropy"]["spectrum_uncentered"]["top1_frac"]},
        "whitened_mean_cos": {"alpha_used": [ka, kb], ta: wa_a, tb: wa_b},
        "note": "Descriptive only. Whether a PR/cosine gap 'explains' the damage "
                "regime difference cannot be decided here (no damage matrices in "
                "these banks). Do NOT over-read a single-seed two-point contrast.",
    }


def build_interpretation_constraints(profiles):
    """Build the interpretation_constraints block from the banks actually loaded.

    PENDING REVIEW (authoring pass 2026-07-04): this replaces a STATIC block whose
    lines were written before the L14 / Qwen banks existed and had gone stale (they
    claimed "no Qwen raw-key bank exists", "cross-model L14-pending", "L8-L12
    code-path only" — directly contradicting a run whose cross_model_contrast is
    populated). The block is now derived from `profiles` so it can never again
    contradict the file's own contents. The still-valid, bank-independent limits
    (no damage join, key-space only, whitening ill-posedness, per-bank single-seed,
    column-permutation, sample-confound) are preserved.
    """
    def _short(m):
        return str(m).rstrip("/").split("/")[-1]

    covered = [f'{_short(p["provenance"]["model"])}@L{p["provenance"]["layer"]}'
               f's{p["provenance"]["seed"]}' for p in profiles]
    seeds = sorted({p["provenance"]["seed"] for p in profiles})
    Ns = sorted({p["provenance"]["N"] for p in profiles})
    ds = sorted({p["provenance"]["d_in"] for p in profiles})
    n_txt = str(Ns[0]) if len(Ns) == 1 else f"{Ns[0]}-{Ns[-1]}"
    d_txt = str(ds[0]) if len(ds) == 1 else f"{ds[0]}-{ds[-1]}"

    cons = [
        f"COVERAGE: this run profiles {len(profiles)} bank(s): {', '.join(covered)}. "
        "All statements below are properties of exactly these banks.",
    ]
    if len(profiles) == 2:
        a, b = covered
        cons.append(
            f"CROSS-MODEL contrast ({a} vs {b}) is DESCRIPTIVE only and single-seed "
            "per model: read RELATIVE deltas, not absolute values, and do not "
            "over-read a two-point contrast (the null BANDS quantify metric sampling "
            "noise, not seed-to-seed variation).")
    else:
        cons.append("SINGLE BANK: no cross-model contrast is computed in this run.")
    cons += [
        "PROBE-PAIR arm UNAVAILABLE: no current bank saves probe keys, so "
        "'random-pair vs matched-probe-pair cosine' cannot be computed. This "
        "measures EDIT-key anisotropy only.",
        "KEY-SPACE ONLY: these banks carry no COS / damage matrices, so NO damage "
        "join is possible from the key banks alone. This characterises the key "
        "distribution; it does NOT by itself show anisotropy CAUSES the damage "
        "regime transition. Attribution needs a damage-joined follow-up.",
        f"n (=N edits, ~{n_txt}) << d (={d_txt}): the d x d covariance has rank <= "
        "N-1, so ~ (d-N+1) directions carry zero empirical variance and the spectrum "
        "tail beyond ~N-1 is zero BY CONSTRUCTION, not signal. Full d-space whitening "
        "is ILL-POSED; only populated-subspace (top-r) whitening is reported, across "
        "a shrinkage grid. Whitening mechanically isotropizes the RETAINED subspace, "
        "so read RELATIVE cross-bank contrasts of the whitened cosine, not the "
        "absolute whitened numbers, and read the alpha-dependence.",
        f"PER-BANK SINGLE SEED (seeds present: {seeds}): each bank is one seed, so no "
        "cross-seed stability is estimated within this file — the null BANDS quantify "
        "metric sampling noise, not seed-to-seed variation. Aggregate the per-seed "
        "output files externally for seed stability.",
        "column_permutation preserves per-neuron variance AND the mean vector, so "
        "its mean_cos is not a valid isotropy null (use the Gaussian band); it is a "
        "valid null ONLY for rotational (off-axis) PR.",
        "Anisotropy is a property of THIS edit-key sample (CounterFact subjects); it "
        "cannot be separated from tokenizer / subject-frequency confounds without a "
        "controlled probe set.",
    ]
    return cons


def main():
    ap = argparse.ArgumentParser(description="Anisotropy / whitening profile of ROME edit-key banks.")
    ap.add_argument("banks", nargs="+", help="1 or 2 killgate --save_vectors npz banks (need raw 'K').")
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument("--n_rep", type=int, default=200, help="baseline resamples for the null bands")
    ap.add_argument("--var_keep", type=float, default=0.95,
                    help="cumulative-variance fraction defining the whitening subspace rank")
    ap.add_argument("--shrink_grid", type=float, nargs="+", default=[0.01, 0.1, 0.5],
                    help="shrinkage alphas (fraction of mean retained variance)")
    args = ap.parse_args()

    if len(args.banks) > 2:
        raise SystemExit("[aniso] pass at most 2 banks (single profile, or a cross-model pair).")

    t0 = time.time()
    rng = np.random.default_rng(RNG_SEED)
    # NB: the two banks SHARE this rng, so bank 2's null bands consume rng state left by bank 1
    # — the null-band numbers are therefore bank-ORDER-sensitive. Left as-is deliberately to keep
    # byte-identity with the s0/s1/s2 outputs already on disk; give each bank its own seeded rng in
    # a future version if order-independence is wanted.
    profiles = [analyze_one(p, args, rng) for p in args.banks]

    report = {
        "analysis": "anisotropy_whitening_edit_keys",
        "pass": "AUTHORING (hostile review gates any paper claim)",
        "rng_seed": RNG_SEED,
        "params": {"n_rep": args.n_rep, "var_keep": args.var_keep, "shrink_grid": args.shrink_grid},
        "banks": profiles,
        "array_provenance_notes": {
            "K": "base-model EDIT keys = down_proj input at edit layer L, subject "
                 "last-token position, raw/un-normalised [N, d_in]. Source: "
                 "killgate_keygeom.py:363-376 (key_for -> _capture_key), saved :814.",
            "knorm": "row norms of K (||k|| per edit), killgate_keygeom.py:823.",
            "layer/model/editor/seed": "run provenance scalars stored in the bank.",
            "NOT_PRESENT": "probe keys, probe cosine matrix, and collateral-damage "
                           "matrices are NOT in these banks (K_probe computed but "
                           "unsaved; GRAM_pre only in SEQ mode). No proxy used.",
        },
        "interpretation_constraints": build_interpretation_constraints(profiles),
        "runtime_s": round(time.time() - t0, 1),
    }
    if len(profiles) == 2:
        report["cross_model_contrast"] = cross_model_contrast(profiles)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(report, fh, indent=2)
    os.replace(tmp, args.out)
    print(f"[aniso] wrote {args.out} ({time.time()-t0:.1f}s)")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
