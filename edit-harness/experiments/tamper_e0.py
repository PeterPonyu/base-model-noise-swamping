"""tamper_e0.py — T1.2 edit-forensics / weight-space TAMPER DETECTION + localization (Phase-0-CPU).

Implements the FROZEN pre-registration docs/plans/PREREG-T12-TAMPER-E0-20260713.md,
Phase-0-CPU (section 5A) EXACTLY. CPU-only (numpy + torch-CPU for reading bf16
safetensors); zero GPU. This script is the kill-gate that decides whether to spend the
(cheap) Phase-0-GPU breadth dump. NO feature or threshold is added beyond the prereg.

WHAT IT DOES (prereg section 5A steps 1-11)
  1. Reconstruct, per ROME cell, the 200 edited target-layer matrices exactly:
     W_edited_i = Wbase + outer(A_i, B_i); hard-assert vectors_valid==1 and
     max(recon_rel_err) <= 1e-3 before use (prereg step 1).
  2. Clean layer bank: load every mlp.down_proj from the checkpoint on CPU (fp32 from
     bf16) -> clean negatives + localization distractors; verify each cell's Wbase equals
     the checkpoint target-layer weight (prereg step 2).
  3-5. Spectra + MP bulk-edge (ARM-N) + reference-based deltas (ARM-R) features (prereg
     steps 3-5), from the admitted feature set only (prereg section 2, features 1-7).
  6. S1 detection floor: ARM-N train-free statistic (1) AUROC edited-vs-clean, pooled and
     per-architecture; logistic secondary over features {1-5} under the frozen split
     (fit on Llama-1B only, test on held-out Qwen-1.5B); ARM-R AUROC as the ceiling.
  7. S2: the 2410.17770 single-matrix RMT-null baseline (eigenvalue-count-beyond-edge)
     head-to-head vs ARM-N (summed-magnitude-excess).
  8. H1 localization (the primary CPU headline): rank all layers of the checkpoint with
     the edited target layer's W_edited substituted in; reference-free ARM-N score; top-1
     rate vs 1/n_layers chance.
  9. H3 fold-in: the SV = ||A||*||B|| = norm_growth identity check (corr ~1 by
     construction), then within-cell Spearman rho(MP-spike, per-edit mean collateral
     damage) from the matching results/matrices/qv_*.npz (damage_logit); both architectures.
 10. H4 (SECONDARY, weight-space reversibility proxy): within-probe rho(pre-edit
     key-cosine, reversal residual R_i); reported, NOT in the gate.
 11. GRACE negative control: only if grace vectors exist (delta==0 => detector must return
     chance); absent here => "GRACE control: SKIPPED (no inputs)".
 12. Evaluate the section-4 CPU PASS/KILL/GREY gate; emit results/tamper_e0/E0_report.json.

DECISION RULE (prereg section 4, CPU gate). PASS iff ALL of:
  (S1) ARM-N AUROC > 0.80 pooled AND > 0.75 per architecture;
  (H1) localization top-1 >= 0.60 AND >= 3x chance on BOTH Llama-1B and Qwen-1.5B;
  (H3) within-cell rho(spike, S) > 0 (sanity ~1) AND rho(spike, mean damage) > 0, both arch;
  (S2) ARM-N AUROC >= the RMT-null baseline AUROC.
KILL iff H1 fails 3x chance on BOTH arch, OR S1 AUROC <= 0.70 at Llama-1B (easiest case).
GREY iff H1 half-meets (passes one arch only) OR 0.70 < S1(Llama) <= 0.80.

ARM-N reference-free statistic (1) = the MP bulk-edge excess. The scalar reporter used
for S1/S2/H1 is the SUMMED NORMALIZED positive excess  sum_i max(sigma_i/sv_edge - 1, 0)
(scale-free across layers of differing weight scale); the count #{sigma_i > sv_edge} is the
2410.17770 RMT-null baseline scalar (S2). sv_edge is the Marchenko-Pastur singular-value
edge tau*(sqrt(m)+sqrt(n)) with per-entry noise tau estimated MAD/median-robustly from the
mid-spectrum (median-of-eigenvalues calibrated against the closed-form MP median for the
matrix aspect beta = min_dim/max_dim). This keys on the MP-edge EXCESS, not raw sigma_1
(prereg threat-to-validity section 7).

Conventions mirror experiments/analyze_matrices.py (canonical `spearman`,
`within_probe_rhos`, edit_ok / pre_p>0.05 known-probe filters) and the HERE/ROOT/RESULTS
layout of the other experiments/ scripts.

Usage:
  python tamper_e0.py --selftest          # synthetic; reads NO real npz; exits nonzero on fail
  python tamper_e0.py                      # run the Phase-0-CPU gate on the real assets
Output: results/tamper_e0/E0_report.json  (numbers keyed by prereg claim IDs S1/S2/H1/H3/H4)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # edit-harness/
RESULTS = os.path.join(ROOT, "results")
VECTORS = os.path.join(RESULTS, "vectors")
MATRICES = os.path.join(RESULTS, "matrices")
MODELS = os.path.join(ROOT, "data", "models")
OUT_DIR = os.path.join(RESULTS, "tamper_e0")

RNG_SEED = 12345          # every stochastic step (perturbations, logistic init) is seeded
RECON_TOL = 1e-3          # prereg vectors_valid gate: max recon_rel_err <= 1e-3
WBASE_TOL = 1e-2          # Wbase-vs-checkpoint sanity (both round through bf16 -> exact 0)

# The Phase-0-CPU asset set (prereg section 3). Each entry is a `vectors_qv_<TAG>.npz`
# whose matching collateral file is `matrices/qv_<TAG>.npz`.
MODEL_DIRS = {"llama1b": "Llama-3.2-1B", "qwen15b": "Qwen2.5-1.5B"}
ARCH_OF = {"llama1b": "llama", "qwen15b": "qwen"}
N_LAYERS = {"llama": 16, "qwen": 28}  # num_hidden_layers (verified from config.json)

CPU_VECTOR_CELLS = [
    "vectors_qv_llama1b_rome_cf_L8_s0.npz",
    "vectors_qv_llama1b_rome_cf_L12_s0.npz",
    "vectors_qv_llama1b_rome_cf_L14_s0.npz",
    "vectors_qv_llama1b_rome_cf_L14_s1.npz",
    "vectors_qv_llama1b_rome_cf_L14_s2.npz",
    "vectors_qv_qwen15b_rome_cf_L14_s0.npz",
    "vectors_qv_qwen15b_rome_cf_L14_s1.npz",
    "vectors_qv_qwen15b_rome_cf_L14_s2.npz",
]

# The pre-committed logistic feature vector (prereg section 2, ARM-N features 1-5).
ARMN_FEATURES = ["summed_excess", "spectral_gap", "stable_rank", "eff_rank",
                 "bulk_edge_ratio", "participation"]


# --------------------------------------------------------------------------------------
# Canonical rank statistics (mirrored verbatim from analyze_matrices.py so the H3/H4
# within-probe numbers are computed with the identical estimator as the B6 gate).
# --------------------------------------------------------------------------------------
def _midrank(x):
    """Tie-averaged ranks (proper Spearman ranks)."""
    x = np.asarray(x, float)
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


def auroc(scores, labels):
    """AUROC via the Mann-Whitney U statistic with tie-averaged ranks. labels: 1=positive."""
    scores = np.asarray(scores, float)
    labels = np.asarray(labels)
    finite = np.isfinite(scores)
    scores, labels = scores[finite], labels[finite]
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = _midrank(np.concatenate([pos, neg]))
    r_pos = ranks[:n_pos].sum()
    u = r_pos - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


# --------------------------------------------------------------------------------------
# Marchenko-Pastur singular-value edge (prereg section 2.1)
# --------------------------------------------------------------------------------------
def mp_median(beta, n_grid=400001):
    """Median of the standard Marchenko-Pastur eigenvalue law (per-entry variance 1) for
    aspect ratio beta = min_dim/max_dim in (0,1]. Support [a,b], a=(1-sqrt b)^2,
    b=(1+sqrt b)^2; density p(l) = sqrt((b-l)(l-a)) / (2 pi beta l). Computed by
    trapezoidal CDF inversion (deterministic)."""
    beta = float(beta)
    a = (1.0 - np.sqrt(beta)) ** 2
    b = (1.0 + np.sqrt(beta)) ** 2
    lam = np.linspace(a, b, n_grid)
    dens = np.sqrt(np.clip((b - lam) * (lam - a), 0.0, None)) / (2.0 * np.pi * beta * lam)
    cdf = np.cumsum(dens) * (lam[1] - lam[0])
    cdf /= cdf[-1]
    return float(np.interp(0.5, cdf, lam))


def sv_edge_and_tau(svals, m, n):
    """Robust MP singular-value bulk edge for an m x n matrix given its singular values
    (descending, length min(m,n)).

    tau (per-entry noise scale) is estimated MAD/median-robustly from the mid-spectrum:
    the eigenvalues of (1/n) W W^T are l_i = sigma_i^2 / n and, under an MP null with
    per-entry variance tau^2, their median equals tau^2 * mp_median(beta). So
    tau^2 = median(l_i) / mp_median(beta). The median over all min(m,n) values is
    insensitive to a handful of top spikes (exactly what we are trying to detect).

    Singular-value edge sv_edge = tau * (sqrt(m) + sqrt(n)) = sqrt(n) * tau * (1 + sqrt beta),
    beta = min(m,n)/max(m,n). Returns (sv_edge, tau)."""
    svals = np.asarray(svals, float)
    p = min(m, n)
    big = max(m, n)
    beta = p / big
    ell = svals[:p] ** 2 / big               # eigenvalues of (1/big) W W^T
    med = np.median(ell)
    tau2 = med / mp_median(beta)
    tau = float(np.sqrt(max(tau2, 0.0)))
    sv_edge = tau * (np.sqrt(m) + np.sqrt(n))
    return float(sv_edge), tau


# --------------------------------------------------------------------------------------
# Spectra via the Gram matrix (efficient + exact for the top spectrum)
# --------------------------------------------------------------------------------------
def gram_eigh(W, want_vectors=False):
    """Singular values (descending) of W [m x n] via eigendecomposition of the smaller
    Gram matrix W W^T (m<=n here: down_proj is d_out x d_in). If want_vectors, also
    return the left singular vectors U [m x m] (descending). Squaring inflates the
    condition number of the tiny bulk but the TOP spectrum (all that S1/H1/H4 use) is
    accurate; selftest cross-checks this against a direct SVD."""
    m, n = W.shape
    if m <= n:
        G = W @ W.T                          # m x m
        if want_vectors:
            w, V = np.linalg.eigh(G)
            idx = np.argsort(w)[::-1]
            svals = np.sqrt(np.clip(w[idx], 0.0, None))
            return svals, V[:, idx]
        w = np.linalg.eigvalsh(G)
        return np.sqrt(np.clip(w[::-1], 0.0, None)), None
    else:
        G = W.T @ W                          # n x n (not hit for down_proj; kept correct)
        if want_vectors:
            w, Vr = np.linalg.eigh(G)
            idx = np.argsort(w)[::-1]
            svals = np.sqrt(np.clip(w[idx], 0.0, None))
            # return left vectors for interface parity: U = W Vr / sigma
            Vr = Vr[:, idx]
            with np.errstate(divide="ignore", invalid="ignore"):
                U = (W @ Vr) / np.where(svals > 0, svals, 1.0)
            return svals, U
        w = np.linalg.eigvalsh(G)
        return np.sqrt(np.clip(w[::-1], 0.0, None)), None


def armn_features(svals, m, n):
    """ARM-N reference-free features (prereg section 2, features 1-5)."""
    svals = np.asarray(svals, float)
    p = len(svals)
    sv_edge, tau = sv_edge_and_tau(svals, m, n)
    excess = np.maximum(svals / sv_edge - 1.0, 0.0)
    summed_excess = float(excess.sum())                      # statistic (1): summed magnitude
    n_beyond = int((svals > sv_edge).sum())                  # statistic (1): count (= RMT-null)
    s1 = svals[0]
    s2 = svals[1] if p > 1 else svals[0]
    spectral_gap = float(s1 / s2) if s2 > 0 else np.inf      # (2)
    t = min(50, p)
    topd = svals[:t]
    denom = np.where(topd[:-1] > 0, topd[:-1], 1.0)
    max_norm_gap = float(((topd[:-1] - topd[1:]) / denom).max()) if t > 1 else 0.0  # (2)
    fro2 = float((svals ** 2).sum())
    stable_rank = float(fro2 / (s1 ** 2)) if s1 > 0 else np.nan                     # (3)
    pspec = svals ** 2 / fro2 if fro2 > 0 else np.ones(p) / p
    eff_rank = float(np.exp(-(pspec * np.log(pspec + 1e-30)).sum()))                # (3)
    bulk_edge_ratio = float(s1 / sv_edge)                                           # (4)
    participation = float(1.0 / (pspec ** 2).sum())                                # (5)
    return {
        "summed_excess": summed_excess, "n_beyond": n_beyond,
        "spectral_gap": spectral_gap, "max_norm_gap": max_norm_gap,
        "stable_rank": stable_rank, "eff_rank": eff_rank,
        "bulk_edge_ratio": bulk_edge_ratio, "participation": participation,
        "sv_edge": sv_edge, "tau": tau, "sigma1": float(s1),
    }


def armr_features(sv_cur, sv_clean, U_cur=None, U_clean=None, W_cur=None, W_clean=None):
    """ARM-R reference-based features (prereg section 2, features 6-7). Feature 7
    (principal-angle / right-subspace overlap over the top-r right-singular subspaces)
    is computed from left vectors: v_i = W^T u_i / sigma_i, then mean cos(principal
    angle) = mean singular value of V_r(cur)^T V_r(clean)."""
    sv_cur = np.asarray(sv_cur, float)
    sv_clean = np.asarray(sv_clean, float)
    k = min(len(sv_cur), len(sv_clean))
    d = np.abs(sv_cur[:k] - sv_clean[:k])
    feats = {
        "sv_delta_k1": float(d[0]) if k >= 1 else np.nan,
        "sv_delta_k5": float(d[:5].sum()) if k >= 5 else float(d.sum()),
        "sv_delta_k20": float(d[:20].sum()) if k >= 20 else float(d.sum()),
    }
    if U_cur is not None and U_clean is not None and W_cur is not None and W_clean is not None:
        r = min(20, U_cur.shape[1], U_clean.shape[1])
        with np.errstate(divide="ignore", invalid="ignore"):
            Vc = (W_cur.T @ U_cur[:, :r]) / np.where(sv_cur[:r] > 0, sv_cur[:r], 1.0)
            Vk = (W_clean.T @ U_clean[:, :r]) / np.where(sv_clean[:r] > 0, sv_clean[:r], 1.0)
        Vc /= (np.linalg.norm(Vc, axis=0, keepdims=True) + 1e-30)
        Vk /= (np.linalg.norm(Vk, axis=0, keepdims=True) + 1e-30)
        cos_pa = np.linalg.svd(Vc.T @ Vk, compute_uv=False)
        feats["subspace_overlap_r20"] = float(np.mean(np.clip(cos_pa, -1, 1)))
    return feats


# --------------------------------------------------------------------------------------
# Efficient per-edit edited-matrix spectrum: Gram = G0 + rank-2 update (no full SVD)
# --------------------------------------------------------------------------------------
def edited_gram_spectrum(G0, Wbase, a, b, want_vectors=False):
    """Spectrum (and optionally left vectors) of W_edited = Wbase + outer(a, b), given the
    precomputed clean Gram G0 = Wbase @ Wbase^T. Exact identity:
        W_edited W_edited^T = G0 + (Wbase b) a^T + a (Wbase b)^T + (b.b) a a^T .
    a is d_out (len m), b is d_in (len n)."""
    p = Wbase @ b                                    # d_out
    bb = float(b @ b)
    G = G0 + np.outer(p, a) + np.outer(a, p) + bb * np.outer(a, a)
    if want_vectors:
        w, V = np.linalg.eigh(G)
        idx = np.argsort(w)[::-1]
        return np.sqrt(np.clip(w[idx], 0.0, None)), V[:, idx]
    w = np.linalg.eigvalsh(G)
    return np.sqrt(np.clip(w[::-1], 0.0, None)), None


# --------------------------------------------------------------------------------------
# Model loading (bf16 safetensors -> fp32 on CPU, down_proj only)
# --------------------------------------------------------------------------------------
def load_down_proj_bank(model_key):
    """Return {layer_idx: W (fp32 d_out x d_in)} for every mlp.down_proj of the checkpoint,
    read on CPU via torch (weights are bf16)."""
    import torch  # noqa: F401  (only for bf16 safetensors read)
    from safetensors import safe_open
    mdir = os.path.join(MODELS, MODEL_DIRS[model_key])
    path = os.path.join(mdir, "model.safetensors")
    bank = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for k in f.keys():
            if k.endswith("mlp.down_proj.weight"):
                li = int(k.split("model.layers.")[1].split(".")[0])
                bank[li] = f.get_tensor(k).float().numpy()
    return bank


# --------------------------------------------------------------------------------------
# Per-cell reconstruction + feature extraction
# --------------------------------------------------------------------------------------
def parse_tag(vec_fname):
    tag = vec_fname[len("vectors_qv_"):-len(".npz")]   # e.g. llama1b_rome_cf_L8_s0
    parts = tag.split("_")
    model_key = parts[0]
    layer = int([q for q in parts if q.startswith("L") and q[1:].isdigit()][0][1:])
    seed = int([q for q in parts if q.startswith("s") and q[1:].isdigit()][0][1:])
    return tag, model_key, ARCH_OF[model_key], layer, seed


def process_cell(vec_fname, clean_banks, limit_edits=None):
    """Reconstruct one ROME cell and extract all per-edit spectra + ARM-N/ARM-R features
    for the edited target-layer matrices. Returns a dict of arrays."""
    tag, model_key, arch, layer, seed = parse_tag(vec_fname)
    vpath = os.path.join(VECTORS, vec_fname)
    d = np.load(vpath, allow_pickle=True)
    # prereg step 1 hard gates
    assert int(d["vectors_valid"]) == 1, f"{vec_fname}: vectors_valid != 1"
    rre = d["recon_rel_err"].astype(float)
    assert float(rre.max()) <= RECON_TOL, f"{vec_fname}: max recon_rel_err {rre.max()} > {RECON_TOL}"
    A = d["A"].astype(np.float64)          # [N, d_out]
    B = d["B"].astype(np.float64)          # [N, d_in]
    Wbase = d["Wbase"].astype(np.float64)  # [d_out, d_in]
    norm_growth = d["norm_growth"].astype(float)
    edit_ok = d["edit_ok"].astype(float) if "edit_ok" in d.files else np.ones(len(A))
    m, n = Wbase.shape
    N = A.shape[0] if limit_edits is None else min(limit_edits, A.shape[0])

    # prereg step 2: Wbase == checkpoint target-layer weight (sanity)
    W_ckpt = clean_banks[model_key][layer].astype(np.float64)
    wbase_maxdiff = float(np.max(np.abs(Wbase - W_ckpt)))
    assert wbase_maxdiff <= WBASE_TOL, f"{vec_fname}: Wbase vs checkpoint maxdiff {wbase_maxdiff}"

    # clean reference spectrum (Wbase) once
    G0 = Wbase @ Wbase.T
    sv_clean, U_clean = gram_eigh(Wbase, want_vectors=True)
    clean_armn = armn_features(sv_clean, m, n)

    per_edit = {kk: [] for kk in ARMN_FEATURES + ["n_beyond", "sv_edge", "sigma1"]}
    armr = {"sv_delta_k1": [], "sv_delta_k5": [], "sv_delta_k20": [], "subspace_overlap_r20": []}
    reversal_R = []            # H4 weight-space residual
    ab_singular = []           # H3 identity: ||A||*||B||
    for i in range(N):
        a, b = A[i], B[i]
        sv_e, U_e = edited_gram_spectrum(G0, Wbase, a, b, want_vectors=True)
        fN = armn_features(sv_e, m, n)
        for kk in ARMN_FEATURES:
            per_edit[kk].append(fN[kk])
        per_edit["n_beyond"].append(fN["n_beyond"])
        per_edit["sv_edge"].append(fN["sv_edge"])
        per_edit["sigma1"].append(fN["sigma1"])
        # ARM-R vs Wbase
        W_e = Wbase + np.outer(a, b)
        fR = armr_features(sv_e, sv_clean, U_e, U_clean, W_e, Wbase)
        for kk in armr:
            armr[kk].append(fR.get(kk, np.nan))
        # H3 identity singular value of the rank-one implant
        ab_singular.append(float(np.linalg.norm(a) * np.linalg.norm(b)))
        # H4: reversal by subtracting the reference-free top-1 component u1 (u1^T W_e)
        u1 = U_e[:, 0]
        w1 = W_e.T @ u1                                  # = sigma1 * v1
        dW_true_fro2 = float((a @ a) * (b @ b))          # ||outer(a,b)||_F^2
        # ||outer(a,b) - outer(u1, w1)||_F^2 (closed form, no dense d_out x d_in)
        cross = float((a @ u1) * (b @ w1))
        resid2 = dW_true_fro2 + float((u1 @ u1) * (w1 @ w1)) - 2.0 * cross
        R = np.sqrt(max(resid2, 0.0)) / np.sqrt(dW_true_fro2) if dW_true_fro2 > 0 else np.nan
        reversal_R.append(R)

    out = {
        "tag": tag, "model_key": model_key, "arch": arch, "layer": layer, "seed": seed,
        "m": int(m), "n": int(n), "N": int(N),
        "wbase_maxdiff": wbase_maxdiff, "recon_rel_err_max": float(rre.max()),
        "norm_growth": norm_growth[:N], "edit_ok": edit_ok[:N],
        "ab_singular": np.array(ab_singular),
        "reversal_R": np.array(reversal_R),
        "clean_armn": clean_armn, "sv_clean": sv_clean,
    }
    for kk in per_edit:
        out[kk] = np.array(per_edit[kk], float)
    for kk in armr:
        out["armr_" + kk] = np.array(armr[kk], float)
    return out


def load_matching_damage(tag, N, known_filter=True):
    """Per-edit mean collateral damage from matrices/qv_<tag>.npz (damage_logit), averaged
    over known probes (pre_p>0.05) if requested, plus the edit_ok mask. Returns
    (mean_damage[N], edit_ok[N]) or (None, None) if the file is absent."""
    mpath = os.path.join(MATRICES, "qv_" + tag + ".npz")
    if not os.path.exists(mpath):
        return None, None
    md = np.load(mpath, allow_pickle=True)
    D = md["damage_logit"].astype(float)           # [N, M]
    if known_filter and "pre_p" in md.files:
        cols = md["pre_p"].astype(float) > 0.05
        if cols.sum() >= 5:
            D = D[:, cols]
    eok = md["edit_ok"].astype(float) if "edit_ok" in md.files else np.ones(D.shape[0])
    mean_dmg = D.mean(axis=1)
    return mean_dmg[:N], eok[:N]


# --------------------------------------------------------------------------------------
# Numpy logistic regression (ARM-N secondary; fit on Llama, test on Qwen)
# --------------------------------------------------------------------------------------
def logistic_fit_predict(Xtr, ytr, Xte, iters=2000, lr=0.1, l2=1e-3, seed=RNG_SEED):
    """Standardize on train, gradient-descent logistic regression, return test scores."""
    mu = Xtr.mean(0)
    sd = Xtr.std(0) + 1e-9
    Xtr_s = (Xtr - mu) / sd
    Xte_s = (Xte - mu) / sd
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 0.01, Xtr_s.shape[1])
    b = 0.0
    n = len(ytr)
    for _ in range(iters):
        z = Xtr_s @ w + b
        pr = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        g = pr - ytr
        gw = Xtr_s.T @ g / n + l2 * w
        gb = g.mean()
        w -= lr * gw
        b -= lr * gb
    return Xte_s @ w + b, (Xtr_s @ w + b)


# --------------------------------------------------------------------------------------
# Main analysis (prereg steps 6-11)
# --------------------------------------------------------------------------------------
def build_detection_sets(cells, clean_banks, rng):
    """S1 detection matrices. Positives = edited target-layer matrices (ARM-N summed_excess
    per edit). Negatives = clean-layer bank (all down_proj) + Gaussian-perturbed clean at
    the cell's median-Delta-W Frobenius scale (a NON-EDIT baseline), balanced to n_pos per
    architecture (prereg section 3). Returns per-arch dicts of scores/labels for ARM-N,
    ARM-R ceiling, and the RMT-null count baseline, plus the logistic feature matrices."""
    # clean layer ARM-N features per model (scored once)
    clean_feats = {}
    for mk, bank in clean_banks.items():
        arch = ARCH_OF[mk]
        rows = []
        for li, W in sorted(bank.items()):
            sv, _ = gram_eigh(W, want_vectors=False)
            rows.append(armn_features(sv, W.shape[0], W.shape[1]))
        clean_feats[mk] = rows

    per_arch = {}
    for arch in ["llama", "qwen"]:
        arch_cells = [c for c in cells if c["arch"] == arch]
        if not arch_cells:
            continue
        mk = arch_cells[0]["model_key"]
        m0, n0 = arch_cells[0]["m"], arch_cells[0]["n"]
        # positives
        pos_summed = np.concatenate([c["summed_excess"] for c in arch_cells])
        pos_count = np.concatenate([c["n_beyond"].astype(float) for c in arch_cells])
        pos_armr = np.concatenate([c["armr_sv_delta_k1"] for c in arch_cells])
        pos_logit = np.column_stack([np.concatenate([c[k] for c in arch_cells])
                                     for k in ARMN_FEATURES])
        n_pos = len(pos_summed)
        # negatives: clean bank + perturbed-clean to balance
        neg_summed, neg_count, neg_armr, neg_logit = [], [], [], []
        for f in clean_feats[mk]:
            neg_summed.append(f["summed_excess"]); neg_count.append(float(f["n_beyond"]))
            neg_armr.append(0.0)  # unperturbed clean: reference == itself => delta 0 (ceiling)
            neg_logit.append([f[k] for k in ARMN_FEATURES])
        # matched-Frobenius full-rank Gaussian perturbations (non-edit baseline). The
        # per-edit rank-one Frobenius norm is exactly the implanted singular value
        # ||A||*||B|| = ab_singular; use the cell-pooled median as the matched scale.
        med_dw_fro = float(np.median(np.concatenate([c["ab_singular"] for c in arch_cells])))
        bank = clean_banks[mk]
        layer_ids = sorted(bank.keys())
        while len(neg_summed) < n_pos:
            li = layer_ids[rng.integers(0, len(layer_ids))]
            W = bank[li].astype(np.float64)
            noise = rng.standard_normal(W.shape)
            noise *= med_dw_fro / (np.linalg.norm(noise) + 1e-30)   # matched Frobenius scale
            Wp = W + noise
            sv, _ = gram_eigh(Wp, want_vectors=False)
            f = armn_features(sv, W.shape[0], W.shape[1])
            neg_summed.append(f["summed_excess"]); neg_count.append(float(f["n_beyond"]))
            svc, _ = gram_eigh(W, want_vectors=False)
            neg_armr.append(abs(sv[0] - svc[0]))                    # ARM-R delta vs pre-perturb
            neg_logit.append([f[k] for k in ARMN_FEATURES])
        neg_summed = np.array(neg_summed[:n_pos]); neg_count = np.array(neg_count[:n_pos])
        neg_armr = np.array(neg_armr[:n_pos]); neg_logit = np.array(neg_logit[:n_pos])

        scores_summed = np.concatenate([pos_summed, neg_summed])
        scores_count = np.concatenate([pos_count, neg_count])
        scores_armr = np.concatenate([pos_armr, neg_armr])
        X = np.vstack([pos_logit, neg_logit])
        y = np.concatenate([np.ones(n_pos), np.zeros(len(neg_summed))])
        per_arch[arch] = {
            "armn_score": scores_summed, "rmt_count": scores_count,
            "armr_score": scores_armr, "X": X, "y": y, "n_pos": n_pos,
        }
    return per_arch


def run_gate(limit_edits=None):
    rng = np.random.default_rng(RNG_SEED)
    # load clean banks
    clean_banks = {mk: load_down_proj_bank(mk) for mk in MODEL_DIRS}
    for mk in clean_banks:
        assert len(clean_banks[mk]) == N_LAYERS[ARCH_OF[mk]], \
            f"{mk}: got {len(clean_banks[mk])} down_proj, expected {N_LAYERS[ARCH_OF[mk]]}"

    # process every CPU cell present
    cells = []
    missing = []
    for vf in CPU_VECTOR_CELLS:
        if not os.path.exists(os.path.join(VECTORS, vf)):
            missing.append(vf)
            continue
        cells.append(process_cell(vf, clean_banks, limit_edits=limit_edits))
    assert cells, "no CPU vector cells found"

    report = {"prereg": "PREREG-T12-TAMPER-E0-20260713.md", "phase": "Phase-0-CPU",
              "cells": [c["tag"] for c in cells], "missing_cells": missing,
              "rng_seed": RNG_SEED}

    # ---- S1 / S2 detection ----
    det = build_detection_sets(cells, clean_banks, rng)
    s1 = {"per_arch": {}, "armr_ceiling": {}}
    s2 = {"per_arch": {}}
    pooled_armn = {"score": [], "y": []}
    pooled_rmt = {"score": [], "y": []}
    for arch, dd in det.items():
        a_armn = auroc(dd["armn_score"], dd["y"])
        a_armr = auroc(dd["armr_score"], dd["y"])
        a_rmt = auroc(dd["rmt_count"], dd["y"])
        s1["per_arch"][arch] = round(a_armn, 4)
        s1["armr_ceiling"][arch] = round(a_armr, 4)
        s2["per_arch"][arch] = round(a_rmt, 4)
        pooled_armn["score"].append(dd["armn_score"]); pooled_armn["y"].append(dd["y"])
        pooled_rmt["score"].append(dd["rmt_count"]); pooled_rmt["y"].append(dd["y"])
    s1_pooled = auroc(np.concatenate(pooled_armn["score"]), np.concatenate(pooled_armn["y"]))
    s2_pooled = auroc(np.concatenate(pooled_rmt["score"]), np.concatenate(pooled_rmt["y"]))
    s1["pooled"] = round(s1_pooled, 4)
    s2["pooled_rmt"] = round(s2_pooled, 4)
    s2["armn_pooled"] = round(s1_pooled, 4)
    s2["armn_ge_rmt"] = bool(s1_pooled >= s2_pooled)

    # logistic secondary under the frozen split (fit Llama, test Qwen)
    if "llama" in det and "qwen" in det:
        te_scores, tr_scores = logistic_fit_predict(det["llama"]["X"], det["llama"]["y"],
                                                     det["qwen"]["X"])
        s1["logistic_secondary"] = {
            "train_arch": "llama", "test_arch": "qwen",
            "train_auroc": round(auroc(tr_scores, det["llama"]["y"]), 4),
            "heldout_auroc": round(auroc(te_scores, det["qwen"]["y"]), 4),
            "note": "reported per prereg section 2; the GATE uses the train-free statistic (1)",
        }

    # ---- H1 localization ----
    # clean ARM-N summed_excess per layer (distractors), scored once per model
    clean_layer_score = {}
    for mk, bank in clean_banks.items():
        sc = {}
        for li, W in bank.items():
            sv, _ = gram_eigh(W, want_vectors=False)
            sc[li] = armn_features(sv, W.shape[0], W.shape[1])["summed_excess"]
        clean_layer_score[mk] = sc
    h1 = {"per_cell": {}, "per_arch": {}}
    arch_hits = {"llama": [0, 0], "qwen": [0, 0]}  # [top1_hits, total]
    for c in cells:
        mk, layer, arch = c["model_key"], c["layer"], c["arch"]
        base = dict(clean_layer_score[mk])
        others = [v for li, v in base.items() if li != layer]
        hits = 0
        for i in range(c["N"]):
            tgt = c["summed_excess"][i]
            # target #1 iff its score exceeds every distractor layer's clean score
            if tgt > max(others):
                hits += 1
        top1 = hits / c["N"]
        h1["per_cell"][c["tag"]] = {"top1": round(top1, 4), "chance": round(1.0 / N_LAYERS[arch], 4)}
        arch_hits[arch][0] += hits
        arch_hits[arch][1] += c["N"]
    for arch in ["llama", "qwen"]:
        if arch_hits[arch][1] == 0:
            continue
        top1 = arch_hits[arch][0] / arch_hits[arch][1]
        chance = 1.0 / N_LAYERS[arch]
        h1["per_arch"][arch] = {"top1": round(top1, 4), "chance": round(chance, 4),
                                "x_chance": round(top1 / chance, 2)}

    # ---- H3 fold-in ----
    h3 = {"per_cell": {}, "per_arch": {}}
    arch_id, arch_dmg = {"llama": [], "qwen": []}, {"llama": [], "qwen": []}
    for c in cells:
        arch = c["arch"]
        spike = c["ab_singular"]                       # = ||A||*||B|| (the implanted SV)
        S = c["norm_growth"]
        rho_id = spearman(spike, S)                    # sanity: ~1 by the identity
        max_rel = float(np.max(np.abs(spike - S) / (np.abs(S) + 1e-12)))
        mean_dmg, eok = load_matching_damage(c["tag"], c["N"])
        entry = {"rho_spike_S": round(rho_id, 4), "identity_max_rel_diff": max_rel}
        # review M-a: a TRUNCATED matrices file (fewer rows than the vectors cell) must
        # skip gracefully like a missing file, not IndexError into spike[mask].
        if mean_dmg is not None and (len(mean_dmg) != c["N"] or len(eok) != c["N"]):
            entry["damage_file"] = "LENGTH_MISMATCH"
            sys.stderr.write(f"[tamper_e0] WARNING: {c['tag']} damage len {len(mean_dmg)} != N {c['N']} — treated as missing\n")
            mean_dmg = None
        if mean_dmg is not None:
            mask = eok > 0.5
            rho_d = spearman(spike[mask], mean_dmg[mask])
            entry["rho_spike_damage"] = round(rho_d, 4)
            entry["n_edit_ok"] = int(mask.sum())
            arch_dmg[arch].append(rho_d)
        else:
            entry["rho_spike_damage"] = None
            entry["damage_file"] = "MISSING"
        h3["per_cell"][c["tag"]] = entry
        arch_id[arch].append(rho_id)
    for arch in ["llama", "qwen"]:
        if not arch_id[arch]:
            continue
        h3["per_arch"][arch] = {
            "mean_rho_spike_S": round(float(np.nanmean(arch_id[arch])), 4),
            "mean_rho_spike_damage": (round(float(np.nanmean(arch_dmg[arch])), 4)
                                      if arch_dmg[arch] else None),
        }

    # ---- H4 secondary (reversibility proxy; NOT gated) ----
    h4 = {"per_cell": {}, "per_arch": {}, "note": "SECONDARY; reported, not in the CPU gate"}
    arch_h4 = {"llama": [], "qwen": []}
    for c in cells:
        mpath = os.path.join(MATRICES, "qv_" + c["tag"] + ".npz")
        rho = None
        if os.path.exists(mpath):
            md = np.load(mpath, allow_pickle=True)
            COS = md["COS"].astype(float)
            eok = md["edit_ok"].astype(float) if "edit_ok" in md.files else np.ones(COS.shape[0])
            R = c["reversal_R"]
            mask = eok[:len(R)] > 0.5
            COSm = COS[:len(R)][mask]
            Rm = R[mask]
            Rb = np.repeat(Rm[:, None], COSm.shape[1], axis=1)
            wp = within_probe_rhos(COSm, Rb)
            rho = float(np.nanmean(wp))
            arch_h4[c["arch"]].append(rho)
        h4["per_cell"][c["tag"]] = {"within_probe_rho_keycos_R": (round(rho, 4) if rho is not None else None),
                                    "mean_reversal_R": round(float(np.nanmean(c["reversal_R"])), 4)}
    for arch in ["llama", "qwen"]:
        if arch_h4[arch]:
            h4["per_arch"][arch] = round(float(np.nanmean(arch_h4[arch])), 4)

    # ---- GRACE negative control ----
    grace_vecs = glob.glob(os.path.join(VECTORS, "vectors_*grace*.npz"))
    grace = {"status": "SKIPPED (no inputs)"} if not grace_vecs else {"status": "present", "files": grace_vecs}

    # ---- Decision rule (prereg section 4, CPU gate) ----
    verdict, reasons = decide(s1, s2, h1, h3)
    report.update({"S1": s1, "S2": s2, "H1": h1, "H3": h3, "H4": h4,
                   "GRACE_control": grace, "verdict": verdict, "verdict_reasons": reasons})
    return report


def decide(s1, s2, h1, h3):
    reasons = []
    s1_llama = s1["per_arch"].get("llama", np.nan)
    s1_qwen = s1["per_arch"].get("qwen", np.nan)
    s1_pooled = s1["pooled"]
    s1_ok = (s1_pooled > 0.80) and (s1_llama > 0.75) and (s1_qwen > 0.75)
    reasons.append(f"S1: pooled={s1_pooled} llama={s1_llama} qwen={s1_qwen} -> {'ok' if s1_ok else 'fail'}")

    def h1_arch_ok(arch):
        e = h1["per_arch"].get(arch)
        if e is None:
            return False
        return (e["top1"] >= 0.60) and (e["top1"] >= 3.0 * e["chance"])

    def h1_arch_beats_3x(arch):
        e = h1["per_arch"].get(arch)
        return e is not None and e["top1"] >= 3.0 * e["chance"]

    h1_ll, h1_qw = h1_arch_ok("llama"), h1_arch_ok("qwen")
    h1_ok = h1_ll and h1_qw
    reasons.append(f"H1: llama_ok={h1_ll} qwen_ok={h1_qw}")

    h3_ok = True
    for arch in ["llama", "qwen"]:
        e = h3["per_arch"].get(arch)
        if e is None:
            h3_ok = False
            continue
        id_ok = (e["mean_rho_spike_S"] or -1) > 0
        dmg_ok = (e["mean_rho_spike_damage"] is not None) and (e["mean_rho_spike_damage"] > 0)
        h3_ok = h3_ok and id_ok and dmg_ok
    reasons.append(f"H3: {h3['per_arch']} -> {'ok' if h3_ok else 'fail'}")

    s2_ok = bool(s2.get("armn_ge_rmt", False))
    reasons.append(f"S2: armn_pooled={s2.get('armn_pooled')} >= rmt_pooled={s2.get('pooled_rmt')} -> {s2_ok}")

    # KILL conditions (evaluated first)
    kill = (not (h1_arch_beats_3x("llama") or h1_arch_beats_3x("qwen"))) or (s1_llama <= 0.70)
    if kill:
        reasons.append("KILL: H1 fails 3x chance on BOTH arch, or S1(llama) <= 0.70")
        return "KILL", reasons
    if s1_ok and h1_ok and h3_ok and s2_ok:
        return "PASS", reasons
    # GREY: H1 half-meets (one arch only), or 0.70 < S1(llama) <= 0.80
    grey = (h1_ll != h1_qw) or (0.70 < s1_llama <= 0.80)
    reasons.append("GREY: H1 half-meets (one arch only) or 0.70 < S1(llama) <= 0.80" if grey
                   else "no PASS conditions met but no explicit GREY trigger -> GREY (partial)")
    return "GREY", reasons


# --------------------------------------------------------------------------------------
# Self-test (synthetic only; reads NO real npz / model / GPU)
# --------------------------------------------------------------------------------------
def _selftest(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)
    checks = {}
    all_ok = True

    def rec(name, ok, detail=""):
        nonlocal all_ok
        checks[name] = {"ok": bool(ok), "detail": detail}
        all_ok = all_ok and bool(ok)
        print(f"[selftest] {'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)

    # (a) MP edge correctness + sane false-positive rate on a NULL Gaussian matrix
    m, n = 300, 1200
    tau_true = 0.7
    beta = m / n
    G = rng.standard_normal((m, n)) * tau_true
    sv, _ = gram_eigh(G, want_vectors=False)
    sv_edge, tau_hat = sv_edge_and_tau(sv, m, n)
    true_edge = tau_true * (np.sqrt(m) + np.sqrt(n))
    rel_edge_err = abs(sv_edge - true_edge) / true_edge
    rec("mp_edge_accuracy_null", rel_edge_err < 0.05,
        f"sv_edge={sv_edge:.3f} true={true_edge:.3f} rel_err={rel_edge_err:.4f} tau_hat={tau_hat:.4f}")
    fN_null = armn_features(sv, m, n)
    # sane FP: a pure null has near-zero summed excess and O(1) count from TW fluctuations
    rec("mp_null_low_excess", fN_null["summed_excess"] < 0.05 and fN_null["n_beyond"] <= 3,
        f"summed_excess={fN_null['summed_excess']:.5f} n_beyond={fN_null['n_beyond']} sigma1/edge={fN_null['bulk_edge_ratio']:.4f}")

    # MP median sanity (monotone, in support) at a couple of betas
    med25 = mp_median(0.25)
    rec("mp_median_in_support", (1 - 0.5) ** 2 < med25 < (1 + 0.5) ** 2, f"mp_median(0.25)={med25:.4f}")

    # (b) planted rank-one spike is detected + top vector recovered
    u = rng.standard_normal(m); u /= np.linalg.norm(u)
    v = rng.standard_normal(n); v /= np.linalg.norm(v)
    spike_mag = 6.0 * true_edge
    Wsp = G + spike_mag * np.outer(u, v)
    sv_sp, U_sp = gram_eigh(Wsp, want_vectors=True)
    fN_sp = armn_features(sv_sp, m, n)
    rec("planted_spike_detected", fN_sp["summed_excess"] > 1.0 and sv_sp[0] > sv_edge,
        f"summed_excess={fN_sp['summed_excess']:.3f} sigma1={sv_sp[0]:.3f} edge={fN_sp['sv_edge']:.3f}")
    align = abs(float(U_sp[:, 0] @ u))
    rec("planted_spike_vector_recovered", align > 0.9, f"|u1.u_plant|={align:.4f}")

    # gram_eigh singular values match a direct SVD (top spectrum)
    sv_direct = np.linalg.svd(Wsp, compute_uv=False)
    top_err = float(np.max(np.abs(sv_sp[:20] - sv_direct[:20]) / sv_direct[:20]))
    rec("gram_vs_direct_svd", top_err < 1e-4, f"max_rel_top20={top_err:.2e}")

    # edited_gram_spectrum rank-2 identity matches a from-scratch spectrum
    Wb = rng.standard_normal((m, n)) * tau_true
    a = rng.standard_normal(m) * 0.3
    b = rng.standard_normal(n) * 0.3
    G0 = Wb @ Wb.T
    sv_fast, _ = edited_gram_spectrum(G0, Wb, a, b, want_vectors=False)
    sv_ref, _ = gram_eigh(Wb + np.outer(a, b), want_vectors=False)
    rec("edited_gram_identity", float(np.max(np.abs(sv_fast[:20] - sv_ref[:20]))) < 1e-6,
        f"max_abs_top20={float(np.max(np.abs(sv_fast[:20]-sv_ref[:20]))):.2e}")

    # (c) localization: plant a spike in exactly one of several null "layers"
    n_layers = 8
    layer_scores = []
    planted = 3
    for li in range(n_layers):
        Wl = rng.standard_normal((m, n)) * tau_true
        if li == planted:
            Wl = Wl + spike_mag * np.outer(u, v)
        svl, _ = gram_eigh(Wl, want_vectors=False)
        layer_scores.append(armn_features(svl, m, n)["summed_excess"])
    top1_layer = int(np.argmax(layer_scores))
    rec("localization_picks_planted", top1_layer == planted,
        f"argmax_layer={top1_layer} planted={planted} scores={[round(x,3) for x in layer_scores]}")

    # H3 identity: singular value of outer(a,b) equals ||a||*||b|| exactly
    sv_ab = np.linalg.svd(np.outer(a, b), compute_uv=False)[0]
    rec("rank1_singular_identity", abs(sv_ab - np.linalg.norm(a) * np.linalg.norm(b)) < 1e-6,
        f"sv={sv_ab:.6f} ||a||*||b||={np.linalg.norm(a)*np.linalg.norm(b):.6f}")

    # AUROC helper: perfect separation -> 1.0; reversed -> 0.0; tie -> 0.5
    rec("auroc_perfect", abs(auroc(np.array([3, 2, 1, 0.0]), np.array([1, 1, 0, 0])) - 1.0) < 1e-9)
    rec("auroc_reversed", abs(auroc(np.array([0, 1, 2, 3.0]), np.array([1, 1, 0, 0])) - 0.0) < 1e-9)

    # spearman monotone sanity
    rec("spearman_monotone", abs(spearman(np.arange(10), 2 * np.arange(10) + 1) - 1.0) < 1e-9)

    report = {"verdict": "SELFTEST-PASS" if all_ok else "SELFTEST-FAIL",
              "all_ok": all_ok, "checks": checks}
    out_json = os.path.join(out_dir, "SELFTEST_report.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[selftest] {report['verdict']} -> {out_json}", flush=True)
    return all_ok, report


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="T1.2 tamper-detection E0 Phase-0-CPU kill-gate (CPU-only).")
    ap.add_argument("--selftest", action="store_true",
                    help="synthetic self-test (MP edge / planted spike / localization / "
                         "identities); reads NO real npz/model/GPU; exits nonzero on failure")
    ap.add_argument("--selftest_out", default=os.path.join(OUT_DIR, "selftest"),
                    help="directory for the selftest report")
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "E0_report.json"),
                    help="output report path")
    ap.add_argument("--limit_edits", type=int, default=None,
                    help="cap edits/cell (debug only; default = all 200 per the prereg)")
    args = ap.parse_args()

    if args.selftest:
        ok, _ = _selftest(args.selftest_out)
        sys.exit(0 if ok else 1)

    os.makedirs(OUT_DIR, exist_ok=True)
    report = run_gate(limit_edits=args.limit_edits)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else o)

    # human-readable summary
    print("\n===== T1.2 TAMPER E0 — Phase-0-CPU =====", flush=True)
    print(f"cells: {report['cells']}")
    print(f"S1 ARM-N AUROC   pooled={report['S1']['pooled']}  per-arch={report['S1']['per_arch']}")
    print(f"   ARM-R ceiling per-arch={report['S1']['armr_ceiling']}")
    if "logistic_secondary" in report["S1"]:
        ls = report["S1"]["logistic_secondary"]
        print(f"   logistic secondary: train(llama)={ls['train_auroc']} heldout(qwen)={ls['heldout_auroc']}")
    print(f"S2 RMT-null AUROC pooled={report['S2']['pooled_rmt']}  ARM-N>=RMT: {report['S2']['armn_ge_rmt']}")
    print(f"H1 localization per-arch={report['H1']['per_arch']}")
    print(f"H3 fold-in per-arch={report['H3']['per_arch']}")
    print(f"H4 (secondary) per-arch={report['H4']['per_arch']}")
    print(f"GRACE control: {report['GRACE_control']['status']}")
    for r in report["verdict_reasons"]:
        print(f"   - {r}")
    print(f"\nVERDICT: {report['verdict']}  ->  {args.out}\n", flush=True)


if __name__ == "__main__":
    main()
