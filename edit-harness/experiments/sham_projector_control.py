"""sham_projector_control.py — DEFECTIVE, DO NOT USE. Results withdrawn 2026-07-26.

!!!! REJECTED BY HOSTILE REVIEW (2026-07-26). The first-order proxy below is
!!!! mathematically degenerate and its outputs in results/sham_control/ are WITHDRAWN.
!!!! Defect (verified independently, residual 1.8e-15 for both structured and random
!!!! projectors): in `predicted_removed`, the ROME denominator rescale kᵀk→kᵀPk exactly
!!!! undoes the along-key projection it models —
!!!!       proj_par = |diag·COS| · (1/|diag|) · ‖A‖ = |COS|·‖A‖ = raw_hit
!!!! — so the output is a projector-FREE monotone function of raw_hit for every P, and
!!!! the "real vs sham" comparison carries no information. Two further blockers:
!!!! predicted_removed is negative for 100% of entries (wrong sign vs ~98% measured
!!!! removal), and the --smoke gate passed on a planted correlation alone (it would
!!!! pass with the proxy deleted).
!!!! Anyone reviving this must model P's effect on ΔW·k_p through the full product
!!!! WITHOUT the self-cancelling rescale — and even then the nonlinearity that produces
!!!! the measured gap is not captured. The replacement is the GPU-level sham (rerun
!!!! AlphaEdit with config["projector"] = a rank-matched random projector); spec in
!!!! run_b6ins.sh comments. Full record: submissions/ieee/revision/SHAM-CONTROL-READOUT-20260726.md
!!!! Kept on disk for provenance only.

Original (now-invalid) intent follows.

sham_projector_control.py — is the AlphaEdit causal result algebraic or empirical?

Referee objection (B6@TETCI, top predicted reject trigger): AlphaEdit projects the
ROME update off the preserved-key subspace, so "damage-removed ∝ key-cosine" could be
a restatement of the update algebra under a projector, not an empirical finding.

Control design (two layers):

1. ALGEBRAIC LAYER (this script, CPU, runs on saved artifacts):
   For each ROME edit we have the exact rank-one factors from --save_vectors dumps
   (results/vectors/vectors_qv_llama1b_rome_cf_L{L}_s{s}.npz: K [E,d_in] edit keys,
   A [E,d_out], B [E,d_in] with ΔW_i = A_i ⊗ B_i, Wbase) and the probe-bank key
   cosines + measured damage (results/matrices/g4_llama1b_{rome,alphaHO}_cf_L{L}_s{s}.npz:
   COS [E,P], damage_logit [E,P]).
   - Rebuild the REAL null-space projector P_real from the probe keys the way
     editors/alphaedit.build_null_projector does (eigh of K_pᵀK_p, keep_ratio energy).
     We do not have the raw probe keys here, so P_real is fit on the EDIT key bank K —
     an in-distribution stand-in fit on the same layer's key geometry (edit keys and
     probe keys are drawn from the same key-space; the paper's holdout cells make the
     same move with a disjoint bank).
   - Build N_SHAM random projectors matched in REMOVED RANK to P_real: P_sham =
     I − U Uᵀ with U a random orthonormal basis of the same rank r (optionally
     cosine-matched to the edit-key distribution via Procrustes-free rejection).
   - First-order damage proxy for projector P on edit i, probe p:
       raw ROME hit:    h_ip  = |B_i · k̂_p|            (along-key transfer of ΔW_i)
       projected hit:   h'_ip = |(P B_i) · k̂_p| · s_i   (s_i = kᵀk / kᵀPk rescale)
       predicted-removed_ip = h_ip − h'_ip
     Within-probe Spearman ρ(COS_ip, predicted-removed_ip) is what ALGEBRA ALONE
     forces. We report it for P_real and the sham distribution.
   - EMPIRICAL COMPARISON: the measured within-probe ρ(COS, damage_removed) from the
     matrices (rome − alphaHO damage). Verdict logic:
       * If measured ρ ≈ algebraic-real ρ AND sham ρ ≈ 0: the geometry-tracking is
         carried by the preserved-key structure of the projector, not by projection
         per se — the causal claim is empirical (a random rank-matched projector
         would NOT reproduce it), and the algebraic prediction is CONFIRMED by the
         full nonlinear forward pass rather than presupposed.
       * If sham ρ ≈ real ρ: the result IS a projection tautology → objection stands.
   The JSON output states both numbers side by side; the paper's rebuttal paragraph
   quotes the sham percentile.

2. EMPIRICAL LAYER (GPU, run_b6ins.sh): the existing killgate --alpha_proj_source
   holdout cells at L10/L14 close the "holdout only exists at L8/L12" gap. A full
   GPU sham (rerunning edits with a random projector through the model) is specced
   in run_b6ins.sh comments as an optional escalation if a reviewer demands it.

Usage:
  python experiments/sham_projector_control.py --smoke          # synthetic self-test
  python experiments/sham_projector_control.py --layer 8 --seed 0
  python experiments/sham_projector_control.py --all            # L8/L12 × s0/1/2
Outputs: results/sham_control/SHAM_projector_control_L{L}_s{s}.json (+ _summary.json
with --all).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HARNESS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HARNESS)
sys.path.insert(0, os.path.join(HARNESS, "experiments"))

N_SHAM_DEFAULT = 20
KEEP_RATIO = 0.99  # match editors/alphaedit.py build_null_projector default


# ---------------------------------------------------------------- shared stats
def _rankdata(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(a), dtype=np.float64)
    # average ties
    sa = a[order]
    i = 0
    while i < len(sa):
        j = i
        while j + 1 < len(sa) and sa[j + 1] == sa[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    ra, rb = _rankdata(a[m]), _rankdata(b[m])
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def within_probe_rho(COS: np.ndarray, D: np.ndarray) -> float:
    """Mean over probes of Spearman(COS[:,p], D[:,p]) — the paper's within-probe
    statistic (matches analyze_matrices.within_probe_rhos convention: correlate
    ACROSS edits within each probe column, then average)."""
    rhos = [spearman(COS[:, p], D[:, p]) for p in range(COS.shape[1])]
    rhos = [r for r in rhos if np.isfinite(r)]
    return float(np.mean(rhos)) if rhos else float("nan")


# ---------------------------------------------------------------- projectors
def build_null_projector_np(K: np.ndarray, keep_ratio: float = KEEP_RATIO) -> tuple[np.ndarray, int]:
    """NumPy port of editors/alphaedit.build_null_projector (eigh on KᵀK,
    remove top-r energy directions). Returns (P, r)."""
    K = K.astype(np.float64)
    C = K.T @ K
    evals, evecs = np.linalg.eigh(C)          # ascending
    total = max(float(evals.sum()), 1e-12)
    desc = evals[::-1]
    cum = np.cumsum(desc) / total
    r = int((cum < keep_ratio).sum()) + 1
    r = max(1, min(r, evecs.shape[1] - 1))
    return _proj_from_basis(evecs[:, -r:]), r


def _proj_from_basis(U: np.ndarray) -> np.ndarray:
    d = U.shape[0]
    return np.eye(d) - U @ U.T


def sham_projector(d: int, r: int, rng: np.random.Generator) -> np.ndarray:
    """Random rank-matched projector: remove a uniformly-random r-dim subspace."""
    G = rng.standard_normal((d, r))
    Q, _ = np.linalg.qr(G)
    return _proj_from_basis(Q[:, :r])


# ---------------------------------------------------------------- core analysis
def analyze(K: np.ndarray, A: np.ndarray, B: np.ndarray, COS: np.ndarray,
            removed_measured: np.ndarray | None, n_sham: int, seed: int,
            keep_ratio: float = KEEP_RATIO) -> dict:
    """K [E,d] edit keys; A [E,d_out], B [E,d] rank-one factors (ΔW_i = A_i⊗B_i);
    COS [E,P] edit-key/probe-key cosines; removed_measured [E,P] or None."""
    E, d = K.shape
    rng = np.random.default_rng(seed)

    P_real, r = build_null_projector_np(K, keep_ratio)

    # First-order along-key hit: probe p's key k̂_p is unobservable here, but
    # COS[i,p] = k̂_i·k̂_p and B_i ∝ k̂_i for ROME (B is the kᵀ/(kᵀk) side).
    # For a projector P the transferred component along k̂_p is
    #   |(P k̂_i)·k̂_p| ≈ |k̂_p·(P k̂_i)|. Decompose k̂_p = COS[i,p]·k̂_i + orth.
    # The k̂_i-parallel part is exact from COS; the orthogonal part is estimated
    # in expectation over the probe sphere: E|orth·(P k̂_i − (k̂_iᵀP k̂_i)k̂_i)|.
    Kn = K / np.maximum(np.linalg.norm(K, axis=1, keepdims=True), 1e-12)

    def predicted_removed(P: np.ndarray) -> np.ndarray:
        PK = Kn @ P.T                                   # [E,d] P·k̂_i
        diag = np.einsum("ed,ed->e", Kn, PK)            # k̂_iᵀ P k̂_i
        # rescale ROME denominator kᵀk→kᵀPk (AlphaEdit update: (v−Wk)(Pk)ᵀ/(kᵀPk))
        scale = 1.0 / np.maximum(np.abs(diag), 1e-6)
        # parallel component transferred to probe p: diag_i * COS[i,p] * scale_i
        # raw ROME transfers COS[i,p] itself (P=I ⇒ diag=1, scale=1)
        anorm = np.linalg.norm(A, axis=1)               # per-edit output magnitude
        raw_hit = np.abs(COS) * anorm[:, None]
        proj_par = np.abs(diag[:, None] * COS) * (scale * anorm)[:, None]
        # orthogonal leakage: ||P k̂_i − diag_i k̂_i|| spread isotropically over the
        # remaining d−1 dims; its expected projection on any fixed unit probe dir
        # scales ~ leak/√d — subdominant, include for honesty
        leak = np.linalg.norm(PK - diag[:, None] * Kn, axis=1)
        proj_orth = (leak * scale * anorm / np.sqrt(max(d - 1, 1)))[:, None]
        proj_hit = np.sqrt(proj_par ** 2 + proj_orth ** 2)
        return raw_hit - proj_hit                       # predicted damage-removed

    rem_real = predicted_removed(P_real)
    rho_alg_real = within_probe_rho(COS, rem_real)

    sham_rhos = []
    for _ in range(n_sham):
        Ps = sham_projector(d, r, rng)
        sham_rhos.append(within_probe_rho(COS, predicted_removed(Ps)))
    sham_rhos = np.array(sham_rhos, dtype=np.float64)

    out = {
        "n_edits": int(E), "dim": int(d), "removed_rank_r": int(r),
        "keep_ratio": keep_ratio, "n_sham": int(n_sham),
        "rho_algebraic_real_projector": round(rho_alg_real, 4),
        "rho_sham_mean": round(float(np.nanmean(sham_rhos)), 4),
        "rho_sham_sd": round(float(np.nanstd(sham_rhos)), 4),
        "rho_sham_p95": round(float(np.nanpercentile(sham_rhos, 95)), 4),
        "real_percentile_in_sham": round(
            float((sham_rhos < rho_alg_real).mean() * 100.0), 1),
    }
    if removed_measured is not None:
        rho_meas = within_probe_rho(COS, removed_measured)
        out["rho_measured_damage_removed"] = round(rho_meas, 4)
        out["measured_minus_algebraic"] = round(rho_meas - rho_alg_real, 4)
    out["interpretation"] = (
        "If rho_measured ≈ rho_algebraic_real AND sham stays near its null, the "
        "geometry-tracking is carried by the preserved-key structure (empirical "
        "under the full forward pass, not reproduced by rank-matched random "
        "projectors). If sham ≈ real, the projection step alone forces the "
        "correlation and the objection stands.")
    return out


# ---------------------------------------------------------------- data loading
def load_cell(layer: int, seed: int) -> tuple[np.ndarray, ...]:
    vec_p = os.path.join(HARNESS, "results/vectors",
                         f"vectors_qv_llama1b_rome_cf_L{layer}_s{seed}.npz")
    rome_p = os.path.join(HARNESS, "results/matrices",
                          f"gate_llama1b_rome_cf_L{layer}_s{seed}.npz")
    ho_p = os.path.join(HARNESS, "results/matrices",
                        f"g4_llama1b_alphaHO_cf_L{layer}_s{seed}.npz")
    for p in (vec_p, rome_p):
        if not os.path.exists(p):
            raise SystemExit(f"[sham] missing artifact: {p}")
    v = np.load(vec_p, allow_pickle=True)
    r = np.load(rome_p, allow_pickle=True)
    K, A, B = v["K"].astype(np.float64), v["A"].astype(np.float64), v["B"].astype(np.float64)
    COS, dmg_rome = r["COS"].astype(np.float64), r["damage_logit"].astype(np.float64)
    removed = None
    if os.path.exists(ho_p):
        h = np.load(ho_p, allow_pickle=True)
        removed = dmg_rome - h["damage_logit"].astype(np.float64)
    # filter to valid edits (edit_ok on the ROME side, matching aggregate_g4_causal)
    ok = r["edit_ok"].astype(bool)
    return K[ok], A[ok], B[ok], COS[ok], (removed[ok] if removed is not None else None)


# ---------------------------------------------------------------- smoke
def smoke() -> int:
    rng = np.random.default_rng(0)
    d, E, P = 32, 40, 15
    # planted geometry: probe keys correlated with a low-dim "preserved" subspace.
    # Real MLP keys are post-activation and their pairwise cosines are mostly
    # non-negative; mimic that with abs() low-dim coordinates so the signed
    # within-probe statistic can see the planted |COS|-shaped signal.
    U_true = np.linalg.qr(rng.standard_normal((d, 6)))[0]
    K = 0.3 * rng.standard_normal((E, d)) \
        + 3.0 * np.abs(rng.standard_normal((E, 6))) @ U_true.T
    probes = 0.3 * rng.standard_normal((P, d)) \
        + 3.0 * np.abs(rng.standard_normal((P, 6))) @ U_true.T
    Kn = K / np.linalg.norm(K, axis=1, keepdims=True)
    Pn = probes / np.linalg.norm(probes, axis=1, keepdims=True)
    COS = Kn @ Pn.T
    A = rng.standard_normal((E, d))
    B = Kn.copy()
    # synthetic measured removal that follows the geometry (planted positive case)
    removed = np.abs(COS) * np.linalg.norm(A, axis=1)[:, None] * 0.8 \
        + 0.05 * rng.standard_normal((E, P))
    res = analyze(K, A, B, COS, removed, n_sham=10, seed=1)
    print("[smoke]", json.dumps(res, indent=2))
    ok = (np.isfinite(res["rho_algebraic_real_projector"])
          and np.isfinite(res["rho_sham_mean"])
          and res["removed_rank_r"] >= 1
          and res["rho_measured_damage_removed"] > 0.3)
    print(f"[smoke] {'PASS' if ok else 'FAIL'}: real-projector algebraic rho="
          f"{res['rho_algebraic_real_projector']}, sham mean={res['rho_sham_mean']}, "
          f"measured(planted)={res['rho_measured_damage_removed']}")
    return 0 if ok else 1


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--all", action="store_true", help="L8/L12 × s0/1/2")
    ap.add_argument("--n_sham", type=int, default=N_SHAM_DEFAULT)
    ap.add_argument("--out_dir", default=os.path.join(HARNESS, "results/sham_control"))
    args = ap.parse_args()

    if args.smoke:
        return smoke()

    cells = ([(l, s) for l in (8, 12) for s in (0, 1, 2)] if args.all
             else [(args.layer, args.seed)])
    if cells[0][0] is None:
        ap.error("--layer required (or --smoke / --all)")
    os.makedirs(args.out_dir, exist_ok=True)

    summary = {}
    for layer, seed in cells:
        K, A, B, COS, removed = load_cell(layer, seed)
        res = analyze(K, A, B, COS, removed, n_sham=args.n_sham, seed=1000 + seed)
        res["cell"] = f"L{layer}_s{seed}"
        outp = os.path.join(args.out_dir, f"SHAM_projector_control_L{layer}_s{seed}.json")
        with open(outp, "w") as f:
            json.dump(res, f, indent=2)
        print(f"[sham] L{layer} s{seed}: algebraic_real="
              f"{res['rho_algebraic_real_projector']} sham_mean={res['rho_sham_mean']} "
              f"measured={res.get('rho_measured_damage_removed')} -> {outp}")
        summary[f"L{layer}_s{seed}"] = res
    if args.all:
        sp = os.path.join(args.out_dir, "SHAM_projector_control_summary.json")
        with open(sp, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[sham] summary -> {sp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
