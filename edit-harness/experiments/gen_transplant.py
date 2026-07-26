"""gen_transplant.py — D3 / T1.3-E0b generational edit TRANSPLANT.

Prereg: docs/plans/PREREG-D3-TRANSPLANT-E0B-20260714.md. Read §1 (THE CORE DESIGN
DECISION) before touching this file.

WHAT IS TRANSPLANTED (and what is not) — the load-bearing decision:
  A ROME edit is the rank-one update  DW = r k^T / (k.k),  r = v - Wk.
    * r (the residual/value correction, dim = HIDDEN = down_proj OUTPUT) is the edit's
      semantic content. HIDDEN matches across the Qwen2.5-14B -> Qwen3-14B generation
      (5120/5120), and the residual stream shares the token-embedding basis, so r is
      mapped across the generation by an orthogonal Procrustes rotation W_align fit on
      SHARED-VOCAB TOKEN EMBEDDINGS (both HIDDEN-dim, directly paired).
    * k (the key, dim = INTERMEDIATE = down_proj INPUT) is a model-specific localization
      whose dim does NOT even match across the generation (13824 vs 17408). It carries no
      semantic content and is cheap to recompute, so it is RE-DERIVED NATIVELY on the
      recipient (_capture_key at the subject's last token). The key space is NOT mapped.
  Install on the recipient:  DW_r = r_map k_r^T / (k_r.k_r),  r_map = W_align^T r_d
  -> (W_r + DW_r) k_r = W_r k_r + r_map  (recipient's own baseline + the transplanted
  correction). This is the RESIDUAL-transplant (map the correction r, not the absolute
  target v = r + Wk, because baselines W k differ across models).

MODES:
  --selftest              pure-math synthetic validation (CPU, deterministic). No models,
                          no downloads. Writes only under results/transplant/selftest/.
                          Prints "ALL CHECKS PASSED" on success (the driver greps for it).
  --phase donor           load DONOR fp32, dump per-edit residuals r_d + donor candidate-
                          anchor embeddings -> donor_bank.npz. Weights restored each edit.
  --phase recipient       load RECIPIENT fp32, read donor_bank.npz, fit Procrustes on
                          shared-vocab anchors, install every condition per edit, measure
                          transplant esr + the per-edit fidelity predictor -> result JSON+npz.
  --phase all             donor then (explicit unload) recipient in one process (for small
                          pairs / local testing only; 14B pair needs the two-phase driver).

PRECISION: value-optimization is fp32 ONLY (ROME fp16/bf16 value-opt silently NaNs). Two
14B fp32 models do not co-reside in 96GB -> sequential load/unload is why donor/recipient
are separate phases with a disk bank between them.

CPU-only for --selftest; GPU (cuda, fp32) for the donor/recipient phases.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

HARNESS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HARNESS)
sys.path.insert(0, os.path.join(HARNESS, "experiments"))

# ------------------------------------------------------------------ pure math (import-light)
# analyze_matrices is numpy-only; safe to import even in --selftest.
from analyze_matrices import spearman, within_probe_rhos  # noqa: E402

RNG_SEED = 12345  # fixed permutation-null seed (matches analyze_matrices)


def orthogonal_procrustes(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Orthogonal W minimizing ||A W - B||_F, fp64. Row-vector convention: donor anchor row
    a maps to recipient row a @ W. Returns W [d, d].

    Solution: M = A^T B, SVD M = U S V^T, W = U V^T. For orthonormal-column A (A^T A = I),
    W recovers a planted rotation EXACTLY (selftest check 1).
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    M = A.T @ B
    U, _s, Vt = np.linalg.svd(M, full_matrices=False)
    return U @ Vt


def map_residual(W_align: np.ndarray, r_donor: np.ndarray) -> np.ndarray:
    """Map a donor residual COLUMN vector r_donor [H] to recipient space.

    Row convention (orthogonal_procrustes): donor row a -> a @ W. A column vector r has
    row form r^T, mapped to (r^T W), i.e. as a column: W^T r. Returns [H]."""
    return np.asarray(W_align, dtype=np.float64).T @ np.asarray(r_donor, dtype=np.float64)


def rank_one_delta(r: np.ndarray, k: np.ndarray) -> np.ndarray:
    """DW = outer(r, k) / (k.k + 1e-8), [H, I]. Installing it makes (W+DW)k = Wk + r."""
    r = np.asarray(r, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    denom = float(k @ k) + 1e-8
    return np.outer(r, k) / denom


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float((a @ b) / (na * nb))


def sample_random_orthogonal(d: int, seed: int) -> np.ndarray:
    """Haar-ish random orthogonal [d, d] via QR of a seeded Gaussian (sign-fixed)."""
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((d, d))
    Q, R = np.linalg.qr(G)
    # fix reflection ambiguity so Q is deterministic given the seed
    Q = Q * np.sign(np.diag(R))
    return Q


def spearman_perm_p(x: np.ndarray, y: np.ndarray, n_perm: int = 10000,
                    seed: int = RNG_SEED) -> Tuple[float, float]:
    """Spearman(x, y) and a permutation p (two-sided, shuffle y). NaN-safe."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 5 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan"), float("nan")
    obs = spearman(x, y)
    if not np.isfinite(obs):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    yc = y.copy()
    ge = 0
    for _ in range(n_perm):
        rng.shuffle(yc)
        s = spearman(x, yc)
        if np.isfinite(s) and abs(s) >= abs(obs):
            ge += 1
    return float(obs), float((ge + 1) / (n_perm + 1))


# ------------------------------------------------------------------ shared-vocab anchors
def shared_vocab_anchor_ids(tok_d, tok_r, cand_range: int) -> List[int]:
    """Ids in [0, cand_range) that decode to the SAME token string in both tokenizers.

    convert_ids_to_tokens is the raw-token (pre-detokenization) view -> a stable string
    identity that does not depend on surrounding context. cand_range caps the cost of the
    embedding block dumped in the donor phase."""
    hi = min(cand_range, len(tok_d), len(tok_r))
    ids = list(range(hi))
    td = tok_d.convert_ids_to_tokens(ids)
    tr = tok_r.convert_ids_to_tokens(ids)
    return [i for i, (a, b) in enumerate(zip(td, tr)) if a is not None and a == b]


# ================================================================== SELFTEST (pure math)
def _selftest() -> int:
    print("[gen_transplant selftest] pure-math synthetic validation (CPU, deterministic)")
    checks: List[Tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = ""):
        checks.append((name, bool(ok), detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")

    rng = np.random.default_rng(0)

    # --- check 1: recover a KNOWN rotation from orthonormal-column anchors (exact) ---
    n, d = 2000, 16
    A0 = rng.standard_normal((n, d))
    A, _ = np.linalg.qr(A0)          # A: [n, d] with orthonormal COLUMNS (A^T A = I_d)
    R_true = sample_random_orthogonal(d, seed=7)
    B = A @ R_true
    W = orthogonal_procrustes(A, B)
    recon = float(np.linalg.norm(A @ W - B))
    rot_err = float(np.linalg.norm(W - R_true))
    check("rotation_recovery_exact", np.allclose(W, R_true, atol=1e-6) and recon < 1e-6,
          f"||W-R_true||={rot_err:.2e} recon={recon:.2e}")

    # --- check 2: rank-one residual-transplant install identity ---
    H, I = 5120 // 10, 13824 // 10   # smaller but same shape class
    W_r = rng.standard_normal((H, I))
    k = rng.standard_normal(I)
    r = rng.standard_normal(H)
    DW = rank_one_delta(r, k)
    lhs = (W_r + DW) @ k
    rhs = W_r @ k + r
    inst_err = float(np.linalg.norm(lhs - rhs))
    check("rank_one_install_identity", np.allclose(lhs, rhs, atol=1e-8),
          f"||(W+DW)k - (Wk+r)||={inst_err:.2e}")

    # --- check 3: identity beats random-rotation on a PLANTED, ALIGNED signal ---
    # aligned bases (R_true = I): anchors identical -> Procrustes ~ I -> mapped == donor.
    n2, H2 = 4000, 64
    E = rng.standard_normal((n2, H2))
    W_al = orthogonal_procrustes(E, E)                 # should be ~ identity
    ident_to_I = float(np.linalg.norm(W_al - np.eye(H2)))
    N_edit = 200
    r_d = rng.standard_normal((N_edit, H2))
    r_target = r_d.copy()                              # aligned -> recipient wants the same r
    # fidelity = mean_i cos(map(r_i), target_i)
    fid_identity = float(np.mean([_cos(r_d[i], r_target[i]) for i in range(N_edit)]))
    fid_proc = float(np.mean([_cos(map_residual(W_al, r_d[i]), r_target[i])
                              for i in range(N_edit)]))
    rand_fids = []
    for j in range(5):
        Rj = sample_random_orthogonal(H2, seed=100 + j)
        rand_fids.append(float(np.mean(
            [_cos(map_residual(Rj, r_d[i]), r_target[i]) for i in range(N_edit)])))
    rand_mean = float(np.mean(rand_fids)); rand_std = float(np.std(rand_fids))
    check("procrustes_recovers_identity_when_aligned", ident_to_I < 1e-6,
          f"||W_align - I||={ident_to_I:.2e}")
    check("identity_beats_random_planted", (fid_identity - rand_mean) > 0.5,
          f"identity={fid_identity:.3f} rand_mean={rand_mean:.3f}(+-{rand_std:.3f})")
    check("procrustes_beats_random_planted", (fid_proc - rand_mean) > 0.5,
          f"procrustes={fid_proc:.3f} rand_mean={rand_mean:.3f}")

    # --- check 4: orthogonality receipt on a NON-trivial (noised) fit ---
    R2 = sample_random_orthogonal(H2, seed=3)
    Bn = E @ R2 + 0.01 * rng.standard_normal((n2, H2))
    Wn = orthogonal_procrustes(E, Bn)
    orth_err = float(np.linalg.norm(Wn @ Wn.T - np.eye(H2)))
    check("procrustes_orthogonality", orth_err < 1e-6, f"||WW^T - I||={orth_err:.2e}")

    # --- check 5: fidelity predictor / perm-p plumbing (a planted monotone relation) ---
    xg = rng.standard_normal(120)
    yg = (xg + 0.3 * rng.standard_normal(120) > 0).astype(float)  # binary success ~ tracks x
    rho, p = spearman_perm_p(xg, yg, n_perm=2000, seed=1)
    check("predictor_spearman_perm_plumbing", np.isfinite(rho) and np.isfinite(p) and rho > 0.2,
          f"rho={rho:.3f} p={p:.4f}")

    # --- check 6 (reviewer): map_residual ORIENTATION — a residual maps by W^T, and using the
    #     TRANSPOSED W (i.e. W r) is a detectably different vector. Anchors B = A @ R_true make
    #     W_align == R_true; the correct recipient image of a donor column r is R_true^T r =
    #     map_residual(W_align, r). Using W r (= R_true r) must NOT match — catches a flipped map. ---
    d6 = 16
    A6, _ = np.linalg.qr(rng.standard_normal((400, d6)))     # orthonormal columns
    R6 = sample_random_orthogonal(d6, seed=11)
    W6 = orthogonal_procrustes(A6, A6 @ R6)                  # == R6
    r6 = rng.standard_normal(d6)
    true_image = R6.T @ r6                                   # correct recipient image
    good = map_residual(W6, r6)                              # W6^T r6  -> must equal true_image
    transposed = W6 @ r6                                     # the WRONG orientation -> must differ
    check("map_residual_orientation",
          np.allclose(good, true_image, atol=1e-10) and not np.allclose(transposed, true_image, atol=1e-6),
          f"||good-true||={np.linalg.norm(good-true_image):.2e} "
          f"||transposed-true||={np.linalg.norm(transposed-true_image):.2e} (must be >>0)")

    out_dir = os.path.join(HARNESS, "results", "transplant", "selftest")
    os.makedirs(out_dir, exist_ok=True)
    rec = {"checks": [{"name": n, "pass": ok, "detail": det} for n, ok, det in checks],
           "all_pass": all(ok for _, ok, _ in checks)}
    tmp = os.path.join(out_dir, "gen_transplant_selftest.json.tmp")
    with open(tmp, "w") as f:
        json.dump(rec, f, indent=2)
    os.replace(tmp, os.path.join(out_dir, "gen_transplant_selftest.json"))

    if rec["all_pass"]:
        print("ALL CHECKS PASSED")
        return 0
    print("SELFTEST FAILED")
    return 1


# ================================================================== real GPU phases
def _load_model(path: str, device: str):
    """fp32 always (ROME value-opt NaNs in fp16/bf16)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).to(device).eval()
    return model, tok


def _resolve_layer(model, layer_arg: str) -> int:
    nL = model.config.num_hidden_layers
    return nL // 2 if layer_arg == "auto" else int(layer_arg)


def phase_donor(args) -> int:
    """Capture per-edit donor residuals r_d + donor candidate-anchor embeddings."""
    import torch
    from experiments.killgate_keygeom import load_counterfact
    from editors.rome_native import apply_edit
    from metrics import efficacy

    t0 = time.time()
    model, tok = _load_model(args.donor, args.device)
    L = _resolve_layer(model, args.layer_donor)
    H = int(model.config.hidden_size)
    I = int(model.model.layers[L].mlp.down_proj.weight.shape[1])
    # shared loader returns a 3rd (E6 holdout) bank — absorb extras (mechanism_dump precedent)
    edits, _probes, *_ = load_counterfact(args.data, args.n_edits, args.n_probes, args.seed)
    print(f"[donor] {args.donor} L{L} H={H} I={I} n_edits={len(edits)} "
          f"{time.time()-t0:.1f}s", flush=True)

    W = model.model.layers[L].mlp.down_proj.weight
    W_base = W.detach().clone()
    cfg = {"layer": L, "steps": args.steps, "lr": args.lr, "v_weight_decay": args.v_weight_decay}

    r_donor = np.full((len(edits), H), np.nan, dtype=np.float32)
    donor_edit_ok = np.zeros(len(edits), dtype=np.float32)
    for i, e in enumerate(edits):
        info = apply_edit(model, tok, e, cfg, args.device)   # applies DW to W in place
        r_donor[i] = np.asarray(info["residual_vec"], dtype=np.float32)
        # efficacy of the donor's OWN edit BEFORE restoring (W is still edited here)
        eff = efficacy(model, tok, e["prompt"], e["target_new"], e.get("target_true"), args.device)
        donor_edit_ok[i] = eff["success"]
        with torch.no_grad():
            W.copy_(W_base)   # restore -> every edit sees the base model
        if (i + 1) % 20 == 0:
            print(f"[donor] edit {i+1}/{len(edits)} {time.time()-t0:.1f}s", flush=True)

    # donor candidate-anchor embedding block (rows 0..cand_range) for Procrustes in phase 2
    emb = model.get_input_embeddings().weight.detach().float().cpu().numpy()
    hi = min(args.anchor_cand, emb.shape[0])
    donor_anchor_emb = emb[:hi].astype(np.float32)

    os.makedirs(os.path.dirname(args.donor_bank) or ".", exist_ok=True)
    # np.savez_compressed appends ".npz" to a path lacking that suffix -> write to an
    # explicit tmp base and rename the realized file atomically into place.
    tmp_base = args.donor_bank + ".tmp"
    np.savez_compressed(
        tmp_base,
        r_donor=r_donor, donor_edit_ok=donor_edit_ok,
        donor_anchor_emb=donor_anchor_emb,
        meta=np.array(json.dumps({
            "donor": args.donor, "layer_donor": L, "H": H, "I_donor": I,
            "n_edits": len(edits), "seed": args.seed, "anchor_cand": hi,
            "steps": args.steps, "lr": args.lr,
        }), dtype="U4096"),
    )
    os.replace(tmp_base + ".npz", args.donor_bank)
    print(f"[donor] wrote bank -> {args.donor_bank} ({time.time()-t0:.1f}s)", flush=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 0


def _install_measure_efficacy(model, tok, W, W_base, L, e, r_map, k_r, device) -> float:
    """Install DW_r = outer(r_map,k_r)/(k_r.k_r) at layer L, return per-edit success, restore."""
    import torch
    from metrics import efficacy
    DW = torch.tensor(rank_one_delta(r_map, k_r), dtype=W.dtype, device=W.device)
    with torch.no_grad():
        W.add_(DW)
    try:
        eff = efficacy(model, tok, e["prompt"], e["target_new"], e.get("target_true"), device)
        succ = float(eff["success"])
    finally:
        with torch.no_grad():
            W.copy_(W_base)
    return succ


def phase_recipient(args) -> int:
    import torch
    from experiments.killgate_keygeom import (
        load_counterfact, find_subject_last_token_index, _capture_key, prob_of_token)
    from editors.rome_native import apply_edit
    from metrics import first_target_token_id
    from transformers import AutoTokenizer

    t0 = time.time()
    bank = np.load(args.donor_bank, allow_pickle=False)
    dmeta = json.loads(str(bank["meta"]))
    r_donor = bank["r_donor"].astype(np.float64)
    donor_edit_ok = bank["donor_edit_ok"].astype(float)
    donor_anchor_emb = bank["donor_anchor_emb"].astype(np.float64)
    tok_d = AutoTokenizer.from_pretrained(args.donor)

    model, tok = _load_model(args.recipient, args.device)
    L = _resolve_layer(model, args.layer_recip)
    H = int(model.config.hidden_size)
    I = int(model.model.layers[L].mlp.down_proj.weight.shape[1])

    # ---- preflight (hard) ----
    if int(dmeta["H"]) != H:
        raise SystemExit(f"[recip] HIDDEN mismatch: donor H={dmeta['H']} != recipient H={H} "
                         "— residual transplant is ill-posed; aborting")
    print(f"[recip] {args.recipient} L{L} H={H} I_recip={I} (I_donor={dmeta['I_donor']}) "
          f"{time.time()-t0:.1f}s", flush=True)
    if I != int(dmeta["I_donor"]):
        print(f"[recip] NOTE key dims differ ({dmeta['I_donor']} vs {I}) — keys re-derived "
              "natively (expected; see prereg §1)", flush=True)

    edits, probes, *_ = load_counterfact(args.data, args.n_edits, args.n_probes, args.seed)
    if len(edits) != int(dmeta["n_edits"]):
        raise SystemExit(f"[recip] ALIGNMENT MISMATCH: {len(edits)} edits vs bank "
                         f"{dmeta['n_edits']} — check seed/n_edits/data")

    # ---- shared-vocab anchors + Procrustes ----
    cand = int(dmeta["anchor_cand"])
    anchor_ids = shared_vocab_anchor_ids(tok_d, tok, cand)
    overlap = len(anchor_ids) / max(1, cand)
    if overlap < args.min_vocab_overlap or len(anchor_ids) < args.min_anchors:
        raise SystemExit(f"[recip] shared-vocab anchors insufficient: n={len(anchor_ids)} "
                         f"overlap={overlap:.3f} (floors: n>={args.min_anchors}, "
                         f"overlap>={args.min_vocab_overlap}) — aborting")
    recip_emb = model.get_input_embeddings().weight.detach().float().cpu().numpy().astype(np.float64)
    A = donor_anchor_emb[anchor_ids]      # [n_anchor, H]
    B = recip_emb[anchor_ids]             # [n_anchor, H]
    W_align = orthogonal_procrustes(A, B)
    orth_err = float(np.linalg.norm(W_align @ W_align.T - np.eye(H)))
    recon_err = float(np.linalg.norm(A @ W_align - B) / (np.linalg.norm(B) + 1e-12))
    print(f"[recip] Procrustes fit: n_anchor={len(anchor_ids)} overlap={overlap:.3f} "
          f"orth_err={orth_err:.2e} rel_recon={recon_err:.3f}", flush=True)

    # ---- per-edit: native residual + native key, then every transplant condition ----
    W = model.model.layers[L].mlp.down_proj.weight
    W_base = W.detach().clone()
    cfg = {"layer": L, "steps": args.steps, "lr": args.lr, "v_weight_decay": args.v_weight_decay}

    N = len(edits)
    n_rand = args.n_rand
    r_native = np.full((N, H), np.nan)
    succ = {c: np.full(N, np.nan) for c in ["native", "identity", "procrustes"]}
    succ_rand = np.full((N, n_rand), np.nan)
    ca = np.full(N, np.nan)            # per-edit fidelity cos(r_map_procrustes, r_native)
    K_edit = np.full((N, I), np.nan)   # recipient native edit keys (reused by scaled + collateral)
    R_rands = [sample_random_orthogonal(H, seed=1000 + j) for j in range(n_rand)]

    for i, e in enumerate(edits):
        k_r = _capture_key(model, tok, L, e["prompt"],
                           find_subject_last_token_index(tok, e["prompt"], e.get("subject")),
                           args.device).float().cpu().numpy().astype(np.float64)
        K_edit[i] = k_r
        # native recipient residual (topline). apply_edit uses the SAME layer/subject key as
        # k_r above, so re-installing outer(rn,k_r)/(k_r.k_r) reproduces the native ROME edit
        # EXACTLY — measured via _install_measure_efficacy for an apples-to-apples esr with the
        # transplant conditions. Restore before the topline measurement so every condition below
        # starts from the base weight.
        info = apply_edit(model, tok, e, cfg, args.device)
        rn = np.asarray(info["residual_vec"], dtype=np.float64)
        r_native[i] = rn
        with torch.no_grad():
            W.copy_(W_base)
        succ["native"][i] = _install_measure_efficacy(
            model, tok, W, W_base, L, e, rn, k_r, args.device)

        r_d = r_donor[i]
        # identity (no alignment)
        succ["identity"][i] = _install_measure_efficacy(
            model, tok, W, W_base, L, e, r_d, k_r, args.device)
        # procrustes
        r_map = map_residual(W_align, r_d)
        ca[i] = _cos(r_map, rn)
        succ["procrustes"][i] = _install_measure_efficacy(
            model, tok, W, W_base, L, e, r_map, k_r, args.device)
        # random rotations
        for j, Rj in enumerate(R_rands):
            succ_rand[i, j] = _install_measure_efficacy(
                model, tok, W, W_base, L, e, Rj.T @ r_d, k_r, args.device)
        if (i + 1) % 20 == 0:
            print(f"[recip] edit {i+1}/{N} {time.time()-t0:.1f}s", flush=True)

    # ---- gate arithmetic on the donor-successful (transplantable) edit set ----
    tset = donor_edit_ok > 0.5
    def esr(a):
        v = a[tset]
        v = v[np.isfinite(v)]
        return float(np.mean(v)) if v.size else float("nan")
    esr_native = esr(succ["native"])
    esr_identity = esr(succ["identity"])
    esr_proc = esr(succ["procrustes"])
    esr_rand_per = [esr(succ_rand[:, j]) for j in range(n_rand)]
    _rf = np.array([x for x in esr_rand_per if np.isfinite(x)], dtype=float)
    esr_rand_mean = float(_rf.mean()) if _rf.size else float("nan")   # nan-safe (empty tset)
    esr_rand_std = float(_rf.std()) if _rf.size else float("nan")

    # ---- prereg §3.1 scaled-variant ablation (NON-GATING): s * W^T r_d,
    #      s = median||r_native|| / median||r_donor|| over the transplantable set. The orthogonal
    #      map is norm-preserving (||r_map|| == ||r_donor||), so s rescales the mapped correction
    #      to the recipient's native median residual norm. One extra efficacy-only pass. ----
    succ_scaled = np.full(N, np.nan)
    rn_norm = np.linalg.norm(r_native, axis=1)
    rd_norm = np.linalg.norm(r_donor, axis=1)
    fin = tset & np.isfinite(rn_norm) & np.isfinite(rd_norm)
    if int(fin.sum()) >= 1:
        s_scale = float(np.median(rn_norm[fin]) / (np.median(rd_norm[fin]) + 1e-12))
        for i, e in enumerate(edits):
            succ_scaled[i] = _install_measure_efficacy(
                model, tok, W, W_base, L, e, s_scale * map_residual(W_align, r_donor[i]),
                K_edit[i], args.device)
        scaled_variant = {"implemented": True, "esr": round(esr(succ_scaled), 4),
                          "scale_s": round(s_scale, 4),
                          "note": "orthogonal map is norm-preserving; s rescales to native median "
                                  "residual norm. NON-GATING ablation (prereg §3.1)."}
    else:
        scaled_variant = {"implemented": True, "esr": None, "scale_s": None,
                          "note": "transplantable set empty — scaled esr undefined."}

    # ---- prereg §3.6 tertiary: does a transplanted edit still obey the B6 collateral law?
    #      within-probe Spearman(|key-cos|, damage) (the E0a clean metric) on the procrustes-
    #      transplanted edits. NON-GATING. Default OFF because it adds an N x Mc probe sweep;
    #      the driver enables it on the primary pair only. ----
    collateral = {"implemented": False,
                  "note": "run with --collateral (default off; adds an N x Mc probe sweep — the "
                          "E0 gate itself is efficacy-only). Prereg §3.6, reported-later otherwise."}
    if args.collateral:
        Mc = min(args.n_collateral_probes, len(probes))
        cpr = probes[:Mc]
        ptok = [first_target_token_id(tok, p["target_true"]) for p in cpr]
        pre_l = np.array([prob_of_token(model, tok, p["prompt"], ptok[j], args.device)[1]
                          for j, p in enumerate(cpr)], dtype=np.float64)
        K_probe = np.stack([
            _capture_key(model, tok, L, p["prompt"],
                         find_subject_last_token_index(tok, p["prompt"], p.get("subject")),
                         args.device).float().cpu().numpy().astype(np.float64) for p in cpr])
        Ke = K_edit / (np.linalg.norm(K_edit, axis=1, keepdims=True) + 1e-8)
        Kp = K_probe / (np.linalg.norm(K_probe, axis=1, keepdims=True) + 1e-8)
        COS = Ke @ Kp.T                                          # [N, Mc]  edit-key vs probe-key
        damage_l = np.full((N, Mc), np.nan)
        for i, e in enumerate(edits):
            DW = torch.tensor(rank_one_delta(map_residual(W_align, r_donor[i]), K_edit[i]),
                              dtype=W.dtype, device=W.device)
            with torch.no_grad():
                W.add_(DW)
            try:
                for j, p in enumerate(cpr):
                    _, ll = prob_of_token(model, tok, p["prompt"], ptok[j], args.device)
                    damage_l[i, j] = pre_l[j] - ll
            finally:
                with torch.no_grad():
                    W.copy_(W_base)
        rhos = within_probe_rhos(np.abs(COS), damage_l)          # E0a metric on transplanted edits
        finite = rhos[np.isfinite(rhos)]
        collateral = {
            "implemented": True, "condition": "procrustes-transplanted edits", "n_probes": Mc,
            "within_probe_mean": (None if finite.size == 0 else round(float(np.mean(finite)), 4)),
            "within_probe_frac_positive": (None if finite.size == 0
                                           else round(float(np.mean(finite > 0)), 3)),
            "mean_signed_damage_logit": round(float(np.nanmean(damage_l)), 5),
            "note": "within-probe Spearman(|key-cos|, damage) on procrustes-transplanted edits; "
                    "compare to the recipient's NATIVE-edit E0a value. NON-GATING (prereg §3.6)."}

    # G3 per-edit geometry predictor (on the transplantable set)
    rho_ca, p_ca = spearman_perm_p(ca[tset], succ["procrustes"][tset],
                                   n_perm=args.n_perm, seed=RNG_SEED)

    G1 = (esr_proc - esr_rand_mean) >= args.margin and \
         esr_proc >= (esr_rand_mean + 2 * esr_rand_std)
    G2 = (esr_proc >= 0.5 * esr_native if np.isfinite(esr_native) and esr_native > 0 else False) \
         and esr_proc >= args.abs_floor
    G3 = np.isfinite(rho_ca) and rho_ca >= args.rho_floor and np.isfinite(p_ca) and p_ca < 0.05
    layer_valid = np.isfinite(esr_native) and esr_native >= args.native_esr_floor

    if not layer_valid:
        verdict = "INCONCLUSIVE_LAYER_INVALID"
    elif not G1:
        verdict = "KILL_K1_not_alignment_specific"
    elif esr_proc < args.abs_floor2:
        verdict = "KILL_K2_transplant_fails"
    elif G1 and G2 and not G3:
        verdict = "PARTIAL_PASS_no_predictor"
    elif G1 and G2 and G3:
        verdict = "PASS"
    else:
        verdict = "INCONCLUSIVE"

    identity_gap = esr_proc - esr_identity
    res: Dict[str, Any] = {
        "prereg": "docs/plans/PREREG-D3-TRANSPLANT-E0B-20260714.md",
        "donor": args.donor, "recipient": args.recipient,
        "layer_donor": int(dmeta["layer_donor"]), "layer_recip": L,
        "H": H, "I_donor": int(dmeta["I_donor"]), "I_recip": I,
        "n_edits": N, "n_transplantable(donor_ok)": int(tset.sum()), "seed": args.seed,
        "procrustes": {"n_anchor": len(anchor_ids), "vocab_overlap": round(overlap, 4),
                       "orth_err": orth_err, "rel_recon_err": round(recon_err, 4)},
        "esr": {"native": round(esr_native, 4), "identity": round(esr_identity, 4),
                "procrustes": round(esr_proc, 4),
                "procrustes_scaled": scaled_variant["esr"],
                "random_mean": round(esr_rand_mean, 4), "random_std": round(esr_rand_std, 4),
                "random_per_sample": [round(x, 4) for x in esr_rand_per]},
        "scaled_variant": scaled_variant,
        "collateral_law_survival": collateral,
        "identity_minus_procrustes_gap": round(identity_gap, 4),
        "predictor_ca_vs_success": {"spearman": (None if not np.isfinite(rho_ca) else round(rho_ca, 4)),
                                    "perm_p": (None if not np.isfinite(p_ca) else round(p_ca, 5))},
        "gates": {"G1_alignment_specific": bool(G1), "G2_nontrivial_absolute": bool(G2),
                  "G3_geometry_predicts": bool(G3), "layer_valid": bool(layer_valid)},
        "thresholds": {"margin": args.margin, "abs_floor_G2": args.abs_floor,
                       "abs_floor_K2": args.abs_floor2, "rho_floor": args.rho_floor,
                       "native_esr_floor": args.native_esr_floor},
        "verdict": verdict,
        "binding_wording": (
            "identity ~= procrustes and both >> random => bases already coincide; say "
            "'identity-alignment suffices, learned rotation not required' (do NOT claim "
            "Procrustes is necessary). Report random_mean prominently — the transplant story "
            "is only as strong as the random null is low."),
    }

    pair = f"L{int(dmeta['layer_donor'])}to{L}_s{args.seed}"
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(res, f, indent=2)
    os.replace(tmp, args.out)
    npz_out = args.out[:-5] + ".npz" if args.out.endswith(".json") else args.out + ".npz"
    # atomic write (parity with the JSON + donor bank): savez appends ".npz" to a suffix-less
    # path, so write to an explicit tmp base and os.replace the realized file into position.
    npz_tmp = npz_out + ".tmp"
    np.savez_compressed(
        npz_tmp,
        succ_native=succ["native"].astype(np.float32),
        succ_identity=succ["identity"].astype(np.float32),
        succ_procrustes=succ["procrustes"].astype(np.float32),
        succ_procrustes_scaled=succ_scaled.astype(np.float32),
        succ_random=succ_rand.astype(np.float32),
        ca=ca.astype(np.float32), donor_edit_ok=donor_edit_ok.astype(np.float32),
    )
    os.replace(npz_tmp + ".npz", npz_out)
    print(f"[recip] verdict={verdict} esr(proc={esr_proc:.3f} rand={esr_rand_mean:.3f} "
          f"native={esr_native:.3f} ident={esr_identity:.3f}) rho_ca={rho_ca} p={p_ca}", flush=True)
    print(f"[recip] wrote {args.out} + {os.path.basename(npz_out)} ({time.time()-t0:.1f}s)",
          flush=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="D3/T1.3-E0b generational edit transplant")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--phase", choices=["donor", "recipient", "all"], default=None)
    ap.add_argument("--donor", default=None)
    ap.add_argument("--recipient", default=None)
    ap.add_argument("--layer_donor", default="auto")
    ap.add_argument("--layer_recip", default="auto")
    ap.add_argument("--data", default=os.path.join(HARNESS, "data", "counterfact.json"))
    ap.add_argument("--n_edits", type=int, default=200)
    ap.add_argument("--n_probes", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--v_weight_decay", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--collateral", action="store_true",
                    help="prereg §3.6 tertiary (NON-GATING): within-probe key-cos->damage law on "
                         "the procrustes-transplanted edits. Default off (adds an N x Mc probe "
                         "sweep); enable on the primary pair only.")
    ap.add_argument("--n_collateral_probes", type=int, default=200,
                    help="probe count for the --collateral sweep (prereg §3.6 uses 200)")
    ap.add_argument("--donor_bank", default=os.path.join(HARNESS, "results", "transplant",
                                                         "donor_bank.npz"))
    ap.add_argument("--out", default=os.path.join(HARNESS, "results", "transplant",
                                                  "D3_transplant_E0b.json"))
    ap.add_argument("--anchor_cand", type=int, default=32768,
                    help="candidate id range for shared-vocab anchors (caps donor embed dump)")
    ap.add_argument("--min_vocab_overlap", type=float, default=0.50)
    ap.add_argument("--min_anchors", type=int, default=1000)
    ap.add_argument("--n_rand", type=int, default=5, help="random-rotation null samples (>=3)")
    ap.add_argument("--n_perm", type=int, default=10000)
    ap.add_argument("--margin", type=float, default=0.10, help="G1/K1 procrustes-minus-random margin")
    ap.add_argument("--abs_floor", type=float, default=0.20, help="G2 absolute esr floor")
    ap.add_argument("--abs_floor2", type=float, default=0.10, help="K2 absolute-fail floor")
    ap.add_argument("--rho_floor", type=float, default=0.15, help="G3 predictor spearman floor")
    ap.add_argument("--native_esr_floor", type=float, default=0.50, help="layer-validity guard")
    args = ap.parse_args()

    if args.selftest:
        return _selftest()
    if args.phase is None:
        raise SystemExit("need --selftest or --phase {donor,recipient,all}")
    if args.phase in ("donor", "all"):
        if not args.donor:
            raise SystemExit("--phase donor needs --donor")
        rc = phase_donor(args)
        if rc != 0 or args.phase == "donor":
            return rc
    if args.phase in ("recipient", "all"):
        if not args.recipient:
            raise SystemExit("--phase recipient needs --recipient")
        return phase_recipient(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
