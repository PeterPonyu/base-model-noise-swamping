"""gradsim_true.py — TRUE-backprop GradSim cell (de-tautologizes the S x C <-> GradSim
comparison; see docs/findings/THEOREM-SXC-DRAFT-2026-07-06.md sections 6.2/9 open issue #1).

BACKGROUND. The on-disk ``gradsim_baseline.py`` "resid" variant is the SAME closed-form
computation as ``mechanism_sc_table.py``'s S x C (both read ``resid_norm``/``COS`` off the
SAME .npz fields) -- so it is not an independent validation of the theorem's Proposition 2
claim that S x C is proportional to a TracIn/GradSim-style first-order training influence.
This script computes that influence with a REAL backward pass (an actual model gradient),
at the pre-edit weights, for two forms:

  DIRECT   (gold standard, no A2/single-read-out assumption):
      infl_direct(e,p) := <grad_W l_p, DW_e>_F
    where l_p is the probe's correct-token (target_true) RAW logit at the prompt's LAST
    token (matching the harness's damage_logit convention), grad_W l_p is the FULL gradient
    of that scalar w.r.t. the edited layer's down_proj.weight (every token position that
    contributes, not just the probe's subject-token position), and DW_e is ROME's rank-one
    update. Since DW_e = r_e k_e^T / (k_e.k_e) is exactly rank-one, the Frobenius inner
    product reduces ALGEBRAICALLY (no approximation, just <A,uv^T>=u^T A v) to
        infl_direct(e,p) = r_e^T (grad_W l_p) k_e / (k_e . k_e)
    which is what's actually computed (one matvec per probe against all edits at once) --
    materializing the full DW_e is only done for the small identity spot-check below.

  FACTORIZED (Eq. 3 of the theorem, needs Assumption A2 -- "the probe's logit depends on W
  only through its OWN read-out r_p = W k_p"):
      infl_factorized(e,p) = alpha(e,p) * S_e * ||k_p|| * C_ep,
      alpha(e,p) := cos(g_p, r_e) * ||g_p||,   g_p := grad_{r_p} l_p
    computed from a SINGLE per-probe gradient w.r.t. the down_proj OUTPUT activation at the
    probe's own subject-token position (cheap: no full-weight backward needed for this form
    alone). ||k_p|| is dropped (a within-probe-column constant -- cancels exactly in the
    within-probe Spearman rank statistic; see THEOREM-SXC-DRAFT-2026-07-06.md section 6.1).

SIGN CONVENTION (reviewed 2026-07-06, F1). ``l_p`` above is the probe's logit; the actual
backward pass is on ``-l_p`` so that ``grad_W l_p``/``g_p``/``infl_direct``/``alpha`` all
point the "higher => MORE damage" way, matching ``damage_logit`` (:= pre_l - post_l,
positive = damaged) and every other predictor in the harness (S x C, gradsim_baseline.py).
infl(e,p) is a first-order estimate of the probe's ACTUAL signed logit change (post-pre),
i.e. -damage; naively backpropping the raw +logit would flip the sign of every downstream
within-probe rho and rank-agreement reported below relative to that convention -- a perfect
NEGATIVE correlation would then be a perfect AGREEMENT misreported as disagreement. The
alpha sign-consistency/CV statistics (the A4' test) are invariant to this global sign flip.

WHICH FORM IS "THE" TRUE INFLUENCE. Both are reported. DIRECT is the one that doesn't
assume the thing being tested; FACTORIZED is the theorem's own Eq. 3 restated with a REAL
g_p instead of substituting the S x C surrogate for it. Comparing the two against each
other (rank_agreement_direct_vs_factorized) is an empirical read on Assumption A2 itself
(not asked for explicitly, reported as a bonus diagnostic).

WHAT THIS SCRIPT OUTPUTS (the A4/A4' test): for each probe, the per-edit alpha(e,p) values
-- sign-consistency rate (fraction sharing the majority sign across edits) and coefficient
of variation -- are exactly Assumption A4/A4' (section 2 of the theorem draft), currently
untested anywhere else in the harness.

PROP.1 IDENTITY CHECK (assert, not just report). The DIRECT form's algebraic shortcut
(r_e^T G_p k_e / denom, no full DW_e materialized) must equal, to float tolerance, the
brute-force <G_p, DW_e>_F with DW_e fully materialized as outer(r_e,k_e)/denom -- this is
Prop. 1's rank-one substitution identity applied at the Frobenius-inner-product level (an
implementation self-check, not a physics/model assumption). Checked on a small
(edit,probe) grid (--identity_max_edits x --identity_max_probes) and hard-asserted.

TWO OPERATING MODES:
  * EXTERNAL (both --gate_npz and --mech_npz given): reuses an EXISTING killgate
    --save_matrices gate .npz (COS/damage_logit/edit_ok/pre_p/norm_growth) and an EXISTING
    mechanism_dump.py --save_vectors npz (resid_vecs) for the SAME (dataset,n_edits,seed) --
    no probe-damage sweep or edit loop is re-run; this is the ~30-GPU-min science path
    (run_gradsim_true.sh). Edit KEYS are still recomputed locally (cheap, no gradient) since
    neither npz stores them.
  * SELF-CONTAINED (neither given): runs its own small restore-every-edit loop (mirrors
    killgate_keygeom.py + mechanism_dump.py's --save_vectors combined) to produce
    COS/damage_logit/edit_ok/pre_p/S/resid_vecs from scratch. This is the "full pipeline"
    CPU-smoke path (no pre-existing artifacts needed).

Usage (external, science):
  python experiments/gradsim_true.py --model data/models/Llama-3.2-1B --layer 12 --seed 0 \
      --gate_npz results/matrices/gate_llama1b_rome_cf_L12_s0.npz \
      --mech_npz results/mechanism/Llama-3.2-1B_L12.npz \
      --known --edit_ok --device cuda \
      --out results/GRADSIM_TRUE_Llama-3.2-1B_L12_s0.json

Usage (self-contained CPU smoke):
  python experiments/gradsim_true.py --model data/models/Qwen2.5-0.5B --layer auto --seed 0 \
      --n_edits 3 --n_probes 8 --steps 2 --device cpu \
      --out results/smoke/GRADSIM_TRUE_qwen05b_cpu_smoke.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HARNESS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HARNESS not in sys.path:
    sys.path.insert(0, HARNESS)
from experiments.killgate_keygeom import load_counterfact, load_zsre  # noqa: E402
from experiments.analyze_matrices import spearman, within_probe_rhos  # noqa: E402
from experiments.gradsim_baseline import _apply_masks  # noqa: E402  (reuse, don't duplicate)
from editors.rome_native import (  # noqa: E402
    apply_edit, _capture_key, find_subject_last_token_index,
)
from metrics import first_target_token_id, next_token_logits, efficacy  # noqa: E402


# --------------------------------------------------------------------------- #
# shared small helpers                                                         #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _prob_of_token(model, tok, prompt, token_id, device):
    logits = next_token_logits(model, tok, prompt, device)
    probs = torch.softmax(logits, dim=-1)
    return float(probs[token_id].item()), float(logits[token_id].item())


@torch.no_grad()
def _key_for(model, tok, layer, prompt, subject, device):
    idx = find_subject_last_token_index(tok, prompt, subject)
    return _capture_key(model, tok, layer, prompt, idx, device).float().cpu().numpy()


def _sign_consistency(vals):
    """Fraction of (finite) entries sharing the MAJORITY sign. NaN-safe."""
    v = np.asarray(vals, float)
    v = v[np.isfinite(v) & (v != 0)]
    if v.size == 0:
        return np.nan
    pos = float(np.mean(v > 0))
    return max(pos, 1.0 - pos)


def _cv(vals):
    """Coefficient of variation |std/mean| of (finite) entries. NaN-safe."""
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    if v.size < 2 or abs(float(np.mean(v))) < 1e-12:
        return np.nan
    return float(np.std(v) / abs(np.mean(v)))


class _NpzLike(dict):
    """Minimal shim so gradsim_baseline._apply_masks (written against a real np.load()
    NpzFile, which exposes .files) also accepts a plain in-memory dict here."""
    @property
    def files(self):
        return list(self.keys())


def _summ(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"mean": None, "median": None, "std": None, "n": 0}
    return {"mean": round(float(np.mean(x)), 4), "median": round(float(np.median(x)), 4),
            "std": round(float(np.std(x)), 4), "n": int(x.size)}


# --------------------------------------------------------------------------- #
# SELF-CONTAINED mode: mini killgate + mechanism_dump combined (restore-every-edit)
# --------------------------------------------------------------------------- #
def self_contained_pipeline(model, tok, edits, probes, layer, steps, lr, v_weight_decay, device):
    """Base-model key capture (all keys pre-edit, since weights restore every edit) +
    per-edit resid_vec/S/edit_ok + full probe damage sweep. Mirrors killgate_keygeom.py's
    key_for/COS/damage_l construction and mechanism_dump.py's --save_vectors capture, at
    small scale, with no dependency on either script's on-disk artifacts."""
    K_edit = np.stack([_key_for(model, tok, layer, e["prompt"], e["subject"], device) for e in edits])
    K_probe = np.stack([_key_for(model, tok, layer, p["prompt"], p["subject"], device) for p in probes])
    Ke = K_edit / (np.linalg.norm(K_edit, axis=1, keepdims=True) + 1e-8)
    Kp = K_probe / (np.linalg.norm(K_probe, axis=1, keepdims=True) + 1e-8)
    COS = (Ke @ Kp.T).astype(np.float32)

    probe_tok = [first_target_token_id(tok, p["target_true"]) for p in probes]
    pre_p = np.zeros(len(probes), np.float32)
    pre_l = np.zeros(len(probes), np.float32)
    for j, p in enumerate(probes):
        pre_p[j], pre_l[j] = _prob_of_token(model, tok, p["prompt"], probe_tok[j], device)

    W = model.model.layers[layer].mlp.down_proj.weight
    W_base = W.detach().clone()
    N, M = len(edits), len(probes)
    damage_l = np.zeros((N, M), np.float32)
    edit_ok = np.zeros(N, np.float32)
    S = np.zeros(N, np.float32)
    resid_vecs = np.zeros((N, W.shape[0]), np.float32)
    cfg = {"layer": layer, "steps": steps, "lr": lr, "v_weight_decay": v_weight_decay}
    for i, e in enumerate(edits):
        info = apply_edit(model, tok, e, cfg, device)
        ng = info["delta_weight_norm"]
        S[i] = float(ng[layer]) if isinstance(ng, dict) else float(ng)  # ||DW||_F == S (Eq.2)
        resid_vecs[i] = np.asarray(info["residual_vec"], dtype=np.float32)
        eff = efficacy(model, tok, e["prompt"], e["target_new"], e.get("target_true"), device)
        edit_ok[i] = eff["success"]
        for j, p in enumerate(probes):
            _, ll = _prob_of_token(model, tok, p["prompt"], probe_tok[j], device)
            damage_l[i, j] = pre_l[j] - ll
        with torch.no_grad():
            W.copy_(W_base)
    return COS, damage_l, edit_ok, pre_p, S, resid_vecs, K_edit


# --------------------------------------------------------------------------- #
# per-probe TRUE gradient: g_p (activation grad, factorized form) AND
# grad_W l_p (full weight grad, direct form) from ONE backward pass.
# --------------------------------------------------------------------------- #
def probe_gradients(model, tok, layer, prompt, subject, target_true, device):
    down = model.model.layers[layer].mlp.down_proj
    tok_index = find_subject_last_token_index(tok, prompt, subject)
    captured = {}

    def hook(_m, _i, output):
        output.retain_grad()
        captured["r"] = output

    h = down.register_forward_hook(hook)
    model.zero_grad(set_to_none=True)
    enc = tok(prompt, return_tensors="pt").to(device)
    logits = model(**enc).logits[0, -1, :].float()   # RAW logit (matches damage_logit convention)
    tgt_id = first_target_token_id(tok, target_true)
    # SIGN CONVENTION (reviewed 2026-07-06, F1): damage_logit is defined as pre_l - post_l
    # (positive = the edit HURT this probe), and every other predictor in the harness
    # (gradsim_baseline.py, S x C) is oriented so higher-predictor => more-damage. infl(e,p)
    # is a first-order estimate of the ACTUAL signed logit change post-pre, i.e. -damage;
    # backprop on the raw +logit would make g_p/G_p/infl point the "logit went UP" way,
    # flipping every downstream rho/rank-agreement's sign relative to that convention (a
    # perfect NEGATIVE correlation would then be a perfect AGREEMENT misreported as
    # disagreement). Backprop on -logit instead so higher infl/alpha => more damage,
    # matching S x C's own sign convention directly.
    loss = -logits[tgt_id]
    loss.backward()
    h.remove()
    g_p = captured["r"].grad[0, tok_index, :].detach().clone()   # [hidden]
    G_p = down.weight.grad.detach().clone()                       # [hidden, intermediate]
    model.zero_grad(set_to_none=True)
    return g_p, G_p


def main():
    ap = argparse.ArgumentParser(description="TRUE-backprop GradSim cell (de-tautologizes S x C).")
    ap.add_argument("--model", default=os.path.join(HARNESS, "data", "models", "Llama-3.2-1B"))
    ap.add_argument("--data", default=os.path.join(HARNESS, "data", "counterfact.json"))
    ap.add_argument("--dataset", choices=["counterfact", "zsre"], default="counterfact")
    ap.add_argument("--n_edits", type=int, default=200)
    ap.add_argument("--n_probes", type=int, default=500)
    ap.add_argument("--layer", default="12", help="int or 'auto' (n_layers//2)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=20, help="self-contained mode only")
    ap.add_argument("--lr", type=float, default=0.1, help="self-contained mode only")
    ap.add_argument("--v_weight_decay", type=float, default=1e-3, help="self-contained mode only")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--known", action="store_true")
    ap.add_argument("--edit_ok", action="store_true")
    ap.add_argument("--gate_npz", default=None,
                    help="EXTERNAL mode: existing killgate --save_matrices npz "
                         "(COS/damage_logit/edit_ok/pre_p/norm_growth). Must be paired with "
                         "--mech_npz (same dataset/n_edits/seed/layer).")
    ap.add_argument("--mech_npz", default=None,
                    help="EXTERNAL mode: existing mechanism_dump.py --save_vectors npz "
                         "(resid_vecs). Must be paired with --gate_npz.")
    ap.add_argument("--identity_max_edits", type=int, default=10)
    ap.add_argument("--identity_max_probes", type=int, default=20)
    ap.add_argument("--identity_rtol", type=float, default=1e-3,
                    help="relative tolerance for the Prop.1 Frobenius-identity assert")
    ap.add_argument("--identity_atol", type=float, default=1e-4,
                    help="absolute tolerance floor (np.allclose-style combined bound), so "
                         "near-zero influence cells don't spuriously fail on fp32 noise")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if bool(args.gate_npz) != bool(args.mech_npz):
        raise SystemExit("[gradsim_true] --gate_npz and --mech_npz must be given TOGETHER "
                         "(external mode) or both omitted (self-contained mode)")
    external = args.gate_npz is not None

    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(args.device).eval()
    nL = model.config.num_hidden_layers
    layer = nL // 2 if args.layer == "auto" else int(args.layer)
    # freeze everything except the edited layer's down_proj.weight -- backward still needs
    # to traverse the WHOLE graph to reach it, but we skip accumulating .grad for every other
    # parameter (memory/compute hygiene; matches rome_native._optimise_value's own pattern).
    for p in model.parameters():
        p.requires_grad_(False)
    down_w = model.model.layers[layer].mlp.down_proj.weight
    down_w.requires_grad_(True)
    print(f"[gradsim_true] loaded {args.model} ({nL} layers, layer={layer}, "
          f"device={args.device}, mode={'external' if external else 'self_contained'}) "
          f"{time.time()-t0:.1f}s", flush=True)

    load_fn = load_counterfact if args.dataset == "counterfact" else load_zsre
    edits, probes, *_ = load_fn(args.data, args.n_edits, args.n_probes, args.seed)
    print(f"[gradsim_true] {args.dataset}: {len(edits)} edits, {len(probes)} probes "
          f"(seed {args.seed})", flush=True)

    if external:
        gate_d = np.load(args.gate_npz)
        COS = gate_d["COS"].astype(np.float32)
        D = gate_d["damage_logit"].astype(np.float32)
        edit_ok_all = gate_d["edit_ok"].astype(np.float32)
        pre_p_all = gate_d["pre_p"].astype(np.float32)
        S_all = gate_d["norm_growth"].astype(np.float32)   # Eq.2-correct S (mechanism_sc_table.py fix)
        if COS.shape != (len(edits), len(probes)):
            raise SystemExit(f"[gradsim_true] --gate_npz shape {COS.shape} != expected "
                             f"({len(edits)}, {len(probes)}) from --n_edits/--n_probes/--seed/"
                             f"--data -- these must be the SAME loader call used to build it")
        mech_d = np.load(args.mech_npz)
        if "resid_vecs" not in mech_d.files:
            raise SystemExit(f"[gradsim_true] --mech_npz {args.mech_npz} has no 'resid_vecs' "
                             f"-- regenerate with mechanism_dump.py --save_vectors")
        resid_vecs_all = mech_d["resid_vecs"].astype(np.float32)
        if resid_vecs_all.shape[0] != len(edits):
            raise SystemExit(f"[gradsim_true] --mech_npz has {resid_vecs_all.shape[0]} edit "
                             f"rows != {len(edits)} from --n_edits/--seed/--data")
        key_norm_stored = mech_d["key_norm"].astype(np.float32) if "key_norm" in mech_d.files else None
    else:
        COS, D, edit_ok_all, pre_p_all, S_all, resid_vecs_all, _K_edit_sc = self_contained_pipeline(
            model, tok, edits, probes, layer, args.steps, args.lr, args.v_weight_decay, args.device)
        key_norm_stored = None
        print(f"[gradsim_true] self-contained pipeline done {time.time()-t0:.1f}s", flush=True)
        # BUG GUARD: rome_native._optimise_value (called inside apply_edit, called inside
        # self_contained_pipeline for every edit) does its OWN `for p in model.parameters():
        # p.requires_grad_(False)` internally and never restores it -- it clobbers the
        # down_w.requires_grad_(True) set above. Re-assert it here, AFTER the edit loop and
        # BEFORE any probe_gradients() call, or every G_p/g_p below silently comes back None.
        down_w.requires_grad_(True)

    # edit KEYS: always recomputed locally (cheap, no gradient) -- neither npz stores the
    # actual key vector, only its norm (mechanism npz) or nothing (gate npz).
    with torch.no_grad():
        K_edit = np.stack([_key_for(model, tok, layer, e["prompt"], e["subject"], args.device)
                           for e in edits]).astype(np.float32)
    key_norm_recomputed = np.linalg.norm(K_edit, axis=1).astype(np.float32)
    key_norm_check = None
    KEY_NORM_TOL = 1e-2
    if key_norm_stored is not None:
        rel = np.abs(key_norm_recomputed - key_norm_stored) / (key_norm_stored + 1e-8)
        key_norm_check = {"max_rel_diff": round(float(np.max(rel)), 6), "tol": KEY_NORM_TOL,
                          "note": ("recomputed edit keys vs mechanism_dump.py's key_norm -- "
                                   "large values would mean key capture drifted between runs")}
        # F4 (reviewed 2026-07-06): a real mismatch here means --mech_npz's resid_vecs are
        # for a DIFFERENT edit bank than the one --n_edits/--seed/--data/--model just
        # reconstructed (stale file, wrong seed, wrong model, drifted key-capture logic
        # between runs) -- exactly the silent-stale-vectors failure a soft WARN can't guard
        # against. Hard-fail instead of proceeding with a row-misaligned resid_vecs/K_edit
        # pairing (every downstream infl/alpha number would be silently wrong).
        if float(np.max(rel)) > KEY_NORM_TOL:
            raise SystemExit(
                f"[gradsim_true] key_norm mismatch vs --mech_npz: max_rel_diff="
                f"{key_norm_check['max_rel_diff']} > tol={KEY_NORM_TOL} -- --mech_npz "
                f"{args.mech_npz} does not match the edit bank reconstructed from "
                f"--model/--data/--dataset/--n_edits/--seed; refusing to proceed with a "
                f"possibly row-misaligned resid_vecs/K_edit pairing")

    # ---- masks (byte-identical convention to analyze_matrices.py / gradsim_baseline.py) ----
    fake_d = _NpzLike(edit_ok=edit_ok_all, pre_p=pre_p_all)
    rows, cols, COSm, Dm = _apply_masks(fake_d, COS, D, args.known, args.edit_ok)
    probe_idx = np.where(cols)[0]
    n_rows, n_cols = int(rows.sum()), int(len(probe_idx))
    print(f"[gradsim_true] masked shape ({n_rows}, {n_cols}) "
          f"(known={args.known}, edit_ok={args.edit_ok})", flush=True)
    if n_rows < 3 or n_cols < 1:
        raise SystemExit(f"[gradsim_true] too few rows/cols after masking ({n_rows},{n_cols})")

    K_edit_m = K_edit[rows]
    resid_vecs_m = resid_vecs_all[rows]
    S_m = S_all[rows]
    denom_m = (K_edit_m.astype(np.float64) ** 2).sum(axis=1).astype(np.float32) + 1e-8

    Kt = torch.tensor(K_edit_m, device=args.device, dtype=torch.float32)      # [n_rows, intermediate]
    Rt = torch.tensor(resid_vecs_m, device=args.device, dtype=torch.float32)  # [n_rows, hidden]
    denom_t = torch.tensor(denom_m, device=args.device, dtype=torch.float32)  # [n_rows]
    Rt_norm = Rt.norm(dim=1)                                                  # [n_rows]

    INFL_DIRECT = np.full((n_rows, n_cols), np.nan, np.float32)
    INFL_FACT = np.full((n_rows, n_cols), np.nan, np.float32)
    ALPHA = np.full((n_rows, n_cols), np.nan, np.float32)

    identity_diffs = []
    n_id_edits = min(args.identity_max_edits, n_rows)
    n_id_probes = min(args.identity_max_probes, n_cols)

    t1 = time.time()
    for k, j in enumerate(probe_idx):
        p = probes[j]
        g_p, G_p = probe_gradients(model, tok, layer, p["prompt"], p["subject"],
                                   p["target_true"], args.device)
        # DIRECT form: exact Frobenius-inner-product shortcut, no A2.
        GK = G_p @ Kt.T                              # [hidden, n_rows]
        num = (Rt.T * GK).sum(dim=0)                 # [n_rows]  == r_e^T G_p k_e
        infl_direct = (num / denom_t).detach().cpu().numpy()
        INFL_DIRECT[:, k] = infl_direct

        # FACTORIZED form (Eq. 3): alpha(e,p) = cos(g_p,r_e)*||g_p|| = (g_p.r_e)/||r_e||
        dot = Rt @ g_p                                # [n_rows]
        alpha = (dot / Rt_norm).detach().cpu().numpy()
        ALPHA[:, k] = alpha
        INFL_FACT[:, k] = alpha * S_m * COSm[:, k]    # ||k_p|| omitted (within-probe constant)

        # Prop.1 Frobenius identity spot-check: brute-force materialize DW_e for a few
        # (edit,probe) cells and compare against the algebraic shortcut used above. Combined
        # abs+rel bound (np.allclose-style: |a-b| <= atol + rtol*|b|), reported as a ratio
        # (<=1 means pass) so a near-zero influence cell can't spuriously fail on fp32 noise.
        # F2 (reviewed 2026-07-06): the brute-force side sums Llama-1B-scale [hidden x
        # intermediate] elementwise products naively in fp32, which alone can lose enough
        # precision (summation-order cancellation over ~2048x8192 terms) to spuriously fail
        # a real-scale cell even though the algebra is correct -- widen ONLY the brute-force
        # reference to float64 (the shortcut/"fast" side stays fp32, matching what's actually
        # used for every other edit/probe -- this is what the check is meant to validate).
        if k < n_id_probes:
            G_p64 = G_p.double()
            for e in range(n_id_edits):
                dW_e = torch.outer(Rt[e], Kt[e]).double() / denom_t[e].double()
                brute = float(torch.sum(G_p64 * dW_e).item())
                fast = float(infl_direct[e])
                bound = args.identity_atol + args.identity_rtol * abs(fast)
                identity_diffs.append(abs(brute - fast) / bound)
        if (k + 1) % 50 == 0:
            print(f"[gradsim_true] probe {k+1}/{n_cols} {time.time()-t1:.1f}s", flush=True)

    identity_diffs = np.asarray(identity_diffs, float)
    identity_pass = bool(identity_diffs.size and float(identity_diffs.max()) <= 1.0)
    prop1_identity_check = {
        "n_pairs_checked": int(identity_diffs.size),
        "n_edits_checked": n_id_edits, "n_probes_checked": n_id_probes,
        "max_bound_ratio": (round(float(identity_diffs.max()), 6) if identity_diffs.size else None),
        "atol": args.identity_atol, "rtol": args.identity_rtol,
        "PASS": identity_pass,
    }
    if not identity_pass:
        raise AssertionError(
            f"[gradsim_true] Prop.1 Frobenius-identity check FAILED: "
            f"max_bound_ratio={prop1_identity_check['max_bound_ratio']} > 1.0 "
            f"(atol={args.identity_atol}, rtol={args.identity_rtol}) -- the algebraic "
            f"shortcut r_e^T G_p k_e/denom does not match brute-force "
            f"<G_p, outer(r_e,k_e)/denom>_F; this is a code bug, not a scientific finding")
    print(f"[gradsim_true] Prop.1 identity check: {prop1_identity_check}", flush=True)

    # ---- within-probe rank statistics ----
    SC = S_m[:, None] * np.abs(COSm)   # existing S x C predictor (reference)
    rho_direct = within_probe_rhos(INFL_DIRECT, Dm)
    rho_fact = within_probe_rhos(INFL_FACT, Dm)
    rho_sc = within_probe_rhos(SC, Dm)
    agree_direct_sc = np.array([spearman(INFL_DIRECT[:, k], SC[:, k]) for k in range(n_cols)])
    agree_direct_fact = np.array([spearman(INFL_DIRECT[:, k], INFL_FACT[:, k]) for k in range(n_cols)])
    sign_consist = np.array([_sign_consistency(ALPHA[:, k]) for k in range(n_cols)])
    cv_alpha = np.array([_cv(ALPHA[:, k]) for k in range(n_cols)])

    res = {
        "model": args.model, "layer": layer, "seed": args.seed, "dataset": args.dataset,
        "mode": "external" if external else "self_contained",
        "gate_npz": args.gate_npz, "mech_npz": args.mech_npz,
        "filters": {"known": args.known, "edit_ok": args.edit_ok},
        "n_edits_total": len(edits), "n_probes_total": len(probes),
        "n_edits_used": n_rows, "n_probes_used": n_cols,
        "key_norm_crosscheck": key_norm_check,
        "influence_form_implemented": {
            "direct": ("<grad_W l_p, DW_e>_F via the exact rank-one shortcut "
                      "r_e^T (grad_W l_p) k_e / (k_e.k_e); grad_W l_p is the FULL weight "
                      "gradient (no A2 assumption); l_p = raw target_true logit at the "
                      "prompt's last token."),
            "factorized": ("Eq.3 alpha(e,p)*S_e*C_ep, alpha=cos(g_p,r_e)*||g_p||, "
                          "g_p=grad_{r_p} l_p at r_p=W k_p (probe's own subject-token "
                          "position) -- the A2-approximate form. ||k_p|| dropped (within-"
                          "probe-column constant; cancels in Spearman rank)."),
        },
        "prop1_identity_check": prop1_identity_check,
        "within_probe_rho": {
            "direct_vs_damage": _summ(rho_direct),
            "factorized_vs_damage": _summ(rho_fact),
            "SC_vs_damage_reference": _summ(rho_sc),
        },
        "rank_agreement": {
            "direct_vs_SC": _summ(agree_direct_sc),
            "direct_vs_factorized_A2_check": _summ(agree_direct_fact),
        },
        "alpha_A4_test": {
            "sign_consistency_rate": _summ(sign_consist),
            "coefficient_of_variation": _summ(cv_alpha),
            "note": ("A4' requires HIGH sign-consistency AND LOW CV per probe (section 2, "
                     "docs/findings/THEOREM-SXC-DRAFT-2026-07-06.md). sign_consistency_rate "
                     "1.0 = alpha never changes sign for that probe across edits."),
        },
        "runtime_s": round(time.time() - t0, 1),
    }
    print(json.dumps({k: v for k, v in res.items() if k not in ("gate_npz", "mech_npz")}, indent=2),
          flush=True)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        tmp = args.out + ".tmp"
        json.dump(res, open(tmp, "w"), indent=2)
        os.replace(tmp, args.out)
        print(f"[gradsim_true] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
