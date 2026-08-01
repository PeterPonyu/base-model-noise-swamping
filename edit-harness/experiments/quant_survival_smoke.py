"""quant_survival_smoke.py — Direction #1 SMOKE: does a knowledge edit AND its collateral
damage survive quantization, and does the key-geometry→damage tie survive too?

SMOKE-SCALE, simulated quantization via a weight ROUND-TRIP (dequant(quant(W))) — the REAL
GGUF/GPTQ/bitsandbytes kernels are deferred to the full paper. The round-trip codecs are a
faithful pure-torch reimplementation (no bitsandbytes dependency, so the whole thing is
CPU-testable and deterministic):
  * INT8  — per-output-row symmetric affine: scale = absmax_row/127, q = round(W/scale), deq = q·scale.
  * NF4   — the QLoRA/bitsandbytes NF4 4-bit codebook (16 normal-quantile levels) with per-block
            absmax scaling (blocksize 64): deq = level·absmax, level = nearest NF4 grid point.
We use the pure-torch path as PRIMARY (portable + selftestable). If bitsandbytes is importable it
could replace these kernels in the full study; the SMOKE deliberately does not depend on it.

DISCLOSURE (required before any paper claim): this simulated NF4 keeps each block's absmax in
FULL fp32, i.e. it OMITS bitsandbytes' DOUBLE QUANTIZATION (the block absmax values are themselves
FP8-quantized in outer blocks of 256). Real NF4 therefore carries a small additional scale-
quantization error our round-trip does not, so this smoke slightly UNDERSTATES real NF4 error /
overstates survival. The full-paper measurement must use the real bitsandbytes/GGUF kernels.

MEASUREMENT (reuses the gate cells' machinery — same metrics, imported primitives, signed
damage_logit, NEVER AUROC):
  * key k = _capture_key at layer L (rome_native); COS[i,j] = cos(k_edit_i, k_probe_j) = key-cos.
  * probe damage_logit[i,j] = pre_l[j] − post_l[i,j]  (pre_l = fp32 UNEDITED probe target-true
    logit; positive = damaged), exactly killgate_keygeom.py's damage_l.
  * edit success esr = efficacy(...)["success"] (argmax == new target).

ARMS per edit i (ROME edit installed fp32 first):
  * fp32                       — install, measure esr + damage (the standard gate measurement).
  * edited_layer × {nf4,int8}  — round-trip ONLY the edited down_proj@L, measure esr + damage.
  * full_model × {nf4,int8}    — round-trip ALL transformer-block linears (edit-then-quantize
                                 deployment order), measure esr + damage.
BASE arm (once, unedited): round-trip the unedited model (both localities × schemes) and measure
the same probes → per-probe base quant noise, so edit-survival is separable from base quant noise
(edit-attributable damage = damage_arm − base_noise, reported alongside raw).

OUTPUT (per-tag via run_quant_smoke.sh: results/quant_smoke/<tag>/QS_smoke_table.json + the raw
QS_smoke_raw.npz; --out_dir/QS_smoke_table.json when the module is run directly):
  * survival rates (esr survival vs fp32; mean esr per arm),
  * damage-delta distributions (added damage = damage_arm − damage_fp32; base quant noise),
  * the MECHANISM tie the venues want: Spearman(key-cos, damage) at fp32 vs each quant arm
    (pooled + within-probe), and Spearman(damage_fp32, damage_arm) rank-survival.

FROZEN PREDICTIONS (directional, EXPLORATORY — see docs/plans/PREREG-QUANT-SMOKE-2026-07-16.md):
  (p1) esr survival > 0.9 under NF4 full-model at 1B;
  (p2) quantization ADDS damage variance but the geometry ranking survives — Spearman(key-cos,
       damage) stays within ±0.15 of fp32;
  (p3) edited-layer-only round-trip ≈ full-model for edit-local metrics.

CPU-only validation: --selftest asserts the codec quantize-dequantize identity bounds (INT8
half-step, NF4 grid-membership + max-gap) and idempotence; plus a best-effort tiny-random-model
end-to-end pipeline run (SKIP if no local tokenizer). NO GPU used to author/validate. Results are
quarantined to results/quant_smoke/. Standing rules: ROME value-opt stays fp32; signed Spearman
never AUROC; PID-only process control (driver side).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HARNESS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HARNESS)

from experiments.merging_m0 import _spearman  # noqa: E402  (reuse the audited signed-Spearman)

SCHEMA_VERSION = "qs.smoke.v1"

# bitsandbytes NF4 codebook (QLoRA / create_normal_map); 16 levels, endpoints ±1, sorted ascending.
NF4_LEVELS = [
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
]


# ============================================================ round-trip codecs (pure torch)
def int8_roundtrip(w, per_row=True):
    """Per-output-row (per_row) symmetric INT8 affine round-trip: deq = round(W/scale)·scale,
    scale = absmax/127. Round-trip only (no packed storage) — the SMOKE measures the numeric
    perturbation, not the memory footprint."""
    import torch
    wf = w.detach().float()
    if per_row and wf.dim() == 2:
        absmax = wf.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    else:
        absmax = wf.abs().amax().clamp(min=1e-12)
    scale = absmax / 127.0
    q = torch.clamp(torch.round(wf / scale), -127.0, 127.0)
    return (q * scale).to(w.dtype)


def _nf4_levels_tensor(device, dtype):
    import torch
    return torch.tensor(NF4_LEVELS, device=device, dtype=dtype)


def nf4_roundtrip(w, blocksize=64):
    """NF4 4-bit round-trip: per contiguous block of `blocksize` weights, normalise by the block
    absmax, snap to the nearest NF4 codebook level (via midpoint bucketize — memory-light), and
    dequantise as level·absmax. Faithful to bitsandbytes' blockwise NF4."""
    import torch
    orig_shape = w.shape
    wf = w.detach().float().reshape(-1)
    n = wf.numel()
    pad = (-n) % blocksize
    if pad:
        wf = torch.cat([wf, wf.new_zeros(pad)])
    blocks = wf.view(-1, blocksize)
    absmax = blocks.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    normed = blocks / absmax                                  # in [-1, 1]
    levels = _nf4_levels_tensor(w.device, torch.float32)
    mids = (levels[1:] + levels[:-1]) / 2.0                   # 15 midpoints
    idx = torch.bucketize(normed, mids)                       # nearest level index 0..15
    deq = levels[idx] * absmax
    deq = deq.reshape(-1)[:n].reshape(orig_shape)
    return deq.to(w.dtype)


def roundtrip(w, scheme, blocksize=64):
    if scheme == "int8":
        return int8_roundtrip(w)
    if scheme == "nf4":
        return nf4_roundtrip(w, blocksize=blocksize)
    if scheme in ("none", "fp32"):
        return w
    raise ValueError(f"unknown quant scheme {scheme!r} (expected nf4/int8/none)")


# ============================================================ model weight bookkeeping
def _quant_linears(model):
    """The nn.Linear modules quantized by the FULL-MODEL arm = every linear inside the transformer
    blocks (attention + MLP projections). lm_head / embeddings are left fp32 (vocab-projection
    quantization is a separate deployment choice, out of scope for this smoke)."""
    import torch.nn as nn
    mods = []
    for blk in model.model.layers:
        for m in blk.modules():
            if isinstance(m, nn.Linear):
                mods.append(m)
    return mods


def _snapshot(mods, device=None):
    """Base fp32 weight snapshot. Default (device=None): kept on the weights' own device
    (fast restore; a 1B model at fp32 is ~5GB, comfortable on a 24GB card — this smoke is
    1B-only). device="cpu" (2026-07-31, B1/B3 enablement): pinned to host RAM instead —
    a 3B fp32 model + on-device snapshot exceeds 24GB (Qwen2.5-3B OOMed the 5090 at
    snapshot; fp32 loading is prereg-bound so the snapshot's PLACEMENT is the only free
    knob). Values are bitwise identical either way; consumers must device-align (see
    _assert_restore_integrity)."""
    if device == "cpu":
        return [m.weight.detach().cpu().clone() for m in mods]
    return [m.weight.detach().clone() for m in mods]


def _restore(mods, snap):
    import torch
    with torch.no_grad():
        for m, w in zip(mods, snap):
            m.weight.copy_(w)


def _assert_restore_integrity(mods, snap, W_edited, w_edited_L, atol=1e-5):
    """MINOR-1 fidelity check (replaces the tautological W==W_base assert): after the full-model
    arms + base-restore + re-add-ΔW, every NON-edited block linear must be back to base, and the
    edited down_proj@L (identified by tensor identity `is W_edited`) must hold the fp32 edit. Full
    O(model) sweep — the caller runs it once (first edit); the install is deterministic so one proof
    of the restore machinery suffices, and the per-scheme edited-layer check runs every edit."""
    import torch
    for m, wb in zip(mods, snap):
        if m.weight is W_edited:
            assert torch.allclose(m.weight, w_edited_L, atol=atol), \
                "[qs] edited down_proj@L not fp32-edited after full-model arm"
        else:
            assert torch.allclose(m.weight, wb.to(m.weight.device), atol=atol), \
                "[qs] a non-edited block linear was not restored to base after full-model arm"


# ============================================================ measurement helpers (gate machinery)
def load_counterfact(path, n_edits, n_probes, seed=0):
    """Edit + probe banks, REIMPLEMENTED VERBATIM from killgate_keygeom.py:62-82 (edits/probes
    slice; holdout dropped) so the (seed, n_edits, n_probes) selection is byte-identical to the
    gate cells and the damage/COS numbers are directly comparable."""
    data = json.load(open(path))
    rng = np.random.default_rng(seed)
    rng.shuffle(data)
    recs = []
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
        if len(recs) >= n_edits + n_probes:
            break
    return recs[:n_edits], recs[n_edits:n_edits + n_probes]


def _prob_of_token(model, tok, prompt, token_id, device):
    """(softmax prob, logit) of `token_id` as the next token — mirrors killgate_keygeom.prob_of_token."""
    import torch
    from metrics import next_token_logits
    logits = next_token_logits(model, tok, prompt, device)   # [V] cpu float
    p = float(torch.softmax(logits, dim=-1)[token_id].item())
    return p, float(logits[token_id].item())


def _probe_logits(model, tok, probes, probe_tok, device):
    """Vector of probe target-true logits under the CURRENT model weights."""
    out = np.zeros(len(probes))
    for j, p in enumerate(probes):
        _, out[j] = _prob_of_token(model, tok, p["prompt"], probe_tok[j], device)
    return out


# ============================================================ analysis
def _within_probe_spearman(COS, damage):
    """Mean over probe columns j of Spearman(COS[:,j], damage[:,j]) across edits — the gate's
    within-probe geometry→damage tie (columns with degenerate ranks are skipped)."""
    rhos = []
    for j in range(COS.shape[1]):
        r = _spearman(COS[:, j], damage[:, j])
        if np.isfinite(r):
            rhos.append(r)
    return (float(np.mean(rhos)) if rhos else float("nan")), len(rhos)


def analyze(res):
    """Build the summary table from the raw per-arm arrays in `res`. All CPU/numpy."""
    COS = res["COS"]
    damage_fp32 = res["damage_fp32"]
    edit_ok_fp32 = res["edit_ok_fp32"]
    arms = res["arms"]                       # {arm_name: {"damage":[N,M], "edit_ok":[N], "locality","scheme"}}
    base = res["base"]                       # {arm_name: base_noise[M]}
    probes_n = COS.shape[1]

    def _rnd(x, k=4):
        return round(float(x), k) if (x is not None and np.isfinite(x)) else None

    rho_fp32_pooled = _spearman(COS.reshape(-1), damage_fp32.reshape(-1))
    rho_fp32_within, wp_n = _within_probe_spearman(COS, damage_fp32)

    n_worked_fp32 = int(np.nansum(edit_ok_fp32 >= 0.5))
    mechanism = {
        "rho_keycos_damage_fp32_pooled": _rnd(rho_fp32_pooled),
        "rho_keycos_damage_fp32_within_probe": _rnd(rho_fp32_within),
        "within_probe_n_cols": wp_n,
    }
    esr = {"mean_esr_fp32": _rnd(float(np.nanmean(edit_ok_fp32)), 4), "n_edits_worked_fp32": n_worked_fp32}
    damage_summ = {}
    arms_out = {}
    for name, a in arms.items():
        dmg = a["damage"]
        eok = a["edit_ok"]
        # base-noise-subtracted, edit-attributable damage (broadcast base over edits)
        bnoise = base.get(name)
        dmg_attr = dmg - bnoise[None, :] if bnoise is not None else dmg
        rho_pool = _spearman(COS.reshape(-1), dmg.reshape(-1))
        rho_pool_attr = _spearman(COS.reshape(-1), dmg_attr.reshape(-1))
        rho_within, _ = _within_probe_spearman(COS, dmg)
        rho_rank_survival = _spearman(damage_fp32.reshape(-1), dmg.reshape(-1))
        added = dmg - damage_fp32
        # esr survival among edits that worked fp32
        worked = edit_ok_fp32 >= 0.5
        surv = (float(np.nanmean((eok >= 0.5)[worked])) if worked.any() else float("nan"))
        arms_out[name] = {
            "locality": a["locality"], "scheme": a["scheme"],
            "mean_esr": _rnd(float(np.nanmean(eok)), 4),
            "esr_survival_given_fp32_worked": _rnd(surv, 4),
            "rho_keycos_damage_pooled": _rnd(rho_pool),
            "rho_keycos_damage_pooled_base_subtracted": _rnd(rho_pool_attr),
            "rho_keycos_damage_within_probe": _rnd(rho_within),
            "delta_rho_vs_fp32_pooled": _rnd((rho_pool - rho_fp32_pooled)
                                             if np.isfinite(rho_pool) and np.isfinite(rho_fp32_pooled)
                                             else float("nan")),
            "rho_damage_fp32_vs_arm_rank_survival": _rnd(rho_rank_survival),
            "added_damage_logit_mean": _rnd(float(np.mean(added)), 5),
            "added_damage_logit_std": _rnd(float(np.std(added)), 5),
            "base_quant_noise_logit_mean_abs": (_rnd(float(np.mean(np.abs(bnoise))), 5)
                                                if bnoise is not None else None),
        }
    # frozen-prediction read-outs (informational; EXPLORATORY)
    def _arm(scheme, loc):
        return arms_out.get(f"{scheme}_{loc}")
    p1 = _arm("nf4", "full_model")
    p2_checks = {n: (abs(a["delta_rho_vs_fp32_pooled"]) <= 0.15
                     if a["delta_rho_vs_fp32_pooled"] is not None else None)
                 for n, a in arms_out.items()}
    ed = _arm("nf4", "edited_layer"); fu = _arm("nf4", "full_model")
    p3_esr_gap = (abs((ed["mean_esr"] or 0) - (fu["mean_esr"] or 0))
                  if ed and fu and ed["mean_esr"] is not None and fu["mean_esr"] is not None else None)

    return {
        "experiment": "quant_survival_smoke",
        "schema_version": SCHEMA_VERSION,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": res["meta"]["model"], "layer": res["meta"]["layer"], "editor": "rome",
        "n_edits": res["meta"]["n_edits"], "n_probes": probes_n, "seed": res["meta"]["seed"],
        "schemes": res["meta"]["schemes"], "blocksize": res["meta"]["blocksize"],
        "quant_note": ("simulated quantization via weight round-trip dequant(quant(W)); INT8 "
                       "per-row symmetric, NF4 bitsandbytes codebook + blockwise absmax; REAL "
                       "GGUF/GPTQ deferred to the full paper"),
        "damage_metric_note": "signed damage_logit = pre_l(fp32 unedited) − post_l; never AUROC",
        "esr": esr,
        "mechanism_tie": mechanism,
        "arms": arms_out,
        "frozen_prediction_readout": {
            "p1_esr_survival_nf4_full_gt_0.9": (
                (p1["esr_survival_given_fp32_worked"] is not None
                 and p1["esr_survival_given_fp32_worked"] > 0.9) if p1 else None),
            "p1_value": (p1["esr_survival_given_fp32_worked"] if p1 else None),
            "p2_geometry_within_pm0.15_by_arm": p2_checks,
            "p3_edited_vs_full_esr_gap_nf4": p3_esr_gap,
            "note": "EXPLORATORY smoke — directional read-outs only, not a gate",
        },
    }


def print_table(table):
    print("\n=== QUANT-SURVIVAL SMOKE ===", flush=True)
    print(f"model={table['model']} L={table['layer']} n_edits={table['n_edits']} "
          f"n_probes={table['n_probes']} schemes={table['schemes']}", flush=True)
    m = table["mechanism_tie"]
    print(f"  fp32 mechanism: rho(key-cos,damage) pooled={m['rho_keycos_damage_fp32_pooled']} "
          f"within-probe={m['rho_keycos_damage_fp32_within_probe']}  mean_esr_fp32={table['esr']['mean_esr_fp32']}",
          flush=True)
    for name, a in table["arms"].items():
        print(f"  [{name}] esr={a['mean_esr']} surv={a['esr_survival_given_fp32_worked']} "
              f"rho(cos,dmg)={a['rho_keycos_damage_pooled']} Δrho_vs_fp32={a['delta_rho_vs_fp32_pooled']} "
              f"rank_surv={a['rho_damage_fp32_vs_arm_rank_survival']} "
              f"added_dmg μ={a['added_damage_logit_mean']}±{a['added_damage_logit_std']}", flush=True)
    fp = table["frozen_prediction_readout"]
    print(f"  read-out: p1(nf4-full esr surv>0.9)={fp['p1_esr_survival_nf4_full_gt_0.9']} "
          f"(={fp['p1_value']}); p3 edited-vs-full esr gap(nf4)={fp['p3_edited_vs_full_esr_gap_nf4']}",
          flush=True)


# ============================================================ phase-1 GPU run
def run_smoke(args):
    import torch
    import transformers as _tf
    from experiments.merging_m0 import _load_edit_model
    from metrics import first_target_token_id, efficacy
    from editors.rome_native import _capture_key, find_subject_last_token_index, apply_edit
    t0 = time.time()
    device = args.device
    model, tok, layer, _nL = _load_edit_model(args.model, args.layer, device)
    schemes = [s for s in str(args.schemes).split(",") if s]
    localities = ["edited_layer", "full_model"]

    edits, probes = load_counterfact(args.data, args.n_edits, args.n_probes, args.seed)
    N, M = len(edits), len(probes)
    print(f"[qs] {N} edits, {M} probes (seed {args.seed}); schemes={schemes}", flush=True)

    def key_for(rec):
        idx = find_subject_last_token_index(tok, rec["prompt"], rec.get("subject"))
        return _capture_key(model, tok, layer, rec["prompt"], idx, device).float().cpu().numpy()

    K_edit = np.stack([key_for(e) for e in edits])
    K_probe = np.stack([key_for(p) for p in probes])
    Ke = K_edit / (np.linalg.norm(K_edit, axis=1, keepdims=True) + 1e-8)
    Kp = K_probe / (np.linalg.norm(K_probe, axis=1, keepdims=True) + 1e-8)
    COS = (Ke @ Kp.T).astype(np.float64)                     # [N, M] key-cos
    probe_tok = [first_target_token_id(tok, p["target_true"]) for p in probes]
    print(f"[qs] keys+cos done {time.time()-t0:.1f}s", flush=True)

    # fp32 unedited baseline probe logits (the damage reference)
    pre_l = _probe_logits(model, tok, probes, probe_tok, device)

    down = model.model.layers[layer].mlp.down_proj
    W = down.weight
    W_base_L = W.detach().clone()
    qlinears = _quant_linears(model)
    # MINOR-3: VRAM/size preflight — an ON-DEVICE base snapshot DUPLICATES the block-linear
    # weights, so a >1B model can silently approach the 24GB ceiling and OOM mid-run. Abort loudly.
    # With --snapshot_device cpu the snapshot lives in host RAM and does NOT charge VRAM
    # (2026-07-31 fix: the gate previously charged it anyway, defeating the CPU path).
    snap_bytes = sum(m.weight.numel() * m.weight.element_size() for m in qlinears)
    snap_on_gpu = getattr(args, "snapshot_device", "cuda") != "cpu"
    if device == "cuda" and snap_on_gpu:
        free, total = torch.cuda.mem_get_info()
        if snap_bytes * 1.4 > free:
            raise SystemExit(
                f"[qs] VRAM preflight: on-device base snapshot needs ~{snap_bytes/1e9:.1f}GB "
                f"(x1.4 margin) but only {free/1e9:.1f}GB / {total/1e9:.1f}GB free — this smoke is "
                f"sized for ~1B models. Use a smaller --model, or pass --snapshot_device cpu "
                f"before running larger models.")
        print(f"[qs] VRAM preflight OK: base snapshot ~{snap_bytes/1e9:.2f}GB, "
              f"{free/1e9:.1f}GB free", flush=True)
    elif device == "cuda":
        print(f"[qs] VRAM preflight: base snapshot ~{snap_bytes/1e9:.2f}GB is CPU-resident "
              f"(no VRAM charge)", flush=True)
    base_snap = _snapshot(qlinears, device=("cpu" if getattr(args, "snapshot_device", "cuda") == "cpu" else None))  # full-model base weights

    # BASE arm (unedited quantized): per (locality, scheme) probe logits -> base quant noise
    base_noise = {}                                          # arm_name -> [M]
    for scheme in schemes:
        # edited_layer locality on the UNEDITED down_proj
        with torch.no_grad():
            W.copy_(roundtrip(W_base_L, scheme, args.blocksize))
        base_noise[f"{scheme}_edited_layer"] = pre_l - _probe_logits(model, tok, probes, probe_tok, device)
        with torch.no_grad():
            W.copy_(W_base_L)
        # full_model locality on the UNEDITED model
        with torch.no_grad():
            for m in qlinears:
                m.weight.copy_(roundtrip(m.weight, scheme, args.blocksize))
        base_noise[f"{scheme}_full_model"] = pre_l - _probe_logits(model, tok, probes, probe_tok, device)
        _restore(qlinears, base_snap)
    print(f"[qs] base-arm quant noise done {time.time()-t0:.1f}s", flush=True)

    # per-arm accumulators
    arm_names = [f"{s}_{loc}" for s in schemes for loc in localities]
    damage = {n: np.zeros((N, M)) for n in arm_names}
    edit_ok = {n: np.full(N, np.nan) for n in arm_names}
    damage_fp32 = np.zeros((N, M))
    edit_ok_fp32 = np.full(N, np.nan)

    def measure_damage():
        return pre_l - _probe_logits(model, tok, probes, probe_tok, device)

    for i, e in enumerate(edits):
        # 1) install ROME edit fp32
        info = apply_edit(model, tok, e, {"layer": layer, "steps": args.steps, "lr": args.lr}, device)
        delta = (W.detach() - W_base_L).clone()             # the edit's ΔW@L (for full-arm re-apply)
        # 2) fp32 measurement
        eff = efficacy(model, tok, e["prompt"], e["target_new"], e.get("target_true"), device)
        edit_ok_fp32[i] = eff["success"]
        damage_fp32[i] = measure_damage()
        # 3) edited_layer arms — round-trip ONLY down_proj@L, restore to edited state after each
        w_edited_L = W.detach().clone()
        for scheme in schemes:
            with torch.no_grad():
                W.copy_(roundtrip(w_edited_L, scheme, args.blocksize))
            ef = efficacy(model, tok, e["prompt"], e["target_new"], e.get("target_true"), device)
            edit_ok[f"{scheme}_edited_layer"][i] = ef["success"]
            damage[f"{scheme}_edited_layer"][i] = measure_damage()
            with torch.no_grad():
                W.copy_(w_edited_L)
        # 4) full_model arms — round-trip ALL block linears (edit in place), restore base + re-add ΔW
        for scheme in schemes:
            with torch.no_grad():
                for m in qlinears:
                    m.weight.copy_(roundtrip(m.weight, scheme, args.blocksize))
            ef = efficacy(model, tok, e["prompt"], e["target_new"], e.get("target_true"), device)
            edit_ok[f"{scheme}_full_model"][i] = ef["success"]
            damage[f"{scheme}_full_model"][i] = measure_damage()
            _restore(qlinears, base_snap)
            with torch.no_grad():
                W.add_(delta.to(W.dtype))                   # back to fp32-edited for the next scheme
            assert torch.allclose(W, w_edited_L, atol=1e-5), \
                "[qs] full-model arm: re-added ΔW@L != fp32-edited state"
        # MINOR-1: full restore-integrity sweep once (first edit) — proves the base-restore machinery
        if i == 0:
            _assert_restore_integrity(qlinears, base_snap, W, w_edited_L)
        # 5) end-of-edit restore
        with torch.no_grad():
            W.copy_(W_base_L)
        if (i + 1) % 10 == 0:
            print(f"[qs] edit {i+1}/{N}  {time.time()-t0:.1f}s", flush=True)

    with torch.no_grad():
        W.copy_(W_base_L)
    assert torch.allclose(W, W_base_L), "[qs] final restore FAILED"

    res = dict(
        COS=COS, damage_fp32=damage_fp32, edit_ok_fp32=edit_ok_fp32,
        arms={n: {"damage": damage[n], "edit_ok": edit_ok[n],
                  "scheme": n.split("_")[0], "locality": "_".join(n.split("_")[1:])}
              for n in arm_names},
        base=base_noise,
        meta=dict(model=args.model, layer=layer, n_edits=N, seed=args.seed, schemes=schemes,
                  blocksize=args.blocksize, torch=torch.__version__, transformers=_tf.__version__),
    )
    _save_raw(res, args.out_dir)
    table = analyze(res)
    out = args.table_out or os.path.join(args.out_dir, "QS_smoke_table.json")
    _write_json(table, out)
    print_table(table)
    print(f"[qs] wrote {out}  total {time.time()-t0:.1f}s", flush=True)
    return table


def _write_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def _save_raw(res, out_dir):
    """Raw per-arm arrays (COS, per-arm damage, esr, base noise) for downstream reanalysis."""
    os.makedirs(out_dir, exist_ok=True)
    arrs = dict(COS=res["COS"], damage_fp32=res["damage_fp32"], edit_ok_fp32=res["edit_ok_fp32"])
    for n, a in res["arms"].items():
        arrs[f"damage__{n}"] = a["damage"]
        arrs[f"esr__{n}"] = a["edit_ok"]
    for n, b in res["base"].items():
        arrs[f"base__{n}"] = b
    tmp = os.path.join(out_dir, "QS_smoke_raw.npz.tmp.npz")
    np.savez_compressed(tmp, **arrs)
    os.replace(tmp, os.path.join(out_dir, "QS_smoke_raw.npz"))


# ============================================================ self-test (CPU)
def _selftest_codecs(rng):
    import torch
    print("[selftest] (a) INT8 per-row round-trip: half-step error bound + idempotence ...", flush=True)
    worst_i = 0.0
    for _ in range(50):
        w = torch.tensor(rng.standard_normal((17, 40)) * rng.uniform(0.1, 5.0))
        deq = int8_roundtrip(w)
        absmax = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
        scale = absmax / 127.0
        err = (deq.float() - w.float()).abs()
        # rounding error must be <= scale/2 per row (+fp slack)
        assert bool((err <= (scale + 1e-6)).all()), "int8 error exceeds one step"
        assert bool((err <= (scale / 2 + 1e-6)).all()), "int8 error exceeds half-step"
        assert torch.allclose(int8_roundtrip(deq), deq, atol=1e-6), "int8 not idempotent"
        worst_i = max(worst_i, float((err / (scale + 1e-12)).max()))
    print(f"[selftest]   INT8 OK — worst err/step over 50 trials = {worst_i:.4f} (<= 0.5)", flush=True)

    print("[selftest] (b) NF4 round-trip: codebook + grid-membership + max-gap bound + idempotence ...",
          flush=True)
    levels = np.array(NF4_LEVELS)
    assert len(levels) == 16 and levels[0] == -1.0 and levels[-1] == 1.0, "NF4 codebook malformed"
    assert bool(np.all(np.diff(levels) > 0)), "NF4 codebook not strictly ascending"
    max_gap = float(np.max(np.diff(levels)))                 # largest adjacent gap
    worst_n = 0.0
    for _ in range(50):
        bs = 64
        w = torch.tensor(rng.standard_normal((3, 2 * bs)) * rng.uniform(0.1, 5.0))
        deq = nf4_roundtrip(w, blocksize=bs)
        blocks = w.reshape(-1, bs)
        absmax = blocks.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
        # every dequant value must be a codebook level * its block absmax
        dq_blocks = deq.reshape(-1, bs)
        ratio = (dq_blocks / absmax).reshape(-1).numpy()
        nearest = levels[np.argmin(np.abs(ratio[:, None] - levels[None, :]), axis=1)]
        assert bool(np.all(np.abs(ratio - nearest) < 1e-5)), "NF4 dequant not on the codebook grid"
        # normalized reconstruction error bounded by half the max gap (compare in block form)
        nerr = float(((dq_blocks - blocks).abs() / absmax).max())
        assert nerr <= max_gap / 2 + 1e-4, f"NF4 error {nerr:.4f} exceeds max_gap/2 {max_gap/2:.4f}"
        assert torch.allclose(nf4_roundtrip(deq, blocksize=bs), deq, atol=1e-5), "NF4 not idempotent"
        worst_n = max(worst_n, nerr)
    print(f"[selftest]   NF4 OK — worst normalized err = {worst_n:.4f} (<= max_gap/2 = {max_gap/2:.4f})",
          flush=True)
    return worst_i, worst_n


def _find_local_tokenizer():
    root = os.path.join(HARNESS, "data", "models")
    if not os.path.isdir(root):
        return None
    for p in ("Qwen2.5-0.5B", "Llama-3.2-1B", "Qwen2.5-1.5B"):
        d = os.path.join(root, p)
        if os.path.isfile(os.path.join(d, "config.json")) and any(
                os.path.isfile(os.path.join(d, f))
                for f in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json")):
            return d
    return None


def _tiny_e2e_selftest():
    """Best-effort: run the whole smoke pipeline on a tiny random-weight Llama (needs a local
    tokenizer; SKIP loudly otherwise). Asserts the table has the expected shape and that survival
    rates / rhos are finite and in-range — the plumbing check, not a science result."""
    tokdir = _find_local_tokenizer()
    if tokdir is None:
        print("[selftest] (c) tiny e2e: SKIP (no local tokenizer under data/models)", flush=True)
        return None
    try:
        import torch
        from transformers import AutoTokenizer, LlamaForCausalLM, LlamaConfig
    except Exception as ex:
        print(f"[selftest] (c) tiny e2e: SKIP (transformers unavailable: {ex})", flush=True)
        return None
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    tok = AutoTokenizer.from_pretrained(tokdir)
    # Size the embedding to COVER every id the tokenizer can emit — tokenizer-agnostic. vocab_size
    # excludes added/special tokens (e.g. Llama-3 BOS id 128000 == vocab_size 128000 -> out of range
    # -> the box IndexError at embed_tokens); len(tok) includes them. Use max(len,vocab_size)+margin.
    vocab = max(int(len(tok)), int(getattr(tok, "vocab_size", 0) or 0)) + 16
    cfg = LlamaConfig(vocab_size=vocab, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=64,
                      tie_word_embeddings=True)
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg).to("cpu").float().eval()
    from metrics import first_target_token_id, efficacy
    from editors.rome_native import _capture_key, find_subject_last_token_index, apply_edit

    edits = [{"subject": "Paris", "prompt": "Paris is the capital of", "target_new": "Spain",
              "target_true": "France"},
             {"subject": "Rome", "prompt": "Rome is the capital of", "target_new": "Egypt",
              "target_true": "Italy"}]
    probes = [{"subject": "Berlin", "prompt": "Berlin is the capital of", "target_new": "Peru",
               "target_true": "Germany"},
              {"subject": "Tokyo", "prompt": "Tokyo is the capital of", "target_new": "Chad",
               "target_true": "Japan"},
              {"subject": "Cairo", "prompt": "Cairo is the capital of", "target_new": "Cuba",
               "target_true": "Egypt"}]
    layer = 2

    class A:
        model = tokdir; layer = 2; steps = 2; lr = 0.1; seed = 0; schemes = "nf4,int8"
        blocksize = 64; n_edits = 2; n_probes = 3; device = "cpu"; data = None
        out_dir = os.path.join(HARNESS, "results", "quant_smoke", "selftest"); table_out = None
    a = A()
    # drive the run_smoke body directly on the in-memory tiny model (bypass _load_edit_model/loader)
    res = _run_on(model, tok, layer, a, edits, probes)
    table = analyze(res)
    for name, arm in table["arms"].items():
        assert arm["mean_esr"] is None or 0.0 <= arm["mean_esr"] <= 1.0, f"esr out of range {name}"
        s = arm["esr_survival_given_fp32_worked"]
        assert s is None or 0.0 <= s <= 1.0, f"survival out of range {name}"
    assert set(table["arms"]) == {f"{s}_{loc}" for s in ("nf4", "int8")
                                  for loc in ("edited_layer", "full_model")}, "arm set wrong"
    print(f"[selftest]   tiny e2e OK — {len(table['arms'])} arms, "
          f"fp32 mean_esr={table['esr']['mean_esr_fp32']}, table shape valid", flush=True)
    return True


def _run_on(model, tok, layer, args, edits, probes):
    """The GPU-body of run_smoke, factored so the tiny e2e self-test can drive it on an in-memory
    model without a loader/model-path. Returns the raw `res` dict analyze() consumes."""
    import torch
    from metrics import first_target_token_id, efficacy
    from editors.rome_native import _capture_key, find_subject_last_token_index, apply_edit
    device = args.device
    schemes = [s for s in str(args.schemes).split(",") if s]
    localities = ["edited_layer", "full_model"]
    N, M = len(edits), len(probes)

    def key_for(rec):
        idx = find_subject_last_token_index(tok, rec["prompt"], rec.get("subject"))
        return _capture_key(model, tok, layer, rec["prompt"], idx, device).float().cpu().numpy()

    K_edit = np.stack([key_for(e) for e in edits]); K_probe = np.stack([key_for(p) for p in probes])
    Ke = K_edit / (np.linalg.norm(K_edit, axis=1, keepdims=True) + 1e-8)
    Kp = K_probe / (np.linalg.norm(K_probe, axis=1, keepdims=True) + 1e-8)
    COS = (Ke @ Kp.T).astype(np.float64)
    probe_tok = [first_target_token_id(tok, p["target_true"]) for p in probes]
    pre_l = _probe_logits(model, tok, probes, probe_tok, device)

    W = model.model.layers[layer].mlp.down_proj.weight
    W_base_L = W.detach().clone()
    qlinears = _quant_linears(model); base_snap = _snapshot(qlinears, device=("cpu" if getattr(args, "snapshot_device", "cuda") == "cpu" else None))

    base_noise = {}
    for scheme in schemes:
        with torch.no_grad():
            W.copy_(roundtrip(W_base_L, scheme, args.blocksize))
        base_noise[f"{scheme}_edited_layer"] = pre_l - _probe_logits(model, tok, probes, probe_tok, device)
        with torch.no_grad():
            W.copy_(W_base_L)
            for m in qlinears:
                m.weight.copy_(roundtrip(m.weight, scheme, args.blocksize))
        base_noise[f"{scheme}_full_model"] = pre_l - _probe_logits(model, tok, probes, probe_tok, device)
        _restore(qlinears, base_snap)

    arm_names = [f"{s}_{loc}" for s in schemes for loc in localities]
    damage = {n: np.zeros((N, M)) for n in arm_names}; edit_ok = {n: np.full(N, np.nan) for n in arm_names}
    damage_fp32 = np.zeros((N, M)); edit_ok_fp32 = np.full(N, np.nan)

    def measure_damage():
        return pre_l - _probe_logits(model, tok, probes, probe_tok, device)

    for i, e in enumerate(edits):
        apply_edit(model, tok, e, {"layer": layer, "steps": args.steps, "lr": args.lr}, device)
        delta = (W.detach() - W_base_L).clone()
        eff = efficacy(model, tok, e["prompt"], e["target_new"], e.get("target_true"), device)
        edit_ok_fp32[i] = eff["success"]; damage_fp32[i] = measure_damage()
        w_edited_L = W.detach().clone()
        for scheme in schemes:
            with torch.no_grad():
                W.copy_(roundtrip(w_edited_L, scheme, args.blocksize))
            ef = efficacy(model, tok, e["prompt"], e["target_new"], e.get("target_true"), device)
            edit_ok[f"{scheme}_edited_layer"][i] = ef["success"]
            damage[f"{scheme}_edited_layer"][i] = measure_damage()
            with torch.no_grad():
                W.copy_(w_edited_L)
        for scheme in schemes:
            with torch.no_grad():
                for m in qlinears:
                    m.weight.copy_(roundtrip(m.weight, scheme, args.blocksize))
            ef = efficacy(model, tok, e["prompt"], e["target_new"], e.get("target_true"), device)
            edit_ok[f"{scheme}_full_model"][i] = ef["success"]
            damage[f"{scheme}_full_model"][i] = measure_damage()
            _restore(qlinears, base_snap)
            with torch.no_grad():
                W.add_(delta.to(W.dtype))
            assert torch.allclose(W, w_edited_L, atol=1e-5), \
                "[qs] full-model arm: re-added ΔW@L != fp32-edited state"
        if i == 0:
            _assert_restore_integrity(qlinears, base_snap, W, w_edited_L)
        with torch.no_grad():
            W.copy_(W_base_L)
    with torch.no_grad():
        W.copy_(W_base_L)
    return dict(
        COS=COS, damage_fp32=damage_fp32, edit_ok_fp32=edit_ok_fp32,
        arms={n: {"damage": damage[n], "edit_ok": edit_ok[n],
                  "scheme": n.split("_")[0], "locality": "_".join(n.split("_")[1:])} for n in arm_names},
        base=base_noise,
        meta=dict(model=getattr(args, "model", "tiny"), layer=layer, n_edits=N, seed=args.seed,
                  schemes=schemes, blocksize=args.blocksize))


def selftest():
    rng = np.random.default_rng(20260716)
    print("[selftest] quant-survival smoke — CPU (codecs + tiny e2e)", flush=True)
    _selftest_codecs(rng)
    print("[selftest] (c) tiny end-to-end pipeline on a tiny random Llama ...", flush=True)
    e2e = _tiny_e2e_selftest()
    tail = "" if e2e else " [tiny e2e SKIPPED — no local tokenizer]"
    print("\n[selftest] ALL CHECKS PASSED (INT8 + NF4 codec bounds/idempotence" +
          (" + tiny e2e pipeline" if e2e else "") + ")" + tail, flush=True)
    return True


# ============================================================ CLI
def main():
    ap = argparse.ArgumentParser(description="Quantization-survival smoke (Direction #1).")
    ap.add_argument("--selftest", action="store_true",
                    help="CPU self-test: codec identity bounds + tiny-model e2e (no GPU).")
    ap.add_argument("--run", action="store_true", help="GPU smoke run.")
    ap.add_argument("--model", default=os.path.join(HARNESS, "data", "models", "Llama-3.2-1B"))
    ap.add_argument("--data", default=os.path.join(HARNESS, "data", "counterfact.json"))
    ap.add_argument("--n_edits", type=int, default=50)
    ap.add_argument("--n_probes", type=int, default=40)
    ap.add_argument("--layer", default="12")
    ap.add_argument("--snapshot_device", choices=["cuda", "cpu"], default="cuda",
                    help="CPU-resident base snapshot lets >1B fp32 models fit 24GB cards")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--schemes", default="nf4,int8")
    ap.add_argument("--blocksize", type=int, default=64, help="NF4 block size (bitsandbytes default 64)")
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--out_dir", default=os.path.join(HARNESS, "results", "quant_smoke"))
    ap.add_argument("--table_out", default=None)
    ap.add_argument("--reanalyze", default=None,
                    help="STANDALONE CPU reanalysis: read a QS_smoke_raw.npz dir and rewrite the table.")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    if args.reanalyze:
        res = _load_raw(args.reanalyze)
        table = analyze(res)
        out = args.table_out or os.path.join(args.reanalyze, "QS_smoke_table.json")
        _write_json(table, out)
        print_table(table)
        print(f"[qs] reanalysis wrote {out}", flush=True)
        return
    if args.run:
        run_smoke(args)
        return
    ap.error("nothing to do: pass --selftest, --run, or --reanalyze")


def _load_raw(d):
    z = dict(np.load(os.path.join(d, "QS_smoke_raw.npz")))
    arms = {}
    base = {}
    for k in list(z):
        if k.startswith("damage__"):
            n = k[len("damage__"):]
            arms.setdefault(n, {})["damage"] = z[k]
            arms[n]["scheme"] = n.split("_")[0]; arms[n]["locality"] = "_".join(n.split("_")[1:])
        elif k.startswith("esr__"):
            n = k[len("esr__"):]
            arms.setdefault(n, {})["edit_ok"] = z[k]
        elif k.startswith("base__"):
            base[k[len("base__"):]] = z[k]
    meta = dict(model="(reanalyze)", layer=None, n_edits=int(z["damage_fp32"].shape[0]),
                seed=None, schemes=sorted({n.split("_")[0] for n in arms}), blocksize=64)
    return dict(COS=z["COS"], damage_fp32=z["damage_fp32"], edit_ok_fp32=z["edit_ok_fp32"],
                arms=arms, base=base, meta=meta)


if __name__ == "__main__":
    main()
