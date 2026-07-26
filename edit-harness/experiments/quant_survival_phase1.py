"""quant_survival_phase1.py — Paper B, Track-1 Phase-1: does a concentrated knowledge edit AND
its geometric collateral-damage RANKING survive REAL post-training quantization, and does the
per-parameter concentration/bin-width mechanism explain it?

This is the smoke's isolated per-edit protocol (experiments/quant_survival_smoke.py), scaled to
n=200 / 3-seed / real bitsandbytes kernels and extended per the Paper-B prereg
(docs/plans/PREREG-PAPERB-QUANTSURVIVAL-DRAFT-2026-07-16.md, companion DESIGN doc). It REUSES the
smoke's audited primitives by IMPORT (the pilot artifact stays byte-stable): the pure-torch INT8
/ NF4 round-trip codecs, the qlinear bookkeeping, the probe-logit machinery and the within-probe
Spearman; and mirrors experiments/killgate_keygeom.py's editor plumbing for {rome,memit,alpha}.

WHAT IS NEW vs the smoke (all per the prereg):
  1. --editor {rome,memit,alpha}. ROME carries C1/C2/C3; memit/alpha are C1/C3-only (their fp32
     geometry tie is dead — the C2 gating is an analysis-time scope rule, recorded in the table
     metadata as "c2_scope").
  2. REAL KERNELS PRIMARY (--codec real, CUDA-only): bitsandbytes NF4 with DOUBLE-QUANT ON
     (quantize_4bit(compress_statistics=True, quant_type="nf4") + dequantize_4bit round-trip) and
     INT8 (bnb int8_vectorwise_quant per-row absmax round-trip). The smoke's pure-torch codecs run
     behind --codec sim as the CPU cross-check + the selftest path (bnb 4-bit kernels need CUDA, so
     the real path is asserted-CUDA-only; --selftest is CPU-only and uses the sim codecs).
  3. Full-model arm CACHE optimization (prereg §7, design §6): quantize-dequant all NON-edited
     transformer-block linears ONCE per (model,scheme), cache the dequantized tensors, and per edit
     re-quantize ONLY the edited tensor(s). Valid for the per-tensor INT8/NF4 codecs (a tensor's
     quantization is independent of the others). --fullmodel_cache {auto,on,off}: `auto` caches on
     the GPU only when it fits in free VRAM (1B/1.5B), else falls back to the smoke's un-optimized
     per-edit full re-quant (correct, memory-lean — used for 3B).
  4. C3 bin-width mechanism (per editor,scheme): MEASURED bin widths — INT8 b_row=absmax_row/127;
     NF4 b_block = absmax_block * local adjacent-codebook gap at each parameter's level (the NF4
     grid is non-uniform). ratio[k]=|ΔW[k]|/b(k); F_above=mean(ratio>=1), median, p90. PLUS the
     competing M-averaging statistic: r_func = ‖(ΔW_q−ΔW)·x‖/‖ΔW·x‖ on the edit's key activation x
     vs r_param = median|ε|/median|ΔW|. Both in every cell's table (prereg C3 / K3).
  5. Uncertainty: permutation null p (shuffle key-cos vs damage, N>=1000) for each quantized ρ;
     bootstrap 95% CI on ρ_geo (resample edits within the cell). base-subtracted (edit-attributable)
     ρ alongside raw. Δρ vs fp32; ρ_rank (fp32-vs-arm damage rank survival).
  6. Generation checks (bounded subsample --gen_check_n): perplexity of edited+quantized vs
     fp32-edited on a FIXED neutral text slice (documented, counterfact-unrelated — no wikitext on
     disk); paraphrase esr (2 CounterFact paraphrase_prompts/edit) pre/post-quant.

METRIC DISCIPLINE (non-negotiable, inherited): damage = signed within-probe damage_logit
(pre_l(fp32 unedited) − post_l); signed Spearman; AUROC BANNED anywhere; ROME value-opt fp32 (the
model loads fp32; the editors keep their own fp32 casts); PID-only process control (driver side).

CPU selftest (--selftest, NO CUDA): sim-codec round-trip bounds; bin-width formula on synthetic
tensors with analytic answers; spearman/permutation sanity; edit/probe disjointness; and a tiny
random-Llama end-to-end pipeline run on CPU (SKIP if no local tokenizer). Prints ALL CHECKS PASSED.

Standing rules honored: ROME value-opt stays fp32; signed Spearman never AUROC; results quarantined
to results/quant_survival/. See the companion prereg + design docs for the frozen predictions.
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

# Reuse the AUDITED primitives by import (keeps the pilot artifact byte-stable — we never edit it):
#   * signed Spearman + partial Spearman (numpy, NaN-safe, never AUROC)
#   * the smoke's pure-torch INT8/NF4 round-trip codecs + block bookkeeping (sim cross-check path)
from experiments.merging_m0 import _spearman  # noqa: E402
from experiments.quant_survival_smoke import (  # noqa: E402
    NF4_LEVELS, int8_roundtrip, nf4_roundtrip,
    _quant_linears, _snapshot, _restore, _prob_of_token, _probe_logits,
    _within_probe_spearman, _write_json,
)

SCHEMA_VERSION = "qs.phase1.v1"

# Fixed, deterministic, CounterFact-UNRELATED neutral text for the perplexity fluency check.
# DOCUMENTED CHOICE (prereg §4 / design §5): no wikitext-style corpus is on disk under data/; the
# GLUE parquet files require pyarrow and carry sentiment/entailment sentences (not neutral prose).
# We therefore use a fixed multi-sentence encyclopaedic paragraph with no overlap with the
# CounterFact subjects/objects, so perplexity(edited+quant) vs perplexity(fp32-edited) isolates the
# quantization's fluency effect, not a data-leak into the edited fact.
NEUTRAL_PPL_TEXT = (
    "The process of photosynthesis converts light energy into chemical energy stored in glucose. "
    "Ocean currents redistribute heat across the planet and influence regional weather patterns. "
    "A prime number has exactly two distinct positive divisors, one and itself. "
    "Sedimentary rock forms over long periods as mineral particles settle and compact. "
    "The circulatory system transports oxygen and nutrients to cells throughout the body."
)


# ============================================================ real bitsandbytes kernels (CUDA-only)
def _int8_real(w):
    """Real bitsandbytes per-row (vectorwise) absmax INT8 quantize/dequantize round-trip.
    Returns (deq_tensor[same shape/dtype/device], absmax_row[d_out] fp32) — absmax_row feeds the
    C3 bin width b_row=absmax_row/127. CUDA-only, and by design RAISES on any bnb API mismatch
    (no silent codec fallback — the sim path is an explicit `--codec sim` choice, never an
    automatic mask). NOTE (bnb 0.49.2): int8_vectorwise_quant's CUDA kernel hard-rejects fp32
    (backends/cuda/ops.py torch._check A.dtype == torch.float16; bnb itself always casts to fp16
    first, autograd/_functions.py) — so the input MUST be cast to fp16 before the call."""
    import torch
    import bitsandbytes.functional as F
    assert w.is_cuda, "bnb int8 kernel requires CUDA"
    wf = w.detach().to(torch.float16).contiguous()    # bnb int8 CUDA kernel requires fp16 input
    out = F.int8_vectorwise_quant(wf)                 # (q_int8, row_stats(absmax), outlier_cols)
    q = out[0]
    row_stats = out[1].to(torch.float32).reshape(-1)  # per-row absmax
    deq = q.to(torch.float32) * (row_stats / 127.0).reshape(-1, 1)
    return deq.to(w.dtype), row_stats


def _nf4_dq_real(w):
    """Real bitsandbytes NF4 4-bit round-trip with DOUBLE QUANTIZATION ON (compress_statistics).
    Returns (deq_tensor, absmax_block[n_blocks] fp32) — the effective per-block absmax used by the
    kernel, which feeds the C3 bin width b_block = absmax_block*local_gap. CUDA-only."""
    import torch
    import bitsandbytes.functional as F
    assert w.is_cuda, "bnb NF4 kernel requires CUDA"
    wf = w.detach().to(torch.float32).contiguous()
    q, state = F.quantize_4bit(wf, blocksize=64, compress_statistics=True, quant_type="nf4")
    deq = F.dequantize_4bit(q, state, quant_type="nf4").reshape(w.shape)
    # effective per-block absmax: under double-quant state.absmax is the QUANTIZED absmax; bnb's
    # get_absmax dequantizes it. Prefer the dequantized value (design: "include double-quant's
    # quantized absmax from the real kernel path when available"); fall back to the tensor's own
    # per-64-block absmax (bnb's first-level absmax) if the state layout differs across versions.
    absmax_block = None
    try:
        am = state.absmax
        if getattr(state, "state2", None) is not None and getattr(state, "offset", None) is not None:
            am = F.dequantize_blockwise(state.absmax, state.state2) + state.offset
        absmax_block = am.detach().to(torch.float32).reshape(-1)
    except Exception:
        absmax_block = None
    if absmax_block is None:
        nblk = (wf.numel() + 63) // 64
        flat = wf.reshape(-1)
        pad = (-flat.numel()) % 64
        if pad:
            flat = torch.cat([flat, flat.new_zeros(pad)])
        absmax_block = flat.view(-1, 64).abs().amax(dim=1).to(torch.float32).reshape(-1)
        assert absmax_block.numel() == nblk
    return deq.to(w.dtype), absmax_block


def quantize_weight(w, scheme, codec):
    """Dispatch a single-tensor quant/dequant round-trip and return (deq_tensor, binmeta).
    scheme in {'int8','nf4dq'}; codec in {'real','sim'}. binmeta carries the MEASURED absmax the
    C3 bin width needs. For sim, nf4dq maps to the pilot NF4 round-trip (double-quant OMITTED — a
    disclosed CPU cross-check only; the headline numbers must be codec='real')."""
    import torch
    if scheme == "int8":
        if codec == "real":
            deq, absmax_row = _int8_real(w)
        else:
            wf = w.detach().to(torch.float32)
            absmax_row = (wf.abs().amax(dim=1) if wf.dim() == 2 else wf.abs().amax().reshape(1)).clamp(min=1e-12)
            deq = int8_roundtrip(w)
        return deq, {"kind": "int8", "absmax_row": absmax_row.detach().to(torch.float32).cpu().numpy()}
    if scheme == "nf4dq":
        if codec == "real":
            deq, absmax_block = _nf4_dq_real(w)
        else:
            deq = nf4_roundtrip(w, blocksize=64)
            wf = w.detach().to(torch.float32).reshape(-1)
            pad = (-wf.numel()) % 64
            if pad:
                wf = torch.cat([wf, wf.new_zeros(pad)])
            absmax_block = wf.view(-1, 64).abs().amax(dim=1).clamp(min=1e-12)
        return deq, {"kind": "nf4", "blocksize": 64,
                     "absmax_block": absmax_block.detach().to(torch.float32).cpu().numpy()}
    raise ValueError(f"unknown scheme {scheme!r} (expected int8/nf4dq)")


# ============================================================ C3 bin-width mechanism (numpy, CPU)
def _nf4_local_gap(normed):
    """Local adjacent-codebook gap at each normalized value (nearest NF4 level's min distance to
    its neighbours). The NF4 grid is non-uniform (gaps ~0.16 near +-1, ~0.08 near 0); use the LOCAL
    gap, not the max gap. normed in [-1,1] (numpy). Returns the local gap per element."""
    levels = np.asarray(NF4_LEVELS, float)
    mids = (levels[1:] + levels[:-1]) / 2.0
    idx = np.clip(np.searchsorted(mids, normed), 0, len(levels) - 1)   # nearest-level index 0..15
    # gap to the neighbour(s) that exist; edges use their single adjacent gap
    gap_lo = np.where(idx > 0, levels[idx] - levels[np.clip(idx - 1, 0, len(levels) - 1)], np.inf)
    gap_hi = np.where(idx < len(levels) - 1, levels[np.clip(idx + 1, 0, len(levels) - 1)] - levels[idx], np.inf)
    gap = np.minimum(gap_lo, gap_hi)
    # both-inf only if the codebook had 1 level (never) — guard anyway
    gap = np.where(np.isfinite(gap), gap, float(np.max(np.diff(levels))))
    return gap


def bin_width_ratios(dW, W_edited, binmeta):
    """ratio[k] = |ΔW[k]| / b(k) for every edited-tensor parameter (numpy, flat row-major).
    INT8: b_row = absmax_row/127 (broadcast over the row's d_in columns).
    NF4:  b_block(w) = absmax_block * local_gap(w/absmax_block); the parameter's own W_edited value
          sets its NF4 level (bin position), block layout = contiguous 64-wide over reshape(-1).
    Returns the flat ratio array. b(k) is measured on W_edited (the tensor actually quantized)."""
    dW = np.asarray(dW, float)
    We = np.asarray(W_edited, float)
    if binmeta["kind"] == "int8":
        absmax_row = np.asarray(binmeta["absmax_row"], float).clip(min=1e-12)   # [d_out]
        b_row = absmax_row / 127.0
        b = np.broadcast_to(b_row[:, None], We.shape)
        return (np.abs(dW) / np.clip(b, 1e-30, None)).reshape(-1)
    # NF4 blockwise
    bs = int(binmeta["blocksize"])
    absmax_block = np.asarray(binmeta["absmax_block"], float).clip(min=1e-12)   # [n_blocks]
    flat_dW = dW.reshape(-1)
    flat_We = We.reshape(-1)
    n = flat_We.size
    pad = (-n) % bs
    if pad:
        flat_We = np.concatenate([flat_We, np.zeros(pad)])
        flat_dW = np.concatenate([flat_dW, np.zeros(pad)])
    blocks_We = flat_We.reshape(-1, bs)
    nblk = blocks_We.shape[0]
    am = absmax_block[:nblk] if absmax_block.size >= nblk else np.pad(
        absmax_block, (0, nblk - absmax_block.size), constant_values=absmax_block[-1] if absmax_block.size else 1.0)
    normed = blocks_We / am[:, None]
    gap = _nf4_local_gap(normed.reshape(-1)).reshape(nblk, bs)
    b = am[:, None] * gap                                                   # [n_blocks, bs]
    ratio = (np.abs(flat_dW.reshape(-1, bs)) / np.clip(b, 1e-30, None)).reshape(-1)
    return ratio[:n]


def m_averaging_stats(dW, eps, x):
    """M-averaging (prereg C3 competitor): r_func = ‖(ΔW_q−ΔW)·x‖/‖ΔW·x‖ = ‖ε·x‖/‖ΔW·x‖ on the
    edit's key activation x (ε = ΔW_q − ΔW = the rounding error on the edited weight); and
    r_param = median|ε|/median|ΔW| (per-parameter error ratio). r_func ≪ r_param is the signature
    of averaging-driven survival. All numpy."""
    dW = np.asarray(dW, float)
    eps = np.asarray(eps, float)
    x = np.asarray(x, float).reshape(-1)
    sig = dW @ x
    err = eps @ x
    denom = float(np.linalg.norm(sig)) + 1e-30
    r_func = float(np.linalg.norm(err)) / denom
    med_dW = float(np.median(np.abs(dW))) + 1e-30
    r_param = float(np.median(np.abs(eps))) / med_dW
    return r_func, r_param


# ============================================================ uncertainty helpers (numpy, CPU)
def permutation_null_p(cos_flat, dmg_flat, rho_obs, n_perm=1000, seed=0):
    """Two-sided permutation p for Spearman(cos,dmg): shuffle cos vs dmg, fraction of |null ρ| >=
    |observed ρ|. N>=1000 (prereg). Returns p (or None if ρ_obs is not finite)."""
    if rho_obs is None or not np.isfinite(rho_obs):
        return None
    a = np.asarray(cos_flat, float)
    b = np.asarray(dmg_flat, float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if a.size < 5:
        return None
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_perm):
        r = _spearman(a, rng.permutation(b))
        if np.isfinite(r) and abs(r) >= abs(rho_obs):
            hits += 1
    return round((hits + 1) / (n_perm + 1), 5)


def bootstrap_ci_rho(cos, dmg, B=1000, seed=0, alpha=0.05):
    """Bootstrap (2.5, 97.5) percentile CI on pooled Spearman(cos,dmg) by resampling EDITS (rows).
    cos,dmg are [N,M]; resample the N edit rows with replacement. Returns [lo, hi] or [None,None]."""
    cos = np.asarray(cos, float)
    dmg = np.asarray(dmg, float)
    if cos.ndim != 2 or cos.shape[0] < 4:
        return [None, None]
    N = cos.shape[0]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        idx = rng.integers(0, N, N)
        r = _spearman(cos[idx].reshape(-1), dmg[idx].reshape(-1))
        if np.isfinite(r):
            vals.append(r)
    if len(vals) < 10:
        return [None, None]
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return [round(float(lo), 4), round(float(hi), 4)]


# ============================================================ data loader (keeps paraphrases)
def load_counterfact_para(path, n_edits, n_probes, seed=0, n_para=2):
    """Edit + probe banks — the loader is REIMPLEMENTED VERBATIM from killgate_keygeom.py:62-82 /
    quant_survival_smoke.load_counterfact (same json.load -> default_rng(seed).shuffle ->
    requested_rewrite parse -> first-(n_edits+n_probes) slice), so the (seed,n_edits,n_probes)
    selection is BYTE-IDENTICAL to the gate/smoke cells and the COS/damage numbers are directly
    comparable. The ONLY addition: it retains up to n_para CounterFact paraphrase_prompts per
    record for the paraphrase-esr fluency check (edit/probe selection is unchanged)."""
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
        recs.append({"subject": subj, "prompt": prompt, "target_new": tnew, "target_true": ttrue,
                     "paraphrase_prompts": list(d.get("paraphrase_prompts") or [])[:n_para]})
        if len(recs) >= n_edits + n_probes:
            break
    return recs[:n_edits], recs[n_edits:n_edits + n_probes]


# ============================================================ generation-quality checks
def _perplexity(model, tok, text, device, max_len=256):
    """exp(mean next-token CE) of `text` under the CURRENT model weights. One forward; deterministic."""
    import torch
    enc = tok(text, return_tensors="pt", truncation=True, max_length=max_len).to(device)
    ids = enc["input_ids"]
    if ids.shape[1] < 2:
        return float("nan")
    out = model(**enc)
    logits = out.logits[0, :-1, :].float()
    tgt = ids[0, 1:]
    ce = torch.nn.functional.cross_entropy(logits, tgt)
    return float(torch.exp(ce).item())


def _paraphrase_esr(model, tok, edit, device):
    """Mean argmax-esr of the edit's target_new over its (<=2) CounterFact paraphrase prompts under
    the CURRENT model weights. NaN if the record carries no paraphrases."""
    from metrics import efficacy
    paras = edit.get("paraphrase_prompts") or []
    if not paras:
        return float("nan")
    ok = []
    for p in paras:
        ok.append(efficacy(model, tok, p, edit["target_new"], edit.get("target_true"), device)["success"])
    return float(np.mean(ok)) if ok else float("nan")


# ============================================================ analysis
def analyze(res):
    """Build the summary table from the raw per-arm arrays in `res` — all CPU/numpy, signed."""
    COS = res["COS"]
    damage_fp32 = res["damage_fp32"]
    edit_ok_fp32 = res["edit_ok_fp32"]
    arms = res["arms"]
    base = res["base"]
    meta = res["meta"]
    c3 = res.get("c3", {})
    gen = res.get("gen", {})
    n_perm = int(meta.get("n_perm", 1000))
    n_boot = int(meta.get("n_boot", 1000))
    seed = int(meta.get("seed", 0))

    def _rnd(x, k=4):
        return round(float(x), k) if (x is not None and np.isfinite(x)) else None

    rho_fp32_pooled = _spearman(COS.reshape(-1), damage_fp32.reshape(-1))
    rho_fp32_within, wp_n = _within_probe_spearman(COS, damage_fp32)
    n_worked_fp32 = int(np.nansum(edit_ok_fp32 >= 0.5))

    mechanism = {
        "rho_keycos_damage_fp32_pooled": _rnd(rho_fp32_pooled),
        "rho_keycos_damage_fp32_within_probe": _rnd(rho_fp32_within),
        "within_probe_n_cols": wp_n,
        "fp32_law_gate_c2_eligible": bool(np.isfinite(rho_fp32_within) and rho_fp32_within >= 0.30),
        "fp32_pooled_ci95_bootstrap_edits": bootstrap_ci_rho(COS, damage_fp32, B=n_boot, seed=seed),
    }
    esr = {"mean_esr_fp32": _rnd(float(np.nanmean(edit_ok_fp32))), "n_edits_worked_fp32": n_worked_fp32}

    arms_out = {}
    worked = edit_ok_fp32 >= 0.5
    for name, a in arms.items():
        dmg = a["damage"]
        eok = a["edit_ok"]
        bnoise = base.get(name)
        dmg_attr = dmg - bnoise[None, :] if bnoise is not None else dmg
        rho_pool = _spearman(COS.reshape(-1), dmg.reshape(-1))
        rho_pool_attr = _spearman(COS.reshape(-1), dmg_attr.reshape(-1))
        rho_within, _ = _within_probe_spearman(COS, dmg)
        rho_within_attr, _ = _within_probe_spearman(COS, dmg_attr)
        rho_rank = _spearman(damage_fp32.reshape(-1), dmg.reshape(-1))
        rho_rank_attr = _spearman((damage_fp32 - (bnoise[None, :] if bnoise is not None else 0)).reshape(-1),
                                  dmg_attr.reshape(-1))
        added = dmg - damage_fp32
        surv = (float(np.nanmean((eok >= 0.5)[worked])) if worked.any() else float("nan"))
        d_rho_pool = (rho_pool - rho_fp32_pooled) if np.isfinite(rho_pool) and np.isfinite(rho_fp32_pooled) else float("nan")
        d_rho_within = (rho_within - rho_fp32_within) if np.isfinite(rho_within) and np.isfinite(rho_fp32_within) else float("nan")
        arms_out[name] = {
            "locality": a["locality"], "scheme": a["scheme"],
            "mean_esr": _rnd(float(np.nanmean(eok))),
            "esr_survival_given_fp32_worked": _rnd(surv),
            "rho_keycos_damage_pooled": _rnd(rho_pool),
            "rho_keycos_damage_pooled_base_subtracted": _rnd(rho_pool_attr),
            "rho_keycos_damage_within_probe": _rnd(rho_within),
            "rho_keycos_damage_within_probe_base_subtracted": _rnd(rho_within_attr),
            "delta_rho_vs_fp32_pooled": _rnd(d_rho_pool),
            "delta_rho_vs_fp32_within_probe": _rnd(d_rho_within),
            "rho_damage_fp32_vs_arm_rank_survival": _rnd(rho_rank),
            "rho_damage_fp32_vs_arm_rank_survival_base_subtracted": _rnd(rho_rank_attr),
            "permutation_null_p_pooled": permutation_null_p(
                COS.reshape(-1), dmg.reshape(-1), rho_pool, n_perm=n_perm, seed=seed),
            "rho_pooled_ci95_bootstrap_edits": bootstrap_ci_rho(COS, dmg, B=n_boot, seed=seed + 1),
            "added_damage_logit_mean": _rnd(float(np.mean(added)), 5),
            "added_damage_logit_std": _rnd(float(np.std(added)), 5),
            "base_quant_noise_logit_mean_abs": (_rnd(float(np.mean(np.abs(bnoise))), 5)
                                                if bnoise is not None else None),
        }

    # C3 aggregate per scheme (bin-width concentration + M-averaging)
    c3_out = {}
    for scheme, cc in c3.items():
        ratios = np.asarray(cc.get("ratio_pooled", []), float)
        rfunc = np.asarray(cc.get("r_func", []), float)
        rparam = np.asarray(cc.get("r_param", []), float)
        c3_out[scheme] = {
            "n_params_pooled": int(ratios.size),
            "F_above_bin": _rnd(float(np.mean(ratios >= 1.0)) if ratios.size else float("nan")),
            "median_ratio": _rnd(float(np.median(ratios)) if ratios.size else float("nan")),
            "p90_ratio": _rnd(float(np.percentile(ratios, 90)) if ratios.size else float("nan")),
            "M_concentration_holds_median_ge_1": (bool(np.median(ratios) >= 1.0) if ratios.size else None),
            "r_func_mean": _rnd(float(np.nanmean(rfunc)) if rfunc.size else float("nan")),
            "r_param_mean": _rnd(float(np.nanmean(rparam)) if rparam.size else float("nan")),
            # the 0.5 factor below is a REPORTING heuristic for the "r_func ≪ r_param" flag, NOT a
            # prereg gate — the prereg adjudicates M-averaging on the raw r_func/r_param magnitudes.
            "M_averaging_r_func_ll_r_param": (
                bool(np.nanmean(rfunc) < 0.5 * np.nanmean(rparam))
                if (rfunc.size and rparam.size and np.isfinite(np.nanmean(rparam))) else None),
            "note": ("F_above/median_ratio are a CONCENTRATION proxy (survival also depends on "
                     "W_base in-bin position) — reported distinct from the measured C1/C2 survival"),
        }

    # generation-quality aggregate (bounded subsample)
    gen_out = {}
    for name, gg in gen.items():
        ppl = np.asarray(gg.get("ppl", []), float)
        para = np.asarray(gg.get("para_esr", []), float)
        gen_out[name] = {
            "n_gen_probes": int(np.isfinite(ppl).sum()),
            "perplexity_mean": _rnd(float(np.nanmean(ppl)) if ppl.size else float("nan"), 4),
            "paraphrase_esr_mean": _rnd(float(np.nanmean(para)) if para.size else float("nan")),
        }

    return {
        "experiment": "quant_survival_phase1",
        "schema_version": SCHEMA_VERSION,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": meta["model"], "layer": meta["layer"], "editor": meta["editor"],
        "n_edits": meta["n_edits"], "n_probes": COS.shape[1], "seed": meta["seed"],
        "schemes": meta["schemes"], "codec": meta["codec"], "blocksize": 64,
        "fullmodel_cache": meta.get("fullmodel_cache"),
        "edited_layers": meta.get("edited_layers"),
        "c2_scope": ("ROME on this cell carries C1/C2/C3; C2 (geometry-ranking survival) is "
                     "evaluated only where the fp32 within-probe law clears rho>=0.30 (see "
                     "mechanism_tie.fp32_law_gate_c2_eligible)" if meta["editor"] == "rome"
                     else f"editor={meta['editor']} is C1/C3-ONLY, never C2 (fp32 geometry tie is "
                          "dead for this editor — a geometry-survival number would be vacuous)"),
        "quant_note": (f"REAL bitsandbytes kernels (codec={meta['codec']}): NF4 double-quant ON, "
                       "INT8 per-row absmax; sim = pilot pure-torch round-trip (NF4 omits "
                       "double-quant), CPU cross-check only" if meta["codec"] == "real"
                       else "SIM pure-torch round-trip codecs (CPU cross-check; NF4 omits "
                            "double-quant — NOT a headline number)"),
        "damage_metric_note": "signed damage_logit = pre_l(fp32 unedited) − post_l; never AUROC",
        "esr": esr,
        "mechanism_tie": mechanism,
        "arms": arms_out,
        "bin_width_mechanism_C3": c3_out,
        "generation_checks": gen_out,
    }


def print_table(table):
    print("\n=== QUANT-SURVIVAL PHASE-1 ===", flush=True)
    print(f"model={table['model']} L={table['layer']} editor={table['editor']} "
          f"codec={table['codec']} n_edits={table['n_edits']} n_probes={table['n_probes']} "
          f"seed={table['seed']} schemes={table['schemes']}", flush=True)
    m = table["mechanism_tie"]
    print(f"  fp32: rho(key-cos,damage) pooled={m['rho_keycos_damage_fp32_pooled']} "
          f"within={m['rho_keycos_damage_fp32_within_probe']} C2-eligible={m['fp32_law_gate_c2_eligible']} "
          f"mean_esr_fp32={table['esr']['mean_esr_fp32']}", flush=True)
    for name, a in table["arms"].items():
        print(f"  [{name}] esr={a['mean_esr']} surv={a['esr_survival_given_fp32_worked']} "
              f"rho(cos,dmg)={a['rho_keycos_damage_pooled']} Δrho={a['delta_rho_vs_fp32_pooled']} "
              f"rank_surv={a['rho_damage_fp32_vs_arm_rank_survival']} "
              f"perm_p={a['permutation_null_p_pooled']}", flush=True)
    for scheme, c in table["bin_width_mechanism_C3"].items():
        print(f"  [C3 {scheme}] F_above={c['F_above_bin']} median_ratio={c['median_ratio']} "
              f"p90={c['p90_ratio']} r_func={c['r_func_mean']} r_param={c['r_param_mean']}", flush=True)


# ============================================================ GPU run
def _editor_apply(editor):
    if editor == "rome":
        from editors.rome_native import apply_edit
    elif editor == "memit":
        from editors.memit import apply_edit
    elif editor == "alpha":
        from editors.alphaedit import apply_edit
    else:
        raise ValueError(f"unknown editor {editor!r}")
    return apply_edit


def run(args):
    import torch
    import transformers as _tf
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from metrics import first_target_token_id, efficacy
    from editors.rome_native import _capture_key, find_subject_last_token_index
    from editors.arch_compat import normalize_arch

    t0 = time.time()
    device = args.device
    schemes = [s for s in str(args.schemes).split(",") if s]
    for s in schemes:
        if s not in ("int8", "nf4dq"):
            raise SystemExit(f"[qsp] unknown scheme {s!r} (expected int8/nf4dq)")
    codec = args.codec
    if codec == "real" and device != "cuda":
        raise SystemExit("[qsp] --codec real requires --device cuda (bnb 4-bit kernels are CUDA-only; "
                         "use --codec sim for a CPU cross-check)")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(device).eval()
    normalize_arch(model, tok, device)
    nL = model.config.num_hidden_layers
    layer = nL // 2 if args.layer == "auto" else int(args.layer)
    apply_edit = _editor_apply(args.editor)
    print(f"[qsp] loaded {args.model} ({nL} layers, edit layer={layer}, editor={args.editor}, "
          f"codec={codec}, device={device}) {time.time()-t0:.1f}s", flush=True)

    # editor-specific setup
    memit_layers = None
    alpha_proj = None
    if args.editor == "memit":
        from editors.memit import parse_memit_layers
        memit_layers = parse_memit_layers(args.memit_layers, layer, nL)
        print(f"[qsp] memit layers={memit_layers} (z-layer={layer}, identity cov)", flush=True)
    edited_layers = memit_layers if args.editor == "memit" else [layer]

    edits, probes = load_counterfact_para(args.data, args.n_edits, args.n_probes, args.seed)
    N, M = len(edits), len(probes)
    print(f"[qsp] {N} edits, {M} probes (seed {args.seed}); schemes={schemes}", flush=True)
    # disjointness receipt (prompts must not overlap edit/probe banks)
    if set(e["prompt"] for e in edits) & set(p["prompt"] for p in probes):
        raise SystemExit("[qsp] edit/probe prompt overlap — banks not disjoint")

    def key_for(rec):
        idx = find_subject_last_token_index(tok, rec["prompt"], rec.get("subject"))
        return _capture_key(model, tok, layer, rec["prompt"], idx, device).float().cpu().numpy()

    K_edit = np.stack([key_for(e) for e in edits])
    K_probe = np.stack([key_for(p) for p in probes])
    Ke = K_edit / (np.linalg.norm(K_edit, axis=1, keepdims=True) + 1e-8)
    Kp = K_probe / (np.linalg.norm(K_probe, axis=1, keepdims=True) + 1e-8)
    COS = (Ke @ Kp.T).astype(np.float64)
    probe_tok = [first_target_token_id(tok, p["target_true"]) for p in probes]
    print(f"[qsp] keys+cos done {time.time()-t0:.1f}s", flush=True)

    if args.editor == "alpha":
        from editors.alphaedit import build_null_projector
        # C1/C3-only cell; the projector source does not enter any C2 claim. By-construction probe
        # keys (recorded in provenance) — mirrors the smoke's alpha-free geometry measurement.
        alpha_proj = build_null_projector(torch.tensor(K_probe, device=device), args.keep_ratio)

    # fp32 unedited baseline probe logits (the damage reference)
    pre_l = _probe_logits(model, tok, probes, probe_tok, device)

    W_refs = {li: model.model.layers[li].mlp.down_proj.weight for li in edited_layers}
    W_bases = {li: w.detach().clone() for li, w in W_refs.items()}
    Wz = W_refs[layer]
    Wz_base = W_bases[layer]
    edited_ids = {id(w) for w in W_refs.values()}

    qlinears = _quant_linears(model)
    non_edited = [m for m in qlinears if id(m.weight) not in edited_ids]
    base_snap = _snapshot(qlinears)

    # VRAM preflight: the on-device base snapshot is ALREADY resident above, so `free` is
    # post-model+post-snapshot memory. Charging snap_bytes×1.4 against it double-counts the
    # resident snapshot and wrongly aborts a 3B cell on a 32GB card (model 12.6GB + snapshot
    # 11.3GB leaves ~9GB — the 3B cells on box 10263 all FAILed rc 1 at this gate 2026-07-20).
    # What the rest of the run actually needs beyond the resident state: transient deq of ONE
    # edited tensor (≤~300MB), bnb workspace, and the optional fm_cache (auto falls back to off
    # when it doesn't fit, see below). Honest gate: post-snapshot headroom >= HEADROOM_BYTES.
    snap_bytes = sum(m.weight.numel() * m.weight.element_size() for m in qlinears)
    free = total = None
    if device == "cuda":
        free, total = torch.cuda.mem_get_info()
        HEADROOM_BYTES = int(3.5 * (1024 ** 3))
        if free < HEADROOM_BYTES:
            raise SystemExit(
                f"[qsp] VRAM preflight: post-snapshot free {free/1e9:.1f}GB < {HEADROOM_BYTES/1e9:.1f}GB "
                f"headroom (model+snapshot already resident ~{snap_bytes/1e9:.1f}GB) — use a smaller "
                f"model or free the card.")
        print(f"[qsp] VRAM preflight OK: snapshot ~{snap_bytes/1e9:.2f}GB resident, {free/1e9:.1f}GB free", flush=True)

    # ---- full-model cache decision (prereg §7 optimization) ----
    # cache = dequant(quant(W_base)) for EVERY qlinear, once per scheme (base_noise-full uses it
    # directly; per-edit full arm overrides ONLY the edited tensors). auto: cache on GPU iff it fits.
    cache_bytes = snap_bytes * len(schemes)  # one fp32 dequant per qlinear per scheme
    if args.fullmodel_cache == "on":
        use_cache = True
    elif args.fullmodel_cache == "off":
        use_cache = False
    else:  # auto
        use_cache = bool(device == "cuda" and free is not None and (cache_bytes * 1.3) < free)
    print(f"[qsp] full-model arm: cache={'ON' if use_cache else 'OFF (un-optimized per-edit re-quant)'} "
          f"(need ~{cache_bytes/1e9:.1f}GB for {len(schemes)} scheme(s); mode={args.fullmodel_cache})",
          flush=True)

    fm_cache = {}   # scheme -> {id(module): dequant tensor (base)} — ALL qlinears, on device
    if use_cache:
        for scheme in schemes:
            fm_cache[scheme] = {}
            for m in qlinears:
                deq, _ = quantize_weight(m.weight.detach(), scheme, codec)
                fm_cache[scheme][id(m)] = deq
            print(f"[qsp] cached full-model base dequant for scheme={scheme} {time.time()-t0:.1f}s", flush=True)

    localities = ["edited_layer", "full_model"]
    arm_names = [f"{s}_{loc}" for s in schemes for loc in localities]

    def measure_damage():
        return pre_l - _probe_logits(model, tok, probes, probe_tok, device)

    # ---- base arm (unedited quantized): per (scheme, locality) probe logits -> base quant noise ----
    base_noise = {}
    for scheme in schemes:
        # edited_layer locality: quant ONLY the edited layer(s) on the UNEDITED weights
        for li in edited_layers:
            deq, _ = quantize_weight(W_bases[li], scheme, codec)
            with torch.no_grad():
                W_refs[li].copy_(deq)
        base_noise[f"{scheme}_edited_layer"] = measure_damage()
        with torch.no_grad():
            for li in edited_layers:
                W_refs[li].copy_(W_bases[li])
        # full_model locality on the UNEDITED model
        if use_cache:
            with torch.no_grad():
                for m in qlinears:
                    m.weight.copy_(fm_cache[scheme][id(m)])
        else:
            with torch.no_grad():
                for m in qlinears:
                    deq, _ = quantize_weight(m.weight.detach(), scheme, codec)
                    m.weight.copy_(deq)
        base_noise[f"{scheme}_full_model"] = measure_damage()
        _restore(qlinears, base_snap)
    print(f"[qsp] base-arm quant noise done {time.time()-t0:.1f}s", flush=True)

    # ---- accumulators ----
    damage = {n: np.zeros((N, M)) for n in arm_names}
    edit_ok = {n: np.full(N, np.nan) for n in arm_names}
    damage_fp32 = np.zeros((N, M))
    edit_ok_fp32 = np.full(N, np.nan)
    c3 = {s: {"ratio_pooled": [], "r_func": [], "r_param": []} for s in schemes}
    gen = {n: {"ppl": [], "para_esr": []} for n in (["fp32"] + arm_names)}
    gen_n = int(args.gen_check_n)

    def restore_edited_to(state):
        with torch.no_grad():
            for li in edited_layers:
                W_refs[li].copy_(W_bases[li] if state == "base" else W_edited_snap[li])

    for i, e in enumerate(edits):
        # 1) install the edit (fp32; editor keeps its own fp32 value-opt)
        if args.editor == "rome":
            cfg = {"layer": layer, "steps": args.steps, "lr": args.lr}
        elif args.editor == "memit":
            cfg = {"layers": memit_layers, "z_layer": layer, "steps": args.steps, "lr": args.lr,
                   "cov": None, "cov_source": "identity"}
        else:  # alpha
            cfg = {"layer": layer, "steps": args.steps, "lr": args.lr, "projector": alpha_proj}
        apply_edit(model, tok, e, cfg, device)
        W_edited_snap = {li: W_refs[li].detach().clone() for li in edited_layers}
        # z-layer ΔW (the geometry layer) for C3 / M-averaging
        dW_z = (W_edited_snap[layer].detach().float() - Wz_base.float()).cpu().numpy()
        We_z = W_edited_snap[layer].detach().float().cpu().numpy()

        # 2) fp32 measurement
        eff = efficacy(model, tok, e["prompt"], e["target_new"], e.get("target_true"), device)
        edit_ok_fp32[i] = eff["success"]
        damage_fp32[i] = measure_damage()
        if i < gen_n:
            gen["fp32"]["ppl"].append(_perplexity(model, tok, NEUTRAL_PPL_TEXT, device))
            gen["fp32"]["para_esr"].append(_paraphrase_esr(model, tok, e, device))

        # 3) edited_layer arms — quant ONLY the edited layer(s), restore to fp32-edited after each
        for scheme in schemes:
            deq_z = None
            for li in edited_layers:
                deq, binmeta = quantize_weight(W_edited_snap[li], scheme, codec)
                with torch.no_grad():
                    W_refs[li].copy_(deq)
                if li == layer:
                    deq_z, binmeta_z = deq, binmeta
            ef = efficacy(model, tok, e["prompt"], e["target_new"], e.get("target_true"), device)
            edit_ok[f"{scheme}_edited_layer"][i] = ef["success"]
            damage[f"{scheme}_edited_layer"][i] = measure_damage()
            if i < gen_n:
                gen[f"{scheme}_edited_layer"]["ppl"].append(_perplexity(model, tok, NEUTRAL_PPL_TEXT, device))
                gen[f"{scheme}_edited_layer"]["para_esr"].append(_paraphrase_esr(model, tok, e, device))
            # C3 bin-width + M-averaging on the z-layer edited-layer quant (the geometry layer)
            ratio = bin_width_ratios(dW_z, We_z, binmeta_z)
            # Subsample the per-edit ratio array BEFORE pooling (2026-07-20, box 10263 post-mortem):
            # the full down_proj tensor is ~16M (1B) / ~25M (3B) params per edit, so the un-sampled
            # pool is N_edits x that (3.3B-5B values) — the percentile stage and the raw npz explode
            # (observed: 24.5GB npz + ~25 min analysis tail at llama1b_rome_L12_s0). A deterministic
            # stride sample to <=16k values/edit (~3.3M pooled) keeps F_above/median/p90 honest
            # (uniform systematic sample over the tensor) while making the tail ~100x smaller.
            MAX_RATIO_PER_EDIT = 16384
            if ratio.size > MAX_RATIO_PER_EDIT:
                stride = int(np.ceil(ratio.size / MAX_RATIO_PER_EDIT))
                ratio = ratio[::stride]
            eps_z = deq_z.detach().float().cpu().numpy() - We_z          # rounding error on W_edited@z
            rf, rp = m_averaging_stats(dW_z, eps_z, K_edit[i])
            c3[scheme]["ratio_pooled"].append(ratio.astype(np.float32))
            c3[scheme]["r_func"].append(rf)
            c3[scheme]["r_param"].append(rp)
            restore_edited_to("edited")

        # 4) full_model arms — cached non-edited dequant + freshly-quantized edited tensor(s)
        for scheme in schemes:
            with torch.no_grad():
                if use_cache:
                    for m in non_edited:
                        m.weight.copy_(fm_cache[scheme][id(m)])
                else:
                    for m in non_edited:
                        deq, _ = quantize_weight(m.weight.detach(), scheme, codec)
                        m.weight.copy_(deq)
                for li in edited_layers:
                    deq, _ = quantize_weight(W_edited_snap[li], scheme, codec)
                    W_refs[li].copy_(deq)
            ef = efficacy(model, tok, e["prompt"], e["target_new"], e.get("target_true"), device)
            edit_ok[f"{scheme}_full_model"][i] = ef["success"]
            damage[f"{scheme}_full_model"][i] = measure_damage()
            if i < gen_n:
                gen[f"{scheme}_full_model"]["ppl"].append(_perplexity(model, tok, NEUTRAL_PPL_TEXT, device))
                gen[f"{scheme}_full_model"]["para_esr"].append(_paraphrase_esr(model, tok, e, device))
            # restore non-edited to base, edited back to fp32-edited
            _restore(non_edited, [base_snap[qlinears.index(m)] for m in non_edited])
            restore_edited_to("edited")

        # MINOR-1 (restored from the pilot): one-time full-model restore-integrity sweep on the
        # first edit — after the full_model arms, every NON-edited block linear must be back to base
        # and each edited down_proj must hold its fp32 edit. The install is deterministic, so one
        # proof of the base-restore machinery suffices; a silent non_edited-restore corruption
        # (e.g. a bad cache-vs-base copy) cannot then survive unnoticed through the whole cell.
        if i == 0:
            for m, wb in zip(qlinears, base_snap):
                if id(m.weight) in edited_ids:
                    li = next(l for l in edited_layers if W_refs[l] is m.weight)
                    assert torch.allclose(m.weight, W_edited_snap[li], atol=1e-5), \
                        "[qsp] edited down_proj not fp32-edited after full-model arm"
                else:
                    assert torch.allclose(m.weight, wb, atol=1e-5), \
                        "[qsp] a non-edited block linear was not restored to base after full-model arm"
            print("[qsp] full-model restore-integrity sweep PASSED (edit 0)", flush=True)

        # 5) end-of-edit restore to base
        restore_edited_to("base")
        if (i + 1) % 10 == 0:
            print(f"[qsp] edit {i+1}/{N}  {time.time()-t0:.1f}s", flush=True)

    # final restore + verification
    with torch.no_grad():
        for li in edited_layers:
            W_refs[li].copy_(W_bases[li])
    assert all(torch.allclose(W_refs[li], W_bases[li]) for li in edited_layers), "[qsp] final restore FAILED"

    # pool the per-edit ratio arrays for C3
    for scheme in schemes:
        parts = c3[scheme]["ratio_pooled"]
        c3[scheme]["ratio_pooled"] = (np.concatenate(parts) if parts else np.array([]))
        c3[scheme]["r_func"] = np.asarray(c3[scheme]["r_func"], float)
        c3[scheme]["r_param"] = np.asarray(c3[scheme]["r_param"], float)

    res = dict(
        COS=COS, damage_fp32=damage_fp32, edit_ok_fp32=edit_ok_fp32,
        arms={n: {"damage": damage[n], "edit_ok": edit_ok[n],
                  "scheme": n.split("_")[0], "locality": "_".join(n.split("_")[1:])}
              for n in arm_names},
        base=base_noise, c3=c3, gen=gen,
        meta=dict(model=args.model, layer=layer, editor=args.editor, n_edits=N, seed=args.seed,
                  schemes=schemes, codec=codec, fullmodel_cache=("on" if use_cache else "off"),
                  edited_layers=edited_layers, n_perm=args.n_perm, n_boot=args.n_boot,
                  torch=torch.__version__, transformers=_tf.__version__),
    )
    _save_raw(res, args.out_dir)
    table = analyze(res)
    out = args.table_out or os.path.join(args.out_dir, "QS_phase1_table.json")
    _write_json(table, out)
    print_table(table)
    print(f"[qsp] wrote {out}  total {time.time()-t0:.1f}s", flush=True)
    return table


def _save_raw(res, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    arrs = dict(COS=res["COS"], damage_fp32=res["damage_fp32"], edit_ok_fp32=res["edit_ok_fp32"])
    for n, a in res["arms"].items():
        arrs[f"damage__{n}"] = a["damage"]
        arrs[f"esr__{n}"] = a["edit_ok"]
    for n, b in res["base"].items():
        arrs[f"base__{n}"] = b
    for s, cc in res["c3"].items():
        arrs[f"c3_ratio__{s}"] = np.asarray(cc["ratio_pooled"], np.float32)
        arrs[f"c3_rfunc__{s}"] = np.asarray(cc["r_func"], np.float32)
        arrs[f"c3_rparam__{s}"] = np.asarray(cc["r_param"], np.float32)
    tmp = os.path.join(out_dir, "QS_phase1_raw.npz.tmp.npz")
    np.savez_compressed(tmp, **arrs)
    os.replace(tmp, os.path.join(out_dir, "QS_phase1_raw.npz"))


# ============================================================ CPU self-test (NO CUDA)
def _selftest_binwidth():
    """Analytic bin-width checks on synthetic tensors (numpy, no model, no CUDA)."""
    print("[selftest] (a) INT8 bin-width ratio: analytic b_row = absmax_row/127 ...", flush=True)
    W = np.array([[0.0, 0.0, 1.27], [0.0, 2.54, 0.0]], float)          # absmax rows: 1.27, 2.54
    dW = np.array([[0.01, 0.005, 0.0], [0.04, 0.0, 0.02]], float)      # b_row0=0.01, b_row1=0.02
    r = bin_width_ratios(dW, W, {"kind": "int8", "absmax_row": np.array([1.27, 2.54])})
    # ratio row0 col0 = 0.01/0.01 = 1.0; row1 col0 = 0.04/0.02 = 2.0; row1 col2 = 0.02/0.02 = 1.0
    exp = np.array([1.0, 0.5, 0.0, 2.0, 0.0, 1.0])
    assert np.allclose(r, exp, atol=1e-6), f"int8 ratio wrong: {r} != {exp}"
    print(f"[selftest]   INT8 OK — ratios {np.round(r,3).tolist()}", flush=True)

    print("[selftest] (b) NF4 bin-width local-gap: monotone + endpoint sanity ...", flush=True)
    levels = np.array(NF4_LEVELS)
    # a value near 0 has the SMALLEST local gap; a value near +-1 the largest — check ordering
    g_mid = _nf4_local_gap(np.array([0.02]))[0]
    g_edge = _nf4_local_gap(np.array([0.97]))[0]
    assert g_mid < g_edge, f"NF4 local gap not smaller near 0 ({g_mid}) than near 1 ({g_edge})"
    assert g_mid > 0 and g_edge <= float(np.max(np.diff(levels))) + 1e-9
    # ratio: a block whose absmax=1, ΔW equal to the exact local gap must give ratio==1 there
    Wb = np.array([levels[8]], float)               # a mid grid point, block of size... pad to 64
    Wb = np.concatenate([Wb, np.zeros(63)])
    dWb = np.zeros(64); dWb[0] = _nf4_local_gap(np.array([levels[8]]))[0] * 1.0
    rb = bin_width_ratios(dWb, Wb, {"kind": "nf4", "blocksize": 64, "absmax_block": np.array([1.0])})
    assert abs(rb[0] - 1.0) < 1e-6, f"NF4 ratio at exact-gap ΔW should be 1.0, got {rb[0]}"
    print(f"[selftest]   NF4 OK — gap(0.02)={g_mid:.4f} < gap(0.97)={g_edge:.4f}, exact-gap ratio=1", flush=True)

    print("[selftest] (c) M-averaging: functional cancellation vs per-param error ...", flush=True)
    rng = np.random.default_rng(0)
    d_out, d_in = 40, 64
    x = rng.standard_normal(d_in)
    dW = np.outer(rng.standard_normal(d_out), x) / (x @ x)             # rank-one aligned with x
    eps = rng.standard_normal((d_out, d_in)) * 0.05                    # zero-mean rounding-like error
    rf, rp = m_averaging_stats(dW, eps, x)
    assert np.isfinite(rf) and np.isfinite(rp) and rf >= 0 and rp >= 0, "M-averaging non-finite"
    print(f"[selftest]   M-averaging OK — r_func={rf:.4f} r_param={rp:.4f}", flush=True)


def _selftest_sim_codecs():
    """The pilot sim codecs must still round-trip within their bounds (imported, not re-implemented)."""
    import torch
    print("[selftest] (d) sim INT8/NF4 round-trip bounds (imported pilot codecs) ...", flush=True)
    rng = np.random.default_rng(1)
    w = torch.tensor(rng.standard_normal((17, 128)) * 2.0)
    d8 = int8_roundtrip(w)
    absmax = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    assert bool(((d8.float() - w.float()).abs() <= absmax / 2 + 1e-6).all()), "sim int8 half-step violated"
    d4 = nf4_roundtrip(w, blocksize=64)
    # NF4 dequant values must lie on level*absmax grid
    blocks = w.reshape(-1, 64); am = blocks.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    ratio = (d4.reshape(-1, 64) / am).reshape(-1).numpy()
    lv = np.array(NF4_LEVELS)
    nearest = lv[np.argmin(np.abs(ratio[:, None] - lv[None, :]), axis=1)]
    assert bool(np.all(np.abs(ratio - nearest) < 1e-5)), "sim NF4 off grid"
    print("[selftest]   sim codecs OK (int8 half-step, NF4 on-grid)", flush=True)


def _selftest_stats():
    print("[selftest] (e) spearman + permutation + bootstrap sanity ...", flush=True)
    rng = np.random.default_rng(2)
    x = rng.standard_normal((30, 8))
    y = x + rng.standard_normal((30, 8)) * 0.3                          # strongly correlated
    r = _spearman(x.reshape(-1), y.reshape(-1))
    assert r > 0.7, f"spearman on correlated data too low: {r}"
    p_sig = permutation_null_p(x.reshape(-1), y.reshape(-1), r, n_perm=200, seed=0)
    assert p_sig is not None and p_sig < 0.05, f"perm p on real signal should be small: {p_sig}"
    yn = rng.standard_normal((30, 8))                                   # independent
    rn = _spearman(x.reshape(-1), yn.reshape(-1))
    p_null = permutation_null_p(x.reshape(-1), yn.reshape(-1), rn, n_perm=200, seed=0)
    assert p_null is not None, "perm p should be defined"
    ci = bootstrap_ci_rho(x, y, B=200, seed=0)
    assert ci[0] is not None and ci[0] <= r <= ci[1] + 1e-6, f"bootstrap CI {ci} excludes rho {r}"
    print(f"[selftest]   stats OK — rho={r:.3f} perm_p_signal={p_sig} ci95={ci}", flush=True)


def _selftest_disjoint_and_loader():
    print("[selftest] (f) loader disjointness (edit/probe banks) ...", flush=True)
    data = os.path.join(HARNESS, "data", "counterfact.json")
    if not os.path.isfile(data):
        print("[selftest]   loader: SKIP (no counterfact.json)", flush=True)
        return
    edits, probes = load_counterfact_para(data, 20, 20, seed=0)
    assert len(edits) == 20 and len(probes) == 20, "loader wrong bank sizes"
    ep = set(e["prompt"] for e in edits); pp = set(p["prompt"] for p in probes)
    assert not (ep & pp), "edit/probe banks overlap"
    # same shuffle as the smoke/killgate loader -> byte-identical selection at (seed,n)
    from experiments.quant_survival_smoke import load_counterfact as smoke_loader
    se, sp = smoke_loader(data, 20, 20, seed=0)
    assert [e["prompt"] for e in edits] == [e["prompt"] for e in se], "selection diverged from pilot loader"
    print(f"[selftest]   loader OK — disjoint, {len(edits)} edits/{len(probes)} probes, "
          "selection matches the pilot loader byte-for-byte", flush=True)


def _tiny_e2e_cpu():
    """Best-effort tiny random-Llama CPU end-to-end (sim codec) — plumbing check, not science.
    SKIP loudly if no local tokenizer. Runs a 2-edit/3-probe cell through run() with device=cpu."""
    root = os.path.join(HARNESS, "data", "models")
    tokdir = None
    for p in ("Qwen2.5-0.5B", "Llama-3.2-1B", "Qwen2.5-1.5B"):
        d = os.path.join(root, p)
        if os.path.isfile(os.path.join(d, "config.json")) and any(
                os.path.isfile(os.path.join(d, f)) for f in ("tokenizer.json", "tokenizer.model")):
            tokdir = d
            break
    if tokdir is None:
        print("[selftest] (g) tiny e2e: SKIP (no local tokenizer under data/models)", flush=True)
        return None
    try:
        import torch
        from transformers import AutoTokenizer, LlamaForCausalLM, LlamaConfig
    except Exception as ex:
        print(f"[selftest] (g) tiny e2e: SKIP (transformers unavailable: {ex})", flush=True)
        return None
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    tok = AutoTokenizer.from_pretrained(tokdir)
    vocab = max(int(len(tok)), int(getattr(tok, "vocab_size", 0) or 0)) + 16
    cfg = LlamaConfig(vocab_size=vocab, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=64,
                      tie_word_embeddings=True)
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg).to("cpu").float().eval()
    edits = [{"subject": "Paris", "prompt": "Paris is the capital of", "target_new": "Spain",
              "target_true": "France", "paraphrase_prompts": ["The capital city Paris belongs to"]},
             {"subject": "Rome", "prompt": "Rome is the capital of", "target_new": "Egypt",
              "target_true": "Italy", "paraphrase_prompts": ["Rome serves as the capital of"]}]
    probes = [{"subject": "Berlin", "prompt": "Berlin is the capital of", "target_new": "Peru",
               "target_true": "Germany", "paraphrase_prompts": []},
              {"subject": "Tokyo", "prompt": "Tokyo is the capital of", "target_new": "Chad",
               "target_true": "Japan", "paraphrase_prompts": []},
              {"subject": "Cairo", "prompt": "Cairo is the capital of", "target_new": "Cuba",
               "target_true": "Egypt", "paraphrase_prompts": []}]

    # monkey-load: drive run()'s body by swapping the loader/model to the tiny in-memory pieces
    class A:
        model = tokdir; data = None; n_edits = 2; n_probes = 3; layer = "2"; seed = 0
        steps = 2; lr = 0.1; schemes = "nf4dq,int8"; codec = "sim"; device = "cpu"
        editor = "rome"; keep_ratio = 0.99; memit_layers = "auto"; gen_check_n = 2
        n_perm = 50; n_boot = 50; fullmodel_cache = "off"
        out_dir = os.path.join(HARNESS, "results", "quant_survival", "selftest"); table_out = None
    a = A()
    table = _run_on_tiny(model, tok, 2, a, edits, probes)
    assert set(table["arms"]) == {f"{s}_{loc}" for s in ("nf4dq", "int8")
                                  for loc in ("edited_layer", "full_model")}, "arm set wrong"
    for name, arm in table["arms"].items():
        s = arm["esr_survival_given_fp32_worked"]
        assert s is None or 0.0 <= s <= 1.0, f"survival out of range {name}"
    assert set(table["bin_width_mechanism_C3"]) == {"nf4dq", "int8"}, "C3 schemes missing"
    print(f"[selftest]   tiny e2e OK — {len(table['arms'])} arms, C3+gen present, "
          f"fp32 mean_esr={table['esr']['mean_esr_fp32']}", flush=True)
    return True


def _run_on_tiny(model, tok, layer, args, edits, probes):
    """The run() GPU-body, factored to drive the tiny CPU e2e on an in-memory model without a loader.
    Mirrors run() exactly for editor=rome / codec=sim / fullmodel_cache=off (the CPU-testable path)."""
    import torch
    from metrics import first_target_token_id, efficacy
    from editors.rome_native import _capture_key, find_subject_last_token_index, apply_edit
    device = "cpu"
    schemes = [s for s in str(args.schemes).split(",") if s]
    codec = "sim"
    N, M = len(edits), len(probes)
    edited_layers = [layer]

    def key_for(rec):
        idx = find_subject_last_token_index(tok, rec["prompt"], rec.get("subject"))
        return _capture_key(model, tok, layer, rec["prompt"], idx, device).float().cpu().numpy()

    K_edit = np.stack([key_for(e) for e in edits]); K_probe = np.stack([key_for(p) for p in probes])
    Ke = K_edit / (np.linalg.norm(K_edit, axis=1, keepdims=True) + 1e-8)
    Kp = K_probe / (np.linalg.norm(K_probe, axis=1, keepdims=True) + 1e-8)
    COS = (Ke @ Kp.T).astype(np.float64)
    probe_tok = [first_target_token_id(tok, p["target_true"]) for p in probes]
    pre_l = _probe_logits(model, tok, probes, probe_tok, device)

    W_refs = {layer: model.model.layers[layer].mlp.down_proj.weight}
    W_bases = {layer: W_refs[layer].detach().clone()}
    Wz_base = W_bases[layer]
    edited_ids = {id(W_refs[layer])}
    qlinears = _quant_linears(model); non_edited = [m for m in qlinears if id(m.weight) not in edited_ids]
    base_snap = _snapshot(qlinears)
    localities = ["edited_layer", "full_model"]
    arm_names = [f"{s}_{loc}" for s in schemes for loc in localities]

    def measure_damage():
        return pre_l - _probe_logits(model, tok, probes, probe_tok, device)

    base_noise = {}
    for scheme in schemes:
        deq, _ = quantize_weight(W_bases[layer], scheme, codec)
        with torch.no_grad():
            W_refs[layer].copy_(deq)
        base_noise[f"{scheme}_edited_layer"] = measure_damage()
        with torch.no_grad():
            W_refs[layer].copy_(W_bases[layer])
            for m in qlinears:
                dq, _ = quantize_weight(m.weight.detach(), scheme, codec); m.weight.copy_(dq)
        base_noise[f"{scheme}_full_model"] = measure_damage()
        _restore(qlinears, base_snap)

    damage = {n: np.zeros((N, M)) for n in arm_names}; edit_ok = {n: np.full(N, np.nan) for n in arm_names}
    damage_fp32 = np.zeros((N, M)); edit_ok_fp32 = np.full(N, np.nan)
    c3 = {s: {"ratio_pooled": [], "r_func": [], "r_param": []} for s in schemes}
    gen = {n: {"ppl": [], "para_esr": []} for n in (["fp32"] + arm_names)}

    for i, e in enumerate(edits):
        apply_edit(model, tok, e, {"layer": layer, "steps": args.steps, "lr": args.lr}, device)
        W_edited = W_refs[layer].detach().clone()
        dW_z = (W_edited.float() - Wz_base.float()).cpu().numpy(); We_z = W_edited.float().cpu().numpy()
        eff = efficacy(model, tok, e["prompt"], e["target_new"], e.get("target_true"), device)
        edit_ok_fp32[i] = eff["success"]; damage_fp32[i] = measure_damage()
        gen["fp32"]["ppl"].append(_perplexity(model, tok, NEUTRAL_PPL_TEXT, device))
        gen["fp32"]["para_esr"].append(_paraphrase_esr(model, tok, e, device))
        for scheme in schemes:
            deq, binmeta = quantize_weight(W_edited, scheme, codec)
            with torch.no_grad():
                W_refs[layer].copy_(deq)
            ef = efficacy(model, tok, e["prompt"], e["target_new"], e.get("target_true"), device)
            edit_ok[f"{scheme}_edited_layer"][i] = ef["success"]; damage[f"{scheme}_edited_layer"][i] = measure_damage()
            gen[f"{scheme}_edited_layer"]["ppl"].append(_perplexity(model, tok, NEUTRAL_PPL_TEXT, device))
            gen[f"{scheme}_edited_layer"]["para_esr"].append(_paraphrase_esr(model, tok, e, device))
            ratio = bin_width_ratios(dW_z, We_z, binmeta)
            eps_z = deq.float().cpu().numpy() - We_z
            rf, rp = m_averaging_stats(dW_z, eps_z, K_edit[i])
            c3[scheme]["ratio_pooled"].append(ratio.astype(np.float32)); c3[scheme]["r_func"].append(rf); c3[scheme]["r_param"].append(rp)
            with torch.no_grad():
                W_refs[layer].copy_(W_edited)
        for scheme in schemes:
            with torch.no_grad():
                for m in non_edited:
                    dq, _ = quantize_weight(m.weight.detach(), scheme, codec); m.weight.copy_(dq)
                dq, _ = quantize_weight(W_edited, scheme, codec); W_refs[layer].copy_(dq)
            ef = efficacy(model, tok, e["prompt"], e["target_new"], e.get("target_true"), device)
            edit_ok[f"{scheme}_full_model"][i] = ef["success"]; damage[f"{scheme}_full_model"][i] = measure_damage()
            gen[f"{scheme}_full_model"]["ppl"].append(_perplexity(model, tok, NEUTRAL_PPL_TEXT, device))
            gen[f"{scheme}_full_model"]["para_esr"].append(_paraphrase_esr(model, tok, e, device))
            _restore(non_edited, [base_snap[qlinears.index(m)] for m in non_edited])
            with torch.no_grad():
                W_refs[layer].copy_(W_edited)
        if i == 0:  # MINOR-1 sweep, exercised on CPU by the tiny e2e
            for m, wb in zip(qlinears, base_snap):
                if id(m.weight) in edited_ids:
                    assert torch.allclose(m.weight, W_edited, atol=1e-5), "[qsp] tiny: edited not fp32-edited"
                else:
                    assert torch.allclose(m.weight, wb, atol=1e-5), "[qsp] tiny: non-edited not restored"
        with torch.no_grad():
            W_refs[layer].copy_(W_bases[layer])
    with torch.no_grad():
        W_refs[layer].copy_(W_bases[layer])
    for scheme in schemes:
        parts = c3[scheme]["ratio_pooled"]
        c3[scheme]["ratio_pooled"] = (np.concatenate(parts) if parts else np.array([]))
        c3[scheme]["r_func"] = np.asarray(c3[scheme]["r_func"], float)
        c3[scheme]["r_param"] = np.asarray(c3[scheme]["r_param"], float)
    res = dict(
        COS=COS, damage_fp32=damage_fp32, edit_ok_fp32=edit_ok_fp32,
        arms={n: {"damage": damage[n], "edit_ok": edit_ok[n], "scheme": n.split("_")[0],
                  "locality": "_".join(n.split("_")[1:])} for n in arm_names},
        base=base_noise, c3=c3, gen=gen,
        meta=dict(model="tiny", layer=layer, editor="rome", n_edits=N, seed=0, schemes=schemes,
                  codec="sim", fullmodel_cache="off", edited_layers=[layer], n_perm=args.n_perm,
                  n_boot=args.n_boot))
    return analyze(res)


def selftest():
    print("[selftest] quant-survival phase-1 — CPU (NO CUDA)", flush=True)
    _selftest_binwidth()
    _selftest_sim_codecs()
    _selftest_stats()
    _selftest_disjoint_and_loader()
    e2e = _tiny_e2e_cpu()
    tail = "" if e2e else " [tiny e2e SKIPPED — no local tokenizer]"
    print("\n[selftest] ALL CHECKS PASSED (bin-width analytic + sim codecs + stats + disjointness" +
          (" + tiny e2e pipeline" if e2e else "") + ")" + tail, flush=True)
    return True


# ============================================================ CLI
def main():
    ap = argparse.ArgumentParser(description="Quant-survival Paper-B Track-1 Phase-1.")
    ap.add_argument("--selftest", action="store_true", help="CPU self-test (no GPU, no bnb kernels).")
    ap.add_argument("--run", action="store_true", help="GPU/CPU cell run.")
    ap.add_argument("--model", default=os.path.join(HARNESS, "data", "models", "Llama-3.2-1B"))
    ap.add_argument("--data", default=os.path.join(HARNESS, "data", "counterfact.json"))
    ap.add_argument("--editor", choices=["rome", "memit", "alpha"], default="rome")
    ap.add_argument("--n_edits", type=int, default=200)
    ap.add_argument("--n_probes", type=int, default=200)
    ap.add_argument("--layer", default="12")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--schemes", default="nf4dq,int8", help="comma list of int8/nf4dq")
    ap.add_argument("--codec", choices=["real", "sim"], default="real",
                    help="real = bitsandbytes kernels (NF4 double-quant + INT8, CUDA-only, PRIMARY); "
                         "sim = pilot pure-torch round-trip (CPU cross-check; NF4 omits double-quant)")
    ap.add_argument("--fullmodel_cache", choices=["auto", "on", "off"], default="auto",
                    help="full-model arm cache: quantize non-edited linears once (auto=on iff it "
                         "fits free VRAM; off=un-optimized per-edit re-quant, memory-lean for 3B)")
    ap.add_argument("--keep_ratio", type=float, default=0.99, help="AlphaEdit projector keep ratio")
    ap.add_argument("--memit_layers", default="auto", help="MEMIT layer span ('auto'=4 ending at --layer)")
    ap.add_argument("--gen_check_n", type=int, default=40,
                    help="edits (first N) that also get perplexity + paraphrase-esr generation checks")
    ap.add_argument("--n_perm", type=int, default=1000, help="permutation-null draws (prereg N>=1000)")
    ap.add_argument("--n_boot", type=int, default=1000, help="bootstrap resamples for the rho CI")
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--out_dir", default=os.path.join(HARNESS, "results", "quant_survival"))
    ap.add_argument("--table_out", default=None)
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    if args.run:
        run(args)
        return
    ap.error("nothing to do: pass --selftest or --run")


if __name__ == "__main__":
    main()
