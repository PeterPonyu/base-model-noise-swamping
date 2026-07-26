"""quantedit_e0.py — QuantEdit E0: oracle-first CPU kill-gate (rank #7 in
EXPANSION-DIRECTIONS-DEEP-2026-07-01.md, S2.7).

Hypothesis. For a rank-one edit ΔW_i = outer(A_i, B_i) applied to Wbase, a
closed-form ROW-EFFECTIVE margin — the edit's per-row shift norm relative to
that row's OWN post-training-quantization round-off noise, computed from the
banked (K, A, B, Wbase) factors alone, with zero GPU / model access — predicts
how much of ΔW_i actually survives 4-bit-class weight quantization. If it
does, the margin is a zero-cost a-priori "will this edit get washed out under
PTQ" predictor; if it does not, QuantEdit is killed here, before any of the
140+ GPU-minute behavioral rungs (E1+) are spent.

CLAIM SCOPE (NARROWED per the 2026-07-11 literature re-audit — VERDICT: NARROWED).
  arXiv 2605.15138 (MANSU, May 2026, *unlearning*) already published BOTH (a) the
  sub-bin-width grid-crossing physics (knowledge-editing updates lie 47-828x below the
  NF4 bin width and are therefore washed out by PTQ) AND (b) a no-retraining
  per-parameter magnitude-FLOOR fix. Therefore:
    - The ONLY novel contribution admissible here is the *per-edit, closed-form, a-priori
      survival PREDICTOR* computed from a rank-one EDIT's own (k, v-Wk) geometry, for
      insertion-class locate-then-edit editing (ROME / MEMIT / AlphaEdit), NOT unlearning.
    - Any margin-floor-FIX is a corollary/application and MUST cite 2605.15138 as the
      mechanism's origin — it is never presented as new. E0 implements NO floor fix
      (oracle + predictor ONLY); the fix, if pursued, lands in a later rung.
  Prior art for the phenomenon (aggregate PTQ x editing collapse) — cited together, none
  of which give a per-edit closed-form a-priori predictor: 2407.06483 (Composable
  Interventions, ICLR'25; aggregate 2-4-bit collapse), 2410.16454 (edit-then-quantize
  degradation), 2605.15138 (MANSU; sub-bin-width mechanism + floor fix, unlearning).

Two closed-form quantities per edit i (bank = one (model, layer, seed) npz):

  MARGIN_i   (predictor, needs Wbase quantized ONCE per rung, not per edit)
      row r shift is EXACTLY A_i[r] * B_i (rank-one row scaling), so
      ||shift_r||_2 = |A_i[r]| * ||B_i||_2 (no approximation). Normalize by
      that row's OWN quantization round-off norm under the rung's grid,
      rowerr_base[r] = ||Q(Wbase)[r,:] - Wbase[r,:]||_2 (computed once per
      rung, shared across all edits in the bank):
          margin_i = median_r( |A_i[r]| * ||B_i||_2 / (rowerr_base[r] + eps) )
      Values >> 1 mean the edit's row-shift dwarfs the ambient quantization
      noise at that row (edit should survive); values << 1 mean the edit is
      smaller than a quantization bin and is expected to be washed out.
      NOTE: this is deliberately NOT the naive entry-wise |ΔW_ij| vs bin-width
      comparison (audited out in the expansion doc — entries trivially scale
      with |B_i[j]| within a row, so a per-entry margin says nothing an
      aggregate doesn't already say cleaner); this is the row-effective
      aggregate the doc calls for.

  SURVIVAL_i (oracle / target, needs one full-matrix requantization per edit)
      survival_i = ||Q(Wbase + ΔW_i) - Q(Wbase)||_F / ||ΔW_i||_F
      where ||ΔW_i||_F = ||A_i||_2 * ||B_i||_2 exactly (rank-one Frobenius
      norm, no need to materialize ΔW_i for this scalar). Q(Wbase+ΔW_i) is
      requantized PER EDIT (group absmax recomputed on the post-edit matrix —
      this assumes PTQ happens AFTER editing, i.e. the deployed model is
      quantized post-hoc; if quantization instead happened before editing,
      this proxy would need to change). survival_i in [0, ~1+]: 1.0 = the
      edit's weight-space footprint fully survives round-trip quantization;
      near 0 = the edit is quantization-noise-indistinguishable from the
      unedited baseline ("washed out").

Quantization schemes simulated (documented here since there is no bnb/GPU
dependency — everything is hand-rolled numpy):
  - uniform RTN (round-to-nearest), SYMMETRIC, per-row-group absmax scaling:
    for a group of `group` contiguous columns, scale = max(|group|) / qmax
    with qmax = 2**(bits-1) - 1 (e.g. bits=4 -> qmax=7), code = clip(round(x
    / scale), -qmax, qmax), dequant = code * scale. This is the standard
    weight-only RTN baseline used by GPTQ/AWQ-style papers minus calibration.
  - NF4: the published bitsandbytes/QLoRA 4-bit NormalFloat codebook (16
    fixed quantiles in [-1, 1], Dettmers et al. 2023, hardcoded below),
    per-group absmax normalization (scale = max(|group|)), nearest-codebook
    lookup via searchsorted, dequant = code_value * scale. NOT included: bnb's
    double-quantization of the scale constants themselves (a further ~0.5
    bit/param compression of the scales) — out of scope, noted as a caveat.
  - Rung ladder (from the pre-registered E0 design, "int8 -> int3-g32 + NF4"):
    int8_g64, int4_g64, int3_g32 (uniform), nf4_g64 (non-uniform). Group
    sizes are hand-picked to common PTQ practice (64 = bnb default block,
    32 for the most aggressive int3 rung), NOT swept — a documented choice,
    not a discovered optimum.

Pre-registered kill-gate (EXPANSION-DIRECTIONS-DEEP-2026-07-01.md:164, E0):
  KILL if Spearman(margin, survival) < 0.8 at the "transition rung" (the rung
  with the most cross-edit variance in survival, i.e. argmax_rung std(survival)),
  OR if survival has no cross-edit variance (std < 0.05) at EVERY rung (all
  edits saturate to ~full survival or ~total washout — no signal to predict).
  Both bars are exposed as --gate_rho / --gate_var_floor (defaults 0.80/0.05)
  but default to the pre-registered values; do not lower them post-hoc to
  manufacture a PASS.

Modes:
  --validate_npz PATH [PATH ...]
      The hard gate killgate_keygeom.py's --save_vectors WARN text points to:
      asserts K/A/B/Wbase present, shapes mutually consistent, and
      recon_rel_err <= --recon_threshold (default 1e-3, matching killgate's
      own vectors_valid definition) for every edit. Exits nonzero on ANY
      structural problem or numerical violation — never silently continues.
      Prints N valid edits (== N on PASS) per file.

  --npz PATH [...]  /  --from_vectors PATH [...]   (default mode: run the E0 science)
      Two spellings of the same input (merged, de-duplicated). --from_vectors is
      the name the E0 spec / run_quantedit_e0.sh use. Validates each bank first
      (aborts that bank on failure — the science never runs on unvalidated
      vectors), then computes MARGIN/SURVIVAL per edit per rung, the gate
      Spearman, and the PASS/KILL verdict. Writes --out (default
      results/QUANTEDIT_E0.json; the driver writes results/quantedit/E0_oracle_table.json).

  --selftest
      CPU-only, no npz/model/GPU. (a) quantizer unit tests (RTN round-trip <=
      half step, monotone level count, NF4 16-level + absmax); (b) a clean
      strength-swept synthetic bank confirming the tool detects a STRONG
      monotone margin<->survival relationship (reports the actual signed rho —
      it is strong-but-NEGATIVE with the Frobenius-ratio oracle, see below) plus
      a direct gate_verdict unit test hitting all three verdict branches; (c) a
      no-variance fixture -> KILL. Exits nonzero on any failure. Writes
      results/quantedit/selftest/SELFTEST_report.json.

  --ladder {spec,legacy}   (default spec)
      'spec' = int8/int6/int4/int3 group-32 + NF4 group-32 (the S2.7 rung ladder
      this file's title claims). 'legacy' = the first-draft int8_g64/int4_g64/
      int3_g32/nf4_g64 ladder (reproduces the original results/QUANTEDIT_E0.json).

Usage:
  python experiments/quantedit_e0.py --selftest
  python experiments/quantedit_e0.py --validate_npz \
        results/vectors/vectors_qv_llama1b_rome_cf_L12_s0.npz
  python experiments/quantedit_e0.py --from_vectors \
        results/vectors/vectors_qv_llama1b_rome_cf_L8_s0.npz \
        results/vectors/vectors_qv_llama1b_rome_cf_L12_s0.npz \
        --ladder spec --out results/quantedit/E0_oracle_table.json

IMPORTANT — what "survival" measures, and why the real KILL is partly definitional.
  survival_i = ||Q(Wbase+ΔW_i) - Q(Wbase)||_F / ||ΔW_i||_F is a weight-space
  Frobenius RATIO. It is a DECREASING function of edit magnitude: as an edit
  shrinks, its own imprint ||ΔW|| falls ~linearly while the residual requant
  noise floor ||Q(Wbase+ΔW)-Q(Wbase)|| falls only ~sqrt(magnitude), so the ratio
  DIVERGES for tiny edits and -> 1 for large edits. The margin INCREASES with
  magnitude. Hence margin and survival are STRUCTURALLY anti-correlated, and the
  gate's signed KILL is expected on clean synthetic data too (see --selftest (b)).
  Read the KILL as "this closed-form margin does not (positively) predict this
  weight-space survival ratio", NOT as a physical claim that geometry cannot
  matter under PTQ. The bounded projection-retention oracle
  <(Q(W+ΔW)-Q(W))·k, r>/<r,r> is now IMPLEMENTED (margin_survival_projret) and
  emitted into the output table. On the real banks it is DEGENERATE — mean ~1.002,
  std ~0.001, no cross-edit variance — i.e. the S2.7 pre-registered live null:
  every edit survives equally, so nothing can be predicted (margin<->projret rho
  is a weak, meaningless +0.10 to +0.14 on the real banks / ~+0.08 on synthetic).
  Both oracles therefore kill QuantEdit at E0, for complementary reasons
  (Frobenius: definitional anti-correlation; projection: no-variance live null).

CPU-only. numpy on existing .npz. No GPU / torch / model download — this is a
SIMULATED quantization-survival oracle, not a real quantized-model behavioral
eval (that is E1 in the pre-registered ladder).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    from analyze_matrices import spearman  # noqa: E402
except Exception:  # pragma: no cover - fallback replica (keeps this script standalone)
    def spearman(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        m = np.isfinite(a) & np.isfinite(b)
        a, b = a[m], b[m]
        if a.size < 3:
            return np.nan
        ar = a.argsort().argsort().astype(float)
        br = b.argsort().argsort().astype(float)
        if ar.std() == 0 or br.std() == 0:
            return np.nan
        return float(np.corrcoef(ar, br)[0, 1])


# Published bitsandbytes/QLoRA NF4 codebook (Dettmers et al. 2023): 16 fixed
# quantiles of a folded/scaled unit normal, in [-1, 1], sorted ascending.
NF4_CODES = np.array([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343976497650, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
    0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
    0.7229568362236023, 1.0,
], dtype=np.float32)

# Rung ladder.
#
# LEGACY (unchanged; reproduces the pre-existing results/QUANTEDIT_E0.json byte-for-byte
# when selected via --ladder legacy). This was the first draft's ladder.
RUNGS = [
    {"name": "int8_g64", "kind": "uniform", "bits": 8, "group": 64},
    {"name": "int4_g64", "kind": "uniform", "bits": 4, "group": 64},
    {"name": "int3_g32", "kind": "uniform", "bits": 3, "group": 32},
    {"name": "nf4_g64", "kind": "nf4", "bits": 4, "group": 64},
]

# SPEC (new default, 2026-07-11): the ladder the E0 design (S2.7) literally enumerates —
# "RTN int8/int6/int4/int3 with group size 32, plus hand-rolled NF4". Differs from LEGACY
# by (a) adding the int6 rung, (b) uniform group size 32 for every uniform rung (LEGACY used
# g64 for int8/int4), (c) NF4 at g32. Selected by default so a bare invocation is
# spec-compliant; LEGACY stays available (and reproducible) via --ladder legacy.
SPEC_RUNGS = [
    {"name": "int8_g32", "kind": "uniform", "bits": 8, "group": 32},
    {"name": "int6_g32", "kind": "uniform", "bits": 6, "group": 32},
    {"name": "int4_g32", "kind": "uniform", "bits": 4, "group": 32},
    {"name": "int3_g32", "kind": "uniform", "bits": 3, "group": 32},
    {"name": "nf4_g32", "kind": "nf4", "bits": 4, "group": 32},
]

LADDERS = {"spec": SPEC_RUNGS, "legacy": RUNGS}

EPS = 1e-8


def quant_uniform(X, bits, group):
    """Symmetric per-row-group absmax RTN quantization, dequantized in place."""
    d_out, d_in = X.shape
    if d_in % group != 0:
        raise ValueError(f"d_in={d_in} not divisible by group={group}")
    Xg = X.reshape(d_out, d_in // group, group)
    qmax = 2 ** (bits - 1) - 1
    scale = np.maximum(np.max(np.abs(Xg), axis=-1, keepdims=True), EPS) / qmax
    code = np.clip(np.round(Xg / scale), -qmax, qmax)
    return (code * scale).reshape(d_out, d_in).astype(np.float32)


def quant_nf4(X, group):
    """Per-row-group absmax NF4 (16-level non-uniform codebook) quantization."""
    d_out, d_in = X.shape
    if d_in % group != 0:
        raise ValueError(f"d_in={d_in} not divisible by group={group}")
    Xg = X.reshape(d_out, d_in // group, group)
    absmax = np.maximum(np.max(np.abs(Xg), axis=-1, keepdims=True), EPS)
    norm = (Xg / absmax).astype(np.float32)
    flat = norm.ravel()
    idx = np.clip(np.searchsorted(NF4_CODES, flat), 1, len(NF4_CODES) - 1)
    left, right = NF4_CODES[idx - 1], NF4_CODES[idx]
    code = np.where((flat - left) > (right - flat), right, left).astype(np.float32)
    return (code.reshape(Xg.shape) * absmax).reshape(d_out, d_in).astype(np.float32)


def make_quantizer(rung):
    if rung["kind"] == "uniform":
        bits, group = rung["bits"], rung["group"]
        return lambda X: quant_uniform(X, bits, group)
    if rung["kind"] == "nf4":
        group = rung["group"]
        return lambda X: quant_nf4(X, group)
    raise ValueError(f"unknown rung kind {rung['kind']!r}")


REQUIRED_FIELDS = ["K", "A", "B", "Wbase", "recon_rel_err", "vectors_valid"]


def validate_npz(path, recon_threshold=1e-3):
    """Hard gate: structural + numerical validity of a --save_vectors npz.

    Returns (report_dict, ok_bool). Raises no exceptions; caller decides how
    to act (main() exits nonzero on failure)."""
    d = np.load(path, allow_pickle=True)
    missing = [k for k in REQUIRED_FIELDS if k not in d.files]
    report = {"path": path, "missing_fields": missing}
    if missing:
        report["ok"] = False
        return report, False

    K, A, B, Wbase = d["K"], d["A"], d["B"], d["Wbase"]
    N, d_in = K.shape
    d_out = A.shape[1]
    shape_problems = []
    if A.shape != (N, d_out):
        shape_problems.append(f"A.shape={A.shape} != (N={N}, d_out={d_out})")
    if B.shape != (N, d_in):
        shape_problems.append(f"B.shape={B.shape} != (N={N}, d_in={d_in})")
    if Wbase.shape != (d_out, d_in):
        shape_problems.append(f"Wbase.shape={Wbase.shape} != (d_out={d_out}, d_in={d_in})")
    recon = np.asarray(d["recon_rel_err"], dtype=float)
    if recon.shape != (N,):
        shape_problems.append(f"recon_rel_err.shape={recon.shape} != (N={N},)")

    report.update({
        "N": int(N), "d_in": int(d_in), "d_out": int(d_out),
        "shape_problems": shape_problems,
        "model": str(d["model"]) if "model" in d.files else None,
        "editor": str(d["editor"]) if "editor" in d.files else None,
        "layer": int(d["layer"]) if "layer" in d.files else None,
        "seed": int(d["seed"]) if "seed" in d.files else None,
    })
    if shape_problems:
        report["ok"] = False
        return report, False

    n_valid = int(np.sum(recon <= recon_threshold))
    max_recon = float(recon.max()) if recon.size else float("nan")
    stored_valid = bool(d["vectors_valid"])
    recomputed_valid = bool(recon.size and max_recon <= recon_threshold)
    report.update({
        "recon_threshold": recon_threshold,
        "n_valid_edits": n_valid,
        "max_recon_rel_err": max_recon,
        "stored_vectors_valid": stored_valid,
        "recomputed_vectors_valid": recomputed_valid,
        "provenance_consistent": stored_valid == recomputed_valid,
    })
    ok = recomputed_valid  # the hard numerical gate; provenance mismatch is reported, not fatal
    report["ok"] = ok
    return report, ok


def margin_survival_projret(A, B, K, Wbase, rung):
    """Per-edit closed-form margin (predictor) + TWO oracles for one bank / rung.
    Returns (margins[N], survivals[N], projrets[N]). A single requantization
    Q(Wbase+ΔW_i) per edit feeds all three (no double cost).

      margins[i]   = median_r( |A_i[r]| * ||B_i||_2 / (rowerr_base[r] + eps) )   (predictor)

      survivals[i] = ||Q(Wbase+ΔW_i) - Q(Wbase)||_F / ||ΔW_i||_F                 (FROBENIUS
                     oracle) — magnitude-slaved: it is a DECREASING function of edit size
                     (noise floor ~sqrt(mag), imprint ~mag), so the margin cannot positively
                     predict it. See the module docstring.

      projrets[i]  = <(Q(Wbase+ΔW_i)-Q(Wbase)) @ k_i, r_i> / <r_i, r_i>          (faithful
                     PROJECTION-RETENTION oracle) — the fraction of the edit's ACTION on its
                     OWN key k that survives requantization. r_i = A_i (= v-Wk, exactly),
                     k_i = K_i. Bounded ~[0,1], inflation-robust. On real ROME banks it is
                     DEGENERATE (mean ~1.002, std ~0.001 — no cross-edit variance), i.e. the
                     S2.7 pre-registered live-null: every edit "survives" equally, so there is
                     nothing for any predictor to rank -> KILL by the no-variance criterion.

    THIS is the K argument's real use (validate_npz only uses K for shape/recon)."""
    q = make_quantizer(rung)
    Wbase = Wbase.astype(np.float32)
    Qbase = q(Wbase)
    rowerr_base = np.linalg.norm(Qbase - Wbase, axis=1)  # [d_out], once per rung

    N = A.shape[0]
    margins = np.empty(N, dtype=np.float64)
    survivals = np.empty(N, dtype=np.float64)
    projrets = np.full(N, np.nan, dtype=np.float64)
    for i in range(N):
        Ai, Bi = A[i].astype(np.float32), B[i].astype(np.float32)
        normA, normB = np.linalg.norm(Ai), np.linalg.norm(Bi)
        margins[i] = float(np.median(np.abs(Ai) * normB / (rowerr_base + EPS)))

        Wnew = Wbase + np.outer(Ai, Bi)
        Qnew = q(Wnew)
        dQ = Qnew - Qbase
        survivals[i] = float(np.linalg.norm(dQ) / max(normA * normB, EPS))

        # projection-retention oracle: project the realized weight-delta's action on the
        # edit's own key k onto the intended residual r=A_i. fp64 reduction for stability.
        r64 = Ai.astype(np.float64)
        rr = float(r64 @ r64)
        if rr > 0.0:
            projrets[i] = float((dQ.astype(np.float64) @ K[i].astype(np.float64)) @ r64 / rr)
    return margins, survivals, projrets


def analyze_bank(path, recon_threshold, rungs=RUNGS):
    report, ok = validate_npz(path, recon_threshold)
    if not ok:
        return None, report

    d = np.load(path, allow_pickle=True)
    K, A, B, Wbase = d["K"], d["A"], d["B"], d["Wbase"]
    bank_tag = {
        "path": path,
        "model": str(d["model"]) if "model" in d.files else None,
        "editor": str(d["editor"]) if "editor" in d.files else None,
        "layer": int(d["layer"]) if "layer" in d.files else None,
        "seed": int(d["seed"]) if "seed" in d.files else None,
        "N": int(K.shape[0]),
    }

    rung_results = []
    projret_stds = []
    for rung in rungs:
        t0 = time.time()
        margins, survivals, projrets = margin_survival_projret(A, B, K, Wbase, rung)
        rho = spearman(margins, survivals)
        rho_pr = spearman(margins, projrets)
        pr = projrets[np.isfinite(projrets)]
        pr_mean = float(np.mean(pr)) if pr.size else float("nan")
        pr_std = float(np.std(pr)) if pr.size else float("nan")
        projret_stds.append(pr_std)
        rung_results.append({
            "rung": rung["name"],
            "rho_margin_survival": None if np.isnan(rho) else round(float(rho), 4),
            "margin_mean": round(float(np.mean(margins)), 4),
            "margin_median": round(float(np.median(margins)), 4),
            "margin_std": round(float(np.std(margins)), 4),
            "survival_mean": round(float(np.mean(survivals)), 4),
            "survival_median": round(float(np.median(survivals)), 4),
            "survival_std": round(float(np.std(survivals)), 4),
            # faithful projection-retention oracle (artifact-backed live-null evidence):
            "projret_mean": None if np.isnan(pr_mean) else round(pr_mean, 4),
            "projret_std": None if np.isnan(pr_std) else round(pr_std, 4),
            "rho_margin_projret": None if np.isnan(rho_pr) else round(float(rho_pr), 4),
            "elapsed_s": round(time.time() - t0, 2),
        })

    # transition rung = the rung with the most cross-edit survival variance
    # (the regime with actual signal to predict, per the pre-registered gate)
    stds = [r["survival_std"] for r in rung_results]
    trans_idx = int(np.argmax(stds))
    transition = rung_results[trans_idx]

    # faithful-oracle live-null diagnostic: if the projection-retention oracle has essentially
    # NO cross-edit variance at EVERY rung, there is nothing for any predictor to rank -> this
    # is the S2.7 pre-registered "no cross-edit variance (std<0.05)" KILL, independent of the
    # (magnitude-slaved) Frobenius oracle's negative rho.
    finite_pr_stds = [s for s in projret_stds if not np.isnan(s)]
    max_pr_std = max(finite_pr_stds) if finite_pr_stds else float("nan")

    return {
        "bank": bank_tag,
        "rungs": rung_results,
        "transition_rung": transition["rung"],
        "transition_survival_std": transition["survival_std"],
        "transition_rho": transition["rho_margin_survival"],
        "projret_max_std_across_rungs": (None if np.isnan(max_pr_std) else round(max_pr_std, 4)),
        "projret_live_null_kill": bool(finite_pr_stds and max_pr_std < 0.05),
    }, report


def gate_verdict(bank_results, gate_rho, gate_var_floor):
    any_variance = any(r["transition_survival_std"] >= gate_var_floor for r in bank_results)
    all_pass_rho = all(
        (r["transition_rho"] is not None and r["transition_rho"] >= gate_rho)
        for r in bank_results
    )
    if not any_variance:
        return "KILL", "no bank has cross-edit survival std >= gate_var_floor at its transition rung (no signal to predict)"
    if not all_pass_rho:
        return "KILL", "at least one bank's transition-rung Spearman(margin, survival) < gate_rho"
    return "PASS", "all banks clear gate_var_floor and gate_rho at their transition rung"


# -------------------------------------------------------------------------------------
# --selftest: CPU-only, no npz/model/GPU. Proves (a) the quantizer math, (b) that on a
# clean strength-swept synthetic bank the tool detects a STRONG monotone margin<->survival
# relationship (reports the actual signed rho — see the (b) note: with this Frobenius-ratio
# oracle it is strong-but-NEGATIVE, the real KILL is partly definitional), plus a direct
# gate_verdict unit test covering all three verdict branches (PASS / low-rho KILL /
# no-variance KILL), and (c) that a no-variance fixture triggers KILL — so a PASS/KILL on
# real banks can be trusted.
# -------------------------------------------------------------------------------------
def _make_synthetic_bank(path, Wbase, A, B, K, seed=0, layer=-1, model="synthetic"):
    """Write a --save_vectors-schema npz from synthetic factors (recon exact -> 0.0)."""
    N = A.shape[0]
    np.savez_compressed(
        path,
        K=K.astype(np.float32), A=A.astype(np.float32), B=B.astype(np.float32),
        Wbase=Wbase.astype(np.float32),
        recon_rel_err=np.zeros(N, dtype=np.float32),
        vectors_valid=np.array(1, dtype=np.int8),
        model=np.array(model, dtype="U64"),
        editor=np.array("synthetic", dtype="U16"),
        dataset=np.array("selftest", dtype="U16"),
        layer=np.array(layer, dtype=np.int64),
        seed=np.array(seed, dtype=np.int64),
        n_edits=np.array(N, dtype=np.int64),
    )


def _selftest(out_dir, ladder, gate_rho=0.80, gate_var_floor=0.05):
    """Return (all_ok: bool, report: dict). Prints progress; asserts via report['all_ok']."""
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(20260711)
    report = {"ladder": [r["name"] for r in ladder],
              "gate_rho": gate_rho, "gate_var_floor": gate_var_floor,
              "quantizer_unit_tests": {}, "fixtures": {}, "all_ok": True}

    def _fail(msg):
        report["all_ok"] = False
        print(f"[selftest] FAIL: {msg}", flush=True)

    # ---------------- (a) quantizer unit tests ----------------
    group = 32
    W = (rng.standard_normal((16, 128)) * 0.1).astype(np.float32)

    # RTN round-trip error must be <= half a grid step (per group), every rung.
    rt = {}
    for bits in (8, 6, 4, 3):
        Wq = quant_uniform(W, bits, group)
        Wg = W.reshape(16, 128 // group, group)
        qmax = 2 ** (bits - 1) - 1
        step = np.maximum(np.max(np.abs(Wg), axis=-1, keepdims=True), EPS) / qmax
        step_full = np.broadcast_to(step, Wg.shape).reshape(16, 128)
        max_ratio = float(np.max(np.abs(W - Wq) / (0.5 * step_full)))
        ok = max_ratio <= 1.0 + 1e-4
        rt[f"int{bits}"] = {"max_err_over_halfstep": round(max_ratio, 6), "ok": ok}
        if not ok:
            _fail(f"RTN int{bits} round-trip error {max_ratio:.4f} > half step")
    report["quantizer_unit_tests"]["rtn_round_trip_halfstep"] = rt

    # Distinct quantization-level count must be non-increasing as bits shrink.
    levelcnt = {}
    for bits in (8, 6, 4, 3):
        Wg = W.reshape(16, 128 // group, group)
        qmax = 2 ** (bits - 1) - 1
        scale = np.maximum(np.max(np.abs(Wg), axis=-1, keepdims=True), EPS) / qmax
        code = np.clip(np.round(Wg / scale), -qmax, qmax)
        levelcnt[f"int{bits}"] = int(np.unique(code).size)
    mono = levelcnt["int8"] >= levelcnt["int6"] >= levelcnt["int4"] >= levelcnt["int3"]
    report["quantizer_unit_tests"]["distinct_level_count"] = {
        **levelcnt, "monotone_nonincreasing": mono}
    if not mono:
        _fail(f"distinct level count not monotone across rungs: {levelcnt}")

    # NF4: exactly 16 codebook levels; absmax element maps to +-absmax (endpoints +-1.0);
    # every dequantized value is (codebook level) * (per-group absmax).
    Wn = rng.standard_normal((4, 64)).astype(np.float32)
    Wnq = quant_nf4(Wn, 32)
    Wg = Wn.reshape(4, 2, 32); Wqg = Wnq.reshape(4, 2, 32)
    absmax = np.max(np.abs(Wg), axis=-1)
    recovered = np.max(np.abs(Wqg), axis=-1)
    len16 = (NF4_CODES.size == 16)
    absmax_ok = bool(np.allclose(recovered, absmax, rtol=1e-4, atol=1e-6))
    # every dequantized value, normalized by its group absmax, must land ON a codebook
    # level. Tolerance-based (float32 divide reintroduces ~1e-7 noise -> exact set
    # membership is too brittle).
    norm_vals = (Wqg / absmax[..., None]).astype(np.float64)
    nearest_dist = np.min(np.abs(norm_vals[..., None] - NF4_CODES.astype(np.float64)), axis=-1)
    on_book = bool(np.max(nearest_dist) < 1e-4)
    report["quantizer_unit_tests"]["nf4"] = {
        "codebook_len": int(NF4_CODES.size), "len_is_16": len16,
        "absmax_reproduced": absmax_ok, "dequant_on_codebook": on_book}
    if not (len16 and absmax_ok and on_book):
        _fail(f"NF4 unit test failed: len16={len16} absmax={absmax_ok} on_book={on_book}")

    # ---------------- (b) strong monotone margin<->oracle relationship (rank corr) --------
    # Clean strength-swept synthetic bank (near-uniform key -> the oracle is well
    # conditioned). We confirm the machinery detects a STRONG monotone relationship at the
    # transition rung and reports the actual signed rho.
    # REAL FINDING (not a bug): with THIS tool's survival oracle,
    #     survival_i = ||Q(Wbase+dW_i) - Q(Wbase)||_F / ||dW_i||_F,
    # the relationship is strong but NEGATIVE. That oracle is a DECREASING function of edit
    # magnitude (as an edit shrinks, the boundary-crossing noise floor ||Q(Wbase+dW)-Q(Wbase)||
    # falls only ~sqrt(magnitude) while the denominator ||dW|| falls ~linearly, so the ratio
    # DIVERGES for small edits and -> 1 for large edits), whereas the margin INCREASES with
    # magnitude. So the gate's SIGNED verdict here is KILL (rho < gate_rho) — the same
    # KILL, for the same structural reason, that the real banks show. The selftest asserts
    # |rho| is high (the machinery correctly finds the monotone relationship and its sign);
    # the PASS branch is proven separately via the gate_verdict unit test below.
    d_out, d_in = 16, 128
    Wbase = (rng.standard_normal((d_out, d_in)) * 0.05).astype(np.float32)
    N = 80
    scales = np.geomspace(2.0, 40.0, N)
    A = np.zeros((N, d_out), np.float32)
    B = np.zeros((N, d_in), np.float32)
    K = np.zeros((N, d_in), np.float32)
    for i in range(N):
        rdir = rng.standard_normal(d_out); rdir /= (np.linalg.norm(rdir) + 1e-12)
        k = (1.0 + 0.02 * rng.standard_normal(d_in)).astype(np.float32)  # near-uniform key
        A[i] = (rdir * scales[i]).astype(np.float32)
        B[i] = (k / float(k @ k)).astype(np.float32)
        K[i] = k
    pth = os.path.join(out_dir, "synthetic_transition.npz")
    _make_synthetic_bank(pth, Wbase, A, B, K, layer=99, model="synthetic_transition")
    res, rep_b = analyze_bank(pth, 1e-3, rungs=ladder)
    assert res is not None, f"(b) synthetic bank unexpectedly failed validate_npz: {rep_b}"
    verdict, reason = gate_verdict([res], gate_rho, gate_var_floor)
    rho_t = res["transition_rho"]; std_t = res["transition_survival_std"]
    abs_rho = abs(rho_t) if rho_t is not None else None
    report["fixtures"]["b_transition"] = {
        "transition_rung": res["transition_rung"], "transition_rho_signed": rho_t,
        "transition_abs_rho": (round(float(abs_rho), 4) if abs_rho is not None else None),
        "transition_survival_std": std_t, "verdict": verdict, "reason": reason,
        "note": "strong NEGATIVE rho is expected & correct (Frobenius-ratio survival "
                "DECREASES with edit magnitude; margin increases) -> signed verdict KILL.",
        "per_rung": res["rungs"]}
    print(f"[selftest] (b) transition rung={res['transition_rung']} "
          f"signed_rho={rho_t} |rho|={abs_rho} survival_std={std_t} -> verdict={verdict}",
          flush=True)
    if not (std_t is not None and std_t >= gate_var_floor):
        _fail(f"(b) synthetic transition has no cross-edit survival variance (std={std_t})")
    if not (abs_rho is not None and abs_rho >= gate_rho):
        _fail(f"(b) margin<->survival |rho| at transition rung={abs_rho} < {gate_rho} "
              f"(expected a strong monotone relationship on clean synthetic data)")

    # -------- gate_verdict branch unit tests (all three outcomes) --------------------------
    # Because the oracle cannot produce a positive rho (see (b)), the PASS branch is proven
    # by exercising gate_verdict directly on constructed per-bank summaries.
    gv = {
        "pass_branch": gate_verdict(
            [{"transition_survival_std": 0.20, "transition_rho": 0.91}],
            gate_rho, gate_var_floor)[0],
        "kill_low_rho_branch": gate_verdict(
            [{"transition_survival_std": 0.20, "transition_rho": 0.42}],
            gate_rho, gate_var_floor)[0],
        "kill_no_variance_branch": gate_verdict(
            [{"transition_survival_std": 0.01, "transition_rho": 0.99}],
            gate_rho, gate_var_floor)[0],
    }
    report["fixtures"]["gate_verdict_unit"] = gv
    print(f"[selftest] gate_verdict branches: {gv}", flush=True)
    if gv["pass_branch"] != "PASS":
        _fail(f"gate_verdict PASS branch returned {gv['pass_branch']!r}, expected PASS")
    if gv["kill_low_rho_branch"] != "KILL":
        _fail(f"gate_verdict low-rho branch returned {gv['kill_low_rho_branch']!r}, expected KILL")
    if gv["kill_no_variance_branch"] != "KILL":
        _fail(f"gate_verdict no-variance branch returned {gv['kill_no_variance_branch']!r}, expected KILL")

    # ---------------- (c) no-variance fixture -> KILL ----------------
    # Identical edits => survival is identical across edits => zero cross-edit variance at
    # every rung => the "no signal to predict" KILL branch must fire. (Mirrors the live null
    # named in S2.7: large-norm edits saturating to ~full survival.)
    N2 = 40
    rdir = rng.standard_normal(d_out); rdir /= (np.linalg.norm(rdir) + 1e-12)
    k0 = rng.standard_normal(d_in).astype(np.float32)
    a0 = (rdir * 25.0).astype(np.float32)
    b0 = (k0 / float(k0 @ k0)).astype(np.float32)
    A2 = np.repeat(a0[None, :], N2, axis=0)
    B2 = np.repeat(b0[None, :], N2, axis=0)
    K2 = np.repeat(k0[None, :], N2, axis=0)
    pth2 = os.path.join(out_dir, "synthetic_novariance.npz")
    _make_synthetic_bank(pth2, Wbase, A2, B2, K2, layer=98, model="synthetic_novariance")
    res2, rep2 = analyze_bank(pth2, 1e-3, rungs=ladder)
    assert res2 is not None, f"(c) synthetic bank unexpectedly failed validate_npz: {rep2}"
    verdict2, reason2 = gate_verdict([res2], gate_rho, gate_var_floor)
    stds2 = {r["rung"]: r["survival_std"] for r in res2["rungs"]}
    report["fixtures"]["c_novariance"] = {
        "survival_std_per_rung": stds2, "verdict": verdict2, "reason": reason2}
    print(f"[selftest] (c) no-variance survival_std={stds2} -> {verdict2}", flush=True)
    if verdict2 != "KILL":
        _fail(f"(c) expected KILL on no-variance fixture, got {verdict2}")
    if not all(v < gate_var_floor for v in stds2.values()):
        _fail(f"(c) no-variance fixture has a rung with std>=floor: {stds2}")

    report["verdict"] = "SELFTEST-PASS" if report["all_ok"] else "SELFTEST-FAIL"
    out_json = os.path.join(out_dir, "SELFTEST_report.json")
    tmp = out_json + ".tmp"
    with open(tmp, "w") as f:
        json.dump(report, f, indent=2)
    os.replace(tmp, out_json)
    print(f"[selftest] {report['verdict']} -> {out_json}", flush=True)
    return report["all_ok"], report


def main():
    ap = argparse.ArgumentParser(
        description="QuantEdit E0: closed-form margin vs simulated-quant survival oracle (CPU-only)."
    )
    ap.add_argument("--validate_npz", nargs="+", default=None,
                     help="hard-gate mode: validate 1+ --save_vectors npz files and exit "
                          "(nonzero on any structural or numerical failure)")
    ap.add_argument("--npz", nargs="+", default=None,
                     help="1+ --save_vectors npz files to run the E0 science on")
    ap.add_argument("--recon_threshold", type=float, default=1e-3,
                     help="max allowed recon_rel_err per edit (matches killgate's own "
                          "vectors_valid threshold)")
    ap.add_argument("--gate_rho", type=float, default=0.80,
                     help="pre-registered kill-gate bar for Spearman(margin, survival) "
                          "at the transition rung (EXPANSION-DIRECTIONS-DEEP-2026-07-01.md:164)")
    ap.add_argument("--gate_var_floor", type=float, default=0.05,
                     help="min cross-edit std(survival) required at some rung, else KILL "
                          "(no signal to predict)")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results", "QUANTEDIT_E0.json"))
    ap.add_argument("--from_vectors", nargs="+", default=None,
                     help="alias for --npz: 1+ --save_vectors banks to score (both are merged)")
    ap.add_argument("--ladder", choices=sorted(LADDERS.keys()), default="spec",
                     help="rung ladder. 'spec' (default) = int8/int6/int4/int3 g32 + NF4 g32 "
                          "(the S2.7 pre-registration). 'legacy' = the first-draft int8_g64/"
                          "int4_g64/int3_g32/nf4_g64 ladder (reproduces the original "
                          "results/QUANTEDIT_E0.json).")
    ap.add_argument("--selftest", action="store_true",
                     help="CPU-only self-test: quantizer unit tests + synthetic survival "
                          "transition (margin<->oracle rank corr) + no-variance KILL fixture. "
                          "No npz/model/GPU. Exits nonzero on any failure.")
    ap.add_argument("--selftest_out",
                     default=os.path.join(HERE, "..", "results", "quantedit", "selftest"),
                     help="directory for --selftest fixtures + SELFTEST_report.json")
    args = ap.parse_args()

    rungs = LADDERS[args.ladder]

    if args.selftest:
        ok, _ = _selftest(args.selftest_out, rungs, args.gate_rho, args.gate_var_floor)
        sys.exit(0 if ok else 1)

    # --from_vectors is a spelling of --npz; merge both (order-preserving, de-duplicated).
    npz_inputs = list(args.npz or []) + list(args.from_vectors or [])
    seen = set()
    npz_inputs = [p for p in npz_inputs if not (p in seen or seen.add(p))]

    if args.validate_npz:
        all_ok = True
        reports = []
        for p in args.validate_npz:
            report, ok = validate_npz(p, args.recon_threshold)
            reports.append(report)
            status = "OK" if ok else "FAIL"
            print(f"[quantedit_e0] --validate_npz {status} {p}")
            print(json.dumps(report, indent=2))
            if ok:
                print(f"[quantedit_e0]   -> {report['n_valid_edits']}/{report['N']} valid edits")
            all_ok = all_ok and ok
        sys.exit(0 if all_ok else 1)

    if not npz_inputs:
        raise SystemExit("[quantedit_e0] pass --selftest, --validate_npz PATH..., "
                         "or --npz/--from_vectors PATH...")

    bank_results = []
    notes = []
    for p in npz_inputs:
        res, report = analyze_bank(p, args.recon_threshold, rungs)
        if res is None:
            notes.append(f"{p}: FAILED validate_npz hard gate — skipped ({report})")
            print(f"[quantedit_e0] ABORT bank {p}: fails --validate_npz hard gate", flush=True)
            continue
        bank_results.append(res)
        print(f"[quantedit_e0] bank {p}: transition_rung={res['transition_rung']} "
              f"rho={res['transition_rho']} (survival_std={res['transition_survival_std']})",
              flush=True)

    if not bank_results:
        raise SystemExit("[quantedit_e0] no bank passed --validate_npz; nothing to score")

    verdict, verdict_reason = gate_verdict(bank_results, args.gate_rho, args.gate_var_floor)

    # faithful-oracle (projection-retention) cross-bank summary: artifact-backed evidence that
    # the KILL ALSO holds under the bounded, inflation-robust oracle — a live null (no cross-edit
    # variance -> nothing for any predictor to rank). Independent of the Frobenius verdict.
    projret_live_null_all = bool(bank_results) and all(
        b.get("projret_live_null_kill") for b in bank_results)
    _pr_stds = [b["projret_max_std_across_rungs"] for b in bank_results
                if b.get("projret_max_std_across_rungs") is not None]
    projret_max_std_overall = max(_pr_stds) if _pr_stds else float("nan")

    out = {
        "experiment": "QuantEdit E0 (rank #7, EXPANSION-DIRECTIONS-DEEP-2026-07-01.md S2.7)",
        "hypothesis": "closed-form row-effective margin (from K,A,B,Wbase alone) predicts "
                       "per-edit survival under simulated 4-bit-class PTQ",
        "margin_definition": "median over d_out rows of |A_i[r]| * ||B_i||_2 / "
                              "(||Q(Wbase)[r,:]-Wbase[r,:]||_2 + eps); Q = the rung's quantizer "
                              "applied to Wbase once (shared across edits in the bank)",
        "survival_definition": "||Q(Wbase+outer(A_i,B_i)) - Q(Wbase)||_F / (||A_i||_2*||B_i||_2); "
                                "Q(Wbase+ΔW_i) is requantized per edit (post-edit absmax) — assumes "
                                "PTQ happens AFTER editing. NOTE: magnitude-slaved (DECREASING in "
                                "edit size) so the margin cannot positively predict it — the KILL "
                                "here is partly definitional; see projret_definition.",
        "projret_definition": "<(Q(Wbase+ΔW_i)-Q(Wbase)) @ k_i, r_i> / <r_i,r_i>, with r_i=A_i=v-Wk "
                               "and k_i=K_i — the faithful projection-RETENTION oracle: the fraction "
                               "of the edit's action on its OWN key that survives requantization "
                               "(bounded ~[0,1], inflation-robust). On real ROME banks this is a "
                               "LIVE NULL (mean ~1.002, std ~0.001, no cross-edit variance) -> "
                               "nothing to predict -> the S2.7 pre-registered no-variance KILL. "
                               "Emitted per rung as projret_mean/projret_std/rho_margin_projret.",
        "nf4_definition": "bitsandbytes/QLoRA 16-level NF4 codebook (Dettmers et al. 2023), "
                           "per-row-group absmax normalization, nearest-codebook dequant; "
                           "NO double-quantization of scale constants (out of scope)",
        "uniform_rtn_definition": "symmetric per-row-group absmax RTN: scale=max(|group|)/qmax, "
                                   "qmax=2**(bits-1)-1, code=clip(round(x/scale),-qmax,qmax), "
                                   "dequant=code*scale",
        "ladder": args.ladder,
        "rungs": [r["name"] for r in rungs],
        "group_sizes": {r["name"]: r["group"] for r in rungs},
        "gate": {
            "gate_rho": args.gate_rho,
            "gate_var_floor": args.gate_var_floor,
            "rule": "KILL if no bank clears gate_var_floor at its transition rung (no signal), "
                    "OR any bank's transition-rung Spearman(margin,survival) < gate_rho",
            "source": "EXPANSION-DIRECTIONS-DEEP-2026-07-01.md:164 (QuantEdit E0 pre-registration)",
        },
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "faithful_oracle_live_null": {
            "projret_live_null_kill_all_banks": projret_live_null_all,
            "projret_max_std_over_all_banks_rungs": (
                None if np.isnan(projret_max_std_overall)
                else round(float(projret_max_std_overall), 4)),
            "meaning": "projection-retention oracle has ~no cross-edit variance at every rung of "
                       "every bank (< gate_var_floor) -> S2.7 pre-registered live-null KILL, "
                       "corroborating the Frobenius verdict with a bounded, non-magnitude-slaved "
                       "oracle. If True, the KILL does NOT rest on the definitional Frobenius rho.",
        },
        "banks": bank_results,
        "notes": notes + [
            "SIMULATED quant survival oracle — hand-rolled numpy RTN/NF4, not a real "
            "quantized-model behavioral eval (that is E1, ~140 GPU-min, deferred pending this gate).",
            "Single seed per layer (s0) — no cross-seed variance yet; that is E2 in the ladder.",
            "Group sizes (64 uniform / 32 for int3) are a documented choice, not swept.",
            "NF4 scale constants are NOT double-quantized (bnb does this for extra compression; "
            "skipped here as out of scope for the closed-form margin claim).",
            "Margin uses a row-median aggregate (not mean/min/quantile) — an unexplored "
            "aggregation-sensitivity axis.",
        ],
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=2)
    os.replace(tmp, args.out)
    print("\n=== QUANTEDIT E0 RESULT ===", flush=True)
    print(json.dumps({k: out[k] for k in ("verdict", "verdict_reason")}, indent=2), flush=True)
    for b in bank_results:
        print(f"  bank L{b['bank']['layer']} s{b['bank']['seed']}: "
              f"transition_rung={b['transition_rung']} rho={b['transition_rho']} "
              f"survival_std={b['transition_survival_std']}", flush=True)
    print(f"[quantedit_e0] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
