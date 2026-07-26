"""halflife_hl0.py — "Edit half-life" HL0 kill-gate (Edit half-life direction, spec 2.2).

CLAIM (a-priori, per-EDIT). The erasure of a single ROME/AlphaEdit edit under a
subsequent gsm8k adaptation is predicted BEFORE the adaptation is run by the scalar

        P_i = || tau_L . k_edit_i ||          (task vector tau computed ONCE, fp32)

where tau_L is the edited-layer down_proj slice of the gsm8k task vector and k_edit_i
is edit i's base-model subject key. First-principles reason it should work: ROME writes
DW_i = (v_i - W k_i) k_i^T / (k_i.k_i), so (W + DW_i) k_i = v_i (the edit installs value
v_i at key k_i). A task-vector adaptation W <- W + alpha*tau_L shifts THAT output by
exactly alpha*(tau_L k_i); the edit's margin therefore decays with || tau_L k_i ||. The
gate asks whether this a-priori scalar survives partialling of ||k_edit||, ||DW||,
S=||v-Wk||, target-token frequency and pre-edit p(target), and whether it beats the
arXiv 2511.05852 frequency-factor baseline head-to-head.

WHY PER-EDIT (not per-probe). B6's killgate_keygeom / analyze_matrices statistic is
WITHIN-PROBE (one rho per probe column, correlating key-cosine vs damage DOWN edits).
Here the unit of analysis is the EDIT: each edit has ONE scalar erasure (its own
target-margin drop under adaptation) and ONE predictor P_i. So HL0 correlates two
length-N per-edit vectors ACROSS edits — the exchangeable unit is the edit, and the
permutation nulls permute edits (mirrors analyze_sequential.perm_null_spearman, which
is likewise a per-edit vector null, NOT the within-probe column null).

STAGES (each --stage sN is independently invokable and idempotent — skips if its output
already exists on disk; every GPU stage CONFIG-skips cleanly with a clear message if the
model or gsm8k cache is absent, and NEVER downloads — HF_HUB_OFFLINE standing policy):

  s1  (GPU)  compute the task vector tau ONCE per arm from a short gsm8k fine-tune:
             * MLP-only arm  -> FT unfreezes ONLY the edited-layer down_proj.weight
             * full-model arm-> FT unfreezes ALL params (2511.05852: non-edited-layer
               FT erases MORE). Saves results/halflife/tau/tau_{arm}.npz (tau_L slice,
               always) + tau_full_state.pt (full arm only, the all-param delta applied
               in s3).
  s2  (GPU)  200 L12 s0 ROME edits + 100 unedited control probes. Per edit records the
             base key k_edit, ||DW|| (norm_growth), S (resid_norm), the exact rank-one
             factors (residual_vec, k, denom) so s3 can REINSTALL DW cheaply without
             re-optimising, the post-edit margin, pre-edit p(target) and target-token
             frequency. Saves results/halflife/hl0_s2_edits.npz.
  s3  (GPU)  the cheap PROXY. For each arm x alpha in {0.5,1,2}: apply W += alpha*tau
             (edited layer only for MLP; ALL params for full via tau_full_state.pt),
             reinstall each edit's rank-one DW on top, FORWARD, read the target margin;
             erasure_i = margin_edited_i - margin_proxy_i (a MARGIN DROP, not binary).
             Also measures the 100 controls' margin drift. Saves hl0_s3_proxy.npz.
             NB (spec deviation, see module note at bottom): s3 needs a real model
             FORWARD per edit — a closed-form margin from alpha*tau_L k would be
             CIRCULAR with the predictor — so s3 runs on --device, not pure-CPU.
  s4  (GPU)  proxy-fidelity ground truth. For n=40 sampled edits, install the edit then
             run a REAL short gsm8k FT (same machinery as s1) and read that edit's
             margin drop under genuine adaptation. Saves hl0_s4_realft.npz.
  s5  (CPU)  the analysis + verdict. Reads s1/s2/s3(/s4), computes P_i per arm, the raw
             and partialled per-edit Spearman ladder (single + joint confounds, with
             Freedman-Lane residual permutation nulls), the frequency-baseline head-to-
             head, and the s3-vs-s4 proxy fidelity with a bootstrap CI. Writes the
             explicit per-criterion verdict block -> results/halflife/HL0_killgate_table.json.

KILL CRITERIA (all computed + printed in the verdict block; MLP arm is primary, headline
alpha = 1.0 = the natural task-vector scale theta_base+tau=theta_ft):
  * erasure variance nil (nothing to predict)                                    -> KILL
  * MLP-arm joint-partialled rho < 0.2 (SIGN-AWARE: rho>0, not |rho|)            -> KILL
  * LOSES to the frequency baseline head-to-head (|freq rho| >= |P rho|)         -> KILL
  * proxy-fidelity bootstrap-CI lower bound < 0.3                                -> KILL
  * full-model arm rho < 0.15 WHILE the MLP arm passes AND was EVALUATED    -> SCOPE-DOWN
    (report MLP-only; NOT a kill). Full arm ABSENT => UNEVALUATED, never scope-down.

TWO CIRCULARITY / HONESTY GUARDS baked into the verdict (hostile-review fixes):
  * FREQUENCY baseline is a within-slice unigram APPROXIMATION (few-hundred-sample counts),
    NOT the pretraining frequency factor of 2511.05852. So "beats frequency" is reported as
    APPROXIMATE-BASELINE-ONLY and is NOT a paper-claimable result; only an actual LOSS kills.
    Flip target_freq_provenance to 'real_corpus_unigram' only with a real corpus table.
  * The s3 PROXY erasure is mechanically alpha*P_i to machine precision, so the arm ladder's
    rho(P, proxy-erasure) is partly TAUTOLOGICAL. The least-circular evidence is
    `nontautological_predictor_vs_realft` = Spearman(P, REAL gsm8k-FT erasure) from s4 (n=40),
    surfaced in the verdict; s4 carries the non-tautological weight, not the s3 proxy.

SELF-TEST (CPU, synthetic, no torch/model/dataset):
  python experiments/halflife_hl0.py --selftest
  Fixtures: POSITIVE (erasure ~ P) -> PASS; KILL (erasure = noise) -> KILL; PLANTED-CONFOUND
  (erasure = f(||k||), P correlated with ||k|| but conditionally independent) -> partial
  COLLAPSES the confound; MECHANICAL (proxy == alpha*P exactly) -> arm rho tautologically ~1
  while the clean P-vs-realFT sits below it; plus M2 (full arm absent -> UNEVALUATED, not
  scope-down) and M3 (freq head-to-head labeled approximate, not paper-claimable). Artifacts
  are quarantined under results/halflife/selftest/.

Statistics mirror the B6 codebase EXACTLY: tie-averaged (midrank) Spearman
(analyze_matrices.spearman), rank-space partial correlation (analyze_sequential.
partial_spearman generalised to >1 confound here), Freedman-Lane residual permutation
null (analyze_sequential.perm_null_partial), RNG_SEED=12345. Signed Spearman only —
AUROC is banned in this codebase. CPU/numpy for s5; torch is imported LAZILY inside the
GPU stages only, so s5 and --selftest never touch CUDA.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

HARNESS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HARNESS)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# ------------------------------------------------------------------ stat primitives
# Import the canonical B6 conventions; fall back to byte-equivalent replicas so this
# file is self-contained if run from an odd cwd (mirrors mechanism_sc_table.py's pattern).
try:
    from analyze_matrices import _midrank, spearman, RNG_SEED  # noqa: E402
except Exception:  # pragma: no cover - fallback replica
    RNG_SEED = 12345

    def _midrank(x):
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


DATA_DEFAULT = os.path.join(HARNESS, "data", "counterfact.json")
MODEL_DEFAULT = os.path.join(HARNESS, "data", "models", "Llama-3.2-1B")
OUTDIR = os.path.join(HARNESS, "results", "halflife")
TAUDIR = os.path.join(OUTDIR, "tau")

ARMS = ("mlp", "full")
ALPHAS = (0.5, 1.0, 2.0)
ALPHA_HEADLINE = 1.0            # pre-registered gate alpha (theta_base + tau = theta_ft)
CONFOUND_NAMES = ("knorm", "norm_growth", "S", "target_freq", "pre_edit_ptarget")


# =================================================================== per-edit stats
def _residualize_multi(y_rank, z_ranks):
    """Least-squares residual of y_rank on [1, z_ranks...] in rank space.

    Generalises analyze_sequential._residualize (single z) to a design matrix of
    >=1 confound rank vectors — the standard partial-correlation construction."""
    cols = [np.ones_like(y_rank)] + [np.asarray(z, float) for z in z_ranks]
    Z = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(Z, y_rank, rcond=None)
    return y_rank - Z @ beta


def partial_spearman_multi(x, y, zlist):
    """Signed partial Spearman rho(x, y | z1..zk): correlate the rank-space residuals
    of x and y after regressing each on ALL confounds jointly. Returns
    (rho, rx_res, ry_res) so the caller can reuse the residuals for a Freedman-Lane
    permutation null (exactly as analyze_sequential.partial_spearman does for one z).

    Per-EDIT: x, y, each z is a length-N per-edit vector; N = #edits."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    zlist = [np.asarray(z, float) for z in zlist]
    m = np.isfinite(x) & np.isfinite(y)
    for z in zlist:
        m = m & np.isfinite(z)
    x, y = x[m], y[m]
    zlist = [z[m] for z in zlist]
    if x.size < 4 + len(zlist):
        return np.nan, None, None
    rx, ry = _midrank(x), _midrank(y)
    rz = [_midrank(z) for z in zlist]
    if rx.std() == 0 or ry.std() == 0 or any(r.std() == 0 for r in rz):
        return np.nan, None, None
    rx_res = _residualize_multi(rx, rz)
    ry_res = _residualize_multi(ry, rz)
    if rx_res.std() == 0 or ry_res.std() == 0:
        return np.nan, None, None
    return float(np.corrcoef(rx_res, ry_res)[0, 1]), rx_res, ry_res


def perm_null_spearman(x, y, obs, n_perm=2000, seed=RNG_SEED):
    """Edit-level null for a raw per-edit Spearman: permute y across edits (the edit is
    the exchangeable unit) and recompute. Returns (p, null_mean, null_std). Mirrors
    analyze_sequential.perm_null_spearman."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 4 or not np.isfinite(obs):
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    nulls = np.empty(n_perm); ge = 0
    for t in range(n_perm):
        nulls[t] = spearman(x, rng.permutation(y))
        if abs(nulls[t]) >= abs(obs):
            ge += 1
    return (ge + 1) / (n_perm + 1), float(np.nanmean(nulls)), float(np.nanstd(nulls))


def perm_null_partial(rx_res, ry_res, obs, n_perm=2000, seed=RNG_SEED):
    """Freedman-Lane residual permutation null for a partial correlation: with every
    confound residualised out of both sides, ry_res is exchangeable under
    H0 (x _||_ y | Z). Permute ry_res, recompute Pearson corr vs fixed rx_res.
    Mirrors analyze_sequential.perm_null_partial."""
    if rx_res is None or ry_res is None or not np.isfinite(obs):
        return np.nan
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(n_perm):
        nu = float(np.corrcoef(rx_res, rng.permutation(ry_res))[0, 1])
        if abs(nu) >= abs(obs):
            ge += 1
    return (ge + 1) / (n_perm + 1)


def boot_ci_spearman(x, y, n_boot=2000, seed=RNG_SEED, lo=2.5, hi=97.5):
    """Bootstrap CI of Spearman(x, y) by resampling the paired EDITS with replacement
    (the correct cluster unit). Returns (rho_point, ci_lo, ci_hi)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    point = spearman(x, y)
    if x.size < 5:
        return point, np.nan, np.nan
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for t in range(n_boot):
        idx = rng.integers(0, x.size, x.size)
        vals[t] = spearman(x[idx], y[idx])
    vals = vals[np.isfinite(vals)]
    if vals.size < 3:
        return point, np.nan, np.nan
    return point, float(np.percentile(vals, lo)), float(np.percentile(vals, hi))


def _r4(v):
    if v is None:
        return None
    v = float(v)
    return None if not np.isfinite(v) else round(v, 4)


# =================================================================== s5 core analysis
def _confound_ladder(P, erasure, confounds, n_perm, seed):
    """Full partialling ladder for one (arm, alpha) view. `confounds` is an ordered dict
    name->[N]. Returns raw rho (+ null), each SINGLE-confound partial (+ FL null), and
    the JOINT partial over all confounds (+ FL null) — the joint is the GATE statistic."""
    raw = spearman(P, erasure)
    raw_p, raw_nm, raw_ns = perm_null_spearman(P, erasure, raw, n_perm, seed)
    singles = {}
    for name, z in confounds.items():
        rho, rxr, ryr = partial_spearman_multi(P, erasure, [z])
        singles[name] = {"partial_rho": _r4(rho),
                         "perm_p": _r4(perm_null_partial(rxr, ryr, rho, n_perm, seed))}
    zlist = [confounds[k] for k in confounds]
    jrho, jrxr, jryr = partial_spearman_multi(P, erasure, zlist)
    joint_p = perm_null_partial(jrxr, jryr, jrho, n_perm, seed)
    return {
        "raw_rho": _r4(raw), "raw_perm_p": _r4(raw_p),
        "raw_null_mean": _r4(raw_nm), "raw_null_std": _r4(raw_ns),
        "single_partials": singles,
        "joint_partial_rho": _r4(jrho), "joint_partial_perm_p": _r4(joint_p),
        "n_edits_used": int((np.isfinite(P) & np.isfinite(erasure)).sum()),
    }


def analyze_hl0(predictors, erasures, confounds, control_drift=None, proxy_real=None,
                alphas=ALPHAS, alpha_headline=ALPHA_HEADLINE, arms=ARMS,
                n_perm=2000, seed=RNG_SEED, provenance=None,
                freq_is_approximate=True):
    """PURE analysis (no torch/IO) shared by the s5 file-path and the self-test.

    predictors:   dict arm -> [N]           P_i = ||tau_L k_i||
    erasures:     dict arm -> {alpha: [N]}  per-edit margin drop under the proxy
    confounds:    dict name -> [N]          ||k||, ||DW||, S, target_freq, pre_edit_ptarget
    control_drift:dict arm -> {alpha: [n_ctrl]}  (optional, reporting-only specificity)
    proxy_real:   optional dict {arm, alpha, proxy_erasure[n], real_erasure[n], edit_idx[n]}

    Returns the full result dict incl. the explicit per-criterion verdict block."""
    arms = [a for a in arms if a in predictors]
    result = {
        "unit_of_analysis": "per-EDIT (one scalar erasure + one predictor P_i per edit; "
                            "N=#edits). Contrast: killgate_keygeom is per-PROBE.",
        "predictor": "P_i = ||tau_L . k_edit_i|| (task vector tau computed once, fp32)",
        "erasure": "margin drop = margin_edited - margin_proxy (logit(target_new)-"
                   "logit(target_true) at the edit prompt); NOT binary.",
        "statistic": "signed midrank Spearman; rank-space partial correlation; "
                     "Freedman-Lane residual permutation null; RNG_SEED=%d." % seed,
        "confound_order": list(confounds.keys()),
        "alphas": list(alphas), "alpha_headline": alpha_headline,
        "n_perm": n_perm,
        "arms": {},
    }
    if provenance:
        result["provenance"] = provenance

    # erasure variance receipt (kill criterion 1) + the full ladder per arm x alpha
    min_erasure_std = np.inf
    for arm in arms:
        P = np.asarray(predictors[arm], float)
        arm_block = {"n_edits": int(P.size), "by_alpha": {}}
        for a in alphas:
            er = np.asarray(erasures[arm][a], float)
            std = float(np.nanstd(er))
            min_erasure_std = min(min_erasure_std, std)
            block = _confound_ladder(P, er, confounds, n_perm, seed)
            block["erasure_std"] = _r4(std)
            block["erasure_mean"] = _r4(float(np.nanmean(er)))
            if control_drift is not None and arm in control_drift and a in control_drift[arm]:
                cd = np.asarray(control_drift[arm][a], float)
                block["control_drift_mean"] = _r4(float(np.nanmean(cd)))
                block["control_drift_p90"] = _r4(float(np.nanpercentile(cd, 90)))
                # specificity: edit erasure should exceed unrelated-control drift
                block["edit_erasure_over_control_mean"] = _r4(
                    float(np.nanmean(er)) / (abs(float(np.nanmean(cd))) + 1e-9))
            arm_block["by_alpha"][str(a)] = block
        result["arms"][arm] = arm_block

    # frequency baseline head-to-head (kill criterion 3), MLP arm at headline alpha
    freq = confounds.get("target_freq")
    prim_arm = "mlp" if "mlp" in arms else arms[0]
    P_prim = np.asarray(predictors[prim_arm], float)
    er_prim = np.asarray(erasures[prim_arm][alpha_headline], float)
    freq_rho = spearman(freq, er_prim) if freq is not None else np.nan
    tauk_rho = spearman(P_prim, er_prim)
    tauk_ctrl_freq, rxr_f, ryr_f = (partial_spearman_multi(P_prim, er_prim, [freq])
                                    if freq is not None else (np.nan, None, None))
    result["frequency_head_to_head"] = {
        "arm": prim_arm, "alpha": alpha_headline,
        "freq_baseline_rho": _r4(freq_rho),
        "tauk_raw_rho": _r4(tauk_rho),
        "tauk_partialled_by_freq_rho": _r4(tauk_ctrl_freq),
        "tauk_partialled_by_freq_perm_p": _r4(
            perm_null_partial(rxr_f, ryr_f, tauk_ctrl_freq, n_perm, seed)),
        "tauk_beats_freq": (bool(abs(tauk_rho) > abs(freq_rho))
                            if (np.isfinite(tauk_rho) and np.isfinite(freq_rho)) else None),
        # M3: the baseline is a degenerate within-slice unigram (~few-hundred-sample counts),
        # NOT the pretraining frequency factor of 2511.05852, unless a real large-corpus
        # unigram table was supplied. A spurious "beats" of a toothless baseline is near-free,
        # so this head-to-head is reported as APPROXIMATE-BASELINE-ONLY and CANNOT support a
        # 'beats frequency' paper claim; only an actual LOSS to it is treated as a KILL signal.
        "baseline_is_approximate": bool(freq_is_approximate),
        "baseline_provenance": ("within_slice_unigram_approx" if freq_is_approximate
                                else "real_corpus_unigram"),
        "beats_freq_is_paper_claimable": bool(
            (not freq_is_approximate)
            and (np.isfinite(tauk_rho) and np.isfinite(freq_rho) and abs(tauk_rho) > abs(freq_rho))),
    }

    # proxy fidelity (kill criterion 4): s3 proxy erasure vs s4 REAL-FT erasure
    fid = None
    if proxy_real is not None:
        pr = np.asarray(proxy_real["proxy_erasure"], float)
        rl = np.asarray(proxy_real["real_erasure"], float)
        pt, lo, hi = boot_ci_spearman(pr, rl, seed=seed)
        fid = {
            "arm": proxy_real.get("arm"), "alpha": proxy_real.get("alpha"),
            "n_real": int(min(pr.size, rl.size)),
            "spearman_proxy_vs_real": _r4(pt),
            "boot_ci95": [_r4(lo), _r4(hi)],
            "ci_lower": _r4(lo),
            "note": ("s3 proxy is MECHANICALLY loaded toward P (applied perturbation norm "
                     "== alpha*P_i to machine precision); high fidelity here is necessary but "
                     "partly tautological — see nontautological_predictor_vs_realft for the "
                     "clean test."),
        }
    result["proxy_fidelity"] = fid

    # M4: the LEAST-CIRCULAR statistic — a-priori predictor P vs REAL gsm8k-FT erasure (s4),
    # free (data already loaded). Because the s3 proxy's perturbation norm IS alpha*P_i, only
    # this P-vs-real-FT correlation carries non-tautological weight. Surfaced in the verdict.
    nontaut = None
    if proxy_real is not None:
        arm_pr = proxy_real.get("arm", prim_arm)
        eidx = np.asarray(proxy_real["edit_idx"], int)
        P_s4 = np.asarray(predictors[arm_pr], float)[eidx]
        real = np.asarray(proxy_real["real_erasure"], float)
        pt2, lo2, hi2 = boot_ci_spearman(P_s4, real, seed=seed)
        ppv, _, _ = perm_null_spearman(P_s4, real, pt2, n_perm, seed)
        nontaut = {
            "arm": arm_pr, "n_real": int(min(P_s4.size, real.size)),
            "spearman_P_vs_realFT": _r4(pt2),
            "boot_ci95": [_r4(lo2), _r4(hi2)], "ci_lower": _r4(lo2),
            "perm_p": _r4(ppv),
            "note": ("LEAST-CIRCULAR evidence: a-priori ||tau_L k|| vs a REAL gradient-descent "
                     "gsm8k FT erasure. Unlike the s3 proxy this is not mechanically P-loaded; "
                     "it (not proxy_fidelity) carries the non-tautological weight. n=40 is "
                     "small — read the CI, not the point estimate."),
        }
    result["nontautological_predictor_vs_realft"] = nontaut

    # ---------------------------------------------------------------- verdict block
    headline = str(alpha_headline)
    mlp_joint = result["arms"].get(prim_arm, {}).get("by_alpha", {}).get(
        headline, {}).get("joint_partial_rho")
    full_joint = None
    if "full" in result["arms"]:
        full_joint = result["arms"]["full"]["by_alpha"].get(headline, {}).get("joint_partial_rho")

    c_var_nil = bool(np.isfinite(min_erasure_std) and min_erasure_std < 1e-6)
    # sign-aware (minor): erasure must INCREASE with the predictor -> require rho > 0, not
    # |rho|. A strong NEGATIVE partial is not a pass (it would mean more tau-reach -> LESS
    # erasure, contradicting the mechanism).
    c_mlp_weak = bool(mlp_joint is None or not np.isfinite(mlp_joint) or mlp_joint < 0.2)
    mlp_passes = bool(mlp_joint is not None and np.isfinite(mlp_joint) and mlp_joint >= 0.2)

    # M3: frequency baseline is (by default) an approximate within-slice unigram. LOSING to
    # even a weak baseline is a red flag (KILL); BEATING it cannot license a paper claim.
    beats = result["frequency_head_to_head"]["tauk_beats_freq"]
    c_lose_freq = bool(beats is False)
    freq_status = ("APPROXIMATE-BASELINE-ONLY — within-slice unigram proxy; a real "
                   "pretraining-frequency head-to-head needs a large-corpus unigram table "
                   "(absent). 'beats frequency' is NOT paper-claimable here; only a LOSS kills."
                   if freq_is_approximate else "real-corpus baseline")

    c_fid_low = bool(fid is not None and fid["ci_lower"] is not None
                     and np.isfinite(fid["ci_lower"]) and fid["ci_lower"] < 0.3)

    # M2: SCOPE-DOWN vs UNEVALUATED must not be conflated. SCOPE-DOWN requires the full arm
    # was actually EVALUATED (finite joint rho) and measured weak. If the full arm is absent
    # (tau_full/s3-full missing), it is UNEVALUATED -> the full-arm criterion is INCOMPLETE,
    # never silently a scope-down. Both use the sign-aware < 0.15 threshold.
    full_evaluated = bool(full_joint is not None and np.isfinite(full_joint))
    # inline `full_joint is not None` (not just full_evaluated) so the `< 0.15` is guarded for
    # BOTH runtime and the static type-checker, which cannot narrow None across full_evaluated.
    c_scope_down = bool(mlp_passes and full_evaluated
                        and full_joint is not None and full_joint < 0.15)
    full_unevaluated = bool(mlp_passes and not full_evaluated)
    # DESCRIPTIVE-ONLY: the full arm was measured but the MLP arm did NOT pass (the gate already
    # KILLed on the MLP arm). A full-arm measurement cannot rescue an MLP kill, so it is reported
    # for completeness only and does not enter the verdict decision.
    full_descriptive_only = bool(full_evaluated and not mlp_passes)

    criteria = {
        "erasure_variance_nil": {
            "kill": c_var_nil, "min_erasure_std": _r4(min_erasure_std)},
        "mlp_joint_partialled_rho_lt_0.2": {
            "kill": c_mlp_weak, "mlp_joint_partial_rho": mlp_joint, "arm": prim_arm,
            "alpha": alpha_headline, "sign_aware": "require rho>0 (not |rho|)"},
        "loses_to_frequency_baseline": {
            "kill": c_lose_freq, "tauk_raw_rho": _r4(tauk_rho),
            "freq_baseline_rho": _r4(freq_rho),
            "baseline_is_approximate": bool(freq_is_approximate),
            "status": freq_status},
        "proxy_fidelity_ci_lower_lt_0.3": {
            "kill": c_fid_low,
            "ci_lower": (fid["ci_lower"] if fid else None),
            "note": ("s4 not provided — criterion UNEVALUATED (not a pass)"
                     if fid is None else "s3-vs-s4; partly tautological, see "
                     "nontautological_predictor_vs_realft")},
        "full_arm": {
            "scope_down_kill": False, "scope_down": c_scope_down,
            "evaluated": full_evaluated, "unevaluated": full_unevaluated,
            "descriptive_only": full_descriptive_only,
            "full_joint_partial_rho": full_joint, "mlp_passes": bool(mlp_passes),
            "note": ("SCOPE-DOWN (not a kill) requires the full arm to be EVALUATED and weak "
                     "(<0.15, sign-aware); if UNEVALUATED the criterion is INCOMPLETE, never "
                     "a scope-down. descriptive_only=true means the full arm is measured but the "
                     "MLP arm already KILLed — reported for completeness, does not affect the "
                     "verdict.")},
    }
    kills = [k for k, v in criteria.items() if v.get("kill")]
    fidelity_unevaluated = fid is None
    freq_pass_clause = ("is not worse than a within-slice unigram approximation (a real "
                        "frequency baseline is pending)" if freq_is_approximate
                        else "beats the frequency baseline")
    if kills:
        verdict = "KILL — " + "; ".join(kills)
    elif fidelity_unevaluated:
        verdict = ("INCOMPLETE — no KILL criterion tripped on available stages, but "
                   "proxy-fidelity (s4) is missing; run s4 before claiming PASS")
        if full_unevaluated:
            verdict += " | full-model arm UNEVALUATED"
        elif c_scope_down:
            verdict += " | SCOPE-DOWN candidate (full arm evaluated-weak, MLP holds)"
    elif c_scope_down:
        verdict = ("SCOPE-DOWN — MLP-only predictor holds but the full-model arm was "
                   "EVALUATED and weak (<0.15); report as MLP-only, not a kill")
    elif full_unevaluated:
        verdict = ("PASS (MLP arm) — a-priori ||tau_L k|| predicts per-edit erasure, "
                   f"survives the joint confound partial, {freq_pass_clause}, and the cheap "
                   "proxy is faithful to real gsm8k FT (see nontautological_predictor_vs_realft "
                   "for the least-circular evidence). FULL-MODEL ARM UNEVALUATED — run it to "
                   "test the 2511.05852 non-edited-layer-erases-more prediction.")
    else:
        verdict = ("PASS — a-priori ||tau_L k|| predicts per-edit erasure; survives the "
                   f"joint confound partial, {freq_pass_clause}, and the cheap proxy is "
                   "faithful to real gsm8k FT (see nontautological_predictor_vs_realft for "
                   "the least-circular evidence).")
    result["verdict_criteria"] = criteria
    result["VERDICT"] = verdict
    return result


# =================================================================== s5 file path
def _load_npz(path):
    return np.load(path, allow_pickle=False)


def run_s5(args):
    tau = {}
    for arm in ARMS:
        p = os.path.join(TAUDIR, f"tau_{arm}.npz")
        if os.path.isfile(p):
            tau[arm] = _load_npz(p)
    s2p = os.path.join(OUTDIR, "hl0_s2_edits.npz")
    s3p = os.path.join(OUTDIR, "hl0_s3_proxy.npz")
    s4p = os.path.join(OUTDIR, "hl0_s4_realft.npz")
    missing = [x for x in (("tau_mlp", "mlp" in tau), ("s2", os.path.isfile(s2p)),
                           ("s3", os.path.isfile(s3p))) if not x[1]]
    if missing:
        raise SystemExit(f"[hl0-s5] missing prerequisite stage outputs: "
                         f"{[m[0] for m in missing]} — run --stage s1/s2/s3 first "
                         f"(look under {OUTDIR})")
    s2 = _load_npz(s2p)
    s3 = _load_npz(s3p)
    K = s2["k_edit"].astype(np.float64)                       # [N, d_in]
    # predictor P_i = ||tau_L k_i|| per arm
    predictors = {}
    for arm in tau:
        tau_L = tau[arm]["tau_L"].astype(np.float64)          # [d_out, d_in]
        proj = tau_L @ K.T                                    # [d_out, N]
        predictors[arm] = np.linalg.norm(proj, axis=0)        # [N]
    confounds = {}
    for name in CONFOUND_NAMES:
        if name in s2.files:
            confounds[name] = s2[name].astype(np.float64)
    # erasures[arm][alpha] and control drift come from s3 (keys: er_{arm}_a{alpha}, ctl_{arm}_a{alpha}).
    # M2 edge: an arm may have a tau_{arm}.npz (so predictors[arm] exists) but NO s3 erasure
    # arrays — e.g. the full arm was skipped in s3 because tau_full_state.pt was missing. DROP
    # that arm (record it) instead of SystemExit'ing the whole s5 and losing the MLP verdict.
    erasures, control_drift = {}, {}
    dropped_arms = []
    for arm in list(predictors):
        erasures[arm], control_drift[arm] = {}, {}
        for a in ALPHAS:
            ek = f"er_{arm}_a{a}"
            if ek in s3.files:
                erasures[arm][a] = s3[ek].astype(np.float64)
            ck = f"ctl_{arm}_a{a}"
            if ck in s3.files:
                control_drift[arm][a] = s3[ck].astype(np.float64)
        if not erasures[arm]:
            dropped_arms.append(arm)
            del erasures[arm], control_drift[arm], predictors[arm]
    if not predictors:
        raise SystemExit("[hl0-s5] s3 npz has NO erasure arrays for ANY arm — re-run --stage s3")
    proxy_real = None
    if os.path.isfile(s4p):
        s4 = _load_npz(s4p)
        arm4 = str(s4["arm"]) if "arm" in s4.files else "mlp"
        a4 = float(s4["alpha"]) if "alpha" in s4.files else ALPHA_HEADLINE
        idx = s4["edit_idx"].astype(int)
        real = s4["real_erasure"].astype(np.float64)
        prox = erasures.get(arm4, {}).get(a4)
        if prox is not None:
            proxy_real = {"arm": arm4, "alpha": a4, "edit_idx": idx,
                          "real_erasure": real, "proxy_erasure": prox[idx]}
    # M3: frequency-baseline provenance. s2 stamps target_freq_provenance; only a real
    # large-corpus unigram table flips this to non-approximate. Absent the stamp, assume the
    # (degenerate) within-slice unigram approximation -> the head-to-head is baseline-only.
    freq_prov = (str(s2["target_freq_provenance"]) if "target_freq_provenance" in s2.files
                 else "within_slice_unigram_approx")
    freq_is_approximate = (freq_prov != "real_corpus_unigram")
    prov = {"tau_arms": list(tau.keys()), "arms_analyzed": list(predictors.keys()),
            "arms_dropped_no_s3_erasure": dropped_arms,
            "s2_npz": os.path.basename(s2p), "s3_npz": os.path.basename(s3p),
            "s4_present": os.path.isfile(s4p),
            "target_freq_provenance": freq_prov,
            "n_edits": int(K.shape[0])}
    res = analyze_hl0(predictors, erasures, confounds, control_drift=control_drift,
                      proxy_real=proxy_real, n_perm=args.n_perm, provenance=prov,
                      freq_is_approximate=freq_is_approximate)
    _atomic_json(args.out, res)
    print("\n=== HL0 KILL-GATE VERDICT ===")
    print(res["VERDICT"])
    print(f"[hl0-s5] wrote {args.out}")
    return res


def _atomic_json(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    json.dump(obj, open(tmp, "w"), indent=2)
    os.replace(tmp, path)


# =================================================================== GPU stages
# torch + transformers + datasets are imported LAZILY inside these functions ONLY, so s5
# and --selftest run on pure numpy with no CUDA init and no heavy imports.
def _lazy_torch():
    import torch  # noqa
    return torch


def _load_facts(data_path, seed, n_edits, n_probes, n_holdout=0):
    """Deterministic CounterFact slice shared by every stage (verbatim selection logic
    from killgate_keygeom.load_counterfact so edit/control identity is identical to the
    B6 harness under the same seed)."""
    data = json.load(open(data_path))
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
        if len(recs) >= n_edits + n_probes + n_holdout:
            break
    edits = recs[:n_edits]
    probes = recs[n_edits:n_edits + n_probes]
    return edits, probes


def _load_gsm8k(n_examples):
    """Load the FIRST n gsm8k (main/train) Q,A pairs from the LOCAL HF cache. NEVER
    downloads: relies on HF_HUB_OFFLINE=1 + the on-disk arrow. Raises a clean SystemExit
    (CONFIG-skip) if the cache is absent so a GPU stage aborts with an actionable message
    instead of hitting the network.

    ASSUMPTION (asserted below, not just typed): the returned object is a map-style,
    NON-STREAMING datasets.Dataset that supports len()/__getitem__. load_dataset returns
    a DatasetDict when NO split= is passed and an IterableDataset when streaming=True —
    both would crash the len(ds)/ds[i] loop at GPU-run time, long after the CPU self-test
    passes (the self-test never calls this). So we pin split='train', streaming=False AND
    hard-assert isinstance(ds, datasets.Dataset) with an actionable message; the arrow
    fallback (datasets.Dataset.from_file) is a Dataset by construction and passes the same
    guard."""
    try:
        import datasets  # noqa
    except Exception as e:  # pragma: no cover
        raise SystemExit(f"[hl0] gsm8k FT needs the `datasets` package: {e}")
    ds = None
    try:
        # explicit split= (=> Dataset, not DatasetDict) and streaming=False (=> not
        # IterableDataset) — both pins are load-bearing for the len()/ds[i] access below.
        ds = datasets.load_dataset("gsm8k", "main", split="train", streaming=False)
    except Exception:
        # direct arrow fallback (survives offline hub-metadata failures). NB the arrow lives
        # TWO dirs deep: main/<version>/<hash>/gsm8k-train.arrow — a one-level glob
        # (main/*/gsm8k-train.arrow) MISSES it and falsely reports the cache absent (this was
        # the "gsm8k absent" symptom the reviewer hit). Use a RECURSIVE glob.
        base = os.path.expanduser("~/.cache/huggingface/datasets/openai___gsm8k/main")
        cand = glob.glob(os.path.join(base, "**", "gsm8k-train.arrow"), recursive=True)
        if not cand:
            raise SystemExit(
                "[hl0] gsm8k cache not found (looked for datasets.load_dataset('gsm8k','main') "
                f"and {base}/*/gsm8k-train.arrow). Downloads are ask-first (HF_HUB_OFFLINE=1) "
                "— provide the gsm8k cache or skip the tau-based stages.")
        ds = datasets.Dataset.from_file(sorted(cand)[0])
    # HARD guard: a DatasetDict (no split=) or IterableDataset (streaming) does NOT support
    # the len()/__getitem__ contract the FT batch builder relies on. Fail loud + actionable
    # instead of AttributeError/KeyError deep in the loop.
    if not isinstance(ds, datasets.Dataset):
        raise SystemExit(
            f"[hl0] gsm8k loaded as {type(ds).__name__}, expected a non-streaming "
            "datasets.Dataset (map-style). This means split= was dropped (DatasetDict) or "
            "streaming was enabled (IterableDataset) — fix the load call; the FT batch "
            "builder needs len()/ds[i].")
    out = []
    for i in range(min(n_examples, len(ds))):
        r = ds[i]
        out.append((r["question"], r["answer"]))
    return out


def _build_ft_batch(tok, gsm_pairs, device, max_len=384):
    """Tokenise gsm8k Q->A pairs; label-mask everything but the answer span (CE only on
    the answer tokens, exactly the ft_editor loss convention). Returns a list of
    (input_ids[1,T], labels[1,T])."""
    torch = _lazy_torch()
    batch = []
    for q, a in gsm_pairs:
        q_ids = tok.encode(q + "\n", add_special_tokens=True)
        a_ids = tok.encode(" " + a.strip(), add_special_tokens=False)
        ids = (q_ids + a_ids)[:max_len]
        input_ids = torch.tensor([ids], device=device)
        labels = torch.full_like(input_ids, -100)
        astart = min(len(q_ids), len(ids))
        if astart < len(ids):
            labels[0, astart:] = torch.tensor(ids[astart:], device=device)
        batch.append((input_ids, labels))
    return batch


def _short_finetune(model, tok, gsm_pairs, device, arm, layer, steps, lr,
                    base_state_cpu=None):
    """Short gsm8k fine-tune producing a task vector tau = theta_ft - theta_base.

    arm='mlp'  : freeze all, unfreeze ONLY model.model.layers[layer].mlp.down_proj.weight.
    arm='full' : unfreeze ALL params (2511.05852 — non-edited-layer FT erases more).
    Returns (tau_L [d_out,d_in] numpy, full_delta_state_dict-on-CPU or None). fp32
    throughout (the model is loaded fp32 by the driver; matches the ROME fp32 rule).

    GPU-MEMORY (24GB laptop, M1 review fix). The full arm needs theta_base to form tau, but
    a SECOND on-GPU clone of the whole model (~4.94GB fp32) on top of run_s1's own base copy
    blew the budget (model 4.94 + 2 clones 9.88 + Adam m/v 9.88 + grads 4.94 = ~29.7 > 24).
    Fix (three parts): (1) the SINGLE base copy lives on CPU (`base_state_cpu`, passed by
    run_s1, needed only for the delta) — no redundant GPU clone; (2) the optimizer state is
    freed BEFORE the delta is read, and the full arm runs under gradient checkpointing; (3)
    the full-arm Adam uses foreach=False (see below) so opt.step() does NOT materialize a
    transient whole-model v-copy. Resident GPU budget: model 4.94 + Adam m/v 9.88 + grads 4.94
    = ~19.8GB RESIDENT + (checkpointed) activations + ~0.7GB foreign context + allocator
    reserve. The FIRST live run OOMed at 22.37GB (76MB free) because the default foreach=True
    Adam allocated an ADDITIONAL transient at _foreach_sqrt on top of that resident state.
    foreach=False removes the transient; the re-run's OBSERVED peak was 22456 MiB (~21.9GiB),
    fitting under the 23.4GiB card cap with ~1.5GiB headroom (with foreach=True the extra
    transient exceeds the cap). The driver also exports PYTORCH_CUDA_ALLOC_CONF=
    expandable_segments:True to cut fragmentation. NB the headroom is modest — a larger model
    or a card carrying more foreign context may still need optimizer-state CPU offload.
    Optimizer choice UNCHANGED (Adam) — the tau convention is not silently altered."""
    torch = _lazy_torch()
    model.train()
    for p in model.parameters():
        p.requires_grad_(False)
    W = model.model.layers[layer].mlp.down_proj.weight
    base_W_gpu = None
    used_gc = False
    if arm == "mlp":
        trainable = [W]
        W.requires_grad_(True)
        base_W_gpu = W.detach().clone()          # tiny (one down_proj); GPU is fine
    else:
        trainable = list(model.parameters())
        for p in trainable:
            p.requires_grad_(True)
        # the ONE base copy (CPU). Reuse run_s1's if given; else build on CPU (no GPU cost).
        if base_state_cpu is None:
            base_state_cpu = {n: p.detach().float().cpu().clone()
                              for n, p in model.named_parameters()}
        # gradient checkpointing (best-effort) trims activation memory for the full arm.
        try:
            model.config.use_cache = False
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
            used_gc = True
        except Exception:
            try:
                model.gradient_checkpointing_enable(); used_gc = True
            except Exception:
                used_gc = False
    batch = _build_ft_batch(tok, gsm_pairs, device)
    # OOM FIX (full arm): torch's default multi-tensor Adam (foreach=True on CUDA) runs
    # _foreach_sqrt over ALL exp_avg_sq tensors at opt.step(), which materializes a TRANSIENT
    # whole-model-sized copy (~4.9GB fp32 for 1B params) on top of model+grads+Adam state —
    # that transient is what OOMed at 22.37GB (torch/optim/adam.py:790 _foreach_sqrt). foreach=
    # False runs a per-parameter Python loop with NO whole-model temporary, dropping peak by
    # ~one v-copy to ~17-18GB. Adam itself is UNCHANGED (same math/state) — only the kernel
    # dispatch changes, so the tau convention is preserved. mlp arm keeps the default (foreach
    # =None): a single small down_proj param, where the multi-tensor path is already cheap.
    foreach = False if arm == "full" else None
    opt = torch.optim.Adam([p for p in trainable if p.requires_grad], lr=lr, foreach=foreach)
    for _ in range(steps):
        for input_ids, labels in batch:
            opt.zero_grad(set_to_none=True)
            out = model(input_ids=input_ids, labels=labels)
            out.loss.backward()
            opt.step()
    # free optimizer state + grads BEFORE reading the delta (peak-memory reduction)
    del opt
    model.zero_grad(set_to_none=True)
    for p in model.parameters():
        p.requires_grad_(False)
    if used_gc:
        try:
            model.gradient_checkpointing_disable()
        except Exception:
            pass
        model.config.use_cache = True
    model.eval()
    with torch.no_grad():
        W_now = model.model.layers[layer].mlp.down_proj.weight.detach().float()
        if arm == "mlp":
            assert base_W_gpu is not None      # set in the mlp branch above (guard + narrows None)
            tau_L = (W_now - base_W_gpu.float()).cpu().numpy().astype(np.float32)
            full_delta = None
        else:
            assert base_state_cpu is not None   # built/reused in the full branch above (guard + narrows)
            base_W_cpu = base_state_cpu[f"model.layers.{layer}.mlp.down_proj.weight"].float()
            tau_L = (W_now.cpu() - base_W_cpu).numpy().astype(np.float32)
            # full delta on CPU, param-by-param (never a second whole-model GPU tensor)
            full_delta = {n: (p.detach().float().cpu() - base_state_cpu[n].float())
                          for n, p in model.named_parameters()}
    return tau_L, full_delta


def _load_model(args):
    torch = _lazy_torch()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not os.path.isdir(args.model):
        raise SystemExit(f"[hl0] model dir not found: {args.model} — CONFIG-SKIP "
                         "(no download; provide the local checkpoint).")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(args.device).eval()
    return model, tok


def _margin(model, tok, prompt, new_id, true_id, device):
    """Signed target margin = logit(target_new_first) - logit(target_true_first) at the
    prompt's next-token position. Higher => model prefers the (new) edit target."""
    from metrics import next_token_logits
    logits = next_token_logits(model, tok, prompt, device)
    return float(logits[new_id].item() - logits[true_id].item())


def run_s1(args):
    """Compute tau ONCE per arm (idempotent per arm)."""
    os.makedirs(TAUDIR, exist_ok=True)
    torch = _lazy_torch()
    todo = [a for a in ARMS if not os.path.isfile(os.path.join(TAUDIR, f"tau_{a}.npz"))]
    if not todo:
        print("[hl0-s1] all tau_{arm}.npz present — skip"); return
    model, tok = _load_model(args)
    layer = args.layer
    gsm = _load_gsm8k(args.n_ft)
    # the SINGLE base copy lives on CPU (M1 OOM fix): it is only needed to restore the model
    # between arms and to form the full-arm delta — never as a second on-GPU model. copy_
    # from a CPU tensor into a CUDA Parameter is a valid cross-device copy.
    base_state = {n: p.detach().float().cpu().clone() for n, p in model.named_parameters()}
    for arm in todo:
        # restore base before each arm (full arm mutates every param)
        with torch.no_grad():
            for n, p in model.named_parameters():
                p.copy_(base_state[n])
        tau_L, full_delta = _short_finetune(model, tok, gsm, args.device, arm, layer,
                                            args.ft_steps, args.ft_lr,
                                            base_state_cpu=base_state)
        # Write ORDER matters for idempotent resume: for the full arm, save the multi-GB
        # tau_full_state.pt FIRST, then tau_{arm}.npz. That makes tau_full.npz the TERMINAL
        # marker whose presence GUARANTEES the .pt is already on disk — so run_s1's per-arm
        # todo check (gate on tau_{arm}.npz) and s3's full-arm gate (also on the .npz) can
        # never see a .npz without its .pt. A crash between them leaves the .pt but no .npz,
        # and the next run re-does the full arm (idempotent, overwrites the .pt).
        if arm == "full" and full_delta is not None:
            _atomic_torch_save(full_delta, os.path.join(TAUDIR, "tau_full_state.pt"), torch)
        _atomic_npz(os.path.join(TAUDIR, f"tau_{arm}.npz"),
                    dict(tau_L=tau_L.astype(np.float32),
                         arm=np.array(arm, dtype="U8"),
                         layer=np.array(layer, dtype=np.int64),
                         ft_steps=np.array(args.ft_steps, dtype=np.int64),
                         ft_lr=np.array(args.ft_lr, dtype=np.float64),
                         n_ft=np.array(args.n_ft, dtype=np.int64)))
        print(f"[hl0-s1] arm={arm}: ||tau_L||_F={float(np.linalg.norm(tau_L)):.4g} "
              f"-> tau_{arm}.npz")
    # leave the model restored to base
    with torch.no_grad():
        for n, p in model.named_parameters():
            p.copy_(base_state[n])


def _rome_factors(model, tok, e, layer, steps, lr, device):
    """Install a ROME edit and return (residual_vec[d_out], k[d_in], denom, dW_norm,
    resid_norm) WITHOUT restoring — caller restores. Reuses the native editor so the
    rank-one factors are byte-identical to a normal killgate edit; DW = outer(residual,k)/denom."""
    from editors.rome_native import apply_edit
    cfg = {"layer": layer, "steps": steps, "lr": lr}
    info = apply_edit(model, tok, e, cfg, device)
    return (info["residual_vec"].astype(np.float32),
            None,  # key filled by caller (captured separately, base model)
            float(info["key_norm"]) ** 2 + 1e-8,
            float(info["delta_weight_norm"]),
            float(info["residual_norm"]))


def run_s2(args):
    """200 ROME edits + 100 controls; record per-edit predictors' inputs + confounds."""
    out = os.path.join(OUTDIR, "hl0_s2_edits.npz")
    if os.path.isfile(out):
        print(f"[hl0-s2] {out} exists — skip"); return
    torch = _lazy_torch()
    from metrics import first_target_token_id
    from editors.rome_native import _capture_key, find_subject_last_token_index
    model, tok = _load_model(args)
    layer = args.layer
    edits, controls = _load_facts(args.data, args.seed, args.n_edits, args.n_controls)
    # target-token frequency (2511.05852 baseline input): corpus unigram freq of each
    # edit's target_new FIRST token over the loaded slice (documented approximation to
    # pretraining frequency — see the reviewer-assumptions note).
    all_first = []
    for r in edits + controls:
        all_first.append(first_target_token_id(tok, r["target_new"]))
    uniq, counts = np.unique(np.array(all_first), return_counts=True)
    freq_map = {int(t): int(c) for t, c in zip(uniq, counts)}
    tot = float(sum(counts))

    W = model.model.layers[layer].mlp.down_proj.weight
    W_base = W.detach().clone()
    N = len(edits)
    d_in = W.shape[1]; d_out = W.shape[0]
    k_edit = np.zeros((N, d_in), np.float32)
    resid = np.zeros((N, d_out), np.float32)
    denom = np.zeros(N, np.float32)
    norm_growth = np.zeros(N, np.float32)
    S = np.zeros(N, np.float32)
    knorm = np.zeros(N, np.float32)
    target_freq = np.zeros(N, np.float32)
    pre_edit_ptarget = np.zeros(N, np.float32)
    margin_edited = np.zeros(N, np.float32)
    edit_ok = np.zeros(N, np.float32)
    new_ids = np.zeros(N, np.int64); true_ids = np.zeros(N, np.int64)

    for i, e in enumerate(edits):
        new_id = first_target_token_id(tok, e["target_new"])
        true_id = first_target_token_id(tok, e["target_true"])
        new_ids[i], true_ids[i] = new_id, true_id
        idx = find_subject_last_token_index(tok, e["prompt"], e["subject"])
        k = _capture_key(model, tok, layer, e["prompt"], idx, args.device).float().cpu().numpy()
        k_edit[i] = k.astype(np.float32); knorm[i] = float(np.linalg.norm(k))
        target_freq[i] = freq_map.get(int(new_id), 0) / tot
        from metrics import next_token_logits
        lg = next_token_logits(model, tok, e["prompt"], args.device)
        pre_edit_ptarget[i] = float(torch.softmax(lg, -1)[new_id].item())
        rv, _, dn, dwn, rn = _rome_factors(model, tok, e, layer, args.steps, args.lr, args.device)
        resid[i] = rv; denom[i] = dn; norm_growth[i] = dwn; S[i] = rn
        margin_edited[i] = _margin(model, tok, e["prompt"], new_id, true_id, args.device)
        eff_lg = next_token_logits(model, tok, e["prompt"], args.device)
        edit_ok[i] = 1.0 if int(torch.argmax(eff_lg).item()) == new_id else 0.0
        with torch.no_grad():
            W.copy_(W_base)
        if (i + 1) % 25 == 0:
            print(f"[hl0-s2] edit {i+1}/{N}", flush=True)

    # controls (unedited): base-model margins for the s3 specificity drift baseline
    cN = len(controls)
    ctl_new = np.zeros(cN, np.int64); ctl_true = np.zeros(cN, np.int64)
    ctl_margin_base = np.zeros(cN, np.float32)
    for j, c in enumerate(controls):
        nid = first_target_token_id(tok, c["target_new"])
        tid = first_target_token_id(tok, c["target_true"])
        ctl_new[j], ctl_true[j] = nid, tid
        ctl_margin_base[j] = _margin(model, tok, c["prompt"], nid, tid, args.device)

    _atomic_npz(out, dict(
        k_edit=k_edit, resid_vec=resid, denom=denom,
        norm_growth=norm_growth, S=S, knorm=knorm,
        target_freq=target_freq, pre_edit_ptarget=pre_edit_ptarget,
        margin_edited=margin_edited, edit_ok=edit_ok,
        new_ids=new_ids, true_ids=true_ids,
        ctl_new=ctl_new, ctl_true=ctl_true, ctl_margin_base=ctl_margin_base,
        layer=np.array(layer, np.int64), seed=np.array(args.seed, np.int64),
        n_edits=np.array(N, np.int64), n_controls=np.array(cN, np.int64),
        W_base=W_base.detach().float().cpu().numpy().astype(np.float32),
        # M3: target_freq here is a within-slice unigram approximation (~n-sample counts),
        # NOT the pretraining frequency factor of 2511.05852. Stamp its provenance so s5
        # reports the frequency head-to-head as APPROXIMATE-BASELINE-ONLY. Flip this to
        # 'real_corpus_unigram' only if target_freq is recomputed from a real large-corpus
        # unigram table (none available locally; ask before downloading one).
        target_freq_provenance=np.array("within_slice_unigram_approx", dtype="U32"),
    ))
    print(f"[hl0-s2] esr={float(edit_ok.mean()):.3f} -> {out}")


def _apply_delta_all(model, torch, full_delta, alpha, sign=1.0):
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in full_delta:
                p.add_(sign * alpha * full_delta[n].to(p.device, p.dtype))


def run_s3(args):
    """The cheap PROXY: W += alpha*tau, reinstall each edit's rank-one DW, forward,
    erasure_i = margin_edited_i - margin_proxy_i. Requires s1 (tau) + s2 (edits)."""
    out = os.path.join(OUTDIR, "hl0_s3_proxy.npz")
    torch = _lazy_torch()
    s2p = os.path.join(OUTDIR, "hl0_s2_edits.npz")
    if not os.path.isfile(s2p):
        raise SystemExit("[hl0-s3] hl0_s2_edits.npz absent — run --stage s2 first")
    # INCREMENTAL (do NOT recompute already-computed arms). The first live run wrote the MLP
    # proxies before the full arm existed; this preserves er_mlp_*/ctl_mlp_* byte-for-byte and
    # computes ONLY the arms still missing from the npz. Decide what to do BEFORE loading the
    # model, so an all-present re-run costs zero GPU.
    existing = {}
    if os.path.isfile(out):
        with np.load(out) as _e:
            existing = {k: _e[k] for k in _e.files}
    tau = {}
    for arm in ARMS:
        pth = os.path.join(TAUDIR, f"tau_{arm}.npz")
        if os.path.isfile(pth):
            tau[arm] = _load_npz(pth)
    if not tau:
        raise SystemExit("[hl0-s3] no tau_{arm}.npz — run --stage s1 first")
    fdp = os.path.join(TAUDIR, "tau_full_state.pt")

    def _needs(arm):
        if f"er_{arm}_a{ALPHAS[0]}" in existing:
            return False                                   # already computed — preserve
        if arm == "full" and not os.path.isfile(fdp):
            print(f"[hl0-s3] arm=full: {fdp} absent — cannot compute full proxies, skipping")
            return False
        return True

    arms_to_do = [a for a in tau if _needs(a)]
    if not arms_to_do:
        print(f"[hl0-s3] all available arms already present in {out} "
              f"(er_ keys: {sorted(k for k in existing if k.startswith('er_'))}) — skip")
        return
    print(f"[hl0-s3] incremental: computing {arms_to_do}; preserving "
          f"{sorted(k for k in existing if k.startswith('er_'))}", flush=True)

    s2 = _load_npz(s2p)
    layer = int(s2["layer"])
    model, tok = _load_model(args)
    W = model.model.layers[layer].mlp.down_proj.weight
    W_base = torch.tensor(s2["W_base"], device=W.device, dtype=W.dtype)
    with torch.no_grad():
        W.copy_(W_base)                                        # ensure clean base
    # base copy on CPU (restore only; keeps GPU headroom — M1 review consistency)
    base_state = {n: p.detach().float().cpu().clone() for n, p in model.named_parameters()}

    K = s2["k_edit"]; RV = s2["resid_vec"]; DEN = s2["denom"]
    new_ids = s2["new_ids"]; true_ids = s2["true_ids"]
    margin_edited = s2["margin_edited"].astype(np.float64)
    N = K.shape[0]
    edits, controls = _load_facts(args.data, int(s2["seed"]), int(s2["n_edits"]),
                                  int(s2["n_controls"]))
    ctl_new = s2["ctl_new"]; ctl_true = s2["ctl_true"]
    ctl_base = s2["ctl_margin_base"].astype(np.float64)

    full_delta = None
    arrs = dict(existing)                                     # start from the preserved arms
    for arm in arms_to_do:
        tau_L = torch.tensor(tau[arm]["tau_L"], device=W.device, dtype=torch.float32)
        if arm == "full":
            full_delta = torch.load(fdp, map_location=W.device)
        for a in ALPHAS:
            # 1) restore base everywhere, 2) apply proxy
            with torch.no_grad():
                for n, p in model.named_parameters():
                    p.copy_(base_state[n])
            if arm == "full":
                _apply_delta_all(model, torch, full_delta, a, sign=1.0)
            else:
                with torch.no_grad():
                    W.add_((a * tau_L).to(W.dtype))
            W_proxy = model.model.layers[layer].mlp.down_proj.weight
            W_proxy_snap = W_proxy.detach().clone()      # proxied edited-layer state
            er = np.zeros(N, np.float64)
            for i in range(N):
                k = torch.tensor(K[i], device=W.device, dtype=torch.float32)
                rv = torch.tensor(RV[i], device=W.device, dtype=torch.float32)
                dW = torch.outer(rv, k) / float(DEN[i])
                with torch.no_grad():
                    W_proxy.add_(dW.to(W_proxy.dtype))       # reinstall edit on proxied W
                mnow = _margin(model, tok, edits[i]["prompt"],
                               int(new_ids[i]), int(true_ids[i]), args.device)
                er[i] = margin_edited[i] - mnow              # margin DROP = erasure
                with torch.no_grad():
                    W_proxy.copy_(W_proxy_snap)              # back to proxied (no edit)
            # control drift under the proxy (no edit installed)
            cd = np.zeros(len(controls), np.float64)
            for j, c in enumerate(controls):
                mnow = _margin(model, tok, c["prompt"], int(ctl_new[j]), int(ctl_true[j]),
                               args.device)
                cd[j] = ctl_base[j] - mnow
            arrs[f"er_{arm}_a{a}"] = er.astype(np.float32)
            arrs[f"ctl_{arm}_a{a}"] = cd.astype(np.float32)
            print(f"[hl0-s3] arm={arm} alpha={a}: mean_erasure={er.mean():.4g} "
                  f"mean_control_drift={cd.mean():.4g}", flush=True)
    # restore base
    with torch.no_grad():
        for n, p in model.named_parameters():
            p.copy_(base_state[n])
    # arms actually present in the merged npz (preserved + newly computed), derived from the
    # er_ keys so the label can never claim an arm whose erasures aren't on disk.
    present_arms = sorted({k.split("_")[1] for k in arrs if k.startswith("er_")})
    arrs["arms"] = np.array(",".join(present_arms), dtype="U16")
    _atomic_npz(out, arrs)
    print(f"[hl0-s3] wrote {out} (arms present: {present_arms})")


def run_s4(args):
    """Proxy fidelity: for n=40 sampled edits, install the edit then run a REAL short
    gsm8k FT (arm=mlp by default) and read the edit's margin drop under genuine
    adaptation. s5 correlates this with the s3 proxy erasure (bootstrap CI)."""
    out = os.path.join(OUTDIR, "hl0_s4_realft.npz")
    if os.path.isfile(out):
        print(f"[hl0-s4] {out} exists — skip"); return
    torch = _lazy_torch()
    from metrics import first_target_token_id
    from editors.rome_native import apply_edit
    s2p = os.path.join(OUTDIR, "hl0_s2_edits.npz")
    if not os.path.isfile(s2p):
        raise SystemExit("[hl0-s4] hl0_s2_edits.npz absent — run --stage s2 first")
    s2 = _load_npz(s2p)
    layer = int(s2["layer"])
    model, tok = _load_model(args)
    edits, _ = _load_facts(args.data, int(s2["seed"]), int(s2["n_edits"]), int(s2["n_controls"]))
    gsm = _load_gsm8k(args.n_ft)
    rng = np.random.default_rng(RNG_SEED)
    N = int(s2["n_edits"])
    idx = np.sort(rng.choice(N, size=min(args.n_real, N), replace=False))
    # base copy on CPU (restore only; keeps GPU headroom — M1 review consistency)
    base_state = {n: p.detach().float().cpu().clone() for n, p in model.named_parameters()}
    margin_edited = s2["margin_edited"].astype(np.float64)
    new_ids = s2["new_ids"]; true_ids = s2["true_ids"]
    real_er = np.zeros(len(idx), np.float64)
    for c, i in enumerate(idx):
        e = edits[int(i)]
        with torch.no_grad():
            for n, p in model.named_parameters():
                p.copy_(base_state[n])
        apply_edit(model, tok, e, {"layer": layer, "steps": args.steps, "lr": args.lr}, args.device)
        # REAL short FT (arm=mlp: unfreeze the edited-layer down_proj only) on the ALREADY
        # edited model — this is the ground-truth adaptation the s3 proxy approximates.
        _short_finetune(model, tok, gsm, args.device, "mlp", layer, args.ft_steps, args.ft_lr)
        mnow = _margin(model, tok, e["prompt"], int(new_ids[int(i)]), int(true_ids[int(i)]),
                       args.device)
        real_er[c] = margin_edited[int(i)] - mnow
        print(f"[hl0-s4] {c+1}/{len(idx)} edit#{int(i)} real_erasure={real_er[c]:.4g}", flush=True)
    with torch.no_grad():
        for n, p in model.named_parameters():
            p.copy_(base_state[n])
    _atomic_npz(out, dict(edit_idx=idx.astype(np.int64), real_erasure=real_er.astype(np.float32),
                          arm=np.array("mlp", dtype="U8"),
                          alpha=np.array(ALPHA_HEADLINE, np.float64),
                          n_real=np.array(len(idx), np.int64)))
    print(f"[hl0-s4] wrote {out}")


def _atomic_npz(path, arrs):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **arrs)
    os.replace(tmp, path)


def _atomic_torch_save(state_dict, path, torch):
    """Atomic (tmp + os.replace) torch.save of the multi-GB full-arm delta, with a
    free-disk-space precheck (minor review fix). A crash mid-write can never leave a
    truncated 'tau_full_state.pt' that s3 would then torch.load and mis-apply."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    need = int(sum(t.numel() * t.element_size() for t in state_dict.values()))
    try:
        free = os.statvfs(os.path.dirname(path) or ".")
        free_bytes = free.f_bavail * free.f_frsize
    except Exception:
        free_bytes = None
    if free_bytes is not None and free_bytes < need + (1 << 30):   # +1GiB margin
        raise SystemExit(f"[hl0-s1] insufficient disk for tau_full_state.pt: need ~"
                         f"{need/1e9:.1f}GB + 1GiB margin, have {free_bytes/1e9:.1f}GB free "
                         f"at {os.path.dirname(path)!r}. Free space or skip the full arm.")
    tmp = path + ".tmp"
    torch.save(state_dict, tmp)
    os.replace(tmp, path)


# =================================================================== self-test (CPU)
def _make_confounds(rng, N):
    return {
        "knorm": rng.normal(5.0, 1.0, N),
        "norm_growth": rng.normal(2.0, 0.5, N),
        "S": rng.normal(1.0, 0.3, N),
        "target_freq": rng.uniform(0, 1, N),
        "pre_edit_ptarget": rng.uniform(0, 0.3, N),
    }


def _synth_bundle(kind, N=200, seed=0):
    """Synthetic per-edit HL0 analysis bundle. kind in {positive, kill, confound, mechanical}.
    Returns (predictors, erasures, confounds, proxy_real)."""
    rng = np.random.default_rng(seed)
    conf = _make_confounds(rng, N)
    P_mlp = rng.normal(0, 1, N)
    P_full = 0.7 * P_mlp + 0.3 * rng.normal(0, 1, N)   # correlated but distinct arm
    predictors = {"mlp": P_mlp, "full": P_full}
    erasures = {"mlp": {}, "full": {}}
    if kind == "positive":
        # erasure grows with the predictor; confounds independent of erasure
        for arm, P in predictors.items():
            for a in ALPHAS:
                erasures[arm][a] = a * (2.0 * _midrank(P) / N + rng.normal(0, 0.3, N))
        # proxy faithful to real: real ~ proxy(mlp, headline) + small noise
        idx = np.sort(rng.choice(N, 40, replace=False))
        prox = erasures["mlp"][ALPHA_HEADLINE][idx]
        real = prox + rng.normal(0, 0.05, idx.size)
        proxy_real = {"arm": "mlp", "alpha": ALPHA_HEADLINE, "edit_idx": idx,
                      "proxy_erasure": prox, "real_erasure": real}
    elif kind == "mechanical":
        # CIRCULARITY MODE: the s3 proxy erasure is MECHANICALLY alpha*P (to machine
        # precision, as the real s3 perturbation norm is), so proxy_fidelity is ~perfect but
        # tautological. real_erasure is an INDEPENDENT (noisier) P-correlated measurement —
        # the non-tautological signal. Tests that the pipeline surfaces BOTH and does not let
        # the tautological proxy masquerade as the clean evidence.
        for arm, P in predictors.items():
            for a in ALPHAS:
                erasures[arm][a] = a * P + rng.normal(0, 1e-6, N)   # == alpha*P_i, machine-eps
        idx = np.sort(rng.choice(N, 40, replace=False))
        prox = erasures["mlp"][ALPHA_HEADLINE][idx]
        real = P_mlp[idx] + rng.normal(0, 0.4, idx.size)           # clean but noisy
        proxy_real = {"arm": "mlp", "alpha": ALPHA_HEADLINE, "edit_idx": idx,
                      "proxy_erasure": prox, "real_erasure": real}
    elif kind == "kill":
        for arm in predictors:
            for a in ALPHAS:
                erasures[arm][a] = rng.normal(0, 1, N)     # pure noise
        idx = np.sort(rng.choice(N, 40, replace=False))
        proxy_real = {"arm": "mlp", "alpha": ALPHA_HEADLINE, "edit_idx": idx,
                      "proxy_erasure": rng.normal(0, 1, 40),
                      "real_erasure": rng.normal(0, 1, 40)}
    else:  # confound: erasure = f(||k||); predictor made to CORRELATE with ||k|| but be
        #   conditionally independent of erasure given ||k||.
        kn = conf["knorm"]
        for arm in predictors:
            # tie the predictor to ||k|| so the RAW rho(P, erasure) is spuriously high
            predictors[arm] = _midrank(kn) + rng.normal(0, N * 0.05, N)
            for a in ALPHAS:
                erasures[arm][a] = a * (3.0 * _midrank(kn) / N + rng.normal(0, 0.25, N))
        idx = np.sort(rng.choice(N, 40, replace=False))
        prox = erasures["mlp"][ALPHA_HEADLINE][idx]
        proxy_real = {"arm": "mlp", "alpha": ALPHA_HEADLINE, "edit_idx": idx,
                      "proxy_erasure": prox, "real_erasure": prox + rng.normal(0, 0.05, idx.size)}
    return predictors, erasures, conf, proxy_real


def run_selftest(args):
    sd = os.path.join(OUTDIR, "selftest")
    os.makedirs(sd, exist_ok=True)
    n_perm = 300
    ok = True

    def check(name, cond):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        ok = ok and bool(cond)

    print("=== HL0 SELF-TEST (synthetic, CPU) ===")

    # ---- POSITIVE fixture: erasure ~ predictor -> PASS-shaped ----
    pr, er, cf, pxr = _synth_bundle("positive", seed=1)
    rp = analyze_hl0(pr, er, cf, proxy_real=pxr, n_perm=n_perm)
    _atomic_json(os.path.join(sd, "positive.json"), rp)
    mlp_joint = rp["arms"]["mlp"]["by_alpha"][str(ALPHA_HEADLINE)]["joint_partial_rho"]
    print(f"  positive: mlp joint-partial rho={mlp_joint}  verdict={rp['VERDICT'][:60]}")
    check("positive: mlp joint-partial rho >= 0.2", mlp_joint is not None and mlp_joint >= 0.2)
    check("positive: not a KILL verdict", not rp["VERDICT"].startswith("KILL"))
    check("positive: tauk beats frequency",
          rp["frequency_head_to_head"]["tauk_beats_freq"] is True)
    check("positive: proxy fidelity CI lower >= 0.3",
          rp["proxy_fidelity"]["ci_lower"] is not None and rp["proxy_fidelity"]["ci_lower"] >= 0.3)
    check("positive: PASS verdict", rp["VERDICT"].startswith("PASS"))

    # ---- KILL fixture: erasure = noise -> KILL-shaped ----
    pr, er, cf, pxr = _synth_bundle("kill", seed=2)
    rk = analyze_hl0(pr, er, cf, proxy_real=pxr, n_perm=n_perm)
    _atomic_json(os.path.join(sd, "kill.json"), rk)
    mlp_joint_k = rk["arms"]["mlp"]["by_alpha"][str(ALPHA_HEADLINE)]["joint_partial_rho"]
    print(f"  kill: mlp joint-partial rho={mlp_joint_k}  verdict={rk['VERDICT'][:60]}")
    check("kill: mlp joint-partial rho < 0.2 (weak)",
          mlp_joint_k is None or abs(mlp_joint_k) < 0.2)
    check("kill: VERDICT is KILL", rk["VERDICT"].startswith("KILL"))
    check("kill: mlp-weak criterion tripped",
          rk["verdict_criteria"]["mlp_joint_partialled_rho_lt_0.2"]["kill"] is True)

    # ---- PLANTED-CONFOUND fixture: erasure = f(||k||), predictor correlated w/ ||k|| ----
    pr, er, cf, pxr = _synth_bundle("confound", seed=3)
    P = pr["mlp"]; E = er["mlp"][ALPHA_HEADLINE]; kn = cf["knorm"]
    raw = spearman(P, E)
    part_k, rxr, ryr = partial_spearman_multi(P, E, [kn])
    print(f"  confound: raw rho(P,erasure)={raw:.3f}  ||k||-partialled={part_k:.3f}")
    check("confound: RAW rho(P, erasure) is high (>0.3) [spurious via ||k||]",
          np.isfinite(raw) and abs(raw) > 0.3)
    check("confound: ||k||-partial COLLAPSES the planted confound (|rho|<0.15)",
          np.isfinite(part_k) and abs(part_k) < 0.15)
    rc = analyze_hl0(pr, er, cf, proxy_real=pxr, n_perm=n_perm)
    _atomic_json(os.path.join(sd, "confound.json"), rc)
    cj = rc["arms"]["mlp"]["by_alpha"][str(ALPHA_HEADLINE)]
    check("confound: single-partial(knorm) < raw (partialling removes signal)",
          cj["single_partials"]["knorm"]["partial_rho"] is not None
          and abs(cj["single_partials"]["knorm"]["partial_rho"]) < abs(cj["raw_rho"]))
    check("confound: joint-partial < 0.2 (killed once ||k|| controlled)",
          cj["joint_partial_rho"] is None or abs(cj["joint_partial_rho"]) < 0.2)

    # ---- MECHANICAL (circularity mode): the s3 proxy erasure IS alpha*P to machine precision,
    #      so the ARM ladder's raw rho(P, proxy-erasure) is tautologically ~1.0 (the circularity
    #      M4 warns about); the honest P-vs-REAL-FT number sits BELOW it and carries the weight ----
    pr, er, cf, pxr = _synth_bundle("mechanical", seed=4)
    rm = analyze_hl0(pr, er, cf, proxy_real=pxr, n_perm=n_perm)
    _atomic_json(os.path.join(sd, "mechanical.json"), rm)
    pf = rm["proxy_fidelity"]; nt = rm["nontautological_predictor_vs_realft"]
    assert pf is not None and nt is not None   # mechanical fixture always supplies proxy_real
    arm_raw = rm["arms"]["mlp"]["by_alpha"][str(ALPHA_HEADLINE)]["raw_rho"]
    print(f"  mechanical: arm raw_rho(P,proxy-erasure)={arm_raw} (TAUTOLOGICAL ~1) "
          f"P_vs_realFT={nt['spearman_P_vs_realFT']} (clean, lower)")
    check("mechanical: arm raw rho is tautologically inflated (>0.98, proxy==alpha*P)",
          arm_raw is not None and arm_raw > 0.98)
    check("mechanical: proxy_fidelity note flags the tautology",
          "tautolog" in (pf.get("note") or "").lower())
    nt_rho = nt["spearman_P_vs_realFT"]   # nt asserted non-None above
    check("mechanical: nontautological P-vs-realFT reported + finite",
          nt_rho is not None and np.isfinite(nt_rho))
    check("mechanical: clean P-vs-realFT positive but BELOW the tautological arm rho",
          nt_rho is not None and arm_raw is not None and nt_rho > 0.2 and nt_rho < arm_raw)

    # ---- M3: frequency head-to-head must be labeled APPROXIMATE + not paper-claimable ----
    check("M3: freq baseline labeled approximate (default)",
          rp["frequency_head_to_head"]["baseline_is_approximate"] is True)
    check("M3: 'beats freq' NOT paper-claimable on the approximate baseline",
          rp["frequency_head_to_head"]["beats_freq_is_paper_claimable"] is False)
    check("M3: PASS verdict does NOT assert 'beats the frequency baseline'",
          "beats the frequency baseline" not in rp["VERDICT"])

    # ---- M2: full arm ABSENT must read as UNEVALUATED, never silently SCOPE-DOWN ----
    pr, er, cf, pxr = _synth_bundle("positive", seed=5)
    pr_mlp = {"mlp": pr["mlp"]}; er_mlp = {"mlp": er["mlp"]}   # drop the full arm entirely
    rmo = analyze_hl0(pr_mlp, er_mlp, cf, proxy_real=pxr, n_perm=n_perm)
    _atomic_json(os.path.join(sd, "mlp_only.json"), rmo)
    fa = rmo["verdict_criteria"]["full_arm"]
    print(f"  mlp_only: full_arm evaluated={fa['evaluated']} unevaluated={fa['unevaluated']} "
          f"verdict={rmo['VERDICT'][:50]}")
    check("M2: full arm reported UNEVALUATED (not evaluated)",
          fa["evaluated"] is False and fa["unevaluated"] is True)
    check("M2: full arm NOT counted as scope-down when unevaluated",
          fa["scope_down"] is False)
    check("M2: verdict surfaces 'UNEVALUATED' for the full arm",
          "UNEVALUATED" in rmo["VERDICT"])

    print(f"\n=== SELF-TEST {'PASSED' if ok else 'FAILED'} ===")
    print(f"artifacts under {sd}/")
    return 0 if ok else 1


# =================================================================== CLI
def build_argparser():
    ap = argparse.ArgumentParser(description="Edit half-life HL0 kill-gate")
    ap.add_argument("--stage", choices=["s1", "s2", "s3", "s4", "s5"],
                    help="s1 tau | s2 edits | s3 proxy | s4 real-FT fidelity | s5 analysis")
    ap.add_argument("--selftest", action="store_true", help="CPU synthetic self-test (no GPU)")
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--data", default=DATA_DEFAULT)
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_edits", type=int, default=200)
    ap.add_argument("--n_controls", type=int, default=100)
    ap.add_argument("--steps", type=int, default=20, help="ROME value-opt steps")
    ap.add_argument("--lr", type=float, default=0.1, help="ROME value-opt lr")
    ap.add_argument("--n_ft", type=int, default=32, help="# gsm8k examples for the tau FT")
    ap.add_argument("--ft_steps", type=int, default=5, help="short-FT epochs over the gsm8k batch")
    ap.add_argument("--ft_lr", type=float, default=1e-4, help="short-FT Adam lr")
    ap.add_argument("--n_real", type=int, default=40, help="# real-FT edits for s4 fidelity")
    ap.add_argument("--n_perm", type=int, default=2000, help="s5 permutation-null count")
    ap.add_argument("--out", default=os.path.join(OUTDIR, "HL0_killgate_table.json"))
    return ap


def main():
    args = build_argparser().parse_args()
    if args.selftest:
        raise SystemExit(run_selftest(args))
    if not args.stage:
        raise SystemExit("specify --stage {s1,s2,s3,s4,s5} or --selftest")
    {"s1": run_s1, "s2": run_s2, "s3": run_s3, "s4": run_s4, "s5": run_s5}[args.stage](args)


if __name__ == "__main__":
    main()
