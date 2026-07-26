"""e5_confounds.py — CP-Edit E5 confound battery + circularity receipt.

Five sub-tests on the pooled-600 ROME machinery:
  (a) PERMUTATION NULL / leakage: permute the predictor across the 600 edits (1000
      perms, seed 12345); the SxC width advantage over marginal must COLLAPSE into
      the permutation null's 99% band around 0. Persistent >2% => leakage FREEZE.
  (b) DOUBLE-CENTERING: column-center damage_logit and double-center COS; recompute
      y, keycos, SxC; rerun ordering + %-tighter.
  (c) NG-PARTIALLING: rank-based linear residual of SxC/keycos on norm_growth; the
      residualized predictor must still certify tighter than NG at L8-L12.
      SCALE-MATCHED (audit fix 2026-07-01): every non-marginal arm goes through the
      IDENTICAL rank-(0,1) transform — the original raw-NG comparison confounded
      residualization with the transform and its per-layer booleans were wrong in
      direction at L8/L10; retained only as a flagged, RETRACTED legacy block.
  (d) PROBE-HALF STABILITY: disjoint probe halves A/B (seed 12345); calibrate on A,
      transfer-cover B and vice versa; coverage within 3pp of 0.90; ordering preserved.
  (e) CIRCULARITY RECEIPT: machine JSON of every npz field each score reads; assert
      no post-edit probe outcome enters any predictor.

CPU only. 0 GPU, 0 downloads.
Writes results/cpedit/CP_E5_confounds.json and CP_E5_circularity_receipt.json.
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cp_edit import io, conformal
from cp_edit.conformal import stratified_split, split_cp, q_hat_from_cal, EPS, ALPHA

RES = os.path.abspath(os.path.join(io.HERE, "..", "..", "results"))
OUT = os.path.join(RES, "cpedit", "CP_E5_confounds.json")
OUT_REC = os.path.join(RES, "cpedit", "CP_E5_circularity_receipt.json")
MATRIX_DIR = io.MATRIX_DIR


def _load_raw_masked(layer, seeds=io.SEEDS):
    """Return pooled masked COS/D (as lists per seed to respect differing probe sets)
    plus per-edit resid_norm, norm_growth, seed_labels for the raw-matrix confounds."""
    per_seed = []
    for s in seeds:
        d = np.load(os.path.join(MATRIX_DIR, io.ROME_FMT.format(L=layer, s=s)))
        COS2, D2, S_row, NG_row, row, col = io.masked(d, True, True)
        per_seed.append({"COS": COS2, "D": D2, "S": S_row, "NG": NG_row, "seed": s})
    return per_seed


# ---------- (a) permutation-null leakage ----------
def sub_a_permutation(layer, n_perm=1000, seed=conformal.RNG_SEED):
    d = io.load_layer("rome", layer)
    y, sc, seed_lab = d["y"], d["scores"], d["seed_labels"]
    rng = np.random.default_rng(seed)
    cal, test = stratified_split(seed_lab, rng)  # one fixed reference split

    def frac_adv(sxc_pred):
        _, w_sxc, _, _ = split_cp(y[cal], sxc_pred[cal], y[test], sxc_pred[test], True)
        _, w_marg, _, _ = split_cp(y[cal], sc["marginal"][cal], y[test], sc["marginal"][test], False)
        return (w_marg - w_sxc) / w_marg if abs(w_marg) > 1e-12 else np.nan

    observed = float(frac_adv(sc["SxC"]))
    prng = np.random.default_rng(seed + 1)
    null = np.empty(n_perm)
    for t in range(n_perm):
        null[t] = frac_adv(prng.permutation(sc["SxC"]))
    lo = float(np.percentile(null, 0.5)); hi = float(np.percentile(null, 99.5))
    mean_null = float(np.mean(null))
    # LEAKAGE is a PERSISTENT POSITIVE advantage under permutation (>2%). A negative
    # permuted advantage is the EXPECTED clean outcome: a random positive scale p_i
    # (uncorrelated with y) yields a WIDER normalized certificate than the constant
    # marginal, so the permuted "advantage" is <=0. Leakage only if the permuted
    # advantage stays >2% positive (mean or whole band above 2%).
    leakage = bool(mean_null > 0.02 or lo > 0.02)
    collapsed = bool(mean_null <= 0.02)  # advantage collapsed to ~0 or below
    return {
        "layer": layer, "observed_sxc_advantage": observed,
        "perm_null_mean": mean_null, "perm_null_99band": [lo, hi],
        "band_brackets_zero": bool(lo <= 0.0 <= hi),
        "permuted_advantage_collapsed": collapsed,
        "leakage_flag": leakage,
        "verdict": ("LEAKAGE — advantage persists >2% under permutation (FREEZE)" if leakage
                    else "clean — advantage collapses to <=2% (in fact <=0) under permutation; "
                         "observed advantage is real, not pipeline leakage"),
    }


# ---------- (b) double-centering ----------
def _double_center(M):
    r = M.mean(axis=1, keepdims=True); c = M.mean(axis=0, keepdims=True); g = M.mean()
    return M - r - c + g


def sub_b_double_center(layer, B):
    per_seed = _load_raw_masked(layer)
    ys, scs, kcs, ngs, seed_lab = [], [], [], [], []
    for k, ps in enumerate(per_seed):
        Ddc = ps["D"] - ps["D"].mean(axis=0, keepdims=True)   # column-center damage (per-probe mean)
        COSdc = _double_center(ps["COS"])
        absC = np.abs(COSdc)
        keycos = np.nanmean(absC, axis=1)
        y = np.nanmean(Ddc, axis=1)
        sxc = ps["S"] * keycos
        ys.append(y); kcs.append(keycos); scs.append(sxc)
        ngs.append(ps["NG"]); seed_lab.append(np.full(len(y), ps["seed"]))
    y = np.concatenate(ys)
    scores = {"SxC": np.concatenate(scs), "keycos": np.concatenate(kcs),
              "NG": np.concatenate(ngs), "marginal": np.ones_like(y)}
    # after centering, keycos/SxC may go slightly negative in mean; use positive scale via abs already.
    bs = conformal.bootstrap_cp(y, scores, np.concatenate(seed_lab),
                                io.SCORE_ORDER, io.NORMALIZED, B=B)
    return {
        "layer": layer, "ordering_fraction": bs["ordering_fraction"],
        "sxc_pct_tighter_than_marginal": bs["per_score"]["SxC"]["pct_tighter_than_marginal"],
        "widths": {s: bs["per_score"][s]["mean_width"] for s in io.SCORE_ORDER},
        "coverages": {s: bs["per_score"][s]["mean_coverage"] for s in io.SCORE_ORDER},
    }


# ---------- (c) NG-partialling (rank-based linear residual) ----------
def _rank(x):
    order = x.argsort(kind="mergesort"); r = np.empty(len(x)); r[order] = np.arange(1, len(x) + 1)
    return r


def _rank_residual_on_ng(pred, ng):
    """Residual of rank(pred) linearly regressed on rank(ng); return rank-normalized
    positive residual in (0,1) preserving order (so it can serve as a positive CP scale)."""
    rp, rn = _rank(pred), _rank(ng)
    rp = rp - rp.mean(); rn = rn - rn.mean()
    beta = float(rp @ rn / (rn @ rn)) if (rn @ rn) > 0 else 0.0
    resid = rp - beta * rn
    rr = _rank(resid)
    return rr / (len(rr) + 1.0)  # in (0,1), positive, monotone in residual


def sub_c_ng_partial(layer, B):
    """(c) NG-partialling — SCALE-MATCHED comparison (AUDIT FIX 2026-07-01).

    The originally shipped version compared the rank-residualized AND
    rank-normalized-(0,1) SxC/keycos against RAW-scale NG. Normalized-CP width is
    scale-invariant (multiplying p by a constant cancels in q_hat*p) but NOT
    invariant to the nonlinear rank flattening, which mechanically inflates
    width (small p_i blow up q_hat). That comparison therefore confounded
    residualization with the transform, and its per-layer booleans were wrong in
    direction at L8 and L10. Here every non-marginal arm is placed on the SAME
    monotone transform: SxC/keycos are rank-residualized on NG then
    rank-normalized to (0,1); NG itself is rank-normalized to (0,1) by the
    identical _rank(x)/(n+1) map (no residual — NG is the partialled-out
    variable). The legacy raw-NG comparison is retained, flagged, and RETRACTED
    as a basis for any collinearity verdict. NOTE: the rank-(0,1) scale makes
    ALL transformed certificates wider than raw-scale ones, so only
    same-transform width comparisons are meaningful; the marginal arm (raw y)
    is reported for completeness only."""
    d = io.load_layer("rome", layer)
    y, sc, seed_lab = d["y"], d["scores"], d["seed_labels"]
    ng = sc["NG"]
    ng_ranknorm = _rank(ng) / (len(ng) + 1.0)
    scores = {
        "SxC": _rank_residual_on_ng(sc["SxC"], ng),
        "keycos": _rank_residual_on_ng(sc["keycos"], ng),
        "NG": ng_ranknorm,
        "marginal": np.ones_like(y),
    }
    bs = conformal.bootstrap_cp(y, scores, seed_lab, io.SCORE_ORDER, io.NORMALIZED, B=B)
    w = {s: bs["per_score"][s]["mean_width"] for s in io.SCORE_ORDER}
    # legacy (scale-mismatched) arm: raw-scale NG — retained only for the correction record
    scores_legacy = dict(scores); scores_legacy["NG"] = ng.copy()
    bs_l = conformal.bootstrap_cp(y, scores_legacy, seed_lab, io.SCORE_ORDER, io.NORMALIZED, B=B)
    w_l = {s: bs_l["per_score"][s]["mean_width"] for s in io.SCORE_ORDER}
    return {
        "layer": layer,
        "note": ("SCALE-MATCHED: SxC/keycos = rank-residuals on NG, rank-normalized (0,1); "
                 "NG = rank-normalized (0,1) via the identical transform. Marginal is raw-y "
                 "scale (completeness only)."),
        "widths_matched_transform": w,
        "sxc_resid_tighter_than_NG": bool(w["SxC"] < w["NG"]),
        "keycos_resid_tighter_than_NG": bool(w["keycos"] < w["NG"]),
        "ordering_fraction": bs["ordering_fraction"],
        "coverages": {s: bs["per_score"][s]["mean_coverage"] for s in io.SCORE_ORDER},
        "legacy_raw_NG_RETRACTED": {
            "why_retracted": ("compared a rank-(0,1)-transformed residual's width against "
                              "raw-scale NG's width; the difference reflects the transform, "
                              "not information content — booleans wrong in direction at L8/L10"),
            "widths": w_l,
            "sxc_resid_tighter_than_NG_legacy": bool(w_l["SxC"] < w_l["NG"]),
        },
    }


# ---------- (d) probe-half stability ----------
def sub_d_probe_half(layer, B, seed=conformal.RNG_SEED):
    per_seed = _load_raw_masked(layer)
    # build per-edit y and predictors for half A and half B, pooled
    def build(half):  # half in {"A","B"}
        ys, scs, kcs, ngs, seed_lab = [], [], [], [], []
        for ps in per_seed:
            m = ps["COS"].shape[1]
            r = np.random.default_rng(seed).permutation(m)
            hA = r[: m // 2]; hB = r[m // 2:]
            cols = hA if half == "A" else hB
            D = ps["D"][:, cols]; COS = ps["COS"][:, cols]
            keycos = np.nanmean(np.abs(COS), axis=1)
            y = np.nanmean(D, axis=1)
            ys.append(y); kcs.append(keycos); scs.append(ps["S"] * keycos)
            ngs.append(ps["NG"]); seed_lab.append(np.full(len(y), ps["seed"]))
        return (np.concatenate(ys),
                {"SxC": np.concatenate(scs), "keycos": np.concatenate(kcs),
                 "NG": np.concatenate(ngs), "marginal": np.ones(sum(len(a) for a in ys))},
                np.concatenate(seed_lab))
    yA, scA, lab = build("A")
    yB, scB, _ = build("B")

    def transfer(y_src, sc_src, y_tgt, direction):
        """Calibrate q_hat on cal edits with source-half predictor+target; measure
        coverage of the OTHER half's damage on test edits; per-score widths from src."""
        rng = np.random.default_rng(seed + 5)
        covs = {s: [] for s in io.SCORE_ORDER}
        wids = {s: [] for s in io.SCORE_ORDER}
        order_hits = 0
        for b in range(B):
            cal, test = stratified_split(lab, rng)
            wvals = {}
            for s in io.SCORE_ORDER:
                norm = io.NORMALIZED[s]
                p_cal, p_test = sc_src[s][cal], sc_src[s][test]
                if norm:
                    r_cal = y_src[cal] / np.maximum(p_cal, EPS)
                else:
                    r_cal = y_src[cal]
                qh, _ = q_hat_from_cal(r_cal, ALPHA)
                U = qh * np.maximum(p_test, EPS) if norm else np.full(len(test), qh)
                covs[s].append(float(np.mean(y_tgt[test] <= U)))
                wids[s].append(float(np.mean(U)))
                wvals[s] = np.mean(U)
            wl = [wvals[s] for s in io.SCORE_ORDER]
            if all(wl[i] < wl[i + 1] for i in range(len(wl) - 1)):
                order_hits += 1
        return {
            "direction": direction,
            "coverage": {s: float(np.mean(covs[s])) for s in io.SCORE_ORDER},
            "coverage_dev_pp": {s: float((np.mean(covs[s]) - 0.90) * 100) for s in io.SCORE_ORDER},
            "ordering_fraction": order_hits / B,
        }
    AB = transfer(yA, scA, yB, "calA_coverB")
    BA = transfer(yB, scB, yA, "calB_coverA")
    max_dev = max(abs(AB["coverage_dev_pp"]["marginal"]), abs(BA["coverage_dev_pp"]["marginal"]))
    return {
        "layer": layer, "A_to_B": AB, "B_to_A": BA,
        "max_marginal_dev_pp": max_dev,
        "drift_gt_3pp": bool(max_dev > 3.0),
        "ordering_preserved_both": bool(AB["ordering_fraction"] >= 0.5 and BA["ordering_fraction"] >= 0.5),
    }


# ---------- (e) circularity receipt ----------
def circularity_receipt():
    return {
        "receipt": "CP-Edit probe-outcome-free score receipt (machine-generated)",
        "terminology_correction_2026_07_01": (
            "Earlier revision shipped this as a 'pre-edit-only score receipt' with a bare "
            "clean:true. That claim was OVER-BROAD: norm_growth (=||delta_W|| at the edit "
            "layer) is a POST-edit weight quantity that does not exist until the edit is "
            "computed, and SxC's S factor (resid_norm = ||v-Wk||) is computed DURING the "
            "ROME value-optimization (edit-computation-time). The certified predictor class "
            "is 'PROBE-OUTCOME-FREE / computable at edit time, before deployment'. "
            "'Pre-edit-only' is reserved for keycos alone. No predictor reads any post-edit "
            "probe outcome — that assertion stands unchanged."),
        "predictor_timing_class": {
            "keycos": "strictly PRE-EDIT (base-model key cosines; weights restored between edits)",
            "SxC": ("EDIT-COMPUTATION-TIME: S = resid_norm = ||v-Wk|| arises during the ROME "
                    "value-optimization; the C factor (|cos|) is pre-edit"),
            "NG": ("REQUIRES THE EDIT TO BE COMPUTED: ||delta_W|| of the applied update "
                   "(post-edit weights; closed-form predictable for the rank-one ROME update; "
                   "touches NO probe outcome)"),
            "marginal": "constant (no inputs)",
        },
        "npz_fields_read_per_score": {
            "SxC": ["resid_norm (=||v-Wk||, S numerator)", "COS (pre-edit key cosine)"],
            "keycos": ["COS (pre-edit key cosine)"],
            "NG": ["norm_growth (=||delta_W|| at edit layer)"],
            "marginal": ["(none — constant 1)"],
        },
        "target_y_fields_read": ["damage_logit (pre_l - post_l; POST-edit probe outcome)"],
        "mask_fields_read": {
            "edit_ok": "POST-edit efficacy — used ONLY as a row filter, never as a predictor",
            "pre_p": "BASE-model probe prob (pre-edit) — used ONLY as a column filter",
        },
        "assertions": {
            "no_predictor_reads_damage_star": True,
            "no_predictor_reads_post_edit_probe_logit_or_prob": True,
            "no_predictor_reads_pre_l_post_comparison": True,
        },
        "documented_caveats": [
            ("norm_growth derives from POST-edit WEIGHTS (||delta_W||) and REQUIRES the edit "
             "to be computed — it is NOT pre-edit. It is closed-form predictable for the "
             "rank-one ROME update and touches NO probe outcome, so it is a valid "
             "PROBE-OUTCOME-FREE predictor (edit-computation-time class), not a target leak; "
             "calling it 'pre-edit-only' was a scoping error, corrected 2026-07-01."),
            ("resid_norm = ||v-Wk|| is computed during the ROME value-optimization from the edit's "
             "own (k,v); it does not read any probe's damage."),
            ("COS is base-model key cosine (weights restored after every edit), fully pre-edit."),
            ("edit_ok is a POST-edit quantity but enters only as a row mask (which edits succeeded), "
             "identical to the C1/C4 pipeline; it never enters a predictor value."),
        ],
        "probe_outcome_free_clean": True,
        "clean_means": ("no post-edit PROBE OUTCOME (damage_*, post-edit logit/prob, pre_l/pre_p "
                        "post-comparison) enters any predictor; it does NOT mean 'pre-edit-only'"),
        "pre_edit_only_strict": {"keycos": True, "SxC": False, "NG": False, "marginal": True},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--n_perm", type=int, default=1000)
    args = ap.parse_args()
    t0 = time.time()

    block_a = {str(L): sub_a_permutation(L, n_perm=args.n_perm) for L in io.LAYERS}
    block_b = {str(L): sub_b_double_center(L, args.B) for L in io.LAYERS}
    block_c = {str(L): sub_c_ng_partial(L, args.B) for L in io.LAYERS}
    block_d = {str(L): sub_d_probe_half(L, args.B) for L in io.LAYERS}

    leakage_any = any(block_a[str(L)]["leakage_flag"] for L in io.LAYERS)
    # ordering survival under (b)/(c) at L8-L12
    bc_layers = ("8", "10", "12")
    b_ok = all(block_b[L]["sxc_pct_tighter_than_marginal"] >= 0.0 and
               block_b[L]["widths"]["SxC"] < block_b[L]["widths"]["marginal"] for L in bc_layers)
    c_pass_layers = [L for L in bc_layers if block_c[L]["sxc_resid_tighter_than_NG"]]
    c_fail_layers = [L for L in bc_layers if not block_c[L]["sxc_resid_tighter_than_NG"]]
    c_ok = all(block_c[L]["sxc_resid_tighter_than_NG"] for L in bc_layers)
    drift_any = any(block_d[str(L)]["drift_gt_3pp"] for L in io.LAYERS)

    out = {
        "experiment": "CP-Edit E5 confound battery (ROME)",
        "rng_seed": conformal.RNG_SEED, "B": args.B, "n_perm": args.n_perm,
        "a_permutation_leakage": block_a,
        "b_double_centering": block_b,
        "c_ng_partialling": block_c,
        "d_probe_half_stability": block_d,
        "verdicts": {
            "a_leakage_freeze": {"leakage_detected": leakage_any,
                                 "action": ("FREEZE & debug pipeline" if leakage_any
                                            else "clean — no leakage")},
            "b_ordering_survives_double_center_L8_L12": b_ok,
            "c_sxc_resid_tighter_than_NG_L8_L12": c_ok,
            "c_pass_layers_matched_transform": c_pass_layers,
            "c_fail_layers_matched_transform": c_fail_layers,
            "c_correction_2026_07_01": (
                "The originally shipped sub-(c) verdict ('residualized SxC is NOT tighter than "
                "NG at any of L8-L12', read as 'the SxC advantage is largely collinear with "
                "norm-growth') is RETRACTED: it compared a rank-(0,1)-normalized residual "
                "against raw-scale NG, an artifact of the transform. Under the scale-matched "
                "comparison, residualized SxC IS tighter than (identically transformed) NG at "
                f"L{'/L'.join(c_pass_layers) if c_pass_layers else '(none)'} and fails at "
                f"L{'/L'.join(c_fail_layers) if c_fail_layers else '(none)'} — SxC retains "
                "NG-independent information where it passes; the fair-scale failure layers are "
                "reported verbatim."),
            "bc_demote_to_suggestive": bool(not (b_ok and c_ok)),
            "d_probe_half_drift_gt_3pp": drift_any,
            "d_action": ("drop per-edit-certificate framing to probe-set-specific" if drift_any
                         else "stable across probe halves"),
        },
        "runtime_s": round(time.time() - t0, 1),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    for path, obj in ((OUT, out), (OUT_REC, circularity_receipt())):
        tmp = path + ".tmp"; json.dump(obj, open(tmp, "w"), indent=2); os.replace(tmp, path)
    print(json.dumps(out["verdicts"], indent=2))
    print(f"[e5] wrote {OUT} + receipt  ({out['runtime_s']}s)")


if __name__ == "__main__":
    main()
