#!/usr/bin/env python3
"""
analysis_deep.py — deep panel analysis for the P2 pre-RL overthinking diagnostic.

CPU + numpy + stdlib ONLY.  Imports (never edits) diagnostic.py for the D
estimators, the cluster bootstrap, and the rank/Spearman helpers.  Reads the
already-computed per-checkpoint records in results/*.json (does NOT recompute
D_pooled / D_within that already exist there) and the raw samples/*.json (for
stratification, k-sensitivity, and the robust log-mean estimator).

Writes results/PANEL_deep_analysis.json (atomic os.replace) and prints a
human-readable summary.

Sections (mirror the task):
  (a) Panel table pulled from results/*.json.
  (b) Difficulty stratification of D_within (per pass-rate band).
  (c) k-sensitivity: half-budget (k=4 of 8) resampling of D_within.
  (d) Robust within-problem log-mean length-difference estimator delta.
  (e) Cross-checkpoint power analysis (Monte Carlo, exact one-sided perm p).
  (f) Usability flag per checkpoint.

Deterministic: every RNG is explicitly seeded.
"""
from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Sequence

import numpy as np

import diagnostic as diag  # sibling module; imported, never edited

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
SAMPLES_DIR = os.path.join(HERE, "samples")
OUT_PATH = os.path.join(RESULTS_DIR, "PANEL_deep_analysis.json")

# Canonical panel order (matches grpo_config.CHECKPOINT_PANEL by family/size).
PANEL: List[str] = [
    "Qwen2.5-0.5B",
    "Qwen2.5-1.5B",
    "Qwen2.5-3B",
    "Llama-3.2-1B",
    "Llama-3.2-3B",
    "gemma-2-2b",
    "Phi-3.5-mini",
]

# Usability rule (stated verbatim in the output JSON).
USABILITY_RULE = (
    "A checkpoint is UNUSABLE for the cross-checkpoint correlation if "
    "n_right < 20 (too few correct traces to control difficulty) OR the 95% "
    "cluster-bootstrap CI width on the primary effect size D_within exceeds "
    "1.5 (estimate too imprecise to rank the panel). Otherwise USABLE."
)
USABILITY_MIN_NRIGHT = 20
USABILITY_MAX_CIW = 1.5

# Global seeds.
SEED_STRATA = 12345      # base seed for stratum bootstraps
SEED_KSENS = 20260710    # k-sensitivity resampling
SEED_ROBUST = 777        # robust-estimator bootstrap
SEED_POWER = 999         # power Monte Carlo

N_BOOT = 2000
N_KSENS_DRAWS = 300      # >= 200 required
N_POWER_SIMS = 8000      # >= 5000 required


# --------------------------------------------------------------------------- #
# small IO helpers
# --------------------------------------------------------------------------- #

def _load_result(ckpt: str) -> Dict[str, Any]:
    with open(os.path.join(RESULTS_DIR, f"{ckpt}.json")) as fh:
        return json.load(fh)


def _load_problems(ckpt: str) -> List[Dict[str, Any]]:
    return diag.load_samples(os.path.join(SAMPLES_DIR, f"{ckpt}.json"))


def _ci_width(d: Dict[str, Any]) -> float:
    return float(d["ci_hi"] - d["ci_lo"])


def _finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


# --------------------------------------------------------------------------- #
# (a) panel table + (f) usability flag
# --------------------------------------------------------------------------- #

def build_panel() -> Dict[str, Any]:
    panel: Dict[str, Any] = {}
    for ckpt in PANEL:
        r = _load_result(ckpt)
        c = r["counts"]
        dp, dw = r["D_pooled"], r["D_within"]
        n_right = int(c["n_right"])
        ciw = _ci_width(dw)
        reasons: List[str] = []
        if n_right < USABILITY_MIN_NRIGHT:
            reasons.append(f"n_right={n_right} < {USABILITY_MIN_NRIGHT}")
        if ciw > USABILITY_MAX_CIW:
            reasons.append(f"D_within_CI_width={ciw:.3f} > {USABILITY_MAX_CIW}")
        panel[ckpt] = {
            "n_right": n_right,
            "n_wrong": int(c["n_wrong"]),
            "n_mixed_problems": int(c["n_mixed_problems"]),
            "mean_len_right": r.get("mean_len_right"),
            "mean_len_wrong": r.get("mean_len_wrong"),
            "D_pooled": {
                "point": dp["point"], "ci_lo": dp["ci_lo"],
                "ci_hi": dp["ci_hi"], "boot_se": dp["boot_se"],
            },
            "D_within": {
                "point": dw["point"], "ci_lo": dw["ci_lo"],
                "ci_hi": dw["ci_hi"], "boot_se": dw["boot_se"],
                "ci_width": ciw,
            },
            "usable": len(reasons) == 0,
            "unusable_reasons": reasons,
        }
    return panel


# --------------------------------------------------------------------------- #
# (b) difficulty stratification
# --------------------------------------------------------------------------- #

_STRATA = [
    ("(0,0.25]", lambda pr: 0.0 < pr <= 0.25),
    ("(0.25,0.5]", lambda pr: 0.25 < pr <= 0.5),
    ("(0.5,1)", lambda pr: 0.5 < pr < 1.0),
]


def _pass_rate(problem: Dict[str, Any]) -> float:
    samples = problem.get("samples", [])
    if not samples:
        return float("nan")
    n_ok = sum(1 for s in samples if diag._is_correct(s))
    return n_ok / len(samples)


def stratify(ckpt: str, problems: Sequence[Dict[str, Any]], seed: int) -> Dict[str, Any]:
    # Only MIXED problems contribute to D_within, so stratify those.
    mixed = []
    for p in problems:
        r, w = diag._problem_lengths(p)
        if r and w:
            mixed.append((p, _pass_rate(p)))
    out: Dict[str, Any] = {"n_mixed_total": len(mixed), "strata": {}}
    for label, pred in _STRATA:
        bucket = [p for (p, pr) in mixed if pred(pr)]
        if bucket:
            ci = diag.bootstrap_ci(bucket, diag.d_within, n_boot=N_BOOT,
                                   seed=seed)
            out["strata"][label] = {
                "n_problems": len(bucket),
                "d_within": ci["point"],
                "ci_lo": ci["ci_lo"],
                "ci_hi": ci["ci_hi"],
                "boot_se": ci["boot_se"],
                "frac_finite": ci["frac_finite"],
                # <5 problems: bootstrap CI is degenerate/unreliable, do not interpret
                "small_stratum_warn": len(bucket) < 5,
            }
        else:
            out["strata"][label] = {"n_problems": 0, "d_within": None,
                                    "note": "empty stratum"}
    return out


# --------------------------------------------------------------------------- #
# (c) k-sensitivity (half budget: k=4 of 8)
# --------------------------------------------------------------------------- #

def _subsample_problem(problem: Dict[str, Any], rng: np.random.Generator,
                       k: int) -> Dict[str, Any]:
    s = problem.get("samples", [])
    if len(s) <= k:
        return problem
    idx = rng.choice(len(s), size=k, replace=False)
    return {"problem": problem.get("problem"), "samples": [s[i] for i in idx]}


def k_sensitivity(problems: Sequence[Dict[str, Any]], seed: int,
                  k: int = 4, n_draws: int = N_KSENS_DRAWS) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    full = diag.d_within(problems)
    vals: List[float] = []
    for _ in range(n_draws):
        sub = [_subsample_problem(p, rng, k) for p in problems]
        v = diag.d_within(sub)
        if math.isfinite(v):
            vals.append(v)
    arr = np.asarray(vals, dtype=float)
    return {
        "k": k,
        "n_draws": n_draws,
        "d_within_full_k8": float(full) if math.isfinite(full) else None,
        "mean_k4": float(arr.mean()) if arr.size else None,
        "sd_k4": float(arr.std(ddof=1)) if arr.size > 1 else None,
        "min_k4": float(arr.min()) if arr.size else None,
        "max_k4": float(arr.max()) if arr.size else None,
        "frac_finite": arr.size / n_draws if n_draws else float("nan"),
    }


# --------------------------------------------------------------------------- #
# (d) robust within-problem log-mean length-difference estimator
# --------------------------------------------------------------------------- #

def delta_logmean(problems: Sequence[Dict[str, Any]]) -> float:
    """Mean over mixed problems of [mean(log wrong_len) - mean(log right_len)].

    exp(delta) is a geometric-mean analog of D_within (both ~ wrong/right length
    ratio) but far less sensitive to a single long outlier trace."""
    diffs: List[float] = []
    for p in problems:
        r, w = diag._problem_lengths(p)
        r = [x for x in r if x > 0]
        w = [x for x in w if x > 0]
        if r and w:
            diffs.append(float(np.mean(np.log(w)) - np.mean(np.log(r))))
    if not diffs:
        return float("nan")
    return float(np.mean(diffs))


def robust_estimator(ckpt: str, problems: Sequence[Dict[str, Any]],
                     d_within_point: float, seed: int) -> Dict[str, Any]:
    ci = diag.bootstrap_ci(problems, delta_logmean, n_boot=N_BOOT, seed=seed)
    d, lo, hi = ci["point"], ci["ci_lo"], ci["ci_hi"]
    return {
        "delta_point": d,
        "delta_ci_lo": lo,
        "delta_ci_hi": hi,
        "boot_se": ci["boot_se"],
        "frac_finite": ci["frac_finite"],
        "exp_delta": float(math.exp(d)) if _finite(d) else None,
        "exp_ci_lo": float(math.exp(lo)) if _finite(lo) else None,
        "exp_ci_hi": float(math.exp(hi)) if _finite(hi) else None,
        "d_within_point": float(d_within_point) if _finite(d_within_point) else None,
    }


# --------------------------------------------------------------------------- #
# (e) cross-checkpoint power analysis
# --------------------------------------------------------------------------- #

# 2026-07-11: the enumeration machinery moved to diagnostic.py as PUBLIC functions
# (exact_null_spearman_dist / one_sided_p_from_null / critical_rho_from_null) so the
# prereg §5 primary test (compute_overthinking_gap.py) and this power analysis share
# ONE implementation.  These aliases keep every call site and the prereg's
# "as in analysis_deep._one_sided_p" pointer valid; behavior is verified identical
# (sha256 of the null vectors, n=3..7).
_null_spearman_dist = diag.exact_null_spearman_dist
_one_sided_p = diag.one_sided_p_from_null
_critical_rho = diag.critical_rho_from_null


def power_analysis(panel: Dict[str, Any]) -> Dict[str, Any]:
    """Plug-in Monte Carlo power for the planned cross-checkpoint Spearman test.

    Assumptions (stated in output):
      * The panel D_within point estimates are treated as the TRUE latent pre-RL
        D of each checkpoint (plug-in).
      * The measured pre-RL D is noisy: observed_D_i ~ N(point_i, boot_se_i).
      * The post-GRPO overthinking gap correlates with the TRUE latent D at a
        target Pearson rho on standardized scores:
            post_i = rho * z(trueD_i) + sqrt(1-rho^2) * eps_i,  eps~N(0,1).
      * Test = one-sided (positive) Spearman(observed_D, post) with EXACT
        permutation p (all n! label permutations). Power = P(p < 0.05).
    """
    rng = np.random.default_rng(SEED_POWER)
    target_rhos = [0.5, 0.7, 0.9]

    # candidate panels
    all7 = list(PANEL)
    n6 = [c for c in PANEL if c != "Llama-3.2-1B"]
    usable = [c for c in PANEL if panel[c]["usable"]]

    def _vecs(ids: List[str]):
        pts = np.array([panel[c]["D_within"]["point"] for c in ids], float)
        ses = np.array([panel[c]["D_within"]["boot_se"] for c in ids], float)
        return pts, ses

    def _power_for(ids: List[str]) -> Dict[str, Any]:
        n = len(ids)
        if n < 3:
            return {"n": n, "note": "n<3: not testable", "power": {}}
        pts, ses = _vecs(ids)
        z = (pts - pts.mean()) / (pts.std() if pts.std() > 0 else 1.0)
        null = _null_spearman_dist(n)
        crit = _critical_rho(null)
        res: Dict[str, Any] = {
            "n": n,
            "ids": ids,
            "min_attainable_p": 1.0 / math.factorial(n),
            "critical_rho_p05_onesided": crit,
            "power": {},
        }
        for tr in target_rhos:
            hits = 0
            for _ in range(N_POWER_SIMS):
                obs = rng.normal(pts, ses)
                eps = rng.normal(0.0, 1.0, size=n)
                post = tr * z + math.sqrt(max(0.0, 1.0 - tr * tr)) * eps
                rho = diag.spearman(obs, post)
                if math.isfinite(rho) and _one_sided_p(rho, null) < 0.05:
                    hits += 1
            res["power"][f"rho={tr}"] = hits / N_POWER_SIMS
        return res

    return {
        "n_sims": N_POWER_SIMS,
        "target_rhos": target_rhos,
        "assumptions": (
            "Plug-in: panel D_within points = true latent D; observed ~ "
            "N(point, boot_se); post = rho*z(trueD)+sqrt(1-rho^2)*N(0,1); "
            "one-sided exact-permutation Spearman(observed, post); power=P(p<0.05)."
        ),
        "n7_all": _power_for(all7),
        "n6_excl_Llama1B": _power_for(n6),
        "n_usable_only": _power_for(usable),
        "usable_ids": usable,
    }


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #

def main() -> None:
    panel = build_panel()

    strat: Dict[str, Any] = {}
    ksens: Dict[str, Any] = {}
    robust: Dict[str, Any] = {}
    for i, ckpt in enumerate(PANEL):
        problems = _load_problems(ckpt)
        strat[ckpt] = stratify(ckpt, problems, seed=SEED_STRATA + i)
        ksens[ckpt] = k_sensitivity(problems, seed=SEED_KSENS + i)
        robust[ckpt] = robust_estimator(
            ckpt, problems, panel[ckpt]["D_within"]["point"],
            seed=SEED_ROBUST + i,
        )

    power = power_analysis(panel)

    usable = [c for c in PANEL if panel[c]["usable"]]
    out = {
        "_meta": {
            "tool": "analysis_deep.py",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "note": "CPU deep analysis; imports diagnostic.py, reads results/ + "
                    "samples/. No GPU / model loading.",
            "seeds": {
                "strata": SEED_STRATA, "ksens": SEED_KSENS,
                "robust": SEED_ROBUST, "power": SEED_POWER,
            },
            "n_boot": N_BOOT,
        },
        "usability_rule": USABILITY_RULE,
        "usable_checkpoints": usable,
        "panel": panel,                      # (a)+(f)
        "difficulty_stratification": strat,  # (b)
        "k_sensitivity": ksens,              # (c)
        "robust_logmean": robust,            # (d)
        "power_analysis": power,             # (e)
    }

    # sanity: every numeric leaf must be finite-or-null (no NaN/Inf in JSON)
    _assert_clean(out)

    tmp = OUT_PATH + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(out, fh, indent=2, allow_nan=False)
    os.replace(tmp, OUT_PATH)

    _print_summary(out)


def _assert_clean(obj: Any, path: str = "root") -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            _assert_clean(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for j, v in enumerate(obj):
            _assert_clean(v, f"{path}[{j}]")
    elif isinstance(obj, float):
        if not math.isfinite(obj):
            raise ValueError(f"non-finite float at {path}: {obj!r}")


def _fmt(x: Any, nd: int = 3) -> str:
    if x is None:
        return "  -  "
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def _print_summary(out: Dict[str, Any]) -> None:
    p = out["panel"]
    print("\n" + "=" * 78)
    print("P2 DEEP PANEL ANALYSIS")
    print("=" * 78)
    print("\n(a)+(f) PANEL TABLE  [rule: " + out["usability_rule"] + "]")
    print(f"{'checkpoint':16s} {'n_r':>4s} {'n_mix':>5s} {'D_pool':>7s} "
          f"{'D_within':>8s} {'CIw':>6s}  usable")
    for c in PANEL:
        r = p[c]
        print(f"{c:16s} {r['n_right']:>4d} {r['n_mixed_problems']:>5d} "
              f"{_fmt(r['D_pooled']['point']):>7s} "
              f"{_fmt(r['D_within']['point']):>8s} "
              f"{_fmt(r['D_within']['ci_width'],2):>6s}  "
              f"{'YES' if r['usable'] else 'no: ' + '; '.join(r['unusable_reasons'])}")
    print(f"\n  usable checkpoints ({len(out['usable_checkpoints'])}): "
          f"{out['usable_checkpoints']}")

    print("\n(b) DIFFICULTY STRATIFICATION of D_within (mixed problems only)")
    for c in PANEL:
        s = out["difficulty_stratification"][c]
        cells = []
        for lab, st in s["strata"].items():
            if st["n_problems"]:
                cells.append(f"{lab}: D={_fmt(st['d_within'])} (n={st['n_problems']})")
            else:
                cells.append(f"{lab}: -")
        print(f"  {c:16s} " + " | ".join(cells))

    print("\n(c) k-SENSITIVITY  (k=4 of 8; mean +/- sd of D_within over draws)")
    for c in PANEL:
        k = out["k_sensitivity"][c]
        print(f"  {c:16s} full_k8={_fmt(k['d_within_full_k8'])}  "
              f"k4_mean={_fmt(k['mean_k4'])}  k4_sd={_fmt(k['sd_k4'])}  "
              f"[{_fmt(k['min_k4'])},{_fmt(k['max_k4'])}]")

    print("\n(d) ROBUST log-mean estimator  (exp(delta) ~ D_within, outlier-robust)")
    for c in PANEL:
        rb = out["robust_logmean"][c]
        print(f"  {c:16s} exp(delta)={_fmt(rb['exp_delta'])} "
              f"[{_fmt(rb['exp_ci_lo'])},{_fmt(rb['exp_ci_hi'])}]  "
              f"vs D_within={_fmt(rb['d_within_point'])}")

    print("\n(e) POWER ANALYSIS  (Monte Carlo, exact one-sided perm p<0.05)")
    pw = out["power_analysis"]
    for key, lab in [("n7_all", "n=7 (all)"),
                     ("n6_excl_Llama1B", "n=6 (excl Llama-3.2-1B)"),
                     ("n_usable_only", f"n={len(pw['usable_ids'])} (usable only)")]:
        blk = pw[key]
        if "power" in blk and blk["power"]:
            crit = blk.get("critical_rho_p05_onesided")
            pstr = "  ".join(f"{k}:{_fmt(v,2)}" for k, v in blk["power"].items())
            print(f"  {lab:26s} crit_rho={_fmt(crit)}  min_p={_fmt(blk['min_attainable_p'],4)}"
                  f"  power[{pstr}]")
        else:
            print(f"  {lab:26s} {blk.get('note','n/a')}")

    print("\nwrote", OUT_PATH)
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
