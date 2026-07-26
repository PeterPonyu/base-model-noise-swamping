#!/usr/bin/env python3
"""
diagnostic.py — pre-RL "overthinking" length-bias diagnostic.

Core statistic
--------------
For one checkpoint we have, per problem, k sampled chain-of-thought traces,
each labelled as `correct` (eventually-right) or not (eventually-wrong), with a
token length `len`.  The diagnostic is:

    D = mean_len(eventually-wrong) / mean_len(eventually-right)

D > 1  => wrong traces are systematically LONGER than right traces, i.e. the
checkpoint already carries a length/verbosity bias BEFORE any RL.  The P2
hypothesis is that this cheap pre-RL D predicts the post-GRPO "overthinking
gap" across the 7-checkpoint panel (cross_checkpoint_spearman scaffold below).

Two flavours of D are reported:
  * D_pooled  — all traces pooled (simple, but confounded by problem difficulty:
                harder problems tend to be both longer AND more often wrong).
  * D_within  — difficulty-controlled: for each problem that has BOTH a right and
                a wrong trace, take mean_wrong_len/mean_right_len, then average
                the per-problem ratios.  This removes the between-problem
                difficulty confound and is the primary effect size.

Uncertainty
-----------
Cluster (problem-level) bootstrap: resample PROBLEMS with replacement, recompute
D each time, take a percentile CI.  Problem-level (not trace-level) resampling is
required because traces within a problem are correlated.

Dependencies: numpy + stdlib ONLY.  This module deliberately does NOT import
trl / unsloth / transformers / torch so it runs on CPU in the shared `dl` env
(or any env with numpy) while those RL deps are broken.  See trl_mergekit_fix.md.
"""
from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ----------------------------------------------------------------------------- #
# I/O
# ----------------------------------------------------------------------------- #

def load_samples(path: str) -> List[Dict[str, Any]]:
    """Load a checkpoint samples JSON.

    Accepted shapes:
      * {"checkpoint": "...", "problems": [ {problem, samples:[...]} , ... ]}
      * [ {problem, samples:[...]}, ... ]                (bare list of problems)
    Each problem: {"problem": <id/str>, "samples": [ {"text":?, "len":?,
    "correct": bool}, ... ]}.
    Returns the list of problem dicts.
    """
    with open(path, "r") as fh:
        obj = json.load(fh)
    if isinstance(obj, dict):
        problems = obj.get("problems", obj.get("data"))
        if problems is None:
            raise ValueError(
                f"{path}: dict payload lacks a 'problems' (or 'data') key"
            )
    elif isinstance(obj, list):
        problems = obj
    else:
        raise ValueError(f"{path}: unsupported top-level JSON type {type(obj)}")
    if not isinstance(problems, list):
        raise ValueError(f"{path}: 'problems' must be a list")
    return problems


# ----------------------------------------------------------------------------- #
# Length / correctness extraction
# ----------------------------------------------------------------------------- #

def _sample_length(sample: Dict[str, Any]) -> Optional[float]:
    """Prefer an explicit numeric `len`; else fall back to whitespace token
    count of `text`; else None (skip)."""
    v = sample.get("len")
    if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
        return float(v)
    text = sample.get("text")
    if isinstance(text, str) and text:
        return float(len(text.split()))
    return None


def _is_correct(sample: Dict[str, Any]) -> bool:
    return bool(sample.get("correct", False))


def _problem_lengths(problem: Dict[str, Any]) -> Tuple[List[float], List[float]]:
    """Return (right_lengths, wrong_lengths) for one problem, skipping samples
    with no usable length."""
    right: List[float] = []
    wrong: List[float] = []
    for s in problem.get("samples", []):
        L = _sample_length(s)
        if L is None:
            continue
        (right if _is_correct(s) else wrong).append(L)
    return right, wrong


# ----------------------------------------------------------------------------- #
# D estimators
# ----------------------------------------------------------------------------- #

def d_pooled(problems: Sequence[Dict[str, Any]]) -> float:
    """D on all traces pooled: mean_len(wrong)/mean_len(right)."""
    right: List[float] = []
    wrong: List[float] = []
    for p in problems:
        r, w = _problem_lengths(p)
        right.extend(r)
        wrong.extend(w)
    if not right or not wrong:
        return float("nan")
    return float(np.mean(wrong) / np.mean(right))


def d_within(problems: Sequence[Dict[str, Any]]) -> float:
    """Difficulty-controlled D: mean over problems (that contain BOTH a right and
    a wrong trace) of mean_wrong_len/mean_right_len.  NaN if no mixed problem."""
    ratios: List[float] = []
    for p in problems:
        r, w = _problem_lengths(p)
        if r and w:
            ratios.append(float(np.mean(w) / np.mean(r)))
    if not ratios:
        return float("nan")
    return float(np.mean(ratios))


def _count(problems: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    n_right = n_wrong = n_mixed = 0
    for p in problems:
        r, w = _problem_lengths(p)
        n_right += len(r)
        n_wrong += len(w)
        if r and w:
            n_mixed += 1
    return {
        "n_problems": len(problems),
        "n_traces": n_right + n_wrong,
        "n_right": n_right,
        "n_wrong": n_wrong,
        "n_mixed_problems": n_mixed,
    }


# ----------------------------------------------------------------------------- #
# Bootstrap CI (cluster = problem)
# ----------------------------------------------------------------------------- #

def bootstrap_ci(
    problems: Sequence[Dict[str, Any]],
    estimator=d_pooled,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> Dict[str, Any]:
    """Percentile CI for `estimator` via problem-level (cluster) resampling.

    Returns point estimate, CI bounds, bootstrap SE and the fraction of
    resamples that yielded a finite value (low fraction => unstable, e.g. too few
    mixed problems for d_within)."""
    problems = list(problems)
    point = estimator(problems)
    rng = np.random.default_rng(seed)
    n = len(problems)
    reps: List[float] = []
    if n > 0:
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            resampled = [problems[i] for i in idx]
            val = estimator(resampled)
            if math.isfinite(val):
                reps.append(val)
    lo = hi = se = float("nan")
    if reps:
        arr = np.asarray(reps, dtype=float)
        alpha = (1.0 - ci) / 2.0
        lo = float(np.quantile(arr, alpha))
        hi = float(np.quantile(arr, 1.0 - alpha))
        se = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    return {
        "point": point,
        "ci_level": ci,
        "ci_lo": lo,
        "ci_hi": hi,
        "boot_se": se,
        "n_boot": n_boot,
        "frac_finite": (len(reps) / n_boot) if n_boot else float("nan"),
        "seed": seed,
    }


# ----------------------------------------------------------------------------- #
# Per-checkpoint aggregation
# ----------------------------------------------------------------------------- #

def aggregate_checkpoint(
    problems: Sequence[Dict[str, Any]],
    checkpoint_id: str = "unknown",
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> Dict[str, Any]:
    """Full single-checkpoint diagnostic record."""
    counts = _count(problems)
    # aggregate mean lengths
    right: List[float] = []
    wrong: List[float] = []
    for p in problems:
        r, w = _problem_lengths(p)
        right.extend(r)
        wrong.extend(w)
    result = {
        "checkpoint_id": checkpoint_id,
        "counts": counts,
        "mean_len_right": float(np.mean(right)) if right else float("nan"),
        "mean_len_wrong": float(np.mean(wrong)) if wrong else float("nan"),
        "D_pooled": bootstrap_ci(problems, d_pooled, n_boot, ci, seed),
        "D_within": bootstrap_ci(problems, d_within, n_boot, ci, seed),
    }
    # Primary reported scalar + a convenience "length bias present" flag:
    # bias present if the difficulty-controlled CI excludes 1.0 from below.
    dw = result["D_within"]
    result["length_bias_flag"] = bool(
        math.isfinite(dw["ci_lo"]) and dw["ci_lo"] > 1.0
    )
    return result


# ----------------------------------------------------------------------------- #
# Rank / correlation helpers (numpy only; no scipy)
# ----------------------------------------------------------------------------- #

def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average ranks (ties -> mean rank), 1-based.  scipy.stats.rankdata clone."""
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    sa = a[order]
    i = 0
    n = len(a)
    while i < n:
        j = i
        while j + 1 < n and sa[j + 1] == sa[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-based average rank
        ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xm = x - x.mean()
    ym = y - y.mean()
    denom = math.sqrt(float(np.dot(xm, xm)) * float(np.dot(ym, ym)))
    if denom == 0.0:
        return float("nan")
    return float(np.dot(xm, ym) / denom)


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation (Pearson on ranks)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) != len(y) or len(x) < 2:
        return float("nan")
    return _pearson(_rankdata(x), _rankdata(y))


def spearman_perm_test(
    x: Sequence[float],
    y: Sequence[float],
    n_perm: int = 10000,
    seed: int = 0,
) -> Dict[str, Any]:
    """Two-sided permutation p-value for Spearman rho by shuffling y's labels.

    This is the workspace's signature permutation-null gate, adapted to the
    cross-checkpoint level.  With only n=7 checkpoints the minimum attainable
    two-sided p is ~2/7! -> effectively coarse; treat as an effect-size-first
    scaffold, not a decisive test (see synthesis: 'n=7 is underpowered')."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    obs = spearman(x, y)
    if not math.isfinite(obs) or n < 2:
        return {"rho": obs, "p_perm": float("nan"), "n": n, "n_perm": n_perm}
    rng = np.random.default_rng(seed)
    rx = _rankdata(x)
    ry = _rankdata(y)
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(ry)
        rho = _pearson(rx, perm)
        if math.isfinite(rho) and abs(rho) >= abs(obs) - 1e-12:
            count += 1
    p = (count + 1) / (n_perm + 1)  # add-one (never 0)
    return {"rho": float(obs), "p_perm": float(p), "n": n, "n_perm": n_perm}


# ----------------------------------------------------------------------------- #
# Exact one-sided permutation Spearman test (PREREG-P2-GRPO-20260710.md §5)
#
# These are the PUBLIC versions of the enumeration machinery that previously
# lived as private helpers inside analysis_deep.py's power analysis
# (_null_spearman_dist / _one_sided_p / _critical_rho — now thin delegations to
# here, so the prereg's primary test and the power analysis can never drift
# apart).  numpy + stdlib only, same as the rest of this module.
# ----------------------------------------------------------------------------- #

# full enumeration is n! — keep an explicit ceiling so a bad caller can't hang
_EXACT_ENUM_MAX_N = 9


def exact_null_spearman_dist(n: int) -> np.ndarray:
    """Exact null distribution of Spearman rho for n DISTINCT-ranKED points.

    For no-tie data, Spearman = Pearson(ranks_x, ranks_y) and the permutation
    null over all n! label assignments depends ONLY on n (it is Pearson between
    a fixed 1..n vector and every permutation of 1..n).  Returns the sorted
    n!-vector.  (Moved verbatim from analysis_deep._null_spearman_dist.)
    """
    if n > _EXACT_ENUM_MAX_N:
        raise ValueError(f"n={n} > {_EXACT_ENUM_MAX_N}: n! enumeration refused")
    from itertools import permutations as _perms
    a = np.arange(1.0, n + 1.0)
    am = a - a.mean()
    denom = float(np.dot(am, am))  # same for x and every permutation of ranks
    vals = np.empty(math.factorial(n), dtype=float)
    for i, perm in enumerate(_perms(a)):
        bm = np.asarray(perm) - a.mean()
        vals[i] = float(np.dot(am, bm) / denom)
    vals.sort()
    return vals


def one_sided_p_from_null(rho_obs: float, null_sorted: np.ndarray) -> float:
    """Exact one-sided (positive-direction) permutation p: fraction of the
    enumerated null Spearman values >= observed (tiny tolerance for float
    identity).  (Moved verbatim from analysis_deep._one_sided_p.)"""
    idx = np.searchsorted(null_sorted, rho_obs - 1e-12, side="left")
    ge = null_sorted.size - idx
    return float(ge) / float(null_sorted.size)


def critical_rho_from_null(null_sorted: np.ndarray, alpha: float = 0.05) -> float:
    """Smallest observed rho whose one-sided p is < alpha; inf if unattainable
    at this n.  (Moved verbatim from analysis_deep._critical_rho.)"""
    uniq = np.unique(null_sorted)
    for r in uniq:
        if one_sided_p_from_null(r, null_sorted) < alpha:
            return float(r)
    return float("inf")


def exact_one_sided_spearman_test(
    x: Sequence[float],
    y: Sequence[float],
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """PREREG §5 primary test: one-sided (positive) Spearman(x, y) with EXACT
    permutation p over ALL n! label permutations of y.

    Unlike exact_null_spearman_dist (which assumes distinct ranks), this
    enumerates permutations of y's ACTUAL rank vector, so it stays exact under
    ties; with no ties the two nulls are identical.  The reported critical rho
    always comes from the enumerated (tie-aware) null.
    """
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    n = len(xa)
    out: Dict[str, Any] = {"n": n, "alpha": alpha, "test": "exact_one_sided_spearman"}
    if len(ya) != n or n < 3:
        out.update({"rho": float("nan"), "p_exact_one_sided": float("nan"),
                    "warning": "n<3 (or length mismatch): not testable"})
        return out
    if n > _EXACT_ENUM_MAX_N:
        raise ValueError(f"n={n} > {_EXACT_ENUM_MAX_N}: n! enumeration refused")
    from itertools import permutations as _perms
    rx = _rankdata(xa)
    ry = _rankdata(ya)
    obs = _pearson(rx, ry)
    if not math.isfinite(obs):
        out.update({"rho": obs, "p_exact_one_sided": float("nan"),
                    "warning": "degenerate (constant) vector: rho undefined"})
        return out
    null = np.empty(math.factorial(n), dtype=float)
    for i, perm in enumerate(_perms(range(n))):
        rho = _pearson(rx, ry[list(perm)])
        null[i] = rho if math.isfinite(rho) else 0.0
    null.sort()
    p = one_sided_p_from_null(obs, null)
    ties = bool(np.unique(rx).size < n or np.unique(ry).size < n)
    out.update({
        "rho": float(obs),
        "p_exact_one_sided": float(p),
        "n_permutations_enumerated": int(null.size),
        "min_attainable_p": 1.0 / float(null.size),
        "critical_rho_p05_onesided": critical_rho_from_null(null, alpha),
        "ties_present": ties,
        "significant": bool(p < alpha),
    })
    return out


# ----------------------------------------------------------------------------- #
# Cross-checkpoint Spearman scaffold
# ----------------------------------------------------------------------------- #

def cross_checkpoint_spearman(
    pre_rl_D: Dict[str, float],
    post_overthinking_gap: Dict[str, float],
    n_perm: int = 10000,
    seed: int = 0,
) -> Dict[str, Any]:
    """Correlate the cheap pre-RL diagnostic (per-checkpoint D) against the
    expensive post-GRPO overthinking gap, across the checkpoint panel.

    pre_rl_D:              {checkpoint_id: D_within (or D_pooled)}
    post_overthinking_gap: {checkpoint_id: measured post-GRPO overthinking gap}

    Keys are aligned on their intersection.  Returns Spearman rho, permutation
    p-value, and the aligned (x, y, ids) used.  This is the P2 payoff test; it
    is a SCAFFOLD because the post-GRPO gap vector is produced by the QUEUED GPU
    job (see make_jobs.py / run_diag.py), not by this CPU module."""
    ids = [k for k in pre_rl_D if k in post_overthinking_gap]
    ids.sort()
    x = [float(pre_rl_D[k]) for k in ids]
    y = [float(post_overthinking_gap[k]) for k in ids]
    out = spearman_perm_test(x, y, n_perm=n_perm, seed=seed)
    out.update({"ids": ids, "pre_rl_D": x, "post_gap": y})
    if out["n"] < 3:
        out["warning"] = (
            "n<3 aligned checkpoints: correlation is not interpretable; "
            "populate more checkpoints before reading rho."
        )
    return out


if __name__ == "__main__":  # tiny self-demo (no external data)
    demo = [
        {"problem": "p1", "samples": [
            {"len": 120, "correct": True}, {"len": 260, "correct": False}]},
        {"problem": "p2", "samples": [
            {"len": 90, "correct": True}, {"len": 300, "correct": False}]},
    ]
    print(json.dumps(aggregate_checkpoint(demo, "demo", n_boot=500), indent=2))
