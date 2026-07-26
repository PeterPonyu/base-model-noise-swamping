"""calibrate.py — does disagreement calibrate forecast uncertainty?

Given, over a set of rolling windows, a per-window disagreement scalar and the
per-window absolute forecast error, this module answers:

- **Rank calibration**: Spearman(disagreement, |error|). A positive, significant
  correlation means larger disagreement flags larger error.
- **Interval calibration**: treat disagreement as a predictive sigma, form
  central Gaussian prediction intervals, and report PICP (coverage) and mean
  interval width at a nominal level.
- **Proper score**: Gaussian CRPS of (point forecast, sigma=disagreement) vs
  truth-implied error.
- **Uncertainty on all of the above**: bootstrap 95% CIs over windows.
- **The kill-gate**: compare cross-architecture disagreement against the
  temperature-resampling null via the bootstrap CI of the Spearman *difference*.

All functions are pure NumPy/SciPy on 1-D arrays; no I/O.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from scipy import stats

Z_TABLE = {0.80: 1.2815515655, 0.90: 1.6448536270, 0.95: 1.9599639845, 0.99: 2.5758293035}


# ---------------------------------------------------------------------------
# Plausibility guard
# ---------------------------------------------------------------------------
def plausible_mask(*arrays: np.ndarray, rel_cap: float = 1e4) -> np.ndarray:
    """Joint mask keeping windows where every array is finite AND not
    astronomically large relative to its own sample.

    ``isfinite`` alone does not reject a *finite but astronomical* decode
    artifact (e.g. 1e148 from a swallowed digit run), which then dominates
    every mean-based metric. A window is kept iff, for each array, the value
    is finite and ``|x| <= rel_cap * median(|finite x|)`` (median floored to
    a tiny positive number so all-zero arrays don't reject everything).
    """
    arrs = [np.asarray(a, dtype=np.float64) for a in arrays]
    mask = np.ones(arrs[0].size, dtype=bool)
    for a in arrs:
        finite = np.isfinite(a)
        med = np.median(np.abs(a[finite])) if finite.any() else 0.0
        cap = rel_cap * max(med, 1e-12)
        mask &= finite & (np.abs(a) <= cap)
    return mask


# ---------------------------------------------------------------------------
# Rank calibration
# ---------------------------------------------------------------------------
def spearman(disagreement: np.ndarray, abs_error: np.ndarray) -> Dict[str, float]:
    """Spearman rank correlation between disagreement and |error|."""
    d = np.asarray(disagreement, dtype=np.float64)
    e = np.asarray(abs_error, dtype=np.float64)
    mask = np.isfinite(d) & np.isfinite(e)
    if mask.sum() < 3:
        return {"rho": float("nan"), "pvalue": float("nan"), "n": int(mask.sum())}
    rho, p = stats.spearmanr(d[mask], e[mask])
    return {"rho": float(rho), "pvalue": float(p), "n": int(mask.sum())}


# ---------------------------------------------------------------------------
# Interval calibration
# ---------------------------------------------------------------------------
def picp_width(
    errors: np.ndarray,
    sigma: np.ndarray,
    level: float = 0.90,
) -> Dict[str, float]:
    """Prediction Interval Coverage Probability and mean width.

    We work in error space: the truth deviates from the point forecast by
    ``error`` (signed residual). A central Gaussian interval of half-width
    ``z*sigma`` covers the truth iff ``|error| <= z*sigma``. ``errors`` may be
    signed residuals or absolute errors; absolute value is taken either way.
    """
    e = np.abs(np.asarray(errors, dtype=np.float64))
    s = np.asarray(sigma, dtype=np.float64)
    mask = np.isfinite(e) & np.isfinite(s) & (s >= 0)
    if mask.sum() == 0:
        return {"picp": float("nan"), "mean_width": float("nan"), "nominal": level, "n": 0}
    z = Z_TABLE.get(round(level, 2), float(stats.norm.ppf(0.5 + level / 2.0)))
    halfwidth = z * s[mask]
    covered = e[mask] <= halfwidth
    return {
        "picp": float(np.mean(covered)),
        "mean_width": float(np.mean(2.0 * halfwidth)),
        "nominal": float(level),
        "n": int(mask.sum()),
    }


# ---------------------------------------------------------------------------
# CRPS (Gaussian closed form)
# ---------------------------------------------------------------------------
def crps_gaussian(
    errors: np.ndarray,
    sigma: np.ndarray,
    sigma_floor: float = 1e-8,
) -> Dict[str, float]:
    """Mean Gaussian CRPS given signed residual ``error`` and predictive sigma.

    CRPS(N(0, sigma), error) with z = error / sigma:
        sigma * [ z*(2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi) ]
    (centering the predictive mean at the point forecast, so the observation's
    offset is exactly the residual ``error``.)
    """
    e = np.asarray(errors, dtype=np.float64)
    s = np.maximum(np.asarray(sigma, dtype=np.float64), sigma_floor)
    mask = np.isfinite(e) & np.isfinite(s)
    if mask.sum() == 0:
        return {"crps": float("nan"), "n": 0}
    z = e[mask] / s[mask]
    crps = s[mask] * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))
    return {"crps": float(np.mean(crps)), "n": int(mask.sum())}


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
def _bootstrap_stat(fn, *arrays, n_boot: int = 2000, seed: int = 0) -> Dict[str, float]:
    arrs = [np.asarray(a, dtype=np.float64) for a in arrays]
    n = arrs[0].size
    rng = np.random.default_rng(seed)
    stats_out = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats_out[b] = fn(*[a[idx] for a in arrs])
    stats_out = stats_out[np.isfinite(stats_out)]
    if stats_out.size == 0:
        return {"mean": float("nan"), "lo95": float("nan"), "hi95": float("nan"), "n_boot": 0}
    lo, hi = np.percentile(stats_out, [2.5, 97.5])
    return {
        "mean": float(np.mean(stats_out)),
        "lo95": float(lo),
        "hi95": float(hi),
        "n_boot": int(stats_out.size),
    }


def bootstrap_spearman(
    disagreement: np.ndarray,
    abs_error: np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
) -> Dict[str, float]:
    """Bootstrap 95% CI of Spearman(disagreement, |error|) over windows."""
    def _rho(d, e):
        m = np.isfinite(d) & np.isfinite(e)
        if m.sum() < 3:
            return float("nan")
        return stats.spearmanr(d[m], e[m]).statistic
    ci = _bootstrap_stat(_rho, disagreement, abs_error, n_boot=n_boot, seed=seed)
    ci["point"] = spearman(disagreement, abs_error)["rho"]
    return ci


# ---------------------------------------------------------------------------
# The kill-gate: cross-architecture vs resampling null
# ---------------------------------------------------------------------------
def compare_cross_vs_resampling(
    cross_disagreement: np.ndarray,
    resampling_disagreement: np.ndarray,
    abs_error: np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
) -> Dict[str, object]:
    """Kill-gate comparison.

    Paired bootstrap over windows of
        delta = Spearman(cross, |err|) - Spearman(resampling, |err|).
    The cross-architecture committee "wins" (gate PASS) iff its Spearman is
    positive AND the 95% CI of delta lies strictly above 0.
    """
    cross = np.asarray(cross_disagreement, dtype=np.float64)
    resamp = np.asarray(resampling_disagreement, dtype=np.float64)
    err = np.asarray(abs_error, dtype=np.float64)

    def _rho(d, e):
        m = np.isfinite(d) & np.isfinite(e)
        if m.sum() < 3:
            return float("nan")
        return stats.spearmanr(d[m], e[m]).statistic

    def _delta(c, r, e):
        return _rho(c, e) - _rho(r, e)

    delta_ci = _bootstrap_stat(_delta, cross, resamp, err, n_boot=n_boot, seed=seed)
    cross_sp = spearman(cross, err)
    resamp_sp = spearman(resamp, err)
    delta_point = cross_sp["rho"] - resamp_sp["rho"]

    gate_pass = bool(
        np.isfinite(cross_sp["rho"])
        and cross_sp["rho"] > 0
        and np.isfinite(delta_ci["lo95"])
        and delta_ci["lo95"] > 0
    )
    return {
        "cross_spearman": cross_sp,
        "resampling_spearman": resamp_sp,
        "delta_point": float(delta_point),
        "delta_ci": delta_ci,
        "gate_pass": gate_pass,
    }


# ---------------------------------------------------------------------------
# Convenience: full calibration report from window arrays
# ---------------------------------------------------------------------------
def calibration_report(
    cross_disagreement: np.ndarray,
    resampling_disagreement: np.ndarray,
    signed_error: np.ndarray,
    level: float = 0.90,
    n_boot: int = 2000,
    seed: int = 0,
    rel_cap: float = 1e4,
) -> Dict[str, object]:
    """Assemble the full P4 calibration report from per-window arrays.

    ``signed_error`` = truth - point_forecast (residual); |.| used where an
    absolute error is required. ``cross_disagreement`` is used as the predictive
    sigma for PICP/CRPS (it is the candidate UQ).

    Windows failing :func:`plausible_mask` (non-finite OR finite-but-
    astronomical decode artifacts, rel_cap x median) are DROPPED up front and
    counted in ``n_windows_dropped_implausible``. Set ``rel_cap=None``/inf to
    keep the old finite-only behaviour.
    """
    cross = np.asarray(cross_disagreement, dtype=np.float64)
    resamp = np.asarray(resampling_disagreement, dtype=np.float64)
    signed = np.asarray(signed_error, dtype=np.float64)
    n_total = int(signed.size)
    if rel_cap is not None and np.isfinite(rel_cap):
        keep = plausible_mask(cross, resamp, signed, rel_cap=rel_cap)
        cross_disagreement, resampling_disagreement, signed_error = (
            cross[keep], resamp[keep], signed[keep])
    abs_err = np.abs(np.asarray(signed_error, dtype=np.float64))
    report = {
        "n_windows": int(abs_err.size),
        "n_windows_total": n_total,
        "n_windows_dropped_implausible": n_total - int(abs_err.size),
        "spearman_cross": spearman(cross_disagreement, abs_err),
        "spearman_resampling": spearman(resampling_disagreement, abs_err),
        "bootstrap_spearman_cross": bootstrap_spearman(cross_disagreement, abs_err, n_boot, seed),
        "picp_width_cross": picp_width(signed_error, cross_disagreement, level),
        "crps_cross": crps_gaussian(signed_error, cross_disagreement),
        "comparison": compare_cross_vs_resampling(
            cross_disagreement, resampling_disagreement, abs_err, n_boot, seed
        ),
    }
    return report


def _selftest() -> int:
    """Synthetic (disagreement, error) where disagreement is built to correlate
    with error — assert Spearman recovers it and CRPS/PICP are finite."""
    rng = np.random.default_rng(7)
    n = 300
    latent = rng.gamma(2.0, 1.0, size=n)  # latent difficulty
    # cross disagreement tracks latent difficulty; resampling is mostly noise
    cross = latent + rng.normal(0, 0.3, size=n)
    resamp = 0.15 * latent + rng.normal(0, 1.0, size=n)
    signed_err = rng.normal(0, 1.0, size=n) * latent  # error scale grows with latent

    sp_cross = spearman(cross, np.abs(signed_err))
    sp_resamp = spearman(resamp, np.abs(signed_err))
    assert sp_cross["rho"] > 0.4, f"cross Spearman too low: {sp_cross}"
    assert sp_cross["rho"] > sp_resamp["rho"], "cross must beat resampling"
    assert sp_cross["pvalue"] < 1e-6

    pw = picp_width(signed_err, cross, level=0.90)
    cr = crps_gaussian(signed_err, cross)
    assert np.isfinite(pw["picp"]) and np.isfinite(pw["mean_width"])
    assert np.isfinite(cr["crps"])

    boot = bootstrap_spearman(cross, np.abs(signed_err), n_boot=500)
    assert boot["lo95"] > 0, "bootstrap lower CI should exclude 0"

    cmp = compare_cross_vs_resampling(cross, resamp, np.abs(signed_err), n_boot=500)
    assert cmp["gate_pass"] is True, "constructed data should PASS the gate"

    print(f"[calibrate] spearman cross rho={sp_cross['rho']:.3f} p={sp_cross['pvalue']:.2e} "
          f"(resamp rho={sp_resamp['rho']:.3f})")
    print(f"[calibrate] bootstrap cross rho 95% CI=[{boot['lo95']:.3f}, {boot['hi95']:.3f}]")
    print(f"[calibrate] PICP@0.90={pw['picp']:.3f} mean_width={pw['mean_width']:.3f}")
    print(f"[calibrate] CRPS={cr['crps']:.4f}")
    print(f"[calibrate] delta(cross-resamp) CI=[{cmp['delta_ci']['lo95']:.3f}, "
          f"{cmp['delta_ci']['hi95']:.3f}]  gate_pass={cmp['gate_pass']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
