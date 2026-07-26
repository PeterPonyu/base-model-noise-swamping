"""disagreement.py — per-window forecast disagreement signals.

Two flavours of disagreement, both reduced to a single scalar per rolling
window (aggregated over the horizon):

1. **Cross-model** (the hypothesis): spread *across architecturally distinct
   models* forecasting the same window. Computed from a [n_models, horizon]
   matrix.
2. **Temperature-resampling** (the null baseline): spread across repeated
   stochastic draws from a *single* model, from a [n_samples, horizon] matrix.

If cross-model disagreement tracks true forecast error substantially better
than the resampling null, architectural diversity carries UQ signal the null
cannot — that is the P4 kill-gate (evaluated in calibrate.py).

Spread is summarised by both IQR and std; the horizon axis is collapsed by a
configurable reducer (mean by default). NaNs (a model that failed to emit a
value at some step) are ignored per-step.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np


def _iqr(a: np.ndarray, axis: int = 0) -> np.ndarray:
    q75, q25 = np.nanpercentile(a, [75, 25], axis=axis)
    return q75 - q25


def _std(a: np.ndarray, axis: int = 0) -> np.ndarray:
    return np.nanstd(a, axis=axis)


def per_step_spread(matrix: np.ndarray) -> Dict[str, np.ndarray]:
    """Per-horizon-step spread across the first axis (models or samples).

    Returns {"iqr": [horizon], "std": [horizon]}.
    """
    m = np.asarray(matrix, dtype=np.float64)
    if m.ndim != 2:
        raise ValueError("expected a 2-D [rows, horizon] matrix")
    return {"iqr": _iqr(m, axis=0), "std": _std(m, axis=0)}


def window_disagreement(
    matrix: np.ndarray,
    metric: str = "std",
    reduce: Callable[[np.ndarray], float] = np.nanmean,
) -> float:
    """Collapse a [rows, horizon] matrix to a single disagreement scalar.

    ``metric`` in {"std", "iqr"}; ``reduce`` collapses the horizon axis.
    """
    spread = per_step_spread(matrix)
    if metric not in spread:
        raise ValueError(f"metric must be one of {list(spread)}")
    val = float(reduce(spread[metric]))
    return val


def cross_model_disagreement(matrix: np.ndarray, metric: str = "std") -> float:
    """Disagreement across architecturally distinct models (the signal)."""
    return window_disagreement(matrix, metric=metric)


def resampling_disagreement(matrix: np.ndarray, metric: str = "std") -> float:
    """Disagreement across temperature-resampled draws of one model (the null)."""
    return window_disagreement(matrix, metric=metric)


def committee_point_forecast(matrix: np.ndarray, how: str = "median") -> np.ndarray:
    """Consensus point forecast [horizon] from a [n_models, horizon] matrix."""
    m = np.asarray(matrix, dtype=np.float64)
    if how == "mean":
        return np.nanmean(m, axis=0)
    if how == "median":
        return np.nanmedian(m, axis=0)
    raise ValueError("how must be 'mean' or 'median'")


def forecast_abs_error(point: np.ndarray, truth: np.ndarray, reduce=np.mean) -> float:
    """Scalar per-window forecast error: reduce(|point - truth|) over horizon."""
    p = np.asarray(point, dtype=np.float64)
    t = np.asarray(truth, dtype=np.float64)
    return float(reduce(np.abs(p - t)))


def _selftest() -> int:
    rng = np.random.default_rng(3)
    horizon = 12
    # a tight matrix (low spread) and a loose one (high spread)
    tight = rng.normal(0, 0.1, size=(5, horizon))
    loose = rng.normal(0, 2.0, size=(5, horizon))
    d_tight = cross_model_disagreement(tight)
    d_loose = cross_model_disagreement(loose)
    assert d_loose > d_tight, "looser matrix must have larger disagreement"
    # NaN tolerance
    withnan = loose.copy()
    withnan[0, 0] = np.nan
    assert np.isfinite(cross_model_disagreement(withnan))
    # point forecast + error
    truth = np.zeros(horizon)
    pt = committee_point_forecast(loose, how="median")
    err = forecast_abs_error(pt, truth)
    assert np.isfinite(err)
    print(f"[disagreement] std tight={d_tight:.4f} loose={d_loose:.4f} (loose>tight OK)")
    print(f"[disagreement] iqr loose={cross_model_disagreement(loose, 'iqr'):.4f}")
    print(f"[disagreement] point-forecast abs err={err:.4f} (finite OK)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
