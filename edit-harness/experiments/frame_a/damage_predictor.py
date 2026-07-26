"""damage_predictor.py — the router's per-edit damage input `d̂(u)`.

Thin wrapper over the B6 zero-parameter predictor: **raw signed key-cosine of the edit key at
the geometry-valid layer L12** (`experiments/d3_benefit_predictor.py:raw_predict_cell`,
held-out per-edit within-cell Spearman ρ=0.725 @ L12). Binding constraints (DESIGN §0):

  * The geometry claim is **L12-ONLY**. This module never returns an L14 geometry prediction;
    L14 lift is norm-growth magnitude, not geometry, and is out of scope for the router input.
  * The predictor is Llama-calibrated. Cross-arch = **recalibrate**, never assume transfer:
    `recalibrate()` fits a per-arch monotone affine map on a disjoint calibration slice; the
    zero-parameter raw signal is the default when no recalibration is supplied.
  * `d̂` is a *pre-edit, key-derived* quantity — it never touches `gt_damage` (which is a
    scorer/oracle input only), so nothing leaks the ground truth into the router.

DRYRUN: `predict(update)` returns `update["key_cos"]` (the stream carries the pre-computed raw
key-cosine per update); no torch, no GPU. The real path (`predict_from_key`) is lazy and used
only off dryrun when an edit key vector must be scored live.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from . import config as C


@dataclass
class DamagePredictor:
    """Maps an update to a predicted per-edit collateral-damage score in [-1, 1]-ish key-cos units.

    `arch` labels the calibrated architecture. `recal` is an optional (scale, bias) affine
    recalibration fitted per-arch on a DISJOINT calibration slice (never on a scored stream);
    when absent the raw signed key-cosine is returned unchanged (the zero-parameter predictor).
    """
    arch: str = "llama"
    layer: int = C.GEOMETRY_LAYER            # L12; guarded against L14 use below.
    recal: Optional[Dict[str, float]] = None  # {"scale": s, "bias": b} or None (raw).

    def __post_init__(self) -> None:
        if self.layer != C.GEOMETRY_LAYER:
            raise ValueError(
                f"damage predictor is geometry-valid at L{C.GEOMETRY_LAYER} only; "
                f"L{self.layer} is magnitude, not geometry (DESIGN §0). Refusing.")

    # ------------------------------------------------------------------ dryrun / stream path
    def predict(self, update: Dict) -> float:
        """d̂(u) from the update's pre-computed raw signed key-cosine (the stream metadata)."""
        raw = float(update["key_cos"])
        return self._apply_recal(raw)

    def _apply_recal(self, raw: float) -> float:
        if self.recal is None:
            return raw
        return self.recal["scale"] * raw + self.recal["bias"]

    # ------------------------------------------------------------------ recalibration (per-arch)
    def recalibrate(self, raw_cos, gt_damage) -> "DamagePredictor":
        """Fit a monotone affine map raw_key_cos -> damage on a DISJOINT calibration slice.

        This is the *method that recalibrates per-arch* (DESIGN desk-reject #2): the router
        logic is arch-agnostic; only these two coefficients move. Least-squares on the slice;
        `gt_damage` here is calibration-slice ground truth, disjoint from every scored stream.
        """
        import numpy as np
        x = np.asarray(raw_cos, float)
        y = np.asarray(gt_damage, float)
        if x.size < 2 or float(x.std()) < 1e-12:
            self.recal = {"scale": 1.0, "bias": 0.0}
            return self
        A = np.stack([x, np.ones_like(x)], axis=1)
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        self.recal = {"scale": float(coef[0]), "bias": float(coef[1])}
        return self

    # ------------------------------------------------------------------ real (off-dryrun) path
    def predict_from_key(self, model, tokenizer, edit_request: Dict, probe_geom,
                         device: str = "cuda", capture_fn=None, subject_idx_fn=None) -> float:
        """Live per-edit `key_cos` at L12 — the (d)-assert primitive. Captures the edit's down_proj
        input key at the subject's last token (the EXACT `rome_native._capture_key` the gate cells
        used), L2-normalizes, and returns the mean signed cosine to `probe_geom`'s base-known probe
        keys — reproducing the stored cell aggregate to fp tolerance. Lazy imports; the capture and
        subject-index fns are injectable so the path is CPU-mockable (see real_replay.make_mock_harness).
        """
        from .real_replay import predict_key_cos
        if capture_fn is None or subject_idx_fn is None:
            import os as _os, sys as _sys
            _exp = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
            if _exp not in _sys.path:
                _sys.path.insert(0, _exp)
            from editors.rome_native import _capture_key, find_subject_last_token_index
            capture_fn = capture_fn or _capture_key
            subject_idx_fn = subject_idx_fn or find_subject_last_token_index
        idx = subject_idx_fn(tokenizer, edit_request["prompt"], edit_request.get("subject"))
        k = capture_fn(model, tokenizer, self.layer, edit_request["prompt"], idx, device)
        import numpy as _np
        k = _np.asarray(k.float().cpu().numpy() if hasattr(k, "float") else k, float)
        return predict_key_cos(k / (_np.linalg.norm(k) + 1e-8), probe_geom)


# ---------------------------------------------------------------- selftest
def _selftest() -> None:
    p = DamagePredictor()
    # raw passthrough of the stream's key_cos:
    assert p.predict({"key_cos": 0.37}) == 0.37
    # monotone recalibration preserves ordering:
    p2 = DamagePredictor().recalibrate([0.0, 0.2, 0.4, 0.6], [0.0, 1.0, 2.0, 3.0])
    lo = p2.predict({"key_cos": 0.1}); hi = p2.predict({"key_cos": 0.5})
    assert hi > lo, "recalibrated predictor must stay monotone in key_cos"
    assert p2.recal is not None and p2.recal["scale"] > 0
    # L14 (or any non-L12) geometry predictor is refused:
    try:
        DamagePredictor(layer=14)
        raise AssertionError("must refuse L14 geometry predictor")
    except ValueError:
        pass
    # flat calibration slice -> identity recal (no divide-by-zero):
    p3 = DamagePredictor().recalibrate([0.3, 0.3], [1.0, 1.0])
    assert p3.recal == {"scale": 1.0, "bias": 0.0}
    print("damage_predictor selftest: PASS")


if __name__ == "__main__":
    _selftest()
