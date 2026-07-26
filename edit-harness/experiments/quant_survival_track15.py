"""quant_survival_track15.py — Paper B Track-1.5 utility: frozen-scale GPTQ/AWQ isolated codec.

CPU-side infrastructure only. The real kernels (AutoGPTQ / AutoAWQ) are optional at import time;
any GPU path is invoked only when this module is called from a driver. Lazy imports and clear
errors if packages are missing — the standing ask-first download rule is respected.

Track-1.5 protocol:
  1. Calibrate the quantizer ONCE on the base (or fully-edited) model — cache per-group scales
     and zero points. This is the "frozen scale".
  2. In the isolated per-edit protocol, apply the cached frozen-scale group-wise codec to ONLY
     the edited tensor(s). No re-calibration.
  3. The full-model arm = frozen non-edited groups + re-rounded edited tensor.

Correctness caveat: freezing scales on the base model and re-rounding only ΔW is an
APPROXIMATION of a true re-calibrated GPTQ/AWQ of the edited model (error-feedback would
redistribute across the group). This is documented in DESIGN-PAPERB-QUANTSURVIVAL-2026-07-16.md.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple


def _require_gptq():
    import importlib.util
    if importlib.util.find_spec("auto_gptq") is None:
        raise RuntimeError(
            "Track 1.5 GPTQ requires AutoGPTQ, which is not installed. "
            "Install is ask-first per project rules; run the driver only after user approval."
        )


def _require_awq():
    import importlib.util
    if importlib.util.find_spec("awq") is None:
        raise RuntimeError(
            "Track 1.5 AWQ requires AutoAWQ, which is not installed. "
            "Install is ask-first per project rules; run the driver only after user approval."
        )


class FrozenScaleCodec:
    """Base class for a frozen-scale group-wise 4-bit codec."""

    def __init__(self, name: str, group_size: int = 128):
        self.name = name
        self.group_size = group_size
        self._scales: Optional[Dict[int, Tuple]] = None   # id(weight) -> (scale, zero, ...)

    def calibrate(self, model) -> None:
        """Run one calibration pass over `model` and cache per-group scales/zero points.

        The exact calibration routine depends on the backend (AutoGPTQ/AutoAWQ). This is a
        CPU/GPU build task implemented at driver time; the skeleton records the contract.
        """
        raise NotImplementedError(f"{self.name} calibration not yet implemented")

    def quantize_tensor(self, w, tensor_id: int):
        """Apply the cached frozen scale to a single edited tensor.

        Args:
            w: fp32 weight tensor to quantize (already detached, on device).
            tensor_id: python id() of the weight object used to look up the cached scale.
        Returns:
            dequantized tensor of the same shape/dtype/device.
        """
        if self._scales is None:
            raise RuntimeError(f"{self.name}: calibrate() must be called before quantize_tensor()")
        if tensor_id not in self._scales:
            raise RuntimeError(f"{self.name}: no cached scale for tensor_id {tensor_id}")
        raise NotImplementedError(f"{self.name} quantize_tensor not yet implemented")

    def cache_info(self) -> str:
        n = len(self._scales) if self._scales is not None else 0
        return f"{self.name}(group_size={self.group_size}, cached_tensors={n})"


class FrozenGPTQCodec(FrozenScaleCodec):
    """Frozen-scale GPTQ-4bit codec (group size 128 by default)."""

    def __init__(self, group_size: int = 128):
        super().__init__("gptq", group_size)

    def calibrate(self, model, calibration_data=None) -> None:
        _require_gptq()
        # Real implementation will use auto_gptq to run one Hessian-based calibration pass
        # and store per-group scales/zeros/zero points keyed by id(weight) for every linear.
        # The calibration set is optional; if None, use a default small C4/WikiText slice.
        raise NotImplementedError("FrozenGPTQCodec.calibrate not yet implemented")

    def quantize_tensor(self, w, tensor_id: int):
        _require_gptq()
        super().quantize_tensor(w, tensor_id)
        # Real implementation: apply cached scale/zero to w, round to 4-bit, dequant back.


class FrozenAWQCodec(FrozenScaleCodec):
    """Frozen-scale AWQ-4bit codec (group size 128 by default)."""

    def __init__(self, group_size: int = 128):
        super().__init__("awq", group_size)

    def calibrate(self, model, calibration_data=None) -> None:
        _require_awq()
        # Real implementation will use awq to run activation-aware scaling once.
        raise NotImplementedError("FrozenAWQCodec.calibrate not yet implemented")

    def quantize_tensor(self, w, tensor_id: int):
        _require_awq()
        super().quantize_tensor(w, tensor_id)


# ------------------------------------------------------------------------
# Driver-facing helper: build the full-model arm for a single edit
# ------------------------------------------------------------------------
def build_full_model_arm_track15(
    model,
    edited_layers: list,
    gptq_codec: Optional[FrozenGPTQCodec] = None,
    awq_codec: Optional[FrozenAWQCodec] = None,
):
    """Reconstruct the full-model arm for Track-1.5.

    Non-edited linears keep their frozen quantized weights (already set in the model by the
    driver). Edited tensors are re-rounded with the cached codec. This function is a wiring
    helper; the actual tensor assignment happens in the driver.
    """
    if gptq_codec is None and awq_codec is None:
        raise ValueError("At least one of gptq_codec / awq_codec must be provided")
    # Real implementation will iterate over edited_layers, call codec.quantize_tensor on the
    # edited weights, and leave non-edited weights unchanged. This skeleton documents the
    # contract and types.
    raise NotImplementedError("build_full_model_arm_track15 not yet implemented")


# ------------------------------------------------------------------------
# Self-test (CPU-only, no model needed)
# ------------------------------------------------------------------------
def selftest():
    """Verify that the skeleton imports and that missing-package errors are clear."""
    print("[track15 selftest] FrozenScaleCodec base import OK")
    for cls, name in [(FrozenGPTQCodec, "gptq"), (FrozenAWQCodec, "awq")]:
        c = cls(group_size=128)
        assert c.name == name
        assert c.group_size == 128
        assert c._scales is None
    print("[track15 selftest] codec constructors OK")

    # Missing-package errors should be informative
    try:
        FrozenGPTQCodec().calibrate(None)
    except NotImplementedError:
        pass  # codec stub raises this after the missing-package guard is implemented
    except RuntimeError as e:
        assert "AutoGPTQ" in str(e), f"unexpected GPTQ error: {e}"
        print(f"[track15 selftest] GPTQ missing-package error is clear: {e}")
    try:
        FrozenAWQCodec().calibrate(None)
    except NotImplementedError:
        pass
    except RuntimeError as e:
        assert "AutoAWQ" in str(e), f"unexpected AWQ error: {e}"
        print(f"[track15 selftest] AWQ missing-package error is clear: {e}")
    print("[track15 selftest] ALL CHECKS PASSED")


if __name__ == "__main__":
    selftest()
