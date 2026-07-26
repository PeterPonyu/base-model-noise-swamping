"""arms/real_asserts.py — the reviewer's first-wave integrity asserts (c)+(d), as pure helpers.

These are torch-free, CPU-testable functions whose LOGIC is exercised by the self-test gate on
synthetic inputs; they are CALLED on the real-GPU path (RealEditArm._real_install and the real
replay in run_stream) where they run against measured tensors. Keeping them pure means the
guarantee is verified build-only even though the GPU wiring is not.

  (c) ΔW parity for the edit arm: ROME installs a RANK-ONE update ΔW = outer(residual, key)/denom.
      `assert_rank_one` checks the installed weight delta is rank-one (2nd singular value ≪ 1st);
      `assert_rank_one_parity` checks it equals the reconstructed outer product;
      `assert_delta_norm_match` checks ‖ΔW‖ matches the editor's reported `delta_weight_norm`.
  (d) live predictor == stream key_cos: `assert_key_cos_match` checks the LIVE raw signed
      key-cosine (recomputed from the loaded model at edit time) equals the stream's stored
      `key_cos` for that edit — catching any drift between the built stream and the live model.
"""
from __future__ import annotations

import numpy as np


def assert_rank_one(delta, rtol: float = 1e-3) -> float:
    """Assert a 2D weight-delta is (numerically) rank-one. Returns s2/s1 for logging."""
    d = np.asarray(delta, dtype=np.float64)
    if d.ndim != 2:
        raise AssertionError(f"delta must be 2D, got shape {d.shape}")
    s = np.linalg.svd(d, compute_uv=False)
    if s[0] <= 0:
        raise AssertionError("delta is all-zero; expected a rank-one edit")
    ratio = float(s[1] / s[0]) if s.size > 1 else 0.0
    if ratio > rtol:
        raise AssertionError(f"installed ΔW is not rank-one: s2/s1={ratio:.2e} > {rtol:.0e} "
                             f"(ROME must install a rank-one update)")
    return ratio


def assert_rank_one_parity(delta_installed, u_vec, v_vec, atol: float = 1e-4) -> None:
    """Assert installed ΔW == outer(u_vec, v_vec) (reconstructed-vs-installed parity)."""
    d = np.asarray(delta_installed, dtype=np.float64)
    recon = np.outer(np.asarray(u_vec, np.float64), np.asarray(v_vec, np.float64))
    if d.shape != recon.shape:
        raise AssertionError(f"parity shape mismatch: installed {d.shape} vs recon {recon.shape}")
    md = float(np.max(np.abs(d - recon)))
    if md > atol:
        raise AssertionError(f"ΔW parity failed: max|installed - outer(u,v)| = {md:.2e} > {atol:.0e}")


def assert_delta_norm_match(delta_installed, reported_norm: float, rtol: float = 1e-3) -> float:
    """Assert ‖installed ΔW‖ matches the editor's reported delta_weight_norm. Returns the rel-err."""
    n = float(np.linalg.norm(np.asarray(delta_installed, np.float64)))
    if reported_norm <= 0:
        raise AssertionError("reported delta_weight_norm must be > 0 for a real edit")
    rel = abs(n - reported_norm) / reported_norm
    if rel > rtol:
        raise AssertionError(f"‖ΔW‖ mismatch: installed {n:.4g} vs reported {reported_norm:.4g} "
                             f"(rel {rel:.2e} > {rtol:.0e})")
    return rel


def assert_key_cos_match(live_key_cos: float, stored_key_cos: float, atol: float = 1e-3) -> None:
    """Assert the live per-edit key-cosine equals the stream's stored key_cos (no drift)."""
    if abs(float(live_key_cos) - float(stored_key_cos)) > atol:
        raise AssertionError(
            f"live key_cos {live_key_cos:.5f} != stream key_cos {stored_key_cos:.5f} "
            f"(|Δ| > {atol:.0e}) — the built stream and the live model disagree on the edit key.")


# ---------------------------------------------------------------- selftest (pure, CPU)
def _selftest() -> None:
    rng = np.random.default_rng(0)
    u = rng.normal(size=64); v = rng.normal(size=128)
    delta = np.outer(u, v)
    # (c) rank-one: a true outer product passes; a full-rank matrix fails.
    assert assert_rank_one(delta) < 1e-6
    try:
        assert_rank_one(rng.normal(size=(64, 128))); raise SystemExit("should have failed")
    except AssertionError:
        pass
    # parity: identical reconstruction passes; a perturbed one fails.
    assert_rank_one_parity(delta, u, v)
    try:
        assert_rank_one_parity(delta + 1.0, u, v, atol=1e-4); raise SystemExit("should have failed")
    except AssertionError:
        pass
    # norm match.
    assert_delta_norm_match(delta, float(np.linalg.norm(delta)))
    try:
        assert_delta_norm_match(delta, float(np.linalg.norm(delta)) * 1.5); raise SystemExit("fail")
    except AssertionError:
        pass
    # (d) key_cos match.
    assert_key_cos_match(0.4213, 0.4215)
    try:
        assert_key_cos_match(0.40, 0.60); raise SystemExit("should have failed")
    except AssertionError:
        pass
    print("arms.real_asserts selftest: PASS")


if __name__ == "__main__":
    _selftest()
