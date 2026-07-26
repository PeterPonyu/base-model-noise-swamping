"""committee.py — cross-architecture LLM committee for numeric TS forecasting.

Given a list of model names and an LLMTime-encoded context prompt, each model
produces a numeric continuation; decoding those continuations yields a forecast
vector per model. The stack of per-model forecasts is the raw material for the
cross-architecture *disagreement* signal (see disagreement.py).

Two backends:
- ``"ollama"``: real HTTP call to the local Ollama server
  (http://localhost:11434/api/generate). The network call lives ONLY inside
  :func:`generate_raw`; nothing here touches the network at import time.
- ``"mock"``: fully synthetic. A deterministic, model-seeded numeric
  continuation is produced and then *encoded with the same codec*, so the mock
  exercises the exact decode path the real backend uses. Tests run entirely on
  this backend and never contact Ollama.

The temperature-resampling null baseline is produced by
:func:`resample_single_model`, which draws several stochastic continuations
from ONE model (real: temperature>0 sampling; mock: seeded per-sample noise).
"""
from __future__ import annotations

import hashlib
import json
import urllib.request
from typing import Dict, List, Optional, Sequence

import numpy as np

from llmtime import LLMTimeCodec

DEFAULT_HOST = "http://localhost:11434"


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
def build_prompt(codec: LLMTimeCodec, context_values: Sequence[float]) -> str:
    """LLMTime-style forecasting prompt: the encoded history, ready to continue.

    The trailing separator invites the model to emit the next value straight
    away, matching the training-free LLMTime setup.
    """
    body = codec.encode(context_values)
    return body + codec.sep


# ---------------------------------------------------------------------------
# Mock backend (deterministic, no network)
# ---------------------------------------------------------------------------
def _seed_from(*parts: object) -> int:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:16], 16)


def _mock_continuation(
    model: str,
    context_values: Sequence[float],
    horizon: int,
    temperature: float,
    sample_idx: int,
) -> np.ndarray:
    """A synthetic per-model forecast: linear extrapolation of the context plus
    a deterministic, model-dependent bias and temperature-scaled noise.

    Distinct models get distinct biases and noise scales -> genuine
    cross-architecture spread. For a single model, ``sample_idx`` reseeds the
    noise so temperature resampling produces a controlled null spread.
    """
    ctx = np.asarray(context_values, dtype=np.float64).ravel()
    n = ctx.size
    # local linear trend from the last min(n, 24) points
    k = min(n, 24)
    xs = np.arange(k, dtype=np.float64)
    ys = ctx[-k:]
    slope, intercept = np.polyfit(xs, ys, 1) if k >= 2 else (0.0, ctx[-1])
    base = intercept + slope * (xs[-1] + 1 + np.arange(horizon, dtype=np.float64))

    spread = float(np.std(ctx)) or 1.0
    mrng = np.random.default_rng(_seed_from(model, "bias"))
    # model-specific systematic bias (fixed per model, drives cross-model IQR)
    bias = mrng.normal(0.0, 0.25 * spread, size=horizon)
    # temperature/sample-specific stochastic component (drives resampling IQR)
    srng = np.random.default_rng(_seed_from(model, sample_idx, round(temperature, 4)))
    noise = srng.normal(0.0, 0.15 * spread * max(temperature, 1e-3), size=horizon)
    return base + bias + noise


# ---------------------------------------------------------------------------
# Raw generation (the ONLY place network I/O happens)
# ---------------------------------------------------------------------------
def generate_raw(
    model: str,
    prompt: str,
    *,
    backend: str = "ollama",
    temperature: float = 1.0,
    host: str = DEFAULT_HOST,
    num_predict: int = 256,
    timeout: float = 120.0,
    seed: Optional[int] = None,
    # mock-only context so the synthetic path can build a realistic series:
    codec: Optional[LLMTimeCodec] = None,
    context_values: Optional[Sequence[float]] = None,
    horizon: Optional[int] = None,
    sample_idx: int = 0,
) -> str:
    """Return the raw *text* continuation from ``model`` for ``prompt``.

    backend="ollama": POST to {host}/api/generate (network).
    backend="mock":   build a synthetic forecast and encode it with ``codec``,
                      so the returned string looks exactly like a real one.
    """
    if backend == "mock":
        if codec is None or context_values is None or horizon is None:
            raise ValueError("mock backend needs codec, context_values and horizon")
        fc = _mock_continuation(model, context_values, horizon, temperature, sample_idx)
        return codec.encode(fc)

    if backend != "ollama":
        raise ValueError(f"unknown backend {backend!r}")

    options: Dict[str, object] = {"temperature": float(temperature), "num_predict": int(num_predict)}
    if seed is not None:
        options["seed"] = int(seed)
    payload = {"model": model, "prompt": prompt, "stream": False, "options": options}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{host}/api/generate", data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        obj = json.loads(resp.read().decode("utf-8"))
    return obj.get("response", "")


# ---------------------------------------------------------------------------
# Forecast vectors
# ---------------------------------------------------------------------------
def forecast_one(
    model: str,
    codec: LLMTimeCodec,
    context_values: Sequence[float],
    horizon: int,
    *,
    backend: str = "ollama",
    temperature: float = 1.0,
    sample_idx: int = 0,
    host: str = DEFAULT_HOST,
    **kw,
) -> np.ndarray:
    """One model -> one horizon-length forecast vector (NaN-padded if short)."""
    prompt = build_prompt(codec, context_values)
    raw = generate_raw(
        model,
        prompt,
        backend=backend,
        temperature=temperature,
        host=host,
        codec=codec,
        context_values=context_values,
        horizon=horizon,
        sample_idx=sample_idx,
        **kw,
    )
    dec = codec.decode(raw, n=horizon)
    out = np.full(horizon, np.nan, dtype=np.float64)
    m = min(dec.size, horizon)
    out[:m] = dec[:m]
    return out


def committee_forecast(
    models: Sequence[str],
    codec: LLMTimeCodec,
    context_values: Sequence[float],
    horizon: int,
    *,
    backend: str = "ollama",
    temperature: float = 1.0,
    host: str = DEFAULT_HOST,
    **kw,
) -> Dict[str, object]:
    """Query every model once; return per-model vectors and their stacked matrix.

    Returns dict with ``matrix`` [n_models, horizon], ``models`` and
    ``per_model`` (name -> vector). NaNs mark values a model failed to emit.
    """
    per_model: Dict[str, np.ndarray] = {}
    rows: List[np.ndarray] = []
    for mdl in models:
        vec = forecast_one(
            mdl, codec, context_values, horizon,
            backend=backend, temperature=temperature, host=host, **kw,
        )
        per_model[mdl] = vec
        rows.append(vec)
    matrix = np.vstack(rows) if rows else np.empty((0, horizon))
    return {"models": list(models), "per_model": per_model, "matrix": matrix}


def resample_single_model(
    model: str,
    codec: LLMTimeCodec,
    context_values: Sequence[float],
    horizon: int,
    n_samples: int,
    *,
    backend: str = "ollama",
    temperature: float = 1.0,
    host: str = DEFAULT_HOST,
    base_seed: int = 0,
    **kw,
) -> np.ndarray:
    """Temperature-resampling null: ``n_samples`` stochastic forecasts from ONE
    model. Returns a [n_samples, horizon] matrix.

    Real backend varies the RNG seed per draw (with temperature>0); mock varies
    ``sample_idx`` so the seeded noise differs per draw.
    """
    rows: List[np.ndarray] = []
    for i in range(n_samples):
        vec = forecast_one(
            model, codec, context_values, horizon,
            backend=backend, temperature=temperature, host=host,
            sample_idx=i, seed=base_seed + i, **kw,
        )
        rows.append(vec)
    return np.vstack(rows) if rows else np.empty((0, horizon))


def _selftest() -> int:
    """Mock-backend smoke test: committee + resampling produce finite spread."""
    rng = np.random.default_rng(1)
    t = np.arange(200)
    series = 5.0 + 2.0 * np.sin(2 * np.pi * t / 24.0) + 0.1 * rng.standard_normal(t.size)
    ctx, horizon = series[:96], 12
    codec = LLMTimeCodec.fit(ctx, prec=3)

    models = ["llama3.1:8b", "qwen2.5:7b", "gemma2:9b", "mistral:7b"]
    out = committee_forecast(models, codec, ctx, horizon, backend="mock", temperature=0.7)
    mat = out["matrix"]
    assert mat.shape == (4, horizon)
    assert np.isfinite(mat).all(), "mock forecasts must be finite"
    cross_spread = float(np.mean(np.std(mat, axis=0)))

    res = resample_single_model(models[0], codec, ctx, horizon, 5, backend="mock", temperature=0.7)
    assert res.shape == (5, horizon)
    resamp_spread = float(np.mean(np.std(res, axis=0)))

    print(f"[committee] committee matrix {mat.shape}, finite=True")
    print(f"[committee] cross-model mean std   = {cross_spread:.4f}")
    print(f"[committee] resampling  mean std   = {resamp_spread:.4f}")
    print(f"[committee] model[0] forecast[:4]  = {np.round(mat[0, :4], 3).tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
