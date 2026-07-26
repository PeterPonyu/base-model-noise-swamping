"""llmtime.py — LLMTime-style digit encoding/decoding of a numeric series.

Implements the core representation from Gruver et al. 2023, "Large Language
Models Are Zero-Shot Time Series Forecasters": a numeric series is rescaled to
a fixed dynamic range, rounded to a fixed decimal precision, and each value is
written as its (space-separated) digit string so the tokenizer sees one token
per digit. Successive time steps are joined by a separator (default `" , "`).

Design goals
------------
- **Exact round-trip** up to the encoding precision: ``decode(encode(x))`` recovers
  ``x`` to within ``scale * 10**-prec`` (half a quantisation step).
- **Signed** scaled values are supported (a forecast may dip below the context
  min), via a leading ``-`` token.
- **Robust decode**: real LLM output is noisy, so decode pulls the first signed
  integer out of each value field with a regex and tolerates trailing prose.

No I/O and no network here — pure, deterministic string <-> ndarray maths, plus
a rolling-origin window builder used by the committee/forecasting pipeline.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass, asdict
from typing import List, Optional, Sequence, Tuple

import numpy as np

# One signed integer per value field, e.g. "- 1 2 3" or "4 0 7".
_INT_RE = re.compile(r"-?\s*\d[\d\s]*")


@dataclass
class LLMTimeCodec:
    """Fixed scaling / precision digit codec for a 1-D numeric series.

    Parameters
    ----------
    prec : int
        Number of decimal places kept in *scaled* space (digits after the
        implied point). Quantisation step in original units is
        ``scale * 10**-prec``.
    scale : float
        Divisor applied after subtracting ``offset``. Chosen by :meth:`fit`
        so typical scaled magnitudes are O(1) and the integer-digit count is
        bounded.
    offset : float
        Subtracted before scaling. :meth:`fit` sets it to the series minimum
        (minus an optional margin) so in-range values scale to >= 0.
    sep : str
        Separator inserted between successive time steps.
    """

    prec: int = 3
    scale: float = 1.0
    offset: float = 0.0
    sep: str = " , "

    # ---- fitting -----------------------------------------------------------
    @classmethod
    def fit(
        cls,
        values: Sequence[float],
        prec: int = 3,
        alpha: float = 0.95,
        margin: float = 0.0,
        sep: str = " , ",
    ) -> "LLMTimeCodec":
        """Fit offset/scale to ``values``.

        offset = min(values) - margin*range  (so values map to >= 0)
        scale  = alpha-quantile of (values - offset), floored to a positive
                 number so scaled magnitudes are ~O(1).
        """
        v = np.asarray(values, dtype=np.float64).ravel()
        if v.size == 0:
            raise ValueError("cannot fit codec on an empty series")
        vmin = float(np.min(v))
        vmax = float(np.max(v))
        rng = vmax - vmin
        offset = vmin - margin * rng
        centered = v - offset
        scale = float(np.quantile(centered, alpha))
        if not np.isfinite(scale) or scale <= 0:
            # constant / degenerate series: fall back to range or unit scale
            scale = rng if rng > 0 else 1.0
        return cls(prec=int(prec), scale=scale, offset=offset, sep=sep)

    # ---- helpers -----------------------------------------------------------
    @property
    def quantum(self) -> float:
        """Original-unit size of one least-significant encoded digit."""
        return self.scale * (10.0 ** (-self.prec))

    def _to_int(self, x: float) -> int:
        scaled = (float(x) - self.offset) / self.scale
        return int(np.round(scaled * (10 ** self.prec)))

    def _from_int(self, n: int) -> float:
        scaled = n / (10 ** self.prec)
        return scaled * self.scale + self.offset

    @staticmethod
    def _digits(n: int) -> str:
        neg = n < 0
        body = " ".join(str(abs(n)))
        return ("- " + body) if neg else body

    # ---- encode / decode ---------------------------------------------------
    def encode(self, values: Sequence[float]) -> str:
        """Encode a 1-D numeric series to a space-separated digit string."""
        v = np.asarray(values, dtype=np.float64).ravel()
        return self.sep.join(self._digits(self._to_int(x)) for x in v)

    def decode(self, text: str, n: Optional[int] = None,
               max_abs_scaled: float = 100.0) -> np.ndarray:
        """Decode ``text`` back to a float ndarray.

        Robust to trailing prose and to a partial final value. If ``n`` is
        given the result is truncated/kept to at most ``n`` values.

        ``max_abs_scaled`` is a plausibility bound in *scaled* space: in-range
        context values scale to ~[0, 1/alpha], so a token whose scaled
        magnitude exceeds the bound is a decode artifact (e.g. a comma-less
        digit run swallowed whole -> ~1e148 in original units), not a
        forecast. Such tokens are REJECTED, not returned; the count of
        rejections is exposed as ``self.n_rejected_last_decode`` so callers
        can log the rejection rate. Set ``max_abs_scaled=None`` to disable.
        """
        # split on the (comma) core of the separator so we tolerate spacing drift
        fields = re.split(r"[,;\n]", text)
        out: List[float] = []
        n_rejected = 0
        for field in fields:
            m = _INT_RE.search(field)
            if not m:
                continue
            token = m.group(0).replace(" ", "")
            if token in ("", "-"):
                continue
            try:
                value_int = int(token)
            except ValueError:
                continue
            if max_abs_scaled is not None and \
                    abs(value_int) / (10 ** self.prec) > max_abs_scaled:
                n_rejected += 1
                continue
            out.append(self._from_int(value_int))
            if n is not None and len(out) >= n:
                break
        self.n_rejected_last_decode = n_rejected
        return np.asarray(out, dtype=np.float64)

    # ---- (de)serialisation for configs/results -----------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "LLMTimeCodec":
        return cls(**{k: d[k] for k in ("prec", "scale", "offset", "sep") if k in d})


# ---- rolling-origin windows -----------------------------------------------
def rolling_windows(
    series: Sequence[float],
    context: int,
    horizon: int,
    stride: int = 1,
    max_windows: Optional[int] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Rolling-origin evaluation windows.

    Yields ``(context_slice, target_slice)`` pairs where the origin advances by
    ``stride``. context_slice has length ``context`` and immediately precedes a
    target_slice of length ``horizon``.
    """
    s = np.asarray(series, dtype=np.float64).ravel()
    n = s.size
    if context < 1 or horizon < 1:
        raise ValueError("context and horizon must be >= 1")
    windows: List[Tuple[np.ndarray, np.ndarray]] = []
    origin = context
    while origin + horizon <= n:
        ctx = s[origin - context:origin]
        tgt = s[origin:origin + horizon]
        windows.append((ctx.copy(), tgt.copy()))
        origin += stride
        if max_windows is not None and len(windows) >= max_windows:
            break
    return windows


def load_series_from_csv(path: str, column, max_rows: Optional[int] = None) -> np.ndarray:
    """Read one numeric column from a CSV file (stdlib only, no pandas).

    ``column`` may be a header name (str) or a 0-based index (int). Rows whose
    cell is empty or non-numeric are skipped.
    """
    vals: List[float] = []
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if isinstance(column, str):
            if header is None or column not in header:
                raise KeyError(f"column {column!r} not in header {header}")
            idx = header.index(column)
        else:
            idx = int(column)
        for row in reader:
            if idx >= len(row):
                continue
            cell = row[idx].strip()
            if cell == "":
                continue
            try:
                vals.append(float(cell))
            except ValueError:
                continue
            if max_rows is not None and len(vals) >= max_rows:
                break
    return np.asarray(vals, dtype=np.float64)


def _selftest() -> int:
    """Round-trip + rolling-window self-test on a synthetic sine series."""
    rng = np.random.default_rng(0)
    t = np.arange(600)
    series = 10.0 + 3.0 * np.sin(2 * np.pi * t / 48.0) + 0.05 * rng.standard_normal(t.size)

    codec = LLMTimeCodec.fit(series, prec=4, alpha=0.95)
    enc = codec.encode(series)
    dec = codec.decode(enc, n=series.size)
    assert dec.size == series.size, f"decoded {dec.size} != {series.size}"
    max_err = float(np.max(np.abs(dec - series)))
    tol = codec.quantum  # half a quantum is the bound; use a full quantum for slack
    assert max_err <= tol, f"round-trip error {max_err:.3e} > tol {tol:.3e}"

    # signed round-trip (values below the fitted offset)
    dip = series - 5.0
    dec_dip = codec.decode(codec.encode(dip), n=dip.size)
    assert float(np.max(np.abs(dec_dip - dip))) <= tol, "signed round-trip failed"

    wins = rolling_windows(series, context=96, horizon=24, stride=24)
    assert len(wins) > 0
    for ctx, tgt in wins:
        assert ctx.shape == (96,) and tgt.shape == (24,)
    # windows must be contiguous: target follows context exactly
    ctx0, tgt0 = wins[0]
    assert np.allclose(np.r_[ctx0, tgt0], series[:120])

    print(f"[llmtime] round-trip OK  max_err={max_err:.3e}  quantum={tol:.3e}")
    print(f"[llmtime] scale={codec.scale:.4g} offset={codec.offset:.4g} prec={codec.prec}")
    print(f"[llmtime] windows={len(wins)} (context=96 horizon=24 stride=24)")
    print(f"[llmtime] sample encode(series[:3])='{codec.encode(series[:3])}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
