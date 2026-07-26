"""run_committee.py — P4 pilot entrypoint.

Consumes a config JSON (e.g. configs/ett_pilot.json) and writes a full
calibration report to results/<id>.json. This is the command a fission-engine
queue job invokes (see make_jobs.py).

Pipeline per config
-------------------
1. Load the series: a CSV column if ``data.csv`` exists on disk, else a
   synthetic sine fallback (keeps CPU/mock dry-runs self-contained).
2. Build rolling-origin windows (context/horizon/stride).
3. For each window:
     - fit an LLMTime codec on the context,
     - query the cross-architecture committee once per model -> matrix,
     - draw the temperature-resampling null from one model,
     - record cross-model disagreement, resampling disagreement, the consensus
       point forecast, and the signed error vs truth.
4. Run the full calibration report (Spearman + bootstrap CI, PICP, CRPS, and the
   cross-vs-resampling kill-gate).
5. Write results/<id>.json.

Backend: ``config.backend`` ("ollama" for real GPU jobs). Pass ``--backend mock``
to force the synthetic backend for a CPU dry-run without Ollama.

Usage
-----
    python3 run_committee.py configs/ett_pilot.json
    python3 run_committee.py configs/ett_pilot.json --backend mock
    python3 run_committee.py configs/ett_pilot.json --backend mock --out results/foo.json
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from typing import Dict, List

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import calibrate  # noqa: E402
import committee  # noqa: E402
import disagreement as dis  # noqa: E402
from llmtime import LLMTimeCodec, load_series_from_csv, rolling_windows  # noqa: E402

RESULTS_DIR = os.path.join(HERE, "results")


def _synthetic_series(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return (
        10.0
        + 4.0 * np.sin(2 * np.pi * t / 96.0)
        + 1.5 * np.sin(2 * np.pi * t / 24.0)
        + 0.3 * rng.standard_normal(n)
    )


def load_series(cfg: Dict) -> np.ndarray:
    data = cfg.get("data", {})
    csv_rel = data.get("csv")
    column = data.get("column", "OT")
    max_rows = data.get("max_rows")
    if csv_rel:
        csv_path = csv_rel if os.path.isabs(csv_rel) else os.path.join(HERE, csv_rel)
        if os.path.isfile(csv_path):
            series = load_series_from_csv(csv_path, column, max_rows=max_rows)
            if series.size:
                print(f"[run] loaded {series.size} points from {csv_path} col={column}")
                return series
        print(f"[run] CSV unavailable ({csv_path}); using synthetic fallback")
    n = int(data.get("synthetic_n", 1500))
    series = _synthetic_series(n, seed=int(data.get("seed", 0)))
    print(f"[run] synthetic sine series, n={series.size}")
    return series


def run(cfg: Dict, backend_override: str = None) -> Dict:
    run_id = cfg.get("id", "p4_run")
    backend = backend_override or cfg.get("backend", "ollama")
    models: List[str] = cfg["models"]
    context = int(cfg.get("context", 96))
    horizon = int(cfg.get("horizon", 24))
    stride = int(cfg.get("stride", horizon))
    max_windows = cfg.get("max_windows")
    prec = int(cfg.get("prec", 3))
    temperature = float(cfg.get("temperature", 0.8))
    n_resamples = int(cfg.get("n_resamples", 5))
    metric = cfg.get("disagreement_metric", "std")
    level = float(cfg.get("picp_level", 0.90))
    n_boot = int(cfg.get("n_boot", 2000))
    host = cfg.get("host", committee.DEFAULT_HOST)
    null_model = cfg.get("null_model", models[0])

    series = load_series(cfg)
    windows = rolling_windows(series, context, horizon, stride=stride, max_windows=max_windows)
    if not windows:
        raise ValueError("no rolling windows — check context/horizon/series length")
    print(f"[run] {len(windows)} windows | models={len(models)} | backend={backend} "
          f"| context={context} horizon={horizon}")

    cross_dis: List[float] = []
    resamp_dis: List[float] = []
    signed_err: List[float] = []
    per_window: List[Dict] = []

    for wi, (ctx, tgt) in enumerate(windows):
        codec = LLMTimeCodec.fit(ctx, prec=prec)
        comm = committee.committee_forecast(
            models, codec, ctx, horizon,
            backend=backend, temperature=temperature, host=host,
        )
        matrix = comm["matrix"]
        d_cross = dis.cross_model_disagreement(matrix, metric=metric)

        resamp = committee.resample_single_model(
            null_model, codec, ctx, horizon, n_resamples,
            backend=backend, temperature=temperature, host=host, base_seed=1000 + wi,
        )
        d_resamp = dis.resampling_disagreement(resamp, metric=metric)

        point = dis.committee_point_forecast(matrix, how="median")
        resid = float(np.nanmean(tgt - point))          # signed residual (per window)
        abs_err = dis.forecast_abs_error(point, tgt)     # scalar |error|

        cross_dis.append(d_cross)
        resamp_dis.append(d_resamp)
        signed_err.append(resid)
        per_window.append({
            "window": wi,
            "cross_disagreement": d_cross,
            "resampling_disagreement": d_resamp,
            "signed_error": resid,
            "abs_error": abs_err,
        })

    cross_arr = np.asarray(cross_dis)
    resamp_arr = np.asarray(resamp_dis)
    signed_arr = np.asarray(signed_err)

    report = calibrate.calibration_report(
        cross_arr, resamp_arr, signed_arr, level=level, n_boot=n_boot,
        seed=int(cfg.get("seed", 0)),
    )

    result = {
        "id": run_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "backend": backend,
        "config": {
            "models": models, "null_model": null_model, "context": context,
            "horizon": horizon, "stride": stride, "prec": prec,
            "temperature": temperature, "n_resamples": n_resamples,
            "disagreement_metric": metric, "picp_level": level, "n_boot": n_boot,
        },
        "n_windows": len(windows),
        "report": report,
        "per_window": per_window,
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = cfg.get("_out") or os.path.join(RESULTS_DIR, f"{run_id}.json")
    with open(out, "w") as fh:
        json.dump(result, fh, indent=2)
    result["_path"] = out
    return result


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description="P4 committee-disagreement calibration run")
    ap.add_argument("config", help="path to a config JSON")
    ap.add_argument("--backend", choices=["ollama", "mock"], default=None,
                    help="override config backend (use 'mock' for CPU dry-run)")
    ap.add_argument("--out", default=None, help="explicit results path")
    args = ap.parse_args(argv)

    with open(args.config) as fh:
        cfg = json.load(fh)
    if args.out:
        cfg["_out"] = args.out

    res = run(cfg, backend_override=args.backend)
    rep = res["report"]
    cmp = rep["comparison"]
    print(f"[run] wrote {res['_path']}")
    print(f"[run] windows={res['n_windows']} backend={res['backend']}")
    print(f"[run] spearman  cross={rep['spearman_cross']['rho']:.3f} "
          f"(p={rep['spearman_cross']['pvalue']:.2e})  "
          f"resamp={rep['spearman_resampling']['rho']:.3f}")
    print(f"[run] bootstrap cross rho 95% CI="
          f"[{rep['bootstrap_spearman_cross']['lo95']:.3f}, "
          f"{rep['bootstrap_spearman_cross']['hi95']:.3f}]")
    print(f"[run] PICP@{rep['picp_width_cross']['nominal']:.2f}="
          f"{rep['picp_width_cross']['picp']:.3f} "
          f"width={rep['picp_width_cross']['mean_width']:.3f}  "
          f"CRPS={rep['crps_cross']['crps']:.4f}")
    print(f"[run] KILL-GATE delta(cross-resamp)={cmp['delta_point']:.3f} "
          f"CI=[{cmp['delta_ci']['lo95']:.3f}, {cmp['delta_ci']['hi95']:.3f}]  "
          f"gate_pass={cmp['gate_pass']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
