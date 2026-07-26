#!/usr/bin/env python3
"""esr_band_analysis.py — P0 of the 2026-07-09 enhancement round (CPU-only, no GPU).

Assembles the EDITABLE-BAND picture across every model family from matrices already on
disk: each killgate npz stores per-edit `edit_ok` (real efficacy: post-edit argmax ==
new target), so esr-vs-depth curves for all families are a pure read. Motivated by the
NeoX-20B discovery (memory/neox20b-esr-depth-collapse-20260709.md): esr collapses
sharply past depth ~0.5-0.64 on NeoX-20B while Llama stays editable through 0.875 —
"each architecture has an editable band; the geometry law lives inside it".

Emits results/esr_band_table.json:
  rows: one per (model, editor, dataset, layer, seed) gate npz with
        esr, n_edits, depth_frac (layer / n_layers), source npz path
  curves: per (model, editor, dataset): depth_frac -> mean esr across seeds

Model n_layers registry is explicit (NOT guessed from filenames) — a file whose model
tag is not in the registry is reported under `unmapped` rather than silently dropped.

Usage: python experiments/esr_band_analysis.py [--matrices results/matrices] \
          [--out results/esr_band_table.json]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np

HARNESS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# model-tag (as it appears in gate_* filenames) -> total decoder layers
N_LAYERS = {
    "llama1b": 16,
    "llama1binstruct": 16,
    "llama3b": 28,
    "llama8b": 32,
    "qwen05b": 24,
    "qwen15b": 28,
    "qwen3b": 36,
    "gemma2b": 26,
    "phi35": 32,
    "gptj": 28,
    "gptneox20b": 44,
    "neox20b": 44,
    "pythia14b": 24,
    "pythia28b": 32,
}

# gate_<model>_<editor>_<dataset>_L<layer>_s<seed>.npz — dataset may contain '_'
PAT = re.compile(r"^gate_([a-z0-9]+)_([a-z_]+?)_([a-z0-9_]+)_L(\d+)_s(\d+)\.npz$")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrices", default=os.path.join(HARNESS, "results", "matrices"))
    ap.add_argument("--out", default=os.path.join(HARNESS, "results", "esr_band_table.json"))
    args = ap.parse_args()

    rows, unmapped, unreadable = [], [], []
    for path in sorted(glob.glob(os.path.join(args.matrices, "gate_*.npz"))):
        name = os.path.basename(path)
        m = PAT.match(name)
        if not m:
            unmapped.append({"file": name, "why": "filename pattern"})
            continue
        model, editor, dataset, layer, seed = (
            m.group(1), m.group(2), m.group(3), int(m.group(4)), int(m.group(5)))
        if model not in N_LAYERS:
            unmapped.append({"file": name, "why": f"model tag {model!r} not in registry"})
            continue
        try:
            with np.load(path) as d:
                if "edit_ok" not in d:
                    unmapped.append({"file": name, "why": "no edit_ok array"})
                    continue
                ok = d["edit_ok"].astype(float)
        except Exception as e:  # corrupt/partial npz — report, never crash the sweep
            unreadable.append({"file": name, "err": str(e)})
            continue
        rows.append({
            "model": model, "editor": editor, "dataset": dataset,
            "layer": layer, "seed": seed,
            "n_layers": N_LAYERS[model],
            "depth_frac": round(layer / N_LAYERS[model], 4),
            "esr": round(float(ok.mean()), 4),
            "n_edits": int(ok.size),
            "npz": os.path.relpath(path, HARNESS),
        })

    # curves: (model, editor, dataset) -> depth_frac -> mean esr across seeds
    curves: dict = {}
    for r in rows:
        key = f"{r['model']}|{r['editor']}|{r['dataset']}"
        curves.setdefault(key, {}).setdefault(str(r["depth_frac"]), []).append(r["esr"])
    curves_mean = {
        key: {df: {"mean_esr": round(float(np.mean(v)), 4), "n_seeds": len(v)}
              for df, v in sorted(pts.items(), key=lambda kv: float(kv[0]))}
        for key, pts in sorted(curves.items())
    }

    out = {
        "n_rows": len(rows),
        "rows": rows,
        "curves": curves_mean,
        "unmapped": unmapped,
        "unreadable": unreadable,
        "note": ("esr = mean(edit_ok) as stored by killgate_keygeom.py (argmax==target). "
                 "Smoke-sized probe npz are excluded by the gate_* glob (probes live in "
                 "results/probe_lr/, smokes in results/smoke_*/)."),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[esr_band] {len(rows)} rows -> {args.out} "
          f"({len(unmapped)} unmapped, {len(unreadable)} unreadable)")
    for key, pts in curves_mean.items():
        prof = "  ".join(f"{df}:{v['mean_esr']}" for df, v in pts.items())
        print(f"[esr_band] {key}: {prof}")


if __name__ == "__main__":
    main()
