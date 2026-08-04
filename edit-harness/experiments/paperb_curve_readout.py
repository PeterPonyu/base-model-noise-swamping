#!/usr/bin/env python3
"""Evaluate the local Paper B curve and create a pre-B4 G-S3 receipt.

Schema version 2: adds per-point rows with noise_to_signal, nf4_rank_survival, and provenance.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
NEW = ROOT / "results" / "quant_survival_curve"
OLD = ROOT / "results" / "quant_survival"
OUT = ROOT / "results" / "quant_survival" / "aggregate" / "curve_local_readout.json"

SCHEMA_VERSION = 2


def sha256_file(path):
    """Return SHA256 hex digest of file."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def arm_value(path):
    d = json.load(open(path))
    arm = d["arms"]["nf4dq_full_model"]
    return float(arm["rho_damage_fp32_vs_arm_rank_survival"]), d


def means(base, tag, layer):
    vals = []
    for s in (0, 1, 2):
        cell = base / f"{tag}_rome_L{layer}_s{s}"
        vals.append(arm_value(cell / "QS_phase1_table.json")[0])
    return float(np.mean(vals)), vals


def main():
    # Enumerate all 9 required cells
    required = [(NEW, "qwen3b", 27), (NEW, "gemma2b", 19), (NEW, "phi35", 24)]

    # Check which cells exist
    missing_cells = []
    for base, tag, layer in required:
        for s in (0, 1, 2):
            cell_dir = base / f"{tag}_rome_L{layer}_s{s}"
            table_path = cell_dir / "QS_phase1_table.json"
            raw_path = cell_dir / "QS_phase1_raw.npz"
            if not (table_path.exists() and raw_path.exists()):
                missing_cells.append(str(table_path))

    # If any required cells missing, return INCOMPLETE
    if missing_cells:
        print(json.dumps({
            "status": "INCOMPLETE",
            "schema_version": SCHEMA_VERSION,
            "missing": missing_cells,
            "note": "Cannot compute gates until all 9 cells (qwen3b/gemma2b/phi35 × L27/L19/L24 × s0/s1/s2) complete"
        }, indent=2))
        return 3

    # All cells exist - compute legacy means for gates
    q15, _ = means(OLD, "qwen15b", 21)
    q3, q3s = means(NEW, "qwen3b", 27)
    gem, gems = means(NEW, "gemma2b", 19)
    phi, phis = means(NEW, "phi35", 24)
    l1, _ = means(OLD, "llama1b", 12)
    l3, _ = means(OLD, "llama3b", 24)

    qwen_monotone = q3 < q15
    llama_partial_monotone = l3 < l1
    family_sep = max(q3, gem, phi, l3) - min(q3, gem, phi, l3) > 0.0582

    # Build per-point rows from all available cells (including OLD for full curve)
    rows = []
    all_cells = [
        (OLD, "llama1b", 12),
        (OLD, "llama3b", 24),
        (OLD, "qwen15b", 21)
    ] + required

    for base, tag, layer in all_cells:
        for s in (0, 1, 2):
            cell_dir = base / f"{tag}_rome_L{layer}_s{s}"
            table_path = cell_dir / "QS_phase1_table.json"
            raw_path = cell_dir / "QS_phase1_raw.npz"

            if not (table_path.exists() and raw_path.exists()):
                continue

            surv, _ = arm_value(table_path)
            with np.load(raw_path) as raw:
                signal = float(np.abs(raw["damage_fp32"]).mean())
                noise = float(np.abs(raw["base__nf4dq_full_model"]).mean())

            if signal > 0:
                noise_to_signal = noise / signal
                row = {
                    "model": tag,
                    "layer": int(layer),
                    "seed": int(s),
                    "noise_to_signal": float(noise_to_signal),
                    "nf4_rank_survival": float(surv),
                    "source_table": str(table_path.relative_to(ROOT)),
                    "source_raw": str(raw_path.relative_to(ROOT)),
                    "table_sha256": sha256_file(table_path),
                    "raw_sha256": sha256_file(raw_path)
                }
                rows.append(row)

    # Compute Spearman correlation if enough points
    if len(rows) >= 6:
        from experiments.merging_m0 import _spearman
        nsr_vals = np.array([r["noise_to_signal"] for r in rows])
        surv_vals = np.array([r["nf4_rank_survival"] for r in rows])
        rho_nsr = float(_spearman(nsr_vals, surv_vals))
    else:
        rho_nsr = None

    q3_pass = rho_nsr is not None and rho_nsr <= -0.3

    verdict = {
        "status": "PRE_B4_READOUT",
        "schema_version": SCHEMA_VERSION,
        "qwen15b_mean": q15,
        "qwen3b_mean": q3,
        "gemma2b_mean": gem,
        "phi35_mean": phi,
        "llama1b_mean": l1,
        "llama3b_mean": l3,
        "Q1_qwen_monotone": qwen_monotone,
        "Q1_llama_partial_monotone": llama_partial_monotone,
        "Q2_family_separation": family_sep,
        "Q3_nsr_rho": rho_nsr,
        "Q3_PASS": q3_pass,
        "G_S3_PASS": qwen_monotone and llama_partial_monotone and q3_pass,
        "seed_values": {
            "qwen3b": q3s,
            "gemma2b": gems,
            "phi35": phis
        },
        "curve_points": rows,
        "note": "B4 remains the separately gated Llama-3.1-8B point; complete three-point Llama Q1 is evaluated only after B4."
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".tmp")
    json.dump(verdict, open(tmp, "w"), indent=2)
    os.replace(tmp, OUT)

    receipt = ROOT / "engine" / "PAPERB_CURVE_GS3_PASS.ok"
    if verdict["G_S3_PASS"]:
        receipt.write_text(f"PASS source={OUT.relative_to(ROOT)}\n")
    elif receipt.exists():
        receipt.unlink()

    print(json.dumps(verdict, indent=2))
    return 0 if verdict["G_S3_PASS"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
