#!/usr/bin/env python3
"""Evaluate the local Paper B curve and create a pre-B4 G-S3 receipt."""
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
    required = [(NEW, "qwen3b", 27), (NEW, "gemma2b", 19), (NEW, "phi35", 24)]
    missing = [str(base / f"{tag}_rome_L{layer}_s{s}" / "QS_phase1_table.json")
               for base, tag, layer in required for s in (0, 1, 2)
               if not (base / f"{tag}_rome_L{layer}_s{s}" / "QS_phase1_table.json").exists()]
    if missing:
        print(json.dumps({"status": "INCOMPLETE", "missing": missing}, indent=2)); return 3
    q15, _ = means(OLD, "qwen15b", 21)
    q3, q3s = means(NEW, "qwen3b", 27)
    gem, gems = means(NEW, "gemma2b", 19)
    phi, phis = means(NEW, "phi35", 24)
    l1, _ = means(OLD, "llama1b", 12)
    l3, _ = means(OLD, "llama3b", 24)
    qwen_monotone = q3 < q15
    llama_partial_monotone = l3 < l1
    family_sep = max(q3, gem, phi, l3) - min(q3, gem, phi, l3) > 0.0582

    rows = []
    for base, tag, layer in [(OLD, "llama1b", 12), (OLD, "llama3b", 24),
                              (OLD, "qwen15b", 21)] + required:
        for s in (0, 1, 2):
            cell = base / f"{tag}_rome_L{layer}_s{s}"
            surv, _ = arm_value(cell / "QS_phase1_table.json")
            with np.load(cell / "QS_phase1_raw.npz") as raw:
                signal = float(np.abs(raw["damage_fp32"]).mean())
                noise = float(np.abs(raw["base__nf4dq_full_model"]).mean())
            if signal > 0:
                rows.append((noise / signal, surv))
    if len(rows) >= 6:
        from experiments.merging_m0 import _spearman
        rho_nsr = float(_spearman(np.array([r[0] for r in rows]), np.array([r[1] for r in rows])))
    else:
        rho_nsr = None
    q3_pass = rho_nsr is not None and rho_nsr <= -0.3
    verdict = {
        "status": "PRE_B4_READOUT",
        "qwen15b_mean": q15, "qwen3b_mean": q3, "gemma2b_mean": gem,
        "phi35_mean": phi, "llama1b_mean": l1, "llama3b_mean": l3,
        "Q1_qwen_monotone": qwen_monotone,
        "Q1_llama_partial_monotone": llama_partial_monotone,
        "Q2_family_separation": family_sep, "Q3_nsr_rho": rho_nsr,
        "Q3_PASS": q3_pass,
        "G_S3_PASS": qwen_monotone and llama_partial_monotone and q3_pass,
        "seed_values": {"qwen3b": q3s, "gemma2b": gems, "phi35": phis},
        "note": "B4 remains the separately gated Llama-3.1-8B point; complete three-point Llama Q1 is evaluated only after B4."
    }
    OUT.parent.mkdir(parents=True, exist_ok=True); tmp = OUT.with_suffix(".tmp")
    json.dump(verdict, open(tmp, "w"), indent=2); os.replace(tmp, OUT)
    receipt = ROOT / "engine" / "PAPERB_CURVE_GS3_PASS.ok"
    if verdict["G_S3_PASS"]: receipt.write_text(f"PASS source={OUT.relative_to(ROOT)}\n")
    elif receipt.exists(): receipt.unlink()
    print(json.dumps(verdict, indent=2)); return 0 if verdict["G_S3_PASS"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
