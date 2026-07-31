#!/usr/bin/env python3
"""Score the cumulative deletion panel and unlock Wave 2 only on preregistered G-D3 PASS."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MATRIX = ROOT / "results" / "matrices"
OUTDIR = ROOT / "results" / "deletion_wave1"
ENGINE = ROOT / "engine"
PY = sys.executable

# Five non-Qwen families and three Qwen families frozen before readout.
FAMILIES = {
    "llama1b": (12, "gate_llama1b_rome_cf_L12", "u1e0_llama1b_delete_refusal_L12", "+"),
    "gemma2b": (13, "gate_gemma2b_rome_cf_L13", "u1e0_gemma2b_delete_refusal_L13", "+"),
    "phi35": (16, "gate_phi35_rome_cf_L16", "u1e0_phi35_delete_refusal_L16", "+"),
    "mistral7b": (24, "gate_mistral7b_rome_cf_L24", "u1e0_mistral7b_delete_refusal_L24", "+"),
    "llama8b": (24, "gate_llama8b_rome_cf_L24", "u1e0_llama8b_delete_refusal_L24", "+"),
    "qwen15b": (21, "gate_qwen15b_rome_cf_L21", "u1e0_qwen15b_delete_refusal_L21", "-"),
    "qwen3b": (18, "gate_qwen3b_rome_cf_L18", "u1e0_qwen3b_delete_refusal_L18", "-"),
    "qwen7b": (21, "gate_qwen7b_rome_cf_L21", "u1e0_qwen7b_delete_refusal_L21", "-"),
}


def score_gate(family: str, seed: int, insertion_tag: str, deletion_tag: str) -> Path:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = OUTDIR / f"GATE_{family}_s{seed}.json"
    if out.exists():
        return out
    ins = MATRIX / f"{insertion_tag}_s{seed}.npz"
    dele = MATRIX / f"{deletion_tag}_s{seed}.npz"
    if not ins.exists() or not dele.exists():
        raise FileNotFoundError(f"missing pair: insertion={ins.exists()} {ins}; deletion={dele.exists()} {dele}")
    subprocess.run([
        PY, str(ROOT / "experiments" / "u1_deletion_gate.py"),
        "--del_npz", str(dele), "--ins_npz", str(ins), "--out", str(out),
    ], check=True)
    return out


def atlas_consistent(dc_rho, dc_perm_p, sign: str) -> bool:
    if dc_rho is None or dc_perm_p is None or float(dc_perm_p) >= 0.05:
        return False
    return float(dc_rho) >= 0.15 if sign == "+" else float(dc_rho) <= -0.15


def causal_llama8b() -> dict:
    from experiments.aggregate_g4_causal import masked_pair, within_probe_rhos
    per_seed = []
    for seed in (0, 1, 2):
        rome = MATRIX / f"u1e0_llama8b_delete_refusal_L24_s{seed}.npz"
        alpha = MATRIX / f"u1e0_llama8b_alphaHO_delete_refusal_L24_s{seed}.npz"
        if not rome.exists() or not alpha.exists():
            raise FileNotFoundError(f"missing Llama-8B causal pair for seed {seed}")
        with np.load(rome) as dr, np.load(alpha) as da:
            src = str(da["alpha_proj_source"]) if "alpha_proj_source" in da.files else "probes"
            if src != "holdout":
                raise ValueError(f"seed {seed}: AlphaEdit projector source={src}, expected holdout")
            got = masked_pair(dr, da, known=True, edit_ok=True)
            if got is None:
                raise ValueError(f"seed {seed}: no usable causal pairs")
            cos, dmg_rome, dmg_alpha = got
            removed = dmg_rome - dmg_alpha
            denom = float(np.mean(np.abs(dmg_rome)))
            reduction = float(np.mean(np.abs(dmg_rome) - np.abs(dmg_alpha)) / denom) if denom > 0 else np.nan
            rho = float(np.nanmean(within_probe_rhos(cos, removed)))
            per_seed.append({"seed": seed, "damage_reduction_fraction": reduction,
                             "rho_keycos_damage_removed": rho})
    reduction_mean = float(np.mean([x["damage_reduction_fraction"] for x in per_seed]))
    rho_mean = float(np.mean([x["rho_keycos_damage_removed"] for x in per_seed]))
    return {"per_seed": per_seed, "mean_damage_reduction_fraction": reduction_mean,
            "mean_rho_keycos_damage_removed": rho_mean,
            "PASS": reduction_mean >= 0.5 and rho_mean >= 0.15}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=1000,
                    help="reserved for compatibility; u1_deletion_gate uses its frozen default")
    args = ap.parse_args()
    family_rows = {}
    for family, (layer, ins_tag, del_tag, sign) in FAMILIES.items():
        seeds = []
        for seed in (0, 1, 2):
            path = score_gate(family, seed, ins_tag, del_tag)
            d = json.load(open(path))
            primary = d["arms"]["known=True|edit_ok=False"]
            seeds.append({"seed": seed, "verdict": d["VERDICT"],
                          "dc_rho": primary.get("dc_rho"),
                          "dc_perm_p": primary.get("dc_perm_p"),
                          "var_ratio": d["variance_receipt"].get("var_ratio")})
        n_expected = sum(atlas_consistent(x["dc_rho"], x["dc_perm_p"], sign) for x in seeds)
        family_rows[family] = {"layer": layer, "expected_sign": sign, "seeds": seeds,
                               "atlas_consistent_2_of_3": n_expected >= 2}

    non_qwen = [f for f in ("llama1b", "gemma2b", "phi35", "mistral7b", "llama8b")
                if family_rows[f]["atlas_consistent_2_of_3"]]
    qwen = [f for f in ("qwen15b", "qwen3b", "qwen7b")
            if family_rows[f]["atlas_consistent_2_of_3"]]
    causal = causal_llama8b()
    result = {
        "panel_frozen": {"non_qwen": ["llama1b", "gemma2b", "phi35", "mistral7b", "llama8b"],
                         "qwen": ["qwen15b", "qwen3b", "qwen7b"]},
        "families": family_rows,
        "non_qwen_consistent": non_qwen,
        "qwen_inverted": qwen,
        "G_D3_non_qwen_PASS": len(non_qwen) >= 4,
        "G_D3_qwen_PASS": len(qwen) == 3,
        "causal": causal,
    }
    result["G_D3_PASS"] = (result["G_D3_non_qwen_PASS"] and result["G_D3_qwen_PASS"]
                            and causal["PASS"])
    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = OUTDIR / "WAVE1_GD3_READOUT.json"
    tmp = out.with_suffix(".tmp")
    json.dump(result, open(tmp, "w"), indent=2)
    os.replace(tmp, out)
    receipt = ENGINE / "DELETION_WAVE1_GD3_PASS.ok"
    if result["G_D3_PASS"]:
        receipt.write_text(f"PASS source={out.relative_to(ROOT)}\n")
    elif receipt.exists():
        receipt.unlink()
    print(json.dumps(result, indent=2))
    return 0 if result["G_D3_PASS"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
