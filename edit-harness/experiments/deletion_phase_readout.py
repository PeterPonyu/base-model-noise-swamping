#!/usr/bin/env python3
"""Evaluate frozen Phase-L deletion gates and create receipts only on literal PASS."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "deletion_phaseL"
ENG = ROOT / "engine"
FAMILIES = {"gemma2b": 13, "phi35": 16, "qwen3b": 18, "qwen15b": 21}


def main():
    gate_decidable = {}
    var_pass = {}
    text_pass = {}
    incomplete = []
    text_path = RES / "TEXT_BASELINE_phaseL.json"
    text = json.load(open(text_path)) if text_path.exists() else {}
    if not text_path.exists():
        incomplete.append(str(text_path))
    for fam, layer in FAMILIES.items():
        decidable = vv = tv = 0
        for seed in (0, 1, 2):
            p = RES / f"GATE_{fam}_L{layer}_s{seed}.json"
            if p.exists():
                d = json.load(open(p))
                verdict = str(d.get("VERDICT", "")).upper()
                decidable += int(
                    verdict and "UNDETERMINED" not in verdict and "FLAG_DEGENERATE" not in verdict
                )
                receipt = d.get("variance_receipt", {})
                ratio = receipt.get("var_ratio")
                vv += int(ratio is not None and float(ratio) >= 0.1)
            else:
                incomplete.append(str(p))
            tag = f"u1e0_{fam}_delete_refusal_L{layer}_s{seed}"
            t = text.get(tag, {})
            incr = t.get("incremental_r2_of_geometry")
            rho = t.get("partial_spearman_geometry_given_text")
            marginal = t.get("spearman_geometry_marginal")
            tv += int(incr is not None and rho is not None and marginal is not None
                      and float(incr) > 0 and abs(float(rho)) >= 0.15
                      and float(rho) * float(marginal) > 0)
        gate_decidable[fam] = decidable >= 2
        var_pass[fam] = vv >= 2
        text_pass[fam] = tv >= 2
    verdict = {
        "status": "INCOMPLETE" if incomplete else "COMPLETE",
        "missing": incomplete,
        "families": {f: {"deletion_gate_decidable": gate_decidable[f], "variance": var_pass[f],
                          "text_increment": text_pass[f]} for f in FAMILIES},
        "G_D1_PASS": sum(gate_decidable.values()) >= 3,
        "G_D2_PASS": sum(var_pass.values()) >= 3,
        "TEXT_PASS": sum(text_pass.values()) >= 3,
    }
    RES.mkdir(parents=True, exist_ok=True)
    out = RES / "PHASEL_GATE_READOUT.json"
    tmp = out.with_suffix(".tmp")
    json.dump(verdict, open(tmp, "w"), indent=2)
    os.replace(tmp, out)
    receipts = {
        "G_D1_PASS": ENG / "DELETION_PHASEL_GD1_PASS.ok",
        "G_D2_PASS": ENG / "DELETION_PHASEL_GD2_PASS.ok",
        "TEXT_PASS": ENG / "DELETION_PHASEL_TEXT_PASS.ok",
    }
    for key, path in receipts.items():
        if verdict[key]:
            path.write_text(f"PASS source={out.relative_to(ROOT)}\n")
        elif path.exists():
            path.unlink()
    print(json.dumps(verdict, indent=2))
    if incomplete:
        return 3
    return 0 if all(verdict[k] for k in receipts) else 2


if __name__ == "__main__":
    raise SystemExit(main())
