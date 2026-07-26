"""quant_survival_analyze.py — Paper B Track-1 Phase-1 aggregate analyzer.

Reads every `results/quant_survival/*_L*_s*/QS_phase1_table.json`, aggregates across seeds,
and emits a gate readout JSON plus a human-readable summary. CPU-only.

Output: results/quant_survival/aggregate/gate_readout.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional

HARNESS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HARNESS, "..", "results", "quant_survival")


def _mean_finite(vals: List[float]) -> Optional[float]:
    ok = [v for v in vals if v is not None and (isinstance(v, float) and (v == v))]
    if not ok:
        return None
    return float(sum(ok) / len(ok))


def _ci_finite(vals: List[float]) -> List[Optional[float]]:
    ok = sorted([v for v in vals if v is not None and (isinstance(v, float) and (v == v))])
    if len(ok) < 2:
        return [None, None]
    lo = ok[int(0.025 * len(ok))]
    hi = ok[int(0.975 * len(ok))]
    return [lo, hi]


def load_tables(root: str):
    tables = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f in ("QS_phase1_table.json", "QS_smoke_table.json"):
                p = os.path.join(dirpath, f)
                try:
                    d = json.load(open(p))
                except Exception as e:
                    print(f"[warn] skip unreadable {p}: {e}", file=sys.stderr)
                    continue
                if d.get("experiment") not in ("quant_survival_phase1", "quant_survival_smoke"):
                    continue
                d["_path"] = p
                tables.append(d)
    return tables


def aggregate(tables: List[Dict[str, Any]]):
    """Aggregate tables by (model, editor, layer)."""
    groups = defaultdict(list)
    for t in tables:
        key = (t["model"], t["editor"], t.get("layer"))
        groups[key].append(t)

    out = {}
    for (model, editor, layer), ts in groups.items():
        # sort by seed, keep only complete tables with arms
        ts = sorted([t for t in ts if t.get("arms")], key=lambda x: x.get("seed", 0))
        if not ts:
            continue

        # fp32 law gate
        fp32_within = [_mean_finite([t["mechanism_tie"].get("rho_keycos_damage_fp32_within_probe")]) for t in ts]
        fp32_within = [v for v in fp32_within if v is not None]
        c2_eligible = bool(fp32_within and min(fp32_within) >= 0.30)

        # arms aggregation
        arm_names = list(ts[0]["arms"].keys())
        arms_agg = {}
        for arm in arm_names:
            surv = [_mean_finite([t["arms"][arm].get("esr_survival_given_fp32_worked")]) for t in ts]
            surv = [v for v in surv if v is not None]
            drho = [_mean_finite([t["arms"][arm].get("delta_rho_vs_fp32_within_probe")]) for t in ts]
            drho = [v for v in drho if v is not None]
            drho_pooled = [_mean_finite([t["arms"][arm].get("delta_rho_vs_fp32_pooled")]) for t in ts]
            drho_pooled = [v for v in drho_pooled if v is not None]
            rank = [_mean_finite([t["arms"][arm].get("rho_damage_fp32_vs_arm_rank_survival")]) for t in ts]
            rank = [v for v in rank if v is not None]
            rank_base = [_mean_finite([t["arms"][arm].get(
                "rho_damage_fp32_vs_arm_rank_survival_base_subtracted")]) for t in ts]
            rank_base = [v for v in rank_base if v is not None]
            perm_p = [_mean_finite([t["arms"][arm].get("permutation_null_p_pooled")]) for t in ts]
            perm_p = [v for v in perm_p if v is not None]
            mean_esr = [_mean_finite([t["arms"][arm].get("mean_esr")]) for t in ts]
            mean_esr = [v for v in mean_esr if v is not None]
            arms_agg[arm] = {
                "n_seeds": len(surv),
                "esr_survival_given_fp32_worked_mean": _mean_finite(surv),
                "esr_survival_given_fp32_worked_ci95": _ci_finite(surv),
                "delta_rho_vs_fp32_within_mean": _mean_finite(drho),
                "delta_rho_vs_fp32_within_ci95": _ci_finite(drho),
                "delta_rho_vs_fp32_pooled_mean": _mean_finite(drho_pooled),
                "delta_rho_vs_fp32_pooled_ci95": _ci_finite(drho_pooled),
                "rho_damage_fp32_vs_arm_rank_survival_mean": _mean_finite(rank),
                "rho_damage_fp32_vs_arm_rank_survival_ci95": _ci_finite(rank),
                "rho_damage_fp32_vs_arm_rank_survival_base_subtracted_mean": _mean_finite(rank_base),
                "rho_damage_fp32_vs_arm_rank_survival_base_subtracted_ci95": _ci_finite(rank_base),
                "permutation_null_p_pooled_mean": _mean_finite(perm_p),
                "permutation_null_p_pooled_max": max(perm_p) if perm_p else None,
                "mean_esr_mean": _mean_finite(mean_esr),
            }

        # C3 aggregation
        c3_agg = {}
        schemes = list(ts[0].get("bin_width_mechanism_C3", {}).keys())
        for scheme in schemes:
            F_above = [_mean_finite([t["bin_width_mechanism_C3"][scheme].get("F_above_bin")]) for t in ts]
            F_above = [v for v in F_above if v is not None]
            med = [_mean_finite([t["bin_width_mechanism_C3"][scheme].get("median_ratio")]) for t in ts]
            med = [v for v in med if v is not None]
            rfunc = [_mean_finite([t["bin_width_mechanism_C3"][scheme].get("r_func_mean")]) for t in ts]
            rfunc = [v for v in rfunc if v is not None]
            rparam = [_mean_finite([t["bin_width_mechanism_C3"][scheme].get("r_param_mean")]) for t in ts]
            rparam = [v for v in rparam if v is not None]
            c3_agg[scheme] = {
                "F_above_mean": _mean_finite(F_above),
                "median_ratio_mean": _mean_finite(med),
                "r_func_mean": _mean_finite(rfunc),
                "r_param_mean": _mean_finite(rparam),
            }

        out[f"{os.path.basename(model)}_{editor}_L{layer}"] = {
            "model": model, "editor": editor, "layer": layer,
            "n_seeds": len(ts),
            "c2_eligible": c2_eligible,
            "fp32_rho_within_probe_mean": _mean_finite(fp32_within),
            "fp32_rho_within_probe_ci95": _ci_finite(fp32_within),
            "arms": arms_agg,
            "c3": c3_agg,
        }
    return out


def gate_readout(agg: Dict[str, Any]):
    """Apply the frozen prereg thresholds and emit gate status."""
    # K1: C2 geometry-ranking survival on validated-law cells
    # K2: esr survival >= 0.8 at 4-bit for ROME on both primary models
    # K3: M-concentration holds
    # K4: bridge handled separately (Track-2)
    readout = {
        "thresholds": {
            "fp32_law_gate": 0.30,
            "esr_survival_4bit": 0.80,
            "esr_survival_8bit": 0.90,
            "delta_rho_tolerance": 0.15,
            "rank_survival_4bit": 0.85,
            "rank_survival_8bit": 0.95,
            "median_ratio_concentration": 1.0,
        },
        "cells": agg,
        "gates": {},
    }

    # K1: only on ROME cells that are c2_eligible
    k1_cells = [k for k, v in agg.items() if v["editor"] == "rome" and v["c2_eligible"]]
    k1_failures = []
    k1_arm_status = {}
    for k in k1_cells:
        v = agg[k]
        k1_arm_status[k] = {}
        for arm, a in v["arms"].items():
            scheme, locality = arm.split("_", 1)
            rank_threshold = 0.85 if scheme == "nf4dq" and locality == "full_model" else 0.95
            arm_failures = []
            for label, value, ci in (
                ("within Δρ", a.get("delta_rho_vs_fp32_within_mean"),
                 a.get("delta_rho_vs_fp32_within_ci95")),
                ("pooled Δρ", a.get("delta_rho_vs_fp32_pooled_mean"),
                 a.get("delta_rho_vs_fp32_pooled_ci95")),
            ):
                if value is None:
                    arm_failures.append(f"{label} missing")
                elif abs(value) > 0.15:
                    arm_failures.append(f"|{label}|={abs(value):.3f} > 0.15")
                elif ci and all(x is not None for x in ci) and max(abs(ci[0]), abs(ci[1])) > 0.15:
                    arm_failures.append(
                        f"{label} seed range=[{ci[0]:.3f},{ci[1]:.3f}] leaves ±0.15")
            for label, value, ci in (
                ("rank survival", a.get("rho_damage_fp32_vs_arm_rank_survival_mean"),
                 a.get("rho_damage_fp32_vs_arm_rank_survival_ci95")),
                ("base-subtracted rank survival", a.get(
                    "rho_damage_fp32_vs_arm_rank_survival_base_subtracted_mean"), a.get(
                    "rho_damage_fp32_vs_arm_rank_survival_base_subtracted_ci95")),
            ):
                if value is None:
                    arm_failures.append(f"{label} missing")
                elif value < rank_threshold:
                    arm_failures.append(f"{label}={value:.3f} < {rank_threshold:.2f}")
                elif ci and ci[0] is not None and ci[0] < rank_threshold:
                    arm_failures.append(
                        f"{label} seed-range lower={ci[0]:.3f} < {rank_threshold:.2f}")
            perm_p = a.get("permutation_null_p_pooled_max")
            if perm_p is None:
                arm_failures.append("permutation p missing")
            elif perm_p >= 0.01:
                arm_failures.append(f"permutation p(max)={perm_p:.3f} >= 0.01")
            k1_arm_status[k][arm] = {
                "status": "FAIL" if arm_failures else "PASS",
                "rank_threshold": rank_threshold,
                "failures": arm_failures,
            }
            k1_failures.extend(f"{k}/{arm}: {msg}" for msg in arm_failures)
    readout["gates"]["K1_geometry_ranking_survival"] = {
        "status": "FAIL" if k1_failures else "PASS",
        "n_cells_evaluated": len(k1_cells),
        "failures": k1_failures,
        "arm_status": k1_arm_status,
        "note": ("A single-cell full-model NF4 failure narrows C2 to INT8 plus edited-layer "
                 "NF4 across both cells and full-model NF4 where supported; it does not kill "
                 "the complete ranking-survival result."),
    }

    # K2: esr survival at 4-bit for the preregistered C1 primary models
    # (Llama-3.2-1B and Qwen2.5-1.5B; Llama-3.2-3B is the mandatory C2 extension).
    k2_primary = [k for k, v in agg.items() if v["editor"] == "rome" and
                  any(x in k for x in ("Llama-3.2-1B", "Qwen2.5-1.5B"))]
    k2_failures = []
    for k in k2_primary:
        v = agg[k]
        for arm, a in v["arms"].items():
            if arm.startswith("nf4dq"):
                s = a.get("esr_survival_given_fp32_worked_mean")
                if s is not None and s < 0.80:
                    k2_failures.append(f"{k}/{arm}: surv={s:.3f} < 0.80")
    readout["gates"]["K2_esr_survival_4bit"] = {
        "status": "FAIL" if k2_failures else "PASS",
        "n_cells_evaluated": len(k2_primary),
        "cells_evaluated": sorted(k2_primary),
        "failures": k2_failures,
        "note": ("K2 covers the preregistered ROME threshold component on Llama-3.2-1B "
                 "and Qwen2.5-1.5B. The stronger C1 prediction that MEMIT/AlphaEdit are "
                 "no less robust than ROME is reported separately."),
    }

    # K3: amended channel-scale x NF4 minimum-gap axis is not available in legacy NPZ.
    # Preserve legacy/local-gap observations as sensitivity diagnostics, but do not
    # adjudicate the preregistered primary direction without a targeted rerun.
    k3_legacy = []
    for k, v in agg.items():
        if v["editor"] != "rome":
            continue
        c3 = v.get("c3", {})
        if "nf4dq" in c3:
            med = c3["nf4dq"].get("median_ratio_mean")
            F_above = c3["nf4dq"].get("F_above_mean")
            if med is not None:
                k3_legacy.append(f"{k}: median_ratio={med:.3f}, F_above={F_above:.3f}")
    readout["gates"]["K3_M_concentration"] = {
        "status": "UNADJUDICATED",
        "legacy_sensitivity_observations": k3_legacy,
        "failures": [],
        "note": "Legacy/local-gap ratios are sensitivity observations only; amended channel-scale x NF4 minimum-gap K3 requires targeted rerun."
    }

    return readout


def print_summary(readout: Dict[str, Any]):
    print("\n=== Paper B Track-1 Phase-1 aggregate gate readout ===")
    th = readout["thresholds"]
    print(f"Thresholds: {json.dumps(th, indent=2)}")
    for gname, g in readout["gates"].items():
        print(f"\n{gname}: {g['status']}")
        if g.get("failures"):
            print("  failures:")
            for f in g["failures"]:
                print(f"    - {f}")

    print("\n--- Cell summary ---")
    for name, v in readout["cells"].items():
        print(f"\n{name}: n_seeds={v['n_seeds']} c2_eligible={v['c2_eligible']} "
              f"fp32_within={v['fp32_rho_within_probe_mean']:.3f}")
        for arm, a in v["arms"].items():
            def _fmt(x):
                return f"{x:.3f}" if x is not None else "None"
            print(f"  {arm}: esr_surv={_fmt(a.get('esr_survival_given_fp32_worked_mean'))} "
                  f"Δρ={_fmt(a.get('delta_rho_vs_fp32_within_mean'))} "
                  f"rank_surv={_fmt(a.get('rho_damage_fp32_vs_arm_rank_survival_mean'))}")
        for scheme, c in v.get("c3", {}).items():
            def _fmt2(x):
                return f"{x:.3f}" if x is not None else "None"
            print(f"  [C3 {scheme}] median_ratio={_fmt2(c.get('median_ratio_mean'))} "
                  f"F_above={_fmt2(c.get('F_above_mean'))}")


def main():
    ap = argparse.ArgumentParser(description="Aggregate Paper B Track-1 Phase-1 results.")
    ap.add_argument("--root", default=ROOT, help="results/quant_survival root")
    ap.add_argument("--out", default=os.path.join(ROOT, "aggregate", "gate_readout.json"),
                    help="output JSON path")
    ap.add_argument("--summary", action="store_true", help="print human summary")
    args = ap.parse_args()

    tables = load_tables(args.root)
    if not tables:
        print(f"[warn] no tables found under {args.root}", file=sys.stderr)
        # still write a minimal readout
        agg = {}
    else:
        agg = aggregate(tables)

    readout = gate_readout(agg)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(readout, f, indent=2)
    print(f"wrote {args.out}")
    if args.summary:
        print_summary(readout)


if __name__ == "__main__":
    main()
