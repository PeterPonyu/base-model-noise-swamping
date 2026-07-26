"""quant_survival_schema.py — Paper B schema validation utility.

Validates the on-disk JSON/JSONL/NPZ artifacts for smoke, phase1, and aggregate readout.
This catches schema drift between the code that writes tables and the analysis/scripts that
read them. CPU-only.
"""
from __future__ import annotations

import json
import os
import sys
from typing import List, Tuple

HARNESS = os.path.dirname(os.path.abspath(__file__))

SMOKE_REQUIRED_TABLE_KEYS = {
    "experiment", "schema_version", "model", "layer", "editor", "n_edits", "n_probes",
    "seed", "schemes", "damage_metric_note", "esr", "mechanism_tie", "arms",
    "frozen_prediction_readout",
}
SMOKE_REQUIRED_ARM_KEYS = {
    "locality", "scheme", "mean_esr", "esr_survival_given_fp32_worked",
    "rho_keycos_damage_pooled", "rho_keycos_damage_within_probe",
    "delta_rho_vs_fp32_pooled", "rho_damage_fp32_vs_arm_rank_survival",
}
SMOKE_REQUIRED_RAW_KEYS = {"COS", "damage_fp32", "edit_ok_fp32"}

PHASE1_REQUIRED_TABLE_KEYS = {
    "experiment", "schema_version", "model", "layer", "editor", "n_edits", "n_probes",
    "seed", "schemes", "codec", "damage_metric_note", "esr", "mechanism_tie", "arms",
    "bin_width_mechanism_C3", "c2_scope", "quant_note",
}
PHASE1_REQUIRED_ARM_KEYS = {
    "locality", "scheme", "mean_esr", "esr_survival_given_fp32_worked",
    "rho_keycos_damage_pooled", "rho_keycos_damage_pooled_base_subtracted",
    "rho_keycos_damage_within_probe", "rho_keycos_damage_within_probe_base_subtracted",
    "delta_rho_vs_fp32_pooled", "delta_rho_vs_fp32_within_probe",
    "rho_damage_fp32_vs_arm_rank_survival", "rho_damage_fp32_vs_arm_rank_survival_base_subtracted",
    "permutation_null_p_pooled", "rho_pooled_ci95_bootstrap_edits",
}
PHASE1_REQUIRED_C3_KEYS = {"F_above_bin", "median_ratio", "p90_ratio", "r_func_mean", "r_param_mean"}
PHASE1_REQUIRED_RAW_KEYS = {"COS", "damage_fp32", "edit_ok_fp32"}


def validate_npz(path: str, required_keys: set) -> Tuple[bool, List[str]]:
    try:
        import numpy as np
        a = np.load(path)
    except Exception as e:
        return False, [f"unreadable npz: {e}"]
    missing = required_keys - set(a.files)
    if missing:
        return False, [f"raw npz missing {missing}"]
    return True, []


def validate_smoke_table(path: str) -> Tuple[bool, List[str]]:
    try:
        d = json.load(open(path))
    except Exception as e:
        return False, [f"unparseable: {e}"]
    missing = SMOKE_REQUIRED_TABLE_KEYS - set(d.keys())
    if missing:
        return False, [f"missing table keys {missing}"]
    for arm_name, arm in d["arms"].items():
        missing = SMOKE_REQUIRED_ARM_KEYS - set(arm.keys())
        if missing:
            return False, [f"arm {arm_name} missing keys {missing}"]
    return True, []


def validate_phase1_table(path: str) -> Tuple[bool, List[str]]:
    try:
        d = json.load(open(path))
    except Exception as e:
        return False, [f"unparseable: {e}"]
    missing = PHASE1_REQUIRED_TABLE_KEYS - set(d.keys())
    if missing:
        return False, [f"missing table keys {missing}"]
    for arm_name, arm in d["arms"].items():
        missing = PHASE1_REQUIRED_ARM_KEYS - set(arm.keys())
        if missing:
            return False, [f"arm {arm_name} missing keys {missing}"]
    for scheme, c in d["bin_width_mechanism_C3"].items():
        missing = PHASE1_REQUIRED_C3_KEYS - set(c.keys())
        if missing:
            return False, [f"C3 scheme {scheme} missing keys {missing}"]
    return True, []


def validate_readout(path: str) -> Tuple[bool, List[str]]:
    try:
        d = json.load(open(path))
    except Exception as e:
        return False, [f"unparseable: {e}"]
    for k in ("thresholds", "cells", "gates"):
        if k not in d:
            return False, [f"missing readout key {k}"]
    for cell_name, cell in d["cells"].items():
        for k in ("n_seeds", "c2_eligible", "fp32_rho_within_probe_mean", "arms", "c3"):
            if k not in cell:
                return False, [f"cell {cell_name} missing {k}"]
    return True, []


def validate_all(root: str = os.path.join(HARNESS, "..", "results")):
    failures = []
    counts = {"smoke": 0, "phase1": 0, "readout": 0}
    for dirpath, _, files in os.walk(root):
        for f in files:
            p = os.path.join(dirpath, f)
            if f == "QS_smoke_table.json":
                counts["smoke"] += 1
                ok, errs = validate_smoke_table(p)
                if not ok:
                    failures.append((p, errs))
                raw = os.path.join(dirpath, "QS_smoke_raw.npz")
                if os.path.exists(raw):
                    ok, errs = validate_npz(raw, SMOKE_REQUIRED_RAW_KEYS)
                    if not ok:
                        failures.append((raw, errs))
            elif f == "QS_phase1_table.json":
                counts["phase1"] += 1
                ok, errs = validate_phase1_table(p)
                if not ok:
                    failures.append((p, errs))
                raw = os.path.join(dirpath, "QS_phase1_raw.npz")
                if os.path.exists(raw):
                    ok, errs = validate_npz(raw, PHASE1_REQUIRED_RAW_KEYS)
                    if not ok:
                        failures.append((raw, errs))
            elif f == "gate_readout.json":
                counts["readout"] += 1
                ok, errs = validate_readout(p)
                if not ok:
                    failures.append((p, errs))
    return counts, failures


def selftest():
    """Test the validators against a synthetic good/bad table."""
    import tempfile
    good_smoke = {
        "experiment": "quant_survival_smoke", "schema_version": "qs.smoke.v1",
        "model": "m", "layer": 1, "editor": "rome", "n_edits": 1, "n_probes": 1,
        "seed": 0, "schemes": ["nf4"], "damage_metric_note": "", "esr": {},
        "mechanism_tie": {"rho_keycos_damage_fp32_pooled": 0.5,
                          "rho_keycos_damage_fp32_within_probe": 0.5,
                          "within_probe_n_cols": 1},
        "arms": {"nf4_edited_layer": {k: 0.0 for k in SMOKE_REQUIRED_ARM_KEYS}},
        "frozen_prediction_readout": {},
    }
    bad_smoke = {"experiment": "quant_survival_smoke", "arms": {}}
    with tempfile.TemporaryDirectory() as td:
        g = os.path.join(td, "QS_smoke_table.json")
        b = os.path.join(td, "bad_smoke_table.json")
        json.dump(good_smoke, open(g, "w"))
        json.dump(bad_smoke, open(b, "w"))
        ok, _ = validate_smoke_table(g)
        assert ok, "good smoke table should validate"
        ok, errs = validate_smoke_table(b)
        assert not ok and "missing" in str(errs), f"bad smoke should fail: {errs}"
    print("[schema selftest] validators OK")


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Validate Paper B artifact schemas.")
    ap.add_argument("--root", default=os.path.join(HARNESS, "..", "results"),
                    help="results root")
    ap.add_argument("--selftest", action="store_true", help="run internal self-test")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    counts, failures = validate_all(args.root)
    print(f"[schema] checked: smoke={counts['smoke']} phase1={counts['phase1']} readout={counts['readout']}")
    if failures:
        print(f"[schema] FAILURES: {len(failures)}")
        for p, errs in failures:
            for e in errs:
                print(f"  {p}: {e}")
        sys.exit(1)
    print("[schema] ALL VALID")


if __name__ == "__main__":
    main()
