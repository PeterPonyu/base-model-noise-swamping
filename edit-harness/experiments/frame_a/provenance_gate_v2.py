"""Fail-closed Frame-A provenance gate staged for use after the live wave drains.

This module preserves provenance_gate.py's report and exit-code contract, then adds four
independent checks: known synthetic cost anchors, cross-seed primary-metric degeneracy,
same-second batch writes, and cutoff-aware runner-stamp enforcement. The variance check
includes A_loc, Q, error_cost_eval, and install_gpu_s; fixed-arm policies can legitimately
collapse A_loc and Q across seeds, so install and error cost are required to distinguish
that case from a replayed cell bundle.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from . import provenance_gate as v1
except ImportError:  # Direct execution from experiments/frame_a/.
    import provenance_gate as v1  # type: ignore

DEFAULT_RUNNER_STAMP_CUTOFF = "2026-07-27T00:00:00Z"
SYNTHETIC_ANCHORS = frozenset({1500.0, 2400.0, 4000.0})
PRIMARY_METRIC_PATHS: Tuple[Tuple[str, ...], ...] = (
    ("quality", "A_loc"),
    ("quality", "Q"),
    ("error_cost_eval",),
    ("cost", "install_gpu_s"),
)
STAMP_REQUIRED_FIELDS = frozenset({
    "code_sha256", "pid", "hostname", "wall_start", "wall_end", "elapsed_s",
    "nvidia_smi_sample", "stamp_version",
})


def _parse_utc(value: str) -> datetime.datetime:
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    parsed = datetime.datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=datetime.timezone.utc)
    return parsed.astimezone(datetime.timezone.utc)


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _path_value(cell: Dict[str, Any], path: Sequence[str]) -> Optional[float]:
    value: Any = cell
    for key in path:
        value = value.get(key) if isinstance(value, dict) else None
    return _number(value)


def _in_scope(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        cell for cell in cells
        if cell.get("model") == v1.EXPECTED_MODEL
        and cell.get("provenance") == v1.EXPECTED_PROVENANCE
        and cell.get("mix") in v1.EXPECTED_MIXES
    ]


def _check_synthetic_anchors(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    for cell in cells:
        cost = cell.get("cost") if isinstance(cell.get("cost"), dict) else {}
        for field in ("install_gpu_s", "serve_gpu_s"):
            value = _number(cost.get(field))
            if value in SYNTHETIC_ANCHORS:
                findings.append({
                    "kind": "synthetic_anchor_v2",
                    "path": cell["_path"],
                    "cell_id": cell["_cell_id"],
                    "field": f"cost.{field}",
                    "value": value,
                    "anchors": sorted(SYNTHETIC_ANCHORS),
                    "severity": "FAIL",
                })
    return findings


def _check_primary_metric_variance(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    siblings: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        siblings[(cell.get("mix"), cell.get("policy"))].append(cell)

    findings: List[Dict[str, Any]] = []
    expected_seeds = set(v1.EXPECTED_SEEDS)
    for (mix, policy), group in sorted(siblings.items()):
        by_seed = {cell.get("seed"): cell for cell in group}
        if set(by_seed) != expected_seeds:
            continue
        arrays: Dict[str, List[float]] = {}
        complete = True
        for metric_path in PRIMARY_METRIC_PATHS:
            values = [_path_value(by_seed[seed], metric_path) for seed in v1.EXPECTED_SEEDS]
            if any(value is None for value in values):
                complete = False
                break
            arrays[".".join(metric_path)] = [float(value) for value in values if value is not None]
        if not complete:
            continue
        zero_variance = all(max(values) - min(values) <= v1.DEGENERACY_ATOL
                            for values in arrays.values())
        if zero_variance:
            findings.append({
                "kind": "primary_metric_zero_cross_seed_variance",
                "mix": mix,
                "policy": policy,
                "seeds": list(v1.EXPECTED_SEEDS),
                "primary_metric_arrays": arrays,
                "atol": v1.DEGENERACY_ATOL,
                "severity": "FAIL",
            })
    return findings


def _check_same_second_writes(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_second: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        by_second[cell["_mtime_sec"]].append(cell)
    findings: List[Dict[str, Any]] = []
    for second, group in sorted(by_second.items()):
        if second < 0 or len(group) < 2:
            continue
        findings.append({
            "kind": "same_second_batch_write",
            "mtime_sec": second,
            "mtime_iso": datetime.datetime.fromtimestamp(
                second, tz=datetime.timezone.utc).isoformat(),
            "count": len(group),
            "paths": sorted(cell["_path"] for cell in group),
            "caveat": "POSIX mtime is not copy provenance: rsync -a / cp -p preserve source mtimes, "
                       "while non-mtime-preserving pulls can rewrite a legitimate bundle to one second; "
                       "confirm against the source-host runner log before acting on this warning.",
            "severity": "WARN",
        })
    return findings


def _stamp_shape_errors(stamp: Any) -> List[str]:
    if not isinstance(stamp, dict):
        return ["runner_stamp must be an object"]
    errors = [f"missing {key}" for key in sorted(STAMP_REQUIRED_FIELDS - set(stamp))]
    if stamp.get("stamp_version") != 1:
        errors.append("stamp_version must equal 1")
    code_hash = stamp.get("code_sha256")
    if not isinstance(code_hash, str) or len(code_hash) != 64:
        errors.append("code_sha256 must be a 64-character hex digest")
    elif any(char not in "0123456789abcdef" for char in code_hash.lower()):
        errors.append("code_sha256 must be hexadecimal")
    if not isinstance(stamp.get("pid"), int) or isinstance(stamp.get("pid"), bool) or stamp.get("pid", 0) <= 0:
        errors.append("pid must be a positive integer")
    if not isinstance(stamp.get("hostname"), str) or not stamp.get("hostname"):
        errors.append("hostname must be a nonempty string")
    elapsed = _number(stamp.get("elapsed_s"))
    if elapsed is None or elapsed < 0:
        errors.append("elapsed_s must be a finite nonnegative number")
    for field in ("wall_start", "wall_end"):
        try:
            _parse_utc(stamp.get(field, ""))
        except (TypeError, ValueError):
            errors.append(f"{field} must be an ISO timestamp")
    if not isinstance(stamp.get("nvidia_smi_sample"), dict):
        errors.append("nvidia_smi_sample must be an object")
    return errors


def _check_runner_stamps(
    cells: List[Dict[str, Any]], cutoff: datetime.datetime,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    required: List[Dict[str, Any]] = []
    legacy: List[Dict[str, Any]] = []
    cutoff_epoch = cutoff.timestamp()
    for cell in cells:
        stamp = cell.get("runner_stamp")
        # A failed stat is not evidence of an old legacy cell. Treat it as new/unknown.
        after_cutoff = cell["_mtime_sec"] < 0 or cell["_mtime_sec"] >= cutoff_epoch
        if stamp is None:
            finding = {
                "kind": "missing_runner_stamp" if after_cutoff else "legacy_missing_runner_stamp",
                "path": cell["_path"],
                "cell_id": cell["_cell_id"],
                "mtime_sec": cell["_mtime_sec"],
                "cutoff_utc": cutoff.isoformat(),
                "severity": "FAIL" if after_cutoff else "WARN",
            }
            (required if after_cutoff else legacy).append(finding)
            continue
        errors = _stamp_shape_errors(stamp)
        if errors:
            required.append({
                "kind": "invalid_runner_stamp",
                "path": cell["_path"],
                "cell_id": cell["_cell_id"],
                "errors": errors,
                "severity": "FAIL",
            })
    return required, legacy


def _max_severity(items: List[Dict[str, Any]]) -> str:
    rank = {"INFO": 0, "WARN": 1, "FAIL": 2}
    value = max((rank.get(item.get("severity", "INFO"), 0) for item in items), default=0)
    return {0: "INFO", 1: "WARN", 2: "FAIL"}[value]


def run_gate(
    cells_dir: str, runner_stamp_cutoff: str = DEFAULT_RUNNER_STAMP_CUTOFF,
) -> Dict[str, Any]:
    report = v1.run_gate(cells_dir)
    cells, _ = v1._load_cells(cells_dir)
    in_scope = _in_scope(cells)
    cutoff = _parse_utc(runner_stamp_cutoff)

    v2_groups = {
        "synthetic_anchor_v2": _check_synthetic_anchors(in_scope),
        "primary_metric_variance_v2": _check_primary_metric_variance(in_scope),
        "same_second_writes_v2": _check_same_second_writes(in_scope),
    }
    stamp_fail, stamp_legacy = _check_runner_stamps(in_scope, cutoff)
    v2_groups["runner_stamp_v2"] = stamp_fail
    v2_groups["legacy_runner_stamp_v2"] = stamp_legacy
    report["findings"].update(v2_groups)
    report["group_severity"].update({key: _max_severity(value)
                                     for key, value in v2_groups.items()})

    any_fail = any(item.get("severity") == "FAIL"
                   for group in report["findings"].values() for item in group)
    n_in_scope = report["counts"]["in_scope_cells"]
    p2_present = any(item.get("kind") == "p2_ok"
                     for item in report["findings"].get("p2_status", []))
    if any_fail:
        report["status"], report["exit_code"] = "FAIL", 1
    elif n_in_scope < v1.EXPECTED_TOTAL:
        report["status"], report["exit_code"] = "INCOMPLETE", 2
    elif n_in_scope == v1.EXPECTED_TOTAL and p2_present:
        report["status"], report["exit_code"] = "PASS", 0
    else:
        report["status"], report["exit_code"] = "FAIL", 1

    report["schema_version"] = "frame_a.provenance_gate.v2"
    report["thresholds"].update({
        "synthetic_anchor_values": sorted(SYNTHETIC_ANCHORS),
        "same_second_batch_min_cells": 2,
        "primary_metric_paths": [".".join(path) for path in PRIMARY_METRIC_PATHS],
        "runner_stamp_cutoff_utc": cutoff.isoformat(),
    })
    return report


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Fail-closed Frame-A provenance gate v2")
    parser.add_argument("--cells_dir", default="results/frame_a/cells")
    parser.add_argument("--report", default=None)
    parser.add_argument("--runner-stamp-cutoff", default=DEFAULT_RUNNER_STAMP_CUTOFF)
    args = parser.parse_args(argv)
    if not os.path.isdir(args.cells_dir):
        print(json.dumps({"status": "USAGE", "error": f"cells_dir not a directory: {args.cells_dir}"}))
        return 3
    try:
        cutoff = _parse_utc(args.runner_stamp_cutoff)
    except (TypeError, ValueError) as error:
        print(json.dumps({"status": "USAGE", "error": f"invalid runner stamp cutoff: {error}"}))
        return 3
    report = run_gate(args.cells_dir, cutoff.isoformat())
    rendered = json.dumps(report, indent=2, allow_nan=True)
    print(rendered)
    if args.report:
        with open(args.report, "w") as handle:
            handle.write(rendered + "\n")
    return report["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
