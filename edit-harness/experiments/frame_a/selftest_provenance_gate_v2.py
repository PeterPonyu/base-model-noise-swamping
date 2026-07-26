"""CPU-only regression tests for provenance_gate_v2.py's fail-closed detectors."""
from __future__ import annotations

import datetime
import json
import os
import shutil
import sys
import tempfile
from typing import Any, Callable, Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import provenance_gate_v2 as pg  # noqa: E402

BASE_TIME = datetime.datetime(2030, 1, 1, tzinfo=datetime.timezone.utc).timestamp()
CLEAN_CUTOFF = "2029-01-01T00:00:00Z"
POLICIES = pg.v1.EXPECTED_POLICIES


def _stamp(index: int) -> Dict[str, Any]:
    start = datetime.datetime.fromtimestamp(BASE_TIME + index * 120, datetime.timezone.utc)
    return {
        "code_sha256": "a" * 64,
        "pid": 1000 + index,
        "hostname": "fixture-host",
        "wall_start": start.isoformat(),
        "wall_end": (start + datetime.timedelta(seconds=3)).isoformat(),
        "elapsed_s": 3.0,
        "nvidia_smi_sample": {"util_pct": 37, "mem_mib": 2048},
        "stamp_version": 1,
    }


def _cell(mix: str, policy: str, seed: int, index: int) -> Dict[str, Any]:
    offset = index / 10000.0
    return {
        "mix": mix,
        "policy": policy,
        "seed": seed,
        "model": pg.v1.EXPECTED_MODEL,
        "provenance": pg.v1.EXPECTED_PROVENANCE,
        "quality": {
            "A_upd": 0.8 + offset,
            "A_loc": 0.7 + offset,
            "A_cum": 0.9 + offset,
            "A_rip": 0.85 + offset,
            "Q": 0.79 + offset,
        },
        "cost": {
            "install_gpu_s": 210.25 + index * 0.37,
            "serve_gpu_s": 6.25 + index * 0.013,
            "total_gpu_s": 216.5 + index * 0.383,
            "serve_overhead_total": 0.2,
            "store_bytes_peak": 128.0,
            "exposure_surface_mean": 0.1,
        },
        "error_cost_eval": 5000.0 + index * 2.75,
        "discovery": {"n_damaging_gt": 60},
        "routing": {"arm_counts": {"edit": 499 - seed, "rag": seed + 1}},
        "stream_hash": f"fixture-{mix}-{policy}-{seed}",
        "runner_stamp": _stamp(index),
    }


def _write_grid(root: str) -> List[str]:
    paths: List[str] = []
    index = 0
    for mix in pg.v1.EXPECTED_MIXES:
        for policy in POLICIES:
            for seed in pg.v1.EXPECTED_SEEDS:
                body = _cell(mix, policy, seed, index)
                name = f"cell_{pg.v1.EXPECTED_MODEL}_real_{mix}_{policy}_s{seed}.json"
                path = os.path.join(root, name)
                with open(path, "w") as handle:
                    json.dump(body, handle, allow_nan=True)
                timestamp = BASE_TIME + index * 120
                os.utime(path, (timestamp, timestamp))
                paths.append(path)
                index += 1
    p2 = {
        "model": pg.v1.EXPECTED_MODEL,
        "provenance": "real",
        "mix": "MIX_C",
        "exposure_edit": 0.0,
        "exposure_rag": 1.0,
        "footprint_delta": 128000.0,
        "overhead_delta": 0.6,
        "router_edit_majority_on_privacy": 0.8,
    }
    with open(os.path.join(root, pg.v1.NAMESPACED_P2_NAME), "w") as handle:
        json.dump(p2, handle)
    return paths


def _mutate(path: str, fn: Callable[[Dict[str, Any]], None]) -> None:
    with open(path) as handle:
        body = json.load(handle)
    fn(body)
    with open(path, "w") as handle:
        json.dump(body, handle, allow_nan=True)


def _find_path(paths: List[str], mix: str, policy: str, seed: int) -> str:
    suffix = f"_{mix}_{policy}_s{seed}.json"
    return next(path for path in paths if path.endswith(suffix))


def _case(label: str, mutate: Callable[[str, List[str]], None], group: str, kind: str) -> bool:
    root = tempfile.mkdtemp(prefix=f"frame_a_v2_{label}_")
    try:
        paths = _write_grid(root)
        mutate(root, paths)
        report = pg.run_gate(root, CLEAN_CUTOFF)
        hit = any(item.get("kind") == kind for item in report["findings"][group])
        ok = report["status"] == "FAIL" and report["exit_code"] != 0 and hit
        print(f"[{'PASS' if ok else 'FAIL'}] {label}: status={report['status']} finding={hit}")
        return ok
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_anchor() -> bool:
    def mutate(_root: str, paths: List[str]) -> None:
        path = _find_path(paths, "MIX_B", "both", 0)
        _mutate(path, lambda body: body["cost"].update(serve_gpu_s=2400.0))
    return _case("exact synthetic anchor 2400", mutate, "synthetic_anchor_v2", "synthetic_anchor_v2")


def test_integer_anchor_1500() -> bool:
    def mutate(_root: str, paths: List[str]) -> None:
        path = _find_path(paths, "MIX_B", "both", 1)
        _mutate(path, lambda body: body["cost"].update(install_gpu_s=1500))
    return _case("integer synthetic anchor 1500", mutate, "synthetic_anchor_v2", "synthetic_anchor_v2")


def test_zero_variance() -> bool:
    def mutate(_root: str, paths: List[str]) -> None:
        for seed in pg.v1.EXPECTED_SEEDS:
            path = _find_path(paths, "MIX_B", "both", seed)
            def fix(body: Dict[str, Any]) -> None:
                body["quality"]["A_loc"] = 0.7123
                body["quality"]["Q"] = 0.8012
                body["error_cost_eval"] = 5123.0
                body["cost"]["install_gpu_s"] = 333.0
            _mutate(path, fix)
    return _case("zero cross-seed primary variance", mutate,
                 "primary_metric_variance_v2", "primary_metric_zero_cross_seed_variance")


def test_same_second() -> bool:
    def mutate(_root: str, paths: List[str]) -> None:
        timestamp = BASE_TIME + 99999
        for path in paths[:2]:
            os.utime(path, (timestamp, timestamp))
    root = tempfile.mkdtemp(prefix="frame_a_v2_same_second_")
    try:
        paths = _write_grid(root)
        mutate(root, paths)
        report = pg.run_gate(root, CLEAN_CUTOFF)
        items = report["findings"]["same_second_writes_v2"]
        hit = any(item.get("kind") == "same_second_batch_write" and
                  item.get("severity") == "WARN" and "caveat" in item for item in items)
        ok = report["status"] == "PASS" and report["exit_code"] == 0 and hit
        print(f"[{'PASS' if ok else 'FAIL'}] same-second write is WARN with caveat: "
              f"status={report['status']} finding={hit}")
        return ok
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_invalid_stamp_shape() -> bool:
    def mutate(_root: str, paths: List[str]) -> None:
        _mutate(paths[0], lambda body: body.update(runner_stamp=[]))
    return _case("invalid runner stamp shape", mutate, "runner_stamp_v2", "invalid_runner_stamp")


def test_missing_stamp() -> bool:
    def mutate(_root: str, paths: List[str]) -> None:
        path = paths[0]
        _mutate(path, lambda body: body.pop("runner_stamp"))
        os.utime(path, (BASE_TIME, BASE_TIME))
    return _case("post-cutoff missing runner stamp", mutate,
                 "runner_stamp_v2", "missing_runner_stamp")


def test_incomplete_exit_code() -> bool:
    root = tempfile.mkdtemp(prefix="frame_a_v2_incomplete_")
    try:
        paths = _write_grid(root)
        os.unlink(paths[-1])
        report = pg.run_gate(root, CLEAN_CUTOFF)
        ok = report["status"] == "INCOMPLETE" and report["exit_code"] == 2
        print(f"[{'PASS' if ok else 'FAIL'}] incomplete grid exit code: "
              f"status={report['status']} exit={report['exit_code']}")
        return ok
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_cutoff_boundary() -> bool:
    root = tempfile.mkdtemp(prefix="frame_a_v2_cutoff_")
    try:
        paths = _write_grid(root)
        cutoff = datetime.datetime.fromtimestamp(BASE_TIME, datetime.timezone.utc)
        _mutate(paths[0], lambda body: body.pop("runner_stamp"))
        os.utime(paths[0], (BASE_TIME, BASE_TIME))
        report = pg.run_gate(root, cutoff.isoformat())
        finding = report["findings"]["runner_stamp_v2"]
        ok = report["status"] == "FAIL" and any(item["kind"] == "missing_runner_stamp" for item in finding)
        print(f"[{'PASS' if ok else 'FAIL'}] cutoff boundary is post-cutoff: status={report['status']}")
        return ok
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_corrupt_utf8_is_fail() -> bool:
    root = tempfile.mkdtemp(prefix="frame_a_v2_utf8_")
    try:
        paths = _write_grid(root)
        with open(paths[0], "wb") as handle:
            handle.write(b"{\"broken\":\xff")
        report = pg.run_gate(root, CLEAN_CUTOFF)
        malformed = report["findings"].get("malformed_json", [])
        ok = report["status"] == "FAIL" and report["exit_code"] == 1 and bool(malformed)
        print(f"[{'PASS' if ok else 'FAIL'}] corrupt UTF-8 is FAIL: status={report['status']}")
        return ok
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_clean() -> bool:
    root = tempfile.mkdtemp(prefix="frame_a_v2_clean_")
    try:
        _write_grid(root)
        report = pg.run_gate(root, CLEAN_CUTOFF)
        ok = report["status"] == "PASS" and report["exit_code"] == 0
        print(f"[{'PASS' if ok else 'FAIL'}] clean stamped grid: status={report['status']}")
        return ok
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_legacy_warn() -> bool:
    root = tempfile.mkdtemp(prefix="frame_a_v2_legacy_")
    try:
        paths = _write_grid(root)
        _mutate(paths[0], lambda body: body.pop("runner_stamp"))
        os.utime(paths[0], (BASE_TIME, BASE_TIME))
        report = pg.run_gate(root, "2031-01-01T00:00:00Z")
        warnings = report["findings"]["legacy_runner_stamp_v2"]
        ok = report["status"] == "PASS" and len(warnings) == 1 and warnings[0]["severity"] == "WARN"
        print(f"[{'PASS' if ok else 'FAIL'}] legacy missing stamp grandfathered: "
              f"status={report['status']} warnings={len(warnings)}")
        return ok
    finally:
        shutil.rmtree(root, ignore_errors=True)


def main() -> int:
    results = [test_anchor(), test_integer_anchor_1500(), test_zero_variance(),
               test_same_second(), test_invalid_stamp_shape(), test_missing_stamp(),
               test_incomplete_exit_code(), test_cutoff_boundary(), test_corrupt_utf8_is_fail(),
               test_clean(), test_legacy_warn()]
    passed = sum(results)
    print(f"RESULT: {passed}/{len(results)} passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
