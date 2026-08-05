#!/usr/bin/env python3
"""Focused tests for the Paper B H11 result puller."""

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
PULL_SCRIPT = ROOT / "engine" / "box_paperb_h11_pull.sh"
CELLS = (
    "gemma2b_rome_L19_s2",
    "qwen3b_rome_L27_s2",
    "phi35_rome_L24_s0",
    "phi35_rome_L24_s1",
    "phi35_rome_L24_s2",
)


def populate_remote(remote_root: Path) -> None:
    stamp = {
        "code_sha256": "fixture-code",
        "pid": 123,
        "hostname": "fixture-host",
        "wall_start": "2026-08-05T00:00:00Z",
        "wall_end": "2026-08-05T00:01:00Z",
        "elapsed_s": 60,
        "nvidia_smi_sample": "0, 10 %, 100 MiB",
    }
    for cell in CELLS:
        cell_dir = remote_root / "results" / "quant_survival_curve" / cell
        cell_dir.mkdir(parents=True)
        (cell_dir / "QS_phase1_table.json").write_text(
            json.dumps({"runner_stamp": stamp, "editor": "rome", "codec": "real"})
        )
        np.savez(
            cell_dir / "QS_phase1_raw.npz",
            runner_stamp_json=np.array(json.dumps(stamp)),
            COS=np.zeros((200, 200), dtype=np.float32),
        )


def mock_rsync(bin_dir: Path) -> None:
    script = bin_dir / "rsync"
    script.write_text(
        """#!/usr/bin/env bash
set -u
args=("$@")
count=${#args[@]}
source_arg="${args[$((count - 2))]}"
destination="${args[$((count - 1))]}"
source_path="${source_arg#*:}"
[ -f "$source_path" ] || exit 23
cp "$source_path" "$destination"
"""
    )
    script.chmod(0o755)


def run_pull(local_root: Path, remote_root: Path, bin_dir: Path):
    (local_root / "engine").mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "HARNESS": str(local_root),
            "PATH": f"{bin_dir}:{env['PATH']}",
            "REMOTE_PORT": "36039",
            "PY": env.get("PY", "python3"),
        }
    )
    return subprocess.run(
        ["bash", str(PULL_SCRIPT), "mockhost", str(remote_root)],
        capture_output=True,
        text=True,
        env=env,
    )


def test_pull_requires_all_ten_critical_artifacts(tmp_path):
    remote_root = tmp_path / "remote"
    local_root = tmp_path / "local"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    populate_remote(remote_root)
    mock_rsync(bin_dir)

    result = run_pull(local_root, remote_root, bin_dir)

    assert result.returncode == 0, result.stdout + result.stderr
    pulled = (local_root / "engine" / "paperb_h11_pulled_this_run.txt").read_text().splitlines()
    assert len(pulled) == 10
    assert len(list((local_root / "results" / "quant_survival_curve").glob("*/*"))) == 10
    assert "All critical files pulled and validated" in result.stdout


def test_missing_remote_file_cannot_hide_behind_stale_local_file(tmp_path):
    remote_root = tmp_path / "remote"
    local_root = tmp_path / "local"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    populate_remote(remote_root)
    mock_rsync(bin_dir)

    missing_rel = Path(
        "results/quant_survival_curve/phi35_rome_L24_s2/QS_phase1_raw.npz"
    )
    (remote_root / missing_rel).unlink()
    stale = local_root / missing_rel
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"STALE\n")

    result = run_pull(local_root, remote_root, bin_dir)

    assert result.returncode == 3
    assert stale.read_bytes() == b"STALE\n"
    pulled = (local_root / "engine" / "paperb_h11_pulled_this_run.txt").read_text().splitlines()
    assert str(missing_rel) not in pulled
    assert f"MISSING CRITICAL: {missing_rel}" in result.stdout


def test_invalid_remote_pair_cannot_overwrite_local_truth(tmp_path):
    remote_root = tmp_path / "remote"
    local_root = tmp_path / "local"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    populate_remote(remote_root)
    mock_rsync(bin_dir)

    cell = "phi35_rome_L24_s2"
    table_rel = Path(f"results/quant_survival_curve/{cell}/QS_phase1_table.json")
    raw_rel = Path(f"results/quant_survival_curve/{cell}/QS_phase1_raw.npz")
    (remote_root / table_rel).write_text("not-json\n")
    local_table = local_root / table_rel
    local_raw = local_root / raw_rel
    local_table.parent.mkdir(parents=True)
    local_table.write_text("LOCAL-TABLE\n")
    local_raw.write_bytes(b"LOCAL-RAW\n")

    result = run_pull(local_root, remote_root, bin_dir)

    assert result.returncode == 3
    assert local_table.read_text() == "LOCAL-TABLE\n"
    assert local_raw.read_bytes() == b"LOCAL-RAW\n"
    assert f"INVALID CRITICAL: {cell}" in result.stdout
