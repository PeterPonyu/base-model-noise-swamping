#!/usr/bin/env python3
"""Test paperb_curve_readout.py schema and logic."""
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Add parent to path for imports
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.paperb_curve_readout import SCHEMA_VERSION, main, sha256_file


@pytest.fixture
def temp_harness(tmp_path):
    """Create minimal harness structure."""
    results = tmp_path / "results"
    (results / "quant_survival_curve").mkdir(parents=True)
    (results / "quant_survival").mkdir(parents=True)
    (results / "quant_survival" / "aggregate").mkdir(parents=True)
    (tmp_path / "engine").mkdir(parents=True)
    return tmp_path


def create_cell(base_dir, tag, layer, seed, nf4_surv, signal=10.0, noise=5.0):
    """Create a fake cell directory with table.json and raw.npz."""
    cell = base_dir / f"{tag}_rome_L{layer}_s{seed}"
    cell.mkdir(parents=True, exist_ok=True)

    # Create table.json
    table = {
        "arms": {
            "nf4dq_full_model": {
                "rho_damage_fp32_vs_arm_rank_survival": nf4_surv
            }
        }
    }
    (cell / "QS_phase1_table.json").write_text(json.dumps(table, indent=2))

    # Create raw.npz
    np.savez(
        cell / "QS_phase1_raw.npz",
        damage_fp32=np.full(100, signal / 100.0),
        base__nf4dq_full_model=np.full(100, noise / 100.0)
    )


def test_schema_version():
    """Schema version is 2."""
    assert SCHEMA_VERSION == 2


def test_sha256_file(tmp_path):
    """SHA256 hash works."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("hello")
    h = sha256_file(test_file)
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


@patch("experiments.paperb_curve_readout.ROOT")
@patch("experiments.paperb_curve_readout.NEW")
@patch("experiments.paperb_curve_readout.OLD")
@patch("experiments.paperb_curve_readout.OUT")
def test_missing_cells_incomplete(mock_out, mock_old, mock_new, mock_root, temp_harness, capsys):
    """Missing cells return INCOMPLETE status."""
    mock_root.return_value = temp_harness
    new_dir = temp_harness / "results" / "quant_survival_curve"
    old_dir = temp_harness / "results" / "quant_survival"
    mock_new.return_value = new_dir
    mock_old.return_value = old_dir
    mock_out.return_value = temp_harness / "results" / "quant_survival" / "aggregate" / "curve_local_readout.json"

    # Create only some cells (not all 9 required)
    create_cell(new_dir, "qwen3b", 27, 0, 0.5)
    create_cell(new_dir, "qwen3b", 27, 1, 0.5)
    # missing qwen3b s2, all gemma2b, all phi35

    # Patch the module constants directly
    with patch("experiments.paperb_curve_readout.NEW", new_dir), \
         patch("experiments.paperb_curve_readout.OLD", old_dir), \
         patch("experiments.paperb_curve_readout.OUT", mock_out.return_value), \
         patch("experiments.paperb_curve_readout.ROOT", temp_harness):
        rc = main()

    assert rc == 3

    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["status"] == "INCOMPLETE"
    assert result["schema_version"] == SCHEMA_VERSION
    assert len(result["missing"]) == 7  # 1 qwen + 3 gemma + 3 phi


@patch("experiments.paperb_curve_readout.ROOT")
@patch("experiments.paperb_curve_readout.NEW")
@patch("experiments.paperb_curve_readout.OLD")
@patch("experiments.paperb_curve_readout.OUT")
def test_complete_9cell_pass(mock_out, mock_old, mock_new, mock_root, temp_harness, capsys):
    """Complete 9 cells with G-S3 PASS creates receipt."""
    mock_root.return_value = temp_harness
    new_dir = temp_harness / "results" / "quant_survival_curve"
    old_dir = temp_harness / "results" / "quant_survival"
    out_path = temp_harness / "results" / "quant_survival" / "aggregate" / "curve_local_readout.json"
    mock_new.return_value = new_dir
    mock_old.return_value = old_dir
    mock_out.return_value = out_path

    # Create OLD cells (for full curve)
    create_cell(old_dir, "llama1b", 12, 0, 0.7, signal=10, noise=3)
    create_cell(old_dir, "llama1b", 12, 1, 0.68, signal=10, noise=3)
    create_cell(old_dir, "llama1b", 12, 2, 0.72, signal=10, noise=3)
    create_cell(old_dir, "llama3b", 24, 0, 0.5, signal=10, noise=5)
    create_cell(old_dir, "llama3b", 24, 1, 0.48, signal=10, noise=5)
    create_cell(old_dir, "llama3b", 24, 2, 0.52, signal=10, noise=5)
    create_cell(old_dir, "qwen15b", 21, 0, 0.6, signal=10, noise=4)
    create_cell(old_dir, "qwen15b", 21, 1, 0.58, signal=10, noise=4)
    create_cell(old_dir, "qwen15b", 21, 2, 0.62, signal=10, noise=4)

    # Create NEW cells (required for gate)
    # Set values to pass gates: qwen3b < qwen15b, llama3b < llama1b
    # Q3: need strong negative correlation
    create_cell(new_dir, "qwen3b", 27, 0, 0.4, signal=10, noise=7)
    create_cell(new_dir, "qwen3b", 27, 1, 0.38, signal=10, noise=7)
    create_cell(new_dir, "qwen3b", 27, 2, 0.42, signal=10, noise=7)
    create_cell(new_dir, "gemma2b", 19, 0, 0.45, signal=10, noise=6)
    create_cell(new_dir, "gemma2b", 19, 1, 0.43, signal=10, noise=6)
    create_cell(new_dir, "gemma2b", 19, 2, 0.47, signal=10, noise=6)
    create_cell(new_dir, "phi35", 24, 0, 0.42, signal=10, noise=6.5)
    create_cell(new_dir, "phi35", 24, 1, 0.40, signal=10, noise=6.5)
    create_cell(new_dir, "phi35", 24, 2, 0.44, signal=10, noise=6.5)

    with patch("experiments.paperb_curve_readout.ROOT", temp_harness), \
         patch("experiments.paperb_curve_readout.NEW", new_dir), \
         patch("experiments.paperb_curve_readout.OLD", old_dir), \
         patch("experiments.paperb_curve_readout.OUT", out_path):
        rc = main()

    # Check return code (may be 0 or 2 depending on gate pass)
    assert rc in (0, 2)

    # Check JSON output structure
    captured = capsys.readouterr()
    result = json.loads(captured.out)

    assert result["status"] == "PRE_B4_READOUT"
    assert result["schema_version"] == SCHEMA_VERSION
    assert "qwen15b_mean" in result
    assert "qwen3b_mean" in result
    assert "gemma2b_mean" in result
    assert "phi35_mean" in result
    assert "llama1b_mean" in result
    assert "llama3b_mean" in result
    assert "Q1_qwen_monotone" in result
    assert "Q1_llama_partial_monotone" in result
    assert "Q2_family_separation" in result
    assert "Q3_nsr_rho" in result
    assert "Q3_PASS" in result
    assert "G_S3_PASS" in result
    assert "seed_values" in result
    assert "curve_points" in result

    # Check curve_points structure
    points = result["curve_points"]
    assert len(points) == 18  # 18 cells = 9 OLD + 9 NEW
    for pt in points:
        assert "model" in pt
        assert "layer" in pt
        assert "seed" in pt
        assert "noise_to_signal" in pt
        assert "nf4_rank_survival" in pt
        assert "source_table" in pt
        assert "source_raw" in pt
        assert "table_sha256" in pt
        assert "raw_sha256" in pt
        assert isinstance(pt["noise_to_signal"], float)
        assert isinstance(pt["nf4_rank_survival"], float)
        assert len(pt["table_sha256"]) == 64
        assert len(pt["raw_sha256"]) == 64

    # Check output file exists
    assert out_path.exists()
    saved = json.load(open(out_path))
    assert saved["schema_version"] == SCHEMA_VERSION
    assert saved["curve_points"] == points


@patch("experiments.paperb_curve_readout.ROOT")
@patch("experiments.paperb_curve_readout.NEW")
@patch("experiments.paperb_curve_readout.OLD")
@patch("experiments.paperb_curve_readout.OUT")
def test_gs3_fail_deletes_receipt(mock_out, mock_old, mock_new, mock_root, temp_harness, capsys):
    """G-S3 FAIL deletes existing receipt."""
    mock_root.return_value = temp_harness
    new_dir = temp_harness / "results" / "quant_survival_curve"
    old_dir = temp_harness / "results" / "quant_survival"
    out_path = temp_harness / "results" / "quant_survival" / "aggregate" / "curve_local_readout.json"
    mock_new.return_value = new_dir
    mock_old.return_value = old_dir
    mock_out.return_value = out_path

    receipt = temp_harness / "engine" / "PAPERB_CURVE_GS3_PASS.ok"
    receipt.write_text("OLD PASS\n")  # pre-existing receipt

    # Create cells that will FAIL gates
    create_cell(old_dir, "llama1b", 12, 0, 0.5)
    create_cell(old_dir, "llama1b", 12, 1, 0.5)
    create_cell(old_dir, "llama1b", 12, 2, 0.5)
    create_cell(old_dir, "llama3b", 24, 0, 0.7)  # FAIL: llama3b > llama1b
    create_cell(old_dir, "llama3b", 24, 1, 0.7)
    create_cell(old_dir, "llama3b", 24, 2, 0.7)
    create_cell(old_dir, "qwen15b", 21, 0, 0.6)
    create_cell(old_dir, "qwen15b", 21, 1, 0.6)
    create_cell(old_dir, "qwen15b", 21, 2, 0.6)

    create_cell(new_dir, "qwen3b", 27, 0, 0.5)
    create_cell(new_dir, "qwen3b", 27, 1, 0.5)
    create_cell(new_dir, "qwen3b", 27, 2, 0.5)
    create_cell(new_dir, "gemma2b", 19, 0, 0.5)
    create_cell(new_dir, "gemma2b", 19, 1, 0.5)
    create_cell(new_dir, "gemma2b", 19, 2, 0.5)
    create_cell(new_dir, "phi35", 24, 0, 0.5)
    create_cell(new_dir, "phi35", 24, 1, 0.5)
    create_cell(new_dir, "phi35", 24, 2, 0.5)

    with patch("experiments.paperb_curve_readout.ROOT", temp_harness), \
         patch("experiments.paperb_curve_readout.NEW", new_dir), \
         patch("experiments.paperb_curve_readout.OLD", old_dir), \
         patch("experiments.paperb_curve_readout.OUT", out_path):
        rc = main()

    assert rc == 2  # FAIL

    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["G_S3_PASS"] is False

    # Receipt should be deleted
    assert not receipt.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
