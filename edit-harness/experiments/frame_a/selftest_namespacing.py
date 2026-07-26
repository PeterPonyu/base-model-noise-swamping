"""selftest_namespacing.py — CPU gate for MAJOR-2 (cell namespacing + provenance refusal).

Kept SEPARATE from `selftest.py` (that file is another builder's slice) and wired into the
DRIVER's gate step (`run_frame_a_wave1.sh`), not into the main self-test. Proves, on a real
synthetic `run_wave` into a throwaway dir (no torch / GPU / network):

  1. run_wave writes `cell_{model}_{provenance}_{MIX}_{policy}_s{seed}.json` filenames AND stamps
     `model` + `provenance` into every cell body; the P2 file is `p2_{model}_{provenance}_MIX_C.json`;
  2. `analyze_frame_a.load_results` with an explicit expectation loads the matching cells and the
     namespaced P2, and `evaluate` runs on them;
  3. a stray foreign-provenance/model cell makes `load_results` HARD-FAIL (offender listed) under an
     expectation AND under no expectation (mixed set) — never silently scored;
  4. removing the stray restores a clean load.

INVOCATION: `python -m experiments.frame_a.selftest_namespacing` from the harness root (package
-relative imports). Exit 0 = GREEN, exit 1 = RED. Validations run from file+argv (never a heredoc
under conda — the stdin-swallow gotcha).
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import sys
import tempfile
import traceback

from . import config as C
from .run_stream import run_wave
from .scorer.analyze_frame_a import load_results, evaluate

MODEL = "testmodel-1b"
PROV = "synth"


def _check(name, fn):
    try:
        fn()
        print(f"  [OK] {name}")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL] {name}: {e}")
        traceback.print_exc()
        return False


def _build(tmp: str) -> None:
    """Run the synthetic wave (MIX_C only → cells + a namespaced P2) into `tmp`."""
    info = run_wave(tmp, synthetic=True, force=True, mixes=["MIX_C"], model_tag=MODEL)
    assert info["model"] == MODEL and info["provenance"] == PROV, info


def _filenames_and_body(tmp: str) -> None:
    cells = glob.glob(os.path.join(tmp, f"cell_{MODEL}_{PROV}_MIX_C_*.json"))
    assert cells, "no namespaced cell_{model}_{prov}_MIX_C_*.json written"
    # every cell file carries the namespace in BOTH the name and the body.
    for path in cells:
        c = json.load(open(path))
        assert c.get("model") == MODEL, f"cell body missing/mismatched model: {path}"
        assert c.get("provenance") == PROV, f"cell body missing/mismatched provenance: {path}"
    # un-namespaced legacy names must NOT be produced.
    assert not glob.glob(os.path.join(tmp, "cell_MIX_C_*.json")), "legacy un-namespaced cell leaked"
    # namespaced P2 present; legacy p2 absent.
    assert os.path.exists(os.path.join(tmp, f"p2_{MODEL}_{PROV}_MIX_C.json")), "namespaced P2 missing"
    assert not os.path.exists(os.path.join(tmp, "p2_MIX_C.json")), "legacy p2_MIX_C.json leaked"


def _clean_load_and_evaluate(tmp: str) -> None:
    res = load_results(tmp, expect_model=MODEL, expect_provenance=PROV)
    assert "MIX_C" in res and "both" in res["MIX_C"], "clean load missing MIX_C/both"
    assert "_p2" in res["MIX_C"], "namespaced P2 not loaded"
    v = evaluate(res)
    # This test builds MIX_C only (it exercises namespacing, not the full grid), so the
    # grid-completeness refusal (M4 fix 3) correctly classes it INCOMPLETE — all this
    # assert needs is that evaluate() runs and returns a recognized verdict class.
    assert v["VERDICT"] in ("PASS", "GREY", "KILL", "INCOMPLETE"), v
    # no-expectation clean load also works (single (model, provenance) present).
    res2 = load_results(tmp)
    assert "MIX_C" in res2


def _stray_offender_refused(tmp: str) -> None:
    stray = os.path.join(tmp, f"cell_othermodel-3b_real_MIX_C_both_s0.json")
    json.dump({"mix": "MIX_C", "policy": "both", "seed": 0, "model": "othermodel-3b",
               "provenance": "real", "quality": {"Q": 0.5}, "cost": {"total_gpu_s": 1.0},
               "discovery": {"recall_at_decile": float("nan"), "n_damaging_gt": 0}},
              open(stray, "w"))
    try:
        # (a) with an expectation: the foreign cell is an offender → hard error listing it.
        raised = False
        try:
            load_results(tmp, expect_model=MODEL, expect_provenance=PROV)
        except ValueError as e:
            raised = True
            assert "othermodel-3b_real" in str(e), f"offender not named in error: {e}"
        assert raised, "load_results must REFUSE a foreign-provenance cell under an expectation"
        # (b) with NO expectation: the dir is a mixed set → hard error.
        raised = False
        try:
            load_results(tmp)
        except ValueError:
            raised = True
        assert raised, "load_results must REFUSE a mixed (model, provenance) dir with no expectation"
    finally:
        os.remove(stray)
    # (c) after removing the stray, the clean expectation load works again.
    res = load_results(tmp, expect_model=MODEL, expect_provenance=PROV)
    assert "MIX_C" in res


def main() -> int:
    print("Frame-A namespacing/provenance self-test (MAJOR-2 driver gate):")
    tmp = tempfile.mkdtemp(prefix="frame_a_ns_")
    ok = True
    try:
        ok &= _check("run_wave builds namespaced synthetic cells", lambda: _build(tmp))
        ok &= _check("cell filenames + body carry model/provenance; namespaced P2", lambda: _filenames_and_body(tmp))
        ok &= _check("clean load (expected model/provenance) + evaluate", lambda: _clean_load_and_evaluate(tmp))
        ok &= _check("foreign cell REFUSED (offender listed) + mixed-dir refused", lambda: _stray_offender_refused(tmp))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("GATE:", "GREEN (namespacing OK)" if ok else "RED (namespacing BLOCKED)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
