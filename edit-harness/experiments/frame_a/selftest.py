"""selftest.py — the MANDATORY pre-wave CPU gate (DESIGN §selftest.py, PREREG §4).

Run from `run_frame_a_wave1.sh` BEFORE any GPU cell is read. A single failed assertion exits
non-zero and BLOCKS the wave (no silent vacuous pass — validations run from file+argv, never a
heredoc under conda). Proves, on synthetic fixtures (no torch / GPU / network):

  1. planted `oracle ≥ router(both) > random` in quality  (⇔ ErrorCost oracle ≤ both < random);
  2. an ANTI-correlated damage predictor routes WORSE than random;
  3. RAG extra-token serving cost > 0;
  4. GRACE serving > install;
  5. edit serving is FLAT in store size N;
  6. injected conflict / damaging counts match the mix rates (+ damaging_gt/synth partition);
  7. ErrorCost_eval arithmetic is exact AND gov-free (exposure never enters it);
  8. every module selftest passes.

Exit 0 = gate GREEN (wave may proceed). Exit 1 = gate RED (wave blocked).

INVOCATION (MINOR-2): this module uses package-relative imports, so it must be run as a MODULE
from the harness root — `python -m experiments.frame_a.selftest` — NOT as a bare script
(`python experiments/frame_a/selftest.py`), which has no package context and fails on the
relative imports. The driver `run_frame_a_wave1.sh` invokes it the `-m` way.
"""
from __future__ import annotations

import os
import sys
import traceback

import numpy as np

from . import config as C
from .cost_harness import SyntheticClock
from .damage_predictor import DamagePredictor
from .router import Router, RandomRouter, OracleRouter
from .stream_builder import StreamBuilder
from .run_stream import _replay
from .scorer.scoring import error_cost_eval, OutcomeRow


def _check(name, fn):
    try:
        fn()
        print(f"  [OK] {name}")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL] {name}: {e}")
        traceback.print_exc()
        return False


# ---------------------------------------------------------------- gate assertions
def _planted_oracle_router_random():
    b = StreamBuilder(synthetic=True)
    ups, _ = b.build_stream("MIX_B", 0)
    pred = DamagePredictor()
    anti = DamagePredictor(); anti.recal = {"scale": -1.0, "bias": 0.0}
    def ec(router):
        rows, _ = _replay(ups, router)
        return error_cost_eval(rows)
    # calibration lands λ in the small-grid range where the planted ordering is clean; the
    # self-test fixes λ=1e-2 (a grid value) to isolate the predictor's contribution.
    ec_oracle = ec(OracleRouter())
    ec_both = ec(Router(predictor=pred, mode="both", lambda_cost=1e-2))
    ec_anti = ec(Router(predictor=anti, mode="both", lambda_cost=1e-2))
    ec_rand = float(np.mean([ec(RandomRouter(seed=s)) for s in range(3)]))
    assert ec_oracle <= ec_both, f"oracle must upper-bound router: oracle={ec_oracle:.0f} both={ec_both:.0f}"
    assert ec_both < ec_rand, f"router(both) must beat random: both={ec_both:.0f} random={ec_rand:.0f}"
    assert ec_anti > ec_rand, f"anti-correlated predictor must route WORSE than random: anti={ec_anti:.0f} random={ec_rand:.0f}"


def _cost_invariants():
    clk = SyntheticClock()
    # RAG extra-token serving cost > 0:
    assert clk.serve("rag", 1, 100, k=5).extra_input_tokens > 0
    assert clk.serve("rag", 1, 100, k=5).gpu_s > clk.serve("edit", 1, 100).gpu_s
    # GRACE serving > install:
    assert clk.serve("grace", 1, 100).gpu_s > clk.install("grace", 1, 100).gpu_s
    # edit serving flat in N:
    assert abs(clk.serve("edit", 1, 10).gpu_s - clk.serve("edit", 1, 10**6).gpu_s) < 1e-9
    # RAG per-query serving constant in N (only the k-fact prefill); GRACE grows with N:
    assert abs(clk.serve("rag", 1, 10, 5).gpu_s - clk.serve("rag", 1, 10**5, 5).gpu_s) < 1e-9
    assert clk.serve("grace", 1, 10**4).gpu_s > clk.serve("grace", 1, 10).gpu_s


def _injection_counts():
    b = StreamBuilder(synthetic=True)
    n = C.STREAM_LEN_WAVE1
    _, manB = b.build_stream("MIX_B", 0)
    conf = manB["conflict_flag_counts"].get("conflict", 0)
    dmg = manB["damaging_partition"]["gt"] + manB["damaging_partition"]["synth"]
    assert abs(conf - 0.30 * n) <= 6, f"MIX-B conflict count {conf}"
    assert abs(dmg - 0.30 * n) <= 6, f"MIX-B damaging count {dmg}"
    assert manB["discovery_scope"]["headline_set"] == "damaging_gt"
    _, manA = b.build_stream("MIX_A", 0)
    dmgA = manA["damaging_partition"]["gt"] + manA["damaging_partition"]["synth"]
    assert abs(dmgA - 0.10 * n) <= 5, f"MIX-A damaging must stay ~0.10 (DOF-1): {dmgA}"


def _errorcost_arithmetic_and_govfree():
    row = OutcomeRow(t=0, arm="edit", fact_type="cf", applied=True, collateral=2.0,
                     stale=False, install_gpu_s=1.0, serve_overhead=0.5, serve_gpu_s=0.0,
                     exposure_surface=1.0)
    ec = error_cost_eval([row], {"C_wrong": 30.0, "C_stale": 9.0, "C_latency": 1.0, "C_compute": 1.0})
    expect = 30 * 2.0 + 9 * 0 + 1 * 0.5 + 1 * 1.0
    assert abs(ec - expect) < 1e-9, f"arithmetic {ec} != {expect}"
    # explicit gov-free check (OPTION A): changing exposure must not move ErrorCost_eval at all.
    a = error_cost_eval([row]); row.exposure_surface = 999.0
    assert abs(error_cost_eval([row]) - a) < 1e-9, "ErrorCost_eval must be exposure/gov-free"


def _determinism_subprocess():
    """CROSS-PROCESS determinism (MODERATE regression guard): build the SAME stream in two FRESH
    interpreters (each with its own random PYTHONHASHSEED) and assert identical stream_hash. The
    in-process check structurally cannot catch PYTHONHASHSEED salting — only a subprocess can."""
    import subprocess
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # edit-harness/
    code = ("from experiments.frame_a.stream_builder import StreamBuilder;"
            "print(StreamBuilder(synthetic=True).build_stream('MIX_A',0)[1]['stream_hash'])")
    hashes = []
    for _ in range(2):
        env = dict(os.environ); env["PYTHONHASHSEED"] = "random"
        out = subprocess.run([sys.executable, "-c", code], cwd=root, env=env,
                             capture_output=True, text=True)
        assert out.returncode == 0, f"subprocess build failed: {out.stderr[-400:]}"
        hashes.append(out.stdout.strip().splitlines()[-1])
    assert hashes[0] == hashes[1], (
        f"CROSS-PROCESS NON-DETERMINISM: stream_hash differs across interpreters "
        f"({hashes[0]} != {hashes[1]}) — a builtin hash() leak (PYTHONHASHSEED salt).")


def _router_view_leak_check():
    """Enumerate ALL forbidden fields and assert router_view leaks NONE of them across a stream
    that CONTAINS damaging updates — incl. the raw `conflict_flag=='damaging'` label (must be
    masked to 'none'). The router must discover damage from key_cos geometry, never a label."""
    from .stream_builder import StreamBuilder, router_view
    forbidden = ("gt_damage", "gt_damage_provenance", "gt_measured", "damaging_gt_eligible",
                 "damaging_kind", "downstream_query_set", "orig_idx")
    b = StreamBuilder(synthetic=True)
    updates, _ = b.build_stream("MIX_B", 0)          # MIX-B has the most damaging updates.
    n_dmg = sum(1 for u in updates if u.get("damaging_kind") == "gt")
    assert n_dmg > 0, "test needs damaging_gt updates present to be meaningful"
    for u in updates:
        rv = router_view(u)
        for f in forbidden:
            assert f not in rv, f"router_view LEAKS forbidden field '{f}'"
        assert rv.get("conflict_flag") != "damaging", "router_view leaks the raw 'damaging' label"
    # positive control: a damaging update's raw conflict_flag is 'damaging' but its view is 'none'.
    dmg_u = next(u for u in updates if u.get("damaging_kind") == "gt")
    assert dmg_u["conflict_flag"] == "damaging" and router_view(dmg_u)["conflict_flag"] == "none"


def _real_join_check():
    """Reviewer-mandated CPU check: the identity geometry-join against the ACTUAL gate_llama1b
    npz (the synthetic gate cannot catch the MAJOR-1 misjoin class). SKIP-safe if the cell is
    absent; asserts covered CF edit records carry cell row orig_idx's key_cos/gt_damage."""
    from .stream_builder import _selftest_real_join
    _selftest_real_join()


def _module_selftests():
    from .cost_harness import _selftest as s1
    from .damage_predictor import _selftest as s2
    from .arms.base import _selftest as s3
    from .arms.real_backends import _selftest as s4
    from .arms.real_asserts import _selftest as s4b
    from .stream_builder import _selftest as s5
    from .router import _selftest as s6
    from .run_stream import _selftest as s7
    from .scorer.scoring import _selftest as s8
    from .scorer.analyze_frame_a import _selftest as s9
    from .real_replay import _selftest as s10   # CPU-mock real replay: (c)+(d) fire+pass.
    for s in (s1, s2, s3, s4, s4b, s5, s6, s7, s8, s9, s10):
        s()


def main() -> int:
    print("Frame-A self-test gate (mandatory pre-wave):")
    ok = True
    ok &= _check("module selftests (all 11)", _module_selftests)
    ok &= _check("planted oracle>=router>random & anti<random", _planted_oracle_router_random)
    ok &= _check("cost invariants (RAG token>0, GRACE serve>install, edit flat-in-N)", _cost_invariants)
    ok &= _check("injection counts (conflict/damaging + gt/synth partition)", _injection_counts)
    ok &= _check("ErrorCost_eval exact + gov-free", _errorcost_arithmetic_and_govfree)
    ok &= _check("cross-process determinism (subprocess stream_hash match)", _determinism_subprocess)
    ok &= _check("router_view leaks no forbidden field (incl. masked 'damaging' label)", _router_view_leak_check)
    ok &= _check("real 3-cell-union geometry-join vs actual npz (MAJOR-1 guard; SKIP-safe)", _real_join_check)
    print("GATE:", "GREEN (wave may proceed)" if ok else "RED (wave BLOCKED)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
