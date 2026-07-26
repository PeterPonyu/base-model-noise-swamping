#!/usr/bin/env python3
"""
compute_overthinking_gap.py — PREREG-P2-GRPO-20260710.md §4–§9, as code.

CPU + numpy + stdlib ONLY (imports diagnostic.py and analysis_deep.py, never
torch/trl).  Reads:

    samples/<id>[_nNNN].json        canonical PRE-RL samples  (largest _nNNN wins)
    results/<id>[_nNNN].json        canonical PRE-RL diagnostics (same rule)
    grpo_out/<id>/train_status.json GRPO terminal status (run_grpo.py contract)
    samples_postRL/<id>.json        POST-RL samples (sample_ckpt.py, unchanged tool)

Writes results/G_overthinking_test.json (atomic, allow_nan=False).

Frozen-prereg mapping (any change to these is a logged DEVIATION):
  §2  usability     n_right >= 20 AND D_within CI width <= 1.5, on the CANONICAL
                    results file.  --usability-only exposes exactly this rule to
                    the driver preflight (single implementation).
  §4  G_c           mean over solved-in-both problems of
                    mean_len(post-RL correct) / mean_len(pre-RL correct).
  §5  primary test  one-sided exact-permutation Spearman(D_within, G) —
                    diagnostic.exact_one_sided_spearman_test (full n! enumeration).
  §6  sensitivities (a) exp(delta) predictor (recomputed from canonical samples)
                    (b) full-panel exploratory (usability filter dropped)
                    (c) D_pooled predictor
                    (d) fixed-common-set G (intersection < 20 -> "unavailable",
                        NEVER relaxed)
  §7  exclusions    non-completed train_status; < 20 solved-in-both problems.
  §8  pass gate     p < 0.05 one-sided exact; adequately registered only at n >= 6.
  §9  kill          if included-usable n < 6: DESCRIPTIVE-ONLY — the confirmatory
                    test AND all rho sensitivities are suppressed, tables only.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import diagnostic as diag          # noqa: E402
import analysis_deep as deep       # noqa: E402  (PANEL, usability constants, delta_logmean)

RESULTS_DIR = os.path.join(HERE, "results")
SAMPLES_DIR = os.path.join(HERE, "samples")
SAMPLES_POST_DIR = os.path.join(HERE, "samples_postRL")
GRPO_OUT = os.path.join(HERE, "grpo_out")
OUT_PATH = os.path.join(RESULTS_DIR, "G_overthinking_test.json")

MIN_SOLVED_IN_BOTH = 20   # §7
MIN_CONFIRMATORY_N = 6    # §8/§9
MIN_COMMON_SET = 20       # §6(d)


# --------------------------------------------------------------------------- #
# canonical file resolution (shared rule: largest _nNNN version wins)
# --------------------------------------------------------------------------- #

def canonical_path(dir_: str, ckpt: str) -> Optional[str]:
    """`<ckpt>_nNNN.json` with the LARGEST N if any exists, else `<ckpt>.json`,
    else None."""
    best_n, best = -1, None
    for p in glob.glob(os.path.join(dir_, f"{ckpt}_n*.json")):
        m = re.match(rf"^{re.escape(ckpt)}_n(\d+)\.json$", os.path.basename(p))
        if m and int(m.group(1)) > best_n:
            best_n, best = int(m.group(1)), p
    if best:
        return best
    base = os.path.join(dir_, f"{ckpt}.json")
    return base if os.path.exists(base) else None


# --------------------------------------------------------------------------- #
# §2 usability (the ONE implementation; driver preflight calls --usability-only)
# --------------------------------------------------------------------------- #

def usability_record(ckpt: str) -> Dict[str, Any]:
    path = canonical_path(RESULTS_DIR, ckpt)
    rec: Dict[str, Any] = {"checkpoint": ckpt, "canonical_results": path,
                           "usable": False, "reasons": []}
    if path is None:
        rec["reasons"].append("no results file")
        return rec
    with open(path) as fh:
        r = json.load(fh)
    n_right = r.get("counts", {}).get("n_right")
    dw = r.get("D_within", {})
    ciw = None
    if isinstance(dw.get("ci_hi"), (int, float)) and isinstance(dw.get("ci_lo"), (int, float)):
        ciw = float(dw["ci_hi"]) - float(dw["ci_lo"])
    rec.update({"n_right": n_right, "D_within": dw.get("point"),
                "D_pooled": r.get("D_pooled", {}).get("point"),
                "ci_width": round(ciw, 4) if ciw is not None else None})
    if not isinstance(n_right, int) or n_right < deep.USABILITY_MIN_NRIGHT:
        rec["reasons"].append(f"n_right={n_right} < {deep.USABILITY_MIN_NRIGHT}")
    if ciw is None or ciw > deep.USABILITY_MAX_CIW:
        rec["reasons"].append(f"D_within_CI_width={ciw} > {deep.USABILITY_MAX_CIW}")
    rec["usable"] = not rec["reasons"]
    return rec


# --------------------------------------------------------------------------- #
# §4 G computation (pure function — selftest target)
# --------------------------------------------------------------------------- #

def compute_g(pre_problems: Sequence[Dict[str, Any]],
              post_problems: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-checkpoint overthinking gap G (§4, verbatim).

    solved-in-both = problem with >=1 correct trace pre-RL AND >=1 correct
    post-RL; per-problem ratio = mean_len(post correct)/mean_len(pre correct);
    G = arithmetic mean of the ratios."""
    pre = {p.get("problem"): p for p in pre_problems}
    post = {p.get("problem"): p for p in post_problems}
    ratios: Dict[str, float] = {}
    for pid in pre.keys() & post.keys():
        pre_right = diag._problem_lengths(pre[pid])[0]
        post_right = diag._problem_lengths(post[pid])[0]
        if pre_right and post_right:
            ratios[pid] = float(np.mean(post_right) / np.mean(pre_right))
    return {
        "G": float(np.mean(list(ratios.values()))) if ratios else None,
        "n_solved_in_both": len(ratios),
        "n_common_problems": len(pre.keys() & post.keys()),
        "per_problem_ratio": ratios,
    }


# --------------------------------------------------------------------------- #
# panel assembly
# --------------------------------------------------------------------------- #

def assemble_panel() -> Dict[str, Any]:
    """Per-checkpoint block: usability (§2), train status (§7), G (§4), and the
    §7 inclusion verdict for the primary test."""
    panel: Dict[str, Any] = {}
    for ckpt in deep.PANEL:
        blk = usability_record(ckpt)
        status_path = os.path.join(GRPO_OUT, ckpt, "train_status.json")
        train_status = None
        if os.path.exists(status_path):
            with open(status_path) as fh:
                train_status = json.load(fh)
        blk["train_status"] = (train_status or {}).get("status")
        blk["train_reason"] = (train_status or {}).get("reason")
        post_path = os.path.join(SAMPLES_POST_DIR, f"{ckpt}.json")
        blk["post_samples"] = post_path if os.path.exists(post_path) else None
        blk["exclusions"] = []
        if blk["train_status"] != "completed":
            blk["exclusions"].append(
                f"train_status={blk['train_status']} (§7: diverged/failed/absent)")
        if blk["post_samples"] is None:
            blk["exclusions"].append("no post-RL samples")
        if not blk["exclusions"]:
            pre_path = canonical_path(SAMPLES_DIR, ckpt)
            blk["canonical_samples"] = pre_path
            g = compute_g(diag.load_samples(pre_path), diag.load_samples(post_path))
            solved_ids = sorted(g.pop("per_problem_ratio").keys())
            blk["_ratios_path"] = None  # ratios re-derivable; keep JSON small
            blk.update(g)
            blk["solved_in_both_ids"] = solved_ids
            if g["n_solved_in_both"] < MIN_SOLVED_IN_BOTH:
                blk["exclusions"].append(
                    f"n_solved_in_both={g['n_solved_in_both']} < {MIN_SOLVED_IN_BOTH} (§7)")
        blk["in_primary"] = blk["usable"] and not blk["exclusions"]
        blk["in_exploratory"] = (not blk["exclusions"])  # §6(b): usability dropped
        panel[ckpt] = blk
    return panel


def _g_for(panel: Dict[str, Any], ckpt: str,
           restrict_ids: Optional[set] = None) -> Optional[float]:
    """G for one checkpoint, optionally restricted to a fixed problem-id set
    (§6d).  Recomputes from the sample files (ratios are not persisted)."""
    blk = panel[ckpt]
    g = compute_g(diag.load_samples(blk["canonical_samples"]),
                  diag.load_samples(blk["post_samples"]))
    ratios = g["per_problem_ratio"]
    if restrict_ids is not None:
        ratios = {k: v for k, v in ratios.items() if k in restrict_ids}
    return float(np.mean(list(ratios.values()))) if ratios else None


# --------------------------------------------------------------------------- #
# §5/§6/§8/§9 tests
# --------------------------------------------------------------------------- #

def run_tests(panel: Dict[str, Any]) -> Dict[str, Any]:
    primary_ids = [c for c in deep.PANEL if panel[c]["in_primary"]]
    n = len(primary_ids)
    out: Dict[str, Any] = {"primary_panel": primary_ids, "n_primary": n}

    if n < MIN_CONFIRMATORY_N:
        out["mode"] = "DESCRIPTIVE_ONLY"
        out["verdict"] = (
            f"KILL CONDITION (§9): only {n} usable checkpoints with valid G "
            f"(< {MIN_CONFIRMATORY_N}). Confirmatory correlation ABANDONED; "
            "per-checkpoint D and G tables reported descriptively; NO rho is "
            "computed (primary or sensitivity). Do not relax the usability rule.")
        return out

    out["mode"] = "CONFIRMATORY"
    d_within = [panel[c]["D_within"] for c in primary_ids]
    g_vec = [panel[c]["G"] for c in primary_ids]
    primary = diag.exact_one_sided_spearman_test(d_within, g_vec)
    primary["predictor"] = "D_within (canonical)"
    # degenerate (constant) vectors return early without significant/critical
    # keys — treat as a non-pass with an explicit verdict, never a crash
    primary["pass_H1"] = bool(primary.get("significant", False))
    out["primary"] = primary
    if "warning" in primary:
        out["gate"] = {"rule": "PASS iff one-sided exact-perm p < 0.05 (§8)",
                       "n": n, "pass": False,
                       "note": f"primary test degenerate: {primary['warning']}"}
        out["verdict"] = ("H1 NOT TESTABLE — degenerate primary vectors "
                          f"({primary['warning']}); no gate verdict")
        return out
    out["gate"] = {
        "rule": "PASS iff one-sided exact-perm p < 0.05 on the frozen panel (§8)",
        "n": n, "pass": primary["pass_H1"],
        "note": ("power at true rho=0.9 is ~0.43-0.54 at n=6-7: a null is "
                 "'no detectable association at this panel size', NEVER refutation (§8)"),
    }

    sens: Dict[str, Any] = {}
    # (a) exp(delta) predictor, recomputed from the canonical pre-RL samples
    exp_delta = []
    for c in primary_ids:
        d = deep.delta_logmean(diag.load_samples(panel[c]["canonical_samples"]))
        exp_delta.append(float(math.exp(d)) if math.isfinite(d) else float("nan"))
    if all(math.isfinite(v) for v in exp_delta):
        t = diag.exact_one_sided_spearman_test(exp_delta, g_vec)
        t["predictor"] = "exp(delta) robust log-mean (recomputed, canonical)"
        sens["a_exp_delta"] = t
    else:
        sens["a_exp_delta"] = {"unavailable": "non-finite delta on the panel"}
    # (b) full-panel exploratory (usability filter dropped)
    expl_ids = [c for c in deep.PANEL if panel[c]["in_exploratory"]]
    if len(expl_ids) >= 3:
        t = diag.exact_one_sided_spearman_test(
            [panel[c]["D_within"] for c in expl_ids],
            [panel[c]["G"] for c in expl_ids])
        t["predictor"] = "D_within (canonical)"
        t["panel"] = expl_ids
        t["note"] = "exploratory: noisy predictors included (§6b)"
        sens["b_full_panel"] = t
    else:
        sens["b_full_panel"] = {"unavailable": f"only {len(expl_ids)} with valid G"}
    # (c) D_pooled predictor
    d_pooled = [panel[c]["D_pooled"] for c in primary_ids]
    if all(isinstance(v, (int, float)) for v in d_pooled):
        t = diag.exact_one_sided_spearman_test(d_pooled, g_vec)
        t["predictor"] = "D_pooled (canonical)"
        sens["c_d_pooled"] = t
    else:
        sens["c_d_pooled"] = {"unavailable": "missing D_pooled on the panel"}
    # (d) fixed common set (§6d): intersection of solved-in-both across the panel
    common: Optional[set] = None
    for c in primary_ids:
        ids = set(panel[c]["solved_in_both_ids"])
        common = ids if common is None else (common & ids)
    n_common = len(common or ())
    if n_common < MIN_COMMON_SET:
        sens["d_fixed_common_set"] = {
            "available": False, "n_common": n_common,
            "note": (f"common solved-in-both set has {n_common} < {MIN_COMMON_SET} "
                     "problems -> reported as unavailable, NOT relaxed (§6d)")}
    else:
        g_common = [_g_for(panel, c, restrict_ids=common) for c in primary_ids]
        t = diag.exact_one_sided_spearman_test(d_within, g_common)
        t["predictor"] = "D_within (canonical)"
        t["G_variant"] = f"G on the fixed common set (n_common={n_common})"
        t["available"] = True
        t["n_common"] = n_common
        sens["d_fixed_common_set"] = t
    out["sensitivities"] = sens

    out["verdict"] = (
        f"H1 {'SUPPORTED' if primary['pass_H1'] else 'NOT SUPPORTED at this panel size'}"
        f" — rho={primary['rho']:.4f}, exact one-sided p={primary['p_exact_one_sided']:.5f}"
        f" (critical rho {primary['critical_rho_p05_onesided']:.4f} at n={n})")
    return out


# --------------------------------------------------------------------------- #
# driver / IO
# --------------------------------------------------------------------------- #

def _strip_heavy(panel: Dict[str, Any]) -> Dict[str, Any]:
    """Drop bulky per-problem id lists from the written report (re-derivable)."""
    slim = {}
    for c, blk in panel.items():
        b = dict(blk)
        b.pop("_ratios_path", None)
        ids = b.pop("solved_in_both_ids", None)
        if ids is not None:
            b["n_solved_in_both_ids"] = len(ids)
        slim[c] = b
    return slim


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--usability-only", action="store_true",
                    help="print the §2 usability verdict per checkpoint and exit "
                         "(the driver preflight's single source of truth)")
    ap.add_argument("--print-meta", metavar="CKPT", default=None,
                    help="print 'n k temp top_p max_new seed dtype' from the canonical "
                         "pre-RL samples' _meta and exit (driver helper; conda run "
                         "swallows heredoc stdin, so inline python is not an option)")
    ap.add_argument("--validate-sample", nargs=2, metavar=("PATH", "N"), default=None,
                    help="assert a samples JSON parses and has exactly N problems; "
                         "exit 0/1 (driver helper)")
    ap.add_argument("--selftest", action="store_true",
                    help="run the offline synthetic-fixture regression and exit")
    ap.add_argument("--out", default=OUT_PATH)
    args = ap.parse_args(argv)

    if args.selftest:
        return _selftest()

    if args.print_meta:
        p = canonical_path(SAMPLES_DIR, args.print_meta)
        if not p:
            print(f"no canonical samples for {args.print_meta}", file=sys.stderr)
            return 1
        with open(p) as fh:
            m = json.load(fh)["_meta"]
        print(m["n_problems"], m["k"], m["temperature"], m["top_p"],
              m["max_new_tokens"], m["seed"], m.get("model_dtype", "fp32"))
        return 0

    if args.validate_sample:
        path, n_expect = args.validate_sample[0], int(args.validate_sample[1])
        try:
            with open(path) as fh:
                d = json.load(fh)
            got = len(d["problems"] if isinstance(d, dict) else d)
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"VALIDATE-FAIL {path}: {e}", file=sys.stderr)
            return 1
        if got != n_expect:
            print(f"VALIDATE-FAIL {path}: n_problems {got} != {n_expect}", file=sys.stderr)
            return 1
        print(f"VALIDATE-OK {path} n_problems={got}")
        return 0

    if args.usability_only:
        recs = {c: usability_record(c) for c in deep.PANEL}
        usable = [c for c, r in recs.items() if r["usable"]]
        print(json.dumps({"rule": deep.USABILITY_RULE, "checkpoints": recs,
                          "usable": usable, "n_usable": len(usable)}, indent=2))
        return 0

    panel = assemble_panel()
    tests = run_tests(panel)
    out = {
        "_meta": {
            "tool": "compute_overthinking_gap.py",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "prereg": "PREREG-P2-GRPO-20260710.md (frozen v1.1)",
            "constants": {"min_solved_in_both": MIN_SOLVED_IN_BOTH,
                          "min_confirmatory_n": MIN_CONFIRMATORY_N,
                          "min_common_set": MIN_COMMON_SET},
        },
        "usability_rule": deep.USABILITY_RULE,
        "panel": _strip_heavy(panel),
        "tests": tests,
    }
    deep._assert_clean(json.loads(json.dumps(out, default=str)))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(out, fh, indent=2, default=str, allow_nan=False)
    os.replace(tmp, args.out)

    print(f"\n== P2 OVERTHINKING GAP — {tests['mode']} ==")
    for c in deep.PANEL:
        b = panel[c]
        g = b.get("G")
        g_str = f"{g:.4f}" if isinstance(g, float) else "-"
        tail = "" if not b["exclusions"] else f"  excl={b['exclusions']}"
        print(f"  {c:16s} usable={'Y' if b['usable'] else 'n'} "
              f"train={b.get('train_status')} G={g_str}{tail}")
    print(f"\n  {tests['verdict']}")
    print(f"  wrote {args.out}\n")
    return 0


# --------------------------------------------------------------------------- #
# selftest (offline, synthetic; no repo files touched)
# --------------------------------------------------------------------------- #

def _mk(pid: str, pre_lens: list, post_lens: list,
        pre_ok: list, post_ok: list):
    """One synthetic problem in pre and post form."""
    pre = {"problem": pid, "samples": [
        {"len": L, "correct": ok} for L, ok in zip(pre_lens, pre_ok)]}
    post = {"problem": pid, "samples": [
        {"len": L, "correct": ok} for L, ok in zip(post_lens, post_ok)]}
    return pre, post


def _selftest() -> int:
    # 1. G arithmetic: ratios 2.0 and 1.0 -> G = 1.5
    p0 = _mk("p0", [100, 50], [200, 70], [True, False], [True, False])
    p1 = _mk("p1", [100, 100, 30], [100, 40], [True, True, False], [True, False])
    g = compute_g([p0[0], p1[0]], [p0[1], p1[1]])
    assert g["n_solved_in_both"] == 2, g
    assert abs(g["G"] - 1.5) < 1e-12, g
    # 2. solved-in-both selection: right in pre only / post only -> excluded
    q0 = _mk("q0", [100], [200], [True], [False])   # pre-only right
    q1 = _mk("q1", [100], [200], [False], [True])   # post-only right
    g2 = compute_g([q0[0], q1[0]], [q0[1], q1[1]])
    assert g2["n_solved_in_both"] == 0 and g2["G"] is None, g2
    # 3. non-overlapping problem ids -> no common problems
    r0 = _mk("r0", [100], [200], [True], [True])
    r1 = _mk("r1", [100], [200], [True], [True])
    g3 = compute_g([r0[0]], [r1[1]])
    assert g3["n_common_problems"] == 0 and g3["n_solved_in_both"] == 0, g3
    # 4. exact test hand-checks (§5 machinery)
    t = diag.exact_one_sided_spearman_test([1, 2, 3], [1, 2, 3])
    assert abs(t["rho"] - 1.0) < 1e-12 and abs(t["p_exact_one_sided"] - 1 / 6) < 1e-12, t
    t2 = diag.exact_one_sided_spearman_test([1, 2, 3, 4], [1, 2, 3, 4])
    assert abs(t2["p_exact_one_sided"] - 1 / 24) < 1e-12, t2
    t3 = diag.exact_one_sided_spearman_test([1, 2, 3], [3, 2, 1])
    assert abs(t3["p_exact_one_sided"] - 1.0) < 1e-12, t3
    # 5. §9 kill-condition branch: n_primary < 6 -> DESCRIPTIVE_ONLY, no rho keys
    fake_panel = {c: {"in_primary": (i < 4), "in_exploratory": (i < 4),
                      "D_within": 1.0 + i, "D_pooled": 1.0 + i, "G": 1.0 + i,
                      "solved_in_both_ids": [f"x{j}" for j in range(25)],
                      "canonical_samples": None, "post_samples": None}
                  for i, c in enumerate(deep.PANEL)}
    tt = run_tests(fake_panel)
    assert tt["mode"] == "DESCRIPTIVE_ONLY" and "primary" not in tt \
        and "sensitivities" not in tt, tt
    # 6. §6(d) unavailability: panel with < MIN_COMMON_SET overlapping solved ids
    #    must report d_fixed_common_set unavailable (run in CONFIRMATORY mode)
    fake6 = {}
    for i, c in enumerate(deep.PANEL):
        # 6 usable checkpoints whose solved-id sets overlap in only 5 problems
        shared = [f"s{j}" for j in range(5)]
        own = [f"c{i}_{j}" for j in range(30)]
        fake6[c] = {"in_primary": (i < 6), "in_exploratory": (i < 6),
                    "D_within": 1.0 + 0.1 * i, "D_pooled": 1.0 + 0.1 * i,
                    "G": 1.0 + 0.05 * i,
                    "solved_in_both_ids": shared + own,
                    "canonical_samples": None, "post_samples": None}
    # sensitivity (a) reads canonical_samples -> stub delta_logmean for the fixture
    real_delta = deep.delta_logmean
    real_load = diag.load_samples
    deep.delta_logmean = lambda problems: 0.1
    diag.load_samples = lambda path: []
    try:
        tt6 = run_tests(fake6)
    finally:
        deep.delta_logmean = real_delta
        diag.load_samples = real_load
    assert tt6["mode"] == "CONFIRMATORY", tt6
    d_sens = tt6["sensitivities"]["d_fixed_common_set"]
    assert d_sens["available"] is False and d_sens["n_common"] == 5, d_sens
    print(json.dumps({"selftest": "OK", "checks": 6}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
