"""analyze_frame_a.py — seed aggregation, the computable Pareto predicate, and the frozen
P1–P4 / PASS-GREY-KILL verdict (PREREG §2–§4). Emits results/frame_a/frame_a_verdict.json.

Bindings:
  * Pareto predicate is COMPUTABLE with seed-level bootstrap 95% CIs (no visual frontier reading).
    Cost axis = total normalised GPU-seconds; quality axis = Q. "matches-then-undercuts" = equal Q
    within CI at strictly lower cost.
  * P2 is the STRUCTURAL conjunction in MIX-C (exposure 0 vs ≈1 AND footprint_delta>0 AND per-query
    overhead_delta>0 AND router-selects-edit-majority) — NEVER read off ErrorCost_eval.
  * VERDICT = PASS (P1 ∧ P2) / KILL (¬P1 ∧ ¬P2) / GREY (exactly one). P3/P4 sharpen, do not pivot.
  * Thresholds are config constants; predictions are not re-chosen after seeing data.

FLAG (A_loc semantics): the `A_loc` component of every cell's quality is a SEQUENTIAL-incremental
stream-locality metric (running baseline, clipped) — NOT the B6 gate cells' per-edit-vs-fixed-base
`damage_logit`. See scoring.quality. The manuscript must not equate the two.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from .. import config as C
from . import scoring

FIXED_STRATEGIES = ("always_edit", "always_grace", "always_rag", "always_ft", "always_reject", "random")


# ---------------------------------------------------------------- bootstrap / Pareto
def _bootstrap_diff(a: List[float], b: List[float], n: int = C.BOOTSTRAP_N,
                    ci: float = C.BOOTSTRAP_CI, seed: int = 0) -> Tuple[float, float, float]:
    """Seed-level bootstrap of mean(a)-mean(b). Resamples the (paired) seed orderings.

    Returns (mean_diff, ci_lo, ci_hi). With a handful of seeds this is the pinned procedure —
    weak power is honestly reflected in wide CIs.
    """
    a = np.asarray(a, float); b = np.asarray(b, float)
    rng = np.random.default_rng(seed)
    m = min(len(a), len(b))
    a, b = a[:m], b[:m]
    diffs = []
    for _ in range(n):
        idx = rng.integers(0, m, size=m)
        diffs.append(a[idx].mean() - b[idx].mean())
    lo = float(np.quantile(diffs, (1 - ci) / 2))
    hi = float(np.quantile(diffs, 1 - (1 - ci) / 2))
    return float(a.mean() - b.mean()), lo, hi


def pareto_dominates(qA: List[float], cA: List[float], qB: List[float], cB: List[float],
                     seed: int = 0) -> Dict:
    """Computable CI-level Pareto predicate (higher Q better, lower cost better).

    A dominates B iff, at the 95% CI level, EITHER
      strictly-higher Q with no significant cost loss, OR
      strictly-lower cost with no significant Q loss  (= matches-then-undercuts).
    strict_better(dX)=CIlo(dX)>0 ; not_worse(dX)=CIhi(dX)>=0 (B not significantly better on X).
    """
    dQ, dQlo, dQhi = _bootstrap_diff(qA, qB, seed=seed)                 # Q(A)-Q(B)
    dC, dClo, dChi = _bootstrap_diff([-x for x in cA], [-x for x in cB], seed=seed + 1)  # cost adv (A cheaper>0)
    strict_Q = dQlo > 0.0
    strict_C = dClo > 0.0
    notworse_Q = dQhi >= 0.0
    notworse_C = dChi >= 0.0
    dominates = (strict_Q and notworse_C) or (strict_C and notworse_Q)
    return {"dominates": bool(dominates), "dQ": dQ, "dQ_ci": [dQlo, dQhi],
            "d_cost_adv": dC, "d_cost_adv_ci": [dClo, dChi],
            "strict_Q": strict_Q, "strict_cost": strict_C}


# ---------------------------------------------------------------- per-mix extraction
def _series(cell_list: List[Dict], key: str) -> List[float]:
    return [float(c[key]) for c in cell_list]


def _q_series(cells: List[Dict]) -> List[float]:
    return [float(c["quality"]["Q"]) for c in cells]


def _cost_series(cells: List[Dict]) -> List[float]:
    return [float(c["cost"]["total_gpu_s"]) for c in cells]


# ---------------------------------------------------------------- predictions
def evaluate(results: Dict[str, Dict[str, List[Dict]]]) -> Dict:
    """results[mix][policy] = list of per-seed cell dicts {quality, cost, discovery, ...}.

    MIX_C additionally carries results[mix]['_p2'] = the structural P2 quantities.
    """
    per_mix = {}
    p1_mixes, p3_mixes = 0, 0
    p4_beats_cost, p4_beats_damage = 0, 0
    for mix, policies in results.items():
        if "both" not in policies:
            continue
        qB, cB = _q_series(policies["both"]), _cost_series(policies["both"])
        # --- P1: both Pareto-dominates every fixed strategy + random.
        p1_here = True
        p1_detail = {}
        for strat in FIXED_STRATEGIES:
            if strat not in policies:
                p1_here = False; p1_detail[strat] = "MISSING"; continue
            dom = pareto_dominates(qB, cB, _q_series(policies[strat]), _cost_series(policies[strat]))
            p1_detail[strat] = dom
            p1_here = p1_here and dom["dominates"]
        if p1_here:
            p1_mixes += 1
        # --- P3: both vs ft_merge, both-way cost parity (== dominates).
        p3_here = False
        if "ft_merge" in policies:
            dom = pareto_dominates(qB, cB, _q_series(policies["ft_merge"]), _cost_series(policies["ft_merge"]))
            p3_here = dom["dominates"]
            if p3_here:
                p3_mixes += 1
        # --- P4: both beats each ablation.
        if "cost_only" in policies and pareto_dominates(
                qB, cB, _q_series(policies["cost_only"]), _cost_series(policies["cost_only"]))["dominates"]:
            p4_beats_cost += 1
        if "damage_only" in policies and pareto_dominates(
                qB, cB, _q_series(policies["damage_only"]), _cost_series(policies["damage_only"]))["dominates"]:
            p4_beats_damage += 1
        # --- discovery aggregation (headline on damaging_gt; MIX-A honours CI-only power floor).
        disc = _aggregate_discovery(policies["both"], mix)
        per_mix[mix] = {"P1_dominates_all_fixed": p1_here, "P1_detail": p1_detail,
                        "P3_beats_ft_merge": p3_here, "discovery": disc}

    P1 = p1_mixes >= C.GATE["P1_min_mixes"]
    P3 = p3_mixes >= C.GATE["P3_min_mixes"]
    P4 = (p4_beats_cost >= 1) and (p4_beats_damage >= 1)
    P2, p2_detail = _p2_structural_verdict(results.get("MIX_C", {}))

    verdict = "PASS" if (P1 and P2) else ("KILL" if (not P1 and not P2) else "GREY")
    # A verdict is only meaningful over the COMPLETE pre-registered grid. Any missing mix or
    # missing fixed-strategy baseline means absent evidence, not negative evidence — a partial
    # wave must never print KILL (wave-1 wrote KILL off 1/9 cells on 2026-07-16). Refuse instead.
    missing = {m: [s for s, d in per_mix[m]["P1_detail"].items() if d == "MISSING"]
               for m in per_mix if any(d == "MISSING" for d in per_mix[m]["P1_detail"].values())}
    absent_mixes = [m for m in C.MIXES if m not in per_mix]
    if missing or absent_mixes:
        return {
            "VERDICT": "INCOMPLETE",
            "incomplete": {"absent_mixes": absent_mixes, "missing_baselines": missing},
            "note": "verdict refused: pre-registered grid not fully on disk; no PASS/GREY/KILL "
                    "is derivable from a partial wave.",
            "per_mix": per_mix,
            "measured_vs_synthetic_cost_ratio_check": _measured_vs_synthetic_cost_ratio(results),
        }
    return {
        "VERDICT": verdict,
        "P1": bool(P1), "P1_mixes": p1_mixes,
        "P2": bool(P2), "P2_detail": p2_detail,
        "P3": bool(P3), "P3_mixes": p3_mixes,
        "P4": bool(P4), "P4_beats_cost_only_mixes": p4_beats_cost,
        "P4_beats_damage_only_mixes": p4_beats_damage,
        "gate": {"P1_min_mixes": C.GATE["P1_min_mixes"], "P3_min_mixes": C.GATE["P3_min_mixes"],
                 "rule": "PASS=P1∧P2 ; KILL=¬P1∧¬P2 ; GREY=exactly one. P3/P4 sharpen only."},
        "measured_vs_synthetic_cost_ratio_check": _measured_vs_synthetic_cost_ratio(results),
        "per_mix": per_mix,
    }


def _aggregate_discovery(both_cells: List[Dict], mix: str) -> Dict:
    recalls = [c["discovery"]["recall_at_decile"] for c in both_cells
               if c["discovery"]["recall_at_decile"] == c["discovery"]["recall_at_decile"]]
    n_gt = int(np.mean([c["discovery"]["n_damaging_gt"] for c in both_cells])) if both_cells else 0
    # rev.5: CI-only unless damaging_gt >= the pinned point-floor (50); config can force CI-only.
    below_floor = n_gt < C.DISCOVERY_POINT_FLOOR
    ci_only = below_floor or bool(C.MIXES[mix].get("ci_only_discovery", False))
    out = {"mean_recall_at_decile": float(np.mean(recalls)) if recalls else float("nan"),
           "mean_lift": (float(np.mean(recalls)) / C.PREDICTOR_TOPDECILE_CHANCE) if recalls else float("nan"),
           "predictor_ceiling": C.PREDICTOR_TOPDECILE_RECALL_CEILING_L12,
           "n_damaging_gt": n_gt, "point_floor": C.DISCOVERY_POINT_FLOOR,
           "ci_only": ci_only, "point_claim_allowed": (not ci_only)}
    if recalls:
        # seed-level percentile CI on the recall — ALWAYS reported (rev.5: report both CI and point).
        out["recall_ci"] = ([float(np.quantile(recalls, 0.025)), float(np.quantile(recalls, 0.975))]
                            if len(recalls) > 1 else [recalls[0], recalls[0]])
        # point estimate reported only when the floor is met (else suppressed, CI stands alone).
        out["point_recall"] = out["mean_recall_at_decile"] if not ci_only else None
    if ci_only:
        out["point_claim"] = "SUPPRESSED_below_point_floor (rev.5: CI-only)"
    # per-fact-type discovery (MINOR-A follow-up): mean recall + mean n_damaging_gt per fact_type,
    # reported ALONGSIDE the pooled headline. Back-compatible with cells lacking the breakdown.
    per_ft = {}
    for ft in C.FACT_TYPES:
        r_ft = [c["discovery"].get("per_fact_type", {}).get(ft, {}).get("recall_at_decile")
                for c in both_cells]
        r_ft = [x for x in r_ft if x is not None and x == x]   # drop None + NaN
        n_ft = [c["discovery"].get("per_fact_type", {}).get(ft, {}).get("n_damaging_gt", 0)
                for c in both_cells]
        per_ft[ft] = {"mean_recall_at_decile": float(np.mean(r_ft)) if r_ft else float("nan"),
                      "n_damaging_gt": int(np.mean(n_ft)) if n_ft else 0}
    out["per_fact_type"] = per_ft
    return out


def _measured_vs_synthetic_cost_ratio(results: Dict) -> Dict:
    """M3: per-arm MEASURED (real-cell routing) vs SYNTHETIC (frozen SyntheticClock) serving GPU-s.

    Wave-1's router decides on the prereg-frozen synthetic cost table; this reports the measured/
    synthetic per-arm serving ratio so a post-wave check can flag material divergence (→ a wave-1b
    sensitivity re-run with measured router costs). Real cells carry routing.serve_gpu_s_by_arm +
    arm_counts; synthetic cells carry neither, so this yields an empty per_arm map for a synth run."""
    from ..cost_harness import SyntheticClock
    clk = SyntheticClock()
    agg: Dict[str, List[float]] = {}
    for _mix, policies in results.items():
        for policy, cells in policies.items():
            if policy == "_p2" or not isinstance(cells, list):
                continue
            for c in cells:
                routing = c.get("routing", {}) or {}
                sgba = routing.get("serve_gpu_s_by_arm", {}) or {}
                ac = routing.get("arm_counts", {}) or {}
                for arm, s in sgba.items():
                    agg.setdefault(arm, [0.0, 0.0])
                    agg[arm][0] += float(s)
                    agg[arm][1] += float(ac.get(arm, 0))
    per_arm = {}
    for arm, (s, n) in agg.items():
        if n > 0:
            measured = s / n
            synth = clk.serve(arm, n_queries=1, store_n=0, k=C.RAG_TOP_K).gpu_s
            per_arm[arm] = {"measured_serve_gpu_s": measured, "synthetic_serve_gpu_s": synth,
                            "measured_over_synthetic": (measured / synth) if synth else None,
                            "n_queries": int(n)}
    return {"per_arm": per_arm,
            "note": ("wave-1 router uses the frozen synthetic cost table; measured-vs-synthetic "
                     "per-arm serving ratios reported here — material divergence triggers a wave-1b "
                     "sensitivity re-run with measured router costs (PREREG §c decision note)."),
            "applicable": bool(per_arm)}


def _p2_structural_verdict(mixC: Dict) -> Tuple[bool, Dict]:
    """P2 = the four-term structural conjunction in MIX-C. NEVER reads ErrorCost_eval."""
    p2 = mixC.get("_p2")
    if not p2:
        return False, {"error": "MIX_C _p2 structural quantities missing"}
    t1 = (p2["exposure_edit"] == 0.0) and (p2["exposure_rag"] > 0.5)
    t2 = p2["footprint_delta"] > 0.0
    t3 = p2["overhead_delta"] > 0.0
    t4 = p2.get("router_edit_majority_on_privacy", 0.0) > 0.5
    detail = {"exposure_edit_lt_rag": bool(t1), "footprint_delta_positive": bool(t2),
              "overhead_delta_positive": bool(t3), "router_selects_edit_majority": bool(t4),
              "values": p2}
    return bool(t1 and t2 and t3 and t4), detail


# ---------------------------------------------------------------- IO
def _load_p2(cells_dir: str, model: Optional[str], provenance: Optional[str]) -> Optional[Dict]:
    """Load the namespaced MIX-C structural P2 file `p2_{model}_{provenance}_MIX_C.json`.

    Falls back to the single namespaced p2 file if expectation is unset, then to the legacy
    un-namespaced `p2_MIX_C.json` (older cells). Ambiguity (multiple namespaced p2 files with no
    expectation given) is a hard error — never guess which model/provenance the P2 belongs to.
    """
    if model is not None and provenance is not None:
        p = os.path.join(cells_dir, f"p2_{model}_{provenance}_MIX_C.json")
        if os.path.exists(p):
            return json.load(open(p))
    cands = sorted(glob.glob(os.path.join(cells_dir, "p2_*_MIX_C.json")))
    if len(cands) == 1 and (model is None or provenance is None):
        return json.load(open(cands[0]))
    if len(cands) > 1 and (model is None or provenance is None):
        raise ValueError(
            f"ambiguous P2: {len(cands)} namespaced p2_*_MIX_C.json files in {cells_dir} "
            f"({[os.path.basename(c) for c in cands]}); pass --expect_model/--expect_provenance.")
    legacy = os.path.join(cells_dir, "p2_MIX_C.json")
    if os.path.exists(legacy):
        return json.load(open(legacy))
    return None


def load_results(cells_dir: str, expect_model: Optional[str] = None,
                 expect_provenance: Optional[str] = None) -> Dict[str, Dict[str, List[Dict]]]:
    """Load cell_*.json produced by run_stream into results[mix][policy]=[cells] (MAJOR-2).

    Cell filename convention: `cell_{model}_{provenance}_{MIX}_{policy}_s{seed}.json`; each cell
    body carries `model` + `provenance` keys (the filename is not parsed — the body is the truth).

    REFUSES a mixed cell set (hard error listing offenders) so a real analyze can never silently
    score stale synthetic cells (or another model's cells) as its own:
      * with an explicit expectation → EVERY loaded cell must match `expect_model`/`expect_provenance`;
        any mismatch is an offender and aborts the run (also aborts if nothing matches);
      * without an expectation → all cells must share ONE (model, provenance); >1 distinct aborts.
    Cells lacking `model`/`provenance` (legacy, pre-MAJOR-2) count as `(None, None)` and are
    offenders whenever an expectation is set.
    """
    results: Dict[str, Dict[str, List[Dict]]] = {}
    offenders: List[Tuple[str, Optional[str], Optional[str]]] = []
    present = set()
    loaded = 0
    expecting = (expect_model is not None) or (expect_provenance is not None)
    for path in sorted(glob.glob(os.path.join(cells_dir, "cell_*.json"))):
        cell = json.load(open(path))
        model, prov = cell.get("model"), cell.get("provenance")
        present.add((model, prov))
        if expecting and ((expect_model is not None and model != expect_model)
                          or (expect_provenance is not None and prov != expect_provenance)):
            offenders.append((os.path.basename(path), model, prov))
            continue
        results.setdefault(cell["mix"], {}).setdefault(cell["policy"], []).append(cell)
        loaded += 1
    if expecting:
        if offenders:
            raise ValueError(
                f"REFUSING mixed cell set in {cells_dir}: expected model={expect_model!r} "
                f"provenance={expect_provenance!r}; {len(offenders)} offender(s): "
                f"{offenders[:20]}{' ...' if len(offenders) > 20 else ''}")
        if loaded == 0:
            raise ValueError(
                f"no cells match model={expect_model!r} provenance={expect_provenance!r} in {cells_dir} "
                f"(present: {sorted(present)})")
    elif len(present) > 1:
        raise ValueError(
            f"REFUSING mixed cell set in {cells_dir}: {len(present)} distinct (model, provenance) "
            f"pairs present {sorted(present)}; pass --expect_model/--expect_provenance to select one.")
    p2 = _load_p2(cells_dir, expect_model, expect_provenance)
    if p2 is not None:
        results.setdefault("MIX_C", {})["_p2"] = p2
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Frame-A frozen-gate analyzer")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--cells_dir", default=os.path.join(C.RESULTS_DIR, "cells"))
    ap.add_argument("--out", default=os.path.join(C.RESULTS_DIR, "frame_a_verdict.json"))
    ap.add_argument("--expect_model", default=None,
                    help="require every loaded cell to carry this model tag (MAJOR-2 provenance guard)")
    ap.add_argument("--expect_provenance", default=None, choices=[None, "synth", "real"],
                    help="require every loaded cell to carry this provenance (synth|real)")
    args = ap.parse_args()
    if args.selftest:
        _selftest(); return
    results = load_results(args.cells_dir, args.expect_model, args.expect_provenance)
    verdict = evaluate(results)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(verdict, open(args.out, "w"), indent=2)
    if verdict["VERDICT"] == "INCOMPLETE":
        print(f"VERDICT=INCOMPLETE (refused: {verdict['incomplete']}) -> {args.out}")
    else:
        print(f"VERDICT={verdict['VERDICT']}  P1={verdict['P1']} P2={verdict['P2']} "
              f"P3={verdict['P3']} P4={verdict['P4']} -> {args.out}")


# ---------------------------------------------------------------- selftest
def _planted(both_better: bool, p2_true: bool) -> Dict:
    """Construct a planted results dict: if both_better, `both` Pareto-dominates all; else it ties."""
    def cells(policy, q, cost, recall):
        return [{"mix": "M", "policy": policy, "seed": s,
                 "quality": {"Q": q + 0.001 * s}, "cost": {"total_gpu_s": cost},
                 "discovery": {"recall_at_decile": recall, "n_damaging_gt": 100,
                               "lift": recall / C.PREDICTOR_TOPDECILE_CHANCE}} for s in C.SEEDS]
    res = {}
    for mix in ("MIX_A", "MIX_B", "MIX_C"):
        pol = {}
        # both: high Q + low cost when both_better; else a genuine TIE with the fixed strategies
        # (same Q AND same cost → no Pareto dominance on either axis).
        pol["both"] = cells("both", 0.85 if both_better else 0.60, 10.0 if both_better else 20.0, 0.44)
        for s in FIXED_STRATEGIES:
            pol[s] = cells(s, 0.60, 20.0, 0.10)
        pol["ft_merge"] = cells("ft_merge", 0.62, 22.0, 0.10)
        pol["cost_only"] = cells("cost_only", 0.55, 12.0, 0.05)
        pol["damage_only"] = cells("damage_only", 0.58, 30.0, 0.42)
        res[mix] = pol
    res["MIX_C"]["_p2"] = {
        "exposure_edit": 0.0, "exposure_rag": 1.0,
        "footprint_delta": 1.0 if p2_true else -1.0,
        "overhead_delta": 0.6 if p2_true else -0.1,
        "router_edit_majority_on_privacy": 0.8 if p2_true else 0.2,
    }
    return res


def _selftest() -> None:
    # PASS world: both dominates + P2 holds.
    v = evaluate(_planted(both_better=True, p2_true=True))
    assert v["VERDICT"] == "PASS", v
    assert v["P1"] and v["P2"] and v["P3"] and v["P4"], v
    # KILL world: both ties (no dominance) + P2 fails.
    v2 = evaluate(_planted(both_better=False, p2_true=False))
    assert v2["P1"] is False and v2["P2"] is False and v2["VERDICT"] == "KILL", v2
    # GREY world: P2 holds but P1 fails.
    v3 = evaluate(_planted(both_better=False, p2_true=True))
    assert v3["VERDICT"] == "GREY", v3
    # Pareto predicate: strictly-better-Q-at-lower-cost dominates; equal ties do not.
    dom = pareto_dominates([0.9, 0.9, 0.9], [10, 10, 10], [0.6, 0.6, 0.6], [20, 20, 20])
    assert dom["dominates"]
    tie = pareto_dominates([0.6, 0.6, 0.6], [20, 20, 20], [0.6, 0.6, 0.6], [20, 20, 20])
    assert not tie["dominates"]
    # matches-then-undercuts: equal Q within CI at strictly lower cost dominates.
    mtu = pareto_dominates([0.60, 0.61, 0.60], [10, 10, 10], [0.60, 0.61, 0.60], [20, 20, 20])
    assert mtu["dominates"], mtu
    print("scorer.analyze_frame_a selftest: PASS")


if __name__ == "__main__":
    main()
