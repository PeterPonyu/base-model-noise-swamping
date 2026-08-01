"""q_ext_analysis.py — the M6 Q_ext gate analysis for Frame-A. CPU-only, ¥0, reads existing cells.

BINDING SPEC: `docs/plans/PREREG-FRAME-A-AMENDMENT-M6-QMETRIC-2026-07-30.md` (STATUS: RATIFIED).
Amends PREREG-FRAME-A-STREAM-2026-07-16 §2 (fixed composite weights) + MINOR-2 (Pareto predicate).

WHY THIS EXISTS. On the pre-registered `Q`, `always_grace` scores 0.9544 (MIX_A) / 0.8912 (MIX_B)
and thereby OUTSCORES THE ORACLE (0.7496 / 0.7333). A policy above the by-construction ceiling is
a self-refuting frontier: GRACE never touches weights, so its collateral is identically zero and
`w_loc=0.30` is a free 0.30 no weight-editing policy can contest. Referee FA-2 raises exactly this.
`Q_ext` charges the three currencies GRACE actually pays — capacity, serving latency, staleness:

    Q_ext = w_upd·A_upd + w_loc·A_loc + w_cum·A_cum + w_rip·A_rip
            − λ_cap·capacity_frac − λ_lat·latency_frac − λ_stale·stale_frac

The four `w` are UNCHANGED from the prereg. λ are FROZEN by the amendment at
(cap .15 / lat .10 / stale .15) and live in `scorer/scoring.py` as `Q_EXT_LAMBDA` — this module
imports them, it does not re-freeze them. Q_ext is EXPLORATORY / POST-HOC by the amendment's own
admission (all 63 cells were measured before it was written); the original `Q` remains the
pre-registered primary and is reported side-by-side in every table below (G-Q4).

WHAT THIS MODULE MAY AND MAY NOT DO. No GPU work is authorised by M6 and none is needed: this
reads `results/frame_a/cells/cell_*.json` only. It NEVER writes a cell, never touches
`run_stream.py` / `real_replay.py` / `config.py` / any scorer file (a refill driver holds those —
see the live-file-edit hazard rule), and it emits exactly one artifact:
`results/frame_a/Q_ext_analysis_20260731.json`.

THE FRACS COME FROM PERSISTED AGGREGATES — no per-record rows are on disk (cells persist only
`quality{...}` / `cost{...}` / `routing{arm_counts, *_by_arm}`), so `scoring.quality_ext(rows=...)`
cannot be called on real data at all. See RESOLVED AMBIGUITY 1 below. Each frac's aggregate
reading, and the conservatism it buys, is documented at its `_frac_*` function.

INTEGRITY (M6 §Integrity): the amendment demands the three fracs be recomputed ONCE from the raw
rows by a reviewer pass that did not write this scorer, before any paper use. This module therefore
prints EVERY frac PER CELL into the JSON under `per_cell`, together with the exact numerator and
denominator it used, so that recompute needs this file's output but NOT this file's code.
Author and review in separate passes. Quarantine failures; never repair in place.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List, Optional, Tuple

from . import config as C
from .scorer import scoring
from .scorer.analyze_frame_a import pareto_dominates

OUT_DEFAULT = os.path.join(C.RESULTS_DIR, "Q_ext_analysis_20260731.json")
CENTRAL = dict(scoring.Q_EXT_LAMBDA)              # {"cap":0.15,"lat":0.10,"stale":0.15} — FROZEN
BUDGET = scoring.Q_EXT_CAPACITY_BUDGET            # 200.0 entries (M6: one per stream edit)
STREAM_LEN = float(C.STREAM_LEN_WAVE1)            # 500 updates per stream instance (wave 1)

# The 11 pre-registered policies (run_stream.POLICIES order; imported by NAME, not from run_stream,
# so this module never imports the file the refill driver is executing).
POLICIES = ("both", "cost_only", "damage_only", "oracle",
            "always_edit", "always_grace", "always_rag", "always_ft", "always_reject",
            "random", "ft_merge")
MIXES = ("MIX_A", "MIX_B")        # wave-1 real cells on disk; MIX_C hosts P2 only (no Q frontier).

# M2 PROVENANCE (verified 2026-07-31 by reading the code, not by trusting a tag). `serve_overhead`
# is NOT measured on the real path: `arms/base.py:85 Arm.serve_overhead` reads `SyntheticClock`, and
# NO class in `arms/real_backends.py` overrides it. Closed-form check reproduces the cells exactly:
#   grace: Σ_{n=1..500} (0.15 + 2e-3·n) = 325.50  == every always_grace cell's serve_overhead_total
#   rag  : 500 · (5 · 12 / 100)         = 300.00  == every always_rag cell's serve_overhead_total
# Under the M2 rule a synthetic value is allowed ONLY with an explicit tag, never silently — so the
# latency term is tagged `synthetic-frozen-clock` throughout, NOT "measured-real-replay". This is a
# CAVEAT ON λ_lat ONLY (max 0.10 of the composite); the capacity and staleness terms are unaffected.
LATENCY_PROVENANCE = "synthetic-frozen-clock"
LATENCY_PROVENANCE_NOTE = (
    "serve_overhead_total is produced by the prereg-frozen SyntheticClock (arms/base.py:85 "
    "Arm.serve_overhead; no real_backends subclass overrides it). Closed forms reproduce the "
    "cells exactly (grace 325.50, rag 300.00), so this is a SYNTHETIC latency under the M2 "
    "explicit-tag rule, not a measured-real-replay one. Affects lambda_lat (<=0.10) only.")


# ================================================================ the three fracs
# Each helper returns (frac, detail-dict). The detail dict carries the raw numerator/denominator so
# the M6 reviewer recompute can be done from the JSON alone, without re-reading this code.
def _frac_capacity(cell: Dict) -> Tuple[float, Dict]:
    """capacity_frac = grace codebook entries / 200, clipped [0,1].

    AGGREGATE READING: `routing.arm_counts['grace']` is the count of updates absorbed by the GRACE
    arm, and one GRACE install == one codebook entry, so arm_counts['grace'] IS the codebook size at
    end of stream. Weight-editing (`edit`, `ft`) and `rag` and `reject` arms store nothing in the
    codebook, so their grace count is absent ⇒ 0 ⇒ capacity_frac 0, exactly as M6 specifies
    ("0 for weight-editing policies, which store nothing").

    DESIGN TENSION, FLAGGED (do not bury): the M6 budget is 200 entries "one per edit in the
    stream", but wave-1 streams are 500 updates long (STREAM_LEN_WAVE1=500, the amendment was
    written against the 200-edit figure). Any policy sending >200 updates to GRACE therefore clips
    to 1.0 and the capacity term saturates — `always_grace` (500/500 to grace) and the router
    (486-490/500) are INDISTINGUISHABLE on this term at the frozen budget. The alternative
    stream-fraction view (grace/500, reported alongside as `capacity_frac_alt_stream`) separates
    them (1.000 vs 0.972) but is a DIFFERENT quantity — a share, not a budget-utilisation. The
    frozen budget=200 view is PRIMARY per the amendment; the alt view is a sensitivity note only
    and no verdict in this file is computed from it.
    """
    ac = (cell.get("routing", {}) or {}).get("arm_counts", {}) or {}
    entries = float(ac.get("grace", 0) or 0)
    frac = min(1.0, max(0.0, entries / BUDGET))
    alt = min(1.0, max(0.0, entries / STREAM_LEN))
    return frac, {"codebook_entries": entries, "budget": BUDGET, "capacity_frac": frac,
                  "capacity_frac_alt_stream": alt, "alt_denominator": STREAM_LEN,
                  "clipped": entries > BUDGET}


def _frac_latency_numerator(cell: Dict) -> float:
    """Per-query serving overhead = serve_overhead_total / 500 (the stream's query volume).

    The cell persists only the SUM over the stream (`cost.serve_overhead_total`); dividing by the
    stream length recovers the mean per-query overhead that `scoring.quality_ext` computes as
    `_mean([r.serve_overhead ...])` over the rows. Identical quantity, aggregate route.
    """
    return float(cell["cost"]["serve_overhead_total"]) / STREAM_LEN


def _frac_latency(cell: Dict, latency_max: float) -> Tuple[float, Dict]:
    """latency_frac = per-query overhead / the MIX's max per-query overhead, clipped [0,1].

    The denominator is the max ACROSS POLICIES WITHIN THE SAME MIX (M6: "the mix's max observed
    overhead across policies"), so policies are comparable and the worst policy in each mix pays
    exactly 1.0. Computed from the pooled per-(mix,policy) means, not per seed, so the normaliser
    does not wobble with seed coverage. Provenance is SYNTHETIC — see LATENCY_PROVENANCE above.
    """
    num = _frac_latency_numerator(cell)
    frac = min(1.0, max(0.0, num / latency_max)) if latency_max > 0 else 0.0
    return frac, {"serve_overhead_total": float(cell["cost"]["serve_overhead_total"]),
                  "query_volume": STREAM_LEN, "per_query_overhead": num,
                  "latency_max_in_mix": latency_max, "latency_frac": frac,
                  "latency_provenance": LATENCY_PROVENANCE}


def _frac_stale(cell: Dict) -> Tuple[float, Dict]:
    """stale_frac = 1 − A_upd (the no-installed-answer rate).

    AGGREGATE READING of M6's per-record definition ("fraction of scored queries answered by
    fallback rather than by the policy's own mechanism; codebook miss for GRACE, no-edit-installed
    rate for weight editors"). On persisted data the two collapse: `A_upd` is the fraction of
    applied updates whose efficacy query recalled `target_new` at end of stream
    (scoring.quality:60), i.e. the rate at which the policy's OWN mechanism answered; its
    complement is precisely the rate at which the unedited model answered instead — a codebook
    miss for GRACE, a not-installed edit for a weight editor, a reject/deferral for the rest.

    TWO DOCUMENTED CONSERVATISMS, both charging the penalty UPWARD (never in GRACE's favour):
      (a) `A_upd` is measured at END of stream, so forgetting-at-end (an FT-merge overwrite) is
          folded into 1−A_upd as well as being charged separately through `A_cum`. The stale term
          is therefore slightly conservative for FT-family policies.
      (b) `A_upd` is computed over APPLIED rows only, so rows the policy rejected outright
          (`always_reject`: A_upd=0) read as fully stale — which is the correct charge for a
          policy that installs nothing, but it means 1−A_upd is a "no answer from the policy's own
          mechanism" rate rather than a strict codebook-miss rate.
    Both readings are visible per cell in the JSON, so the reviewer recompute can substitute a
    strict per-record definition when the rows are re-derived.
    """
    a_upd = float(cell["quality"]["A_upd"])
    frac = min(1.0, max(0.0, 1.0 - a_upd))
    return frac, {"A_upd": a_upd, "stale_frac": frac,
                  "definition": "1 - A_upd (aggregate reading of the M6 per-record fallback rate)"}


# ================================================================ Q_ext from persisted aggregates
def q_ext_from_cell(cell: Dict, latency_max: float,
                    lambdas: Optional[Dict[str, float]] = None) -> Dict:
    """Q_ext for ONE cell from its persisted aggregates.

    The formula, the frozen λ, and the capacity budget all come from `scorer.scoring` (imported,
    never re-implemented). What this function supplies is only the AGGREGATE→frac mapping that
    `scoring.quality_ext` would otherwise take from per-record rows — which are not persisted.
    The arithmetic is asserted identical to `scoring.quality_ext` in `--selftest`.
    """
    lam = dict(CENTRAL) if lambdas is None else dict(lambdas)
    cap, cap_d = _frac_capacity(cell)
    lat, lat_d = _frac_latency(cell, latency_max)
    stale, stale_d = _frac_stale(cell)
    q = float(cell["quality"]["Q"])
    q_ext = q - lam["cap"] * cap - lam["lat"] * lat - lam["stale"] * stale
    return {"Q": q, "Q_ext": q_ext,
            "capacity_frac": cap, "latency_frac": lat, "stale_frac": stale,
            "lambda_cap": lam["cap"], "lambda_lat": lam["lat"], "lambda_stale": lam["stale"],
            "penalty_total": q - q_ext,
            "detail": {"capacity": cap_d, "latency": lat_d, "stale": stale_d}}


# ================================================================ load + pool
def load_cells(cells_dir: str, expect_model: Optional[str] = None,
               expect_provenance: Optional[str] = "real") -> Tuple[Dict, Dict]:
    """cells[mix][policy] = [cell,...] plus a coverage report.

    SKIPS MISSING CELLS GRACEFULLY BY DESIGN: a refill driver is re-running s2 cells right now, so
    absence is a coverage fact to report, never an error to raise (contrast `analyze_frame_a`,
    which must refuse a partial grid because it emits a PASS/GREY/KILL verdict; M6's gates are
    per-mix comparisons that remain computable from the seeds present, with the coverage stated).
    Provenance guard retained from MAJOR-2: a cell whose body disagrees with the expectation is an
    offender and is EXCLUDED, and every exclusion is listed in the report.
    """
    cells: Dict[str, Dict[str, List[Dict]]] = {}
    offenders: List[Dict] = []
    for path in sorted(glob.glob(os.path.join(cells_dir, "cell_*.json"))):
        try:
            cell = json.load(open(path))
        except (json.JSONDecodeError, OSError) as e:      # a mid-write refill cell: skip, report.
            offenders.append({"file": os.path.basename(path), "reason": f"unreadable: {e}"})
            continue
        model, prov = cell.get("model"), cell.get("provenance")
        if (expect_model is not None and model != expect_model) or \
           (expect_provenance is not None and prov != expect_provenance):
            offenders.append({"file": os.path.basename(path), "model": model, "provenance": prov,
                              "reason": "provenance/model mismatch (MAJOR-2 guard)"})
            continue
        if cell.get("mix") not in MIXES:                  # MIX_C carries no Q frontier (P2 only).
            continue
        cells.setdefault(cell["mix"], {}).setdefault(cell["policy"], []).append(cell)
    report = _coverage(cells, offenders)
    return cells, report


def _coverage(cells: Dict, offenders: List[Dict]) -> Dict:
    per_mix = {}
    n_present = n_expected = 0
    for mix in MIXES:
        pol = cells.get(mix, {})
        seeds = {p: sorted(c["seed"] for c in pol.get(p, [])) for p in POLICIES}
        missing = {p: [s for s in C.SEEDS if s not in seeds[p]] for p in POLICIES
                   if [s for s in C.SEEDS if s not in seeds[p]]}
        present = sum(len(v) for v in seeds.values())
        n_present += present
        n_expected += len(POLICIES) * len(C.SEEDS)
        per_mix[mix] = {"seeds_present": seeds, "missing_seeds": missing,
                        "n_present": present, "n_expected": len(POLICIES) * len(C.SEEDS),
                        "absent_policies": [p for p in POLICIES if not seeds[p]]}
    return {"per_mix": per_mix, "n_present": n_present, "n_expected": n_expected,
            "complete": n_present == n_expected, "excluded_files": offenders,
            "note": ("cells missing => reported as coverage, not an error: a refill driver may be "
                     "re-running s2 cells. Every gate below is computed over the seeds PRESENT and "
                     "its seed count is stated, so a later full-coverage rerun is comparable.")}


def latency_max_per_mix(cells: Dict) -> Dict[str, float]:
    """The per-mix latency normaliser: max over POLICIES of the seed-pooled per-query overhead.

    Pooled-then-maxed (not maxed over raw cells) so the denominator is a property of the mix's
    policy set rather than of whichever seed happened to land highest.
    """
    out = {}
    for mix, pol in cells.items():
        vals = []
        for cl in pol.values():
            if cl:
                vals.append(sum(_frac_latency_numerator(c) for c in cl) / len(cl))
        out[mix] = max(vals) if vals else 0.0
    return out


def pool(cells: Dict, lambdas: Optional[Dict[str, float]] = None) -> Dict:
    """Per (mix, policy): average over the seeds PRESENT (M6 inherits MINOR-2's mean-over-seeds).

    Pooling order: compute Q_ext per seed-cell, then average. Since Q_ext is affine in the fracs
    and the fracs are per-cell, mean(Q_ext) == Q_ext(mean fracs) exactly — the order is stated only
    so the reviewer recompute matches to the last decimal.
    """
    lat_max = latency_max_per_mix(cells)
    out: Dict[str, Dict[str, Dict]] = {}
    for mix in sorted(cells):
        out[mix] = {}
        for policy, cl in sorted(cells[mix].items()):
            if not cl:
                continue
            rows = [q_ext_from_cell(c, lat_max[mix], lambdas) for c in cl]
            out[mix][policy] = {
                "n_seeds": len(cl), "seeds": sorted(c["seed"] for c in cl),
                "Q": _mean([r["Q"] for r in rows]),
                "Q_ext": _mean([r["Q_ext"] for r in rows]),
                "capacity_frac": _mean([r["capacity_frac"] for r in rows]),
                "latency_frac": _mean([r["latency_frac"] for r in rows]),
                "stale_frac": _mean([r["stale_frac"] for r in rows]),
                "cost_total_gpu_s": _mean([float(c["cost"]["total_gpu_s"]) for c in cl]),
                "Q_series": [r["Q"] for r in rows],
                "Q_ext_series": [r["Q_ext"] for r in rows],
                "cost_series": [float(c["cost"]["total_gpu_s"]) for c in cl],
            }
    return {"per_mix": out, "latency_max_per_mix": lat_max}


def _mean(xs: List[float]) -> float:
    xs = [x for x in xs if x == x]
    return float(sum(xs) / len(xs)) if xs else float("nan")


# ================================================================ G-Q1 (sanity: nothing beats oracle)
def gate_q1(cells: Dict) -> Dict:
    """G-Q1: under Q_ext, NO policy exceeds the oracle in ANY of the 27 settings, in BOTH mixes.

    M6 is explicit that failure here is terminal for the metric: "If any policy still beats the
    oracle, Q_ext is also mis-specified: report that, do not proceed to G-Q2/G-Q3, and do not
    publish a frontier under it." This function therefore reports the violating (mix, policy,
    setting) triples exhaustively rather than a bare boolean — the identity of the violator is what
    tells the next amendment which currency is still uncharged.

    `oracle` itself is excluded from the comparison set (it cannot exceed itself), and the strict
    inequality is `>` with a 1e-12 tolerance so floating-point ties are not read as violations.
    """
    TOL = 1e-12
    grid = _grid_settings()
    violations, per_mix = [], {}
    for mix in sorted(cells):
        if "oracle" not in cells[mix]:
            per_mix[mix] = {"evaluable": False,
                            "reason": "oracle cells absent — G-Q1 not evaluable for this mix"}
            continue
        n_viol_settings = 0
        worst = {"policy": None, "margin": float("-inf"), "setting": None}
        for lam in grid:
            p = pool(cells, lam)["per_mix"][mix]
            q_or = p["oracle"]["Q_ext"]
            hit = False
            for policy, v in p.items():
                if policy == "oracle":
                    continue
                margin = v["Q_ext"] - q_or
                if margin > TOL:
                    hit = True
                    violations.append({"mix": mix, "policy": policy, "setting": lam,
                                       "Q_ext_policy": v["Q_ext"], "Q_ext_oracle": q_or,
                                       "margin": margin})
                    if margin > worst["margin"]:
                        worst = {"policy": policy, "margin": margin, "setting": lam}
            n_viol_settings += int(hit)
        per_mix[mix] = {"evaluable": True, "n_settings": len(grid),
                        "n_settings_with_violation": n_viol_settings,
                        "clean_settings": len(grid) - n_viol_settings,
                        "worst_violator": worst if worst["policy"] else None,
                        "passes": n_viol_settings == 0}
    evaluable = [m for m in per_mix if per_mix[m].get("evaluable")]
    passes = bool(evaluable) and all(per_mix[m]["passes"] for m in evaluable)
    return {"gate": "G-Q1", "passes": passes, "per_mix": per_mix,
            "n_violations": len(violations), "violations": violations[:200],
            "violations_truncated": len(violations) > 200,
            "rule": ("under Q_ext no policy may exceed the oracle in any of the 27 settings, in "
                     "both mixes; failure ⇒ report and do NOT proceed to G-Q2/G-Q3 verdicts, and "
                     "do NOT publish a frontier under Q_ext (M6 G-Q1)"),
            "evaluable_mixes": sorted(evaluable),
            "not_evaluable": {m: per_mix[m].get("reason") for m in per_mix
                              if not per_mix[m].get("evaluable")}}


# ================================================================ G-Q2 / G-Q3 (the primary verdict)
def gate_q2(cells: Dict, lambdas: Optional[Dict[str, float]] = None) -> Dict:
    """G-Q2 (primary): under Q_ext at the CENTRAL setting, `both` Pareto-dominates `always_grace`
    in AT LEAST ONE mix, using the UNCHANGED CI-level Pareto predicate from MINOR-2 (seed-level
    bootstrap over orderings 0/1/2).

    EXACT WORDING HONOURED: the amendment's headline for G-Q2 in the gate list is "the router
    (`both`) Pareto-dominates `always_grace` in at least one mix"; the task brief phrases the same
    gate as "always_grace must NOT dominate in both mixes". These are DIFFERENT propositions (a
    mutual non-dominance tie satisfies the second and fails the first), so both are computed and
    reported separately — `passes` follows the AMENDMENT's wording, and `always_grace_dominates_both`
    carries the brief's reading, which is also exactly the antecedent G-Q3 needs. No verdict is
    left implicit.

    Cost axis and predicate are imported unchanged from `analyze_frame_a.pareto_dominates`; only the
    QUALITY series is swapped Q→Q_ext. That is the whole of M6's intervention on the predicate.
    """
    lam = dict(CENTRAL) if lambdas is None else dict(lambdas)
    p = pool(cells, lam)["per_mix"]
    per_mix, router_wins, grace_dominates = {}, [], []
    for mix in sorted(p):
        if "both" not in p[mix] or "always_grace" not in p[mix]:
            per_mix[mix] = {"evaluable": False,
                            "reason": "need both `both` and `always_grace` cells in this mix"}
            continue
        b, g = p[mix]["both"], p[mix]["always_grace"]
        fwd = pareto_dominates(b["Q_ext_series"], b["cost_series"],
                              g["Q_ext_series"], g["cost_series"])
        rev = pareto_dominates(g["Q_ext_series"], g["cost_series"],
                              b["Q_ext_series"], b["cost_series"])
        # Q-only comparison reported alongside: with a 3-seed bootstrap the CI predicate is weak,
        # and a reader must be able to see whether a non-dominance is a genuine tie or a power floor.
        per_mix[mix] = {
            "evaluable": True,
            "n_seeds_both": b["n_seeds"], "n_seeds_always_grace": g["n_seeds"],
            "both_Q_ext": b["Q_ext"], "always_grace_Q_ext": g["Q_ext"],
            "both_Q": b["Q"], "always_grace_Q": g["Q"],
            "delta_Q_ext_both_minus_grace": b["Q_ext"] - g["Q_ext"],
            "both_cost": b["cost_total_gpu_s"], "always_grace_cost": g["cost_total_gpu_s"],
            "router_dominates_always_grace": fwd["dominates"], "router_vs_grace_detail": fwd,
            "always_grace_dominates_router": rev["dominates"], "grace_vs_router_detail": rev,
            "mutual_non_dominance": (not fwd["dominates"]) and (not rev["dominates"]),
        }
        router_wins.append(fwd["dominates"])
        grace_dominates.append(rev["dominates"])
    evaluable = [m for m in per_mix if per_mix[m].get("evaluable")]
    return {"gate": "G-Q2", "lambdas": lam,
            "passes": bool(any(router_wins)),
            "n_mixes_router_dominates": int(sum(router_wins)),
            "always_grace_dominates_both": bool(grace_dominates) and all(grace_dominates),
            "n_mixes_always_grace_dominates": int(sum(grace_dominates)),
            "rule": ("M6 G-Q2 as written: router (`both`) Pareto-dominates always_grace in >=1 mix "
                     "at the central lambda, CI-level MINOR-2 predicate on (Q_ext, total_gpu_s). "
                     "The brief's complementary reading (always_grace must NOT dominate in both) is "
                     "reported as always_grace_dominates_both."),
            "per_mix": per_mix, "evaluable_mixes": sorted(evaluable)}


def gate_q3(cells: Dict, grid_summary: Dict) -> Dict:
    """G-Q3 (honest-negative gate). Triggered iff `always_grace` still dominates in BOTH mixes
    across the MAJORITY of the grid (>13/27 settings, i.e. >50%).

    If triggered, the finding to publish is: on this stream, a zero-damage codebook IS the right
    policy and routing does not pay. M6 forbids extending the metric further to manufacture a
    router win, so this function emits the statement text rather than a suggestion to iterate.
    A triggered G-Q3 is a publishable result, not a failed wave.
    """
    n = grid_summary["n_settings"]
    k = grid_summary["verdicts"]["always_grace_dominates_both_mixes"]["n_settings_holding"]
    triggered = k > n / 2.0
    return {"gate": "G-Q3", "triggered": bool(triggered),
            "n_settings_always_grace_dominates_both": k, "n_settings": n,
            "fraction": f"{k}/{n}",
            "rule": "triggered iff always_grace dominates in BOTH mixes in a majority of the grid",
            "statement": (
                "HONEST NEGATIVE (M6 G-Q3): on this stream a zero-damage codebook is the right "
                "policy and routing does not pay. Reported as the finding; extending the metric "
                "further to manufacture a router win is forbidden by the amendment."
                if triggered else
                "not triggered: always_grace does not dominate both mixes across a grid majority"),
            "forbidden_next_step": "further metric extension to manufacture a router win (M6 G-Q3)"}


# ================================================================ G-Q4 (no-op check / rank shifts)
def gate_q4(cells: Dict, lambdas: Optional[Dict[str, float]] = None) -> Dict:
    """G-Q4: Q vs Q_ext side by side for all 11 policies × 2 mixes; any policy whose RANK moves by
    more than 2 positions is flagged and explained in text (M6: the extension's effect must be
    visible, not buried in a frontier plot).

    Ranks are 1 = best (highest score), computed within a mix over the policies PRESENT; ties take
    the average rank so a tie cannot manufacture a spurious shift. The `explanation` string is
    generated from the dominant penalty term, so the flag arrives with its cause attached.
    """
    lam = dict(CENTRAL) if lambdas is None else dict(lambdas)
    p = pool(cells, lam)["per_mix"]
    out = {}
    for mix in sorted(p):
        pol = p[mix]
        rq = _ranks({k: v["Q"] for k, v in pol.items()})
        rx = _ranks({k: v["Q_ext"] for k, v in pol.items()})
        table, flagged = [], []
        for policy in sorted(pol, key=lambda k: -pol[k]["Q_ext"]):
            v = pol[policy]
            shift = rq[policy] - rx[policy]        # >0 = moved UP (better) under Q_ext
            row = {"policy": policy, "n_seeds": v["n_seeds"],
                   "Q": v["Q"], "Q_ext": v["Q_ext"], "penalty": v["Q"] - v["Q_ext"],
                   "rank_Q": rq[policy], "rank_Q_ext": rx[policy], "rank_shift": shift,
                   "capacity_frac": v["capacity_frac"], "latency_frac": v["latency_frac"],
                   "stale_frac": v["stale_frac"]}
            if abs(shift) > 2:
                row["flag"] = "RANK_SHIFT_GT_2"
                row["explanation"] = _explain_shift(policy, v, lam, shift)
                flagged.append(row)
            table.append(row)
        out[mix] = {"n_policies": len(pol), "table": table, "flagged": flagged,
                    "n_flagged": len(flagged)}
    return {"gate": "G-Q4", "lambdas": lam, "per_mix": out,
            "rule": ("report Q and Q_ext side by side for every policy in every mix; |rank shift| "
                     "> 2 must be explained in text (M6 G-Q4)"),
            "n_flagged_total": sum(v["n_flagged"] for v in out.values())}


def _ranks(scores: Dict[str, float]) -> Dict[str, float]:
    """1 = best (highest). Ties share the average rank (so ties never fake a shift)."""
    order = sorted(scores.items(), key=lambda kv: -kv[1])
    ranks: Dict[str, float] = {}
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and abs(order[j + 1][1] - order[i][1]) < 1e-12:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k][0]] = avg
        i = j + 1
    return ranks


def _explain_shift(policy: str, v: Dict, lam: Dict[str, float], shift: float) -> str:
    terms = {"capacity": lam["cap"] * v["capacity_frac"],
             "latency": lam["lat"] * v["latency_frac"],
             "staleness": lam["stale"] * v["stale_frac"]}
    dom = max(terms, key=lambda k: terms[k])
    direction = "DOWN" if shift < 0 else "UP"
    return (f"{policy} moved {direction} {abs(shift):.1f} positions: total penalty "
            f"{sum(terms.values()):.4f} (capacity {terms['capacity']:.4f}, latency "
            f"{terms['latency']:.4f}, staleness {terms['staleness']:.4f}); dominated by the "
            f"{dom} term. A {direction}ward move means the extension charged this policy "
            f"{'more' if shift < 0 else 'less'} than its Q-neighbours, which is the intended "
            f"visible effect of M6 rather than a metric artifact.")


# ================================================================ 27-setting sensitivity grid
def _grid_settings() -> List[Dict[str, float]]:
    """The full 27-setting grid, iterated in the SAME nesting order as
    `scoring.q_ext_sensitivity_grid` (cap → lat → stale) so grid index i in this file's output and
    grid index i from the scorer's own helper refer to the same λ triple. The λ values themselves
    are read from `scoring.Q_EXT_GRID` — frozen there by M6, never redefined here."""
    g = scoring.Q_EXT_GRID
    return [{"cap": lc, "lat": ll, "stale": ls}
            for lc in g["cap"] for ll in g["lat"] for ls in g["stale"]]


def sensitivity_grid(cells: Dict) -> Dict:
    """Per-setting verdicts over the full grid, reported as EXACT fractions.

    M6: "A verdict that holds in fewer than 27/27 must be reported with its exact fraction, never
    rounded up to 'robust'." So every verdict carries `n_settings_holding`, the literal `k/27`
    string, and the explicit `robust_all_settings` boolean — there is no field in this output that
    a reader can mistake for a rounded claim.
    """
    grid = _grid_settings()
    verdicts = {
        "no_policy_exceeds_oracle_both_mixes": [],
        "router_dominates_always_grace_at_least_one_mix": [],
        "always_grace_dominates_both_mixes": [],
        "always_grace_Q_ext_above_oracle_MIX_A": [],
        "always_grace_Q_ext_above_oracle_MIX_B": [],
    }
    per_setting = []
    for lam in grid:
        p = pool(cells, lam)["per_mix"]
        # (1) oracle ceiling respected in every mix that has an oracle
        ok_oracle, above = True, {}
        for mix in sorted(p):
            if "oracle" not in p[mix]:
                continue
            q_or = p[mix]["oracle"]["Q_ext"]
            worse = [k for k, v in p[mix].items() if k != "oracle" and v["Q_ext"] > q_or + 1e-12]
            above[mix] = worse
            ok_oracle = ok_oracle and not worse
        # (2)/(3) router-vs-grace dominance both directions, per mix
        router_win, grace_win = [], []
        for mix in sorted(p):
            if "both" not in p[mix] or "always_grace" not in p[mix]:
                continue
            b, g = p[mix]["both"], p[mix]["always_grace"]
            router_win.append(pareto_dominates(b["Q_ext_series"], b["cost_series"],
                                               g["Q_ext_series"], g["cost_series"])["dominates"])
            grace_win.append(pareto_dominates(g["Q_ext_series"], g["cost_series"],
                                              b["Q_ext_series"], b["cost_series"])["dominates"])
        verdicts["no_policy_exceeds_oracle_both_mixes"].append(ok_oracle)
        verdicts["router_dominates_always_grace_at_least_one_mix"].append(bool(any(router_win)))
        verdicts["always_grace_dominates_both_mixes"].append(
            bool(grace_win) and all(grace_win))
        for mix, key in (("MIX_A", "always_grace_Q_ext_above_oracle_MIX_A"),
                         ("MIX_B", "always_grace_Q_ext_above_oracle_MIX_B")):
            v = (mix in p and "always_grace" in p[mix] and "oracle" in p[mix]
                 and p[mix]["always_grace"]["Q_ext"] > p[mix]["oracle"]["Q_ext"] + 1e-12)
            verdicts[key].append(bool(v))
        per_setting.append({
            "lambdas": lam, "oracle_ceiling_respected": ok_oracle,
            "policies_above_oracle": above,
            "router_dominates_grace_per_mix": router_win,
            "grace_dominates_router_per_mix": grace_win,
            "Q_ext": {mix: {k: v["Q_ext"] for k, v in p[mix].items()} for mix in sorted(p)},
        })
    n = len(grid)
    summary = {}
    for k, flags in verdicts.items():
        holding = int(sum(flags))
        summary[k] = {"n_settings_holding": holding, "n_settings": n,
                      "fraction": f"{holding}/{n}",
                      "fraction_float": holding / n if n else float("nan"),
                      "robust_all_settings": holding == n,
                      "reporting_rule": "exact fraction; never rounded up to 'robust' (M6)"}
    return {"n_settings": n, "verdicts": summary, "per_setting": per_setting,
            "lambda_grid": {k: list(v) for k, v in scoring.Q_EXT_GRID.items()}}


# ================================================================ integrity dump (M6 §Integrity)
def per_cell_dump(cells: Dict) -> List[Dict]:
    """EVERY frac for EVERY cell, machine-readable, with numerators and denominators.

    This is the M6 integrity artifact: the amendment requires an independent reviewer who did not
    write this scorer to recompute `capacity_frac`, `latency_frac` and `stale_frac` once from the
    raw rows before any paper use. Emitting the inputs (arm_counts.grace, serve_overhead_total,
    A_upd) alongside the outputs means the reviewer's recompute can be checked against this file
    WITHOUT executing it — a disagreement localises to a specific cell and a specific frac.
    """
    lat_max = latency_max_per_mix(cells)
    dump = []
    for mix in sorted(cells):
        for policy in sorted(cells[mix]):
            for c in sorted(cells[mix][policy], key=lambda x: x["seed"]):
                r = q_ext_from_cell(c, lat_max[mix], CENTRAL)
                dump.append({
                    "mix": mix, "policy": policy, "seed": c["seed"],
                    "model": c.get("model"), "provenance": c.get("provenance"),
                    "stream_hash": c.get("stream_hash"),
                    "arm_counts": (c.get("routing", {}) or {}).get("arm_counts", {}),
                    "inputs": {"A_upd": float(c["quality"]["A_upd"]),
                               "A_loc": float(c["quality"]["A_loc"]),
                               "A_cum": float(c["quality"]["A_cum"]),
                               "A_rip": float(c["quality"]["A_rip"]),
                               "Q_on_disk": float(c["quality"]["Q"]),
                               "serve_overhead_total": float(c["cost"]["serve_overhead_total"]),
                               "total_gpu_s": float(c["cost"]["total_gpu_s"])},
                    "fracs": {"capacity_frac": r["capacity_frac"],
                              "latency_frac": r["latency_frac"],
                              "stale_frac": r["stale_frac"]},
                    "frac_derivation": r["detail"],
                    "Q": r["Q"], "Q_ext_central": r["Q_ext"], "penalty": r["penalty_total"],
                })
    return dump


# ================================================================ driver
def analyze(cells_dir: str, expect_model: Optional[str] = None,
            expect_provenance: Optional[str] = "real") -> Dict:
    """Run the whole M6 gate battery and return the artifact dict (no file IO here)."""
    cells, coverage = load_cells(cells_dir, expect_model, expect_provenance)
    if not cells:
        return {"amendment": "M6", "ERROR": "no usable cells found",
                "cells_dir": cells_dir, "coverage": coverage}
    central = pool(cells, CENTRAL)
    grid = sensitivity_grid(cells)
    q1 = gate_q1(cells)
    q4 = gate_q4(cells, CENTRAL)                      # computed for the record regardless of G-Q1
    # M6 G-Q1 is a HARD STOP on the VERDICTS (not on the tables): "do not proceed to G-Q2/G-Q3, and
    # do not publish a frontier under it". So G-Q2/G-Q3 are still COMPUTED (the record must show
    # what they would have said) but flagged `verdict_suppressed_by_G_Q1` and must not be quoted.
    q2 = gate_q2(cells, CENTRAL)
    q3 = gate_q3(cells, grid)
    if not q1["passes"]:
        for g in (q2, q3):
            g["verdict_suppressed_by_G_Q1"] = True
            g["suppression_note"] = (
                "G-Q1 FAILED: Q_ext is still mis-specified, so this gate's verdict must NOT be "
                "quoted and no frontier may be published under Q_ext (M6 G-Q1). The numbers are "
                "retained for the record only.")
    return {
        "amendment": "PREREG-FRAME-A-AMENDMENT-M6-QMETRIC-2026-07-30 (STATUS: RATIFIED)",
        "generated_by": "experiments/frame_a/q_ext_analysis.py",
        "status_labels": {
            "Q": "PRE-REGISTERED PRIMARY (PREREG-FRAME-A-STREAM-2026-07-16 §2) — reported in full",
            "Q_ext": "EXPLORATORY / POST-HOC (M6 is post-data by its own admission)",
        },
        "formula": ("Q_ext = w_upd*A_upd + w_loc*A_loc + w_cum*A_cum + w_rip*A_rip "
                    "- lambda_cap*capacity_frac - lambda_lat*latency_frac "
                    "- lambda_stale*stale_frac"),
        "w_weights": dict(C.Q_WEIGHTS),
        "lambda_central_frozen": dict(CENTRAL),
        "capacity_budget": BUDGET, "stream_len": STREAM_LEN,
        "frac_sources": {
            "capacity_frac": "routing.arm_counts['grace'] / 200, clipped [0,1] "
                             "(one GRACE install = one codebook entry; edit/ft/rag/reject store "
                             "nothing in the codebook => 0)",
            "capacity_frac_alt_stream": "routing.arm_counts['grace'] / 500 — SENSITIVITY NOTE ONLY; "
                                        "no verdict uses it (see design tension below)",
            "latency_frac": "(cost.serve_overhead_total / 500) / max-over-policies-in-same-mix, "
                            "clipped [0,1]",
            "stale_frac": "1 - quality.A_upd (aggregate reading of the M6 per-record fallback rate)",
        },
        "design_tensions": [
            "CAPACITY BUDGET vs STREAM LENGTH: the M6 budget is 200 entries ('one per edit in the "
            "stream') but wave-1 streams carry 500 updates, so every policy sending >200 updates to "
            "GRACE clips to capacity_frac=1.0. always_grace (500 to grace) and the router (486-490 "
            "to grace) are therefore INDISTINGUISHABLE on the capacity term at the frozen budget. "
            "The alt stream-fraction view separates them (1.000 vs ~0.972) but measures a share, "
            "not budget utilisation; it is reported as a note and drives no verdict. Resolving this "
            "properly needs a USER decision on the budget (200 as frozen, or 500 = one per update).",
            "LATENCY PROVENANCE: " + LATENCY_PROVENANCE_NOTE,
            "STALE_FRAC CONSERVATISM: 1-A_upd is measured at end of stream, so FT-merge forgetting "
            "is charged both here and through A_cum; and rows a policy rejected outright read as "
            "fully stale. Both push the penalty UP, never in GRACE's favour.",
        ],
        "coverage": coverage,
        "latency_provenance": LATENCY_PROVENANCE,
        "latency_provenance_note": LATENCY_PROVENANCE_NOTE,
        "central": central,
        "gates": {"G_Q1": q1, "G_Q2": q2, "G_Q3": q3, "G_Q4": q4},
        "sensitivity_grid": grid,
        "per_cell": per_cell_dump(cells),
        "integrity_note": (
            "M6 §Integrity: capacity_frac / latency_frac / stale_frac must each be recomputed once "
            "from the raw per-record rows by a reviewer pass that did NOT write this scorer, before "
            "any paper use. `per_cell` carries every numerator and denominator so the recompute is "
            "checkable without running this code. Author and review in separate passes. Quarantine "
            "failures; never repair in place."),
    }


# ================================================================ human summary
def print_summary(art: Dict) -> None:
    if "ERROR" in art:
        print(f"ERROR: {art['ERROR']} (cells_dir={art.get('cells_dir')})")
        return
    cov = art["coverage"]
    print("=" * 78)
    print("M6 Q_ext GATE ANALYSIS — Frame-A   [Q_ext is EXPLORATORY/POST-HOC; Q stays primary]")
    print("=" * 78)
    print(f"coverage: {cov['n_present']}/{cov['n_expected']} cells"
          f"{' (COMPLETE)' if cov['complete'] else ' (PARTIAL — refill in flight?)'}")
    for mix, d in cov["per_mix"].items():
        if d["missing_seeds"]:
            print(f"  {mix}: missing {d['missing_seeds']}")
        if d["absent_policies"]:
            print(f"  {mix}: ABSENT POLICIES {d['absent_policies']}")
    if cov["excluded_files"]:
        print(f"  excluded {len(cov['excluded_files'])} file(s): "
              f"{[e['file'] for e in cov['excluded_files'][:5]]}")
    print(f"lambda central (FROZEN): {art['lambda_central_frozen']}   budget={art['capacity_budget']}"
          f"   latency provenance={art['latency_provenance']}")

    print("\n--- G-Q4: Q vs Q_ext side by side (rank 1 = best) ---")
    for mix, d in art["gates"]["G_Q4"]["per_mix"].items():
        print(f"\n  [{mix}]  n_policies={d['n_policies']}")
        print(f"    {'policy':<14}{'sd':>3}{'Q':>9}{'Q_ext':>9}{'pen':>8}"
              f"{'rQ':>5}{'rX':>5}{'shift':>7}{'cap':>6}{'lat':>6}{'stl':>6}")
        for r in d["table"]:
            flag = "  <== FLAG" if r.get("flag") else ""
            print(f"    {r['policy']:<14}{r['n_seeds']:>3}{r['Q']:>9.4f}{r['Q_ext']:>9.4f}"
                  f"{r['penalty']:>8.4f}{r['rank_Q']:>5.1f}{r['rank_Q_ext']:>5.1f}"
                  f"{r['rank_shift']:>+7.1f}{r['capacity_frac']:>6.2f}"
                  f"{r['latency_frac']:>6.2f}{r['stale_frac']:>6.2f}{flag}")
        for r in d["flagged"]:
            print(f"      EXPLAIN {r['explanation']}")

    q1 = art["gates"]["G_Q1"]
    print(f"\n--- G-Q1 (sanity: nothing may beat the oracle) : "
          f"{'PASS' if q1['passes'] else 'FAIL'} ---")
    for mix, d in q1["per_mix"].items():
        if not d.get("evaluable"):
            print(f"  {mix}: NOT EVALUABLE — {d.get('reason')}")
            continue
        print(f"  {mix}: clean in {d['clean_settings']}/{d['n_settings']} settings"
              f"{'' if d['passes'] else '  VIOLATED'}")
        if d.get("worst_violator"):
            w = d["worst_violator"]
            print(f"      worst violator: {w['policy']} by {w['margin']:+.4f} at {w['setting']}")
    if not q1["passes"]:
        print("  => M6 G-Q1 HARD STOP: Q_ext is still mis-specified. G-Q2/G-Q3 verdicts are")
        print("     computed for the record but MUST NOT be quoted; no frontier under Q_ext.")

    q2 = art["gates"]["G_Q2"]
    print(f"\n--- G-Q2 (primary: router dominates always_grace in >=1 mix) : "
          f"{'PASS' if q2['passes'] else 'FAIL'}"
          f"{'  [SUPPRESSED by G-Q1]' if q2.get('verdict_suppressed_by_G_Q1') else ''} ---")
    for mix, d in q2["per_mix"].items():
        if not d.get("evaluable"):
            print(f"  {mix}: NOT EVALUABLE — {d.get('reason')}")
            continue
        print(f"  {mix}: both Q_ext={d['both_Q_ext']:.4f} (cost {d['both_cost']:.1f}) vs "
              f"always_grace Q_ext={d['always_grace_Q_ext']:.4f} (cost {d['always_grace_cost']:.1f})"
              f"  dQ_ext={d['delta_Q_ext_both_minus_grace']:+.4f}")
        print(f"      router dominates grace: {d['router_dominates_always_grace']}   "
              f"grace dominates router: {d['always_grace_dominates_router']}   "
              f"mutual non-dominance: {d['mutual_non_dominance']}")
    print(f"  always_grace dominates BOTH mixes at central lambda: "
          f"{q2['always_grace_dominates_both']}")

    q3 = art["gates"]["G_Q3"]
    print(f"\n--- G-Q3 (honest-negative gate) : "
          f"{'TRIGGERED' if q3['triggered'] else 'not triggered'} "
          f"({q3['fraction']} settings) ---")
    print(f"  {q3['statement']}")

    print("\n--- sensitivity: exact fractions over the 27-setting grid (NEVER rounded up) ---")
    for k, v in art["sensitivity_grid"]["verdicts"].items():
        print(f"  {v['fraction']:>7}  {'ROBUST' if v['robust_all_settings'] else '      '}  {k}")

    print("\n--- design tensions flagged ---")
    for t in art["design_tensions"]:
        print(f"  * {t}")
    print("=" * 78)


def main() -> None:
    ap = argparse.ArgumentParser(description="M6 Q_ext gate analysis (CPU-only, reads cells)")
    ap.add_argument("--selftest", action="store_true",
                    help="synthetic-cell mechanics check for G-Q1/G-Q4 (no real cells touched)")
    ap.add_argument("--cells_dir", default=os.path.join(C.RESULTS_DIR, "cells"))
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--expect_model", default=None,
                    help="require every loaded cell to carry this model tag (MAJOR-2 guard)")
    ap.add_argument("--expect_provenance", default="real", choices=["real", "synth"],
                    help="require every loaded cell to carry this provenance (default: real)")
    ap.add_argument("--no_write", action="store_true", help="print the summary, write nothing")
    args = ap.parse_args()
    if args.selftest:
        _selftest(); return
    art = analyze(args.cells_dir, args.expect_model, args.expect_provenance)
    print_summary(art)
    if not args.no_write:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(art, open(args.out, "w"), indent=2, allow_nan=True)
        print(f"\nwrote {args.out}")


# ================================================================ selftest (synthetic cells only)
def _mk_cell(mix: str, policy: str, seed: int, *, A_upd: float, A_loc: float, A_cum: float,
             A_rip: float, soh: float, gpu: float, grace: int = 0, edit: int = 0) -> Dict:
    """A synthetic cell in the on-disk schema. NOTHING real is read or written by the selftest."""
    Q = (C.Q_WEIGHTS["A_upd"] * A_upd + C.Q_WEIGHTS["A_loc"] * A_loc
         + C.Q_WEIGHTS["A_cum"] * A_cum + C.Q_WEIGHTS["A_rip"] * A_rip)
    ac = {}
    if grace:
        ac["grace"] = grace
    if edit:
        ac["edit"] = edit
    return {"mix": mix, "policy": policy, "seed": seed, "model": "synthtest",
            "provenance": "synth", "stream_hash": "deadbeef",
            "quality": {"A_upd": A_upd, "A_loc": A_loc, "A_cum": A_cum, "A_rip": A_rip, "Q": Q},
            "cost": {"install_gpu_s": gpu, "serve_gpu_s": 0.0, "total_gpu_s": gpu,
                     "serve_overhead_total": soh, "store_bytes_peak": 0.0,
                     "exposure_surface_mean": 0.0},
            "error_cost_eval": 0.0,
            "discovery": {"n_damaging_gt": 40, "recall_at_decile": 0.5},
            "routing": {"arm_counts": ac, "install_gpu_s_by_arm": {},
                        "serve_gpu_s_by_arm": {}},
            "lambda_cost": 0.0, "n_d_assert": 0}


def _synth_world(grace_wins: bool) -> Dict:
    """Two mixes × the 11 policies × 3 seeds.

    `grace_wins=True` plants the 07-26 pathology: a zero-damage full-codebook GRACE with high
    efficacy against a WEAK oracle that itself pays heavy capacity/latency/staleness, so GRACE
    outscores the ceiling even after the penalties ⇒ G-Q1 must FAIL. `grace_wins=False` plants a
    STRONG oracle (small codebook, low overhead, near-zero fallback) against a high-staleness GRACE,
    so the charged metric puts every policy below the ceiling ⇒ G-Q1 must PASS. Both are mechanics
    fixtures, not realism claims: what is being tested is that the gate fires on the right side.
    """
    cells: Dict[str, Dict[str, List[Dict]]] = {}
    for mix in MIXES:
        cells[mix] = {}
        for policy in POLICIES:
            rows = []
            for s in C.SEEDS:
                if policy == "always_grace":
                    rows.append(_mk_cell(mix, policy, s, A_upd=(0.95 if grace_wins else 0.20),
                                         A_loc=1.0, A_cum=1.0, A_rip=1.0,
                                         soh=325.5, gpu=253.0, grace=500))
                elif policy == "oracle":
                    # weak oracle (heavily charged) vs strong oracle (barely charged)
                    rows.append(_mk_cell(mix, policy, s,
                                         A_upd=(0.60 if grace_wins else 0.95),
                                         A_loc=1.0, A_cum=1.0, A_rip=1.0,
                                         soh=(250.0 if grace_wins else 100.0),
                                         gpu=180.0, grace=(190 if grace_wins else 50)))
                elif policy in ("both", "damage_only"):
                    rows.append(_mk_cell(mix, policy, s, A_upd=0.55,
                                         A_loc=1.0, A_cum=1.0, A_rip=1.0,
                                         soh=309.6, gpu=(120.0 if policy == "both" else 250.0),
                                         grace=486, edit=14))
                else:                                  # weight-editing / reject family
                    rows.append(_mk_cell(mix, policy, s, A_upd=0.40, A_loc=0.90, A_cum=0.90,
                                         A_rip=0.50, soh=0.0, gpu=300.0, edit=500))
            cells[mix][policy] = rows
    return cells


def _selftest() -> None:
    # ---- (0) frac arithmetic agrees with scoring.quality_ext on rows that mirror a cell ----
    # Build OutcomeRows whose aggregates equal a synthetic cell's, then assert this module's
    # aggregate route and the scorer's per-record route produce the SAME Q_ext. This is the
    # anti-drift check: the formula lives in scoring.py and is never re-implemented here.
    rows = []
    for t in range(100):
        rows.append(scoring.OutcomeRow(t=t, arm="grace", fact_type="cf", applied=True,
                                       stale=(t < 40), collateral=0.0,
                                       efficacy_correct=(t >= 40), serve_overhead=0.5))
    ref = scoring.quality_ext(rows, codebook_entries=200.0, latency_max=0.5)
    assert abs(ref["capacity_frac"] - 1.0) < 1e-12
    assert abs(ref["latency_frac"] - 1.0) < 1e-12
    assert abs(ref["stale_frac"] - 0.40) < 1e-12
    # the mirror cell: A_upd = 0.60 (= 1 - stale 0.40), serve_overhead_total = 0.5*500
    mirror = _mk_cell("MIX_A", "always_grace", 0, A_upd=ref["A_upd"], A_loc=ref["A_loc"],
                      A_cum=ref["A_cum"], A_rip=ref["A_rip"], soh=0.5 * STREAM_LEN,
                      gpu=1.0, grace=200)
    got = q_ext_from_cell(mirror, latency_max=0.5, lambdas=CENTRAL)
    assert abs(got["capacity_frac"] - ref["capacity_frac"]) < 1e-12, got
    assert abs(got["latency_frac"] - ref["latency_frac"]) < 1e-12, got
    assert abs(got["stale_frac"] - ref["stale_frac"]) < 1e-12, got
    assert abs(got["Q_ext"] - ref["Q_ext"]) < 1e-12, (got["Q_ext"], ref["Q_ext"])
    # zero-penalty no-op: a policy that stores nothing, adds no overhead, never falls back.
    clean = _mk_cell("MIX_A", "always_edit", 0, A_upd=1.0, A_loc=0.9, A_cum=1.0, A_rip=1.0,
                     soh=0.0, gpu=1.0, edit=500)
    z = q_ext_from_cell(clean, latency_max=0.65, lambdas=CENTRAL)
    assert abs(z["Q_ext"] - z["Q"]) < 1e-12, "zero fracs must make Q_ext a no-op on Q"
    # frozen lambdas are the amendment's, imported not redefined
    assert CENTRAL == {"cap": 0.15, "lat": 0.10, "stale": 0.15}, CENTRAL
    assert BUDGET == 200.0 and len(_grid_settings()) == 27

    # ---- (1) capacity: clip at the frozen budget + the alt stream view separates the clipped ----
    capg, dg = _frac_capacity(_mk_cell("MIX_A", "always_grace", 0, A_upd=.9, A_loc=1., A_cum=1.,
                                       A_rip=1., soh=1., gpu=1., grace=500))
    capb, db = _frac_capacity(_mk_cell("MIX_A", "both", 0, A_upd=.9, A_loc=1., A_cum=1., A_rip=1.,
                                       soh=1., gpu=1., grace=486, edit=14))
    assert capg == capb == 1.0 and dg["clipped"] and db["clipped"], "both clip at budget=200"
    assert dg["capacity_frac_alt_stream"] > db["capacity_frac_alt_stream"], \
        "the alt stream view must separate what the frozen budget cannot"
    cape, _ = _frac_capacity(_mk_cell("MIX_A", "always_edit", 0, A_upd=.9, A_loc=.9, A_cum=.9,
                                      A_rip=.5, soh=0., gpu=1., edit=500))
    assert cape == 0.0, "weight-editing policies store nothing in the codebook"

    # ---- (2) G-Q1 mechanics: fails when a policy beats the oracle, passes when charged enough ----
    bad = _synth_world(grace_wins=True)
    q1_bad = gate_q1(bad)
    assert q1_bad["passes"] is False, "planted above-oracle GRACE must FAIL G-Q1"
    assert q1_bad["n_violations"] > 0
    assert any(v["policy"] == "always_grace" for v in q1_bad["violations"]), \
        "the violator must be identified by name (M6 needs to know which currency is uncharged)"
    assert all(q1_bad["per_mix"][m]["n_settings"] == 27 for m in MIXES)
    good = _synth_world(grace_wins=False)
    q1_good = gate_q1(good)
    assert q1_good["passes"] is True, f"charged GRACE must PASS G-Q1: {q1_good['violations'][:2]}"
    assert all(q1_good["per_mix"][m]["clean_settings"] == 27 for m in MIXES)

    # ---- (3) the G-Q1 hard stop propagates: G-Q2/G-Q3 verdicts get suppressed, tables survive ----
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        for mix, pol in bad.items():
            for policy, cl in pol.items():
                for c in cl:
                    json.dump(c, open(os.path.join(
                        td, f"cell_synthtest_synth_{mix}_{policy}_s{c['seed']}.json"), "w"))
        art = analyze(td, expect_model="synthtest", expect_provenance="synth")
    assert art["gates"]["G_Q1"]["passes"] is False
    assert art["gates"]["G_Q2"].get("verdict_suppressed_by_G_Q1") is True
    assert art["gates"]["G_Q3"].get("verdict_suppressed_by_G_Q1") is True
    assert art["gates"]["G_Q4"]["per_mix"]["MIX_A"]["n_policies"] == len(POLICIES), \
        "G-Q4 tables must still be computed for the record when G-Q1 fails"
    assert art["coverage"]["complete"] is True and art["coverage"]["n_present"] == 66
    assert len(art["per_cell"]) == 66, "the integrity dump must carry EVERY cell"
    for pc in art["per_cell"]:                         # reviewer-recompute inputs must be present
        assert set(pc["fracs"]) == {"capacity_frac", "latency_frac", "stale_frac"}
        assert "serve_overhead_total" in pc["inputs"] and "A_upd" in pc["inputs"]
        assert pc["frac_derivation"]["capacity"]["budget"] == BUDGET
    assert art["latency_provenance"] == LATENCY_PROVENANCE != "measured-real-replay"
    # grid fractions are exact k/27 strings, never a rounded word
    for k, v in art["sensitivity_grid"]["verdicts"].items():
        assert v["n_settings"] == 27 and v["fraction"].endswith("/27"), (k, v)
        assert isinstance(v["robust_all_settings"], bool)
    assert json.dumps(art) and True                    # artifact must be JSON-serialisable

    # ---- (4) missing cells are coverage, not an error (a refill driver may be mid-flight) ----
    with tempfile.TemporaryDirectory() as td:
        for mix, pol in good.items():
            for policy, cl in pol.items():
                for c in cl:
                    if c["seed"] == 2 and policy in ("both", "always_grace"):
                        continue                       # simulate the in-flight s2 refill
                    json.dump(c, open(os.path.join(
                        td, f"cell_synthtest_synth_{mix}_{policy}_s{c['seed']}.json"), "w"))
        art2 = analyze(td, expect_model="synthtest", expect_provenance="synth")
    assert art2["coverage"]["complete"] is False
    assert art2["coverage"]["per_mix"]["MIX_A"]["missing_seeds"]["both"] == [2]
    assert art2["gates"]["G_Q2"]["per_mix"]["MIX_A"]["n_seeds_both"] == 2, "pools over seeds PRESENT"
    assert art2["gates"]["G_Q1"]["passes"] is True, "partial coverage must still be evaluable"

    # ---- (5) G-Q4 rank mechanics: ties share a rank; a >2 shift is flagged AND explained ----
    r = _ranks({"a": 0.9, "b": 0.9, "c": 0.5})
    assert r["a"] == r["b"] == 1.5 and r["c"] == 3.0, r
    shifted = _synth_world(grace_wins=True)
    for mix in MIXES:                                  # plant a big Q_ext-only demotion:
        # mid-table on Q (above the weight-editing family), then crushed by all three penalties:
        # a full codebook, the mix's max overhead, and a 70% fallback rate.
        for c in shifted[mix]["always_rag"]:
            c["quality"].update({"A_upd": 0.30, "A_loc": 1.0, "A_cum": 1.0, "A_rip": 1.0})
            c["quality"]["Q"] = (0.4 * 0.30 + 0.3 + 0.2 + 0.1)
            c["cost"]["serve_overhead_total"] = 400.0
            c["routing"]["arm_counts"] = {"grace": 500}
    q4 = gate_q4(shifted, CENTRAL)
    assert q4["per_mix"]["MIX_A"]["n_policies"] == len(POLICIES)
    tbl = {r["policy"]: r for r in q4["per_mix"]["MIX_A"]["table"]}
    assert tbl["always_rag"]["rank_shift"] < 0, "the crushed policy must move DOWN under Q_ext"
    assert abs(tbl["always_rag"]["rank_shift"]) > 2 and tbl["always_rag"]["flag"] == "RANK_SHIFT_GT_2"
    assert "dominated by the" in tbl["always_rag"]["explanation"], "a flag must carry its cause"
    assert q4["n_flagged_total"] >= 2                   # one per mix
    # every policy appears in the side-by-side table with BOTH metrics (G-Q4's whole point)
    for mix in MIXES:
        for row in q4["per_mix"][mix]["table"]:
            assert "Q" in row and "Q_ext" in row and "rank_Q" in row and "rank_Q_ext" in row

    # ---- (6) G-Q3 arithmetic: majority-of-grid threshold, exact fraction reported ----
    g_bad = sensitivity_grid(bad)
    q3 = gate_q3(bad, g_bad)
    assert q3["n_settings"] == 27 and q3["fraction"].endswith("/27")
    assert q3["triggered"] == (q3["n_settings_always_grace_dominates_both"] > 13.5)
    assert "forbidden" in q3["forbidden_next_step"] or "manufacture" in q3["forbidden_next_step"]

    print("frame_a.q_ext_analysis selftest: PASS")


if __name__ == "__main__":
    main()
