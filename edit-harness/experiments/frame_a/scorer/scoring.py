"""scoring.py — per-stream quality, evaluation cost, discovery, and cost vector.

Consumes a list of per-update OUTCOME rows (produced by `run_stream.replay`) and the stream's
CostLedger. Encodes the frozen definitions:

  * quality: A_upd (efficacy), A_loc (1 − normalised probe-bank collateral; SIGNED within-probe,
    never AUROC), A_cum (end-of-stream retention; punishes FT forgetting), A_rip (multi-hop /
    logical-consequence). Q = fixed-weight composite (summary only; the frontier is the claim).
  * ErrorCost_eval (OPTION A): C_wrong·wrong + C_stale·stale + C_latency·latency + C_compute·gpu_s.
    **NO governance/exposure term** — that liability is router-internal and never on this scalar.
  * discovery: recall@decile + lift on the `damaging_gt` set ONLY (damaging_synth excluded from
    the headline). chance = 0.0993; predictor ceiling = 0.4407 (cited). Quantile-lift, never AUROC.
  * cost vector: install/serve gpu-seconds, per-query serve overhead, footprint/store bytes,
    and the exposure surface (reported SEPARATELY, never folded into ErrorCost_eval).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .. import config as C


@dataclass
class OutcomeRow:
    """Realised effect of one routed update (dryrun-synthetic or real replay)."""
    t: int
    arm: str
    fact_type: str
    conflict_flag: str = "none"
    damaging_kind: Optional[str] = None       # "gt" | "synth" | None
    applied: bool = False                     # did the fact become answerable this step?
    stale: bool = False                       # unapplied (reject / not-yet-flushed at end)?
    forgotten_at_end: bool = False            # lost to a later FT merge overwrite?
    collateral: float = 0.0                   # signed probe-bank damage inflicted (edit only)
    gt_damage: float = 0.0                    # scorer/oracle ground truth (never a router input)
    efficacy_correct: bool = False            # efficacy query recalled target_new at end?
    ripple_correct: Optional[bool] = None     # multi-hop / logical-consequence (None if N/A)
    install_gpu_s: float = 0.0
    serve_gpu_s: float = 0.0
    serve_overhead: float = 0.0               # per-query, above base forward
    exposure_surface: float = 0.0             # reported separately (NOT in ErrorCost_eval)
    store_bytes: float = 0.0
    routed_away_from_edit: bool = False       # for the discovery metric


# ---------------------------------------------------------------- quality
def quality(rows: List[OutcomeRow]) -> Dict[str, float]:
    """A_upd/A_loc/A_cum/A_rip → Q.

    FLAG (metric semantics — the paper must NOT equate these): in the real replay `A_loc` is a
    SEQUENTIAL-incremental STREAM-locality metric — each update's `collateral` is the probe
    correct-token logit DROP measured against a RUNNING baseline (previous state), clipped at 0,
    accumulated over the sequential stream (no restore between updates). This is semantically
    DISTINCT from the B6 gate cells' per-edit `damage_logit`, which is measured per edit against a
    FIXED base model with restore. A_loc answers "how much locality did the stream erode end-to-end",
    not "what did edit i do to a pristine model". Do not conflate the two numbers in the manuscript.
    """
    applied = [r for r in rows if r.applied]
    A_upd = _mean([1.0 if r.efficacy_correct else 0.0 for r in applied])
    # A_loc: 1 − normalised total collateral (signed within-probe magnitude). Bounded [0,1].
    total_col = sum(max(0.0, r.collateral) for r in rows)
    A_loc = 1.0 / (1.0 + total_col / max(1, len(rows)))
    # A_cum: end-of-stream retention over all facts that were ever applied.
    ever = [r for r in rows if r.applied]
    A_cum = _mean([0.0 if r.forgotten_at_end else 1.0 for r in ever])
    rip = [r for r in rows if r.ripple_correct is not None]
    A_rip = _mean([1.0 if r.ripple_correct else 0.0 for r in rip]) if rip else 0.0
    Q = (C.Q_WEIGHTS["A_upd"] * A_upd + C.Q_WEIGHTS["A_loc"] * A_loc
         + C.Q_WEIGHTS["A_cum"] * A_cum + C.Q_WEIGHTS["A_rip"] * A_rip)
    return {"A_upd": A_upd, "A_loc": A_loc, "A_cum": A_cum, "A_rip": A_rip, "Q": Q}


# ---------------------------------------------------------------- evaluation cost (OPTION A)
def error_cost_eval(rows: List[OutcomeRow], ratios: Optional[Dict[str, float]] = None) -> float:
    """ErrorCost_eval — the ONLY cost scalar on the P1/P3 frontier. NO governance term.

    wrong  = collateral-driven wrong answers (∝ realised collateral of applied edits).
    stale  = unapplied updates whose fact a query then missed (reject / never-flushed).
    latency= Σ per-query serve overhead ; gpu_s = Σ install+serve GPU-seconds.
    """
    r = ratios or C.EVAL_COST_RATIOS
    wrong = sum(max(0.0, row.collateral) for row in rows)          # collateral → wrong answers
    stale = sum(1.0 for row in rows if row.stale)
    latency = sum(row.serve_overhead for row in rows)
    gpu_s = sum(row.install_gpu_s + row.serve_gpu_s for row in rows)
    return (r["C_wrong"] * wrong + r["C_stale"] * stale
            + r["C_latency"] * latency + r["C_compute"] * gpu_s)


# ---------------------------------------------------------------- discovery (damaging_gt ONLY)
def discovery(rows: List[OutcomeRow]) -> Dict[str, Any]:
    """recall@decile + lift on the `damaging_gt` set only (headline). AUROC banned.

    Return is a HETEROGENEOUS dict (floats, ints, a `metric` str, and the nested `per_fact_type`
    breakdown), so the value type is `Any` — `Dict[str, float]` was a false narrowing that made a
    static checker flag `set()`/subscripting on the per-fact-type sub-dict at the call sites.

    recall = fraction of ground-truth top-decile (`damaging_gt`) updates the router routed AWAY
    from the weight-edit arm (to GRACE/reject). Reported vs chance (0.0993) as lift, and relative
    to the predictor's measured ceiling (0.4407). `damaging_synth` is EXCLUDED here (separate col).
    """
    gt = [r for r in rows if r.damaging_kind == "gt"]
    synth = [r for r in rows if r.damaging_kind == "synth"]
    recall = _mean([1.0 if r.routed_away_from_edit else 0.0 for r in gt]) if gt else float("nan")
    chance = C.PREDICTOR_TOPDECILE_CHANCE
    ceiling = C.PREDICTOR_TOPDECILE_RECALL_CEILING_L12
    # per-fact-type discovery breakdown (MINOR-A follow-up) — reported ALONGSIDE the pooled
    # headline; the pooled `recall_at_decile` remains the headline. On the damaging_gt set only.
    per_fact_type = {}
    for ft in C.FACT_TYPES:
        ft_gt = [r for r in gt if r.fact_type == ft]
        per_fact_type[ft] = {
            "n_damaging_gt": len(ft_gt),
            "recall_at_decile": (
                _mean([1.0 if r.routed_away_from_edit else 0.0 for r in ft_gt]) if ft_gt else float("nan")),
        }
    out = {
        "n_damaging_gt": len(gt),
        "recall_at_decile": recall,
        "chance": chance,
        "lift": (recall / chance) if (gt and chance > 0) else float("nan"),
        "predictor_ceiling": ceiling,
        "recall_over_ceiling": (recall / ceiling) if (gt and ceiling > 0) else float("nan"),
        "per_fact_type": per_fact_type,
        # robustness-only column (excluded from the headline):
        "synth_recall_at_decile": (
            _mean([1.0 if r.routed_away_from_edit else 0.0 for r in synth]) if synth else float("nan")),
        "n_damaging_synth": len(synth),
        "metric": "recall_at_decile+lift (quantile-lift; AUROC banned)",
    }
    return out


# ---------------------------------------------------------------- cost vector
def cost_vector(rows: List[OutcomeRow]) -> Dict[str, float]:
    install = sum(r.install_gpu_s for r in rows)
    serve = sum(r.serve_gpu_s for r in rows)
    return {
        "install_gpu_s": install,
        "serve_gpu_s": serve,
        "total_gpu_s": install + serve,                 # the P1/P3 cost axis.
        "serve_overhead_total": sum(r.serve_overhead for r in rows),
        "store_bytes_peak": max([r.store_bytes for r in rows], default=0.0),
        # exposure is reported SEPARATELY — never folded into ErrorCost_eval or the frontier.
        "exposure_surface_mean": _mean([r.exposure_surface for r in rows]) if rows else 0.0,
    }


# ---------------------------------------------------------------- P2 structural quantities (MIX-C)
def p2_structural(arm_exposure: Dict[str, float], arm_footprint: Dict[str, float],
                  arm_overhead: Dict[str, float]) -> Dict:
    """The arm-level P2 structural deltas (never from ErrorCost_eval).

    (1) exposure_edit=0 < exposure_rag≈1 ; (2) footprint_delta>0 ; (3) per-query overhead_delta>0.
    The 4th P2 term (router selects `edit` for the majority of privacy/footprint MIX-C updates)
    needs the per-update serving_hint + chosen arm and is computed by `analyze_frame_a` from the
    routing log, not here.
    """
    return {
        "exposure_edit": arm_exposure.get("edit", 0.0),
        "exposure_rag": arm_exposure.get("rag", 1.0),
        "footprint_delta": arm_footprint.get("rag", 0.0) - arm_footprint.get("edit", 0.0),
        "overhead_delta": arm_overhead.get("rag", 0.0) - arm_overhead.get("edit", 0.0),
    }


# ---------------------------------------------------------------- helpers
def _mean(xs: List[float]) -> float:
    xs = [x for x in xs if x == x]     # drop NaN
    return float(sum(xs) / len(xs)) if xs else 0.0


# ---------------------------------------------------------------- Q_ext (M6 amendment)
# PREREG-FRAME-A-AMENDMENT-M6-QMETRIC-2026-07-30 — **DRAFT, NOT yet user-ratified.**
# CODE ARM ONLY: these functions exist so the gate can be reviewed, but NO Q_ext number
# may be computed on real cells before ratification (the 07-26 always_grace>oracle artifact
# showed Q hands GRACE a free w_loc=0.30 on zero-by-construction damage; Q_ext charges
# capacity/latency/staleness). Synthetic selftest rows are fine — they compute nothing real.
Q_EXT_LAMBDA = {"cap": 0.15, "lat": 0.10, "stale": 0.15}      # FROZEN by the M6 amendment
Q_EXT_CAPACITY_BUDGET = 200.0                                  # one entry per stream edit
Q_EXT_GRID = {"cap": (0.05, 0.15, 0.30),
              "lat": (0.05, 0.10, 0.20),
              "stale": (0.05, 0.15, 0.30)}                     # 27-setting sensitivity grid


def quality_ext(rows: List[OutcomeRow], *, codebook_entries: float = 0.0,
                latency_max: float = 0.0, latency_provenance: str = "measured",
                lambdas: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """Q_ext = Q − λ_cap·capacity_frac − λ_lat·latency_frac − λ_stale·stale_frac.

    * capacity_frac = codebook_entries / 200, clipped [0,1] (0 for weight-editing policies,
      which store nothing).
    * latency_frac = mean per-query serve_overhead / the MIX's max observed overhead across
      policies (caller supplies latency_max for the whole mix so policies are comparable),
      clipped [0,1]. M2 provenance rule: synthetic latencies must arrive with an explicit
      latency_provenance tag, never silently "measured".
    * stale_frac = fraction of rows answered by fallback rather than the policy's own
      mechanism — read from the existing OutcomeRow.stale field (no new measurement).
    """
    q = quality(rows)
    lam = dict(Q_EXT_LAMBDA) if lambdas is None else dict(lambdas)
    capacity_frac = min(1.0, max(0.0, codebook_entries / Q_EXT_CAPACITY_BUDGET))
    per_query_overhead = _mean([r.serve_overhead for r in rows])
    latency_frac = (min(1.0, max(0.0, per_query_overhead / latency_max))
                    if latency_max > 0 else 0.0)
    stale_frac = _mean([1.0 if r.stale else 0.0 for r in rows])
    q_ext = (q["Q"] - lam["cap"] * capacity_frac
             - lam["lat"] * latency_frac - lam["stale"] * stale_frac)
    return {**q,
            "capacity_frac": capacity_frac, "latency_frac": latency_frac,
            "stale_frac": stale_frac, "latency_provenance": latency_provenance,
            "lambda_cap": lam["cap"], "lambda_lat": lam["lat"], "lambda_stale": lam["stale"],
            "Q_ext": q_ext}


def q_ext_sensitivity_grid(rows: List[OutcomeRow], *, codebook_entries: float = 0.0,
                           latency_max: float = 0.0,
                           latency_provenance: str = "measured") -> List[Dict[str, float]]:
    """The full 27-setting grid. A verdict holds only where it survives ALL settings —
    the M6 amendment: fractions reported exactly, never rounded up to 'robust'."""
    out = []
    for lc in Q_EXT_GRID["cap"]:
        for ll in Q_EXT_GRID["lat"]:
            for ls in Q_EXT_GRID["stale"]:
                out.append(quality_ext(rows, codebook_entries=codebook_entries,
                                       latency_max=latency_max,
                                       latency_provenance=latency_provenance,
                                       lambdas={"cap": lc, "lat": ll, "stale": ls}))
    return out


# ---------------------------------------------------------------- selftest
def _selftest() -> None:
    # Build two tiny synthetic streams: a good policy (routes damage away) and a bad one (edits all).
    def mk(arm_for_damaging, n=50):
        rows = []
        for t in range(n):
            is_gt = (t % 10 == 0)                       # ~10% damaging_gt
            gt_dmg = 3.0 if is_gt else 0.2
            arm = arm_for_damaging if is_gt else "edit"
            applied = arm != "reject"
            collateral = (gt_dmg if arm == "edit" else 0.0)
            rows.append(OutcomeRow(
                t=t, arm=arm, fact_type="cf",
                damaging_kind=("gt" if is_gt else None),
                applied=applied, stale=(arm == "reject"),
                collateral=collateral, gt_damage=gt_dmg,
                efficacy_correct=applied,
                install_gpu_s=(8.0 if arm == "edit" else 2.0 if arm == "grace" else 0.05),
                serve_gpu_s=0.0, serve_overhead=(0.6 if arm == "rag" else 0.0),
                exposure_surface=C.EXPOSURE_SURFACE.get(arm, 0.0),
                routed_away_from_edit=(arm != "edit")))
        return rows
    good = mk("grace")       # routes damaging_gt to GRACE (away from edit)
    bad = mk("edit")         # edits everything (never routes away)
    qg, qb = quality(good), quality(bad)
    assert qg["A_loc"] > qb["A_loc"], "routing damage away must improve locality"
    # ErrorCost_eval: bad (more collateral) must cost MORE, and NO gov term is present.
    eg, eb = error_cost_eval(good), error_cost_eval(bad)
    assert eb > eg, "policy that edits all damaging facts must have higher ErrorCost_eval"
    # exposure is NOT in ErrorCost_eval: adding exposure to every row must not change the eval cost.
    for r in good:
        r.exposure_surface = 1.0
    assert abs(error_cost_eval(good) - eg) < 1e-9, "ErrorCost_eval must be gov-free (exposure excluded)"
    # discovery on damaging_gt only:
    dg = discovery(good)
    assert dg["recall_at_decile"] == 1.0 and dg["lift"] > 1.0
    # per-fact-type breakdown present; cf (the only type here) matches the pooled headline.
    assert set(dg["per_fact_type"]) == set(C.FACT_TYPES)
    assert dg["per_fact_type"]["cf"]["recall_at_decile"] == dg["recall_at_decile"]
    assert dg["per_fact_type"]["cf"]["n_damaging_gt"] == dg["n_damaging_gt"]
    assert dg["per_fact_type"]["zsre"]["recall_at_decile"] != dg["per_fact_type"]["zsre"]["recall_at_decile"]  # NaN (no zsre gt)
    db = discovery(bad)
    assert db["recall_at_decile"] == 0.0
    # error-cost arithmetic exact on a hand-checked row:
    row = OutcomeRow(t=0, arm="edit", fact_type="cf", applied=True, collateral=2.0,
                     stale=False, install_gpu_s=1.0, serve_overhead=0.5, serve_gpu_s=0.0)
    ec = error_cost_eval([row], {"C_wrong": 30.0, "C_stale": 9.0, "C_latency": 1.0, "C_compute": 1.0})
    assert abs(ec - (30 * 2.0 + 9 * 0 + 1 * 0.5 + 1 * 1.0)) < 1e-9, f"arithmetic {ec}"
    # ---- Q_ext (M6 code arm; synthetic rows only — amendment NOT yet ratified) ----
    # no-op check: zero fracs ⇒ Q_ext == Q exactly.
    qx0 = quality_ext(good)
    assert abs(qx0["Q_ext"] - qx0["Q"]) < 1e-12, "zero penalties must be a no-op"
    # planted GRACE-like policy: zero collateral, full codebook, 60% fallback, overhead at max.
    grace_like = mk("grace")
    for r in grace_like:
        r.stale = (r.t % 5 < 3)           # 60% fallback
        r.serve_overhead = 0.6
    qg_ext = quality_ext(grace_like, codebook_entries=200.0, latency_max=0.6)
    expected = qg_ext["Q"] - 0.15 * 1.0 - 0.10 * 1.0 - 0.15 * 0.6
    assert abs(qg_ext["Q_ext"] - expected) < 1e-9, f"penalty arithmetic {qg_ext['Q_ext']} vs {expected}"
    # G-Q1-style sanity on synthetic rows: a full-capacity, high-staleness, max-latency
    # zero-damage policy must NOT beat a perfect oracle policy under Q_ext.
    oracle_like = mk("edit")
    for r in oracle_like:
        r.collateral = 0.0; r.applied = True; r.efficacy_correct = True
        r.forgotten_at_end = False; r.stale = False; r.serve_overhead = 0.0
    qo_ext = quality_ext(oracle_like, codebook_entries=0.0, latency_max=0.6)
    assert qg_ext["Q_ext"] < qo_ext["Q_ext"], "charged GRACE-like must not beat oracle under Q_ext"
    # grid: exactly 27 settings, and the verdict here survives all of them.
    grid_g = q_ext_sensitivity_grid(grace_like, codebook_entries=200.0, latency_max=0.6)
    grid_o = q_ext_sensitivity_grid(oracle_like, codebook_entries=0.0, latency_max=0.6)
    assert len(grid_g) == 27 and len(grid_o) == 27
    surv = sum(1 for a, b in zip(grid_g, grid_o) if a["Q_ext"] < b["Q_ext"])
    assert surv == 27, f"synthetic verdict must survive 27/27, got {surv}"
    print("scorer.scoring selftest: PASS")


if __name__ == "__main__":
    _selftest()
