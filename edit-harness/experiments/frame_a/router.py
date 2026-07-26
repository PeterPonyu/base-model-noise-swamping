"""router.py — the interpretable expected-error-cost routing policy (DESIGN §c, PREREG §2).

For each update `u` and arm `a`:

    score(u, a) = expected_error_cost(u, a) + λ_cost · serving_cost(u, a)
    expected_error_cost(u, a) = C_wrong · P_collateral(u, a)     # damage-driven
                              + C_stale · P_miss(u, a)           # arm fails to apply → stale
                              + C_gov  · P_expose(u, a)          # ROUTER-INTERNAL governance surface

and the router picks `argmin_a score(u, a)`. Every term is an explicit, inspectable function of
router-visible inputs, so the decision is auditable (the per-arm breakdown is logged for the
ESWA decision-support figure).

BINDING (OPTION A, rev.4):
  * `C_gov · P_expose` lives **only** here — it steers routing toward `edit` on
    privacy/footprint updates — and is a **pinned** constant `C_gov = C_stale` (no free knob).
    It NEVER enters `ErrorCost_eval` (that scalar is the scorer's job) so the router cannot
    "win" the P1/P3 Pareto frontier on a soft self-assigned term.
  * NO `C_forget` term (DOF-3): forgetting is priced by the scorer through `A_cum → Q`.
  * `λ_cost` is the SOLE fitted knob, chosen by grid-minimising dev-slice `ErrorCost_eval` over
    the fixed log grid on the DISJOINT calibration slice only (`calibrate_lambda`).
  * The router reads ONLY `router_view(u)` (geometry = raw signed key-cos at L12); `gt_damage`
    is never visible to it. Capacity uses the two-regime federation bound (g≤5 geometry-valid;
    5<g≤10 magnitude-only/degraded; g>10 weight-edit blocked).

DRYRUN-safe: no torch/GPU; serving-cost table comes from `cost_harness.SyntheticClock`, exposure
from `config.EXPOSURE_SURFACE`. Real runs swap the cost table for measured harness readings.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from . import config as C
from .cost_harness import SyntheticClock
from .damage_predictor import DamagePredictor
from .stream_builder import router_view


def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


@dataclass
class RouterState:
    """Mutable capacity state the router maintains as it replays a stream.

    `subject_edits` = per subject-neighborhood count of in-flight weight edits (the federation
    `g`). Weight edits to a re-used subject (conflict injection) accumulate g there; distinct
    subjects stay at g=0/1. Capacity is read BEFORE the arm is chosen, updated after.
    """
    subject_edits: Dict[str, int] = field(default_factory=dict)
    store_n: int = 0                     # RAG/GRACE store size (for serving-cost table).
    n_routed: Dict[str, int] = field(default_factory=dict)

    def g_for(self, subject_key: str) -> int:
        return self.subject_edits.get(subject_key, 0)

    def commit(self, arm: str, subject_key: str) -> None:
        self.n_routed[arm] = self.n_routed.get(arm, 0) + 1
        if arm == "edit":
            self.subject_edits[subject_key] = self.subject_edits.get(subject_key, 0) + 1
        if arm in ("rag", "grace"):
            self.store_n += 1


@dataclass
class Router:
    """The interpretable per-update router.

    mode ∈ {"both", "cost_only", "damage_only"} selects the ablation:
      * both        — full score (the headline router).
      * cost_only   — damage term OFF (λ_cost large): ignores collateral → over-edits cheaply.
      * damage_only — λ_cost = 0: ignores cost → over-uses GRACE/reject.
    """
    predictor: DamagePredictor = field(default_factory=DamagePredictor)
    mode: str = "both"
    lambda_cost: float = 1e-2
    editor: str = C.DEFAULT_EDITOR
    _clock: SyntheticClock = field(default_factory=SyntheticClock)

    # cost/decision constants (pinned).
    C_wrong: float = C.EVAL_COST_RATIOS["C_wrong"]
    C_stale: float = C.EVAL_COST_RATIOS["C_stale"]
    C_gov: float = C.ROUTER_C_GOV

    # ------------------------------------------------------------------ P_* terms
    def _serve_overhead(self, arm: str, store_n: int) -> float:
        """Per-query serving cost above a base forward (reference-forward multiples)."""
        return self._clock.serve(arm, n_queries=1, store_n=store_n, k=C.RAG_TOP_K).gpu_s - 1.0

    def _p_collateral(self, arm: str, dhat: float, g: int) -> Optional[float]:
        """Damage probability for weight edits; 0 for non-parametric arms. None = capacity-blocked.

        Monotone in `d̂` (raw signed key-cos at L12) × affected-probe mass (a pre-edit,
        key-derived quantity). Federation capacity:
          g ≤ 5   : geometry-valid — full d̂ trust.
          5 < g≤10: magnitude-only (geometry inert) — floor the estimate (cannot trust a LOW
                     prediction; the router must stay cautious). Binding: no geometry claim here.
          g > 10  : weight-edit BLOCKED (return None → arm removed from the candidate set).
        """
        if arm not in ("edit",):
            return 0.0
        if g > C.G_MAGNITUDE_ONLY:
            return None                              # blocked
        # key-derived collateral estimate, monotone in d̂, mapped to [0,1].
        pc = _clip01((dhat + 0.2) / 0.8)             # key_cos range ~[-0.2,0.6] -> [0,1]
        mass = _clip01(0.5 + dhat)                   # affected-probe mass proxy (pre-edit, key-derived)
        est = pc * mass
        if g > C.G_GEOMETRY_VALID:                   # magnitude-only band: floor (geometry inert)
            est = max(est, 0.5)
        return est

    def _p_miss(self, arm: str, fact_type: str) -> float:
        """Probability the arm fails to apply the fact (→ stale). NOT 1 for RAG offline (M3)."""
        if arm == "reject":
            return 1.0
        if arm == "rag":
            base = 0.10                              # paraphrase/lexical retrieval gap.
            return base + (0.10 if fact_type in ("mquake_mh", "ripple") else 0.0)
        if arm == "ft":
            return 0.30                              # DEFERRED until the next merge flush → the
            #                                         fact is stale for ~part of the K-interval.
        return 0.0                                    # edit / grace apply immediately.

    # serving_hint sensitivity for the exposure term: the raw-store liability is only *costly*
    # where the deployment is privacy/footprint/offline-sensitive (DESIGN §c: P_expose is keyed
    # on those tags); elsewhere a low baseline (the store exists but is not the priority).
    _EXPOSE_SENSITIVITY = {"privacy_sensitive": 1.0, "footprint": 1.0, "offline": 1.0,
                           "low_latency": 0.15, "none": 0.15}

    def _p_expose(self, arm: str, serving_hint: str) -> float:
        """Deployment-surface / plaintext-exposure liability (router-internal only), keyed on the
        update's serving_hint so RAG is penalised where privacy/footprint/offline matters."""
        return C.EXPOSURE_SURFACE[arm] * self._EXPOSE_SENSITIVITY.get(serving_hint, 0.15)

    # ------------------------------------------------------------------ scoring
    def _score_arm(self, arm: str, u_view: Dict, state: RouterState) -> Optional[Dict]:
        dhat = self.predictor.predict(u_view)
        g = state.g_for(u_view.get("subject_key", ""))
        pc = self._p_collateral(arm, dhat, g)
        if pc is None:
            return None                              # capacity-blocked candidate.
        pm = self._p_miss(arm, u_view["fact_type"])
        pe = self._p_expose(arm, u_view.get("serving_hint", "none"))
        use_damage = self.mode != "cost_only"
        use_cost = self.mode != "damage_only"
        use_gov = self.mode != "cost_only"           # exposure steering off in the cost-only ablation.
        lam = (1e6 if self.mode == "cost_only" else self.lambda_cost)
        err = (self.C_wrong * pc if use_damage else 0.0) \
            + self.C_stale * pm \
            + (self.C_gov * pe if use_gov else 0.0)
        # serving cost the router trades off: per-query overhead × estimated downstream query
        # volume (a router-visible input). edit serves free (0); grace/rag pay per query.
        qvol = max(1, int(u_view.get("est_qvol", 1)))
        serving = self._serve_overhead(arm, state.store_n) * qvol
        cost = lam * serving if use_cost else 0.0
        return {"arm": arm, "score": err + cost, "P_collateral": pc, "P_miss": pm,
                "P_expose": pe, "serving": serving, "dhat": dhat, "g": g,
                "err": err, "cost_term": cost}

    def route(self, update: Dict, state: RouterState) -> Dict:
        """Return the routing decision + full per-arm score breakdown (audit trail)."""
        u_view = router_view(update)
        breakdown = []
        for arm in C.ARMS:
            sc = self._score_arm(arm, u_view, state)
            if sc is not None:
                breakdown.append(sc)
        breakdown.sort(key=lambda d: (d["score"], C.ARMS.index(d["arm"])))
        chosen = breakdown[0]["arm"]
        editor = self._pick_editor(chosen, u_view, state)
        state.commit(chosen, u_view.get("subject_key", ""))
        return {"arm": chosen, "editor": editor, "breakdown": breakdown,
                "decision_rule": self._decision_rule(chosen, u_view, breakdown)}

    def _pick_editor(self, arm: str, u_view: Dict, state: RouterState) -> Optional[str]:
        """Within the edit arm, MEMIT for batched arrivals / AlphaEdit if projector present."""
        if arm != "edit":
            return None
        return self.editor                            # ROME default; run_stream may batch to MEMIT.

    @staticmethod
    def _decision_rule(arm: str, u_view: Dict, breakdown: List[Dict]) -> str:
        """Human-readable decision-list label (DESIGN §c list) for the audit figure."""
        hint = u_view.get("serving_hint", "none")
        if arm == "edit" and hint in ("privacy_sensitive", "footprint", "offline"):
            return "R1: privacy/footprint & d̂ acceptable → edit (P_expose(rag)>0=P_expose(edit))"
        if arm in ("grace", "reject"):
            return "R2/R3: top-decile d̂ or capacity-blocked or low-value conflict → GRACE/reject"
        if arm == "edit":
            return "R4: moderate d̂, capacity ok → edit/MEMIT"
        return "R5: cheap-serving & non-damaging → RAG (or batch to FT)"

    # ------------------------------------------------------------------ λ_cost calibration
    def calibrate_lambda(self, dev_updates: List[Dict], eval_cost_fn) -> float:
        """Grid-minimise dev-slice ErrorCost_eval over the FIXED log grid (DOF-2).

        `eval_cost_fn(router, dev_updates) -> float` replays the dev slice through this router at
        a candidate λ and returns the EVALUATION error cost (scorer.ErrorCost_eval; NO gov term).
        The λ with the lowest dev-slice cost is frozen and returned. Calibration touches ONLY the
        disjoint dev slice, never a scored stream.
        """
        best_lam, best_cost = self.lambda_cost, float("inf")
        saved = self.lambda_cost
        for lam in C.LAMBDA_COST_GRID:
            self.lambda_cost = lam
            cost = eval_cost_fn(self, dev_updates)
            if cost < best_cost:
                best_cost, best_lam = cost, lam
        self.lambda_cost = best_lam
        return best_lam


# ---------------------------------------------------------------- baselines / oracle
@dataclass
class FixedRouter:
    """always-<arm> fixed strategy."""
    arm: str

    def route(self, update: Dict, state: RouterState) -> Dict:
        subj = update.get("subject_key", update.get("edit", {}).get("subject", ""))
        state.commit(self.arm, subj)
        return {"arm": self.arm, "editor": (C.DEFAULT_EDITOR if self.arm == "edit" else None),
                "breakdown": [], "decision_rule": f"fixed:{self.arm}"}


@dataclass
class RandomRouter:
    seed: int = 0
    _rng: object = None

    def __post_init__(self):
        import numpy as np
        self._rng = np.random.default_rng(self.seed)

    def route(self, update: Dict, state: RouterState) -> Dict:
        arm = str(self._rng.choice(C.ARMS))
        subj = update.get("subject_key", update.get("edit", {}).get("subject", ""))
        state.commit(arm, subj)
        return {"arm": arm, "editor": (C.DEFAULT_EDITOR if arm == "edit" else None),
                "breakdown": [], "decision_rule": "random"}


@dataclass
class OracleRouter:
    """Upper bound: the TRUE greedy per-update argmin on realised ErrorCost_eval, using perfect
    foresight of `gt_damage` (and the store-dependent cost table). The oracle is the ONLY policy
    allowed to read `gt_damage` — it defines η = router/oracle. Mirrors `run_stream._replay`'s
    per-arm cost accounting so nothing can beat it on the per-update collateral/stale/compute terms.
    """
    ratios: Optional[Dict[str, float]] = None
    _clock: SyntheticClock = field(default_factory=SyntheticClock)

    def route(self, update: Dict, state: RouterState) -> Dict:
        r = self.ratios or C.EVAL_COST_RATIOS
        gt = float(update["gt_damage"])               # oracle-only ground truth.
        subj = update.get("subject_key", update.get("edit", {}).get("subject", ""))
        g = state.g_for(subj)
        qvol = max(1, int(update.get("est_qvol", len(update.get("downstream_query_set", {}).get("efficacy", [1])))))
        n = state.store_n
        cand = {}
        for arm in C.ARMS:
            if arm == "edit" and g > C.G_MAGNITUDE_ONLY:
                continue                              # capacity-blocked.
            collateral = gt if arm == "edit" else 0.0
            stale = 1.0 if arm == "reject" else 0.0
            install = self._clock.install(arm, 1, n).gpu_s
            over = self._clock.serve(arm, 1, n, C.RAG_TOP_K).gpu_s - 1.0
            serve_gpu = qvol * (1.0 + over)
            realised = (r["C_wrong"] * collateral + r["C_stale"] * stale
                        + r["C_latency"] * (qvol * over) + r["C_compute"] * (install + serve_gpu))
            cand[arm] = realised
        arm = min(cand, key=lambda a: (cand[a], C.ARMS.index(a)))
        state.commit(arm, subj)
        return {"arm": arm, "editor": (C.DEFAULT_EDITOR if arm == "edit" else None),
                "breakdown": [{"arm": arm, "gt_damage": gt, "realised": cand}],
                "decision_rule": "oracle(argmin realised ErrorCost)"}


# ---------------------------------------------------------------- selftest
def _selftest() -> None:
    from .stream_builder import StreamBuilder
    b = StreamBuilder(synthetic=True)
    updates, _ = b.build_stream("MIX_B", seed=0)

    # 1) high predicted-damage update routes AWAY from the weight-edit arm.
    r = Router(mode="both", lambda_cost=1e-2)
    st = RouterState()
    hi = dict(updates[0]); hi["key_cos"] = 0.6; hi["serving_hint"] = "none"
    dec = r.route(hi, st)
    assert dec["arm"] != "edit", f"high-d̂ update should avoid edit, got {dec['arm']}"

    # 2) capacity: after >10 edits to one subject, the edit arm is BLOCKED (removed from candidates).
    st2 = RouterState()
    r2 = Router(mode="cost_only")                     # cost_only over-edits, so it will try edit
    lo = dict(updates[0]); lo["key_cos"] = -0.1; lo["subject_key"] = "S"; lo["serving_hint"] = "none"
    picks = [r2.route(dict(lo), st2)["arm"] for _ in range(13)]
    assert st2.g_for("S") <= C.G_MAGNITUDE_ONLY + 1
    # once g>10 no further edit decisions increment g (edit removed):
    assert picks.count("edit") <= C.G_MAGNITUDE_ONLY + 1, "edit must be capacity-blocked past g=10"

    # 3) privacy_sensitive update: exposure term pushes edit over rag.
    r3 = Router(mode="both", lambda_cost=1e-2)
    st3 = RouterState()
    priv = dict(updates[0]); priv["key_cos"] = 0.0; priv["serving_hint"] = "privacy_sensitive"
    b3 = r3.route(priv, st3)["breakdown"]
    s_edit = next(d["score"] for d in b3 if d["arm"] == "edit")
    s_rag = next(d["score"] for d in b3 if d["arm"] == "rag")
    assert s_edit < s_rag, "privacy update: edit must score below rag (P_expose steering)"

    # 4) cost_only ignores collateral (edits a high-damage benign-serving fact); damage_only avoids it.
    stc, std = RouterState(), RouterState()
    hi2 = dict(updates[1]); hi2["key_cos"] = 0.55; hi2["serving_hint"] = "none"; hi2["subject_key"] = "Z"
    assert Router(mode="cost_only").route(dict(hi2), stc)["arm"] == "edit"
    assert Router(mode="damage_only").route(dict(hi2), std)["arm"] != "edit"

    # 5) λ calibration picks a grid value that minimises a stubbed dev cost.
    def stub_cost(router, devs):
        # synthetic: prefer a mid λ (parabola minimum at 1e-2).
        import math
        return (math.log10(router.lambda_cost + 1e-6) - math.log10(1e-2)) ** 2
    lam = Router().calibrate_lambda(updates[:20], stub_cost)
    assert lam in C.LAMBDA_COST_GRID and abs(lam - 1e-2) < 3e-2, f"calibrated λ={lam}"

    # 6) oracle reads gt_damage and avoids edit on a truly-damaging fact.
    ost = RouterState()
    dmg = dict(updates[0]); dmg["gt_damage"] = 3.0
    assert OracleRouter().route(dmg, ost)["arm"] == "grace"
    print("router selftest: PASS")


if __name__ == "__main__":
    _selftest()
