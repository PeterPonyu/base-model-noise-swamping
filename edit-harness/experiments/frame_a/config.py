"""config.py — the FROZEN Frame-A contract (single source of truth for all modules).

Every constant here mirrors the pre-registration
`docs/plans/PREREG-FRAME-A-STREAM-2026-07-16.md` (rev. 4, OPTION A) and the design
`docs/plans/DESIGN-FRAME-A-2026-07-16.md`. NOTHING here is tuned to an outcome; changing a
value is a prereg amendment, not a code edit. Kept import-light (numpy only, lazily) so
`config` loads under a bare interpreter with no torch / no GPU.

BINDING invariants encoded here (do not silently violate elsewhere):
  * damage metric = signed within-probe damage_logit; AUROC is BANNED.
  * geometry claim is L12-ONLY; L14 is norm-growth magnitude, never geometry.
  * two cost objects are DISJOINT (OPTION A):
      - EVAL_COST_RATIOS drive ErrorCost_eval / the P1,P3 Pareto frontier — NO gov term.
      - ROUTER_C_GOV lives ONLY in the router-internal score(); never in ErrorCost_eval.
  * P2 is judged by the STRUCTURAL predicate, never off the ErrorCost frontier.
  * federation capacity: geometry-valid g<=5; magnitude-only band 5<g<=10; blocked g>10.
"""
from __future__ import annotations

# ---------------------------------------------------------------- geometry / predictor
GEOMETRY_LAYER = 12          # the geometry-VALID layer (Llama-1B). L14 is magnitude, not geometry.
GEOMETRY_LAYER_IS_LLAMA_FAMILY_ONLY = True   # wave-1 arch claim scoped to the Llama family.
PREDICTOR_HELDOUT_RHO_L12 = 0.725            # clean held-out per-edit within-cell Spearman.
PREDICTOR_TOPDECILE_RECALL_CEILING_L12 = 0.4407   # measured decile recall (chance ~0.0993).
PREDICTOR_TOPDECILE_CHANCE = 0.0993
DAMAGE_METRIC = "signed_within_probe_damage_logit"   # AUROC banned.
KNOWN_PROBE_PRE_P = 0.05     # base-known probe column mask (pre_p > 0.05), as in d3.

# ---------------------------------------------------------------- federation capacity bound
G_GEOMETRY_VALID = 5         # geometry-trustworthy up to here.
G_MAGNITUDE_ONLY = 10        # 5 < g <= 10 : magnitude-only (degraded); g > 10 : weight-edit blocked.

# ---------------------------------------------------------------- stream construction
STREAM_LEN_WAVE1 = 500       # updates per stream instance (wave 1). Full = 1000.
PROBE_BANK_SIZE = 500        # held-out locality probe bank (record-disjoint from edits).
SEEDS = (0, 1, 2)            # 3 orderings per mix.
FACT_TYPES = ("cf", "zsre", "mquake_mh", "ripple")
CONFLICT_FLAGS = ("none", "conflict", "damaging")
SERVING_HINTS = ("none", "offline", "low_latency", "privacy_sensitive", "footprint")

# Three stream mixes. rho_conflict / rho_damaging are INTENDED per-mix (DOF-1: MIX-A stays 0.10).
MIXES = {
    "MIX_A": {  # steady maintenance (low-churn) — CI-only power (never inflate rho_damaging).
        "desc": "steady maintenance",
        "fact_type_weights": {"cf": 0.45, "zsre": 0.25, "mquake_mh": 0.15, "ripple": 0.15},
        "rho_conflict": 0.10, "rho_damaging": 0.10,
        "serving_hint_weights": {"none": 0.9, "low_latency": 0.1},
        "ci_only_discovery": True,   # DOF-1: report with CIs, no point claim below the power floor.
    },
    "MIX_B": {  # adversarial / high-churn — the damage-awareness regime.
        "desc": "adversarial high-churn",
        "fact_type_weights": {"cf": 0.4, "zsre": 0.3, "mquake_mh": 0.15, "ripple": 0.15},
        "rho_conflict": 0.30, "rho_damaging": 0.30,
        "serving_hint_weights": {"none": 0.85, "low_latency": 0.15},
        "ci_only_discovery": False,
    },
    "MIX_C": {  # ripple-heavy / offline-serving — hosts the P2 must-win regime.
        "desc": "ripple-heavy / privacy-footprint serving",
        "fact_type_weights": {"cf": 0.15, "zsre": 0.15, "mquake_mh": 0.35, "ripple": 0.35},
        "rho_conflict": 0.15, "rho_damaging": 0.15,
        "serving_hint_weights": {"none": 0.25, "offline": 0.2, "low_latency": 0.2,
                                 "privacy_sensitive": 0.2, "footprint": 0.15},
        "ci_only_discovery": False,
    },
}
MIXA_POWER_FLOOR_DAMAGING_GT = 80    # (legacy rev.4) MIX-A power floor; superseded by the rev.5 pin.
# rev.5 (PREREG amendment 2026-07-16): with the 3-cell CF union enlarging the covered pool,
# discovery is CI-only per mix UNLESS post-join damaging_gt >= this pin, in which case a point
# estimate is ALSO reported (both CI and point). MIX-A stays CI-only by its config flag (DOF-1).
DISCOVERY_POINT_FLOOR = 50
# The 3 CF measured cells whose covered records are UNION-ed for coverage (rev.5). Lowest seed
# wins on collision; damaging_gt label = top-decile WITHIN EACH CELL's OWN damage distribution
# (per-cell quantile — cross-cell raw damage is NOT comparable: different probe banks per cell).
CF_UNION_CELL_SEEDS = (0, 1, 2)

# ---------------------------------------------------------------- quality composite (Q)
# Fixed weights for the SUMMARY scalar only; the frontier is the claim, not Q.
Q_WEIGHTS = {"A_upd": 0.40, "A_loc": 0.30, "A_cum": 0.20, "A_rip": 0.10}

# ---------------------------------------------------------------- EVALUATION cost (OPTION A)
# ErrorCost_eval = C_wrong*wrong + C_stale*stale + C_latency*latency + C_compute*gpu_s
# NO governance term here. Sourced anchor: enterprise support KB, C_wrong/C_serve ~30-40x
# (vendor figures, order-of-magnitude only). Primary point 30:9:1:1.
EVAL_COST_RATIOS = {"C_wrong": 30.0, "C_stale": 9.0, "C_latency": 1.0, "C_compute": 1.0}
# Sensitivity grid (conclusion must survive ALL). Includes the C_wrong/C_compute cross-ratio.
SENSITIVITY_GRID = {
    "C_wrong_over_C_stale": (2.0, 5.0, 10.0),
    "C_wrong_over_C_compute": (5.0, 15.0, 30.0, 60.0),
    "C_compute_over_C_latency": (0.5, 1.0, 2.0),
}

# ---------------------------------------------------------------- ROUTER-INTERNAL only
# C_gov lives ONLY in the router's decision score(); NEVER in ErrorCost_eval. Pinned to
# C_stale (no free knob). exposure_surface in {0 (edit), ~1 (rag), ~0.3 (grace)}.
ROUTER_C_GOV = EVAL_COST_RATIOS["C_stale"]     # = 9.0 at the primary point.
EXPOSURE_SURFACE = {"edit": 0.0, "grace": 0.3, "rag": 1.0, "ft": 0.0, "reject": 0.0}

# lambda_cost: the SOLE fitted router knob. Chosen by grid-minimising dev-slice ErrorCost_eval
# over this FIXED log grid, on the DISJOINT dev slice only. Selected value is reported.
LAMBDA_COST_GRID = (0.0, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0)

# thresholds (reported constants, not tuned)
TAU_HIGH_DECILE = 0.90       # "top-decile predicted damage" quantile for the reject/GRACE rule.

# ---------------------------------------------------------------- arms
ARMS = ("edit", "grace", "rag", "ft", "reject")
EDITORS = ("rome", "memit", "alphaedit")       # weight-editor choices for the edit arm.
DEFAULT_EDITOR = "rome"                          # damage-prone primary (the arm to route away from).
RAG_TOP_K = 5                                    # BM25 injected facts (constant-in-N prefill).
RAG_BM25_K1 = 1.5
RAG_BM25_B = 0.75
FT_LORA_R = 16
FT_LORA_ALPHA = 32
FT_LORA_LR = 1e-4
FT_LORA_STEPS = 100
FT_MERGE_INTERVAL_K = 50
FT_LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj")  # gate_proj EXCLUDED (pinned).
FT_LORA_INIT_SEED = 0        # pinned A-init seed for reproducibility.
FT_MIN_FREE_VRAM_GB = 4.0    # VRAM guard: below this free, the FT training round DEFERS (no OOM).

# ---------------------------------------------------------------- frozen predictions / gate
# Directional only. PASS = P1 AND P2 ; KILL = (not P1) AND (not P2) ; else GREY.
GATE = {
    "P1_min_mixes": 2,       # router Pareto-beats every fixed strategy in >= 2/3 mixes.
    "P3_min_mixes": 2,       # router beats FT-merge at cost parity (both ways) in >= 2/3 mixes.
    "P2_structural_terms": (  # ALL must hold in MIX-C. Computed by scorer/analyze, NEVER off frontier.
        "exposure_edit_lt_rag",       # exposure_edit == 0 < exposure_rag ~ 1
        "footprint_delta_positive",   # footprint_rag - footprint_edit > 0
        "overhead_delta_positive",    # per-query serve_overhead(rag) - serve_overhead(edit) > 0 (constant in k)
        "router_selects_edit_majority",  # router picks edit for majority of privacy/footprint MIX-C updates
    ),
}
BOOTSTRAP_N = 1000           # seed-level bootstrap resamples for CIs.
BOOTSTRAP_CI = 0.95

# ---------------------------------------------------------------- on-disk assets
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
HARNESS_ROOT = _os.path.dirname(_os.path.dirname(_HERE))   # edit-harness/
DATA_DIR = _os.path.join(HARNESS_ROOT, "data")
RESULTS_DIR = _os.path.join(HARNESS_ROOT, "results", "frame_a")
DATASETS = {
    "cf": _os.path.join(DATA_DIR, "counterfact.json"),
    "zsre": _os.path.join(DATA_DIR, "zsre_eval.json"),
    "mquake_mh": _os.path.join(DATA_DIR, "mquake_cf3k.json"),
    "ripple": _os.path.join(DATA_DIR, "rippleedits", "popular.json"),
}
# gt_damage cells (scorer/oracle input ONLY — never a router input).
GT_DAMAGE_GLOB = _os.path.join(HARNESS_ROOT, "results", "matrices",
                               "gate_llama1b_rome_cf_L{L}_s{s}.npz")
