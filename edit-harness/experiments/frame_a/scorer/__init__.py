"""scorer/ — quality + evaluation-cost scoring and the frozen P1–P4 / kill-gate analysis.

`scoring`  : per-stream A_upd/A_loc/A_cum/A_rip → Q, ErrorCost_eval (OPTION A, NO gov term),
             discovery (recall@decile + lift on damaging_gt ONLY), cost vector.
`analyze_frame_a` : seed aggregation + bootstrap CIs, computable Pareto predicate, the P1–P4
             structural predicates, and the mechanical PASS/GREY/KILL verdict.
"""
from .scoring import (  # noqa: F401
    quality, error_cost_eval, discovery, cost_vector, OutcomeRow,
)
