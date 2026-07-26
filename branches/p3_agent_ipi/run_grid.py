"""run_grid.py -- B4 extended-lineage-grid entrypoint (thin driver over run_ipi.run).

Runs the lineage-vs-architecture IPI contrast on an EXTENDED panel (grid.py tiers) at a
chosen scenario seed, reusing run_ipi's exact sweep + error-nulling + contrast-gate + result
schema (so audit_unmatched.py and the analyze pipeline consume the output unchanged).

    # metadata dry-run (no daemon, no inference):
    python run_grid.py --tier core --backend mock --n 8 --dry_run

    # real GPU run (Ollama on GPU; residency verified externally by run_p3_gpu.sh):
    python run_grid.py --tier core --backend ollama --n 30 --seed 0 --allow_gpu \
        --run_id ipi_grid_core_s0

The B4 EXTENSION axes are: --tier (model breadth), --seed (scenario-content seed; item->
target stays seed-invariant), and --n (scenario count -- more items => tighter correlations,
lower permutation p, no seed machinery needed).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, H)

import grid  # noqa: E402
import run_ipi  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="B4 extended lineage-grid IPI run.")
    ap.add_argument("--tier", default="core", choices=sorted(grid.TIERS))
    ap.add_argument("--backend", choices=["mock", "ollama"], default="mock")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0, help="scenario-content seed (0 == frozen build)")
    ap.add_argument("--n_perm", type=int, default=1000)
    ap.add_argument("--metric", choices=["pearson", "jaccard"], default="pearson")
    ap.add_argument("--match_mode", choices=["name_and_sentinel", "name_only"],
                    default="name_and_sentinel")
    ap.add_argument("--run_id", default=None)
    ap.add_argument("--allow_gpu", action="store_true")
    ap.add_argument("--allow_singleton_lineage_drop", action="store_true",
                    help="opt-in (default off): a dead SINGLETON in-group lineage (e.g. "
                         "wave-3 openthinker) is dropped instead of suppressing the whole "
                         "seed's contrast, when a multi-member anchor lineage is untouched "
                         "and an attackable architecture pair survives. See PREREG-WAVE3-"
                         "LINEAGE-DRAFT-20260711.md sec 3a before enabling for a real launch.")
    ap.add_argument("--dry_run", action="store_true",
                    help="resolve panel + build scenarios and print the plan; NO model calls")
    args = ap.parse_args(argv)

    names = grid.tier_names(args.tier)
    panel = grid.resolve_panel(names, backend=args.backend,
                               overrides=grid.tier_overrides(args.tier))
    scenarios, source_statuses = grid.seeded_scenarios(n=args.n, seed=args.seed)
    run_id = args.run_id or f"ipi_grid_{args.tier}_n{args.n}_s{args.seed}"

    if args.dry_run:
        plan = {
            "run_id": run_id, "tier": args.tier, "backend": args.backend,
            "n_models": len(panel), "n_scenarios": len(scenarios), "seed": args.seed,
            "models": [{"name": m["name"], "group": m.get("group"), "lineage": m["lineage"],
                        "supports_tools": m.get("supports_tools")} for m in panel],
            "scenario_by_category": {c: sum(1 for x in scenarios if x["attack_category"] == c)
                                     for c in sorted({x["attack_category"] for x in scenarios})},
        }
        print(json.dumps(plan, indent=2, default=str))
        return 0

    res = run_ipi.run(backend=args.backend, panel=panel, scenarios=scenarios,
                      source_statuses=source_statuses, n_perm=args.n_perm, metric=args.metric,
                      match_mode=args.match_mode, run_id=run_id, allow_gpu=args.allow_gpu,
                      allow_singleton_lineage_drop=args.allow_singleton_lineage_drop)
    c = res["contrast"]
    summary = {"run_id": res["run_id"], "out": res["_out_path"], "tier": args.tier,
               "backend": res["backend"], "n_models": len(res["models"]),
               "n_scenarios": res["n_scenarios"], "per_model_asr": res["per_model_asr"],
               "contrast_excluded_models": res["contrast_excluded_models"],
               "contrast_note": res["contrast_note"],
               "contrast": None if c is None else {
                   "mean_lineage_corr": c["mean_lineage_corr"],
                   "mean_architecture_corr": c["mean_architecture_corr"],
                   "observed_diff": c["observed_diff"], "p_value": c["p_value"],
                   "lineage_gt_architecture": c["lineage_gt_architecture"],
                   "dropped_singleton_lineages": c.get("dropped_singleton_lineages")}}
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
