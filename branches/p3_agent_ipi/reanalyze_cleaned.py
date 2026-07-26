"""reanalyze_cleaned.py — CPU re-analysis of the real (ollama) P3 run with
API-dead models excluded.

The 2026-06-30 real run's three deepseek-r1 rows are all-zero because every
call failed (HTTP 400: the r1 distills have supports_tools=false and the
Ollama tool API rejects them) — NOT because they resisted injection. This
script (1) detects dead models from per_model_records, (2) recomputes ASR and
the lineage-vs-architecture contrast on the surviving models only, and (3)
records explicitly whether the design remains testable after exclusion (it
does not, if the whole r1 lineage group is dead — that is the finding).

Pure CPU on the saved result JSON. NO network, NO Ollama.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import contrast  # noqa: E402

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "results", "ipi_20260630_ollama_n30.json")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "results", "ipi_20260630_ollama_n30_cleaned.json")


def main():
    d = json.load(open(SRC))
    models = d["models"]
    matrix = d["success_matrix"]
    recs = d["per_model_records"]

    by_name = {r.get("model"): r for r in recs} if isinstance(recs, list) else recs
    dead = []
    for i, m in enumerate(models):
        r = by_name.get(m["name"], {})
        items = r.get("items", []) if isinstance(r, dict) else []
        errs = sum(1 for it in items if isinstance(it, dict) and it.get("error"))
        if items and errs == len(items):
            dead.append(i)

    keep = [i for i in range(len(models)) if i not in dead]
    sub_models = [models[i] for i in keep]
    sub_matrix = [matrix[i] for i in keep]

    import math
    cleaned = contrast(sub_matrix, sub_models, metric="pearson", n_perm=1000, seed=0)
    od = cleaned.get("observed_diff")
    testable = bool(cleaned.get("lineage_pairs")) and bool(cleaned.get("architecture_pairs")) \
        and od is not None and isinstance(od, (int, float)) and not math.isnan(od)
    out = {
        "source": os.path.basename(SRC),
        "dead_models_excluded": [models[i]["name"] for i in dead],
        "dead_reason": "every per-model record carries an error (HTTP 400: "
                       "supports_tools=false vs Ollama tool API) — all-zero rows are "
                       "API artifacts, not injection resistance",
        "n_models_kept": len(keep),
        "per_model_asr_kept": {models[i]["name"]: d["per_model_asr"][i]
                               if isinstance(d["per_model_asr"], list)
                               else d["per_model_asr"][models[i]["name"]]
                               for i in keep},
        "cleaned_contrast": cleaned,
        "design_testable_after_exclusion": testable,
        "verdict": ("DEGENERATE: excluding the API-dead models kills the entire r1-distill "
                    "lineage group (all 3 r1 rows dead), so no architecture-matched "
                    "lineage-vs-base pair survives (observed_diff=NaN). The lineage-vs-"
                    "architecture hypothesis is UNTESTED by this run, not null. A valid run "
                    "needs a tool-call transport that works for non-tool-API models "
                    "(prompt-format tool calling), or an r1-distill served with a tool API."
                    if not testable else
                    "Contrast recomputed on surviving models; see cleaned_contrast."),
    }
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"dead: {out['dead_models_excluded']}")
    print(f"kept: {out['n_models_kept']} models; lineage_pairs="
          f"{cleaned.get('lineage_pairs')}")
    print(f"observed_diff={cleaned.get('observed_diff'):.4f} p={cleaned.get('p_value'):.3f} "
          f"label_perm_p={cleaned.get('label_perm_p'):.3f}")
    print(f"verdict: {out['verdict'][:120]}")
    print(f"[reanalyze] wrote {OUT}")


if __name__ == "__main__":
    main()
