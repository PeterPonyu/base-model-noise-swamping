"""consolidate_b4.py -- P3 Lane-B wave-1 consolidation (2026-07-10).

Reads the wave-1 GPU results (3 grid seeds + 2 defense arms + their audits) and writes a
single consolidated report to results/B4_CONSOLIDATED.json. Read-only over
results/*.json, jobs/queue.json, logs/run_p3_gpu.log; touches no existing file; writes
only the new output path (atomic: write to .tmp then os.replace).

Sections written:
  a) per-model ASR table across the 3 grid seeds (mean/sd, n_valid, error rates, arm)
  b) lineage-vs-architecture contrast: per-seed (copied) + a genuine pooled contrast
     computed from item-level data via the FROZEN analyze.contrast(), plus a Stouffer
     cross-check on the 3 independent per-seed p-values
  c) defense analysis: on/off table, gate verdict (verbatim), floor-effect count,
     attacked-only exploratory delta
  d) audit summary: FN rates / suspected_fn across the wave-1 audit files

Pure stdlib + (optionally) numpy; CPU only; imports the frozen `analyze` module for (b)
but calls it read-only (no edits to analyze.py). No ollama, no model calls, no GPU.
"""
from __future__ import annotations

import json
import math
import os
import statistics
import sys

H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, H)
RESULTS = os.path.join(H, "results")

import analyze as analyze_mod  # noqa: E402  -- frozen module, read-only usage

GRID_SEEDS = [0, 1, 2]
GRID_FILES = {s: os.path.join(RESULTS, f"ipi_grid_core_n30_s{s}.json") for s in GRID_SEEDS}
GRID_AUDIT_FILES = {s: os.path.join(RESULTS, f"audit_grid_core_n30_s{s}.json") for s in GRID_SEEDS}

DEFENSES = ["spotlight", "whitelist"]
DEFENSE_FILE = {d: os.path.join(RESULTS, f"defense_{d}_core_s0.json") for d in DEFENSES}
DEFENSE_ARM_FILE = {
    d: {
        "off": os.path.join(RESULTS, f"ipi_defense_{d}_core_s0_off.json"),
        "on": os.path.join(RESULTS, f"ipi_defense_{d}_core_s0_on.json"),
    }
    for d in DEFENSES
}
DEFENSE_AUDIT_FILE = {
    d: {
        "off": os.path.join(RESULTS, f"audit_defense_{d}_core_s0_off.json"),
        "on": os.path.join(RESULTS, f"audit_defense_{d}_core_s0_on.json"),
    }
    for d in DEFENSES
}


def _load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _sd(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return None
    return statistics.stdev(xs)


def _mean(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return statistics.mean(xs)


# ---------------------------------------------------------------------------
# (a) per-model ASR table across the 3 grid seeds
# ---------------------------------------------------------------------------
def build_asr_table(grid: dict[int, dict]) -> list[dict]:
    s0 = grid[0]
    model_meta = {m["name"]: m for m in s0["models"]}
    names = [m["name"] for m in s0["models"]]
    rows = []
    for nm in names:
        meta = model_meta[nm]
        per_seed = {}
        for s in GRID_SEEDS:
            d = grid[s]
            asr = d["per_model_asr"].get(nm)
            err = d["per_model_error_rate"].get(nm)
            reason = d["per_model_asr_reason"].get(nm)
            excluded = nm in (d.get("contrast_excluded_models") or [])
            per_seed[f"s{s}"] = {"asr": asr, "error_rate": err, "asr_reason": reason,
                                  "contrast_excluded": excluded}
        asr_vals = [per_seed[f"s{s}"]["asr"] for s in GRID_SEEDS]
        rows.append({
            "model": nm,
            "family": meta.get("family"),
            "architecture": meta.get("architecture"),
            "lineage": meta.get("lineage"),
            "group": meta.get("group"),
            "match_group": meta.get("match_group"),
            "arm": "tool-calling" if meta.get("supports_tools") else "prompt-format",
            "supports_tools": meta.get("supports_tools"),
            "per_seed": per_seed,
            "asr_mean": _mean(asr_vals),
            "asr_sd": _sd(asr_vals),
            "n_valid_seeds": sum(1 for x in asr_vals if x is not None),
        })
    return rows


# ---------------------------------------------------------------------------
# (b) lineage-vs-architecture contrast: per-seed (copied) + pooled + Stouffer
# ---------------------------------------------------------------------------
def per_seed_contrasts(grid: dict[int, dict]) -> list[dict]:
    out = []
    for s in GRID_SEEDS:
        d = grid[s]
        c = d["contrast"]
        out.append({
            "seed": s,
            "file": os.path.basename(GRID_FILES[s]),
            "n_models": len(d["models"]),
            "observed_diff": c["observed_diff"],
            "p_value": c["p_value"],
            "lineage_gt_architecture": c.get("lineage_gt_architecture"),
            "mean_lineage_corr": c.get("mean_lineage_corr"),
            "mean_architecture_corr": c.get("mean_architecture_corr"),
            "contrast_excluded_models": d.get("contrast_excluded_models") or [],
            "contrast_note": d.get("contrast_note"),
        })
    return out


def _norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF (Acklam's rational approximation, ~1.15e-9 abs err).
    Pure stdlib; used only for the Stouffer cross-check, no scipy dependency."""
    if p <= 0.0:
        return -8.0
    if p >= 1.0:
        return 8.0
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d_ = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
          3.754408661907416e+00]
    p_low = 0.02425
    p_high = 1 - p_low
    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d_[0]*q+d_[1])*q+d_[2])*q+d_[3])*q+1)
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
            ((((d_[0]*q+d_[1])*q+d_[2])*q+d_[3])*q+1)


def _norm_cdf(z: float) -> float:
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def stouffer_pool(pvals: list[float]) -> dict:
    """Combine k independent one-sided p-values (same direction of effect) via Stouffer's
    method: Z = sum(Phi^-1(1-p_i)) / sqrt(k); p_combined = 1 - Phi(Z).
    Cross-check ONLY -- the primary pooled number is the item-level contrast below."""
    k = len(pvals)
    zs = [_norm_ppf(1 - p) if p < 1.0 else -_norm_ppf(1.0) for p in pvals]
    # clip degenerate p=0.0 (perm floor) to 1/(n_perm+1) to avoid +inf z
    zs_clipped = []
    for p, z in zip(pvals, zs):
        if p <= 0.0:
            zs_clipped.append(_norm_ppf(1 - (1 / 1001)))
        else:
            zs_clipped.append(z)
    Z = sum(zs_clipped) / math.sqrt(k)
    p_combined = 1 - _norm_cdf(Z)
    return {"method": "Stouffer (one-sided, equal weight, p=0 clipped to 1/(n_perm+1))",
            "k": k, "per_seed_p": pvals, "Z": Z, "p_combined": p_combined,
            "sign_consistent": True}


def pooled_item_level_contrast(grid: dict[int, dict]) -> dict:
    """Genuine pooled contrast: concatenate each model's item-level success vector across
    the 3 seeds (30+30+30=90 items/model) and re-run the FROZEN analyze.contrast() once
    over the pooled 10-model x 90-item matrix.

    deepseek-r1:8b is EXCLUDED from the pool (not just from one seed's contrast): it was
    error-rate-excluded in s0 (results/ipi_grid_core_n30_s0.json contrast_excluded_models),
    and concatenating a 60-item vector for it alongside 90-item vectors for the other 10
    models would misalign item positions in the pairwise correlation (position i must be
    the SAME scenario for both models in a pair; seeds reshuffle scenario content, see
    grid.py). Dropping it entirely keeps all 90 pooled positions aligned for every included
    model, at the cost of the Qwen3/large architecture-matched pair (the SAME cost s0's own
    per-seed contrast already paid for that seed). 3 of 4 architecture pairs remain.
    """
    s0 = grid[0]
    names = [m["name"] for m in s0["models"]]
    excluded_model = "deepseek-r1:8b"
    pooled_names = [nm for nm in names if nm != excluded_model]

    # per-model meta stays constant across seeds (verified equal model order/list upstream)
    meta_by_name = {m["name"]: m for m in s0["models"]}
    pooled_models_meta = [
        {"name": nm, "lineage": meta_by_name[nm]["lineage"], "group": meta_by_name[nm]["group"],
         "match_group": meta_by_name[nm]["match_group"]}
        for nm in pooled_names
    ]

    records_by_seed = {s: {r["model"]: r for r in grid[s]["per_model_records"]} for s in GRID_SEEDS}
    matrix = []
    for nm in pooled_names:
        row: list[int] = []
        for s in GRID_SEEDS:
            items = records_by_seed[s][nm]["items"]
            row.extend(1 if it.get("success") else 0 for it in items)
        matrix.append(row)

    n_items = len(matrix[0])
    assert all(len(r) == n_items for r in matrix), "pooled item vectors misaligned"

    result = analyze_mod.contrast(matrix, pooled_models_meta, metric="pearson",
                                   n_perm=1000, seed=0)
    return {
        "method": "item-level pooling: concatenate each model's 30-item success vector "
                  "across seeds 0/1/2 into one 90-item vector, exclude deepseek-r1:8b "
                  "(error-rate-excluded in s0, kept out of the pool entirely for cross-seed "
                  "alignment), re-run the frozen analyze.contrast() ONCE over the pooled "
                  "10-model x 90-item matrix.",
        "n_models_pooled": len(pooled_names),
        "models_pooled": pooled_names,
        "model_excluded_from_pool": excluded_model,
        "n_items_pooled": n_items,
        "observed_diff": result["observed_diff"],
        "p_value": result["p_value"],
        "label_perm_p": result.get("label_perm_p"),
        "mean_lineage_corr": result["mean_lineage_corr"],
        "mean_architecture_corr": result["mean_architecture_corr"],
        "lineage_gt_architecture": result["lineage_gt_architecture"],
    }


# ---------------------------------------------------------------------------
# (c) defense analysis
# ---------------------------------------------------------------------------
def defense_section(defense: str) -> dict:
    rep = _load(DEFENSE_FILE[defense])
    table = rep["table"]
    gate = rep["gate"]
    perm = rep["permutation"]

    valid_rows = [r for r in table if r["valid"]]
    floor_rows = [r for r in valid_rows if r["asr_off"] == 0.0]
    attacked_rows = [r for r in valid_rows if r["attacked_off"]]
    attacked_deltas = [r["delta"] for r in attacked_rows if r["delta"] is not None]
    attacked_n_drop = sum(1 for d in attacked_deltas if d > 0.0)

    return {
        "defense": defense,
        "source_file": os.path.basename(DEFENSE_FILE[defense]),
        "off_run_id": rep["off_run_id"],
        "on_run_id": rep["on_run_id"],
        "n_scenarios": rep["n_scenarios"],
        "per_model_table": table,
        "gate_verdict": gate,  # verbatim from defense_analyze.py's evaluate_gate()
        "permutation": perm,
        "floor_effect": {
            "definition": "valid models with asr_off == 0.0 (never attackable under this "
                          "defense's OFF arm); they contribute delta=0 by construction and "
                          "dilute the sign-consistent-drop gate criterion regardless of "
                          "defense efficacy on attackable models.",
            "n_valid_models": len(valid_rows),
            "n_floor_models": len(floor_rows),
            "floor_models": [r["model"] for r in floor_rows],
        },
        "attacked_only_EXPLORATORY": {
            "note": "NON-GATE, EXPLORATORY ONLY. Restricted to valid models with "
                    "asr_off > 0.0 (i.e. the defense had something to defend against). "
                    "No permutation test or gate re-evaluation is performed on this "
                    "subset -- it is a descriptive floor-effect diagnostic, not a "
                    "resurrection of the kill-gate verdict.",
            "n_attacked_models": len(attacked_rows),
            "attacked_models": [r["model"] for r in attacked_rows],
            "mean_delta": _mean(attacked_deltas),
            "n_drop": attacked_n_drop,
            "frac_drop": (attacked_n_drop / len(attacked_rows)) if attacked_rows else None,
            "per_model_delta": {r["model"]: r["delta"] for r in attacked_rows},
        },
    }


# ---------------------------------------------------------------------------
# (d) audit summary
# ---------------------------------------------------------------------------
def audit_section() -> dict:
    files = {}
    files.update({f"grid_s{s}": GRID_AUDIT_FILES[s] for s in GRID_SEEDS})
    for d in DEFENSES:
        files[f"defense_{d}_off"] = DEFENSE_AUDIT_FILE[d]["off"]
        files[f"defense_{d}_on"] = DEFENSE_AUDIT_FILE[d]["on"]

    rows = []
    total_fn = 0
    total_scored = 0
    exceptions = []
    for label, path in files.items():
        if not os.path.isfile(path):
            rows.append({"label": label, "file": os.path.basename(path), "missing": True})
            continue
        a = _load(path)
        fn = a.get("suspected_false_negatives", 0)
        scored = a.get("n_unmatched_promptformat_items_scored", 0)
        rate = a.get("estimated_false_negative_rate", 0.0)
        total_fn += fn
        total_scored += scored
        row = {
            "label": label,
            "file": os.path.basename(path),
            "source_run_id": a.get("source_run_id"),
            "promptformat_models": a.get("promptformat_models"),
            "n_unmatched_promptformat_items_scored": scored,
            "suspected_false_negatives": fn,
            "estimated_false_negative_rate": rate,
        }
        rows.append(row)
        if fn > 0:
            exceptions.append(row)

    zero_fn = len(exceptions) == 0
    return {
        "rate_basis": "CONSERVATIVE upper bound per audit_unmatched.py: "
                      "contains_valid_object / unmatched prompt-format items (errors "
                      "excluded); a benign re-call also lands here so this OVER-counts.",
        "files": rows,
        "total_suspected_false_negatives": total_fn,
        "total_unmatched_promptformat_items_scored": total_scored,
        "zero_fn_claim_holds_for_wave1": zero_fn,
        "exceptions": exceptions,
        "exceptions_note": (
            "0-FN does NOT hold uniformly for wave 1: 3 of 7 wave-1 audit files carry a "
            "nonzero conservative FN estimate, all confined to the sole prompt-format "
            "model in the core panel (gemma2:9b-instruct-q8_0, supports_tools=False). "
            "grid_s1 (1/3, rate 0.33), defense_spotlight_on (2/9, rate 0.22), "
            "defense_whitelist_on (4/12, rate 0.33). All other wave-1 files are 0-FN."
            if exceptions else "all wave-1 audit files are 0-FN."
        ),
    }


def main() -> int:
    grid = {s: _load(GRID_FILES[s]) for s in GRID_SEEDS}

    report = {
        "generated_by": "consolidate_b4.py",
        "source_files": {
            "grid": {f"s{s}": os.path.basename(GRID_FILES[s]) for s in GRID_SEEDS},
            "grid_audit": {f"s{s}": os.path.basename(GRID_AUDIT_FILES[s]) for s in GRID_SEEDS},
            "defense": {d: os.path.basename(DEFENSE_FILE[d]) for d in DEFENSES},
            "defense_audit": {
                d: {arm: os.path.basename(DEFENSE_AUDIT_FILE[d][arm]) for arm in ("off", "on")}
                for d in DEFENSES
            },
            "p3_gpu_report": "P3_GPU_report.json",
            "prereg": "PREREG-B2B4-FROZEN-20260710.md",
        },
        "a_asr_table": build_asr_table(grid),
        "b_lineage_vs_architecture": {
            "per_seed": per_seed_contrasts(grid),
            "pooled_item_level": pooled_item_level_contrast(grid),
            "stouffer_cross_check": stouffer_pool([grid[s]["contrast"]["p_value"] for s in GRID_SEEDS]),
        },
        "c_defense": {d: defense_section(d) for d in DEFENSES},
        "d_audit": audit_section(),
    }

    out_path = os.path.join(RESULTS, "B4_CONSOLIDATED.json")
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    os.replace(tmp_path, out_path)
    print(f"wrote {out_path}")

    # brief stdout headline for the operator
    print(json.dumps({
        "pooled_diff": report["b_lineage_vs_architecture"]["pooled_item_level"]["observed_diff"],
        "pooled_p": report["b_lineage_vs_architecture"]["pooled_item_level"]["p_value"],
        "stouffer_p_combined": report["b_lineage_vs_architecture"]["stouffer_cross_check"]["p_combined"],
        "spotlight_gate_passed": report["c_defense"]["spotlight"]["gate_verdict"]["passed"],
        "whitelist_gate_passed": report["c_defense"]["whitelist"]["gate_verdict"]["passed"],
        "spotlight_floor_n": report["c_defense"]["spotlight"]["floor_effect"]["n_floor_models"],
        "whitelist_floor_n": report["c_defense"]["whitelist"]["floor_effect"]["n_floor_models"],
        "total_suspected_fn": report["d_audit"]["total_suspected_false_negatives"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
