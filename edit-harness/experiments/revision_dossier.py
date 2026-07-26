"""revision_dossier.py — B6 revision-readiness dossier core.

B6 was submitted to an IEEE journal with 4 extension causal cells at single seed s0
(MQuAKE, ripple, instruct, 8B alpha-holdout). A local GPU queue (Lane A) produces seeds
s1/s2 for those cells plus an extra 8B layer (L28). This script compares the submitted
s0-only numbers against whatever 3-seed evidence has landed so far and flags which paper
claims are STABLE vs SHIFTED, so a revision response can be drafted same-day.

Two kinds of ground truth are consulted, and neither is recomputed here (stdlib only,
no numpy/torch — this must stay loadable anywhere):
  - CAUSAL / GATE cells (MQuAKE, instruct, 8B): the canonical within-probe (partialled)
    Spearman rho is only ever computed from matched .npz matrices by
    aggregate_g4_causal.py (causal: key-cos vs damage_removed) or analyze_matrices.py
    (gate: key-cos vs damage, standalone). Those scripts' output aggregate JSONs
    (results/C4_causal_*_table*.json, results/C3_mquake_alpha_L12_3seed.json) are the
    ONLY source for that statistic — the per-seed raw run JSONs (g4_*, gate_*) do NOT
    carry it, only a pooled/flat spearman_cos_damage that the project's own C3 aggregate
    labels "_INFLATED" and treats as non-canonical. So a raw per-seed JSON existing on
    disk does not by itself produce a dossier data point for these families; it only
    counts toward "raw seeds on disk" so we can flag a STALE aggregate (raw seeds exist
    that the aggregate hasn't picked up yet -> rerun the aggregator).
  - RIPPLE cells: within_probe_rho_logit.ripple / .unrelated IS carried directly in each
    per-seed ripple_*.json, so those are read directly (results/RIPPLE_depth_profile.json
    is consulted only as a cross-check, not as the primary source).

REVINS ADDITIONS (2026-07-11, run_revins.sh — a second local GPU queue, independent of Lane
A, producing 4 more cell families the paper may get asked about on revision):
  - Cell A: mquake_causal_holdout_L12 — same aggregate-is-canonical rule as the other CAUSAL
    cells, honest --alpha_proj_source holdout variant of the existing (circular,
    probes-sourced) mquake_causal_L12; cross-checked against it directly (see
    cross_check_mquake_holdout).
  - Cell B: grace_damage_L12 — NOT a rho family. grace's ΔW≡0 codebook mechanism makes
    damage_logit identically zero on unrelated probes by construction, so within-probe rho
    is undefined; reported as a DESCRIPTIVE row from results/GRACE_damage_report_revins.json
    instead (verdict in {PENDING, DESCRIPTIVE_CONFIRMED, DESCRIPTIVE_ANOMALY}).
  - Cell C: gradsim_true_L{8,10,12,14} — True-GradSim rank-agreement (A4' test). No
    aggregate_*.py table exists for this family; each GRADSIM_TRUE_*.json already carries
    rank_agreement.direct_vs_SC.mean directly, read the same way as RIPPLE cells.
  - Cell D: gptj_alphaHO_L21 — GPT-J causal seed parity, fits the existing CAUSAL_CELLS
    shape exactly (same aggregate_g4_causal.py --proj_source holdout tool, new model).

Run from edit-harness/:
  python experiments/revision_dossier.py
  python experiments/revision_dossier.py --results_dir results --out results/REVISION_DOSSIER.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
from pathlib import Path

SEED_NPZ_RE = re.compile(r"_s(\d+)\.npz$")

# ---------------------------------------------------------------------------
# cell family definitions
# ---------------------------------------------------------------------------

CAUSAL_CELLS = [
    # key, agg_file, agg_layer_key, alpha_raw_pattern, rome_raw_pattern, note
    ("instruct_alphaHO_L12", "C4_causal_instruct_table_3seed.json", "12",
     "g4_instruct_alphaHO_cf_L12_s{s}.json", "gate_instruct_rome_cf_L12_s{s}.json"),
    ("8b_alphaHO_L16", "C4_causal_8b_table_3seed.json", "16",
     "g4_llama8b_alphaHO_cf_L16_s{s}.json", "gate_llama8b_rome_cf_L16_s{s}.json"),
    ("8b_alphaHO_L24", "C4_causal_8b_table_3seed.json", "24",
     "g4_llama8b_alphaHO_cf_L24_s{s}.json", "gate_llama8b_rome_cf_L24_s{s}.json"),
    ("8b_alphaHO_L28", "C4_causal_8b_table_3seed.json", "28",
     "g4_llama8b_alphaHO_cf_L28_s{s}.json", "gate_llama8b_rome_cf_L28_s{s}.json"),
    ("mquake_causal_L12", "C4_causal_mquake_table_3seed_probesrc.json", "12",
     "gate_llama1b_alpha_mquake_L12_s{s}.json", "gate_llama1b_rome_mquake_L12_s{s}.json"),
    # --- revins additions (2026-07-11) ---
    # Cell A: the HONEST holdout-projector MQuAKE causal cell (run_revins.sh Cell A). The
    # entry above (mquake_causal_L12) fits the projector on the SAME probes it's scored
    # against (--alpha_proj_source probes, circular — memory: c4-alphaedit-projector-
    # circularity.md); this one uses --alpha_proj_source holdout. Same ROME reference npz
    # (gate_llama1b_rome_mquake_L12_s{s}) is shared by both — only the alpha arm differs.
    ("mquake_causal_holdout_L12", "C4_causal_mquake_holdout_table_3seed.json", "12",
     "g4_llama1b_alphaHO_mquake_L12_s{s}.json", "gate_llama1b_rome_mquake_L12_s{s}.json"),
    # Cell D: GPT-J AlphaEdit(-holdout) causal seed parity (run_revins.sh Cell D). s0 was
    # the original cross-arch single-seed cell; s1/s2 are the revins additions.
    ("gptj_alphaHO_L21", "C4_causal_gptj_table_3seed.json", "21",
     "g4_gptj_alphaHO_cf_L21_s{s}.json", "gate_gptj_rome_cf_L21_s{s}.json"),
]
CAUSAL_METRIC_LABEL = ("within_probe_spearman(key-cos, damage_removed) "
                        "— causal AlphaEdit(-holdout) collateral-damage-removed vs ROME")

GATE_CELLS = [
    # key, agg_file, raw_pattern (for esr; same file the npz name embeds a seed)
    ("mquake_gate_L12", "C3_mquake_alpha_L12_3seed.json",
     "gate_llama1b_alpha_mquake_L12_s{s}.json"),
]
GATE_METRIC_LABEL = ("within_probe_mean(key-cos, damage) "
                      "— standalone AlphaEdit gate test, partialled Spearman")

RIPPLE_CELLS = [
    # key_prefix, raw_pattern, layer
    ("ripple_rome_L8", "ripple_llama1b_rome_popular_L8_s{s}.json", 8),
    ("ripple_rome_L10", "ripple_llama1b_rome_popular_L10_s{s}.json", 10),
    ("ripple_rome_L12", "ripple_llama1b_rome_popular_L12_s{s}.json", 12),
    ("ripple_rome_L14", "ripple_llama1b_rome_popular_L14_s{s}.json", 14),
    ("ripple_alpha_L12", "ripple_llama1b_alpha_popular_L12_s{s}.json", 12),
]
RIPPLE_DEPTH_PROFILE = "RIPPLE_depth_profile.json"

# Cell C (run_revins.sh, revins 2026-07-11): True-GradSim multi-seed/multi-layer expansion.
# The submitted claim is a SINGLE cell (L12 s0): rank_agreement.direct_vs_SC.mean = 0.087
# (memory: gradsim-true-result-20260707.md, "A4' half-met"). gradsim_true.py has no
# aggregator of its own (unlike the causal/gate families) — its per-(layer,seed) JSON
# already carries the canonical statistic directly, so these are read the same way as the
# RIPPLE_CELLS direct-per-seed family, not through an aggregate_*.py table. L12 gets true
# seed parity (s0 existing + s1/s2 new); L8/L10/L14 are brand-new layers at s0 only (no
# prior claim to compare against — informational, will read PENDING until more seeds land).
GRADSIM_TRUE_CELLS = [
    # key, raw_pattern, layer
    ("gradsim_true_L8", "GRADSIM_TRUE_Llama-3.2-1B_L8_s{s}.json", 8),
    ("gradsim_true_L10", "GRADSIM_TRUE_Llama-3.2-1B_L10_s{s}.json", 10),
    ("gradsim_true_L12", "GRADSIM_TRUE_Llama-3.2-1B_L12_s{s}.json", 12),
    ("gradsim_true_L14", "GRADSIM_TRUE_Llama-3.2-1B_L14_s{s}.json", 14),
]
GRADSIM_TRUE_METRIC_LABEL = ("rank_agreement.direct_vs_SC.mean — TRUE-backprop direct "
                              "influence vs S×C surrogate rank agreement (A4' "
                              "de-tautologization test)")

# Cell B (run_revins.sh, revins 2026-07-11): GRACE EGL damage report. NOT a Spearman-rho
# family — grace's ΔW≡0 codebook mechanism leaves damage_logit identically zero on
# unrelated probes by construction, so a within-probe rho over a constant column is
# undefined (NaN). run_revins.sh's own post-run step already reduces this to a direct
# damage-identically-zero check (results/GRACE_damage_report_revins.json); the dossier just
# surfaces that report as a DESCRIPTIVE row (verdict in {PENDING, DESCRIPTIVE_CONFIRMED,
# DESCRIPTIVE_ANOMALY}), never computing or expecting a rho for it.
GRACE_DAMAGE_REPORT_FILE = "GRACE_damage_report_revins.json"
GRACE_METRIC_LABEL = ("damage_logit identically-zero check (descriptive, NOT a Spearman rho "
                       "— see run_revins.sh Cell B header / GRACE_damage_report_revins.json)")

SEEDS = (0, 1, 2)


# ---------------------------------------------------------------------------
# io helpers
# ---------------------------------------------------------------------------

def load_json(path: Path, generated_from: list, pending: list):
    """Load a JSON file if present; track it in generated_from/pending. Never raises."""
    if not path.exists():
        pending.append(str(path))
        return None
    try:
        with open(path, "r") as f:
            data = json.load(f)
        generated_from.append(str(path))
        return data
    except Exception as e:  # corrupt / partial write mid-queue
        pending.append(f"{path} (unreadable: {e})")
        return None


def dget(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


# ---------------------------------------------------------------------------
# verdict logic
# ---------------------------------------------------------------------------

def verdict_for(s0, per_seed_vals):
    """per_seed_vals: dict[int seed] -> float, only seeds with a computed value.

    Returns (verdict, mean, sd, sign_consistent).
    """
    vals = [v for v in per_seed_vals.values() if v is not None]
    n = len(vals)
    if n < 2:
        return "PENDING", None, None, None

    mean = statistics.mean(vals)
    sd = statistics.stdev(vals) if n >= 2 else None
    signs = {(1 if v > 0 else (-1 if v < 0 else 0)) for v in vals}
    sign_consistent = len(signs) <= 1

    # Near-zero cells: sign flips among values that are all ~0 are noise around a
    # null, not a shifted claim (seen live: mquake gate L12 = -0.074/+0.022/-0.066).
    # Report them as their own verdict so a null result reads as stable-null.
    all_vals = vals + ([s0] if s0 is not None else [])
    if all(abs(v) < 0.1 for v in all_vals):
        return "STABLE_NULL", mean, sd, sign_consistent

    shifted = not sign_consistent
    if s0 is not None and abs(mean - s0) > 0.15:
        shifted = True

    threshold = max(0.15, 0.5 * abs(s0)) if s0 is not None else 0.15
    rng = max(vals) - min(vals)
    stable_range = sign_consistent and (rng <= threshold)

    verdict = "STABLE" if (stable_range and not shifted) else "SHIFTED"
    return verdict, mean, sd, sign_consistent


def esr_lookup(results_dir, raw_pattern, generated_from, pending, seeds=SEEDS):
    """Return {seed: edit_success_rate or None} by reading each seed's raw JSON."""
    out = {}
    for s in seeds:
        p = results_dir / raw_pattern.format(s=s)
        d = load_json(p, generated_from, pending)
        out[s] = d.get("edit_success_rate") if d is not None else None
    return out


def raw_seeds_on_disk(results_dir, raw_pattern, seeds=SEEDS):
    return [s for s in seeds if (results_dir / raw_pattern.format(s=s)).exists()]


# ---------------------------------------------------------------------------
# per-family builders
# ---------------------------------------------------------------------------

def build_causal_cell(results_dir, key, agg_file, agg_layer, alpha_pat, rome_pat,
                       generated_from, pending, notes_out):
    agg_path = results_dir / agg_file
    agg = load_json(agg_path, generated_from, pending)

    per_seed_rho = {}
    aggregate_seeds_used = []
    if agg is not None:
        layer_obj = dget(agg, "layers", agg_layer)
        if layer_obj is not None:
            aggregate_seeds_used = layer_obj.get("seeds_used", []) or []
            rhos = layer_obj.get("within_probe_spearman_per_seed", []) or []
            for s, r in zip(aggregate_seeds_used, rhos):
                per_seed_rho[int(s)] = r
        else:
            notes_out.append(f"{key}: aggregate '{agg_file}' has no layer {agg_layer} entry yet")

    esr_per_seed = esr_lookup(results_dir, alpha_pat, generated_from, pending)
    alpha_raw_seeds = raw_seeds_on_disk(results_dir, alpha_pat)
    rome_raw_seeds = raw_seeds_on_disk(results_dir, rome_pat)

    stale_seeds = sorted((set(alpha_raw_seeds) & set(rome_raw_seeds)) - set(aggregate_seeds_used))
    if stale_seeds:
        notes_out.append(
            f"{key}: raw materials for seed(s) {stale_seeds} exist on disk for BOTH "
            f"alpha and rome arms but the aggregate table only reflects "
            f"{sorted(aggregate_seeds_used)} — rerun aggregate_g4_causal.py to refresh."
        )

    s0 = per_seed_rho.get(0)
    verdict, mean, sd, sign_consistent = verdict_for(s0, per_seed_rho)
    esr_warn = any(v is not None and v < 0.9 for v in esr_per_seed.values())

    return {
        "metric": CAUSAL_METRIC_LABEL,
        "source": str(agg_path),
        "s0": s0,
        "per_seed": {f"s{s}": per_seed_rho.get(s) for s in SEEDS},
        "mean": mean,
        "sd": sd,
        "sign_consistent": sign_consistent,
        "n_seeds_available": len([v for v in per_seed_rho.values() if v is not None]),
        "verdict": verdict,
        "esr_warn": esr_warn,
        "esr_per_seed": {f"s{s}": esr_per_seed.get(s) for s in SEEDS},
        "raw_alpha_seeds_on_disk": alpha_raw_seeds,
        "raw_rome_seeds_on_disk": rome_raw_seeds,
        "aggregate_seeds_used": sorted(int(s) for s in aggregate_seeds_used),
    }


def build_gate_cell(results_dir, key, agg_file, raw_pat, generated_from, pending, notes_out):
    agg_path = results_dir / agg_file
    agg = load_json(agg_path, generated_from, pending)

    per_seed_rho = {}
    if agg is not None:
        for entry in agg.get("per_seed", []) or []:
            npz_name = entry.get("npz", "")
            m = SEED_NPZ_RE.search(npz_name)
            if m:
                per_seed_rho[int(m.group(1))] = entry.get("within_probe_mean")

    esr_per_seed = esr_lookup(results_dir, raw_pat, generated_from, pending)
    raw_seeds = raw_seeds_on_disk(results_dir, raw_pat)
    stale_seeds = sorted(set(raw_seeds) - set(per_seed_rho.keys()))
    if stale_seeds:
        notes_out.append(
            f"{key}: raw run JSON for seed(s) {stale_seeds} exist on disk but the "
            f"standalone gate aggregate '{agg_file}' has not been regenerated for them."
        )

    s0 = per_seed_rho.get(0)
    verdict, mean, sd, sign_consistent = verdict_for(s0, per_seed_rho)
    esr_warn = any(v is not None and v < 0.9 for v in esr_per_seed.values())

    return {
        "metric": GATE_METRIC_LABEL,
        "source": str(agg_path),
        "s0": s0,
        "per_seed": {f"s{s}": per_seed_rho.get(s) for s in SEEDS},
        "mean": mean,
        "sd": sd,
        "sign_consistent": sign_consistent,
        "n_seeds_available": len([v for v in per_seed_rho.values() if v is not None]),
        "verdict": verdict,
        "esr_warn": esr_warn,
        "esr_per_seed": {f"s{s}": esr_per_seed.get(s) for s in SEEDS},
        "raw_seeds_on_disk": raw_seeds,
    }


def build_ripple_cells(results_dir, key_prefix, raw_pat, layer, generated_from, pending, notes_out):
    """Returns {key_prefix + '_ripple': cell, key_prefix + '_unrelated': cell}."""
    per_seed = {"ripple": {}, "unrelated": {}}
    esr_per_seed = {}
    for s in SEEDS:
        p = results_dir / raw_pat.format(s=s)
        d = load_json(p, generated_from, pending)
        if d is None:
            continue
        rho = dget(d, "within_probe_rho_logit", default={}) or {}
        if not rho:
            notes_out.append(
                f"{key_prefix}: {p} loaded but has no 'within_probe_rho_logit' object "
                f"(seed {s} would otherwise silently score as a missing rho, not a "
                f"data-quality problem — check the run)."
            )
        per_seed["ripple"][s] = rho.get("ripple")
        per_seed["unrelated"][s] = rho.get("unrelated")
        esr_per_seed[s] = d.get("edit_success_rate")

    out = {}
    for sub in ("ripple", "unrelated"):
        s0 = per_seed[sub].get(0)
        verdict, mean, sd, sign_consistent = verdict_for(s0, per_seed[sub])
        esr_warn = any(v is not None and v < 0.9 for v in esr_per_seed.values())
        out[f"{key_prefix}_{sub}"] = {
            "metric": f"within_probe_rho_logit.{sub} (RippleEdits {sub} probes, direct per-seed)",
            "source": str(results_dir / raw_pat.format(s="{seed}")),
            "layer": layer,
            "s0": s0,
            "per_seed": {f"s{s}": per_seed[sub].get(s) for s in SEEDS},
            "mean": mean,
            "sd": sd,
            "sign_consistent": sign_consistent,
            "n_seeds_available": len([v for v in per_seed[sub].values() if v is not None]),
            "verdict": verdict,
            "esr_warn": esr_warn,
            "esr_per_seed": {f"s{s}": esr_per_seed.get(s) for s in SEEDS},
        }
    return out


def build_gradsim_true_cell(results_dir, key, raw_pat, layer, generated_from, pending, notes_out):
    """Cell C: read rank_agreement.direct_vs_SC.mean directly from each seed's
    GRADSIM_TRUE_*.json (no aggregate_*.py table exists for this family — see the
    GRADSIM_TRUE_CELLS comment)."""
    per_seed = {}
    for s in SEEDS:
        p = results_dir / raw_pat.format(s=s)
        d = load_json(p, generated_from, pending)
        if d is None:
            continue
        per_seed[s] = dget(d, "rank_agreement", "direct_vs_SC", "mean")

    raw_seeds = raw_seeds_on_disk(results_dir, raw_pat)
    if raw_seeds and len(raw_seeds) < 2:
        notes_out.append(
            f"{key}: only seed(s) {raw_seeds} on disk so far — this layer needs 2+ seeds "
            f"before a STABLE/SHIFTED verdict is meaningful (single-layer cells at L8/L10/"
            f"L14 are new exploration, not seed parity, per run_revins.sh Cell C)."
        )

    s0 = per_seed.get(0)
    verdict, mean, sd, sign_consistent = verdict_for(s0, per_seed)

    return {
        "metric": GRADSIM_TRUE_METRIC_LABEL,
        "source": str(results_dir / raw_pat.format(s="{seed}")),
        "layer": layer,
        "s0": s0,
        "per_seed": {f"s{s}": per_seed.get(s) for s in SEEDS},
        "mean": mean,
        "sd": sd,
        "sign_consistent": sign_consistent,
        "n_seeds_available": len([v for v in per_seed.values() if v is not None]),
        "verdict": verdict,
        "esr_warn": False,
        "raw_seeds_on_disk": raw_seeds,
    }


def build_grace_damage_cell(results_dir, generated_from, pending, notes_out):
    """Cell B: descriptive damage-identically-zero check, NOT a rho row (see
    GRACE_DAMAGE_REPORT_FILE comment)."""
    path = results_dir / GRACE_DAMAGE_REPORT_FILE
    d = load_json(path, generated_from, pending)
    common = {
        "metric": GRACE_METRIC_LABEL,
        "source": str(path),
        "descriptive": True,
        "s0": None, "mean": None, "sd": None, "sign_consistent": None,
        "esr_warn": False,
    }
    if d is None:
        return {**common, "per_seed": {}, "n_seeds_available": 0, "verdict": "PENDING"}

    per_seed_rows = d.get("per_seed", []) or []
    n_seeds = d.get("n_seeds_found", len(per_seed_rows))
    all_zero = d.get("damage_identically_zero_all_seeds")
    per_seed_zero = {}
    for row in per_seed_rows:
        m = SEED_NPZ_RE.search(row.get("npz", ""))
        if m:
            per_seed_zero[f"s{m.group(1)}"] = row.get("damage_identically_zero")

    if n_seeds < len(SEEDS):
        notes_out.append(
            f"grace_damage_L12: only {n_seeds}/{len(SEEDS)} seed(s) present in "
            f"{GRACE_DAMAGE_REPORT_FILE} so far (queue mid-run) — descriptive verdict may "
            f"firm up once the remaining seeds land."
        )

    if all_zero is None:
        verdict = "PENDING"
    elif all_zero:
        verdict = "DESCRIPTIVE_CONFIRMED"
    else:
        verdict = "DESCRIPTIVE_ANOMALY"
        notes_out.append(
            "grace_damage_L12: damage_identically_zero_all_seeds=False — grace produced "
            "NON-zero collateral damage on some seed, contradicting the ΔW≡0 codebook "
            "argument in run_revins.sh Cell B header. Investigate before writing this up."
        )

    return {
        **common,
        "per_seed": per_seed_zero,
        "n_seeds_available": n_seeds,
        "verdict": verdict,
        "damage_identically_zero_all_seeds": all_zero,
    }


def cross_check_mquake_holdout(cells, notes_out):
    """Best-effort sanity check: does the new honest-holdout MQuAKE causal cell (Cell A)
    confirm the existing probes-sourced (circular) one, the same way the CF holdout table
    confirmed the CF probes-sourced number (memory: c4-alphaedit-projector-circularity.md,
    "CF holdout precedent L8 0.390 / L12 0.590")? Never fatal — just a note either way."""
    probes_mean = dget(cells, "mquake_causal_L12", "mean")
    holdout_mean = dget(cells, "mquake_causal_holdout_L12", "mean")
    if probes_mean is None or holdout_mean is None:
        return
    diff = abs(probes_mean - holdout_mean)
    if diff > 0.15:
        notes_out.append(
            f"mquake_causal_holdout_L12 ({holdout_mean:.3f}) diverges from the circular "
            f"probes-sourced mquake_causal_L12 ({probes_mean:.3f}) by {diff:.3f} (>0.15) — "
            f"the holdout projector does NOT confirm the probes-sourced number; investigate "
            f"before relying on either for a revision response."
        )
    else:
        notes_out.append(
            f"mquake_causal_holdout_L12 ({holdout_mean:.3f}) confirms the circular "
            f"probes-sourced mquake_causal_L12 ({probes_mean:.3f}), diff {diff:.3f} — "
            f"same pattern as the CF holdout precedent (L8 0.390 / L12 0.590)."
        )


def cross_check_ripple(results_dir, cells, generated_from, pending, notes_out):
    """Best-effort sanity check against RIPPLE_depth_profile.json. Never fatal."""
    path = results_dir / RIPPLE_DEPTH_PROFILE
    prof = load_json(path, generated_from, pending)
    if prof is None:
        return

    def check(mean_val, profile_mean, label):
        if mean_val is None or profile_mean is None:
            return
        if abs(mean_val - profile_mean) > 0.02:
            notes_out.append(
                f"cross-check mismatch: {label} dossier mean {mean_val:.4f} vs "
                f"{RIPPLE_DEPTH_PROFILE} mean {profile_mean:.4f} (>0.02 apart)"
            )

    rome_depth = prof.get("rome_depth_profile", {}) or {}
    for L in (8, 10, 12, 14):
        node = rome_depth.get(f"L{L}")
        if not node:
            continue
        check(dget(cells, f"ripple_rome_L{L}_ripple", "mean"), node.get("rho_ripple_mean"),
              f"ripple_rome_L{L}_ripple")
        check(dget(cells, f"ripple_rome_L{L}_unrelated", "mean"), node.get("rho_unrelated_mean"),
              f"ripple_rome_L{L}_unrelated")

    alpha_per_seed = prof.get("alpha_L12_per_seed", []) or []
    if alpha_per_seed:
        rr = [e.get("rho_ripple") for e in alpha_per_seed if e.get("rho_ripple") is not None]
        ru = [e.get("rho_unrelated") for e in alpha_per_seed if e.get("rho_unrelated") is not None]
        if rr:
            check(dget(cells, "ripple_alpha_L12_ripple", "mean"), statistics.mean(rr),
                  "ripple_alpha_L12_ripple")
        if ru:
            check(dget(cells, "ripple_alpha_L12_unrelated", "mean"), statistics.mean(ru),
                  "ripple_alpha_L12_unrelated")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="B6 revision-readiness dossier (s0-only vs 3-seed).")
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--out", default="results/REVISION_DOSSIER.json")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    generated_from: list = []
    pending: list = []
    notes: list = []

    cells = {}

    for key, agg_file, agg_layer, alpha_pat, rome_pat in CAUSAL_CELLS:
        cells[key] = build_causal_cell(results_dir, key, agg_file, agg_layer, alpha_pat, rome_pat,
                                        generated_from, pending, notes)

    for key, agg_file, raw_pat in GATE_CELLS:
        cells[key] = build_gate_cell(results_dir, key, agg_file, raw_pat,
                                      generated_from, pending, notes)

    for key_prefix, raw_pat, layer in RIPPLE_CELLS:
        cells.update(build_ripple_cells(results_dir, key_prefix, raw_pat, layer,
                                         generated_from, pending, notes))

    for key, raw_pat, layer in GRADSIM_TRUE_CELLS:
        cells[key] = build_gradsim_true_cell(results_dir, key, raw_pat, layer,
                                              generated_from, pending, notes)

    cells["grace_damage_L12"] = build_grace_damage_cell(results_dir, generated_from, pending, notes)

    cross_check_ripple(results_dir, cells, generated_from, pending, notes)
    cross_check_mquake_holdout(cells, notes)

    n_stable = sum(1 for c in cells.values() if c["verdict"] in ("STABLE", "STABLE_NULL"))
    n_shifted = sum(1 for c in cells.values() if c["verdict"] == "SHIFTED")
    n_pending = sum(1 for c in cells.values() if c["verdict"] == "PENDING")
    n_esr_warn = sum(1 for c in cells.values() if c.get("esr_warn"))
    # Descriptive cells (grace: PENDING/DESCRIPTIVE_CONFIRMED/DESCRIPTIVE_ANOMALY) never hit
    # "SHIFTED", so a real anomaly could otherwise sit in the table while this one-line
    # summary reads "shifted=0" all-clear (review MINOR, 2026-07-11) — surface it explicitly.
    n_descriptive_confirmed = sum(1 for c in cells.values() if c["verdict"] == "DESCRIPTIVE_CONFIRMED")
    n_descriptive_anomaly = sum(1 for c in cells.values() if c["verdict"] == "DESCRIPTIVE_ANOMALY")

    out = {
        "generated_from": sorted(set(generated_from)),
        "pending": sorted(set(pending)),
        "cells": cells,
        "summary": {
            "n_cells": len(cells),
            "n_stable": n_stable,
            "n_shifted": n_shifted,
            "n_pending": n_pending,
            "n_descriptive_confirmed": n_descriptive_confirmed,
            "n_descriptive_anomaly": n_descriptive_anomaly,
            "n_esr_warn": n_esr_warn,
            "notes": notes,
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(out, f, indent=2)
    os.replace(tmp_path, out_path)

    print_table(cells)
    print(f"\n[revision_dossier] wrote {out_path}")
    anomaly_flag = f" [DESCRIPTIVE_ANOMALY x{n_descriptive_anomaly} -- SEE TABLE]" if n_descriptive_anomaly else ""
    print(f"[revision_dossier] cells={len(cells)} stable={n_stable} shifted={n_shifted} "
          f"pending={n_pending} descriptive_confirmed={n_descriptive_confirmed} "
          f"descriptive_anomaly={n_descriptive_anomaly} esr_warn={n_esr_warn}{anomaly_flag}")
    return 0


def print_table(cells):
    rows = []
    for key, c in sorted(cells.items()):
        s0 = c.get("s0")
        mean = c.get("mean")
        sd = c.get("sd")
        s0_str = f"{s0:.3f}" if isinstance(s0, (int, float)) else "—"
        if mean is None:
            mean_str = "—"
        elif sd is None:
            mean_str = f"{mean:.3f}±—"
        else:
            mean_str = f"{mean:.3f}±{sd:.3f}"
        flag = " [ESR<0.9]" if c.get("esr_warn") else ""
        rows.append((key, s0_str, mean_str, c["verdict"] + flag))

    widths = [max(len(r[i]) for r in rows + [("cell", "s0", "mean±sd", "verdict")])
              for i in range(4)]
    header = ("cell", "s0", "mean±sd", "verdict")
    print("| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(header)) + " |")
    print("|-" + "-|-".join("-" * w for w in widths) + "-|")
    for r in rows:
        print("| " + " | ".join(r[i].ljust(widths[i]) for i in range(4)) + " |")


if __name__ == "__main__":
    sys.exit(main())
