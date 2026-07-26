#!/usr/bin/env python3
"""Paired hierarchical bootstrap on editor ordering of conditional edit-efficacy survival.

Paper B (quantization survival) claims that MEMIT and AlphaEdit are "no less robust"
than ROME, and reports that ordering as FAILING on the Qwen NF4 arms. That readout
compares three point estimates with no uncertainty attached. This script attaches it.

Estimand per (arch, arm, editor):
    conditional survival = mean_seeds( mean_i( edit_ok_arm[i] | edit_ok_fp32[i] ) )
matching the point policy of the canonical repair artefact
(`quant_survival_repair_v1.json`, point_kind = mean_of_per_seed_spearmans).

The three editors are run on the SAME edit set and the SAME probe set within each
(model, seed) -- verified by byte-identical COS matrices -- so editor contrasts can be
bootstrapped PAIRED: each resample draws one set of edit indices and scores every
editor on it. Pairing removes the edit-sampling variance that is common to the editors
and is what makes the contrast, rather than each editor's level, the unit of inference.

Bootstrap: hierarchical, seeds-then-edits, matching the paper's existing methodology.
    Stage 1: resample the n_seeds seeds with replacement.
    Stage 2: within each drawn seed, resample the n_edits edit indices with replacement;
             the SAME drawn indices are applied to all three editors.
Reported: percentile CI95 of the paired difference and a two-sided bootstrap p-value.

Usage:  python paperb_editor_ordering_bootstrap.py [--n_boot 10000] [--out PATH]
CPU only; runtime ~1 min.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import pathlib
import sys
from datetime import datetime, timezone

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
RESULTS = REPO / "edit-harness" / "results" / "quant_survival"

MODELS = {
    "llama1b": {"layer": 12, "display": "Llama-3.2-1B L12"},
    "llama3b": {"layer": 24, "display": "Llama-3.2-3B L24"},
    "qwen15b": {"layer": 21, "display": "Qwen-2.5-1.5B L21"},
}
EDITORS = ["rome", "memit", "alpha"]
EDITOR_DISPLAY = {"rome": "ROME", "memit": "MEMIT", "alpha": "AlphaEdit"}
ARMS = ["nf4dq_edited_layer", "nf4dq_full_model", "int8_edited_layer", "int8_full_model"]
SEEDS = [0, 1, 2]


def load_cell(slug: str, editor: str, seed: int) -> dict:
    """Return the boolean success vectors for one (model, editor, seed) cell."""
    layer = MODELS[slug]["layer"]
    path = RESULTS / f"{slug}_{editor}_L{layer}_s{seed}" / "QS_phase1_raw.npz"
    with np.load(path) as z:
        out = {"ok_fp32": z["edit_ok_fp32"].astype(bool)}
        for arm in ARMS:
            out[arm] = z[f"esr__{arm}"].astype(bool)
        out["cos_hash"] = hashlib.sha256(z["COS"].tobytes()).hexdigest()[:16]
    return out


def conditional_survival(ok_fp32: np.ndarray, ok_arm: np.ndarray) -> float:
    """mean(ok_arm | ok_fp32); NaN when the conditioning set is empty in this resample."""
    denom = ok_fp32.sum()
    if denom == 0:
        return np.nan
    return float(ok_arm[ok_fp32].sum() / denom)


def point_estimate(cells: dict, slug: str, editor: str, arm: str) -> float:
    """Mean over seeds of the per-seed conditional survival."""
    per_seed = [
        conditional_survival(cells[(slug, editor, s)]["ok_fp32"], cells[(slug, editor, s)][arm])
        for s in SEEDS
    ]
    return float(np.nanmean(per_seed))


def paired_bootstrap(cells: dict, slug: str, arm: str, n_boot: int, rng_seed: int) -> dict:
    """Hierarchical paired bootstrap of all pairwise editor contrasts for one (arch, arm)."""
    rng = np.random.default_rng(rng_seed)
    n_edits = cells[(slug, "rome", 0)]["ok_fp32"].shape[0]
    pairs = list(itertools.combinations(EDITORS, 2))
    draws = {p: np.full(n_boot, np.nan) for p in pairs}
    levels = {e: np.full(n_boot, np.nan) for e in EDITORS}

    for b in range(n_boot):
        seed_draw = rng.choice(SEEDS, size=len(SEEDS), replace=True)
        # One edit-index resample per drawn seed, shared across all editors (the pairing).
        idx_draw = [rng.integers(0, n_edits, size=n_edits) for _ in seed_draw]
        per_editor = {}
        for editor in EDITORS:
            vals = []
            for s, idx in zip(seed_draw, idx_draw):
                c = cells[(slug, editor, s)]
                vals.append(conditional_survival(c["ok_fp32"][idx], c[arm][idx]))
            per_editor[editor] = np.nanmean(vals)
        for e in EDITORS:
            levels[e][b] = per_editor[e]
        for a, bb in pairs:
            draws[(a, bb)][b] = per_editor[a] - per_editor[bb]

    out = {"levels": {}, "contrasts": {}}
    for e in EDITORS:
        d = levels[e][np.isfinite(levels[e])]
        out["levels"][e] = {
            "point": point_estimate(cells, slug, e, arm),
            "boot_dist_mean": float(d.mean()),
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "boot_n_finite": int(d.size),
        }
    for a, bb in pairs:
        d = draws[(a, bb)][np.isfinite(draws[(a, bb)])]
        obs = point_estimate(cells, slug, a, arm) - point_estimate(cells, slug, bb, arm)
        lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
        # Two-sided bootstrap p: doubled smaller tail. Both tails count the mass exactly at
        # zero, so a degenerate all-zero contrast returns p = 1 (no evidence of a difference)
        # rather than the p = 0 an exclusive comparison would produce.
        frac_le = float((d <= 0).mean())
        frac_ge = float((d >= 0).mean())
        p_two = min(1.0, 2.0 * min(frac_le, frac_ge))
        p_two = max(p_two, 1.0 / d.size)
        excludes_zero = (lo > 0.0) or (hi < 0.0)
        out["contrasts"][f"{a}_minus_{bb}"] = {
            "observed_diff": obs,
            "boot_dist_mean": float(d.mean()),
            "ci95": [lo, hi],
            "p_two_sided": p_two,
            "boot_n_finite": int(d.size),
            "distinguishable_at_95": bool(excludes_zero),
            "verdict": (
                "distinguishable" if excludes_zero else "not statistically distinguishable"
            ),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--rng_seed", type=int, default=12345)
    ap.add_argument(
        "--out",
        type=str,
        default=str(RESULTS / "aggregate" / "editor_ordering_bootstrap_20260726.json"),
    )
    args = ap.parse_args()

    cells = {}
    for slug in MODELS:
        for editor in EDITORS:
            for s in SEEDS:
                cells[(slug, editor, s)] = load_cell(slug, editor, s)

    # Pairing precondition: identical edit/probe geometry across editors within (model, seed).
    pairing = {}
    for slug in MODELS:
        for s in SEEDS:
            hashes = {cells[(slug, e, s)]["cos_hash"] for e in EDITORS}
            pairing[f"{slug}_s{s}"] = {
                "cos_hash": sorted(hashes)[0],
                "identical_across_editors": len(hashes) == 1,
            }
    if not all(v["identical_across_editors"] for v in pairing.values()):
        print("FATAL: edit/probe sets are not shared across editors; pairing is invalid.")
        return 1

    results = {}
    for slug in MODELS:
        for arm in ARMS:
            key = f"{slug}__{arm}"
            results[key] = paired_bootstrap(cells, slug, arm, args.n_boot, args.rng_seed)
            print(f"[{key}]")
            for e in EDITORS:
                lv = results[key]["levels"][e]
                print(f"    {EDITOR_DISPLAY[e]:10} {lv['point']:.4f}")
            for name, c in results[key]["contrasts"].items():
                print(
                    f"    {name:20} d={c['observed_diff']:+.4f} "
                    f"CI95 [{c['ci95'][0]:+.4f},{c['ci95'][1]:+.4f}] "
                    f"p={c['p_two_sided']:.4f}  {c['verdict']}"
                )

    n_disting = sum(
        1 for r in results.values() for c in r["contrasts"].values() if c["distinguishable_at_95"]
    )
    n_total = sum(len(r["contrasts"]) for r in results.values())

    payload = {
        "module": "paperb_editor_ordering_bootstrap",
        "version": "1.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "estimand": (
            "conditional survival = mean_seeds( mean_edits( edit_ok_arm | edit_ok_fp32 ) ); "
            "point = mean of per-seed values, matching quant_survival_repair_v1 point policy"
        ),
        "bootstrap": {
            "kind": "hierarchical paired, seeds-then-edits",
            "pairing": (
                "one edit-index resample per drawn seed, applied identically to all three "
                "editors; valid because the edit and probe sets are byte-identical across "
                "editors within each (model, seed)"
            ),
            "n_boot": args.n_boot,
            "rng_seed": args.rng_seed,
            "ci": "percentile 95",
            "p_value": "two-sided; 2 x min(P(d<=0), P(d>0)), floored at 1/n_boot",
        },
        "pairing_audit": pairing,
        "grid": {
            "models": list(MODELS),
            "editors": EDITORS,
            "arms": ARMS,
            "seeds": SEEDS,
            "n_edits": int(cells[("llama1b", "rome", 0)]["ok_fp32"].shape[0]),
        },
        "summary": {
            "n_contrasts": n_total,
            "n_distinguishable_at_95": n_disting,
            "n_not_distinguishable": n_total - n_disting,
        },
        "results": results,
        "numpy_version": np.__version__,
    }
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n{n_disting}/{n_total} editor contrasts distinguishable at 95%")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
