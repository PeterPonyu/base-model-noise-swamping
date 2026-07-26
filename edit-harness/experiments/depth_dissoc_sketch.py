"""depth_dissoc_sketch.py — T1.1 depth-dissociation E0, DESCRIPTIVE (CPU) arm.

=============================================================================
AUTHORING PASS. A separate hostile-review lane gates any paper claim built on
this. This is the Llama-only DESCRIPTIVE SKETCH the prereg explicitly permits
now (docs/plans/PREREG-T11-DEPTH-DISSOCIATION-E0-20260713.md). The
pre-registered PASS gate CANNOT fire here: it requires a SECOND architecture
with raw-K at >=3 depths, and no such depth profile exists in cache (Qwen-1.5B
has raw K at L14 only — prereg s2.4). Every code path below says so, and the
final stdout line states it verbatim.
=============================================================================

WHAT THIS DOES (prereg s4 numbered protocol, CPU-only, read-only inputs):
  1. Collateral profile C(L) from the PINNED CANONICAL G1_stability_L*_v2.json
     (field aggregate.within_probe_mean_across_seeds).
  2. Merge profile M(L) from results/merging/RG_operating_curve_table{,_L8,_L14}
     .json: per-layer mean of partial_rho_geom over the c2-coherent qualifying
     g in {2,3} cells across seeds, pre-committed g=2-then-g=3 (prereg s2.2).
  3. Dissociation signal D(L) = z_grid[M] - z_grid[C] on the shared usable grid
     {L8,L12,L14} (L10 excluded — no RG L10 run, no raw-K bank), plus the
     review-mandated fragility guard: raw dM/dC signs beside the z-based dD sign.
  4. The three admitted key-space statistics on the raw edit-key banks K[N,d]
     (prereg s1.2): PR (pr_frac), anisotropy (mean_cos / mean_abs_cos), kurtosis
     proxy kappa. PR + anisotropy REUSE analyze_aniso.py (same formulas as the
     prereg); kurtosis is implemented here per the prereg (see KURTOSIS note).
  5. Per-statistic monotone-tracking test + sign-test p + Holm-Bonferroni over
     the three (arch-1 only; informational — no gate can fire).
  6. Emit results/depth_dissoc/sketch_report.json + a stdout table.

KURTOSIS divergence note: the prereg proposes kappa as a ~10-line --emit_kurtosis
flag added to analyze_aniso.py. The T1.1 task binds this deliverable to ONE new
file, so kappa is implemented locally here (pooled excess kurtosis of the
column-standardized centered keys, exactly the prereg s1.2 H3 formula). PR and
anisotropy are IMPORTED from analyze_aniso (formulas verified identical to the
prereg), so no formula is duplicated. This divergence is recorded in the JSON.

Conventions match experiments/: fixed inputs, numpy only, 4-dp rounding,
deterministic. Sign convention per statistic is pre-committed in the prereg
(PR: sign(dD) vs sign(-d pr_frac); A: vs sign(+dA); kappa: vs sign(+dkappa)).

Usage:
  python experiments/depth_dissoc_sketch.py            # real descriptive run
  python experiments/depth_dissoc_sketch.py --selftest # synthetic, no real IO
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
# Reuse the exact prereg-referenced primitives — do NOT reimplement.
from analyze_aniso import (  # noqa: E402
    participation_ratio,   # PR = (sum l)^2 / sum l^2         (prereg H1)
    pairwise_cos_stats,    # mean_cos / mean_abs_cos on raw K  (prereg H2)
    _gram_eigs,            # nonzero eigs of centered Gram      (prereg s1.2)
    load_bank,             # loud-fail loader; requires raw K
)

HARNESS = os.path.dirname(HERE)                        # edit-harness/
RESULTS = os.path.join(HARNESS, "results")
VECDIR = os.path.join(RESULTS, "vectors")
MERGEDIR = os.path.join(RESULTS, "merging")
OUTDIR = os.path.join(RESULTS, "depth_dissoc")
OUTJSON = os.path.join(OUTDIR, "sketch_report.json")

# Shared usable grid: layers with C AND M AND a raw-K bank all present.
# L10 has C but NO RG-merge run and NO raw-K bank -> excluded (prereg s2.2/s2.4).
GRID_LAYERS = [8, 12, 14]

# Pre-committed sign conventions (prereg s1.2). Each maps a statistic key to the
# signed delta whose sign should equal sign(dD) at every consecutive pair.
#   PR   : track sign(dD) vs sign(-d pr_frac)     -> multiplier -1 on pr_frac
#   A    : track sign(dD) vs sign(+d mean_cos)    -> multiplier +1
#   kappa: track sign(dD) vs sign(+d kappa)       -> multiplier +1
STAT_CONVENTIONS = [
    ("pr_frac", -1, "H1 participation-ratio fraction (primary): sign(-d pr_frac)"),
    ("mean_cos", +1, "H2 key anisotropy (secondary): sign(+d mean_cos)"),
    ("kappa", +1, "H3 kurtosis proxy (tertiary): sign(+d kappa)"),
]


# ------------------------------------------------------------------- statistics
def kurtosis_proxy(K):
    """Prereg s1.2 H3: pooled excess kurtosis of the column-standardized centered
    keys. Center K by column mean, standardize each coordinate j by its column
    std sigma_j over the N edits, pool all N*d z-values, kappa = mean(z^4) - 3.
    Columns with sigma_j == 0 carry no signal and are dropped (no 0/0)."""
    Kc = K - K.mean(axis=0, keepdims=True)
    sig = Kc.std(axis=0)                       # per-coordinate std over N edits
    keep = sig > 0
    Z = Kc[:, keep] / sig[keep]
    z2 = Z * Z
    kappa = float((z2 * z2).mean() - 3.0)
    return kappa, int(keep.sum()), int((~keep).sum())


def stats_one_bank(npz_path):
    """All three admitted statistics for one raw-K bank. PR + anisotropy are the
    imported analyze_aniso primitives; kurtosis is the local H3 proxy."""
    K, meta = load_bank(npz_path)
    N, d = K.shape
    eig_cen = _gram_eigs(K - K.mean(axis=0))          # centered-Gram spectrum
    pr = participation_ratio(eig_cen)                 # effective # directions
    pr_frac = pr / min(N - 1, d)                       # depth-comparable fraction
    cos = pairwise_cos_stats(K)                        # raw-key pairwise cosines
    kappa, n_kept, n_dropped = kurtosis_proxy(K)
    return {
        "npz": os.path.basename(npz_path),
        "layer": meta["layer"], "seed": meta["seed"],
        "N": N, "d": d, "vectors_valid": meta["vectors_valid"],
        "pr": round(pr, 4), "pr_frac": round(pr_frac, 6),
        "mean_cos": round(cos["mean_cos"], 6),
        "mean_abs_cos": round(cos["mean_abs_cos"], 6),
        "kappa": round(kappa, 6),
        "kappa_cols_kept": n_kept, "kappa_cols_dropped": n_dropped,
    }


def _agg_over_seeds(rows, key):
    """mean +/- std (ddof=0) of `key` over per-seed stat rows for one layer."""
    vals = np.array([r[key] for r in rows], dtype=float)
    return {"mean": round(float(vals.mean()), 6),
            "std": round(float(vals.std(ddof=0)), 6),
            "n_seeds": int(vals.size),
            "seeds": sorted(int(r["seed"]) for r in rows)}


# --------------------------------------------------------------- collateral C(L)
def read_collateral():
    """C(L) = aggregate.within_probe_mean_across_seeds from the pinned canonical
    G1_stability_L{L}_v2.json, with the std / perm-p uncertainty band."""
    out = {}
    for L in GRID_LAYERS:
        f = os.path.join(RESULTS, f"G1_stability_L{L}_v2.json")
        agg = json.load(open(f))["aggregate"]
        out[L] = {
            "C": float(agg["within_probe_mean_across_seeds"]),
            "std": float(agg.get("within_probe_std_across_seeds", float("nan"))),
            "max_within_probe_perm_p": agg.get("max_within_probe_perm_p"),
            "n_seeds": int(agg.get("n_seeds", 0)),
            "source": os.path.basename(f),
        }
    return out


# -------------------------------------------------------------------- merge M(L)
def _cell_qualifies(c):
    """A merge cell qualifies iff it is c2-coherent, non-negligible, not saturated
    (prereg s2.2 'c2-coherent qualifying cells')."""
    return bool(c.get("c2_coherent")) and bool(c.get("non_negligible")) \
        and not bool(c.get("saturated"))


def _select_merge_cells(cells, seeds):
    """Prereg s2.2 g-selection RULE, extracted so depth_dissoc_gate.py (the arch-2
    gate extension) can reuse it verbatim instead of re-deriving it — a single
    place this can't drift out of sync (review MINOR-1, 2026-07-14). Prefer g=2;
    it is usable for the layer iff EVERY present seed's g2 cell qualifies;
    otherwise fall back to the qualifying g3 cells (dropping seeds whose g3 also
    fails). Returns (chosen_g, chosen) where chosen is a list of
    (seed, cell_key, cell_dict) triples — identical shape/semantics to the
    inline block this replaces, so read_merge()'s output is unchanged."""
    def collect(g):
        got = []
        for s in seeds:
            k = f"g{g}_s{s}"
            if k in cells and _cell_qualifies(cells[k]):
                got.append((s, k, cells[k]))
        return got

    g2 = collect(2)
    g2_all_present = all((f"g2_s{s}" in cells) for s in seeds)
    # g=2 usable iff every present-seed g2 cell qualifies.
    if g2_all_present and len(g2) == len(seeds):
        return 2, g2
    return 3, collect(3)


def read_merge():
    """M(L) per layer, pre-committed g=2-where-c2-coherent-else-g=3 (prereg s2.2).
    Rule: prefer g=2. g=2 is used for the layer iff EVERY present seed's g2 cell
    qualifies; otherwise fall back to the qualifying g3 cells (dropping seeds
    whose g3 also fails). M = mean(partial_rho_geom) over the selected cells;
    own-magnitude partial and rho_I_cos_drop retained as guards."""
    files = {8: "RG_operating_curve_table_L8.json",
             12: "RG_operating_curve_table.json",
             14: "RG_operating_curve_table_L14.json"}
    out = {}
    for L, fn in files.items():
        d = json.load(open(os.path.join(MERGEDIR, fn)))
        cells = d["cells"]
        seeds = [int(s) for s in d.get("seeds", [0, 1, 2])]

        chosen_g, chosen = _select_merge_cells(cells, seeds)

        prho = np.array([c["partial_rho_geom"] for _, _, c in chosen], dtype=float)
        own = np.array([c["partial_rho_geom_ownmag"] for _, _, c in chosen], dtype=float)
        icos = np.array([c["rho_I_cos_drop"] for _, _, c in chosen], dtype=float)
        if prho.size == 0:  # review M-c: degenerate merge profile must not zero the tracking silently
            sys.stderr.write(
                f"[depth_dissoc] WARNING: layer L{L} has NO qualifying merge cell "
                f"(g2 all-fail and g3 empty) -> M=NaN; ΔD signs touching L{L} are non-informative.\n")
        out[L] = {
            "M": round(float(prho.mean()), 6) if prho.size else float("nan"),
            "M_std": round(float(prho.std(ddof=0)), 6) if prho.size else float("nan"),
            "M_ownmag_mean": round(float(own.mean()), 6) if own.size else float("nan"),
            "rho_I_cos_drop_mean": round(float(icos.mean()), 6) if icos.size else float("nan"),
            "chosen_g": chosen_g,
            "cells_used": [k for _, k, _ in chosen],
            "n_cells": int(prho.size),
            "source": os.path.basename(files[L]),
            "layer_field": d.get("layer"),
        }
    return out


# ---------------------------------------------------------- dissociation signal
def _z(vals):
    """z-score across the grid (ddof=0). Sign of dD is invariant to the ddof
    choice — see prereg s1.3 fragility note — so ddof=0 is used consistently."""
    v = np.asarray(vals, dtype=float)
    sd = v.std(ddof=0)
    if sd == 0:
        return np.zeros_like(v), 0.0
    return (v - v.mean()) / sd, float(sd)


def _sign(x, eps=1e-12):
    if x > eps:
        return 1
    if x < -eps:
        return -1
    return 0


def build_dissociation(layers, C_by_L, M_by_L):
    """D(L) = z_grid[M] - z_grid[C]; per-pair dD sign + the review-mandated raw
    dM/dC concordance guard (prereg s1.3)."""
    C = [C_by_L[L] for L in layers]
    M = [M_by_L[L] for L in layers]
    zC, sC = _z(C)
    zM, sM = _z(M)
    D = (zM - zC)
    profile = [{"layer": L, "C": round(C[i], 6), "M": round(M[i], 6),
                "zC": round(float(zC[i]), 6), "zM": round(float(zM[i]), 6),
                "D": round(float(D[i]), 6)} for i, L in enumerate(layers)]
    pairs = []
    d_profile_ambiguous = False
    for i in range(len(layers) - 1):
        dM = M[i + 1] - M[i]
        dC = C[i + 1] - C[i]
        dD = float(D[i + 1] - D[i])
        s_dD, s_dM, s_dC = _sign(dD), _sign(dM), _sign(dC)
        # Raw picture is UNAMBIGUOUS only when M and C move oppositely
        # (D up iff M rises & C falls). Same-sign moves make sign(dD) depend on
        # the 3-sample std ratio -> flagged AMBIGUOUS per prereg s1.3.
        raw_unambiguous = (s_dM != 0 and s_dC != 0 and s_dM == -s_dC)
        if not raw_unambiguous:
            d_profile_ambiguous = True
        pairs.append({
            "pair": f"L{layers[i]}->L{layers[i+1]}",
            "dM": round(dM, 6), "dC": round(dC, 6), "dD": round(dD, 6),
            "sign_dM": s_dM, "sign_dC": s_dC, "sign_dD": s_dD,
            "raw_unambiguous": raw_unambiguous,
        })
    return {"profile": profile, "sigma_M": round(sM, 6), "sigma_C": round(sC, 6),
            "pairs": pairs, "d_profile_ambiguous": d_profile_ambiguous}


# ------------------------------------------------------ per-statistic tracking
def tracking_test(layers, dissoc_pairs, stat_by_L):
    """Per admitted statistic: match sign(dD) against sign(mult * dS) at each
    consecutive pair; monotone-tracking = concordant at ALL pairs; sign-test
    p = 2^-(K-1) when monotone (prereg s3.1). Then Holm-Bonferroni over three."""
    K = len(layers)
    n_pairs = K - 1
    per_stat = []
    for key, mult, desc in STAT_CONVENTIONS:
        vals = [stat_by_L[L][key] for L in layers]
        pair_rows, n_conc = [], 0
        for i, dp in enumerate(dissoc_pairs):
            dS = vals[i + 1] - vals[i]
            s_expected = _sign(mult * dS)
            s_dD = dp["sign_dD"]
            conc = (s_expected != 0 and s_expected == s_dD)
            n_conc += int(conc)
            pair_rows.append({"pair": dp["pair"], "dS": round(float(dS), 6),
                              "sign_expected": s_expected, "sign_dD": s_dD,
                              "concordant": conc})
        tracks = (n_conc == n_pairs)
        per_stat.append({
            "stat": key, "convention": desc,
            "values_on_grid": [round(float(v), 6) for v in vals],
            "pairs": pair_rows, "n_concordant": n_conc, "n_pairs": n_pairs,
            "monotone_tracks_arch1": tracks,
            "sign_test_p": (2.0 ** (-(n_pairs))) if tracks else None,
        })
    # Holm-Bonferroni over the three p-values (tracking->0.25 at K=3; else fail).
    m = len(per_stat)
    scored = [(s["sign_test_p"] if s["sign_test_p"] is not None else 1.0, s["stat"])
              for s in per_stat]
    order = sorted(range(m), key=lambda i: scored[i][0])
    holm = {}
    for rank, idx in enumerate(order):
        p, name = scored[idx]
        thr = 0.05 / (m - rank)
        holm[name] = {"p": p, "holm_threshold": round(thr, 6),
                      "passes_holm": bool(p <= thr)}
    for s in per_stat:
        s["holm"] = holm[s["stat"]]
    # Priority-ordered arch-1 winner candidate (PR->A->kappa), if any tracks.
    winner = next((s["stat"] for s in per_stat if s["monotone_tracks_arch1"]), None)
    return per_stat, winner


# ------------------------------------------------------------------- real run
def run_real():
    os.makedirs(OUTDIR, exist_ok=True)
    layers = GRID_LAYERS

    coll = read_collateral()
    merge = read_merge()
    C_by_L = {L: coll[L]["C"] for L in layers}
    M_by_L = {L: merge[L]["M"] for L in layers}

    # Statistic banks per layer (single-seed at L8/L12; 3-seed at L14).
    bank_glob = {
        8:  ["vectors_qv_llama1b_rome_cf_L8_s0.npz"],
        12: ["vectors_qv_llama1b_rome_cf_L12_s0.npz"],
        14: ["vectors_qv_llama1b_rome_cf_L14_s0.npz",
             "vectors_qv_llama1b_rome_cf_L14_s1.npz",
             "vectors_qv_llama1b_rome_cf_L14_s2.npz"],
    }
    stat_rows_by_L, stat_by_L = {}, {}
    for L in layers:
        rows = [stats_one_bank(os.path.join(VECDIR, b)) for b in bank_glob[L]]
        stat_rows_by_L[L] = rows
        stat_by_L[L] = {
            "pr_frac": _agg_over_seeds(rows, "pr_frac")["mean"],
            "mean_cos": _agg_over_seeds(rows, "mean_cos")["mean"],
            "mean_abs_cos": _agg_over_seeds(rows, "mean_abs_cos")["mean"],
            "kappa": _agg_over_seeds(rows, "kappa")["mean"],
            "bands": {k: _agg_over_seeds(rows, k)
                      for k in ("pr_frac", "mean_cos", "mean_abs_cos", "kappa")},
            "single_seed": len(rows) == 1,
        }

    dissoc = build_dissociation(layers, C_by_L, M_by_L)
    per_stat, winner = tracking_test(layers, dissoc["pairs"], stat_by_L)

    report = {
        "experiment": "T1.1_depth_dissociation_E0_descriptive_sketch",
        "prereg": "docs/plans/PREREG-T11-DEPTH-DISSOCIATION-E0-20260713.md",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "arch": "Llama-3.2-1B (ROME, CounterFact)",
        "grid_layers": layers,
        "grid_note": "L10 excluded: has C but no RG-merge run and no raw-K bank "
                     "(prereg s2.2/s2.4). Shared usable grid = {L8,L12,L14}.",
        "collateral_C": coll,
        "merge_M": merge,
        "dissociation": dissoc,
        "statistics_per_layer": stat_by_L,
        "statistics_per_seed": stat_rows_by_L,
        "tracking": per_stat,
        "arch1_winner_candidate": winner,
        "verdict": "DESCRIPTIVE_ONLY_GATE_NOT_DECIDABLE",
        "gate_status": {
            "pass_gate_decidable": False,
            "reason": "PASS requires a 2nd architecture with raw-K at >=3 depths; "
                      "cache has Qwen-1.5B raw-K at L14 only -> no arch-2 depth "
                      "profile exists (prereg s2.4). A one-architecture 3-point "
                      "match is KILL-for-gate-purposes (prereg s3.3 AMBIGUOUS "
                      "guard); reported descriptively only.",
            "d_profile_ambiguous": dissoc["d_profile_ambiguous"],
        },
        "binding_wording": [
            "z-grid-relative reading only: D compares depth-SHAPE, not absolute "
            "partial-rho levels between the two families.",
            "single-seed statistic point estimates at L8 and L12 (raw-K 3-seed "
            "only at L14) -> arch-1 tracking rests on point estimates there.",
            "merge M(L) carries two boundaries per house rule (geometry-valid "
            "g<=5; gradated g=10); this sketch uses only qualifying g in {2,3}.",
            "kurtosis kappa is implemented in-file (not the analyze_aniso "
            "--emit_kurtosis flag) per the one-file task constraint; PR and "
            "anisotropy are imported from analyze_aniso (identical formulas).",
        ],
    }
    with open(OUTJSON, "w") as fh:
        json.dump(report, fh, indent=2)
    print_table(report)
    return report


def print_table(report):
    layers = report["grid_layers"]
    coll, merge = report["collateral_C"], report["merge_M"]
    sbl = report["statistics_per_layer"]
    dpairs = {p["pair"]: p for p in report["dissociation"]["pairs"]}
    Dprof = {p["layer"]: p for p in report["dissociation"]["profile"]}

    print(f"\n=== T1.1 depth-dissociation E0 — DESCRIPTIVE sketch ({report['arch']}) ===")
    print(f"grid = {layers}  ({report['grid_note']})\n")
    hdr = (f"{'L':>3} {'C':>7} {'M':>7} {'D':>7} "
           f"{'pr_frac':>9} {'aniso':>8} {'|aniso|':>8} {'kurt':>10} {'seeds':>6}")
    print(hdr)
    print("-" * len(hdr))
    for L in layers:
        st = sbl[L]
        nseed = st["bands"]["kappa"]["n_seeds"]
        print(f"L{L:>2} {coll[L]['C']:>7.3f} {merge[L]['M']:>7.3f} "
              f"{Dprof[L]['D']:>7.3f} {st['pr_frac']:>9.4f} {st['mean_cos']:>8.4f} "
              f"{st['mean_abs_cos']:>8.4f} {st['kappa']:>10.4f} {nseed:>6}")

    print("\n-- dissociation dD sign + raw dM/dC concordance guard (prereg s1.3) --")
    for p in report["dissociation"]["pairs"]:
        tag = "unambiguous" if p["raw_unambiguous"] else "AMBIGUOUS(same-sign dM,dC)"
        print(f"  {p['pair']:>12}: dD={p['dD']:+.3f} (sign {p['sign_dD']:+d})  "
              f"dM={p['dM']:+.3f} dC={p['dC']:+.3f}  [{tag}]")

    print("\n-- per-statistic arch-1 monotone-tracking (informational; no gate) --")
    for s in report["tracking"]:
        p = s["sign_test_p"]
        pstr = f"p={p:.3f}" if p is not None else "p=n/a"
        print(f"  {s['stat']:>9}: {s['n_concordant']}/{s['n_pairs']} concordant  "
              f"tracks={s['monotone_tracks_arch1']}  {pstr}  "
              f"holm_pass={s['holm']['passes_holm']}  [{s['convention']}]")
    print(f"\n  arch-1 winner candidate (PR->A->kappa priority): "
          f"{report['arch1_winner_candidate']}")
    print(f"  D-profile ambiguous (raw-sign guard): "
          f"{report['dissociation']['d_profile_ambiguous']}")
    print(f"\n  report -> {os.path.relpath(OUTJSON, HARNESS)}")
    print("DESCRIPTIVE ONLY — PASS gate not decidable "
          "(arch-2 depth profile absent; see prereg §2.4)")


# --------------------------------------------------------------------- selftest
def _synth_bank(n, d, aniso, rng):
    """Synthetic key bank with a PLANTED anisotropy level in [0,1]. aniso=0 is
    near-isotropic; aniso->1 collapses keys onto one shared direction with a
    heavy-tailed coordinate mixed in. Returns K[n,d]."""
    iso = rng.standard_normal((n, d))
    shared = rng.standard_normal(d)
    shared /= np.linalg.norm(shared)
    # same-sign projection => a real directional CONE (raises signed mean_cos),
    # not a bipolar +/- axis (which would average the signed cosine to ~0).
    coeff = np.abs(rng.standard_normal((n, 1))) + 0.5
    cone = coeff * shared[None, :]                 # rank-1 cone component
    # heavy-tailed spike on a few coordinates -> lifts pooled kurtosis
    spike = np.zeros((n, d))
    hot = rng.integers(0, d, size=8)
    spike[:, hot] = rng.standard_t(3, size=(n, hot.size)) * 3.0
    return (1 - aniso) * iso + aniso * (4.0 * cone + spike)


def run_selftest():
    """Prove the three statistics move in the pre-committed directions on planted
    data. Reads NO real files. Anisotropy gradient over 3 synthetic 'layers':
    expect pr_frac DOWN, mean_cos UP, kappa UP as anisotropy rises."""
    rng = np.random.default_rng(20260713)
    n, d = 200, 512
    anisos = [0.05, 0.35, 0.75]
    rows = []
    for lv, a in enumerate(anisos):
        K = _synth_bank(n, d, a, rng)
        eig = _gram_eigs(K - K.mean(axis=0))
        pr_frac = participation_ratio(eig) / min(n - 1, d)
        cos = pairwise_cos_stats(K)
        kappa, _, _ = kurtosis_proxy(K)
        rows.append({"level": lv, "aniso": a, "pr_frac": pr_frac,
                     "mean_cos": cos["mean_cos"], "kappa": kappa})

    print("\n=== SELFTEST — planted anisotropy gradient (no real IO) ===")
    print(f"{'lvl':>3} {'aniso':>6} {'pr_frac':>9} {'mean_cos':>9} {'kappa':>10}")
    for r in rows:
        print(f"{r['level']:>3} {r['aniso']:>6.2f} {r['pr_frac']:>9.4f} "
              f"{r['mean_cos']:>9.4f} {r['kappa']:>10.4f}")

    def strictly(key, want):
        seq = [r[key] for r in rows]
        diffs = np.diff(seq)
        ok = bool(np.all(diffs > 0)) if want == "up" else bool(np.all(diffs < 0))
        return ok, seq

    checks = [("pr_frac", "down", "-d pr_frac (PR convention)"),
              ("mean_cos", "up", "+d mean_cos (A convention)"),
              ("kappa", "up", "+d kappa (kurtosis convention)")]
    all_ok = True
    print("\n-- expected monotone directions --")
    for key, want, desc in checks:
        ok, seq = strictly(key, want)
        all_ok &= ok
        print(f"  {key:>9} strictly {want:>4}: {'PASS' if ok else 'FAIL'}  "
              f"({desc})  values={[round(v,4) for v in seq]}")
    print(f"\nSELFTEST {'PASS' if all_ok else 'FAIL'} — "
          f"statistics move in the pre-committed directions."
          if all_ok else
          f"\nSELFTEST FAIL — a statistic did not move as planted.")
    return 0 if all_ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true",
                    help="run the synthetic-data self-test (no real file IO)")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(run_selftest())
    run_real()


if __name__ == "__main__":
    main()
