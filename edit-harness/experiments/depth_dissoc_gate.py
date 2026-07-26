"""depth_dissoc_gate.py — T1.1 depth-dissociation E0, GATE-GRADE (arch-2) extension.

=============================================================================
AUTHORING PASS. A separate hostile-review lane gates any paper claim built on
this. This EXTENDS depth_dissoc_sketch.py (Llama-only descriptive arm, left
UNCHANGED — its own CLI entrypoint and JSON output are untouched) to evaluate
the pre-registered PASS gate from
docs/plans/PREREG-T11-DEPTH-DISSOCIATION-E0-20260713.md s2.4/s3 against a
SECOND architecture, once its raw-K vector banks exist (arch-1 = Llama-3.2-1B,
arch-2 = Qwen2.5-1.5B by default, but both are CLI-parameterized).

Every statistic FORMULA (PR, anisotropy, kurtosis) and the arch-1 machinery
(collateral-vs-merge dissociation D(L), the tracking test, Holm-Bonferroni) are
IMPORTED from depth_dissoc_sketch.py / analyze_aniso.py and reused VERBATIM —
nothing here reimplements a statistic. This file adds only: (a) path-
parameterized readers so arch-1's inputs are no longer tied to this machine's
absolute layout, (b) arch-2 raw-K discovery + stats, (c) an arch-2 REPLICATION
TARGET, and (d) the PASS/KILL/UNDECIDABLE gate evaluator (prereg s3.3).

=============================================================================
GATE-INTERPRETATION NOTE (read before trusting any PASS/KILL from this file)
=============================================================================
The prereg (s1.1) defines the arch-1 target as a TWO-FAMILY dissociation
  D(L) = z_grid[ M(L) ] - z_grid[ C(L) ]
where M(L) is a merge-interference (RG) measurement and C(L) is the within-
probe collateral-damage rho (the G1 gate). This file supports the LITERAL
two-family gate for arch-2 whenever the data exists, and falls back to a
single-sided proxy when it does not:

  * LITERAL gate (bare PASS/KILL/AMBIGUOUS). Fired when arch-2 has, at >=3
    OVERLAPPING depths, all three of: collateral C(L) (matrices auto-discovery
    or --arch2_collateral_json), TAG-NAMESPACED merge M(L) read from
    RG_operating_curve_table_<tag>_L<L>.json (e.g. *_qwen15b_L21.json — note
    these carry the arch tag, unlike arch-1's untagged names), AND a raw-K stat
    bank. Then D_arch2(L)=z[M]-z[|C|] is built with the SAME build_dissociation
    machinery as arch-1 and the prereg s3.3 replication gate is evaluated; a
    bare "PASS" is legitimate ONLY here.
  * PROXY fallback (PASS_PROXY_TARGET/KILL/KILL_FOR_GATE_PURPOSES). When no
    tag-namespaced merge / stat-bank overlap of >=3 exists (e.g. the authoring-
    time cache: qwen15b matrices L14 only, merge L21 only, bank L14 only), the
    arch-2 test uses the single-sided proxy T_arch2(L)=z_grid[|C_arch2(L)|] —
    arch-2's collateral-dominance MAGNITUDE alone. A PASS_PROXY_TARGET is a
    WEAKER claim than the literal gate and is tagged
    target_kind="single_sided_proxy_abs_C".

The absolute value on C is REQUIRED and load-bearing in BOTH regimes, not
cosmetic: Qwen-1.5B's within-probe rho is SIGN-INVERTED at measured depths
(rho_C < 0; project memory crossarch-transfer-verdict-2026-07-02) while
Llama-1B's is positive throughout its grid. Tracking the raw SIGNED rho would
conflate "how strongly geometry governs damage at this depth" with an
incidental global sign flip that has nothing to do with depth-SHAPE — exactly
the failure mode the task brief warned about ("don't assume both profiles are
positive"). The abs() move is gated by a THREE-REGIME SIGN POLICY over the
arch-2 gate-depth C (review MAJOR-1, see _evaluate_literal_gate): POSITIVE C ->
abs is a no-op, D is prereg-literal, a pass is the BARE token "PASS"; NEGATIVE
C (globally sign-inverted, e.g. Qwen) -> abs is a documented DEVIATION from
prereg s1.1's signed-C wording, a pass is the DISTINCT token
"PASS_ABS_CONVENTION" (admissible as pre-registered ONLY after a prereg
amendment — pending user decision); MIXED C -> abs is UNjustified (it conflates
opposite couplings), the gate is FORCED "AMBIGUOUS". Because z[-x]=-z[x] when
all C share a sign, the signed-C variant is ALSO computed as an auditable
shadow — but gate.winner_robust_to_signed_C is an AUDIT field only and CANNOT
gate (expected-False in the legitimate negative regime). For arch-1 abs() is a
strict no-op (its C(L) is already all-positive on the prereg grid), so
build_dissociation()/tracking_test() are called UNCHANGED and arch-1 numbers
are byte-identical to depth_dissoc_sketch.py's own run.

C_arch2(L) itself comes from (a) auto-discovery: within_probe_rhos (imported
from analyze_matrices.py — the SAME primitive the G1 gate uses, not a re-
implementation) computed directly on any local results/matrices/gate_<tag>_
rome_cf_L<layer>_s*.npz found, point-estimate only (no permutation null — out
of scope for a depth-shape target), and/or (b) an external JSON supplied via
--arch2_collateral_json, which OVERRIDES the auto value per layer when both
exist. Layers with neither source are ABSENT from the profile, never treated
as zero. On the data situation named in the task brief (qwen15b raw-K planned
at L14/L17/L21/L24, but this repo's matrices only exist at L14 — L17/L21/L24
have mechanism_dump.py S-factor summaries only, which carry no COS/damage
matrix), the DEFAULT on-box run with no external JSON is expected to return
arch2.decidable=False. That is disclosed, not hidden.

VERDICT TOKENS by regime. PROXY fallback (literal_gate_decidable=false):
"KILL_FOR_GATE_PURPOSES" when an arch is undecidable (prereg s3.3's phrase for
a one-architecture match — more data could still change it); "KILL" when both
arches are measured but no statistic replicates on the proxy target;
"PASS_PROXY_TARGET" when a statistic replicates on the proxy (weaker than the
literal gate). LITERAL gate (literal_gate_decidable=true): bare "PASS" (the
pre-registered gate fired), "KILL" (both arches measured, no statistic
replicates on the literal two-family target), or "AMBIGUOUS" (prereg s1.3
raw-sign fragility guard tripped on either D-profile). All non-PASS outcomes
mean descriptive-only publication; the tokens preserve WHY for a hostile
reviewer.

Usage:
  python experiments/depth_dissoc_gate.py                      # real gate-grade run
  python experiments/depth_dissoc_gate.py --arch2_collateral_json FILE.json
  python experiments/depth_dissoc_gate.py --selftest            # synthetic, no real IO
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# Reuse the exact statistic implementations + arch-1 machinery — do NOT reimplement.
from depth_dissoc_sketch import (   # noqa: E402
    stats_one_bank,        # per-bank PR/aniso/kurtosis (prereg s1.2, H1/H2/H3)
    _agg_over_seeds,       # mean/std over per-seed stat rows
    _z, _sign,              # z-score + tri-state sign helpers
    STAT_CONVENTIONS,       # the 3 pre-committed (stat, mult, description) tuples
    _select_merge_cells,    # g2/g3 whole-layer selection RULE (prereg s2.2) — the
                            # single place this can drift; NOT reimplemented here
                            # (review MINOR-1, 2026-07-14)
    _cell_qualifies,        # per-cell qualify predicate (c2_coherent ∧ non_negligible
                            # ∧ ¬saturated) — reused for honest per-layer merge status
                            # diagnosis (review MAJOR-2, 2026-07-14)
    build_dissociation,     # arch-1 D(L) = z[M]-z[C] + raw dM/dC concordance guard
    tracking_test,          # generic ΔD-vs-ΔS sign-concordance test + Holm-Bonferroni
    _synth_bank,            # planted-anisotropy synthetic K generator (selftest reuse)
)
from analyze_matrices import within_probe_rhos  # noqa: E402 — reuse core stat only

HARNESS = os.path.dirname(HERE)  # edit-harness/


# ============================================================== atomic IO
def _atomic_write_json(path, obj):
    d = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=2)
    os.replace(tmp, path)


# ==================================================== arch-1: parameterized readers
# Faithful, path-configurable ports of depth_dissoc_sketch.py's read_collateral /
# read_merge (which hardcode this machine's results/ layout). No statistic or
# selection RULE is reimplemented: `_select_merge_cells` is imported, not rewritten
# (review MINOR-1 — the original port re-derived the g2/g3 whole-layer selection
# inline, risking drift from the sketch's own copy; now there is exactly one
# implementation of that rule and both files call it).
def read_collateral_generic(results_dir, layers):
    """C(L) = aggregate.within_probe_mean_across_seeds from the pinned canonical
    G1_stability_L{L}_v2.json. Returns {layer: None} for any layer whose file is
    absent (never raises) so the caller can report a precise UNDECIDABLE reason."""
    out = {}
    for L in layers:
        f = os.path.join(results_dir, f"G1_stability_L{L}_v2.json")
        if not os.path.exists(f):
            out[L] = None
            continue
        agg = json.load(open(f))["aggregate"]
        out[L] = {
            "C": float(agg["within_probe_mean_across_seeds"]),
            "std": float(agg.get("within_probe_std_across_seeds", float("nan"))),
            "max_within_probe_perm_p": agg.get("max_within_probe_perm_p"),
            "n_seeds": int(agg.get("n_seeds", 0)),
            "source": os.path.basename(f),
        }
    return out


def _merge_filename(L):
    """arch-1 (untagged) naming convention already used at home: the canonical
    (no-suffix) file is L12; every other layer is L-suffixed. Generalized here to
    any layer so a future L10 RG run (prereg s2.4) needs no code change."""
    return "RG_operating_curve_table.json" if L == 12 else f"RG_operating_curve_table_L{L}.json"


def _merge_filename_tagged(tag, L):
    """arch-2 (tag-namespaced) naming produced by the RG runner for non-Llama
    architectures, e.g. RG_operating_curve_table_qwen15b_L21.json. There is NO
    untagged/canonical variant for a tagged arch (verified on disk 2026-07-14:
    only *_qwen15b_L21.json exists, no bare *_qwen15b.json), so EVERY layer is
    L-suffixed — unlike _merge_filename's L12-special-case for arch-1."""
    return f"RG_operating_curve_table_{tag}_L{L}.json"


def _diagnose_merge_cells(cells, seeds):
    """Honest per-g/per-seed accounting of WHY a merge layer has no qualifying cell
    (review MAJOR-2): for g∈{2,3} record which seeds' cells are present and, for
    each, the three qualification flags + the specific fail reasons (not_c2_coherent
    / negligible / saturated). Reuses the imported `_cell_qualifies` predicate."""
    out = {}
    for g in (2, 3):
        present, per_seed = [], {}
        for s in seeds:
            c = cells.get(f"g{g}_s{s}")
            if c is None:
                continue
            present.append(s)
            reasons = []
            if not bool(c.get("c2_coherent")):
                reasons.append("not_c2_coherent")
            if not bool(c.get("non_negligible")):
                reasons.append("negligible")
            if bool(c.get("saturated")):
                reasons.append("saturated")
            per_seed[str(s)] = {
                "c2_coherent": bool(c.get("c2_coherent")),
                "non_negligible": bool(c.get("non_negligible")),
                "saturated": bool(c.get("saturated")),
                "qualifies": bool(_cell_qualifies(c)),
                "fail_reasons": reasons,
            }
        out[f"g{g}"] = {"present_seeds": present, "per_seed": per_seed}
    return out


def _read_merge(mergedir, layers, fname_fn, status_out=None):
    """Shared merge-table reader. The g2/g3 whole-layer selection RULE itself
    (prereg s2.2) is NOT reimplemented here — `_select_merge_cells` is imported
    verbatim from depth_dissoc_sketch.py (review MINOR-1) so there is exactly one
    copy of that rule to keep in sync. This body only adds path-parameterization
    and the aggregation/JSON-shaping; `fname_fn(L)` supplies the per-layer
    filename so the SAME code serves arch-1 (untagged) and arch-2 (tag-namespaced)
    RG tables.

    Return value {layer: None|dict} is UNCHANGED (arch-1 byte-compat): None for a
    missing file OR a present-but-degenerate (no qualifying g2/g3 cell) layer, the
    full M dict otherwise. review MAJOR-2 adds two honest-accounting side channels
    that DO NOT alter the return value: (a) when `status_out` is provided it is
    populated per layer with status ∈ {file_absent, present_but_no_qualifying_g2g3_
    cell (+ per-g/seed diagnosis), qualified}; (b) a present-but-degenerate layer
    emits the sketch's degeneracy stderr WARNING (re-added) so a silently-dropped
    merge depth is never invisible."""
    out = {}
    for L in layers:
        f = os.path.join(mergedir, fname_fn(L))
        fname = os.path.basename(f)
        if not os.path.exists(f):
            out[L] = None
            if status_out is not None:
                status_out[L] = {"status": "file_absent", "file": fname}
            continue
        d = json.load(open(f))
        cells = d["cells"]
        seeds = [int(s) for s in d.get("seeds", [0, 1, 2])]

        chosen_g, chosen = _select_merge_cells(cells, seeds)

        if not chosen:
            out[L] = None
            sys.stderr.write(
                f"[depth_dissoc_gate] WARNING: merge layer L{L} file {fname} is "
                f"PRESENT but has NO qualifying g2/g3 cell (c2_coherent ∧ "
                f"non_negligible ∧ ¬saturated) -> M unavailable at this depth; any "
                f"ΔD touching L{L} is non-informative (distinct from merge-file-absent).\n")
            if status_out is not None:
                status_out[L] = {"status": "present_but_no_qualifying_g2g3_cell",
                                 "file": fname, "cells": _diagnose_merge_cells(cells, seeds)}
            continue
        prho = np.array([c["partial_rho_geom"] for _, _, c in chosen], dtype=float)
        own = np.array([c["partial_rho_geom_ownmag"] for _, _, c in chosen], dtype=float)
        icos = np.array([c["rho_I_cos_drop"] for _, _, c in chosen], dtype=float)
        out[L] = {
            "M": round(float(prho.mean()), 6),
            "M_std": round(float(prho.std(ddof=0)), 6),
            "M_ownmag_mean": round(float(own.mean()), 6),
            "rho_I_cos_drop_mean": round(float(icos.mean()), 6),
            "chosen_g": chosen_g,
            "cells_used": [k for _, k, _ in chosen],
            "n_cells": int(prho.size),
            "source": fname,
            "layer_field": d.get("layer"),
        }
        if status_out is not None:
            status_out[L] = {"status": "qualified", "file": fname,
                             "chosen_g": chosen_g, "M": out[L]["M"],
                             "cells_used": out[L]["cells_used"]}
    return out


def read_merge_generic(mergedir, layers):
    """arch-1 merge M(L) from untagged RG tables. Delegates to `_read_merge`;
    behavior is byte-identical to the previous inline implementation (arch-1
    numbers must not move — reviewed constraint)."""
    return _read_merge(mergedir, layers, _merge_filename)


def read_merge_tagged(mergedir, tag, layers, status_out=None):
    """arch-2 merge M(L) from tag-namespaced RG tables
    (RG_operating_curve_table_<tag>_L<L>.json), same selection/aggregation as
    arch-1 via the shared `_read_merge`. `status_out` (review MAJOR-2) captures the
    honest per-layer merge status so the literal path can report WHY a depth has no
    M — file-absent vs present-but-no-qualifying-g2/g3-cell."""
    return _read_merge(mergedir, layers, lambda L: _merge_filename_tagged(tag, L),
                       status_out=status_out)


def _glob_layer_banks(vecdir, tag, L):
    return sorted(glob.glob(os.path.join(vecdir, f"vectors_qv_{tag}_rome_cf_L{L}_s*.npz")))


def run_arch1(args):
    layers = sorted(int(x) for x in str(args.arch1_layers).split(","))
    coll = read_collateral_generic(args.results_dir, layers)
    merge = read_merge_generic(args.mergedir, layers)
    missing = [L for L in layers if coll.get(L) is None or merge.get(L) is None]
    if missing:
        return {"decidable": False,
                "reason": f"arch-1 grid missing collateral C(L) or merge M(L) at "
                          f"layers {missing} (see collateral_raw/merge_raw for what "
                          f"WAS found).",
                "layers": layers, "collateral_raw": coll, "merge_raw": merge}

    C_by_L = {L: coll[L]["C"] for L in layers}
    M_by_L = {L: merge[L]["M"] for L in layers}

    stat_rows_by_L, stat_by_L, no_bank = {}, {}, []
    for L in layers:
        paths = _glob_layer_banks(args.arch1_vector_dir, args.arch1_tag, L)
        if not paths:
            no_bank.append(L)
            continue
        rows = [stats_one_bank(p) for p in paths]
        stat_rows_by_L[L] = rows
        stat_by_L[L] = {
            "pr_frac": _agg_over_seeds(rows, "pr_frac")["mean"],
            "mean_cos": _agg_over_seeds(rows, "mean_cos")["mean"],
            "mean_abs_cos": _agg_over_seeds(rows, "mean_abs_cos")["mean"],
            "kappa": _agg_over_seeds(rows, "kappa")["mean"],
            "bands": {k: _agg_over_seeds(rows, k)
                      for k in ("pr_frac", "mean_cos", "mean_abs_cos", "kappa")},
            "n_seeds": len(rows),
        }
    if no_bank:
        return {"decidable": False,
                "reason": f"no raw-K bank for arch-1 at layer(s) {no_bank} "
                          f"(glob vectors_qv_{args.arch1_tag}_rome_cf_L<L>_s*.npz "
                          f"in {args.arch1_vector_dir}).",
                "layers": layers, "collateral_raw": coll, "merge_raw": merge}

    dissoc = build_dissociation(layers, C_by_L, M_by_L)   # reused verbatim, unmodified
    per_stat, winner = tracking_test(layers, dissoc["pairs"], stat_by_L)  # reused verbatim
    return {
        "decidable": True, "layers": layers,
        "collateral_C": coll, "merge_M": merge,
        "statistics_per_layer": stat_by_L, "statistics_per_seed": stat_rows_by_L,
        "dissociation": dissoc, "tracking": per_stat,
        "arch1_winner_candidate": winner,
    }


# ============================================================== arch-2: replication
def read_arch2_collateral_auto(matrices_dir, tag, layers, metric="logit",
                                known=True, edit_ok=True):
    """Best-effort C_arch2(L) computed FROM LOCAL matrices npz where present
    (results/matrices/gate_<tag>_rome_cf_L<L>_s*.npz), reusing the imported
    within_probe_rhos primitive (analyze_matrices.py) — point estimate only, no
    permutation null (out of scope for a depth-shape target). Layers with no
    matching npz are simply absent from the returned dict."""
    out = {}
    dkey = "damage_logit" if metric == "logit" else "damage_prob"
    for L in layers:
        paths = sorted(glob.glob(os.path.join(matrices_dir, f"gate_{tag}_rome_cf_L{L}_s*.npz")))
        vals = []
        for p in paths:
            try:
                d = np.load(p)
                COS = d["COS"].astype(float)
                D = d[dkey].astype(float)
                if edit_ok and "edit_ok" in d:
                    rows = d["edit_ok"].astype(float) > 0.5
                    COS, D = COS[rows], D[rows]
                if known and "pre_p" in d:
                    cols = d["pre_p"].astype(float) > 0.05
                    if cols.sum() >= 5:
                        COS, D = COS[:, cols], D[:, cols]
                wp = within_probe_rhos(COS, D)
                vals.append(float(np.nanmean(wp)))
            except Exception as e:
                sys.stderr.write(f"[depth_dissoc_gate] WARNING: failed reading {p}: {e}\n")
        if vals:
            out[L] = {"C": float(np.mean(vals)), "n_seeds": len(vals),
                       "source": f"auto:within_probe_rhos over {len(vals)} seed npz "
                                 f"(gate_{tag}_rome_cf_L{L}_s*.npz)"}
    return out


def build_target_arch2(layers, C_by_L):
    """Single-sided z-scored PROXY replication target — see the module docstring's
    GATE-INTERPRETATION note for why abs() is required and why this is not a
    literal two-family D(L). Returns a dict shaped like build_dissociation()'s
    output (profile + pairs with a 'sign_dD' field) so tracking_test() can be
    reused unmodified on either arch."""
    Ls = sorted(layers)
    vals = [abs(C_by_L[L]) for L in Ls]
    z, s = _z(vals)
    profile = [{"layer": L, "C_signed": round(C_by_L[L], 6), "C_abs": round(vals[i], 6),
                "zT": round(float(z[i]), 6)} for i, L in enumerate(Ls)]
    pairs = []
    for i in range(len(Ls) - 1):
        dT = float(z[i + 1] - z[i])
        pairs.append({"pair": f"L{Ls[i]}->L{Ls[i+1]}", "dD": round(dT, 6),
                      "sign_dD": _sign(dT), "dM": None, "dC": None,
                      "raw_unambiguous": None,
                      "note": "single-sided proxy pair; no dM/dC concordance guard "
                              "applies (see module docstring)."})
    return {"profile": profile, "sigma_T": round(s, 6), "pairs": pairs,
            "d_profile_ambiguous": None,
            "target_kind": "single_sided_proxy_abs_C"}


def _load_external_collateral(path):
    """Parse an optional {layer: rho_C} (or {'layers': {...}}) JSON into
    {int layer: {'C': float, 'source': str, ...}}. Malformed entries fail LOUD
    here (SystemExit naming the offending layer + missing key), not as a bare
    KeyError later — review MINOR-2, extracted verbatim so both the proxy and the
    literal C sources share one loud-fail path."""
    raw = json.load(open(path))
    raw = raw.get("layers", raw)  # allow either {"14": v} or {"layers": {"14": v}}
    out = {}
    for k, v in raw.items():
        if isinstance(v, dict):
            if "C" not in v:
                raise SystemExit(
                    f"[depth_dissoc_gate] malformed --arch2_collateral_json "
                    f"{path!r}: entry for layer {k!r} is a dict but has no 'C' key "
                    f"(got keys {sorted(v.keys())}). Expected either a bare number "
                    f"or a dict containing 'C'.")
            entry = dict(v)
        else:
            try:
                entry = {"C": float(v)}
            except (TypeError, ValueError) as e:
                raise SystemExit(
                    f"[depth_dissoc_gate] malformed --arch2_collateral_json "
                    f"{path!r}: entry for layer {k!r} ({v!r}) is neither a number "
                    f"nor a dict with 'C': {e}")
        entry["source"] = f"external:{os.path.basename(path)}"
        out[int(k)] = entry
    return out


def build_literal_target_arch2(layers, C_signed_by_L, M_by_L):
    """LITERAL two-family dissociation target for arch-2 (prereg s1.1):
      D_arch2(L) = z_grid[ M(L) ] - z_grid[ |C(L)| ].

    SIGN-HANDLING CHOICE (the prereg-ambiguous point for a negative-C arch —
    documented here, in the module docstring, and in a gate_interpretation entry;
    made fully auditable in the JSON): the collateral side is entered as |C(L)|,
    NOT the raw signed rho. Rationale — identical to the already-reviewed proxy
    convention on this same architecture: the prereg's "collateral-geometry
    dominance" is a MAGNITUDE concept (how strongly key-geometry governs damage at
    a depth), and Qwen-1.5B's within-probe rho is globally sign-inverted
    (rho_C < 0; memory crossarch-transfer-verdict-2026-07-02). Entering the raw
    signed rho would let that incidental global sign flip masquerade as depth-
    SHAPE. For a positive-C arch (Llama) abs() is a strict no-op, so this is the
    same D the prereg literally defines there. Because z[-x] = -z[x] when all C
    share a sign, signed-vs-abs genuinely flips the C-contribution's depth
    derivative — so the SIGNED-C variant is ALSO computed as an auditable shadow,
    and evaluate_gate records whether the winning statistic survives it
    (winner_robust_to_signed_C). Both build_dissociation calls (abs + shadow) are
    the arch-1 machinery, unmodified.

    M(L) is entered as read (partial_rho_geom); its sign is not flagged inverted
    and the prereg reads it directly. build_dissociation's own raw dM/dC guard
    (prereg s1.3) still runs and sets d_profile_ambiguous on the primary target."""
    Ls = sorted(layers)
    absC = {L: abs(float(C_signed_by_L[L])) for L in Ls}
    primary = build_dissociation(Ls, absC, M_by_L)                 # z[M] - z[|C|]
    shadow = build_dissociation(Ls, {L: float(C_signed_by_L[L]) for L in Ls}, M_by_L)  # z[M]-z[C]
    return {
        "target_kind": "literal_two_family",
        "sign_convention": "abs_C_primary_signed_C_shadow",
        "C_signed_by_L": {L: round(float(C_signed_by_L[L]), 6) for L in Ls},
        "C_abs_by_L": {L: round(float(absC[L]), 6) for L in Ls},
        "M_by_L": {L: round(float(M_by_L[L]), 6) for L in Ls},
        "abs_C": primary,
        "signed_C_shadow": shadow,
        # Top-level convenience mirrors of the PRIMARY (abs-C) target so the gate
        # evaluator + emitters can read `pairs` / `d_profile_ambiguous` uniformly.
        "pairs": primary["pairs"],
        "profile": primary["profile"],
        "d_profile_ambiguous": primary["d_profile_ambiguous"],
    }


def run_arch2(args):
    layers_req = sorted(int(x) for x in str(args.arch2_layers).split(","))
    vecdir = args.arch2_vector_dir or args.arch1_vector_dir
    bank_by_L = {L: _glob_layer_banks(vecdir, args.arch2_tag, L) for L in layers_req}
    bank_by_L = {L: p for L, p in bank_by_L.items() if p}
    layers_with_bank = sorted(bank_by_L.keys())
    if len(layers_with_bank) < 3:
        return {"arch2_tag": args.arch2_tag, "decidable": False,
                "reason": f"raw-K found at only {len(layers_with_bank)} depth(s) "
                          f"{layers_with_bank} (requested {layers_req} in "
                          f"{vecdir}); prereg s2.4 requires >=3 depths.",
                "layers_requested": layers_req, "layers_with_bank": layers_with_bank}

    stat_rows_by_L, stat_by_L = {}, {}
    for L in layers_with_bank:
        rows = [stats_one_bank(p) for p in bank_by_L[L]]
        stat_rows_by_L[L] = rows
        stat_by_L[L] = {
            "pr_frac": _agg_over_seeds(rows, "pr_frac")["mean"],
            "mean_cos": _agg_over_seeds(rows, "mean_cos")["mean"],
            "mean_abs_cos": _agg_over_seeds(rows, "mean_abs_cos")["mean"],
            "kappa": _agg_over_seeds(rows, "kappa")["mean"],
            "bands": {k: _agg_over_seeds(rows, k)
                      for k in ("pr_frac", "mean_cos", "mean_abs_cos", "kappa")},
            "n_seeds": len(rows),
        }

    matdir = args.arch2_matrices_dir or os.path.join(args.results_dir, "matrices")
    C_meta = dict(read_arch2_collateral_auto(matdir, args.arch2_tag, layers_with_bank))
    # review MINOR-2 loud-fail parsing now lives in _load_external_collateral so
    # the proxy C source and the literal C source share ONE validated path.
    ext_collateral = (_load_external_collateral(args.arch2_collateral_json)
                      if args.arch2_collateral_json else {})
    C_meta.update(ext_collateral)  # external OVERRIDES auto per layer, by design

    layers_with_C = sorted(set(layers_with_bank) & set(C_meta.keys()))
    if len(layers_with_C) < 3:
        return {"arch2_tag": args.arch2_tag, "decidable": False,
                "reason": "fewer than 3 depths have BOTH a raw-K bank AND a "
                          "collateral C(L) value (auto-discovered from local "
                          "matrices npz, or supplied via --arch2_collateral_json); "
                          "the arch-2 replication target needs >=3 depths "
                          "(prereg s2.4/s3.1).",
                "layers_requested": layers_req,
                "layers_with_bank": layers_with_bank,
                "layers_with_collateral": sorted(C_meta.keys()),
                "layers_used": layers_with_C,
                "collateral_detail": C_meta,
                "statistics_per_layer": stat_by_L}

    C_by_L = {L: C_meta[L]["C"] for L in layers_with_C}
    target = build_target_arch2(layers_with_C, C_by_L)
    per_stat, winner = tracking_test(layers_with_C, target["pairs"], stat_by_L)

    # ---- LITERAL two-family arch-2 target (NEW 2026-07-14) --------------------
    # Buildable D_arch2(L)=z[M]-z[|C|] wherever BOTH arch-2 collateral C(L)
    # (matrices auto-discovery ∪ external override, over ALL requested depths — NOT
    # restricted to bank depths, so a C-only depth still contributes to the
    # informative profile) AND arch-2 merge M(L) (tag-namespaced RG) exist. The
    # full statistic-REPLICATION gate additionally needs a raw-K stat bank at those
    # depths (prereg s3.1), so literal.decidable requires >=3 depths with C AND M
    # AND bank. When literal.decidable is False the caller (evaluate_gate) falls
    # back EXACTLY to the single-sided proxy gate (today's behavior), unchanged.
    C_all = dict(read_arch2_collateral_auto(matdir, args.arch2_tag, layers_req))
    C_all.update(ext_collateral)
    merge_status = {}   # review MAJOR-2: honest per-layer merge accounting
    M_all = read_merge_tagged(args.mergedir, args.arch2_tag, layers_req,
                              status_out=merge_status)
    C_lit = {L: C_all[L]["C"] for L in C_all}
    M_lit = {L: v["M"] for L, v in M_all.items() if v is not None}
    depths_D = sorted(set(C_lit) & set(M_lit))
    depths_gate = sorted(set(depths_D) & set(layers_with_bank))
    # review MAJOR-2: is EVERY requested merge depth present-but-non-qualifying (the
    # Qwen small-g-window case) vs merely absent? This distinguishes "wrong statistic
    # window" from "no data".
    n_present = sum(1 for v in merge_status.values() if v["status"] != "file_absent")
    n_qualified = sum(1 for v in merge_status.values() if v["status"] == "qualified")
    n_present_no_qual = sum(1 for v in merge_status.values()
                            if v["status"] == "present_but_no_qualifying_g2g3_cell")
    all_present_are_nonqualifying = (n_present > 0 and n_qualified == 0
                                     and n_present_no_qual == n_present)
    literal = {
        "C_signed_by_L": {L: round(float(C_lit[L]), 6) for L in sorted(C_lit)},
        "M_by_L": {L: round(float(M_lit[L]), 6) for L in sorted(M_lit)},
        "merge_detail": {str(L): merge_status[L] for L in sorted(merge_status)},
        "merge_status_counts": {"present": n_present, "qualified": n_qualified,
                                "present_but_no_qualifying_g2g3_cell": n_present_no_qual},
        "depths_C_and_M": depths_D,
        "depths_C_and_M_and_bank": depths_gate,
        "merge_source_convention": (
            f"RG_operating_curve_table_{args.arch2_tag}_L<L>.json (tag-namespaced; "
            f"NOT the untagged arch-1 RG names)"),
    }
    # The frozen §2.2 small-g window failed everywhere present -> the honest,
    # amendment-required verdict (review MAJOR-2, canned wording).
    QWEN_SMALLG_REASON = (
        "Literal gate UNDECIDABLE_AS_PREREGISTERED — Qwen merge M(L) has no "
        "c2-coherent qualifying cell at the pre-registered g∈{2,3} at any depth; its "
        "interference signal lives at g=10–20, OUTSIDE the frozen §2.2 small-g "
        "window. The two-family dissociation cannot be built on Qwen's pre-registered "
        "merge statistic without a prereg amendment. This is DISTINCT from "
        "merge-data-absent.")
    if len(depths_D) >= 3:
        # Informative D-profile over every C&M depth (may include bank-less depths).
        literal["D_profile_buildable"] = build_literal_target_arch2(
            depths_D, {L: C_lit[L] for L in depths_D}, {L: M_lit[L] for L in depths_D})
    if len(depths_gate) >= 3:
        gate_target = build_literal_target_arch2(
            depths_gate, {L: C_lit[L] for L in depths_gate},
            {L: M_lit[L] for L in depths_gate})
        stat_sub = {L: stat_by_L[L] for L in depths_gate}
        per_stat_lit, winner_lit = tracking_test(
            depths_gate, gate_target["abs_C"]["pairs"], stat_sub)
        per_stat_sig, winner_sig = tracking_test(
            depths_gate, gate_target["signed_C_shadow"]["pairs"], stat_sub)
        literal.update({
            "decidable": True,
            "gate_target": gate_target,
            "tracking": per_stat_lit,
            "winner_candidate": winner_lit,
            "tracking_signedC": per_stat_sig,
            "winner_candidate_signedC": winner_sig,
        })
    elif all_present_are_nonqualifying:
        # Distinct from data-absent: files are there, but the pre-registered small-g
        # window has no qualifying cell -> amendment required (review MAJOR-2).
        literal["decidable"] = False
        literal["merge_window_failure"] = True
        literal["reason"] = QWEN_SMALLG_REASON
    elif len(depths_D) >= 3:
        literal["decidable"] = False
        literal["reason"] = (
            f"literal D_arch2 buildable at depths {depths_D} (both C and M) but only "
            f"{depths_gate} also have a raw-K stat bank; the statistic-replication "
            f"gate needs >=3 bank depths (prereg s3.1). The buildable D-profile is "
            f"recorded under 'D_profile_buildable' for audit; gate falls back to the "
            f"single-sided proxy.")
    else:
        literal["decidable"] = False
        literal["reason"] = (
            f"fewer than 3 depths have BOTH arch-2 collateral C(L) AND merge M(L): "
            f"C at {sorted(C_lit)}, M at {sorted(M_lit)}, overlap {depths_D} "
            f"(merge status counts {literal['merge_status_counts']}); the literal "
            f"two-family D_arch2 needs >=3 overlapping depths (prereg s2.4). Gate "
            f"falls back to the single-sided proxy.")

    return {
        "arch2_tag": args.arch2_tag, "decidable": True,
        "layers_used": layers_with_C,
        "layers_with_bank_no_collateral": sorted(set(layers_with_bank) - set(layers_with_C)),
        "collateral_detail": C_meta,
        "statistics_per_layer": stat_by_L, "statistics_per_seed": stat_rows_by_L,
        "target": target, "tracking": per_stat,
        "winner_candidate": winner,
        "literal": literal,
    }


# ================================================================== gate evaluator
# Fallback sentinel for the prereg's LITERAL two-family gate (s1.1/s3.3: sign(dD)
# tracking where D=z[M]-z[|C|] on ARCH-2). It is emitted ONLY when the literal gate
# cannot be decided (arch-2 lacks tag-namespaced merge M(L) and/or a stat bank at
# >=3 depths overlapping its collateral C(L)) — i.e. exactly the pre-2026-07-14
# situation, in which the script falls back to the single-sided proxy gate.
# When arch-2 DOES have C(L) AND M(L) AND a raw-K stat bank at >=3 overlapping
# depths (the tag-namespaced RG tables landed), evaluate_gate decides the LITERAL
# gate and literal_gate_verdict carries PASS / KILL / AMBIGUOUS instead.
LITERAL_GATE_VERDICT = "UNDECIDABLE_AS_PREREGISTERED"


def _classify_C_sign(C_signed_by_L):
    """Sign regime of arch-2 collateral C over the gate depths (review MAJOR-1):
      'mixed'    -> at least one +C AND at least one -C: |C| collapses genuinely
                    opposite couplings into the same magnitude, so the abs() move is
                    UNjustified -> the gate is forced AMBIGUOUS (never PASS).
      'negative' -> all C <= 0 (>=1 strictly negative): globally sign-inverted arch;
                    |C| is a DEVIATION from prereg s1.1 signed-C wording -> a PASS is
                    emitted as the DISTINCT token PASS_ABS_CONVENTION, never bare PASS.
      'positive' -> all C >= 0: abs() is a strict no-op, D is prereg-literal -> a PASS
                    is the bare pre-registered token."""
    vals = [float(v) for v in C_signed_by_L.values()]
    has_pos = any(v > 0 for v in vals)
    has_neg = any(v < 0 for v in vals)
    if has_pos and has_neg:
        return "mixed"
    if has_neg:
        return "negative"
    return "positive"


def _evaluate_literal_gate(arch1, arch2, a1_winner):
    """LITERAL two-family gate (prereg s3.3), fired only when arch-2 has a decidable
    literal target (C AND M AND stat bank at >=3 overlapping depths). The winner is
    the highest-priority statistic (PR->A->kappa) that monotone-tracks BOTH arch-1's
    D(L) AND arch-2's literal D_arch2(L)=z[M]-z[|C|].

    VERDICT POLICY (review MAJOR-1, sign-aware — the arch-2 C sign regime over the
    gate depths decides which PASS token, if any, is admissible):
      - AMBIGUOUS iff (a) the arch-2 C sign regime is MIXED (both +C and -C present:
        |C| conflates opposite couplings, abs() is unjustified -> never PASS), OR
        (b) either arch's D-profile trips the prereg s1.3 raw-sign fragility guard.
        AMBIGUOUS PRECEDES any PASS (s1.3: "reported as AMBIGUOUS ... regardless of
        the tracking test").
      - PASS (bare) iff a winner exists, not ambiguous, AND the regime is POSITIVE
        (abs() is a strict no-op, so D is the prereg-literal signed-C target). This
        is the ONLY path that may emit the bare token "PASS".
      - PASS_ABS_CONVENTION (distinct token, NEVER bare PASS) iff a winner exists,
        not ambiguous, AND the regime is NEGATIVE (globally sign-inverted arch, e.g.
        Qwen). |C| is a documented DEVIATION from prereg s1.1's signed-C wording;
        this outcome is admissible AS the pre-registered result ONLY after a prereg
        amendment ratifies the |C| convention for sign-inverted architectures — a
        pending user decision (see gate_interpretation).
      - KILL otherwise (both arches measured, no statistic replicates on both).

    winner_robust_to_signed_C records whether the winner ALSO tracks the signed-C
    SHADOW D — but it CANNOT gate: because z[-x] = -z[x], in the legitimate NEGATIVE
    regime the signed-C target is a depth-reflection of the |C| target, so the winner
    tracking the shadow is EXPECTED to be False there. It is an audit field only; the
    sign-regime split above, not this flag, decides the token."""
    lit = arch2["literal"]
    target_kind = lit["gate_target"]["target_kind"]
    if target_kind != "literal_two_family":
        # Inverted guard (task 2026-07-14): a bare "PASS" is legitimate ONLY on the
        # literal target, so this branch must be reached ONLY with that target_kind.
        raise RuntimeError(
            f"_evaluate_literal_gate(): unexpected literal target_kind={target_kind!r} "
            "— the bare-PASS path assumes literal_two_family; revisit before trusting.")
    priority = [k for k, _, _ in STAT_CONVENTIONS]  # PR -> A -> kappa, prereg s3.2
    a1_tracks = {s["stat"]: bool(s["monotone_tracks_arch1"]) for s in arch1["tracking"]}
    a2_tracks = {s["stat"]: bool(s["monotone_tracks_arch1"]) for s in lit["tracking"]}
    a2_tracks_signed = {s["stat"]: bool(s["monotone_tracks_arch1"])
                        for s in lit["tracking_signedC"]}
    both = {k: (a1_tracks.get(k, False) and a2_tracks.get(k, False)) for k in priority}
    winner = next((k for k in priority if both[k]), None)
    a1_amb = bool(arch1["dissociation"]["d_profile_ambiguous"])
    a2_amb = bool(lit["gate_target"]["d_profile_ambiguous"])
    winner_robust_signedC = bool(winner and a1_tracks.get(winner)
                                 and a2_tracks_signed.get(winner))
    sign_regime = _classify_C_sign(lit["gate_target"]["C_signed_by_L"])
    sign_consistent = (sign_regime != "mixed")
    if a1_amb or a2_amb or sign_regime == "mixed":
        verdict = "AMBIGUOUS"
    elif winner and sign_regime == "positive":
        verdict = "PASS"
    elif winner and sign_regime == "negative":
        verdict = "PASS_ABS_CONVENTION"
    else:
        verdict = "KILL"
    if verdict == "PASS":
        # Bare PASS ONLY with a literal target AND a sign-consistent-POSITIVE C
        # (abs() a no-op -> prereg-literal). Negative/mixed can never reach here.
        assert target_kind == "literal_two_family" and sign_regime == "positive", \
            "bare PASS requires the literal target AND sign-consistent-positive C"
    return {
        "verdict": verdict,
        "literal_gate_decidable": True,
        "literal_gate_verdict": verdict,
        "target_kind": target_kind,
        "winning_statistic": winner,
        "arch2_C_sign_regime": sign_regime,
        "arch2_C_sign_consistent": sign_consistent,
        "winner_robust_to_signed_C": winner_robust_signedC,
        "d_profile_ambiguous": {"arch1": a1_amb, "arch2_literal": a2_amb},
        "depths_gate": lit["depths_C_and_M_and_bank"],
        "arch1_tracks": a1_tracks, "arch2_tracks": a2_tracks,
        "arch2_tracks_signed_C": a2_tracks_signed,
        "arch1_winner_candidate": a1_winner,
        "arch2_literal_winner_candidate": lit["winner_candidate"],
        "arch2_literal_winner_candidate_signedC": lit["winner_candidate_signedC"],
        "note": "LITERAL two-family gate (prereg s3.3): D_arch2(L)=z[M]-z[|C|]. Sign "
                "regime of arch-2 C over the gate depths gates the token: POSITIVE -> "
                "bare 'PASS' (prereg-literal, abs a no-op); NEGATIVE -> "
                "'PASS_ABS_CONVENTION' (documented deviation from prereg s1.1 signed-C, "
                "admissible only after a prereg amendment — pending user decision); "
                "MIXED -> forced 'AMBIGUOUS' (abs unjustified). AMBIGUOUS also fires on "
                "the prereg s1.3 raw-sign guard. winner_robust_to_signed_C is an audit "
                "field only and CANNOT gate (False is expected in the negative regime "
                "since z[-x]=-z[x]). See gate_interpretation.",
    }


def evaluate_gate(arch1, arch2):
    """Prereg s3.3. Two regimes (review MAJOR 2026-07-14 + literal extension
    2026-07-14):

    LITERAL gate — when arch-2 has a decidable literal two-family target
    (arch2['literal']['decidable'] is True: C(L) AND tag-namespaced merge M(L) AND
    a raw-K stat bank at >=3 overlapping depths). Then gate['verdict'] is the
    pre-registered outcome PASS / KILL / AMBIGUOUS, gate['literal_gate_decidable']
    is True and gate['literal_gate_verdict'] carries the same token. A bare "PASS"
    is legitimate here and ONLY here (see _evaluate_literal_gate).

    PROXY fallback — when the literal target is NOT decidable, behavior is EXACTLY
    the pre-literal-extension logic: arch-2's target is the single-sided proxy
    T_arch2(L)=z[|C_arch2(L)|], gate['verdict'] is PASS_PROXY_TARGET / KILL /
    KILL_FOR_GATE_PURPOSES (bare "PASS" NEVER emitted, asserted below),
    gate['literal_gate_decidable'] is False and gate['literal_gate_verdict'] is
    LITERAL_GATE_VERDICT ("UNDECIDABLE_AS_PREREGISTERED").

    See module docstring for the (separate) KILL vs KILL_FOR_GATE_PURPOSES
    distinction. `tracking_test`'s per-stat field is named "monotone_tracks_arch1"
    (inherited unmodified from depth_dissoc_sketch.py, which only ever ran arch-1);
    when this function reads it out of arch2's own tracking list, it means "tracks
    arch-2's own target", not literally arch-1 — a naming quirk of reuse, not a bug,
    flagged here for the reviewer."""
    if not arch1.get("decidable"):
        return {"verdict": "KILL_FOR_GATE_PURPOSES", "literal_gate_decidable": False,
                "literal_gate_verdict": LITERAL_GATE_VERDICT,
                "reason": "arch-1 undecidable: " + arch1.get("reason", "?")}
    a1_winner = arch1["arch1_winner_candidate"]
    if not arch2.get("decidable"):
        return {"verdict": "KILL_FOR_GATE_PURPOSES", "literal_gate_decidable": False,
                "literal_gate_verdict": LITERAL_GATE_VERDICT,
                "reason": "arch-2 undecidable: " + arch2.get("reason", "?"),
                "arch1_winner_candidate": a1_winner}

    # LITERAL gate takes precedence when arch-2's literal target is decidable.
    if arch2.get("literal", {}).get("decidable"):
        return _evaluate_literal_gate(arch1, arch2, a1_winner)

    # ---- PROXY fallback (unchanged pre-literal-extension behavior) ------------
    target_kind = arch2.get("target", {}).get("target_kind")
    if target_kind != "single_sided_proxy_abs_C":
        # Defensive, not speculative forward-compat: this codebase has exactly
        # ONE arch-2 target constructor (build_target_arch2), and it always
        # stamps this target_kind. If that ever changes, the PASS_PROXY_TARGET /
        # literal_gate_decidable=False guarantee below no longer automatically
        # holds — fail loudly here rather than silently mislabel a result.
        raise RuntimeError(
            f"evaluate_gate(): unexpected arch2 target_kind={target_kind!r} — "
            "the PASS_PROXY_TARGET / literal_gate_decidable=False logic assumes "
            "single_sided_proxy_abs_C; revisit before trusting this verdict.")

    priority = [k for k, _, _ in STAT_CONVENTIONS]  # PR -> A -> kappa, prereg s3.2
    a1_tracks = {s["stat"]: bool(s["monotone_tracks_arch1"]) for s in arch1["tracking"]}
    a2_tracks = {s["stat"]: bool(s["monotone_tracks_arch1"]) for s in arch2["tracking"]}
    both = {k: (a1_tracks.get(k, False) and a2_tracks.get(k, False)) for k in priority}
    winner = next((k for k in priority if both[k]), None)
    verdict = "PASS_PROXY_TARGET" if winner else "KILL"
    assert verdict != "PASS", "bare PASS must never be emitted against a proxy target"
    return {
        "verdict": verdict,
        "literal_gate_decidable": False,
        "literal_gate_verdict": LITERAL_GATE_VERDICT,
        "target_kind": target_kind,
        "winning_statistic": winner,
        "arch1_tracks": a1_tracks, "arch2_tracks": a2_tracks,
        "arch1_winner_candidate": a1_winner,
        "arch2_winner_candidate": arch2["winner_candidate"],
        "note": "PASS_PROXY_TARGET/KILL here are against arch-2's SINGLE-SIDED "
                "PROXY target (target_kind=single_sided_proxy_abs_C), a weaker "
                "claim than the prereg's literal two-family D(L) — see "
                "literal_gate_verdict and the module docstring GATE-"
                "INTERPRETATION note. PASS_PROXY_TARGET is NOT the pre-"
                "registered gate firing; it is this script's practical, "
                "disclosed-as-weaker substitute.",
    }


GATE_INTERPRETATION = [
    "TWO gate regimes. LITERAL two-family gate (prereg s1.1 D(L)=z[M]-z[C]) fires "
    "when arch-2 has tag-namespaced merge M(L) (RG_operating_curve_table_<tag>_"
    "L<L>.json), collateral C(L), AND a raw-K stat bank at >=3 OVERLAPPING depths; "
    "then gate.verdict is the pre-registered PASS/KILL/AMBIGUOUS and "
    "gate.literal_gate_decidable=true. Otherwise the script FALLS BACK to a "
    "SINGLE-SIDED PROXY target T_arch2(L)=z_grid[|C_arch2(L)|] (arch-2's own "
    "collateral-dominance MAGNITUDE, not a merge-vs-collateral dissociation): "
    "gate.verdict is PASS_PROXY_TARGET/KILL/KILL_FOR_GATE_PURPOSES, "
    "gate.literal_gate_decidable=false and gate.literal_gate_verdict="
    "'UNDECIDABLE_AS_PREREGISTERED'. A PASS_PROXY_TARGET is WEAKER than the "
    "literal gate and must be reported as such (see arch2.target.target_kind).",
    "SIGN POLICY (review MAJOR-1; three regimes by the sign of arch-2 C over the "
    "gate depths, recorded as gate.arch2_C_sign_regime / arch2_C_sign_consistent): "
    "(i) POSITIVE (all C>=0) -> |C| is a strict no-op, D_arch2=z[M]-z[C] is exactly "
    "prereg s1.1, so a gate pass is the BARE token 'PASS'; (ii) NEGATIVE (all C<=0, "
    "globally sign-inverted arch e.g. Qwen) -> the |C| target is a DEVIATION from "
    "prereg s1.1's signed-C wording, so a gate pass is the DISTINCT token "
    "'PASS_ABS_CONVENTION' (never bare PASS); (iii) MIXED (both +C and -C) -> |C| "
    "conflates genuinely opposite couplings into one magnitude, abs() is UNjustified, "
    "so the gate is FORCED to 'AMBIGUOUS' regardless of tracking (the reviewer's "
    "{-0.5,+0.3,-0.1} case must never PASS). abs() is used because the prereg's "
    "'collateral-geometry dominance' is a MAGNITUDE and Qwen's within-probe rho is "
    "sign-inverted (memory crossarch-transfer-verdict-2026-07-02). The SIGNED-C "
    "variant is an auditable shadow (arch2.literal.gate_target.signed_C_shadow) and "
    "gate.winner_robust_to_signed_C is an AUDIT field ONLY — it cannot gate, because "
    "z[-x]=-z[x] makes it expected-False in the legitimate negative regime. For "
    "arch-1 (positive C) abs() is a no-op: build_dissociation()/tracking_test() run "
    "UNMODIFIED and arch-1 numbers are byte-identical to depth_dissoc_sketch.py.",
    "PASS_ABS_CONVENTION is NOT the bare pre-registered outcome. It marks a gate "
    "pass on a globally-NEGATIVE-C architecture where the two-family D was built with "
    "|C| — a documented deviation from prereg s1.1, which is written for signed C. It "
    "is admissible AS the pre-registered result ONLY after a prereg AMENDMENT ratifies "
    "the |C| convention for sign-inverted architectures. That ratification is a "
    "PENDING USER DECISION; until then, a PASS_ABS_CONVENTION is a conditional "
    "positive, reported distinctly so no reader can mistake it for bare PASS.",
    "AMBIGUOUS (literal gate only) fires when (a) the arch-2 C sign regime is MIXED "
    "(above), or (b) per prereg s1.3 EITHER arch's D-profile trips the raw dM/dC "
    "fragility guard (d_profile_ambiguous — M and C not cleanly opposite at some pair, "
    "so sign(dD) depends on a 3-sample std ratio). AMBIGUOUS and KILL both mean "
    "descriptive-only publication, but a reviewer needs the distinction: AMBIGUOUS = "
    "the target/convention is unjustified or fragile; KILL = the target is clean but "
    "no statistic replicated on both arches.",
    "MERGE HONESTY (review MAJOR-2): _read_merge records per-layer status "
    "(arch2.literal.merge_detail) distinguishing file_absent / "
    "present_but_no_qualifying_g2g3_cell (with the per-g/seed reason: not_c2_coherent "
    "/ negligible / saturated) / qualified, and emits a stderr WARNING for a "
    "present-but-degenerate layer. When EVERY present arch-2 merge file has no "
    "qualifying g2/g3 cell (the real Qwen case: its signal lives at g=10-20, outside "
    "the frozen s2.2 small-g window), literal.reason states the two-family "
    "dissociation cannot be built on Qwen's pre-registered merge statistic without a "
    "prereg amendment — DISTINCT from merge-data-absent (literal.merge_window_failure=true).",
    "C_arch2(L) is (a) auto-discovered by running within_probe_rhos (imported from "
    "analyze_matrices.py, the G1 gate's own primitive) on any local results/"
    "matrices/gate_<tag>_rome_cf_L<L>_s*.npz found, point estimate only (no "
    "permutation null — out of scope for a depth-shape target), and/or (b) "
    "supplied via --arch2_collateral_json, which OVERRIDES the auto value per "
    "layer. Layers with neither are ABSENT from the profile, never treated as 0. "
    "For the LITERAL D-profile, C is discovered over ALL requested depths (a "
    "C-only depth with no bank still contributes to the informative "
    "D_profile_buildable), but the statistic-replication GATE additionally "
    "requires a raw-K bank at each depth (prereg s3.1).",
    "On the data situation at authoring time (2026-07-14): qwen15b matrices exist "
    "at L14 only, tag-namespaced merge at L21 only, raw-K bank at L14 only — zero "
    "overlap, so the DEFAULT on-box run returns arch2.decidable via the proxy path "
    "and literal_gate_decidable=false. As the local chain lands "
    "gate_qwen15b_rome_cf_L{17,21,24} matrices + RG_operating_curve_table_qwen15b_"
    "L{14,24} merge (and, for the full gate, raw-K banks at those depths), the "
    "literal gate activates automatically with no code change. Bare 'PASS' is "
    "emitted ONLY on the literal two-family target — a proxy-target match can "
    "never be mistaken for the pre-registered gate firing.",
]


# ======================================================================= real run
def build_arg_parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", default=os.path.join(HARNESS, "results"),
                     help="root for G1_stability_L*.json (arch-1 collateral) and, "
                          "unless --mergedir/--arch2_matrices_dir override, "
                          "results/merging + results/matrices under it")
    ap.add_argument("--mergedir", default=None,
                     help="default: <results_dir>/merging")
    ap.add_argument("--arch1_vector_dir", default=None,
                     help="default: <results_dir>/vectors")
    ap.add_argument("--arch1_tag", default="llama1b")
    ap.add_argument("--arch1_layers", default="8,12,14",
                     help="comma layers; must each have C(L), M(L), AND a raw-K bank")
    ap.add_argument("--arch2_vector_dir", default=None,
                     help="default: same as --arch1_vector_dir")
    ap.add_argument("--arch2_tag", default="qwen15b")
    ap.add_argument("--arch2_layers", default="14,17,21,24",
                     help="comma CANDIDATE layers to look for raw-K banks at "
                          "(>=3 must actually be present to be gate-eligible)")
    ap.add_argument("--arch2_matrices_dir", default=None,
                     help="default: <results_dir>/matrices (for collateral "
                          "auto-discovery)")
    ap.add_argument("--arch2_collateral_json", default=None,
                     help="optional {layer: rho_C} (or {'layers': {...}}) JSON; "
                          "overrides auto-discovered values per layer")
    ap.add_argument("--out", default=None,
                     help="default: <results_dir>/analysis/T11_gate_report.json")
    ap.add_argument("--selftest", action="store_true",
                     help="run synthetic PASS/FAIL/UNDECIDABLE fixtures, no real IO")
    return ap


def _resolve_defaults(args):
    if args.mergedir is None:
        args.mergedir = os.path.join(args.results_dir, "merging")
    if args.arch1_vector_dir is None:
        args.arch1_vector_dir = os.path.join(args.results_dir, "vectors")
    if args.arch2_vector_dir is None:
        args.arch2_vector_dir = args.arch1_vector_dir
    if args.arch2_matrices_dir is None:
        args.arch2_matrices_dir = os.path.join(args.results_dir, "matrices")
    if args.out is None:
        args.out = os.path.join(args.results_dir, "analysis", "T11_gate_report.json")
    return args


def run_real(args):
    arch1 = run_arch1(args)
    arch2 = run_arch2(args)
    gate = evaluate_gate(arch1, arch2)
    report = {
        "experiment": "T1.1_depth_dissociation_E0_GATE",
        "prereg": "docs/plans/PREREG-T11-DEPTH-DISSOCIATION-E0-20260713.md",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cli_args": {k: v for k, v in vars(args).items() if k != "selftest"},
        "arch1": arch1, "arch2": arch2, "gate": gate,
        "gate_interpretation": GATE_INTERPRETATION,
    }
    _atomic_write_json(args.out, report)
    print_summary(report)
    return report


def print_summary(report):
    g = report["gate"]
    print(f"\n=== T1.1 depth-dissociation E0 — GATE-GRADE ({report['experiment']}) ===")
    print(f"arch1.decidable={report['arch1'].get('decidable')}  "
          f"arch2.decidable={report['arch2'].get('decidable')}")
    print(f"VERDICT={g['verdict']}  target_kind={g.get('target_kind')}")
    print(f"literal_gate_decidable={g['literal_gate_decidable']}  "
          f"literal_gate_verdict={g.get('literal_gate_verdict')}")
    if g.get("literal_gate_decidable"):
        pre = ("IS the pre-registered outcome" if g["verdict"] in ("PASS", "KILL")
               else "is the literal-gate outcome")
        print(f"LITERAL two-family gate DECIDED (target_kind=literal_two_family). "
              f"'{g['verdict']}' {pre}.")
        print(f"  arch2_C_sign_regime={g.get('arch2_C_sign_regime')}  "
              f"arch2_C_sign_consistent={g.get('arch2_C_sign_consistent')}")
        print(f"  d_profile_ambiguous={g.get('d_profile_ambiguous')}  "
              f"winner_robust_to_signed_C={g.get('winner_robust_to_signed_C')} "
              f"(audit only — cannot gate)")
        if g["verdict"] == "PASS_ABS_CONVENTION":
            print("  NOTE: PASS_ABS_CONVENTION is NOT bare PASS — globally-negative-C "
                  "arch, |C| convention; admissible as pre-registered ONLY after a "
                  "prereg amendment (pending user decision).")
        if g["verdict"] == "AMBIGUOUS":
            print("  NOTE: AMBIGUOUS = MIXED-sign C (abs unjustified) and/or prereg "
                  "s1.3 raw-sign fragility guard (NOT a mechanism claim; descriptive-only).")
    elif g["verdict"] == "PASS_PROXY_TARGET":
        print("NOTE: PASS_PROXY_TARGET is NOT the pre-registered gate firing — "
              "see literal_gate_verdict above and the module docstring.")
    # Buildable literal D-profile can exist even when the full gate is not decidable
    # (C&M at >=3 depths but stat banks missing) — surface it so a partial-data run
    # shows the literal constructor working.
    lit = report["arch2"].get("literal", {}) if isinstance(report.get("arch2"), dict) else {}
    if not g.get("literal_gate_decidable") and lit.get("D_profile_buildable"):
        print(f"literal D_arch2 BUILDABLE at depths {lit.get('depths_C_and_M')} "
              f"(C&M present) but gate not decidable: {lit.get('reason')}")
    if g.get("reason"):
        print(f"reason: {g['reason']}")
    if g.get("winning_statistic"):
        print(f"winning_statistic: {g['winning_statistic']}")
    print(f"report -> {report['cli_args'].get('out') or '(unset)'}")


# =================================================================== selftest
def _write_vec_bank(path, K, layer, seed, model="synthtest"):
    np.savez(path, K=K.astype(np.float32),
              knorm=np.linalg.norm(K, axis=1).astype(np.float32),
              layer=np.array(layer), seed=np.array(seed), model=np.array(model),
              editor=np.array("rome"), dataset=np.array("cf"),
              vectors_valid=np.array(1))


def _write_g1(path, C):
    _atomic_write_json(path, {"aggregate": {
        "within_probe_mean_across_seeds": C, "within_probe_std_across_seeds": 0.01,
        "max_within_probe_perm_p": 0.001, "max_within_probe_perm_p_editlevel": 0.001,
        "n_seeds": 3}})


def _write_rg(path, layer, M):
    cells = {}
    for s in (0, 1, 2):
        cells[f"g2_s{s}"] = {"c2_coherent": True, "non_negligible": True,
                              "saturated": False, "partial_rho_geom": M,
                              "partial_rho_geom_ownmag": M, "rho_I_cos_drop": 0.3}
    _atomic_write_json(path, {"layer": layer, "seeds": [0, 1, 2], "cells": cells})


def _write_rg_tagged(mergedir, tag, layer, M):
    """arch-2 tag-namespaced RG table (RG_operating_curve_table_<tag>_L<L>.json),
    same qualifying-cell shape as _write_rg — exercises read_merge_tagged."""
    _write_rg(os.path.join(mergedir, _merge_filename_tagged(tag, layer)), layer, M)


def _mk_arch1_fixture(tmpdir, C_by_L, M_by_L, aniso_by_L, n=60, d=64, rng=None):
    """Writes G1/RG jsons + raw-K vector banks (1 seed/layer, planted anisotropy
    level per layer via the imported _synth_bank) for one architecture."""
    rng = rng or np.random.default_rng(1)
    results_dir = os.path.join(tmpdir, "results")
    os.makedirs(os.path.join(results_dir, "merging"), exist_ok=True)
    vecdir = os.path.join(results_dir, "vectors")
    os.makedirs(vecdir, exist_ok=True)
    for L in C_by_L:
        _write_g1(os.path.join(results_dir, f"G1_stability_L{L}_v2.json"), C_by_L[L])
        _write_rg(os.path.join(results_dir, "merging", _merge_filename(L)), L, M_by_L[L])
        K = _synth_bank(n, d, aniso_by_L[L], rng)
        _write_vec_bank(os.path.join(vecdir, f"vectors_qv_llama1b_rome_cf_L{L}_s0.npz"), K, L, 0)
    return results_dir, vecdir


def _mk_arch2_banks(tmpdir_vec, tag, aniso_by_L, n=60, d=64, rng=None):
    rng = rng or np.random.default_rng(2)
    for L, a in aniso_by_L.items():
        K = _synth_bank(n, d, a, rng)
        _write_vec_bank(os.path.join(tmpdir_vec, f"vectors_qv_{tag}_rome_cf_L{L}_s0.npz"), K, L, 0)


def run_selftest():
    ok_all = True

    def check(name, cond, detail=""):
        nonlocal ok_all
        ok_all &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))

    ap = build_arg_parser()
    print("\n=== SELFTEST — synthetic two-architecture fixtures (no real IO) ===")

    # The planted _synth_bank aniso gradient at levels [0.05, 0.35, 0.75] moves ALL
    # three statistics in their CONCORDANT direction simultaneously (pr_frac down,
    # mean_cos up, kappa up) — proven by depth_dissoc_sketch.py's own selftest. So
    # for any 3-layer grid using this gradient, sign(dS)=[+1,+1] for every stat
    # under its pre-committed convention. We control PASS/KILL/UNDECIDABLE purely
    # via the TARGET (D on arch-1, T on arch-2), not the statistic gradient.
    aniso_by_L = {8: 0.05, 12: 0.35, 14: 0.75}

    # ---- Scenario A: PASS -------------------------------------------------
    with tempfile.TemporaryDirectory() as td:
        C1 = {8: 0.1, 12: 0.1, 14: 0.1}          # flat -> zC=[0,0,0]
        M1 = {8: 0.0, 12: 1.0, 14: 2.0}          # strictly increasing -> D up, up
        results_dir, vecdir = _mk_arch1_fixture(td, C1, M1, aniso_by_L)
        arch2_vecdir = os.path.join(td, "arch2vec")
        os.makedirs(arch2_vecdir, exist_ok=True)
        _mk_arch2_banks(arch2_vecdir, "synth2", aniso_by_L)  # same gradient, reused
        coll_json = os.path.join(td, "arch2_C.json")
        # C increasing (raw, positive) -> |C| increasing too -> T up, up (matches D)
        _atomic_write_json(coll_json, {"8": 0.1, "12": 1.0, "14": 2.0})
        args = ap.parse_args(["--results_dir", results_dir,
                               "--arch1_vector_dir", vecdir, "--arch1_layers", "8,12,14",
                               "--arch2_vector_dir", arch2_vecdir, "--arch2_tag", "synth2",
                               "--arch2_layers", "8,12,14",
                               "--arch2_collateral_json", coll_json,
                               "--out", os.path.join(td, "out.json")])
        args = _resolve_defaults(args)
        arch1 = run_arch1(args); arch2 = run_arch2(args); gate = evaluate_gate(arch1, arch2)
        check("A: arch1 decidable", arch1.get("decidable"))
        check("A: arch2 decidable", arch2.get("decidable"))
        check("A: verdict PASS_PROXY_TARGET (never bare PASS)",
              gate["verdict"] == "PASS_PROXY_TARGET", gate.get("verdict"))
        check("A: bare 'PASS' is impossible under the proxy target",
              gate["verdict"] != "PASS", gate.get("verdict"))
        check("A: literal_gate_decidable False (review MAJOR: never True on a "
              "proxy target)", gate["literal_gate_decidable"] is False)
        check("A: literal_gate_verdict UNDECIDABLE_AS_PREREGISTERED",
              gate.get("literal_gate_verdict") == "UNDECIDABLE_AS_PREREGISTERED",
              gate.get("literal_gate_verdict"))
        check("A: target_kind propagated to gate dict",
              gate.get("target_kind") == "single_sided_proxy_abs_C",
              gate.get("target_kind"))
        check("A: winning_statistic pr_frac (top priority)",
              gate.get("winning_statistic") == "pr_frac", gate.get("winning_statistic"))

    # ---- Scenario B: KILL (both decidable, sign-handling exercised) -------
    with tempfile.TemporaryDirectory() as td:
        C1 = {8: 0.1, 12: 0.1, 14: 0.1}
        M1 = {8: 0.0, 12: 1.0, 14: 2.0}           # arch-1 D still up, up (as A)
        results_dir, vecdir = _mk_arch1_fixture(td, C1, M1, aniso_by_L)
        arch2_vecdir = os.path.join(td, "arch2vec")
        os.makedirs(arch2_vecdir, exist_ok=True)
        _mk_arch2_banks(arch2_vecdir, "synth2", aniso_by_L)
        coll_json = os.path.join(td, "arch2_C.json")
        # NEGATIVE, INCREASING raw rho (-2 -> -1 -> 0): "less negative" as depth
        # rises, mirroring Qwen's sign-inverted rho_C. Raw signed C is INCREASING
        # (-2,-1,0) but |C| is DECREASING (2,1,0) -> T down, down -> mismatches
        # every statistic's concordant [+1,+1] expectation -> KILL. This is the
        # exact case the abs()-handling note in the module docstring targets: had
        # this script used raw signed C, sign(dT) would come out [+1,+1] (matching
        # the naive convention) and WRONGLY fire PASS on an artifact of the sign
        # flip rather than genuine depth-shape agreement.
        _atomic_write_json(coll_json, {"8": -2.0, "12": -1.0, "14": 0.0})
        args = ap.parse_args(["--results_dir", results_dir,
                               "--arch1_vector_dir", vecdir, "--arch1_layers", "8,12,14",
                               "--arch2_vector_dir", arch2_vecdir, "--arch2_tag", "synth2",
                               "--arch2_layers", "8,12,14",
                               "--arch2_collateral_json", coll_json,
                               "--out", os.path.join(td, "out.json")])
        args = _resolve_defaults(args)
        arch1 = run_arch1(args); arch2 = run_arch2(args); gate = evaluate_gate(arch1, arch2)
        t = arch2["target"]["profile"]
        c_abs_decreasing = t[0]["C_abs"] > t[1]["C_abs"] > t[2]["C_abs"]
        check("B: abs(C) profile is decreasing (sign-inversion handled correctly)",
              c_abs_decreasing, [r["C_abs"] for r in t])
        check("B: arch2 winner_candidate is None (no stat tracks a decreasing "
              "target under the increasing convention)",
              arch2["winner_candidate"] is None)
        check("B: verdict KILL", gate["verdict"] == "KILL", gate.get("verdict"))
        check("B: literal_gate_decidable False (always, per review MAJOR)",
              gate["literal_gate_decidable"] is False)
        check("B: literal_gate_verdict UNDECIDABLE_AS_PREREGISTERED",
              gate.get("literal_gate_verdict") == "UNDECIDABLE_AS_PREREGISTERED",
              gate.get("literal_gate_verdict"))

    # ---- Scenario C: UNDECIDABLE (bank/collateral overlap < 3) ------------
    with tempfile.TemporaryDirectory() as td:
        C1 = {8: 0.1, 12: 0.1, 14: 0.1}
        M1 = {8: 0.0, 12: 1.0, 14: 2.0}
        results_dir, vecdir = _mk_arch1_fixture(td, C1, M1, aniso_by_L)
        arch2_vecdir = os.path.join(td, "arch2vec")
        os.makedirs(arch2_vecdir, exist_ok=True)
        # 3 raw-K bank layers present...
        _mk_arch2_banks(arch2_vecdir, "synth2", {14: 0.05, 17: 0.35, 21: 0.75})
        coll_json = os.path.join(td, "arch2_C.json")
        # ...but collateral only supplied for 2 of them -> overlap = 2 < 3.
        _atomic_write_json(coll_json, {"14": 0.1, "17": 1.0})
        args = ap.parse_args(["--results_dir", results_dir,
                               "--arch1_vector_dir", vecdir, "--arch1_layers", "8,12,14",
                               "--arch2_vector_dir", arch2_vecdir, "--arch2_tag", "synth2",
                               "--arch2_layers", "14,17,21",
                               "--arch2_collateral_json", coll_json,
                               "--out", os.path.join(td, "out.json")])
        args = _resolve_defaults(args)
        arch1 = run_arch1(args); arch2 = run_arch2(args); gate = evaluate_gate(arch1, arch2)
        check("C: arch2 NOT decidable", arch2.get("decidable") is False)
        check("C: reason cites the overlap<3 condition",
              "3 depths" in arch2.get("reason", ""), arch2.get("reason"))
        check("C: verdict KILL_FOR_GATE_PURPOSES",
              gate["verdict"] == "KILL_FOR_GATE_PURPOSES", gate.get("verdict"))
        check("C: literal_gate_decidable False", gate["literal_gate_decidable"] is False)
        check("C: literal_gate_verdict UNDECIDABLE_AS_PREREGISTERED",
              gate.get("literal_gate_verdict") == "UNDECIDABLE_AS_PREREGISTERED",
              gate.get("literal_gate_verdict"))

    # ---- end-to-end real-run smoke: writes+reads back an atomic JSON report ---
    with tempfile.TemporaryDirectory() as td:
        C1 = {8: 0.1, 12: 0.1, 14: 0.1}
        M1 = {8: 0.0, 12: 1.0, 14: 2.0}
        results_dir, vecdir = _mk_arch1_fixture(td, C1, M1, aniso_by_L)
        arch2_vecdir = os.path.join(td, "arch2vec")
        os.makedirs(arch2_vecdir, exist_ok=True)
        _mk_arch2_banks(arch2_vecdir, "synth2", {14: 0.05, 17: 0.35, 21: 0.75})
        outp = os.path.join(td, "sub", "out.json")  # nested dir -> exercises makedirs
        args = ap.parse_args(["--results_dir", results_dir,
                               "--arch1_vector_dir", vecdir, "--arch1_layers", "8,12,14",
                               "--arch2_vector_dir", arch2_vecdir, "--arch2_tag", "synth2",
                               "--arch2_layers", "14,17,21", "--out", outp])
        args = _resolve_defaults(args)
        report = run_real(args)
        reread = json.load(open(outp))
        check("D: report written and re-readable", reread["gate"]["verdict"] == report["gate"]["verdict"])
        check("D: no .tmp file left behind", not os.path.exists(outp + ".tmp"))

    # ---- Scenario E: malformed --arch2_collateral_json fails loud (review MINOR-2) ---
    with tempfile.TemporaryDirectory() as td:
        C1 = {8: 0.1, 12: 0.1, 14: 0.1}
        M1 = {8: 0.0, 12: 1.0, 14: 2.0}
        results_dir, vecdir = _mk_arch1_fixture(td, C1, M1, aniso_by_L)
        arch2_vecdir = os.path.join(td, "arch2vec")
        os.makedirs(arch2_vecdir, exist_ok=True)
        _mk_arch2_banks(arch2_vecdir, "synth2", {8: 0.05, 12: 0.35, 14: 0.75})
        coll_json = os.path.join(td, "arch2_C_bad.json")
        # entry for layer 12 is a dict with no "C" key -> must fail loud, not KeyError
        _atomic_write_json(coll_json, {"8": 0.1, "12": {"note": "typo, no C"}, "14": 2.0})
        args = ap.parse_args(["--results_dir", results_dir,
                               "--arch1_vector_dir", vecdir, "--arch1_layers", "8,12,14",
                               "--arch2_vector_dir", arch2_vecdir, "--arch2_tag", "synth2",
                               "--arch2_layers", "8,12,14",
                               "--arch2_collateral_json", coll_json,
                               "--out", os.path.join(td, "out.json")])
        args = _resolve_defaults(args)
        try:
            run_arch2(args)
            check("E: malformed collateral entry raises (missing 'C' key)", False,
                  "no exception raised")
        except SystemExit as e:
            msg = str(e)
            check("E: malformed collateral entry raises SystemExit, not KeyError",
                  True)
            check("E: error message names the offending layer",
                  "'12'" in msg or "12" in msg, msg)
            check("E: error message names the missing key",
                  "'C'" in msg, msg)
        except KeyError as e:
            check("E: malformed collateral entry raises SystemExit, not KeyError",
                  False, f"got bare KeyError: {e}")

    # ==== LITERAL two-family gate scenarios (arch-2 has tag-namespaced merge) ====
    # arch-1 fixture reused across F/G: C DECREASING + M INCREASING so arch-1's own
    # D(L) is unambiguous (dM,dC opposite) and pr_frac tracks it (rising aniso).
    a1_C = {8: 2.0, 12: 1.0, 14: 0.1}
    a1_M = {8: 0.0, 12: 1.0, 14: 2.0}

    # ---- Scenario F: LITERAL gate PASS (bare PASS, positive C, robust) --------
    with tempfile.TemporaryDirectory() as td:
        results_dir, vecdir = _mk_arch1_fixture(td, a1_C, a1_M, aniso_by_L)
        mergedir = os.path.join(results_dir, "merging")
        arch2_vecdir = os.path.join(td, "arch2vec")
        os.makedirs(arch2_vecdir, exist_ok=True)
        _mk_arch2_banks(arch2_vecdir, "synth2", aniso_by_L)           # rising aniso
        for L, M in {8: 0.0, 12: 1.0, 14: 2.0}.items():              # M increasing
            _write_rg_tagged(mergedir, "synth2", L, M)
        coll_json = os.path.join(td, "arch2_C.json")
        # positive DECREASING C -> |C| decreasing; with M increasing -> literal D
        # up,up & unambiguous; pr_frac tracks -> PASS. Positive C => signed==abs =>
        # winner robust to the signed-C shadow.
        _atomic_write_json(coll_json, {"8": 2.0, "12": 1.0, "14": 0.1})
        args = ap.parse_args(["--results_dir", results_dir,
                               "--arch1_vector_dir", vecdir, "--arch1_layers", "8,12,14",
                               "--arch2_vector_dir", arch2_vecdir, "--arch2_tag", "synth2",
                               "--arch2_layers", "8,12,14",
                               "--arch2_collateral_json", coll_json,
                               "--out", os.path.join(td, "out.json")])
        args = _resolve_defaults(args)
        arch1 = run_arch1(args); arch2 = run_arch2(args); gate = evaluate_gate(arch1, arch2)
        check("F: arch2 literal target decidable",
              arch2["literal"].get("decidable") is True, arch2["literal"].get("reason"))
        check("F: gate target_kind literal_two_family",
              gate.get("target_kind") == "literal_two_family", gate.get("target_kind"))
        check("F: verdict is bare PASS (literal gate fired)",
              gate["verdict"] == "PASS", gate.get("verdict"))
        check("F: arch2_C_sign_regime positive (abs a no-op -> bare PASS admissible)",
              gate.get("arch2_C_sign_regime") == "positive", gate.get("arch2_C_sign_regime"))
        check("F: literal_gate_decidable True", gate["literal_gate_decidable"] is True)
        check("F: literal_gate_verdict PASS",
              gate.get("literal_gate_verdict") == "PASS", gate.get("literal_gate_verdict"))
        check("F: winning_statistic pr_frac (top priority tracks both arches)",
              gate.get("winning_statistic") == "pr_frac", gate.get("winning_statistic"))
        check("F: neither D-profile ambiguous",
              gate.get("d_profile_ambiguous") == {"arch1": False, "arch2_literal": False},
              gate.get("d_profile_ambiguous"))
        check("F: winner robust to signed-C shadow (positive C: signed==abs)",
              gate.get("winner_robust_to_signed_C") is True,
              gate.get("winner_robust_to_signed_C"))

    # ---- Scenario G: LITERAL gate KILL (NEGATIVE C exercises abs; clean KILL) --
    with tempfile.TemporaryDirectory() as td:
        results_dir, vecdir = _mk_arch1_fixture(td, a1_C, a1_M, aniso_by_L)
        mergedir = os.path.join(results_dir, "merging")
        arch2_vecdir = os.path.join(td, "arch2vec")
        os.makedirs(arch2_vecdir, exist_ok=True)
        # FALLING aniso -> arch-2 pr_frac RISES (opposite of arch-1) -> no stat tracks
        _mk_arch2_banks(arch2_vecdir, "synth2", {8: 0.75, 12: 0.35, 14: 0.05})
        for L, M in {8: 0.0, 12: 1.0, 14: 2.0}.items():              # M increasing
            _write_rg_tagged(mergedir, "synth2", L, M)
        coll_json = os.path.join(td, "arch2_C.json")
        # NEGATIVE C, |C| = 2,1,0.1 DECREASING (Qwen-like sign flip). With M
        # increasing, literal D up,up & UNAMBIGUOUS (via |C|) -> clean KILL, not
        # AMBIGUOUS; had we used raw signed C (-2,-1,-0.1 INCREASING) the target
        # would flip -> the exact abs()-load-bearing case.
        _atomic_write_json(coll_json, {"8": -2.0, "12": -1.0, "14": -0.1})
        args = ap.parse_args(["--results_dir", results_dir,
                               "--arch1_vector_dir", vecdir, "--arch1_layers", "8,12,14",
                               "--arch2_vector_dir", arch2_vecdir, "--arch2_tag", "synth2",
                               "--arch2_layers", "8,12,14",
                               "--arch2_collateral_json", coll_json,
                               "--out", os.path.join(td, "out.json")])
        args = _resolve_defaults(args)
        arch1 = run_arch1(args); arch2 = run_arch2(args); gate = evaluate_gate(arch1, arch2)
        cabs = arch2["literal"]["gate_target"]["C_abs_by_L"]
        check("G: literal gate decidable", arch2["literal"].get("decidable") is True)
        check("G: |C| decreasing in literal target (sign-inversion handled via abs)",
              cabs["8"] > cabs["12"] > cabs["14"] if all(isinstance(k, str) for k in cabs)
              else cabs[8] > cabs[12] > cabs[14], cabs)
        check("G: verdict KILL", gate["verdict"] == "KILL", gate.get("verdict"))
        check("G: literal_gate_decidable True", gate["literal_gate_decidable"] is True)
        check("G: literal_gate_verdict KILL",
              gate.get("literal_gate_verdict") == "KILL", gate.get("literal_gate_verdict"))
        check("G: winning_statistic None (no stat tracks both arches)",
              gate.get("winning_statistic") is None, gate.get("winning_statistic"))
        check("G: clean KILL, not AMBIGUOUS (both D-profiles unambiguous)",
              gate.get("d_profile_ambiguous") == {"arch1": False, "arch2_literal": False},
              gate.get("d_profile_ambiguous"))

    # ---- Scenario H: partial merge (2 depths) -> proxy FALLBACK ---------------
    with tempfile.TemporaryDirectory() as td:
        C1 = {8: 0.1, 12: 0.1, 14: 0.1}      # proxy-style arch-1 (flat C, as scenario A)
        M1 = {8: 0.0, 12: 1.0, 14: 2.0}
        results_dir, vecdir = _mk_arch1_fixture(td, C1, M1, aniso_by_L)
        mergedir = os.path.join(results_dir, "merging")
        arch2_vecdir = os.path.join(td, "arch2vec")
        os.makedirs(arch2_vecdir, exist_ok=True)
        _mk_arch2_banks(arch2_vecdir, "synth2", aniso_by_L)          # banks at 8,12,14
        for L, M in {8: 0.0, 12: 1.0}.items():                      # merge at only 2 depths
            _write_rg_tagged(mergedir, "synth2", L, M)
        coll_json = os.path.join(td, "arch2_C.json")
        _atomic_write_json(coll_json, {"8": 0.1, "12": 1.0, "14": 2.0})  # C increasing (proxy PASS)
        args = ap.parse_args(["--results_dir", results_dir,
                               "--arch1_vector_dir", vecdir, "--arch1_layers", "8,12,14",
                               "--arch2_vector_dir", arch2_vecdir, "--arch2_tag", "synth2",
                               "--arch2_layers", "8,12,14",
                               "--arch2_collateral_json", coll_json,
                               "--out", os.path.join(td, "out.json")])
        args = _resolve_defaults(args)
        arch1 = run_arch1(args); arch2 = run_arch2(args); gate = evaluate_gate(arch1, arch2)
        check("H: arch2 literal NOT decidable (merge at <3 overlapping depths)",
              arch2["literal"].get("decidable") is False)
        check("H: literal reason cites the C&M overlap<3 condition",
              "overlap" in arch2["literal"].get("reason", ""), arch2["literal"].get("reason"))
        check("H: gate FELL BACK to the single-sided proxy target",
              gate.get("target_kind") == "single_sided_proxy_abs_C", gate.get("target_kind"))
        check("H: literal_gate_decidable False (fallback)",
              gate["literal_gate_decidable"] is False)
        check("H: literal_gate_verdict UNDECIDABLE_AS_PREREGISTERED (fallback)",
              gate.get("literal_gate_verdict") == "UNDECIDABLE_AS_PREREGISTERED",
              gate.get("literal_gate_verdict"))
        check("H: proxy verdict PASS_PROXY_TARGET (never bare PASS on fallback)",
              gate["verdict"] == "PASS_PROXY_TARGET", gate.get("verdict"))

    # ---- Scenario I: literal D buildable but gate undecidable (banks miss the ---
    #      deep C&M layers) -> proxy FALLBACK. Mirrors tonight's expected partial
    #      activation: matrices+merge land at deep layers before their raw-K banks.
    with tempfile.TemporaryDirectory() as td:
        C1 = {8: 0.1, 12: 0.1, 14: 0.1}
        M1 = {8: 0.0, 12: 1.0, 14: 2.0}
        results_dir, vecdir = _mk_arch1_fixture(td, C1, M1, aniso_by_L)
        mergedir = os.path.join(results_dir, "merging")
        arch2_vecdir = os.path.join(td, "arch2vec")
        os.makedirs(arch2_vecdir, exist_ok=True)
        _mk_arch2_banks(arch2_vecdir, "synth2", aniso_by_L)          # banks ONLY at 8,12,14
        for L, M in {17: 0.0, 21: 1.0, 24: 2.0}.items():            # merge at DEEP layers (no banks)
            _write_rg_tagged(mergedir, "synth2", L, M)
        coll_json = os.path.join(td, "arch2_C.json")
        # C at the shallow (bank) layers for the proxy AND the deep layers for the
        # buildable literal D. depths_D = C∩M = {17,21,24} (3, D buildable);
        # depths_gate = depths_D ∩ bank = {} (<3) -> literal gate undecidable.
        _atomic_write_json(coll_json, {"8": 0.1, "12": 1.0, "14": 2.0,
                                       "17": 2.0, "21": 1.0, "24": 0.1})
        args = ap.parse_args(["--results_dir", results_dir,
                               "--arch1_vector_dir", vecdir, "--arch1_layers", "8,12,14",
                               "--arch2_vector_dir", arch2_vecdir, "--arch2_tag", "synth2",
                               "--arch2_layers", "8,12,14,17,21,24",
                               "--arch2_collateral_json", coll_json,
                               "--out", os.path.join(td, "out.json")])
        args = _resolve_defaults(args)
        arch1 = run_arch1(args); arch2 = run_arch2(args); gate = evaluate_gate(arch1, arch2)
        lit = arch2["literal"]
        check("I: literal gate NOT decidable (no bank at the deep C&M layers)",
              lit.get("decidable") is False)
        check("I: informative D_profile_buildable IS present (C&M at 3 depths)",
              "D_profile_buildable" in lit)
        check("I: D_profile_buildable spans the deep C&M depths",
              lit.get("depths_C_and_M") == [17, 21, 24], lit.get("depths_C_and_M"))
        check("I: literal reason cites the missing stat bank",
              "bank" in lit.get("reason", ""), lit.get("reason"))
        check("I: gate FELL BACK to the proxy target",
              gate.get("target_kind") == "single_sided_proxy_abs_C", gate.get("target_kind"))
        check("I: literal_gate_decidable False", gate["literal_gate_decidable"] is False)

    # ---- Scenario J: MIXED-sign arch-2 C -> forced AMBIGUOUS (review MAJOR-1) --
    # The reviewer's case {-0.5,+0.3,-0.1}: |C|={0.5,0.3,0.1} decreasing so abs WOULD
    # find a tracking winner (abs-would-PASS), but mixed sign makes |C| conflate
    # opposite couplings -> the gate MUST force AMBIGUOUS, never PASS.
    with tempfile.TemporaryDirectory() as td:
        results_dir, vecdir = _mk_arch1_fixture(td, a1_C, a1_M, aniso_by_L)
        mergedir = os.path.join(results_dir, "merging")
        arch2_vecdir = os.path.join(td, "arch2vec")
        os.makedirs(arch2_vecdir, exist_ok=True)
        _mk_arch2_banks(arch2_vecdir, "synth2", aniso_by_L)           # rising aniso
        for L, M in {8: 0.0, 12: 1.0, 14: 2.0}.items():
            _write_rg_tagged(mergedir, "synth2", L, M)
        coll_json = os.path.join(td, "arch2_C.json")
        _atomic_write_json(coll_json, {"8": -0.5, "12": 0.3, "14": -0.1})  # MIXED sign
        args = ap.parse_args(["--results_dir", results_dir,
                               "--arch1_vector_dir", vecdir, "--arch1_layers", "8,12,14",
                               "--arch2_vector_dir", arch2_vecdir, "--arch2_tag", "synth2",
                               "--arch2_layers", "8,12,14",
                               "--arch2_collateral_json", coll_json,
                               "--out", os.path.join(td, "out.json")])
        args = _resolve_defaults(args)
        arch1 = run_arch1(args); arch2 = run_arch2(args); gate = evaluate_gate(arch1, arch2)
        check("J: literal gate decidable", arch2["literal"].get("decidable") is True)
        check("J: arch2_C_sign_regime MIXED", gate.get("arch2_C_sign_regime") == "mixed",
              gate.get("arch2_C_sign_regime"))
        check("J: arch2_C_sign_consistent False", gate.get("arch2_C_sign_consistent") is False)
        check("J: abs WOULD have found a winner (winning_statistic pr_frac)...",
              gate.get("winning_statistic") == "pr_frac", gate.get("winning_statistic"))
        check("J: ...but verdict is FORCED to AMBIGUOUS (never PASS on mixed sign)",
              gate["verdict"] == "AMBIGUOUS", gate.get("verdict"))
        check("J: bare PASS is impossible under mixed-sign C",
              gate["verdict"] != "PASS", gate.get("verdict"))

    # ---- Scenario K: globally-NEGATIVE arch-2 C, |C| gate passes ---------------
    #      -> DISTINCT token PASS_ABS_CONVENTION, never bare PASS (review MAJOR-1).
    with tempfile.TemporaryDirectory() as td:
        results_dir, vecdir = _mk_arch1_fixture(td, a1_C, a1_M, aniso_by_L)
        mergedir = os.path.join(results_dir, "merging")
        arch2_vecdir = os.path.join(td, "arch2vec")
        os.makedirs(arch2_vecdir, exist_ok=True)
        _mk_arch2_banks(arch2_vecdir, "synth2", aniso_by_L)           # rising aniso -> pr_frac tracks
        for L, M in {8: 0.0, 12: 1.0, 14: 2.0}.items():
            _write_rg_tagged(mergedir, "synth2", L, M)
        coll_json = os.path.join(td, "arch2_C.json")
        # all-negative, |C| = 2,1,0.1 decreasing -> literal D up,up & unambiguous;
        # pr_frac tracks both arches -> a PASS, but on a sign-inverted arch.
        _atomic_write_json(coll_json, {"8": -2.0, "12": -1.0, "14": -0.1})
        args = ap.parse_args(["--results_dir", results_dir,
                               "--arch1_vector_dir", vecdir, "--arch1_layers", "8,12,14",
                               "--arch2_vector_dir", arch2_vecdir, "--arch2_tag", "synth2",
                               "--arch2_layers", "8,12,14",
                               "--arch2_collateral_json", coll_json,
                               "--out", os.path.join(td, "out.json")])
        args = _resolve_defaults(args)
        arch1 = run_arch1(args); arch2 = run_arch2(args); gate = evaluate_gate(arch1, arch2)
        check("K: arch2_C_sign_regime NEGATIVE",
              gate.get("arch2_C_sign_regime") == "negative", gate.get("arch2_C_sign_regime"))
        check("K: verdict PASS_ABS_CONVENTION (distinct token, not bare PASS)",
              gate["verdict"] == "PASS_ABS_CONVENTION", gate.get("verdict"))
        check("K: bare 'PASS' is IMPOSSIBLE on a globally-negative-C arch",
              gate["verdict"] != "PASS", gate.get("verdict"))
        check("K: literal_gate_verdict mirrors the distinct token",
              gate.get("literal_gate_verdict") == "PASS_ABS_CONVENTION",
              gate.get("literal_gate_verdict"))
        check("K: winning_statistic pr_frac", gate.get("winning_statistic") == "pr_frac",
              gate.get("winning_statistic"))
        check("K: winner_robust_to_signed_C False (expected in neg regime; z[-x]=-z[x])",
              gate.get("winner_robust_to_signed_C") is False,
              gate.get("winner_robust_to_signed_C"))

    print(f"\nSELFTEST {'PASS' if ok_all else 'FAIL'}")
    return 0 if ok_all else 1


def main():
    ap = build_arg_parser()
    args = ap.parse_args()
    if args.selftest:
        sys.exit(run_selftest())
    args = _resolve_defaults(args)
    run_real(args)


if __name__ == "__main__":
    main()
