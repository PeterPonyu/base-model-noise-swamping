"""mechanism_sc_table.py — C1 S x C decomposition receipt (per model/layer group).

The closed-form ROME rank-one update imposes  DW k_p = S * (||k_p|| * cos)  where
S = ||v - W k|| / ||k|| = ||DW||_F (Eq. 2 of sections/03_method.tex). In the gate
.npz this normalized S is stored per edit as ``norm_growth`` (== ||DW||_F by
construction — see the Frobenius-norm identity in
docs/findings/THEOREM-SXC-DRAFT-2026-07-06.md §6.3). This script summarizes, per
(model, layer) group:

  S        = mean ||DW||_F = mean ||v-Wk||/||k||     (from norm_growth)
  |C|      = mean |cos|
  rho_C    = signed within-probe Spearman(|cos|, damage)      (probe-marginal-clean)
  rho_SC   = signed within-probe Spearman(S * |cos|, damage)

FIX (2026-07-06, review-gated): this script previously used ``resid_norm`` (the
UNNORMALIZED ||v-Wk|| numerator, missing the /||k|| term) as "S". resid_norm and
norm_growth differ by the per-edit factor ||k_e||, which is NOT exactly constant
across edits (checked: implied ||k_e|| = resid_norm/norm_growth has mean 4.59, std
0.25 on gate_llama1b_rome_cf_L8_s0 — a ~5.5% relative spread), so the substitution
shifted within-probe rho_SC by ~3% at L8 (0.3899 -> 0.4015) and, worse, changed
which of {S x C, raw key-cosine} has the larger within-probe rho at L8/L10 (S x C
now leads at ALL four layers L8/L10/L12/L14 once correctly normalized — the
"S x C loses to raw key-cosine at L8/L10" claim in the manuscript was a
normalization artifact, not a real finding — see THEOREM-SXC-DRAFT §6.3 and the
review that confirmed it). resid_norm is still recorded per group (as
``mean_S_resid_norm_unnormalized``) for provenance/debugging ONLY — it is not a
valid S for Eq. 2 and must never be read as one.

Scientific payoff: Llama groups should show LARGE S and cosine-tracks-damage; Qwen
groups should show MUCH SMALLER S (predicted 4-8x) and near-zero within-probe rho —
evidence damage is driven by residual NORM (S), not raw key orthogonality.

CPU-only. numpy on existing .npz. No GPU / torch / downloads.

Usage (verbatim from run_deep_until1900.sh):
  python experiments/mechanism_sc_table.py \
    --npz 'results/matrices/gate_llama1b_rome_cf_L*_s*.npz' \
          'results/matrices/gate_qwen05b_rome_cf_L12_s*.npz' \
          'results/matrices/gate_qwen15b_rome_cf_L14_s*.npz' \
    --known --edit_ok \
    --out results/C1_mechanism_sc_table.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    from analyze_matrices import spearman, within_probe_rhos  # noqa: E402
except Exception:  # pragma: no cover - fallback replica
    def spearman(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        m = np.isfinite(a) & np.isfinite(b)
        a, b = a[m], b[m]
        if a.size < 3:
            return np.nan
        ar = a.argsort().argsort().astype(float)
        br = b.argsort().argsort().astype(float)
        if ar.std() == 0 or br.std() == 0:
            return np.nan
        return float(np.corrcoef(ar, br)[0, 1])

    def within_probe_rhos(COS, D):
        return np.array([spearman(COS[:, j], D[:, j]) for j in range(COS.shape[1])])


# gate_<model>_<editor>_<dataset>_L<layer>_s<seed>.npz  ->  (model, editor, dataset, layer)
# NB: capture the EDITOR token too (2nd group). Previously discarded -> a --npz glob spanning
# >1 editor for the same model+layer silently POOLED them (MEMIT-vs-ROME bug, fixed 2026-07-03).
# VERIFICATION 2026-07-04: fix confirmed empirically (authoring/verification pass only). Per-editor
# rho_C now: MEMIT L8 0.0156 / L12 0.0357 (was pooled 0.211 / 0.319), matching the raw-npz signed
# peek and the independent C3_memit null (within-probe 0.0194 / 0.0374). A separate HOSTILE REVIEW
# still gates these numbers into the paper; do NOT quote the MEMIT S x C figure until that clears.
#
# PENDING REVIEW (authoring pass 2026-07-04, revised per review): capture the DATASET token too
# (3rd group, SINGLE token). Previously the dataset token was matched but DISCARDED, so a glob
# spanning cf+zsre at the same (model, editor, layer) (e.g. llama1b rome L10) silently POOLED
# them — the same bug class as the editor axis, moved to the dataset axis. Now it separates.
#
# u1e0 is DELIBERATELY EXCLUDED from the prefix alternation (was gate|g4|u1e0). Rationale: u1e0
# has its own pipeline, and its filename layout puts the edit MODE where this regex expects the
# editor token — u1e0_<model>_<mode>_<variant>_L..  (e.g. u1e0_llama1b_delete_refusal_L12) and
# u1e0_<model>_<editor>_<mode>_<variant>_L.. (e.g. u1e0_llama1b_alpha_delete_refusal_L12). A
# single-token dataset pattern WOULD still match the 2-token-mode names (parsing editor="delete",
# dataset="refusal"), mislabeling a rome/alpha-deletion cell as a "delete" editor. Dropping u1e0
# from the prefix makes every u1e0 file fail the regex and go through the notes/logged-skip path
# (no silent drop, no wrong label). Re-add u1e0 only with a u1e0-aware pattern. VERIFY: no-op on
# single-dataset cf-only globs (byte-identical group values); separates cf vs zsre on a mixed glob;
# all u1e0 files logged-skipped. See ../findings-MEMIT-SC-RECONCILIATION-2026-07-04.md.
# 2026-07-13: model token is now GREEDY (.+) so multi-underscore model tags parse
# (qwen25_7b, llama31_8bi, qwen3_8b, qwen25_14b, qwen3_14b, qwen3_32b) — the fp32/bf16
# precision twin is NOT covered here (its _fp32/_bf16 suffix correctly fails _s\d+.npz$;
# the plan routes it through analyze_matrices on explicit paths) —
# the old single-token ([^_]+) NO-MATCHed all of them, silently emptying the S x C table
# for every non-single-token family. editor+dataset are ANCHORED to known enumerations so
# the greedy model can't swallow them AND the intentional u1e0/deletion exclusion is
# preserved (e.g. ..._rome_delete_refusal_L... has dataset='delete' not in the set -> NO
# MATCH -> logged-skip, exactly as before). An unrecognised editor/dataset token fails the
# regex and skips-with-note (fail-safe), never a silent mislabel. VERIFIED against the
# on-disk corpus + the deletion filenames; extend the enumerations if a new editor/dataset lands.
TAG_RE = re.compile(
    r"(?:gate|g4)_(.+)_(rome|ft|alpha|alphaHO|memit|grace)_(cf|zsre|mquake|mquaket|popular)_L(\d+)_s(\d+)\.npz$")


def parse_tag(path):
    m = TAG_RE.search(os.path.basename(path))
    if not m:
        return None
    # (model, editor, dataset, layer)
    return m.group(1), m.group(2), m.group(3), int(m.group(4))


def masked(d, known, edit_ok):
    """Return (COS2, D2, S_row, S_unnorm_row) masked to edit_ok rows and known cols.

    COS2, D2 are [n_rows, n_cols]. S_row is the Eq.2-correct per-edit strength
    S = ||v-Wk||/||k|| == ||DW||_F, read from ``norm_growth`` (None if unavailable).
    S_unnorm_row is the UNNORMALIZED ||v-Wk|| numerator, read from ``resid_norm`` --
    retained for provenance/debugging ONLY; it is NOT a valid S x C statistic (see
    module docstring, fix 2026-07-06)."""
    COS = d["COS"].astype(float)
    D = d["damage_logit"].astype(float)
    if "S" in d.files:
        S_full = d["S"].astype(float)
    elif "norm_growth" in d.files:
        S_full = d["norm_growth"].astype(float)  # ||DW||_F == ||v-Wk||/||k|| == Eq.2's S
    else:
        S_full = None
    S_unnorm_full = d["resid_norm"].astype(float) if "resid_norm" in d.files else None

    row = np.ones(COS.shape[0], bool)
    if edit_ok and "edit_ok" in d.files:
        row = d["edit_ok"].astype(float) > 0.5
    col = np.ones(COS.shape[1], bool)
    if known and "pre_p" in d.files:
        c = d["pre_p"].astype(float) > 0.05
        if c.sum() >= 5:
            col = c
    COS2 = COS[row][:, col]
    D2 = D[row][:, col]
    S_row = None if S_full is None else S_full[row]
    S_unnorm_row = None if S_unnorm_full is None else S_unnorm_full[row]
    return COS2, D2, S_row, S_unnorm_row


def analyze_group(paths, known, edit_ok):
    S_means, S_unnorm_means, absC_means = [], [], []
    rho_C_seeds, rho_SC_seeds = [], []
    n_pairs = 0
    s_available = True
    for p in paths:
        d = np.load(p)
        COS2, D2, S_row, S_unnorm_row = masked(d, known, edit_ok)
        if COS2.size < 20:
            continue
        absC = np.abs(COS2)
        absC_means.append(float(np.nanmean(absC)))
        rho_C_seeds.append(float(np.nanmean(within_probe_rhos(absC, D2))))
        n_pairs += int(COS2.size)
        if S_row is None:
            s_available = False
        else:
            S_means.append(float(np.nanmean(S_row)))
            SC = S_row[:, None] * absC  # S * |cos|  broadcast to [n_rows, n_cols]
            rho_SC_seeds.append(float(np.nanmean(within_probe_rhos(SC, D2))))
        if S_unnorm_row is not None:
            S_unnorm_means.append(float(np.nanmean(S_unnorm_row)))

    if not absC_means:
        return None
    return {
        "n_files": len(paths),
        "n_pairs": n_pairs,
        "mean_S": (round(float(np.nanmean(S_means)), 4) if S_means else None),
        "mean_S_resid_norm_unnormalized": (
            round(float(np.nanmean(S_unnorm_means)), 4) if S_unnorm_means else None),
        "S_available": s_available and bool(S_means),
        "mean_absC": round(float(np.nanmean(absC_means)), 4),
        "within_probe_rho_C": round(float(np.nanmean(rho_C_seeds)), 4),
        "within_probe_rho_SC": (round(float(np.nanmean(rho_SC_seeds)), 4) if rho_SC_seeds else None),
    }


def main():
    ap = argparse.ArgumentParser(description="C1 S x C mechanism table grouped by (model, layer).")
    ap.add_argument("--npz", nargs="+", required=True, help=">=1 glob patterns for gate rome npz")
    ap.add_argument("--known", action="store_true", help="restrict to probes the base model knows (pre_p>0.05)")
    ap.add_argument("--edit_ok", action="store_true", help="drop failed edits (edit_ok==0)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    paths = sorted({p for pat in args.npz for p in glob.glob(pat)})
    groups = {}
    notes = []
    for p in paths:
        tag = parse_tag(p)
        if tag is None:
            notes.append(f"could not parse model/editor/dataset/layer from {os.path.basename(p)} — skipped")
            continue
        groups.setdefault(tag, []).append(p)

    rows = []
    saw_memit = False
    for (model, editor, dataset, layer) in sorted(groups):
        res = analyze_group(sorted(groups[(model, editor, dataset, layer)]), args.known, args.edit_ok)
        if res is None:
            notes.append(f"{model}/{editor}/{dataset} L{layer}: no usable pairs after filtering — skipped")
            continue
        if not res["S_available"]:
            notes.append(f"{model}/{editor}/{dataset} L{layer}: S (||v-Wk||) unavailable in npz")
        # rho_SC is a SINGLE-LAYER rank-one ROME identity. MEMIT spreads each edit across
        # several layers while damage is the aggregate, so masked()'s per-file S*|cos| is NOT
        # a valid S x C statistic for MEMIT — flag it rather than let it be read as one.
        rho_sc_valid = editor.lower() == "rome"
        if not rho_sc_valid and res.get("within_probe_rho_SC") is not None:
            notes.append(
                f"{model}/{editor}/{dataset} L{layer}: within_probe_rho_SC is NOT a valid "
                f"S x C statistic — the S x C closed form is a single-layer rank-one ROME "
                f"identity; this editor is not single-layer rank-one ROME. Quote rho_C only.")
        if editor.lower() == "memit":
            saw_memit = True
        rows.append({"model": model, "editor": editor, "dataset": dataset, "layer": layer,
                     "rho_SC_valid_sxc": rho_sc_valid, **res})

    out = {
        "statistic": ("S=mean||v-Wk||/||k||=mean||DW||_F (from norm_growth; Eq.2), "
                      "|C|=mean|cos|; signed within-probe Spearman(|cos|,dmg) and (S*|cos|,dmg). "
                      "mean_S_resid_norm_unnormalized is provenance-only, NOT a valid S."),
        "rho_SC_validity": (
            "within_probe_rho_SC is the closed-form single-layer rank-one ROME identity S*|cos|. "
            "It is a valid S x C statistic ONLY for editor==rome (rho_SC_valid_sxc==true). For "
            "MEMIT (multi-layer aggregate edit) and non-ROME editors it is NOT a valid S x C "
            "measurement — quote rho_C only. See findings-MEMIT-SC-RECONCILIATION-2026-07-04.md."),
        "filters": {"known": args.known, "edit_ok": args.edit_ok},
        "globs": args.npz,
        "groups": rows,
        "notes": notes,
    }
    if saw_memit:
        out["notes"].append(
            "MEMIT present: rho_SC rows carry rho_SC_valid_sxc=false — do NOT cite them as 'MEMIT S x C'.")
    print(json.dumps(out, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=2)
        print(f"[c1] wrote {args.out}")


if __name__ == "__main__":
    main()
