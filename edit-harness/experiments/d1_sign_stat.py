"""d1_sign_stat.py — D1 sign-inversion mechanism atlas, E0 CPU analysis.

=============================================================================
AUTHORING PASS. A separate hostile-review lane gates any paper claim built on
this. Reads only cached raw-K banks; never touches the network or the GPU.
Pre-registration: docs/plans/PREREG-D1-SIGN-E0-20260714.md (this file encodes
its §1-§3 decision rule verbatim; if the two ever disagree, the prereg wins and
this code is the bug).
=============================================================================

THE QUESTION (prereg §0): the signed within-probe collateral law rho_C has a
FAMILY-determined SIGN (Llama/Mistral/Gemma POSITIVE; Qwen NEGATIVE at every
tier and both generations). Does a per-family KEY-SPACE statistic — a covariance
-shape / anisotropy summary of the raw pre-edit edit-key bank K[N,d] — PREDICT
sign(rho_C) across families, with leave-one-family-out (LOFO) generalization?

WHAT THIS COMPUTES (prereg §1-§3, CPU-only, read-only inputs):
  1. For every cached raw-K bank vectors_qv_<tag>_rome_cf_L*_s*.npz it can find,
     the four admitted key-space statistics on the RAW edit keys K (prereg §1.2),
     each REUSED from experiments/analyze_aniso.py + depth_dissoc_sketch.py (the
     T1.1 primitives — NOT reinvented here):
        pr_frac   participation-ratio fraction of the centered key covariance
        mean_cos  signed pairwise edit-key cosine (directional anisotropy)
        abs_cos   |cos| pairwise (cone concentration, sign-blind)
        kappa     pooled excess kurtosis of the standardized key coordinates
  2. Each raw statistic is converted to a WIDTH/NORM-INVARIANT EXCESS over a
     norm-matched ISOTROPIC-Gaussian null MEAN at that bank's own (N,d) (prereg
     §1.4 — the deconfound: raw |cos| shrinks like 1/sqrt(d), so families with
     wider intermediate_size would separate mechanically; subtracting the (N,d)
     -matched null MEAN removes that. NOT a z-score — the null std is d-dependent
     and ~0 at N<<d, so dividing by it would reintroduce width; only the null
     LOCATION is removed. The isotropic draw is the same construction as
     analyze_aniso.isotropic_gaussian_band, extended to all four statistics. This
     removes width/norm but NOT the lineage confound — see §4 / the binding wording).
  3. Per-family aggregate excess (mean over that family's banks/seeds) + a sign
     label from the frozen atlas (prereg §2.1).
  4. The LOFO sign-prediction gate (prereg §3): a 1-D nearest-class-mean
     classifier whose threshold + direction are re-learned on the training
     families every fold (no global direction is peeked), plus the in-sample
     full-separation margin and an EXACT class-size-preserving label-permutation
     p (fixed-sequence over the four statistics in the fixed priority order; Holm
     reported as a conservative secondary). NB the permutation null assumes label
     EXCHANGEABILITY, which the single-Qwen-lineage negative class violates (prereg
     §3.2) — effective negative-side n is ~1 lineage, not the family count.
  5. Verdict PASS / PASS_SEPARATION / AMBIGUOUS / KILL / INSUFFICIENT_FAMILIES
     (prereg §3.3) with the binding-wording caveats, to
     <analysis>/D1_sign_stat_table.json.

Conventions match experiments/: fixed inputs, numpy only, 4-dp rounding,
deterministic. No AUROC.

Runs ON-BOX or locally: point --vector_dir at wherever the raw-K banks live (they
stay on-box after the dump wave — only the small JSON table is pulled home).

Usage:
  python experiments/d1_sign_stat.py                    # cached banks in results/vectors/
  python experiments/d1_sign_stat.py --vector_dir DIR   # banks elsewhere (e.g. on-box)
  python experiments/d1_sign_stat.py --selftest         # synthetic, no real IO
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
# Reuse the exact prereg-referenced primitives — do NOT reimplement (task rule).
from analyze_aniso import (  # noqa: E402
    participation_ratio,   # PR = (sum l)^2 / sum l^2
    pairwise_cos_stats,    # mean_cos / mean_abs_cos on raw K
    _gram_eigs,            # nonzero eigs of the centered Gram
    load_bank,             # loud-fail loader; requires raw K
)
from depth_dissoc_sketch import kurtosis_proxy  # noqa: E402  (prereg-identical H3)

HARNESS = os.path.dirname(HERE)                       # edit-harness/
RESULTS = os.path.join(HARNESS, "results")
VECDIR = os.path.join(RESULTS, "vectors")
OUTDIR = os.path.join(RESULTS, "analysis")
OUTJSON = os.path.join(OUTDIR, "D1_sign_stat_table.json")

RNG_SEED = 12345  # matches analyze_aniso.RNG_SEED so the null bands are reproducible.

# Frozen 8-family sign atlas (docs/plans/RESEARCH-DIRECTIONS-SYNTHESIS-20260714.md §A,
# 3-seed within-probe rho_C). +1 = POSITIVE, -1 = NEGATIVE. Tags are the harness bank
# tags (basename of the qv --out). Families with no cached raw-K bank are simply absent
# at run time (the analysis degrades gracefully). qwen15b (Qwen2.5-1.5B) is the smaller
# Qwen tier — NEGATIVE like the whole family (crossfamily atlas). The three commented
# tags are the BALANCING negatives whose dump lowers the permutation floor below 0.05
# (prereg §3.2); left in the map so their banks are auto-used the moment they exist.
FAMILY_SIGN = {
    "llama1b": +1,      # Llama-3.2-1B          (POS)  raw-K L8/L12/L14 present
    "mistral7b": +1,    # Mistral-7B-v0.3       (POS)  dump wave
    "llama31_8bi": +1,  # Llama-3.1-8B-Instruct (POS)  dump wave
    "gemma9b": +1,      # gemma-2-9b            (POS)  dump wave
    "qwen15b": -1,      # Qwen2.5-1.5B          (NEG)  raw-K L14 present
    "qwen3_8b": -1,     # Qwen3-8B-Base         (NEG)  dump wave
    "qwen25_7b": -1,    # Qwen2.5-7B            (NEG)  balancing dump (optional)
    "qwen25_14b": -1,   # Qwen2.5-14B           (NEG)  balancing dump (optional)
    "qwen3_14b": -1,    # Qwen3-14B             (NEG)  balancing dump (optional)
}

# Admitted statistics in the pre-committed PRIORITY ORDER (prereg §1.2/§3.2). Each is the
# EXCESS of a raw key-space statistic over its norm-matched (N,d) isotropic-null MEAN — the
# 1/sqrt(d) floor subtracted off (prereg §1.4). Excess, not a z-score: dividing by the null
# STD (itself d-dependent, and ~0 at N<<d) would REINTRODUCE the width confound through the
# scale and explode numerically — so only the null LOCATION is removed. The LOFO classifier
# learns the sign convention per fold, so no per-statistic direction is fixed a priori.
STAT_PRIORITY = [
    ("e_abs_cos", "H1 key-cone anisotropy: |cos| excess over the (N,d)-matched isotropic null"),
    ("e_pr_frac", "H2 participation-ratio-fraction excess (effective-rank / covariance shape)"),
    ("e_kappa",   "H3 coordinate-kurtosis excess (superposition / heavy tails)"),
    ("e_mean_cos", "H4 signed-anisotropy excess (robustness variant of H1)"),
]


# ------------------------------------------------------------------- statistics
def raw_stats(K):
    """The four raw key-space statistics for one bank K[N,d] (all from reused
    primitives). Returns a dict of floats + (N,d)."""
    N, d = K.shape
    eig_cen = _gram_eigs(K - K.mean(axis=0))
    pr = participation_ratio(eig_cen)
    pr_frac = pr / min(N - 1, d)
    cos = pairwise_cos_stats(K)
    kappa, _, _ = kurtosis_proxy(K)
    return {"pr_frac": float(pr_frac), "mean_cos": float(cos["mean_cos"]),
            "abs_cos": float(cos["mean_abs_cos"]), "kappa": float(kappa),
            "N": int(N), "d": int(d)}


def isotropic_null_bands(K, n_rep, rng):
    """Norm-matched isotropic-Gaussian null for ALL FOUR statistics at K's own (N,d).

    Same construction as analyze_aniso.isotropic_gaussian_band (draw N iid N(0,I_d)
    rows, rescale each to a shuffled empirical key norm — a random rotation of an
    isotropic cloud), EXTENDED from {PR, mean_cos} to the full 4-stat vector so every
    admitted statistic gets the same finite-(N,d) no-anisotropy floor. Reuses the
    imported primitives per draw; the draw itself is the 4-line standard construction.

    NB the caller passes ONE shared rng across all banks, so each bank's null draws
    consume the rng state the previous bank left — the null MEANS are therefore
    (deterministically) bank-order-dependent. This is negligible at an adequate n_rep
    (the null mean is a stable population quantity; only its Monte-Carlo estimate moves
    by O(std/sqrt(n_rep)) with the seed stream), and it keeps byte-identity with the
    analyze_aniso convention. Give each bank its own seeded rng if order-independence
    is ever wanted.
    """
    N, d = K.shape
    norms = np.linalg.norm(K, axis=1)
    acc = {"pr_frac": [], "mean_cos": [], "abs_cos": [], "kappa": []}
    for _ in range(n_rep):
        G = rng.standard_normal((N, d))
        G *= (rng.permutation(norms) / (np.linalg.norm(G, axis=1) + 1e-12))[:, None]
        eig = _gram_eigs(G - G.mean(axis=0))
        acc["pr_frac"].append(participation_ratio(eig) / min(N - 1, d))
        c = pairwise_cos_stats(G)
        acc["mean_cos"].append(c["mean_cos"])
        acc["abs_cos"].append(c["mean_abs_cos"])
        acc["kappa"].append(kurtosis_proxy(G)[0])
    out = {}
    for k, v in acc.items():
        a = np.asarray(v, float)
        out[k] = {"mean": float(a.mean()), "std": float(a.std(ddof=0)), "n_rep": int(a.size)}
    return out


def _excess(x, band):
    """Excess of raw stat x over the isotropic-null MEAN (the 1/sqrt(d) floor removed).
    Width/norm-invariant; the null STD is deliberately NOT divided out (see STAT_PRIORITY)."""
    return float(x - band["mean"])


def bank_stat_vector(K, n_rep, rng):
    """The width-invariant excess vector (e_pr_frac, e_mean_cos, e_abs_cos, e_kappa)."""
    raw = raw_stats(K)
    band = isotropic_null_bands(K, n_rep, rng)
    return {
        "e_pr_frac": _excess(raw["pr_frac"], band["pr_frac"]),
        "e_mean_cos": _excess(raw["mean_cos"], band["mean_cos"]),
        "e_abs_cos": _excess(raw["abs_cos"], band["abs_cos"]),
        "e_kappa": _excess(raw["kappa"], band["kappa"]),
        "raw": {k: round(raw[k], 6) for k in ("pr_frac", "mean_cos", "abs_cos", "kappa")},
        "null_mean": {k: round(band[k]["mean"], 6) for k in band},
        "N": raw["N"], "d": raw["d"],
    }


# ------------------------------------------------------------------ LOFO gate
def lofo_accuracy(S, y):
    """Leave-one-family-out nearest-class-mean sign accuracy (prereg §3.1). Threshold
    AND direction are re-derived on the training families every fold — nothing about the
    held-out family enters its own prediction. A degenerate fold (train missing a class)
    scores that family wrong (the classifier is undefined there)."""
    S = np.asarray(S, float); y = np.asarray(y, int)
    n = S.size; correct = 0
    for i in range(n):
        tr = np.arange(n) != i
        Str, ytr = S[tr], y[tr]
        pos, neg = Str[ytr > 0], Str[ytr < 0]
        if pos.size == 0 or neg.size == 0:
            continue  # undefined -> counts as wrong
        thr = 0.5 * (pos.mean() + neg.mean())
        direction = np.sign(pos.mean() - neg.mean())
        if direction == 0:
            continue
        pred = 1 if direction * (S[i] - thr) > 0 else -1
        correct += int(pred == y[i])
    return correct / n


def separation_margin(S, y):
    """Signed in-sample separation margin: (min of the higher class) - (max of the lower
    class). >0 iff the statistic FULLY separates the two sign classes (prereg §3.1)."""
    S = np.asarray(S, float); y = np.asarray(y, int)
    pos, neg = S[y > 0], S[y < 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    if pos.mean() >= neg.mean():
        return float(pos.min() - neg.max())
    return float(neg.min() - pos.max())


def exact_perm_p(S, y, acc_obs):
    """Exact class-size-preserving label-permutation p for the LOFO accuracy: over all
    C(n, n_neg) ways to place the negatives, p = fraction of labelings whose LOFO accuracy
    >= observed (prereg §3.2). Floor = 1/C(n, n_neg)."""
    S = np.asarray(S, float); y = np.asarray(y, int)
    n = y.size; idx = np.arange(n)
    n_neg = int((y < 0).sum())
    ge = 0; total = 0
    for neg_pos in itertools.combinations(idx, n_neg):
        yp = np.ones(n, int); yp[list(neg_pos)] = -1
        total += 1
        if lofo_accuracy(S, yp) >= acc_obs - 1e-12:
            ge += 1
    return ge / total, total, 1.0 / total


def run_gate(fam_names, fam_signs, fam_stat):
    """Full pre-registered gate over the priority-ordered statistics. `fam_stat` maps
    family -> {z_stat: value}. Returns the gate block + verdict."""
    y = np.array([fam_signs[f] for f in fam_names], int)
    n = y.size
    n_pos, n_neg = int((y > 0).sum()), int((y < 0).sum())
    perm_floor = 1.0 / max(1, len(list(itertools.combinations(range(n), min(n_pos, n_neg)))))

    # Insufficient-power guard (prereg §3.3): need >=2 families in EACH class so every
    # LOFO training fold keeps both classes.
    if n < 4 or n_pos < 2 or n_neg < 2:
        return {
            "n_families": n, "n_pos": n_pos, "n_neg": n_neg,
            "verdict": "INSUFFICIENT_FAMILIES",
            "reason": f"need >=2 families per sign class for a LOFO gate; have "
                      f"{n_pos} POS / {n_neg} NEG. Descriptive statistics only; run the "
                      f"raw-K dump wave (prereg §2.3) to reach the gate.",
            "per_stat": [], "winner": None, "permutation_floor": round(perm_floor, 6),
        }, "INSUFFICIENT_FAMILIES"

    per_stat = []
    for key, desc in STAT_PRIORITY:
        S = np.array([fam_stat[f][key] for f in fam_names], float)
        acc = lofo_accuracy(S, y)
        margin = separation_margin(S, y)
        p, n_perm, floor = exact_perm_p(S, y, acc)
        per_stat.append({
            "stat": key, "desc": desc,
            "family_values": {f: round(float(v), 4) for f, v in zip(fam_names, S)},
            "lofo_accuracy": round(acc, 4),
            "separation_margin": (round(margin, 4) if margin == margin else None),
            "fully_separates": bool(margin == margin and margin > 0),
            "perm_p": round(p, 6), "n_perm": n_perm, "perm_floor": round(floor, 6),
        })

    # Multiplicity control (prereg §3.2): PRIMARY is a FIXED-SEQUENCE test — the four
    # statistics are tested in the a-priori priority order STAT_PRIORITY and the FIRST that
    # qualifies (full separation + LOFO==1.0) carries the gate at raw alpha=0.05; a fixed
    # a-priori order controls FWER without a Bonferroni penalty (Holm would over-correct: the
    # nearest-centroid classifier is sign-flip symmetric, so even an ideal BALANCED separator
    # has perm_p = 2/C(n,n_neg) and Holm/4 blocks it at the ideal). Holm columns are still
    # reported as a conservative SECONDARY read, but are NOT the gate.
    m = len(per_stat)
    order = sorted(range(m), key=lambda i: per_stat[i]["perm_p"])
    for rank, idx in enumerate(order):
        thr = 0.05 / (m - rank)
        per_stat[idx]["holm_threshold"] = round(thr, 6)
        per_stat[idx]["passes_holm"] = bool(per_stat[idx]["perm_p"] <= thr)

    # Winner = highest-PRIORITY statistic that fully separates AND LOFO-classifies every
    # family (prereg §3.3). Priority order is STAT_PRIORITY, not p-value order.
    winner = next((s for s in per_stat
                   if s["fully_separates"] and s["lofo_accuracy"] >= 1.0 - 1e-9), None)
    any_sep = any(s["fully_separates"] for s in per_stat)

    if winner is not None and winner["perm_p"] <= 0.05:
        verdict = "PASS"                      # fixed-sequence winner clears raw alpha=0.05
    elif winner is not None:
        verdict = "PASS_SEPARATION"           # full sep + full LOFO but perm_p > 0.05 (imbalance)
    elif any_sep:
        verdict = "AMBIGUOUS"                 # separates in-sample but LOFO leaks
    else:
        verdict = "KILL"                      # no key-space statistic separates the signs

    return {
        "n_families": n, "n_pos": n_pos, "n_neg": n_neg,
        "permutation_floor": round(perm_floor, 6),
        "per_stat": per_stat,
        "winner": (winner["stat"] if winner else None),
        "verdict": verdict,
    }, verdict


# ------------------------------------------------------------------- real run
def discover_banks(vecdir):
    """Map family tag -> sorted list of its cached raw-K bank paths (may be empty)."""
    fam = {}
    for tag in FAMILY_SIGN:
        pat = os.path.join(vecdir, f"vectors_qv_{tag}_rome_cf_L*_s*.npz")
        fam[tag] = sorted(glob.glob(pat))
    return fam


def run_real(n_rep, vecdir=VECDIR, outjson=OUTJSON):
    os.makedirs(os.path.dirname(os.path.abspath(outjson)), exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)
    banks = discover_banks(vecdir)

    per_family, fam_stat, fam_names = {}, {}, []
    for tag, paths in banks.items():
        if not paths:
            per_family[tag] = {"sign": FAMILY_SIGN[tag], "n_banks": 0,
                               "status": "ABSENT (no cached raw-K bank — awaits dump wave)"}
            continue
        rows = []
        for p in paths:
            K, meta = load_bank(p)
            sv = bank_stat_vector(K, n_rep, rng)
            sv["npz"] = os.path.basename(p)
            sv["layer"] = meta["layer"]; sv["seed"] = meta["seed"]
            sv["vectors_valid"] = meta["vectors_valid"]
            rows.append(sv)
        agg = {k: float(np.mean([r[k] for r in rows]))
               for k in ("e_pr_frac", "e_mean_cos", "e_abs_cos", "e_kappa")}
        fam_stat[tag] = agg
        fam_names.append(tag)
        per_family[tag] = {
            "sign": FAMILY_SIGN[tag], "n_banks": len(rows),
            "d_intermediate": sorted({r["d"] for r in rows}),
            "layers": sorted({r["layer"] for r in rows}),
            "seeds": sorted({r["seed"] for r in rows}),
            "single_seed": len({r["seed"] for r in rows}) == 1,
            "excess_aggregate": {k: round(v, 6) for k, v in agg.items()},
            "per_bank": [{"npz": r["npz"], "layer": r["layer"], "seed": r["seed"],
                          "N": r["N"], "d": r["d"], "vectors_valid": r["vectors_valid"],
                          "raw": r["raw"], "null_mean": r["null_mean"],
                          "excess": {k: round(r[k], 6) for k in
                                     ("e_pr_frac", "e_mean_cos", "e_abs_cos", "e_kappa")}}
                         for r in rows],
        }

    # Order the gate families by sign then name for a stable, readable table.
    fam_names = sorted(fam_names, key=lambda f: (FAMILY_SIGN[f], f))
    gate, verdict = run_gate(fam_names, FAMILY_SIGN, fam_stat)

    report = {
        "experiment": "D1_sign_inversion_keyspace_statistic_E0",
        "prereg": "docs/plans/PREREG-D1-SIGN-E0-20260714.md",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "rng_seed": RNG_SEED, "n_rep_isotropic_null": n_rep,
        "atlas_sign_table": FAMILY_SIGN,
        "families_used_in_gate": fam_names,
        "families_absent": [t for t in FAMILY_SIGN if not banks[t]],
        "statistic_definition": (
            "Each statistic is the EXCESS of a raw key-space quantity (pr_frac / mean_cos / "
            "|cos| / kurtosis) over its NORM-MATCHED ISOTROPIC-Gaussian null MEAN at that "
            "bank's own (N,d) — the 1/sqrt(d) floor subtracted off, removing the width "
            "(intermediate_size) and key-norm confounds (prereg §1.4). This does NOT close "
            "the lineage/tokenizer confound (see binding_wording). Not a z-score: the null "
            "STD is NOT divided out (it is d-dependent and ~0 at N<<d, which would reintroduce "
            "width and explode numerically). Per family = mean excess over its cached banks/seeds."),
        "per_family": per_family,
        "gate": gate,
        "verdict": verdict,
        "binding_wording": [
            "DOWNSCOPED CLAIM (review MAJOR-1): a PASS/PASS_SEPARATION means a key-space "
            "statistic SEPARATES the Qwen lineage from the Llama/Mistral/Gemma lineages, "
            "CONSISTENT WITH the sign split. It does NOT 'predict the family sign' in a "
            "generalizable sense and is NOT a mechanism (no intervention; see next item).",
            "LABEL EXCHANGEABILITY VIOLATED: the exact-permutation null assumes the sign "
            "labels are exchangeable, but ALL negatives are ONE correlated lineage (Qwen: "
            "shared architecture + tokenizer + training recipe). Effective negative-side n is "
            "~1 LINEAGE, not the negative FAMILY count, so the permutation p is optimistic and "
            "the LOFO 'held-out family' is really a held-out Qwen TIER, not an independent draw.",
            "HARD CEILING: a genuine negative-side generalization test needs a NON-Qwen "
            "negative family — which the atlas shows does not exist (every negative family is "
            "Qwen). The balancing dump (3 more Qwen negatives) only improves the permutation "
            "FLOOR; it does NOT relieve this ceiling. So the strongest admissible result here "
            "is a lineage-separation / sign-consistency statement, never family-sign prediction.",
            "KILL is the explicit 'sign is LINEAGE-determined with no lower-dimensional "
            "key-space summary' outcome (prereg §3.3) -> ship the sign atlas as a "
            "descriptive taxonomy, no mechanism claim.",
            "LINEAGE confound (architecture + tokenizer + training recipe), NOT closed by the "
            "isotropic-excess deconfound (that removes width/norm only): a statistic that "
            "merely proxies 'is-Qwen-lineage' is indistinguishable here from a key-geometry "
            "mechanism. The prereg §4.2 shared-single-token-subject control addresses only the "
            "tokenizer facet; architecture/recipe cannot be separated without a non-Qwen "
            "negative family. Run the control before ANY mechanism-flavored wording.",
            "Embedding-tying is PRE-KILLED as the sign driver (prereg §4.1): tied models "
            "sit on BOTH sign sides (Llama-3.2-1B/gemma-2 tied & POSITIVE; Qwen tied & "
            "NEGATIVE; Llama-3.1-8B untied & POSITIVE).",
            "Class balance sets the exact permutation p of a perfect separator (fixed-"
            "sequence, raw alpha=0.05): with 4 POS / 2 NEG only the true partition "
            "separates -> p=1/15=0.067>0.05 -> at best PASS_SEPARATION; dumping the 3 "
            "balancing Qwen negatives (4 POS / 4 NEG) drops it to p=2/70=0.029<=0.05 "
            "(the factor 2 is the sign-flip mirror of the nearest-centroid rule). NB this "
            "lowers only the FLOOR; the exchangeability ceiling above still caps the claim.",
            "Most families are single-seed (s0) raw-K; per-family excess rests on point "
            "estimates except llama1b(L14)/qwen15b(L14) which carry 3 seeds.",
        ],
    }
    with open(outjson + ".tmp", "w") as fh:
        json.dump(report, fh, indent=2)
    os.replace(outjson + ".tmp", outjson)
    print_table(report, outjson)
    return report


def print_table(report, outjson=OUTJSON):
    g = report["gate"]
    print(f"\n=== D1 sign-inversion key-space E0 ({report['created']}) ===")
    print(f"families in gate: {report['families_used_in_gate']}")
    print(f"families absent : {report['families_absent']}")
    print(f"\n{'family':>14} {'sign':>5} {'nB':>3} {'e_abs_cos':>10} {'e_pr_frac':>10} "
          f"{'e_kappa':>9} {'e_mean_cos':>11}")
    for f in report["families_used_in_gate"]:
        pf = report["per_family"][f]; e = pf["excess_aggregate"]
        print(f"{f:>14} {pf['sign']:>+5d} {pf['n_banks']:>3} {e['e_abs_cos']:>10.4f} "
              f"{e['e_pr_frac']:>10.4f} {e['e_kappa']:>9.4f} {e['e_mean_cos']:>11.4f}")
    print(f"\n-- LOFO sign-prediction gate (n_pos={g.get('n_pos')} n_neg={g.get('n_neg')} "
          f"perm_floor={g.get('permutation_floor')}) --")
    for s in g.get("per_stat", []):
        print(f"  {s['stat']:>11}: LOFO={s['lofo_accuracy']:.3f}  "
              f"sep={s['separation_margin']}  fully_sep={s['fully_separates']}  "
              f"perm_p={s['perm_p']:.4f}  holm_pass={s.get('passes_holm')}")
    print(f"\n  winner  = {g.get('winner')}")
    print(f"  VERDICT = {report['verdict']}")
    print(f"  report -> {os.path.relpath(outjson, HARNESS)}")


# --------------------------------------------------------------------- selftest
def _synth_bank(n, d, aniso, rng):
    """Reuse the planted-anisotropy generator from depth_dissoc_sketch (imported lazily so
    a missing symbol never breaks import). aniso in [0,1]: 0 ~ isotropic, ->1 = cone +
    heavy tail. Raises signed & |cos| & kurtosis, lowers PR — the exact statistics."""
    from depth_dissoc_sketch import _synth_bank as gen
    return gen(n, d, aniso, rng)


def _fixture(fam_aniso, fam_d, rng, n_rep):
    """Build a per-family z-stat table from planted banks (2 s0 banks per family).
    Returns (names, signs, fam_stat)."""
    names, signs, fam_stat = [], {}, {}
    for tag, (aniso, sign) in fam_aniso.items():
        d = fam_d[tag]
        svs = []
        for _ in range(2):  # 2 synthetic layers/banks per family
            K = _synth_bank(200, d, aniso, rng)
            svs.append(bank_stat_vector(K, n_rep, rng))
        names.append(tag); signs[tag] = sign
        fam_stat[tag] = {k: float(np.mean([s[k] for s in svs]))
                         for k in ("e_pr_frac", "e_mean_cos", "e_abs_cos", "e_kappa")}
    names = sorted(names, key=lambda f: (signs[f], f))
    return names, signs, fam_stat


def run_selftest():
    """Three provable fixtures, no real IO — one per gate tier the real data can hit:
      PASS         — BALANCED 4 POS / 4 NEG, POS planted high-anisotropy, NEG near-isotropic,
                     DIFFERENT widths d per family (proves the excess deconfound). A separating
                     statistic must fully separate + LOFO 8/8 + perm_p 2/70=0.029<=0.05 -> PASS.
      PASS_SEPARATION — IMBALANCED 4 POS / 2 NEG (the ACTUAL post-dump-wave shape), same planted
                     separation. A separating statistic fully separates + LOFO 6/6 but perm_p=
                     1/15=0.067>0.05 -> PASS_SEPARATION (never a strict PASS). This is the tier
                     the real gate will report.
      KILL         — 4/4 but every family at the SAME middling anisotropy (label is noise) ->
                     no statistic separates -> verdict KILL/AMBIGUOUS (never PASS)."""
    rng = np.random.default_rng(20260714)
    n_rep = 40  # small null band for speed; separation is large by construction
    widths = {"pA": 4096, "pB": 8192, "pC": 14336, "pD": 8960,
              "nA": 4864, "nB": 11008, "nC": 3584, "nD": 5120}
    # Tight within-class clusters with a large gap so a genuinely-separating statistic scores a
    # perfect LOFO under the nearest-centroid rule (a wide-spread class can lose its extreme
    # held-out member to the midpoint threshold — the honest failure mode).
    sep_aniso = {"pA": (0.80, +1), "pB": (0.82, +1), "pC": (0.78, +1), "pD": (0.81, +1),
                 "nA": (0.03, -1), "nB": (0.05, -1), "nC": (0.04, -1), "nD": (0.06, -1)}

    print("\n=== SELFTEST fixture 1: PASS (BALANCED 4/4, planted anisotropy, mixed widths) ===")
    names, signs, fam_stat = _fixture(sep_aniso, widths, rng, n_rep)
    g1, v1 = run_gate(names, signs, fam_stat)
    for s in g1["per_stat"]:
        print(f"  {s['stat']:>11}: LOFO={s['lofo_accuracy']:.3f} fully_sep={s['fully_separates']} "
              f"perm_p={s['perm_p']:.4f} holm_pass={s.get('passes_holm')}")
    print(f"  winner={g1['winner']}  verdict={v1}  perm_floor={g1['permutation_floor']}")
    ok_pass = (v1 == "PASS" and g1["winner"] is not None)

    print("\n=== SELFTEST fixture 2: PASS_SEPARATION (IMBALANCED 4 POS / 2 NEG — real shape) ===")
    imb_aniso = {k: v for k, v in sep_aniso.items() if k not in ("nC", "nD")}  # drop 2 negatives
    names_i, signs_i, fam_stat_i = _fixture(imb_aniso, widths, rng, n_rep)
    gI, vI = run_gate(names_i, signs_i, fam_stat_i)
    for s in gI["per_stat"]:
        print(f"  {s['stat']:>11}: LOFO={s['lofo_accuracy']:.3f} fully_sep={s['fully_separates']} "
              f"perm_p={s['perm_p']:.4f}")
    print(f"  winner={gI['winner']}  verdict={vI}  perm_floor={gI['permutation_floor']}")
    ok_sep = (vI == "PASS_SEPARATION" and gI["winner"] is not None)

    print("\n=== SELFTEST fixture 3: KILL (label is noise, same anisotropy everywhere) ===")
    null_aniso = {"pA": (0.30, +1), "pB": (0.32, +1), "pC": (0.28, +1), "pD": (0.31, +1),
                  "nA": (0.29, -1), "nB": (0.30, -1), "nC": (0.31, -1), "nD": (0.30, -1)}
    names2, signs2, fam_stat2 = _fixture(null_aniso, widths, rng, n_rep)
    g2, v2 = run_gate(names2, signs2, fam_stat2)
    for s in g2["per_stat"]:
        print(f"  {s['stat']:>11}: LOFO={s['lofo_accuracy']:.3f} fully_sep={s['fully_separates']} "
              f"perm_p={s['perm_p']:.4f}")
    print(f"  winner={g2['winner']}  verdict={v2}")
    ok_kill = (v2 in ("KILL", "AMBIGUOUS") and g2["winner"] is None)

    all_ok = ok_pass and ok_sep and ok_kill
    print(f"\nSELFTEST {'PASS' if all_ok else 'FAIL'} — "
          f"fixture1={'PASS' if ok_pass else 'FAIL(expected PASS)'} / "
          f"fixture2={'PASS' if ok_sep else 'FAIL(expected PASS_SEPARATION)'} / "
          f"fixture3={'PASS' if ok_kill else 'FAIL(expected KILL/AMBIGUOUS)'}")
    return 0 if all_ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true", help="synthetic self-test (no real IO)")
    ap.add_argument("--n_rep", type=int, default=50,
                    help="isotropic-null resamples per bank for the excess floor (real run; "
                         "the null MEAN is a stable population quantity so ~40-60 suffices — "
                         "200 was overkill, ~tens of minutes over the full bank set)")
    ap.add_argument("--vector_dir", default=VECDIR,
                    help="dir holding the raw-K banks (default results/vectors; point at the "
                         "ON-BOX vectors dir to analyse before the JSON is pulled home)")
    ap.add_argument("--out", default=OUTJSON,
                    help="output JSON path (default results/analysis/D1_sign_stat_table.json)")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(run_selftest())
    run_real(args.n_rep, args.vector_dir, args.out)


if __name__ == "__main__":
    main()
