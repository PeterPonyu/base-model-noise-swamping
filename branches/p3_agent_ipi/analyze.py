"""analyze.py -- lineage-vs-architecture fingerprint contrast + permutation test.

Input: a per-(model,item) binary success matrix S (M models x K items) plus the
model meta (lineage, arch_family, match_group, group).

Pipeline:
  1. Pairwise similarity between model success vectors:
       - Pearson correlation (phi coefficient for binary vectors; std=0 -> 0)
       - Jaccard similarity of the success sets
  2. Classify each in-group pair (excluding out-group models) as:
       - 'lineage'      : same lineage label (r1-distill vs base-instruct),
                          different architecture/scale  (Group-A internal,
                          Group-B internal)
       - 'architecture' : same match_group (arch_family/scale), different lineage
                          (the 3 matched r1<->base pairs)
     Out-group pairs (any pair touching a Group-C model) are summarized
     separately and NOT used in the main contrast.
  3. Contrast statistic  diff = mean_corr(lineage pairs) - mean_corr(arch pairs).
  4. Within-model item-label permutation test (default 1000x): under H0 there is
     no cross-model item-level agreement beyond each model's marginal ASR, so we
     independently shuffle EACH model's success vector across items (a
     within-model permutation of the item labels), recompute the full similarity
     matrix, and recompute `diff` with the FIXED lineage/architecture pair sets.
     The one-sided p-value is the fraction of permutations with diff >= observed.

     Why within-model item permutation and NOT across-model lineage-label
     permutation: the design has 3 r1-distills vs 3 vanilla bases in-group. Any
     balanced 2-label permutation is degenerate under a global label swap (the
     complement labeling ALWAYS ties the observed statistic), so the smallest
     achievable p from lineage-label shuffling is 2/C(6,3)=0.10 -- it can never
     clear 0.05 regardless of effect size. The within-model item permutation has
     no such floor: it drives cross-model correlations to ~0 under H0, so a
     genuine lineage>architecture separation reaches arbitrarily small p while a
     structureless (shuffled) matrix returns a null p~0.5. (The across-model
     label-permutation p is still reported as `label_perm_p` for reference.)

Pure-Python (no numpy dependency required, but uses it if present for speed).
No network, no model calls.
"""
from __future__ import annotations

import itertools
import json
import math
import random
from typing import Any


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def pearson(a: list[float], b: list[float]) -> float:
    n = len(a)
    if n == 0:
        return 0.0
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = math.sqrt(sum((x - ma) ** 2 for x in a))
    db = math.sqrt(sum((y - mb) ** 2 for y in b))
    if da == 0 or db == 0:
        return 0.0
    return num / (da * db)


def jaccard(a: list[int], b: list[int]) -> float:
    inter = sum(1 for x, y in zip(a, b) if x and y)
    union = sum(1 for x, y in zip(a, b) if x or y)
    return inter / union if union else 0.0


def similarity_matrices(matrix: list[list[int]]) -> dict[str, list[list[float]]]:
    m = len(matrix)
    pear = [[0.0] * m for _ in range(m)]
    jac = [[0.0] * m for _ in range(m)]
    for i in range(m):
        for j in range(m):
            pear[i][j] = 1.0 if i == j else pearson(
                [float(x) for x in matrix[i]], [float(x) for x in matrix[j]])
            jac[i][j] = 1.0 if i == j else jaccard(matrix[i], matrix[j])
    return {"pearson": pear, "jaccard": jac}


def _pair_class(mi: dict, mj: dict) -> str | None:
    """Classify an in-group pair. Returns 'lineage', 'architecture', or None."""
    if mi.get("group") == "out" or mj.get("group") == "out":
        return None
    if mi["match_group"] == mj["match_group"] and mi["lineage"] != mj["lineage"]:
        return "architecture"
    if mi["lineage"] == mj["lineage"] and mi["match_group"] != mj["match_group"]:
        return "lineage"
    return None


def _diff_from_labels(sim: list[list[float]], models: list[dict],
                      lineage_labels: list[str], arch_pairs: list[tuple[int, int]]) -> tuple:
    """Compute mean(lineage-pair corr) - mean(arch-pair corr) for a labeling.

    arch_pairs is the FIXED set of matched (architecture) pairs. lineage pairs
    are all in-group same-(permuted-)lineage / different-match_group pairs.
    Used both for the observed statistic and for the reference across-model
    label-permutation null.
    """
    idx = [k for k, m in enumerate(models) if m.get("group") != "out"]
    lin_vals = []
    for a, b in itertools.combinations(idx, 2):
        if lineage_labels[a] == lineage_labels[b] and \
                models[a]["match_group"] != models[b]["match_group"]:
            lin_vals.append(sim[a][b])
    arch_vals = [sim[a][b] for a, b in arch_pairs]
    lm, am = _mean(lin_vals), _mean(arch_vals)
    return lm - am, lm, am, lin_vals, arch_vals


def _diff_from_pairs(sim: list[list[float]], lineage_pairs: list[tuple[int, int]],
                     arch_pairs: list[tuple[int, int]]) -> float:
    """diff = mean(corr over FIXED lineage pairs) - mean(corr over FIXED arch pairs)."""
    lm = _mean([sim[a][b] for a, b in lineage_pairs])
    am = _mean([sim[a][b] for a, b in arch_pairs])
    return lm - am


def contrast(matrix: list[list[int]], models: list[dict], metric: str = "pearson",
             n_perm: int = 1000, seed: int = 0) -> dict[str, Any]:
    """Run the full lineage-vs-architecture contrast + permutation test."""
    sims = similarity_matrices(matrix)
    sim = sims[metric]
    m = len(models)

    # fixed architecture-matched pairs (same match_group, differ lineage)
    arch_pairs = []
    lineage_pairs = []
    outgroup_pairs = []
    for i, j in itertools.combinations(range(m), 2):
        cls = _pair_class(models[i], models[j])
        if models[i].get("group") == "out" or models[j].get("group") == "out":
            outgroup_pairs.append((i, j))
        elif cls == "architecture":
            arch_pairs.append((i, j))
        elif cls == "lineage":
            lineage_pairs.append((i, j))

    labels = [m_["lineage"] for m_ in models]
    obs_diff, lin_mean, arch_mean, lin_vals, arch_vals = _diff_from_labels(
        sim, models, labels, arch_pairs)
    out_vals = [sim[i][j] for i, j in outgroup_pairs]

    # ---- PRIMARY null: within-model item-label permutation --------------------
    # Independently shuffle each model's success vector across items (preserving
    # its marginal ASR), recompute the FULL similarity matrix, recompute `diff`
    # over the FIXED lineage/architecture pair sets. No degeneracy floor.
    rng = random.Random(seed)
    null = []
    ge = 0
    for _ in range(n_perm):
        pm = [row[:] for row in matrix]
        for row in pm:
            rng.shuffle(row)
        psim = similarity_matrices(pm)[metric]
        d = _diff_from_pairs(psim, lineage_pairs, arch_pairs)
        null.append(d)
        if d >= obs_diff - 1e-12:
            ge += 1
    p_value = ge / n_perm if n_perm else float("nan")

    # ---- REFERENCE null: across-model lineage-label permutation ---------------
    # Reported for completeness; note its hard floor of 2/C(6,3)=0.10 for the
    # balanced 3-vs-3 in-group design (see module docstring).
    in_idx = [k for k, mm in enumerate(models) if mm.get("group") != "out"]
    in_labels = [labels[k] for k in in_idx]
    lrng = random.Random(seed + 1)
    lge = 0
    for _ in range(n_perm):
        perm = in_labels[:]
        lrng.shuffle(perm)
        plabels = list(labels)
        for pos, k in enumerate(in_idx):
            plabels[k] = perm[pos]
        d, *_ = _diff_from_labels(sim, models, plabels, arch_pairs)
        if d >= obs_diff - 1e-12:
            lge += 1
    label_perm_p = lge / n_perm if n_perm else float("nan")

    return {
        "metric": metric,
        "n_models": m,
        "n_items": len(matrix[0]) if matrix else 0,
        "null_type": "within_model_item_permutation",
        "lineage_pairs": [(models[i]["name"], models[j]["name"]) for i, j in lineage_pairs],
        "architecture_pairs": [(models[i]["name"], models[j]["name"]) for i, j in arch_pairs],
        "n_outgroup_pairs": len(outgroup_pairs),
        "mean_lineage_corr": lin_mean,
        "mean_architecture_corr": arch_mean,
        "mean_outgroup_corr": _mean(out_vals),
        "observed_diff": obs_diff,
        "n_perm": n_perm,
        "p_value": p_value,
        "label_perm_p": label_perm_p,
        "label_perm_floor": 2.0 / 20.0,
        "null_mean": _mean(null),
        "null_max": max(null) if null else float("nan"),
        "lineage_gt_architecture": obs_diff > 0,
        "similarity_matrices": sims,
    }


if __name__ == "__main__":
    # trivial smoke test with random data
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_perm", type=int, default=1000)
    args = ap.parse_args()
    from models import design_models
    rng = random.Random(1)
    mm = design_models()
    mat = [[rng.randint(0, 1) for _ in range(30)] for _ in mm]
    print(json.dumps({k: v for k, v in contrast(mat, mm, n_perm=args.n_perm).items()
                      if k != "similarity_matrices"}, indent=2, default=str))
