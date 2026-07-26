"""geometry_router.py — D3 geometry-gated editor routing evaluation.

Turns the (architecture-conditioned) key-geometry law into a DEPLOYABLE routing
policy and scores it on the matrices we already have on disk. PURE CPU / numpy —
NO torch, NO model load, NO network. Safe to run while the GPU is busy with the
keystone.

Routing policy (per model/layer config), gated on PRE-EDIT geometry only:
  mean_cos = mean over the (known/edit-ok masked) edit-probe pairs of |COS|.
    - mean_cos >= --cos_threshold  (Llama-like, high-cosine)   -> route to AlphaEdit
        (the null-space projection buys real collateral protection here).
    - mean_cos <  --cos_threshold  (Qwen-like, near-orthogonal) -> route to vanilla
        ROME and SKIP the null-space projection  (projection-compute SAVED — the
        geometry says there is nothing to protect).

For each config we report mean_cos, the routing decision, whether the projection
was skipped, and — where a matched AlphaEdit matrix exists on the SAME
model/layer/seed — the expected mean collateral damage under routing vs.
always-ROME and always-AlphaEdit, on the SHARED (both-editors) masks so the three
numbers are directly comparable. Where no matched alpha matrix exists we emit the
routing DECISION only (no damage delta), as the spec requires.

Metric discipline: mean SIGNED damage (damage_logit), never AUROC. All new/optional
npz keys are guarded with `if k in d.files` per the analyze_g4 pattern.

Usage:
  python geometry_router.py \
      --gate_glob 'results/matrices/gate_*_rome_cf_*_s0.npz' \
      --alpha_glob 'results/matrices/g4_llama1b_alpha_cf_L*_s0.npz' \
      --cos_threshold 0.05 --known --edit_ok --out results/D3_routing_eval.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np

# gate_{model}_rome_{dataset}_L{layer}_s{seed}.npz  (model tags carry no underscore)
GATE_RE = re.compile(r"gate_(?P<model>.+?)_rome_(?P<dataset>[a-z0-9]+)_L(?P<layer>\d+)_s(?P<seed>\d+)\.npz$")
# g4_{model}_alpha_{dataset}_L{layer}_s{seed}.npz
ALPHA_RE = re.compile(r"g4_(?P<model>.+?)_alpha_(?P<dataset>[a-z0-9]+)_L(?P<layer>\d+)_s(?P<seed>\d+)\.npz$")


def parse_name(path, rx):
    m = rx.search(os.path.basename(path))
    if not m:
        return None
    g = m.groupdict()
    return (g["model"], g["dataset"], int(g["layer"]), int(g["seed"]))


def masks(d, known, edit_ok):
    """Row (edit) and column (probe) boolean masks for a single npz."""
    COS = d["COS"].astype(float)
    row = np.ones(COS.shape[0], bool)
    if edit_ok and "edit_ok" in d.files:
        row = d["edit_ok"].astype(float) > 0.5
    col = np.ones(COS.shape[1], bool)
    if known and "pre_p" in d.files:
        c = d["pre_p"].astype(float) > 0.05
        if c.sum() >= 5:
            col = c
    return row, col


def masked_mean(A, row, col):
    sub = A[row][:, col]
    sub = sub[np.isfinite(sub)]
    return float(sub.mean()) if sub.size else float("nan")


def r4(x):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), 5)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gate_glob", required=True, help="glob of ROME gate .npz (one per config; --known/--edit_ok masks applied)")
    ap.add_argument("--alpha_glob", default=None, help="glob of matched AlphaEdit .npz (g4_..._alpha_...)")
    ap.add_argument("--cos_threshold", type=float, default=0.05,
                    help="mean|COS| >= threshold -> AlphaEdit; below -> vanilla ROME (skip projection)")
    ap.add_argument("--known", action="store_true", help="restrict to probes the base model knows (pre_p>0.05)")
    ap.add_argument("--edit_ok", action="store_true", help="restrict to edits that succeeded (edit_ok>0.5)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    gate_paths = sorted(set(glob.glob(args.gate_glob)))
    if not gate_paths:
        raise SystemExit(f"no gate .npz matched: {args.gate_glob}")

    # index matched alpha matrices by (model,dataset,layer,seed)
    alpha_index = {}
    if args.alpha_glob:
        for p in sorted(set(glob.glob(args.alpha_glob))):
            key = parse_name(p, ALPHA_RE)
            if key is not None:
                alpha_index[key] = p

    rows = []
    for gp in gate_paths:
        key = parse_name(gp, GATE_RE)
        if key is None:
            # not a rome gate matrix (e.g. an ft matrix caught by a loose glob) -> skip
            continue
        model, dataset, layer, seed = key
        dg = np.load(gp)
        rmask, cmask = masks(dg, args.known, args.edit_ok)
        COS = dg["COS"].astype(float)
        mean_cos = masked_mean(np.abs(COS), rmask, cmask)

        route_alpha = np.isfinite(mean_cos) and (mean_cos >= args.cos_threshold)
        decision = ("AlphaEdit (high-cos: apply null-space projection)"
                    if route_alpha else
                    "ROME (low-cos: SKIP null-space projection -> compute saved)")
        projection_skipped = (not route_alpha)

        row = {
            "config": f"{model}_{dataset}_L{layer}_s{seed}",
            "model": model, "dataset": dataset, "layer": layer, "seed": seed,
            "gate_npz": os.path.basename(gp),
            "n_edits_masked": int(rmask.sum()), "n_probes_masked": int(cmask.sum()),
            "mean_abs_cos": r4(mean_cos),
            "cos_threshold": args.cos_threshold,
            "routing_decision": decision,
            "projection_skipped": bool(projection_skipped),
        }

        ap_path = alpha_index.get(key)
        if ap_path is not None:
            da = np.load(ap_path)
            Dr = dg["damage_logit"].astype(float)
            Da = da["damage_logit"].astype(float)
            if COS.shape == Dr.shape == Da.shape:
                # SHARED masks (analyze_g4 convention): both editors must succeed on an edit;
                # probe-known column mask comes from the base-model pre_p (editor-invariant).
                srow = rmask.copy()
                if args.edit_ok and "edit_ok" in da.files:
                    srow = srow & (da["edit_ok"].astype(float) > 0.5)
                always_rome = masked_mean(Dr, srow, cmask)
                always_alpha = masked_mean(Da, srow, cmask)
                routed = always_alpha if route_alpha else always_rome
                row.update({
                    "alpha_available": True,
                    "alpha_npz": os.path.basename(ap_path),
                    "n_edits_shared": int(srow.sum()),
                    "always_rome_mean_damage": r4(always_rome),
                    "always_alpha_mean_damage": r4(always_alpha),
                    "routed_mean_damage": r4(routed),
                    "routing_delta_vs_always_rome": r4(
                        (routed - always_rome) if np.isfinite(routed) and np.isfinite(always_rome) else None),
                    "routing_delta_vs_always_alpha": r4(
                        (routed - always_alpha) if np.isfinite(routed) and np.isfinite(always_alpha) else None),
                })
            else:
                row.update({"alpha_available": False,
                            "note": f"alpha shape mismatch {COS.shape}/{Dr.shape}/{Da.shape} — decision only"})
        else:
            row.update({"alpha_available": False, "note": "no matched alpha npz — routing decision only"})

        rows.append(row)

    # ---- summary over configs ----
    n_alpha = sum(1 for r in rows if r["routing_decision"].startswith("AlphaEdit"))
    n_rome = sum(1 for r in rows if r["routing_decision"].startswith("ROME"))
    with_alpha = [r for r in rows if r.get("alpha_available")]

    def _mean(vals):
        vals = [v for v in vals if v is not None and np.isfinite(v)]
        return round(float(np.mean(vals)), 5) if vals else None

    summary = {
        "n_configs": len(rows),
        "cos_threshold": args.cos_threshold,
        "filters": {"known": args.known, "edit_ok": args.edit_ok},
        "n_routed_alphaedit": n_alpha,
        "n_routed_rome_projection_saved": n_rome,   # configs where projection compute is skipped
        "n_configs_with_matched_alpha": len(with_alpha),
        "expected_mean_damage_under_routing": _mean([r.get("routed_mean_damage") for r in with_alpha]),
        "expected_mean_damage_always_rome": _mean([r.get("always_rome_mean_damage") for r in with_alpha]),
        "expected_mean_damage_always_alpha": _mean([r.get("always_alpha_mean_damage") for r in with_alpha]),
    }
    res = {"summary": summary, "by_config": rows}
    print(json.dumps(res, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(res, open(args.out, "w"), indent=2)
        print(f"[route] wrote {args.out}")


if __name__ == "__main__":
    main()
