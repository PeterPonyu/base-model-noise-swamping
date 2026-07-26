"""mechanism_dump.py — per-edit ROME S-factor (mechanism) dump.

Materializes the *edit-strength* factor of the closed-form ROME S x C decomposition.
For the rank-one update  DW = (v - W k) k^T / (k^T k),  the change a single edit
imposes on any probe key k_p is

    DW k_p = (v - W k) * (k . k_p) / (k . k)
           = (v - W k) * ||k_p|| cos(k, k_p) / ||k||
           = S * (||k_p|| * cos)          with   S = ||v - W k|| / ||k||.

So per edit the scalar

    S = ||v - W k|| / ||k||   (residual_norm / key_norm)

is the model/edit-specific magnitude that multiplies the geometric factor C = cos.
This script runs the native ROME editor over a bank of edits (weights RESTORED
after each edit, exactly like killgate), captures ``residual_norm`` (||v-Wk||) and
``key_norm`` (||k||) already returned by ``editors.rome_native.apply_edit``, and
dumps the per-edit arrays plus a small mean/median-S summary, one file per
(model, layer):

    results/mechanism/<model>_L<layer>.npz     # per-edit arrays
    results/mechanism/<model>_L<layer>.json    # mean/median S summary

The claim to materialize downstream (C1 mechanism receipt): Llama S ~= 22.9 (large)
vs Qwen S small (predicted 4-8x smaller) -> damage tracks residual NORM (S), not raw
key orthogonality.

CPU-only validation is possible on a tiny-random Llama (``--device cpu``); GPU runs
use fp32 (``--device cuda``), never fp16 (ROME value-opt NaNs in fp16).

``--save_vectors`` (additive, default OFF, 2026-07-06): also persists the per-edit
residual VECTOR ``r_e = v - Wk`` (dim hidden_size) as npz key ``resid_vecs`` [N, hidden]
-- feeds ``experiments/gradsim_true.py``'s true-backprop influence check. The npz is
byte-identical to the pre-existing schema when the flag is omitted.

Usage:
  python experiments/mechanism_dump.py --model data/models/Qwen2.5-0.5B \
      --dataset counterfact --data data/counterfact.json \
      --n_edits 200 --layer 12 --seed 0 --steps 20 --lr 0.1 \
      --device cuda --out_dir results/mechanism
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

import numpy as np
import torch

HARNESS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HARNESS)
# reuse the SAME edit loaders as killgate so edit ordering aligns 1:1 with the gate
# npz for a given (dataset, seed, n_edits) -> the per-edit S here indexes the same edits.
from experiments.killgate_keygeom import load_counterfact, load_zsre  # noqa: E402
from metrics import efficacy  # noqa: E402
from editors.rome_native import apply_edit  # noqa: E402


def model_tag(model_path: str) -> str:
    """Filesystem-safe model name for the per-model/per-layer dump filename."""
    return os.path.basename(os.path.normpath(model_path))


def main():
    ap = argparse.ArgumentParser(description="Per-edit ROME S = ||v-Wk||/||k|| dump.")
    ap.add_argument("--model", required=True, help="local model dir (0-download)")
    ap.add_argument("--data", required=True, help="counterfact.json or zsre_eval.json")
    ap.add_argument("--dataset", choices=["counterfact", "zsre"], default="counterfact")
    ap.add_argument("--n_edits", type=int, default=200)
    ap.add_argument("--n_probes", type=int, default=0,
                    help="only affects which slice becomes 'edits' via the shared loader; "
                         "S is computed on the edit bank only")
    ap.add_argument("--layer", default="auto")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--v_weight_decay", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda", help="cuda for real runs; cpu for tiny-random validation")
    ap.add_argument("--out_dir", default=os.path.join(HARNESS, "results", "mechanism"))
    ap.add_argument("--save_vectors", action="store_true",
                    help="ADDITIVE (2026-07-06): also persist the per-edit residual VECTOR "
                         "r_e = v - Wk (dim hidden_size, the down_proj OUTPUT dim -- NOT the "
                         "intermediate/key dim key_norm lives in) as npz key 'resid_vecs' "
                         "[N, hidden]. Feeds experiments/gradsim_true.py's true-backprop "
                         "influence check. Default OFF: the npz is byte-identical to the "
                         "pre-existing schema without this flag.")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    # fp32 ALWAYS: the ROME value-optimization (Adam + log_softmax) silently NaNs in fp16.
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(args.device).eval()
    nL = model.config.num_hidden_layers
    layer = nL // 2 if args.layer == "auto" else int(args.layer)
    print(f"[mech] loaded {args.model} ({nL} layers, edit layer={layer}) "
          f"dev={args.device} {time.time()-t0:.1f}s", flush=True)

    load_fn = load_counterfact if args.dataset == "counterfact" else load_zsre
    # reuse killgate's (n_edits + n_probes) slicing so ordering matches the gate npz.
    # NB: the shared loader grew a 3rd return (E6 holdout bank) — absorb extras so this
    # script tracks future loader returns (API-drift bug caught by run8h 2026-07-02).
    edits, _probes, *_ = load_fn(args.data, args.n_edits, args.n_probes, args.seed)
    print(f"[mech] {args.dataset}: {len(edits)} edits (seed {args.seed})", flush=True)

    # snapshot the editable weight for fast restore (weights RESTORED after each edit).
    W = model.model.layers[layer].mlp.down_proj.weight
    W_base = W.detach().clone()

    N = len(edits)
    resid_norm = np.full(N, np.nan)   # ||v - W k||
    key_norm = np.full(N, np.nan)     # ||k||
    S = np.full(N, np.nan)            # ||v - W k|| / ||k||
    norm_growth = np.full(N, np.nan)  # ||DW|| (ENCORE baseline)
    value_norm = np.full(N, np.nan)   # ||v||
    solve_resid = np.full(N, np.nan)  # ||(W+DW)k - v||  (~0 confirms rank-one solve)
    edit_ok = np.zeros(N)             # per-edit efficacy (argmax == new target)
    # always a concrete list (never None) so the conditional .append() below has a stable
    # static type; stays empty and unused when --save_vectors is off.
    resid_vec_list: list[np.ndarray] = []

    cfg = {"layer": layer, "steps": args.steps, "lr": args.lr,
           "v_weight_decay": args.v_weight_decay}
    for i, e in enumerate(edits):
        info = apply_edit(model, tok, e, cfg, args.device)
        kn = float(info.get("key_norm", np.nan))
        rn = float(info.get("residual_norm", np.nan))
        key_norm[i] = kn
        resid_norm[i] = rn
        S[i] = rn / kn if (np.isfinite(kn) and kn > 0) else np.nan
        ng = info.get("delta_weight_norm", np.nan)
        norm_growth[i] = float(ng[layer]) if isinstance(ng, dict) else float(ng)
        value_norm[i] = float(info.get("value_norm", np.nan))
        solve_resid[i] = float(info.get("rank_one_solve_residual", np.nan))
        eff = efficacy(model, tok, e["prompt"], e["target_new"], e.get("target_true"), args.device)
        edit_ok[i] = eff["success"]
        if args.save_vectors:
            resid_vec_list.append(np.asarray(info["residual_vec"], dtype=np.float32))
        with torch.no_grad():
            W.copy_(W_base)  # restore -> every edit sees the base-model weight
        if (i + 1) % 10 == 0:
            print(f"[mech] edit {i+1}/{N}  {time.time()-t0:.1f}s", flush=True)

    def _stat(fn, arr):
        a = arr[np.isfinite(arr)]
        return None if a.size == 0 else round(float(fn(a)), 6)

    tag = model_tag(args.model)
    os.makedirs(args.out_dir, exist_ok=True)
    npz_path = os.path.join(args.out_dir, f"{tag}_L{layer}.npz")
    # dict[str, Any], not dict[str, ndarray]: np.savez_compressed's real signature is
    # (file, *args, allow_pickle=True, **kwds) -- a precisely-typed dict[str, ndarray]
    # splat makes pyright worry the dict could supply an ndarray for the allow_pickle:
    # bool slot (reportArgumentType). Any sidesteps that false positive; it's also
    # honest, since additive fields elsewhere in the harness aren't always ndarray
    # (e.g. killgate_keygeom.py's np.array(..., dtype="U16") string-scalar fields).
    arrs: dict[str, Any] = dict(
        resid_norm=resid_norm.astype(np.float32),   # [N] ||v - W k||
        key_norm=key_norm.astype(np.float32),        # [N] ||k||
        S=S.astype(np.float32),                      # [N] ||v-Wk|| / ||k||   (the mechanism factor)
        norm_growth=norm_growth.astype(np.float32),  # [N] ||DW||
        value_norm=value_norm.astype(np.float32),    # [N] ||v||
        solve_resid=solve_resid.astype(np.float32),  # [N] rank-one solve residual (~0)
        edit_ok=edit_ok.astype(np.float32),          # [N] per-edit efficacy
    )
    # ADDITIVE (--save_vectors only): [N, hidden] residual VECTORS r_e = v-Wk. hidden dim,
    # matching g_p = grad_{r_p} l_p in experiments/gradsim_true.py -- NOT the intermediate
    # (key) dim. Absent entirely (not even an empty array) when the flag is off, so the npz
    # is byte-identical to the pre-existing schema by default.
    if args.save_vectors:
        arrs["resid_vecs"] = np.stack(resid_vec_list).astype(np.float32)  # [N, hidden]
    np.savez_compressed(npz_path, **arrs)

    summary = {
        "model": args.model,
        "model_tag": tag,
        "dataset": args.dataset,
        "layer": layer,
        "n_edits": N,
        "seed": args.seed,
        "steps": args.steps,
        "lr": args.lr,
        "device": args.device,
        "edit_success_rate": _stat(np.mean, edit_ok),
        "S_mean": _stat(np.mean, S),
        "S_median": _stat(np.median, S),
        "S_std": _stat(np.std, S),
        "resid_norm_mean": _stat(np.mean, resid_norm),   # ||v-Wk|| (S-numerator)
        "resid_norm_median": _stat(np.median, resid_norm),
        "key_norm_mean": _stat(np.mean, key_norm),       # ||k|| (S-denominator)
        "key_norm_median": _stat(np.median, key_norm),
        "norm_growth_mean": _stat(np.mean, norm_growth),
        "n_finite_S": int(np.isfinite(S).sum()),
        "npz": npz_path,
        "runtime_s": round(time.time() - t0, 1),
    }
    json_path = os.path.join(args.out_dir, f"{tag}_L{layer}.json")
    tmp = json_path + ".tmp"
    json.dump(summary, open(tmp, "w"), indent=2)
    os.replace(tmp, json_path)  # atomic

    print(f"[mech] saved arrays -> {npz_path}", flush=True)
    print(f"[mech] saved summary -> {json_path}", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
