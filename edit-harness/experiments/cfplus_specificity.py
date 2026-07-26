"""cfplus_specificity.py — CounterFact+ (CF+) hard-neighborhood specificity eval.

Reimplements, from the METHOD DESCRIPTION only (no network access to the original
apartresearch/specificityplus repo or its exact prompt-construction code -- see
DIVERGENCES FROM THE PUBLISHED METHOD below), the core idea of Hoelscher-Obermaier et al.
2023 ("Detecting Edit Failures in Large Language Models: An Improved Specificity
Benchmark", CounterFact+ / specificity+): standard CounterFact neighborhood prompts are
"easy" because they never mention the edited fact at all, so a model that FAILED to learn
the edit at all can still look locally specific by accident. CF+ instead evaluates
specificity under a DISTRIBUTION SHIFT TOWARD the edit -- prompts that first invoke the
edited fact, then ask about a neighboring (unrelated) subject -- plus a distributional
(KL) locality metric instead of an accuracy-only one.

This module computes, given an EDITED model and a CounterFact-schema record `e` (with
`neighborhood_prompts` already attached, e.g. via ``egl_metrics.attach_egl_fields``):

  (i)   standard NS  -- reproduces ``egl_metrics``'s own Neighborhood Score EXACTLY
        (import, not duplicate): for each neighborhood prompt, the neighbor is
        UNDAMAGED iff mean-logprob(target_true) > mean-logprob(target_new) (flipped
        inequality vs. ES/PS -- see egl_metrics.py's module docstring).
  (ii)  CF+ hard-NS -- the SAME neighborhood prompts, each PREFIXED with a declarative
        sentence stating the edit's OWN (rewritten) fact, then scored with the SAME
        flipped-inequality rule.
  (iii) neighborhood KL(pre||post) over the FULL VOCAB at the answer position (next-token
        distribution right after the prompt), for both the standard and hard-prefixed
        neighbor prompts -- a distributional locality metric, not accuracy-only.

DIVERGENCES FROM THE PUBLISHED METHOD (flagged per instructions -- these are honest
approximations, not verified against the original repo):
  * EDIT-STATEMENT CONSTRUCTION. CF+'s exact prefix wording/templates are not available
    offline. We construct the simplest well-defined declarative sentence available from
    local CounterFact fields: the edit's own cloze `prompt` completed with `target_new`,
    e.g. prompt="The mother tongue of {subj} is" + target_new="English" ->
    "The mother tongue of {subj} is English." This is prepended verbatim to each
    neighborhood prompt (space-joined). The published method may use paraphrased or
    templated statements instead -- ours is the most literal, no-network reconstruction.
  * SCORING PAIR FOR THE HARD VARIANT. Hard-NS reuses the EDIT's own (target_new,
    target_true) pair on the (now prefixed) neighborhood prompt, exactly like standard NS
    does (this is the harness's own existing NS convention, extended to the new prompt
    text) -- CF+ may define its own per-neighbor target pair; not reproducible offline.
  * KL METRIC. "KL-divergence-based neighborhood metrics" is stated only qualitatively in
    the prompt; we implement the most direct reading -- full-vocabulary KL(pre||post) at
    the single next-token position -- rather than, e.g., a multi-token generation KL or a
    restricted-vocabulary variant.
  * MEMORY. Precomputing pre-edit full-vocab distributions for every neighbor prompt (both
    variants) for every edit is O(n_edits * max_neighborhood * 2 * |vocab|) floats; kept
    float32 and capped by `--max_neighborhood` (default 5, deliberately smaller than
    egl_metrics's 10) -- NOT yet exercised at killgate's n_edits=200/n_probes=500 science
    scale. A driver-level decision on how to bound this is left for later (per 2b).

Follows the harness's restore-every-edit split (mirrors egl_metrics.py's
precompute_egl_baselines / measure_egl_one): pre-edit quantities are captured BEFORE any
edit is applied; measure_cfplus_one is called on the EDITED model, strictly before restore.

Intended invocation once wired into a driver (NOT built here -- queue decision deferred,
per instructions 2b):
  # inside killgate_keygeom.py's per-edit loop, analogous to --egl:
  #   base_i = precompute_cfplus_baselines(model, tok, edits, device, args.cfplus_max_neighborhood)[i]
  #   ... apply_edit(...) ...
  #   rec = measure_cfplus_one(model, tok, e, base_i, device, args.cfplus_max_neighborhood)
  #   ... restore ...

Standalone CLI (this file's own smoke/test entry point -- runs a self-contained
restore-every-edit loop; NOT a --save_matrices-integrated harness driver):
  python experiments/cfplus_specificity.py --model data/models/Qwen2.5-0.5B \
      --n_edits 2 --max_neighborhood 4 --steps 2 --device cpu \
      --out results/smoke/cfplus_qwen05b_cpu_smoke.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import torch

HARNESS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HARNESS not in sys.path:
    sys.path.insert(0, HARNESS)
# dual-path import (killgate_keygeom.py's own lesson, 2026-07-03 r3 smoke bug): this module
# may be invoked as `python3 experiments/cfplus_specificity.py` (script dir on sys.path,
# bare import needed) or imported package-qualified by another experiments/ script.
try:
    from experiments.egl_metrics import full_target_scores, attach_egl_fields  # noqa: E402
except ModuleNotFoundError:
    from egl_metrics import full_target_scores, attach_egl_fields  # noqa: E402
from metrics import next_token_logits  # noqa: E402
from editors.rome_native import find_subject_last_token_index, _capture_key  # noqa: E402


# --------------------------------------------------------------------------- #
# 0. edit-statement construction (the one CF+-specific, undocumented-offline choice)
# --------------------------------------------------------------------------- #
def edit_statement(e: dict) -> str:
    """Declarative sentence stating the EDIT's rewritten fact, from local CF fields only.
    See module docstring, "EDIT-STATEMENT CONSTRUCTION" divergence."""
    return f"{e['prompt']} {e['target_new']}."


def hard_prompt(e: dict, neighbor_prompt: str) -> str:
    return f"{edit_statement(e)} {neighbor_prompt}"


# --------------------------------------------------------------------------- #
# 1. full-vocab next-token distribution + KL (pure, reused by both pre/post reads)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _full_vocab_probs(model, tok, prompt: str, device: str) -> np.ndarray:
    logits = next_token_logits(model, tok, prompt, device)   # [V] fp32 cpu (metrics.py contract)
    return torch.softmax(logits, dim=-1).numpy().astype(np.float32)


def _kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """KL(p || q), clipped for numerical safety. Pure numpy, no model access."""
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * np.log(p / q)))


# --------------------------------------------------------------------------- #
# 2. pre-edit baselines (BEFORE any edit is applied)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def precompute_cfplus_baselines(model, tok, edits: List[dict], device: str,
                                max_neighborhood: int) -> List[dict]:
    """One entry per edit, in edit order. Each entry carries the PRE-EDIT full-vocab
    next-token distribution for every (capped) neighborhood prompt, both variants."""
    baselines = []
    for e in edits:
        nps = (e.get("neighborhood_prompts") or [])[:max_neighborhood]
        std_pre = [_full_vocab_probs(model, tok, np_, device) for np_ in nps]
        hard_pre = [_full_vocab_probs(model, tok, hard_prompt(e, np_), device) for np_ in nps]
        baselines.append({"neighborhood_prompts": nps, "std_pre": std_pre, "hard_pre": hard_pre})
    return baselines


# --------------------------------------------------------------------------- #
# 3. post-edit measurement (ON THE EDITED MODEL, strictly before restore)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def measure_cfplus_one(model, tok, e: dict, base_i: dict, device: str,
                       max_neighborhood: int) -> dict:
    nps = (e.get("neighborhood_prompts") or [])[:max_neighborhood]
    tnew, ttrue = e["target_new"], e.get("target_true")

    ns_vals, hard_ns_vals, kl_std, kl_hard = [], [], [], []
    for k, np_ in enumerate(nps):
        # (i) standard NS -- reproduces egl_metrics's own flipped-inequality convention.
        # full_target_scores's return is typed Dict[str, Optional[float]] uniformly, but
        # 'lp_new' is only None-typed for symmetry with 'lp_true' -- it's always a real
        # float (mean_logprob_full_target(target_new) is unconditional). Only 'lp_true'
        # can genuinely be None (when target_true is falsy), so only it needs the guard.
        ft_std = full_target_scores(model, tok, np_, tnew, ttrue, device)
        lp_true_std, lp_new_std = ft_std["lp_true"], ft_std["lp_new"]
        undamaged_std = 1.0 if (lp_true_std is not None and lp_new_std is not None
                                and lp_true_std > lp_new_std) else 0.0
        ns_vals.append(undamaged_std)

        # (ii) CF+ hard-NS -- same rule, prefixed prompt.
        hp = hard_prompt(e, np_)
        ft_hard = full_target_scores(model, tok, hp, tnew, ttrue, device)
        lp_true_hard, lp_new_hard = ft_hard["lp_true"], ft_hard["lp_new"]
        undamaged_hard = 1.0 if (lp_true_hard is not None and lp_new_hard is not None
                                 and lp_true_hard > lp_new_hard) else 0.0
        hard_ns_vals.append(undamaged_hard)

        # (iii) full-vocab KL(pre||post) at the answer position, both variants.
        if k < len(base_i.get("std_pre", [])):
            post_std = _full_vocab_probs(model, tok, np_, device)
            kl_std.append(_kl(base_i["std_pre"][k], post_std))
        if k < len(base_i.get("hard_pre", [])):
            post_hard = _full_vocab_probs(model, tok, hp, device)
            kl_hard.append(_kl(base_i["hard_pre"][k], post_hard))

    return {
        "subject": e.get("subject"),
        "NS": float(np.mean(ns_vals)) if ns_vals else None,
        "NS_hard": float(np.mean(hard_ns_vals)) if hard_ns_vals else None,
        "KL_std_mean": float(np.mean(kl_std)) if kl_std else None,
        "KL_hard_mean": float(np.mean(kl_hard)) if kl_hard else None,
        "n_neighborhood": len(nps),
    }


def summarize_cfplus(records: List[dict]) -> dict:
    def _mean(key):
        vals = [r[key] for r in records if r.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    ns_mean, ns_hard_mean = _mean("NS"), _mean("NS_hard")
    return {
        "n_edits": len(records),
        "NS": ns_mean, "NS_hard": ns_hard_mean,
        "KL_std_mean": _mean("KL_std_mean"), "KL_hard_mean": _mean("KL_hard_mean"),
        "hard_ns_le_std_ns": (
            None if (ns_mean is None or ns_hard_mean is None)
            else bool(ns_hard_mean <= ns_mean)),
        "hard_ns_le_std_ns_note": ("HYPOTHESIS, not asserted: CF+'s harder distribution-"
                                   "shifted prompts are expected to expose MORE collateral "
                                   "damage than standard neighborhood prompts, i.e. "
                                   "NS_hard <= NS. Reported, not gated on."),
    }


def write_cfplus_sidecar(path: str, obj: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# standalone smoke/test CLI: self-contained restore-every-edit loop
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="CF+ hard-neighborhood specificity smoke/test CLI.")
    ap.add_argument("--model", default=os.path.join(HARNESS, "data", "models", "Qwen2.5-0.5B"))
    ap.add_argument("--data", default=os.path.join(HARNESS, "data", "counterfact.json"))
    ap.add_argument("--n_edits", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layer", default="auto")
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--max_neighborhood", type=int, default=5)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    # ---- editor selection (2026-07-08 extension for run_cfplus.sh's {rome,memit,alpha}
    # sweep; killgate_keygeom.py's dispatch, cut down to this driver's needs — no
    # probe/holdout distinction, the fit bank below plays that role for both editors) ----
    ap.add_argument("--editor", choices=["rome", "memit", "alpha"], default="rome")
    ap.add_argument("--n_fit", type=int, default=100,
                    help="memit/alpha only: size of the disjoint fact bank used to fit the "
                         "MEMIT layer covariance / AlphaEdit null-space projector. Loaded via "
                         "load_counterfact's n_probes slot, right after the n_edits edits (so "
                         "disjoint from them, same shuffle/seed).")
    ap.add_argument("--keep_ratio", type=float, default=0.99,
                    help="alpha: fraction of preserved-key energy to project out")
    ap.add_argument("--memit_layers", default="auto",
                    help="comma ints or 'auto' = span of 4 ending at --layer (memit's own "
                         "hard requirement: max(memit_layers) == --layer)")
    ap.add_argument("--memit_cov_tokens", type=int, default=20000)
    ap.add_argument("--memit_cov_reg", type=float, default=1e-2)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    HARNESS_ = HARNESS
    if HARNESS_ not in sys.path:
        sys.path.insert(0, HARNESS_)
    from experiments.killgate_keygeom import load_counterfact  # noqa: E402

    if args.editor == "rome":
        from editors.rome_native import apply_edit  # noqa: E402
    elif args.editor == "memit":
        from editors.memit import apply_edit, estimate_layer_covariances, parse_memit_layers  # noqa: E402
    else:  # alpha
        from editors.alphaedit import apply_edit, build_null_projector  # noqa: E402

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(args.device).eval()
    nL = model.config.num_hidden_layers
    layer = nL // 2 if args.layer == "auto" else int(args.layer)
    if args.editor == "memit":
        arch_name = model.config.model_type
        if arch_name in ("gpt2", "gptj"):
            raise SystemExit(f"[cfplus] editor=memit is not supported on {arch_name}-family "
                             "models (residual-stream hook needs the real decoder-layer Module)")
        memit_layers = parse_memit_layers(args.memit_layers, layer, nL)
    else:
        memit_layers = None
    print(f"[cfplus] loaded {args.model} ({nL} layers, layer={layer}, editor={args.editor}, "
          f"device={args.device}) {time.time()-t0:.1f}s", flush=True)

    # n_fit is 0 for rome (no fit bank needed); for memit/alpha it's a disjoint bank of
    # facts (SAME shuffle/seed as the edits, drawn right after them) used ONLY to fit the
    # covariance/projector — never scored as neighborhood prompts.
    n_fit = args.n_fit if args.editor != "rome" else 0
    edits, fit_records, *_ = load_counterfact(args.data, args.n_edits, n_fit, args.seed)
    n_match = attach_egl_fields(edits, args.data, "counterfact")
    print(f"[cfplus] {len(edits)} edits, {n_match} carry neighborhood_prompts, "
          f"{len(fit_records)} fit-bank records {time.time()-t0:.1f}s", flush=True)

    # ---- memit covariance / alpha projector: fit ONCE from fit_records, before any edit ----
    memit_cov = None
    alpha_proj = None
    if args.editor == "memit":
        cov_prompts = [r["prompt"] for r in fit_records]
        memit_cov = estimate_layer_covariances(model, tok, cov_prompts, memit_layers, args.device,
                                               max_tokens=args.memit_cov_tokens, reg=args.memit_cov_reg)
        print(f"[cfplus] memit cov fit on {len(cov_prompts)} prompts, layers={memit_layers} "
              f"{time.time()-t0:.1f}s", flush=True)
    elif args.editor == "alpha":
        def key_for(prompt, subject):
            idx = find_subject_last_token_index(tok, prompt, subject)
            return _capture_key(model, tok, layer, prompt, idx, args.device).float().cpu().numpy()
        K_fit = np.stack([key_for(r["prompt"], r["subject"]) for r in fit_records])
        alpha_proj = build_null_projector(torch.tensor(K_fit, device=args.device), args.keep_ratio)
        print(f"[cfplus] alpha projector fit on {K_fit.shape[0]} keys "
              f"{time.time()-t0:.1f}s", flush=True)

    base = precompute_cfplus_baselines(model, tok, edits, args.device, args.max_neighborhood)
    print(f"[cfplus] baselines precomputed {time.time()-t0:.1f}s", flush=True)

    restore_layers = memit_layers if args.editor == "memit" else [layer]
    W_refs = {li: model.model.layers[li].mlp.down_proj.weight for li in restore_layers}
    W_bases = {li: w.detach().clone() for li, w in W_refs.items()}
    records = []
    for i, e in enumerate(edits):
        if args.editor == "rome":
            cfg = {"layer": layer, "steps": args.steps, "lr": args.lr}
        elif args.editor == "memit":
            cfg = {"layers": memit_layers, "z_layer": layer, "steps": args.steps,
                   "lr": args.lr, "cov": memit_cov, "cov_source": "generic"}
        else:  # alpha
            cfg = {"layer": layer, "steps": args.steps, "lr": args.lr, "projector": alpha_proj}
        apply_edit(model, tok, e, cfg, args.device)
        records.append(measure_cfplus_one(model, tok, e, base[i], args.device, args.max_neighborhood))
        with torch.no_grad():
            for li in restore_layers:
                W_refs[li].copy_(W_bases[li])
        print(f"[cfplus] edit {i+1}/{len(edits)} {time.time()-t0:.1f}s", flush=True)

    res = {
        "model": args.model, "layer": layer, "editor": args.editor, "seed": args.seed,
        "steps": args.steps, "max_neighborhood": args.max_neighborhood,
        "n_fit": len(fit_records),
        "memit_layers": memit_layers,
        "records": records,
        "summary": summarize_cfplus(records),
        "runtime_s": round(time.time() - t0, 1),
    }
    print(json.dumps(res, indent=2), flush=True)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        write_cfplus_sidecar(args.out, res)
        print(f"[cfplus] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
