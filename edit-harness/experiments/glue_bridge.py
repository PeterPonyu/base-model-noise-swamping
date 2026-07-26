#!/usr/bin/env python3
"""glue_bridge.py — P1 of the 2026-07-09 enhancement round: the geometry -> CAPABILITY
bridge. Runs the exact killgate_keygeom.py restore-every-edit protocol, but with GLUE
task examples as the probe set: does the edit-key <-> task-prompt-key cosine predict
per-example TASK damage (label-margin drop / prediction flips), the way it predicts
factual-probe logit damage?

WHY A NEW FILE (memory/live-file-edit-hazard-under-running-queue.md): glue_downstream.py
is the sequential-trajectory tool (cumulative edits, checkpointed task accuracy, coarse);
this module is the per-edit-pair-level law test (one edit at a time, restore between,
COS x damage matrices in killgate npz format so analyze_matrices.within_probe_rhos works
UNCHANGED). Neither replaces the other; nothing shared is mutated.

REUSED, NOT REIMPLEMENTED:
  * edits + holdout bank: killgate_keygeom.load_counterfact (same sampling semantics,
    same seed -> same edit set as the gate_* rows, enabling cross-experiment joins)
  * GLUE loading/templates/label words: glue_downstream.{load_glue_task, build_prompt,
    LABEL_WORDS} (same zero-shot two-way forced-choice construction)
  * editing: editors.rome_native.apply_edit / editors.alphaedit.{apply_edit,
    build_null_projector}; arch via editors.arch_compat.normalize_arch
  * scoring: metrics.{next_token_logits, first_target_token_id, _capture_key,
    find_subject_last_token_index, efficacy}

PROBE-KEY CONVENTION: GLUE prompts have no edit subject — the key is captured at the
LAST prompt token (the position whose layer-L MLP output feeds the label logit), the
same fallback rippleedits_loader.py documents for non-cloze prompts.

DAMAGE SEMANTICS (per edit i, per GLUE example j, matching killgate sign convention
"positive = damaged"):
  margin m_j  = logit(correct label word) - logit(other label word)  [pre and post]
  damage_margin[i,j] = m_pre[j] - m_post[i,j]
  damage_flip[i,j]   = 1.0 if (pre-correct AND post-incorrect) else 0.0

Output: --out json (summary + within-example rho) and, under --save_matrices,
<matrix_dir>/<basename(out)>.npz with COS/damage_margin/damage_flip/edit_ok/pre_margin/
task_id — killgate-shaped [N_e x M_glue] matrices.

CPU smoke (no GPU, tiny):
  python experiments/glue_bridge.py --model data/models/Qwen2.5-0.5B --n_edits 3 \
      --n_glue_samples 4 --steps 2 --device cpu --layer 12 --out /tmp/gb_smoke.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HARNESS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HARNESS not in sys.path:
    sys.path.insert(0, HARNESS)
_EXP = os.path.dirname(os.path.abspath(__file__))
if _EXP not in sys.path:
    sys.path.insert(0, _EXP)

from metrics import (  # noqa: E402
    next_token_logits, first_target_token_id, efficacy,
)
from editors.rome_native import (  # noqa: E402
    _capture_key, find_subject_last_token_index,
)
from editors.arch_compat import normalize_arch  # noqa: E402

try:  # dual-path import (same pattern as killgate's egl_metrics import)
    from experiments.killgate_keygeom import load_counterfact  # noqa: E402
    from experiments.glue_downstream import (  # noqa: E402
        LABEL_WORDS, build_prompt, load_glue_task)
except ModuleNotFoundError:
    from killgate_keygeom import load_counterfact  # noqa: E402
    from glue_downstream import LABEL_WORDS, build_prompt, load_glue_task  # noqa: E402


def within_example_rhos(C: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Per-COLUMN Spearman across edits — the same estimand as analyze_matrices.
    within_probe_rhos (imported when available; local fallback keeps the CPU smoke
    runnable from a bare checkout)."""
    try:
        from analyze_matrices import within_probe_rhos  # noqa: E402
        return within_probe_rhos(C, D)
    except Exception:
        from scipy.stats import spearmanr
        out = np.full(C.shape[1], np.nan)
        for j in range(C.shape[1]):
            if np.nanstd(C[:, j]) > 0 and np.nanstd(D[:, j]) > 0:
                out[j] = spearmanr(C[:, j], D[:, j])[0]
        return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(HARNESS, "data", "models", "Llama-3.2-1B"))
    ap.add_argument("--data", default=os.path.join(HARNESS, "data", "counterfact.json"))
    ap.add_argument("--glue_dir", default=os.path.join(HARNESS, "data", "glue"))
    ap.add_argument("--tasks", default="sst2,mrpc,rte")
    ap.add_argument("--n_glue_samples", type=int, default=100,
                    help="validation examples PER task (probe columns = tasks x this)")
    ap.add_argument("--n_edits", type=int, default=200)
    ap.add_argument("--editor", choices=["rome", "alpha"], default="rome")
    ap.add_argument("--alpha_proj_source", choices=["holdout"], default="holdout",
                    help="alpha projector fit bank: HOLDOUT CounterFact keys only (the "
                         "honest protocol; probes here are GLUE examples, so a probes-"
                         "sourced projector would be a different experiment entirely)")
    ap.add_argument("--holdout_frac", type=float, default=1.0,
                    help="holdout bank size as a fraction of n_glue probes-equivalent "
                         "(passed to load_counterfact as n_holdout=round(frac*100))")
    ap.add_argument("--keep_ratio", type=float, default=0.98)
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--model_dtype", choices=["fp32", "bf16"], default="fp32")
    ap.add_argument("--save_matrices", action="store_true")
    ap.add_argument("--matrix_dir", default=os.path.join(HARNESS, "results", "matrices"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t0 = time.time()
    device = args.device
    tok = AutoTokenizer.from_pretrained(args.model)
    load_dtype = torch.float32 if args.model_dtype == "fp32" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=load_dtype).to(device).eval()
    arch = normalize_arch(model, tok, device)
    nL = model.config.num_hidden_layers
    layer = int(args.layer)
    if not (0 <= layer < nL):
        raise SystemExit(f"[gb] --layer {layer} out of range for {nL}-layer model")
    print(f"[gb] loaded {args.model} ({nL} layers, edit layer={layer}, arch={arch}, "
          f"dtype={args.model_dtype}) {time.time()-t0:.1f}s", flush=True)

    # ---- edits (+ holdout bank for the alpha projector) — killgate loader, same seed
    # semantics as the gate_* rows so edit sets are joinable across experiments ----
    n_holdout = int(round(args.holdout_frac * 100)) if args.editor == "alpha" else 0
    edits, _probes_unused, holdout = load_counterfact(
        args.data, args.n_edits, 0, args.seed, n_holdout)
    if args.editor == "alpha" and len(holdout) < 5:
        raise SystemExit(f"[gb] alpha holdout bank too small ({len(holdout)})")
    print(f"[gb] {args.editor}: {len(edits)} edits, {len(holdout)} holdout", flush=True)

    # ---- GLUE probe set ----
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    glue: list = []          # flat probe list: {"task","prompt","tid_correct","tid_other"}
    for t in tasks:
        for ex in load_glue_task(args.glue_dir, t, args.n_glue_samples, args.seed):
            words = LABEL_WORDS[t]
            lab = int(ex["label"])
            glue.append({
                "task": t,
                "prompt": build_prompt(t, ex),
                "tid_correct": first_target_token_id(tok, words[lab]),
                "tid_other": first_target_token_id(tok, words[1 - lab]),
            })
    M = len(glue)
    if M == 0:
        raise SystemExit("[gb] empty GLUE probe set")
    print(f"[gb] GLUE probes: {M} examples across {tasks}", flush=True)

    # ---- keys ----
    def edit_key(e):
        idx = find_subject_last_token_index(tok, e["prompt"], e["subject"])
        return _capture_key(model, tok, layer, e["prompt"], idx, device).float().cpu().numpy()

    def last_token_key(prompt):
        idx = len(tok.encode(prompt, add_special_tokens=True)) - 1
        return _capture_key(model, tok, layer, prompt, idx, device).float().cpu().numpy()

    K_edit = np.stack([edit_key(e) for e in edits])                    # [N, d]
    K_glue = np.stack([last_token_key(g["prompt"]) for g in glue])     # [M, d]
    Ke = K_edit / (np.linalg.norm(K_edit, axis=1, keepdims=True) + 1e-8)
    Kg = K_glue / (np.linalg.norm(K_glue, axis=1, keepdims=True) + 1e-8)
    COS = (Ke @ Kg.T).astype(np.float32)                               # [N, M]
    print(f"[gb] keys+cosine done {time.time()-t0:.1f}s", flush=True)

    # ---- editor setup ----
    if args.editor == "rome":
        from editors.rome_native import apply_edit
        alpha_proj = None
    else:
        from editors.alphaedit import apply_edit, build_null_projector
        hk = np.stack([edit_key(h) for h in holdout])  # subject-last-token holdout keys
        alpha_proj = build_null_projector(torch.tensor(hk, device=device), args.keep_ratio)
        print(f"[gb] alpha projector [holdout]: fit on {hk.shape[0]} keys", flush=True)

    # ---- pre-edit margins ----
    @torch.no_grad()
    def margin(g) -> float:
        logits = next_token_logits(model, tok, g["prompt"], device)
        return float(logits[g["tid_correct"]]) - float(logits[g["tid_other"]])

    pre_margin = np.array([margin(g) for g in glue], dtype=np.float32)
    pre_correct = (pre_margin > 0).astype(np.float32)
    task_names = sorted(set(g["task"] for g in glue))
    task_id = np.array([task_names.index(g["task"]) for g in glue], dtype=np.int32)
    print(f"[gb] pre accuracy: " + "  ".join(
        f"{t}:{pre_correct[task_id == k].mean():.3f}"
        for k, t in enumerate(task_names)), flush=True)

    # ---- snapshot the edited weight for exact restore (single-layer editors only) ----
    W_ref = model.model.layers[layer].mlp.down_proj.weight
    W_base = W_ref.detach().clone()

    N = len(edits)
    damage_margin = np.zeros((N, M), dtype=np.float32)
    damage_flip = np.zeros((N, M), dtype=np.float32)
    edit_ok = np.zeros(N, dtype=np.float32)
    norm_growth = np.zeros(N, dtype=np.float32)

    for i, e in enumerate(edits):
        if args.editor == "rome":
            cfg = {"layer": layer, "steps": args.steps, "lr": args.lr}
        else:
            cfg = {"layer": layer, "steps": args.steps, "lr": args.lr, "projector": alpha_proj}
        info = apply_edit(model, tok, e, cfg, device)
        ng = info["delta_weight_norm"]
        norm_growth[i] = float(ng[layer]) if isinstance(ng, dict) else float(ng)
        eff = efficacy(model, tok, e["prompt"], e["target_new"], e.get("target_true"), device)
        edit_ok[i] = eff["success"]
        for j, g in enumerate(glue):
            m_post = margin(g)
            damage_margin[i, j] = pre_margin[j] - m_post
            damage_flip[i, j] = 1.0 if (pre_correct[j] > 0 and m_post <= 0) else 0.0
        with torch.no_grad():
            W_ref.data.copy_(W_base)
        if (i + 1) % 10 == 0:
            print(f"[gb] edit {i+1}/{N}  {time.time()-t0:.1f}s", flush=True)

    # restore sanity: weights must be bitwise-identical to base after the loop
    if not torch.equal(W_ref.data, W_base):
        raise SystemExit("[gb] RESTORE TRIPWIRE: post-loop weight != base snapshot")

    # ---- analysis: within-example rho on pre-correct columns (margin damage is only
    # meaningful where the model had the example right to begin with) ----
    mask_e = edit_ok > 0
    mask_c = pre_correct > 0

    def nanmean_safe(a: np.ndarray) -> float:
        return float(np.nanmean(a)) if np.isfinite(a).any() else float("nan")

    res_rho = {}
    for name, Dm in (("margin", damage_margin), ("flip", damage_flip)):
        rhos_all = within_example_rhos(COS, Dm)
        rhos_ok = (within_example_rhos(COS[mask_e][:, mask_c], Dm[mask_e][:, mask_c])
                   if mask_e.sum() >= 10 and mask_c.sum() >= 10 else np.array([np.nan]))
        res_rho[name] = {
            "within_example_mean_all": round(nanmean_safe(rhos_all), 4),
            "within_example_mean_editok_precorrect": round(nanmean_safe(rhos_ok), 4),
            "n_cols_all": int(np.isfinite(rhos_all).sum()),
            "n_cols_filtered": int(np.isfinite(rhos_ok).sum()),
        }

    res = {
        "model": args.model, "editor": args.editor, "layer": layer, "seed": args.seed,
        "steps": args.steps, "lr": args.lr, "n_edits": N, "n_glue": M,
        "tasks": task_names,
        "edit_success_rate": round(float(edit_ok.mean()), 3),
        "pre_accuracy": {t: round(float(pre_correct[task_id == k].mean()), 4)
                         for k, t in enumerate(task_names)},
        "mean_norm_growth": round(float(norm_growth.mean()), 4),
        "rho": res_rho,
        "provenance": {
            "protocol": "restore-every-edit, GLUE examples as probes, last-token keys",
            "alpha_proj_source": (args.alpha_proj_source if args.editor == "alpha" else None),
            "model_dtype": args.model_dtype,
        },
        "runtime_s": round(time.time() - t0, 1),
    }
    if args.save_matrices:
        os.makedirs(args.matrix_dir, exist_ok=True)
        npz_path = os.path.join(
            args.matrix_dir, os.path.splitext(os.path.basename(args.out))[0] + ".npz")
        np.savez_compressed(
            npz_path, COS=COS, damage_margin=damage_margin, damage_flip=damage_flip,
            edit_ok=edit_ok, pre_margin=pre_margin, pre_correct=pre_correct,
            task_id=task_id, norm_growth=norm_growth,
            tasks=np.array(task_names))
        res["matrices_npz"] = os.path.relpath(npz_path, HARNESS)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"[gb] wrote {args.out} ({res['runtime_s']}s)", flush=True)


if __name__ == "__main__":
    main()
