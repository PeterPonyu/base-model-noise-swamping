"""ft_editor.py — constrained fine-tuning editor (FT-L style).

Legitimate, fully-testable editing baseline. Idea:

  * Freeze the whole model.
  * Unfreeze ONLY the ``down_proj.weight`` of a chosen MLP layer (or a small
    set of layers) — this is the "FT-L" (fine-tune, locality-constrained)
    restriction used as a baseline in the ROME / MEMIT papers.
  * Take a few Adam steps minimising cross-entropy of the *new target* tokens
    appended to the edit prompt (loss is masked to the target span only).
  * Constrain the update to stay near the original weights so unrelated facts
    are preserved:
        - L2 anchor:  lambda_l2 * ||W - W0||^2   (always on)
        - optional KL: lambda_kl * KL(p_orig || p_now) on neighbourhood prompts
          (only if ``neighborhood_prompts`` are provided in the edit request).

The public entry point is ``apply_edit(model, tokenizer, edit_request, config,
device)`` which mutates ``model`` in place and returns an info dict. The runner
is responsible for snapshotting / restoring weights between edits.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List

import torch
import torch.nn.functional as F

# Allow `import metrics` whether run as module or script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metrics import target_token_ids  # noqa: E402


def _select_down_proj(model, layers: List[int]):
    """Return list of (layer_idx, weight Parameter) for the chosen down_proj layers."""
    out = []
    for li in layers:
        w = model.model.layers[li].mlp.down_proj.weight
        out.append((li, w))
    return out


def apply_edit(
    model,
    tokenizer,
    edit_request: Dict,
    config: Dict,
    device: str = "cpu",
) -> Dict:
    """Apply an FT-L edit in place.

    edit_request keys used:
      * ``prompt``      (str)  — the edit prompt (contains the subject).
      * ``target_new``  (str)  — the new object the model should produce.
      * ``neighborhood_prompts`` (List[str], optional) — for the KL anchor.

    config keys (all optional, with defaults):
      * ``layers``     (List[int])  default [n_layers-1]  — which MLP layers.
      * ``steps``      (int)        default 25
      * ``lr``         (float)      default 5e-3
      * ``lambda_l2``  (float)      default 1e-3
      * ``lambda_kl``  (float)      default 1.0  (only used if neighbours given)
    """
    prompt: str = edit_request["prompt"]
    target_new: str = edit_request["target_new"]
    neighborhood: List[str] = edit_request.get("neighborhood_prompts", []) or []

    n_layers = model.config.num_hidden_layers
    layers: List[int] = config.get("layers", [n_layers - 1])
    steps: int = int(config.get("steps", 25))
    lr: float = float(config.get("lr", 5e-3))
    lambda_l2: float = float(config.get("lambda_l2", 1e-3))
    lambda_kl: float = float(config.get("lambda_kl", 1.0))

    model.to(device)
    model.eval()  # disable dropout; we still backprop into the selected weights

    # --- freeze everything, then unfreeze the selected down_proj weights ---
    for p in model.parameters():
        p.requires_grad_(False)
    selected = _select_down_proj(model, layers)
    originals = {li: w.detach().clone() for li, w in selected}
    for _, w in selected:
        w.requires_grad_(True)

    # --- build the training batch: prompt + target, loss masked to target ---
    tgt_ids = target_token_ids(tokenizer, target_new)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    input_ids = torch.tensor([prompt_ids + tgt_ids], device=device)
    labels = torch.full_like(input_ids, -100)
    labels[0, len(prompt_ids):] = torch.tensor(tgt_ids, device=device)

    # --- optional KL anchor: snapshot original neighbourhood distributions ---
    kl_inputs = None
    orig_logp = None
    if neighborhood and lambda_kl > 0:
        kl_inputs = [tokenizer(p, return_tensors="pt").to(device) for p in neighborhood]
        with torch.no_grad():
            orig_logp = [F.log_softmax(model(**ki).logits[0, -1], dim=-1) for ki in kl_inputs]

    opt = torch.optim.Adam([w for _, w in selected], lr=lr)

    history: List[float] = []
    for _ in range(steps):
        opt.zero_grad()
        out = model(input_ids=input_ids, labels=labels)
        loss = out.loss
        # L2 anchor toward original weights
        l2 = sum(((w - originals[li]) ** 2).sum() for li, w in selected)
        loss = loss + lambda_l2 * l2
        # optional KL anchor on neighbourhood prompts
        if kl_inputs is not None:
            kl = 0.0
            for ki, olp in zip(kl_inputs, orig_logp):
                cur_logp = F.log_softmax(model(**ki).logits[0, -1], dim=-1)
                kl = kl + F.kl_div(cur_logp, olp, log_target=True, reduction="sum")
            loss = loss + lambda_kl * (kl / len(kl_inputs))
        loss.backward()
        opt.step()
        history.append(float(loss.detach().item()))

    # detach edited weights from the autograd graph; freeze again for inference
    for _, w in selected:
        w.requires_grad_(False)
    model.eval()

    delta_norms = {li: float((w.detach() - originals[li]).norm().item()) for li, w in selected}
    return {
        "editor": "ft_editor",
        "layers": layers,
        "steps": steps,
        "lr": lr,
        "lambda_l2": lambda_l2,
        "lambda_kl": lambda_kl if neighborhood else 0.0,
        "used_kl": kl_inputs is not None,
        "loss_history": history,
        "final_loss": history[-1] if history else None,
        "delta_weight_norm": delta_norms,
        "target_token_ids": tgt_ids,
    }
