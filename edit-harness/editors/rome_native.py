"""rome_native.py — native ROME-style rank-one editor.

Implements the core of ROME (Rank-One Model Editing) directly on top of a
HuggingFace ``LlamaForCausalLM`` (or any model exposing
``model.model.layers[i].mlp.down_proj``), with NO EasyEdit dependency.

Pipeline
--------
1. **Locate** the target MLP layer ``L`` and its ``down_proj`` linear
   ``W : R^{intermediate} -> R^{hidden}``.
2. **Key** ``k``: run the edit prompt once and capture, via a forward hook, the
   *input* to ``down_proj`` at the subject's last token. This vector (dim =
   intermediate_size) is the "key" the MLP associates with the fact.
3. **Value** ``v``: optimise a target output vector ``v`` (dim = hidden_size)
   such that, when ``down_proj``'s output at the subject token is *replaced* by
   ``v``, the model predicts ``target_new``. We start ``v = W k`` and take a few
   gradient steps (a small anchor keeps ``v`` near the original output).
4. **Rank-one update**: solve for ``ΔW`` so that ``(W + ΔW) k = v`` while moving
   every other key as little as possible:

       ΔW = (v - W k) k^T / (k^T k)            # implemented here

   and apply it in place: ``W <- W + ΔW``.

TODO (MEMIT covariance refinement) — see the marked block below: ROME/MEMIT
replace ``k^T k`` with a whitening by the pre-computed second-moment matrix
``C = E[k k^T]`` of keys over a large text sample:

       ΔW = (v - W k) (C^{-1} k)^T / (k^T C^{-1} k)

which spreads the edit along the natural key covariance and improves locality.
Estimating ``C`` requires a forward pass over ~100k tokens; left as a TODO so
the implemented simplified update stays correct and self-contained.

Public entry point: ``apply_edit(model, tokenizer, edit_request, config, device)``.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metrics import target_token_ids  # noqa: E402
# tp_edit_util is imported LAZILY (inside _optimise_value/apply_edit below), not here at
# module top level: killgate_keygeom.py imports find_subject_last_token_index/_capture_key
# from this module at ITS OWN top level for EVERY --editor (ft/memit/grace too, not just
# rome/alpha) — a top-level import failure here would break every editor's killgate
# invocation, not just the TP-relevant ones. Deferring the import confines any tp_edit_util
# breakage to the call sites that actually need it.


def find_subject_last_token_index(tokenizer, prompt: str, subject: Optional[str]) -> int:
    """Index (into the tokenised prompt) of the subject's LAST token.

    If ``subject`` is found in ``prompt`` we tokenise the prefix up to and
    including the subject and take its final position. Otherwise we fall back to
    the last token of the whole prompt (a safe, if less canonical, edit site).
    """
    full_ids = tokenizer.encode(prompt, add_special_tokens=True)
    if subject and subject in prompt:
        end = prompt.index(subject) + len(subject)
        prefix_ids = tokenizer.encode(prompt[:end], add_special_tokens=True)
        idx = len(prefix_ids) - 1
        # clamp into range (tokeniser merges across the boundary occasionally)
        return max(0, min(idx, len(full_ids) - 1))
    return len(full_ids) - 1


def _capture_key(model, tokenizer, layer_idx: int, prompt: str, tok_index: int, device: str):
    """Forward-hook the down_proj of ``layer_idx`` and return its input at ``tok_index``."""
    down = model.model.layers[layer_idx].mlp.down_proj
    captured = {}

    def hook(_module, inputs, _output):
        # inputs[0]: [batch, seq, intermediate] — the activation entering down_proj
        captured["k"] = inputs[0][0, tok_index, :].detach().clone()

    h = down.register_forward_hook(hook)
    try:
        with torch.no_grad():
            enc = tokenizer(prompt, return_tensors="pt").to(device)
            model(**enc)
    finally:
        h.remove()
    return captured["k"]


def _optimise_value(
    model,
    tokenizer,
    layer_idx: int,
    prompt: str,
    tok_index: int,
    target_new: str,
    device: str,
    steps: int,
    lr: float,
    v_weight_decay: float,
):
    """Optimise the replacement value ``v`` so the model predicts ``target_new``.

    Returns ``v`` (dim = hidden_size). A forward hook overwrites down_proj's
    output at ``tok_index`` with the learnable ``v`` so gradients flow only into
    ``v`` (the model weights stay frozen).
    """
    down = model.model.layers[layer_idx].mlp.down_proj
    for p in model.parameters():
        p.requires_grad_(False)

    # TP-safety: `device` above is the INPUT (embedding) device threaded through the
    # whole killgate pipeline for tokenizer(...).to(device) below — under --device_map
    # the EDITED layer can sit on a DIFFERENT card (accelerate shards at layer
    # boundaries). `v` must live on THIS layer's own device: the `inject` hook further
    # below writes it into down_proj's *output* tensor, which accelerate places on
    # layer_device, never on `device` if the two differ. Single-device path (no
    # --device_map): layer_device == device always, so every line below is unchanged
    # (see tp_edit_util.py).
    from tp_edit_util import resolve_layer_device  # noqa: E402 (lazy — see module header)
    layer_device = resolve_layer_device(model, layer_idx)

    # initialise v at the current output W k (so the edit starts as a no-op)
    init_holder = {}

    def capture_out(_m, _i, output):
        init_holder["v0"] = output[0, tok_index, :].detach().clone()

    h0 = down.register_forward_hook(capture_out)
    with torch.no_grad():
        enc = tokenizer(prompt, return_tensors="pt").to(device)
        model(**enc)
    h0.remove()

    # bf16 boundary: the OPTIMIZATION math (the v parameter, Adam state, the
    # anchor) stays fp32 regardless of model dtype — bf16/fp16 value-opt is the
    # known silent-NaN failure mode. Under an fp32 model .float() is an exact
    # no-op (returns self), so the default path is byte-identical.
    v = init_holder["v0"].clone().detach().float().to(layer_device).requires_grad_(True)
    v0 = init_holder["v0"].clone().detach().float()  # already on layer_device (captured
    # straight off down_proj's own output above); .float() is device-preserving.

    def inject(_m, _i, output):
        output = output.clone()
        # bf16 boundary: cast the injected fp32 v to the hidden-state dtype so a
        # bf16 model never sees an fp32 tensor mid-forward (downstream bf16
        # matmuls would RuntimeError). Grad still flows to the fp32 v through
        # the cast. Under fp32, .to(fp32) returns v itself — byte-identical.
        output[0, tok_index, :] = v.to(output.dtype)
        return output

    # target = first token of target_new appended right after the prompt
    tgt_id = target_token_ids(tokenizer, target_new)[0]
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    opt = torch.optim.Adam([v], lr=lr)
    history: List[float] = []

    for _ in range(steps):
        opt.zero_grad()
        h = down.register_forward_hook(inject)
        try:
            logits = model(**enc).logits[0, -1, :]
        finally:
            h.remove()
        # bf16 boundary: log_softmax + NLL in fp32 (bf16 logp is too coarse for
        # a 2-step Adam signal). .float() is a differentiable cast and an exact
        # no-op under an fp32 model.
        logp = torch.log_softmax(logits.float(), dim=-1)
        nll = -logp[tgt_id]
        reg = v_weight_decay * ((v - v0) ** 2).sum()
        loss = nll + reg
        loss.backward()
        opt.step()
        history.append(float(loss.detach().item()))

    return v.detach(), v0, history


def apply_edit(
    model,
    tokenizer,
    edit_request: Dict,
    config: Dict,
    device: str = "cpu",
) -> Dict:
    """Apply a native ROME rank-one edit in place.

    edit_request keys: ``prompt``, ``target_new``, and optional ``subject``.
    config keys (optional):
      * ``layer``  (int)   default n_layers//2  — target MLP layer.
      * ``steps``  (int)   default 25           — value-optimisation steps.
      * ``lr``     (float) default 5e-1         — value-optimisation lr.
      * ``v_weight_decay`` (float) default 1e-3 — anchor v near W k.
    """
    prompt: str = edit_request["prompt"]
    target_new: str = edit_request["target_new"]
    subject: Optional[str] = edit_request.get("subject")

    n_layers = model.config.num_hidden_layers
    layer_idx: int = int(config.get("layer", n_layers // 2))
    steps: int = int(config.get("steps", 25))
    lr: float = float(config.get("lr", 5e-1))
    v_weight_decay: float = float(config.get("v_weight_decay", 1e-3))

    from tp_edit_util import safe_model_to  # noqa: E402 (lazy — see module header)
    safe_model_to(model, device)  # no-op under --device_map TP (see tp_edit_util.py) —
    # a plain model.to(device) would collapse the accelerate shard placement here, on
    # EVERY edit, since apply_edit() runs once per edit in the killgate loop.
    model.eval()

    tok_index = find_subject_last_token_index(tokenizer, prompt, subject)

    # 1) key k (input to down_proj at the subject's last token)
    k = _capture_key(model, tokenizer, layer_idx, prompt, tok_index, device).float()

    # 2) value v (optimise so the model predicts the new object)
    v, v0, history = _optimise_value(
        model, tokenizer, layer_idx, prompt, tok_index,
        target_new, device, steps, lr, v_weight_decay,
    )
    v = v.float()

    # 3) rank-one update:  ΔW = (v - W k) k^T / (k^T k)
    W = model.model.layers[layer_idx].mlp.down_proj.weight
    W_dtype = W.dtype
    Wk = (W.detach().float() @ k)                      # current output for key k
    residual = (v - Wk)                                # what v must add (dim hidden)
    denom = float((k @ k).item()) + 1e-8
    # ---------------------------------------------------------------- #
    # TODO(MEMIT): replace `k / denom` with whitened `C^{-1} k / (k^T C^{-1} k)`
    # where C = E[k k^T] is the pre-computed second-moment matrix of keys over a
    # large corpus sample. That covariance refines locality; the line below is
    # the correct *simplified* (identity-covariance) ROME update.
    delta = torch.outer(residual, k) / denom          # [hidden, intermediate]
    # ---------------------------------------------------------------- #

    with torch.no_grad():
        before = W.detach().clone()
        W.add_(delta.to(W_dtype))
        applied_norm = float((W.detach() - before).norm().item())

    # sanity: did the edited output now match v at this key?
    new_Wk = (W.detach().float() @ k)
    residual_after = float((new_Wk - v).norm().item())

    return {
        "editor": "rome_native",
        "layer": layer_idx,
        "subject_last_token_index": tok_index,
        "steps": steps,
        "lr": lr,
        "v_weight_decay": v_weight_decay,
        "key_norm": float(k.norm().item()),
        "value_norm": float(v.norm().item()),
        "value_init_norm": float(v0.float().norm().item()),
        "residual_norm": float(residual.norm().item()),
        "delta_weight_norm": applied_norm,
        "rank_one_solve_residual": residual_after,  # ~0 confirms (W+ΔW)k == v
        "value_loss_history": history,
        "final_value_loss": history[-1] if history else None,
        "covariance_used": False,
        "todo": "MEMIT second-moment covariance C (see TODO block) not yet estimated.",
        # ADDITIVE (2026-07-06, true-backprop GradSim cell): the residual VECTOR itself,
        # r = v - Wk, dim = hidden_size (down_proj's OUTPUT dim) -- NOT the intermediate
        # (down_proj INPUT / key) dim that key_norm lives in. Existing callers only read
        # scalar fields from this dict and never enumerate its keys, so this is safe to
        # add unconditionally; no consumer's behavior changes.
        "residual_vec": residual.detach().cpu().numpy().astype(np.float32),
    }
