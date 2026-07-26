"""grace_editor.py — GRACE-style codebook key-value memory editor.

Instead of modifying model weights (ΔW), GRACE (Hartvigsen et al., NeurIPS 2023)
attaches an external codebook of (key, value) pairs to a chosen MLP layer's
down_proj. At inference, for any token whose down_proj INPUT (the same "key"
rome_native captures) is cosine-close to a stored key (>= grace_eps_cos), a forward
hook REPLACES that token's down_proj OUTPUT with the stored value; unmatched
tokens pass through W unchanged. ΔW is exactly zero by construction — the edit
lives entirely in the codebook, never in the weight matrix. This is the "hard
replacement" variant (a single shared cosine threshold, not a learned per-entry
epsilon-ball radius).

Reuses rome_native's key-capture + value-optimisation (same fp32 discipline: the
value-opt math — Adam, the v parameter, log_softmax — stays fp32 regardless of
model dtype; only the value actually written into the forward pass is cast to the
hidden-state dtype, mirroring rome_native's own `inject` hook).

Codebook + hook lifecycle
--------------------------
The codebook is a Python list of (key: Tensor[d_in] fp32, value: Tensor[d_out]
fp32) pairs stashed as attributes on the down_proj module itself, so it survives
across sequential ``apply_edit`` calls on the SAME model object with no extra
state threaded through the caller. ``apply_edit`` is idempotent-safe: if a hook is
already registered on this layer's down_proj it is REUSED (the new entry is
appended to the existing codebook) rather than re-registered — never more than
one grace hook per module.

The harness restores model WEIGHTS between edits but knows nothing about hooks or
codebooks. Since ΔW is always 0, a weight restore is a true no-op for grace — the
only state that needs resetting between isolated per-edit measurements is the
codebook itself. Any caller that restores weights between edits (the killgate
per-edit loop) MUST also call ``clear_grace(model)``, or damage measured after
edit i+1 will silently include edit i's codebook entry too (this is the exact
analogue of a weight restore, applied to codebook state instead of W).

Public entry points: ``apply_edit(model, tok, edit_request, config, device)``,
``clear_grace(model)``.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from editors.rome_native import (  # noqa: E402
    find_subject_last_token_index, _capture_key, _optimise_value,
)

_HOOK_ATTR = "_grace_hook_handle"
_CODEBOOK_ATTR = "_grace_codebook"   # List[Tuple[Tensor[d_in] fp32, Tensor[d_out] fp32]]
_EPS_ATTR = "_grace_eps_cos"


def _grace_forward_hook(module, inputs, output):
    """Replace down_proj's output at any position whose input key cosine-matches
    the codebook above eps_cos; positions with no match pass W's real output
    through untouched."""
    codebook = getattr(module, _CODEBOOK_ATTR, None)
    if not codebook:
        return output
    eps = getattr(module, _EPS_ATTR, 0.99)
    key_in = inputs[0]                                    # [batch, seq, d_in] — down_proj INPUT
    K = torch.stack([k for k, _ in codebook]).to(key_in.device, torch.float32)   # [C, d_in]
    Kn = K / (K.norm(dim=-1, keepdim=True) + 1e-8)
    flat_in = key_in.reshape(-1, key_in.shape[-1]).float()
    flat_n = flat_in / (flat_in.norm(dim=-1, keepdim=True) + 1e-8)
    sims = flat_n @ Kn.t()                                # [batch*seq, C]
    best_sim, best_idx = sims.max(dim=-1)
    hit = best_sim >= eps
    if not bool(hit.any()):
        return output
    out = output.clone()
    flat_out = out.reshape(-1, out.shape[-1])
    V = torch.stack([v for _, v in codebook]).to(out.device, out.dtype)         # [C, d_out]
    flat_out[hit] = V[best_idx[hit]]
    return flat_out.reshape(out.shape)


def _get_or_register_hook(model, layer_idx: int):
    """Return down_proj at layer_idx, installing (or reusing) its grace hook + codebook."""
    down = model.model.layers[layer_idx].mlp.down_proj
    if getattr(down, _HOOK_ATTR, None) is None:
        handle = down.register_forward_hook(_grace_forward_hook)
        setattr(down, _HOOK_ATTR, handle)
        setattr(down, _CODEBOOK_ATTR, [])
    return down


def clear_grace(model) -> int:
    """Remove every GRACE hook + codebook attached to `model`'s MLP down_proj layers.

    Returns the number of layers that had an active grace hook (0 if none — safe
    no-op). Callers that restore weights between edits MUST call this alongside
    the weight restore (see module docstring) or codebook state silently leaks
    across edits that are supposed to be measured in isolation.
    """
    n = 0
    for layer in model.model.layers:
        down = layer.mlp.down_proj
        handle = getattr(down, _HOOK_ATTR, None)
        if handle is not None:
            handle.remove()
            delattr(down, _HOOK_ATTR)
            n += 1
        if hasattr(down, _CODEBOOK_ATTR):
            delattr(down, _CODEBOOK_ATTR)
        if hasattr(down, _EPS_ATTR):
            delattr(down, _EPS_ATTR)
    return n


def apply_edit(
    model,
    tok,
    edit_request: Dict,
    config: Dict,
    device: str = "cpu",
) -> Dict:
    """Register a GRACE codebook entry for one edit in place. ΔW is always 0.

    edit_request keys: ``prompt``, ``target_new``, and optional ``subject`` (as ROME).
    config keys (optional):
      * ``layer``          (int)   default n_layers//2  — target MLP layer.
      * ``steps``          (int)   default 25           — value-optimisation steps.
      * ``lr``             (float) default 5e-1         — value-optimisation lr.
      * ``v_weight_decay``  (float) default 1e-3         — anchor v near W k.
      * ``grace_eps_cos``  (float) default 0.99 — cosine-match threshold shared by
        every entry in this layer's codebook (a single global epsilon, not a
        learned per-entry radius — the simplification this hard-replacement
        variant makes).
    """
    prompt: str = edit_request["prompt"]
    target_new: str = edit_request["target_new"]
    subject: Optional[str] = edit_request.get("subject")

    n_layers = model.config.num_hidden_layers
    layer_idx: int = int(config.get("layer", n_layers // 2))
    steps: int = int(config.get("steps", 25))
    lr: float = float(config.get("lr", 5e-1))
    v_weight_decay: float = float(config.get("v_weight_decay", 1e-3))
    eps_cos: float = float(config.get("grace_eps_cos", 0.99))

    model.to(device)
    model.eval()

    down = _get_or_register_hook(model, layer_idx)
    setattr(down, _EPS_ATTR, eps_cos)  # last-writer-wins; one shared eps per layer's codebook

    tok_index = find_subject_last_token_index(tok, prompt, subject)

    # 1) key k (input to down_proj at the subject's last token) — _capture_key's own
    # hook only reads `inputs[0]`, so it is unaffected by hook registration order
    # relative to the grace hook (which only rewrites `output`).
    k = _capture_key(model, tok, layer_idx, prompt, tok_index, device).float()

    # 2) value v: optimise so the model predicts target_new. rome_native's
    # _optimise_value installs its own temporary `inject` hook that unconditionally
    # overwrites down_proj's output at tok_index every step, so the final value at
    # that position is always v regardless of whether the (persistent) grace hook
    # ran first — the fp32 value-opt discipline (Adam/log_softmax/v in fp32) is
    # untouched, same as rome_native and alphaedit.
    v, v0, history = _optimise_value(
        model, tok, layer_idx, prompt, tok_index,
        target_new, device, steps, lr, v_weight_decay,
    )
    v = v.float()

    # 3) S factor (‖v − W k‖) for provenance/mechanism parity with rome/alpha — W is
    # the UNMODIFIED weight (grace never writes to it), so this is well-defined even
    # though no rank-one update is ever solved or applied.
    W = model.model.layers[layer_idx].mlp.down_proj.weight
    Wk = (W.detach().float() @ k)
    residual = v - Wk

    # 4) store the entry — ΔW stays exactly 0.0.
    codebook: List = getattr(down, _CODEBOOK_ATTR)
    codebook.append((k.detach().clone(), v.detach().clone()))

    return {
        "editor": "grace",
        "layer": layer_idx,
        "subject_last_token_index": tok_index,
        "steps": steps,
        "lr": lr,
        "v_weight_decay": v_weight_decay,
        "grace_eps_cos": eps_cos,
        "key_norm": float(k.norm().item()),
        "value_norm": float(v.norm().item()),
        "value_init_norm": float(v0.float().norm().item()),
        "residual_norm": float(residual.norm().item()),
        "delta_weight_norm": 0.0,
        "n_codebook_entries": len(codebook),
        "final_value_loss": history[-1] if history else None,
    }
