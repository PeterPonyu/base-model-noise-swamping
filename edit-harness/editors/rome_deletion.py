"""rome_deletion.py — ROME-style rank-one DELETION editor (eos / suppress objectives).

For U1-E0. Same ``apply_edit(model, tok, edit_request, config, device) -> info``
contract as every editor. rome_native.py gets ZERO diff — its value-opt loop
hardcodes NLL(target_new), so the deletion objectives are reimplemented here:

  * eos:      drive the next token toward tok.eos_token_id (taken DIRECTLY —
              never via the target string: metrics.target_token_ids prepends a
              leading space, so ' <eos>' first-token != eos_id).
  * suppress: bounded unlikelihood on the TRUE target's first token:
              loss = -log(clamp(1 - p_true, min=1e-6))  (raw +logp diverges).

The rank-one update is copied from rome_native (delta = outer(residual, k)/(k@k+1e-8)),
with an OPTIONAL config['projector'] applied exactly as alphaedit applies it
(Pk = P@k; denom = k@Pk; delta = outer(residual, Pk)/denom) for the future E5
deletion arm; default None. All value-opt math in fp32 (fp16 silently NaNs).

NO restore inside — the runner snapshots/restores (ft_editor convention).
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from editors.rome_native import (  # noqa: E402
    find_subject_last_token_index, _capture_key,
)
from metrics import first_target_token_id  # noqa: E402


def _optimise_value_deletion(model, tokenizer, layer_idx: int, prompt: str, tok_index: int,
                             variant: str, target_true: Optional[str], device: str,
                             steps: int, lr: float, v_weight_decay: float):
    """Clone of rome_native._optimise_value with the deletion objective switch."""
    down = model.model.layers[layer_idx].mlp.down_proj
    for p in model.parameters():
        p.requires_grad_(False)

    init_holder = {}

    def capture_out(_m, _i, output):
        init_holder["v0"] = output[0, tok_index, :].detach().clone()

    h0 = down.register_forward_hook(capture_out)
    with torch.no_grad():
        enc = tokenizer(prompt, return_tensors="pt").to(device)
        model(**enc)
    h0.remove()

    # bf16 boundary (same pattern as rome_native._optimise_value): the v parameter,
    # Adam state and anchor stay fp32 under any model dtype; .float() is an exact
    # no-op on the fp32 default path.
    v = init_holder["v0"].clone().detach().float().to(device).requires_grad_(True)
    v0 = init_holder["v0"].clone().detach().float()

    def inject(_m, _i, output):
        output = output.clone()
        # bf16 boundary: cast the injected fp32 v to the hidden-state dtype so bf16
        # downstream matmuls never see fp32 (grad still reaches the fp32 v); exact
        # no-op under fp32.
        output[0, tok_index, :] = v.to(output.dtype)
        return output

    if variant == "eos":
        if tokenizer.eos_token_id is None:
            raise SystemExit("[rome_deletion] eos variant: tokenizer.eos_token_id is None — "
                             "refusing to optimise a wrong token")
        tgt_id = int(tokenizer.eos_token_id)
        true_id = None
    elif variant == "suppress":
        if not target_true:
            raise SystemExit("[rome_deletion] suppress variant needs edit_request['target_true']")
        true_id = first_target_token_id(tokenizer, target_true)
        tgt_id = None
    else:
        raise SystemExit(f"[rome_deletion] unknown delete_variant {variant!r} "
                         f"(refusal is data-layer, handled by rome_native)")

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
        if variant == "eos":
            logp = torch.log_softmax(logits.float(), dim=-1)
            obj = -logp[tgt_id]
        else:  # suppress — bounded unlikelihood in fp32
            probs = torch.softmax(logits.float(), dim=-1)
            p_true = probs[true_id]
            obj = -torch.log(torch.clamp(1.0 - p_true, min=1e-6))
        reg = v_weight_decay * ((v - v0) ** 2).sum()
        loss = obj + reg
        loss.backward()
        opt.step()
        history.append(float(loss.detach().item()))

    target_first_id = tgt_id if variant == "eos" else true_id
    return v.detach(), v0, history, target_first_id


def apply_edit(model, tok, edit_request: Dict, config: Dict, device: str = "cpu") -> Dict:
    """Rank-one deletion edit in place (eos/suppress variants)."""
    prompt = edit_request["prompt"]
    subject = edit_request.get("subject")
    target_true = edit_request.get("target_true")

    n_layers = model.config.num_hidden_layers
    layer_idx = int(config.get("layer", n_layers // 2))
    steps = int(config.get("steps", 25))
    lr = float(config.get("lr", 5e-1))
    v_wd = float(config.get("v_weight_decay", 1e-3))
    variant = str(config.get("delete_variant", "eos"))

    model.to(device)
    model.eval()

    tok_index = find_subject_last_token_index(tok, prompt, subject)
    k = _capture_key(model, tok, layer_idx, prompt, tok_index, device).float()
    v, v0, history, target_first_id = _optimise_value_deletion(
        model, tok, layer_idx, prompt, tok_index, variant, target_true,
        device, steps, lr, v_wd)
    v = v.float()

    # optional null-space projector (alphaedit application pattern) for the E5 deletion arm
    P = config.get("projector")
    W = model.model.layers[layer_idx].mlp.down_proj.weight
    W_dtype = W.dtype
    Wk = (W.detach().float() @ k)
    residual = (v - Wk)
    if P is not None:
        P = P.to(device)
        Pk = P @ k
        denom = float((k @ Pk).item()) + 1e-8
        delta = torch.outer(residual, Pk) / denom
    else:
        denom = float((k @ k).item()) + 1e-8
        delta = torch.outer(residual, k) / denom

    with torch.no_grad():
        before = W.detach().clone()
        W.add_(delta.to(W_dtype))
        applied_norm = float((W.detach() - before).norm().item())

    return {
        "editor": "rome_deletion",
        "delete_variant": variant,
        "layer": layer_idx,
        "projected": P is not None,
        "key_norm": float(k.norm().item()),
        "value_norm": float(v.norm().item()),
        "residual_norm": float(residual.norm().item()),   # the S factor
        "delta_weight_norm": applied_norm,
        "target_first_id": int(target_first_id),
        "value_loss_history": history,
        "final_value_loss": history[-1] if history else None,
        "covariance_used": False,
    }
