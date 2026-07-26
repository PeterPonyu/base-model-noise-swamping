"""alphaedit.py — native AlphaEdit-style editor (null-space projected ROME).

For G4 (the ONLY causal test in the key-geometry paper) and D3 (geometry-gated
routing). AlphaEdit (Fang et al., ICLR 2025) applies the ROME rank-one update
but projects it into the NULL SPACE of the preserved-knowledge key covariance
C_K = K_p^T K_p, so the edit barely disturbs facts whose keys live in C_K's
column space.

Mechanism prediction this editor is built to test (G4):
  vanilla ROME damages a probe ∝ cos(k_edit, k_probe).  AlphaEdit projects
  k_edit off the preserved-key subspace, so it should disproportionately PROTECT
  high-cosine probes (their keys are in that subspace) while low-cosine probe
  damage is ~unchanged. If true, that is causal evidence for the geometry account.

Reuses rome_native's key-capture + value-optimisation; only the final rank-one
update direction differs (P·k instead of k).

    ROME:      ΔW = (v − W k) kᵀ / (kᵀ k)
    AlphaEdit: ΔW = (v − W k) (P k)ᵀ / (kᵀ P k),   P = I − U_r U_rᵀ
    where U_r spans the top-r eigenvectors of C_K (the preserved-key subspace).

STATUS: pre-written, NOT yet GPU-tested. Integration TODO in killgate: build the
projector once from the probe-bank keys K_probe (already captured there) and pass
config["projector"], or pass config["preserved_keys"]=K_probe.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, Optional

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from editors.rome_native import (  # noqa: E402
    find_subject_last_token_index, _capture_key, _optimise_value,
)
# resolve_layer_device/safe_model_to (tp_edit_util.py) are imported LAZILY, inside
# _resolve_projector/apply_edit below, not here at module top level — mirrors editors/
# rome_native.py's own header comment (killgate_keygeom.py imports THAT module
# unconditionally at its own top level for every --editor, so a broken tp_edit_util there
# would break every editor). This module is only imported by killgate when --editor alpha
# is actually selected (already conditional at the killgate call site), but keeping the
# import lazy here too means a tp_edit_util breakage can never take down
# build_null_projector — the by-construction-reference code path this module also
# exports — which doesn't touch tp_edit_util at all.

# cache projectors per (model id, layer, keep_ratio) so the eigdecomp runs once
_PROJ_CACHE: Dict[tuple, torch.Tensor] = {}


def build_null_projector(K_p: torch.Tensor, keep_ratio: float = 0.99) -> torch.Tensor:
    """Null-space projector P (d×d) that projects OUT the preserved-key subspace.

    K_p: [P, d] preserved keys (each row a key vector at the edited layer).
    keep_ratio: fraction of C_K spectral energy treated as the "preserved"
    subspace to remove (0.99 → project out directions holding 99% of key energy).
    """
    K_p = K_p.float()
    C = K_p.t() @ K_p                                   # [d, d] second moment
    evals, evecs = torch.linalg.eigh(C)                 # ascending eigenvalues
    total = float(evals.sum().clamp(min=1e-12))
    desc = torch.flip(evals, [0])
    cum = torch.cumsum(desc, 0) / total
    r = int((cum < keep_ratio).sum().item()) + 1        # top-r directions = preserved subspace
    r = max(1, min(r, evecs.shape[1] - 1))
    U_r = evecs[:, -r:]                                 # [d, r] top-r eigenvectors
    d = C.shape[0]
    P = torch.eye(d, device=C.device, dtype=C.dtype) - U_r @ U_r.t()
    return P  # rank removed = d − trace(P); recover from P itself (no fragile attribute)


def _resolve_projector(model, tok, layer_idx, config, device, k_dim) -> Optional[torch.Tensor]:
    """Get the null-space projector from config, building/caching as needed.

    TP-safety: `P` must live on the EDITED layer's device — it is matmul'd against
    `k` below, and `k` (captured by _capture_key's forward hook) already lives on
    that same layer's device. `device` here is still used, correctly, only to encode
    `preserved_prompts` (an INPUT-side forward pass) when the projector has to be
    built from scratch. Single-device path (no --device_map): layer_device == device
    always, so every `.to()` below is a byte-identical no-op vs. the old `.to(device)`.
    """
    from tp_edit_util import resolve_layer_device  # noqa: E402 (lazy — see module header)
    layer_device = resolve_layer_device(model, layer_idx)
    if config.get("projector") is not None:
        return config["projector"].to(layer_device)
    key = (id(model), layer_idx, float(config.get("keep_ratio", 0.99)))
    if key in _PROJ_CACHE:
        return _PROJ_CACHE[key].to(layer_device)
    K_p = config.get("preserved_keys")
    if K_p is None:
        # build from preserved prompts if given, else no projection (= vanilla ROME)
        prompts = config.get("preserved_prompts")
        subjects = config.get("preserved_subjects")
        if not prompts:
            return None
        rows = []
        for p, s in zip(prompts, subjects or [None] * len(prompts)):
            idx = find_subject_last_token_index(tok, p, s)
            rows.append(_capture_key(model, tok, layer_idx, p, idx, device).float())
        K_p = torch.stack(rows)
    if not torch.is_tensor(K_p):
        K_p = torch.as_tensor(K_p)
    P = build_null_projector(K_p.to(layer_device), float(config.get("keep_ratio", 0.99)))
    _PROJ_CACHE[key] = P
    return P


def apply_edit(model, tok, edit_request: Dict, config: Dict, device: str = "cpu") -> Dict:
    """AlphaEdit rank-one edit (null-space projected). Same call signature as ROME/FT.

    config: layer, steps, lr, v_weight_decay (as ROME) PLUS one of:
      projector [d,d] | preserved_keys [P,d] | preserved_prompts(+preserved_subjects).
      keep_ratio (default 0.99). If no preserved info → falls back to vanilla ROME.
    """
    prompt = edit_request["prompt"]
    target_new = edit_request["target_new"]
    subject = edit_request.get("subject")

    n_layers = model.config.num_hidden_layers
    layer_idx = int(config.get("layer", n_layers // 2))
    steps = int(config.get("steps", 25))
    lr = float(config.get("lr", 5e-1))
    v_wd = float(config.get("v_weight_decay", 1e-3))

    from tp_edit_util import safe_model_to  # noqa: E402 (lazy — see module header)
    safe_model_to(model, device); model.eval()  # no-op under --device_map TP — see
    # editors/rome_native.py::apply_edit's identical comment / tp_edit_util.py
    tok_index = find_subject_last_token_index(tok, prompt, subject)
    k = _capture_key(model, tok, layer_idx, prompt, tok_index, device).float()
    v, v0, history = _optimise_value(model, tok, layer_idx, prompt, tok_index,
                                     target_new, device, steps, lr, v_wd)
    v = v.float()

    P = _resolve_projector(model, tok, layer_idx, config, device, k.shape[0])
    projected = P is not None
    Pk = (P @ k) if projected else k                    # null-space-projected key

    W = model.model.layers[layer_idx].mlp.down_proj.weight
    W_dtype = W.dtype
    Wk = (W.detach().float() @ k)
    residual = (v - Wk)
    denom = float((k @ Pk).item()) + 1e-8
    delta = torch.outer(residual, Pk) / denom

    with torch.no_grad():
        before = W.detach().clone()
        W.add_(delta.to(W_dtype))
        applied_norm = float((W.detach() - before).norm().item())

    new_Wk = (W.detach().float() @ k)
    return {
        "editor": "alphaedit",
        "layer": layer_idx,
        "projected": projected,
        "rank_removed": int(round(P.shape[0] - float(torch.trace(P)))) if projected else 0,
        "key_norm": float(k.norm().item()),
        "value_norm": float(v.norm().item()),
        "residual_norm": float(residual.norm().item()),
        "delta_weight_norm": applied_norm,
        "rank_one_solve_residual": float((new_Wk - v).norm().item()),
        "final_value_loss": history[-1] if history else None,
        "keep_ratio": float(config.get("keep_ratio", 0.99)),
    }
