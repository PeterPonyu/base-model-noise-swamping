"""tp_edit_util.py — device-resolution helpers for editing under accelerate's
device_map (tensor-parallel / multi-GPU sharded) model placement.

Why this file exists: killgate_keygeom.py + editors/rome_native.py + editors/
alphaedit.py all thread a single `device` string/torch.device through their call
chains. That is correct for the INPUT side (tokenizer encode -> first forward):
under accelerate dispatch (--device_map), its forward hooks move activations
across shard boundaries automatically, so every `tokenizer(...).to(device)` call
site in this harness only ever needs the model's INPUT (embedding) device.

It is WRONG for any tensor that is built from scratch and then combined
arithmetically with the EDITED LAYER's own weight/activation — under
--device_map that layer can sit on a different card than the embedding (e.g.
NeoX-20B's early layers on cuda:0, later layers on cuda:1, with the ~44-layer
model split roughly in half across the two 24GB cards). Two call sites in this
harness build such a tensor and previously routed it through the wrong (input)
device: the ROME/AlphaEdit optimized value `v` (editors/rome_native.py::
_optimise_value) and the AlphaEdit null-space projector `P` (editors/
alphaedit.py::_resolve_projector). Both now route through resolve_layer_device()
below instead of the ambient `device` parameter. A third call site,
editors/rome_native.py::apply_edit's `model.to(device)`, is even more acute: run
unconditionally it COLLAPSES an accelerate-dispatched model back onto one device
on every single apply_edit() call — since the killgate calls apply_edit() once
per edit, an unmodified TP run would silently undo its own --device_map sharding
on the second edit of any cell. safe_model_to() below guards that.

Single-device path (no --device_map): every layer's down_proj.weight lives on
the SAME device the killgate resolved for `device`, so resolve_layer_device
always returns that same device and safe_model_to always finds hf_device_map
absent — every fix wired to this module is a byte-identical no-op there,
verified by the CPU smoke (experiments/smoke_neox20b_cpu.py).
"""
from __future__ import annotations

import torch


def resolve_layer_device(model, layer_idx: int) -> torch.device:
    """The actual device the edited layer's down_proj weight lives on.

    Pure attribute read — no forward pass, no allocation, so it is safe to call
    before any tensor touching that layer exists yet. Works identically whether
    model.model.layers is a real Llama-family ModuleList or the Llama-shaped
    view arch_compat.py grafts onto GPT-2/GPT-J/GPT-NeoX (a SimpleNamespace
    wrapping the SAME live nn.Linear, so .weight.device resolves to the real
    device either way — the graft never changes where a weight lives).
    """
    return model.model.layers[layer_idx].mlp.down_proj.weight.device


def safe_model_to(model, device) -> None:
    """model.to(device) IF AND ONLY IF the model is not accelerate-dispatched.

    A plain model.to(device) collapses an accelerate device_map (TP) placement
    back onto a single device — calling it unconditionally inside apply_edit()
    (as every editor did before this pass) would silently UNDO --device_map
    sharding on the very next edit of a run. accelerate stamps `hf_device_map`
    on any model loaded via `from_pretrained(..., device_map=...)`; skip the
    collapse when present. The default single-device path (no --device_map)
    never carries that attribute, so this is byte-identical to the old
    unconditional `.to()` call it replaces.
    """
    if not hasattr(model, "hf_device_map"):
        model.to(device)


def resolve_input_device(model, device):
    """Re-resolve the INPUT (tokenizer-encode) device after a possible accelerate
    device_map load. Mirrors experiments/killgate_keygeom.py's post-load re-resolution
    (`device = model.get_input_embeddings().weight.device`): under --device_map the
    embedding need not land on `device` verbatim (e.g. balanced_low_0 reserves card 0
    for other work), so every downstream `tokenizer(...).to(device)` call site must use
    the REAL embedding device, not the string the caller originally requested.

    Single-device path (no --device_map, no `hf_device_map` attribute): returns `device`
    unchanged — byte-identical no-op, matching resolve_layer_device/safe_model_to's
    convention above. Added 2026-07-16 for merging_m0.py's --device_map plumbing (R-C
    revision-wave cell); pure attribute read, safe to call any time after model load.
    """
    if hasattr(model, "hf_device_map"):
        return model.get_input_embeddings().weight.device
    return device
