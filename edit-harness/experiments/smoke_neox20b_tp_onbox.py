"""smoke_neox20b_tp_onbox.py — ON-BOX preflight smoke for run_neox20b.sh (WP3,
2026-07-08). Exercises the REAL accelerate.dispatch_model mechanism (the same one
transformers' `from_pretrained(..., device_map=...)` calls internally) on a TINY
synthetic 2-layer GPTNeoXForCausalLM split across the box's actual GPU cards, one
layer per card, BEFORE the real ~40GB NeoX-20B checkpoint is downloaded or loaded.

Why this exists (see experiments/smoke_neox20b_cpu.py's own docstring for the full
picture): that CPU smoke proves every piece of this build's logic in isolation —
including tp_edit_util.resolve_layer_device/safe_model_to against MOCKED
heterogeneous devices, since no free second GPU was available where this build was
authored (a single local card, reserved for a concurrent job — see this file's own
author-time validation note below). It explicitly could NOT prove that accelerate's
dispatch really puts two layers on two different PHYSICAL cards and moves
activations across that boundary during a real forward/backward pass. THIS script
closes that gap on the real box, in seconds, using a model ~9 orders of magnitude
smaller than the real one — if this fails, the real 20B run will fail identically
(same code path, same device_map shape) but ~40GB and hours later.

Usage (on the AutoDL box, both cards attached, called from run_neox20b.sh's preflight):
    python3 experiments/smoke_neox20b_tp_onbox.py
Exits 0 (PASS — safe to proceed to the download) or 1 (FAIL — do not download).

--devices a,b overrides the auto-detected pair (default: cuda:0,cuda:1 when >=2 CUDA
devices are visible, else a hard FAIL — <2 GPUs means this box cannot tensor-parallel-
shard NeoX-20B at all, so failing HERE, before any download, is the whole point).
AUTHOR-TIME VALIDATION NOTE: this exact script (device_map SHAPE, dispatch_model
call, resolve_layer_device/safe_model_to wiring, the ROME+AlphaEdit edit calls) was
run and verified PASS with `--devices cpu,cpu` on the build box, which has only one
CUDA device and that device was off-limits (reserved for a concurrent GPU job) — see
the WP3 build report. That run proves the device_map SHAPE and the full edit-call
chain end-to-end via accelerate's REAL dispatch_model API (not a mock); only the
device VALUES (cpu -> cuda:0/cuda:1) are untested and are exactly what this script
exists to check on the real box.
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

HARNESS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HARNESS)

try:
    from experiments.smoke_neox20b_cpu import _build_tiny_gptneox, _build_tiny_tokenizer
except ModuleNotFoundError:  # invoked as `python3 experiments/smoke_neox20b_tp_onbox.py`
    from smoke_neox20b_cpu import _build_tiny_gptneox, _build_tiny_tokenizer  # noqa: E402
from editors.arch_compat import normalize_arch  # noqa: E402
from editors.rome_native import apply_edit as rome_apply_edit  # noqa: E402
from editors.alphaedit import apply_edit as alpha_apply_edit  # noqa: E402
from tp_edit_util import resolve_layer_device, safe_model_to  # noqa: E402


def _default_devices():
    n = torch.cuda.device_count()
    return ["cuda:0", "cuda:1"] if n >= 2 else None


def _device_map(d0, d1):
    """gpt_neox.layers.0 -> d0, gpt_neox.layers.1 -> d1: the two-card split under
    test. Every OTHER submodule must also be covered (accelerate's dispatch_model
    requires full coverage) — embed_in/emb_dropout/rotary_emb ride with layer 0's
    card (the input side), final_layer_norm/embed_out ride with layer 1's card (the
    output side), mirroring how a real device_map="auto"/"balanced" split of a
    44-layer NeoX-20B would keep the embedding with the first shard and the head
    with the last."""
    return {
        "gpt_neox.embed_in": d0, "gpt_neox.emb_dropout": d0, "gpt_neox.rotary_emb": d0,
        "gpt_neox.layers.0": d0, "gpt_neox.layers.1": d1,
        "gpt_neox.final_layer_norm": d1, "embed_out": d1,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", default=None,
                    help="comma pair of accelerate device targets, e.g. cuda:0,cuda:1. "
                         "Default: auto (cuda:0,cuda:1 if >=2 CUDA devices visible, else FAIL).")
    args = ap.parse_args()

    if args.devices:
        d0, d1 = args.devices.split(",")
    else:
        auto = _default_devices()
        if auto is None:
            print(f"[tp-onbox-smoke] FAIL: torch.cuda.device_count()="
                  f"{torch.cuda.device_count()} < 2 — this box cannot tensor-parallel-shard "
                  f"NeoX-20B. Do NOT proceed to the 20B download/run.", flush=True)
            sys.exit(1)
        d0, d1 = auto

    from accelerate import dispatch_model
    ok = True

    def check(name, cond, detail=None):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"[tp-onbox-smoke] {name}: {'PASS' if cond else 'FAIL'} {detail or ''}", flush=True)

    model, cfg = _build_tiny_gptneox()
    tok = _build_tiny_tokenizer()
    dmap = _device_map(d0, d1)
    model = dispatch_model(model, device_map=dmap)
    check("dispatch_model_real", hasattr(model, "hf_device_map"),
          {"hf_device_map": model.hf_device_map})

    arch = normalize_arch(model, tok, d0)
    check("normalize_arch_gptneox_cross_device", arch == "gptneox", {"arch": arch})

    dev_l0 = resolve_layer_device(model, 0)
    dev_l1 = resolve_layer_device(model, 1)
    # the distinctness half of this check is only meaningful when d0 != d1 (it is
    # vacuously true under this build's own --devices cpu,cpu degenerate validation
    # run, where d0 == d1 by construction — see the module docstring).
    lands_correctly = (str(dev_l0).startswith(d0.split(":")[0])
                       and str(dev_l1).startswith(d1.split(":")[0]))
    distinct_ok = (d0 == d1) or (str(dev_l0) != str(dev_l1))
    check("layers_land_on_the_requested_devices",
          lands_correctly and distinct_ok,
          {"layer0": str(dev_l0), "layer1": str(dev_l1), "requested": [d0, d1]})

    # safe_model_to must be a no-op post-dispatch (else every apply_edit() call would
    # collapse this split right back onto one device on the very first edit).
    before = model.model.layers[1].mlp.down_proj.weight.data_ptr()
    safe_model_to(model, d0)
    after = model.model.layers[1].mlp.down_proj.weight.data_ptr()
    check("safe_model_to_does_not_collapse_the_split", before == after)

    req = {"prompt": "The capital of Testland is the city of", "subject": "Testland",
          "target_new": "wonderful"}
    # Edit the layer that sits on the SECOND device while `device` (input encoding)
    # points at the FIRST — the exact cross-device case this build's fix targets
    # (rome_native._optimise_value's v / alphaedit._resolve_projector's P used to be
    # built on the ambient input device, not the edited layer's own device).
    # EXPECTED near-zero delta_weight_norm: layer 1 IS the last of this 2-layer tiny
    # model, so a mid-prompt subject-token edit there has a near-degenerate gradient
    # path to the final-position logits — the same documented artifact as
    # experiments/smoke_memit_cpu.py's T1 (real L12-of-16 cells never hit this). The
    # check below only asserts finite + nonzero, which holds regardless.
    snap1 = model.model.layers[1].mlp.down_proj.weight.detach().clone()
    try:
        info = rome_apply_edit(model, tok, req, {"layer": 1, "steps": 3, "lr": 0.1}, d0)
        W1 = model.model.layers[1].mlp.down_proj.weight.detach()
        check("rome_edit_on_the_second_devices_layer",
              bool(torch.isfinite(W1).all()) and bool((W1 - snap1).abs().max() > 0),
              {"delta_weight_norm": info.get("delta_weight_norm")})
    except Exception as e:
        check("rome_edit_on_the_second_devices_layer", False, {"err": repr(e)})
    finally:
        with torch.no_grad():
            model.model.layers[1].mlp.down_proj.weight.copy_(snap1)

    try:
        info2 = alpha_apply_edit(model, tok, req, {"layer": 1, "steps": 3, "lr": 0.1}, d0)
        W1b = model.model.layers[1].mlp.down_proj.weight.detach()
        check("alpha_edit_on_the_second_devices_layer",
              bool(torch.isfinite(W1b).all()) and bool((W1b - snap1).abs().max() > 0),
              {"delta_weight_norm": info2.get("delta_weight_norm")})
    except Exception as e:
        check("alpha_edit_on_the_second_devices_layer", False, {"err": repr(e)})
    finally:
        with torch.no_grad():
            model.model.layers[1].mlp.down_proj.weight.copy_(snap1)

    print(f"[tp-onbox-smoke] "
          f"{'ALL PASS — safe to proceed to the 20B download/run' if ok else 'FAIL — DO NOT proceed'}",
          flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
