"""arch_compat.py — LOAD-TIME architecture normalization to the Llama view.

Why this file exists: the whole harness (killgate_keygeom + all 4 editors)
addresses the edited weight as ``model.model.layers[li].mlp.down_proj`` — an
``nn.Linear`` whose weight is ``[d_out, d_in]`` (forward ``x @ W.T + b``).
GPT-2 exposes the same object as ``model.transformer.h[li].mlp.c_proj`` — a
``transformers`` ``Conv1D`` with the TRANSPOSED layout (weight ``[d_in, d_out]``,
forward ``x @ W + b``). Rather than teach every downstream call site both
layouts (a silent-transpose bug factory), ALL GPT-2 quirks are confined here,
at load time:

  1. every block's ``mlp.c_proj`` Conv1D is replaced by a byte-equivalent
     ``nn.Linear`` (``linear.weight = Parameter(conv.weight.T.contiguous())``;
     ``linear.bias`` is the SAME Parameter object — no copy, no drift).
     ``attn.c_proj`` is deliberately NOT touched (the harness never edits it).
  2. an EQUIVALENCE PROOF (one forward on a fixed prompt before vs after the
     swap, logits allclose) hard-stops the run if the conversion changed
     anything numerically.
  3. a Llama-shaped view ``model.model.layers[li].mlp.down_proj`` is grafted on
     as ``SimpleNamespace``s pointing at the SAME nn.Linear modules, so every
     existing reference across the killgate and the 4 editors resolves with
     ZERO edits to those call sites (keys, hooks, rank-one write-backs and the
     restore snapshot all act on the same Parameter).

GPT-J (GPTJForCausalLM, e.g. EleutherAI/gpt-j-6b) shares GPT-2's top-level
``model.transformer.h`` layout (both are pre-Llama-refactor `transformers`
families) but its MLP is GPT-NeoX-style parallel attn+MLP with
``mlp.fc_in``/``mlp.fc_out`` — and critically ``fc_out`` is ALREADY an
``nn.Linear`` (weight ``[d_out, d_in]``), never a ``Conv1D``. So GPT-J needs
NO conversion step, only step 3 (the Llama-view graft) pointed at ``fc_out``
instead of ``c_proj``; step 1/2 (Conv1D transpose + its equivalence proof) are
gpt2-only. The two families are distinguished by which attribute is present on
block 0's ``mlp`` (``c_proj`` => gpt2, ``fc_out`` => gptj) since both otherwise
look identical from ``hasattr(model, "transformer")``/``hasattr(.., "h")``.

GPT-NeoX (GPTNeoXForCausalLM, e.g. EleutherAI/gpt-neox-20b) is a THIRD,
separate top-level layout: ``model.gpt_neox.layers`` (not ``model.transformer.
h`` at all — GPT-J's "GPT-NeoX-style parallel attn+MLP" note above describes
its MLP shape only, not this top-level container). Its MLP is
``mlp.dense_h_to_4h`` / ``mlp.dense_4h_to_h``, and like GPT-J's ``fc_out``,
``dense_4h_to_h`` is ALREADY an ``nn.Linear`` (weight ``[d_out, d_in]``) — so
GPT-NeoX needs NO conversion step either, only the step-3 graft pointed at
``dense_4h_to_h``. Detected via ``hasattr(model, "gpt_neox") and hasattr(model.
gpt_neox, "layers")``, checked strictly after the transformer.h dispatch so a
GPT-2/GPT-J model (which has no ``gpt_neox`` attribute) never reaches this
branch and vice versa.

Native Llama-family models are returned untouched ("native") — this function
is a no-op on every pre-existing code path.

Known limitation (guarded in the killgate, not here): editors/memit.py's
``_hidden_at`` hooks the DECODER-LAYER Module itself (``model.model.layers[l]``)
for the residual stream; under the graft that object is a plain SimpleNamespace
with no ``register_forward_hook`` — so memit is fenced out on GPT-2, GPT-J AND
GPT-NeoX.
"""
from __future__ import annotations

import types

import torch

# fixed, deterministic prompt for the conversion equivalence proof
_PROOF_PROMPT = "The Eiffel Tower is located in the city of"


def normalize_arch(model, tok, device: str) -> str:
    """Detect + normalize the model architecture. Returns "native", "gpt2", "gptj", or
    "gptneox".

    "native"  — model already exposes model.model.layers[li].mlp.down_proj
                (Llama/Qwen/gemma/Phi families). NOTHING is touched.
    "gpt2"    — model.transformer.h[li].mlp.c_proj (Conv1D) converted to
                nn.Linear + Llama view grafted (see module docstring).
    "gptj"    — model.transformer.h[li].mlp.fc_out (already nn.Linear, GPT-J/
                GPT-NeoX-style parallel attn+MLP) — no conversion, Llama view
                grafted directly onto fc_out (see module docstring).
    "gptneox" — model.gpt_neox.layers[li].mlp.dense_4h_to_h (already nn.Linear)
                — no conversion, Llama view grafted directly onto
                dense_4h_to_h (see module docstring).
    Anything else -> SystemExit (an unsupported family must fail loudly, not
    produce silently-wrong geometry).
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return "native"  # pre-existing path: untouched by construction
    has_transformer_h = hasattr(model, "transformer") and hasattr(model.transformer, "h")
    has_gpt_neox = hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers")
    if not (has_transformer_h or has_gpt_neox):
        raise SystemExit("[arch_compat] unsupported architecture: model has neither "
                         "model.model.layers (Llama view), transformer.h (GPT-2/GPT-J), "
                         "nor gpt_neox.layers (GPT-NeoX)")

    # transformers maps num_hidden_layers -> n_layer/num_hidden_layers via attribute_map
    # for each family; verify instead of assuming (the killgate's nL and --layer bounds
    # depend on it).
    blocks = model.transformer.h if has_transformer_h else model.gpt_neox.layers
    n_blocks = len(blocks)
    if int(model.config.num_hidden_layers) != n_blocks:
        raise SystemExit(f"[arch_compat] config.num_hidden_layers="
                         f"{model.config.num_hidden_layers} != len(blocks)={n_blocks}")

    if has_transformer_h:
        # distinguish the two transformer.h families by their block-0 mlp attribute —
        # both look identical up to this point (hasattr(model,"transformer")/hasattr(..,"h")).
        first_mlp = blocks[0].mlp
        if hasattr(first_mlp, "c_proj"):
            family = "gpt2"
        elif hasattr(first_mlp, "fc_out"):
            family = "gptj"
        else:
            raise SystemExit(f"[arch_compat] transformer.h present but block-0 mlp "
                             f"({type(first_mlp).__name__}) has neither c_proj (GPT-2 Conv1D) "
                             f"nor fc_out (GPT-J nn.Linear) — unrecognized family")
    else:
        family = "gptneox"

    # GPT-2/GPT-J/GPT-NeoX ship no pad token; killgate/metrics only do single-prompt
    # encodes (never pad), but generate_text falls back to eos anyway — set it so any
    # future batched path is safe rather than crashing on tokenizer(pad=True).
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # ---- equivalence-proof reference logits BEFORE any change ----
    enc = tok(_PROOF_PROMPT, return_tensors="pt").to(device)
    with torch.no_grad():
        logits_before = model(**enc).logits.detach().clone()

    if family == "gpt2":
        # ---- Conv1D -> nn.Linear, byte-equivalent, mlp.c_proj ONLY ----
        for b in blocks:
            conv = b.mlp.c_proj                   # Conv1D: weight [d_in, d_out], y = x@W + b
            d_in, d_out = conv.weight.shape
            lin = torch.nn.Linear(d_in, d_out, bias=True,
                                  device=conv.weight.device, dtype=conv.weight.dtype)
            with torch.no_grad():
                # transpose ONCE at load; .contiguous() so downstream outer-product
                # add_ / copy_ restore paths behave exactly like a native Linear
                lin.weight = torch.nn.Parameter(conv.weight.detach().t().contiguous())
                lin.bias = conv.bias              # SAME Parameter object (no copy)
            b.mlp.c_proj = lin                    # nn.Module setattr: swaps the submodule
        down_attr = "c_proj"
    elif family == "gptj":  # fc_out is ALREADY nn.Linear; nothing to convert
        for b in blocks:
            if not isinstance(b.mlp.fc_out, torch.nn.Linear):
                raise SystemExit(f"[arch_compat] gptj: mlp.fc_out is "
                                 f"{type(b.mlp.fc_out).__name__}, expected nn.Linear — "
                                 f"architecture assumption violated, refusing to graft")
        down_attr = "fc_out"
    else:  # family == "gptneox" — dense_4h_to_h is ALREADY nn.Linear; nothing to convert
        for b in blocks:
            if not isinstance(b.mlp.dense_4h_to_h, torch.nn.Linear):
                raise SystemExit(f"[arch_compat] gptneox: mlp.dense_4h_to_h is "
                                 f"{type(b.mlp.dense_4h_to_h).__name__}, expected nn.Linear — "
                                 f"architecture assumption violated, refusing to graft")
        down_attr = "dense_4h_to_h"

    # ---- equivalence PROOF: the change (or, for gptj/gptneox, the graft below) must be
    #      a numerical no-op ----
    with torch.no_grad():
        logits_after = model(**enc).logits
    # fp32: x@W.T+b (addmm) vs Conv1D's addmm(b, x, W) may differ only in reduction
    # order. 1e-4 (not 1e-5): drift scales with the contraction dim, and GPT2-XL's
    # ~6400-dim MLP can exceed 1e-5 on some BLAS while still being 4+ orders below
    # logit magnitude (hostile-review finding 2026-07-02). Loud SystemExit either
    # way — never silent-wrong. bf16 rounding needs a looser bar. gptj/gptneox touch
    # no weight at all, so this is a pure sanity check that nothing upstream mutated
    # state; the bar is the same fp32/bf16 split for consistency.
    atol = 1e-4 if next(model.parameters()).dtype == torch.float32 else 5e-2
    if not torch.allclose(logits_before, logits_after, atol=atol):
        raise SystemExit(f"[arch_compat] {family} equivalence proof FAILED: "
                         f"max|Δlogit|={float((logits_before - logits_after).abs().max()):.3g} "
                         f"> atol={atol}")

    # ---- graft the Llama-compatible view (namespaces wrap the LIVE modules) ----
    view = types.SimpleNamespace(layers=[
        types.SimpleNamespace(mlp=types.SimpleNamespace(down_proj=getattr(b.mlp, down_attr)))
        for b in blocks
    ])
    try:
        # nn.Module.__setattr__ stores non-Module/non-Parameter values in __dict__;
        # "model" collides with no GPT-2/GPT-J param/buffer/submodule name, so this
        # works — the except branch is belt-and-braces for exotic subclasses.
        model.model = view
    except (TypeError, AttributeError):
        object.__setattr__(model, "model", view)

    print(f"[arch_compat] {family} normalized: {n_blocks} mlp.{down_attr} "
          f"{'Conv1D->Linear' if family == 'gpt2' else '(already nn.Linear, graft-only)'} "
          f"(equivalence proof max|Δlogit|="
          f"{float((logits_before - logits_after).abs().max()):.3g}, atol={atol})", flush=True)
    return family
