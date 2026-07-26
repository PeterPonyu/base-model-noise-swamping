"""smoke_neox20b_cpu.py — standalone CPU smoke for the GPT-NeoX-20B TP-editing build
(WP3, 2026-07-08). Tests T0-T5 below. NO DOWNLOAD, NO GPU: builds a tiny synthetic
GPTNeoXForCausalLM in memory (hidden=32, layers=2, heads=4, intermediate=128) so the
whole graft+edit pipeline is exercised without the real ~40GB checkpoint.

Two things this smoke CANNOT prove, by construction (no CUDA on this box): that
accelerate's device_map actually splits a model across two physical cards, and that
tensors really move between them. What it DOES prove:
  T0-T2: arch_compat.py's new "gptneox" branch grafts correctly and is a numerical
         no-op (equivalence proof), and _capture_key/find_subject_last_token_index
         resolve through the graft with zero code changes (same guarantee gpt2/gptj
         already had).
  T3:    a ROME edit applies through the graft with no NaN.
  T4:    tp_edit_util.resolve_layer_device reads the ACTUAL per-layer weight device,
         not any ambient default — proven with a MOCKED model whose two layers are
         fabricated on two DIFFERENT torch.device objects (torch.device("cuda:N") is
         constructible without a GPU present; it is never used to allocate real
         storage here, only read back as an attribute). This is the load-bearing
         property that keeps editors/rome_native.py::_optimise_value's `v` and
         editors/alphaedit.py::_resolve_projector's `P` on the correct card under
         real 2-GPU TP.
  T5:    tp_edit_util.safe_model_to only calls .to() when hf_device_map is ABSENT —
         proven by spying on nn.Module.to for two model stand-ins, one with the
         attribute stamped (accelerate-dispatched, real TP) and one without
         (single-device default path).
The genuinely untested residual risk — whether accelerate really places NeoX-20B's
layers across two physical AutoDL cards and whether the forward/backward graph moves
activations across that boundary the way this smoke assumes — is deferred to
run_neox20b.sh's on-box preflight (see that file's Phase 0a3), which runs this exact
graft+edit pipeline again but on a REAL 2-GPU device_map before the 20B download.

Smoke gates are STRUCTURAL asserts, never efficacy (esr=0 at steps=2 is expected).
"""
from __future__ import annotations

import json
import os
import sys

import torch

HARNESS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HARNESS)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from editors.arch_compat import normalize_arch  # noqa: E402
from editors.rome_native import (  # noqa: E402
    apply_edit, _capture_key, find_subject_last_token_index,
)
from tp_edit_util import resolve_layer_device, safe_model_to  # noqa: E402

DEVICE = "cpu"
OUT = os.path.join(HARNESS, "results", "smoke_infra", "neox20b_smoke.json")

TINY_CFG = dict(vocab_size=256, hidden_size=32, num_hidden_layers=2,
                num_attention_heads=4, intermediate_size=128,
                max_position_embeddings=64)


def _build_tiny_gptneox():
    """A tiny synthetic GPTNeoXForCausalLM, built in-memory — no download, no network."""
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
    cfg = GPTNeoXConfig(**TINY_CFG)
    model = GPTNeoXForCausalLM(cfg).to(torch.float32).eval()
    return model, cfg


def _build_tiny_tokenizer():
    """A dependency-free BYTE tokenizer paired 1:1 with the synthetic model's own
    vocab_size=256 above — deliberately NOT gpt-neox-20b's real GPT-2-style BPE
    tokenizer, which would need a network fetch or a local HF cache this box may
    not have. This smoke tests the EDITING pipeline (graft, key capture, rank-one
    write-back) which is tokenizer-agnostic — it only needs valid int token ids,
    never real subword semantics — so a byte tokenizer is architecturally
    sufficient, not a stand-in for a missing real one."""
    return _ByteTokenizer()


class _ByteTokenizer:
    """encode = ord() per char (mod vocab_size), decode = chr(). Exercises the exact
    call surface arch_compat.py / rome_native.py / metrics.py need: encode(text,
    add_special_tokens=...), __call__(text, return_tensors="pt").to(device), and the
    eos_token/eos_token_id/pad_token attributes."""
    eos_token = "<eos>"
    eos_token_id = 0
    pad_token = None

    def encode(self, text, add_special_tokens=True):
        ids = [1 + (ord(c) % (TINY_CFG["vocab_size"] - 1)) for c in text]
        return ([self.eos_token_id] + ids) if add_special_tokens else ids

    def __call__(self, text, return_tensors=None):
        from transformers import BatchEncoding
        ids = self.encode(text, add_special_tokens=True)
        return BatchEncoding({"input_ids": torch.tensor([ids]),
                              "attention_mask": torch.ones(1, len(ids), dtype=torch.long)})


def main():
    res = {"model": "synthetic-tiny-gptneox", "device": DEVICE, "tests": {}}
    ok_all = True

    def record(name, passed, detail):
        nonlocal ok_all
        res["tests"][name] = {"pass": bool(passed), **detail}
        ok_all = ok_all and bool(passed)
        print(f"[neox20b-smoke] {name}: {'PASS' if passed else 'FAIL'} {detail}", flush=True)

    # ---------------------------------------------------------------- T0: build + graft
    model, cfg = _build_tiny_gptneox()
    tok = _build_tiny_tokenizer()
    try:
        arch = normalize_arch(model, tok, DEVICE)
        record("T0_normalize_arch_returns_gptneox", arch == "gptneox", {"arch": arch})
    except SystemExit as e:
        record("T0_normalize_arch_returns_gptneox", False, {"err": str(e)})
        arch = None

    # ---------------------------------------------------------------- T1: graft identity
    # model.model.layers[i].mlp.down_proj must be the SAME nn.Linear object as
    # model.gpt_neox.layers[i].mlp.dense_4h_to_h — not a copy (data_ptr equality proves
    # the restore/write-back path in rome_native.py acts on the real weight, not a clone).
    if arch == "gptneox":
        same = all(
            model.model.layers[i].mlp.down_proj.weight.data_ptr()
            == model.gpt_neox.layers[i].mlp.dense_4h_to_h.weight.data_ptr()
            for i in range(cfg.num_hidden_layers)
        )
        is_linear = all(
            isinstance(model.model.layers[i].mlp.down_proj, torch.nn.Linear)
            for i in range(cfg.num_hidden_layers)
        )
        record("T1_graft_is_same_live_module", same and is_linear,
              {"same_data_ptr": same, "is_nn_linear": is_linear})
    else:
        record("T1_graft_is_same_live_module", False, {"skipped": "T0 failed"})

    # ---------------------------------------------------------------- T2: key capture through the graft
    prompt, subject = "The capital of Testland is the city of", "Testland"
    try:
        idx = find_subject_last_token_index(tok, prompt, subject)
        k = _capture_key(model, tok, 0, prompt, idx, DEVICE)
        record("T2_capture_key_through_graft",
              torch.is_tensor(k) and k.shape == (cfg.intermediate_size,) and torch.isfinite(k).all(),
              {"shape": list(k.shape), "finite": bool(torch.isfinite(k).all())})
    except Exception as e:
        record("T2_capture_key_through_graft", False, {"err": repr(e)})

    # ---------------------------------------------------------------- T3: ROME edit, no NaN
    snap = model.model.layers[0].mlp.down_proj.weight.detach().clone()
    try:
        req = {"prompt": prompt, "subject": subject, "target_new": "wonderful"}
        info = apply_edit(model, tok, req, {"layer": 0, "steps": 3, "lr": 0.1}, DEVICE)
        W_after = model.model.layers[0].mlp.down_proj.weight.detach()
        finite = bool(torch.isfinite(W_after).all())
        changed = bool((W_after - snap).abs().max() > 0)
        record("T3_rome_edit_no_nan", finite and changed,
              {"finite": finite, "changed": changed,
               "delta_weight_norm": info.get("delta_weight_norm"),
               "rank_one_solve_residual": info.get("rank_one_solve_residual")})
    except Exception as e:
        record("T3_rome_edit_no_nan", False, {"err": repr(e)})
    finally:
        with torch.no_grad():
            model.model.layers[0].mlp.down_proj.weight.copy_(snap)

    # ---------------------------------------------------------------- T4: device-resolution
    # logic, MOCKED heterogeneous devices (no real CUDA needed — torch.device("cuda:N")
    # is constructible without hardware; it is only ever read back here, never used to
    # allocate a real tensor). Proves resolve_layer_device reads the layer's OWN device,
    # not any ambient default passed elsewhere — the exact property that keeps
    # rome_native._optimise_value's `v` and alphaedit._resolve_projector's `P` on the
    # right card under real 2-GPU TP.
    import types
    mock_model = types.SimpleNamespace(model=types.SimpleNamespace(layers=[
        types.SimpleNamespace(mlp=types.SimpleNamespace(
            down_proj=types.SimpleNamespace(weight=types.SimpleNamespace(
                device=torch.device("cuda:0"))))),
        types.SimpleNamespace(mlp=types.SimpleNamespace(
            down_proj=types.SimpleNamespace(weight=types.SimpleNamespace(
                device=torch.device("cuda:1"))))),
    ]))
    d0 = resolve_layer_device(mock_model, 0)
    d1 = resolve_layer_device(mock_model, 1)
    record("T4_resolve_layer_device_mock_heterogeneous",
          d0 == torch.device("cuda:0") and d1 == torch.device("cuda:1") and d0 != d1,
          {"layer0_device": str(d0), "layer1_device": str(d1)})

    # ---------------------------------------------------------------- T5: safe_model_to guard
    class _SpyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.to_called_with = None
        def to(self, *a, **kw):  # noqa: A003 (shadowing builtin .to is the module's own API)
            self.to_called_with = (a, kw)
            return self

    m_dispatched = _SpyModule()
    m_dispatched.hf_device_map = {"gpt_neox.layers.0": 0, "gpt_neox.layers.1": 1}
    safe_model_to(m_dispatched, "cuda:0")
    dispatched_skipped = m_dispatched.to_called_with is None

    m_plain = _SpyModule()
    safe_model_to(m_plain, "cpu")
    plain_called = m_plain.to_called_with is not None

    record("T5_safe_model_to_guard", dispatched_skipped and plain_called,
          {"dispatched_model_to_skipped": dispatched_skipped,
           "plain_model_to_called": plain_called})

    res["ALL_PASS"] = ok_all
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tmp = OUT + ".tmp"
    json.dump(res, open(tmp, "w"), indent=2)
    os.replace(tmp, OUT)
    print(f"[neox20b-smoke] wrote {OUT} ALL_PASS={ok_all}", flush=True)
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
