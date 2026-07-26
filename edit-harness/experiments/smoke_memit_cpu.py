"""smoke_memit_cpu.py — standalone CPU smoke of editors/memit.py (tests T1-T6).

Runs on HuggingFaceM4/tiny-random-LlamaForCausalLM, fp32, CPU, no killgate import.
Writes results/smoke_infra/memit_smoke.json atomically (tmp + os.replace).
Smoke gates are STRUCTURAL asserts, never efficacy (esr=0 at steps=2 is expected).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

HARNESS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HARNESS)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from editors import rome_native  # noqa: E402
from editors.memit import (  # noqa: E402
    apply_edit, apply_batch_edit, estimate_layer_covariances, parse_memit_layers,
)
from metrics import next_token_logits  # noqa: E402

TINY = "HuggingFaceM4/tiny-random-LlamaForCausalLM"
DEVICE = "cpu"
OUT = os.path.join(HARNESS, "results", "smoke_infra", "memit_smoke.json")


def _snap(model, layers):
    return {l: model.model.layers[l].mlp.down_proj.weight.detach().clone() for l in layers}


def _restore(model, snap):
    with torch.no_grad():
        for l, w in snap.items():
            model.model.layers[l].mlp.down_proj.weight.copy_(w)


def main():
    res = {"model": TINY, "device": DEVICE, "tests": {}}
    ok_all = True

    def record(name, passed, detail):
        nonlocal ok_all
        res["tests"][name] = {"pass": bool(passed), **detail}
        ok_all = ok_all and bool(passed)
        print(f"[memit-smoke] {name}: {'PASS' if passed else 'FAIL'} {detail}", flush=True)

    # pure-function asserts on parse_memit_layers
    try:
        assert parse_memit_layers("auto", 12, 16) == [9, 10, 11, 12]
        assert parse_memit_layers("auto", 1, 2) == [0, 1]          # clamp at 0
        assert parse_memit_layers("9,12,10,11", 12, 16) == [9, 10, 11, 12]
        bad = False
        try:
            parse_memit_layers("9,10", 12, 16)                    # max != z_layer
        except ValueError:
            bad = True
        assert bad
        record("T0_parse_memit_layers", True, {})
    except AssertionError as e:
        record("T0_parse_memit_layers", False, {"err": str(e)})

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TINY)
    model = AutoModelForCausalLM.from_pretrained(TINY, dtype=torch.float32).to(DEVICE).eval()
    nL = model.config.num_hidden_layers
    z = nL - 1
    # NOTE: the tiny model has 2 layers, so z=1 is the LAST layer. A last-layer MLP
    # edit at a mid-prompt subject token has ZERO gradient path to the final-position
    # logits (nothing downstream reads it), making dz degenerate-zero. The smoke
    # therefore uses SUBJECT-FINAL prompts so the z-layer value-opt is gradient-live.
    # Real cells (e.g. L12 of 16) never hit this; documented tiny-model ceiling.
    req = {"prompt": "Paris is the capital city of France", "subject": "France",
           "target_new": "Rome"}
    cfg_common = {"steps": 2, "lr": 0.1}

    # T1 identity-equivalence: memit(layers=[z], cov=None) delta == rome delta
    snap = _snap(model, [z])
    info_m = apply_edit(model, tok, req, {"layers": [z], "z_layer": z, "cov": None,
                                          "cov_source": "identity", **cfg_common}, DEVICE)
    D_memit = (model.model.layers[z].mlp.down_proj.weight.detach() - snap[z]).float().clone()
    _restore(model, snap)
    rome_native.apply_edit(model, tok, req, {"layer": z, **cfg_common}, DEVICE)
    D_rome = (model.model.layers[z].mlp.down_proj.weight.detach() - snap[z]).float().clone()
    _restore(model, snap)
    rel = float((D_memit - D_rome).norm() / (D_rome.norm() + 1e-30))
    record("T1_identity_equivalence", rel < 1e-5, {"rel_err": rel})

    # T2 rank-one exactness (from the run above)
    record("T2_rank_one_exactness", info_m["rank_one_solve_residual"] < 1e-4,
           {"max_solve_resid": info_m["rank_one_solve_residual"]})

    # T3 spread closure: layers=[z-1, z] -> shortfall_ratio < 0.05
    layers2 = [z - 1, z]
    snap2 = _snap(model, layers2)
    info_s = apply_edit(model, tok, req, {"layers": layers2, "z_layer": z, "cov": None,
                                          "cov_source": "identity", **cfg_common}, DEVICE)
    _restore(model, snap2)
    record("T3_spread_closure", info_s["shortfall_ratio"] < 0.05,
           {"shortfall_ratio": info_s["shortfall_ratio"],
            "delta_norms": info_s["delta_weight_norm"]})

    # T4 covariance path: fit on 20 counterfact prompts, whitened edit finite
    cf = json.load(open(os.path.join(HARNESS, "data", "counterfact.json")))
    prompts = []
    for d in cf[:40]:
        rr = d.get("requested_rewrite", d)
        try:
            prompts.append(rr["prompt"].format(rr["subject"]) if "{}" in rr["prompt"] else rr["prompt"])
        except Exception:
            continue
        if len(prompts) >= 20:
            break
    cov = estimate_layer_covariances(model, tok, prompts, layers2, DEVICE,
                                     max_tokens=2000, reg=1e-2)
    cov_ok = all(torch.isfinite(cov[l]["chol"]).all() and cov[l]["n_tokens"] > 0
                 and cov[l]["reg_used"] > 0 for l in layers2)
    snap3 = _snap(model, layers2)
    info_w = apply_edit(model, tok, req, {"layers": layers2, "z_layer": z, "cov": cov,
                                          "cov_source": "generic", **cfg_common}, DEVICE)
    D_ws = {l: (model.model.layers[l].mlp.down_proj.weight.detach() - snap3[l]) for l in layers2}
    finite = all(torch.isfinite(D).all() for D in D_ws.values())
    _restore(model, snap3)
    record("T4_covariance_path", cov_ok and finite and info_w["covariance_used"],
           {"n_tokens": {l: cov[l]["n_tokens"] for l in layers2},
            "reg_used": {l: cov[l]["reg_used"] for l in layers2},
            "deltas_finite": finite, "shortfall_ratio": info_w["shortfall_ratio"]})

    # T5 batch (EXPERIMENTAL path): B=3, small lambda for the tiny-model scale
    # (default lambda=10000 is MEMIT's mom2_update_weight for real-scale covariances;
    #  at tiny-random scale with identity C it would damp the update to nothing, so
    #  the smoke uses a scale-appropriate lambda — structural check, not science).
    reqs = [req,
            {"prompt": "I love visiting the Eiffel Tower", "subject": "Eiffel Tower",
             "target_new": "Berlin"},
            {"prompt": "The chemical formula of water", "subject": "water",
             "target_new": "iron"}]
    snap4 = _snap(model, layers2)
    info_b = apply_batch_edit(model, tok, reqs, {"layers": layers2, "z_layer": z, "cov": None,
                                                 "memit_lambda": 1e-8, **cfg_common}, DEVICE)
    D_bs = {l: (model.model.layers[l].mlp.down_proj.weight.detach() - snap4[l]) for l in layers2}
    finite_b = all(torch.isfinite(D).all() for D in D_bs.values())
    _restore(model, snap4)
    record("T5_batch", finite_b and all(r < 0.25 for r in info_b["shortfall_ratios"]),
           {"shortfall_ratios": info_b["shortfall_ratios"], "finite": finite_b,
            "memit_lambda_used": 1e-8})

    # T6 restore-dict rehearsal (rehearses the killgate restore refactor)
    held = "The largest planet is"
    logits_pre = next_token_logits(model, tok, held, DEVICE)
    snap5 = _snap(model, layers2)
    apply_edit(model, tok, req, {"layers": layers2, "z_layer": z, "cov": None,
                                 "cov_source": "identity", **cfg_common}, DEVICE)
    _restore(model, snap5)
    maxdiff = max(float((model.model.layers[l].mlp.down_proj.weight.detach() - snap5[l])
                        .abs().max()) for l in layers2)
    logits_post = next_token_logits(model, tok, held, DEVICE)
    logits_same = bool(torch.equal(logits_pre, logits_post))
    record("T6_restore_dict", maxdiff == 0.0 and logits_same,
           {"max_abs_weight_diff": maxdiff, "logits_identical": logits_same})

    res["ALL_PASS"] = ok_all
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tmp = OUT + ".tmp"
    json.dump(res, open(tmp, "w"), indent=2)
    os.replace(tmp, OUT)
    print(f"[memit-smoke] wrote {OUT} ALL_PASS={ok_all}", flush=True)
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
