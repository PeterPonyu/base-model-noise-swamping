"""memit.py — native MEMIT-style editor (multi-layer whitened spread, batch-capable).

Same ``apply_edit(model, tok, edit_request, config, device) -> info`` contract as
``rome_native.py`` / ``alphaedit.py``. Implements the whitened update from the
rome_native.py docstring formula (its TODO block):

    ΔW_l = r_l (C_l^{-1} k_l)^T / (k_l^T C_l^{-1} k_l)

spread across a span of layers ending at the z-layer, with keys RECAPTURED under
the partially-edited model (MEMIT's own recompute-per-layer convention).
``rome_native.py`` itself is NOT modified — its identity-covariance ROME stays
the pristine baseline.

Covariance note (0-download constraint): C_l is fit on ALL token positions of an
IN-HARNESS prompt bank (holdout prompts for --memit_cov_source generic, probe
prompts for probes), NOT MEMIT's ~100k-token external Wikipedia mom2. The paper
implementation appendix must state this; prereg gate M2 (identity-cov ablation)
polices whether the word "MEMIT" is even earned. If the identity ablation matches
generic within |Δrho| < 0.05 at all layers, the text must say
"MEMIT-style multi-layer spread", not "MEMIT". If V3 (identity ablation) never
runs, the DEFAULT is the demoted name (pre-registered).

NG-partialling prereg (design rev, frozen before any GPU result): for memit
cells the norm-growth confound baseline uses delta_norm_total (sum over edited
layers, stored as a NEW npz array by the killgate) as PRIMARY and the z-layer
delta norm (the legacy norm_growth array) as SECONDARY. The CPU consumer is
experiments/analyze_memit_ngtotal.py. Prereg branch precedence when evaluating
the MEMIT outcome: (c) KILLED -> (a) GENERALIZES -> (b) ATTENUATED, with the
seed-level perm-p aggregated as max over the 3 seeds (analyze_matrices
aggregate convention).

VRAM note: C_l is [d_int, d_int] (256MB fp32 at d=8192). C/chol stay CPU-resident
by design with per-edit CPU cholesky_solve (O(d^2)) — zero VRAM cost. Do NOT
move chol to GPU when an 8B model arrives (~800MB/layer at 14336).

fp32 is load-bearing throughout the value-opt path (fp16 silently NaNs).
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


def parse_memit_layers(spec: str, z_layer: int, n_layers: int, span: int = 4) -> List[int]:
    """'auto' -> sorted [max(0, z_layer-span+1) .. z_layer]; else comma ints.

    Pure function. Sorts, dedupes, bounds-checks 0 <= l < n_layers; raises
    ValueError unless max == z_layer (COS geometry refers to the deepest edited
    layer = z-layer — documented decision). Imported by the killgate so the
    runner makes zero layer decisions.
    """
    if spec == "auto":
        layers = list(range(max(0, z_layer - span + 1), z_layer + 1))
    else:
        try:
            layers = sorted({int(x) for x in spec.split(",") if x.strip() != ""})
        except ValueError as e:
            raise ValueError(f"--memit_layers must be 'auto' or comma ints, got {spec!r}") from e
    if not layers:
        raise ValueError("--memit_layers parsed to an empty list")
    for l in layers:
        if not (0 <= l < n_layers):
            raise ValueError(f"memit layer {l} out of range [0, {n_layers})")
    if max(layers) != z_layer:
        raise ValueError(f"max(memit_layers)={max(layers)} must equal the z-layer (--layer={z_layer}) "
                         f"so the precomputed COS geometry and the z-layer coincide")
    return sorted(layers)


@torch.no_grad()
def estimate_layer_covariances(model, tok, prompts, layers, device,
                               max_tokens: int = 20000, reg: float = 1e-2) -> Dict[int, Dict]:
    """ONE hooked forward per prompt capturing the down_proj INPUT at ALL positions
    for all requested layers simultaneously; accumulate C_l += X_l^T X_l in fp32 on
    CPU (never store the full K bank). Stop at max_tokens. A = C_hat + reg*mean(diag)*I;
    chol with up to 3 retries at 10x reg on failure, else raise.

    Deterministic given the prompt list (no sampling -> seed-free).
    Returns {l: {'chol': L (CPU fp32), 'n_tokens': int, 'reg_used': float}}.
    """
    layers = sorted(set(int(l) for l in layers))
    d_int = model.model.layers[layers[0]].mlp.down_proj.weight.shape[1]
    C = {l: torch.zeros(d_int, d_int, dtype=torch.float32) for l in layers}
    n_tok = {l: 0 for l in layers}
    cap: Dict[int, torch.Tensor] = {}

    hooks = []

    def mk_hook(l):
        def hook(_m, inputs, _o):
            cap[l] = inputs[0][0].detach()  # [seq, d_int]
        return hook

    for l in layers:
        hooks.append(model.model.layers[l].mlp.down_proj.register_forward_hook(mk_hook(l)))
    try:
        for prompt in prompts:
            if min(n_tok.values()) >= max_tokens:
                break
            enc = tok(prompt, return_tensors="pt").to(device)
            model(**enc)
            for l in layers:
                X = cap[l].float().cpu()  # [seq, d_int] fp32 CPU
                take = min(X.shape[0], max(0, max_tokens - n_tok[l]))
                if take > 0:
                    Xs = X[:take]
                    C[l] += Xs.t() @ Xs
                    n_tok[l] += take
    finally:
        for h in hooks:
            h.remove()

    out: Dict[int, Dict] = {}
    for l in layers:
        if n_tok[l] == 0:
            raise RuntimeError(f"memit covariance: 0 tokens accumulated at layer {l}")
        C_hat = C[l] / float(n_tok[l])
        mean_diag = float(C_hat.diagonal().mean().item())
        reg_used = reg
        chol = None
        for _try in range(4):  # initial + up to 3 retries at 10x
            A = C_hat + reg_used * mean_diag * torch.eye(d_int, dtype=torch.float32)
            try:
                chol = torch.linalg.cholesky(A)
                break
            except Exception:
                reg_used *= 10.0
        if chol is None:
            raise RuntimeError(f"memit covariance: cholesky failed at layer {l} even at reg={reg_used}")
        out[l] = {"chol": chol, "n_tokens": int(n_tok[l]), "reg_used": float(reg_used)}
    return out


def _solve_cov(cov_entry: Optional[Dict], k: torch.Tensor) -> torch.Tensor:
    """C^{-1} k via CPU cholesky_solve (O(d^2) per edit); k unchanged when cov None."""
    if cov_entry is None:
        return k
    k_cpu = k.detach().float().cpu()
    sol = torch.cholesky_solve(k_cpu[:, None], cov_entry["chol"])[:, 0]
    return sol.to(k.device)


def _hidden_at(model, tok, prompt, z_layer, tok_index, device, capture_layer=None):
    """One forward; returns (h^{z_layer+1}[tok_index] fp32, captured key at capture_layer or None).

    h^{z_layer+1} = RAW output of decoder layer z_layer (pre-norm residual stream),
    captured via a forward hook on the decoder-layer module itself. NOTE: this is
    deliberately NOT output_hidden_states — in current transformers the LAST entry
    of hidden_states has the final RMSNorm already applied, which silently breaks
    the additivity invariant when z_layer is the last layer (caught by smoke T3).
    A delta added to ANY earlier layer's down_proj output at this token shifts
    h^{z_layer+1} by +delta to first order — the same approximation MEMIT makes.
    """
    cap = {}
    hooks = []
    layer_mod = model.model.layers[z_layer]

    def h_hook(_m, _inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        cap["h"] = hs[0, tok_index, :].detach().float()
    hooks.append(layer_mod.register_forward_hook(h_hook))

    if capture_layer is not None:
        down = model.model.layers[capture_layer].mlp.down_proj

        def k_hook(_m, inputs, _o):
            cap["k"] = inputs[0][0, tok_index, :].detach().float()
        hooks.append(down.register_forward_hook(k_hook))
    try:
        with torch.no_grad():
            enc = tok(prompt, return_tensors="pt").to(device)
            model(**enc)
    finally:
        for h in hooks:
            h.remove()
    return cap["h"], cap.get("k")


def apply_edit(model, tok, edit_request: Dict, config: Dict, device: str = "cpu") -> Dict:
    """Single-fact multi-layer MEMIT-style edit, exact harness contract.

    config: layers (sorted list, max == z_layer), z_layer, steps, lr, cov
    (dict from estimate_layer_covariances or None = identity), cov_source (str).
    NO restore inside — the runner snapshots/restores (ft_editor convention).
    """
    prompt = edit_request["prompt"]
    target_new = edit_request["target_new"]
    subject = edit_request.get("subject")

    layers = sorted(int(l) for l in config["layers"])
    z_layer = int(config.get("z_layer", max(layers)))
    assert max(layers) == z_layer, "memit: max(layers) must equal z_layer"
    steps = int(config.get("steps", 25))
    lr = float(config.get("lr", 5e-1))
    v_wd = float(config.get("v_weight_decay", 1e-3))
    cov = config.get("cov")  # {l: {'chol','n_tokens','reg_used'}} or None
    cov_source = str(config.get("cov_source", "identity" if cov is None else "generic"))

    model.to(device)
    model.eval()

    tok_index = find_subject_last_token_index(tok, prompt, subject)

    # (2) optimise the z-layer value (fp32 in/out — the .float() casts are load-bearing)
    v, v0, history = _optimise_value(model, tok, z_layer, prompt, tok_index,
                                     target_new, device, steps, lr, v_wd)
    dz = (v.float() - v0.float())
    residual_norm = float(dz.norm().item())  # the S factor consumed by the harness

    # (4) pre-pass: base residual-stream state at the z-layer output
    h_base, _ = _hidden_at(model, tok, prompt, z_layer, tok_index, device)
    z_target = h_base + dz

    delta_norms: Dict[int, float] = {}
    solve_resids: List[float] = []
    n = len(layers)
    for i, l in enumerate(layers):
        # (5) recapture key + current hidden UNDER THE PARTIALLY-EDITED MODEL
        h_cur, k_l = _hidden_at(model, tok, prompt, z_layer, tok_index, device, capture_layer=l)
        shortfall = z_target - h_cur
        r_l = shortfall / float(n - i)
        Ck = _solve_cov(cov.get(l) if cov is not None else None, k_l)
        denom = float((k_l @ Ck).item()) + 1e-8
        delta = torch.outer(r_l, Ck) / denom  # the rome_native docstring whitened formula
        W_l = model.model.layers[l].mlp.down_proj.weight
        with torch.no_grad():
            W_l.add_(delta.to(W_l.dtype))
        delta_norms[l] = float(delta.norm().item())
        exact = float((delta.float() @ k_l - r_l).norm().item()) / (float(r_l.norm().item()) + 1e-12)
        solve_resids.append(exact)

    # (6) verification forward
    h_final, _ = _hidden_at(model, tok, prompt, z_layer, tok_index, device)
    shortfall_final = float((z_target - h_final).norm().item())

    return {
        "editor": "memit_native",
        "layers": layers,
        "z_layer": z_layer,
        "delta_weight_norm": delta_norms,               # dict {int l: float} -> ng[z_layer] in the runner
        "delta_weight_norm_total": float(sum(delta_norms.values())),
        "residual_norm": residual_norm,                 # ||dz|| = ||v - Wk|| at the z-layer
        "shortfall_final": shortfall_final,
        "shortfall_ratio": shortfall_final / (residual_norm + 1e-8),
        "rank_one_solve_residual": max(solve_resids) if solve_resids else float("nan"),
        "final_value_loss": history[-1] if history else None,
        "covariance_used": cov is not None,
        "cov_source": cov_source,
        "cov_n_tokens": ({l: cov[l]["n_tokens"] for l in layers} if cov is not None else None),
    }


def apply_batch_edit(model, tok, edit_requests: List[Dict], config: Dict, device: str = "cpu") -> Dict:
    """Batch MEMIT proper — EXPERIMENTAL; not called by the killgate per-edit loop (v1).

    Ready for the sequential/batch arm. Per layer l ascending:
        A = lambda * C_hat_l + K_l^T K_l  (fp32 CPU; lambda = config['memit_lambda'],
        default 10000 = MEMIT's mom2_update_weight — a config-dict key ONLY, never a
        killgate CLI flag), adjK = solve(A, K_l^T), DeltaW_l = R_l^T @ adjK^T.
    """
    layers = sorted(int(l) for l in config["layers"])
    z_layer = int(config.get("z_layer", max(layers)))
    steps = int(config.get("steps", 25))
    lr = float(config.get("lr", 5e-1))
    v_wd = float(config.get("v_weight_decay", 1e-3))
    cov = config.get("cov")
    lam = float(config.get("memit_lambda", 10000.0))

    model.to(device)
    model.eval()

    # per-fact z targets
    facts = []
    for e in edit_requests:
        idx = find_subject_last_token_index(tok, e["prompt"], e.get("subject"))
        v, v0, _ = _optimise_value(model, tok, z_layer, e["prompt"], idx,
                                   e["target_new"], device, steps, lr, v_wd)
        dz = (v.float() - v0.float())
        h_base, _ = _hidden_at(model, tok, e["prompt"], z_layer, idx, device)
        facts.append({"e": e, "idx": idx, "z_target": h_base + dz, "dz_norm": float(dz.norm().item())})

    d_int = model.model.layers[layers[0]].mlp.down_proj.weight.shape[1]
    delta_norms: Dict[int, float] = {}
    n = len(layers)
    for i, l in enumerate(layers):
        K_rows, R_rows = [], []
        for f in facts:
            h_cur, k_l = _hidden_at(model, tok, f["e"]["prompt"], z_layer, f["idx"], device,
                                    capture_layer=l)
            K_rows.append(k_l.detach().float().cpu())
            R_rows.append(((f["z_target"] - h_cur) / float(n - i)).detach().float().cpu())
        K = torch.stack(K_rows)          # [B, d_int]
        R = torch.stack(R_rows)          # [B, h]
        if cov is not None and l in cov:
            Lc = cov[l]["chol"]
            C_hat = Lc @ Lc.t()          # includes its ridge — acceptable for the experimental path
        else:
            C_hat = torch.eye(d_int, dtype=torch.float32)
        A = lam * C_hat + K.t() @ K
        adjK = torch.linalg.solve(A, K.t())          # [d_int, B]
        deltaW = (R.t() @ adjK.t())                  # [h, d_int]
        W_l = model.model.layers[l].mlp.down_proj.weight
        with torch.no_grad():
            W_l.add_(deltaW.to(W_l.dtype).to(W_l.device))
        delta_norms[l] = float(deltaW.norm().item())

    ratios = []
    for f in facts:
        h_final, _ = _hidden_at(model, tok, f["e"]["prompt"], z_layer, f["idx"], device)
        ratios.append(float((f["z_target"] - h_final).norm().item()) / (f["dz_norm"] + 1e-8))
    return {
        "editor": "memit_native_batch",
        "layers": layers,
        "z_layer": z_layer,
        "delta_weight_norm": delta_norms,
        "delta_weight_norm_total": float(sum(delta_norms.values())),
        "shortfall_ratios": ratios,
        "memit_lambda": lam,
    }
