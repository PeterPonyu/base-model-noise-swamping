"""ripple_geometry.py — does key geometry predict RippleEdits-style RELATED-fact damage
the same way it predicts unrelated collateral damage (the harness's core H1 question,
extended to a third probe class beyond CounterFact/zsRE/MQuAKE)?

For each edit in a RippleEdits sample (experiments/rippleedits_loader.py):
  1. its OWN ripple implications (the criterion test_queries RippleEdits ships for that
     edit) are checked for post-edit correctness — the benchmark's native per-criterion
     accuracy metric (Logical_Generalization / Compositionality_I/II / Subject_Aliasing /
     Relation_Specificity / Forgetfulness — see rippleedits_loader's module docstring for
     why "Forgetfulness" and not "Preservation").
  2. the SAME restore-every-edit / precompute-once-probe-baseline / per-edit damage-sweep
     design as killgate_keygeom.py is run TWICE — once against the pooled ripple-probe bank
     (all criteria' test_queries across the whole sample), once against an "unrelated"
     bank built the same way killgate builds its probes (other records' facts, unrelated
     to any edit's own ripple criteria) — producing two independent COS[edit,probe] /
     damage[edit,probe] matrices.
  3. within-probe Spearman(key-cosine, damage) (metrics.py convention: SIGNED Spearman,
     not AUROC — memory: editing-damage-metric-signed-spearman-not-auroc) is computed on
     BOTH matrices via analyze_matrices.within_probe_rhos, so the headline comparison is a
     single number pair: rho_ripple vs rho_unrelated.

DAMAGE TARGET ASYMMETRY (documented, not a bug): "damage" always means "drop in P/logit
of the token that SHOULD be correct for that probe right now". For unrelated probes that
is target_true (the pre-existing fact, killgate's convention — these should NOT change).
For ripple probes that is target_new (the NEW fact's logical implication, which SHOULD
hold after the edit — RippleEdits' own accuracy framing) — there is no independent "prior
truth" for a ripple implication distinct from what the edit is trying to establish.

Restricted to rome/alpha (single-layer, rank-one, restore-based editors) — same restore
mechanics as killgate; memit/ft/grace are out of scope here (not asked for, and memit is
fenced on gpt2/gptj by arch_compat's own graft limitation, ft has no rank-one key/value
factorization to reuse this cleanly, grace's deletion semantics don't fit a "damage" frame).

NO SUBJECT STRING (RippleEdits limitation, see rippleedits_loader.py): every key is
captured with subject=None, so find_subject_last_token_index falls back to "last token of
the prompt" — a known fidelity approximation, not this script's invention.

Usage:
  python experiments/ripple_geometry.py --model data/models/Llama-3.2-1B \\
      --data data/rippleedits/popular.json --editor rome --layer 12 \\
      --n_edits 50 --n_unrelated_probes 50 --steps 20 --lr 0.1 --seed 0 \\
      --out results/ripple_llama1b_rome_popular_L12_s0.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HARNESS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HARNESS)
from metrics import next_token_logits, first_target_token_id, efficacy  # noqa: E402
from editors.rome_native import _capture_key, find_subject_last_token_index  # noqa: E402
from editors.arch_compat import normalize_arch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rippleedits_loader import load_ripple_edits, CRITERIA  # noqa: E402
try:
    from experiments.analyze_matrices import within_probe_rhos  # noqa: E402
except ModuleNotFoundError:
    from analyze_matrices import within_probe_rhos  # noqa: E402


@torch.no_grad()
def prob_of_token(model, tok, prompt, token_id, device):
    logits = next_token_logits(model, tok, prompt, device)
    probs = torch.softmax(logits, dim=-1)
    return float(probs[token_id].item()), float(logits[token_id].item())


def cap_pool(probes_by_crit, cap, seed):
    """Seeded per-criterion cap so a huge Relation_Specificity list doesn't dominate
    runtime; returns (flat_list, per_criterion_kept_counts)."""
    rng = np.random.default_rng(seed + 12345)
    flat, kept = [], {}
    for crit, probes in probes_by_crit.items():
        if len(probes) > cap:
            idx = rng.choice(len(probes), size=cap, replace=False)
            probes = [probes[k] for k in sorted(idx)]
        kept[crit] = len(probes)
        flat.extend(probes)
    return flat, kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="data/rippleedits/{popular,random}.json "
                    "(recent.json is out of scope — see rippleedits_loader.py)")
    ap.add_argument("--criteria", default=",".join(CRITERIA))
    ap.add_argument("--n_edits", type=int, default=50)
    ap.add_argument("--n_unrelated_probes", type=int, default=50)
    ap.add_argument("--max_probes_per_criterion", type=int, default=40,
                    help="seeded per-criterion cap on the pooled ripple-probe bank")
    ap.add_argument("--editor", choices=["rome", "alpha"], default="rome")
    ap.add_argument("--layer", default="auto")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--keep_ratio", type=float, default=0.99,
                    help="alpha: fraction of preserved-key energy to project out "
                         "(projector fit on the unrelated-probe key bank)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--model_dtype", choices=["fp32", "bf16"], default="fp32")
    ap.add_argument("--save_matrices", action="store_true")
    ap.add_argument("--matrix_dir", default=os.path.join(HARNESS, "results", "matrices"))
    ap.add_argument("--out", default=os.path.join(HARNESS, "results", "ripple_geometry.json"))
    args = ap.parse_args()

    criteria = tuple(c for c in args.criteria.split(",") if c)
    t0 = time.time()

    edits, ripple_by_crit, unrelated, meta = load_ripple_edits(
        args.data, args.n_edits, args.n_unrelated_probes, args.seed, criteria=criteria)
    ripple_probes, kept_per_crit = cap_pool(ripple_by_crit, args.max_probes_per_criterion,
                                           args.seed)
    # damage needs a concrete "correct token" — unrelated probes with an unknown prior
    # value (target_true == "", see rippleedits_loader._diff_edit) have nothing to measure
    # damage against; drop them here rather than silently falling back to target_new
    # (which would blur the unrelated/ripple damage-target distinction this script exists
    # to keep separate).
    n_unrel_before = len(unrelated)
    unrelated = [u for u in unrelated if u.get("target_true")]
    print(f"[ripple] loaded: {len(edits)} edits, {len(ripple_probes)} ripple probes "
          f"(capped from {sum(len(v) for v in ripple_by_crit.values())}, per-criterion "
          f"{kept_per_crit}), {len(unrelated)}/{n_unrel_before} unrelated probes "
          f"(dropped {n_unrel_before - len(unrelated)} with unknown prior value)",
          flush=True)
    if len(edits) == 0 or len(ripple_probes) == 0 or len(unrelated) == 0:
        raise SystemExit(f"[ripple] insufficient data: edits={len(edits)} "
                         f"ripple_probes={len(ripple_probes)} unrelated={len(unrelated)}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    if args.editor == "rome":
        from editors.rome_native import apply_edit
    else:
        from editors.alphaedit import apply_edit, build_null_projector

    tok = AutoTokenizer.from_pretrained(args.model)
    load_dtype = torch.float32 if args.model_dtype == "fp32" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=load_dtype).to(args.device).eval()
    actual_dtype = str(next(model.parameters()).dtype).replace("torch.", "")
    arch = normalize_arch(model, tok, args.device)
    nL = model.config.num_hidden_layers
    layer = nL // 2 if args.layer == "auto" else int(args.layer)
    print(f"[ripple] loaded {args.model} ({nL} layers, edit layer={layer}, "
          f"device={args.device}, dtype={actual_dtype}, arch={arch}) {time.time()-t0:.1f}s",
          flush=True)

    def key_for(prompt):
        idx = find_subject_last_token_index(tok, prompt, None)  # no subject string; see docstring
        return _capture_key(model, tok, layer, prompt, idx, args.device).float().cpu().numpy()

    K_edit = np.stack([key_for(e["prompt"]) for e in edits])
    K_ripple = np.stack([key_for(p["prompt"]) for p in ripple_probes])
    K_unrel = np.stack([key_for(p["prompt"]) for p in unrelated])
    Ke = K_edit / (np.linalg.norm(K_edit, axis=1, keepdims=True) + 1e-8)
    COS_ripple = Ke @ (K_ripple / (np.linalg.norm(K_ripple, axis=1, keepdims=True) + 1e-8)).T
    COS_unrel = Ke @ (K_unrel / (np.linalg.norm(K_unrel, axis=1, keepdims=True) + 1e-8)).T
    print(f"[ripple] keys+cosine done {time.time()-t0:.1f}s", flush=True)

    alpha_proj = None
    if args.editor == "alpha":
        alpha_proj = build_null_projector(torch.tensor(K_unrel, device=args.device),
                                          args.keep_ratio)
        print(f"[ripple] alphaedit projector fit on {K_unrel.shape[0]} unrelated keys "
              f"(keep_ratio={args.keep_ratio})", flush=True)

    ripple_tok = [first_target_token_id(tok, p["target_new"]) for p in ripple_probes]
    unrel_tok = [first_target_token_id(tok, p["target_true"]) for p in unrelated]
    pre_p_r = np.zeros(len(ripple_probes)); pre_l_r = np.zeros(len(ripple_probes))
    for j, p in enumerate(ripple_probes):
        pre_p_r[j], pre_l_r[j] = prob_of_token(model, tok, p["prompt"], ripple_tok[j], args.device)
    pre_p_u = np.zeros(len(unrelated)); pre_l_u = np.zeros(len(unrelated))
    for j, p in enumerate(unrelated):
        pre_p_u[j], pre_l_u[j] = prob_of_token(model, tok, p["prompt"], unrel_tok[j], args.device)

    W = model.model.layers[layer].mlp.down_proj.weight
    W_base = W.detach().clone()

    N_e, M_r, M_u = len(edits), len(ripple_probes), len(unrelated)
    dmg_p_r = np.zeros((N_e, M_r)); dmg_l_r = np.zeros((N_e, M_r))
    dmg_p_u = np.zeros((N_e, M_u)); dmg_l_u = np.zeros((N_e, M_u))
    edit_ok = np.zeros(N_e)
    crit_hits, crit_total = {c: 0 for c in criteria}, {c: 0 for c in criteria}

    for i, e in enumerate(edits):
        cfg = {"layer": layer, "steps": args.steps, "lr": args.lr}
        if args.editor == "alpha":
            cfg["projector"] = alpha_proj
        apply_edit(model, tok, e, cfg, args.device)
        eff = efficacy(model, tok, e["prompt"], e["target_new"], e.get("target_true"), args.device)
        edit_ok[i] = eff["success"]

        for j, p in enumerate(ripple_probes):
            if p["source_edit_index"] != i:
                continue
            crit = p["criterion"]
            r = efficacy(model, tok, p["prompt"], p["target_new"], None, args.device)
            crit_total[crit] += 1
            crit_hits[crit] += int(r["success"])

        for j in range(M_r):
            pp, ll = prob_of_token(model, tok, ripple_probes[j]["prompt"], ripple_tok[j], args.device)
            dmg_p_r[i, j] = pre_p_r[j] - pp
            dmg_l_r[i, j] = pre_l_r[j] - ll
        for j in range(M_u):
            pp, ll = prob_of_token(model, tok, unrelated[j]["prompt"], unrel_tok[j], args.device)
            dmg_p_u[i, j] = pre_p_u[j] - pp
            dmg_l_u[i, j] = pre_l_u[j] - ll

        with torch.no_grad():
            W.copy_(W_base)
        if (i + 1) % max(1, N_e // 5) == 0 or i + 1 == N_e:
            print(f"[ripple] edit {i+1}/{N_e} done {time.time()-t0:.1f}s", flush=True)

    rho_ripple_logit = float(np.nanmean(within_probe_rhos(COS_ripple, dmg_l_r)))
    rho_unrel_logit = float(np.nanmean(within_probe_rhos(COS_unrel, dmg_l_u)))
    rho_ripple_p = float(np.nanmean(within_probe_rhos(COS_ripple, dmg_p_r)))
    rho_unrel_p = float(np.nanmean(within_probe_rhos(COS_unrel, dmg_p_u)))
    per_criterion_accuracy = {
        c: (crit_hits[c] / crit_total[c] if crit_total[c] else None) for c in criteria
    }

    out = {
        "model": args.model, "data": args.data, "editor": args.editor, "layer": layer,
        "n_layers": nL, "arch": arch, "seed": args.seed, "steps": args.steps, "lr": args.lr,
        "n_edits": N_e, "n_ripple_probes": M_r, "n_unrelated_probes": M_u,
        "edit_success_rate": float(np.mean(edit_ok)),
        "per_criterion_accuracy": per_criterion_accuracy,
        "per_criterion_n": crit_total,
        "within_probe_rho_logit": {"ripple": rho_ripple_logit, "unrelated": rho_unrel_logit},
        "within_probe_rho_p": {"ripple": rho_ripple_p, "unrelated": rho_unrel_p},
        "loader_meta": meta,
        "runtime_s": time.time() - t0,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"[ripple] rho(key-cos, damage) logit: ripple={rho_ripple_logit:+.4f} "
          f"unrelated={rho_unrel_logit:+.4f}  |  p: ripple={rho_ripple_p:+.4f} "
          f"unrelated={rho_unrel_p:+.4f}", flush=True)
    print(f"[ripple] per-criterion accuracy: {per_criterion_accuracy}", flush=True)

    if args.save_matrices:
        os.makedirs(args.matrix_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(args.out))[0]
        np.savez(os.path.join(args.matrix_dir, base + ".npz"),
                 COS_ripple=COS_ripple, damage_p_ripple=dmg_p_r, damage_logit_ripple=dmg_l_r,
                 COS_unrelated=COS_unrel, damage_p_unrelated=dmg_p_u, damage_logit_unrelated=dmg_l_u,
                 edit_ok=edit_ok)
        print(f"[ripple] matrices -> {os.path.join(args.matrix_dir, base + '.npz')}", flush=True)


if __name__ == "__main__":
    main()
