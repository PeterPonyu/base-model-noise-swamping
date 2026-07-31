"""prospective_admission.py — R-E prospective admission-policy evaluation.

GATED: the GPU path runs only when --prereg points at a prereg markdown containing a line
reading exactly "STATUS: RATIFIED" — a line the USER writes. docs/plans/PREREG-D2-PROSPECTIVE-
2026-07-26.md is the file to ratify (it supersedes the 07-16 DRAFT and resolves its three open
decision points). The driver run_d2_prospective.sh passes --prereg but cannot ratify: it fails
closed on the same guard. --selftest (CPU, synthetic, no model) is safe to run any time.

WHY THIS EXISTS. Every existing D2 federation result (merging_m0.py, merging_editors.py,
d3_benefit_predictor.py, rg_admission_benefit.py) is RETROSPECTIVE: edits are measured, merged,
and only THEN is the geometry statistic read off and correlated with the observed damage. This
module asks the PROSPECTIVE question instead: build an admission policy that uses geometry to
ADMIT/REJECT edits BEFORE any merging happens, form groups from the admitted edits, install the
merges on the REAL model, and measure real behavioral outcomes — not a closed-form estimate.

DESIGN (frozen in the prereg doc; read it before changing anything here):
  - reference cell: Llama-3.2-1B L12, ROME (identity-cov), CounterFact.
  - candidate pool: N=100 edits/seed; screening score is the Eq-1 closed form generalised to the
    WHOLE POOL as the cross-talk universe (admission precedes grouping, so the score cannot
    depend on a group assignment that doesn't exist yet):
        I_cos(a) = ||k_a|| * sum_{b!=a in pool} S_b * |cos(k_b,k_a)|
        I_mag(a) = ||k_a|| * sum_{b!=a in pool} S_b                      (cosine=1 bound)
    identical definition to merging_m0._regime_stat's I_cos/I_mag, summed over the pool instead
    of one measured merge group.
  - budget 25%; three policies: geometry (bottom-25% by I_cos), magnitude (bottom-25% by I_mag),
    random (3 independent uniform draws/seed). "Bottom" = admit the LOWEST-score (safest,
    least-predicted-interference) edits — same convention as rg_admission_benefit.admit_stats.
  - admitted edits partitioned into g=5 random groups of 5 (disjoint tiling, like
    merging_m0._tiled_groups), each group's ΔW installed/measured/restored exactly like
    merging_m0._merge_factors / _measure_merged_groups (no new merge math).
  - behavioral outcomes measured on the DEPLOYED merged model: (a) edit success rate
    (argmax_ok_post), (b) neighborhood/specificity damage via egl_metrics's canonical NS
    machinery (CounterFact neighborhood_prompts, scored with the edit's own target_new/
    target_true pair), (c) general retention on a FIXED 200-prompt held-out set (mean
    full-target log-prob shift, base vs merged), (d) target-logit drop (logit_solo - logit_post),
    the same field every retrospective RG/M0 table already reports.
  - 3 seeds (0,1,2), each an independent 100-candidate pool draw.

NEIGHBORHOOD-DAMAGE REFERENCE (--ns_reference, decision point now IMPLEMENTED, not just
designed): outcome (b)'s "pre-merge" reference for neighbor NS was a design ambiguity in the
prereg's original text (it said the reference is the "solo-edit baseline", but the shipped code
actually measured it at the unedited BASE model, since merging_m0._compute_solo restores every
edit after its own solo pass -- the weights at that point are indistinguishable from base).
Both readings are now real, flag-selected options, so ratifying is choosing a value, not writing
code:
  - --ns_reference base  (option ii, the prior de facto behavior): neighbor NS measured at the
    UNEDITED base model. A neighbor that edit `a`'s OWN solo edit already damages is never
    counted as merge-induced (it reads as "already damaged" before federation even happens) --
    INCLUDES solo-edit collateral in what counts as pre-existing damage.
  - --ns_reference solo  (option i, true federation-added damage): neighbor NS measured with
    edit `a` installed ALONE (its own solo ΔW, reconstructed via merging_m0._merge_factors with
    a singleton group -- bit-identical to the editor's own per-edit ΔW, no new editor math; see
    _solo_delta_w below), then restored. Isolates the damage the FEDERATION adds on top of what
    the solo edit already did -- literally what the prereg's original prose described.
No default: the GPU path REFUSES until --ns_reference is passed explicitly (mirrors, and does
NOT weaken, the prereg-ratification SystemExit below).

REUSE (imported, not reimplemented): experiments/merging_m0.py's load_counterfact,
_load_edit_model, _compute_solo, _merge_factors, _spearman, _savez_atomic, _write_table,
_model_tag; experiments/egl_metrics.py's attach_egl_fields, mean_logprob_full_target,
full_target_scores. New here: the pool-wide (not group-restricted) I_cos/I_mag screening
function, the three admission policies, the random group partition, and the behavioral-
measurement loop.

Standing workspace rules honored: ROME value-opt stays fp32 (merging_m0._compute_solo /
editors/rome_native.py's own .float() casts — never overridden here); the reported geometry
statistic is signed Spearman, never AUROC (used only in --selftest's sanity checks, this module
does not itself compute a Spearman verdict — it reports raw behavioral aggregates per the
prereg); PID-only process control belongs to the (not-yet-written) driver, not this module.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time

import numpy as np

HARNESS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HARNESS not in sys.path:
    sys.path.insert(0, HARNESS)

from experiments.merging_m0 import (  # noqa: E402
    load_counterfact, _load_edit_model, _compute_solo, _merge_factors, _spearman,
    _savez_atomic, _write_table, _model_tag,
)

SCHEMA_VERSION = "prospadm.v1"
POLICIES_SCORED = ("geometry", "magnitude")   # random handled separately (n_random_draws draws)
RETENTION_SEED = 999                          # fixed, --seed-independent (prereg "General retention")
# Fixed per-policy salt for the group-partition RNG (run_admission_seed). NOT hash(pname): Python
# randomizes str hashes per-process by default (PYTHONHASHSEED), which would make the group
# partition non-reproducible across re-runs at the SAME --seed — silently breaking this harness's
# standing seed-determinism convention. A fixed dict is stable across processes/Python versions.
POLICY_SALT = {"geometry": 11, "magnitude": 23, "random": 37}


# ============================================================ pool-wide screening (Eq-1, numpy)
def pool_screening_stats(K, S, key_norm):
    """I_cos(a)/I_mag(a) over the FULL candidate pool (every other candidate is a potential
    federation partner — admission precedes grouping, so the score must not assume a group).
    K: [N,d_in] raw keys, S: [N] = ||r||/||k||, key_norm: [N] = ||k||. Vectorised, O(N^2) (N=100
    is trivial). Reduces exactly to merging_m0._regime_stat's per-observation I_cos/I_mag formula
    when "others" == the whole pool minus self."""
    K = np.asarray(K, float); S = np.asarray(S, float); key_norm = np.asarray(key_norm, float)
    N = K.shape[0]
    Kn = K / (key_norm[:, None] + 1e-12)
    absc = np.abs(Kn @ Kn.T)
    np.fill_diagonal(absc, 0.0)                 # excludes b==a from every sum below
    I_cos = key_norm * (absc @ S)
    I_mag = key_norm * (float(S.sum()) - S)
    return I_cos, I_mag


def _assert_pool_screening_bruteforce(rng, trials=20, N=17, d_in=9, tol=1e-9):
    """--selftest (a): pool_screening_stats vs an explicit O(N^2) python double loop."""
    worst = 0.0
    for _ in range(trials):
        K = rng.standard_normal((N, d_in))
        key_norm = np.linalg.norm(K, axis=1)
        S = rng.uniform(0.1, 2.0, N)
        I_cos, I_mag = pool_screening_stats(K, S, key_norm)
        for a in range(N):
            ic = im = 0.0
            for b in range(N):
                if b == a:
                    continue
                cos_ab = abs(float(K[b] @ K[a]) / (key_norm[b] * key_norm[a] + 1e-12))
                ic += S[b] * cos_ab
                im += S[b]
            ic *= key_norm[a]; im *= key_norm[a]
            worst = max(worst, abs(ic - I_cos[a]), abs(im - I_mag[a]))
        assert np.allclose(I_cos, [sum(S[b] * abs(float(K[b] @ K[a]) /
                                        (key_norm[b] * key_norm[a] + 1e-12))
                                        for b in range(N) if b != a) * key_norm[a]
                                   for a in range(N)], atol=1e-9)
    return worst


# ============================================================ admission policies
def admit_bottom_q(score, budget):
    """Admit the LOWEST-`budget`-fraction by `score` (least predicted interference) — same
    "bottom-q" convention as experiments/rg_admission_benefit.py's admit_stats. Stable sort so
    ties resolve by original (pool) index, making the selection deterministic given `score`."""
    score = np.asarray(score, float)
    n = score.shape[0]
    k = max(1, int(np.floor(budget * n)))
    idx = np.argsort(score, kind="stable")[:k]
    return np.sort(idx)


def admit_random(n, budget, rng):
    """Uniform random admission of floor(budget*n) of the n pool indices, via the CALLER-owned
    `rng` (so each of the n_random_draws draws gets its own independent stream)."""
    k = max(1, int(np.floor(budget * n)))
    idx = rng.permutation(n)[:k]
    return np.sort(idx)


def partition_groups(admitted_idx, group_size, rng):
    """Random disjoint partition of the admitted (pool-index) array into floor(k/group_size)
    groups of exactly group_size (drops the remainder) — mirrors merging_m0._tiled_groups'
    convention, applied to the ADMITTED subset rather than the whole pool. Returns a list of
    1-D int arrays of pool indices (NOT admission-local indices)."""
    admitted_idx = np.asarray(admitted_idx)
    k = admitted_idx.shape[0]
    perm = rng.permutation(k)
    n_groups = k // group_size
    return [admitted_idx[perm[i * group_size:(i + 1) * group_size]] for i in range(n_groups)]


def _assert_admission_and_partition(rng, trials=20, N=100, budget=0.25, group_size=5):
    """--selftest (b): budget rounding/determinism + partition shape/coverage/disjointness."""
    for _ in range(trials):
        score = rng.standard_normal(N)
        idx = admit_bottom_q(score, budget)
        k = int(np.floor(budget * N))
        assert idx.shape[0] == k, f"admit_bottom_q: expected {k} admitted, got {idx.shape[0]}"
        assert np.array_equal(idx, np.sort(idx)), "admit_bottom_q: not sorted"
        assert set(idx.tolist()) == set(np.argsort(score)[:k].tolist()), \
            "admit_bottom_q: wrong bottom-k set"
        # determinism: same score -> same admitted set
        idx2 = admit_bottom_q(score, budget)
        assert np.array_equal(idx, idx2), "admit_bottom_q: non-deterministic"

        ridx = admit_random(N, budget, rng)
        assert ridx.shape[0] == k, f"admit_random: expected {k}, got {ridx.shape[0]}"
        assert len(set(ridx.tolist())) == k, "admit_random: duplicate indices"

        groups = partition_groups(idx, group_size, rng)
        assert len(groups) == k // group_size, "partition_groups: wrong group count"
        seen = []
        for g in groups:
            assert len(g) == group_size, "partition_groups: wrong group size"
            seen.extend(g.tolist())
        assert len(seen) == len(set(seen)), "partition_groups: overlapping groups"
        assert set(seen).issubset(set(idx.tolist())), "partition_groups: index outside admitted set"


# ============================================================ retention held-out set (pool-disjoint)
def load_retention_prompts(data_path, n_pool_per_seed, seeds, n_retention=200, seed=RETENTION_SEED):
    """A FIXED, seed-independent 200-prompt CounterFact sample, disjoint from every seed's own
    n_pool_per_seed-candidate pool BY CONSTRUCTION: load_counterfact's shuffle is
    default_rng(seed).shuffle on the SAME file — for a DIFFERENT seed (RETENTION_SEED, fixed,
    never equal to any of the experiment's --seeds by construction below) the resulting order is
    an independent permutation, so exact disjointness is not guaranteed by seed-difference alone.
    We therefore draw retention prompts explicitly EXCLUDING (subject, prompt) pairs that appear
    in ANY of the per-seed candidate pools, so retention never doubles as a candidate at any
    seed. Returns a list of {subject, prompt, target_new, target_true} dicts (same schema as
    load_counterfact)."""
    if seed in set(int(s) for s in seeds):
        raise ValueError(f"[prospadm] RETENTION_SEED={seed} collides with an experiment --seeds "
                         f"value {seeds} — pick a retention seed outside the experiment's seed set")
    excluded = set()
    for s in seeds:
        for e in load_counterfact(data_path, n_pool_per_seed, int(s)):
            excluded.add((e["subject"], e["prompt"]))
    # over-fetch then filter, so excluding the pools still leaves >= n_retention prompts
    pool = load_counterfact(data_path, n_retention + len(excluded) + 50, seed)
    out = [e for e in pool if (e["subject"], e["prompt"]) not in excluded]
    if len(out) < n_retention:
        raise RuntimeError(f"[prospadm] only {len(out)} pool-disjoint retention prompts available "
                           f"(<{n_retention}) — increase the over-fetch margin")
    return out[:n_retention]


def _nvidia_smi_sample():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20)
        return out.stdout.strip().splitlines()[0].strip() if out.returncode == 0 else None
    except Exception:
        return None


def _code_sha256():
    h = hashlib.sha256()
    for rel in ("experiments/prospective_admission.py", "experiments/merging_m0.py",
                "experiments/egl_metrics.py"):
        p = os.path.join(HARNESS, rel)
        try:
            with open(p, "rb") as f:
                h.update(f.read())
        except OSError:
            h.update(b"<missing:" + rel.encode() + b">")
    return h.hexdigest()


def _runner_stamp(wall_start, wall_end, gpu_before, gpu_after):
    """Compute-time provenance stamp, written by the process that actually produced the
    numbers (the Frame-A synthetic-relabel incident is why this cannot be added downstream:
    a stamp attached after the fact certifies nothing). Field set mirrors
    experiments/frame_a/provenance_gate_v2.py's STAMP_REQUIRED_FIELDS so the same gate can
    read it."""
    return {
        "stamp_version": "runner_stamp.v2",
        "code_sha256": _code_sha256(),
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "wall_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(wall_start)),
        "wall_end": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(wall_end)),
        "elapsed_s": round(float(wall_end - wall_start), 3),
        "nvidia_smi_sample": {"before": gpu_before, "after": gpu_after},
    }


def _solo_delta_w(K, R, denom, a, device):
    """Reconstruct edit `a`'s OWN solo ΔW as a singleton "group" through
    merging_m0._merge_factors (Rt=R, Ktsc=K/denom; merged = Rt[a].outer(Ktsc[a])) -- bit-
    identical to the real editor's per-edit ΔW (merging_m0._compute_solo's own recon_rel_err
    check already asserts this <1e-3 for every edit at capture time). No new editor/ROME math:
    this is exactly _measure_merged_groups' merge formula with a group of size 1. Used by
    --ns_reference solo (GPU path) AND by --selftest (CPU, synthetic K/R/denom, device='cpu') --
    the same code is exercised both places."""
    import torch
    Rt, Ktsc = _merge_factors(K, R, denom, device)
    gi = torch.tensor([int(a)], device=Rt.device, dtype=torch.long)
    return Rt.index_select(0, gi).t() @ Ktsc.index_select(0, gi)   # [d_out, d_in]


# ============================================================ GPU: behavioral measurement
def _neighborhood_damage_rate(model, tok, edits, admitted_idx, base_ns, device):
    """Merge-INDUCED neighborhood damage rate over the ADMITTED edits' neighborhood_prompts
    (egl_metrics's canonical NS convention: undamaged iff mean-logprob(target_true) >
    mean-logprob(target_new), scored with the EDIT's own (target_new, target_true) pair — see
    egl_metrics.py's module docstring). `base_ns[a]` = list of booleans (NS-undamaged?) measured
    ONCE per neighbor prompt of edit a, at the reference state chosen by --ns_reference ('base' =
    unedited base model; 'solo' = edit a installed alone — see _base_ns_for / the module
    docstring's "NEIGHBORHOOD-DAMAGE REFERENCE" section); this function reads the CURRENT
    (merged) weights and returns the fraction of (edit, neighbor) pairs that were undamaged at
    that reference but DAMAGED post-merge (the federation's contribution under the chosen
    reference — that base-vs-solo distinction is exactly what --ns_reference selects)."""
    from experiments.egl_metrics import full_target_scores
    n_pairs = 0
    n_induced = 0
    for a in admitted_idx:
        e = edits[a]
        neighbors = e.get("neighborhood_prompts") or []
        pre = base_ns.get(int(a), [])
        for j, nprompt in enumerate(neighbors):
            sc = full_target_scores(model, tok, nprompt, e["target_new"], e["target_true"], device)
            post_undamaged = sc["lp_true"] is not None and sc["lp_true"] > sc["lp_new"]
            n_pairs += 1
            pre_undamaged = pre[j] if j < len(pre) else True
            if pre_undamaged and not post_undamaged:
                n_induced += 1
    return (float(n_induced) / n_pairs if n_pairs > 0 else float("nan")), n_pairs


def _retention_mean_logprob(model, tok, retention_prompts, device):
    from experiments.egl_metrics import mean_logprob_full_target
    vals = [mean_logprob_full_target(model, tok, e["prompt"], e["target_true"], device)
           for e in retention_prompts]
    return float(np.mean(vals))


def _measure_one_group(model, tok, device, W, W_base, Rt, Ktsc, target_tok, edits, group,
                       base_ns, retention_prompts, base_retention_mean):
    """Install ONE group's merged ΔW, measure (a) success, (b) neighborhood-damage, (c) retention
    shift, (d) drop for every member; restore. Mirrors merging_m0._measure_merged_groups' single-
    group install/restore discipline exactly (reused math: Rt/Ktsc from _merge_factors)."""
    import torch
    from metrics import next_token_logits
    gi = torch.tensor(group.tolist(), device=W.device, dtype=torch.long)
    merged = Rt.index_select(0, gi).t() @ Ktsc.index_select(0, gi)
    with torch.no_grad():
        W.add_(merged.to(W.dtype))
    rows = []
    for a in group:
        a = int(a)
        logits = next_token_logits(model, tok, edits[a]["prompt"], device)
        lg = float(logits[int(target_tok[a])].item())
        am = int(logits.argmax().item())
        rows.append({"edit": a, "logit_post": lg,
                    "argmax_ok_post": 1.0 if am == int(target_tok[a]) else 0.0})
    dmg_rate, n_pairs = _neighborhood_damage_rate(model, tok, edits, group, base_ns, device)
    ret_post = _retention_mean_logprob(model, tok, retention_prompts, device)
    with torch.no_grad():
        W.copy_(W_base)
    return rows, dmg_rate, n_pairs, (ret_post - base_retention_mean)


def run_admission_seed(model, tok, layer, device, args, seed, retention_prompts):
    """One seed's full pipeline: pool solo-capture -> pool-wide screening -> 3 admission
    policies (geometry/magnitude/random x n_random_draws) -> g=group_size groups -> install +
    measure each group -> aggregate. Returns a JSON-able dict for this seed."""
    from experiments.egl_metrics import attach_egl_fields, full_target_scores
    import torch
    t0 = time.time()
    pool = load_counterfact(args.data, args.n_pool, seed)
    attach_egl_fields(pool, args.data, "counterfact")
    N = len(pool)
    print(f"[prospadm] seed {seed}: {N}-edit pool loaded + EGL-attached", flush=True)

    vec, W, W_base = _compute_solo(model, tok, layer, device, pool, args.steps, args.lr, t0)
    K, S, key_norm = vec["K"].astype(float), vec["S"].astype(float), vec["key_norm"].astype(float)
    I_cos, I_mag = pool_screening_stats(K, S, key_norm)
    print(f"[prospadm] seed {seed}: solo capture + pool screening done "
          f"({time.time()-t0:.1f}s)", flush=True)

    # neighborhood NS reference, ONCE per admitted-anywhere edit, per --ns_reference (see the
    # module docstring's "NEIGHBORHOOD-DAMAGE REFERENCE" section): base_ns[a][j] = True iff
    # neighbor j of edit a is undamaged at the CHOSEN reference state — the "was this neighbor
    # already fine" baseline the merge-INDUCED metric needs. Computed lazily, only for edits that
    # end up admitted somewhere.
    base_ns_cache = {}
    ns_mode = args.ns_reference

    def _base_ns_for(a):
        """'base' (option ii): measure at the unedited base model — weights are already ==
        W_base here (every _compute_solo edit was individually restored). A neighbor edit `a`'s
        OWN solo edit already damages is never counted as merge-induced.
        'solo' (option i): install edit `a`'s own solo ΔW alone (_solo_delta_w — bit-identical to
        the real editor's ΔW, no new editor math), measure, restore. Isolates the damage the
        FEDERATION adds on top of what the solo edit already did."""
        a = int(a)
        if a in base_ns_cache:
            return base_ns_cache[a]
        e = pool[a]
        neighbors = e.get("neighborhood_prompts") or []

        def _measure():
            out = []
            for nprompt in neighbors:
                sc = full_target_scores(model, tok, nprompt, e["target_new"], e["target_true"], device)
                out.append(bool(sc["lp_true"] is not None and sc["lp_true"] > sc["lp_new"]))
            return out

        if ns_mode == "base":
            out = _measure()
        elif ns_mode == "solo":
            delta = _solo_delta_w(vec["K"], vec["R"], vec["denom"], a, W.device)
            with torch.no_grad():
                W.add_(delta.to(W.dtype))
            out = _measure()
            with torch.no_grad():
                W.copy_(W_base)
        else:
            raise ValueError(f"[prospadm] unknown --ns_reference {ns_mode!r} (expected solo/base)")
        base_ns_cache[a] = out
        return out

    base_retention_mean = _retention_mean_logprob(model, tok, retention_prompts, device)

    policies = {}
    admitted_by_policy = {
        "geometry": [admit_bottom_q(I_cos, args.budget)],
        "magnitude": [admit_bottom_q(I_mag, args.budget)],
        "random": [admit_random(N, args.budget, np.random.default_rng(1_000_000 * seed + d))
                  for d in range(args.n_random_draws)],
    }
    for pname, draws in admitted_by_policy.items():
        draw_reports = []
        for d, admitted in enumerate(draws):
            for a in admitted:
                _base_ns_for(a)
            group_rng = np.random.default_rng(2_000_000 * seed + 7919 * d + POLICY_SALT[pname])
            groups = partition_groups(admitted, args.group_size, group_rng)
            all_rows, dmg_rates, n_pairs_list, ret_shifts = [], [], [], []
            Rt, Ktsc = _merge_factors(vec["K"], vec["R"], vec["denom"], W.device)
            for group in groups:
                rows, dmg_rate, n_pairs, ret_shift = _measure_one_group(
                    model, tok, device, W, W_base, Rt, Ktsc, vec["target_tok"], pool, group,
                    base_ns_cache, retention_prompts, base_retention_mean)
                for r in rows:
                    a = r["edit"]
                    r["drop"] = float(vec["logit_solo"][a] - r["logit_post"])
                all_rows.extend(rows)
                if not np.isnan(dmg_rate):
                    dmg_rates.append(dmg_rate)
                n_pairs_list.append(n_pairs)
                ret_shifts.append(ret_shift)
            succ = float(np.mean([r["argmax_ok_post"] for r in all_rows])) if all_rows else None
            drop_mean = float(np.mean([r["drop"] for r in all_rows])) if all_rows else None
            drop_abs_mean = float(np.mean([abs(r["drop"]) for r in all_rows])) if all_rows else None
            draw_reports.append({
                "n_admitted": int(len(admitted)), "n_groups": len(groups),
                "edit_success_rate": succ,
                "neighborhood_damage_rate_induced": (float(np.mean(dmg_rates))
                                                     if dmg_rates else None),
                "n_neighborhood_pairs": int(sum(n_pairs_list)),
                "retention_shift_mean_logprob": (float(np.mean(ret_shifts)) if ret_shifts else None),
                "target_logit_drop_mean": drop_mean, "target_logit_drop_abs_mean": drop_abs_mean,
            })
            print(f"[prospadm] seed {seed} policy={pname} draw={d}: "
                 f"succ={succ} dmg_induced={draw_reports[-1]['neighborhood_damage_rate_induced']} "
                 f"ret_shift={draw_reports[-1]['retention_shift_mean_logprob']} "
                 f"drop_mean={drop_mean}  ({time.time()-t0:.1f}s)", flush=True)
        policies[pname] = {
            "draws": draw_reports,
            "mean_over_draws": {
                k: (float(np.mean([d[k] for d in draw_reports if d[k] is not None]))
                    if any(d[k] is not None for d in draw_reports) else None)
                for k in ("edit_success_rate", "neighborhood_damage_rate_induced",
                         "retention_shift_mean_logprob", "target_logit_drop_mean",
                         "target_logit_drop_abs_mean")
            },
        }

    with torch.no_grad():
        W.copy_(W_base)
    return {
        "seed": seed, "n_pool": N,
        "esr_solo": round(float(vec["argmax_ok_solo"].mean()), 4),
        "base_retention_mean_logprob": base_retention_mean,
        "policies": policies,
    }


def run_admission(args):
    """Top-level GPU entrypoint: load model once, run every seed, write the results JSON with a
    compute-time runner_stamp. Reached only past main()'s ratification guard."""
    import torch
    t0 = time.time()
    gpu_before = _nvidia_smi_sample()
    model, tok, layer, _nL = _load_edit_model(args.model, args.layer, args.device,
                                              model_dtype=args.model_dtype)
    seeds = [int(x) for x in str(args.seeds).split(",") if x != ""]
    retention_prompts = load_retention_prompts(args.data, args.n_pool, seeds,
                                               n_retention=args.n_retention)
    print(f"[prospadm] {len(retention_prompts)} pool-disjoint retention prompts loaded", flush=True)

    seed_reports = []
    for s in seeds:
        s_t0 = time.time()
        rep = run_admission_seed(model, tok, layer, args.device, args, s, retention_prompts)
        rep["seed_wall_start"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(s_t0))
        rep["seed_wall_end"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        rep["seed_elapsed_s"] = round(time.time() - s_t0, 3)
        seed_reports.append(rep)

    report = {
        "experiment": "prospective_admission", "schema_version": SCHEMA_VERSION,
        "status": "PROSPECTIVE_PREREGISTERED",
        "prereg": os.path.relpath(args.prereg, os.path.dirname(HARNESS)) if args.prereg else None,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": args.model, "model_tag": _model_tag(args.model), "layer": layer,
        "model_dtype": args.model_dtype,
        "dataset": "counterfact", "n_pool": args.n_pool, "budget": args.budget,
        "group_size": args.group_size, "n_random_draws": args.n_random_draws,
        "n_retention": args.n_retention, "seeds": seeds, "ns_reference": args.ns_reference,
        "seed_reports": seed_reports,
        "runner_stamp": _runner_stamp(t0, time.time(), gpu_before, _nvidia_smi_sample()),
    }
    out = args.table_out or os.path.join(args.out_dir, "prospective_admission_table.json")
    _write_table(report, out)
    print(f"[prospadm] wrote {out}  total {time.time()-t0:.1f}s", flush=True)
    return report


# ============================================================ self-test (CPU, no model, no GPU)
def _simulate_ns_reference_dispatch(mode, K, R, denom, a, W):
    """CPU simulation of _base_ns_for's --ns_reference dispatch (the real function needs a model
    to call full_target_scores; this exercises the SAME branching + W install/restore mechanics
    with a fake "measurement" that just snapshots the current W). Returns (measured_snapshot,
    W_after, W_before) so --selftest can assert: 'base' never touches W and measures the
    untouched base state; 'solo' perturbs W with edit `a`'s own solo ΔW during "measurement" then
    restores exactly — the two decision-point options from the module docstring's
    "NEIGHBORHOOD-DAMAGE REFERENCE" section, reduced to their pure-tensor mechanics."""
    import torch
    W_before = W.clone()
    if mode == "base":
        snapshot = W.clone()
    elif mode == "solo":
        delta = _solo_delta_w(K, R, denom, a, W.device)
        with torch.no_grad():
            W.add_(delta.to(W.dtype))
        snapshot = W.clone()          # the fake "measurement" happens here, edit a installed
        with torch.no_grad():
            W.copy_(W_before)
    else:
        raise ValueError(f"unknown ns_reference mode {mode!r}")
    return snapshot, W.clone(), W_before


def selftest():
    rng = np.random.default_rng(20260716)
    print("[selftest] (a) pool-wide I_cos/I_mag vs brute-force O(N^2) ...", flush=True)
    w1 = _assert_pool_screening_bruteforce(rng)
    print(f"[selftest]   worst abs err {w1:.3e} — PASS", flush=True)

    print("[selftest] (b) admission budget rounding/determinism + group partition ...", flush=True)
    _assert_admission_and_partition(rng)
    print("[selftest]   PASS", flush=True)

    print("[selftest] (c) end-to-end aggregation on synthetic per-group measurements ...",
         flush=True)
    N, budget, group_size = 100, 0.25, 5
    K = rng.standard_normal((N, 12))
    key_norm = np.linalg.norm(K, axis=1)
    S = rng.uniform(0.1, 2.0, N)
    I_cos, I_mag = pool_screening_stats(K, S, key_norm)
    geo = admit_bottom_q(I_cos, budget)
    mag = admit_bottom_q(I_mag, budget)
    rnd = [admit_random(N, budget, rng) for _ in range(3)]
    assert len(geo) == len(mag) == int(np.floor(budget * N)) == 25
    for r in rnd:
        assert len(r) == 25
    groups_geo = partition_groups(geo, group_size, rng)
    assert len(groups_geo) == 5
    # synthetic per-group aggregation (no model): fabricate rows, run the SAME reduction the GPU
    # path uses (mean success/drop), assert shapes/types are JSON-able.
    synth_rows = []
    for g in groups_geo:
        for a in g:
            synth_rows.append({"edit": int(a), "argmax_ok_post": float(rng.random() > 0.1),
                              "drop": float(rng.standard_normal())})
    succ = float(np.mean([r["argmax_ok_post"] for r in synth_rows]))
    drop_mean = float(np.mean([r["drop"] for r in synth_rows]))
    assert 0.0 <= succ <= 1.0
    json.dumps({"succ": succ, "drop_mean": drop_mean})  # must be JSON-serialisable
    print(f"[selftest]   synthetic aggregation OK (succ={succ:.3f}, drop_mean={drop_mean:.3f})",
         flush=True)

    print("[selftest] (d) --ns_reference mechanics: solo-ΔW reconstruction + both dispatch "
         "modes (synthetic K/R/denom, CPU, no model) ...", flush=True)
    import torch
    N2, d_in2, d_out2 = 12, 9, 7
    Ksyn = rng.standard_normal((N2, d_in2)).astype(np.float32)
    Rsyn = rng.standard_normal((N2, d_out2)).astype(np.float32)
    denomsyn = (Ksyn ** 2).sum(axis=1).astype(np.float64) + 1e-8
    a_test = int(rng.integers(0, N2))
    delta = _solo_delta_w(Ksyn, Rsyn, denomsyn, a_test, "cpu")
    manual = torch.outer(torch.tensor(Rsyn[a_test]),
                        torch.tensor(Ksyn[a_test] / denomsyn[a_test], dtype=torch.float32))
    assert torch.allclose(delta, manual, atol=1e-6), \
        "_solo_delta_w singleton-group reconstruction != manual outer(r_a, k_a/denom_a)"

    W_base_syn = torch.tensor(rng.standard_normal((d_out2, d_in2)).astype(np.float32))
    snap_base, W_after_base, W_before_base = _simulate_ns_reference_dispatch(
        "base", Ksyn, Rsyn, denomsyn, a_test, W_base_syn.clone())
    assert torch.allclose(snap_base, W_before_base), \
        "'base' mode must measure at the UNTOUCHED base W"
    assert torch.allclose(W_after_base, W_before_base), "'base' mode must never perturb W"

    W_solo_syn = torch.tensor(rng.standard_normal((d_out2, d_in2)).astype(np.float32))
    snap_solo, W_after_solo, W_before_solo = _simulate_ns_reference_dispatch(
        "solo", Ksyn, Rsyn, denomsyn, a_test, W_solo_syn.clone())
    assert not torch.allclose(snap_solo, W_before_solo), \
        "'solo' mode must measure at a PERTURBED (edit-a-installed) W"
    assert torch.allclose(W_after_solo, W_before_solo), \
        "'solo' mode install→restore round-trip FAILED"
    print("[selftest]   solo-ΔW reconstruction matches manual outer(r_a,k_a/denom_a); "
         "'base' measures the untouched W (never perturbs it), 'solo' measures a perturbed W "
         "then restores exactly — PASS", flush=True)

    print("\n[selftest] ALL CHECKS PASSED (pool screening vs brute-force, admission/partition "
         "correctness, synthetic end-to-end aggregation, --ns_reference solo/base dispatch "
         "mechanics) — NO MODEL, NO GPU. This does NOT validate the GPU measurement path "
         "(_measure_one_group / egl_metrics integration, or --ns_reference against a real "
         "full_target_scores call); that requires a real model and is gated behind the "
         "--prereg 'STATUS: RATIFIED' guard.",
         flush=True)


def main():
    ap = argparse.ArgumentParser(description="R-E prospective admission-policy evaluation. The "
                                             "GPU path requires --prereg pointing at a prereg "
                                             "marked 'STATUS: RATIFIED' by the user.")
    ap.add_argument("--selftest", action="store_true",
                    help="CPU self-test: pool-screening vs brute-force + admission/partition "
                         "correctness + synthetic end-to-end aggregation. No model, no GPU.")
    ap.add_argument("--model", default=os.path.join(HARNESS, "data", "models", "Llama-3.2-1B"))
    ap.add_argument("--data", default=os.path.join(HARNESS, "data", "counterfact.json"))
    ap.add_argument("--layer", default="12")
    ap.add_argument("--n_pool", type=int, default=100)
    ap.add_argument("--budget", type=float, default=0.25)
    ap.add_argument("--group_size", type=int, default=5)
    ap.add_argument("--n_random_draws", type=int, default=3)
    ap.add_argument("--n_retention", type=int, default=200)
    ap.add_argument("--ns_reference", choices=["solo", "base"], default=None,
                    help="Neighborhood-damage reference (see the module docstring's "
                         "'NEIGHBORHOOD-DAMAGE REFERENCE' section). 'base' = measure against the "
                         "unedited base model (option ii, includes solo-edit collateral). 'solo' "
                         "= measure with edit a installed alone (option i, true federation-added "
                         "damage). NO DEFAULT -- must be passed explicitly; this is the "
                         "ratification choice, both options are already implemented.")
    ap.add_argument("--prereg", default=None,
                    help="Path to the RATIFIED prereg markdown. The GPU path refuses unless this "
                         "file exists AND contains a line reading exactly 'STATUS: RATIFIED' "
                         "(the user writes that line; no driver or agent may add it on the "
                         "user's behalf).")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--model_dtype", choices=["fp32", "bf16"], default="fp32",
                    help="Frozen-forward dtype, passed through to merging_m0._load_edit_model. "
                         "fp32 (default) is byte-identical to the reference cell. bf16 is for "
                         "the >=7B cloud confirmation arm on a 24GB card; ROME value-opt stays "
                         "fp32 regardless (the editors' own .float() casts).")
    ap.add_argument("--out_dir", default=os.path.join(HARNESS, "results", "prospective_admission"))
    ap.add_argument("--table_out", default=None)
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    if not args.ns_reference:
        raise SystemExit(
            "[prospadm] REFUSING: --ns_reference {solo,base} was not passed. This is ratification "
            "decision point 1 (PREREG-D2-PROSPECTIVE-2026-07-26.md, 'NEIGHBORHOOD-DAMAGE "
            "REFERENCE'; recommended: solo): 'base' = neighbor NS measured against "
            "the unedited base model (option ii, includes solo-edit collateral); 'solo' = "
            "measured with edit a installed alone (option i, true federation-added damage). Both "
            "are implemented -- ratifying means passing one explicitly, not waiting on new code.")
    # Ratification guard. The original revision raised unconditionally because no ratified
    # prereg could exist yet. It now checks the real thing instead: a prereg file the USER has
    # marked 'STATUS: RATIFIED'. Weakening this to a flag the driver can pass itself would make
    # the guard decorative — the whole point is that the authorizing act is the user's.
    if not args.prereg:
        raise SystemExit(
            "[prospadm] REFUSING to run the GPU path: --prereg <ratified-prereg.md> was not "
            "passed. Point it at the FROZEN prereg (docs/plans/PREREG-D2-PROSPECTIVE-2026-07-26"
            ".md); the GPU path runs only once that file contains a line reading exactly "
            "'STATUS: RATIFIED', written by the user. Run --selftest for the CPU-only checks.")
    if not os.path.isfile(args.prereg):
        raise SystemExit(f"[prospadm] REFUSING: --prereg {args.prereg!r} does not exist.")
    with open(args.prereg, encoding="utf-8") as f:
        ratified = any(ln.strip() == "STATUS: RATIFIED" for ln in f)
    if not ratified:
        raise SystemExit(
            f"[prospadm] REFUSING to run the GPU path: {args.prereg} contains no line reading "
            "exactly 'STATUS: RATIFIED'. The prereg is still unratified. The user writes that "
            "line after resolving the decision points; do not add it on their behalf and do not "
            "patch around this guard.")
    run_admission(args)


if __name__ == "__main__":
    main()
