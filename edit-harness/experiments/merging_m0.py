"""merging_m0.py — Merging law M0 kill-gate (direction 2.4, EXPANSION-DIRECTIONS-DEEP).

CLAIM (docs/plans/EXPANSION-DIRECTIONS-DEEP-2026-07-01.md sec 2.4). For two
independently-computed rank-one ROME edits merged by task arithmetic (ΔW_merged =
ΔW_a + ΔW_b, a plain tensor add), the perturbation the OTHER edit imposes on edit
a's key is EXACT and closed-form:

        ΔW_b @ k_a = r_b (k_b · k_a) / ||k_b||^2                              (identity)

    because  ΔW_b = outer(r_b, k_b) / (k_b·k_b)   (editors/rome_native.py:236),
    so       ΔW_b @ k_a = r_b (k_b·k_a) / (k_b·k_b).

The AGGREGATE screening statistic over a merge set is

        I(a) = sum_{b != a} S_b * ||k_a|| * |cos(k_b, k_a)|,   S_b = ||r_b||/||k_b||

and it is exactly the sum of per-cross-term OUTPUT-perturbation norms, since
        ||ΔW_b @ k_a|| = ||r_b|| |k_b·k_a| / ||k_b||^2
                       = (||r_b||/||k_b||) (|k_b·k_a|/||k_b||) ||k_a|| / ||k_a||
                       = S_b ||k_a|| |cos(k_b,k_a)|.
So I(a) is a triangle-inequality bound on ||delta_a|| = ||sum_{b!=a} ΔW_b @ k_a||
and is fully computable on CPU from the saved (k, r) factors — no model needed.

M0 kill-gate (75 GPU-min, 1 seed). 200 independent L12 s0 ROME edits with saved
per-edit (k, r, S, denom); THREE merge regimes measured on the model
(post-merge target-token logit + argmax per participating edit):
  * natural_pairwise  — pool of `pair_pool` edits (default 10 -> C(10,2)=45 pairwise
                        2-edit merges). This is the spec's "natural 20+20 (45
                        pairwise)"; the parenthetical "(45 pairwise)" is honored
                        exactly (pool 10 -> 45 unordered pairs), which also gives
                        the cleanest single-cross-term test of the identity.
  * natural_group     — one merge of `group_size` edits (default 200 = the spec's
                        "100+100"; ΔW addition is associative so two 100-groups
                        summed == one 200-sum). The "safe-federation-at-scale"
                        regime; many-term aggregate I(a).
  * enriched_conflict — conflict pairs (same-relation and/or highest key-cosine),
                        the "near-orthogonality risk confronted head-on" regime.

KILL criteria (all computed + PRINTED by the analysis; see analyze()):
  (1) interference negligible EVEN in the enriched regime
      (median |drop| < 0.1 logit AND argmax-loss rate < 5%)                    -> KILL
  (2) rho(I_cos, drop) < rho_min in any regime where interference is non-neg   -> KILL/MIXED
  (3) GEOMETRY (RE-SCOPED, RG ruling 2026-07-12). The discriminant is the PARTIAL Spearman
      rho(I_cos, drop | I_mag) — the RAW rho-delta (cos vs mag) is MIS-SPECIFIED because I_cos
      and I_mag share the ||k_a||*sum(S) scaffold, so their raw-rho difference hides geometry's
      contribution; the raw delta is kept only as provenance. COHERENCE RULE: a regime carries
      the c3 decision ONLY if it is non-negligible AND NOT saturated (argmax_loss <= sat_argmax,
      i.e. the outcome is not maxed out) AND c2-coherent (rho(I_cos,drop) >= rho_min). Decision
      on the federation-at-scale PRIMARY = natural_group: saturated -> UNINTERPRETABLE (run the
      RG curve); negligible -> fall back to natural_pairwise; else PASS iff partial >= partial_min
      AND geometry survives own-magnitude partialling rho(I_cos,drop | ||k_a||,S_a). enriched is
      EXCLUDED (its high cosines make I_cos ~ I_mag, degenerating the partial by construction).
Negligible interference in a NATURAL regime is NOT a kill (it is the positive "safe to federate"
headline) — criterion (1) keys on the ENRICHED regime only. The RG operating-curve mode
(--rg / --rg_phase2_dir) sweeps group sizes x seeds and applies the pre-registered pass rule
(PRE_REG_PASS_RULE) on the same partial metric to find where scoped federation is safe.

TWO PHASES:
  phase 1 (GPU, run LATER): compute the 200 edits, save (k, r, S, denom, solo
    efficacy) + per-regime membership + per-regime post-merge logits/argmax into
    results/merging/<run_tag>/{phase1_vectors.npz, phase1_regimes.npz,
    phase1_meta.json}; the merge itself is CPU tensor addition of the saved rank-one
    factors (merged = R_g^T @ (K/denom)_g, bit-identical to summing the editor's own
    ΔW_b). Then auto-runs phase 2 and writes results/merging/M0_killgate_table.json.
  phase 2 (CPU): reads the saved arrays, recomputes I(a) and the exact cross-terms
    from the (k, r) factors, the drop = solo_logit - merged_logit from the saved
    logits, the kill-gate stats, and the verdict. Rerunnable STANDALONE from the
    saved vectors (no model, no GPU): merging_m0.py --phase2_dir <run_tag dir>.

SELF-TEST (CPU, no model, no GPU): merging_m0.py --selftest
  (a) asserts the exact identity ΔW_b @ k_a == r_b (k_b·k_a)/||k_b||^2 to fp64 tol
      on random rank-one edits;
  (b) synthesizes the full phase-1 schema for four M0 fixtures — NEGLIGIBLE (-> KILL via
      criterion 1), STRONG (-> PASS, natural_group partial well clear of partial_min),
      GROUP_FAILS (federation-at-scale SATURATED -> c3 UNINTERPRETABLE, exercising the
      coherence rule: the saturated primary is neither passed nor rerouted to g=2 pairwise),
      and HIDDEN_GEOMETRY (geometry hidden under shared magnitude: RAW delta ~0 so the old
      metric misses it, but the PARTIAL detects it > 0.3 -> PASS) — plus two RG operating-curve
      bundles (pass/kill) exercising the pre-registered pass rule. Runs the whole phase-2 /
      analyze_rg pipeline end-to-end and asserts the verdict logic. Writes only under
      results/merging/selftest/ (quarantined).

REUSE (read-only) of the live harness:
  * editors/rome_native.py {_capture_key, find_subject_last_token_index, apply_edit}
    and metrics.py {next_token_logits, first_target_token_id} — imported LAZILY,
    only inside run_phase1 (GPU). Phase 2 + self-test are numpy-only.
  * load_counterfact is REIMPLEMENTED VERBATIM below (cite killgate_keygeom.py:62-82)
    rather than imported, matching the u1_transplant.py / lexical_sbert_baseline.py
    pattern (avoids any coupling to a file a live GPU queue is importing, and keeps
    edit selection byte-aligned with other cells at the same --seed/--n_edits).

Standing workspace rules honored: ROME value-opt stays fp32 (the editor keeps its
own .float() casts; this file never overrides model_dtype); the reported statistic
is signed Spearman, never AUROC; PID-only process control (driver side).
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

HARNESS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HARNESS)

SCHEMA_VERSION = "m0.v1"

# kill-gate thresholds (spec 2.4 defaults; all overridable on the CLI)
DEF_NEG_LOGIT = 0.1     # median |drop| below this (logit) => interference negligible
DEF_NEG_ARGMAX = 0.05   # argmax-loss rate below this => interference negligible
DEF_RHO_MIN = 0.3       # rho(I_cos, drop) below this => regime is NOT c2-coherent (cannot carry c3)
DEF_DRHO_MIN = 0.1      # SECONDARY raw-delta margin (mis-specified; kept for provenance only)
# --- RE-SCOPED metric (RG ruling 2026-07-12): geometry discriminant is the PARTIAL Spearman ---
DEF_PARTIAL_MIN = 0.15  # partial rho(I_cos, drop | I_mag) below this => geometry does not add signal
DEF_SAT_ARGMAX = 0.8    # argmax-loss rate above this => outcome SATURATED => UNINTERPRETABLE for c3


# ============================================================ small numpy stats
def _midrank(x):
    """Tie-averaged ranks (1..n), numpy-only (equivalent to scipy rankdata 'average').
    Same convention analyze_matrices._midrank / the project's Spearman use."""
    x = np.asarray(x, float)
    n = x.size
    order = np.argsort(x, kind="mergesort")
    sx = x[order]
    ranks = np.empty(n, float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def _spearman(a, b):
    """Signed Spearman rho over paired scalars (NaN-safe, midrank ties). Never AUROC."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if a.size < 3:
        return float("nan")
    ra, rb = _midrank(a), _midrank(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _residualize_multi(y_rank, z_ranks):
    """Least-squares residual of y_rank on [1, z_ranks...] in rank space. Mirrors
    experiments/halflife_hl0.py:_residualize_multi (which generalises analyze_sequential.
    _residualize to >=1 confound) — the house partial-correlation construction."""
    cols = [np.ones_like(y_rank)] + [np.asarray(z, float) for z in z_ranks]
    Z = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(Z, y_rank, rcond=None)
    return y_rank - Z @ beta


def partial_spearman_multi(x, y, zlist):
    """Signed partial Spearman rho(x, y | z1..zk): correlate the rank-space residuals of x
    and y after regressing each on ALL confounds jointly. Returns rho only (NaN-safe).
    Verbatim convention of experiments/halflife_hl0.py:partial_spearman_multi (rank-transform
    x, y, each z via _midrank; residualize both sides on [1, rank(z)...]; Pearson of residuals).
    This is the PRIMARY geometry discriminant: I_cos and I_mag share the ||k_a||*sum(S) scaffold,
    so their raw-rho DIFFERENCE hides geometry's contribution — residualizing I_mag out of both
    I_cos and drop isolates the unique cosine signal."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    zlist = [np.asarray(z, float) for z in zlist]
    m = np.isfinite(x) & np.isfinite(y)
    for z in zlist:
        m = m & np.isfinite(z)
    x, y = x[m], y[m]
    zlist = [z[m] for z in zlist]
    if x.size < 4 + len(zlist):
        return float("nan")
    rx, ry = _midrank(x), _midrank(y)
    rz = [_midrank(z) for z in zlist]
    if rx.std() == 0 or ry.std() == 0 or any(r.std() == 0 for r in rz):
        return float("nan")
    rx_res = _residualize_multi(rx, rz)
    ry_res = _residualize_multi(ry, rz)
    if rx_res.std() == 0 or ry_res.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx_res, ry_res)[0, 1])


# ============================================================ rank-one merge math
def rank_one_delta(r_b, k_b, denom_b):
    """The exact ΔW_b the ROME editor applies: outer(r_b, k_b)/denom_b, [d_out, d_in].
    denom_b == k_b·k_b (+1e-8 in the live editor). Replicates editors/rome_native.py:236."""
    return np.outer(np.asarray(r_b, float), np.asarray(k_b, float)) / float(denom_b)


def cross_term(r_b, k_b, k_a, denom_b):
    """Exact ΔW_b @ k_a == r_b (k_b·k_a)/denom_b, [d_out]. The closed-form identity —
    computed WITHOUT materializing the dense ΔW_b."""
    r_b = np.asarray(r_b, float)
    return r_b * (float(np.dot(k_b, k_a)) / float(denom_b))


def _relation(prompt, subject):
    """Prompt with the subject string removed = crude relation template.
    Reimplemented verbatim from experiments/u1_transplant.py:107-109."""
    return str(prompt).lower().replace(str(subject).lower(), " ").strip()


# ============================================================ counterfact loader
def load_counterfact(path, n_edits, seed=0):
    """Edit-bank loader, REIMPLEMENTED VERBATIM from killgate_keygeom.py:62-82 (n_probes
    slice dropped: M0 only needs the edit bank). Same json.load -> default_rng(seed).shuffle
    -> requested_rewrite parse -> first-n slice, so the edit ORDER is byte-identical to any
    killgate/mechanism cell at the same (seed, n_edits)."""
    data = json.load(open(path))
    rng = np.random.default_rng(seed)
    rng.shuffle(data)
    recs = []
    for d in data:
        rr = d.get("requested_rewrite", d)
        try:
            subj = rr["subject"]
            prompt = rr["prompt"].format(subj) if "{}" in rr["prompt"] else rr["prompt"]
            tnew = rr["target_new"]["str"] if isinstance(rr["target_new"], dict) else rr["target_new"]
            ttrue = rr["target_true"]["str"] if isinstance(rr["target_true"], dict) else rr["target_true"]
        except Exception:
            continue
        recs.append({"subject": subj, "prompt": prompt, "target_new": tnew, "target_true": ttrue})
        if len(recs) >= n_edits:
            break
    return recs[:n_edits]


# ============================================================ regime construction
def build_enriched_pairs(K, key_norm, edits_meta, n_enriched, max_appear=3):
    """Conflict pairs for the enriched regime: same-relation pairs AND highest
    |cos(k_a,k_b)| pairs (the geometric near-orthogonality risk the law is about),
    unioned and greedily capped so each edit appears in <= max_appear pairs (keeps
    variety, avoids one hub edit dominating). Returns list of [a, b] index pairs."""
    N = len(K)
    Kn = K / (key_norm[:, None] + 1e-12)
    absc = np.abs(Kn @ Kn.T)
    np.fill_diagonal(absc, -1.0)

    cand = {}  # (a,b) a<b -> |cos|, deduped
    # (1) same-relation pairs (semantic conflict)
    buckets = defaultdict(list)
    for i, e in enumerate(edits_meta):
        buckets[_relation(e["prompt"], e["subject"])].append(i)
    for members in buckets.values():
        if len(members) >= 2:
            for a, b in itertools.combinations(sorted(members), 2):
                cand[(a, b)] = float(absc[a, b])
    # (2) globally highest key-cosine pairs (geometric conflict)
    if N >= 2:
        iu = np.triu_indices(N, 1)
        vals = absc[iu]
        take = min(len(vals), max(n_enriched * 4, n_enriched))
        for t in np.argsort(-vals)[:take]:
            a, b = int(iu[0][t]), int(iu[1][t])
            cand.setdefault((a, b), float(vals[t]))

    # greedy select by descending |cos| with per-edit appearance cap
    used = defaultdict(int)
    out = []
    for a, b in sorted(cand, key=lambda p: -cand[p]):   # pairs by descending |cos|
        if used[a] < max_appear and used[b] < max_appear:
            out.append([a, b])
            used[a] += 1
            used[b] += 1
        if len(out) >= n_enriched:
            break
    return out


def build_regimes(K, key_norm, edits_meta, N, pair_pool, group_size, n_enriched):
    """Return [(name, groups)], groups = list of edit-index lists (each list is one merge)."""
    regimes = []
    pool = min(pair_pool, N)
    pairs = [[a, b] for a, b in itertools.combinations(range(pool), 2)]
    regimes.append(("natural_pairwise", pairs))

    g = min(group_size, N)
    regimes.append(("natural_group", [list(range(g))]))

    regimes.append(("enriched_conflict",
                    build_enriched_pairs(K, key_norm, edits_meta, n_enriched)))
    return regimes


# ============================================================ schema I/O
def save_phase1(run_dir, vectors, regimes_arr, meta):
    os.makedirs(run_dir, exist_ok=True)
    # atomic tmp+os.replace for the npz too (matches killgate_keygeom.py:963 precedent):
    # np.savez_compressed keeps a path already ending in .npz as-is, so ".tmp.npz" is the
    # written file and the replace target has no accidental double-suffix.
    for fname, arrs in (("phase1_vectors.npz", vectors), ("phase1_regimes.npz", regimes_arr)):
        final = os.path.join(run_dir, fname)
        tmp = final + ".tmp.npz"
        np.savez_compressed(tmp, **arrs)
        os.replace(tmp, final)
    tmp = os.path.join(run_dir, "phase1_meta.json.tmp")
    with open(tmp, "w") as f:
        json.dump(meta, f, indent=2)
    os.replace(tmp, os.path.join(run_dir, "phase1_meta.json"))


def load_phase1(run_dir):
    v = dict(np.load(os.path.join(run_dir, "phase1_vectors.npz")))
    r = dict(np.load(os.path.join(run_dir, "phase1_regimes.npz")))
    with open(os.path.join(run_dir, "phase1_meta.json")) as f:
        meta = json.load(f)
    return v, r, meta


def _default_thresholds():
    return dict(neg_logit=DEF_NEG_LOGIT, neg_argmax=DEF_NEG_ARGMAX, rho_min=DEF_RHO_MIN,
                drho_min=DEF_DRHO_MIN, partial_min=DEF_PARTIAL_MIN, sat_argmax=DEF_SAT_ARGMAX)


def _regime_stat(vecs, obs_edit, obs_group, obs_logit_post, obs_argmax_ok_post, members, t):
    """Per-regime stats from the merge observations (each = edit a in merge group g).
    members: {group_id -> [edit indices]}. Computes, over observations:
      drop = solo_logit - merged_logit; I_cos, I_mag; exact ||delta_a||;
      raw rho(I_cos,drop) / rho(I_mag,drop) / raw delta (SECONDARY, mis-specified — I_cos & I_mag
        share the ||k_a||*sum(S) scaffold so their raw-rho difference hides geometry);
      partial rho(I_cos, drop | I_mag)         [PRIMARY geometry discriminant];
      partial rho(I_cos, drop | ||k_a||, S_a)  [own-magnitude partial];
    and the coherence flags: saturated (argmax_loss > sat_argmax => outcome maxed out =>
    UNINTERPRETABLE), c2_coherent (rho(I_cos,drop) >= rho_min), c3_eligible (non-neg AND not
    saturated AND c2-coherent), collapses_under_ownmag_partial (own-mag partial < partial_min)."""
    K, R, denom, S, key_norm, logit_solo, argmax_ok_solo = vecs
    I_cos, I_mag, normdelta, drop = [], [], [], []
    kn_a, S_a, argmax_loss, worked = [], [], [], []
    for a, g, lp, ap in zip(obs_edit, obs_group, obs_logit_post, obs_argmax_ok_post):
        a, g = int(a), int(g)
        others = [b for b in members[g] if b != a]
        if others:
            ob = np.array(others)
            coses = np.array([abs(float(np.dot(K[b], K[a])) /
                                  (key_norm[b] * key_norm[a] + 1e-12)) for b in others])
            ic = float(key_norm[a] * np.sum(S[ob] * coses))
            im = float(key_norm[a] * np.sum(S[ob]))
            d_a = np.zeros(R.shape[1])
            for b in others:
                d_a += cross_term(R[b], K[b], K[a], denom[b])
            nd = float(np.linalg.norm(d_a))
        else:
            ic = im = nd = 0.0
        I_cos.append(ic); I_mag.append(im); normdelta.append(nd)
        drop.append(float(logit_solo[a] - lp))
        kn_a.append(float(key_norm[a])); S_a.append(float(S[a]))
        ws = argmax_ok_solo[a] > 0.5
        worked.append(bool(ws)); argmax_loss.append(bool(ws and ap < 0.5))

    I_cos = np.array(I_cos); I_mag = np.array(I_mag); normdelta = np.array(normdelta)
    drop = np.array(drop); kn_a = np.array(kn_a); S_a = np.array(S_a)
    argmax_loss = np.array(argmax_loss, bool); worked = np.array(worked, bool)

    n_worked = int(worked.sum())
    # negligibility gate uses the median of the ABSOLUTE drop, not the signed median: a merge
    # can push a target logit either way, so a half-up/half-down regime would show signed median
    # ~0 yet is clearly NOT negligible. |drop| cannot hide real perturbation behind sign
    # cancellation (conservative vs spurious KILLs); signed mean is reported as mean_drop_logit.
    med_abs_drop = float(np.median(np.abs(drop))) if drop.size else float("nan")
    argmax_loss_rate = (float(argmax_loss.sum() / n_worked) if n_worked > 0 else float("nan"))
    rho_cos = _spearman(I_cos, drop)
    rho_mag = _spearman(I_mag, drop)
    drho = (float(rho_cos - rho_mag) if np.isfinite(rho_cos) and np.isfinite(rho_mag)
            else float("nan"))
    partial_geom = partial_spearman_multi(I_cos, drop, [I_mag])
    partial_ownmag = partial_spearman_multi(I_cos, drop, [kn_a, S_a])
    xcheck = _spearman(I_cos, normdelta)

    non_negligible = bool((np.isfinite(med_abs_drop) and med_abs_drop >= t["neg_logit"]) or
                          (np.isfinite(argmax_loss_rate) and argmax_loss_rate >= t["neg_argmax"]))
    saturated = bool(np.isfinite(argmax_loss_rate) and argmax_loss_rate > t["sat_argmax"])
    c2_coherent = bool(np.isfinite(rho_cos) and rho_cos >= t["rho_min"])
    c3_eligible = bool(non_negligible and (not saturated) and c2_coherent)
    collapses_ownmag = not bool(np.isfinite(partial_ownmag) and partial_ownmag >= t["partial_min"])

    def rnd(x):
        return round(float(x), 4) if (x is not None and np.isfinite(x)) else None
    return {
        "n_obs": int(len(drop)),
        "n_groups": int(len(members)),
        "n_worked_solo": n_worked,
        "median_abs_drop_logit": rnd(med_abs_drop),
        "mean_drop_logit": (round(float(np.mean(drop)), 5) if drop.size else None),
        "argmax_loss_rate": rnd(argmax_loss_rate),
        "rho_I_cos_drop": rnd(rho_cos),
        "rho_I_magonly_drop": rnd(rho_mag),
        # PRIMARY geometry metric (RG ruling): partial rho(I_cos, drop | I_mag)
        "partial_rho_geom": rnd(partial_geom),
        # own-magnitude partial: rho(I_cos, drop | ||k_a||, S_a)
        "partial_rho_geom_ownmag": rnd(partial_ownmag),
        # SECONDARY (mis-specified; provenance only — kept so the old metric stays auditable)
        "delta_rho_cos_minus_mag": rnd(drho),
        "rho_I_cos_normdelta_cpu_xcheck": rnd(xcheck),
        "non_negligible": non_negligible,
        "saturated": saturated,
        "c2_coherent": c2_coherent,
        "c3_eligible": c3_eligible,
        "collapses_under_ownmag_partial": collapses_ownmag,
    }


# ============================================================ phase 2 analysis (CPU)
def analyze(vectors, regimes_arr, meta, thr=None):
    """Recompute I(a), exact cross-terms, drops, kill-gate stats + verdict — all CPU,
    only from the saved arrays. `thr` overrides the default thresholds."""
    t = _default_thresholds()
    if thr:
        t.update(thr)

    K = vectors["K"].astype(float)
    R = vectors["R"].astype(float)
    denom = vectors["denom"].astype(float)
    S = vectors["S"].astype(float)
    key_norm = vectors["key_norm"].astype(float)
    logit_solo = vectors["logit_solo"].astype(float)
    argmax_ok_solo = vectors["argmax_ok_solo"].astype(float)

    names = [str(x) for x in regimes_arr["regime_names"]]
    obs_regime = regimes_arr["obs_regime"].astype(int)
    obs_group = regimes_arr["obs_group"].astype(int)
    obs_edit = regimes_arr["obs_edit"].astype(int)
    obs_logit_post = regimes_arr["obs_logit_post"].astype(float)
    obs_argmax_ok_post = regimes_arr["obs_argmax_ok_post"].astype(float)
    mem_regime = regimes_arr["mem_regime"].astype(int)
    mem_group = regimes_arr["mem_group"].astype(int)
    mem_edit = regimes_arr["mem_edit"].astype(int)

    # (regime, group) -> member edit indices
    members_all = defaultdict(list)
    for rr, gg, ee in zip(mem_regime, mem_group, mem_edit):
        members_all[(int(rr), int(gg))].append(int(ee))

    vecs = (K, R, denom, S, key_norm, logit_solo, argmax_ok_solo)
    regime_stats = {}
    for rid, name in enumerate(names):
        sel = np.where(obs_regime == rid)[0]
        members = {int(g): members_all[(rid, int(g))] for g in set(obs_group[sel].tolist())}
        regime_stats[name] = _regime_stat(
            vecs, obs_edit[sel], obs_group[sel], obs_logit_post[sel], obs_argmax_ok_post[sel],
            members, t)

    verdict = _verdict(regime_stats, t)
    table = {
        "experiment": "merging_law_M0",
        "schema_version": SCHEMA_VERSION,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": meta.get("model"),
        "layer": meta.get("layer"),
        "seed": meta.get("seed"),
        "n_edits": meta.get("n_edits"),
        "dataset": meta.get("dataset", "counterfact"),
        "thresholds": t,
        "drop_metric_note": ("negligibility uses median|drop_logit| (conservative vs spurious "
                             "kills; sign-cancellation cannot hide real perturbation); "
                             "mean_drop_logit reports signed directionality per regime"),
        "geometry_metric_note": ("PRIMARY geometry discriminant = partial_rho_geom = partial "
                                 "Spearman rho(I_cos, drop | I_mag). The raw delta_rho_cos_minus_mag "
                                 "is MIS-SPECIFIED (I_cos & I_mag share the ||k_a||*sum(S) scaffold, "
                                 "so their raw-rho difference hides geometry) and kept only as "
                                 "provenance. A regime carries the c3 decision only if it is "
                                 "non-negligible, NOT saturated (argmax_loss<=sat_argmax), and "
                                 "c2-coherent (rho(I_cos,drop)>=rho_min)."),
        "regimes": regime_stats,
        "verdict": verdict,
    }
    return table


def _verdict(regime_stats, t):
    """KILL / PASS / MIXED per criterion + overall. See module docstring."""
    enriched = next((n for n in regime_stats if n.startswith("enriched")), None)
    nonneg = [n for n in regime_stats if regime_stats[n]["non_negligible"]]

    # criterion 1 — interference negligible EVEN in the enriched regime
    if enriched is None:
        c1 = "N/A"
    else:
        c1 = "PASS" if regime_stats[enriched]["non_negligible"] else "KILL"

    # criterion 2 — rho(I,drop) >= rho_min wherever interference is non-negligible
    def _passes_rho(n):
        r = regime_stats[n]["rho_I_cos_drop"]
        return (r is not None) and (r >= t["rho_min"])
    if not nonneg:
        c2 = "N/A"
    else:
        pr = [_passes_rho(n) for n in nonneg]
        c2 = "PASS" if all(pr) else ("KILL" if not any(pr) else "MIXED")

    # criterion 3 — RE-SCOPED (RG ruling 2026-07-12). Geometry discriminant = PARTIAL Spearman
    # rho(I_cos, drop | I_mag); the raw delta is mis-specified (I_cos & I_mag share the
    # ||k_a||*sum(S) scaffold), kept only as provenance. COHERENCE RULE: a regime carries the c3
    # decision only if non-negligible AND NOT saturated (argmax_loss<=sat_argmax) AND c2-coherent
    # (rho(I_cos,drop)>=rho_min). Decision tree on the federation-at-scale PRIMARY (natural_group):
    #   saturated              -> UNINTERPRETABLE (outcome maxed out; geometry-at-scale
    #                             unanswerable from this run -> that is what the RG curve is for);
    #                             do NOT substitute pairwise (g=2 is not the federation question).
    #   negligible             -> fall back to natural_pairwise (safe federation at scale; decide
    #                             c3 where interference is actually real). enriched NEVER decides.
    #   non-neg, c2-incoherent -> KILL (interference real but I_cos does not even raw-correlate).
    #   c3_eligible            -> decide by PASS iff partial_rho_geom>=partial_min AND geometry
    #                             does NOT collapse under own-magnitude partialling; else KILL.
    def _decide(st):
        p = st["partial_rho_geom"]
        ok = (p is not None) and (p >= t["partial_min"]) and (not st["collapses_under_ownmag_partial"])
        return "PASS" if ok else "KILL"
    ng_st = regime_stats.get("natural_group")
    npw_st = regime_stats.get("natural_pairwise")
    if ng_st is None:
        c3, c3_regime = "N/A", None
    elif ng_st["saturated"]:
        c3, c3_regime = "UNINTERPRETABLE", "natural_group"
    elif not ng_st["non_negligible"]:
        if npw_st is not None and npw_st["c3_eligible"]:
            c3, c3_regime = _decide(npw_st), "natural_pairwise"
        else:
            c3, c3_regime = "N/A", None
    elif not ng_st["c2_coherent"]:
        c3, c3_regime = "KILL", "natural_group"
    else:
        c3, c3_regime = _decide(ng_st), "natural_group"

    c3_rule = ("geometry metric = partial rho(I_cos, drop | I_mag) >= partial_min; decision regime "
               "primary=natural_group, fallback=natural_pairwise (only if group negligible), "
               "enriched=excluded-by-construction; a regime carries c3 only if non-negligible AND "
               f"not saturated (argmax_loss<={t['sat_argmax']}) AND c2-coherent "
               f"(rho(I_cos,drop)>={t['rho_min']}); PASS also requires geometry survives "
               "own-magnitude partialling (rho(I_cos,drop|||k_a||,S_a)>=partial_min)")
    partial_by_regime = {n: regime_stats[n]["partial_rho_geom"] for n in regime_stats}
    ownmag_by_regime = {n: regime_stats[n]["partial_rho_geom_ownmag"] for n in regime_stats}
    drho_by_regime = {n: regime_stats[n]["delta_rho_cos_minus_mag"] for n in regime_stats}

    if "KILL" in (c1, c2, c3):
        overall = "KILL"
    elif c3 in ("UNINTERPRETABLE", "N/A") or c2 == "MIXED":
        overall = "MIXED"
    else:
        overall = "PASS"

    if overall == "KILL":
        interp = ("geometry does not predict merge interference where it is testable — abandon "
                  "or heavily re-scope the merging-law direction")
    elif overall == "PASS":
        interp = ("partial rho(I_cos, drop | I_mag) clears the margin in the c3 decision regime — "
                  "geometry adds signal beyond magnitude, survives own-magnitude partialling")
    elif c3 == "UNINTERPRETABLE":
        interp = ("federation-at-scale regime is SATURATED (outcome maxed out, argmax_loss>"
                  f"{t['sat_argmax']}) so geometry-at-scale is UNINTERPRETABLE from this run — "
                  "run the RG operating curve over smaller group sizes (do not read the raw "
                  "delta at scale as evidence)")
    else:
        interp = ("mixed / inconclusive — geometry signal is not uniform across testable "
                  "regimes; scope with the RG operating curve, do not kill")

    return {
        "overall": overall,
        "criterion_1_negligible_even_enriched": c1,
        "criterion_2_rho_where_nonnegligible": c2,
        "criterion_3_geometry_partial": c3,
        "criterion_3_metric": "partial_rho(I_cos, drop | I_mag)",
        "criterion_3_rule": c3_rule,
        "criterion_3_decided_on": c3_regime,
        "criterion_3_partial_by_regime": partial_by_regime,
        "criterion_3_ownmag_partial_by_regime": ownmag_by_regime,
        "criterion_3_delta_rho_by_regime_SECONDARY": drho_by_regime,
        "enriched_regime": enriched,
        "non_negligible_regimes": nonneg,
        "saturated_regimes": [n for n in regime_stats if regime_stats[n]["saturated"]],
        "c3_eligible_regimes": [n for n in regime_stats if regime_stats[n]["c3_eligible"]],
        "interpretation": interp,
    }


def print_table(table):
    v = table["verdict"]
    print("\n=== MERGING LAW M0 KILL-GATE ===", flush=True)
    print(f"model={table['model']} layer={table['layer']} seed={table['seed']} "
          f"n_edits={table['n_edits']}", flush=True)
    for name, st in table["regimes"].items():
        print(f"  [{name}] n_obs={st['n_obs']} groups={st['n_groups']} "
              f"med|drop|={st['median_abs_drop_logit']} "
              f"argmax_loss={st['argmax_loss_rate']} "
              f"rho(Icos,drop)={st['rho_I_cos_drop']} "
              f"PARTIAL(Icos,drop|Imag)={st['partial_rho_geom']} "
              f"partial|ownmag={st['partial_rho_geom_ownmag']} "
              f"raw_drho={st['delta_rho_cos_minus_mag']} "
              f"nonneg={st['non_negligible']} sat={st['saturated']} c3elig={st['c3_eligible']}",
              flush=True)
    print(f"  VERDICT overall={v['overall']} | c1(neg-enriched)={v['criterion_1_negligible_even_enriched']} "
          f"c2(rho)={v['criterion_2_rho_where_nonnegligible']} "
          f"c3(geom-partial)={v['criterion_3_geometry_partial']} "
          f"[c3 decided_on={v['criterion_3_decided_on']}]", flush=True)
    print(f"  -> {v['interpretation']}", flush=True)


# ============================================================ phase 1 GPU helpers (shared M0+RG)
def _load_edit_model(model_path, layer_arg, device, model_dtype="fp32", device_map="none"):
    """Load the model (fp32 default), arch-normalize, resolve edit layer.

    model_dtype/device_map are OFF by default (fp32, single `.to(device)`) — byte-identical
    to the old hardcode for every existing caller (merging_editors.py's 4 call sites included,
    which never pass these kwargs). ROME/AlphaEdit/MEMIT value-opt stays fp32 regardless of
    model_dtype (the editors' own `.float()` casts — see editors/rome_native.py's "bf16
    boundary" comments); only the frozen forward runs at model_dtype.

    device_map (R-C revision-wave addition, 2026-07-16): 'none' is the old path unchanged.
    Any other value ('auto'/'balanced'/'balanced_low_0'/'sequential') hands loading to
    accelerate for TENSOR-PARALLEL sharding across multiple GPUs (mirrors
    experiments/killgate_keygeom.py's --device_map path exactly, incl. skipping the
    collapsing `.to(device)` call and re-resolving `device` to the input-embedding device
    afterward). Callers that need the layer-edited weight's own device for GPU-tensor
    construction (the RG merge path) must separately call
    tp_edit_util.resolve_layer_device(model, layer) — this function only fixes the INPUT
    (tokenizer-encode) device, returned implicitly by re-deriving it via
    tp_edit_util.resolve_input_device(model, device) at the call site (the return signature
    here is intentionally left as the old 4-tuple so merging_editors.py's imports never break;
    see run_phase_rg for the device_map-aware caller)."""
    import torch  # lazy — phase 2 + self-test stay numpy-only
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from editors.arch_compat import normalize_arch  # noqa: E402
    from tp_edit_util import resolve_input_device  # noqa: E402
    tok = AutoTokenizer.from_pretrained(model_path)
    load_dtype = torch.float32 if model_dtype == "fp32" else torch.bfloat16
    if device_map == "none":
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=load_dtype).to(device).eval()
    else:
        # TP path: do NOT call .to(device) here — on an accelerate-dispatched model that
        # collapses the shard placement back onto one card (tp_edit_util.py docstring).
        model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=load_dtype, device_map=device_map).eval()
        # hf_device_map is only stamped when accelerate actually dispatches; on a single
        # large card (Pro-6000 96GB) device_map="auto" can keep the model whole and leave
        # the attribute unset — never let this diagnostic print kill the run (box 29246
        # rc-1-at-49s, 2026-07-19).
        print(f"[m0] --device_map={device_map}: hf_device_map={getattr(model, 'hf_device_map', None)}", flush=True)
    device = resolve_input_device(model, device)
    normalize_arch(model, tok, device)
    nL = model.config.num_hidden_layers
    layer = nL // 2 if layer_arg == "auto" else int(layer_arg)
    print(f"[m0] loaded {model_path} ({nL} layers, edit layer={layer}, device={device}, "
          f"dtype={model_dtype}, device_map={device_map})", flush=True)
    return model, tok, layer, nL


def _compute_solo(model, tok, layer, device, edits, steps, lr, t0=None):
    """Compute per-edit ROME rank-one factors + solo efficacy (weights RESTORED after each edit
    => every edit sees base weights). Returns (vectors dict, W, W_base). fp32/restore/recon logic
    lives HERE only so M0 phase-1 and the RG operating curve share one audited implementation."""
    import torch
    from metrics import next_token_logits, first_target_token_id  # noqa: E402
    from editors.rome_native import (  # noqa: E402
        _capture_key, find_subject_last_token_index, apply_edit,
    )
    if t0 is None:
        t0 = time.time()
    N = len(edits)
    W = model.model.layers[layer].mlp.down_proj.weight
    W_base = W.detach().clone()
    d_out, d_in = int(W.shape[0]), int(W.shape[1])
    K = np.zeros((N, d_in), np.float32); Rv = np.zeros((N, d_out), np.float32)
    denom = np.zeros(N, np.float64); key_norm = np.zeros(N, np.float32)
    resid_norm = np.zeros(N, np.float32); S = np.zeros(N, np.float32)
    target_tok = np.zeros(N, np.int64); logit_solo = np.full(N, np.nan, np.float32)
    argmax_ok_solo = np.zeros(N, np.float32); recon_rel_err = np.zeros(N, np.float32)
    for i, e in enumerate(edits):
        idx = find_subject_last_token_index(tok, e["prompt"], e["subject"])
        k = _capture_key(model, tok, layer, e["prompt"], idx, device).float().cpu().numpy()
        tgt = first_target_token_id(tok, e["target_new"])
        info = apply_edit(model, tok, e, {"layer": layer, "steps": steps, "lr": lr}, device)
        r = np.asarray(info["residual_vec"], np.float32)   # v - Wk, dim d_out
        dn = float(k @ k) + 1e-8                            # editor's exact denom
        recon_norm = float(np.linalg.norm(rank_one_delta(r, k, dn)))
        applied = float(info["delta_weight_norm"])
        rre = abs(recon_norm - applied) / (applied + 1e-30)
        recon_rel_err[i] = rre
        if rre > 1e-3:
            print(f"[m0] WARN recon_rel_err={rre:.3g} at edit {i} (>1e-3)", flush=True)
        logits = next_token_logits(model, tok, e["prompt"], device)  # under EDITED weights
        lg = float(logits[tgt].item()); am = int(logits.argmax().item())
        K[i] = k; Rv[i] = r; denom[i] = dn
        key_norm[i] = float(np.linalg.norm(k)); resid_norm[i] = float(np.linalg.norm(r))
        S[i] = resid_norm[i] / (key_norm[i] + 1e-12)
        target_tok[i] = tgt; logit_solo[i] = lg; argmax_ok_solo[i] = 1.0 if am == tgt else 0.0
        with torch.no_grad():
            W.copy_(W_base)
        if (i + 1) % 20 == 0:
            print(f"[m0] edit {i+1}/{N}  {time.time()-t0:.1f}s", flush=True)
    with torch.no_grad():
        W.copy_(W_base)
    assert torch.allclose(W, W_base), "[m0] solo-loop final restore FAILED"
    print(f"[m0] edits done; esr={float(argmax_ok_solo.mean()):.3f} "
          f"max_recon_rel_err={float(recon_rel_err.max()):.2e}  {time.time()-t0:.1f}s", flush=True)
    vectors = dict(
        K=K, R=Rv, denom=denom.astype(np.float64), S=S, key_norm=key_norm, resid_norm=resid_norm,
        target_tok=target_tok, logit_solo=logit_solo, argmax_ok_solo=argmax_ok_solo,
        recon_rel_err=recon_rel_err,
    )
    return vectors, W, W_base


def _merge_factors(K, R, denom, device):
    """Torch factors for the fast merged-ΔW = R_g^T @ (K/denom)_g (== summing the editor's own
    rank-one ΔW_b exactly, in fp32)."""
    import torch
    Rt = torch.tensor(np.asarray(R), device=device, dtype=torch.float32)
    Ktsc = torch.tensor(np.asarray(K) / np.asarray(denom)[:, None], device=device, dtype=torch.float32)
    return Rt, Ktsc


def _measure_merged_groups(model, tok, device, W, W_base, Rt, Ktsc, target_tok, edits, groups,
                           input_device=None):
    """For each group (list of edit idx): form merged ΔW, apply, measure every member's
    post-merge target-token logit + argmax, restore. Returns parallel lists
    (gid, edit, logit_post, argmax_ok_post); gid = index into `groups`.

    `device` here is the device Rt/Ktsc/W live on (the EDITED LAYER's device — under
    single-GPU --device_map none this is the same value as the tokenizer/input device, so
    every pre-existing caller that passes just `device` positionally is unaffected).
    `input_device` (optional, R-C addition 2026-07-16) is the tokenizer-encode device for
    next_token_logits; defaults to `device` when omitted, which is exactly the old
    behaviour. Under multi-GPU --device_map the two can differ (the edited layer may sit on
    a different card than the embedding) — see tp_edit_util.py / run_phase_rg."""
    import torch
    from metrics import next_token_logits  # noqa: E402
    input_device = device if input_device is None else input_device
    G_id, G_edit, G_lp, G_ap = [], [], [], []
    for gid, group in enumerate(groups):
        gi = torch.tensor(group, device=device, dtype=torch.long)
        merged = Rt.index_select(0, gi).t() @ Ktsc.index_select(0, gi)   # [d_out, d_in]
        with torch.no_grad():
            W.add_(merged.to(W.dtype))
        for a in group:
            logits = next_token_logits(model, tok, edits[a]["prompt"], input_device)
            lg = float(logits[int(target_tok[a])].item()); am = int(logits.argmax().item())
            G_id.append(gid); G_edit.append(a); G_lp.append(lg)
            G_ap.append(1.0 if am == int(target_tok[a]) else 0.0)
        with torch.no_grad():
            W.copy_(W_base)
    with torch.no_grad():
        W.copy_(W_base)
    assert torch.allclose(W, W_base), "[m0] merge-measure final restore FAILED"
    return G_id, G_edit, G_lp, G_ap


def _tiled_groups(N, g, seed):
    """Disjoint random tiling of N edits into floor(N/g) groups of size g (drops the remainder).
    ~N observations per group size. seed offset keeps the tiling distinct from edit-selection."""
    rng = np.random.default_rng(int(seed) + 20260712)
    perm = rng.permutation(N)
    return [perm[i * g:(i + 1) * g].tolist() for i in range(N // g)]


# ============================================================ phase 1 (GPU, LATER) — M0
def run_phase1(args):
    """Compute the 200 edits, measure the 3 M0 merge regimes on the model, save vectors +
    per-regime post-merge logits, then run phase 2 and write the table. GPU / model here."""
    import torch
    import transformers as _tf
    t0 = time.time()
    device = args.device
    model, tok, layer, _nL = _load_edit_model(args.model, args.layer, device,
                                              model_dtype=args.model_dtype)
    edits = load_counterfact(args.data, args.n_edits, args.seed)
    N = len(edits)
    print(f"[m0] {N} edits (seed {args.seed})", flush=True)

    vectors, W, W_base = _compute_solo(model, tok, layer, device, edits, args.steps, args.lr, t0)
    K = vectors["K"]; d_out, d_in = int(W.shape[0]), int(W.shape[1])

    regimes = build_regimes(K.astype(float), vectors["key_norm"].astype(float), edits, N,
                            args.pair_pool, args.group_size, args.n_enriched)
    Rt, Ktsc = _merge_factors(vectors["K"], vectors["R"], vectors["denom"], device)

    obs_regime, obs_group, obs_edit = [], [], []
    obs_logit_post, obs_argmax_ok_post = [], []
    mem_regime, mem_group, mem_edit = [], [], []
    for rid, (name, groups) in enumerate(regimes):
        gid, ed, lp, ap = _measure_merged_groups(
            model, tok, device, W, W_base, Rt, Ktsc, vectors["target_tok"], edits, groups)
        for g, e2, l2, a2 in zip(gid, ed, lp, ap):
            obs_regime.append(rid); obs_group.append(g); obs_edit.append(e2)
            obs_logit_post.append(l2); obs_argmax_ok_post.append(a2)
            mem_regime.append(rid); mem_group.append(g); mem_edit.append(e2)
        print(f"[m0] regime '{name}' measured ({len(groups)} groups)  {time.time()-t0:.1f}s",
              flush=True)

    regimes_arr = dict(
        regime_names=np.array([n for n, _ in regimes], dtype="U32"),
        obs_regime=np.array(obs_regime, np.int32), obs_group=np.array(obs_group, np.int32),
        obs_edit=np.array(obs_edit, np.int32), obs_logit_post=np.array(obs_logit_post, np.float32),
        obs_argmax_ok_post=np.array(obs_argmax_ok_post, np.float32),
        mem_regime=np.array(mem_regime, np.int32), mem_group=np.array(mem_group, np.int32),
        mem_edit=np.array(mem_edit, np.int32),
    )
    meta = dict(
        schema_version=SCHEMA_VERSION, model=args.model, model_tag=_model_tag(args.model),
        dataset="counterfact", layer=layer, seed=args.seed, n_edits=N,
        steps=args.steps, lr=args.lr, device=device, d_in=d_in, d_out=d_out,
        pair_pool=args.pair_pool, group_size=args.group_size, n_enriched=args.n_enriched,
        regime_group_counts={n: len(g) for n, g in regimes},
        esr_solo=round(float(vectors["argmax_ok_solo"].mean()), 4),
        max_recon_rel_err=float(vectors["recon_rel_err"].max()),
        torch=torch.__version__, transformers=_tf.__version__, numpy=np.__version__,
    )
    run_dir = os.path.join(args.out_dir, f"{_model_tag(args.model)}_L{layer}_s{args.seed}")
    save_phase1(run_dir, vectors, regimes_arr, meta)
    print(f"[m0] saved phase-1 vectors -> {run_dir}", flush=True)

    table = analyze(vectors, regimes_arr, meta)
    _write_table(table, args.table_out or os.path.join(args.out_dir, "M0_killgate_table.json"))
    _write_table(table, os.path.join(run_dir, "M0_killgate_table.json"))
    print_table(table)
    print(f"[m0] wrote {args.table_out or os.path.join(args.out_dir, 'M0_killgate_table.json')}  "
          f"total {time.time()-t0:.1f}s", flush=True)


def _model_tag(model_path):
    return os.path.basename(os.path.normpath(model_path))


def _write_table(table, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(table, f, indent=2)
    os.replace(tmp, path)


def _savez_atomic(path, arrs):
    """np.savez_compressed with tmp+os.replace (path ends in .npz; killgate_keygeom.py:963)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **arrs)
    os.replace(tmp, path)


# ============================================================ RG operating curve (re-scoped gate)
PRE_REG_PASS_RULE = (
    "PASS iff partial rho(I_cos, drop | I_mag) >= partial_min (0.15) in >= 2 non-negligible, "
    "c2-passing (rho(I_cos,drop) >= rho_min) group sizes across >= 2 seeds; KILL if geometry's "
    "partial < partial_min everywhere testable OR geometry collapses under own-magnitude "
    "partialling (rho(I_cos,drop | ||k_a||,S_a) < partial_min) at every testable cell."
)


def save_rg(rg_dir, per_seed_vectors, measurements, meta):
    os.makedirs(rg_dir, exist_ok=True)
    for s, v in per_seed_vectors.items():
        _savez_atomic(os.path.join(rg_dir, f"rg_seed{int(s)}_vectors.npz"), v)
    _savez_atomic(os.path.join(rg_dir, "rg_measurements.npz"), measurements)
    tmp = os.path.join(rg_dir, "rg_meta.json.tmp")
    with open(tmp, "w") as f:
        json.dump(meta, f, indent=2)
    os.replace(tmp, os.path.join(rg_dir, "rg_meta.json"))


def load_rg(rg_dir):
    with open(os.path.join(rg_dir, "rg_meta.json")) as f:
        meta = json.load(f)
    seeds = [int(s) for s in meta["seeds"]]
    per_seed = {s: dict(np.load(os.path.join(rg_dir, f"rg_seed{s}_vectors.npz"))) for s in seeds}
    meas = dict(np.load(os.path.join(rg_dir, "rg_measurements.npz")))
    return per_seed, meas, meta


def analyze_rg(per_seed_vectors, measurements, meta, thr=None):
    """CPU operating-curve analysis: per (group_size g, seed) regime stats via _regime_stat with
    the fixed PARTIAL metric, then the pre-registered pass rule. Standalone-rerunnable."""
    t = _default_thresholds()
    if thr:
        t.update(thr)
    seeds = [int(s) for s in meta["seeds"]]
    gsizes = [int(g) for g in meta["group_sizes"]]
    obs_seed = measurements["obs_seed"].astype(int); obs_g = measurements["obs_g"].astype(int)
    obs_group = measurements["obs_group"].astype(int); obs_edit = measurements["obs_edit"].astype(int)
    obs_lp = measurements["obs_logit_post"].astype(float)
    obs_ap = measurements["obs_argmax_ok_post"].astype(float)
    mem_seed = measurements["mem_seed"].astype(int); mem_g = measurements["mem_g"].astype(int)
    mem_group = measurements["mem_group"].astype(int); mem_edit = measurements["mem_edit"].astype(int)

    members_all = defaultdict(list)
    for s, g, gr, ed in zip(mem_seed, mem_g, mem_group, mem_edit):
        members_all[(int(s), int(g), int(gr))].append(int(ed))

    cells = {}
    for s in seeds:
        v = per_seed_vectors[s]
        vecs = (v["K"].astype(float), v["R"].astype(float), v["denom"].astype(float),
                v["S"].astype(float), v["key_norm"].astype(float),
                v["logit_solo"].astype(float), v["argmax_ok_solo"].astype(float))
        for g in gsizes:
            sel = np.where((obs_seed == s) & (obs_g == g))[0]
            if sel.size == 0:
                continue
            members = {int(gr): members_all[(s, g, int(gr))] for gr in set(obs_group[sel].tolist())}
            cells[f"g{g}_s{s}"] = _regime_stat(
                vecs, obs_edit[sel], obs_group[sel], obs_lp[sel], obs_ap[sel], members, t)

    def _cell(g, s):
        return cells.get(f"g{g}_s{s}")

    def _geom_pass(st):
        p = st["partial_rho_geom"]
        return bool(st["c3_eligible"] and p is not None and p >= t["partial_min"]
                    and not st["collapses_under_ownmag_partial"])

    per_g = {}
    for g in gsizes:
        sc = {s: _cell(g, s) for s in seeds if _cell(g, s) is not None}
        pass_seeds = [s for s, st in sc.items() if _geom_pass(st)]
        per_g[str(g)] = {
            "n_seeds_measured": len(sc),
            "n_seeds_eligible": sum(1 for st in sc.values() if st["c3_eligible"]),
            "n_seeds_pass_geometry": len(pass_seeds),
            "pass_seeds": pass_seeds,
            "partial_by_seed": {str(s): st["partial_rho_geom"] for s, st in sc.items()},
            "eligible_by_seed": {str(s): st["c3_eligible"] for s, st in sc.items()},
            "argmax_loss_by_seed": {str(s): st["argmax_loss_rate"] for s, st in sc.items()},
            "qualifies": len(pass_seeds) >= 2,   # geometry passes in >= 2 seeds at this g
        }
    qualifying = [g for g in gsizes if per_g[str(g)]["qualifies"]]

    elig = [(g, s) for g in gsizes for s in seeds
            if _cell(g, s) is not None and _cell(g, s)["c3_eligible"]]
    any_geom_pass = any(_geom_pass(_cell(g, s)) for (g, s) in elig)
    all_collapse = bool(elig) and all(_cell(g, s)["collapses_under_ownmag_partial"] for (g, s) in elig)

    if not elig:
        overall = "INCONCLUSIVE"
    elif len(qualifying) >= 2:
        overall = "PASS"
    elif (not any_geom_pass) or all_collapse:
        overall = "KILL"
    else:
        overall = "MIXED"

    # scoped-federation boundary: largest g where damage is still partial (not saturated)
    boundary = {}
    for s in seeds:
        nonsat = [g for g in gsizes if _cell(g, s) is not None and not _cell(g, s)["saturated"]]
        boundary[str(s)] = (max(nonsat) if nonsat else None)
    bvals = [b for b in boundary.values() if b is not None]

    table = {
        "experiment": "merging_law_RG_operating_curve",
        "schema_version": SCHEMA_VERSION,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": meta.get("model"), "layer": meta.get("layer"),
        "seeds": seeds, "group_sizes": gsizes, "n_edits": meta.get("n_edits"),
        "thresholds": t,
        "pass_rule": PRE_REG_PASS_RULE,
        "geometry_metric": "partial_rho(I_cos, drop | I_mag)  [own-mag guard: rho(...|||k_a||,S_a)]",
        "cells": cells,
        "per_g_summary": per_g,
        "scoped_federation_boundary": {
            "per_seed_largest_nonsaturated_g": boundary,
            "conservative_min_across_seeds": (min(bvals) if bvals else None),
            "note": ("largest group size g where damage stays partial (argmax_loss < sat_argmax) "
                     "i.e. measurable/gradated — the scoped-federation limit"),
        },
        "verdict": {
            "overall": overall,
            "qualifying_group_sizes": qualifying,
            "n_qualifying_group_sizes": len(qualifying),
            "n_testable_cells": len(elig),
            "any_geometry_pass": any_geom_pass,
            "all_collapse_under_ownmag": all_collapse,
            "pass_rule": PRE_REG_PASS_RULE,
            "interpretation": {
                "PASS": ("geometry (partial rho) predicts merge interference at >= 2 group sizes "
                         "across >= 2 seeds -> scoped federation engineering-rule is real"),
                "KILL": ("geometry partial < partial_min everywhere testable (or collapses under "
                         "own-magnitude partialling) -> shelve the merging-law direction"),
                "MIXED": ("geometry passes somewhere but not at >= 2 group sizes x >= 2 seeds -> "
                          "under-powered / scope narrower"),
                "INCONCLUSIVE": ("no non-negligible, non-saturated, c2-coherent cell -> the curve "
                                 "cannot test geometry (all saturated or all negligible)"),
            }[overall],
        },
    }
    return table


def print_rg_table(table):
    v = table["verdict"]
    print("\n=== MERGING LAW RG OPERATING CURVE ===", flush=True)
    print(f"model={table['model']} layer={table['layer']} seeds={table['seeds']} "
          f"group_sizes={table['group_sizes']}", flush=True)
    for g in table["group_sizes"]:
        s = table["per_g_summary"][str(g)]
        print(f"  g={g:>3}: partial_by_seed={s['partial_by_seed']} "
              f"eligible={s['n_seeds_eligible']}/{s['n_seeds_measured']} "
              f"pass_geometry={s['n_seeds_pass_geometry']} qualifies={s['qualifies']}", flush=True)
    b = table["scoped_federation_boundary"]
    print(f"  scoped-federation boundary (largest non-saturated g): per_seed={b['per_seed_largest_nonsaturated_g']} "
          f"conservative={b['conservative_min_across_seeds']}", flush=True)
    print(f"  VERDICT overall={v['overall']} qualifying_g={v['qualifying_group_sizes']} "
          f"testable_cells={v['n_testable_cells']}", flush=True)
    print(f"  -> {v['interpretation']}", flush=True)


def run_phase_rg(args):
    """RG operating curve (GPU): group sizes x seeds at one layer; reuse existing per-seed solo
    vectors, compute fresh only where absent; tiled-group merges measured with the fixed metric.

    --device_map (R-C revision-wave addition, 2026-07-16): under multi-GPU sharding the edited
    layer's weight (W, and the Rt/Ktsc merge tensors built from it) can live on a DIFFERENT card
    than the input embedding, so two devices are tracked: `input_device` (tokenizer encodes,
    re-resolved via tp_edit_util.resolve_input_device — feeds _compute_solo/next_token_logits)
    and `layer_device` (Rt/Ktsc/gi construction — via tp_edit_util.resolve_layer_device, feeds
    _merge_factors/_measure_merged_groups). Single-GPU (--device_map none, the default) has
    input_device == layer_device == the old `device` string everywhere, so this is a
    byte-identical no-op for every existing invocation (main() also hard-fences --device_map to
    --rg mode only; run_phase1's regime-merge path is not made device_map-aware here)."""
    import torch
    import transformers as _tf
    from tp_edit_util import resolve_input_device, resolve_layer_device  # noqa: E402
    t0 = time.time()
    input_device = args.device
    model, tok, layer, _nL = _load_edit_model(args.model, args.layer, input_device,
                                              model_dtype=args.model_dtype,
                                              device_map=args.device_map)
    input_device = resolve_input_device(model, input_device)
    layer_device = resolve_layer_device(model, layer)
    seeds = [int(x) for x in str(args.rg_seeds).split(",") if x != ""]
    gsizes = [int(x) for x in str(args.rg_group_sizes).split(",") if x != ""]
    tag = _model_tag(args.model)
    rg_dir = os.path.join(args.out_dir, f"{tag}_L{layer}_RG")

    per_seed_vectors = {}
    reuse_note = {}
    obs_seed, obs_g, obs_group, obs_edit, obs_lp, obs_ap = [], [], [], [], [], []
    mem_seed, mem_g, mem_group, mem_edit = [], [], [], []
    for s in seeds:
        edits = load_counterfact(args.data, args.n_edits, s)
        N = len(edits)
        reuse_path = os.path.join(args.out_dir, f"{tag}_L{layer}_s{s}", "phase1_vectors.npz")
        if os.path.isfile(reuse_path):
            v = dict(np.load(reuse_path))
            W = model.model.layers[layer].mlp.down_proj.weight
            W_base = W.detach().clone()
            reuse_note[str(s)] = f"reused {reuse_path}"
            print(f"[rg] seed {s}: reused solo vectors {reuse_path}", flush=True)
        else:
            v, W, W_base = _compute_solo(model, tok, layer, input_device, edits, args.steps,
                                         args.lr, t0)
            _savez_atomic(os.path.join(args.out_dir, f"{tag}_L{layer}_s{s}", "phase1_vectors.npz"), v)
            reuse_note[str(s)] = "fresh 200-edit phase-1 (saved for future reuse)"
            print(f"[rg] seed {s}: computed fresh solo vectors", flush=True)
        per_seed_vectors[s] = v
        Rt, Ktsc = _merge_factors(v["K"], v["R"], v["denom"], layer_device)
        for g in gsizes:
            groups = _tiled_groups(N, g, s)
            gid, ed, lp, ap = _measure_merged_groups(
                model, tok, layer_device, W, W_base, Rt, Ktsc, v["target_tok"], edits, groups,
                input_device=input_device)
            for gg, e2, l2, a2 in zip(gid, ed, lp, ap):
                obs_seed.append(s); obs_g.append(g); obs_group.append(gg); obs_edit.append(e2)
                obs_lp.append(l2); obs_ap.append(a2)
                mem_seed.append(s); mem_g.append(g); mem_group.append(gg); mem_edit.append(e2)
            print(f"[rg] seed {s} g={g}: {len(groups)} groups measured  {time.time()-t0:.1f}s",
                  flush=True)

    measurements = dict(
        obs_seed=np.array(obs_seed, np.int32), obs_g=np.array(obs_g, np.int32),
        obs_group=np.array(obs_group, np.int32), obs_edit=np.array(obs_edit, np.int32),
        obs_logit_post=np.array(obs_lp, np.float32), obs_argmax_ok_post=np.array(obs_ap, np.float32),
        mem_seed=np.array(mem_seed, np.int32), mem_g=np.array(mem_g, np.int32),
        mem_group=np.array(mem_group, np.int32), mem_edit=np.array(mem_edit, np.int32),
    )
    meta = dict(
        schema_version=SCHEMA_VERSION, experiment="RG", model=args.model, model_tag=tag,
        dataset="counterfact", layer=layer, seeds=seeds, group_sizes=gsizes, n_edits=args.n_edits,
        steps=args.steps, lr=args.lr, device=str(input_device), layer_device=str(layer_device),
        model_dtype=args.model_dtype, device_map=args.device_map, reuse=reuse_note,
        esr_by_seed={str(s): round(float(per_seed_vectors[s]["argmax_ok_solo"].mean()), 4) for s in seeds},
        max_recon_rel_err_by_seed={str(s): (float(per_seed_vectors[s]["recon_rel_err"].max())
                                            if "recon_rel_err" in per_seed_vectors[s] else None)
                                   for s in seeds},
        torch=torch.__version__, transformers=_tf.__version__, numpy=np.__version__,
    )
    save_rg(rg_dir, per_seed_vectors, measurements, meta)
    print(f"[rg] saved RG bundle -> {rg_dir}", flush=True)
    table = analyze_rg(per_seed_vectors, measurements, meta)
    out = args.table_out or os.path.join(args.out_dir, "RG_operating_curve_table.json")
    _write_table(table, out)
    _write_table(table, os.path.join(rg_dir, "RG_operating_curve_table.json"))
    print_rg_table(table)
    print(f"[rg] wrote {out}  total {time.time()-t0:.1f}s", flush=True)


# ============================================================ self-test (CPU, no model)
def _assert_identity(rng, trials=200, d_in=17, d_out=11, tol=1e-9):
    """(a) ΔW_b @ k_a == r_b (k_b·k_a)/||k_b||^2 to fp64 tolerance on random rank-one edits."""
    worst = 0.0
    for _ in range(trials):
        k_a = rng.standard_normal(d_in)
        k_b = rng.standard_normal(d_in)
        r_b = rng.standard_normal(d_out)
        denom_b = float(k_b @ k_b)                 # exact ||k_b||^2 (no epsilon: pure identity)
        dW = np.outer(r_b, k_b) / denom_b          # [d_out, d_in]
        lhs = dW @ k_a
        rhs = r_b * (float(k_b @ k_a) / denom_b)
        # also exercise the helper that phase-2 actually uses
        rhs2 = cross_term(r_b, k_b, k_a, denom_b)
        err = max(float(np.max(np.abs(lhs - rhs))), float(np.max(np.abs(lhs - rhs2))))
        worst = max(worst, err)
        assert np.allclose(lhs, rhs, rtol=tol, atol=1e-12), f"identity broke: {err:.3e}"
        assert np.allclose(lhs, rhs2, rtol=tol, atol=1e-12), f"cross_term broke: {err:.3e}"
    return worst


def _crowded_keys(rng, N, d_in, frac_aligned=0.35):
    """Keys with BIMODAL crowding (Sigma_b |cos|) DECOUPLED from key-norm: ~frac_aligned of the
    edits align strongly with ONE shared direction (a high-mutual-cosine cluster), the rest stay
    near-orthogonal; then each key is rescaled to a random norm independent of alignment (cosine
    is scale-invariant, so crowding is preserved). The bimodal split makes the crowding term X_a
    VAR-DOMINATE the shared key-norm factor, so in the many-partner natural_group regime I_cos
    genuinely beats magnitude-only I_mag (~key-norm ranking) instead of tying it — the fair,
    non-vacuous test the criterion-3 primary-regime rule needs (verified robust over 6 seeds:
    strong group drho >~0.42, group_fails group drho <~ -0.42, both far from the 0.1 margin)."""
    base = 0.4 * rng.standard_normal((N, d_in))
    s = rng.standard_normal(d_in)
    s /= (np.linalg.norm(s) + 1e-12)
    aligned = rng.random(N) < frac_aligned
    alpha = np.where(aligned, rng.uniform(3.0, 7.0, N), rng.uniform(0.0, 0.2, N))
    base += alpha[:, None] * s[None, :]
    target_norm = rng.uniform(0.7, 1.4, N)           # real but modest spread, indep. of crowding
    return base / (np.linalg.norm(base, axis=1, keepdims=True) + 1e-12) * target_norm[:, None]


def _make_fixture(mode, rng, N=40, d_in=16, d_out=8,
                  pair_pool=10, group_size=40, n_enriched=30):
    """Synthesize the full phase-1 schema for a fixture. Modes:
      negligible  — tiny residuals, drop==0 everywhere  -> criterion 1 KILL (enriched negligible)
      strong      — crowded keys, drop ~ I_cos everywhere -> PASS (group drho clears >> drho_min)
      group_fails — crowded keys, drop ~ I_cos in pairwise/enriched but ~ I_mag in natural_group
                    -> KILL: proves the primary-regime rule bites (pairwise would PASS under an
                    any-clears rule, but the federation-at-scale regime decides c3 -> KILL).
    Returns (vectors, regimes_arr, meta). No model, no torch."""
    if mode == "negligible":
        K = rng.standard_normal((N, d_in)).astype(np.float64)
    else:
        K = _crowded_keys(rng, N, d_in).astype(np.float64)
    R = rng.standard_normal((N, d_out)).astype(np.float64)
    if mode == "negligible":
        R *= 0.02                                    # tiny residuals => tiny cross-terms
    key_norm = np.linalg.norm(K, axis=1)
    resid_norm = np.linalg.norm(R, axis=1)
    denom = key_norm ** 2 + 1e-8
    S = resid_norm / (key_norm + 1e-12)
    target_tok = np.arange(N, dtype=np.int64)
    logit_solo = (5.0 + rng.standard_normal(N)).astype(np.float64)
    argmax_ok_solo = np.ones(N, np.float64)          # every edit worked solo

    edits_meta = [{"prompt": f"probe {i} lives in", "subject": f"probe {i}"} for i in range(N)]
    regimes = build_regimes(K, key_norm, edits_meta, N, pair_pool, group_size, n_enriched)

    def _zs(x):
        x = np.asarray(x, float)
        sd = x.std()
        return (x - x.mean()) / sd if sd > 0 else x * 0.0

    def _resid_raw(y, z):
        Z = np.column_stack([np.ones_like(z), z])
        beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
        return y - Z @ beta

    # signal spec + target argmax-loss (saturation) per regime, chosen so the coherence rule is
    # exercised: strong keeps natural_group NON-saturated (0.4) so it can carry c3 via the
    # partial; group_fails SATURATES natural_group (0.95>sat_argmax) so it reads UNINTERPRETABLE;
    # hidden_geometry drives drop by I_mag + I_cos_perp so RAW delta~0 but the PARTIAL detects it.
    def _spec(md, name):
        if md == "strong":
            return "cos", (0.4 if name == "natural_group" else 0.2 if name == "natural_pairwise" else 0.3)
        if md == "group_fails":
            return ("mag" if name == "natural_group" else "cos"), (0.95 if name == "natural_group" else 0.2)
        if md == "hidden_geometry":
            return "hidden", (0.4 if name == "natural_group" else 0.25)
        return "cos", 0.2

    obs_regime, obs_group, obs_edit = [], [], []
    obs_logit_post, obs_argmax_ok_post = [], []
    mem_regime, mem_group, mem_edit = [], [], []
    for rid, (name, groups) in enumerate(regimes):
        # pass 1: collect per-obs aggregates for this regime
        rows = []                     # (gid, a, I_cos, I_mag)
        for gid, group in enumerate(groups):
            for a in group:
                others = [b for b in group if b != a]
                if others:
                    ob = np.array(others)
                    coses = np.array([abs(float(K[b] @ K[a]) /
                                          (key_norm[b] * key_norm[a] + 1e-12)) for b in others])
                    ic = float(key_norm[a] * np.sum(S[ob] * coses))
                    im = float(key_norm[a] * np.sum(S[ob]))
                else:
                    ic = im = 0.0
                rows.append((gid, a, ic, im))
        Ic = np.array([r[2] for r in rows]); Im = np.array([r[3] for r in rows])
        # pass 2: build drops from the regime's signal, then flip the top-`target` fraction argmax
        if mode == "negligible":
            drops = np.zeros(len(rows)); am = np.ones(len(rows))
        else:
            which, target = _spec(mode, name)
            if which == "cos":
                sig = _zs(Ic)
            elif which == "mag":
                sig = _zs(Im)
            else:  # hidden: magnitude drives most of the variance, geometry hides in the residual
                sig = 1.0 * _zs(Im) + 0.55 * _zs(_resid_raw(Ic, Im))
            drops = 0.7 * sig + 0.05 * rng.standard_normal(len(rows))
            am = np.ones(len(rows))
            if target > 0 and len(rows) > 0:
                k_flip = int(round(target * len(rows)))
                if k_flip > 0:
                    flip = np.argsort(-drops)[:k_flip]     # largest drops lose the argmax
                    am[flip] = 0.0
        for (gid, a, _ic, _im), dr, ao in zip(rows, drops, am):
            obs_regime.append(rid); obs_group.append(gid); obs_edit.append(a)
            obs_logit_post.append(logit_solo[a] - float(dr))
            obs_argmax_ok_post.append(float(ao))
            mem_regime.append(rid); mem_group.append(gid); mem_edit.append(a)

    vectors = dict(
        K=K.astype(np.float32), R=R.astype(np.float32), denom=denom.astype(np.float64),
        S=S.astype(np.float32), key_norm=key_norm.astype(np.float32),
        resid_norm=resid_norm.astype(np.float32), target_tok=target_tok,
        logit_solo=logit_solo.astype(np.float32), argmax_ok_solo=argmax_ok_solo.astype(np.float32),
        recon_rel_err=np.zeros(N, np.float32),
    )
    regimes_arr = dict(
        regime_names=np.array([n for n, _ in regimes], dtype="U32"),
        obs_regime=np.array(obs_regime, np.int32), obs_group=np.array(obs_group, np.int32),
        obs_edit=np.array(obs_edit, np.int32), obs_logit_post=np.array(obs_logit_post, np.float32),
        obs_argmax_ok_post=np.array(obs_argmax_ok_post, np.float32),
        mem_regime=np.array(mem_regime, np.int32), mem_group=np.array(mem_group, np.int32),
        mem_edit=np.array(mem_edit, np.int32),
    )
    meta = dict(schema_version=SCHEMA_VERSION, model=f"SYNTHETIC_{mode}", model_tag=f"synthetic_{mode}",
                dataset="synthetic", layer=-1, seed=0, n_edits=N, d_in=d_in, d_out=d_out,
                pair_pool=pair_pool, group_size=group_size, n_enriched=n_enriched)
    return vectors, regimes_arr, meta


def _make_rg_fixture(mode, seeds=(0, 1, 2), gsizes=(2, 3, 5, 10, 20, 50), N=60, d_in=16, d_out=8):
    """Synthesize an RG operating-curve bundle. mode='pass' drives drop by I_cos (geometry beyond
    magnitude) so small (non-saturated) g pass the partial in >=2 seeds -> PASS; mode='kill'
    drives drop by I_mag only so the partial is ~0 everywhere testable -> KILL. Large g are
    saturated (argmax_loss 0.9) to exercise the scoped-federation boundary. No model, no torch."""
    per_seed = {}
    O = defaultdict(list)   # collects obs_/mem_ columns
    for s in seeds:
        srng = np.random.default_rng(9000 + s + (0 if mode == "pass" else 500))
        if mode == "pass":
            K = _crowded_keys(srng, N, d_in).astype(np.float64)
        else:
            # kill: EXACTLY collinear keys (all same direction, varied norm) => every pairwise
            # |cos|==1 => I_cos == I_mag identically => the partial residualizes to 0 (NaN), so
            # magnitude-driven drop leaves NO geometry for the partial to find -> clean KILL.
            shared = srng.standard_normal(d_in); shared /= (np.linalg.norm(shared) + 1e-12)
            K = (shared[None, :] * srng.uniform(0.7, 1.4, N)[:, None]).astype(np.float64)
        R = srng.standard_normal((N, d_out)).astype(np.float64)
        key_norm = np.linalg.norm(K, axis=1); resid_norm = np.linalg.norm(R, axis=1)
        denom = key_norm ** 2 + 1e-8; S = resid_norm / (key_norm + 1e-12)
        logit_solo = 5.0 + srng.standard_normal(N)
        per_seed[s] = dict(
            K=K.astype(np.float32), R=R.astype(np.float32), denom=denom.astype(np.float64),
            S=S.astype(np.float32), key_norm=key_norm.astype(np.float32),
            resid_norm=resid_norm.astype(np.float32), target_tok=np.arange(N, dtype=np.int64),
            logit_solo=logit_solo.astype(np.float32), argmax_ok_solo=np.ones(N, np.float32),
            recon_rel_err=np.zeros(N, np.float32))
        for g in gsizes:
            groups = _tiled_groups(N, g, s)
            if not groups:
                continue
            rows = []
            for gid, group in enumerate(groups):
                for a in group:
                    others = [b for b in group if b != a]
                    ob = np.array(others)
                    coses = np.array([abs(float(K[b] @ K[a]) /
                                          (key_norm[b] * key_norm[a] + 1e-12)) for b in others])
                    rows.append((gid, a, float(key_norm[a] * np.sum(S[ob] * coses)),
                                 float(key_norm[a] * np.sum(S[ob]))))
            Ic = np.array([r[2] for r in rows]); Im = np.array([r[3] for r in rows])
            sig = (Ic if mode == "pass" else Im)
            sig = (sig - sig.mean()) / (sig.std() + 1e-12)
            drops = 0.7 * sig + 0.05 * srng.standard_normal(len(rows))
            target = 0.2 if g <= 20 else 0.9   # small g measurable; large g saturated (boundary)
            am = np.ones(len(rows)); kf = int(round(target * len(rows)))
            if kf > 0:
                am[np.argsort(-drops)[:kf]] = 0.0
            for (gid, a, _ic, _im), dr, ao in zip(rows, drops, am):
                O["obs_seed"].append(s); O["obs_g"].append(g); O["obs_group"].append(gid)
                O["obs_edit"].append(a); O["obs_logit_post"].append(float(logit_solo[a] - dr))
                O["obs_argmax_ok_post"].append(float(ao))
                O["mem_seed"].append(s); O["mem_g"].append(g); O["mem_group"].append(gid)
                O["mem_edit"].append(a)
    measurements = dict(
        obs_seed=np.array(O["obs_seed"], np.int32), obs_g=np.array(O["obs_g"], np.int32),
        obs_group=np.array(O["obs_group"], np.int32), obs_edit=np.array(O["obs_edit"], np.int32),
        obs_logit_post=np.array(O["obs_logit_post"], np.float32),
        obs_argmax_ok_post=np.array(O["obs_argmax_ok_post"], np.float32),
        mem_seed=np.array(O["mem_seed"], np.int32), mem_g=np.array(O["mem_g"], np.int32),
        mem_group=np.array(O["mem_group"], np.int32), mem_edit=np.array(O["mem_edit"], np.int32))
    meta = dict(schema_version=SCHEMA_VERSION, experiment="RG", model=f"SYNTHETIC_RG_{mode}",
                model_tag=f"synthetic_rg_{mode}", dataset="synthetic", layer=-1,
                seeds=list(seeds), group_sizes=list(gsizes), n_edits=N)
    return per_seed, measurements, meta


def selftest(selftest_dir):
    os.makedirs(selftest_dir, exist_ok=True)
    rng = np.random.default_rng(20260711)

    print("[selftest] (a) exact cross-term identity ΔW_b @ k_a == r_b(k_b·k_a)/||k_b||^2 ...",
          flush=True)
    worst = _assert_identity(rng)
    print(f"[selftest]   IDENTITY OK — worst fp64 abs err over 200 random edits = {worst:.3e}",
          flush=True)

    print("[selftest] (b) M0 phase-2 on synthetic fixtures (partial metric + coherence rule) ...",
          flush=True)
    results = {}
    for mode, expect_overall in [("negligible", "KILL"), ("strong", "PASS"),
                                 ("group_fails", "MIXED"), ("hidden_geometry", "PASS")]:
        vectors, regimes_arr, meta = _make_fixture(mode, rng)
        run_dir = os.path.join(selftest_dir, mode)
        save_phase1(run_dir, vectors, regimes_arr, meta)          # round-trips through disk
        v2, r2, m2 = load_phase1(run_dir)
        table = analyze(v2, r2, m2)
        _write_table(table, os.path.join(run_dir, "M0_killgate_table.json"))
        print_table(table)
        vd = table["verdict"]
        got = vd["overall"]
        results[mode] = vd
        assert got == expect_overall, \
            f"[selftest] {mode} fixture: overall verdict {got!r} != expected {expect_overall!r}"

        if mode == "negligible":
            assert vd["criterion_1_negligible_even_enriched"] == "KILL", \
                "[selftest] negligible: c1 should KILL (enriched negligible)"

        if mode == "strong":
            grp = table["regimes"]["natural_group"]
            assert not grp["saturated"] and grp["c3_eligible"], \
                "[selftest] strong: natural_group must be non-saturated and c3-eligible"
            assert vd["criterion_3_decided_on"] == "natural_group", \
                "[selftest] strong: c3 must be decided on natural_group (federation regime)"
            assert vd["criterion_3_geometry_partial"] == "PASS", \
                "[selftest] strong: c3 geometry-partial should PASS"
            assert (grp["partial_rho_geom"] or 0) >= 2 * DEF_PARTIAL_MIN, \
                (f"[selftest] strong: natural_group partial={grp['partial_rho_geom']} should clear "
                 f"2*partial_min={2*DEF_PARTIAL_MIN} (comfortable margin vs RNG)")

        if mode == "group_fails":
            grp = table["regimes"]["natural_group"]
            pw = table["regimes"]["natural_pairwise"]
            # COHERENCE RULE: the federation-at-scale regime is SATURATED, so it reads
            # UNINTERPRETABLE and does NOT carry c3 — and is NOT substituted by g=2 pairwise
            # (even though pairwise is eligible with a high partial).
            assert grp["saturated"] and not grp["c3_eligible"], \
                "[selftest] group_fails: natural_group must be saturated + NOT c3-eligible"
            assert vd["criterion_3_geometry_partial"] == "UNINTERPRETABLE", \
                "[selftest] group_fails: c3 should be UNINTERPRETABLE (saturated federation regime)"
            assert pw["c3_eligible"] and (pw["partial_rho_geom"] or 0) >= DEF_PARTIAL_MIN, \
                "[selftest] group_fails: pairwise SHOULD be eligible+strong (proves it was not rerouted)"

        if mode == "hidden_geometry":
            grp = table["regimes"]["natural_group"]
            # geometry is HIDDEN under shared magnitude: the raw delta MISSES it (< drho_min, so
            # the OLD metric would not credit geometry), but the PARTIAL detects it (> 0.3).
            assert vd["criterion_3_decided_on"] == "natural_group" \
                and vd["criterion_3_geometry_partial"] == "PASS", \
                "[selftest] hidden_geometry: c3 should PASS on natural_group via the partial"
            assert (grp["partial_rho_geom"] or 0) >= 0.3, \
                f"[selftest] hidden_geometry: partial={grp['partial_rho_geom']} should exceed 0.3"
            assert (grp["delta_rho_cos_minus_mag"] or 0) < DEF_DRHO_MIN, \
                (f"[selftest] hidden_geometry: raw delta={grp['delta_rho_cos_minus_mag']} should be "
                 f"< drho_min={DEF_DRHO_MIN} (raw metric MISSES what the partial catches)")
        print(f"[selftest]   {mode}: verdict={got} c3={vd['criterion_3_geometry_partial']} "
              f"(on={vd['criterion_3_decided_on']}) — as expected", flush=True)

    print("[selftest] (c) RG operating-curve on synthetic bundles (pre-registered pass rule) ...",
          flush=True)
    for mode, expect in [("pass", "PASS"), ("kill", "KILL")]:
        per_seed, meas, meta = _make_rg_fixture(mode)
        rg_dir = os.path.join(selftest_dir, f"rg_{mode}")
        save_rg(rg_dir, per_seed, meas, meta)                     # disk round-trip
        ps2, me2, mt2 = load_rg(rg_dir)
        rgt = analyze_rg(ps2, me2, mt2)
        _write_table(rgt, os.path.join(rg_dir, "RG_operating_curve_table.json"))
        print_rg_table(rgt)
        got = rgt["verdict"]["overall"]
        assert got == expect, f"[selftest] RG {mode}: overall {got!r} != expected {expect!r}"
        if mode == "pass":
            assert rgt["verdict"]["n_qualifying_group_sizes"] >= 2, \
                "[selftest] RG pass: need >= 2 qualifying group sizes"
            assert rgt["scoped_federation_boundary"]["conservative_min_across_seeds"] is not None, \
                "[selftest] RG pass: scoped-federation boundary should be recorded"
        if mode == "kill":
            assert not rgt["verdict"]["any_geometry_pass"], \
                "[selftest] RG kill: no cell should pass geometry"
        print(f"[selftest]   RG {mode}: verdict={got} "
              f"qualifying_g={rgt['verdict']['qualifying_group_sizes']} — as expected", flush=True)

    print("\n[selftest] ALL CHECKS PASSED (identity + M0 partial-metric/coherence "
          "[negligible->KILL, strong->PASS, group_fails->UNINTERPRETABLE/MIXED, "
          "hidden_geometry->PASS-via-partial] + RG pass/kill, disk round-trips)", flush=True)
    return results


# ============================================================ CLI
def main():
    ap = argparse.ArgumentParser(description="Merging law M0 kill-gate (spec 2.4).")
    ap.add_argument("--selftest", action="store_true",
                    help="CPU self-test: exact-identity assertion + full phase-2 pipeline on "
                         "synthetic fixtures (no model, no GPU). Writes only under --selftest_dir.")
    ap.add_argument("--selftest_dir",
                    default=os.path.join(HARNESS, "results", "merging", "selftest"))
    ap.add_argument("--phase2_dir", default=None,
                    help="STANDALONE CPU reanalysis: read a phase-1 run dir and (re)write the "
                         "kill-gate table from the saved vectors — no model, no GPU.")
    # RG operating-curve (re-scoped gate, ruling 2026-07-12)
    ap.add_argument("--rg", action="store_true",
                    help="RG operating curve (GPU): group sizes x seeds at one layer, partial metric.")
    ap.add_argument("--rg_phase2_dir", default=None,
                    help="STANDALONE CPU reanalysis of a saved RG bundle dir — no model, no GPU.")
    ap.add_argument("--rg_seeds", default="0,1,2", help="comma seeds for the RG curve")
    ap.add_argument("--rg_group_sizes", default="2,3,5,10,20,50,100",
                    help="comma group sizes g for the RG operating curve")
    # phase-1 (GPU) args
    ap.add_argument("--model", default=os.path.join(HARNESS, "data", "models", "Llama-3.2-1B"))
    ap.add_argument("--data", default=os.path.join(HARNESS, "data", "counterfact.json"))
    ap.add_argument("--n_edits", type=int, default=200)
    ap.add_argument("--layer", default="12")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--model_dtype", choices=["fp32", "bf16"], default="fp32",
                    help="model LOAD dtype. fp32 = default, byte-identical to the old hardcode. "
                         "bf16 halves weight RAM for big-model cells; ROME's value-optimization "
                         "math stays fp32 inside the editor regardless (editors/rome_native.py's "
                         "bf16-boundary casts) — only the frozen forward runs at model_dtype.")
    ap.add_argument("--device_map", choices=["none", "auto", "balanced", "balanced_low_0",
                                             "sequential"], default="none",
                    help="accelerate device_map for TENSOR-PARALLEL sharded loading across "
                         "multiple GPUs (e.g. a ~13B model on 2x24GB cards — R-C revision-wave "
                         "cell). 'none' (default) is byte-identical to the old single-.to(device) "
                         "load path. Wired ONLY for --rg mode (run_phase_rg); --device_map with "
                         "the plain phase-1 kill-gate path is refused below (its regime-merge "
                         "code assumes single-device tensors — see tp_edit_util.py).")
    ap.add_argument("--pair_pool", type=int, default=10,
                    help="natural_pairwise pool -> C(pool,2) pairwise merges (10 => 45)")
    ap.add_argument("--group_size", type=int, default=200,
                    help="natural_group single-merge size (spec '100+100'; assoc. => 200-sum)")
    ap.add_argument("--n_enriched", type=int, default=45,
                    help="number of conflict pairs in the enriched regime")
    ap.add_argument("--out_dir", default=os.path.join(HARNESS, "results", "merging"))
    ap.add_argument("--table_out", default=None,
                    help="kill-gate table path (default <out_dir>/M0_killgate_table.json)")
    # thresholds
    ap.add_argument("--neg_logit", type=float, default=DEF_NEG_LOGIT)
    ap.add_argument("--neg_argmax", type=float, default=DEF_NEG_ARGMAX)
    ap.add_argument("--rho_min", type=float, default=DEF_RHO_MIN)
    ap.add_argument("--drho_min", type=float, default=DEF_DRHO_MIN)
    ap.add_argument("--partial_min", type=float, default=DEF_PARTIAL_MIN)
    ap.add_argument("--sat_argmax", type=float, default=DEF_SAT_ARGMAX)
    args = ap.parse_args()

    if args.device_map != "none":
        if args.device != "cuda":
            raise SystemExit("[m0] --device_map requires --device cuda (accelerate device_map "
                             "places shards on CUDA devices)")
        if not args.rg:
            raise SystemExit("[m0] --device_map is wired only for --rg mode (run_phase_rg). "
                             "run_phase1's 3-regime kill-gate path (_merge_factors/"
                             "_measure_merged_groups) assumes single-device Rt/Ktsc/W tensors "
                             "and has not been made device_map-aware — pass --rg or use "
                             "--device_map none.")

    thr = dict(neg_logit=args.neg_logit, neg_argmax=args.neg_argmax, rho_min=args.rho_min,
               drho_min=args.drho_min, partial_min=args.partial_min, sat_argmax=args.sat_argmax)

    if args.selftest:
        selftest(args.selftest_dir)
        return
    if args.phase2_dir:
        v, r, meta = load_phase1(args.phase2_dir)
        table = analyze(v, r, meta, thr)
        out = args.table_out or os.path.join(args.phase2_dir, "M0_killgate_table.json")
        _write_table(table, out)
        print_table(table)
        print(f"[m0] phase-2 reanalysis wrote {out}", flush=True)
        return
    if args.rg_phase2_dir:
        per_seed, meas, meta = load_rg(args.rg_phase2_dir)
        table = analyze_rg(per_seed, meas, meta, thr)
        out = args.table_out or os.path.join(args.rg_phase2_dir, "RG_operating_curve_table.json")
        _write_table(table, out)
        print_rg_table(table)
        print(f"[rg] RG reanalysis wrote {out}", flush=True)
        return
    if args.rg:
        run_phase_rg(args)
        return
    run_phase1(args)


if __name__ == "__main__":
    main()
