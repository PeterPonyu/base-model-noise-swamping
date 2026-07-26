#!/usr/bin/env python3
"""Permutation null for the pre-registered RG federation gate (referee ask, D2 paper).

Question: with the gate's quorum rule (partial rho(I_cos, drop | I_mag) >= partial_min
in >= 2 non-negligible, c2-coherent group sizes, in >= 2 of 3 seeds), what is the
false-positive rate under the null of NO geometry-damage association?

Null construction: within every (g, seed) cell of the reference bundle, permute the
per-observation drop vector against the (I_cos, I_mag) pairs. Negligibility and
saturation flags are permutation-invariant (they depend only on the drop distribution
and argmax outcomes, not the pairing), so each replicate re-evaluates exactly the
data-dependent parts of the rule: c2 coherence (rho(I_cos, drop) >= rho_min) and the
partial threshold.

Outputs, per bundle:
  - observed per-(g,s) partials (asserted against the stored operating-curve table)
  - per-(g,s) one-sided permutation p for the observed partial
  - full-gate false-positive rate over N_PERM joint replicates

CPU-only; reads the frozen bundle npz files, writes results/merging/RG_permutation_null_<date>.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from merging_m0 import (_spearman, partial_spearman_multi,  # noqa: E402
                        DEF_RHO_MIN, DEF_PARTIAL_MIN, DEF_NEG_LOGIT, DEF_NEG_ARGMAX,
                        DEF_SAT_ARGMAX)


def load_cell_arrays(bundle_dir, seed, g):
    """Reproduce _regime_stat's I_cos/I_mag/drop/flags for one (g, seed) cell."""
    z = np.load(bundle_dir / "rg_measurements.npz")
    v = np.load(bundle_dir / f"rg_seed{seed}_vectors.npz")
    K, S = v["K"], v["S"]
    # SCHEMA GUARD: m0.v1 single-layer bundles only — editors.v1 multi-layer bundles
    # (MEMIT/AlphaEdit) store K as (n_edits, n_layers, hidden) and would silently
    # mis-shape the geometry below (verifier finding 2026-07-16). For those, use the
    # per-obs arrays in rg_measurements.npz (obs_drop etc.) directly.
    if K.ndim != 2:
        raise ValueError(f"{bundle_dir}: K shape {K.shape} — editors.v1 multi-layer "
                         "bundle; load_cell_arrays supports m0.v1 single-layer only")
    key_norm, logit_solo, argmax_ok_solo = v["key_norm"], v["logit_solo"], v["argmax_ok_solo"]

    msel = (z["mem_seed"] == seed) & (z["mem_g"] == g)
    members = {}
    for grp, ed in zip(z["mem_group"][msel], z["mem_edit"][msel]):
        members.setdefault(int(grp), []).append(int(ed))

    osel = (z["obs_seed"] == seed) & (z["obs_g"] == g)
    I_cos, I_mag, drop, argmax_loss, worked = [], [], [], [], []
    for a, grp, lp, ap in zip(z["obs_edit"][osel], z["obs_group"][osel],
                              z["obs_logit_post"][osel], z["obs_argmax_ok_post"][osel]):
        a, grp = int(a), int(grp)
        others = [b for b in members[grp] if b != a]
        if others:
            ob = np.array(others)
            coses = np.array([abs(float(np.dot(K[b], K[a])) /
                                  (key_norm[b] * key_norm[a] + 1e-12)) for b in others])
            ic = float(key_norm[a] * np.sum(S[ob] * coses))
            im = float(key_norm[a] * np.sum(S[ob]))
        else:
            ic = im = 0.0
        I_cos.append(ic)
        I_mag.append(im)
        drop.append(float(logit_solo[a] - lp))
        ws = argmax_ok_solo[a] > 0.5
        worked.append(bool(ws))
        argmax_loss.append(bool(ws and ap < 0.5))
    I_cos, I_mag, drop = map(np.array, (I_cos, I_mag, drop))
    argmax_loss = np.array(argmax_loss, bool)
    worked = np.array(worked, bool)

    med_abs_drop = float(np.median(np.abs(drop)))
    loss_rate = float(argmax_loss.sum() / max(worked.sum(), 1))
    non_negligible = (med_abs_drop >= DEF_NEG_LOGIT) or (loss_rate >= DEF_NEG_ARGMAX)
    saturated = loss_rate > DEF_SAT_ARGMAX
    return I_cos, I_mag, drop, non_negligible, saturated


def gate_fires(qual_by_seed):
    """>= 2 qualifying group sizes in >= 2 seeds."""
    return sum(1 for q in qual_by_seed.values() if q >= 2) >= 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="results/merging/Llama-3.2-1B_L12_RG")
    ap.add_argument("--gsizes", type=int, nargs="+", default=[2, 3, 5, 10, 20])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n_perm", type=int, default=2000)
    ap.add_argument("--out", default="results/merging/RG_permutation_null_20260716.json")
    args = ap.parse_args()

    bundle = Path(args.bundle)
    stored = json.load(open(bundle / "RG_operating_curve_table.json"))["cells"]

    cells = {}
    for s in args.seeds:
        for g in args.gsizes:
            I_cos, I_mag, drop, non_neg, sat = load_cell_arrays(bundle, s, g)
            obs_partial = partial_spearman_multi(I_cos, drop, [I_mag])
            key = f"g{g}_s{s}"
            ref = stored[key]["partial_rho_geom"]
            if abs(obs_partial - ref) > 5e-3:
                raise AssertionError(f"{key}: recomputed partial {obs_partial:.4f} != stored {ref:.4f}")
            cells[key] = dict(I_cos=I_cos, I_mag=I_mag, drop=drop,
                              non_negligible=non_neg, saturated=sat,
                              obs_partial=float(obs_partial),
                              obs_rho_cos=float(_spearman(I_cos, drop)))

    rng = np.random.default_rng(20260716)
    perm_exceed = {k: 0 for k in cells}
    gate_hits = 0
    for _ in range(args.n_perm):
        qual_by_seed = {s: 0 for s in args.seeds}
        for s in args.seeds:
            for g in args.gsizes:
                key = f"g{g}_s{s}"
                c = cells[key]
                pd = c["drop"][rng.permutation(len(c["drop"]))]
                partial = partial_spearman_multi(c["I_cos"], pd, [c["I_mag"]])
                if partial >= c["obs_partial"]:
                    perm_exceed[key] += 1
                c2 = _spearman(c["I_cos"], pd) >= DEF_RHO_MIN
                if (c["non_negligible"] and not c["saturated"] and c2
                        and partial >= DEF_PARTIAL_MIN):
                    qual_by_seed[s] += 1
        if gate_fires(qual_by_seed):
            gate_hits += 1

    out = dict(
        experiment="RG_permutation_null",
        bundle=str(bundle),
        n_perm=args.n_perm,
        thresholds=dict(rho_min=DEF_RHO_MIN, partial_min=DEF_PARTIAL_MIN,
                        neg_logit=DEF_NEG_LOGIT, neg_argmax=DEF_NEG_ARGMAX,
                        sat_argmax=DEF_SAT_ARGMAX),
        gate_rule=">=2 qualifying g in >=2 seeds; qualify = non-negligible & !saturated & c2 & partial>=partial_min",
        gate_false_positive_rate=gate_hits / args.n_perm,
        gate_hits=gate_hits,
        cells={k: dict(obs_partial=v["obs_partial"], obs_rho_cos=v["obs_rho_cos"],
                       non_negligible=v["non_negligible"], saturated=v["saturated"],
                       perm_p_one_sided=(perm_exceed[k] + 1) / (args.n_perm + 1))
               for k, v in cells.items()},
    )
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"gate FPR = {gate_hits}/{args.n_perm} = {out['gate_false_positive_rate']:.4f}")
    for k, v in out["cells"].items():
        print(f"  {k}: obs_partial={v['obs_partial']:+.3f} perm_p={v['perm_p_one_sided']:.4f} "
              f"nonneg={v['non_negligible']} sat={v['saturated']}")


if __name__ == "__main__":
    main()
