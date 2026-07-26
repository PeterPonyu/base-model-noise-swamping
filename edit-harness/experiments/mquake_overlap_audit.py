"""mquake_overlap_audit.py — edit-vs-probe fact-overlap robustness check for the
MQuAKE / MQuAKE-T law-replication cells (2026-07-10).

load_mquake() splits shuffled records positionally and does NOT dedup edit facts
against probe facts on (subject, relation_id); single-hop facts recur across MQuAKE
cases, so ~4-5% (CF-3k) / ~9-10% (T) of probe columns collide with an edit row.
A colliding (edit, probe) pair is not "collateral" damage — the edit targets the
probe's own fact — so the within-probe rho could in principle be inflated.

This audit recomputes the canonical within-probe statistics with colliding pairs
EXCLUDED and reports the delta. It reconstructs the (subject, relation_id) identity
of every edit row / probe column by replaying load_mquake's deterministic shuffle
(same file, same numpy default_rng(seed) — verified to reproduce the split), because
the gate npz stores no identity fields.

Masking and statistics mirror mechanism_sc_table.py exactly (--known --edit_ok):
edit_ok rows, pre_p>0.05 cols, rho_C = mean per-probe Spearman(|cos|, damage_logit),
rho_SC = same with norm_growth*|cos|.

CPU-only. numpy on existing .npz + the raw dataset JSONs. No GPU / torch / downloads.
Usage:  python experiments/mquake_overlap_audit.py
"""
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from mechanism_sc_table import spearman  # noqa: E402  (canonical rank corr)

DATA_FILE = {"mquake": "data/mquake_cf3k.json", "mquaket": "data/mquake_t.json"}
# guard against a dataset-file swap silently breaking the shuffle replay
EXPECTED_N = {"mquake": 3000, "mquaket": 1868}
TAG_RE = re.compile(r"gate_([^_]+)_([^_]+)_(mquake|mquaket)_L(\d+)_s(\d+)\.npz$")


def replay_split(dataset, seed, n_edits=200, n_probes=500):
    """Replay load_mquake's shuffle/skip logic; return (subject, relation_id) lists."""
    raw = json.load(open(DATA_FILE[dataset]))
    assert len(raw) == EXPECTED_N[dataset], (
        f"{dataset}: {len(raw)} records != expected {EXPECTED_N[dataset]} — "
        "dataset file changed since the runs; shuffle replay is not trustworthy")
    data = list(raw)
    rng = np.random.default_rng(seed)
    rng.shuffle(data)
    recs = []
    for d in data:
        rrs = d.get("requested_rewrite") or []
        if not rrs:
            continue
        rr = rrs[0]
        try:
            subj = rr["subject"]
            _ = rr["prompt"]
            tn = rr["target_new"]
            tt = rr["target_true"]
            tn = tn["str"] if isinstance(tn, dict) else tn
            tt = tt["str"] if isinstance(tt, dict) else tt
        except Exception:
            continue
        recs.append((subj, rr.get("relation_id", "NA")))
        if len(recs) >= n_edits + n_probes:
            break
    assert len(recs) == n_edits + n_probes, f"only {len(recs)} usable records"
    return recs[:n_edits], recs[n_edits:]


def audit_cell(path):
    m = TAG_RE.search(os.path.basename(path))
    assert m is not None, f"unparseable cell filename: {path}"
    model, editor, dataset, layer, seed = (
        m.group(1), m.group(2), m.group(3), int(m.group(4)), int(m.group(5)))
    d = np.load(path, allow_pickle=True)
    COS = np.abs(d["COS"].astype(float))
    D = d["damage_logit"].astype(float)
    S = d["norm_growth"].astype(float)
    rows = d["edit_ok"].astype(float) > 0.5
    cols = d["pre_p"].astype(float) > 0.05
    if cols.sum() < 5:
        cols = np.ones(COS.shape[1], bool)

    edits, probes = replay_split(dataset, seed, *COS.shape)
    collide = np.zeros(COS.shape, bool)
    probe_key_to_cols = {}
    for j, key in enumerate(probes):
        probe_key_to_cols.setdefault(key, []).append(j)
    for i, key in enumerate(edits):
        for j in probe_key_to_cols.get(key, []):
            collide[i, j] = True

    def mean_within_probe(x_full, drop_collisions):
        vals = []
        for j in np.where(cols)[0]:
            keep = rows.copy()
            if drop_collisions:
                keep &= ~collide[:, j]
            if keep.sum() < 3:
                continue
            vals.append(spearman(x_full[keep, j], D[keep, j]))
        return float(np.nanmean(vals))

    SC = S[:, None] * COS
    out = {"model": model, "editor": editor, "dataset": dataset,
           "layer": layer, "seed": seed,
           "n_colliding_pairs": int(collide.sum()),
           # NB: colliding probes are counted among the KNOWN (pre_p>0.05) columns —
           # the population the rho statistics are computed over. Use n_known_probes
           # as the denominator for any "fraction affected" claim, NOT n_probes_total.
           "n_colliding_probes": int((collide.any(axis=0) & cols).sum()),
           "n_known_probes": int(cols.sum()),
           "n_probes_total": int(COS.shape[1]),
           "frac_known_probes_colliding": float(
               (collide.any(axis=0) & cols).sum() / max(1, cols.sum())),
           "rho_C": mean_within_probe(COS, False),
           "rho_C_nooverlap": mean_within_probe(COS, True),
           "rho_SC": mean_within_probe(SC, False),
           "rho_SC_nooverlap": mean_within_probe(SC, True)}
    return out


def main():
    paths = sorted(glob.glob("results/matrices/gate_llama1b_rome_mquake_L*_s*.npz")
                   + glob.glob("results/matrices/gate_llama1b_rome_mquaket_L*_s*.npz"))
    cells = [audit_cell(p) for p in paths]
    for c in cells:
        print(f"{c['dataset']:8s} L{c['layer']:<3d} s{c['seed']}  "
              f"pairs={c['n_colliding_pairs']:3d} probes={c['n_colliding_probes']:3d}  "
              f"rho_C {c['rho_C']:.4f} -> {c['rho_C_nooverlap']:.4f}  "
              f"rho_SC {c['rho_SC']:.4f} -> {c['rho_SC_nooverlap']:.4f}")
    with open("results/MQUAKE_overlap_audit.json", "w") as f:
        json.dump({"cells": cells,
                   "note": "within-probe rho with edit-vs-probe (subject,relation_id) "
                           "collisions excluded; split replayed from load_mquake's "
                           "deterministic shuffle; masking mirrors mechanism_sc_table "
                           "--known --edit_ok"}, f, indent=1)
    print("wrote results/MQUAKE_overlap_audit.json")


if __name__ == "__main__":
    main()
