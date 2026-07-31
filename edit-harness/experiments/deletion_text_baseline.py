"""deletion_text_baseline.py — PreUnlearn-style TEXT/SET-feature baseline for deletion collateral.

WHY THIS EXISTS (binding amendment from the G-D0 scoop check, 2026-07-26):
PreUnlearn (Su/Shah/Le, arXiv:2606.18473) predicts SET-PAIR aggregate collateral damage for
gradient unlearning from TEXT features (set size/diversity/compactness, cross-set centroid
distances over sentence embeddings). Our claim is model-internal, per-fact, editor-aware.
That differentiation is only *credible* if we show geometry adds something OVER text features
on the same facts. So this module computes text-only predictors with the SAME interface as the
geometry predictor, and reports the INCREMENTAL value of key-geometry over them.

If the increment is null, the honest outcome is a negative result ("for deletion edits,
key geometry adds nothing beyond textual similarity"), NOT a reframed positive.

DESIGN NOTES
- The binding baseline uses a locally cached sentence-transformer plus stable hashed lexical
  features. It never downloads at run time and fails closed if the semantic encoder is absent.
- Per-fact granularity (ours), not set-pair (theirs): each edit is one row, its "forget set"
  is that fact's prompt+subject, and the probe bank is the retain set.
- Incremental value = LOFO (leave-one-feature-out) R^2 drop and partial Spearman of the
  geometry feature given the text features, computed on HELD-OUT folds.

Usage:
  python experiments/deletion_text_baseline.py --smoke
  python experiments/deletion_text_baseline.py --del_npz <npz> --data <counterfact.json> --out <json>
  python experiments/deletion_text_baseline.py --phaseL --out <json>     # sweeps Phase L cells
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

HARNESS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEXT_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
N_HASH = 512
N_FOLDS = 5


# ---------------------------------------------------------------- text featurisation
def _tokens(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(s).lower())


def _stable_hash(tok: str) -> int:
    """Deterministic across processes. Python's builtin hash() is randomised per-process
    unless PYTHONHASHSEED is pinned, which would make the feature matrix (and therefore
    every reported number) irreproducible between runs."""
    h = 2166136261
    for ch in tok.encode("utf-8"):          # FNV-1a
        h = ((h ^ ch) * 16777619) & 0xFFFFFFFF
    return h


def _hashed_bow(texts: list[str], n_dim: int = N_HASH) -> np.ndarray:
    """Hashed bag-of-words with sublinear tf and L2 norm (a TF-IDF stand-in that needs no fit)."""
    X = np.zeros((len(texts), n_dim), dtype=np.float64)
    for i, t in enumerate(texts):
        for tok in _tokens(t):
            X[i, _stable_hash(tok) % n_dim] += 1.0
    X = np.log1p(X)
    # idf-ish: downweight dimensions present in most rows
    df = (X > 0).mean(0)
    X *= np.log(1.0 / np.maximum(df, 1e-6))
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(nrm, 1e-12)


def _sentence_vectors(texts: list[str], model_name: str) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("sentence-transformers is required for the binding text baseline") from exc
    model = SentenceTransformer(model_name, local_files_only=True)
    return np.asarray(model.encode(texts, normalize_embeddings=True, show_progress_bar=False),
                      dtype=np.float64)


def text_features(edit_texts: list[str], probe_texts: list[str],
                  encoder: str = TEXT_ENCODER) -> tuple[np.ndarray, list[str]]:
    """PreUnlearn-flavoured per-edit features against the probe (retain) bank.

    The binding space is a cached sentence encoder. Stable hashed lexical features remain
    as complementary covariates and make the baseline stronger, not as a silent fallback.
    """
    E = len(edit_texts)
    all_texts = edit_texts + probe_texts
    X_sem = _sentence_vectors(all_texts, encoder)
    X_lex = _hashed_bow(all_texts)
    Xe_sem, Xp_sem = X_sem[:E], X_sem[E:]
    Xe_lex, Xp_lex = X_lex[:E], X_lex[E:]

    feats, names = [], []
    for prefix, Xe, Xp in (("semantic", Xe_sem, Xp_sem), ("lexical", Xe_lex, Xp_lex)):
        cent = Xp.mean(0)
        cent /= max(np.linalg.norm(cent), 1e-12)
        sim = Xe @ Xp.T
        feats.append(sim.mean(1)); names.append(f"{prefix}_mean_sim_to_retain")
        feats.append(sim.max(1)); names.append(f"{prefix}_max_sim_to_retain")
        feats.append(np.percentile(sim, 90, axis=1)); names.append(f"{prefix}_p90_sim_to_retain")
        feats.append(sim.std(1)); names.append(f"{prefix}_std_sim_to_retain")
        feats.append(Xe @ cent); names.append(f"{prefix}_cos_to_retain_centroid")
        feats.append(np.linalg.norm(Xe - cent[None, :], axis=1)); names.append(
            f"{prefix}_dist_to_retain_centroid")

    feats.append(np.array([len(_tokens(t)) for t in edit_texts], float)); names.append("n_tokens")
    feats.append(np.array([len(set(_tokens(t))) / max(len(_tokens(t)), 1)
                           for t in edit_texts], float)); names.append("lexical_diversity")
    feats.append(np.array([len(str(t)) for t in edit_texts], float)); names.append("n_chars")
    pvocab = set()
    for t in probe_texts:
        pvocab.update(_tokens(t))
    feats.append(np.array([
        sum(tok in pvocab for tok in _tokens(t)) / max(len(_tokens(t)), 1)
        for t in edit_texts
    ], float)); names.append("retain_vocab_overlap")
    return np.column_stack(feats), names


# ---------------------------------------------------------------- stats
def _rank(a: np.ndarray) -> np.ndarray:
    o = np.argsort(a, kind="mergesort")
    r = np.empty(len(a), float)
    r[o] = np.arange(len(a), dtype=float)
    s = a[o]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            r[o[i:j + 1]] = r[o[i:j + 1]].mean()
        i = j + 1
    return r


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    ra, rb = _rank(a[m]), _rank(b[m])
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def partial_spearman(x: np.ndarray, y: np.ndarray, Z: np.ndarray,
                     n_folds: int = N_FOLDS, seed: int = 0) -> float:
    """Cross-fitted Spearman(x, y | Z); every residual is predicted out of fold."""
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(Z).all(1)
    if m.sum() < n_folds * 2:
        return float("nan")
    rx, ry = _rank(x[m]), _rank(y[m])
    RZ = np.column_stack([_rank(Z[m][:, j]) for j in range(Z.shape[1])])
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(len(rx)), n_folds)
    ex, ey = np.full(len(rx), np.nan), np.full(len(ry), np.nan)
    for f, te in enumerate(folds):
        tr = np.concatenate([folds[g] for g in range(n_folds) if g != f])
        Xtr = np.column_stack([np.ones(len(tr)), RZ[tr]])
        Xte = np.column_stack([np.ones(len(te)), RZ[te]])
        lam = 1e-6 * np.trace(Xtr.T @ Xtr) / Xtr.shape[1]
        A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1])
        ex[te] = rx[te] - Xte @ np.linalg.solve(A, Xtr.T @ rx[tr])
        ey[te] = ry[te] - Xte @ np.linalg.solve(A, Xtr.T @ ry[tr])
    return spearman(ex, ey)


def _kfold_r2(X: np.ndarray, y: np.ndarray, n_folds: int = N_FOLDS, seed: int = 0) -> float:
    """Held-out R^2 of ridge regression y ~ X (standardised, k-fold)."""
    n = len(y)
    if n < n_folds * 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    folds = np.array_split(idx, n_folds)
    preds = np.full(n, np.nan)
    for f in range(n_folds):
        te = folds[f]
        tr = np.concatenate([folds[g] for g in range(n_folds) if g != f])
        Xtr, Xte = X[tr], X[te]
        mu, sd = Xtr.mean(0), Xtr.std(0)
        sd = np.where(sd < 1e-12, 1.0, sd)
        Xtr = np.column_stack([np.ones(len(tr)), (Xtr - mu) / sd])
        Xte = np.column_stack([np.ones(len(te)), (Xte - mu) / sd])
        lam = 1.0
        A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1])
        b = np.linalg.solve(A, Xtr.T @ y[tr])
        preds[te] = Xte @ b
    ss_res = float(((y - preds) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def compare(text_F: np.ndarray, names: list[str], geom: np.ndarray,
            damage: np.ndarray, seed: int = 0) -> dict:
    """The decisive comparison: does geometry add anything OVER text features?"""
    ok = np.isfinite(damage) & np.isfinite(geom) & np.isfinite(text_F).all(1)
    F, g, y = text_F[ok], geom[ok], damage[ok]
    r2_text = _kfold_r2(F, y, seed=seed)
    r2_both = _kfold_r2(np.column_stack([F, g]), y, seed=seed)
    r2_geom = _kfold_r2(g.reshape(-1, 1), y, seed=seed)
    lofo = {}
    for j, nm in enumerate(names):
        keep = [k for k in range(F.shape[1]) if k != j]
        lofo[nm] = round(r2_both - _kfold_r2(np.column_stack([F[:, keep], g]), y, seed=seed), 4)
    return {
        "n": int(ok.sum()),
        "r2_text_only": round(r2_text, 4),
        "r2_geometry_only": round(r2_geom, 4),
        "r2_text_plus_geometry": round(r2_both, 4),
        "incremental_r2_of_geometry": round(r2_both - r2_text, 4),
        "lofo_drop_when_geometry_removed": round(r2_both - r2_text, 4),
        "partial_spearman_geometry_given_text": round(partial_spearman(g, y, F, seed=seed), 4),
        "spearman_geometry_marginal": round(spearman(g, y), 4),
        "lofo_per_text_feature": lofo,
        "VERDICT_RULE": ("geometry is load-bearing iff incremental_r2_of_geometry > 0 AND "
                         "partial_spearman_geometry_given_text is non-null with the same sign "
                         "as the marginal; otherwise report the honest negative"),
    }


# ---------------------------------------------------------------- data loading
def load_cell(del_npz: str, data_json: str | None) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    d = np.load(del_npz, allow_pickle=True)
    COS = d["COS"].astype(np.float64)             # [E, P]
    dmg = d["damage_logit"].astype(np.float64)    # [E, P]
    ok = d["edit_ok"].astype(bool) if "edit_ok" in d.files else np.ones(COS.shape[0], bool)
    if "resid_norm" not in d.files:
        raise ValueError("matrix has no resid_norm; cannot construct preregistered SxC geometry")
    S = d["resid_norm"].astype(np.float64)
    geom = S[ok] * np.nanmean(np.abs(COS[ok]), axis=1)  # per-edit |SxC| summary
    damage = np.nanmean(dmg[ok], axis=1)          # per-edit mean collateral
    edit_texts, probe_texts = [], []
    if data_json and os.path.exists(data_json):
        seed_match = re.search(r"_s(\d+)(?:\.npz)?$", os.path.basename(del_npz))
        if seed_match is None:
            raise ValueError("cannot recover selection seed from matrix filename")
        seed = int(seed_match.group(1))
        try:
            from experiments.killgate_keygeom import load_counterfact
        except ModuleNotFoundError:
            from killgate_keygeom import load_counterfact
        edits, probes, _ = load_counterfact(data_json, COS.shape[0], COS.shape[1], seed)
        if len(edits) != COS.shape[0] or len(probes) != COS.shape[1]:
            raise ValueError(
                f"loader returned {len(edits)} edits/{len(probes)} probes for matrix {COS.shape}"
            )
        edit_texts = [f"{r['prompt']} {r['subject']}" for r, keep in zip(edits, ok) if keep]
        probe_texts = [f"{r['prompt']} {r['subject']}" for r in probes]
    return geom, damage, edit_texts, probe_texts


# ---------------------------------------------------------------- smoke
def smoke() -> int:
    rng = np.random.default_rng(0)
    E, P = 120, 40
    subj = [f"entity{i%37}" for i in range(E)]
    edit_texts = [f"The capital of {s} is located in" for s in subj]
    probe_texts = [f"The capital of entity{i%23} is located in" for i in range(P)]
    F, names = text_features(edit_texts, probe_texts)
    geom = rng.random(E)
    # planted: damage driven by geometry PLUS a text feature, so both signals are real
    damage = 2.0 * geom + 1.0 * (F[:, 0] - F[:, 0].mean()) / max(F[:, 0].std(), 1e-9) \
        + 0.3 * rng.standard_normal(E)
    res = compare(F, names, geom, damage)
    print(json.dumps({k: v for k, v in res.items() if k != "lofo_per_text_feature"}, indent=2))
    ok = (res["n"] == E and np.isfinite(res["r2_text_only"])
          and res["incremental_r2_of_geometry"] > 0.05
          and res["partial_spearman_geometry_given_text"] > 0.3)
    print(f"[smoke] {'PASS' if ok else 'FAIL'}: planted geometry recovered over text features "
          f"(incr R2={res['incremental_r2_of_geometry']}, "
          f"partial rho={res['partial_spearman_geometry_given_text']})")
    # negative control: geometry pure noise, unrelated to damage -> increment must collapse
    damage2 = 1.0 * (F[:, 0] - F[:, 0].mean()) / max(F[:, 0].std(), 1e-9) + 0.3 * rng.standard_normal(E)
    res2 = compare(F, names, rng.random(E), damage2)
    ok2 = res2["incremental_r2_of_geometry"] < 0.05
    print(f"[smoke] {'PASS' if ok2 else 'FAIL'}: null control gives no increment "
          f"(incr R2={res2['incremental_r2_of_geometry']})")
    return 0 if (ok and ok2) else 1


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--del_npz")
    ap.add_argument("--data", default=os.path.join(HARNESS, "data", "counterfact.json"))
    ap.add_argument("--phaseL", action="store_true", help="sweep every Phase L deletion cell on disk")
    ap.add_argument("--out")
    args = ap.parse_args()

    if args.smoke:
        return smoke()

    if args.del_npz:
        cells = [args.del_npz]
    elif args.phaseL:
        cells = []
        for tag, layer in (("gemma2b", 13), ("phi35", 16), ("qwen3b", 18), ("qwen15b", 21)):
            cells.extend(glob.glob(os.path.join(
                HARNESS, f"results/matrices/u1e0_{tag}_delete_refusal_L{layer}_s*.npz")))
        cells = sorted(cells)
    else:
        cells = sorted(glob.glob(os.path.join(
            HARNESS, "results/matrices/u1e0_*_delete_refusal_L*_s*.npz")))
    if not cells:
        print("[textbase] no deletion npz found — run the deletion cells first"); return 2

    out = {}
    for c in cells:
        tag = os.path.basename(c).replace(".npz", "")
        try:
            geom, damage, et, pt = load_cell(c, args.data)
        except Exception as e:          # noqa: BLE001 — one bad cell must not sink the sweep
            out[tag] = {"error": str(e)}; print(f"[textbase] {tag}: ERROR {e}"); continue
        if not et or not pt:
            out[tag] = {"error": "no text available (dataset json missing) — geometry-only fallback",
                        "spearman_geometry_marginal": round(spearman(geom, damage), 4)}
            print(f"[textbase] {tag}: text unavailable, marginal rho only"); continue
        F, names = text_features(et, pt)
        n = min(len(F), len(geom), len(damage))
        out[tag] = compare(F[:n], names, geom[:n], damage[:n])
        print(f"[textbase] {tag}: incr_R2={out[tag]['incremental_r2_of_geometry']} "
              f"partial_rho={out[tag]['partial_spearman_geometry_given_text']}")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=2)
        print(f"[textbase] -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
