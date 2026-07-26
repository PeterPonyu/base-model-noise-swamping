"""lexical_sbert_baseline.py — G3 lexical + SBERT damage predictors.

Closes the reviewer's H2 rebuttal: "high key-cosine is just topical / lexical /
semantic similarity between the edit and the probe, so any old surface-similarity
predictor would work." We build two surface competitors on the edit↔probe text
pairs and score them on the SAME clean within-probe partialled Spearman metric as
the key-cosine headline:

  LEXICAL  : token-Jaccard overlap (whitespace+lowercase), CPU/numpy, no model.
  SBERT    : cosine of local sentence-embeddings (BAAI/bge-m3 default, fallback
             Alibaba-NLP/gte-Qwen2-1.5B-instruct). LOCAL HF CACHE ONLY — the import
             AND the model load are guarded and run with HF_HUB_OFFLINE=1; if
             unavailable the SBERT arm degrades gracefully (reported as null) and
             the lexical arm still runs. No download is ever attempted.

Row/col alignment: the .npz rows are edits and columns are probes in the order
produced by killgate's load_counterfact / load_zsre. We REIMPLEMENT those loaders
VERBATIM here (same json.load → default_rng(seed).shuffle → same requested_rewrite
parsing → same n_edits/n_probes slicing), so LEX[N,M]/SBERT[N,M] align 1:1 with the
npz COS/damage. The --seed/--n_edits/--n_probes MUST match the run that made the npz.

METRIC DISCIPLINE (project rule): primary = SIGNED within-probe partialled
Spearman(predictor, damage); AUROC is NOT used. Always also report MEAN signed
damage. Degenerate constant columns → Spearman NaN, which is COUNTED and reported,
never silently dropped.

Usage:
  python experiments/lexical_sbert_baseline.py \
      results/matrices/gate_llama1b_rome_cf_L8_s0.npz \
      --dataset counterfact --data data/counterfact.json \
      --n_edits 200 --n_probes 500 --seed 0 \
      --field prompt --metric logit --known --edit_ok \
      --out results/H2_lexsbert_L8_s0.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_matrices import spearman, within_probe_rhos  # noqa: E402

# local SBERT candidates present in the HF cache (checked at runtime, offline)
SBERT_DEFAULT = "BAAI/bge-m3"
SBERT_FALLBACK = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"


# ---- loaders REIMPLEMENTED VERBATIM from killgate_keygeom.py (order must match npz) ----
def load_counterfact(path, n_edits, n_probes, seed=0):
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
        if len(recs) >= n_edits + n_probes:
            break
    edits = recs[:n_edits]
    probes = recs[n_edits:n_edits + n_probes]
    return edits, probes


def load_zsre(path, n_edits, n_probes, seed=0):
    data = json.load(open(path))
    rng = np.random.default_rng(seed)
    rng.shuffle(data)
    recs = []
    for d in data:
        s, p, alt, pred = d.get("subject"), d.get("src"), d.get("alt"), d.get("pred")
        if not (s and p and alt and pred):
            continue
        recs.append({"subject": s, "prompt": p, "target_new": alt, "target_true": pred})
        if len(recs) >= n_edits + n_probes:
            break
    return recs[:n_edits], recs[n_edits:n_edits + n_probes]


# ---- lexical predictor ----
def _toks(s):
    return set(s.lower().split())


def jaccard_matrix(edit_texts, probe_texts):
    """LEX[i,j] = |A∩B|/|A∪B| token-Jaccard. Higher overlap ⇒ more predicted damage
    (sign matches COS). Empty∪empty ⇒ 0.0."""
    et = [_toks(t) for t in edit_texts]
    pt = [_toks(t) for t in probe_texts]
    LEX = np.zeros((len(et), len(pt)), float)
    for i, a in enumerate(et):
        for j, b in enumerate(pt):
            u = len(a | b)
            LEX[i, j] = (len(a & b) / u) if u else 0.0
    return LEX


# ---- SBERT predictor (guarded local-only) ----
def sbert_matrix(edit_texts, probe_texts, model_name, fallback):
    """Return (SBERT[N,M] cosine, info). Never downloads; guards import + load.
    On any failure returns (None, {...error...}) so the lexical arm still reports."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    try:
        from sentence_transformers import SentenceTransformer  # guarded import
    except Exception as e:  # noqa: BLE001
        return None, {"sbert": "SKIPPED", "reason": f"import failed: {type(e).__name__}: {e}"}

    last_err = None
    for name in [model_name, fallback]:
        if name is None:
            continue
        try:
            # device left to library default; CPU is fine for these short strings.
            m = SentenceTransformer(name, trust_remote_code=True)
            Ee = m.encode(edit_texts, normalize_embeddings=True, show_progress_bar=False)
            Ep = m.encode(probe_texts, normalize_embeddings=True, show_progress_bar=False)
            SB = np.asarray(Ee, float) @ np.asarray(Ep, float).T   # cosine (already L2-normed)
            return SB, {"sbert": "OK", "model": name}
        except Exception as e:  # noqa: BLE001
            last_err = f"{name}: {type(e).__name__}: {e}"
            continue
    return None, {"sbert": "SKIPPED", "reason": f"local load failed: {last_err}"}


def _apply_masks(d, known, edit_ok_filter, N, M):
    rows = np.ones(N, bool)
    cols = np.ones(M, bool)
    if edit_ok_filter and "edit_ok" in d.files:
        rows = d["edit_ok"].astype(float) > 0.5
    if known and "pre_p" in d.files:
        c = d["pre_p"].astype(float) > 0.05
        if c.sum() >= 5:
            cols = c
    return rows, cols


def _wp(pred, D):
    rhos = within_probe_rhos(pred, D)
    n_nan = int(np.sum(~np.isfinite(rhos)))
    return {
        "within_probe_mean": (None if np.all(np.isnan(rhos))
                              else round(float(np.nanmean(rhos)), 4)),
        "within_probe_frac_positive": (None if np.all(np.isnan(rhos))
                                       else round(float(np.nanmean(rhos > 0)), 3)),
        "nan_column_count": n_nan,
        "n_columns": int(pred.shape[1]),
    }


def main():
    ap = argparse.ArgumentParser(description="G3 lexical + SBERT surface-similarity baselines")
    ap.add_argument("npz", help="the killgate .npz whose rows(edits)/cols(probes) to align to")
    ap.add_argument("--dataset", choices=["counterfact", "zsre"], default="counterfact")
    ap.add_argument("--data", required=True)
    ap.add_argument("--n_edits", type=int, default=200)
    ap.add_argument("--n_probes", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0, help="MUST match the seed that produced the npz")
    ap.add_argument("--field", choices=["subject", "prompt"], default="prompt")
    ap.add_argument("--metric", choices=["logit", "prob"], default="logit")
    ap.add_argument("--known", action="store_true")
    ap.add_argument("--edit_ok", action="store_true")
    ap.add_argument("--sbert_model", default=SBERT_DEFAULT)
    ap.add_argument("--sbert_fallback", default=SBERT_FALLBACK)
    ap.add_argument("--no_sbert", action="store_true", help="skip the SBERT arm entirely")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # 1) re-derive edits/probes in the SAME order as the npz
    load_fn = load_counterfact if args.dataset == "counterfact" else load_zsre
    edits, probes = load_fn(args.data, args.n_edits, args.n_probes, args.seed)

    d = np.load(args.npz)
    COS = d["COS"].astype(float)
    D = (d["damage_logit"] if args.metric == "logit" else d["damage_prob"]).astype(float)
    N, M = COS.shape
    if len(edits) != N or len(probes) != M:
        raise SystemExit(
            f"ALIGNMENT MISMATCH: re-derived {len(edits)} edits x {len(probes)} probes "
            f"but npz is {N} x {M}. Check --seed/--n_edits/--n_probes/--dataset/--data "
            f"match the run that produced {os.path.basename(args.npz)}.")

    edit_texts = [e[args.field] for e in edits]
    probe_texts = [p[args.field] for p in probes]

    # 2) build predictors on the FULL matrix, then mask identically to COS/D
    LEX = jaccard_matrix(edit_texts, probe_texts)
    SB, sbert_info = (None, {"sbert": "DISABLED (--no_sbert)"}) if args.no_sbert else \
        sbert_matrix(edit_texts, probe_texts, args.sbert_model, args.sbert_fallback)

    rows, cols = _apply_masks(d, args.known, args.edit_ok, N, M)
    ix = np.ix_(rows, cols)
    COSm, Dm, LEXm = COS[ix], D[ix], LEX[ix]
    SBm = SB[ix] if SB is not None else None

    res = {
        "npz": os.path.basename(args.npz),
        "dataset": args.dataset, "field": args.field, "metric": args.metric,
        "seed": args.seed, "n_edits": args.n_edits, "n_probes": args.n_probes,
        "filters": {"known": args.known, "edit_ok": args.edit_ok},
        "shape_after_masks": [int(COSm.shape[0]), int(COSm.shape[1])],
        "mean_signed_damage": round(float(np.nanmean(Dm)), 5),   # project rule
        "keycos": _wp(COSm, Dm),
        "lexical": _wp(LEXm, Dm),
        "sbert_info": sbert_info,
        "sbert": (_wp(SBm, Dm) if SBm is not None else None),
    }
    # H2 verdict: key-cosine must beat lexical (and SBERT when available)
    km = res["keycos"]["within_probe_mean"]
    lm = res["lexical"]["within_probe_mean"]
    sm = res["sbert"]["within_probe_mean"] if res["sbert"] else None
    beats_lex = None if (km is None or lm is None) else bool(abs(km) > abs(lm))
    beats_sbert = None if (km is None or sm is None) else bool(abs(km) > abs(sm))
    res["verdict_detail"] = {"keycos_beats_lexical": beats_lex,
                             "keycos_beats_sbert": beats_sbert}
    if beats_lex and (beats_sbert is None or beats_sbert):
        res["VERDICT"] = "key-cosine BEATS surface similarity (H2 rebuttal closed)"
    elif beats_lex is None:
        res["VERDICT"] = "UNDETERMINED"
    else:
        res["VERDICT"] = "surface similarity MATCHES/BEATS key-cosine — H2 concern stands"

    print(json.dumps(res, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(res, open(args.out, "w"), indent=2)
        print(f"[lexsbert] wrote {args.out}")


if __name__ == "__main__":
    main()
