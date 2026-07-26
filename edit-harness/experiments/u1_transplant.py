"""u1_transplant.py — U1-E1 PreUnlearn-style transplanted-feature baseline (CODE ONLY tonight).

Transplant extension of lexical_sbert_baseline.py. The PreUnlearn-style
(arXiv 2606.18473) feature set needs NO target-model forward pass — that is the
whole point of the transplant ("data-only"): if a purely data/lexical/embedding
feature set predicts per-probe collateral as well as the model-internal S×C
statistic, then S×C's forward-pass cost is not worth it. U1-E1 is therefore the
GENUINELY FALSIFIABLE gate:

    KILL/demote U1 if the data-only transplant >= S×C  (Δρ = ρ(S×C) − ρ(transplant) <= 0).

TONIGHT'S SCOPE (per spec): CODE + CPU smoke-validation on the existing INSERTION
npz gate_llama1b_rome_cf_L12_s0.npz. This validates loader row/col alignment and
NaN accounting ONLY — smoke numbers are NOT a U1 result. U1-E0's deletion npz
(refusal-target, L12 s0, 200x500) is a 40-GPU-min run scheduled LATER and is NOT
run tonight; when it lands, this module scores it into
results/U1_E1_transplant_del_L12_s0.json.

HARD REQUIREMENTS carried over verbatim from lexical_sbert_baseline.py:
  (1) load_counterfact/load_zsre reimplemented VERBATIM (same json.load ->
      default_rng(seed).shuffle -> requested_rewrite parsing -> n_edits/n_probes
      slicing) so feature matrices align 1:1 with npz rows/cols. --seed/--n_edits/
      --n_probes MUST match the run that made the npz.
  (2) HF_HUB_OFFLINE=1; SBERT from local cache only (BAAI/bge-m3, fallback
      Alibaba-NLP/gte-Qwen2-1.5B-instruct); graceful null degradation, no download.
  (3) metric = SIGNED within-probe partialled Spearman (imported from
      analyze_matrices); AUROC banned; mean signed damage always reported;
      NaN/degenerate-constant columns COUNTED and reported, never silently dropped.

CPU only. 0 GPU tonight. 0 downloads.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_matrices import _midrank, spearman, within_probe_rhos  # noqa: E402

SBERT_DEFAULT = "BAAI/bge-m3"
SBERT_FALLBACK = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
TINY_LLAMA = "HuggingFaceM4/tiny-random-LlamaForCausalLM"  # cached; CPU env-sanity only


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
    return recs[:n_edits], recs[n_edits:n_edits + n_probes]


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


# ---------- data-only PreUnlearn-style features (no target-model forward pass) ----------
def _toks(s):
    return set(str(s).lower().split())


def _corpus_token_freq(path, dataset):
    """Unigram frequency of target strings across the WHOLE corpus (data-only)."""
    data = json.load(open(path))
    freq = {}
    for d in data:
        if dataset == "counterfact":
            rr = d.get("requested_rewrite", d)
            tt = rr.get("target_true", {})
            tt = tt["str"] if isinstance(tt, dict) else tt
        else:
            tt = d.get("pred")
        if tt:
            for w in str(tt).lower().split():
                freq[w] = freq.get(w, 0) + 1
    return freq


def _relation(prompt, subject):
    """Prompt with the subject string removed = crude relation template."""
    return str(prompt).lower().replace(str(subject).lower(), " ").strip()


def build_transplant_features(edits, probes, corpus_freq):
    """Return dict name->[N,M] feature matrix, all data-only.

    Higher feature value => more predicted damage (sign aligned with COS)."""
    N, M = len(edits), len(probes)
    e_tok = [_toks(e["prompt"]) for e in edits]
    p_tok = [_toks(p["prompt"]) for p in probes]
    e_subj = [_toks(e["subject"]) for e in edits]
    p_subj = [_toks(p["subject"]) for p in probes]
    e_rel = [_toks(_relation(e["prompt"], e["subject"])) for e in edits]
    p_rel = [_toks(_relation(p["prompt"], p["subject"])) for p in probes]
    p_len = np.array([len(t) for t in p_tok], float)
    # per-probe target rarity: lower corpus freq -> rarer -> (hypothesized) more fragile
    def freq_of(s):
        ws = str(s).lower().split()
        return float(np.mean([corpus_freq.get(w, 0) for w in ws])) if ws else 0.0
    p_targ_rarity = np.array([1.0 / (1.0 + freq_of(p["target_true"])) for p in probes], float)
    e_targ_tokens = [_toks(e["target_new"]) for e in edits]

    JAC = np.zeros((N, M)); SUBJ = np.zeros((N, M)); REL = np.zeros((N, M)); CONTAIN = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            u = len(e_tok[i] | p_tok[j]); JAC[i, j] = len(e_tok[i] & p_tok[j]) / u if u else 0.0
            us = len(e_subj[i] | p_subj[j]); SUBJ[i, j] = len(e_subj[i] & p_subj[j]) / us if us else 0.0
            ur = len(e_rel[i] | p_rel[j]); REL[i, j] = len(e_rel[i] & p_rel[j]) / ur if ur else 0.0
            CONTAIN[i, j] = 1.0 if (e_targ_tokens[i] & p_tok[j]) else 0.0
    PLEN = np.repeat(p_len[None, :], N, axis=0)
    PRAR = np.repeat(p_targ_rarity[None, :], N, axis=0)
    return {"jaccard": JAC, "subj_overlap": SUBJ, "rel_overlap": REL,
            "target_contain": CONTAIN, "probe_len": PLEN, "probe_target_rarity": PRAR}


def _sha16(arr):
    """sha256 (first 16 hex) of a float64-contiguous array — mirrors cp_edit/kg0._hash."""
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(arr, dtype=np.float64).tobytes()).hexdigest()[:16]


# pooling modes that MUST hold for a model to be usable (fix pass 2026-07-02:
# the pin is now ENFORCED, not merely recorded)
EXPECTED_POOLING = {"BAAI/bge-m3": "cls"}


def _resolve_st_snapshot(model_name):
    """Resolve MODEL_NAME to an explicit local HF-cache snapshot directory that
    contains modules.json (a real sentence-transformers config). Loading by bare
    repo name under HF_HUB_OFFLINE can resolve a snapshot WITHOUT modules.json,
    in which case SentenceTransformer silently fabricates a mean-pooling head
    ('Creating a new one with mean pooling') — for bge-m3 (canonical pooling =
    CLS) that changes the embeddings and the U1 gate metric. Returns
    (snapshot_path_or_None, note)."""
    cache = os.environ.get("HF_HUB_CACHE",
                           os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub"))
    repo_dir = os.path.join(cache, "models--" + model_name.replace("/", "--"))
    snap_root = os.path.join(repo_dir, "snapshots")
    if not os.path.isdir(snap_root):
        return None, f"no local cache dir for {model_name}"
    # prefer the snapshot refs/main points to, then any other snapshot
    candidates = []
    ref_main = os.path.join(repo_dir, "refs", "main")
    if os.path.isfile(ref_main):
        rev = open(ref_main).read().strip()
        candidates.append(os.path.join(snap_root, rev))
    for s in sorted(os.listdir(snap_root)):
        p = os.path.join(snap_root, s)
        if p not in candidates:
            candidates.append(p)
    for p in candidates:
        if os.path.isfile(os.path.join(p, "modules.json")):
            return p, "snapshot with modules.json (real ST config incl. pooling head)"
    return None, (f"no cached snapshot of {model_name} contains modules.json — "
                  "ST would fabricate a mean-pooling head; refusing to load")


def sbert_matrix(edit_texts, probe_texts, model_name, fallback, strict=False):
    """SBERT cosine feature — GATE-INTEGRITY PIN, ENFORCED (fix pass 2026-07-02).

    delta_rho_SxC_minus_transplant is the U1 KILL/demote metric and swings ~0.03
    with SBERT numerics, so the execution environment is FORCED (device=cpu,
    float32, TF32 off) and RECORDED in the JSON (versions, device, dtype, TF32
    flags, pooling mode, snapshot path, sha256 of both embedding matrices).

    POOLING PIN (review finding, 2026-07-02): loading by bare repo name under
    HF_HUB_OFFLINE can resolve a cached snapshot lacking modules.json, making
    SentenceTransformer silently fabricate a MEAN-pooling head for bge-m3
    (canonical pooling = CLS). Two enforcement steps replace the old
    record-only behavior:
      (1) the model is loaded from an EXPLICIT snapshot path that contains
          modules.json (_resolve_st_snapshot); bare-name loading is never used;
      (2) after load, the REALIZED pooling_mode is checked against
          EXPECTED_POOLING; a mismatch is a hard failure for that model —
          never silently proceed with the wrong pooling. With strict=True
          (gate runs) an unusable SBERT raises SystemExit; with strict=False
          (smoke) it degrades to sbert=SKIPPED with the reason recorded and
          NO sbert_cos feature.
    """
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    try:
        import torch
        import sentence_transformers
        from sentence_transformers import SentenceTransformer
    except Exception as e:  # noqa: BLE001
        info = {"sbert": "SKIPPED", "reason": f"import failed: {type(e).__name__}: {e}"}
        if strict:
            raise SystemExit(f"[u1] SBERT pin violated on a gate run: {info['reason']}")
        return None, info
    # pin numerics BEFORE any encode
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    failures = []
    for name in [model_name, fallback]:
        if name is None:
            continue
        snap, snap_note = _resolve_st_snapshot(name)
        if snap is None:
            failures.append(f"{name}: {snap_note}")
            continue
        try:
            m = SentenceTransformer(snap, trust_remote_code=True, device="cpu")
            m = m.to(torch.float32)
            p = next(m.parameters())
            # assert-and-log the pin (gate run must be cpu/float32)
            assert p.device.type == "cpu" and p.dtype == torch.float32, (
                f"SBERT env not pinned: device={p.device}, dtype={p.dtype}")
            pooling = "unknown"
            try:
                for mod in m:
                    if hasattr(mod, "get_pooling_mode_str"):
                        pooling = mod.get_pooling_mode_str()
            except Exception:  # noqa: BLE001
                pass
            expected = EXPECTED_POOLING.get(name)
            if expected is not None and pooling != expected:
                # ENFORCED pin: wrong pooling is a hard failure for this model,
                # never a silently different feature.
                failures.append(f"{name}: realized pooling_mode={pooling!r} != "
                                f"required {expected!r} (pin ENFORCED)")
                continue
            Ee = np.asarray(m.encode(edit_texts, normalize_embeddings=True,
                                     show_progress_bar=False), float)
            Ep = np.asarray(m.encode(probe_texts, normalize_embeddings=True,
                                     show_progress_bar=False), float)
            info = {
                "sbert": "OK", "model": name,
                "env_pin": {
                    "device": str(p.device), "model_dtype": str(p.dtype),
                    "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
                    "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
                    "torch_version": torch.__version__,
                    "sentence_transformers_version": sentence_transformers.__version__,
                    "numpy_version": np.__version__,
                    "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"),
                    "snapshot_path": snap,
                    "snapshot_revision": os.path.basename(snap),
                    "pooling_mode": pooling,
                    "pooling_pin": (f"ENFORCED: pooling_mode must be {expected!r} "
                                    "(load hard-fails otherwise)" if expected is not None
                                    else "no pin registered for this model"),
                    "emb_sha256_edits": _sha16(Ee),
                    "emb_sha256_probes": _sha16(Ep),
                    "emb_shapes": [list(Ee.shape), list(Ep.shape)],
                },
            }
            return Ee @ Ep.T, info
        except Exception as e:  # noqa: BLE001
            failures.append(f"{name}: {type(e).__name__}: {e}")
    reason = "; ".join(failures) if failures else "no model name given"
    if strict:
        raise SystemExit(f"[u1] SBERT pin violated on a gate run — refusing to score "
                         f"the U1 gate without the pinned SBERT feature: {reason}")
    return None, {"sbert": "SKIPPED", "reason": reason}


def _colrank(M):
    """Within-probe (down-column) MIDRANK-normalize each feature to (0,1).

    FIX (review finding, 2026-07-02): the previous implementation used a stable
    argsort with NO tie handling, so ties were broken by ROW INDEX — for the
    mostly-zero sparse transplant features (target_contain, subj_overlap) this
    replaced the feature with row-order noise (measured: target_contain
    within-probe rho -0.134 raw -> +0.009 after the old _colrank), and turned
    per-column-CONSTANT features into exactly row_index/(n+1). Midranks
    (analyze_matrices._midrank, the same tie-averaged ranks the project's
    spearman uses) preserve tie structure; constant columns map to a constant."""
    R = np.empty_like(M, float)
    n = M.shape[0]
    for j in range(M.shape[1]):
        R[:, j] = _midrank(M[:, j]) / (n + 1.0)
    return R


def _within_col_constant(M):
    """True iff the feature is constant DOWN EVERY column (across edits) — such a
    feature is definitionally NaN under the within-probe metric and must not enter
    the combiner (it contributes only a per-column constant, i.e. nothing, or —
    under a broken ranker — row-index noise)."""
    return bool(np.all(np.nanmax(M, axis=0) == np.nanmin(M, axis=0)))


def cross_fitted_rank_combo(features, D, K=8, seed=0):
    """Cross-fitted linear rank combination across EDITS (rows). For each of K folds,
    learn per-feature sign+weight from the within-probe correlation of the feature's
    column-midranks with damage on the TRAIN edits, then apply to the held-out edits.
    Features constant within every column are EXCLUDED (review finding 2026-07-02).
    Returns (COMBINED[N,M], used_names, excluded_names). NaN-safe."""
    names = [k for k in features if not _within_col_constant(features[k])]
    excluded = [k for k in features if k not in names]
    Rs = {k: _colrank(features[k]) for k in names}
    N, M = D.shape
    rng = np.random.default_rng(seed)
    folds = rng.integers(0, K, N)
    COMB = np.zeros((N, M))
    for f in range(K):
        te = np.where(folds == f)[0]
        tr = np.where(folds != f)[0]
        if len(tr) < 5 or len(te) == 0:
            te = np.arange(N); tr = np.arange(N)
        w = {}
        for k in names:
            # weight = mean within-probe Spearman(feature-rank, damage) over train edits
            rho = within_probe_rhos(Rs[k][tr, :], D[tr, :])
            w[k] = float(np.nanmean(rho)) if np.isfinite(np.nanmean(rho)) else 0.0
        for k in names:
            COMB[te, :] += w[k] * Rs[k][te, :]
    return COMB, names, excluded


def _wp(pred, D):
    rhos = within_probe_rhos(pred, D)
    n_nan = int(np.sum(~np.isfinite(rhos)))
    return {"within_probe_mean": (None if np.all(np.isnan(rhos)) else round(float(np.nanmean(rhos)), 4)),
            "within_probe_frac_positive": (None if np.all(np.isnan(rhos)) else round(float(np.nanmean(rhos > 0)), 3)),
            "nan_column_count": n_nan, "n_columns": int(pred.shape[1])}


def _tiny_llama_cpu_sanity():
    """ENV SANITY ONLY (not used by the data-only transplant): confirm the cached
    tiny-random-Llama loads + forwards on CPU, offline. Guarded; failure is reported."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(TINY_LLAMA)
        model = AutoModelForCausalLM.from_pretrained(TINY_LLAMA, dtype=torch.float32).to("cpu").eval()
        with torch.no_grad():
            ids = tok("Paris is the capital of", return_tensors="pt")
            out = model(**ids)
        return {"status": "OK", "logits_shape": list(out.logits.shape), "device": "cpu"}
    except Exception as e:  # noqa: BLE001
        return {"status": "SKIPPED", "reason": f"{type(e).__name__}: {e}"}


def main():
    ap = argparse.ArgumentParser(description="U1-E1 transplanted PreUnlearn-style baseline")
    ap.add_argument("--npz", default=None, help="npz whose rows(edits)/cols(probes) to align to")
    ap.add_argument("--dataset", choices=["counterfact", "zsre"], default="counterfact")
    ap.add_argument("--data", default=None)
    ap.add_argument("--n_edits", type=int, default=200)
    ap.add_argument("--n_probes", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--metric", choices=["logit", "prob"], default="logit")
    ap.add_argument("--known", action="store_true")
    ap.add_argument("--edit_ok", action="store_true")
    ap.add_argument("--K", type=int, default=8, help="cross-fit folds across edits")
    ap.add_argument("--no_sbert", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="CPU smoke on the INSERTION npz; marks output smoke=true, NOT a U1 result")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    HARNESS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if args.smoke:
        args.npz = args.npz or os.path.join(HARNESS, "results", "matrices",
                                            "gate_llama1b_rome_cf_L12_s0.npz")
        args.data = args.data or os.path.join(HARNESS, "data", "counterfact.json")
        args.known = True; args.edit_ok = True; args.seed = 0
        args.out = args.out or os.path.join(HARNESS, "results", "U1_E1_transplant_SMOKE_L12_s0.json")
    if not (args.npz and args.data):
        raise SystemExit("need --npz and --data (or --smoke)")

    load_fn = load_counterfact if args.dataset == "counterfact" else load_zsre
    edits, probes = load_fn(args.data, args.n_edits, args.n_probes, args.seed)

    d = np.load(args.npz)
    COS = d["COS"].astype(float)
    D = (d["damage_logit"] if args.metric == "logit" else d["damage_prob"]).astype(float)
    N, M = COS.shape
    if len(edits) != N or len(probes) != M:
        raise SystemExit(f"ALIGNMENT MISMATCH: re-derived {len(edits)}x{len(probes)} but npz is {N}x{M}. "
                         f"Check --seed/--n_edits/--n_probes/--dataset/--data.")

    corpus_freq = _corpus_token_freq(args.data, args.dataset)
    feats = build_transplant_features(edits, probes, corpus_freq)
    # strict pin on gate runs: an unusable SBERT must hard-fail, not silently
    # weaken the transplant baseline (which would bias the gate toward U1 surviving)
    SB, sbert_info = (None, {"sbert": "DISABLED (--no_sbert)"}) if args.no_sbert else \
        sbert_matrix([e["prompt"] for e in edits], [p["prompt"] for p in probes],
                     SBERT_DEFAULT, SBERT_FALLBACK, strict=not args.smoke)
    if SB is not None:
        feats["sbert_cos"] = SB

    # masks (same as C1/C4)
    rows = np.ones(N, bool); cols = np.ones(M, bool)
    if args.edit_ok and "edit_ok" in d.files:
        rows = d["edit_ok"].astype(float) > 0.5
    if args.known and "pre_p" in d.files:
        c = d["pre_p"].astype(float) > 0.05
        if c.sum() >= 5:
            cols = c
    ix = np.ix_(rows, cols)
    COSm, Dm = COS[ix], D[ix]
    feats_m = {k: v[ix] for k, v in feats.items()}

    # cross-fitted combined transplant predictor (within-column-constant features excluded)
    COMB, comb_used, comb_excluded = cross_fitted_rank_combo(feats_m, Dm, K=args.K, seed=args.seed)

    # S x C reference (mechanism_sc_table convention): S=resid_norm, C=|cos|
    S_row = d["resid_norm"].astype(float)[rows] if "resid_norm" in d.files else None
    absC = np.abs(COSm)
    if S_row is not None:
        SC = S_row[:, None] * absC
        rho_SxC = float(np.nanmean(within_probe_rhos(SC, Dm)))
    else:
        rho_SxC = None

    res = {
        "smoke": bool(args.smoke),
        "IS_U1_RESULT": (False if args.smoke else True),
        "note": ("SMOKE on INSERTION npz — validates loader alignment + NaN accounting ONLY; "
                 "NOT a U1 result. U1 needs U1-E0's DELETION (refusal-target) npz."
                 if args.smoke else "scored on provided npz"),
        "npz": os.path.basename(args.npz), "dataset": args.dataset, "metric": args.metric,
        "seed": args.seed, "n_edits": args.n_edits, "n_probes": args.n_probes,
        "filters": {"known": args.known, "edit_ok": args.edit_ok},
        "shape_after_masks": [int(COSm.shape[0]), int(COSm.shape[1])],
        "mean_signed_damage": round(float(np.nanmean(Dm)), 5),
        "keycos": _wp(absC, Dm),
        "SxC_within_probe_mean": (round(rho_SxC, 4) if rho_SxC is not None else None),
        "transplant_features": {k: _wp(feats_m[k], Dm) for k in feats_m},
        "transplant_combined": _wp(COMB, Dm),
        "combiner_features_used": comb_used,
        "combiner_features_excluded_within_col_constant": comb_excluded,
        "sbert_info": sbert_info,
        "cross_fit_K": args.K,
        "env_pin_note": ("FIX PASS 2026-07-02: (1) SBERT pooling pin is now ENFORCED — "
                         "model loaded from an explicit cached snapshot containing "
                         "modules.json, realized pooling_mode checked against "
                         "EXPECTED_POOLING (bge-m3 -> 'cls'); mismatch hard-fails gate "
                         "runs / degrades to sbert=SKIPPED on smoke. Environment "
                         "(cpu/float32/TF32-off), snapshot path and sha256 of both "
                         "embedding matrices recorded in sbert_info.env_pin. "
                         "(2) _colrank now uses tie-averaged MIDRANKS "
                         "(analyze_matrices._midrank); the previous stable-argsort "
                         "ranking broke ties by row index, gutting sparse features in "
                         "the combiner; within-column-constant features are excluded "
                         "from the combiner. This artifact was regenerated from the "
                         "checked-in code after both fixes."),
    }
    # best SINGLE transplant feature (finite rho only), so the gate is not
    # hostage to the combiner (review finding 2026-07-02)
    singles = {k: res["transplant_features"][k]["within_probe_mean"]
               for k in res["transplant_features"]
               if res["transplant_features"][k]["within_probe_mean"] is not None}
    if singles:
        best_name = max(singles, key=lambda k: singles[k])
        res["transplant_best_single"] = {"feature": best_name,
                                         "within_probe_mean": singles[best_name]}
    rho_tr = res["transplant_combined"]["within_probe_mean"]
    if rho_SxC is not None and rho_tr is not None:
        delta = round(rho_SxC - rho_tr, 4)
        res["delta_rho_SxC_minus_transplant"] = delta
        # HOSTILE transplant score = max(combined, best single feature); the
        # pre-registered KILL/demote gate is evaluated against this
        best_tr = max([rho_tr] + list(singles.values())) if singles else rho_tr
        delta_best = round(rho_SxC - best_tr, 4)
        res["transplant_best_of_combined_and_singles"] = round(float(best_tr), 4)
        res["delta_rho_SxC_minus_best_transplant"] = delta_best
        res["U1_E1_gate_note"] = (
            "SMOKE — gate NOT scored on insertion data; the falsifiable gate KILL/demote "
            "(Δρ<=0) applies only to U1-E0's DELETION npz." if args.smoke else
            ("KILL/DEMOTE U1 — data-only transplant >= S×C (Δρ<=0 vs best transplant)"
             if delta_best <= 0
             else "U1 survives — S×C beats data-only transplant (Δρ>0 vs best of "
                  "combined and single features)"))
    if args.smoke:
        res["tiny_llama_cpu_env_sanity"] = _tiny_llama_cpu_sanity()

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        tmp = args.out + ".tmp"; json.dump(res, open(tmp, "w"), indent=2); os.replace(tmp, args.out)
        print(f"[u1] wrote {args.out}")
    print(json.dumps({k: res[k] for k in ("smoke", "IS_U1_RESULT", "shape_after_masks",
                                          "keycos", "SxC_within_probe_mean",
                                          "transplant_combined",
                                          "delta_rho_SxC_minus_transplant")
                      if k in res}, indent=2))


if __name__ == "__main__":
    main()
