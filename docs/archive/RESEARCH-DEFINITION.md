# Research Definition — clarifying "what we are going to do" (post-workflow-verification)

> Source: 8-agent workflow (6-way literature reconnaissance + code build + synthesis), real web/HF retrieval, 2026-06-30.
> Answers to the three questions: code infrastructure = built and independently verified✅ · peer papers = retrieved✅ · what to do = locked in✅

---

## 1. Literature reconnaissance conclusion: novelty ranking of the 6 branches (honest version)

| Rank | Branch | Novelty | 5090 feasibility | Chosen as first paper? |
|---|---|---|---|---|
| **1** | **B6 mechanism: falsifiable "per-fact edit-damage predictor"** | **contested (descriptive side is crowded, but a PREDICTIVE predictor is a genuine gap)** | **very high** (frozen 1-7B + rank-one editing, harness already ready) | ✅ **Yes** |
| 2 | B4 routing: RL/bandit learns "edit site" | contested (nobody has done reward-driven site selection) | medium-high | No (better suited to a second paper) |
| 3 | B2 multi-hop: decontaminated metrics + pre-edit ripple prediction | contested (descriptive side saturated) | high | No (crowded track: RippleCOT/CaKE/K-Edit…) |
| 4 | B5 multimodal VLM editing | contested (UniKE's cross-modal 92% vs 18.5% gap is quite new) | medium (would need to build a VLM harness from scratch) | No (no harness leverage, scoop risk) |
| 5 | B3 lifelong editing / stopping criteria | **crowded** (already claimed by NAS/RLSEdit/SPHERE/CrispEdit) | high | No (easy but crowded = poor first choice) |
| 6 | B1 general methods/benchmarks | **crowded** (AlphaEdit + dominated by large labs) | low (a single GPU can't beat AlphaEdit) | No (most crowded) |

**Key correction**: my previous default assumption that **B1 multi-hop was judged crowded**—RippleEdits/MQuAKE are already mature, and a single GPU has no advantage going head-to-head. The workflow re-judged the first paper as **B6 mechanism prediction**, with solid reasoning.

---

## 2. First paper (locked in)

### Title
**Does Key Geometry Predict Collateral Damage? An A-Priori, Per-Fact Locality-Damage Forecaster for Knowledge Editing**

### Research question
Given a single parametric edit (ROME/MEMIT/AlphaEdit) to a frozen LM, can we predict **before** the edit is applied **which** unrelated facts will be broken—based on "the geometric overlap (cosine/Gram) between the edit subject's key and the probe fact's key at the edited layer"? And is this key-overlap signal **better** than ENCORE's "matrix norm-growth" signal, and better than Hase's "causal-tracing layer match"?

### Falsifiable hypotheses
The drop in a probe fact's locality after editing is monotonically predicted by the **pre-edit** `cosine(k_edit, k_probe)` (same layer):
- **H1**: Spearman ρ(key-cosine, damage) ≥ 0.4 and broken-vs-preserved AUROC ≥ 0.75 (CounterFact)
- **H2**: the key-overlap predictor **beats** the norm-growth predictor + lexical/embedding-similarity baselines
- **H3**: AlphaEdit's null-space projection **specifically and disproportionately** reduces damage for high-overlap probes (causal validation of the geometric explanation)

### Why it's novel (benchmarked against real SOTA)
- Goes beyond **Knowledge in Superposition (2408.07413)**: it proves interference **exists** but does not predict **which specific** facts break.
- Goes beyond **ENCORE (2502.01636)**: it attributes damage to matrix norm-growth with no per-target prediction; this paper is the **first** to put norm-growth vs key-overlap in **direct head-to-head competition** (reconnaissance explicitly flagged this experiment as missing).
- Goes beyond **Hase 2023 (NeurIPS)**: tests whether the locus of the effect is key-covariance overlap rather than the causal-tracing-identified layer, and **explains why AlphaEdit (ICLR 2025) works**—reconnaissance flagged this reconciliation as an "open & testable" question.
- **No surveyed paper provides a falsifiable, a-priori, per-edit, per-fact damage predictor.**

### Baselines / data / metrics
- **Baselines**: ENCORE norm-growth · lexical + sentence-embedding similarity · Hase causal-tracing layer match (negative control) · GradSim (EMNLP'24) · random floor
- **Data**: CounterFact (primary) · zsRE (generalization) · ~500-1000 unrelated probe-fact pool · RippleEdits (optional extension)
- **Metrics**: per-fact Δprob/Δlogit · Spearman/Pearson · broken-vs-preserved AUROC/AUPRC · predictor head-to-head ΔAUROC (bootstrap CI) · H3 causal test

### Kill-gate (2-3 days)
Run the cleanest setup first: **Llama-3.2-1B (or GPT-J-6B) × ROME × CounterFact, ~200 edits × ~500 probes**.
**If** key-cosine is essentially unrelated to per-fact damage—`|ρ|<0.2` and AUROC<0.6 and it fails to beat norm-growth—**then the central hypothesis is dead; pivot or stop.** Soft warning: if key-overlap works but is indistinguishable from norm-growth → collapse to "unifying the two explanations," still publishable but needs reframing.

### First experiment (pure forward pass + rank-one, no training, reusing already-verified code)
Frozen Llama-3.2-1B fp16, using `rome_native.py` + `metrics.py`: (1) load ~200 CounterFact edits + ~500 probes; (2) forward-hook to capture `k_edit` (edited layer, subject's last token); (3) capture each `k_probe` at the same layer; (4) compute cosine for all edit×probe pairs **before** editing; (5) snapshot probe locality; (6) apply rank-one ROME, re-measure, compute Δprob/Δlogit; (7) `load_state_dict` to restore, repeat per edit; (8) compute Spearman + AUROC, and compute ENCORE norm-growth as a competing predictor. MEMIT covariance C and AlphaEdit H3 are deferred to phase 2.

---

## 3. Code infrastructure status (independently verified, not agent self-report)

`edit-harness/` has been built and **I personally re-ran it to confirm**:
- `metrics.py` (240) efficacy/generalization/locality/fluency + shared target_token_ids
- `editors/ft_editor.py` (141) FT-L constrained fine-tuning — re-run: `efficacy pre.success=0.0→post.success=1.0` ✅
- `editors/rome_native.py` (232) native ROME rank-one; `rank_one_solve_residual≈1.25e-4` (≈0, (W+ΔW)k=v holds) ✅; MEMIT covariance C flagged as a TODO
- `runner.py` (219) config→edit→metrics→JSON, can `load_state_dict` to restore
- `queue/run_all.sh` (61) serially consumes queue/*.json (one GPU job at a time)
- produced results JSON is saved to disk with complete fields

> **All code needed for the first experiment already exists and has been verified to run**—the first experiment is mainly "batch-capture keys with existing hooks + compute geometry," not writing new training code.

---

## 4. Next steps (awaiting your go-ahead)
The first paper can start with **Llama-3.2-1B** (very small, ~2.5GB) or GPT-J-6B. Recommendation: download the smallest Llama-3.2-1B first and run the kill-gate—within a **few hours** you can get the first signal on "does key geometry actually predict damage or not"—this validates true/false faster than downloading the 14GB GPT-J.
- Minimal starting download: Llama-3.2-1B (~2.5GB) + CounterFact/zsRE (~2GB) ≈ **~5GB**
- Once past the H1 kill-gate, move on to GPT-J-6B / AlphaEdit / MEMIT-C.
