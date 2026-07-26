# DEEP-RESEARCH SYNTHESIS — knowledge-editing directions (2026-06-30)

> Source: 6-agent Sonnet-5 deep-research workflow (live web/arXiv + skeptical re-check of on-disk JSONs). Covers B6, D1, D2, D3, B-family (B1/B2/B4), and the S×C mechanism. Read alongside PORTFOLIO.md / RESEARCH-DEFINITION.md / NEXT-DIRECTIONS.md.

## TL;DR — two findings that reshape the program

1. **B6's flagship niche is no longer a vacuum.** **CLaRE-ty Amid Chaos (arXiv 2603.19297, ACL 2026 Findings, Mar 2026)** already delivers an *a-priori, per-fact, forward-activation representation-geometry* predictor of **both** ripple **and** unrelated-fact (locality) damage, across **ROME/MEMIT/AlphaEdit**, vs **GradSim**, on 11,427 facts (+62.2% Spearman, 2.7× faster). RESEARCH-DEFINITION.md's "no surveyed paper provides a falsifiable a-priori per-fact damage predictor / true vacuum" claim is now **false as written** and must be corrected before any drafting. This also **effectively scoops B1**.
2. **The decisive gate (G1) has never run.** Every on-disk "PASS — key geometry predicts damage" is a **flat/pooled Spearman over ~60k edit×probe pairs from only ~150 edits × ~400 probes** — exactly the non-independence + probe-marginal-leakage confound the project's own BREADTH-ANALYSIS.md flags. The confound-retiring test (within-probe partialled Spearman + column-permutation null; coded in `analyze_matrices.py`, needs `--save_matrices`) has **no `.npz` and no `GATE_*.json` on disk**. It is blocked behind the still-running `engine.py` + queued `run_gate.sh`. **The true headline number does not exist yet.**

## Per-direction verdicts

| Dir | Novelty | Verdict | One-line |
|---|---|---|---|
| **B6** flagship | CONTESTED (was "vacuum") | **PURSUE-AFTER-GATE** | reframe as mechanism-first, editor/arch-conditioned; complementary to CLaRE, not "first predictor" |
| **D1** FT paradox | CONTESTED | **RESCOPE** | "4×/ρc≈0" is a cherry-pick; Adam-rank mechanism is unmeasured text; competing paper argues opposite |
| **D2** arch safety audit | CONTESTED | **PURSUE-AFTER-GATE** | "200×" is unmatched-depth; honest matched gap ~9–26×; table doesn't exist yet |
| **D3** geometry routing | **OPEN (best)** | **PURSUE-AFTER-GATE** | genuinely unoccupied, but fully premise-gated on G4 (AlphaEdit never run); cosine-only rule already falsified on disk |
| **B1/B2/B4** | B1 SCOOPED · B2 crowded · B4 open | mixed | drop B1; B2 hedge only; **B4≡D3 is the best 2nd paper** |
| **Mechanism** S×C | CONTESTED | **PURSUE-AFTER-GATE** | crossover + FT dissociation are REAL; Adam-rank + Qwen-distributed-storage are ASSERTED-not-measured |

## What's REAL on disk vs ASSERTED (skeptical audit)

**Verified real (spot-checked against raw JSON):**
- Layer crossover: known-probe ρ(cos,damage) L4 0.333 → L8 0.389 → L12 0.496 → **drops** L14 0.232, norm-growth overtakes cosine only at L14. Genuine regime transition.
- Editor dissociation: ROME ρc 0.30–0.50 vs FT-L ρc 0.036–0.143 (near-null). Reproducible.
- The one PASS killgate (Llama-1B/L8/seed0): known ρ=0.421, AUROC=0.727, beats norm-growth (ρ=0.084). Real, but single-seed + flat/pooled.
- Methodology point (D2): AUROC stays ~0.79 even at ~0 mean damage (Qwen0.5B L6). Solid, defensible.

**Asserted but NOT measured (do not publish as mechanism until run):**
- **Adam rank-inflation** (FT-null cause): zero SVD/effective-rank ever computed; also *in tension* with GaLore/LoRA evidence that Adam FT updates are often **low** rank. Needs `torch.linalg.svdvals(ΔW)`.
- **Qwen distributed storage / S≈0**: **no Qwen config anywhere records `mean_residual_norm`** (only Llama-1B-L8 22.87 and Phi-3.5-L8 26.16 have it). The "1-line, <15 min" residual-norm dump the docs promise was never run.
- **"200×" arch gap**: unmatched-depth cherry-pick (Llama L8=0.50 depth vs Qwen L6=0.25). Depth-matched = ~9.4×; cosine-matched = ~26×. Real but ~an order smaller than advertised; won't survive a reviewer sanity check.
- **"4× FT damage / ρc≈0"** (D1): one config; range across configs is 4×–15×+ and ρc 0.036–0.143. KL "fix" only ever ran on a `tiny-random-Llama` stub.
- **"mean_cos collapses to 0.44"** (docs): **backwards** — 0.44 is the *highest* mean-cosine measured (L14); cosine *rises/saturates* while its *correlation* with damage collapses. Fix the phrasing.
- **"26× at matched cosine"**: a 2-config coincidence, not a designed matched-cosine grid. Replicate ×3–4 before leaning on it.

**Overstatement pattern (systemic):** harness self-labels "PASS" at a looser internal bar (ρ≥0.2, AUROC≥0.6) than the locked H1 (ρ≥0.4, AUROC≥0.75) — the real known-probe AUROC 0.727 actually **misses** H1's 0.75. Grade against the pre-registered bar, not the code default. (Same class of overstatement the sibling reliability-portfolio re-audit just flagged.)

## Scoop map (2025-2026)
- **CLaRE-ty (2603.19297)** — scoops B6's core claim + B1. Read the full paper (not abstract) to see if it already ran ENCORE-norm-growth / AlphaEdit-causal baselines; that determines how much of H2/H3 is still open.
- **Fine-tuning Done Right (2509.22072)** — argues FT badness is a *fixable pipeline/batching artifact*, opposite of D1's "structural Adam" thesis. Must cite/reconcile.
- **Detecting Edit Failures (2305.17553)** — ROME/MEMIT damage *more* than FT under adversarial near-neighbor probes → D1's random-probe ranking may be a probe-selection artifact.
- **Benchmarking & Rethinking KE (2505.18690)** — arch-dependent locality rankings already public → D2's "no one treats arch as safety axis" must narrow to "zero-edit pre-hoc index."
- **The Fall of ROME (2406.11263)** — rival per-arch mechanism (first-token key-statistics / denominator blowups) competes with "key orthogonality."
- **MechLens (2606.07978, Jun 2026)** — GQA (Qwen-like) crystallizes facts later/more distributed than MHA (Llama-like): independent *support* for "Qwen stores facts differently."
- **D3/B4 routing** — no scoop found (WilKE routes layers; RRDA routes adapters; EvoEdit cheapens projection). Genuinely open.

## Recommended plan of attack (revised)

1. **[0 GPU, minutes] Read CLaRE-ty in full** → correct the "vacuum" claim in RESEARCH-DEFINITION.md; decide exactly which of H1/H2/H3 remain novel (H2 norm-growth head-to-head + H3 AlphaEdit causal test are the likely survivors).
2. **[cheap, decisive] Run G1** the moment the GPU frees: `killgate_keygeom.py --save_matrices` on Llama-1B L8 seeds 0/1/2 (~1 GPU-hr) → `analyze_matrices.py` (CPU). Grade H1 against the **locked** thresholds. This single step decides whether B6/D2-ρc/D3/B4 have any signal left. **Do not draft before `GATE_*.json` exists.**
3. **[cheap mechanism receipts] While at it:** dump Qwen `mean_residual_norm` (1 line, ~12 min) to test distributed-storage; `svdvals(ΔW_FT vs ΔW_ROME)` (0 GPU) to test Adam-rank. Converts two asserted mechanisms into measured ones — or kills them.
4. **[re-anchor D2]** recompute the arch gap depth+cosine-matched (→ ~9–26×, drop "200×"); add success-filtered + probability/KL-space damage; actually download Mistral; fill the 4-arch table.
5. **[if G1 passes] Run G4 (AlphaEdit causal test)** → this is the single most novelty-defensible experiment vs CLaRE, and the shared premise D3/B4 depend on. Then build the D3/B4 router with a **2-feature rule (cosine + residual-norm)**, not cosine-only.
6. **Drop B1** (CLaRE). **B2** only as a narrow small-budget precedence hedge vs the spectral/energy/MPES cluster. **B4≡D3 is the recommended second paper.**

**If G1 dies** (within-probe ρ<0.10): pivot to the already-drafted fallback — *"When does locate-then-edit collateral damage become geometrically predictable? An editor- and architecture-conditioned dissociation"* — which rests only on the three verified dissociations and survives even without a clean per-fact predictor.
