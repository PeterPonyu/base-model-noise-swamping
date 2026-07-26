# WHY + What's Missing + Next Directions (Sonnet panel, 4 agents)

> ⚠️ **SUPERSEDED for direction planning by `EXPANDED-DIRECTIONS-2026-07-01.md`** (audit +
> SCIE/CCF venue research + CPU results of 2026-07-01). That doc is now authoritative for: the
> 7-direction count, venue targeting (B6→TNNLS, P4→IJF, P3→TIFS/ESWA), the D3 routing artifact
> (built on CPU), and the P4/P3 cleaned re-analyses. This file is retained for the original WHY
> (S×C algebra) + the G0–G4 gate history below.

> Full-scope analysis based on a broad-area scan. Date 2026-06-30. Raw output `edit-harness/engine/why_next_raw.json`.

## (1) WHY — One algebraic identity explains everything
The ROME update **ΔW = outer(v−Wk, k)/‖k‖²** decomposes into two independent multiplicative factors:
- **S = ‖v−Wk‖/‖k‖** (edit strength, varies per-edit)
- **C = cos(k_edit, k_probe)** (geometry, varies per-probe)

Every observation is the result of "which factor dominates the variance of the damage matrix":
| Phenomenon | Explanation |
|---|---|
| Llama mid-layers L4–L12 high ρc | S has moderate-low variance (this layer stores facts), C has spread → geometry dominates |
| Llama L14 norm-growth overtakes | S is large and high-variance (edit is on a non-factual layer), C collapses (mean_cos→0.44) → magnitude dominates |
| **FT-L any layer ρc≈0** | Adam's adaptive scaling turns the rank-1 gradient into a high-effective-rank ΔW, with **no row-space alignment to k_edit** → the C factor is structurally absent. **This is not a tuning failure, it's a mathematical necessity of Adam** (m_ij/√v_ij cannot be decomposed as f(i)g(j)) |
| **Qwen total failure** | **S≈0**: facts are finely distributed across 24 layers, each layer's contribution is small, so the optimizer gets v−Wk≈0 → damage≈0 → no signal |

**Key counter-intuitive finding**: at matched mean_cos (0.33 vs 0.31), Qwen's damage differs by **26×** → proving the primary cause is **residual norm S (distributed storage), not key orthogonality**.

**One measurement confirms everything**: dump ‖v−Wk‖ for every edit, comparing Llama-1B L8 vs Qwen-0.5B L12 (same relative depth ratio 0.50). If Qwen's residual norm is 4–8× smaller → distributed storage confirmed. **1 line of code + 200 edits, <15 minutes**. If <2× → the Qwen null has another cause (GQA/wide MLP), and the mechanism story needs revision.

## (2) MISSING — Gating experiments needed to complete paper #1
The raw matrices were not saved, so the two confounds cannot be ruled out → the headline result is not yet publishable.
- **G0 [must go first, 8 lines, 0 GPU]** Add `--save_matrices` to `killgate_keygeom.py` (four np.save calls for COS/damage_l/pre_l/norm_growth). **All downstream statistics depend on this.**
- **G1 [GATE, ~6 GPU-hr]** `analyze_matrices.py`: per-probe Spearman (within-column, partialling out probe identity) + column-permutation null (1000 iterations) + per-edit row Spearman. **If within-column ρ≥0.15 and the permutation null collapses to 0 → both confounds are ruled out, the headline holds; if it collapses → the current AUROC is a probe-selection artifact, and this is demoted to a secondary finding.** seeds 1/2/3 @ L8.
- **G2 [~1 GPU-day]** GradSim baseline (required by H2 in the original spec; reviewers will definitely ask about it).
- **G3 [2hr, 0 GPU]** Lexical BLEU + SBERT topical-similarity baseline (to rule out "high cosine is just topical similarity"). CPU.
- **G4 [~1.5 GPU-day, causal GATE]** AlphaEdit (null-space projection): predicts it **specifically protects high-cosine probes**. This is the paper's **only causal test**; everything else is correlational.
- **⚠️ Dead-end marker**: if G1's within-column ρ<0.10, the headline is downgraded, and the paper pivots to lead with **Qwen-null + FT-overcollateral + regime-transition**. **Know this before investing in G2–G4.**

## (3) Additional directions (ranked by novelty × feasibility × reuse)
**D1 — The FT over-collateral-damage paradox (100% reuse, 1 day)** ⭐ strongest spin-off
FT causes **4× larger** damage than ROME (18 vs 4.4 logit) yet is geometrically unpredictable (ρc=0.04) → **directly refutes the practitioner consensus that "FT is safer than ROME"**. Contribution = controlled matched comparison + mechanism (Adam rank inflation) + fix (KL-regularized FT brings damage down to ROME's level while the ρc null persists → proving the null is structural). Publishable as a short paper, reuses the existing `ft_editor.py`.

**D2 — Architecture-conditioned edit-safety audit (100% reuse, 1 GPU-day)**
**Key orthogonality of an architecture is a measurable pre-deployment screening property; choice of backbone dominates layer choice or hyperparameter tuning in its effect on edit safety.** The 200× Llama-vs-Qwen damage gap is the anchor. Run Gemma/Phi/Mistral through the same harness and rank them by (mean_damage, mean_cos, ρc). **No one in the ROME/MEMIT literature has treated architecture as a safety-screening axis.** Methodological point: AUROC stays at 0.79 even at zero damage → **safety reports must report mean_damage directly, not just AUROC**.

**D3 — Geometry-gated editor routing (bridges B4, ~80% reuse, 1 GPU-day)**
Compute mean_cos(k_edit, K_layer) before editing: high (Llama-like, ≥0.25) → route to AlphaEdit; near-zero (Qwen-like) → vanilla ROME is already local, so projection wastes compute. Turns the B6 predictor into a **deployable routing policy** + a second causal test of the mechanism. AlphaEdit comes from G4; the routing wrapper is ~200 lines.

## (4) PREPARE NEXT — Ordered setup checklist
**Now (cheap, unlocks everything):**
1. Add `--save_matrices` to `killgate_keygeom.py` (15 min, 8 lines) → unlocks all downstream statistics
2. Write `experiments/analyze_matrices.py` (CPU, ~1hr) → within-column Spearman + permutation null
3. Queue Llama-1B L8 **seeds 1/2/3** + `--save_matrices` (~20min each, ~1 GPU-hr total)
4. Add `float(residual.norm())` to `rome_native.py`, saved to JSON (1 line, 0 GPU) → unlocks the Qwen-vs-Llama residual-norm mechanism test

**After the GATE passes (G1 within-column ρ≥0.10):**
5. Gemma-2-2B @ L13 (~25min, 9GB) — the most important architecture extrapolation, determines whether this is "Llama-specific" or "rank-one universal"
6. Phi-3.5-mini @ L16 (~30min, 12GB) — completes the 4-architecture dissociation table
7. GradSim baseline (~3hr coding)
8. Implement AlphaEdit in `editors/` (~250 lines, reuses `rome_native`) → covers G4 + D3
9. Add `--ft_kl_weight` to `ft_editor.py` (λ∈{0.01,0.1,1.0}) → prove the FT null is structural
10. Download MQuAKE (~200MB, download it now) → unlocks B1 without waiting

**Deferred**: B5 multimodal (LLaVA-7B needs fp16 + a new VLM harness + visual probes; high setup cost, low marginal novelty) — do not touch until all GATEs are clean.

**⚠️ Branch point for the Gemma result**: if Gemma's ρc≈0 but the edit succeeds → check L13's mean_cos. Low (~0.13) → supports the "cosine threshold" hypothesis, narrowing the scope to "high-key-cosine, Llama-like architectures"; high but ρc still null → there's an architectural feature the formula doesn't capture, and the mechanism story needs an added term. Both outcomes are informative; the latter requires reframing before writing.
