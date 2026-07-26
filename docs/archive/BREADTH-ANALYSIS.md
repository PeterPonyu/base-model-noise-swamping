# Full-Scope Analysis — Knowledge-Editing Breadth Scan (workflow, 5 agents)

> Data: 20 configs (Llama-1B/3B, Qwen-0.5B/1.5B/3B × ROME/FT-L × CounterFact/zsRE × layers); gemma/Phi/seeds still being added.
> Method: 4 parallel analysis lenses (architecture×scale, editor×layer, baseline×metric, skeptic) + synthesis. Date 2026-06-30.

## One-sentence conclusion (scoped, no longer universal)
**Pre-edit key cosine predicting which facts a ROME edit will damage holds only for the Llama family + rank-one ROME + middle layers (L4–L12)** (ρc=0.33–0.50, AUROC 0.71–0.83, beating norm-growth; Llama-3B replicates at ρc≈0.24–0.30). **It fails everywhere the rank-one × collateral-damage premise is absent**: FT-L (full-rank update, ρc≈0.04–0.14 ≈ random), the last Llama layer (L14, norm-growth ρn=0.57 overtakes), and **all of Qwen** (ρc=−0.14~+0.13).

## Why Qwen fails — a genuine null, not a bug
- Qwen edit_success=0.98–1.00 (not an editing failure).
- **Qwen's ROME edit collateral damage ≈0 logit (Llama is +3.0, a ~200x difference)**: Qwen's wider MLP + near-orthogonal keys make the rank-one update naturally local → **there is no damage to predict**.
- Qwen's negative ρc is "noise on a zero-variance target," not an inverse correlation.
- → Qwen actually **corroborates** the mechanism (damage ∝ key alignment), turning a "negative result" into "positive evidence."

## ⚠️ Two unresolved confounds (the Llama effect is "directionally real, but not yet clean")
1. **Non-independence**: each ρ is computed over ~18k pairs, but there are actually only ~150 edits × ~70–150 reused probes → **the effect size is usable, but the p-values/confidence intervals are severely inflated**.
2. **Probe-marginal leakage**: AUROC on Qwen remains 0.65–0.80 even when ρc≈0 → indicating AUROC is partly driven by "which probe is inherently fragile" (column structure), not edit-specific pairwise geometry. **The same contamination is present in Llama's AUROC too** → the current headline is only a "flattened-level correlation," not yet proven to be "within-probe pairwise geometry."
- The decisive test **cannot be done offline**: the raw COS/damage matrices were not saved, only scalar summaries.

## How to rewrite the paper (the original "universal predictor" is dead, but it's publishable)
**Reframe as a conditional, mechanistic boundary result**:
> **"When does locate-then-edit collateral damage become geometrically predictable? An architecture- and editor-conditioned phenomenon"**
- The contribution is not a universal predictor, but a **mechanistic dissociation**:
  1. The ROME update is analytically rank-one (ΔW=outer(v−Wk,k)/‖k‖²) → each probe's perturbation = (v−Wk)·cos(k_edit,k_probe) → **"cosine predicts damage" is built into the rank-one structure**; FT-L's full-rank gradient has no such alignment → not predictable. **The ROME-vs-FT dissociation is the cleanest, most defensible result.**
  2. The predictor is **layer-restricted**: strongest in the middle-layer fact-storage band; in the final layers key anisotropy rises and magnitude (norm-growth) dominates.
  3. **Architecture gates on key geometry**: Qwen's orthogonal, wide MLP → edits are local → the phenomenon and the damage vanish together.
- **Honesty requirements before submission**: use signed Spearman as the primary metric (not AUROC, to expose the heteroscedasticity artifact); report within-probe partialled statistics + permutation null; report mean signed damage per config (so that rows with ≈0 damage are explicitly marked "unmeasurable"); **do not** sell Qwen's ρc<0 as a counter-prediction. Position it as a focused mechanism/diagnostic short paper (workshop/short-paper), not a universal interference law.

## Next steps (cheap-first, single GPU)
1. **[Hours, decisive, GATE]** Re-run Llama-1B L8/L10/L12 ROME, **dump the raw COS/damage matrices + ≥3 seeds**, compute **within-probe partialled Spearman + column-permutation null**. This step directly retires both major threats (non-independence + column leakage), confirming or killing the headline. **Do this before anything else.**
2. **[Cheap]** Add gemma-2-2b + Phi-3.5 ROME+CF middle layers + test key cosine vs. mean damage → determine whether the boundary is "Llama-specific" or a general law of "high key cosine ⇒ damage present ⇒ geometrically predictable" (if a non-Llama, high-cosine model also holds → upgrade to a key-cosine-gated law, a major framing upgrade).
3. **[Cheap]** Wide-MLP hypothesis: regress mean damage on the intermediate/hidden ratio (data already available); for causal evidence, edit Qwen at the layer/projection that maximizes key cosine and see whether damage and predictability reappear.
4. **[Cheap]** A regularized FT-L (add a KL neighbor term, lower the lr) to prove that FT's failure is structural (full-rank, not k-aligned) rather than simply over-damage → preempts reviewers.
5. **[Moderate]** MEMIT/batched ROME: how far the geometric signal holds up as it approaches FT.
- Deferred: larger models, datasets beyond CF/zsRE — these won't change the conclusion until #1 resolves the partialling issue.

## Notes
- The raw workflow output is stored at `edit-harness/engine/breadth_analysis_raw.json`.
- gemma/Phi/seeds will be added to the table once complete; this does not change the conclusions above (the Qwen null is settled).
