# B6 paper skeleton — honest, CLaRE-differentiated (2026-07-01; rewritten 2026-07-04)

> **AUTHORING PASS (2026-07-04).** This is a writer pass only — content herein has NOT been
> through the hostile submission review; a separate reviewer/verifier lane gates any
> submission. Every quantitative claim below was checked against the canonical JSONs in
> `edit-harness/results/` (not peek files, not recollection); the claim→evidence map at the
> end lists the exact source file for each number, and unverifiable numbers are marked
> `[UNVERIFIED]` inline.
>
> Green-lit after a full-text read of CLaRE-ty (arXiv 2603.19297, ACL 2026 Findings): all 4
> candidate gaps are OPEN. B6 is a real paper IF framed as mechanism + causal + dissociation
> (complementary to CLaRE's empirical predictor), NOT as "first a-priori predictor" (scooped).
> Grounded in the audit-corrected `findings-G1-gate-2026-07-01.md`, the 07-02/07-03 GPU
> campaign (`STATUS-AND-PLAN-2026-07-02-EOD.md`), and the two 07-04 reconciliation findings
> (`findings-MEMIT-SC-RECONCILIATION-2026-07-04.md`, `findings-SEQ-ANALYSIS-2026-07-04.md`).

## Scope banner (binding — set before anything else)
**The signed key-cosine→damage law is Llama-family-specific.** The paper is framed as a
Llama-family mechanistic account **plus a cross-architecture dissociation arm**, not a
universal predictor. Two laws with different reach:
- **Signed law** (sign and rank of ρ(key-cos, damage)): Llama-family only. Off-Llama it
  either goes null (gemma, Phi) or **inverts** (Qwen).
- **Magnitude law** (|key-cos| → |damage|): transfers on **4 of 5** families; **gemma is the
  documented exception** (0.084 ± 0.029, 3-seed).

## Working title
**"When and why does key geometry predict locate-then-edit collateral damage? A closed-form,
editor- and architecture-conditioned account of the Llama family, with a causal test."**

## One-sentence thesis
For rank-one locate-then-edit updates (ROME) on the Llama family, the key-vector geometry that
ranks collateral damage is a **provable consequence of the update's algebra** (a closed-form
S×C decomposition that doubles as a zero-cost surrogate for first-order gradient influence), is
**conditional on editor and architecture** (it vanishes for Adam full-rank fine-tuning and for
MEMIT's multi-layer spread, and inverts on Qwen), undergoes a **depth/regime transition** whose
sign tracks the sign of the damage regime, and — causally — **null-space projection (AlphaEdit)
removes exactly the geometry-predicted damage, whatever its sign**, including for a
refusal-**deletion** edit that was never in the original design.

## Positioning vs CLaRE (cite prominently as concurrent; differentiate on 4 axes)
CLaRE ships an *empirical, single-layer hidden-state cosine* per-fact ripple predictor,
baselined only against GradSim, **correlational by authors' own admission**, over 5
locate-then-edit editors × 3 MHA models (GPT2-XL/GPT-J/Llama3). It does NOT provide:

| # | B6 contribution CLaRE lacks | Evidence status (2026-07-04) |
|---|---|---|
| C1 | **Closed-form ROME S×C derivation** — ΔW=outer(v−Wk,k)/‖k‖² factorizes into edit-strength S=‖v−Wk‖/‖k‖ × geometry C=cos(k_edit,k_probe); "cosine ranks damage" is *built into* the rank-one algebra, and S×C is a **zero-cost closed-form surrogate for GradSim** first-order influence | derived + measured (S×C ≈ GradSim resid to ~2 decimals; `C1_mechanism_sc_table.json`, `G2_gradsim_L8.json`) |
| C2 | **Norm-growth (ENCORE) head-to-head + depth/regime transition** — on the confound-clean within-probe metric, key-cos beats norm-growth mid-layer, norm-growth overtakes at L14, and the sign of the coupling tracks the sign of the damage regime across scale | ✅ measured (G1 + regime cells) |
| C3 | **Editor / architecture dissociation** — geometry-predictability is ROME-specific (dies for Adam full-rank FT-L and for multi-layer MEMIT; near-null on gemma/Phi; **inverts** on Qwen) | ✅ measured 3-seed (`C3_null_*_v2.json`, `C3_memit_*_r3.json`) |
| C4 | **AlphaEdit CAUSAL test (the keystone CLaRE explicitly disclaims)** — null-space projection removes damage in proportion to key-cosine at every layer, erases the Qwen inversion, and collapses a refusal-deletion edit's coupling | ✅ measured; E6 holdout retires the projector-circularity objection |

**Verify-on-write note:** the 4-axis differentiation still reads correctly under the new
Llama-scoping — CLaRE's predictive claim is itself only demonstrated on MHA models incl.
Llama3, so a Llama-scoped mechanistic+causal account is complementary, not subsumed. Axis 3
gets *stronger* under scoping (we now have measured 3-seed nulls/inversions on 5 architectures
and 4 editors, which CLaRE never runs predictively).

## Claims map (each claim → the table/JSON that feeds it)

**Core Llama law (C1/C2).**
- **G1 within-probe gate PASS, 4 layers.** ρ(key-cos, damage): L8 **0.395**, L10 **0.534**,
  L12 **0.602 (peak)**, L14 **0.301**; frac_positive 1.0; perm-p at the 1/301 floor; survives
  double-centering and norm-growth partialling. → `G1_L{8,10,12,14}_analysis.json`,
  `G2_gradsim_L8.json` (keycos 0.395).
- **S×C is a surrogate, not a winner.** rho_SC vs rho_C: L8 0.390/0.395, L10 0.528/0.534,
  L12 0.628/0.602, L14 0.498/0.301 — **S×C loses to raw key-cos at L8 and L10** (and on gemma);
  it equals GradSim-resid to the third decimal (L8 rho_SC 0.3899 = GradSim resid 0.3899).
  Frame S×C as a **zero-cost closed-form surrogate for first-order gradient influence**, never
  "beats key-cosine." → `C1_mechanism_sc_table.json`, `G2_gradsim_L8.json`.
- **Regime transition (C2).** Norm-growth→damage within-probe baseline L8 0.083 / L10 0.246 /
  L12 0.509 / **L14 0.502**: key-cos wins L8/L10/L12, **norm-growth overtakes at L14** (0.502 >
  0.301, all 3 seeds). → `findings-G1-gate-2026-07-01.md`, `G1_L14_analysis.json`.

**Causal arm (C4 + E6).**
- AlphaEdit removes ~95–98% of ROME damage; the **damage removed rises monotonically with
  pre-edit key-cosine at every layer, incl. L14**. By-construction (projector fit on the probes)
  within-probe ρ(key-cos, damage-removed) 3-seed: L8 0.397 / L10 0.532 / L12 0.597 / L14 0.302;
  L14 quartile removed 4.05 → 4.82 → 5.55 → 6.80. → `C4_causal_table.json`.
- **E6 (projector circularity retired) — PRIMARY causal numbers, 3-seed holdout.** With a
  **holdout** projector (fit on held-out keys), within-probe ρ(key-cos, damage-removed) 3-seed:
  L8 **0.390** (0.401 / 0.391 / 0.380), L12 **0.590** (0.590 / 0.610 / 0.570) — matching
  by-construction (0.397 / 0.597); mean damage ROME→Alpha 4.385 → 0.136 (L8), 3.006 → 0.146
  (L12); removed top-vs-bottom quartile ratio 2.60 / 3.13. → `C4_causal_holdout_table_3seed.json`
  (the seed-0-only `C4_causal_holdout_table.json` remains on disk as the earlier record). The
  generic-projector variant exists only as a gate-level result and is not quoted in the main
  text (footnote it).

**Editor dissociation (C3) — locality/coupling spectrum (L8, ρ_signed / mean |damage|).**
FT **0.024 / 18.1** → KL-FT **0.132 / 15.2** (3-seed) → ROME **0.406 / 4.45** (seed 0; 0.395
3-seed) → MEMIT **0.019 / 0.03** → AlphaEdit **~0 / 0.14** (3-seed holdout floor). The law is
locate-then-edit-mechanism-specific, not "more damage." → `C3_null_ft_L8.json`,
`C3_null_ftkl_L8_v2.json`, `gate_llama1b_{ft,ftkl,rome,memit}_cf_L8_s0.json` (mean-damage),
`C3_memit_L8_r3.json`.
- **KL-ladder dose–response (3-seed, run_u5).** Rising KL strength raises the coupling and lowers
  the damage, but at 3 seeds the coupling **rises then plateaus** (~0.15 by kl 0.3): FT (kl 0)
  **0.024 / 18.1** → kl 0.03 **0.091 / 16.3** → kl 0.1 **0.132 / 15.2** → kl 0.3 **0.150 / 13.7** →
  kl 1.0 **0.149 / 11.9**; damage monotone, edit-success 1.0 at every rung. →
  `C3_klladder_{003,030,100}_L8_seeds_u5.json` (3-seed), `C3_null_ftkl_L8_v2.json` (kl 0.1),
  `gate_llama1b_ftkl{003,030,100}_cf_L8_s0.json`. **L12 non-replication (NEW, single-seed):** at
  the ROME peak layer the ladder is DEAD/borderline and non-monotone — ρ **0.119 / 0.097 / 0.088 /
  0.120** (kl 0.03/0.1/0.3/1.0), all ≈ or < the 0.10 DEAD threshold; KL-FT's partial restoration
  is an L8 phenomenon. → `C3_klladder_{003,010,030,100}_L12_u5.json`.
- **MEMIT full layer profile (completed run_u4)** — DEAD at every layer: ρ_C L8 **0.019** /
  L10 **0.034** / L12 **0.037** / L14 **0.012** (L8/L12 3-seed; L10/L14 single-seed), all below
  the 0.10 DEAD threshold vs ROME 0.395/0.534/0.602/0.301. → `C3_memit_L{8,12}_r3.json`,
  `C3_memit_L{10,14}_u4.json`.
- **MEMIT wording is binding** (`findings-MEMIT-SC-RECONCILIATION-2026-07-04.md`): quote
  **ρ_C** only, 3-seed C3 means, DEAD (<0.10), vs ROME 0.395/0.602.
  **NEVER write "MEMIT S×C"** — the S×C closed form is a single-layer rank-one ROME identity;
  MEMIT spreads each edit across 4 layers, so its within_probe_rho_SC is not a valid S×C
  statistic. Approved sentence: *"For MEMIT, the within-probe Spearman between key-cosine |C|
  and collateral damage is negligible — 0.019 (L8) and 0.037 (L12), 3-seed means, both below
  the 0.10 DEAD threshold — versus ROME's 0.41 (L8) / 0.60 (L12). Geometry does not predict
  MEMIT collateral damage."* (Retires the stale single-seed 0.016 editor-spectrum number.)

**Architecture dissociation (C3) — 3-seed signed within-probe ρ.**
gemma-2-2b L13 **0.084 ± 0.029** (DEAD; magnitude-law exception) / Phi-3.5 L16 **0.017**
(signed-blind) / Qwen-0.5B L12 **0.103** (borderline) / Qwen-1.5B L14 **−0.172**
(sign-inverted) / Qwen-3B L18 **−0.119** (inversion scale-persistent). Qwen inversion is
NG-clean and **causally geometry-carried** (AlphaEdit erases it, C4/E6). → `C3_null_{gemma2b_L13,
phi35_L16,qwen05b_L12,qwen15b_L14,qwen3b_L18}_v2.json` (Qwen-1.5B: `C3_null_qwen15b_L14.json`).

**Magnitude law (C3) — 3-seed within-probe Spearman(|key-cos|, |damage|), canonical.**
The magnitude law |C|→|dmg| transfers on **4 of 5** non-Llama families: Llama-1B L12
**0.613 ± 0.019** (L8 0.398 / L10 0.551 / L14 0.306) / Qwen-0.5B L12 **0.320** / Qwen-1.5B L14
**0.412** / Qwen-3B L18 **0.401** / Phi-3.5 L16 **0.321** / **gemma-2-2b L13 0.086 (DEAD,
edit-level perm-p 0.117)** — the sole exception. **gemma is now DOUBLE-DEAD** (signed
0.084 ± 0.029 AND magnitude 0.086 n.s.), which strengthens the anomaly. → `C1_magnitude_table.json`
(`--known --edit_ok`, 3-seed, PASS 8/9 cells). Note: Phi's canonical magnitude is **0.321**, vs
**≈0.36** in an earlier unfiltered peek (the peek omitted `--known`; exact 0.362 was loosely
remembered — reviewer reproduces 0.365 no-known / 0.3623 seed-0-with-known) — footnote this.
`[AUTHORING-PASS — C1_magnitude_table.json has not itself been through hostile review; fold it
into the paper review scope.]`

**Regime law (C2 extension, scale) — 3-seed both sides.**
Sign of the geometry–damage coupling tracks the sign of the mean-damage (regime): Llama-3B L24
**+0.376** (3-seed 0.351–0.393, positive regime, mean per-edit dmg +0.13…+0.18) vs Llama-8B L24
**−0.097** (3-seed −0.078…−0.117, improvement regime, mean per-edit dmg −0.02…−0.05). Within
8B, sign flips with depth: L24 **−0.097** (3-seed mean; per-seed −0.117 / −0.094 / −0.078),
L16 **+0.173** / L28 **+0.155** (seed-0). Norm-growth
transition tracks **relative** depth (8B L28 = 0.875 rel depth, NG-dominance directionally
confirmed, attenuated ~55% vs 1B). The **sign tracks the damage regime (3-seed both sides);
the coupling magnitude is attenuated at 8B (|ρ| 0.07–0.17 vs 0.38 at 3B)** — it is not a
scale-invariant magnitude. → `C3_regime_3b_L24_r4.json`, `C3_regime_8b_L24_r4.json`,
`C3_llama8b_r3.json`.

**Deletion collateral (U1) — geometry-governed (new section; now 3-seed on the causal cells).**
Refusal-deletion collateral is geometry-governed: L12 3-seed within-probe (non-DC) ρ
**0.657 / 0.699 / 0.681** (DC 0.437 / 0.508 / 0.500), **above** the rewrite reference 0.602;
layer profile (raw key-cos within-probe, 3-seed) L8 **0.461 ± 0.017** / L14 **0.519 ± 0.018**
(L12 raw aggregate 0.663; gate S×C arm agrees) → `C3_u1_blockB_L{8,14}_seeds_u5.json`. Variant robustness (now 3-seed, run_u4-resolved) — **two distinct statistics, do not conflate**:
(a) gate **S×C** non-DC + DC (seed-0, `u1_gate_*`); (b) **raw key-cos within-probe** 3-seed
(`C3_u1_blockC_*_seeds_u4.json`). **eos** robust on both — gate non-DC S×C **0.653**, raw
within-probe **0.616 ± 0.008** (per 0.605 / 0.624 / 0.619). **suppress** is the **fragile**
variant — gate non-DC S×C **0.621** but **DC-FRAGILE → ~0.16**, AND raw key-cos within-probe
**0.073 ± 0.024** (3-seed, DEAD). These are different statistics, not a discrepancy: the weak raw
coupling is CONSISTENT with and strengthens the DC-fragility reading (the non-DC S×C coupling is
not robust geometry). Report suppress as the negative/fragile variant; never average or swap the
two numbers. **AlphaEdit-delete collapses damage 4.10 → ~0.10 AND coupling to 0.036 ± 0.014**
(3-seed, DEAD — geometry-carried damage removed whatever its sign). **Qwen-1.5B-delete
−0.066 ± 0.013** (3-seed, mean damage −0.036) — the Llama-scoping replicates in the deletion
setting. Transplant gate: S×C beats a transplant baseline by Δρ (vs **best** transplant)
**0.59 / 0.55 / 0.61** (L12 seeds), **0.31** (L8), **0.38** (L14). → `u1_gate_refusal_L12_s{0,1,2}.json`,
`u1_gate_refusal_L8_s0.json`, `u1_gate_refusal_L14_s0.json`, `C3_u1_blockC_{eos,suppress}_seeds_u4.json`,
`C3_u1_blockD_alphadelete_seeds_u2.json`, `C3_u1_blockE_qwen15b_seeds_u2.json`,
`U1_E1_transplant_GATE_alphadelete_L12_s{0,1,2}.json` (damage 0.10 / 0.10 / 0.11),
`U1_E1_transplant_GATE_*.json`.
**Deletion transfers to zsRE (dataset generality; single-seed, refusal, L10).** Raw key-cos
within-probe **0.241** (PASS) / gate S×C non-DC **0.234** → DC **0.311** (**DC-ROBUST**, unlike
suppress) — below the zsRE rewrite law (0.361, same layer) and CF-deletion (0.657, L12): geometry
governs deletion collateral across datasets. → `C3_u1_zsre_delete_L10_u5.json`,
`u1_gate_zsre_refusal_L10_s0.json` (canonical run_u5 files, byte-identical to the pre-computation).
(U1 L8/L14 refusal layer profile is now 3-seed — folded above; no longer a TODO.)

**Sequential no-restore (new section, DESCRIPTIVE ONLY; 4-stream, run_u4-promoted).**
Survival **collapses** after 50 edits — per-stream **10 / 14 / 42 / 36%** (pooled **25.5%**);
report the range, not a single point, and NOT "monotonic". Position fragility ρ(position,
survival) per-stream **0.31 / 0.20 / 0.48 / 0.51** (~0.20–0.51), **pooled 0.372 (p=0.0005)** —
*"later-applied edits survive modestly more often"* (reviewed 2-stream baseline pooled 0.25,
p=0.009). The pooled 0.372 mixes streams with heterogeneous survival base rates (10–42%), which
the per-stream range already exposes — report both, lean on the range. **The ρ≈0.55 figure is
RETRACTED — never use it.** H1 geometry-attribution stays
**UNSETTLED at 4 streams**: pooled position-partialled S×C ρ **0.097 (p=0.176)**; per-stream
partials −0.01 / +0.08 / +0.03 / +0.16 — **NO geometry-attribution language admissible.** Verdict
language governed by `findings-SEQ-ANALYSIS-2026-07-04.md`. → `SEQ_analysis_L12_4stream.json`
(shared streams reproduce `SEQ_analysis_L12.json` to 1e-4).
**Flank layers (run_u5, descriptive, 2-stream):** *survival collapse* is hardest at L14 —
pooled survival **28%** (L8, per-stream 28/28) / **5%** (L14, 8/2) vs 25.5% (L12); *position
fragility* is modest and **non-monotone** across layers, ρ **0.57 (L8) / 0.372 (L12) / 0.36 (L14)**
(mildest at L14). Does not touch the H1 verdict. → `SEQ_analysis_L{8,14}.json`.

**Generality (C3 breadth).**
zsRE L10 **0.361 ± 0.014** signed / 0.495 magnitude (3-seed) — not a CounterFact artifact;
Llama-3B L14 **0.291** (1B ref 0.301) — replicates at 3× scale; GPT-2-XL sanity ES **.98/.985**,
PS **.74/.77** (PS gap = documented no-context-templates), NS **.74/.71**, esr **0.84/0.795**
(both VALIDATE-WARN → limitations footnote). → `C3_null_llama1b_zsre_L10.json`,
`C3_null_llama3b_L14.json`, `sanity_gpt2xl_rome_cf_L{5,17}_s0.egl.json`.

**Canonical EGL table (Llama-1B L12, CounterFact, 2-seed means).** Efficacy / paraphrase /
neighborhood-specificity per editor, the standard edit-quality view of the locality spectrum:
ROME **ES 1.00 / PS 0.991 / NS 0.043** (worst neighbor preservation) → AlphaEdit
**0.998 / 0.966 / 0.578** → MEMIT **0.995 / 0.955 / 0.741** (best NS). The NS ordering (ROME ≪
AlphaEdit < MEMIT) matches the mean-damage spectrum. → `egl_llama1b_{rome,memit,alpha}_cf_L12_s{0,1}.egl.json` (2-seed).

**Anisotropy (why-L14 / why-Qwen) — Discussion, descriptive only.** Llama-1B L14 edit keys are
strongly coned: mean pairwise key-cosine **0.460**, **uncentered** top-1 eigenvalue fraction
**0.47** (inflated by the shared mean direction, mean-cos 0.46), **centered** participation ratio
**36.7** (vs column-permutation null ~106 and isotropic-Gaussian null ~190). Qwen-1.5B L14 keys
are far less coned (mean pairwise cos **0.200**, uncentered top-1 **0.22**, centered PR **60.4**). The crowding contrast **replicates across 3 seeds** (Llama mean-cos 0.460/0.430/0.431 vs Qwen
0.200/0.196/0.197). → `ANISO_analysis_L14.json` + `ANISO_analysis_L14_s{1,2}.json`. **Write as
DESCRIPTIVE, key-space-only (top-1/PR quoted at seed 0) — NO damage join, NO causal language**
(the module's `interpretation_constraints` forbid attributing the damage regime to anisotropy).
`[FLAG — the file's `interpretation_constraints` block still says "no Qwen raw-key bank exists /
cross-model L14-pending," but the file itself now contains the Qwen-1.5B L14 bank and a populated
`cross_model_contrast`. The constraints text is stale relative to the data; the `cross_model_contrast.note`
("Descriptive only… do NOT over-read a single-seed two-point contrast") is the governing caveat.
No gemma bank exists in this file, so no gemma anisotropy claim is admissible. Reconcile in review.]`

**D3 (reframed).** Routing is **degenerate** (always-AlphaEdit, 10/10 configs) — geometry does
**not** predict editor *choice*. It predicts **benefit magnitude**: ROME→AlphaEdit damage-ratio
tracks coupling, Llama L8/L10/L12/L14 ≈ 68× / 32× / 36× / 71× → near-zero-damage families
(Phi/Qwen) give unstable/near-zero ratios. → `D3_routing_eval_v2.json`,
`D3_routing_per_edit_sweep.json`.
`[UNVERIFIED — the specific "27–41× Llama → 10.4× Qwen-0.5B → 9.6× Qwen-1.5B → 6.7× Phi"
monotone chain from STATUS §1.7 is not the ratio computed in D3_routing_eval_v2.json (which
gives Llama 32–71× and negative/degenerate ratios for near-zero-damage families). Keep D3
qualitative ("benefit magnitude tracks coupling, largest on the high-damage Llama regime") or
recompute the per-family benefit statistic before quoting the chain.]`

## Section plan
1. **Intro** — collateral damage in locate-then-edit; the a-priori-predictability question;
   CLaRE as concurrent empirical predictor; our mechanism/causal/dissociation angle; the
   Llama-family scope stated up front.
2. **The S×C decomposition (C1)** — analytic; the GradSim-surrogate framing; predicts C2/C3/C4
   as consequences.
3. **Confound-clean measurement (G1 within-probe gate)** — method + non-independence /
   probe-marginal-leakage retirement (permutation null, double-centering, NG-partialling).
4. **Depth & regime transition (C2)** — key-cos vs norm-growth across depth; the L12→L14
   crossover; scale/regime law (sign tracks damage-regime sign; relative-depth NG transition).
5. **Editor & architecture dissociation (C3)** — ROME vs FT/KL-FT (dose–response ladder)/MEMIT
   (full DEAD layer profile); Llama vs gemma/Phi/Qwen (magnitude-transfers-4/5, canonical
   magnitude table, gemma double-dead, Qwen inversion); canonical EGL table; tie each to (3).
6. **Causal test (C4 + E6)** — AlphaEdit removes the geometry-predicted damage at every layer,
   erases the Qwen inversion; E6 holdout projector retires circularity; the L14 sharp test.
7. **Deletion collateral (U1)** — geometry governs refusal/eos deletion damage too (3-seed);
   suppress is the fragile/DEAD variant; AlphaEdit-delete collapse (3-seed); transplant gate;
   Qwen-delete replication (3-seed).
8. **Sequential no-restore (descriptive)** — survival collapse + modest position fragility; H1
   geometry-attribution explicitly null (4-stream, H1 pooled partial 0.097 n.s.).
9. **Generality** — zsRE, 3B scale, GPT-2-XL sanity, canonical EGL table.
10. **Discussion — anisotropy (why-L14 / why-Qwen)** — descriptive, single-seed, key-space-only,
    no causal attribution.
11. **Related work** — CLaRE (4-axis differentiation), ENCORE, Knowledge-in-Superposition, Hase.
12. **Limitations** — Llama-family scope for the signed law; ROME rank-one mechanism; honest
    null/inversion arms (FT, MEMIT, gemma, Phi, Qwen); remaining low-seed cells (single-seed:
    MEMIT L10/L14, KL-L12 ladder, zsRE-deletion, Llama-3B L14, generic-projector; 2-seed: EGL;
    2-stream: seq flank); GPT-2-XL esr VALIDATE-WARN; sequential geometry-attribution unsettled;
    anisotropy descriptive/not damage-joined (crowding contrast now 3-seed).

## Target venue
- **PRIMARY: ARR → EMNLP/NAACL main** (CCF-B, archival; NOT Findings/BlackboxNLP). The
  mechanism + causal + dissociation package is a main-track contribution once scoped honestly.
- **SECONDARY / later: KBS (SCIE Q1) journal extension** after ARR — carries the D3
  benefit-magnitude predictor + breadth. Per standing venue rule: SCIE-indexed / CCF-ranked is
  the first filter; TMLR / BlackboxNLP / COLM / EACL do NOT qualify.
- Nothing to TNNLS/Neurocomputing while evidence is ≤8B and there is no theorem-bearing
  artifact.

## TODO slots — DONE (run_u2/u4 folded 2026-07-04)
- ~~U1 AlphaEdit-delete seeds~~ → 3-seed, coupling 0.036 ± 0.014, damage 4.10→~0.10.
- ~~U1 Qwen-1.5B-delete seeds~~ → 3-seed, −0.066 ± 0.013.
- ~~KL-ladder dose–response~~ → folded (2-seed for the new rungs) into §5.
- ~~E6 holdout 3-seed~~ → `C4_causal_holdout_table_3seed.json`.
- ~~MEMIT full layer profile~~ → L8/L10/L12/L14 all DEAD.
- ~~Canonical magnitude table, EGL table, suppress/eos 3-seed, aniso L14~~ → folded.

## TODO hooks — ALL run_u5 cells folded (2026-07-04, 22/22 drained clean)
- ~~U1 L8/L14 refusal seeds~~ → 3-seed raw within-probe L8 0.461 ± 0.017 / L14 0.519 ± 0.018.
- ~~KL-ladder kl 1.0 s2~~ → 3-seed L8 rungs (kl 0.03/0.3/1.0 = 0.091/0.150/0.149); coupling plateaus.
- ~~Sequential flank layers~~ → L8/L14 folded (2-stream, descriptive).
- ~~Aniso seed contrasts~~ → 3-seed crowding contrast (Llama 0.460/0.430/0.431 vs Qwen 0.200/0.196/0.197).
- ~~zsRE-deletion cell~~ → canonical files confirmed byte-identical to the pre-computation.
- **NEW (folded):** KL-L12 ladder (single-seed, DEAD/borderline — dose-response does not replicate at the peak layer); EGL now 2-seed.
- **Remaining low-seed (in Limitations):** MEMIT L10/L14 (single-seed), KL-L12 ladder (single-seed),
  zsRE-deletion (single-seed), Llama-3B L14 (single-seed), generic-projector (gate-level); EGL (2-seed),
  seq flank layers (2-stream).
(Authoring-pass modules still needing their own review: `C1_magnitude_table.json` / magnitude_table.py,
`ANISO_analysis_L14.json` / analyze_aniso.py, `make_figures.py` output — fold into the paper review.)

## Decision
B6 is worth finishing as a Llama-family mechanistic+causal+dissociation paper. The causal arm
(C4/E6) is the keystone and the sharpest CLaRE differentiator; the U1 deletion result extends
the mechanism to a genuinely new edit type. Frame honestly, cite CLaRE as complementary, hold
the scope to the Llama family for the signed law, and route ARR/EMNLP-main first.
