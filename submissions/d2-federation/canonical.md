# When Do Rank-One Knowledge Edits Merge? A Gain-Screened Two-Regime Law of Edit Federation

**STATUS: SUPERSEDED as source of truth (2026-07-16). `main.tex` is now canonical** —
it has evolved far past this draft (blind-panel response round: robustness statistics,
threshold statement, TSV engagement, reproducibility/ledger sections, expanded
tables/figures). This file remains as the v0.1 design record only; do NOT port from it.

**STATUS: DRAFT v0.1 (2026-07-15) — canonical version; source of truth for the LaTeX
rendering. Every number below traces to an on-disk artifact (provenance in §A). TODO
slots are marked `[TODO-*]` and enumerated in §B.**

**Target venue: Knowledge-Based Systems (user-confirmed 2026-07-16);** Neurocomputing
fallback. **Review model: SINGLE-anonymized (verified live)** — real byline, B6
companion citable normally (the old TODO-ANON is RESOLVED). Venue facts + desk-review
mitigation: `VENUE-KBS-FACTS-2026-07-16.md`; binding build rules: `REVIEW-RULES.md`.
NOTE (updated 07-17, submission state): main.tex is now the SOLE source of truth for
the whole manuscript — the 07-16/17 revision rounds (referee-feedback abstract reframe
+ caveat hoisting, ~245 words; §7/caption compression; consistency fixes; byline; both
GenAI declarations; live Data-availability links DOI 10.5281/zenodo.21405273 +
github.com/PeterPonyu/edit-federation-map) were applied in main.tex ONLY. This
canonical is a HISTORICAL design record as of the pre-reframe draft (its abstract and
cell counts are stale: 19 cells/six families vs the submitted 22/seven, 1–20B). Do NOT
port canonical→tex anymore; for any future revision, edit main.tex and treat this file
as background.

---

## Abstract (draft)

Locate-then-edit methods such as ROME install knowledge as rank-one weight updates,
making it natural to *federate* edits — merge many independently-authored updates into
one deployed model. Whether merged edits interfere, and whether interference is
predictable before merging, is unmeasured beyond small scales. We present the first
systematic operating map of rank-one edit federation: 19 model×layer cells across six
architecture families (Llama, Mistral, Qwen, Gemma, Phi, GPT-2; 0.5–14B), group sizes
g=2–100, three seeds per cell, with pre-registered evaluation gates. We find a
**two-regime structure**. In *high-gain* regimes (all Llama/Mistral cells; shallow-to-mid
layers elsewhere), merge cross-talk is destructive at every group size, and its magnitude
is predicted by key-geometry coherence (partial rank correlations up to 0.64). In
*low-gain* regimes (deep layers of Qwen and Phi; GPT-2-XL broadly), small-group
cross-talk is **constructive** — aligned interference *raises* the member edits' target
logits — and crosses over to destructive at an architecture-dependent group size. The
regime is screened by a single measurable scalar, the layer's perturbation *gain*
(logit response per unit relative cross-talk dose; rank correlation with constructive
fraction −0.81 across all 19 cells), is depth-gated within a model at fixed scale, and
is neither scale-specific nor family-specific. We rule out cosine sign-cancellation
mechanically (interference indices are identical under signed and absolute cosine;
key mass is non-negative in every model) and localize the flip to what aligned
cross-talk *does* at the readout rather than how much arrives (cross-talk magnitude is
geometry-tracked at ρ≈0.99 everywhere). We translate the map into a gain-screened,
geometry-ordered admission rule for edit federation and evaluate it retrospectively
across the map `[TODO-BENEFIT]`. All gates, thresholds, and predictions were frozen
before the corresponding runs; three intermediate interpretations falsified by our own
follow-up tests are reported as such.

---

## 1. Introduction

Deployed language models increasingly receive *post-hoc knowledge edits*: rank-one or
low-rank weight updates that install, correct, or delete individual facts without
retraining (ROME, MEMIT, AlphaEdit). As editing matures, the unit of deployment shifts
from *one edit* to *fleets of edits*: multiple teams author updates independently and a
maintainer must decide which to merge into the next model build. This is *edit
federation*, and it inherits a question the model-merging literature has studied only
for task vectors and fine-tuned adapters: **when do independently authored updates
interfere, and can interference be predicted from the updates alone, before merging?**

Prior work gives a monotone answer: overlap causes interference. Task-arithmetic
merging attributes damage to sign conflict and magnitude collision (TIES, DARE);
editing-composition methods enforce orthogonality between successive edits precisely
because overlap is assumed harmful (MEMIT-Merge, O-Edit, orthogonal sequential
editing); and interference predictors built from update geometry report stable,
non-inverting signs (Demystifying Mergeability). The nearest neighbor, MergeProbe,
adds one nonmonotonicity — near-parallel adapter updates over-amplify and hurt — but
its direction is uniformly "similarity hurts," measured on LoRA adapters with training
probes, without a scale or architecture axis.

We show the picture for rank-one knowledge edits is qualitatively richer.
Contributions:

1. **An operating map of edit federation** (19 model×layer cells, six architecture
   families, 0.5–14B, g=2–100, 3 seeds, pre-registered gates): where key-geometry
   coherence predicts merge interference, at which group sizes the signal is testable,
   and where damage saturates. To our knowledge this includes the first edit-federation
   measurements at 14B scale.
2. **A two-regime law.** High-gain regimes are destructive at all group sizes with
   geometry-predicted damage (over-amplification, consistent with MergeProbe's
   direction). Low-gain regimes are *constructive at small group sizes* — aligned
   cross-talk raises the member edits' target logits — with an architecture-dependent
   crossover. The constructive regime has, to our knowledge, no precedent in the
   merging or editing literature.
3. **A measurable screen.** A single scalar per model×layer — the perturbation gain,
   estimable from the same vectors the edits are built from — orders the regimes
   (Spearman −0.81 with constructive fraction across all 19 cells; prediction frozen
   before 9 of the 19 cells were run). The regime is depth-gated *within* a model at
   fixed scale (Phi-3.5 and Qwen-3B flip regime between 50% and 75% relative depth,
   with gain collapsing ~10×), is not scale-gated (GPT-2-XL at 1.5B is the most extreme
   constructive cell), and is not family-gated (Qwen and Phi and GPT-2 all exhibit it).
   We report explicitly what the screen does *not* explain: at matched gain,
   architecture residuals in constructiveness remain.
4. **Mechanistic localization.** The flip is not cosine sign-cancellation (signed and
   absolute interference indices are numerically identical everywhere; negative-cosine
   key mass is negligible in all 19 cells) and not differential geometry-tracking (the
   received cross-talk *norm* is geometry-predicted at ρ≈0.99 in every cell). Aligned
   cross-talk arrives everywhere (~90–97% of observations); what flips is its effect at
   the readout — a boost in low-gain regimes (ρ(dose, drop) −0.45..−0.73 at small g), a
   penalty in high-gain regimes (+0.32..+0.87) — consistent with over-amplification
   past an operating point in high-gain layers and small-signal linear reinforcement in
   low-gain layers.
5. **A deployable admission rule** for federation — gain-screen the target layer, then
   geometry-order candidate edits — with a retrospective benefit evaluation across the
   map `[TODO-BENEFIT]`, connected to a validated per-edit damage predictor (per-edit
   gate ρ=0.725 on held-out seeds).

Methodologically, every gate, threshold, and directional prediction was frozen in
writing before the corresponding runs, and the paper reports three intermediate
interpretations that our own follow-up tests refuted (a single dose-continuum; a
"sharp 7B→14B jump" that was an aggregation artifact; Qwen-specificity of the
constructive regime).

## 2. Related work

**Model merging and task arithmetic.** TIES (2306.01708) and DARE (2311.03099) treat
interference as sign conflict plus magnitude collision; Task Singular Vectors
(2412.00081) refines this per-layer. Interference *prediction* from update geometry:
Demystifying Mergeability (2601.22285) fits interpretable linear predictors (CLIP-only;
stable non-inverting signs); Will it Merge? (2601.06672) attributes mergeability to
base-model knowledge rather than geometry; MergeProbe (2606.19549) predicts set-level
interference from update alignment and adds the over-amplification nonmonotonicity
(near-parallel LoRA updates ~2× and hurt). None has a scale or architecture-family
axis; none reports a constructive regime. **Distinct from all of these, our units are
train-free rank-one edits** whose key/value structure gives interference a closed
per-edit form (§3), and our map spans six families to 14B.

**Knowledge-editing composition.** Batch and sequential editing assume overlap is
harmful and engineer it away: MEMIT-Merge (2502.07322) resolves identical-key
conflicts; O-Edit (2410.11469) and orthogonal sequential editing (2606.22627,
2601.07873) enforce orthogonality; an AlphaEdit reproduction (2606.26783) reports
positive co-directional damage accumulation. We measure, rather than assume, the sign
of overlap's effect — and find regimes where it reverses.

**Interference under aligned updates.** In multi-task learning, negative transfer can
be *worse* under highly consistent gradients (ForkMerge, 2301.12618; cf. PCGrad,
CAGrad): the cleanest precedent that alignment is not protective. In continual
learning, task similarity is double-edged. These literatures never index the effect's
sign by architecture, depth, or a measurable layer property.

**Architecture norm regimes.** Qwen2.5 models carry much larger embedding/activation
norms than Llama peers, compressing relative perturbations [CITATION RETRACTED 2026-07-16 — unverifiable ID; claim unsourced, dropped from the submission]. Our gain
screen gives this observation an operational form and an interference consequence,
while our GPT-2 and Phi results show the low-gain regime is not Qwen-specific.

**Relation to our companion mechanism work.** A companion line establishes that
*single-edit* collateral damage is key-geometry-predictable within a family and
causally removable, with a family-determined sign (positive Llama/Mistral/Gemma,
negative Qwen at every scale). The federation law studied here is distinct: its regime
is set by depth/gain rather than family, small-scale Qwen merging is
destructive-leaning while its damage law is already sign-inverted, and GPT-2 — outside
the atlas — exhibits the constructive regime. `[TODO-ANON: cite as anonymous companion
or fold minimal background into §3.]`

## 3. Preliminaries: rank-one edits and exact merge cross-talk

**Rank-one edits.** A ROME-style edit to a down-projection W installs
ΔW_a = r_a k_aᵀ / (k_aᵀ k_a), where k_a is the edit's key (subject representation at
the edited layer) and r_a its value residual. We use a native ROME implementation
(fp32 value optimization; n=200 CounterFact edits per cell; three seeds drawing
disjoint edit sets).

**Federation operator.** Merging a group G of edits deploys W + Σ_{b∈G} ΔW_b. The
value delivered at edit a's key is then r_a + d_a, with the **exact cross-talk**
d_a = Σ_{b≠a} r_b (k_bᵀ k_a)/(k_bᵀ k_b) — computable in closed form from the stored
vectors, no model forward needed.

**Outcome.** For each *observation* (edit a in group G), drop = logit_solo(a) −
logit_merged(a) at a's target token: drop > 0 means merging damaged the edit,
drop < 0 means merging *helped* it.

**Interference indices.** I_cos(a) = ‖k_a‖ Σ_b S_b |cos(k_b, k_a)| (geometry-weighted)
and I_mag(a) = ‖k_a‖ Σ_b S_b (magnitude-only scaffold); the geometry discriminant is
the partial Spearman ρ(I_cos, drop | I_mag), guarded by an own-magnitude partial. A
pre-registered gate (frozen before any multi-model run) declares a cell's geometry
signal real iff the partial exceeds +0.30 in ≥2 seeds at ≥2 group sizes, with
negligibility, saturation (argmax-loss > threshold), and coherence eligibility rules.

**Dose and gain.** The dimensionless *dose* of observation a is
(d_a·r_a)/‖r_a‖² — received cross-talk relative to the edit's own delivery. A cell's
**gain** is the median of |drop|/dose over positive-dose observations (g≤20 pooled):
the logit response per unit relative dose, measuring how strongly the layer transmits
value-space perturbations to the readout.

## 4. The operating map under the pre-registered gate

**Reference cell (Llama-3.2-1B, L12 = 75% depth).** The gate PASSES: geometry predicts
merge damage at g=2 (partials 0.488/0.444/0.476 across seeds), g=3
(0.418/0.479/0.557), g=5 (0.316/0.305/0.340); g=10 fails coherence (transition band);
g≥20 saturates (argmax-loss > 0.85). **Two distinct boundaries** (binding): the
*geometry-valid* boundary g≤5, and the *damage-still-gradated* boundary g=10 — the law
must never be quoted as holding to g=10.

**Width/scale series (frozen prereg; boundary = max qualifying g in {2,3,5,10,20}).**

| cell | verdict | qualifying window | boundary |
|---|---|---|---|
| Llama-3.2-1B L12 | PASS | {2,3,5} | 5 |
| Llama-3.1-8B L24 | PASS | {2..20} full | 20 |
| Mistral-7B L24 | PASS | {2..20} full | 20 |
| Qwen2.5-1.5B L21 | PASS | {10,20} | 20 |
| Qwen2.5-7B L21 | PASS | {5,10,20} | 20 |
| Qwen2.5-14B L36 | INCONCLUSIVE (0 testable cells under the positive gate) | — | undefined |
| Mistral-Nemo-12B L30 | MIXED | {20} | 20 |
| gemma-2-9b L31 | INCONCLUSIVE (coherence) | — | undefined |
| gpt-neox-20b L33 | INCONCLUSIVE (negligible damage) | — | undefined |

Width, not family, explains the original small-vs-mid scale contrast on the positive
side: Llama-8B's full window matches Mistral-7B's (H-Llama). Qwen's window edge moves
*down* with width (1.5B {10,20} → 7B {5,10,20}) — opposite to Llama — a family
signature on the window's lower edge. The Qwen-14B INCONCLUSIVE is not absence of
signal; it is the gate's structural blindness to what §5 shows. **Scaling Mistral to
12B reproduces the deep-Qwen window shape in a second family**: Mistral-Nemo-12B's
small-g partials sit near zero and grow to a qualifying window at g=20 — the
"window-shifts-up" signature that at 7B appeared Qwen-specific. gemma-2-9b fails the
coherence rule (the known gemma anomaly), and at gpt-neox-20b merges barely perturb
targets at all (median drops 0.003–0.04), so the negligibility rule correctly declares
the cell untestable — both are regime information in their own right (§5.3).

## 5. The two-regime law

### 5.1 The 14B anomaly and its resolution

Under a *signed* re-analysis (the frozen gate thresholds only positive partials), the
Qwen-14B L36 cell shows strongly negative geometry partials at small g
(g=2: −0.432/−0.419/−0.523 across seeds). Decomposition shows this is not geometry
failing but the outcome flipping: **81–88% of merge observations at 14B have drop < 0
— merging raises the member edits' target logits — and higher geometry coherence
predicts a *larger boost*.** Saturation is ruled out (argmax-loss ≤ 0.13); cosine
sign-cancellation is ruled out mechanically (signed, positive-part, and absolute-cosine
indices are numerically identical in every cell; Llama-1B has literally zero
negative-cosine mass); and the *amount* of cross-talk an edit receives is
geometry-tracked everywhere (ρ(I_cos, ‖d_a‖) = 0.95–1.00 in all 19 cells). What flips
is what aligned cross-talk *does*.

### 5.2 Alignment is universal; its effect is regime-dependent

Received cross-talk aligns with the edit's own residual direction in *every* model
(cos(d_a, r_a) > 0 for ~90–97% of observations — a consequence of the shared
anisotropy cone of keys and values). The discriminant is the effect of aligned dose:
at Qwen-14B g=2, ρ(dose, drop) = −0.73 (more aligned cross-talk → larger gain — the
naive linear prediction); at Llama-1B L12 and Mistral-7B, +0.40 and +0.32 (more
aligned cross-talk → more damage — over-amplification past the operating point).
Bundles do **not** collapse onto a single dose-response curve: at matched dimensionless
dose, the same relative perturbation produces ~60× different logit effects across
models. The missing variable is the layer's gain.

### 5.3 The gain screen (19 cells, six families; predictions frozen pre-run)

| cell | gain | frac(drop<0) | ρ(dose,drop) |
|---|---|---|---|
| Llama-1B L8 | 64.5 | 0.041 | +0.70 |
| Qwen-3B L18 (50%) | 37.2 | 0.332 | +0.47 |
| Llama-3B L21 | 32.0 | 0.022 | +0.87 |
| Llama-1B L14 | 28.7 | 0.013 | +0.78 |
| Mistral-7B L24 | 28.3 | 0.278 | +0.60 |
| Qwen-1.5B L14 (50%) | 28.1 | 0.242 | +0.66 |
| Llama-1B L12 | 27.2 | 0.025 | +0.78 |
| Llama-8B L24 | 22.2 | 0.127 | +0.82 |
| gemma-2b L13 (50%) | 20.0 | 0.287 | +0.25 |
| Phi-3.5 L16 (50%) | 16.9 | 0.251 | +0.36 |
| gemma-2b L19 | 12.5 | 0.115 | +0.57 |
| Qwen-3B L27 | 3.6 | 0.430 | +0.50 |
| Qwen-7B L21 | 3.5 | 0.349 | +0.40 |
| GPT-2-XL L24 (50%) | 3.4 | 0.711 | −0.10 |
| Qwen-1.5B L21 | 3.3 | 0.420 | +0.38 |
| **Qwen-14B L36** | **3.1** | **0.836** | **−0.34** |
| Phi-3.5 L24 | 1.9 | 0.556 | +0.12 |
| Qwen-1.5B L24 | 1.1 | 0.471 | +0.23 |
| **GPT-2-XL L36** | **0.84** | **0.725** | **−0.45** |
| gemma-2-9b L31 | 12.5 | 0.119 | +0.53 |
| **Mistral-Nemo-12B L30** | **10.7** | **0.423** | **+0.41** |
| **gpt-neox-20b L33** | **0.103** | **0.634** | **−0.12** |

Spearman(gain, frac(drop<0)) = **−0.82** across all 22 cells (seven families,
0.5B–20B); the ≤ −0.7 threshold and every per-cell directional prediction were frozen
in writing before the corresponding cells ran (prediction document in supplementary
material; intermediate waves: −0.855 at 14 cells, −0.81 at 19).

**Scale lowers gain within a family; the flip follows gain, not scale.** Scaling
Mistral from 7B to 12B drops gain three-fold (28.3 → 10.7) and moves the cell to the
constructive boundary (constructive fraction 0.278 → 0.423) exactly where the ordering
places it, while reproducing the deep-Qwen "window-shifts-up" profile (§4). gemma's
gain is scale-stable (12.5 at both 2B and 9B — a family signature). The pre-modern
line goes deepest: at gpt-neox-20b, gain collapses to 0.103 and merge cross-talk
becomes both negligible in magnitude (median drops 0.003–0.04) and
majority-constructive — the endpoint of the low-gain regime.

**Depth gates the regime at fixed scale.** Phi-3.5: gain 16.9 → 1.9 and g=2
ρ(proj,drop) +0.14 → −0.35 between 50% and 75% depth. Qwen-3B: 37.2 → 3.6 and +0.13 →
−0.12. Same model, same scale — the regime is a property of the layer.

**Scale is not necessary; family is not the carrier.** GPT-2-XL (1.5B, pre-modern
architecture) has the lowest measured gain and the most extreme constructive profile —
ρ(proj, drop) negative at *every* g with constructive fraction *rising* with g
(0.59→0.70), the only cell where it rises. Qwen-14B and GPT-2-XL-deep are the two
cells whose crossover lies beyond g=20.

**g-resolved profiles.** Every low-gain cell is constructive at g=2 with the
aligned-boost signature (negative ρ(proj,drop)); crossover to destructive occurs at
g≈10 (Qwen-1.5B deep), g≈5 (Qwen-3B/7B, Phi-3.5 deep), beyond g=20 (Qwen-14B,
GPT-2-XL deep). Every high-gain cell is destructive with positive ρ(proj,drop) from
g=2. Pooled statistics hide this: a cell whose crossover sits below the pooling window
looks "destructive on average" — the source of an intermediate "sharp 7B→14B jump"
reading that the g-resolved analysis retired (§7).

**What the screen does not explain (reported, not smoothed).** At matched gain (~3.4),
GPT-2-XL is far more constructive than Qwen deep layers (0.711 vs 0.35–0.43), and
crossover-g is not a clean function of gain (Qwen-1.5B L24, gain 1.1, crosses at ~10;
Qwen-14B, gain 3.1, does not cross by 20). Gain is a rank-level *screen*, not a
sufficient statistic; architecture residuals are real. The Qwen small-g boost is also
non-monotone in scale (1.5B stronger than 3B/7B) — unexplained.

## 5.4 The law is editor-general, and editor choice dominates federation safety

All results above federate ROME edits. A pre-registered editor-generality wave
(predictions frozen before any run; an exact-equivalence anchor ties the extended
pipeline to the ROME cells at machine precision, and a ΔW-fidelity gate verifies the
re-derived updates against each real editor's installed weights on-model before any
science cell runs) extends the map to MEMIT-style multi-layer edits (identity
covariance; per-layer rank-one spread over four layers) and AlphaEdit (null-space
projected) at one high-gain cell (Llama-1B L12) and one low-gain cell (Qwen-1.5B L21),
three seeds, g=2–20:

| cell | editor | verdict | median drop g=2 / g=20 | small-g partials |
|---|---|---|---|---|
| Llama-1B L12 | ROME (canonical) | PASS | 1.97 / 17.6 | +0.44..+0.49 |
| Llama-1B L12 | MEMIT-style | PASS | 1.57 / 19.2 | **+0.56..+0.71** |
| Llama-1B L12 | AlphaEdit | PASS | **0.10 / 1.77** | +0.05..+0.28 |
| Qwen-1.5B L21 | ROME (canonical) | PASS {10,20} | 0.096 / 1.65 | −0.04..+0.02 |
| Qwen-1.5B L21 | MEMIT-style | MIXED | 0.119 / 2.79 | −0.09..+0.01 |
| Qwen-1.5B L21 | AlphaEdit | INCONCLUSIVE (negligible damage) | **0.015 / 0.15** | −0.03..−0.01 |
| Llama-8B L24 | ROME (canonical) | PASS {2..20} | 0.25 / 10.9* | +0.29..+0.44 |
| Llama-8B L24 | MEMIT-style | PASS | 0.31 / 16.8 | **+0.39..+0.49** |
| Llama-8B L24 | AlphaEdit | MIXED | **0.038 / 0.77** | +0.06..+0.22 |
| Qwen-14B L36 | AlphaEdit | INCONCLUSIVE (negligible damage) | **0.014 / 0.12** | −0.17..−0.33 |

*Llama-8B ROME med-drop values from the canonical scale cell, seed 0, for
comparability with the editor cells (all 8B rows are seed-0 displays; the canonical
cell's 3-seed statistics appear in §4). [CORRECTED 2026-07-16: an earlier revision
showed 0.14/5.6 here — those were Mistral-7B's values, a transcription error.]
The 8B extension replicates both editor findings at scale: MEMIT-style keeps the
destructive regime with a stronger small-g geometry signal than ROME (as at 1B), and
**AlphaEdit's federation-safety advantage holds at 8B — median damage 6–14× below
ROME and up to 22× below MEMIT across group sizes** (0.038 vs 0.25/0.31 at g=2;
0.77 vs 10.9/16.8 at g=20). At the constructive-regime extreme (Qwen-14B), AlphaEdit
suppresses cross-talk magnitude about five-fold in BOTH directions (median 0.014 vs
ROME's 0.072 at g=2) while the constructive regime itself is preserved (constructive
fraction 0.837 vs ROME's 0.836) — the regime belongs to the layer, not the editor —
and, consistent with the operating-point account, the reduced dose keeps the cell in
the small-signal constructive window past g=20 (geometry partials remain negative at
g=20, where ROME had already reverted positive). For a maintainer: AlphaEdit shrinks
the magnitude of federation cross-talk everywhere, damping harm in destructive regimes
and damping — but not removing — the reinforcement in constructive ones.

Three findings. **(1) The two-regime structure is editor-general:** every editor cell
lands in the regime its layer's gain predicts, and the combined ordering across all 31
cells (22 ROME + 9 editor- and dataset-varied) is Spearman(gain, constructive fraction)
= **−0.820** (p = 1.7e-08; frozen threshold ≤ −0.7, first evaluated at 23 cells where it
gave −0.826 — the ordering is stable under every subsequent extension). AlphaEdit at the Qwen deep layer exhibits the
full constructive signature (constructive fraction 0.639, ρ(dose, drop) = −0.30) — a
third editor in the low-gain regime. **(2) Editor choice dominates federation safety:**
AlphaEdit's null-space projection — designed to protect preserved knowledge in
sequential editing — also suppresses federation cross-talk by roughly an order of
magnitude (6–20×) in BOTH regimes, at every group size. For a maintainer this is the
single largest safety lever in the map, ahead of group size and admission ordering.
**(3) MEMIT-style multi-layer spread does not buy safety:** damage is at or slightly
above ROME's at matched cells, with an even stronger small-g geometry signal — the
multi-layer distribution spreads, but does not shrink, the interference surface.

**Dataset generality.** ROME federation on zsRE replicates the CounterFact cells at
both reference layers: Llama-1B L12 zsRE is PASS with partials tracking the
CounterFact values within seed noise at every group size (g=2: +0.40..+0.58 vs
+0.44..+0.49) and matching damage magnitudes; Qwen-1.5B L21 zsRE is PASS with the same
low-gain window shape. The operating map is not an artifact of one fact distribution.

## 6. A gain-screened admission rule for edit federation

The map converts into a two-stage admission rule for a maintainer deciding which of N
candidate edits to co-deploy at a target layer:

1. **Screen the layer.** Estimate gain from a small calibration set of edits (the same
   vectors the edits are built from; ~91 s of GPU time per cell in our setting). High
   gain ⇒ destructive regime: admit conservatively, geometry-ordered, within the
   geometry-valid boundary (g≤5 at the reference cell). Low gain ⇒ constructive
   small-g regime: small merge groups are safe and can even reinforce edits, but the
   architecture-dependent crossover bounds group size; the positive-gate boundary
   table (§4) gives per-cell limits.
2. **Order candidates by predicted received interference.** Within a group, each
   edit's exact expected cross-talk d_a is closed-form; rank and cap by I_cos. This
   connects to a validated per-edit damage predictor (held-out per-edit gate ρ=0.725 at
   the reference layer) and a retrospective capacity-constrained routing evaluation
   (oracle efficiency η=0.842, +0.32 over random, 3/3 seeds; retrospective and
   zero-parameter — no fitted thresholds).

**Retrospective benefit table (computed under the spec frozen above; code
`experiments/rg_admission_benefit.py`, artifact
`results/merging/RG_admission_benefit_20260715.json`).** Per (cell, g≤5), admission =
bottom-q I_cos observations at fixed group composition; primary metric = mean signed
drop among admitted edits vs the same-budget random baseline (exactly the overall
mean); magnitude-only ordering (bottom-q I_mag) as the second baseline; seeds
separated; no fitted thresholds. Small-g regime aggregates:

| regime | budget q | benefit (geometry) | by seed | benefit (magnitude-only) |
|---|---|---|---|---|
| high-gain | 25% | **+0.716 logits/edit** | 0.813 / 0.678 / 0.658 | +0.443 |
| high-gain | 50% | +0.534 | 0.669 / 0.467 / 0.467 | +0.289 |
| low-gain | 25% | +0.013 | 0.017 / 0.021 / 0.003 | +0.001 |
| low-gain | 50% | +0.020 | 0.026 / 0.024 / 0.010 | +0.006 |

Geometry-ordered admission avoids substantial damage exactly where damage exists
(high-gain regimes), outperforms magnitude-only ordering by ~1.6–1.8×, and is neutral
in low-gain regimes — where the two-regime law already says small groups are safe. The
rule's value is therefore concentrated, and the gain screen tells the maintainer in
advance whether ordering will pay.

## 7. Falsified intermediate interpretations (methods transparency)

1. *Single dose-continuum*: refuted by the master-curve test (no collapse at matched
   dimensionless dose; ~60× cross-model effect differences at the same dose).
2. *"Sharp 7B→14B jump"*: an aggregation artifact of pooling g≤20 — retired by the
   g-resolved profiles (§5.3).
3. *Qwen-specificity*: refuted by Phi-3.5 (majority-constructive at 3.8B) and GPT-2-XL
   (most extreme constructive cell at 1.5B).
4. *"Gain is the carrier"*: weakened to "gain is a rank-level screen" by the
   matched-gain architecture residuals (§5.3).

## 8. Limitations and threats to validity

- **Editor generality.** Demonstrated at two reference cells (§5.4: MEMIT-style +
  AlphaEdit, both regimes, pre-registered); the remaining gap is editor-varied cells at
  ≥8B scale `[E2: llama-8B editor cells staged]` and true-covariance MEMIT (our MEMIT
  arm uses identity covariance — "MEMIT-style multi-layer spread", named as such
  throughout).
- **Dataset.** CounterFact primary; zsRE replication at both reference cells (§5.4).
  MQuAKE/RippleEdits-style streams are exercised in the companion decision-framework
  work, not here.
- **Scale ceiling.** One ≥14B family (Qwen). `[TODO-C-CELL: non-Qwen 12–14B cell —
  Mistral-Nemo-12B — anchors the scale axis; ~2–3 GPU-h.]`
- **Retrospective evaluations.** The admission rule's benefit table `[TODO-BENEFIT]`
  and the routing η are retrospective; a confirmatory prospective arm is future work.
- **Gain estimation.** Gain is measured from merge observations here; a cheaper
  edit-free estimator (e.g., from base-model norm statistics) is future work.
- **Outcome metric.** Target-logit drop on the edited fact; downstream/global damage
  under federation is inherited from the companion damage-law work, not re-measured
  per group size.

## 9. Conclusion

Rank-one edit federation is governed by a two-regime, gain-screened law: destructive
geometry-predicted over-amplification in high-gain layers; constructive small-group
reinforcement with architecture-dependent crossover in low-gain layers. The regimes
are depth-gated within a model, present from 1.5B to 14B and across six architecture
families, and screened — though not fully explained — by one measurable scalar. For
practitioners this yields an actionable admission rule; for the science of editing it
overturns the assumption, inherited from task-vector merging, that overlap between
updates is uniformly harmful.

---

## §A. Provenance (numbers → artifacts; NOT part of the submission)

| claim | artifact |
|---|---|
| gate/partials, boundary table | `results/merging/RG_operating_curve_table*.json`, `M0_killgate_table.json` |
| 19-cell gain table, ordering −0.811 | `results/merging/RG_gain_law_20260715.json` |
| signed decomposition, frac(drop<0), sign-cancellation exclusion, ρ≈0.99 x-check | `results/merging/RG_signed_reanalysis_20260715.json` (156+ cells reproduce canonical tables) |
| alignment/ρ(proj,drop)/g-profiles | `results/merging/RG_crossterm_alignment_20260715.json` |
| frozen predictions | `docs/plans/PREDICTIONS-GAIN-WAVE-2026-07-15.md` (+2 addenda) |
| lit positions / claim shape | `docs/findings/SCOUT-MERGING-SIGN-INVERSION-2026-07-15.md` |
| narrative + corrections | `docs/findings/findings-RG-SIGNED-REANALYSIS-2026-07-15.md` |
| η=0.842 routing (retrospective) | `results/D3_benefit_predictor_eval.json`, `docs/plans/ANALYSIS-D4-ROUTING-E0-20260714.md` |
| width series | memory `p2-wave-results-20260715` + per-cell tables |

## §B. TODO register (drafting gaps, ranked)

1. ~~`[TODO-BENEFIT]`~~ **DONE 2026-07-15** — table in §6;
   `results/merging/RG_admission_benefit_20260715.json`.
2. ~~`[TODO-EDITORS]`~~ **DONE 2026-07-16** — §5.4: MEMIT-style + AlphaEdit at both
   reference cells, all 4 prereg predictions confirmed; 23-cell combined ordering
   −0.826. GRACE paragraph still to add in prose port (damage≡0 codebook ⇒ no weight
   interference by construction). Remaining: 8B editor cells (E2).
3. ~~`[TODO-ZSRE]`~~ **DONE 2026-07-16** — §5.4 dataset-generality block (PASS at both
   cells, partials track cf within noise).
4. `[TODO-C-CELL]` — non-Qwen 12–14B cell (E2 box wave, ~¥40–80 with the 8B editor
   cells; Mistral-Nemo-Base-2407 download ~24 GB, user gate).
5. `[TODO-ANON]` — companion-paper citation strategy under double-anonymity.
6. Figures: (F1) gain vs constructive fraction scatter; (F2) g-resolved profiles by
   regime; (F3) depth-gate bars (Phi/Qwen 50% vs 75%); (F4) admission-benefit bars.
   Pipeline: R/ggplot2 → tikzDevice (house standard; % SOURCE provenance headers);
   pdflatex render; VISUAL audit of rendered pages (never trust logs alone).
