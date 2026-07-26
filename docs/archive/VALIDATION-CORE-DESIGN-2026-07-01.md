# Fission validation-core design + journal-adequacy audit — 2026-07-01

Scope: design the scientific **validation core** for every planned direction (B6 + P2/P3/P4),
and audit whether the current designs are adequate for **SCIE Q1/Q2 CS journals**
(Neurocomputing, Expert Systems with Applications, Knowledge-Based Systems, IEEE TNNLS).
Grounded in the *actual* result JSONs on disk, not the plan docs' aspirations.

> **Reality check first.** Only ONE of the four directions currently has a defensible result.
> The others range from "partial on fallback data" to "failing its own kill-gate in real data"
> to "synthetic-only scaffold." This document says so plainly per direction and designs the
> core each needs to become journal-grade.

---

## Part 0 — The venue mismatch that must be resolved first

The B6 skeleton (`B6-PAPER-SKELETON-2026-07-01.md` §venue) explicitly says: **"NOT the Elsevier
journals (wrong reviewer pool for a ROME/AlphaEdit mechanism paper)"** and targets TMLR /
BlackboxNLP. The workspace `CLAUDE.md` and the current request instead target
**Neurocomputing / ESWA / KBS / TNNLS**. These pull in opposite directions and the design must
pick one, because *the validation core differs by venue*:

| Venue class | What the reviewer pool rewards | Consequence for the core |
|---|---|---|
| **ML/NLP (TMLR, *ACL)** | mechanism, novelty, honesty about scope; a single clean model is OK | current B6 core is ~sufficient |
| **KBS / Neurocomputing** | mechanism **+ a reusable method/system + multi-setting empirics** | need a deployable artifact (D3 routing) + ≥3 architectures + ≥2 datasets, all on the clean metric |
| **ESWA** | an **application/system** solving a real problem with strong evaluation | mechanism alone is a poor fit; must be framed as a *safe-editing tool* with a benchmarked pipeline |
| **TNNLS** | learning-theory / NN mechanism with rigor + breadth | closest to B6's strength; wants breadth + theory, tolerates less "application" |

**Recommendation.** Target **TNNLS or KBS** for B6 (the mechanism + the D3 routing artifact
gives KBS its required "knowledge-based method"). Treat ESWA as the home for a *P-series
applied* paper, not B6. Everywhere below, "journal-grade" means the KBS/TNNLS bar: **≥3
architectures, ≥2 datasets, ≥3 seeds, effect sizes with CIs, a permutation/null control, and a
deployable method with a head-to-head baseline.** That bar is the through-line of every core
design below.

---

## Part 1 — Cross-cutting validation-core principles (apply to all four)

These are the standing requirements a Q1 CS reviewer will check; the current branches satisfy
them unevenly (B6: mostly; P2/P3/P4: not yet).

1. **A falsifiable primary hypothesis with a pre-registered kill-gate** (effect-size threshold +
   significance). B6 has this (`analyze_matrices.py` verdict); P4 has it and **fails** it; P2/P3
   have the machinery but no real data through it.
2. **The correct exchangeable unit for the null.** Not the flattened pair count. B6 now uses the
   edit-level permutation null + edit-cluster bootstrap (added today). P2 uses problem-level
   cluster bootstrap (correct). P4 uses a temperature-resampling null (correct *design*). P3 uses
   a within-model item permutation (correct). This is the one thing all four got right in *design*.
3. **A head-to-head against the obvious competitor**, on the *same* metric — not a strawman.
   B6: norm-growth + gradsim + lexical/SBERT (done, all layers, today). P4: cross-model vs
   resampling disagreement (done — and cross loses). P3: lineage vs architecture (done on
   fallback data). P2: pre-RL D vs a trivial length baseline (**missing**).
4. **Effect sizes with CIs, reported per setting, including the near-zero settings** (do not hide
   the nulls — Qwen's ρ≈0 is *evidence for* the mechanism, not a failure to bury).
5. **Multi-setting generality**: ≥3 model architectures and ≥2 datasets for any "law"/"predictor"
   claim. B6 is single-family (Llama) on the *clean* metric; the cross-arch sweeps exist only on
   the inflated flat metric and at single seed. This is the biggest shared gap.
6. **Provenance + reproducibility**: library versions, seeds, git-clean result JSONs (added to
   B6's killgate today; P2/P3/P4 have `_meta` timestamps but not lib versions).

---

## Part 2 — B6 (knowledge-editing mechanism): the mature core, and what closes it

### 2.1 Current evidence (verified from disk today)
- **G1 within-probe gate PASSES at all 4 layers under the strict edit-level null** (re-run today):
  L8 0.395 / L10 0.534 / L12 0.602 / L14 0.301; edit-level perm-p = 0.001 (floor), z = 6.5–16.4;
  edit-cluster CIs all clearly > 0 (weakest L14 [0.19, 0.34]). This is genuinely solid.
- **C1 S×C decomposition** is analytic and correct (Eq. 3 in the draft) and now has a receipt
  (`C1_mechanism_sc_table.json`: mean_S ≈ 22.9–24.7 for Llama).
- **C4 causal** (AlphaEdit) had a **circularity flaw fixed today**: projector was fit on the same
  probes it protected. Holdout/generic projector sources are now implemented + staged (E6).
- **The honest complication found today (must reframe around it):** the S×C surrogate
  (norm-growth × cosine) **beats raw key-cosine at every layer** (L12 0.602 vs 0.677; L14 0.301
  vs 0.504). This is *predicted by Eq. 3* and thus supports the mechanism — but the paper cannot
  lead with "raw key-cosine predicts damage."

### 2.2 The cross-architecture threat (the single biggest B6 risk)
The sweep numbers (single-seed, *flat/inflated* metric) already contain a warning the plan docs
underweight:

| model | layer | flat ρ(cos) | flat ρ(ng) | mean_dmg | mean_cos | edit_ok |
|---|---|---|---|---|---|---|
| Llama-3B | L14 | **0.300** | 0.130 | 0.89 | 0.222 | 1.00 |
| gemma-2-2b | L13 | 0.120 | −0.113 | **3.20** | **0.519** | 0.913 |
| Qwen-0.5B | L12 | 0.098 | 0.137 | 0.17 | 0.332 | 0.993 |
| Qwen-1.5B | L14 | −0.141 | 0.028 | −0.10 | 0.186 | 0.993 |
| Qwen-3B | L18 | −0.107 | 0.057 | −0.04 | 0.150 | 1.00 |
| Phi-3.5 | L16 | −0.002 | 0.137 | 0.08 | 0.166 | 0.993 |

**The problem:** gemma-2-2b has the **highest** key-cosine (0.52) AND real damage (3.2) — exactly
the regime where the "high-cosine ⇒ geometry-predictable" story predicts a strong ρ — yet shows
only flat ρ=0.12. If that holds on the clean within-probe metric, the "key-cosine-gated law"
framing is dead and B6 collapses to "a Llama-family phenomenon." Conversely, if the *within-probe*
gemma ρ is strong (the flat number is known to understate it — cf. Llama flat 0.51 vs within 0.60),
the law upgrades to cross-architecture. **This is the experiment that decides B6's scope and title.**

### 2.3 B6 validation core — the experiments that make it journal-grade

**Tier A (decisive; do first, all staged in `run_deep_until1900.sh`):**
- **A1 — cross-arch on the CLEAN metric, ≥3 seeds.** Re-run gemma-2-2b L13, Phi-3.5 L16, Qwen
  {0.5B,1.5B,3B} with `--save_matrices` at **3 seeds** (currently seed-0 flat only), then
  `analyze_matrices.py` within-probe + strict null. Decision rule: gemma within-probe ρ ≥ 0.3
  ⇒ "cross-architecture geometry law" (KBS/TNNLS-grade); gemma ρ ≈ 0 at mean_cos 0.52 ⇒ retitle
  to the *dissociation* result (still publishable, weaker). ~4 GPU-hr.
- **A2 — C4 honest causal (E6).** Run holdout + generic AlphaEdit at L8/L12 (staged today). Report
  `C4_causal_holdout_table.json` as primary; the probes-fit table becomes a by-construction
  reference. Decision rule: monotone damage-removed by cosine quartile **survives** on holdout ⇒
  causal claim stands. ~72 GPU-min.
- **A3 — S×C as the headline predictor.** Re-run C1/C4 through the upgraded midrank Spearman;
  present S×C (not raw cosine) as the predictor, with raw-cosine and norm-growth as the two
  factored special cases. Pure CPU.

**Tier B (breadth + the KBS "method" hook):**
- **B1 — second dataset (zsRE), 3 seeds** at the peak layer (staged E5). Closes "CounterFact-only."
- **B2 — D3 geometry-gated routing as a deployable artifact.** `geometry_router.py` exists; make
  it a *benchmarked method*: on a held-out edit stream, route high-cos→AlphaEdit / low-cos→ROME,
  and report (collateral damage ↓, edit-success held, compute saved) vs always-ROME and
  always-AlphaEdit. **This is what turns a mechanism paper into a KBS/TNNLS paper.** ~2 GPU-hr.
- **B3 — FT-L structural null on the clean metric + KL-FT control (D1, staged E3).** Makes the
  ROME-vs-FT dissociation defensible on matched stats and pre-empts "FT is safer" reviewers.

**Tier C (mechanism receipts, cheap, strengthen C1):**
- Effective-rank/SVD of ΔW_FT vs ΔW_ROME (asserted, not measured).
- Qwen residual-norm S dump vs Llama (the "S≈0 distributed storage" claim — the sweep already
  hints it: Qwen mean_dmg ≈ 0 at matched cosine).

### 2.4 B6 adequacy verdict
- **For TMLR/*ACL:** already ~adequate after A1–A2.
- **For KBS/TNNLS:** adequate **only with B2 (routing artifact) + A1 (≥3 arch clean) + B1 (2nd
  dataset).** Without the deployable method, KBS/ESWA will desk-reject as "analysis, not a method."
- **For ESWA:** not a fit; do not target.

---

## Part 3 — P4 (temporal-UQ conformal committee): FAILING in real data — salvage or pivot

### 3.1 What the real runs actually show (ETTh2/ETTm1/ETTm2, ollama, 60 windows each)
- **Core hypothesis rejected.** `gate_pass: False` on **all three** datasets. Cross-model
  disagreement→error Spearman ρ = −0.19 / 0.12 / 0.05 (p = 0.15 / 0.37 / 0.68). The
  cross-vs-resampling delta CI straddles zero everywhere ([−0.46, 0.32], [−0.05, 0.63],
  [−0.23, 0.56]). **Cross-model disagreement does not predict forecast error better than a
  single model's temperature resampling.** That is the whole thesis, and it is currently dead.
- **Calibration broken.** PICP = 0.33 / 0.36 / 0.58 against a **0.90** target — severe
  under-coverage; the "conformal" guarantee is not being delivered.
- **Numerics broken.** CRPS = 1.17e148 / 133.9 / 1.32e106 — LLMTime occasionally decodes a
  catastrophic-magnitude number and the error/CRPS path doesn't reject it (the `isfinite` masks
  in `calibrate.py` don't catch a *finite but astronomical* decode).

### 3.2 P4 core redesign (three independent fixes, then a re-gate)
This direction is *not* publishable as-is anywhere. To salvage it needs, in order:

1. **Fix the LLMTime decode robustness (prerequisite, CPU/GPU cheap).** Reject/winsorize decoded
   values outside a data-scaled band (e.g. median ± k·MAD of the context); log the rejection rate.
   Until CRPS is finite the calibration numbers are meaningless. **Re-run the 3 real datasets.**
2. **Fix conformal calibration (the coverage failure).** PICP 0.33 at nominal 0.90 means the
   interval construction is wrong, not just noisy. Use **split-conformal** calibration of the
   disagreement→width map on a held-out window block; PICP must land near 0.90 by construction.
   If PICP is right and the intervals are wide, that is an honest (publishable) negative-coverage
   story about LLM point forecasts.
3. **Re-test the central hypothesis with the fixes + more power.** n=60 windows is thin; extend
   to the full ETT panel (ETTh1 + weather/electricity) and ≥5 rolling blocks. **Pre-registered
   re-gate:** cross-model disagreement beats resampling by Δρ ≥ 0.15 with a CI excluding 0.
   - **If it passes:** genuine result → IJF/Neurocomputing viable.
   - **If it fails again (likely, given 3/3 current fails):** pivot the paper to the *honest
     negative*: "cross-model committee disagreement is **not** a usable UQ signal for LLM time-
     series forecasting; here is why (models agree on the same wrong answer — correlated errors),
     and here is the resampling/conformal baseline that *does* calibrate." A rigorous negative +
     a working baseline is publishable in Neurocomputing/ESWA; the current silent failure is not.

### 3.3 P4 adequacy verdict
**Not adequate for any journal today** (fails own gate, broken calibration, exploding CRPS).
Salvageable only via the 3-step redesign; realistically it becomes a **negative-result + working-
baseline** paper unless step 3 surprises. Do not invest GPU in scaling P4 until steps 1–2 make the
metrics finite and calibrated. Estimated: ~1 GPU-day for the full re-run after the CPU-side fixes.

---

## Part 4 — P3 (agentic indirect prompt injection): partial, on fallback data

### 4.1 State
- Real structure exists: 9-model panel with `family/architecture/lineage/group` tags, an M×K
  success matrix, per-model ASR, and a **lineage-vs-architecture contrast with a within-model
  item-permutation null** (correct null design). Ran under both mock and ollama.
- **But the data source fell back:** every `source_status` shows `data_found: false,
  used_fallback: true` ("gated data not downloaded"). So the 30 scenarios are the harness's
  fallback constructions, **not** a recognized IPI benchmark. n=30 is small.
- The scientific question — *does IPI vulnerability track model lineage (shared pretraining) more
  than architecture?* — is genuinely interesting and not obviously scooped.

### 4.2 P3 validation core
1. **Get a real benchmark.** InjecAgent / AgentDojo / a de-gated xLAM subset. Requires a download
   (ask first per standing rule). Without real scenarios the contrast is on synthetic strings and
   any reviewer kills it. This is the gating prerequisite.
2. **Power up:** ≥150 scenarios (n=30 gives the permutation test almost no resolution), ≥3
   lineage families with matched-architecture controls (the design already tags these).
3. **Primary statistic:** lineage-pair success-correlation vs architecture-pair, with the
   within-model item-permutation null (already implemented) + a cluster bootstrap over scenarios.
4. **Defensive half (what makes it TIFS/ESWA-shaped):** a mitigation (tool-whitelist / memory
   isolation / injected-instruction detector) with a before/after ASR table. A pure vulnerability-
   correlation finding is thin; the attack→defense pair is the publishable unit.

### 4.3 P3 adequacy verdict
**Not adequate today** (fallback data, n=30, no defense). The design's *statistics* are sound; the
*evidence* is placeholder. IEEE TIFS (CCF-A) is a very high bar — realistically retarget to
**ESWA / Neurocomputing** (security-application framing with the attack+defense pipeline) unless
the real-benchmark result is exceptionally strong. Needs the download + a defense module + Ollama
GPU time (~1–2 GPU-days for the model panel × scenarios).

---

## Part 5 — P2 (pre-RL overthinking diagnostic): scaffold-only, zero real evidence

### 5.1 State
- The diagnostic statistic is **well-designed**: D_within (difficulty-controlled wrong/right length
  ratio) + problem-level cluster bootstrap — the right unit, numpy-only, clean.
- **But the only results are SYNTHETIC** (`SYNTH-bias` D=1.61, `SYNTH-control` D=1.01) — these
  merely confirm the estimator recovers a planted bias. The cross-checkpoint correlation has
  **n=2** and its own file warns *"correlation is not interpretable."*
- The actual predictive claim — *cheap pre-RL D predicts the post-GRPO overthinking gap across a
  checkpoint panel* — has **no real checkpoint, no real GRPO run, no real post-gap.** The GRPO
  environment (`dl-rl`) is not even built yet (SETUP.md is instructions-only).

### 5.2 P2 validation core
1. **Build the real panel:** ≥7 checkpoints spanning a length/verbosity gradient (different base
   models or SFT stages), sample k CoT traces each on GSM8K/MATH, label correctness. GPU sampling.
2. **Run the real GRPO** (the isolated `dl-rl` env, per SETUP.md) to get the *post-RL* overthinking
   gap per checkpoint. This is the expensive, un-started part — GRPO on 7 checkpoints is the bulk
   of the workload.
3. **Primary test:** Spearman(pre-RL D, post-GRPO gap) across the ≥7 checkpoints, cluster-
   bootstrapped, with a **length-only baseline** (does D beat "mean trace length" as a predictor?
   — currently missing and a reviewer's first question).
4. **Kill-gate:** ρ ≥ 0.5 with CI excluding 0 across ≥7 checkpoints AND D beats the length
   baseline. Below that, the diagnostic has no predictive value and the paper does not exist.

### 5.3 P2 adequacy verdict
**Least mature of the four — currently a validated estimator with no scientific result.** Not
adequate for anything until the real panel + GRPO land. TNNLS/Neurocomputing/AAAI are plausible
*if* the predictive gate passes on ≥7 real checkpoints. Largest GPU workload of all branches
(GRPO training × panel). Recommend gating this behind B6 completion — do not open the GRPO front
while B6 is one experiment from done.

---

## Part 6 — Workload, GPU budget, and sequencing (single RTX 5090, serial)

| Dir | Maturity | Blocking gap | GPU to journal-grade | Priority |
|---|---|---|---|---|
| **B6** | **High** (1 gate passed, causal fixed) | cross-arch clean metric + routing artifact | ~8–10 GPU-hr (mostly staged) | **1 — finish it** |
| **P4** | Failing in real data | fix decode+conformal, then re-gate | ~1 GPU-day after CPU fixes | 2 — cheap CPU fixes decide fate |
| **P3** | Partial (fallback data) | real benchmark + defense module | ~1–2 GPU-days + a download | 3 — needs download decision |
| **P2** | Scaffold only | real checkpoint panel + GRPO | largest (GRPO ×7) | 4 — gate behind B6 |

**Sequencing rule (unchanged from ROADMAP §0 but now evidence-weighted):** finish B6 first — it is
the only direction with a live result and the cheapest path to a submission. Run P4's *CPU-side*
decode/conformal fixes in parallel (no GPU) to learn whether P4 is salvageable before spending a
GPU-day on it. Hold P3 pending a download decision. Hold P2's GRPO until B6 is submitted.

---

## Part 7 — One-line adequacy scorecard (target: KBS / Neurocomputing / TNNLS / ESWA)

- **B6** — *Adequate after Tier-A + the D3 routing artifact.* The mechanism is real and now
  survives strict statistics; the routing method is what earns the "knowledge-based method" label.
  Biggest risk: gemma clean-metric cross-arch result. **Closest to submittable.**
- **P4** — *Not adequate; failing its own gate in real data.* Salvage via decode+conformal fixes;
  most likely becomes an honest-negative + working-baseline paper. Decide with cheap CPU work.
- **P3** — *Not adequate; placeholder data.* Sound statistics, needs a real benchmark + a defense
  module; retarget from TIFS to ESWA/Neurocomputing.
- **P2** — *Not adequate; no real result at all.* A good estimator awaiting its experiment; gate
  behind B6.

**Bottom line:** the previous designs are adequate in *statistical machinery* (the nulls, the
bootstraps, the kill-gates are all correctly conceived) but **inadequate in evidence** for three
of four directions, and B6 needs the cross-architecture clean-metric result plus a deployable
routing artifact to clear the Q1-journal bar rather than the workshop bar its own skeleton aimed at.
