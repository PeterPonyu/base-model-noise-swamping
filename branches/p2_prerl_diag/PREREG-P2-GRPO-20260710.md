# PREREG — P2 pre-RL diagnostic → post-GRPO overthinking (FROZEN 2026-07-10, v1.1)

Frozen pre-registration for the **future** GPU GRPO validation wave. NOT launched by
this document. Any deviation from a numbered clause below MUST be logged in the results
writeup as an explicit "DEVIATION FROM PREREG" note with its reason.

*v1.1 (2026-07-10, same-day PRE-LAUNCH amendment from the hostile review pass — nothing
had run against v1.0): §4 selection-bias caveat added; §6(a) reworded (Jensen gap vs
outlier-robustness conflation); §6(d) common-set sensitivity added; §2 gemma unusability
reasons completed. No thresholds, endpoints, or gates were changed.*

Evidence base: `results/PANEL_deep_analysis.json` (from `analysis_deep.py`, seeded, CPU).

## 1. Hypothesis (H1)
The cheap pre-RL length-bias diagnostic **D_within** (difficulty-controlled
mean_len(wrong)/mean_len(right), `diagnostic.d_within`) POSITIVELY predicts, across the
checkpoint panel, the expensive post-GRPO **overthinking gap** G (def. §4). Direction is
pre-specified positive → one-sided test.

## 2. Checkpoint inclusion (driven by the Task-1 usability flag)
Rule (frozen, from Task 1): a checkpoint is USABLE iff `n_right >= 20` AND the 95%
cluster-bootstrap CI width on D_within `<= 1.5`.
- At the current pre-RL freeze, USABLE = {Qwen2.5-0.5B, Qwen2.5-1.5B, Qwen2.5-3B,
  Phi-3.5-mini} (n=4). UNUSABLE = Llama-3.2-1B (n_right=5<20 AND CIw>1.5), gemma-2-2b
  (n_right=16<20 AND CIw 1.71), Llama-3.2-3B (CIw 3.19; n_right=27 passes).
- **Registered target set is n>=6.** Before the confirmatory test, additional pre-RL
  sampling (same `sample_ckpt.py`, k=8) of gemma-2-2b, Llama-3.2-3B, Llama-3.2-1B is run
  until each reaches USABLE, restoring the panel toward n=6–7. Checkpoints still UNUSABLE
  at sampling-freeze are excluded from the primary test (logged).
- Rationale: the three under-sampled models carry the entire D_within>1 range; excluding
  them collapses the design, but their pre-RL D is currently too noisy to rank — so the
  registered path is to *measure them better*, not to test on the narrow usable-4.

## 3. Training config (pointer — do not edit)
`grpo_config.py::GRPOScaffold` (matched budget: LoRA r=32, k=8 rollouts, max_steps=500,
lr 1e-6, beta 0.04, boxed rule reward). Identical for every checkpoint — no per-model
tuning. Trainer via `build_grpo_trainer()` inside the patched `dl-rl` env only.

## 4. Primary endpoint — "post-RL overthinking gap" G (ONE frozen definition)
Per checkpoint c, sample k=8 traces post-GRPO on the same GSM8K problem set. Let a problem
be **solved-in-both** if it has >=1 correct trace pre-RL AND >=1 correct trace post-RL.

    G_c = mean over solved-in-both problems of
          [ mean_len(post-RL correct traces) / mean_len(pre-RL correct traces) ]

G_c > 1 ⇔ post-RL inflates chain length on problems the model already solved (length
growth without accuracy benefit = overthinking). Accuracy is controlled by restricting to
correct-in-both traces; per-problem ratios averaged arithmetically to match D_within's
construction. G_c is the scalar fed as `post_overthinking_gap[c]` to
`diagnostic.cross_checkpoint_spearman`.

**Registered CAVEAT (v1.1): selection bias in "solved-in-both".** Conditioning on
solved-in-both selects an easier AND checkpoint-DEPENDENT problem subset (stronger
checkpoints retain more, and harder, problems), partially reintroducing the
between-checkpoint difficulty confound D_within was built to remove. The bias direction
on the cross-checkpoint rho is not determinable a priori. The definition above stays the
frozen primary; §6(d) registers the fixed-common-set sensitivity that removes this
selection.

## 5. Primary test statistic
One-sided (positive) Spearman(pre-RL D_within, G) with **exact permutation p** over all n!
label permutations (enumeration, as in `analysis_deep._one_sided_p`; not the sampled
two-sided `spearman_perm_test`, which is reported only as a secondary cross-check).

## 6. Pre-specified sensitivity analyses (not primary)
(a) Replace pre-RL predictor with `exp(delta)` (within-problem log-length difference,
§Task-1d). **v1.1 wording:** exp(delta) is a geometric-mean analog and sits systematically
BELOW the arithmetic D_within even with zero outliers (Jensen gap: well-sampled
checkpoints show exp(delta)/D_within ≈ 0.77–0.86 — Qwen×3 0.80/0.83/0.86, Phi 0.77), so
the two estimators are NOT point-comparable and "exp(delta)<1" alone does not prove a
D_within>1 was outlier-driven. What Task 1 actually supports: gemma-2-2b (ratio 0.43),
Llama-3.2-3B (0.48) and Llama-3.2-1B (0.58) fall FAR below the typical Jensen offset —
consistent with heavy-tail/outlier influence on their D_within — and under the robust
estimator only Phi-3.5 (exp(delta)=1.15, CI [1.08,1.23]) and Llama-3.2-3B (1.70, CI
[1.10,2.64]) remain >1 with CIs excluding 1; gemma-2-2b (0.74 [0.45,1.24]) and
Llama-3.2-1B (0.72 [0.33,1.40]) are indistinguishable from no bias.
(b) Full-panel n=7 exploratory run (all checkpoints, noisy predictors included).
(c) D_pooled as predictor.
(d) **(v1.1)** G recomputed on the FIXED COMMON problem set — problems solved-in-both by
EVERY checkpoint included in the primary test — removing §4's checkpoint-dependent
selection at the cost of restricting to the easiest stratum. If the common set has <20
problems, report this sensitivity as unavailable (logged), do not relax it.

## 7. Exclusion rules (pre-stated)
- Checkpoint excluded from primary if UNUSABLE at sampling-freeze (§2).
- Problem excluded from G_c if not solved-in-both.
- Checkpoint excluded from G_c (and thus the test) if it has <20 solved-in-both problems.
- Any checkpoint whose GRPO run diverges / collapses (reward → 0 or NaN loss under the
  matched budget) is excluded and logged.

## 8. Pass gate (justified by the Task-1e power analysis)
Power is plug-in Monte Carlo (8000 sims): observed_D ~ N(point, boot_se); post ~
rho·z(trueD)+√(1-rho²)·noise; exact one-sided perm p. Critical observed rho for p<0.05:
**n=7 → 0.714, n=6 → 0.829, n=4 → 1.000** (perfect order). Power at true rho=0.9:
n=7 0.54, n=6 0.43, n=4 0.23.

- **PASS (H1 supported):** on the frozen set of size n, one-sided exact-perm p < 0.05,
  i.e. observed Spearman rho >= critical_rho(n) [0.714 at n=7, 0.829 at n=6].
- The study is declared **adequately registered only at n>=6.** A null at n<6 (esp. the
  usable-4, which needs perfect ordering) is NON-informative — reported as "underpowered,
  inconclusive", NEVER as evidence against H1.
- Because even n=7 reaches only 0.54 power at rho=0.9, a null at n=6–7 is reported as
  "no detectable association at this panel size", not as refutation.

## 9. Kill condition
If, after the budgeted extra sampling, the panel cannot reach n>=6 USABLE (e.g. a model's
correct-rate is so low that n_right>=20 is unreachable within budget), ABANDON the
confirmatory correlation and report the panel **descriptively only** (per-checkpoint D
table + G table, no rho gate). Do not relax the usability rule to force n up.

## 10. Frozen constants
Usability: n_right>=20 AND D_within CIw<=1.5. Bootstrap n_boot=2000. Test: one-sided exact
permutation, alpha=0.05. Seeds and all Task-1 numbers: `results/PANEL_deep_analysis.json`.
Deviations from any clause above must be logged as such in the writeup.
