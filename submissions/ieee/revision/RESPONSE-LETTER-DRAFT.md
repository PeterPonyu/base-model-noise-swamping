# Draft Response to the Reviewers

> **Status:** Point-by-point skeleton for the B6 TETCI revision round. Replace page and line references after the revised manuscript is rebuilt. Evidence marked `{H1-PENDING}`, `{H5-PENDING}`, or `{H6-PENDING}` must not be converted to numerical claims until the corresponding preregistered cells and readouts are complete.

This draft records both changes already applied to the revision manuscript and changes queued behind the labeled evidence slots. The revision package replaces stale single-seed entries with verified multi-seed estimates, discloses boundary cases, separates algebraic consequences from empirical findings where already folded, and reserves unfinished controls for explicit pending slots.

## B6-1. Is the AlphaEdit causal result algebraically guaranteed?

**Reviewer objection.** *Because AlphaEdit projects updates away from preserved-key directions, is the observed relationship between key cosine and damage removed simply guaranteed by the construction rather than evidence about knowledge-key geometry?*

**Response.** We agree that, once a projector removes nearly all preserved-key energy, ordering the cancelled component by edit–probe key cosine is close to definitional; this response concedes that point explicitly, and the same concession is queued for the manuscript. The empirical claims that remain are the projector-free ROME association, the magnitude of removal, and the L14 transition result, while the ratified A-RAND control tests whether matched-norm random rank-one directions reproduce the same ordering. Its frozen design contains 12 cells, L8/L10/L12/L14 × seeds 0/1/2, and will be reported under gates G-R1–G-R4 without post-hoc layer, sign, or threshold substitution.

**Artifact evidence.** `docs/plans/PREREG-B6-RANDOM-DIRECTION-CONTROL-2026-07-30.md`; `submissions/ieee/revision/PROJECTOR-CONTROL-ILLPOSED-20260726.md`; `edit-harness/results/C4_causal_holdout_table_3seed.json`.

**Number destination.** `{H5-PENDING}` → per-layer A-OPT minus A-RAND Δρ, A-RAND permutation decisions, and the paired norm-growth readout in the response letter and the key-versus-random panel. No A-RAND value may enter the manuscript before the preregistered 12-cell readout is complete.

**Manuscript change.** Queued after H5: replace the remaining strong mechanism wording with the explicit algebraic concession, preserve the projector-free G1 result as the primary geometry-to-damage evidence, and add the A-RAND paragraph and panel.

## B6-2. Why is the held-out projector primary when only two depths are reported?

**Reviewer objection.** *The manuscript calls the held-out-key projector the primary causal protocol, but held-out estimates are available only at L8 and L12; how can the depth claim be assessed at L10 and L14?*

**Response.** The current held-out estimates are ρ = 0.390 at L8 and 0.590 at L12, each pooled over three seeds (`C4_causal_holdout_table_3seed.json`). We have queued the identical held-out protocol at L10 and L14 for seeds 0/1/2 so that the complete four-depth profile is evaluated under one projector-source convention (`PLAN-GAP-CLOSURE-MASTER-2026-07-31.md`, H6; `run_b6ins.sh`, Cell H).

**Artifact evidence.** `edit-harness/results/C4_causal_holdout_table_3seed.json`; `docs/plans/PLAN-GAP-CLOSURE-MASTER-2026-07-31.md` (H6); `edit-harness/run_b6ins.sh` (Cell H).

**Number destination.** `{H6-PENDING}` → proposed `\hoRhoLten` and `\hoRhoLfourteen` macros, the held-out column of Table I, and the corresponding response-letter sentence. The existing `\hoRhoLeight = 0.390` and `\hoRhoLtwelve = 0.590` remain unchanged.

**Manuscript change.** Table I completion and the four-depth held-out paragraph are queued for the H6 fold; the manuscript will not imply that L10/L14 held-out values already exist.

## B6-3. Table II is stale relative to the authors' own artifacts

**Reviewer objection.** *Several Table II entries appear to be single-seed values even though three-seed extension artifacts are available; why were the current estimates and their variability not reported?*

**Response.** We agree and have replaced the stale entries with the verified three-seed means: 1B-Instruct L12 ρ = 0.552, GPT-J-6B L21 ρ = −0.184, Llama-8B L16 ρ = 0.212, and Llama-8B L24 ρ = −0.087 (`C4_causal_instruct_table_3seed.json`, `C4_causal_gptj_table_3seed.json`, and `C4_causal_8b_table_3seed.json`). The corresponding ROME→AlphaEdit mean-damage entries are 2.298→0.086, −0.034→−0.002, 0.260→0.022, and −0.050→0.020; NeoX-20B L16 remains explicitly labeled single-seed (`TABLE2-3SEED-UPDATE.md`).

**Artifact evidence.** `submissions/ieee/revision/TABLE2-3SEED-UPDATE.md`; `edit-harness/results/C4_causal_instruct_table_3seed.json`; `edit-harness/results/C4_causal_gptj_table_3seed.json`; `edit-harness/results/C4_causal_8b_table_3seed.json`.

**Number destination.** Already verified and folded into the Table II macro block and table body; no pending slot. The response will retain the per-seed provenance and will not present the NeoX row as multi-seed.

**Manuscript change.** Completed in the July 31 fold: three-seed Table II values, matching damage columns, and an explicit seed-count qualification for NeoX-20B.

## B6-4. Quartile ratios amplify noise near zero

**Reviewer objection.** *Top-to-bottom quartile ratios are unstable when the bottom-quartile mean is approximately zero and can rhetorically magnify negligible absolute differences. Why are such ratios emphasized?*

**Response.** We agree and have retired the GPT-J quartile ratio because its denominator is only about 0.0099 logits. We now report the absolute quartile means, −0.010/−0.021/−0.035/−0.060, and apply the same rule to sign-crossing or near-zero 8B cells (`TABLE2-3SEED-UPDATE.md`).

**Artifact evidence.** `submissions/ieee/revision/TABLE2-3SEED-UPDATE.md`; `edit-harness/results/C4_causal_gptj_table_3seed.json`; `submissions/ieee/macros.tex` (retired `\gptjRatio`).

**Number destination.** Already verified in the Table II note and causal prose; no pending slot. No ratio is reported for GPT-J, Llama-8B L24, or Llama-8B L28.

**Manuscript change.** Completed in the July 31 fold: `\gptjRatio` is retired from `macros.tex`, the multiplier language is removed, and the table note states the absolute-scale reporting rule.

## B6-5. The Phi signed/magnitude split could look like claim-shopping

**Reviewer objection.** *The manuscript describes Phi-3.5 as null under the signed estimand but positive under the magnitude estimand. Was the estimand selected after observing which version passed?*

**Response.** The revision states that signed within-probe correlation is the primary direction-sensitive estimand and that the magnitude analysis answers a distinct, secondary question about absolute damage. More importantly, the original Phi cells are now invalidated by the tokenizer-integrity audit, so neither the old signed nor magnitude value will be defended or retained until the corrected three-seed insertion and deletion cells are complete (`findings-PHI35-TOKENIZER-COLLISION-2026-07-30.md`; `PLAN-GAP-CLOSURE-MASTER-2026-07-31.md`, H1).

**Artifact evidence.** `docs/findings/findings-PHI35-TOKENIZER-COLLISION-2026-07-30.md`; `docs/plans/PLAN-GAP-CLOSURE-MASTER-2026-07-31.md` (H1); `edit-harness/engine/manifests/phi_refix_b6.txt`.

**Number destination.** `{H1-PENDING}` → corrected `\phiSigned`, `\magPhi`, and `\magPhiSD` only after all six fixed Phi cells are present and the preregistered 1,000-permutation readout is rerun. No two-seed preview may appear in the response or manuscript.

**Manuscript change.** The July 31 prose fold makes the signed/magnitude distinction explicit; the current Phi macros are queued for replacement, not affirmation, after H1.

## B6-6. The MQuAKE causal cell is a boundary case, not a headline replication

**Reviewer objection.** *On MQuAKE-CF, AlphaEdit removes much less damage and the lowest quartile is slightly negative. Does this weaken the claim that geometry identifies removable collateral damage?*

**Response.** We now present MQuAKE only as a consistency check: with a held-out projector, the coupling is ρ = 0.495 over three seeds (sd = 0.004), while ROME→AlphaEdit damage changes from 3.092 to 1.773, or approximately 43% removed (`C4_causal_mquake_holdout_table_3seed.json`). Damage removed rises from −0.035 in Q1 to 3.65 in Q4, but we do not quote the sign-crossing quartile ratio and do not combine this harder multi-hop setting with the 91–96% CounterFact removal headline.

**Artifact evidence.** `edit-harness/results/C4_causal_mquake_holdout_table_3seed.json`; `submissions/ieee/revision/TABLE2-3SEED-UPDATE.md`; `submissions/ieee/macros.tex` (`\mquakeCausal*`).

**Number destination.** Already verified in `\mquakeCausalRho`, `\mquakeCausalRhoSD`, `\mquakeCausalQuartLo`, `\mquakeCausalQuartHi`, and `\mquakeCausalFracRemoved`; no pending slot.

**Manuscript change.** Completed in the July 31 fold: MQuAKE is kept out of Table II and added to §8 as a held-out-projector consistency check with the lower-removal and negative-Q1 caveats.

## B6-7. What does the theorem buy empirically?

**Reviewer objection.** *The gain of S×C over raw cosine is modest, and its agreement with true first-order influence is only about 0.09. Is the formal result being used to imply a mechanism that the data do not establish?*

**Response.** We have narrowed the proposed claim: S×C is a zero-backprop algebraic estimator obtained from the ROME rank-one update, not a proof of the downstream mechanism and not a faithful reconstruction of true-influence rankings. The normalized S×C correlations are 0.402/0.553/0.677/0.504 across L8/L10/L12/L14, while the independent L12 backward-pass cell finds true influence predicts damage at ρ = 0.474; the stabilized three-seed S×C/true-influence rank agreement is only 0.075 (sd = 0.011; per-seed 0.0874/0.0706/0.0660). We also do not transfer the reduction to MEMIT: its measured key-coupling ρ is 0.019 at L8 and 0.037 at L12, both below the 0.10 DEAD threshold and reported as ρ_C, never “MEMIT S×C.”

**Artifact evidence.** `edit-harness/results/C1_mechanism_sc_table.json`; `edit-harness/results/GRADSIM_TRUE_Llama-3.2-1B_L12_s{0,1,2}.json`; `edit-harness/results/REVISION_DOSSIER.json` (`gradsim_true_L12`); `edit-harness/results/C3_memit_L{8,12}_r3.json`; `docs/findings/THEOREM-SXC-DRAFT-2026-07-06.md`; `docs/findings/findings-MEMIT-SC-RECONCILIATION-2026-07-04.md`.

**Number destination.** The normalized S×C and true-backprop limitation values are verified; no pending slot. The response-letter and manuscript theorem paragraph must use the three-seed rank-agreement summary rather than presenting 0.0874 without its seed-0 label.

**Manuscript change.** Queued in the theorem-value fold: replace the remaining “mechanistic surrogate” wording with “zero-cost algebraic estimator,” retain the exact rank-one reduction under stated assumptions, and update the true-influence limitation to the three-seed agreement.

## Projector circularity

**Reviewer objection.** *If the AlphaEdit projector is estimated from the same type of facts used to measure damage, could the causal coupling be a projector-fit artifact?*

**Response.** The primary CounterFact projector is fitted on held-out facts disjoint from the damage probes and yields ρ = 0.390 at L8 and 0.590 at L12 over three seeds (`C4_causal_holdout_table_3seed.json`); the revision dossier reports the same qualitative result under the comparison projector source. The independent MQuAKE held-out cell is also stable at ρ = 0.495, and the full revision dossier records 23 tracked cells, 22 stable, zero shifted, and one descriptively confirmed (`C4_causal_mquake_holdout_table_3seed.json`; `REVISION_DOSSIER.json`). We nevertheless treat the cancelled-component ordering as partly construction-driven and do not present projector-source agreement as proof that the algebraic concern disappears.

**Artifact evidence.** `edit-harness/results/C4_causal_holdout_table_3seed.json`; `edit-harness/results/C4_causal_table.json`; `edit-harness/results/C4_causal_mquake_holdout_table_3seed.json`; `edit-harness/results/REVISION_DOSSIER.json`; `submissions/ieee/revision/PROJECTOR-CONTROL-ILLPOSED-20260726.md`.

**Number destination.** Existing L8/L12 values remain in `\hoRhoLeight` and `\hoRhoLtwelve`; `{H6-PENDING}` completes L10/L14. `{H5-PENDING}` supplies the separate matched-norm random-direction test of algebraic inevitability.

**Manuscript change.** The held-out projector is already primary in the revision and the MQuAKE held-out consistency check is in §8. The explicit algebraic concession and A-RAND result are queued after H5; no sham-projector result will be revived.

## Theorem value and scope

**Reviewer objection.** *Does the proposition add a testable insight, or merely restate the ROME rank-one update with assumptions strong enough to force the conclusion?*

**Response.** Proposition 1 is retained as an exact readout identity, while the loss-level ranking statement is explicitly conditional and no longer presented as an unconditional theorem. Its practical value is computational and diagnostic: S×C is available without backpropagation and improves on raw key cosine at all four reported depths, but the stabilized L12 true-influence rank agreement is only 0.075 (sd = 0.011), so it is not a faithful surrogate.

**Artifact evidence.** `docs/findings/THEOREM-SXC-DRAFT-2026-07-06.md`; `edit-harness/results/C1_mechanism_sc_table.json`; `edit-harness/results/GRADSIM_TRUE_Llama-3.2-1B_L12_s{0,1,2}.json`; `edit-harness/results/REVISION_DOSSIER.json` (`gradsim_true_L12`).

**Number destination.** The formal identity, normalized S×C values, and three-seed true-influence limitation are verified; no pending slot. The response cites the proposition/remark and the true-backprop limitation together so that the algebraic and empirical scopes cannot be conflated.

**Manuscript change.** Queued in the theorem-value fold: remove the remaining “mechanistic surrogate” phrase, preserve the explicit assumptions, and present S×C as a zero-cost algebraic estimator with a measured three-seed limitation.

## Seed variance and stability

**Reviewer objection.** *The causal conclusions rely on isolated seeds. How stable are the signs, effect sizes, and qualitative conclusions across projector seeds?*

**Response.** Across the four Table II cells now available at three seeds, no sign changes and no qualitative threshold is crossed; the seed sd ranges from 0.013 to 0.032 in ρ, with the largest max–min spread 0.062 at Llama-8B L16 (`TABLE2-3SEED-UPDATE.md`). The means and sd values are: 1B-Instruct 0.552 (0.017), GPT-J −0.184 (0.018), Llama-8B L16 0.212 (0.032), and Llama-8B L24 −0.087 (0.013), while NeoX-20B remains transparently single-seed. The broader `REVISION_DOSSIER.json` summary is 23 cells, 22 stable, zero shifted, zero pending, and one descriptive confirmation.

**Artifact evidence.** `submissions/ieee/revision/TABLE2-3SEED-UPDATE.md`; `edit-harness/results/C4_causal_instruct_table_3seed.json`; `edit-harness/results/C4_causal_gptj_table_3seed.json`; `edit-harness/results/C4_causal_8b_table_3seed.json`; `edit-harness/results/REVISION_DOSSIER.json`.

**Number destination.** Already verified in Table II, its note, and the response-letter seed table; no pending slot. H6 will add the missing held-out L10/L14 depth estimates but does not alter the status of the existing Table II seed fold.

**Manuscript change.** Completed in the July 31 fold: Table II uses three-seed values where available, reports variability, and marks the NeoX exception instead of pooling it with multi-seed cells.

## Phi tokenizer integrity

**Reviewer objection.** *How do the authors know the Phi-3.5 edits targeted the intended answer token rather than a tokenizer artifact, and what happens to conclusions based on the affected cells?*

**Response.** The audit found that Phi-3.5's leading whitespace token, id 29871, was returned for every target, invalidating seven matrices: three insertion, one AlphaEdit, and three deletion cells (`findings-PHI35-TOKENIZER-COLLISION-2026-07-30.md`). Edit success could not detect the defect—the invalid insertion cells reported success near 0.995—so the harness now skips whitespace-only leading tokens and requires `assert_targets_distinguishable` before GPU execution; the affected artifacts are quarantined rather than repaired in place. All Phi-dependent B6 values remain withdrawn until the corrected six-cell insertion/deletion set and final readouts are complete.

**Artifact evidence.** `docs/findings/findings-PHI35-TOKENIZER-COLLISION-2026-07-30.md`; `edit-harness/results/_invalid_phi35_tokencollision_20260730/`; `edit-harness/engine/run_phi_b6_refix.sh`; `edit-harness/engine/manifests/phi_refix_b6.txt`.

**Number destination.** `{H1-PENDING}` → final three-seed `\phiSigned`, `\magPhi`, and `\magPhiSD`, generated only after `run_phi_b6_refix.sh` completes and `magnitude_table.py --n_perm 1000` plus `analyze_matrices.py` are rerun. The two-seed preview is prohibited from the response letter, macros, tables, and prose.

**Manuscript change.** Queued in the July 31 fold: replace the invalid Phi macros and signed/magnitude sentence, add the tokenizer-integrity receipt to the implementation details, and rebuild the PDF only after H1 is final.

## Evidence-Slot Inventory

| Slot | Evidence still required | Cells | Destination | Release condition |
|---|---|---:|---|---|
| `{H1-PENDING}` | Corrected Phi insertion and deletion readouts | 4 currently missing; 6 fixed cells required in the final set | `\phiSigned`, `\magPhi`, `\magPhiSD`; architecture/magnitude prose | All six fixed cells present; 1,000-permutation magnitude readout and signed analysis complete; no preview values |
| `{H5-PENDING}` | Matched-norm A-RAND rank-one control | 12: L8/L10/L12/L14 × seeds 0/1/2 | Response paragraph, per-layer control table, key-versus-random panel | Ratified G-R1–G-R4 readout complete with raw-npz recomputation |
| `{H6-PENDING}` | Held-out AlphaEdit depth completion | 6: L10/L14 × seeds 0/1/2 | Proposed `\hoRhoLten`, `\hoRhoLfourteen`; Table I held-out column | All six cells valid and pooled three-seed estimates generated |

## Artifact Index

- `submissions/ieee/revision/TABLE2-3SEED-UPDATE.md`
- `submissions/ieee/revision/SHAM-CONTROL-READOUT-20260726.md`
- `submissions/ieee/revision/PROJECTOR-CONTROL-ILLPOSED-20260726.md`
- `submissions/ieee/revision/PORTAL-PDF-AUDIT-20260726.md`
- `docs/plans/PLAN-REVIEWER-CLOSURE-CAMPAIGN-2026-07-30.md`
- `docs/plans/PLAN-GAP-CLOSURE-MASTER-2026-07-31.md`
- `docs/plans/PREREG-B6-RANDOM-DIRECTION-CONTROL-2026-07-30.md`
- `docs/findings/findings-PHI35-TOKENIZER-COLLISION-2026-07-30.md`
- `docs/findings/THEOREM-SXC-DRAFT-2026-07-06.md`
- `edit-harness/results/REVISION_DOSSIER.json`
- `edit-harness/results/C4_causal_holdout_table_3seed.json`
- `edit-harness/results/C4_causal_mquake_holdout_table_3seed.json`
