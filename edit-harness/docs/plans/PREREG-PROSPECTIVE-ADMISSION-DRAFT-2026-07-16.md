# PREREG DRAFT — Prospective admission-policy evaluation (R-E)   2026-07-16

**STATUS: DRAFT. USER MUST RATIFY BEFORE ANY GPU RUN.** This document freezes the design and
predictions BEFORE `experiments/prospective_admission.py` is ever pointed at a real model. Do
not launch (no driver ships with this revision on purpose — see "Launch" below). If the user
amends any line below, the amendment must land here, in writing, before a run.

## Why this exists (blind-referee framing)

Every existing D2 federation result (`experiments/merging_m0.py`, `merging_editors.py`,
`d3_benefit_predictor.py`, `rg_admission_benefit.py`) is **retrospective**: edits are measured,
merged, and THEN the geometry statistic is read off and correlated with the observed damage.
A blind referee's natural question is whether the geometry statistic is useful **prospectively**
— i.e., as an actual pre-merge admission/screening rule, not just a post-hoc correlate. This
experiment answers that directly: build a policy that ADMITS edits by geometry BEFORE any
merging happens, and measure real behavioral outcomes on the deployed merged model.

## Reference cell (frozen)

- Model: Llama-3.2-1B, layer L12 (the canonical B6/D2 reference cell — same as
  `results/merging/Llama-3.2-1B_L12_RG/`).
- Editor: ROME (`editors/rome_native.py`), identity-covariance, single-layer rank-one — reuses
  `experiments/merging_m0.py`'s `_load_edit_model` / `_compute_solo` / `_merge_factors` verbatim.
  No new editor code; this experiment is entirely a NEW ADMISSION + MEASUREMENT layer on top of
  the existing ROME federation primitives.
- Dataset: CounterFact (`data/counterfact.json`), same loader convention
  (`load_counterfact`, `default_rng(seed).shuffle` → first-N slice) as every other cell in this
  harness, so edit selection is byte-reproducible across re-runs at the same seed.

## Candidate pool and admission (frozen)

- **Candidate pool**: N = 100 edits per seed (`load_counterfact(data, 100, seed)`).
- **Pre-admission screening score**: the Eq-1 closed form generalised to the WHOLE POOL as the
  cross-talk universe (not a fixed merge group — admission happens *before* grouping, so the
  score must not depend on a group assignment that does not exist yet):

      I_cos(a) = ||k_a|| * sum_{b != a in pool} S_b * |cos(k_b, k_a)|
      I_mag(a) = ||k_a|| * sum_{b != a in pool} S_b                      (cosine=1 bound)

  identical definition to `merging_m0._regime_stat`'s per-observation I_cos/I_mag, just summed
  over the full 100-candidate pool rather than one measured merge group. S_b = ||r_b||/||k_b||
  (ROME's per-edit residual-to-key ratio), computed by the SAME solo-capture pass
  (`merging_m0._compute_solo`) every other cell in this harness uses — no new geometry formula.
- **Budget**: 25% admission (k = floor(0.25 * 100) = 25 admitted edits).
- **Three policies**, all evaluated on the SAME 100-candidate pool per seed:
  1. **geometry** — admit the 25 candidates with the LOWEST I_cos (least predicted received
     interference; same "bottom-q" convention as `rg_admission_benefit.py`'s `admit_stats`).
  2. **magnitude** — admit the 25 candidates with the LOWEST I_mag (magnitude-only baseline,
     the standard ablation this whole research line uses to isolate geometry's marginal value
     beyond raw magnitude).
  3. **random** — a uniform random 25-of-100 draw, **3 independent draws** per seed (distinct
     RNG streams, reported individually AND averaged) so the random baseline carries its own
     variance estimate rather than being a single lucky/unlucky draw.

## Grouping and installation (frozen)

- After admission, partition the 25 admitted edits into **g = 5 random groups of 5**
  (`floor(25/5) = 5` groups, no remainder — matches `merging_m0._tiled_groups`' disjoint-tiling
  convention, re-seeded independently of the admission draw).
- Each group's combined ΔW is installed exactly as `merging_m0._merge_factors` /
  `_measure_merged_groups` already do (`R_g^T @ (K/denom)_g`, bit-identical to summing the
  editor's own per-edit ΔW), measured, then the base weights are restored before the next group
  — no new merge mathematics, only new things measured post-merge.

## Behavioral outcomes measured (frozen; all on the ACTUALLY DEPLOYED merged model, not a
closed-form estimate)

For every admitted edit `a` in every installed group:

  - **(a) Edit success rate** — `argmax_ok_post` on `a`'s own rewrite prompt (same convention as
    `argmax_ok_solo`/`argmax_ok_post` everywhere else in this harness): does the post-merge
    argmax still equal the edited target token?
  - **(b) Specificity / neighborhood damage** — CounterFact's own `neighborhood_prompts` field
    (content-matched via `experiments.egl_metrics.attach_egl_fields`, the canonical ES/PS/NS
    machinery already in this harness — reused, not reinvented), scored post-merge with
    `egl_metrics.full_target_scores`'s NS convention (neighbor is undamaged iff
    mean-logprob(target_true) > mean-logprob(target_new)). Reported as the merge-INDUCED
    neighborhood damage rate: the fraction of (edit, neighbor-prompt) pairs whose NS flips from
    "undamaged" (measured once, at the reference state selected by `--ns_reference` — see the
    decision-point amendment right below) to "damaged" post-merge.

### Decision point: neighborhood-damage reference (RESOLVED behind a flag, 2026-07-16 amendment)

The "measured once, pre-merge" reference above was ambiguous as originally written: it read as
the solo-edit baseline, but the code that shipped alongside this draft actually measured it at
the unedited BASE model (an artifact of `merging_m0._compute_solo` restoring every edit after
its own solo pass — the weights at that point are indistinguishable from base). Both readings
are now real, IMPLEMENTED options behind `experiments/prospective_admission.py --ns_reference`,
so ratifying this decision point is choosing a flag value, not writing new code:

  - **`--ns_reference base`** (option ii, the prior de facto behavior): neighbor NS measured at
    the unedited base model. A neighbor that edit `a`'s OWN solo edit already damages is never
    counted as merge-induced — INCLUDES solo-edit collateral in what counts as pre-existing
    damage.
  - **`--ns_reference solo`** (option i, true federation-added damage): neighbor NS measured with
    edit `a` installed ALONE (its own solo ΔW, reconstructed via `merging_m0._merge_factors` with
    a singleton group — bit-identical to the editor's own per-edit ΔW, no new editor math), then
    restored. Isolates the damage the FEDERATION adds on top of what the solo edit already did —
    this is what the original prose above described.

No default: the module REFUSES to run the GPU path if `--ns_reference` is omitted, mirroring
(and not weakening) the prereg-ratification `SystemExit` guard. CPU-validated at build time via
`--selftest` part (d): the singleton-group ΔW reconstruction is checked against a manual
`outer(r_a, k_a/denom_a)` recomputation, and the install→restore round-trip for both dispatch
branches is asserted exact (no model, no GPU).
  - **(c) General retention** — a FIXED, seed-independent, 200-prompt held-out CounterFact
    sample (disjoint from every seed's 100-candidate pool by construction: drawn from the
    complementary index range of the same shuffled file, seed=999 fixed, so it never overlaps
    any candidate pool at any of the 3 seeds), scored by mean full-target log-probability
    (`egl_metrics.mean_logprob_full_target` on each held-out prompt's OWN true-answer
    continuation) once at BASE weights and once per installed group; reported as the mean shift
    (post-merge minus base), the standard "does merging degrade unrelated knowledge" probe.
  - **(d) Target-logit drop** — `drop = logit_solo[a] - logit_post_merge[a]`, THE SAME quantity
    every retrospective RG/M0 table already reports (continuity field — this is what lets R-E's
    numbers sit in the same column as `RG_admission_benefit_20260715.json`'s retrospective
    benefit table).

## Seeds and repetition (frozen)

- 3 seeds (0, 1, 2), each drawing an INDEPENDENT 100-candidate pool (`load_counterfact(..., seed)`)
  and running all 3 policies (geometry / magnitude / random-x3) end-to-end on that pool.
- Total GPU cost per seed: 100 solo-capture edits (ROME value-opt, ~identical cost to any
  existing 100-edit RG seed) + a small number of cheap forward-only group-merge measurements
  (5 groups x geometry, 5 x magnitude, 5 x 3 random draws = 25 group-installs per seed, each one
  cheap merge-add + a handful of forwards) + 200 held-out retention forwards x (1 base + up to
  7 group configurations) per seed. Order of magnitude comparable to one existing RG seed
  (~minutes, not hours) — no honest GPU-minute estimate exists yet because this has never run;
  treat the first real invocation's timing as the first honest number (same convention as every
  other new driver in this harness).

## Frozen predictions (directions only — NO magnitude committed, kill-gate discipline applies)

If the D2 geometry-screening law has PROSPECTIVE, not just retrospective, value:

- **(P1)** geometry-admission shows a HIGHER mean edit success rate among admitted edits than
  magnitude-admission and random-admission (the lowest-I_cos edits are, almost by construction,
  also less likely to have their own success clobbered by federation cross-talk).
- **(P2)** geometry-admission shows LOWER merge-induced neighborhood damage (b) than magnitude
  and random.
- **(P3)** geometry-admission shows a SMALLER general-retention shift (c) than magnitude and
  random (fewer/weaker cross-terms landing on unrelated knowledge).
- **(P4)** geometry-admission shows a SMALLER aggregate |drop| (d) than magnitude and random,
  continuing the retrospective finding (`d3_benefit_predictor.py`, `rg_admission_benefit.py`)
  into a prospective setting.
- A **KILL** outcome (geometry indistinguishable from or worse than magnitude/random on most of
  P1–P4) is a legitimate, reportable finding under this workspace's truth-first / kill-gate
  discipline (`docs/plans/TRUTH-FIRST-RESET-2026-07-15.md`) — it would mean the retrospective
  correlation does not translate into a deployable admission rule at this budget/group-size, and
  must be written up as such, not suppressed or re-scoped after the fact.
- No p-value/threshold is pre-committed beyond "3 seeds, report all seeds separately, then
  pooled" (the house convention throughout B6/D2) — magnitude comparisons are exploratory.

## What is reused vs. new (so a reviewer can audit scope)

Reused, unmodified: `experiments/merging_m0.py` (`load_counterfact`, `_load_edit_model`,
`_compute_solo`, `_merge_factors`, `_spearman`, `_savez_atomic`, `_write_table`, `_model_tag`),
`experiments/egl_metrics.py` (`attach_egl_fields`, `mean_logprob_full_target`,
`full_target_scores`). New: the pool-wide (not group-restricted) I_cos/I_mag screening function,
the three admission policies, the random group partition, and the behavioral-measurement loop
that installs each group and reads off (a)-(d) — all in `experiments/prospective_admission.py`.
No new ROME/editor code, no new dataset loader, no new geometry formula.

## CPU validation performed at build time (before any GPU run)

`experiments/prospective_admission.py --selftest` (no model, no GPU, synthetic pool):
asserts the pool-wide I_cos/I_mag formula against a brute-force O(N^2) recomputation; asserts
`admit_bottom_q`/`admit_random` budget rounding and determinism; asserts `partition_groups`
produces `floor(k/group_size)` disjoint groups covering exactly `group_size * n_groups` of the
admitted indices; runs the aggregation/table-writing path on synthetic per-group measurements
end-to-end; asserts the `--ns_reference` singleton-group solo-ΔW reconstruction against a manual
recomputation and the install→restore round-trip for both `base`/`solo` dispatch branches (part
(d), see the decision-point amendment above). `bash -n` / `py_compile` clean. See
`docs/plans/REVWAVE-BUILD-NOTES-2026-07-16.md` for the exact commands run and their output.

## Launch (NOT provided — ratification gate)

No `run_revwave_re.sh` ships with this revision. Deliberate: this experiment's design (admission
budget, group size, retention-set size, neighborhood-damage definition) is exactly the kind of
frozen-before-numbers-exist choice this workspace's prereg discipline exists to protect, and it
has not been reviewed by anyone but its author yet. Once ratified, the launch command is (NOTE:
`--ns_reference {solo,base}` is now REQUIRED — no default — picking its value IS the
neighborhood-damage-reference half of ratification, see the decision-point amendment above):

```
python experiments/prospective_admission.py \
    --model data/models/Llama-3.2-1B --layer 12 --data data/counterfact.json \
    --n_pool 100 --budget 0.25 --group_size 5 --n_retention 200 --n_random_draws 3 \
    --ns_reference solo \
    --seeds 0,1,2 --steps 20 --lr 0.1 --device cuda \
    --out_dir results/prospective_admission
```
(a driver mirroring `run_merging_editors.sh`'s preflight/idle-gate/budget skeleton should be
built alongside ratification, not before it).
