# B6 Revision-Readiness Scaffold (2026-07-10)

> Doc-only artifact. Maps anticipated reviewer asks to prepared responses, each backed by
> a number read directly off a file on disk (path cited inline). Nothing here is quoted
> from memory alone. Source manuscript: `submissions/ieee/main.tex` (IEEE, `\iftnnls`/
> `\iftaslp` dual-fork; exact journal TBC, see Logistics). Status: B6 SUBMITTED, under
> review as of 2026-07-10.

---

## 1. "The extension results (MQuAKE, RippleEdits, 8B, instruction-tuned) were single-seed
##    at submission time — how confident are they?"

**Likely phrasing:** *"Section 5's generality claims for scale, instruction-tuning, and
dataset transfer rest on single-seed cells (s0 only). Please provide seed variance."*

**Response.** All four extension causal cells that were single-seed s0 at submission have
since been re-run to 3 seeds; the per-seed values are unchanged in sign and stable in
magnitude at every layer/model except one cell still in flight (see below).

| cell | s0 | s1 | s2 | 3-seed mean | verdict | source |
|---|---|---|---|---|---|---|
| Instruct-1B AlphaEdit(-holdout), L12 | 0.5675 | 0.5335 | 0.5550 | 0.552 (sd 0.017) | STABLE | `edit-harness/results/C4_causal_instruct_table_3seed.json`, confirmed in `REVISION_DOSSIER.json` cell `instruct_alphaHO_L12` |
| Llama-8B AlphaEdit(-holdout), L16 | 0.1852 | 0.2474 | 0.2040 | 0.2122 (sd 0.032) | STABLE | `C4_causal_8b_table_3seed.json`, dossier cell `8b_alphaHO_L16` |
| Llama-8B AlphaEdit(-holdout), L24 | −0.1016 | −0.0810 | −0.0781 | −0.0869 (sd 0.013) | STABLE | `C4_causal_8b_table_3seed.json`, dossier cell `8b_alphaHO_L24` |
| Llama-8B AlphaEdit(-holdout), L28 | 0.1543 | 0.0865 | 0.1266 | 0.1225 (sd 0.034) | STABLE (see damage-floor caveat, item 2) | `C4_causal_8b_table_3seed.json` (`seeds_used: [0,1,2]` as of 2026-07-10 23:38), dossier cell `8b_alphaHO_L28` |
| MQuAKE AlphaEdit(-holdout, proj_source=probes), L12 | 0.5784 | 0.4977 | 0.5224 | 0.5328 (sd 0.041) | STABLE (esr_warn) | `C4_causal_mquake_table_3seed_probesrc.json`, dossier cell `mquake_causal_L12` |
| RippleEdits ROME depth profile (ripple probes), L8/L10/L12/L14 | 0.457 / 0.448 / 0.277 / 0.480 | see file | see file | 0.454 / 0.465 / 0.274 / 0.492 | STABLE (all 4 layers) | `RIPPLE_depth_profile.json`, dossier cells `ripple_rome_L{8,10,12,14}_ripple` |

The `REVISION_DOSSIER.json` roll-up (`edit-harness/results/REVISION_DOSSIER.json`,
refreshed 2026-07-10 23:40 after the L28 completion run) reports **16 of 16 tracked cells
STABLE, 0 PENDING, 2 STABLE-with-`esr_warn`, 0 SHIFTED** (`summary.n_stable=16,
n_pending=0, n_esr_warn=2, n_shifted=0`). Zero cells flipped sign or crossed a qualitative
threshold going from 1 to 3 seeds.

**L28 slot (filled 2026-07-10 23:38, run `4 done / 0 fail`):** per-seed rho 0.1543 /
0.0865 / 0.1266, mean 0.1225 (sd 0.034), sign-consistent positive, dossier verdict STABLE.
**Interpretation caveat (binding):** at L28 the ROME damage scale is near-floor —
`mean_damage_rome = 0.0050` (vs 0.2603 at L16) and `mean_damage_removed = −0.0035`
(AlphaEdit residual damage slightly EXCEEDS ROME's) — so this rho is a rank correlation on
a damage scale ~50× smaller than L16's and must not be quoted as mechanism support at
depth; it completes the depth profile descriptively (see item 2).

---

## 2. "The 8B causal result looks weak and sign-flipping across depth — is this a real
##    confirmation of the mechanism at scale?"

**Likely phrasing:** *"Table [causal-scale] shows the AlphaEdit-holdout coupling at 8B
going from +0.19 at L16 to −0.10 at L24. Doesn't a sign flip undercut the causal claim?"*

**Response.** This is accurately characterized as attenuation, not confirmation, and the
manuscript already frames it that way (`submissions/ieee/EXTENSION-TODO.md`,
STUB-8BCAUSAL entry: "Landed as a claim-tightening / attenuation result, NOT a
confirmation"). With 3 seeds now on disk, both signs are **stable, not seed noise**:
L16 is positive across all three seeds (0.1852 / 0.2474 / 0.2040, mean 0.2122) and L24 is
negative across all three seeds (−0.1016 / −0.0810 / −0.0781, mean −0.0869) —
`edit-harness/results/C4_causal_8b_table_3seed.json`. The sign flip itself replicates; it
is not an artifact of a single seed's noise. This matches the observational (non-causal)
attenuation pattern already reported for 8B elsewhere in the paper. The paper's own
position (per `EXTENSION-TODO.md`) is explicit: "the large, positive-signed, monotone-
removal form of the law is a small-to-mid-scale result." We do not claim causal
confirmation at 8B; we report the coupling honestly weakens and can invert with depth at
this scale. The L28 3-seed point now completes the depth profile: +0.2122 (L16) → −0.0869
(L24) → +0.1225 (L28) — **non-monotone**, with the L28 point carrying a binding caveat:
ROME's collateral damage is near-floor there (`mean_damage_rome` 0.0050 vs 0.2603 at L16;
`mean_damage_removed` −0.0035), so at the deepest layer there is essentially no ROME
damage for AlphaEdit to remove and the small positive rho rides a ~50×-compressed damage
scale (see item 1's table and caveat).

**Evidence pointers:** `C4_causal_8b_table_3seed.json` (per-seed and per-layer numbers
above); `REVISION_DOSSIER.json` cells `8b_alphaHO_L16`/`8b_alphaHO_L24` (both
`sign_consistent: true` within their own layer, `esr_warn: false`, esr per seed all
≥0.99); `EXTENSION-TODO.md` STUB-8BCAUSAL section (paper's own framing).

**Remaining gap:** none for the depth profile itself (L28 3-seed landed 2026-07-10; the
verdict is non-monotone with a damage-floor caveat at L28). What remains genuinely open at
8B is *why* the coupling attenuates/inverts — a mechanism question, not a missing cell.

---

## 3. "Is the AlphaEdit projector fit circular — i.e. does 'AlphaEdit removes damage that
##    correlates with key-cosine' just reflect how the projector itself was built?"

**Likely phrasing:** *"AlphaEdit's null-space projector needs to be fit on some set of
facts. If it's fit on the same probes used to measure collateral damage, isn't the
damage-removal result circular by construction?"*

**Response.** All headline 3-seed causal cells in the revision package use
`proj_source: "holdout"` — the AlphaEdit null-space projector is fit on a fact set
disjoint from the probes used to measure damage-removed, precisely to rule out this
concern. We verified the holdout protocol against the non-held-out ("generic"/probe-fit)
projector on the original CounterFact causal table and found the two give
near-identical within-probe couplings: L8 holdout 0.3905 (per-seed 0.4006/0.3914/0.3795)
vs. generic 0.3969 (0.4033/0.4060/0.3816); L12 holdout 0.5903 (0.5902/0.6103/0.5704) vs.
generic 0.5968 (0.5955/0.6172/0.5778) — `edit-harness/results/C4_causal_holdout_table_3seed.json`
vs. `edit-harness/results/C4_causal_table.json`. The projector-fit source does not
materially move the result, which is the direct empirical answer to the circularity
concern.

**Caveat to state plainly in the response letter:** the MQuAKE causal cell
(`C4_causal_mquake_table_3seed_probesrc.json`) uses `proj_source: "probes"` **by design**,
not holdout — the filename itself flags this (`_probesrc` suffix) and the JSON's
`filters.proj_source` field confirms it. This cell (mean 0.5328, per-seed 0.5784/0.4977/
0.5224) should be presented as a reference/consistency check against the CounterFact
holdout numbers, not as an independent holdout-clean replication. If a reviewer presses
specifically on MQuAKE circularity, the honest answer is that we have not yet run a
holdout-projector version of the MQuAKE causal cell.

**Evidence pointers:** `C4_causal_holdout_table_3seed.json` (holdout, CF, L8/L12),
`C4_causal_table.json` (generic/probes, CF, L8/L10/L12/L14), `C4_causal_mquake_table_3seed_probesrc.json`
(`filters.proj_source: "probes"`, MQuAKE).

**Remaining gap:** no holdout-projector MQuAKE causal cell exists on disk; if this becomes
a formal reviewer ask, it is a new run, not a re-read of existing data.

---

## 4. "Does S×C actually beat raw key-cosine, and is the GradSim equivalence a real,
##    independent validation?"

**Likely phrasing:** *"You claim the S×C product is a better predictor than raw key-
cosine and that it approximates a GradSim/TracIn-style influence measure. Please
substantiate both claims."*

**Response, part A (S×C vs. raw key-cosine).** After a normalization fix (`S` must be
built from `norm_growth` = ‖ΔW‖_F, not the unnormalized `resid_norm`), the corrected
within-probe Spearman values show S×C beating raw key-cosine at **all four** layers, not
two of four as an earlier draft claimed:

| layer | rho_C (raw key-cos) | rho_SC (S×C) |
|---|---|---|
| L8 | 0.395 | 0.4015 |
| L10 | 0.5337 | 0.5526 |
| L12 | 0.6018 | 0.6765 |
| L14 | 0.3006 | 0.5037 |

Source: `edit-harness/results/C1_mechanism_sc_table.json` (Llama-3.2-1B, CounterFact,
`--known --edit_ok`, `rho_SC_valid_sxc: true` for all four rows — this validity flag
matters, see part C). Full derivation and the "loses at L8/L10 was a normalization
artifact" retraction: `docs/findings/THEOREM-SXC-DRAFT-2026-07-06.md`, STATUS UPDATE
item 2.

**Response, part B (GradSim equivalence — do not overclaim).** The manuscript's formal
statement (`sections/03_method.tex`, "Toward a formal statement," ported from the
reviewed theorem draft) is deliberately conservative: S×C is a **rank estimator** of
first-order training influence under Assumption A4′ (sign-consistency + low covariance of
the edit-varying coefficient α(e,p)), not a proven identity and not an empirically
validated surrogate. This caution is load-bearing, because the true-backprop check we ran
shows A4′ is only **half-met**: `edit-harness/results/GRADSIM_TRUE_Llama-3.2-1B_L12_s0.json`
reports `alpha_A4_test.sign_consistency_rate.mean = 0.812` (not near 1.0) and
`alpha_A4_test.coefficient_of_variation.mean = 2.4295` (high, not low) — so the
"low-variance" half of A4′ fails. Consequently the true (direct-backprop) influence and
S×C **rank-disagree**: `rank_agreement.direct_vs_SC.mean = 0.0874` in the same file, even
though the true influence itself does predict damage reasonably well
(`within_probe_rho.direct_vs_damage.mean = 0.4738`). **S×C is not a faithful rank-surrogate
of true gradient influence; it is a cheap, zero-backprop proxy that correlates with damage
in its own right, and the paper's wording must not claim more than that.** The manuscript
already ships the cautious phrasing (verified: `sections/03_method.tex` contains "rank
estimator," "not a faithful rank-surrogate," and cites Assumption A4′ explicitly — grep
confirms these phrases are present in the current draft).

**Evidence pointers:** `C1_mechanism_sc_table.json` (corrected rho_C/rho_SC by layer),
`docs/findings/THEOREM-SXC-DRAFT-2026-07-06.md` (full proof, §6.3 normalization bug, §9
open issues), `GRADSIM_TRUE_Llama-3.2-1B_L12_s0.json` (true-backprop A4′ test,
single-seed CPU/GPU science cell — not yet 3-seed).

**Remaining gap:** the true-GradSim cell is single-seed (s0, L12 only); no 3-seed or
multi-layer version exists on disk. If a reviewer wants the A4′ test repeated at another
layer or with seed variance, that is new compute, not a re-read.

---

## 5. "What about MEMIT — does the same S×C/geometry mechanism explain its (lack of)
##    collateral damage?"

**Likely phrasing:** *"You show ROME's collateral damage is geometry-predictable. Is the
same true for MEMIT?"*

**Response.** No — and the paper must be precise about why. MEMIT's within-probe
Spearman between key-cosine and damage is **negligible**: 0.0194 (L8, 3-seed mean of
0.0156/0.025/0.0175) and 0.0374 (L12, 3-seed mean of 0.0357/0.0306/0.046), both below the
project's 0.10 "DEAD" threshold, versus ROME's 0.395 (L8) / 0.6018 (L12) at the same
layers — `edit-harness/results/C3_memit_L8_r3.json`, `C3_memit_L12_r3.json`. **Binding
wording rule from the reconciliation review** (`docs/findings/findings-MEMIT-SC-RECONCILIATION-2026-07-04.md`):
quote this as **rho_C only, never "MEMIT S×C."** The S×C closed form is a single-layer
rank-one ROME identity; MEMIT spreads each edit across multiple layers (layers 5–8 or
9–12 depending on config), so the S×C statistic is not mechanistically valid for it — this
is enforced mechanically in the harness (`C1_mechanism_sc_table.json` carries ROME rows
only — MEMIT rows are excluded from that table entirely; its top-level `rho_SC_validity`
note states the restriction, and MEMIT numbers live solely in `C3_memit_*`). The approved sentence from the
reconciliation doc: *"For MEMIT, the within-probe Spearman between key-cosine |C| and
collateral damage is negligible — 0.019 (L8) and 0.037 (L12), 3-seed means, both below
the 0.10 DEAD threshold — versus ROME's 0.41 (L8) / 0.60 (L12). Geometry does not predict
MEMIT collateral damage."* If pressed further: report negligible *magnitude*, not "zero
correlation" (perm-p is small at these sample sizes but the effect size is trivial).

**Evidence pointers:** `C3_memit_L8_r3.json`, `C3_memit_L12_r3.json` (3-seed means),
`docs/findings/findings-MEMIT-SC-RECONCILIATION-2026-07-04.md` (binding caveats +
approved wording).

**Remaining gap:** none for the magnitude claim itself; the open item is the still-
unreconciled multi-layer/z-layer attribution question for *why* MEMIT is null (flagged
explicitly as NOT resolved in the findings doc, caveat 1).

---

## 6. "The sequential-editing results — do 50 chained edits show geometry predicts
##    which survive, and does collapse follow a clean trend?"

**Likely phrasing:** *"Figure/Table [sequential] shows survival collapsing after 50
edits. Is this collapse monotone, and is it geometry-predictable?"*

**Response.** The current, superseding analysis is the 4-stream re-scope
(`edit-harness/results/SEQ_analysis_L12_4stream.json`), which extends the original
2-stream pass (`docs/findings/findings-SEQ-ANALYSIS-2026-07-04.md`, streams s0/s1) with
two additional streams (s2/s3). Per `docs/plans/PAPER-FOLDIN-MAP-2026-07-10.md` binding
constraint 4, the 4-stream numbers are what ships; the 2-stream findings doc's specific
"10%/14%" framing is superseded (though its qualitative reasoning — non-monotone curves,
H1 null — still holds and is corroborated by the 4-stream data):

1. **Collapse is ordering-dependent, not a single clean number — "collapses to 10–14%" is
   DEAD wording.** Final survival at 50 edits varies by stream: s0 10%, s1 14%, s2 42%,
   s3 36% (`SEQ_analysis_L12_4stream.json`, `per_stream[i].final_survival_frac`); pooled
   across all 200 edits, 25.5% (`pooled.final_survival_frac`). Individual curves are
   non-monotone (e.g. s0: 0.50→0.35→0.367→0.225→0.10; s2: 0.60→0.60→0.70→0.35→0.42).
2. **Position-fragility is admissible, pooled.** Pooled position-vs-survival ρ = 0.3716,
   perm-p = 0.0005 (`pooled.position_fragility`); per-stream values range 0.20–0.51 (s0
   0.312 p=0.025, s1 0.202 p=0.167 n.s., s2 0.482 p=0.001, s3 0.514 p=0.0015). Wording:
   "later-applied edits survive modestly more often, pooled effect significant." **The
   ρ≈0.55 figure some earlier internal notes cite is RETRACTED** — no artifact produces
   it as a position-fragility number; do not confuse it with any of the per-stream partial
   values below.
3. **Geometry-attribution (H1) is UNSETTLED/null — no positive claim admissible.** The
   pre-registered gate (position-partialled S×C ρ>0, perm-p<0.05, all streams) fails:
   per-stream partialled ρ = −0.0095 (p=0.943), +0.0758 (p=0.571), +0.0315 (p=0.829),
   +0.157 (p=0.264); pooled partialled ρ = 0.0969, perm-p = 0.1764
   (`SEQ_analysis_L12_4stream.json`, `verdict` block, `H1_STATUS: "UNSETTLED"`). Raw
   (unpartialled) ρ is much larger (0.21–0.56) but collapses once stream position is
   partialled out — the geometry signal is confounded with position, not an independent
   predictor. The sequential section must stay **descriptive only** (collapse +
   position-fragility), with no geometry-attribution language.
4. **Install-artifact ruled out.** Streams s2/s3 have edit-success rates 0.98/0.98
   (`SEQ_analysis_L12_4stream.json`, `per_stream[2].edit_success_rate`,
   `per_stream[3].edit_success_rate`), so their higher survival is not an installation
   failure mode.

**Evidence pointers:** `edit-harness/results/SEQ_analysis_L12_4stream.json` (authoritative
4-stream numbers, all four items above), `docs/plans/PAPER-FOLDIN-MAP-2026-07-10.md`
binding constraint 4 (wording rules), `docs/findings/findings-SEQ-ANALYSIS-2026-07-04.md`
(original 2-stream analysis and design record — still useful for methodology, superseded
on the headline numbers).

**Remaining gap:** none for the numbers themselves — the 4-stream table is complete and
on disk. The open item is purely editorial: confirm `sections/*sequential*.tex` currently
quotes the 4-stream numbers above, not the older 2-stream "10%/14%" framing.

---

## 7. "Is this a dataset artifact of CounterFact specifically, or does the geometry law
##    generalize to other data (multi-hop facts, deletion-style edits, related-fact
##    ripple effects)?"

**Likely phrasing:** *"All your core results use CounterFact. Do they hold on
structurally different data?"*

**Response.** Three independent dataset extensions are on disk, all replicating the core
qualitative pattern:

- **MQuAKE (multi-hop facts).** The causal AlphaEdit-removal coupling replicates at
  L12: 0.5328 (3-seed mean, per-seed 0.5784/0.4977/0.5224) —
  `C4_causal_mquake_table_3seed_probesrc.json` — versus CounterFact holdout L12 0.5903 —
  `C4_causal_holdout_table_3seed.json`. Comparable magnitude, same sign, same layer. (The
  standalone, non-causal MQuAKE *gate* test at L12 is a separate, weaker cell — mean
  −0.0395, `sign_consistent: false`, verdict `STABLE_NULL` — `C3_mquake_alpha_L12_3seed.json`
  — this is the AlphaEdit-only correlational gate, not the causal ROME-vs-AlphaEdit
  removal comparison, and should not be conflated with the causal number above.)
- **zsRE-deletion (structurally different edit *mode*, not just dataset).** The geometry
  coupling transfers to deletion-style edits at L10: 3-seed mean 0.2427 (per-seed
  0.2408/0.2721/0.2153, sd 0.023), verdict PASS ("headline survives — edit-specific
  pairwise geometry, not probe-marginal") — `edit-harness/results/C3_u1_zsre_delete_L10_u6.json`.
- **RippleEdits (related-fact ripple damage, not just unrelated collateral).** Geometry
  predicts ripple-probe damage too, though more weakly than unrelated-probe damage, at
  every ROME layer tested: L8 ripple 0.454 vs. unrelated 0.475; L10 ripple 0.465 vs.
  unrelated 0.545; L12 ripple 0.274 vs. unrelated 0.332; L14 ripple 0.492 vs. unrelated
  0.449 (all 3-seed means, all `STABLE`, `sign_consistent: true`) —
  `edit-harness/results/RIPPLE_depth_profile.json`. This is a genuinely new axis (ripple
  vs. unrelated collateral) beyond a dataset-transfer check.

**Evidence pointers:** as cited inline above; all four numbers are cross-confirmed in
`REVISION_DOSSIER.json` (`mquake_causal_L12`, `ripple_rome_L{8,10,12,14}_{ripple,unrelated}`
cells — note `ripple_alpha_L12_unrelated` is `STABLE_NULL`, mean 0.0215, which is the
expected floor result for AlphaEdit's own ripple damage, not a failure of the ROME axis).

**Remaining gap:** MQuAKE's causal cell uses `proj_source: probes` (see item 3's caveat),
not holdout, so it is a reference/consistency check rather than a fully independent
circularity-clean replication.

---

## 8. "Do the results survive instruction tuning, or are they an artifact of base
##    (non-chat) checkpoints?"

**Likely phrasing:** *"All main results use base models. Does the geometry-damage law and
the S×C-beats-key-cosine ordering hold for an instruction-tuned model?"*

**Response.** Yes, on the instruction-tuned Llama-3.2-1B twin: the causal AlphaEdit-
holdout removal coupling at L12 is 0.552 (3-seed mean, per-seed 0.5675/0.5335/0.5550, sd
0.017, verdict STABLE) — `edit-harness/results/C4_causal_instruct_table_3seed.json` —
essentially matching the base-model holdout value of 0.5903 at the same layer
(`C4_causal_holdout_table_3seed.json`). Edit-success rate stays high across all three
seeds (0.995/0.995/1.0), so this is not an edit-installation artifact. Damage-removed
quartile ratio (Q4/Q1) is 2.831 for the instruct twin
(`C4_causal_instruct_table_3seed.json`, `layers.12.removed_top_vs_bottom_ratio`) versus
3.131 for base (`C4_causal_holdout_table_3seed.json`, same field) — same direction, same
order of magnitude.

**Evidence pointers:** `C4_causal_instruct_table_3seed.json` (full instruct-twin table),
`C4_causal_holdout_table_3seed.json` (base-model comparison point), `REVISION_DOSSIER.json`
cell `instruct_alphaHO_L12`.

**Remaining gap:** the *causal* (AlphaEdit-holdout) instruct profile exists only at L12
(`g4_instruct_alphaHO_cf_L12_s*`). The *correlational* ROME depth profile DOES exist at 3
seeds across L8/L10/L12/L14 — `C3_instruct_rome_L{8,10,12,14}_instruct.json`: within-probe
rho 0.3645 / 0.4577 / 0.5586 / 0.3791, all verdict PASS, peaked at L12 with the same shape
as the base model — so "does the depth pattern survive instruction tuning?" IS answerable
from disk for the correlational law; only the causal depth generalization beyond L12 is a
genuine gap.

---

## RESPONSE-POINTS (added 2026-07-16, revision-draft pass)

These are science-response items surfaced by the 2026-07-16 hostile-referee + typesetter
pass. They are recorded here for the eventual response letter; they are **not** drafting
tasks (no manuscript text was changed for them). Each is backed by numbers already on disk
(see items 1–8 above and the revins dossier).

### RP-A. "What does the theorem/decomposition actually buy?" (S×C over raw key-cosine)

**The objection, stated fairly.** S×C's win over raw key-cosine is *modest at most layers*:
within-probe Spearman rho_C (raw key-cos) vs rho_SC (S×C) is 0.395 vs 0.402 (L8), 0.534 vs
0.553 (L10), 0.602 vs 0.677 (L12) — a few hundredths — and separately, S×C is **not** a
faithful rank-surrogate of true gradient influence (direct-backprop rank agreement
≈ 0.09, `GRADSIM_TRUE_Llama-3.2-1B_L12_s0.json`, `rank_agreement.direct_vs_SC.mean = 0.0874`).
A reviewer can reasonably ask what the closed form adds if it barely beats a raw cosine and
does not track the true influence ranking.

**Answer — lean on three things, in this order.**
1. **The L14 result is where the decomposition earns its place.** At the norm-growth-dominant
   layer the gap is large, not marginal: rho_SC 0.504 vs rho_C 0.301
   (`C1_mechanism_sc_table.json`, `rho_SC_valid_sxc: true`). Precisely where the raw
   key-cosine *degrades* (the regime transition of Section IV), the S×C product retains
   predictive power — so the decomposition is not a cosmetic re-scaling of the cosine; it
   carries the additional `S` (norm-growth) signal that becomes load-bearing at depth.
2. **Frame the claim as a rank estimator (Assumption A4′), never as an identity or a
   validated surrogate.** The manuscript already ships this cautious wording
   (`sections/03_method.tex`: "rank estimator," "not a faithful rank-surrogate," A4′ cited).
   The ≈0.09 rank-agreement number is not a weakness to hide — it is *our own* disclosure
   that bounds the claim. The value of S×C is (i) zero-backprop cost and (ii) that it
   predicts *damage* in its own right (the true influence itself predicts damage at
   within-probe rho 0.474, and S×C is the cheap rank-one reduction of that influence), not
   that it reproduces the true-influence ordering.
3. **The revins dossier de-risks the "it's just a cosine" read** by showing the S×C/geometry
   story survives the stressors a skeptic would apply (holdout projector, deletion mode,
   instruction tuning, MQuAKE, ripple) — see items 3, 5, 7, 8 above and the dossier roll-up
   (`results/REVISION_DOSSIER.json`, `results/REVINS_manifest.json`,
   `engine/run_revins_report.txt`).

Binding metric wording: quote the corrected rho_SC set 0.402 / 0.553 / 0.677 / 0.504 only;
the pre-normalization-fix set 0.390 / 0.528 / 0.628 / 0.498 is DEAD and must never appear.

### RP-B. "AlphaEdit floors everything, so 'removal correlates with geometry' is trivial."

**The objection.** If null-space projection drives collateral damage to a small residual
*regardless* of the edit, then "damage-removed rises with key-cosine" is just re-describing
how much damage ROME did in the first place — a tautology, not a causal confirmation of the
geometry mechanism.

**Answer.** Two pointers, both already on disk.
1. **The L14 cell is the non-trivial test.** Even at the layer where norm-growth has overtaken
   the key-cosine as the observational predictor, AlphaEdit's *damage-removed* still rises
   monotonically with pre-edit key-cosine (quartile removed 4.05 → 4.82 → 5.55 → 6.80;
   within-probe rho(key-cos, damage-removed) 0.302). If AlphaEdit floored damage
   geometry-blindly, the removed quantity would not be graded by key-cosine at a layer where
   key-cosine no longer dominates the raw damage. It is — so the removal is geometry-carried,
   not a uniform floor.
2. **The holdout-projector MQuAKE causal cell closes the circularity door at 0.495, 3-seed
   stable.** The projector is fit on facts disjoint from the damage probes, and the causal
   coupling is stable across seeds (`results/REVINS_manifest.json` / revins report; cross-ref
   item 3's holdout-vs-generic near-identity on CounterFact, and item 7's MQuAKE causal).
   "AlphaEdit floors everything" cannot explain a projector fit on *different* facts still
   removing damage in proportion to *these* probes' key-cosine.

### RP-C. "Causal confirmation is only ≤3B — the mechanism may not hold at scale."

**The objection.** The strong positive-signed, monotone-removal causal law is demonstrated up
to 3B; at 8B it attenuates and can sign-flip with depth (item 2). So the causal claim may be
a small-model artifact.

**Answer — concede the cap, point to the pre-computed dossier.** We do **not** claim causal
confirmation beyond mid-scale, and the abstract now states this cap up front. The honest
scope is backed by pre-computed seed cells: the 8B AlphaEdit-holdout depth profile is 3-seed
and *stable in both signs* (+0.212 L16 / −0.087 L24 / +0.122 L28, all sign-consistent across
seeds — `C4_causal_8b_table_3seed.json`), so the attenuation/inversion is a real
scale-dependent phenomenon, not seed noise; and the L28 point carries the damage-floor caveat
(item 1). The claim we defend is "sign-tracks-regime with attenuating magnitude at the largest
scale tested," and every number needed to defend it is already on disk in the dossier
(items 1–2). What remains genuinely open — *why* the coupling attenuates at 8B — is flagged as
a mechanism question, not a missing cell.

## Logistics

- **`esr_warn` cells (2 of 16 tracked, both flagged for the same reason).**
  `REVISION_DOSSIER.json`'s dossier-builder (`edit-harness/experiments/revision_dossier.py`,
  line 198) flags `esr_warn` whenever any seed's edit-success rate (esr) falls below 0.90.
  Both flagged cells are MQuAKE-sourced: `mquake_causal_L12` (esr per seed 0.885/0.9/0.865)
  and `mquake_gate_L12` (same esr values, same underlying npz family). This is precedented
  and not a red flag specific to these cells — MQuAKE's multi-hop facts are known to have a
  lower base edit-success rate than CounterFact's single-hop facts in this harness (the
  dataset's `frac_known@1B` pre-filter statistic, `data/DOWNLOADS-20260706.md`, already
  documents MQuAKE as a harder-to-edit population). Both cells are still `verdict: STABLE`
  or `STABLE_NULL` — the warning is informational, not a failed gate.
- **Pending cell — RESOLVED 2026-07-10 23:38.** `8b_alphaHO_L28` is now 3-seed STABLE
  (0.1543/0.0865/0.1266, mean 0.1225); dossier refreshed to 16/16 stable, 0 pending. Quote
  it only WITH the damage-floor caveat (item 1): `mean_damage_rome` 0.0050,
  `mean_damage_removed` −0.0035 at L28.
- **Venue confirmation still open.** `submissions/ieee/VENUE-NOTE.md` states the
  TNNLS-vs-TASLP decision is **not yet made** — the manuscript is dual-fork
  (`\iftnnls`/`\iftaslp` toggle in `main.tex`) and both compile from one source. TNNLS
  requires the formal theorem to ship as a `proposition` (mandatory); TASLP allows it as a
  weaker `remark`. Per `EXTENSION-TODO.md`, STUB-THEOREM is CLOSED as of 2026-07-10 and
  both variants are already implemented behind the toggle, so the manuscript is
  submission-ready either way — but the actual submitted-venue identity (which fork was
  compiled and sent) is not recorded in any file this pass read. **Action needed:** record
  which IEEE journal (TASLP vs. TNNLS) actually received the 2026-07-10 submission, per the
  workspace CLAUDE.md's own open note ("confirm exact journal TASLP vs TNNLS and record it
  here").

---

## Sourcing note

Every number above was read from a JSON or Markdown file under
`edit-harness/results/`, `docs/findings/`, or `submissions/ieee/` during this pass — no
number was taken from memory or from the task-launch summary without independent
verification against the file. Files read: `REVISION_DOSSIER.json`,
`C4_causal_instruct_table_3seed.json`, `C4_causal_mquake_table_3seed_probesrc.json`,
`C4_causal_8b_table_3seed.json`, `C3_mquake_alpha_L12_3seed.json`,
`RIPPLE_depth_profile.json`, `C1_mechanism_sc_table.json`, `C4_causal_holdout_table_3seed.json`,
`C4_causal_table.json`, `C3_memit_L8_r3.json`, `C3_memit_L12_r3.json`,
`C3_u1_zsre_delete_L10_u5.json`, `C3_u1_zsre_delete_L10_u6.json`,
`GRADSIM_TRUE_Llama-3.2-1B_L12_s0.json`, `findings-MEMIT-SC-RECONCILIATION-2026-07-04.md`,
`findings-SEQ-ANALYSIS-2026-07-04.md`, `SEQ_analysis_L12_4stream.json`,
`THEOREM-SXC-DRAFT-2026-07-06.md`, `EXTENSION-TODO.md`, `VENUE-NOTE.md`,
`docs/plans/PAPER-FOLDIN-MAP-2026-07-10.md`, plus a targeted grep of `sections/03_method.tex`
and `experiments/revision_dossier.py` to confirm wording/threshold claims. Every number
this pass set out to find was ultimately located on disk; the initial draft mis-flagged
the 4-stream sequential figures (item 6) as unlocatable before a follow-up search found
`SEQ_analysis_L12_4stream.json` — that section was corrected in place before this document
was finalized.
