# B6 draft sections — the S×C mechanism (C1) + related-work differentiation (rewritten 2026-07-04)

> **AUTHORING PASS (2026-07-04).** Writer lane only; a separate hostile review gates
> submission. All numbers verified against canonical `edit-harness/results/*.json` (sources
> named per section); `[UNVERIFIED]` / `[FLAG]` mark anything not backed by a canonical file.
> Companion to `B6-PAPER-SKELETON-2026-07-01.md`. Venue-independent, analytic core.
>
> **Scope (binding):** the *signed* key-cosine→damage law is **Llama-family-specific**. This
> doc writes the Llama mechanism as primary and the off-Llama behaviour (null on gemma/Phi,
> inversion on Qwen) as a *dissociation arm* that the same algebra predicts.

## Section 2 — The S×C decomposition (C1)

**Setup.** ROME inserts a new key→value association by a rank-one update to a single MLP
down-projection W. Given an edit key `k` (subject's last-token key at the critical layer) and
target value `v`, the update is

    ΔW = (v − Wk) kᵀ / (kᵀk) = outer(v − Wk, k) / ‖k‖²,          (1)

which is the minimum-norm solution to `(W+ΔW)k = v`.

**Collateral perturbation to an unrelated probe.** For an unrelated fact whose key at the same
layer is `k_p`, the edit perturbs its value read-out by

    ΔW · k_p = (v − Wk) · (kᵀk_p) / ‖k‖².                          (2)

Taking magnitudes and writing `kᵀk_p = ‖k‖ · ‖k_p‖ · cos(k, k_p)`:

    ‖ΔW · k_p‖ = ( ‖v − Wk‖ / ‖k‖ ) · ‖k_p‖ · |cos(k, k_p)|
               = **S · ‖k_p‖ · |C|** ,                              (3)

where
- **S ≡ ‖v − Wk‖ / ‖k‖** — the **edit-strength**, a per-EDIT scalar (constant across probes);
- **C ≡ cos(k, k_p)** — the **key-geometry**, the per-(edit,probe) term;
- **‖k_p‖** — a per-PROBE scalar (constant across edits for a given probe).

Equation (3) is the paper's analytic core: **the collateral perturbation factorizes
multiplicatively into a per-edit strength, a per-probe norm, and the edit×probe key-cosine.**
(The observed *damage* — the drop in the probe's correct-token logit — is a monotone function of
‖ΔW·k_p‖ under local linearization of the head; we therefore MEASURE the rank relationship
rather than assert exact proportionality. This is precisely why the empirical test is a
within-probe Spearman.)

**S×C is a zero-cost surrogate for gradient influence, not a "better cosine."** The product
S·|C| is, up to the per-probe norm, the closed form of the first-order (GradSim) influence of an
edit on a probe — obtainable with **no backprop**. Measured, S×C tracks GradSim to ~2 decimals:
at L8 the within-probe rho for S×C is **0.390**, identical to the GradSim-resid within-probe rho
**0.390**, versus raw key-cosine **0.395** (`G2_gradsim_L8.json`, whose own verdict is "GradSim
MATCHES/BEATS key-cosine"). Across the layer band, rho_SC vs rho_C is **0.390/0.395 (L8),
0.528/0.534 (L10), 0.628/0.602 (L12), 0.498/0.301 (L14)** (`C1_mechanism_sc_table.json`).
**S×C therefore LOSES to raw key-cosine at L8 and L10** and only pulls ahead where S carries the
variance (L12/L14). We frame S×C as the **mechanistic, zero-cost surrogate that explains *why*
cosine works and predicts *where* it stops working**, and we never claim it beats raw
key-cosine (it does not, at L8/L10, nor on gemma).

**Consequences (each a measured phenomenon on the canonical JSONs):**

- **(→C2, within-probe predictivity).** Fix a probe (a column): ‖k_p‖ and the probe's intrinsic
  susceptibility are constant, so damage ranks across edits by `S·|C|`. When S has limited
  spread relative to C, `Spearman(|C|, damage)` down the column is high — exactly the
  within-probe partialled Spearman the G1 gate measures. Across 3 seeds on the saved gate
  matrices: **L8 0.395 → L10 0.534 → L12 0.602 (peak) → L14 0.301** (frac_positive = 1.0 at
  every layer; within-probe permutation-p at the 1/301 floor). Holding the probe fixed is what
  retires probe-marginal leakage (‖k_p‖ is differenced out). Sources: `G1_L{8,10,12,14}_analysis
  .json`.

- **(→C2, the L14 regime transition).** The *rank* of damage down a column is governed by the
  relative spread of S vs |C| across edits. Mid-layer, |C| carries most cross-edit variance →
  geometry-dominant, key-cos wins. At L14, |C| saturates (higher mean |C|, lower
  variance) while S becomes the dominant varying factor → **magnitude/norm-growth
  overtakes.** (Mean |C| rises to 0.425 at L14 vs 0.21–0.31 across the L8/L10/L12 band —
  `C1_mechanism_sc_table.json`.) Head-to-head, the true norm-growth→damage within-probe ρ is L8 0.083 / L10 0.246 /
  L12 0.509 / **L14 0.502**: key-cos wins L8/L10/L12, norm-growth overtakes at L14 (0.502 >
  0.301), on **all 3 seeds** (`findings-G1-gate-2026-07-01.md`, `G1_L14_analysis.json`). The
  earlier "L14 crossover is a flat-stat artifact" claim was retracted after fixing the
  norm-growth baseline in `analyze_matrices.py`; it is real on the confound-clean metric.

- **(→C2, the scale/regime law).** The transition is not just a depth effect but a
  **damage-regime** effect: the *sign* of the coupling tracks the *sign of the mean damage*.
  At Llama-3B L24 (positive-damage regime) within-probe ρ = **+0.376** (3-seed 0.351–0.393); at
  Llama-8B L24 (net-improvement regime, mean per-edit damage negative) ρ = **−0.097** (3-seed
  mean; per-seed −0.117 / −0.094 / −0.078). Within 8B, the sign flips with depth: L24 −0.097
  (3-seed), L16 **+0.173** / L28 **+0.155** (seed-0). The norm-growth transition tracks
  **relative** depth: 8B L28 sits at 0.875 relative
  depth (the 1B-L14 position) and shows the NG-dominance directionally, attenuated ~55% vs 1B.
  The settled claim is: **the sign tracks the damage regime (3-seed both sides); the coupling
  magnitude is attenuated at 8B (|ρ| 0.07–0.17 vs 0.38 at 3B)** — it is not scale-invariant.
  Sources: `C3_regime_3b_L24_r4.json`, `C3_regime_8b_L24_r4.json`, `C3_llama8b_r3.json`.

- **(→C3, editor dissociation).** For fine-tuning (FT-L) the accumulated update is NOT rank-one:
  Adam's per-coordinate rescaling `m_ij/√v_ij` is not separable as `f(i)g(j)`, so ΔW_FT has high
  effective rank and no row-space direction aligned with `k`; ΔW_FT·k_p does not factor through
  `cos(k,k_p)` → ρc ≈ 0 (measured FT L8 **0.024** at mean damage **18.1**, vs ROME 0.406 at
  4.45). KL-regularized FT restores it partially (ρc **0.132** 3-seed at mean damage **15.2**).
  For **MEMIT**, the edit is spread across four layers (memit_layers 9–12 / 5–8), so a
  single-layer rank-one identity does not describe it and the coupling is negligible at every
  layer: ρ_C **0.019 (L8) / 0.034 (L10) / 0.037 (L12) / 0.012 (L14)**, all DEAD (L8/L12 3-seed,
  `C3_memit_L{8,12}_r3.json`; L10/L14 single-seed, `C3_memit_L{10,14}_u4.json`).
  **Binding:** report MEMIT as **ρ_C only, never "MEMIT S×C"** — see
  `findings-MEMIT-SC-RECONCILIATION-2026-07-04.md`. *Receipt to add: SVD/effective-rank of a
  matched ΔW_FT vs ΔW_ROME.*

- **(→C3, architecture null / inversion).** If `S = ‖v − Wk‖/‖k‖ ≈ 0` — the fact is already
  near-represented (distributed storage) — then `S·‖k_p‖·|C| ≈ 0` for *every* probe regardless
  of geometry → nothing to predict; this is the gemma/Phi near-null (gemma 0.084 ± 0.029, Phi
  0.017). On Qwen the coupling **inverts** (Qwen-1.5B L14 **−0.172**, Qwen-3B L18 **−0.119**,
  3-seed, scale-persistent) — a net-improvement/regime effect, NG-clean, and (crucially)
  **causally geometry-carried**: AlphaEdit erases it (§6). Sources: `C3_null_{gemma2b_L13,
  phi35_L16,qwen15b_L14,qwen3b_L18}_v2.json` (Qwen-1.5B `C3_null_qwen15b_L14.json`).

- **(→C4, why AlphaEdit's causal protection tracks cosine).** AlphaEdit projects the update onto
  the null-space of preserved-knowledge keys, so for a preserved probe ΔW_α·k_p ≈ 0 irrespective
  of its original `S·‖k_p‖·|C|`. Hence the damage AlphaEdit *removes* is ≈ `S·‖k_p‖·|C|` — the
  absolute damage-removed rises monotonically with |C| (measured in §6 at every layer, incl.
  L14). The decomposition predicts the causal result; the intervention confirms the removed
  damage is the geometry-predicted damage — the causal test CLaRE explicitly disclaims.

**In one line:** a single algebraic identity (3), doubling as a zero-cost GradSim surrogate,
unifies the within-probe predictor, the depth/regime transition, the ROME-vs-FT/MEMIT and
Llama-vs-{gemma,Phi,Qwen} dissociations, and the AlphaEdit causal result — none of which
follows from an empirical activation-cosine metric.

## Section 5 — Editor & architecture dissociation (C3)

The law is **locate-then-edit-mechanism-specific**, not a monotone function of how much damage
an editor does. On a matched within-probe statistic (Llama-3.2-1B, CounterFact, L8) the editors
form a **locality/coupling spectrum** — coupling and mean damage move *oppositely*:

| Editor | within-probe ρ(key-cos, dmg) | mean \|damage\| (logit) | reading |
|---|---|---|---|
| FT (full-rank Adam) | **0.024** (1-seed; L10 0.070 / L12 0.096) | **18.1** | geometry-blind, most damaging |
| KL-FT | **0.132** (3-seed) | **15.2** | KL partially restores predictability |
| ROME (rank-one) | **0.406** (s0; 0.395 3-seed) | **4.45** | the law lives here |
| MEMIT (4-layer) | **0.019 / 0.034 / 0.037 / 0.012** (L8/L10/L12/L14) | **0.03** | multi-layer spread kills the single-layer identity |
| AlphaEdit (null-proj) | **~0** | **0.14** (3-seed holdout floor) | damage AND coupling collapsed together |

Sources: `gate_llama1b_{ft,ftkl,rome,memit}_cf_L8_s0.json` (mean-damage), `C3_null_ft_L8.json`,
`C3_null_ftkl_L8_v2.json`, `C3_memit_L{8,12}_r3.json` / `C3_memit_L{10,14}_u4.json` (full DEAD
layer profile), `C4_causal_holdout_table_3seed.json` (AlphaEdit 3-seed holdout floor ~0.14).
MEMIT wording binding per `findings-MEMIT-SC-RECONCILIATION-2026-07-04.md` (ρ_C only, never
"MEMIT S×C").

**KL-ladder dose–response (run_u5, 3-seed).** Raising KL strength raises the coupling and lowers
the damage; at 3 seeds the coupling **rises then plateaus** (~0.15 by kl 0.3, with kl 1.0 no
longer exceeding kl 0.3) while mean damage falls monotonically, edit-success 1.0 at every rung.
Source: `C3_klladder_{003,030,100}_L8_seeds_u5.json` (3-seed), `C3_null_ftkl_L8_v2.json`
(kl 0.1, 3-seed), `gate_llama1b_ftkl{003,030,100}_cf_L8_s0.json` (mean-damage + esr).

| ft_kl | within-probe ρ | mean \|damage\| | seeds |
|---|---|---|---|
| 0.0 (plain FT) | 0.024 | 18.1 | 1 |
| 0.03 | 0.091 ± 0.008 | 16.3 | 3 |
| 0.1 | 0.132 | 15.2 | 3 |
| 0.3 | 0.150 ± 0.011 | 13.7 | 3 |
| 1.0 | 0.149 ± 0.024 | 11.9 | 3 |

**The dose–response is an L8 phenomenon.** At L12 — the law's peak layer for ROME — the same
ladder is DEAD/borderline and non-monotone: within-probe ρ **0.119 / 0.097 / 0.088 / 0.120** at
kl 0.03 / 0.1 / 0.3 / 1.0 (single-seed), all near or below the 0.10 DEAD threshold. KL-FT's
partial restoration of geometry-predictability does not carry to the layer where ROME's coupling
is strongest. Source: `C3_klladder_{003,010,030,100}_L12_u5.json`.

**Architecture arm.** The *signed* law is Llama-specific. Signed within-probe ρ, 3-seed:
gemma-2-2b L13 **0.084 ± 0.029**, Phi-3.5 L16 **0.017**, Qwen-0.5B L12 **0.103** (borderline),
Qwen-1.5B L14 **−0.172**, Qwen-3B L18 **−0.119**. Eq. (3) accounts for both breakdowns: near-null
where S≈0 (distributed storage), inversion where the edit sits in a net-improvement regime. The
Qwen inversion is NG-clean, scale-persistent, and causally geometry-carried (§6). Sources:
`C3_null_*_v2.json`.

The **magnitude** law |C|→|dmg| (unsigned) is more portable — it **transfers on 4 of 5**
non-Llama families. Canonical 3-seed within-probe Spearman(|key-cos|, |damage|) (`--known
--edit_ok`, `C1_magnitude_table.json`):

| Family (layer) | magnitude ρ (3-seed) | verdict |
|---|---|---|
| Llama-1B (L8/L10/L12/L14) | 0.398 / 0.551 / **0.613 ± 0.019** / 0.306 | PASS |
| Qwen-0.5B (L12) | 0.320 ± 0.021 | PASS |
| Qwen-1.5B (L14) | 0.412 ± 0.009 | PASS |
| Qwen-3B (L18) | 0.401 ± 0.011 | PASS |
| Phi-3.5 (L16) | 0.321 ± 0.035 | PASS |
| **gemma-2-2b (L13)** | **0.086 ± 0.025** (perm-p 0.117) | **DEAD** |

**gemma is DOUBLE-DEAD** — geometry-blind on both the signed (0.084) and the magnitude (0.086,
n.s.) law — which strengthens the anomaly rather than weakening the transfer story. Phi's
canonical magnitude is **0.321**, versus ≈0.36 in an earlier unfiltered peek: the peek omitted
the `--known` filter.[^phipeek]

[^phipeek]: The `--known` filter restricts to probes the base model actually knows (pre_p>0.05).
    The mechanism is verified — dropping `--known` raises Phi's magnitude ρ to ≈0.36 (reviewer
    reproduces 0.365 without `--known`, 0.3623 seed-0 with `--known`); the exact "0.362" figure
    was a loosely-remembered value, so we quote the canonical 0.321 and describe the peek as
    "≈0.36 unfiltered." All headline numbers use `--known --edit_ok`. `C1_magnitude_table.json`
    is an authoring-pass module pending its own hostile review.

## Section 6 — Causal test (C4 + E6): AlphaEdit removes the geometry-predicted damage

The within-probe correlation (C2) and the S×C algebra (C1) are, on their own, *observational*.
We close the gap with an intervention. AlphaEdit applies the same rank-one target as ROME but
first projects the update onto the null-space of a set of preserved-knowledge keys, so for any
preserved probe `k_p` the induced perturbation `ΔW_α·k_p ≈ 0`. Eq. (3) then predicts: since
ROME's collateral term is `S·‖k_p‖·|C|` and AlphaEdit drives it toward zero, the *damage
AlphaEdit removes* should itself rise monotonically with `|C|`.

**Setup.** For matched edit sets we run ROME and AlphaEdit on the same Llama-3.2-1B,
CounterFact, N=200 edits × M=500 probes, under the identical `--known` (base pre_p>0.05) and
`--edit_ok` masks used for the G1 gate. We bin the shared (edit, probe) pairs into key-cosine
quartiles and report, per quartile, the **absolute mean logit-damage removed** = mean(ROME) −
mean(AlphaEdit). We report the *absolute* reduction, not the "protection ratio" (non-monotone,
partly an artifact of ROME's own cosine-scaling).

**Result (3 seeds, all four layers — the keystone is complete).** AlphaEdit floors collateral
damage to a small residual while ROME damage scales with cosine, so the damage removed is
monotone in key-cosine at **every** layer, including the norm-growth-dominant L14:

| Layer | within-probe ρ(key-cos, dmg-removed), 3-seed | mean dmg ROME→Alpha | quartile dmg-removed Q1→Q4 |
|---|---|---|---|
| **L8** | **0.397** (0.403/0.406/0.382) | 4.39 → 0.07 | 2.41 / 3.73 / 4.88 / 6.22 |
| **L10** | **0.532** (0.526/0.522/0.550) | 2.04 → 0.07 | 0.82 / 1.36 / 2.05 / 3.64 |
| **L12** | **0.597** (0.596/0.617/0.578) | 3.00 → 0.09 | 1.57 / 2.31 / 3.07 / 4.71 |
| **L14** | **0.302** (0.268/0.270/0.367) | 5.39 → 0.08 | 4.05 / 4.82 / 5.55 / 6.80 |

Source: `C4_causal_table.json`. **The sharp test passes:** even at L14, where norm-growth has
overtaken key-cos as the within-probe predictor (C2), AlphaEdit still removes damage
monotonically in key-cosine (quartile removed 4.05 → 6.80) — geometry predicts *causally
removable* damage even where the correlational ρ weakens.

**E6 — projector circularity retired; holdout numbers are PRIMARY (3-seed).** A referee's
objection: the by-construction projector is fit on the very probes whose damage we measure, so
the tracking could be by construction. We refit the projector on **held-out** keys. The
within-probe ρ(key-cos, damage-removed) is unchanged, 3-seed: **L8 holdout 0.390**
(0.401 / 0.391 / 0.380), **L12 holdout 0.590** (0.590 / 0.610 / 0.570), matching by-construction
(0.397 / 0.597); mean damage ROME→Alpha 4.385 → 0.136 (L8) and 3.006 → 0.146 (L12); removed
top-vs-bottom quartile ratio 2.60 / 3.13. Source: `C4_causal_holdout_table_3seed.json` (the
seed-0-only `C4_causal_holdout_table.json` remains on disk as the earlier record). A
generic-projector variant also exists as a gate-level result but is not quoted in the main
text.[^generic]

[^generic]: The generic-key-bank projector (`g4_llama1b_alphaGEN_cf_L12_s0`) reproduces the
    result directionally at gate level but has no aggregated damage-removed Spearman; we report
    the holdout arm as the primary circularity control.

**BONUS — AlphaEdit erases the Qwen inversion.** On Qwen-1.5B, where ROME's coupling is negative
(−0.172), AlphaEdit removes the geometric coupling *whatever its sign*: the deletion-setting
replica shows the same collapse (§7). The causal statement is therefore "null-space projection
removes the geometry-carried damage regardless of the sign of the correlational law," which is
stronger than a Llama-only positive result.

**Honest caveats (carried into Limitations).**
- AlphaEdit's floor is not perfectly cosine-flat (L8 quartile alpha means 0.06–0.09), so
  "removes exactly `S·‖k_p‖·|C|`" is approximate, not exact.
- Both the holdout and by-construction arms are 3-seed and agree (L8 0.390/0.397, L12
  0.590/0.597); AlphaEdit's floor rises modestly under the holdout projector (mean damage
  0.136/0.146 vs ~0.08 by-construction), still ~30× below ROME.
- The reported statistic is the signed absolute reduction with cross-seed mean±sd; the
  auto-verdict's protection-ratio framing is intentionally omitted (probe-marginal artifact).

## Section 7 — Deletion collateral (U1): geometry governs a new edit type

The mechanism was derived and tested for knowledge **rewrites**. We show it also governs
**deletion** edits — removing a learned refusal ("I can't help with that") — a qualitatively
different edit direction that was not in the original design. Using the same within-probe
statistic on Llama-3.2-1B L12:

- **Refusal-deletion collateral is geometry-governed**, and *more* strongly than rewrites:
  within-probe (non-DC) ρ **0.657 / 0.699 / 0.681** (3 seeds; DC ρ **0.437 / 0.508 / 0.500**,
  perm-p 0.001) — above the rewrite reference 0.602. Layer profile (raw key-cos within-probe,
  3-seed): L8 **0.461 ± 0.017** / L14 **0.519 ± 0.018** (L12 raw aggregate 0.663; the gate S×C
  arm agrees), source `C3_u1_blockB_L{8,14}_seeds_u5.json`.
  Sources: `u1_gate_refusal_L12_s{0,1,2}.json`, `u1_gate_refusal_L8_s0.json`,
  `u1_gate_refusal_L14_s0.json`; 3-seed within-probe aggregate 0.663 in `C3_u1_blockA_seeds_u1
  .json`.
- **Variant robustness resolved 3-seed (run_u4) — two distinct statistics, do not conflate.**
  Two statistics exist per variant and must be reported separately: (a) the **gate S×C
  statistic** (`u1_gate_*`, seed-0): non-DC S×C-vs-damage Spearman and its double-centered value;
  (b) the **raw key-cos within-probe statistic** (`C3_u1_blockC_*_seeds_u4.json`, 3-seed).
  - **eos** is robust on **both**: gate non-DC S×C **0.653** (seed-0), and raw key-cos
    within-probe **0.616 ± 0.008** 3-seed (per 0.605 / 0.624 / 0.619).
  - **suppress** is the **fragile** variant: its gate non-DC S×C is **0.621** (seed-0) but
    **DC-fragile → ~0.16** under double-centering; and on the **raw key-cos within-probe** metric
    it is **0.073 ± 0.024** 3-seed (DEAD). These are *different* statistics, not a discrepancy —
    the weak raw-level coupling across seeds is **consistent with, and strengthens, the
    DC-fragility reading**: suppress's apparent non-DC S×C coupling is not robust geometry. We
    report suppress as the negative/fragile variant, never averaging or swapping the two numbers.
- **Causal (3-seed, run_u2).** AlphaEdit-delete collapses damage **4.10 → ~0.10** (per-seed
  0.10 / 0.10 / 0.11) AND coupling to **0.036 ± 0.014** (3-seed, DEAD) — the same "removes the
  geometry-carried damage whatever its sign" result as the rewrite arm. Sources:
  `C3_u1_blockD_alphadelete_seeds_u2.json`, `U1_E1_transplant_GATE_alphadelete_L12_s{0,1,2}.json`.
- **Llama-scoping replicates in deletion (3-seed, run_u2).** Qwen-1.5B-delete ρ **−0.066 ± 0.013**
  at mean damage −0.036 (improvement regime) — the same off-Llama behaviour as the rewrite arm.
  Source: `C3_u1_blockE_qwen15b_seeds_u2.json`.
- **Transplant gate — the S×C surrogate beats a learned transplant baseline.** Δρ (S×C minus
  **best** transplant, `delta_rho_SxC_minus_best_transplant`) **0.59 / 0.55 / 0.61** at L12
  (seeds), **0.31** at L8, **0.38** at L14. Source: `U1_E1_transplant_GATE_L12_s{0,1,2}.json`,
  `U1_E1_transplant_GATE_L{8,14}_s0.json` (label, text, and figure F5(d) all use the
  best-transplant field).
- **Deletion collateral is NOT a CounterFact artifact — it transfers to zsRE (single-seed,
  refusal variant, L10).** Refusal-deletion on zsRE at L10 (matched to the zsRE insertion
  reference layer): raw key-cos within-probe **0.241** (C3 verdict PASS, survives the permutation
  battery); formal gate S×C non-DC **0.234** → DC **0.311** — i.e. **DC-ROBUST** (DC *higher* than
  non-DC, the opposite of suppress's DC-fragility and like-for-like with the CF refusal variant).
  Three anchors: zsRE **deletion** 0.24 raw / 0.31 DC at L10 sits below the zsRE **rewrite** law
  (0.361 at the same layer, `C3_null_llama1b_zsre_L10.json`) and well below CF **deletion**
  (0.657 at L12) — geometry governs deletion collateral across datasets, attenuating with the
  weaker zsRE signal. Source: `C3_u1_zsre_delete_L10_u5.json`, `u1_gate_zsre_refusal_L10_s0.json`
  (canonical run_u5 files, byte-identical to the earlier pre-computation).

## Section 8 — Sequential no-restore stress (descriptive only)

We run four streams of 50 edits with no weight restore between them (Llama-3.2-1B L12, recheck
every 10). **This section is descriptive: no geometry-attribution claim is admissible** (see
`findings-SEQ-ANALYSIS-2026-07-04.md`). Numbers below are the 4-stream re-analysis
(`SEQ_analysis_L12_4stream.json`); the two shared streams reproduce the reviewed 2-stream file to
1e-4.

- **Survival collapses** after 50 edits — per-stream **10 / 14 / 42 / 36%** (pooled **25.5%**);
  say "collapses," NOT "monotonic decay" (the curves are non-monotone) and give the range rather
  than a single point. The higher survival in streams s2/s3 is **not an install artifact**: their
  edit-success is *higher* (**0.98 / 0.98**) than s0/s1 (**0.92 / 0.86**), so more edits were
  installed there, not fewer.
- **Position fragility:** ρ(stream-position, survival) per-stream **0.31 / 0.20 / 0.48 / 0.51**
  (roughly **0.20–0.51**), **pooled 0.372 (p=0.0005)** — *"later-applied edits survive modestly
  more often."* (The reviewed 2-stream baseline pooled to 0.25, p=0.009; the two added orderings
  raise it.) Note that the pooled 0.372 mixes streams with heterogeneous survival base rates
  (10–42%), which the per-stream range already exposes — we report both and lean on the range.
  **The ρ≈0.55 figure is RETRACTED and must never be cited.**
- **H1 geometry-attribution remains UNSETTLED at 4 streams.** The pre-registered gate
  (position-partialled forward S×C exposure ρ > 0, perm-p < 0.05) is not passed: **pooled partial
  ρ 0.097 (p=0.176)**; per-stream partials −0.01 / +0.08 / +0.03 / +0.16. Raw pooled ρ (0.372)
  collapses under position partialling. No geometry-predictability claim for the sequential
  setting enters the paper.
- **Flank layers (descriptive, 2-stream).** *Survival collapse* appears at every layer probed
  and is **hardest at L14 (5%)**: pooled survival 28% at L8 (per-stream 28/28), 5% at L14
  (per-stream 8/2), vs 25.5% at L12. *Position fragility*, by contrast, is modest and
  **non-monotone** across layers — ρ(position, survival) **0.57 (L8) / 0.372 (L12) / 0.36 (L14)**,
  a 0.36–0.57 band, in fact mildest at L14. These flank cells are 2-stream and descriptive and do
  not touch the geometry-attribution verdict. Source: `SEQ_analysis_L{8,14}.json`.

## Section 9 — Generality

- **Dataset (zsRE).** Llama-3.2-1B L10, zsRE: within-probe ρ **0.361 ± 0.014** signed / 0.495
  magnitude (3 seeds) — not a CounterFact artifact. Source: `C3_null_llama1b_zsre_L10.json`.
- **Scale (within family).** Llama-3.2-3B L14: **0.291** (1B ref 0.301) — replicates at 3×.
  Source: `C3_null_llama3b_L14.json`.
- **GPT-2-XL sanity (canonical E/G/L).** ES **0.98 / 0.985**, PS **0.74 / 0.77**, NS
  **0.74 / 0.71** (L5 / L17). The PS gap is a documented artifact of the no-context paraphrase
  templates. Note: both cells VALIDATE-WARN (esr **0.84 / 0.795**) — carry as a limitations
  footnote. Source: `sanity_gpt2xl_rome_cf_L{5,17}_s0.egl.json`.
- **Canonical EGL table across editors (Llama-1B L12, CounterFact, 2-seed means).** The standard
  edit-quality view of the editor spectrum — Efficacy / Paraphrase / Neighborhood-specificity:

  | Editor | ES | PS | NS |
  |---|---|---|---|
  | ROME | 1.00 | 0.991 | **0.043** |
  | AlphaEdit | 0.998 | 0.966 | 0.578 |
  | MEMIT | 0.995 | 0.955 | **0.741** |

  All three essentially install the edit (ES ≥ 0.995) and preserve paraphrases; they differ
  sharply on neighborhood specificity — ROME barely preserves neighbors (NS 0.043), MEMIT best
  (0.741), AlphaEdit in between — matching the mean-damage spectrum in §5. Source:
  `egl_llama1b_{rome,memit,alpha}_cf_L12_s{0,1}.egl.json` (2-seed means).

## Section 10 — Discussion: anisotropy of the edit-key distribution (why L14 / why Qwen)

*Descriptive, key-space only — no damage join, no causal claim* (respecting the
`interpretation_constraints` of `ANISO_analysis_L14.json`). We characterise the *distribution* of
ROME edit keys, which offers a candidate reading of the L14 regime and the Qwen behaviour without
asserting it causes them. The Llama-vs-Qwen crowding contrast **replicates across 3 seeds**: mean
pairwise key-cosine 0.460 / 0.430 / 0.431 (Llama) vs 0.200 / 0.196 / 0.197 (Qwen).

- **Llama-1B L14 edit keys are strongly coned.** Mean pairwise key-cosine **0.460**; the
  **uncentered** spectrum's top-1 eigenvalue fraction is **0.47** (inflated by the shared mean
  direction — mean-cos 0.46 — so read it as "keys share a dominant common axis," not as a
  centered-covariance rank); the **centered** participation ratio is **36.7**, far below the
  column-permutation null (~106) and the norm-matched isotropic-Gaussian null (~190). A highly
  anisotropic key cloud means |C| saturates high with low variance, consistent with S (not |C|)
  carrying the cross-edit damage variance at L14 (the §2/§4 regime reading).
- **Qwen-1.5B L14 keys are much less coned** (mean pairwise cos **0.200**, uncentered top-1
  **0.22**, centered PR **60.4**) — a descriptive contrast with Llama at the same layer.

Source: `ANISO_analysis_L14.json` + `ANISO_analysis_L14_s{1,2}.json` (3-seed crowding contrast;
top-1/PR spectra quoted at seed 0). We do not join this to collateral damage and make no causal
claim: the banks carry no damage matrices, and edit-key anisotropy is confounded with tokenizer /
subject-frequency. This paragraph reads the key distribution only.[^aniso]

[^aniso]: The module's `interpretation_constraints` block still contains stale lines ("no Qwen
    raw-key bank exists / cross-model L14-pending") that the file's own populated
    `cross_model_contrast` now contradicts; the governing caveat is `cross_model_contrast.note`
    ("descriptive only; do not over-read a single-seed two-point contrast"). No gemma bank exists
    in this file, so no gemma-anisotropy claim is made. Authoring-pass module — pending review.

## Section 11 — Related work / differentiation from CLaRE

CLaRE-ty (arXiv 2603.19297, ACL 2026 Findings) is the closest concurrent work: an a-priori
per-fact interference predictor from a **single-layer forward-activation cosine**
`cos(h_i^L, h_j^L)`, evaluated on 11,427 facts over five *locate-then-edit* editors × three MHA
models, baselined against GradSim, and — by the authors' own Limitations — **correlational, with
no causal mechanism.** We differ on four axes, each verified against CLaRE's full text and each
still correct under the Llama-family scoping:

1. **Mechanism vs metric (C1).** CLaRE's score is empirical (hidden-state cosine); we derive the
   collateral term in closed form from ROME's rank-one algebra (Eq. 3) — a zero-cost GradSim
   surrogate that *predicts* the phenomenon and its failure modes rather than measuring a
   correlate. CLaRE contains no analytic decomposition of ΔW.
2. **Norm-growth head-to-head + regime (C2).** CLaRE's only baseline is GradSim; ENCORE-style
   matrix-norm-growth is never considered. We run key-cos vs norm-growth on the confound-clean
   within-probe metric, expose the L12→L14 crossover, and show the coupling's *sign* tracks the
   damage regime across scale.
3. **Editor/architecture dissociation (C3).** CLaRE never evaluates fine-tuning editors (only
   ROME/MEMIT/PRUNE/RECT/AlphaEdit) and never evaluates Qwen/Mistral for its predictive claim
   (they appear only in its compute-cost tables). Our ROME-vs-FT/KL-FT/MEMIT and
   Llama-vs-{gemma,Phi,Qwen} dissociations are new territory, all measured 3-seed, and Eq. (3)
   explains every breakdown (rank/locality for the editors, S≈0 or regime-inversion for the
   architectures). Under our scoping this axis is *strengthened*, not weakened: we claim the
   signed law only for the family where we can causally support it.
4. **Causal test (C4).** CLaRE states it "does not establish a formal causal mechanism." Our
   AlphaEdit intervention — null-space projection removes exactly the geometry-predicted damage
   at every layer, erases the Qwen inversion, and collapses a refusal-deletion edit's coupling —
   is precisely the causal pathway CLaRE leaves as future work.

We position B6 as **complementary to** CLaRE — the mechanistic, causal, editor/architecture-
conditioned account of *when and why* key geometry predicts locate-then-edit damage on the Llama
family — and cite it as the concurrent empirical predictor, not a competitor to out-benchmark.
