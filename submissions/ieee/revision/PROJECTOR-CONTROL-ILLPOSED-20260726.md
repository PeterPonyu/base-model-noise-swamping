# The projector-substitution control is ill-posed (2026-07-26)

Decision record for the B6/TETCI objection *"AlphaEdit's damage-removed ∝ key-cosine is
algebraically guaranteed by the projector."* Two controls were built and both are dead;
this documents why, so nobody rebuilds them a third time.

## Attempt 1 — CPU first-order proxy: REJECTED (degenerate)
The ROME denominator rescale kᵀk→kᵀPk cancelled the projection identically (residual
1.8e-15 for any P), so both the "real" and "sham" columns were the projector-free quantity
ρ(COS, |COS|·‖A‖). Withdrawn in full: `SHAM-CONTROL-READOUT-20260726.md`.

## Attempt 2 — GPU rank-matched random projector: ILL-POSED (measured, not argued)
Measured on the real edit keys of `vectors_qv_llama1b_rome_cf_L12_s0.npz`
(n=200 keys, d=8192, key-matrix rank 200; `build_null_projector` at keep_ratio=0.99 → r=176):

| projector | mean kᵀPk/‖k‖² (key energy KEPT) | subspace overlap w/ honest |
|---|---|---|
| honest (top-176 of key span) | **0.0099** | — |
| rank-matched random in R^d (the patch as written) | **0.9777** | ≈0 |
| energy-matched (reviewer's proposed fix) | 0.0099 | **1.000** |
| random r inside top-181 key span | 0.0323 | high |
| band[20:196] of the key spectrum | 0.5901 | 0.886 |

Two facts kill the whole control family:

1. **Rank-matching is a no-op projection.** A random r-dim subspace of R^8192 with
   r/d ≈ 0.021 keeps 97.8% of every key's energy, so Pk ≈ k and the sham arm is
   essentially plain ROME. It cannot fail → it proves nothing. (Reviewer's HIGH-1,
   independently confirmed.)
2. **Energy-matching degenerates to the honest projector.** The key spectrum is extremely
   concentrated — the top 10 directions carry 50% of the energy, and the whole key matrix
   has rank 200. Removing ~99% of key energy therefore *forces* you to remove those same
   top directions: overlap with the honest subspace is 1.000. "Same energy, different
   directions" does not exist in this geometry.

**Conclusion: you cannot separate "removed the preserved-key subspace" from "removed 99%
of preserved-key energy" here — they are the same operation.** No projector substitution
can adjudicate the objection.

## What this means for the paper (framing decision — USER)

The referee's objection has **real force and should be partly conceded**, not fought with
another control. The honest position:

- **Concede:** given a projector that removes ~99% of preserved-key energy, the fact that
  the cancelled damage on probe *p* is ordered by cos(k_edit, k_p) is close to definitional.
  The paper should say so plainly rather than let a reviewer discover it.
- **Relocate the empirical content** to the parts the algebra does NOT give you:
  (a) the *magnitude* — ~98% of total damage removed is not implied by cancelling the
  along-key component, since downstream nonlinearity could have preserved it;
  (b) the L14 result — geometry predicts causally-removable damage at the layer where the
  *correlational* signal has already been overtaken by norm-growth;
  (c) the G1 gate itself, which involves **no projector at all** (ROME-only, within-probe,
  confound-controlled) and is where the geometry→damage law actually lives.
- Note the existing circularity triple (holdout 0.590 ≈ generic 0.574) cuts the same way:
  the result does not depend on fitting the projector on fact keys specifically. Report it
  as a scope statement, not as a robustness win.

## The one control that would still be informative (NOT built, cheap)

Test a *different* version of the objection — whether the law is about **knowledge keys**
or about **any rank-one direction**: apply a ROME-shaped rank-one update whose key is a
random unit direction with matched norm (not the fact's key), and measure whether damage
still tracks cos(update-direction, probe-key). No projector involved, so none of the above
degeneracy applies.
- If damage tracks cosine to an arbitrary direction → the "key geometry" law is a generic
  property of rank-one updates in this layer, a much weaker (but honest, and publishable)
  claim.
- If it does not → key structure is load-bearing, which is a *stronger* result than the
  projector test could ever have given.
Cost ~25 GPU-min/cell. This is a G1-side control, not a C4-side one, and it would need its
own preregistration before running.

## Actions taken
- `run_b6ins.sh` Cell S: **disarmed** (guard now refuses on principle, with a pointer here).
- `experiments/patches/alpha_sham_projector_20260726.patch`: **kept, NOT to be applied**;
  header rewritten to say so.
- `engine/chain_after_bc_drain_20260726.sh` stage S1c: **removed** (no longer applies the
  patch post-drain). The runner-stamp patch (S1) is unaffected.
- No GPU time spent on either dead control.
