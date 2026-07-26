# Sham-projector control — WITHDRAWN (2026-07-26)

> **STATUS: REJECTED BY HOSTILE REVIEW. DO NOT CITE ANY NUMBER BELOW IN THE TETCI
> REVISION LETTER.** The CPU first-order proxy is mathematically degenerate. The
> interpretation and rebuttal paragraph that previously appeared in this file are
> WITHDRAWN in full. The objection this control was built to answer remains OPEN and
> now requires the GPU-level sham (see "What must replace it").

## Why it was rejected (independent review, verified against the code)

**Blocker 1 — the projection cancels identically.** In `experiments/sham_projector_control.py`
(the `predicted_removed` closure), the parallel term is

    proj_par = |diag·COS| · (1/|diag|) · ‖A‖  =  |COS|·‖A‖  =  raw_hit

exactly: the ROME denominator rescale kᵀk→kᵀPk undoes the along-key projection it was
meant to model. So `predicted_removed = raw_hit − sqrt(raw_hit² + orth²)` for EVERY
projector, real or sham, and both columns collapse to a monotone function of `raw_hit`.
The reported "sham ≈ 0.85–0.92" is just ρ(COS, |COS|·‖A‖) — a projector-free quantity
(0.970/0.975/0.966 at L8, 0.912/0.937/0.910 at L12 with no projector at all).
**The control therefore contains no projector information and cannot separate real from sham.**

**Blocker 2 — sign inversion.** `predicted_removed` is negative for 100% of entries: the
proxy predicts the projector ADDS damage everywhere, while the measurement shows ~98%
removal. A proxy whose central prediction has the wrong sign cannot license any claim
about what "the algebra predicts".

**Blocker 3 — the smoke gate was vacuous.** It passed on a planted `rho_measured > 0.3`
alone and reported real < sham (the opposite ordering from the real cells); it would have
passed with the proxy deleted.

**Also found (would matter even after a rebuild):**
- Signed/abs mismatch: the proxy uses |·|, the measured damage-removed is signed with ~6%
  negative mass at L12 s0. Same estimand only because COS happens to be 100% positive here.
- Dropped 1/‖k_i‖: ROME carries k/‖k‖², the proxy used |COS|·‖A‖ only; ‖k_i‖ spans 2.05×
  within probe columns.
- The "real projector" rebuild is fit on 199 edit keys → rank r=175, whereas the editor's
  holdout projector is fit on ~500 disjoint probe keys. That is a **different-rank**
  projector, not merely a different sample — the earlier disclosure understated it.

**Confirmed sound (kept for the record):** B_i ∝ k̂_i exactly for ROME (cos = 1.000000 over
199 edits); AlphaEdit projects the KEY side, ΔW=(v−Wk)(Pk)ᵀ/(kᵀPk), which the code matched;
the six reported numbers do faithfully reproduce from the script (the script is what is wrong);
independent spot-recompute of L12 s0 measured within-probe ρ = 0.5723, matching.

## The honest, much narrower statement that survives

Any damage surrogate monotone in |COS| correlates ~0.95 with COS. The measured
damage-removed correlations (0.35–0.61) are lower, which shows the full forward pass
**attenuates** geometry-tracking relative to a first-order surrogate. **That says nothing
about the projector and does not rebut the "algebraically guaranteed" objection.**

## What must replace it (required, not optional)

GPU-level sham: rerun AlphaEdit through the model with `config["projector"]` replaced by a
rank-matched random projector, at L12 s0 first (~25 GPU-min per draw), and compare
within-probe ρ(key-cos, damage-removed) against the real holdout-projector cell. This is
the only version that puts a projector in the causal path. Spec lives in the comments of
`edit-harness/run_b6ins.sh`; it is currently NOT queued — queue it after the alphaHO
L10/L14 rows drain, and do not iterate further on CPU algebra (fixing Blocker 1 requires
modeling P's effect on ΔW·k_p through the full product without the self-cancelling
rescale, and would still miss the nonlinearity that produces the measured gap).

Artifacts retained for provenance only, all superseded:
`edit-harness/results/sham_control/*.json`, `edit-harness/experiments/sham_projector_control.py`
(defect at the `predicted_removed` closure).
