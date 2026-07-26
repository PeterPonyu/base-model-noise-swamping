# PLAN-U1-PAPER — U1 Deletion Program (2026-07-03)

Both U1 gates passed today at L12/s0 (C3 rho=.638; U1-E1 transplant Delta-rho=.61), but
the formal prereg scorer (u1_deletion_gate.py) has only run on a tiny CPU dev-smoke —
never on real data. `run_u1.sh` is the one-command launcher for the next campaign.

## Blocks
- **A — seed hardening** (refusal L12 s1,s2). H: L12 rho survives seeds. Kill: any seed
  rho<0.15 or perm_p>=0.05 (u1_deletion_gate.py PREREG). Cost: 60 GPU-min.
- **B — layer profile** (refusal L8,L14 s0). H: geometry->deletion-damage link holds
  off-peak, mirroring the insertion regime curve (L8 0.395/L14 0.301 rho analog). Kill:
  same prereg bar; a peak-only result narrows the paper's claim, not fatal. 60 GPU-min.
- **C — variant dissociation** (eos, suppress; L12 s0; editors/rome_deletion.py, never
  run at GPU scale). H: the geometry link is objective-general, not an artifact of the
  refusal-string data-layer swap. Kill: prereg fails for BOTH eos and suppress ->
  refusal-specific artifact, demote to a footnote. 70 GPU-min.
- **D — mitigation** (AlphaEdit delete-refusal L12 s0). H: null-space projection removes
  deletion collateral the way it removes rewrite collateral (C4 causal precedent). Kill:
  no damage reduction vs raw ROME-delete -> AlphaEdit's protection is rewrite-specific.
  35 GPU-min.
- **E — STRETCH cross-arch** (qwen15b, refusal, L14 s0 -- matches qwen15b's existing
  rewrite-law layer so it drops into the same C3-null scaffold). H: the deletion-geometry
  law generalizes off-Llama the way the magnitude REWRITE law does (4/5 families), not the
  way the SIGNED rewrite law fails to (crossarch-transfer-verdict-2026-07-02). Kill: not
  gating -- a negative/inverted result here is itself a reportable dissociation, same
  epistemic status as the Qwen sign-inversion finding in the rewrite arm. Same code path as
  A/B (rome+delete+refusal, already proven to completion today), so no fresh smoke gate.
  35 GPU-min.
- **FILLER — QuantEdit-delete arm** (rome+delete+refusal+save_vectors; guard-legal, never
  exercised combo). Populates the rank-one vector oracle for a future QuantEdit-E5
  deletion unlock; not itself gated. 35 GPU-min.

## Guard legality (verified against killgate_keygeom.py:196-238)
All 10 row-types legal — see authoring report's guard-legality table. No row dropped.

## Cost / venue
Smokes 4x4=16m + Science 260m + Stretch-E 35m = **~311 GPU-min (~5.2 GPU-h)**, budget 480m
(8h), headroom for retries/model-load. Venue: ARR/EMNLP main next cycle (CCF-B, archival)
first per the 07-02 pivot; KBS (SCIE Q1) as the journal extension once C/D land — same
sequencing as B6.

## Post-run (CPU, always)
- `u1_deletion_gate.py` (the FORMAL prereg SxC-DC scorer) runs on every deletion cell against
  its matched insertion npz, including the new qwen15b_L14_s0 pair
  (`gate_qwen15b_rome_cf_L14_s0.npz`, on disk from 2026-07-02).
- `u1_transplant.py` (U1-E1 falsifiable gate) runs on every NEW refusal-VARIANT npz this
  driver can produce: L12 s1/s2, L8 s0, L14 s0, the AlphaEdit-delete mitigation cell, and
  the qwen15b cross-arch cell. It is defined only for the refusal transplant question
  (loader reimplements the pre-swap target_new), so eos/suppress/qv cells are out of scope
  for it by design.
- `analyze_matrices.py` C3-style within-probe groups, one per block/config (never pooled
  across layers, editors, or delete variants).

## Launch
```
cd edit-harness && BUDGET_MIN=480 ./run_u1.sh >> engine/run_u1.nohup.log 2>&1 &
```
Or arm `engine/chain_u1.sh` to fire it automatically when a trigger marker appears (see
that script's header for the exact marker path/name).
