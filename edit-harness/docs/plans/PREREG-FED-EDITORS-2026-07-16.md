# PREREG — Editor-general federation RG (MEMIT + AlphaEdit)   2026-07-16

Short prereg addendum extending the ROME-only edit-federation law (the merging RG
operating curve, `merging_m0.py`, PASS at Llama-3.2-1B L12: geometry-valid g≤5,
damage-gradated g=10; two-regime gain-screened law) to the other two native editors, to
test whether the two-regime federation law is **editor-general**.

Code: `experiments/merging_editors.py` (+ driver `run_merging_editors.sh`). ROME through
this module reproduces `merging_m0.py` exactly (equivalence anchor, `--selftest` (c)).
This addendum does NOT modify `merging_m0.py` or re-open the ROME verdict.

## Frozen design

- **Editors:** `memit`, `alpha` (ROME is the equivalence baseline, already landed).
- **Models (local ¥0 wave):** Llama-3.2-1B @ L12 and Qwen2.5-1.5B @ L21 (=floor(28·0.75),
  the 75%-depth rule). Two families × two editors = 4 science cells.
- **Seeds:** 0, 1, 2. **Group sizes g:** 2, 3, 5, 10, 20 (the geometry-valid + transition
  band from the ROME curve; g=50/100 dropped — MEMIT's multi-layer cost per edit is ~L×
  ROME and the ROME curve already saturates by g≈20).
- **Metric (unchanged from the RG ruling 2026-07-12):** partial Spearman
  ρ(I_cos, drop | I_mag), own-magnitude guard ρ(I_cos, drop | ‖k_a‖, S_a), pre-registered
  PASS rule = geometry passes at ≥2 group sizes across ≥2 seeds. Editor-general
  quantities generalise to multi-layer + effective-key form (module docstring):
  cross-term ΔW_b^l@k_a^l = r_b^l(kk_b^l·k_a^l)/denom_b^l, kk = k/Pk/C⁻¹k for
  ROME/Alpha/MEMIT; I_cos/I_mag/dose aggregated over edited layers.
- **AlphaEdit config:** keep_ratio 0.99, preserved-key bank = 50 held-out CounterFact
  edits (disjoint from the edit set, same shuffle → byte-stable).
- **MEMIT config:** `--memit_cov identity` for the primary wave (ROME-style, no external
  corpus; earns the name "MEMIT-style multi-layer spread", not "MEMIT" — the memit.py
  covariance caveat). Span 4 (auto → layers z−3..z). A `generic` covariance arm
  (holdout-bank C_l) is available but is a SECONDARY, opt-in comparison, not the frozen
  primary.
- **Dataset:** `cf` primary; `zsre` is an opt-in generality arm (module `--dataset zsre`).

## Frozen predictions (directions, not thresholds)

- **(a) ROME-equivalence anchor MUST hold.** merging_editors ROME == merging_m0 to fp64
  on the shared analysis fields (asserted CPU-side in `--selftest`; any real-model ROME
  re-run must land on the existing L12 numbers). This is a correctness gate, not a science
  bet — if it fails, nothing downstream is trusted.
- **(b) AlphaEdit federation shows LOWER interference than ROME at matched cells.** Its
  null-space projection removes the edit's component along the preserved-key subspace by
  construction, so cross-talk onto in-subspace keys is attenuated → smaller median |drop|
  and a lower / higher-boundary saturation curve than ROME at the same (model, layer, g).
  Direction only; no magnitude committed.
- **(c) The two-regime ordering extends to editor-varied cells.** Where interference is
  non-negligible and non-saturated, geometry (the partial ρ) still predicts the drop for
  MEMIT and AlphaEdit — i.e. the gain-screened ordering is not ROME-specific. A per-editor
  PASS is the confirming outcome; a KILL is a real finding (the law is ROME-specific) and
  is reported as such (kill-gate discipline; truth-first reset).
- **(d) MEMIT regime placement is EXPLORATORY.** MEMIT spreads one fact over several
  layers, so its per-edit cross-talk aggregates differently; we do NOT pre-commit to where
  MEMIT lands on the geometry-valid / damage-gradated / saturated axis. The RG curve +
  scoped-federation boundary are reported descriptively for MEMIT; only the ROME-anchor
  (a) and the direction (b) are hard predictions.

## Launch (when a GPU wave is authorised — NOT launched by the author)

```
MODEL_DIR=data/models/Llama-3.2-1B  MODEL_TAG=llama1b  EDITOR=memit LAYER=12 ./run_merging_editors.sh
MODEL_DIR=data/models/Llama-3.2-1B  MODEL_TAG=llama1b  EDITOR=alpha LAYER=12 ./run_merging_editors.sh
MODEL_DIR=data/models/Qwen2.5-1.5B  MODEL_TAG=qwen15b  EDITOR=memit          ./run_merging_editors.sh
MODEL_DIR=data/models/Qwen2.5-1.5B  MODEL_TAG=qwen15b  EDITOR=alpha          ./run_merging_editors.sh
```

Driver mirrors `run_merging_width.sh` (GPU-idle gate util<25 & mem<1500 ×3, CPU self-test
smoke gate, budget, DRYRUN, refuse-clobber, PID-by-file / `kill -0` only). Two extra gates
vs the width driver: (1) a first-run **ΔW-fidelity gate** (`--smoke`) that installs a couple
of edits with the REAL editor and asserts the re-derived ΔW matches to Frobenius rel-err
< 1e-4 per layer before any RG cell is trusted (cached by a `.ok` marker); (2) outputs are
**dataset-tagged** (`<model>_<editor>_<dataset>_L<layer>_RG`) so cf and zsre cells coexist.
The driver passes `--no_refuse_clobber` (it owns the valid-bundle refuse in preflight; the
module's guard only refuses a COMPLETE bundle, so a timed-out/half-written cell is re-runnable,
never wedged).

CPU-validated build only: `bash -n`; `merging_editors.py --selftest` (cross-term, additivity,
ROME-equivalence anchor exact to 0.000e+00, RG pass/kill, ΔW-fidelity on a tiny random Llama for
all 3 editors); `--smoke` on the real Qwen2.5-0.5B (CPU) for all 3 editors (rel-err ≤ 5e-6);
DRYRUN for both wave models. Zero GPU used to author or verify.
