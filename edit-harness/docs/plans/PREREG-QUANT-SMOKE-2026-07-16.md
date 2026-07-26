# PREREG — Quantization-survival SMOKE (Direction #1)   2026-07-16

Smoke test: does a knowledge edit AND its collateral damage survive quantization, and does the
key-geometry→damage tie survive too? SMOKE-scale, simulated quantization via a weight round-trip
(`dequant(quant(W))`); REAL GGUF/GPTQ/bitsandbytes kernels deferred to the full paper.

Code: `experiments/quant_survival_smoke.py` (+ driver `run_quant_smoke.sh`). Reuses the gate cells'
damage machinery (imported `_capture_key` / `efficacy` / probe-logit; signed damage_logit, never
AUROC). Does not modify any existing experiment; results quarantined to `results/quant_smoke/`.

**Simulation-fidelity disclosure (required before any paper claim):** the simulated NF4 keeps each
block's absmax in full fp32 — it OMITS bitsandbytes' DOUBLE QUANTIZATION (the block absmax values
are themselves FP8-quantized in outer blocks of 256). Real NF4 thus carries a small extra scale-
quantization error our round-trip does not, so this smoke slightly UNDERSTATES real NF4 error /
overstates survival. The full-paper numbers must come from the real bitsandbytes/GGUF kernels.

## Frozen design
- Model Llama-3.2-1B (local), layer **L12**, editor **ROME**, **n=50** CounterFact edits, **40
  probes**, seed 0 (add seed 1 if the budget allows). Codecs: **INT8** (per-row symmetric affine)
  + **NF4** (bitsandbytes 16-level codebook, blockwise absmax, blocksize 64), pure-torch round-trip.
- Arms per edit: **fp32** (standard gate measurement) · **edited_layer × {nf4,int8}** (round-trip
  only the edited down_proj@L) · **full_model × {nf4,int8}** (round-trip all transformer-block
  linears, edit-then-quantize order). **BASE arm** (unedited, both localities × schemes) gives the
  per-probe base quant noise so edit-survival is separable from base noise (edit-attributable
  damage = damage_arm − base_noise, reported alongside raw).
- Metrics: esr = efficacy success; damage_logit[i,j] = pre_l(fp32 unedited) − post_l; key-cos =
  cos(k_edit, k_probe); mechanism tie = Spearman(key-cos, damage) pooled + within-probe, at fp32
  vs each quant arm, plus Spearman(damage_fp32, damage_arm) rank-survival.

## Frozen predictions (directional only — ALL EXPLORATORY, not gates)
- **(p1)** esr survival > 0.9 under NF4 full-model at 1B (the edit itself is robust to deployment quant).
- **(p2)** quantization ADDS damage variance but the geometry ranking survives — Spearman(key-cos,
  damage) stays within ±0.15 of fp32 at every arm.
- **(p3)** edited-layer-only round-trip ≈ full-model for edit-local metrics (esr and the geometry tie).

## Launch (when authorised — NOT launched by the author; ≤40 GPU-min; respects the idle gate)
```
MODEL_DIR=data/models/Llama-3.2-1B MODEL_TAG=llama1b LAYER=12 SEED=0 ./run_quant_smoke.sh
```
Driver mirrors `run_merging_editors.sh` (GPU-idle gate util<25 & mem<1500 ×3, CPU self-test smoke
gate, budget, DRYRUN, refuse-guard on a valid table, PID-by-file / `kill -0`). CPU-validated build
only: `bash -n`; `--selftest` (INT8 half-step bound, NF4 grid-membership + max-gap/2 bound +
idempotence, tiny-random-model end-to-end pipeline); `--reanalyze` roundtrip; DRYRUN. Zero GPU used
to author or verify.
