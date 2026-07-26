"""experiments/frame_a/ — Paper A (Cost-Aware Knowledge-Maintenance Router) implementation.

Build-only pipeline over the FROZEN rev.4 design (docs/plans/DESIGN-FRAME-A-2026-07-16.md) +
prereg (docs/plans/PREREG-FRAME-A-STREAM-2026-07-16.md). Modules: config, stream_builder, arms/,
router, cost_harness, damage_predictor, run_stream, scorer/, selftest. Nothing here launches a
GPU run; the synthetic-model path exercises the whole pipeline on CPU and the mandatory
`selftest` gate guards the wave.
"""
