# D2 revision-wave build notes (R-C / R-D / R-E / R-F)   2026-07-16

Author pass only — built to answer blind-referee gaps on the D2 federation paper
(`submissions/d2-federation/`). **Nothing was launched on GPU.** A separate hostile review runs
after this. Target runtime: AutoDL box, 2x4090D (24GB each), code at
`/root/edit-harness-code-20260716/`, models under `/root/autodl-tmp/models/`.

## R-F — two low-gain AlphaEdit federation cells (smallest)

**Files**: `run_revwave_rf.sh` (new). No Python changes — pure orchestration over the
already-reviewed `run_merging_editors.sh`.

Cells: gpt2-xl L36 alpha cf, Phi-3.5-mini L24 alpha cf, 3 seeds each (RG_SEEDS/RG_GROUP_SIZES/
N_EDITS left at `run_merging_editors.sh`'s own defaults — `0,1,2` / `2,3,5,10,20` / `200` —
which already match the spec exactly). `LAYER=auto75` lets each model's own `config.json`
determine the layer; verified live (DRYRUN) that gpt2-xl (48 layers) resolves to L36 and
Phi-3.5-mini (32 layers) resolves to L24, matching the deliverable's stated layers exactly, so
no hardcoded LAYER was needed.

**Validated**: `bash -n`; a full `DRYRUN=1` pass against locally-present model directories
(`data/models/gpt2-xl`, `data/models/Phi-3.5-mini`) — both cells produced correct, non-colliding
table/bundle paths and the exact `--layer` values above.

**Open risk**: the Phi-3.5 directory name on the AutoDL box is unconfirmed — the team-lead
message said "Phi-3.5-mini-instruct?" and the local dev copy here is named `Phi-3.5-mini` (no
`-instruct` suffix). Driver defaults to `/root/autodl-tmp/models/Phi-3.5-mini-instruct`;
override with `PHI35_DIR=` if that's wrong on the box. gpt2-xl needs arch_compat's Conv1D→Linear
graft (already exercised elsewhere in this harness, e.g. other gpt2-xl cells); Phi-3.5-mini is
native-Llama-shaped, no arch_compat work needed.

**Launch**: `GPT2XL_DIR=/root/autodl-tmp/models/gpt2-xl PHI35_DIR=/root/autodl-tmp/models/Phi-3.5-mini-instruct ./run_revwave_rf.sh`

## R-D — MEMIT true-covariance (wiki-or-CF-fallback)

**Files changed**: `experiments/merging_editors.py` (new: `_cov_cache_path`, `_save_cov_cache`,
`_load_cov_cache`, `_wiki_corpus_candidates`, `_load_wiki_or_fallback_prompts`,
`_get_or_build_memit_cov`, `_cov_variant_suffix`; `--memit_cov` choices extended to include
`"wiki"`; `_editor_context`'s memit branch now dispatches through `_get_or_build_memit_cov`);
`run_merging_editors.sh` (RG_DIR/TABLE naming gets a `${COV_SUFFIX}` when `EDITOR=memit` and
`MEMIT_COV != identity`). **New file**: `run_revwave_rd.sh` (thin wrapper, EDITOR=memit,
MEMIT_COV=wiki, LAYER=12, Llama-3.2-1B, cf, 3 seeds).

**Design**: `estimate_layer_covariances` (editors/memit.py) is reused UNCHANGED — only the
prompt SOURCE and an on-disk cache (`results/merging_editors/cov_cache/<model>_L<layer>_
<source>.npz`) are new. Ridge/invertibility (`eps*trace/d`) was already implemented correctly
inside `estimate_layer_covariances` (`A = C_hat + reg_used*mean_diag*I`, `mean_diag ==
trace(C_hat)/d`) — nothing new needed there. **No wikitext corpus exists under `data/` as of
2026-07-16** (verified `ls data/`) — `--memit_cov wiki` therefore falls through to the
documented CF-fallback (a broad, cell-independent CounterFact prompt sample, `seed=999`,
independent of `--seed`), logged as `source=cf_fallback`. The cache key includes the source
string, so a future real wikitext drop would build its own cache, never silently reuse the
fallback one. The ΔW-fidelity gate compares against the REAL `editors/memit.py apply_edit`
install — same anchor as the existing `generic` arm, no downgrade to decomposition-level was
needed (editors/memit.py already supports true, non-identity covariance).

**IMPORTANT BUG CAUGHT AND FIXED**: the RG bundle directory is derived solely from
`{model_basename}_{editor}_{dataset}_{layer}_RG` — it does **not** depend on `--memit_cov`. A
landed identity-cov bundle already exists at exactly this cell's target path
(`results/merging_editors/Llama-3.2-1B_memit_cf_L12_RG/`, the prereg's PRIMARY arm). Running
`--memit_cov wiki` there unmodified would have either silently clobbered that landed result (via
the driver's `--no_refuse_clobber`) or silently no-op'ed past it (module's own refuse-guard
seeing an existing valid table and treating the cell as "already done" — the table would report
identity-cov numbers under a wiki-cov filename). Fixed via `_cov_variant_suffix` in
`merging_editors.py` + the matching `COV_SUFFIX` line in `run_merging_editors.sh` — `identity`
paths are BYTE-IDENTICAL to before (verified via DRYRUN diff below); `wiki`/`generic` get their
own directory.

**Validated**:
- `py_compile` clean; `--selftest` (cross-term/additivity/ROME-equivalence/RG-pass-kill/
  ΔW-fidelity/large-vocab) still ALL PASSES after the change.
- DRYRUN identity-cov: resolves to `results/merging_editors/Llama-3.2-1B_memit_cf_L12_RG` and
  `RG_editors_table_llama1b_memit_cf_L12.json` — **identical to pre-change paths**.
- DRYRUN wiki-cov: resolves to `results/merging_editors/Llama-3.2-1B_wiki_memit_cf_L12_RG` and
  `RG_editors_table_llama1b_truecov_memit_cf_wiki_L12.json` — **distinct, non-colliding**.
- Real-model CPU smoke (Qwen2.5-0.5B, `data/models/Qwen2.5-0.5B`, `--memit_cov wiki`,
  `cov_max_tokens=60` to keep CPU runtime reasonable — the real spec's ~50k tokens is
  GPU-production-scale, not CPU-smoke-scale): `_editor_context("memit", ...)` built + cached a
  real 2-layer covariance (`layers=[11,12]`, chol shape `(4864,4864)`), logged
  `source=cf_fallback` as expected (no wikitext present); a SECOND call hit the cache (0.79s vs
  28.7s) and returned numerically identical `chol` arrays (`np.allclose` assertion passed). Test
  artifact cleaned up afterward (no stray files left in `results/`).

**Open risk**: `estimate_layer_covariances`'s per-prompt CPU forward cost means the real
`cov_max_tokens=20000` (module default) target at GPU speed is untested here — only a
`max_tokens=60` CPU smoke was run for time reasons. The GPU wave should watch the first seed's
timing (the covariance build is a one-time cost, cached and reused by seeds 1/2).

**Launch**: `MODEL_DIR=/root/autodl-tmp/models/Llama-3.2-1B ./run_revwave_rd.sh`

## R-C — 13B ROME federation cell, device_map-sharded

**Files changed**: `tp_edit_util.py` (new `resolve_input_device`, mirroring
`resolve_layer_device`); `experiments/merging_m0.py` (`_load_edit_model` gains
`model_dtype`/`device_map` params, mirroring `experiments/killgate_keygeom.py`'s existing
`--device_map` path exactly — both OFF by default, byte-identical old behavior when unset;
`_measure_merged_groups` gains an optional `input_device` param so the tokenizer-encode device
can differ from the Rt/Ktsc/W device; `run_phase_rg` now resolves BOTH `input_device` (tokenizer
encodes) and `layer_device` (GPU-tensor construction) via `tp_edit_util`, since under sharding
the edited layer can sit on a different card than the embedding; `main()` gains
`--model_dtype {fp32,bf16}` / `--device_map {none,auto,balanced,balanced_low_0,sequential}`,
hard-fenced to `--rg` mode only — `run_phase1`'s 3-regime kill-gate merge path was **not** made
device_map-aware, since only the RG path (`g=2..20` sweep) is needed here; a SystemExit guard
raises if `--device_map` is passed without `--rg`). **New file**: `run_revwave_rc.sh` (own
skeleton, modeled on `run_merging_width.sh`, MODEL_DIR/MODEL_TAG required with no default model
— per the "DO NOT download, user decides" instruction — MODEL_DTYPE=bf16/DEVICE_MAP=auto
defaults, with a loud WARN if a caller overrides to fp32+device_map=none on a ~13B model).

**Validated**:
- `py_compile` clean; `merging_m0.py --selftest` (pure numpy, no model) still ALL PASSES
  unchanged after the device_map plumbing — confirms the CPU-only analysis path is untouched.
- `bash -n` clean; DRYRUN with a locally-present 8B model (`data/models/Llama-3.1-8B`, standing
  in for the eventual 13B) resolved `--model_dtype bf16 --device_map auto` correctly into the
  `merging_m0.py --rg` command line and a correctly-named table/bundle path; a second DRYRUN
  with `MODEL_DTYPE=fp32 DEVICE_MAP=none` correctly printed the OOM-risk WARN.
- Missing MODEL_DIR/MODEL_TAG correctly refuses with a usage message (rc=1).

**OPEN RISK (the one thing NOT validated)**: the `--device_map`/bf16 CODE PATH ITSELF has **not**
run on real CUDA hardware — GPU launches were explicitly out of scope for this build. Nothing
in this harness currently has 2 visible GPUs to test true multi-card sharding, and testing on
this machine's single 5090 would only exercise "device_map=auto with one card visible"
(accelerate would just place everything on that one card, which does NOT exercise the
layer-device != input-device case the `_merge_factors`/`_measure_merged_groups` fix is for).
**Recommend the hostile reviewer specifically re-derive the device-handling logic by hand**
(trace `run_phase_rg`'s `input_device`/`layer_device` variables through `_compute_solo` →
`_merge_factors` → `_measure_merged_groups`) rather than trusting a smoke test, and treat the
FIRST real on-box invocation's log as the actual validation, watching for any `RuntimeError:
... expected all tensors to be on the same device`.

**Model choice**: MODEL_DIR/MODEL_TAG are required env vars with NO default — Llama-2-13b-hf
(40 layers → L30) and OLMo-2-1124-13B are both candidates per the task instructions; neither
was downloaded. `LAYER=auto75` handles either correctly once a model is chosen.

**Launch** (once a 13B model is on the box): `MODEL_DIR=/root/autodl-tmp/models/<the-13B-model> MODEL_TAG=<tag> ./run_revwave_rc.sh`

## R-E — prospective admission-policy evaluation (biggest; DRAFT-GATED)

**Files**: `experiments/prospective_admission.py` (new), `docs/plans/PREREG-PROSPECTIVE-
ADMISSION-DRAFT-2026-07-16.md` (new, marked DRAFT). **No driver ships** — deliberate: the
module's `main()` hard-refuses the GPU path with a SystemExit unless the prereg has been
ratified (there is no flag to bypass this; the guard must be edited out by hand after
ratification, which is intentional friction).

**Design** (frozen in the prereg doc — read it in full before running): Llama-3.2-1B L12 ROME
reference cell; 100-edit candidate pool/seed; pool-wide (not group-restricted) Eq-1 I_cos/I_mag
screening (generalizes `merging_m0._regime_stat`'s per-observation formula to "every other pool
candidate is a potential federation partner", since admission precedes grouping); 3 admission
policies at 25% budget (geometry/magnitude = bottom-25% by score, random = 3 independent draws);
admitted edits partitioned into g=5 groups of 5; each group's real ΔW installed/measured/
restored via `merging_m0._merge_factors` (no new merge math); behavioral outcomes on the
DEPLOYED merged model: (a) edit success rate, (b) merge-INDUCED neighborhood damage rate (via
`egl_metrics`'s canonical NS convention, pre- vs post-merge), (c) general retention on a fixed,
pool-disjoint 200-prompt held-out set (mean full-target log-prob shift), (d) target-logit drop
(continuity with the retrospective RG/M0 tables); 3 seeds.

**Validated (CPU only, no model, no GPU)**:
- `py_compile` clean.
- `--selftest`: (a) the pool-wide I_cos/I_mag closed form vs an explicit O(N²) brute-force
  recomputation (worst abs error 1.6e-11); (b) admission budget rounding/determinism +
  `partition_groups` shape/coverage/disjointness; (c) a synthetic end-to-end aggregation pass
  (fabricated per-edit rows → the same success/drop reduction the GPU path uses → JSON-
  serializability). All PASS.
- `load_retention_prompts` (the pool-disjoint 200-prompt held-out set) verified directly: loads
  200 prompts, **zero overlap** with any of the 3 seeds' 100-candidate pools (checked by
  explicit set intersection against `load_counterfact` at each seed), and is **byte-
  deterministic** across repeated calls.
- The GPU-only refuse-guard fires correctly (`rc=1`, clear message) when invoked without
  `--selftest`.

**BUG CAUGHT AND FIXED DURING BUILD**: the group-partition RNG seed for each (seed, policy,
draw) originally used `hash(pname)` — Python randomizes string hashes per-process by default
(`PYTHONHASHSEED`), which would have made the group partition **non-reproducible across
re-runs at the same `--seed`**, silently breaking this harness's standing seed-determinism
convention. Replaced with a fixed `POLICY_SALT` dict.

**NOT validated (by design, since GPU is out of scope)**: the actual GPU measurement loop
(`_measure_one_group`, `_neighborhood_damage_rate`, `_retention_mean_logprob`, and their
integration with `egl_metrics.full_target_scores`/`mean_logprob_full_target` and
`merging_m0._compute_solo`/`_load_edit_model`) has never executed against a real model. All
imported symbols were confirmed to exist via a static attribute check, and the call shapes were
built by close reading of `merging_m0.py`/`egl_metrics.py`'s existing usage, but this is the
single largest unverified surface in this wave — flagging for the hostile reviewer to read
closely before any GPU time is spent, and for the USER to ratify the prereg doc before either
review or launch.

**Launch**: NONE until the user ratifies `docs/plans/PREREG-PROSPECTIVE-ADMISSION-DRAFT-2026-
07-16.md`; the doc's own "Launch" section gives the eventual command + notes that a driver
should be built alongside ratification, not before it.

## Cross-cutting notes

- No file a live queue imports was touched (checked: no local queue currently running from
  `edit-harness/`; `experiments/frame_a/*` was not touched at all, per the boundary
  instruction).
- All four deliverables' identity/default-path behavior was verified byte-identical to
  pre-change (DRYRUN diffs for R-D/R-F/R-C; unchanged `--selftest` output for merging_m0.py/
  merging_editors.py).
- Nothing was launched on GPU by this build pass; the one real-model exercise (R-D's covariance
  cache smoke) ran entirely on CPU (`device="cpu"`) with a tiny local model and cleaned up its
  own test artifact afterward.
