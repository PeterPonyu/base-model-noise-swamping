# cloud/EXTENSION-WAVE-RUNBOOK.md — 2026-07-11 extension wave

Built + CPU-tested 2026-07-11 (`bash -n` on every new script, DRYRUN launcher
simulations, live `config.json` fetches to confirm the 4 new model repos/architectures —
see "What's verified" below). **No AutoDL box was provisioned, no GPU, no model download
was performed to build this.** Builds ON TOP OF the existing `cloud/` orchestration
(`run_cloud_wave.sh`, `setup_autodl.sh`, `sync_results.sh`, `gpu_idle_lib.sh`) — read
`cloud/README.md` first, this doc only covers what's NEW.

## What this wave answers

- **Track 1 — does the signed key-geometry->damage law + "S x C beats raw key-cos at
  all layers" transfer to the modern 7-9B instruction-tuned tier?** The local zoo tops
  out at Llama-3.1-8B base + GPT-J-6B (2026-07-11 scout finding) — the entire 7-9B
  instruction-tuned family is cloud-only. `run_family_transfer.sh` runs correlational
  ROME cells + a holdout-projector AlphaEdit causal cell + `mechanism_sc_table.py` for
  Mistral-7B-v0.3, Qwen2.5-7B, gemma-2-9b, and Llama-3.1-8B-**Instruct** (the 8B
  instruct twin — local only has the 1B twin, `run_instruct.sh`).
- **Track 2 — 3-seed-harden the cross-arch causal cells that are currently single-seed
  s0 and weak/negative on disk.** `run_extension_causal_seeds.sh` adds seeds s1/s2 to
  neox20b, pythia14b, pythia28b's AlphaEdit-holdout causal cells (GPT-J is local, has its
  own driver, NOT touched here).

## New files

| File | Purpose |
|---|---|
| `cloud/dl_extension_models.py` | Downloads the 4 Track-1 models (mirrors `cloud/dl_pythia.py`) |
| `run_family_transfer.sh` | Track 1 driver (repo root, like every other `run_*.sh`) |
| `run_extension_causal_seeds.sh` | Track 2 driver (repo root) |
| `cloud/run_extension_wave.sh` | Wave launcher, mirrors `cloud/run_cloud_wave.sh` |
| `cloud/failsafe_extension.sh` | 30h hard power-off, mirrors `cloud/failsafe_enhance.sh` |

None of the 3 existing chain-locked drivers (`run_ripple.sh`, `run_mquake_law.sh`,
`run_8bcausal.sh`) or `run_neox20b.sh`/`run_pythia.sh` were touched — every new driver is
a standalone file, per the live-file-edit-hazard lesson (memory
`live-file-edit-hazard-under-running-queue.md`).

## Download list — ASK-FIRST (per standing policy, unchanged)

| Model | Repo (primary, ungated mirror) | Gated official alternative | Size (bf16) |
|---|---|---|---|
| Mistral-7B-v0.3 | `mistralai/Mistral-7B-v0.3` (itself ungated) | — | ~14.5GB |
| Qwen2.5-7B | `Qwen/Qwen2.5-7B` (ungated) | — | ~15.2GB |
| gemma-2-9b | `unsloth/gemma-2-9b` | `google/gemma-2-9b` (needs license accept) | ~18.5GB |
| Llama-3.1-8B-Instruct | `unsloth/Meta-Llama-3.1-8B-Instruct` | `meta-llama/Llama-3.1-8B-Instruct` (**separate** gate from the already-accepted Llama-3.2-1B-Instruct — do not assume it carries over) | ~16GB |

Total ~64GB new download (Track 1) + Track 2 needs `pythia-1.4b`/`pythia-2.8b`
(`cloud/dl_pythia.py`, ~8.5GB) and `gpt-neox-20b` (`setup_autodl.sh download --with-20b`,
~40GB) — **none of these 3 exist in the local zoo either** (verified 2026-07-11: no
`data/models/{pythia-1.4b,pythia-2.8b,gpt-neox-20b}` locally), so on a fresh box Track 2
also needs its models re-downloaded even though the s0 science already exists (only JSON
results were synced back from the 07-08 wave — see memory
`cloud-wave-complete-20260710.md`'s "JSON-only sync" note). Grand total download ≈ 112GB;
check disk headroom before `all --with-20b`.

The unsloth mirrors are the SAME choice this repo's own `setup_autodl.sh` already makes
for `unsloth/Llama-3.2-3B`/`unsloth/gemma-2-2b` — verified live (2026-07-11) that all 4
repos above return a clean `config.json` fetch with no auth wall; architecture facts below
come from those live fetches, not guesses.

## Also required — NOT in the download list, provision via RSYNC (not git-clone)

`run_family_transfer.sh`'s preflight hard-aborts (`pf_fail=1`, `exit 3`) without two
inputs that already exist LOCALLY but are not fetched by any downloader script above:

| Input | Why | Local path |
|---|---|---|
| `data/models/Llama-3.2-1B` | bf16 equiv-gate reference model (Phase A's ROME row, `--model_dtype bf16`) — gated repo `meta-llama/Llama-3.2-1B` | already local |
| `results/matrices/gate_llama1b_rome_cf_L12_s0.npz` | fp32 equiv comparator the bf16 gate diffs against (hard preflight check, `run_family_transfer.sh`'s "equiv comparator fp32 npz" line) | already local |

Both exist on the local laptop, so `rsync`-provisioning the box carries them over
transparently — **this is why the box MUST be provisioned by rsync/scp of the local
`edit-harness/` tree, NOT a fresh `git clone`** (this repo's `results/` and `data/models/`
are gitignored; a git-clone box would hard-abort `run_family_transfer.sh`'s preflight on
BOTH Track-1 model rows AND the Phase-0 equiv-gate step, i.e. Track 1 produces zero
science). A downloader-only box (models fetched via `dl_extension_models.py` but the repo
itself git-cloned) hits the identical failure — the fix is provisioning method, not an
extra download. If step 1 below clones instead of rsyncs, copy these two paths over
separately before step 5.

## Verified architecture facts (live `config.json` fetch, 2026-07-11)

| Tag | Layers | Hidden | `--expect_params` | Layer band (0.50 / 0.75 depth) |
|---|---:|---:|---:|---|
| `mistral7b` | 32 | 4096 | 7.248e9 (computed from config, matches published 7.25B) | L16 / L24 |
| `qwen7b` | 28 | 3584 | 7.61e9 (published) | L14 / L21 |
| `gemma9b` | 42 | 3584 | 9.2422e9 (published, matches gemma-2-9b-it's 9,242,164,736) | L21 / L32 (round(31.5)->32) |
| `llama8binst` | 32 | 4096 | 8.03e9 (same as the already-local Llama-3.1-8B base — identical arch) | L16 / L24 (paired with the base model's own L16/L24 cells) |

All 4 are **native** architectures per `editors/arch_compat.py`'s structural check
(`hasattr(model.model, "layers")`) — confirmed by the same code path already working for
the local `gemma-2-2b` (gemma2 family). No graft, no tensor-parallel needed for Track 1 —
every row is a single-card bf16 edit, same shape as `run_8bcausal.sh`'s Llama-3.1-8B rows.
`--expect_params` values are the same "guessed, not measured" honesty convention as every
other driver's header — `integrity_check.py`'s 1% band is the real arbiter on the box.

## Track 2 layer/lr reconciliation — READ THIS BEFORE LAUNCHING

`run_neox20b.sh` on disk still shows its ORIGINAL plan (L33 peak, lr 0.1). That cell is
dead: `results/gate_neox20b_rome_cf_L33_s0.json.DEAD-LR01` (esr collapses to ~0.01 at 20B
scale, memory `neox20b-esr-depth-collapse-20260709.md`). The battery that actually ran
(07-08/09, likely via an archived wave-2 orchestrator not present in this checkout — only
JSON results were synced back) redesigned to a shallow band at lr 0.5; its causal cell is
`g4_neox20b_alphaHO_cf_L16_s0.json` (esr 0.935, confirmed by reading the file directly —
`provenance.lr=0.5`, `model_dtype_arg=bf16` — AND matching the team-lead brief's own
"neox20b L16 rho 0.049" line). **`run_extension_causal_seeds.sh` targets L16/lr0.5 to
match the REAL cell on disk, not `run_neox20b.sh`'s stale header.** pythia14b (L6/lr0.1)
and pythia28b (L8/lr0.1) have no such discrepancy — `run_pythia.sh` is unchanged and its
adaptive selector's choice is directly confirmed from the s0 result files.

## Cost estimate — READ BEFORE PROVISIONING

Per-row minute estimates (guessed from `run_8bcausal.sh`'s MEASURED ~100min/row for
Llama-3.1-8B ROME COMMON, scaled by hidden-size/layer-count ratios — same honesty
convention as every "GPU COST — FLAGGED" header in this repo; **none of these 4 new
architectures have ever run in this harness**):

| Tag | ROME row (~min) | AlphaEdit-holdout causal row (~min) |
|---|---:|---:|
| mistral7b | 100 | 120 |
| qwen7b | 95 | 115 |
| gemma9b | 130 | 155 |
| llama8binst | 100 | 120 |

**"Floor" cost** (peak+mid layer ROME s0 + peak-layer causal s0, per model — what
`run_family_transfer.sh`'s breadth-first Phase 1-3 guarantees first):
mistral7b 320m, qwen7b 305m, gemma9b 415m, llama8binst 320m.

**Card assignment** (`cloud/run_extension_wave.sh`'s default): card0 = {mistral7b,
gemma9b} floor 735m; card1 = {qwen7b, llama8binst} floor 625m + Track-2 pythia gap-fill
(4 rows: 50+50+70+70=240m) = 865m. ~18% imbalance, accepted (estimates carry real
uncertainty; retune via `FAMILY_MODELS_CARD0`/`FAMILY_MODELS_CARD1` if a real run shows a
better split).

**Two cost tiers.** `NEOX_SEEDS` is read ONLY by `run_extension_causal_seeds.sh`'s
neox20b rows, which run EXCLUSIVELY in the `tp2` phase (step 7 below) — setting it on
the `both`/step-5 command has **no effect** (that phase never touches neox20b; see the
launcher header). The tier decision happens at step 7 (or is baked into step 8's
one-liner), not step 5.

| Design | card0/card1 phase (parallel) | tp2 phase (neox20b, both cards) | Total wall-clock | Est. cost @ ~$0.33/card-h* |
|---|---:|---:|---:|---:|
| **Trimmed (DEFAULT)** — `run_extension_causal_seeds.sh` now defaults `NEOX_SEEDS="1"`, no flag needed | ~865min (14.4h) | 1 row x 300m = 300m (5h) | ~19.4h | **~$13** |
| **Full** — explicit `NEOX_SEEDS="1 2"` opt-in, both gap-fill seeds | ~865min (14.4h, card1-bound) | 2 rows x 300m = 600m (10h) | ~24.4h | **~$16** |

\* rate inferred from the 07-08->10 wave's own "~20.5h dual-4090 -> $12-15" precedent
(memory `cloud-wave-autodl-ready-20260708.md` + `cloud-wave-complete-20260710.md`), NOT an
independently looked-up AutoDL price (this build has no network pricing lookup) — verify
the box's actual hourly rate before committing either tier. **Trimmed is the DEFAULT for
exactly this reason — it's the tier that matches the standing ~¥100-class ($12-15)
pre-approved ceiling. Full runs ~$1-4 over it and requires an explicit `NEOX_SEEDS="1 2"`
— get a go-ahead before choosing it.**

Set `BUDGET_MIN_FAMILY` (default 650/card) higher for a more complete Track-1 seed sweep
if the cost ceiling is relaxed; at the 650 default, card0's heavier model (gemma9b) may
lose its mid-layer row to the budget gate (logged as `BUDGET-SKIP`, not silently dropped —
check `cloud/logs_ext/card0.log`).

## Ordered runbook

1. Open the AutoDL box in "无卡模式" (no-GPU tier). `ssh` in, **rsync/scp this repo over
   from local (NOT a fresh `git clone`)** — see "Also required" above: `results/` and
   `data/models/` are gitignored, and `run_family_transfer.sh` hard-aborts its preflight
   without `data/models/Llama-3.2-1B` + `results/matrices/gate_llama1b_rome_cf_L12_s0.npz`,
   both of which only exist locally.
2. **Ask-first gate**: confirm the download list above with the user before proceeding
   (unchanged policy). Once approved:
   - `bash cloud/setup_autodl.sh all` (the ORIGINAL core battery, if this is a fresh box —
     skip if already done) then, separately:
   - `( source /etc/network_turbo; python cloud/dl_extension_models.py ) &` — Track 1's 4
     models (~64GB, ~2-3h depending on mirror bandwidth)
   - `( source /etc/network_turbo; python cloud/dl_pythia.py ) &` — Track 2's pythia
     pair (~8.5GB) if not already present
   - `bash cloud/setup_autodl.sh download --with-20b` — Track 2's neox20b (~40GB) if not
     already present
   - All three can run concurrently (network+disk only, no GPU needed yet) — same pattern
     as `run_enhance_4090.sh`'s background pythia download.
3. `bash cloud/setup_autodl.sh patch-drivers` — **only if this box will ALSO run the
   original `cloud/run_cloud_wave.sh`** (patches the 3 chain-locked drivers; the extension
   wave's own new drivers already natively source `cloud/gpu_idle_lib.sh` and honor
   `CLOUD_PY`, no patching needed for them — see "Portable H/PY" note below).
4. Switch to the 2x4090 GPU tier. `bash cloud/setup_autodl.sh gpu-check`.
5. Launch the per-card phase — `NEOX_SEEDS` has NO effect here (see cost-tier note
   above), the cost-tier choice happens at step 7/8:
   `bash cloud/run_extension_wave.sh both`
   > **CAVEAT (review NIT, 2026-07-11): NEVER manually launch `card0` and `card1` as
   > separate commands on a fresh box** — the two subcommands race the Phase-0 equiv
   > gate (each may try to derive `engine/r3_equiv_bf16.ok` concurrently). Only
   > `both` (which hoists Phase-0 to a single card before fanning out) and `full`
   > are safe entry points.
   (this also runs Phase-0: a single-card, before-fan-out derivation of the shared
   `engine/r3_equiv_bf16.ok` marker if it's absent/stale — see `phase0_equiv_gate` in
   the launcher; no action needed, it's automatic and a no-op if the marker is fresh).
6. `bash cloud/run_extension_wave.sh wait` — blocks until both per-card workers exit.
7. Pick a cost tier HERE (this is where `NEOX_SEEDS` actually matters):
   `bash cloud/run_extension_wave.sh tp2` — **Trimmed (default)**, one gap-fill seed, no
   flag needed; or
   `NEOX_SEEDS="1 2" bash cloud/run_extension_wave.sh tp2` — **Full**, both gap-fill
   seeds (explicit opt-in, get a go-ahead first).
   Needs BOTH cards free, hence step 6 first. `bash cloud/run_extension_wave.sh
   wait_tp2` to block until it drains.
   - Or skip steps 5-7's manual sequencing entirely: `bash cloud/run_extension_wave.sh
     full` runs both -> wait -> tp2 -> wait_tp2 -> a zero-new-results shutdown guard (mirrors
     `run_enhance_4090.sh`) in one call, Trimmed by default (see step 8) — pair with
     `nohup ... &` + `cloud/failsafe_extension.sh` per the pattern below.
8. Money-safety, same pattern as `run_enhance_4090.sh` (Trimmed shown; prepend
   `NEOX_SEEDS="1 2"` to the second line for Full — explicit opt-in only):
   ```
   nohup bash cloud/failsafe_extension.sh &          # 30h hard cap, cancel: touch /root/NO_SHUTDOWN
   nohup bash cloud/run_extension_wave.sh full >> cloud/logs_ext/full.log 2>&1 &
   ```
9. Periodically, from **local**: `bash cloud/sync_results.sh --host <ip> --port <port>
   --key ~/.ssh/id_autodl` to pull results back (unchanged from the original wave).
10. After drain, final `sync_results.sh` pull, then tear down.

## Model-gate contract (per team-lead brief's requirement)

Every row in both new drivers goes through `run_row`'s existing built-in model-directory
check (extracts `--model <dir>` from the command, `CONFIG-SKIP`s cleanly if the dir is
absent — this is `run_instruct.sh`'s exact pattern, not new code) PLUS an explicit
`integrity_check.py --expect_params` soft-gate per model (`engine/family_<tag>_
integrity.ok` / `engine/extcausal_<tag>_integrity.ok`) that re-derives every launch and
never hard-aborts the whole driver — a missing or still-downloading model produces loud
`MODEL-ABSENT`/`INTEGRITY-FAIL` log lines and every row for that tag `CONFIG-SKIP`s, the
driver still completes normally and the post-processing steps just skip that tag's table.
Provisioning a box with any subset of the 7 new models missing will NOT wedge the queue.

## What's verified vs. what isn't

**Verified (CPU, no GPU/network beyond 4 read-only `config.json` fetches):**
- `bash -n` clean on all 5 new `.sh` files.
- `cloud/run_extension_wave.sh both`/`wait`/`tp2` DRYRUN-simulated (fake 2-worker launch,
  PID files, distinct log paths, no collision with `cloud/logs/` from the original wave).
- `run_family_transfer.sh`/`run_extension_causal_seeds.sh` DRYRUN output inspected —
  correct `--model`/`--layer`/`--lr`/`--seed`/`--out` per row, correct `needs` gating
  chain, `FAMILY_MODELS`/`TRACK2_SCOPE`/`PYTHIA_SEEDS`/`NEOX_SEEDS` subsetting confirmed.
- All 4 Track-1 model repos + architectures confirmed live (no auth wall, layer counts as
  tabled above) via a direct `config.json` fetch, 2026-07-11.
- Every driver invocation/flag referenced (`killgate_keygeom.py`'s `--device_map`,
  `--model_dtype`, `--alpha_proj_source`, `--holdout_frac`; `mechanism_sc_table.py`,
  `aggregate_g4_causal.py`'s CLI) checked against `--help` output / existing driver
  precedent, not assumed.

**NOT verified (flag loudly, matches `cloud/README.md`'s own precedent for the original
wave):**
- No AutoDL box, no real GPU, no real download was touched.
- The 5 per-row minute estimates (Track 1) are scaled guesses, not measurements — none of
  these 4 architectures have ever run a ROME/AlphaEdit edit in this harness before.
- gemma-2-9b's ~24GB VRAM headroom at bf16 (~18.5GB weights + COMMON-settings activations)
  is a tighter fit than the already-proven Llama-3.1-8B (~16GB weights) — if this OOMs on
  a real 4090, the fix is almost certainly `--n_probes` reduction, not a code change, but
  this build has no way to confirm without the real box.
- The $/¥ cost table's per-card-hour rate is inferred from the prior wave's own "~20.5h ->
  $12-15" precedent, not looked up independently.
- neox20b's `run_extension_causal_seeds.sh` TP path reuses `experiments/
  smoke_neox20b_tp_onbox.py`'s existing hard gate (proven logic, unchanged) — but this
  driver's own SMOKE row (`alphaHO_neox20b_L16`, the alpha+L16+lr0.5 combo specifically)
  has never run for real, unlike `run_neox20b.sh`'s original L33 smoke.
