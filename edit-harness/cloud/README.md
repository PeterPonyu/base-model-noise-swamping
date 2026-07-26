# cloud/ — AutoDL dual-4090 orchestration runbook

Built + CPU-tested 2026-07-08. **No GPU, no downloads, no network were used to build
this** — everything here was verified with `bash -n`, a CPU-only launcher simulation
(`selftest.sh`), and code review against the local drivers it references read-only.
Do not treat anything as "GPU-verified" until it has actually run on the AutoDL box.

**2026-07-08 wave-review update:** the original design here was **seed**-sharded (both
cards run the same 6 drivers, distinguished only by a `SEED_OVERRIDE` env var). Review
found `SEED_OVERRIDE` is not actually read by any driver, so both cards would execute
the identical `--out`/`.npz` paths concurrently — a real corruption risk given
killgate's non-atomic npz write, not just wasted compute. Re-sharded to **driver**-level
instead: each of the 6 drivers now runs on exactly one card (see the map below), so no
two workers ever target the same output path. `SEED_OVERRIDE` is still exported for
forward-compat but is inert against every driver in this repo — do not rely on it.

## Driver-shard map

| Card                | Drivers                                              | Est. total |
|----------------------|-------------------------------------------------------|-----------:|
| AutoDL card 0        | `run_8bcausal.sh`, `run_ripple.sh`, `run_cfplus.sh`    | ~723 min   |
| AutoDL card 1        | `run_mquake_law.sh`, `run_mquake_t.sh`, `run_glue_seq.sh` | ~748 min |

Balanced from each driver's own SCIENCE+SMOKE minute estimates (summed from its
`run_row` calls), ~3% apart. card0 deliberately carries the heaviest single driver
(`run_8bcausal.sh`, the 8B-model cells) paired with two light ones; card1 carries the
two mid-weight MQuAKE drivers (~350 min each) balanced by the smallest driver
(`glue_seq`). Override via `DRIVERS_CARD0`/`DRIVERS_CARD1` env vars, or per-invocation
with `run_cloud_wave.sh card0/card1 [seed] [drivers]`.

Every driver already sweeps its own seeds 0/1/2 internally (see e.g.
`run_mquake_law.sh`'s `gate_llama1b_rome_mquake_L8_s{0,1,2}` rows), so running each
driver once — on one card — captures the full 3-seed local plan without duplicating
seed-0 work that already ran on the local machine. (The local machine's own seed-0 runs
predate this cloud wave and are separate — see workspace `CLAUDE.md`/memory.)

## Ordered runbook

1. **Open the AutoDL box** in "无卡模式" (no-GPU billing tier).
2. `ssh` in, clone/pull this repo to the box (adjust `cloud/sync_results.sh
   --remote-root` to wherever it lands — default assumption is `/root/edit-harness`).
3. `bash cloud/setup_autodl.sh all` — asserts the `/root/autodl-tmp` data disk, points
   `HF_HOME`/`HF_ENDPOINT` at it via hf-mirror, verifies the python/torch env, and
   downloads the ≤24GB-cell model set straight into `<repo-root>/data/models/<name>`
   via `hf download --local-dir` (idempotent — safe to re-run; NOT the HF cache — every
   driver hardcodes `--model data/models/<name>`, so weights must land there). Add
   `--with-20b` only if/when you're about to run WP3's tensor-parallel phase (~40GB
   extra, lands at `data/models/gpt-neox-20b`).
4. `bash cloud/setup_autodl.sh patch-drivers` — **box-only, run once, right here** (see
   "driver idle-gate contract" and "portable H/PY" below for why). Sed-patches the 3
   existing chain-locked drivers' (`run_ripple.sh`, `run_mquake_law.sh`,
   `run_8bcausal.sh`) on THIS on-box copy only, in two independent passes: idle-gate
   (`cloud/patch_idle_gate.sed`, honors `SKIP_IDLE_GATE`) and H/PY portability
   (`cloud/patch_h_py.sed`, fixes the hardcoded local repo path + python interpreter —
   without this every driver's `cd "$H"` fails on the box and exits immediately, zero
   science). Each pass is independently idempotent (grep-guarded on its own marker).
   **Hard-guarded against accidental local execution** — refuses unless at least one
   on-box signal is present (`$H` not under `/home/zeyufu`, `AUTODL_BOX=1`, or the data
   disk exists); see `cloud/setup_autodl.sh`'s `phase_patch_drivers()`/`on_box()`.
5. **Switch the instance to the 2x4090 GPU tier** in the AutoDL console.
6. `bash cloud/setup_autodl.sh gpu-check` — confirms `torch.cuda.device_count() == 2`.
7. `bash cloud/run_cloud_wave.sh both` — the one command. Launches both driver-shard
   workers together (card0/card1 per the map above). No longer required to start in
   lockstep for gate safety (see below), but `both` remains the simplest path.
8. Periodically, from **local**: `bash cloud/sync_results.sh --host <ip> --port <port>
   --key ~/.ssh/id_autodl [--dry-run]` to pull results back without waiting for the
   whole wave to finish.
9. After the driver-shard phase drains (`bash cloud/run_cloud_wave.sh wait` blocks until
   both workers exit): `bash cloud/run_cloud_wave.sh tp20b` for WP3's 20B
   tensor-parallel phase, which needs **both** cards together, not one each — hence it
   runs only after the per-card phase above has released them.
10. Final `sync_results.sh` pull, then tear down the instance.

## Driver idle-gate contract — the decision, spelled out

**As of the 2026-07-08 wave-review fix, this is resolved for all 6 drivers, provided
step 4 above (`patch-drivers`) was run.** The underlying problem, still worth
understanding: `run_ripple.sh`, `run_mquake_law.sh`, `run_8bcausal.sh` each inline their
own idle-gate loop as
```
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | head -1
```
— no `-i <device>` flag. `nvidia-smi`'s enumeration is **not** affected by
`CUDA_VISIBLE_DEVICES` (that's a CUDA-runtime-only remap the separate `nvidia-smi`
binary never sees), so on a 2-GPU box `head -1` **always** returns physical GPU 0's
row, regardless of which card the caller is pinned to. A worker pinned to card 1 via
`CUDA_VISIBLE_DEVICES=1` would gate on **GPU 0's** load, not GPU 1's — so a driver
re-gating mid-wave (e.g. after a restart) could stall up to 30 minutes waiting on the
wrong card, or falsely proceed while its own card is genuinely busy.

Two fixes landed for this:

- **The 3 existing chain-locked drivers** (`run_ripple.sh`, `run_mquake_law.sh`,
  `run_8bcausal.sh`) cannot be edited locally — a live local GPU chain imports them
  right now. Fix: `cloud/setup_autodl.sh patch-drivers` (step 4 above) sed-patches
  **the on-box copy only**, via `cloud/patch_idle_gate.sed`, wrapping their inline gate
  loop so `SKIP_IDLE_GATE=1` (exported by `run_cloud_wave.sh` by default) bypasses it
  entirely. This never touches the local repo — see the sed script's own header and
  `setup_autodl.sh`'s `phase_patch_drivers()` comment for the guard.
- **The 3 new WP2 drivers** (`run_cfplus.sh`, `run_glue_seq.sh`, `run_mquake_t.sh`) are
  not chain-locked and were authored/updated to `source cloud/gpu_idle_lib.sh;
  idle_gate_wait` directly — no patching needed, they honor `SKIP_IDLE_GATE` and
  `IDLE_GATE_DEVICE` natively.

**If step 4 is skipped** (e.g. a fresh box where you forgot `patch-drivers`), the 3
existing drivers fall back to their original unpatched behavior — in that case, launch
both workers together (`run_cloud_wave.sh both`) so both inline gates poll GPU 0 (idle,
since nothing has started yet) and both clear before either driver's first real job
starts; do not restart a single worker mid-wave in that fallback mode, since the
restarted worker's gate would see the busy card (always GPU 0) and either hang or
falsely proceed. This fallback caveat does not apply to the 3 WP2 drivers, which always
gate correctly regardless of `patch-drivers`.

## Portable H/PY — the B4 fix

Every driver in this repo (all 6 wave drivers + `run_neox20b.sh`) was authored on the
local laptop and hardcodes `H=/home/zeyufu/Desktop/idea-feasibility-analysis/edit-harness`
+ `cd "$H"` and `PY=/home/zeyufu/miniconda3/envs/dl/bin/python3`. **Neither exists on the
AutoDL box** — the repo lands at wherever you rsync/clone it (e.g. `/root/edit-harness`),
and the box's own image ships its own python. Unpatched, `cd "$H"` fails immediately and
every driver exits before doing any science — this is why `patch-drivers` (step 4) and
`CLOUD_PY` both matter, not just the idle-gate.

Two fixes, same split as the idle-gate one above:

- **The 3 WP2 drivers + `run_neox20b.sh`** (not chain-locked) were edited directly: `H`
  now derives from the script's own location (`H="$(cd "$(dirname "$0")" && pwd)"`), and
  `PY` gets a `PY="${CLOUD_PY:-$PY}"` line right after the original assignment — the
  local hardcoded path stays as the fallback default, so nothing changes for local runs;
  `CLOUD_PY`, once set, overrides it.
- **The 3 chain-locked drivers** get the same rewrite via `cloud/patch_h_py.sed`
  (part of step 4's `patch-drivers`, applied only to the on-box copy).

`cloud/run_cloud_wave.sh` resolves `CLOUD_PY` once at load time (defaults to `python3` on
`PATH`, override with `CLOUD_PY=<path>` if the box's python isn't on `PATH`) and exports
it into every worker subshell, so all 6 drivers pick it up automatically once patched.

## What's genuinely untested (flag loudly)

- **No real AutoDL box, no real SSH, no real GPUs were touched.** `selftest.sh` proves
  the launcher's process/PID/file-collision mechanics on CPU with a 2-line fake driver
  — it does not prove any actual driver script runs correctly on a 4090.
- **`/root/autodl-tmp` as the data-disk mount point** is an assumption about AutoDL's
  default layout, not verified against a live instance.
- **Exact HF repo IDs** in `setup_autodl.sh` (`meta-llama/Llama-3.2-1B`,
  `meta-llama/Llama-3.1-8B`, etc.) are guessed from local `data/models/` directory
  names and this repo's existing `download_models.py`/`download_manifest.sh`
  precedent — not independently re-verified. The two `meta-llama/*` ones are gated;
  the download phase checks `hf auth whoami` and warns, but does not verify the
  license has actually been accepted for *this* HF account before attempting the pull.
- **hf-mirror.com reachability/behavior from inside AutoDL** is asserted per the task
  brief, not tested (no network in this build environment).
- **`hf download --local-dir` behavior for large multi-file repos on hf-mirror.com**
  (retry/resume semantics, whether it stages via `.incomplete` the same way the cache
  path does) is assumed to match the documented CLI behavior — not verified against a
  live pull.
- **WP2's drivers** (`run_cfplus.sh`, `run_glue_seq.sh`, `run_mquake_t.sh`) and **WP3's**
  (`run_neox20b.sh`) now exist and are wired into `run_cloud_wave.sh`'s default
  `DRIVERS_CARD0`/`DRIVERS_CARD1` split and the `tp20b` subcommand respectively, but
  have never run on a real GPU through this launcher — the launcher checks `[ -f "$d" ]`
  and skips missing ones with a logged note, so it degrades gracefully if any are absent.
- **The `patch-drivers` sed patches** (`cloud/patch_idle_gate.sed` + `cloud/patch_h_py.sed`)
  were verified with `bash -n` and a functional CPU harness against scratch copies of
  the 3 target drivers (idle-gate bypass short-circuits with zero `nvidia-smi` calls
  under `SKIP_IDLE_GATE=1`, original poll loop unchanged otherwise; H/PY patch resolves
  a portable `$H` and honors `CLOUD_PY` while preserving the local default; both passes
  are idempotent on re-run) — never applied to a real on-box checkout. The `on_box()`
  hard guard was verified to refuse a real invocation against this local repo (exit 1,
  zero bytes changed) and to correctly pass for each of its 3 bypass signals tested in
  isolation (non-local `$H`, `AUTODL_BOX=1`, data-disk present).
