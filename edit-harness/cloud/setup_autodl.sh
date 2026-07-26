#!/bin/bash
# cloud/setup_autodl.sh — AutoDL dual-4090 (2x24GB) box bootstrap (2026-07-08). Run this
# ON the AutoDL instance after SSH. Two-phase by design:
#
#   PHASE 1 (disk + deps + model download) — run in AutoDL's "无卡模式" (no-GPU billing
#   tier, near-free) BEFORE attaching GPUs. Downloading weights needs no CUDA; paying
#   for idle GPU time during a multi-hour model pull is wasted money.
#   PHASE 2 (gpu-check) — run AFTER switching to the GPU instance, right before
#   cloud/run_cloud_wave.sh, to confirm torch sees both cards.
#
# Usage:
#   bash cloud/setup_autodl.sh disk                     # phase 1: assert data disk, set HF env
#   bash cloud/setup_autodl.sh deps                      # phase 1: verify python/torch/transformers
#   bash cloud/setup_autodl.sh download [--with-20b]     # phase 1, no-GPU mode
#   bash cloud/setup_autodl.sh gpu-check                 # phase 2, needs GPUs attached
#   bash cloud/setup_autodl.sh all [--with-20b]          # disk+deps+download in sequence
#
# DRYRUN=1 prints every command instead of running it. This build has no network/GPU
# access, so it was authored and verified with DRYRUN=1 (see cloud/selftest.sh) — on
# the real box, run with DRYRUN unset (defaults to 0, i.e. actually do it).
set -u
DRYRUN=${DRYRUN:-0}
DATA_DISK=${DATA_DISK:-/root/autodl-tmp}
HF_HOME_DIR="$DATA_DISK/hf_cache"
PY=${CLOUD_PY:-python3}
# Repo root (the edit-harness checkout this script lives in) — NOT the data disk. Drivers
# reference weights via the relative path `data/models/<name>` from here, so downloaded
# models must land under $H/data/models/, separate from $HF_HOME_DIR's cache.
H="$(cd "$(dirname "$0")/.." && pwd)"

run(){ if [ "$DRYRUN" -eq 1 ]; then echo "DRYRUN: $*"; else eval "$*"; fi; }
log(){ echo "[setup_autodl $(date '+%F %T')] $*"; }

phase_disk(){
  log "asserting data disk ${DATA_DISK}"
  if [ ! -d "$DATA_DISK" ]; then
    log "FATAL: ${DATA_DISK} not found — AutoDL data disk not mounted. Check the instance's"
    log "  disk config in the AutoDL console; the data disk should be pre-mounted at boot."
    log "  ASSUMPTION FLAGGED: this path (/root/autodl-tmp) is AutoDL's documented default"
    log "  data-disk mount point as of this writing — override with DATA_DISK=<path> if the"
    log "  actual box differs."
    return 1
  fi
  local avail
  avail=$(df --output=avail -BG "$DATA_DISK" 2>/dev/null | tail -1 | tr -dc 0-9)
  log "data disk avail: ${avail:-unknown}G"
  run "mkdir -p '${HF_HOME_DIR}'"
  # Persist for subsequent shells and for run_cloud_wave.sh's launched workers.
  run "grep -q 'HF_ENDPOINT=https://hf-mirror.com' ~/.bashrc 2>/dev/null || printf '%s\n' 'export HF_ENDPOINT=https://hf-mirror.com' 'export HF_HOME=${HF_HOME_DIR}' >> ~/.bashrc"
  export HF_ENDPOINT=https://hf-mirror.com
  export HF_HOME="$HF_HOME_DIR"
  log "HF_ENDPOINT=${HF_ENDPOINT} HF_HOME=${HF_HOME} (on data disk, not system disk)"
}

phase_deps(){
  log "verifying python env (ASSUMPTION FLAGGED: assumes a pre-existing torch env is"
  log "  already active — AutoDL community images ship one; this script does not create"
  log "  or pip-install one. If any import below fails: activate/create the env, then"
  log "  'pip install torch transformers numpy', then re-run 'deps')"
  run "${PY} -c 'import sys; print(\"python\", sys.version.split()[0])'"
  run "${PY} -c 'import torch; print(\"torch\", torch.__version__, \"cuda_available:\", torch.cuda.is_available())'"
  run "${PY} -c 'import transformers; print(\"transformers\", transformers.__version__)'"
  run "${PY} -c 'import numpy; print(\"numpy\", numpy.__version__)'"
}

# ASSUMPTION FLAGGED: exact repo IDs guessed to match local data/models/ naming
# (Llama-3.2-1B, Llama-3.2-3B, Llama-3.1-8B, Qwen2.5-1.5B, gemma-2-2b, Phi-3.5-mini,
# gpt-j-6b) and this repo's own download_models.py/download_manifest.sh precedent for
# the ungated ones (unsloth/Llama-3.2-3B, Qwen/Qwen2.5-*, unsloth/gemma-2-2b,
# microsoft/Phi-3.5-mini-instruct, EleutherAI/gpt-j-6b). Llama-3.2-1B and Llama-3.1-8B
# are GATED meta-llama repos on the real box (see workspace memory: license accepted
# 2026-07-06 for 1B-Instruct) — verify with `hf auth whoami` below and expect 403s if
# this HF account hasn't accepted the license for the exact repos pulled here.
MODELS_CORE="meta-llama/Llama-3.2-1B unsloth/Llama-3.2-3B meta-llama/Llama-3.1-8B Qwen/Qwen2.5-1.5B unsloth/gemma-2-2b microsoft/Phi-3.5-mini-instruct EleutherAI/gpt-j-6b"
MODEL_20B="EleutherAI/gpt-neox-20b"   # ~40GB — gated behind --with-20b, WP3's TP phase only

# repo_id -> local basename under data/models/, matching this repo's OWN naming exactly
# (verified against `ls data/models/` and data/DOWNLOADS-20260706.md on the local box —
# 2026-07-08 fix for the B1 model-path blocker: every run_*.sh driver hardcodes
# `--model data/models/<basename>`, so weights MUST land there, not in the HF cache).
model_basename(){
  case "$1" in
    meta-llama/Llama-3.2-1B) echo "Llama-3.2-1B" ;;
    unsloth/Llama-3.2-3B) echo "Llama-3.2-3B" ;;
    meta-llama/Llama-3.1-8B) echo "Llama-3.1-8B" ;;
    Qwen/Qwen2.5-1.5B) echo "Qwen2.5-1.5B" ;;
    unsloth/gemma-2-2b) echo "gemma-2-2b" ;;
    microsoft/Phi-3.5-mini-instruct) echo "Phi-3.5-mini" ;;
    EleutherAI/gpt-j-6b) echo "gpt-j-6b" ;;
    EleutherAI/gpt-neox-20b) echo "gpt-neox-20b" ;;
    *) return 1 ;;
  esac
}

# Downloads ONE repo straight into $H/data/models/<basename> via `hf download --local-dir`
# (real files, not the HF-cache symlink farm) — the fix for B1. Idempotent: skips if the
# target dir already holds files and no .incomplete markers.
download_one(){
  local repo="$1" name target
  name=$(model_basename "$repo") || { log "FATAL: no local basename mapping for ${repo} — add one to model_basename()"; return 1; }
  target="${H}/data/models/${name}"
  if [ "$DRYRUN" -ne 1 ] && [ -d "$target" ] && [ -n "$(find "$target" -maxdepth 1 -type f 2>/dev/null)" ] && [ -z "$(find "$target" -name '*.incomplete' 2>/dev/null)" ]; then
    log "skip ${repo} (weights present at ${target}, no .incomplete files — idempotent)"
    return 0
  fi
  run "mkdir -p '${target}'"
  run "hf download '${repo}' --local-dir '${target}' --exclude '*.bin' --exclude '*.pth' --exclude '*.h5'"
}

phase_download(){
  local with20b=0
  [ "${1:-}" = "--with-20b" ] && with20b=1
  export HF_ENDPOINT=https://hf-mirror.com
  export HF_HOME="$HF_HOME_DIR"
  log "download phase — HF_HOME=${HF_HOME} via mirror ${HF_ENDPOINT}, weights -> ${H}/data/models/"
  log "REMINDER: run this in AutoDL's 无卡模式 (no-GPU tier) — no CUDA needed to pull weights"
  run "hf auth whoami || echo 'NOT LOGGED IN — gated meta-llama pulls below will 403; run: hf auth login'"
  local m
  for m in $MODELS_CORE; do
    download_one "$m"
  done
  if [ "$with20b" -eq 1 ]; then
    log "20B phase (~40GB, --with-20b given) — confirm disk headroom before this runs for real"
    run "df --output=avail -BG '${DATA_DISK}'"
    download_one "$MODEL_20B"
  else
    log "skipping GPT-NeoX-20B (pass --with-20b to include; only needed for WP3's TP phase, run AFTER GPUs attach)"
  fi
}

phase_gpu_check(){
  run "nvidia-smi -L"
  run "${PY} -c 'import torch; print(\"cuda device count:\", torch.cuda.device_count())'"
}

# BOX-ONLY — wave-review B3 (idle-gate) + B4 (portable H/PY) fixes (2026-07-08).
# Patches the CLOUD COPY of the 3 existing chain-locked drivers (run_ripple.sh,
# run_mquake_law.sh, run_8bcausal.sh) at $H via cloud/patch_idle_gate.sed (SKIP_IDLE_GATE
# support) and cloud/patch_h_py.sed (portable H + CLOUD_PY override — without this, `cd
# "$H"` fails on the box since H is hardcoded to this laptop's path, and every driver
# exits before doing any science). These 3 files are read/imported by a LIVE local GPU
# chain right now and must NEVER be edited on the local machine — this subcommand is
# never invoked automatically (not part of the `all` sequence) and is meant to run ONLY
# here, on the box, right after the repo is rsynced/cloned to $H. Each sed's effect is
# independently idempotent (grep-guarded on its own marker) so re-running doesn't
# double-patch either piece.
PATCH_DRIVERS="run_ripple.sh run_mquake_law.sh run_8bcausal.sh"
PATCH_MARKER_GATE="SKIP_IDLE_GATE-bypass (patched by cloud/setup_autodl.sh patch-drivers)"
PATCH_MARKER_HPY='PY="${CLOUD_PY:-$PY}"'

# HARD GUARD (2026-07-08, wave-review safety must-fix): phase_patch_drivers sed-edits
# files at $H in place. If $H ever resolves to the LOCAL repo (e.g. this is run by hand
# on the laptop instead of on the AutoDL box), it would corrupt the 3 chain-locked
# drivers a live local GPU chain is importing right now — refuse unless at least one
# genuine on-box signal is present. Any ONE of these is sufficient:
#   - $H does not start with /home/zeyufu (this repo's known local path)
#   - AUTODL_BOX=1 is explicitly set (manual confirmation override)
#   - the AutoDL data-disk mount ${DATA_DISK} (default /root/autodl-tmp) exists
on_box(){
  case "$H" in /home/zeyufu/*) ;; *) return 0 ;; esac
  [ "${AUTODL_BOX:-0}" = "1" ] && return 0
  [ -d "$DATA_DISK" ] && return 0
  return 1
}

phase_patch_drivers(){
  if ! on_box; then
    log "REFUSING: patch-drivers looks like it's running on the LOCAL machine (H=${H})."
    log "  This subcommand sed-edits the 3 chain-locked drivers a LIVE local GPU chain"
    log "  imports right now — refusing to risk corrupting it. Run this ON THE AUTODL"
    log "  BOX after rsync (where H won't be under /home/zeyufu and/or ${DATA_DISK}"
    log "  exists), or set AUTODL_BOX=1 if you are certain this really is an isolated"
    log "  on-box checkout."
    return 1
  fi
  log "on-box guard passed (H=${H} AUTODL_BOX=${AUTODL_BOX:-0} ${DATA_DISK}=$([ -d "$DATA_DISK" ] && echo present || echo absent))"
  log "patching cloud-copy drivers at ${H}: ${PATCH_DRIVERS} (idle-gate + portable H/PY — box-only, see header comment above)"
  local f target
  for f in $PATCH_DRIVERS; do
    target="${H}/${f}"
    if [ ! -f "$target" ]; then
      log "SKIP ${f} (not found at ${target} — repo layout mismatch?)"
      continue
    fi
    if grep -qF "$PATCH_MARKER_GATE" "$target" 2>/dev/null; then
      log "skip ${f} idle-gate patch (already patched — idempotent)"
    else
      run "sed -i -f '${H}/cloud/patch_idle_gate.sed' '${target}'"
      log "patched ${f} (idle-gate)"
    fi
    if grep -qF "$PATCH_MARKER_HPY" "$target" 2>/dev/null; then
      log "skip ${f} H/PY patch (already patched — idempotent)"
    else
      run "sed -i -f '${H}/cloud/patch_h_py.sed' '${target}'"
      log "patched ${f} (H/PY)"
    fi
  done
}

case "${1:-help}" in
  disk) phase_disk ;;
  deps) phase_deps ;;
  download) shift; phase_download "$@" ;;
  gpu-check) phase_gpu_check ;;
  patch-drivers) phase_patch_drivers ;;
  all) shift; phase_disk && phase_deps && phase_download "$@" ;;
  *) echo "usage: bash cloud/setup_autodl.sh {disk|deps|download [--with-20b]|gpu-check|patch-drivers|all [--with-20b]}"; exit 1 ;;
esac
