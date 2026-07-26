#!/bin/bash
# cloud/gpu_idle_lib.sh — sourceable per-card GPU idle gate for the cloud multi-worker
# launcher (2026-07-08). NOT currently sourced by any existing run_*.sh at the repo
# root (those are read-only reference for this build) — this is the CONTRACT that
# WP-review should wire new/edited drivers to, so that two workers pinned to two
# different physical cards each gate on THEIR OWN card instead of racing on
# nvidia-smi's default GPU-0-first output (see cloud/README.md "driver idle-gate
# contract" for the full story and why the zero-edit path is `both` + SKIP_IDLE_GATE
# rather than this lib, until drivers actually source it).
#
# Usage (inside a driver, in place of the inline util<25/mem<1500/head -1 poll loop
# that run_ripple.sh / run_mquake_law.sh / run_8bcausal.sh each hand-roll today):
#   source "$(dirname "$0")/cloud/gpu_idle_lib.sh"
#   idle_gate_wait || exit 2
#
# Honors two env knobs, both default to today's single-GPU inline behavior if unset:
#   SKIP_IDLE_GATE=1        — bypass entirely. Safe ONLY on a dedicated box with no
#                              other GPU consumers (e.g. a freshly-provisioned AutoDL
#                              instance running exactly this wave, nothing else).
#   IDLE_GATE_DEVICE=<idx>  — restrict the nvidia-smi query to physical GPU <idx> via
#                              `-i <idx>`, instead of the unqualified query whose
#                              `head -1` always returns GPU 0 regardless of which card
#                              the caller actually holds. CUDA_VISIBLE_DEVICES does NOT
#                              affect nvidia-smi's enumeration — it is a CUDA-runtime
#                              remap, invisible to the separate nvidia-smi binary.
set -u

UTIL_MAX=${UTIL_MAX:-25}
MEM_MAX_MIB=${MEM_MAX_MIB:-1500}
IDLE_GATE_TIMEOUT_S=${IDLE_GATE_TIMEOUT_S:-1800}
IDLE_GATE_POLL_S=${IDLE_GATE_POLL_S:-30}
IDLE_GATE_CONSEC=${IDLE_GATE_CONSEC:-3}

idle_gate_wait() {
  if [ "${SKIP_IDLE_GATE:-0}" -eq 1 ]; then
    echo "[gpu_idle_lib] SKIP_IDLE_GATE=1 — bypassing idle gate (dedicated-box path)"
    return 0
  fi
  local dev_args=()
  [ -n "${IDLE_GATE_DEVICE:-}" ] && dev_args=(-i "$IDLE_GATE_DEVICE")
  local t0 consec=0
  t0=$(date +%s)
  while [ "$consec" -lt "$IDLE_GATE_CONSEC" ]; do
    local line util mem
    line=$(nvidia-smi "${dev_args[@]}" --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    util=$(echo "$line" | awk -F, '{gsub(/[^0-9]/,"",$1); print $1}')
    mem=$(echo "$line"  | awk -F, '{gsub(/[^0-9]/,"",$2); print $2}')
    if [ -n "$util" ] && [ -n "$mem" ] && [ "$util" -lt "$UTIL_MAX" ] && [ "$mem" -lt "$MEM_MAX_MIB" ]; then
      consec=$((consec+1))
    else
      consec=0
      if [ $(( $(date +%s) - t0 )) -gt "$IDLE_GATE_TIMEOUT_S" ]; then
        echo "[gpu_idle_lib] ABORT: GPU (dev=${IDLE_GATE_DEVICE:-any}) busy >${IDLE_GATE_TIMEOUT_S}s at gate" >&2
        return 2
      fi
    fi
    echo "[gpu_idle_lib] poll dev=${IDLE_GATE_DEVICE:-any} util=${util:-NA} mem=${mem:-NA} consec=${consec}/${IDLE_GATE_CONSEC}"
    [ "$consec" -lt "$IDLE_GATE_CONSEC" ] && sleep "$IDLE_GATE_POLL_S"
  done
  echo "[gpu_idle_lib] GPU (dev=${IDLE_GATE_DEVICE:-any}) idle — window opens now"
  return 0
}
