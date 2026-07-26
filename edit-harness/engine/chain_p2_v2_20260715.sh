#!/bin/bash
# chain_p2_20260715.sh — Pro-6000 (box 29246) P2 wave: D3 transplant E0b, then D2 width
# cells 7B -> 8B -> 14B (14B strictly SOLO — nothing co-scheduled, per prereg ops note).
# Serial by design: fp32 footprints (7B 30G / 8B 32G / 14B 59G) + D3's sequential 14B
# loads make concurrency not worth the risk on one card.
# House rules: pidfile + kill -0 only; per-driver rc captured; idempotent (each driver
# has its own skip-if-exists); budget enforced by each driver's own clock.
set -u
H=/root/edit-harness
cd "$H" || exit 1
export CLOUD_PY=/root/miniconda3/bin/python3
export PY=/root/miniconda3/bin/python3   # width-driver convention (rc=3 preflight fix)
export WAVE_BOX=pro6000
LOG=engine/chain_p2_20260715.log
op=$(cat engine/chain_p2_20260715.pid 2>/dev/null); if kill -0 "$op" 2>/dev/null; then echo "another chain instance alive ($op) — refusing double-launch" >&2; exit 9; fi
echo $$ > engine/chain_p2_20260715.pid
log(){ echo "[chainP2 $(date '+%F %T')] $*" >> "$LOG"; }
log "================ CHAIN_P2 START (pid $$) ================"

run(){ # name, cmd...
  local name=$1; shift
  log "RUN $name"
  "$@" >> "engine/chainP2_${name}.out" 2>&1
  local rc=$?
  log "DONE $name rc=$rc"
  echo "$rc" > "engine/chainP2_${name}.rc"
  return $rc
}

# 1) D3 transplant E0b (both pairs; driver has own gates/budget)
run transplant ./run_transplant_e0b.sh || log "WARN transplant rc!=0 — continuing to D2 (independent science)"

# 2) D2 width cells (merging_m0 fp32; driver has own GPU-idle+selftest gates)
run width_7b  env MODEL_DIR=data/models/Qwen2.5-7B   MODEL_TAG=qwen25_7b  RG_SEEDS=0,1,2 RG_GROUP_SIZES=2,3,5,10,20 BUDGET_MIN=120 ./run_merging_width.sh || log "WARN width_7b failed"
run width_8b  env MODEL_DIR=data/models/Llama-3.1-8B MODEL_TAG=llama31_8b RG_SEEDS=0,1,2 RG_GROUP_SIZES=2,3,5,10,20 BUDGET_MIN=120 ./run_merging_width.sh || log "WARN width_8b failed"
run width_14b env MODEL_DIR=data/models/Qwen2.5-14B  MODEL_TAG=qwen25_14b RG_SEEDS=0,1,2 RG_GROUP_SIZES=2,3,5,10,20 BUDGET_MIN=180 ./run_merging_width.sh || log "WARN width_14b failed"

log "================ CHAIN_P2 COMPLETE ================"
echo "CHAIN_P2_DONE" >> "$LOG"
