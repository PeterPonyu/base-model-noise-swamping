#!/bin/bash
# run_t11_unblock_20260714.sh — LOCAL 5090, ¥0. User-approved 2026-07-14 ("option 2").
# Serial chain, two stages:
#   A) 9 qwen15b gate cells (L17/21/24 × s0/1/2) with --save_matrices → gives the T1.1
#      arch-2 within-probe collateral C(L) that exists nowhere today (only L14 does).
#      Invocation cloned VERBATIM from archive/drivers/run_deep_until1900.sh (the driver
#      that produced the comparable gate_qwen15b_rome_cf_L14 cells): CF/COMMON/lr identical.
#   B) 2 merging-RG depth cells via run_merging_width.sh at explicit LAYER=14 and LAYER=24
#      (L21 already done today) → gives arch-2 merge M(L) at 3 depths aligned with raw-K.
# Together these make the T1.1 literal two-family gate DATA-complete (the gate script still
# needs its literal arch-2 target constructor added — separate authoring+review task).
# House rules: pidfile + kill -0 only (never pgrep), budget clock from WORK start,
# skip-if-exists per cell, per-cell timeout, killgate does its own atomic writes.
set -u
H=/home/zeyufu/Desktop/idea-feasibility-analysis/edit-harness
cd "$H" || exit 1
PY=/home/zeyufu/miniconda3/envs/dl/bin/python3
KG="experiments/killgate_keygeom.py"
ENVP="env -u ALL_PROXY -u all_proxy HF_HUB_OFFLINE=1"
CF="--dataset counterfact --data data/counterfact.json"
COMMON="--n_edits 200 --n_probes 500 --steps 20 --save_matrices --matrix_dir results/matrices"
LOG=engine/run_t11_unblock_20260714.log
BUDGET_MIN=${BUDGET_MIN:-300}
CELL_TIMEOUT_MIN=${CELL_TIMEOUT_MIN:-45}
echo $$ > engine/run_t11_unblock_20260714.pid
log(){ echo "[t11unblock $(date '+%F %T')] $*" >> "$LOG"; }
log "================ START (pid $$, budget ${BUDGET_MIN}m) ================"

# GPU idle gate: util<25 && mem<1500, 3 consecutive polls (house pattern)
consec=0
for i in $(seq 1 60); do
  read -r util mem <<<"$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | head -1 | tr -d ',')"
  if [ "${util:-99}" -lt 25 ] && [ "${mem:-99999}" -lt 1500 ]; then consec=$((consec+1)); else consec=0; fi
  [ "$consec" -ge 3 ] && break
  sleep 20
done
[ "$consec" -lt 3 ] && { log "ABORT: GPU never idle"; exit 4; }
T0=$(date +%s)   # budget clock starts at WORK start (post-gate)
over_budget(){ [ $(( ($(date +%s) - T0) / 60 )) -ge "$BUDGET_MIN" ]; }

fails=0
# -------- Stage A: 9 gate cells
for L in 17 21 24; do
  for S in 0 1 2; do
    outj="results/gate_qwen15b_rome_cf_L${L}_s${S}.json"
    outm="results/matrices/gate_qwen15b_rome_cf_L${L}_s${S}.npz"
    if [ -f "$outj" ] && [ -f "$outm" ]; then log "skip (exists): L${L} s${S}"; continue; fi
    over_budget && { log "BUDGET reached before L${L} s${S} — stopping"; exit 6; }
    log "RUN gate qwen15b L${L} s${S}"
    timeout $((CELL_TIMEOUT_MIN*60)) $ENVP $PY $KG --model data/models/Qwen2.5-1.5B \
      --editor rome $CF $COMMON --lr 0.1 --layer "$L" --seed "$S" --out "$outj" \
      >> "$LOG" 2>&1
    rc=$?
    if [ $rc -ne 0 ] || [ ! -f "$outm" ]; then
      log "FAIL L${L} s${S} rc=$rc (matrices present: $([ -f "$outm" ] && echo yes || echo no))"
      fails=$((fails+1)); [ $fails -ge 2 ] && { log "ABORT: 2 consecutive fails"; exit 7; }
    else
      log "OK L${L} s${S}"; fails=0
    fi
  done
done

# -------- Stage B: 2 RG depth cells (driver has its own gates; run serially here)
for L in 14 24; do
  tab="results/merging/RG_operating_curve_table_qwen15b_L${L}.json"
  if [ -f "$tab" ]; then log "skip (exists): RG L${L}"; continue; fi
  over_budget && { log "BUDGET reached before RG L${L} — stopping"; exit 6; }
  log "RUN merging-width RG qwen15b L${L}"
  MODEL_DIR=data/models/Qwen2.5-1.5B MODEL_TAG=qwen15b LAYER=$L \
    RG_SEEDS=0,1,2 RG_GROUP_SIZES=2,3,5,10,20,50,100 BUDGET_MIN=60 \
    ./run_merging_width.sh >> "$LOG" 2>&1
  rc=$?
  [ $rc -ne 0 ] || [ ! -f "$tab" ] && { log "FAIL RG L${L} rc=$rc"; exit 8; }
  log "OK RG L${L}"
done

log "================ COMPLETE ================"
echo "T11_UNBLOCK_DONE" >> "$LOG"
