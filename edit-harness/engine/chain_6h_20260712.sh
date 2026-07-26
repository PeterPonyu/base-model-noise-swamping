#!/bin/bash
# chain_6h_20260712.sh — user-authorized 6-hour local-5090 window (2026-07-12).
# Sequence: [merging M0 already RUNNING, launched separately] -> wait its drain ->
#   gradsim seed gap-fill (L8/L10/L14 x s1/s2, closes the last 3 dossier PENDINGs) ->
#   half-life HL0 kill-gate (review CONFIRMED-CLEAN 2026-07-12) with the REMAINING
#   window as its budget -> dossier refresh + report.
# All stages idempotent; wait by PID only (kill -0), never pgrep/pkill (standing rule);
# DONE markers validated with the last-DONE-after-last-START ordering check (append-only
# logs contain stale DRYRUN markers — same trap chain_gated_20260711.sh guards against).
# DRYRUN=1: print plan + live gate states, no waits, no launches, no pid/marker writes.
# Namespace: chain_6h_* only. Exit codes: 0 ok / 3 merging-wait timeout / 5 thermal-defer
# (not used here — drivers carry their own gates) / 6 internal.
set -u

H="/home/zeyufu/Desktop/idea-feasibility-analysis/edit-harness"
cd "$H" || exit 6
PY="${PY:-/home/zeyufu/miniconda3/envs/dl/bin/python3}"
command -v "$PY" >/dev/null 2>&1 || PY=python3
ENVP="env -u ALL_PROXY -u all_proxy HF_HUB_OFFLINE=1"
LOG="engine/chain_6h_20260712.log"
PIDFILE="engine/chain_6h_20260712.pid"
REPORT="engine/chain_6h_report.txt"
MERGING_PIDFILE="engine/run_merging_kg0.pid"
MERGING_LOG="engine/run_merging_kg0.log"
WINDOW_MIN="${WINDOW_MIN:-355}"   # hard window: 6h minus margin
T0=$(date +%s)
DEADLINE=$(( T0 + WINDOW_MIN * 60 ))
DRYRUN="${DRYRUN:-0}"

log() { echo "[chain-6h $(date '+%F %T')] $*" | tee -a "$LOG"; }
rem_min() { echo $(( (DEADLINE - $(date +%s)) / 60 )); }

# last-DONE-after-last-START ordering check on the merging log (stale DRYRUN DONE exists)
merging_done() {
  [ -f "$MERGING_LOG" ] || return 1
  local ld ls
  ld=$(grep -n "RUN_MERGING_KG0_DONE" "$MERGING_LOG" | tail -1 | cut -d: -f1)
  ls=$(grep -n "RUN_MERGING_KG0 START" "$MERGING_LOG" | tail -1 | cut -d: -f1)
  [ -n "$ld" ] && [ -n "$ls" ] && [ "$ld" -gt "$ls" ]
}
merging_alive() {
  local p
  p=$(cat "$MERGING_PIDFILE" 2>/dev/null) || return 1
  [ -n "$p" ] && kill -0 "$p" 2>/dev/null
}

gpu_idle_wait() {  # util<25 && mem<1500, 3 consecutive polls @20s; never zero-compute-apps
  local ok=0 u m
  while [ "$ok" -lt 3 ]; do
    read -r u m <<<"$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | head -1 | tr -d ',')"
    if [ "${u:-100}" -lt 25 ] && [ "${m:-99999}" -lt 1500 ]; then ok=$((ok+1)); else ok=0; fi
    [ "$ok" -lt 3 ] && sleep 20
    [ "$(rem_min)" -le 0 ] && { log "DEADLINE hit inside idle-wait"; return 1; }
  done
  return 0
}

if [ "$DRYRUN" -eq 1 ]; then
  echo "DRYRUN chain_6h plan (window ${WINDOW_MIN}m):"
  echo "  stage1: wait merging drain — alive_now=$(merging_alive && echo yes || echo no) done_marker_ordered=$(merging_done && echo yes || echo no)"
  for L in 8 10 14; do for s in 1 2; do
    out="results/GRADSIM_TRUE_Llama-3.2-1B_L${L}_s${s}.json"
    gate="results/matrices/gate_llama1b_rome_cf_L${L}_s${s}.npz"
    echo "  stage2: L${L}_s${s} gate_npz=$( [ -f "$gate" ] && echo present || echo MISSING ) out=$( [ -s "$out" ] && echo EXISTS-skip || echo to-run )"
  done; done
  echo "  stage3: HL0 with BUDGET_MIN=remaining (driver: run_halflife_hl0.sh, review CONFIRMED-CLEAN)"
  echo "  stage4: revision_dossier.py refresh + report"
  exit 0
fi

echo $$ > "$PIDFILE"
log "================ CHAIN_6H START (pid $$, window ${WINDOW_MIN}m) ================"

# ---- stage 1: wait for merging M0 to drain (it was launched before this chain) ----
log "stage1: waiting for run_merging_kg0.sh (pid $(cat "$MERGING_PIDFILE" 2>/dev/null || echo '?'))"
DEAD_SINCE=""
while :; do
  if merging_done; then log "stage1: merging DONE marker ordered-valid"; break; fi
  if merging_alive; then DEAD_SINCE=""; else
    if [ -z "$DEAD_SINCE" ]; then DEAD_SINCE=$(date +%s); fi
    if [ $(( $(date +%s) - DEAD_SINCE )) -gt 600 ]; then
      log "stage1: merging pid dead >10m without ordered DONE — proceeding anyway (its own report records the failure); rc noted"
      break
    fi
  fi
  [ "$(rem_min)" -le 200 ] && { log "stage1: waited too long (rem $(rem_min)m <= 200m) — merging overran; ABORT chain (exit 3)"; exit 3; }
  sleep 60
done
MERGING_RC_NOTE="$(grep -a "RUN_MERGING_KG0 REPORT\|COMPLETE" "$MERGING_LOG" | tail -2 | tr '\n' ' ')"
log "stage1: merging status: ${MERGING_RC_NOTE:-unknown}"

# ---- stage 2: gradsim seed gap-fill L{8,10,14} x s{1,2} (mirrors run_revins.sh Cell C) ----
gpu_idle_wait || exit 3
N2_DONE=0; N2_SKIP=0; N2_FAIL=0
for L in 8 10 14; do for s in 1 2; do
  outdir="results/mechanism/s${s}"; mkdir -p "$outdir"
  mechnpz="${outdir}/Llama-3.2-1B_L${L}.npz"
  gatenpz="results/matrices/gate_llama1b_rome_cf_L${L}_s${s}.npz"
  gsout="results/GRADSIM_TRUE_Llama-3.2-1B_L${L}_s${s}.json"
  if [ -s "$gsout" ]; then log "stage2: L${L}_s${s} SKIP (output exists)"; N2_SKIP=$((N2_SKIP+1)); continue; fi
  if [ ! -f "$gatenpz" ]; then log "stage2: L${L}_s${s} SKIP (gate npz missing: $gatenpz)"; N2_SKIP=$((N2_SKIP+1)); continue; fi
  if [ ! -s "$mechnpz" ]; then
    log "stage2: mech dump L${L}_s${s} -> $mechnpz"
    timeout --signal=TERM --kill-after=60 1500 $ENVP "$PY" experiments/mechanism_dump.py \
      --model data/models/Llama-3.2-1B --data data/counterfact.json --dataset counterfact \
      --n_edits 200 --layer "$L" --seed "$s" --steps 20 --lr 0.1 --device cuda \
      --save_vectors --out_dir "$outdir" >> "$LOG" 2>&1
    [ -s "$mechnpz" ] || { log "stage2: mech dump L${L}_s${s} FAIL"; N2_FAIL=$((N2_FAIL+1)); continue; }
  fi
  log "stage2: gradsim_true L${L}_s${s} -> $gsout"
  timeout --signal=TERM --kill-after=60 1200 $ENVP "$PY" experiments/gradsim_true.py \
    --model data/models/Llama-3.2-1B --layer "$L" --seed "$s" --n_edits 200 --n_probes 500 \
    --gate_npz "$gatenpz" --mech_npz "$mechnpz" --known --edit_ok --device cuda \
    --out "$gsout" >> "$LOG" 2>&1
  if [ -s "$gsout" ]; then N2_DONE=$((N2_DONE+1)); else log "stage2: gradsim L${L}_s${s} FAIL"; N2_FAIL=$((N2_FAIL+1)); fi
  [ "$(rem_min)" -le 230 ] && { log "stage2: budget guard (rem $(rem_min)m) — stop gap-fill, move on"; break 2; }
done; done
log "stage2: gap-fill done=${N2_DONE} skip=${N2_SKIP} fail=${N2_FAIL}"

# ---- stage 3: half-life HL0 (review CONFIRMED-CLEAN 2026-07-12) ----
HL0_RC="not-run"
REM=$(rem_min)
if [ "$REM" -ge 60 ]; then
  gpu_idle_wait || exit 3
  log "stage3: launching run_halflife_hl0.sh BUDGET_MIN=${REM}"
  BUDGET_MIN="$REM" ./run_halflife_hl0.sh >> engine/run_halflife_hl0.nohup.log 2>&1
  HL0_RC=$?
  log "stage3: HL0 rc=${HL0_RC}"
else
  log "stage3: SKIP HL0 (remaining ${REM}m < 60m)"
fi

# ---- stage 4: dossier refresh + report ----
DOSS="$($PY experiments/revision_dossier.py --results_dir results --out results/REVISION_DOSSIER.json 2>&1 | grep '^\[revision_dossier\] .*stable=' | tail -1)"
log "stage4: dossier: ${DOSS:-no summary}"
{
  echo "CHAIN_6H REPORT $(date '+%F %T') (window ${WINDOW_MIN}m, remaining $(rem_min)m)"
  echo "merging: ${MERGING_RC_NOTE:-unknown}"
  echo "gradsim gap-fill: done=${N2_DONE} skip=${N2_SKIP} fail=${N2_FAIL}"
  echo "HL0 rc: ${HL0_RC}  (verdict table: results/halflife/HL0_killgate_table.json if produced)"
  echo "M0 verdict table: results/merging/M0_killgate_table.json (if produced)"
  echo "dossier: ${DOSS:-n/a}"
} > "$REPORT"
log "================ CHAIN_6H COMPLETE ================"
echo "CHAIN_6H_DONE" >> "$LOG"
