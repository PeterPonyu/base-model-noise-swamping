#!/usr/bin/env bash
# Paper B local curve chain (2026-07-31): refill -> B2 gemma -> B1 qwen3b + B3 phi35.
# gemma runs with the default on-GPU snapshot (fits 24GB); qwen3b/phi35 run with
# SNAPSHOT_DEVICE=cpu (host-RAM snapshot — fp32 loading unchanged, prereg-bound).
# The G-S3 readout inside the driver returns INCOMPLETE (rc 3 -> driver rc 11) until all
# three families are on disk; only the FINAL stage's readout can write PAPERB_CURVE_GS3_PASS.ok.
set -u
H="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$H" || exit 2
LOG=engine/chain_paperb_local.log
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

wait_pidfile_gone(){  # $1=pidfile $2=timeout_s $3=label
  local t0; t0=$(date +%s)
  while [ -f "$1" ]; do
    [ $(( $(date +%s) - t0 )) -le "$2" ] || { log "ABORT: $3 still present after $2s"; exit 9; }
    sleep 60
  done
}

wait_pidfile_gone engine/run_mixab_refill.pid 10800 "MIX_A/B refill"
tail -2 engine/run_mixab_refill.log | tee -a "$LOG"

log "stage 1: B2 gemma-2-2b L19 (default snapshot)"
GRID="gemma2b:data/models/gemma-2-2b:19" setsid bash run_paperb_curve_local.sh >> engine/paperb_curve_gemma.nohup.log 2>&1
rc=$?
log "stage 1 rc=$rc (11 = readout INCOMPLETE, expected at this stage)"
[ "$rc" -eq 0 ] || [ "$rc" -eq 11 ] || { log "ABORT: gemma stage failed rc=$rc"; exit "$rc"; }

wait_pidfile_gone engine/run_paperb_curve_local.pid 60 "stale curve pidfile"
log "stage 2: B1 qwen3b L27 + B3 phi35 L24 (SNAPSHOT_DEVICE=cpu)"
GRID="qwen3b:data/models/Qwen2.5-3B:27 phi35:data/models/Phi-3.5-mini:24" \
  SNAPSHOT_DEVICE=cpu setsid bash run_paperb_curve_local.sh >> engine/paperb_curve_qwen_phi.nohup.log 2>&1
rc=$?
log "stage 2 rc=$rc"
tail -3 engine/run_paperb_curve_local.log | tee -a "$LOG"
exit "$rc"
