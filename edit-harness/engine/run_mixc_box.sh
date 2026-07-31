#!/usr/bin/env bash
# run_mixc_box.sh — Frame-A MIX_C (33 cells) + 3 quarantined MIX_A s2 cells, ON-BOX,
# dual-card shard (H14, PLAN-GAP-CLOSURE-MASTER-2026-07-31).
#
#   card0: MIX_C policies both,cost_only,damage_only,oracle,always_edit,always_grace (18 cells)
#          + MIX_A rerun cost_only (its s2 cell was quarantined 2026-07-26)
#   card1: MIX_C policies always_rag,always_ft,always_reject,random,ft_merge (15 cells)
#          + MIX_A rerun ft_merge,random (2 quarantined s2 cells)
#
# run_stream is idempotent per-cell (skip-on-exists), so MIX_A reruns fill ONLY the
# quarantined s2 gaps. setsid + TERM trap per the 07-29 4x-SIGTERM incident (I18):
# a wrapper/monitor SIGTERM must leave the python cell alive.
# GATE RULE (binding): no analysis/number may regenerate from these cells until
# experiments/frame_a/provenance_gate_v2.py PASSES on them and writes
# engine/FRAME_A_GATE_V2_PASS.ok. This driver runs cells ONLY — it never writes that receipt.
set -u
H="${H:-/root/edit-harness-deploy-20260727}"; cd "$H" || exit 2
PY="${PY:-/root/autodl-tmp/venvs/ifa-20260727/bin/python}"
SHARD="${SHARD:-}"; GPU_ID="${GPU_ID:-}"
[ "$SHARD" = card0 ] || [ "$SHARD" = card1 ] || { echo "ABORT: SHARD must be card0 or card1" >&2; exit 2; }
[ -n "$GPU_ID" ] || { echo "ABORT: GPU_ID is required" >&2; exit 2; }
PIDFILE="engine/run_mixc_box_${SHARD}.pid"
LOG="engine/run_mixc_box_${SHARD}.log"
mkdir -p engine results/frame_a/cells
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
[ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE" 2>/dev/null)" 2>/dev/null \
  && { echo "REFUSE: already running (pid $(cat "$PIDFILE"))" >&2; exit 7; }
echo $$ > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT

case "$SHARD" in
  card0) MIXC_POLICIES="both,cost_only,damage_only,oracle,always_edit,always_grace"
         MIXA_POLICIES="cost_only" ;;
  card1) MIXC_POLICIES="always_rag,always_ft,always_reject,random,ft_merge"
         MIXA_POLICIES="ft_merge,random" ;;
esac

# ---- preflight: model + tokenizer collision gate (07-30 regression lock, I22) ----
[ -d data/models/Llama-3.2-1B ] || { log "ABORT: missing data/models/Llama-3.2-1B"; exit 3; }
$PY experiments/selftest_target_token.py --tokenizer data/models/Llama-3.2-1B >/dev/null 2>&1 \
  || { log "ABORT: tokenizer gate FAIL on Llama-3.2-1B"; exit 3; }
# ---- per-card idle gate (never zero-compute-apps; util<25 && mem<1500 x3) ----
consec=0; tries=0
while [ "$consec" -lt 3 ]; do
  [ "$tries" -ge 60 ] && { log "ABORT: card $GPU_ID never idle after 30min"; exit 9; }
  tries=$((tries+1))
  line=$(nvidia-smi -i "$GPU_ID" --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  util=$(echo "$line" | awk -F, '{gsub(/[^0-9]/,"",$1); print $1}')
  mem=$(echo "$line"  | awk -F, '{gsub(/[^0-9]/,"",$2); print $2}')
  if [ -n "${util:-}" ] && [ -n "${mem:-}" ] && [ "$util" -lt 25 ] && [ "$mem" -lt 1500 ]; then
    consec=$((consec+1)); else consec=0; fi
  [ "$consec" -lt 3 ] && sleep 30
done
log "card $GPU_ID idle — $SHARD window opens (MIX_C policies: $MIXC_POLICIES)"

ENVP="env -u ALL_PROXY -u all_proxy HF_HUB_OFFLINE=1"
run_cell(){  # TAG CMD...  — setsid, survives wrapper SIGTERM, rc from wait
  local tag="$1"; shift
  log "RUN $tag"
  setsid env CUDA_VISIBLE_DEVICES="$GPU_ID" $ENVP "$@" >> "$LOG" 2>&1 &
  local child=$!
  trap 'log "WRAPPER TERM/INT — setsid cell pid '"$child"' stays alive; relaunch resumes"; exit 143' TERM INT
  wait "$child"; local rc=$?
  trap - TERM INT
  [ "$rc" -eq 0 ] && log "DONE $tag" || { log "FAIL $tag rc=$rc"; return "$rc"; }
}

rc_all=0
# ---- MIX_C (shard policies) — smoke gate is enforced inside run_stream (M4) ----
run_cell "mixc_${SHARD}" $PY -m experiments.frame_a.run_stream --run --real \
  --mixes MIX_C --policies "$MIXC_POLICIES" --model_dir data/models/Llama-3.2-1B || rc_all=$?
# ---- MIX_A quarantined s2 refills (skip-on-exists => only the 3 gaps regenerate) ----
if [ "$rc_all" -eq 0 ]; then
  run_cell "mixa_refill_${SHARD}" $PY -m experiments.frame_a.run_stream --run --real \
    --mixes MIX_A --policies "$MIXA_POLICIES" --model_dir data/models/Llama-3.2-1B || rc_all=$?
fi

# ---- verification: count cells (never gate numbers on this — gate v2 decides) ----
mixc=$(ls results/frame_a/cells/cell_llama-3.2-1b_real_MIX_C_*.json 2>/dev/null | wc -l)
mixa=$(ls results/frame_a/cells/cell_llama-3.2-1b_real_MIX_A_*.json 2>/dev/null | wc -l)
log "cell counts: MIX_C=$mixc/33 MIX_A=$mixa/33 (analysis BLOCKED until FRAME_A_GATE_V2_PASS.ok)"
{
  echo "MIXC BOX REPORT shard=$SHARD host=$(hostname) rc=$rc_all MIX_C=$mixc/33 MIX_A=$mixa/33"
  grep -E 'RUN |DONE |FAIL |ABORT' "$LOG" | tail -40
} > "engine/run_mixc_box_${SHARD}.report"
log "======== MIXC BOX $SHARD END rc=$rc_all ========"
exit "$rc_all"
