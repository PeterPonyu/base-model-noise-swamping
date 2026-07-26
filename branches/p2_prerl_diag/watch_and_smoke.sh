#!/bin/bash
# watch_and_smoke.sh — wait (by PID) for the P2 resample driver to drain, then run
# the deferred GRPO GPU smoke in isolation and refresh the §9 usability verdict.
#
# Non-science, no full wave: this ONLY proves run_grpo.py's train->callback->
# merge->save path end-to-end with --smoke-steps 2 (2 optimization steps, tiny),
# writing to grpo_out_smoke/ which the science pipeline never reads.
#
# Rules honored (workspace burned-in): wait by PID kill -0 (NEVER pgrep/pkill a
# pattern — watcher cmdlines self-match); GPU-idle gate on util+mem (NOT zero
# compute-apps — a persistent CUDA context never clears).
set -u
H=/home/zeyufu/Desktop/idea-feasibility-analysis
B=$H/branches/p2_prerl_diag
cd "$H" || exit 1
RESAMPLE_PID=${RESAMPLE_PID:-318667}
LOG=$B/watch_and_smoke.log
echo $$ > $B/watch_and_smoke.pid
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
log "================ WATCH_AND_SMOKE START (pid $$, waiting on resample pid $RESAMPLE_PID) ================"

# 1. wait for the resample driver to exit (by PID; identity-agnostic — we only
#    need it GONE, and we never signal it)
while kill -0 "$RESAMPLE_PID" 2>/dev/null; do sleep 120; done
log "resample pid $RESAMPLE_PID has exited — proceeding to GPU idle gate"
# let any final CUDA teardown settle
sleep 30

# 2. GPU idle gate (util<25 && mem<4000, x3 consecutive, 60s apart; 60min ceiling)
gate_t0=$(date +%s); consec=0
while [ "$consec" -lt 3 ]; do
  line=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  util=$(echo "$line" | awk -F, '{gsub(/[^0-9]/,"",$1); print $1}')
  mem=$(echo "$line"  | awk -F, '{gsub(/[^0-9]/,"",$2); print $2}')
  if [ -n "$util" ] && [ -n "$mem" ] && [ "$util" -lt 25 ] && [ "$mem" -lt 4000 ]; then
    consec=$((consec+1))
  else
    consec=0
    if [ $(( $(date +%s) - gate_t0 )) -gt 3600 ]; then log "ABORT: GPU busy >60min at gate"; exit 2; fi
  fi
  log "gpu poll util=${util:-NA} mem=${mem:-NA} consec=${consec}/3"
  [ "$consec" -lt 3 ] && sleep 60
done
log "GPU idle — running GRPO smoke (Qwen2.5-0.5B, --smoke-steps 2)"

# 3. the smoke (cap 30min; a 2-step run on the 0.5B is minutes, cap is a backstop)
t=$(date +%s)
timeout --signal=TERM --kill-after=60 1800s \
  env -u ALL_PROXY -u all_proxy conda run -n dl-rl python3 -u $B/run_grpo.py \
    --checkpoint Qwen2.5-0.5B --smoke-steps 2 >> "$B/watch_and_smoke_run.log" 2>&1
rc=$?
dt=$(( $(date +%s) - t ))
smoke_status=$(python3 -c "
import json, os
p='$B/grpo_out_smoke/Qwen2.5-0.5B/train_status.json'
print(json.load(open(p)).get('status','absent') if os.path.exists(p) else 'absent')
" 2>/dev/null || echo absent)
merged_ok=no
[ -f "$B/grpo_out_smoke/Qwen2.5-0.5B/merged/config.json" ] && merged_ok=yes
log "smoke: rc=$rc status=$smoke_status merged=$merged_ok (${dt}s)"

# 4. refresh the §9 usability verdict from the now-complete resample results
log "refreshing usability verdict (compute_overthinking_gap.py --usability-only)"
conda run -n dl python3 $B/compute_overthinking_gap.py --usability-only \
  > "$B/results/.USABILITY_POST_RESAMPLE.json.tmp" 2>>"$LOG" \
  && mv "$B/results/.USABILITY_POST_RESAMPLE.json.tmp" "$B/results/USABILITY_POST_RESAMPLE.json" \
  || { rm -f "$B/results/.USABILITY_POST_RESAMPLE.json.tmp"; log "usability refresh FAILED"; }
n_usable=$(python3 -c "
import json, os
p='$B/results/USABILITY_POST_RESAMPLE.json'
print(json.load(open(p)).get('n_usable','?') if os.path.exists(p) else '?')
" 2>/dev/null || echo '?')

# 5. one-line verdict banner for the morning glance
{
  echo "SMOKE  rc=$rc  status=$smoke_status  merged=$merged_ok  (${dt}s)"
  echo "PANEL  n_usable=$n_usable  (>=6 => n=6 confirmatory; <6 => descriptive-only or more resampling)"
  echo "next   if smoke=completed+merged=yes: launch run_p2_grpo.sh (FORCE_DESCRIPTIVE=1 iff n_usable<6)"
} > "$B/SMOKE_REPORT.txt"
log "wrote SMOKE_REPORT.txt"
log "================ WATCH_AND_SMOKE COMPLETE ================"
