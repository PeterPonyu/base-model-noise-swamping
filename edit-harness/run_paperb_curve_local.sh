#!/usr/bin/env bash
# Paper B local B1-B3 curve extension, RTX 5090, fp32 only.
set -u
cd "$(dirname "$0")" || exit 1
PY=${PY:-/home/zeyufu/miniconda3/envs/dl/bin/python3}; GPU_ID=${GPU_ID:-0}; DRYRUN=${DRYRUN:-0}
BUDGET_MIN=${BUDGET_MIN:-600}; JOB_CAP_MIN=${JOB_CAP_MIN:-120}
PREREG=${PREREG:-../docs/plans/PREREG-PAPERB-CURVE-2026-07-26.md}
[ -f "$PREREG" ] && grep -qx 'STATUS: RATIFIED' "$PREREG" || { echo "ABORT: Paper B curve prereg not ratified" >&2; exit 5; }
mkdir -p engine results/quant_survival_curve results/smoke_paperb_curve
PIDFILE=engine/run_paperb_curve_local.pid; LOG=engine/run_paperb_curve_local.log
echo "$BASHPID" > "$PIDFILE"; trap 'rm -f "$PIDFILE"' EXIT
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
Q=experiments/quant_survival_phase1.py; DATA=data/counterfact.json
GRID="qwen3b:data/models/Qwen2.5-3B:27 gemma2b:data/models/gemma-2-2b:19 phi35:data/models/Phi-3.5-mini:24"
for f in "$Q" "$DATA"; do [ -f "$f" ] || exit 3; done
for spec in $GRID; do rest=${spec#*:}; model=${rest%:*}; [ -d "$model" ] || { log "ABORT missing $model"; exit 3; }; done
if [ "$DRYRUN" = 1 ]; then log "DRYRUN $GRID"; exit 0; fi
CUDA_VISIBLE_DEVICES="" "$PY" "$Q" --selftest >engine/paperb_curve_selftest.log 2>&1 && grep -q 'ALL CHECKS PASSED' engine/paperb_curve_selftest.log || exit 4
consec=0; gate_start=$(date +%s)
while [ "$consec" -lt 3 ]; do
 line=$(nvidia-smi -i "$GPU_ID" --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null|head -1); util=$(printf '%s' "$line"|cut -d, -f1|tr -dc 0-9); mem=$(printf '%s' "$line"|cut -d, -f2|tr -dc 0-9)
 if [ -n "$util" ] && [ -n "$mem" ] && [ "$util" -lt 25 ] && [ "$mem" -lt 1500 ]; then consec=$((consec+1)); else consec=0; fi
 [ $(( $(date +%s)-gate_start )) -le 1800 ] || exit 8; [ "$consec" -eq 3 ] || sleep 30
done
T0=$(date +%s); failures=0
for spec in $GRID; do
  tag=${spec%%:*}; rest=${spec#*:}; model=${rest%:*}; layer=${spec##*:}
  smoke_dir="results/smoke_paperb_curve/${tag}"; smoke_table="$smoke_dir/QS_phase1_table.json"
  if [ ! -f "$smoke_table" ]; then
    mkdir -p "$smoke_dir"
    CUDA_VISIBLE_DEVICES="$GPU_ID" timeout 2400 "$PY" "$Q" --run --model "$model" --data "$DATA" --editor rome --n_edits 3 --n_probes 8 --layer "$layer" --seed 0 --steps 2 --lr 0.1 --schemes nf4dq,int8 --codec real --fullmodel_cache off --n_perm 20 --n_boot 20 --gen_check_n 0 --device cuda --out_dir "$smoke_dir" --table_out "$smoke_table" >>"engine/paperb_curve_smoke_${tag}.log" 2>&1 || { log "ABORT smoke $tag"; exit 10; }
  fi
done
validate(){ "$PY" - "$1" "$2" <<'PY'
import json,sys,numpy as np
d=json.load(open(sys.argv[1])); a=np.load(sys.argv[2],allow_pickle=True); s=d.get('runner_stamp') or {}
need={'code_sha256','pid','hostname','wall_start','wall_end','elapsed_s','nvidia_smi_sample'}
assert not (need-set(s)); assert json.loads(str(a['runner_stamp_json'].item()))['code_sha256']==s['code_sha256']
assert a['COS'].shape==(200,200) and d['editor']=='rome' and d['codec']=='real'
PY
}
for spec in $GRID; do tag=${spec%%:*}; rest=${spec#*:}; model=${rest%:*}; layer=${spec##*:}
 for seed in 0 1 2; do
  dir="results/quant_survival_curve/${tag}_rome_L${layer}_s${seed}"; table="$dir/QS_phase1_table.json"; raw="$dir/QS_phase1_raw.npz"; mkdir -p "$dir"
  if [ -f "$table" ] && [ -f "$raw" ] && validate "$table" "$raw" >/dev/null 2>&1; then log "SKIP $tag s$seed"; continue; fi
  elapsed=$(( ($(date +%s)-T0)/60 )); [ $((elapsed+60)) -le "$BUDGET_MIN" ] || { log "BUDGET-STOP"; break 2; }
  CUDA_VISIBLE_DEVICES="$GPU_ID" timeout --signal=TERM --kill-after=60 "$((JOB_CAP_MIN*60))s" "$PY" "$Q" --run --model "$model" --data "$DATA" --editor rome --n_edits 200 --n_probes 200 --layer "$layer" --seed "$seed" --steps 20 --lr 0.1 --schemes nf4dq,int8 --codec real --fullmodel_cache auto --n_perm 1000 --n_boot 1000 --device cuda --out_dir "$dir" --table_out "$table" >>"engine/paperb_curve_${tag}_s${seed}.log" 2>&1; rc=$?
  if [ "$rc" -eq 0 ] && validate "$table" "$raw" >/dev/null 2>&1; then log "DONE $tag s$seed"; else log "FAIL $tag s$seed rc=$rc"; failures=$((failures+1)); fi
  [ "$failures" -lt 2 ] || exit 9
 done
done
"$PY" experiments/paperb_curve_readout.py >>"$LOG" 2>&1
readout_rc=$?
[ "$readout_rc" -eq 0 ] && log "G-S3 PASS receipt written" || log "G-S3 not passed (rc=$readout_rc); B4 remains locked"
log "COMPLETE failures=$failures readout_rc=$readout_rc"
[ "$failures" -eq 0 ] || exit 9
[ "$readout_rc" -eq 3 ] && exit 11
exit 0
