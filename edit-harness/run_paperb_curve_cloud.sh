#!/usr/bin/env bash
# Paper B conditional B4: Llama-3.1-8B fp32 on Pro-6000 96GB.
set -u
H=${H:-/root/edit-harness}; cd "$H" || exit 1
PY=${CLOUD_PY:-python3}; GPU_ID=${GPU_ID:-0}; WAVE_BOX=${WAVE_BOX:-}; DRYRUN=${DRYRUN:-0}
BUDGET_MIN=${BUDGET_MIN:-300}; JOB_CAP_MIN=${JOB_CAP_MIN:-150}
PREREG=${PREREG:-$H/docs/plans/PREREG-PAPERB-CURVE-2026-07-26.md}
[ -n "$WAVE_BOX" ] && [ "$WAVE_BOX" = "$(hostname)" ] || { echo "ABORT: WAVE_BOX mismatch" >&2; exit 6; }
[ -f "$PREREG" ] && grep -qx 'STATUS: RATIFIED' "$PREREG" || { echo "ABORT: prereg not ratified" >&2; exit 5; }
READY="$H/engine/BOX_READY_paperb-curve.ok"
[ -f "$READY" ] || { echo "ABORT: run box_prepare_wave.sh paperb-curve check first" >&2; exit 8; }
expected_sha=$(sha256sum "$0" | cut -d' ' -f1)
prepare_sha=$(sha256sum "$H/engine/box_prepare_wave.sh" | cut -d' ' -f1)
grep -qx "driver_sha256=$expected_sha" "$READY" || { echo "ABORT: stale BOX_READY receipt for a different driver hash" >&2; exit 8; }
grep -qx "prepare_sha256=$prepare_sha" "$READY" || { echo "ABORT: stale BOX_READY receipt for a different prepare hash" >&2; exit 8; }
[ -f engine/PAPERB_CURVE_GS3_PASS.ok ] || { echo "ABORT: missing G-S3 PASS receipt" >&2; exit 7; }
MODEL=data/models/Llama-3.1-8B; [ -d "$MODEL" ] || { echo "ABORT: missing $MODEL" >&2; exit 3; }
mkdir -p engine results/quant_survival_curve_cloud results/smoke_paperb_curve_cloud
PIDFILE=engine/run_paperb_curve_cloud.pid; LOG=engine/run_paperb_curve_cloud.log
echo "$BASHPID" > "$PIDFILE"; trap 'rm -f "$PIDFILE"' EXIT
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
Q=experiments/quant_survival_phase1.py; DATA=data/counterfact.json
if [ "$DRYRUN" = 1 ]; then log "DRYRUN Llama-3.1-8B L24 ROME seeds 0,1,2 fp32"; exit 0; fi
CUDA_VISIBLE_DEVICES="" "$PY" "$Q" --selftest >engine/paperb_curve_cloud_selftest.log 2>&1 && grep -q 'ALL CHECKS PASSED' engine/paperb_curve_cloud_selftest.log || exit 4
consec=0; gate_start=$(date +%s)
while [ "$consec" -lt 3 ]; do
 line=$(nvidia-smi -i "$GPU_ID" --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null|head -1); util=$(printf '%s' "$line"|cut -d, -f1|tr -dc 0-9); mem=$(printf '%s' "$line"|cut -d, -f2|tr -dc 0-9)
 if [ -n "$util" ] && [ -n "$mem" ] && [ "$util" -lt 25 ] && [ "$mem" -lt 1500 ]; then consec=$((consec+1)); else consec=0; fi
 [ $(( $(date +%s)-gate_start )) -le 1800 ] || exit 8; [ "$consec" -eq 3 ] || sleep 30
done
T0=$(date +%s); failures=0; completed=0; expected=3
SMOKE_DIR=results/smoke_paperb_curve_cloud/llama8b; SMOKE_TABLE="$SMOKE_DIR/QS_phase1_table.json"
if [ ! -f "$SMOKE_TABLE" ]; then
 mkdir -p "$SMOKE_DIR"
 CUDA_VISIBLE_DEVICES="$GPU_ID" timeout 3000 "$PY" "$Q" --run --model "$MODEL" --data "$DATA" --editor rome --n_edits 3 --n_probes 8 --layer 24 --seed 0 --steps 2 --lr 0.1 --schemes nf4dq,int8 --codec real --fullmodel_cache off --n_perm 20 --n_boot 20 --gen_check_n 0 --device cuda --out_dir "$SMOKE_DIR" --table_out "$SMOKE_TABLE" >>engine/paperb_curve_cloud_smoke.log 2>&1 || { log "ABORT model smoke"; exit 10; }
fi
validate(){ "$PY" - "$1" "$2" <<'PY'
import json,sys,numpy as np
d=json.load(open(sys.argv[1])); a=np.load(sys.argv[2],allow_pickle=True); s=d.get('runner_stamp') or {}
need={'code_sha256','pid','hostname','wall_start','wall_end','elapsed_s','nvidia_smi_sample'}
assert not (need-set(s)); assert json.loads(str(a['runner_stamp_json'].item()))['code_sha256']==s['code_sha256']
assert a['COS'].shape==(200,200) and d['model'].endswith('Llama-3.1-8B') and d['editor']=='rome'
PY
}
for seed in 0 1 2; do
 dir="results/quant_survival_curve_cloud/llama8b_rome_L24_s${seed}"; table="$dir/QS_phase1_table.json"; raw="$dir/QS_phase1_raw.npz"; mkdir -p "$dir"
 if [ -f "$table" ] && [ -f "$raw" ] && validate "$table" "$raw" >/dev/null 2>&1; then log "SKIP s$seed"; completed=$((completed+1)); continue; fi
 elapsed=$(( ($(date +%s)-T0)/60 )); [ $((elapsed+70)) -le "$BUDGET_MIN" ] || { log "BUDGET-STOP"; break; }
 CUDA_VISIBLE_DEVICES="$GPU_ID" timeout --signal=TERM --kill-after=60 "$((JOB_CAP_MIN*60))s" "$PY" "$Q" --run --model "$MODEL" --data "$DATA" --editor rome --n_edits 200 --n_probes 200 --layer 24 --seed "$seed" --steps 20 --lr 0.1 --schemes nf4dq,int8 --codec real --fullmodel_cache off --n_perm 1000 --n_boot 1000 --device cuda --out_dir "$dir" --table_out "$table" >>"engine/paperb_curve_llama8b_s${seed}.log" 2>&1; rc=$?
 if [ "$rc" -eq 0 ] && validate "$table" "$raw" >/dev/null 2>&1; then log "DONE s$seed"; completed=$((completed+1)); else log "FAIL s$seed rc=$rc"; failures=$((failures+1)); fi
 [ "$failures" -lt 2 ] || exit 9
done
{
 echo "PAPERB CURVE CLOUD REPORT host=$(hostname) failures=$failures completed=$completed expected=$expected"
 grep -E 'DONE |FAIL |SKIP |BUDGET|ABORT|COMPLETE' "$LOG" | tail -80
} > engine/run_paperb_curve_cloud.report
log "COMPLETE failures=$failures completed=$completed/$expected"
[ "$failures" -eq 0 ] && [ "$completed" -eq "$expected" ]
