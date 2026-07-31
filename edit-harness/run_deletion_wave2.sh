#!/usr/bin/env bash
# Pro-6000 deletion Wave 2, conditional on Wave-1 G-D3.
set -u
H=${H:-/root/edit-harness}; cd "$H" || exit 1
PY=${CLOUD_PY:-python3}; GPU_ID=${GPU_ID:-0}; WAVE_BOX=${WAVE_BOX:-}; DRYRUN=${DRYRUN:-0}
BUDGET_MIN=${BUDGET_MIN:-1260}; JOB_CAP_MIN=${JOB_CAP_MIN:-150}
PREREG=${PREREG:-$H/docs/plans/PREREG-DELETION-PREDICTOR-2026-07-26.md}
[ -n "$WAVE_BOX" ] && [ "$WAVE_BOX" = "$(hostname)" ] || { echo "ABORT: WAVE_BOX mismatch" >&2; exit 6; }
[ -f "$PREREG" ] && grep -qx 'STATUS: RATIFIED' "$PREREG" || { echo "ABORT: prereg not ratified" >&2; exit 5; }
READY="$H/engine/BOX_READY_deletion-wave2.ok"
[ -f "$READY" ] || { echo "ABORT: run box_prepare_wave.sh deletion-wave2 check first" >&2; exit 8; }
expected_sha=$(sha256sum "$0" | cut -d' ' -f1)
prepare_sha=$(sha256sum "$H/engine/box_prepare_wave.sh" | cut -d' ' -f1)
grep -qx "driver_sha256=$expected_sha" "$READY" || { echo "ABORT: stale BOX_READY receipt for a different driver hash" >&2; exit 8; }
grep -qx "prepare_sha256=$prepare_sha" "$READY" || { echo "ABORT: stale BOX_READY receipt for a different prepare hash" >&2; exit 8; }
[ -f engine/DELETION_WAVE1_GD3_PASS.ok ] || { echo "ABORT: missing engine/DELETION_WAVE1_GD3_PASS.ok" >&2; exit 7; }
mkdir -p engine results/matrices results/smoke_deletion_wave2
PIDFILE=engine/run_deletion_wave2.pid; LOG=engine/run_deletion_wave2.log
echo "$BASHPID" > "$PIDFILE"; trap 'rm -f "$PIDFILE"' EXIT
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
KG=experiments/killgate_keygeom.py; DATA=data/counterfact.json
COMMON="--dataset counterfact --data $DATA --n_edits 200 --n_probes 500 --steps 20 --lr 0.1 --model_dtype fp32 --save_matrices --matrix_dir results/matrices"
GRID="llama13b:data/models/Llama-2-13b-hf:30 qwen14b:data/models/Qwen2.5-14B:36"
for f in "$KG" "$DATA"; do [ -f "$f" ] || { log "ABORT missing $f"; exit 3; }; done
for spec in $GRID; do rest=${spec#*:}; model=${rest%:*}; [ -d "$model" ] || { log "ABORT missing model $model"; exit 3; }; done
if [ "$DRYRUN" = 1 ]; then log "DRYRUN gpu=$GPU_ID grid=$GRID"; exit 0; fi
consec=0; gate_start=$(date +%s)
while [ "$consec" -lt 3 ]; do
 line=$(nvidia-smi -i "$GPU_ID" --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
 util=$(printf '%s' "$line"|cut -d, -f1|tr -dc 0-9); mem=$(printf '%s' "$line"|cut -d, -f2|tr -dc 0-9)
 if [ -n "$util" ] && [ -n "$mem" ] && [ "$util" -lt 25 ] && [ "$mem" -lt 1500 ]; then consec=$((consec+1)); else consec=0; fi
 [ $(( $(date +%s)-gate_start )) -le 1800 ] || { log "ABORT GPU busy >30m"; exit 8; }; [ "$consec" -eq 3 ] || sleep 30
done
T0=$(date +%s); failures=0; completed=0; expected=13
validate(){ "$PY" - "$1" "$2" <<'PY'
import json,sys,numpy as np
d=json.load(open(sys.argv[1])); a=np.load(sys.argv[2],allow_pickle=True); s=d.get('runner_stamp') or {}
need={'code_sha256','pid','hostname','wall_start','wall_end','elapsed_s','nvidia_smi_sample'}
assert not (need-set(s)); assert 'runner_stamp_json' in a.files; assert a['COS'].shape==(200,500)
PY
}
run_cell(){ tag=$1; est=$2; shift 2; out="results/$tag.json"; npz="results/matrices/$tag.npz"
 if [ -f "$out" ] && [ -f "$npz" ] && validate "$out" "$npz" >/dev/null 2>&1; then log "SKIP $tag"; completed=$((completed+1)); return; fi
 elapsed=$(( ($(date +%s)-T0)/60 )); [ $((elapsed+est)) -le "$BUDGET_MIN" ] || { log "BUDGET-STOP $tag"; return; }
 CUDA_VISIBLE_DEVICES="$GPU_ID" timeout --signal=TERM --kill-after=60 "$((JOB_CAP_MIN*60))s" "$@" --out "$out" >>"engine/$tag.log" 2>&1; rc=$?
 if [ "$rc" -eq 0 ] && validate "$out" "$npz" >/dev/null 2>&1; then log "DONE $tag"; completed=$((completed+1)); else log "FAIL $tag rc=$rc"; failures=$((failures+1)); fi
 [ "$failures" -lt 2 ] || exit 9
}
for spec in $GRID; do tag=${spec%%:*}; rest=${spec#*:}; model=${rest%:*}; layer=${spec##*:}
 smoke="results/smoke_deletion_wave2/$tag.json"
 [ -f "$smoke" ] || CUDA_VISIBLE_DEVICES="$GPU_ID" timeout 2400 "$PY" "$KG" --model "$model" --editor rome --dataset counterfact --data "$DATA" --n_edits 3 --n_probes 8 --steps 2 --lr 0.1 --layer "$layer" --model_dtype fp32 --out "$smoke" >>"engine/smoke_$tag.log" 2>&1 || exit 10
 for seed in 0 1 2; do
  run_cell "gate_${tag}_rome_cf_L${layer}_s${seed}" 80 "$PY" "$KG" --model "$model" --editor rome $COMMON --layer "$layer" --seed "$seed"
  run_cell "u1e0_${tag}_delete_refusal_L${layer}_s${seed}" 80 "$PY" "$KG" --model "$model" --editor rome --edit_mode delete --delete_variant refusal $COMMON --layer "$layer" --seed "$seed"
 done
 if [ "$tag" = llama13b ]; then run_cell "u1e0_llama13b_alphaHO_delete_refusal_L30_s0" 100 "$PY" "$KG" --model "$model" --editor alpha --alpha_proj_source holdout --holdout_frac 1.0 --edit_mode delete --delete_variant refusal $COMMON --layer 30 --seed 0; fi
done
{
 echo "DELETION WAVE2 REPORT host=$(hostname) failures=$failures completed=$completed expected=$expected"
 grep -E 'RUN |DONE |FAIL |SKIP |BUDGET|ABORT|COMPLETE' "$LOG" | tail -120
} > engine/run_deletion_wave2.report
log "COMPLETE failures=$failures completed=$completed/$expected"
[ "$failures" -eq 0 ] && [ "$completed" -eq "$expected" ]
