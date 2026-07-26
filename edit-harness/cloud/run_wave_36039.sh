#!/bin/bash
# cloud/run_wave_36039.sh — 7-9B family wave on the 36039 dual-4090D box (2026-07-13).
#
# SCIENCE (Phase-A recipe from PLAN-PRO6000-ZOO-UTILIZATION, retargeted to the <=8B tier
# that now lives on this box): per family model —
#   gate band: ROME killgate at 4 relative-depth layers x seeds {0,1,2}   (12 cells)
#   causal:    AlphaEdit HOLDOUT-projector at the 2 middle band layers, s0 (2 cells)
# Layer bands are relative-depth-matched to the Llama-1B L8/10/12/14-of-16 convention
# (50 / 62.5 / 75 / 87.5 % of num_hidden_layers, floor-rounded).
#
# GATES per model: (a) model dir present; (b) its smoke json PASSes (esr>0.9) —
# house rule: no battery before the 2-min smoke; (c) config num_hidden_layers matches
# the hardcoded band (else CONFIG-skip, never guess layers).
#
# SHARDING: driver-level, one worker per card (run_cloud_wave.sh pattern; no shared
# --out paths between cards). card0: mistral7b, qwen3_8b, gemma9b(late). card1:
# qwen25_7b, llama31_8bi.
#
# OPS: idempotent (skip if out json exists), per-job timeout JOB_CAP_MIN (default 100,
# the Lane-B lesson), per-card budget BUDGET_MIN (default 600), stop-file
# engine/STOP_WAVE36039, pidfiles engine/wave36039_card{0,1}.pid, wait by PID only.
set -u
H="$(cd "$(dirname "$0")/.." && pwd)"; cd "$H" || exit 1
PY=${CLOUD_PY:-/root/miniconda3/bin/python}
M=${MODELS_DIR:-/root/autodl-tmp/models}
DATA="$H/data/counterfact.json"
SMK="$H/results/smoke36039"
RES="$H/results"
ENG="$H/engine"
JOB_CAP_MIN=${JOB_CAP_MIN:-100}
BUDGET_MIN=${BUDGET_MIN:-900}   # per card; card0 carries 3 models (gemma lands late) — review sizing note
STOP="$ENG/STOP_WAVE36039"
mkdir -p "$RES" "$ENG" cloud/logs

log(){ echo "[wave36039 $(date '+%F %T')] $*" | tee -a "$ENG/wave36039.log"; }

# tag -> "dir n_layers band(4 layers space-sep)"
spec(){
  case "$1" in
    mistral7b)   echo "$M/Mistral-7B-v0.3 32 16 20 24 28" ;;
    qwen25_7b)   echo "$M/Qwen2.5-7B 28 14 17 21 24" ;;
    llama31_8bi) echo "$M/Llama-3.1-8B-Instruct 32 16 20 24 28" ;;
    qwen3_8b)    echo "$M/Qwen3-8B-Base 36 18 22 27 31" ;;
    gemma9b)     echo "$M/gemma-2-9b-bf16 42 21 26 31 36" ;;
    *) echo ""; return 1 ;;
  esac
}

model_gate(){ # tag dir n_layers -> 0 ok / 1 skip
  local tag=$1 dir=$2 nl=$3
  [ -d "$dir" ] || { log "CONFIG-skip $tag: model dir absent"; return 1; }
  local smoke="$SMK/smoke_${tag}.json"
  # smoke json name for gemma9b is smoke_gemma9b_bf16.json (smoke driver tag)
  [ "$tag" = gemma9b ] && smoke="$SMK/smoke_gemma9b_bf16.json"
  if ! "$PY" - "$smoke" <<'EOF'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(1)
esr = d.get("esr", d.get("edit_success_rate", 0)) or 0
sys.exit(0 if esr > 0.9 else 1)
EOF
  then log "CONFIG-skip $tag: smoke gate not passed ($smoke)"; return 1; fi
  local cfg_nl
  cfg_nl=$("$PY" - "$dir/config.json" <<'EOF'
import json, sys
print(json.load(open(sys.argv[1])).get("num_hidden_layers", -1))
EOF
) || cfg_nl=-1
  if [ "$cfg_nl" != "$nl" ]; then
    log "CONFIG-skip $tag: num_hidden_layers=$cfg_nl != expected $nl (band would be wrong)"
    return 1
  fi
  return 0
}

run_cell(){ # card tag dir editor layer seed extra_flags... -> writes out json
  local card=$1 tag=$2 dir=$3 editor=$4 layer=$5 seed=$6; shift 6
  local out
  if [ "$editor" = alpha ]; then
    out="$RES/g4_${tag}_alphaHO_cf_L${layer}_s${seed}.json"
  else
    out="$RES/gate_${tag}_${editor}_cf_L${layer}_s${seed}.json"
  fi
  # skip requires BOTH json and npz: the gate analysis reads the npz (analysis-plan
  # catch 2026-07-13 — cells without --save_matrices are unusable for the within-probe gate)
  local npz="$RES/matrices/$(basename "$out" .json).npz"
  [ -f "$out" ] && [ -f "$npz" ] && { log "skip (exists): $(basename "$out")"; return 0; }
  [ -f "$STOP" ] && { log "STOP-file present — not starting $(basename "$out")"; return 9; }
  log "card$card RUN $(basename "$out")"
  CUDA_VISIBLE_DEVICES=$card timeout "$((JOB_CAP_MIN * 60))" "$PY" experiments/killgate_keygeom.py \
    --model "$dir" --data "$DATA" --dataset counterfact \
    --n_edits 200 --n_probes 500 --steps 20 --lr 0.1 \
    --editor "$editor" --layer "$layer" --seed "$seed" --model_dtype bf16 \
    --save_matrices "$@" --out "$out" > "cloud/logs/wave_${tag}_${editor}_L${layer}_s${seed}.log" 2>&1
  local rc=$?
  [ $rc -ne 0 ] && log "card$card FAIL rc=$rc $(basename "$out")"
  return $rc
}

card_worker(){ # card tag...
  local card=$1; shift
  # $BASHPID, not $$: this runs in a backgrounded subshell and $$ would record the
  # parent driver's PID in BOTH pidfiles (review M2).
  echo "$BASHPID" > "$ENG/wave36039_card${card}.pid"
  local t0 fails=0
  t0=$(date +%s)
  for tag in "$@"; do
    read -r dir nl b1 b2 b3 b4 <<< "$(spec "$tag")"
    [ -z "${dir:-}" ] && continue
    model_gate "$tag" "$dir" "$nl" || continue
    for layer in $b1 $b2 $b3 $b4; do
      for seed in 0 1 2; do
        if [ $(( ($(date +%s) - t0) / 60 )) -ge "$BUDGET_MIN" ]; then
          log "card$card BUDGET_MIN=$BUDGET_MIN reached — stopping"; return 0
        fi
        run_cell "$card" "$tag" "$dir" rome "$layer" "$seed"
        rc=$?
        [ $rc -eq 9 ] && return 0
        if [ $rc -ne 0 ]; then
          fails=$((fails + 1))
          if [ $fails -ge 2 ]; then log "card$card 2 consecutive fails — stopping (wedge?)"; return 1; fi
        else
          fails=0
        fi
      done
    done
    for layer in $b2 $b3; do
      [ -f "$STOP" ] && return 0
      if [ $(( ($(date +%s) - t0) / 60 )) -ge "$BUDGET_MIN" ]; then
        log "card$card BUDGET_MIN=$BUDGET_MIN reached before alpha cells — stopping"; return 0
      fi
      run_cell "$card" "$tag" "$dir" alpha "$layer" 0 --alpha_proj_source holdout
      rc=$?
      # rc=9 = STOP-file sentinel from run_cell (timeout emits 124/137 and killgate
      # exits 0/1, so 9 cannot come from a real job today)
      [ $rc -eq 9 ] && return 0
      if [ $rc -ne 0 ]; then
        fails=$((fails + 1))
        if [ $fails -ge 2 ]; then log "card$card 2 consecutive fails in alpha — stopping (wedge?)"; return 1; fi
      else
        fails=0
      fi
    done
  done
  return 0
}

echo $$ > "$ENG/wave36039_main.pid"
# clear stale terminal markers from earlier attempts (a dead run's PARTIAL.err would
# otherwise make every watcher instantly misread the fresh run — burned 2026-07-13)
rm -f "$ENG/WAVE36039_DONE.ok" "$ENG/WAVE36039_PARTIAL.err"
log "wave start: JOB_CAP_MIN=$JOB_CAP_MIN BUDGET_MIN=$BUDGET_MIN"
card_worker 0 mistral7b qwen3_8b gemma9b > "cloud/logs/wave36039_card0.log" 2>&1 &
W0=$!
card_worker 1 qwen25_7b llama31_8bi > "cloud/logs/wave36039_card1.log" 2>&1 &
W1=$!
log "workers: card0 pid $W0, card1 pid $W1"
wait $W0; RC0=$?
wait $W1; RC1=$?
log "wave done: card0 rc=$RC0 card1 rc=$RC1"
n_done=$(ls "$RES"/gate_*_cf_L*_s*.json "$RES"/g4_*_alphaHO_cf_L*_s0.json 2>/dev/null | wc -l)
log "result jsons on disk: $n_done"
if [ $RC0 -eq 0 ] && [ $RC1 -eq 0 ]; then touch "$ENG/WAVE36039_DONE.ok"; else touch "$ENG/WAVE36039_PARTIAL.err"; fi
