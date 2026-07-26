#!/bin/bash
# cloud/run_wave_pro6000.sh — Tier-A "generational core" wave on box 29246
# (RTX Pro 6000 Blackwell 96GB, single card; 2026-07-13, user-approved scope ¥120-160).
#
# SCIENCE:
#   P0 smokes (house rule, this box): qwen25_14b, qwen3_14b, qwen3_32b, llama31_8b.
#   P1 (two lanes, 2x~35G <= 96G): Qwen2.5-14B band L{24,30,36,42} x s{0,1} ROME
#      + alphaHO L36 s0  ||  Qwen3-14B band L{20,25,30,35} x s{0,1} + alphaHO L30 s0.
#      -> the matched-dim (5120) generational pair, B1 of the zoo plan.
#   P2 (solo, ~75G): Qwen3-32B mini-rung L{32,40,48,56} x s0 ROME + alphaHO L48 s0.
#   P3 (two lanes, ~42G+~40G): Llama-3.1-8B fp32-vs-bf16 twin killgate at L16 s0
#      (the precision ground-truth cell)  ||  merging RG at 7B fp32
#      (Mistral-7B L24=75%-depth, seeds 0,1,2, groups 2,3,5,10,20).
#   Bands are relative-depth-matched (floor(nl x {.5,.625,.75,.875})); alphaHO uses
#   --alpha_proj_source holdout (E6 circularity rule). 14B/32B cells run bf16;
#   the fp32 twin and merging (fp32-only code) are the deliberate exceptions.
#
# OPS: single GPU — concurrency is by VRAM-budgeted phases, NOT CUDA_VISIBLE_DEVICES.
# Idempotent, JOB_CAP_MIN (default 150; 32B cells ~90-110 min), BUDGET_MIN (default
# 1300) checked before EVERY cell, stop-file engine/STOP_WAVEPRO, $BASHPID pidfiles,
# 2-consecutive-fail stop per lane, markers engine/WAVEPRO_{DONE.ok,PARTIAL.err}.
set -u
H="$(cd "$(dirname "$0")/.." && pwd)"; cd "$H" || exit 1
PY=${CLOUD_PY:-/root/miniconda3/bin/python}
M=${MODELS_DIR:-/root/autodl-tmp/models}
DATA="$H/data/counterfact.json"
RES="$H/results"
SMK="$RES/smokepro"
ENG="$H/engine"
JOB_CAP_MIN=${JOB_CAP_MIN:-150}
BUDGET_MIN=${BUDGET_MIN:-1300}
STOP="$ENG/STOP_WAVEPRO"
mkdir -p "$RES" "$SMK" "$ENG" cloud/logs
T0=$(date +%s)

log(){ echo "[wavepro $(date '+%F %T')] $*" | tee -a "$ENG/wavepro.log"; }
over_budget(){ [ $(( ($(date +%s) - T0) / 60 )) -ge "$BUDGET_MIN" ]; }

spec(){ # tag -> "dir n_layers band(4)"
  case "$1" in
    qwen25_14b)  echo "$M/Qwen2.5-14B 48 24 30 36 42" ;;
    qwen3_14b)   echo "$M/Qwen3-14B-Base 40 20 25 30 35" ;;
    qwen3_32b)   echo "$M/Qwen3-32B 64 32 40 48 56" ;;
    llama31_8b)  echo "$M/Llama-3.1-8B 32 16 20 24 28" ;;
    *) echo ""; return 1 ;;
  esac
}

smoke(){ # tag dir -> writes $SMK/smoke_<tag>.json; rc 0 = PASS
  local tag=$1 dir=$2 out="$SMK/smoke_${tag}.json"
  if [ ! -f "$out" ]; then
    log "smoke RUN $tag"
    timeout 1800 "$PY" experiments/killgate_keygeom.py \
      --model "$dir" --data "$DATA" --dataset counterfact \
      --n_edits 24 --n_probes 60 --steps 20 --editor rome --layer auto \
      --seed 0 --model_dtype bf16 --out "$out" > "cloud/logs/smokepro_${tag}.log" 2>&1 \
      || { log "smoke FAIL(run) $tag"; return 1; }
  fi
  "$PY" - "$out" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
esr = d.get("esr", d.get("edit_success_rate", 0)) or 0
sys.exit(0 if esr > 0.9 else 1)
EOF
}

model_gate(){ # tag dir n_layers
  local tag=$1 dir=$2 nl=$3
  [ -d "$dir" ] || { log "CONFIG-skip $tag: dir absent"; return 1; }
  smoke "$tag" "$dir" || { log "CONFIG-skip $tag: smoke gate failed"; return 1; }
  local cfg_nl
  cfg_nl=$("$PY" - "$dir/config.json" <<'EOF'
import json, sys
print(json.load(open(sys.argv[1])).get("num_hidden_layers", -1))
EOF
) || cfg_nl=-1
  [ "$cfg_nl" = "$nl" ] || { log "CONFIG-skip $tag: num_hidden_layers=$cfg_nl != $nl"; return 1; }
  return 0
}

run_cell(){ # tag dir editor layer seed dtype out extra... ; rc 9 = STOP sentinel
  local tag=$1 dir=$2 editor=$3 layer=$4 seed=$5 dtype=$6 out=$7; shift 7
  # skip requires BOTH json and npz (analysis-plan catch 2026-07-13: gate analysis
  # reads the npz; JSON-only cells are unusable for the within-probe gate)
  local npz="$RES/matrices/$(basename "$out" .json).npz"
  [ -f "$out" ] && [ -f "$npz" ] && { log "skip (exists): $(basename "$out")"; return 0; }
  [ -f "$STOP" ] && { log "STOP-file — not starting $(basename "$out")"; return 9; }
  over_budget && { log "BUDGET_MIN=$BUDGET_MIN reached — not starting $(basename "$out")"; return 9; }
  log "RUN $(basename "$out")"
  timeout "$((JOB_CAP_MIN * 60))" "$PY" experiments/killgate_keygeom.py \
    --model "$dir" --data "$DATA" --dataset counterfact \
    --n_edits 200 --n_probes 500 --steps 20 --lr 0.1 \
    --editor "$editor" --layer "$layer" --seed "$seed" --model_dtype "$dtype" \
    --save_matrices "$@" --out "$out" > "cloud/logs/wavepro_$(basename "$out" .json).log" 2>&1
  local rc=$?
  [ $rc -ne 0 ] && log "FAIL rc=$rc $(basename "$out")"
  return $rc
}

band_worker(){ # lane tag seeds(csv) [rome_seed_mode: all|s0] — 4-layer band + alphaHO
  local lane=$1 tag=$2 seeds=$3
  echo "$BASHPID" > "$ENG/wavepro_lane${lane}.pid"
  local dir nl b1 b2 b3 b4 fails=0 rc
  read -r dir nl b1 b2 b3 b4 <<< "$(spec "$tag")"
  [ -z "${dir:-}" ] && return 0
  # review M-a: a gate-skip must not let the wave end as DONE — flag it for the summary
  model_gate "$tag" "$dir" "$nl" || { touch "$ENG/wavepro_skipped_${tag}.flag"; return 0; }
  for layer in $b1 $b2 $b3 $b4; do
    for seed in ${seeds//,/ }; do
      run_cell "$tag" "$dir" rome "$layer" "$seed" bf16 \
        "$RES/gate_${tag}_rome_cf_L${layer}_s${seed}.json"
      rc=$?
      [ $rc -eq 9 ] && return 0
      if [ $rc -ne 0 ]; then
        fails=$((fails + 1)); [ $fails -ge 2 ] && { log "lane$lane 2-fails — stop"; return 1; }
      else fails=0; fi
    done
  done
  # alphaHO causal at the 3rd band layer, s0
  run_cell "$tag" "$dir" alpha "$b3" 0 bf16 \
    "$RES/g4_${tag}_alphaHO_cf_L${b3}_s0.json" --alpha_proj_source holdout
  rc=$?
  [ $rc -eq 9 ] && return 0
  [ $rc -ne 0 ] && { log "lane$lane alpha FAIL rc=$rc ($tag)"; return 1; }
  return 0
}

twin8b_worker(){ # fp32-vs-bf16 precision ground-truth on Llama-3.1-8B L16 s0
  echo "$BASHPID" > "$ENG/wavepro_lane0.pid"
  local dir nl b1 b2 b3 b4 rc
  read -r dir nl b1 b2 b3 b4 <<< "$(spec llama31_8b)"
  model_gate llama31_8b "$dir" "$nl" || { touch "$ENG/wavepro_skipped_llama31_8b.flag"; return 0; }
  run_cell llama31_8b "$dir" rome 16 0 fp32 "$RES/gate_llama31_8b_rome_cf_L16_s0_fp32.json"
  rc=$?; [ $rc -eq 9 ] && return 0; [ $rc -ne 0 ] && return 1
  run_cell llama31_8b "$dir" rome 16 0 bf16 "$RES/gate_llama31_8b_rome_cf_L16_s0_bf16.json"
  rc=$?; [ $rc -eq 9 ] && return 0; [ $rc -ne 0 ] && return 1
  return 0
}

merging_worker(){ # merging RG at 7B fp32 (merging_m0 is fp32-only by design)
  echo "$BASHPID" > "$ENG/wavepro_lane1.pid"
  local out_dir="$RES/merging/Mistral-7B-v0.3_L24_RG"
  # explicit --table_out (2026-07-14 fix, D2 width-series hazard finding): without it,
  # merging_m0.py --rg defaults to $out_dir_top/RG_operating_curve_table.json — the SAME
  # top-level path the canonical Llama-3.2-1B L12 gate owns (see docs/plans/
  # PREREG-D2-WIDTH-RG-20260714.md). Tag+layer-namespaced, same convention as
  # run_merging_width.sh, so a future restart of this wave can never clobber that table.
  local table_out="$RES/merging/RG_operating_curve_table_mistral7b_L24.json"
  [ -d "$out_dir" ] && { log "skip (exists): merging RG dir"; return 0; }
  [ -f "$STOP" ] && return 0
  over_budget && { log "BUDGET_MIN reached — skipping merging"; return 0; }
  log "RUN merging RG Mistral-7B L24 (fp32, seeds 0,1,2, g=2..20)"
  timeout $((JOB_CAP_MIN * 60 * 2)) "$PY" experiments/merging_m0.py --rg \
    --model "$M/Mistral-7B-v0.3" --data "$DATA" --layer 24 \
    --rg_seeds 0,1,2 --rg_group_sizes 2,3,5,10,20 \
    --table_out "$table_out" \
    > cloud/logs/wavepro_merging_rg.log 2>&1
  local rc=$?
  [ $rc -ne 0 ] && { log "merging FAIL rc=$rc"; return 1; }
  return 0
}

echo $$ > "$ENG/wavepro_main.pid"
rm -f "$ENG"/wavepro_skipped_*.flag "$ENG/WAVEPRO_DONE.ok" "$ENG/WAVEPRO_PARTIAL.err"
log "wave start: JOB_CAP_MIN=$JOB_CAP_MIN BUDGET_MIN=$BUDGET_MIN (Tier A)"

log "=== P1: 14B generational pair (two lanes) ==="
band_worker 0 qwen25_14b 0,1 > cloud/logs/wavepro_lane0_p1.log 2>&1 & L0=$!
band_worker 1 qwen3_14b  0,1 > cloud/logs/wavepro_lane1_p1.log 2>&1 & L1=$!
wait $L0; R0=$?
wait $L1; R1=$?
log "P1 done: lane0 rc=$R0 lane1 rc=$R1"

log "=== P2: Qwen3-32B mini-rung (solo) ==="
band_worker 0 qwen3_32b 0 > cloud/logs/wavepro_p2.log 2>&1 & L2=$!
wait $L2; R2=$?
log "P2 done: rc=$R2"

log "=== P3: 8B precision twin || merging RG ==="
twin8b_worker   > cloud/logs/wavepro_lane0_p3.log 2>&1 & L3=$!
merging_worker  > cloud/logs/wavepro_lane1_p3.log 2>&1 & L4=$!
wait $L3; R3=$?
wait $L4; R4=$?
log "P3 done: twin rc=$R3 merging rc=$R4"

n_done=$(ls "$RES"/gate_qwen*_cf_L*.json "$RES"/gate_llama31_8b_rome_cf_L16_s0_*.json "$RES"/g4_*_alphaHO_cf_L*_s0.json 2>/dev/null | wc -l)
log "wave done: cells on disk=$n_done rcs=$R0/$R1/$R2/$R3/$R4"
n_skipped=$(ls "$ENG"/wavepro_skipped_*.flag 2>/dev/null | wc -l)
if [ $((R0 + R1 + R2 + R3 + R4)) -eq 0 ] && [ "$n_skipped" -eq 0 ]; then
  touch "$ENG/WAVEPRO_DONE.ok"
else
  [ "$n_skipped" -gt 0 ] && log "PARTIAL: $n_skipped model(s) gate-skipped"
  touch "$ENG/WAVEPRO_PARTIAL.err"
fi
