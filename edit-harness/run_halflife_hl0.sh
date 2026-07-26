#!/usr/bin/env bash
# run_halflife_hl0.sh — Edit half-life HL0 kill-gate driver (~220 GPU-min, 1 seed).
#
# Skeleton verbatim from run_revins.sh / run_u6.sh: CPU preflight, GPU-idle gate
# (util<25 && mem<1500 MiB x3 consecutive polls), DRYRUN=1 plan mode, per-stage
# budget/timeout/idempotent run_stage with HALFLIFE-namespaced pid/log/markers.
# Chains experiments/halflife_hl0.py stages s1->s2->s3->s4->s5.
#
# GPU/SCIENCE stages (s1-s4) load Llama-3.2-1B + gsm8k (LOCAL HF cache only; NEVER
# downloads — HF_HUB_OFFLINE=1). Each stage CONFIG-skips cleanly if its model/dataset
# gate marker is missing. s5 is CPU-only (numpy on the stage npz). This driver is
# author-verified via `bash -n` + DRYRUN=1 only and is NOT launched by the author.
#
# Stage cost anchors (1B, L12, single seed; from the ROME-edit precedent in
# engine/*.log where a comparable cell exists, else conservative upper bounds):
#   s1 tau     ~15m  (2 arms x short gsm8k FT; full arm dominates)
#   s2 edits   ~70m  (200 ROME edits + 100 controls; ~ one killgate 200-edit pass)
#   s3 proxy   ~90m  (arms x 3 alpha x (200 edit forwards + 100 control forwards))
#   s4 realFT  ~40m  (40 edits x (ROME edit + short real gsm8k FT))
#   s5 analysis <2m  (CPU)
# Total ~217 GPU-min. Ordered s1->s2->s3->s4->s5 (hard data dependency chain).
#
# Usage:
#   DRYRUN=1 ./run_halflife_hl0.sh                 # print the plan; no GPU, no STAGE outputs
#     (DRYRUN is plan-only for the GPU stages, but it is NOT zero-disk: the CPU preflight
#      still runs `--selftest`, writing results/halflife/selftest/*.json, and the driver
#      writes its own bookkeeping engine/halflife_hl0.{pid,log} + engine/halflife_round_start.
#      What DRYRUN never writes: any tau_*/hl0_s*/HL0_killgate_table stage artifact.)
#   nohup ./run_halflife_hl0.sh >> engine/halflife_hl0.nohup.log 2>&1 &   # real run
#   BUDGET_MIN=120 ./run_halflife_hl0.sh           # tighter window (drops later stages)
set -u
H=/home/zeyufu/Desktop/idea-feasibility-analysis/edit-harness
cd "$H" || exit 1
PY=/home/zeyufu/miniconda3/envs/dl/bin/python3
HL=experiments/halflife_hl0.py
LOG=engine/halflife_hl0.log
BUDGET_MIN=${BUDGET_MIN:-300}
mkdir -p engine results/halflife results/halflife/tau results/halflife/selftest
echo $$ > engine/halflife_hl0.pid
[ -f engine/halflife_round_start ] || stat -c %Y engine/halflife_hl0.pid > engine/halflife_round_start
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
log "================ RUN_HALFLIFE_HL0 START (pid $$, budget ${BUDGET_MIN}m) ================"

# ---------------------------------------------------------------- Phase 0a: CPU preflight (HARD: code/tool/data presence only)
pf_fail=0
pf(){ if eval "$2"; then log "preflight OK: $1"; else log "PREFLIGHT-FAIL: $1"; pf_fail=1; fi; }
pf "python env (torch+numpy)" "$PY -c 'import torch, numpy' 2>/dev/null"
pf "halflife_hl0.py present" "[ -f $HL ]"
pf "halflife_hl0.py compiles" "$PY -m py_compile $HL 2>/dev/null"
pf "halflife_hl0 --selftest GREEN" "$PY $HL --selftest >/dev/null 2>&1"
pf "counterfact.json" "[ -f data/counterfact.json ]"
pf "rome_native editor" "[ -f editors/rome_native.py ]"
pf "metrics.py" "[ -f metrics.py ]"
if [ "$pf_fail" -ne 0 ]; then log "ABORT: preflight failed"; exit 3; fi

# ---------------------------------------------------------------- Phase 0a2: per-stage readiness gates (marker files)
DRYRUN=${DRYRUN:-0}
# gsm8k train arrow lives TWO dirs deep: openai___gsm8k/main/<version>/<hash>/gsm8k-train.arrow.
# A one-level glob (main/*/gsm8k-train.arrow) MISSES it and falsely reports the cache absent —
# use `find` (recursive). Presence check ONLY; never downloads.
GSM8K_DIR="$HOME/.cache/huggingface/datasets/openai___gsm8k/main"
GSM8K_ARROW=$(find "$GSM8K_DIR" -name gsm8k-train.arrow 2>/dev/null | head -1)
if [ "$DRYRUN" -ne 1 ]; then
  rm -f engine/halflife_model.ok engine/halflife_gsm8k.ok
  [ -d data/models/Llama-3.2-1B ] && { : > engine/halflife_model.ok; log "gate OK: model Llama-3.2-1B"; } \
    || log "MODEL-ABSENT: data/models/Llama-3.2-1B — s1..s4 CONFIG-skip"
  if [ -n "$GSM8K_ARROW" ]; then
    : > engine/halflife_gsm8k.ok; log "gate OK: gsm8k train arrow -> $GSM8K_ARROW"
  else
    # HARD gate (reviewer-requested loud line): without gsm8k there is no tau -> no s1/s3/s5
    # -> NO VERDICT, so the whole driver is pointless. Abort loudly; do NOT download.
    log "DATASET-ABSENT: gsm8k train arrow not found under ${GSM8K_DIR}/**/gsm8k-train.arrow"
    log "DATASET-ABSENT: gsm8k is required for tau (s1) and therefore for a verdict (s3/s5)."
    log "DATASET-ABSENT: ask the user before downloading (HF_HUB_OFFLINE=1 standing policy)."
    echo "DATASET-ABSENT: gsm8k train arrow not found under ${GSM8K_DIR}/**/gsm8k-train.arrow" >&2
    echo "DATASET-ABSENT: required for a verdict; ask the user before downloading (no auto-download)." >&2
    exit 4
  fi
fi
[ "$DRYRUN" -eq 1 ] && log "DRYRUN gsm8k probe: ${GSM8K_ARROW:-ABSENT} (no abort in plan mode)"

# ---------------------------------------------------------------- Phase 0b: GPU idle gate (util<25 && mem<1500 MiB x3)
if [ "$DRYRUN" -eq 1 ]; then
  log "DRYRUN=1 -- skipping GPU idle gate, printing every run_stage call without executing"
else
  gate_t0=$(date +%s); consec=0
  while [ "$consec" -lt 3 ]; do
    line=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    util=$(echo "$line" | awk -F, '{gsub(/[^0-9]/,"",$1); print $1}')
    mem=$(echo "$line"  | awk -F, '{gsub(/[^0-9]/,"",$2); print $2}')
    if [ -n "$util" ] && [ -n "$mem" ] && [ "$util" -lt 25 ] && [ "$mem" -lt 1500 ]; then
      consec=$((consec+1))
    else
      consec=0
      if [ $(( $(date +%s) - gate_t0 )) -gt 1800 ]; then log "ABORT: GPU busy >30min at gate"; exit 2; fi
    fi
    log "gpu poll util=${util:-NA} mem=${mem:-NA} consec=${consec}/3"
    [ "$consec" -lt 3 ] && sleep 30
  done
  log "GPU idle -- window opens now"
fi
T0=$(date +%s)
elapsed_min(){ echo $(( ( $(date +%s) - T0 ) / 60 )); }

# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True cuts allocator fragmentation for the s1
# full-arm FT (the OOM row) and the s3 full-arm delta apply; harmless for every other stage.
ENVP="env -u ALL_PROXY -u all_proxy HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
wedge_fail=0; MAXWEDGE=2
n_done=0; n_fail=0; n_skip=0; QUEUE_ABORT=0

# run_stage CLASS TAG EST_MIN NEEDS(comma-list or -) OUTCHECK(file or -) CMD...
# idempotent (skip if OUTCHECK exists), budget-gated, timeout-capped, wedge-aware —
# same contract as run_revins.sh run_row2 (stages own their exact output path).
run_stage(){
  local class="$1" tag="$2" est="$3" needs="$4" outcheck="$5"; shift 5; local cmd="$*"
  if [ "$DRYRUN" -eq 1 ]; then
    echo "DRYRUN ${tag} [${class}] est=${est}m needs=${needs} outcheck=${outcheck} cmd: ${cmd}"
    log "DRYRUN ${tag} [${class}] est=${est}m needs=${needs} outcheck=${outcheck} cmd: ${cmd}"
    return
  fi
  [ "$QUEUE_ABORT" -ne 0 ] && { log "QUEUE-ABORT active — skip ${tag}"; n_skip=$((n_skip+1)); return; }
  local now; now=$(elapsed_min)
  if [ $(( now + est + 2 )) -gt "$BUDGET_MIN" ]; then
    log "BUDGET-SKIP ${tag} (elapsed ${now}m + est ${est}m > ${BUDGET_MIN}m)"; n_skip=$((n_skip+1)); return; fi
  if [ "$needs" != "-" ]; then
    local mk; for mk in ${needs//,/ }; do
      if [ ! -f "$mk" ]; then log "CONFIG-SKIP ${tag} (missing gate marker ${mk})"; n_skip=$((n_skip+1)); return; fi
    done
  fi
  if [ "$outcheck" != "-" ] && [ -f "$outcheck" ]; then
    log "skip ${tag} (output exists: ${outcheck})"; return; fi
  local cap=$(( est * 60 * 3 + 900 ))
  log "RUN ${tag} [${class}] (est ${est}m, cap ${cap}s, elapsed ${now}m) -> engine/halflife_${tag}.log"
  local t rc dt
  t=$(date +%s)
  timeout --signal=TERM --kill-after=60 "${cap}s" bash -c "$cmd" >> "engine/halflife_${tag}.log" 2>&1
  rc=$?; dt=$(( $(date +%s) - t ))
  if [ "$rc" -eq 0 ] && { [ "$outcheck" = "-" ] || [ -f "$outcheck" ]; }; then
    log "done ${tag} (${dt}s)"; n_done=$((n_done+1)); wedge_fail=0
  else
    if [ "$rc" -eq 124 ] || [ "$dt" -ge $(( est * 60 / 2 )) ]; then
      wedge_fail=$((wedge_fail+1)); n_fail=$((n_fail+1))
      log "FAIL ${tag} (rc ${rc}, ${dt}s) WEDGE-LIKE consec=${wedge_fail}/${MAXWEDGE}"
      [ "$wedge_fail" -ge "$MAXWEDGE" ] && { log "ABORT: wedge-like failures"; QUEUE_ABORT=1; }
    else
      n_fail=$((n_fail+1))
      log "FAIL ${tag} (rc ${rc}, ${dt}s) FAST/CONFIG -- not counted toward wedge abort"
    fi
  fi
}
heartbeat(){ log "PROGRESS jobs=${n_done}done/${n_fail}fail/${n_skip}skip elapsed=$(elapsed_min)m budget_left=$(( BUDGET_MIN - $(elapsed_min) ))m"; }

MODEL="--model data/models/Llama-3.2-1B --data data/counterfact.json --device cuda --layer 12 --seed 0"
TAU_MLP="results/halflife/tau/tau_mlp.npz"
TAU_FULL="results/halflife/tau/tau_full.npz"
S2NPZ="results/halflife/hl0_s2_edits.npz"
S3NPZ="results/halflife/hl0_s3_proxy.npz"
S4NPZ="results/halflife/hl0_s4_realft.npz"
S5OUT="results/halflife/HL0_killgate_table.json"

# ---------------------------------------------------------------- Stage chain (hard s1->s2->s3->s4->s5 dependency)
# s1: tau (needs model + gsm8k). s2: edits (needs model only — no gsm8k). s3: proxy
# (needs s1 tau + s2 npz). s4: real-FT fidelity (needs model + gsm8k + s2 npz). s5: CPU.
# Idempotency notes:
#  * s1 outcheck = tau_FULL (the TERMINAL artifact — written last, and only after tau_full_state.pt);
#    run_s1 is per-arm idempotent internally, so if tau_mlp exists but tau_full does not, s1 re-runs
#    and computes ONLY the full arm. (Using tau_mlp here would wrongly skip s1 with the full arm missing.)
#  * s3 outcheck = "-" so run_s3 is ALWAYS invoked; it is INCREMENTAL (adds only arms missing from the
#    npz, preserving already-computed ones) and self-skips cheaply BEFORE loading the model if complete.
#  * s5 outcheck = "-" so the table is ALWAYS refreshed to reflect the current on-disk stages.
run_stage SCIENCE s1 15 "engine/halflife_model.ok,engine/halflife_gsm8k.ok" "$TAU_FULL" \
  "$ENVP $PY $HL --stage s1 $MODEL"
heartbeat
run_stage SCIENCE s2 70 "engine/halflife_model.ok" "$S2NPZ" \
  "$ENVP $PY $HL --stage s2 $MODEL"
heartbeat
run_stage SCIENCE s3 90 "engine/halflife_model.ok,${TAU_MLP},${S2NPZ}" "-" \
  "$ENVP $PY $HL --stage s3 $MODEL"
heartbeat
run_stage SCIENCE s4 40 "engine/halflife_model.ok,engine/halflife_gsm8k.ok,${S2NPZ}" "$S4NPZ" \
  "$ENVP $PY $HL --stage s4 $MODEL"
heartbeat
# s5 (CPU): analysis + verdict. needs s3 (s4 optional — s5 marks fidelity UNEVALUATED if absent).
run_stage CPU s5 2 "${S3NPZ}" "-" \
  "$ENVP $PY $HL --stage s5 --out $S5OUT"
heartbeat

# ---------------------------------------------------------------- report
{
  echo "RUN_HALFLIFE_HL0 REPORT $(date '+%F %T')  jobs: ${n_done} done / ${n_fail} fail / ${n_skip} skipped  elapsed $(elapsed_min)m/${BUDGET_MIN}m  abort=${QUEUE_ABORT}"
  [ -f "$S5OUT" ] && $PY -c "import json;d=json.load(open('$S5OUT'));print('VERDICT:',d.get('VERDICT'))" 2>/dev/null
  grep -E 'RUN |done |FAIL |SKIP|ABORT|PROGRESS|gate OK|MODEL-ABSENT|DATASET-ABSENT' "$LOG" | tail -60
} > engine/halflife_hl0_report.txt
log "================ RUN_HALFLIFE_HL0 COMPLETE (${n_done} done / ${n_fail} fail / ${n_skip} skip) ================"
echo "RUN_HALFLIFE_HL0_DONE" >> "$LOG"
