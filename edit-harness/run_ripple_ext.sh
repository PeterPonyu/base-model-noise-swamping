#!/bin/bash
# run_ripple_ext.sh — P3 of the 2026-07-09 enhancement round (4090 box): ripple-axis
# layer completion. Coverage today is L12-only (rome s0-2 + alpha s0, popular split;
# the cloud-wave "ripple L8/L10/L14" plan never materialized as rows — run_ripple.sh
# on the box validated-skipped its L12 set). memory/ripple-geometry-result-20260707.md:
# rho=0.27 (3-seed) "needs more layers". This driver adds the depth profile:
#   rome popular L8/L10/L14 x s0/1/2  (SCIENCE, 9 rows, est 25m each)
#   alpha popular L12 s1/s2           (FILLER, completes the causal contrast's seeds)
# Rows are IDENTICAL in protocol to run_ripple.sh's (same CLI, same COMMON) — only the
# layer/seed grid differs, so existing + new rows pool into one analysis.
set -u
H="$(cd "$(dirname "$0")" && pwd)"
cd "$H" || exit 1
PY=/home/zeyufu/miniconda3/envs/dl/bin/python3
PY="${CLOUD_PY:-$PY}"
LOG=engine/run_ripple_ext.log
BUDGET_MIN=${BUDGET_MIN:-300}
mkdir -p engine results/smoke_ripple
echo $$ > engine/run_ripple_ext.pid
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
log "================ RUN_RIPPLE_EXT START (pid $$, budget ${BUDGET_MIN}m) ================"

# ---------------------------------------------------------------- Phase 0a: CPU pre-flight
pf_fail=0
pf(){ if eval "$2"; then log "preflight OK: $1"; else log "PREFLIGHT-FAIL: $1"; pf_fail=1; fi; }
pf "python env" "$PY -c 'import torch, numpy, scipy' 2>/dev/null"
pf "ripple_geometry.py" "[ -f experiments/ripple_geometry.py ]"
pf "rippleedits_loader.py" "[ -f experiments/rippleedits_loader.py ]"
pf "data/rippleedits/popular.json" "[ -f data/rippleedits/popular.json ]"
pf "model Llama-3.2-1B" "[ -d data/models/Llama-3.2-1B ]"
pf "disk >=10GB free" "[ \$(df --output=avail -BG . | tail -1 | tr -dc 0-9) -ge 10 ]"
rm -f engine/smoke_ripple_ext_*.ok
if [ "$pf_fail" -ne 0 ]; then log "ABORT: preflight failed"; exit 3; fi

# ---------------------------------------------------------------- Phase 0b: GPU idle gate
DRYRUN=${DRYRUN:-0}
if [ "$DRYRUN" -eq 1 ]; then
  log "DRYRUN=1 — skipping GPU idle gate, printing every run_row call without executing"
else
  source "$H/cloud/gpu_idle_lib.sh"
  idle_gate_wait || { log "ABORT: idle gate failed"; exit 2; }
fi
T0=$(date +%s)

# ---------------------------------------------------------------- helpers (run_ripple.sh, verbatim CLI)
ENVP="env -u ALL_PROXY -u all_proxy HF_HUB_OFFLINE=1"
RG="experiments/ripple_geometry.py"
POP="--data data/rippleedits/popular.json"
COMMON="--n_edits 200 --n_unrelated_probes 200 --max_probes_per_criterion 40 --steps 20"
SMK="--n_edits 3 --n_unrelated_probes 4 --max_probes_per_criterion 3 --steps 3"
elapsed_min(){ echo $(( ( $(date +%s) - T0 ) / 60 )); }

validate(){
  $PY - "$1" <<'EOF'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    assert isinstance(d, dict) and d, "empty json"
    esr = d.get("edit_success_rate")
    if esr is None:
        print("VALIDATE-OK")
    else:
        print("VALIDATE-OK" if esr >= 0.9 else f"VALIDATE-WARN esr={esr}<0.9")
except Exception as e:
    print(f"VALIDATE-FAIL {e}")
EOF
}

wedge_fail=0; MAXWEDGE=2
n_done=0; n_fail=0; n_skip=0
QUEUE_ABORT=0

run_row(){
  local class="$1" tag="$2" est="$3" needs="$4"; shift 4; local cmd="$*"
  if [ "$DRYRUN" -eq 1 ]; then
    echo "DRYRUN ${tag} [${class}] est=${est}m needs=${needs} cmd: ${cmd}"
    log "DRYRUN ${tag} [${class}] est=${est}m needs=${needs} cmd: ${cmd}"
    return
  fi
  local now; now=$(elapsed_min)
  if [ $(( now + est + 2 )) -gt "$BUDGET_MIN" ]; then
    log "BUDGET-SKIP ${tag} (elapsed ${now}m + est ${est}m > ${BUDGET_MIN}m)"; n_skip=$((n_skip+1)); return; fi
  if [ "$needs" != "-" ]; then
    local mk; for mk in ${needs//,/ }; do
      if [ ! -f "$mk" ]; then log "CONFIG-SKIP ${tag} (missing gate marker ${mk})"; n_skip=$((n_skip+1)); return; fi
    done
  fi
  local outj; outj=$(echo "$cmd" | grep -oE -- '--out [^ ]+' | awk '{print $2}')
  if [ -n "$outj" ] && [ -f "$outj" ]; then
    local v; v=$(validate "$outj")
    if echo "$v" | grep -q VALIDATE-FAIL; then
      mv "$outj" "$outj.INVALID" 2>/dev/null
      log "STALE-INVALID ${tag} — quarantined; re-running"
    else
      log "skip ${tag} (exists, validated)"
      [ "$class" = "SMOKE" ] && : > "engine/smoke_ripple_ext_${tag}.ok"
      return
    fi
  fi
  local cap=$(( est * 60 * 3 + 1200 ))
  log "RUN ${tag} [${class}] (est ${est}m, cap ${cap}s, elapsed ${now}m) -> engine/ripple_ext_${tag}.log"
  local t rc dt
  t=$(date +%s)
  timeout --signal=TERM --kill-after=60 "${cap}s" bash -c "$cmd" >> "engine/ripple_ext_${tag}.log" 2>&1
  rc=$?; dt=$(( $(date +%s) - t ))
  if [ "$rc" -eq 0 ] && [ -n "$outj" ] && [ -f "$outj" ]; then
    local v; v=$(validate "$outj")
    if echo "$v" | grep -q VALIDATE-FAIL; then
      mv "$outj" "$outj.INVALID" 2>/dev/null
      log "FAIL ${tag} (${dt}s) OUTPUT-INVALID: ${v}"; n_fail=$((n_fail+1))
    else
      log "done ${tag} (${dt}s) ${v}"; n_done=$((n_done+1)); wedge_fail=0
      [ "$class" = "SMOKE" ] && : > "engine/smoke_ripple_ext_${tag}.ok"
    fi
  else
    if [ "$rc" -eq 124 ] || [ "$dt" -ge $(( est * 60 / 2 )) ]; then
      wedge_fail=$((wedge_fail+1)); n_fail=$((n_fail+1))
      log "FAIL ${tag} (rc ${rc}, ${dt}s) WEDGE-LIKE consec=${wedge_fail}/${MAXWEDGE}"
      [ "$wedge_fail" -ge "$MAXWEDGE" ] && { log "ABORT: wedge-like failures"; QUEUE_ABORT=1; }
    else
      n_fail=$((n_fail+1))
      log "FAIL ${tag} (rc ${rc}, ${dt}s) FAST/CONFIG — not counted toward wedge abort"
    fi
  fi
}
heartbeat(){ log "PROGRESS jobs=${n_done}done/${n_fail}fail/${n_skip}skip elapsed=$(elapsed_min)m budget_left=$(( BUDGET_MIN - $(elapsed_min) ))m"; }

# ---------------------------------------------------------------- rows
run_row SMOKE rome_ripple_ext_smoke 4 - "$ENVP $PY $RG --model data/models/Llama-3.2-1B --editor rome $POP $SMK --lr 0.1 --layer 8 --seed 0 --out results/smoke_ripple/rome_ripple_ext.json"
heartbeat
for L in 8 10 14; do
  for s in 0 1 2; do
    [ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE ripple_llama1b_rome_popular_L${L}_s${s} 25 engine/smoke_ripple_ext_rome_ripple_ext_smoke.ok "$ENVP $PY $RG --model data/models/Llama-3.2-1B --editor rome $POP $COMMON --lr 0.1 --layer ${L} --seed ${s} --out results/ripple_llama1b_rome_popular_L${L}_s${s}.json"
  done
  heartbeat
done
for s in 1 2; do
  [ "$QUEUE_ABORT" -eq 0 ] && run_row FILLER ripple_llama1b_alpha_popular_L12_s${s} 30 engine/smoke_ripple_ext_rome_ripple_ext_smoke.ok "$ENVP $PY $RG --model data/models/Llama-3.2-1B --editor alpha $POP $COMMON --lr 0.1 --layer 12 --seed ${s} --out results/ripple_llama1b_alpha_popular_L12_s${s}.json"
done
heartbeat
log "================ RUN_RIPPLE_EXT COMPLETE (${n_done} done / ${n_fail} fail / ${n_skip} skip) ================"
echo "RUN_RIPPLE_EXT_DONE"
