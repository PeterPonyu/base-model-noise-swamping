#!/bin/bash
# run_glue_bridge.sh — P1 of the 2026-07-09 enhancement round (4090 box).
# Template = run_glue_seq.sh (standalone-CLI driver, one json per cell), GB-namespaced.
# Calls experiments/glue_bridge.py: killgate restore-every-edit protocol with GLUE
# examples as probes — does edit-key<->task-prompt-key cosine predict per-example task
# damage (margin drop / flips)? npz lands in results/matrices in killgate matrix shape.
#
# GRID: rome L12 x s0/1/2 (core) + alpha-holdout L12 x s0/1/2 (causal contrast) +
# rome L8/L14 s0 (depth flanks, FILLER). est ~50m/row (200 edits x ~300 GLUE probes,
# unmeasured — flagged; the SMOKE row calibrates before the queue commits).
set -u
H="$(cd "$(dirname "$0")" && pwd)"
cd "$H" || exit 1
PY=/home/zeyufu/miniconda3/envs/dl/bin/python3
PY="${CLOUD_PY:-$PY}"
LOG=engine/run_glue_bridge.log
BUDGET_MIN=${BUDGET_MIN:-480}
mkdir -p engine results/smoke_glue_bridge results/glue_bridge results/matrices
echo $$ > engine/run_glue_bridge.pid
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
log "================ RUN_GLUE_BRIDGE START (pid $$, budget ${BUDGET_MIN}m) ================"

# ---------------------------------------------------------------- Phase 0a: CPU pre-flight
pf_fail=0
pf(){ if eval "$2"; then log "preflight OK: $1"; else log "PREFLIGHT-FAIL: $1"; pf_fail=1; fi; }
pf "python env" "$PY -c 'import torch, numpy, pandas, scipy' 2>/dev/null"
pf "glue_bridge.py" "[ -f experiments/glue_bridge.py ]"
pf "glue_downstream.py (templates reused)" "[ -f experiments/glue_downstream.py ]"
pf "counterfact.json" "[ -f data/counterfact.json ]"
pf "data/glue/sst2 validation parquet" "[ -f data/glue/sst2/validation-00000-of-00001.parquet ]"
pf "data/glue/mrpc validation parquet" "[ -f data/glue/mrpc/validation-00000-of-00001.parquet ]"
pf "data/glue/rte validation parquet" "[ -f data/glue/rte/validation-00000-of-00001.parquet ]"
pf "model Llama-3.2-1B" "[ -d data/models/Llama-3.2-1B ]"
pf "disk >=10GB free" "[ \$(df --output=avail -BG . | tail -1 | tr -dc 0-9) -ge 10 ]"
rm -f engine/smoke_glue_bridge_*.ok
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

# ---------------------------------------------------------------- helpers
ENVP="env -u ALL_PROXY -u all_proxy HF_HUB_OFFLINE=1"
GB="experiments/glue_bridge.py"
M1B="--model data/models/Llama-3.2-1B"
COMMON="--n_edits 200 --n_glue_samples 100 --steps 20 --save_matrices"
SMK="--n_edits 4 --n_glue_samples 8 --steps 2 --save_matrices --matrix_dir results/smoke_glue_bridge"
elapsed_min(){ echo $(( ( $(date +%s) - T0 ) / 60 )); }

validate(){   # $1=json  $2=required npz path ("" for smoke rows)
  $PY - "$1" "${2:-}" <<'EOF'
import json, os, sys
try:
    d = json.load(open(sys.argv[1]))
    assert "rho" in d and "edit_success_rate" in d, "missing rho/esr"
    npz = sys.argv[2]
    if npz and not os.path.exists(npz):
        print(f"VALIDATE-FAIL missing npz {npz}"); raise SystemExit
    esr = d["edit_success_rate"]
    warns = []
    if esr < 0.9:
        warns.append(f"esr={esr}<0.9")
    # review MEDIUM-3: if GLUE pre-accuracy sits at chance, the pre-correct column
    # filter keeps noise columns and the rho is uninterpretable — warn loudly.
    sst2 = (d.get("pre_accuracy") or {}).get("sst2")
    if sst2 is not None and sst2 <= 0.55:
        warns.append(f"PREACC-WARN sst2={sst2}<=0.55 (near chance — margin columns suspect)")
    print("VALIDATE-OK" if not warns else "VALIDATE-WARN " + " ".join(warns))
except SystemExit:
    pass
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
  local outj outn=""
  outj=$(echo "$cmd" | grep -oE -- '--out [^ ]+' | awk '{print $2}')
  if [ "$class" != "SMOKE" ]; then
    outn="results/matrices/$(basename "${outj%.json}").npz"   # review LOW-1: npz sidecar is part of idempotence
  fi
  if [ -n "$outj" ] && [ -f "$outj" ]; then
    local v; v=$(validate "$outj" "$outn")
    if echo "$v" | grep -q VALIDATE-FAIL; then
      mv "$outj" "$outj.INVALID" 2>/dev/null
      log "STALE-INVALID ${tag} — quarantined; re-running"
    else
      log "skip ${tag} (exists, validated)"
      [ "$class" = "SMOKE" ] && : > "engine/smoke_glue_bridge_${tag}.ok"
      return
    fi
  fi
  local mdir; mdir=$(echo "$cmd" | grep -oE -- '--model [^ ]+' | awk '{print $2}')
  case "$mdir" in data/*|/*)
    if [ -n "$mdir" ] && [ ! -d "$mdir" ]; then log "CONFIG-SKIP ${tag} (no model ${mdir})"; n_skip=$((n_skip+1)); return; fi;;
  esac
  local cap=$(( est * 60 * 3 + 1200 ))
  log "RUN ${tag} [${class}] (est ${est}m, cap ${cap}s, elapsed ${now}m) -> engine/glue_bridge_${tag}.log"
  local t rc dt
  t=$(date +%s)
  timeout --signal=TERM --kill-after=60 "${cap}s" bash -c "$cmd" >> "engine/glue_bridge_${tag}.log" 2>&1
  rc=$?; dt=$(( $(date +%s) - t ))
  if [ "$rc" -eq 0 ] && [ -n "$outj" ] && [ -f "$outj" ]; then
    local v; v=$(validate "$outj" "$outn")
    if echo "$v" | grep -q VALIDATE-FAIL; then
      mv "$outj" "$outj.INVALID" 2>/dev/null
      log "FAIL ${tag} (${dt}s) OUTPUT-INVALID: ${v}"; n_fail=$((n_fail+1))
    else
      log "done ${tag} (${dt}s) ${v}"; n_done=$((n_done+1)); wedge_fail=0
      [ "$class" = "SMOKE" ] && : > "engine/smoke_glue_bridge_${tag}.ok"
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
run_row SMOKE gb_smoke 4 - "$ENVP $PY $GB $M1B $SMK --lr 0.1 --layer 12 --seed 0 --out results/smoke_glue_bridge/gb_smoke.json"
heartbeat
for s in 0 1 2; do
  [ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE gb_llama1b_rome_L12_s${s} 50 engine/smoke_glue_bridge_gb_smoke.ok "$ENVP $PY $GB $M1B --editor rome $COMMON --lr 0.1 --layer 12 --seed ${s} --out results/glue_bridge/gb_llama1b_rome_L12_s${s}.json"
done
heartbeat
for s in 0 1 2; do
  [ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE gb_llama1b_alpha_L12_s${s} 55 engine/smoke_glue_bridge_gb_smoke.ok "$ENVP $PY $GB $M1B --editor alpha --alpha_proj_source holdout $COMMON --lr 0.1 --layer 12 --seed ${s} --out results/glue_bridge/gb_llama1b_alpha_L12_s${s}.json"
done
heartbeat
for L in 8 14; do
  [ "$QUEUE_ABORT" -eq 0 ] && run_row FILLER gb_llama1b_rome_L${L}_s0 50 engine/smoke_glue_bridge_gb_smoke.ok "$ENVP $PY $GB $M1B --editor rome $COMMON --lr 0.1 --layer ${L} --seed 0 --out results/glue_bridge/gb_llama1b_rome_L${L}_s0.json"
done
heartbeat
log "================ RUN_GLUE_BRIDGE COMPLETE (${n_done} done / ${n_fail} fail / ${n_skip} skip) ================"
echo "RUN_GLUE_BRIDGE_DONE"
