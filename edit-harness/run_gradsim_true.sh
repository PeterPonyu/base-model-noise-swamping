#!/bin/bash
# run_gradsim_true.sh — TRUE-backprop GradSim cell driver (2026-07-06). Template = run_u5/u6
# (verbatim skeleton: preflight, GPU-idle gate, run_row/validate/heartbeat), GRADSIM_TRUE-
# namespaced (own pid/log/markers). Standalone driver, NOT wired into the run_u* queue --
# the queue decision is deferred (matches the CF+ deliverable's own "queue decision comes
# later" scoping).
#
# Fills docs/findings/THEOREM-SXC-DRAFT-2026-07-06.md open issue #1 (section 9): the
# on-disk gradsim_baseline.py "resid" variant is the SAME closed-form computation as
# mechanism_sc_table.py's S x C (both read resid_norm/COS off the same npz fields), so it
# is not an independent validation. This driver runs a REAL backward pass instead:
#   Row 1 (SCIENCE, prereq): mechanism_dump.py --save_vectors at L12 s0 (llama1b, CF,
#     standard COMMON n_edits/steps) — persists the per-edit residual VECTOR r_e = v-Wk
#     (new, additive npz field), needed by gradsim_true.py's direct-influence shortcut.
#   Row 2 (SCIENCE): gradsim_true.py over Row 1's mechanism npz + the ALREADY-EXISTING
#     killgate gate npz (results/matrices/gate_llama1b_rome_cf_L12_s0.npz, confirmed present
#     — no probe-damage sweep is re-run, keeping this driver to ~30 GPU-min total rather than
#     re-paying killgate_keygeom.py's own ~30min COST for the same cell).
# A CPU micro-smoke (Row 0) exercises gradsim_true.py's SELF-CONTAINED mode (no pre-existing
# npz needed) on Qwen2.5-0.5B before the GPU rows run, gating Row 2 via a readiness marker.
#
# GUESSED CONVENTIONS (flagged per instructions):
#   - "llama1b" gate-npz model tag: this short tag is not programmatically derivable from
#     --model (it's a manual driver-naming convention throughout this harness, e.g.
#     mechanism_sc_table.py's TAG_RE) — hardcoded here to match the EXISTING file
#     results/matrices/gate_llama1b_rome_cf_L12_s0.npz (verified present at authoring time).
#   - est minutes: mechanism_dump (no probe sweep, 200 edits, ~10-15m by analogy to
#     killgate's own per-edit-loop share of its ~30m COMMON runtime); gradsim_true (500
#     probe backward passes, no edit loop in EXTERNAL mode) est ~15m — NEITHER measured on
#     GPU yet at this scale; treat as provisional, matching run_u6.sh's own disclaimer style.
set -u
H=/home/zeyufu/Desktop/idea-feasibility-analysis/edit-harness
cd "$H" || exit 1
PY=/home/zeyufu/miniconda3/envs/dl/bin/python3
LOG=engine/run_gradsim_true.log
BUDGET_MIN=${BUDGET_MIN:-60}
mkdir -p engine results/mechanism results/smoke_gradsim_true
echo $$ > engine/run_gradsim_true.pid
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
log "================ RUN_GRADSIM_TRUE START (pid $$, budget ${BUDGET_MIN}m) ================"

# ---------------------------------------------------------------- Phase 0a: CPU pre-flight
pf_fail=0
pf(){ if eval "$2"; then log "preflight OK: $1"; else log "PREFLIGHT-FAIL: $1"; pf_fail=1; fi; }
pf "python env" "$PY -c 'import torch, numpy' 2>/dev/null"
pf "counterfact.json" "[ -f data/counterfact.json ]"
pf "mechanism_dump.py --save_vectors flag" "grep -q -- '--save_vectors' experiments/mechanism_dump.py"
pf "gradsim_true.py present" "[ -f experiments/gradsim_true.py ]"
pf "model Llama-3.2-1B" "[ -d data/models/Llama-3.2-1B ]"
pf "model Qwen2.5-0.5B (smoke)" "[ -d data/models/Qwen2.5-0.5B ]"
pf "EXISTING gate npz L12 s0 (reused, not regenerated)" "[ -f results/matrices/gate_llama1b_rome_cf_L12_s0.npz ]"
pf "disk >=5GB free" "[ \$(df --output=avail -BG /home | tail -1 | tr -dc 0-9) -ge 5 ]"
rm -f engine/gradsim_true_smoke.ok
if [ "$pf_fail" -ne 0 ]; then log "ABORT: preflight failed"; exit 3; fi

# ---------------------------------------------------------------- Phase 0b: GPU idle gate
DRYRUN=${DRYRUN:-0}
if [ "$DRYRUN" -eq 1 ]; then
  log "DRYRUN=1 — skipping GPU idle gate, printing every run_row call without executing"
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
log "GPU idle — window opens now"
fi
T0=$(date +%s)

# ---------------------------------------------------------------- helpers (u5/u6 template, verbatim)
ENVP="env -u ALL_PROXY -u all_proxy HF_HUB_OFFLINE=1"
elapsed_min(){ echo $(( ( $(date +%s) - T0 ) / 60 )); }
n_done=0; n_fail=0; n_skip=0; QUEUE_ABORT=0

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
  local cap=$(( est * 60 * 3 + 900 ))
  log "RUN ${tag} [${class}] (est ${est}m, cap ${cap}s, elapsed ${now}m) -> engine/gradsim_true_${tag}.log"
  local t rc dt
  t=$(date +%s)
  timeout --signal=TERM --kill-after=60 "${cap}s" bash -c "$cmd" >> "engine/gradsim_true_${tag}.log" 2>&1
  rc=$?; dt=$(( $(date +%s) - t ))
  if [ "$rc" -eq 0 ]; then
    log "done ${tag} (${dt}s)"; n_done=$((n_done+1))
    [ "$class" = "SMOKE" ] && : > "engine/gradsim_true_${tag}.ok"
  else
    n_fail=$((n_fail+1))
    log "FAIL ${tag} (rc ${rc}, ${dt}s) — see engine/gradsim_true_${tag}.log"
    [ "$class" != "SMOKE" ] && QUEUE_ABORT=1
  fi
}
heartbeat(){ log "PROGRESS jobs=${n_done}done/${n_fail}fail/${n_skip}skip elapsed=$(elapsed_min)m budget_left=$(( BUDGET_MIN - $(elapsed_min) ))m"; }

# ---------------------------------------------------------------- Row 0: CPU self-contained smoke
# gradsim_true.py's OWN full pipeline (self-contained mode, no pre-existing npz) at tiny
# scale — also exercises the Prop.1 identity assert (hard-fails the row on any mismatch).
run_row SMOKE smoke 4 - "$ENVP $PY experiments/gradsim_true.py --model data/models/Qwen2.5-0.5B --layer auto --n_edits 3 --n_probes 8 --steps 2 --device cpu --out results/smoke_gradsim_true/qwen05b_cpu_smoke.json"
heartbeat

# ---------------------------------------------------------------- Row 1: mechanism_dump --save_vectors
run_row SCIENCE mech_l12_s0 15 engine/gradsim_true_smoke.ok "$ENVP $PY experiments/mechanism_dump.py --model data/models/Llama-3.2-1B --data data/counterfact.json --dataset counterfact --n_edits 200 --layer 12 --seed 0 --steps 20 --lr 0.1 --device cuda --save_vectors --out_dir results/mechanism"
heartbeat

# ---------------------------------------------------------------- Row 2: gradsim_true (external, reuses existing gate npz)
# needs= Row 1's OWN output file (not a synthetic .ok marker): if Row 1 gets BUDGET-SKIPPED
# or CONFIG-SKIPPED, Row 2 must not run against a stale/absent mechanism npz (F4, reviewed
# 2026-07-06) -- gradsim_true.py itself now also hard-fails on a key_norm mismatch, but
# skipping the row here avoids paying for a doomed GPU cell in the first place.
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE gradsim_true_l12_s0 15 results/mechanism/Llama-3.2-1B_L12.npz "$ENVP $PY experiments/gradsim_true.py --model data/models/Llama-3.2-1B --layer 12 --seed 0 --n_edits 200 --n_probes 500 --gate_npz results/matrices/gate_llama1b_rome_cf_L12_s0.npz --mech_npz results/mechanism/Llama-3.2-1B_L12.npz --known --edit_ok --device cuda --out results/GRADSIM_TRUE_Llama-3.2-1B_L12_s0.json"
heartbeat

log "================ RUN_GRADSIM_TRUE DONE (${n_done} done, ${n_fail} fail, ${n_skip} skip) ================"
