#!/bin/bash
# run_glue_seq.sh — GLUE downstream-accuracy tracking across a sequential ROME edit
# trajectory (2026-07-08). Template = run_cfplus.sh / run_ripple.sh (standalone-CLI
# driver, target script writes ONE json per cell, no --save_matrices/.npz), GLUESEQ-
# namespaced (own pid/log/markers). Calls experiments/glue_downstream.py (a NEW module,
# see its docstring — reuses editors.rome_native.apply_edit + metrics.py's next-token-
# logit scoring, does not reimplement either).
#
# SCIENCE QUESTION: the "table-stakes downstream tracking" damage claims need — as
# UNRELATED CounterFact edits accumulate (never restored, same cumulative-sequence
# concept as killgate_keygeom.py's SEQ/--no_restore mode), does zero-shot accuracy on
# SST-2/MRPC/RTE degrade? Checkpoints at 0/10/50/100 edits (0 = the unedited baseline,
# always measured first).
#
# SCOPE NOTE: rome only (per the task spec: "apply a SEQUENCE of N ROME edits"); a
# single sequence per seed (checkpoints are read off ONE growing trajectory, not
# independent per-checkpoint runs — cheaper, and matches how a real deployment
# accumulates edits).
set -u
H="$(cd "$(dirname "$0")" && pwd)"
cd "$H" || exit 1
PY=/home/zeyufu/miniconda3/envs/dl/bin/python3
PY="${CLOUD_PY:-$PY}"   # 2026-07-08 B4 fix: box's own python (run_cloud_wave.sh exports CLOUD_PY); local default unchanged
LOG=engine/run_glue_seq.log
BUDGET_MIN=${BUDGET_MIN:-180}
mkdir -p engine results/smoke_glue_seq results/glue_seq
echo $$ > engine/run_glue_seq.pid
[ -f engine/glue_seq_round_start ] || stat -c %Y engine/run_glue_seq.pid > engine/glue_seq_round_start
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
log "================ RUN_GLUE_SEQ START (pid $$, budget ${BUDGET_MIN}m) ================"

# ---------------------------------------------------------------- Phase 0a: CPU pre-flight
pf_fail=0
pf(){ if eval "$2"; then log "preflight OK: $1"; else log "PREFLIGHT-FAIL: $1"; pf_fail=1; fi; }
pf "python env" "$PY -c 'import torch, numpy, pandas' 2>/dev/null"
pf "glue_downstream.py" "[ -f experiments/glue_downstream.py ]"
pf "counterfact.json" "[ -f data/counterfact.json ]"
pf "data/glue/sst2 validation parquet" "[ -f data/glue/sst2/validation-00000-of-00001.parquet ]"
pf "data/glue/mrpc validation parquet" "[ -f data/glue/mrpc/validation-00000-of-00001.parquet ]"
pf "data/glue/rte validation parquet" "[ -f data/glue/rte/validation-00000-of-00001.parquet ]"
pf "model Llama-3.2-1B" "[ -d data/models/Llama-3.2-1B ]"
pf "disk >=15GB free" "[ \$(df --output=avail -BG /home | tail -1 | tr -dc 0-9) -ge 15 ]"
rm -f engine/smoke_glue_seq_*.ok
if [ "$pf_fail" -ne 0 ]; then log "ABORT: preflight failed"; exit 3; fi

# ---------------------------------------------------------------- Phase 0b: GPU idle gate
# Per-card gate via cloud/gpu_idle_lib.sh (2026-07-08 cloud-wave fix) — honors
# SKIP_IDLE_GATE (dedicated-box bypass) and IDLE_GATE_DEVICE (gate on the physical card
# this worker actually owns, not always GPU0). See cloud/README.md "driver idle-gate
# contract" for why the old unqualified `nvidia-smi | head -1` loop was unsafe on a
# multi-card box.
DRYRUN=${DRYRUN:-0}
if [ "$DRYRUN" -eq 1 ]; then
  log "DRYRUN=1 — skipping GPU idle gate, printing every run_row call without executing"
else
  source "$H/cloud/gpu_idle_lib.sh"
  idle_gate_wait || { log "ABORT: idle gate failed"; exit 2; }
fi
T0=$(date +%s)

# ---------------------------------------------------------------- helpers (cfplus/ripple template, adapted: JSON-only cells, no npz)
ENVP="env -u ALL_PROXY -u all_proxy HF_HUB_OFFLINE=1"
GS="experiments/glue_downstream.py"
GSDATA="--data data/counterfact.json --glue_dir data/glue"
COMMON="--n_glue_samples 100 --n_edits 100 --checkpoints 0,10,50,100 --steps 20 --device cuda"
SMK="--n_glue_samples 2 --n_edits 4 --checkpoints 0,2,4 --steps 2 --device cpu"
elapsed_min(){ echo $(( ( $(date +%s) - T0 ) / 60 )); }

validate(){
  $PY - "$1" <<'EOF'
import json, sys
j = sys.argv[1]
try:
    d = json.load(open(j))
except Exception as e:
    print(f"VALIDATE-FAIL json unparseable: {e}"); sys.exit(1)
need = {"trajectory", "checkpoints", "tasks"}
missing = need - set(d.keys())
if missing: print(f"VALIDATE-FAIL json missing keys {missing}"); sys.exit(1)
traj = d["trajectory"]
if not traj: print("VALIDATE-FAIL empty trajectory"); sys.exit(1)
for row in traj:
    for t, acc in row.get("accuracy", {}).items():
        if acc is None or not (0.0 <= acc <= 1.0):
            print(f"VALIDATE-FAIL accuracy out of [0,1] at n_edits={row.get('n_edits')} task={t}: {acc}")
            sys.exit(1)
if traj[0].get("n_edits") != 0:
    print("VALIDATE-FAIL trajectory must start at the n_edits=0 baseline checkpoint"); sys.exit(1)
print("VALIDATE-OK")
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
    if validate "$outj" | grep -q VALIDATE-FAIL; then
      mv "$outj" "$outj.INVALID" 2>/dev/null
      log "STALE-INVALID ${tag} — quarantined; re-running"
    else
      log "skip ${tag} (exists, validated)"
      [ "$class" = "SMOKE" ] && : > "engine/smoke_glue_seq_${tag}.ok"
      return
    fi
  fi
  local mdir; mdir=$(echo "$cmd" | grep -oE -- '--model [^ ]+' | awk '{print $2}')
  case "$mdir" in data/*|/*)
    if [ -n "$mdir" ] && [ ! -d "$mdir" ]; then log "CONFIG-SKIP ${tag} (no model ${mdir})"; n_skip=$((n_skip+1)); return; fi;;
  esac
  local cap=$(( est * 60 * 3 + 1200 ))
  log "RUN ${tag} [${class}] (est ${est}m, cap ${cap}s, elapsed ${now}m) -> engine/glue_seq_${tag}.log"
  local t rc dt
  t=$(date +%s)
  timeout --signal=TERM --kill-after=60 "${cap}s" bash -c "$cmd" >> "engine/glue_seq_${tag}.log" 2>&1
  rc=$?; dt=$(( $(date +%s) - t ))
  if [ "$rc" -eq 0 ] && [ -n "$outj" ] && [ -f "$outj" ]; then
    local v; v=$(validate "$outj")
    if echo "$v" | grep -q VALIDATE-FAIL; then
      mv "$outj" "$outj.INVALID" 2>/dev/null
      log "FAIL ${tag} (${dt}s) OUTPUT-INVALID: ${v}"; n_fail=$((n_fail+1))
    else
      log "done ${tag} (${dt}s) ${v}"; n_done=$((n_done+1)); wedge_fail=0
      [ "$class" = "SMOKE" ] && : > "engine/smoke_glue_seq_${tag}.ok"
      if [ "$class" != "SMOKE" ] && [ "$dt" -gt $(( est * 60 * 14 / 10 )) ]; then
        local pw; pw=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits 2>/dev/null | head -1)
        log "THERMAL-WATCH ${tag} ran ${dt}s > 1.4x est; power.draw now=${pw:-NA}W (wedge if <100 under load)"
      fi
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

# ---------------------------------------------------------------- Phase 0c: micro-smoke
run_row SMOKE rome_smoke 3 - "$ENVP $PY $GS --model data/models/Llama-3.2-1B $GSDATA $SMK --lr 0.1 --layer 12 --seed 0 --out results/smoke_glue_seq/glue_seq_smoke.json"
heartbeat

# ---------------------------------------------------------------- Block G: rome L12, 3 seeds, 0/10/50/100-edit trajectory
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE glue_seq_llama1b_rome_L12_s0 15 engine/smoke_glue_seq_rome_smoke.ok "$ENVP $PY $GS --model data/models/Llama-3.2-1B $GSDATA $COMMON --lr 0.1 --layer 12 --seed 0 --out results/glue_seq/glue_seq_llama1b_rome_L12_s0.json"
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE glue_seq_llama1b_rome_L12_s1 15 engine/smoke_glue_seq_rome_smoke.ok "$ENVP $PY $GS --model data/models/Llama-3.2-1B $GSDATA $COMMON --lr 0.1 --layer 12 --seed 1 --out results/glue_seq/glue_seq_llama1b_rome_L12_s1.json"
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE glue_seq_llama1b_rome_L12_s2 15 engine/smoke_glue_seq_rome_smoke.ok "$ENVP $PY $GS --model data/models/Llama-3.2-1B $GSDATA $COMMON --lr 0.1 --layer 12 --seed 2 --out results/glue_seq/glue_seq_llama1b_rome_L12_s2.json"
heartbeat

# ---------------------------------------------------------------- Post-run (CPU, ALWAYS): 3-seed pooled trajectory
log "---------------- POST-RUN (CPU) ----------------"
$PY - > results/GLUE_SEQ_aggregate.json 2>>"$LOG" <<'EOF'
import json, glob, statistics as st
from collections import defaultdict
by_ckpt = defaultdict(lambda: defaultdict(list))  # n_edits -> task -> [acc, ...]
n_seeds = 0
for j in sorted(glob.glob('results/glue_seq/glue_seq_llama1b_rome_L12_s*.json')):
    d = json.load(open(j)); n_seeds += 1
    for row in d['trajectory']:
        for t, acc in row['accuracy'].items():
            by_ckpt[row['n_edits']][t].append(acc)
out = []
for n_edits in sorted(by_ckpt):
    tasks = {t: {'mean': st.mean(vals), 'per_seed': vals} for t, vals in by_ckpt[n_edits].items()}
    out.append({'n_edits': n_edits, 'tasks': tasks})
print(json.dumps({'n_seeds': n_seeds, 'trajectory': out}, indent=1))
EOF
log "post: GLUE_SEQ_aggregate.json attempted"

{
  echo "RUN_GLUE_SEQ REPORT $(date '+%F %T')  jobs: ${n_done} done / ${n_fail} fail / ${n_skip} skipped  elapsed $(elapsed_min)m/${BUDGET_MIN}m  abort=${QUEUE_ABORT}"
  grep -E 'RUN |done |FAIL |SKIP|ABORT|PROGRESS|THERMAL' "$LOG" | tail -60
} > engine/run_glue_seq_report.txt
log "================ RUN_GLUE_SEQ COMPLETE (${n_done} done / ${n_fail} fail / ${n_skip} skip) ================"
echo "RUN_GLUE_SEQ_DONE" >> "$LOG"