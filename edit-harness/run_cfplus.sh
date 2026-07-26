#!/bin/bash
# run_cfplus.sh — CF+ (CounterFact+) hard-neighborhood specificity battery (2026-07-08).
# Template = run_ripple.sh (verbatim skeleton for a standalone-CLI driver whose target
# script writes ONE json per cell, no --save_matrices/.npz — cfplus_specificity.py has no
# probe-matrix machinery, unlike killgate_keygeom.py), CFPLUS-namespaced (own pid/log/
# markers). Calls experiments/cfplus_specificity.py directly (NOT wired into
# killgate_keygeom.py — deliberately, per that module's own docstring: "Intended
# invocation once wired into a driver (NOT built here...) — queue decision deferred").
#
# SCIENCE QUESTION: standard CounterFact neighborhood prompts are "easy" locality probes
# (they never mention the edited fact); CF+ instead evaluates specificity under a
# distribution shift TOWARD the edit (prompt prefixed with the edit's own rewritten fact) —
# does the editor's apparent locality (NS) survive that harder test (NS_hard), and does
# that pattern hold across editors of different aggressiveness (rome/memit/alpha)? See
# cfplus_specificity.py's module docstring for the full method + its honest divergences
# from the published CF+ paper (no network access to verify the exact prompt templates).
#
# EDITOR SUPPORT (2026-07-08 extension to cfplus_specificity.py's own CLI, added
# alongside this driver): --editor {rome,memit,alpha}. memit/alpha fit their
# covariance/null-projector from a --n_fit-sized disjoint fact bank (drawn right after
# the edits, same shuffle/seed) — cfplus has no probe/holdout split of its own, so this
# fit bank plays that role for both editors (by-construction, reference-quality; NOT the
# honest holdout causal test — same caveat as killgate's default alpha_proj_source=probes).
#
# MEMORY-BOUND CAVEAT (cfplus_specificity.py's own docstring): baselines are
# O(n_edits * max_neighborhood * 2 * |vocab|) float32, kept in memory for the whole run
# — NOT yet exercised at killgate's n_edits=200 science scale. This driver keeps
# n_edits MODEST (40) per that caveat, not 200.
set -u
H="$(cd "$(dirname "$0")" && pwd)"
cd "$H" || exit 1
PY=/home/zeyufu/miniconda3/envs/dl/bin/python3
PY="${CLOUD_PY:-$PY}"   # 2026-07-08 B4 fix: box's own python (run_cloud_wave.sh exports CLOUD_PY); local default unchanged
LOG=engine/run_cfplus.log
BUDGET_MIN=${BUDGET_MIN:-180}
mkdir -p engine results/smoke_cfplus results/cfplus
echo $$ > engine/run_cfplus.pid
[ -f engine/cfplus_round_start ] || stat -c %Y engine/run_cfplus.pid > engine/cfplus_round_start
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
log "================ RUN_CFPLUS START (pid $$, budget ${BUDGET_MIN}m) ================"

# ---------------------------------------------------------------- Phase 0a: CPU pre-flight
pf_fail=0
pf(){ if eval "$2"; then log "preflight OK: $1"; else log "PREFLIGHT-FAIL: $1"; pf_fail=1; fi; }
pf "python env" "$PY -c 'import torch, numpy' 2>/dev/null"
pf "cfplus_specificity.py" "[ -f experiments/cfplus_specificity.py ]"
pf "cfplus_specificity.py --editor support" "grep -q -- '\-\-editor' experiments/cfplus_specificity.py"
pf "counterfact.json" "[ -f data/counterfact.json ]"
pf "model Llama-3.2-1B" "[ -d data/models/Llama-3.2-1B ]"
pf "disk >=15GB free" "[ \$(df --output=avail -BG /home | tail -1 | tr -dc 0-9) -ge 15 ]"
rm -f engine/smoke_cfplus_*.ok
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

# ---------------------------------------------------------------- helpers (ripple/mquake_law template, adapted: JSON-only cells, no npz)
ENVP="env -u ALL_PROXY -u all_proxy HF_HUB_OFFLINE=1"
CF="experiments/cfplus_specificity.py"
CFDATA="--data data/counterfact.json"
COMMON="--n_edits 40 --n_fit 100 --max_neighborhood 5 --steps 20 --device cuda"
SMK="--n_edits 2 --n_fit 8 --max_neighborhood 4 --steps 2 --device cpu"
elapsed_min(){ echo $(( ( $(date +%s) - T0 ) / 60 )); }

validate(){
  $PY - "$1" <<'EOF'
import json, sys
j = sys.argv[1]
try:
    d = json.load(open(j))
except Exception as e:
    print(f"VALIDATE-FAIL json unparseable: {e}"); sys.exit(1)
need = {"summary", "records", "editor"}
missing = need - set(d.keys())
if missing: print(f"VALIDATE-FAIL json missing keys {missing}"); sys.exit(1)
s = d["summary"]
if s.get("n_edits", 0) == 0: print("VALIDATE-FAIL summary n_edits==0"); sys.exit(1)
import math
def has_nan(o):
    if isinstance(o, dict): return any(has_nan(v) for v in o.values())
    if isinstance(o, list): return any(has_nan(v) for v in o)
    if isinstance(o, float): return math.isnan(o)
    return False
if has_nan(d): print("VALIDATE-FAIL NaN present in output"); sys.exit(1)
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
      [ "$class" = "SMOKE" ] && : > "engine/smoke_cfplus_${tag}.ok"
      return
    fi
  fi
  local mdir; mdir=$(echo "$cmd" | grep -oE -- '--model [^ ]+' | awk '{print $2}')
  case "$mdir" in data/*|/*)
    if [ -n "$mdir" ] && [ ! -d "$mdir" ]; then log "CONFIG-SKIP ${tag} (no model ${mdir})"; n_skip=$((n_skip+1)); return; fi;;
  esac
  local cap=$(( est * 60 * 3 + 1200 ))
  log "RUN ${tag} [${class}] (est ${est}m, cap ${cap}s, elapsed ${now}m) -> engine/cfplus_${tag}.log"
  local t rc dt
  t=$(date +%s)
  timeout --signal=TERM --kill-after=60 "${cap}s" bash -c "$cmd" >> "engine/cfplus_${tag}.log" 2>&1
  rc=$?; dt=$(( $(date +%s) - t ))
  if [ "$rc" -eq 0 ] && [ -n "$outj" ] && [ -f "$outj" ]; then
    local v; v=$(validate "$outj")
    if echo "$v" | grep -q VALIDATE-FAIL; then
      mv "$outj" "$outj.INVALID" 2>/dev/null
      log "FAIL ${tag} (${dt}s) OUTPUT-INVALID: ${v}"; n_fail=$((n_fail+1))
    else
      log "done ${tag} (${dt}s) ${v}"; n_done=$((n_done+1)); wedge_fail=0
      [ "$class" = "SMOKE" ] && : > "engine/smoke_cfplus_${tag}.ok"
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

# ---------------------------------------------------------------- Phase 0c: micro-smoke (all 3 editors, tiny)
run_row SMOKE rome_smoke 3 - "$ENVP $PY $CF --model data/models/Llama-3.2-1B --editor rome $CFDATA $SMK --lr 0.1 --layer 12 --seed 0 --out results/smoke_cfplus/cfplus_rome_smoke.json"
run_row SMOKE memit_smoke 4 - "$ENVP $PY $CF --model data/models/Llama-3.2-1B --editor memit $CFDATA $SMK --lr 0.1 --layer 12 --seed 0 --out results/smoke_cfplus/cfplus_memit_smoke.json"
run_row SMOKE alpha_smoke 4 - "$ENVP $PY $CF --model data/models/Llama-3.2-1B --editor alpha $CFDATA $SMK --lr 0.1 --layer 12 --seed 0 --out results/smoke_cfplus/cfplus_alpha_smoke.json"
heartbeat

# ---------------------------------------------------------------- Block C: rome/memit/alpha x L12 x 3 seeds
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE cfplus_llama1b_rome_L12_s0 12 engine/smoke_cfplus_rome_smoke.ok "$ENVP $PY $CF --model data/models/Llama-3.2-1B --editor rome $CFDATA $COMMON --lr 0.1 --layer 12 --seed 0 --out results/cfplus/cfplus_llama1b_rome_L12_s0.json"
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE cfplus_llama1b_rome_L12_s1 12 engine/smoke_cfplus_rome_smoke.ok "$ENVP $PY $CF --model data/models/Llama-3.2-1B --editor rome $CFDATA $COMMON --lr 0.1 --layer 12 --seed 1 --out results/cfplus/cfplus_llama1b_rome_L12_s1.json"
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE cfplus_llama1b_rome_L12_s2 12 engine/smoke_cfplus_rome_smoke.ok "$ENVP $PY $CF --model data/models/Llama-3.2-1B --editor rome $CFDATA $COMMON --lr 0.1 --layer 12 --seed 2 --out results/cfplus/cfplus_llama1b_rome_L12_s2.json"
heartbeat
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE cfplus_llama1b_memit_L12_s0 15 engine/smoke_cfplus_memit_smoke.ok "$ENVP $PY $CF --model data/models/Llama-3.2-1B --editor memit $CFDATA $COMMON --lr 0.1 --layer 12 --seed 0 --out results/cfplus/cfplus_llama1b_memit_L12_s0.json"
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE cfplus_llama1b_memit_L12_s1 15 engine/smoke_cfplus_memit_smoke.ok "$ENVP $PY $CF --model data/models/Llama-3.2-1B --editor memit $CFDATA $COMMON --lr 0.1 --layer 12 --seed 1 --out results/cfplus/cfplus_llama1b_memit_L12_s1.json"
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE cfplus_llama1b_memit_L12_s2 15 engine/smoke_cfplus_memit_smoke.ok "$ENVP $PY $CF --model data/models/Llama-3.2-1B --editor memit $CFDATA $COMMON --lr 0.1 --layer 12 --seed 2 --out results/cfplus/cfplus_llama1b_memit_L12_s2.json"
heartbeat
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE cfplus_llama1b_alpha_L12_s0 15 engine/smoke_cfplus_alpha_smoke.ok "$ENVP $PY $CF --model data/models/Llama-3.2-1B --editor alpha $CFDATA $COMMON --lr 0.1 --layer 12 --seed 0 --out results/cfplus/cfplus_llama1b_alpha_L12_s0.json"
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE cfplus_llama1b_alpha_L12_s1 15 engine/smoke_cfplus_alpha_smoke.ok "$ENVP $PY $CF --model data/models/Llama-3.2-1B --editor alpha $CFDATA $COMMON --lr 0.1 --layer 12 --seed 1 --out results/cfplus/cfplus_llama1b_alpha_L12_s1.json"
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE cfplus_llama1b_alpha_L12_s2 15 engine/smoke_cfplus_alpha_smoke.ok "$ENVP $PY $CF --model data/models/Llama-3.2-1B --editor alpha $CFDATA $COMMON --lr 0.1 --layer 12 --seed 2 --out results/cfplus/cfplus_llama1b_alpha_L12_s2.json"
heartbeat

# ---------------------------------------------------------------- Post-run (CPU, ALWAYS): small aggregation JSON
log "---------------- POST-RUN (CPU) ----------------"
$PY - > results/CFPLUS_aggregate.json 2>>"$LOG" <<'EOF'
import json, glob, statistics as st
by_editor = {}
for j in sorted(glob.glob('results/cfplus/cfplus_llama1b_*_L12_s*.json')):
    d = json.load(open(j))
    ed = d.get('editor', 'unknown')
    by_editor.setdefault(ed, []).append(d['summary'])
out = {}
for ed, rows in by_editor.items():
    def mean(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return st.mean(vals) if vals else None
    out[ed] = {
        'n_seeds': len(rows),
        'NS_mean': mean('NS'), 'NS_hard_mean': mean('NS_hard'),
        'KL_std_mean': mean('KL_std_mean'), 'KL_hard_mean': mean('KL_hard_mean'),
        'per_seed_NS': [r.get('NS') for r in rows],
        'per_seed_NS_hard': [r.get('NS_hard') for r in rows],
    }
print(json.dumps({'n_editors': len(out), 'by_editor': out}, indent=1))
EOF
log "post: CFPLUS_aggregate.json attempted"

{
  echo "RUN_CFPLUS REPORT $(date '+%F %T')  jobs: ${n_done} done / ${n_fail} fail / ${n_skip} skipped  elapsed $(elapsed_min)m/${BUDGET_MIN}m  abort=${QUEUE_ABORT}"
  grep -E 'RUN |done |FAIL |SKIP|ABORT|PROGRESS|THERMAL' "$LOG" | tail -60
} > engine/run_cfplus_report.txt
log "================ RUN_CFPLUS COMPLETE (${n_done} done / ${n_fail} fail / ${n_skip} skip) ================"
echo "RUN_CFPLUS_DONE" >> "$LOG"