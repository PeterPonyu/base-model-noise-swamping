#!/bin/bash
# run_pythia.sh — P2 of the 2026-07-09 enhancement round (4090 box): does the NeoX-20B
# editable-band collapse (memory/neox20b-esr-depth-collapse-20260709.md) come from the
# GPT-NeoX ARCHITECTURE or from 20B SCALE? Pythia-1.4B/2.8B share NeoX's architecture +
# tokenizer (editors/arch_compat.py's "gptneox" branch runs them unchanged) at sizes
# where Llama-family models show esr>=0.93 through depth 0.875 (results/esr_band_table
# .json). If Pythia's band also ends early -> architectural; if wide -> scale.
#
# TWO-STAGE ADAPTIVE DESIGN (the 20B lesson: NEVER commit 3-seed science to a layer/lr
# nobody probed — the L33 lr-0.1 row burned 64 min producing esr 0.015):
#   Stage A: smoke-size esr probes (20 edits/50 probes/steps 20) at 4 depth fractions
#            x lr {0.1, 0.5} per model -> results/probe_pythia/*.json
#   Stage B: python selector keeps layers with esr>=0.30 (cap 3, best-lr per layer),
#            runs killgate gate rows x 3 seeds there + one alpha-holdout row at the
#            best layer. Zero selected layers -> loud log, model's science skipped.
# Science rows are STANDARD killgate cells (fp32, cf, 200x500x20) -> gate_pythia{14,28}b
# npz feed the existing analysis + esr_band_analysis.py registry unchanged.
#
# MODELS (downloaded by cloud/dl_pythia.py, possibly still in flight when this driver
# starts — each model block CONFIG-skips cleanly if its dir is absent/incomplete).
set -u
H="$(cd "$(dirname "$0")" && pwd)"
cd "$H" || exit 1
PY=/home/zeyufu/miniconda3/envs/dl/bin/python3
PY="${CLOUD_PY:-$PY}"
LOG=engine/run_pythia.log
BUDGET_MIN=${BUDGET_MIN:-720}
# NOTE (review LOW-3): this driver has NO SMOKE row — the Stage-A PROBE rows are the
# gate (a science row only runs on a layer whose probe cleared esr>=0.30); the SMOKE
# branch inside run_row is template plumbing that never fires here.
mkdir -p engine results/probe_pythia results/matrices
echo $$ > engine/run_pythia.pid
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
log "================ RUN_PYTHIA START (pid $$, budget ${BUDGET_MIN}m) ================"

# ---------------------------------------------------------------- Phase 0a: CPU pre-flight
pf_fail=0
pf(){ if eval "$2"; then log "preflight OK: $1"; else log "PREFLIGHT-FAIL: $1"; pf_fail=1; fi; }
pf "python env" "$PY -c 'import torch, numpy, scipy' 2>/dev/null"
pf "killgate_keygeom.py" "[ -f experiments/killgate_keygeom.py ]"
pf "counterfact.json" "[ -f data/counterfact.json ]"
pf "arch_compat gptneox branch" "grep -q -- 'family = \"gptneox\"' editors/arch_compat.py"
pf "integrity_check.py" "[ -f experiments/tools/integrity_check.py ]"
pf "disk >=10GB free" "[ \$(df --output=avail -BG . | tail -1 | tr -dc 0-9) -ge 10 ]"
rm -f engine/smoke_pythia_*.ok
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
KG="experiments/killgate_keygeom.py"
CF="--dataset counterfact --data data/counterfact.json"
COMMON="--n_edits 200 --n_probes 500 --steps 20 --save_matrices --matrix_dir results/matrices"
PRB="--n_edits 20 --n_probes 50 --steps 20 --save_matrices --matrix_dir results/probe_pythia/matrices"
elapsed_min(){ echo $(( ( $(date +%s) - T0 ) / 60 )); }

validate(){   # json must parse and carry edit_success_rate; SCIENCE also needs its npz
  $PY - "$1" "${2:-}" <<'EOF'
import json, os, sys
try:
    d = json.load(open(sys.argv[1]))
    esr = d["edit_success_rate"]
    npz = sys.argv[2]
    if npz and not os.path.exists(npz):
        print(f"VALIDATE-FAIL missing npz {npz}"); raise SystemExit
    print("VALIDATE-OK" if esr >= 0.9 else f"VALIDATE-WARN esr={esr}<0.9")
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
  if [ "$class" = "SCIENCE" ]; then
    outn="results/matrices/$(basename "${outj%.json}").npz"
  fi
  if [ -n "$outj" ] && [ -f "$outj" ]; then
    local v; v=$(validate "$outj" "$outn")
    if echo "$v" | grep -q VALIDATE-FAIL; then
      mv "$outj" "$outj.INVALID" 2>/dev/null
      log "STALE-INVALID ${tag} — quarantined; re-running"
    else
      log "skip ${tag} (exists, validated)"
      [ "$class" = "SMOKE" ] && : > "engine/smoke_pythia_${tag}.ok"
      return
    fi
  fi
  local mdir; mdir=$(echo "$cmd" | grep -oE -- '--model [^ ]+' | awk '{print $2}')
  case "$mdir" in data/*|/*)
    if [ -n "$mdir" ] && [ ! -d "$mdir" ]; then log "CONFIG-SKIP ${tag} (no model ${mdir})"; n_skip=$((n_skip+1)); return; fi;;
  esac
  local cap=$(( est * 60 * 3 + 1200 ))
  log "RUN ${tag} [${class}] (est ${est}m, cap ${cap}s, elapsed ${now}m) -> engine/pythia_${tag}.log"
  local t rc dt
  t=$(date +%s)
  # </dev/null (review LOW-2): run_row is invoked inside `while read <<< "$sel"` — an
  # inner command that read stdin would consume the selector list and truncate the loop
  timeout --signal=TERM --kill-after=60 "${cap}s" bash -c "$cmd" >> "engine/pythia_${tag}.log" 2>&1 </dev/null
  rc=$?; dt=$(( $(date +%s) - t ))
  if [ "$rc" -eq 0 ] && [ -n "$outj" ] && [ -f "$outj" ]; then
    local v; v=$(validate "$outj" "$outn")
    if echo "$v" | grep -q VALIDATE-FAIL; then
      mv "$outj" "$outj.INVALID" 2>/dev/null
      log "FAIL ${tag} (${dt}s) OUTPUT-INVALID: ${v}"; n_fail=$((n_fail+1))
    else
      log "done ${tag} (${dt}s) ${v}"; n_done=$((n_done+1)); wedge_fail=0
      [ "$class" = "SMOKE" ] && : > "engine/smoke_pythia_${tag}.ok"
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

# selector: read Stage-A probe jsons for a model tag, keep layers with best-lr esr>=0.30
# (cap 3, esr-desc), emit "layer lr" lines. Missing/failed probe jsons are just absent.
select_layers(){   # select_layers <model_tag>
  $PY - "$1" <<'EOF'
import glob, json, re, sys
tag = sys.argv[1]
best = {}
for p in glob.glob(f"results/probe_pythia/{tag}_L*_lr*.json"):
    m = re.search(r"_L(\d+)_lr(\d+)\.json$", p)
    if not m:
        continue
    L, lr = int(m.group(1)), {"01": 0.1, "05": 0.5}.get(m.group(2))
    if lr is None:
        continue
    try:
        esr = json.load(open(p))["edit_success_rate"]
    except Exception:
        continue
    if L not in best or esr > best[L][1]:
        best[L] = (lr, esr)
keep = sorted(((L, lr, esr) for L, (lr, esr) in best.items() if esr >= 0.30),
              key=lambda x: -x[2])[:3]
for L, lr, esr in keep:
    print(f"{L} {lr} {esr}")
EOF
}

# ---------------------------------------------------------------- per-model battery
#   model_battery <tag> <model_dir> <expect_params> <probe_layers space-sep> <row_est_min>
model_battery(){
  local tag="$1" mdir="$2" expparams="$3" probe_layers="$4" est="$5"
  if [ ! -d "$mdir" ]; then
    if [ "$DRYRUN" -eq 1 ]; then
      log "DRYRUN: ${mdir} absent — expanding probe rows anyway (on-box the download precedes this driver)"
    else
      log "MODEL-ABSENT ${tag}: ${mdir} — battery skipped (download still running?)"; return
    fi
  fi
  if [ "$DRYRUN" -ne 1 ]; then
    if ! $PY experiments/tools/integrity_check.py "$mdir" --expect_params "$expparams" >> "$LOG" 2>&1; then
      log "INTEGRITY-FAIL ${tag}: ${mdir} — battery skipped (incomplete download?)"; return
    fi
    log "integrity OK: ${tag}"
  fi
  # Stage A: esr probes (idempotent; ~4-6 min each)
  local L lrtag lr
  for L in $probe_layers; do
    for lr in 0.1 0.5; do
      lrtag=$(echo "$lr" | tr -d .)
      [ "$QUEUE_ABORT" -eq 0 ] && run_row PROBE ${tag}_L${L}_lr${lrtag} 8 - "$ENVP $PY $KG --model $mdir --editor rome $CF $PRB --lr $lr --layer $L --seed 0 --out results/probe_pythia/${tag}_L${L}_lr${lrtag}.json"
    done
  done
  heartbeat
  if [ "$DRYRUN" -eq 1 ]; then
    log "DRYRUN: selector + science rows for ${tag} not expanded (depend on probe results)"
    return
  fi
  # Stage B: science on the selected band
  local sel; sel=$(select_layers "$tag")
  if [ -z "$sel" ]; then
    log "BAND-EMPTY ${tag}: no probed layer reached esr>=0.30 — science skipped (that null IS a result; probes stay in results/probe_pythia/)"
    return
  fi
  log "BAND ${tag}: $(echo "$sel" | tr '\n' ';')"
  local first=1 bestL="" bestlr=""
  while read -r L lr esr; do
    [ -z "$L" ] && continue
    if [ "$first" -eq 1 ]; then bestL="$L"; bestlr="$lr"; first=0; fi
    local s
    for s in 0 1 2; do
      [ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE gate_${tag}_rome_cf_L${L}_s${s} "$est" - "$ENVP $PY $KG --model $mdir --editor rome $CF $COMMON --lr $lr --layer $L --seed ${s} --out results/gate_${tag}_rome_cf_L${L}_s${s}.json"
    done
    heartbeat
  done <<< "$sel"
  # alpha-holdout causal row at the best-esr layer
  [ "$QUEUE_ABORT" -eq 0 ] && [ -n "$bestL" ] && run_row SCIENCE g4_${tag}_alphaHO_cf_L${bestL}_s0 $(( est + 15 )) - "$ENVP $PY $KG --model $mdir --editor alpha $CF $COMMON --lr $bestlr --layer $bestL --seed 0 --alpha_proj_source holdout --holdout_frac 1.0 --out results/g4_${tag}_alphaHO_cf_L${bestL}_s0.json"
  heartbeat
}

# pythia-1.4b: 24 layers -> probe {6,12,15,18} = depth {0.25,0.50,0.625,0.75}
model_battery pythia14b data/models/pythia-1.4b 1.5153e9 "6 12 15 18" 35
# pythia-2.8b: 32 layers -> probe {8,16,20,24} = same fractions
model_battery pythia28b data/models/pythia-2.8b 2.9094e9  "8 16 20 24" 55

log "================ RUN_PYTHIA COMPLETE (${n_done} done / ${n_fail} fail / ${n_skip} skip) ================"
echo "RUN_PYTHIA_DONE"
