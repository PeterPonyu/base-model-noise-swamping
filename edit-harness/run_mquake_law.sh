#!/bin/bash
# run_mquake_law.sh — MQuAKE (multi-hop CounterFact-schema) law-replication driver
# (2026-07-05). Template = run_u6.sh (verbatim skeleton), MQUAKE-namespaced (own
# pid/log/markers — never reuses u5/u6 names). Third-dataset support added to
# killgate_keygeom.py (experiments/killgate_keygeom.py: load_mquake, --dataset mquake).
#
# SCIENCE: replicate the signed key-geometry law (Spearman(key-cos, damage) rising with
# layer, peaking ~L12) on MQuAKE-CF-3k's single-hop edits (requested_rewrite[0]) — a
# THIRD dataset beyond CounterFact/zsRE, using the SAME within-probe metric. Rows:
#   Llama-3.2-1B x rome x {L8,L10,L12,L14} x {s0,s1,s2}   (12 cells; the law replication)
#   Llama-3.2-1B x alpha x L12 x s0                        (1 cell; causal anchor, by-
#     construction probes-sourced projector — reference-only, NOT the honest holdout test;
#     see memory/c4-alphaedit-projector-circularity.md. A single quick anchor cell, not a
#     substitute for the CF/zsRE E6 holdout causal table.)
#
# HARD PRECONDITION: data/mquake_cf3k.json (MQuAKE-CF-3k.json from princeton-nlp/MQuAKE)
# must already exist on disk. Downloads are ask-first in this workspace — this driver
# NEVER fetches it; preflight aborts cleanly if absent (mirrors killgate's own startup
# guard for --dataset mquake, added alongside the loader).
set -u
H=/home/zeyufu/Desktop/idea-feasibility-analysis/edit-harness
cd "$H" || exit 1
PY=/home/zeyufu/miniconda3/envs/dl/bin/python3
LOG=engine/run_mquake_law.log
BUDGET_MIN=${BUDGET_MIN:-260}
mkdir -p engine results/matrices results/smoke_mquake_law/matrices
echo $$ > engine/run_mquake_law.pid
[ -f engine/mquake_law_round_start ] || stat -c %Y engine/run_mquake_law.pid > engine/mquake_law_round_start
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
log "================ RUN_MQUAKE_LAW START (pid $$, budget ${BUDGET_MIN}m) ================"

# ---------------------------------------------------------------- Phase 0a: CPU pre-flight
pf_fail=0
pf(){ if eval "$2"; then log "preflight OK: $1"; else log "PREFLIGHT-FAIL: $1"; pf_fail=1; fi; }
pf "python env" "$PY -c 'import torch, numpy' 2>/dev/null"
pf "mquake_cf3k.json (HARD: no auto-download — ask-first policy)" "[ -f data/mquake_cf3k.json ]"
pf "dataset mquake wired into killgate" "grep -q -- 'mquake' experiments/killgate_keygeom.py"
pf "load_mquake present" "grep -q -- 'def load_mquake' experiments/killgate_keygeom.py"
pf "model Llama-3.2-1B" "[ -d data/models/Llama-3.2-1B ]"
pf "analyze_matrices.py" "[ -f experiments/analyze_matrices.py ]"
pf "mechanism_sc_table.py" "[ -f experiments/mechanism_sc_table.py ]"
pf "disk >=20GB free" "[ \$(df --output=avail -BG /home | tail -1 | tr -dc 0-9) -ge 20 ]"
rm -f engine/smoke_mquake_law_*.ok
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

# ---------------------------------------------------------------- helpers (u2/u4/u5/u6 template, verbatim)
ENVP="env -u ALL_PROXY -u all_proxy HF_HUB_OFFLINE=1"
KG="experiments/killgate_keygeom.py"
MQ="--dataset mquake --data data/mquake_cf3k.json"
COMMON="--n_edits 200 --n_probes 500 --steps 20 --save_matrices --matrix_dir results/matrices"
SMK="--n_edits 4 --n_probes 40 --steps 2 --save_matrices --matrix_dir results/smoke_mquake_law/matrices"
elapsed_min(){ echo $(( ( $(date +%s) - T0 ) / 60 )); }

validate(){
  $PY - "$1" "$2" "${3:-full}" <<'EOF'
import json, os, sys, numpy as np
j, z, mode = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    d = json.load(open(j))
except Exception as e:
    print(f"VALIDATE-FAIL json unparseable: {e}"); sys.exit(1)
try:
    a = np.load(z)
except Exception as e:
    print(f"VALIDATE-FAIL npz unreadable: {e}"); sys.exit(1)
need = {"COS", "damage_logit", "norm_growth", "edit_ok"}
missing = need - set(a.files)
if missing: print(f"VALIDATE-FAIL npz missing {missing}"); sys.exit(1)
for k in need:
    arr = a[k].astype(float)
    if np.isnan(arr).all():
        print(f"VALIDATE-FAIL {k} all-NaN"); sys.exit(1)
if mode == "full":
    esr = d.get("edit_success_rate")
    if esr == 0: print("VALIDATE-FAIL esr==0 at real steps"); sys.exit(1)
    if esr is not None and esr < 0.9: print(f"VALIDATE-WARN esr={esr}<0.9")
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
  local outj outn
  outj=$(echo "$cmd" | grep -oE -- '--out [^ ]+' | awk '{print $2}')
  outn="results/matrices/$(basename "${outj%.json}").npz"
  case "$cmd" in *smoke_mquake_law*) outn="results/smoke_mquake_law/matrices/$(basename "${outj%.json}").npz";; esac
  if [ -n "$outj" ] && [ -f "$outj" ] && [ -f "$outn" ]; then
    local pvmode="full"; [ "$class" = "SMOKE" ] && pvmode="smoke"
    if validate "$outj" "$outn" "$pvmode" | grep -q VALIDATE-FAIL; then
      mv "$outj" "$outj.INVALID" 2>/dev/null; mv "$outn" "$outn.INVALID" 2>/dev/null
      log "STALE-INVALID ${tag} — quarantined; re-running"
    else
      log "skip ${tag} (exists, validated)"
      [ "$class" = "SMOKE" ] && : > "engine/smoke_mquake_law_${tag}.ok"
      return
    fi
  fi
  local mdir; mdir=$(echo "$cmd" | grep -oE -- '--model [^ ]+' | awk '{print $2}')
  case "$mdir" in data/*|/*)
    if [ -n "$mdir" ] && [ ! -d "$mdir" ]; then log "CONFIG-SKIP ${tag} (no model ${mdir})"; n_skip=$((n_skip+1)); return; fi;;
  esac
  local cap=$(( est * 60 * 3 + 1200 ))
  log "RUN ${tag} [${class}] (est ${est}m, cap ${cap}s, elapsed ${now}m) -> engine/mquake_law_${tag}.log"
  local t rc dt
  t=$(date +%s)
  timeout --signal=TERM --kill-after=60 "${cap}s" bash -c "$cmd" >> "engine/mquake_law_${tag}.log" 2>&1
  rc=$?; dt=$(( $(date +%s) - t ))
  local vmode="full"; [ "$class" = "SMOKE" ] && vmode="smoke"
  if [ "$rc" -eq 0 ] && [ -n "$outj" ] && [ -f "$outj" ] && [ -f "$outn" ]; then
    local v; v=$(validate "$outj" "$outn" "$vmode")
    if echo "$v" | grep -q VALIDATE-FAIL; then
      mv "$outj" "$outj.INVALID" 2>/dev/null; mv "$outn" "$outn.INVALID" 2>/dev/null
      log "FAIL ${tag} (${dt}s) OUTPUT-INVALID: ${v}"; n_fail=$((n_fail+1))
    else
      log "done ${tag} (${dt}s) ${v}"; n_done=$((n_done+1)); wedge_fail=0
      [ "$class" = "SMOKE" ] && : > "engine/smoke_mquake_law_${tag}.ok"
      # thermal sentinel: >1.4x est at full steps = possible 60W SW-wedge (memory:
      # gpu-60w-thermal-cap-reboot-fix)
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
# mquake has never been driven end-to-end on GPU; smoke the loader+editor combo first.
run_row SMOKE mquake_rome_smoke 4 - "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor rome $MQ $SMK --lr 0.1 --layer 12 --seed 0 --out results/smoke_mquake_law/mquake_rome.json"
heartbeat

# ---------------------------------------------------------------- Block R: law replication (rome, 4 layers x 3 seeds)
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE gate_llama1b_rome_mquake_L8_s0 25 engine/smoke_mquake_law_mquake_rome_smoke.ok "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor rome $MQ $COMMON --lr 0.1 --layer 8 --seed 0 --out results/gate_llama1b_rome_mquake_L8_s0.json"
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE gate_llama1b_rome_mquake_L8_s1 25 engine/smoke_mquake_law_mquake_rome_smoke.ok "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor rome $MQ $COMMON --lr 0.1 --layer 8 --seed 1 --out results/gate_llama1b_rome_mquake_L8_s1.json"
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE gate_llama1b_rome_mquake_L8_s2 25 engine/smoke_mquake_law_mquake_rome_smoke.ok "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor rome $MQ $COMMON --lr 0.1 --layer 8 --seed 2 --out results/gate_llama1b_rome_mquake_L8_s2.json"
heartbeat
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE gate_llama1b_rome_mquake_L10_s0 25 engine/smoke_mquake_law_mquake_rome_smoke.ok "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor rome $MQ $COMMON --lr 0.1 --layer 10 --seed 0 --out results/gate_llama1b_rome_mquake_L10_s0.json"
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE gate_llama1b_rome_mquake_L10_s1 25 engine/smoke_mquake_law_mquake_rome_smoke.ok "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor rome $MQ $COMMON --lr 0.1 --layer 10 --seed 1 --out results/gate_llama1b_rome_mquake_L10_s1.json"
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE gate_llama1b_rome_mquake_L10_s2 25 engine/smoke_mquake_law_mquake_rome_smoke.ok "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor rome $MQ $COMMON --lr 0.1 --layer 10 --seed 2 --out results/gate_llama1b_rome_mquake_L10_s2.json"
heartbeat
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE gate_llama1b_rome_mquake_L12_s0 25 engine/smoke_mquake_law_mquake_rome_smoke.ok "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor rome $MQ $COMMON --lr 0.1 --layer 12 --seed 0 --out results/gate_llama1b_rome_mquake_L12_s0.json"
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE gate_llama1b_rome_mquake_L12_s1 25 engine/smoke_mquake_law_mquake_rome_smoke.ok "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor rome $MQ $COMMON --lr 0.1 --layer 12 --seed 1 --out results/gate_llama1b_rome_mquake_L12_s1.json"
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE gate_llama1b_rome_mquake_L12_s2 25 engine/smoke_mquake_law_mquake_rome_smoke.ok "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor rome $MQ $COMMON --lr 0.1 --layer 12 --seed 2 --out results/gate_llama1b_rome_mquake_L12_s2.json"
heartbeat
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE gate_llama1b_rome_mquake_L14_s0 32 engine/smoke_mquake_law_mquake_rome_smoke.ok "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor rome $MQ $COMMON --lr 0.1 --layer 14 --seed 0 --out results/gate_llama1b_rome_mquake_L14_s0.json"
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE gate_llama1b_rome_mquake_L14_s1 32 engine/smoke_mquake_law_mquake_rome_smoke.ok "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor rome $MQ $COMMON --lr 0.1 --layer 14 --seed 1 --out results/gate_llama1b_rome_mquake_L14_s1.json"
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE gate_llama1b_rome_mquake_L14_s2 32 engine/smoke_mquake_law_mquake_rome_smoke.ok "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor rome $MQ $COMMON --lr 0.1 --layer 14 --seed 2 --out results/gate_llama1b_rome_mquake_L14_s2.json"
heartbeat

# ---------------------------------------------------------------- Block A: causal anchor (alpha, L12, s0)
# by-construction probes-sourced projector (default --alpha_proj_source probes) — quick
# reference-only anchor, NOT the honest holdout causal test (memory:
# c4-alphaedit-projector-circularity.md). A single cell; do not overclaim from it.
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE gate_llama1b_alpha_mquake_L12_s0 25 engine/smoke_mquake_law_mquake_rome_smoke.ok "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor alpha $MQ $COMMON --lr 0.1 --layer 12 --seed 0 --out results/gate_llama1b_alpha_mquake_L12_s0.json"
heartbeat

# ---------------------------------------------------------------- Post-run (CPU, ALWAYS)
log "---------------- POST-RUN (CPU) ----------------"
$PY - > results/mquake_law_validation.json 2>>"$LOG" <<'EOF'
import json, glob, os, numpy as np
t0 = float(open('engine/mquake_law_round_start').read().strip())
out = []
for j in sorted(glob.glob('results/gate_llama1b_*mquake*.json')):
    base = os.path.basename(j)[:-5]
    if os.path.getmtime(j) < t0: continue
    z = 'results/matrices/' + base + '.npz'
    row = {'json': j, 'npz_found': os.path.exists(z)}
    try:
        d = json.load(open(j)); row['json_ok'] = True; row['esr'] = d.get('edit_success_rate')
    except Exception as e:
        row['json_ok'] = False; row['err'] = str(e)
    out.append(row)
print(json.dumps({'n': len(out), 'rows': out}, indent=1))
EOF
log "post: validation sweep -> results/mquake_law_validation.json"

# per-layer 3-seed pooling (rome only; mirrors run_u6's C3_memit_L10/L14 pattern)
for spec in "C3_mquake_rome_L8:results/matrices/gate_llama1b_rome_mquake_L8_s*.npz" \
            "C3_mquake_rome_L10:results/matrices/gate_llama1b_rome_mquake_L10_s*.npz" \
            "C3_mquake_rome_L12:results/matrices/gate_llama1b_rome_mquake_L12_s*.npz" \
            "C3_mquake_rome_L14:results/matrices/gate_llama1b_rome_mquake_L14_s*.npz" \
            "C3_mquake_alpha_L12:results/matrices/gate_llama1b_alpha_mquake_L12_s*.npz"; do
  outn="${spec%%:*}"; glob="${spec#*:}"
  if compgen -G "$glob" >/dev/null; then
    $PY experiments/analyze_matrices.py $glob --metric logit --known --edit_ok \
      --out "results/${outn}_mquake_law.json" >> "$LOG" 2>&1 && log "post: ${outn}_mquake_law done" || log "post: ${outn}_mquake_law FAIL"
  fi
done

# S x C law-replication table across all 4 layers (the headline mquake result) — same
# invocation pattern as mechanism_sc_table.py's own docstring / run_u5/u6 post-processing.
if compgen -G "results/matrices/gate_llama1b_rome_mquake_L*_s*.npz" >/dev/null; then
  $PY experiments/mechanism_sc_table.py \
    --npz 'results/matrices/gate_llama1b_rome_mquake_L*_s*.npz' \
    --known --edit_ok \
    --out results/MQUAKE_mechanism_sc_table.json >> "$LOG" 2>&1 \
    && log "post: MQUAKE_mechanism_sc_table done" || log "post: MQUAKE_mechanism_sc_table FAIL"
fi

{
  echo "RUN_MQUAKE_LAW REPORT $(date '+%F %T')  jobs: ${n_done} done / ${n_fail} fail / ${n_skip} skipped  elapsed $(elapsed_min)m/${BUDGET_MIN}m  abort=${QUEUE_ABORT}"
  grep -E 'RUN |done |FAIL |SKIP|ABORT|PROGRESS|THERMAL' "$LOG" | tail -60
} > engine/run_mquake_law_report.txt
log "================ RUN_MQUAKE_LAW COMPLETE (${n_done} done / ${n_fail} fail / ${n_skip} skip) ================"
echo "RUN_MQUAKE_LAW_DONE" >> "$LOG"
