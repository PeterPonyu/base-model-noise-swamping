#!/bin/bash
# run_ripple.sh — RippleEdits geometry battery (2026-07-06). Template = run_mquake_law.sh
# (verbatim skeleton for a THIRD-dataset law-replication driver), RIPPLE-namespaced (own
# pid/log/markers). Calls experiments/ripple_geometry.py (NOT killgate_keygeom.py — a
# separate script, since RippleEdits' probe structure — per-criterion implication banks
# keyed to a SPECIFIC source edit, plus a pooled "unrelated" bank — doesn't fit killgate's
# single flat probe matrix without either forcing a schema RippleEdits doesn't have or
# duplicating killgate's damage-loop machinery under a different probe-selection rule; see
# ripple_geometry.py's module docstring for the full design rationale).
#
# SCIENCE QUESTION: does key geometry predict RippleEdits' RELATED-fact (ripple-implication)
# damage the same way it predicts UNRELATED collateral damage? Two within-probe Spearman
# rhos per cell (ripple vs unrelated), plus RippleEdits' own per-criterion accuracy
# (Logical_Generalization / Compositionality_I/II / Subject_Aliasing / Relation_Specificity
# / Forgetfulness — NOT "Preservation": the actual downloaded data uses "Forgetfulness" in
# that slot, verified against all 3 files, 0 "Preservation" keys anywhere — see
# rippleedits_loader.py's module docstring).
#
# DATA SOURCE: data/rippleedits/popular.json (885 records; random.json/recent.json also
# exist but recent.json has NO original_fact for any record and is out of scope for
# rippleedits_loader's rewrite-diffing method — see that file's docstring). popular.json
# chosen over random.json for the main battery: "popular" entities have richer ripple
# implication banks (more test_queries per criterion in the CPU smoke: e.g.
# Relation_Specificity 272 vs random's 90 at n_edits=50), giving more probes per within-
# probe correlation column.
#
# NO SUBJECT STRING (RippleEdits limitation, not this driver's invention — see
# rippleedits_loader.py): every captured key falls back to "last token of the cloze
# prompt stem" (editors/rome_native.py's pre-existing, documented fallback for
# subject=None). Flagged so nobody reads RippleEdits keys as having CounterFact's
# exact-subject-token fidelity.
set -u
H=/home/zeyufu/Desktop/idea-feasibility-analysis/edit-harness
cd "$H" || exit 1
PY=/home/zeyufu/miniconda3/envs/dl/bin/python3
LOG=engine/run_ripple.log
BUDGET_MIN=${BUDGET_MIN:-180}
mkdir -p engine results/matrices results/smoke_ripple/matrices
echo $$ > engine/run_ripple.pid
[ -f engine/ripple_round_start ] || stat -c %Y engine/run_ripple.pid > engine/ripple_round_start
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
log "================ RUN_RIPPLE START (pid $$, budget ${BUDGET_MIN}m) ================"

# ---------------------------------------------------------------- Phase 0a: CPU pre-flight
pf_fail=0
pf(){ if eval "$2"; then log "preflight OK: $1"; else log "PREFLIGHT-FAIL: $1"; pf_fail=1; fi; }
pf "python env" "$PY -c 'import torch, numpy' 2>/dev/null"
pf "rippleedits_loader.py" "[ -f experiments/rippleedits_loader.py ]"
pf "ripple_geometry.py" "[ -f experiments/ripple_geometry.py ]"
pf "data/rippleedits/popular.json" "[ -f data/rippleedits/popular.json ]"
pf "model Llama-3.2-1B" "[ -d data/models/Llama-3.2-1B ]"
pf "analyze_matrices.py (within_probe_rhos)" "grep -q -- 'def within_probe_rhos' experiments/analyze_matrices.py"
pf "disk >=15GB free" "[ \$(df --output=avail -BG /home | tail -1 | tr -dc 0-9) -ge 15 ]"
rm -f engine/smoke_ripple_*.ok
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

# ---------------------------------------------------------------- helpers (u6/mquake_law template, adapted for ripple_geometry.py)
ENVP="env -u ALL_PROXY -u all_proxy HF_HUB_OFFLINE=1"
RG="experiments/ripple_geometry.py"
POP="--data data/rippleedits/popular.json"
COMMON="--n_edits 200 --n_unrelated_probes 200 --max_probes_per_criterion 40 --steps 20"
SMK="--n_edits 3 --n_unrelated_probes 4 --max_probes_per_criterion 3 --steps 3"
elapsed_min(){ echo $(( ( $(date +%s) - T0 ) / 60 )); }

validate(){
  $PY - "$1" "${2:-full}" <<'EOF'
import json, sys
j, mode = sys.argv[1], sys.argv[2]
try:
    d = json.load(open(j))
except Exception as e:
    print(f"VALIDATE-FAIL json unparseable: {e}"); sys.exit(1)
need = {"within_probe_rho_logit", "per_criterion_accuracy", "edit_success_rate"}
missing = need - set(d.keys())
if missing: print(f"VALIDATE-FAIL json missing keys {missing}"); sys.exit(1)
esr = d.get("edit_success_rate")
# smoke cells run steps=3 on 3-4 edits/probes — esr=0 there is the MEASURED common case
# (confirmed live: the authoring smoke on Qwen2.5-0.5B, steps=3, produced esr=0.0 and a
# fully valid output), same as run_gptj.sh/run_instruct.sh's smoke bypass. Only hard-fail
# esr==0 for SCIENCE-mode cells (real steps=20).
if mode == "full":
    if esr == 0: print("VALIDATE-FAIL esr==0 at real steps"); sys.exit(1)
    if esr is not None and esr < 0.9: print(f"VALIDATE-WARN esr={esr}<0.9")
else:
    if esr is not None and esr < 0.9: print(f"VALIDATE-NOTE smoke esr={esr}")
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
  local pvmode="full"; [ "$class" = "SMOKE" ] && pvmode="smoke"
  if [ -n "$outj" ] && [ -f "$outj" ]; then
    if validate "$outj" "$pvmode" | grep -q VALIDATE-FAIL; then
      mv "$outj" "$outj.INVALID" 2>/dev/null
      log "STALE-INVALID ${tag} — quarantined; re-running"
    else
      log "skip ${tag} (exists, validated)"
      [ "$class" = "SMOKE" ] && : > "engine/smoke_ripple_${tag}.ok"
      return
    fi
  fi
  local mdir; mdir=$(echo "$cmd" | grep -oE -- '--model [^ ]+' | awk '{print $2}')
  case "$mdir" in data/*|/*)
    if [ -n "$mdir" ] && [ ! -d "$mdir" ]; then log "CONFIG-SKIP ${tag} (no model ${mdir})"; n_skip=$((n_skip+1)); return; fi;;
  esac
  local cap=$(( est * 60 * 3 + 1200 ))
  log "RUN ${tag} [${class}] (est ${est}m, cap ${cap}s, elapsed ${now}m) -> engine/ripple_${tag}.log"
  local t rc dt
  t=$(date +%s)
  timeout --signal=TERM --kill-after=60 "${cap}s" bash -c "$cmd" >> "engine/ripple_${tag}.log" 2>&1
  rc=$?; dt=$(( $(date +%s) - t ))
  if [ "$rc" -eq 0 ] && [ -n "$outj" ] && [ -f "$outj" ]; then
    local v; v=$(validate "$outj" "$pvmode")
    if echo "$v" | grep -q VALIDATE-FAIL; then
      mv "$outj" "$outj.INVALID" 2>/dev/null
      log "FAIL ${tag} (${dt}s) OUTPUT-INVALID: ${v}"; n_fail=$((n_fail+1))
    else
      log "done ${tag} (${dt}s) ${v}"; n_done=$((n_done+1)); wedge_fail=0
      [ "$class" = "SMOKE" ] && : > "engine/smoke_ripple_${tag}.ok"
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
# ripple_geometry.py has never run end-to-end against the real downloaded RippleEdits
# files on GPU (CPU-validated only, on Qwen2.5-0.5B, during authoring — see this driver's
# accompanying report).
run_row SMOKE rome_ripple_smoke 4 - "$ENVP $PY $RG --model data/models/Llama-3.2-1B --editor rome $POP $SMK --lr 0.1 --layer 12 --seed 0 --out results/smoke_ripple/rome_ripple.json"
heartbeat

# ---------------------------------------------------------------- Block P: ROME L12 x 3 seeds (popular.json)
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE ripple_llama1b_rome_popular_L12_s0 25 engine/smoke_ripple_rome_ripple_smoke.ok "$ENVP $PY $RG --model data/models/Llama-3.2-1B --editor rome $POP $COMMON --lr 0.1 --layer 12 --seed 0 --out results/ripple_llama1b_rome_popular_L12_s0.json"
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE ripple_llama1b_rome_popular_L12_s1 25 engine/smoke_ripple_rome_ripple_smoke.ok "$ENVP $PY $RG --model data/models/Llama-3.2-1B --editor rome $POP $COMMON --lr 0.1 --layer 12 --seed 1 --out results/ripple_llama1b_rome_popular_L12_s1.json"
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE ripple_llama1b_rome_popular_L12_s2 25 engine/smoke_ripple_rome_ripple_smoke.ok "$ENVP $PY $RG --model data/models/Llama-3.2-1B --editor rome $POP $COMMON --lr 0.1 --layer 12 --seed 2 --out results/ripple_llama1b_rome_popular_L12_s2.json"
heartbeat

# ---------------------------------------------------------------- Block A: AlphaEdit L12 s0 (by-construction reference projector — the loader's own probe/unrelated split IS the preserved set by default; see ripple_geometry.py)
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE ripple_llama1b_alpha_popular_L12_s0 30 engine/smoke_ripple_rome_ripple_smoke.ok "$ENVP $PY $RG --model data/models/Llama-3.2-1B --editor alpha $POP $COMMON --lr 0.1 --layer 12 --seed 0 --out results/ripple_llama1b_alpha_popular_L12_s0.json"
heartbeat

# ---------------------------------------------------------------- Post-run (CPU, ALWAYS)
log "---------------- POST-RUN (CPU) ----------------"
$PY - > results/ripple_validation.json 2>>"$LOG" <<'EOF'
import json, glob, os
t0 = float(open('engine/ripple_round_start').read().strip())
out = []
for j in sorted(glob.glob('results/ripple_llama1b_*.json')):
    if os.path.getmtime(j) < t0: continue
    row = {'json': j}
    try:
        d = json.load(open(j)); row['json_ok'] = True
        row['rho_ripple'] = d.get('within_probe_rho_logit', {}).get('ripple')
        row['rho_unrelated'] = d.get('within_probe_rho_logit', {}).get('unrelated')
        row['esr'] = d.get('edit_success_rate')
        row['per_criterion_accuracy'] = d.get('per_criterion_accuracy')
    except Exception as e:
        row['json_ok'] = False; row['err'] = str(e)
    out.append(row)
print(json.dumps({'n': len(out), 'rows': out}, indent=1))
EOF
log "post: validation sweep -> results/ripple_validation.json"

# 3-seed pooled rho (ripple vs unrelated), same "does geometry predict RELATED damage
# like UNRELATED damage" headline the driver exists to answer.
$PY - >> "$LOG" 2>&1 <<'EOF'
import json, glob
rows = []
for j in sorted(glob.glob('results/ripple_llama1b_rome_popular_L12_s*.json')):
    d = json.load(open(j))
    rows.append((d['within_probe_rho_logit']['ripple'], d['within_probe_rho_logit']['unrelated']))
if rows:
    import statistics as st
    rr = [r[0] for r in rows]; ru = [r[1] for r in rows]
    summary = {'n_seeds': len(rows), 'rho_ripple_mean': st.mean(rr), 'rho_unrelated_mean': st.mean(ru),
              'rho_ripple_per_seed': rr, 'rho_unrelated_per_seed': ru}
    json.dump(summary, open('results/RIPPLE_rho_summary_L12.json', 'w'), indent=1)
    print(f"[ripple post] {summary}")
else:
    print("[ripple post] no ripple_llama1b_rome_popular_L12_s* rows yet — skipping summary")
EOF
log "post: RIPPLE_rho_summary_L12 attempted"

{
  echo "RUN_RIPPLE REPORT $(date '+%F %T')  jobs: ${n_done} done / ${n_fail} fail / ${n_skip} skipped  elapsed $(elapsed_min)m/${BUDGET_MIN}m  abort=${QUEUE_ABORT}"
  grep -E 'RUN |done |FAIL |SKIP|ABORT|PROGRESS|THERMAL' "$LOG" | tail -60
} > engine/run_ripple_report.txt
log "================ RUN_RIPPLE COMPLETE (${n_done} done / ${n_fail} fail / ${n_skip} skip) ================"
echo "RUN_RIPPLE_DONE" >> "$LOG"
