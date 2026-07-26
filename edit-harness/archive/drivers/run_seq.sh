#!/bin/bash
# run_seq.sh — ROUND 5 (SEQUENTIAL-COLLAPSE): the ordering-controlled battery.
# Template = run_r4.sh (preflight/gate/validate/run_row/post-run structure copied verbatim,
# r5-namespaced). Manifest: order_seed micro-smoke -> L12/L8 ordering battery (SAME edit
# SET per selection seed, DIFFERENT insertion order via --order_seed) -> post-run runs
# experiments/seq/analyze_sequential.py over the battery + writes a report.
set -u
H=/home/zeyufu/Desktop/idea-feasibility-analysis/edit-harness
cd "$H" || exit 1
PY=/home/zeyufu/miniconda3/envs/dl/bin/python3
LOG=engine/run_seq.log
BUDGET_MIN=${BUDGET_MIN:-300}
mkdir -p engine results/matrices results/smoke_r5/matrices
echo $$ > engine/run_seq.pid
[ -f engine/r5_round_start ] || stat -c %Y engine/run_seq.pid > engine/r5_round_start
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
log "================ RUN_SEQ (r5) START (pid $$, budget ${BUDGET_MIN}m) ================"

# ---------------------------------------------------------------- Phase 0a: CPU pre-flight
pf_fail=0
pf(){ if eval "$2"; then log "preflight OK: $1"; else log "PREFLIGHT-FAIL: $1"; pf_fail=1; fi; }
pf "python env" "$PY -c 'import torch, numpy' 2>/dev/null"
pf "counterfact.json" "[ -f data/counterfact.json ]"
pf "no_restore flag" "grep -q -- '--no_restore' experiments/killgate_keygeom.py"
pf "order_seed flag" "grep -q -- '--order_seed' experiments/killgate_keygeom.py"
pf "analyze_sequential.py" "[ -f experiments/seq/analyze_sequential.py ]"
pf "model Llama-3.2-1B" "[ -d data/models/Llama-3.2-1B ]"
pf "disk >=20GB free" "[ \$(df --output=avail -BG /home | tail -1 | tr -dc 0-9) -ge 20 ]"
# fail-closed marker lifecycle (r3/r4 review HIGH #1)
rm -f engine/smoke_r5_*.ok
if [ "$pf_fail" -ne 0 ]; then log "ABORT: preflight failed"; exit 3; fi

# ---------------------------------------------------------------- Phase 0b: GPU idle gate
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
T0=$(date +%s)

# ---------------------------------------------------------------- helpers (r4 template, verbatim)
ENVP="env -u ALL_PROXY -u all_proxy HF_HUB_OFFLINE=1"
KG="experiments/killgate_keygeom.py"
CF="--dataset counterfact --data data/counterfact.json"
SEQCOMMON="--n_edits 50 --n_probes 500 --steps 20 --save_matrices --matrix_dir results/matrices --recheck_every 10"
SMK="--n_edits 4 --n_probes 40 --steps 2 --recheck_every 2 --save_matrices --matrix_dir results/smoke_r5/matrices"
elapsed_min(){ echo $(( ( $(date +%s) - T0 ) / 60 )); }

validate(){
  $PY - "$1" "$2" "${3:-full}" <<'EOF'
import json, sys, numpy as np
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
    # SEQ probe_stride writes intentional NaN rows; all-NaN is still invalid
    if np.isnan(arr).all():
        print(f"VALIDATE-FAIL {k} all-NaN"); sys.exit(1)
# seq cells: the prior-edit overwrite panel is the H1 science — gate its presence
if "seq_no_restore" in a.files and int(a["seq_no_restore"]) == 1:
    if "prior_eff" not in a.files or np.isnan(a["prior_eff"].astype(float)).all():
        print("VALIDATE-FAIL seq cell missing/all-NaN prior_eff panel"); sys.exit(1)
    if "GRAM_pre" not in a.files:
        print("VALIDATE-FAIL seq cell missing GRAM_pre"); sys.exit(1)
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
  case "$cmd" in *smoke_r5*) outn="results/smoke_r5/matrices/$(basename "${outj%.json}").npz";; esac
  if [ -n "$outj" ] && [ -f "$outj" ] && [ -f "$outn" ]; then
    local pvmode="full"; [ "$class" = "SMOKE" ] && pvmode="smoke"
    if validate "$outj" "$outn" "$pvmode" | grep -q VALIDATE-FAIL; then
      mv "$outj" "$outj.INVALID" 2>/dev/null; mv "$outn" "$outn.INVALID" 2>/dev/null
      log "STALE-INVALID ${tag} — quarantined; re-running"
    else
      log "skip ${tag} (exists, validated)"
      [ "$class" = "SMOKE" ] && : > "engine/smoke_r5_${tag}.ok"
      return
    fi
  fi
  local mdir; mdir=$(echo "$cmd" | grep -oE -- '--model [^ ]+' | awk '{print $2}')
  case "$mdir" in data/*|/*)
    if [ -n "$mdir" ] && [ ! -d "$mdir" ]; then log "CONFIG-SKIP ${tag} (no model ${mdir})"; n_skip=$((n_skip+1)); return; fi;;
  esac
  local cap=$(( est * 60 * 3 + 1200 ))
  log "RUN ${tag} [${class}] (est ${est}m, cap ${cap}s, elapsed ${now}m) -> engine/r5_${tag}.log"
  local t rc dt
  t=$(date +%s)
  timeout --signal=TERM --kill-after=60 "${cap}s" bash -c "$cmd" >> "engine/r5_${tag}.log" 2>&1
  rc=$?; dt=$(( $(date +%s) - t ))
  local vmode="full"; [ "$class" = "SMOKE" ] && vmode="smoke"
  if [ "$rc" -eq 0 ] && [ -n "$outj" ] && [ -f "$outj" ] && [ -f "$outn" ]; then
    local v; v=$(validate "$outj" "$outn" "$vmode")
    if echo "$v" | grep -q VALIDATE-FAIL; then
      mv "$outj" "$outj.INVALID" 2>/dev/null; mv "$outn" "$outn.INVALID" 2>/dev/null
      log "FAIL ${tag} (${dt}s) OUTPUT-INVALID: ${v}"; n_fail=$((n_fail+1))
    else
      log "done ${tag} (${dt}s) ${v}"; n_done=$((n_done+1)); wedge_fail=0
      [ "$class" = "SMOKE" ] && : > "engine/smoke_r5_${tag}.ok"
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

# ---------------------------------------------------------------- Phase 0c: micro-smoke (never-run path)
run_row SMOKE ord_smoke 4 - "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor rome --no_restore --order_seed 1 $CF $SMK --lr 0.1 --layer 12 --seed 0 --out results/smoke_r5/ord_smoke.json"
heartbeat

# ---------------------------------------------------------------- Block A: L12 ordering battery
# selection seed 0 x order_seed {0,1,2,3,4} — SAME edit set, 5 different insertion orders.
for os in 0 1 2 3 4; do
  [ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE seq_ord_llama1b_L12_s0_o${os} 10 engine/smoke_r5_ord_smoke.ok \
    "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor rome --no_restore --order_seed ${os} $CF $SEQCOMMON --lr 0.1 --layer 12 --seed 0 --out results/seq_ord_llama1b_L12_s0_o${os}.json"
done
heartbeat

# selection seed 1 x order_seed {0,1,2} — second selection seed, ordering-controlled again.
for os in 0 1 2; do
  [ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE seq_ord_llama1b_L12_s1_o${os} 10 engine/smoke_r5_ord_smoke.ok \
    "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor rome --no_restore --order_seed ${os} $CF $SEQCOMMON --lr 0.1 --layer 12 --seed 1 --out results/seq_ord_llama1b_L12_s1_o${os}.json"
done
heartbeat

# ---------------------------------------------------------------- Block B: L8 ordering battery (cross-layer check)
# selection seed 0 x order_seed {0,1} at L8 — does the ordering effect replicate off L12?
for os in 0 1; do
  [ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE seq_ord_llama1b_L8_s0_o${os} 10 engine/smoke_r5_ord_smoke.ok \
    "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor rome --no_restore --order_seed ${os} $CF $SEQCOMMON --lr 0.1 --layer 8 --seed 0 --out results/seq_ord_llama1b_L8_s0_o${os}.json"
done
heartbeat

# ---------------------------------------------------------------- Post-run (CPU, ALWAYS)
log "---------------- POST-RUN (CPU) ----------------"
$PY - > results/r5_validation.json 2>>"$LOG" <<'EOF'
import json, glob, os, numpy as np
t0 = float(open('engine/r5_round_start').read().strip())
out = []
for j in sorted(glob.glob('results/seq_ord_*.json')):
    base = os.path.basename(j)[:-5]
    if os.path.getmtime(j) < t0: continue
    z = 'results/matrices/' + base + '.npz'
    row = {'json': j, 'npz_found': os.path.exists(z)}
    try:
        d = json.load(open(j)); row['json_ok'] = True; row['esr'] = d.get('edit_success_rate')
        row['order_seed'] = d.get('order_seed')
    except Exception as e:
        row['json_ok'] = False; row['err'] = str(e)
    if row['npz_found']:
        a = np.load(z)
        row['npz_keys_ok'] = {'COS','damage_logit','norm_growth','edit_ok','GRAM_pre','prior_eff'} <= set(a.files)
        row['all_nan'] = bool(np.isnan(a['COS'].astype(float)).all())
    out.append(row)
print(json.dumps({'n': len(out), 'rows': out}, indent=1))
EOF
log "post: validation sweep -> results/r5_validation.json"

if compgen -G "results/matrices/seq_ord_llama1b_L12_s*.npz" >/dev/null; then
  $PY experiments/seq/analyze_sequential.py "results/matrices/seq_ord_llama1b_L12_s*.npz" \
    --out results/seq_ordering_battery_L12_r5.json >> "$LOG" 2>&1 \
    && log "post: seq_ordering_battery_L12_r5 done" || log "post: seq_ordering_battery_L12_r5 FAIL"
fi
if compgen -G "results/matrices/seq_ord_llama1b_L8_s*.npz" >/dev/null; then
  $PY experiments/seq/analyze_sequential.py "results/matrices/seq_ord_llama1b_L8_s*.npz" \
    --out results/seq_ordering_battery_L8_r5.json >> "$LOG" 2>&1 \
    && log "post: seq_ordering_battery_L8_r5 done" || log "post: seq_ordering_battery_L8_r5 FAIL"
fi
if compgen -G "results/matrices/seq_ord_llama1b_L*_s*.npz" >/dev/null; then
  $PY experiments/seq/analyze_sequential.py "results/matrices/seq_ord_llama1b_L*_s*.npz" \
    --out results/seq_ordering_battery_ALL_r5.json >> "$LOG" 2>&1 \
    && log "post: seq_ordering_battery_ALL_r5 done" || log "post: seq_ordering_battery_ALL_r5 FAIL"
fi

{
  echo "RUN_SEQ (r5) REPORT $(date '+%F %T')  jobs: ${n_done} done / ${n_fail} fail / ${n_skip} skipped  elapsed $(elapsed_min)m/${BUDGET_MIN}m  abort=${QUEUE_ABORT}"
  grep -E 'RUN |done |FAIL |SKIP|ABORT|PROGRESS' "$LOG" | tail -60
} > engine/run_seq_report.txt
log "================ RUN_SEQ COMPLETE (${n_done} done / ${n_fail} fail / ${n_skip} skip) ================"
echo "RUN_SEQ_DONE" >> "$LOG"
