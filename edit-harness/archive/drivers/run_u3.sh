#!/bin/bash
# run_u3.sh — L14 raw-key vector dumps for Phase 2.4 anisotropy (2026-07-04). Template =
# run_u2.sh (verbatim preflight/idle-gate/validate/run_row/heartbeat skeleton), U3-namespaced.
# WHY: the banked aniso L14 dumps (results/aniso_*_L14) hold only 7 per-edit SCALARS —
# mechanism_dump.py never saves raw keys — so the Phase 2.4 anisotropy/whitening analysis
# is data-blocked at L14 (aniso-analysis agent STOP report, 07-04). The only raw-key banks
# are killgate --save_vectors at Llama L8/L12. This driver produces the two missing L14
# banks via the PROVEN qv_ row config (run_8h.sh:187-188, measured ~1660s each on Llama):
#   qv_llama1b_rome_cf_L14_s0  + qv_qwen15b_rome_cf_L14_s0
# save_vectors has never run on Qwen — one micro-smoke gates that cell (house rule:
# never-run config paths get smoked first). NOTE: probe keys are saved by NO current code
# path; the matched-probe arm of the analysis stays data-unavailable by design here.
set -u
H=/home/zeyufu/Desktop/idea-feasibility-analysis/edit-harness
cd "$H" || exit 1
PY=/home/zeyufu/miniconda3/envs/dl/bin/python3
LOG=engine/run_u3.log
BUDGET_MIN=${BUDGET_MIN:-120}
mkdir -p engine results/matrices results/vectors results/smoke_u3/matrices
echo $$ > engine/run_u3.pid
[ -f engine/u3_round_start ] || stat -c %Y engine/run_u3.pid > engine/u3_round_start
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
log "================ RUN_U3 START (pid $$, budget ${BUDGET_MIN}m) ================"

# ---------------------------------------------------------------- Phase 0a: CPU pre-flight
pf_fail=0
pf(){ if eval "$2"; then log "preflight OK: $1"; else log "PREFLIGHT-FAIL: $1"; pf_fail=1; fi; }
pf "python env" "$PY -c 'import torch, numpy' 2>/dev/null"
pf "counterfact.json" "[ -f data/counterfact.json ]"
pf "save_vectors flag" "grep -q -- '--save_vectors' experiments/killgate_keygeom.py"
pf "model Llama-3.2-1B" "[ -d data/models/Llama-3.2-1B ]"
pf "model Qwen2.5-1.5B" "[ -d data/models/Qwen2.5-1.5B ]"
pf "disk >=20GB free" "[ \$(df --output=avail -BG /home | tail -1 | tr -dc 0-9) -ge 20 ]"
rm -f engine/smoke_u3_*.ok
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

# ---------------------------------------------------------------- helpers (u2 template, verbatim)
ENVP="env -u ALL_PROXY -u all_proxy HF_HUB_OFFLINE=1"
KG="experiments/killgate_keygeom.py"
CF="--dataset counterfact --data data/counterfact.json"
COMMON="--n_edits 200 --n_probes 500 --steps 20 --save_matrices --matrix_dir results/matrices"
SMK="--n_edits 4 --n_probes 40 --steps 2 --save_matrices --matrix_dir results/smoke_u3/matrices"
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
  case "$cmd" in *smoke_u3*) outn="results/smoke_u3/matrices/$(basename "${outj%.json}").npz";; esac
  if [ -n "$outj" ] && [ -f "$outj" ] && [ -f "$outn" ]; then
    local pvmode="full"; [ "$class" = "SMOKE" ] && pvmode="smoke"
    if validate "$outj" "$outn" "$pvmode" | grep -q VALIDATE-FAIL; then
      mv "$outj" "$outj.INVALID" 2>/dev/null; mv "$outn" "$outn.INVALID" 2>/dev/null
      log "STALE-INVALID ${tag} — quarantined; re-running"
    else
      log "skip ${tag} (exists, validated)"
      [ "$class" = "SMOKE" ] && : > "engine/smoke_u3_${tag}.ok"
      return
    fi
  fi
  local mdir; mdir=$(echo "$cmd" | grep -oE -- '--model [^ ]+' | awk '{print $2}')
  case "$mdir" in data/*|/*)
    if [ -n "$mdir" ] && [ ! -d "$mdir" ]; then log "CONFIG-SKIP ${tag} (no model ${mdir})"; n_skip=$((n_skip+1)); return; fi;;
  esac
  local cap=$(( est * 60 * 3 + 1200 ))
  log "RUN ${tag} [${class}] (est ${est}m, cap ${cap}s, elapsed ${now}m) -> engine/u3_${tag}.log"
  local t rc dt
  t=$(date +%s)
  timeout --signal=TERM --kill-after=60 "${cap}s" bash -c "$cmd" >> "engine/u3_${tag}.log" 2>&1
  rc=$?; dt=$(( $(date +%s) - t ))
  local vmode="full"; [ "$class" = "SMOKE" ] && vmode="smoke"
  if [ "$rc" -eq 0 ] && [ -n "$outj" ] && [ -f "$outj" ] && [ -f "$outn" ]; then
    local v; v=$(validate "$outj" "$outn" "$vmode")
    if echo "$v" | grep -q VALIDATE-FAIL; then
      mv "$outj" "$outj.INVALID" 2>/dev/null; mv "$outn" "$outn.INVALID" 2>/dev/null
      log "FAIL ${tag} (${dt}s) OUTPUT-INVALID: ${v}"; n_fail=$((n_fail+1))
    else
      log "done ${tag} (${dt}s) ${v}"; n_done=$((n_done+1)); wedge_fail=0
      [ "$class" = "SMOKE" ] && : > "engine/smoke_u3_${tag}.ok"
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
# save_vectors has NEVER run on Qwen (only Llama L8/L12) — smoke that path. The Llama
# save_vectors path is proven at science scale (run8h qv_ cells) — no smoke needed.
run_row SMOKE savevec_qwen_L14 4 - "$ENVP $PY $KG --model data/models/Qwen2.5-1.5B --editor rome --save_vectors --vector_dir results/smoke_u3/vectors $CF $SMK --lr 0.1 --layer 14 --seed 0 --out results/smoke_u3/savevec_qwen_L14.json"
heartbeat

# ---------------------------------------------------------------- Science: L14 raw-key banks
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE qv_llama1b_rome_cf_L14_s0 30 - "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor rome --save_vectors $CF $COMMON --lr 0.1 --layer 14 --seed 0 --out results/qv_llama1b_rome_cf_L14_s0.json"
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE qv_qwen15b_rome_cf_L14_s0 38 engine/smoke_u3_savevec_qwen_L14.ok "$ENVP $PY $KG --model data/models/Qwen2.5-1.5B --editor rome --save_vectors $CF $COMMON --lr 0.1 --layer 14 --seed 0 --out results/qv_qwen15b_rome_cf_L14_s0.json"
heartbeat

# ---------------------------------------------------------------- Post-run (CPU): vector-bank check
log "---------------- POST-RUN (CPU) ----------------"
$PY - > results/u3_validation.json 2>>"$LOG" <<'EOF'
import json, glob, os, numpy as np
out = []
for z in sorted(glob.glob('results/vectors/vectors_qv_*_L14_s0.npz')):
    a = np.load(z)
    row = {'npz': z,
           'K_shape': list(a['K'].shape) if 'K' in a.files else None,
           'vectors_valid': bool(a['vectors_valid']) if 'vectors_valid' in a.files else None,
           'recon_rel_err_max': float(a['recon_rel_err'].max()) if 'recon_rel_err' in a.files else None,
           'K_all_finite': bool(np.isfinite(a['K']).all()) if 'K' in a.files else None}
    out.append(row)
print(json.dumps({'n': len(out), 'rows': out}, indent=1))
EOF
log "post: vector-bank sweep -> results/u3_validation.json"

{
  echo "RUN_U3 REPORT $(date '+%F %T')  jobs: ${n_done} done / ${n_fail} fail / ${n_skip} skipped  elapsed $(elapsed_min)m/${BUDGET_MIN}m  abort=${QUEUE_ABORT}"
  grep -E 'RUN |done |FAIL |SKIP|ABORT|PROGRESS' "$LOG" | tail -30
} > engine/run_u3_report.txt
log "================ RUN_U3 COMPLETE (${n_done} done / ${n_fail} fail / ${n_skip} skip) ================"
echo "RUN_U3_DONE" >> "$LOG"
