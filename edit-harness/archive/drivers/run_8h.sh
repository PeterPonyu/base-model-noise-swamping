#!/bin/bash
# run_8h.sh — 8-hour budgeted serial GPU window (2026-07-02). See PLAN-8H-RUN-2026-07-02.md.
# Design-against-failure: measured ests + 3x caps; smoke-gated new code; config-fail never
# aborts; per-job output validation; zero-risk filler; polite idle-gate abort; atomic skips.
set -u
H=/home/zeyufu/Desktop/idea-feasibility-analysis/edit-harness
cd "$H" || exit 1
PY=/home/zeyufu/miniconda3/envs/dl/bin/python3
LOG=engine/run8h.log
BUDGET_MIN=${BUDGET_MIN:-480}
mkdir -p engine results/matrices results/smoke8h/matrices
echo $$ > engine/run8h.pid
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
log "================ RUN8H START (pid $$, budget ${BUDGET_MIN}m) ================"

# ---------------------------------------------------------------- Phase 0a: CPU pre-flight
pf_fail=0
pf(){ if eval "$2"; then log "preflight OK: $1"; else log "PREFLIGHT-FAIL: $1"; pf_fail=1; fi; }
pf "python env (torch+numpy import)" "$PY -c 'import torch, numpy' 2>/dev/null"
for m in Qwen2.5-0.5B Qwen2.5-1.5B Qwen2.5-3B gemma-2-2b Phi-3.5-mini Llama-3.2-1B Llama-3.2-3B; do
  pf "model $m" "[ -d data/models/$m ]"
done
pf "counterfact.json" "[ -f data/counterfact.json ]"
pf "editors/memit.py" "[ -f experiments/editors/memit.py ] || [ -f editors/memit.py ]"
for fl in save_vectors alpha_proj_source ft_kl holdout_frac; do
  pf "killgate flag --$fl" "grep -q -- \"--$fl\" experiments/killgate_keygeom.py"
done
pf "disk >=50GB free" "[ \$(df --output=avail -BG /home | tail -1 | tr -dc 0-9) -ge 50 ]"
if [ "$pf_fail" -ne 0 ]; then log "ABORT: preflight failed — nothing was run"; exit 3; fi

# ---------------------------------------------------------------- Phase 0b: GPU idle gate
# util<25 AND mem<1500MiB for 3 consecutive 30s polls; ABORT after 30min instead of jamming.
gate_t0=$(date +%s); consec=0
while [ "$consec" -lt 3 ]; do
  line=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  util=$(echo "$line" | awk -F, '{gsub(/[^0-9]/,"",$1); print $1}')
  mem=$(echo "$line"  | awk -F, '{gsub(/[^0-9]/,"",$2); print $2}')
  if [ -n "$util" ] && [ -n "$mem" ] && [ "$util" -lt 25 ] && [ "$mem" -lt 1500 ]; then
    consec=$((consec+1))
  else
    consec=0
    if [ $(( $(date +%s) - gate_t0 )) -gt 1800 ]; then
      log "ABORT: GPU busy >30min at gate (util=${util:-NA} mem=${mem:-NA}) — window yielded"; exit 2
    fi
  fi
  log "gpu poll util=${util:-NA} mem=${mem:-NA} consec=${consec}/3"
  [ "$consec" -lt 3 ] && sleep 30
done
log "GPU idle — window opens now"
T0=$(date +%s)   # budget starts when the GPU is actually ours

# ---------------------------------------------------------------- helpers
ENVP="env -u ALL_PROXY -u all_proxy HF_HUB_OFFLINE=1"
KG="experiments/killgate_keygeom.py"
CF="--dataset counterfact --data data/counterfact.json"
COMMON="--n_edits 200 --n_probes 500 --steps 20 --save_matrices --matrix_dir results/matrices"
SMK="--n_edits 4 --n_probes 40 --steps 2 --save_matrices --matrix_dir results/smoke8h/matrices"

elapsed_min(){ echo $(( ( $(date +%s) - T0 ) / 60 )); }

# validate <json> <npz> <mode>: parses json, npz keys, no all-NaN; esr gates only in "full"
# mode — SMOKE rows run at steps=2 where esr==0 is EXPECTED (review fix 2026-07-02 #1).
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
    if np.isnan(a[k].astype(float)).all():
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

# run_row <class> <tag> <est_min> <needs> <cmd...>
#   class: SMOKE|SCIENCE|FILLER. needs: "-" or comma list of marker files that must exist.
run_row(){
  local class="$1" tag="$2" est="$3" needs="$4"; shift 4; local cmd="$*"
  local now; now=$(elapsed_min)
  # budget admission
  if [ $(( now + est + 2 )) -gt "$BUDGET_MIN" ]; then
    log "BUDGET-SKIP ${tag} (elapsed ${now}m + est ${est}m > ${BUDGET_MIN}m)"; n_skip=$((n_skip+1)); return; fi
  # smoke-marker gating (the code-bug firewall)
  if [ "$needs" != "-" ]; then
    local mk; for mk in ${needs//,/ }; do
      if [ ! -f "$mk" ]; then log "CONFIG-SKIP ${tag} (missing gate marker ${mk})"; n_skip=$((n_skip+1)); return; fi
    done
  fi
  # idempotent skip on committed artifacts (json is the commit marker, written last+atomic)
  local outj outn
  outj=$(echo "$cmd" | grep -oE -- '--out [^ ]+' | awk '{print $2}')
  outn="results/matrices/$(basename "${outj%.json}").npz"
  case "$cmd" in *smoke8h*) outn="results/smoke8h/matrices/$(basename "${outj%.json}").npz";; esac
  if [ -n "$outj" ] && [ -f "$outj" ] && [ -f "$outn" ]; then
    local pvmode="full"; [ "$class" = "SMOKE" ] && pvmode="smoke"
    if validate "$outj" "$outn" "$pvmode" | grep -q VALIDATE-FAIL; then
      mv "$outj" "$outj.INVALID" 2>/dev/null; mv "$outn" "$outn.INVALID" 2>/dev/null
      log "STALE-INVALID ${tag} — pre-existing pair failed validation, quarantined; re-running"
    else
      log "skip ${tag} (exists, validated)"
      [ "$class" = "SMOKE" ] && : > "engine/smoke8h_${tag}.ok"
      return
    fi
  fi
  # model precheck = config skip, not failure
  local mdir; mdir=$(echo "$cmd" | grep -oE -- '--model [^ ]+' | awk '{print $2}')
  if [ -n "$mdir" ] && [ ! -d "$mdir" ]; then log "CONFIG-SKIP ${tag} (no model ${mdir})"; n_skip=$((n_skip+1)); return; fi
  # run with 3x cap
  local cap=$(( est * 60 * 3 + 1200 ))
  log "RUN ${tag} [${class}] (est ${est}m, cap ${cap}s, elapsed ${now}m) -> engine/run8h_${tag}.log"
  local t rc dt
  t=$(date +%s)
  timeout --signal=TERM --kill-after=60 "${cap}s" bash -c "$cmd" >> "engine/run8h_${tag}.log" 2>&1
  rc=$?; dt=$(( $(date +%s) - t ))
  local vmode="full"; [ "$class" = "SMOKE" ] && vmode="smoke"
  if [ "$rc" -eq 0 ] && [ -n "$outj" ] && [ -f "$outj" ] && [ -f "$outn" ]; then
    local v; v=$(validate "$outj" "$outn" "$vmode")
    if echo "$v" | grep -q VALIDATE-FAIL; then
      mv "$outj" "$outj.INVALID" 2>/dev/null; mv "$outn" "$outn.INVALID" 2>/dev/null
      log "FAIL ${tag} (${dt}s) OUTPUT-INVALID: ${v} — artifacts quarantined"; n_fail=$((n_fail+1))
    else
      log "done ${tag} (${dt}s) ${v}"; n_done=$((n_done+1)); wedge_fail=0
      [ "$class" = "SMOKE" ] && : > "engine/smoke8h_${tag}.ok"
    fi
  else
    if [ "$rc" -eq 124 ] || [ "$dt" -ge $(( est * 60 / 2 )) ]; then
      wedge_fail=$((wedge_fail+1)); n_fail=$((n_fail+1))
      log "FAIL ${tag} (rc ${rc}, ${dt}s) WEDGE-LIKE consec=${wedge_fail}/${MAXWEDGE}"
      if [ "$wedge_fail" -ge "$MAXWEDGE" ]; then log "ABORT: ${MAXWEDGE} consecutive wedge-like failures"; QUEUE_ABORT=1; fi
    else
      n_fail=$((n_fail+1))
      log "FAIL ${tag} (rc ${rc}, ${dt}s) FAST/CONFIG — not counted toward wedge abort"
    fi
  fi
}

heartbeat(){ log "PROGRESS jobs=${n_done}done/${n_fail}fail/${n_skip}skip elapsed=$(elapsed_min)m budget_left=$(( BUDGET_MIN - $(elapsed_min) ))m"; }

QUEUE_ABORT=0
# ---------------------------------------------------------------- Phase 0c: GPU micro-smokes
run_row SMOKE memit_L12   4 - "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor memit $CF $SMK --lr 0.1 --layer 12 --seed 0 --out results/smoke8h/memit_L12.json"
run_row SMOKE savevec_L12 4 - "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor rome --save_vectors $CF $SMK --lr 0.1 --layer 12 --seed 0 --out results/smoke8h/savevec_L12.json"
run_row SMOKE alphaphi_L16 5 - "$ENVP $PY $KG --model data/models/Phi-3.5-mini --editor alpha $CF $SMK --lr 0.1 --layer 16 --seed 0 --out results/smoke8h/alphaphi_L16.json"
heartbeat

# ---------------------------------------------------------------- Phase B1: router + aniso
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE g4_qwen05b_alpha_cf_L12_s0 20 - "$ENVP $PY $KG --model data/models/Qwen2.5-0.5B --editor alpha $CF $COMMON --lr 0.1 --layer 12 --seed 0 --out results/g4_qwen05b_alpha_cf_L12_s0.json"
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE g4_qwen15b_alpha_cf_L14_s0 35 - "$ENVP $PY $KG --model data/models/Qwen2.5-1.5B --editor alpha $CF $COMMON --lr 0.1 --layer 14 --seed 0 --out results/g4_qwen15b_alpha_cf_L14_s0.json"
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE g4_phi35_alpha_cf_L16_s0 62 engine/smoke8h_alphaphi_L16.ok "$ENVP $PY $KG --model data/models/Phi-3.5-mini --editor alpha $CF $COMMON --lr 0.1 --layer 16 --seed 0 --out results/g4_phi35_alpha_cf_L16_s0.json"
if [ "$QUEUE_ABORT" -eq 0 ]; then
  for spec in "llama1b:Llama-3.2-1B:14" "qwen15b:Qwen2.5-1.5B:14"; do
    IFS=':' read -r nm md ly <<< "$spec"
    if [ -d "results/aniso_${nm}_L${ly}" ] && [ -n "$(ls -A results/aniso_${nm}_L${ly} 2>/dev/null)" ]; then
      log "skip aniso_dump_${nm}_L${ly} (exists)"
    elif [ $(( $(elapsed_min) + 10 )) -le "$BUDGET_MIN" ]; then
      log "RUN aniso_dump_${nm}_L${ly} (mechanism_dump)"
      timeout 1800 $ENVP $PY experiments/mechanism_dump.py --model data/models/$md --layer $ly \
        --dataset counterfact --data data/counterfact.json --n_edits 50 --n_probes 100 --steps 20 --seed 0 \
        --out_dir results/aniso_${nm}_L${ly} >> engine/run8h_aniso_${nm}.log 2>&1 \
        && { log "done aniso_dump_${nm}_L${ly}"; n_done=$((n_done+1)); } \
        || { log "FAIL aniso_dump_${nm}_L${ly} (non-fatal)"; n_fail=$((n_fail+1)); }
    else log "BUDGET-SKIP aniso_dump_${nm}"; n_skip=$((n_skip+1)); fi
  done
fi
heartbeat

# ---------------------------------------------------------------- Phase B2: MEMIT + QuantEdit-E0
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE gate_llama1b_memit_cf_L12_s0 30 engine/smoke8h_memit_L12.ok "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor memit $CF $COMMON --lr 0.1 --layer 12 --seed 0 --out results/gate_llama1b_memit_cf_L12_s0.json"
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE gate_llama1b_memit_cf_L8_s0 30 engine/smoke8h_memit_L12.ok "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor memit $CF $COMMON --lr 0.1 --layer 8 --seed 0 --out results/gate_llama1b_memit_cf_L8_s0.json"
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE qv_llama1b_rome_cf_L8_s0 20 engine/smoke8h_savevec_L12.ok "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor rome --save_vectors $CF $COMMON --lr 0.1 --layer 8 --seed 0 --out results/qv_llama1b_rome_cf_L8_s0.json"
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE qv_llama1b_rome_cf_L12_s0 20 engine/smoke8h_savevec_L12.ok "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor rome --save_vectors $CF $COMMON --lr 0.1 --layer 12 --seed 0 --out results/qv_llama1b_rome_cf_L12_s0.json"
heartbeat

# ---------------------------------------------------------------- Phase C: zero-risk seed hardening
FILLER=(
  "gate_gemma2b_rome_cf_L13_s1|52|--model data/models/gemma-2-2b --editor rome $CF $COMMON --lr 0.1 --layer 13 --seed 1"
  "gate_gemma2b_rome_cf_L13_s2|52|--model data/models/gemma-2-2b --editor rome $CF $COMMON --lr 0.1 --layer 13 --seed 2"
  "gate_phi35_rome_cf_L16_s1|57|--model data/models/Phi-3.5-mini --editor rome $CF $COMMON --lr 0.1 --layer 16 --seed 1"
  "gate_qwen3b_rome_cf_L18_s1|50|--model data/models/Qwen2.5-3B --editor rome $CF $COMMON --lr 0.1 --layer 18 --seed 1"
  "g4_llama1b_alphaHO_cf_L12_s1|22|--model data/models/Llama-3.2-1B --editor alpha $CF $COMMON --lr 0.1 --layer 12 --seed 1 --alpha_proj_source holdout --holdout_frac 1.0"
  "g4_llama1b_alphaHO_cf_L8_s1|22|--model data/models/Llama-3.2-1B --editor alpha $CF $COMMON --lr 0.1 --layer 8 --seed 1 --alpha_proj_source holdout --holdout_frac 1.0"
  "gate_llama1b_ftkl_cf_L8_s1|25|--model data/models/Llama-3.2-1B --editor ft $CF $COMMON --ft_lr 5e-3 --ft_kl 0.1 --ft_kl_n 5 --layer 8 --seed 1"
  "gate_phi35_rome_cf_L16_s2|57|--model data/models/Phi-3.5-mini --editor rome $CF $COMMON --lr 0.1 --layer 16 --seed 2"
  "gate_qwen3b_rome_cf_L18_s2|50|--model data/models/Qwen2.5-3B --editor rome $CF $COMMON --lr 0.1 --layer 18 --seed 2"
  "g4_llama1b_alphaHO_cf_L12_s2|22|--model data/models/Llama-3.2-1B --editor alpha $CF $COMMON --lr 0.1 --layer 12 --seed 2 --alpha_proj_source holdout --holdout_frac 1.0"
  "g4_llama1b_alphaHO_cf_L8_s2|22|--model data/models/Llama-3.2-1B --editor alpha $CF $COMMON --lr 0.1 --layer 8 --seed 2 --alpha_proj_source holdout --holdout_frac 1.0"
  "gate_llama1b_ftkl_cf_L8_s2|25|--model data/models/Llama-3.2-1B --editor ft $CF $COMMON --ft_lr 5e-3 --ft_kl 0.1 --ft_kl_n 5 --layer 8 --seed 2"
  "gate_llama1b_ft_cf_L10_s1|25|--model data/models/Llama-3.2-1B --editor ft $CF $COMMON --ft_lr 5e-3 --layer 10 --seed 1"
  "gate_llama1b_ft_cf_L12_s1|25|--model data/models/Llama-3.2-1B --editor ft $CF $COMMON --ft_lr 5e-3 --layer 12 --seed 1"
)
for row in "${FILLER[@]}"; do
  [ "$QUEUE_ABORT" -ne 0 ] && break
  IFS='|' read -r tag est args <<< "$row"
  run_row FILLER "$tag" "$est" - "$ENVP $PY $KG $args --out results/${tag}.json"
  heartbeat
done

# ---------------------------------------------------------------- Post-run CPU pass (ALWAYS)
log "---------------- POST-RUN (CPU) ----------------"
# D3 router re-eval with the new non-Llama alpha cells
if compgen -G "results/matrices/g4_qwen*_alpha_cf_*_s0.npz" >/dev/null || compgen -G "results/matrices/g4_phi35_alpha_cf_*_s0.npz" >/dev/null; then
  $PY experiments/geometry_router.py --gate_glob 'results/matrices/gate_*_rome_cf_*_s0.npz' \
    --alpha_glob 'results/matrices/g4_*_alpha*_cf_*_s0.npz' --cos_threshold 0.05 --known \
    --out results/D3_routing_eval_v2.json >> "$LOG" 2>&1 && log "post: D3 v2 done" || log "post: D3 v2 FAIL"
fi
# C3 nulls for any group that reached >=2 seeds this window
for spec in "C3_null_gemma2b_L13:results/matrices/gate_gemma2b_rome_cf_L13_s*.npz" \
            "C3_null_phi35_L16:results/matrices/gate_phi35_rome_cf_L16_s*.npz" \
            "C3_null_qwen3b_L18:results/matrices/gate_qwen3b_rome_cf_L18_s*.npz" \
            "C3_null_ftkl_L8:results/matrices/gate_llama1b_ftkl_cf_L8_s*.npz" \
            "C3_null_memit:results/matrices/gate_llama1b_memit_cf_L*_s0.npz"; do
  outn="${spec%%:*}"; glob="${spec#*:}"
  if compgen -G "$glob" >/dev/null; then
    $PY experiments/analyze_matrices.py $glob --metric logit --known --edit_ok \
      --out "results/${outn}_v2.json" >> "$LOG" 2>&1 && log "post: ${outn}_v2 done" || log "post: ${outn}_v2 FAIL"
  fi
done
# validation sweep of every artifact this window produced (json mtime > window start)
$PY - > results/run8h_validation.json 2>>"$LOG" <<'EOF'
import json, glob, os, numpy as np
t0 = os.path.getmtime('engine/run8h.pid')
out = []
for j in glob.glob('results/*.json') + glob.glob('results/smoke8h/*.json'):
    if os.path.getmtime(j) < t0 or j.endswith('run8h_validation.json'): continue
    base = os.path.basename(j)[:-5]
    z = ('results/smoke8h/matrices/' if 'smoke8h' in j else 'results/matrices/') + base + '.npz'
    row = {'json': j, 'npz_found': os.path.exists(z)}
    try:
        d = json.load(open(j)); row['json_ok'] = True; row['esr'] = d.get('edit_success_rate')
    except Exception as e:
        row['json_ok'] = False; row['err'] = str(e)
    if row['npz_found']:
        try:
            a = np.load(z)
            row['npz_keys_ok'] = {'COS','damage_logit','norm_growth','edit_ok'} <= set(a.files)
            row['all_nan'] = bool(np.isnan(a['COS'].astype(float)).all()) if 'COS' in a.files else None
        except Exception as e:
            row['npz_keys_ok'] = False; row['err_npz'] = str(e)
    out.append(row)
print(json.dumps({'n': len(out), 'rows': out}, indent=1))
EOF
log "post: validation sweep -> results/run8h_validation.json"
# MEMIT vs ROME S x C comparison (only if MEMIT cells landed)
if compgen -G "results/matrices/gate_llama1b_memit_cf_L*_s0.npz" >/dev/null; then
  $PY experiments/mechanism_sc_table.py \
    --npz 'results/matrices/gate_llama1b_memit_cf_L*_s0.npz' 'results/matrices/gate_llama1b_rome_cf_L8_s0.npz' 'results/matrices/gate_llama1b_rome_cf_L12_s0.npz' \
    --known --edit_ok --out results/MEMIT_vs_ROME_sc.json >> "$LOG" 2>&1 \
    && log "post: MEMIT_vs_ROME_sc done" || log "post: MEMIT_vs_ROME_sc FAIL"
fi
# consolidated report
{
  echo "RUN8H REPORT $(date '+%F %T')  jobs: ${n_done} done / ${n_fail} fail / ${n_skip} skipped  elapsed $(elapsed_min)m/${BUDGET_MIN}m  abort=${QUEUE_ABORT}"
  grep -E 'RUN |done |FAIL |SKIP|ABORT|PROGRESS' "$LOG" | tail -80
} > engine/run8h_report.txt
log "================ RUN8H COMPLETE (${n_done} done / ${n_fail} fail / ${n_skip} skip) ================"
echo "RUN8H_DONE" >> "$LOG"
