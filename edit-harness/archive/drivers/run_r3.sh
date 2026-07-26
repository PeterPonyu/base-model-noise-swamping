#!/bin/bash
# run_r3.sh — ROUND 3: new-model science (2026-07-03). Template = run_8h.sh (hostile-reviewed)
# with the two 07-03 bug fixes: (1) validation sweep scoped to cell artifacts only (analysis
# JSONs have no npz BY DESIGN and were false-flagged); (2) model rows gate on INTEGRITY
# markers (headers+size+params), not dir-exists — a resumable-download partial dir passes
# dir-exists but must never reach science.
# Manifest: bf16 equivalence gate -> GPT-2-XL ROME sanity (+EGL) -> 8B scale (L24/L16)
#           -> relative-depth test (3B L24) -> MEMIT seeds -> stretch (8B L28, MEMIT s2).
set -u
H=/home/zeyufu/Desktop/idea-feasibility-analysis/edit-harness
cd "$H" || exit 1
PY=/home/zeyufu/miniconda3/envs/dl/bin/python3
LOG=engine/run_r3.log
BUDGET_MIN=${BUDGET_MIN:-480}
mkdir -p engine results/matrices results/smoke_r3/matrices
echo $$ > engine/run_r3.pid
[ -f engine/r3_round_start ] || stat -c %Y engine/run_r3.pid > engine/r3_round_start   # survives resumes (review MED #4)
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
log "================ RUN_R3 START (pid $$, budget ${BUDGET_MIN}m) ================"

# ---------------------------------------------------------------- Phase 0a: CPU pre-flight
pf_fail=0
pf(){ if eval "$2"; then log "preflight OK: $1"; else log "PREFLIGHT-FAIL: $1"; pf_fail=1; fi; }
pf "python env" "$PY -c 'import torch, numpy' 2>/dev/null"
pf "counterfact.json" "[ -f data/counterfact.json ]"
pf "arch_compat module" "[ -f editors/arch_compat.py ]"
pf "model_dtype flag" "grep -q -- '--model_dtype' experiments/killgate_keygeom.py"
pf "egl flag" "grep -q -- '--egl' experiments/killgate_keygeom.py"
pf "equiv fp32 canonical npz" "[ -f results/matrices/gate_llama1b_rome_cf_L12_s0.npz ]"
pf "disk >=30GB free" "[ \$(df --output=avail -BG /home | tail -1 | tr -dc 0-9) -ge 30 ]"
# NEW-MODEL INTEGRITY (bug-fix 2: marker-gated, tolerates still-downloading models by
# CONFIG-skipping their rows instead of failing the whole preflight)
# Clear ALL round gate markers: equiv/smoke markers surviving a resume would fail-OPEN
# the bf16 8B safety gate after a code change (hostile-review HIGH #1, 2026-07-03).
# The comparator re-derives equiv from the persisted npz at zero GPU cost.
rm -f engine/r3_integrity_8b.ok engine/r3_integrity_gpt2xl.ok engine/r3_equiv_bf16.ok engine/smoke_r3_*.ok
$PY experiments/tools/integrity_check.py data/models/Llama-3.1-8B --expect_params 8.03e9 >> "$LOG" 2>&1 \
  && { : > engine/r3_integrity_8b.ok; log "integrity OK: Llama-3.1-8B"; } \
  || log "integrity NOT-READY: Llama-3.1-8B (its rows will CONFIG-skip)"
$PY experiments/tools/integrity_check.py data/models/gpt2-xl --expect_params 1.608e9 >> "$LOG" 2>&1 \
  && { : > engine/r3_integrity_gpt2xl.ok; log "integrity OK: gpt2-xl"; } \
  || log "integrity NOT-READY: gpt2-xl (its rows will CONFIG-skip)"
if [ "$pf_fail" -ne 0 ]; then log "ABORT: preflight failed — nothing was run"; exit 3; fi

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
    if [ $(( $(date +%s) - gate_t0 )) -gt 1800 ]; then
      log "ABORT: GPU busy >30min at gate — window yielded"; exit 2
    fi
  fi
  log "gpu poll util=${util:-NA} mem=${mem:-NA} consec=${consec}/3"
  [ "$consec" -lt 3 ] && sleep 30
done
log "GPU idle — window opens now"
T0=$(date +%s)

# ---------------------------------------------------------------- helpers (run_8h template)
ENVP="env -u ALL_PROXY -u all_proxy HF_HUB_OFFLINE=1"
KG="experiments/killgate_keygeom.py"
CF="--dataset counterfact --data data/counterfact.json"
COMMON="--n_edits 200 --n_probes 500 --steps 20 --save_matrices --matrix_dir results/matrices"
SMK="--n_edits 4 --n_probes 40 --steps 2 --save_matrices --matrix_dir results/smoke_r3/matrices"
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
  case "$cmd" in *smoke_r3*) outn="results/smoke_r3/matrices/$(basename "${outj%.json}").npz";; esac
  if [ -n "$outj" ] && [ -f "$outj" ] && [ -f "$outn" ]; then
    local pvmode="full"; [ "$class" = "SMOKE" ] && pvmode="smoke"
    if validate "$outj" "$outn" "$pvmode" | grep -q VALIDATE-FAIL; then
      mv "$outj" "$outj.INVALID" 2>/dev/null; mv "$outn" "$outn.INVALID" 2>/dev/null
      log "STALE-INVALID ${tag} — quarantined; re-running"
    else
      log "skip ${tag} (exists, validated)"
      [ "$class" = "SMOKE" ] && : > "engine/smoke_r3_${tag}.ok"
      return
    fi
  fi
  # dir-precheck only for LOCAL model paths; HF-cache ids (e.g. sshleifer/tiny-gpt2) are
  # resolved by transformers from the cache and falsely fail -d (bug found 2026-07-03).
  local mdir; mdir=$(echo "$cmd" | grep -oE -- '--model [^ ]+' | awk '{print $2}')
  case "$mdir" in data/*|/*)
    if [ -n "$mdir" ] && [ ! -d "$mdir" ]; then log "CONFIG-SKIP ${tag} (no model ${mdir})"; n_skip=$((n_skip+1)); return; fi;;
  esac
  local cap=$(( est * 60 * 3 + 1200 ))
  log "RUN ${tag} [${class}] (est ${est}m, cap ${cap}s, elapsed ${now}m) -> engine/r3_${tag}.log"
  local t rc dt
  t=$(date +%s)
  timeout --signal=TERM --kill-after=60 "${cap}s" bash -c "$cmd" >> "engine/r3_${tag}.log" 2>&1
  rc=$?; dt=$(( $(date +%s) - t ))
  local vmode="full"; [ "$class" = "SMOKE" ] && vmode="smoke"
  if [ "$rc" -eq 0 ] && [ -n "$outj" ] && [ -f "$outj" ] && [ -f "$outn" ]; then
    local v; v=$(validate "$outj" "$outn" "$vmode")
    if echo "$v" | grep -q VALIDATE-FAIL; then
      mv "$outj" "$outj.INVALID" 2>/dev/null; mv "$outn" "$outn.INVALID" 2>/dev/null
      log "FAIL ${tag} (${dt}s) OUTPUT-INVALID: ${v}"; n_fail=$((n_fail+1))
    else
      log "done ${tag} (${dt}s) ${v}"; n_done=$((n_done+1)); wedge_fail=0
      [ "$class" = "SMOKE" ] && : > "engine/smoke_r3_${tag}.ok"
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

# ---------------------------------------------------------------- Phase 0c: micro-smokes
# gpt2-xl path incl. EGL (tiny-gpt2 proxy is cached; the XL row itself is integrity-gated)
run_row SMOKE gpt2egl 5 - "$ENVP $PY $KG --model sshleifer/tiny-gpt2 --device cuda --editor rome --egl $CF $SMK --lr 0.1 --layer 1 --seed 0 --out results/smoke_r3/gpt2egl.json"
# 8B bf16 load+edit micro-smoke (only if integrity passed)
run_row SMOKE bf16_8b 15 engine/r3_integrity_8b.ok "$ENVP $PY $KG --model data/models/Llama-3.1-8B --model_dtype bf16 --editor rome $CF $SMK --lr 0.1 --layer 16 --seed 0 --out results/smoke_r3/bf16_8b.json"
heartbeat

# ---------------------------------------------------------------- Phase A: bf16 EQUIVALENCE GATE
# Full-size Llama-1B L12 s0 in bf16; within-probe rho must match the fp32 canonical
# (gate_llama1b_rome_cf_L12_s0.npz) within |drho| < 0.02, else NO 8B science runs.
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE equiv_llama1b_bf16_L12_s0 22 - "$ENVP $PY $KG --model data/models/Llama-3.2-1B --model_dtype bf16 --editor rome $CF $COMMON --lr 0.1 --layer 12 --seed 0 --out results/equiv_llama1b_bf16_L12_s0.json"
if [ -f results/matrices/equiv_llama1b_bf16_L12_s0.npz ] && [ -f results/matrices/gate_llama1b_rome_cf_L12_s0.npz ]; then
  $PY - >> "$LOG" 2>&1 <<'EOF'
import numpy as np, sys, os
sys.path.insert(0, 'experiments')
from analyze_matrices import within_probe_rhos
def rho(f):
    d = np.load(f); C = d['COS'].astype(float); D = d['damage_logit'].astype(float)
    m = d['edit_ok'].astype(float) > 0; c = d['pre_p'].astype(float) > 0.05
    return float(np.nanmean(within_probe_rhos(C[m][:, c], D[m][:, c])))
r_fp32 = rho('results/matrices/gate_llama1b_rome_cf_L12_s0.npz')
r_bf16 = rho('results/matrices/equiv_llama1b_bf16_L12_s0.npz')
d = abs(r_fp32 - r_bf16)
print(f"[equiv-gate] fp32 rho={r_fp32:+.4f}  bf16 rho={r_bf16:+.4f}  |drho|={d:.4f}  bar=0.02")
if d < 0.02:
    open('engine/r3_equiv_bf16.ok', 'w').close()
    print("[equiv-gate] PASS — 8B science admitted")
else:
    print("[equiv-gate] FAIL — bf16 rows stay CONFIG-skipped; investigate before any 8B claim")
EOF
fi
heartbeat

# ---------------------------------------------------------------- Phase B: the science
# B1: GPT-2-XL ROME sanity (canonical ROME layer 17, with EGL for the published-numbers table)
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE sanity_gpt2xl_rome_cf_L17_s0 75 engine/r3_integrity_gpt2xl.ok,engine/smoke_r3_gpt2egl.ok "$ENVP $PY $KG --model data/models/gpt2-xl --editor rome --egl $CF $COMMON --lr 0.1 --layer 17 --seed 0 --out results/sanity_gpt2xl_rome_cf_L17_s0.json"
# B2: 8B scale — L24 (0.75 rel depth = the 1B-L12 peak-equivalent), then L16 (0.5 mid)
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE gate_llama8b_rome_cf_L24_s0 100 engine/r3_integrity_8b.ok,engine/smoke_r3_bf16_8b.ok,engine/r3_equiv_bf16.ok "$ENVP $PY $KG --model data/models/Llama-3.1-8B --model_dtype bf16 --editor rome $CF $COMMON --lr 0.1 --layer 24 --seed 0 --out results/gate_llama8b_rome_cf_L24_s0.json"
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE gate_llama8b_rome_cf_L16_s0 100 engine/r3_integrity_8b.ok,engine/smoke_r3_bf16_8b.ok,engine/r3_equiv_bf16.ok "$ENVP $PY $KG --model data/models/Llama-3.1-8B --model_dtype bf16 --editor rome $CF $COMMON --lr 0.1 --layer 16 --seed 0 --out results/gate_llama8b_rome_cf_L16_s0.json"
# B3: relative-depth test on the EXISTING 3B (28 layers; L24 = 0.857 rel = 1B-L14 regime)
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE gate_llama3b_rome_cf_L24_s0 55 - "$ENVP $PY $KG --model data/models/Llama-3.2-3B --editor rome $CF $COMMON --lr 0.1 --layer 24 --seed 0 --out results/gate_llama3b_rome_cf_L24_s0.json"
# B4: MEMIT seeds (canonical multi-seed for the editor spectrum)
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE gate_llama1b_memit_cf_L12_s1 32 - "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor memit $CF $COMMON --lr 0.1 --layer 12 --seed 1 --out results/gate_llama1b_memit_cf_L12_s1.json"
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE gate_llama1b_memit_cf_L8_s1 32 - "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor memit $CF $COMMON --lr 0.1 --layer 8 --seed 1 --out results/gate_llama1b_memit_cf_L8_s1.json"
heartbeat
# STRETCH (budget-gated): 8B relative-depth (L28 = 0.875) + MEMIT s2 + gpt2-xl second layer
[ "$QUEUE_ABORT" -eq 0 ] && run_row FILLER gate_llama8b_rome_cf_L28_s0 100 engine/r3_integrity_8b.ok,engine/r3_equiv_bf16.ok "$ENVP $PY $KG --model data/models/Llama-3.1-8B --model_dtype bf16 --editor rome $CF $COMMON --lr 0.1 --layer 28 --seed 0 --out results/gate_llama8b_rome_cf_L28_s0.json"
[ "$QUEUE_ABORT" -eq 0 ] && run_row FILLER gate_llama1b_memit_cf_L12_s2 32 - "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor memit $CF $COMMON --lr 0.1 --layer 12 --seed 2 --out results/gate_llama1b_memit_cf_L12_s2.json"
[ "$QUEUE_ABORT" -eq 0 ] && run_row FILLER gate_llama1b_memit_cf_L8_s2 32 - "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor memit $CF $COMMON --lr 0.1 --layer 8 --seed 2 --out results/gate_llama1b_memit_cf_L8_s2.json"
[ "$QUEUE_ABORT" -eq 0 ] && run_row FILLER sanity_gpt2xl_rome_cf_L5_s0 75 engine/r3_integrity_gpt2xl.ok,engine/smoke_r3_gpt2egl.ok "$ENVP $PY $KG --model data/models/gpt2-xl --editor rome --egl $CF $COMMON --lr 0.1 --layer 5 --seed 0 --out results/sanity_gpt2xl_rome_cf_L5_s0.json"
heartbeat

# ---------------------------------------------------------------- Post-run (CPU, ALWAYS)
log "---------------- POST-RUN (CPU) ----------------"
# validation sweep — BUG-FIX 1: cell artifacts only (gate_/g4_/qv_/sanity_/equiv_ with npz);
# analysis JSONs are out of scope by design.
$PY - > results/r3_validation.json 2>>"$LOG" <<'EOF'
import json, glob, os, numpy as np
t0 = float(open('engine/r3_round_start').read().strip())
out = []
for j in sorted(glob.glob('results/*.json')):
    base = os.path.basename(j)[:-5]
    if os.path.getmtime(j) < t0: continue
    if not base.startswith(('gate_', 'g4_', 'qv_', 'sanity_', 'equiv_')): continue
    z = 'results/matrices/' + base + '.npz'
    row = {'json': j, 'npz_found': os.path.exists(z)}
    try:
        d = json.load(open(j)); row['json_ok'] = True; row['esr'] = d.get('edit_success_rate')
    except Exception as e:
        row['json_ok'] = False; row['err'] = str(e)
    if row['npz_found']:
        a = np.load(z)
        row['npz_keys_ok'] = {'COS','damage_logit','norm_growth','edit_ok'} <= set(a.files)
        row['all_nan'] = bool(np.isnan(a['COS'].astype(float)).all())
        row['model_dtype'] = str(a['model_dtype']) if 'model_dtype' in a.files else 'float32'
    out.append(row)
print(json.dumps({'n': len(out), 'rows': out}, indent=1))
EOF
log "post: validation sweep (cells only) -> results/r3_validation.json"
# per-new-group C3 analysis (multi-seed where available)
for spec in "C3_memit_L12:results/matrices/gate_llama1b_memit_cf_L12_s*.npz" \
            "C3_memit_L8:results/matrices/gate_llama1b_memit_cf_L8_s*.npz" \
            "C3_llama8b:results/matrices/gate_llama8b_rome_cf_L*_s0.npz" \
            "C3_reldepth_3b_L24:results/matrices/gate_llama3b_rome_cf_L24_s0.npz" \
            "C3_gpt2xl:results/matrices/sanity_gpt2xl_rome_cf_L*_s0.npz"; do
  outn="${spec%%:*}"; glob="${spec#*:}"
  if compgen -G "$glob" >/dev/null; then
    $PY experiments/analyze_matrices.py $glob --metric logit --known --edit_ok \
      --out "results/${outn}_r3.json" >> "$LOG" 2>&1 && log "post: ${outn}_r3 done" || log "post: ${outn}_r3 FAIL"
  fi
done
{
  echo "RUN_R3 REPORT $(date '+%F %T')  jobs: ${n_done} done / ${n_fail} fail / ${n_skip} skipped  elapsed $(elapsed_min)m/${BUDGET_MIN}m  abort=${QUEUE_ABORT}"
  grep -E 'RUN |done |FAIL |SKIP|ABORT|equiv-gate|integrity|PROGRESS' "$LOG" | tail -60
} > engine/run_r3_report.txt
log "================ RUN_R3 COMPLETE (${n_done} done / ${n_fail} fail / ${n_skip} skip) ================"
echo "RUN_R3_DONE" >> "$LOG"
