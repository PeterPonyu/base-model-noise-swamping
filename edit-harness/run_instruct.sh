#!/bin/bash
# run_instruct.sh — Llama-3.2-1B-Instruct twin battery (2026-07-06). Template = run_u6.sh
# (verbatim skeleton), INSTRUCT-namespaced (own pid/log/markers). Mirrors the existing
# Llama-3.2-1B (base) primary rows so base-vs-instruct is a clean paired comparison at
# IDENTICAL layers/seeds/settings — same architecture (16 layers, hidden 2048), same
# expected param count (1,235,814,400 — read directly off the ALREADY-downloaded base
# model's safetensors header via integrity_check.py, not guessed; Instruct ships the same
# vocab/config so this should match within the tool's 1% band).
#
# MODEL STATUS AS OF AUTHORING (2026-07-06): data/models/Llama-3.2-1B-Instruct does NOT
# exist. data/DOWNLOADS-20260706.md item 5: download FAILED — meta-llama/Llama-3.2-1B-
# Instruct is gated and no HF_TOKEN is configured anywhere on this machine (no env var, no
# ~/.cache/huggingface/token file); this is a pure auth gap, not a license-acceptance gap.
# Per this workspace's standing policy (downloads are ask-first; auth failures are reported,
# never worked around), THIS DRIVER DOES NOT ATTEMPT TO FETCH THE MODEL. Every row below is
# gated on engine/instruct_integrity.ok (derived below, NOT as a pf() hard preflight gate —
# see Phase 0a2) so a missing model produces a clean, loud CONFIG-SKIP log line per row
# and the script still completes and writes its report normally. Once the user exports
# HF_TOKEN and re-accepts the license, re-running this driver with NO changes will pick up
# the model automatically the next time integrity_check.py finds it.
set -u
H=/home/zeyufu/Desktop/idea-feasibility-analysis/edit-harness
cd "$H" || exit 1
PY=/home/zeyufu/miniconda3/envs/dl/bin/python3
LOG=engine/run_instruct.log
BUDGET_MIN=${BUDGET_MIN:-260}
mkdir -p engine results/matrices results/smoke_instruct/matrices
echo $$ > engine/run_instruct.pid
[ -f engine/instruct_round_start ] || stat -c %Y engine/run_instruct.pid > engine/instruct_round_start
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
log "================ RUN_INSTRUCT START (pid $$, budget ${BUDGET_MIN}m) ================"

# ---------------------------------------------------------------- Phase 0a: CPU pre-flight
# NOTE: the Instruct model dir is deliberately NOT checked here (see header + Phase 0a2) —
# these pf() checks are for infra that must exist regardless of whether the gated model
# has landed yet.
pf_fail=0
pf(){ if eval "$2"; then log "preflight OK: $1"; else log "PREFLIGHT-FAIL: $1"; pf_fail=1; fi; }
pf "python env" "$PY -c 'import torch, numpy' 2>/dev/null"
pf "counterfact.json" "[ -f data/counterfact.json ]"
pf "zsre_eval.json" "[ -f data/zsre_eval.json ]"
pf "egl flag" "grep -q -- '--egl' experiments/killgate_keygeom.py"
pf "alpha_proj_source flag" "grep -q -- '--alpha_proj_source' experiments/killgate_keygeom.py"
pf "integrity_check.py" "[ -f experiments/tools/integrity_check.py ]"
pf "analyze_matrices.py" "[ -f experiments/analyze_matrices.py ]"
pf "mechanism_sc_table.py" "[ -f experiments/mechanism_sc_table.py ]"
pf "base Llama-3.2-1B (comparator)" "[ -d data/models/Llama-3.2-1B ]"
pf "base ref L8 s0 (comparator)" "[ -f results/matrices/gate_llama1b_rome_cf_L8_s0.npz ]"
pf "base ref L14 s0 (comparator)" "[ -f results/matrices/gate_llama1b_rome_cf_L14_s0.npz ]"
pf "disk >=15GB free" "[ \$(df --output=avail -BG /home | tail -1 | tr -dc 0-9) -ge 15 ]"
rm -f engine/smoke_instruct_*.ok
if [ "$pf_fail" -ne 0 ]; then log "ABORT: preflight failed"; exit 3; fi

# ---------------------------------------------------------------- Phase 0a2: re-derive engine/instruct_integrity.ok
# Soft gate (NOT a pf() hard abort): the model may simply not be here yet (gated repo,
# auth pending — see header). Header-only check, no GPU, cheap to run every launch.
rm -f engine/instruct_integrity.ok
if [ -d data/models/Llama-3.2-1B-Instruct ]; then
  $PY experiments/tools/integrity_check.py data/models/Llama-3.2-1B-Instruct --expect_params 1.235814e9 >> "$LOG" 2>&1 \
    && { : > engine/instruct_integrity.ok; log "integrity OK: Llama-3.2-1B-Instruct"; } \
    || log "integrity NOT-READY: Llama-3.2-1B-Instruct (download incomplete or corrupt — rows CONFIG-skip)"
else
  log "MODEL-ABSENT: data/models/Llama-3.2-1B-Instruct not on disk (gated repo, HF_TOKEN required — data/DOWNLOADS-20260706.md item 5) — every row in this driver will CONFIG-skip cleanly until the user retries the download"
fi

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
KG="experiments/killgate_keygeom.py"
CF="--dataset counterfact --data data/counterfact.json"
ZS="--dataset zsre --data data/zsre_eval.json"
COMMON="--n_edits 200 --n_probes 500 --steps 20 --save_matrices --matrix_dir results/matrices"
SMK="--n_edits 4 --n_probes 40 --steps 2 --save_matrices --matrix_dir results/smoke_instruct/matrices"
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
  case "$cmd" in *smoke_instruct*) outn="results/smoke_instruct/matrices/$(basename "${outj%.json}").npz";; esac
  if [ -n "$outj" ] && [ -f "$outj" ] && [ -f "$outn" ]; then
    local pvmode="full"; [ "$class" = "SMOKE" ] && pvmode="smoke"
    if validate "$outj" "$outn" "$pvmode" | grep -q VALIDATE-FAIL; then
      mv "$outj" "$outj.INVALID" 2>/dev/null; mv "$outn" "$outn.INVALID" 2>/dev/null
      log "STALE-INVALID ${tag} — quarantined; re-running"
    else
      log "skip ${tag} (exists, validated)"
      [ "$class" = "SMOKE" ] && : > "engine/smoke_instruct_${tag}.ok"
      return
    fi
  fi
  local mdir; mdir=$(echo "$cmd" | grep -oE -- '--model [^ ]+' | awk '{print $2}')
  case "$mdir" in data/*|/*)
    if [ -n "$mdir" ] && [ ! -d "$mdir" ]; then log "CONFIG-SKIP ${tag} (no model ${mdir})"; n_skip=$((n_skip+1)); return; fi;;
  esac
  local cap=$(( est * 60 * 3 + 1200 ))
  log "RUN ${tag} [${class}] (est ${est}m, cap ${cap}s, elapsed ${now}m) -> engine/instruct_${tag}.log"
  local t rc dt
  t=$(date +%s)
  timeout --signal=TERM --kill-after=60 "${cap}s" bash -c "$cmd" >> "engine/instruct_${tag}.log" 2>&1
  rc=$?; dt=$(( $(date +%s) - t ))
  local vmode="full"; [ "$class" = "SMOKE" ] && vmode="smoke"
  if [ "$rc" -eq 0 ] && [ -n "$outj" ] && [ -f "$outj" ] && [ -f "$outn" ]; then
    local v; v=$(validate "$outj" "$outn" "$vmode")
    if echo "$v" | grep -q VALIDATE-FAIL; then
      mv "$outj" "$outj.INVALID" 2>/dev/null; mv "$outn" "$outn.INVALID" 2>/dev/null
      log "FAIL ${tag} (${dt}s) OUTPUT-INVALID: ${v}"; n_fail=$((n_fail+1))
    else
      log "done ${tag} (${dt}s) ${v}"; n_done=$((n_done+1)); wedge_fail=0
      [ "$class" = "SMOKE" ] && : > "engine/smoke_instruct_${tag}.ok"
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
NEEDS="engine/instruct_integrity.ok"
run_row SMOKE rome_instruct 6 "$NEEDS" "$ENVP $PY $KG --model data/models/Llama-3.2-1B-Instruct --editor rome $CF $SMK --lr 0.1 --layer 12 --seed 0 --out results/smoke_instruct/rome_instruct.json"
heartbeat

# ---------------------------------------------------------------- Block R: rome layer band x 3 seeds on CF
# mirrors the base model's own gate_llama1b_rome_cf_L{8,10,12,14}_s{0,1,2} cells exactly
# (same layers/seeds/settings) for a clean base-vs-instruct pairing.
for L in 8 10 12 14; do
  for s in 0 1 2; do
    est=25; [ "$L" = "14" ] && est=32
    [ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE gate_instruct_rome_cf_L${L}_s${s} "$est" "$NEEDS,engine/smoke_instruct_rome_instruct.ok" "$ENVP $PY $KG --model data/models/Llama-3.2-1B-Instruct --editor rome $CF $COMMON --lr 0.1 --layer ${L} --seed ${s} --out results/gate_instruct_rome_cf_L${L}_s${s}.json"
  done
  heartbeat
done

# ---------------------------------------------------------------- Block Z: zsRE spot check (L12, s0)
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE gate_instruct_rome_zsre_L12_s0 25 "$NEEDS,engine/smoke_instruct_rome_instruct.ok" "$ENVP $PY $KG --model data/models/Llama-3.2-1B-Instruct --editor rome $ZS $COMMON --lr 0.1 --layer 12 --seed 0 --out results/gate_instruct_rome_zsre_L12_s0.json"
heartbeat

# ---------------------------------------------------------------- Block E: EGL (L12) x 3 seeds
for s in 0 1 2; do
  [ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE egl_instruct_rome_cf_L12_s${s} 27 "$NEEDS,engine/smoke_instruct_rome_instruct.ok" "$ENVP $PY $KG --model data/models/Llama-3.2-1B-Instruct --editor rome --egl $CF $COMMON --lr 0.1 --layer 12 --seed ${s} --out results/egl_instruct_rome_cf_L12_s${s}.json"
done
heartbeat

# ---------------------------------------------------------------- Block A: matched alpha-holdout causal pair (L12, s0)
run_row SMOKE alphaHO_instruct 6 "$NEEDS" "$ENVP $PY $KG --model data/models/Llama-3.2-1B-Instruct --editor alpha $CF $SMK --lr 0.1 --layer 12 --seed 0 --alpha_proj_source holdout --holdout_frac 1.0 --out results/smoke_instruct/alphaHO_instruct.json"
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE g4_instruct_alphaHO_cf_L12_s0 25 "$NEEDS,engine/smoke_instruct_alphaHO_instruct.ok" "$ENVP $PY $KG --model data/models/Llama-3.2-1B-Instruct --editor alpha $CF $COMMON --lr 0.1 --layer 12 --seed 0 --alpha_proj_source holdout --holdout_frac 1.0 --out results/g4_instruct_alphaHO_cf_L12_s0.json"
heartbeat

# ---------------------------------------------------------------- Post-run (CPU, ALWAYS)
log "---------------- POST-RUN (CPU) ----------------"
$PY - > results/instruct_validation.json 2>>"$LOG" <<'EOF'
import json, glob, os
t0 = float(open('engine/instruct_round_start').read().strip())
out = []
for j in sorted(glob.glob('results/*instruct*.json')):
    base = os.path.basename(j)[:-5]
    z = 'results/matrices/' + base + '.npz'
    row = {'json': j, 'npz_found': os.path.exists(z), 'touched_this_run': os.path.getmtime(j) >= t0}
    try:
        d = json.load(open(j)); row['json_ok'] = True; row['esr'] = d.get('edit_success_rate')
    except Exception as e:
        row['json_ok'] = False; row['err'] = str(e)
    out.append(row)
print(json.dumps({'n': len(out), 'rows': out}, indent=1))
EOF
log "post: validation sweep -> results/instruct_validation.json"

for spec in "C3_instruct_rome_L8:results/matrices/gate_instruct_rome_cf_L8_s*.npz" \
            "C3_instruct_rome_L10:results/matrices/gate_instruct_rome_cf_L10_s*.npz" \
            "C3_instruct_rome_L12:results/matrices/gate_instruct_rome_cf_L12_s*.npz" \
            "C3_instruct_rome_L14:results/matrices/gate_instruct_rome_cf_L14_s*.npz" \
            "C3_instruct_egl_L12:results/matrices/egl_instruct_rome_cf_L12_s*.npz"; do
  outn="${spec%%:*}"; glob="${spec#*:}"
  if compgen -G "$glob" >/dev/null; then
    $PY experiments/analyze_matrices.py $glob --metric logit --known --edit_ok \
      --out "results/${outn}_instruct.json" >> "$LOG" 2>&1 && log "post: ${outn}_instruct done" || log "post: ${outn}_instruct FAIL"
  fi
done

# base-vs-instruct S x C law comparison table across the shared layer band
if compgen -G "results/matrices/gate_instruct_rome_cf_L*_s*.npz" >/dev/null; then
  $PY experiments/mechanism_sc_table.py \
    --npz 'results/matrices/gate_instruct_rome_cf_L*_s*.npz' \
    --known --edit_ok \
    --out results/INSTRUCT_mechanism_sc_table.json >> "$LOG" 2>&1 \
    && log "post: INSTRUCT_mechanism_sc_table done" || log "post: INSTRUCT_mechanism_sc_table FAIL"
fi

if [ -f experiments/aggregate_g4_causal.py ] \
   && compgen -G "results/matrices/g4_instruct_alphaHO_cf_L12_s0.npz" >/dev/null; then
  tmp_out="results/.C4_causal_instruct_table.json.tmp"
  $PY experiments/aggregate_g4_causal.py \
    --rome_glob 'results/matrices/gate_instruct_rome_cf_L{L}_s0.npz' \
    --alpha_glob 'results/matrices/g4_instruct_alphaHO_cf_L{L}_s0.npz' \
    --layers 12 --known --edit_ok --proj_source holdout \
    --out "$tmp_out" >> "$LOG" 2>&1 \
    && mv "$tmp_out" results/C4_causal_instruct_table.json \
    && log "post: C4_causal_instruct_table done (atomic)" \
    || { rm -f "$tmp_out"; log "FAIL aggregate_g4_causal instruct"; }
else
  log "skip C4-instruct (aggregate_g4_causal.py or holdout alpha npz missing)"
fi

{
  echo "RUN_INSTRUCT REPORT $(date '+%F %T')  jobs: ${n_done} done / ${n_fail} fail / ${n_skip} skipped  elapsed $(elapsed_min)m/${BUDGET_MIN}m  abort=${QUEUE_ABORT}"
  grep -E 'RUN |done |FAIL |SKIP|ABORT|PROGRESS|THERMAL|integrity|MODEL-ABSENT' "$LOG" | tail -80
} > engine/run_instruct_report.txt
log "================ RUN_INSTRUCT COMPLETE (${n_done} done / ${n_fail} fail / ${n_skip} skip) ================"
echo "RUN_INSTRUCT_DONE" >> "$LOG"
