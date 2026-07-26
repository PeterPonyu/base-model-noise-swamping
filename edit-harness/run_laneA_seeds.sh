#!/bin/bash
# run_laneA_seeds.sh — Lane A of docs/plans/LOCAL-COMPUTE-PLAN-2026-07-10.md: B6
# revision-readiness seed/layer gap-fill on the freed local 5090. Template =
# run_8bcausal.sh skeleton (npz validate + bf16 equiv gate), LANEA-namespaced (own
# pid/log/markers — never reuses 8bcausal/instruct/mquake script-scoped names, but
# deliberately REUSES the shared model-fact markers engine/instruct_integrity.ok,
# engine/r3_integrity_8b.ok, engine/r3_equiv_bf16.ok — those certify facts about the
# models/code, not about any one driver).
#
# WHAT THIS FILLS (gap analysis 2026-07-10, from results/ on disk):
#   - MQuAKE causal anchor was s0-only  -> add gate_llama1b_alpha_mquake_L12_s{1,2}
#     (probes-source projector, IDENTICAL protocol to the s0 anchor — reference-only
#     circularity caveat stands, memory/c4-alphaedit-projector-circularity.md; the s0
#     cell logged esr=0.885 VALIDATE-WARN, so a WARN here is precedented, not new)
#   - Instruct causal was s0-only       -> add g4_instruct_alphaHO_cf_L12_s{1,2}
#     (holdout projector; rome comparators s0-2 already exist from run_instruct.sh)
#   - 8B causal was s0-only             -> add g4_llama8b_alphaHO_cf_L{16,24}_s{1,2}
#     + gate_llama8b_rome_cf_L16_s2 (completes the rome triple; L24 rome s0-2 already
#     exist via run_r3/r4; L16 rome s0/s1 exist)
#   - A2 extra 8B causal layer          -> g4_llama8b_alphaHO_cf_L28_s0 (pairs the
#     ALREADY-EXISTING gate_llama8b_rome_cf_L28_s0 from run_r3 — a third 8B layer for
#     the weak/sign-flipping 8B causal story, memory/causal-8b-attenuation-20260707.md)
#   - A0 (gptj verify) is NOT here — it is a CPU check in engine/chain_laneA_20260710.sh
#   - A3 (ripple layers/seeds) needs NO GPU: rows landed via the 07-10 cloud wave
#     (ripple_llama1b_rome_popular_L{8,10,14}_s{0,1,2} + alpha L12 s{0,1,2} all on
#     disk); this driver only adds the CPU pooling summaries in its post-run.
#
# MEASURED per-cell timings (NOT guessed — from engine/run_mquake_law.log,
# engine/run_instruct.log, engine/run_8bcausal.log, engine/archive/run_r4_report.txt):
#   1B mquake/instruct cells ~1200-1300s (~21m, est 25m) ; 8B rome ~3100s (est 65m) ;
#   8B alphaHO ~2650s (est 55m; L28 unmeasured for alpha -> est 65m).
# Full queue ~8.4h inc. smokes/equiv — under the 12h reboot rule, but reboot first
# anyway if the box has been under load (memory/gpu-60w-thermal-cap-reboot-fix.md).
set -u
H=/home/zeyufu/Desktop/idea-feasibility-analysis/edit-harness
cd "$H" || exit 1
PY=/home/zeyufu/miniconda3/envs/dl/bin/python3
LOG=engine/run_lanea_seeds.log
BUDGET_MIN=${BUDGET_MIN:-560}
mkdir -p engine results/matrices results/smoke_lanea/matrices
echo $$ > engine/run_lanea_seeds.pid
[ -f engine/lanea_seeds_round_start ] || stat -c %Y engine/run_lanea_seeds.pid > engine/lanea_seeds_round_start
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
log "================ RUN_LANEA_SEEDS START (pid $$, budget ${BUDGET_MIN}m) ================"

# ---------------------------------------------------------------- Phase 0a: CPU pre-flight
pf_fail=0
pf(){ if eval "$2"; then log "preflight OK: $1"; else log "PREFLIGHT-FAIL: $1"; pf_fail=1; fi; }
pf "python env" "$PY -c 'import torch, numpy' 2>/dev/null"
pf "counterfact.json" "[ -f data/counterfact.json ]"
pf "mquake_cf3k.json (ask-first: never auto-download)" "[ -f data/mquake_cf3k.json ]"
pf "dataset mquake wired into killgate" "grep -q -- 'def load_mquake' experiments/killgate_keygeom.py"
pf "alpha_proj_source flag" "grep -q -- '--alpha_proj_source' experiments/killgate_keygeom.py"
pf "model_dtype flag" "grep -q -- '--model_dtype' experiments/killgate_keygeom.py"
pf "model Llama-3.2-1B" "[ -d data/models/Llama-3.2-1B ]"
pf "model Llama-3.2-1B-Instruct" "[ -d data/models/Llama-3.2-1B-Instruct ]"
pf "model Llama-3.1-8B" "[ -d data/models/Llama-3.1-8B ]"
pf "integrity_check.py" "[ -f experiments/tools/integrity_check.py ]"
pf "aggregate_g4_causal.py" "[ -f experiments/aggregate_g4_causal.py ]"
pf "analyze_matrices.py" "[ -f experiments/analyze_matrices.py ]"
pf "mquake alpha s0 anchor (protocol reference)" "[ -f results/matrices/gate_llama1b_alpha_mquake_L12_s0.npz ]"
pf "instruct alphaHO s0 (protocol reference)" "[ -f results/matrices/g4_instruct_alphaHO_cf_L12_s0.npz ]"
pf "8B rome L16 s0 npz (pair base)" "[ -f results/matrices/gate_llama8b_rome_cf_L16_s0.npz ]"
pf "8B rome L24 s0 npz (pair base)" "[ -f results/matrices/gate_llama8b_rome_cf_L24_s0.npz ]"
pf "8B rome L28 s0 npz (A2 pair base)" "[ -f results/matrices/gate_llama8b_rome_cf_L28_s0.npz ]"
pf "equiv comparator fp32 npz" "[ -f results/matrices/gate_llama1b_rome_cf_L12_s0.npz ]"
pf "disk >=20GB free" "[ \$(df --output=avail -BG /home | tail -1 | tr -dc 0-9) -ge 20 ]"
rm -f engine/smoke_lanea_*.ok
if [ "$pf_fail" -ne 0 ]; then log "ABORT: preflight failed"; exit 3; fi

# ---------------------------------------------------------------- Phase 0a2: re-derive shared integrity markers
# Header-only checks, no GPU, safe to run unconditionally every launch (r3/instruct pattern).
rm -f engine/instruct_integrity.ok
$PY experiments/tools/integrity_check.py data/models/Llama-3.2-1B-Instruct --expect_params 1.235814e9 >> "$LOG" 2>&1 \
  && { : > engine/instruct_integrity.ok; log "integrity OK: Llama-3.2-1B-Instruct"; } \
  || log "integrity NOT-READY: Llama-3.2-1B-Instruct (its rows will CONFIG-skip)"
rm -f engine/r3_integrity_8b.ok
$PY experiments/tools/integrity_check.py data/models/Llama-3.1-8B --expect_params 8.03e9 >> "$LOG" 2>&1 \
  && { : > engine/r3_integrity_8b.ok; log "integrity OK: Llama-3.1-8B"; } \
  || log "integrity NOT-READY: Llama-3.1-8B (its rows will CONFIG-skip)"
# r3_equiv_bf16.ok is re-derived in Phase A below (needs the GPU-gated equiv row if stale).
rm -f engine/r3_equiv_bf16.ok

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

# ---------------------------------------------------------------- helpers (8bcausal template, verbatim)
ENVP="env -u ALL_PROXY -u all_proxy HF_HUB_OFFLINE=1"
KG="experiments/killgate_keygeom.py"
CF="--dataset counterfact --data data/counterfact.json"
MQ="--dataset mquake --data data/mquake_cf3k.json"
COMMON="--n_edits 200 --n_probes 500 --steps 20 --save_matrices --matrix_dir results/matrices"
SMK="--n_edits 4 --n_probes 40 --steps 2 --save_matrices --matrix_dir results/smoke_lanea/matrices"
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
  case "$cmd" in *smoke_lanea*) outn="results/smoke_lanea/matrices/$(basename "${outj%.json}").npz";; esac
  if [ -n "$outj" ] && [ -f "$outj" ] && [ -f "$outn" ]; then
    local pvmode="full"; [ "$class" = "SMOKE" ] && pvmode="smoke"
    if validate "$outj" "$outn" "$pvmode" | grep -q VALIDATE-FAIL; then
      mv "$outj" "$outj.INVALID" 2>/dev/null; mv "$outn" "$outn.INVALID" 2>/dev/null
      log "STALE-INVALID ${tag} — quarantined; re-running"
    else
      log "skip ${tag} (exists, validated)"
      [ "$class" = "SMOKE" ] && : > "engine/smoke_lanea_${tag}.ok"
      return
    fi
  fi
  local mdir; mdir=$(echo "$cmd" | grep -oE -- '--model [^ ]+' | awk '{print $2}')
  case "$mdir" in data/*|/*)
    if [ -n "$mdir" ] && [ ! -d "$mdir" ]; then log "CONFIG-SKIP ${tag} (no model ${mdir})"; n_skip=$((n_skip+1)); return; fi;;
  esac
  local cap=$(( est * 60 * 3 + 1200 ))
  log "RUN ${tag} [${class}] (est ${est}m, cap ${cap}s, elapsed ${now}m) -> engine/lanea_${tag}.log"
  local t rc dt
  t=$(date +%s)
  timeout --signal=TERM --kill-after=60 "${cap}s" bash -c "$cmd" >> "engine/lanea_${tag}.log" 2>&1
  rc=$?; dt=$(( $(date +%s) - t ))
  local vmode="full"; [ "$class" = "SMOKE" ] && vmode="smoke"
  if [ "$rc" -eq 0 ] && [ -n "$outj" ] && [ -f "$outj" ] && [ -f "$outn" ]; then
    local v; v=$(validate "$outj" "$outn" "$vmode")
    if echo "$v" | grep -q VALIDATE-FAIL; then
      mv "$outj" "$outj.INVALID" 2>/dev/null; mv "$outn" "$outn.INVALID" 2>/dev/null
      log "FAIL ${tag} (${dt}s) OUTPUT-INVALID: ${v}"; n_fail=$((n_fail+1))
    else
      log "done ${tag} (${dt}s) ${v}"; n_done=$((n_done+1)); wedge_fail=0
      [ "$class" = "SMOKE" ] && : > "engine/smoke_lanea_${tag}.ok"
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

# ---------------------------------------------------------------- Block M: MQuAKE causal-anchor seeds (fastest first)
# Combo (mquake+alpha, probes-source) has run before (s0, 07-07) — smoke is a cheap
# regression guard against code drift since then, not a first-run gate.
run_row SMOKE alpha_mq 4 - "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor alpha $MQ $SMK --lr 0.1 --layer 12 --seed 0 --out results/smoke_lanea/alpha_mq.json"
heartbeat
for s in 1 2; do
  [ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE gate_llama1b_alpha_mquake_L12_s${s} 25 engine/smoke_lanea_alpha_mq.ok "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor alpha $MQ $COMMON --lr 0.1 --layer 12 --seed ${s} --out results/gate_llama1b_alpha_mquake_L12_s${s}.json"
done
heartbeat

# ---------------------------------------------------------------- Block I: Instruct alpha-holdout seeds
run_row SMOKE alphaHO_instr 6 engine/instruct_integrity.ok "$ENVP $PY $KG --model data/models/Llama-3.2-1B-Instruct --editor alpha $CF $SMK --lr 0.1 --layer 12 --seed 0 --alpha_proj_source holdout --holdout_frac 1.0 --out results/smoke_lanea/alphaHO_instr.json"
heartbeat
for s in 1 2; do
  [ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE g4_instruct_alphaHO_cf_L12_s${s} 25 engine/instruct_integrity.ok,engine/smoke_lanea_alphaHO_instr.ok "$ENVP $PY $KG --model data/models/Llama-3.2-1B-Instruct --editor alpha $CF $COMMON --lr 0.1 --layer 12 --seed ${s} --alpha_proj_source holdout --holdout_frac 1.0 --out results/g4_instruct_alphaHO_cf_L12_s${s}.json"
done
heartbeat

# ---------------------------------------------------------------- Phase A: bf16 EQUIVALENCE GATE (re-derive engine/r3_equiv_bf16.ok)
# Verbatim run_8bcausal.sh logic: if the cached bf16 comparator predates the current
# killgate_keygeom.py, quarantine it so run_row's idempotency can't skip it, then let
# the row execute for real. Cheap CPU reuse when the cache is fresh (~0 GPU).
if [ "$DRYRUN" -ne 1 ] && [ -f results/matrices/equiv_llama1b_bf16_L12_s0.npz ] \
   && [ -f experiments/killgate_keygeom.py ] \
   && [ "$(stat -c %Y results/matrices/equiv_llama1b_bf16_L12_s0.npz)" -lt "$(stat -c %Y experiments/killgate_keygeom.py)" ]; then
  log "equiv comparator npz is STALE (older than killgate_keygeom.py) — quarantining to force re-run"
  mv results/equiv_llama1b_bf16_L12_s0.json results/equiv_llama1b_bf16_L12_s0.json.STALE 2>/dev/null
  mv results/matrices/equiv_llama1b_bf16_L12_s0.npz results/matrices/equiv_llama1b_bf16_L12_s0.npz.STALE 2>/dev/null
fi
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE equiv_llama1b_bf16_L12_s0 22 engine/r3_integrity_8b.ok "$ENVP $PY $KG --model data/models/Llama-3.2-1B --model_dtype bf16 --editor rome $CF $COMMON --lr 0.1 --layer 12 --seed 0 --out results/equiv_llama1b_bf16_L12_s0.json"
if [ "$DRYRUN" -ne 1 ] && [ -f results/matrices/equiv_llama1b_bf16_L12_s0.npz ] && [ -f results/matrices/gate_llama1b_rome_cf_L12_s0.npz ]; then
  $PY - >> "$LOG" 2>&1 <<'EOF'
import numpy as np, sys
sys.path.insert(0, 'experiments')
from analyze_matrices import within_probe_rhos
def rho(f):
    d = np.load(f); C = d['COS'].astype(float); D = d['damage_logit'].astype(float)
    m = d['edit_ok'].astype(float) > 0; c = d['pre_p'].astype(float) > 0.05
    return float(np.nanmean(within_probe_rhos(C[m][:, c], D[m][:, c])))
r_fp32 = rho('results/matrices/gate_llama1b_rome_cf_L12_s0.npz')
r_bf16 = rho('results/matrices/equiv_llama1b_bf16_L12_s0.npz')
d = abs(r_fp32 - r_bf16)
print(f"[lanea equiv-gate] fp32 rho={r_fp32:+.4f} bf16 rho={r_bf16:+.4f} |drho|={d:.4f} bar=0.02")
if d < 0.02:
    open('engine/r3_equiv_bf16.ok', 'w').close()
    print("[lanea equiv-gate] PASS — 8B science admitted")
else:
    print("[lanea equiv-gate] FAIL — 8B rows stay CONFIG-skipped; investigate before any 8B claim")
EOF
fi
heartbeat

# ---------------------------------------------------------------- Block E: 8B rome triple completion + alpha-holdout seeds + A2 layer
run_row SMOKE alphaHO_8b 15 engine/r3_integrity_8b.ok "$ENVP $PY $KG --model data/models/Llama-3.1-8B --model_dtype bf16 --editor alpha $CF $SMK --lr 0.1 --layer 16 --seed 0 --alpha_proj_source holdout --holdout_frac 1.0 --out results/smoke_lanea/alphaHO_8b.json"
heartbeat
G8="engine/r3_integrity_8b.ok,engine/r3_equiv_bf16.ok"
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE gate_llama8b_rome_cf_L16_s2 65 "$G8" "$ENVP $PY $KG --model data/models/Llama-3.1-8B --model_dtype bf16 --editor rome $CF $COMMON --lr 0.1 --layer 16 --seed 2 --out results/gate_llama8b_rome_cf_L16_s2.json"
heartbeat
for cell in "16 1" "16 2" "24 1" "24 2"; do
  set -- $cell; L=$1; s=$2
  [ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE g4_llama8b_alphaHO_cf_L${L}_s${s} 55 "$G8,engine/smoke_lanea_alphaHO_8b.ok" "$ENVP $PY $KG --model data/models/Llama-3.1-8B --model_dtype bf16 --editor alpha $CF $COMMON --lr 0.1 --layer ${L} --seed ${s} --alpha_proj_source holdout --holdout_frac 1.0 --out results/g4_llama8b_alphaHO_cf_L${L}_s${s}.json"
  heartbeat
done
# A2: third 8B causal layer — alpha pairs the pre-existing rome L28 s0 row (run_r3).
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE g4_llama8b_alphaHO_cf_L28_s0 65 "$G8,engine/smoke_lanea_alphaHO_8b.ok" "$ENVP $PY $KG --model data/models/Llama-3.1-8B --model_dtype bf16 --editor alpha $CF $COMMON --lr 0.1 --layer 28 --seed 0 --alpha_proj_source holdout --holdout_frac 1.0 --out results/g4_llama8b_alphaHO_cf_L28_s0.json"
heartbeat

# A2 completion (2026-07-10 resume): L28 was left single-seed by the first pass —
# REVISION_DOSSIER 8b_alphaHO_L28 = PENDING. aggregate_g4_causal intersects rome/alpha
# seed maps, so BOTH sides need s1/s2 (rome s1/s2 never existed for L28).
for s in 1 2; do
  [ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE gate_llama8b_rome_cf_L28_s${s} 65 "$G8" "$ENVP $PY $KG --model data/models/Llama-3.1-8B --model_dtype bf16 --editor rome $CF $COMMON --lr 0.1 --layer 28 --seed ${s} --out results/gate_llama8b_rome_cf_L28_s${s}.json"
  heartbeat
  [ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE g4_llama8b_alphaHO_cf_L28_s${s} 65 "$G8,engine/smoke_lanea_alphaHO_8b.ok" "$ENVP $PY $KG --model data/models/Llama-3.1-8B --model_dtype bf16 --editor alpha $CF $COMMON --lr 0.1 --layer 28 --seed ${s} --alpha_proj_source holdout --holdout_frac 1.0 --out results/g4_llama8b_alphaHO_cf_L28_s${s}.json"
  heartbeat
done

# ---------------------------------------------------------------- Post-run (CPU, ALWAYS)
log "---------------- POST-RUN (CPU) ----------------"
$PY - > results/lanea_seeds_validation.json 2>>"$LOG" <<'EOF'
import json, glob, os
t0 = float(open('engine/lanea_seeds_round_start').read().strip())
targets = ['results/gate_llama1b_alpha_mquake_L12_s*.json',
           'results/g4_instruct_alphaHO_cf_L12_s*.json',
           'results/gate_llama8b_rome_cf_L16_s2.json',
           'results/g4_llama8b_alphaHO_cf_L*.json']
out = []
for pat in targets:
    for j in sorted(glob.glob(pat)):
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
log "post: validation sweep -> results/lanea_seeds_validation.json"

# MQuAKE: 3-seed pooled alpha anchor + causal table. proj_source=probes is BY DESIGN the
# reference-only (circular-projector) protocol matching the s0 anchor — the output name
# carries the caveat so it can never be quoted as the honest holdout number
# (memory/c4-alphaedit-projector-circularity.md). Invocation CPU-tested 2026-07-10 on the
# existing s0 npz before this driver was committed.
if compgen -G "results/matrices/gate_llama1b_alpha_mquake_L12_s*.npz" >/dev/null; then
  tmp_c3="results/.C3_mquake_alpha_L12_3seed.json.tmp"
  $PY experiments/analyze_matrices.py results/matrices/gate_llama1b_alpha_mquake_L12_s*.npz \
    --metric logit --known --edit_ok \
    --out "$tmp_c3" >> "$LOG" 2>&1 \
    && mv "$tmp_c3" results/C3_mquake_alpha_L12_3seed.json \
    && log "post: C3_mquake_alpha_L12_3seed done (atomic)" \
    || { rm -f "$tmp_c3"; log "post: C3_mquake_alpha_L12_3seed FAIL"; }
  tmp_out="results/.C4_causal_mquake_table_3seed_probesrc.json.tmp"
  $PY experiments/aggregate_g4_causal.py \
    --rome_glob 'results/matrices/gate_llama1b_rome_mquake_L{L}_s*.npz' \
    --alpha_glob 'results/matrices/gate_llama1b_alpha_mquake_L{L}_s*.npz' \
    --layers 12 --known --edit_ok --proj_source probes \
    --out "$tmp_out" >> "$LOG" 2>&1 \
    && mv "$tmp_out" results/C4_causal_mquake_table_3seed_probesrc.json \
    && log "post: C4_causal_mquake_table_3seed_probesrc done (atomic)" \
    || { rm -f "$tmp_out"; log "FAIL aggregate_g4_causal mquake"; }
fi

# Instruct: 3-seed honest holdout causal table (rome comparators already 3-seed on disk).
if compgen -G "results/matrices/g4_instruct_alphaHO_cf_L12_s*.npz" >/dev/null; then
  tmp_out="results/.C4_causal_instruct_table_3seed.json.tmp"
  $PY experiments/aggregate_g4_causal.py \
    --rome_glob 'results/matrices/gate_instruct_rome_cf_L{L}_s*.npz' \
    --alpha_glob 'results/matrices/g4_instruct_alphaHO_cf_L{L}_s*.npz' \
    --layers 12 --known --edit_ok --proj_source holdout \
    --out "$tmp_out" >> "$LOG" 2>&1 \
    && mv "$tmp_out" results/C4_causal_instruct_table_3seed.json \
    && log "post: C4_causal_instruct_table_3seed done (atomic)" \
    || { rm -f "$tmp_out"; log "FAIL aggregate_g4_causal instruct 3seed"; }
fi

# 8B: multi-seed honest holdout causal table across L16/L24 (+L28 single-seed A2 layer;
# aggregate pools whatever seeds exist per layer and records seeds_used).
if compgen -G "results/matrices/g4_llama8b_alphaHO_cf_L*_s*.npz" >/dev/null; then
  tmp_out="results/.C4_causal_8b_table_3seed.json.tmp"
  $PY experiments/aggregate_g4_causal.py \
    --rome_glob 'results/matrices/gate_llama8b_rome_cf_L{L}_s*.npz' \
    --alpha_glob 'results/matrices/g4_llama8b_alphaHO_cf_L{L}_s*.npz' \
    --layers 16 24 28 --known --edit_ok --proj_source holdout \
    --out "$tmp_out" >> "$LOG" 2>&1 \
    && mv "$tmp_out" results/C4_causal_8b_table_3seed.json \
    && log "post: C4_causal_8b_table_3seed done (atomic)" \
    || { rm -f "$tmp_out"; log "FAIL aggregate_g4_causal 8b 3seed"; }
fi

# Ripple (A3, CPU-only): the layer/seed rows already landed via the 07-10 cloud wave —
# pool the depth profile (rome L8/10/12/14 x s0-2) + the alpha L12 3-seed causal contrast.
$PY - >> "$LOG" 2>&1 <<'EOF'
import json, glob, statistics as st
prof = {}
for L in (8, 10, 12, 14):
    rows = []
    for j in sorted(glob.glob(f'results/ripple_llama1b_rome_popular_L{L}_s*.json')):
        d = json.load(open(j))
        r = d.get('within_probe_rho_logit', {})
        if r.get('ripple') is None or r.get('unrelated') is None:
            print(f"[lanea post] SKIP {j}: missing ripple/unrelated rho"); continue
        rows.append((r.get('ripple'), r.get('unrelated'), d.get('edit_success_rate')))
    if rows:
        prof[f'L{L}'] = {
            'n_seeds': len(rows),
            'rho_ripple_mean': st.mean(x[0] for x in rows),
            'rho_unrelated_mean': st.mean(x[1] for x in rows),
            'rho_ripple_per_seed': [x[0] for x in rows],
            'rho_unrelated_per_seed': [x[1] for x in rows],
            'esr_per_seed': [x[2] for x in rows]}
alpha = []
for j in sorted(glob.glob('results/ripple_llama1b_alpha_popular_L12_s*.json')):
    d = json.load(open(j))
    r = d.get('within_probe_rho_logit', {})
    alpha.append({'json': j, 'rho_ripple': r.get('ripple'), 'rho_unrelated': r.get('unrelated'),
                  'esr': d.get('edit_success_rate')})
out = {'rome_depth_profile': prof, 'alpha_L12_per_seed': alpha,
       'note': 'pooled from per-cell jsons (run_ripple/run_ripple_ext protocol, popular split)'}
import os
tmp = 'results/.RIPPLE_depth_profile.json.tmp'
json.dump(out, open(tmp, 'w'), indent=1)
os.replace(tmp, 'results/RIPPLE_depth_profile.json')
print(f"[lanea post] RIPPLE_depth_profile: layers={list(prof)} alpha_seeds={len(alpha)}")
EOF
log "post: RIPPLE_depth_profile attempted"

{
  echo "RUN_LANEA_SEEDS REPORT $(date '+%F %T')  jobs: ${n_done} done / ${n_fail} fail / ${n_skip} skipped  elapsed $(elapsed_min)m/${BUDGET_MIN}m  abort=${QUEUE_ABORT}"
  grep -E 'RUN |done |FAIL |SKIP|ABORT|PROGRESS|THERMAL|equiv-gate|integrity' "$LOG" | tail -80
} > engine/run_lanea_seeds_report.txt
log "================ RUN_LANEA_SEEDS COMPLETE (${n_done} done / ${n_fail} fail / ${n_skip} skip) ================"
echo "RUN_LANEA_SEEDS_DONE" >> "$LOG"
