#!/bin/bash
# run_8bcausal.sh — STUB-8BCAUSAL (2026-07-05): matched ROME + AlphaEdit causal pairs at
# Llama-3.1-8B bf16, honest (holdout-projector) protocol. Template = run_r3.sh/run_r4.sh
# (verbatim skeleton for the 8B gates), 8BCAUSAL-namespaced (own pid/log — never reuses
# r3/r4/u5 script-scoped names). Deliberately REUSES the shared marker names
# engine/r3_integrity_8b.ok and engine/r3_equiv_bf16.ok (per the orchestrator's brief) rather
# than re-namespacing them, since those two gates certify a fact about the model/code, not
# about this driver, and other future 8B drivers are expected to look for the same names.
#
# STATE CHECKED 2026-07-05: neither marker exists on disk (fresh checkout / torn-down
# session) — Phase 0a below RE-DERIVES both from scratch (integrity_check.py is header-only,
# no GPU; the equiv-gate reuses the EXISTING equiv_llama1b_bf16_L12_s0.npz + gate_llama1b_
# rome_cf_L12_s0.npz pair, both already on disk, so this costs ~0 GPU time — no re-run of the
# bf16 equivalence row itself is needed). Also: results/matrices/gate_llama8b_rome_cf_L24_s0.npz
# and .../gate_llama8b_rome_cf_L16_s0.npz ALREADY EXIST (run_r3.sh, valid) — the 2 ROME rows
# below will therefore idempotency-skip; only the 2 AlphaEdit(holdout) rows are real new GPU
# work. Cells: L24 (positive-damage-regime counterpart, already characterized for ROME) and
# L16 (the other 8B layer already characterized), seed s0, --alpha_proj_source holdout (E6
# honest protocol — see memory/c4-alphaedit-projector-circularity.md).
#
# GUESSED / FLAGGED: AlphaEdit-at-8B has NO cost precedent anywhere in the harness (only ROME
# has run at 8B scale, 100m/row in run_r3.sh). 1B alpha vs rome cost roughly at parity (22m
# each, run_8h.sh), but AlphaEdit's null-space projector fit scales with hidden-dim^2 more than
# ROME's edit — the 120m/row estimate below is a conservative guess, NOT measured. The
# orchestrator's brief said "~2-3 GPU-h for 4 rows," but 2 of those 4 rows are free skips (ROME
# already done) — real GPU cost here is ~2 alpha rows x ~120m = ~4h, higher than that 2-3h
# figure if taken literally against all 4 rows. Flagging the discrepancy rather than silently
# matching it.
set -u
H=/home/zeyufu/Desktop/idea-feasibility-analysis/edit-harness
cd "$H" || exit 1
PY=/home/zeyufu/miniconda3/envs/dl/bin/python3
LOG=engine/run_8bcausal.log
BUDGET_MIN=${BUDGET_MIN:-300}
mkdir -p engine results/matrices results/smoke_8bcausal/matrices
echo $$ > engine/run_8bcausal.pid
[ -f engine/8bcausal_round_start ] || stat -c %Y engine/run_8bcausal.pid > engine/8bcausal_round_start
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
log "================ RUN_8BCAUSAL START (pid $$, budget ${BUDGET_MIN}m) ================"

# ---------------------------------------------------------------- Phase 0a: CPU pre-flight
pf_fail=0
pf(){ if eval "$2"; then log "preflight OK: $1"; else log "PREFLIGHT-FAIL: $1"; pf_fail=1; fi; }
pf "python env" "$PY -c 'import torch, numpy' 2>/dev/null"
pf "counterfact.json" "[ -f data/counterfact.json ]"
pf "model_dtype flag" "grep -q -- '--model_dtype' experiments/killgate_keygeom.py"
pf "alpha_proj_source flag" "grep -q -- '--alpha_proj_source' experiments/killgate_keygeom.py"
pf "model Llama-3.1-8B" "[ -d data/models/Llama-3.1-8B ]"
pf "integrity_check.py" "[ -f experiments/tools/integrity_check.py ]"
pf "aggregate_g4_causal.py" "[ -f experiments/aggregate_g4_causal.py ]"
pf "ROME ref L24 s0 (expect pre-existing, idempotent skip)" "[ -f results/matrices/gate_llama8b_rome_cf_L24_s0.npz ]"
pf "ROME ref L16 s0 (expect pre-existing, idempotent skip)" "[ -f results/matrices/gate_llama8b_rome_cf_L16_s0.npz ]"
pf "equiv comparator fp32 npz" "[ -f results/matrices/gate_llama1b_rome_cf_L12_s0.npz ]"
pf "equiv comparator bf16 npz" "[ -f results/matrices/equiv_llama1b_bf16_L12_s0.npz ]"
pf "disk >=20GB free" "[ \$(df --output=avail -BG /home | tail -1 | tr -dc 0-9) -ge 20 ]"
rm -f engine/smoke_8bcausal_*.ok
if [ "$pf_fail" -ne 0 ]; then log "ABORT: preflight failed"; exit 3; fi

# ---------------------------------------------------------------- Phase 0a2: re-derive engine/r3_integrity_8b.ok
# Doesn't exist on disk as of 2026-07-05 (torn-down session) — DON'T skip this gate; header-
# only check, no GPU, safe to run unconditionally every launch (mirrors r3.sh/r4.sh).
rm -f engine/r3_integrity_8b.ok
$PY experiments/tools/integrity_check.py data/models/Llama-3.1-8B --expect_params 8.03e9 >> "$LOG" 2>&1 \
  && { : > engine/r3_integrity_8b.ok; log "integrity OK: Llama-3.1-8B"; } \
  || log "integrity NOT-READY: Llama-3.1-8B (its rows will CONFIG-skip)"
# engine/r3_equiv_bf16.ok is NOT re-derived here: it requires the equiv_llama1b_bf16_L12_s0
# GPU row, which must run inside the GPU-idle-gated section below (Phase A), not in CPU
# preflight. See Phase A for the freshness check + forced re-run if the cached comparator
# predates the current killgate_keygeom.py (checked 2026-07-05: it DOES predate it — the
# gate will force a real ~22min re-run on first launch, not a free reuse. Flagged loudly:
# do not assume this marker is free).
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

# ---------------------------------------------------------------- helpers (u2/u4/u5/r3/r4 template, verbatim)
ENVP="env -u ALL_PROXY -u all_proxy HF_HUB_OFFLINE=1"
KG="experiments/killgate_keygeom.py"
CF="--dataset counterfact --data data/counterfact.json"
COMMON="--n_edits 200 --n_probes 500 --steps 20 --save_matrices --matrix_dir results/matrices"
SMK="--n_edits 4 --n_probes 40 --steps 2 --save_matrices --matrix_dir results/smoke_8bcausal/matrices"
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
  case "$cmd" in *smoke_8bcausal*) outn="results/smoke_8bcausal/matrices/$(basename "${outj%.json}").npz";; esac
  if [ -n "$outj" ] && [ -f "$outj" ] && [ -f "$outn" ]; then
    local pvmode="full"; [ "$class" = "SMOKE" ] && pvmode="smoke"
    if validate "$outj" "$outn" "$pvmode" | grep -q VALIDATE-FAIL; then
      mv "$outj" "$outj.INVALID" 2>/dev/null; mv "$outn" "$outn.INVALID" 2>/dev/null
      log "STALE-INVALID ${tag} — quarantined; re-running"
    else
      log "skip ${tag} (exists, validated)"
      [ "$class" = "SMOKE" ] && : > "engine/smoke_8bcausal_${tag}.ok"
      return
    fi
  fi
  local mdir; mdir=$(echo "$cmd" | grep -oE -- '--model [^ ]+' | awk '{print $2}')
  case "$mdir" in data/*|/*)
    if [ -n "$mdir" ] && [ ! -d "$mdir" ]; then log "CONFIG-SKIP ${tag} (no model ${mdir})"; n_skip=$((n_skip+1)); return; fi;;
  esac
  local cap=$(( est * 60 * 3 + 1200 ))
  log "RUN ${tag} [${class}] (est ${est}m, cap ${cap}s, elapsed ${now}m) -> engine/8bcausal_${tag}.log"
  local t rc dt
  t=$(date +%s)
  timeout --signal=TERM --kill-after=60 "${cap}s" bash -c "$cmd" >> "engine/8bcausal_${tag}.log" 2>&1
  rc=$?; dt=$(( $(date +%s) - t ))
  local vmode="full"; [ "$class" = "SMOKE" ] && vmode="smoke"
  if [ "$rc" -eq 0 ] && [ -n "$outj" ] && [ -f "$outj" ] && [ -f "$outn" ]; then
    local v; v=$(validate "$outj" "$outn" "$vmode")
    if echo "$v" | grep -q VALIDATE-FAIL; then
      mv "$outj" "$outj.INVALID" 2>/dev/null; mv "$outn" "$outn.INVALID" 2>/dev/null
      log "FAIL ${tag} (${dt}s) OUTPUT-INVALID: ${v}"; n_fail=$((n_fail+1))
    else
      log "done ${tag} (${dt}s) ${v}"; n_done=$((n_done+1)); wedge_fail=0
      [ "$class" = "SMOKE" ] && : > "engine/smoke_8bcausal_${tag}.ok"
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

# ---------------------------------------------------------------- Phase A: bf16 EQUIVALENCE GATE (re-derive engine/r3_equiv_bf16.ok)
# Same freshness principle as run_r4.sh's reuse-guard, but this is the ORIGINAL-derivation
# path (run_r3.sh Phase A), so instead of refusing to certify a stale comparator we FORCE a
# fresh re-run: if the cached npz predates killgate_keygeom.py, quarantine it so run_row's
# idempotency can't skip it, then let the row execute for real inside the GPU-gated queue.
# Checked 2026-07-05: equiv_llama1b_bf16_L12_s0.npz (mtime-old) DOES predate killgate_keygeom.py
# (mtime-newer) — this branch WILL fire on a real launch, costing one real ~22min GPU row that
# is not reflected in the "already exists, 0 cost" assumption elsewhere in this file's header.
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
print(f"[8bcausal equiv-gate] fp32 rho={r_fp32:+.4f} bf16 rho={r_bf16:+.4f} |drho|={d:.4f} bar=0.02")
if d < 0.02:
    open('engine/r3_equiv_bf16.ok', 'w').close()
    print("[8bcausal equiv-gate] PASS — 8B science admitted")
else:
    print("[8bcausal equiv-gate] FAIL — bf16 rows stay CONFIG-skipped; investigate before any 8B claim")
EOF
fi
heartbeat

# ---------------------------------------------------------------- Phase 0c: micro-smoke
# AlphaEdit + bf16 + 8B + holdout-projector combo has never run — gated on integrity marker.
run_row SMOKE alphaHO_8b 15 engine/r3_integrity_8b.ok "$ENVP $PY $KG --model data/models/Llama-3.1-8B --model_dtype bf16 --editor alpha $CF $SMK --lr 0.1 --layer 16 --seed 0 --alpha_proj_source holdout --holdout_frac 1.0 --out results/smoke_8bcausal/alphaHO_8b.json"
heartbeat

# ---------------------------------------------------------------- Block: matched ROME + AlphaEdit(holdout) pairs
# ROME rows are enumerated for pair completeness/reproducibility of the manifest; both are
# already-valid on disk (run_r3.sh) and will idempotency-skip at ~0 cost.
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE gate_llama8b_rome_cf_L24_s0 100 engine/r3_integrity_8b.ok,engine/r3_equiv_bf16.ok "$ENVP $PY $KG --model data/models/Llama-3.1-8B --model_dtype bf16 --editor rome $CF $COMMON --lr 0.1 --layer 24 --seed 0 --out results/gate_llama8b_rome_cf_L24_s0.json"
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE gate_llama8b_rome_cf_L16_s0 100 engine/r3_integrity_8b.ok,engine/r3_equiv_bf16.ok "$ENVP $PY $KG --model data/models/Llama-3.1-8B --model_dtype bf16 --editor rome $CF $COMMON --lr 0.1 --layer 16 --seed 0 --out results/gate_llama8b_rome_cf_L16_s0.json"
heartbeat
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE g4_llama8b_alphaHO_cf_L24_s0 120 engine/r3_integrity_8b.ok,engine/smoke_8bcausal_alphaHO_8b.ok,engine/r3_equiv_bf16.ok "$ENVP $PY $KG --model data/models/Llama-3.1-8B --model_dtype bf16 --editor alpha $CF $COMMON --lr 0.1 --layer 24 --seed 0 --alpha_proj_source holdout --holdout_frac 1.0 --out results/g4_llama8b_alphaHO_cf_L24_s0.json"
[ "$QUEUE_ABORT" -eq 0 ] && run_row SCIENCE g4_llama8b_alphaHO_cf_L16_s0 120 engine/r3_integrity_8b.ok,engine/smoke_8bcausal_alphaHO_8b.ok,engine/r3_equiv_bf16.ok "$ENVP $PY $KG --model data/models/Llama-3.1-8B --model_dtype bf16 --editor alpha $CF $COMMON --lr 0.1 --layer 16 --seed 0 --alpha_proj_source holdout --holdout_frac 1.0 --out results/g4_llama8b_alphaHO_cf_L16_s0.json"
heartbeat

# ---------------------------------------------------------------- Post-run (CPU, ALWAYS)
log "---------------- POST-RUN (CPU) ----------------"
$PY - > results/8bcausal_validation.json 2>>"$LOG" <<'EOF'
import json, glob, os
t0 = float(open('engine/8bcausal_round_start').read().strip())
out = []
for j in sorted(glob.glob('results/*8b*.json')):
    base = os.path.basename(j)[:-5]
    if not base.startswith(('gate_llama8b_rome_cf_L24_s0', 'gate_llama8b_rome_cf_L16_s0', 'g4_llama8b_alphaHO_cf_')):
        continue
    z = 'results/matrices/' + base + '.npz'
    row = {'json': j, 'npz_found': os.path.exists(z), 'touched_this_run': os.path.getmtime(j) >= t0}
    try:
        d = json.load(open(j)); row['json_ok'] = True; row['esr'] = d.get('edit_success_rate')
    except Exception as e:
        row['json_ok'] = False; row['err'] = str(e)
    out.append(row)
print(json.dumps({'n': len(out), 'rows': out}, indent=1))
EOF
log "post: validation sweep -> results/8bcausal_validation.json"

# C4 causal aggregation (honest holdout-projector protocol) over the new 8B rome/alpha pairs,
# invocation pattern copied from run_deep_until1900.sh's C4-HONEST block (layers 8 12 -> 16 24).
if [ -f experiments/aggregate_g4_causal.py ] \
   && compgen -G "results/matrices/g4_llama8b_alphaHO_cf_L*_s0.npz" >/dev/null; then
  log "aggregate_g4_causal (8B, holdout projector) -> results/C4_causal_8b_table.json"
  tmp_out="results/.C4_causal_8b_table.json.tmp"
  $PY experiments/aggregate_g4_causal.py \
    --rome_glob 'results/matrices/gate_llama8b_rome_cf_L{L}_s0.npz' \
    --alpha_glob 'results/matrices/g4_llama8b_alphaHO_cf_L{L}_s0.npz' \
    --layers 16 24 --known --edit_ok --proj_source holdout \
    --out "$tmp_out" >> "$LOG" 2>&1 \
    && mv "$tmp_out" results/C4_causal_8b_table.json \
    && log "post: C4_causal_8b_table done (atomic)" \
    || { rm -f "$tmp_out"; log "FAIL aggregate_g4_causal 8b"; }
else
  log "skip C4-8B (aggregate_g4_causal.py or holdout alpha npz missing)"
fi

{
  echo "RUN_8BCAUSAL REPORT $(date '+%F %T')  jobs: ${n_done} done / ${n_fail} fail / ${n_skip} skipped  elapsed $(elapsed_min)m/${BUDGET_MIN}m  abort=${QUEUE_ABORT}"
  grep -E 'RUN |done |FAIL |SKIP|ABORT|PROGRESS|THERMAL|equiv-gate|integrity' "$LOG" | tail -60
} > engine/run_8bcausal_report.txt
log "================ RUN_8BCAUSAL COMPLETE (${n_done} done / ${n_fail} fail / ${n_skip} skip) ================"
echo "RUN_8BCAUSAL_DONE" >> "$LOG"
