#!/bin/bash
# run_quantedit_e0.sh — QuantEdit E0 oracle-first kill-gate driver (2026-07-11).
# Template = run_revins.sh skeleton (preflight / DRYRUN gate / GPU-idle gate / run_row /
# report), QUANTEDIT-namespaced (own pid/log/markers; never reuses revins/u6/gptj names).
#
# WHAT E0 IS. A per-edit closed-form "will this ROME edit survive post-training quantization"
# PREDICTOR, validated against a CPU-numpy EXACT PTQ oracle, BEFORE any behavioral GPU rung
# (E1+, ~140 GPU-min) is spent. Scored by experiments/quantedit_e0.py (CPU-only). The GPU is
# used for ONE thing only: dumping the per-edit rank-one factors (k, v-Wk) + base weight so
# the CPU oracle has something to quantize. Everything after the dump is CPU.
#
# CLAIM SCOPE (NARROWED per the 2026-07-11 literature re-audit — VERDICT: NARROWED).
#   arXiv 2605.15138 (MANSU, May 2026, *unlearning*) already published BOTH the sub-bin-width
#   grid-crossing physics AND a no-retraining per-parameter magnitude-FLOOR fix. Therefore the
#   ONLY admissible novel contribution here is the *per-edit, closed-form, a-priori survival
#   PREDICTOR* from a rank-one EDIT's own (k, v-Wk) geometry, for insertion-class locate-then-
#   edit editing (ROME / MEMIT / AlphaEdit), NOT unlearning. A margin-floor FIX would be a
#   corollary citing 2605.15138 as the mechanism's origin — E0 implements NO fix (oracle +
#   predictor only). Prior art cited together: 2407.06483 (Composable Interventions, ICLR'25),
#   2410.16454 (edit-then-quantize degradation), 2605.15138 (MANSU sub-bin-width + floor fix).
#
# DUMP-SOURCE DEVIATION (flagged; deliberate, with reason). The E0 spec text names
# `mechanism_dump.py --save_vectors` as the dump side. That is INSUFFICIENT for the EXACT
# oracle: mechanism_dump.py --save_vectors persists ONLY the residual VECTOR r=v-Wk
# (npz key `resid_vecs` [N, hidden]) — it does NOT save the edit key k, nor the base weight
# Wbase. The exact oracle Q(Wbase+ΔW)-Q(Wbase) requires the FULL base weight and the rank-one
# factors. The correct dump is `killgate_keygeom.py --save_vectors` (experiments/
# killgate_keygeom.py:948-993), which is explicitly namespaced "scored by
# experiments/quantedit_e0.py" and writes results/vectors/vectors_<tag>.npz with
# K [N,d_in], A=v-Wk [N,d_out], B=k/(k·k) [N,d_in], Wbase [d_out,d_in], recon_rel_err,
# vectors_valid, + provenance. This driver therefore dumps via killgate_keygeom.py
# --save_vectors. (The 8 real banks results/vectors/vectors_qv_*.npz were produced this way.)
#
# SEED-TAGGING (collision hazard, avoided). mechanism_dump.py names its output
# `{model_tag}_L{layer}.npz` with NO SEED -> two seeds at one layer silently overwrite (the
# hazard documented in run_revins.sh's header). killgate_keygeom.py --save_vectors instead
# names the file `vectors_<basename(--out)>.npz`, so putting L{L}_s{S} in --out makes every
# dump seed+layer-tagged with no collision. This driver does exactly that.
#
# GPU-HOUR ESTIMATE. The DEFAULT cell (L12 s0) already has a valid bank on disk
# (results/vectors/vectors_qv_llama1b_rome_cf_L12_s0.npz), so a default run is CPU-ONLY
# (idempotent skip of the dump). A dump of a MISSING cell is ~1 killgate pass at n_edits=200
# / n_probes=50 (~15-20 GPU-min, "mostly CPU" as the spec bills it). Scoring is CPU (a few
# minutes for the full 5-rung spec ladder over the requested banks).
#
# MODES / ENV:
#   DRYRUN=1     print every planned command (selftest / dump / score) without executing;
#                appends to the log ($LOG) only — NO pidfile write, NO results/ writes, NO GPU
#                gate (so a DRYRUN can never clobber a concurrent live run's pidfile).
#   LAYERS="12"  space/comma-separated edit layers to ensure a bank for (default "12").
#   SEED=0       seed for the dump cells (default 0).
#   BUDGET_MIN   wall-clock budget for the GPU-idle gate wait (default 600).
set -u
H=/home/zeyufu/Desktop/idea-feasibility-analysis/edit-harness
cd "$H" || exit 1
PY=/home/zeyufu/miniconda3/envs/dl/bin/python3
LOG=engine/run_quantedit_e0.log
DRYRUN=${DRYRUN:-0}
LAYERS=${LAYERS:-12}
SEED=${SEED:-0}
BUDGET_MIN=${BUDGET_MIN:-600}
LAYERS=${LAYERS//,/ }
mkdir -p engine   # needed for $LOG in every mode
# a live run owns the pidfile + results/ tree; a DRYRUN must NOT clobber a concurrent live
# run's pidfile (fixes the header's "no pidfile" claim) — so gate both behind non-DRYRUN.
if [ "$DRYRUN" -ne 1 ]; then
  mkdir -p results/vectors results/quantedit results/quantedit/selftest
  echo $$ > engine/run_quantedit_e0.pid
fi
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; echo "[qe0] $*"; }
ENVP="env -u ALL_PROXY -u all_proxy HF_HUB_OFFLINE=1"
KG="experiments/killgate_keygeom.py"
QE="experiments/quantedit_e0.py"
MODEL="data/models/Llama-3.2-1B"
log "================ RUN_QUANTEDIT_E0 START (pid $$, DRYRUN=${DRYRUN}, layers='${LAYERS}' seed=${SEED}) ================"

# ---------------------------------------------------------------- Phase 0: CPU preflight (HARD: code/tool presence only)
pf_fail=0
pf(){ if eval "$2"; then log "preflight OK: $1"; else log "PREFLIGHT-FAIL: $1"; pf_fail=1; fi; }
pf "python env (torch+numpy)"          "$PY -c 'import torch, numpy' 2>/dev/null"
pf "counterfact.json"                  "[ -f data/counterfact.json ]"
pf "killgate --save_vectors flag"      "grep -q -- '--save_vectors' $KG"
pf "killgate --vector_dir flag"        "grep -q -- '--vector_dir' $KG"
pf "quantedit_e0.py present"           "[ -f $QE ]"
pf "quantedit_e0 --from_vectors flag"  "grep -q -- '--from_vectors' $QE"
pf "quantedit_e0 --selftest flag"      "grep -q -- '--selftest' $QE"
pf "quantedit_e0 --ladder flag"        "grep -q -- '--ladder' $QE"
pf "quantedit_e0 --validate_npz flag"  "grep -q -- '--validate_npz' $QE"
pf "model Llama-3.2-1B"                 "[ -d $MODEL ]"
if [ "$pf_fail" -ne 0 ]; then log "ABORT: preflight failed (code/tool presence)"; exit 3; fi

# ---------------------------------------------------------------- Phase 1: CPU selftest gate (validate the math BEFORE any GPU)
# A defined-but-unproven oracle is worse than none. Run the CPU selftest first; if the
# quantizer math or the gate logic is broken, abort before spending GPU on a dump.
if [ "$DRYRUN" -eq 1 ]; then
  log "DRYRUN cmd (selftest): $ENVP $PY $QE --selftest --ladder spec"
else
  log "RUN selftest (CPU, quantizer + gate-logic validation)"
  if $ENVP $PY $QE --selftest --ladder spec >> "$LOG" 2>&1; then
    log "selftest PASS"
  else
    log "ABORT: --selftest FAILED (quantizer/gate math broken) — see $LOG"; exit 4
  fi
fi

# ---------------------------------------------------------------- Phase 2: ensure a bank per (layer, seed) cell
# Idempotent: if a bank already exists and passes the quantedit_e0 --validate_npz HARD gate,
# skip the GPU dump entirely (the default L12 s0 cell is already on disk -> CPU-only run).
BANKS=""
gpu_gate_opened=0
gpu_idle_gate(){
  # open the GPU-idle gate at most ONCE, lazily, only if a real dump is actually needed.
  [ "$gpu_gate_opened" -eq 1 ] && return 0
  local gate_t0 consec line util mem; gate_t0=$(date +%s); consec=0
  while [ "$consec" -lt 3 ]; do
    line=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    util=$(echo "$line" | awk -F, '{gsub(/[^0-9]/,"",$1); print $1}')
    mem=$(echo "$line"  | awk -F, '{gsub(/[^0-9]/,"",$2); print $2}')
    if [ -n "$util" ] && [ -n "$mem" ] && [ "$util" -lt 25 ] && [ "$mem" -lt 1500 ]; then
      consec=$((consec+1))
    else
      consec=0
      if [ $(( $(date +%s) - gate_t0 )) -gt $(( BUDGET_MIN * 60 )) ]; then
        log "ABORT: GPU busy > ${BUDGET_MIN}min at dump gate"; exit 2; fi
    fi
    log "gpu poll util=${util:-NA} mem=${mem:-NA} consec=${consec}/3"
    [ "$consec" -lt 3 ] && sleep 30
  done
  gpu_gate_opened=1; log "GPU idle — dump window open"
}

for L in $LAYERS; do
  bank="results/vectors/vectors_qv_llama1b_rome_cf_L${L}_s${SEED}.npz"
  BANKS="$BANKS $bank"
  if [ "$DRYRUN" -eq 1 ]; then
    log "DRYRUN cell L${L} s${SEED}: bank=${bank}"
    log "DRYRUN   validate: $ENVP $PY $QE --validate_npz ${bank}"
    log "DRYRUN   dump-if-missing: $ENVP $PY $KG --model $MODEL --editor rome --dataset counterfact --data data/counterfact.json --n_edits 200 --n_probes 50 --steps 20 --lr 0.1 --layer ${L} --seed ${SEED} --save_vectors --vector_dir results/vectors --out results/vectors/qv_llama1b_rome_cf_L${L}_s${SEED}.json"
    continue
  fi
  if [ -f "$bank" ] && $ENVP $PY $QE --validate_npz "$bank" >/dev/null 2>&1; then
    log "cell L${L} s${SEED}: bank exists + validates -> SKIP dump (no GPU)"
    continue
  fi
  log "cell L${L} s${SEED}: bank missing/invalid -> GPU dump via killgate --save_vectors"
  gpu_idle_gate
  dump_out="results/vectors/qv_llama1b_rome_cf_L${L}_s${SEED}.json"
  $ENVP $PY $KG --model "$MODEL" --editor rome \
      --dataset counterfact --data data/counterfact.json \
      --n_edits 200 --n_probes 50 --steps 20 --lr 0.1 \
      --layer "$L" --seed "$SEED" \
      --save_vectors --vector_dir results/vectors --out "$dump_out" >> "$LOG" 2>&1
  rc=$?
  if [ "$rc" -ne 0 ] || ! $ENVP $PY $QE --validate_npz "$bank" >/dev/null 2>&1; then
    log "FAIL cell L${L} s${SEED}: dump rc=${rc} or bank still invalid"
  else
    log "done cell L${L} s${SEED}: bank dumped + validated"
  fi
done

# ---------------------------------------------------------------- Phase 3: CPU scoring -> E0 oracle table
BANKS=$(echo "$BANKS" | tr -s ' ')
if [ "$DRYRUN" -eq 1 ]; then
  log "DRYRUN cmd (score): $ENVP $PY $QE --from_vectors${BANKS} --ladder spec --out results/quantedit/E0_oracle_table.json"
  log "================ RUN_QUANTEDIT_E0 DRYRUN COMPLETE ================"
  exit 0
fi
# score only banks that actually exist + validate (a failed dump must not poison the table)
score_banks=""
for b in $BANKS; do
  if [ -f "$b" ] && $ENVP $PY $QE --validate_npz "$b" >/dev/null 2>&1; then score_banks="$score_banks $b"; fi
done
if [ -z "$(echo "$score_banks" | tr -d ' ')" ]; then
  log "ABORT: no valid bank to score"; exit 5
fi
log "RUN score (CPU): banks =${score_banks}"
if $ENVP $PY $QE --from_vectors $score_banks --ladder spec \
     --out results/quantedit/E0_oracle_table.json >> "$LOG" 2>&1; then
  verdict=$($PY -c "import json;print(json.load(open('results/quantedit/E0_oracle_table.json'))['verdict'])" 2>/dev/null)
  log "score DONE -> results/quantedit/E0_oracle_table.json  VERDICT=${verdict:-?}"
else
  log "FAIL score"; exit 6
fi
log "================ RUN_QUANTEDIT_E0 COMPLETE ================"
