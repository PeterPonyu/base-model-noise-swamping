#!/bin/bash
# onbox/d1_rawk_dump.sh — D1 sign-atlas raw-K dump driver (36039 dual-4090D box).
#
# WHAT: produce the cached raw edit-key banks the D1 sign-E0 needs
# (experiments/d1_sign_stat.py + docs/plans/PREREG-D1-SIGN-E0-20260714.md). Four
# families that currently have NO raw-K bank —
#     mistral7b     Mistral-7B-v0.3        32L  gate layers 16 20 24 28   (sign +)
#     llama31_8bi   Llama-3.1-8B-Instruct  32L  gate layers 16 20 24 28   (sign +)
#     gemma9b       gemma-2-9b(-bf16)      42L  gate layers 21 26 31 36   (sign +)
#     qwen3_8b      Qwen3-8B-Base          36L  gate layers 18 22 27 31   (sign -)
# each at its 4 gate layers, seed s0 -> 16 cells. Existing banks llama1b (L8/L12/L14)
# + qwen15b (L14) already cover 2 more families; after this wave the gate has 4 POS + 2
# NEG families (prereg §2.1/§3.2 — the 3 balancing Qwen negatives are the OPTIONAL
# FAMILIES-env extension that lowers the permutation floor below 0.05).
#
# HOW: this is NOT a new dump program. killgate_keygeom.py --save_vectors already emits
# the exact bank format d1_sign_stat.py reads (K[N,d], knorm, norm_growth, edit_ok,
# vectors_valid, model/editor/dataset/layer/seed provenance — verified byte-identical to
# the existing vectors_qv_* banks). The vectors path is derived by killgate from --out:
#   --out results/qv_<tag>_rome_cf_L<L>_s0.json  ->  results/vectors/vectors_qv_<tag>_rome_cf_L<L>_s0.npz
# so no experiments/d1_rawk_dump.py is needed and none is written (task's "if a new
# python entry is needed" — it is not).
#
# ROME value-opt stays fp32 INSIDE the editor (rome-edit-must-be-fp32 rule) even though
# the model loads bf16 — --model_dtype bf16 only sets the frozen-weight/forward dtype, the
# same setting the sign atlas itself was measured under (cloud/run_wave_36039.sh), so the
# dumped keys are the matched key-space of the runs that produced sign(rho_C). Never pass
# --model_dtype fp32 here (would OOM a 4090D at 9B AND change the key-space vs the atlas).
#
# OPS (burned-in lessons): explicit WAVE_BOX guard (no silent default-box run); budget
# clock starts at WORK start, not process start; skip-if-exists+valid per cell; killgate's
# own atomic tmp+os.replace writes; pid file waited by `kill -0` ONLY (never pgrep/pkill
# -f — watcher/self-match hazard); per-cell timeout; clear .done/.fail markers; DRYRUN=1
# prints the full command list and reports missing on-box models without executing.
#
# SHARDING: run twice with disjoint FAMILIES + CARD (no shared --out paths, no in-script
# backgrounding, no pattern-matched waits), e.g. on 36039:
#   CARD=0 FAMILIES="mistral7b gemma9b"   WAVE_BOX=36039 nohup ./onbox/d1_rawk_dump.sh &
#   CARD=1 FAMILIES="llama31_8bi qwen3_8b" WAVE_BOX=36039 nohup ./onbox/d1_rawk_dump.sh &
set -u
H="$(cd "$(dirname "$0")/.." && pwd)"; cd "$H" || exit 1
PY=${CLOUD_PY:-/root/miniconda3/bin/python}
M=${MODELS_DIR:-/root/autodl-tmp/models}
DATA="$H/data/counterfact.json"
RES="$H/results"; VEC="$RES/vectors"; ENG="$H/engine"
CARD=${CARD:-0}                      # defined BEFORE the card-namespaced paths below (set -u)
# Card-namespaced pid/log/report so two concurrent shard invocations (CARD=0 / CARD=1, per the
# header's dual-shard instruction) never clobber each other's files — same per-card suffix
# convention as cloud/run_wave_36039.sh's wave36039_card${card}.pid.
LOG="$ENG/d1_rawk_dump_card${CARD}.log"
BUDGET_MIN=${BUDGET_MIN:-260}        # per card/invocation; clock starts at WORK start below
JOB_CAP_MIN=${JOB_CAP_MIN:-45}       # per-cell hard timeout (7-9B ROME 200 edits ~15-30m)
EST_MIN=${EST_MIN:-30}               # per-cell budget-skip estimate
N_EDITS=${N_EDITS:-200}
N_PROBES=${N_PROBES:-100}            # K is PROBE-INDEPENDENT (captured during the edit, not
                                     # the probe sweep) -> 100 is enough for killgate to run
                                     # its pipeline; lower than the atlas's 500 only to hit
                                     # the ~2-3 GPU-h target. Does NOT change the dumped K.
SEED=0
FAMILIES=${FAMILIES:-"mistral7b llama31_8bi gemma9b qwen3_8b"}
WAVE_BOX=${WAVE_BOX:-}               # MUST be set to 36039 to actually run (no default-box)
EXPECT_BOX=36039
DRYRUN=${DRYRUN:-0}
STOP="$ENG/STOP_D1_RAWK"
mkdir -p "$RES" "$VEC" "$ENG" cloud/logs
echo $$ > "$ENG/d1_rawk_dump_card${CARD}.pid"
log(){ echo "[d1_rawk $(date '+%F %T')] $*" | tee -a "$LOG"; }
log "================ D1_RAWK_DUMP START (pid $$, box=${WAVE_BOX:-UNSET} card=${CARD} families=[${FAMILIES}] budget=${BUDGET_MIN}m dry=${DRYRUN}) ================"

# tag -> "dir n_layers L1 L2 L3 L4 expect_params"
spec(){
  case "$1" in
    mistral7b)   echo "$M/Mistral-7B-v0.3 32 16 20 24 28 7.248e9" ;;
    llama31_8bi) echo "$M/Llama-3.1-8B-Instruct 32 16 20 24 28 8.03e9" ;;
    gemma9b)     echo "$M/gemma-2-9b-bf16 42 21 26 31 36 9.2422e9" ;;
    qwen3_8b)    echo "$M/Qwen3-8B-Base 36 18 22 27 31 8.19e9" ;;
    *) return 1 ;;
  esac
}

# --------------------------------------------------------------- Phase 0: WAVE_BOX guard
# The local 5090 belongs to THIS workspace's B6 revision lane, not to this cloud dump; and
# the "WAVE_BOX default trap" (a defaulted box id silently ran a wave on the wrong machine)
# is a burned lesson. Require an explicit WAVE_BOX==36039. DRYRUN bypasses (plan-only).
if [ "$DRYRUN" -ne 1 ]; then
  if [ "$WAVE_BOX" != "$EXPECT_BOX" ]; then
    log "ABORT: WAVE_BOX='${WAVE_BOX:-UNSET}' != ${EXPECT_BOX}. Set WAVE_BOX=${EXPECT_BOX} to run "
    log "       on the intended dual-4090D box (this refuses to dump on any other machine)."
    exit 3
  fi
fi

# --------------------------------------------------------------- Phase 0a: CPU pre-flight (DRYRUN-soft)
pf_fail=0
pf(){ if eval "$2"; then log "preflight OK: $1"; else log "PREFLIGHT-FAIL: $1"; pf_fail=1; fi; }
if [ "$DRYRUN" -ne 1 ]; then
  pf "python env (torch+numpy)" "$PY -c 'import torch, numpy' 2>/dev/null"
  pf "killgate_keygeom.py present" "[ -f experiments/killgate_keygeom.py ]"
  pf "killgate has --save_vectors" "grep -q -- '--save_vectors' experiments/killgate_keygeom.py"
  pf "counterfact.json" "[ -f \"$DATA\" ]"
  if [ "$pf_fail" -ne 0 ]; then log "ABORT: CPU preflight failed"; exit 3; fi
else
  log "DRYRUN=1 — skipping python-exec preflight + GPU idle gate; printing the plan only"
fi

# --------------------------------------------------------------- Phase 0b: per-family model + confound preflight
# tie_word_embeddings is recorded (NOT a gate) — the prereg pre-kills embedding-tying as the
# sign driver (tied models on BOTH sign sides); this logs gemma/mistral/qwen3 tie values as
# the on-box confound evidence the prereg §4 asks for. Missing model dir -> the family's
# cells CONFIG-skip cleanly (no crash), so DRYRUN on the laptop reports absence gracefully.
tie_of(){ # dir -> true/false/absent(->gemma2 default true)/NA
  local cfg="$1/config.json"
  [ -f "$cfg" ] || { echo "NA(no config)"; return; }
  "$PY" - "$cfg" 2>/dev/null <<'EOF' || echo "NA(parse)"
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    print("NA(parse)"); raise SystemExit
if "tie_word_embeddings" in d:
    print(str(d["tie_word_embeddings"]).lower())
else:
    print("absent(gemma2-default-true)")
EOF
}
for tag in $FAMILIES; do
  s=$(spec "$tag") || { log "UNKNOWN-FAMILY ${tag} — skipping"; continue; }
  read -r dir nl _ <<< "$s"
  if [ -d "$dir" ]; then
    if [ "$DRYRUN" -ne 1 ]; then
      cfg_nl=$("$PY" - "$dir/config.json" 2>/dev/null <<'EOF'
import json,sys
try: print(json.load(open(sys.argv[1])).get("num_hidden_layers",-1))
except Exception: print(-1)
EOF
)
      [ "$cfg_nl" = "$nl" ] && log "model OK: ${tag} (${dir}, num_hidden_layers=${cfg_nl}, tie=$(tie_of "$dir"))" \
                            || log "MODEL-CONFIG-MISMATCH ${tag}: num_hidden_layers=${cfg_nl} != ${nl} — its cells CONFIG-skip"
    else
      log "model present: ${tag} (${dir})"
    fi
  else
    log "MODEL-ABSENT: ${tag} (${dir}) — its cells CONFIG-skip cleanly (dump not run on this box?)"
  fi
done

# --------------------------------------------------------------- Phase 0c: GPU idle gate (skipped on DRYRUN)
if [ "$DRYRUN" -ne 1 ]; then
  gate_t0=$(date +%s); consec=0
  while [ "$consec" -lt 3 ]; do
    # nvidia-smi IGNORES CUDA_VISIBLE_DEVICES (driver tool, not CUDA) — must use -i $CARD,
    # else every shard polls GPU 0 and non-zero cards wedge at the gate (burned 2026-07-14).
    line=$(nvidia-smi -i "$CARD" --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    util=$(echo "$line" | awk -F, '{gsub(/[^0-9]/,"",$1); print $1}')
    mem=$(echo "$line"  | awk -F, '{gsub(/[^0-9]/,"",$2); print $2}')
    if [ -n "$util" ] && [ -n "$mem" ] && [ "$util" -lt 25 ] && [ "$mem" -lt 1500 ]; then
      consec=$((consec+1))
    else
      consec=0
      if [ $(( $(date +%s) - gate_t0 )) -gt 1800 ]; then log "ABORT: GPU(card${CARD}) busy >30min at gate"; exit 2; fi
    fi
    log "gpu(card${CARD}) poll util=${util:-NA} mem=${mem:-NA} consec=${consec}/3"
    [ "$consec" -lt 3 ] && sleep 30
  done
  log "GPU(card${CARD}) idle — window opens now"
fi

# BUDGET CLOCK starts HERE (work start), not at process start (burned lesson).
T0=$(date +%s)
elapsed_min(){ echo $(( ( $(date +%s) - T0 ) / 60 )); }
ENVP="env -u ALL_PROXY -u all_proxy HF_HUB_OFFLINE=1"
n_done=0; n_fail=0; n_skip=0

# vectors-bank validity check (K present + vectors_valid==1) — reused for skip + post-run.
valid_vec(){
  "$PY" - "$1" 2>/dev/null <<'EOF'
import sys, numpy as np
try:
    d = np.load(sys.argv[1], allow_pickle=False)
except Exception as e:
    print(f"VEC-FAIL unreadable: {e}"); sys.exit(1)
if "K" not in d.files:
    print("VEC-FAIL no K"); sys.exit(1)
if d["K"].ndim != 2 or d["K"].shape[0] < 2:
    print(f"VEC-FAIL bad K shape {d['K'].shape}"); sys.exit(1)
vv = int(d["vectors_valid"].item()) if "vectors_valid" in d.files else 0
print(f"VEC-OK shape={tuple(d['K'].shape)} vectors_valid={vv}")
sys.exit(0 if vv == 1 else 2)
EOF
}

run_cell(){ # tag dir layer
  local tag="$1" dir="$2" L="$3"
  local out="$RES/qv_${tag}_rome_cf_L${L}_s${SEED}.json"
  local vnpz="$VEC/vectors_qv_${tag}_rome_cf_L${L}_s${SEED}.npz"
  local mk_done="$ENG/d1_rawk_${tag}_L${L}_s${SEED}.done"
  local mk_fail="$ENG/d1_rawk_${tag}_L${L}_s${SEED}.fail"
  local cmd="$ENVP CUDA_VISIBLE_DEVICES=$CARD $PY experiments/killgate_keygeom.py \
--model $dir --data $DATA --dataset counterfact \
--n_edits $N_EDITS --n_probes $N_PROBES --steps 20 --lr 0.1 \
--editor rome --layer $L --seed $SEED --model_dtype bf16 \
--save_vectors --out $out"

  if [ "$DRYRUN" -eq 1 ]; then
    echo "DRYRUN ${tag} L${L} s${SEED} -> ${vnpz}"
    echo "DRYRUN cmd: ${cmd}"
    log  "DRYRUN ${tag} L${L}: ${cmd}"
    return
  fi
  [ -f "$STOP" ] && { log "STOP-file present — not starting ${tag} L${L}"; return 9; }
  # skip-if-exists + valid
  if [ -f "$out" ] && [ -f "$vnpz" ] && valid_vec "$vnpz" >/dev/null 2>&1; then
    log "skip ${tag} L${L} (bank exists + valid)"; : > "$mk_done"; n_skip=$((n_skip+1)); return 0
  fi
  # model presence (CONFIG-skip, never crash)
  if [ ! -d "$dir" ]; then log "CONFIG-SKIP ${tag} L${L} (model dir ${dir} absent)"; n_skip=$((n_skip+1)); return 0; fi
  # budget
  local now; now=$(elapsed_min)
  if [ $(( now + EST_MIN + 2 )) -gt "$BUDGET_MIN" ]; then
    log "BUDGET-SKIP ${tag} L${L} (elapsed ${now}m + est ${EST_MIN}m > ${BUDGET_MIN}m)"; n_skip=$((n_skip+1)); return 0; fi

  local cap=$(( JOB_CAP_MIN * 60 )) t rc dt
  rm -f "$mk_fail"
  log "RUN ${tag} L${L} s${SEED} (cap ${JOB_CAP_MIN}m, elapsed ${now}m) -> cloud/logs/d1_${tag}_L${L}.log"
  t=$(date +%s)
  timeout --signal=TERM --kill-after=60 "${cap}s" bash -c "$cmd" > "cloud/logs/d1_${tag}_L${L}.log" 2>&1 </dev/null
  rc=$?; dt=$(( $(date +%s) - t ))
  if [ "$rc" -eq 0 ] && [ -f "$out" ] && [ -f "$vnpz" ]; then
    local v; v=$(valid_vec "$vnpz")
    if echo "$v" | grep -q "VEC-OK"; then
      log "done ${tag} L${L} (${dt}s) ${v}"; : > "$mk_done"; n_done=$((n_done+1))
    else
      log "FAIL ${tag} L${L} (${dt}s) INVALID-VEC: ${v}"; : > "$mk_fail"; n_fail=$((n_fail+1))
    fi
  else
    log "FAIL ${tag} L${L} (rc ${rc}, ${dt}s)"; : > "$mk_fail"; n_fail=$((n_fail+1))
  fi
}

# --------------------------------------------------------------- the dump sweep
for tag in $FAMILIES; do
  s=$(spec "$tag") || { log "UNKNOWN-FAMILY ${tag}"; continue; }
  read -r dir nl L1 L2 L3 L4 _ <<< "$s"
  for L in $L1 $L2 $L3 $L4; do
    [ -f "$STOP" ] && { log "STOP-file — halting sweep"; break 2; }
    run_cell "$tag" "$dir" "$L"
  done
done

# --------------------------------------------------------------- report
{
  echo "D1_RAWK_DUMP REPORT $(date '+%F %T')  card=${CARD}  ${n_done} done / ${n_fail} fail / ${n_skip} skip  elapsed $( [ "$DRYRUN" -eq 1 ] && echo dryrun || elapsed_min )m/${BUDGET_MIN}m"
  echo "banks -> $VEC/vectors_qv_{mistral7b,llama31_8bi,gemma9b,qwen3_8b}_rome_cf_L*_s0.npz"
  echo "next (ON-BOX CPU — banks are large, keep them on-box): python experiments/d1_sign_stat.py --vector_dir $VEC"
  echo "     then pull ONLY results/analysis/D1_sign_stat_table.json home (prereg docs/plans/PREREG-D1-SIGN-E0-20260714.md)"
} | tee "$ENG/d1_rawk_dump_card${CARD}_report.txt" | while read -r l; do log "$l"; done
log "================ D1_RAWK_DUMP COMPLETE (${n_done} done / ${n_fail} fail / ${n_skip} skip) ================"
echo "D1_RAWK_DUMP_DONE" >> "$LOG"
