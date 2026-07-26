#!/bin/bash
# run_p2_grpo.sh — PREREG-P2-GRPO-20260710.md §3-§9 GRPO validation-wave driver.
#
# Serial, GPU-idle-gated orchestrator (conventions mirror the reviewed
# run_p2_resample.sh: pidfile+identity, pf preflight, DRYRUN=1, idle gate
# util<25 && mem<4000 x3, per-job timeout, skip-if-valid, quarantine, atomic post).
#
# Per USABLE checkpoint (§2 rule via compute_overthinking_gap.py --usability-only,
# the single implementation), smallest model first:
#   1. train+merge : conda run -n dl-rl run_grpo.py --checkpoint <id>
#                    (rc 3=diverged / 4=oom / 5=timeout are §7 exclusions, NOT
#                     driver failures — logged and the queue continues)
#   2. post-RL gen : conda run -n dl sample_ckpt.py --model grpo_out/<id>/merged
#                    with n/k/temp/top_p/max_new/seed/dtype mirrored VERBATIM from
#                    the canonical pre-RL samples' _meta (comparability).
# Post: compute_overthinking_gap.py -> results/G_overthinking_test.json.
#
# §9 kill condition is enforced at preflight: usable panel < 6 aborts (exit 3).
# FORCE_DESCRIPTIVE=1 overrides ONLY to spend GPU on a descriptive-only panel —
# the analysis itself will still refuse to compute any rho below n=6.
#
# Env knobs: DRYRUN=1, JOB_CAP_MIN (train cap, default 720), SAMPLE_CAP_MIN
# (post-RL gen cap, default 300), FORCE_DESCRIPTIVE=1.
set -u
H=/home/zeyufu/Desktop/idea-feasibility-analysis
B=$H/branches/p2_prerl_diag
cd "$H" || exit 1
PY_ENV="env -u ALL_PROXY -u all_proxy"
RUN_DL="conda run -n dl"
RUN_DLRL="conda run -n dl-rl"
LOG=$B/run_p2_grpo.log
PIDFILE=$B/run_p2_grpo.pid
DRYRUN=${DRYRUN:-0}
JOB_CAP_MIN=${JOB_CAP_MIN:-720}
SAMPLE_CAP_MIN=${SAMPLE_CAP_MIN:-300}
FORCE_DESCRIPTIVE=${FORCE_DESCRIPTIVE:-0}

log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }

# refuse concurrent runs (identity-checked, never trust a stale pidfile blindly)
if [ -f "$PIDFILE" ]; then
  op=$(cat "$PIDFILE" 2>/dev/null)
  if [ -n "$op" ] && kill -0 "$op" 2>/dev/null \
     && tr '\0' ' ' < /proc/$op/cmdline 2>/dev/null | grep -q run_p2_grpo; then
    echo "run_p2_grpo: another instance (pid $op) is alive — refusing" >&2
    exit 2
  fi
fi
echo $$ > "$PIDFILE"
log "================ RUN_P2_GRPO START (pid $$) ================"
log "knobs: DRYRUN=$DRYRUN JOB_CAP_MIN=$JOB_CAP_MIN SAMPLE_CAP_MIN=$SAMPLE_CAP_MIN FORCE_DESCRIPTIVE=$FORCE_DESCRIPTIVE"

# ---------------------------------------------------------------- preflight (CPU)
pf_fail=0
pf(){ if eval "$2"; then log "preflight OK: $1"; else log "PREFLIGHT-FAIL: $1"; pf_fail=1; fi; }
pf "run_grpo.py"                 "[ -f $B/run_grpo.py ]"
pf "compute_overthinking_gap.py" "[ -f $B/compute_overthinking_gap.py ]"
pf "sample_ckpt.py"              "[ -f $B/sample_ckpt.py ]"
pf "dl-rl GRPOTrainer import (mergekit patch)" \
   "$RUN_DLRL python3 -c 'from trl import GRPOTrainer' 2>/dev/null"
pf "dl env"                      "$RUN_DL python3 -c 'import torch, datasets, numpy' 2>/dev/null"
pf "disk >=60GB free"            "[ \$(df --output=avail -BG /home | tail -1 | tr -dc 0-9) -ge 60 ]"
if [ "$pf_fail" -ne 0 ]; then log "ABORT: preflight failed"; exit 3; fi

# usability (§2) — the ONE implementation; also the §9 kill-condition gate
USA_JSON=$($RUN_DL python3 $B/compute_overthinking_gap.py --usability-only 2>>"$LOG")
if [ -z "$USA_JSON" ]; then log "ABORT: usability check produced no output"; exit 3; fi
N_USABLE=$(echo "$USA_JSON" | python3 -c "import json,sys; print(json.load(sys.stdin)['n_usable'])" 2>>"$LOG")
USABLE_LIST=$(echo "$USA_JSON" | python3 -c "import json,sys; print(' '.join(json.load(sys.stdin)['usable']))" 2>>"$LOG")
# review m4: a malformed (non-empty) usability JSON would leave N_USABLE empty and
# silently BYPASS the §9 gate — require a clean integer or abort
case "$N_USABLE" in
  ''|*[!0-9]*) log "ABORT: usability output unparseable (N_USABLE='$N_USABLE')"; exit 3;;
esac
log "usability: n_usable=$N_USABLE usable=[$USABLE_LIST]"
if [ "$N_USABLE" -lt 6 ] && [ "$FORCE_DESCRIPTIVE" -ne 1 ]; then
  log "ABORT (§9 kill condition): usable panel $N_USABLE < 6 at sampling-freeze."
  log "  Either more resampling restores the panel, or the study goes descriptive-only."
  log "  FORCE_DESCRIPTIVE=1 re-runs this driver for a descriptive-only GPU spend"
  log "  (the analysis will still refuse to compute rho below n=6 — that part is frozen)."
  exit 3
fi

# smallest-first queue order for early signal, filtered to the usable panel
SIZE_ORDER="Qwen2.5-0.5B Llama-3.2-1B Qwen2.5-1.5B gemma-2-2b Qwen2.5-3B Llama-3.2-3B Phi-3.5-mini"
QUEUE=""
for id in $SIZE_ORDER; do
  case " $USABLE_LIST " in *" $id "*) QUEUE="$QUEUE $id";; esac
done
log "queue (smallest first):$QUEUE"

# ---------------------------------------------------------------- GPU idle gate
# Positive predicate: the resample driver (identity-checked) must be dead — its
# inter-job CPU diag windows look GPU-idle and would invite a barge-in.
RESAMPLE_PIDFILE=$B/run_p2_resample.pid
resample_busy(){
  if [ -f "$RESAMPLE_PIDFILE" ]; then
    local p; p=$(cat "$RESAMPLE_PIDFILE" 2>/dev/null)
    if [ -n "$p" ] && kill -0 "$p" 2>/dev/null \
       && tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -q run_p2_resample; then return 0; fi
  fi
  return 1
}
if [ "$DRYRUN" -ne 1 ]; then
gate_t0=$(date +%s); consec=0
while [ "$consec" -lt 3 ]; do
  line=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  util=$(echo "$line" | awk -F, '{gsub(/[^0-9]/,"",$1); print $1}')
  mem=$(echo "$line"  | awk -F, '{gsub(/[^0-9]/,"",$2); print $2}')
  if ! resample_busy && [ -n "$util" ] && [ -n "$mem" ] && [ "$util" -lt 25 ] && [ "$mem" -lt 4000 ]; then
    consec=$((consec+1))
  else
    consec=0
    if [ $(( $(date +%s) - gate_t0 )) -gt 14400 ]; then log "ABORT: GPU busy >240min at gate"; exit 2; fi
  fi
  log "gpu poll util=${util:-NA} mem=${mem:-NA} resample_busy=$(resample_busy && echo yes || echo no) consec=${consec}/3"
  [ "$consec" -lt 3 ] && sleep 60
done
log "GPU idle + resample drained — starting GRPO queue"
fi

n_done=0; n_excl=0; n_fail=0

# NOTE: `conda run` SWALLOWS heredoc/piped stdin (verified live 2026-07-11:
# `echo 'print(1)' | conda run -n dl python3 -` runs python on EMPTY stdin, rc 0,
# no output — so inline-python validations through conda run pass VACUOUSLY).
# Every python helper below is therefore a FILE invocation with argv, never stdin.
train_status_of(){
  # prints the status field of grpo_out/<id>/train_status.json, or "absent"
  # (plain system python3: stdlib-only one-liner, no conda needed)
  python3 -c "
import json, sys, os
p = sys.argv[1]
print(json.load(open(p)).get('status', 'absent') if os.path.exists(p) else 'absent')
" "$B/grpo_out/$1/train_status.json" 2>/dev/null || echo absent
}

run_ckpt(){
  local id="$1"
  local post_out="$B/samples_postRL/${id}.json"

  # ---- canonical pre-RL meta (mirrored VERBATIM into the post-RL run) ----
  local meta
  meta=$($RUN_DL python3 $B/compute_overthinking_gap.py --print-meta "$id" 2>>"$LOG")
  if [ -z "$meta" ]; then log "FAIL ${id}: cannot read canonical pre-RL _meta"; n_fail=$((n_fail+1)); return; fi
  local n k temp topp maxnew seed dtype
  read -r n k temp topp maxnew seed dtype <<< "$meta"
  local dflag=""; [ "$dtype" = "bf16" ] && dflag="--model_dtype bf16"
  # review m2: SAMPLE_CAP_MIN is calibrated for n=200; the canonical n is mirrored
  # per checkpoint (gemma 600, a usable Llama-1B would be 1000) — scale the cap
  # proportionally per started 200-problem block so large-n rows can't time out
  # into a spurious quarantine on an already-marginal panel
  local cap_min=$(( SAMPLE_CAP_MIN * ( (n + 199) / 200 ) ))

  if [ "$DRYRUN" -eq 1 ]; then
    log "DRYRUN ${id}: train (cap ${JOB_CAP_MIN}m) -> grpo_out/${id}/merged; then sample n=${n} k=${k} dtype=${dtype} (cap ${cap_min}m) -> $post_out"
    return
  fi

  # ---- 1. train + merge (run_grpo.py is itself idempotent) ----
  local st; st=$(train_status_of "$id")
  if [ "$st" != "completed" ]; then
    log "RUN train ${id} (cap ${JOB_CAP_MIN}m)"
    local t rc; t=$(date +%s)
    timeout --signal=TERM --kill-after=120 $((JOB_CAP_MIN*60))s \
      $PY_ENV $RUN_DLRL python3 -u $B/run_grpo.py --checkpoint "$id" \
      >> "$B/run_p2_grpo_${id}.log" 2>&1
    rc=$?
    local dt=$(( $(date +%s) - t ))
    st=$(train_status_of "$id")
    case "$rc" in
      0) log "done train ${id} (${dt}s, status=$st)";;
      3|4|5) log "EXCLUDED ${id} (§7): rc=${rc} status=$st (${dt}s) — queue continues"
             n_excl=$((n_excl+1)); return;;
      # 124 = driver cap (SIGTERM honored); 137 = kill-after escalation (SIGKILL:
      # the python handler never ran, train_status may still say "running" —
      # compute_overthinking_gap excludes on status!=completed either way)
      124|137) log "EXCLUDED ${id} (§7): driver cap hit (rc=${rc}), status=$st (${dt}s)"
           n_excl=$((n_excl+1)); return;;
      *) log "FAIL train ${id} (rc ${rc}, status=$st, ${dt}s)"; n_fail=$((n_fail+1)); return;;
    esac
  else
    log "skip train ${id} (already completed)"
  fi

  # ---- 2. post-RL sampling (sample_ckpt.py UNCHANGED; params mirrored) ----
  if [ -f "$post_out" ] && $RUN_DL python3 $B/compute_overthinking_gap.py \
       --validate-sample "$post_out" "$n" >> "$LOG" 2>&1
  then
    log "skip post-RL gen ${id} (valid sample exists)"; n_done=$((n_done+1)); return
  fi
  [ -f "$post_out" ] && { log "STALE-INVALID post sample ${id} — quarantining"; mv "$post_out" "$post_out.INVALID"; }
  log "RUN post-RL gen ${id} n=${n} k=${k} dtype=${dtype} (cap ${cap_min}m)"
  local t2 rc2; t2=$(date +%s)
  timeout --signal=TERM --kill-after=60 $((cap_min*60))s \
    $PY_ENV $RUN_DL python3 $B/sample_ckpt.py \
      --model $B/grpo_out/${id}/merged --dataset openai/gsm8k --config main \
      --split test --n-problems "$n" --k "$k" --temperature "$temp" --top-p "$topp" \
      --max-new-tokens "$maxnew" $dflag --seed "$seed" --out "$post_out" \
      >> "$B/run_p2_grpo_${id}.log" 2>&1
  rc2=$?
  local dt2=$(( $(date +%s) - t2 ))
  if [ "$rc2" -ne 0 ]; then log "FAIL post-RL gen ${id} (rc ${rc2}, ${dt2}s)"; n_fail=$((n_fail+1)); return; fi
  if ! $RUN_DL python3 $B/compute_overthinking_gap.py \
       --validate-sample "$post_out" "$n" >> "$LOG" 2>&1
  then log "FAIL validate post-RL ${id} — quarantining"; mv "$post_out" "$post_out.INVALID"; n_fail=$((n_fail+1)); return; fi
  log "done post-RL gen ${id} (${dt2}s)"
  n_done=$((n_done+1))
}

mkdir -p "$B/samples_postRL" "$B/grpo_out"
for id in $QUEUE; do run_ckpt "$id"; done

# ---------------------------------------------------------------- post: §4-§9 analysis (CPU)
if [ "$DRYRUN" -eq 1 ]; then log "DRYRUN complete"; exit 0; fi
log "post: compute_overthinking_gap.py"
if $RUN_DL python3 $B/compute_overthinking_gap.py >> "$LOG" 2>&1 \
   && [ -f "$B/results/G_overthinking_test.json" ]; then
  log "post: wrote results/G_overthinking_test.json"
else
  log "post: G analysis FAILED (see log)"
fi
log "================ RUN_P2_GRPO COMPLETE (${n_done} sampled / ${n_excl} excluded / ${n_fail} fail) ================"
