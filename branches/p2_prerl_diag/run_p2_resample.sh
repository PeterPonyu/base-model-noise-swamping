#!/bin/bash
# run_p2_resample.sh — PREREG-P2-GRPO-20260710.md §2 "measure them better" path:
# fresh, LARGER pre-RL samples for the three UNUSABLE checkpoints so the panel can
# reach n>=6 USABLE before any GRPO spend. Protocol identical to the 07-10 filler
# runs (sample_ckpt.py, GSM8K test, k=8, temp 0.9, top-p 1.0, max-new 640, seed 0);
# only --n-problems grows. --n-problems is a PREFIX slice (sample_ckpt.py:126), so
# each new file supersets the original 200 problems.
#
# Sizing (from the 07-10 measured rates, results/<id>.json):
#   gemma-2-2b   n=600  — n_right rate ~1.0%  -> E[n_right]~48>=20; CIw 1.71/sqrt(3)~1.0<=1.5
#   Llama-3.2-1B n=1000 — n_right rate ~0.31% -> E[n_right]~25>=20 (n_right-MARGINAL, the
#                          riskiest cell; prereg §9 kill condition applies if it misses)
#   Llama-3.2-3B n=1000 — CIw-MARGINAL: forward-computed 5x problems -> ~120 mixed ->
#                          CIw ~ 3.19/sqrt(5) ~ 1.43 vs the 1.5 gate — a squeaker (review
#                          m3); n_right already passes (27). If CIw lands >1.5, §9
#                          exclusion applies — do NOT relax the rule. (bf16, as 07-10)
# Measured 200-problem gen times: gemma 2836s / L1B 2064s / L3B 2178s -> est 8.5k/10.3k/10.9k s.
# Outputs are VERSIONED (<id>_nNNN.json) — originals untouched; prereg usability is
# evaluated on the new ids at sampling-freeze.
#
# --- POST-DRAIN FIX (2026-07-11, applied via apply_resample_fix.sh) ---
# 3 inline `conda run -n dl python3 - <<'EOF' ... EOF` validation/rollup
# sites below were VACUOUS: conda run swallows heredoc stdin, so each ran
# python on EMPTY stdin (rc 0, silent) — see workspace memory
# conda-run-swallows-stdin.md. Replaced with file+argv invocations:
# compute_overthinking_gap.py --validate-sample (2 sites, existing flag)
# and resample_usability_rollup.py (1 site, new small helper — the
# original inline rollup rule/output shape didn't match any existing
# compute_overthinking_gap.py mode, so it was NOT reused as-is). Applied
# after the driver had already drained; behavior contracts unchanged.
#
set -u
H=/home/zeyufu/Desktop/idea-feasibility-analysis
B=$H/branches/p2_prerl_diag
cd "$H" || exit 1
PY_ENV="env -u ALL_PROXY -u all_proxy"
CONDA_RUN="conda run -n dl"
LOG=$B/run_p2_resample.log
echo $$ > $B/run_p2_resample.pid
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
log "================ RUN_P2_RESAMPLE START (pid $$) ================"

# ---------------------------------------------------------------- preflight (CPU)
pf_fail=0
pf(){ if eval "$2"; then log "preflight OK: $1"; else log "PREFLIGHT-FAIL: $1"; pf_fail=1; fi; }
pf "sample_ckpt.py"        "[ -f $B/sample_ckpt.py ]"
pf "run_diag.py"           "[ -f $B/run_diag.py ]"
pf "model gemma-2-2b"      "[ -d $H/edit-harness/data/models/gemma-2-2b ]"
pf "model Llama-3.2-1B"    "[ -d $H/edit-harness/data/models/Llama-3.2-1B ]"
pf "model Llama-3.2-3B"    "[ -d $H/edit-harness/data/models/Llama-3.2-3B ]"
pf "python env"            "$CONDA_RUN python3 -c 'import torch, datasets' 2>/dev/null"
pf "disk >=20GB free"      "[ \$(df --output=avail -BG /home | tail -1 | tr -dc 0-9) -ge 20 ]"
if [ "$pf_fail" -ne 0 ]; then log "ABORT: preflight failed"; exit 3; fi

# ---------------------------------------------------------------- GPU idle gate (review-hardened, M1)
# Positive P3-done predicate + wider mem threshold. The instantaneous util/mem read alone
# is NOT sufficient: the outgroup sweep dips to idle-looking levels between model loads
# (barge-in risk), and the persistent baseline sits ~1.4GB (1500-jam risk — this
# workspace's burned-in idle-gate failure mode).
P3_PIDFILE=$H/branches/p3_agent_ipi/logs/run_p3_gpu.pid
p3_busy(){
  # (a) P3 driver alive (identity-checked: never trust a stale pidfile blindly)
  if [ -f "$P3_PIDFILE" ]; then
    local p; p=$(cat "$P3_PIDFILE" 2>/dev/null)
    if [ -n "$p" ] && kill -0 "$p" 2>/dev/null \
       && tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -q run_p3_gpu; then return 0; fi
  fi
  # (b) any ollama/llama-server compute-app still holding VRAM
  nvidia-smi --query-compute-apps=process_name --format=csv,noheader 2>/dev/null \
    | grep -qiE 'ollama|llama-server|llama-cpp' && return 0
  return 1
}
DRYRUN=${DRYRUN:-0}
if [ "$DRYRUN" -ne 1 ]; then
gate_t0=$(date +%s); consec=0
while [ "$consec" -lt 3 ]; do
  line=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  util=$(echo "$line" | awk -F, '{gsub(/[^0-9]/,"",$1); print $1}')
  mem=$(echo "$line"  | awk -F, '{gsub(/[^0-9]/,"",$2); print $2}')
  if ! p3_busy && [ -n "$util" ] && [ -n "$mem" ] && [ "$util" -lt 25 ] && [ "$mem" -lt 4000 ]; then
    consec=$((consec+1))
  else
    consec=0
    if [ $(( $(date +%s) - gate_t0 )) -gt 14400 ]; then log "ABORT: GPU busy >240min at gate"; exit 2; fi
  fi
  log "gpu poll util=${util:-NA} mem=${mem:-NA} p3_busy=$(p3_busy && echo yes || echo no) consec=${consec}/3"
  [ "$consec" -lt 3 ] && sleep 60
done
log "GPU idle + P3 drained — starting resample queue"
fi

n_done=0; n_fail=0
run_cell(){
  local id="$1" n="$2" cap="$3" dtype_flag="$4"
  local out="$B/samples/${id}_n${n}.json"
  local diag_out="$B/results/${id}_n${n}.json"
  if [ "$DRYRUN" -eq 1 ]; then
    log "DRYRUN ${id} n=${n} cap=${cap}s dtype='${dtype_flag}' -> $out"
    return
  fi
  if [ -f "$out" ] && [ -f "$diag_out" ]; then
    log "skip ${id}_n${n} (both outputs exist)"; return
  fi
  # review m2: never re-burn GPU-hours because the cheap CPU diag failed — skip gen
  # whenever a VALID sample already exists (sample_ckpt writes atomically, so an
  # existing $out is never truncated), and run diag independently below.
  if [ -f "$out" ] && $CONDA_RUN python3 $B/compute_overthinking_gap.py --validate-sample "$out" "$n" >> "$LOG" 2>&1
  then
    log "skip gen ${id}_n${n} (valid sample exists) — diag only"
  else
  [ -f "$out" ] && { log "STALE-INVALID sample ${id}_n${n} — quarantining"; mv "$out" "$out.INVALID"; }
  log "RUN gen ${id} n=${n} (cap ${cap}s) -> $out"
  local t rc; t=$(date +%s)
  timeout --signal=TERM --kill-after=60 "${cap}s" \
    $PY_ENV $CONDA_RUN python3 $B/sample_ckpt.py \
      --model edit-harness/data/models/${id} --dataset openai/gsm8k --config main \
      --split test --n-problems ${n} --k 8 --temperature 0.9 --top-p 1.0 \
      --max-new-tokens 640 ${dtype_flag} --seed 0 --out "$out" \
      >> "$B/run_p2_resample_${id}.log" 2>&1
  rc=$?
  local dt=$(( $(date +%s) - t ))
  if [ "$rc" -ne 0 ]; then log "FAIL gen ${id} (rc ${rc}, ${dt}s)"; n_fail=$((n_fail+1)); return; fi
  # validate: parses + expected problem count
  if ! $CONDA_RUN python3 $B/compute_overthinking_gap.py --validate-sample "$out" "$n" >> "$LOG" 2>&1
  then log "FAIL validate ${id} — quarantining"; mv "$out" "$out.INVALID"; n_fail=$((n_fail+1)); return; fi
  log "done gen ${id} (${dt}s)"
  fi
  log "RUN diag ${id}_n${n}"
  if $CONDA_RUN python3 $B/run_diag.py "$out" --id "${id}_n${n}" --n-boot 2000 --seed 0 >> "$B/run_p2_resample_${id}.log" 2>&1 \
     && [ -f "$diag_out" ]; then
    log "done diag ${id}_n${n}"; n_done=$((n_done+1))
  else
    log "FAIL diag ${id}_n${n}"; n_fail=$((n_fail+1))
  fi
}

# shortest first for early signal; bf16 ONLY for the 3B (matches 07-10 protocol exactly)
run_cell gemma-2-2b   600  13000 ""
run_cell Llama-3.2-1B 1000 15500 ""
run_cell Llama-3.2-3B 1000 16500 "--model_dtype bf16"

# ---------------------------------------------------------------- post: prereg usability check (CPU)
[ "$DRYRUN" -eq 1 ] && { log "DRYRUN complete"; exit 0; }
$CONDA_RUN python3 $B/resample_usability_rollup.py > $B/results/.RESAMPLE_usability_20260711.json.tmp 2>>"$LOG"
if [ -s $B/results/.RESAMPLE_usability_20260711.json.tmp ]; then
  mv $B/results/.RESAMPLE_usability_20260711.json.tmp $B/results/RESAMPLE_usability_20260711.json
  log "post: usability -> results/RESAMPLE_usability_20260711.json"
else
  rm -f $B/results/.RESAMPLE_usability_20260711.json.tmp
  log "post: usability roll-up FAILED (empty output; see log)"
fi
log "================ RUN_P2_RESAMPLE COMPLETE (${n_done} done / ${n_fail} fail) ================"
