#!/usr/bin/env bash
# chain_36039_20260731.sh — dual-card gap-closure master chain, ON-BOX (2026-07-31).
#
# Orchestrates every unblocked GPU hole from PLAN-GAP-CLOSURE-MASTER-2026-07-31:
#   lane0 (GPU 0): H1  — resume run_phi_b6_refix.sh (idempotent: insertion s2 + 3 deletion cells)
#                  H5  — A-RAND random-direction control, 12 cells (prereg L8/L10/L12/L14 × s0/1/2;
#                        master-plan said 9 — PREREG BINDS, 12 run). GATED on RATIFIED prereg.
#   lane1 (GPU 1): H6  — alphaHO L10/L14 × s0/1/2 (6 cells, run_b6ins.sh Cell H CLI verbatim)
#                  H3  — u1e0_qwen15b_delete_refusal_L21 s1/s2 (2 cells)
#                  H8  — d2-prospective wave (existing infra). GATED on RATIFIED prereg.
# Gated cells SKIP CLEANLY (log + rc 0) when their prereg is not STATUS: RATIFIED on-box —
# no agent ratifies anything; the user edits the prereg STATUS line and re-syncs.
#
# Idempotent: every cell fast-skips on existing validated output. Kill-by-PID only.
set -u
H="${H:-/root/edit-harness-deploy-20260727}"; cd "$H" || exit 2
PY="${PY:-/root/autodl-tmp/venvs/ifa-20260727/bin/python}"
KG="experiments/killgate_keygeom.py"
CF="--dataset counterfact --data data/counterfact.json"
COMMON="--n_edits 200 --n_probes 500 --steps 20 --lr 0.1 --save_matrices --matrix_dir results/matrices"
ENVP="env -u ALL_PROXY -u all_proxy HF_HUB_OFFLINE=1"
PREREG_RAND="docs/plans/PREREG-B6-RANDOM-DIRECTION-CONTROL-2026-07-30.md"
PREREG_D2P="docs/plans/PREREG-D2-PROSPECTIVE-2026-07-26.md"
LOG="engine/chain_36039_20260731.log"
mkdir -p engine results/matrices
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# ---------------------------------------------------------------- lane machinery
idle_gate(){  # $1 = GPU index; util<25 && mem<1500, 3 consecutive, 30s apart
  local gpu="$1" consec=0 tries=0 line util mem
  while [ "$consec" -lt 3 ]; do
    [ "$tries" -ge 60 ] && { log "lane gpu$gpu ABORT: card never idle after 30min"; return 9; }
    tries=$((tries+1))
    line=$(nvidia-smi -i "$gpu" --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    util=$(echo "$line" | awk -F, '{gsub(/[^0-9]/,"",$1); print $1}')
    mem=$(echo "$line"  | awk -F, '{gsub(/[^0-9]/,"",$2); print $2}')
    if [ -n "${util:-}" ] && [ -n "${mem:-}" ] && [ "$util" -lt 25 ] && [ "$mem" -lt 1500 ]; then
      consec=$((consec+1)); else consec=0; fi
    [ "$consec" -lt 3 ] && sleep 30
  done
  log "lane gpu$gpu idle — window opens"; return 0
}

run_cell(){  # GPU TAG EST_MIN OUT_JSON CMD...  (skip-if-done, 2-fails abort per lane)
  local gpu="$1" tag="$2" est="$3" out="$4"; shift 4
  local npz="results/matrices/$(basename "${out%.json}").npz"
  if [ -e "$out" ] && [ -e "$npz" ]; then log "SKIP $tag (exists)"; return 0; fi
  [ "$(cat "engine/chain0731_lane${gpu}.fails" 2>/dev/null || echo 0)" -ge 2 ] && { log "ABORT-AFTER-2-FAILS before $tag"; return 8; }
  log "RUN $tag (gpu$gpu, est ${est}m)"
  CUDA_VISIBLE_DEVICES="$gpu" timeout --signal=TERM --kill-after=60 "$(( ${JOB_CAP_MIN:-60} * 60 ))s" \
    bash -c "$*" >> "engine/chain0731_${tag}.log" 2>&1
  local rc=$?
  if [ "$rc" -eq 0 ] && [ -e "$out" ]; then
    log "DONE $tag"; return 0
  fi
  log "FAIL $tag rc=$rc"; echo $(( $(cat "engine/chain0731_lane${gpu}.fails" 2>/dev/null || echo 0) + 1 )) > "engine/chain0731_lane${gpu}.fails"
  return 1
}

# ---------------------------------------------------------------- lane0: H1 resume, then H5 (gated)
lane0(){
  idle_gate 0 || return 9
  log "lane0: H1 — relaunch run_phi_b6_refix.sh (idempotent resume)"
  CUDA_VISIBLE_DEVICES=0 bash engine/run_phi_b6_refix.sh >> engine/chain0731_phi_refix.log 2>&1
  local rc=$?
  log "lane0: phi refix driver exited rc=$rc"
  # ---- H5 A-RAND (gated on user-ratified prereg) ----
  if [ -f "$PREREG_RAND" ] && grep -qx 'STATUS: RATIFIED' "$PREREG_RAND"; then
    log "lane0: H5 A-RAND prereg RATIFIED — 12 cells"
    for L in 8 10 12 14; do
      for s in 0 1 2; do
        run_cell 0 "arand_llama1b_L${L}_s${s}" 20 "results/arand_llama1b_rome_cf_L${L}_s${s}.json" \
          "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor rome $CF $COMMON \
           --layer $L --seed $s --rank_one_random \
           --rank_one_random_norms results/matrices/gate_llama1b_rome_cf_L${L}_s${s}.npz \
           --out results/arand_llama1b_rome_cf_L${L}_s${s}.json"
      done
    done
  else
    log "lane0: H5 SKIPPED — $PREREG_RAND not RATIFIED on-box (user gate; re-sync after ratification)"
  fi
  log "lane0 COMPLETE"; return 0
}

# ---------------------------------------------------------------- lane1: H6 → H3 → H8 (H8 gated)
lane1(){
  idle_gate 1 || return 9
  log "lane1: H6 — alphaHO L10/L14 × s0/1/2 (run_b6ins.sh Cell H CLI verbatim)"
  for L in 10 14; do
    for s in 0 1 2; do
      run_cell 1 "alphaHO_L${L}_s${s}" 30 "results/g4_llama1b_alphaHO_cf_L${L}_s${s}.json" \
        "$ENVP $PY $KG --model data/models/Llama-3.2-1B --editor alpha $CF $COMMON \
         --layer $L --seed $s --alpha_proj_source holdout --holdout_frac 1.0 \
         --out results/g4_llama1b_alphaHO_cf_L${L}_s${s}.json"
    done
  done
  log "lane1: H3 — qwen15b delete refusal L21 s1/s2"
  for s in 1 2; do
    run_cell 1 "u1e0_qwen15b_delete_refusal_L21_s${s}" 45 "results/u1e0_qwen15b_delete_refusal_L21_s${s}.json" \
      "$ENVP $PY $KG --model data/models/Qwen2.5-1.5B --editor rome --edit_mode delete \
       --delete_variant refusal $CF $COMMON --layer 21 --seed $s \
       --out results/u1e0_qwen15b_delete_refusal_L21_s${s}.json"
  done
  # ---- H8 d2-prospective (gated on user-ratified prereg) ----
  if [ -f "$PREREG_D2P" ] && grep -qx 'STATUS: RATIFIED' "$PREREG_D2P"; then
    log "lane1: H8 d2-prospective prereg RATIFIED — prepare+launch wave"
    bash engine/box_prepare_wave.sh d2-prospective check \
      && WAVE_BOX="$(hostname)" bash engine/box_launch_wave.sh d2-prospective \
      || log "lane1: H8 wave launch blocked (see box_prepare_wave FAIL lines)"
  else
    log "lane1: H8 SKIPPED — $PREREG_D2P not RATIFIED on-box (user gate; re-sync after ratification)"
  fi
  log "lane1 COMPLETE"; return 0
}

# ---- lane dispatch FIRST: a lane re-invocation must skip the pidfile guard and
# preflight below (the parent already holds the pid and passed preflight).
if [ "${1:-}" = "--lane0" ]; then lane0; exit $?; fi
if [ "${1:-}" = "--lane1" ]; then lane1; exit $?; fi

# ---------------------------------------------------------------- main: guard + preflight + launch
[ -f engine/chain_36039_20260731.pid ] && kill -0 "$(cat engine/chain_36039_20260731.pid 2>/dev/null)" 2>/dev/null \
  && { echo "REFUSE: already running (pid $(cat engine/chain_36039_20260731.pid))" >&2; exit 7; }
echo $$ > engine/chain_36039_20260731.pid
trap 'rm -f engine/chain_36039_20260731.pid' EXIT

CLOUD_PY=/root/autodl-tmp/venvs/ifa-20260727/bin/python bash engine/box_preflight_onbox.sh generic || { log "ABORT: preflight blocked"; exit 3; }
for m in Phi-3.5-mini Llama-3.2-1B Qwen2.5-1.5B; do
  [ -d "data/models/$m" ] || { log "ABORT: missing model data/models/$m (download before chain)"; exit 3; }
  $PY experiments/selftest_target_token.py --tokenizer "data/models/$m" >/dev/null 2>&1 \
    || { log "ABORT: tokenizer gate FAIL on $m (07-30 collision class)"; exit 3; }
done
log "preflight GREEN (3 models + tokenizer gates)"

rm -f engine/chain0731_lane0.fails engine/chain0731_lane1.fails
setsid bash "$0" --lane0 >> "$LOG" 2>&1 &
L0=$!
setsid bash "$0" --lane1 >> "$LOG" 2>&1 &
L1=$!
echo "$L0" > engine/chain0731_lane0.pid; echo "$L1" > engine/chain0731_lane1.pid
log "lanes launched: lane0 pid=$L0 lane1 pid=$L1"
wait "$L0"; rc0=$?
wait "$L1"; rc1=$?
rm -f engine/chain0731_lane0.pid engine/chain0731_lane1.pid
{
  echo "CHAIN 36039 20260731 REPORT host=$(hostname) lane0_rc=$rc0 lane1_rc=$rc1"
  grep -E 'RUN |DONE |FAIL |SKIP |ABORT|COMPLETE' "$LOG" | tail -80
} > engine/chain_36039_20260731.report
log "CHAIN END lane0_rc=$rc0 lane1_rc=$rc1"
[ "$rc0" -eq 0 ] && [ "$rc1" -eq 0 ]
