#!/usr/bin/env bash
# GPU driver chain 2026-07-07 — run the 7 queued drivers fast->slow, GPU-serial,
# PID-tracked, each idempotent/resumable. Fast ones finish first; slowest (gptj) last.
# Launch: nohup ./engine/chain_20260707.sh >> engine/chain_20260707.nohup.log 2>&1 &
# Stop:   kill by PID from engine/chain_20260707.pid (NEVER pkill -f).
set -u
cd /home/zeyufu/Desktop/idea-feasibility-analysis/edit-harness || exit 2
LOG=engine/chain_20260707.log
echo "$$" > engine/chain_20260707.pid
echo "[chain] START $(date '+%F %T %Z') pid=$$" | tee -a "$LOG"

# Enable the 6th-editor (GRACE) EGL row inside run_u6 (review+smoke passed).
touch engine/grace_ready.ok
echo "[chain] created engine/grace_ready.ok (grace EGL row armed)" | tee -a "$LOG"

run() {              # run <name> <BUDGET_MIN|-> <driver.sh>
  local name="$1" budget="$2" script="$3"
  local env=""; [ "$budget" != "-" ] && env="BUDGET_MIN=$budget"
  echo "[chain] >>> $name START $(date '+%T') ($env)" | tee -a "$LOG"
  env $env bash "$script" >> "engine/chain_${name}.log" 2>&1
  local rc=$?
  echo "[chain] <<< $name DONE rc=$rc $(date '+%T')" | tee -a "$LOG"
}

# order = ascending estimated GPU time; gptj (slowest) last, capped to its top cells
run gradsim_true  -    ./run_gradsim_true.sh   #  ~0.5h
run ripple        -    ./run_ripple.sh         #  ~2h
run 8bcausal      400  ./run_8bcausal.sh       #  ~4.5h
run mquake_law    500  ./run_mquake_law.sh     #  ~5.5h
run instruct      600  ./run_instruct.sh       #  ~7.5h
run u6            700  ./run_u6.sh             #  ~8.4h
run gptj          520  ./run_gptj.sh           #  ~8.7h (top cells; last)

echo "[chain] ALL DONE $(date '+%F %T %Z')" | tee -a "$LOG"
rm -f engine/chain_20260707.pid
