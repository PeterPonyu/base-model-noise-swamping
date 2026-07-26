#!/bin/bash
# run_enhance_4090.sh — master orchestrator for the 2026-07-09 enhancement round on the
# single-4090 box (P1 GLUE bridge, P2 Pythia arch-vs-scale, P3 ripple layer completion).
#
# PHASE ORDER + the concurrent download (user directive 2026-07-09): the Pythia download
# (cloud/dl_pythia.py, ~8.5GB safetensors-only) starts FIRST in the background and runs
# during P1+P3 GPU work (~7-9h vs ~20min of download — enormous slack); P2 runs last and
# self-gates per model (MODEL-ABSENT / INTEGRITY-FAIL -> clean skip, never a crash).
#
# MONEY-SAFETY (this week's lessons, all baked in):
#   * NO_SHUTDOWN escape hatch checked before power-off (touch /root/NO_SHUTDOWN)
#   * zero-new-results guard: if the round produced NOTHING new, the box STAYS UP for
#     diagnosis instead of burning a boot cycle (96GB-box lesson, run 1-3)
#   * pair with cloud/failsafe_enhance.sh (24h hard power-off) at launch
#   * launcher keeps NO_SHUTDOWN armed until the P1 smoke row is verified running
set -u
H="$(cd "$(dirname "$0")" && pwd)"
cd "$H" || exit 1
LOG=${ENHANCE_LOG:-/root/enhance_4090.log}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} SKIP_IDLE_GATE=1 IDLE_GATE_DEVICE=${IDLE_GATE_DEVICE:-0}
export CLOUD_PY=${CLOUD_PY:-/root/miniconda3/bin/python}
mlog(){ echo "[enhance $(date '+%F %T')] $*" >> "$LOG"; }
mlog "================ ENHANCE-4090 START (pid $$) ================"

# result census for the shutdown guard — pure bash arithmetic (review MEDIUM-1: `bc`
# may be absent on a minimal cloud image; an empty count would have silently defeated
# the guard). ABSOLUTE count, not a baseline diff (review MEDIUM-2): on an idempotent
# resume where everything validated-skips, prior results still exist -> count>0 ->
# clean shutdown; only a round that has produced NOTHING usable keeps the box up.
count_round_results(){
  local a b c
  a=$(ls results/glue_bridge/gb_*.json 2>/dev/null | wc -l)
  b=$(ls results/ripple_llama1b_rome_popular_L8_s*.json results/ripple_llama1b_rome_popular_L10_s*.json \
     results/ripple_llama1b_rome_popular_L14_s*.json results/ripple_llama1b_alpha_popular_L12_s[12].json 2>/dev/null | wc -l)
  c=$(ls results/gate_pythia*_s*.json results/g4_pythia*_s*.json 2>/dev/null | wc -l)
  echo $(( a + b + c ))
}
mlog "round-result count at start: $(count_round_results)"

# Pythia download — background, network+disk only, concurrent with P1/P3 GPU work
( source /etc/network_turbo 2>/dev/null; exec "$CLOUD_PY" cloud/dl_pythia.py ) >> /root/dl_pythia.log 2>&1 &
DL_PID=$!
mlog "pythia download pid=${DL_PID} (log /root/dl_pythia.log)"

for d in run_glue_bridge.sh run_ripple_ext.sh; do
  mlog ">>> ${d} START"
  bash "$d" >> "$LOG" 2>&1
  mlog "<<< ${d} rc=$?"
done

if kill -0 "$DL_PID" 2>/dev/null; then mlog "P1/P3 done, download still running — waiting"; fi
wait "$DL_PID"; mlog "pythia download rc=$?"

mlog ">>> run_pythia.sh START"
bash run_pythia.sh >> "$LOG" 2>&1
mlog "<<< run_pythia.sh rc=$?"

# P0 refresh (CPU): editable-band table over everything now on disk
"$CLOUD_PY" experiments/esr_band_analysis.py >> "$LOG" 2>&1 \
  && mlog "esr_band_analysis refreshed" || mlog "esr_band_analysis FAILED (non-fatal)"

FINAL_COUNT=$(count_round_results)
mlog "final round-result count: ${FINAL_COUNT}"
if [ "${FINAL_COUNT:-0}" -eq 0 ]; then
  mlog "ZERO ROUND RESULTS — NOT shutting down; box stays up for diagnosis"
  exit 1
fi
mlog "ALL DONE — powering off in 300s (cancel: touch /root/NO_SHUTDOWN)"
sync
sleep 300
[ -f /root/NO_SHUTDOWN ] && { mlog "shutdown CANCELLED by /root/NO_SHUTDOWN"; exit 0; }
mlog "shutdown -h now"
shutdown -h now
