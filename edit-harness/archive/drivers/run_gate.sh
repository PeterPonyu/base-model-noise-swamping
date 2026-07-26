#!/usr/bin/env bash
# G1 GATE: after the breadth engine finishes, re-run Llama-1B ROME at L8/L10/L12
# with raw-matrix dumps x 3 seeds, then run the partialled-correlation analyzer.
# Idempotent (skips configs whose JSON exists). Launch detached; it self-waits.
set -u
H=/home/zeyufu/Desktop/idea-feasibility-analysis/edit-harness
cd "$H"
PY=/home/zeyufu/miniconda3/envs/dl/bin/python3
LLAMA=data/models/Llama-3.2-1B
GLOG=engine/gate.log
log(){ echo "[$(date '+%F %T')] $*" >> "$GLOG"; }

log "gate runner: waiting for breadth engine to finish (no GPU contention)..."
while pgrep -f "python3 engine.py" >/dev/null; do sleep 60; done
if grep -q STOPPED_GPU_WEDGE engine/state.json 2>/dev/null; then
  log "ABORT: engine stopped on a GPU wedge — fix nvidia_uvm (sudo rmmod/modprobe), "
  log "       re-run engine.py, then relaunch this script. Not starting on a wedged GPU."
  exit 1
fi
log "engine done — starting GATE runs (Llama-1B L8/L10/L12 x seeds 0,1,2, N=200 M=500)"

for L in 8 10 12; do
  for S in 0 1 2; do
    tag=gate_llama1b_rome_cf_L${L}_s${S}
    if [ -f "results/${tag}.json" ]; then log "skip ${tag} (done)"; continue; fi
    log "run ${tag}"
    env -u ALL_PROXY -u all_proxy HF_HUB_OFFLINE=1 "$PY" experiments/killgate_keygeom.py \
      --model "$LLAMA" --editor rome --dataset counterfact --data data/counterfact.json \
      --n_edits 200 --n_probes 500 --steps 20 --lr 0.1 --layer "$L" --seed "$S" \
      --save_matrices --out "results/${tag}.json" >> "$GLOG" 2>&1 \
      && log "done ${tag}" || log "FAIL ${tag}"
  done
done

log "GATE runs complete — analyzing (within-probe partialled Spearman + permutation null)"
for L in 8 10 12; do
  "$PY" experiments/analyze_matrices.py "results/matrices/gate_llama1b_rome_cf_L${L}_s"*.npz \
    --metric logit --known --edit_ok --out "results/GATE_L${L}.json" >> "$GLOG" 2>&1 \
    && log "analyzed L${L} -> results/GATE_L${L}.json"
done
log "=== GATE COMPLETE — see results/GATE_L{8,10,12}.json for the verdict ==="
echo "GATE_DONE" >> "$GLOG"
