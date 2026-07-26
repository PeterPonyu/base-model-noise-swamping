#!/usr/bin/env bash
# G4 causal test: after the GATE runner finishes (produces ROME L8/L10 seed-0
# matrices), run MATCHED AlphaEdit at the same configs, then compare damage by
# cosine quartile. Idempotent. Launch detached; it self-waits.
set -u
H=/home/zeyufu/Desktop/idea-feasibility-analysis/edit-harness
cd "$H"
PY=/home/zeyufu/miniconda3/envs/dl/bin/python3
LLAMA=data/models/Llama-3.2-1B
G4LOG=engine/g4.log
log(){ echo "[$(date '+%F %T')] $*" >> "$G4LOG"; }

log "g4: waiting for engine + GATE runner to finish..."
while pgrep -f "python3 engine.py" >/dev/null || pgrep -f "run_gate.sh" >/dev/null; do sleep 60; done
if grep -q STOPPED_GPU_WEDGE engine/state.json 2>/dev/null; then
  log "ABORT: engine stopped on a GPU wedge — fix nvidia_uvm, re-run, then relaunch. Not starting."
  exit 1
fi
log "gate done — running matched AlphaEdit (L8/L10, seed 0, N=200 M=500)"

for L in 8 10; do
  rome_npz="results/matrices/gate_llama1b_rome_cf_L${L}_s0.npz"
  if [ ! -f "$rome_npz" ]; then log "SKIP L${L}: matched ROME matrix missing ($rome_npz)"; continue; fi
  tag=g4_llama1b_alpha_cf_L${L}_s0
  if [ ! -f "results/${tag}.json" ]; then
    log "run ${tag}"
    env -u ALL_PROXY -u all_proxy HF_HUB_OFFLINE=1 "$PY" experiments/killgate_keygeom.py \
      --model "$LLAMA" --editor alpha --dataset counterfact --data data/counterfact.json \
      --n_edits 200 --n_probes 500 --steps 20 --lr 0.1 --layer "$L" --seed 0 --keep_ratio 0.99 \
      --save_matrices --out "results/${tag}.json" >> "$G4LOG" 2>&1 && log "done ${tag}" || log "FAIL ${tag}"
  fi
  "$PY" experiments/analyze_g4.py --rome "$rome_npz" \
    --alpha "results/matrices/${tag}.npz" --known --edit_ok \
    --out "results/G4_L${L}.json" >> "$G4LOG" 2>&1 && log "analyzed L${L} -> results/G4_L${L}.json"
done
log "=== G4 COMPLETE — see results/G4_L{8,10}.json for the causal verdict ==="
echo "G4_DONE" >> "$G4LOG"
