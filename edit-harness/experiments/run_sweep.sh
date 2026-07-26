#!/usr/bin/env bash
# Breadth sweep: run killgate across a list of configs serially (GPU = one at a time).
# Each config writes results/sweep_<tag>.json. Extend the CONFIGS array to widen coverage.
set -u
H=/home/zeyufu/Desktop/idea-feasibility-analysis/edit-harness
cd "$H"
RUN() { env -u ALL_PROXY -u all_proxy HF_HUB_OFFLINE=1 conda run -n dl python3 experiments/killgate_keygeom.py "$@"; }
N=150; M=400   # moderate per-config size for broad coverage (60k pairs, ~10min each)
LLAMA=data/models/Llama-3.2-1B

log(){ echo "[$(date +%H:%M:%S)] $*" >> results/sweep.log; }

# ---- Axis 1: layer sweep (Llama-3.2-1B, ROME, CounterFact) ----
for L in 4 6 8 10 12 14; do
  log "layer $L start"
  RUN --model $LLAMA --data data/counterfact.json --n_edits $N --n_probes $M \
      --steps 20 --lr 0.1 --layer $L \
      --out results/sweep_llama1b_rome_cf_L${L}.json >> results/sweep.log 2>&1
  log "layer $L done"
done
echo "SWEEP_LAYERS_DONE" >> results/sweep.log
