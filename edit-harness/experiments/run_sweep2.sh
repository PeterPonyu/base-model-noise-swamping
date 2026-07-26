#!/usr/bin/env bash
# Breadth axes 3 (editor) + 4 (dataset) + cross-architecture (Qwen). Run AFTER layer sweep.
set -u
H=/home/zeyufu/Desktop/idea-feasibility-analysis/edit-harness
cd "$H"
RUN(){ env -u ALL_PROXY -u all_proxy HF_HUB_OFFLINE=1 conda run -n dl python3 experiments/killgate_keygeom.py "$@"; }
N=150; M=400
LLAMA=data/models/Llama-3.2-1B
QWEN=data/models/Qwen2.5-1.5B
log(){ echo "[$(date +%H:%M:%S)] $*" >> results/sweep.log; }

# Axis 3 — editor: FT-L on Llama (key geometry should predict damage for FT-L too, not just ROME)
log "ft-L start"
RUN --model $LLAMA --editor ft --dataset counterfact --data data/counterfact.json \
    --n_edits $N --n_probes $M --steps 25 --ft_lr 5e-3 --layer 8 \
    --out results/sweep_llama1b_ft_cf_L8.json >> results/sweep.log 2>&1
log "ft-L done"

# Axis 4 — dataset: zsRE on Llama, ROME
log "zsre start"
RUN --model $LLAMA --editor rome --dataset zsre --data data/zsre_eval.json \
    --n_edits $N --n_probes $M --steps 20 --lr 0.1 --layer 8 \
    --out results/sweep_llama1b_rome_zsre_L8.json >> results/sweep.log 2>&1
log "zsre done"

# Cross-architecture — Qwen2.5-1.5B (different arch), ROME, CounterFact (only if downloaded)
if [ -f "$QWEN/model.safetensors" ]; then
  log "qwen start"
  RUN --model $QWEN --editor rome --dataset counterfact --data data/counterfact.json \
      --n_edits $N --n_probes $M --steps 20 --lr 0.1 --layer auto \
      --out results/sweep_qwen1.5b_rome_cf_Lauto.json >> results/sweep.log 2>&1
  log "qwen done"
else
  log "qwen SKIPPED (weights not present)"
fi
echo "SWEEP2_DONE" >> results/sweep.log
