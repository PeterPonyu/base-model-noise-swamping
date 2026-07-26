#!/bin/bash
# G4 extend: make the H3 AlphaEdit causal test paper-grade — all 4 GATE layers x 3 seeds.
# Mirrors run_g4.sh exactly (--editor alpha, keep_ratio 0.99, N=200 M=500, --save_matrices).
# Idempotent: skips any (layer,seed) whose AlphaEdit result already exists.
# L14 is the scientifically key layer (norm-growth-dominant per the corrected G1 regime band):
# does AlphaEdit null-space projection still track key-cosine there, or track norm-growth?
set -u
H=/home/zeyufu/Desktop/idea-feasibility-analysis/edit-harness
cd "$H"
PY=/home/zeyufu/miniconda3/envs/dl/bin/python3
LLAMA=data/models/Llama-3.2-1B
LOG=engine/g4_extend.log
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
log "=== G4 extend START: AlphaEdit L8/L10/L12/L14 x seeds 0/1/2 (H3 paper-grade) ==="
for L in 8 10 12 14; do
  for S in 0 1 2; do
    rome_npz="results/matrices/gate_llama1b_rome_cf_L${L}_s${S}.npz"
    if [ ! -f "$rome_npz" ]; then log "SKIP L${L}s${S}: matched ROME matrix missing"; continue; fi
    tag=g4_llama1b_alpha_cf_L${L}_s${S}
    if [ ! -f "results/${tag}.json" ]; then
      log "run ${tag}"
      env -u ALL_PROXY -u all_proxy HF_HUB_OFFLINE=1 "$PY" experiments/killgate_keygeom.py \
        --model "$LLAMA" --editor alpha --dataset counterfact --data data/counterfact.json \
        --n_edits 200 --n_probes 500 --steps 20 --lr 0.1 --layer "$L" --seed "$S" --keep_ratio 0.99 \
        --save_matrices --out "results/${tag}.json" >> "$LOG" 2>&1 && log "done ${tag}" || { log "FAIL ${tag}"; continue; }
    else
      log "skip ${tag} (exists)"
    fi
    "$PY" experiments/analyze_g4.py --rome "$rome_npz" \
      --alpha "results/matrices/${tag}.npz" --known --edit_ok \
      --out "results/G4_L${L}_s${S}.json" >> "$LOG" 2>&1 && log "analyzed -> results/G4_L${L}_s${S}.json"
  done
done
log "=== G4 EXTEND COMPLETE — per-seed results/G4_L{8,10,12,14}_s{0,1,2}.json ==="
echo "G4_EXTEND_DONE" >> "$LOG"
