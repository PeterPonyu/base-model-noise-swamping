#!/bin/bash
# ESR layer-sweep probe for gpt2-xl (tracer recommendation 2026-07-16): discriminates
# H5 (layer-placement — auto75 sits past GPT-2's causal band) from H2 (value-opt
# budget). ROME only via the AUDITED shared path (merging_m0._compute_solo — the same
# function every RG cell used), n=50, seed 0, layers L14/L17/L20/L24/L30/L36.
# ~5-10 GPU-min total on the 5090. Fire ONLY at a wave-drain window (idle gate below).
#
# Expected under H5: esr peaks shallow (~L14-L20) and falls toward L36.
# Expected under H2: esr low everywhere -> instead test a steps/lr bump.
#
# Output: results/esr_probe_gpt2xl/esr_by_layer.json
set -u
H="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$H" || exit 1
PY=${PY:-/home/zeyufu/miniconda3/envs/dl/bin/python3}  # pin the dl env like every other driver (hostile review 2026-07-18: bare python3 works today but drifts)
MODEL_DIR=${MODEL_DIR:-data/models/gpt2-xl}
mkdir -p results/esr_probe_gpt2xl
[ -d "$MODEL_DIR" ] || { echo "ABORT: MODEL_DIR $MODEL_DIR missing"; exit 1; }

# idle gate: THIS card only (nvidia-smi ignores CUDA_VISIBLE_DEVICES; use -i). 30-min busy-abort
# (matches run_quant_smoke.sh's gate); no sleep after consec==3 (the window is open, do not stall).
GPU_ID=${CUDA_VISIBLE_DEVICES:-0}; GPU_ID=${GPU_ID%%,*}
gate_t0=$(date +%s); consec=0
while [ $consec -lt 3 ]; do
  read -r util mem <<<"$(nvidia-smi -i "$GPU_ID" --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | tr -d ',')"
  if [ "${util:-100}" -lt 25 ] && [ "${mem:-99999}" -lt 1500 ]; then
    consec=$((consec+1))
  else
    consec=0
    if [ $(( $(date +%s) - gate_t0 )) -gt 1800 ]; then echo "ABORT: GPU busy >30min at gate"; exit 2; fi
  fi
  echo "gpu poll util=$util mem=$mem consec=$consec/3"
  [ $consec -lt 3 ] && sleep 20
done

MODEL_DIR="$MODEL_DIR" timeout 3600 $PY - <<'EOF'
import json, os, sys, time
sys.path.insert(0, ".")
sys.path.insert(0, "experiments")
from merging_m0 import _load_edit_model, _compute_solo, load_counterfact

model_dir = os.environ["MODEL_DIR"]
# layer_arg only seeds the returned default; _compute_solo takes the layer explicitly.
model, tok, _, nL = _load_edit_model(model_dir, "14", "cuda")
print(f"[probe] loaded {model_dir} ({nL} layers)", flush=True)
edits = load_counterfact("data/counterfact.json", 50, seed=0)
out = {}
for L in (14, 17, 20, 24, 30, 36):
    t0 = time.time()
    vectors, W, W_base = _compute_solo(model, tok, L, "cuda", edits, 20, 0.1)
    esr = float(vectors["argmax_ok_solo"].mean())
    out[str(L)] = esr
    print(f"RESULT layer={L} esr={esr:.3f}  ({time.time()-t0:.0f}s)", flush=True)
json.dump({"model": model_dir, "n_edits": 50, "seed": 0, "steps": 20, "lr": 0.1,
           "esr_by_layer": out},
          open("results/esr_probe_gpt2xl/esr_by_layer.json", "w"), indent=1)
print("ESR PROBE DONE -> results/esr_probe_gpt2xl/esr_by_layer.json", flush=True)
EOF
