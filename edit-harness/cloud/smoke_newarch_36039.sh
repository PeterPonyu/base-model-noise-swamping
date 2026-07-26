#!/bin/bash
# cloud/smoke_newarch_36039.sh — house-rule smoke cells for the 36039 dual-4090D box
# (2026-07-13): every box-new architecture gets a ~2-min cell BEFORE any battery.
# Targets: Mistral-7B-v0.3, Qwen3-8B-Base, gemma-2-9b-bf16 (the 100GB-preset newcomers).
# Real steps=20 (steps=2 smoke gives a FALSE esr=0 — see smoke-QA 2026-07-02 lesson).
# --model_dtype bf16 is REQUIRED at 7-9B on 24G cards (killgate defaults to fp32).
# One model per card in parallel, then the third on card 0.
set -u
H="$(cd "$(dirname "$0")/.." && pwd)"; cd "$H" || exit 1
PY=${CLOUD_PY:-/root/miniconda3/bin/python}
DATA="$H/data/counterfact.json"
M=${MODELS_DIR:-/root/autodl-tmp/models}
OUT="$H/results/smoke36039"
mkdir -p "$OUT" cloud/logs
[ -f "$DATA" ] || { echo "ABORT: $DATA missing"; exit 2; }

run_smoke(){ # card src tag
  if [ ! -d "$2" ]; then echo "MODEL-ABSENT: $2 — skipped"; return 0; fi
  if [ -f "$OUT/smoke_$3.json" ]; then echo "[smoke] $3 already done — skipped"; return 0; fi
  echo "[smoke] card$1 $3 starting $(date '+%T')"
  CUDA_VISIBLE_DEVICES=$1 "$PY" experiments/killgate_keygeom.py \
    --model "$2" --data "$DATA" --dataset counterfact \
    --n_edits 24 --n_probes 60 --steps 20 --editor rome --layer auto \
    --seed 0 --model_dtype bf16 --out "$OUT/smoke_$3.json" \
    > "cloud/logs/smoke_$3.log" 2>&1
  echo "[smoke] card$1 $3 rc=$? $(date '+%T')"
}

run_smoke 0 "$M/Mistral-7B-v0.3"  mistral7b &
P0=$!
run_smoke 1 "$M/Qwen3-8B-Base"    qwen3_8b &
P1=$!
wait $P0 $P1
# review M1: the wave driver gates EVERY model on its smoke json — card1's models
# (known families, but fresh checkpoints on this box) need cells too.
run_smoke 0 "$M/Qwen2.5-7B"             qwen25_7b &
P0=$!
run_smoke 1 "$M/Llama-3.1-8B-Instruct"  llama31_8bi &
P1=$!
wait $P0 $P1
run_smoke 0 "$M/gemma-2-9b-bf16"  gemma9b_bf16

echo "=== smoke verdicts (gate: esr>0.9, frac_known sane) ==="
"$PY" - "$OUT" <<'EOF'
import json, glob, sys
ok = True
for f in sorted(glob.glob(sys.argv[1] + "/smoke_*.json")):
    d = json.load(open(f))
    esr = d.get("esr", d.get("edit_success_rate"))
    fk = d.get("frac_known")
    if esr is None:
        scalars = {k: v for k, v in d.items() if isinstance(v, (int, float))}
        print(f, "NO-ESR-KEY; scalar fields:", scalars)
        ok = False
        continue
    verdict = "PASS" if esr > 0.9 else "FAIL"
    if verdict == "FAIL":
        ok = False
    print(f, "esr=%.3f" % esr, "frac_known=", fk, "->", verdict)
print("SMOKE_ALL_PASS" if ok else "SMOKE_HAS_FAILURES")
EOF
