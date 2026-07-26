#!/bin/bash
# gpu_power_probe.sh — after tonight's chain completes: 25min cool-down, then a 60s max-power
# FP16 GEMM burn while logging power/clocks. Discriminates thermal-soak vs platform-cap.
set -u
H=/home/zeyufu/Desktop/idea-feasibility-analysis/edit-harness
cd "$H" || exit 1
echo $$ > engine/power_probe.pid
L=engine/power_probe.log
lg(){ echo "[$(date '+%F %T')] $*" >> "$L"; }
lg "probe armed: waiting for aniso-chain COMPLETE"
until grep -q 'aniso-chain: COMPLETE' engine/chain_aniso.log 2>/dev/null; do sleep 60; done
lg "chain complete — 25min cool-down"
sleep 1500
lg "cool-down done; pre-burn state:"
nvidia-smi --query-gpu=power.draw,clocks.sm,temperature.gpu --format=csv,noheader >> "$L"
( for i in $(seq 1 40); do nvidia-smi --query-gpu=power.draw,utilization.gpu,clocks.sm,temperature.gpu --format=csv,noheader >> "$L"; sleep 2; done ) &
SAMPLER=$!
timeout 90 /home/zeyufu/miniconda3/envs/dl/bin/python3 - >> "$L" 2>&1 <<'PYEOF'
import torch, time
a = torch.randn(8192, 8192, dtype=torch.float16, device='cuda')
b = torch.randn(8192, 8192, dtype=torch.float16, device='cuda')
t0 = time.time()
n = 0
while time.time() - t0 < 60:
    c = a @ b; n += 1
torch.cuda.synchronize()
print(f"GEMM burn: {n} matmuls in 60s ({n*2*8192**3/60/1e12:.1f} TFLOP/s fp16)")
PYEOF
wait $SAMPLER 2>/dev/null
peak=$(grep -oE '^[0-9]+\.[0-9]+ W' "$L" | sort -rn | head -1)
lg "VERDICT: peak draw during burn = ${peak:-unknown}. >120W => thermal-soak was the cap (remedy: cooling/pacing). ~60W even cool => platform/EC or driver cap (next: asusctl profile, -rgc, BIOS, reboot)."
