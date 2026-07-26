#!/bin/bash
# E2 card-window chain for box 29246 (Pro-6000 96GB), 2026-07-16.
# Runbook: engine/runbook_e2_20260716.md. Fast->slow; NeoX-20B runs LAST and SOLO
# (tightest VRAM fit; OOM there must not cost earlier cells). Kill only by PID from
# engine/chain_e2_20260716.pid. Idempotent: driver-level skip-if-valid everywhere.
# All ROME cells: run_merging_width.sh (n_layer fallback in place).
# Editor cells: run_merging_editors.sh (carries its own ΔW-fidelity smoke gate).
set -u
cd "$(dirname "$0")/.." || exit 2
PIDFILE=engine/chain_e2_20260716.pid
LOG=engine/chain_e2_20260716.log
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }

if [ -f "$PIDFILE" ]; then
  oldpid=$(cat "$PIDFILE" 2>/dev/null)
  if [ -n "$oldpid" ] && kill -0 "$oldpid" 2>/dev/null; then
    echo "REFUSE: chain already running (pid $oldpid)" >&2; exit 7
  fi
fi
echo $$ > "$PIDFILE"
log "================ E2 CHAIN START pid=$$ ================"

M=${M:-/root/autodl-tmp/models}
# box python (ops lesson 07-15: drivers default PY to the LOCAL laptop path; export BOTH
# conventions for on-box runs)
PY=${PY:-/root/miniconda3/bin/python}
export PY
CLOUD_PY=${CLOUD_PY:-$PY}
export CLOUD_PY

# 25s GEMM thermal-burn gate (house pattern: SW-thermal caps must abort before science)
burn_w=$(timeout 90 "$PY" - <<'PY' 2>/dev/null
import torch, time
if not torch.cuda.is_available():
    print(0); raise SystemExit
a = torch.randn(8192, 8192, device="cuda"); b = torch.randn(8192, 8192, device="cuda")
t0 = time.time()
while time.time() - t0 < 25:
    a @ b
torch.cuda.synchronize()
import subprocess
out = subprocess.run(["nvidia-smi","--query-gpu=power.draw","--format=csv,noheader,nounits"],
                     capture_output=True, text=True).stdout.strip().splitlines()[0]
print(int(float(out)))
PY
)
log "thermal burn gate: ${burn_w:-NA} W"
if [ -z "${burn_w:-}" ] || [ "${burn_w:-0}" -lt 120 ]; then
  log "THERMAL-DEFER: burn power ${burn_w:-NA} W < 120 W — reboot box, relaunch this chain"
  echo "THERMAL-DEFER" > engine/E2_THERMAL_DEFER.txt
  exit 5
fi

rome_cell(){ # dir tag [budget]
  local d="$1" t="$2" bud="${3:-180}"
  log "RUN rome $t"
  MODEL_DIR="$d" MODEL_TAG="$t" RG_GROUP_SIZES=2,3,5,10,20 BUDGET_MIN="$bud" ./run_merging_width.sh >> "$LOG" 2>&1
  local rc=$?; log "DONE rome $t rc=$rc"; return $rc
}
editor_cell(){ # dir tag editor [budget]
  local d="$1" t="$2" e="$3" bud="${4:-150}"
  log "RUN $e $t"
  MODEL_DIR="$d" MODEL_TAG="$t" EDITOR="$e" DATASET=cf RG_GROUP_SIZES=2,3,5,10,20 BUDGET_MIN="$bud" ./run_merging_editors.sh >> "$LOG" 2>&1
  local rc=$?; log "DONE $e $t rc=$rc"; return $rc
}

fails=0
# 1) gemma-2-9b ROME RG @75% (42L -> L31 auto): gemma family scale point (~1-1.5h)
rome_cell "$M/gemma-2-9b" gemma9b 150            || fails=$((fails+1))
# 2) Llama-3.1-8B editor-generality at 8B (memit, alpha) (~1h each)
editor_cell "$M/Llama-3.1-8B" llama8b memit 150  || fails=$((fails+1))
editor_cell "$M/Llama-3.1-8B" llama8b alpha 150  || fails=$((fails+1))
# 3) C-CELL: Mistral-Nemo-12B ROME RG @75% (40L -> L30 auto) — the scale/family anchor (~2-2.5h)
rome_cell "$M/Mistral-Nemo-Base-2407" nemo12b 240 || fails=$((fails+1))
# 4) OPTIONAL editor extreme: Qwen2.5-14B alpha (low-gain regime, 14B) (~2h)
[ "${SKIP_QWEN14B_ALPHA:-0}" = "1" ] || editor_cell "$M/Qwen2.5-14B" qwen14b alpha 210 || fails=$((fails+1))
# 5) LAST + SOLO: gpt-neox-20b ROME RG @75% (44L -> L33 auto) — tightest fit, OOM-guarded (~2-4h)
[ "${SKIP_NEOX:-0}" = "1" ] || rome_cell "$M/gpt-neox-20b" neox20b 300 || fails=$((fails+1))

log "================ E2 CHAIN END fails=$fails ================"
touch "engine/chain_e2_20260716.done"
exit "$fails"
