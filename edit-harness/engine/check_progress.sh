#!/bin/bash
# check_progress.sh — 快速检查所有任务进度
cd "$(dirname "$0")/.."

echo "======== Frame-A Recovery Progress ========"
echo ""
echo "Active Processes:"
for pf in engine/resume_mixb_missing.pid engine/run_mixc.pid engine/chain_after_bc_drain_20260726.pid engine/monitor_and_chain.pid; do
  if [ -f "$pf" ]; then
    pid=$(cat "$pf" 2>/dev/null)
    name=$(basename "$pf" .pid)
    if kill -0 "$pid" 2>/dev/null; then
      etime=$(ps -p "$pid" -o etime= 2>/dev/null | xargs)
      echo "  ✓ $name (pid $pid, runtime $etime)"
    else
      echo "  ✗ $name (pid $pid, DEAD)"
    fi
  fi
done

echo ""
echo "Cell Counts:"
mixb=$(ls results/frame_a/cells/cell_*_MIX_B_*.json 2>/dev/null | wc -l)
mixc=$(ls results/frame_a/cells/cell_*_MIX_C_*.json 2>/dev/null | wc -l)
mixa=$(ls results/frame_a/cells/cell_*_MIX_A_*.json 2>/dev/null | wc -l)
echo "  MIX_A: $mixa (expect 33 after chain S2)"
echo "  MIX_B: $mixb/33"
echo "  MIX_C: $mixc/33"

echo ""
echo "Missing MIX_B s2 cells:"
for p in ft_merge random; do
  f="results/frame_a/cells/cell_llama-3.2-1b_real_MIX_B_${p}_s2.json"
  if [ -f "$f" ]; then
    echo "  ✓ $p"
  else
    echo "  ✗ $p"
  fi
done

echo ""
echo "Chain Markers:"
for m in engine/STAMP_CUTOFF_UTC.txt engine/FRAME_A_GATE_V2_PASS.ok engine/CHAIN_BC_DRAIN_STOP.txt engine/CHAIN_ABORT.txt; do
  if [ -f "$m" ]; then
    echo "  ✓ $(basename $m): $(cat $m 2>/dev/null | head -1)"
  fi
done

echo ""
echo "GPU Status:"
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader 2>/dev/null || echo "  (nvidia-smi unavailable)"

echo ""
echo "Recent Log Tails:"
for log in engine/resume_mixb_missing.log engine/run_mixc.log engine/chain_after_bc_drain.log engine/monitor_and_chain.log; do
  if [ -f "$log" ]; then
    echo ""
    echo "--- $(basename $log) (last 3 lines) ---"
    tail -3 "$log"
  fi
done
