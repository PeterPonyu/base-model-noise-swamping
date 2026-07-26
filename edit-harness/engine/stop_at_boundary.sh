#!/bin/bash
# stop_at_boundary.sh — wait for the in-flight qwen3b_s1 cell to commit, then stop EVERYTHING
# (probe, aniso chain, run8h queue) in the order that prevents auto-restarts. Pre-reboot cleanup.
set -u
H=/home/zeyufu/Desktop/idea-feasibility-analysis/edit-harness
cd "$H" || exit 1
L=engine/stop_at_boundary.log
lg(){ echo "[$(date '+%F %T')] $*" >> "$L"; }
lg "watching for qwen3b_s1 boundary"
until grep -qE '(done|FAIL) gate_qwen3b_rome_cf_L18_s1' engine/run8h.log; do sleep 10; done
lg "boundary reached — tearing down (probe -> aniso chain -> queue)"
for pf in engine/power_probe.pid engine/chain_aniso.pid; do
  P=$(cat $pf 2>/dev/null); [ -n "${P:-}" ] && kill -0 $P 2>/dev/null && { kill -TERM -- -$P 2>/dev/null || kill -TERM $P 2>/dev/null; lg "stopped $(basename $pf) ($P)"; }
done
./engine/stop_deep_queue.sh engine/run8h.pid >> "$L" 2>&1
lg "ALL STOPPED — safe to reboot. GPU state: $(nvidia-smi --query-gpu=power.draw,utilization.gpu,memory.used --format=csv,noheader)"
