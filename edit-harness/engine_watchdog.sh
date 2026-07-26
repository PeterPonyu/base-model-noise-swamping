#!/usr/bin/env bash
# Cron watchdog: keep the fan-out engine advancing across crashes / reboots.
# Relaunches engine.py if it's not running, UNLESS the plan is DONE or the GPU
# is wedged (engine sets STOPPED_GPU_WEDGE — relaunching would just fail again).
# engine.py is idempotent (skips jobs whose output JSON exists), so relaunch is safe.
H=/home/zeyufu/Desktop/idea-feasibility-analysis/edit-harness
CONDA=/home/zeyufu/miniconda3/bin/conda
cd "$H" || exit 0

pgrep -f "python3 engine.py" >/dev/null && exit 0          # already running
state=$(cat engine/state.json 2>/dev/null)
echo "$state" | grep -q '"round": "DONE"' && exit 0        # finished
echo "$state" | grep -q "STOPPED_GPU_WEDGE" && exit 0      # don't relaunch into a wedge

nohup env -u ALL_PROXY -u all_proxy /home/zeyufu/miniconda3/envs/dl/bin/python3 engine.py \
  >> engine/engine_stdout.log 2>&1 &
echo "[$(date '+%F %T')] watchdog relaunched engine" >> engine/watchdog.log
