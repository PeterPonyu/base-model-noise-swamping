#!/usr/bin/env bash
# chain_after_bc_drain_20260726.sh — post-drain continuation chain (2026-07-26).
#
# WAITS for the live Frame-A MIX_B/C wave to fully drain, then in strict order:
#   S1  apply the runner-stamp patch to run_stream.py (staging-reviewed,
#       APPROVE-WITH-FIXES applied; dry-run gated — aborts on failure)
#   S2  rerun the 3 quarantined MIX_A cells (cost_only/ft_merge/random, seed 2)
#       WITH stamps, via targeted --policies + --cf_cell_seed calls
#   S3  provenance gate v2 over the full cells dir — REQUIRE exit 0 (PASS);
#       exit 2 (INCOMPLETE) or FAIL aborts the chain and leaves a flag file
#   S4  B6 insurance queue (run_b6ins.sh: alphaHO L10/L14 x 3 seeds, ~150 GPU-min)
#
# Wave-drain detection = the wrapper PID from engine/run_frame_a_bc_real.pid is DEAD
# AND the 66 target cell JSONs exist (33 MIX_B + 33 MIX_C), OR the wrapper is dead
# with fewer cells (aborted wave) — in that case we STOP with a flag file instead of
# proceeding (S2/S3 on a partial grid would mislabel INCOMPLETE as contamination).
#
# Launch:  cd edit-harness && nohup ./engine/chain_after_bc_drain_20260726.sh \
#            > engine/chain_after_bc_drain.nohup.log 2>&1 &
# Stop:    kill by PID from engine/chain_after_bc_drain.pid (NEVER pgrep/pkill).
#          NOTE: S4 runs run_b6ins.sh in the FOREGROUND, so the chain pidfile stays held
#          for that queue's ~5h. Killing the chain PID does NOT stop the queue — stop that
#          one separately by the PID in engine/run_b6ins.pid.
# Idempotent: every stage checks its own completion marker / output first.

set -u
cd "$(dirname "$0")/.."   # edit-harness/
mkdir -p engine

PY=${PY:-/home/zeyufu/miniconda3/envs/dl/bin/python3}
# HIGH-4: mirror the wrapper's environment hardening (SOCKS proxy breaks HF loading).
ENVP="env -u ALL_PROXY -u all_proxy HF_HUB_OFFLINE=1"
CELLS="results/frame_a/cells"
LOG="engine/chain_after_bc_drain.log"
PIDFILE="engine/chain_after_bc_drain.pid"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
flag() { echo "$*" > "engine/CHAIN_BC_DRAIN_STOP.txt"; log "STOP-FLAG: $*"; }

# MEDIUM: refuse to start if a previous instance is alive (double-launch → GPU collision).
if [ -f "$PIDFILE" ]; then
  old=$(cat "$PIDFILE" 2>/dev/null || true)
  if [ -n "$old" ] && kill -0 "$old" 2>/dev/null; then
    log "REFUSE: chain already running as pid $old"; exit 3
  fi
  log "stale pidfile (pid ${old:-?} dead) — taking over"
fi
# MEDIUM: an earlier abort must be acknowledged before a relaunch re-enters stage 0.
if [ -f engine/CHAIN_BC_DRAIN_STOP.txt ] && [ "${IGNORE_STOP_FLAG:-0}" != "1" ]; then
  log "REFUSE: engine/CHAIN_BC_DRAIN_STOP.txt exists ($(cat engine/CHAIN_BC_DRAIN_STOP.txt))."
  log "        Investigate, delete the flag, or relaunch with IGNORE_STOP_FLAG=1."
  exit 3
fi
echo $$ > "$PIDFILE"
# MEDIUM: never leave a stale pidfile behind on any exit path.
trap 'rm -f "$PIDFILE"' EXIT

# HIGH-2: the gate's default cutoff (2026-07-27T00:00Z) would grandfather cells this chain
# itself writes today, making the stamp requirement vacuous. We pin it instead — but NOT to
# chain start: the live wave keeps writing ~47 more cells with the UNPATCHED runner while we
# wait, and those legitimately cannot carry stamps. The only correct boundary is the moment
# the patch takes effect (end of S1), captured into STAMP_CUTOFF_UTC there. Everything the
# patched runner writes (the 3 MIX_A reruns) must then be stamped or S3 FAILs.
STAMP_CUTOFF_UTC=""     # set at S1

log "======== CHAIN-AFTER-BC-DRAIN START pid=$$ ========"

# ---------------------------------------------------------------- stage 0: wait for drain
while true; do
  wp=$(cat engine/run_frame_a_bc_real.pid 2>/dev/null || true)
  alive=0
  [ -n "$wp" ] && kill -0 "$wp" 2>/dev/null && alive=1
  # LOW: match cell_* only — the namespaced p2 file also contains MIX_C and would inflate.
  b=$(ls $CELLS/cell_*_MIX_B_*.json 2>/dev/null | wc -l)
  c=$(ls $CELLS/cell_*_MIX_C_*.json 2>/dev/null | wc -l)
  if [ "$alive" -eq 0 ]; then
    if [ "$b" -ge 33 ] && [ "$c" -ge 33 ]; then
      log "wave drained clean: MIX_B $b/33 MIX_C $c/33"
      break
    else
      flag "wave wrapper dead but grid incomplete (MIX_B $b/33 MIX_C $c/33) — investigate before resuming; chain will NOT proceed"
      exit 5
    fi
  fi
  log "waiting: wrapper $wp alive, MIX_B $b/33 MIX_C $c/33"
  sleep 600
done

# extra settle: GPU idle gate (util<25 && mem<1500 x3, 30s apart).
# MEDIUM: bounded — a failing nvidia-smi must not hang the chain forever.
consec=0; settle_tries=0
while [ "$consec" -lt 3 ]; do
  if [ "$settle_tries" -ge 60 ]; then     # 60 x 30s = 30 min
    flag "GPU settle gate timed out after 30min (nvidia-smi unreadable or card never idle)"; exit 9
  fi
  settle_tries=$((settle_tries+1))
  line=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  util=$(echo "$line" | awk -F, '{gsub(/[^0-9]/,"",$1); print $1}')
  mem=$(echo "$line"  | awk -F, '{gsub(/[^0-9]/,"",$2); print $2}')
  if [ -n "${util:-}" ] && [ -n "${mem:-}" ] && [ "$util" -lt 25 ] && [ "$mem" -lt 1500 ]; then
    consec=$((consec+1)); else consec=0; fi
  log "gpu settle util=${util:-NA} mem=${mem:-NA} consec=${consec}/3 try=${settle_tries}/60"
  [ "$consec" -lt 3 ] && sleep 30
done

# ---------------------------------------------------------------- stage 1: apply runner-stamp patch
if grep -q "runner_stamp" experiments/frame_a/run_stream.py 2>/dev/null; then
  log "S1 SKIP: run_stream.py already stamped"
  # Restart case: recover the boundary written by the original S1 run.
  if [ -f engine/STAMP_CUTOFF_UTC.txt ]; then
    STAMP_CUTOFF_UTC=$(cat engine/STAMP_CUTOFF_UTC.txt)
    log "S1 cutoff recovered: $STAMP_CUTOFF_UTC"
  else
    flag "run_stream.py is patched but engine/STAMP_CUTOFF_UTC.txt is missing — cannot establish an honest stamp boundary; set it manually to the patch time and relaunch"
    exit 6
  fi
else
  log "S1: applying runner-stamp patch"
  # LOW: --forward --batch so an already-applied hunk fails closed instead of prompting.
  if ! ( cd experiments/frame_a && patch --dry-run --forward --batch -p0 \
         < patches/runner_stamp_20260726.patch ) >> "$LOG" 2>&1; then
    flag "S1 patch dry-run FAILED — run_stream.py diverged; manual review needed"; exit 6
  fi
  if ! ( cd experiments/frame_a && patch --forward --batch -p0 \
         < patches/runner_stamp_20260726.patch ) >> "$LOG" 2>&1; then
    flag "S1 patch apply FAILED after a clean dry-run — inspect run_stream.py"; exit 6
  fi
  $PY -c "import ast,sys; ast.parse(open('experiments/frame_a/run_stream.py').read())" \
    || { flag "S1 post-patch parse FAILED"; exit 6; }
  # CPU synthetic selftest (fast structural check; NOT a substitute for the GPU smoke below)
  $PY -m experiments.frame_a.run_stream --selftest >> "$LOG" 2>&1 \
    || { flag "S1 post-patch selftest FAILED"; exit 6; }
  # (Stage S1c REMOVED 2026-07-26: it used to apply the AlphaEdit sham-projector patch.
  # That control was measured to be ILL-POSED — rank-matching is a no-op projection (keeps
  # 97.8% of key energy vs the honest projector's 0.99%), while energy-matching degenerates
  # to the honest projector itself (subspace overlap 1.000) because the key spectrum is
  # rank-200 and top-heavy. No projector substitution can adjudicate the tautology
  # objection. See submissions/ieee/revision/PROJECTOR-CONTROL-ILLPOSED-20260726.md.
  # The patch file is retained but must NOT be applied.)

  # The patch is now live: every cell written from here on MUST carry a stamp.
  STAMP_CUTOFF_UTC=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
  echo "$STAMP_CUTOFF_UTC" > engine/STAMP_CUTOFF_UTC.txt
  log "S1 DONE: patch applied + synthetic selftest green; stamp cutoff = $STAMP_CUTOFF_UTC"
fi

# ---------------------------------------------------------------- stage 1b: RE-SMOKE the patched runner
# HIGH-1: _frame_a_code_checksum() hashes every frame_a/**/*.py, so patching run_stream.py
# invalidates engine/SMOKE_PASS.ok. The real-wave dispatch does NOT enforce the marker, so
# without this stage S2 would run freshly-patched GPU code with zero smoke coverage — exactly
# what the smoke gate exists to prevent. Mirrors run_frame_a_bc_real_20260721.sh's smoke call.
log "S1b: GPU smoke on the patched runner"
if ! $ENVP $PY -m experiments.frame_a.run_stream --run --real --smoke \
     --model_dir data/models/Llama-3.2-1B >> "$LOG" 2>&1; then
  flag "S1b post-patch GPU smoke FAILED — do NOT run science cells on this runner"; exit 6
fi
log "S1b DONE: patched runner smoke green"

# ---------------------------------------------------------------- stage 2: rerun 3 MIX_A cells
# Quarantined 2026-07-26: MIX_A cost_only_s2 / ft_merge_s2 / random_s2 (synthetic batch).
# run_stream skips existing JSONs, so only the missing 3 will run (~3 GPU-h max).
need=0
for p in cost_only ft_merge random; do
  [ -e "$CELLS/cell_llama-3.2-1b_real_MIX_A_${p}_s2.json" ] || need=1
done
if [ "$need" -eq 0 ]; then
  log "S2 SKIP: all 3 MIX_A rerun cells present"
else
  log "S2: rerunning quarantined MIX_A cells (policies cost_only,ft_merge,random; seed grid includes s2)"
  $ENVP $PY -m experiments.frame_a.run_stream --run --real \
    --mixes MIX_A --policies cost_only,ft_merge,random \
    --model_dir data/models/Llama-3.2-1B >> "$LOG" 2>&1 \
    || { flag "S2 MIX_A rerun FAILED"; exit 7; }
  # HIGH-2: assert the cells EXIST *and* carry a runner_stamp — file presence alone would
  # pass even if the patch silently emitted nothing.
  for p in cost_only ft_merge random; do
    f="$CELLS/cell_llama-3.2-1b_real_MIX_A_${p}_s2.json"
    [ -e "$f" ] || { flag "S2 rerun finished but cell ${p}_s2 missing"; exit 7; }
    $PY - "$f" <<'PYEOF' || { flag "S2 cell missing runner_stamp — patch did not take effect"; exit 7; }
import json, sys
d = json.loads(open(sys.argv[1]).read().replace("NaN", "null"))
st = d.get("runner_stamp")
assert isinstance(st, dict) and st.get("code_sha256") and st.get("stamp_version"), \
    f"no valid runner_stamp in {sys.argv[1]}"
PYEOF
  done
  log "S2 DONE: 3 MIX_A cells rerun and stamp-verified"
fi

# ---------------------------------------------------------------- stage 3: gate v2 must PASS
if [ -z "$STAMP_CUTOFF_UTC" ]; then
  flag "S3 refused: STAMP_CUTOFF_UTC empty (S1 did not establish a patch boundary)"; exit 8
fi
# Reviewer LOW: the one remaining way to get a vacuous boundary is a human editing
# STAMP_CUTOFF_UTC.txt to a late value. Refuse any cutoff that postdates the cells the
# patched runner just wrote — those must fall INSIDE the stamp-required window.
newest_rerun=$(ls -t "$CELLS"/cell_llama-3.2-1b_real_MIX_A_{cost_only,ft_merge,random}_s2.json 2>/dev/null | head -1)
if [ -n "$newest_rerun" ]; then
  cutoff_epoch=$(date -u -d "$STAMP_CUTOFF_UTC" +%s 2>/dev/null || echo 0)
  cell_epoch=$(stat -c %Y "$newest_rerun" 2>/dev/null || echo 0)
  if [ "$cutoff_epoch" -eq 0 ] || [ "$cutoff_epoch" -gt "$cell_epoch" ]; then
    flag "S3 refused: stamp cutoff $STAMP_CUTOFF_UTC postdates (or fails to parse against) the newest rerun cell $(basename "$newest_rerun") — the stamp requirement would be vacuous"
    exit 8
  fi
fi
log "S3: provenance gate v2 over $CELLS (stamp cutoff = patch time $STAMP_CUTOFF_UTC)"
$PY experiments/frame_a/provenance_gate_v2.py --cells_dir "$CELLS" \
  --runner-stamp-cutoff "$STAMP_CUTOFF_UTC" \
  --report engine/gate_v2_report_20260726.json >> "$LOG" 2>&1
rc=$?
if [ "$rc" -eq 0 ]; then
  log "S3 DONE: gate v2 PASS (report engine/gate_v2_report_20260726.json)"
  echo "PASS $(date '+%F %T')" > engine/FRAME_A_GATE_V2_PASS.ok
else
  flag "S3 gate v2 rc=$rc (0=PASS required) — DO NOT run Frame-A analysis; see engine/gate_v2_report_20260726.json"
  exit 8
fi

# ---------------------------------------------------------------- stage 4: B6 insurance queue
# HIGH-3: do NOT skip on the alphaHO marker — run_b6ins.sh is row-idempotent and its CPU
# sham tail runs AFTER the alphaHO rows; an early skip would silently drop the sham control.
log "S4: launching run_b6ins.sh (row-idempotent: alphaHO L10/L14 + CPU sham tail)"
if [ -f engine/run_b6ins.pid ]; then
  bp=$(cat engine/run_b6ins.pid 2>/dev/null || true)
  if [ -n "$bp" ] && kill -0 "$bp" 2>/dev/null; then
    flag "S4 SKIPPED: run_b6ins.sh already running as pid $bp"; exit 10
  fi
fi
./run_b6ins.sh >> engine/run_b6ins.nohup.log 2>&1
b6rc=$?     # capture BEFORE any other command consumes $?
log "S4 DONE rc=$b6rc (see engine/run_b6ins.log)"

log "======== CHAIN-AFTER-BC-DRAIN END ========"
