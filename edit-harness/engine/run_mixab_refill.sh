#!/bin/bash
# run_mixab_refill.sh — refill the 9 gate-v2-quarantined Frame-A cells, LOCAL 5090 (2026-07-31).
#   6x MIX_B s2 (always_ft/always_grace/always_rag/always_reject/ft_merge/random — moved to
#     .synthetic-relabel-bak/*.GATE-V2-FAIL-20260731 after failing runner_stamp_v2)
#   3x MIX_A s2 (cost_only/ft_merge/random — synthetic-relabel quarantine 2026-07-26)
# run_stream skip-on-exists => only the 9 gaps regenerate, now WITH runner stamps (gate v2
# schema, run_stream.py 2026-07-31). setsid cell + TERM trap per the 07-29 incident (I18).
set -u
H="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$H/.." || exit 2
PY=${PY:-/home/zeyufu/miniconda3/envs/dl/bin/python3}
PIDFILE=engine/run_mixab_refill.pid
LOG=engine/run_mixab_refill.log
mkdir -p engine
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE" 2>/dev/null)" 2>/dev/null; then
  echo "REFUSE: already running (pid $(cat "$PIDFILE"))" >&2; exit 7
fi
echo $$ > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT
log "======== MIX_A/B REFILL START pid=$$ (9 cells: 6 MIX_B + 3 MIX_A) ========"
ENVP="env -u ALL_PROXY -u all_proxy HF_HUB_OFFLINE=1"
CHILD_PIDFILE=engine/run_mixab_refill.child.pid
rc_all=0
run_part(){  # TAG MIX POLICIES
  local tag="$1" mix="$2" pol="$3"
  log "RUN $tag ($mix policies=$pol)"
  setsid $ENVP $PY -m experiments.frame_a.run_stream --run --real \
      --mixes "$mix" --policies "$pol" \
      --model_dir data/models/Llama-3.2-1B >> "$LOG" 2>&1 &
  local child=$!
  echo "$child" > "$CHILD_PIDFILE"
  trap 'log "WRAPPER TERM/INT — setsid cell pid '"$child"' stays alive; relaunch resumes"; exit 143' TERM INT
  wait "$child"; local rc=$?
  trap - TERM INT
  [ "$rc" -eq 0 ] && log "DONE $tag" || { log "FAIL $tag rc=$rc"; rc_all=$rc; }
}
run_part mixb_s2 MIX_B "always_ft,always_grace,always_rag,always_reject,ft_merge,random"
[ "$rc_all" -eq 0 ] && run_part mixa_s2 MIX_A "cost_only,ft_merge,random"
rm -f "$CHILD_PIDFILE"
# ---- verification: the 9 cells exist AND carry schema-valid runner stamps ----
$PY - <<'PY'
import json, sys
sys.path.insert(0, 'experiments/frame_a')
import provenance_gate_v2 as pg
want = [f"cell_llama-3.2-1b_real_MIX_B_{p}_s2.json" for p in
        ("always_ft","always_grace","always_rag","always_reject","ft_merge","random")] + \
       [f"cell_llama-3.2-1b_real_MIX_A_{p}_s2.json" for p in ("cost_only","ft_merge","random")]
bad = []
for w in want:
    try:
        cell = json.load(open(f"results/frame_a/cells/{w}"))
    except OSError:
        bad.append((w, "MISSING")); continue
    errs = pg._stamp_shape_errors(cell.get("runner_stamp"))
    if errs:
        bad.append((w, errs))
if bad:
    print("REFILL VERIFY FAIL:", bad); sys.exit(1)
print("REFILL VERIFY PASS: 9/9 cells present with gate-v2-valid runner stamps")
PY
vrc=$?
[ "$vrc" -eq 0 ] && log "VERIFIED: 9/9 refills stamped" || { log "VERIFY FAILED rc=$vrc"; rc_all=1; }
log "======== MIX_A/B REFILL END rc=$rc_all ========"
exit "$rc_all"
