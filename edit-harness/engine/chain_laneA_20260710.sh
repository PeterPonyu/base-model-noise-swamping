#!/usr/bin/env bash
# Lane A chain 2026-07-10 — B6 revision-readiness pre-runs on the freed 5090
# (docs/plans/LOCAL-COMPUTE-PLAN-2026-07-10.md, sequencing step 1: A0 verify -> A1 -> A3 -> A2;
# A1/A2 are GPU rows inside run_laneA_seeds.sh, A3 needs CPU pooling only — also in that
# driver's post-run — and A0 is the CPU verify below).
# Launch: cd edit-harness && nohup ./engine/chain_laneA_20260710.sh >> engine/chain_laneA_20260710.nohup.log 2>&1 &
# Stop:   kill by PID from engine/chain_laneA_20260710.pid (NEVER pkill -f).
# Reminders (memory-backed): laptop lid OPEN while running (nvidia_uvm wedge);
# reboot first if the box has been under load (60W SW-thermal cap history).
set -u
cd /home/zeyufu/Desktop/idea-feasibility-analysis/edit-harness || exit 2
LOG=engine/chain_laneA_20260710.log
echo "$$" > engine/chain_laneA_20260710.pid
PY=/home/zeyufu/miniconda3/envs/dl/bin/python3
echo "[chain-laneA] START $(date '+%F %T %Z') pid=$$" | tee -a "$LOG"

# ---------------------------------------------------------------- A0: VERIFY gptj rows (CPU, never re-run GPU on a false alarm)
# Plan item A0: the 07-08 gptj preflight failure was believed fixed — confirm the C3/C4
# artifacts on disk actually parse and carry the expected aggregate keys before anyone
# schedules a gptj rerun. This is a VERIFY-only step: it logs a verdict, it never launches.
$PY - >> "$LOG" 2>&1 <<'EOF'
import json, glob, os
ok, bad = [], []
files = sorted(glob.glob('results/C3_gptj_rome_L*.json'))
expect_layers = {'14', '18', '21', '24'}
seen = set()
for f in files:
    try:
        d = json.load(open(f))
        assert 'aggregate' in d and 'per_seed' in d, 'missing aggregate/per_seed'
        L = f.split('_L')[1].split('_')[0]
        seen.add(L); ok.append(f)
    except Exception as e:
        bad.append((f, str(e)))
c4 = 'results/C4_causal_gptj_table.json'
c4ok = False
try:
    d = json.load(open(c4)); c4ok = bool(d.get('layers')); ok.append(c4)
except Exception as e:
    bad.append((c4, str(e)))
missing = expect_layers - seen
verdict = 'PASS' if (not bad and not missing and c4ok) else 'FAIL'
print(f"[A0-verify] {verdict}: C3 layers found={sorted(seen)} missing={sorted(missing)} "
      f"C4_table_ok={c4ok} bad={bad}")
print(f"[A0-verify] gptj rerun is {'NOT needed' if verdict=='PASS' else 'POSSIBLY needed — inspect before scheduling'}")
EOF
grep -h "A0-verify" "$LOG" | tail -2

run() {              # run <name> <BUDGET_MIN|-> <driver.sh>
  local name="$1" budget="$2" script="$3"
  local env=""; [ "$budget" != "-" ] && env="BUDGET_MIN=$budget"
  echo "[chain-laneA] >>> $name START $(date '+%T') ($env)" | tee -a "$LOG"
  env $env bash "$script" >> "engine/chain_laneA_${name}.log" 2>&1
  local rc=$?
  echo "[chain-laneA] <<< $name DONE rc=$rc $(date '+%T')" | tee -a "$LOG"
}

# A1 + A2 + A3-pooling, one idempotent driver (~8.4h at measured per-cell timings).
run lanea_seeds 560 ./run_laneA_seeds.sh

echo "[chain-laneA] ALL DONE $(date '+%F %T %Z')" | tee -a "$LOG"
echo "[chain-laneA] report: engine/run_lanea_seeds_report.txt ; A0 verdict above; new tables:" | tee -a "$LOG"
ls -la results/C4_causal_instruct_table_3seed.json results/C4_causal_8b_table_3seed.json \
      results/C4_causal_mquake_table_3seed_probesrc.json results/C3_mquake_alpha_L12_3seed.json \
      results/RIPPLE_depth_profile.json 2>&1 | tee -a "$LOG"
rm -f engine/chain_laneA_20260710.pid
