#!/usr/bin/env bash
# chain_lanes_20260710.sh — MASTER lane supervisor for LOCAL-COMPUTE-PLAN-2026-07-10.
# Sequences: [foreign GPU job drains] -> Lane A (B6 revision seed gap-fill; relaunching
# its chain if its 30-min idle-gate abort fired) -> [idle + thermal gates] -> Lane B
# (P3 GPU-Ollama under PREREG-B2B4-FROZEN-20260710.md).
#
# WHY stage 0 exists: at 2026-07-10 05:41 a job from ANOTHER workspace
# (~/Desktop/labs/active/HetCLOP run_biological.py, pid 3464654) took ~468MiB VRAM,
# pinning nvidia-smi mem above Lane A's 1500MiB idle threshold. Lane A's own gate
# hard-aborts after 30min of busy polls (exit 2). This supervisor waits for the foreign
# job BY PID (kill -0 — never pgrep/pkill a pattern), then relaunches Lane A if needed.
# We never touch the foreign job itself — not ours.
#
# Launch: cd edit-harness && FOREIGN_PIDS=3464654 nohup ./engine/chain_lanes_20260710.sh \
#           >> engine/chain_lanes_20260710.nohup.log 2>&1 &
# Stop:   kill by PID from engine/chain_lanes_20260710.pid (NEVER pkill -f — this command
#         line contains the very names a pattern would match).
#
# Gates before Lane B:
#   idle    — util<25 AND mem<1500MiB, 3 consecutive checks 60s apart (codified gate;
#             NEVER zero-compute-apps — persistent contexts, memory/gpu-idle-gate lesson).
#   thermal — 20s fp16 GEMM burn must reach >=120W; the A+B stint spans >12h and the 60W
#             SW-thermal cap (memory/gpu-60w-thermal-cap-reboot-fix) would wedge the
#             Ollama queue's per-job caps. On <120W: DEFER (exit 5 + note file), reboot,
#             relaunch this chain — it fast-passes everything already done.
#   prereg  — PREREG-B2B4-FROZEN-20260710.md must exist; queue must have pending jobs.
# Idempotent: safe to relaunch after any abort/defer at any stage.
set -u
H=/home/zeyufu/Desktop/idea-feasibility-analysis/edit-harness
P3=/home/zeyufu/Desktop/idea-feasibility-analysis/branches/p3_agent_ipi
cd "$H" || exit 2
LOG=engine/chain_lanes_20260710.log
PIDFILE=engine/chain_lanes_20260710.pid
PY=/home/zeyufu/miniconda3/envs/dl/bin/python3
FOREIGN_PIDS="${FOREIGN_PIDS:-}"
MAX_LANEA_LAUNCHES="${MAX_LANEA_LAUNCHES:-2}"
echo "$$" > "$PIDFILE"
log(){ echo "[chain-lanes $(date '+%F %T')] $*" | tee -a "$LOG"; }
finish(){ rm -f "$PIDFILE"; }
trap finish EXIT
log "START pid=$$ foreign_pids='${FOREIGN_PIDS}' max_laneA_launches=${MAX_LANEA_LAUNCHES}"

# ---------------------------------------------------------------- stage 0: foreign GPU job(s)
# Identity-checked wait: if the kernel recycles the PID to a different process, treat the
# original job as exited (liveness guard — we still never signal anything here).
for fp in $FOREIGN_PIDS; do
  if kill -0 "$fp" 2>/dev/null; then
    ident0="$(tr '\0' ' ' < "/proc/$fp/cmdline" 2>/dev/null)"
    log "stage0: waiting for foreign pid $fp ('${ident0}') to exit (kill -0 poll, 120s; NOT killing it)"
    while kill -0 "$fp" 2>/dev/null; do
      ident="$(tr '\0' ' ' < "/proc/$fp/cmdline" 2>/dev/null)"
      if [ "$ident" != "$ident0" ]; then log "stage0: pid $fp identity changed ('${ident}') -> original job exited (PID reuse)"; break; fi
      sleep 120
    done
    log "stage0: foreign pid $fp done"
  else
    log "stage0: foreign pid $fp already gone"
  fi
done

# ---------------------------------------------------------------- stage 1: Lane A drained (or run it)
LANEA_TARGETS="results/gate_llama1b_alpha_mquake_L12_s1.json results/gate_llama1b_alpha_mquake_L12_s2.json \
results/g4_instruct_alphaHO_cf_L12_s1.json results/g4_instruct_alphaHO_cf_L12_s2.json \
results/g4_llama8b_alphaHO_cf_L16_s1.json results/g4_llama8b_alphaHO_cf_L16_s2.json \
results/g4_llama8b_alphaHO_cf_L24_s1.json results/g4_llama8b_alphaHO_cf_L24_s2.json \
results/gate_llama8b_rome_cf_L16_s2.json results/g4_llama8b_alphaHO_cf_L28_s0.json"
missing_targets(){ local n=0 f; for f in $LANEA_TARGETS; do [ -f "$f" ] || n=$((n+1)); done; echo "$n"; }
wait_pidfile(){ # wait_pidfile <file> <label> — waits while the recorded PID is alive
  local f="$1" label="$2" p
  while :; do
    [ -f "$f" ] || { log "stage1: $label pidfile gone -> not running"; return 0; }
    p="$(cat "$f" 2>/dev/null)"
    { [ -n "$p" ] && kill -0 "$p" 2>/dev/null; } || { log "stage1: $label (pid ${p:-?}) not running"; return 0; }
    sleep 120
  done
}
lanea_launches=0; prev_missing=-1
while :; do
  wait_pidfile engine/chain_laneA_20260710.pid "laneA chain"
  wait_pidfile engine/run_lanea_seeds.pid      "laneA driver"
  m="$(missing_targets)"
  if [ "$m" -eq 0 ]; then log "stage1: Lane A complete (all 10 target cells on disk)"; break; fi
  if [ "$lanea_launches" -ge 1 ] && [ "$m" -eq "$prev_missing" ]; then
    log "stage1: ${m} cell(s) still missing and UNCHANGED by relaunch #${lanea_launches} —"
    log "        stable skip/fail set (e.g. bf16 equiv-gate CONFIG-skips). Not relaunching;"
    log "        proceeding to Lane B. Inspect engine/run_lanea_seeds_report.txt."
    break
  fi
  if [ "$lanea_launches" -ge "$MAX_LANEA_LAUNCHES" ]; then
    log "stage1: ${m} target cell(s) still missing after ${lanea_launches} (re)launches —"
    log "        NOT relaunching again (real failures need eyes, not retries)."
    log "        Proceeding to Lane B anyway: the lanes are independent; inspect"
    log "        engine/run_lanea_seeds_report.txt + engine/chain_laneA_lanea_seeds.log."
    break
  fi
  lanea_launches=$((lanea_launches+1)); prev_missing="$m"
  log "stage1: Lane A not running, ${m}/10 target cells missing -> (re)launch #${lanea_launches} of chain_laneA (idempotent)"
  nohup ./engine/chain_laneA_20260710.sh >> engine/chain_laneA_20260710.nohup.log 2>&1 &
  sleep 30   # let it write its pidfile before we wait on it
done
log "stage1 done. Lane A report tail:"
tail -3 engine/run_lanea_seeds_report.txt 2>/dev/null | tee -a "$LOG"

# ---------------------------------------------------------------- stage 2: GPU idle gate (util+mem, x3)
idle_ok=0; tries=0
while [ "$idle_ok" -lt 3 ]; do
  tries=$((tries+1))
  if [ "$tries" -gt 120 ]; then log "ABORT: GPU never idled within ~2h of Lane A draining"; exit 3; fi
  line="$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)"
  util="$(echo "$line" | awk -F, '{gsub(/[^0-9]/,"",$1);print $1}')"
  mem="$(echo "$line"  | awk -F, '{gsub(/[^0-9]/,"",$2);print $2}')"
  if [ -n "$util" ] && [ -n "$mem" ] && [ "$util" -lt 25 ] && [ "$mem" -lt 1500 ]; then
    idle_ok=$((idle_ok+1)); log "idle gate: PASS ${idle_ok}/3 (util=${util}% mem=${mem}MiB)"
  else
    idle_ok=0; log "idle gate: busy (util=${util:-?}% mem=${mem:-?}MiB), resetting"
  fi
  [ "$idle_ok" -lt 3 ] && sleep 60
done

# ---------------------------------------------------------------- stage 3: thermal burn gate (>=120W)
log "thermal gate: 25s GEMM burn (require >=120W peak; 60W cap => DEFER + reboot needed)"
# timeout wrap (review MAJOR): on a wedged GPU the CUDA calls block uninterruptibly and an
# unbounded wait would hang the supervisor in exactly the state this gate exists to detect.
timeout --signal=KILL 90 "$PY" - <<'EOF' >> "$LOG" 2>&1 &
import torch, time
a = torch.randn(8192, 8192, device='cuda', dtype=torch.float16)
b = torch.randn(8192, 8192, device='cuda', dtype=torch.float16)
t0 = time.time()
while time.time() - t0 < 25:
    a @ b
torch.cuda.synchronize()
EOF
BURN=$!
# sample after a settle so cold torch import / CUDA init doesn't under-read a healthy card
peak=0
sleep 6
for _ in 1 2 3 4; do
  w="$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits 2>/dev/null | head -1 | cut -d. -f1 | tr -dc 0-9)"
  [ -n "$w" ] && [ "$w" -gt "$peak" ] && peak="$w"
  sleep 5
done
wait "$BURN" 2>/dev/null; burn_rc=$?
log "thermal gate: peak ${peak}W during burn (burn rc=${burn_rc})"
if [ "$burn_rc" -ne 0 ]; then log "ABORT: burn script failed/timed out (rc=${burn_rc}) — GPU state suspect, not launching Lane B"; exit 5; fi
if [ "$peak" -lt 120 ]; then
  {
    echo "THERMAL-DEFER $(date '+%F %T'): GEMM burn peaked at ${peak}W (<120W) after the Lane A stint."
    echo "The 60W SW-thermal cap has likely re-appeared. REBOOT the box, then relaunch:"
    echo "  cd $H && nohup ./engine/chain_lanes_20260710.sh >> engine/chain_lanes_20260710.nohup.log 2>&1 &"
    echo "(Everything already done fast-passes; only the remaining stages run.)"
  } | tee -a "$LOG" > engine/LANEB_THERMAL_DEFER.txt
  exit 5
fi

# ---------------------------------------------------------------- stage 4: pre-reg + queue preconditions
[ -f "$P3/PREREG-B2B4-FROZEN-20260710.md" ] || { log "ABORT: pre-reg freeze file missing — Lane B is gated on it"; exit 4; }
pending="$("$PY" - <<'EOF'
import json
q = json.load(open("/home/zeyufu/Desktop/idea-feasibility-analysis/branches/p3_agent_ipi/jobs/queue.json"))
print(sum(1 for j in q if j.get("gpu_required") and j.get("status") != "done"))
EOF
)"
log "pre-reg OK; pending GPU jobs: ${pending:-?}"
if [ -z "$pending" ] || [ "$pending" -eq 0 ]; then log "nothing pending — Lane B already drained. DONE."; exit 0; fi

# ---------------------------------------------------------------- stage 5: launch Lane B, wait by PID
log "launching Lane B: run_p3_gpu.sh (BUDGET_MIN=420 JOB_CAP_MIN=100)"
cd "$P3" || { log "ABORT: cannot cd to p3 branch"; exit 2; }
BUDGET_MIN=420 JOB_CAP_MIN=100 nohup ./run_p3_gpu.sh >> logs/run_p3_gpu.nohup.log 2>&1 &
BP=$!
log "Lane B pid=$BP (also in $P3/logs/run_p3_gpu.pid)"
wait "$BP"; brc=$?
cd "$H" || exit 2
if [ "$brc" -ne 0 ]; then
  log "LANE B FAILED rc=${brc} (2=card/daemon, 3=preflight, 4=NOT-on-GPU CPU-fallback refusal)"
else
  log "Lane B finished rc=0."
fi
log "run_p3_gpu.log tail:"
tail -6 "$P3/logs/run_p3_gpu.log" 2>/dev/null | tee -a "$LOG"
if [ -f "$P3/results/P3_GPU_report.json" ]; then
  log "report: $P3/results/P3_GPU_report.json"
else
  log "WARNING: no P3_GPU_report.json — inspect $P3/logs/run_p3_gpu.log before trusting anything"
fi
log "ALL DONE"
