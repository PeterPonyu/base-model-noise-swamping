#!/bin/bash
# run_p3_gpu.sh -- P3 agentic-IPI GPU-Ollama orchestration (2026-07-10, Lane B).
#
# POLICY CONTEXT: B6 is submitted, so the Ollama CPU-pin is LIFTED. Ollama now serves from
# the local RTX 5090. This driver therefore INVERTS the old safety net: the 2026-07-06 sweep
# had to prove Ollama was NOT on the GPU; this one must prove it IS -- and abort loudly if it
# silently falls back to CPU mid-run.
#
# It: (0) CPU preflight, (1) checks the card is free enough to host models, (2) starts
# `ollama serve` FRESH with NO CPU-pin env (no CUDA_VISIBLE_DEVICES / OLLAMA_NUM_GPU /
# OLLAMA_VULKAN masking), PID-file tracked, (3) does a real first inference and VERIFIES
# GPU residency (an ollama/llama-server compute-app holding >=1.5 GiB VRAM) or aborts,
# (4) runs jobs/queue.json serially with per-job timeout + idempotent skip + wedge-abort
# after 2 consecutive timeout-like failures, (5) an inverted background watchdog asserts
# Ollama stays alive (by PID) AND on the GPU while a job runs, (6) post-run runs the BINDING
# audit_unmatched.py gate on every ipi_* result + prints defense gate verdicts + a report.
#
# Process rules (workspace standing): wait by PID (kill -0), NEVER pgrep/pkill a pattern.
# Only ever kills the ollama serve THIS script started, by its recorded PID.
#
# Usage:
#   cd branches/p3_agent_ipi
#   # populate the queue first, e.g.:
#   python make_jobs.py --kind grid    --tier core --seeds 0,1,2 --n 30
#   python make_jobs.py --kind defense --defenses spotlight,whitelist --tier core --seeds 0 --n 30
#   nohup ./run_p3_gpu.sh >> logs/run_p3_gpu.nohup.log 2>&1 &
#
# Env knobs: BUDGET_MIN (default 240), JOB_CAP_MIN (per-job cap, default 100),
#   DRYRUN=1 (print plan, no daemon/GPU),
#   REUSE_OLLAMA=1 (reuse an already-up daemon after verifying GPU residency),
#   PY (python interpreter), MIN_FREE_MIB (default 8000 VRAM headroom to start).
set -u

H="/home/zeyufu/Desktop/idea-feasibility-analysis/branches/p3_agent_ipi"
cd "$H" || exit 1
PY="${PY:-/home/zeyufu/miniconda3/envs/dl/bin/python3}"
command -v "$PY" >/dev/null 2>&1 || PY="python3"
OLLAMA_BIN="${OLLAMA_BIN:-/home/zeyufu/.local/bin/ollama}"
OLLAMA_URL="http://localhost:11434"
BUDGET_MIN="${BUDGET_MIN:-240}"
# Per-job hard cap. The scope brief's own estimates: core-tier grid sweep ~30-45 min/job,
# defense job = 2 whole arms ~60-90 min/job -- a 30-min cap would timeout every job and
# wedge-abort the queue. 100 min covers the defense worst case with headroom.
JOB_CAP_MIN="${JOB_CAP_MIN:-100}"
MIN_FREE_MIB="${MIN_FREE_MIB:-8000}"
# GPU-residency proof threshold. A true CPU fallback shows NO ollama/llama-server
# compute-app at all; any such app holding real VRAM proves GPU serving. 1500 was
# miscalibrated: the smallest probe (qwen2.5:0.5b) resides at ~1176 MiB and 1.5b-q4
# models sit near ~1.1GB — both false-aborted as "CPU fallback" (seen live 14:15:58
# 2026-07-10). 600 MiB is well above any non-model ollama footprint and below every
# model in the panel.
RESIDENCY_MIN_MIB="${RESIDENCY_MIN_MIB:-600}"
DRYRUN="${DRYRUN:-0}"
REUSE_OLLAMA="${REUSE_OLLAMA:-0}"

mkdir -p logs results jobs
LOG="logs/run_p3_gpu.log"
OLLAMA_LOG="logs/ollama_serve_p3gpu.log"
PIDFILE="logs/run_p3_gpu.pid"
OLLAMA_PIDFILE="logs/ollama_p3gpu.pid"
WD_PIDFILE="logs/watchdog_p3gpu.pid"
ABORT_FLAG="logs/p3_abort.flag"
RUNNING_FLAG="logs/p3_job_running.flag"
DONE_FLAG="logs/p3_done.flag"
echo $$ > "$PIDFILE"
rm -f "$ABORT_FLAG" "$RUNNING_FLAG" "$DONE_FLAG"

log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
log "================ RUN_P3_GPU START (pid $$, budget ${BUDGET_MIN}m, DRYRUN=${DRYRUN}) ================"

STARTED_OLLAMA=0
cleanup(){
  : > "$DONE_FLAG"
  if [ -f "$WD_PIDFILE" ]; then
    local wp; wp="$(cat "$WD_PIDFILE" 2>/dev/null)"
    [ -n "$wp" ] && kill -0 "$wp" 2>/dev/null && kill "$wp" 2>/dev/null && log "stopped watchdog pid=$wp"
  fi
  if [ "$STARTED_OLLAMA" -eq 1 ] && [ -f "$OLLAMA_PIDFILE" ]; then
    local op; op="$(cat "$OLLAMA_PIDFILE" 2>/dev/null)"
    if [ -n "$op" ] && kill -0 "$op" 2>/dev/null; then
      kill "$op" 2>/dev/null && log "stopped ollama serve pid=$op (started by this run)"
    fi
  else
    log "left ollama serve running (not started by this run, or no pidfile)"
  fi
  rm -f "$RUNNING_FLAG"
}
trap cleanup EXIT

# ---------------------------------------------------------------- Phase 0: CPU preflight
pf_fail=0
pf(){ if eval "$2"; then log "preflight OK: $1"; else log "PREFLIGHT-FAIL: $1"; pf_fail=1; fi; }
pf "python compiles run_ipi"   "$PY -m py_compile run_ipi.py"
pf "python compiles grid"      "$PY -m py_compile grid.py"
pf "python compiles defenses"  "$PY -m py_compile defenses.py"
pf "python compiles defense_analyze" "$PY -m py_compile defense_analyze.py"
pf "python compiles run_grid"  "$PY -m py_compile run_grid.py"
pf "python compiles run_defense" "$PY -m py_compile run_defense.py"
pf "python compiles audit"     "$PY -m py_compile audit_unmatched.py"
pf "ollama binary present"     "[ -x '$OLLAMA_BIN' ]"
pf "queue.json present"        "[ -f jobs/queue.json ]"
pf "nvidia-smi present"        "command -v nvidia-smi >/dev/null 2>&1"
if [ "$pf_fail" -ne 0 ]; then log "ABORT: preflight failed"; exit 3; fi

# pending GPU jobs (run_id \t out \t space-joined args-after-python), status != done, out missing/invalid
list_pending(){
  "$PY" - <<'EOF'
import json, os
q = json.load(open("jobs/queue.json")) if os.path.isfile("jobs/queue.json") else []
for j in q:
    if not j.get("gpu_required"): continue
    if j.get("status") == "done": continue
    cmd = j.get("cmd", [])
    if not cmd: continue
    args = cmd[1:]  # drop the leading "python"
    print("\t".join([j["run_id"], j.get("out",""), " ".join(str(a) for a in args)]))
EOF
}

mapfile -t PENDING < <(list_pending)
log "queue: ${#PENDING[@]} pending GPU job(s)"
if [ "${#PENDING[@]}" -eq 0 ]; then
  log "nothing to do -- populate jobs/queue.json with make_jobs.py (grid/defense). Exiting 0."
  exit 0
fi

if [ "$DRYRUN" -eq 1 ]; then
  log "DRYRUN=1 -- planned jobs (no daemon, no GPU):"
  for row in "${PENDING[@]}"; do
    rid="${row%%$'\t'*}"; rest="${row#*$'\t'}"; out="${rest%%$'\t'*}"; args="${rest#*$'\t'}"
    log "DRYRUN job ${rid} -> ${out}  cmd: $PY ${args}"
    echo "DRYRUN job ${rid}: $PY ${args}"
  done
  exit 0
fi

# ---------------------------------------------------------------- Phase 1: card free enough?
smi_line(){ nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1; }
line="$(smi_line)"
mem_used="$(echo "$line" | awk -F, '{gsub(/[^0-9]/,"",$2);print $2}')"
mem_tot="$(echo "$line"  | awk -F, '{gsub(/[^0-9]/,"",$3);print $3}')"
free_mib=$(( ${mem_tot:-0} - ${mem_used:-0} ))
log "GPU free ~${free_mib} MiB (used ${mem_used:-NA}/${mem_tot:-NA})"
# Warn (do not hard-block) if another compute-app already holds the card -- the card is serial;
# a concurrent B6 Lane-A job should finish first. Abort only if we cannot fit a model at all.
others="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | grep -viE 'ollama|llama-server|llama-cpp' || true)"
[ -n "$others" ] && log "NOTE: non-ollama GPU compute-app(s) present (card is serial):" && echo "$others" | while read -r l; do log "   $l"; done
if [ "$free_mib" -lt "$MIN_FREE_MIB" ]; then
  log "ABORT: only ${free_mib} MiB free (< MIN_FREE_MIB=${MIN_FREE_MIB}); free the card before P3 GPU run."
  exit 2
fi

# ---------------------------------------------------------------- Phase 2: start ollama on GPU
ollama_up(){ curl -s -m 4 "$OLLAMA_URL/api/version" >/dev/null 2>&1; }
if ollama_up; then
  if [ "$REUSE_OLLAMA" -eq 1 ]; then
    log "ollama already up; REUSE_OLLAMA=1 -> will verify GPU residency and reuse it"
    STARTED_OLLAMA=0
    # try to record its pid for the watchdog (best-effort, by listening socket owner)
    echo "" > "$OLLAMA_PIDFILE"
  else
    log "ABORT: an ollama daemon is already up but REUSE_OLLAMA!=1. Its GPU/env state is"
    log "       unverifiable from here. Stop it (by PID) and rerun, or set REUSE_OLLAMA=1."
    exit 2
  fi
else
  log "starting ollama serve FRESH on GPU (no CPU-pin env)"
  # Explicitly clear any inherited CPU-pin masks so the daemon uses the GPU (2026-07-10 policy).
  # OLLAMA_KEEP_ALIVE keeps a model resident so the residency watchdog is stable between calls.
  env -u CUDA_VISIBLE_DEVICES -u OLLAMA_NUM_GPU -u OLLAMA_VULKAN -u GGML_VK_VISIBLE_DEVICES \
      -u ALL_PROXY -u all_proxy \
      OLLAMA_HOME="/home/zeyufu/.ollama" OLLAMA_FLASH_ATTENTION=1 OLLAMA_KEEP_ALIVE=30m \
      nohup "$OLLAMA_BIN" serve >> "$OLLAMA_LOG" 2>&1 &
  OP=$!
  echo "$OP" > "$OLLAMA_PIDFILE"
  STARTED_OLLAMA=1
  log "ollama serve launched pid=$OP -> $OLLAMA_LOG"
  # wait for the HTTP endpoint (up to ~40s)
  up=0
  for _ in $(seq 1 40); do
    if ollama_up; then up=1; break; fi
    kill -0 "$OP" 2>/dev/null || { log "ABORT: ollama serve died during startup (see $OLLAMA_LOG)"; exit 2; }
    sleep 1
  done
  [ "$up" -eq 1 ] || { log "ABORT: ollama endpoint did not come up in ~40s"; exit 2; }
  log "ollama endpoint is up"
fi

# ---------------------------------------------------------------- Phase 3: verify GPU residency
# Pick the smallest model present as the probe (first found among a small-first list).
probe_model=""
for m in "qwen2.5:0.5b" "deepseek-r1:1.5b" "qwen2.5:1.5b" "llama3.2:1b" "gemma3:1b"; do
  if "$OLLAMA_BIN" show "$m" >/dev/null 2>&1; then probe_model="$m"; break; fi
done
[ -n "$probe_model" ] || { log "ABORT: no small probe model found via 'ollama show'"; exit 2; }
log "GPU-residency probe: real inference on '$probe_model'"
curl -s -m 120 "$OLLAMA_URL/api/generate" \
  -d "{\"model\":\"$probe_model\",\"prompt\":\"hi\",\"stream\":false,\"options\":{\"num_predict\":8}}" \
  > logs/p3_probe.json 2>>"$LOG"
# after a real inference the model must be resident: an ollama/llama-server compute-app holding VRAM
resident_mib(){
  nvidia-smi --query-compute-apps=process_name,used_memory --format=csv,noheader 2>/dev/null \
    | grep -iE 'ollama|llama-server|llama-cpp' \
    | grep -oE '[0-9]+ MiB' | grep -oE '[0-9]+' | sort -rn | head -1
}
res="$(resident_mib)"
log "post-probe ollama VRAM residency: ${res:-0} MiB"
if [ -z "$res" ] || [ "$res" -lt "$RESIDENCY_MIN_MIB" ]; then
  log "ABORT: Ollama is NOT on the GPU after a real inference (residency ${res:-0} MiB < ${RESIDENCY_MIN_MIB})."
  log "       This is the exact CPU-fallback failure mode; refusing to run the sweep on CPU."
  exit 4
fi
log "VERIFIED: Ollama is serving from the GPU (${res} MiB resident)"

# ---------------------------------------------------------------- Phase 4: inverted watchdog
# Background: while a job runs, assert ollama alive (by PID) AND on GPU. On silent CPU
# fallback or a dead daemon, write ABORT_FLAG so the main loop stops the queue. Two
# consecutive off-GPU polls (while a job runs) are required to avoid a keep-alive race.
OP_FOR_WD="$(cat "$OLLAMA_PIDFILE" 2>/dev/null)"
(
  miss=0
  while true; do
    sleep 45
    [ -f "$DONE_FLAG" ] && exit 0
    if [ -n "$OP_FOR_WD" ] && ! kill -0 "$OP_FOR_WD" 2>/dev/null; then
      echo "ollama serve pid=$OP_FOR_WD is dead" > "$ABORT_FLAG"; exit 1
    fi
    if [ -f "$RUNNING_FLAG" ]; then
      r="$(nvidia-smi --query-compute-apps=process_name,used_memory --format=csv,noheader 2>/dev/null \
            | grep -iE 'ollama|llama-server|llama-cpp' | grep -oE '[0-9]+ MiB' | grep -oE '[0-9]+' | sort -rn | head -1)"
      if [ -z "$r" ] || [ "$r" -lt "$RESIDENCY_MIN_MIB" ]; then
        miss=$((miss+1))
        [ "$miss" -ge 2 ] && { echo "ollama fell off the GPU during a job (residency ${r:-0} MiB)" > "$ABORT_FLAG"; exit 1; }
      else
        miss=0
      fi
    else
      miss=0
    fi
  done
) &
echo $! > "$WD_PIDFILE"
log "inverted watchdog started pid=$(cat "$WD_PIDFILE") (asserts Ollama stays on GPU during jobs)"

# ---------------------------------------------------------------- Phase 5: serial job loop
T0=$(date +%s)
elapsed_min(){ echo $(( ( $(date +%s) - T0 ) / 60 )); }
n_done=0; n_fail=0; n_skip=0; wedge=0; MAXWEDGE=2; QUEUE_ABORT=0

# validate a run_ipi/defense result JSON (parse + expected keys); mode ipi|defense
validate(){
  "$PY" - "$1" "$2" <<'EOF'
import json, sys
path, mode = sys.argv[1], sys.argv[2]
try:
    d = json.load(open(path))
except Exception as e:
    print(f"VALIDATE-FAIL unparseable: {e}"); sys.exit(1)
if mode == "defense":
    need = {"defense", "table", "permutation", "gate"}
else:
    need = {"run_id", "success_matrix", "per_model_asr", "models"}
missing = need - set(d.keys())
if missing:
    print(f"VALIDATE-FAIL missing keys {missing}"); sys.exit(1)
print("VALIDATE-OK")
EOF
}

run_job(){
  local rid="$1" out="$2" args="$3"
  local now; now=$(elapsed_min)
  if [ -f "$ABORT_FLAG" ]; then log "ABORT (watchdog): $(cat "$ABORT_FLAG")"; QUEUE_ABORT=1; return; fi
  if [ $(( now + 5 )) -gt "$BUDGET_MIN" ]; then log "BUDGET-STOP before ${rid} (elapsed ${now}m/${BUDGET_MIN}m)"; QUEUE_ABORT=1; return; fi
  local mode="ipi"; [[ "$rid" == defense_* ]] && mode="defense"
  # idempotent skip
  if [ -n "$out" ] && [ -f "$out" ]; then
    if validate "$out" "$mode" | grep -q VALIDATE-FAIL; then
      mv "$out" "$out.INVALID" 2>/dev/null; log "STALE-INVALID ${rid} -- quarantined; re-running"
    else
      log "skip ${rid} (exists, validated)"; n_skip=$((n_skip+1)); return
    fi
  fi
  local cap=$(( JOB_CAP_MIN * 60 ))   # per-job hard cap (JOB_CAP_MIN, default 100m; defense jobs are 2 arms)
  log "RUN ${rid} [${mode}] (elapsed ${now}m, cap ${cap}s) -> logs/job_${rid}.log"
  : > "$RUNNING_FLAG"
  local t rc dt
  t=$(date +%s)
  timeout --signal=TERM --kill-after=30 "${cap}s" env -u ALL_PROXY -u all_proxy \
      "$PY" $args >> "logs/job_${rid}.log" 2>&1
  rc=$?; dt=$(( $(date +%s) - t ))
  rm -f "$RUNNING_FLAG"
  if [ -f "$ABORT_FLAG" ]; then log "ABORT (watchdog) after ${rid}: $(cat "$ABORT_FLAG")"; n_fail=$((n_fail+1)); QUEUE_ABORT=1; return; fi
  if [ "$rc" -eq 0 ] && [ -n "$out" ] && [ -f "$out" ]; then
    local v; v="$(validate "$out" "$mode")"
    if echo "$v" | grep -q VALIDATE-FAIL; then
      mv "$out" "$out.INVALID" 2>/dev/null; log "FAIL ${rid} (${dt}s) OUTPUT-INVALID: ${v}"; n_fail=$((n_fail+1))
    else
      log "done ${rid} (${dt}s) ${v}"; n_done=$((n_done+1)); wedge=0
      # mark done in the queue so a rerun skips it
      "$PY" - "$rid" <<'EOF'
import json, sys, os
rid = sys.argv[1]; qp = "jobs/queue.json"
q = json.load(open(qp)) if os.path.isfile(qp) else []
for j in q:
    if j.get("run_id") == rid: j["status"] = "done"
json.dump(q, open(qp, "w"), indent=2)
EOF
    fi
  else
    if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
      wedge=$((wedge+1)); n_fail=$((n_fail+1))
      log "FAIL ${rid} (rc ${rc}, ${dt}s) TIMEOUT/WEDGE consec=${wedge}/${MAXWEDGE}"
      [ "$wedge" -ge "$MAXWEDGE" ] && { log "ABORT: ${MAXWEDGE} consecutive timeout-like failures"; QUEUE_ABORT=1; }
    else
      n_fail=$((n_fail+1)); log "FAIL ${rid} (rc ${rc}, ${dt}s) -- not counted toward wedge abort"
    fi
  fi
  log "PROGRESS ${n_done}done/${n_fail}fail/${n_skip}skip elapsed=$(elapsed_min)m"
}

for row in "${PENDING[@]}"; do
  [ "$QUEUE_ABORT" -eq 0 ] || break
  rid="${row%%$'\t'*}"; rest="${row#*$'\t'}"; out="${rest%%$'\t'*}"; args="${rest#*$'\t'}"
  run_job "$rid" "$out" "$args"
done

# ---------------------------------------------------------------- Phase 6: post-run (CPU, ALWAYS)
: > "$DONE_FLAG"
log "---------------- POST-RUN (CPU) ----------------"
# BINDING condition: audit every ipi_* result produced/paired this round for parser FNs.
"$PY" - >> "$LOG" 2>&1 <<'EOF'
import glob, json, os, subprocess, sys
audited = []
for f in sorted(glob.glob("results/ipi_*.json")):
    base = os.path.basename(f)
    if base.endswith("_cleaned.json"): continue
    aud = os.path.join("results", "audit_" + base[len("ipi_"):])
    try:
        r = subprocess.run([sys.executable, "audit_unmatched.py", f, "--out", aud],
                           capture_output=True, text=True, timeout=120)
        rep = json.load(open(aud)) if os.path.isfile(aud) else {}
        audited.append({"result": base,
                        "fn_rate": rep.get("estimated_false_negative_rate"),
                        "suspected_fn": rep.get("suspected_false_negatives")})
    except Exception as e:
        audited.append({"result": base, "audit_error": str(e)})
print("[post] audit_unmatched over", len(audited), "ipi_* results")
for a in audited:
    print("   ", a)
EOF

# defense gate verdicts + report
"$PY" - > results/P3_GPU_report.json 2>>"$LOG" <<'EOF'
import glob, json, os
report = {"defense_gates": [], "grid_contrasts": []}
for f in sorted(glob.glob("results/defense_*.json")):
    try:
        d = json.load(open(f))
        report["defense_gates"].append({
            "file": os.path.basename(f), "defense": d.get("defense"), "tier": d.get("tier"),
            "observed_mean_delta": d.get("permutation", {}).get("observed_mean_delta"),
            "p_value": d.get("permutation", {}).get("p_value"),
            "gate_passed": d.get("gate", {}).get("passed")})
    except Exception as e:
        report["defense_gates"].append({"file": os.path.basename(f), "error": str(e)})
for f in sorted(glob.glob("results/ipi_grid_*.json")):
    try:
        d = json.load(open(f)); c = d.get("contrast") or {}
        report["grid_contrasts"].append({
            "file": os.path.basename(f), "n_models": len(d.get("models", [])),
            "observed_diff": c.get("observed_diff"), "p_value": c.get("p_value"),
            "lineage_gt_arch": c.get("lineage_gt_architecture"),
            "contrast_note": d.get("contrast_note")})
    except Exception as e:
        report["grid_contrasts"].append({"file": os.path.basename(f), "error": str(e)})
print(json.dumps(report, indent=2, default=str))
EOF
log "post: wrote results/P3_GPU_report.json"

{
  echo "RUN_P3_GPU REPORT $(date '+%F %T')  jobs: ${n_done} done / ${n_fail} fail / ${n_skip} skipped  elapsed $(elapsed_min)m/${BUDGET_MIN}m  abort=${QUEUE_ABORT}"
  [ -f "$ABORT_FLAG" ] && echo "  ABORT reason: $(cat "$ABORT_FLAG")"
} | tee -a "$LOG"
log "================ RUN_P3_GPU END ================"
exit 0
