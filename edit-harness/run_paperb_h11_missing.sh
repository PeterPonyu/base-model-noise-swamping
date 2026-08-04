#!/usr/bin/env bash
# Paper B H11 missing 5 cells: gemma2b L19 s2, qwen3b L27 s2, phi35 L24 s{0,1,2}
# Frozen parameters from run_paperb_curve_local.sh; dual-card shardable for remote box.
set -u
cd "$(dirname "$0")" || exit 1

# Environment
PY=${PY:-/home/zeyufu/miniconda3/envs/dl/bin/python3}
GPU_ID=${GPU_ID:-0}
DRYRUN=${DRYRUN:-0}
BUDGET_MIN=${BUDGET_MIN:-300}
JOB_CAP_MIN=${JOB_CAP_MIN:-120}
SNAPSHOT_DEVICE=${SNAPSHOT_DEVICE:-cuda}
SHARD=${SHARD:-all}  # all | card0 | card1
WAVE_BOX=${WAVE_BOX:-local}
H=${H:-$(pwd)}

# Prereg check (reuses Paper B curve prereg)
PREREG=${PREREG:-../docs/plans/PREREG-PAPERB-CURVE-2026-07-26.md}
[ -f "$PREREG" ] && grep -qx 'STATUS: RATIFIED' "$PREREG" || {
  echo "ABORT: Paper B curve prereg not ratified" >&2
  exit 5
}

# Paths
mkdir -p engine results/quant_survival_curve
PIDFILE=engine/run_paperb_h11_missing_${SHARD}.pid
LOG=engine/run_paperb_h11_missing_${SHARD}.log
echo "$BASHPID" > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# Files
Q=experiments/quant_survival_phase1.py
DATA=data/counterfact.json
for f in "$Q" "$DATA"; do
  [ -f "$f" ] || { log "ABORT: missing $f"; exit 3; }
done

# Model specs: tag:path:layer
ALL_SPECS="gemma2b:data/models/gemma-2-2b:19 qwen3b:data/models/Qwen2.5-3B:27 phi35:data/models/Phi-3.5-mini:24"

# Shard distribution (card0 gets 3 cells, card1 gets 2 cells for balance)
case "$SHARD" in
  all)
    GRID="$ALL_SPECS"
    SEEDS_gemma2b="2"
    SEEDS_qwen3b="2"
    SEEDS_phi35="0 1 2"
    ;;
  card0)
    GRID="gemma2b:data/models/gemma-2-2b:19 phi35:data/models/Phi-3.5-mini:24"
    SEEDS_gemma2b="2"
    SEEDS_qwen3b=""
    SEEDS_phi35="0 2"
    ;;
  card1)
    GRID="qwen3b:data/models/Qwen2.5-3B:27 phi35:data/models/Phi-3.5-mini:24"
    SEEDS_gemma2b=""
    SEEDS_qwen3b="2"
    SEEDS_phi35="1"
    ;;
  *)
    log "ABORT: invalid SHARD=$SHARD (must be all|card0|card1)"
    exit 2
    ;;
esac

# Model existence check
for spec in $GRID; do
  rest=${spec#*:}
  model=${rest%:*}
  [ -d "$model" ] || { log "ABORT: missing $model"; exit 3; }
done

# CounterFact SHA256 check
EXPECTED_DATA_SHA="d017056125178a13728594e66a801357a8db9ed7973a7425554bb4271de9fc6f"
actual=$(sha256sum "$DATA" | cut -d' ' -f1)
[ "$actual" = "$EXPECTED_DATA_SHA" ] || {
  log "ABORT: CounterFact sha256 mismatch (expected $EXPECTED_DATA_SHA, got $actual)"
  exit 3
}

# Tokenizer gate: verify all three models load and have reasonable vocab sizes
log "Tokenizer gate check..."
"$PY" - <<'PY' || { log "ABORT: tokenizer gate failed"; exit 4; }
from transformers import AutoTokenizer
import sys

models = [
    ("gemma2b", "data/models/gemma-2-2b", 256000, 262000),
    ("qwen3b", "data/models/Qwen2.5-3B", 150000, 152100),
    ("phi35", "data/models/Phi-3.5-mini", 32000, 32100),
]

for tag, path, min_vocab, max_vocab in models:
    try:
        tok = AutoTokenizer.from_pretrained(path)
        vocab_size = len(tok)
        if not (min_vocab <= vocab_size <= max_vocab):
            print(f"FAIL {tag}: vocab_size={vocab_size} outside [{min_vocab}, {max_vocab}]", file=sys.stderr)
            sys.exit(1)
        print(f"PASS {tag}: vocab_size={vocab_size}")
    except Exception as e:
        print(f"FAIL {tag}: {e}", file=sys.stderr)
        sys.exit(1)
print("Tokenizer gate PASS")
PY

# Model integrity parameter check
log "Model integrity check..."
"$PY" - <<'PY' || { log "ABORT: model integrity failed"; exit 4; }
import sys
sys.path.insert(0, ".")
from experiments.tools.integrity_check import check_integrity

models = [
    ("gemma2b", "data/models/gemma-2-2b", "2.507e9"),
    ("qwen3b", "data/models/Qwen2.5-3B", "3.09e9"),
    ("phi35", "data/models/Phi-3.5-mini", "3.821e9"),
]

for tag, path, expected in models:
    print(f"Checking {tag} at {path}...")
    check_integrity(path, expect_params=expected)
    print(f"PASS {tag}")
print("Model integrity PASS")
PY

if [ "$DRYRUN" = 1 ]; then
  log "DRYRUN mode: would run SHARD=$SHARD GRID=$GRID"
  exit 0
fi

# CPU selftest
CUDA_VISIBLE_DEVICES="" "$PY" "$Q" --selftest > engine/paperb_h11_missing_selftest.log 2>&1 && \
  grep -q 'ALL CHECKS PASSED' engine/paperb_h11_missing_selftest.log || {
  log "ABORT: CPU selftest failed"
  exit 4
}
log "CPU selftest PASS"

# GPU idle gate (skip on remote box if WAVE_BOX != local)
if [ "$WAVE_BOX" = "local" ]; then
  consec=0
  gate_start=$(date +%s)
  while [ "$consec" -lt 3 ]; do
    line=$(nvidia-smi -i "$GPU_ID" --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    util=$(printf '%s' "$line" | cut -d, -f1 | tr -dc 0-9)
    mem=$(printf '%s' "$line" | cut -d, -f2 | tr -dc 0-9)
    if [ -n "$util" ] && [ -n "$mem" ] && [ "$util" -lt 25 ] && [ "$mem" -lt 1500 ]; then
      consec=$((consec + 1))
    else
      consec=0
    fi
    elapsed=$(($(date +%s) - gate_start))
    [ "$elapsed" -le 1800 ] || { log "ABORT: GPU idle gate timeout"; exit 8; }
    [ "$consec" -eq 3 ] || sleep 30
  done
  log "GPU idle gate PASS"
fi

# Validation function (runner_stamp + schema check)
validate() {
  "$PY" - "$1" "$2" <<'PY'
import json, sys, numpy as np
table_path, raw_path = sys.argv[1], sys.argv[2]
d = json.load(open(table_path))
a = np.load(raw_path, allow_pickle=True)
s = d.get('runner_stamp') or {}
need = {'code_sha256', 'pid', 'hostname', 'wall_start', 'wall_end', 'elapsed_s', 'nvidia_smi_sample'}
assert not (need - set(s)), f"Missing runner_stamp fields: {need - set(s)}"
assert json.loads(str(a['runner_stamp_json'].item()))['code_sha256'] == s['code_sha256'], "code_sha256 mismatch"
assert a['COS'].shape == (200, 200), f"COS shape={a['COS'].shape}, expected (200,200)"
assert d['editor'] == 'rome', f"editor={d['editor']}, expected rome"
assert d['codec'] == 'real', f"codec={d['codec']}, expected real"
PY
}

# Main execution loop
T0=$(date +%s)
failures=0

for spec in $GRID; do
  tag=${spec%%:*}
  rest=${spec#*:}
  model=${rest%:*}
  layer=${spec##*:}

  # Get seeds for this model/shard
  seeds_var="SEEDS_${tag}"
  seeds="${!seeds_var:-}"
  [ -n "$seeds" ] || continue

  for seed in $seeds; do
    dir="results/quant_survival_curve/${tag}_rome_L${layer}_s${seed}"
    table="$dir/QS_phase1_table.json"
    raw="$dir/QS_phase1_raw.npz"
    mkdir -p "$dir"

    # Skip if valid
    if [ -f "$table" ] && [ -f "$raw" ] && validate "$table" "$raw" >/dev/null 2>&1; then
      log "SKIP $tag L$layer s$seed (already valid)"
      continue
    fi

    # Budget check
    elapsed=$((($(date +%s) - T0) / 60))
    if [ $((elapsed + 60)) -gt "$BUDGET_MIN" ]; then
      log "BUDGET-STOP at ${elapsed}min (limit ${BUDGET_MIN}min)"
      break 2
    fi

    # Run cell
    log "START $tag L$layer s$seed"
    CUDA_VISIBLE_DEVICES="$GPU_ID" timeout --signal=TERM --kill-after=60 "$((JOB_CAP_MIN * 60))s" \
      "$PY" "$Q" \
        --run \
        --model "$model" \
        --data "$DATA" \
        --editor rome \
        --n_edits 200 \
        --n_probes 200 \
        --layer "$layer" \
        --seed "$seed" \
        --steps 20 \
        --lr 0.1 \
        --schemes nf4dq,int8 \
        --codec real \
        --fullmodel_cache auto \
        --n_perm 1000 \
        --n_boot 1000 \
        --device cuda \
        --snapshot_device "$SNAPSHOT_DEVICE" \
        --out_dir "$dir" \
        --table_out "$table" \
      >> "engine/paperb_h11_missing_${tag}_s${seed}.log" 2>&1
    rc=$?

    if [ "$rc" -eq 0 ] && validate "$table" "$raw" >/dev/null 2>&1; then
      log "DONE $tag L$layer s$seed"
    else
      log "FAIL $tag L$layer s$seed rc=$rc"
      failures=$((failures + 1))
    fi

    # Abort after 2 failures
    [ "$failures" -lt 2 ] || { log "ABORT: 2 failures"; exit 9; }
  done
done

# Summary readout (aggregates from both old and new cells)
log "Running aggregate readout..."
"$PY" experiments/paperb_curve_readout.py >> "$LOG" 2>&1
readout_rc=$?

if [ "$readout_rc" -eq 0 ]; then
  log "G-S3 PASS receipt written"
elif [ "$readout_rc" -eq 3 ]; then
  log "G-S3 INCOMPLETE (some cells still missing)"
else
  log "G-S3 not passed (rc=$readout_rc)"
fi

log "COMPLETE failures=$failures readout_rc=$readout_rc elapsed=$((($(date +%s) - T0) / 60))min"

[ "$failures" -eq 0 ] || exit 9
[ "$readout_rc" -eq 3 ] && exit 11
exit 0
