#!/usr/bin/env bash
# Prepare Paper B H11 missing cells wave on a fresh AutoDL box.
# Usage: bash engine/box_paperb_h11_prepare.sh {deps|download|check}
set -u

ACTION="${1:-}"
H="${HARNESS:-/root/edit-harness}"
PY="${CLOUD_PY:-python3}"
DATA_DISK="${DATA_DISK:-/root/autodl-tmp}"
MODEL_ROOT="$H/data/models"
DATA="$H/data/counterfact.json"
EXPECTED_DATA_SHA="d017056125178a13728594e66a801357a8db9ed7973a7425554bb4271de9fc6f"
FAILED=0

usage() {
  echo "usage: $0 {deps|download|check}" >&2
  exit 2
}
[ -n "$ACTION" ] || usage

log() { echo "[paperb-h11-prepare] $*"; }
fail() { log "FAIL $*"; FAILED=1; }
need_file() { [ -f "$1" ] || fail "missing file $1"; }
need_dir() { [ -d "$1" ] || fail "missing directory $1"; }

# Models needed: gemma-2-2b, Qwen2.5-3B, Phi-3.5-mini
# Total ~14GB safetensors + ~2GB config/tokenizer = ~16GB disk
# VRAM: dual 4090D (24GB each) or better
MODEL_SPECS="
google/gemma-2-2b|gemma-2-2b|2614341888
Qwen/Qwen2.5-3B|Qwen2.5-3B|3085938688
microsoft/Phi-3.5-mini-instruct|Phi-3.5-mini|3821079552
"

phase_deps() {
  command -v "$PY" >/dev/null 2>&1 || { fail "$PY not found"; return; }
  "$PY" - <<'PY' || fail "CUDA-compatible torch is absent"
import torch
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
assert torch.cuda.is_available(), "CUDA must be available"
PY

  local missing
  missing=$("$PY" - <<'PY'
import importlib.util
mods = {
    "numpy": "numpy",
    "scipy": "scipy",
    "transformers": "transformers",
    "huggingface_hub": "huggingface_hub",
    "bitsandbytes": "bitsandbytes"
}
print(" ".join(pkg for pkg, mod in mods.items() if importlib.util.find_spec(mod) is None))
PY
)
  if [ -n "$missing" ]; then
    log "installing dependencies (missing: $missing)"
    "$PY" -m pip install -r "$H/requirements-box-waves.txt" || fail "dependency installation failed"
  fi

  "$PY" - <<'PY' || fail "runtime dependency import failed"
import numpy, scipy, transformers, huggingface_hub, bitsandbytes
print("dependency imports PASS")
PY

  [ "$FAILED" -eq 0 ] && log "DEPS READY"
}

phase_download() {
  need_dir "$DATA_DISK"
  need_dir "$H"
  [ "$FAILED" -eq 0 ] || return

  # Need at least 20GB free
  avail=$(df --output=avail -BG "$DATA_DISK" 2>/dev/null | tail -1 | tr -dc 0-9)
  [ -n "$avail" ] && [ "$avail" -ge 20 ] || {
    fail "$DATA_DISK has ${avail:-?}GB free; wave needs at least 20GB"
    return
  }

  mkdir -p "$DATA_DISK/models" "$H/data"

  # Link model root to data disk
  if [ -d "$MODEL_ROOT" ] && [ ! -L "$MODEL_ROOT" ]; then
    if [ -z "$(find "$MODEL_ROOT" -mindepth 1 -maxdepth 1 2>/dev/null)" ]; then
      rmdir "$MODEL_ROOT"
    else
      fail "$MODEL_ROOT is a non-empty real directory; move it to $DATA_DISK/models"
      return
    fi
  fi
  ln -sfn "$DATA_DISK/models" "$MODEL_ROOT"

  export HF_HOME="$DATA_DISK/hf_cache"
  unset HF_ENDPOINT ALL_PROXY all_proxy
  [ -f /etc/network_turbo ] && source /etc/network_turbo

  # Download models
  while IFS='|' read -r repo name expected; do
    [ -n "$repo" ] || continue
    target="$MODEL_ROOT/$name"

    if [ -d "$target" ] && "$PY" "$H/experiments/tools/integrity_check.py" "$target" --expect_params "$expected" >/dev/null 2>&1; then
      log "skip $repo: integrity check passes at $target"
      continue
    fi

    if [ -d "$target" ]; then
      log "$target exists but integrity incomplete; snapshot_download will resume"
    fi

    log "downloading $repo -> $target"
    REPO_ID="$repo" TARGET_DIR="$target" "$PY" - <<'PY' || { fail "download failed: $repo"; break; }
import os
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id=os.environ["REPO_ID"],
    local_dir=os.environ["TARGET_DIR"],
    ignore_patterns=["*.bin", "*.pth", "*.h5", "*.msgpack"],
)
PY
  done <<< "$MODEL_SPECS"

  # Verify CounterFact
  if [ -f "$DATA" ]; then
    actual=$(sha256sum "$DATA" | cut -d' ' -f1)
    if [ "$actual" = "$EXPECTED_DATA_SHA" ]; then
      log "CounterFact sha256 verified"
    else
      fail "CounterFact sha256 mismatch"
    fi
  else
    fail "CounterFact data missing: $DATA"
  fi

  [ "$FAILED" -eq 0 ] && log "DOWNLOAD READY"
}

check_gpu() {
  command -v nvidia-smi >/dev/null 2>&1 || { fail "nvidia-smi unavailable"; return; }

  # Need 2 GPUs with >=23GB VRAM each
  mapfile -t mems < <(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | tr -dc '0-9\n')
  [ "${#mems[@]}" -ge 2 ] || fail "need 2 GPU(s), found ${#mems[@]}"

  for i in 0 1; do
    [ "${mems[$i]}" -ge 23000 ] || fail "GPU $i has ${mems[$i]}MiB, need >=23000MiB"
  done

  "$PY" - <<'PY' || fail "torch does not see CUDA"
import torch
assert torch.cuda.is_available()
count = torch.cuda.device_count()
assert count >= 2, f"need 2 CUDA devices, found {count}"
print(f"torch CUDA devices: {count}")
PY
}

check_prereg() {
  local prereg="$H/docs/plans/PREREG-PAPERB-CURVE-2026-07-26.md"
  need_file "$prereg"
  grep -qx 'STATUS: RATIFIED' "$prereg" || fail "prereg not ratified: $prereg"
}

check_code() {
  need_file "$H/run_paperb_h11_missing.sh"
  need_file "$H/experiments/quant_survival_phase1.py"
  need_file "$H/experiments/paperb_curve_readout.py"
  need_file "$H/experiments/tools/integrity_check.py"

  # Verify driver is executable
  [ -x "$H/run_paperb_h11_missing.sh" ] || chmod +x "$H/run_paperb_h11_missing.sh"
}

check_models() {
  while IFS='|' read -r repo name expected; do
    [ -n "$repo" ] || continue
    target="$MODEL_ROOT/$name"
    log "checking $name..."
    "$PY" "$H/experiments/tools/integrity_check.py" "$target" --expect_params "$expected" || fail "$name integrity check failed"
  done <<< "$MODEL_SPECS"
}

phase_check() {
  check_gpu
  check_prereg
  check_code
  check_models

  if [ "$FAILED" -eq 0 ]; then
    driver="$H/run_paperb_h11_missing.sh"
    driver_sha=$(sha256sum "$driver" | cut -d' ' -f1)
    ready="$H/engine/BOX_READY_paperb_h11_missing.ok"
    {
      echo "wave=paperb-h11-missing"
      echo "host=$(hostname)"
      echo "timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      echo "driver_sha256=$driver_sha"
      echo "models=gemma-2-2b,Qwen2.5-3B,Phi-3.5-mini"
    } > "$ready"
    log "CHECK COMPLETE → $ready"
  else
    log "CHECK FAILED (see errors above)"
    return 1
  fi
}

case "$ACTION" in
  deps) phase_deps ;;
  download) phase_download ;;
  check) phase_check ;;
  *) usage ;;
esac

[ "$FAILED" -eq 0 ] || exit 1
exit 0
