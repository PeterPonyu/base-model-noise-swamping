#!/usr/bin/env bash
# box_preflight_onbox.sh — run ON A FRESHLY RENTED BOX before any spend (I21).
# Verifies the box image assumptions the wave drivers silently make. FAIL = do not
# launch; WARN = proceed but note. Laptop-side prerequisites are a different script
# (box_preflight_local.sh); box_prepare_wave.sh check is per-wave and runs AFTER this.
#
# Usage:  bash engine/box_preflight_onbox.sh [wave]
#   wave optional; deletion-wave1 additionally requires 2 UNIFORM cards.
set -u
WAVE="${1:-generic}"
FAILED=0
fail(){ echo "FAIL $*"; FAILED=1; }
warn(){ echo "WARN $*"; }
ok(){ echo "OK   $*"; }

PY="${CLOUD_PY:-python3}"
DATA_DISK="${DATA_DISK:-/root/autodl-tmp}"

# 1. torch + CUDA (image assumption: preinstalled DL image)
if command -v "$PY" >/dev/null 2>&1 && "$PY" - <<'PY' 2>/dev/null
import torch, sys
assert torch.cuda.is_available(), "torch present but CUDA not available"
print(f"torch {torch.__version__} cuda {torch.version.cuda} devices {torch.cuda.device_count()}")
PY
then ok "torch+CUDA"; else fail "torch absent or CUDA-blind (wrong image?)"; fi

# 2. nvidia-smi + card count/uniformity
if command -v nvidia-smi >/dev/null 2>&1; then
  mapfile -t mems < <(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | tr -dc '0-9\n')
  ok "nvidia-smi sees ${#mems[@]} card(s): ${mems[*]} MiB"
  if [ "$WAVE" = deletion-wave1 ]; then
    [ "${#mems[@]}" -ge 2 ] || fail "deletion-wave1 needs 2 cards, found ${#mems[@]}"
    if [ "${#mems[@]}" -ge 2 ] && [ "${mems[0]}" != "${mems[1]}" ]; then
      fail "deletion-wave1 needs UNIFORM cards, found ${mems[0]} vs ${mems[1]} MiB"
    fi
  fi
else
  fail "nvidia-smi unavailable (no-card mode? switch instance to GPU)"
fi

# 3. DATA_DISK exists and has headroom
if [ -d "$DATA_DISK" ]; then
  avail=$(df --output=avail -BG "$DATA_DISK" 2>/dev/null | tail -1 | tr -dc 0-9)
  ok "DATA_DISK=$DATA_DISK avail=${avail:-?}GB"
  [ -n "$avail" ] && [ "$avail" -ge 20 ] || warn "DATA_DISK low (<20GB); check wave_spec need_gb before download"
else
  fail "DATA_DISK=$DATA_DISK missing (image assumption broken; set DATA_DISK= explicitly)"
fi

# 4. HF token — WARN only (public models download fine without it; gated ones don't)
if [ -n "${HF_TOKEN:-}" ] || [ -f "${HF_HOME:-$HOME/.cache/huggingface}/token" ]; then
  ok "HF token present"
else
  warn "no HF token — gated repos (Llama) will 403; public repos unaffected"
fi

# 5. outbound net to HF (cheap, 5s)
if curl -sI --max-time 5 https://huggingface.co >/dev/null 2>&1; then
  ok "huggingface.co reachable"
else
  warn "huggingface.co unreachable in 5s — downloads will stall (try /etc/network_turbo)"
fi

if [ "$FAILED" -eq 0 ]; then echo "PREFLIGHT_GREEN wave=$WAVE host=$(hostname)"; exit 0; fi
echo "PREFLIGHT_BLOCKED wave=$WAVE host=$(hostname) — fix every FAIL before spend" >&2
exit 3
