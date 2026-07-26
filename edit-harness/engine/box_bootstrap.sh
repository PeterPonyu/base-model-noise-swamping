#!/usr/bin/env bash
# box_bootstrap.sh — bring a FRESH AutoDL box to launch-ready. Run ON THE BOX.
#
# Encodes every ops lesson this workspace paid for (07-08 download campaign, 07-13/14
# two-box wave). Idempotent: safe to re-run. FAILS LOUDLY on unmet preconditions rather
# than continuing into an expensive mistake.
#
# Usage (on box):
#   bash engine/box_bootstrap.sh              # full check + setup
#   bash engine/box_bootstrap.sh --check      # verify only, change nothing
#   NEED_GB=60 bash engine/box_bootstrap.sh   # require 60G free on the DATA disk
#
# What it does NOT do: download models (that needs your explicit repo list and, for gated
# repos, a token you scp yourself), and never launches science.

set -u
CHECK_ONLY=0
[ "${1:-}" = "--check" ] && CHECK_ONLY=1
NEED_GB="${NEED_GB:-50}"
FAILED=0

say()  { echo "[boot] $*"; }
ok()   { echo "[boot]  OK   $*"; }
bad()  { echo "[boot] FAIL  $*"; FAILED=1; }
warn() { echo "[boot] WARN  $*"; }

say "=========== BOX BOOTSTRAP $(date '+%F %T') ==========="

# ---------------------------------------------------------------- 1. GPU
# nvidia-smi IGNORES CUDA_VISIBLE_DEVICES — always query by -i index (burned lesson).
if ! command -v nvidia-smi >/dev/null 2>&1; then
  bad "nvidia-smi not found — is this a no-card (无卡) boot? Downloads are fine here, compute is NOT."
else
  ncards=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
  if [ "$ncards" -eq 0 ]; then
    warn "no CUDA devices visible — no-card mode. OK for downloads ONLY (0.5 core: too weak for smokes)."
  else
    ok "$ncards GPU(s):"
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader | sed 's/^/[boot]       /'
    # foreign-process check: never kill blind, just report (may belong to another session)
    foreign=$(nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader 2>/dev/null)
    if [ -n "$foreign" ]; then
      warn "processes already resident on the card(s) — identify before arming any idle gate:"
      echo "$foreign" | sed 's/^/[boot]       /'
    else
      ok "no foreign compute processes on the card(s)"
    fi
  fi
fi

# ---------------------------------------------------------------- 2. disks
# System disk / is ~30G and WILL overflow; models must live on the DATA disk.
DATA_DIR="/root/autodl-tmp"
if [ -d "$DATA_DIR" ]; then
  avail=$(df -BG --output=avail "$DATA_DIR" 2>/dev/null | tail -1 | tr -dc '0-9')
  if [ -n "$avail" ] && [ "$avail" -ge "$NEED_GB" ]; then
    ok "data disk $DATA_DIR has ${avail}G free (need ${NEED_GB}G)"
  else
    bad "data disk $DATA_DIR has only ${avail:-?}G free, need ${NEED_GB}G — downloads will fail mid-way"
  fi
  sysavail=$(df -BG --output=avail / 2>/dev/null | tail -1 | tr -dc '0-9')
  [ -n "$sysavail" ] && [ "$sysavail" -lt 5 ] && warn "system disk / has only ${sysavail}G free"
else
  bad "$DATA_DIR missing — this does not look like an AutoDL box; models would land on / and overflow it"
fi

# ---------------------------------------------------------------- 3. model storage symlink
HARNESS="${HARNESS:-/root/edit-harness}"
if [ -d "$HARNESS" ]; then
  ok "harness present at $HARNESS"
  if [ -L "$HARNESS/data/models" ]; then
    ok "data/models is a symlink -> $(readlink -f "$HARNESS/data/models")"
  elif [ -d "$HARNESS/data/models" ]; then
    warn "data/models is a REAL DIR on this disk — if that is /, big models will overflow it."
    warn "  fix: mv it to $DATA_DIR/models and symlink back."
  else
    if [ "$CHECK_ONLY" -eq 0 ]; then
      mkdir -p "$DATA_DIR/models" "$HARNESS/data"
      ln -sfn "$DATA_DIR/models" "$HARNESS/data/models"
      ok "created data/models -> $DATA_DIR/models"
    else
      warn "data/models missing (would create symlink -> $DATA_DIR/models)"
    fi
  fi
else
  bad "harness not found at $HARNESS — run engine/box_sync_up.sh from the laptop FIRST (code before anything else)"
fi

# ---------------------------------------------------------------- 4. python env
PY="${PY:-python3}"
if command -v "$PY" >/dev/null 2>&1; then
  ok "python: $($PY -V 2>&1)"
  for m in torch transformers numpy scipy; do
    v=$($PY -c "import $m; print(getattr($m,'__version__','?'))" 2>/dev/null)
    if [ -n "$v" ]; then ok "  $m $v"; else bad "  $m NOT importable"; fi
  done
  cuda_ok=$($PY -c "import torch;print(torch.cuda.is_available())" 2>/dev/null)
  [ "$cuda_ok" = "True" ] && ok "  torch sees CUDA" || warn "  torch.cuda.is_available()=False (fine in no-card mode)"
else
  bad "$PY not found"
fi

# ---------------------------------------------------------------- 5. download environment
# THE recipe (07-08, learned the hard way):
#   - hf_xet must be REMOVED from site-packages; HF_HUB_DISABLE_XET=1 alone does NOT work
#   - source /etc/network_turbo (reaches the xet CDN from China); do NOT set HF_ENDPOINT
#   - SOCKS proxy vars break the hf CLI
if [ "$CHECK_ONLY" -eq 0 ]; then
  xet=$(find /root -maxdepth 6 -name 'hf_xet*' -path '*site-packages*' 2>/dev/null | head -3)
  if [ -n "$xet" ]; then
    say "removing hf_xet (mandatory — disable flag is not enough):"
    echo "$xet" | sed 's/^/[boot]       /'
    echo "$xet" | xargs rm -rf 2>/dev/null && ok "hf_xet removed" || bad "could not remove hf_xet"
  else
    ok "hf_xet already absent"
  fi
else
  find /root -maxdepth 6 -name 'hf_xet*' -path '*site-packages*' 2>/dev/null | head -1 | grep -q . \
    && warn "hf_xet present (would remove)" || ok "hf_xet absent"
fi

[ -f /etc/network_turbo ] && ok "/etc/network_turbo present (source it before downloading)" \
                          || warn "/etc/network_turbo missing — HF CDN may be unreachable from China"
[ -n "${HF_ENDPOINT:-}" ] && warn "HF_ENDPOINT is set ($HF_ENDPOINT) — UNSET it; mirrors redirect to xet and break gated repos"
[ -n "${ALL_PROXY:-}${all_proxy:-}" ] && warn "SOCKS proxy vars set — they break the hf CLI; drivers should use: env -u ALL_PROXY -u all_proxy"
[ -f /root/.cache/huggingface/token ] && ok "HF token present" || warn "no HF token (only needed for gated repos; you scp it yourself)"

# ---------------------------------------------------------------- 6. env file for drivers
ENVFILE="$HARNESS/engine/box_env.sh"
if [ "$CHECK_ONLY" -eq 0 ] && [ -d "$HARNESS/engine" ]; then
  cat > "$ENVFILE" <<'EOF'
# sourced by drivers on-box; written by box_bootstrap.sh
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_HUB_DISABLE_XET=1          # belt-and-braces; hf_xet is also physically removed
unset HF_ENDPOINT                     # real HF, needed for gated repos
export ENVP="env -u ALL_PROXY -u all_proxy"
export PY=python3
EOF
  ok "wrote $ENVFILE (drivers should: source engine/box_env.sh)"
fi

# ---------------------------------------------------------------- 7. verdict + checklist
echo
if [ "$FAILED" -eq 0 ]; then
  say "RESULT: READY (no blocking failures)"
else
  say "RESULT: BLOCKED — fix the FAIL lines above before spending money"
fi
cat <<'EOF'
[boot]
[boot] REMAINING MANUAL STEPS (bootstrap deliberately does not do these):
[boot]   1. Models: run your download script in NO-CARD mode (cheap, network-only).
[boot]      source /etc/network_turbo first. Use snapshot_download in a .py file, NOT the
[boot]      hf CLI (its quoting mangles through nested SSH). Launch detached:
[boot]        setsid python dl.py <repo> <dest> > dl.log 2>&1 < /dev/null &
[boot]   2. Switch to GPU mode for compute (no-card = 0.5 core, too weak for smokes).
[boot]   3. Launch a driver detached so it survives SSH teardown:
[boot]        cd /root/edit-harness && setsid -f env <VARS> ./run_X.sh > /root/X.out 2>&1 </dev/null
[boot]      Verify aliveness from a FRESH ssh via the driver's own pidfile — never ps|grep.
[boot]   4. Auto-shutdown: arm it keyed to THIS wave's own DONE markers (old campaigns'
[boot]      markers are already satisfied and would fire immediately). touch /root/NO_SHUTDOWN
[boot]      cancels any pending shutdown.
[boot]   5. Budget clock: pass BUDGET_MIN explicitly; never let a driver count budget from a
[boot]      wait loop (the 07-14 mopup bug consumed 10h of budget waiting and ran nothing).
EOF
exit "$FAILED"
