#!/usr/bin/env bash
# box36039_autofire.sh — wait for box 36039 SSH, then sync + preflight + launch the
# gap-closure chain, unattended (user blanket authorization 2026-07-31).
# Idempotent: sync is code-only; chain cells skip-if-done; chain preflight aborts loudly.
# Kill by PID only (pidfile engine/box36039_autofire.pid).
set -u
cd "$(dirname "$0")/.." || exit 2   # edit-harness/
LOG=engine/box36039_autofire.log
PIDFILE=engine/box36039_autofire.pid
HOST="${HOST:-root@connect.cqa1.seetacloud.com}"
PORT="${PORT:-36039}"
DEST="${DEST:-/root/edit-harness-deploy-20260727}"
SSH="ssh -p $PORT -o ConnectTimeout=10 -o BatchMode=yes -o ServerAliveInterval=15"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
[ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE" 2>/dev/null)" 2>/dev/null \
  && { echo "REFUSE: already running (pid $(cat "$PIDFILE"))" >&2; exit 7; }
echo $$ > "$PIDFILE"; trap 'rm -f "$PIDFILE"' EXIT

# ---- 1) wait for SSH (max 12h) ----
t0=$(date +%s)
until timeout 25 $SSH "$HOST" 'echo SSH_OK' 2>/dev/null | grep -q SSH_OK; do
  [ $(( $(date +%s) - t0 )) -le 43200 ] || { log "ABORT: box unreachable after 12h"; exit 10; }
  sleep 60
done
log "SSH OK — box is up"

# ---- 2) environment survey ----
$SSH "$HOST" 'hostname; nvidia-smi -L; echo ---; ls /root/autodl-tmp/venvs/ 2>/dev/null; ls "$HOME" | head' \
  >> "$LOG" 2>&1 || { log "ABORT: survey failed"; exit 11; }
grep -q 'edit-harness-deploy-20260727' "$LOG" || log "WARN: deploy dir not confirmed in survey (continuing; sync creates DEST)"

# ---- 3) sync code up ----
log "sync up -> $DEST"
PORT="$PORT" DEST="$DEST" bash engine/box_sync_up.sh "$HOST" --go >> "$LOG" 2>&1 \
  || { log "ABORT: sync failed"; exit 12; }

# ---- 4) on-box preflight + model check/download (small public models only) ----
$SSH "$HOST" "cd $DEST && bash engine/box_preflight_onbox.sh generic" >> "$LOG" 2>&1 \
  || { log "ABORT: on-box preflight blocked"; exit 13; }
for spec in "Qwen/Qwen2.5-1.5B|Qwen2.5-1.5B" "microsoft/Phi-3.5-mini-instruct|Phi-3.5-mini" "meta-llama/Llama-3.2-1B|Llama-3.2-1B"; do
  repo="${spec%%|*}"; name="${spec##*|}"
  if ! $SSH "$HOST" "[ -d $DEST/data/models/$name ] || [ -d /root/autodl-tmp/models/$name ]" 2>/dev/null; then
    log "model $name missing on box — downloading $repo (public check)"
    $SSH "$HOST" "cd $DEST && mkdir -p data/models /root/autodl-tmp/hf_cache && \
      HF_HOME=/root/autodl-tmp/hf_cache /root/autodl-tmp/venvs/ifa-20260727/bin/python -c \"
from huggingface_hub import snapshot_download
snapshot_download('$repo', local_dir='$DEST/data/models/$name', ignore_patterns=['*.bin','*.pth','*.h5','*.msgpack'])
print('downloaded $name')\"" >> "$LOG" 2>&1 \
      || { log "ABORT: download failed for $repo (gated? needs token?)"; exit 14; }
  else
    log "model $name present"
  fi
done

# ---- 5) launch the dual-card chain, detached on-box ----
log "launching chain_36039_20260731.sh"
$SSH "$HOST" "cd $DEST && setsid nohup bash engine/chain_36039_20260731.sh \
  >> engine/chain_36039_20260731.nohup.log 2>&1 & echo CHAIN_LAUNCHED pid=\$!" \
  >> "$LOG" 2>&1 || { log "ABORT: chain launch failed"; exit 15; }
sleep 20
$SSH "$HOST" "tail -5 $DEST/engine/chain_36039_20260731.log 2>/dev/null" >> "$LOG" 2>&1 || true
log "AUTOFIRE COMPLETE — chain running on box; pull manifests: phi_refix_b6.txt, phi_refix.txt, chain36039_gapbatch1.txt"
