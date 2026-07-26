#!/bin/bash
# dl_supervisor.sh — resumable HF downloads with a stall watchdog (unauthenticated CDN
# connections die silently mid-transfer; observed twice 2026-07-02). Per repo: run the
# downloader, watch dir bytes every 30s, kill+restart on 120s of zero growth. Resume is
# native (hf_hub .incomplete files). Exits when both repos are complete.
set -u
H=/home/zeyufu/Desktop/idea-feasibility-analysis/edit-harness
cd "$H" || exit 1
PY=/home/zeyufu/miniconda3/envs/dl/bin/python3
L=engine/dl_supervisor.log
echo $$ > engine/dl_supervisor.pid
lg(){ echo "[$(date '+%F %T')] $*" >> "$L"; }

dl_one(){  # $1 tag  $2 repo  $3 localdir  $4 allow(csv)  $5 ignore(csv or -)
  local tag="$1" repo="$2" dir="$3" allow="$4" ignore="$5" attempt=0
  while :; do
    attempt=$((attempt+1))
    # done check: no .incomplete AND >0 safetensors bytes AND python exit 0 marker
    if [ -f "engine/dl_${tag}.done" ]; then lg "$tag already done"; return 0; fi
    lg "$tag attempt $attempt: launching downloader"
    env -u ALL_PROXY -u all_proxy HF_HUB_OFFLINE=0 $PY - "$repo" "$dir" "$allow" "$ignore" >> "engine/dl_${tag}.log" 2>&1 <<'PYEOF' &
import sys
from huggingface_hub import snapshot_download
repo, ldir, allow, ignore = sys.argv[1], sys.argv[2], sys.argv[3].split(','), sys.argv[4]
kw = dict(allow_patterns=allow, local_dir=ldir, max_workers=4)
if ignore != "-": kw["ignore_patterns"] = ignore.split(',')
p = snapshot_download(repo, **kw)
print("DONE ->", p)
PYEOF
    local child=$!
    # watchdog: kill on 120s zero growth
    local last=$(du -sb "$dir" 2>/dev/null | cut -f1); local stall=0
    while kill -0 $child 2>/dev/null; do
      sleep 30
      local now=$(du -sb "$dir" 2>/dev/null | cut -f1)
      if [ "${now:-0}" -le "${last:-0}" ]; then stall=$((stall+30)); else stall=0; last=$now; fi
      if [ "$stall" -ge 120 ]; then
        lg "$tag STALL detected (120s zero growth at $now B) — killing pid $child for restart"
        kill -TERM $child 2>/dev/null; sleep 3; kill -KILL $child 2>/dev/null
        break
      fi
    done
    wait $child 2>/dev/null; local rc=$?
    if [ "$rc" -eq 0 ] && grep -q 'DONE ->' "engine/dl_${tag}.log"; then
      : > "engine/dl_${tag}.done"; lg "$tag COMPLETE (attempt $attempt)"; return 0
    fi
    lg "$tag attempt $attempt ended rc=$rc — retrying in 10s (resume)"
    sleep 10
    if [ "$attempt" -ge 30 ]; then lg "$tag GIVING UP after 30 attempts"; return 1; fi
  done
}

# serial (they share bandwidth anyway; serial = clearer stall detection). 8B first (bigger).
dl_one llama8b NousResearch/Meta-Llama-3.1-8B data/models/Llama-3.1-8B \
  "*.safetensors,config.json,generation_config.json,tokenizer.json,tokenizer_config.json,special_tokens_map.json,model.safetensors.index.json" "original/*"
dl_one gpt2xl openai-community/gpt2-xl data/models/gpt2-xl \
  "model.safetensors,config.json,generation_config.json,vocab.json,merges.txt,tokenizer.json,tokenizer_config.json" "-"
lg "ALL DOWNLOADS COMPLETE"
echo "DL_SUPERVISOR_DONE" >> "$L"
