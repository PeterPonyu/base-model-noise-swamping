#!/bin/bash
# dl_supervisor2.sh — curl-based model downloader. Why not hf_hub: its 1.x .incomplete
# files carry a per-PROCESS session suffix -> killed/restarted processes CANNOT resume
# each other's partials (verified 2026-07-03 00:05; two orphaned GB discarded). curl -C -
# resumes byte-exactly from the target file itself across any restart/reboot.
set -u
H=/home/zeyufu/Desktop/idea-feasibility-analysis/edit-harness
cd "$H" || exit 1
L=engine/dl_supervisor2.log
echo $$ > engine/dl_supervisor2.pid
lg(){ echo "[$(date '+%F %T')] $*" >> "$L"; }
lg "================ DL2 START (pid $$) ================"

fetch(){  # $1 url  $2 target
  local url="$1" tgt="$2" expected have attempt=0
  mkdir -p "$(dirname "$tgt")"
  # x-linked-size = HF's TRUE file size for LFS/Xet files; the 302 redirect body
  # content-length (1052) is NOT the file — that bug thrashed the loop (2026-07-03).
  local head; head=$(curl -sIL "$url" 2>/dev/null)
  expected=$(echo "$head" | grep -i '^x-linked-size:' | tail -1 | tr -dc 0-9)
  [ -z "$expected" ] && expected=$(echo "$head" | grep -i '^content-length:' | tail -1 | tr -dc 0-9)
  if [ -z "$expected" ]; then lg "HEAD failed for $(basename "$tgt") — will retry whole file later"; return 1; fi
  while :; do
    have=$(stat -c%s "$tgt" 2>/dev/null || echo 0)
    if [ "$have" -eq "$expected" ]; then lg "DONE $(basename "$tgt") ($have B)"; return 0; fi
    if [ "$have" -gt "$expected" ]; then lg "OVERSIZED $(basename "$tgt") ($have > $expected) — restarting from 0"; rm -f "$tgt"; fi
    attempt=$((attempt+1))
    lg "GET $(basename "$tgt") attempt $attempt from byte $have / $expected"
    # --speed-limit/time: abort (rc 28) if <1KB/s for 90s -> loop resumes; never orphans bytes.
    # --max-time 1800: force re-negotiation every 30min so a connection that started on a
    # congested route re-samples a fresh path once the link recovers (resume is byte-exact).
    curl -sS -L -C - --speed-limit 1024 --speed-time 90 --connect-timeout 30 --max-time 1800 -o "$tgt" "$url" 2>>"$L"
    local rc=$?
    [ "$rc" -ne 0 ] && lg "curl rc=$rc on $(basename "$tgt") — retrying in 15s"; sleep 15
    if [ "$attempt" -ge 200 ]; then lg "GIVING UP $(basename "$tgt") after 200 attempts"; return 1; fi
  done
}

B=https://huggingface.co
ok=0; fail=0
# Llama-3.1-8B: 4 shards (small files already complete on disk)
for f in model-00001-of-00004.safetensors model-00002-of-00004.safetensors model-00003-of-00004.safetensors model-00004-of-00004.safetensors; do
  fetch "$B/NousResearch/Meta-Llama-3.1-8B/resolve/main/$f" "data/models/Llama-3.1-8B/$f" && ok=$((ok+1)) || fail=$((fail+1))
done
# gpt2-xl: the single weights file (small files already complete on disk)
fetch "$B/openai-community/gpt2-xl/resolve/main/model.safetensors" "data/models/gpt2-xl/model.safetensors" && ok=$((ok+1)) || fail=$((fail+1))

lg "================ DL2 COMPLETE ok=$ok fail=$fail ================"
echo "DL2_DONE ok=$ok fail=$fail" >> "$L"
