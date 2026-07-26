#!/bin/bash
# dl_llama2_watcher.sh — poll for Llama-2 license acceptance, then download to the R-C path.
# Runs on box 29246 (Pro-6000). HF gotchas honored: HF_HUB_DISABLE_XET=1 (xet breaks on this
# route), network_turbo sourced. User accepted the license in-browser; this fires on detection.
DEST=/root/autodl-tmp/models/Llama-2-13b-hf
LOG=/root/dl_llama2_watcher.log
export HF_TOKEN=$(cat /root/.cache/huggingface/token)
export HF_HUB_DISABLE_XET=1
source /etc/network_turbo 2>/dev/null
echo "[$(date +%T)] watcher started, polling for license..." >> $LOG
while true; do
  code=$(curl -s -o /dev/null -w "%{http_code}" -I -L -H "Authorization: Bearer $HF_TOKEN" --max-time 30 https://huggingface.co/meta-llama/Llama-2-13b-hf/resolve/main/config.json)
  echo "[$(date +%T)] config.json HEAD: $code" >> $LOG
  [ "$code" = "200" ] && break
  sleep 300
done
echo "[$(date +%T)] LICENSE ACCEPTED — starting download" >> $LOG
mkdir -p $DEST
/root/miniconda3/bin/hf download meta-llama/Llama-2-13b-hf --local-dir $DEST >> $LOG 2>&1
rc=$?
echo "[$(date +%T)] hf download rc=$rc" >> $LOG
if [ $rc -eq 0 ] && [ -f $DEST/config.json ] && ls $DEST/model-0000*-of-0000*.safetensors >/dev/null 2>&1; then
  n=$(ls $DEST/model-0000*-of-0000*.safetensors | wc -l)
  sz=$(du -sh $DEST | cut -f1)
  echo "[$(date +%T)] DOWNLOAD COMPLETE: $n shards, $sz -> $DEST" >> $LOG
  touch /root/LLAMA2_DOWNLOAD_DONE.ok
else
  echo "[$(date +%T)] DOWNLOAD FAILED or incomplete (rc=$rc) — see log" >> $LOG
fi
