#!/bin/bash
# cloud/failsafe_enhance.sh — billing failsafe for the enhancement round: hard power-off
# at 24h no matter what (estimate is ~16-18h). Cancel with: touch /root/NO_SHUTDOWN
sleep 86400
[ -f /root/NO_SHUTDOWN ] && exit 0
echo "[failsafe $(date)] 24h elapsed — forcing shutdown" >> /root/enhance_4090.log
shutdown -h now
