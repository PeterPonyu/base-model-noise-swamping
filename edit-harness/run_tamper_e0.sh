#!/bin/bash
cd /home/zeyufu/Desktop/idea-feasibility-analysis/edit-harness
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
echo $$ > /tmp/tamper_e0.pid
echo "[tamper] start $(date)"
timeout 28800 python3 experiments/tamper_e0.py > results/tamper_e0/run.log 2>&1
echo "[tamper] rc=$? $(date)" >> results/tamper_e0/run.log
