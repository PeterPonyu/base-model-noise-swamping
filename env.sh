#!/usr/bin/env bash
# Source this before ANY GPU/HF command:  source ~/Desktop/idea-feasibility-analysis/env.sh
# Fixes the two env blockers found in pre-flight (2026-06-30).

# BLOCKER 1: ALL_PROXY=socks://... breaks httpx-based huggingface_hub ("Unknown scheme").
#   The same local proxy serves HTTP on the same port; keep HTTPS_PROXY(http), drop socks.
unset ALL_PROXY all_proxy
# (HTTPS_PROXY/HTTP_PROXY=http://127.0.0.1:7897 are valid and kept; direct also works.)

# Activate the dl env (Blackwell torch 2.12.1+cu130).
# Use:  conda run -n dl <cmd>   OR   conda activate dl
export OMC_EDIT_ENV=dl

# Optional: set HF token to lift rate limits (recommended for bulk download)
# export HF_TOKEN=hf_xxx

# Ollama (LLM backend) — restart after reboot:
#   OLLAMA_HOME=/home/zeyufu/.ollama OLLAMA_FLASH_ATTENTION=1 \
#     /home/zeyufu/miniconda3/envs/dl/bin/ollama serve &
# NEVER set CUDA_VISIBLE_DEVICES for ollama (breaks GPU discovery).

echo "[env] ALL_PROXY unset; dl env ready; torchvision 0.27.1+cu130 (fixed)."
