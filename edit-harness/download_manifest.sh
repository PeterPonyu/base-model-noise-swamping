#!/usr/bin/env bash
# STAGED bulk-download manifest for the knowledge-editing fan-out engine.
# NOT auto-run. Run a wave explicitly:  bash download_manifest.sh wave0
# Pre-flight already validated proxy + GPU + edit primitives on tiny proxies (see PREFLIGHT.md).
set -euo pipefail
source "$(dirname "$0")/../env.sh"        # unsets socks proxy (download blocker fix)
HF() { conda run -n dl hf download "$@"; } # new HF CLI; resumes; skips cached
DATA="$(dirname "$0")/data"; mkdir -p "$DATA"
get() { [ -f "$DATA/$2" ] && echo "  cached $2" || conda run -n dl curl -fL --retry 3 -o "$DATA/$2" "$1"; }

wave0_trunk() {   # ~14 GB — canonical ROME/MEMIT baseline; unblocks trunk + B1/B2/B3/B6
  echo "== Wave0: trunk model + editing datasets =="
  HF EleutherAI/gpt-j-6b --include "*.json" "*.safetensors" "*.model" "tokenizer*"   # ungated, ~12GB fp16
  get https://rome.baulab.info/data/dsets/counterfact.json        counterfact.json
  get https://rome.baulab.info/data/dsets/zsre_mend_eval.json     zsre_eval.json
  get https://rome.baulab.info/data/dsets/zsre_mend_train.json    zsre_train.json
  echo "Wave0 done. Run trunk gate: MEMIT×GPT-J-6B×CounterFact 5-metric baseline."
}

wave1_b1() {      # +~0.1 GB — MQuAKE multi-hop (B1 collapse story)
  echo "== Wave1: MQuAKE multi-hop =="
  for f in MQuAKE-CF-3k.json MQuAKE-CF.json MQuAKE-T.json; do
    get "https://raw.githubusercontent.com/princeton-nlp/MQuAKE/main/datasets/$f" "$f"
  done
  echo "Wave1 done. Run B1 gate: multi-hop consistency vs single-hop efficacy gap."
}

wave2_b5() {      # +~18 GB — multimodal VLM editing (do AFTER text line mature)
  echo "== Wave2: VLM for multimodal editing =="
  HF Qwen/Qwen2-VL-7B-Instruct --include "*.json" "*.safetensors" "tokenizer*" "*.txt"
  echo "Wave2 done. (MMEdit/E-VQA editing benchmark: fetch per B5 spec.)"
}

optional_llama() {  # alt/extra trunk model (ungated mirror; modern relevance)
  echo "== Optional: Llama-3-8B (ungated mirror) =="
  HF NousResearch/Meta-Llama-3-8B --include "*.json" "*.safetensors" "tokenizer*"   # ~16GB
}

case "${1:-help}" in
  wave0) wave0_trunk ;;
  wave1) wave1_b1 ;;
  wave2) wave2_b5 ;;
  llama) optional_llama ;;
  all)   wave0_trunk; wave1_b1 ;;        # text line only; B5 separate
  *) echo "usage: bash download_manifest.sh {wave0|wave1|wave2|llama|all}";
     echo "  wave0 ~14GB trunk(GPT-J-6B+CF/zsRE) | wave1 +0.1GB MQuAKE | wave2 +18GB VLM | llama +16GB alt";;
esac
