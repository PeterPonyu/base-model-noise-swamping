#!/usr/bin/env python3
"""dl_pythia.py — box-side Pythia download for P2 of the 2026-07-09 enhancement round.

Downloads EleutherAI/pythia-1.4b then pythia-2.8b straight into data/models/<name>
(real files, the B1 local-dir convention). Runs CONCURRENTLY with the P1/P3 GPU work
(network+disk only); run_pythia.sh gates each model on dir presence + integrity, so a
still-running or failed download degrades to a clean MODEL-ABSENT/INTEGRITY-FAIL skip.

LESSONS BAKED IN (memory/autodl-download-routine-20260708.md + the 20260709 NeoX
near-overflow): allow_patterns restricts to safetensors + tokenizer/config files —
dual-format repos otherwise download BOTH weight formats (the neox repo cost 2x disk);
HF_HUB_DISABLE_XET=1 (xet hangs); caller sources /etc/network_turbo.

Usage (on the box):  ( source /etc/network_turbo; python cloud/dl_pythia.py ) &
"""
from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from huggingface_hub import snapshot_download  # noqa: E402

HARNESS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALLOW = ["*.safetensors", "*.safetensors.index.json", "config.json", "generation_config.json",
         "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
         "vocab.json", "merges.txt"]
MODELS = [("EleutherAI/pythia-1.4b", "pythia-1.4b"),
          ("EleutherAI/pythia-2.8b", "pythia-2.8b")]

rc = 0
for repo, name in MODELS:
    dst = os.path.join(HARNESS, "data", "models", name)
    ok = False
    for attempt in range(1, 4):
        try:
            snapshot_download(repo, local_dir=dst, allow_patterns=ALLOW, max_workers=8)
            print(f"[dl_pythia] {name} DOWNLOAD-OK -> {dst}", flush=True)
            ok = True
            break
        except Exception as e:
            print(f"[dl_pythia] {name} attempt {attempt} failed: {e}", flush=True)
            time.sleep(30)
    if not ok:
        rc = 1
        print(f"[dl_pythia] {name} FAILED after 3 attempts — run_pythia.sh will "
              f"MODEL-ABSENT/INTEGRITY-FAIL skip it", flush=True)
sys.exit(rc)
