#!/usr/bin/env python3
"""cloud/dl_extension_models.py — box-side downloader for the 2026-07-11 extension wave's
Track-1 7-9B family-transfer battery (run_family_transfer.sh). Companion to
cloud/dl_pythia.py (same shape: snapshot_download straight into data/models/<name>, retry
x3, safetensors-only allow_patterns, HF_HUB_DISABLE_XET=1, caller sources
/etc/network_turbo) — NOT wired into setup_autodl.sh's MODELS_CORE list, kept standalone
like dl_pythia.py was, since this is a separate track with its own ask-first gate.

MODELS (4, ~64GB combined bf16-safetensors-only — 14.5+15.2+18.5+16GB per
cloud/EXTENSION-WAVE-RUNBOOK.md's download table — verify disk headroom before running):
  - mistralai/Mistral-7B-v0.3      UNGATED (verified via a live config.json fetch,
    2026-07-11 — no HF auth needed)
  - Qwen/Qwen2.5-7B                UNGATED (verified same way)
  - unsloth/gemma-2-9b             UNGATED MIRROR of the gated google/gemma-2-9b (same
    42-layer gemma2 architecture, config.json fetched clean with no auth — this repo
    already uses the identical unsloth-mirror pattern locally for gemma-2-2b, see
    data/models/gemma-2-2b). Set EXT_GEMMA9B_REPO=google/gemma-2-9b to use the official
    gated repo instead (needs `hf auth login` + license acceptance first).
  - unsloth/Meta-Llama-3.1-8B-Instruct   UNGATED MIRROR of the gated
    meta-llama/Llama-3.1-8B-Instruct (identical 32-layer llama architecture, verified
    clean). Set EXT_LLAMA8BINST_REPO=meta-llama/Llama-3.1-8B-Instruct for the official
    gated repo (separate license gate from the already-accepted Llama-3.2-1B-Instruct —
    do not assume it carries over).
ALL FOUR ARE ASK-FIRST PER WORKSPACE POLICY — this script does not run itself; the
runbook invokes it as an explicit, user-approved step.

Usage (on the box, no-GPU tier):
  ( source /etc/network_turbo; python cloud/dl_extension_models.py ) &
Env overrides: EXT_GEMMA9B_REPO, EXT_LLAMA8BINST_REPO (see above).
"""
from __future__ import annotations

import hashlib
import os
import sys
import time

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # burned-in lesson: xet hangs behind this box's proxy/mirror
from huggingface_hub import snapshot_download  # noqa: E402

HARNESS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALLOW = ["*.safetensors", "*.safetensors.index.json", "config.json", "generation_config.json",
         "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
         "tokenizer.model", "vocab.json", "merges.txt", "added_tokens.json"]

MODELS = [
    ("mistralai/Mistral-7B-v0.3", "Mistral-7B-v0.3"),
    ("Qwen/Qwen2.5-7B", "Qwen2.5-7B"),
    (os.environ.get("EXT_GEMMA9B_REPO", "unsloth/gemma-2-9b"), "gemma-2-9b"),
    (os.environ.get("EXT_LLAMA8BINST_REPO", "unsloth/Meta-Llama-3.1-8B-Instruct"), "Llama-3.1-8B-Instruct"),
]


def sha256_sample(dst: str, max_files: int = 1) -> None:
    """Provenance logging only (NOT a pre-registered expected-hash check — HF snapshot
    downloads don't ship a single canonical sha256 the way the GitHub-sourced dataset
    files in data/DOWNLOADS-20260706.md did). Hashes the largest safetensors shard so the
    log carries a verifiable fingerprint if this download is ever disputed; real integrity
    verification is integrity_check.py's --expect_params header check (run by the driver's
    own preflight, same as every other model in this repo)."""
    try:
        shards = [os.path.join(dst, f) for f in os.listdir(dst) if f.endswith(".safetensors")]
        if not shards:
            return
        biggest = max(shards, key=os.path.getsize)
        h = hashlib.sha256()
        with open(biggest, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        print(f"[dl_extension_models] sha256({os.path.basename(biggest)}) = {h.hexdigest()}", flush=True)
    except Exception as e:
        print(f"[dl_extension_models] sha256 sample skipped: {e}", flush=True)


rc = 0
for repo, name in MODELS:
    dst = os.path.join(HARNESS, "data", "models", name)
    ok = False
    for attempt in range(1, 4):
        try:
            snapshot_download(repo, local_dir=dst, allow_patterns=ALLOW, max_workers=8)
            print(f"[dl_extension_models] {name} ({repo}) DOWNLOAD-OK -> {dst}", flush=True)
            ok = True
            break
        except Exception as e:
            print(f"[dl_extension_models] {name} ({repo}) attempt {attempt} failed: {e}", flush=True)
            time.sleep(30)
    if not ok:
        rc = 1
        print(f"[dl_extension_models] {name} FAILED after 3 attempts — "
              f"run_family_transfer.sh will MODEL-ABSENT/INTEGRITY-FAIL skip it", flush=True)
        continue
    sha256_sample(dst)

sys.exit(rc)
