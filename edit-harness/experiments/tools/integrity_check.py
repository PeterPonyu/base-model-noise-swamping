#!/usr/bin/env python3
"""Model-dir integrity check (no weights loaded into RAM — headers only).

Verifies: config.json parses; every *.safetensors header is readable; summed
parameter count matches --expect_params within 1%; index.json (if present)
references exactly the shard files on disk. Exit 0 = usable, 1 = not.

Also supports legacy ``pytorch_model*.bin`` checkpoints (e.g. EleutherAI/gpt-j-6b's
``float16`` revision, which predates safetensors) via a meta-device torch.load — see
bin_meta_params(). A model dir is checked as safetensors if any *.safetensors file is
present, else as pytorch_model*.bin if any is present, else FAIL.

Usage: integrity_check.py MODEL_DIR --expect_params 8.03e9
"""
import argparse, json, os, struct, sys


def st_params(path):
    """(param_count, size_ok) from a safetensors header — no tensor data read.

    size_ok compares actual file size to 8 + header_len + max data_offset:
    a resumable-download PARTIAL has a complete header but truncated data, so
    header-declared params alone would falsely pass (hole found 2026-07-03)."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    total, max_end = 0, 0
    for k, v in hdr.items():
        if k == "__metadata__":
            continue
        cnt = 1
        for d in v["shape"]:
            cnt *= d
        total += cnt
        max_end = max(max_end, v["data_offsets"][1])
    expected_size = 8 + n + max_end
    return total, os.path.getsize(path) == expected_size, expected_size


def bin_meta_params(path):
    """Param count from a legacy pytorch_model*.bin checkpoint via a META-DEVICE
    torch.load: parses the zip+pickle container and registers every tensor's shape/dtype
    WITHOUT allocating real storage bytes (no weight data ever enters RAM) — same
    "headers only" guarantee as st_params() above, just via torch's own container format
    instead of safetensors'. A truncated/resumable-partial download corrupts the zip's
    central directory (stored at the file's END) or the pickle stream, so torch.load
    raises rather than silently returning a partial state dict. Unlike st_params(), there
    is no declared data_offsets total to cross-check file size against, so "loaded
    without error" is the only truncation signal available here.

    FLOATING-POINT TENSORS ONLY (bug found 2026-07-06 review, before any GPT-J launch):
    a pytorch_model.bin state_dict — unlike a safetensors checkpoint — persists registered
    BUFFERS alongside real weight Parameters, and GPT-J registers a non-trainable BOOL
    causal-mask buffer (``transformer.h.{i}.attn.bias``, shape [1,1,2048,2048]) per layer.
    Summing ALL tensors' numel() counts those 28 buffers as "params" — 117,440,512 extra
    elements on gpt-j-6b — which pushed the total outside the --expect_params 1% band and
    would have made engine/gptj_integrity.ok never get written (every GPT-J science row
    silently CONFIG-skipping). Filtering to dtype.is_floating_point excludes bool/int
    buffers while keeping every real fp16/bf16/fp32 weight.
    """
    import torch
    sd = torch.load(path, map_location="meta", weights_only=True)
    return sum(v.numel() for v in sd.values()
              if hasattr(v, "numel") and getattr(v, "dtype", None) is not None
              and v.dtype.is_floating_point)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--expect_params", type=float, required=True)
    args = ap.parse_args()
    d = args.model_dir
    fails = []

    cfg = os.path.join(d, "config.json")
    try:
        json.load(open(cfg))
    except Exception as e:
        fails.append(f"config.json: {e}")

    shards = sorted(f for f in os.listdir(d) if f.endswith(".safetensors"))
    bin_shards = sorted(f for f in os.listdir(d)
                        if f.startswith("pytorch_model") and f.endswith(".bin"))
    total = 0
    if shards:
        for s in shards:
            try:
                cnt, size_ok, exp = st_params(os.path.join(d, s))
                total += cnt
                if not size_ok:
                    fails.append(f"{s}: TRUNCATED — {os.path.getsize(os.path.join(d, s)):,} B "
                                 f"on disk vs {exp:,} B implied by header")
            except Exception as e:
                fails.append(f"{s}: unreadable header ({e})")

        idx = os.path.join(d, "model.safetensors.index.json")
        if os.path.exists(idx):
            try:
                ref = set(json.load(open(idx))["weight_map"].values())
                if ref != set(shards):
                    fails.append(f"index/shard mismatch: index refs {sorted(ref)} vs disk {shards}")
            except Exception as e:
                fails.append(f"index.json: {e}")
        n_shard_kind = len(shards)
    elif bin_shards:
        # legacy checkpoint (e.g. EleutherAI/gpt-j-6b float16 revision, which ships
        # pytorch_model.bin and predates safetensors adoption) — see bin_meta_params().
        for s in bin_shards:
            try:
                total += bin_meta_params(os.path.join(d, s))
            except Exception as e:
                fails.append(f"{s}: unreadable/truncated pytorch bin ({e})")

        idx = os.path.join(d, "pytorch_model.bin.index.json")
        if os.path.exists(idx):
            try:
                ref = set(json.load(open(idx))["weight_map"].values())
                if ref != set(bin_shards):
                    fails.append(f"index/shard mismatch: index refs {sorted(ref)} vs disk {bin_shards}")
            except Exception as e:
                fails.append(f"pytorch_model.bin.index.json: {e}")
        n_shard_kind = len(bin_shards)
    else:
        fails.append("no .safetensors or pytorch_model*.bin files")
        n_shard_kind = 0

    if not fails and abs(total - args.expect_params) / args.expect_params > 0.01:
        fails.append(f"param count {total:,} vs expected {args.expect_params:,.0f} (>1% off)")

    if fails:
        print(f"INTEGRITY-FAIL {d}: " + "; ".join(fails))
        sys.exit(1)
    print(f"INTEGRITY-OK {d}: {n_shard_kind} shard(s), {total:,} params")


if __name__ == "__main__":
    main()
