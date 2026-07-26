"""download_models.py — fetch the overnight breadth-campaign models via curl
(the only reliable path through this proxy). Handles single & sharded
safetensors, resumes on drops, writes a COMPLETE marker per model.

Download order matches the engine's round order so each model is ready when
its round comes. Runs unattended; ~26 GB total.
"""
import json
import os
import subprocess
import time

H = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(H, "data", "models")
LOG = os.path.join(H, "engine", "download.log")
os.makedirs(os.path.join(H, "engine"), exist_ok=True)

ENV = dict(os.environ)
for k in ("ALL_PROXY", "all_proxy"):
    ENV.pop(k, None)

# (repo_id, local_dir) — order = engine round order
MODELS = [
    ("unsloth/Llama-3.2-3B", "Llama-3.2-3B"),
    ("Qwen/Qwen2.5-3B", "Qwen2.5-3B"),
    ("Qwen/Qwen2.5-0.5B", "Qwen2.5-0.5B"),
    ("unsloth/gemma-2-2b", "gemma-2-2b"),
    ("microsoft/Phi-3.5-mini-instruct", "Phi-3.5-mini"),
]
SMALL_FILES = ["config.json", "generation_config.json", "tokenizer.json",
               "tokenizer_config.json", "special_tokens_map.json", "vocab.json",
               "merges.txt", "tokenizer.model", "added_tokens.json"]


def log(msg):
    with open(LOG, "a") as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")


def curl(args, **kw):
    return subprocess.run(["curl", *args], env=ENV, **kw)


def head_size(url):
    r = curl(["-sLI", url], capture_output=True, text=True)
    for line in reversed(r.stdout.splitlines()):
        if line.lower().startswith("content-length:"):
            try:
                return int(line.split(":")[1].strip())
            except ValueError:
                pass
    return None


def fetch_resumable(url, path, tries=80):
    target = head_size(url)
    if not target:
        # unknown size => HEAD failed / 404 / bad index. Do NOT declare success:
        # curl -o would have written a 404/HTML body that looks like a file.
        try:
            os.remove(path)
        except OSError:
            pass
        return False
    for _ in range(tries):
        sz = os.path.getsize(path) if os.path.exists(path) else 0
        if sz >= target:
            return True
        curl(["-L", "-C", "-", "--retry", "5", "--retry-delay", "2",
              "--max-time", "900", "-s", "-o", path, url])
    return os.path.exists(path) and os.path.getsize(path) >= target


def download_model(repo, dirname):
    md = os.path.join(MODELS_DIR, dirname)
    os.makedirs(md, exist_ok=True)
    if os.path.exists(os.path.join(md, "COMPLETE")):
        log(f"{dirname}: already complete")
        return
    base = f"https://huggingface.co/{repo}/resolve/main"
    log(f"{dirname}: start")
    # small files (ignore 404s)
    for f in SMALL_FILES:
        r = curl(["-fsSL", "-o", os.path.join(md, f), f"{base}/{f}"],
                 capture_output=True)
        if r.returncode != 0:
            try:
                os.remove(os.path.join(md, f))
            except OSError:
                pass
    # determine shards
    idx_path = os.path.join(md, "model.safetensors.index.json")
    r = curl(["-fsSL", "-o", idx_path, f"{base}/model.safetensors.index.json"],
             capture_output=True)
    if r.returncode == 0 and os.path.exists(idx_path):
        wm = json.load(open(idx_path))["weight_map"]
        shards = sorted(set(wm.values()))
        log(f"{dirname}: {len(shards)} shards")
    else:
        shards = ["model.safetensors"]
    ok = True
    for s in shards:
        if not fetch_resumable(f"{base}/{s}", os.path.join(md, s)):
            ok = False
            log(f"{dirname}: shard {s} INCOMPLETE")
    # verify config + a tokenizer + every shard (present & nonzero) BEFORE marking COMPLETE,
    # so from_pretrained can never hit a dir that is missing tokenizer/config or has 0-byte weights.
    def nonzero(p):
        return os.path.exists(p) and os.path.getsize(p) > 0
    cfg_ok = nonzero(os.path.join(md, "config.json"))
    tok_ok = any(nonzero(os.path.join(md, t)) for t in ("tokenizer.json", "tokenizer.model", "vocab.json"))
    shards_ok = ok and all(nonzero(os.path.join(md, s)) for s in shards)
    if cfg_ok and tok_ok and shards_ok:
        open(os.path.join(md, "COMPLETE"), "w").close()
        sz = sum(os.path.getsize(os.path.join(md, s)) for s in shards) // 10**6
        log(f"{dirname}: COMPLETE ({sz} MB)")
    else:
        log(f"{dirname}: FAILED (cfg={cfg_ok} tok={tok_ok} shards={shards_ok}) "
            "— NO COMPLETE marker; engine will wait then skip")


def main():
    log("=== download manager start ===")
    for repo, d in MODELS:
        try:
            download_model(repo, d)
        except Exception as e:
            log(f"{d}: ERROR {e}")
    log("=== download manager done ===")


if __name__ == "__main__":
    main()
