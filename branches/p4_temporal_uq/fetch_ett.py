"""fetch_ett.py — download the ETT benchmark CSVs (network; guarded).

Grabs ETTh1/ETTh2/ETTm1/ETTm2 from the public ETDataset repo into ``data/``.
Network I/O happens ONLY when :func:`fetch_all` is called (never at import),
and each file is skipped if it already exists on disk, so re-runs and
network-free environments are safe.

Usage
-----
    python3 fetch_ett.py                 # fetch missing CSVs into ./data
    python3 fetch_ett.py --check         # report presence only, no network
    python3 fetch_ett.py --force         # re-download even if present

Datasets: ETTh1/ETTh2 are hourly, ETTm1/ETTm2 are 15-minute. Target column is
``OT`` (oil temperature). Total size is small (~10MB); the wider "~150MB" figure
in the plan is generous headroom, not a hard requirement.
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")

BASE = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small"
FILES: Dict[str, str] = {
    "ETTh1.csv": f"{BASE}/ETTh1.csv",
    "ETTh2.csv": f"{BASE}/ETTh2.csv",
    "ETTm1.csv": f"{BASE}/ETTm1.csv",
    "ETTm2.csv": f"{BASE}/ETTm2.csv",
}
TARGET_COLUMN = "OT"


def local_path(name: str) -> str:
    return os.path.join(DATA_DIR, name)


def present() -> Dict[str, bool]:
    return {name: os.path.isfile(local_path(name)) for name in FILES}


def fetch_one(name: str, url: str, timeout: float = 120.0) -> str:
    """Download a single CSV (network). Returns the local path."""
    os.makedirs(DATA_DIR, exist_ok=True)
    dst = local_path(name)
    tmp = dst + ".part"
    with urllib.request.urlopen(url, timeout=timeout) as resp, open(tmp, "wb") as fh:
        while True:
            chunk = resp.read(1 << 16)
            if not chunk:
                break
            fh.write(chunk)
    os.replace(tmp, dst)
    return dst


def fetch_all(force: bool = False, allow_network: bool = True) -> Dict[str, str]:
    """Fetch every missing ETT CSV. Guarded: if ``allow_network`` is False or an
    individual download fails, that file is simply reported as missing rather
    than aborting the batch.
    """
    results: Dict[str, str] = {}
    for name, url in FILES.items():
        dst = local_path(name)
        if os.path.isfile(dst) and not force:
            results[name] = dst
            print(f"[fetch_ett] present, skip: {name}")
            continue
        if not allow_network:
            print(f"[fetch_ett] missing (network disabled): {name}")
            continue
        try:
            print(f"[fetch_ett] downloading {name} <- {url}")
            results[name] = fetch_one(name, url)
            size = os.path.getsize(results[name])
            print(f"[fetch_ett] wrote {name} ({size/1e6:.2f} MB)")
        except Exception as exc:  # network/DNS/proxy — report, keep going
            print(f"[fetch_ett] FAILED {name}: {exc}", file=sys.stderr)
    return results


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description="download ETT CSVs (guarded)")
    ap.add_argument("--check", action="store_true", help="report presence only, no network")
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    args = ap.parse_args(argv)

    if args.check:
        for name, ok in present().items():
            print(f"[fetch_ett] {'OK ' if ok else '-- '} {name} -> {local_path(name)}")
        return 0

    fetch_all(force=args.force, allow_network=True)
    missing = [n for n, ok in present().items() if not ok]
    if missing:
        print(f"[fetch_ett] still missing: {missing}", file=sys.stderr)
        return 1
    print("[fetch_ett] all ETT CSVs available.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
