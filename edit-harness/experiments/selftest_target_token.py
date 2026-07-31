#!/usr/bin/env python3
"""selftest_target_token.py — regression test for the Phi-3.5 first-token collision.

Defect record: docs/findings/findings-PHI35-TOKENIZER-COLLISION-2026-07-30.md

Runs against every local checkpoint (CPU, tokenizer-only, no weights loaded) and
asserts that distinct edit targets map to distinct first CONTENT tokens. Before
the 2026-07-30 fix, Phi-3.5 failed this on every pair.

Usage:
    python3 experiments/selftest_target_token.py            # all local models
    python3 experiments/selftest_target_token.py --tokenizer <path_or_repo>
"""
from __future__ import annotations

import argparse
import os
import sys

HARNESS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HARNESS)

from metrics import (  # noqa: E402
    assert_targets_distinguishable,
    first_target_token_id,
    target_token_ids,
)

# Representative CounterFact-style objects plus the deletion refusal string.
TARGETS = ["Paris", "Michael", "London", "English", "I cannot answer"]


def check(name: str, tok) -> bool:
    ids = {t: target_token_ids(tok, t) for t in TARGETS}
    firsts = {t: first_target_token_id(tok, t) for t in TARGETS}
    distinct = len(set(firsts.values())) == len(firsts)
    # No returned first token may be whitespace-only.
    ws = [t for t, i in firsts.items() if tok.decode([i]).strip() == ""]
    try:
        assert_targets_distinguishable(tok, TARGETS)
        guard_ok = True
    except ValueError:
        guard_ok = False
    ok = distinct and not ws and guard_ok
    print(f"{'PASS' if ok else 'FAIL'} {name}")
    print(f"     raw first ids : {[ids[t][0] for t in TARGETS]}")
    print(f"     content ids   : {[firsts[t] for t in TARGETS]}")
    if ws:
        print(f"     whitespace-only returned for: {ws}")
    if not distinct:
        print("     COLLISION: distinct targets share a first content token")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", help="single tokenizer path or repo id")
    args = ap.parse_args()
    from transformers import AutoTokenizer

    if args.tokenizer:
        targets = [(args.tokenizer, args.tokenizer)]
    else:
        root = os.path.join(HARNESS, "data", "models")
        targets = [(d, os.path.join(root, d)) for d in sorted(os.listdir(root))
                   if os.path.isdir(os.path.join(root, d))]

    results = []
    for name, path in targets:
        try:
            tok = AutoTokenizer.from_pretrained(path)
        except Exception as e:  # noqa: BLE001 — one unreadable tokenizer must not sink the sweep
            print(f"SKIP {name}: {e}")
            continue
        results.append(check(name, tok))

    if not results:
        print("no tokenizers checked")
        return 2
    failed = results.count(False)
    print(f"\n{len(results) - failed}/{len(results)} tokenizers PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
