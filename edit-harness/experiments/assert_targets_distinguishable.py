#!/usr/bin/env python3
"""Fail closed when a tokenizer collapses a wave's edit targets.

This is the tokenizer-only wave-preflight entry point for the 2026-07-30
Phi-3.5 first-token collision. It loads no model weights and performs no GPU
work.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterable
from typing import Any

HARNESS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HARNESS)

from metrics import assert_targets_distinguishable  # noqa: E402


def _target_text(value: Any) -> str | None:
    if isinstance(value, dict):
        value = value.get("str")
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def load_targets(path: str) -> list[str]:
    """Collect every true/new target that a CounterFact-style wave may sample."""
    with open(path, encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError(f"dataset root must be a list, got {type(records).__name__}")

    targets: list[str] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        rewrites = record.get("requested_rewrite", record)
        if isinstance(rewrites, dict):
            rewrite_rows: Iterable[dict[str, Any]] = (rewrites,)
        elif isinstance(rewrites, list):
            rewrite_rows = (row for row in rewrites if isinstance(row, dict))
        else:
            continue
        for rewrite in rewrite_rows:
            for key in ("target_new", "target_true"):
                text = _target_text(rewrite.get(key))
                if text:
                    targets.append(text)
    if not targets:
        raise ValueError(f"no target_new/target_true strings found in {path}")
    return targets


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Assert that a local tokenizer distinguishes the wave target set."
    )
    parser.add_argument("--tokenizer", required=True, help="local tokenizer/model directory")
    parser.add_argument("--data", required=True, help="wave dataset JSON")
    parser.add_argument("--label", help="model label used in diagnostics")
    parser.add_argument(
        "--extra-target",
        action="append",
        default=[],
        help="additional generated target, such as the deletion refusal string",
    )
    parser.add_argument("--min-ratio", type=float, default=0.5)
    args = parser.parse_args()

    label = args.label or args.tokenizer
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer, local_files_only=True
        )
        targets = load_targets(args.data) + args.extra_target
        report = assert_targets_distinguishable(
            tokenizer, targets, min_ratio=args.min_ratio
        )
    except Exception as exc:  # fail closed: an unreadable tokenizer/data file is not a pass
        print(f"TOKENIZER-GATE FAIL {label}: {exc}", file=sys.stderr)
        return 1

    print(
        f"TOKENIZER-GATE PASS {label}: "
        f"{report['n_first_tokens']}/{report['n_targets']} distinct first tokens "
        f"(ratio {report['ratio']:.3f}, {report['n_collisions']} colliding pairs)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
