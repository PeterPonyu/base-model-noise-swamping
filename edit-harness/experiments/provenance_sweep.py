#!/usr/bin/env python3
"""Read-only forensic provenance sweep for experiment result trees."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

SKIP_DIRS = {".synthetic-relabel-bak", "archive", "smoke"}
SEED_TOKEN_RE = re.compile(r"(?i)(?:^|[_-])s([0-2])(?=[_.-]|$)|(?:^|[_-])seed[_-]?([0-2])(?=[_.-]|$)")
SEED_KEY_RE = re.compile(r"(?i)(?:^|[_-])(?:random_)?seed(?:$|[_-])")
AGGREGATE_RE = re.compile(
    r"(?i)(?:^|[_-])(analysis|aggregate|aggregated|table|manifest|validation|verdict|report|summary|meta|stats|curve|eval)(?:[_\-.]|$)"
)
MEASUREMENT_RE = re.compile(r"(?i)(?:^|[_-])(cell|measurement|result|run|trial)(?:[_\-.]|$)")
TIMING_COST_RE = re.compile(
    r"(?i)(cost|tim(?:e|ing)|elapsed|duration|runtime|latency|second|minute|hour|wall[_-]?clock|gpu[_-]?h)"
)
METRIC_RE = re.compile(
    r"(?i)(metric|score|loss|damage|drop|rho|corr|accuracy|rate|asr|esr|success|benefit|cos|norm|logit|prob|kl|perplex|effect|survival|fragility|gain|interference|error)"
)
CONFIG_RE = re.compile(
    r"(?i)(?:^n_|neighborhood|^id$|^index$|^seed$|(?:^|[_-])(?:random_)?seed(?:$|[_-])|layer|steps?|epoch|batch|n_edits|n_items|n_obs|n_groups|count|size|rank|device|dtype|schema|version)$"
)
ROUND_ANCHORS = {1500.0, 2400.0, 4000.0}


@dataclass
class JsonRecord:
    path: Path
    rel: str
    tree: str
    mtime_second: int
    kind: str
    seed: int | None
    data: Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Result root to sweep recursively")
    parser.add_argument("--recent-seconds", type=int, default=300, help="Skip files newer than this age")
    parser.add_argument("--output", type=Path, help="Write JSON findings to this path (outside result root)")
    return parser.parse_args()


def skipped_path(path: Path, root: Path) -> bool:
    try:
        parts = path.relative_to(root).parts[:-1]
    except ValueError:
        return True
    return any(part in SKIP_DIRS for part in parts)


def tree_for(path: Path, root: Path) -> str:
    parts = path.relative_to(root).parts
    return parts[0] if len(parts) > 1 else "(root)"


def seed_from_path(rel: str) -> int | None:
    matches = list(SEED_TOKEN_RE.finditer(rel))
    if not matches:
        return None
    match = matches[-1]
    return int(match.group(1) or match.group(2))


def seed_from_data(data: Any) -> int | None:
    found: set[int] = set()

    def visit(value: Any, key: str = "") -> None:
        if isinstance(value, dict):
            for child_key, child in value.items():
                if SEED_KEY_RE.search(str(child_key)) and isinstance(child, int) and not isinstance(child, bool):
                    if child in (0, 1, 2):
                        found.add(child)
                elif isinstance(child, (dict, list)):
                    visit(child, str(child_key))
        elif isinstance(value, list):
            for child in value:
                if isinstance(child, (dict, list)):
                    visit(child, key)

    visit(data)
    return next(iter(found)) if len(found) == 1 else None


def classify(rel: str, seed: int | None) -> str:
    name = Path(rel).name
    if AGGREGATE_RE.search(rel):
        return "aggregate"
    if seed is not None:
        return "measurement"
    if MEASUREMENT_RE.search(name):
        return "measurement"
    return "unclassified"


def canonical_seed_path(rel: str) -> str:
    return SEED_TOKEN_RE.sub(lambda match: match.group(0).replace(match.group(1) or match.group(2), "{seed}"), rel)


def stable_stat(path: Path, cutoff: float) -> os.stat_result | None:
    try:
        stat = path.stat()
    except (FileNotFoundError, OSError):
        return None
    return stat if stat.st_mtime <= cutoff else None


def flatten_typed_numbers(value: Any, prefix: str = "$") -> dict[str, list[tuple[float, bool]]]:
    fields: dict[str, list[tuple[float, bool]]] = defaultdict(list)

    def visit(item: Any, path: str) -> None:
        if isinstance(item, bool) or item is None:
            return
        if isinstance(item, (int, float)):
            if isinstance(item, float) and not math.isfinite(item):
                return
            fields[path].append((float(item), isinstance(item, float)))
        elif isinstance(item, dict):
            for key, child in sorted(item.items(), key=lambda pair: str(pair[0])):
                normalized_key = re.sub(r"(?i)(?:^|[_-])s[0-2](?:$|[_-])", "_{seed}_", str(key))
                visit(child, f"{path}.{normalized_key}")
        elif isinstance(item, list):
            for child in item:
                visit(child, f"{path}[]")

    visit(value, prefix)
    return dict(fields)


def flatten_numbers(value: Any, prefix: str = "$") -> dict[str, list[float]]:
    return {
        path: [number for number, _is_float in values]
        for path, values in flatten_typed_numbers(value, prefix).items()
    }


def is_primary_metric(path: str) -> bool:
    leaf = path.rsplit(".", 1)[-1].replace("[]", "")
    return bool(METRIC_RE.search(leaf)) and not bool(CONFIG_RE.search(leaf))


def numeric_payload(data: Any) -> dict[str, list[float]]:
    return {
        path: values
        for path, values in flatten_numbers(data).items()
        if not SEED_KEY_RE.search(path.rsplit(".", 1)[-1]) and not CONFIG_RE.search(path.rsplit(".", 1)[-1].replace("[]", ""))
    }


def same_numeric_payload(records: list[JsonRecord]) -> bool:
    payloads = [numeric_payload(record.data) for record in records]
    return bool(payloads[0]) and all(payload == payloads[0] for payload in payloads[1:])


def identical_metric_arrays(records: list[JsonRecord]) -> list[str]:
    flat = [flatten_numbers(record.data) for record in records]
    common = set.intersection(*(set(item) for item in flat)) if flat else set()
    identical: list[str] = []
    for path in sorted(common):
        arrays = [item[path] for item in flat]
        if not is_primary_metric(path) or len(arrays[0]) < 2:
            continue
        if all(array == arrays[0] for array in arrays[1:]):
            identical.append(path)
    return identical


def round_anchor_findings(record: JsonRecord) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fields = flatten_typed_numbers(record.data)
    anchors: list[dict[str, Any]] = []
    round_fields: list[dict[str, Any]] = []
    for field, typed_values in fields.items():
        values = [value for value, _is_float in typed_values]
        if TIMING_COST_RE.search(field):
            for value in sorted(set(values) & ROUND_ANCHORS):
                anchors.append({"field": field, "value": value})
        float_values = [value for value, is_float in typed_values if is_float]
        if len(float_values) >= 2:
            rounded = sum(value != 0.0 and value % 100.0 == 0.0 for value in float_values)
            if rounded / len(float_values) > 0.5:
                round_fields.append(
                    {
                        "field": field,
                        "n": len(float_values),
                        "round_hundreds": rounded,
                        "fraction": rounded / len(float_values),
                        "values": sorted(set(float_values))[:12],
                    }
                )
    return anchors, round_fields


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scan(root: Path, recent_seconds: int) -> dict[str, Any]:
    root = root.resolve()
    started = time.time()
    cutoff = started - recent_seconds
    errors: list[dict[str, str]] = []
    skipped_recent: list[str] = []
    records: list[JsonRecord] = []

    for path in root.rglob("*.json"):
        if skipped_path(path, root):
            continue
        stat = stable_stat(path, cutoff)
        if stat is None:
            try:
                if path.exists() and path.stat().st_mtime > cutoff:
                    skipped_recent.append(str(path.relative_to(root)))
            except OSError:
                pass
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            errors.append({"file": str(path.relative_to(root)), "error": f"{type(exc).__name__}: {exc}"})
            continue
        rel = str(path.relative_to(root))
        seed = seed_from_path(rel)
        if seed is None:
            seed = seed_from_data(data)
        records.append(
            JsonRecord(
                path=path,
                rel=rel,
                tree=tree_for(path, root),
                mtime_second=int(stat.st_mtime),
                kind=classify(rel, seed),
                seed=seed,
                data=data,
            )
        )

    batch_groups: dict[tuple[str, int], list[JsonRecord]] = defaultdict(list)
    for record in records:
        batch_groups[(record.tree, record.mtime_second)].append(record)
    batch_write = []
    for (tree, second), group in sorted(batch_groups.items()):
        if len(group) < 2:
            continue
        kinds = {record.kind for record in group}
        classification = next(iter(kinds)) if len(kinds) == 1 else "mixed"
        batch_write.append(
            {
                "tree": tree,
                "mtime_second": second,
                "classification": classification,
                "files": [record.rel for record in sorted(group, key=lambda item: item.rel)],
            }
        )

    seed_groups: dict[tuple[str, str], list[JsonRecord]] = defaultdict(list)
    for record in records:
        if record.seed is not None:
            seed_groups[(record.tree, canonical_seed_path(record.rel))].append(record)
    zero_variance = []
    seed_groups_examined = 0
    for (tree, group_key), group in sorted(seed_groups.items()):
        distinct = {record.seed for record in group}
        if len(distinct) < 2:
            continue
        seed_groups_examined += 1
        group = sorted(group, key=lambda item: (item.seed if item.seed is not None else -1, item.rel))
        identical_payload = same_numeric_payload(group)
        identical_arrays = identical_metric_arrays(group)
        if identical_payload or identical_arrays:
            zero_variance.append(
                {
                    "tree": tree,
                    "group": group_key,
                    "seeds": sorted(distinct),
                    "files": [record.rel for record in group],
                    "identical_numeric_payload": identical_payload,
                    "identical_primary_metric_arrays": identical_arrays,
                }
            )

    round_anchors = []
    round_hundreds = []
    for record in records:
        anchors, rounded = round_anchor_findings(record)
        if anchors:
            round_anchors.append({"tree": record.tree, "file": record.rel, "matches": anchors})
        if rounded:
            round_hundreds.append({"tree": record.tree, "file": record.rel, "matches": rounded})

    matrix_root = root / "matrices"
    npz_groups: dict[str, list[tuple[Path, int]]] = defaultdict(list)
    npz_scanned = 0
    if matrix_root.is_dir():
        for path in matrix_root.rglob("*.npz"):
            if skipped_path(path, root):
                continue
            stat = stable_stat(path, cutoff)
            if stat is None:
                try:
                    if path.exists() and path.stat().st_mtime > cutoff:
                        skipped_recent.append(str(path.relative_to(root)))
                except OSError:
                    pass
                continue
            rel = str(path.relative_to(root))
            seed = seed_from_path(rel)
            if seed is not None:
                npz_groups[canonical_seed_path(rel)].append((path, seed))
            npz_scanned += 1
    duplicate_npz = []
    npz_groups_examined = 0
    for group_key, group in sorted(npz_groups.items()):
        if len({seed for _, seed in group}) < 2:
            continue
        npz_groups_examined += 1
        hashes: dict[str, list[str]] = defaultdict(list)
        for path, _seed in group:
            try:
                hashes[hash_file(path)].append(str(path.relative_to(root)))
            except OSError as exc:
                errors.append({"file": str(path.relative_to(root)), "error": f"{type(exc).__name__}: {exc}"})
        for digest, files in hashes.items():
            if len(files) >= 2:
                duplicate_npz.append({"group": group_key, "sha256": digest, "files": sorted(files)})

    tree_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"json": 0, "measurement": 0, "aggregate": 0, "unclassified": 0})
    for record in records:
        tree_counts[record.tree]["json"] += 1
        tree_counts[record.tree][record.kind] += 1

    return {
        "schema_version": "1.0",
        "root": str(root),
        "started_unix": started,
        "recent_seconds": recent_seconds,
        "skip_dirs": sorted(SKIP_DIRS),
        "summary": {
            "json_scanned": len(records),
            "json_errors": len(errors),
            "recent_files_skipped": len(set(skipped_recent)),
            "seed_groups_examined": seed_groups_examined,
            "npz_scanned_in_matrices": npz_scanned,
            "npz_seed_groups_examined": npz_groups_examined,
        },
        "tree_counts": dict(sorted(tree_counts.items())),
        "batch_write_clusters": batch_write,
        "zero_variance_seed_groups": zero_variance,
        "round_anchor_files": round_anchors,
        "round_hundreds_files": round_hundreds,
        "duplicate_seed_variant_npz": duplicate_npz,
        "skipped_recent": sorted(set(skipped_recent)),
        "errors": errors,
    }


def main() -> int:
    args = parse_args()
    if not args.root.is_dir():
        raise SystemExit(f"not a directory: {args.root}")
    result = scan(args.root, args.recent_seconds)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
