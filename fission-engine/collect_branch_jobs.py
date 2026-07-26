"""collect_branch_jobs.py — convert branch-native job specs into engine JobSpecs.

Branches historically emit jobs in ad-hoc formats that ``schema.JobSpec`` (which
rejects unknown fields) will NOT accept as-is:

  * **P4** — one JSON *object per file* with keys::
        id, created, gpu_required, env="dl"(string), cwd, config,
        cmd(string), expect_result(SINGULAR string), dataset, backend, notes
  * **P3** — one ``queue.json`` holding a *list* of objects with keys::
        run_id, backend, gpu_required, created, cmd(list), out, status
  * **P2** — one JSON *object per file* (queue/*.json) with keys::
        job_kind("gen"|"diag"), job_id, gpu_required, created, checkpoint_id,
        env="dl"(string), cmd(string, root-relative paths), _note, depends_on,
        spec{..., out}(gen only; diag output is results/<checkpoint_id>.json)

This collector DISCOVERS those specs (an explicit list of paths/globs, defaulting
to the two known branch locations), CONVERTS each into a valid engine JobSpec
dict, and WRITES one ``<id>.json`` per job into a queue dir. It only ever READS
the sources — originals are left byte-for-byte untouched (it COPIES).

Field mapping (branch-native -> engine JobSpec)::

    run_id / id            -> id
    expect_result / out    -> expect_outputs   (wrapped as a 1-element list)
    cwd                    -> cwd              (P4: carried; P3: default branch dir)
    cmd (list or str)      -> cmd              (kept as-is)
    gpu_required           -> gpu_required
    created                -> created          (carried when present)
    notes                  -> description       (P4 only; nice-to-have)
    env="dl" (string)      -> DROPPED           (cmds already `conda run -n dl ...`;
                                                 engine env must be a dict/omitted)
    config/dataset/backend/status/...          -> DROPPED (not JobSpec fields)

Idempotent: a job is skipped when ``<queue_dir>/<id>.json`` already exists OR its
``expect_outputs`` are all already present on disk (already satisfied).

Usage::

    # default: collect both known branches into the CENTRAL engine queue
    python -m fission_engine.collect_branch_jobs

    # collect explicit sources into an alternate queue (verification / scratch)
    python -m fission_engine.collect_branch_jobs \
        --queue-dir /path/to/scratch_queue \
        --source /.../p4_temporal_uq/fission-engine/queue/p4_etth1_20260630_2359.json \
        --source /.../p3_agent_ipi/jobs/queue.json

    python -m fission_engine.collect_branch_jobs --dry-run   # print, write nothing
"""
from __future__ import annotations

import argparse
import glob as _glob
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# robust import for both `python -m fission_engine.collect_branch_jobs` and
# `python collect_branch_jobs.py`
if __package__:
    from . import queue as jobqueue
    from . import schema
else:  # pragma: no cover - direct-script fallback
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import queue as jobqueue  # type: ignore
    import schema  # type: ignore

JobSpec = schema.JobSpec

# The two branch locations we default to. Each is (glob, branch-tag, default_cwd).
_ANALYSIS_ROOT = Path(__file__).resolve().parent.parent  # idea-feasibility-analysis/
_P4_BRANCH = _ANALYSIS_ROOT / "branches" / "p4_temporal_uq"
_P3_BRANCH = _ANALYSIS_ROOT / "branches" / "p3_agent_ipi"
_P2_BRANCH = _ANALYSIS_ROOT / "branches" / "p2_prerl_diag"


@dataclass
class Source:
    """One place branch specs live, plus how to tag/anchor its jobs."""

    pattern: str          # a path or glob (may match many files)
    branch: str           # engine branch tag applied to converted jobs
    default_cwd: str      # cwd for jobs that don't declare their own


DEFAULT_SOURCES: List[Source] = [
    Source(
        pattern=str(_P4_BRANCH / "fission-engine" / "queue" / "*.json"),
        branch="P4",
        default_cwd=str(_P4_BRANCH),
    ),
    Source(
        pattern=str(_P3_BRANCH / "jobs" / "queue.json"),
        branch="P3",
        default_cwd=str(_P3_BRANCH),
    ),
    Source(
        # P2 cmds use workspace-root-relative paths ("branches/p2_prerl_diag/...",
        # "edit-harness/data/models/..."), so the cwd anchor is the WORKSPACE ROOT.
        pattern=str(_P2_BRANCH / "queue" / "*.json"),
        branch="P2",
        default_cwd=str(_ANALYSIS_ROOT),
    ),
]


def _as_list(raw: Any) -> List[Dict[str, Any]]:
    """A source file is either a single job object (P4) or a list (P3)."""
    if isinstance(raw, list):
        return [r for r in raw if isinstance(r, dict)]
    if isinstance(raw, dict):
        return [raw]
    raise ValueError(f"unexpected JSON top-level type: {type(raw).__name__}")


def convert_one(raw: Dict[str, Any], source: Source) -> JobSpec:
    """Convert one branch-native job dict into a validated engine JobSpec.

    Raises schema.JobSpecError (via JobSpec validation) on anything malformed."""
    job_id = raw.get("id") or raw.get("run_id") or raw.get("job_id")
    if not job_id:
        raise ValueError("branch job has none of 'id' / 'run_id' / 'job_id'")

    cmd = raw.get("cmd")
    if not cmd:
        raise ValueError(f"branch job {job_id!r} has no 'cmd'")

    # expect_result (P4, singular) / out (P3, singular) / spec.out (P2 gen) -> 1-element list.
    expect = (raw.get("expect_result") or raw.get("out")
              or (raw.get("spec") or {}).get("out"))
    if not expect and raw.get("job_kind") == "diag":
        # P2 diag: run_diag.py names its output from the cmd's --id (override wins
        # over payload/filename), writing results/<id>.json. Derive from the cmd
        # itself so the expectation can never drift from what actually gets written;
        # checkpoint_id is only the fallback.
        cmd_str = cmd if isinstance(cmd, str) else " ".join(str(c) for c in cmd)
        m = re.search(r"--id\s+(\S+)", cmd_str)
        diag_id = m.group(1) if m else raw.get("checkpoint_id")
        if diag_id:
            expect = f"branches/p2_prerl_diag/results/{diag_id}.json"
    expect_outputs = [expect] if expect else []

    spec_dict: Dict[str, Any] = {
        "id": str(job_id),
        "branch": source.branch,
        "gpu_required": bool(raw.get("gpu_required", False)),
        "cmd": cmd,  # list or str, kept as-is
        "cwd": raw.get("cwd") or source.default_cwd,
        "expect_outputs": expect_outputs,
    }
    # carry created when present (preserves FIFO ordering intent); else JobSpec
    # auto-fills it.
    if raw.get("created"):
        spec_dict["created"] = str(raw["created"])
    # P4 carries prose in "notes", P2 in "_note" -> engine description (P3 has none).
    if raw.get("notes"):
        spec_dict["description"] = str(raw["notes"])
    elif raw.get("_note"):
        desc = str(raw["_note"])
        if raw.get("depends_on"):
            # the engine has no dependency edges — FIFO by created covers P2's
            # gen-before-diag ordering; carry the intent for humans.
            desc += f" [depends_on={raw['depends_on']}]"
        spec_dict["description"] = desc
    # NB: env="dl" (a string) is intentionally NOT carried — the cmds already
    # invoke `conda run -n dl`, and engine env must be a dict[str,str] or omitted.

    return JobSpec.from_dict(spec_dict)


def _discover(patterns: List[str], sources: List[Source]) -> List[tuple]:
    """Expand each source's glob into (path, source) pairs (sorted, stable)."""
    pairs: List[tuple] = []
    for src in sources:
        matched = sorted(_glob.glob(src.pattern))
        for m in matched:
            pairs.append((Path(m), src))
    return pairs


def collect(
    sources: Optional[List[Source]] = None,
    queue_dir: Optional[Union[str, Path]] = None,
    dry_run: bool = False,
    root: Optional[Path] = None,
) -> Dict[str, List[str]]:
    """Discover -> convert -> write. Returns a summary dict of ids per bucket.

    ``queue_dir=None`` targets the central engine queue (jobqueue.QUEUE_DIR).
    Idempotent: skips ids already queued or whose outputs already exist. Never
    mutates the source files."""
    sources = sources if sources is not None else DEFAULT_SOURCES
    root = root or jobqueue.ROOT
    qd, _, _ = jobqueue._dirs_for(Path(queue_dir) if queue_dir is not None else None)

    written: List[str] = []
    skipped_exists: List[str] = []
    skipped_satisfied: List[str] = []
    errors: List[str] = []

    for path, src in _discover([s.pattern for s in sources], sources):
        try:
            raw_json = json.loads(path.read_text())
            for raw in _as_list(raw_json):
                try:
                    spec = convert_one(raw, src)
                except Exception as e:  # one bad job never aborts the batch
                    errors.append(f"{path.name}: {e}")
                    continue
                target = qd / f"{spec.id}.json"
                if target.exists():
                    skipped_exists.append(spec.id)
                    continue
                if spec.outputs_present(root):
                    skipped_satisfied.append(spec.id)
                    continue
                if not dry_run:
                    qd.mkdir(parents=True, exist_ok=True)
                    target.write_text(json.dumps(spec.to_dict(), indent=2))
                written.append(spec.id)
        except Exception as e:
            errors.append(f"{path}: {e}")

    return {
        "queue_dir": [str(qd)],
        "written": written,
        "skipped_exists": skipped_exists,
        "skipped_satisfied": skipped_satisfied,
        "errors": errors,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--queue-dir", metavar="DIR", default=None,
        help="write converted JobSpecs here (default: central engine queue / "
             "$FISSION_QUEUE_DIR). Pass a scratch dir to verify without touching "
             "the central queue.",
    )
    ap.add_argument(
        "--source", metavar="PATH_OR_GLOB", action="append", default=None,
        help="a branch spec path/glob to convert (repeatable). Omit to use the "
             "two known branch locations (P4 queue/*.json + P3 jobs/queue.json).",
    )
    ap.add_argument(
        "--branch", default="B", metavar="TAG",
        help="branch tag for --source jobs that can't be auto-classified "
             "(default 'B'). Ignored for the default known sources.",
    )
    ap.add_argument("--dry-run", action="store_true",
                    help="convert + report but write nothing")
    args = ap.parse_args(argv)

    if args.source:
        # explicit sources: classify P4/P3 by path, else fall back to --branch.
        sources: List[Source] = []
        for pat in args.source:
            low = pat.lower()
            if "p4_temporal_uq" in low:
                sources.append(Source(pat, "P4", str(_P4_BRANCH)))
            elif "p3_agent_ipi" in low:
                sources.append(Source(pat, "P3", str(_P3_BRANCH)))
            elif "p2_prerl_diag" in low:
                sources.append(Source(pat, "P2", str(_ANALYSIS_ROOT)))
            else:
                sources.append(Source(pat, args.branch, str(Path(pat).resolve().parent)))
    else:
        sources = DEFAULT_SOURCES

    summary = collect(sources=sources, queue_dir=args.queue_dir, dry_run=args.dry_run)
    tag = "[dry-run] " if args.dry_run else ""
    print(f"{tag}queue_dir = {summary['queue_dir'][0]}")
    print(f"{tag}written           ({len(summary['written'])}): {summary['written']}")
    print(f"{tag}skipped_exists    ({len(summary['skipped_exists'])}): {summary['skipped_exists']}")
    print(f"{tag}skipped_satisfied ({len(summary['skipped_satisfied'])}): {summary['skipped_satisfied']}")
    if summary["errors"]:
        print(f"{tag}errors ({len(summary['errors'])}):")
        for e in summary["errors"]:
            print(f"    - {e}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
