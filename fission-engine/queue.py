"""queue.py — the on-disk GPU job queue (the "printer spooler" of ROADMAP §3).

Layout (all under the fission-engine package dir)::

    queue/
      <id>.json          # pending jobs live here at the top level
      done/<id>.json     # succeeded jobs are moved here + <id>.log/<id>.result.json
      failed/<id>.json   # failed jobs moved here + <id>.log/<id>.result.json

"Pending" == the job JSON is still at the top level of queue/ AND its declared
``expect_outputs`` are not all already present (idempotent / restartable: a job
whose outputs already exist is considered satisfied and is skipped).

This module is stdlib-only and does not import the runner, so design/analysis
lanes can enqueue and inspect the queue without loading any GPU stack.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Union

# absolute/robust import so this works both as `-m fission_engine.queue` and as a
# plain script on sys.path.
try:
    from . import schema as _schema
except ImportError:  # pragma: no cover - direct-script fallback
    import schema as _schema  # type: ignore

JobSpec = _schema.JobSpec
ResultRecord = _schema.ResultRecord

PKG_DIR = Path(__file__).resolve().parent
# ROOT = the workspace root that job cmds run from (parent of fission-engine).
# Override with FISSION_ROOT if you relocate the engine.
import os as _os

ROOT = Path(_os.environ.get("FISSION_ROOT", str(PKG_DIR.parent))).resolve()
# QUEUE_DIR honors FISSION_QUEUE_DIR (mirrors FISSION_ROOT above) so an analysis
# or branch lane can point the spooler at an alternate queue without editing
# code; DONE_DIR/FAILED_DIR always derive from it.
QUEUE_DIR = Path(_os.environ.get("FISSION_QUEUE_DIR", str(PKG_DIR / "queue"))).resolve()
DONE_DIR = QUEUE_DIR / "done"
FAILED_DIR = QUEUE_DIR / "failed"


def _dirs_for(queue_dir: Optional[Path] = None):
    """Return (queue, done, failed) dirs for ``queue_dir``.

    ``queue_dir=None`` yields the module globals (which already honor
    FISSION_QUEUE_DIR); an explicit path is resolved and its done/failed
    subdirs derived. This is the single place every helper resolves dirs, so a
    caller-supplied ``queue_dir`` overrides the global consistently."""
    if queue_dir is None:
        return QUEUE_DIR, DONE_DIR, FAILED_DIR
    qd = Path(queue_dir).resolve()
    return qd, qd / "done", qd / "failed"


def _ensure_dirs(queue_dir: Optional[Path] = None) -> None:
    for d in _dirs_for(queue_dir):
        d.mkdir(parents=True, exist_ok=True)


def enqueue(job: Union[Dict, JobSpec], queue_dir: Optional[Path] = None) -> Path:
    """Validate and write a job to ``<queue_dir>/<id>.json``. Returns the path.

    Accepts either a plain dict (validated against the JobSpec contract) or an
    already-built JobSpec. Refuses to clobber an existing pending job id.
    ``queue_dir=None`` uses the module default (honoring FISSION_QUEUE_DIR)."""
    _ensure_dirs(queue_dir)
    qd, _, _ = _dirs_for(queue_dir)
    spec = job if isinstance(job, JobSpec) else JobSpec.from_dict(job)
    path = qd / f"{spec.id}.json"
    if path.exists():
        raise FileExistsError(f"job id already queued: {path}")
    path.write_text(json.dumps(spec.to_dict(), indent=2))
    spec._path = str(path)
    return path


def _top_level_jobs(queue_dir: Optional[Path] = None) -> List[JobSpec]:
    _ensure_dirs(queue_dir)
    qd, _, _ = _dirs_for(queue_dir)
    jobs: List[JobSpec] = []
    for p in sorted(qd.glob("*.json")):
        jobs.append(_schema.load_job(p))
    return jobs


def list_pending(queue_dir: Optional[Path] = None, root: Path = ROOT) -> List[JobSpec]:
    """Return top-level jobs whose expected outputs are not all present yet,
    sorted by ``created`` (oldest first) so the queue drains FIFO."""
    pending = [j for j in _top_level_jobs(queue_dir) if not j.outputs_present(root)]
    pending.sort(key=lambda j: (j.created, j.id))
    return pending


def list_satisfied(queue_dir: Optional[Path] = None, root: Path = ROOT) -> List[JobSpec]:
    """Top-level jobs already satisfied (outputs present) — candidates to sweep
    into done/ without re-running."""
    return [j for j in _top_level_jobs(queue_dir) if j.outputs_present(root)]


def get_job(job_id: str, queue_dir: Optional[Path] = None) -> Optional[JobSpec]:
    qd, _, _ = _dirs_for(queue_dir)
    p = qd / f"{job_id}.json"
    return _schema.load_job(p) if p.exists() else None


def _move_with_log(
    job: JobSpec,
    dest_dir: Path,
    result: ResultRecord,
    log_text: str = "",
    queue_dir: Optional[Path] = None,
) -> Path:
    _ensure_dirs(queue_dir)
    qd, _, _ = _dirs_for(queue_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    src = Path(job._path) if job._path else (qd / f"{job.id}.json")
    dest_json = dest_dir / f"{job.id}.json"
    # write a sibling .log (human-readable) and .result.json (structured)
    log_path = dest_dir / f"{job.id}.log"
    result.log_path = str(log_path)
    log_body = result.summary_line() + "\n"
    if log_text:
        log_body += "\n----- captured output -----\n" + log_text
    log_path.write_text(log_body)
    (dest_dir / f"{job.id}.result.json").write_text(
        json.dumps(result.to_dict(), indent=2)
    )
    if src.exists():
        shutil.move(str(src), str(dest_json))
    else:  # job file already gone (e.g. re-run); still persist its json snapshot
        dest_json.write_text(json.dumps(job.to_dict(), indent=2))
    return dest_json


def mark_done(
    job: JobSpec, result: ResultRecord, log_text: str = "",
    queue_dir: Optional[Path] = None,
) -> Path:
    _, done_dir, _ = _dirs_for(queue_dir)
    result.status = "done"
    return _move_with_log(job, done_dir, result, log_text, queue_dir)


def mark_failed(
    job: JobSpec, result: ResultRecord, log_text: str = "",
    queue_dir: Optional[Path] = None,
) -> Path:
    _, _, failed_dir = _dirs_for(queue_dir)
    result.status = "failed"
    return _move_with_log(job, failed_dir, result, log_text, queue_dir)


if __name__ == "__main__":  # tiny CLI: `python queue.py` -> print pending
    _ensure_dirs()
    print(f"ROOT      = {ROOT}")
    print(f"QUEUE_DIR = {QUEUE_DIR}")
    ps = list_pending()
    print(f"pending ({len(ps)}):")
    for j in ps:
        gp = "GPU" if j.gpu_required else "cpu"
        print(f"  - {j.id}  [{gp}] branch={j.branch} created={j.created}")
