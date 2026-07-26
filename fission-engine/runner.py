"""runner.py — the serial GPU consumer (G-lane of the fission engine).

The trunk that hosts all branches. It drains ``queue/`` one job at a time,
NEVER concurrently (single GPU), gating gpu_required jobs on GPU idleness:

  1. list pending jobs, sorted by ``created`` (FIFO),
  2. for each: if gpu_required and GPU busy -> wait_for_gpu(poll, max_wait),
  3. run the job's cmd via subprocess from ROOT (or the job's cwd),
  4. success == rc==0 AND every expect_outputs file exists -> queue/done/,
     otherwise -> queue/failed/ (both with a sibling .log + .result.json),
  5. repeat.

Flags
-----
  --once      drain the currently-pending set, then exit (default cadence for
              "morning harvest / evening enqueue" driving from cron or by hand)
  --watch     loop forever, sleeping --idle-sleep between empty polls
  --dry-run   skip the GPU-idle wait entirely but still run the cmd as given
              (CPU jobs run normally; GPU jobs are attempted without gating)
  --job <id>  run only the one job with this id, then exit
  --poll-s / --max-wait-s   GPU-idle poll interval and cap
  --idle-sleep              seconds to sleep between empty polls in --watch

A file lock (queue/.runner.lock) guarantees only ONE runner touches the GPU at a
time even across processes. This is NOT a daemon: nothing here backgrounds or
self-launches; you invoke it (or a cron/systemd unit you write) explicitly.
"""
from __future__ import annotations

import argparse
import datetime
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

# robust import for both `python -m fission_engine.runner` and `python runner.py`
if __package__:
    from . import gpuguard
    from . import queue as jobqueue
    from . import schema
else:  # pragma: no cover - direct-script fallback
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import gpuguard  # type: ignore
    import queue as jobqueue  # type: ignore
    import schema  # type: ignore

ROOT = jobqueue.ROOT
QUEUE_DIR = jobqueue.QUEUE_DIR
LOCK_PATH = QUEUE_DIR / ".runner.lock"
RUN_LOG = QUEUE_DIR / "runner.log"


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def log(msg: str) -> None:
    line = f"[{_now()}] {msg}"
    print(line, flush=True)
    try:
        with open(RUN_LOG, "a") as f:
            f.write(line + "\n")
    except OSError:
        pass


# --------------------------------------------------------------------------
# single-runner lock (prevents two runners contending the GPU concurrently)
# --------------------------------------------------------------------------
class RunnerLock:
    def __init__(self, path: Path):
        self.path = path
        self.fd: Optional[int] = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            other = ""
            try:
                other = self.path.read_text().strip()
            except OSError:
                pass
            raise SystemExit(
                f"[runner] another runner holds the lock ({self.path}); "
                f"contents={other!r}. Refusing to run a second GPU consumer."
            )
        os.write(self.fd, f"pid={os.getpid()} started={_now()}\n".encode())
        return self

    def __exit__(self, *exc):
        if self.fd is not None:
            os.close(self.fd)
        try:
            self.path.unlink()
        except OSError:
            pass


# --------------------------------------------------------------------------
# job execution
# --------------------------------------------------------------------------
def _run_cmd(job: schema.JobSpec):
    cwd = job.resolve_cwd(ROOT)
    cwd.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    # ALL_PROXY(socks) breaks HF httpx (see env.sh); scrub it for every job.
    for k in ("ALL_PROXY", "all_proxy"):
        env.pop(k, None)
    env.update(job.env)
    shell = isinstance(job.cmd, str)
    proc = subprocess.run(
        job.cmd,
        cwd=str(cwd),
        env=env,
        shell=shell,
        capture_output=True,
        text=True,
        timeout=job.timeout_s,
    )
    return proc


def process_job(job: schema.JobSpec, dry_run: bool, poll_s: float, max_wait_s: Optional[float],
                queue_dir: Optional[Path] = None) -> str:
    """Run one job to completion. Returns 'done' | 'failed'. Never concurrent.

    ``queue_dir`` (when set) is where the job's done/failed record is written,
    so the runner can drain a queue other than the package default."""
    started = _now()
    t0 = time.time()

    # --- GPU gate ---------------------------------------------------------
    if job.gpu_required and not dry_run:
        if not gpuguard.is_gpu_idle():
            log(f"gpu busy; waiting (poll={poll_s}s max_wait={max_wait_s}s) for {job.id}")

            def _on_wait(waited, pids):
                log(f"  ...still busy after {waited:.0f}s (pids={pids}) — {job.id}")

            if not gpuguard.wait_for_gpu(poll_s=poll_s, max_wait_s=max_wait_s, on_wait=_on_wait):
                rec = schema.ResultRecord(
                    id=job.id, status="failed", reason="gpu_wait_timeout",
                    started=started, finished=_now(), duration_s=time.time() - t0,
                    cmd=job.cmd, cwd=str(job.resolve_cwd(ROOT)),
                    outputs_missing=job.missing_outputs(ROOT), branch=job.branch,
                )
                jobqueue.mark_failed(job, rec, log_text="GPU never became idle within max_wait_s",
                                     queue_dir=queue_dir)
                log(f"FAILED (gpu wait timeout): {job.id}")
                return "failed"
    elif job.gpu_required and dry_run:
        log(f"[dry-run] skipping GPU-idle wait for gpu_required job {job.id}")

    # --- run --------------------------------------------------------------
    log(f"run: {job.id}  cmd={job.cmd}")
    try:
        proc = _run_cmd(job)
        rc: Optional[int] = proc.returncode
        out_text = (proc.stdout or "") + (proc.stderr or "")
    except subprocess.TimeoutExpired as e:
        rc = None
        out_text = f"TIMEOUT after {job.timeout_s}s\n{e}"
    except FileNotFoundError as e:
        rc = 127
        out_text = f"command not found: {e}"

    present = [str(p) for p in job.output_paths(ROOT) if p.exists()]
    missing = job.missing_outputs(ROOT)
    ok = (rc == 0) and (not missing)
    reason = "ok" if ok else ("rc!=0" if rc != 0 else "missing_outputs")

    rec = schema.ResultRecord(
        id=job.id, status="done" if ok else "failed", returncode=rc, reason=reason,
        started=started, finished=_now(), duration_s=time.time() - t0,
        cmd=job.cmd, cwd=str(job.resolve_cwd(ROOT)),
        outputs_present=present, outputs_missing=missing, branch=job.branch,
    )
    if ok:
        jobqueue.mark_done(job, rec, log_text=out_text, queue_dir=queue_dir)
        log(f"done: {job.id} ({rec.duration_s:.1f}s)")
        return "done"
    jobqueue.mark_failed(job, rec, log_text=out_text, queue_dir=queue_dir)
    log(f"FAILED: {job.id} (rc={rc}, reason={reason}, missing={missing})")
    return "failed"


# --------------------------------------------------------------------------
# main loop
# --------------------------------------------------------------------------
def _drain_once(args, queue_dir: Optional[Path] = None) -> int:
    pending: List[schema.JobSpec] = jobqueue.list_pending(queue_dir=queue_dir)
    if args.job:
        pending = [j for j in pending if j.id == args.job]
        if not pending:
            # maybe it exists but its outputs already satisfied, or not found
            existing = jobqueue.get_job(args.job, queue_dir=queue_dir)
            if existing is None:
                log(f"--job {args.job}: not found in queue/")
            else:
                log(f"--job {args.job}: outputs already satisfied; nothing to run")
            return 0
    if not pending:
        return 0
    n = 0
    for job in pending:
        process_job(job, args.dry_run, args.poll_s, args.max_wait_s, queue_dir=queue_dir)
        n += 1
        if args.job:  # single-job mode stops after the one
            break
    return n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="fission engine serial GPU runner")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true",
                      help="drain the currently-pending set, then exit")
    mode.add_argument("--watch", action="store_true",
                      help="loop forever, sleeping between empty polls")
    ap.add_argument("--dry-run", action="store_true",
                    help="skip GPU-idle wait but still run cmds (CPU-friendly)")
    ap.add_argument("--job", metavar="ID", help="run only this one job id, then exit")
    ap.add_argument("--queue-dir", metavar="DIR", default=None,
                    help="drain this queue dir instead of the package default "
                         "(else $FISSION_QUEUE_DIR, else fission-engine/queue). "
                         "The runner lock and run log are recomputed under it.")
    ap.add_argument("--poll-s", type=float, default=30.0,
                    help="GPU-idle poll interval (default 30)")
    ap.add_argument("--max-wait-s", type=float, default=6 * 3600,
                    help="cap on GPU-idle wait per job (default 6h; <=0 => wait forever)")
    ap.add_argument("--idle-sleep", type=float, default=60.0,
                    help="seconds between empty polls in --watch (default 60)")
    args = ap.parse_args(argv)

    if args.max_wait_s is not None and args.max_wait_s <= 0:
        args.max_wait_s = None  # wait indefinitely

    # default mode: --once if neither --once nor --watch given
    if not args.once and not args.watch:
        args.once = True

    # Resolve the effective queue dir: --queue-dir (CLI) wins, else the queue
    # module default (which already honors $FISSION_QUEUE_DIR). Rebind the
    # module globals so the lock + run log live under the queue we actually
    # drain, and thread the same dir into every queue call.
    global QUEUE_DIR, LOCK_PATH, RUN_LOG
    QUEUE_DIR = Path(args.queue_dir).resolve() if args.queue_dir else jobqueue.QUEUE_DIR
    LOCK_PATH = QUEUE_DIR / ".runner.lock"
    RUN_LOG = QUEUE_DIR / "runner.log"
    queue_dir = QUEUE_DIR

    with RunnerLock(LOCK_PATH):
        log(f"=== fission runner start (ROOT={ROOT}, QUEUE_DIR={QUEUE_DIR}, "
            f"mode={'watch' if args.watch else 'once'}"
            f"{', dry-run' if args.dry_run else ''}) ===")
        if args.watch:
            try:
                while True:
                    ran = _drain_once(args, queue_dir)
                    if ran == 0:
                        time.sleep(args.idle_sleep)
                    if args.job:  # --job + --watch: one shot then stop
                        break
            except KeyboardInterrupt:
                log("interrupted — exiting watch loop")
        else:
            ran = _drain_once(args, queue_dir)
            log(f"=== drained {ran} job(s); exiting (--once) ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
