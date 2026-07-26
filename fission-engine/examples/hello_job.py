"""examples/hello_job.py — the canonical CPU smoke job.

Builds and enqueues a `gpu_required=false` job whose cmd writes
`results/hello.json` (relative to the job cwd = the fission-engine dir). Running
this then draining the queue proves the whole loop end-to-end without touching
the GPU — the pattern any branch copies to register real work.

Usage::

    python -m fission_engine.examples.hello_job         # enqueue it
    python -m fission_engine.runner --once              # drain (writes results/hello.json)
"""
from __future__ import annotations

import sys
from pathlib import Path

if __package__:
    from .. import queue as jobqueue
    from ..schema import JobSpec
else:  # pragma: no cover - direct-script fallback
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import queue as jobqueue  # type: ignore
    from schema import JobSpec  # type: ignore

PKG_DIR = Path(__file__).resolve().parents[1]  # the fission-engine dir

# A tiny, dependency-free python script that writes results/hello.json.
_HELLO_SRC = (
    "import json,os,time,pathlib;"
    "p=pathlib.Path('results/hello.json');"
    "p.parent.mkdir(parents=True,exist_ok=True);"
    "p.write_text(json.dumps({'hello':'fission','pid':os.getpid(),"
    "'ts':time.time(),'ok':True},indent=2));"
    "print('wrote',p.resolve())"
)


def make_hello_job() -> JobSpec:
    """Return the hello JobSpec (id is stable so re-enqueue is idempotent)."""
    return JobSpec(
        id="hello_cpu",
        branch="G",
        gpu_required=False,
        # run the same interpreter that enqueued us; CPU-only, needs only stdlib
        cmd=[sys.executable, "-c", _HELLO_SRC],
        # cwd relative to ROOT (workspace); the fission-engine dir keeps output local
        cwd=str(PKG_DIR),
        expect_outputs=["results/hello.json"],
        description="CPU smoke: write results/hello.json",
    )


def main() -> int:
    job = make_hello_job()
    # clear a stale copy so the demo is repeatable
    stale = jobqueue.QUEUE_DIR / f"{job.id}.json"
    if stale.exists():
        stale.unlink()
    out = (jobqueue.PKG_DIR / "results" / "hello.json")
    if out.exists():
        out.unlink()
    path = jobqueue.enqueue(job)
    print(f"[hello_job] enqueued {path}")
    print(f"[hello_job] now run:  python -m fission_engine.runner --once")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
