"""fission-engine — the general, workspace-level fission engine (the trunk).

A branch-agnostic serial GPU job scheduler distilled from ROADMAP.md §0-4:
the 5090 is the one serial resource, so all GPU work queues into ``queue/`` and
is drained one job at a time behind a GPU-idle gate, while design/analysis/
writing lanes run in parallel off-GPU.

This is a *general* engine and deliberately does NOT import or touch the
editing-specific ``edit-harness/`` engine, which remains live and independent.

Public surface:
    schema.JobSpec, schema.ResultRecord, schema.load_job
    gpuguard.is_gpu_idle, gpuguard.wait_for_gpu
    queue.enqueue, queue.list_pending, queue.mark_done, queue.mark_failed
    runner.main  (python -m fission_engine.runner ...)
"""
from __future__ import annotations

__all__ = ["schema", "gpuguard", "queue", "runner"]
__version__ = "0.1.0"

from . import gpuguard  # noqa: F401
from . import queue  # noqa: F401
from . import schema  # noqa: F401
