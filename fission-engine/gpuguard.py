"""gpuguard.py — the single-GPU idle gate.

The whole scheduling premise (ROADMAP §0, §3): the 5090 is the *only* serial
resource. Before a gpu_required job runs, we must confirm no other job is
actually WORKING the GPU.

LESSON BURNED IN (2026-07-01, cost ~8h of idle GPU): do NOT gate on
"zero compute apps" (`nvidia-smi --query-compute-apps`). Long-lived services
(mcp_litchron ~1.1GB, ollama, etc.) hold a PERMANENT CUDA context that never
clears, so a zero-apps gate jams forever even though the card is idle. The
correct gate — the one `run_deep_until1900.sh` uses — is:

    utilization.gpu < UTIL_MAX  AND  memory.used < MEM_MAX_MIB

which tolerates small resident contexts while still blocking on a real job
(any training/edit job pins util near 100 and holds multi-GB memory).
`compute_app_pids()` is kept as a DIAGNOSTIC (who is resident), not a gate.
"""
from __future__ import annotations

import shutil
import subprocess
import time
from typing import List, Optional, Tuple

NVIDIA_SMI = "nvidia-smi"
QUERY_APPS_ARGS = ["--query-compute-apps=pid", "--format=csv,noheader"]
QUERY_LOAD_ARGS = [
    "--query-gpu=utilization.gpu,memory.used",
    "--format=csv,noheader,nounits",
]

# Idle thresholds (match run_deep_until1900.sh). MEM_MAX_MIB must stay above the
# sum of persistent service contexts (litchron ~1.1GB) but far below any real job.
UTIL_MAX = 10        # percent
MEM_MAX_MIB = 1500   # MiB


def _run_smi(args: List[str], timeout_s: float) -> str:
    if shutil.which(NVIDIA_SMI) is None:
        raise FileNotFoundError("nvidia-smi not found on PATH")
    try:
        out = subprocess.run(
            [NVIDIA_SMI, *args],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"nvidia-smi timed out after {timeout_s}s") from e
    if out.returncode != 0:
        raise RuntimeError(
            f"nvidia-smi failed (rc={out.returncode}): {out.stderr.strip()}"
        )
    return out.stdout


def compute_app_pids(timeout_s: float = 15.0) -> List[int]:
    """PIDs currently holding a CUDA compute context. DIAGNOSTIC ONLY — a
    non-empty list does NOT mean busy (persistent service contexts linger)."""
    pids: List[int] = []
    for line in _run_smi(QUERY_APPS_ARGS, timeout_s).splitlines():
        line = line.strip()
        if not line:
            continue
        # rows may be just "1234" or occasionally "[N/A]" — keep only integers
        try:
            pids.append(int(line.split(",")[0].strip()))
        except ValueError:
            continue
    return pids


def gpu_load(timeout_s: float = 15.0) -> Tuple[Optional[int], Optional[int]]:
    """(utilization_percent, memory_used_mib) for GPU 0; (None, None) if unparseable."""
    first = ""
    for line in _run_smi(QUERY_LOAD_ARGS, timeout_s).splitlines():
        if line.strip():
            first = line.strip()
            break
    parts = [p.strip() for p in first.split(",")]
    try:
        return int(float(parts[0])), int(float(parts[1]))
    except (ValueError, IndexError):
        return None, None


def is_gpu_idle(timeout_s: float = 15.0) -> bool:
    """True iff util < UTIL_MAX and memory.used < MEM_MAX_MIB.

    CPU-only host (no nvidia-smi) => True. An unparseable nvidia-smi reading
    => False (fail closed: never start a GPU job on an unknown-state card)."""
    try:
        util, mem = gpu_load(timeout_s=timeout_s)
    except FileNotFoundError:
        # No GPU tooling at all: nothing can be contending, so treat as idle.
        return True
    if util is None or mem is None:
        return False
    return util < UTIL_MAX and mem < MEM_MAX_MIB


def wait_for_gpu(
    poll_s: float = 30.0,
    max_wait_s: Optional[float] = 6 * 3600,
    on_wait=None,
    consecutive: int = 3,
) -> bool:
    """Block until the GPU is idle for ``consecutive`` polls in a row (guards
    against sampling the gap between a job's phases).

    Returns True once idle. Returns False if ``max_wait_s`` elapses first
    (``max_wait_s=None`` waits indefinitely). ``on_wait(waited_s, pids)`` is an
    optional callback invoked each poll while still busy (used for logging)."""
    waited = 0.0
    streak = 0
    while True:
        if is_gpu_idle():
            streak += 1
            if streak >= max(1, consecutive):
                return True
        else:
            streak = 0
        if max_wait_s is not None and waited >= max_wait_s:
            return False
        if streak == 0 and on_wait is not None:
            try:
                pids = compute_app_pids()
            except Exception:
                pids = []
            on_wait(waited, pids)
        time.sleep(poll_s)
        waited += poll_s


if __name__ == "__main__":  # tiny CLI: `python gpuguard.py`
    try:
        util, mem = gpu_load()
        pids = compute_app_pids()
        print(f"util={util}% mem={mem}MiB compute_app_pids={pids}")
        print("GPU is IDLE" if is_gpu_idle() else "GPU is BUSY")
    except FileNotFoundError:
        print("nvidia-smi absent -> reporting IDLE (CPU-only host)")
