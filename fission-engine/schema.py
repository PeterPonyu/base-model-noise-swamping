"""schema.py — the JobSpec / ResultRecord contract for the fission engine.

This is the *general* (workspace-level) engine's data contract. It is
deliberately editing-agnostic: a "job" is just a shell command that the engine
runs serially on the single GPU box, guarded by GPU-idle gating and validated
by expected output files. Any branch (B1..B6, or any future direction) registers
work by writing JobSpec JSON into ``queue/``.

Stdlib only — no torch/transformers import here, so this module loads instantly
and can be used from a CPU-only design/analysis/writing lane.

JobSpec (the contract)
----------------------
{
  "id":            "b1_mquake_hop2_s0",     # unique; auto-derived if omitted
  "branch":        "G",                       # lane/branch tag (G/D/A/W or B1..)
  "created":       "2026-06-30T22:00:00",    # ISO ts; auto-filled if omitted
  "gpu_required":  true,                       # true => wait for GPU idle first
  "cmd":           ["python3", "b1/sweep.py", "--hop", "2"],  # list OR string
  "cwd":           "b1",                        # optional; resolved vs ROOT
  "env":           {"HF_HUB_OFFLINE": "1"},    # optional per-job env overrides
  "expect_outputs":["results/b1_hop2.json"],   # files that must exist on success
  "timeout_s":     3000,                        # optional per-job wall clock
  "description":   "MQuAKE 2-hop consistency sweep, seed 0"
}

ResultRecord (written next to the moved job as <id>.log / <id>.result.json)
--------------------------------------------------------------------------
Captures rc, timing, which expected outputs were present/missing, and the log
path, so the analysis (A) lane can consume outcomes without re-parsing stdout.
"""
from __future__ import annotations

import datetime
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def _now_iso() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def _slug(text: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    return s.strip("_") or "job"


class JobSpecError(ValueError):
    """Raised when a queue job JSON does not satisfy the JobSpec contract."""


@dataclass
class JobSpec:
    """A single, self-contained unit of work for the serial GPU queue."""

    cmd: Union[List[str], str]
    id: str = ""
    branch: str = "G"
    created: str = field(default_factory=_now_iso)
    gpu_required: bool = False
    cwd: Optional[str] = None
    env: Dict[str, str] = field(default_factory=dict)
    expect_outputs: List[str] = field(default_factory=list)
    timeout_s: Optional[int] = None
    description: str = ""
    # populated by the loader so callers know where the job lives on disk
    _path: Optional[str] = None

    def __post_init__(self) -> None:
        self.validate()

    # -- validation ------------------------------------------------------
    def validate(self) -> "JobSpec":
        if not self.cmd:
            raise JobSpecError("JobSpec.cmd is required (list[str] or non-empty str)")
        if isinstance(self.cmd, list):
            if not all(isinstance(x, str) for x in self.cmd):
                raise JobSpecError("JobSpec.cmd list must contain only strings")
        elif not isinstance(self.cmd, str):
            raise JobSpecError("JobSpec.cmd must be a list[str] or a str")
        if not isinstance(self.gpu_required, bool):
            raise JobSpecError("JobSpec.gpu_required must be a bool")
        if not isinstance(self.expect_outputs, list) or not all(
            isinstance(x, str) for x in self.expect_outputs
        ):
            raise JobSpecError("JobSpec.expect_outputs must be a list[str]")
        if not isinstance(self.env, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in self.env.items()
        ):
            raise JobSpecError("JobSpec.env must be a dict[str, str]")
        if self.timeout_s is not None and (
            not isinstance(self.timeout_s, int) or self.timeout_s <= 0
        ):
            raise JobSpecError("JobSpec.timeout_s must be a positive int or null")
        if not self.id:
            self.id = self._auto_id()
        if not re.fullmatch(r"[A-Za-z0-9._-]+", self.id):
            raise JobSpecError(f"JobSpec.id has illegal characters: {self.id!r}")
        return self

    def _auto_id(self) -> str:
        base = self.description or (
            " ".join(self.cmd) if isinstance(self.cmd, list) else self.cmd
        )
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return f"{_slug(self.branch)}_{_slug(base)[:40]}_{stamp}"

    # -- (de)serialisation ----------------------------------------------
    def to_dict(self, include_path: bool = False) -> Dict[str, Any]:
        d = asdict(self)
        if not include_path:
            d.pop("_path", None)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "JobSpec":
        if not isinstance(d, dict):
            raise JobSpecError("job JSON must decode to an object/dict")
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        unknown = set(d) - known
        if unknown:
            raise JobSpecError(f"unknown JobSpec fields: {sorted(unknown)}")
        try:
            return cls(**d)
        except TypeError as e:  # missing required field (cmd) -> clean contract error
            raise JobSpecError(f"invalid JobSpec: {e}") from e

    # -- path resolution helpers ----------------------------------------
    def resolve_cwd(self, root: Path) -> Path:
        if not self.cwd:
            return root
        p = Path(self.cwd)
        return p if p.is_absolute() else (root / p)

    def output_paths(self, root: Path) -> List[Path]:
        base = self.resolve_cwd(root)
        out = []
        for o in self.expect_outputs:
            p = Path(o)
            out.append(p if p.is_absolute() else (base / p))
        return out

    def outputs_present(self, root: Path) -> bool:
        """True iff every declared expected output exists (all present)."""
        paths = self.output_paths(root)
        if not paths:
            return False  # no declared outputs => never 'already satisfied'
        return all(p.exists() for p in paths)

    def missing_outputs(self, root: Path) -> List[str]:
        return [str(p) for p in self.output_paths(root) if not p.exists()]


@dataclass
class ResultRecord:
    """Outcome of running one JobSpec. Serialised beside the moved job file."""

    id: str
    status: str  # "done" | "failed"
    returncode: Optional[int] = None
    reason: str = ""  # short machine-ish reason, e.g. "ok", "rc!=0", "gpu_wait_timeout"
    started: str = ""
    finished: str = ""
    duration_s: float = 0.0
    cmd: Union[List[str], str, None] = None
    cwd: str = ""
    outputs_present: List[str] = field(default_factory=list)
    outputs_missing: List[str] = field(default_factory=list)
    log_path: str = ""
    branch: str = "G"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def summary_line(self) -> str:
        return (
            f"[{self.status.upper()}] {self.id} "
            f"rc={self.returncode} reason={self.reason} "
            f"dur={self.duration_s:.1f}s "
            f"missing={len(self.outputs_missing)}"
        )


# --------------------------------------------------------------------------
# loader / validator helpers
# --------------------------------------------------------------------------
def load_job(path: Union[str, Path]) -> JobSpec:
    """Load and validate one queue job JSON, tagging it with its on-disk path."""
    path = Path(path)
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise JobSpecError(f"{path}: invalid JSON: {e}") from e
    job = JobSpec.from_dict(raw)
    job._path = str(path)
    return job


def validate_job_dict(d: Dict[str, Any]) -> JobSpec:
    """Validate a plain dict against the JobSpec contract (raises JobSpecError)."""
    return JobSpec.from_dict(d)


def dump_job(job: JobSpec, path: Union[str, Path]) -> Path:
    path = Path(path)
    path.write_text(json.dumps(job.to_dict(), indent=2))
    return path
