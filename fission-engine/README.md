# fission-engine — the general (workspace-level) fission engine

The **trunk that hosts all branches**. A branch-agnostic, single-GPU job
scheduler distilled from [`../ROADMAP.md`](../ROADMAP.md) §0–4.

> First principle (ROADMAP §0): the 5090 is the *one* serial resource. So all
> GPU work queues into `queue/` and is drained **one job at a time** behind a
> GPU-idle gate, while design / analysis / writing lanes run in parallel
> off-GPU. `fission speed = GPU utilization × branch parallelism`.

This engine is **general** and deliberately does **not** import or interfere
with the editing-specific `../edit-harness/` engine (which is live and
independent). Any direction — the knowledge-editing branches B1–B6, or a future
non-editing project — registers work here the same way.

---

## Architecture

```
fission-engine/                 (import name: fission_engine — symlink at repo root)
├── schema.py     JobSpec + ResultRecord dataclasses/validators; load/validate queue JSON
├── gpuguard.py   is_gpu_idle() / wait_for_gpu() via nvidia-smi compute-apps
├── queue.py      enqueue / list_pending / mark_done / mark_failed (the spooler)
├── runner.py     the serial G-lane loop: gate -> run -> validate -> done|failed
├── examples/
│   └── hello_job.py   CPU smoke job maker (writes results/hello.json)
├── queue/
│   ├── <id>.json      pending jobs
│   ├── done/          <id>.json + <id>.log + <id>.result.json
│   └── failed/        <id>.json + <id>.log + <id>.result.json
└── results/           job outputs (declared per-job via expect_outputs)
```

- **ROOT** — the directory job `cmd`s run from. Defaults to the workspace root
  (parent of `fission-engine/`); override with `FISSION_ROOT`. A job may set its
  own `cwd` (resolved against ROOT). `expect_outputs` are resolved against the
  job's effective cwd.
- **Serial guarantee** — the runner is a single-threaded for-loop *and* holds an
  exclusive lock (`queue/.runner.lock`), so two jobs never contend the 24 GB GPU
  (which would OOM an 8B fp16 edit — ROADMAP §3.4). **Not a daemon**: it never
  backgrounds or self-launches. You invoke it (or a cron/systemd unit you write).
- **Idempotent / restartable** — "pending" = job JSON still at the top of
  `queue/` **and** its `expect_outputs` are not all present. A job whose outputs
  already exist is skipped, so re-running `--once` after a crash is safe.

### The JobSpec contract (`schema.py`)

```jsonc
{
  "id":            "b1_mquake_hop2_s0",              // unique; auto-derived if omitted
  "branch":        "G",                                // lane/branch tag (G/D/A/W or B1..)
  "created":       "2026-06-30T22:00:00",             // ISO ts; auto-filled if omitted
  "gpu_required":  true,                                // true => wait for GPU idle first
  "cmd":           ["python3", "b1/sweep.py", "--hop", "2"],  // list (exec) OR string (shell)
  "cwd":           "b1",                                // optional; resolved vs ROOT
  "env":           {"HF_HUB_OFFLINE": "1"},            // optional per-job env overrides
  "expect_outputs":["results/b1_hop2.json"],           // must all exist for success
  "timeout_s":     3000,                                // optional per-job wall clock
  "description":   "MQuAKE 2-hop consistency sweep"
}
```

A **ResultRecord** (`<id>.result.json`, plus a human `<id>.log`) is written
beside every moved job: `status`, `returncode`, `reason`, timing,
`outputs_present` / `outputs_missing`, and the log path — so the A-lane consumes
outcomes structurally without re-parsing stdout.

---

## The 4-lane cadence (ROADMAP §4)

> At any moment the GPU runs **branch N** while you/agents design **N+1**,
> analyze **N−1**, and write **N−2**. GPU never waits on a human; humans never
> wait on the GPU.

| Lane | Does | Who | When | Touches GPU? |
|---|---|---|---|---|
| **G** (GPU, serial) | drains `queue/` sweeps | this engine (`runner.py`) | 24/7, mainly overnight | **yes (serial)** |
| **D** (design) | write next batch of `JobSpec`s, prep data, ablation tables | you + subagent | daytime | no |
| **A** (analysis) | parse `queue/done/*.result.json`, make tables/plots | scientist subagent | after harvest | no |
| **W** (writing) | turn matured analysis into paper sections | writer subagent | while GPU runs | no |

**Daily loop**

```
morning:  harvest overnight G results -> A lane (dispatch scientist agent)
midday:   D lane designs the next batch of JobSpecs + W lane writes matured branches
evening:  enqueue the new JobSpecs -> start the overnight G drain
```

Because D/A/W never touch the GPU, six branches flow through different stages at
once on a single card: analyze B1's last batch, design B2's sequential loop, and
draft B6's theory — all while B1's next sweep runs on the GPU.

---

## How a branch registers jobs

A branch is just a producer of `JobSpec`s. Three equivalent ways:

**1. Python (recommended — validated + auto-id):**
```python
from fission_engine import queue
queue.enqueue({
    "branch": "B1",
    "gpu_required": True,
    "cmd": ["python3", "b1/mquake_sweep.py", "--hop", "2", "--seed", "0"],
    "cwd": "b1",
    "expect_outputs": ["results/b1_mquake_hop2_s0.json"],
    "description": "MQuAKE 2-hop consistency, seed 0",
})
```
Loop over `method × model × dataset × seed × hop-depth` to enqueue a whole
overnight sweep (ROADMAP §3.5: seed×3 fills the queue and pins GPU utilization).

**2. Drop a JSON file** straight into `queue/<id>.json` (matching the contract).

**3. Copy the example maker** `examples/hello_job.py` and adapt `cmd` /
`expect_outputs`.

Then drain:
```bash
# from the repo root (source ../env.sh first: unsets ALL_PROXY, activates dl)
python -m fission_engine.runner --once     # drain current pending, exit (cron-friendly)
python -m fission_engine.runner --watch    # loop, sleeping between empty polls
python -m fission_engine.runner --job b1_mquake_hop2_s0   # run one job only
python -m fission_engine.runner --once --dry-run          # skip GPU wait; still runs cmds
```

Runner flags: `--once` | `--watch` (mutually exclusive; default `--once`),
`--dry-run` (skip the GPU-idle gate, still run cmds — CPU-friendly), `--job ID`,
`--poll-s` (GPU poll interval), `--max-wait-s` (cap per job; `<=0` = wait
forever), `--idle-sleep` (empty-poll sleep in `--watch`).

---

## Quick self-test (CPU only, no GPU needed)

```bash
python -m fission_engine.examples.hello_job   # enqueue the CPU hello job
python -m fission_engine.runner --once        # drain it -> writes results/hello.json
cat fission-engine/results/hello.json
ls  fission-engine/queue/done/                # hello_cpu.{json,log,result.json}

python fission-engine/gpuguard.py             # prints GPU IDLE / BUSY right now
```

> Import name is `fission_engine` (underscore) via a symlink at the repo root to
> the `fission-engine/` directory, so `python -m fission_engine.…` works while
> keeping the on-disk name from the ROADMAP.

---

## Operations & gotchas

Hard-won lessons; read before scripting around the engine.

### (a) Wait on external processes by PID — never `pgrep -f <script-name>`

To block a follow-up job until another process finishes, poll its **PID**:

```bash
while kill -0 "$PID" 2>/dev/null; do sleep 60; done   # correct
```

Do **not** gate on `pgrep -f serial_drain.py` (or any script name): a watcher's
own command line *contains that string*, so `pgrep -f` matches **itself** and the
wait never ends — a self-match deadlock. This bit us this session.

Likewise the engine's GPU gate (`gpuguard.py`) deliberately asks **nvidia-smi**
for compute-app PIDs (`--query-compute-apps=pid`), never `pgrep`. Occupancy is a
property of the GPU, not of a process name. **Keep it that way** — do not
"simplify" gpuguard to a `pgrep`/name match.

### (b) How a branch registers work into the central queue

The engine only accepts the `JobSpec` contract (see above). Branch-native
formats — P4's per-file `{id, env:"dl", expect_result, …}` objects and P3's
`queue.json` **list** of `{run_id, out, …}` — are **rejected** by
`JobSpec.validate` (it errors on unknown fields). Two supported paths:

1. **Convert** with `collect_branch_jobs.py` (COPIES; originals untouched):
   ```bash
   # default: both known branches -> the central engine queue
   python -m fission_engine.collect_branch_jobs
   # explicit sources -> an alternate queue (idempotent; skips already-satisfied)
   python -m fission_engine.collect_branch_jobs --queue-dir /path/to/queue \
       --source /…/p4_temporal_uq/fission-engine/queue/p4_etth2_20260630_2359.json \
       --source /…/p3_agent_ipi/jobs/queue.json
   ```
   It maps `run_id→id`, `expect_result`/`out→expect_outputs` (1-element list),
   carries `cwd` (P3 defaults to the branch dir), keeps `cmd`, carries
   `gpu_required`/`created`, and **drops** `env:"dl"` (the cmds already
   `conda run -n dl …`; engine `env` must be a `dict[str,str]` or omitted).

2. **Write engine-format `JobSpec` JSON** directly (via `queue.enqueue(...)` or a
   file dropped into the queue dir). Do **not** hand the engine the old ad-hoc
   P4/P3 formats.

> ⚠️ Do not `collect` branch jobs into the **central** queue while a separate
> orchestrator (e.g. `serial_drain.py`) is already driving those same branch
> specs directly — that double-runs them. Collect into a scratch `--queue-dir`
> for verification instead.

### (c) Pointing the engine at another queue — `FISSION_QUEUE_DIR` / `--queue-dir`

`queue.QUEUE_DIR` defaults to `fission-engine/queue/` but honors the env var
`FISSION_QUEUE_DIR` (mirroring `FISSION_ROOT` for `ROOT`); `done/` and `failed/`
always derive from it. The runner also takes `--queue-dir DIR`, which **wins**
over the env var and recomputes the runner lock (`.runner.lock`) and `runner.log`
under that dir:

```bash
FISSION_QUEUE_DIR=/tmp/scratch_q python -m fission_engine.queue            # list its pending
FISSION_QUEUE_DIR=/tmp/scratch_q python -m fission_engine.runner --once --dry-run
python -m fission_engine.runner --queue-dir /tmp/scratch_q --once --dry-run  # flag form
```

Every queue helper (`enqueue`, `list_pending`, `get_job`, `mark_done`,
`mark_failed`) takes an optional `queue_dir=` that overrides the module default,
so a lane can drive an alternate queue without mutating globals.

> ⚠️ `--dry-run` skips only the **GPU-idle wait**, not execution: without
> `--job <id>`, a bare `--once --dry-run` **runs every pending cmd**. On a queue
> holding real GPU jobs that will actually launch them — scope with `--job` (or a
> CPU-only scratch queue) when you only mean to smoke-test the plumbing.

### (d) The stdlib-`queue` shadow

This package's module is named `queue.py`, which **shadows** Python's stdlib
`queue` for code inside the package. That's intentional and harmless here (the
engine never needs stdlib `queue`), but it's why `python -m fission_engine.queue`
prints a benign `RuntimeWarning: 'fission_engine.queue' found in sys.modules …`
— a `runpy` artifact of running a submodule by name, not an error. If you ever
need the stdlib queue from within this package, import it as
`import queue as _stdlib_queue` only after ensuring `sys.path` order won't
re-resolve to this file, or rename accordingly.
