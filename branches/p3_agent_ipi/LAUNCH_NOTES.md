# P3 B4 lineage sweep — launch notes

## Binding post-run condition (standing, applies to every future run of this sweep)
Any `results/ipi_*.json` produced by `run_ipi.py --backend ollama` MUST be passed through
`audit_unmatched.py` before any ASR/lineage/contrast number from it is quoted or trusted.
That script quantifies arm-specific parser false-negatives (concentrated on the prompt-format
/ r1-distill arm) that otherwise bias the lineage-vs-architecture contrast. Do not cite a
run's `contrast` field pre-audit.

## 2026-07-06 launch attempt — ABORTED, GPU-safety blocker (unresolved)

Attempted the reviewed launch (`python3 run_ipi.py --backend ollama --n 30`, per the
docstring's documented invocation) with Ollama started as:

    env CUDA_VISIBLE_DEVICES="" OLLAMA_NUM_GPU=0 nohup ollama serve

This is **insufficient to keep this Ollama build off the GPU.** Ollama 0.30.4 on this
machine defaults to a **Vulkan** backend (`OLLAMA_VULKAN:true` in its own logged server
config) which auto-discovers the RTX 5090 Laptop GPU via Vulkan (`Vulkan0 ... NVIDIA GeForce
RTX 5090 Laptop GPU`) independent of `CUDA_VISIBLE_DEVICES`. `OLLAMA_NUM_GPU=0` did not
prevent this either. `GGML_VK_VISIBLE_DEVICES` was present but empty in the server's env
dump and did NOT suppress Vulkan device discovery (unlike the CUDA convention, an empty
string here did not mean "no devices").

Sequence observed: ollama serve started clean (no GPU process) -> sweep launched -> first
scenario dispatched a real chat request -> the model loaded lazily and `llama-server`
offloaded 29/29 layers to the GPU (`load_tensors: offloaded 29/29 layers to GPU`) -> a new
GPU compute-app process appeared (`llama-server`, ~2 GiB) and `nvidia-smi` memory-used rose
from the pre-existing 11594 MiB (the unrelated `run_20260705_gapB_b1_muonlr.py` fleet) to
13656 MiB, GPU util to 95%.

**This means `run_ipi.py`'s own built-in guard, `_assert_ollama_not_on_gpu()`, is not
sufficient to catch this failure mode**: it checks `nvidia-smi --query-compute-apps` once at
the top of `run()`, before any request is sent — at that point ollama serve is up but has not
yet lazily loaded a model onto GPU, so the guard passes. The GPU allocation only happens on
the *first real inference call*, after the guard has already cleared the sweep to proceed.

Killed by PID immediately on detection (never pgrep/pkill by pattern, per the workspace
standing rule): sweep (`run_ipi.py`), `llama-server`, `ollama serve`, in that order. GPU
confirmed restored to baseline (6 pre-existing `python3` processes, ~11.6 GiB, no
ollama/llama-server compute-app) within seconds. No `results/*.json` was written (the kill
landed before the first model finished its first item, and `run_ipi.py` only writes its
output JSON once, at the very end of `run()`).

**Not resolved by this launch attempt — needs a decision before relaunching:**
a verified way to force this Ollama build fully off GPU (candidates, UNTESTED,
requires verification before use — do not assume): `OLLAMA_VULKAN=0` /
`OLLAMA_VULKAN=false` at `ollama serve` start, or an explicit invalid/negative
`GGML_VK_VISIBLE_DEVICES` value, or an Ollama version/flag that disables Vulkan discovery
outright. Whatever is chosen must be reverified with the same before/after
`nvidia-smi --query-compute-apps` check used here, through the point where the model
actually loads (not just server start).

## 2026-07-06 relaunch — SUCCESS (`OLLAMA_VULKAN=0`, verified post-inference + watchdog)

`OLLAMA_VULKAN=0` (the first untested candidate above) works. Started ollama as:

    env OLLAMA_VULKAN=0 CUDA_VISIBLE_DEVICES="" OLLAMA_NUM_GPU=0 nohup ollama serve

The server's own logged config confirmed `OLLAMA_VULKAN:false`. A probe inference
(`qwen2.5:1.5b`, tiny prompt via `/api/generate`) completed with NO offload log lines
(compare to the earlier failed attempt's `load_tensors: offloaded 29/29 layers to GPU`) and
`nvidia-smi --query-compute-apps` showed no new process, memory unchanged from baseline.

A watchdog (`logs/gpu_watchdog.sh`, new file, does not touch the reviewed sweep code) was
attached alongside the real sweep launch: polls `nvidia-smi --query-compute-apps` every 60s
for ollama/llama-server/llama-cpp; on a hit it kills the sweep + ollama server by PID and
appends an ABORT line to the sweep log, then self-exits. Confirmed quiet (no ABORT) through
7 completed real sweep items (~2.5 min wall) spanning multiple GPU checks, all clean.

Observed throughput on the smallest model (`deepseek-r1:1.5b`, first in the design panel,
CPU-only q-something inference, ~6 tok/s generation): roughly 1 item per 25-30s wall clock.
Larger models in the 9-model panel (7B/8B/9B, several q8) will be substantially slower per
item than this floor — do not extrapolate the 270-item (9 models x 30 scenarios) total from
the 1.5B rate alone; expect multi-hour wall time and monitor rather than assume a fixed ETA.

Live PIDs (see pidfiles under `logs/`): `ollama serve` 2440220 (`logs/ollama_b.pid`),
sweep 2448156 (`logs/sweep_b.pid`), watchdog 2448159 (`logs/watchdog.pid`). Logs:
`logs/ollama_serve_20260706b.log`, `logs/p3_sweep_20260706b.log`,
`logs/watchdog_20260706b.log`. Output lands at `results/ipi_<timestamp>.json` when `run()`
completes — remember the binding audit condition above before trusting it.
