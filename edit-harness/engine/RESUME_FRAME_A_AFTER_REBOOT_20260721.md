# Frame-A BC real wave resume after shutdown

Paused at: 2026-07-22 08:31:32 local time (user-requested shutdown)

## Frozen state

- Driver PID before stop: `34368`
- Active MIX_B Python PID before stop: `47159`
- Stop method: `SIGTERM` to Python PID only; child and wrapper both exited within 1 second.
- Wrapper recorded the expected `ABORT: MIX_B failed rc=143`.
- Concurrent-session conflict during shutdown: the older active idea session `...23129` created `frame-a-bc-real-20260722.service` at 08:34:21 and, after that unit was stopped, created corrected `frame-a-bc-real-20260722-v2.service` at 08:37:08. Both reached only smoke/model load and wrote no target cell. Its completion/provenance watchers were stopped, both exact units were stopped and collected, and the conflicting session was interrupted back to an idle prompt so it could not relaunch again.
- Final stable state after a 15-second observation: both units inactive/not found, no matching user timer or shutdown inhibitor, no Frame-A runner/wrapper/provenance watcher, and GPU idle (2% utilization, about 779 MiB).
- Completed target cells: 18/66 total (`MIX_B` 18/33, `MIX_C` 0/33).
- Completed MIX_B cells: all 11 policies for seed 0; for seed 1, `both`, `cost_only`, `damage_only`, `oracle`, `always_edit`, `always_grace`, and `always_rag`.
- Interrupted cell: `MIX_B/always_ft/s1`; it was not present on disk and must rerun from the beginning.
- All 18 completed files parse under the project's NaN-tolerant JSON convention and declare `model=llama-3.2-1b`, `provenance=real`, and `mix=MIX_B`.
- No new Frame-A `tmp`, `partial`, or `lock` file was left by this interruption. The old quarantined `frame_a_verdict_llama-3.2-1b.json.INVALID-PARTIAL-20260716` is unrelated and remains untouched.
- GPU returned idle after stop (1% utilization; only desktop GPU processes remained).

## Resume guarantee

`experiments/frame_a/run_stream.py:393-414` skips an existing target JSON unless `--force` is used. Each missing cell first restores and verifies the base model, then writes its JSON only after replay and scoring finish. Relaunching the wrapper therefore preserves the 18 completed cells and fully reruns `MIX_B/always_ft/s1`.

## After power-on

Keep the laptop lid open. Check GPU health first:

```bash
cd ~/Desktop/idea-feasibility-analysis/edit-harness
nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader
```

Then resume exactly once using this single entrypoint (do not also create/start a transient systemd service for the same wrapper):

```bash
cd ~/Desktop/idea-feasibility-analysis/edit-harness
nohup ./run_frame_a_bc_real_20260721.sh > engine/run_frame_a_bc_real.nohup.log 2>&1 &
```

The wrapper reruns its smoke test, invokes MIX_B, skips the 18 completed files, resumes at `always_ft/s1`, and then runs the remaining MIX_B cells followed by MIX_C plus the namespaced P2 file.

Monitor by PID from `engine/run_frame_a_bc_real.pid`; never use `pgrep` or `pkill -f`.

## Post-wave provenance gate

Run this only after the wrapper exits successfully:

```bash
cd ~/Desktop/idea-feasibility-analysis/edit-harness
python3 -m experiments.frame_a.provenance_gate \
  --cells_dir results/frame_a/cells \
  --report results/frame_a/provenance_gate_real_bc.json
```

Trust MIX_B/C only if this returns `PASS` with exit code 0. The provenance watcher was not running at shutdown, so do not infer completion from a watcher marker; use the wrapper exit status and this fail-closed gate.
