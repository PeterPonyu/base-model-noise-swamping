# PATCH (apply at MIX_A drain — run_stream.py is imported by the LIVE local wave)

## Defect (box smoke incident 2026-07-16)
`run_stream --smoke` writes `engine/SMOKE_PASS.ok` as soon as asserts (a)–(d) fire,
BEFORE the micro-stream completes. A crash after the asserts (e.g. the peft
ModuleNotFoundError at the FT flush on box 36039) still leaves a valid marker, and the
driver would launch the real wave onto a broken environment. The marker must be written
only at clean smoke completion.

## Fix
In `experiments/frame_a/run_stream.py`, locate the smoke path where the marker is
written (search `SMOKE_PASS.ok`). Move the marker write to AFTER the smoke stream's
final statement (end of the smoke branch, after all arms/flushes complete), i.e. the
last action before the smoke path returns/exits — and add a comment:
```python
# marker ONLY at clean completion: a crash after the asserts must NOT leave a valid
# marker (box incident 2026-07-16 — peft missing, flush crashed, marker already on disk)
```
Keep `--check_smoke_marker` semantics unchanged (model_dir + code checksum).

## Verification after applying
1. `python3 -m py_compile experiments/frame_a/run_stream.py`
2. `python3 -m experiments.frame_a.selftest` GREEN
3. Grep test: simulate by reading the code — the write must be the LAST statement of
   the smoke branch, after any `arms["ft"].flush` / final measurement.
4. Note: applying this changes the frame_a code checksum → next real-wave launch
   requires a fresh SMOKE=1 run (by design).

## RESOLVED-LOCALLY (verified 2026-07-18, hostile review)
The LOCAL `experiments/frame_a/run_stream.py` does NOT have this defect: `write_smoke_marker()`
is called exactly once (line ~351), as the LAST statement of `run_smoke` — after the
micro-stream, the restore probes, the FT flush, and the NaN tripwire. A crash at the FT flush
(the box incident) cannot leave a marker with this code. The defect existed on the BOX 36039
copy (older rsync). **Do NOT "apply" this patch locally** — a no-op reorder would still change
the frame_a code checksum and needlessly invalidate the fresh SMOKE_PASS.ok. If box 36039
restarts, check ITS copy of run_stream.py before trusting its marker.
