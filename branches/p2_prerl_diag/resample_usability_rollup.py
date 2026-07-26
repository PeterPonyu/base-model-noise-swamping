#!/usr/bin/env python3
"""resample_usability_rollup.py — run_p2_resample.sh site-3 helper.

Extracted 2026-07-11 (post-drain fix) from the driver's inline
`$CONDA_RUN python3 - <<'EOF' ... EOF` heredoc, which was VACUOUS: `conda run
-n dl` swallows heredoc stdin (workspace memory conda-run-swallows-stdin.md),
so the heredoc body never executed and results/RESAMPLE_usability_20260711.json
was never written (the driver always fell into its "empty output" failure
branch). Logic below is byte-identical to the original heredoc body — only
the invocation mechanism changed (file + argv, no stdin). Must be run with
cwd == the workspace root (the driver `cd`s to $H before calling this), since
the glob below is relative, matching the original.
"""
import glob
import json
import os

rule = "USABLE iff n_right >= 20 AND (ci_hi - ci_lo) of D_within <= 1.5  [PREREG-P2-GRPO-20260710.md sec 2]"
out = {"rule": rule, "checkpoints": {}}
for f in sorted(glob.glob("branches/p2_prerl_diag/results/*_n[0-9]*.json")):
    d = json.load(open(f))
    dw = d.get("D_within", {})
    nr = d.get("counts", {}).get("n_right")
    ciw = None
    if dw.get("ci_hi") is not None and dw.get("ci_lo") is not None:
        ciw = dw["ci_hi"] - dw["ci_lo"]
    usable = (nr is not None and nr >= 20) and (ciw is not None and ciw <= 1.5)
    out["checkpoints"][d.get("checkpoint_id", os.path.basename(f))] = {
        "file": f, "n_right": nr, "D_within": dw.get("point"),
        "ci_width": round(ciw, 4) if ciw is not None else None, "USABLE": bool(usable)}
print(json.dumps(out, indent=1))
