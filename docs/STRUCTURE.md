# Workspace structure rules (2026-07-23)

One page. If a new file doesn't have an obvious home under these rules, that's a design
smell — stop and decide, don't nest.

## Fixed top level (never grows sideways)
```
CLAUDE.md README.md PLAN.md env.sh          # the only durable root FILES
docs/          findings | plans | portfolio | archive   (+ this file)
edit-harness/  the B6 experiment harness (see below)
branches/      p2_prerl_diag | p3_agent_ipi | p4_temporal_uq  (self-contained each)
paper-arr/     shelved ARR draft (frozen at 11pp — re-fit to 8+2 gate before any revival)
submissions/   ieee | d2-federation | frame-a-eswa | paper-b-neurocomputing
               + lightweight kbs/tnnls extension-plan placeholders
fission-engine/  (+ fission_engine symlink — intentional, python import name)
.omc/          OMC session state — ROOT ONLY, see litter rules
.claude/       project-local Claude configuration (tool-managed)
.cursor/       project-local Cursor configuration (tool-managed)
```
A dated `_trash-YYYYMMDD/` may appear temporarily during a reviewed cleanup. It is not a
new archive or a durable workspace component.

## edit-harness internal map
- `editors/` one editor module per file (uniform apply_edit contract)
- `experiments/` flat, one concern per file; `experiments/tools/` for standalone checkers
- `engine/` runtime markers/logs/pids of the drivers
- `data/` datasets (flat files or one dir per dataset) + `data/models/<Name>/` (flat HF snapshot)
- `results/` canonical JSONs flat at top; `results/matrices/` npz; `results/smoke*/`
  driver-isolated smoke output (exempt from queue skip logic)
- `run_*.sh` at harness root = PENDING/ACTIVE queues only; completed one-shots move to
  `archive/drivers/` (grep for non-comment references first)
- `archive/` the ONLY archive inside the harness

## Depth cap
Durable, hand-authored files live at **≤4 levels** below the workspace root
(e.g. `submissions/ieee/sections/x.tex` = 3, fine). Anything hand-authored appearing at
5+ levels means a new nesting layer was invented — redesign instead of nesting.
Machine output (HF caches, npz shards) may go deeper but falls under litter rules.

## Litter rules (delete-on-sight after confirming no live writer)
- `.omc/` anywhere except workspace root — caused by launching agents with a subdir cwd;
  contains only transient session state. **Prevention: launch agents from workspace root.**
- `.pytest_cache/`, `__pycache__/`, and `.playwright-mcp/` — regenerable tool state; never archive.
- `data/**/.cache/` (HF download staging) — deletable once the artifact is sha256-verified.
- Root-level `.log`, `.aux`, `.spl`, and ad-hoc QA raster files — misplaced build output;
  paper-specific QA belongs under that submission's `figures-qa/`.
- Stale/superseded result artifacts: rename with a `-STALE` suffix only if a review record
  references them, else delete. Never leave a misleading number under its original name.

## Archives & quarantine
Two archives total: `docs/archive/`, `edit-harness/archive/`. No new archive dirs.
Bulk delete-candidates go through a dated root quarantine (`_trash-YYYYMMDD/`) before any
deletion. Every quarantine must contain `MANIFEST.tsv` with original paths and `RESTORE.md`
with exact rollback commands. Inspect and validate the workspace before a later explicit
`rm -rf`; the cleanup pass itself does not delete the quarantine.
