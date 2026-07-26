# P3 wave-3 lineage-arm download manifest — 2026-07-11 — NOT PULLED, ask-first per standing rule

Companion to `PREREG-WAVE3-LINEAGE-DRAFT-20260711.md` (DRAFT, not frozen) and
`grid.py`'s new `EXTENDED_DESIGN` "Group D" rows / `TIERS["lineage_arm"]`. None of the
4 commands below have been run. Per this workspace's standing rule ("ask before large
network downloads" — unchanged by the 2026-07-10 policy shift, which only lifted the
Ollama CPU-pin and the all-local/GPU-serial rule, not the download-approval gate), these
need explicit user go-ahead before `ollama pull`.

## Pull commands

```
ollama pull hermes3:8b
ollama pull dolphin3:8b
ollama pull tulu3:8b
ollama pull openthinker:7b
```

**Anchor model — NO download needed.** The `lineage_arm` tier also uses
`llama3.1:8b-instruct-q8_0` as the `Llama3.1/large` cluster anchor, but that model is ALREADY
local (present in `~/.ollama/models/manifests`, measured ASR ~0.10 in wave-1/2). Wave 3 only
**relabels** it in-group for this tier (`grid.py` `TIER_OVERRIDES["lineage_arm"]`, tier-local
— its global out-group row is untouched). So the 4 pulls above are the ONLY new bytes; the
anchor de-risks the cluster (3 of the 6 Llama arch pairs involve a known-attackable model)
without any additional download.

## Size / disk

Est. ~5GB each (q4/q5-class default Ollama library quantization for 7-8B models) —
**~20GB total**. Current free space on `/` (where `~/.ollama/models` lives):
**417GB free of 1.9TB** (checked 2026-07-11), so headroom is not a concern.

## Risk note: `openthinker:7b` may be structurally degenerate (design hazard §8b)

`openthinker:7b` is a reasoning-distill fine-tune (Bespoke-Stratos-style, trained on
DeepSeek-R1 reasoning traces) — the same category of model as `deepseek-r1:*`, which the
existing roster already documents as NOT reliably advertising Ollama `tools` capability
(`grid.py`'s `_mock_tools_hint`: `lineage == "r1-distill"` -> `supports_tools_hint=False`).
If `openthinker:7b`'s Ollama build behaves the same way, its real (live, `ollama show`-derived)
`supports_tools` will resolve `False` or it will error out and get nulled by
`ERROR_RATE_THRESHOLD=0.2` (`run_ipi.py:51`) — i.e. it could reproduce exactly the
structural-0 problem this wave-3 arm exists to fix, just relocated to the `Qwen2.5/mid`
pair instead of fixed.

**This is a known, accepted risk, not a blocker** — because:
1. The design is NOT solely dependent on `openthinker:7b`. The 3 Llama3.1/large pairs
   (hermes3 x dolphin3, hermes3 x tulu3, dolphin3 x tulu3) are all plain instruction-tuned
   chat fine-tunes (not reasoning distills) and are expected to be attackable on the same
   basis as the existing attackable Llama/Qwen base-instruct models in the roster.
2. **CORRECTED 2026-07-11 (hostile-review MAJOR-1):** the claim that a degenerate
   `openthinker:7b` "will simply fail to contribute a surviving pair rather than silently
   corrupting the headline number" was FALSE as originally written. `openthinker` is a
   SINGLETON in-group lineage (its only member). Under `run_ipi.py::_gate_contrast` as it
   shipped, ANY in-group lineage — including a singleton — losing all its members to
   `ERROR_RATE_THRESHOLD=0.2` (run_ipi.py:51) suppresses the ENTIRE seed's contrast
   (`contrast=None`), not just its own pair. A degenerate `openthinker:7b` would take the
   6 healthy Llama3.1/large pairs down with it, not merely fail to contribute its own.
   This still fails SAFE — it never emits a silently-wrong number — but it is not
   "stands alone." A 2026-07-11 fix adds an opt-in `--allow_singleton_lineage_drop` flag
   (default OFF, `run_grid.py` / `make_jobs.py --kind grid`) that lets a dead singleton
   lineage be dropped instead of suppressing the whole contrast, gated on (i) only
   singleton lineage(s) being lost, never a multi-member one, and (ii) an attackable
   (ASR>0 both sides) architecture pair still surviving the drop. See
   `PREREG-WAVE3-LINEAGE-DRAFT-20260711.md` sec 3a for exact semantics; using it for the
   real launch is a conscious per-run choice, not a default.
3. If `openthinker:7b` IS degenerate, the Qwen2.5/mid cross-family replication drops. The
   Llama3.1/large arch-pair result (the primary fix target) stands on its own ONLY if
   `--allow_singleton_lineage_drop` is explicitly set for that launch — WITHOUT the flag,
   openthinker's degeneracy suppresses the entire seed's contrast, Llama pairs included.
   State whichever launch mode was actually used explicitly in any writeup, not glossed
   over.

## Post-pull verification (before trusting anything downstream)

1. `ollama show hermes3:8b` / `dolphin3:8b` / `tulu3:8b` / `openthinker:7b` — confirm each
   resolves and check advertised capabilities (does `tools` appear?).
2. Re-run `grid.py --tier lineage_arm --backend ollama` (LIVE daemon, still metadata-only —
   no inference) and diff `supports_tools` against the `mock`-backend hints already recorded
   in this build's offline verification, to catch any hint/reality mismatch before the real
   sweep.
3. Only after (1)-(2) look reasonable: launch per
   `PREREG-WAVE3-LINEAGE-DRAFT-20260711.md` sec 4.
