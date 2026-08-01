# Paper B honesty audit (2026-08-01)

Scope: read-only audit of `main.tex`; no manuscript edits. Claim values below resolve the macros in `macros.tex` only to report values already present in the canonical manuscript inputs.

## Part 1: honesty verdicts

### K1 / C2 narrow failure

**Verdict: PASS for honesty.** The manuscript never turns the narrow failure into a pass. Its principal numerical adjudications state the 0.85 threshold explicitly, while shorter captions retain the one-pass/one-fail verdict without repeating the number. It also says that only 1/9 broader NF4 full-model cells clear the threshold.

Sentences/claims quoting or adjudicating K1/C2:

1. `main.tex:56-60` reports the ROME ordering as 0.904 (Llama-1B), 0.680 (Llama-3B), and the Qwen value, then says only 1/9 cells clear the preregistered 0.85 threshold. **Verdict: honest; the 3B failure is not rounded into a pass.**
2. `main.tex:127-132` repeats the ordered full-model NF4 result and says only 1/9 cells clear the preregistered 4-bit threshold. **Verdict: honest.**
3. `main.tex:170-177` says the preregistered 4-bit full-model rank-survival threshold fails at two of three models and explicitly rejects narrowing the paper to the surviving cell. **Verdict: honest.**
4. `main.tex:254-267` states “fails at 4-bit full-model,” then identifies Llama-1B L12 as 0.904 PASS and Llama-3B L24 as 0.680 FAIL against 0.85; it also reports only 1/9 broader cells above threshold. **Verdict: strongest explicit PASS/FAIL statement; honest.**
5. `main.tex:490-498` says Llama-1B L12 clears at 0.904 while Llama-3B L24 falls below at 0.680 versus 0.85, then narrows C2 to INT8/edited-layer NF4 across both cells and full-model NF4 at Llama-1B only. **Verdict: honest.**
6. `main.tex:682-687` calls K1 a narrowed result and says one validated ROME NF4 full-model cell passes and one fails. **Verdict: honest, though numerical details remain in the surrounding figure/table.**
7. `main.tex:948-953` explains that K1 fires only if both validated cells fail; one failure narrows C2 rather than killing it. `main.tex:960` gives the K1 status through `\gateKoneDisplayStatus`, whose displayed manuscript wording is the narrowed verdict. **Verdict: honest.**
8. `main.tex:995-1002` says the threshold failed at two of three models and that focusing on the one passing cell would hide the finding. **Verdict: honest and explicit about preregistration divergence.**
9. `main.tex:1052-1056` again says only 1/9 cells clears 0.85 and that the preregistered gate failed. **Verdict: honest.**

**H-Llama width-law evidence check: FAIL / absent framing.** Neither “H-Llama” nor a width-law explanation of the old 1B-vs-Mistral confound occurs in `main.tex`; consequently, no evidence citation is attached to that framing. The manuscript instead warns that the observed ordering “is not a scale law” at `main.tex:61`. If the H-Llama framing is intended to remain a Paper B claim, the current manuscript does not carry it or cite its evidence.

### K3 and K3-prime

**Verdict: PASS.** K3 is presented as a failure on the measured axis, not as unadjudicated:

- `main.tex:159-168`: the competing concentration hypothesis is “dead” on the measurable axis.
- `main.tex:289-296`: C3 outcome is “killed,” with essentially all edits sub-bin-width.
- `main.tex:898-902`: the figure caption says the reconstruction ordering is descriptive and M-concentration is killed on the measured axis.
- `main.tex:906-914`: the section heading and argument call K3 a dead predictor, “not an inconclusive one.”
- `main.tex:916-928`: denominator limitations are disclosed, but the manuscript explicitly adjudicates K3 as “killed on the measured axis rather than deferring it”; the unmeasured amended channel-scale axis is separately identified.
- `main.tex:930-940`: K3-prime is limited to a descriptive codec property: `r_func < r_param` in all 18 scheme blocks, with no mechanism claim.
- `main.tex:948-963`: the gate table lists K3 as “KILLED (measured axis)” and K3-prime as “holds (18/18, descriptive).”

A stale source comment remains in `macros.tex:283-285` calling amended K3 `UNADJUDICATED`, but the rendered manuscript does not use that status for K3 and directly explains why measured-axis K3 is killed. This is source-maintenance debt, not a prose honesty regression.

### Preregistration divergence

**Verdict: PASS; the 2026-07-26 correction is intact.** `main.tex:972-983` labels the central base-noise mechanism post hoc, unregistered, and not prospectively tested. `main.tex:985-993` discloses that the DRAFT preregistration was not ratified before runs and separates prospectively fixed thresholds from chronologically unresolved protocol details. Most importantly, `main.tex:995-1002` preserves the exact strict reading: a failed gate plus an unregistered mechanism hypothesis for why it failed. Nothing later in the manuscript reverses that status; `main.tex:1055-1058` again pairs “the preregistered gate failed” with “the mechanism, offered post hoc.”

## Part 2: M-averaging mechanism deepening

Canonical source: `edit-harness/results/quant_survival/aggregate/gate_readout.json`, fields `cells.*.c3.{nf4dq,int8}.{r_func_mean,r_param_mean}`. The source has nine model/editor/layer cells and two schemes, giving 18 aggregate points; all 18 satisfy `r_func < r_param` descriptively.

### What can be said by layer

Only a **cross-model, layer-indexed breakdown** is available:

- ROME: L12 has the smallest `r_func` under both NF4 and INT8 among the three available ROME cells.
- AlphaEdit: L12 has the smallest `r_func` under both NF4 and INT8.
- MEMIT: L24 has the smallest `r_func` under both NF4 and INT8.

This is not evidence that those depths cause stronger functional cancellation. Layer is perfectly confounded with model: Llama-3.2-1B is observed only at L12, Qwen-2.5-1.5B only at L21, and Llama-3.2-3B only at L24. There is no within-model layer sweep in the existing canonical artifacts, so a genuine “functional cancellation by depth” conclusion is unavailable.

### Artifact needed for an actual depth test

Produce one canonical JSON containing repeated layers within each fixed model/editor/seed/scheme, with one row per observation and at least:

- `model`, `layer`, `editor`, `seed`, `scheme`;
- `r_func = ||epsilon x|| / ||Delta W x||`;
- `r_param = median|epsilon| / median|Delta W|`;
- the denominators or validity flags needed to reject non-finite/degenerate ratios;
- source cell identifiers and schema/provenance version.

At least three layers per model, with the same seeds and editors at every layer, are needed to separate depth from architecture. Preserving per-seed rows, rather than only means, is required for uncertainty intervals and paired layer contrasts.

### Figure specification

Script: `figures-src/fig05_functional_cancellation_by_depth.R`.

- Panel A: x = edited layer; y = `r_func` on a log scale; color = model; shape = editor; facets = NF4 and INT8. Each plotted mark is one canonical model/editor/layer/scheme aggregate point (18 total).
- Panel B: same encodings, with y = `r_func / r_param` on a log scale to display the descriptive cancellation gap directly.
- No lines connect layers because the x positions are different models, not a within-model sweep.
- Subtitle/caption must state: “one layer per model; cross-model layer-indexed view, not a within-model depth sweep.”
- The script reads only `gate_readout.json`, validates all 18 points and the 18/18 ordering, supports `--dry`, and adds `% SOURCE` headers to its tikz output.

## Part 3: F3 curve schema audit

Script: `figures-src/figF3_noise_signal_rank_survival.R`.

`paperb_curve_readout.py:12-14` reads new cells from `results/quant_survival_curve/` but writes `results/quant_survival/aggregate/curve_local_readout.json`. Its current output schema (`paperb_curve_readout.py:65-75`) contains status, six model means, Q1/Q2/Q3 gate fields, three new-family seed-survival arrays, and a note. Although lines 48-61 compute per-seed `(noise_to_signal, rank_survival)` pairs from NPZ arrays, those rows are discarded and never serialized.

Therefore the exact canonical JSON schema cannot currently support the requested curve: it has no per-point noise-to-signal x-values and no complete per-point model/seed metadata. The R skeleton performs `--dry` validation against the exact current schema when `curve_local_readout.json` exists (or when an explicit `--input` fixture/artifact is supplied), then reports the schema gap; before B1-B3 land, the default canonical input is absent. It does not invent values or a nonexistent JSON field. To unblock plotting while preserving the canonical-JSON-only figure rule, `paperb_curve_readout.py` must serialize the already-computed rows, including `model`, `layer`, `seed`, `noise_to_signal`, and `nf4_rank_survival`, into its canonical output under an explicitly versioned schema; the R script should then be updated to that real schema.
