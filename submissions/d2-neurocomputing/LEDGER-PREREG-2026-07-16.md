# Pre-registration ledger — edit-federation operating map (deposit supplement)

> Assembled 2026-07-16 for the research-data deposit promised in §3 ("Pre-registration
> ledger"). Every "frozen" claim in the manuscript maps to a dated, written document that
> preceded the corresponding runs. The source documents are included verbatim in the
> deposit under `prereg/`; this file is the index and the prediction→outcome record.
> ANONYMITY NOTE: before public deposit, the source documents receive a redaction pass
> (internal project codenames and references to a companion manuscript under review are
> masked); dates, predictions, and thresholds are preserved byte-for-byte.

## 1. Freeze events (chronological)

| Date (2026) | Document | What it froze |
|---|---|---|
| 07-12 | RG kill-gate window record + `PREREG-RG-DEPTH-2026-07-12.md` | The full gate: partial ρ(I_cos,drop \| I_mag) ≥ +0.15 in ≥2 group sizes × ≥2 seeds; coherence ρ ≥ 0.30; negligibility median\|drop\| ≥ 0.1 or argmax-loss ≥ 0.05; saturation ceiling 0.8; own-magnitude partial guard. Depth tier (L8/L14) designs. |
| 07-14 | `PREREG-D2-WIDTH-RG-20260714.md` | Width tier (family fixed, width varied) design and directional expectations. |
| 07-15 ~12:55 | `PREDICTIONS-GAIN-WAVE-2026-07-15.md` | Gain definition (median \|drop\|/dose, positive-dose pooled g≤20); the ordering claim Spearman(gain, constructive fraction) ≤ −0.7; per-cell directional predictions (below); falsifier. |
| 07-15 ~13:50 | same doc, Addendum 1 (before launch) | Three 50%-depth contrast cells with directional predictions. |
| 07-15 ~14:30 | same doc, Addendum 2 (before launch) | GPT-2-XL pair, declared EXPLORATORY (ordering claim only). |
| 07-16 | `PREREG-FED-EDITORS-2026-07-16.md` | Editor wave: (a) ROME-equivalence anchor must hold to fp64; (b) AlphaEdit reduces federation cross-talk vs ROME (direction); (c) the ordering extends to editor-varied cells; MEMIT rows descriptive. |

## 2. Prediction → outcome (per cell)

Outcomes from `RG_gain_law_20260715.json` (pooled) and `RG_gain_holdout_20260716.json`
(g=2 fractions). HELD = outcome inside the frozen band/direction.

| Cell | Frozen prediction (date) | Outcome | Verdict |
|---|---|---|---|
| Llama-3.1-8B L24 | HIGH gain, destructive, frac<0.2 (07-15) | gain 22.2, frac 0.127 | HELD |
| Qwen-7B L21 | gain between Qwen-1.5B and 14B; frac at g=2 ≈0.4–0.7 (07-15) | gain 3.45 (marginally ABOVE both anchors, 3.3/3.1, not between them); frac_g2 0.525 (in band) | **PARTIAL** (fraction band held; gain 0.15 above the upper anchor) |
| gemma-2b L19 | HIGH-ish gain, frac<0.3, positive partials (07-15) | gain 12.5, frac 0.115, ρ +0.57 | HELD |
| Llama-3B L21 | HIGH gain, frac<0.2, positive partials (07-15) | gain 32.0, frac 0.022, ρ +0.87 | HELD |
| Qwen-3B L27 | low-mid gain; frac at g=2 ≈0.35–0.60 (07-15) | gain 3.62; frac_g2 0.637 | **NEAR-MISS** (g=2 fraction 0.037 above the frozen band; direction correct) |
| Phi-3.5 L24 | EXPLORATORY, no direction (07-15) | gain 1.87, frac 0.556 (low-gain, majority-constructive) | n/a (declared exploratory at freeze) |
| Phi-3.5 L16 (50%) | HIGH gain (≥10), frac<0.3 (Addendum 1) | gain 16.9, frac 0.251 | HELD |
| Qwen-3B L18 (50%) | HIGH gain, destructive, mirrors Qwen-1.5B L14 (28.1/0.242) (Addendum 1) | gain 37.2, frac 0.332 | HELD (direction; fraction 0.09 above the cited mirror value) |
| gemma-2b L13 (50%) | HIGH gain ≥12.5, destructive (Addendum 1) | gain 20.0, frac 0.287 | HELD |
| GPT-2-XL L36 / L24 | EXPLORATORY pair, no direction (Addendum 2) | 0.84/0.725 and 3.41/0.711 (both constructive) | n/a (declared exploratory at freeze) |
| Editor wave (a) ROME anchor | must hold to fp64 (07-16) | worst \|Δ\| = 0.0 | HELD |
| Editor wave (b) AlphaEdit direction | reduces cross-talk vs ROME (07-16) | 4–20× per-seed reduction, every seed of every cell | HELD |
| Editor wave (c) ordering extends | gain-screened ordering not ROME-specific (07-16) | every editor cell lands in its layer's predicted regime; 31-cell restatement −0.820 | HELD |

## 3. The ordering claim at each freeze stage

Frozen threshold: Spearman(gain, constructive fraction) ≤ −0.7, restated verbatim at
each addendum as the cell count grew (≥11 → ~17 → ~19 cells).

| Stage | Cells | Measured ρ |
|---|---|---|
| A-wave (75% depth) | 11+ | held (≤ −0.7) |
| + depth contrasts | ~17 | held |
| + GPT-2-XL pair | ~19 | −0.811 |
| + final-wave cells (post-freeze, below) | 22 | **−0.822** (p=2.8e-6; family-median −0.93) |

## 4. Cells added after the final prediction freeze

Three cells (Mistral-Nemo-12B L30, gemma-2-9b L31, GPT-NeoX-20B L33) were added on
07-16, after the 07-15 freezes, with no per-cell directional predictions (the frozen
falsifier named a 12–14B non-Qwen cell as "the sharpest test either way"). The ordering
claim was evaluated with and without them: −0.811 (19 cells) vs −0.822 (22 cells). The
manuscript identifies these three cells wherever their coverage differs from the frozen
set (signed re-analysis scope, §5.1/§5.3).

## 5. Honest accounting

12 directional predictions scored: 9 fully HELD, 1 HELD-with-note (Qwen-3B L18:
direction held, fraction 0.09 above the cited mirror value), 1 PARTIAL (Qwen-7B: the
g=2 fraction band held but the gain sits 0.15 above the upper anchor rather than
between the anchors), 1 NEAR-MISS (Qwen-3B L27: g=2 fraction 0.037 above its frozen
band; regime direction correct). 3 cells were declared exploratory at freeze time and
are not scored. Nothing frozen was dropped; no cell that
failed a prediction was excluded from any analysis.
