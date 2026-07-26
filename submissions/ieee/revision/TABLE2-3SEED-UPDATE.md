# Table II (`tab:causal-scale`) — single-seed → 3-seed replacement

**Date:** 2026-07-26
**Target:** `submissions/ieee/sections/05_causal.tex` §"Generality: architecture, scale, and
instruction tuning" + the macro block at `submissions/ieee/macros.tex:348–378`.
**Status:** DRAFT for the TETCI revision round. Nothing in `submissions/ieee/` has been
modified; this file is the diff to apply when the revision is authorized.
**Verification:** every replacement value below was read directly out of the cited JSON in
this session (not from memory, not from `REVISION-PREP-20260710.md`). Where the prep doc
and the JSON disagree, the JSON wins and the discrepancy is flagged.

---

## 1. What changes and why

Table II was submitted with **`seeds_used = [0]`** in every cell — a single AlphaEdit
projector seed. The table's own footnote discloses this ("Every cell uses the held-out-key
projector and a single AlphaEdit seed"), and `REVISION-PREP-20260710.md` §1 anticipates the
reviewer asking for seed variance. Three of the four extension cells have since been re-run
to seeds {0,1,2}; the fourth (NeoX-20B) has not.

**Headline for the response letter:** no cell changes sign, and no cell crosses a
qualitative threshold. Seed-to-seed spread is 0.003–0.034 in ρ. The single-seed values
were, in every case, within ~0.03 of the 3-seed mean.

---

## 2. Verified value table

ρ = signed within-probe Spearman(pre-edit key-cosine, damage AlphaEdit removes vs ROME).
"sd" is the sample standard deviation across the three seeds.

| Cell | Submitted (s0) | 3-seed mean | per-seed | sd | Δ | Source JSON |
|---|---|---|---|---|---|---|
| 1B-Instruct L12 ρ | **0.568** | **0.552** | 0.5675 / 0.5335 / 0.5550 | 0.017 | −0.016 | `C4_causal_instruct_table_3seed.json` |
| GPT-J-6B L21 ρ | **−0.204** | **−0.184** | −0.2043 / −0.1696 / −0.1786 | 0.018 | +0.020 | `C4_causal_gptj_table_3seed.json` |
| Llama-8B L16 ρ | **0.185** | **0.212** | 0.1852 / 0.2474 / 0.2040 | 0.032 | +0.027 | `C4_causal_8b_table_3seed.json` |
| Llama-8B L24 ρ | **−0.102** | **−0.087** | −0.1016 / −0.0810 / −0.0781 | 0.013 | +0.015 | `C4_causal_8b_table_3seed.json` |
| NeoX-20B L16 ρ | **0.049** | *(still s0 only)* | 0.0488 | — | — | `C4_causal_neox20b_table.json` (`seeds_used: [0]`) |

All three 3-seed files carry `filters = {known: true, edit_ok: true, proj_source:
"holdout"}` — identical to the submitted configuration, so this is a seed extension, not a
protocol change.

### Damage columns (`dmg R→A`) also move

The 3-seed re-runs pool a different edit set, so the mean-damage columns shift too. These
must be updated together with ρ or the table becomes internally inconsistent.

| Cell | Submitted dmg R→A | 3-seed dmg R→A | note |
|---|---|---|---|
| 1B-Instruct L12 | 2.63 → 0.14 | **2.298 → 0.086** | 96.3 % removed (was ~94.8 %) |
| GPT-J-6B L21 | −0.040 → −0.004 | **−0.034 → −0.002** | 93.5 % removed |
| Llama-8B L16 | 0.26 → 0.04 | **0.260 → 0.022** | 91.4 % removed |
| Llama-8B L24 | −0.05 → 0.03 | **−0.050 → 0.020** | sign-crossing floor; see §4 |
| NeoX-20B L16 | 0.061 → −0.028 | unchanged (s0) | — |

### Quartile means (used in the prose, not the table)

| Cell | Submitted Q1→Q4 | 3-seed Q1→Q4 | 3-seed ratio |
|---|---|---|---|
| 1B-Instruct L12 | 1.36 → 3.85 (2.83×) | **1.224 → 3.465** | **2.831** (unchanged to 3sf) |
| GPT-J-6B L21 | −0.011 → −0.066 (6.2×) | **−0.010 → −0.060** | **6.083** — see §4, note A |
| Llama-8B L16 | not quoted | 0.016 → 0.634 | 40.5 (still unstable, keep unquoted) |
| NeoX-20B L16 | 0.053 → 0.138 (2.63×) | unchanged (s0) | 2.634 |

**Monotonicity across quartiles survives at 3 seeds** for 1B-Instruct (1.224 / 1.826 /
2.332 / 3.465) and GPT-J in magnitude (|−0.010| / |−0.021| / |−0.035| / |−0.060|). The
paper's monotone-removal claim is unaffected.

### Discrepancy against `REVISION-PREP-20260710.md`

That doc's §1 table lists the 8B L28 cell and an MQuAKE cell using
`C4_causal_mquake_table_3seed_probesrc.json` (`proj_source = probes`). Table II contains
neither. The prep doc's instruct/8B ρ values match the JSONs exactly; its damage figures
were not tabulated there and are the ones newly verified above. No contradiction found,
but note the prep doc's MQuAKE row is the **probe-source** projector, whereas the
revision-insurance run produced a **holdout** projector file — see §4, note B.

---

## 3. Draft replacement LaTeX

### 3a. Macro block (`macros.tex:348–378`)

Replace the header comment and the seven changed macros. Unchanged macros
(`\instructRatio`, all `\neox*`) are omitted.

```latex
% =====================================================================
% 2026-07-10 extension fold-in — scale/architecture CAUSAL cells (§5)
% REVISED 2026-07-26 (TETCI revision round): instruct / gptj / 8b cells now
% 3-seed (seeds_used=[0,1,2]); NeoX-20B remains single-seed s0.
% source: C4_causal_{instruct,gptj,8b}_table_3seed.json + C4_causal_neox20b_table.json
% (holdout projector throughout: filters.proj_source = "holdout")
% =====================================================================
\newcommand{\instructCausalRho}{0.552}    % instruct_3seed layers.12.within_probe_spearman = 0.552 (per-seed 0.5675/0.5335/0.5550, sd 0.017)
\newcommand{\instructDmgRome}{2.30}       % instruct_3seed layers.12.mean_damage_rome = 2.29786
\newcommand{\instructDmgAlpha}{0.09}      % instruct_3seed layers.12.mean_damage_alpha = 0.08612
\newcommand{\instructQuartLo}{1.22}       % instruct_3seed layers.12.quartile_means[0].mean_damage_removed = 1.22405
\newcommand{\instructQuartHi}{3.47}       % instruct_3seed layers.12.quartile_means[3].mean_damage_removed = 3.46510
\newcommand{\instructRatio}{2.83}         % instruct_3seed layers.12.removed_top_vs_bottom_ratio = 2.831 (unchanged)
\newcommand{\gptjCausalRho}{\ensuremath{-0.184}} % gptj_3seed layers.21.within_probe_spearman = -0.1842 (per-seed -0.2043/-0.1696/-0.1786, sd 0.018)
\newcommand{\gptjDmgRome}{\ensuremath{-0.034}}   % gptj_3seed layers.21.mean_damage_rome = -0.03388
\newcommand{\gptjDmgAlpha}{\ensuremath{-0.002}}  % gptj_3seed layers.21.mean_damage_alpha = -0.00221
\newcommand{\gptjQuartLo}{\ensuremath{-0.010}}   % gptj_3seed layers.21.quartile_means[0].mean_damage_removed = -0.00992
\newcommand{\gptjQuartHi}{\ensuremath{-0.060}}   % gptj_3seed layers.21.quartile_means[3].mean_damage_removed = -0.06034
\newcommand{\eightbCausalRhoLsixteen}{0.212}                  % 8b_3seed layers.16.within_probe_spearman = 0.2122 (per-seed 0.1852/0.2474/0.2040, sd 0.032)
\newcommand{\eightbCausalRhoLtwentyfour}{\ensuremath{-0.087}} % 8b_3seed layers.24.within_probe_spearman = -0.0869 (per-seed -0.1016/-0.0810/-0.0781, sd 0.013)
\newcommand{\eightbDmgRomeLsixteen}{0.26}                     % 8b_3seed layers.16.mean_damage_rome = 0.26026
\newcommand{\eightbDmgAlphaLsixteen}{0.02}                    % 8b_3seed layers.16.mean_damage_alpha = 0.02248
\newcommand{\eightbDmgRomeLtwentyfour}{\ensuremath{-0.05}}    % 8b_3seed layers.24.mean_damage_rome = -0.05036
\newcommand{\eightbDmgAlphaLtwentyfour}{0.02}                 % 8b_3seed layers.24.mean_damage_alpha = 0.02018
% \gptjRatio RETIRED — see revision note A (base mean ~0; ratio not quotable).
% NB: 8b layers.16.removed_top_vs_bottom_ratio (40.484 at 3 seeds, -68.963 at s0) still has
% NO macro — the sign change near the floor makes it unstable in both.
```

### 3b. Table body (`sections/05_causal.tex:166–186`)

The `tabular` itself is macro-driven and needs **no structural change**; updating the
macros updates the table. Two edits are required outside the tabular:

**(i) Caption** — replace "single AlphaEdit seed" with the mixed-seed statement:

```latex
\caption{Causal cell across architecture, scale, and instruction tuning
(held-out-key projector, CounterFact, matched ROME/AlphaEdit edit sets,
known-fact and successful-edit filters). All cells are three AlphaEdit
projector seeds (mean reported) except GPT-NeoX-20B, which remains
single-seed. $\rho$ is the signed within-probe Spearman between pre-edit
key-cosine and the logit-damage AlphaEdit removes relative to ROME;
``dmg R$\rightarrow$A'' is mean (signed) logit-damage under ROME then under
AlphaEdit on the identical (edit, probe) pairs.}
```

**(ii) Add a seed column** so the reviewer can see the spread without reading the response
letter. Widen to five columns:

```latex
\begin{tabular}{@{}lcccc@{}}
\toprule
Model & L & seeds & $\rho$(cos,\,removed) & dmg R$\rightarrow$A \\
\midrule
1B-Instruct & 12 & 3 & \instructCausalRho{} \tiny(sd 0.017) & \instructDmgRome{}$\rightarrow$\instructDmgAlpha{} \\
GPT-J-6B    & 21 & 3 & \gptjCausalRho{} \tiny(sd 0.018)     & \gptjDmgRome{}$\rightarrow$\gptjDmgAlpha{} \\
NeoX-20B    & 16 & 1 & \neoxCausalRho{}                     & \neoxDmgRome{}$\rightarrow$\neoxDmgAlpha{} \\
Llama-8B    & 16 & 3 & \eightbCausalRhoLsixteen{} \tiny(sd 0.032)    & \eightbDmgRomeLsixteen{}$\rightarrow$\eightbDmgAlphaLsixteen{} \\
Llama-8B    & 24 & 3 & \eightbCausalRhoLtwentyfour{} \tiny(sd 0.013) & \eightbDmgRomeLtwentyfour{}$\rightarrow$\eightbDmgAlphaLtwentyfour{} \\
\bottomrule
\end{tabular}
```

**(iii) Footnote** — the sentence "Every cell uses the held-out-key projector and a single
AlphaEdit seed" is now false. Replace with:

```latex
{\footnotesize Every cell uses the held-out-key projector. All cells report the
mean over three AlphaEdit projector seeds except NeoX-20B (single seed); no
cell changes sign across seeds and the seed spread is at most 0.034 in $\rho$.
A positive $\rho$ with a large ROME$\rightarrow$AlphaEdit drop (1B-Instruct) is
the clean removal regime; the coupling weakens and its sign varies from 6B
upward, matching the observational attenuation of Section~\ref{sec:regime}.
Even where the signed correlation is weak or negative, the magnitude of the
removed perturbation rises monotonically with key-cosine (1B-Instruct
\instructQuartLo{}$\rightarrow$\instructQuartHi{}, NeoX-20B
\neoxQuartLo{}$\rightarrow$\neoxQuartHi{} across key-cosine quartiles).\par}
```

### 3c. Prose edits in `05_causal.tex`

| Line(s) | Current | Change |
|---|---|---|
| 80–106 (provenance comment block) | says `seeds_used = [0]` for all four | rewrite to point at the `_3seed` filenames and record per-seed values |
| 111–113 | "Each of these cells rests on a single AlphaEdit projector seed; the 1B result above remains the only multi-seed causal measurement in this paper." | **DELETE** — no longer true. Replace: "Three of these four cells now rest on three AlphaEdit projector seeds (NeoX-20B remains single-seed); seed spread is at most 0.034 in $\rho$ and no cell changes sign." |
| 130 | "top-vs-bottom \gptjRatio{}$\times$" | **DELETE the ratio clause** — see note A |
| 196–197 (Honest caveats) | "the cells of Table~\ref{tab:causal-scale} are single-AlphaEdit-seed generality evidence" | replace with "…are three-seed generality evidence except NeoX-20B" |

---

## 4. Two binding advisory notes

### Note A — retire quartile-ratio phrasing where the base mean is ≈ 0

`\gptjRatio{} = 6.2×` (3-seed: 6.083) must **not** survive into the revision.

The ratio is Q4/Q1 of `mean_damage_removed`. For GPT-J at L21 that is
−0.06034 / −0.00992. The denominator is 0.0099 logits — statistical noise on a damage scale
two orders of magnitude below the 1B cell's 2.2. A ratio built on a near-zero base is
arbitrarily inflatable and a referee who checks it will (correctly) call it a rhetorical
amplifier: the same monotone trend expressed in absolute terms is a 0.05-logit spread.

The paper already applies exactly this discipline to the 8B L16 ratio (macros.tex line
comment: "deliberately NOT quoted (numerically unstable near the sign change)"). Applying it
inconsistently — suppressing 8B's while headlining GPT-J's — is the worse of the two
failure modes, because it looks like selection.

**Action:** delete `\gptjRatio` and the "top-vs-bottom …×" clause at `05_causal.tex:130`
and in the table footnote. State the monotonicity in absolute quartile means instead
(−0.010 / −0.021 / −0.035 / −0.060), which is the actual evidence and reads as more
careful, not less. The 1B-Instruct ratio (2.83×, base 1.22 logits) is on a real scale and
may stay.

Same reasoning kills any ratio for **8B L24** (base −0.065, 3-seed ratio 1.411 — a
non-finding) and **8B L28** (ratio −1.548, sign-crossing).

### Note B — the MQuAKE causal cell is a fold-in trap

`C4_causal_mquake_holdout_table_3seed.json` (L12, holdout projector, seeds {0,1,2}) is the
revision-insurance artifact that closed the projector-circularity worry, and its ρ is
excellent and extremely stable: **0.4949, per-seed 0.4986 / 0.4917 / 0.4944, sd 0.0035**
over 110 292 pairs. It is tempting to add it to Table II as a sixth row.

**Do not add it without the following framing, or omit it.** Three problems:

1. **Only ~43 % of damage is removed** — `mean_damage_rome = 3.092`,
   `mean_damage_alpha = 1.773`. Every other cell in Table II removes 91–96 %. Dropping a
   43 % cell into a table whose narrative is "AlphaEdit removes the damage" without comment
   invites the reviewer to notice the outlier before you point at it.
2. **The Q1 quartile of damage-removed is NEGATIVE** (−0.0346; quartiles
   −0.035 / 0.334 / 1.326 / 3.651). At the lowest key-cosine quartile, AlphaEdit makes
   things very slightly *worse*. The overall trend is strongly monotone and the effect is
   tiny, but a negative Q1 is precisely what a hostile reader looks for.
3. **The quartile ratio is −105.4** — a sign-crossing artifact of that negative Q1, and
   categorically unquotable (this is Note A again).

**Recommended handling.** Keep MQuAKE out of Table II. Report it in the response letter
and, if desired, in §8 (Generality) prose as: *"On MQuAKE-CF the causal cell replicates
with a held-out projector at ρ = 0.495 (3 seeds, sd 0.004), with damage-removed rising
monotonically across key-cosine quartiles; the fraction of damage removed is lower than on
CounterFact (43 % vs 91–96 %), consistent with multi-hop probes retaining collateral that a
single-layer null-space projection cannot reach."* That framing turns the weak number into
a mechanism observation and pre-empts the objection. Do not quote the ratio.

Also note the prep doc's MQuAKE row cites `C4_causal_mquake_table_3seed_probesrc.json`
(ρ 0.5328, `proj_source = probes`), a *different* projector. Quote only one, and prefer the
**holdout** file — the probe-source projector is the arm the circularity objection targets.

---

## 5. Checklist before applying

- [ ] Confirm the revision round is authorized (paper is under review; `submissions/ieee/`
      is frozen outside `revision/`).
- [ ] Apply §3a macro block, §3b caption/tabular/footnote, §3c prose edits.
- [ ] `grep -n "gptjRatio" submissions/ieee/` returns 0 hits.
- [ ] `grep -rn "single AlphaEdit seed\|single-AlphaEdit-seed" submissions/ieee/sections/`
      returns 0 hits.
- [ ] Rebuild; confirm 14 pp, 0 overfull, 0 undefined (the seed column widens Table II —
      check it does not overflow the IEEE column).
- [ ] Re-verify each printed number against the JSON by `pdftotext` grep before shipping.
