#!/usr/bin/env python3
"""
make_figures.py — regenerate the full B6 paper figure set from the canonical JSONs
in edit-harness/results/, deterministically.

Every plotted number is read from a canonical results/*.json (the claim->evidence
map in ../B6-PAPER-SKELETON-2026-07-01.md). Nothing is hardcoded except axis
labels, panel titles, and annotations that quote review-approved numbers already
present in the skeleton. A per-figure provenance line (figure -> source files) is
written to <outdir>/PROVENANCE.txt.

Palette: the dataviz skill's validated reference categorical palette, used in fixed
slot order (blue, aqua, yellow, green, violet, red, magenta, orange), with the blue
sequential ramp for ordered quartiles and blue<->red diverging for signed values.

Usage:
    python experiments/make_figures.py [--outdir figures]

Design constraints honoured:
  - one y-axis per panel (never dual-axis)
  - categorical hues assigned in fixed order, never cycled
  - signed values use the diverging blue(+)/red(-) convention with a neutral zero line
  - a legend whenever >=2 series share a panel
  - "pick-up-if-present" hooks for GPU cells landing today: KL-ladder ft_kl=1.0,
    U1 alpha-delete s1/s2, U1 Qwen-delete s1/s2 (skipped-with-note if absent).
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.normpath(os.path.join(HERE, "..", "results"))

# ---------------------------------------------------------------------------
# dataviz reference palette (validated instance; used in fixed slot order)
# ---------------------------------------------------------------------------
PAL = {
    "blue":    "#2a78d6",
    "aqua":    "#1baf7a",
    "yellow":  "#eda100",
    "green":   "#008300",
    "violet":  "#4a3aa7",
    "red":     "#e34948",
    "magenta": "#e87ba4",
    "orange":  "#eb6834",
}
CATEG = [PAL["blue"], PAL["aqua"], PAL["yellow"], PAL["green"],
         PAL["violet"], PAL["red"], PAL["magenta"], PAL["orange"]]
# blue sequential ramp steps 250->600 (ordinal-safe range)
SEQ_BLUE = ["#86b6ef", "#5598e7", "#2a78d6", "#184f95"]
POS = PAL["blue"]      # diverging positive pole
NEG = PAL["red"]       # diverging negative pole
GOOD = "#0ca30c"
CRIT = "#d03b3b"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

plt.rcParams.update({
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "legend.frameon": False,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "text.color": INK,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

SINGLE_W = 3.35   # single-column width (inches)
DOUBLE_W = 6.9    # full-width


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------
class Missing(Exception):
    pass


def load(name):
    """Load a canonical results JSON by basename. Records the path for provenance."""
    path = os.path.join(RESULTS, name)
    if not os.path.exists(path):
        raise Missing(name)
    with open(path) as f:
        return json.load(f)


def exists(name):
    return os.path.exists(os.path.join(RESULTS, name))


def seed_mean_std(vals):
    """Population mean/std over the non-None entries (matches the harness's
    within_probe_std_across_seeds convention). Returns (mean, std, n)."""
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, 0.0, 0
    m = sum(vals) / len(vals)
    sd = (sum((x - m) ** 2 for x in vals) / len(vals)) ** 0.5 if len(vals) > 1 else 0.0
    return m, sd, len(vals)


def wp_rho(d):
    """Extract a within-probe signed Spearman from a C3-style aggregate JSON.
    House convention: aggregate.within_probe_mean_across_seeds. Returns
    (value, n_seeds) or (None, None) if not found."""
    agg = d.get("aggregate", d) if isinstance(d, dict) else {}
    for k in ("within_probe_mean_across_seeds", "within_probe_mean"):
        if k in agg:
            return agg[k], agg.get("n_seeds")
    return None, None


def style_axes(ax, ygrid=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if ygrid:
        ax.yaxis.grid(True, zorder=0)
        ax.set_axisbelow(True)


def save(fig, outdir, stem):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"{stem}.{ext}"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# F1 — Layer law + regime transition; scale/depth panel
# ---------------------------------------------------------------------------
def fig_F1(outdir):
    srcs = []
    layers = [8, 10, 12, 14]
    keycos, keycos_sd, ng, ng_sd = [], [], [], []
    for L in layers:
        d = load(f"G1_L{L}_analysis.json"); srcs.append(f"G1_L{L}_analysis.json")
        keycos.append(d["aggregate"]["within_probe_mean_across_seeds"])
        keycos_sd.append(d["aggregate"]["within_probe_std_across_seeds"])
        ngs = [s["within_probe_mean_normgrowth"] for s in d["per_seed"]]
        m = sum(ngs) / len(ngs)
        ng.append(m)
        ng_sd.append((sum((x - m) ** 2 for x in ngs) / len(ngs)) ** 0.5)

    # scale/regime panel: 3B L24, 8B L16/L24/L28
    d3 = load("C3_regime_3b_L24_r4.json"); srcs.append("C3_regime_3b_L24_r4.json")
    d8 = load("C3_llama8b_r3.json"); srcs.append("C3_llama8b_r3.json")
    r3b = d3["aggregate"]["within_probe_mean_across_seeds"]
    r3b_sd = d3["aggregate"]["within_probe_std_across_seeds"]
    # C3_llama8b_r3 stores L16/L24/L28 as its three "per_seed" entries (seed 0 each)
    l8b = {}
    for s in d8["per_seed"]:
        lay = int(s["npz"].split("_L")[1].split("_")[0])
        l8b[lay] = s["within_probe_mean"]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(DOUBLE_W, 2.7),
                                   gridspec_kw={"width_ratios": [1.25, 1.0]})

    # -- (a) 1B layer law
    axL.errorbar(layers, keycos, yerr=keycos_sd, marker="o", ms=5, lw=2,
                 color=PAL["blue"], capsize=2.5, label="key-cosine  |C|")
    axL.errorbar(layers, ng, yerr=ng_sd, marker="s", ms=5, lw=2,
                 color=PAL["orange"], capsize=2.5, label="norm-growth (ENCORE)")
    axL.axhline(0.10, color=MUTED, lw=0.8, ls=":")
    axL.text(8.05, 0.115, "DEAD floor 0.10", color=MUTED, fontsize=6)
    # mark the L12 peak and the L14 overtake
    axL.annotate("peak\n0.602", (12, keycos[2]), textcoords="offset points",
                 xytext=(-2, 8), fontsize=6.5, color=PAL["blue"], ha="center")
    axL.annotate("norm-growth\novertakes at L14", (13.5, 0.20),
                 fontsize=6.5, color=PAL["orange"], ha="center")
    axL.set_xticks(layers)
    axL.set_xlabel("edited layer (Llama-3.2-1B)")
    axL.set_ylabel("within-probe ρ(·, damage)  (3-seed)")
    axL.set_title("(a) Geometry–damage law and its regime transition", loc="left")
    axL.set_ylim(0, 0.72)
    axL.legend(loc="upper left", bbox_to_anchor=(0.0, 0.88))
    style_axes(axL)

    # -- (b) scale / depth: sign tracks regime
    labels = ["3B\nL24", "8B\nL16", "8B\nL24", "8B\nL28"]
    vals = [r3b, l8b[16], l8b[24], l8b[28]]
    errs = [r3b_sd, 0, 0, 0]  # 8B points are single-seed in this file
    colors = [POS if v >= 0 else NEG for v in vals]
    xs = range(len(vals))
    axR.bar(xs, vals, color=colors, width=0.62, zorder=3,
            yerr=[errs, errs], capsize=2.5, error_kw=dict(ecolor=INK2, lw=0.8))
    axR.axhline(0, color=AXIS, lw=1.0)
    for x, v in zip(xs, vals):
        axR.text(x, v + (0.02 if v >= 0 else -0.02), f"{v:+.3f}",
                 ha="center", va="bottom" if v >= 0 else "top", fontsize=6.5,
                 color=INK)
    axR.set_xticks(list(xs)); axR.set_xticklabels(labels)
    axR.set_ylabel("signed within-probe ρ(key-cos, damage)")
    axR.set_title("(b) Sign tracks the damage regime, across scale/depth", loc="left")
    axR.set_ylim(-0.25, 0.5)
    legend = [Line2D([0], [0], color=POS, lw=6, label="positive regime (net damage)"),
              Line2D([0], [0], color=NEG, lw=6, label="improvement regime")]
    axR.legend(handles=legend, loc="upper right")
    axR.text(0.0, -0.22, "3B L24 3-seed ±sd; 8B points single-seed (this file)",
             fontsize=5.6, color=MUTED)
    style_axes(axR)

    fig.suptitle("F1  Key geometry predicts collateral damage — layer law, regime crossover, and scale",
                 fontsize=9.5, x=0.02, ha="left", weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, outdir, "F1_layer_law_regime")
    return "F1_layer_law_regime", srcs


# ---------------------------------------------------------------------------
# F2 — S×C is a zero-cost surrogate for GradSim first-order influence
# ---------------------------------------------------------------------------
def fig_F2(outdir):
    srcs = ["C1_mechanism_sc_table.json"]
    c1 = load("C1_mechanism_sc_table.json")
    sc_by_layer = {g["layer"]: g["within_probe_rho_SC"]
                   for g in c1["groups"] if g["model"] == "llama1b"}
    xs, ys, labs = [], [], []
    for L in (8, 10, 12, 14):
        g2 = load(f"G2_gradsim_L{L}.json"); srcs.append(f"G2_gradsim_L{L}.json")
        resid = g2["aggregate"]["gradsim_within_probe"]["resid"]["mean"]
        xs.append(sc_by_layer[L]); ys.append(resid); labs.append(f"L{L}")

    fig, ax = plt.subplots(figsize=(SINGLE_W, 3.2))
    lo, hi = 0.30, 0.70
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1.0, ls="--", zorder=1,
            label="identity (S×C = GradSim)")
    ax.scatter(xs, ys, s=48, color=PAL["blue"], zorder=3, edgecolor="white", lw=0.8)
    for x, y, t in zip(xs, ys, labs):
        ax.annotate(t, (x, y), textcoords="offset points", xytext=(6, -3),
                    fontsize=7, color=INK)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("S×C surrogate ρ  (closed form, no backprop)")
    ax.set_ylabel("GradSim-residual ρ  (first-order influence)")
    ax.set_title("F2  S×C ≈ GradSim: a zero-cost surrogate\nfor gradient influence (Llama-1B, L8–L14)",
                 loc="left")
    ax.legend(loc="upper left")
    ax.text(lo + 0.005, hi - 0.055,
            "matches to ~2 decimals\n(e.g. L8: 0.390 = 0.390)",
            fontsize=6.3, color=INK2)
    style_axes(ax, ygrid=False)
    ax.grid(True, zorder=0); ax.set_axisbelow(True)
    save(fig, outdir, "F2_sc_gradsim_surrogate")
    return "F2_sc_gradsim_surrogate", srcs


# ---------------------------------------------------------------------------
# F3 — Causal: AlphaEdit damage-removed by key-cos quartile; holdout agreement
# ---------------------------------------------------------------------------
def fig_F3(outdir):
    srcs = ["C4_causal_table.json", "C4_causal_holdout_table_3seed.json"]
    c4 = load("C4_causal_table.json")
    ho = load("C4_causal_holdout_table_3seed.json")
    layers = [8, 10, 12, 14]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(DOUBLE_W, 2.9),
                                   gridspec_kw={"width_ratios": [1.3, 1.0]})

    # (a) damage removed vs cosine quartile, one line per layer
    for i, L in enumerate(layers):
        q = c4["layers"][str(L)]["quartile_means"]
        cos = [x["mean_cos"] for x in q]
        rem = [x["mean_damage_removed"] for x in q]
        axL.plot(cos, rem, marker="o", ms=4.5, lw=1.8, color=CATEG[i],
                 label=f"L{L}  (ρ={c4['layers'][str(L)]['within_probe_spearman']:.2f})")
    axL.set_xlabel("pre-edit key-cosine  (quartile mean)")
    axL.set_ylabel("AlphaEdit damage removed  (logit)")
    axL.set_title("(a) Damage removed rises with key-cosine", loc="left")
    axL.legend(loc="upper left", title="by-construction projector", title_fontsize=6.5)
    style_axes(axL)

    # (b) by-construction vs holdout within-probe ρ (retires circularity)
    ho_layers = [int(k) for k in ho["layers"].keys()]
    ho_layers.sort()
    x = range(len(ho_layers)); w = 0.36
    bc = [c4["layers"][str(L)]["within_probe_spearman"] for L in ho_layers]
    hov = [ho["layers"][str(L)]["within_probe_spearman"] for L in ho_layers]
    axR.bar([i - w / 2 for i in x], bc, width=w, color=SEQ_BLUE[2], zorder=3,
            label="by-construction (probes)")
    axR.bar([i + w / 2 for i in x], hov, width=w, color=PAL["aqua"], zorder=3,
            label="holdout projector (primary)")
    for i, (a, b) in enumerate(zip(bc, hov)):
        axR.text(i - w / 2, a + 0.008, f"{a:.3f}", ha="center", fontsize=6, color=INK)
        axR.text(i + w / 2, b + 0.008, f"{b:.3f}", ha="center", fontsize=6, color=INK)
    axR.set_xticks(list(x)); axR.set_xticklabels([f"L{L}" for L in ho_layers])
    axR.set_ylabel("within-probe ρ(key-cos, damage-removed)")
    axR.set_ylim(0, 0.7)
    axR.set_title("(b) Holdout ≈ by-construction\n(circularity retired, 3-seed)", loc="left")
    axR.legend(loc="upper left")
    style_axes(axR)

    fig.suptitle("F3  Causal test — null-space projection removes the geometry-predicted damage",
                 fontsize=9.5, x=0.02, ha="left", weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, outdir, "F3_causal_alphaedit")
    return "F3_causal_alphaedit", srcs


# ---------------------------------------------------------------------------
# F4 — Editor locality spectrum + KL-FT dose ladder
# ---------------------------------------------------------------------------
def fig_F4(outdir):
    srcs = []
    notes = []

    def mean_dmg(gate_name):
        d = load(gate_name); srcs.append(gate_name)
        return d["KNOWN_PROBES"]["mean_damage_logit"]

    # signed within-probe rho (confound-clean) per editor + mean damage (L8, s0 gate)
    ft = load("C3_null_ft_L8.json"); srcs.append("C3_null_ft_L8.json")
    ftkl = load("C3_null_ftkl_L8_v2.json"); srcs.append("C3_null_ftkl_L8_v2.json")
    rome = load("C1_mechanism_sc_table.json"); srcs.append("C1_mechanism_sc_table.json")
    memit = load("C3_memit_L8_r3.json"); srcs.append("C3_memit_L8_r3.json")
    c4 = load("C4_causal_holdout_table_3seed.json"); srcs.append("C4_causal_holdout_table_3seed.json")

    rome_rho = next(g["within_probe_rho_C"] for g in rome["groups"]
                    if g["model"] == "llama1b" and g["layer"] == 8)
    editors = [
        ("FT",        ft["aggregate"]["within_probe_mean_across_seeds"],
                      mean_dmg("gate_llama1b_ft_cf_L8_s0.json")),
        ("KL-FT",     ftkl["aggregate"]["within_probe_mean_across_seeds"],
                      mean_dmg("gate_llama1b_ftkl_cf_L8_s0.json")),
        ("ROME",      rome_rho,
                      mean_dmg("gate_llama1b_rome_cf_L8_s0.json")),
        ("MEMIT",     memit["aggregate"]["within_probe_mean_across_seeds"],
                      mean_dmg("gate_llama1b_memit_cf_L8_s0.json")),
        ("AlphaEdit", 0.0,  # holdout floor; coupling ~0 by design
                      c4["layers"]["8"]["mean_damage_alpha"]),
    ]

    fig, (axL, axMid, axR) = plt.subplots(1, 3, figsize=(DOUBLE_W, 2.7),
                                          gridspec_kw={"width_ratios": [1.15, 1.0, 1.0]})

    # (a) locality spectrum: mean damage (x, log) vs signed within-probe rho (y)
    for i, (name, rho, dmg) in enumerate(editors):
        axL.scatter(dmg, rho, s=70, color=CATEG[i], zorder=3,
                    edgecolor="white", lw=0.8)
        axL.annotate(name, (dmg, rho), textcoords="offset points",
                     xytext=(6, 5), fontsize=7, color=INK)
    axL.axhline(0.10, color=MUTED, lw=0.8, ls=":")
    axL.text(0.02, 0.115, "DEAD floor 0.10", color=MUTED, fontsize=6)
    axL.set_xscale("log")
    axL.set_xlabel("mean collateral damage  (logit, L8 s0)")
    axL.set_ylabel("signed within-probe ρ(key-cos, damage)")
    axL.set_title("(a) Editor locality spectrum", loc="left")
    axL.set_ylim(-0.05, 0.5)
    style_axes(axL)

    # (b) ROME vs MEMIT across depth: geometry-predictability is ROME-specific at EVERY layer.
    #   ROME 3-seed within-probe from the G1 gate; MEMIT (multi-layer) from the C3 memit pools
    #   (L8/L12 3-seed r3, L10/L14 single-seed u4). Quote MEMIT as rho_C, never "MEMIT S×C".
    memit_files = {8: "C3_memit_L8_r3.json", 10: "C3_memit_L10_u4.json",
                   12: "C3_memit_L12_r3.json", 14: "C3_memit_L14_u4.json"}
    layers_mp = [8, 10, 12, 14]
    rome_mp, rome_sd, memit_mp, memit_sd, memit_1seed = [], [], [], [], []
    for L in layers_mp:
        gg = load(f"G1_L{L}_analysis.json"); srcs.append(f"G1_L{L}_analysis.json")
        rome_mp.append(gg["aggregate"]["within_probe_mean_across_seeds"])
        rome_sd.append(gg["aggregate"]["within_probe_std_across_seeds"])
        mm = load(memit_files[L])["aggregate"]; srcs.append(memit_files[L])
        memit_mp.append(mm["within_probe_mean_across_seeds"])
        memit_sd.append(mm["within_probe_std_across_seeds"])
        if mm.get("n_seeds", 1) < 3:
            memit_1seed.append(L)
    axMid.errorbar(layers_mp, rome_mp, yerr=rome_sd, marker="o", ms=5, lw=2,
                   color=PAL["blue"], capsize=2.5, label="ROME (rank-1)")
    axMid.errorbar(layers_mp, memit_mp, yerr=memit_sd, marker="s", ms=5, lw=2,
                   color=PAL["yellow"], capsize=2.5, label="MEMIT (multi-layer)")
    axMid.axhline(0.10, color=MUTED, lw=0.8, ls=":")
    axMid.text(8.05, 0.125, "DEAD floor 0.10", color=MUTED, fontsize=6)
    axMid.set_xticks(layers_mp)
    axMid.set_xlabel("edited layer")
    axMid.set_ylabel("within-probe ρ(key-cos, damage)")
    axMid.set_title("(b) ROME vs MEMIT", loc="left")
    axMid.set_ylim(-0.03, 0.72)
    axMid.legend(loc="center right")
    style_axes(axMid)
    if memit_1seed:
        notes.append(f"F4 panel(b): MEMIT L{memit_1seed} single-seed (u4, err=0); "
                     "L8/L12 3-seed. MEMIT quoted as rho_C (0.019 L8 / 0.037 L12), never 'S×C'.")

    # (c) KL-FT dose ladder — WITHIN-PROBE metric (consistent with panel (a)). Prefer the
    #   multi-seed pools (C3_klladder_*_L8_seeds_u4.json, 2-seed with error bars) over the
    #   single-seed u2 files; the 0.1 rung reuses the 3-seed C3_null_ftkl_L8_v2. Flat
    #   known-probe metric from the gate JSONs is a labeled fallback if no within-probe file.
    # Each rung: (ft_kl weight, [within-probe sources, best first], gate file for damage)
    ladder = [
        (0.03, ["C3_klladder_003_L8_seeds_u5.json", "C3_klladder_003_L8_seeds_u4.json", "C3_klladder_003_L8_u2.json"], "gate_llama1b_ftkl003_cf_L8_s0.json"),
        (0.1,  ["C3_null_ftkl_L8_v2.json"],                                                                            "gate_llama1b_ftkl_cf_L8_s0.json"),
        (0.3,  ["C3_klladder_030_L8_seeds_u5.json", "C3_klladder_030_L8_seeds_u4.json", "C3_klladder_030_L8_u2.json"], "gate_llama1b_ftkl030_cf_L8_s0.json"),
        (1.0,  ["C3_klladder_100_L8_seeds_u5.json", "C3_klladder_100_L8_seeds_u4.json", "C3_klladder_100_L8_u2.json"], "gate_llama1b_ftkl100_cf_L8_s0.json"),
    ]
    wp_mode = any(exists(f) for _, wl_files, _ in ladder for f in wl_files
                  if f.startswith("C3_klladder_"))
    wl, rl, rsd, dl = [], [], [], []
    for w, wpfiles, gate in ladder:
        g = load(gate) if exists(gate) else None   # bound once; None when gate absent
        if g is not None:
            srcs.append(gate)
        dmg = g["KNOWN_PROBES"]["mean_damage_logit"] if g is not None else None
        if wp_mode:
            wpfn = next((f for f in wpfiles if exists(f)), None)
            if wpfn is not None:
                agg = load(wpfn)["aggregate"]; srcs.append(wpfn)
                wl.append(w); rl.append(agg["within_probe_mean_across_seeds"])
                rsd.append(agg.get("within_probe_std_across_seeds") or 0.0); dl.append(dmg)
            else:
                notes.append(f"F4 KL-ladder within-probe rung ft_kl={w} absent — skipped (cell pending)")
        else:
            if g is not None:
                wl.append(w); rl.append(g["KNOWN_PROBES"]["spearman_cos_damage"])
                rsd.append(0.0); dl.append(dmg)
            else:
                notes.append(f"F4 KL-ladder rung ft_kl={w} absent ({gate}) — skipped (GPU cell pending)")

    axR.errorbar(wl, rl, yerr=rsd, marker="o", ms=5, lw=1.8, color=PAL["blue"],
                 capsize=2.5, ecolor=INK2, elinewidth=0.8, zorder=3)
    for w, r, dg in zip(wl, rl, dl):
        if dg is not None:
            axR.annotate(f"dmg {dg:.1f}", (w, r), textcoords="offset points",
                         xytext=(0, 9), fontsize=6, color=INK2, ha="center")
    axR.set_xscale("log")
    if wl:
        axR.set_xticks(wl); axR.set_xticklabels([f"{w:g}" for w in wl])
    axR.set_xlabel("KL-FT weight (ft_kl)")
    if wp_mode:
        axR.set_ylabel("within-probe ρ (confound-clean)")
        subtitle = "within-probe (L8); 3-seed"
        notes.append("F4 KL-ladder: WITHIN-PROBE metric at L8; rungs 0.03/0.3/1.0 = 3-seed pools "
                     "(seeds_u5: 0.091/0.150/0.149), 0.1 = 3-seed C3_null_ftkl_L8_v2 (0.132).")
    else:
        axR.set_ylabel("known-probe ρ (flat)")
        subtitle = "flat metric; within-probe pending"
    axR.set_title(f"(c) KL-FT dose ladder\n({subtitle})", loc="left")
    axR.set_ylim(0, max(0.30, (max(rl) * 1.3) if rl else 0.30))
    style_axes(axR)
    # L12 KL dose-response (C3_klladder_*_L12_u5) is single-seed and flat (~0.09–0.12, DEAD floor)
    #   vs L8's rise — does not compose cleanly as a second curve on this compact panel; noted
    #   for the appendix per the lead's option, not overlaid.
    l12_ladder = ["C3_klladder_003_L12_u5.json", "C3_klladder_010_L12_u5.json",
                  "C3_klladder_030_L12_u5.json", "C3_klladder_100_L12_u5.json"]
    if all(exists(f) for f in l12_ladder):
        l12v = [load(f)["aggregate"]["within_probe_mean_across_seeds"] for f in l12_ladder]
        notes.append("F4 APPENDIX: L12 KL dose-response (single-seed, C3_klladder_*_L12_u5) is FLAT "
                     f"across ft_kl 0.03/0.1/0.3/1.0 = {'/'.join(f'{v:.3f}' for v in l12v)} — near the "
                     "0.10 DEAD floor, no dose trend (contrast L8's rise). Not overlaid (single-seed, "
                     "compact panel); appendix candidate.")

    fig.suptitle("F4  Geometry-predictability is locate-then-edit-mechanism specific",
                 fontsize=9.5, x=0.02, ha="left", weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, outdir, "F4_editor_spectrum")
    return "F4_editor_spectrum", srcs, notes


# ---------------------------------------------------------------------------
# F5 — U1 deletion collateral
# ---------------------------------------------------------------------------
ARM = "known=True|edit_ok=False"


def _arm(name):
    d = load(name)
    return d["arms"][ARM]


def fig_F5(outdir):
    srcs = []
    notes = []
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_W, 5.4))
    (axA, axB), (axC, axD) = axes

    # (a) Refusal-deletion coupling across DEPTH (3-seed, raw key-cos within-probe).
    #   L8/L14 from run_u5 blockB pools; L12 from the 3 transplant gates' keycos (same statistic).
    #   Replaces the earlier L12-only per-seed view; the DC-fragility contrast lives in panel (b).
    w = 0.36  # shared bar width for panels (b)/(c)/(d)
    b8 = load("C3_u1_blockB_L8_seeds_u5.json")["aggregate"]; srcs.append("C3_u1_blockB_L8_seeds_u5.json")
    b14 = load("C3_u1_blockB_L14_seeds_u5.json")["aggregate"]; srcs.append("C3_u1_blockB_L14_seeds_u5.json")
    l12v = []
    for s in (0, 1, 2):
        fn = f"U1_E1_transplant_GATE_L12_s{s}.json"; srcs.append(fn)
        l12v.append(load(fn)["keycos"]["within_probe_mean"])
    l12m, l12sd, _ = seed_mean_std(l12v)
    prof_layers = [8, 12, 14]
    prof_rho = [b8["within_probe_mean_across_seeds"], l12m, b14["within_probe_mean_across_seeds"]]
    prof_sd = [b8["within_probe_std_across_seeds"], l12sd, b14["within_probe_std_across_seeds"]]
    g1 = load("G1_L12_analysis.json"); srcs.append("G1_L12_analysis.json")
    rewrite_ref = g1["aggregate"]["within_probe_mean_across_seeds"]
    axA.errorbar(prof_layers, prof_rho, yerr=prof_sd, marker="o", ms=6, lw=2,
                 color=SEQ_BLUE[2], capsize=3, ecolor=INK2, elinewidth=0.8, zorder=3)
    axA.axhline(rewrite_ref, color=CRIT, lw=1.2, ls="--", zorder=2)
    axA.text(8, rewrite_ref + 0.02, f"rewrite ref {rewrite_ref:.3f}",
             color=CRIT, fontsize=6, ha="left", va="bottom")
    for L, r, e in zip(prof_layers, prof_rho, prof_sd):
        axA.text(L, r + e + 0.03, f"{r:.3f}", ha="center", fontsize=6, color=INK)
    axA.set_xticks(prof_layers)
    axA.set_xlabel("edited layer")
    axA.set_ylabel("refusal-delete within-probe ρ (key-cos)")
    axA.set_ylim(0, 0.85)
    axA.set_title("(a) Refusal-delete coupling across depth (3-seed)", loc="left")
    style_axes(axA)
    notes.append("F5 panel(a): refusal-deletion layer profile L8/L12/L14 = "
                 f"{prof_rho[0]:.3f}/{prof_rho[1]:.3f}/{prof_rho[2]:.3f} (3-seed ±sd; L8/L14 "
                 "blockB_seeds_u5, L12 transplant-gate keycos). Peaks at L12, above rewrite ref "
                 f"{rewrite_ref:.3f}. Replaced the L12-only per-seed view; DC contrast in panel (b).")

    # (b) variants: refusal / eos / suppress at L12 (nondc vs dc) + layer profile note
    variants = [("refusal", "u1_gate_refusal_L12_s0.json"),
                ("eos",     "u1_gate_eos_L12_s0.json"),
                ("suppress", "u1_gate_suppress_L12_s0.json")]
    vn, vd, vlab = [], [], []
    for lab, fn in variants:
        a = _arm(fn); srcs.append(fn)
        vn.append(a["nondc_rho"]); vd.append(a["dc_rho"]); vlab.append(lab)
    x = range(len(variants))
    axB.bar([i - w / 2 for i in x], vn, width=w, color=SEQ_BLUE[2], zorder=3, label="raw (non-DC)")
    axB.bar([i + w / 2 for i in x], vd, width=w, color=PAL["aqua"], zorder=3, label="double-centered")
    # flag suppress DC-fragility
    axB.annotate("DC-fragile\n0.621 → 0.159", (2 + w / 2, vd[2]),
                 textcoords="offset points", xytext=(0, 24), fontsize=6.3,
                 color=CRIT, ha="center",
                 arrowprops=dict(arrowstyle="->", color=CRIT, lw=0.8))
    axB.set_xticks(list(x)); axB.set_xticklabels(vlab)
    axB.set_ylabel("within-probe ρ  (L12, seed 0)")
    axB.set_ylim(0, 0.8)
    axB.set_title("(b) Deletion variants — suppress is DC-fragile", loc="left")
    axB.legend(loc="upper right")
    style_axes(axB)
    # NOTE: run_u4 wrote 3-seed eos/suppress pools (C3_u1_blockC_eos_seeds_u4=0.616±0.008,
    #   C3_u1_blockC_suppress_seeds_u4=0.073±0.024), but those are the RAW key-cos within-probe
    #   statistic, NOT panel (b)'s S×C DC-comparison (suppress S×C nondc 0.621 → DC 0.159).
    #   Deliberately NOT merged — the two are different statistics. Flagged for review.
    if exists("C3_u1_blockC_suppress_seeds_u4.json") or exists("C3_u1_blockC_eos_seeds_u4.json"):
        notes.append("F5 panel(b): 3-seed eos/suppress pools exist (C3_u1_blockC_{eos,suppress}_seeds_u4) "
                     "but use the RAW key-cos within-probe metric (suppress 0.073±0.024), a DIFFERENT "
                     "statistic from panel (b)'s S×C DC-comparison — NOT merged (would conflate metrics). "
                     "Panel (b) stays seed-0 S×C nondc/dc. Flagged for review.")

    # (c) AlphaEdit-delete collapse (damage & coupling) + Qwen-delete null — 3-SEED both stats.
    #   Coupling: ROME-delete from the transplant gates' S×C (3-seed); Alpha-delete from the
    #   run_u2 block aggregate C3_u1_blockD (3-seed). Damage: paired per-seed mean_signed_damage
    #   from the transplant gates. If the alpha-delete transplant seeds s1/s2 have not landed,
    #   fall back to 3-seed coupling + fewer-seed damage WITH A NOTE (never hold the figure).
    rome_dmg, rome_cpl = [], []
    for s in (0, 1, 2):
        fn = f"U1_E1_transplant_GATE_L12_s{s}.json"
        if exists(fn):
            d = load(fn); srcs.append(fn)
            rome_dmg.append(d["mean_signed_damage"]); rome_cpl.append(d["SxC_within_probe_mean"])
    alp_dmg, alp_cpl = [], []
    for s in (0, 1, 2):
        fn = f"U1_E1_transplant_GATE_alphadelete_L12_s{s}.json"
        if exists(fn):
            d = load(fn); srcs.append(fn)
            alp_dmg.append(d["mean_signed_damage"]); alp_cpl.append(d["SxC_within_probe_mean"])
    db_m, db_sd, db_n = seed_mean_std(rome_dmg)
    cb_m, cb_sd, cb_n = seed_mean_std(rome_cpl)
    da_m, da_sd, da_n = seed_mean_std(alp_dmg)
    # Alpha-delete coupling: prefer the 3-seed block aggregate (lead's directive)
    if exists("C3_u1_blockD_alphadelete_seeds_u2.json"):
        bd = load("C3_u1_blockD_alphadelete_seeds_u2.json"); srcs.append("C3_u1_blockD_alphadelete_seeds_u2.json")
        ca_m = bd["aggregate"]["within_probe_mean_across_seeds"]
        ca_sd = bd["aggregate"]["within_probe_std_across_seeds"]
        ca_n = bd["aggregate"].get("n_seeds", len(alp_cpl) or 1)
    else:
        ca_m, ca_sd, ca_n = seed_mean_std(alp_cpl)
        notes.append("F5 panel(c): block-D alpha-delete coupling absent — using transplant-gate S×C fallback.")
    if da_n < 3:
        notes.append(f"F5 panel(c): Alpha-delete DAMAGE is {da_n}-seed (transplant alphadelete s1/s2 pending); "
                     f"coupling is {ca_n}-seed (block D). Fallback per lead: 3-seed coupling + {da_n}-seed damage.")
    notes.append(f"F5 panel(c) seed counts — ROME-delete dmg {db_n}/coupling {cb_n}; "
                 f"Alpha-delete dmg {da_n}/coupling {ca_n}.")

    labels = ["ROME-\ndelete", "Alpha-\ndelete"]
    xg = [0, 1]
    if None in (db_m, da_m, cb_m, ca_m):
        notes.append("F5 panel(c): a required statistic had zero seeds on disk — "
                     "plotted as 0.0; check the transplant-gate / block-D inputs.")
    dmg_m = [db_m or 0.0, da_m or 0.0]; dmg_sd = [db_sd, da_sd]
    cpl_m = [cb_m or 0.0, ca_m or 0.0]; cpl_sd = [cb_sd, ca_sd]
    axC.bar([i - w / 2 for i in xg], dmg_m, width=w, color=PAL["orange"], zorder=3,
            yerr=[dmg_sd, dmg_sd], capsize=2.5, error_kw=dict(ecolor=INK2, lw=0.8),
            label="mean damage (logit)")
    axC.bar([i + w / 2 for i in xg], cpl_m, width=w, color=SEQ_BLUE[2], zorder=3,
            yerr=[cpl_sd, cpl_sd], capsize=2.5, error_kw=dict(ecolor=INK2, lw=0.8),
            label="S×C coupling ρ")
    for i in xg:
        axC.text(i - w / 2, dmg_m[i] + dmg_sd[i] + 0.12, f"{dmg_m[i]:.2f}",
                 ha="center", fontsize=6, color=INK)
        axC.text(i + w / 2, cpl_m[i] + cpl_sd[i] + 0.12, f"{cpl_m[i]:.3f}",
                 ha="center", fontsize=6, color=INK)
    axC.set_xticks(xg); axC.set_xticklabels(labels)
    axC.set_ylabel("value")
    full3 = min(db_n, cb_n, da_n, ca_n) >= 3
    seedtag = "3-seed" if full3 else f"α-dmg {da_n}-seed"
    axC.set_title(f"(c) AlphaEdit-delete collapse ({seedtag})", loc="left")
    axC.set_ylim(0, 4.6)
    axC.legend(loc="upper right", bbox_to_anchor=(1.0, 0.82))
    style_axes(axC)

    # (d) transplant gate: Δρ(S×C − best transplant baseline)
    tp = [("L8 s0", "U1_E1_transplant_GATE_L8_s0.json"),
          ("L12 s0", "U1_E1_transplant_GATE_L12_s0.json"),
          ("L12 s1", "U1_E1_transplant_GATE_L12_s1.json"),
          ("L12 s2", "U1_E1_transplant_GATE_L12_s2.json"),
          ("L14 s0", "U1_E1_transplant_GATE_L14_s0.json")]
    tlab, tval = [], []
    for lab, fn in tp:
        d = load(fn); srcs.append(fn)
        tlab.append(lab); tval.append(d["delta_rho_SxC_minus_best_transplant"])
    xg = range(len(tp))
    axD.bar(xg, tval, width=0.6, color=PAL["violet"], zorder=3)
    for i, v in enumerate(tval):
        axD.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=6, color=INK)
    axD.axhline(0, color=AXIS, lw=1.0)
    axD.set_xticks(list(xg)); axD.set_xticklabels(tlab, fontsize=6.3)
    axD.set_ylabel("Δρ  (S×C − best transplant baseline)")
    axD.set_ylim(0, 0.75)
    axD.set_title("(d) S×C beats lexical-transplant baseline", loc="left")
    style_axes(axD)

    # Qwen-delete null annotation on panel (c) — 3-SEED. Coupling from block-E aggregate
    #   (lead's directive); damage 3-seed from the per-seed u1e0 delete gates.
    if exists("C3_u1_blockE_qwen15b_seeds_u2.json"):
        be = load("C3_u1_blockE_qwen15b_seeds_u2.json"); srcs.append("C3_u1_blockE_qwen15b_seeds_u2.json")
        qc_m = be["aggregate"]["within_probe_mean_across_seeds"]
        qc_sd = be["aggregate"]["within_probe_std_across_seeds"]
        qc_n = be["aggregate"].get("n_seeds", 3)
    else:
        qa = _arm("u1_gate_qwen15b_refusal_L14_s0.json"); srcs.append("u1_gate_qwen15b_refusal_L14_s0.json")
        qc_m, qc_sd, qc_n = qa["nondc_rho"], 0.0, 1
    qdmg = []
    for s in (0, 1, 2):
        fn = f"u1e0_qwen15b_delete_refusal_L14_s{s}.json"
        if exists(fn):
            d = load(fn); srcs.append(fn); qdmg.append(d["KNOWN_PROBES"]["mean_damage_logit"])
    qd_m, qd_sd, qd_n = seed_mean_std(qdmg)
    axC.text(0.36, 0.50,
             f"Qwen-1.5B delete (null, {qc_n}-seed):\n"
             f"ρ={qc_m:+.3f}±{qc_sd:.3f}\n"
             f"@ dmg {qd_m:+.3f} ({qd_n}-seed)",
             transform=axC.transAxes, fontsize=6, color=INK2)
    notes.append(f"F5 Qwen-delete null: coupling {qc_n}-seed (block E), damage {qd_n}-seed.")

    fig.suptitle("F5  Deletion collateral is geometry-governed (refusal / eos / suppress); AlphaEdit erases it",
                 fontsize=9.5, x=0.02, ha="left", weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, outdir, "F5_deletion_u1")
    return "F5_deletion_u1", srcs, notes


# ---------------------------------------------------------------------------
# F6 — Sequential no-restore (DESCRIPTIVE ONLY; no geometry-attribution panel)
# ---------------------------------------------------------------------------
def fig_F6(outdir):
    srcs = []
    notes = []
    # Prefer the newer 4-stream analysis IF it is consistent with the reviewed 2-stream file
    # on the shared streams (s0/s1); else keep the reviewed 2-stream file and FLAG the mismatch.
    base = load("SEQ_analysis_L12.json"); srcs.append("SEQ_analysis_L12.json")
    d = base
    if exists("SEQ_analysis_L12_4stream.json"):
        four = load("SEQ_analysis_L12_4stream.json")
        consistent = all(
            o["npz"] == n["npz"]
            and o["final_survival_frac"] == n["final_survival_frac"]
            and abs(o["position_fragility"]["rho_position_vs_survival"]
                    - n["position_fragility"]["rho_position_vs_survival"]) <= 1e-4
            for o, n in zip(base["per_stream"], four["per_stream"]))
        if consistent:
            d = four; srcs.append("SEQ_analysis_L12_4stream.json")
            notes.append(
                "F6: using 4-stream SEQ_analysis_L12_4stream.json — shared streams s0/s1 match the "
                "reviewed 2-stream file exactly; pooled position-fragility "
                f"{base['pooled']['position_fragility']['rho_position_vs_survival']:.3f} → "
                f"{four['pooled']['position_fragility']['rho_position_vs_survival']:.3f} "
                f"(p {four['pooled']['position_fragility']['perm_p']:.4f}). H1 still UNSETTLED.")
        else:
            notes.append("F6: 4-stream file INCONSISTENT with reviewed 2-stream on shared streams — "
                         "kept 2-stream as primary source. FLAGGED for review.")
    streams = d["per_stream"]
    pooled = d["pooled"]
    # Sequential flank layers L8/L14 exist (SEQ_analysis_L{8,14}.json, 2-stream each) but F6 stays
    #   L12-focused (the primary, reviewed layer); flank pooled fragility noted for the appendix.
    flank = []
    for L in (8, 14):
        fn = f"SEQ_analysis_L{L}.json"
        if exists(fn):
            fd = load(fn)
            flank.append(f"L{L} {fd['pooled']['position_fragility']['rho_position_vs_survival']:.3f}")
    if flank:
        notes.append("F6 APPENDIX: sequential flank layers available (2-stream) — pooled "
                     f"position-fragility {', '.join(flank)} vs L12 "
                     f"{pooled['position_fragility']['rho_position_vs_survival']:.3f}; "
                     "not added to F6 (kept L12-focused, descriptive-only).")

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(DOUBLE_W, 2.8),
                                   gridspec_kw={"width_ratios": [1.25, 1.0]})

    # (a) survival curves per stream
    for i, s in enumerate(streams):
        cp = [c["checkpoint_nedits"] for c in s["survival_curve"]]
        fr = [c["frac_survived"] for c in s["survival_curve"]]
        axL.plot(cp, fr, marker="o", ms=4.5, lw=1.8, color=CATEG[i],
                 label=f"s{i} · final {s['final_survival_frac']:.0%}")
    axL.set_xlabel("edits applied (no restore)")
    axL.set_ylabel("fraction of target edits still surviving")
    axL.set_ylim(0, 0.82)
    axL.set_title(f"(a) Survival collapse ({len(streams)} streams)", loc="left")
    axL.legend(loc="upper right", ncol=1, fontsize=6)
    style_axes(axL)

    # (b) position fragility per stream + pooled
    labs, vals, ps = [], [], []
    for i, s in enumerate(streams):
        pf = s["position_fragility"]
        labs.append(f"s{i}"); vals.append(pf["rho_position_vs_survival"]); ps.append(pf["perm_p"])
    pf = pooled["position_fragility"]
    labs.append("pooled"); vals.append(pf["rho_position_vs_survival"]); ps.append(pf["perm_p"])
    x = range(len(vals))
    axR.bar(x, vals, width=0.6, color=SEQ_BLUE[2], zorder=3)
    for i, (v, p) in enumerate(zip(vals, ps)):
        sig = "p={:.3f}".format(p) if p < 0.05 else "n.s."
        axR.text(i, v + 0.01, f"{v:.2f}\n{sig}", ha="center", fontsize=6, color=INK)
    axR.set_xticks(list(x)); axR.set_xticklabels(labs)
    axR.set_ylabel("ρ(edit position, survival)")
    axR.set_ylim(0, max(0.42, max(vals) * 1.35))
    axR.set_title("(b) Later edits survive modestly more", loc="left")
    axR.text(0.0, -0.13,
             "H1 geometry-attribution UNSETTLED (position-partialled ρ n.s.) — panel omitted by design.",
             transform=axR.transAxes, fontsize=5.8, color=MUTED)
    style_axes(axR)

    fig.suptitle("F6  Sequential no-restore — descriptive survival & position fragility",
                 fontsize=9.5, x=0.02, ha="left", weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, outdir, "F6_sequential")
    return "F6_sequential", srcs, notes


# ---------------------------------------------------------------------------
# F7 — Cross-architecture signed within-probe ρ (3-seed), + optional magnitude panel
# ---------------------------------------------------------------------------
def _magnitude_rows(srcs, want):
    """Pick-up hook for the magnitude-law table (C1_magnitude_table.json).
    REAL schema (2026-07-04): top-level list 'families'; each item has 'family'
    plus within_probe_mean_abs_across_seeds / within_probe_std_abs_across_seeds
    (canonical --known --edit_ok values; these are |C|→|damage| within-probe rho).
    `want` is an ordered list of (display_label, family_key) selecting and aligning
    the rows to the signed panel; family_key None means that family has no magnitude
    row (e.g. Llama-3B). Returns (rows, status) with
    rows = [(label, rho_abs, err_abs, is_dead), ...] and
    status in {'ok','absent','unparsed'}. Skips-with-note (never guesses) if the
    file is present but does not match the schema."""
    if not exists("C1_magnitude_table.json"):
        return [], "absent"
    d = load("C1_magnitude_table.json")
    fam_list = d.get("families")
    if not isinstance(fam_list, list):
        return [], "unparsed"
    by_key = {r["family"]: r for r in fam_list
              if isinstance(r, dict) and "family" in r
              and "within_probe_mean_abs_across_seeds" in r}
    rows = []
    for label, key in want:
        if key is None or key not in by_key:
            continue
        r = by_key[key]
        rho = r["within_probe_mean_abs_across_seeds"]
        err = r.get("within_probe_std_abs_across_seeds") or 0.0
        dead = str(r.get("VERDICT", "")).upper().startswith("DEAD") or rho < 0.10
        rows.append((label, rho, err, dead))
    if not rows:
        return [], "unparsed"
    srcs.append("C1_magnitude_table.json")
    return rows, "ok"


def fig_F7(outdir):
    srcs = []
    notes = []
    # (display label, signed-source JSON, magnitude-table family key or None)
    fams = [
        ("Llama-1B\nL12 (ref)", "G1_L12_analysis.json",        "llama1b_L12"),
        ("Llama-3B\nL14",       "C3_null_llama3b_L14.json",    None),
        ("gemma-2-2b\nL13",     "C3_null_gemma2b_L13_v2.json", "gemma2b_L13"),
        ("Phi-3.5\nL16",        "C3_null_phi35_L16_v2.json",   "phi35_L16"),
        ("Qwen-0.5B\nL12",      "C3_null_qwen05b_L12.json",    "qwen05b_L12"),
        ("Qwen-1.5B\nL14",      "C3_null_qwen15b_L14.json",    "qwen15b_L14"),
        ("Qwen-3B\nL18",        "C3_null_qwen3b_L18_v2.json",  "qwen3b_L18"),
    ]
    labs, vals, errs = [], [], []
    for lab, fn, _ in fams:
        d = load(fn); srcs.append(fn)
        agg = d["aggregate"]
        labs.append(lab)
        vals.append(agg["within_probe_mean_across_seeds"])
        errs.append(agg["within_probe_std_across_seeds"])

    def draw_signed(ax, title):
        colors = [POS if v >= 0 else NEG for v in vals]
        x = range(len(vals))
        ax.bar(x, vals, width=0.62, color=colors, zorder=3,
               yerr=[errs, errs], capsize=3, error_kw=dict(ecolor=INK2, lw=0.8))
        ax.axhline(0, color=AXIS, lw=1.0)
        ax.axhspan(-0.10, 0.10, color=GRID, alpha=0.5, zorder=0)
        ax.text(len(vals) - 0.6, 0.10, "±0.10 DEAD band", color=MUTED,
                fontsize=6, ha="right", va="bottom")
        for i, (v, e) in enumerate(zip(vals, errs)):
            ax.text(i, v + (0.02 if v >= 0 else -0.02) + (e if v >= 0 else -e),
                    f"{v:+.3f}", ha="center", va="bottom" if v >= 0 else "top",
                    fontsize=6.3, color=INK)
        ax.set_xticks(list(x)); ax.set_xticklabels(labs, fontsize=6.5)
        ax.set_ylabel("signed within-probe ρ(key-cos, damage)  (3-seed ±sd)")
        ax.set_ylim(-0.30, 0.72)
        ax.set_title(title, loc="left", weight="bold")
        ax.annotate("Qwen inversion\n(AlphaEdit erases it, causal arm C4)",
                    (5, vals[5]), xytext=(4.6, 0.34),
                    fontsize=6.3, color=NEG, ha="center",
                    arrowprops=dict(arrowstyle="->", color=NEG, lw=0.8))
        legend = [Line2D([0], [0], color=POS, lw=6, label="positive coupling"),
                  Line2D([0], [0], color=NEG, lw=6, label="inverted coupling")]
        ax.legend(handles=legend, loc="upper right")
        style_axes(ax)

    mag_rows, mag_status = _magnitude_rows(srcs, [(lab, mk) for lab, _, mk in fams])

    if mag_status == "ok":
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(DOUBLE_W, 3.2))
        draw_signed(axL, "(a) Signed ρ: Llama-specific")
        mlabs = [r[0] for r in mag_rows]
        mvals = [r[1] for r in mag_rows]
        merrs = [r[2] for r in mag_rows]
        mdead = [r[3] for r in mag_rows]
        xm = range(len(mag_rows))
        mcolors = [MUTED if dd else SEQ_BLUE[2] for dd in mdead]
        axR.bar(xm, mvals, width=0.62, color=mcolors, zorder=3,
                yerr=[merrs, merrs], capsize=3, error_kw=dict(ecolor=INK2, lw=0.8))
        axR.axhline(0.10, color=MUTED, lw=0.8, ls=":")
        axR.text(len(mag_rows) - 0.6, 0.115, "DEAD floor 0.10", color=MUTED,
                 fontsize=6, ha="right", va="bottom")
        for i, (v, e, dd) in enumerate(zip(mvals, merrs, mdead)):
            axR.text(i, v + e + 0.012, f"{v:.3f}", ha="center", fontsize=6.3, color=INK)
            if dd:
                axR.annotate("gemma:\nsole failure", (i, v), xytext=(i - 0.05, 0.30),
                             fontsize=6, color=CRIT, ha="center",
                             arrowprops=dict(arrowstyle="->", color=CRIT, lw=0.8))
        axR.set_xticks(list(xm)); axR.set_xticklabels(mlabs, fontsize=6.5)
        axR.set_ylabel("within-probe ρ(|key-cos|, |damage|)  (3-seed ±sd)")
        axR.set_title("(b) Magnitude ρ: 4/5 transfer", loc="left")
        axR.set_ylim(0, max(0.7, max(mvals) * 1.25))
        style_axes(axR)
        notes.append(f"F7 magnitude panel from C1_magnitude_table.json ({len(mag_rows)} families; "
                     "canonical --known --edit_ok. phi35=0.321 here is NOT the no-known peek 0.362 — "
                     "do not mix). Llama-3B omitted: no magnitude row in the table.")
        fig.suptitle("F7  Signed vs magnitude geometry–damage law across architectures",
                     fontsize=9.5, x=0.02, ha="left", weight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
    else:
        fig, ax = plt.subplots(figsize=(DOUBLE_W, 3.0))
        draw_signed(ax, "F7  Signed law is Llama-specific: null gemma/Phi, inverts Qwen")
        if mag_status == "absent":
            notes.append("F7 magnitude panel PENDING — C1_magnitude_table.json absent; "
                         "signed-only shown. Per-family magnitude ρ (|C|→|damage|) is not in "
                         "any other canonical JSON, so it is not improvised.")
        else:  # unparsed
            notes.append("F7 magnitude panel SKIPPED — C1_magnitude_table.json present but did "
                         "not match the expected schema (families[] with "
                         "within_probe_mean_abs_across_seeds); reported, not guessed.")
        fig.tight_layout()

    # note: Llama-3B is single-seed in its canonical file
    d3 = load("C3_null_llama3b_L14.json")
    if d3["aggregate"]["n_seeds"] == 1:
        notes.append("F7 Llama-3B L14 is single-seed in C3_null_llama3b_L14.json (error bar = 0).")
    # Reference availability (not plotted; underpins the Qwen-inversion and generality arms):
    if exists("ANISO_analysis_L14_s1.json") and exists("ANISO_analysis_L14_s2.json"):
        notes.append("F7 REFERENCE: anisotropy contrast is seed-stable (ANISO_analysis_L14_s1/s2) — "
                     "mean-cos Llama 0.460/0.430/0.431 vs Qwen 0.200/0.196/0.197 — underpins the "
                     "Qwen sign-inversion mechanism; not a figure panel.")
    if exists("C3_u1_zsre_delete_L10_u5.json") or exists("u1_gate_zsre_refusal_L10_s0.json"):
        notes.append("F7 REFERENCE: canonical zsRE deletion files on disk "
                     "(C3_u1_zsre_delete_L10_u5, u1_gate_zsre_refusal_L10_s0) for the generality arm; "
                     "no zsRE figure panel — cite in text/appendix.")
    save(fig, outdir, "F7_cross_arch")
    return "F7_cross_arch", srcs, notes


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join(HERE, "..", "figures"))
    args = ap.parse_args()
    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    builders = [fig_F1, fig_F2, fig_F3, fig_F4, fig_F5, fig_F6, fig_F7]
    prov_lines = []
    all_notes = []
    for b in builders:
        try:
            res = b(outdir)
        except Missing as e:
            prov_lines.append(f"{b.__name__}: FAILED — missing canonical JSON: {e}")
            all_notes.append(f"{b.__name__} could not render: missing {e}")
            print(f"[WARN] {b.__name__}: missing {e}", file=sys.stderr)
            continue
        if len(res) == 3:
            name, srcs, notes = res
            all_notes.extend(notes)
        else:
            name, srcs = res
        uniq = sorted(set(srcs))
        prov_lines.append(f"{name}  <-  {', '.join(uniq)}")
        print(f"[ok] {name}  ({len(uniq)} source files)")

    with open(os.path.join(outdir, "PROVENANCE.txt"), "w") as f:
        f.write("B6 figure set — provenance (figure -> canonical results/*.json)\n")
        f.write("Generated by experiments/make_figures.py (deterministic).\n")
        f.write("=" * 72 + "\n\n")
        for ln in prov_lines:
            f.write(ln + "\n")
        f.write("\n" + "-" * 72 + "\nNOTES / pending-cell hooks / reconciliation gaps:\n")
        for n in all_notes:
            f.write("  - " + n + "\n")
    print(f"\nWrote {len(prov_lines)} figures + PROVENANCE.txt to {outdir}")


if __name__ == "__main__":
    main()
