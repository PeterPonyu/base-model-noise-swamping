#!/usr/bin/env python3
"""
make_figures_tex.py — TeX-auditable figure set for the B6 ARR submission.

Sibling to make_figures.py. Same canonical-JSON-only rule, same hook-based data
loading (loaders imported directly from make_figures). Emits pgfplots/tikz code
into paper-arr/figures-tex/*.tex with the DATA INLINE (\addplot coordinates), and
above every data series a comment block citing the exact source JSON path + field
and the plotted value — so every number can be audited by READING the .tex against
edit-harness/results/*.json, no rendering required.

Consolidation (footprint reduction) vs the exploratory make_figures.py set:
  - F1 + F7  ->  one 2-column figure* (2x2 groupplot): layer law | scale/regime |
                 signed cross-arch | magnitude transfer. Labels fig:layerlaw +
                 fig:crossarch both attach to it.
  - F4       ->  tightened 3-panel groupplot (spectrum | ROME-vs-MEMIT | KL ladder).
  - F5       ->  2x2 groupplot (depth profile | variants | collapse | transplant).
  - F2/F3/F6 ->  compact column-native single figures (unchanged content).
All compose-vs-note decisions from make_figures.py carry over verbatim.

Usage:
    python experiments/make_figures_tex.py            # -> paper-arr/figures-tex/
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from make_figures import (  # noqa: E402  (import after sys.path tweak)
    load, exists, wp_rho, seed_mean_std, RESULTS,
)

OUT = os.path.normpath(os.path.join(HERE, "..", "..", "paper-arr", "figures-tex"))

# --------------------------------------------------------------------------
# audit-comment + emission helpers
# --------------------------------------------------------------------------
_PROV = []  # (figure, [source basenames])


def S(basename, field, value):
    """One audit line: cite the exact source JSON path + field + plotted value."""
    if isinstance(value, float):
        value = f"{value:.4f}"
    return f"    % SOURCE: results/{basename} :: {field} = {value}"


def coords(pairs):
    return " ".join(f"({x},{y})" for x, y in pairs)


def fmt(v, nd=4):
    return f"{v:.{nd}f}"


def write_fig(name, body, sources):
    # NO \resizebox: each figure sizes its axes explicitly (width/height budgeted
    # from \columnwidth=219pt / \textwidth=455pt) so pgfplots lays out at real
    # dimensions and LaTeX reports genuine overfulls (resizebox masked both the
    # overfulls and cramped the fonts). See paperpanel in _pgfpreamble.tex.
    path = os.path.join(OUT, f"{name}.tex")
    with open(path, "w") as f:
        f.write(body if body.endswith("\n") else body + "\n")
    _PROV.append((name, sorted(set(sources))))
    print(f"[ok] {name}.tex  ({len(set(sources))} source files)")


# ==========================================================================
# FigA  (F1 + F7 merged): layer law | scale/regime | signed cross-arch | magnitude
# ==========================================================================
def fig_lawtransfer():
    srcs = []
    L = [8, 10, 12, 14]

    # -- panel 1: key-cos vs norm-growth within-probe rho across depth (Llama-1B)
    kc, kc_sd, ng = [], [], []
    p1_src = []
    for l in L:
        d = load(f"G1_L{l}_analysis.json"); srcs.append(f"G1_L{l}_analysis.json")
        a = d["aggregate"]
        kc.append(a["within_probe_mean_across_seeds"]); kc_sd.append(a["within_probe_std_across_seeds"])
        ngs = [s["within_probe_mean_normgrowth"] for s in d["per_seed"]]
        m = sum(ngs) / len(ngs); ng.append(m)
        p1_src.append(S(f"G1_L{l}_analysis.json", "aggregate.within_probe_mean_across_seeds", a["within_probe_mean_across_seeds"]))
        p1_src.append(S(f"G1_L{l}_analysis.json", "mean(per_seed[].within_probe_mean_normgrowth)", m))
    kc_pts = coords(zip(L, [fmt(v) for v in kc]))
    kc_err = " ".join(f"({x},{fmt(v)}) +- (0,{fmt(e)})" for x, v, e in zip(L, kc, kc_sd))
    ng_pts = coords(zip(L, [fmt(v) for v in ng]))

    # -- panel 2: signed rho at deep layers across scale (3B L24, 8B L16/L24/L28)
    d3 = load("C3_regime_3b_L24_r4.json"); srcs.append("C3_regime_3b_L24_r4.json")
    r3 = d3["aggregate"]["within_probe_mean_across_seeds"]; r3sd = d3["aggregate"]["within_probe_std_across_seeds"]
    d8 = load("C3_llama8b_r3.json"); srcs.append("C3_llama8b_r3.json")
    l8b = {int(s["npz"].split("_L")[1].split("_")[0]): s["within_probe_mean"] for s in d8["per_seed"]}
    scale = [("3B L24", r3), ("8B L16", l8b[16]), ("8B L24", l8b[24]), ("8B L28", l8b[28])]
    p2_src = [S("C3_regime_3b_L24_r4.json", "aggregate.within_probe_mean_across_seeds", r3)]
    for lay in (16, 24, 28):
        p2_src.append(S("C3_llama8b_r3.json", f"per_seed[L{lay}].within_probe_mean", l8b[lay]))
    pos2 = [(i, v) for i, (_, v) in enumerate(scale) if v >= 0]
    neg2 = [(i, v) for i, (_, v) in enumerate(scale) if v < 0]

    # -- panel 3: signed within-probe rho per architecture family (7)
    fams = [
        ("Llama-1B L12", "G1_L12_analysis.json", "llama1b_L12"),
        ("Llama-3B L14", "C3_null_llama3b_L14.json", None),
        ("gemma-2-2b L13", "C3_null_gemma2b_L13_v2.json", "gemma2b_L13"),
        ("Phi-3.5 L16", "C3_null_phi35_L16_v2.json", "phi35_L16"),
        ("Qwen-0.5B L12", "C3_null_qwen05b_L12.json", "qwen05b_L12"),
        ("Qwen-1.5B L14", "C3_null_qwen15b_L14.json", "qwen15b_L14"),
        ("Qwen-3B L18", "C3_null_qwen3b_L18_v2.json", "qwen3b_L18"),
    ]
    fv, fsd, p3_src = [], [], []
    for lab, fn, _ in fams:
        a = load(fn)["aggregate"]; srcs.append(fn)
        fv.append(a["within_probe_mean_across_seeds"]); fsd.append(a["within_probe_std_across_seeds"])
        p3_src.append(S(fn, "aggregate.within_probe_mean_across_seeds", a["within_probe_mean_across_seeds"]))
    pos3 = [(i, v, e) for i, (v, e) in enumerate(zip(fv, fsd)) if v >= 0]
    neg3 = [(i, v, e) for i, (v, e) in enumerate(zip(fv, fsd)) if v < 0]

    # -- panel 4: magnitude law |C|->|damage| per family (6)
    mt = load("C1_magnitude_table.json"); srcs.append("C1_magnitude_table.json")
    by = {r["family"]: r for r in mt["families"]}
    want = [("Llama-1B L12", "llama1b_L12"), ("gemma-2-2b L13", "gemma2b_L13"),
            ("Phi-3.5 L16", "phi35_L16"), ("Qwen-0.5B L12", "qwen05b_L12"),
            ("Qwen-1.5B L14", "qwen15b_L14"), ("Qwen-3B L18", "qwen3b_L18")]
    mv, msd, mdead, p4_src = [], [], [], []
    for lab, key in want:
        r = by[key]
        v = r["within_probe_mean_abs_across_seeds"]; e = r.get("within_probe_std_abs_across_seeds") or 0.0
        dead = str(r.get("VERDICT", "")).upper().startswith("DEAD") or v < 0.10
        mv.append(v); msd.append(e); mdead.append(dead)
        p4_src.append(S("C1_magnitude_table.json", f"families[family={key}].within_probe_mean_abs_across_seeds", v))
    mlive = [(i, v, e) for i, (v, e, dd) in enumerate(zip(mv, msd, mdead)) if not dd]
    mdd = [(i, v, e) for i, (v, e, dd) in enumerate(zip(mv, msd, mdead)) if dd]

    short = {"Llama-1B L12": "Llama-1B", "Llama-3B L14": "Llama-3B",
             "gemma-2-2b L13": "gemma", "Phi-3.5 L16": "Phi-3.5",
             "Qwen-0.5B L12": "Qwen-0.5B", "Qwen-1.5B L14": "Qwen-1.5B",
             "Qwen-3B L18": "Qwen-3B"}
    famxticks = ",".join(str(i) for i in range(7))
    famxlabels = ",".join("{" + short[lab] + "}" for lab, _, _ in fams)
    magxticks = ",".join(str(i) for i in range(6))
    magxlabels = ",".join("{" + short[lab] + "}" for lab, _ in want)
    sc_xlabels = ",".join("{" + s[0].replace(" ", "/") + "}" for s in scale)

    tex = rf"""% =====================================================================
% fig:layertransfer  (consolidates F1 + F7).  Claims C2 (depth/regime) + C3 (arch).
% Every plotted number is audited below against edit-harness/results/*.json.
% Regenerate: python edit-harness/experiments/make_figures_tex.py
% =====================================================================
\begin{{figure*}}[t]
\centering
\begin{{tikzpicture}}
\begin{{groupplot}}[
  group style={{group size=2 by 2, horizontal sep=1.7cm, vertical sep=2.2cm}},
  paperpanel, width=5.8cm, height=3.8cm,
]
% ---- panel (a): geometry-damage law across depth (Llama-3.2-1B) ----
\nextgroupplot[title={{(a) depth}},
  xlabel={{edited layer}}, ylabel={{within-probe $\rho$}},
  xtick={{8,10,12,14}}, ymin=0, ymax=0.72, xmin=7.4, xmax=14.6,
  ]
{chr(10).join(p1_src)}
\addplot+[vizblue, mark=*, mark options={{fill=vizblue, scale=0.7}}] plot[error bars/.cd, y dir=both, y explicit]
  coordinates {{{kc_err}}};
\addlegendentry{{key-cos $|C|$}}
\addplot+[vizorange, mark=square*, mark options={{fill=vizorange, scale=0.7}}]
  coordinates {{{ng_pts}}};
\addlegendentry{{norm-growth}}
\draw[vizmuted, densely dotted, line width=0.4pt] (axis cs:7.4,0.10) -- (axis cs:14.6,0.10);
\node[vizlabelm, anchor=south west] at (axis cs:7.5,0.10) {{DEAD 0.10}};
% ---- panel (b): sign tracks damage regime across scale/depth ----
\nextgroupplot[title={{(b) regime}},
  ylabel={{signed $\rho$}},
  xtick={{0,1,2,3}}, xticklabels={{{sc_xlabels}}}, x tick label style={{rotate=42, anchor=east, font=\tiny}},
  ymin=-0.25, ymax=0.5, xmin=-0.6, xmax=3.6, ybar, bar width=7pt]
{chr(10).join(p2_src)}
\addplot+[draw=vizblue, fill=vizblue] coordinates {{{coords((i, fmt(v)) for i, v in pos2)}}};
\addplot+[draw=vizred, fill=vizred] coordinates {{{coords((i, fmt(v)) for i, v in neg2)}}};
\draw[vizmuted, line width=0.4pt] (axis cs:-0.6,0) -- (axis cs:3.6,0);
% ---- panel (c): signed law per architecture family ----
\nextgroupplot[title={{(c) signed}},
  ylabel={{signed $\rho$}},
  xtick={{{famxticks}}}, xticklabels={{{famxlabels}}}, x tick label style={{rotate=42, anchor=east, font=\tiny}},
  ymin=-0.28, ymax=0.72, xmin=-0.6, xmax=6.6, ybar, bar width=5pt]
{chr(10).join(p3_src)}
\draw[vizgrid, line width=6pt, opacity=0.5] (axis cs:-0.6,0) -- (axis cs:6.6,0);
\addplot+[draw=vizblue, fill=vizblue] plot[error bars/.cd, y dir=both, y explicit]
  coordinates {{{" ".join(f"({i},{fmt(v)}) +- (0,{fmt(e)})" for i, v, e in pos3)}}};
\addplot+[draw=vizred, fill=vizred] plot[error bars/.cd, y dir=both, y explicit]
  coordinates {{{" ".join(f"({i},{fmt(v)}) +- (0,{fmt(e)})" for i, v, e in neg3)}}};
\draw[vizmuted, line width=0.4pt] (axis cs:-0.6,0) -- (axis cs:6.6,0);
% ---- panel (d): magnitude law transfers 4/5 off Llama ----
\nextgroupplot[title={{(d) magnitude}},
  ylabel={{$\rho(|C|,|$dmg$|)$}},
  xtick={{{magxticks}}}, xticklabels={{{magxlabels}}}, x tick label style={{rotate=42, anchor=east, font=\tiny}},
  ymin=0, ymax=0.72, xmin=-0.6, xmax=5.6, ybar, bar width=6pt]
{chr(10).join(p4_src)}
\addplot+[draw=vizseqblue, fill=vizseqblue] plot[error bars/.cd, y dir=both, y explicit]
  coordinates {{{" ".join(f"({i},{fmt(v)}) +- (0,{fmt(e)})" for i, v, e in mlive)}}};
\addplot+[draw=vizmuted, fill=vizmuted] plot[error bars/.cd, y dir=both, y explicit]
  coordinates {{{" ".join(f"({i},{fmt(v)}) +- (0,{fmt(e)})" for i, v, e in mdd)}}};
\draw[vizmuted, densely dotted, line width=0.4pt] (axis cs:-0.6,0.10) -- (axis cs:5.6,0.10);
\end{{groupplot}}
\end{{tikzpicture}}
\caption{{The Llama-family law across depth, scale, and architecture (consolidates
the former F1+F7). \textbf{{(a)}} within-probe $\rho$(key-cos, damage) traces an
inverted-U over depth on Llama-3.2-1B (peak $0.602$ at L12), with matrix-norm-growth
overtaking key-cosine at L14. \textbf{{(b)}} the coupling's sign tracks the sign of the
mean-damage regime across scale (Llama-3B L24 positive; Llama-8B L24 improvement).
\textbf{{(c)}} the \emph{{signed}} law is Llama-specific: null on gemma/Phi, inverted on
Qwen. \textbf{{(d)}} the \emph{{magnitude}} law $|C|\!\to\!|$dmg$|$ transfers on four of five
non-Llama families; gemma is the sole failure (grey, below the 0.10 DEAD floor).
Error bars are 3-seed s.d.\ where seeds exist.}}
\label{{fig:layerlaw}}
\label{{fig:crossarch}}
\end{{figure*}}
"""
    write_fig("figA_lawtransfer", tex, srcs)


# ==========================================================================
# F2 surrogate — S x C vs GradSim identity scatter (Llama-1B L8-L14)
# ==========================================================================
def fig_surrogate():
    srcs = ["C1_mechanism_sc_table.json"]
    c1 = load("C1_mechanism_sc_table.json")
    sc = {g["layer"]: g["within_probe_rho_SC"] for g in c1["groups"] if g["model"] == "llama1b"}
    pts, ptsrc = [], []
    for l in (8, 10, 12, 14):
        g2 = load(f"G2_gradsim_L{l}.json"); srcs.append(f"G2_gradsim_L{l}.json")
        resid = g2["aggregate"]["gradsim_within_probe"]["resid"]["mean"]
        pts.append((sc[l], resid, l))
        ptsrc.append(S("C1_mechanism_sc_table.json", f"groups[llama1b,L{l}].within_probe_rho_SC", sc[l]))
        ptsrc.append(S(f"G2_gradsim_L{l}.json", "aggregate.gradsim_within_probe.resid.mean", resid))
    scat = coords((fmt(x), fmt(y)) for x, y, _ in pts)
    # Points sit on the identity line; anchor labels just below-right of each
    # point (into the empty lower-right half-plane) so no label grazes the line.
    labels = "\n".join(rf"\node[vizlabel, anchor=north west, xshift=1.5pt, yshift=-1pt] "
                       rf"at (axis cs:{fmt(x)},{fmt(y)}) {{L{l}}};"
                       for x, y, l in pts)
    tex = rf"""% =====================================================================
% fig:surrogate (F2).  Claim C1: S x C is a zero-cost GradSim surrogate.
% Regenerate: python edit-harness/experiments/make_figures_tex.py
% =====================================================================
\begin{{figure}}[t]
\centering
\begin{{tikzpicture}}
\begin{{axis}}[paperpanel, width=6.0cm, height=4.7cm,
  title={{(F2) $S\times C \approx$ GradSim, no backprop}},
  xlabel={{$S\times C$ surrogate $\rho$}}, ylabel={{GradSim-residual $\rho$}},
  xmin=0.30, xmax=0.70, ymin=0.30, ymax=0.70, axis equal image=false,
  xtick={{0.3,0.4,0.5,0.6,0.7}}, ytick={{0.3,0.4,0.5,0.6,0.7}}]
\addplot[vizmuted, dashed, line width=0.5pt, forget plot] coordinates {{(0.30,0.30) (0.70,0.70)}};
{chr(10).join(ptsrc)}
\addplot[only marks, vizblue, mark=*, mark options={{fill=vizblue, scale=0.9}}]
  coordinates {{{scat}}};
{labels}
\node[vizlabelm, anchor=north west] at (axis cs:0.305,0.665)
  {{identity: $\rho_{{SC}}{{=}}$GradSim}};
\end{{axis}}
\end{{tikzpicture}}
\caption{{\SxC{{}} as a zero-cost GradSim surrogate. The closed-form product
$\rho_{{SC}}$ tracks the GradSim-residual within-probe $\rho$ to $\sim$2 decimals
across L8--L14 (points on the identity line).}}
\label{{fig:surrogate}}
\end{{figure}}
"""
    write_fig("figB_surrogate", tex, srcs)


# ==========================================================================
# F3 causal — quartile damage-removed per layer + by-construction vs holdout
# ==========================================================================
def fig_causal():
    srcs = ["C4_causal_table.json", "C4_causal_holdout_table_3seed.json"]
    c4 = load("C4_causal_table.json"); ho = load("C4_causal_holdout_table_3seed.json")
    layers = [8, 10, 12, 14]
    colors = ["vizblue", "vizaqua", "vizyellow", "vizgreen"]
    series, ssrc = [], []
    for l in layers:
        q = c4["layers"][str(l)]["quartile_means"]
        pts = [(x["mean_cos"], x["mean_damage_removed"]) for x in q]
        series.append((l, pts))
        for x in q:
            ssrc.append(S("C4_causal_table.json",
                          f"layers.{l}.quartile_means[cos={x['mean_cos']:.3f}].mean_damage_removed",
                          x["mean_damage_removed"]))
    hol = sorted(int(k) for k in ho["layers"].keys())
    bc = [c4["layers"][str(l)]["within_probe_spearman"] for l in hol]
    hv = [ho["layers"][str(l)]["within_probe_spearman"] for l in hol]
    hsrc = []
    for l, b, h in zip(hol, bc, hv):
        hsrc.append(S("C4_causal_table.json", f"layers.{l}.within_probe_spearman", b))
        hsrc.append(S("C4_causal_holdout_table_3seed.json", f"layers.{l}.within_probe_spearman", h))

    plotlines = []
    # Direct-label the four depth curves at their right endpoints (no legend box
    # for panel (a)) so its legend cannot collide with panel (b)'s legend above
    # the narrow adjacent panels. L8/L14 endpoints are close, so nudge them apart.
    ystag = {8: "-2.5pt", 14: "2.5pt"}
    for (l, pts), c in zip(series, colors):
        plotlines.append(rf"\addplot+[{c}, mark=*, mark options={{fill={c}, scale=0.6}}] "
                         rf"coordinates {{{coords((fmt(x,3), fmt(y,3)) for x, y in pts)}}};")
        ex, ey = pts[-1]
        plotlines.append(rf"\node[vizlabel, anchor=west, xshift=2pt, yshift={ystag.get(l, '0pt')}] "
                         rf"at (axis cs:{fmt(ex, 3)},{fmt(ey, 3)}) {{L{l}}};")

    hx = list(range(len(hol)))
    hxlab = ",".join(f"L{l}" for l in hol)
    tex = rf"""% =====================================================================
% fig:causal (F3).  Claim C4: AlphaEdit removes the geometry-predicted damage.
% Regenerate: python edit-harness/experiments/make_figures_tex.py
% =====================================================================
\begin{{figure}}[t]
\centering
\begin{{tikzpicture}}
\begin{{groupplot}}[group style={{group size=2 by 1, horizontal sep=1.5cm}},
  paperpanel, width=2.9cm, height=3.8cm]
\nextgroupplot[title={{(a) removed}},
  xlabel={{pre-edit key-cosine}}, ylabel={{AlphaEdit damage removed}},
  xmin=0.08, xmax=0.72, ymin=0.5]
{chr(10).join(ssrc)}
{chr(10).join(plotlines)}
\nextgroupplot[title={{(b) $\approx$ holdout}},
  ylabel={{within-probe $\rho$}}, ybar, bar width=7pt,
  xtick={{{",".join(map(str, hx))}}}, xticklabels={{{hxlab}}},
  ymin=0, ymax=0.72, xmin=-0.6, xmax={len(hol) - 0.4},
  ]
{chr(10).join(hsrc)}
\addplot+[draw=vizseqblue, fill=vizseqblue] coordinates {{{coords((i, fmt(v,3)) for i, v in zip(hx, bc))}}};
\addlegendentry{{by-construction}}
\addplot+[draw=vizaqua, fill=vizaqua] coordinates {{{coords((i, fmt(v,3)) for i, v in zip(hx, hv))}}};
\addlegendentry{{holdout}}
\end{{groupplot}}
\end{{tikzpicture}}
\caption{{AlphaEdit removes the geometry-predicted damage. \textbf{{(a)}} absolute damage
removed rises monotonically across key-cosine quartiles at all four layers.
\textbf{{(b)}} the held-out projector matches the by-construction arm, retiring the
circularity objection.}}
\label{{fig:causal}}
\end{{figure}}
"""
    write_fig("figC_causal", tex, srcs)


# ==========================================================================
# F4 editor — spectrum | ROME-vs-MEMIT depth | KL ladder  (3-panel groupplot)
# ==========================================================================
def fig_editor():
    srcs = []

    def mdmg(g):
        d = load(g); srcs.append(g)
        return d["KNOWN_PROBES"]["mean_damage_logit"]

    ft = load("C3_null_ft_L8.json"); srcs.append("C3_null_ft_L8.json")
    ftkl = load("C3_null_ftkl_L8_v2.json"); srcs.append("C3_null_ftkl_L8_v2.json")
    rome = load("C1_mechanism_sc_table.json"); srcs.append("C1_mechanism_sc_table.json")
    memit = load("C3_memit_L8_r3.json"); srcs.append("C3_memit_L8_r3.json")
    c4 = load("C4_causal_holdout_table_3seed.json"); srcs.append("C4_causal_holdout_table_3seed.json")
    rome_rho = next(g["within_probe_rho_C"] for g in rome["groups"] if g["model"] == "llama1b" and g["layer"] == 8)
    # (name, y=rho, x=mean-damage, y-source file, y-field, x-source file, x-field, label-anchor, xshift, yshift)
    editors = [
        ("FT", ft["aggregate"]["within_probe_mean_across_seeds"], mdmg("gate_llama1b_ft_cf_L8_s0.json"),
         "C3_null_ft_L8.json", "aggregate.within_probe_mean_across_seeds",
         "gate_llama1b_ft_cf_L8_s0.json", "KNOWN_PROBES.mean_damage_logit", "south", "0pt", "2pt"),
        ("KL-FT", ftkl["aggregate"]["within_probe_mean_across_seeds"], mdmg("gate_llama1b_ftkl_cf_L8_s0.json"),
         "C3_null_ftkl_L8_v2.json", "aggregate.within_probe_mean_across_seeds",
         "gate_llama1b_ftkl_cf_L8_s0.json", "KNOWN_PROBES.mean_damage_logit", "south", "0pt", "2pt"),
        ("ROME", rome_rho, mdmg("gate_llama1b_rome_cf_L8_s0.json"),
         "C1_mechanism_sc_table.json", "groups[llama1b,L8].within_probe_rho_C",
         "gate_llama1b_rome_cf_L8_s0.json", "KNOWN_PROBES.mean_damage_logit", "south", "0pt", "2pt"),
        ("MEMIT", memit["aggregate"]["within_probe_mean_across_seeds"], mdmg("gate_llama1b_memit_cf_L8_s0.json"),
         "C3_memit_L8_r3.json", "aggregate.within_probe_mean_across_seeds",
         "gate_llama1b_memit_cf_L8_s0.json", "KNOWN_PROBES.mean_damage_logit", "south", "0pt", "4pt"),
        ("AlphaEdit", 0.0, c4["layers"]["8"]["mean_damage_alpha"],
         "C4_causal_holdout_table_3seed.json", "layers.8: AlphaEdit coupling ~0 by design (floor)",
         "C4_causal_holdout_table_3seed.json", "layers.8.mean_damage_alpha", "west", "3pt", "0pt"),
    ]
    espec_src, espec_pts = [], []
    ecolors = ["vizblue", "vizaqua", "vizyellow", "vizgreen", "vizviolet"]
    for (name, rho, dmg, yf, yfield, xf, xfield, anch, xsh, ysh), c in zip(editors, ecolors):
        espec_src.append(S(yf, yfield + " [y=rho]", rho))
        espec_src.append(S(xf, xfield + " [x=mean damage]", dmg))
        espec_pts.append((name, dmg, rho, c, anch, xsh, ysh))

    # ROME vs MEMIT across depth
    memit_files = {8: "C3_memit_L8_r3.json", 10: "C3_memit_L10_u4.json", 12: "C3_memit_L12_r3.json", 14: "C3_memit_L14_u4.json"}
    layers = [8, 10, 12, 14]
    rome_l, rome_sd, memit_l, memit_sd, mp_src = [], [], [], [], []
    for l in layers:
        g = load(f"G1_L{l}_analysis.json")["aggregate"]; srcs.append(f"G1_L{l}_analysis.json")
        rome_l.append(g["within_probe_mean_across_seeds"]); rome_sd.append(g["within_probe_std_across_seeds"])
        mm = load(memit_files[l])["aggregate"]; srcs.append(memit_files[l])
        memit_l.append(mm["within_probe_mean_across_seeds"]); memit_sd.append(mm["within_probe_std_across_seeds"])
        mp_src.append(S(f"G1_L{l}_analysis.json", "aggregate.within_probe_mean_across_seeds", g["within_probe_mean_across_seeds"]))
        mp_src.append(S(memit_files[l], "aggregate.within_probe_mean_across_seeds", mm["within_probe_mean_across_seeds"]))

    # KL ladder (3-seed L8)
    ladder = [(0.03, ["C3_klladder_003_L8_seeds_u5.json", "C3_klladder_003_L8_seeds_u4.json", "C3_klladder_003_L8_u2.json"]),
              (0.1, ["C3_null_ftkl_L8_v2.json"]),
              (0.3, ["C3_klladder_030_L8_seeds_u5.json", "C3_klladder_030_L8_seeds_u4.json", "C3_klladder_030_L8_u2.json"]),
              (1.0, ["C3_klladder_100_L8_seeds_u5.json", "C3_klladder_100_L8_seeds_u4.json", "C3_klladder_100_L8_u2.json"])]
    kx, ky, ksd, kl_src = [], [], [], []
    for w, files in ladder:
        fn = next((f for f in files if exists(f)), None)
        a = load(fn)["aggregate"]; srcs.append(fn)
        kx.append(w); ky.append(a["within_probe_mean_across_seeds"]); ksd.append(a.get("within_probe_std_across_seeds") or 0.0)
        kl_src.append(S(fn, "aggregate.within_probe_mean_across_seeds", a["within_probe_mean_across_seeds"]))

    espec_plot = "\n".join(
        rf"\addplot[only marks, {c}, mark=*, mark options={{fill={c}, scale=1.0}}] coordinates {{({fmt(dmg,3)},{fmt(rho,3)})}};"
        rf" \node[vizlabel, anchor={anch}, xshift={xsh}, yshift={ysh}] at (axis cs:{fmt(dmg,3)},{fmt(rho,3)}) {{{name}}};"
        for (name, dmg, rho, c, anch, xsh, ysh) in espec_pts)
    kl_err = " ".join(f"({fmt(w,2)},{fmt(v,3)}) +- (0,{fmt(e,3)})" for w, v, e in zip(kx, ky, ksd))
    rome_err = " ".join(f"({l},{fmt(v,3)}) +- (0,{fmt(e,3)})" for l, v, e in zip(layers, rome_l, rome_sd))
    memit_err = " ".join(f"({l},{fmt(v,3)}) +- (0,{fmt(e,3)})" for l, v, e in zip(layers, memit_l, memit_sd))

    tex = rf"""% =====================================================================
% fig:editor (F4).  Claim C3: geometry-predictability is ROME-specific.
% Regenerate: python edit-harness/experiments/make_figures_tex.py
% =====================================================================
\begin{{figure*}}[t]
\centering
\begin{{tikzpicture}}
\begin{{groupplot}}[group style={{group size=3 by 1, horizontal sep=1.55cm}},
  paperpanel, width=3.9cm, height=3.1cm]
% ---- (a) spectrum ----
\nextgroupplot[title={{(a) spectrum}}, xmode=log,
  xlabel={{mean damage (logit, L8 s0)}}, ylabel={{signed within-probe $\rho$}},
  ymin=-0.05, ymax=0.5, xmin=0.02, xmax=40]
{chr(10).join(espec_src)}
{espec_plot}
\draw[vizmuted, densely dotted, line width=0.4pt] (axis cs:0.02,0.10) -- (axis cs:40,0.10);
% ---- (b) ROME vs MEMIT across depth ----
\nextgroupplot[title={{(b) ROME/MEMIT}}, xlabel={{edited layer}},
  ylabel={{within-probe $\rho$}}, xtick={{8,10,12,14}}, ymin=-0.03, ymax=0.72,
  legend style={{at={{(0.98,0.98)}}, anchor=north east, draw=none, fill=none, font=\tiny, inner sep=1pt, row sep=-2pt}}, legend columns=1,
  ]
{chr(10).join(mp_src)}
\addplot+[vizblue, mark=*, mark options={{fill=vizblue, scale=0.6}}] plot[error bars/.cd, y dir=both, y explicit]
  coordinates {{{rome_err}}};
\addlegendentry{{ROME}}
\addplot+[vizyellow, mark=square*, mark options={{fill=vizyellow, scale=0.6}}] plot[error bars/.cd, y dir=both, y explicit]
  coordinates {{{memit_err}}};
\addlegendentry{{MEMIT}}
\draw[vizmuted, densely dotted, line width=0.4pt] (axis cs:8,0.10) -- (axis cs:14,0.10);
% ---- (c) KL-FT dose ladder (within-probe, L8, 3-seed) ----
\nextgroupplot[title={{(c) KL ladder}}, xmode=log,
  xlabel={{KL-FT weight}}, ylabel={{within-probe $\rho$}},
  xtick={{0.03,0.1,0.3,1.0}}, xticklabels={{0.03,0.1,0.3,1.0}}, log ticks with fixed point,
  ymin=0, ymax=0.22, xmin=0.025, xmax=1.2,
  scaled y ticks=false, yticklabel style={{/pgf/number format/fixed, /pgf/number format/precision=2}}]
{chr(10).join(kl_src)}
\addplot+[vizblue, mark=*, mark options={{fill=vizblue, scale=0.6}}] plot[error bars/.cd, y dir=both, y explicit]
  coordinates {{{kl_err}}};
\end{{groupplot}}
\end{{tikzpicture}}
\caption{{Editor dissociation. \textbf{{(a)}} the coupling/locality spectrum
(FT$\to$KL-FT$\to$ROME$\to$MEMIT$\to$AlphaEdit): only ROME's damage is geometry-ranked.
\textbf{{(b)}} ROME's within-probe $\rho$ peaks at L12 while MEMIT stays below the 0.10
DEAD floor at every depth (MEMIT quoted as $\rho_C$, never ``$S\times C$''; L10/L14
single-seed). \textbf{{(c)}} the KL-FT dose-response at L8 (3-seed).}}
\label{{fig:editor}}
\end{{figure*}}
"""
    write_fig("figD_editor", tex, srcs)


# ==========================================================================
# F5 deletion — depth profile | variants | collapse | transplant (2x2 groupplot)
# ==========================================================================
ARM = "known=True|edit_ok=False"


def fig_deletion():
    srcs = []

    # (a) depth profile 3-seed
    b8 = load("C3_u1_blockB_L8_seeds_u5.json")["aggregate"]; srcs.append("C3_u1_blockB_L8_seeds_u5.json")
    b14 = load("C3_u1_blockB_L14_seeds_u5.json")["aggregate"]; srcs.append("C3_u1_blockB_L14_seeds_u5.json")
    l12v = []
    for s in (0, 1, 2):
        fn = f"U1_E1_transplant_GATE_L12_s{s}.json"; srcs.append(fn)
        l12v.append(load(fn)["keycos"]["within_probe_mean"])
    l12m, l12sd, _ = seed_mean_std(l12v)
    g1 = load("G1_L12_analysis.json"); srcs.append("G1_L12_analysis.json")
    rref = g1["aggregate"]["within_probe_mean_across_seeds"]
    prof = [(8, b8["within_probe_mean_across_seeds"], b8["within_probe_std_across_seeds"]),
            (12, l12m, l12sd),
            (14, b14["within_probe_mean_across_seeds"], b14["within_probe_std_across_seeds"])]
    a_src = [S("C3_u1_blockB_L8_seeds_u5.json", "aggregate.within_probe_mean_across_seeds", b8["within_probe_mean_across_seeds"]),
             S("U1_E1_transplant_GATE_L12_s{0,1,2}.json", "mean(keycos.within_probe_mean)", l12m),
             S("C3_u1_blockB_L14_seeds_u5.json", "aggregate.within_probe_mean_across_seeds", b14["within_probe_mean_across_seeds"]),
             S("G1_L12_analysis.json", "aggregate.within_probe_mean_across_seeds (rewrite ref)", rref)]

    # (b) variants nondc/dc (seed 0, S x C DC-comparison)
    variants = [("refusal", "u1_gate_refusal_L12_s0.json"), ("eos", "u1_gate_eos_L12_s0.json"),
                ("suppress", "u1_gate_suppress_L12_s0.json")]
    vn, vd, b_src = [], [], []
    for lab, fn in variants:
        a = load(fn)["arms"][ARM]; srcs.append(fn)
        vn.append(a["nondc_rho"]); vd.append(a["dc_rho"])
        b_src.append(S(fn, f"arms['{ARM}'].nondc_rho", a["nondc_rho"]))
        b_src.append(S(fn, f"arms['{ARM}'].dc_rho", a["dc_rho"]))

    # (c) collapse: ROME-delete vs Alpha-delete (3-seed both)
    rome_dmg, rome_cpl, c_src = [], [], []
    for s in (0, 1, 2):
        fn = f"U1_E1_transplant_GATE_L12_s{s}.json"
        d = load(fn); srcs.append(fn)
        rome_dmg.append(d["mean_signed_damage"]); rome_cpl.append(d["SxC_within_probe_mean"])
    alp_dmg = []
    for s in (0, 1, 2):
        fn = f"U1_E1_transplant_GATE_alphadelete_L12_s{s}.json"
        if exists(fn):
            alp_dmg.append(load(fn)["mean_signed_damage"]); srcs.append(fn)
    db_m, db_sd, _ = seed_mean_std(rome_dmg)
    cb_m, cb_sd, _ = seed_mean_std(rome_cpl)
    da_m, da_sd, _ = seed_mean_std(alp_dmg)
    bd = load("C3_u1_blockD_alphadelete_seeds_u2.json")["aggregate"]; srcs.append("C3_u1_blockD_alphadelete_seeds_u2.json")
    ca_m, ca_sd = bd["within_probe_mean_across_seeds"], bd["within_probe_std_across_seeds"]
    c_src = [S("U1_E1_transplant_GATE_L12_s{0,1,2}.json", "mean(mean_signed_damage)", db_m),
             S("U1_E1_transplant_GATE_L12_s{0,1,2}.json", "mean(SxC_within_probe_mean)", cb_m),
             S("U1_E1_transplant_GATE_alphadelete_L12_s{0,1,2}.json", "mean(mean_signed_damage)", da_m),
             S("C3_u1_blockD_alphadelete_seeds_u2.json", "aggregate.within_probe_mean_across_seeds", ca_m)]

    # (d) transplant Delta-rho
    tp = [("L8 s0", "U1_E1_transplant_GATE_L8_s0.json"), ("L12 s0", "U1_E1_transplant_GATE_L12_s0.json"),
          ("L12 s1", "U1_E1_transplant_GATE_L12_s1.json"), ("L12 s2", "U1_E1_transplant_GATE_L12_s2.json"),
          ("L14 s0", "U1_E1_transplant_GATE_L14_s0.json")]
    tval, d_src = [], []
    for lab, fn in tp:
        d = load(fn); srcs.append(fn)
        tval.append(d["delta_rho_SxC_minus_best_transplant"])
        d_src.append(S(fn, "delta_rho_SxC_minus_best_transplant", d["delta_rho_SxC_minus_best_transplant"]))

    prof_err = " ".join(f"({l},{fmt(v,3)}) +- (0,{fmt(e,3)})" for l, v, e in prof)
    vx = [0, 1, 2]
    tx = list(range(len(tp)))
    txlab = ",".join(f"{{{lab}}}" for lab, _ in tp)

    tex = rf"""% =====================================================================
% fig:deletion (F5).  Section 7: deletion collateral is geometry-governed.
% Regenerate: python edit-harness/experiments/make_figures_tex.py
% =====================================================================
\begin{{figure*}}[tb]
\centering
\begin{{tikzpicture}}
\begin{{groupplot}}[group style={{group size=4 by 1, horizontal sep=1.3cm}},
  paperpanel, width=3.0cm, height=2.6cm]
% ---- (a) refusal-delete coupling across depth (3-seed) ----
\nextgroupplot[title={{(a) depth}}, xlabel={{layer}},
  ylabel={{within-probe $\rho$}}, xtick={{8,12,14}}, ymin=0, ymax=0.85, xmin=7, xmax=15]
{chr(10).join(a_src)}
\addplot+[vizseqblue, mark=*, mark options={{fill=vizseqblue, scale=0.7}}] plot[error bars/.cd, y dir=both, y explicit]
  coordinates {{{prof_err}}};
\draw[vizcrit, dashed, line width=0.6pt] (axis cs:7,{fmt(rref,3)}) -- (axis cs:15,{fmt(rref,3)});
\node[vizlabelm, anchor=south east, text=vizcrit] at (axis cs:14.9,0.03) {{rewrite $\rho{{=}}{fmt(rref,3)}$}};
% ---- (b) variants nondc vs dc ----
\nextgroupplot[title={{(b) variants}}, ybar, bar width=6pt,
  ylabel={{$\rho$ (s0)}}, xtick={{0,1,2}}, xticklabels={{refusal,eos,suppress}},
  x tick label style={{rotate=40, anchor=east, font=\tiny}}, ymin=0, ymax=0.85, xmin=-0.6, xmax=2.6,
  legend style={{at={{(0.98,0.99)}}, anchor=north east, draw=none, fill=none, font=\tiny, inner sep=1pt, row sep=-2pt}}, legend columns=1, ]
{chr(10).join(b_src)}
\addplot+[draw=vizseqblue, fill=vizseqblue] coordinates {{{coords((i, fmt(v,3)) for i, v in zip(vx, vn))}}};
\addlegendentry{{raw}}
\addplot+[draw=vizaqua, fill=vizaqua] coordinates {{{coords((i, fmt(v,3)) for i, v in zip(vx, vd))}}};
\addlegendentry{{DC}}
% ---- (c) collapse (3-seed) ----
\nextgroupplot[title={{(c) collapse}}, ybar, bar width=6pt,
  ylabel={{value}}, xtick={{0,1}}, xticklabels={{ROME-del,Alpha-del}},
  x tick label style={{rotate=40, anchor=east, font=\tiny}}, ymin=0, ymax=4.9, xmin=-0.6, xmax=1.6,
  legend style={{at={{(0.98,0.99)}}, anchor=north east, draw=none, fill=none, font=\tiny, inner sep=1pt, row sep=-2pt}}, legend columns=1, ]
{chr(10).join(c_src)}
\addplot+[draw=vizorange, fill=vizorange] plot[error bars/.cd, y dir=both, y explicit]
  coordinates {{(0,{fmt(db_m,2)}) +- (0,{fmt(db_sd,2)}) (1,{fmt(da_m,2)}) +- (0,{fmt(da_sd,2)})}};
\addlegendentry{{mean dmg}}
\addplot+[draw=vizseqblue, fill=vizseqblue] plot[error bars/.cd, y dir=both, y explicit]
  coordinates {{(0,{fmt(cb_m,3)}) +- (0,{fmt(cb_sd,3)}) (1,{fmt(ca_m,3)}) +- (0,{fmt(ca_sd,3)})}};
\addlegendentry{{$S\times C$ $\rho$}}
% ---- (d) transplant gate Delta-rho ----
\nextgroupplot[title={{(d) transplant}}, ybar, bar width=6pt,
  ylabel={{$\Delta\rho$ transpl.}}, xtick={{{",".join(map(str, tx))}}}, xticklabels={{{txlab}}},
  x tick label style={{rotate=42, anchor=east, font=\tiny}}, ymin=0, ymax=0.75, xmin=-0.6, xmax={len(tp) - 0.4}]
{chr(10).join(d_src)}
\addplot+[draw=vizviolet, fill=vizviolet] coordinates {{{coords((i, fmt(v,3)) for i, v in zip(tx, tval))}}};
\end{{groupplot}}
\end{{tikzpicture}}
\caption{{Deletion collateral is geometry-governed. \textbf{{(a)}} refusal-deletion
coupling across depth (3-seed) peaks at L12, above the rewrite reference $\rho{{=}}{fmt(rref,3)}$.
\textbf{{(b)}} eos is robust and suppress is the DC-fragile variant ($0.621\!\to\!0.159$
under double-centering). \textbf{{(c)}} AlphaEdit-delete collapses both damage and coupling
(3-seed; $\alpha$-damage $\approx{fmt(da_m,2)}$). \textbf{{(d)}} $S\times C$ beats a lexical
transplant baseline at every cell.}}
\label{{fig:deletion}}
\end{{figure*}}
"""
    write_fig("figE_deletion", tex, srcs)


# ==========================================================================
# F6 sequential — survival curves + position fragility (descriptive)
# ==========================================================================
def fig_sequential():
    srcs = []
    base = load("SEQ_analysis_L12.json"); srcs.append("SEQ_analysis_L12.json")
    d = base
    if exists("SEQ_analysis_L12_4stream.json"):
        four = load("SEQ_analysis_L12_4stream.json")
        consistent = all(
            o["npz"] == n["npz"] and o["final_survival_frac"] == n["final_survival_frac"]
            and abs(o["position_fragility"]["rho_position_vs_survival"] - n["position_fragility"]["rho_position_vs_survival"]) <= 1e-4
            for o, n in zip(base["per_stream"], four["per_stream"]))
        if consistent:
            d = four; srcs.append("SEQ_analysis_L12_4stream.json")
    streams = d["per_stream"]; pooled = d["pooled"]
    src_tag = "SEQ_analysis_L12_4stream.json" if d is not base else "SEQ_analysis_L12.json"
    src_tag_tex = src_tag.replace("_", r"\_")
    colors = ["vizblue", "vizaqua", "vizyellow", "vizgreen"]
    curve_plots, cs_src = [], []
    for i, s in enumerate(streams):
        pts = [(c["checkpoint_nedits"], c["frac_survived"]) for c in s["survival_curve"]]
        for c in s["survival_curve"]:
            cs_src.append(S(src_tag, f"per_stream[{i}].survival_curve[{c['checkpoint_nedits']}].frac_survived", c["frac_survived"]))
        curve_plots.append(rf"\addplot+[{colors[i]}, mark=*, mark options={{fill={colors[i]}, scale=0.5}}] "
                           rf"coordinates {{{coords((x, fmt(y,3)) for x, y in pts)}}};")
        curve_plots.append(rf"\addlegendentry{{s{i}}}")
    labs, vals, ps, pf_src = [], [], [], []
    for i, s in enumerate(streams):
        pf = s["position_fragility"]
        labs.append(f"s{i}"); vals.append(pf["rho_position_vs_survival"]); ps.append(pf["perm_p"])
        pf_src.append(S(src_tag, f"per_stream[{i}].position_fragility.rho_position_vs_survival", pf["rho_position_vs_survival"]))
    labs.append("pool"); vals.append(pooled["position_fragility"]["rho_position_vs_survival"]); ps.append(pooled["position_fragility"]["perm_p"])
    pf_src.append(S(src_tag, "pooled.position_fragility.rho_position_vs_survival", pooled["position_fragility"]["rho_position_vs_survival"]))
    px = list(range(len(vals)))
    ymax_pf = max(0.42, max(vals) * 1.35)
    tex = rf"""% =====================================================================
% fig:sequential (F6).  Section 8: sequential no-restore (DESCRIPTIVE only;
% no geometry-attribution panel -- H1 UNSETTLED, hard constraint).
% Regenerate: python edit-harness/experiments/make_figures_tex.py
% =====================================================================
\begin{{figure}}[tb]
\centering
\begin{{tikzpicture}}
\begin{{groupplot}}[group style={{group size=2 by 1, horizontal sep=1.5cm}},
  paperpanel, width=2.85cm, height=2.55cm]
\nextgroupplot[title={{(a) survival}},
  xlabel={{edits applied}}, ylabel={{fraction surviving}}, ymin=0, ymax=0.82,
  legend style={{at={{(0.99,0.99)}}, anchor=north east, draw=none, fill=none, font=\tiny, inner sep=1pt, row sep=-2pt, /tikz/every even column/.append style={{column sep=3pt}}}}, legend columns=2,
  ]
{chr(10).join(cs_src)}
{chr(10).join(curve_plots)}
\nextgroupplot[title={{(b) fragility}}, ybar, bar width=7pt,
  ylabel={{$\rho$(pos, surv)}}, xtick={{{",".join(map(str, px))}}}, xticklabels={{{",".join(labs)}}},
  x tick label style={{rotate=30, anchor=east, font=\tiny}},
  ymin=0, ymax={fmt(ymax_pf,2)}, xmin=-0.6, xmax={len(vals) - 0.4}]
{chr(10).join(pf_src)}
\addplot+[draw=vizseqblue, fill=vizseqblue] coordinates {{{coords((i, fmt(v,3)) for i, v in zip(px, vals))}}};
\end{{groupplot}}
\end{{tikzpicture}}
\caption{{Sequential no-restore stress (descriptive). Survival collapses to an
ordering-dependent range after 50 edits, with modest position fragility (pooled
$\rho{{=}}{fmt(pooled['position_fragility']['rho_position_vs_survival'],3)}$). The pre-registered
geometry-attribution gate is not passed; no geometry panel is shown.}}
\label{{fig:sequential}}
\end{{figure}}
"""
    write_fig("figF_sequential", tex, srcs)


def main():
    os.makedirs(OUT, exist_ok=True)
    fig_lawtransfer()
    fig_surrogate()
    fig_causal()
    fig_editor()
    fig_deletion()
    fig_sequential()
    with open(os.path.join(OUT, "PROVENANCE-TEX.txt"), "w") as f:
        f.write("B6 TeX-auditable figure set — provenance (figure -> canonical results/*.json)\n")
        f.write("Generated by edit-harness/experiments/make_figures_tex.py (deterministic).\n")
        f.write("Every \\addplot in each .tex is preceded by % SOURCE lines citing the exact\n")
        f.write("JSON path + field + plotted value; audit by reading .tex against results/*.json.\n")
        f.write("=" * 72 + "\n\n")
        for name, ss in _PROV:
            f.write(f"{name}.tex  <-  {', '.join(ss)}\n")
    print(f"\nWrote {len(_PROV)} figure .tex files + PROVENANCE-TEX.txt to {OUT}")


if __name__ == "__main__":
    main()
