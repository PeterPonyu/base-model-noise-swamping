#!/usr/bin/env Rscript
# =====================================================================
# make_figures.R — R/ggplot2 -> tikzDevice figure set for the B6 ARR
# submission. Parallel alternative to the pgfplots set in ../figures-tex/.
#
# SAME canonical-JSON-only rule as make_figures_tex.py: every plotted
# number is read at run time from edit-harness/results/*.json (nothing
# hardcoded but axis labels, titles, and the DEAD/reference constants
# already fixed in the paper). Above each emitted fig{A..F}.tex a
# provenance comment block cites the exact JSON path + field + value per
# series, so auditability is retained at the file level.
#
# Palette: the dataviz-skill validated reference categorical instance,
# used in the SAME fixed slot order as _pgfpreamble.tex (blue, aqua,
# yellow, green, violet, red, ...) so the two figure sets are directly
# comparable and the paper's color semantics (blue=+/key-cos, red=-,
# grey=DEAD) are preserved across engines.
#
# Usage:  Rscript figures-r/make_figures.R    # -> figures-r/fig{A..F}.tex
# =====================================================================

suppressPackageStartupMessages({
  library(jsonlite)
  library(ggplot2)
  library(patchwork)
  library(tikzDevice)
  library(scales)
})

HERE    <- tryCatch(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))),
                    error = function(e) ".")
if (length(HERE) == 0 || HERE == "") HERE <- "."
RESULTS <- normalizePath(file.path(HERE, "..", "..", "edit-harness", "results"))
OUT     <- normalizePath(file.path(HERE), mustWork = FALSE)

# ---- dataviz reference palette (fixed slot order; matches _pgfpreamble.tex) ----
viz <- list(
  blue    = "#2A78D6", aqua = "#1BAF7A", yellow = "#EDA100", green = "#008300",
  violet  = "#4A3AA7", red  = "#E34948", magenta = "#E87BA4", orange = "#EB6834",
  seqblue = "#2A78D6", ink  = "#0B0B0B", ink2 = "#333333", tick = "#666666",
  muted   = "#898781", grid = "#E1E0D9", crit = "#D03B3B"
)

# ---------------------------------------------------------------------
# JSON loading (pure nested lists, mirrors python dict/list access)
# ---------------------------------------------------------------------
.load <- function(name) {
  p <- file.path(RESULTS, name)
  if (!file.exists(p)) stop("missing results JSON: ", name)
  fromJSON(readLines(p, warn = FALSE), simplifyVector = FALSE)
}
.exists <- function(name) file.exists(file.path(RESULTS, name))
.first_existing <- function(names) { for (n in names) if (.exists(n)) return(n); NULL }
.num <- function(x) as.numeric(x)
# population mean/std over non-null (matches harness within_probe_std convention)
seed_mean_std <- function(vals) {
  vals <- .num(unlist(vals)); vals <- vals[!is.na(vals)]
  if (!length(vals)) return(c(mean = NA, sd = 0))
  m <- mean(vals)
  sd <- if (length(vals) > 1) sqrt(sum((vals - m)^2) / length(vals)) else 0
  c(mean = m, sd = sd)
}

# ---- provenance: one audit line per plotted series (== python S()) ----
.prov <- character(0)
S <- function(basename, field, value) {
  v <- if (is.numeric(value)) formatC(value, format = "f", digits = 4) else value
  .prov[[length(.prov) + 1L]] <<- sprintf("%% SOURCE: results/%s :: %s = %s", basename, field, v)
  invisible(value)
}
prov_reset <- function() .prov <<- character(0)

# ---------------------------------------------------------------------
# shared professional theme (theme_minimal-derived)
#   - subtle y-major gridlines only (matches pgfplots ymajorgrids)
#   - clean left-aligned bold panel titles, muted ticks, no axis lines
#   - legends OUTSIDE the panel (top), compact keys
# ---------------------------------------------------------------------
theme_b6 <- function(base = 9) {
  # All TEXT wears near-black ink (holds up at ACL 3.03in column width);
  # grey is reserved for gridlines ONLY. Ticks are a mid-grey (structure, not ink).
  theme_minimal(base_size = base) +
    theme(
      text              = element_text(colour = viz$ink),
      plot.title        = element_text(hjust = 0, face = "bold", size = base,
                                       colour = viz$ink, margin = margin(b = 2)),
      axis.title.x      = element_text(size = base - 1, colour = viz$ink, margin = margin(t = 1)),
      axis.title.y      = element_text(size = base - 1, colour = viz$ink, margin = margin(r = 1)),
      axis.text         = element_text(size = base - 2, colour = viz$ink2),
      panel.grid.major.y = element_line(colour = viz$grid, linewidth = 0.25),
      panel.grid.major.x = element_blank(),
      panel.grid.minor  = element_blank(),
      axis.ticks        = element_line(colour = viz$tick, linewidth = 0.3),
      axis.ticks.length = unit(2, "pt"),
      axis.line         = element_blank(),
      legend.position   = "top",
      legend.justification = "center",
      legend.title      = element_blank(),
      legend.text       = element_text(size = base - 2, colour = viz$ink),
      legend.key.size   = unit(7, "pt"),
      legend.key.height = unit(6, "pt"),
      legend.spacing.x  = unit(2, "pt"),
      legend.margin     = margin(0, 0, 0, 0),
      legend.box.margin = margin(0, 0, -6, 0),
      legend.box.spacing = unit(1, "pt"),
      plot.margin       = margin(2, 5, 2, 2)
    )
}
# small helpers -------------------------------------------------------
lbl_geom  <- function(x, y, lab, ...) annotate("text", x = x, y = y, label = lab,
                                               colour = viz$ink, size = 2.1, ...)
dead_line <- function(y = 0.10, xr = NULL) geom_hline(yintercept = y, colour = viz$muted,
                                                      linetype = "dotted", linewidth = 0.3)
errbar    <- function(...) geom_errorbar(..., width = 0, colour = viz$muted, linewidth = 0.35)

# =====================================================================
# FigA  (F1 + F7): depth law | regime | signed cross-arch | magnitude
# =====================================================================
fig_A <- function() {
  prov_reset(); L <- c(8, 10, 12, 14)
  # (a) key-cos vs norm-growth across depth
  kc <- kcsd <- ng <- numeric(0)
  for (l in L) {
    d <- .load(sprintf("G1_L%d_analysis.json", l)); a <- d[["aggregate"]]
    kc  <- c(kc,  S(sprintf("G1_L%d_analysis.json", l), "aggregate.within_probe_mean_across_seeds", .num(a[["within_probe_mean_across_seeds"]])))
    kcsd<- c(kcsd, .num(a[["within_probe_std_across_seeds"]]))
    ngs <- sapply(d[["per_seed"]], function(s) .num(s[["within_probe_mean_normgrowth"]]))
    ng  <- c(ng,  S(sprintf("G1_L%d_analysis.json", l), "mean(per_seed[].within_probe_mean_normgrowth)", mean(ngs)))
  }
  da <- rbind(
    data.frame(layer = L, value = kc, sd = kcsd, series = "key-cos $|C|$"),
    data.frame(layer = L, value = ng, sd = 0,    series = "norm-growth"))
  da$series <- factor(da$series, levels = c("key-cos $|C|$", "norm-growth"))
  pa <- ggplot(da, aes(layer, value, colour = series, shape = series)) +
    dead_line() +
    errbar(data = subset(da, sd > 0), aes(ymin = value - sd, ymax = value + sd)) +
    geom_line(linewidth = 0.5) + geom_point(size = 1.5) +
    lbl_geom(8.4, 0.135, "DEAD 0.10", hjust = 0, colour = viz$muted, size = 1.9) +
    scale_colour_manual(values = c("key-cos $|C|$" = viz$blue, "norm-growth" = viz$orange)) +
    scale_shape_manual(values = c("key-cos $|C|$" = 16, "norm-growth" = 15)) +
    scale_x_continuous(breaks = L) +
    coord_cartesian(ylim = c(0, 0.72), xlim = c(7.4, 14.6)) +
    labs(title = "(a) depth", x = "edited layer", y = "within-probe $\\rho$") +
    theme_b6() + theme(legend.key.width = unit(11, "pt"))

  # (b) regime: 3B L24, 8B L16/L24/L28
  d3 <- .load("C3_regime_3b_L24_r4.json")
  r3 <- S("C3_regime_3b_L24_r4.json", "aggregate.within_probe_mean_across_seeds",
          .num(d3[["aggregate"]][["within_probe_mean_across_seeds"]]))
  d8 <- .load("C3_llama8b_r3.json")
  l8b <- sapply(d8[["per_seed"]], function(s) {
    lay <- as.integer(sub(".*_L(\\d+)_.*", "\\1", s[["npz"]])); setNames(.num(s[["within_probe_mean"]]), lay) })
  for (lay in c(16, 24, 28)) S("C3_llama8b_r3.json", sprintf("per_seed[L%d].within_probe_mean", lay), l8b[[as.character(lay)]])
  db <- data.frame(
    x = factor(c("3B/L24", "8B/L16", "8B/L24", "8B/L28"),
               levels = c("3B/L24", "8B/L16", "8B/L24", "8B/L28")),
    value = c(r3, l8b[["16"]], l8b[["24"]], l8b[["28"]]))
  db$fill <- ifelse(db$value >= 0, viz$blue, viz$red)
  pb <- ggplot(db, aes(x, value, fill = fill)) +
    geom_col(width = 0.62) + geom_hline(yintercept = 0, colour = viz$muted, linewidth = 0.3) +
    scale_fill_identity() + coord_cartesian(ylim = c(-0.25, 0.5)) +
    labs(title = "(b) regime", x = NULL, y = "signed $\\rho$") +
    theme_b6() + theme(axis.text.x = element_text(angle = 42, hjust = 1, size = 6.5))

  # (c) signed law per architecture family
  fams <- list(
    c("Llama-1B", "G1_L12_analysis.json"), c("Llama-3B", "C3_null_llama3b_L14.json"),
    c("gemma",    "C3_null_gemma2b_L13_v2.json"), c("Phi-3.5", "C3_null_phi35_L16_v2.json"),
    c("Qwen-0.5B","C3_null_qwen05b_L12.json"), c("Qwen-1.5B", "C3_null_qwen15b_L14.json"),
    c("Qwen-3B",  "C3_null_qwen3b_L18_v2.json"))
  fv <- fsd <- numeric(0); fl <- character(0)
  for (f in fams) {
    a <- .load(f[2])[["aggregate"]]
    fv  <- c(fv,  S(f[2], "aggregate.within_probe_mean_across_seeds", .num(a[["within_probe_mean_across_seeds"]])))
    fsd <- c(fsd, .num(a[["within_probe_std_across_seeds"]])); fl <- c(fl, f[1])
  }
  dc <- data.frame(x = factor(fl, levels = fl), value = fv, sd = fsd,
                   fill = ifelse(fv >= 0, viz$blue, viz$red))
  pc <- ggplot(dc, aes(x, value, fill = fill)) +
    geom_col(width = 0.68) + geom_hline(yintercept = 0, colour = viz$muted, linewidth = 0.3) +
    errbar(data = subset(dc, sd > 0), aes(ymin = value - sd, ymax = value + sd)) +
    scale_fill_identity() + coord_cartesian(ylim = c(-0.28, 0.72)) +
    labs(title = "(c) signed", x = NULL, y = "signed $\\rho$") +
    theme_b6() + theme(axis.text.x = element_text(angle = 42, hjust = 1, size = 6.5))

  # (d) magnitude law
  mt <- .load("C1_magnitude_table.json")
  by <- setNames(mt[["families"]], sapply(mt[["families"]], function(r) r[["family"]]))
  want <- list(c("Llama-1B","llama1b_L12"), c("gemma","gemma2b_L13"), c("Phi-3.5","phi35_L16"),
               c("Qwen-0.5B","qwen05b_L12"), c("Qwen-1.5B","qwen15b_L14"), c("Qwen-3B","qwen3b_L18"))
  mv <- msd <- numeric(0); ml <- character(0); mdead <- logical(0)
  for (w in want) {
    r <- by[[w[2]]]; v <- .num(r[["within_probe_mean_abs_across_seeds"]])
    e <- .num(r[["within_probe_std_abs_across_seeds"]]); if (length(e) == 0 || is.na(e)) e <- 0
    dd <- toupper(substr(r[["VERDICT"]], 1, 4)) == "DEAD" || v < 0.10
    mv <- c(mv, S("C1_magnitude_table.json", sprintf("families[family=%s].within_probe_mean_abs_across_seeds", w[2]), v))
    msd <- c(msd, e); ml <- c(ml, w[1]); mdead <- c(mdead, dd)
  }
  dd <- data.frame(x = factor(ml, levels = ml), value = mv, sd = msd,
                   fill = ifelse(mdead, viz$muted, viz$seqblue))
  pd <- ggplot(dd, aes(x, value, fill = fill)) +
    dead_line() + geom_col(width = 0.68) +
    errbar(data = subset(dd, sd > 0), aes(ymin = value - sd, ymax = value + sd)) +
    scale_fill_identity() + coord_cartesian(ylim = c(0, 0.72)) +
    labs(title = "(d) magnitude", x = NULL, y = "$\\rho(|C|,|$dmg$|)$") +
    theme_b6() + theme(axis.text.x = element_text(angle = 42, hjust = 1, size = 6.5))

  list(plot = (pa | pb) / (pc | pd), prov = .prov)
}

# =====================================================================
# FigB  surrogate: S x C == GradSim identity scatter
# =====================================================================
fig_B <- function() {
  prov_reset()
  c1 <- .load("C1_mechanism_sc_table.json")
  sc <- list()
  for (g in c1[["groups"]]) if (g[["model"]] == "llama1b") sc[[as.character(g[["layer"]])]] <- .num(g[["within_probe_rho_SC"]])
  pts <- data.frame()
  for (l in c(8, 10, 12, 14)) {
    x <- S("C1_mechanism_sc_table.json", sprintf("groups[llama1b,L%d].within_probe_rho_SC", l), sc[[as.character(l)]])
    y <- S(sprintf("G2_gradsim_L%d.json", l), "aggregate.gradsim_within_probe.resid.mean",
           .num(.load(sprintf("G2_gradsim_L%d.json", l))[["aggregate"]][["gradsim_within_probe"]][["resid"]][["mean"]]))
    pts <- rbind(pts, data.frame(x = x, y = y, lab = sprintf("L%d", l)))
  }
  p <- ggplot(pts, aes(x, y)) +
    geom_abline(slope = 1, intercept = 0, colour = viz$muted, linetype = "dashed", linewidth = 0.35) +
    geom_point(colour = viz$blue, size = 2.1) +
    geom_text(aes(label = lab), colour = viz$ink, size = 2.2, hjust = -0.35, vjust = 1.35) +
    lbl_geom(0.315, 0.675, "identity: $\\rho_{SC}{=}$GradSim", hjust = 0, colour = viz$muted, size = 2.0) +
    coord_cartesian(xlim = c(0.30, 0.70), ylim = c(0.30, 0.70)) +
    scale_x_continuous(breaks = seq(0.3, 0.7, 0.1)) + scale_y_continuous(breaks = seq(0.3, 0.7, 0.1)) +
    labs(title = "(F2) $S\\times C \\approx$ GradSim, no backprop",
         x = "$S\\times C$ surrogate $\\rho$", y = "GradSim-residual $\\rho$") +
    theme_b6() + theme(panel.grid.major.x = element_line(colour = viz$grid, linewidth = 0.25))
  list(plot = p, prov = .prov)
}

# =====================================================================
# FigC  causal: quartile damage-removed | by-construction vs holdout
# =====================================================================
fig_C <- function() {
  prov_reset()
  c4 <- .load("C4_causal_table.json"); ho <- .load("C4_causal_holdout_table_3seed.json")
  layers <- c(8, 10, 12, 14); cols <- c(L8 = viz$blue, L10 = viz$aqua, L12 = viz$yellow, L14 = viz$green)
  qd <- data.frame()
  for (l in layers) {
    q <- c4[["layers"]][[as.character(l)]][["quartile_means"]]
    for (x in q) {
      mc <- .num(x[["mean_cos"]]); dr <- .num(x[["mean_damage_removed"]])
      S("C4_causal_table.json", sprintf("layers.%d.quartile_means[cos=%.3f].mean_damage_removed", l, mc), dr)
      qd <- rbind(qd, data.frame(cos = mc, dmg = dr, layer = sprintf("L%d", l)))
    }
  }
  qd$layer <- factor(qd$layer, levels = names(cols))
  ends <- do.call(rbind, lapply(split(qd, qd$layer), function(s) s[which.max(s$cos), ]))
  pa <- ggplot(qd, aes(cos, dmg, colour = layer)) +
    geom_line(linewidth = 0.5) + geom_point(size = 1.2) +
    geom_text(data = ends, aes(label = layer), colour = viz$ink, size = 2.0, hjust = -0.3) +
    scale_colour_manual(values = cols, guide = "none") +
    coord_cartesian(xlim = c(0.08, 0.75), ylim = c(0.5, 7.4)) +
    labs(title = "(a) removed", x = "pre-edit key-cosine", y = "AlphaEdit damage removed") +
    theme_b6()

  hol <- sort(as.integer(names(ho[["layers"]])))
  hb <- data.frame()
  for (l in hol) {
    b <- S("C4_causal_table.json", sprintf("layers.%d.within_probe_spearman", l),
           .num(c4[["layers"]][[as.character(l)]][["within_probe_spearman"]]))
    h <- S("C4_causal_holdout_table_3seed.json", sprintf("layers.%d.within_probe_spearman", l),
           .num(ho[["layers"]][[as.character(l)]][["within_probe_spearman"]]))
    hb <- rbind(hb,
                data.frame(layer = sprintf("L%d", l), series = "by-construction", value = b),
                data.frame(layer = sprintf("L%d", l), series = "holdout", value = h))
  }
  hb$layer  <- factor(hb$layer, levels = sprintf("L%d", hol))   # keep paper order L8,L12 (not alpha)
  hb$series <- factor(hb$series, levels = c("by-construction", "holdout"))
  pb <- ggplot(hb, aes(layer, value, fill = series)) +
    geom_col(position = position_dodge(width = 0.72), width = 0.62) +
    scale_fill_manual(values = c("by-construction" = viz$seqblue, "holdout" = viz$aqua)) +
    coord_cartesian(ylim = c(0, 0.72)) +
    labs(title = "(b) $\\approx$ holdout", x = NULL, y = "within-probe $\\rho$") +
    theme_b6()
  list(plot = pa | pb, prov = .prov)
}

# =====================================================================
# FigD  editor: spectrum | ROME-vs-MEMIT depth | KL ladder
# =====================================================================
fig_D <- function() {
  prov_reset()
  mdmg <- function(g) .num(.load(g)[["KNOWN_PROBES"]][["mean_damage_logit"]])
  ft   <- .num(.load("C3_null_ft_L8.json")[["aggregate"]][["within_probe_mean_across_seeds"]])
  ftkl <- .num(.load("C3_null_ftkl_L8_v2.json")[["aggregate"]][["within_probe_mean_across_seeds"]])
  rome <- .load("C1_mechanism_sc_table.json")
  rome_rho <- NA; for (g in rome[["groups"]]) if (g[["model"]] == "llama1b" && g[["layer"]] == 8) rome_rho <- .num(g[["within_probe_rho_C"]])
  memit <- .num(.load("C3_memit_L8_r3.json")[["aggregate"]][["within_probe_mean_across_seeds"]])
  c4 <- .load("C4_causal_holdout_table_3seed.json")
  # (name, y=rho, x=mean-damage, colour, y-anchor vjust)
  spec <- data.frame(
    name = c("FT", "KL-FT", "ROME", "MEMIT", "AlphaEdit"),
    x = c(mdmg("gate_llama1b_ft_cf_L8_s0.json"), mdmg("gate_llama1b_ftkl_cf_L8_s0.json"),
          mdmg("gate_llama1b_rome_cf_L8_s0.json"), mdmg("gate_llama1b_memit_cf_L8_s0.json"),
          .num(c4[["layers"]][["8"]][["mean_damage_alpha"]])),
    y = c(ft, ftkl, rome_rho, memit, 0.0),
    col = c(viz$blue, viz$aqua, viz$yellow, viz$green, viz$violet),
    vj = c(-0.8, -0.8, -0.8, -0.9, 0.4), hj = c(0.5, 0.5, 0.5, 0.5, -0.18))
  S("C3_null_ft_L8.json", "aggregate.within_probe_mean_across_seeds [y=rho]", ft)
  S("gate_llama1b_ft_cf_L8_s0.json", "KNOWN_PROBES.mean_damage_logit [x=mean damage]", spec$x[1])
  S("C3_null_ftkl_L8_v2.json", "aggregate.within_probe_mean_across_seeds [y=rho]", ftkl)
  S("gate_llama1b_ftkl_cf_L8_s0.json", "KNOWN_PROBES.mean_damage_logit [x=mean damage]", spec$x[2])
  S("C1_mechanism_sc_table.json", "groups[llama1b,L8].within_probe_rho_C [y=rho]", rome_rho)
  S("gate_llama1b_rome_cf_L8_s0.json", "KNOWN_PROBES.mean_damage_logit [x=mean damage]", spec$x[3])
  S("C3_memit_L8_r3.json", "aggregate.within_probe_mean_across_seeds [y=rho]", memit)
  S("gate_llama1b_memit_cf_L8_s0.json", "KNOWN_PROBES.mean_damage_logit [x=mean damage]", spec$x[4])
  S("C4_causal_holdout_table_3seed.json", "layers.8: AlphaEdit coupling ~0 by design (floor) [y=rho]", 0.0)
  S("C4_causal_holdout_table_3seed.json", "layers.8.mean_damage_alpha [x=mean damage]", spec$x[5])
  pa <- ggplot(spec, aes(x, y)) +
    dead_line() +
    geom_point(aes(colour = col), size = 2.4) +
    geom_text(aes(label = name, vjust = vj, hjust = hj), colour = viz$ink, size = 2.0) +
    scale_colour_identity() +
    scale_x_log10(limits = c(0.02, 40), breaks = c(0.1, 1, 10), labels = c("0.1", "1", "10")) +
    coord_cartesian(ylim = c(-0.05, 0.5)) +
    labs(title = "(a) spectrum", x = "mean damage (logit, L8 s0)", y = "signed within-probe $\\rho$") +
    theme_b6()

  # (b) ROME vs MEMIT across depth
  mfiles <- c(`8` = "C3_memit_L8_r3.json", `10` = "C3_memit_L10_u4.json",
              `12` = "C3_memit_L12_r3.json", `14` = "C3_memit_L14_u4.json")
  layers <- c(8, 10, 12, 14); rm <- data.frame()
  for (l in layers) {
    g <- .load(sprintf("G1_L%d_analysis.json", l))[["aggregate"]]
    m <- .load(mfiles[[as.character(l)]])[["aggregate"]]
    rv <- S(sprintf("G1_L%d_analysis.json", l), "aggregate.within_probe_mean_across_seeds", .num(g[["within_probe_mean_across_seeds"]]))
    mv <- S(mfiles[[as.character(l)]], "aggregate.within_probe_mean_across_seeds", .num(m[["within_probe_mean_across_seeds"]]))
    rm <- rbind(rm,
      data.frame(layer = l, value = rv, sd = .num(g[["within_probe_std_across_seeds"]]), series = "ROME"),
      data.frame(layer = l, value = mv, sd = .num(m[["within_probe_std_across_seeds"]]), series = "MEMIT"))
  }
  rm$series <- factor(rm$series, levels = c("ROME", "MEMIT"))
  pb <- ggplot(rm, aes(layer, value, colour = series, shape = series)) +
    dead_line() +
    errbar(data = subset(rm, sd > 0), aes(ymin = value - sd, ymax = value + sd)) +
    geom_line(linewidth = 0.5) + geom_point(size = 1.5) +
    scale_colour_manual(values = c("ROME" = viz$blue, "MEMIT" = viz$yellow)) +
    scale_shape_manual(values = c("ROME" = 16, "MEMIT" = 15)) +
    scale_x_continuous(breaks = layers) + coord_cartesian(ylim = c(-0.03, 0.72)) +
    labs(title = "(b) ROME/MEMIT", x = "edited layer", y = "within-probe $\\rho$") +
    theme_b6()

  # (c) KL ladder
  ladder <- list(list(0.03, c("C3_klladder_003_L8_seeds_u5.json","C3_klladder_003_L8_seeds_u4.json","C3_klladder_003_L8_u2.json")),
                 list(0.10, c("C3_null_ftkl_L8_v2.json")),
                 list(0.30, c("C3_klladder_030_L8_seeds_u5.json","C3_klladder_030_L8_seeds_u4.json","C3_klladder_030_L8_u2.json")),
                 list(1.00, c("C3_klladder_100_L8_seeds_u5.json","C3_klladder_100_L8_seeds_u4.json","C3_klladder_100_L8_u2.json")))
  kd <- data.frame()
  for (it in ladder) {
    fn <- .first_existing(it[[2]]); a <- .load(fn)[["aggregate"]]
    v <- S(fn, "aggregate.within_probe_mean_across_seeds", .num(a[["within_probe_mean_across_seeds"]]))
    e <- .num(a[["within_probe_std_across_seeds"]]); if (length(e) == 0 || is.na(e)) e <- 0
    kd <- rbind(kd, data.frame(w = it[[1]], value = v, sd = e))
  }
  pc <- ggplot(kd, aes(w, value)) +
    errbar(data = subset(kd, sd > 0), aes(ymin = value - sd, ymax = value + sd)) +
    geom_line(colour = viz$blue, linewidth = 0.5) + geom_point(colour = viz$blue, size = 1.5) +
    scale_x_log10(breaks = c(0.03, 0.1, 0.3, 1.0), labels = c("0.03","0.1","0.3","1.0")) +
    coord_cartesian(ylim = c(0, 0.22), xlim = c(0.025, 1.2)) +
    labs(title = "(c) KL ladder", x = "KL-FT weight", y = "within-probe $\\rho$") +
    theme_b6()
  list(plot = pa | pb | pc, prov = .prov)
}

# =====================================================================
# FigE  deletion: depth | variants | collapse | transplant
# =====================================================================
ARM <- "known=True|edit_ok=False"
fig_E <- function() {
  prov_reset()
  # (a) depth profile 3-seed
  b8  <- .load("C3_u1_blockB_L8_seeds_u5.json")[["aggregate"]]
  b14 <- .load("C3_u1_blockB_L14_seeds_u5.json")[["aggregate"]]
  l12v <- sapply(0:2, function(s) .num(.load(sprintf("U1_E1_transplant_GATE_L12_s%d.json", s))[["keycos"]][["within_probe_mean"]]))
  l12 <- seed_mean_std(l12v)
  rref <- .num(.load("G1_L12_analysis.json")[["aggregate"]][["within_probe_mean_across_seeds"]])
  S("C3_u1_blockB_L8_seeds_u5.json", "aggregate.within_probe_mean_across_seeds", .num(b8[["within_probe_mean_across_seeds"]]))
  S("U1_E1_transplant_GATE_L12_s{0,1,2}.json", "mean(keycos.within_probe_mean)", l12[["mean"]])
  S("C3_u1_blockB_L14_seeds_u5.json", "aggregate.within_probe_mean_across_seeds", .num(b14[["within_probe_mean_across_seeds"]]))
  S("G1_L12_analysis.json", "aggregate.within_probe_mean_across_seeds (rewrite ref)", rref)
  pfd <- data.frame(layer = c(8, 12, 14),
                    value = c(.num(b8[["within_probe_mean_across_seeds"]]), l12[["mean"]], .num(b14[["within_probe_mean_across_seeds"]])),
                    sd = c(.num(b8[["within_probe_std_across_seeds"]]), l12[["sd"]], .num(b14[["within_probe_std_across_seeds"]])))
  pa <- ggplot(pfd, aes(layer, value)) +
    geom_hline(yintercept = rref, colour = viz$crit, linetype = "dashed", linewidth = 0.4) +
    lbl_geom(14.8, 0.05, sprintf("rewrite $\\rho{=}%.3f$", rref), hjust = 1, colour = viz$crit, size = 1.9) +
    errbar(data = subset(pfd, sd > 0), aes(ymin = value - sd, ymax = value + sd)) +
    geom_line(colour = viz$seqblue, linewidth = 0.5) + geom_point(colour = viz$seqblue, size = 1.6) +
    scale_x_continuous(breaks = c(8, 12, 14)) + coord_cartesian(ylim = c(0, 0.85), xlim = c(7, 15)) +
    labs(title = "(a) depth", x = "layer", y = "within-probe $\\rho$") + theme_b6()

  # (b) variants nondc/dc
  vars <- list(c("refusal","u1_gate_refusal_L12_s0.json"), c("eos","u1_gate_eos_L12_s0.json"),
               c("suppress","u1_gate_suppress_L12_s0.json"))
  vb <- data.frame()
  for (v in vars) {
    a <- .load(v[2])[["arms"]][[ARM]]
    nd <- S(v[2], sprintf("arms['%s'].nondc_rho", ARM), .num(a[["nondc_rho"]]))
    dc <- S(v[2], sprintf("arms['%s'].dc_rho", ARM), .num(a[["dc_rho"]]))
    vb <- rbind(vb, data.frame(x = v[1], series = "raw", value = nd),
                    data.frame(x = v[1], series = "DC", value = dc))
  }
  vb$x <- factor(vb$x, levels = c("refusal","eos","suppress"))
  vb$series <- factor(vb$series, levels = c("raw","DC"))
  pb <- ggplot(vb, aes(x, value, fill = series)) +
    geom_col(position = position_dodge(width = 0.72), width = 0.62) +
    scale_fill_manual(values = c("raw" = viz$seqblue, "DC" = viz$aqua)) +
    coord_cartesian(ylim = c(0, 0.8)) +
    labs(title = "(b) variants", x = NULL, y = "$\\rho$ (s0)") +
    theme_b6() + theme(axis.text.x = element_text(angle = 40, hjust = 1, size = 6.5))

  # (c) collapse
  rome_dmg <- rome_cpl <- numeric(0)
  for (s in 0:2) { d <- .load(sprintf("U1_E1_transplant_GATE_L12_s%d.json", s))
    rome_dmg <- c(rome_dmg, .num(d[["mean_signed_damage"]])); rome_cpl <- c(rome_cpl, .num(d[["SxC_within_probe_mean"]])) }
  alp_dmg <- numeric(0)
  for (s in 0:2) { fn <- sprintf("U1_E1_transplant_GATE_alphadelete_L12_s%d.json", s)
    if (.exists(fn)) alp_dmg <- c(alp_dmg, .num(.load(fn)[["mean_signed_damage"]])) }
  bd <- .load("C3_u1_blockD_alphadelete_seeds_u2.json")[["aggregate"]]
  dbm <- seed_mean_std(rome_dmg); cbm <- seed_mean_std(rome_cpl); dam <- seed_mean_std(alp_dmg)
  cam <- c(mean = .num(bd[["within_probe_mean_across_seeds"]]), sd = .num(bd[["within_probe_std_across_seeds"]]))
  S("U1_E1_transplant_GATE_L12_s{0,1,2}.json", "mean(mean_signed_damage)", dbm[["mean"]])
  S("U1_E1_transplant_GATE_L12_s{0,1,2}.json", "mean(SxC_within_probe_mean)", cbm[["mean"]])
  S("U1_E1_transplant_GATE_alphadelete_L12_s{0,1,2}.json", "mean(mean_signed_damage)", dam[["mean"]])
  S("C3_u1_blockD_alphadelete_seeds_u2.json", "aggregate.within_probe_mean_across_seeds", cam[["mean"]])
  cd <- rbind(
    data.frame(x = "ROME-del", series = "mean dmg", value = dbm[["mean"]], sd = dbm[["sd"]]),
    data.frame(x = "Alpha-del", series = "mean dmg", value = dam[["mean"]], sd = dam[["sd"]]),
    data.frame(x = "ROME-del", series = "$S\\times C$ $\\rho$", value = cbm[["mean"]], sd = cbm[["sd"]]),
    data.frame(x = "Alpha-del", series = "$S\\times C$ $\\rho$", value = cam[["mean"]], sd = cam[["sd"]]))
  cd$x <- factor(cd$x, levels = c("ROME-del","Alpha-del"))
  cd$series <- factor(cd$series, levels = c("mean dmg", "$S\\times C$ $\\rho$"))
  pc <- ggplot(cd, aes(x, value, fill = series)) +
    geom_col(position = position_dodge(width = 0.72), width = 0.62) +
    errbar(data = subset(cd, sd > 0), aes(ymin = value - sd, ymax = value + sd),
           position = position_dodge(width = 0.72)) +
    scale_fill_manual(values = setNames(c(viz$orange, viz$seqblue), levels(cd$series))) +
    coord_cartesian(ylim = c(0, 4.6)) +
    labs(title = "(c) collapse", x = NULL, y = "value") +
    theme_b6() + theme(axis.text.x = element_text(angle = 40, hjust = 1, size = 6.5),
                       legend.key.width = unit(9, "pt"))

  # (d) transplant
  tp <- list(c("L8 s0","U1_E1_transplant_GATE_L8_s0.json"), c("L12 s0","U1_E1_transplant_GATE_L12_s0.json"),
             c("L12 s1","U1_E1_transplant_GATE_L12_s1.json"), c("L12 s2","U1_E1_transplant_GATE_L12_s2.json"),
             c("L14 s0","U1_E1_transplant_GATE_L14_s0.json"))
  td <- data.frame()
  for (t in tp) { d <- .load(t[2])
    v <- S(t[2], "delta_rho_SxC_minus_best_transplant", .num(d[["delta_rho_SxC_minus_best_transplant"]]))
    td <- rbind(td, data.frame(x = t[1], value = v)) }
  td$x <- factor(td$x, levels = sapply(tp, `[`, 1))
  pd <- ggplot(td, aes(x, value)) + geom_col(fill = viz$violet, width = 0.68) +
    coord_cartesian(ylim = c(0, 0.75)) +
    labs(title = "(d) transplant", x = NULL, y = "$\\Delta\\rho$ transpl.") +
    theme_b6() + theme(axis.text.x = element_text(angle = 42, hjust = 1, size = 6.5))
  list(plot = (pa | pb) / (pc | pd), prov = .prov)
}

# =====================================================================
# FigF  sequential: survival curves | position fragility (descriptive)
# =====================================================================
fig_F <- function() {
  prov_reset()
  src <- "SEQ_analysis_L12.json"; d <- .load(src)
  if (.exists("SEQ_analysis_L12_4stream.json")) {
    four <- .load("SEQ_analysis_L12_4stream.json")
    nb <- length(d[["per_stream"]])                       # zip() stops at the shorter (base)
    ok <- all(mapply(function(o, n) o[["npz"]] == n[["npz"]] &&
                       isTRUE(all.equal(o[["final_survival_frac"]], n[["final_survival_frac"]])) &&
                       abs(.num(o[["position_fragility"]][["rho_position_vs_survival"]]) -
                             .num(n[["position_fragility"]][["rho_position_vs_survival"]])) <= 1e-4,
                     d[["per_stream"]], four[["per_stream"]][seq_len(nb)]))
    if (ok) { d <- four; src <- "SEQ_analysis_L12_4stream.json" }
  }
  cols <- c(s0 = viz$blue, s1 = viz$aqua, s2 = viz$yellow, s3 = viz$green)
  sv <- data.frame()
  for (i in seq_along(d[["per_stream"]])) {
    s <- d[["per_stream"]][[i]]; sid <- sprintf("s%d", i - 1)
    for (c in s[["survival_curve"]]) {
      fv <- .num(c[["frac_survived"]]); ne <- c[["checkpoint_nedits"]]
      S(src, sprintf("per_stream[%d].survival_curve[%d].frac_survived", i - 1, ne), fv)
      sv <- rbind(sv, data.frame(nedits = ne, frac = fv, stream = sid))
    }
  }
  sv$stream <- factor(sv$stream, levels = names(cols))
  pa <- ggplot(sv, aes(nedits, frac, colour = stream, shape = stream)) +
    geom_line(linewidth = 0.45) + geom_point(size = 1.2) +
    scale_colour_manual(values = cols) + scale_shape_manual(values = c(16, 17, 15, 18)) +
    coord_cartesian(ylim = c(0, 0.82)) +
    labs(title = "(a) survival", x = "edits applied", y = "fraction surviving") +
    theme_b6() + theme(legend.key.width = unit(8, "pt"))

  fr <- data.frame(); labs <- character(0)
  for (i in seq_along(d[["per_stream"]])) {
    pf <- d[["per_stream"]][[i]][["position_fragility"]][["rho_position_vs_survival"]]
    v <- S(src, sprintf("per_stream[%d].position_fragility.rho_position_vs_survival", i - 1), .num(pf))
    fr <- rbind(fr, data.frame(x = sprintf("s%d", i - 1), value = v)); labs <- c(labs, sprintf("s%d", i - 1))
  }
  poolv <- S(src, "pooled.position_fragility.rho_position_vs_survival",
             .num(d[["pooled"]][["position_fragility"]][["rho_position_vs_survival"]]))
  fr <- rbind(fr, data.frame(x = "pool", value = poolv)); labs <- c(labs, "pool")
  fr$x <- factor(fr$x, levels = labs)
  ymax_pf <- max(0.42, max(fr$value) * 1.35)
  pb <- ggplot(fr, aes(x, value)) + geom_col(fill = viz$seqblue, width = 0.68) +
    coord_cartesian(ylim = c(0, ymax_pf)) +
    labs(title = "(b) fragility", x = NULL, y = "$\\rho$(pos, surv)") +
    theme_b6() + theme(axis.text.x = element_text(angle = 30, hjust = 1, size = 6.5))
  list(plot = pa | pb, prov = .prov)
}

# =====================================================================
# emit: render one figure to tikz + prepend provenance comment header
# =====================================================================
emit <- function(name, obj, width, height, headline) {
  path <- file.path(OUT, paste0(name, ".tex"))
  tikz(path, width = width, height = height, standAlone = FALSE, sanitize = FALSE,
       verbose = FALSE)
  print(obj$plot); dev.off()
  body <- readLines(path, warn = FALSE)
  # drop tikzDevice's timestamp line so two runs are byte-identical (deterministic)
  body <- body[!grepl("^% Created by tikzDevice", body)]
  header <- c(
    "% =====================================================================",
    sprintf("%% %s  — R/ggplot2 -> tikzDevice (parallel to ../figures-tex/%s)", name, headline),
    "% Canonical-JSON-only. Regenerate: Rscript figures-r/make_figures.R",
    "% Provenance (JSON path :: field = plotted value), one line per series:",
    obj$prov,
    "% =====================================================================")
  writeLines(c(header, body), path)
  cat(sprintf("[ok] %s.tex  (%d source lines, %.2f x %.2f in)\n", name, length(obj$prov), width, height))
}

# ---- dimensions: figure* -> \textwidth 6.3in ; figure -> \columnwidth 3.03in
main <- function() {
  dir.create(OUT, showWarnings = FALSE, recursive = TRUE)
  # times/amsmath so tikzDevice string metrics match the ACL (Times) target
  options(tikzLatexPackages = c(
    "\\usepackage{tikz}", "\\usepackage[active,tightpage,psfixbb]{preview}",
    "\\PreviewEnvironment{pictureenvironment}", "\\usepackage{times}",
    "\\usepackage{amsmath}", "\\usepackage{amssymb}"))
  # widths: \textwidth = 6.3in (figure*); \columnwidth = 219.09pt = 3.03in (figure).
  # heights: reduced to fit the ACL 8-page budget (§7/§8 floats were stranding).
  emit("figA", fig_A(), 6.3,  3.2, "figA_lawtransfer.tex")
  emit("figB", fig_B(), 3.03, 2.2, "figB_surrogate.tex")
  emit("figC", fig_C(), 3.03, 2.0, "figC_causal.tex")
  emit("figD", fig_D(), 6.3,  2.0, "figD_editor.tex")
  emit("figE", fig_E(), 6.3,  2.7, "figE_deletion.tex")
  emit("figF", fig_F(), 3.03, 2.0, "figF_sequential.tex")
  cat("\nWrote 6 R figure .tex files to ", OUT, "\n", sep = "")
}
main()
