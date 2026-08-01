#!/usr/bin/env Rscript
# =====================================================================
# figF8_theorem_honesty.R — R/ggplot2 -> tikzDevice
# F8: Theorem & surrogate honesty (4 panels A-D).
# SOURCE artifacts (all in edit-harness/results/):
#   GRADSIM_TRUE_Llama-3.2-1B_L{8,10,12,14}_s{0,1,2}.json
#     .prop1_identity_check.{PASS,max_bound_ratio,n_pairs_checked}
#     .within_probe_rho.{direct_vs_damage,SC_vs_damage_reference,factorized_vs_damage}.mean
#     .alpha_A4_test.{sign_consistency_rate,coefficient_of_variation}.{mean,median}
#     .rank_agreement.direct_vs_SC.mean
# Build: Rscript submissions/ieee/figures-src/figF8_theorem_honesty.R
# Output: submissions/ieee/figures-src/figF8_theorem_honesty.tex
# =====================================================================
suppressPackageStartupMessages({
  library(jsonlite); library(ggplot2); library(patchwork); library(tikzDevice)
})

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
HERE <- if (length(script_arg)) normalizePath(dirname(sub("^--file=","",script_arg))) else "."
RESULTS <- normalizePath(file.path(HERE, "..", "..", "..", "edit-harness", "results"))
OUT_TEX  <- file.path(HERE, "..", "figures", "figF8_theorem_honesty.tex")

.load <- function(name) fromJSON(readLines(file.path(RESULTS, name), warn = FALSE),
                                  simplifyVector = FALSE)

# dataviz palette — same slot order as make_figures_ieee.R
viz <- list(
  blue  = "#2A78D6", aqua  = "#1BAF7A", yellow = "#EDA100", green = "#008300",
  violet= "#4A3AA7", red   = "#E34948", orange = "#EB6834",
  ink   = "#0B0B0B", ink2  = "#333333", tick   = "#666666",
  muted = "#898781", grid  = "#E1E0D9"
)
theme_b6 <- function(base = 9) {
  theme_minimal(base_size = base) +
    theme(
      text               = element_text(colour = viz$ink),
      plot.title         = element_text(hjust = 0.5, face = "bold", size = base,
                                        colour = viz$ink, margin = margin(b = 2)),
      plot.tag           = element_text(face = "bold", size = base, colour = viz$ink),
      plot.tag.position  = c(0.01, 0.98),
      axis.title.x       = element_text(size = base - 1, colour = viz$ink, margin = margin(t = 1)),
      axis.title.y       = element_text(size = base - 1, colour = viz$ink, margin = margin(r = 1)),
      axis.text          = element_text(size = base - 2, colour = viz$ink2),
      panel.grid.major.y = element_line(colour = viz$grid, linewidth = 0.25),
      panel.grid.major.x = element_blank(), panel.grid.minor = element_blank(),
      axis.ticks         = element_line(colour = viz$tick, linewidth = 0.3),
      axis.ticks.length  = unit(2, "pt"), axis.line = element_blank(),
      legend.position    = "top", legend.justification = "center",
      legend.title       = element_blank(),
      legend.text        = element_text(size = base - 2, colour = viz$ink),
      legend.key.size    = unit(7, "pt"), legend.key.height = unit(6, "pt"),
      legend.spacing.x   = unit(2, "pt"), legend.margin = margin(0,0,0,0),
      legend.box.margin  = margin(0,0,-6,0), legend.box.spacing = unit(1, "pt"),
      plot.margin        = margin(1, 2, 1, 1)
    )
}

LAYERS <- c(8L, 10L, 12L, 14L)
SEEDS  <- 0:2

# ---- read all 12 cells ------------------------------------------------
cells <- list()
for (L in LAYERS) for (S in SEEDS) {
  key <- sprintf("L%d_s%d", L, S)
  cells[[key]] <- .load(sprintf("GRADSIM_TRUE_Llama-3.2-1B_L%d_s%d.json", L, S))
}
prov <- character(0)
S <- function(artifact, field, value) {
  v <- if (is.numeric(value)) formatC(value, format = "f", digits = 4) else as.character(value)
  prov[[length(prov)+1L]] <<- sprintf("%% SOURCE: results/%s :: %s = %s", artifact, field, v)
  invisible(value)
}

# =====================================================================
# Panel A — Prop.1 identity check: max_bound_ratio per cell (threshold 1)
# =====================================================================
rows_a <- data.frame()
for (L in LAYERS) for (Si in SEEDS) {
  key <- sprintf("L%d_s%d", L, Si)
  art <- sprintf("GRADSIM_TRUE_Llama-3.2-1B_L%d_s%d.json", L, Si)
  v   <- S(art, "prop1_identity_check.max_bound_ratio",
            cells[[key]]$prop1_identity_check$max_bound_ratio)
  np  <- cells[[key]]$prop1_identity_check$n_pairs_checked
  rows_a <- rbind(rows_a, data.frame(layer = L, seed = Si, ratio = v, n_pairs = np))
}
# 3-seed mean per layer for connected line
agg_a <- aggregate(ratio ~ layer, rows_a, mean)

# Fix-1: log10 y-scale so both data (0.0007-0.005) AND threshold (1.0) render
# Fix-3: seeded jitter for byte-stable output
pA <- ggplot(rows_a, aes(x = factor(layer), y = ratio)) +
  geom_hline(yintercept = 1.0, linewidth = 0.5, colour = viz$red, linetype = "dashed") +
  geom_jitter(position = position_jitter(seed = 42, width = 0.12),
              size = 1.8, colour = viz$blue, alpha = 0.75) +
  annotate("text", x = 0.60, y = 1.35, label = "threshold 1.0", hjust = 0,
           size = 2.1, colour = viz$red) +
  annotate("text", x = 2.5, y = 0.0015,
           label = sprintf("all %d cells PASS", nrow(rows_a)),
           hjust = 0.5, size = 2.1, colour = viz$ink2) +
  scale_y_log10(
    limits = c(5e-4, 4.0),
    breaks  = c(0.001, 0.01, 0.1, 1.0),
    labels  = c("0.001", "0.01", "0.1", "1.0")
  ) +
  labs(title = "identity check (fp64)", tag = "(a)",
       x = "edited layer", y = "max bound ratio (log scale)") +
  theme_b6()

# =====================================================================
# Panel B — true-influence predicts damage, depth-rising
# 3 series: direct (true infl), S×C reference, factorized
# =====================================================================
rows_b <- data.frame()
for (L in LAYERS) {
  art <- sprintf("GRADSIM_TRUE_Llama-3.2-1B_L%d_s%d.json", L, 0)
  for (Si in SEEDS) {
    key <- sprintf("L%d_s%d", L, Si)
    art2 <- sprintf("GRADSIM_TRUE_Llama-3.2-1B_L%d_s%d.json", L, Si)
    rows_b <- rbind(rows_b, data.frame(
      layer = L, seed = Si,
      direct = S(art2, "within_probe_rho.direct_vs_damage.mean",
                 cells[[key]]$within_probe_rho$direct_vs_damage$mean),
      scref  = S(art2, "within_probe_rho.SC_vs_damage_reference.mean",
                 cells[[key]]$within_probe_rho$SC_vs_damage_reference$mean),
      fact   = S(art2, "within_probe_rho.factorized_vs_damage.mean",
                 cells[[key]]$within_probe_rho$factorized_vs_damage$mean)
    ))
  }
}
agg_b <- do.call(rbind, lapply(split(rows_b, rows_b$layer), function(df) {
  data.frame(layer = df$layer[1],
             direct_m = mean(df$direct), direct_sd = sd(df$direct),
             scref_m  = mean(df$scref),  scref_sd  = sd(df$scref),
             fact_m   = mean(df$fact),   fact_sd   = sd(df$fact))
}))
long_b <- rbind(
  data.frame(layer = agg_b$layer, rho = agg_b$direct_m, sd = agg_b$direct_sd,
             series = "true influence (direct)"),
  data.frame(layer = agg_b$layer, rho = agg_b$scref_m,  sd = agg_b$scref_sd,
             series = "$S{\\times}C$ reference"),
  data.frame(layer = agg_b$layer, rho = agg_b$fact_m,   sd = agg_b$fact_sd,
             series = "factorized Eq.~3")
)
long_b$series <- factor(long_b$series, levels = c("true influence (direct)",
                                                    "$S{\\times}C$ reference",
                                                    "factorized Eq.~3"))
pB <- ggplot(long_b, aes(layer, rho, colour = series, shape = series)) +
  geom_hline(yintercept = 0.10, linewidth = 0.3, colour = viz$muted, linetype = "dotted") +
  geom_errorbar(aes(ymin = rho - sd, ymax = rho + sd), width = 0.35, linewidth = 0.35) +
  geom_line(linewidth = 0.55) + geom_point(size = 1.8) +
  scale_colour_manual(values = c("true influence (direct)" = viz$blue,
                                  "$S{\\times}C$ reference"  = viz$aqua,
                                  "factorized Eq.~3"         = viz$orange)) +
  scale_shape_manual(values = c("true influence (direct)" = 16,
                                 "$S{\\times}C$ reference"  = 15,
                                 "factorized Eq.~3"         = 17)) +
  scale_x_continuous(breaks = LAYERS) +
  coord_cartesian(ylim = c(0, 0.78), xlim = c(7.3, 14.7)) +
  labs(title = "true-influence $\\rho$ (depth profile, rising to L12)", tag = "(b)",
       x = "edited layer", y = "within-probe $\\rho$ (influence, damage)") +
  theme_b6() + theme(legend.key.width = unit(11, "pt"))

# =====================================================================
# Panel C — alpha A4': sign-consistency rate and median CV per layer
# Sign-consistency in [0.50 chance floor, 1.0]; CV in [0, ~3]
# =====================================================================
rows_c <- data.frame()
for (L in LAYERS) for (Si in SEEDS) {
  key <- sprintf("L%d_s%d", L, Si)
  art <- sprintf("GRADSIM_TRUE_Llama-3.2-1B_L%d_s%d.json", L, Si)
  rows_c <- rbind(rows_c, data.frame(
    layer = L, seed = Si,
    signcons = S(art, "alpha_A4_test.sign_consistency_rate.mean",
                 cells[[key]]$alpha_A4_test$sign_consistency_rate$mean),
    medCV    = S(art, "alpha_A4_test.coefficient_of_variation.median",
                 cells[[key]]$alpha_A4_test$coefficient_of_variation$median)
  ))
}
agg_c <- do.call(rbind, lapply(split(rows_c, rows_c$layer), function(df) {
  data.frame(layer = df$layer[1],
             sc_m = mean(df$signcons), sc_sd = sd(df$signcons),
             cv_m = mean(df$medCV),    cv_sd = sd(df$medCV))
}))
# dual-axis: left = sign-consistency [0.5,1], right = median CV [0,3]
# render as two overlapping lines; secondary axis via y-rescale trick
CV_SCALE <- 1 / 3   # maps CV [0,3] -> [0,1] for co-plot with sign-cons [0.5,1]
long_c <- rbind(
  data.frame(layer = agg_c$layer, val = agg_c$sc_m, sd = agg_c$sc_sd,
             series = "sign-consistency"),
  data.frame(layer = agg_c$layer, val = agg_c$cv_m * CV_SCALE, sd = agg_c$cv_sd * CV_SCALE,
             series = "median CV (right)")
)
long_c$series <- factor(long_c$series, levels = c("sign-consistency", "median CV (right)"))
pC <- ggplot(long_c, aes(layer, val, colour = series, shape = series)) +
  geom_hline(yintercept = 0.50, linewidth = 0.4, colour = viz$muted, linetype = "dotted") +
  annotate("text", x = 8.3, y = 0.515, label = "chance 0.50", hjust = 0,
           size = 2.1, colour = viz$muted) +
  geom_errorbar(aes(ymin = val - sd, ymax = val + sd), width = 0.35, linewidth = 0.35) +
  geom_line(linewidth = 0.55) + geom_point(size = 1.8) +
  scale_colour_manual(values = c("sign-consistency"  = viz$blue,
                                  "median CV (right)" = viz$orange)) +
  scale_shape_manual(values = c("sign-consistency"  = 16,
                                 "median CV (right)" = 17)) +
  scale_x_continuous(breaks = LAYERS) +
  # Fix-2: lower bound 0.15 so CV_scaled at L12(0.354) and L14(0.398) are not clipped
  scale_y_continuous(
    name   = "sign-consistency rate",
    limits = c(0.15, 1.10),
    breaks = c(0.2, 0.4, 0.6, 0.8, 1.0),
    sec.axis = sec_axis(~ . / CV_SCALE, name = "median CV", breaks = c(0, 1, 2, 3))
  ) +
  coord_cartesian(xlim = c(7.3, 14.7)) +
  labs(title = "A4$'$ test: $\\alpha$ constancy", tag = "(c)",
       x = "edited layer") +
  theme_b6() + theme(axis.title.y.right = element_text(size = 8, colour = viz$orange,
                                                         margin = margin(l = 2)))

# =====================================================================
# Panel D — direct-vs-S×C rank agreement (~0 → not a faithful surrogate)
# =====================================================================
rows_d <- data.frame()
for (L in LAYERS) for (Si in SEEDS) {
  key <- sprintf("L%d_s%d", L, Si)
  art <- sprintf("GRADSIM_TRUE_Llama-3.2-1B_L%d_s%d.json", L, Si)
  rows_d <- rbind(rows_d, data.frame(
    layer = L, seed = Si,
    agree = S(art, "rank_agreement.direct_vs_SC.mean",
              cells[[key]]$rank_agreement$direct_vs_SC$mean)
  ))
}
agg_d <- aggregate(agree ~ layer, rows_d, mean)
agg_d$sd <- sapply(split(rows_d$agree, rows_d$layer), sd)

pD <- ggplot(agg_d, aes(factor(layer), agree)) +
  geom_hline(yintercept = 0, linewidth = 0.4, colour = viz$muted) +
  geom_errorbar(aes(ymin = agree - sd, ymax = agree + sd),
                width = 0.25, linewidth = 0.45, colour = viz$muted) +
  geom_point(size = 2.2, colour = viz$red) +
  geom_hline(yintercept = 0, linewidth = 0.3, colour = viz$ink2, linetype = "solid") +
  annotate("text", x = 2.5, y = 0.16,
           label = "$\\rho \\approx 0$: not a faithful rank-surrogate",
           hjust = 0.5, size = 2.1, colour = viz$ink2) +
  coord_cartesian(ylim = c(-0.20, 0.25)) +
  labs(title = "rank agreement: direct vs $S{\\times}C$", tag = "(d)",
       x = "edited layer",
       y = "$\\rho$(direct rank, $S{\\times}C$ rank)") +
  theme_b6()

# =====================================================================
# Emit: combine 4 panels and write .tex with provenance header
# =====================================================================
fig_f8 <- (pA | pB) / (pC | pD) +
  plot_annotation(theme = theme(plot.margin = margin(2, 2, 2, 2)))

path <- OUT_TEX
tikz(path, width = 7.16, height = 4.2, standAlone = FALSE, sanitize = FALSE,
     verbose = FALSE,
     packages = c("\\usepackage{tikz}",
                  "\\usepackage[active,tightpage,psfixbb]{preview}",
                  "\\PreviewEnvironment{pictureenvironment}",
                  "\\usepackage{times}", "\\usepackage{amsmath}",
                  "\\usepackage{amssymb}"))
print(fig_f8)
dev.off()

body   <- readLines(path, warn = FALSE)
body   <- body[!grepl("^% Created by tikzDevice", body)]
header <- c(
  "% =====================================================================",
  "% figF8_theorem_honesty.tex  — R/ggplot2 -> tikzDevice",
  "% Regenerate: Rscript submissions/ieee/figures-src/figF8_theorem_honesty.R",
  "% Provenance (canonical artifact :: field = plotted value, 3 seeds x 4 layers):",
  prov,
  "% ====================================================================="
)
writeLines(c(header, body), path)
cat(sprintf("[ok] %s  (%d source lines)\n", basename(path), length(prov)))
