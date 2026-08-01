#!/usr/bin/env Rscript
# =====================================================================
# figF1_geometry_law.R — R/ggplot2 -> tikzDevice
# F1: Geometry law across depth (4 panels A-D).
# SOURCE artifacts (all in edit-harness/results/):
#   C1_mechanism_sc_table.json  — groups[model=llama1b,editor=rome,dataset=cf]
#     .within_probe_rho_C    (rho_key-cos, = G1 agg wp)
#     .within_probe_rho_SC   (rho_S×C corrected)
#   G2_gradsim_L{8,10,12,14}.json — aggregate
#     .keycos_within_probe.mean/std
#     .gradsim_within_probe.normgrowth.mean/std
#   G1_stability_L{8,10,12,14}_v2.json — aggregate + per_seed
#     flat_spearman_INFLATED  (the inflated flat metric)
#     within_probe_mean       (the within-probe gate metric)
#     .aggregate.within_probe_mean_across_seeds
#     .per_seed[].{editlevel_null_mean,editlevel_null_std,editlevel_z}
#     .aggregate.max_within_probe_perm_p / max_within_probe_perm_p_editlevel
# Build: Rscript submissions/ieee/figures-src/figF1_geometry_law.R
# Output: submissions/ieee/figures-src/figF1_geometry_law.tex
# =====================================================================
suppressPackageStartupMessages({
  library(jsonlite); library(ggplot2); library(patchwork); library(tikzDevice)
})

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
HERE <- if (length(script_arg)) normalizePath(dirname(sub("^--file=","",script_arg))) else "."
RESULTS <- normalizePath(file.path(HERE, "..", "..", "..", "edit-harness", "results"))
OUT_TEX  <- file.path(HERE, "..", "figures", "figF1_geometry_law.tex")

.load <- function(name) fromJSON(readLines(file.path(RESULTS, name), warn = FALSE),
                                  simplifyVector = FALSE)

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
      axis.title.x       = element_text(size = base-1, colour = viz$ink, margin = margin(t=1)),
      axis.title.y       = element_text(size = base-1, colour = viz$ink, margin = margin(r=1)),
      axis.text          = element_text(size = base-2, colour = viz$ink2),
      panel.grid.major.y = element_line(colour = viz$grid, linewidth = 0.25),
      panel.grid.major.x = element_blank(), panel.grid.minor = element_blank(),
      axis.ticks         = element_line(colour = viz$tick, linewidth = 0.3),
      axis.ticks.length  = unit(2, "pt"), axis.line = element_blank(),
      legend.position    = "top", legend.justification = "center",
      legend.title       = element_blank(),
      legend.text        = element_text(size = base-2, colour = viz$ink),
      legend.key.size    = unit(7, "pt"), legend.key.height = unit(6, "pt"),
      legend.spacing.x   = unit(2, "pt"), legend.margin = margin(0,0,0,0),
      legend.box.margin  = margin(0,0,-6,0), legend.box.spacing = unit(1, "pt"),
      plot.margin        = margin(1, 2, 1, 1)
    )
}

LAYERS <- c(8L, 10L, 12L, 14L)
prov   <- character(0)
S <- function(artifact, field, value) {
  v <- if (is.numeric(value)) formatC(value, format = "f", digits = 4) else as.character(value)
  prov[[length(prov)+1L]] <<- sprintf("%% SOURCE: results/%s :: %s = %s", artifact, field, v)
  invisible(value)
}

# ---- load C1 mechanism table (corrected S×C values) ----------------
c1 <- .load("C1_mechanism_sc_table.json")
llama_grps <- Filter(function(g) g$model == "llama1b" && g$editor == "rome" && g$dataset == "cf",
                     c1$groups)
rho_c  <- setNames(sapply(llama_grps, function(g)
  S("C1_mechanism_sc_table.json",
    sprintf("groups[llama1b,rome,cf,L%d].within_probe_rho_C", g$layer),
    g$within_probe_rho_C)), sapply(llama_grps, `[[`, "layer"))
rho_sc <- setNames(sapply(llama_grps, function(g)
  S("C1_mechanism_sc_table.json",
    sprintf("groups[llama1b,rome,cf,L%d].within_probe_rho_SC", g$layer),
    g$within_probe_rho_SC)), sapply(llama_grps, `[[`, "layer"))

# ---- load G2_gradsim (S×C normgrowth variant, std as error bars) ----
gradsim_ng <- list(); gradsim_kc <- list()
for (L in LAYERS) {
  g  <- .load(sprintf("G2_gradsim_L%d.json", L))$aggregate
  gradsim_kc[[as.character(L)]] <- list(
    m  = S(sprintf("G2_gradsim_L%d.json", L), "aggregate.keycos_within_probe.mean",
           g$keycos_within_probe$mean),
    sd = S(sprintf("G2_gradsim_L%d.json", L), "aggregate.keycos_within_probe.std",
           g$keycos_within_probe$std))
  gradsim_ng[[as.character(L)]] <- list(
    m  = S(sprintf("G2_gradsim_L%d.json", L), "aggregate.gradsim_within_probe.normgrowth.mean",
           g$gradsim_within_probe$normgrowth$mean),
    sd = S(sprintf("G2_gradsim_L%d.json", L), "aggregate.gradsim_within_probe.normgrowth.std",
           g$gradsim_within_probe$normgrowth$std))
}

# ---- load G1_stability_v2 (flat INFLATED, within-probe, perm-null moments) ----
g1v2 <- list(); g1v2_agg <- list()
for (L in LAYERS) {
  dat <- .load(sprintf("G1_stability_L%d_v2.json", L))
  g1v2_agg[[as.character(L)]] <- dat$aggregate
  g1v2[[as.character(L)]] <- dat$per_seed
}

# =====================================================================
# Panel A — ρ_SC layer profile (corrected S×C, 3-seed aggregate from G2)
# Two series: raw key-cos ρ_C (blue) and S×C ρ_SC (aqua)
# =====================================================================
dA <- rbind(
  data.frame(layer = LAYERS,
             rho   = sapply(as.character(LAYERS), function(l) gradsim_kc[[l]]$m),
             sd    = sapply(as.character(LAYERS), function(l) gradsim_kc[[l]]$sd),
             series = "key-cos $|C|$"),
  data.frame(layer = LAYERS,
             rho   = sapply(as.character(LAYERS), function(l) gradsim_ng[[l]]$m),
             sd    = sapply(as.character(LAYERS), function(l) gradsim_ng[[l]]$sd),
             series = "$S{\\times}C$")
)
dA$series <- factor(dA$series, levels = c("key-cos $|C|$", "$S{\\times}C$"))

pA <- ggplot(dA, aes(layer, rho, colour = series, shape = series)) +
  geom_hline(yintercept = 0.10, linewidth = 0.3, colour = viz$muted, linetype = "dotted") +
  annotate("text", x = 8.4, y = 0.115, label = "DEAD 0.10", hjust = 0,
           size = 1.9, colour = viz$muted) +
  geom_errorbar(aes(ymin = rho - sd, ymax = rho + sd), width = 0.35, linewidth = 0.35) +
  geom_line(linewidth = 0.55) + geom_point(size = 1.8) +
  scale_colour_manual(values = c("key-cos $|C|$" = viz$blue, "$S{\\times}C$" = viz$aqua)) +
  scale_shape_manual(values  = c("key-cos $|C|$" = 16,       "$S{\\times}C$" = 15)) +
  scale_x_continuous(breaks = LAYERS) +
  coord_cartesian(ylim = c(0, 0.78), xlim = c(7.3, 14.7)) +
  labs(title = "$\\rho_{SC}$ layer profile", tag = "(a)",
       x = "edited layer", y = "within-probe $\\rho$") +
  theme_b6() + theme(legend.key.width = unit(11, "pt"))

# =====================================================================
# Panel B — S×C wins over raw key-cos at all 4 layers
# Bar chart: rho_C vs rho_SC per layer (C1 canonical table values)
# =====================================================================
dB <- rbind(
  data.frame(layer = LAYERS,
             rho   = unname(rho_c[as.character(LAYERS)]),
             metric = "key-cos $\\rho_C$"),
  data.frame(layer = LAYERS,
             rho   = unname(rho_sc[as.character(LAYERS)]),
             metric = "$S{\\times}C$ $\\rho_{SC}$")
)
dB$metric <- factor(dB$metric, levels = c("key-cos $\\rho_C$", "$S{\\times}C$ $\\rho_{SC}$"))
dB$layer_f <- factor(sprintf("L%d", dB$layer), levels = sprintf("L%d", LAYERS))

pB <- ggplot(dB, aes(layer_f, rho, fill = metric)) +
  geom_col(position = position_dodge(width = 0.72), width = 0.62) +
  geom_text(aes(label = sprintf("%.3f", rho)),
            position = position_dodge(width = 0.72),
            vjust = -0.4, size = 2.0) +
  scale_fill_manual(values = c("key-cos $\\rho_C$"    = viz$blue,
                                "$S{\\times}C$ $\\rho_{SC}$" = viz$aqua)) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.20))) +
  labs(title = "$S{\\times}C$ beats key-cos (all layers)", tag = "(b)",
       x = "edited layer", y = "within-probe $\\rho$", fill = NULL) +
  theme_b6() + theme(legend.position = "top")

# =====================================================================
# Panel C — within-probe vs flat (AUROC-artifact contrast)
# Per layer: show flat_spearman_INFLATED (grey) vs within_probe_mean (blue)
# 3-seed error bars from G1_stability_v2 per_seed records
# =====================================================================
rows_c <- data.frame()
for (L in LAYERS) {
  for (ps in g1v2[[as.character(L)]]) {
    art <- sprintf("G1_stability_L%d_v2.json", L)
    rows_c <- rbind(rows_c, data.frame(
      layer = L,
      flat  = S(art, sprintf("per_seed[%s].flat_spearman_INFLATED", ps$npz),
                ps$flat_spearman_INFLATED),
      wp    = S(art, sprintf("per_seed[%s].within_probe_mean", ps$npz),
                ps$within_probe_mean)
    ))
  }
}
agg_c_flat <- aggregate(flat ~ layer, rows_c, function(x) c(m=mean(x), sd=sd(x)))
agg_c_wp   <- aggregate(wp   ~ layer, rows_c, function(x) c(m=mean(x), sd=sd(x)))
dC <- rbind(
  data.frame(layer = agg_c_flat$layer,
             rho   = agg_c_flat$flat[,"m"],
             sd    = agg_c_flat$flat[,"sd"],
             type  = "flat (unpartialled)"),
  data.frame(layer = agg_c_wp$layer,
             rho   = agg_c_wp$wp[,"m"],
             sd    = agg_c_wp$wp[,"sd"],
             type  = "within-probe (gate)")
)
dC$type <- factor(dC$type, levels = c("within-probe (gate)", "flat (unpartialled)"))

pC <- ggplot(dC, aes(layer, rho, colour = type, shape = type)) +
  geom_hline(yintercept = 0.10, linewidth = 0.3, colour = viz$muted, linetype = "dotted") +
  annotate("text", x = 8.4, y = 0.115, label = "DEAD 0.10", hjust = 0,
           size = 1.9, colour = viz$muted) +
  geom_errorbar(aes(ymin = rho - sd, ymax = rho + sd), width = 0.35, linewidth = 0.35) +
  geom_line(linewidth = 0.55) + geom_point(size = 1.8) +
  scale_colour_manual(values = c("within-probe (gate)" = viz$blue,
                                  "flat (unpartialled)" = viz$muted)) +
  scale_shape_manual(values  = c("within-probe (gate)" = 16,
                                  "flat (unpartialled)" = 4)) +
  scale_x_continuous(breaks = LAYERS) +
  coord_cartesian(ylim = c(0.20, 0.72), xlim = c(7.3, 14.7)) +
  labs(title = "within-probe vs flat", tag = "(c)",
       x = "edited layer", y = "$\\rho$ (key-cos, damage)") +
  theme_b6() + theme(legend.key.width = unit(11, "pt"))

# =====================================================================
# Panel D — perm-null distributions per layer (z-score bars)
# Visualise edit-level null: observed wp mean vs null (mean ± sd) as
# normalised z-scores; one point per seed (jitter) + layer mean
# =====================================================================
rows_d <- data.frame()
for (L in LAYERS) {
  for (ps in g1v2[[as.character(L)]]) {
    art <- sprintf("G1_stability_L%d_v2.json", L)
    nm  <- S(art, sprintf("per_seed[%s].editlevel_null_mean", ps$npz),
              ps$editlevel_null_mean)
    nsd <- S(art, sprintf("per_seed[%s].editlevel_null_std",  ps$npz),
              ps$editlevel_null_std)
    z   <- S(art, sprintf("per_seed[%s].editlevel_z",         ps$npz),
              ps$editlevel_z)
    rows_d <- rbind(rows_d, data.frame(layer = L, null_m = nm, null_sd = nsd, z = z))
  }
}
# aggregate mean z per layer, plus individual seed points
agg_d <- aggregate(z ~ layer, rows_d, mean)
agg_d$sd <- sapply(split(rows_d$z, rows_d$layer), sd)
# edit-level perm-p floor from the artifacts (never hard-code the value)
max_p_editlevel <- max(sapply(LAYERS, function(L) {
  art <- sprintf("G1_stability_L%d_v2.json", L)
  S(art, "aggregate.max_within_probe_perm_p_editlevel",
    .load(art)$aggregate$max_within_probe_perm_p_editlevel)
}))

pD <- ggplot(agg_d, aes(factor(layer), z)) +
  geom_hline(yintercept = 0, linewidth = 0.3, colour = viz$muted) +
  geom_col(fill = viz$blue, width = 0.55, alpha = 0.85) +
  geom_errorbar(aes(ymin = z - sd, ymax = z + sd),
                width = 0.25, linewidth = 0.5, colour = viz$ink2) +
  geom_jitter(data = rows_d, aes(x = factor(layer), y = z),
              position = position_jitter(seed = 42, width = 0.10), size = 1.5, colour = viz$ink2, alpha = 0.65, inherit.aes = FALSE) +
  labs(title = "perm-null $z$-scores (edit-level)", tag = "(d)",
       subtitle = sprintf("edit-level perm-$p \\leq %s$ for every seed",
                          formatC(max_p_editlevel, format = "f", digits = 3)),
       x = "edited layer", y = "$z$-score vs edit-level null") +
  theme_b6()

# =====================================================================
# Emit
# =====================================================================
fig_f1 <- (pA | pB) / (pC | pD) +
  plot_annotation(theme = theme(plot.margin = margin(2, 2, 2, 2)))

tikz(OUT_TEX, width = 7.16, height = 4.2, standAlone = FALSE, sanitize = FALSE,
     verbose = FALSE,
     packages = c("\\usepackage{tikz}",
                  "\\usepackage[active,tightpage,psfixbb]{preview}",
                  "\\PreviewEnvironment{pictureenvironment}",
                  "\\usepackage{times}", "\\usepackage{amsmath}",
                  "\\usepackage{amssymb}"))
print(fig_f1)
dev.off()

body   <- readLines(OUT_TEX, warn = FALSE)
body   <- body[!grepl("^% Created by tikzDevice", body)]
header <- c(
  "% =====================================================================",
  "% figF1_geometry_law.tex  — R/ggplot2 -> tikzDevice",
  "% Regenerate: Rscript submissions/ieee/figures-src/figF1_geometry_law.R",
  "% Provenance (canonical artifact :: field = plotted value):",
  prov,
  "% ====================================================================="
)
writeLines(c(header, body), OUT_TEX)
cat(sprintf("[ok] %s  (%d source lines)\n", basename(OUT_TEX), length(prov)))
