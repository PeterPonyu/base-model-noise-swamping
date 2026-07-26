# D2 federation paper figures — R/ggplot2 -> tikzDevice (house standard).
# SOURCE (figA): results/merging/RG_gain_law_20260715.json
# SOURCE (figB): results/merging/RG_crossterm_alignment_20260715.json
# Palette: Okabe-Ito blue #0072B2 (high-gain / destructive) + vermillion #D55E00
# (low-gain / constructive); validated CVD-safe (dataviz validator, all checks PASS);
# shape encodes family as secondary encoding. Run from this directory:
#   Rscript make_figures.R
suppressPackageStartupMessages({library(jsonlite); library(ggplot2); library(tikzDevice)})

HARNESS <- normalizePath(file.path("..", "..", "..", "edit-harness"))
gain <- fromJSON(file.path(HARNESS, "results/merging/RG_gain_law_20260715.json"))
# per-cell bootstrap CIs (SOURCE: RG_gain_holdout_20260716.json, 2000-rep obs bootstrap)
hold <- fromJSON(file.path(HARNESS, "results/merging/RG_gain_holdout_20260716.json"))
alig <- fromJSON(file.path(HARNESS, "results/merging/RG_crossterm_alignment_20260715.json"))

C_HIGH <- "#0072B2"; C_LOW <- "#D55E00"; GAIN_CUT <- 8

fam_of <- function(n) {
  if (grepl("^Llama", n)) "Llama" else if (grepl("^Mistral", n)) "Mistral"
  else if (grepl("^Qwen", n)) "Qwen" else if (grepl("^gemma", n)) "Gemma"
  else if (grepl("^Phi", n)) "Phi" else if (grepl("neox", n)) "GPT-NeoX" else "GPT-2"
}
lab_of <- function(n) gsub("_RG$", "", gsub("_", " ", n))

## ---------------- figA: gain vs constructive fraction (19 cells) ----------------
rows <- list()
for (n in names(gain$bundles)) {
  b <- gain$bundles[[n]]
  if (is.null(b$gain_median_absdrop_per_dose)) next
  h <- hold$cells[[n]]
  rows[[n]] <- data.frame(name = lab_of(n), family = fam_of(n),
                          gain = b$gain_median_absdrop_per_dose,
                          frac = b$frac_drop_negative,
                          gain_lo = h$gain_ci95[1], gain_hi = h$gain_ci95[2],
                          frac_lo = h$frac_ci95[1], frac_hi = h$frac_ci95[2])
}
dA <- do.call(rbind, rows)
dA$regime <- ifelse(dA$gain >= GAIN_CUT, "high-gain (destructive)", "low-gain (constructive)")

tikz("figA_gain_vs_frac.tex", width = 5.0, height = 3.1, standAlone = FALSE)
print(
  ggplot(dA, aes(gain, frac, colour = regime, shape = family)) +
    geom_hline(yintercept = 0.5, linewidth = 0.3, colour = "grey70", linetype = "22") +
    geom_errorbar(aes(ymin = frac_lo, ymax = frac_hi), width = 0, linewidth = 0.3, alpha = 0.55) +
    geom_errorbar(aes(xmin = gain_lo, xmax = gain_hi), orientation = "y", width = 0, linewidth = 0.3, alpha = 0.55) +
    geom_point(size = 2.1, stroke = 0.7) +
    annotate("text", x = 3.1 * 1.18, y = 0.836, label = "Qwen2.5-14B L36",
             hjust = 0, size = 2.65, colour = C_LOW) +
    annotate("text", x = 0.84, y = 0.725, label = "gpt2-xl L36",
             hjust = 0.5, vjust = -1.0, size = 2.65, colour = C_LOW) +
    scale_x_log10(breaks = c(0.1, 1, 3, 10, 30, 60), limits = c(0.07, 75)) +
    scale_colour_manual(values = c("high-gain (destructive)" = C_HIGH,
                                   "low-gain (constructive)" = C_LOW),
                        labels = c("high-gain (destructive)" = "gain $\\geq 8$ (destructive)",
                                   "low-gain (constructive)" = "gain $< 8$ (constructive)")) +
    scale_shape_manual(values = c(Llama = 16, Mistral = 17, Qwen = 15,
                                  Gemma = 3, Phi = 8, `GPT-2` = 4, `GPT-NeoX` = 7)) +
    labs(x = "perturbation gain (median $|$drop$|$/dose, log scale)",
         y = "constructive fraction  frac(drop $<$ 0)",
         colour = NULL, shape = NULL) +
    theme_classic(base_size = 10) +
    theme(legend.position = "right", legend.key.height = grid::unit(10, "pt"),
          legend.text = element_text(size = 8),
          plot.margin = margin(4, 8, 4, 10))
)
dev.off()

## ---------------- figB: g-resolved rho(proj, drop) profiles ----------------
rows <- list(); i <- 0
for (n in names(alig$bundles)) {
  b <- alig$bundles[[n]]
  if (is.null(b$cells)) next
  gn <- gain$bundles[[paste0(n, "_RG")]]$gain_median_absdrop_per_dose
  if (is.null(gn)) next
  for (g in c(2, 3, 5, 10, 20)) {
    vals <- c()
    for (s in b$seeds) {
      cell <- b$cells[[sprintf("g%d_s%d", g, s)]]
      if (!is.null(cell) && !is.null(cell$rho_proj_drop)) vals <- c(vals, cell$rho_proj_drop)
    }
    if (!length(vals)) next
    i <- i + 1
    rows[[i]] <- data.frame(name = lab_of(n), g = g, rho = mean(vals), gain = gn)
  }
}
dB <- do.call(rbind, rows)
dB$regime <- ifelse(dB$gain >= GAIN_CUT, "high-gain", "low-gain")
labB <- dB[dB$g == 20 & dB$name %in% c("Qwen2.5-14B L36", "gpt2-xl L36", "Llama-3.1-8B L24", "Mistral-Nemo-Base-2407 L30"), ]
labB$col <- ifelse(labB$regime == "high-gain", C_HIGH, C_LOW)
labB$name <- sub("Mistral-Nemo-Base-2407", "Mistral-Nemo-12B", labB$name)

tikz("figB_g_profiles.tex", width = 5.0, height = 3.1, standAlone = FALSE)
p <- ggplot(dB, aes(g, rho, group = name, colour = regime)) +
    geom_hline(yintercept = 0, linewidth = 0.3, colour = "grey55") +
    geom_line(linewidth = 0.45, alpha = 0.85) +
    geom_point(size = 1.1) +
    scale_x_log10(breaks = c(2, 3, 5, 10, 20), limits = c(2, 45)) +
    scale_colour_manual(values = c(`high-gain` = C_HIGH, `low-gain` = C_LOW),
                        labels = c(`high-gain` = "gain $\\geq 8$", `low-gain` = "gain $< 8$")) +
    labs(x = "merge group size $g$ (log scale)",
         y = "$\\rho(\\mathrm{proj},\\ \\mathrm{drop})$",
         colour = NULL) +
    theme_classic(base_size = 10) +
    theme(legend.position = "right", legend.key.height = grid::unit(10, "pt"),
          legend.text = element_text(size = 8),
          plot.margin = margin(4, 8, 4, 10))
for (i in seq_len(nrow(labB)))
  p <- p + annotate("text", x = 20 * 1.12, y = labB$rho[i], label = labB$name[i],
                    hjust = 0, size = 2.65, colour = labB$col[i])
print(p)
dev.off()
cat("wrote figA_gain_vs_frac.tex figB_g_profiles.tex\n")

## ---------------- figC: admission-benefit bars (SOURCE: RG_admission_benefit_20260715.json) ----------------
ben <- fromJSON(file.path(HARNESS, "results/merging/RG_admission_benefit_20260715.json"))
agg <- ben$regime_aggregates_small_g
rows <- list(); i <- 0
for (reg in c("high_gain","low_gain")) for (q in c("q25","q50")) {
  a <- agg[[paste0(reg,"_",q)]]
  grp <- paste0(ifelse(reg=="high_gain","high-gain","low-gain"), " ", ifelse(q=="q25","25\\%","50\\%"))
  i <- i+1; rows[[i]] <- data.frame(group=grp, method="geometry-ordered", benefit=a$benefit_geometry_mean)
  i <- i+1; rows[[i]] <- data.frame(group=grp, method="magnitude-only",  benefit=a$benefit_magnitude_mean)
  for (s in names(a$benefit_geometry_by_seed)) {
    i <- i+1; rows[[i]] <- data.frame(group=grp, method="seedpt", benefit=a$benefit_geometry_by_seed[[s]])
  }
}
dC <- do.call(rbind, rows)
dC$group <- factor(dC$group, levels=c("high-gain 25\\%","high-gain 50\\%","low-gain 25\\%","low-gain 50\\%"))
bars <- dC[dC$method != "seedpt",]
pts  <- dC[dC$method == "seedpt",]
tikz("figC_admission_benefit.tex", width = 5.0, height = 2.9, standAlone = FALSE)
print(
  ggplot(bars, aes(group, benefit, fill = method)) +
    geom_col(position = position_dodge(width = 0.72), width = 0.62) +
    geom_point(data = pts, aes(x = group, y = benefit), inherit.aes = FALSE,
               position = position_nudge(x = -0.18), size = 1.1, shape = 21,
               fill = "white", colour = "#0072B2", stroke = 0.5) +
    geom_text(aes(label = sprintf("%.2f", benefit)),
              position = position_dodge(width = 0.72), vjust = -0.45, size = 2.65) +
    scale_fill_manual(values = c(`geometry-ordered` = "#0072B2", `magnitude-only` = "#B0B0B0")) +
    scale_y_continuous(expand = expansion(mult = c(0, 0.18))) +
    labs(x = NULL, y = "damage avoided per admitted edit (logits)", fill = NULL) +
    theme_classic(base_size = 10) +
    theme(legend.position = c(0.82, 0.88), legend.background = element_blank(),
          legend.key.height = grid::unit(10, "pt"), legend.text = element_text(size = 8),
          plot.margin = margin(6, 8, 2, 10))
)
dev.off()
cat("wrote figC_admission_benefit.tex\n")

## ---------------- figD: damage dose-response (SOURCE: RG_map_evidence_20260716.json) ----------------
mev <- fromJSON(file.path(HARNESS, "results/merging/RG_map_evidence_20260716.json"))
rows <- list(); i <- 0
for (n in names(mev$cells)) {
  b <- mev$cells[[n]]
  for (g in names(b$per_g)) {
    i <- i + 1
    rows[[i]] <- data.frame(name = gsub(" RG$", "", lab_of(n)), family = fam_of(n),
                            gain = b$gain, regime = b$regime, g = as.integer(g),
                            med = b$per_g[[g]]$median_abs_drop_med3)
  }
}
dD <- do.call(rbind, rows)
dD$regcol <- ifelse(dD$regime == "high-gain", "high-gain (destructive)", "low-gain (constructive)")
labD <- dD[dD$g == 20 & dD$name %in% c("Llama-3.2-1B L12", "gpt-neox-20b L33", "Qwen2.5-14B L36", "Llama-3.1-8B L24"), ]
labD$name <- sub("gpt-neox-20b", "GPT-NeoX-20B", labD$name)
labD$col <- ifelse(labD$regime == "high-gain", C_HIGH, C_LOW)

tikz("figD_dose_response.tex", width = 5.0, height = 3.1, standAlone = FALSE)
p <- ggplot(dD, aes(g, med, group = name, colour = regcol)) +
    geom_line(linewidth = 0.4, alpha = 0.8) +
    geom_point(size = 1.0) +
    scale_x_log10(breaks = c(2, 3, 5, 10, 20), limits = c(2, 55)) +
    scale_y_log10(breaks = c(0.003, 0.03, 0.3, 3, 20),
                  labels = c("0.003", "0.03", "0.3", "3", "20")) +
    scale_colour_manual(values = c(`high-gain (destructive)` = C_HIGH,
                                   `low-gain (constructive)` = C_LOW),
                        labels = c(`high-gain (destructive)` = "gain $\\geq 8$ (destructive)",
                                   `low-gain (constructive)` = "gain $< 8$ (constructive)")) +
    labs(x = "merge group size $g$ (log scale)",
         y = "median $|$drop$|$ per edit (logits, log scale)",
         colour = NULL) +
    theme_classic(base_size = 10) +
    theme(legend.position = c(0.80, 0.14), legend.background = element_blank(),
          legend.key.height = grid::unit(10, "pt"), legend.text = element_text(size = 8),
          plot.margin = margin(4, 8, 4, 10))
for (j in seq_len(nrow(labD)))
  p <- p + annotate("text", x = 20 * 1.12, y = labD$med[j], label = labD$name[j],
                    hjust = 0, size = 2.65, colour = labD$col[j])
print(p)
dev.off()
cat("wrote figD_dose_response.tex\n")

## ---------------- figE: gate evidence, partial rho by g (SOURCE: RG_map_evidence_20260716.json) ----------------
rows <- list(); i <- 0
for (n in names(mev$cells)) {
  b <- mev$cells[[n]]
  for (g in names(b$per_g)) {
    pg <- b$per_g[[g]]
    i <- i + 1
    rows[[i]] <- data.frame(name = gsub(" RG$", "", lab_of(n)), regime = b$regime,
                            g = as.integer(g),
                            mid = pg$partial_rho_mean, lo = pg$partial_rho_min,
                            hi = pg$partial_rho_max,
                            sat = mean(unlist(pg$saturated)) > 0.5)
  }
}
dE <- do.call(rbind, rows)
dE$panel <- ifelse(dE$regime == "high-gain", "gain $\\geq 8$ cells", "gain $< 8$ cells")

tikz("figE_gate_evidence.tex", width = 5.0, height = 3.3, standAlone = FALSE)
print(
  ggplot(dE, aes(g, mid, group = name, colour = regime)) +
    geom_hline(yintercept = 0.15, linewidth = 0.35, colour = "grey40", linetype = "22") +
    geom_hline(yintercept = 0, linewidth = 0.25, colour = "grey75") +
    geom_text(data = data.frame(g = 2.6, mid = 0.215, name = "gatelab",
                                 regime = "low-gain", panel = "gain $< 8$ cells"),
              label = "gate $+0.15$", size = 2.65, colour = "grey30", show.legend = FALSE) +
    geom_ribbon(aes(ymin = lo, ymax = hi, fill = regime), alpha = 0.12, colour = NA) +
    geom_line(linewidth = 0.4, alpha = 0.85) +
    geom_point(aes(shape = sat), size = 1.1) +
    facet_wrap(~panel) +
    scale_x_log10(breaks = c(2, 3, 5, 10, 20)) +
    scale_shape_manual(values = c(`FALSE` = 16, `TRUE` = 1),
                       labels = c(`FALSE` = "unsaturated", `TRUE` = "saturated"),
                       name = NULL) +
    scale_colour_manual(values = c(`high-gain` = C_HIGH, `low-gain` = C_LOW), guide = "none") +
    scale_fill_manual(values = c(`high-gain` = C_HIGH, `low-gain` = C_LOW), guide = "none") +
    labs(x = "merge group size $g$ (log scale)",
         y = "partial $\\rho(I_{\\cos},\\mathrm{drop} \\mid I_{\\mathrm{mag}})$") +
    theme_classic(base_size = 10) +
    theme(legend.position = "bottom", legend.text = element_text(size = 8),
          strip.background = element_rect(linewidth = 0.4),
          plot.margin = margin(4, 8, 2, 10))
)
dev.off()
cat("wrote figE_gate_evidence.tex\n")
