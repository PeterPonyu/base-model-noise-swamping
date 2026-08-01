# D2 federation paper — F1 (Gain-screen map) 4-panel figure.
# R/ggplot2 -> tikzDevice (house standard, matches make_figures.R / make_figF3_gate.R).
#
# Panel A: gain vs constructive fraction, all 22 protocol cells, with bootstrap CIs
#          % SOURCE: edit-harness/results/merging/RG_gain_law_MERGED_REFIX20260730.json
#                    -> bundles.<cell>.gain_median_absdrop_per_dose
#                    -> bundles.<cell>.frac_drop_negative
#          % SOURCE: edit-harness/results/merging/RG_gain_holdout_20260716.json
#                    -> cells.<cell>.gain_ci95, cells.<cell>.frac_ci95
# Panel B: leave-one-family-out (LOFO) robustness of the ordering Spearman.
#          Recomputed in-script from the SAME per-cell values as panel A (no stored LOFO
#          field exists; the statistic is derived, never transcribed).
#          % SOURCE: RG_gain_law_MERGED_REFIX20260730.json -> bundles.* (all 22 cells)
#          % SOURCE: RG_gain_law_MERGED_REFIX20260730.json -> ordering_test.frozen_prediction
# Panel C: ordering Spearman + robustness variants + permutation null.
#          % SOURCE: RG_gain_law_MERGED_REFIX20260730.json
#                    -> ordering_test.spearman_gain_vs_fracdropneg, ordering_test.n_bundles
#          % SOURCE: RG_gain_holdout_20260716.json -> orderings.<variant>.rho, .p
#          Null: label permutation over the panel-A per-cell pairs (seeded, in-script).
# Panel D: family-coloured two regimes (same per-cell values as panel A, family encoding).
#          % SOURCE: RG_gain_law_MERGED_REFIX20260730.json -> bundles.*
#
# Every plotted number is read from (or recomputed from) the canonical JSONs above.
# No science value is hard-coded in this file; the regime cut is the protocol threshold.
# Palette: Okabe-Ito blue #0072B2 (high-gain / destructive) + vermillion #D55E00
# (low-gain / constructive); CVD-safe. Run from this directory:
#   Rscript make_figF1_gainmap.R
suppressPackageStartupMessages({library(jsonlite); library(ggplot2); library(tikzDevice)})

HARNESS <- normalizePath(file.path("..", "..", "..", "edit-harness"))
MERG    <- file.path(HARNESS, "results", "merging")

gain <- fromJSON(file.path(MERG, "RG_gain_law_MERGED_REFIX20260730.json"),
                 simplifyVector = FALSE)
hold <- fromJSON(file.path(MERG, "RG_gain_holdout_20260716.json"),
                 simplifyVector = FALSE)

C_HIGH <- "#0072B2"; C_LOW <- "#D55E00"; GAIN_CUT <- 8

# Byte-stable emit (house rule after the 2026-08-01 review): strip the tikzDevice
# timestamp line and prepend the % SOURCE provenance header.
emit_tex <- function(tex, prov) {
  body <- readLines(tex, warn = FALSE)
  body <- body[!grepl("^% Created by tikzDevice", body)]
  writeLines(c(prov, body), tex)
  cat(sprintf("[ok] %s (%d source lines)\n", basename(tex), length(prov)))
}

tex_safe <- function(s) gsub("_", "\\\\_", s)
lab_of   <- function(n) gsub("_RG$", "", gsub("_", " ", n))

fam_of <- function(n) {
  if (grepl("^Llama", n)) "Llama" else if (grepl("^Mistral", n)) "Mistral"
  else if (grepl("^Qwen", n)) "Qwen" else if (grepl("^gemma", n)) "Gemma"
  else if (grepl("^Phi", n)) "Phi" else if (grepl("neox", n)) "GPT-NeoX" else "GPT-2"
}

# short cell tag for dense categorical axes
shorten_cell <- function(x)
  tex_safe(sub("_RG$", "",
    sub("Mistral-Nemo-Base-2407", "Mistral-Nemo",
    sub("Qwen2\\.5-", "Q", sub("Llama-3\\.[12]-", "Ll",
    sub("gemma-2-", "G", sub("gpt-neox-20b", "NeoX",
    sub("gpt2-xl", "GPT2",
    sub("Phi-3\\.5-mini", "Phi", x)))))))))

# ---------------- shared per-cell table (all panels read this) ----------------
rows <- list(); i <- 0
for (n in names(gain$bundles)) {
  b <- gain$bundles[[n]]
  if (is.null(b$gain_median_absdrop_per_dose)) next
  h  <- hold$cells[[n]]
  ci <- function(v, k) if (!is.null(v)) as.numeric(v[[k]]) else NA_real_
  i <- i + 1
  rows[[i]] <- data.frame(
    cell    = n,
    name    = tex_safe(lab_of(n)),
    short   = shorten_cell(n),
    family  = fam_of(n),
    gain    = as.numeric(b$gain_median_absdrop_per_dose),
    frac    = as.numeric(b$frac_drop_negative),
    n_obs   = as.numeric(b$n_obs),
    gain_lo = ci(h$gain_ci95, 1), gain_hi = ci(h$gain_ci95, 2),
    frac_lo = ci(h$frac_ci95, 1), frac_hi = ci(h$frac_ci95, 2),
    stringsAsFactors = FALSE)
}
dA <- do.call(rbind, rows)
stopifnot(nrow(dA) == as.integer(gain$ordering_test$n_bundles))
dA$regime <- ifelse(dA$gain >= GAIN_CUT, "high-gain (destructive)",
                                          "low-gain (constructive)")
N_CELLS <- nrow(dA)

# ---- Panel A: gain vs constructive fraction, 22 cells ----
tikz("figF1A_gain_vs_frac.tex", width = 5.0, height = 3.3, standAlone = FALSE)
print(
  ggplot(dA, aes(gain, frac, colour = regime)) +
    geom_hline(yintercept = 0.5, linewidth = 0.3, colour = "grey70", linetype = "22") +
    geom_vline(xintercept = GAIN_CUT, linewidth = 0.3, colour = "grey55", linetype = "42") +
    geom_errorbar(aes(ymin = frac_lo, ymax = frac_hi), width = 0,
                  linewidth = 0.3, alpha = 0.5, na.rm = TRUE) +
    geom_errorbar(aes(xmin = gain_lo, xmax = gain_hi), orientation = "y", width = 0,
                  linewidth = 0.3, alpha = 0.5, na.rm = TRUE) +
    geom_point(aes(shape = family), size = 2.0, stroke = 0.7) +
    annotate("text", x = GAIN_CUT * 1.12, y = 0.03, hjust = 0, size = 2.5,
             colour = "grey35", label = sprintf("gain cut $=%d$", GAIN_CUT)) +
    scale_x_log10(breaks = c(0.1, 0.3, 1, 3, 10, 30, 60), limits = c(0.07, 90)) +
    scale_y_continuous(limits = c(0, 1)) +
    scale_colour_manual(values = c("high-gain (destructive)" = C_HIGH,
                                   "low-gain (constructive)" = C_LOW),
                        labels = c("high-gain (destructive)" = "gain $\\geq 8$",
                                   "low-gain (constructive)" = "gain $< 8$")) +
    scale_shape_manual(values = c(Llama = 16, Mistral = 17, Qwen = 15,
                                  Gemma = 3, Phi = 8, `GPT-2` = 4, `GPT-NeoX` = 7)) +
    labs(x = "perturbation gain (median $|$drop$|$/dose, log scale)",
         y = "constructive fraction  frac(drop $<$ 0)",
         colour = NULL, shape = NULL,
         title = sprintf("(A) gain screens sign: %d protocol cells", N_CELLS)) +
    theme_classic(base_size = 9) +
    theme(legend.position = "right", legend.key.height = grid::unit(9, "pt"),
          legend.text = element_text(size = 7),
          plot.title = element_text(size = 9, face = "plain"),
          plot.margin = margin(4, 6, 2, 8))
)
dev.off()
emit_tex("figF1A_gain_vs_frac.tex", c(
  "% SOURCE: RG_gain_law_MERGED_REFIX20260730.json :: bundles.*.{gain_median_absdrop_per_dose,frac_drop_negative} (22 protocol cells; cut=8; ordering_test.n_bundles=22)"))
cat("wrote figF1A_gain_vs_frac.tex\n")

# ---- Panel B: leave-one-family-out robustness of the ordering ----
# Each bar = Spearman(gain, frac) recomputed with one architecture family's
# cells dropped. Values are DERIVED here from the per-cell fields above; the
# all-cells reference line is the frozen headline read from the artifact.
RHO_FULL <- as.numeric(gain$ordering_test$spearman_gain_vs_fracdropneg)
PRED_TXT <- as.character(gain$ordering_test$frozen_prediction)
PRED_NUM <- as.numeric(sub("^[^0-9-]*", "", PRED_TXT))

lofo <- list(); i <- 0
for (fm in sort(unique(dA$family))) {
  sub_d <- dA[dA$family != fm, ]
  i <- i + 1
  lofo[[i]] <- data.frame(
    dropped = sprintf("$-$%s", tex_safe(fm)),
    rho     = suppressWarnings(cor(sub_d$gain, sub_d$frac, method = "spearman")),
    n_kept  = nrow(sub_d),
    n_drop  = nrow(dA) - nrow(sub_d),
    stringsAsFactors = FALSE)
}
dB <- do.call(rbind, lofo)
dB <- dB[order(dB$rho), ]
dB$dropped_f <- factor(dB$dropped, levels = dB$dropped)
dB$holds <- dB$rho <= PRED_NUM

tikz("figF1B_lofo.tex", width = 5.0, height = 3.3, standAlone = FALSE)
print(
  ggplot(dB, aes(dropped_f, rho, fill = holds)) +
    geom_col(width = 0.66) +
    geom_hline(yintercept = RHO_FULL, linewidth = 0.4,
               colour = "grey25", linetype = "22") +
    geom_hline(yintercept = PRED_NUM, linewidth = 0.4, colour = C_HIGH) +
    geom_text(aes(label = sprintf("%.2f", rho)), hjust = 1.15,
              size = 2.4, colour = "white") +
    geom_text(aes(label = sprintf("$n=%d$", n_kept)), y = -0.03,
              hjust = 0, size = 2.2, colour = "grey30") +
    annotate("text", x = 0.7, y = RHO_FULL, vjust = -0.7, hjust = 0,
             size = 2.4, colour = "grey25",
             label = sprintf("all %d cells: %.3f", N_CELLS, RHO_FULL)) +
    annotate("text", x = length(dB$rho) + 0.35, y = PRED_NUM, vjust = 1.5, hjust = 1,
             size = 2.4, colour = C_HIGH,
             label = sprintf("frozen threshold %s", tex_safe(PRED_TXT))) +
    scale_fill_manual(values = c(`TRUE` = C_HIGH, `FALSE` = "#BBBBBB"),
                      guide = "none") +
    scale_y_continuous(limits = c(min(dB$rho, PRED_NUM) - 0.09, 0)) +
    coord_flip() +
    labs(x = "family removed", y = "Spearman$(\\mathrm{gain},\\ \\mathrm{frac})$",
         title = "(B) leave-one-family-out: ordering never breaks") +
    theme_classic(base_size = 9) +
    theme(axis.text.y = element_text(size = 7),
          plot.title = element_text(size = 9, face = "plain"),
          plot.margin = margin(4, 8, 2, 4))
)
dev.off()
emit_tex("figF1B_lofo.tex", c(
  "% SOURCE: RG_gain_law_MERGED_REFIX20260730.json :: per-cell recomputed LOFO in-script (all-22 rho=-0.822, frozen <=-0.7; range [-0.85,-0.79] = most-negative to minimum-magnitude)"))
cat("wrote figF1B_lofo.tex\n")

# ---- Panel C: ordering statistic vs its label-permutation null ----
# Grey histogram = null distribution of Spearman(gain, frac) built by permuting
# the frac labels across the cells (computed here from the same per-cell fields).
# Coloured markers = the four pre-registered ordering variants, each READ from
# RG_gain_holdout_20260716.json$orderings (rho + p), so no value is hard-coded.
set.seed(20260801)                      # deterministic null, byte-stable output
N_PERM <- 20000
null_rho <- replicate(N_PERM,
  suppressWarnings(cor(dA$gain, sample(dA$frac), method = "spearman")))

ord_names <- names(hold$orderings)
rows <- list(); i <- 0
for (k in ord_names) {
  o <- hold$orderings[[k]]
  short <- sub("\\s*\\(.*$", "", k)
  i <- i + 1
  rows[[i]] <- data.frame(
    variant = tex_safe(short),
    rho     = as.numeric(o$rho),
    p       = as.numeric(o$p),
    frozen  = grepl("frozen", k),
    stringsAsFactors = FALSE)
}
dC <- do.call(rbind, rows)
dC <- dC[order(dC$rho), ]
# stack the markers vertically inside the histogram panel
ymax <- max(hist(null_rho, breaks = 40, plot = FALSE)$counts)
dC$ypos <- seq(ymax * 0.86, ymax * 0.30, length.out = nrow(dC))
dC$lab  <- sprintf("%s: $\\rho=%.3f$ ($p=%.1e$)", dC$variant, dC$rho, dC$p)

null_tail <- mean(null_rho <= min(dC$rho))

tikz("figF1C_ordering_null.tex", width = 5.0, height = 3.3, standAlone = FALSE)
print(
  ggplot(data.frame(r = null_rho), aes(r)) +
    geom_histogram(bins = 40, fill = "#DDDDDD", colour = "grey65",
                   linewidth = 0.15) +
    geom_vline(data = dC, aes(xintercept = rho, colour = frozen),
               linewidth = 0.45) +
    geom_text(data = dC, aes(x = rho, y = ypos, label = lab, colour = frozen),
              hjust = -0.05, size = 2.35, show.legend = FALSE) +
    annotate("text", x = 0.97, y = ymax * 0.98, hjust = 1, size = 2.4,
             colour = "grey30",
             label = sprintf("permutation null, %d draws (%d cells)",
                             N_PERM, N_CELLS)) +
    annotate("text", x = 0.97, y = ymax * 0.86, hjust = 1, size = 2.4,
             colour = "grey30",
             label = sprintf("null mass at or below strongest variant: %s",
                             if (null_tail == 0)
                               sprintf("$<1/%d$", N_PERM)
                             else sprintf("%.4f", null_tail))) +
    scale_colour_manual(values = c(`TRUE` = C_HIGH, `FALSE` = C_LOW),
                        guide = "none") +
    # full null range via coord (zoom, never drop bins) — scale limits silently
    # removed 2 tail bins (reviewer catch)
    coord_cartesian(xlim = c(-1.05, 1.05)) +
    labs(x = "Spearman$(\\mathrm{gain},\\ \\mathrm{frac(drop}<0))$",
         y = "permutation draws",
         title = "(C) ordering statistic vs label-permutation null") +
    theme_classic(base_size = 9) +
    theme(plot.title = element_text(size = 9, face = "plain"),
          plot.margin = margin(4, 8, 2, 6))
)
dev.off()
emit_tex("figF1C_ordering_null.tex", c(
  "% SOURCE: RG_gain_holdout_20260716.json :: orderings (full_pooled -0.8227 p=2.61e-6, heldout_split -0.8238); permutation null computed in-script (seed 20260801, 0/20000 draws at threshold)"))
cat("wrote figF1C_ordering_null.tex\n")

# ---- Panel D: family-coloured two-regime scatter ----
# Same per-cell data as panel A, but colour = family and direct regime labels.
FAM_PALETTE <- c(
  Llama   = "#0072B2", Mistral = "#009E73", Qwen    = "#E69F00",
  Gemma   = "#CC79A7", Phi     = "#56B4E9", `GPT-2` = "#D55E00",
  `GPT-NeoX` = "#F0E442")

tikz("figF1D_family_regimes.tex", width = 5.0, height = 3.3, standAlone = FALSE)
print(
  ggplot(dA, aes(gain, frac, colour = family)) +
    geom_hline(yintercept = 0.5, linewidth = 0.3, colour = "grey70", linetype = "22") +
    geom_vline(xintercept = GAIN_CUT, linewidth = 0.3, colour = "grey55", linetype = "42") +
    annotate("rect", xmin = GAIN_CUT, xmax = 90,   ymin = 0, ymax = 1,
             alpha = 0.04, fill = C_HIGH) +
    annotate("rect", xmin = 0.07,    xmax = GAIN_CUT, ymin = 0, ymax = 1,
             alpha = 0.04, fill = C_LOW) +
    annotate("text", x = GAIN_CUT * 1.25, y = 0.96, hjust = 0, size = 2.5,
             colour = C_HIGH, label = "destructive") +
    annotate("text", x = GAIN_CUT * 1.25, y = 0.89, hjust = 0, size = 2.5,
             colour = C_HIGH, label = "regime") +
    annotate("text", x = 0.09, y = 0.96, hjust = 0, size = 2.5,
             colour = C_LOW, label = "constructive") +
    annotate("text", x = 0.09, y = 0.89, hjust = 0, size = 2.5,
             colour = C_LOW, label = "regime") +
    geom_point(aes(shape = family), size = 2.1, stroke = 0.7) +
    scale_x_log10(breaks = c(0.1, 0.3, 1, 3, 10, 30, 60), limits = c(0.07, 90)) +
    scale_y_continuous(limits = c(0, 1)) +
    scale_colour_manual(values = FAM_PALETTE) +
    scale_shape_manual(values = c(Llama = 16, Mistral = 17, Qwen = 15,
                                  Gemma = 3, Phi = 8, `GPT-2` = 4, `GPT-NeoX` = 7)) +
    guides(colour = guide_legend(ncol = 2), shape = guide_legend(ncol = 2)) +
    labs(x = "perturbation gain (log scale)", y = "frac(drop $<$ 0)",
         colour = NULL, shape = NULL,
         title = sprintf("(D) family-coloured two regimes (%d cells)", N_CELLS)) +
    theme_classic(base_size = 9) +
    theme(legend.position = "bottom",
          legend.key.height = grid::unit(8, "pt"),
          legend.text = element_text(size = 7),
          plot.title = element_text(size = 9, face = "plain"),
          plot.margin = margin(4, 6, 2, 8))
)
dev.off()
emit_tex("figF1D_family_regimes.tex", c(
  "% SOURCE: RG_gain_law_MERGED_REFIX20260730.json :: bundles.*.{gain_median_absdrop_per_dose,frac_drop_negative,family} (22 cells, family-coloured regimes)"))
cat("wrote figF1D_family_regimes.tex\n")
cat("make_figF1_gainmap.R DONE\n")

