# D2 federation paper — F2 (Two-regime law) 4-panel figure.
# R/ggplot2 -> tikzDevice (house standard, matches make_figures.R / make_figF3_gate.R).
#
# Panel A: dose-response curves — HIGH-GAIN cells (destructive regime).
#          % SOURCE: edit-harness/results/merging/RG_map_evidence_REFIX20260801.json
#                    -> cells.<cell>.per_g.<g>.median_abs_drop_med3
#                    -> cells.<cell>.regime, cells.<cell>.gain
# Panel B: dose-response curves — LOW-GAIN cells (constructive regime).
#          % SOURCE: RG_map_evidence_REFIX20260801.json (same; regime = "low-gain")
# Panel C: architecture-dependent crossover g.
#          Crossover = first g at which frac_drop_negative > 0.5 (majority constructive).
#          % SOURCE: edit-harness/results/merging/RG_signed_reanalysis_REFIX20260801.json
#                    -> bundles.<cell>.cells.<g_sX>.frac_drop_negative (seed-avg per g)
#                    -> bundles.<cell>.group_sizes
#          Gain for regime colour:
#          % SOURCE: RG_gain_law_MERGED_REFIX20260730.json -> bundles.<cellRG>.gain_median_absdrop_per_dose
# Panel D: signed reanalysis — frac(drop<0) vs g, highlighting 14B constructive regime.
#          % SOURCE: RG_signed_reanalysis_REFIX20260801.json
#                    -> bundles.<cell>.cells.<g_sX>.frac_drop_negative (seed-avg per g)
#
# Every plotted number is read from the canonical JSONs listed above.
# No science value is hard-coded; GAIN_CUT is the protocol threshold (= 8).
# Palette: Okabe-Ito blue #0072B2 (high-gain) + vermillion #D55E00 (low-gain); CVD-safe.
# Run from this directory:
#   Rscript make_figF2_tworegime.R
suppressPackageStartupMessages({library(jsonlite); library(ggplot2); library(tikzDevice)})

HARNESS <- normalizePath(file.path("..", "..", "..", "edit-harness"))
MERG    <- file.path(HARNESS, "results", "merging")

mev  <- fromJSON(file.path(MERG, "RG_map_evidence_REFIX20260801.json"),
                 simplifyVector = FALSE)
sign <- fromJSON(file.path(MERG, "RG_signed_reanalysis_REFIX20260801.json"),
                 simplifyVector = FALSE)
gain <- fromJSON(file.path(MERG, "RG_gain_law_MERGED_REFIX20260730.json"),
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

tex_safe     <- function(s) gsub("_", "\\\\_", s)
lab_of       <- function(n) gsub("_RG$", "", gsub("_", " ", n))
fam_of       <- function(n) {
  if (grepl("^Llama", n)) "Llama" else if (grepl("^Mistral", n)) "Mistral"
  else if (grepl("^Qwen", n)) "Qwen" else if (grepl("^gemma", n)) "Gemma"
  else if (grepl("^Phi", n)) "Phi" else if (grepl("neox", n)) "GPT-NeoX" else "GPT-2"
}
shorten_cell <- function(x)
  tex_safe(sub("_RG$", "",
    sub("Mistral-Nemo-Base-2407", "Mistral-Nemo",
    sub("Qwen2\\.5-", "Q", sub("Llama-3\\.[12]-", "Ll",
    sub("gemma-2-", "G", sub("gpt-neox-20b", "NeoX",
    sub("gpt2-xl", "GPT2", sub("Phi-3\\.5-mini", "Phi", x)))))))))

# ---- shared dose-response table (panels A + B) ----
rows_dr <- list(); i <- 0
for (nm in names(mev$cells)) {
  b  <- mev$cells[[nm]]
  for (gs in names(b$per_g)) {
    pg <- b$per_g[[gs]]
    i  <- i + 1
    rows_dr[[i]] <- data.frame(
      cell   = nm,
      name   = tex_safe(sub(" RG$", "", lab_of(nm))),
      short  = shorten_cell(nm),
      family = fam_of(nm),
      regime = b$regime,
      gain   = as.numeric(b$gain),
      g      = as.integer(gs),
      med    = as.numeric(pg$median_abs_drop_med3),
      stringsAsFactors = FALSE)
  }
}
dDR <- do.call(rbind, rows_dr)
dDR$regcol <- ifelse(dDR$regime == "high-gain", "high-gain (destructive)",
                                                  "low-gain (constructive)")

# helper: annotate a handful of landmark curves at rightmost g
label_at_end <- function(df, nms, p, x_mult = 1.12) {
  sub_l <- df[df$g == max(df$g) & df$name %in% nms, ]
  sub_l$col <- ifelse(sub_l$regime == "high-gain", C_HIGH, C_LOW)
  for (j in seq_len(nrow(sub_l)))
    p <- p + annotate("text", x = max(df$g) * x_mult, y = sub_l$med[j],
                      label = sub_l$name[j], hjust = 0, size = 2.35,
                      colour = sub_l$col[j])
  p
}

# ---- Panel A: dose-response, HIGH-GAIN ----
dA <- dDR[dDR$regime == "high-gain", ]
p_A <- ggplot(dA, aes(g, med, group = name, colour = regcol)) +
  geom_line(linewidth = 0.45, alpha = 0.85) +
  geom_point(size = 1.0) +
  scale_x_log10(breaks = c(2, 3, 5, 10, 20), limits = c(2, 55)) +
  scale_y_log10(breaks = c(0.03, 0.3, 3, 20),
                labels = c("0.03", "0.3", "3", "20")) +
  scale_colour_manual(values = c("high-gain (destructive)" = C_HIGH,
                                  "low-gain (constructive)" = C_LOW),
                      guide = "none") +
  labs(x = "merge group size $g$ (log scale)",
       y = "median $|$drop$|$ per edit (logits, log scale)",
       title = "(A) dose-response: high-gain regime (destructive)") +
  theme_classic(base_size = 9) +
  theme(plot.title = element_text(size = 9, face = "plain"),
        plot.margin = margin(4, 8, 2, 6))

lnms_A <- c("Ll 1B L8 RG", "Ll 1B L12 RG", "Ll 8B L24 RG", "G 9b L31 RG")
lnms_A <- intersect(lnms_A, unique(dA$name))
# annotate top + bottom of the high-gain bundle — extrema computed ON the subset,
# indexed INTO the subset (the reviewer's indexing-bug catch: previously the
# subset's position was used as a row index of the full table, mislabeling curves)
sub_A   <- dA[dA$g == max(dA$g), ]
top_nm  <- sub_A$name[which.max(sub_A$med)]
bot_nm  <- sub_A$name[which.min(sub_A$med)]
anno_nms <- unique(c(top_nm, bot_nm))
p_A <- label_at_end(dA, anno_nms, p_A)

tikz("figF2A_dose_high.tex", width = 5.0, height = 3.1, standAlone = FALSE)
print(p_A); dev.off()
emit_tex("figF2A_dose_high.tex", c(
  "% SOURCE: submissions/d2-neurocomputing/zenodo-deposit/results/merging/RG_map_evidence_REFIX20260801.json :: cells.*.per_g.{2,3,5,10,20}.median_abs_drop_med3 + cells.*.regime (series selection); extrema labels computed on g=max subset"))
cat("wrote figF2A_dose_high.tex\n")

# ---- Panel B: dose-response, LOW-GAIN (constructive) ----
dB <- dDR[dDR$regime == "low-gain", ]
p_B <- ggplot(dB, aes(g, med, group = name, colour = regcol)) +
  geom_line(linewidth = 0.45, alpha = 0.85) +
  geom_point(size = 1.0) +
  scale_x_log10(breaks = c(2, 3, 5, 10, 20), limits = c(2, 55)) +
  scale_y_log10(breaks = c(0.003, 0.03, 0.3, 3),
                labels = c("0.003", "0.03", "0.3", "3")) +
  scale_colour_manual(values = c("high-gain (destructive)" = C_HIGH,
                                  "low-gain (constructive)" = C_LOW),
                      guide = "none") +
  labs(x = "merge group size $g$ (log scale)",
       y = "median $|$drop$|$ per edit (logits, log scale)",
       title = "(B) dose-response: low-gain regime (constructive)") +
  theme_classic(base_size = 9) +
  theme(plot.title = element_text(size = 9, face = "plain"),
        plot.margin = margin(4, 8, 2, 6))

sub_B    <- dB[dB$g == max(dB$g), ]
top_nm_B <- sub_B$name[which.max(sub_B$med)]
bot_nm_B <- sub_B$name[which.min(sub_B$med)]
p_B <- label_at_end(dB, unique(c(top_nm_B, bot_nm_B)), p_B)

tikz("figF2B_dose_low.tex", width = 5.0, height = 3.1, standAlone = FALSE)
print(p_B); dev.off()
emit_tex("figF2B_dose_low.tex", c(
  "% SOURCE: submissions/d2-neurocomputing/zenodo-deposit/results/merging/RG_map_evidence_REFIX20260801.json :: cells.*.per_g.{2,3,5,10,20}.median_abs_drop_med3 + cells.*.regime (series selection); extrema labels computed on g=max subset"))
cat("wrote figF2B_dose_low.tex\n")

# ---- Panel C: architecture-dependent crossover g ----
# For each bundle in RG_signed_reanalysis, find the first g at which the
# seed-averaged frac_drop_negative exceeds 0.5. Bundles that never cross = NA
# (displayed as "> max g"). Gain is joined from RG_gain_law_MERGED_REFIX for colour.
rows_c <- list(); ic <- 0
for (nm in names(sign$bundles)) {
  b <- sign$bundles[[nm]]
  gs_all <- sort(as.integer(b$group_sizes))
  # seed-averaged frac per g
  gfrac <- sapply(gs_all, function(g) {
    vals <- sapply(b$cells, function(cv) {
      if (as.integer(sub("^g(\\d+)_.*$", "\\1", names(b$cells)[1])) == g) return(NULL)
      NULL
    })
    # directly iterate b$cells, match key prefix g<g>_
    prefix <- sprintf("g%d_", g)
    matched <- b$cells[grepl(paste0("^", prefix), names(b$cells))]
    if (!length(matched)) return(NA_real_)
    mean(sapply(matched, function(cv) as.numeric(cv$frac_drop_negative)))
  })
  # first g where majority constructive (frac > 0.5)
  cross_idx <- which(gfrac > 0.5)
  cross_g   <- if (length(cross_idx)) gs_all[cross_idx[1]] else NA_integer_

  # gain for colour: look up in gain bundles (key = nm + "_RG")
  gain_key <- paste0(nm, "_RG")
  gv <- if (!is.null(gain$bundles[[gain_key]]))
    as.numeric(gain$bundles[[gain_key]]$gain_median_absdrop_per_dose) else NA_real_

  ic <- ic + 1
  rows_c[[ic]] <- data.frame(
    cell     = nm,
    short    = shorten_cell(paste0(nm, "_RG")),
    family   = fam_of(nm),
    gain     = gv,
    cross_g  = cross_g,
    max_g    = max(gs_all),
    stringsAsFactors = FALSE)
}
dC <- do.call(rbind, rows_c)
dC$regime <- ifelse(!is.na(dC$gain) & dC$gain >= GAIN_CUT,
                    "high-gain", "low-gain")
# display value: NA -> max_g + 0.5 to mark "no crossing found"
dC$plot_g <- ifelse(is.na(dC$cross_g), dC$max_g + 0.5, as.numeric(dC$cross_g))
dC$no_cross <- is.na(dC$cross_g)
dC <- dC[order(dC$regime, dC$plot_g), ]
dC$short_f <- factor(dC$short, levels = rev(dC$short))

tikz("figF2C_crossover_g.tex", width = 5.0, height = 4.0, standAlone = FALSE)
print(
  ggplot(dC, aes(plot_g, short_f, colour = regime, shape = no_cross)) +
    geom_vline(xintercept = 0.5, linewidth = 0.3, colour = "grey70") +
    geom_point(size = 2.2, stroke = 0.7) +
    geom_segment(data = dC[dC$no_cross, ],
                 aes(x = max_g * 0.95, xend = plot_g, y = short_f, yend = short_f),
                 arrow = grid::arrow(length = grid::unit(4, "pt"), type = "open"),
                 linewidth = 0.4) +
    scale_colour_manual(values = c("high-gain" = C_HIGH, "low-gain" = C_LOW),
                        labels = c("high-gain" = "gain $\\geq 8$",
                                   "low-gain"  = "gain $< 8$")) +
    scale_shape_manual(values = c(`FALSE` = 16, `TRUE` = 1),
                       labels = c(`FALSE` = "crossing found",
                                  `TRUE`  = "no crossing (arrow = lower bound)")) +
    labs(x = "first $g$ where frac(drop $<$ 0) $> 0.5$",
         y = NULL, colour = NULL, shape = NULL,
         title = "(C) architecture-dependent crossover $g$") +
    theme_classic(base_size = 9) +
    theme(axis.text.y = element_text(size = 6.5),
          legend.position = "bottom", legend.key.height = grid::unit(8, "pt"),
          legend.text = element_text(size = 7),
          plot.title = element_text(size = 9, face = "plain"),
          plot.margin = margin(4, 8, 2, 4))
)
dev.off()
emit_tex("figF2C_crossover_g.tex", c(
  "% SOURCE: submissions/d2-neurocomputing/zenodo-deposit/results/merging/RG_signed_reanalysis_REFIX20260801.json :: bundles.*.cells.g{2,3,5,10,20}.frac_drop_negative over all stored group_sizes {2,3,5,10,20,50,100} (crossover g per cell; 14B 0.813@2, 1B never crosses) + regime colouring from submissions/d2-neurocomputing/zenodo-deposit/results/merging/RG_gain_law_MERGED_REFIX20260730.json"))
cat("wrote figF2C_crossover_g.tex\n")

# ---- Panel D: signed reanalysis — frac(drop<0) vs g, 14B highlighted ----
# Shows ALL bundles as thin background lines; 14B Qwen constructive highlighted.
rows_d <- list(); id <- 0
for (nm in names(sign$bundles)) {
  b  <- sign$bundles[[nm]]
  gs <- as.integer(b$group_sizes)
  gain_key <- paste0(nm, "_RG")
  gv <- if (!is.null(gain$bundles[[gain_key]]))
    as.numeric(gain$bundles[[gain_key]]$gain_median_absdrop_per_dose) else NA_real_
  for (g in gs) {
    prefix  <- sprintf("g%d_", g)
    matched <- b$cells[grepl(paste0("^", prefix), names(b$cells))]
    if (!length(matched)) next
    frac_mean <- mean(sapply(matched, function(cv) as.numeric(cv$frac_drop_negative)))
    id <- id + 1
    rows_d[[id]] <- data.frame(
      cell    = nm,
      short   = shorten_cell(paste0(nm, "_RG")),
      family  = fam_of(nm),
      gain    = gv,
      g       = g,
      frac    = frac_mean,
      stringsAsFactors = FALSE)
  }
}
dD <- do.call(rbind, rows_d)
dD$regime <- ifelse(!is.na(dD$gain) & dD$gain >= GAIN_CUT, "high-gain", "low-gain")
dD$highlight14B <- dD$cell == "Qwen2.5-14B_L36"

tikz("figF2D_signed_reanalysis.tex", width = 5.0, height = 3.3, standAlone = FALSE)
print(
  ggplot() +
    geom_hline(yintercept = 0.5, linewidth = 0.3, colour = "grey55", linetype = "22") +
    # background: all non-14B bundles
    geom_line(data = dD[!dD$highlight14B, ],
              aes(g, frac, group = cell, colour = regime),
              linewidth = 0.3, alpha = 0.45) +
    # foreground: 14B highlighted
    geom_line(data = dD[dD$highlight14B, ],
              aes(g, frac, group = cell),
              colour = C_LOW, linewidth = 1.0) +
    geom_point(data = dD[dD$highlight14B, ],
               aes(g, frac), colour = C_LOW, size = 2.5, shape = 15) +
    annotate("text",
             x = max(dD$g[dD$highlight14B]) * 1.08,
             y = dD$frac[dD$highlight14B & dD$g == max(dD$g[dD$highlight14B])][1] + 0.05,
             label = "Qwen2.5-14B L36", hjust = 0,
             size = 2.4, colour = C_LOW) +
    annotate("text",
             x = max(dD$g[dD$highlight14B]) * 1.08,
             y = dD$frac[dD$highlight14B & dD$g == max(dD$g[dD$highlight14B])][1] - 0.05,
             label = "(constructive)", hjust = 0,
             size = 2.4, colour = C_LOW) +
    scale_x_log10(breaks = c(2, 3, 5, 10, 20)) +
    scale_y_continuous(limits = c(0, 1),
                       labels = function(x) sprintf("%.1f", x)) +
    scale_colour_manual(values = c("high-gain" = C_HIGH, "low-gain" = C_LOW),
                        labels = c("high-gain" = "gain $\\geq 8$",
                                   "low-gain"  = "gain $< 8$")) +
    labs(x = "merge group size $g$ (log scale)",
         y = "frac(drop $<$ 0), seed mean",
         colour = NULL,
         title = "(D) signed reanalysis: 14B constructive highlighted") +
    theme_classic(base_size = 9) +
    theme(legend.position = c(0.82, 0.18), legend.background = element_blank(),
          legend.key.height = grid::unit(8, "pt"),
          legend.text = element_text(size = 7),
          plot.title = element_text(size = 9, face = "plain"),
          plot.margin = margin(4, 8, 2, 6))
)
dev.off()
emit_tex("figF2D_signed_reanalysis.tex", c(
  "% SOURCE: submissions/d2-neurocomputing/zenodo-deposit/results/merging/RG_signed_reanalysis_REFIX20260801.json :: bundles.*.cells.g*.frac_drop_negative (all 19 bundles plotted; 14B 0.665@g20, 1B 0.000@g20 highlighted) + regime colouring from submissions/d2-neurocomputing/zenodo-deposit/results/merging/RG_gain_law_MERGED_REFIX20260730.json"))
cat("wrote figF2D_signed_reanalysis.tex\n")
cat("make_figF2_tworegime.R DONE\n")
