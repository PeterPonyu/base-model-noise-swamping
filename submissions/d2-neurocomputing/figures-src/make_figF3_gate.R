# D2 federation paper — F3 (Geometry gate) 4-panel figure.
# R/ggplot2 -> tikzDevice (house standard, matches make_figures.R).
#
# Panel A: partial rho(I_cos | I_mag) by g, all 22 cells (SOURCE: RG_map_evidence_REFIX20260801.json)
# Panel B: c2 coherence bands by (cell, g)              (SOURCE: RG_map_evidence_REFIX20260801.json)
# Panel C: permutation null FPR across all 22 cells      (SOURCE: perm_null_allcells/*.json)
# Panel D: two-boundary annotation (geometry-valid g<=5; damage-gradated g=10)
#          (SOURCE: per-cell verdicts in RG_operating_curve_table.json files via RG_map_evidence)
#
# Palette: Okabe-Ito blue #0072B2 (high-gain) + vermillion #D55E00 (low-gain); CVD-safe.
# Run from this directory:  Rscript make_figF3_gate.R

suppressPackageStartupMessages({
  library(jsonlite)
  library(ggplot2)
  library(tikzDevice)
})

HARNESS <- normalizePath(file.path("..", "..", "..", "edit-harness"))
DEPOSIT <- normalizePath(file.path("..", "..", "..", "submissions",
                                   "d2-neurocomputing", "zenodo-deposit",
                                   "results", "merging"))

C_HIGH <- "#0072B2"; C_LOW <- "#D55E00"; GAIN_CUT <- 8

# Byte-stable emit (house rule after the 2026-08-01 review): strip the tikzDevice
# timestamp line and prepend the % SOURCE provenance header so two clean runs are
# byte-identical and every panel carries exact artifact :: field provenance.
emit_tex <- function(tex, prov) {
  body <- readLines(tex, warn = FALSE)
  body <- body[!grepl("^% Created by tikzDevice", body)]
  writeLines(c(prov, body), tex)
  cat(sprintf("[ok] %s (%d source lines)\n", basename(tex), length(prov)))
}

# Escape underscores for LaTeX axis labels (tikzDevice renders label text via TeX)
tex_safe <- function(s) gsub("_", "\\\\_", s)

fam_of <- function(n) {
  if (grepl("^Llama-3\\.2-1B", n)) "Llama-1B"
  else if (grepl("^Llama-3\\.2-3B", n)) "Llama-3B"
  else if (grepl("^Llama-3\\.1-8B", n)) "Llama-8B"
  else if (grepl("^Mistral-7B", n)) "Mistral-7B"
  else if (grepl("^Mistral-Nemo", n)) "Mistral-12B"
  else if (grepl("^Qwen2\\.5-1\\.5B", n)) "Qwen-1.5B"
  else if (grepl("^Qwen2\\.5-3B", n)) "Qwen-3B"
  else if (grepl("^Qwen2\\.5-7B", n)) "Qwen-7B"
  else if (grepl("^Qwen2\\.5-14B", n)) "Qwen-14B"
  else if (grepl("^gemma-2-2b", n)) "Gemma-2B"
  else if (grepl("^gemma-2-9b", n)) "Gemma-9B"
  else if (grepl("^Phi-3\\.5.*L16", n)) "Phi-L16"
  else if (grepl("^Phi-3\\.5.*L24", n)) "Phi-L24"
  else if (grepl("^gpt2-xl_L24", n)) "GPT-2-XL L24"
  else if (grepl("^gpt2-xl_L36", n)) "GPT-2-XL L36"
  else if (grepl("^gpt-neox", n)) "GPT-NeoX-20B"
  else n
}

# --------------- load map evidence ---------------
mev <- fromJSON(file.path(DEPOSIT, "RG_map_evidence_REFIX20260801.json"),
                simplifyVector = FALSE)
cells_list <- mev$cells

rows <- list(); i <- 0
for (nm in names(cells_list)) {
  b   <- cells_list[[nm]]
  reg <- b$regime
  gv  <- b$gain
  for (gs in c(2, 3, 5, 10, 20)) {
    pg <- b$per_g[[as.character(gs)]]
    if (is.null(pg)) next
    c2_pass <- sum(unlist(pg$c2_coherent)) >= 2   # majority of 3 seeds
    nn_pass  <- sum(unlist(pg$non_negligible)) >= 2
    sat_any  <- sum(unlist(pg$saturated)) >= 2
    i <- i + 1
    rows[[i]] <- data.frame(
      cell    = nm,
      family  = fam_of(nm),
      regime  = reg,
      gain    = gv,
      g       = gs,
      mid     = pg$partial_rho_mean,
      lo      = pg$partial_rho_min,
      hi      = pg$partial_rho_max,
      c2      = c2_pass,
      nn      = nn_pass,
      sat     = sat_any,
      stringsAsFactors = FALSE
    )
  }
}
dAB <- do.call(rbind, rows)
dAB$panel_label <- ifelse(dAB$regime == "high-gain",
                          "gain $\\geq 8$ cells",
                          "gain $< 8$ cells")

# ---- Panel A: partial rho by g, faceted by regime ----
tikz("figF3A_partial_rho_by_g.tex", width = 5.0, height = 3.3, standAlone = FALSE)
print(
  ggplot(dAB, aes(g, mid, group = cell, colour = regime)) +
    geom_hline(yintercept = 0.15, linewidth = 0.35, colour = "grey40",
               linetype = "22") +
    geom_hline(yintercept = 0,    linewidth = 0.25, colour = "grey75") +
    geom_text(data = data.frame(g = 2.3, mid = 0.22, cell = "lab",
                                 regime = "low-gain",
                                 panel_label = "gain $< 8$ cells"),
              label = "gate $+0.15$", size = 2.65, colour = "grey30",
              show.legend = FALSE) +
    geom_ribbon(aes(ymin = lo, ymax = hi, fill = regime),
                alpha = 0.10, colour = NA) +
    geom_line(linewidth = 0.4, alpha = 0.85) +
    geom_point(aes(shape = sat), size = 1.1) +
    facet_wrap(~panel_label) +
    scale_x_log10(breaks = c(2, 3, 5, 10, 20)) +
    scale_shape_manual(values = c(`FALSE` = 16, `TRUE` = 1),
                       labels = c(`FALSE` = "unsaturated", `TRUE` = "saturated"),
                       name = NULL) +
    scale_colour_manual(values = c(`high-gain` = C_HIGH, `low-gain` = C_LOW),
                        guide = "none") +
    scale_fill_manual(values   = c(`high-gain` = C_HIGH, `low-gain` = C_LOW),
                      guide = "none") +
    labs(x = "merge group size $g$ (log scale)",
         y = "partial $\\rho(I_{\\cos}, \\mathrm{drop} \\mid I_{\\mathrm{mag}})$",
         title = "(A) Geometry gate: partial $\\rho$ by $g$") +
    theme_classic(base_size = 10) +
    theme(legend.position = "bottom",
          legend.text = element_text(size = 8),
          strip.background = element_rect(linewidth = 0.4),
          plot.title = element_text(size = 9, face = "plain"),
          plot.margin = margin(4, 8, 2, 10))
)
dev.off()
emit_tex("figF3A_partial_rho_by_g.tex", c(
  "% SOURCE: submissions/d2-neurocomputing/zenodo-deposit/results/merging/RG_map_evidence_REFIX20260801.json :: cells.*.per_g.{2,3,5,10,20}.partial_rho_{s0,s1,s2,mean}; facets cells.*.regime; gate line partial=0.15"))
cat("wrote figF3A_partial_rho_by_g.tex\n")

# ---- Panel B: c2 coherence bands ----
# Show, per cell per g: whether all/most seeds pass the c2 coherence criterion
# (rho_raw >= 0.30). Display as a heatmap tile: pass = filled, fail = open.
dB <- dAB[, c("cell","family","regime","gain","g","c2","nn","sat","mid")]
dB$c2_label <- ifelse(!dB$nn,  "negligible",
                ifelse( dB$sat, "saturated",
                ifelse( dB$c2,  "c2 pass",   "c2 fail")))
shorten_cell <- function(x)
  tex_safe(sub("_RG$", "",
    sub("Mistral-Nemo-Base-2407", "Mistral-Nemo",
    sub("Qwen2\\.5-", "Q", sub("Llama-3\\.[12]-", "Ll",
    sub("gemma-2-", "G", sub("gpt-neox", "NeoX",
    sub("gpt2-xl", "GPT2",
    sub("Phi-3\\.5-mini", "Phi", x)))))))))

dB$cell_short  <- shorten_cell(dB$cell)
dB$cell_short2 <- shorten_cell(dB$cell)
# order cells by gain descending
ord <- dAB[!duplicated(dAB$cell), c("cell","gain")]
ord <- ord[order(-ord$gain), ]
cell_levels <- shorten_cell(ord$cell)
dB$cell_f <- factor(dB$cell_short2, levels = rev(cell_levels))
dB$g_f    <- factor(dB$g)
c2_colors <- c("c2 pass"   = "#0072B2",
               "c2 fail"   = "#E69F00",
               "negligible"= "#BBBBBB",
               "saturated" = "#CC79A7")
tikz("figF3B_c2_coherence.tex", width = 5.0, height = 4.2, standAlone = FALSE)
print(
  ggplot(dB, aes(g_f, cell_f, fill = c2_label)) +
    geom_tile(colour = "white", linewidth = 0.3) +
    scale_fill_manual(values = c2_colors, name = NULL) +
    labs(x = "merge group size $g$",
         y = NULL,
         title = "(B) c2 coherence bands") +
    theme_classic(base_size = 9) +
    theme(axis.text.y   = element_text(size = 6.5),
          legend.position = "bottom",
          legend.text = element_text(size = 7.5),
          plot.title = element_text(size = 9, face = "plain"),
          plot.margin = margin(4, 8, 2, 6))
)
dev.off()
emit_tex("figF3B_c2_coherence.tex", c(
  "% SOURCE: submissions/d2-neurocomputing/zenodo-deposit/results/merging/RG_map_evidence_REFIX20260801.json :: cells.*.per_g.{2,3,5,10,20}.{c2_coherent,non_negligible,saturated} per seed; cells ordered by gain"))
cat("wrote figF3B_c2_coherence.tex\n")

# ---- Panel C: permutation null — FPR across all 22 cells ----
# SOURCE: perm_null_allcells/*.json
perm_dir <- file.path(DEPOSIT, "perm_null_allcells")
pfiles   <- list.files(perm_dir, pattern = "\\.json$", full.names = TRUE)
rows_p <- list(); i <- 0
for (pf in pfiles) {
  pd  <- fromJSON(pf, simplifyVector = FALSE)
  nm  <- sub("_RG\\.json$", "", basename(pf))
  fpr <- as.numeric(pd$gate_false_positive_rate)   # always 0.0
  np  <- as.integer(pd$n_perm)
  # per-cell min p-value across qualifying cells
  pvals <- sapply(pd$cells, function(cc) {
    v <- cc$perm_p_one_sided
    if (is.null(v)) NA else as.numeric(v)
  })
  pvals_clean <- pvals[!is.na(pvals)]
  gv <- tryCatch({
    gain_nm <- paste0(nm, "_RG")
    gain_data <- fromJSON(file.path(DEPOSIT,
                                     "RG_gain_law_MERGED_REFIX20260730.json"),
                          simplifyVector = FALSE)
    gain_data$bundles[[gain_nm]]$gain_median_absdrop_per_dose
  }, error = function(e) NA)
  i <- i + 1
  rows_p[[i]] <- data.frame(
    cell  = nm,
    fpr   = fpr,
    nperm = np,
    min_p = if (length(pvals_clean)) min(pvals_clean) else NA,
    gain  = if (!is.na(gv)) as.numeric(gv) else NA,
    stringsAsFactors = FALSE
  )
}
dC <- do.call(rbind, rows_p)
dC$regime <- ifelse(is.na(dC$gain), "unknown",
                    ifelse(dC$gain >= GAIN_CUT, "high-gain", "low-gain"))
dC$cell_short <- shorten_cell(dC$cell)
dC <- dC[order(-dC$gain, na.last = TRUE), ]
dC$cell_f <- factor(dC$cell_short, levels = rev(dC$cell_short))
# min_p at floor 1/2001
floor_p <- 1 / 2001
# the one off-scale cell (never silently omitted — rendered as an edge arrow)
gpt2xl_minp <- dC$min_p[dC$cell_short == tex_safe("GPT2_L36")]
gpt2xl_row  <- which(levels(dC$cell_f) == tex_safe("GPT2_L36"))
tikz("figF3C_perm_null.tex", width = 5.0, height = 4.2, standAlone = FALSE)
print(
  ggplot(dC, aes(min_p, cell_f, colour = regime)) +
    geom_vline(xintercept = 0.05, linewidth = 0.35, colour = "grey55",
               linetype = "22") +
    geom_point(data = dC[is.na(dC$min_p) | dC$min_p <= 0.12, ],
               size = 2.0, shape = 16) +   # GPT2_L36 (0.83) is rendered as the edge
                                           # arrow below instead — dropped from the
                                           # points explicitly, never by scale limits
    annotate("text", x = 0.052, y = 1.5, label = "$\\alpha = 0.05$",
             hjust = 0, size = 2.5, colour = "grey40") +
  { if (length(gpt2xl_row) && length(gpt2xl_minp) && !is.na(gpt2xl_minp))
      list(
        annotate("segment", x = 0.118, xend = 0.105, y = gpt2xl_row, yend = gpt2xl_row,
                 arrow = arrow(length = unit(0.10, "cm")), colour = "grey30", linewidth = 0.4),
        annotate("text", x = 0.12, y = gpt2xl_row,
                 label = sprintf("GPT2\\_L36 min $p = %.2f$ (off scale)", gpt2xl_minp),
                 hjust = 1, size = 2.3, colour = "grey30")
      ) } +
    scale_x_continuous(limits = c(0, 0.12),
                       breaks = c(0, floor_p, 0.05, 0.10),
                       labels = c("0", "$1/2001$", "0.05", "0.10")) +
    scale_colour_manual(values = c(`high-gain` = C_HIGH, `low-gain` = C_LOW),
                        guide = "none") +
    labs(x = "min one-sided perm $p$ (qualifying sub-cells)",
         y = NULL,
         title = "(C) Permutation null: gate FPR 0/2{,}000 per cell, 22/22 cells") +
    theme_classic(base_size = 9) +
    theme(axis.text.y   = element_text(size = 6.5),
          plot.title = element_text(size = 9, face = "plain"),
          plot.margin = margin(4, 8, 2, 6))
)
dev.off()
emit_tex("figF3C_perm_null.tex", c(
  "% SOURCE: submissions/d2-neurocomputing/zenodo-deposit/results/merging/perm_null_allcells/*.json :: gate_false_positive_rate (0/2000 all 22), n_perm=2000, cells.*.perm_p_one_sided (min plotted; GPT2_L36 min 0.83 off-scale, arrow-marked); gain for colouring: submissions/d2-neurocomputing/zenodo-deposit/results/merging/RG_gain_law_MERGED_REFIX20260730.json :: bundles.*.gain_median_absdrop_per_dose"))
cat("wrote figF3C_perm_null.tex\n")

# ---- Panel D: two-boundary diagram ----
# Show, per cell, the qualifying window (geometry-valid: g in qualifying_group_sizes)
# vs the largest non-saturated g (damage-gradated boundary).
# SOURCE: RG_operating_curve_table files as collected in map evidence
# We derive directly from per_g in mev: qualifying = c2 pass + partial>=0.15 in >=2 seeds;
# gradated = unsaturated (argmax_loss < 0.8 in >=2 seeds)
rows_d <- list(); i <- 0
for (nm in names(cells_list)) {
  b  <- cells_list[[nm]]
  gv <- b$gain
  reg <- b$regime
  qual_g   <- c()
  grad_g   <- c()
  for (gs in c(2, 3, 5, 10, 20)) {
    pg <- b$per_g[[as.character(gs)]]
    if (is.null(pg)) next
    # gradated: not majority-saturated
    if (sum(unlist(pg$saturated)) < 2) grad_g <- c(grad_g, gs)
    # qualifying: c2 majority + partial mean >= 0.15
    if (sum(unlist(pg$c2_coherent)) >= 2 && pg$partial_rho_mean >= 0.15)
      qual_g <- c(qual_g, gs)
  }
  max_qual <- if (length(qual_g)) max(qual_g) else NA
  max_grad <- if (length(grad_g)) max(grad_g) else NA
  i <- i + 1
  rows_d[[i]] <- data.frame(cell = nm, gain = gv, regime = reg,
                             max_qual = max_qual, max_grad = max_grad,
                             stringsAsFactors = FALSE)
}
dD <- do.call(rbind, rows_d)
dD$cell_short <- shorten_cell(dD$cell)
dD <- dD[order(-dD$gain, na.last = TRUE), ]
dD$cell_f <- factor(dD$cell_short, levels = rev(dD$cell_short))
# honest NA handling (review 2026-08-01): cells with NO qualifying g keep their row —
# no filled dot, an explicit "no qualifying g" cross at the left edge; never the old
# x=0.8 sentinel silently discarded by the axis limit (10 cells vanished that way).
dD$qual_na  <- is.na(dD$max_qual)
dD$max_qual_j <- ifelse(dD$qual_na, NA, as.numeric(dD$max_qual))
dD$max_grad_j <- ifelse(is.na(dD$max_grad), NA, as.numeric(dD$max_grad))
dD_seg   <- dD[!dD$qual_na & !is.na(dD$max_grad_j) & dD$max_grad_j > dD$max_qual_j, ]
dD_qual  <- dD[!dD$qual_na, ]
dD_grad  <- dD[!is.na(dD$max_grad_j), ]
dD_noqual <- dD[dD$qual_na, ]
tikz("figF3D_two_boundaries.tex", width = 5.0, height = 4.2, standAlone = FALSE)
print(
  ggplot(dD) +
    geom_segment(data = dD_seg,
                 aes(y = cell_f, yend = cell_f,
                     x = max_qual_j, xend = max_grad_j, colour = regime),
                 linewidth = 0.6, alpha = 0.5) +
    geom_point(data = dD_qual,
               aes(x = max_qual_j, y = cell_f, colour = regime),
               shape = 16, size = 2.1) +
    geom_point(data = dD_grad,
               aes(x = max_grad_j, y = cell_f, colour = regime),
               shape = 1, size = 2.1, stroke = 0.7) +
    geom_point(data = dD_noqual,
               aes(x = 1.45, y = cell_f), shape = 4, size = 1.8,
               stroke = 0.7, colour = "grey40") +
    geom_vline(xintercept = 5,  linewidth = 0.35, colour = "#009E73",
               linetype = "22") +
    geom_vline(xintercept = 10, linewidth = 0.35, colour = "#F0E442",
               linetype = "22") +
    annotate("text", x = 5.3,  y = 0.8, label = "geometry gate",
             hjust = 0, size = 2.5, colour = "#009E73") +
    annotate("text", x = 10.3, y = 1.8, label = "damage gradated",
             hjust = 0, size = 2.5, colour = "#F0E442") +
    annotate("text", x = 1.55, y = nrow(dD) - 0.5,
             label = sprintf("$\\times$ = no qualifying $g$ (%d cells)", sum(dD$qual_na)),
             hjust = 0, size = 2.3, colour = "grey40") +
    scale_x_continuous(breaks = c(2, 3, 5, 10, 20)) +
    coord_cartesian(xlim = c(1.2, 26)) +
    scale_colour_manual(values = c(`high-gain` = C_HIGH, `low-gain` = C_LOW),
                        guide = "none") +
    labs(x = "group size $g$ ($\\bullet$ = geometry boundary; $\\circ$ = gradated boundary)",
         y = NULL,
         title = "(D) Two boundaries: geometry-valid $g\\leq5$ vs gradated $g=10$") +
    theme_classic(base_size = 9) +
    theme(axis.text.y   = element_text(size = 6.5),
          plot.title = element_text(size = 9, face = "plain"),
          plot.margin = margin(4, 16, 2, 6))
)
dev.off()
emit_tex("figF3D_two_boundaries.tex", c(
  "% SOURCE: submissions/d2-neurocomputing/zenodo-deposit/results/merging/RG_map_evidence_REFIX20260801.json :: cells.*.per_g.{2,3,5,10,20}.{c2_coherent(>=2 seeds)+partial_rho_mean>=0.15 => qualifying g; saturated(<2 seeds) => gradated g; cells with no qualifying g marked with crosses, never dropped"))
cat("wrote figF3D_two_boundaries.tex\n")
cat("F3 done: figF3A_partial_rho_by_g.tex figF3B_c2_coherence.tex figF3C_perm_null.tex figF3D_two_boundaries.tex\n")
