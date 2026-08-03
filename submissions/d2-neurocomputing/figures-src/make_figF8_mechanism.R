# D2 federation paper — F8 (Mechanism, Prop 1) 4-panel figure.
# R/ggplot2 -> tikzDevice (house standard, matches make_figures.R).
#
# Panel A: cross-term alignment, all 19 bundles at g=2 (mean over 3 seeds)
#          SOURCE: RG_crossterm_alignment_ALL_REFIX20260801.json
# Panel B: readout-slope sign = regime (rho_proj_drop sign vs regime, g=2 per bundle)
#          SOURCE: RG_crossterm_alignment_ALL_REFIX20260801.json
# Panel C: gain ≈ slope magnitude as rank-level proxy
#          (Spearman of gain vs median |drop|/dose, already in gain law table)
#          SOURCE: RG_gain_law_MERGED_REFIX20260730.json + RG_map_evidence_REFIX20260801.json
# Panel D: A4'-free statement panel — proposition scope illustration
#          Text panel: first-order identity; assumptions named; "no A4' needed" annotation
#
# Palette: Okabe-Ito blue #0072B2 (high-gain) + vermillion #D55E00 (low-gain); CVD-safe.
# Run from this directory:  Rscript make_figF8_mechanism.R

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
# timestamp line and prepend the % SOURCE provenance header.
emit_tex <- function(tex, prov) {
  body <- readLines(tex, warn = FALSE)
  body <- body[!grepl("^% Created by tikzDevice", body)]
  writeLines(c(prov, body), tex)
  cat(sprintf("[ok] %s (%d source lines)\n", basename(tex), length(prov)))
}

# Escape underscores for LaTeX axis labels
tex_safe <- function(s) gsub("_", "\\\\_", s)

shorten_bundle <- function(x)
  tex_safe(sub("_L[0-9]+$", "",
    sub("Mistral-Nemo-Base-2407", "Mistral-Nemo",
    sub("Qwen2\\.5-", "Q", sub("Llama-3\\.[12]-", "Ll",
    sub("gemma-2-", "G", sub("gpt-neox-20b", "NeoX",
    sub("gpt2-xl", "GPT2",
    sub("Phi-3\\.5-mini", "Phi", x)))))))))

shorten_bundle_with_layer <- function(x)
  tex_safe(sub("Mistral-Nemo-Base-2407", "Mistral-Nemo",
    sub("Qwen2\\.5-", "Q", sub("Llama-3\\.[12]-", "Ll",
    sub("gemma-2-", "G", sub("gpt-neox-20b", "NeoX",
    sub("gpt2-xl", "GPT2",
    sub("Phi-3\\.5-mini", "Phi", x))))))))

fam_of <- function(n) {
  if (grepl("^Llama-3\\.2-1B|^Llama-3.2-1B", n))  "Llama"
  else if (grepl("^Llama-3\\.2-3B|^Llama-3.2-3B", n)) "Llama"
  else if (grepl("^Llama-3\\.1-8B|^Llama-3.1-8B", n)) "Llama"
  else if (grepl("^Mistral-7B",  n))  "Mistral"
  else if (grepl("^Mistral-Nemo", n)) "Mistral"
  else if (grepl("^Qwen",  n))  "Qwen"
  else if (grepl("^gemma", n))  "Gemma"
  else if (grepl("^Phi",   n))  "Phi"
  else if (grepl("neox",   n))  "GPT-NeoX"
  else "GPT-2"
}

shape_of <- c(Llama = 16, Mistral = 17, Qwen = 15,
              Gemma = 3,  Phi = 8, `GPT-2` = 4, `GPT-NeoX` = 7)

# --------------- load gain law ---------------
gain_raw <- fromJSON(file.path(DEPOSIT, "RG_gain_law_MERGED_REFIX20260730.json"),
                     simplifyVector = FALSE)

# --------------- load crossterm alignment ---------------
al_raw <- fromJSON(file.path(DEPOSIT, "RG_crossterm_alignment_ALL_REFIX20260801.json"),
                   simplifyVector = FALSE)

# --------------- load map evidence (for Panel C g-resolved) ---------------
mev <- fromJSON(file.path(DEPOSIT, "RG_map_evidence_REFIX20260801.json"),
                simplifyVector = FALSE)

# ===== Panel A: cross-term alignment at g=2, per bundle (mean over seeds) =====
# For each bundle: mean_cos_align (averaged over 3 seeds), frac_pos (mean over seeds),
# gain, regime.
rows <- list(); i <- 0
for (nm in names(al_raw$bundles)) {
  b  <- al_raw$bundles[[nm]]
  grg <- paste0(nm, "_RG")
  gb  <- gain_raw$bundles[[grg]]
  gv  <- if (!is.null(gb)) gb$gain_median_absdrop_per_dose else NA
  reg <- if (!is.na(gv) && gv >= GAIN_CUT) "high-gain" else "low-gain"
  seeds <- b$seeds
  vals_align <- c(); vals_frac <- c(); vals_rho <- c()
  for (sd in seeds) {
    cell <- b$cells[[paste0("g2_s", sd)]]
    if (!is.null(cell)) {
      vals_align <- c(vals_align, cell$mean_cos_align)
      vals_frac  <- c(vals_frac,  cell$frac_cos_align_pos)
      vals_rho   <- c(vals_rho,   cell$rho_proj_drop)
    }
  }
  if (!length(vals_align)) next
  i <- i + 1
  rows[[i]] <- data.frame(
    bundle    = nm,
    family    = fam_of(nm),
    gain      = as.numeric(gv),
    regime    = reg,
    cos_align = mean(vals_align),
    frac_pos  = mean(vals_frac),
    rho_proj  = mean(vals_rho),
    stringsAsFactors = FALSE
  )
}
dA <- do.call(rbind, rows)
dA <- dA[order(dA$gain), ]
dA$bundle_lab_layer <- shorten_bundle_with_layer(dA$bundle)
dA$bundle_f <- factor(dA$bundle_lab_layer, levels = dA$bundle_lab_layer)

tikz("figF8A_crossterm_alignment.tex", width = 5.2, height = 3.6, standAlone = FALSE)
# Compute median frac_pos across bundles (mean over 3 seeds per bundle) at g=2
med_frac_g2 <- median(tapply(dA$frac_pos, dA$bundle, mean))
print(
  ggplot(dA, aes(bundle_f, frac_pos, fill = regime)) +
    geom_col(width = 0.7) +
    geom_hline(yintercept = med_frac_g2, linewidth = 0.3, colour = "grey40",
               linetype = "22") +
    annotate("text", x = 0.6, y = med_frac_g2 + 0.03, hjust = 0, size = 2.5,
             colour = "grey30",
             label = sprintf("median $= %.2f$", med_frac_g2)) +
    scale_fill_manual(values = c(`high-gain` = C_HIGH, `low-gain` = C_LOW),
                      name = NULL) +
    scale_y_continuous(limits = c(0, 1.08),
                       breaks = c(0, 0.25, 0.5, 0.75, 1.0)) +
    labs(x = NULL,
         y = "frac($\\cos(d_a, r_a)>0$) at $g{=}2$",
         title = "(A) Cross-term alignment: all 19 bundles, $g{=}2$") +
    theme_classic(base_size = 9) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 6),
          legend.position = "top",
          legend.text = element_text(size = 8),
          plot.title = element_text(size = 9, face = "plain"),
          plot.margin = margin(4, 12, 2, 6))
)
dev.off()
emit_tex("figF8A_crossterm_alignment.tex", c(
  "% SOURCE: submissions/d2-neurocomputing/zenodo-deposit/results/merging/RG_crossterm_alignment_ALL_REFIX20260801.json :: bundles.*.cells.g2_s{0,1,2}.{frac_cos_align_pos,rho_proj_drop} (3-seed means per bundle, 19 bundles)"))
cat("wrote figF8A_crossterm_alignment.tex\n")

# ===== Panel B: readout-slope sign = regime =====
# rho(proj, drop) sign: negative in low-gain (constructive) cells,
# positive in high-gain (destructive) cells.  Show per-bundle at g=2.
tikz("figF8B_slope_sign_regime.tex", width = 5.2, height = 3.6, standAlone = FALSE)
print(
  ggplot(dA, aes(bundle_f, rho_proj, fill = regime)) +
    geom_col(width = 0.7) +
    geom_hline(yintercept = 0, linewidth = 0.4, colour = "grey40") +
    scale_fill_manual(values = c(`high-gain` = C_HIGH, `low-gain` = C_LOW),
                      name = NULL) +
    labs(x = NULL,
         y = "$\\rho(\\mathrm{proj},\\ \\mathrm{drop})$ at $g{=}2$",
         title = "(B) Readout-slope sign $=$ regime") +
    theme_classic(base_size = 9) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 6),
          legend.position = "top",
          legend.text = element_text(size = 8),
          plot.title = element_text(size = 9, face = "plain"),
          plot.margin = margin(4, 12, 2, 6))
)
dev.off()
emit_tex("figF8B_slope_sign_regime.tex", c(
  "% SOURCE: submissions/d2-neurocomputing/zenodo-deposit/results/merging/RG_crossterm_alignment_ALL_REFIX20260801.json :: bundles.*.cells.g2_s{0,1,2}.rho_proj_drop (sign split by regime from submissions/d2-neurocomputing/zenodo-deposit/results/merging/RG_gain_law_MERGED_REFIX20260730.json :: bundles.*.gain_median_absdrop_per_dose vs cut 8)"))
cat("wrote figF8B_slope_sign_regime.tex\n")

# ===== Panel C: gain ≈ slope magnitude as rank-level proxy =====
# Show gain (median |drop|/dose) vs per-g=2 median |drop|/dose from map evidence,
# and scatter over all 22 bundles to illustrate the rank relationship.
rows_c <- list(); ic <- 0
for (nm in names(mev$cells)) {
  b  <- mev$cells[[nm]]
  pg2 <- b$per_g[["2"]]
  if (is.null(pg2)) next
  ic <- ic + 1
  rows_c[[ic]] <- data.frame(
    cell   = nm,
    family = fam_of(nm),
    regime = b$regime,
    gain   = b$gain,
    med_g2 = pg2$median_abs_drop_med3,
    stringsAsFactors = FALSE
  )
}
dC <- do.call(rbind, rows_c)
# Spearman between gain and med_g2
sp_c <- cor(dC$gain, dC$med_g2, method = "spearman")
sp_p <- cor.test(dC$gain, dC$med_g2, method = "spearman", exact = FALSE)$p.value

tikz("figF8C_gain_slope_proxy.tex", width = 4.4, height = 3.4, standAlone = FALSE)
print(
  ggplot(dC, aes(gain, med_g2, colour = regime, shape = family)) +
    geom_smooth(method = "lm", se = FALSE, linewidth = 0.4,
                colour = "grey60", formula = y ~ x) +
    geom_point(size = 2.1, stroke = 0.7) +
    annotate("text", x = 5, y = max(dC$med_g2) * 0.92,
             label = sprintf("Spearman $\\rho = %.2f$, $p = %.1e$", sp_c, sp_p),
             hjust = 0, size = 2.8, colour = "grey30") +
    scale_x_log10(breaks = c(0.1, 1, 3, 10, 30, 70),
                  limits = c(0.07, 80)) +
    scale_y_log10(breaks = c(0.003, 0.03, 0.3, 3, 15),
                  labels = c("0.003", "0.03", "0.3", "3", "15")) +
    scale_colour_manual(values = c(`high-gain` = C_HIGH, `low-gain` = C_LOW),
                        name = NULL) +
    scale_shape_manual(values = shape_of, name = NULL) +
    labs(x = "perturbation gain (median $|$drop$|$/dose, pooled $g\\leq20$)",
         y = "median $|$drop$|$ at $g{=}2$ (logits)",
         title = "(C) Gain $\\approx$ slope magnitude: rank-level proxy") +
    theme_classic(base_size = 9) +
    theme(legend.position = "right",
          legend.key.height = grid::unit(9, "pt"),
          legend.text = element_text(size = 7),
          plot.title = element_text(size = 9, face = "plain"),
          plot.margin = margin(4, 8, 2, 10))
)
dev.off()
emit_tex("figF8C_gain_slope_proxy.tex", c(
  "% SOURCE: submissions/d2-neurocomputing/zenodo-deposit/results/merging/RG_map_evidence_REFIX20260801.json :: cells.*.{gain, per_g.\"2\".median_abs_drop_med3}; Spearman computed in-script (rank-level proxy; rho=0.81, p=4e-6)"))
cat("wrote figF8C_gain_slope_proxy.tex\n")

# ===== Panel D: A4'-free statement panel =====
# Pure text/annotation panel: Proposition scope illustration.
# Shows the first-order identity and marks which assumptions are needed,
# explicitly noting "no A4' (true-backprop) required".
dD_dummy <- data.frame(x = 0:1, y = 0:1)
lines_D <- list(
  list(x=0.05, y=0.93, label="\\textbf{Proposition} (first-order readout identity)",        sz=2.9, col="black"),
  list(x=0.05, y=0.84, label="$\\mathrm{drop}_a = -\\ell_a'(1)\\,\\delta_a + o(|\\delta_a|)$", sz=3.2, col="black"),
  list(x=0.05, y=0.73, label="\\textit{Assumptions used:}",                                  sz=2.6, col="grey30"),
  list(x=0.08, y=0.65, label="A1. Scalar readout along delivery axis ($w_a$ is $o(|\\delta_a|)$)",sz=2.5, col="grey20"),
  list(x=0.08, y=0.57, label="A2. Local smoothness of $\\ell_a$ near $t=1$",                sz=2.5, col="grey20"),
  list(x=0.05, y=0.46, label="\\textit{Not required:}",                                      sz=2.6, col="grey30"),
  list(x=0.08, y=0.38, label="$\\times$ A4$'$ (true backprop / GradSim surrogate)",         sz=2.5, col="#D55E00"),
  list(x=0.08, y=0.30, label="$\\times$ any specific architecture or depth",                 sz=2.5, col="#D55E00"),
  list(x=0.05, y=0.18, label="\\textit{Consequence:} gain = rank proxy for $|\\ell_a'(1)|$;",sz=2.5, col="black"),
  list(x=0.05, y=0.10, label="\\phantom{Consequence:} constr.\\ fraction = proxy for $\\Pr[\\ell_a'(1)>0]$",sz=2.5,col="black")
)
tikz("figF8D_prop1_scope.tex", width = 4.8, height = 3.4, standAlone = FALSE)
p_D <- ggplot(dD_dummy, aes(x, y)) +
  geom_blank() +
  geom_rect(data = data.frame(dummy = 1),
            aes(xmin=0.02, xmax=0.98, ymin=0.79, ymax=0.99),
            inherit.aes=FALSE, fill="#EEF4FB", colour="#0072B2",
            linewidth=0.4) +
  geom_rect(data = data.frame(dummy = 1),
            aes(xmin=0.02, xmax=0.98, ymin=0.42, ymax=0.70),
            inherit.aes=FALSE, fill="#F9F9F9", colour="grey70",
            linewidth=0.3) +
  geom_rect(data = data.frame(dummy = 1),
            aes(xmin=0.02, xmax=0.98, ymin=0.23, ymax=0.41),
            inherit.aes=FALSE, fill="#FFF2EC", colour="#D55E00",
            linewidth=0.3)
for (ll in lines_D) {
  p_D <- p_D + annotate("text", x=ll$x, y=ll$y, label=ll$label,
                         hjust=0, vjust=0.5, size=ll$sz, colour=ll$col)
}
p_D <- p_D +
  labs(title = "(D) Prop.\\ 1 scope: A4$'$-free statement") +
  theme_void(base_size = 9) +
  theme(plot.title = element_text(size=9, face="plain",
                                   margin=margin(b=4)),
        plot.margin = margin(6, 8, 4, 6))
print(p_D)
dev.off()
emit_tex("figF8D_prop1_scope.tex", c(
  "% SOURCE: proposition scope statement (text panel; no plotted data; Prop. 1 first-order identity, A1/A2 named, A4' not required)"))
cat("wrote figF8D_prop1_scope.tex\n")
cat("F8 done: figF8A_crossterm_alignment.tex figF8B_slope_sign_regime.tex figF8C_gain_slope_proxy.tex figF8D_prop1_scope.tex\n")
