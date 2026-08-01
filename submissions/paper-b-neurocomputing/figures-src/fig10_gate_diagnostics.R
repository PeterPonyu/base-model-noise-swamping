#!/usr/bin/env Rscript
# SOURCE: edit-harness/results/quant_survival/aggregate/gate_readout.json
# SOURCE FIELDS: thresholds.*; gates.K{1,2,3}_*; cells.*.arms.*; cells.*.c3.nf4dq.F_above_mean
# SOURCE: docs/plans/PREREG-PAPERB-QUANTSURVIVAL-DRAFT-2026-07-16.md sections C1--C3 and K1--K3

suppressPackageStartupMessages({
  library(jsonlite)
  library(ggplot2)
  library(patchwork)
  library(tikzDevice)
})

args <- commandArgs(trailingOnly = TRUE)
has_flag <- function(flag) flag %in% args
arg_value <- function(flag, default) {
  hit <- match(flag, args)
  if (is.na(hit)) return(default)
  if (hit == length(args)) stop(flag, " requires a value", call. = FALSE)
  args[[hit + 1L]]
}

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (length(script_arg) != 1L) stop("Run with Rscript so the repository root can be resolved", call. = FALSE)
script_path <- normalizePath(sub("^--file=", "", script_arg), mustWork = TRUE)
repo_root <- normalizePath(file.path(dirname(script_path), "..", "..", ".."), mustWork = TRUE)
default_input <- file.path(repo_root, "edit-harness", "results", "quant_survival", "aggregate", "gate_readout.json")
default_prereg <- file.path(repo_root, "docs", "plans", "PREREG-PAPERB-QUANTSURVIVAL-DRAFT-2026-07-16.md")
default_output <- file.path(dirname(dirname(script_path)), "fig10_gate_diagnostics.tex")
input_path <- normalizePath(arg_value("--input", default_input), mustWork = TRUE)
prereg_path <- normalizePath(arg_value("--prereg", default_prereg), mustWork = TRUE)
output_path <- arg_value("--output", default_output)
dry_run <- has_flag("--dry")

x <- fromJSON(input_path, simplifyVector = FALSE)
required_cells <- c("Llama-3.2-1B_rome_L12", "Llama-3.2-3B_rome_L24", "Qwen2.5-1.5B_rome_L21")
if (!all(required_cells %in% names(x$cells))) stop("schema: missing required ROME gate cells", call. = FALSE)
if (!identical(x$gates$K1_geometry_ranking_survival$status, "FAIL")) stop("contract: canonical K1 raw gate status changed", call. = FALSE)
if (!identical(x$gates$K2_esr_survival_4bit$status, "PASS")) stop("contract: canonical K2 status changed", call. = FALSE)
if (length(x$gates$K3_M_concentration$legacy_sensitivity_observations) != 3L) stop("contract: expected three K3 sensitivity observations", call. = FALSE)

rank_threshold <- as.numeric(x$thresholds$rank_survival_4bit)
esr_threshold <- as.numeric(x$thresholds$esr_survival_4bit)
if (!isTRUE(all.equal(rank_threshold, 0.85))) stop("contract: K1 4-bit rank threshold changed", call. = FALSE)
if (!isTRUE(all.equal(esr_threshold, 0.80))) stop("contract: K2 4-bit ESR threshold changed", call. = FALSE)

rank_cells <- required_cells[1:2]
k1 <- data.frame(
  label = c("Llama-1B\nL12", "Llama-3B\nL24"),
  value = vapply(rank_cells, function(n) x$cells[[n]]$arms$nf4dq_full_model$rho_damage_fp32_vs_arm_rank_survival_mean, numeric(1)),
  stringsAsFactors = FALSE
)
k1$status <- ifelse(k1$value >= rank_threshold, "PASS", "FAIL")
if (!identical(k1$status, c("PASS", "FAIL"))) stop("contract: expected K1 one-pass/one-fail pattern", call. = FALSE)

k2_cells <- unlist(x$gates$K2_esr_survival_4bit$cells_evaluated, use.names = FALSE)
if (!setequal(k2_cells, c(required_cells[1], required_cells[3]))) stop("contract: K2 evaluated-cell set changed", call. = FALSE)
k2 <- data.frame(
  label = c("Llama-1B\nL12", "Qwen-1.5B\nL21"),
  value = vapply(c(required_cells[1], required_cells[3]), function(n) {
    min(x$cells[[n]]$arms$nf4dq_edited_layer$esr_survival_given_fp32_worked_mean,
        x$cells[[n]]$arms$nf4dq_full_model$esr_survival_given_fp32_worked_mean)
  }, numeric(1)),
  stringsAsFactors = FALSE
)
if (any(k2$value < esr_threshold)) stop("contract: K2 no longer passes", call. = FALSE)

k3 <- data.frame(
  label = c("Llama-1B\nL12", "Llama-3B\nL24"),
  value = vapply(rank_cells, function(n) x$cells[[n]]$c3$nf4dq$F_above_mean, numeric(1)),
  stringsAsFactors = FALSE
)
if (min(k3$value) < 0.001 || max(k3$value) > 0.004) stop("contract: K3 validated-cell F_above range changed", call. = FALSE)

prereg <- readLines(prereg_path, warn = FALSE)
required_prereg <- c("K1:", "K2:", "K3:", "M-averaging")
if (!all(vapply(required_prereg, function(s) any(grepl(s, prereg, fixed = TRUE)), logical(1)))) {
  stop("contract: preregistration no longer contains K1--K3 and M-averaging clauses", call. = FALSE)
}

cat(sprintf("K1: %.4f PASS vs %.4f FAIL at %.2f\n", k1$value[1], k1$value[2], rank_threshold))
cat(sprintf("K2: minima %.4f and %.4f; PASS at %.2f\n", k2$value[1], k2$value[2], esr_threshold))
cat(sprintf("K3 measured axis: F_above %.4f--%.4f; FAIL\n", min(k3$value), max(k3$value)))
cat("disclosure: failed preregistered gate plus unregistered explanatory mechanism\n")
if (dry_run) quit(status = 0L)

C_BLUE <- "#0072B2"
C_ORANGE <- "#D55E00"
C_TEAL <- "#009E73"
C_PURPLE <- "#6A3D9A"
status_colors <- c("PASS" = C_TEAL, "FAIL" = C_ORANGE)

paper_theme <- theme_classic(base_size = 10.5, base_family = "sans") +
  theme(
    plot.title = element_text(face = "bold", size = 10.5),
    plot.subtitle = element_text(size = 8.1),
    axis.title = element_text(size = 9),
    axis.text = element_text(size = 8),
    legend.position = "none",
    plot.margin = margin(5, 6, 6, 6)
  )

p_a <- ggplot(k1, aes(label, value, fill = status)) +
  geom_hline(yintercept = rank_threshold, linetype = "22", colour = "grey45", linewidth = 0.4) +
  geom_col(width = 0.58) +
  geom_text(aes(label = sprintf("%.3f %s", value, status)), vjust = -0.45, size = 2.65) +
  annotate("text", x = 2.38, y = rank_threshold, label = "0.85", hjust = 0, vjust = -0.35, size = 2.4, colour = "grey35") +
  scale_fill_manual(values = status_colors) +
  scale_y_continuous(limits = c(0, 1.02), breaks = c(0, 0.5, 0.85, 1), expand = c(0, 0)) +
  labs(title = "K1: narrow result", subtitle = "NF4 full-model rank survival", x = NULL, y = "rank survival") +
  paper_theme

k2$fill <- "PASS"
p_b <- ggplot(k2, aes(label, value, fill = fill)) +
  geom_hline(yintercept = esr_threshold, linetype = "22", colour = "grey45", linewidth = 0.4) +
  geom_col(width = 0.58) +
  geom_text(aes(label = sprintf("%.3f", value)), vjust = -0.45, size = 2.65) +
  annotate("text", x = 2.38, y = esr_threshold, label = "0.80", hjust = 0, vjust = -0.35, size = 2.4, colour = "grey35") +
  scale_fill_manual(values = c("PASS" = C_TEAL)) +
  scale_y_continuous(limits = c(0, 1.035), breaks = c(0, 0.5, 0.8, 1), expand = c(0, 0)) +
  labs(title = "K2: PASS", subtitle = "Worst NF4 conditional ESR by gate cell", x = NULL, y = "conditional ESR") +
  paper_theme

p_c <- ggplot(k3, aes(label, value)) +
  geom_col(width = 0.58, fill = C_ORANGE) +
  geom_text(aes(label = sprintf("%.4f", value)), vjust = -0.45, size = 2.65) +
  scale_y_continuous(limits = c(0, 0.0042), breaks = c(0, 0.002, 0.004), expand = c(0, 0)) +
  labs(title = "K3: FAIL on measured axis", subtitle = expression(F[above] == Pr(abs(Delta*W) > b)), x = NULL, y = expression(F[above])) +
  paper_theme

flow <- data.frame(
  x = c(1, 2, 3), y = c(1, 1, 1),
  label = c("registered\nrank gate", "failed", "exploratory\nmechanism")
)
p_d <- ggplot(flow, aes(x, y)) +
  annotate("segment", x = 1.28, xend = 1.72, y = 1, yend = 1, arrow = arrow(length = unit(0.12, "cm")), linewidth = 0.55, colour = "grey40") +
  annotate("segment", x = 2.28, xend = 2.72, y = 1, yend = 1, arrow = arrow(length = unit(0.12, "cm")), linewidth = 0.55, colour = "grey40") +
  geom_point(shape = 21, size = 9, stroke = 0.7, fill = c("white", "#FDE8DF", "white"), colour = c(C_BLUE, C_ORANGE, C_PURPLE)) +
  geom_text(aes(label = label), size = 2.65, fontface = c("plain", "bold", "plain")) +
  annotate("text", x = 2, y = 0.52, label = "failed gate + unregistered hypothesis", size = 2.45) +
  coord_cartesian(xlim = c(0.45, 3.55), ylim = c(0.3, 1.55), clip = "off") +
  labs(title = "Preregistration divergence", subtitle = "The explanatory law is post hoc") +
  theme_void(base_size = 10.5, base_family = "sans") +
  theme(plot.title = element_text(face = "bold", size = 10.5), plot.subtitle = element_text(size = 8.1), plot.margin = margin(5, 6, 6, 6))

plot <- (p_a + p_b) / (p_c + p_d) +
  plot_annotation(
    title = "Gate diagnostics",
    caption = "A: K1 narrow result. B: K2 gate cells. C: K3 measured axis. D: registered failure versus exploratory explanation.",
    tag_levels = "A",
    theme = theme(
      plot.tag = element_text(face = "bold", size = 10),
      plot.title = element_text(face = "bold", size = 11),
      plot.caption = element_text(size = 7.2, hjust = 0),
      plot.caption.position = "plot"
    )
  )

dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
tikz(output_path, width = 7.35, height = 6.0, standAlone = FALSE)
print(plot)
dev.off()
body <- readLines(output_path, warn = FALSE)
body <- body[!grepl("^% Created by tikzDevice", body)]
writeLines(c(
  "% SOURCE: edit-harness/results/quant_survival/aggregate/gate_readout.json",
  "% SOURCE FIELDS F10-A: thresholds.rank_survival_4bit; cells.{Llama-3.2-1B_rome_L12,Llama-3.2-3B_rome_L24}.arms.nf4dq_full_model.rho_damage_fp32_vs_arm_rank_survival_mean; gates.K1_geometry_ranking_survival.*.",
  "% SOURCE FIELDS F10-B: thresholds.esr_survival_4bit; gates.K2_esr_survival_4bit.*; cells.{Llama-3.2-1B_rome_L12,Qwen2.5-1.5B_rome_L21}.arms.{nf4dq_edited_layer,nf4dq_full_model}.esr_survival_given_fp32_worked_mean.",
  "% SOURCE FIELDS F10-C: cells.{Llama-3.2-1B_rome_L12,Llama-3.2-3B_rome_L24}.c3.nf4dq.F_above_mean; gates.K3_M_concentration.legacy_sensitivity_observations.",
  "% SOURCE F10-D: docs/plans/PREREG-PAPERB-QUANTSURVIVAL-DRAFT-2026-07-16.md sections C1--C3 and K1--K3; disclosure wording matches manuscript limitations but main.tex is not an input.",
  body
), output_path, useBytes = TRUE)
cat("wrote", normalizePath(output_path, mustWork = TRUE), "\n")
