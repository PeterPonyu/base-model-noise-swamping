#!/usr/bin/env Rscript
# SOURCE: edit-harness/results/quant_survival/aggregate/quant_survival_repair_v1.json
# SOURCE FIELDS: cells[*].{slug,editor,layer,absolute_fp32_esr}; cells[*].arms[*].{absolute_quantized_esr.point,conditional_survival_given_fp32_worked.point}
# SOURCE: edit-harness/results/quant_survival/{llama1b_rome_L12_s*,llama3b_rome_L24_s*}/QS_phase1_table.json
# SOURCE FIELDS: arms.nf4dq_full_model.rho_damage_fp32_vs_arm_rank_survival

suppressPackageStartupMessages({
  library(jsonlite)
  library(ggplot2)
  library(patchwork)
  library(tikzDevice)
})

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
stopifnot(length(script_arg) == 1L)
script_path <- normalizePath(sub("^--file=", "", script_arg), mustWork = TRUE)
repo_root <- normalizePath(file.path(dirname(script_path), "..", "..", ".."), mustWork = TRUE)
paper_dir <- dirname(dirname(script_path))
harness <- file.path(repo_root, "edit-harness")
aggregate_path <- file.path(harness, "results", "quant_survival", "aggregate", "quant_survival_repair_v1.json")
output_path <- file.path(paper_dir, "fig02_efficacy_survival.tex")
report_path <- file.path(paper_dir, "fig02_recompute_report.txt")

finite_scalar <- function(x, label) {
  if (length(x) != 1L || !is.numeric(x) || !is.finite(x)) stop(label, " must be finite", call. = FALSE)
  as.numeric(x)
}

s <- fromJSON(aggregate_path, simplifyVector = FALSE)
stopifnot(s$module_provenance$version == "1.2.1", s$module_provenance$n_boot == 500)

rows <- list(); i <- 0L
for (cell in s$cells) {
  for (arm in names(cell$arms)) {
    x <- cell$arms[[arm]]; i <- i + 1L
    rows[[i]] <- data.frame(
      model = cell$slug,
      editor = toupper(cell$editor),
      layer = as.integer(cell$layer),
      arm = arm,
      fp32 = finite_scalar(cell$absolute_fp32_esr$point, "absolute_fp32_esr.point"),
      absolute = finite_scalar(x$absolute_quantized_esr$point, "absolute_quantized_esr.point"),
      conditional = finite_scalar(x$conditional_survival_given_fp32_worked$point, "conditional_survival_given_fp32_worked.point"),
      stringsAsFactors = FALSE
    )
  }
}
d <- do.call(rbind, rows)
stopifnot(nrow(d) == 36L)
model_labels <- c(llama1b = "Llama-1B L12", qwen15b = "Qwen-1.5B L21", llama3b = "Llama-3B L24")
editor_labels <- c(ALPHA = "AlphaEdit", MEMIT = "MEMIT", ROME = "ROME")
editor_colors <- c(ALPHA = "#D55E00", MEMIT = "#009E73", ROME = "#0072B2")
d$model_label <- factor(unname(model_labels[d$model]), levels = unname(model_labels))
d$editor_label <- factor(unname(editor_labels[d$editor]), levels = unname(editor_labels))

paper_theme <- theme_classic(base_size = 10.2, base_family = "sans") + theme(
  plot.title = element_text(face = "bold", size = 10.2),
  plot.subtitle = element_text(size = 8.0),
  axis.title = element_text(size = 8.8), axis.text = element_text(size = 7.8),
  strip.text = element_text(face = "bold", size = 8.2),
  legend.position = "bottom", legend.text = element_text(size = 7.8),
  plot.caption = element_text(size = 7.0, hjust = 0), plot.caption.position = "plot",
  plot.margin = margin(5, 6, 6, 6)
)

p_a_data <- aggregate(absolute ~ model_label + editor_label, d, mean)
p_a <- ggplot(p_a_data, aes(model_label, absolute, fill = editor_label)) +
  geom_col(position = position_dodge(.72), width = .64) +
  scale_fill_manual(values = setNames(editor_colors, editor_labels)) +
  scale_y_continuous(limits = c(0, 1.04), breaks = c(0, .5, 1), expand = c(0, .01)) +
  labs(title = "Absolute efficacy survival", subtitle = "Mean over four quantization arms", x = NULL, y = "absolute ESR", fill = "editor") +
  paper_theme + theme(axis.text.x = element_text(angle = 25, hjust = 1), legend.position = "none")

p_b_data <- aggregate(conditional ~ model_label + editor_label, d, mean)
p_b <- ggplot(p_b_data, aes(model_label, conditional, fill = editor_label)) +
  geom_col(position = position_dodge(.72), width = .64) +
  scale_fill_manual(values = setNames(editor_colors, editor_labels)) +
  scale_y_continuous(limits = c(0, 1.04), breaks = c(0, .5, 1), expand = c(0, .01)) +
  labs(title = "Conditional on FP32 success", subtitle = "Mean survival among successful FP32 edits", x = NULL, y = "conditional ESR", fill = "editor") +
  paper_theme + theme(axis.text.x = element_text(angle = 25, hjust = 1), legend.position = "none")

p_c_data <- d[d$arm %in% c("int8_edited_layer", "int8_full_model"), ]
p_c_data$locality <- factor(ifelse(p_c_data$arm == "int8_edited_layer", "edited layer", "full model"), levels = c("edited layer", "full model"))
p_c <- ggplot(p_c_data, aes(model_label, conditional, colour = editor_label, shape = locality)) +
  geom_hline(yintercept = .9, colour = "#777777", linetype = "22", linewidth = .35) +
  geom_point(position = position_dodge(.55), size = 2.0) +
  scale_colour_manual(values = setNames(editor_colors, editor_labels)) +
  scale_shape_manual(values = c("edited layer" = 16, "full model" = 17)) +
  scale_y_continuous(limits = c(.75, 1.015), breaks = c(.8, .9, 1), expand = expansion(mult = c(.02, .03))) +
  labs(title = "8-bit survival", subtitle = "Dashed: 0.90 efficacy-survival gate", x = NULL, y = "conditional ESR", colour = "editor", shape = "scope") +
  paper_theme + theme(axis.text.x = element_text(angle = 25, hjust = 1), legend.position = "bottom")

validated <- list(llama1b = list(layer = 12L), llama3b = list(layer = 24L))
k1_rows <- list(); j <- 0L
for (slug in names(validated)) {
  per_seed <- numeric()
  for (seed in 0:2) {
    raw_path <- file.path(harness, "results", "quant_survival", sprintf("%s_rome_L%d_s%d", slug, validated[[slug]]$layer, seed), "QS_phase1_table.json")
    raw <- fromJSON(raw_path, simplifyVector = FALSE)
    per_seed <- c(per_seed, finite_scalar(raw$arms$nf4dq_full_model$rho_damage_fp32_vs_arm_rank_survival, basename(raw_path)))
  }
  recomputed <- mean(per_seed)
  agg_cell <- Filter(function(x) x$slug == slug && x$editor == "rome", s$cells)[[1L]]
  canonical <- finite_scalar(agg_cell$arms$nf4dq_full_model$flat_rank$point, paste0(slug, " canonical flat rank"))
  if (abs(recomputed - canonical) > 5e-5) stop(slug, " raw recompute != canonical: ", recomputed, " vs ", canonical, call. = FALSE)
  j <- j + 1L
  k1_rows[[j]] <- data.frame(
    model = unname(model_labels[slug]), value = recomputed,
    lo = min(per_seed), hi = max(per_seed),
    verdict = ifelse(recomputed >= .85, "PASS", "FAIL"),
    per_seed = paste(sprintf("%.4f", per_seed), collapse = "/"), stringsAsFactors = FALSE
  )
}
k1 <- do.call(rbind, k1_rows)
k1$model <- factor(k1$model, levels = unname(model_labels[c("llama1b", "llama3b")]))
p_d <- ggplot(k1, aes(model, value)) +
  geom_hline(yintercept = .85, colour = "#777777", linetype = "22", linewidth = .35) +
  geom_errorbar(aes(ymin = lo, ymax = hi), width = .12, linewidth = .45, colour = editor_colors[["ROME"]]) +
  geom_point(size = 2.7, colour = editor_colors[["ROME"]]) +
  geom_text(aes(label = sprintf("%.3f %s", value, verdict)), vjust = -1.0, size = 2.7, colour = editor_colors[["ROME"]]) +
  scale_y_continuous(limits = c(.62, .96), breaks = c(.68, .85, .90), expand = expansion(mult = c(.02, .08))) +
  labs(title = "4-bit full-model K1", subtitle = "ROME only; labels report threshold verdicts", x = NULL, y = "Spearman rank survival") +
  paper_theme + theme(axis.text.x = element_text(angle = 25, hjust = 1), legend.position = "none")

plot <- (p_a | p_b) / (p_c | p_d) +
  plot_annotation(
    title = "Efficacy and survival under quantization",
    subtitle = "Model labels include the single evaluated layer; no layer curve is implied.",
    caption = "Panels A-C: canonical v1.2.1 (hierarchical bootstrap n=500). Panel D: raw three-seed ROME recomputation; labels report K1 threshold verdicts.",
    tag_levels = "A",
    theme = theme(plot.tag = element_text(face = "bold", size = 10), plot.title = element_text(face = "bold", size = 11), plot.subtitle = element_text(size = 8.2), plot.caption = element_text(size = 7.1, hjust = 0), plot.caption.position = "plot")
  )

tikz(output_path, width = 7.25, height = 6.0, standAlone = FALSE)
print(plot)
dev.off()
body <- readLines(output_path, warn = FALSE)
body <- body[!grepl("^% Created by tikzDevice", body)]
writeLines(c(
  "% SOURCE: edit-harness/results/quant_survival/aggregate/quant_survival_repair_v1.json",
  "% SOURCE FIELDS A-C: cells[*].{slug,editor,layer,absolute_fp32_esr.point}; cells[*].arms[*].{absolute_quantized_esr.point,conditional_survival_given_fp32_worked.point}; module_provenance.{version,n_boot}",
  "% SOURCE D: edit-harness/results/quant_survival/llama1b_rome_L12_s0/QS_phase1_table.json; edit-harness/results/quant_survival/llama1b_rome_L12_s1/QS_phase1_table.json; edit-harness/results/quant_survival/llama1b_rome_L12_s2/QS_phase1_table.json",
  "% SOURCE D: edit-harness/results/quant_survival/llama3b_rome_L24_s0/QS_phase1_table.json; edit-harness/results/quant_survival/llama3b_rome_L24_s1/QS_phase1_table.json; edit-harness/results/quant_survival/llama3b_rome_L24_s2/QS_phase1_table.json",
  "% SOURCE FIELDS D: arms.nf4dq_full_model.rho_damage_fp32_vs_arm_rank_survival; plotted point = arithmetic mean over seeds 0,1,2; range = seed min-max; threshold = 0.85",
  body
), output_path, useBytes = TRUE)

report <- c(
  "Paper B F2 per-panel recomputation",
  paste0("canonical=", aggregate_path),
  "A absolute efficacy by model/editor = mean(cells[*].arms[*].absolute_quantized_esr.point over four arms)",
  paste(capture.output(print(p_a_data, row.names = FALSE)), collapse = "\n"),
  "B conditional efficacy by model/editor = mean(cells[*].arms[*].conditional_survival_given_fp32_worked.point over four arms)",
  paste(capture.output(print(p_b_data, row.names = FALSE)), collapse = "\n"),
  "C INT8 conditional efficacy rows (edited-layer and full-model retained separately)",
  paste(capture.output(print(p_c_data[c("model_label", "editor_label", "locality", "conditional")], row.names = FALSE)), collapse = "\n"),
  "D K1 raw recompute = mean over three raw per-seed flat rank-survival values; threshold=0.85",
  paste(capture.output(print(k1, row.names = FALSE)), collapse = "\n"),
  "ASSERT: each recomputed K1 point matches quant_survival_repair_v1.json flat_rank.point within 5e-5"
)
writeLines(report, report_path, useBytes = TRUE)
cat("wrote", output_path, "and", report_path, "\n")
