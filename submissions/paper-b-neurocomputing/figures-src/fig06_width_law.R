#!/usr/bin/env Rscript
# SOURCE: edit-harness/results/merging/*/RG_operating_curve_table.json
# SOURCE FIELDS: model; layer; per_g_summary[*].qualifies; verdict.overall
# Boundary definition: max(integer g where per_g_summary[g].qualifies is true); undefined for INCONCLUSIVE.

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
merge_root <- file.path(repo_root, "edit-harness", "results", "merging")
output_path <- file.path(paper_dir, "fig06_width_law.tex")
report_path <- file.path(paper_dir, "fig06_recompute_report.txt")

spec <- data.frame(
  model = c("Llama-3.2-1B", "Llama-3.1-8B", "Mistral-7B-v0.3", "Qwen2.5-1.5B", "Qwen2.5-7B", "Qwen2.5-14B"),
  family = c("Llama", "Llama", "Mistral", "Qwen", "Qwen", "Qwen"),
  width_b = c(1, 8, 7, 1.5, 7, 14),
  layer = c(12L, 24L, 24L, 21L, 21L, 36L),
  rel_depth = c(.75, .75, .75, .75, .75, .75),
  source_rel = c(
    "Llama-3.2-1B_L12_RG/RG_operating_curve_table.json",
    "Llama-3.1-8B_L24_RG/RG_operating_curve_table.json",
    "Mistral-7B-v0.3_L24_RG/RG_operating_curve_table.json",
    "Qwen2.5-1.5B_L21_RG/RG_operating_curve_table.json",
    "Qwen2.5-7B_L21_RG/RG_operating_curve_table.json",
    "Qwen2.5-14B_L36_RG/RG_operating_curve_table.json"
  ), stringsAsFactors = FALSE
)

extract_boundary <- function(source_rel, expected_model, expected_layer) {
  path <- normalizePath(file.path(merge_root, source_rel), mustWork = TRUE)
  x <- fromJSON(path, simplifyVector = FALSE)
  stopifnot(as.integer(x$layer) == expected_layer)
  observed_name <- basename(x$model)
  if (observed_name != expected_model) stop("model mismatch: ", observed_name, " != ", expected_model, call. = FALSE)
  if (!is.list(x$per_g_summary) || length(x$per_g_summary) == 0L) stop("missing per_g_summary: ", source_rel, call. = FALSE)
  qualifying <- sort(as.integer(names(x$per_g_summary)[vapply(x$per_g_summary, function(g) isTRUE(g$qualifies), logical(1))]))
  verdict <- as.character(x$verdict$overall)
  if (verdict == "INCONCLUSIVE" && length(qualifying) > 0L) stop("inconclusive cell has qualifying g: ", source_rel, call. = FALSE)
  list(boundary = if (length(qualifying)) max(qualifying) else NA_real_, qualifying = qualifying, verdict = verdict, path = path)
}

computed <- lapply(seq_len(nrow(spec)), function(i) extract_boundary(spec$source_rel[i], spec$model[i], spec$layer[i]))
spec$boundary <- vapply(computed, `[[`, numeric(1), "boundary")
spec$qualifying <- vapply(computed, function(x) if (length(x$qualifying)) paste(x$qualifying, collapse = ",") else "none", character(1))
spec$verdict <- vapply(computed, `[[`, character(1), "verdict")
spec$source_path <- vapply(computed, `[[`, character(1), "path")
spec$cell_label <- sprintf("%s L%d", sub("Llama-3\\.2-", "Llama-", sub("Llama-3\\.1-", "Llama-", sub("Qwen2\\.5-", "Qwen-", sub("-v0\\.3", "", spec$model)))), spec$layer)

stopifnot(spec$boundary[spec$model == "Llama-3.2-1B"] == 5)
stopifnot(spec$boundary[spec$model == "Llama-3.1-8B"] == 20)
stopifnot(spec$boundary[spec$model == "Mistral-7B-v0.3"] == 20)
stopifnot(is.na(spec$boundary[spec$model == "Qwen2.5-14B"]), spec$verdict[spec$model == "Qwen2.5-14B"] == "INCONCLUSIVE")

family_colors <- c(Llama = "#0072B2", Mistral = "#009E73", Qwen = "#D55E00")
paper_theme <- theme_classic(base_size = 10.2, base_family = "sans") + theme(
  plot.title = element_text(face = "bold", size = 10.2), plot.subtitle = element_text(size = 8.0),
  axis.title = element_text(size = 8.8), axis.text = element_text(size = 7.8),
  legend.position = "bottom", legend.text = element_text(size = 7.8),
  plot.caption = element_text(size = 7.0, hjust = 0), plot.caption.position = "plot",
  plot.margin = margin(5, 6, 6, 6)
)

p_a <- ggplot(spec, aes(width_b, boundary, colour = family, shape = family)) +
  geom_line(data = spec[spec$family %in% c("Llama", "Qwen") & !is.na(spec$boundary), ], aes(group = family), linewidth = .55) +
  geom_point(data = spec[!is.na(spec$boundary), ], size = 2.5) +
  geom_point(data = spec[is.na(spec$boundary), ], aes(y = 1), shape = 4, size = 3, stroke = .7) +
  geom_text(data = spec[is.na(spec$boundary), ], aes(y = 2.4, label = "undefined"), size = 2.4, hjust = 1, show.legend = FALSE) +
  scale_colour_manual(values = family_colors) + scale_shape_manual(values = c(Llama = 16, Mistral = 17, Qwen = 15)) +
  scale_x_log10(breaks = c(1, 1.5, 7, 8, 14), labels = c("1", "1.5", "7", "8", "14")) +
  scale_y_continuous(limits = c(0, 21), breaks = c(0, 5, 10, 20), expand = expansion(mult = c(.01, .03))) +
  labs(title = "Geometry-valid boundary by width", subtitle = "All cells at 75\\% relative depth", x = "model width (B parameters, log scale)", y = "max qualifying group size", colour = "family", shape = "family") + paper_theme

within <- spec[spec$family %in% c("Llama", "Qwen"), ]
within$status_y <- ifelse(is.na(within$boundary), 1, within$boundary)
within$status <- ifelse(is.na(within$boundary), "undefined", paste0("g=", within$boundary))
# Qwen 7B (g=20 at x=7) sits adjacent to Llama 8B (g=20 at x=8) on the log axis:
# place the Qwen label below its point so the two g=20 labels never collide.
within$label_vjust <- ifelse(!is.na(within$boundary) & within$family == "Qwen" & within$width_b == 7, 1.6, -1.0)
p_b <- ggplot(within, aes(width_b, status_y, colour = family, shape = family, group = family)) +
  geom_line(data = within[!is.na(within$boundary), ], linewidth = .65) + geom_point(size = 2.6) +
  geom_text(aes(label = status, vjust = label_vjust, hjust = ifelse(within$family == "Qwen" & is.na(within$boundary), 1, 0.5)), size = 2.5, show.legend = FALSE) +
  scale_colour_manual(values = family_colors) +
  scale_shape_manual(values = c(Llama = 16, Qwen = 15)) +
  scale_x_log10(breaks = c(1, 1.5, 7, 8, 14), labels = c("1", "1.5", "7", "8", "14")) +
  scale_y_continuous(limits = c(0, 22), breaks = c(0, 5, 10, 20), expand = expansion(mult = c(.01, .05))) +
  labs(title = "Within-family ordering", subtitle = "Llama holds; Qwen 14B is undefined, not zero", x = "model width (B parameters, log scale)", y = "geometry-valid boundary", colour = NULL, shape = NULL) + paper_theme + theme(legend.position = "bottom")

confound <- spec[spec$model %in% c("Llama-3.2-1B", "Mistral-7B-v0.3", "Llama-3.1-8B"), ]
confound$comparison <- factor(ifelse(confound$model == "Mistral-7B-v0.3", "old cross-family", "same-family control"), levels = c("old cross-family", "same-family control"))
p_c <- ggplot(confound, aes(width_b, boundary, colour = family, shape = comparison)) +
  geom_segment(aes(x = 1, xend = 7, y = 5, yend = 20), colour = "#777777", linewidth = .45, linetype = "22") +
  geom_segment(aes(x = 1, xend = 8, y = 5, yend = 20), colour = "#0072B2", linewidth = .7) +
  geom_point(size = 2.8) +
  geom_text(aes(label = cell_label, vjust = ifelse(confound$model == "Mistral-7B-v0.3", -1.0, 1.6)), size = 2.35, show.legend = FALSE) +
  scale_colour_manual(values = family_colors) + scale_shape_manual(values = c("old cross-family" = 17, "same-family control" = 16)) +
  scale_x_continuous(limits = c(.5, 8.6), breaks = c(1, 7, 8)) +
  scale_y_continuous(limits = c(3, 22), breaks = c(5, 10, 20)) +
  labs(title = "Old 1B-vs-Mistral confound", subtitle = "The 8B Llama control reaches the same boundary", x = "model width (B parameters)", y = "geometry-valid boundary", colour = "family", shape = "comparison") + paper_theme

llama <- spec[spec$family == "Llama", ]
p_d <- ggplot(llama, aes(width_b, boundary)) +
  annotate("segment", x = min(llama$width_b), xend = max(llama$width_b),
           y = llama$boundary[which.min(llama$width_b)], yend = llama$boundary[which.max(llama$width_b)],
           colour = family_colors[["Llama"]], linewidth = .8) +
  geom_point(size = 3, colour = family_colors[["Llama"]]) +
  geom_text(aes(label = sprintf("%s: g=%d", sub("Llama-", "", cell_label), boundary)), vjust = -1.0, size = 2.45) +
  annotate("text", x = 4.5, y = 10.5, label = "H-Llama: 5 <= 20", colour = family_colors[["Llama"]], fontface = "bold", size = 3.0) +
  scale_x_continuous(limits = c(.5, 8.6), breaks = c(1, 8)) +
  scale_y_continuous(limits = c(3, 22), breaks = c(5, 10, 20)) +
  labs(title = "H-Llama holds", subtitle = "Same relative depth; different single layers", x = "model width (B parameters)", y = "geometry-valid boundary") + paper_theme

plot <- (p_a | p_b) / (p_c | p_d) +
  plot_annotation(
    title = "Width and the geometry-valid federation boundary",
    subtitle = "Width comparisons use one preregistered 75\\%-relative-depth layer per model; no depth law is inferred.",
    caption = "Boundary = max qualifying g. Cross = INCONCLUSIVE/undefined. Lines join widths within families, not layers within models.",
    tag_levels = "A",
    theme = theme(plot.tag = element_text(face = "bold", size = 10), plot.title = element_text(face = "bold", size = 11), plot.subtitle = element_text(size = 8.2), plot.caption = element_text(size = 7.1, hjust = 0), plot.caption.position = "plot")
  )

tikz(output_path, width = 7.25, height = 6.15, standAlone = FALSE)
print(plot)
dev.off()
body <- readLines(output_path, warn = FALSE)
body <- body[!grepl("^% Created by tikzDevice", body)]
source_lines <- c(
  "% SOURCE: edit-harness/results/merging/Llama-3.2-1B_L12_RG/RG_operating_curve_table.json fields model; layer; per_g_summary[*].qualifies; verdict.overall",
  "% SOURCE: edit-harness/results/merging/Llama-3.1-8B_L24_RG/RG_operating_curve_table.json fields model; layer; per_g_summary[*].qualifies; verdict.overall",
  "% SOURCE: edit-harness/results/merging/Mistral-7B-v0.3_L24_RG/RG_operating_curve_table.json fields model; layer; per_g_summary[*].qualifies; verdict.overall",
  "% SOURCE: edit-harness/results/merging/Qwen2.5-1.5B_L21_RG/RG_operating_curve_table.json fields model; layer; per_g_summary[*].qualifies; verdict.overall",
  "% SOURCE: edit-harness/results/merging/Qwen2.5-7B_L21_RG/RG_operating_curve_table.json fields model; layer; per_g_summary[*].qualifies; verdict.overall",
  "% SOURCE: edit-harness/results/merging/Qwen2.5-14B_L36_RG/RG_operating_curve_table.json fields model; layer; per_g_summary[*].qualifies; verdict.overall",
  "% DERIVATION: boundary=max(integer g where per_g_summary[g].qualifies==true); INCONCLUSIVE maps to undefined, never zero.",
  "% SCOPE: all selected cells are 75% relative depth; model width and edited layer remain distinct fields; no within-model depth curve is plotted."
)
writeLines(c(source_lines, body), output_path, useBytes = TRUE)

report <- c(
  "Paper B F6 per-panel recomputation",
  "Boundary definition: max(integer g where per_g_summary[g].qualifies is true); INCONCLUSIVE => undefined (NA), not zero.",
  paste(capture.output(print(spec[c("model", "family", "width_b", "layer", "rel_depth", "qualifying", "boundary", "verdict", "source_rel")], row.names = FALSE)), collapse = "\n"),
  "A: all six width rows; Qwen-14B plotted as undefined cross.",
  "B: within-family ordering. Llama 1B->8B is 5->20 (holds). Qwen 1.5B->7B is 20->20; 14B is INCONCLUSIVE/undefined, so no three-point monotonic claim.",
  "C: old cross-family Llama-1B->Mistral-7B is 5->20; same-family Llama-1B->Llama-8B is also 5->20, removing family as a necessary explanation for that widening.",
  "D: H-Llama recomputed directly as boundary(Llama-1B)=5 <= boundary(Llama-8B)=20.",
  "SCOPE ASSERT: all six rows use relative_depth=0.75; each model contributes exactly one layer, which is retained in labels and never treated as a depth trajectory."
)
writeLines(report, report_path, useBytes = TRUE)
cat("wrote", output_path, "and", report_path, "\n")
