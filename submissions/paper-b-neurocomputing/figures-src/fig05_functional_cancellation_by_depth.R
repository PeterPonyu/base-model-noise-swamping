#!/usr/bin/env Rscript
# SOURCE: edit-harness/results/quant_survival/aggregate/gate_readout.json
# SOURCE FIELDS: cells.*.c3.{nf4dq,int8}.{r_func_mean,r_param_mean,F_above_mean}; cells.*.{model,editor,layer}
# One canonical point is a model/editor/layer/scheme aggregate over the source cell's seeds.

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
default_output <- file.path(dirname(dirname(script_path)), "fig05_functional_cancellation_by_depth.tex")
input_path <- normalizePath(arg_value("--input", default_input), mustWork = TRUE)
output_path <- arg_value("--output", default_output)
dry_run <- has_flag("--dry")

scalar_number <- function(x, label) {
  if (length(x) != 1L || !is.numeric(x) || !is.finite(x)) stop(label, " must be one finite number", call. = FALSE)
  as.numeric(x)
}

source <- fromJSON(input_path, simplifyVector = FALSE)
if (!is.list(source$cells) || length(source$cells) == 0L) stop("schema: cells must be a non-empty object", call. = FALSE)

rows <- list()
i <- 0L
for (cell_name in names(source$cells)) {
  matched <- regexec("^(.*)_(rome|memit|alpha)_L([0-9]+)$", cell_name, ignore.case = TRUE)
  fields <- regmatches(cell_name, matched)[[1L]]
  if (length(fields) != 4L) stop("schema: unsupported cell key: ", cell_name, call. = FALSE)
  model <- fields[[2L]]
  editor <- toupper(fields[[3L]])
  if (editor == "ALPHA") editor <- "AlphaEdit"
  if (editor == "ROME") editor <- "ROME"
  if (editor == "MEMIT") editor <- "MEMIT"
  layer <- as.integer(fields[[4L]])
  c3 <- source$cells[[cell_name]]$c3
  if (!is.list(c3)) stop("schema: missing c3 object for ", cell_name, call. = FALSE)

  for (scheme_key in c("nf4dq", "int8")) {
    block <- c3[[scheme_key]]
    if (!is.list(block)) stop("schema: missing c3.", scheme_key, " for ", cell_name, call. = FALSE)
    r_func <- scalar_number(block$r_func_mean, paste0(cell_name, ".c3.", scheme_key, ".r_func_mean"))
    r_param <- scalar_number(block$r_param_mean, paste0(cell_name, ".c3.", scheme_key, ".r_param_mean"))
    f_above <- scalar_number(block$F_above_mean, paste0(cell_name, ".c3.", scheme_key, ".F_above_mean"))
    if (r_func <= 0 || r_param <= 0) stop("schema: ratios must be positive for log scales: ", cell_name, call. = FALSE)
    if (f_above < 0 || f_above > 1) stop("schema: F_above_mean must lie in [0,1]: ", cell_name, call. = FALSE)
    i <- i + 1L
    rows[[i]] <- data.frame(
      cell = cell_name,
      model = model,
      layer = layer,
      editor = editor,
      scheme = if (scheme_key == "nf4dq") "NF4" else "INT8",
      r_func = r_func,
      r_param = r_param,
      f_above = f_above,
      stringsAsFactors = FALSE
    )
  }
}

d <- do.call(rbind, rows)
if (nrow(d) != 18L) stop("schema: expected 18 model/editor/scheme points, found ", nrow(d), call. = FALSE)
if (!all(d$r_func < d$r_param)) stop("data check: r_func < r_param does not hold in all 18 points", call. = FALSE)
model_layers <- unique(d[c("model", "layer")])
if (any(duplicated(model_layers$model))) stop("scope changed: a within-model layer sweep now exists; revise figure interpretation", call. = FALSE)

cat("schema OK:", nrow(d), "points;", nrow(model_layers), "model-layer pairs; r_func < r_param in all points\n")
cat("scope: one layer per model; depth is confounded with model\n")

required_rome <- c("Llama-3.2-1B_rome_L12", "Qwen2.5-1.5B_rome_L21", "Llama-3.2-3B_rome_L24")
if (!all(required_rome %in% d$cell)) stop("schema: missing a required ROME cross-model cell", call. = FALSE)
validated_rome <- c("Llama-3.2-1B_rome_L12", "Llama-3.2-3B_rome_L24")

if (dry_run) quit(status = 0L)

C_BLUE <- "#0072B2"
C_ORANGE <- "#D55E00"
C_TEAL <- "#009E73"
C_PURPLE <- "#6A3D9A"
scheme_colors <- c("NF4" = C_ORANGE, "INT8" = C_BLUE)
editor_shapes <- c("ROME" = 16, "MEMIT" = 17, "AlphaEdit" = 15)
model_colors <- c("Llama-3.2-1B" = C_BLUE, "Qwen2.5-1.5B" = C_TEAL, "Llama-3.2-3B" = C_ORANGE)

paper_theme <- theme_classic(base_size = 10.5, base_family = "sans") +
  theme(
    plot.title = element_text(face = "bold", size = 10.5),
    plot.subtitle = element_text(size = 8.1),
    axis.title = element_text(size = 9),
    axis.text = element_text(size = 8),
    legend.position = "bottom",
    legend.text = element_text(size = 7.7),
    plot.caption = element_text(size = 7.2, hjust = 0),
    plot.caption.position = "plot",
    plot.margin = margin(5, 6, 6, 6)
  )

# A: all 18 model/editor/layer/scheme aggregates.
p_a <- ggplot(d, aes(r_param, r_func, colour = scheme, shape = editor)) +
  geom_abline(slope = 1, intercept = 0, linetype = "22", colour = "grey55", linewidth = 0.35) +
  geom_point(size = 2.1, alpha = 0.88) +
  scale_x_log10() + scale_y_log10() +
  scale_colour_manual(values = scheme_colors) +
  scale_shape_manual(values = editor_shapes) +
  labs(title = "Functional cancellation", subtitle = "All 18 aggregates fall below equality", x = expression(r[param]), y = expression(r[func]), colour = NULL, shape = NULL) +
  paper_theme

# B: measured K3 axis on the two validated ROME cells.
b <- d[d$cell %in% validated_rome & d$scheme == "NF4", ]
b$label <- factor(b$model, levels = c("Llama-3.2-1B", "Llama-3.2-3B"), labels = c("Llama-1B\nL12", "Llama-3B\nL24"))
p_b <- ggplot(b, aes(label, f_above, fill = model)) +
  geom_col(width = 0.56) +
  geom_text(aes(label = sprintf("%.4f", f_above)), vjust = -0.45, size = 2.6) +
  scale_fill_manual(values = model_colors, guide = "none") +
  scale_y_continuous(limits = c(0, 0.0042), breaks = c(0, 0.002, 0.004), expand = c(0, 0)) +
  labs(title = "Bin-width sensitivity", subtitle = expression(F[above] == Pr(abs(Delta*W) > b)), x = NULL, y = expression(F[above])) +
  paper_theme

# C: exactly three ROME/NF4 single-layer points; no line is permitted.
c <- d[d$cell %in% required_rome & d$scheme == "NF4", ]
c$label <- factor(c$model, levels = c("Llama-3.2-1B", "Qwen2.5-1.5B", "Llama-3.2-3B"), labels = c("Llama-1B\nL12", "Qwen-1.5B\nL21", "Llama-3B\nL24"))
p_c <- ggplot(c, aes(label, r_func, colour = model)) +
  geom_point(size = 3) +
  geom_text(aes(label = sprintf("%.3f", r_func)), vjust = -0.8, size = 2.6, show.legend = FALSE) +
  scale_colour_manual(values = model_colors, guide = "none") +
  scale_y_log10(limits = c(0.01, 0.13), breaks = c(0.02, 0.05, 0.1)) +
  labs(title = "Cross-model snapshot", subtitle = "NOT a depth profile: one layer per model", x = NULL, y = expression(r[func])) +
  paper_theme

# D: visual scope notice; no quantitative claim is encoded here.
notice <- data.frame(
  x = c(1, 2, 3, 2), y = c(2, 2, 2, 1),
  label = c("model width", "+", "layer index", "multi-layer sweep")
)
p_d <- ggplot() +
  annotate("segment", x = 1.25, xend = 1.78, y = 2, yend = 2, linewidth = 0.5, colour = "grey45") +
  annotate("segment", x = 2.22, xend = 2.75, y = 2, yend = 2, linewidth = 0.5, colour = "grey45") +
  annotate("segment", x = 2, xend = 2, y = 1.78, yend = 1.25, arrow = arrow(length = unit(0.12, "cm")), linewidth = 0.5, colour = C_PURPLE) +
  geom_point(data = notice[notice$label != "+", ], aes(x, y), shape = 21, size = 6.3, stroke = 0.6, fill = "white", colour = c(C_BLUE, C_ORANGE, C_PURPLE)) +
  geom_text(data = notice, aes(x, y, label = label), size = 2.65, fontface = c("plain", "bold", "plain", "bold")) +
  annotate("text", x = 2, y = 0.62, label = "needed to separate them", size = 2.55) +
  coord_cartesian(xlim = c(0.45, 3.55), ylim = c(0.35, 2.45), clip = "off") +
  labs(title = "Width--depth confound", subtitle = "Current data cannot identify either effect") +
  theme_void(base_size = 10.5, base_family = "sans") +
  theme(plot.title = element_text(face = "bold", size = 10.5), plot.subtitle = element_text(size = 8.1), plot.margin = margin(5, 6, 6, 6))

plot <- (p_a + p_b) / (p_c + p_d) +
  plot_layout(guides = "collect") +
  plot_annotation(
    title = "Functional cancellation and its scope",
    caption = "A: all cell-arm aggregates. B: validated ROME/NF4 cells. C: three single-layer ROME/NF4 observations; no connecting line. D: identification limit.",
    tag_levels = "A",
    theme = theme(
      plot.tag = element_text(face = "bold", size = 10),
      plot.title = element_text(face = "bold", size = 11),
      plot.caption = element_text(size = 7.2, hjust = 0),
      plot.caption.position = "plot",
      legend.position = "bottom"
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
  "% SOURCE FIELDS F5-A: cells.*.c3.{nf4dq,int8}.{r_func_mean,r_param_mean} (18 model/editor/layer/scheme aggregates).",
  "% SOURCE FIELDS F5-B: cells.{Llama-3.2-1B_rome_L12,Llama-3.2-3B_rome_L24}.c3.nf4dq.F_above_mean.",
  "% SOURCE FIELDS F5-C: cells.{Llama-3.2-1B_rome_L12,Qwen2.5-1.5B_rome_L21,Llama-3.2-3B_rome_L24}.{model,layer,c3.nf4dq.r_func_mean}.",
  "% SCOPE F5-C/F5-D: one layer per model; depth is confounded with model; this is not a within-model depth profile.",
  body
), output_path, useBytes = TRUE)
cat("wrote", normalizePath(output_path, mustWork = TRUE), "\n")
