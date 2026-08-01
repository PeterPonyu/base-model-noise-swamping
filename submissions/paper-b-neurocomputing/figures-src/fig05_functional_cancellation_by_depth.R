#!/usr/bin/env Rscript
# SOURCE: edit-harness/results/quant_survival/aggregate/gate_readout.json
# SOURCE FIELDS: cells.*.c3.{nf4dq,int8}.{r_func_mean,r_param_mean}
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
    if (r_func <= 0 || r_param <= 0) stop("schema: ratios must be positive for log scales: ", cell_name, call. = FALSE)
    i <- i + 1L
    rows[[i]] <- data.frame(
      cell = cell_name,
      model = model,
      layer = layer,
      editor = editor,
      scheme = if (scheme_key == "nf4dq") "NF4" else "INT8",
      r_func = r_func,
      r_param = r_param,
      cancellation_ratio = r_func / r_param,
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
if (dry_run) quit(status = 0L)

model_labels <- c(
  "Llama-3.2-1B" = "Llama-1B",
  "Qwen2.5-1.5B" = "Qwen-1.5B",
  "Llama-3.2-3B" = "Llama-3B"
)
model_colors <- c(
  "Llama-3.2-1B" = "#0072B2",
  "Qwen2.5-1.5B" = "#009E73",
  "Llama-3.2-3B" = "#D55E00"
)
editor_shapes <- c("ROME" = 16, "MEMIT" = 17, "AlphaEdit" = 15)
d$scheme <- factor(d$scheme, levels = c("NF4", "INT8"))
d$editor <- factor(d$editor, levels = c("ROME", "MEMIT", "AlphaEdit"))

paper_theme <- theme_classic(base_size = 10.5, base_family = "sans") +
  theme(
    plot.title = element_text(face = "bold", size = 10.5),
    plot.subtitle = element_text(size = 8.2),
    axis.title = element_text(size = 9),
    axis.text = element_text(size = 8.2),
    strip.text = element_text(face = "bold", size = 8.5),
    legend.position = "bottom",
    legend.text = element_text(size = 8),
    plot.caption = element_text(size = 7.4, hjust = 0),
    plot.caption.position = "plot",
    plot.margin = margin(5, 6, 6, 6)
  )

common <- list(
  geom_point(aes(colour = model, shape = editor), size = 2.2, alpha = 0.85),
  facet_wrap(~scheme, nrow = 1),
  scale_x_continuous(breaks = sort(unique(d$layer))),
  scale_colour_manual(values = model_colors, labels = model_labels),
  scale_shape_manual(values = editor_shapes),
  paper_theme
)

p_a <- ggplot(d, aes(layer, r_func)) + common +
  scale_y_log10() +
  labs(title = "Functional reconstruction ratio", x = "edited layer", y = expression(r[func]), colour = "model", shape = "editor")

p_b <- ggplot(d, aes(layer, cancellation_ratio)) + common +
  scale_y_log10() +
  labs(title = "Functional-to-parameter gap", x = "edited layer", y = expression(r[func] / r[param]), colour = "model", shape = "editor")

plot <- p_a / p_b +
  plot_layout(guides = "collect") +
  plot_annotation(
    title = "Functional cancellation by layer-indexed cell",
    subtitle = "One layer per model: cross-model view, not a within-model depth sweep.",
    caption = "Each point is one model/editor/scheme aggregate. Lines are omitted because layer and model are confounded.",
    tag_levels = "A",
    theme = theme(
      plot.tag = element_text(face = "bold", size = 10),
      plot.title = element_text(face = "bold", size = 11),
      plot.subtitle = element_text(size = 8.5),
      plot.caption = element_text(size = 7.4, hjust = 0),
      plot.caption.position = "plot",
      legend.position = "bottom"
    )
  )

dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
tikz(output_path, width = 7.2, height = 5.2, standAlone = FALSE)
print(plot)
dev.off()
body <- readLines(output_path, warn = FALSE)
body <- body[!grepl("^% Created by tikzDevice", body)]
writeLines(c(
  "% SOURCE: edit-harness/results/quant_survival/aggregate/gate_readout.json",
  "% SOURCE FIELDS: cells.*.c3.{nf4dq,int8}.{r_func_mean,r_param_mean}",
  "% SCOPE: one layer per model; cross-model layer-indexed view, not a within-model depth sweep.",
  body
), output_path, useBytes = TRUE)
cat("wrote", normalizePath(output_path, mustWork = TRUE), "\n")
