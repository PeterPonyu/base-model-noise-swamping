#!/usr/bin/env Rscript
# SOURCE: edit-harness/results/quant_survival/aggregate/curve_local_readout.json
# SOURCE SCHEMA: paperb_curve_readout.py schema_version=2 with curve_points array.
# This script intentionally refuses to invent curve points absent from that canonical JSON.

suppressPackageStartupMessages({
  library(jsonlite)
  library(ggplot2)
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
default_input <- file.path(repo_root, "edit-harness", "results", "quant_survival", "aggregate", "curve_local_readout.json")
default_output <- file.path(dirname(dirname(script_path)), "figF3_noise_signal_rank_survival.tex")
input_arg <- path.expand(arg_value("--input", default_input))
output_path <- arg_value("--output", default_output)
dry_run <- has_flag("--dry")

if (!file.exists(input_arg)) {
  cat("Status: INCOMPLETE - canonical curve readout does not exist\n")
  cat("Missing input: ", input_arg, "\n", sep = "")
  if (dry_run) quit(status = 3L)
  stop("Cannot generate figure: canonical curve readout is missing", call. = FALSE)
}
input_path <- normalizePath(input_arg, mustWork = TRUE)

x <- fromJSON(input_path, simplifyVector = FALSE)

# Validate schema version
if (is.null(x$schema_version) || !identical(x$schema_version, 2L)) {
  stop("schema: schema_version must be 2 (got: ", x$schema_version, ")", call. = FALSE)
}

# Check status
if (is.null(x$status) || !x$status %in% c("PRE_B4_READOUT", "INCOMPLETE")) {
  stop("schema: status must be PRE_B4_READOUT or INCOMPLETE", call. = FALSE)
}

is_incomplete <- identical(x$status, "INCOMPLETE")

if (is_incomplete) {
  cat("Status: INCOMPLETE - cannot plot without all required cells\n")
  if (!is.null(x$missing)) {
    cat("Missing cells:\n")
    for (m in x$missing) cat("  ", m, "\n", sep = "")
  }
  if (dry_run) quit(status = 3L)
  stop("Cannot generate figure: data INCOMPLETE", call. = FALSE)
}

# Validate complete schema
required_scalars <- c(
  "status", "qwen15b_mean", "qwen3b_mean", "gemma2b_mean", "phi35_mean",
  "llama1b_mean", "llama3b_mean", "Q1_qwen_monotone", "Q1_llama_partial_monotone",
  "Q2_family_separation", "Q3_nsr_rho", "Q3_PASS", "G_S3_PASS", "note",
  "schema_version", "curve_points"
)
missing <- setdiff(required_scalars, names(x))
if (length(missing) > 0L) {
  stop("schema: missing fields: ", paste(missing, collapse = ", "), call. = FALSE)
}

numeric_fields <- c("qwen15b_mean", "qwen3b_mean", "gemma2b_mean", "phi35_mean", "llama1b_mean", "llama3b_mean")
for (field in numeric_fields) {
  value <- x[[field]]
  if (length(value) != 1L || !is.numeric(value) || !is.finite(value)) {
    stop("schema: ", field, " must be one finite number", call. = FALSE)
  }
}

for (field in c("Q1_qwen_monotone", "Q1_llama_partial_monotone", "Q2_family_separation", "Q3_PASS", "G_S3_PASS")) {
  if (length(x[[field]]) != 1L || !is.logical(x[[field]])) {
    stop("schema: ", field, " must be one boolean", call. = FALSE)
  }
}

if (is.null(x$Q3_nsr_rho)) {
  if (isTRUE(x$Q3_PASS) || isTRUE(x$G_S3_PASS)) {
    stop("schema: null Q3_nsr_rho requires Q3_PASS and G_S3_PASS to be false", call. = FALSE)
  }
} else if (length(x$Q3_nsr_rho) != 1L || !is.numeric(x$Q3_nsr_rho) || !is.finite(x$Q3_nsr_rho)) {
  stop("schema: Q3_nsr_rho must be one finite number or null", call. = FALSE)
}

if (!is.list(x$seed_values) || !setequal(names(x$seed_values), c("qwen3b", "gemma2b", "phi35"))) {
  stop("schema: seed_values must contain exactly qwen3b, gemma2b, and phi35", call. = FALSE)
}

for (family in names(x$seed_values)) {
  values <- unlist(x$seed_values[[family]], use.names = FALSE)
  if (length(values) != 3L || !is.numeric(values) || any(!is.finite(values))) {
    stop("schema: seed_values.", family, " must contain three finite numbers", call. = FALSE)
  }
}

# Validate curve_points
if (!is.list(x$curve_points) || length(x$curve_points) == 0L) {
  stop("schema: curve_points must be a non-empty list", call. = FALSE)
}

required_point_fields <- c("model", "layer", "seed", "noise_to_signal", "nf4_rank_survival",
                           "source_table", "source_raw", "table_sha256", "raw_sha256")
for (i in seq_along(x$curve_points)) {
  pt <- x$curve_points[[i]]
  missing_pt <- setdiff(required_point_fields, names(pt))
  if (length(missing_pt) > 0L) {
    stop("schema: curve_points[", i, "] missing: ", paste(missing_pt, collapse = ", "), call. = FALSE)
  }
  if (!is.character(pt$model) || length(pt$model) != 1L) {
    stop("schema: curve_points[", i, "].model must be one string", call. = FALSE)
  }
  if (!is.numeric(pt$layer) || length(pt$layer) != 1L) {
    stop("schema: curve_points[", i, "].layer must be one number", call. = FALSE)
  }
  if (!is.numeric(pt$seed) || length(pt$seed) != 1L) {
    stop("schema: curve_points[", i, "].seed must be one number", call. = FALSE)
  }
  if (!is.numeric(pt$noise_to_signal) || length(pt$noise_to_signal) != 1L || !is.finite(pt$noise_to_signal)) {
    stop("schema: curve_points[", i, "].noise_to_signal must be one finite number", call. = FALSE)
  }
  if (!is.numeric(pt$nf4_rank_survival) || length(pt$nf4_rank_survival) != 1L || !is.finite(pt$nf4_rank_survival)) {
    stop("schema: curve_points[", i, "].nf4_rank_survival must be one finite number", call. = FALSE)
  }
}

cat("schema OK: PRE_B4_READOUT schema_version=2 with ", length(x$curve_points), " curve points\n", sep = "")
cat("G-S3 status: ", if (isTRUE(x$G_S3_PASS)) "PASS" else "FAIL", "\n", sep = "")

if (dry_run) {
  cat("dry-run: would plot ", length(x$curve_points), " points\n", sep = "")
  quit(status = 0L)
}

# Convert curve_points to data frame
df <- data.frame(
  model = sapply(x$curve_points, `[[`, "model"),
  layer = sapply(x$curve_points, `[[`, "layer"),
  seed = sapply(x$curve_points, `[[`, "seed"),
  noise_to_signal = sapply(x$curve_points, `[[`, "noise_to_signal"),
  nf4_rank_survival = sapply(x$curve_points, `[[`, "nf4_rank_survival"),
  stringsAsFactors = FALSE
)

# Map model names to families for coloring
df$family <- ifelse(grepl("^llama", df$model), "Llama",
             ifelse(grepl("^qwen", df$model), "Qwen",
             ifelse(grepl("^gemma", df$model), "Gemma",
             ifelse(grepl("^phi", df$model), "Phi", "Other"))))

# Create plot
p <- ggplot(df, aes(x = noise_to_signal, y = nf4_rank_survival, color = family)) +
  geom_point(size = 2, alpha = 0.7) +
  scale_x_log10(
    name = "Noise-to-Signal Ratio (log scale)",
    breaks = 10^seq(-2, 2, by = 1)
  ) +
  scale_y_continuous(
    name = "NF4 Rank Survival ($\\rho$)",
    limits = c(0, 1)
  ) +
  scale_color_manual(
    name = "Model Family",
    values = c("Llama" = "#1f77b4", "Qwen" = "#ff7f0e", "Gemma" = "#2ca02c", "Phi" = "#d62728")
  ) +
  theme_minimal() +
  theme(
    legend.position = "right",
    panel.grid.minor = element_blank()
  )

# Add correlation annotation if available
if (!is.null(x$Q3_nsr_rho) && is.finite(x$Q3_nsr_rho)) {
  rho_text <- sprintf("$\\rho_{\\mathrm{Spearman}} = %.3f$", x$Q3_nsr_rho)
  pass_text <- if (isTRUE(x$Q3_PASS)) "Q3 PASS" else "Q3 FAIL"
  annotation_text <- paste0(rho_text, "\n", pass_text)

  p <- p + annotate("text",
                    x = min(df$noise_to_signal) * 1.15,
                    y = 0.95,
                    label = annotation_text,
                    hjust = 0,
                    size = 2.8)
}

# Generate tikz output
cat("Generating tikz output to:", output_path, "\n")
tikz(output_path, width = 5, height = 3.5, standAlone = FALSE)
print(p)
dev.off()

# Prepend SOURCE headers
original <- readLines(output_path)
header <- c(
  "% SOURCE: edit-harness/results/quant_survival/aggregate/curve_local_readout.json",
  paste0("% SOURCE SCHEMA: paperb_curve_readout.py schema_version=", x$schema_version),
  paste0("% G-S3 STATUS: ", if (isTRUE(x$G_S3_PASS)) "PASS" else "FAIL"),
  paste0("% CURVE POINTS: ", nrow(df)),
  paste0("% GENERATED: ", format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z")),
  ""
)
writeLines(c(header, original), output_path)

cat("Figure generated successfully\n")
cat("Output:", output_path, "\n")
