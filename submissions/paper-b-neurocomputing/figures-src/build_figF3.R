#!/usr/bin/env Rscript
# Standalone build for Figure F3 - generates both .tex and a standalone PDF for QA

suppressPackageStartupMessages({
  library(jsonlite)
  library(ggplot2)
  library(tikzDevice)
})

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (length(script_arg) != 1L) stop("Run with Rscript", call. = FALSE)
script_path <- normalizePath(sub("^--file=", "", script_arg), mustWork = TRUE)
repo_root <- normalizePath(file.path(dirname(script_path), "..", "..", ".."), mustWork = TRUE)

# Paths
input_path <- file.path(repo_root, "edit-harness", "results", "quant_survival", "aggregate", "curve_local_readout.json")
tex_output <- file.path(dirname(dirname(script_path)), "figF3_noise_signal_rank_survival.tex")
standalone_dir <- file.path(dirname(dirname(script_path)), "standalone_qa")
dir.create(standalone_dir, showWarnings = FALSE, recursive = TRUE)
standalone_tex <- file.path(standalone_dir, "figF3_standalone.tex")
standalone_pdf <- file.path(standalone_dir, "figF3_standalone.pdf")

if (!file.exists(input_path)) {
  cat("Input file not found:", input_path, "\n")
  cat("Status: INCOMPLETE (data not yet computed)\n")
  quit(status = 3L)
}

# Load and validate
x <- fromJSON(input_path, simplifyVector = FALSE)
if (is.null(x$schema_version) || !identical(x$schema_version, 2L)) {
  stop("schema_version must be 2", call. = FALSE)
}

if (identical(x$status, "INCOMPLETE")) {
  cat("Status: INCOMPLETE - cannot generate figure\n")
  if (!is.null(x$missing)) {
    cat("Missing", length(x$missing), "cells\n")
  }
  quit(status = 3L)
}

if (!identical(x$status, "PRE_B4_READOUT")) {
  stop("status must be PRE_B4_READOUT or INCOMPLETE", call. = FALSE)
}

# Minimal validation (full validation is in figF3_noise_signal_rank_survival.R)
if (is.null(x$curve_points) || length(x$curve_points) == 0L) {
  stop("curve_points is empty", call. = FALSE)
}

# Convert to data frame
df <- data.frame(
  model = sapply(x$curve_points, `[[`, "model"),
  noise_to_signal = sapply(x$curve_points, `[[`, "noise_to_signal"),
  nf4_rank_survival = sapply(x$curve_points, `[[`, "nf4_rank_survival"),
  stringsAsFactors = FALSE
)

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

if (!is.null(x$Q3_nsr_rho) && is.finite(x$Q3_nsr_rho)) {
  rho_text <- sprintf("$\\rho_{\\text{Spearman}} = %.3f$", x$Q3_nsr_rho)
  pass_text <- if (isTRUE(x$Q3_PASS)) " (Q3 PASS)" else " (Q3 FAIL)"
  annotation_text <- paste0(rho_text, pass_text)
  p <- p + annotate("text", x = max(df$noise_to_signal) * 0.5, y = 0.95,
                    label = annotation_text, hjust = 0.5, size = 3)
}

# Generate paper .tex
cat("Generating paper figure:", tex_output, "\n")
tikz(tex_output, width = 5, height = 3.5, standAlone = FALSE)
print(p)
dev.off()

original <- readLines(tex_output)
header <- c(
  "% SOURCE: edit-harness/results/quant_survival/aggregate/curve_local_readout.json",
  paste0("% SOURCE SCHEMA: paperb_curve_readout.py schema_version=", x$schema_version),
  paste0("% G-S3 STATUS: ", if (isTRUE(x$G_S3_PASS)) "PASS" else "FAIL"),
  paste0("% CURVE POINTS: ", nrow(df)),
  paste0("% GENERATED: ", format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z")),
  ""
)
writeLines(c(header, original), tex_output)

# Generate standalone PDF for QA
cat("Generating standalone PDF for QA:", standalone_pdf, "\n")
tikz(standalone_tex, width = 5, height = 3.5, standAlone = TRUE)
print(p)
dev.off()

# Compile standalone PDF
system2("pdflatex", args = c("-interaction=nonstopmode", "-output-directory", standalone_dir, standalone_tex),
        stdout = FALSE, stderr = FALSE)

if (file.exists(standalone_pdf)) {
  cat("SUCCESS: Figure generated\n")
  cat("  Paper .tex:", tex_output, "\n")
  cat("  QA PDF:", standalone_pdf, "\n")
  cat("  G-S3:", if (isTRUE(x$G_S3_PASS)) "PASS" else "FAIL", "\n")
  cat("  Points:", nrow(df), "\n")
} else {
  cat("WARNING: Paper .tex generated but standalone PDF compilation failed\n")
  cat("  Check LaTeX installation and run manually:\n")
  cat("  cd", standalone_dir, "&& pdflatex figF3_standalone.tex\n")
}
