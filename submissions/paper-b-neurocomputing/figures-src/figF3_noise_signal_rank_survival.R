#!/usr/bin/env Rscript
# SOURCE: edit-harness/results/quant_survival/aggregate/curve_local_readout.json
# SOURCE SCHEMA: paperb_curve_readout.py verdict object at lines 65-75.
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
input_path <- normalizePath(arg_value("--input", default_input), mustWork = TRUE)
output_path <- arg_value("--output", default_output)
dry_run <- has_flag("--dry")

x <- fromJSON(input_path, simplifyVector = FALSE)
required_scalars <- c(
  "status", "qwen15b_mean", "qwen3b_mean", "gemma2b_mean", "phi35_mean",
  "llama1b_mean", "llama3b_mean", "Q1_qwen_monotone", "Q1_llama_partial_monotone",
  "Q2_family_separation", "Q3_nsr_rho", "Q3_PASS", "G_S3_PASS", "note"
)
missing <- setdiff(required_scalars, names(x))
if (length(missing) > 0L) stop("schema: missing fields: ", paste(missing, collapse = ", "), call. = FALSE)
if (!identical(x$status, "PRE_B4_READOUT")) stop("schema: status must be PRE_B4_READOUT", call. = FALSE)

numeric_fields <- c("qwen15b_mean", "qwen3b_mean", "gemma2b_mean", "phi35_mean", "llama1b_mean", "llama3b_mean")
for (field in numeric_fields) {
  value <- x[[field]]
  if (length(value) != 1L || !is.numeric(value) || !is.finite(value)) stop("schema: ", field, " must be one finite number", call. = FALSE)
}
for (field in c("Q1_qwen_monotone", "Q1_llama_partial_monotone", "Q2_family_separation", "Q3_PASS", "G_S3_PASS")) {
  if (length(x[[field]]) != 1L || !is.logical(x[[field]])) stop("schema: ", field, " must be one boolean", call. = FALSE)
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

cat("schema OK: exact PRE_B4_READOUT summary object from paperb_curve_readout.py\n")
cat("schema gap: no serialized per-point noise_to_signal values or complete model/layer/seed rows\n")
if (dry_run) quit(status = 0L)

stop(
  paste(
    "Cannot plot noise-to-signal versus NF4 rank survival from the canonical JSON.",
    "paperb_curve_readout.py computes per-seed pairs from QS_phase1_raw.npz but does not serialize them.",
    "Update the canonical readout schema to export model, layer, seed, noise_to_signal, and nf4_rank_survival;",
    "then revise this script against that real versioned schema. No values were invented."
  ),
  call. = FALSE
)

# Plot implementation intentionally follows, but remains unreachable until a real canonical
# point schema is defined. Do not add a guessed field name here: bind this code to the schema
# emitted by paperb_curve_readout.py in the same change that adds the payload.
#
# Expected design after that schema exists:
#   x: noise-to-signal ratio (log scale)
#   y: NF4 full-model rho_damage_fp32_vs_arm_rank_survival
#   point: one model/layer/seed cell
#   colour: model family
#   shape: model size or family, only if represented in the canonical payload
#   smooth: none unless prospectively specified; show the reported Spearman rho as annotation
#   output: tikzDevice vector source with % SOURCE field headers prepended
