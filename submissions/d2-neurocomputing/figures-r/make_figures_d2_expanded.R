#!/usr/bin/env Rscript

# Expanded figure-generation scaffold for the D2 Neurocomputing submission.
# Later figure tasks will add panel construction and tikzDevice output here.
# Run from submissions/d2-neurocomputing so data and output paths remain stable.

library(ggplot2)
library(dplyr)
library(patchwork)
library(jsonlite)
library(tikzDevice)

GAIN_LAW_JSON <- "../../edit-harness/results/merging/RG_gain_law_20260715.json"
SIGNED_REANALYSIS_JSON <- "../../edit-harness/results/merging/RG_signed_reanalysis_20260715.json"
CROSSTERM_ALIGNMENT_JSON <- "../../edit-harness/results/merging/RG_crossterm_alignment_ALL_REFIX20260801.json"
ADMISSION_BENEFIT_JSON <- "../../edit-harness/results/merging/RG_admission_benefit_REFIX20260730.json"
D3_BENEFIT_PREDICTOR_JSON <- "../../edit-harness/results/D3_benefit_predictor_eval.json"
ESR_BY_CELL_JSON <- "../../edit-harness/results/merging/RG_esr_by_cell_20260716.json"

stopifnot(file.exists(GAIN_LAW_JSON))
stopifnot(file.exists(SIGNED_REANALYSIS_JSON))
stopifnot(file.exists(CROSSTERM_ALIGNMENT_JSON))
stopifnot(file.exists(ADMISSION_BENEFIT_JSON))
stopifnot(file.exists(D3_BENEFIT_PREDICTOR_JSON))
stopifnot(file.exists(ESR_BY_CELL_JSON))

if ("--dry-run" %in% commandArgs(trailingOnly = TRUE)) {
  cat("DRY-RUN OK\n")
  quit(status = 0)
}

OUT_DIR <- "figures-src"
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

make_figure <- function(name, panels, source_json) {
  cat(sprintf("Figure scaffold: %s (%d panel(s)); source: %s\n",
              name, length(panels), source_json))
}
