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

make_figF1 <- function() {
  gain_law <- fromJSON(GAIN_LAW_JSON, simplifyVector = FALSE)
  bundle_names <- names(gain_law$bundles)

  rows <- lapply(bundle_names, function(bundle_name) {
    bundle <- gain_law$bundles[[bundle_name]]
    model_name <- basename(bundle$model)
    family_raw <- sub("-.*$", "", model_name)
    family <- dplyr::recode(
      tolower(family_raw),
      llama = "Llama",
      qwen2.5 = "Qwen",
      phi = "Phi",
      gemma = "Gemma",
      gpt = "GPT",
      gpt2 = "GPT",
      mistral = "Mistral",
      .default = family_raw
    )
    layer_from_key <- as.integer(sub(".*_L([0-9]+)_RG$", "\\1", bundle_name))
    # Hardcoded model depths — robust to future family additions
    n_layers_lookup <- c(
      "Llama-3.2-1B" = 16L, "Llama-3.2-3B" = 28L, "Llama-3.1-8B" = 32L,
      "Mistral-7B-v0.3" = 32L, "Mistral-Nemo-Minitron-8B" = 32L,
      "Phi-3.5-mini-instruct" = 32L,
      "Qwen2.5-1.5B" = 28L, "Qwen2.5-3B" = 36L, "Qwen2.5-7B" = 28L, "Qwen2.5-14B" = 48L,
      "gemma-2-2b" = 26L, "gemma-2-9b" = 42L,
      "gpt2-xl" = 48L, "gpt-neox-20b" = 40L
    )
    n_layers_for_model <- n_layers_lookup[model_name]
    if (is.na(n_layers_for_model)) n_layers_for_model <- layer_from_key  # fallback: rel_depth=1
    rel_depth <- layer_from_key / n_layers_for_model

    data.frame(
      bundle = bundle_name,
      family = family,
      layer = layer_from_key,
      rel_depth = rel_depth,
      n_obs = as.integer(bundle$n_obs),
      gain = as.numeric(bundle$gain_median_absdrop_per_dose),
      frac_drop_negative = as.numeric(bundle$frac_drop_negative),
      regime = if (as.numeric(bundle$frac_drop_negative) > 0.5) {
        "Constructive"
      } else {
        "Destructive"
      },
      stringsAsFactors = FALSE
    )
  })
  gain_df <- bind_rows(rows)

  stopifnot(
    nrow(gain_df) == 22L,
    all(gain_df$family %in% c("Llama", "Qwen", "Phi", "Gemma", "GPT", "Mistral")),
    all(is.finite(gain_df$layer)),
    all(is.finite(gain_df$gain)),
    all(gain_df$gain >= 0),
    all(is.finite(gain_df$frac_drop_negative)),
    all(gain_df$frac_drop_negative >= 0 & gain_df$frac_drop_negative <= 1)
  )
  gain_df$regime <- factor(
    gain_df$regime,
    levels = c("Constructive", "Destructive")
  )

  family_colours <- c(
    Llama = "#0072B2",
    Qwen = "#D55E00",
    Phi = "#009E73",
    Gemma = "#CC79A7",
    GPT = "#E69F00",
    Mistral = "#6A3D9A"
  )
  regime_colours <- c(
    Constructive = "#0072B2",
    Destructive = "#D73027"
  )
  panel_theme <- theme_minimal(base_size = 8) +
    theme(
      legend.position = "bottom",
      legend.title = element_text(size = 7),
      legend.text = element_text(size = 6.5),
      legend.key.width = grid::unit(0.9, "lines"),
      panel.grid.minor = element_blank(),
      plot.margin = margin(3, 4, 3, 4),
      axis.title = element_text(size = 7.5),
      axis.text = element_text(size = 6.5)
    )

  pa <- ggplot(gain_df, aes(x = gain, y = frac_drop_negative, colour = family)) +
    geom_hline(yintercept = 0.5, linewidth = 0.3, linetype = "dashed", colour = "grey45") +
    geom_point(size = 2.1, alpha = 0.9) +
    scale_x_log10() +
    scale_colour_manual(values = family_colours, name = "Family") +
    labs(
      x = "Median gain (absolute drop / dose)",
      y = "Fraction of negative drops"
    ) +
    guides(colour = guide_legend(nrow = 1, byrow = TRUE)) +
    panel_theme

  pb <- ggplot(gain_df, aes(x = gain, y = frac_drop_negative, colour = regime)) +
    geom_hline(yintercept = 0.5, linewidth = 0.3, linetype = "dashed", colour = "grey45") +
    geom_point(size = 2.2, alpha = 0.9) +
    scale_x_log10() +
    scale_colour_manual(values = regime_colours, name = "Observed regime") +
    labs(
      x = "Median gain (absolute drop / dose)",
      y = "Fraction of negative drops"
    ) +
    guides(colour = guide_legend(nrow = 1, byrow = TRUE)) +
    panel_theme

  pc <- ggplot(gain_df, aes(x = rel_depth, y = gain, colour = regime, size = log1p(n_obs))) +
    geom_point(alpha = 0.9) +
    scale_y_log10() +
    scale_size_continuous(range = c(1, 4), guide = "none") +
    scale_colour_manual(values = regime_colours, name = "Observed regime") +
    labs(
      x = "Relative layer depth",
      y = "Median gain (absolute drop / dose)"
    ) +
    guides(colour = guide_legend(nrow = 1, byrow = TRUE)) +
    panel_theme +
    theme(axis.title.y = element_text(margin = margin(r = 8)))

  gain_threshold <- 8
  pd <- ggplot() +
    annotate(
      "rect", xmin = 0.1, xmax = gain_threshold,
      ymin = 0, ymax = 1, fill = regime_colours[["Constructive"]], alpha = 0.13
    ) +
    annotate(
      "rect", xmin = gain_threshold, xmax = 70,
      ymin = 0, ymax = 1, fill = regime_colours[["Destructive"]], alpha = 0.13
    ) +
    annotate(
      "segment", x = gain_threshold, xend = gain_threshold,
      y = 0.08, yend = 0.92, linewidth = 0.5, linetype = "dashed"
    ) +
    annotate(
      "text", x = gain_threshold, y = 0.94,
      label = "Gain screen = 8", hjust = 0.5, vjust = 0, size = 2.5
    ) +
    annotate(
      "text", x = 0.75, y = 0.56,
      label = "Low gain", colour = regime_colours[["Constructive"]], size = 3
    ) +
    annotate(
      "text", x = 0.75, y = 0.40,
      label = "constructive enriched", colour = regime_colours[["Constructive"]], size = 2.4
    ) +
    annotate(
      "text", x = 24, y = 0.56,
      label = "High gain", colour = regime_colours[["Destructive"]], size = 3
    ) +
    annotate(
      "text", x = 24, y = 0.40,
      label = "destructive dominant", colour = regime_colours[["Destructive"]], size = 2.4
    ) +
    scale_x_log10(limits = c(0.1, 70)) +
    coord_cartesian(ylim = c(0, 1), clip = "off") +
    labs(x = "Median gain (absolute drop / dose)", y = NULL) +
    panel_theme +
    theme(
      axis.text.y = element_blank(),
      axis.ticks.y = element_blank(),
      panel.grid = element_blank(),
      legend.position = "none"
    )

  figure <- wrap_plots(pa, pb, pc, pd, ncol = 2) +
    plot_annotation(tag_levels = "a") &
    theme(
      plot.tag = element_text(face = "bold", size = 9),
      plot.tag.position = c(0.01, 0.99)
    )

  out_file <- file.path(OUT_DIR, "figF1_falsification.tex")
  tikz(out_file, width = 6.5, height = 5, standAlone = FALSE)
  print(figure)
  dev.off()

  source_header <- paste0("% SOURCE: ", GAIN_LAW_JSON)
  tex_lines <- readLines(out_file, warn = FALSE)
  tex_lines <- tex_lines[!grepl("^% Created by tikzDevice version ", tex_lines)]
  writeLines(c(source_header, tex_lines), out_file, useBytes = TRUE)
  cat(sprintf("Wrote %s (%d bundles)\n", out_file, nrow(gain_df)))
}

make_figF1()
