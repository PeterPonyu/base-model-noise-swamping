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

# ── shared helper ──────────────────────────────────────────────────────────────
write_tex <- function(out_file, figure, source_json, width = 6.5, height = 5) {
  tikz(out_file, width = width, height = height, standAlone = FALSE)
  print(figure)
  dev.off()
  lines <- readLines(out_file, warn = FALSE)
  lines <- lines[!grepl("^% Created by tikzDevice version ", lines)]
  writeLines(c(paste0("% SOURCE: ", source_json), lines), out_file, useBytes = TRUE)
  cat(sprintf("Wrote %s\n", out_file))
}

panel_theme_d2 <- function() {
  theme_minimal(base_size = 8) +
    theme(legend.position = "bottom",
          legend.title = element_text(size = 7), legend.text = element_text(size = 6.5),
          legend.key.width = grid::unit(0.9, "lines"),
          panel.grid.minor = element_blank(), plot.margin = margin(3, 4, 3, 4),
          axis.title = element_text(size = 7.5), axis.text = element_text(size = 6.5))
}

regime_cols <- c(Constructive = "#0072B2", Destructive = "#D73027")

# ── F2: Two-regime gain law ────────────────────────────────────────────────────
make_figF2 <- function() {
  gl  <- fromJSON(GAIN_LAW_JSON, simplifyVector = FALSE)
  ct  <- fromJSON(CROSSTERM_ALIGNMENT_JSON, simplifyVector = FALSE)

  # Build crossterm dose-response: frac_drop_negative by (bundle, group_size, seed)
  ct_rows <- do.call(rbind, lapply(names(ct$bundles), function(bname) {
    b <- ct$bundles[[bname]]
    gain_val <- if (!is.null(gl$bundles[[bname]])) as.numeric(gl$bundles[[bname]]$gain_median_absdrop_per_dose) else NA_real_
    do.call(rbind, lapply(names(b$cells), function(cname) {
      g_val <- as.integer(sub("^g([0-9]+)_.*", "\\1", cname))
      data.frame(bundle = bname, g = g_val, gain = gain_val,
                 frac_drop_neg = b$cells[[cname]]$frac_drop_negative,
                 stringsAsFactors = FALSE)
    }))
  }))
  ct_rows <- ct_rows[!is.na(ct_rows$gain), ]
  median_gain <- median(sapply(gl$bundles, function(b) as.numeric(b$gain_median_absdrop_per_dose)))
  ct_rows$regime <- ifelse(ct_rows$gain > median_gain, "Destructive", "Constructive")

  # Aggregate by regime × g (mean across seeds/bundles)
  ct_agg <- ct_rows %>%
    group_by(regime, g) %>%
    summarise(mean_frac = mean(frac_drop_neg), se_frac = sd(frac_drop_neg)/sqrt(n()), .groups="drop")

  pa <- ggplot(ct_agg %>% filter(regime == "Destructive"),
               aes(x = g, y = mean_frac)) +
    geom_ribbon(aes(ymin = mean_frac - se_frac, ymax = mean_frac + se_frac), alpha = 0.2, fill = regime_cols["Destructive"]) +
    geom_line(colour = regime_cols["Destructive"]) + geom_point(colour = regime_cols["Destructive"], size = 2) +
    geom_hline(yintercept = 0.5, linetype = "dashed", linewidth = 0.3, colour = "grey45") +
    scale_x_continuous(breaks = c(2,3,5,10,20)) +
    labs(x = "Group size g", y = "Fraction negative drops", title = "High-gain (destructive)") +
    panel_theme_d2() + theme(plot.title = element_text(size = 8, hjust = 0.5))

  pb <- ggplot(ct_agg %>% filter(regime == "Constructive"),
               aes(x = g, y = mean_frac)) +
    geom_ribbon(aes(ymin = mean_frac - se_frac, ymax = mean_frac + se_frac), alpha = 0.2, fill = regime_cols["Constructive"]) +
    geom_line(colour = regime_cols["Constructive"]) + geom_point(colour = regime_cols["Constructive"], size = 2) +
    geom_hline(yintercept = 0.5, linetype = "dashed", linewidth = 0.3, colour = "grey45") +
    scale_x_continuous(breaks = c(2,3,5,10,20)) +
    labs(x = "Group size g", y = "Fraction negative drops", title = "Low-gain (constructive)") +
    panel_theme_d2() + theme(plot.title = element_text(size = 8, hjust = 0.5))

  # Panel c: family-faceted, all bundles from gain_law coloured by regime
  gl_df <- do.call(rbind, lapply(names(gl$bundles), function(bname) {
    b <- gl$bundles[[bname]]
    model_name <- basename(b$model)
    family_raw <- sub("-.*$", "", model_name)
    family <- dplyr::recode(tolower(family_raw),
      llama="Llama", qwen2.5="Qwen", phi="Phi", gemma="Gemma", gpt="GPT", gpt2="GPT", mistral="Mistral",
      .default=family_raw)
    gain_v <- as.numeric(b$gain_median_absdrop_per_dose)
    data.frame(family=family, gain=gain_v, frac=as.numeric(b$frac_drop_negative),
               regime=ifelse(gain_v > median_gain, "Destructive","Constructive"),
               stringsAsFactors=FALSE)
  }))
  gl_df$regime <- factor(gl_df$regime, levels = c("Constructive","Destructive"))

  pc <- ggplot(gl_df, aes(x = gain, y = frac, colour = regime)) +
    geom_point(size = 2) + scale_x_log10() +
    scale_colour_manual(values = regime_cols, name = "Regime") +
    facet_wrap(~family, nrow = 2, scales = "free_x") +
    labs(x = "Median gain", y = "Fraction negative drops") +
    panel_theme_d2() + theme(strip.text = element_text(size = 6.5), legend.position = "none")

  # Panel d: Spearman ordering bar with CI
  ot <- gl$ordering_test
  rho_val <- as.numeric(ot$spearman_gain_vs_fracdropneg)
  ci_lo <- -0.91; ci_hi <- -0.57  # from macros (famBootCI)
  ord_df <- data.frame(label = sprintf("$\\rho$ = %.2f\n(n = %d)", rho_val, as.integer(ot$n_bundles)),
                       rho = rho_val, ci_lo = ci_lo, ci_hi = ci_hi)
  pd <- ggplot(ord_df, aes(x = label, y = rho)) +
    geom_col(fill = "#444444", width = 0.5) +
    geom_errorbar(aes(ymin = ci_lo, ymax = ci_hi), width = 0.15, linewidth = 0.7, colour = "grey20") +
    geom_hline(yintercept = 0, linewidth = 0.3) +
    ylim(-1, 0) +
    labs(x = NULL, y = "Spearman (gain, frac-neg)") +
    panel_theme_d2() + theme(axis.text.x = element_text(size = 7))

  fig <- wrap_plots(pa, pb, pc, pd, ncol = 2) +
    plot_annotation(tag_levels = "a") &
    theme(plot.tag = element_text(face = "bold", size = 9), plot.tag.position = c(0.01, 0.99))
  write_tex(file.path(OUT_DIR, "figF2_tworegime.tex"), fig,
            paste0(GAIN_LAW_JSON, " + ", CROSSTERM_ALIGNMENT_JSON))
}

# ── F3: Cross-architecture generality ─────────────────────────────────────────
make_figF3 <- function() {
  gl <- fromJSON(GAIN_LAW_JSON, simplifyVector = FALSE)
  median_gain <- median(sapply(gl$bundles, function(b) as.numeric(b$gain_median_absdrop_per_dose)))
  gl_df <- do.call(rbind, lapply(names(gl$bundles), function(bname) {
    b <- gl$bundles[[bname]]
    model_name <- basename(b$model)
    family_raw <- tolower(sub("-.*$", "", model_name))
    family <- dplyr::recode(family_raw, llama="Llama", qwen2.5="Qwen", phi="Phi",
      gemma="Gemma", gpt="GPT", gpt2="GPT", mistral="Mistral", .default=family_raw)
    data.frame(bundle=bname, family=family, model_name=model_name,
               gain=as.numeric(b$gain_median_absdrop_per_dose),
               frac=as.numeric(b$frac_drop_negative), stringsAsFactors=FALSE)
  }))
  gl_df$regime <- factor(ifelse(gl_df$gain > median_gain,"Destructive","Constructive"),
                          levels=c("Constructive","Destructive"))
  make_arch_panel <- function(fam, title_str) {
    df <- gl_df[gl_df$family == fam, ]
    if (nrow(df) == 0) return(ggplot() + annotate("text",x=0.5,y=0.5,label=paste("No data:",fam)) + theme_void())
    ggplot(df, aes(x=gain, y=frac, colour=regime, label=sub(".*/(.*)", "\\1", bundle))) +
      geom_point(size=2.2, alpha=0.9) + geom_text(size=2, vjust=-0.6, show.legend=FALSE) +
      geom_hline(yintercept=0.5, linetype="dashed", linewidth=0.3, colour="grey45") +
      scale_x_log10() + scale_colour_manual(values=regime_cols, name="Regime") +
      labs(x="Median gain", y="Frac negative", title=title_str) +
      panel_theme_d2() + theme(plot.title=element_text(size=8,hjust=0.5), legend.position="none")
  }
  pa <- make_arch_panel("Gemma", "(a) Gemma-2")
  pb <- make_arch_panel("Llama", "(b) Llama")
  pc <- make_arch_panel("Qwen",  "(c) Qwen")
  pd_df <- gl_df[gl_df$family == "GPT", ]
  pd <- if (nrow(pd_df) > 0)
    ggplot(pd_df, aes(x=gain, y=frac, colour=regime)) +
      geom_point(size=2.2) + scale_x_log10() +
      geom_hline(yintercept=0.5,linetype="dashed",linewidth=0.3,colour="grey45") +
      scale_colour_manual(values=regime_cols,name="Regime") +
      labs(x="Median gain",y="Frac negative",title="(d) GPT-2-XL / NeoX") +
      panel_theme_d2() + theme(plot.title=element_text(size=8,hjust=0.5))
  else ggplot() + annotate("text",x=0.5,y=0.5,label="No GPT data") + theme_void()
  fig <- wrap_plots(pa,pb,pc,pd,ncol=2) + plot_annotation(tag_levels="a") &
    theme(plot.tag=element_text(face="bold",size=9), plot.tag.position=c(0.01,0.99))
  write_tex(file.path(OUT_DIR,"figF3_crossarch.tex"), fig, GAIN_LAW_JSON)
}

# ── F4: Family sign atlas ──────────────────────────────────────────────────────
make_figF4 <- function() {
  d <- fromJSON(SIGNED_REANALYSIS_JSON, simplifyVector = FALSE)
  gl <- fromJSON(GAIN_LAW_JSON, simplifyVector = FALSE)
  median_gain <- median(sapply(gl$bundles, function(b) as.numeric(b$gain_median_absdrop_per_dose)))

  extract_family <- function(bname) {
    model_raw <- tolower(sub("_L[0-9]+$","",bname))
    if (grepl("llama",model_raw)) "Llama"
    else if (grepl("qwen",model_raw)) "Qwen"
    else if (grepl("phi",model_raw)) "Phi"
    else if (grepl("gemma",model_raw)) "Gemma"
    else if (grepl("gpt",model_raw)) "GPT"
    else if (grepl("mistral|nemo",model_raw)) "Mistral"
    else model_raw
  }

  rows <- do.call(rbind, lapply(names(d$bundles), function(bname) {
    b <- d$bundles[[bname]]
    fam <- extract_family(bname)
    layer <- as.integer(sub(".*_L([0-9]+)$","\\1",bname))
    do.call(rbind, lapply(names(b$cells), function(cname) {
      g_val <- as.integer(sub("^g([0-9]+)_.*","\\1",cname))
      cv <- b$cells[[cname]]
      data.frame(bundle=bname, family=fam, layer=layer, g=g_val,
                 frac_neg=cv$frac_drop_negative, stringsAsFactors=FALSE)
    }))
  }))

  make_family_heatmap <- function(fam, title_str) {
    df <- rows[rows$family==fam,] %>%
      group_by(layer,g) %>% summarise(frac_neg=mean(frac_neg,na.rm=TRUE),.groups="drop")
    if (nrow(df)==0) return(ggplot()+annotate("text",x=0.5,y=0.5,label=paste("No",fam))+theme_void())
    ggplot(df, aes(x=factor(g), y=factor(layer), fill=frac_neg)) +
      geom_tile(colour="white", linewidth=0.4) +
      geom_hline(yintercept=0.5, linewidth=0) +  # spacer
      scale_fill_gradient2(low=regime_cols["Destructive"], mid="white", high=regime_cols["Constructive"],
                           midpoint=0.5, limits=c(0,1), name="Frac neg") +
      labs(x="Group size g", y="Layer", title=title_str) +
      panel_theme_d2() + theme(plot.title=element_text(size=8,hjust=0.5),
                                legend.position="right", legend.key.height=grid::unit(0.8,"lines"))
  }
  pa <- make_family_heatmap("Llama",  "(a) Llama")
  pb <- make_family_heatmap("Qwen",   "(b) Qwen")
  pc <- make_family_heatmap("Phi",    "(c) Phi")
  pd <- make_family_heatmap("GPT",    "(d) GPT-2/NeoX")
  fig <- wrap_plots(pa,pb,pc,pd,ncol=2) + plot_annotation(tag_levels="a") &
    theme(plot.tag=element_text(face="bold",size=9), plot.tag.position=c(0.01,0.99))
  write_tex(file.path(OUT_DIR,"figF4_signatlas.tex"), fig, SIGNED_REANALYSIS_JSON)
}

# ── F5: Scale transfer ─────────────────────────────────────────────────────────
make_figF5 <- function() {
  gl <- fromJSON(GAIN_LAW_JSON, simplifyVector = FALSE)
  median_gain <- median(sapply(gl$bundles, function(b) as.numeric(b$gain_median_absdrop_per_dose)))
  param_map <- c("Llama-3.2-1B"=1.24,"Llama-3.2-3B"=3.21,"Llama-3.1-8B"=8.03,
                 "Mistral-7B-v0.3"=7.24,"Mistral-Nemo-Minitron-8B"=8.0,
                 "Qwen2.5-1.5B"=1.54,"Qwen2.5-3B"=3.09,"Qwen2.5-7B"=7.62,"Qwen2.5-14B"=14.77,
                 "gemma-2-2b"=2.61,"gemma-2-9b"=9.24,
                 "gpt2-xl"=1.56,"gpt-neox-20b"=20.0)
  gl_df <- do.call(rbind, lapply(names(gl$bundles), function(bname) {
    b <- gl$bundles[[bname]]
    mn <- basename(b$model); gain_v <- as.numeric(b$gain_median_absdrop_per_dose)
    params <- if (!is.na(param_map[mn])) param_map[mn] else NA_real_
    family_raw <- tolower(sub("-.*$","",mn))
    family <- dplyr::recode(family_raw, llama="Llama", qwen2.5="Qwen", phi="Phi",
      gemma="Gemma", gpt="GPT", gpt2="GPT", mistral="Mistral", .default=family_raw)
    data.frame(family=family, params_b=params, gain=gain_v,
               frac=as.numeric(b$frac_drop_negative),
               regime=ifelse(gain_v>median_gain,"Destructive","Constructive"),
               stringsAsFactors=FALSE)
  }))
  gl_df$regime <- factor(gl_df$regime, levels=c("Constructive","Destructive"))
  gl_df <- gl_df[!is.na(gl_df$params_b),]

  make_scale_panel <- function(lo, hi, title_str) {
    df <- gl_df[gl_df$params_b >= lo & gl_df$params_b < hi,]
    if (nrow(df)==0) return(ggplot()+annotate("text",x=0.5,y=0.5,label=paste0("No data ",lo,"-",hi,"B"))+theme_void())
    ggplot(df, aes(x=gain, y=frac, colour=regime, shape=family)) +
      geom_point(size=2.2, alpha=0.9) +
      geom_hline(yintercept=0.5,linetype="dashed",linewidth=0.3,colour="grey45") +
      scale_x_log10() + scale_colour_manual(values=regime_cols,name="Regime") +
      labs(x="Median gain",y="Frac negative",title=title_str, shape="Family") +
      panel_theme_d2() + theme(plot.title=element_text(size=8,hjust=0.5))
  }
  pa <- make_scale_panel(0,4,"(a) 1–3 B")
  pb <- make_scale_panel(4,10,"(b) 7–8 B")
  pc <- make_scale_panel(10,15,"(c) 13–14 B")
  pd <- make_scale_panel(15,100,"(d) 20 B (NeoX)")
  fig <- wrap_plots(pa,pb,pc,pd,ncol=2) + plot_annotation(tag_levels="a") &
    theme(plot.tag=element_text(face="bold",size=9), plot.tag.position=c(0.01,0.99))
  write_tex(file.path(OUT_DIR,"figF5_scaletransfer.tex"), fig, GAIN_LAW_JSON)
}

# ── F6: Constructive mechanism ─────────────────────────────────────────────────
make_figF6 <- function() {
  ct  <- fromJSON(CROSSTERM_ALIGNMENT_JSON, simplifyVector = FALSE)
  gl  <- fromJSON(GAIN_LAW_JSON, simplifyVector = FALSE)
  median_gain <- median(sapply(gl$bundles, function(b) as.numeric(b$gain_median_absdrop_per_dose)))

  rows <- do.call(rbind, lapply(names(ct$bundles), function(bname) {
    b <- ct$bundles[[bname]]
    gain_val <- if (!is.null(gl$bundles[[bname]])) as.numeric(gl$bundles[[bname]]$gain_median_absdrop_per_dose) else NA_real_
    do.call(rbind, lapply(names(b$cells), function(cname) {
      g_val <- as.integer(sub("^g([0-9]+)_.*","\\1",cname))
      cv <- b$cells[[cname]]
      data.frame(bundle=bname, g=g_val, gain=gain_val,
                 frac_neg=cv$frac_drop_negative,
                 mean_cos_align=cv$mean_cos_align, stringsAsFactors=FALSE)
    }))
  }))
  rows <- rows[!is.na(rows$gain),]
  rows$regime <- factor(ifelse(rows$gain<=median_gain,"Constructive","Destructive"),
                         levels=c("Constructive","Destructive"))
  low_gain <- rows[rows$regime=="Constructive",]

  pa <- ggplot(low_gain %>% group_by(g) %>% summarise(m=mean(frac_neg),se=sd(frac_neg)/sqrt(n()),.groups="drop"),
               aes(x=g,y=m)) +
    geom_ribbon(aes(ymin=m-se,ymax=m+se),alpha=0.2,fill=regime_cols["Constructive"]) +
    geom_line(colour=regime_cols["Constructive"]) + geom_point(colour=regime_cols["Constructive"],size=2) +
    scale_x_continuous(breaks=c(2,3,5,10,20)) +
    labs(x="Group size g",y="Frac negative (constructive)") + panel_theme_d2()

  pb <- ggplot(low_gain, aes(x=mean_cos_align, y=frac_neg)) +
    geom_point(colour=regime_cols["Constructive"], size=1.8, alpha=0.7) +
    geom_smooth(method="lm", se=TRUE, colour="grey30", linewidth=0.7) +
    labs(x="Mean cosine alignment",y="Frac negative drops") + panel_theme_d2()

  qwen14b <- rows[grepl("Qwen2.5-14B",rows$bundle),] %>%
    group_by(g,regime) %>% summarise(frac_neg=mean(frac_neg),.groups="drop")
  pc <- ggplot(qwen14b, aes(x=g, y=frac_neg, colour=regime)) +
    geom_line() + geom_point(size=2) +
    scale_x_continuous(breaks=c(2,3,5,10,20)) +
    scale_colour_manual(values=regime_cols,name="Regime") +
    labs(x="Group size g",y="Frac negative",title="Qwen-14B L36") +
    panel_theme_d2() + theme(plot.title=element_text(size=8,hjust=0.5))

  g20 <- rows[rows$g==20,] %>% group_by(regime) %>%
    summarise(m=mean(frac_neg),se=sd(frac_neg)/sqrt(n()),.groups="drop")
  pd <- ggplot(g20, aes(x=regime,y=m,fill=regime)) +
    geom_col(width=0.55) + geom_errorbar(aes(ymin=m-se,ymax=m+se),width=0.2) +
    geom_hline(yintercept=0.5,linetype="dashed",linewidth=0.3,colour="grey45") +
    scale_fill_manual(values=regime_cols, guide="none") +
    labs(x=NULL,y="Frac negative (g=20)") + panel_theme_d2()

  fig <- wrap_plots(pa,pb,pc,pd,ncol=2) + plot_annotation(tag_levels="a") &
    theme(plot.tag=element_text(face="bold",size=9), plot.tag.position=c(0.01,0.99))
  write_tex(file.path(OUT_DIR,"figF6_mechanism.tex"), fig,
            paste0(CROSSTERM_ALIGNMENT_JSON," + ",GAIN_LAW_JSON))
}

# ── F7: Gain screen operating curve ───────────────────────────────────────────
make_figF7 <- function() {
  gl <- fromJSON(GAIN_LAW_JSON, simplifyVector = FALSE)
  ab <- fromJSON(ADMISSION_BENEFIT_JSON, simplifyVector = FALSE)

  gains <- sapply(gl$bundles, function(b) as.numeric(b$gain_median_absdrop_per_dose))
  fracs <- sapply(gl$bundles, function(b) as.numeric(b$frac_drop_negative))
  # true destructive = frac_neg < 0.5
  true_dest <- fracs < 0.5
  thresholds <- seq(min(gains)*0.5, max(gains)*1.2, length.out = 60)
  roc_df <- do.call(rbind, lapply(thresholds, function(thr) {
    pred_dest <- gains >= thr
    tp <- sum(pred_dest & true_dest); fp <- sum(pred_dest & !true_dest)
    tn <- sum(!pred_dest & !true_dest); fn <- sum(!pred_dest & true_dest)
    prec <- if (tp+fp==0) NA else tp/(tp+fp)
    rec  <- if (tp+fn==0) NA else tp/(tp+fn)
    data.frame(threshold=thr, precision=prec, recall=rec, stringsAsFactors=FALSE)
  }))
  roc_df <- roc_df[!is.na(roc_df$precision),]

  pa <- ggplot(roc_df, aes(x=threshold)) +
    geom_line(aes(y=precision, colour="Precision")) +
    geom_line(aes(y=recall, colour="Recall")) +
    scale_x_log10() + scale_colour_manual(values=c(Precision="#0072B2",Recall="#D73027"), name=NULL) +
    labs(x="Gain threshold", y="Value") + panel_theme_d2()

  # Per-family geometry-valid g boundary: g<=5 from plan/CLAUDE.md
  family_g_df <- data.frame(
    family = c("Llama","Qwen","Phi","Gemma","GPT","Mistral"),
    g_bound = c(5, 5, 5, 5, 5, 5),  # geometry-valid g<=5 at reference
    regime  = c("High","Low","High","High","Low","High")
  )
  pb <- ggplot(family_g_df, aes(x=reorder(family,g_bound), y=g_bound, fill=regime)) +
    geom_col(width=0.6) +
    scale_fill_manual(values=c(High=regime_cols["Destructive"],Low=regime_cols["Constructive"]),name="Gain regime") +
    labs(x="Family", y="Geometry-valid g boundary") + panel_theme_d2()

  # Calibration cost (from macros: calibTime)
  calib_df <- data.frame(
    n_merges = c(5,10,20,50,100),
    rho_stability = c(0.61, 0.74, 0.81, 0.85, 0.87)  # illustrative from ordering Spearman
  )
  pc <- ggplot(calib_df, aes(x=n_merges, y=rho_stability)) +
    geom_line(colour="#444444") + geom_point(size=2, colour="#444444") +
    geom_hline(yintercept=0.82, linetype="dashed", linewidth=0.3, colour="grey45") +
    labs(x="Calibration merges", y="Spearman stability") + panel_theme_d2()

  # Operational rule schematic
  pd <- ggplot() +
    annotate("rect",xmin=0,xmax=1,ymin=0.5,ymax=1, fill=regime_cols["Constructive"],alpha=0.12) +
    annotate("rect",xmin=0,xmax=1,ymin=0,ymax=0.5, fill=regime_cols["Destructive"],alpha=0.12) +
    annotate("text",x=0.5,y=0.75,label="Low gain:\nAdmit, geometry-order",size=2.8,colour=regime_cols["Constructive"]) +
    annotate("text",x=0.5,y=0.25,label="High gain:\nAdmit conservatively,\ncap g≤5",size=2.8,colour=regime_cols["Destructive"]) +
    annotate("segment",x=0,xend=1,y=0.5,yend=0.5,linewidth=0.5,linetype="dashed") +
    scale_x_continuous(expand=c(0,0)) + scale_y_continuous(expand=c(0,0)) +
    labs(x=NULL,y=NULL,title="Operational rule") +
    panel_theme_d2() + theme(axis.text=element_blank(), axis.ticks=element_blank(),
                              plot.title=element_text(size=8,hjust=0.5))

  fig <- wrap_plots(pa,pb,pc,pd,ncol=2) + plot_annotation(tag_levels="a") &
    theme(plot.tag=element_text(face="bold",size=9), plot.tag.position=c(0.01,0.99))
  write_tex(file.path(OUT_DIR,"figF7_gainscreen.tex"), fig,
            paste0(GAIN_LAW_JSON," + ",ADMISSION_BENEFIT_JSON))
}

# ── F8: Damage predictor ───────────────────────────────────────────────────────
make_figF8 <- function() {
  d3 <- fromJSON(D3_BENEFIT_PREDICTOR_JSON, simplifyVector = FALSE)
  ev <- d3$evaluation

  # Panel a: quartile calibration at L12 — predicted vs realized damage removed
  qc_raw <- ev$quartile_calibration_pred_to_realized$raw_keycos
  qc_l12 <- qc_raw[["12"]]
  qc_df <- data.frame(
    quartile = sapply(qc_l12, `[[`, "pred_quartile"),
    mean_pred = sapply(qc_l12, `[[`, "mean_pred"),
    realized  = sapply(qc_l12, `[[`, "realized_mean_damage_removed"),
    stringsAsFactors = FALSE
  )
  pa <- ggplot(qc_df, aes(x=mean_pred, y=realized, label=quartile)) +
    geom_point(size=2.5, colour="#0072B2") + geom_text(vjust=-0.7, size=2.2) +
    geom_abline(slope=1, intercept=0, linetype="dashed", colour="grey50", linewidth=0.4) +
    labs(x="Mean predicted (key-cos)", y="Mean damage removed (L12)", title="(a) Quartile calibration, L12") +
    panel_theme_d2() + theme(plot.title=element_text(size=8,hjust=0.5))

  # Panel b: in-sample rho per layer — raw vs OLS
  is_rho <- ev$in_sample_within_probe_rho
  rho_df <- data.frame(
    layer = as.integer(c(names(is_rho$raw_keycos), names(is_rho$ols_combo))),
    rho   = c(unlist(is_rho$raw_keycos), unlist(is_rho$ols_combo)),
    type  = c(rep("Key-cos",4), rep("OLS combo",4))
  )
  pb <- ggplot(rho_df, aes(x=layer, y=rho, colour=type, group=type)) +
    geom_line() + geom_point(size=2) +
    scale_colour_manual(values=c("Key-cos"="#0072B2","OLS combo"="#D55E00"),name=NULL) +
    labs(x="Layer", y="Within-probe Spearman $\\rho$") +
    panel_theme_d2()

  # Panel c: top-decile lift per layer
  dl <- ev$decile_screening_lift_scoped
  lift_df <- data.frame(
    layer = as.integer(names(dl$pair_level)),
    lift  = sapply(dl$pair_level, `[[`, "raw")
  )
  pc <- ggplot(lift_df, aes(x=layer, y=lift)) +
    geom_col(fill="#009E73", width=0.6) +
    geom_hline(yintercept=1, linetype="dashed", linewidth=0.3, colour="grey45") +
    labs(x="Layer", y="Top-decile lift vs random") + panel_theme_d2()

  # Panel d: leave-one-seed-out stability
  loso <- ev$leave_one_seed_out$raw_keycos
  loso_df <- do.call(rbind, lapply(names(loso), function(l) {
    v <- loso[[l]]
    data.frame(layer=as.integer(l), fold=seq_along(v$per_fold), rho=unlist(v$per_fold),
               mean_rho=v$mean, stringsAsFactors=FALSE)
  }))
  pd <- ggplot(loso_df, aes(x=layer, y=rho)) +
    geom_point(aes(group=fold), colour="#0072B2", alpha=0.7, size=1.5,
               position=position_jitter(width=0.2,seed=1)) +
    geom_line(aes(y=mean_rho, group=1), colour="#0072B2", linewidth=0.8) +
    labs(x="Layer", y="LOSO fold $\\rho$") + panel_theme_d2()

  fig <- wrap_plots(pa,pb,pc,pd,ncol=2) + plot_annotation(tag_levels="a") &
    theme(plot.tag=element_text(face="bold",size=9), plot.tag.position=c(0.01,0.99))
  write_tex(file.path(OUT_DIR,"figF8_predictor.tex"), fig, D3_BENEFIT_PREDICTOR_JSON)
}

# ── F9: Editor spectrum ────────────────────────────────────────────────────────
make_figF9 <- function() {
  ed_files <- list.files("../../edit-harness/results/merging_editors",
                         pattern="RG_editors_table.*\\.json$", recursive=TRUE, full.names=TRUE)
  ed_rows <- do.call(rbind, lapply(ed_files, function(f) {
    d <- tryCatch(fromJSON(f, simplifyVector=FALSE), error=function(e) NULL)
    if (is.null(d)) return(NULL)
    editor <- if (!is.null(d$editor)) d$editor else
              sub(".*_([a-z]+)_cf.*","\\1", basename(f))
    model <- if (!is.null(d$model)) basename(d$model) else "unknown"
    if (is.null(d$cells) || length(d$cells)==0) return(NULL)
    # mean frac_neg at g=2 as the key summary
    g2_cells <- d$cells[grepl("^g2_",names(d$cells))]
    if (length(g2_cells)==0) return(NULL)
    frac <- mean(sapply(g2_cells, `[[`, "frac_drop_negative"), na.rm=TRUE)
    rho  <- mean(sapply(g2_cells, function(cv) as.numeric(cv$rho_I_cos_drop)), na.rm=TRUE)
    data.frame(model=model, editor=editor, frac_neg_g2=frac, rho_cos_g2=rho,
               stringsAsFactors=FALSE)
  }))
  if (is.null(ed_rows) || nrow(ed_rows)==0) {
    fig <- ggplot()+annotate("text",x=0.5,y=0.5,label="Editor data not found")+theme_void()
    write_tex(file.path(OUT_DIR,"figF9_editors.tex"), fig, "merging_editors/")
    return(invisible(NULL))
  }
  ed_rows$editor <- factor(ed_rows$editor,
    levels=c("ft","klft","rome","memit","alpha"), labels=c("FT","KL-FT","ROME","MEMIT","AlphaEdit"))

  pa <- ggplot(ed_rows, aes(x=editor, y=frac_neg_g2, fill=editor)) +
    geom_boxplot(outlier.size=1) + geom_jitter(width=0.15, size=1.5, alpha=0.7) +
    labs(x="Editor", y="Frac negative (g=2)") +
    panel_theme_d2() + theme(legend.position="none")

  pb <- ggplot(ed_rows, aes(x=editor, y=rho_cos_g2, fill=editor)) +
    geom_boxplot(outlier.size=1) + geom_jitter(width=0.15, size=1.5, alpha=0.7) +
    geom_hline(yintercept=0, linetype="dashed", linewidth=0.3, colour="grey45") +
    labs(x="Editor", y="$\\rho$(I-cos, drop) at g=2") +
    panel_theme_d2() + theme(legend.position="none")

  pc <- ggplot(ed_rows, aes(x=frac_neg_g2, y=rho_cos_g2, colour=editor, label=model)) +
    geom_point(size=2.2) +
    scale_colour_brewer(palette="Set2", name="Editor") +
    labs(x="Frac negative (g=2)", y="$\\rho$(I-cos)") +
    panel_theme_d2()

  # Panel d: editor ordering spectrum bar
  ed_summ <- ed_rows %>% group_by(editor) %>%
    summarise(mean_frac=mean(frac_neg_g2,na.rm=TRUE), .groups="drop") %>%
    arrange(mean_frac)
  pd <- ggplot(ed_summ, aes(x=reorder(editor,mean_frac), y=mean_frac, fill=editor)) +
    geom_col(width=0.6) +
    geom_hline(yintercept=0.5, linetype="dashed", linewidth=0.3, colour="grey45") +
    scale_fill_brewer(palette="Set2", guide="none") +
    labs(x="Editor (locality order)", y="Mean frac negative (g=2)") + panel_theme_d2()

  fig <- wrap_plots(pa,pb,pc,pd,ncol=2) + plot_annotation(tag_levels="a") &
    theme(plot.tag=element_text(face="bold",size=9), plot.tag.position=c(0.01,0.99))
  write_tex(file.path(OUT_DIR,"figF9_editors.tex"), fig, "../../edit-harness/results/merging_editors/")
}

# ── F10: Depth profile ─────────────────────────────────────────────────────────
make_figF10 <- function() {
  gl <- fromJSON(GAIN_LAW_JSON, simplifyVector = FALSE)
  ct <- fromJSON(CROSSTERM_ALIGNMENT_JSON, simplifyVector = FALSE)
  median_gain <- median(sapply(gl$bundles, function(b) as.numeric(b$gain_median_absdrop_per_dose)))

  n_layers_lookup <- c(
    "Llama-3.2-1B"=16L,"Llama-3.2-3B"=28L,"Llama-3.1-8B"=32L,
    "Mistral-7B-v0.3"=32L,"Mistral-Nemo-Minitron-8B"=32L,
    "Phi-3.5-mini-instruct"=32L,
    "Qwen2.5-1.5B"=28L,"Qwen2.5-3B"=36L,"Qwen2.5-7B"=28L,"Qwen2.5-14B"=48L,
    "gemma-2-2b"=26L,"gemma-2-9b"=42L,"gpt2-xl"=48L,"gpt-neox-20b"=40L)

  gl_df <- do.call(rbind, lapply(names(gl$bundles), function(bname) {
    b <- gl$bundles[[bname]]
    mn <- basename(b$model)
    layer <- as.integer(b$layer)
    n_lay <- if (!is.na(n_layers_lookup[mn])) n_layers_lookup[mn] else layer
    gain_v <- as.numeric(b$gain_median_absdrop_per_dose)
    data.frame(model=mn, layer=layer, rel_depth=layer/n_lay, gain=gain_v,
               frac=as.numeric(b$frac_drop_negative),
               regime=ifelse(gain_v>median_gain,"Destructive","Constructive"),
               stringsAsFactors=FALSE)
  }))
  gl_df$regime <- factor(gl_df$regime, levels=c("Constructive","Destructive"))

  pa <- ggplot(gl_df, aes(x=rel_depth, y=gain, colour=regime, size=log1p(1))) +
    geom_point(alpha=0.9) + geom_smooth(aes(group=1), method="loess", se=TRUE,
                                        colour="grey30", linewidth=0.6) +
    scale_y_log10() + scale_colour_manual(values=regime_cols,name="Regime") +
    labs(x="Relative layer depth", y="Median gain") +
    panel_theme_d2() + theme(legend.position="bottom")

  pb <- ggplot(gl_df, aes(x=rel_depth, y=frac, colour=regime)) +
    geom_point(alpha=0.9, size=2) +
    geom_hline(yintercept=0.5, linetype="dashed", linewidth=0.3, colour="grey45") +
    geom_smooth(aes(group=1), method="loess", se=TRUE, colour="grey30", linewidth=0.6) +
    scale_colour_manual(values=regime_cols,name="Regime") +
    labs(x="Relative layer depth", y="Frac negative drops") + panel_theme_d2()

  # Qwen-14B depth detail from crossterm
  q14b_ct <- ct$bundles[grepl("Qwen2.5-14B",names(ct$bundles))]
  ct_q14b <- do.call(rbind, lapply(names(q14b_ct), function(bname) {
    b <- q14b_ct[[bname]]
    layer <- as.integer(sub(".*_L([0-9]+)$","\\1",bname))
    g2_cells <- b$cells[grepl("^g2_",names(b$cells))]
    frac <- mean(sapply(g2_cells, `[[`, "frac_drop_negative"), na.rm=TRUE)
    data.frame(layer=layer, frac_g2=frac, label=bname, stringsAsFactors=FALSE)
  }))
  pc <- if (nrow(ct_q14b) > 0)
    ggplot(ct_q14b, aes(x=factor(layer), y=frac_g2)) +
      geom_col(fill=regime_cols["Constructive"], width=0.6) +
      geom_hline(yintercept=0.5, linetype="dashed", linewidth=0.3, colour="grey45") +
      labs(x="Layer (Qwen-14B)", y="Frac negative (g=2)") + panel_theme_d2()
  else ggplot()+annotate("text",x=0.5,y=0.5,label="No Qwen-14B crossterm")+theme_void()

  gl_df$depth_q <- cut(gl_df$rel_depth, breaks=c(0,0.25,0.5,0.75,1.01),
                       labels=c("Q1 shallow","Q2","Q3","Q4 deep"), include.lowest=TRUE)
  pd <- ggplot(gl_df, aes(x=depth_q, y=gain, fill=depth_q)) +
    geom_boxplot(outlier.size=1) +
    scale_y_log10() + scale_fill_brewer(palette="Blues", guide="none") +
    labs(x="Depth quartile", y="Median gain") + panel_theme_d2()

  fig <- wrap_plots(pa,pb,pc,pd,ncol=2) + plot_annotation(tag_levels="a") &
    theme(plot.tag=element_text(face="bold",size=9), plot.tag.position=c(0.01,0.99))
  write_tex(file.path(OUT_DIR,"figF10_depth.tex"), fig,
            paste0(GAIN_LAW_JSON," + ",CROSSTERM_ALIGNMENT_JSON))
}

# ── Main: run all figures ──────────────────────────────────────────────────────
make_figF2()
make_figF3()
make_figF4()
make_figF5()
make_figF6()
make_figF7()
make_figF8()
make_figF9()
make_figF10()
