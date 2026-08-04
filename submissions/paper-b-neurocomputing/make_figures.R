# Paper B figure package
suppressPackageStartupMessages({library(jsonlite); library(ggplot2); library(tikzDevice); library(patchwork)})
HARNESS <- "/home/zeyufu/Desktop/idea-feasibility-analysis/edit-harness"
SURV <- file.path(HARNESS, "results/quant_survival/aggregate/quant_survival_repair_v1.json")
GATE <- file.path(HARNESS, "results/quant_survival/aggregate/gate_readout.json")
# SOURCE: quant_survival_repair_v1.json fields module_provenance.version and module_provenance.n_boot.
s <- fromJSON(SURV, simplifyVector=FALSE)
stopifnot(s$module_provenance$version == "1.2.1", s$module_provenance$n_boot == 500)
g <- fromJSON(GATE, simplifyVector=FALSE)
C_BLUE <- "#0072B2"; C_ORANGE <- "#D55E00"; C_TEAL <- "#009E73"; C_GREY <- "#6A3D9A"
theme_p <- theme_classic(base_size=10.5, base_family="sans") + theme(
  plot.title=element_text(face="bold", size=10.5),
  plot.subtitle=element_text(size=8.5), axis.title=element_text(size=9),
  axis.text=element_text(size=8.2), legend.position="bottom",
  legend.text=element_text(size=8),
  plot.caption=element_text(size=7.5, hjust=0, margin=margin(t=4)),
  plot.caption.position="plot", plot.margin=margin(5,6,6,6)
)
write_tikz <- function(plot, path, width, height, source_lines) {
  tikz(path, width=width, height=height, standAlone=FALSE)
  print(plot)
  dev.off()
  body <- readLines(path, warn=FALSE)
  body <- body[!grepl("^% Created by tikzDevice", body)]
  writeLines(c(source_lines, body), path, useBytes=TRUE)
}

arm_levels <- c("nf4dq_edited_layer", "nf4dq_full_model", "int8_edited_layer", "int8_full_model")
arm_labels <- c("NF4/EL", "NF4/FM", "INT8/EL", "INT8/FM")

# Figure 3 source fields: cells[*].arms[*].flat_rank.point,
# cells[*].arms[*].within_probe_rank.point, and cells[*].arms[*].edit_level_ranks.signed_mean.point.
rows <- list(); z <- 0
for (cell in s$cells) for (arm in names(cell$arms)) {
  x <- cell$arms[[arm]]; z <- z + 1
  rows[[z]] <- data.frame(
    model=cell$slug, editor=toupper(cell$editor), arm=arm,
    flat=x$flat_rank$point, within=x$within_probe_rank$point,
    edit=x$edit_level_ranks$signed_mean$point
  )
}
r <- do.call(rbind, rows)
r3 <- rbind(
  transform(r, estimand="cross-probe", value=flat),
  transform(r, estimand="within-probe", value=within),
  transform(r, estimand="edit-level", value=edit)
)
r3$estimand <- factor(r3$estimand, levels=c("cross-probe", "within-probe", "edit-level"))
rank_panels <- lapply(seq_along(arm_levels), function(i) {
  ggplot(r3[r3$arm == arm_levels[i], ], aes(estimand, value, colour=estimand)) +
    geom_point(position=position_jitter(width=.11, height=0, seed=121), size=1.15, alpha=.75) +
    scale_colour_manual(values=c("cross-probe"=C_GREY, "within-probe"=C_BLUE, "edit-level"=C_ORANGE)) +
    scale_y_continuous(limits=c(0,1.04), breaks=c(0,.5,1), expand=c(0,.01)) +
    labs(title=arm_labels[i], x=NULL, y=if (i == 1) "Spearman rank survival" else NULL, colour=NULL) +
    theme_p + theme(axis.text.x=element_blank(), axis.ticks.x=element_blank(), legend.position="none")
})
p3 <- wrap_plots(rank_panels, nrow=1) + plot_layout(guides="collect") +
  plot_annotation(
    title="Rank-survival estimands",
    caption="Purple: cross-probe; blue: within-probe; orange: edit-level.",
    tag_levels="A", theme=theme(plot.tag=element_text(face="bold", size=10),
                                plot.caption=element_text(size=7.5, hjust=0,
                                                          margin=margin(t=5)),
                                plot.caption.position="plot")
  )
write_tikz(p3, "fig03_rank_survival.tex", 7.2, 3.45, c(
  "% SOURCE: quant_survival_repair_v1.json fields cells[*].arms[*].flat_rank.point; cells[*].arms[*].within_probe_rank.point; cells[*].arms[*].edit_level_ranks.signed_mean.point.",
  "% SOURCE: quant_survival_repair_v1.json fields module_provenance.version; module_provenance.n_boot (asserted v1.2.1 and 500)."
))

# Figure 4 source fields: cells.*.c3.{nf4dq,int8}.median_ratio_mean,
# cells.*.c3.{nf4dq,int8}.r_func_mean, and cells.*.c3.{nf4dq,int8}.r_param_mean.
cell_codes <- c(
  "Llama-3.2-1B_memit_L12"="L1-M", "Llama-3.2-1B_alpha_L12"="L1-A",
  "Llama-3.2-3B_alpha_L24"="L3-A", "Qwen2.5-1.5B_memit_L21"="Q1-M",
  "Qwen2.5-1.5B_alpha_L21"="Q1-A", "Llama-3.2-3B_memit_L24"="L3-M",
  "Llama-3.2-1B_rome_L12"="L1-R", "Llama-3.2-3B_rome_L24"="L3-R",
  "Qwen2.5-1.5B_rome_L21"="Q1-R"
)
rows <- list(); z <- 0
for (nm in names(g$cells)) for (q in c("nf4dq", "int8")) {
  x <- g$cells[[nm]]$c3[[q]]; z <- z + 1
  rows[[z]] <- data.frame(
    cell=unname(cell_codes[nm]), scheme=ifelse(q == "nf4dq", "NF4", "INT8"),
    ratio=x$median_ratio_mean, func=x$r_func_mean, param=x$r_param_mean
  )
}
gd <- do.call(rbind, rows)
gd$cell <- factor(gd$cell, levels=unname(cell_codes[names(g$cells)]))
gap_panel <- function(field, heading, ylabel, show_legend=FALSE) {
  ggplot(gd, aes(cell, .data[[field]], fill=scheme)) +
    geom_col(position=position_dodge(.72), width=.62) +
    scale_fill_manual(values=c(NF4=C_ORANGE, INT8=C_BLUE)) +
    scale_y_continuous(expand=expansion(mult=c(0,.08))) +
    labs(title=heading, x=NULL, y=ylabel, fill=NULL) + theme_p +
    theme(axis.text.x=element_text(angle=45, hjust=1, vjust=1, size=6.5),
          legend.position=if (show_legend) "bottom" else "none")
}
p4 <- gap_panel("ratio", "Median |dW|/b", "median ratio") +
  gap_panel("func", "Function-space gap", expression(r[func])) +
  gap_panel("param", "Parameter-space gap", expression(r[param]), TRUE) +
  plot_layout(guides="collect", widths=c(1,1,1)) +
  plot_annotation(
    title="Reconstruction gap and bin-width sensitivity",
    caption="Codes: L1/L3 = Llama 1B/3B; Q1 = Qwen 1.5B; M/A/R = MEMIT/AlphaEdit/ROME.\nOrange = NF4; blue = INT8; K3 concentration is KILLED on this axis.",
    tag_levels="A", theme=theme(plot.tag=element_text(face="bold", size=10),
                                plot.caption=element_text(size=7.3, hjust=0,
                                                          margin=margin(t=5)),
                                plot.caption.position="plot",
                                legend.position="bottom")
  )
write_tikz(p4, "fig04_reconstruction_gap.tex", 7.4, 4.0, c(
  "% SOURCE: gate_readout.json fields cells.*.c3.{nf4dq,int8}.median_ratio_mean; cells.*.c3.{nf4dq,int8}.r_func_mean; cells.*.c3.{nf4dq,int8}.r_param_mean."
))
cat("wrote fig03--fig04; canonical version=", s$module_provenance$version,
    " n_boot=", s$module_provenance$n_boot, "\n")

# ── Fig 07: Model-by-model flat-rank survival with bootstrap CIs ───────────────
{
  rows <- list(); z <- 0
  for (cell in s$cells) for (arm in names(cell$arms)) {
    x <- cell$arms[[arm]]; z <- z + 1
    rows[[z]] <- data.frame(
      model   = cell$slug,
      editor  = toupper(cell$editor),
      arm     = arm,
      point   = x$flat_rank$point,
      lo      = x$flat_rank$range_min_max[[1]],
      hi      = x$flat_rank$range_min_max[[2]],
      stringsAsFactors = FALSE
    )
  }
  df7 <- do.call(rbind, rows)
  df7$arm <- factor(df7$arm, levels=arm_levels, labels=arm_labels)
  df7$model_ed <- paste0(df7$model,"\n",df7$editor)

  pa7 <- ggplot(df7[df7$arm %in% c("NF4\nEL","NF4\nFM"),],
                aes(x=model_ed, y=point, ymin=lo, ymax=hi, colour=arm, group=arm)) +
    geom_errorbar(position=position_dodge(0.5), width=0.3, linewidth=0.6) +
    geom_point(position=position_dodge(0.5), size=2) +
    geom_hline(yintercept=0.5, linetype="dashed", colour="grey50", linewidth=0.4) +
    scale_colour_manual(values=c("NF4\nEL"=C_ORANGE,"NF4\nFM"=C_TEAL), name="Arm") +
    labs(x=NULL, y="Flat rank survival", title="NF4") +
    theme_p + theme(axis.text.x=element_text(size=7), plot.title=element_text(size=9,hjust=0.5))

  pb7 <- ggplot(df7[df7$arm %in% c("INT8\nEL","INT8\nFM"),],
                aes(x=model_ed, y=point, ymin=lo, ymax=hi, colour=arm, group=arm)) +
    geom_errorbar(position=position_dodge(0.5), width=0.3, linewidth=0.6) +
    geom_point(position=position_dodge(0.5), size=2) +
    geom_hline(yintercept=0.5, linetype="dashed", colour="grey50", linewidth=0.4) +
    scale_colour_manual(values=c("INT8\nEL"=C_BLUE,"INT8\nFM"=C_GREY), name="Arm") +
    labs(x=NULL, y="Flat rank survival", title="INT8") +
    theme_p + theme(axis.text.x=element_text(size=7), plot.title=element_text(size=9,hjust=0.5))

  # Panel c: cond survival (given fp32 worked) all arms
  rows_c <- Filter(Negate(is.null), lapply(s$cells, function(cell) {
    do.call(rbind, Filter(Negate(is.null), lapply(names(cell$arms), function(arm) {
      cs <- cell$arms[[arm]]$conditional_survival_given_fp32_worked
      if (length(cs) == 0 || is.null(cs$point)) return(NULL)
      data.frame(model=cell$slug, editor=toupper(cell$editor), arm=arm,
                 point=cs$point, lo=cs$ci95[[1]], hi=cs$ci95[[2]], stringsAsFactors=FALSE)
    })))
  }))
  if (length(rows_c) > 0) {
    df7c <- do.call(rbind, rows_c)
    df7c$arm <- factor(df7c$arm, levels=arm_levels, labels=arm_labels)
    pc7 <- ggplot(df7c, aes(x=arm, y=point, ymin=lo, ymax=hi, colour=model, group=interaction(model,editor))) +
      geom_errorbar(position=position_dodge(0.6), width=0.25) +
      geom_point(position=position_dodge(0.6), size=2) +
      geom_hline(yintercept=0.5, linetype="dashed", colour="grey50", linewidth=0.4) +
      labs(x="Arm", y="Conditional survival", colour="Model") +
      theme_p
  } else {
    pc7 <- ggplot() + annotate("text",x=0.5,y=0.5,label="No conditional survival data") + theme_void()
  }

  # Panel d: flat-rank survival by arm as grouped bar (simpler and correct)
  pd7 <- ggplot(df7, aes(x=arm, y=point, ymin=lo, ymax=hi, fill=model, group=interaction(model,editor))) +
    geom_col(position=position_dodge(0.7), width=0.6, alpha=0.85) +
    geom_errorbar(position=position_dodge(0.7), width=0.2) +
    geom_hline(yintercept=0.5, linetype="dashed", colour="grey50", linewidth=0.4) +
    labs(x="Arm", y="Flat rank survival", fill="Model") +
    theme_p

  fig7 <- (pa7 | pb7) / (pc7 | pd7) +
    plot_annotation(tag_levels="a", theme=theme(plot.tag=element_text(face="bold",size=9)))
  write_tikz(fig7, "fig07_model_breakdown.tex", 7.2, 5.5, c(
    "% SOURCE: quant_survival_repair_v1.json fields cells.*.arms.*.{flat_rank,conditional_survival_given_fp32_worked,absolute_quantized_esr}."
  ))
  cat("wrote fig07_model_breakdown.tex\n")
}

# ── Fig 08: Base-noise mechanism (4-panel) ─────────────────────────────────────
{
  NOISE_FILE <- file.path(HARNESS, "results/quant_survival/aggregate/base_noise_swamping_20260726.json")
  stopifnot(file.exists(NOISE_FILE))
  nb <- fromJSON(NOISE_FILE, simplifyVector=FALSE)
  pc_noise <- nb$per_cell
  if (is.list(pc_noise) && !is.null(names(pc_noise))) {
    nrows <- do.call(rbind, lapply(names(pc_noise), function(k) {
      v <- pc_noise[[k]]
      data.frame(cell=k, slug=v$slug, editor=toupper(v$editor), arm=v$arm,
                 nsr=v$noise_to_signal_ratio, flat=v$flat_rank_survival,
                 noise=v$base_quant_noise_mean_abs, stringsAsFactors=FALSE)
    }))
    nrows$arm <- factor(nrows$arm, levels=arm_levels, labels=arm_labels)
  } else {
    nrows <- data.frame()
  }
  corrs <- nb$correlations_over_36_arm_cells
  rho_nsr_flat <- corrs$nsr_vs_flat_rank$rho
  rho_noise_flat <- corrs$raw_noise_vs_flat_rank$rho

  if (nrow(nrows) > 0) {
    pa8 <- ggplot(nrows, aes(x=nsr, y=flat)) +
      geom_point(aes(colour=editor), size=2, alpha=0.8) +
      geom_smooth(method="lm", se=TRUE, colour="grey30", linewidth=0.6) +
      annotate("text",x=Inf,y=Inf, hjust=1.1,vjust=1.5,
               label=sprintf("$\\rho$ = %.3f", rho_nsr_flat), size=3) +
      labs(x="Noise-to-signal ratio", y="Flat rank survival", colour="Editor") + theme_p

    pb8 <- ggplot(nrows, aes(x=noise, y=flat)) +
      geom_point(aes(colour=editor), size=2, alpha=0.8) +
      geom_smooth(method="lm", se=TRUE, colour="grey30", linewidth=0.6) +
      annotate("text",x=Inf,y=Inf, hjust=1.1,vjust=1.5,
               label=sprintf("$\\rho$ = %.3f", rho_noise_flat), size=3) +
      labs(x="Base quant noise (mean abs)", y="Flat rank survival", colour="Editor") + theme_p

    pc8 <- ggplot(nrows, aes(x=arm, y=nsr, fill=editor)) +
      geom_boxplot(outlier.size=1) +
      labs(x="Arm", y="Noise-to-signal ratio", fill="Editor") +
      theme_p + theme(axis.text.x=element_text(size=7))

    nf4_only <- nrows[grepl("nf4",nrows$arm),]
    pd8 <- ggplot(nf4_only, aes(x=arm, y=flat, fill=slug)) +
      geom_boxplot(outlier.size=1) +
      geom_hline(yintercept=0.5, linetype="dashed", colour="grey50", linewidth=0.4) +
      labs(x="NF4 arm", y="Flat rank survival", fill="Model") +
      theme_p + theme(axis.text.x=element_text(size=7))
  } else {
    pa8 <- pb8 <- pc8 <- pd8 <- ggplot() + annotate("text",x=0.5,y=0.5,label="No noise data") + theme_void()
  }

  fig8 <- (pa8 | pb8) / (pc8 | pd8) +
    plot_annotation(tag_levels="a", theme=theme(plot.tag=element_text(face="bold",size=9)))
  write_tikz(fig8, "fig08_noise_mechanism.tex", 7.2, 5.5, c(
    "% SOURCE: base_noise_swamping_20260726.json fields per_cell.*.{noise_to_signal_ratio,flat_rank_survival,base_quant_noise_mean_abs}."
  ))
  cat("wrote fig08_noise_mechanism.tex\n")
}

# ── Fig 09: Editor ordering null result (4-panel) ─────────────────────────────
{
  ED_FILE <- file.path(HARNESS, "results/quant_survival/aggregate/editor_ordering_bootstrap_20260726.json")
  stopifnot(file.exists(ED_FILE))
  eb <- fromJSON(ED_FILE, simplifyVector=FALSE)
  results <- eb$results
  # Extract all contrasts across model×arm cells
  contrast_rows <- do.call(rbind, lapply(names(results), function(k) {
    r <- results[[k]]
    parts <- strsplit(k,"__")[[1]]
    slug <- parts[1]; arm_raw <- paste(parts[-1],collapse="__")
    do.call(rbind, lapply(names(r$contrasts), function(cn) {
      cv <- r$contrasts[[cn]]
      ci <- unlist(cv$ci95)
      data.frame(slug=slug, arm_raw=arm_raw, contrast=gsub("_","-",cn),
                 obs_diff=as.numeric(cv$observed_diff),
                 ci_lo=ci[[1]], ci_hi=ci[[2]],
                 p=as.numeric(cv$p_two_sided),
                 distingishable=isTRUE(cv$distinguishable_at_95),
                 stringsAsFactors=FALSE)
    }))
  }))
  contrast_rows$arm <- factor(contrast_rows$arm_raw, levels=arm_levels, labels=arm_labels)

  pa9 <- ggplot(contrast_rows, aes(x=contrast, y=obs_diff, fill=slug)) +
    geom_col(position=position_dodge(0.8), width=0.7) +
    geom_errorbar(aes(ymin=ci_lo, ymax=ci_hi), position=position_dodge(0.8), width=0.2) +
    geom_hline(yintercept=0, linewidth=0.4) +
    facet_wrap(~arm, ncol=2, scales="free_x") +
    labs(x="Contrast", y="Observed diff", fill="Model") +
    theme_p + theme(axis.text.x=element_text(size=6.5, angle=20, hjust=1),
                    strip.text=element_text(size=6.5))

  # panel b: p-values
  pb9 <- ggplot(contrast_rows, aes(x=contrast, y=-log10(pmax(p,1e-4)), fill=distingishable)) +
    geom_col(position=position_dodge(0.8), width=0.7) +
    geom_hline(yintercept=-log10(0.05), linetype="dashed", colour="grey40") +
    scale_fill_manual(values=c("FALSE"="grey70","TRUE"="#D73027"), name="Sig.") +
    labs(x="Contrast", y="-log10(p)", fill="p<0.05") +
    theme_p + theme(axis.text.x=element_text(size=6.5, angle=20, hjust=1))

  # panel c: CI overlap illustration
  levels_df <- do.call(rbind, lapply(names(results), function(k) {
    r <- results[[k]]; parts <- strsplit(k,"__")[[1]]; slug <- parts[1]; arm_raw <- paste(parts[-1],collapse="__")
    do.call(rbind, lapply(names(r$levels), function(ed) {
      lv <- r$levels[[ed]]
      data.frame(slug=slug, arm_raw=arm_raw, editor=toupper(ed),
                 point=lv$point, lo=lv$ci95[[1]], hi=lv$ci95[[2]], stringsAsFactors=FALSE)
    }))
  }))
  levels_df$arm <- factor(levels_df$arm_raw, levels=arm_levels, labels=arm_labels)
  pc9 <- ggplot(levels_df, aes(x=editor, y=point, ymin=lo, ymax=hi,
                                colour=slug, group=interaction(slug,arm))) +
    geom_pointrange(position=position_dodge(0.6), size=0.4) +
    labs(x="Editor", y="Flat rank survival", colour="Model") +
    theme_p

  # panel d: summary text table as tile plot
  summ <- eb$summary
  summ_df <- data.frame(
    metric=c("Models", "Editors", "Arms", "Contrasts", "Distinguishable at 95%"),
    value =c(as.character(length(eb$grid$models)),
             as.character(length(eb$grid$editors)),
             as.character(length(eb$grid$arms)),
             as.character(summ$n_contrasts),
             as.character(summ$n_distinguishable_at_95))
  )
  pd9 <- ggplot(summ_df, aes(x=1, y=reorder(metric,seq_len(nrow(summ_df))), label=value)) +
    geom_text(hjust=0, nudge_x=0.05, size=3) +
    geom_text(aes(x=0.5, label=metric), hjust=1, size=3, colour="grey30") +
    xlim(0,2) + labs(title="Bootstrap summary") +
    theme_void() + theme(plot.title=element_text(size=9,hjust=0.5))

  fig9 <- (pa9 | pb9) / (pc9 | pd9) +
    plot_annotation(tag_levels="a", theme=theme(plot.tag=element_text(face="bold",size=9)))
  write_tikz(fig9, "fig09_editor_null.tex", 7.2, 5.5, c(
    "% SOURCE: editor_ordering_bootstrap_20260726.json fields results.*.contrasts.*.{observed_diff,ci95,p_two_sided}."
  ))
  cat("wrote fig09_editor_null.tex\n")
}
