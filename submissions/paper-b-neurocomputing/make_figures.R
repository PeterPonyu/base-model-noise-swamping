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

# Figure 2 source fields: cells[*].arms[*].absolute_quantized_esr.point and
# cells[*].arms[*].conditional_survival_given_fp32_worked.point.
rows <- list(); z <- 0
for (cell in s$cells) for (arm in names(cell$arms)) {
  x <- cell$arms[[arm]]; z <- z + 1
  rows[[z]] <- data.frame(
    model=cell$slug, editor=toupper(cell$editor), arm=arm,
    esr=x$absolute_quantized_esr$point,
    cond=x$conditional_survival_given_fp32_worked$point
  )
}
d <- do.call(rbind, rows)
arm_levels <- c("nf4dq_edited_layer", "nf4dq_full_model", "int8_edited_layer", "int8_full_model")
arm_labels <- c("NF4\nEL", "NF4\nFM", "INT8\nEL", "INT8\nFM")
d$arm <- factor(d$arm, levels=arm_levels, labels=arm_labels)
common_esr <- list(
  geom_col(position=position_dodge(.7), width=.62),
  geom_hline(yintercept=c(.8,.9), linetype="22", colour="grey65", linewidth=.3),
  facet_wrap(~model, nrow=1),
  scale_fill_manual(values=c(ALPHA=C_ORANGE, MEMIT=C_TEAL, ROME=C_BLUE)),
  scale_y_continuous(limits=c(0,1.04), breaks=c(0,.8,.9,1), expand=c(0,.01)),
  theme_p
)
p2a <- ggplot(d, aes(arm, esr, fill=editor)) + common_esr +
  labs(title="Absolute quantized efficacy", x=NULL, y="absolute ESR", fill="editor")
p2b <- ggplot(d, aes(arm, cond, fill=editor)) + common_esr +
  labs(title="Conditional survival given FP32 success", x=NULL, y="conditional ESR", fill="editor")
p2 <- p2a / p2b + plot_annotation(
  tag_levels="A", caption="Canonical v1.2.1; hierarchical bootstrap n=500.",
  theme=theme(plot.tag=element_text(face="bold", size=10))
)
write_tikz(p2, "fig02_efficacy_survival.tex", 6.8, 5.0, c(
  "% SOURCE: quant_survival_repair_v1.json fields cells[*].arms[*].absolute_quantized_esr.point; cells[*].arms[*].conditional_survival_given_fp32_worked.point.",
  "% SOURCE: quant_survival_repair_v1.json fields module_provenance.version; module_provenance.n_boot (asserted v1.2.1 and 500)."
))

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
cat("wrote fig02--fig04; canonical version=", s$module_provenance$version,
    " n_boot=", s$module_provenance$n_boot, "\n")
