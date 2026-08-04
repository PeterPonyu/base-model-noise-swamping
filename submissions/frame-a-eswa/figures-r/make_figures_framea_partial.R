#!/usr/bin/env Rscript
# Frame-A partial figures — uses MIX_A (30/33 real) data available now.
# F2/F3/F5/F7/F8 (Pareto, predictor, gate status, MIX_B, MIX_C) are PENDING gate v2.
# Run from submissions/frame-a-eswa/

suppressPackageStartupMessages({
  library(jsonlite); library(ggplot2); library(dplyr); library(patchwork); library(tikzDevice)
})

HARNESS <- "../../edit-harness"
CELLS_DIR <- file.path(HARNESS, "results/frame_a/cells")
OUT_DIR <- "figures-src"
dir.create(OUT_DIR, showWarnings=FALSE, recursive=TRUE)

# Load all MIX_A cells
load_mix <- function(mix_name) {
  files <- list.files(CELLS_DIR, pattern=paste0("cell_.*_real_", mix_name, "_.*\\.json"),
                      full.names=TRUE)
  rows <- lapply(files, function(f) {
    # Use readLines + jsonlite::fromJSON with NaN handling
    txt <- paste(readLines(f, warn=FALSE), collapse="\n")
    txt <- gsub("\\bNaN\\b", "null", txt)  # replace bare NaN with JSON null
    d <- tryCatch(fromJSON(txt, simplifyVector=TRUE), error=function(e) NULL)
    if (is.null(d)) return(NULL)
    q_raw <- d$quality; c_raw <- d$cost; disc_raw <- d$discovery
    data.frame(
      mix     = as.character(d$mix %||% mix_name),
      policy  = as.character(d$policy),
      seed    = as.integer(d$seed),
      Q       = as.numeric(if (is.list(q_raw)) q_raw$Q else q_raw["Q"]),
      A_upd   = as.numeric(if (is.list(q_raw)) q_raw$A_upd else q_raw["A_upd"]),
      total_gpu_s = as.numeric(if (is.list(c_raw)) c_raw$total_gpu_s else c_raw["total_gpu_s"]),
      serve_gpu_s = as.numeric(if (is.list(c_raw)) c_raw$serve_gpu_s else c_raw["serve_gpu_s"]),
      exposure    = as.numeric(if (is.list(c_raw)) c_raw$exposure_surface_mean else NA_real_),
      recall_lift = as.numeric(if (is.list(disc_raw)) disc_raw$lift else NA_real_),
      stringsAsFactors = FALSE
    )
  })
  rows <- Filter(Negate(is.null), rows)
  do.call(rbind, rows)
}
`%||%` <- function(a, b) if (!is.null(a)) a else b

stopifnot(dir.exists(CELLS_DIR))
mixa <- load_mix("MIX_A")
mixa$policy <- gsub("_", "-", mixa$policy)  # LaTeX-safe labels
POLICY_COLS <- setNames(
  c("#2CA02C","#1F77B4","#AEC7E8","#D62728","#FF7F0E","#9467BD",
    "#8C564B","#E377C2","#7F7F7F","#BCBD22","#17BECF"),
  c("oracle","always-grace","both","always-rag","always-edit","always-ft",
    "ft-merge","cost-only","damage-only","always-reject","random")
)

write_tex <- function(plot, path, source_json, width=6.5, height=5) {
  tikz(path, width=width, height=height, standAlone=FALSE)
  print(plot); dev.off()
  lines <- readLines(path, warn=FALSE)
  lines <- lines[!grepl("^% Created by tikzDevice version", lines)]
  writeLines(c(paste0("% SOURCE: ", source_json), lines), path, useBytes=TRUE)
  cat(sprintf("Wrote %s\n", path))
}

pt <- theme_minimal(base_size=8) +
  theme(legend.position="bottom", legend.text=element_text(size=6.5),
        legend.title=element_text(size=7), panel.grid.minor=element_blank(),
        axis.title=element_text(size=7.5), axis.text=element_text(size=6.5),
        plot.title=element_text(size=8, hjust=0.5))

# ── Fig F1: Operator routing problem (4-panel) ─────────────────────────────────
{
  # Aggregate MIX_A by policy (mean across seeds)
  summ <- mixa %>%
    group_by(policy) %>%
    summarise(Q_mean=mean(Q,na.rm=TRUE), Q_sd=sd(Q,na.rm=TRUE),
              cost_mean=mean(total_gpu_s,na.rm=TRUE), cost_sd=sd(total_gpu_s,na.rm=TRUE),
              exposure_mean=mean(exposure,na.rm=TRUE), .groups="drop")

  # Panel a: Q × cost scatter (MIX_A operating points)
  pa <- ggplot(summ, aes(x=cost_mean, y=Q_mean, colour=policy, label=policy)) +
    geom_point(size=2.5, alpha=0.9) +
    geom_text(size=2, vjust=-0.7, show.legend=FALSE) +
    scale_colour_manual(values=POLICY_COLS, guide="none") +
    labs(x="Mean total GPU-s", y="Mean quality Q") + pt

  # Panel b: quality distribution per policy
  pb <- ggplot(mixa, aes(x=reorder(policy,Q,median), y=Q, fill=policy)) +
    geom_boxplot(outlier.size=0.8) +
    scale_fill_manual(values=POLICY_COLS, guide="none") +
    labs(x=NULL, y="Quality Q") +
    pt + theme(axis.text.x=element_text(angle=30, hjust=1, size=6))

  # Panel c: cost distribution per policy
  pc <- ggplot(mixa, aes(x=reorder(policy,total_gpu_s,median), y=total_gpu_s/3600, fill=policy)) +
    geom_boxplot(outlier.size=0.8) +
    scale_fill_manual(values=POLICY_COLS, guide="none") +
    labs(x=NULL, y="Total GPU-h") +
    pt + theme(axis.text.x=element_text(angle=30, hjust=1, size=6))

  # Panel d: exposure surface mean per policy
  pd <- ggplot(summ, aes(x=reorder(policy,exposure_mean), y=exposure_mean, fill=policy)) +
    geom_col(width=0.7) +
    scale_fill_manual(values=POLICY_COLS, guide="none") +
    labs(x=NULL, y="Mean exposure surface") +
    pt + theme(axis.text.x=element_text(angle=30, hjust=1, size=6))

  fig <- wrap_plots(pa, pb, pc, pd, ncol=2) +
    plot_annotation(tag_levels="a") &
    theme(plot.tag=element_text(face="bold", size=9), plot.tag.position=c(0.01,0.99))
  write_tex(fig, file.path(OUT_DIR, "figF1_routing_problem.tex"),
            "../../edit-harness/results/frame_a/cells/cell_*_real_MIX_A_*.json")
}

# ── Fig F6: MIX_A policy spread — full 4-panel ────────────────────────────────
{
  summ <- mixa %>%
    group_by(policy) %>%
    summarise(Q_mean=mean(Q,na.rm=TRUE), Q_lo=quantile(Q,0.25,na.rm=TRUE),
              Q_hi=quantile(Q,0.75,na.rm=TRUE),
              cost_mean=mean(total_gpu_s,na.rm=TRUE), .groups="drop")
  summ$on_pareto <- summ$policy %in% c("always_grace","oracle","both")

  pa <- ggplot(summ, aes(x=cost_mean/3600, y=Q_mean, colour=policy, shape=on_pareto)) +
    geom_point(size=3) +
    scale_colour_manual(values=POLICY_COLS, guide="none") +
    scale_shape_manual(values=c("TRUE"=17,"FALSE"=16), name="On Pareto") +
    geom_text(aes(label=policy), size=1.9, vjust=-0.8, show.legend=FALSE) +
    labs(x="Mean cost (GPU-h)", y="Mean quality Q") + pt

  # Panel b: always_grace dominant position annotated
  pb <- ggplot(summ, aes(x=cost_mean/3600, y=Q_mean)) +
    geom_point(aes(colour=policy), size=2.5) +
    geom_point(data=summ[summ$policy=="always_grace",],
               colour="#1F77B4", size=4, shape=17) +
    annotate("text", x=summ$cost_mean[summ$policy=="always_grace"]/3600,
             y=summ$Q_mean[summ$policy=="always_grace"],
             label="always\\_grace", hjust=-0.15, size=2.5, colour="#1F77B4") +
    scale_colour_manual(values=POLICY_COLS, guide="none") +
    labs(x="Mean cost (GPU-h)", y="Mean quality Q") + pt

  # Panel c: IQR bars per policy for Q
  pc <- ggplot(summ, aes(x=reorder(policy,-Q_mean), y=Q_mean,
                          ymin=Q_lo, ymax=Q_hi, colour=policy)) +
    geom_pointrange() +
    geom_hline(yintercept=summ$Q_mean[summ$policy=="random"], linetype="dashed",
               colour="grey50", linewidth=0.4) +
    scale_colour_manual(values=POLICY_COLS, guide="none") +
    labs(x=NULL, y="Quality Q (IQR)") +
    pt + theme(axis.text.x=element_text(angle=30,hjust=1,size=6))

  # Panel d: damage recall lift
  lift_df <- mixa %>% filter(!is.na(recall_lift)) %>%
    group_by(policy) %>% summarise(lift=mean(recall_lift,na.rm=TRUE), .groups="drop")
  pd <- if (nrow(lift_df) > 0)
    ggplot(lift_df, aes(x=reorder(policy,-lift), y=lift, fill=policy)) +
      geom_col(width=0.7) +
      geom_hline(yintercept=1, linetype="dashed", colour="grey50", linewidth=0.4) +
      scale_fill_manual(values=POLICY_COLS, guide="none") +
      labs(x=NULL, y="Damage recall lift") +
      pt + theme(axis.text.x=element_text(angle=30,hjust=1,size=6))
  else ggplot() + annotate("text",x=0.5,y=0.5,label="No recall lift data (MIX_A)") + theme_void()

  fig <- wrap_plots(pa, pb, pc, pd, ncol=2) +
    plot_annotation(tag_levels="a") &
    theme(plot.tag=element_text(face="bold",size=9), plot.tag.position=c(0.01,0.99))
  write_tex(fig, file.path(OUT_DIR, "figF6_mixa_spread.tex"),
            "../../edit-harness/results/frame_a/cells/cell_*_real_MIX_A_*.json")
}

# ── Fig F9: Policy failure modes (MIX_A panels a/b; c/d PENDING) ──────────────
{
  summ <- mixa %>% group_by(policy,seed) %>%
    summarise(Q=mean(Q,na.rm=TRUE), A_upd=mean(A_upd,na.rm=TRUE),
              cost=mean(total_gpu_s,na.rm=TRUE), .groups="drop")

  # Panel a: edit arm failure — always_edit vs oracle quality
  cmp_df <- summ[summ$policy %in% c("always_edit","oracle","always_grace","cost_only","damage_only"),]
  pa <- ggplot(cmp_df, aes(x=policy, y=Q, fill=policy)) +
    geom_boxplot(outlier.size=0.8) +
    scale_fill_manual(values=POLICY_COLS, guide="none") +
    labs(x=NULL, y="Quality Q") +
    pt + theme(axis.text.x=element_text(angle=20,hjust=1,size=6.5))

  # Panel b: RAG arm cost comparison
  rag_df <- summ[summ$policy %in% c("always_rag","always_grace","always_edit"),]
  pb <- ggplot(rag_df, aes(x=policy, y=cost/3600, fill=policy)) +
    geom_boxplot(outlier.size=0.8) +
    scale_fill_manual(values=POLICY_COLS, guide="none") +
    labs(x=NULL, y="Total GPU-h") +
    pt + theme(axis.text.x=element_text(angle=20,hjust=1,size=6.5))

  # Panels c/d: PENDING — require gate v2 and Q_ext
  pc <- ggplot() +
    annotate("text",x=0.5,y=0.5,label="(c) PENDING:\ngrace-optimal regions\n(requires gate v2)",
             hjust=0.5, vjust=0.5, size=3, colour="grey40") + theme_void()
  pd <- ggplot() +
    annotate("text",x=0.5,y=0.5,label="(d) PENDING:\nedge cases\n(requires MIX_C + gate v2)",
             hjust=0.5, vjust=0.5, size=3, colour="grey40") + theme_void()

  fig <- wrap_plots(pa, pb, pc, pd, ncol=2) +
    plot_annotation(tag_levels="a") &
    theme(plot.tag=element_text(face="bold",size=9), plot.tag.position=c(0.01,0.99))
  write_tex(fig, file.path(OUT_DIR, "figF9_failure_modes.tex"),
            "../../edit-harness/results/frame_a/cells/cell_*_real_MIX_A_*.json")
}

# ── Fig F10: Governance exposure (MIX_A panels a/b; c/d PENDING) ──────────────
{
  gov_df <- mixa %>%
    group_by(policy) %>%
    summarise(exp_mean=mean(exposure,na.rm=TRUE), exp_sd=sd(exposure,na.rm=TRUE),
              Q_mean=mean(Q,na.rm=TRUE), cost_mean=mean(total_gpu_s,na.rm=TRUE), .groups="drop")

  pa <- ggplot(gov_df, aes(x=reorder(policy,exp_mean), y=exp_mean,
                            ymin=pmax(0,exp_mean-exp_sd), ymax=exp_mean+exp_sd, fill=policy)) +
    geom_col(width=0.7) +
    geom_errorbar(width=0.2) +
    scale_fill_manual(values=POLICY_COLS, guide="none") +
    labs(x=NULL, y="Mean exposure surface") +
    pt + theme(axis.text.x=element_text(angle=30,hjust=1,size=6))

  # Panel b: router footprint vs always_rag baseline
  ref_rag <- gov_df$exp_mean[gov_df$policy == "always-rag"]
  if (length(ref_rag) == 0 || ref_rag == 0) ref_rag <- 1
  gov_df$rel_exp <- gov_df$exp_mean / ref_rag
  pb <- ggplot(gov_df, aes(x=reorder(policy,rel_exp), y=rel_exp, fill=policy)) +
    geom_col(width=0.7) +
    geom_hline(yintercept=1, linetype="dashed", colour="grey50", linewidth=0.4) +
    scale_fill_manual(values=POLICY_COLS, guide="none") +
    labs(x=NULL, y="Exposure rel. to always\\_rag") +
    pt + theme(axis.text.x=element_text(angle=30,hjust=1,size=6))

  pc <- ggplot() +
    annotate("text",x=0.5,y=0.5,label="(c) PENDING:\nrouter footprint across mixes\n(requires MIX_B + MIX_C + gate v2)",
             hjust=0.5,vjust=0.5,size=3,colour="grey40") + theme_void()
  pd <- ggplot() +
    annotate("text",x=0.5,y=0.5,label="(d) PENDING:\nalways\\_rag contrast (full data)\n(requires gate v2)",
             hjust=0.5,vjust=0.5,size=3,colour="grey40") + theme_void()

  fig <- wrap_plots(pa, pb, pc, pd, ncol=2) +
    plot_annotation(tag_levels="a") &
    theme(plot.tag=element_text(face="bold",size=9), plot.tag.position=c(0.01,0.99))
  write_tex(fig, file.path(OUT_DIR, "figF10_governance.tex"),
            "../../edit-harness/results/frame_a/cells/cell_*_real_MIX_A_*.json")
}

cat("Frame-A partial figures complete.\n")
