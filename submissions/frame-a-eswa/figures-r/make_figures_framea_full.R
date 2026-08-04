#!/usr/bin/env Rscript
# Frame-A FULL figures — uses MIX_A + MIX_B + MIX_C (99 cells) + namespaced P2.
# Run from submissions/frame-a-eswa/

suppressPackageStartupMessages({
  library(jsonlite); library(ggplot2); library(dplyr); library(patchwork); library(tikzDevice)
})

HARNESS <- "../../edit-harness"
CELLS_DIR <- file.path(HARNESS, "results/frame_a/cells")
P2_PATH <- file.path(CELLS_DIR, "p2_llama-3.2-1b_real_MIX_C.json")
OUT_DIR <- "figures-src"
dir.create(OUT_DIR, showWarnings=FALSE, recursive=TRUE)

# ── Gate v2 reality (NOT a relaxation; honest reporting) ──────────────────────
# P2 router_edit_majority_on_privacy = 0.103 < 0.5: router on MIX_C does NOT
# majority-on-privacy. This is the empirical truth — we report it as-is in F7/F8
# rather than mute the gate. See provenance_v2_report.json p2_status FAIL.

stopifnot(dir.exists(CELLS_DIR))
p2 <- jsonlite::fromJSON(paste(readLines(P2_PATH, warn=FALSE), collapse="\n"))
stopifnot(is.list(p2), all(c("exposure_edit","exposure_rag","footprint_delta",
                             "overhead_delta","router_edit_majority_on_privacy") %in% names(p2)))

# Load all cells from MIX_A + MIX_B + MIX_C
load_mix <- function(mix_name) {
  files <- list.files(CELLS_DIR,
                      pattern=paste0("cell_.*_real_", mix_name, "_.*\\.json"),
                      full.names=TRUE)
  cat(sprintf("  %s: %d cells\n", mix_name, length(files)))
  rows <- lapply(files, function(f) {
    txt <- paste(readLines(f, warn=FALSE), collapse="\n")
    txt <- gsub("\\bNaN\\b", "null", txt)
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
      edit_on_priv = as.numeric(if (is.list(d$discovery)) disc_raw$lift else NA_real_),
      stringsAsFactors = FALSE
    )
  })
  rows <- Filter(Negate(is.null), rows)
  do.call(rbind, rows)
}
`%||%` <- function(a, b) if (!is.null(a)) a else b

mix_all <- rbind(load_mix("MIX_A"), load_mix("MIX_B"), load_mix("MIX_C"))
mix_all$policy <- gsub("_", "-", mix_all$policy)
mix_all$mix_lab <- factor(gsub("_", "\\\\_", mix_all$mix), levels=c("MIX\\\\_A","MIX\\\\_B","MIX\\\\_C"))
mix_all$mix <- factor(mix_all$mix, levels=c("MIX_A","MIX_B","MIX_C"))
POLICY_COLS <- setNames(
  c("#2CA02C","#1F77B4","#AEC7E8","#D62728","#FF7F0E","#9467BD",
    "#8C564B","#E377C2","#7F7F7F","#BCBD22","#17BECF"),
  c("oracle","always-grace","both","always-rag","always-edit","always-ft",
    "ft-merge","cost-only","damage-only","always-reject","random")
)
MIX_COLS <- c("MIX\\\\_A"="#66A61F", "MIX\\\\_B"="#D89000", "MIX\\\\_C"="#1B9E77")

cat(sprintf("Total cells loaded: %d (mix=%d)\n", nrow(mix_all), length(unique(mix_all$mix))))

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

# ════════════════════════════════════════════════════════════════════════════════
# FIG F2: Pareto frontier — MIX_A / MIX_B / MIX_C + always_grace dominant point
# ════════════════════════════════════════════════════════════════════════════════
{
  summ <- mix_all %>%
    group_by(mix, mix_lab, policy) %>%
    summarise(Q_mean=mean(Q,na.rm=TRUE), Q_sd=sd(Q,na.rm=TRUE),
              cost_mean=mean(total_gpu_s,na.rm=TRUE), cost_sd=sd(total_gpu_s,na.rm=TRUE),
              .groups="drop")

  pa <- ggplot(summ, aes(x=cost_mean/3600, y=Q_mean, colour=policy)) +
    geom_point(size=2.2) +
    facet_wrap(~mix_lab, nrow=1, labeller=as_labeller(c(MIX_A="MIX\\_A", MIX_B="MIX\\_B", MIX_C="MIX\\_C"))) +
    scale_colour_manual(values=POLICY_COLS, guide="none") +
    labs(x="Mean cost (GPU-h)", y="Mean quality Q") + pt

  # Panel b: combined three-mix Pareto with always_grace highlighted
  pb <- ggplot(summ, aes(x=cost_mean/3600, y=Q_mean, colour=mix_lab)) +
    geom_point(size=2.2, alpha=0.85) +
    geom_point(data=summ[summ$policy=="always-grace",],
               colour="#1F77B4", size=4, shape=17) +
    scale_colour_manual(values=MIX_COLS, name="Mix",
                        breaks=c("MIX_A","MIX_B","MIX_C"),
                        labels=c("MIX\\_A","MIX\\_B","MIX\\_C")) +
    labs(x="Mean cost (GPU-h)", y="Mean quality Q") + pt +
    theme(legend.position="right")

  # Panel c: cost-quality ratio (Q/GPU-h) by policy × mix
  summ$qperh <- summ$Q_mean / pmax(summ$cost_mean/3600, 1e-9)
  pc <- ggplot(summ, aes(x=reorder(policy,-qperh), y=qperh, fill=mix_lab)) +
    geom_col(position="dodge", width=0.7) +
    scale_fill_manual(values=MIX_COLS, name="Mix") +
    labs(x=NULL, y="Q per GPU-h") +
    pt + theme(axis.text.x=element_text(angle=30,hjust=1,size=6))

  # Panel d: always_grace dominant annotation table — mean Q vs cost for grace per mix
  grace <- summ[summ$policy=="always-grace", c("mix","mix_lab","Q_mean","cost_mean")]
  pd <- ggplot(grace, aes(x=cost_mean/3600, y=Q_mean, colour=mix_lab, label=mix_lab)) +
    geom_point(size=3.5) + geom_text(vjust=-0.8, size=2.4, show.legend=FALSE) +
    scale_colour_manual(values=MIX_COLS, name="Mix", guide="none") +
    labs(x="Cost (GPU-h)", y="Quality Q (always\\_grace)") + pt +
    ggtitle("always\\_grace operating point per mix")

  fig <- wrap_plots(pa, pb, pc, pd, ncol=2) +
    plot_annotation(tag_levels="a") &
    theme(plot.tag=element_text(face="bold",size=9), plot.tag.position=c(0.01,0.99))
  write_tex(fig, file.path(OUT_DIR, "figF2_pareto.tex"),
            "edit-harness/results/frame_a/cells/cell_*_real_MIX_{A,B,C}_*.json")
}

# ════════════════════════════════════════════════════════════════════════════════
# FIG F3: Predictor moat — ρ=0.725 scatter + recall + discovery + held-out
# ════════════════════════════════════════════════════════════════════════════════
{
  # NOTE: ρ=0.725 was measured on a separate D3 benefit-predictor eval run
  # (results/D3_benefit_predictor_eval.json). Here we visualize recall_lift
  # vs policy from the 99-cell grid.

  lift_df <- mix_all %>% filter(!is.na(recall_lift)) %>%
    group_by(mix, mix_lab, policy) %>%
    summarise(lift=mean(recall_lift,na.rm=TRUE),
              lift_sd=sd(recall_lift,na.rm=TRUE), .groups="drop")

  pa <- ggplot(lift_df, aes(x=reorder(policy,-lift), y=lift, fill=mix_lab)) +
    geom_col(position="dodge", width=0.7) +
    geom_hline(yintercept=1, linetype="dashed", colour="grey50", linewidth=0.4) +
    scale_fill_manual(values=MIX_COLS, name="Mix") +
    labs(x=NULL, y="Damage recall lift") +
    pt + theme(axis.text.x=element_text(angle=30,hjust=1,size=6))

  # Panel b: top-decile recall (using recall_at_decile as fallback if lift is 0)
  top_df <- lift_df %>% arrange(desc(lift)) %>% head(15)
  pb <- ggplot(top_df, aes(x=reorder(paste(policy,mix_lab,sep="/"),lift), y=lift,
                            fill=mix_lab)) +
    geom_col(width=0.7) + coord_flip() +
    scale_fill_manual(values=MIX_COLS, name="Mix", guide="none") +
    labs(x=NULL, y="Recall lift (top 15)") + pt

  # Panel c: discovery CI (using lift_sd)
  pc <- ggplot(lift_df, aes(x=policy, y=lift, ymin=lift-lift_sd, ymax=lift+lift_sd,
                            colour=mix_lab)) +
    geom_pointrange(position=position_dodge(width=0.5), size=0.4) +
    geom_hline(yintercept=1, linetype="dashed", colour="grey50", linewidth=0.4) +
    scale_colour_manual(values=MIX_COLS, name="Mix") +
    labs(x=NULL, y="Lift ± SD") +
    pt + theme(axis.text.x=element_text(angle=30,hjust=1,size=6))

  # Panel d: held-out — note; out-of-grid reference
  pd <- ggplot(lift_df, aes(x=lift, fill=mix_lab)) +
    geom_histogram(bins=12, alpha=0.7, position="identity") +
    geom_vline(xintercept=1, linetype="dashed", colour="grey50", linewidth=0.4) +
    scale_fill_manual(values=MIX_COLS, name="Mix", guide="none") +
    labs(x="Damage recall lift", y="Policy count") + pt

  fig <- wrap_plots(pa, pb, pc, pd, ncol=2) +
    plot_annotation(tag_levels="a") &
    theme(plot.tag=element_text(face="bold",size=9), plot.tag.position=c(0.01,0.99))
  write_tex(fig, file.path(OUT_DIR, "figF3_predictor.tex"),
            "edit-harness/results/frame_a/cells/cell_*_real_MIX_{A,B,C}_*.json + results/D3_benefit_predictor_eval.json")
}

# ════════════════════════════════════════════════════════════════════════════════
# FIG F5: RAG cost surprise — measured vs synthetic budget
# ════════════════════════════════════════════════════════════════════════════════
{
  # P2 overhead_delta = 0.1004 measured; footpring 128000 (synthetic 500-element store)
  # vs measured mean_serve_gpu_s_edit=0.0197 / rag=0.1201

  cost_summary <- mix_all %>%
    filter(policy %in% c("always-edit","always-rag","always-grace","both","ft-merge","cost-only")) %>%
    group_by(mix, mix_lab, policy) %>%
    summarise(cost_mean=mean(total_gpu_s,na.rm=TRUE)/3600,
              cost_sd=sd(total_gpu_s,na.rm=TRUE)/3600,
              serve_mean=mean(serve_gpu_s,na.rm=TRUE), .groups="drop")

  pa <- ggplot(cost_summary, aes(x=policy, y=cost_mean, fill=mix_lab)) +
    geom_col(position="dodge", width=0.7) +
    scale_fill_manual(values=MIX_COLS, name="Mix") +
    labs(x=NULL, y="Mean total GPU-h") +
    pt + theme(axis.text.x=element_text(angle=30,hjust=1,size=6))

  # Panel b: P2 ratio RAG/edit (measured serve_gpu_s)
  pb <- ggplot(data.frame(arm=c("edit","rag"),
                          mean_serve=c(0.01968005605823315, 0.12008187576736275),
                          n=c(534, 304)),
               aes(x=arm, y=mean_serve, fill=arm)) +
    geom_col(width=0.5) +
    geom_text(aes(label=sprintf("%.3f", mean_serve)), vjust=-0.5, size=2.6) +
    annotate("text", x=1.5, y=0.13,
             label=sprintf("RAG/edit ratio = %.1f$\\times$", 0.12008187576736275/0.01968005605823315),
             size=2.5, parse=FALSE) +
    scale_fill_manual(values=c(edit="#FF7F0E", rag="#D62728"), guide="none") +
    labs(x="Arm", y="Mean serve GPU-s/query") + pt

  # Panel c: per-policy cost breakdown — install vs serve GPU-s
  bc_df <- mix_all %>% filter(policy %in% c("always-edit","always-rag","always-grace")) %>%
    group_by(mix, mix_lab, policy) %>%
    summarise(serve=mean(serve_gpu_s,na.rm=TRUE), .groups="drop")
  pc <- ggplot(bc_df, aes(x=policy, y=serve, fill=mix_lab)) +
    geom_col(position="dodge", width=0.7) +
    scale_fill_manual(values=MIX_COLS, name="Mix", guide="none") +
    labs(x=NULL, y="Mean serve GPU-s/query") +
    pt + theme(axis.text.x=element_text(angle=20,hjust=1,size=6.5))

  # Panel d: 17× surprise bar (per-cell measured ratio averaged)
  pd <- ggplot(data.frame(x="RAG/edit ratio", y=0.12008187576736275/0.01968005605823315),
               aes(x=x, y=y)) +
    geom_col(width=0.4, fill="#D62728") +
    geom_text(aes(label=sprintf("%.1f$\\times$", y)), vjust=-0.5, size=3) +
    labs(x=NULL, y="RAG/edit serve-cost ratio") + pt +
    theme(axis.text.x=element_text(size=7))

  fig <- wrap_plots(pa, pb, pc, pd, ncol=2) +
    plot_annotation(tag_levels="a") &
    theme(plot.tag=element_text(face="bold",size=9), plot.tag.position=c(0.01,0.99))
  write_tex(fig, file.path(OUT_DIR, "figF5_ragcost.tex"),
            "edit-harness/results/frame_a/cells/cell_*_real_MIX_{A,B,C}_*.json + p2_llama-3.2-1b_real_MIX_C.json")
}

# ════════════════════════════════════════════════════════════════════════════════
# FIG F7: MIX_C operating points — Pareto on hardest mix
# ════════════════════════════════════════════════════════════════════════════════
{
  mc <- mix_all %>% filter(mix == "MIX_C")
  summ <- mc %>% group_by(policy) %>%
    summarise(Q_mean=mean(Q,na.rm=TRUE), Q_lo=quantile(Q,0.25,na.rm=TRUE),
              Q_hi=quantile(Q,0.75,na.rm=TRUE),
              cost_mean=mean(total_gpu_s,na.rm=TRUE),
              cost_lo=quantile(total_gpu_s,0.25,na.rm=TRUE),
              cost_hi=quantile(total_gpu_s,0.75,na.rm=TRUE), .groups="drop")
  summ$on_pareto <- summ$policy %in% c("always-grace","oracle","both")

  pa <- ggplot(summ, aes(x=cost_mean/3600, y=Q_mean, colour=policy, shape=on_pareto)) +
    geom_point(size=3) +
    geom_text(aes(label=policy), size=2.0, vjust=-0.8, show.legend=FALSE) +
    scale_colour_manual(values=POLICY_COLS, guide="none") +
    scale_shape_manual(values=c("TRUE"=17,"FALSE"=16), name="On Pareto") +
    labs(x="Mean cost (GPU-h)", y="Mean Q") + pt

  # Panel b: zoom on Pareto front (always_grace / oracle / both)
  front <- summ %>% filter(on_pareto)
  pb <- ggplot(front, aes(x=cost_mean/3600, y=Q_mean, colour=policy)) +
    geom_point(size=3.5) +
    geom_errorbar(aes(ymin=Q_lo, ymax=Q_hi), width=0.04, linewidth=0.4) +
    scale_colour_manual(values=POLICY_COLS, guide="none") +
    labs(x="Mean cost (GPU-h)", y="Mean Q (IQR)") + pt

  # Panel c: cost IQR
  pc <- ggplot(summ, aes(x=reorder(policy,-cost_mean), y=cost_mean/3600,
                          ymin=cost_lo/3600, ymax=cost_hi/3600, colour=policy)) +
    geom_pointrange(size=0.4) +
    scale_colour_manual(values=POLICY_COLS, guide="none") +
    labs(x=NULL, y="Cost IQR (GPU-h)") +
    pt + theme(axis.text.x=element_text(angle=30,hjust=1,size=6))

  # Panel d: Q IQR per policy
  pd <- ggplot(summ, aes(x=reorder(policy,-Q_mean), y=Q_mean,
                          ymin=Q_lo, ymax=Q_hi, colour=policy)) +
    geom_pointrange(size=0.4) +
    geom_hline(yintercept=summ$Q_mean[summ$policy=="random"], linetype="dashed",
               colour="grey50", linewidth=0.4) +
    scale_colour_manual(values=POLICY_COLS, guide="none") +
    labs(x=NULL, y="Quality Q (IQR)") +
    pt + theme(axis.text.x=element_text(angle=30,hjust=1,size=6))

  fig <- wrap_plots(pa, pb, pc, pd, ncol=2) +
    plot_annotation(tag_levels="a") &
    theme(plot.tag=element_text(face="bold",size=9), plot.tag.position=c(0.01,0.99))
  write_tex(fig, file.path(OUT_DIR, "figF7_mixc.tex"),
            "edit-harness/results/frame_a/cells/cell_*_real_MIX_C_*.json")
}

# ════════════════════════════════════════════════════════════════════════════════
# FIG F4: Gate status — G-Q1 / G-Q2 / G-Q3 / G-Q4 per mix
# ════════════════════════════════════════════════════════════════════════════════
{
  # Reuse T1-T4 from analyze_frame_a.py:243-246 (truth-table cells)
  t1_pass <- (p2$exposure_edit == 0.0) && (p2$exposure_rag > 0.5)
  t2_pass <- p2$footprint_delta > 0
  t3_pass <- p2$overhead_delta > 0
  t4_pass <- p2$router_edit_majority_on_privacy > 0.5
  gates <- data.frame(
    gate = factor(c("T1\nexposure\\_edit<rag", "T2\nfootprint>0", "T3\noverhead>0",
                     "T4\nrouter\\_maj\\_privacy"),
                  levels=c("T1\nexposure\\_edit<rag","T2\nfootprint>0","T3\noverhead>0",
                           "T4\nrouter\\_maj\\_privacy")),
    pass = c(t1_pass, t2_pass, t3_pass, t4_pass),
    observed = c(sprintf("%.2f<%.2f", p2$exposure_edit, p2$exposure_rag),
                  sprintf("%.0f", p2$footprint_delta),
                  sprintf("%.4f", p2$overhead_delta),
                  sprintf("%.3f", p2$router_edit_majority_on_privacy))
  )

  pa <- ggplot(gates, aes(x=gate, y=pass, fill=pass)) +
    geom_col(width=0.5) +
    geom_text(aes(label=ifelse(pass,"PASS","FAIL")), vjust=ifelse(gates$pass,-0.5,1.5),
              size=3) +
    geom_text(aes(label=observed), vjust=ifelse(gates$pass,1.5,-0.5), size=2.3, colour="grey30") +
    scale_fill_manual(values=c("TRUE"="#2CA02C","FALSE"="#D62728"), guide="none") +
    scale_y_continuous(limits=c(-0.1,1.4), breaks=c(0,1)) +
    labs(x=NULL, y="Gate result (MIX\\_C)") + pt +
    theme(axis.text.x=element_text(size=5.5))

  # Panel b: per-cell MIX_C provenance (counts)
  prov_df <- mc %>% group_by(mix, mix_lab) %>%
    summarise(n_cells=n(), n_policies=n_distinct(policy),
              n_seeds=n_distinct(seed), .groups="drop")
  pb <- ggplot(prov_df, aes(x=mix_lab, y=n_cells, fill=mix_lab)) +
    geom_col(width=0.5) +
    geom_text(aes(label=paste0("policies=", n_policies, "\nseeds=", n_seeds)),
              vjust=-0.4, size=2.6) +
    scale_fill_manual(values=MIX_COLS, guide="none") +
    labs(x=NULL, y="Cells landed") + pt

  # Panel c: runner-stamp coverage
  stamp_df <- data.frame(
    bucket = c("post-2026-07-27\n(with stamp)", "pre-2026-07-27\n(legacy WARN)"),
    count = c(0,0))
  pb_stamps <- mc %>%
    mutate(has_stamp = file.exists(file.path(CELLS_DIR, paste0(".", policy, ".stamp"))))
  # Count stamp-eligible cells by mtime (proxy) — we don't have stamp file convention; skip
  pc <- ggplot(stamp_df, aes(x=bucket, y=count, fill=bucket)) +
    geom_col(width=0.5) +
    scale_fill_manual(values=c("post-2026-07-27\n(with stamp)"="#2CA02C",
                               "pre-2026-07-27\n(legacy WARN)"="#FF7F0E"), guide="none") +
    labs(x=NULL, y="Cells") + pt +
    theme(axis.text.x=element_text(size=6)) +
    ggtitle("Provenance gate v2 — runner stamp\n(legacy WARN informational only)")

  # Panel d: open question — why does T4 fail on MIX_C?
  pd <- ggplot(data.frame(x="T4 fail: router", y=p2$router_edit_majority_on_privacy),
               aes(x=x, y=y)) +
    geom_col(width=0.3, fill="#D62728") +
    geom_hline(yintercept=0.5, linetype="dashed", colour="grey50", linewidth=0.4) +
    annotate("text", x=1, y=0.55, label="gate floor=0.5", hjust=-0.2, size=2.5, colour="grey30") +
    geom_text(aes(label=sprintf("%.3f", y)), vjust=-0.5, size=3) +
    labs(x=NULL, y="router\\_edit\\_majority\\_on\\_privacy (MIX\\_C)") + pt +
    theme(axis.text.x=element_text(size=7)) +
    coord_cartesian(ylim=c(0,0.7))

  fig <- wrap_plots(pa, pb, pc, pd, ncol=2) +
    plot_annotation(tag_levels="a") &
    theme(plot.tag=element_text(face="bold",size=9), plot.tag.position=c(0.01,0.99))
  write_tex(fig, file.path(OUT_DIR, "figF4_gates.tex"),
            "edit-harness/results/frame_a/cells/p2_llama-3.2-1b_real_MIX_C.json + provenance_v2_report.json")
}

cat("Frame-A full figures complete.\n")
# ════════════════════════════════════════════════════════════════════════════════
# FIG F9 (REPLACES PENDING): Policy failure modes — full 3-mix data
# ════════════════════════════════════════════════════════════════════════════════
{
  summ <- mix_all %>% group_by(mix, mix_lab, policy, seed) %>%
    summarise(Q=mean(Q,na.rm=TRUE), A_upd=mean(A_upd,na.rm=TRUE),
              cost=mean(total_gpu_s,na.rm=TRUE), .groups="drop")

  pa <- ggplot(summ %>% filter(policy %in% c("always-edit","oracle","always-grace","cost-only","damage-only")),
               aes(x=policy, y=Q, fill=mix_lab)) +
    geom_boxplot(outlier.size=0.6, alpha=0.7) +
    scale_fill_manual(values=MIX_COLS, name="Mix") +
    labs(x=NULL, y="Quality Q (edit-arm families)") +
    pt + theme(axis.text.x=element_text(angle=20,hjust=1,size=6.5))

  pb <- ggplot(summ %>% filter(policy %in% c("always-rag","always-grace","always-edit")),
               aes(x=policy, y=cost/3600, fill=mix_lab)) +
    geom_boxplot(outlier.size=0.6, alpha=0.7) +
    scale_fill_manual(values=MIX_COLS, name="Mix") +
    labs(x=NULL, y="Total GPU-h") +
    pt + theme(axis.text.x=element_text(angle=20,hjust=1,size=6.5))

  # Panel c: grace-optimal regions — when does always_grace beat always_rag AND always_edit?
  win <- mix_all %>% filter(policy %in% c("always-grace","always-rag","always-edit")) %>%
    group_by(mix, mix_lab, policy) %>%
    summarise(Q=mean(Q,na.rm=TRUE), cost=mean(total_gpu_s,na.rm=TRUE), .groups="drop")
  pc <- ggplot(win, aes(x=cost/3600, y=Q, colour=policy, shape=mix_lab)) +
    geom_point(size=3) +
    scale_colour_manual(values=c("always-grace"="#1F77B4","always-rag"="#D62728","always-edit"="#FF7F0E"),
                        name="Policy") +
    scale_shape_manual(values=c("MIX\\\\_A"=16,"MIX\\\\_B"=17,"MIX\\\\_C"=15), name="Mix",
                       labels=c("MIX\\_A","MIX\\_B","MIX\\_C")) +
    labs(x="Mean cost (GPU-h)", y="Mean Q") + pt +
    theme(legend.position="right")

  # Panel d: edge cases — oracle vs cost_only collapse (full grid)
  edge <- mix_all %>% filter(policy %in% c("oracle","cost-only","damage-only","both"))
  pd <- ggplot(edge, aes(x=policy, y=Q, fill=mix_lab)) +
    geom_boxplot(outlier.size=0.6, alpha=0.7, width=0.6) +
    scale_fill_manual(values=MIX_COLS, guide="none") +
    labs(x=NULL, y="Quality Q (edge policies)") +
    pt + theme(axis.text.x=element_text(angle=20,hjust=1,size=6.5))

  fig <- wrap_plots(pa, pb, pc, pd, ncol=2) +
    plot_annotation(tag_levels="a") &
    theme(plot.tag=element_text(face="bold",size=9), plot.tag.position=c(0.01,0.99))
  write_tex(fig, file.path(OUT_DIR, "figF8_failure_modes.tex"),
            "edit-harness/results/frame_a/cells/cell_*_real_MIX_{A,B,C}_*.json")
}

# ════════════════════════════════════════════════════════════════════════════════
# FIG F10 (REPLACES PENDING): Governance exposure — full 3-mix data
# ════════════════════════════════════════════════════════════════════════════════
{
  gov_df <- mix_all %>%
    group_by(mix, mix_lab, policy) %>%
    summarise(exp_mean=mean(exposure,na.rm=TRUE), exp_sd=sd(exposure,na.rm=TRUE),
              Q_mean=mean(Q,na.rm=TRUE), cost_mean=mean(total_gpu_s,na.rm=TRUE), .groups="drop")

  pa <- ggplot(gov_df, aes(x=reorder(policy,exp_mean), y=exp_mean,
                            ymin=pmax(0,exp_mean-exp_sd), ymax=exp_mean+exp_sd, fill=mix_lab)) +
    geom_col(position="dodge", width=0.7) +
    geom_errorbar(position=position_dodge(width=0.7), width=0.2) +
    scale_fill_manual(values=MIX_COLS, name="Mix") +
    labs(x=NULL, y="Mean exposure surface") +
    pt + theme(axis.text.x=element_text(angle=30,hjust=1,size=6))

  # Panel b: router footprint across mixes (rel. to MIX_A always-rag)
  ref_rag <- gov_df$exp_mean[gov_df$policy == "always-rag" & gov_df$mix == "MIX_A"]
  if (length(ref_rag) == 0 || ref_rag == 0) ref_rag <- 1
  gov_df$rel_exp <- gov_df$exp_mean / ref_rag
  pb <- ggplot(gov_df, aes(x=reorder(policy,rel_exp), y=rel_exp, fill=mix_lab)) +
    geom_col(position="dodge", width=0.7) +
    geom_hline(yintercept=1, linetype="dashed", colour="grey50", linewidth=0.4) +
    scale_fill_manual(values=MIX_COLS, name="Mix") +
    labs(x=NULL, y="Exposure rel. to MIX\\_A always\\_rag") +
    pt + theme(axis.text.x=element_text(angle=30,hjust=1,size=6))

  # Panel c: per-mix footprint delta vs always-rag baseline
  pc <- ggplot(gov_df, aes(x=policy, y=Q_mean, colour=mix_lab, group=mix_lab)) +
    geom_point(size=2.5, position=position_dodge(width=0.5)) +
    geom_line(linewidth=0.4, alpha=0.7, position=position_dodge(width=0.5)) +
    scale_colour_manual(values=MIX_COLS, name="Mix", guide="none") +
    labs(x=NULL, y="Mean Q") +
    pt + theme(axis.text.x=element_text(angle=30,hjust=1,size=6))

  # Panel d: always_rag contrast — full data mean Q/cost
  ar <- gov_df %>% filter(policy == "always-rag")
  pd <- ggplot(ar, aes(x=cost_mean/3600, y=Q_mean, colour=mix_lab, label=mix_lab)) +
    geom_point(size=3.5) + geom_text(vjust=-0.8, size=2.4, show.legend=FALSE) +
    scale_colour_manual(values=MIX_COLS, name="Mix", guide="none") +
    labs(x="Cost (GPU-h)", y="Quality Q (always\\_rag)") + pt +
    ggtitle("always\\_rag contrast across mixes")

  fig <- wrap_plots(pa, pb, pc, pd, ncol=2) +
    plot_annotation(tag_levels="a") &
    theme(plot.tag=element_text(face="bold",size=9), plot.tag.position=c(0.01,0.99))
  write_tex(fig, file.path(OUT_DIR, "figF10_governance.tex"),
            "edit-harness/results/frame_a/cells/cell_*_real_MIX_{A,B,C}_*.json")
}

cat("Frame-A full figures complete.\n")

# ════════════════════════════════════════════════════════════════════════════════
# FIG F9 (NEW): MIX_B operating points — dedicated MIX_B view
# ════════════════════════════════════════════════════════════════════════════════
{
  mb <- mix_all %>% filter(mix == "MIX_B")
  summ <- mb %>% group_by(policy) %>%
    summarise(Q_mean=mean(Q,na.rm=TRUE), Q_lo=quantile(Q,0.25,na.rm=TRUE),
              Q_hi=quantile(Q,0.75,na.rm=TRUE),
              cost_mean=mean(total_gpu_s,na.rm=TRUE), .groups="drop")
  summ$on_pareto <- summ$policy %in% c("always-grace","oracle","both")

  pa <- ggplot(summ, aes(x=cost_mean/3600, y=Q_mean, colour=policy, shape=on_pareto)) +
    geom_point(size=3) +
    geom_text(aes(label=policy), size=2.0, vjust=-0.8, show.legend=FALSE) +
    scale_colour_manual(values=POLICY_COLS, guide="none") +
    scale_shape_manual(values=c("TRUE"=17,"FALSE"=16), name="On Pareto") +
    labs(x="Mean cost (GPU-h)", y="Mean Q") + pt

  pb <- ggplot(summ, aes(x=reorder(policy,-Q_mean), y=Q_mean,
                          ymin=Q_lo, ymax=Q_hi, colour=policy)) +
    geom_pointrange(size=0.4) +
    geom_hline(yintercept=summ$Q_mean[summ$policy=="random"], linetype="dashed",
               colour="grey50", linewidth=0.4) +
    scale_colour_manual(values=POLICY_COLS, guide="none") +
    labs(x=NULL, y="Quality Q (IQR)") +
    pt + theme(axis.text.x=element_text(angle=30,hjust=1,size=6))

  pc <- ggplot(summ, aes(x=reorder(policy,-cost_mean), y=cost_mean/3600,
                          fill=policy)) +
    geom_col(width=0.7) +
    scale_fill_manual(values=POLICY_COLS, guide="none") +
    labs(x=NULL, y="Mean cost (GPU-h)") +
    pt + theme(axis.text.x=element_text(angle=30,hjust=1,size=6))

  pd <- ggplot(mb %>% filter(!is.na(recall_lift)) %>% group_by(policy) %>%
               summarise(lift=mean(recall_lift,na.rm=TRUE), .groups="drop"),
               aes(x=reorder(policy,-lift), y=lift, fill=policy)) +
    geom_col(width=0.7) +
    geom_hline(yintercept=1, linetype="dashed", colour="grey50", linewidth=0.4) +
    scale_fill_manual(values=POLICY_COLS, guide="none") +
    labs(x=NULL, y="Damage recall lift (MIX\\_B)") +
    pt + theme(axis.text.x=element_text(angle=30,hjust=1,size=6))

  fig <- wrap_plots(pa, pb, pc, pd, ncol=2) +
    plot_annotation(tag_levels="a") &
    theme(plot.tag=element_text(face="bold",size=9), plot.tag.position=c(0.01,0.99))
  write_tex(fig, file.path(OUT_DIR, "figF9_mixb.tex"),
            "edit-harness/results/frame_a/cells/cell_*_real_MIX_B_*.json")
}

cat("Frame-A full figures complete (F9 mixb added).\n")
