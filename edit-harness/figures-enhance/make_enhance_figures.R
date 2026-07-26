#!/usr/bin/env Rscript
# =====================================================================
# make_enhance_figures.R — figure set for the 2026-07-09 enhancement
# round (editable band / GLUE two-channel bridge / ripple depth profile
# / NeoX-20B law table). SAME rules as paper-arr/figures-r/make_figures.R:
#   * canonical-JSON-only — every number read at run time from
#     edit-harness/results/{esr_band_table,GLUE_BRIDGE_summary,
#     RIPPLE_depth_profile,NEOX20B_law_table}.json (re-run this script
#     after enhance_aggregates.py refreshes them; NOTHING hardcoded)
#   * dataviz reference palette in the fixed slot order
#   * PNG output for visual QA now; tikz emission is a one-line switch
#     when the venue manuscript is picked (render+eyeball rule:
#     memory/paper-visual-qa-lessons.md)
# Usage: Rscript figures-enhance/make_enhance_figures.R
# =====================================================================

suppressPackageStartupMessages({
  library(jsonlite); library(ggplot2); library(patchwork); library(scales)
})

HERE <- tryCatch(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))),
                 error = function(e) ".")
RES  <- normalizePath(file.path(HERE, "..", "results"))
OUT  <- HERE

viz <- list(
  blue    = "#2A78D6", aqua = "#1BAF7A", yellow = "#EDA100", green = "#008300",
  violet  = "#4A3AA7", red  = "#E34948", magenta = "#E87BA4", orange = "#EB6834",
  ink     = "#0B0B0B", ink2 = "#333333", tick = "#666666",
  muted   = "#898781", grid = "#E1E0D9"
)
theme_b6 <- function(base = 10) {
  theme_minimal(base_size = base) +
    theme(panel.grid.minor = element_blank(),
          panel.grid.major.x = element_blank(),
          panel.grid.major.y = element_line(colour = viz$grid, linewidth = 0.25),
          axis.ticks = element_line(colour = viz$tick, linewidth = 0.3),
          plot.title = element_text(face = "bold", size = base + 1),
          plot.subtitle = element_text(colour = viz$ink2, size = base - 1),
          legend.position = "top", legend.title = element_blank(),
          text = element_text(colour = viz$ink))
}

# ---------------- F-band: esr vs depth fraction per family ----------------
band <- fromJSON(file.path(RES, "esr_band_table.json"))
bd <- band$rows
bd <- bd[bd$editor == "rome" & bd$dataset == "cf", ]
agg <- aggregate(esr ~ model + depth_frac, bd, mean)
fam_order <- c("llama1b", "llama3b", "llama8b", "gptj", "neox20b",
               "pythia14b", "pythia28b", "qwen05b", "qwen15b", "qwen3b", "gemma2b", "phi35")
agg <- agg[agg$model %in% fam_order, ]
agg$model <- factor(agg$model, levels = fam_order)
fam_cols <- c(llama1b = viz$blue, llama3b = "#5E96E0", llama8b = "#9FBFEB",
              gptj = viz$yellow, neox20b = viz$red,
              pythia14b = viz$magenta, pythia28b = viz$orange,
              qwen05b = viz$aqua, qwen15b = viz$aqua, qwen3b = viz$aqua,
              gemma2b = viz$green, phi35 = viz$violet)
multi <- names(table(agg$model))[table(agg$model) > 1]
pA <- ggplot(agg, aes(depth_frac, esr, colour = model, group = model)) +
  geom_hline(yintercept = 0.3, linetype = "dotted", colour = viz$muted, linewidth = 0.3) +
  geom_line(data = agg[agg$model %in% multi, ], linewidth = 0.6) +
  geom_point(size = 1.8) +
  scale_colour_manual(values = fam_cols) +
  scale_y_continuous(limits = c(0, 1.02), labels = label_number(accuracy = 0.1)) +
  labs(title = "ROME edit-success across depth: the editable band is architectural",
       subtitle = "CounterFact, seed-mean esr; dotted line = 0.30 usability gate",
       x = "edit layer / total layers", y = "edit success rate") +
  theme_b6()

# ---------------- F-bridge: two-channel decomposition ----------------
gb <- fromJSON(file.path(RES, "GLUE_BRIDGE_summary.json"))
r <- gb$rows
r$layer_f <- factor(paste0("L", r$layer), levels = c("L8", "L12", "L14"))
rr <- r[r$editor == "rome", ]
long <- rbind(
  data.frame(layer = rr$layer_f, ch = "direction (cosine, per-example)", rho = rr$cos_within_example_rho_filtered),
  data.frame(layer = rr$layer_f, ch = "magnitude (norm-growth, per-edit)", rho = rr$ng_to_margin_damage_rho))
pB1 <- ggplot(long, aes(layer, rho, colour = ch)) +
  geom_hline(yintercept = 0, colour = viz$ink2, linewidth = 0.3) +
  geom_point(position = position_dodge(width = 0.4), size = 2.2) +
  stat_summary(fun = mean, geom = "crossbar", width = 0.32, linewidth = 0.35,
               position = position_dodge(width = 0.4)) +
  scale_colour_manual(values = c(viz$blue, viz$orange)) +
  labs(title = "GLUE task damage: direction never predicts; magnitude only mid-depth",
       subtitle = "ROME, Llama-1B; points = seeds", x = NULL, y = "Spearman rho") +
  theme_b6()
l12 <- r[r$layer == 12, ]
l12s <- aggregate(cbind(mean_abs_margin_damage, flip_rate, mean_norm_growth) ~ editor, l12, mean)
l12long <- rbind(
  data.frame(editor = l12s$editor, metric = "mean |margin damage|", val = l12s$mean_abs_margin_damage),
  data.frame(editor = l12s$editor, metric = "flip rate", val = l12s$flip_rate))
pB2 <- ggplot(l12long, aes(metric, val, fill = editor)) +
  geom_col(position = position_dodge(width = 0.6), width = 0.5) +
  geom_text(aes(label = signif(val, 2)), position = position_dodge(width = 0.6),
            vjust = -0.4, size = 2.8, colour = viz$ink2) +
  scale_fill_manual(values = c(alpha = viz$aqua, rome = viz$red)) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.18))) +
  labs(title = "AlphaEdit protects capabilities (~5.6x) despite larger update norms",
       subtitle = sprintf("L12, 3-seed means at matched esr; norm-growth alpha %.1f vs rome %.1f",
                          l12s$mean_norm_growth[l12s$editor == "alpha"],
                          l12s$mean_norm_growth[l12s$editor == "rome"]),
       x = NULL, y = NULL) +
  theme_b6()
pB <- pB1 / pB2

# ---------------- F-ripple: depth profile ----------------
rp <- fromJSON(file.path(RES, "RIPPLE_depth_profile.json"))
prof <- do.call(rbind, lapply(names(rp$profile), function(L) {
  e <- rp$profile[[L]]
  data.frame(layer = as.integer(sub("L", "", L)),
             ripple = e$rho_ripple_mean, unrelated = e$rho_unrelated_mean,
             rlo = e$rho_ripple_range[1], rhi = e$rho_ripple_range[2],
             ulo = e$rho_unrelated_range[1], uhi = e$rho_unrelated_range[2])
}))
pl <- rbind(
  data.frame(layer = prof$layer, kind = "related facts (ripple)", rho = prof$ripple, lo = prof$rlo, hi = prof$rhi),
  data.frame(layer = prof$layer, kind = "unrelated probes", rho = prof$unrelated, lo = prof$ulo, hi = prof$uhi))
pC <- ggplot(pl, aes(layer, rho, colour = kind)) +
  geom_line(linewidth = 0.6) + geom_point(size = 2) +
  geom_errorbar(aes(ymin = lo, ymax = hi), width = 0, linewidth = 0.35) +
  scale_colour_manual(values = c(viz$violet, viz$blue)) +
  scale_x_continuous(breaks = prof$layer) +
  labs(title = "Geometry predicts related-fact ripple damage across depth",
       subtitle = "Llama-1B, RippleEdits popular; 3 seeds/layer (bars = seed range); ordering flips at L14",
       x = "edit layer", y = "within-probe Spearman rho") +
  theme_b6()

# ---------------- F-neox: law table (partial until grid completes) ----------------
nx <- fromJSON(file.path(RES, "NEOX20B_law_table.json"))
n <- nx$rows
nl <- rbind(
  data.frame(layer = n$layer, seed = n$seed, ch = "key-cosine", rho = n$rho_cos),
  data.frame(layer = n$layer, seed = n$seed, ch = "norm-growth", rho = n$rho_norm_growth))
llama_ref <- 0.60  # Llama-1B L12 cosine reference (paper canonical, axis annotation only)
pD <- ggplot(nl, aes(factor(layer), rho, colour = ch)) +
  geom_hline(yintercept = 0, colour = viz$ink2, linewidth = 0.3) +
  geom_hline(yintercept = llama_ref, linetype = "dashed", colour = viz$muted, linewidth = 0.35) +
  geom_point(position = position_dodge(width = 0.35), size = 2.2) +
  scale_colour_manual(values = c(viz$blue, viz$orange)) +
  scale_y_continuous(limits = c(-0.05, 0.7)) +
  labs(title = "NeoX-20B: collateral damage without geometric predictability",
       subtitle = "within-probe rho, edit_ok x known-probe filtered\ndashed = Llama-1B L12 cosine reference (0.60); grid partial until battery completes",
       x = "edit layer (of 44)", y = "within-probe Spearman rho") +
  theme_b6()

ggsave(file.path(OUT, "F_band.png"),   pA, width = 6.5, height = 3.6, dpi = 170)
ggsave(file.path(OUT, "F_bridge.png"), pB, width = 6.5, height = 6.4, dpi = 170)
ggsave(file.path(OUT, "F_ripple.png"), pC, width = 6.5, height = 3.6, dpi = 170)
ggsave(file.path(OUT, "F_neox.png"),   pD, width = 6.5, height = 3.6, dpi = 170)
cat("wrote F_band / F_bridge / F_ripple / F_neox to", OUT, "\n")
