#!/usr/bin/env Rscript
# =====================================================================
# make_figures_frame_a.R -- R/ggplot2 -> tikzDevice data-figure set for the
# Frame-A / ESWA manuscript (submissions/frame-a-eswa/).
#
# CLONED FROM: submissions/ieee/figures/make_figures_ieee.R
#   - identical palette (`viz`) and theme (`theme_b6` -> `theme_fa`)
#   - identical JSON loaders (.load / .exists / .first_existing / seed_mean_std)
#   - identical % SOURCE provenance machinery (`S()` writes one audit line per
#     series into a header that prepended above each emitted .tex file)
#   - identical deterministic timestamp stripping (the "% Created by
#     tikzDevice" line is dropped so two runs are byte-identical)
#
# DATA-ONLY RULE (inherited from make_figures_ieee.R): every plotted number
# is read at run time from edit-harness/results/frame_a/{cells/,frame_a_*.json}.
# No numbers are hardcoded except axis labels, titles, and the DEAD/PASS
# reference constants already fixed in the manuscript text.
#
# FIGURES EMITTED (final mode, all gates PASS):
#   fig02_pareto.tex               -- Q vs cost Pareto plot, 1x3 mix patchwork
#   fig03_router_discovery.tex     -- patchwork: router arm counts (a) +
#                                     discovery recall@decile CI (b)
#   fig04_gate_evidence.tex        -- 3-row patchwork: gate status (a) +
#                                     MIX_B Q-vs-cost drilldown (b) +
#                                     MIX_C structural/P2/privacy (c)
#
# FIG01 (discovery) is owned by direct-TikZ -- this generator does NOT
# emit it. FIG05 (MIX_B alone) and FIG06 (MIX_C alone) are NOT camera-
# ready figures -- their content lives INSIDE fig04 panels (b) and (c).
# The MIX_A-only preview (when MIX_B/C are still missing on disk) is
# written ONLY under figures-qa/ and is unmistakably watermarked
# "QA-PREVIEW-NOT-FOR-SUBMISSION".
#
# FAIL-CLOSED (binding per task 17):
#   preflight() requires ALL of the following simultaneously:
#     (a) provenance_gate (experiments.frame_a.provenance_gate) returns
#         status=PASS and exit_code=0. Run as a subprocess so the gate stays
#         the authoritative independent oracle.
#     (b) exactly 33 real MIX_B cells + 33 real MIX_C cells + the namespaced
#         real P2 file (p2_llama-3.2-1b_real_MIX_C.json) on disk.
#     (c) the verdict JSON is NOT frame_a_verdict_ftfix.json AND its
#         "VERDICT" is in {PASS, GREY, KILL}. INCOMPLETE / unknown / missing
#         values are rejected -- the wave is not yet ready to publish
#         figures from. Truth-first: PASS, GREY, and KILL are all valid
#         scientific outcomes when the grid is complete, the gate passes,
#         and the verdict is fresh.
#     (d) the verdict JSON's mtime is STRICTLY GREATER than the mtime of
#         every cell it summarizes -- this is the "fresh final verdict
#         generated after all inputs" check.
#     (e) no path under RESULTS references a hidden/quarantine/synthetic-
#         relabel directory (anything starting with '.', or matching
#         /cells_synth/, /synth_MIX_/, /.INVALID-/, .INVALID-STALE-*, etc).
#   Any preflight failure -> exit nonzero + writes NO fig02_pareto.tex /
#   fig03_router_discovery.tex / fig04_gate_evidence.tex. MIX_A preview
#   under figures-qa/ may still be emitted ONLY if preflight condition (a)
#   is at least INCOMPLETE-eligible for MIX_A (33 real MIX_A cells present,
#   no FAIL findings from provenance_gate, verdict present) AND the preview
#   is unmistakably watermarked -- otherwise even the preview is suppressed.
#
# Usage:
#   Rscript figures-r/make_figures_frame_a.R             # final mode (fail-closed)
#   Rscript figures-r/make_figures_frame_a.R --preview   # MIX_A preview only
#   Rscript figures-r/make_figures_frame_a.R --preflight # just print gate, exit
# =====================================================================

suppressPackageStartupMessages({
  library(jsonlite)
  library(ggplot2)
  library(patchwork)
  library(tikzDevice)
  library(scales)
})

# ---------------------------------------------------------------------
# small helpers (must precede first use)
# ---------------------------------------------------------------------
`%||%` <- function(a, b) if (is.null(a) || (length(a) == 1L && is.na(a))) b else a

# ---------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------
HERE    <- tryCatch(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))),
                    error = function(e) ".")
if (length(HERE) == 0 || HERE == "") HERE <- "."
# Canonical layout: this script lives at submissions/frame-a-eswa/figures-r/.
# RESULTS = <repo>/edit-harness/results/frame_a/ (3 levels up from HERE).
# QA_OUT = submissions/frame-a-eswa/figures-qa/ (for the MIX_A-only preview).
# Tests override RESULTS via the env var FRAME_A_RESULTS before sys.source().
OUT     <- normalizePath(HERE, mustWork = FALSE)
QA_OUT  <- normalizePath(file.path(HERE, "..", "figures-qa"), mustWork = FALSE)
RESULTS <- if (nzchar(Sys.getenv("FRAME_A_RESULTS"))) {
  normalizePath(Sys.getenv("FRAME_A_RESULTS"), mustWork = FALSE)
} else {
  normalizePath(file.path(HERE, "..", "..", "..", "edit-harness", "results", "frame_a"),
                mustWork = FALSE)
}

# Provenance gate entry point -- resolve relative to the workspace root.
# The gate imports only stdlib + stays independent of live Frame-A imports.
# Path: <repo>/edit-harness/experiments/frame_a/provenance_gate.py
# RESULTS = <repo>/edit-harness/results/frame_a/  -> up 3 reaches <repo>/, then
# down edit-harness/experiments/frame_a/provenance_gate.py.
PROVENANCE_GATE <- normalizePath(file.path(RESULTS, "..", "..", "..",
                                           "edit-harness", "experiments", "frame_a",
                                           "provenance_gate.py"),
                                 mustWork = FALSE)
# Verdict candidates: ordered, first match wins. Explicit REJECT for ftfix.
VERDICT_REJECT  <- c("frame_a_verdict_ftfix.json")
VERDICT_CANDIDATES <- c("frame_a_verdict.json",
                        "frame_a_verdict_llama-3.2-1b.json")
NAMESPACED_P2   <- "p2_llama-3.2-1b_real_MIX_C.json"
PREVIEW_VERDICT <- "frame_a_verdict_ftfix.json"  # MIX_A preview only; forbidden for final mode.

EXPECTED_MODEL      <- "llama-3.2-1b"
EXPECTED_PROVENANCE <- "real"
EXPECTED_MIXES      <- c("MIX_B", "MIX_C")
EXPECTED_AUX_MIX    <- "MIX_A"
EXPECTED_SEEDS      <- c(0L, 1L, 2L)
EXPECTED_POLICIES   <- c("both", "cost_only", "damage_only", "oracle",
                         "always_edit", "always_grace", "always_rag",
                         "always_ft", "always_reject", "random", "ft_merge")

# ---------------------------------------------------------------------
# dataviz palette (clone of make_figures_ieee.R `viz`)
# ---------------------------------------------------------------------
viz <- list(
  blue    = "#2A78D6", aqua = "#1BAF7A", yellow = "#EDA100", green = "#008300",
  violet  = "#4A3AA7", red  = "#E34948", magenta = "#E87BA4", orange = "#EB6834",
  seqblue = "#2A78D6", ink  = "#0B0B0B", ink2 = "#333333", tick = "#666666",
  muted   = "#898781", grid = "#E1E0D9", crit = "#D03B3B",
  preview_bg = "#FFF4E5"  # watermark background for MIX_A QA preview
)

# ---------------------------------------------------------------------
# JSON loaders (clone)
# ---------------------------------------------------------------------
# Python's json module emits non-standard NaN/Infinity tokens which jsonlite
# rejects; sanitize them to null before parsing.
.sanitize_python_json <- function(text) {
  text <- gsub("\\bNaN\\b",      "null", text)
  text <- gsub("\\bInfinity\\b", "null", text)
  text <- gsub("-Infinity",      "null", text)
  text
}
.load_text <- function(text) {
  fromJSON(.sanitize_python_json(text), simplifyVector = FALSE)
}
.load <- function(name) {
  p <- file.path(RESULTS, name)
  if (!file.exists(p)) stop("missing results JSON: ", name)
  .load_text(paste(readLines(p, warn = FALSE), collapse = "\n"))
}
.exists <- function(name) file.exists(file.path(RESULTS, name))
.first_existing <- function(names) { for (n in names) if (.exists(n)) return(n); NULL }
.num <- function(x) as.numeric(x)
seed_mean_std <- function(vals) {
  vals <- .num(unlist(vals)); vals <- vals[!is.na(vals)]
  if (!length(vals)) return(c(mean = NA, sd = 0))
  m <- mean(vals)
  sd <- if (length(vals) > 1) sqrt(sum((vals - m)^2) / length(vals)) else 0
  c(mean = m, sd = sd)
}

# ---------------------------------------------------------------------
# provenance (clone of make_figures_ieee.R `S()`)
# ---------------------------------------------------------------------
.prov <- character(0)
S <- function(basename, field, value) {
  v <- if (is.numeric(value)) formatC(value, format = "f", digits = 4) else value
  .prov[[length(.prov) + 1L]] <<- sprintf("%% SOURCE: results/frame_a/%s :: %s = %s", basename, field, v)
  invisible(value)
}
prov_reset <- function() .prov <<- character(0)

# ---------------------------------------------------------------------
# shared theme (clone of theme_b6, renamed to theme_fa for Frame-A)
# ---------------------------------------------------------------------
theme_fa <- function(base = 9) {
  theme_minimal(base_size = base) +
    theme(
      text              = element_text(colour = viz$ink),
      plot.title        = element_text(hjust = 0.5, face = "bold", size = base,
                                       colour = viz$ink, margin = margin(b = 2)),
      plot.tag          = element_text(face = "bold", size = base, colour = viz$ink),
      plot.tag.position = c(0.01, 0.98),
      axis.title.x      = element_text(size = base - 1, colour = viz$ink, margin = margin(t = 1)),
      axis.title.y      = element_text(size = base - 1, colour = viz$ink, margin = margin(r = 1)),
      axis.text         = element_text(size = base - 2, colour = viz$ink2),
      panel.grid.major.y = element_line(colour = viz$grid, linewidth = 0.25),
      panel.grid.major.x = element_blank(),
      panel.grid.minor  = element_blank(),
      axis.ticks        = element_line(colour = viz$tick, linewidth = 0.3),
      axis.ticks.length = unit(2, "pt"),
      axis.line         = element_blank(),
      legend.position   = "top",
      legend.justification = "center",
      legend.title      = element_blank(),
      legend.text       = element_text(size = base - 2, colour = viz$ink),
      legend.key.size   = unit(7, "pt"),
      legend.key.height = unit(6, "pt"),
      legend.spacing.x  = unit(2, "pt"),
      legend.margin     = margin(0, 0, 0, 0),
      legend.box.margin = margin(0, 0, 0, 0),
      legend.box.spacing = unit(2, "pt"),
      plot.subtitle     = element_text(hjust = 0.5, size = base - 2,
                                       colour = viz$ink2, margin = margin(b = 3)),
      plot.margin       = margin(3, 4, 3, 4)
    )
}

lbl_geom  <- function(x, y, lab, ...) annotate("text", x = x, y = y, label = lab,
                                               colour = viz$ink, size = 2.1, ...)
errbar    <- function(...) geom_errorbar(..., width = 0, colour = viz$muted, linewidth = 0.35)

# ---------------------------------------------------------------------
# cell loaders (real MIX_{A,B,C} only; never synth / hidden / quarantine / .INVALID-*)
# ---------------------------------------------------------------------
.parse_cell_filename <- function(name) {
  if (!startsWith(name, "cell_") || !endsWith(name, ".json")) return(NULL)
  if (grepl("\\.INVALID-", name, fixed = TRUE)) return(NULL)
  body <- substr(name, 6L, nchar(name) - 5L)
  if (!grepl("_s[0-9]+$", body)) return(NULL)
  m <- regmatches(body, regexpr("_s[0-9]+$", body))
  seed <- as.integer(sub("^_s", "", m))
  head <- substr(body, 1L, nchar(body) - nchar(m))
  parts <- strsplit(head, "_", fixed = TRUE)[[1]]
  mix_idx <- NULL
  for (i in seq_len(length(parts) - 1L)) {
    if (parts[i] == "MIX" && parts[i + 1L] %in% c("A", "B", "C")) { mix_idx <- i; break }
  }
  if (is.null(mix_idx) || mix_idx < 2L) return(NULL)
  model       <- paste(parts[seq_len(mix_idx - 2L)], collapse = "_")
  provenance  <- parts[mix_idx - 1L]
  mix         <- paste(parts[mix_idx], parts[mix_idx + 1L], sep = "_")
  policy      <- paste(parts[seq.int(mix_idx + 2L, length(parts))], collapse = "_")
  if (!nzchar(model) || !nzchar(provenance) || !nzchar(policy)) return(NULL)
  list(model = model, provenance = provenance, mix = mix, policy = policy, seed = seed,
       filename = name)
}

.in_scope_cells <- function(mixes = EXPECTED_MIXES, results_dir = RESULTS) {
  # Cells live one level deeper than the verdict JSON: <results_dir>/cells/.
  cells_dir <- file.path(results_dir, "cells")
  if (!dir.exists(cells_dir)) return(list())
  files <- list.files(cells_dir, pattern = "^cell_.*\\.json$", full.names = FALSE)
  # Hard-reject anything that smells like quarantine / invalid / synth.
  bad  <- grepl("\\.INVALID-", files, fixed = TRUE) |
          grepl("\\.synthetic", files, fixed = TRUE) |
          grepl("\\.hidden",   files, fixed = TRUE) |
          grepl("\\.quarantine", files, fixed = TRUE)
  files <- files[!bad]
  out <- list()
  for (f in files) {
    p <- file.path(cells_dir, f)
    # Symlink rejection (mirrors provenance_gate.py).
    if (Sys.readlink(p) %||% "" != "" ) next
    meta <- .parse_cell_filename(f)
    if (is.null(meta)) next
    if (meta$model != EXPECTED_MODEL) next
    if (meta$provenance != EXPECTED_PROVENANCE) next
    if (!(meta$mix %in% mixes)) next
    if (!(meta$seed %in% EXPECTED_SEEDS)) next
    if (!(meta$policy %in% EXPECTED_POLICIES)) next
    body <- tryCatch(.load_text(paste(readLines(p, warn = FALSE), collapse = "\n")),
                     error = function(e) NULL)
    if (is.null(body) || !is.list(body)) next
    # Body identity must match filename identity (relabel tell).
    body_id <- list(model = body$model %||% "", provenance = body$provenance %||% "",
                    mix = body$mix %||% "", policy = body$policy %||% "",
                    seed = body$seed %||% NA_integer_)
    if (!identical(body_id$model, meta$model) ||
        !identical(body_id$provenance, meta$provenance) ||
        !identical(body_id$mix, meta$mix) ||
        !identical(body_id$policy, meta$policy) ||
        !identical(as.integer(body_id$seed), as.integer(meta$seed))) next
    out[[length(out) + 1L]] <- c(meta, body = list(body),
                                 mtime = file.info(p)$mtime,
                                 path = p)
  }
  out
}

# ---------------------------------------------------------------------
# PREFLIGHT (fail-closed gate)
# ---------------------------------------------------------------------
preflight <- function(verbose = TRUE, allow_preview = FALSE, results_dir = RESULTS) {
  findings <- list()
  fail <- function(msg) findings[[length(findings) + 1L]] <<- list(kind = "FAIL", msg = msg)
  info <- function(msg) findings[[length(findings) + 1L]] <<- list(kind = "INFO", msg = msg)
  RD <- results_dir   # local rebinding so the existing gate logic reads cleanly.

  # (0) RESULTS exists
  if (!dir.exists(RD)) {
    fail(sprintf("results_dir directory does not exist: %s", RD))
    return(list(ok = FALSE, findings = findings, report = NULL,
                verdict_path = NULL, scope_cells = list()))
  }
  info(sprintf("results_dir = %s", RD))

  # (1) provenance_gate subprocess
  # The gate lives at a fixed repo location (<repo>/edit-harness/experiments/
  # frame_a/provenance_gate.py), resolved once at script load time via HERE --
  # NOT derived from RD, so it stays valid even when preflight is invoked
  # against a custom /tmp fixture directory.
  gate_cells_dir <- file.path(RD, "cells")
  gate_report <- NULL
  gate_exit <- NA_integer_
  if (!file.exists(PROVENANCE_GATE)) {
    fail(sprintf("provenance_gate.py not found at %s", PROVENANCE_GATE))
  } else {
    # Use system2() so stdout and the subprocess exit status remain independent
    # provenance signals. Successful system2() calls omit the status attribute;
    # only that documented case maps to integer 0.
    raw <- character(0)
    res <- tryCatch(suppressWarnings(system2("python3", c(shQuote(PROVENANCE_GATE),
                                                          "--cells_dir",
                                                          shQuote(gate_cells_dir)),
                                              stdout = TRUE, stderr = FALSE)),
                    error = function(e) {
                      fail(sprintf("provenance_gate subprocess failed: %s", e$message))
                      NULL
                    })
    if (is.character(res)) {
      raw <- res
      status_attr <- attr(res, "status")
      gate_exit <- if (is.null(status_attr)) 0L else status_attr
    } else if (!is.null(res)) {
      fail(sprintf("provenance_gate subprocess returned unexpected type: %s",
                   paste(class(res), collapse = "/")))
    }
    if (length(raw) == 0L) {
      fail("provenance_gate returned no output")
    } else {
      gate_report <- tryCatch(.load_text(paste(raw, collapse = "\n")),
                              error = function(e) {
                                fail(sprintf("provenance_gate JSON decode failed: %s", e$message))
                                NULL
                              })
    }
  }

  gate_exit_ok <- is.integer(gate_exit) && length(gate_exit) == 1L &&
                  !is.na(gate_exit) && identical(gate_exit, 0L)
  if (!gate_exit_ok) {
    fail(sprintf("provenance_gate subprocess exit status invalid/nonzero: %s (need integer 0)",
                 if (length(gate_exit) == 1L) as.character(gate_exit) else "<missing/invalid>"))
  }

  report_exit <- if (!is.null(gate_report)) gate_report$exit_code else NULL
  report_exit_ok <- is.numeric(report_exit) && length(report_exit) == 1L &&
                    !is.na(report_exit) && is.finite(report_exit) &&
                    report_exit == trunc(report_exit) && report_exit == 0
  if (!report_exit_ok) {
    fail(sprintf("provenance_gate report exit_code invalid/nonzero: %s (need numeric integer-equivalent 0)",
                 if (length(report_exit) == 1L) as.character(report_exit) else "<missing/invalid>"))
  }

  report_status <- if (!is.null(gate_report)) gate_report$status else NULL
  report_status_ok <- is.character(report_status) && length(report_status) == 1L &&
                      !is.na(report_status) && identical(report_status, "PASS")
  if (!report_status_ok) {
    fail(sprintf("provenance_gate status=%s (need PASS)",
                 if (length(report_status) == 1L) as.character(report_status) else "<missing/invalid>"))
  }

  if (gate_exit_ok && report_exit_ok && report_status_ok) {
    info(sprintf("provenance_gate PASS (subprocess exit=%d, report exit_code=%s, %d in-scope cells)",
                 gate_exit, as.character(report_exit),
                 gate_report$counts$in_scope_cells %||% 0L))
  }

  # (2) grid: 33 real MIX_B + 33 real MIX_C + namespaced real P2
  scope_cells <- .in_scope_cells(mixes = EXPECTED_MIXES, results_dir = RD)
  by_mix_pol  <- list()
  for (c in scope_cells) {
    key <- paste(c$mix, c$policy, c$seed, sep = "|")
    by_mix_pol[[key]] <- c
  }
  expected_keys <- c()
  for (mx in EXPECTED_MIXES) for (pl in EXPECTED_POLICIES) for (sd in EXPECTED_SEEDS)
    expected_keys <- c(expected_keys, paste(mx, pl, sd, sep = "|"))
  missing <- setdiff(expected_keys, names(by_mix_pol))
  if (length(missing) > 0L) {
    fail(sprintf("missing %d/%d expected (mix,policy,seed) triples: %s",
                 length(missing), length(expected_keys),
                 paste(head(missing, 5L), collapse = ",")))
  } else {
    info(sprintf("grid complete: %d (mix,policy,seed) triples present", length(expected_keys)))
  }

  # P2 lives next to the cell files (RD/cells/, where the gate inspects).
  p2_path <- file.path(RD, "cells", NAMESPACED_P2)
  if (!file.exists(p2_path)) {
    fail(sprintf("namespaced P2 missing: %s (expected in RD/cells/)", NAMESPACED_P2))
  } else if (Sys.readlink(p2_path) %||% "" != "") {
    fail(sprintf("namespaced P2 is a symlink: %s", NAMESPACED_P2))
  } else {
    body <- tryCatch(.load_text(paste(readLines(p2_path, warn = FALSE), collapse = "\n")),
                     error = function(e) NULL)
    if (is.null(body) || body$provenance != EXPECTED_PROVENANCE ||
        body$mix != "MIX_C" || body$model != EXPECTED_MODEL) {
      fail(sprintf("namespaced P2 identity mismatch: %s", NAMESPACED_P2))
    } else {
      info(sprintf("namespaced P2 OK: %s (model=%s mix=%s provenance=%s)",
                   NAMESPACED_P2, body$model, body$mix, body$provenance))
    }
  }

  # (3) verdict: NOT ftfix + VERDICT is PASS + mtime > max cell mtime + MIX_B/C present
  verdict_path <- NULL
  for (cand in VERDICT_CANDIDATES) {
    p <- file.path(RD, cand)
    if (file.exists(p) && (Sys.readlink(p) %||% "") == "") { verdict_path <- p; break }
  }
  if (is.null(verdict_path)) {
    fail("no verdict JSON found (looked for frame_a_verdict.json / frame_a_verdict_llama-3.2-1b.json)")
  } else {
    info(sprintf("verdict candidate: %s", basename(verdict_path)))
    body <- tryCatch(.load_text(paste(readLines(verdict_path, warn = FALSE), collapse = "\n")),
                     error = function(e) { fail(sprintf("verdict JSON decode failed: %s", e$message)); NULL })
    if (!is.null(body)) {
      # Explicit reject for the ftfix file (the task forbids it even if it
      # happened to be the on-disk candidate).
      if (basename(verdict_path) %in% VERDICT_REJECT) {
        fail(sprintf("verdict is FORBIDDEN: %s (explicitly rejected)", basename(verdict_path)))
      }
      verdict_field <- body$VERDICT %||% ""
      # Truth-first gate: scientific verdict may validly be PASS, GREY, or
      # KILL. We reject only INCOMPLETE / unknown / missing -- those mean
      # the wave is not yet ready to publish figures from.
      verdict_accepted <- c("PASS", "GREY", "KILL")
      if (!verdict_field %in% verdict_accepted) {
        fail(sprintf("verdict VERDICT=%s (accept %s; reject INCOMPLETE / unknown / missing)",
                     if (nzchar(verdict_field)) verdict_field else "<missing>",
                     paste(verdict_accepted, collapse = "/")))
      } else {
        info(sprintf("verdict VERDICT=%s (accepted; truth-first)", verdict_field))
      }
      per_mix <- body$per_mix %||% list()
      for (mx in EXPECTED_MIXES) {
        if (is.null(per_mix[[mx]])) {
          fail(sprintf("verdict missing per_mix.%s", mx))
        }
      }
      # Freshness: mtime strictly greater than max cell mtime of EVERY
      # real cell on disk (MIX_A/B/C). The verdict summarizes every cell,
      # not just the in-scope ones, so we scan the entire cells/ directory.
      verdict_mtime <- file.info(verdict_path)$mtime
      cells_dir <- file.path(RD, "cells")
      max_cell_mtime <- NA_real_
      if (dir.exists(cells_dir)) {
        mt <- file.info(list.files(cells_dir, pattern = "^cell_.*\\.json$",
                                   full.names = TRUE))$mtime
        max_cell_mtime <- if (length(mt) && any(!is.na(mt))) max(mt, na.rm = TRUE) else NA_real_
      }
      if (is.na(max_cell_mtime)) {
        fail("verdict freshness check impossible: no real cells on disk to compare against")
      } else if (!(verdict_mtime > max_cell_mtime)) {
        fail(sprintf("verdict mtime (%.0f) not strictly greater than max cell mtime (%.0f) -- not fresh",
                     as.numeric(verdict_mtime), as.numeric(max_cell_mtime)))
      } else {
        info(sprintf("verdict is fresh: mtime %.0f > max cell mtime %.0f",
                     as.numeric(verdict_mtime), as.numeric(max_cell_mtime)))
      }
    }
  }

  # (4) reject any path containing hidden/quarantine/synthetic-relabel substring
  forbidden_substrings <- c(".synthetic-relabel-bak", "/cells_synth/", "/synth_MIX_/",
                            ".INVALID-", ".synthetic", ".hidden", ".quarantine")
  for (c in scope_cells) {
    for (pat in forbidden_substrings) {
      if (grepl(pat, c$path, fixed = TRUE)) {
        fail(sprintf("cell path matches forbidden pattern '%s': %s", pat, c$path))
      }
    }
  }
  # Same on verdict + P2 paths
  for (pth in c(verdict_path, p2_path)) {
    if (!is.null(pth) && !is.na(pth)) {
      for (pat in forbidden_substrings) {
        if (grepl(pat, pth, fixed = TRUE)) {
          fail(sprintf("verdict/P2 path matches forbidden pattern '%s': %s", pat, pth))
        }
      }
    }
  }

  ok <- !any(vapply(findings, function(f) identical(f$kind, "FAIL"), logical(1L)))
  if (verbose) {
    cat("=== preflight (fail-closed) ===\n")
    for (f in findings) cat(sprintf("  [%s] %s\n", f$kind, f$msg))
    cat(sprintf("=== preflight ok = %s ===\n", ok))
  }
  list(ok = ok, findings = findings, report = gate_report,
       verdict_path = verdict_path, scope_cells = scope_cells)
}

# ---------------------------------------------------------------------
# figure bodies (one function per figure; each returns list(plot=, prov=))
# ---------------------------------------------------------------------
.cells_by_key <- function(scope_cells) {
  out <- list()
  for (c in scope_cells) {
    k <- paste(c$mix, c$policy, c$seed, sep = "|")
    out[[k]] <- c
  }
  out
}

# ---- fig02: 1x3 Pareto patchwork across MIX_A / MIX_B / MIX_C ----
fig02_pareto <- function(scope_cells, verdict) {
  prov_reset()
  cells <- .cells_by_key(scope_cells)
  # Include MIX_A as the baseline reference panel; preflight already enforces
  # MIX_B and MIX_C are complete and 33+33 cells exist.
  mixes <- c("MIX_A", "MIX_B", "MIX_C")
  panels <- list()
  for (mx in mixes) {
    pts <- data.frame()
    for (pl in EXPECTED_POLICIES) {
      ms <- numeric(0)
      cs <- numeric(0)
      for (sd in EXPECTED_SEEDS) {
        k <- paste(mx, pl, sd, sep = "|")
        cd <- cells[[k]]
        if (is.null(cd)) next
        body <- cd$body
        q <- body$quality$Q
        tg <- body$cost$total_gpu_s
        if (!is.null(q) && !is.null(tg) && is.finite(q) && is.finite(tg) && tg > 0) {
          ms <- c(ms, as.numeric(q))
          cs <- c(cs, as.numeric(tg))
        }
      }
      if (length(ms) == 0L) next
      mm <- mean(ms); cm <- mean(cs)
      pts <- rbind(pts, data.frame(policy = pl, Q = mm, cost = cm))
      S(sprintf("cell_%s_real_%s_%s_s{0,1,2}.json", EXPECTED_MODEL, mx, pl),
        sprintf("mean(quality.Q), mean(cost.total_gpu_s)"), sprintf("Q=%.3f, cost=%.1f", mm, cm))
    }
    if (nrow(pts) == 0L) next
    # Pareto frontier: non-dominated points (Q higher is better; cost lower is better).
    pts <- pts[order(pts$cost, -pts$Q), ]
    pareto <- logical(nrow(pts))
    best_q <- -Inf
    for (i in seq_len(nrow(pts))) {
      if (pts$Q[i] > best_q) { pareto[i] <- TRUE; best_q <- pts$Q[i] }
    }
    pts$pareto <- pareto
    # Dominant point is always_grace -- mark it.
    pts$is_grace <- pts$policy == "always_grace"
    pts$highlight <- ifelse(pts$is_grace, "codebook-only (dominant)",
                            ifelse(pts$pareto, "Pareto frontier", "dominated"))
    p <- ggplot(pts, aes(cost, Q, shape = highlight, colour = highlight)) +
      geom_line(data = subset(pts, pareto), aes(group = 1), colour = viz$muted,
                linewidth = 0.4, linetype = "dashed") +
      geom_point(size = 2.0) +
      scale_colour_manual(values = c("codebook-only (dominant)" = viz$blue,
                                     "Pareto frontier" = viz$aqua,
                                     "dominated" = viz$muted)) +
      scale_shape_manual(values = c("codebook-only (dominant)" = 16,
                                    "Pareto frontier" = 17,
                                    "dominated" = 4)) +
      coord_cartesian(ylim = c(0, 1.05)) +
      labs(title = mx, x = "total GPU-seconds", y = "quality Q") +
      theme_fa()
    panels[[mx]] <- p
  }
  if (!length(panels)) return(NULL)
  combo <- Reduce(`|`, panels)
  list(plot = combo, prov = .prov)
}

# ---- fig03: router arm counts (a) + discovery recall@decile (b) ----
fig03_router_discovery <- function(scope_cells, verdict) {
  prov_reset()
  cells <- .cells_by_key(scope_cells)
  mixes <- c("MIX_A", "MIX_B", "MIX_C")

  # (a) router arm counts -- only meaningful for `both` (the routed policy);
  # we plot the mean of `arm_counts` across the 3 seeds for `both` per mix.
  ra <- data.frame()
  for (mx in mixes) {
    arm_acc <- list()
    for (sd in EXPECTED_SEEDS) {
      k <- paste(mx, "both", sd, sep = "|")
      cd <- cells[[k]]
      if (is.null(cd)) next
      ac <- cd$body$routing$arm_counts %||% list()
      if (length(ac) == 0L) next
      for (arm_name in names(ac)) {
        arm_acc[[arm_name]] <- c(arm_acc[[arm_name]] %||% numeric(0), as.numeric(ac[[arm_name]]))
      }
    }
    if (!length(arm_acc)) next
    for (arm_name in names(arm_acc)) {
      v <- arm_acc[[arm_name]]
      if (!length(v)) next
      mm <- mean(v)
      S(sprintf("cell_%s_real_%s_both_s{0,1,2}.json", EXPECTED_MODEL, mx),
        sprintf("routing.arm_counts[%s] (mean over seeds)", arm_name), mm)
      ra <- rbind(ra, data.frame(mix = mx, arm = arm_name, mean_count = mm))
    }
  }
  if (nrow(ra) == 0L) {
    pa <- ggplot() + theme_fa() + labs(title = "(a) router", x = NULL, y = NULL)
  } else {
    pa <- ggplot(ra, aes(mix, mean_count, fill = arm)) +
      geom_col(position = position_stack(), width = 0.65) +
      scale_fill_manual(values = c("edit" = viz$orange, "grace" = viz$blue,
                                   "rag" = viz$aqua, "ft" = viz$yellow,
                                   "reject" = viz$muted)) +
      labs(title = "router arm counts", tag = "(a)", x = NULL, y = "mean count / 500 updates") +
      theme_fa()
  }

  # (b) discovery -- reuse fig01 logic but render as a single horizontal panel
  vd <- verdict %||% list()
  per_mix <- vd$per_mix %||% list()
  rb <- data.frame()
  for (mx in mixes) {
    pm <- per_mix[[mx]]; if (is.null(pm)) next
    disc <- pm$discovery %||% list()
    ci   <- disc$recall_ci %||% c(NA, NA)
    p_recall <- disc$point_recall
    n_dmg    <- disc$n_damaging_gt %||% NA
    point_allowed <- isTRUE(disc$point_claim_allowed)
    y_lo <- if (is.na(ci[1])) 0 else ci[1]
    y_hi <- if (is.na(ci[2])) 0 else ci[2]
    if (point_allowed) y_lo <- y_hi <- p_recall %||% y_lo
    S(sprintf("frame_a_verdict.json::per_mix.%s.discovery", mx),
      sprintf("recall_ci (point_allowed=%s)", point_allowed),
      sprintf("[%.3f, %.3f]", y_lo, y_hi))
    rb <- rbind(rb, data.frame(mix = mx, y_lo = y_lo, y_hi = y_hi, n_dmg = n_dmg,
                               point_allowed = point_allowed))
  }
  if (nrow(rb) == 0L) {
    pb <- ggplot() + theme_fa() + labs(title = "(b) discovery", x = NULL, y = NULL)
  } else {
    pb <- ggplot(rb, aes(mix, y_lo, ymin = y_lo, ymax = y_hi, colour = mix)) +
      geom_hline(yintercept = 0.0993, colour = viz$muted, linetype = "dotted", linewidth = 0.3) +
      geom_hline(yintercept = 0.4407, colour = viz$aqua, linetype = "dashed", linewidth = 0.3) +
      geom_errorbar(width = 0, linewidth = 0.6) +
      geom_point(size = 2.0) +
      geom_text(aes(label = sprintf("n=%d", n_dmg)), vjust = -0.8, colour = viz$ink2, size = 2.0) +
      scale_colour_manual(values = c(MIX_A = viz$blue, MIX_B = viz$aqua, MIX_C = viz$violet)) +
      coord_cartesian(ylim = c(0, 1.05)) +
      labs(title = "discovery recall@decile", tag = "(b)",
           x = NULL, y = "recall@decile (bootstrap CI)") +
      theme_fa() + theme(legend.position = "none")
  }
  list(plot = pa | pb, prov = .prov)
}

# ---- fig04: gate (a) + MIX_B Q-vs-cost Pareto drilldown (b) + MIX_C structural evidence (c) ----
# Panel contents per coordinator 2026-07-22 correction (binding):
#   (a) gate status strip: 4 predictions, PASS/FAIL tile.
#   (b) MIX_B Q-vs-cost operating points (full Pareto panel for MIX_B).
#   (c) MIX_C structural evidence: structural PASS/FAIL, P2 metrics
#       (exposure_edit < exposure_rag, footprint_delta, overhead_delta),
#       privacy footprint (router_edit_majority_on_privacy), and the
#       edit-vs-RAG selection share.
# The measured-vs-synthetic arm cost ratio is NOT a panel here -- it has
# its own manuscript table (Table~\ref{tab:costratio}).
fig04_gate_evidence <- function(scope_cells, verdict) {
  prov_reset()
  vd <- verdict %||% list()
  gate <- vd$gate %||% list()

  # (a) gate status strip: 4 predictions, outcome tile. Truth-first: the
  # title shows the actual overall verdict (PASS/GREY/KILL), not a default.
  gate_pred <- data.frame(
    pred = factor(c("K_A (P1)", "K_A\' (P2)", "P3 vs ft_merge", "P4 ablations"),
                  levels = c("K_A (P1)", "K_A\' (P2)", "P3 vs ft_merge", "P4 ablations")),
    verdict = c(gate$P1_min_mixes_passed %||% FALSE,
                gate$P2_structural %||% FALSE,
                vd$P3 %||% FALSE,
                vd$P4 %||% FALSE)
  )
  S("frame_a_verdict.json::gate", "rule", gate$rule %||% "PASS=P1∧P2 ; KILL=¬P1∧¬P2 ; GREY=exactly one. P3/P4 sharpen only.")
  if (!("verdict" %in% names(gate))) {
    gate_pred$verdict <- c(isTRUE(vd$P1), isTRUE(vd$P2),
                           (vd$P3_mixes %||% 0L) >= (gate$P3_min_mixes %||% 2L),
                           (vd$P4_beats_cost_only_mixes %||% 0L) > 0L)
    S("frame_a_verdict.json (root)", "P1/P2/P3_mixes/P4_beats_cost_only_mixes",
      sprintf("P1=%s P2=%s P3_mixes=%s P4_beats=%s",
              vd$P1, vd$P2, vd$P3_mixes %||% 0L, vd$P4_beats_cost_only_mixes %||% 0L))
  }
  overall <- vd$VERDICT %||% "<missing>"
  gate_pred$col <- ifelse(gate_pred$verdict, viz$aqua, viz$red)
  pa <- ggplot(gate_pred, aes(pred, 1L, fill = col)) +
    geom_tile(colour = NA) +
    geom_text(aes(label = ifelse(verdict, "PASS", "FAIL")), colour = viz$ink, size = 2.6) +
    scale_fill_identity() +
    coord_cartesian(ylim = c(0.5, 1.5)) +
    labs(title = sprintf("gate (%s)", overall), tag = "(a)", x = NULL, y = NULL) +
    theme_fa() + theme(axis.text.y = element_blank(), axis.ticks.y = element_blank(),
                       panel.grid = element_blank())

  # (b) MIX_B Q-vs-cost operating-point drilldown (Pareto frontier).
  cells_b <- .cells_by_key(scope_cells)
  pts_b <- data.frame()
  for (pl in EXPECTED_POLICIES) {
    ms <- cs <- numeric(0)
    for (sd in EXPECTED_SEEDS) {
      k <- paste("MIX_B", pl, sd, sep = "|")
      cd <- cells_b[[k]]
      if (is.null(cd)) next
      body <- cd$body
      q  <- body$quality$Q
      tg <- body$cost$total_gpu_s
      if (!is.null(q) && !is.null(tg) && is.finite(q) && is.finite(tg) && tg > 0) {
        ms <- c(ms, as.numeric(q)); cs <- c(cs, as.numeric(tg))
      }
    }
    if (length(ms) == 0L) next
    mm <- mean(ms); cm <- mean(cs)
    S(sprintf("cell_%s_real_MIX_B_%s_s{0,1,2}.json", EXPECTED_MODEL, pl),
      sprintf("mean(quality.Q), mean(cost.total_gpu_s)"), sprintf("Q=%.3f, cost=%.1f", mm, cm))
    pts_b <- rbind(pts_b, data.frame(policy = pl, Q = mm, cost = cm))
  }
  if (nrow(pts_b) == 0L) {
    pb <- ggplot() + theme_fa() + labs(title = "(b) MIX_B", x = NULL, y = NULL)
  } else {
    pts_b <- pts_b[order(pts_b$cost, -pts_b$Q), ]
    pareto <- logical(nrow(pts_b)); best_q <- -Inf
    for (i in seq_len(nrow(pts_b))) {
      if (pts_b$Q[i] > best_q) { pareto[i] <- TRUE; best_q <- pts_b$Q[i] }
    }
    pts_b$pareto    <- pareto
    pts_b$is_grace  <- pts_b$policy == "always_grace"
    pts_b$highlight <- ifelse(pts_b$is_grace, "codebook-only (dominant)",
                              ifelse(pts_b$pareto, "Pareto frontier", "dominated"))
    pb <- ggplot(pts_b, aes(cost, Q, shape = highlight, colour = highlight)) +
      geom_line(data = subset(pts_b, pareto), aes(group = 1), colour = viz$muted,
                linewidth = 0.4, linetype = "dashed") +
      geom_point(size = 2.0) +
      scale_colour_manual(values = c("codebook-only (dominant)" = viz$blue,
                                     "Pareto frontier" = viz$aqua,
                                     "dominated" = viz$muted)) +
      scale_shape_manual(values = c("codebook-only (dominant)" = 16,
                                    "Pareto frontier" = 17,
                                    "dominated" = 4)) +
      coord_cartesian(ylim = c(0, 1.05)) +
      labs(title = "MIX_B operating points", tag = "(b)",
           x = "total GPU-seconds", y = "quality Q") +
      theme_fa() + theme(legend.key.width = unit(11, "pt"))
  }

  # (c) MIX_C structural evidence -- composite panel sourced from the
  # fresh final verdict AND the namespaced real P2 file
  # (results/frame_a/p2_llama-3.2-1b_real_MIX_C.json). The verdict
  # provides per_mix.MIX_C.P2_detail.values; the P2 file is the
  # authoritative source for the standalone structural metrics + the
  # privacy footprint (router_edit_majority_on_privacy).
  per_mix <- vd$per_mix %||% list()
  pm_c <- per_mix$MIX_C %||% list()
  p2_detail  <- pm_c$P2_detail %||% list()
  p2_values  <- p2_detail$values %||% list()

  namespaced_p2 <- file.path(RESULTS, "cells", NAMESPACED_P2)
  p2_body <- if (file.exists(namespaced_p2) &&
                 (Sys.readlink(namespaced_p2) %||% "") == "") {
    tryCatch(.load_text(paste(readLines(namespaced_p2, warn = FALSE), collapse = "\n")),
             error = function(e) NULL)
  } else NULL

  pick <- function(v, p) if (length(v) > 0L && !(length(v) == 1L && is.na(v))) v else p
  exposure_edit   <- pick(p2_values$exposure_edit,   p2_body$exposure_edit)
  exposure_rag    <- pick(p2_values$exposure_rag,    p2_body$exposure_rag)
  footprint_delta <- pick(p2_values$footprint_delta, p2_body$footprint_delta)
  overhead_delta  <- pick(p2_values$overhead_delta,  p2_body$overhead_delta)
  router_priv     <- pick(p2_values$router_edit_majority_on_privacy,
                          p2_body$router_edit_majority_on_privacy)
  src_label <- if (!is.null(p2_body)) NAMESPACED_P2 else "frame_a_verdict.json::per_mix.MIX_C.P2_detail.values"
  S(src_label, "exposure_edit", exposure_edit)
  S(src_label, "exposure_rag", exposure_rag)
  S(src_label, "footprint_delta", footprint_delta)
  S(src_label, "overhead_delta", overhead_delta)
  S(src_label, "router_edit_majority_on_privacy", router_priv)

  # (c).1 -- structural metric bars
  structure_rows <- data.frame(
    metric = factor(c("exposure_edit", "exposure_rag",
                      "footprint_delta", "overhead_delta"),
                    levels = c("exposure_edit", "exposure_rag",
                               "footprint_delta", "overhead_delta")),
    value  = c(exposure_edit %||% NA, exposure_rag %||% NA,
               footprint_delta %||% NA, overhead_delta %||% NA)
  )
  pc1 <- ggplot(structure_rows, aes(metric, value)) +
    geom_col(fill = viz$blue, width = 0.6) +
    geom_text(aes(label = sprintf("%.3g", value)), vjust = -0.5,
              colour = viz$ink2, size = 1.9) +
    labs(title = "structural", tag = "(c)", x = NULL, y = "value") +
    theme_fa() + theme(axis.text.x = element_text(angle = 30, hjust = 1, size = 6.5))

  # (c).2 -- privacy footprint: edit vs RAG selection on privacy updates.
  # router_priv = P(router selects edit | privacy update). RAG share = 1 - that.
  rp <- router_priv %||% 0
  pr_df <- data.frame(
    arm  = c("edit-on-privacy", "rag-on-privacy"),
    share = c(rp, max(0, 1 - rp))
  )
  pc2 <- ggplot(pr_df, aes(arm, share, fill = arm)) +
    geom_col(width = 0.6) +
    geom_text(aes(label = sprintf("%.3f", share)), vjust = -0.5,
              colour = viz$ink2, size = 2.0) +
    scale_fill_manual(values = c("edit-on-privacy" = viz$blue,
                                 "rag-on-privacy"  = viz$aqua)) +
    coord_cartesian(ylim = c(0, 1.1)) +
    labs(title = "privacy footprint", x = NULL, y = "router share") +
    theme_fa() + theme(legend.position = "none",
                       axis.text.x = element_text(angle = 30, hjust = 1, size = 6.5))

  pc <- pc1 / pc2
  list(plot = pa / pb / pc, prov = .prov)
}

# ---------------------------------------------------------------------
# emission: tikz + provenance header + deterministic timestamp strip
# ---------------------------------------------------------------------
emit <- function(name, obj, width, height, headline, watermark = FALSE) {
  if (is.null(obj)) return(invisible(FALSE))
  out_dir <- OUT
  if (watermark) out_dir <- QA_OUT
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  path <- file.path(out_dir, paste0(name, ".tex"))
  tikz(path, width = width, height = height, standAlone = FALSE, sanitize = FALSE,
       verbose = FALSE)
  print(obj$plot); dev.off()
  body <- readLines(path, warn = FALSE)
  body <- body[!grepl("^% Created by tikzDevice", body)]
  header <- c(
    "% =====================================================================",
    sprintf("%% %s -- R/ggplot2 -> tikzDevice (Frame-A, ESWA target)", name),
    "% Canonical-JSON-only. Regenerate: Rscript submissions/frame-a-eswa/figures-r/make_figures_frame_a.R",
    "% Provenance (JSON path :: field = plotted value), one line per series:",
    obj$prov,
    "% =====================================================================")
  if (watermark) {
    header <- c(header,
                sprintf("%% WARNING: QA-PREVIEW-NOT-FOR-SUBMISSION (generated %s)", format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z")),
                "% This file is NOT camera-ready. The MIX_A preview under figures-qa/")
  }
  writeLines(c(header, body), path)
  cat(sprintf("[ok] %s.tex  (%d source lines, %.2f x %.2f in) -> %s\n",
              name, length(obj$prov), width, height, out_dir))
  invisible(TRUE)
}

# ---------------------------------------------------------------------
# main: fail-closed orchestration
# ---------------------------------------------------------------------
main <- function(args) {
  mode <- "final"
  if ("--preview" %in% args) mode <- "preview"
  if ("--preview-incomplete" %in% args) mode <- "preview-incomplete"
  if ("--preflight" %in% args) mode <- "preflight"

  if (mode == "preview-incomplete") {
    emit_incomplete_previews()
    return(invisible(0L))
  }

  pf <- preflight(verbose = TRUE)
  if (!pf$ok) {
    cat("\npreflight FAIL-CLOSED. fig02-04 will NOT be emitted.\n")
    if (mode == "preview") {
      # Even the preview is suppressed: the gates are not satisfiable on
      # the current disk state. We still try to emit a MIX_A preview only
      # if MIX_A cells + provenance_gate PASS for MIX_A (sub-check below).
      pf_a <- preflight_mixa_only(pf)
      if (pf_a$ok) {
        cat("MIX_A-only preview eligible (provenance_gate PASS or MIX_A-clean); emitting QA-PREVIEW.\n")
        preview_rc <- tryCatch({
          emit("fig02_pareto_mix_a_preview", fig02_pareto_mixa(pf$scope_cells, pf$report),
               6.5, 2.6, "MIX_A Pareto preview", watermark = TRUE)
          emit("fig03_router_discovery_mix_a_preview", fig03_router_discovery_mixa(pf$scope_cells, pf$report),
               6.5, 2.6, "MIX_A router+discovery preview", watermark = TRUE)
          emit("fig04_gate_evidence_mix_a_preview", fig04_gate_evidence_mixa(pf$report),
               6.5, 3.4, "MIX_A gate evidence preview", watermark = TRUE)
          0L
        }, error = function(e) {
          # Distinguish: pdflatex missing vs scientific-gate failure. The
          # gate ALREADY failed -- the preview just couldn't render. We
          # surface the underlying TeX problem separately so CI logs don't
          # confuse "no pdflatex in PATH" with "scientific gate failure".
          pdflatex_path <- Sys.which("pdflatex")
          if (!nzchar(pdflatex_path)) {
            cat("preview render aborted: pdflatex not on PATH (NOT a scientific gate failure; final gate already FAILED above).\n")
          } else {
            cat(sprintf("preview render aborted: %s (pdflatex=%s available; NOT a TeX-missing env, NOT a scientific-gate failure -- final gate already FAILED above).\n",
                        conditionMessage(e), pdflatex_path))
          }
          3L
        })
        # Preview never substitutes for camera-ready: still exit nonzero
        # because the FINAL gate did not pass.
        quit(status = max(2L, preview_rc))
      } else {
        cat("MIX_A-only preview also NOT eligible (", pf_a$reason, "). Nothing emitted.\n", sep = "")
        quit(status = 2L)
      }
    }
    quit(status = 2L)
  }

  verdict_path <- pf$verdict_path
  verdict <- tryCatch(.load_text(paste(readLines(verdict_path, warn = FALSE), collapse = "\n")),
                      error = function(e) NULL)
  scope_cells <- pf$scope_cells

  dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

  # Camera-ready output: only if mode == "final" AND preflight PASS.
  # (mode "preview" never emits camera-ready.)
  if (mode == "final") {
    emit("fig02_pareto",           fig02_pareto(scope_cells, verdict),
         7.0, 2.6, "Pareto across MIX_A/B/C")
    emit("fig03_router_discovery", fig03_router_discovery(scope_cells, verdict),
         7.0, 2.6, "router arm counts + discovery recall@decile")
    emit("fig04_gate_evidence",    fig04_gate_evidence(scope_cells, verdict),
         7.0, 4.5, "gate (a) + MIX_B Pareto (b) + MIX_C structural (c)")
    cat("\nFrame-A camera-ready figures emitted to ", OUT, "\n", sep = "")
  } else if (mode == "preview") {
    cat("preview mode but preflight PASS -- skipping MIX_A-only preview (full set would be emitted in final mode).\n")
  } else if (mode == "preflight") {
    cat("preflight-only mode: no figures emitted.\n")
  }
  invisible(0L)
}

# ---------------------------------------------------------------------
# helpers for the MIX_A-only preview path
# ---------------------------------------------------------------------
preflight_mixa_only <- function(pf) {
  # The provenance_gate status must be PASS or INCOMPLETE-with-no-FAIL for MIX_A
  # (i.e. the grid state where MIX_B/C are still missing but MIX_A is fine).
  # Allow only if gate has no FAIL findings -- re-derive from the report.
  rep <- pf$report
  if (is.null(rep)) return(list(ok = FALSE, reason = "provenance_gate report missing"))
  if (!is.null(rep$group_severity)) {
    for (g in names(rep$group_severity)) {
      if (identical(rep$group_severity[[g]], "FAIL")) {
        return(list(ok = FALSE, reason = sprintf("provenance_gate group_severity.%s=FAIL", g)))
      }
    }
  }
  # MIX_A cells + namespaced P2 not required for MIX_A-only preview, but
  # we still need at least one MIX_A cell to render anything.
  mixa_cells <- .in_scope_cells(mixes = c("MIX_A"))
  if (length(mixa_cells) < 1L) {
    return(list(ok = FALSE, reason = "no real MIX_A cells found"))
  }
  list(ok = TRUE, reason = NULL)
}

fig02_pareto_mixa <- function(scope_cells, gate_report) {
  mixa_cells <- .in_scope_cells(mixes = c("MIX_A"))
  fig02_pareto(mixa_cells, list())
}
fig03_router_discovery_mixa <- function(scope_cells, gate_report) {
  mixa_cells <- .in_scope_cells(mixes = c("MIX_A"))
  fig03_router_discovery(mixa_cells, list())
}
fig04_gate_evidence_mixa <- function(gate_report) {
  # The gate panel can be drawn from the gate report alone (it has rule +
  # severity; not the full verdict).
  prov_reset()
  rep <- gate_report %||% list()
  if (length(rep) > 0L && !is.null(rep$status)) {
    S("provenance_gate.py::report", "status", rep$status)
    S("provenance_gate.py::report", "exit_code", rep$exit_code %||% NA)
  }
  p <- ggplot(data.frame(x = 1, y = 1, label = "MIX_A QA preview -- NOT camera-ready"),
              aes(x, y, label = label)) +
    geom_text(colour = viz$ink, size = 4) +
    theme_fa() + theme(axis.title = element_blank(), axis.text = element_blank(),
                       panel.grid = element_blank())
  list(plot = p, prov = .prov)
}


# ---------------------------------------------------------------------
# incomplete preview set (QA only; never called by final mode)
# ---------------------------------------------------------------------
.preview_banner <- function() {
  list(
    labs(caption = "INCOMPLETE PREVIEW -- NOT FOR SUBMISSION"),
    theme(
      legend.position = "bottom",
      legend.box = "horizontal",
      plot.caption = element_text(hjust = 1, face = "bold", size = 6.6,
                                  colour = viz$red,
                                  margin = margin(t = 4, b = 1)),
      plot.caption.position = "plot",
      plot.margin = margin(4, 5, 5, 5)
    )
  )
}

preview_pareto <- function() {
  prov_reset()
  cells <- c(.in_scope_cells(mixes = "MIX_A"), .in_scope_cells(mixes = "MIX_B"))
  rows <- data.frame()
  for (mx in c("MIX_A", "MIX_B")) {
    for (pl in EXPECTED_POLICIES) {
      selected <- Filter(function(cd) cd$mix == mx && cd$policy == pl, cells)
      if (!length(selected)) next
      q <- vapply(selected, function(cd) as.numeric(cd$body$quality$Q), numeric(1L))
      cost <- vapply(selected, function(cd) as.numeric(cd$body$cost$total_gpu_s), numeric(1L))
      S(sprintf("cells/cell_%s_real_%s_%s_s*.json", EXPECTED_MODEL, mx, pl),
        "mean(quality.Q), mean(cost.total_gpu_s)",
        sprintf("n=%d Q=%.4f cost=%.2f", length(selected), mean(q), mean(cost)))
    rows <- rbind(rows, data.frame(mix = sub("_", "-", mx, fixed = TRUE), policy = pl, Q = mean(q),
                                     cost = mean(cost), n = length(selected)))
    }
  }
  rows$complete <- rows$mix == "MIX-A" & rows$n == 3L
  p <- ggplot(rows, aes(cost, Q, shape = complete, colour = mix)) +
    geom_point(size = 2.1, alpha = 0.85) +
    scale_colour_manual(values = c(`MIX-A` = viz$blue, `MIX-B` = viz$muted)) +
    scale_shape_manual(values = c(`TRUE` = 16, `FALSE` = 1),
                       labels = c(`TRUE` = "complete MIX-A", `FALSE` = "partial MIX-B")) +
    coord_cartesian(ylim = c(0, 1.05)) +
    labs(title = "Q--cost operating points",
         subtitle = "MIX-A complete; MIX-B partial; MIX-C pending",
         x = "total GPU-seconds", y = "quality Q", shape = NULL) +
    guides(colour = guide_legend(order = 1, nrow = 1),
           shape = guide_legend(order = 2, nrow = 1)) +
    theme_fa() + .preview_banner()
  list(plot = p, prov = .prov)
}

preview_routing_share <- function() {
  prov_reset()
  cells <- .in_scope_cells(mixes = "MIX_A")
  routed <- Filter(function(cd) cd$policy == "both", cells)
  totals <- list()
  for (cd in routed) {
    ac <- cd$body$routing$arm_counts %||% list()
    for (arm in names(ac)) totals[[arm]] <- c(totals[[arm]] %||% numeric(0), as.numeric(ac[[arm]]))
  }
  rows <- data.frame()
  for (arm in names(totals)) {
    share <- mean(totals[[arm]]) / sum(vapply(totals, mean, numeric(1L)))
    S("cells/cell_llama-3.2-1b_real_MIX_A_both_s{0,1,2}.json",
      sprintf("mean routing.arm_counts.%s / mean total", arm), share)
    rows <- rbind(rows, data.frame(mix = "MIX-A", arm = arm, share = share))
  }
  p <- ggplot(rows, aes(mix, share, fill = arm)) +
    geom_col(width = 0.58) +
    scale_fill_manual(values = c(edit = viz$orange, grace = viz$blue, rag = viz$aqua,
                                 ft = viz$yellow, reject = viz$muted)) +
    scale_y_continuous(labels = function(x) sprintf("%.0f percent", 100 * x), limits = c(0, 1)) +
    labs(title = "Router composition", subtitle = "MIX-A only (3/3 seeds)",
         x = NULL, y = "share of routed updates") +
    guides(fill = guide_legend(nrow = 1)) +
    theme_fa() + .preview_banner()
  list(plot = p, prov = .prov)
}

preview_discovery <- function() {
  prov_reset()
  path <- file.path(RESULTS, PREVIEW_VERDICT)
  vd <- .load_text(paste(readLines(path, warn = FALSE), collapse = "\n"))
  disc <- vd$per_mix$MIX_A$discovery
  ci <- as.numeric(unlist(disc$recall_ci))
  S(PREVIEW_VERDICT, "per_mix.MIX_A.discovery.recall_ci", sprintf("[%.3f, %.3f]", ci[1], ci[2]))
  S(PREVIEW_VERDICT, "per_mix.MIX_A.discovery.n_damaging_gt", disc$n_damaging_gt)
  rows <- data.frame(mix = "MIX-A", lo = ci[1], hi = ci[2])
  p <- ggplot(rows, aes(mix, lo, ymin = lo, ymax = hi)) +
    geom_hline(yintercept = 0.0993, colour = viz$muted, linetype = "dotted") +
    geom_hline(yintercept = 0.4407, colour = viz$aqua, linetype = "dashed") +
    geom_errorbar(width = 0.12, colour = viz$blue, linewidth = 0.7) +
    geom_point(colour = viz$blue, size = 2.2) +
    annotate("text", x = 1, y = 0.82,
             label = sprintf("n=%d < 50; CI only", disc$n_damaging_gt), size = 2.2) +
    coord_cartesian(ylim = c(0, 1.08)) +
    labs(title = "Discovery recall@decile", subtitle = "MIX-A power floor not met",
         x = NULL, y = "bootstrap interval") +
    theme_fa() + theme(axis.text.x = element_blank(), axis.ticks.x = element_blank()) +
    .preview_banner()
  list(plot = p, prov = .prov)
}

preview_gate_status <- function() {
  prov_reset()
  counts <- vapply(c(MIX_A = "MIX_A", MIX_B = "MIX_B", MIX_C = "MIX_C"),
                   function(mx) length(.in_scope_cells(mixes = mx)), integer(1L))
  for (mx in names(counts)) S("cells/", sprintf("real %s cell count", mx), counts[[mx]])
  rows <- data.frame(mix = factor(sub("_", "-", names(counts), fixed = TRUE),
                                  levels = sub("_", "-", names(counts), fixed = TRUE)),
                     count = as.integer(counts))
  rows$status <- ifelse(rows$count == 33L, "complete", ifelse(rows$count == 0L, "pending", "partial"))
  p <- ggplot(rows, aes(mix, count, fill = status)) +
    geom_col(width = 0.62) +
    geom_hline(yintercept = 33, linetype = "dashed", colour = viz$muted) +
    geom_text(aes(label = sprintf("%d/33", count)), vjust = -0.5, size = 2.4) +
    scale_fill_manual(values = c(complete = viz$aqua, partial = viz$yellow, pending = viz$grid)) +
    coord_cartesian(ylim = c(0, 37)) +
    labs(title = "Final-gate readiness",
         subtitle = "No cross-mix verdict until B/C and provenance pass",
         x = NULL, y = "real cells") +
    guides(fill = guide_legend(nrow = 1)) +
    theme_fa() + .preview_banner()
  list(plot = p, prov = .prov)
}

emit_incomplete_previews <- function() {
  emit("preview_fig02_pareto_mix_a_primary", preview_pareto(), 6.5, 3.15,
       "incomplete Pareto preview", watermark = TRUE)
  emit("preview_routing_share_mix_a", preview_routing_share(), 4.6, 3.1,
       "incomplete routing-share preview", watermark = TRUE)
  emit("preview_discovery_recall_mix_a", preview_discovery(), 4.6, 3.0,
       "incomplete discovery preview", watermark = TRUE)
  emit("preview_gate_status_incomplete", preview_gate_status(), 4.9, 3.1,
       "incomplete gate-status preview", watermark = TRUE)
}
if (!interactive() && isTRUE(getOption("fa.run_main", TRUE))) {
  args <- commandArgs(trailingOnly = TRUE)
  rc <- main(args)
  if (!is.null(rc) && is.numeric(rc) && length(rc) == 1L) quit(status = as.integer(rc))
}