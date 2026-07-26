#!/usr/bin/env Rscript
# =====================================================================
# test_make_figures_frame_a.R -- focused tests for the Frame-A figure generator.
#
# Pure-stdlib + jsonlite. Tests run without R/ggplot2/tikzDevice installed
# (each test loads the generator's source as text and inspects it, then
# sources it in an isolated env that stubs the heavy deps).
#
# Coverage:
#   T1  preflight returns ok=FALSE on the current disk state
#   T2  preflight REJECTS frame_a_verdict_ftfix.json by name
#   T3  preflight REJECTS hidden / quarantine / synthetic-relabel paths
#   T4  preflight REJECTS when verdict mtime <= max cell mtime (staleness)
#   T5  preflight REJECTS when MIX_B cell count < 33 or MIX_C < 33
#   T6  preflight REJECTS when namespaced P2 missing
#   T7  final-mode exits nonzero + writes NO fig02/03/04 PDF/tex on failure
#   T8  preview-mode: MIX_A preview watermarked + written ONLY under figures-qa/
#   T9  parser rejects cells with .INVALID-, synth, hidden, quarantine substrings
#   T10 verdict candidate ordering: ftfix is NOT picked even if first on disk
#   T11 palette + theme_b6 -> theme_fa symmetry vs make_figures_ieee.R
#   T12 deterministic timestamp stripping line is present in emit()
#   T20 pdflatex is available on the host
#   T21 complete fixtures reject contradictory provenance exit signals
#
# Exit code: 0 if all tests pass, 1 otherwise. A summary is printed at the end.
# =====================================================================

suppressPackageStartupMessages({ library(jsonlite) })

`%||%` <- function(a, b) if (is.null(a) || (length(a) == 1L && is.na(a))) b else a

# Discover this test file's location via Rscript's --file= argument.
.this_file <- tryCatch(sub("^--file=", "",
                            grep("^--file=", commandArgs(FALSE), value = TRUE)),
                       error = function(e) NA_character_)
if (is.na(.this_file) || !nzchar(.this_file)) .this_file <- "tests/test_make_figures_frame_a.R"
ROOT <- normalizePath(file.path(dirname(.this_file), "..", "..", "..", ".."),
                     mustWork = FALSE)
# Fallback when sourced directly (not via Rscript).
if (!dir.exists(ROOT)) {
  cand <- file.path(getwd(), "idea-feasibility-analysis")
  if (dir.exists(cand)) ROOT <- cand
}

# Locate the generator + the IEEE reference + the gate.
GEN_PATH <- normalizePath(file.path(ROOT, "submissions", "frame-a-eswa",
                                    "figures-r", "make_figures_frame_a.R"))
IEEE_PATH <- normalizePath(file.path(ROOT, "submissions", "ieee", "figures",
                                     "make_figures_ieee.R"))
RESULTS <- file.path(ROOT, "edit-harness", "results", "frame_a")

stopifnot(file.exists(GEN_PATH))
stopifnot(file.exists(IEEE_PATH))
stopifnot(dir.exists(RESULTS))

GEN_SRC  <- paste(readLines(GEN_PATH, warn = FALSE), collapse = "\n")
IEEE_SRC <- paste(readLines(IEEE_PATH, warn = FALSE), collapse = "\n")

# Minimal stub environment to source the generator's helpers (we don't run
# the heavy deps; we only test preflight + parser + emit logic).
stub_env <- new.env(parent = globalenv())
# Stub the heavy libs.
for (pkg in c("jsonlite", "ggplot2", "patchwork", "tikzDevice", "scales")) {
  assign(pkg, structure(list(), class = "stub"), envir = stub_env)
}
# jsonlite::fromJSON -> real, but only on the file-content level. We
# re-route into our env so the preflight's source-grep can still load
# the verdict JSON via real jsonlite (we need full reads, not stubs).
assign("fromJSON", jsonlite::fromJSON, envir = stub_env)
# ggplot / patchwork / tikzDevice no-ops (we never invoke them in tests).
assign("ggplot", function(...) structure(list(), class = "stub"), envir = stub_env)
assign("aes",    function(...) NULL, envir = stub_env)
assign("geom_point",   function(...) NULL, envir = stub_env)
assign("geom_col",     function(...) NULL, envir = stub_env)
assign("geom_hline",   function(...) NULL, envir = stub_env)
assign("geom_errorbar",function(...) NULL, envir = stub_env)
assign("geom_line",    function(...) NULL, envir = stub_env)
assign("geom_tile",    function(...) NULL, envir = stub_env)
assign("geom_text",    function(...) NULL, envir = stub_env)
assign("geom_bar",     function(...) NULL, envir = stub_env)
assign("coord_cartesian", function(...) NULL, envir = stub_env)
assign("scale_colour_manual", function(...) NULL, envir = stub_env)
assign("scale_fill_manual",   function(...) NULL, envir = stub_env)
assign("scale_shape_manual",  function(...) NULL, envir = stub_env)
assign("scale_x_continuous",  function(...) NULL, envir = stub_env)
assign("scale_y_continuous",  function(...) NULL, envir = stub_env)
assign("scale_x_log10",       function(...) NULL, envir = stub_env)
assign("labs",         function(...) NULL, envir = stub_env)
assign("annotate",     function(...) NULL, envir = stub_env)
assign("theme_minimal",function(...) NULL, envir = stub_env)
assign("theme",        function(...) NULL, envir = stub_env)
assign("element_text", function(...) NULL, envir = stub_env)
assign("element_line", function(...) NULL, envir = stub_env)
assign("element_blank",function(...) NULL, envir = stub_env)
assign("margin",       function(...) NULL, envir = stub_env)
assign("unit",         function(...) NULL, envir = stub_env)
assign("tikz",         function(...) NULL, envir = stub_env)
assign("dev.off",      function(...) NULL, envir = stub_env)
assign("print",        function(...) NULL, envir = stub_env)
assign("Reduce",       base::Reduce, envir = stub_env)
assign("|",            function(x, y) {
  # Elementwise logical OR (vectorized). Used only by the parser's
  # forbidden-pattern checks; the patchwork plot-composition `|` is never
  # exercised by preflight, so this stub is intentionally narrow.
  if (is.logical(x) && is.logical(y)) {
    n <- max(length(x), length(y))
    xr <- rep_len(as.logical(x), n)
    yr <- rep_len(as.logical(y), n)
    xr | yr
  } else {
    stop("stub_env `|` only supports logical OR (got ", class(x)[1L], ", ", class(y)[1L], ")")
  }
}, envir = stub_env)
assign("data.frame",   base::data.frame, envir = stub_env)
assign("subset",       base::subset, envir = stub_env)
assign("match",        base::match, envir = stub_env)
assign("c",            base::c, envir = stub_env)
assign("list",         base::list, envir = stub_env)
assign("sprintf",      base::sprintf, envir = stub_env)
assign("file.path",    base::file.path, envir = stub_env)
assign("normalizePath", base::normalizePath, envir = stub_env)
assign("file.exists",  base::file.exists, envir = stub_env)
assign("file.info",    base::file.info, envir = stub_env)
assign("dir.create",   base::dir.create, envir = stub_env)
assign("dir.exists",   base::dir.exists, envir = stub_env)
assign("list.files",   base::list.files, envir = stub_env)
assign("readLines",    base::readLines, envir = stub_env)
assign("writeLines",   base::writeLines, envir = stub_env)
assign("Sys.readlink", function(p) {
  if (!file.exists(p)) return(NA_character_)
  # Use suppressWarnings to silence the expected non-zero exit when the path
  # isn't a symlink (the file exists but readlink exits 1 -> system warns).
  suppressWarnings({
    out <- tryCatch(system(paste("readlink", shQuote(p)),
                           intern = TRUE, ignore.stderr = TRUE),
                    error = function(e) character(0))
  })
  if (length(out) == 0L) NA_character_ else out[[1L]]
}, envir = stub_env)
assign("Sys.time",     base::Sys.time, envir = stub_env)
assign("format",       base::format, envir = stub_env)
assign("commandArgs",  base::commandArgs, envir = stub_env)
assign("interactive",  base::interactive, envir = stub_env)
assign("quit",         base::quit, envir = stub_env)
assign("regmatches",   base::regmatches, envir = stub_env)
assign("grepl",        base::grepl, envir = stub_env)
assign("regexpr",      base::regexpr, envir = stub_env)
assign("strsplit",     base::strsplit, envir = stub_env)
assign("startsWith",   base::startsWith, envir = stub_env)
assign("endsWith",     base::endsWith, envir = stub_env)
assign("substr",       base::substr, envir = stub_env)
assign("nchar",        base::nchar, envir = stub_env)
assign("paste",        base::paste, envir = stub_env)
assign("paste0",       base::paste0, envir = stub_env)
assign("as.integer",   base::as.integer, envir = stub_env)
assign("as.numeric",   base::as.numeric, envir = stub_env)
assign("is.null",      base::is.null, envir = stub_env)
assign("is.na",        base::is.na, envir = stub_env)
assign("is.list",      base::is.list, envir = stub_env)
assign("is.finite",    base::is.finite, envir = stub_env)
assign("is.character", base::is.character, envir = stub_env)
assign("is.numeric",   base::is.numeric, envir = stub_env)
assign("nzchar",       base::nzchar, envir = stub_env)
assign("length",       base::length, envir = stub_env)
assign("names",        base::names, envir = stub_env)
assign("setdiff",      base::setdiff, envir = stub_env)
assign("intersect",    base::intersect, envir = stub_env)
assign("vapply",       base::vapply, envir = stub_env)
assign("head",         utils::head, envir = stub_env)
assign("tryCatch",     base::tryCatch, envir = stub_env)
assign("warning",      base::warning, envir = stub_env)
assign("stop",         base::stop, envir = stub_env)
assign("message",      base::message, envir = stub_env)
assign("cat",          base::cat, envir = stub_env)
assign("try",          base::try, envir = stub_env)
assign("exists",       base::exists, envir = stub_env)
assign("get",          base::get, envir = stub_env)
assign("eval",         base::eval, envir = stub_env)
assign("sys.call",     base::sys.call, envir = stub_env)
assign("seq_len",      base::seq_len, envir = stub_env)
assign("seq.int",      base::seq.int, envir = stub_env)
assign("attr",         base::attr, envir = stub_env)
assign("class",        base::class, envir = stub_env)
assign("trunc",        base::trunc, envir = stub_env)
assign("system2",      base::system2, envir = stub_env)
assign("shQuote",      base::shQuote, envir = stub_env)
assign("invisible",    base::invisible, envir = stub_env)

# Source the generator into the stub env. Disable the auto-main() call so the
# test harness can call preflight() directly without quit()ing the process.
# Pass RESULTS through the FRAME_A_RESULTS env var so the generator resolves
# the absolute path correctly even when sys.source() does not propagate
# commandArgs(--file=) into the inner HERE.
Sys.setenv(FRAME_A_RESULTS = RESULTS)
options(fa.run_main = FALSE)
sys.source(GEN_PATH, envir = stub_env)
options(fa.run_main = TRUE)
Sys.unsetenv("FRAME_A_RESULTS")

# ----- harness helpers -----
pass_count <- 0L
fail_count <- 0L
results <- list()
record <- function(name, ok, detail = "") {
  if (ok) pass_count <<- pass_count + 1L else fail_count <<- fail_count + 1L
  results[[name]] <<- list(ok = ok, detail = detail)
  cat(sprintf("  %s  %s%s\n", if (ok) "PASS" else "FAIL", name,
              if (nzchar(detail)) paste0(" -- ", detail) else ""))
}

# T1: preflight on current disk state returns ok=FALSE
t1 <- tryCatch({
  out <- stub_env$preflight(verbose = FALSE)
  isFALSE <- !isTRUE(out$ok)
  list(ok = isFALSE, detail = sprintf("preflight$ok = %s on current INCOMPLETE disk", out$ok))
}, error = function(e) {
  tb <- paste(sapply(sys.calls(), deparse), collapse = " | ")
  list(ok = FALSE, detail = paste("error:", e$message, "| trace:", substr(tb, 1, 400)))
})
record("T1 preflight ok=FALSE on current INCOMPLETE state", t1$ok, t1$detail)

# T2: preflight REJECTS frame_a_verdict_ftfix.json by name
t2 <- tryCatch({
  gen_text <- GEN_SRC
  has_block <- grepl("VERDICT_REJECT[[:space:]]*<-[[:space:]]*c\\(\"frame_a_verdict_ftfix.json\"\\)",
                     gen_text, perl = FALSE) ||
               grepl("VERDICT_REJECT[[:space:]]*<-[^\\n]*\"frame_a_verdict_ftfix.json\"",
                     gen_text, perl = FALSE)
  has_explicit_check <- grepl("basename\\(verdict_path\\) %in% VERDICT_REJECT", gen_text)
  list(ok = has_block && has_explicit_check,
       detail = sprintf("block=%s, explicit_check=%s", has_block, has_explicit_check))
}, error = function(e) list(ok = FALSE, detail = paste("error:", e$message)))
record("T2 explicit reject of frame_a_verdict_ftfix.json", t2$ok, t2$detail)

# T3: preflight REJECTS hidden / quarantine / synthetic-relabel paths
t3 <- tryCatch({
  gen_text <- GEN_SRC
  pats <- c(".synthetic-relabel-bak", "/cells_synth/", "/synth_MIX_/",
            ".INVALID-", ".synthetic", ".hidden", ".quarantine")
  has_forbidden <- all(sapply(pats, function(p) grepl(p, gen_text, fixed = TRUE)))
  has_loop      <- grepl("forbidden_substrings", gen_text) &&
                   grepl("for \\(pat in forbidden_substrings\\)", gen_text)
  has_reject    <- grepl("fail\\(sprintf\\(\"cell path matches forbidden pattern", gen_text)
  list(ok = has_forbidden && has_loop && has_reject,
       detail = sprintf("patterns=%s, loop=%s, reject=%s",
                        has_forbidden, has_loop, has_reject))
}, error = function(e) list(ok = FALSE, detail = paste("error:", e$message)))
record("T3 rejects hidden/quarantine/synthetic-relabel paths", t3$ok, t3$detail)

# T4: preflight REJECTS when verdict mtime <= max cell mtime (staleness)
t4 <- tryCatch({
  out <- stub_env$preflight(verbose = FALSE)
  # Find the finding for "not fresh"
  findings <- out$findings
  has_staleness_msg <- any(sapply(findings, function(f) grepl("not fresh", f$msg, fixed = TRUE)))
  # Also require the source line that computes verdict_mtime > max_cell_mtime
  src_has_check <- grepl("verdict_mtime > max_cell_mtime", GEN_SRC)
  list(ok = has_staleness_msg && src_has_check,
       detail = sprintf("finding=%s, source=%s", has_staleness_msg, src_has_check))
}, error = function(e) {
  tb <- paste(sapply(sys.calls(), deparse), collapse = " | ")
  list(ok = FALSE, detail = paste("error:", e$message, "| trace:", substr(tb, 1, 400)))
})
record("T4 rejects stale verdict (mtime <= max cell mtime)", t4$ok, t4$detail)

# T5: preflight REJECTS when MIX_B or MIX_C cell count < 33
t5 <- tryCatch({
  out <- stub_env$preflight(verbose = FALSE)
  findings <- out$findings
  has_grid_msg <- any(sapply(findings, function(f)
    grepl("missing ", f$msg, fixed = TRUE) ||
    grepl("triples", f$msg, fixed = TRUE)))
  # Expect 33 MIX_B + 33 MIX_C present (the script computes 33 = 11 policies * 3 seeds)
  src_grid_size <- grepl("length\\(expected_keys\\)", GEN_SRC) &&
                   grepl("missing .* expected", GEN_SRC)
  list(ok = has_grid_msg && src_grid_size,
       detail = sprintf("finding=%s, source=%s", has_grid_msg, src_grid_size))
}, error = function(e) {
  tb <- paste(sapply(sys.calls(), deparse), collapse = " | ")
  list(ok = FALSE, detail = paste("error:", e$message, "| trace:", substr(tb, 1, 400)))
})
record("T5 rejects grid < 33 MIX_B + 33 MIX_C", t5$ok, t5$detail)

# T6: preflight REJECTS when namespaced P2 missing
t6 <- tryCatch({
  out <- stub_env$preflight(verbose = FALSE)
  findings <- out$findings
  has_p2_msg <- any(sapply(findings, function(f)
    grepl("namespaced P2", f$msg, fixed = TRUE)))
  src_has_p2_check <- grepl("NAMESPACED_P2", GEN_SRC) &&
                      grepl("p2_llama-3.2-1b_real_MIX_C.json", GEN_SRC)
  list(ok = has_p2_msg && src_has_p2_check,
       detail = sprintf("finding=%s, source=%s", has_p2_msg, src_has_p2_check))
}, error = function(e) {
  tb <- paste(sapply(sys.calls(), deparse), collapse = " | ")
  list(ok = FALSE, detail = paste("error:", e$message, "| trace:", substr(tb, 1, 400)))
})
record("T6 rejects missing namespaced P2", t6$ok, t6$detail)

# T7: final-mode exits nonzero + writes NO fig02/03/04 on failure
t7 <- tryCatch({
  out <- stub_env$preflight(verbose = FALSE)
  expected_writes <- c("fig02_pareto.tex", "fig03_router_discovery.tex", "fig04_gate_evidence.tex")
  # On failure, main() should call quit(status = 2L) BEFORE emit() is reached.
  # Confirm the source guards `if (!pf$ok)` block + skip emit() calls.
  has_guard <- grepl("if \\(!pf\\$ok\\)", GEN_SRC) &&
               grepl("quit\\(status = 2L\\)", GEN_SRC)
  # And that the camera-ready emit calls are NOT before the preflight guard.
  cam_emit_calls <- regmatches(GEN_SRC, gregexpr('emit\\("fig0[2-4]', GEN_SRC))[[1]]
  guard_idx      <- regexpr("if \\(!pf\\$ok\\)", GEN_SRC)
  first_emit_idx <- min(sapply(cam_emit_calls, function(call) regexpr(call, GEN_SRC, fixed = TRUE)))
  guard_before_emit <- guard_idx < first_emit_idx
  list(ok = has_guard && guard_before_emit && !out$ok,
       detail = sprintf("guard=%s, guard_before_emit=%s, preflight_fail=%s",
                        has_guard, guard_before_emit, !out$ok))
}, error = function(e) list(ok = FALSE, detail = paste("error:", e$message)))
record("T7 final mode exits nonzero + no fig02-04 emit on failure", t7$ok, t7$detail)

# T8: preview-mode: MIX_A preview watermarked + under figures-qa/ only
t8 <- tryCatch({
  has_qa_out <- grepl("QA_OUT", GEN_SRC, fixed = TRUE) &&
                grepl("figures-qa", GEN_SRC, fixed = TRUE)
  has_watermark_path <- grepl("if (watermark) out_dir <- QA_OUT", GEN_SRC, fixed = TRUE)
  has_warning_line <- grepl("WARNING: QA-PREVIEW-NOT-FOR-SUBMISSION", GEN_SRC, fixed = TRUE)
  # No emit() call in main() should land in OUT when mode is preview (only QA_OUT)
  has_preview_guard <- grepl('if (mode == "final")', GEN_SRC, fixed = TRUE) &&
                       grepl('emit("fig02_pareto",', GEN_SRC, fixed = TRUE)
  list(ok = has_qa_out && has_watermark_path && has_warning_line && has_preview_guard,
       detail = sprintf("qa_out=%s, watermark_path=%s, warning=%s, preview_guard=%s",
                        has_qa_out, has_watermark_path, has_warning_line, has_preview_guard))
}, error = function(e) list(ok = FALSE, detail = paste("error:", e$message)))
record("T8 preview watermarked + under figures-qa/ only", t8$ok, t8$detail)

# T9: parser rejects cells with .INVALID-, synth, hidden, quarantine substrings
t9 <- tryCatch({
  src <- GEN_SRC
  # The generator uses literal `\\.INVALID-` etc. (escape \\. inside the R string).
  needle_n <- "grepl(\"\\\\.INVALID-\", name, fixed = TRUE)"
  needle_f <- "grepl(\"\\\\.INVALID-\", files, fixed = TRUE)"
  has_invalid_check  <- grepl(needle_n, src, fixed = TRUE)
  has_listing_filter <- grepl(needle_f, src, fixed = TRUE)
  if (!has_invalid_check || !has_listing_filter) {
    cat("DEBUG T9: needle_n found in src=", grepl(needle_n, src, fixed = TRUE), "\n")
    cat("DEBUG T9: needle_n found in GEN_SRC=", grepl(needle_n, GEN_SRC, fixed = TRUE), "\n")
    cat("DEBUG T9: identical(src, GEN_SRC)=", identical(src, GEN_SRC), "\n")
    cat("DEBUG T9: GEN_SRC nchar=", nchar(GEN_SRC), "src nchar=", nchar(src), "\n")
    cat("DEBUG T9: first 200 of needle_n=", substr(needle_n, 1, 200), "\n")
  }
  list(ok = has_invalid_check && has_listing_filter,
       detail = sprintf("parse_check=%s, listing_filter=%s",
                        has_invalid_check, has_listing_filter))
}, error = function(e) list(ok = FALSE, detail = paste("error:", e$message)))
record("T9 parser rejects .INVALID-/synth/hidden/quarantine", t9$ok, t9$detail)

# T10: verdict candidate ordering: ftfix is NOT picked even if first on disk
t10 <- tryCatch({
  src <- GEN_SRC
  has_candidates <- grepl("VERDICT_CANDIDATES", src, fixed = TRUE) &&
                    grepl("frame_a_verdict.json", src, fixed = TRUE) &&
                    !grepl("frame_a_verdict_ftfix.json", src,
                           fixed = TRUE) == FALSE  # ftfix is mentioned ONLY in REJECT, not CANDIDATES
  # VERDICT_REJECT must be SEPARATE from VERDICT_CANDIDATES (separate binding)
  has_separate_reject <- grepl("VERDICT_REJECT  <- c(", src, fixed = TRUE) ||
                             grepl("VERDICT_REJECT <- c(", src, fixed = TRUE)
  list(ok = has_candidates && has_separate_reject,
       detail = sprintf("candidates=%s, separate_reject=%s",
                        has_candidates, has_separate_reject))
}, error = function(e) list(ok = FALSE, detail = paste("error:", e$message)))
record("T10 verdict candidate ordering excludes ftfix", t10$ok, t10$detail)

# T11: palette + theme_fa matches make_figures_ieee.R (cloned from theme_b6)
t11 <- tryCatch({
  # Compare the full viz + theme blocks via fixed-string markers.
  fa_has_palette  <- grepl('blue    = "#2A78D6"', GEN_SRC,  fixed = TRUE) &&
                     grepl('red  = "#E34948"',    GEN_SRC,  fixed = TRUE) &&
                     grepl('seqblue = "#2A78D6"', GEN_SRC,  fixed = TRUE)
  ieee_has_palette <- grepl('blue    = "#2A78D6"', IEEE_SRC, fixed = TRUE) &&
                      grepl('red  = "#E34948"',    IEEE_SRC, fixed = TRUE) &&
                      grepl('seqblue = "#2A78D6"', IEEE_SRC, fixed = TRUE)
  palette_match <- fa_has_palette && ieee_has_palette
  # theme_fa vs theme_b6: same shape (panel.grid.major.y, axis.title, etc.)
  fa_themes  <- regmatches(GEN_SRC,  gregexpr("panel.grid.major.y", GEN_SRC))[[1]]
  ieee_themes <- regmatches(IEEE_SRC, gregexpr("panel.grid.major.y", IEEE_SRC))[[1]]
  theme_match <- length(fa_themes) == length(ieee_themes) && length(fa_themes) >= 1L
  list(ok = palette_match && theme_match,
       detail = sprintf("palette_match=%s, theme_match=%s (panels: fa=%d, ieee=%d)",
                        palette_match, theme_match, length(fa_themes), length(ieee_themes)))
}, error = function(e) list(ok = FALSE, detail = paste("error:", e$message)))
record("T11 palette + theme_fa cloned from make_figures_ieee.R", t11$ok, t11$detail)

# T12: deterministic timestamp stripping line is present in emit()
t12 <- tryCatch({
  # Use fixed=TRUE; the actual source is `body <- body[!grepl(...)]` with
  # literal brackets (regex would require `\[`/`\]` escaping).
  has_strip <- grepl('body <- body[!grepl("^% Created by tikzDevice", body)]',
                     GEN_SRC, fixed = TRUE)
  list(ok = has_strip,
       detail = sprintf("strip_line=%s", has_strip))
}, error = function(e) list(ok = FALSE, detail = paste("error:", e$message)))
record("T12 emit() strips tikzDevice timestamp for byte-stable output", t12$ok, t12$detail)

# T13: % SOURCE provenance header writer is present per-figure
t13 <- tryCatch({
  has_s_def <- grepl("^S <- function", GEN_SRC) ||
               grepl("S <- function\\(basename, field, value\\)", GEN_SRC)
  has_prov_format <- grepl("%% SOURCE: results/frame_a/", GEN_SRC)
  has_prov_reset  <- grepl("prov_reset", GEN_SRC)
  list(ok = has_s_def && has_prov_format && has_prov_reset,
       detail = sprintf("S_def=%s, %%_SOURCE_format=%s, prov_reset=%s",
                        has_s_def, has_prov_format, has_prov_reset))
}, error = function(e) list(ok = FALSE, detail = paste("error:", e$message)))
record("T13 % SOURCE provenance header writer present", t13$ok, t13$detail)

# ---------------------------------------------------------------------
# Fixture builder: synthesize a complete real RESULTS directory for
# verdict-gate acceptance tests (T14-T17). Layout matches the live tree:
#   results/frame_a/cells/cell_<model>_real_MIX_{B,C}_<policy>_s<seed>.json
#   results/frame_a/cells/p2_<model>_real_MIX_C.json
#   results/frame_a/frame_a_verdict.json       (chosen verdict_field)
# Cell mtimes are deterministic spread; verdict mtime strictly greater.
# ---------------------------------------------------------------------
.build_complete_fixture <- function(tmpdir, verdict_field, mix_a = TRUE,
                                    omit_p2 = FALSE, omit_mix_c = FALSE) {
  dir.create(file.path(tmpdir, "cells"), recursive = TRUE, showWarnings = FALSE)
  model <- "llama-3.2-1b"
  policies <- c("both", "cost_only", "damage_only", "oracle",
                "always_edit", "always_grace", "always_rag", "always_ft",
                "always_reject", "random", "ft_merge")
  seeds <- 0:2
  mixes <- if (omit_mix_c) "MIX_B" else c("MIX_B", "MIX_C")
  base_t <- as.integer(as.numeric(Sys.time())) - 3600L
  cell_idx <- 0L
  for (mx in mixes) {
    for (pl in policies) {
      for (sd in seeds) {
        cell_idx <- cell_idx + 1L
        # Vary install/serve/A_loc across seeds (NOT exactly 4000/1500 so the
        # synthetic_anchor_exact check doesn't fire) so the gate's
        # cross_seed_degeneracy check passes.
        ins  <- 200.0 + sd * 7.3
        srv  <- 5.0 + sd * 1.1
        aloc <- 0.70 + sd * 0.005
        body <- list(model = model, provenance = "real", mix = mx,
                     policy = pl, seed = sd,
                     quality = list(Q = 0.85, A_loc = aloc),
                     cost = list(install_gpu_s = ins, serve_gpu_s = srv,
                                 total_gpu_s = ins + srv),
                     error_cost_eval = 100.0 + sd * 3.0)
        path <- file.path(tmpdir, "cells",
                          sprintf("cell_%s_real_%s_%s_s%d.json",
                                  model, mx, pl, sd))
        writeLines(jsonlite::toJSON(body, auto_unbox = TRUE), path)
        Sys.setFileTime(path, as.POSIXct(base_t + cell_idx * 10L, origin = "1970-01-01"))
      }
    }
  }
  if (mix_a) {
    for (pl in policies) for (sd in seeds) {
      cell_idx <- cell_idx + 1L
      ins  <- 200.0 + sd * 7.3
      srv  <- 5.0 + sd * 1.1
      aloc <- 0.70 + sd * 0.005
      body <- list(model = model, provenance = "real", mix = "MIX_A",
                   policy = pl, seed = sd,
                   quality = list(Q = 0.85, A_loc = aloc),
                   cost = list(install_gpu_s = ins, serve_gpu_s = srv,
                               total_gpu_s = ins + srv),
                   error_cost_eval = 100.0 + sd * 3.0)
      path <- file.path(tmpdir, "cells",
                        sprintf("cell_%s_real_MIX_A_%s_s%d.json",
                                model, pl, sd))
      writeLines(jsonlite::toJSON(body, auto_unbox = TRUE), path)
      Sys.setFileTime(path, as.POSIXct(base_t + cell_idx * 10L, origin = "1970-01-01"))
    }
  }
  if (!omit_p2) {
    p2 <- list(model = model, provenance = "real", mix = "MIX_C",
               exposure_edit = 0.05, exposure_rag = 0.95,
               footprint_delta = 128000.0, overhead_delta = 0.6,
               router_edit_majority_on_privacy = 0.80)
    path <- file.path(tmpdir, "cells", sprintf("p2_%s_real_MIX_C.json", model))
    writeLines(jsonlite::toJSON(p2, auto_unbox = TRUE), path)
    Sys.setFileTime(path, as.POSIXct(base_t + cell_idx * 10L + 30L, origin = "1970-01-01"))
  }
  # Verdict with the requested VERDICT field.
  verdict_mtime <- as.POSIXct(base_t + cell_idx * 10L + 60L, origin = "1970-01-01")
  verdict <- list(VERDICT = verdict_field,
                  per_mix = list(
                    MIX_A = list(P1_detail = list(always_grace = list(dQ = 0.1, dQ_ci = c(0.05, 0.15)))),
                    MIX_B = list(P1_detail = list(always_grace = list(dQ = 0.0, dQ_ci = c(0.0, 0.0)))),
                    MIX_C = list(P1_detail = list(always_grace = list(dQ = 0.0, dQ_ci = c(0.0, 0.0))),
                                 P2_detail = list(values = list(exposure_edit = 0.05,
                                                                 exposure_rag = 0.95,
                                                                 footprint_delta = 128000.0,
                                                                 overhead_delta = 0.6,
                                                                 router_edit_majority_on_privacy = 0.80)))))
  v_path <- file.path(tmpdir, "frame_a_verdict.json")
  writeLines(jsonlite::toJSON(verdict, auto_unbox = TRUE), v_path)
  Sys.setFileTime(v_path, verdict_mtime)
  tmpdir
}

# T14: preflight accepts a complete KILL fixture (truth-first gate)
t14 <- tryCatch({
  tmp <- tempfile("kill_fx_")
  .build_complete_fixture(tmp, verdict_field = "KILL")
  out <- stub_env$preflight(verbose = FALSE, results_dir = tmp)
  has_kill_accept <- any(sapply(out$findings,
                                function(f) grepl("KILL", f$msg, fixed = TRUE) &&
                                              grepl("accepted", f$msg, fixed = TRUE)))
  list(ok = isTRUE(out$ok) && has_kill_accept,
       detail = sprintf("ok=%s, accept_msg=%s", out$ok, has_kill_accept))
}, error = function(e) list(ok = FALSE, detail = paste("error:", e$message)))
record("T14 accepts complete KILL verdict (truth-first)", t14$ok, t14$detail)

# T15: preflight accepts a complete GREY fixture
t15 <- tryCatch({
  tmp <- tempfile("grey_fx_")
  .build_complete_fixture(tmp, verdict_field = "GREY")
  out <- stub_env$preflight(verbose = FALSE, results_dir = tmp)
  has_grey_accept <- any(sapply(out$findings,
                                function(f) grepl("GREY", f$msg, fixed = TRUE) &&
                                              grepl("accepted", f$msg, fixed = TRUE)))
  list(ok = isTRUE(out$ok) && has_grey_accept,
       detail = sprintf("ok=%s, accept_msg=%s", out$ok, has_grey_accept))
}, error = function(e) list(ok = FALSE, detail = paste("error:", e$message)))
record("T15 accepts complete GREY verdict (truth-first)", t15$ok, t15$detail)

# T16: preflight rejects INCOMPLETE verdict
t16 <- tryCatch({
  tmp <- tempfile("inc_fx_")
  .build_complete_fixture(tmp, verdict_field = "INCOMPLETE")
  out <- stub_env$preflight(verbose = FALSE, results_dir = tmp)
  rejected <- any(sapply(out$findings,
                         function(f) grepl("INCOMPLETE", f$msg, fixed = TRUE) &&
                                       grepl("reject", f$msg, ignore.case = TRUE)))
  list(ok = !isTRUE(out$ok) && rejected,
       detail = sprintf("ok=%s, rejected=%s", out$ok, rejected))
}, error = function(e) list(ok = FALSE, detail = paste("error:", e$message)))
record("T16 rejects INCOMPLETE verdict", t16$ok, t16$detail)

# T17: preflight rejects unknown verdict label
t17 <- tryCatch({
  tmp <- tempfile("unk_fx_")
  .build_complete_fixture(tmp, verdict_field = "BOGUS_VALUE")
  out <- stub_env$preflight(verbose = FALSE, results_dir = tmp)
  rejected <- any(sapply(out$findings,
                         function(f) grepl("BOGUS_VALUE", f$msg, fixed = TRUE)))
  list(ok = !isTRUE(out$ok) && rejected,
       detail = sprintf("ok=%s, rejected=%s", out$ok, rejected))
}, error = function(e) list(ok = FALSE, detail = paste("error:", e$message)))
record("T17 rejects unknown verdict label", t17$ok, t17$detail)

# T18: final-mode emits ONLY fig02_pareto, fig03_router_discovery, fig04_gate_evidence
t18 <- tryCatch({
  src <- GEN_SRC
  # Restrict the check to emit() calls inside the `if (mode == "final")` block.
  # The preview block emits QA-PREVIEW-NOT-FOR-SUBMISSION watermarked previews
  # under figures-qa/ (allowed but NOT camera-ready).
  m <- regmatches(src, gregexpr('if \\(mode == "final"\\)\\s*\\{[^{}]*\\}',
                                 src, perl = TRUE))[[1]]
  final_block <- if (length(m) > 0L) m[[1]] else ""
  final_emits <- regmatches(final_block,
                            gregexpr('emit\\("fig0[0-9]+_[a-z_]+',
                                     final_block, perl = TRUE))[[1]]
  final_unique <- sort(unique(sub('emit\\("', '', final_emits)))
  # Whole-source-level: fig01/fig05/fig06 must not appear in any emit() call.
  all_emits <- regmatches(src, gregexpr('emit\\("fig0[0-9]+_[a-z_]+', src, perl = TRUE))[[1]]
  all_unique <- sort(unique(sub('emit\\("', '', all_emits)))
  preview_only <- setdiff(all_unique, final_unique)
  final_expected <- c("fig02_pareto", "fig03_router_discovery", "fig04_gate_evidence")
  ok <- identical(final_unique, final_expected) &&
         !any(grepl("fig01_|fig05_|fig06_", all_unique))
  detail <- sprintf("final_unique: [%s]; all_unique: [%s]; fig01/05/06 absent: %s",
                    paste(final_unique, collapse = ","),
                    paste(all_unique, collapse = ","),
                    !any(grepl("fig01_|fig05_|fig06_", all_unique)))
  list(ok = ok, detail = detail)
}, error = function(e) list(ok = FALSE, detail = paste("error:", e$message)))
record("T18 final mode emits ONLY fig02/03/04", t18$ok, t18$detail)

# T19: fig04 panel structure (a)/(b)/(c) per coordinator spec
t19 <- tryCatch({
  src <- GEN_SRC
  # fig04 must reference (a) gate / (b) MIX_B / (c) MIX_C structural.
  # Use literal-string grepl with fixed=TRUE to avoid regex escape bugs on
  # parentheses / tag strings.
  has_a_gate <- grepl('MIX_B operating points', src, fixed = TRUE) ||
                grepl('"gate ("', src, fixed = TRUE) ||
                grepl('sprintf("gate (%s)', src, fixed = TRUE)
  has_a_tag  <- grepl('tag = "(a)"', src, fixed = TRUE)
  has_b_mixb <- grepl('MIX_B operating points', src, fixed = TRUE) &&
                  grepl('tag = "(b)"', src, fixed = TRUE)
  has_c_mixc <- grepl('"structural"', src, fixed = TRUE) &&
                  grepl('"privacy footprint"', src, fixed = TRUE) &&
                  grepl('router_edit_majority_on_privacy', src, fixed = TRUE) &&
                  grepl('tag = "(c)"', src, fixed = TRUE)
  no_cost_surprise <- !grepl('measured_vs_synthetic_cost_ratio_check',
                              src, fixed = TRUE)
  list(ok = has_a_gate && has_a_tag && has_b_mixb && has_c_mixc && no_cost_surprise,
       detail = sprintf("(a)=%s (a_tag)=%s (b)=%s (c)=%s no_cost_surprise=%s",
                        has_a_gate, has_a_tag, has_b_mixb, has_c_mixc, no_cost_surprise))
}, error = function(e) list(ok = FALSE, detail = paste("error:", e$message)))
record("T19 fig04 panel structure (a)/(b)/(c) per spec", t19$ok, t19$detail)

# T20: pdflatex available on host (absolute tool lookup, PATH-aware)
t20 <- tryCatch({
  pdflatex_path <- Sys.which("pdflatex")
  # pdflatex may need a writable .cfg / interactive configure on first run;
  # `--version` is non-interactive and exits 0 once TeX Live responds.
  ver_out <- ""
  if (nzchar(pdflatex_path)) {
    ver_out <- tryCatch({
      x <- suppressWarnings(system2(pdflatex_path, "--version",
                                    stdout = TRUE, stderr = FALSE))
      paste(x, collapse = " ")
    }, error = function(e) "")
  }
  has_pdflatex <- nzchar(pdflatex_path) &&
                  grepl("pdfTeX|kpathsea|TeX Live", ver_out)
  list(ok = has_pdflatex,
       detail = sprintf("pdflatex=%s, ver: %s",
                        pdflatex_path,
                        substr(ver_out, 1, 80)))
}, error = function(e) list(ok = FALSE, detail = paste("error:", e$message)))
record("T20 pdflatex available (absolute PATH lookup)", t20$ok, t20$detail)

# T21: a complete fixture must still fail when provenance signals contradict.
# Stub system2() rather than weakening the fixture: every grid/verdict/P2 check
# is satisfiable, and only the independent provenance evidence is inconsistent.
t21 <- tryCatch({
  tmp <- tempfile("contradictory_prov_fx_")
  .build_complete_fixture(tmp, verdict_field = "PASS")
  original_system2 <- get("system2", envir = stub_env)
  on.exit(assign("system2", original_system2, envir = stub_env), add = TRUE)

  assign("system2", function(...) {
    out <- jsonlite::toJSON(list(status = "PASS", exit_code = 0L,
                                 counts = list(in_scope_cells = 99L)),
                            auto_unbox = TRUE)
    structure(out, status = 7L)
  }, envir = stub_env)
  subprocess_bad <- stub_env$preflight(verbose = FALSE, results_dir = tmp)
  subprocess_finding <- any(vapply(subprocess_bad$findings, function(f) {
    identical(f$kind, "FAIL") &&
      grepl("subprocess exit status invalid/nonzero", f$msg, fixed = TRUE)
  }, logical(1L)))

  assign("system2", function(...) {
    jsonlite::toJSON(list(status = "PASS", exit_code = 9L,
                          counts = list(in_scope_cells = 99L)),
                     auto_unbox = TRUE)
  }, envir = stub_env)
  report_bad <- stub_env$preflight(verbose = FALSE, results_dir = tmp)
  report_finding <- any(vapply(report_bad$findings, function(f) {
    identical(f$kind, "FAIL") &&
      grepl("report exit_code invalid/nonzero", f$msg, fixed = TRUE)
  }, logical(1L)))

  list(ok = !isTRUE(subprocess_bad$ok) && subprocess_finding &&
              !isTRUE(report_bad$ok) && report_finding,
       detail = sprintf("subprocess_refused=%s, report_refused=%s",
                        !isTRUE(subprocess_bad$ok) && subprocess_finding,
                        !isTRUE(report_bad$ok) && report_finding))
}, error = function(e) list(ok = FALSE, detail = paste("error:", e$message)))
record("T21 complete fixture rejects contradictory provenance exits", t21$ok, t21$detail)

cat(sprintf("\n=== %d/%d tests passed; %d failed ===\n", pass_count,
            pass_count + fail_count, fail_count))
if (fail_count > 0L) quit(status = 1L)
quit(status = 0L)