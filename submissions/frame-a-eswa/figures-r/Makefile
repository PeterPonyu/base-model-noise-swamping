# Makefile -- build targets for the Frame-A figure generator (CPU-only).
#
# Targets:
#   make test       -- run focused tests (submits to Rscript, exits 0 iff all PASS)
#   make preflight  -- run only the fail-closed preflight gate (no figure emit)
#   make preview    -- run in MIX_A preview mode (writes ONLY under figures-qa/, watermarked)
#   make final      -- run in final mode (emits fig02-04 to figures-r/ iff all gates PASS)
#   make clean      -- remove generated fig02-04 files and any preview under figures-qa/
#   make check      -- print a one-line summary of the on-disk gate state
#
# Fail-closed semantics: every target that COULD write a camera-ready file
# (final) refuses to do so when the preflight gate does not PASS. The preview
# target is the only path that may emit ANY .tex under figures-qa/, and only
# when MIX_A is independently clean. See make_figures_frame_a.R for details.

ROOT       := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
GEN        := $(ROOT)/make_figures_frame_a.R
TESTS      := $(ROOT)/tests/test_make_figures_frame_a.R
BUILD_SH   := $(ROOT)/build.sh

.PHONY: all test preflight preview preview-incomplete final clean check

all: preflight

test:
	Rscript $(TESTS)

preflight:
	$(BUILD_SH) --preflight

preview:
	$(BUILD_SH) --preview

preview-incomplete:
	$(BUILD_SH) --preview-incomplete

final:
	$(BUILD_SH)

clean:
	rm -f $(ROOT)/fig0[2-4]_*.tex $(ROOT)/fig0[2-4]_*.pdf
	rm -f $(ROOT)/../figures-qa/fig0[2-4]_*.tex $(ROOT)/../figures-qa/fig0[2-4]_*.pdf

check:
	@echo "==== on-disk gate state ===="
	@echo "MIX_B cells:    $$(ls $(ROOT)/../../../edit-harness/results/frame_a/cells 2>/dev/null | grep -c cell_llama-3.2-1b_real_MIX_B_ || echo 0)"
	@echo "MIX_C cells:    $$(ls $(ROOT)/../../../edit-harness/results/frame_a/cells 2>/dev/null | grep -c cell_llama-3.2-1b_real_MIX_C_ || echo 0)"
	@echo "namespaced P2:  $$(test -f $(ROOT)/../../../edit-harness/results/frame_a/p2_llama-3.2-1b_real_MIX_C.json && echo PRESENT || echo MISSING)"
	@echo "ftfix verdict:  $$(test -f $(ROOT)/../../../edit-harness/results/frame_a/frame_a_verdict_ftfix.json && echo PRESENT || echo MISSING)"
	@echo "verdict candidates (non-ftfix): $$(ls $(ROOT)/../../../edit-harness/results/frame_a/frame_a_verdict*.json 2>/dev/null | grep -v ftfix || echo none)"
	@echo "hidden/quarantine dirs: $$(ls -d $(ROOT)/../../../edit-harness/results/frame_a/cells/.[a-zA-Z]* 2>/dev/null || echo none)"