# Paper B figure package build
R := Rscript
TEX := pdflatex
FIGURE_PDFS := figures/fig01_codec_scope.pdf figures/fig02_efficacy_survival.pdf figures/fig03_rank_survival.pdf figures/fig04_reconstruction_gap.pdf figures/fig06_width_law.pdf
QA_PNGS := $(patsubst figures/%.pdf,figures-qa/%.png,$(FIGURE_PDFS))
GENERATED_TEX := fig02_efficacy_survival.tex fig03_rank_survival.tex fig04_reconstruction_gap.tex fig06_width_law.tex
FIGURE_INTERMEDIATES := $(foreach n,01 02 03 04 06,figures-src/fig$(n).aux figures-src/fig$(n).log figures-src/fig$(n).pdf)

all: $(FIGURE_PDFS)

fig02_efficacy_survival.tex: figures-src/fig02_efficacy_survival.R
	$(R) figures-src/fig02_efficacy_survival.R

fig06_width_law.tex: figures-src/fig06_width_law.R
	$(R) figures-src/fig06_width_law.R

fig03_rank_survival.tex fig04_reconstruction_gap.tex &: make_figures.R
	$(R) make_figures.R

figures/fig01_codec_scope.pdf: figures-src/fig01_codec_scope.tex
	mkdir -p figures
	$(TEX) -interaction=nonstopmode -halt-on-error -output-directory=figures-src -jobname=fig01 figures-src/fig01_codec_scope.tex >/dev/null
	cp figures-src/fig01.pdf $@

figures/fig02_efficacy_survival.pdf: fig02_efficacy_survival.tex figures-src/standalone_fig02.tex
	mkdir -p figures
	$(TEX) -interaction=nonstopmode -halt-on-error -output-directory=figures-src -jobname=fig02 figures-src/standalone_fig02.tex >/dev/null
	cp figures-src/fig02.pdf $@

figures/fig03_rank_survival.pdf: fig03_rank_survival.tex figures-src/standalone_fig03.tex
	mkdir -p figures
	$(TEX) -interaction=nonstopmode -halt-on-error -output-directory=figures-src -jobname=fig03 figures-src/standalone_fig03.tex >/dev/null
	cp figures-src/fig03.pdf $@

figures/fig04_reconstruction_gap.pdf: fig04_reconstruction_gap.tex figures-src/standalone_fig04.tex
	mkdir -p figures
	$(TEX) -interaction=nonstopmode -halt-on-error -output-directory=figures-src -jobname=fig04 figures-src/standalone_fig04.tex >/dev/null
	cp figures-src/fig04.pdf $@

figures/fig06_width_law.pdf: fig06_width_law.tex figures-src/standalone_fig06.tex
	mkdir -p figures
	$(TEX) -interaction=nonstopmode -halt-on-error -output-directory=figures-src -jobname=fig06 figures-src/standalone_fig06.tex >/dev/null
	cp figures-src/fig06.pdf $@

qa: $(QA_PNGS)
	python3 tests/validation.py

figures-qa/%.png: figures/%.pdf
	mkdir -p figures-qa
	pdftoppm -png -r 200 -singlefile "$<" "figures-qa/$*" >/dev/null

clean:
	rm -f $(FIGURE_PDFS) $(QA_PNGS) $(GENERATED_TEX) $(FIGURE_INTERMEDIATES)

.PHONY: all qa clean
