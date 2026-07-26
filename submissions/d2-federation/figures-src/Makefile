# D2 federation figures: R/ggplot2 -> tikzDevice -> standalone pdflatex -> PDF.
# The submission main.tex includes ONLY figA.pdf / figB.pdf (portal-safe, no inline TikZ).
all: figA.pdf figB.pdf figC.pdf figD.pdf figE.pdf
figA_gain_vs_frac.tex figB_g_profiles.tex: make_figures.R
	Rscript make_figures.R
figA.pdf: standalone_figA.tex figA_gain_vs_frac.tex
	pdflatex -interaction=nonstopmode -jobname=figA standalone_figA.tex >/dev/null
figB.pdf: standalone_figB.tex figB_g_profiles.tex
	pdflatex -interaction=nonstopmode -jobname=figB standalone_figB.tex >/dev/null
clean:
	rm -f figA.pdf figB.pdf *.aux *.log
.PHONY: all clean
figC_admission_benefit.tex: make_figures.R
	Rscript make_figures.R
figC.pdf: standalone_figC.tex figC_admission_benefit.tex
	pdflatex -interaction=nonstopmode -jobname=figC standalone_figC.tex >/dev/null
figD_dose_response.tex figE_gate_evidence.tex: make_figures.R
	Rscript make_figures.R
figD.pdf: standalone_figD.tex figD_dose_response.tex
	pdflatex -interaction=nonstopmode -jobname=figD standalone_figD.tex >/dev/null
figE.pdf: standalone_figE.tex figE_gate_evidence.tex
	pdflatex -interaction=nonstopmode -jobname=figE standalone_figE.tex >/dev/null
