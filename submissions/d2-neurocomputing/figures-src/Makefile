# D2 federation figures: R/ggplot2 -> tikzDevice -> standalone pdflatex -> PDF.
# The submission main.tex includes ONLY figA.pdf / figB.pdf (portal-safe, no inline TikZ).
all: figA.pdf figB.pdf figC.pdf figD.pdf figE.pdf \
     figF1A.pdf figF1B.pdf figF1C.pdf figF1D.pdf \
     figF2A.pdf figF2B.pdf figF2C.pdf figF2D.pdf \
     figF3A.pdf figF3B.pdf figF3C.pdf figF3D.pdf \
     figF8A.pdf figF8B.pdf figF8C.pdf figF8D.pdf
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

# F1 / F2 — gain-screen map + two-regime law (2026-08-01)
figF1A_gain_vs_frac.tex figF1B_lofo.tex figF1C_ordering_null.tex figF1D_family_regimes.tex: make_figF1_gainmap.R
	Rscript make_figF1_gainmap.R
figF1A.pdf: standalone_figF1A.tex figF1A_gain_vs_frac.tex
	pdflatex -interaction=nonstopmode -jobname=figF1A standalone_figF1A.tex >/dev/null
figF1B.pdf: standalone_figF1B.tex figF1B_lofo.tex
	pdflatex -interaction=nonstopmode -jobname=figF1B standalone_figF1B.tex >/dev/null
figF1C.pdf: standalone_figF1C.tex figF1C_ordering_null.tex
	pdflatex -interaction=nonstopmode -jobname=figF1C standalone_figF1C.tex >/dev/null
figF1D.pdf: standalone_figF1D.tex figF1D_family_regimes.tex
	pdflatex -interaction=nonstopmode -jobname=figF1D standalone_figF1D.tex >/dev/null

figF2A_dose_high.tex figF2B_dose_low.tex figF2C_crossover_g.tex figF2D_signed_reanalysis.tex: make_figF2_tworegime.R
	Rscript make_figF2_tworegime.R
figF2A.pdf: standalone_figF2A.tex figF2A_dose_high.tex
	pdflatex -interaction=nonstopmode -jobname=figF2A standalone_figF2A.tex >/dev/null
figF2B.pdf: standalone_figF2B.tex figF2B_dose_low.tex
	pdflatex -interaction=nonstopmode -jobname=figF2B standalone_figF2B.tex >/dev/null
figF2C.pdf: standalone_figF2C.tex figF2C_crossover_g.tex
	pdflatex -interaction=nonstopmode -jobname=figF2C standalone_figF2C.tex >/dev/null
figF2D.pdf: standalone_figF2D.tex figF2D_signed_reanalysis.tex
	pdflatex -interaction=nonstopmode -jobname=figF2D standalone_figF2D.tex >/dev/null


figF3A_partial_rho_by_g.tex figF3B_c2_coherence.tex \
figF3C_perm_null.tex figF3D_two_boundaries.tex: make_figF3_gate.R
	Rscript make_figF3_gate.R
figF3A.pdf: standalone_figF3A.tex figF3A_partial_rho_by_g.tex
	pdflatex -interaction=nonstopmode -jobname=figF3A standalone_figF3A.tex >/dev/null
figF3B.pdf: standalone_figF3B.tex figF3B_c2_coherence.tex
	pdflatex -interaction=nonstopmode -jobname=figF3B standalone_figF3B.tex >/dev/null
figF3C.pdf: standalone_figF3C.tex figF3C_perm_null.tex
	pdflatex -interaction=nonstopmode -jobname=figF3C standalone_figF3C.tex >/dev/null
figF3D.pdf: standalone_figF3D.tex figF3D_two_boundaries.tex
	pdflatex -interaction=nonstopmode -jobname=figF3D standalone_figF3D.tex >/dev/null

figF8A_crossterm_alignment.tex figF8B_slope_sign_regime.tex \
figF8C_gain_slope_proxy.tex figF8D_prop1_scope.tex: make_figF8_mechanism.R
	Rscript make_figF8_mechanism.R
figF8A.pdf: standalone_figF8A.tex figF8A_crossterm_alignment.tex
	pdflatex -interaction=nonstopmode -jobname=figF8A standalone_figF8A.tex >/dev/null
figF8B.pdf: standalone_figF8B.tex figF8B_slope_sign_regime.tex
	pdflatex -interaction=nonstopmode -jobname=figF8B standalone_figF8B.tex >/dev/null
figF8C.pdf: standalone_figF8C.tex figF8C_gain_slope_proxy.tex
	pdflatex -interaction=nonstopmode -jobname=figF8C standalone_figF8C.tex >/dev/null
figF8D.pdf: standalone_figF8D.tex figF8D_prop1_scope.tex
	pdflatex -interaction=nonstopmode -jobname=figF8D standalone_figF8D.tex >/dev/null
