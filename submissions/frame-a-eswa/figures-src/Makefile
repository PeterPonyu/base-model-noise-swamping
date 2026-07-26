# CPU-only build for the hand-authored Paper A routing schematic.
# Camera-ready output is vector PDF; the 200-dpi PNG and 390pt-inclusion
# PNG are QA-only. The 390pt render simulates ESWA preprint `\linewidth`
# inclusion so text clipping/overlap can be inspected at effective sizes.
HERE      := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
SOURCE    := $(HERE)/fig01_routing_framework.tex
BUILD     := $(HERE)/.build
PDF       := $(HERE)/../figures/fig01_routing_framework.pdf
QA        := $(HERE)/../figures-qa/fig01_routing_framework_200dpi.png
QA390     := $(HERE)/../figures-qa/fig01_routing_framework_390pt_200dpi.png
JOB       := fig01_routing_framework
LOG       := $(BUILD)/$(JOB).log
PDF390    := $(BUILD)/$(JOB)_390pt.pdf

.PHONY: all pdf qa qa390 check clean

all: pdf qa qa390 check

pdf: $(PDF)

$(PDF): $(SOURCE)
	@mkdir -p $(BUILD) $(dir $(PDF))
	pdflatex -interaction=nonstopmode -halt-on-error -file-line-error \
	  -jobname=$(JOB) -output-directory=$(BUILD) $(SOURCE)
	cp $(BUILD)/$(JOB).pdf $(PDF)

qa: $(QA)

$(QA): $(PDF)
	@mkdir -p $(dir $(QA))
	pdftoppm -png -r 200 -singlefile $(PDF) $(basename $(QA))

qa390: $(QA390)

$(QA390): $(PDF)
	@mkdir -p $(BUILD) $(dir $(QA390))
	gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dQUIET \
	  -sOutputFile=$(PDF390) \
	  -dDEVICEWIDTHPOINTS=390 -dDEVICEHEIGHTPOINTS=147 \
	  -dFIXEDMEDIA -dPDFFitPage $(PDF)
	pdftoppm -png -r 200 -singlefile $(PDF390) $(basename $(QA390))

check: $(PDF) $(QA) $(QA390)
	@test -s $(PDF)
	@test -s $(QA)
	@test -s $(QA390)
	@pdfinfo $(PDF) | grep -q '^Pages:[[:space:]]*1$$'
	@pdfinfo $(PDF390) | grep -q '^Page size:[[:space:]]*390 x'
	@pdftotext $(PDF) - | tr '\n' ' ' | grep -Eq 'Frozen[[:space:]]+decision[[:space:]]+gate'
	@pdftotext $(PDF) - | tr '\n' ' ' | grep -Eq 'measured costs, frozen governance rules,'
	@identify -format '%m %wx%h\n' $(QA) | grep -Eq '^PNG [1-9][0-9]*x[1-9][0-9]*$$'
	@identify -format '%m %wx%h\n' $(QA390) | grep -Eq '^PNG 10[0-9]{2,3}x[1-9][0-9]*$$'
	@! grep -Eq 'Overfull|Underfull|LaTeX Error|Package .* Error|Undefined control sequence' $(LOG)
	@echo "PASS: vector PDF (1p), 200-dpi QA PNG, 390pt-inclusion QA PNG, clean LaTeX log"

clean:
	rm -rf $(BUILD)
	rm -f $(PDF) $(QA) $(QA390)