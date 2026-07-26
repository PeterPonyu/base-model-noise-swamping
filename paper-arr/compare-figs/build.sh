#!/usr/bin/env bash
# Build side-by-side comparison PNGs: pgfplots (left) vs R/ggplot2 (right).
# Each figure's tikzpicture is compiled as a cropped standalone, rasterized,
# and stitched. Does NOT touch main.tex or sections/.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"     # paper-arr/
CMP="$ROOT/compare-figs"; BUILD="$CMP/build"; PNG="$CMP/png"
PGF="$ROOT/figures-tex"; RFIG="$ROOT/figures-r"
mkdir -p "$BUILD" "$PNG"
cp "$PGF/_pgfpreamble.tex" "$BUILD/ppreamble.tex"

pgf_pre() { cat <<'EOF'
\documentclass[border=6pt]{standalone}
\usepackage[T1]{fontenc}\usepackage{times}
\usepackage{amsmath}\usepackage{amssymb}\usepackage{xspace}
\usepackage{pgfplots}\usepgfplotslibrary{groupplots}\usetikzlibrary{patterns}
\newcommand{\SxC}{\ensuremath{S\!\times\!C}\xspace}
\input{ppreamble}
\begin{document}
EOF
}
r_pre() { cat <<'EOF'
\documentclass[border=6pt]{standalone}
\usepackage[T1]{fontenc}\usepackage{times}
\usepackage{amsmath}\usepackage{amssymb}
\usepackage{tikz}
\begin{document}
EOF
}

for f in A B C D E F; do
  src=$(ls "$PGF"/fig${f}_*.tex)
  # extract just the tikzpicture body from the pgfplots figure float
  awk '/\\begin\{tikzpicture\}/{p=1} p{print} /\\end\{tikzpicture\}/{p=0}' "$src" > "$BUILD/pgf_body.tex"
  { pgf_pre; cat "$BUILD/pgf_body.tex"; echo '\end{document}'; } > "$BUILD/pgf_${f}.tex"
  { r_pre;  cat "$RFIG/fig${f}.tex"; echo '\end{document}'; } > "$BUILD/r_${f}.tex"
  for v in pgf r; do
    ( cd "$BUILD" && pdflatex -interaction=nonstopmode -halt-on-error "${v}_${f}.tex" >"${v}_${f}.log" 2>&1 )
    if [ ! -f "$BUILD/${v}_${f}.pdf" ]; then echo "FAIL ${v}_${f} (see log)"; continue; fi
    pdftoppm -png -r 220 "$BUILD/${v}_${f}.pdf" "$BUILD/${v}_${f}" >/dev/null 2>&1
    mv "$BUILD/${v}_${f}-1.png" "$BUILD/${v}_${f}.png" 2>/dev/null || true
  done
  if [ -f "$BUILD/pgf_${f}.png" ] && [ -f "$BUILD/r_${f}.png" ]; then
    # label each panel, normalize to same height, stitch with a divider
    convert "$BUILD/pgf_${f}.png" -gravity North -background white -splice 0x34 \
            -pointsize 22 -fill black -annotate +0+6 "pgfplots (current)" "$BUILD/L_${f}.png"
    convert "$BUILD/r_${f}.png"   -gravity North -background white -splice 0x34 \
            -pointsize 22 -fill black -annotate +0+6 "R / ggplot2 (new)" "$BUILD/R_${f}.png"
    H=$(convert "$BUILD/L_${f}.png" "$BUILD/R_${f}.png" -format "%[fx:max(u.h,v.h)]\n" info: | head -1)
    convert "$BUILD/L_${f}.png" -resize x${H} "$BUILD/L2_${f}.png"
    convert "$BUILD/R_${f}.png" -resize x${H} "$BUILD/R2_${f}.png"
    convert "$BUILD/L2_${f}.png" -bordercolor '#cccccc' -border 1x0 \
            "$BUILD/R2_${f}.png" +append -background white -gravity center "$PNG/compare_fig${f}.png"
    echo "OK  compare_fig${f}.png"
  fi
done
echo "--- done; PNGs in $PNG ---"
ls -1 "$PNG"
