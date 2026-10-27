#!/bin/bash
# Render Additional files 1 and 2 (Markdown) to PDF with pandoc + pdflatex, landscape, for the submission/review copy.
# Usage: bash build_additional.sh   (from any folder). Output: latex/additional/additional_file_{1,2}.pdf
cd "$(dirname "$0")/.." || exit 1
mkdir -p latex/additional
HEADER=$(mktemp)
cat > "$HEADER" <<'TEX'
\usepackage{newunicodechar}
\newunicodechar{×}{\ensuremath{\times}}\newunicodechar{−}{\ensuremath{-}}\newunicodechar{≤}{\ensuremath{\leq}}
\newunicodechar{≥}{\ensuremath{\geq}}\newunicodechar{²}{\textsuperscript{2}}\newunicodechar{³}{\textsuperscript{3}}
\newunicodechar{⁻}{\textsuperscript{--}}\newunicodechar{⁵}{\textsuperscript{5}}\newunicodechar{¹}{\textsuperscript{1}}
\newunicodechar{⁰}{\textsuperscript{0}}\newunicodechar{ρ}{\ensuremath{\rho}}\newunicodechar{α}{\ensuremath{\alpha}}
\newunicodechar{δ}{\ensuremath{\delta}}\newunicodechar{γ}{\ensuremath{\gamma}}\newunicodechar{β}{\ensuremath{\beta}}
\newunicodechar{φ}{\ensuremath{\phi}}\newunicodechar{†}{\dag}\newunicodechar{‡}{\ddag}\newunicodechar{•}{\textbullet}
TEX
for f in additional_file_1 additional_file_2; do
  # Citation keys are not resolved here; the only one used is the TRIPOD+AI statement
  sed -E 's/\[@R-TRIPODAI\]/(Collins et al., BMJ 2024;385:e078378)/' "$f.md" |
    pandoc -f markdown -o "latex/additional/$f.pdf" --pdf-engine=pdflatex -H "$HEADER" \
      -V geometry:"landscape,margin=1.5cm" -V fontsize=9pt -V documentclass=extarticle
  echo "Written: latex/additional/$f.pdf"
done
rm -f "$HEADER"
