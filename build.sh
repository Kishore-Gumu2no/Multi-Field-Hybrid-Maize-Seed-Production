#!/usr/bin/env bash
# Build the manuscript end to end from existing results.
set -e
cd "$(dirname "$0")"

echo "[1/3] figures"
python3 make_figures.py

echo "[2/3] inject results"
python3 fill_paper.py

echo "[3/3] compile"
cd paper
# three passes: bibliography labels, then cross-references, then page refs
for i in 1 2 3; do
  pdflatex -interaction=nonstopmode paper_filled.tex > /dev/null 2>&1 || true
done
echo "unresolved refs: $(grep -c 'undefined' paper_filled.log || true)"
rm -f paper_filled.out
echo "-> paper/paper_filled.pdf"
