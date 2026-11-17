#!/usr/bin/env bash
# Build the manuscript from the real-data (R3) results.
#
# Assumes run_real_rep.py has already produced results/R3_*.  Regenerates
# every figure the manuscript uses, injects the numbers, and compiles.
set -e
cd "$(dirname "$0")"

echo "[1/4] validation + real-data figures"
python3 validate_cimmyt.py > /dev/null
python3 make_figures.py 1 > /dev/null 2>&1 || python3 make_figures.py > /dev/null 2>&1 || true
python3 make_figures_real.py 8 9

echo "[2/4] inject results"
python3 fill_real.py

echo "[3/4] compile"
cd paper
# three passes: labels, then cross-references, then page references
for i in 1 2 3; do
  pdflatex -interaction=nonstopmode paper_filled.tex > /dev/null 2>&1 || true
done

echo "[4/4] check"
if grep -q "^!" paper_filled.log; then
  echo "LaTeX errors:"; grep -n "^!" paper_filled.log | head
fi
echo "undefined refs : $(grep -c 'undefined' paper_filled.log || true)"
echo "overfull boxes : $(grep -c 'Overfull' paper_filled.log || true)"
echo "pages          : $(pdfinfo paper_filled.pdf | awk '/Pages/{print $2}')"
rm -f paper_filled.out
echo "-> paper/paper_filled.pdf"
