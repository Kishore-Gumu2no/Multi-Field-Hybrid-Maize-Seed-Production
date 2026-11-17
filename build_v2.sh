#!/usr/bin/env bash
# Build the revised manuscript (paper_v2) from existing results.
# Assumes results/R3_*, R4_*, R5_*, R6_raw.csv, R7_raw.csv and V1_* exist.
#   python3 run_full21.py   # R6: assignment on all 21 seasons (~1 h, 2 cores)
#   python3 run_curves.py   # R7: yield-curve sensitivity (~1 h)
set -e
cd "$(dirname "$0")"
echo "[1/3] figures";  python3 make_figure_practice.py > /dev/null
echo "[2/3] inject";   python3 fill_v2.py
echo "[3/3] compile";  cd paper
for i in 1 2 3; do pdflatex -interaction=nonstopmode paper_v2_filled.tex > /dev/null 2>&1 || true; done
grep -q "^!" paper_v2_filled.log && { echo "LaTeX errors:"; grep -n "^!" paper_v2_filled.log | head; }
echo "undefined refs : $(grep -c 'undefined' paper_v2_filled.log || true)"
echo "overfull boxes : $(grep -c 'Overfull' paper_v2_filled.log || true)"
echo "pages          : $(pdfinfo paper_v2_filled.pdf | awk '/Pages/{print $2}')"
rm -f paper_v2_filled.out
echo "-> paper/paper_v2_filled.pdf"
