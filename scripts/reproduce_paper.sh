#!/usr/bin/env bash
# Regenerate every table (paper/tables/*.tex) and figure (paper/figures/*.pdf) in the report
# from the seeded generator. Requires the `paper` extra (numpy, scipy, matplotlib).
set -euo pipefail
cd "$(dirname "$0")/../paper"
mkdir -p figures tables
python make_data.py
python figures.py
echo "Tables in paper/tables, figures in paper/figures."
