#!/bin/bash
# The likelihood maps of every variant on the (g, log10 v) grid (plane/plane_scan.py, data/plane_scan_u1.npz), the best
# fits (plane/best_fits.py), the free spectra of figure 3 (plane/pta_free_spectra.py), the fits without the causal tail
# (plane/notail_scan.py) and B1 without its prior (binaries/bhb_lnl_without_prior.py). About 1 h on 56 cores.
set -euo pipefail
source "$(dirname "$0")/env.sh"
"$PY" "$REPRO/plane/plane_scan.py" --workers "$NCPU"
BEST_FIT_WORKERS=$NCPU "$PY" "$REPRO/plane/best_fits.py"
"$PY" "$REPRO/plane/pta_free_spectra.py"
"$PY" "$REPRO/plane/notail_scan.py" "$NCPU"
"$PY" "$REPRO/binaries/bhb_lnl_without_prior.py"
