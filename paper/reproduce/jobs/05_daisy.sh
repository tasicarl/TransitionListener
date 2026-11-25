#!/bin/bash
# The four-dimensional daisy rates N4 (couplings run to pi T) and N5 (fixed couplings): actions per coupling
# (daisy/compute_daisy_actions.py, into DAISY_CACHE;
# the three files of g = 0.5 that stage A reads are copied to data/daisy_cache), the tables on the plane (daisy/build_daisy_table.py, work
# folder; daisy/prepare_daisy_table.py, data/daisy_*table_plane*.npz), the line (daisy/daisy_line.py,
# data/daisy_line*.json), the scale band of N4 at g = 0.5 and the N5 curve of figure 2 (daisy/daisy_scale_band.py,
# data/scale_band_g0.500.json). Several hours on 56 cores.
set -euo pipefail
source "$(dirname "$0")/env.sh"
for kind in bounce bounce_rg direct tl; do
  "$PY" "$REPRO/daisy/compute_daisy_actions.py" --kind $kind --workers "$NCPU"
done
for m in 0.5 2; do
  MU_OVER_PIT=$m "$PY" "$REPRO/daisy/compute_daisy_actions.py" --kind bounce_rg --tag _mu$m --g 0.5 --workers "$NCPU"
done
for kind in bounce bounce_rg direct tl; do
  "$PY" "$REPRO/daisy/build_daisy_table.py" --kind $kind --workers "$NCPU"
done
for kind in bounce rg direct tl; do
  "$PY" "$REPRO/daisy/prepare_daisy_table.py" $kind
done
# the further couplings of the line curves of figure 2, densely at the edges where the transition stops completing
for kind in direct bounce bounce_rg; do
  "$PY" "$REPRO/daisy/compute_daisy_actions.py" --kind $kind --line --workers "$NCPU"
done
DAISY_LINE_KIND=bounce "$PY" "$REPRO/daisy/daisy_line.py"
DAISY_LINE_KIND=bounce_rg "$PY" "$REPRO/daisy/daisy_line.py"
WORKERS=$NCPU "$PY" "$REPRO/daisy/daisy_scale_band.py"
cp "$DAISY_CACHE"/daisy_bounce_rg_g0.500.json "$DAISY_CACHE"/daisy_bounce_rg_mu0.5_g0.500.json \
   "$DAISY_CACHE"/daisy_bounce_rg_mu2_g0.500.json "$REPRO/../data/daisy_cache/"
