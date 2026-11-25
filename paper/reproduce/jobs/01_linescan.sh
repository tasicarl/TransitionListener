#!/bin/bash
# TransitionListener's line scan v = e^{1/6} mu0/g (mu0 = 1 GeV): per coupling the 4D daisy rate, the 3D LO action
# and anchor, the Delta S tables at mu4 x 0.5, 1, 2 and the 3D NLO runs (linescan/scan_point.py, resumable, into
# LINESCAN_POINTS), then data/linescan_u1_mu0_1GeV.json, and the three points that stage A reads copied into
# data/linescan_points. Couplings run in parallel, three processes each (one per mu4 value); 4 min on 56 cores.
#   bash 01_linescan.sh [g ...]
set -euo pipefail
source "$(dirname "$0")/env.sh"
GS=${*:-"0.40 0.41 0.42 0.43 0.44 0.45 0.46 0.47 0.475 0.48 0.49 0.50 0.52 0.55 0.60 0.65 0.70 0.80 0.90 1.00"}
mkdir -p "$LINESCAN_POINTS"
run_g() {
  local v
  v=$("$PY" -c "import math, sys; print(repr(math.exp(1/6) / float(sys.argv[1])))" "$1")
  "$PY" "$REPRO/linescan/scan_point.py" --g "$1" --model u1 --mu3-choice gT --v "$v" --band 0.5,1,2 --l-max 70 \
    > "$LINESCAN_POINTS/scan_g$1.log" 2>&1
}
export -f run_g
printf '%s\n' $GS | xargs -P "$(( NCPU / 3 > 0 ? NCPU / 3 : 1 ))" -I{} bash -c 'run_g {}'
# scan_point.py logs a failed run as "(exit N)" and carries on; stop here instead of collecting an incomplete scan
if grep -l "(exit [1-9]" "$LINESCAN_POINTS"/scan_g*.log; then echo "failed runs in the logs above"; exit 1; fi
"$PY" "$REPRO/linescan/collect_linescan.py" "$LINESCAN_POINTS" --tag u1 --any-v \
  --out "$REPRO/../data/linescan_u1_mu0_1GeV.json"
for g in 0.475 0.500 0.700; do
  src=$(ls -d "$LINESCAN_POINTS"/g${g}_u1_v*)
  dst="$REPRO/../data/linescan_points/$(basename "$src")"
  mkdir -p "$dst"
  cp "$src"/summary.json "$src"/table_mu0.5.json "$src"/table_mu1.json "$src"/table_mu2.json "$dst"/
done
