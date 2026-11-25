#!/bin/bash
# The bounce actions of every rate variant on the temperature grids of the line-scan anchors (rate/compute_actions.py):
# the base grid, the low- and high-temperature extensions and the LO matching-scale band at g = 0.5 of figure 2, into
# data/action_cache (ACTION_CACHE_DIR). Resumable. Of order a day on 56 cores.
set -euo pipefail
source "$(dirname "$0")/env.sh"
export ACTION_WORKERS=$NCPU
for flag in "" --low-all --high --higher --highest --fig1 --edge; do
  "$PY" "$REPRO/rate/compute_actions.py" $flag
done
