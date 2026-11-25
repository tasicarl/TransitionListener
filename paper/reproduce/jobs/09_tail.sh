#!/bin/bash
# The slow tail of the NANOGrav region on a fine grid (g = 0.400 to 0.440 by 1e-4, log10 v/GeV = 0.20 to 2.0 by
# 0.005): without and with the binary foreground (plane/tail_fine.py, binaries/tail_fog.py, tail_fog_plateau.py,
# tail_fog_prior.py, tail_fog_prof_prior.py), in two parts joined by plane/merge_high.py, into data/tail_check/.
# About 6 h on 56 cores.
set -euo pipefail
source "$(dirname "$0")/env.sh"
T=$REPRO/../data/tail_check
W=${PAPER_WORK:-$REPRO/work}
mkdir -p "$T"
export WORKERS=$NCPU TAIL_WORKERS=$NCPU
TAIL_BHB=1 "$PY" "$REPRO/plane/tail_fine.py" 0.400 0.440 0.0001 0.20 1.30 0.005 "$T/tail_patch.npz"
"$PY" "$REPRO/binaries/tail_fog.py" "$T/tail_fog.npz"
"$PY" "$REPRO/binaries/tail_fog_plateau.py"
PRIOR=recent "$PY" "$REPRO/binaries/tail_fog_prior.py"
PRIOR=recent "$PY" "$REPRO/binaries/tail_fog_prof_prior.py"
# the extension to the upper edge of the prior range, v = 100 GeV, in the work folder, joined to the patches above
# by merge_high.py
TAIL_BHB=1 "$PY" "$REPRO/plane/tail_fine.py" 0.400 0.440 0.0001 1.305 2.0 0.005 "$W/tail_patch_high.npz"
TAIL_GRID="$W/tail_patch_high.npz" "$PY" "$REPRO/binaries/tail_fog.py" "$W/tail_fog_high.npz"
TAIL_GRID="$W/tail_patch_high.npz" PRIOR=recent TAIL_OUT="$W/tail_fog_recent_high.npz" "$PY" "$REPRO/binaries/tail_fog_prior.py"
"$PY" "$REPRO/plane/merge_high.py"
