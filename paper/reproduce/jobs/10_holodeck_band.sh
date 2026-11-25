#!/bin/bash
# The binaryfog prior band of figure 3 from holodeck itself (binaries/binaries_prior_band_holodeck.py), in an
# environment with holodeck and binaryfog 1.0.0 (PY_HOLODECK). Hours on 40 cores.
set -euo pipefail
source "$(dirname "$0")/env.sh"
"${PY_HOLODECK:-$PY}" "$REPRO/binaries/binaries_prior_band_holodeck.py" 240 "$NCPU" recent
