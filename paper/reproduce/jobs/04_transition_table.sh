#!/bin/bash
# The transition table on the (g, log10 v) grid by exact percolation (rate/build_exact_table.py), its false-vacuum
# background variant (rate/build_falsevac_table.py), merged into data/transition_table.npz (rate/merge_tables.py); the
# line budget data/line_budget.json (rate/line_budget.py, needs 02_tl_line.sh for its TransitionListener columns).
# About 2 h on 56 cores.
set -euo pipefail
source "$(dirname "$0")/env.sh"
W=${PAPER_WORK:-$REPRO/work}; mkdir -p "$W"
EXACT_TABLE_OUT="$W/exact_table.npz" "$PY" "$REPRO/rate/build_exact_table.py"
"$PY" "$REPRO/rate/build_falsevac_table.py" --background fvbg --out "$W/falsevac_table.npz" --procs "$NCPU"
"$PY" "$REPRO/rate/merge_tables.py" "$REPRO/../data/transition_table.npz" "$W/exact_table.npz" "$W/falsevac_table.npz"
"$PY" "$REPRO/rate/line_budget.py"
