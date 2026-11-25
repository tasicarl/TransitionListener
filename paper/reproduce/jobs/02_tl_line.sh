#!/bin/bash
# TransitionListener with its defaults on the benchmark line (linescan/capture_tl.py, one process per coupling of
# linescan/merged_points.txt) and its accuracy against the direct integration (linescan/accuracy_table.py, into the
# work folder; read by rate/line_budget.py). About a minute on 18 cores.
set -euo pipefail
source "$(dirname "$0")/env.sh"
xargs -P "$NCPU" -I{} "$PY" "$REPRO/linescan/capture_tl.py" --g {} < "$REPRO/linescan/merged_points.txt"
"$PY" "$REPRO/linescan/accuracy_table.py"
