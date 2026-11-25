#!/bin/bash
# TransitionListener at the nodes of plane/tl_nodes.txt (reheating temperature, degrees of freedom and efficiency;
# plane/tl_nodes.py, one process per node, 30 min limit), collected into data/tl_nodes_table.npz and
# data/kappa_nodes_table.npz (plane/collect_tl_nodes.py). About 4 min on 56 cores.
set -euo pipefail
source "$(dirname "$0")/env.sh"
OUT=${PAPER_WORK:-$REPRO/work}/tl_nodes; mkdir -p "$OUT"
export OUT
one() { out=$OUT/g$1_lv$2.json; [ -s "$out" ] || timeout 1800 "$PY" "$REPRO/plane/tl_nodes.py" --g $1 --logv $2 --out "$out" > /dev/null 2>&1 || echo "node g=$1 lv=$2 failed"; }
export -f one
xargs -P "$NCPU" -L 1 bash -c 'one $0 $1' < "$REPRO/plane/tl_nodes.txt"
"$PY" "$REPRO/plane/collect_tl_nodes.py"
