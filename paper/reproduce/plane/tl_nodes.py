"""TransitionListener at one node (g, log10 v) of the transition table, end to end.

The classically conformal U(1) of models/TL_conformal_u1_no_fermions_3d.py with TL's defaults and nothing else: the
three-dimensional rate at NLO (matching at mu4 = 7.0553 T, mu3 = g T, zero_quartic input) with the NLO correction
Delta S and the dimension-six shift of the grid shipped with TL, and the ODE percolation integral on the
adaptive grid. A user who runs TL on this model gets these numbers. TL_DIM6=0 leaves the dimension-six shift out.

One node per process (several runs in one interpreter share state). Every numeric derived
parameter of the run is written, among them Tperc_SM_GeV, betaH_RH, alpha, Treh_SM_GeV and the
degrees of freedom right after reheating, g_eff_tot_reh and h_eff_tot_reh.

Run with PYTHONPATH=<repo>/src (reproduce/jobs/06_tl_nodes.sh runs every node of tl_nodes.txt into <work>/tl_nodes):
  python tl_nodes.py --g 0.45 --logv 0.5 --out <work>/tl_nodes/g0.450_lv+0.500.json
"""
import argparse
import json
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import paths  # noqa: E402  (src/, models/ and the folders of paper/reproduce on sys.path)
CODE = paths.CODE      # the tree that holds src/ and models/ (TL_CODE_ROOT)

import numpy as np  # noqa: E402
from transitionlistener.interface import _compute_single_point  # noqa: E402
from transitionlistener.interface.single_point import load_potential  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--g", type=float, required=True)
ap.add_argument("--logv", type=float, required=True, help="log10(v/GeV)")
ap.add_argument("--out", required=True)
args = ap.parse_args()
v = 10.0 ** args.logv
record = dict(g=args.g, logv=args.logv, v_GeV=v, observables={}, error=None)
t0 = time.time()

try:
    base = load_potential(os.path.join(CODE, "models", "TL_conformal_u1_no_fermions_3d.py"), "specific_potential")

    def factory(p, *a, **kw):
        pot = base(p, *a, **kw)
        if os.environ.get("TL_DIM6", "1") == "0":
            pot.config.nucleationConf.include_dim6 = False
        return pot

    context, error = _compute_single_point({"g": args.g, "v_GeV": v}, factory, timeout=-1, verbose=False,
                                           include_smbhb=False, max_stage=3)
    if error is not None:
        record["error"] = str(error)
    res = context if isinstance(context, dict) else getattr(context, "__dict__", {})

    def walk(d, depth=0):
        if depth > 3 or not isinstance(d, dict):
            return
        for k, val in d.items():
            if isinstance(val, (int, float, np.floating)) and not isinstance(val, bool):
                record["observables"].setdefault(k, float(val))
            elif isinstance(val, np.ndarray) and val.size == 1:
                record["observables"].setdefault(k, float(val))
            elif isinstance(val, dict):
                walk(val, depth + 1)

    walk(res)
except Exception:
    record["error"] = traceback.format_exc()
record["seconds"] = time.time() - t0
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
json.dump(record, open(args.out, "w"), indent=1)
