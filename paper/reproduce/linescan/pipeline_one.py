"""Run ONE treatment of ONE parameter point through the pipeline; write JSON.

One mode per process is required, not stylistic: several modes in one
interpreter share state and all return the four-dimensional answer.

  --mode 4d     the 4D daisy baseline
  --mode lo3d   3D EFT, leading-order action
  --mode nlo3d  3D EFT, next-to-leading-order action (needs --table)

Errors are recorded in the output, never raised, so a scan keeps going.
"""
import argparse
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402  (src/, models/ and the folders of paper/reproduce on sys.path)
ROOT = paths.CODE

from transitionlistener.eft_rate import abelian_higgs_3d  # noqa: E402
from transitionlistener.interface import _compute_single_point  # noqa: E402
from transitionlistener.interface.single_point import load_potential  # noqa: E402

MU4_CENTRAL = float(os.environ.get("MU4_CENTRAL", "7.0553"))     # 2 pi for the settings of arXiv:2609.12098
OBSERVABLES = ("Tperc_SM_GeV", "Tnuc_SM_GeV", "alpha", "alpha_theta",
               "betaH_S3", "betaH_RH", "Treh_SM_GeV")

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--g", type=float, required=True)
ap.add_argument("--y", type=float, default=None, help="Yukawa (dark_u1 model only)")
ap.add_argument("--model", default="dark_u1", choices=("dark_u1", "u1"),
                help="dark_u1: TL_conformal_u1_fermions_3d (two Weyl fermions); "
                     "u1: TL_conformal_u1_no_fermions_3d (no fermions, DRalgo run 6)")
ap.add_argument("--mu4-over-T", type=float, default=None,
                help="absolute mu4/T; overrides --mu4 (e.g. 2 pi for arXiv:2609.12098)")
ap.add_argument("--input-scheme", default=None,
                help="nucleationConf.input_scheme (CW, zero_quartic)")
ap.add_argument("--mu3-choice", default=None, help="u1 model: g3sq (default) or gT")
ap.add_argument("--v", type=float, required=True, help="v in GeV")
ap.add_argument("--mode", required=True, choices=("4d", "lo3d", "nlo3d"))
ap.add_argument("--mu4", type=float, default=1.0, help="multiple of 7.0553 T")
ap.add_argument("--matching", default="NLO", choices=("LO", "NLO"))
ap.add_argument("--table", default=None, help="JSON with T_GeV and delta_S")
ap.add_argument("--algorithm", default="adaptive_step_size",
                choices=("adaptive_step_size", "fixed_step_size"),
                help="percolation solver. fixed_step_size is the "
                     "alternative (agrees with adaptive + double_integral to "
                     "0.04% at benchmark A); used where the adaptive solver's "
                     "jump-acceptance check rejects a region its own refinement "
                     "declines to refine (g = 0.50 on the line scan).")
ap.add_argument("--out", required=True)
args = ap.parse_args()

if args.model == "dark_u1":
    if args.y is None:
        ap.error("--y is required for the dark_u1 model")
    params = {"g": args.g, "y": args.y, "v_GeV": args.v}
    MODEL_FILE = "models/TL_conformal_u1_fermions_3d.py"
else:
    params = {"g": args.g, "v_GeV": args.v}
    MODEL_FILE = "models/TL_conformal_u1_no_fermions_3d.py"
record = dict(params=params, model=args.model, mode=args.mode, mu4=args.mu4,
              mu4_over_T=args.mu4_over_T, input_scheme=args.input_scheme,
              mu3_choice=args.mu3_choice, algorithm=args.algorithm,
              matching=args.matching, table=args.table, observables={},
              temperature_keys={}, error=None, extrapolating=None,
              table_range=None)
t0 = time.time()

corr = None
if args.mode == "nlo3d":
    t = json.load(open(args.table)) if args.table and os.path.exists(args.table) else {}
    if "delta_S" not in t:
        record["error"] = (f"nlo3d needs a finished table; none usable at "
                           f"{args.table!r}")
    else:
        corr = abelian_higgs_3d.NLOCorrection(t["T_GeV"], t["delta_S"], S3_LO=t.get("S3_LO"))
        record["table_range"] = list(corr.T_range)

if record["error"] is None:
    base = load_potential(os.path.join(ROOT, MODEL_FILE), "specific_potential")

    def factory(p, *a, **kw):
        pot = base(p, *a, **kw)
        conf = pot.config.nucleationConf
        # the 3D model switches on its NLO rate (Delta S and S_6 from the shipped grid) by default; every mode sets
        # what it computes explicitly
        conf.include_determinants = conf.include_dim6 = False
        if args.mode == "4d":
            conf.rate_mode = "S3overT_4D"          # the 4D daisy rate of the parent model
        else:
            conf.rate_mode = "model"
            conf.mu4_over_T = (args.mu4_over_T if args.mu4_over_T is not None
                               else MU4_CENTRAL * args.mu4)
            conf.matching_order = args.matching
            if args.input_scheme is not None:
                conf.input_scheme = args.input_scheme
            if args.mu3_choice is not None:
                conf.mu3_choice = args.mu3_choice
        if corr is not None:                       # nlo3d: Delta S of this point's own table, without S_6
            pot.nlo_correction = corr
            conf.include_determinants = True
        # the default ode integral backend fails in 3D mode, so the double integral is used
        pc = pot.config.percolationConf
        pc.integral_method = "double_integral"
        pc.algorithm_mode = args.algorithm
        return pot

    try:
        context, error = _compute_single_point(
            dict(params), factory, timeout=-1, verbose=False,
            include_smbhb=False, max_stage=3)
        if error is not None:
            record["error"] = str(error)
        res = context if isinstance(context, dict) else getattr(context, "__dict__", {})

        def walk(d, depth=0):
            if depth > 3 or not isinstance(d, dict):
                return
            for k, v in d.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    if k in OBSERVABLES:
                        record["observables"].setdefault(k, float(v))
                    if k.startswith(("Tperc", "Tnuc", "Tc", "Treh")):
                        record["temperature_keys"].setdefault(k, float(v))
                elif isinstance(v, dict):
                    walk(v, depth + 1)

        walk(res)
    except Exception:
        record["error"] = traceback.format_exc()

tp = record["observables"].get("Tperc_SM_GeV")
if corr is not None and tp is not None:
    record["extrapolating"] = bool(corr.extrapolating(tp))
record["seconds"] = time.time() - t0
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
with open(args.out, "w") as f:
    json.dump(record, f, indent=2)
