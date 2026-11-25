"""Run TransitionListener at one point of the benchmark line with its defaults (the 3D rate at NLO with the
shipped grid, zero_quartic, mu4 = 7.0553 T, mu3 = gT, the ODE percolation integral, adaptive step size), and
capture the interpolants that enter the mean bubble separation: the action S(T), the true-vacuum fraction P(T),
H(T), 3 c_s^2(T) and ln s(T), evaluated on a dense temperature grid. Writes
<work>/tl_capture/capture_g<g><tag>.npz, which accuracy_table.py reads.

The rate includes the dimension-six shift of the grid; TL_DIM6=0 leaves it out. The point's v is read from the
name of its folder g<g>_u1_v<v> in paths.LINESCAN_POINTS (linescan/scan_point.py).

Usage: python capture_tl.py --g 0.6 [--src /path/to/TL/src] [--scheme zero_quartic|CW] [--integral ...] [--tag _x]"""
import argparse
import glob
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--g", type=float, required=True)
ap.add_argument("--src", default=None)
ap.add_argument("--tag", default="")
ap.add_argument("--integral", default=None, help="percolationConf.integral_method; default: TL's default (the ODE)")
ap.add_argument("--scheme", default=None, help="nucleationConf.input_scheme; default: TL's (zero_quartic)")
args = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import paths  # noqa: E402  (src/, models/ and the folders of paper/reproduce on sys.path)
CODE = paths.CODE      # the tree that holds src/ and models/ (TL_CODE_ROOT)
if args.src:
    sys.path.insert(0, args.src)
import numpy as np  # noqa: E402
from transitionlistener import bubbledynamics as bd, transitionObservables as TO  # noqa: E402
from transitionlistener.interface import _compute_single_point  # noqa: E402
from transitionlistener.interface.single_point import load_potential  # noqa: E402
print("TL from", bd.__file__)

d = glob.glob(os.path.join(paths.LINESCAN_POINTS, f"g{args.g:.3f}_u1_v*"))[0]
v = float(d.split("_v")[-1])
base = load_potential(os.path.join(CODE, "models", "TL_conformal_u1_no_fermions_3d.py"), "specific_potential")
POT = {}


def factory(p, *a, **kw):
    pot = base(p, *a, **kw)
    if args.scheme:
        pot.config.nucleationConf.input_scheme = args.scheme
    if os.environ.get("TL_DIM6", "1") == "0":
        pot.config.nucleationConf.include_dim6 = False
    if args.integral:
        pot.config.percolationConf.integral_method = args.integral
    POT["pot"] = pot
    return pot


CAP = {}
_orig = TO.calcMeanBubbleSeparation


def spy(T, Tmax, Sint, Pint, Hint, entropyInt=None, coolingInt=None, verbose=False, *, logEntropyInt=None):
    CAP.update(T=T, Tmax=Tmax, Sint=Sint, Pint=Pint, Hint=Hint, coolingInt=coolingInt, logEntropyInt=logEntropyInt)
    R = _orig(T, Tmax, Sint, Pint, Hint, entropyInt=entropyInt, coolingInt=coolingInt, verbose=verbose,
              logEntropyInt=logEntropyInt)
    CAP["R"] = R
    return R


TO.calcMeanBubbleSeparation = spy
ctx, err = _compute_single_point({"g": args.g, "v_GeV": v}, factory, timeout=-1, verbose=False,
                                 include_smbhb=False, max_stage=3)
print("error:", err)
res = ctx if isinstance(ctx, dict) else ctx.__dict__
obs = {}


def walk(x, depth=0):
    if depth > 3 or not isinstance(x, dict):
        return
    for k, val in x.items():
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            obs.setdefault(k, float(val))
        elif isinstance(val, dict):
            walk(val, depth + 1)


walk(res)
pot = POT["pot"]
Tp, Tmax = CAP["T"], CAP["Tmax"]
T = np.geomspace(Tp / 1.5, min(Tmax, 3 * Tp), 6000)
out = dict(g=args.g, v=v, conv=pot.conversionFactor, Tp=Tp, Tmax=Tmax, R=CAP["R"], T=T,
           S=CAP["Sint"](T), P=CAP["Pint"](T), H=CAP["Hint"](T),
           cool=CAP["coolingInt"](T) if CAP["coolingInt"] else np.ones_like(T),
           lns=CAP["logEntropyInt"](T) if CAP["logEntropyInt"] else 3 * np.log(T),
           obs_keys=np.array(list(obs.keys())), obs_vals=np.array(list(obs.values())))
# the support grids of the interpolants (interp1d keeps its nodes in .x/.y)
for name in ("Sint", "Pint", "Hint"):
    f = CAP[name]
    if hasattr(f, "x") and hasattr(f, "y"):
        out[name + "_x"], out[name + "_y"] = np.asarray(f.x), np.asarray(f.y)
    elif hasattr(f, "x"):
        out[name + "_x"] = np.asarray(f.x)
os.makedirs(paths.work("tl_capture"), exist_ok=True)
np.savez(paths.work("tl_capture", f"capture_g{args.g:.3f}{args.tag}.npz"), **out)
print("Tp_GeV", Tp * pot.conversionFactor, "RH", CAP["R"] * CAP["Hint"](Tp), "betaH_RH", obs.get("betaH_RH"))
