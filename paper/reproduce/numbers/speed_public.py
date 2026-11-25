"""Seconds per point, potential to GW spectrum, warm interpreter, one core: the public TransitionListener release with
the 4D + Daisy rate against the TransitionListener of this repository with the 3D NLO rate (appendix B). Every point
prints one line "RESULT {json}".

  python speed_public.py ROOT CASE [g v_GeV]...

ROOT: the tree whose src and models are used. CASE:
  public       ROOT = public release, its shipped TL_conformal_dark_u1 (two fermions) at y = 1e-3, the smallest y of its example
  public_nof   ROOT = public release, with the fermion-free model files of this repository copied into its models/
               (the model of the paper, variant N5)
  repo_4d    ROOT = this repository, 4D + Daisy, fermion-free
  repo_3d    ROOT = this repository, 3D EFT at NLO from the shipped table (the default of the paper)"""
import json
import os
import platform
import sys
import time

import numpy as np

ROOT, CASE = os.path.abspath(sys.argv[1]), sys.argv[2]
P = [float(x) for x in sys.argv[3:]] or [0.45, 2.9, 0.48, 2.5, 0.50, 2.0, 0.51, 1.9, 0.55, 1.0, 0.60, 0.5, 0.70, 1.0]
POINTS = list(zip(P[::2], P[1::2]))
MODEL = dict(public="TL_conformal_dark_u1", public_nof="TL_conformal_u1_no_fermions",
             repo_4d="TL_conformal_u1_no_fermions", repo_3d="TL_conformal_u1_no_fermions_3d")[CASE]
sys.argv = sys.argv[:1]
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "models")]
import transitionlistener  # noqa: E402
from transitionlistener.interface.single_point import load_potential  # noqa: E402
from transitionlistener.interface import _compute_single_point  # noqa: E402

assert transitionlistener.__file__.startswith(ROOT), transitionlistener.__file__
model = load_potential(os.path.join(ROOT, "models", MODEL + ".py"), "specific_potential")
cpu = next(line.split(":", 1)[1].strip() for line in open("/proc/cpuinfo") if line.startswith("model name"))


def point(g, v):
    p = dict(g=g, v_GeV=v)
    if CASE == "public":
        p = dict(g=g, y=1e-3, v_GeV=v)
    t = time.perf_counter()
    ctx, err = _compute_single_point(p, model, timeout=-1, verbose=False, include_smbhb=False, max_stage=4)
    dt = time.perf_counter() - t
    found = {}

    def walk(d, depth=0):
        if depth > 4 or not isinstance(d, dict):
            return
        for k, x in d.items():
            if str(k) in ("Tperc", "Treh_SM_GeV", "betaH_RH") and isinstance(x, (float, np.floating)):
                found.setdefault(str(k), float(x))
            elif isinstance(x, dict):
                walk(x, depth + 1)
    walk(ctx if isinstance(ctx, dict) else vars(ctx))
    return dict(case=CASE, g=g, v_GeV=v, seconds=dt, error=None if err is None else str(err)[:160], **found)


if CASE.startswith("repo_"):
    point(0.58, 1.0)                          # compiles and loads
for g, v in POINTS:
    r = point(g, v)
    r.update(host=platform.node(), cpu=cpu, tl=transitionlistener.__file__)
    print("RESULT " + json.dumps(r), flush=True)
