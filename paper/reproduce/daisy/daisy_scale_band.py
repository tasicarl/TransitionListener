"""Scale variations of the 4D daisy rate (V_4D + Daisy) for the rate figure, at g = 0.5 on the line v = e^{1/6} mu0/g.

The 4D one-loop potential of the model file has no scale of its own: its logarithms are normalised to the masses at
phi = v and its gauge coupling is g(mu0). As for the three-dimensional EFT, whose leading-order band comes from the
running of the couplings to the matching scale mu_4 = 4 pi e^{-gamma_E} T, the band takes the gauge coupling of the
whole potential at mu = xi mu_4(T), xi = 1/2, 1, 2, with its one-loop running g^2(mu) = g^2 / (1 - g^2 ln(mu/mu0) / 24 pi^2)
(scalar QED with one charged scalar), and the quartic again from the radiative condition at that coupling. v is kept.
Kinds mu00.5 and mu02 take g at mu0/2 and 2 mu0, so that the band can span the vacuum scale as well. Kind 'n5' is the
default potential, g(mu0), checked against <DAISY_CACHE>/daisy_bounce_g0.500.json; the rate figure draws it as the
curve of variant N5, without a band.
Kinds c0.5, c1, c2 (a band for the fixed couplings): the couplings fixed, the logarithms normalised at
c m_i(v) and the quartic again from radiative symmetry breaking (models/TL_conformal_u1_no_fermions_scale.py); c1 is
the default potential, a check against n5. The varying-g kinds above mix a running g with a fixed quartic. The paper
uses none of the kinds but n5; the band of variant N4 comes from the bounce_rg caches of compute_daisy_actions.py.
    python daisy_scale_band.py                 -> paper/data/scale_band_g0.500.json (all kinds)
    KINDS=c0.5,c1,c2 python daisy_scale_band.py -> the same file, these kinds added or replaced"""
import json
import math
import os
import sys
import warnings
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import paths  # noqa: E402  (the folders of paper/reproduce on sys.path)
import compute_daisy_actions as C  # noqa: E402

G0 = 0.5
V = C.v_ref(G0)
KINDS = tuple(os.environ["KINDS"].split(",")) if os.environ.get("KINDS") else (
    "n5", "0.5", "1", "2", "mu00.5", "mu02", "c0.5", "c1", "c2")   # xi mu_4(T); mu0/2 and 2 mu0; c m_i(v)
MODEL_FILE_SCALE = os.path.join(C.ROOT, "models", "TL_conformal_u1_no_fermions_scale.py")
MU4 = 4 * math.pi * math.exp(-0.5772156649015329)


def g_run(mu):
    return G0 / math.sqrt(1 - G0 ** 2 * math.log(mu / C.MU0) / (24 * math.pi ** 2))


def one(args):
    kind, T = args
    warnings.filterwarnings("ignore")
    from transitionlistener.interface.single_point import load_potential
    if kind.startswith("c"):
        g = G0
        pot = load_potential(MODEL_FILE_SCALE, "specific_potential")({"g": g, "v_GeV": V, "mu_over_mv": float(kind[1:])},
                                                                    verbose=False)
    else:
        base = load_potential(C.MODEL_FILE, "specific_potential")
        g = G0 if kind == "n5" else g_run(float(kind[3:]) * C.MU0) if kind.startswith("mu0") else g_run(float(kind) * MU4 * T)
        pot = base({"g": g, "v_GeV": V}, verbose=False)
    T_int = T / pot.conversionFactor
    S, ok = C.action_bounce(pot, T_int)
    return kind, T, g, S


if __name__ == "__main__":
    ref = json.load(open(os.path.join(C.CACHE, "daisy_bounce_g0.500.json")))
    Ts = [r["T_GeV"] for r in ref["rows"] if 2e-6 < r["T_GeV"] < 4.5e-4]
    jobs = [(k, T) for k in KINDS for T in Ts]
    with Pool(int(os.environ.get("WORKERS", "40"))) as p:
        res = p.map(one, jobs, chunksize=1)
    path = paths.data("scale_band_g0.500.json")
    out = json.load(open(path)) if os.path.exists(path) else {}
    assert not out or out["T_GeV"] == Ts, "the temperature grid of the existing file differs"
    out.update({"g": G0, "v_GeV": V, "T_GeV": Ts,
                "ref_S3_over_T": [r["S3_over_T"] for r in ref["rows"] if 2e-6 < r["T_GeV"] < 4.5e-4]})
    for k in KINDS:
        out[k] = {"g": [r[2] for r in res if r[0] == k], "S3_over_T": [r[3] for r in res if r[0] == k]}
    json.dump(out, open(path, "w"), indent=1)
    print("n5 vs cache max |diff|", np.nanmax(np.abs(np.array(out["n5"]["S3_over_T"]) - np.array(out["ref_S3_over_T"]))))
    for k in [k for k in KINDS if k != "n5"]:
        d = np.array(out[k]["S3_over_T"]) - np.array(out["n5"]["S3_over_T"])
        print(k, "g", min(out[k]["g"]), max(out[k]["g"]), "S - S_n5:", np.nanmin(d), np.nanmax(d))
