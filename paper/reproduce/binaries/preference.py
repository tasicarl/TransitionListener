"""The preference for the transition over the binaries alone, per prior set of binaryfog.priors, for three likelihoods:
    cp_v2   NANOGrav 15 yr CURN (2026 release): the likelihood of prior_study.py
    hd_v2   NANOGrav 15 yr Hellings-Downs (2026 release), the same population emulator
    ipta2   IPTA DR2 CURN with the population emulated on the IPTA frequencies (Background(maps="ipta2"))
The gain is G = max over (g, v, theta) of [ln L + ln prior(theta)] minus the same maximum of the binaries alone, as in
binaryfog, converted with binaryfog.significance.sigma_from_gain for two free parameters (g, v). The maximum
over (g, v) is taken over candidate grid points: the 300 largest of the profile map of <work>/prior_study_<prior>.npz (CURN),
of the likelihood without foreground of this data set, and of its likelihood marginalised over the NANOGrav prior
(paper/data/ipta_fog.npz for IPTA). At each candidate theta is profiled by Nelder-Mead from the best of 64 prior draws and from the
profile of the binaries alone.
    python preference.py          -> paper/data/preference.json
    TAU_MIN=1 PRIORS=nanograv OUT=x.json python preference.py   lower edge of tau_f at 1 Gyr, one prior set, into
                                                                paper/data/x.json"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import paths  # noqa: E402  (src/, models/ and the folders of paper/reproduce on sys.path)
sys.argv = sys.argv[:1]
import prior_study as pst  # noqa: E402
import plane_scan as ps  # noqa: E402
import marginal_foreground as mf  # noqa: E402
import ipta_fog as ipf  # noqa: E402
import binaryfog as bf  # noqa: E402
from binaryfog import priors as bfp  # noqa: E402
from binaryfog.significance import sigma_from_gain  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

# the short names of two binaryfog prior sets (prior_study.BF_NAME, same values) are accepted here as well
for _old, _new in pst.BF_NAME.items():
    bfp.GAUSSIAN_SETS.setdefault(_old, bfp.GAUSSIAN_SETS[_new])

WORKERS = int(os.environ.get("WORKERS", "40"))
PRIORS = os.environ.get("PRIORS", "nanograv,literature,recent").split(",")
# lower edge of the binary lifetime tau_f in Gyr (the box of binaryfog, 0.1, by default): the short lifetimes assume
# that the binaries pass the final parsec; a larger edge removes them from the prior and from the profile
TAU_MIN = float(os.environ.get("TAU_MIN", "0.1"))
OUT = os.environ.get("OUT", "preference.json")
if TAU_MIN != bfp.PRIOR_BOX["hard_time"][0]:
    bfp.PRIOR_BOX = dict(bfp.PRIOR_BOX, hard_time=(TAU_MIN, bfp.PRIOR_BOX["hard_time"][1]))
    pst.LO[list(bfp.PARAM_NAMES).index("hard_time")] = TAU_MIN
NTOP = 300
LIKES = {"cp_v2": ps.BF_FOG,
         "hd_v2": bf.Likelihood(ps.BF_DATA["hd_v2"], bf.Background()),
         "ipta2": bf.Likelihood(ipf.DATA, ipf.BG)}


def profile(like, pt, which, starts):
    def neg(th):
        lp = bfp.ln_prior(th, which)
        return 1e30 if not np.isfinite(lp) else -(like.ln_like(pt, th) + lp)
    best = None
    for x0 in starts:
        r = minimize(neg, x0, method="Nelder-Mead", options={"maxfev": 1500, "xatol": 1e-4, "fatol": 1e-4})
        if best is None or r.fun < best.fun:
            best = r
    return -best.fun, best.x


def job(args):
    key, which, pt, seeds, extra = args
    like = LIKES[key]
    vals = np.array([like.ln_like(pt, th) + bfp.ln_prior(th, which) for th in seeds])
    return profile(like, pt, which, [seeds[int(np.argmax(vals))], extra])[0]


def top(a, n=NTOP):
    a = np.where(np.isfinite(a), a, -np.inf)
    return set(map(tuple, np.argwhere(a >= np.sort(a.ravel())[-n])))


def top_of_prior_study(which):
    """top() of the profiled map of <work>/prior_study_<which>.npz, as indices of the grid of plane_scan (the map may
    cover only part of that grid, so that its own indices can point to other nodes)."""
    d = np.load(paths.work(f"prior_study_{which}.npz"))
    i, j = [np.argmin(np.abs(grid[:, None] - d[key][None, :]), axis=0) for grid, key in ((ps.G, "G"), (ps.LOGV, "LOGV"))]
    assert np.allclose(ps.G[i], d["G"], atol=1e-8) and np.allclose(ps.LOGV[j], d["LOGV"], atol=1e-8), "grids differ"
    return {(int(i[a]), int(j[b])) for a, b in top(d["lnL_prof"])}


if __name__ == "__main__":
    from multiprocessing import get_context
    t0 = time.time()
    spec = {}
    with get_context("fork").Pool(WORKERS) as pool:
        ng = np.full((len(ps.G), len(ps.LOGV), mf.NB), np.nan)
        for i, row in pool.imap_unordered(mf.spectrum_row, range(len(ps.G))):
            ng[i] = row
        ip = np.full((len(ps.G), len(ps.LOGV), ipf.NB), np.nan)
        for i, row in pool.imap_unordered(ipf.spectrum_row, range(len(ps.G))):
            ip[i] = row
    spec = {"cp_v2": ng, "hd_v2": ng, "ipta2": ip}
    print(f"spectra: {time.time() - t0:.0f} s", flush=True)
    IF = np.load(paths.data("ipta_fog.npz"))
    assert np.allclose(IF["LOGV"], ps.LOGV, atol=0) and np.allclose(IF["G"], ps.G, atol=0)
    nofg = {}
    for key in ("cp_v2", "hd_v2"):
        data = LIKES[key].data
        a = np.full(ng.shape[:2], np.nan)
        ok = np.all(np.isfinite(ng), axis=-1)
        a[ok] = [bf.Likelihood(data).ln_like(p) for p in ng[ok]]
        nofg[key] = a
    res = {}
    for key in LIKES:
        res[key] = {}
        for which in PRIORS:
            rng = np.random.default_rng(20261106)
            th = pst.draws(rng, 4096, which)
            like = LIKES[key]
            l0 = np.array([like.ln_like(None, x) + bfp.ln_prior(x, which) for x in th])
            lbin, thbin = profile(like, None, which, list(th[np.argsort(-l0)[:8]]))
            cand = top_of_prior_study(which)
            if key == "ipta2":
                cand |= top(IF["lnL"]) | top(IF["lnL_holodeck"])
            else:
                cand |= top(nofg[key])
            cand = sorted(c for c in cand if np.all(np.isfinite(spec[key][c])))
            with get_context("fork").Pool(WORKERS) as pool:
                vals = pool.map(job, [(key, which, spec[key][c], th[:64], thbin) for c in cand], chunksize=4)
            k = int(np.argmax(vals))
            gain = float(vals[k] - lbin)
            res[key][which] = {"lnL_prof_max": float(vals[k]), "at_g": float(ps.G[cand[k][0]]),
                               "at_log10v": float(ps.LOGV[cand[k][1]]), "lnL_bin_prof": float(lbin),
                               "gain": gain, "sigma_2dof": float(sigma_from_gain(gain, 2)),
                               "sigma_3dof": float(sigma_from_gain(gain, 3)), "ncand": len(cand),
                               "theta_bin": [float(x) for x in thbin], "tau_min": TAU_MIN}
            print(key, which, {k_: round(v, 3) if isinstance(v, float) else v for k_, v in res[key][which].items()},
                  f"{time.time() - t0:.0f} s", flush=True)
            json.dump(res, open(paths.data(OUT), "w"), indent=1)
    print(f"wrote {OUT}", flush=True)
