"""How much the PTA fits improve when the binaryfog population is added to the transition (the Delta ln L column of
the switch tables for the population rows of B1).

For each data set and population prior set: the maximum of ln L + ln pi over (g, log10 v) and the six astrophysical
parameters theta (outer Nelder-Mead in (g, log10 v), inner Nelder-Mead in theta as prior_study.profile_one), and
ln L alone, without ln pi, at that maximum. The reference is the maximum of ln L without binaries, evaluated with the
same likelihood (NANOGrav: binaryfog cp_v2 as the scan; IPTA DR2: binaryfog ipta2_cp_v1 with the IPTA maps, as
ipta_fog.py). As a check, the maximum with theta free inside the library box and no prior.
Run with PYTHONPATH=<repo>/src in the environment of plane_scan.py:
    python bhb_population_dlnl.py       -> paper/data/bhb_population_dlnl.json"""
import json
import os
import sys
from multiprocessing import get_context

import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv = sys.argv[:1]
sys.path.insert(0, os.path.dirname(HERE))
import paths  # noqa: E402  (src/, models/ and the folders of paper/reproduce on sys.path)
import plane_scan as ps  # noqa: E402
import prior_study as pst  # noqa: E402
import ipta_fog as ipf  # noqa: E402
from binaryfog import priors as bfp  # noqa: E402

PRIORS = ("recent", "nanograv", "literature")


def spectrum(x, freqs):
    g, lv = float(x[0]), float(x[1])
    if not (ps.G[0] <= g <= ps.G[-1] and ps.LOGV[0] <= lv <= ps.LOGV[-1]):
        return None
    try:
        _, bh, al, bv = ps.central(g, lv)
        if not (np.isfinite(bh) and np.isfinite(al) and bv > 0):
            return None
        sp = ps.make_spectrum(ps.COSMO_VARIANTS[0][1], ps.reheating(g, lv), bh, al, bv)
        return None if sp is None else sp(freqs)
    except Exception:
        return None


def lnl_fns(ds):
    """(frequencies, ln L without binaries, ln L with the population at theta)."""
    if ds == "NG15":
        F = ps.BF_FOG
        return F.data.freqs, (lambda pt: float(F.ln_like(pt))), (lambda pt, th: float(F.ln_like(pt, th)))
    return ipf.FREQS, (lambda pt: float(ipf.LIKE.ln_like(pt))), (lambda pt, th: float(ipf.lnl_holodeck(pt[None, :], th)[0]))


def inner(pt, l_th, which, seeds, prev=None):
    """theta maximising ln L + ln pi (which = None: ln L alone inside the library box); returns (ln L, ln pi, theta)."""
    def lp(th):
        if which is None:
            return 0.0 if np.all((th >= pst.LO) & (th <= pst.HI)) else -np.inf
        return bfp.ln_prior(th, pst.BF_NAME.get(which, which))

    def neg(th):
        p = lp(th)
        return 1e30 if not np.isfinite(p) else -(l_th(pt, th) + p)
    vals = [neg(th) for th in seeds]
    starts = [seeds[int(np.argmin(vals))]] + ([prev] if prev is not None else [])
    best = min((minimize(neg, x0, method="Nelder-Mead", options={"maxfev": 1500, "xatol": 1e-4, "fatol": 1e-4})
                for x0 in starts), key=lambda r: r.fun)
    return l_th(pt, best.x), lp(best.x), best.x


def job(args):
    ds, which, x0s = args
    freqs, l0, l_th = lnl_fns(ds)
    seeds = pst.draws(np.random.default_rng(20261106), 256, which if which in PRIORS else "recent")
    state = {}

    def neg_outer(x):
        pt = spectrum(x, freqs)
        if pt is None:
            return 1e30
        if which == "none":
            return -l0(pt)
        l, p, th = inner(pt, l_th, None if which == "free" else which, seeds, state.get("th"))
        state["th"] = th
        return -(l + p)
    best = None
    for x0 in x0s:
        x0 = np.array(x0, float)
        r = minimize(neg_outer, x0, method="Nelder-Mead",
                     options=dict(initial_simplex=[x0, x0 + [0.004, 0.0], x0 + [0.0, 0.04]], xatol=2e-5, fatol=1e-3, maxfev=200))
        if best is None or r.fun < best.fun:
            best = r
    pt = spectrum(best.x, freqs)
    if which == "none":
        return ds, which, dict(at=list(map(float, best.x)), lnL=-float(best.fun))
    l, p, th = inner(pt, l_th, None if which == "free" else which, seeds)
    return ds, which, dict(at=list(map(float, best.x)), lnL=float(l), lnpi=float(p), theta=list(map(float, th)),
                           lnL_plus_lnpi=float(l + p))


if __name__ == "__main__":
    bf = json.load(open(paths.data("best_fits.json")))["nlo"]
    jobs = []
    for ds in ("NG15", "IPTA2"):
        x0s = [bf[ds]["at"]]
        if ds + "_bhb" in bf:
            x0s.append(bf[ds + "_bhb"]["at"])
        for which in ("none", "free") + PRIORS:
            jobs.append((ds, which, x0s))
    out = {}
    with get_context("fork").Pool(len(jobs)) as pool:
        for ds, which, r in pool.imap_unordered(job, jobs):
            out.setdefault(ds, {})[which] = r
            print(ds, which, json.dumps(r), flush=True)
    for ds in out:
        ref = out[ds]["none"]["lnL"]
        for which in out[ds]:
            if which != "none":
                out[ds][which]["dlnL"] = out[ds][which]["lnL"] - ref
                print(f"{ds} {which}: Delta ln L = {out[ds][which]['dlnL']:+.3f}", flush=True)
    json.dump(out, open(paths.data("bhb_population_dlnl.json"), "w"), indent=1)
