"""The NANOGrav 15 yr likelihood of the transition with the binaryfog population, per prior set of
binaryfog.priors, profiled and marginalised over the six astrophysical parameters, on the (g, log10 v) grid of
plane_scan.py; and the binaries alone, profiled and marginalised (evidence).

    PRIORS=nanograv,recent python prior_study.py        (WORKERS processes; NDRAW prior draws, default 32768)
Writes prior_study_<prior>.npz to the work folder (paths.WORK); collect_prior_study.py copies what the figures read
into paper/data."""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import paths  # noqa: E402  (src/, models/ and the folders of paper/reproduce on sys.path)
sys.argv = sys.argv[:1]                       # marginal_foreground reads draw numbers from argv
import plane_scan as ps  # noqa: E402
import marginal_foreground as mf  # noqa: E402
from binaryfog import priors as bfp  # noqa: E402
from binaryfog.background import PARAM_NAMES, PRIOR_BOX  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

WORKERS = int(os.environ.get("WORKERS", "56"))
NDRAW = int(os.environ.get("NDRAW", "32768"))
# binaryfog names two prior sets differently (same values): the file names here keep the short ones
BF_NAME = {"literature": "tomczak_kormendy", "recent": "survey_average"}
PRIORS = os.environ.get("PRIORS", "nanograv,literature,recent").split(",")
FOG = ps.BF_FOG
LO = np.array([PRIOR_BOX[k][0] for k in PARAM_NAMES]); HI = np.array([PRIOR_BOX[k][1] for k in PARAM_NAMES])


def draws(rng, n, which):
    """Draws from a binaryfog prior set truncated to the library box: its Gaussians, flat on the rest."""
    gauss = bfp.GAUSSIAN_SETS[BF_NAME.get(which, which)]
    out = []
    while sum(len(x) for x in out) < n:
        th = rng.uniform(LO, HI, size=(4 * n, len(PARAM_NAMES)))
        for k, name in enumerate(PARAM_NAMES):
            if name in gauss:
                th[:, k] = rng.normal(*gauss[name], size=len(th))
        out.append(th[np.all((th >= LO) & (th <= HI), axis=1)])
    return np.concatenate(out)[:n]


def profile_one(pt, which, starts):
    """max over theta of ln L(pt, theta) + ln prior(theta), Nelder-Mead from the given starts."""
    def neg(th):
        lp = bfp.ln_prior(th, BF_NAME.get(which, which))
        return 1e30 if not np.isfinite(lp) else -(FOG.ln_like(pt, th) + lp)
    best = None
    for x0 in starts:
        r = minimize(neg, x0, method="Nelder-Mead", options={"maxfev": 600, "xatol": 1e-3, "fatol": 1e-3})
        if best is None or r.fun < best.fun:
            best = r
    return -best.fun, best.x


def profile_row(job):
    """The profile along one row of the grid (fixed g), warm-started from the previous v."""
    i, row, which, seeds = job
    out = np.full(len(row), np.nan)
    prev = None
    for j, pt in enumerate(row):
        if not np.all(np.isfinite(pt)):
            continue
        vals = np.array([FOG.ln_like(pt, th) + bfp.ln_prior(th, BF_NAME.get(which, which)) for th in seeds])
        starts = [seeds[int(np.argmax(vals))]] + ([prev] if prev is not None else [])
        out[j], prev = profile_one(pt, which, starts)
    return i, out


if __name__ == "__main__":
    from multiprocessing import get_context
    t0 = time.time()
    spec = np.full((len(ps.G), len(ps.LOGV), mf.NB), np.nan)
    with get_context("fork").Pool(WORKERS) as pool:
        for i, row in pool.imap_unordered(mf.spectrum_row, range(len(ps.G))):
            spec[i] = row
    ok = np.all(np.isfinite(spec), axis=-1)
    pts = spec[ok]
    print(f"spectra: {ok.sum()} of {ok.size} points with a transition, {time.time() - t0:.0f} s", flush=True)
    for which in PRIORS:
        rng = np.random.default_rng(20261106)
        th = draws(rng, NDRAW, which)
        out = {"G": ps.G, "LOGV": ps.LOGV, "prior": which, "ndraw": NDRAW}
        # marginal of the combined model, in two halves for the convergence
        halves = {}
        for h, sel in (("A", slice(0, NDRAW // 2)), ("B", slice(NDRAW // 2, NDRAW))):
            blocks = np.array_split(th[sel], 4 * WORKERS)
            with get_context("fork").Pool(WORKERS) as pool:
                parts = pool.map(mf.accumulate, [("holodeck", b, pts) for b in blocks])
            halves[h] = mf.combine(parts)
        m, s1, s2 = mf.combine([halves["A"], halves["B"]])
        for key, (mm, ss, nn) in (("lnL_marg", (m, s1, NDRAW)), ("lnL_marg_A", (halves["A"][0], halves["A"][1], NDRAW // 2)),
                                  ("lnL_marg_B", (halves["B"][0], halves["B"][1], NDRAW // 2))):
            a = np.full(ok.shape, np.nan); a[ok] = mm + np.log(ss) - np.log(nn); out[key] = a
        ess = np.full(ok.shape, np.nan); ess[ok] = s1 ** 2 / s2; out["ess"] = ess
        # binaries alone: evidence over the prior, and the profile under the prior
        zero = np.zeros((1, mf.NB))
        l0 = np.array([mf.lnl_holodeck(zero, x)[0] for x in th])
        mx = l0.max(); out["lnZ_bin"] = float(mx + np.log(np.mean(np.exp(l0 - mx))))
        out["lnZ_bin_halves"] = np.array([float(mx + np.log(np.mean(np.exp(l0[s] - mx))))
                                          for s in (slice(0, NDRAW // 2), slice(NDRAW // 2, NDRAW))])
        seeds = th[np.argsort(-l0)[:8]]
        out["lnL_bin_prof"], out["theta_bin_prof"] = profile_one(None, which, list(seeds))
        # profile of the combined model on the grid
        seeds64 = th[:64]
        prof = np.full(ok.shape, np.nan)
        with get_context("fork").Pool(WORKERS) as pool:
            for i, row in pool.imap_unordered(profile_row, [(i, spec[i], which, seeds64) for i in range(len(ps.G))]):
                prof[i] = row
        out["lnL_prof"] = prof
        np.savez(paths.work(f"prior_study_{which}.npz"), **out)
        print(f"{which}: max lnL_marg {np.nanmax(out['lnL_marg']):.3f}, max lnL_prof {np.nanmax(prof):.3f}, "
              f"lnZ_bin {out['lnZ_bin']:.3f}, lnL_bin_prof {out['lnL_bin_prof']:.3f}; {time.time() - t0:.0f} s", flush=True)
