"""The preference for the phase transition at fixed binary lifetime tau_f, with the inner hardening index nu_inner
profiled together with the other four astrophysical parameters (Nelder-Mead under their prior), (g, v) maximised over
the same 40 candidates as preference_map.py. Seeds for nu_inner: five values across its prior and the best grid cells of preference_map.npz.
    python preference_tau_profile.py   -> paper/data/preference_tau_profile.npz"""
import os
import time

import numpy as np

import preference_map as M  # noqa: E402   (same likelihoods, priors and candidate selection)
from preference_map import P, ps, mf, ipf, bf, bfp, pst, sigma_from_gain, minimize, LIKES, top  # noqa: E402

HERE = M.HERE
TAU = np.unique(np.round(np.concatenate([np.geomspace(0.1, 11.0, 49), np.linspace(0.1, 11.0, 110)]), 5))  # log and lin
FREE5 = [k for k in range(len(bfp.PARAM_NAMES)) if k != M.I_TAU]
J_NU = FREE5.index(M.I_NU)


def full5(x5, tau):
    th = np.empty(len(bfp.PARAM_NAMES)); th[FREE5] = x5; th[M.I_TAU] = tau
    return th


def prof(like, pt, which, tau, starts):
    def neg(x5):
        th = full5(x5, tau); lp = bfp.ln_prior(th, which)
        return 1e30 if not np.isfinite(lp) else -(like.ln_like(pt, th) + lp)
    best = None
    for x0 in starts:
        r = minimize(neg, x0, method="Nelder-Mead", options={"maxfev": 3000, "xatol": 1e-4, "fatol": 1e-4})
        if best is None or r.fun < best.fun:
            best = r
    return -best.fun, best.x


def cell(args):
    key, which, tau, pts, seeds, nu_seeds = args
    like = LIKES[key]
    starts = []
    for nu in nu_seeds:          # the best four-parameter seeds at each seed value of nu_inner
        cand = [np.insert(x, J_NU, nu) for x in seeds]
        v = [like.ln_like(None, full5(x, tau)) + bfp.ln_prior(full5(x, tau), which) for x in cand]
        starts += [cand[i] for i in np.argsort(v)[::-1][:2]]
    lbin, xb = prof(like, None, which, tau, starts)
    lpt, xp = -np.inf, None
    for pt in pts:
        v = [like.ln_like(pt, full5(x, tau)) + bfp.ln_prior(full5(x, tau), which) for x in starts]
        l, x = prof(like, pt, which, tau, [starts[int(np.argmax(v))], xb])
        if l > lpt:
            lpt, xp = l, x
    return key, which, tau, lpt, lbin, xp[J_NU], xb[J_NU]


if __name__ == "__main__":
    from multiprocessing import get_context
    t0 = time.time()
    MAP = np.load(P.paths.work("preference_map.npz"))
    with get_context("fork").Pool(P.WORKERS) as pool:
        ng = np.full((len(ps.G), len(ps.LOGV), mf.NB), np.nan)
        for i, row in pool.imap_unordered(mf.spectrum_row, range(len(ps.G))):
            ng[i] = row
    data = LIKES["cp_v2"].data
    a = np.full(ng.shape[:2], np.nan); ok = np.all(np.isfinite(ng), axis=-1)
    a[ok] = [bf.Likelihood(data).ln_like(p) for p in ng[ok]]
    jobs = []
    for key, which in M.PANELS:
        rng = np.random.default_rng(20261106)
        th = pst.draws(rng, 4096, which)
        cand = P.top_of_prior_study(which) | top(a)
        cand = sorted(c for c in cand if np.all(np.isfinite(ng[c])))
        l0 = np.array([LIKES[key].ln_like(None, x) + bfp.ln_prior(x, which) for x in th])
        lbin0, thbin = P.profile(LIKES[key], None, which, list(th[np.argsort(-l0)[:8]]))
        with get_context("fork").Pool(P.WORKERS) as pool:
            vals = pool.map(P.job, [(key, which, ng[c], th[:64], thbin) for c in cand], chunksize=4)
        pts = [ng[cand[i]] for i in np.argsort(vals)[::-1][:M.NKEEP]]
        seeds = th[:256][:, M.FREE]
        tag = f"{key}_{which}"
        for tau in TAU:
            # nu_inner seeds: five values across the prior, and the grid cells of the map that maximise each
            # likelihood at the two neighbouring tau_f (the best nu_inner can jump between the edges of the prior)
            nus = {-1.5, -1.125, -0.75, -0.375, 0.0}
            for i in np.argsort(abs(np.log(MAP["tau"]) - np.log(tau)))[:2]:
                nus |= {float(MAP["nu"][np.argmax(MAP[f"lnL_pt_{tag}"][i])]),
                        float(MAP["nu"][np.argmax(MAP[f"lnL_bin_{tag}"][i])])}
            jobs.append((key, which, tau, pts, seeds, sorted(nus)))
        print(f"{tag}: {len(cand)} candidates; {time.time() - t0:.0f} s", flush=True)
    with get_context("fork").Pool(P.WORKERS) as pool:
        res = pool.map(cell, jobs, chunksize=1)
    out = {}
    for key, which in M.PANELS:
        tag = f"{key}_{which}"
        r = sorted([x for x in res if (x[0], x[1]) == (key, which)], key=lambda x: x[2])
        LPT, LB = np.array([x[3] for x in r]), np.array([x[4] for x in r])
        out[f"lnL_pt_{tag}"], out[f"lnL_bin_{tag}"] = LPT, LB
        out[f"nu_pt_{tag}"], out[f"nu_bin_{tag}"] = np.array([x[5] for x in r]), np.array([x[6] for x in r])
        out[f"gain_{tag}"] = LPT - LB
        out[f"sigma_{tag}"] = np.array([sigma_from_gain(x, 2) for x in LPT - LB])
        print(tag, "sigma", out[f"sigma_{tag}"].round(2), flush=True)
    np.savez(P.paths.data("preference_tau_profile.npz"), tau=TAU, **out)
    print(f"wrote preference_tau_profile.npz; {time.time() - t0:.0f} s", flush=True)
