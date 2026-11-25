"""The preference for the phase transition at fixed binary lifetime tau_f and inner hardening index nu_inner, for the
figure of the dependence on the hardening of the binaries. Same likelihoods, priors and conversion as preference.py; the
other four astrophysical parameters are profiled under their prior (Nelder-Mead), g and v maximised over the 40 grid
points with the largest profiled likelihood of preference.py's candidate set (the transition fixes (g, v) whatever the
binaries do).
    python preference_map.py          -> <work>/preference_map.npz (read by preference_tau_profile.py)
The minimum of the map over tau_f >= tau_min (and all nu_inner) reproduces the runs of preference.py with TAU_MIN
(paper/data/preference_tau*.json)."""
import os
import sys
import time

import numpy as np

os.environ.setdefault("PRIORS", "nanograv")
import preference as P  # noqa: E402   (imports plane_scan, binaryfog; TAU_MIN at its default)
from preference import ps, mf, ipf, bf, bfp, pst, sigma_from_gain, minimize, LIKES, top  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TAU = np.round(np.geomspace(0.1, 11.0, 25), 4)
NU = np.round(np.linspace(-1.5, 0.0, 13), 4)
PANELS = [("cp_v2", "nanograv"), ("cp_v2", "literature"), ("cp_v2", "recent")]
NKEEP = 40
I_TAU = list(bfp.PARAM_NAMES).index("hard_time")
I_NU = list(bfp.PARAM_NAMES).index("hard_gamma_inner")
FREE = [k for k in range(len(bfp.PARAM_NAMES)) if k not in (I_TAU, I_NU)]


def full(x4, tau, nu):
    th = np.empty(len(bfp.PARAM_NAMES)); th[FREE] = x4; th[I_TAU] = tau; th[I_NU] = nu
    return th


def prof_fixed(like, pt, which, tau, nu, starts):
    def neg(x4):
        th = full(x4, tau, nu); lp = bfp.ln_prior(th, which)
        return 1e30 if not np.isfinite(lp) else -(like.ln_like(pt, th) + lp)
    best = None
    for x0 in starts:
        r = minimize(neg, x0, method="Nelder-Mead", options={"maxfev": 1500, "xatol": 1e-4, "fatol": 1e-4})
        if best is None or r.fun < best.fun:
            best = r
    return -best.fun, best.x


def cell(args):
    key, which, tau, nu, pts, seeds = args
    like = LIKES[key]
    vals = [like.ln_like(None, full(x, tau, nu)) + bfp.ln_prior(full(x, tau, nu), which) for x in seeds]
    starts = [seeds[i] for i in np.argsort(vals)[::-1][:4]]
    lbin, xb = prof_fixed(like, None, which, tau, nu, starts)
    lpt = -np.inf
    for pt in pts:
        v = [like.ln_like(pt, full(x, tau, nu)) + bfp.ln_prior(full(x, tau, nu), which) for x in seeds[:64]]
        l, _ = prof_fixed(like, pt, which, tau, nu, [seeds[int(np.argmax(v))], xb])
        lpt = max(lpt, l)
    return key, which, tau, nu, lpt, lbin


if __name__ == "__main__":
    from multiprocessing import get_context
    t0 = time.time()
    with get_context("fork").Pool(P.WORKERS) as pool:
        ng = np.full((len(ps.G), len(ps.LOGV), mf.NB), np.nan)
        for i, row in pool.imap_unordered(mf.spectrum_row, range(len(ps.G))):
            ng[i] = row
        ip = np.full((len(ps.G), len(ps.LOGV), ipf.NB), np.nan)
        for i, row in pool.imap_unordered(ipf.spectrum_row, range(len(ps.G))):
            ip[i] = row
    spec = {"cp_v2": ng, "ipta2": ip}
    IF = np.load(P.paths.data("ipta_fog.npz"))
    data = LIKES["cp_v2"].data
    a = np.full(ng.shape[:2], np.nan); ok = np.all(np.isfinite(ng), axis=-1)
    a[ok] = [bf.Likelihood(data).ln_like(p) for p in ng[ok]]
    jobs, keep = [], {}
    for key, which in PANELS:
        rng = np.random.default_rng(20261106)
        th = pst.draws(rng, 4096, which)
        cand = P.top_of_prior_study(which)
        cand |= (top(IF["lnL"]) | top(IF["lnL_holodeck"])) if key == "ipta2" else top(a)
        cand = sorted(c for c in cand if np.all(np.isfinite(spec[key][c])))
        # rank the candidates by the profiled likelihood with free hardening, as preference.py, and keep the best
        l0 = np.array([LIKES[key].ln_like(None, x) + bfp.ln_prior(x, which) for x in th])
        lbin0, thbin = P.profile(LIKES[key], None, which, list(th[np.argsort(-l0)[:8]]))
        with get_context("fork").Pool(P.WORKERS) as pool:
            vals = pool.map(P.job, [(key, which, spec[key][c], th[:64], thbin) for c in cand], chunksize=4)
        best = [cand[i] for i in np.argsort(vals)[::-1][:NKEEP]]
        keep[(key, which)] = best
        seeds = th[:, FREE]
        pts = [spec[key][c] for c in best]
        for tau in TAU:
            for nu in NU:
                jobs.append((key, which, tau, nu, pts, seeds[:256]))
        print(f"{key} {which}: {len(cand)} candidates, best {max(vals):.3f}; {time.time() - t0:.0f} s", flush=True)
    out = {}
    with get_context("fork").Pool(P.WORKERS) as pool:
        res = pool.map(cell, jobs, chunksize=1)
    for key, which in PANELS:
        G = np.full((len(TAU), len(NU)), np.nan); LPT = G.copy(); LB = G.copy()
        for k_, w_, tau, nu, lpt, lbin in res:
            if (k_, w_) == (key, which):
                i, j = int(np.where(TAU == tau)[0][0]), int(np.where(NU == nu)[0][0])
                LPT[i, j], LB[i, j], G[i, j] = lpt, lbin, lpt - lbin
        tag = f"{key}_{which}"
        out[f"gain_{tag}"] = G; out[f"lnL_pt_{tag}"] = LPT; out[f"lnL_bin_{tag}"] = LB
        out[f"sigma_{tag}"] = np.vectorize(lambda x: sigma_from_gain(x, 2))(G)
        print(tag, "sigma range", np.nanmin(out[f"sigma_{tag}"]).round(2), np.nanmax(out[f"sigma_{tag}"]).round(2), flush=True)
    np.savez(P.paths.work("preference_map.npz"), tau=TAU, nu=NU, **out)
    print(f"wrote preference_map.npz; {time.time() - t0:.0f} s", flush=True)
