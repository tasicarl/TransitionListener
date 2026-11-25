"""IPTA DR2 with the binaryfog population (B2 for IPTA), on the grid of plane_scan.py.

The population emulator on the IPTA DR2 frequencies (binaryfog Background(maps="ipta2"): the shipped six-parameter maps
fitted again to the same 2000 holodeck models on the 30 IPTA bins)
and the IPTA DR2 free spectrum through binaryfog (ipta2_cp_v1, the product PTArcade reads), 13 bins as the scan.
    lnL          IPTA through binaryfog, no foreground: the reference of B1 and B2, so that both maps of a row share one
                 evaluation of the data. Near the best fit it differs from the scan's IPTA2 map (PTArcade, ceffyl) by a
                 constant, 22.94 +- 0.05; far from the data single bins differ by up to 3.5 (ceffyl reads the cell of
                 the KDE, binaryfog interpolates between cells).
    lnL_holodeck the population marginalised over the NANOGrav astrophysical prior (marginal_foreground.prior_draws_holodeck,
                 32768 draws, halves A and B, effective sample size ess_holodeck), as B2 for NANOGrav.
    lnL_pl       the (A, gamma) power law marginalised over the PTArcade IPTA prior (16384 draws, halves), as B1 for NANOGrav
    lnZ_bin, lnL_bin_prof   the binaries alone: evidence over the prior, and the profile under the prior (Nelder-Mead),
                 for the preference of the transition over the binaries alone (binaryfog's gain, Chernoff)
The transition spectrum is that of the scan (plane_scan.central, COSMO_VARIANTS[0]).

    python ipta_fog.py [NDRAW]          (WORKERS processes, default 56)  -> paper/data/ipta_fog.npz"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NDRAW = int(sys.argv[1]) if len(sys.argv) > 1 else 32768
sys.argv = sys.argv[:1]
sys.path.insert(0, os.path.dirname(HERE))
import paths  # noqa: E402  (src/, models/ and the folders of paper/reproduce on sys.path)
import plane_scan as ps  # noqa: E402
import marginal_foreground as mf  # noqa: E402
import binaryfog as bf  # noqa: E402
from binaryfog.likelihood import _Z, _W  # noqa: E402

WORKERS = int(os.environ.get("WORKERS", "56"))
DATA = bf.FreeSpectrum.load("ipta2_cp_v1", nbins=ps.DATASETS["IPTA2"])
FREQS, NB = DATA.freqs, DATA.nbins
BG = bf.Background(FREQS, maps="ipta2")
LIKE = bf.Likelihood(DATA)


def spectrum_row(i):
    g = ps.G[i]
    out = np.full((len(ps.LOGV), NB), np.nan)
    for j, lv in enumerate(ps.LOGV):
        treh, bh, al, bv = ps.central(g, lv)
        if not (np.isfinite(bh) and np.isfinite(al) and bv > 0):
            continue
        sp = ps.make_spectrum(ps.COSMO_VARIANTS[0][1], ps.reheating(g, lv), bh, al, bv)
        if sp is None:
            continue
        out[j] = sp(FREQS)
    return i, out


def lnl_holodeck(pts, theta):
    """As marginal_foreground.lnl_holodeck, on the IPTA data with the IPTA maps."""
    med, sigma, skew = BG.predict(theta)
    c = max(float(skew), 1e-3)
    z = np.maximum(_Z, -0.5 / c)
    tot = np.zeros(len(pts))
    for i in range(NB):
        s = max(float(sigma[i]), 1e-4)
        b = 10.0 ** (np.log10(max(med[i], 1e-300)) + s * (z + c * z ** 2))
        x = DATA.to_log10rho(pts[:, i, None] + b[None, :], i)
        val = np.exp(DATA.ln_density(x, i)) @ _W
        with np.errstate(divide="ignore"):
            tot += np.log(val)
    return tot


def lnl_powerlaw(pts, x):
    pl = ps._powerlaw(FREQS, *x)
    tot = np.zeros(len(pts))
    for i in range(NB):
        with np.errstate(divide="ignore"):
            tot += DATA.ln_density(DATA.to_log10rho(pts[:, i] + pl[i], i), i)
    return tot


def accumulate(job):
    thetas, pts = job[:2]
    f = lnl_powerlaw if len(job) > 2 and job[2] == "pl" else lnl_holodeck
    m = np.full(len(pts), -np.inf); s1 = np.zeros(len(pts)); s2 = np.zeros(len(pts))
    for th in thetas:
        l = f(pts, th)
        new = np.maximum(m, l)
        fin = np.isfinite(new)
        a = np.where(fin, np.exp(np.where(fin, m - new, 0.0)), 0.0)
        b = np.where(fin, np.exp(np.where(fin, l - new, 0.0)), 0.0)
        s1 = s1 * a + b; s2 = s2 * a * a + b * b; m = new
    return m, s1, s2


if __name__ == "__main__":
    from multiprocessing import get_context
    t0 = time.time()
    spec = np.full((len(ps.G), len(ps.LOGV), NB), np.nan)
    with get_context("fork").Pool(WORKERS) as pool:
        for i, row in pool.imap_unordered(spectrum_row, range(len(ps.G))):
            spec[i] = row
    ok = np.all(np.isfinite(spec), axis=-1)
    pts = spec[ok]
    print(f"spectra: {ok.sum()} of {ok.size} points, {time.time() - t0:.0f} s", flush=True)
    out = {"G": ps.G, "LOGV": ps.LOGV, "freqs": FREQS}
    l0 = np.full(ok.shape, np.nan); l0[ok] = [LIKE.ln_like(p) for p in pts]; out["lnL"] = l0
    rng = np.random.default_rng(20261108)
    th = mf.prior_draws_holodeck(rng, NDRAW)
    halves = {}
    for h, sel in (("A", slice(0, NDRAW // 2)), ("B", slice(NDRAW // 2, NDRAW))):
        with get_context("fork").Pool(WORKERS) as pool:
            parts = pool.map(accumulate, [(b, pts) for b in np.array_split(th[sel], 4 * WORKERS)])
        halves[h] = mf.combine(parts)
    m, s1, s2 = mf.combine([halves["A"], halves["B"]])
    for key, (mm, ss, nn) in (("lnL_holodeck", (m, s1, NDRAW)), ("lnL_holodeck_A", (halves["A"][0], halves["A"][1], NDRAW // 2)),
                              ("lnL_holodeck_B", (halves["B"][0], halves["B"][1], NDRAW // 2))):
        a = np.full(ok.shape, np.nan); a[ok] = mm + np.log(ss) - np.log(nn); out[key] = a
    ess = np.full(ok.shape, np.nan); ess[ok] = s1 ** 2 / s2; out["ess_holodeck"] = ess
    lm = out["lnL_holodeck"]; near = lm > np.nanmax(lm) - 6
    print(f"holodeck: max ln L_marg {np.nanmax(lm):.3f} (no foreground {np.nanmax(l0):.3f}); halves differ by at most "
          f"{np.nanmax(np.abs(out['lnL_holodeck_A'] - out['lnL_holodeck_B'])[near]):.3f} within 6 of the maximum, "
          f"ESS there >= {np.nanmin(ess[near]):.0f}; {time.time() - t0:.0f} s", flush=True)
    # the power law, marginalised over the IPTA prior of the scan's profile
    mu, cov = ps.signal_builder.bhb_priors["IPTA2"]
    thp = rng.multivariate_normal(np.array(mu, float), np.array(cov, float), size=16384)
    hp = {}
    for h, sel in (("A", slice(0, 8192)), ("B", slice(8192, 16384))):
        with get_context("fork").Pool(WORKERS) as pool:
            parts = pool.map(accumulate, [(b, pts, "pl") for b in np.array_split(thp[sel], 4 * WORKERS)])
        hp[h] = mf.combine(parts)
    m, s1, s2 = mf.combine([hp["A"], hp["B"]])
    for key, (mm, ss, nn) in (("lnL_pl", (m, s1, 16384)), ("lnL_pl_A", (hp["A"][0], hp["A"][1], 8192)),
                              ("lnL_pl_B", (hp["B"][0], hp["B"][1], 8192))):
        a = np.full(ok.shape, np.nan); a[ok] = mm + np.log(ss) - np.log(nn); out[key] = a
    # the binaries alone
    from scipy.optimize import minimize
    from binaryfog import priors as bfp
    zero = np.zeros((1, NB))
    l0 = np.array([lnl_holodeck(zero, x)[0] for x in th])
    mx = l0.max(); out["lnZ_bin"] = float(mx + np.log(np.mean(np.exp(l0 - mx))))
    neg = lambda x: 1e30 if not np.isfinite(bfp.ln_prior(x, "nanograv")) else -(lnl_holodeck(zero, x)[0] + bfp.ln_prior(x, "nanograv"))
    best = min((minimize(neg, x0, method="Nelder-Mead", options={"maxfev": 2000, "xatol": 1e-4, "fatol": 1e-4})
                for x0 in th[np.argsort(-l0)[:8]]), key=lambda r: r.fun)
    out["lnL_bin_prof"], out["theta_bin_prof"] = -best.fun, best.x
    print(f"power law: max ln L_marg {np.nanmax(out['lnL_pl']):.3f}; binaries alone: lnZ {out['lnZ_bin']:.3f}, profile "
          f"{out['lnL_bin_prof']:.3f}; {time.time() - t0:.0f} s", flush=True)
    np.savez(paths.data("ipta_fog.npz"), **out)
    print("wrote ipta_fog.npz", flush=True)
