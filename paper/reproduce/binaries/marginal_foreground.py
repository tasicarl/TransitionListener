"""The NANOGrav likelihood with the binary foreground marginalised over its parameters, on the grid of
plane_scan.py, for the comparison of profile likelihood and marginal posterior (appendix E):

    L_marg(g, v) = integral d theta  pi(theta) L(g, v, theta),

by Monte Carlo over draws from the prior pi. Two foregrounds, both on the corrected 2026 NANOGrav
free spectrum through binaryfog, as in plane_scan.lnL:
  pl       the (A, gamma) power law under the PTArcade population prior (ptarcade
           signal_builder.bhb_priors, as the scan and the chains), whose profile is NG15_bhb;
  holodeck the six-parameter binaryfog population under the NANOGrav astrophysical priors
           (binaryfog.priors 'nanograv', truncated to the library box), whose profile is the
           NG15_binaryfog row (B2).
The transition spectrum is the default of the scan (plane_scan.central, COSMO_VARIANTS[0]); points
without a transition, or with (beta/H)_Gamma <= 0, have no likelihood, as in the scan.
The draws are split into two halves, A and B, whose maps test the convergence; ESS is the effective
number of draws, (sum w)^2 / sum w^2, at every point.

    python marginal_foreground.py [NDRAW_HOLODECK] [NDRAW_PL]      (WORKERS processes, default 56)
Writes paper/data/marginal_foreground.npz."""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import paths  # noqa: E402  (src/, models/ and the folders of paper/reproduce on sys.path)
import plane_scan as ps  # noqa: E402
from binaryfog import priors as bfp  # noqa: E402
from binaryfog.background import PARAM_NAMES, PRIOR_BOX  # noqa: E402
from binaryfog.likelihood import _Z, _W  # noqa: E402

N_HOLO = int(sys.argv[1]) if len(sys.argv) > 1 else 32768
N_PL = int(sys.argv[2]) if len(sys.argv) > 2 else 16384
WORKERS = int(os.environ.get("WORKERS", "56"))
LIKE = ps.BF_LIKE["cp_v2"]
DATA = LIKE.data
FREQS = DATA.freqs
NB = DATA.nbins
BG = ps.BF_FOG.background          # the population emulator on the frequencies of the data


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


def prior_draws_holodeck(rng, n):
    """Draws from the NANOGrav astrophysical prior, truncated to the library box: Gaussians on the
    four parameters of binaryfog.priors.NANOGRAV_GAUSSIAN, flat on the other two."""
    lo = np.array([PRIOR_BOX[k][0] for k in PARAM_NAMES])
    hi = np.array([PRIOR_BOX[k][1] for k in PARAM_NAMES])
    out = []
    while sum(len(x) for x in out) < n:
        th = rng.uniform(lo, hi, size=(4 * n, len(PARAM_NAMES)))
        for k, name in enumerate(PARAM_NAMES):
            if name in bfp.NANOGRAV_GAUSSIAN:
                mu, sd = bfp.NANOGRAV_GAUSSIAN[name]
                th[:, k] = rng.normal(mu, sd, size=len(th))
        ok = np.all((th >= lo) & (th <= hi), axis=1)
        out.append(th[ok])
    return np.concatenate(out)[:n]


def lnl_holodeck(pts, theta):
    """ln L of every point (rows of pts, the transition spectrum per bin) with the population at
    theta, marginalised over realisations as binaryfog.Likelihood.ln_like does."""
    med, sigma, skew = BG.predict(theta)
    c = max(float(skew), 1e-3)
    z = np.maximum(_Z, -0.5 / c)
    tot = np.zeros(len(pts))
    for i in range(NB):
        s = max(float(sigma[i]), 1e-4)
        b = 10.0 ** (np.log10(max(med[i], 1e-300)) + s * (z + c * z ** 2))           # (NQ,)
        x = DATA.to_log10rho(pts[:, i, None] + b[None, :], i)                        # (P, NQ)
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
    """log-sum-exp of ln L over a block of draws, and the sums for the effective sample size."""
    kind, thetas, pts = job
    f = lnl_holodeck if kind == "holodeck" else lnl_powerlaw
    m = np.full(len(pts), -np.inf)
    s1 = np.zeros(len(pts)); s2 = np.zeros(len(pts))
    for th in thetas:
        l = f(pts, th)
        new = np.maximum(m, l)
        fin = np.isfinite(new)
        a = np.where(fin, np.exp(np.where(fin, m - new, 0.0)), 0.0)
        b = np.where(fin, np.exp(np.where(fin, l - new, 0.0)), 0.0)
        s1 = s1 * a + b
        s2 = s2 * a * a + b * b
        m = new
    return m, s1, s2


def combine(parts):
    m = np.max([p[0] for p in parts], axis=0)
    fin = np.isfinite(m)
    s1 = sum(np.where(fin, p[1] * np.exp(np.where(fin, p[0] - m, 0.0)), 0.0) for p in parts)
    s2 = sum(np.where(fin, p[2] * np.exp(2 * np.where(fin, p[0] - m, 0.0)), 0.0) for p in parts)
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
    print(f"spectra: {ok.sum()} points with a transition of {ok.size}, {time.time() - t0:.0f} s", flush=True)
    # a check against the scan: the likelihood without foreground at these points
    l0 = np.array([LIKE.ln_like(p) for p in pts[:200]])
    d = dict(np.load(paths.data("plane_scan_u1.npz"), allow_pickle=True))
    k = list(d["names"]).index("nlo")
    ref = d["NG15"][k][ok][:200]
    print(f"check against the scan (no foreground, 200 points): max |difference| {np.nanmax(np.abs(l0 - ref)):.2e}", flush=True)

    rng = np.random.default_rng(20261011)
    out = {"G": ps.G, "LOGV": ps.LOGV}
    for kind, n in (("holodeck", N_HOLO), ("pl", N_PL)):
        if kind == "holodeck":
            th = prior_draws_holodeck(rng, n)
        else:                                   # the prior of the scan's profile and of the chains
            from ptarcade import signal_builder
            mu, cov = signal_builder.bhb_priors["NG15"]
            th = rng.multivariate_normal(np.array(mu, float), np.array(cov, float), size=n)
        halves = {}
        for h, sel in (("A", slice(0, n // 2)), ("B", slice(n // 2, n))):
            blocks = np.array_split(th[sel], 4 * WORKERS)
            with get_context("fork").Pool(WORKERS) as pool:
                parts = pool.map(accumulate, [(kind, b, pts) for b in blocks])
            halves[h] = combine(parts)
        m, s1, s2 = combine([halves["A"], halves["B"]])
        lnm = np.full(ok.shape, np.nan); ess = np.full(ok.shape, np.nan)
        lnm[ok] = m + np.log(s1) - np.log(n)
        ess[ok] = s1 ** 2 / s2
        out[f"lnL_{kind}"] = lnm
        out[f"ess_{kind}"] = ess
        for h in ("A", "B"):
            mh, s1h, _ = halves[h]
            a = np.full(ok.shape, np.nan)
            a[ok] = mh + np.log(s1h) - np.log(n // 2)
            out[f"lnL_{kind}_{h}"] = a
        print(f"{kind}: {n} draws, max ln L_marg {np.nanmax(lnm):.3f}, halves differ by at most "
              f"{np.nanmax(np.abs(out[f'lnL_{kind}_A'] - out[f'lnL_{kind}_B'])[lnm > np.nanmax(lnm) - 6]):.3f} "
              f"within 6 of the maximum; ESS there {np.nanmin(ess[lnm > np.nanmax(lnm) - 6]):.0f} to "
              f"{np.nanmax(ess[lnm > np.nanmax(lnm) - 6]):.0f}; {time.time() - t0:.0f} s", flush=True)
    np.savez(paths.data("marginal_foreground.npz"), **out)
    print("wrote marginal_foreground.npz", flush=True)
