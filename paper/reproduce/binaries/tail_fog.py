"""The NANOGrav likelihood with the binary foreground on the grid of the tail patch (tail_patch.npz: g = 0.400 to
0.440 in steps of 0.0001, log10 v/GeV = 0.20 to 1.30 in steps of 0.005), where the strip at g ~ 0.41 is about
0.001 wide in g and the plane scan samples it at single cells. Same spectra as tail_fine.py (tp_variants,
cosmo_inputs, make_spectrum), same foreground machinery as marginal_foreground.py and
prior_study.py:
    lnL          NG15 without foreground (a check against tail_patch.npz lnL)
    lnL_holodeck the binaryfog population marginalised over its NANOGrav prior (32768 draws, halves A and B)
    lnL_pl       the (A, gamma) power law marginalised over its PTArcade prior (16384 draws, halves)
    lnL_fog_prof the population profiled, ln L + ln prior maximised at every point (Nelder-Mead, warm start)
    python tail_fog.py [OUT]      (WORKERS processes, default 56; OUT defaults to paper/data/tail_check/tail_fog.npz)
The grid is read from paper/data/tail_check/tail_patch.npz, or from $TAIL_GRID."""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import paths  # noqa: E402  (src/, models/ and the folders of paper/reproduce on sys.path)
TAIL = paths.data("tail_check")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(TAIL, "tail_fog.npz")   # read before prior_study resets argv
sys.argv = sys.argv[:1]
import plane_scan as PS  # noqa: E402
import marginal_foreground as mf  # noqa: E402
import prior_study as pst  # noqa: E402

WORKERS = int(os.environ.get("WORKERS", "56"))
T = np.load(os.environ.get("TAIL_GRID", os.path.join(TAIL, "tail_patch.npz")))   # the grid: G, LOGV, lnL
G, L = T["G"], T["LOGV"]
C = PS.COSMO_VARIANTS[0]


def spectrum_row(i):
    g = G[i]
    out = np.full((len(L), mf.NB), np.nan)
    for j, lv in enumerate(L):
        _, bh, alpha, bhv = PS.tp_variants(g, lv)["nlo"]
        if not (np.isfinite(bh) and np.isfinite(alpha)):
            continue
        reh = PS.cosmo_inputs(C, g, lv, alpha)
        if not np.isfinite(reh[0]):
            continue
        sp = PS.make_spectrum(C[1], reh, bh, alpha, bhv)
        if sp is None:
            continue
        out[j] = sp(mf.FREQS)
    return i, out


def marginal(kind, th, pts, ok):
    from multiprocessing import get_context
    n = len(th)
    halves = {}
    for h, sel in (("A", slice(0, n // 2)), ("B", slice(n // 2, n))):
        with get_context("fork").Pool(WORKERS) as pool:
            parts = pool.map(mf.accumulate, [(kind, b, pts) for b in np.array_split(th[sel], 4 * WORKERS)])
        halves[h] = mf.combine(parts)
    m, s1, s2 = mf.combine([halves["A"], halves["B"]])
    res = {}
    for key, (mm, ss, nn) in (("", (m, s1, n)), ("_A", (halves["A"][0], halves["A"][1], n // 2)),
                              ("_B", (halves["B"][0], halves["B"][1], n // 2))):
        a = np.full(ok.shape, np.nan); a[ok] = mm + np.log(ss) - np.log(nn); res[key] = a
    ess = np.full(ok.shape, np.nan); ess[ok] = s1 ** 2 / s2
    return res, ess


if __name__ == "__main__":
    from multiprocessing import get_context
    t0 = time.time()
    spec = np.full((len(G), len(L), mf.NB), np.nan)
    with get_context("fork").Pool(WORKERS) as pool:
        for i, row in pool.imap_unordered(spectrum_row, range(len(G))):
            spec[i] = row
    ok = np.all(np.isfinite(spec), axis=-1)
    pts = spec[ok]
    print(f"spectra: {ok.sum()} of {ok.size} points, {time.time() - t0:.0f} s", flush=True)
    out = {"G": G, "LOGV": L}
    l0 = np.full(ok.shape, np.nan); l0[ok] = [mf.LIKE.ln_like(p) for p in pts]; out["lnL"] = l0
    both = np.isfinite(l0) & np.isfinite(T["lnL"])
    print(f"check against tail_patch.npz lnL: max |diff| {np.max(np.abs(l0 - T['lnL'])[both]):.2e} over {both.sum()} "
          f"points", flush=True)
    rng = np.random.default_rng(20261108)
    th_h = mf.prior_draws_holodeck(rng, 32768)
    r, ess = marginal("holodeck", th_h, pts, ok)
    for k, v in r.items():
        out["lnL_holodeck" + k] = v
    out["ess_holodeck"] = ess
    print(f"holodeck marginal done, {time.time() - t0:.0f} s", flush=True)
    mu, cov = PS.signal_builder.bhb_priors["NG15"]          # the prior of the scan's profile, as marginal_foreground
    th_p = rng.multivariate_normal(np.array(mu, float), np.array(cov, float), size=16384)
    r, ess = marginal("pl", th_p, pts, ok)
    for k, v in r.items():
        out["lnL_pl" + k] = v
    out["ess_pl"] = ess
    print(f"power-law marginal done, {time.time() - t0:.0f} s", flush=True)
    seeds64 = th_h[:64]
    prof = np.full(ok.shape, np.nan)
    with get_context("fork").Pool(WORKERS) as pool:
        for i, row in pool.imap_unordered(pst.profile_row, [(i, spec[i], "nanograv", seeds64) for i in range(len(G))]):
            prof[i] = row
    out["lnL_fog_prof"] = prof
    np.savez(OUT, **out)
    print(f"wrote {OUT}, {time.time() - t0:.0f} s", flush=True)
