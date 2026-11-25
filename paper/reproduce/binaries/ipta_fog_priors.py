"""IPTA DR2 with the binaryfog population marginalised over the other prior sets of binaryfog.priors (B2 for IPTA with
the priors of tab:preference), as lnL_holodeck of ipta_fog.py, which uses the NANOGrav priors. Same spectra, same
emulator on the IPTA frequencies, same draws as prior_study.py (draws(), seed 20261106).
    PRIORS=literature,recent python ipta_fog_priors.py [NDRAW]      -> paper/data/ipta_fog_<prior>.npz"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NDRAW = int(sys.argv[1]) if len(sys.argv) > 1 else 32768
sys.argv = sys.argv[:1]
sys.path.insert(0, os.path.dirname(HERE))
import paths  # noqa: E402  (src/, models/ and the folders of paper/reproduce on sys.path)
import ipta_fog as ipf  # noqa: E402
import marginal_foreground as mf  # noqa: E402
import prior_study as pst  # noqa: E402
import plane_scan as ps  # noqa: E402

PRIORS = os.environ.get("PRIORS", "literature,recent").split(",")

if __name__ == "__main__":
    from multiprocessing import get_context
    t0 = time.time()
    spec = np.full((len(ps.G), len(ps.LOGV), ipf.NB), np.nan)
    with get_context("fork").Pool(ipf.WORKERS) as pool:
        for i, row in pool.imap_unordered(ipf.spectrum_row, range(len(ps.G))):
            spec[i] = row
    ok = np.all(np.isfinite(spec), axis=-1)
    pts = spec[ok]
    print(f"spectra: {ok.sum()} points, {time.time() - t0:.0f} s", flush=True)
    for which in PRIORS:
        th = pst.draws(np.random.default_rng(20261106), NDRAW, which)
        out = {"G": ps.G, "LOGV": ps.LOGV, "prior": which}
        halves = {}
        for h, sel in (("A", slice(0, NDRAW // 2)), ("B", slice(NDRAW // 2, NDRAW))):
            with get_context("fork").Pool(ipf.WORKERS) as pool:
                parts = pool.map(ipf.accumulate, [(b, pts) for b in np.array_split(th[sel], 4 * ipf.WORKERS)])
            halves[h] = mf.combine(parts)
        m, s1, s2 = mf.combine([halves["A"], halves["B"]])
        for key, (mm, ss, nn) in (("lnL_holodeck", (m, s1, NDRAW)), ("lnL_holodeck_A", (halves["A"][0], halves["A"][1], NDRAW // 2)),
                                  ("lnL_holodeck_B", (halves["B"][0], halves["B"][1], NDRAW // 2))):
            a = np.full(ok.shape, np.nan); a[ok] = mm + np.log(ss) - np.log(nn); out[key] = a
        ess = np.full(ok.shape, np.nan); ess[ok] = s1 ** 2 / s2; out["ess_holodeck"] = ess
        lm = out["lnL_holodeck"]; near = lm > np.nanmax(lm) - 6
        print(f"{which}: max ln L_marg {np.nanmax(lm):.3f}; halves differ by at most "
              f"{np.nanmax(np.abs(out['lnL_holodeck_A'] - out['lnL_holodeck_B'])[near]):.3f} within 6 of the maximum, "
              f"ESS there >= {np.nanmin(ess[near]):.0f}; {time.time() - t0:.0f} s", flush=True)
        np.savez(paths.data(f"ipta_fog_{which}.npz"), **out)
    print("done", flush=True)
