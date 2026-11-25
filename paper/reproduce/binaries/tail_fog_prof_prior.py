"""The NANOGrav likelihood with the binaryfog population profiled under another prior set of binaryfog.priors
(ln L + ln prior maximised at every point), on the grid of the tail patch, as lnL_fog_prof of tail_fog.py (which uses
the NANOGrav priors). Same seeds as prior_study.py (the first 64 of draws(), seed 20261106),
so that at the nodes shared with the plane scan the result reproduces its lnL_prof.
    PRIOR=recent python tail_fog_prof_prior.py        (WORKERS processes, default 56)
                                                      -> paper/data/tail_check/tail_fog_prof_<PRIOR>.npz, or $TAIL_OUT
The reference for the check is paper/data/ng15_fog_prof_<PRIOR>.npz, or $REF."""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv = sys.argv[:1]
sys.path.insert(0, HERE)
import tail_fog as tf  # noqa: E402

PRIOR = os.environ.get("PRIOR", "recent")
REF = os.environ.get("REF", tf.paths.data(f"ng15_fog_prof_{PRIOR}.npz"))

if __name__ == "__main__":
    from multiprocessing import get_context
    t0 = time.time()
    spec = np.full((len(tf.G), len(tf.L), tf.mf.NB), np.nan)
    with get_context("fork").Pool(tf.WORKERS) as pool:
        for i, row in pool.imap_unordered(tf.spectrum_row, range(len(tf.G))):
            spec[i] = row
    ok = np.all(np.isfinite(spec), axis=-1)
    print(f"spectra: {ok.sum()} of {ok.size} points, {time.time() - t0:.0f} s", flush=True)
    seeds64 = tf.pst.draws(np.random.default_rng(20261106), 32768, PRIOR)[:64]
    prof = np.full(ok.shape, np.nan)
    with get_context("fork").Pool(tf.WORKERS) as pool:
        for i, row in pool.imap_unordered(tf.pst.profile_row, [(i, spec[i], PRIOR, seeds64) for i in range(len(tf.G))]):
            prof[i] = row
    out = {"G": tf.G, "LOGV": tf.L, "prior": PRIOR, "lnL_fog_prof": prof}
    P = np.load(REF)
    ig = [int(np.argmin(np.abs(P["G"] - g))) for g in tf.G]
    il = [int(np.argmin(np.abs(P["LOGV"] - l))) for l in tf.L]
    sg = np.array([abs(P["G"][j] - g) < 1e-9 for j, g in zip(ig, tf.G)])
    sl = np.array([abs(P["LOGV"][j] - l) < 1e-9 for j, l in zip(il, tf.L)])
    a = prof[np.ix_(sg, sl)]
    b = P["lnL_prof"][np.ix_(np.array(ig)[sg], np.array(il)[sl])]
    both = np.isfinite(a) & np.isfinite(b)
    print(f"check against {os.path.basename(REF)} at {both.sum()} shared nodes: max |diff| "
          f"{np.max(np.abs(a - b)[both]):.3g}", flush=True)
    path = os.environ.get("TAIL_OUT", os.path.join(tf.TAIL, f"tail_fog_prof_{PRIOR}.npz"))
    np.savez(path, **out)
    print(f"wrote {path}, {time.time() - t0:.0f} s", flush=True)
