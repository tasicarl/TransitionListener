"""The NANOGrav likelihood with the binaryfog population marginalised over another prior set of binaryfog.priors, on
the grid of the tail patch, as lnL_holodeck of tail_fog.py (which uses the NANOGrav priors). Same draws as
prior_study.py (draws(), seed 20261106, 32768 draws), so that at the nodes shared with the
plane scan the result reproduces data/ng15_fog_priors.npz.
    PRIOR=recent python tail_fog_prior.py        (WORKERS processes, default 56)
                                                 -> paper/data/tail_check/tail_fog_<PRIOR>.npz, or $TAIL_OUT"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv = sys.argv[:1]
sys.path.insert(0, HERE)
import tail_fog as tf  # noqa: E402

PRIOR = os.environ.get("PRIOR", "recent")

if __name__ == "__main__":
    from multiprocessing import get_context
    t0 = time.time()
    spec = np.full((len(tf.G), len(tf.L), tf.mf.NB), np.nan)
    with get_context("fork").Pool(tf.WORKERS) as pool:
        for i, row in pool.imap_unordered(tf.spectrum_row, range(len(tf.G))):
            spec[i] = row
    ok = np.all(np.isfinite(spec), axis=-1)
    pts = spec[ok]
    print(f"spectra: {ok.sum()} of {ok.size} points, {time.time() - t0:.0f} s", flush=True)
    th = tf.pst.draws(np.random.default_rng(20261106), 32768, PRIOR)
    r, ess = tf.marginal("holodeck", th, pts, ok)
    out = {"G": tf.G, "LOGV": tf.L, "prior": PRIOR, "ess_holodeck": ess}
    for k, v in r.items():
        out["lnL_holodeck" + k] = v
    P = np.load(tf.paths.data("ng15_fog_priors.npz"))
    ig = [int(np.argmin(np.abs(P["G"] - g))) for g in tf.G]
    il = [int(np.argmin(np.abs(P["LOGV"] - l))) for l in tf.L]
    sg = np.array([abs(P["G"][j] - g) < 1e-9 for j, g in zip(ig, tf.G)])
    sl = np.array([abs(P["LOGV"][j] - l) < 1e-9 for j, l in zip(il, tf.L)])
    a = out["lnL_holodeck"][np.ix_(sg, sl)]
    b = P["lnL_marg_" + PRIOR][np.ix_(np.array(ig)[sg], np.array(il)[sl])]
    both = np.isfinite(a) & np.isfinite(b)
    print(f"check against ng15_fog_priors.npz at {both.sum()} shared nodes: max |diff| {np.max(np.abs(a - b)[both]):.3g}",
          flush=True)
    # the no-signal value (transition spectrum zero), as tail_fog_plateau.py adds to tail_fog.npz
    zero = np.zeros((1, tf.mf.NB))
    l0 = np.array([tf.mf.lnl_holodeck(zero, x)[0] for x in th])
    out["plateau_lnL_holodeck"] = float(l0.max() + np.log(np.mean(np.exp(l0 - l0.max()))))
    path = os.environ.get("TAIL_OUT", os.path.join(tf.TAIL, f"tail_fog_{PRIOR}.npz"))
    np.savez(path, **out)
    print(f"wrote {path}, {time.time() - t0:.0f} s", flush=True)
