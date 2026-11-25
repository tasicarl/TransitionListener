"""The maximum of ln L for every variant of plane_scan.py and both data sets, found by maximising
over (g, log10 v) instead of taking the largest value on the grid.

The grid maximum of plane_scan_u1.npz is the start; Nelder-Mead then maximises the same likelihood
(the same transition table, reheating, spectrum and free-spectrum likelihood as the scan) in the
continuous plane, and the best of three starts (the grid maximum and its two runners-up that are at
least two cells away) is kept. Points without a transition have ln L = -inf. The IPTA likelihood of
PTArcade is piecewise constant in the spectrum (histogram of a kernel density estimate), so its
maximum is resolved only to about 0.05 in ln L.

Run with PYTHONPATH=<repo>/src in the environment of plane_scan.py:
  python best_fits.py        writes paper/data/best_fits.json
  BEST_FIT_ONLY="4D daisy RG" python best_fits.py   refits only the variants of these names (comma-separated) and
                                                    replaces them in the existing best_fits.json
"""
import json
import os
import sys
import time
from multiprocessing import get_context

import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import paths  # noqa: E402  (src/, models/ and the folders of paper/reproduce on sys.path)
import plane_scan as ps  # noqa: E402

D = dict(np.load(paths.data("plane_scan_u1.npz"), allow_pickle=True))
NAMES = [str(n) for n in D["names"]]


def lnl_at(k, ds, x, bhb=False):
    g, lv = float(x[0]), float(x[1])
    if not (ps.G[0] <= g <= ps.G[-1] and ps.LOGV[0] <= lv <= ps.LOGV[-1]):
        return -np.inf
    grp, name, pk, c = ps.VARIANT_LIST[k]
    try:
        _, bh, alpha, bhv = ps.tp_variants(g, lv)[pk]
        if not (np.isfinite(bh) and np.isfinite(alpha)):
            return -np.inf
        reh = ps.cosmo_inputs(c, g, lv, alpha)
        if not np.isfinite(reh[0]):
            return -np.inf
        return float(ps.lnL(ds, ps.make_spectrum(c[1], reh, bh, alpha, bhv), smbhb=bhb, optimise=bhb))
    except Exception:
        return -np.inf


def starts(lnl):
    """The grid maximum and up to two runners-up at least two cells away from the chosen ones."""
    flat = np.argsort(np.nan_to_num(lnl, nan=-np.inf).ravel())[::-1]
    out = []
    for f in flat:
        i, j = np.unravel_index(f, lnl.shape)
        if not np.isfinite(lnl[i, j]):
            break
        if all(abs(i - a) > 2 or abs(j - b) > 2 for a, b in out):
            out.append((i, j))
        if len(out) == 3:
            break
    return out


def job(args):
    k, ds, bhb = args
    lnl = D[ds + ("_bhb" if bhb else "")][k]
    if not np.isfinite(lnl).any():
        return k, ds, None
    grid_best = float(np.nanmax(lnl))
    best = (grid_best, None)
    for i, j in starts(lnl):
        x0 = np.array([D["G"][i], D["LOGV"][j]])
        r = minimize(lambda x: -lnl_at(k, ds, x, bhb), x0, method="Nelder-Mead",
                     options=dict(initial_simplex=[x0, x0 + [0.004, 0.0], x0 + [0.0, 0.04]],
                                  xatol=2e-5, fatol=2e-3, maxfev=300))
        if np.isfinite(r.fun) and -r.fun > best[0]:
            best = (-float(r.fun), [float(r.x[0]), float(r.x[1])])
    return k, ds + ("_bhb" if bhb else ""), dict(grid=grid_best, max=best[0], at=best[1])


if __name__ == "__main__":
    t0 = time.time()
    # the scan file must have been written with the variant list of the plane_scan.py imported here
    n_scan = D[next(iter(ps.DATASETS))].shape[0]
    assert n_scan == len(ps.VARIANT_LIST), \
        f"plane_scan_u1.npz has {n_scan} variants, plane_scan.py {len(ps.VARIANT_LIST)}: rerun the scan"
    only = [n for n in os.environ.get("BEST_FIT_ONLY", "").split(",") if n]
    names = [v[1] for v in ps.VARIANT_LIST]
    assert all(n in names for n in only), f"unknown variants {only}"
    tasks = [(k, ds, False) for k in range(len(ps.VARIANT_LIST)) for ds in ps.DATASETS if not only or names[k] in only]
    # the power-law binary foreground (B1) of the central prescription, profiled as in the scan
    k_nlo = names.index("nlo")
    if not only:
        tasks += [(k_nlo, ds, True) for ds in ps.DATASETS]
    out = json.load(open(paths.data("best_fits.json"))) if only else {}
    for n in only:
        out[n] = {}
    with get_context("fork").Pool(int(os.environ.get("BEST_FIT_WORKERS", "24"))) as pool:
        for k, ds, r in pool.imap_unordered(job, tasks):
            out.setdefault(ps.VARIANT_LIST[k][1], {})[ds] = r
    json.dump(out, open(paths.data("best_fits.json"), "w"), indent=1)
    for name, v in out.items():
        print(f"{name:40s}", "  ".join(f"{ds} grid {r['grid']:+.3f} max {r['max']:+.3f}" if r else f"{ds} --"
                                       for ds, r in v.items()))
    print(f"done in {time.time() - t0:.0f} s")
