"""B1, the (A, gamma) power-law binaries: ln L without the prior term ln pi at the maximum of ln L + ln pi
(best_fits.json, "NG15_bhb"/"IPTA2_bhb"), which the switch tables print against the default maximum, and as a check
the maximum of ln L with (A, gamma) free and no prior (the binaries then switch off: Delta ln L = 0).
Run with PYTHONPATH=<repo>/src in the environment of plane_scan.py; writes paper/data/bhb_lnl_without_prior.json."""
import json, os, sys
import numpy as np
from scipy.optimize import minimize
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import paths  # noqa: E402  (src/, models/ and the folders of paper/reproduce on sys.path)
import best_fits as B
ps = B.ps
bf = json.load(open(paths.data("best_fits.json")))["nlo"]
k = B.NAMES.index("nlo")
out = {}
for ds in ("NG15", "IPTA2"):
    grp, name, pk, c = ps.VARIANT_LIST[k]
    def spec_at(x):
        _, bh, alpha, bhv = ps.tp_variants(float(x[0]), float(x[1]))[pk]
        reh = ps.cosmo_inputs(c, float(x[0]), float(x[1]), alpha)
        return ps.make_spectrum(c[1], reh, bh, alpha, bhv)
    like = ps.BF_LIKE["cp_v2" if ds == "NG15" else "ipta2"]; f = like.data.freqs
    def inner(x, prior=True):
        s = spec_at(x)(f)
        nodes, lnw = ps.BHB_NODES[ds]
        vals = np.array([like.ln_like(s + ps._powerlaw(f, *y)) for y in nodes]) + (lnw if prior else 0)
        obj = lambda y: -(float(like.ln_like(s + ps._powerlaw(f, *y))) + (ps._bhb_lnprior(ds, y) if prior else 0.0))
        best = None
        for y0 in [nodes[int(np.argmax(vals))]] + ([np.array([-17.0, 13 / 3])] if not prior else []):
            r = minimize(obj, y0, method="Nelder-Mead", options={"xatol": 1e-5, "fatol": 1e-5, "maxfev": 2000})
            if best is None or r.fun < best.fun: best = r
        y = best.x
        return float(like.ln_like(s + ps._powerlaw(f, *y))), ps._bhb_lnprior(ds, y), y, float(like.ln_like(s))
    x = bf[ds + "_bhb"]["at"]
    lnl, lnpi, y, lnl_nobin = inner(x)
    # (b): maximise ln L over (g, log10 v, log10 A, gamma) without prior; start at the default best fit and at x
    def neg(z):
        try: return -inner(z[:2], prior=False)[0]
        except Exception: return np.inf
    rb = [minimize(neg, np.array(z0), method="Nelder-Mead", options=dict(xatol=2e-5, fatol=2e-3, maxfev=300,
          initial_simplex=[z0, np.add(z0, [0.004, 0]), np.add(z0, [0, 0.04])])) for z0 in (bf[ds]["at"], x)]
    lb = max(-r.fun for r in rb)
    out[ds] = dict(default=bf[ds]["max"], profiled_post=bf[ds + "_bhb"]["max"], at=x, lnL_at_profile=lnl, lnpi=lnpi,
                   A_gamma=list(map(float, y)), lnL_free_max=lb,
                   d_a=lnl - bf[ds]["max"], d_b=lb - bf[ds]["max"], d_old=bf[ds + "_bhb"]["max"] - bf[ds]["max"])
    print(ds, json.dumps(out[ds]), flush=True)
json.dump(out, open(paths.data("bhb_lnl_without_prior.json"), "w"), indent=1)
