"""Numbers of sections 2.3, 3.2, 3.4 and appendix C not printed elsewhere: alpha in the 1 sigma regions, the posterior below (beta/H)_Gamma
= 8, ln(T_p/T_f) in the 1 sigma region where the false-vacuum volume grows, H_p/H_V, and unrounded distances."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import paths  # noqa: E402
sys.path.insert(0, os.path.join(paths.PAPER, "figs"))
import make_figures as mf  # noqa: E402

d = dict(np.load(paths.data("plane_scan_u1.npz"), allow_pickle=True))
names = list(d["names"])
par = {k[4:]: d[k] for k in d if k.startswith("par_")}
print(sorted(par))
k = names.index("nlo")
for ds in ("NG15", "IPTA2"):
    lv, p = mf._levels(d[ds][k])
    for lvl, nm in ((1, "68"), (0, "95")):
        r = p >= lv[lvl]
        print(f"{ds} {nm}%: alpha {np.nanmin(par['alpha'][r]):.3g}-{np.nanmax(par['alpha'][r]):.3g}; T_p max {np.nanmax(par['T_p'][r])*1e6:.1f} keV")
    q = np.where(np.isfinite(d[ds][k]), np.exp(d[ds][k] - np.nanmax(d[ds][k])), 0.0)
    print(f"  posterior with (beta/H)_Gamma < 8: {q[par['beta_Gamma_H'] < 8].sum() / q.sum():.3f}")
    if "T_f" in par:
        r = (p >= lv[1]) & (par["dlnV_dt_over_H"] > 0)
        e = np.log(par["T_p"][r] / par["T_f"][r])
        print(f"  ln(T_p/T_f) in the 1 sigma cells with a growing false-vacuum volume: {np.nanmin(e):.2f}-{np.nanmax(e):.2f}")
