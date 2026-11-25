"""The numbers behind the prior range 10 MeV < v < 100 GeV (make_figures.PRIOR_LOGV): the posterior mass of every variant
outside it, the B1 rows on other ranges, the posterior mean of B1 (survey-averaged priors) and the fits without the
causal tail (footnote fn:causal)."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import paths  # noqa: E402  (src/, models/ and the folders of paper/reproduce on sys.path)
sys.path.insert(0, os.path.join(paths.PAPER, "figs"))
import numpy as np  # noqa: E402
import make_figures as M  # noqa: E402

d = np.load(paths.data("plane_scan_u1.npz"), allow_pickle=True)
L, names = d["LOGV"], list(d["names"])
print("posterior mass (prior 3.2 MeV - 10 TeV) below 10 MeV and above 100 GeV, rows without binaries:")
for ds in ("NG15", "IPTA2"):
    lo = hi = (0.0, "")
    for k, n in enumerate(names):
        p = np.exp(np.nan_to_num(d[ds][k] - np.nanmax(d[ds][k]), nan=-1e9)); p /= p.sum()
        lo = max(lo, (p[:, L < -2 - 1e-9].sum(), n)); hi = max(hi, (p[:, L > 2 + 1e-9].sum(), n))
    print(f"  {ds}: largest below 10 MeV {lo[0]:.1e} ({lo[1]}), largest above 100 GeV {hi[0]:.1e} ({hi[1]})")

print("B1 rows, D and 2 sigma upper end of g, for upper edges of v:")
for lim in ((-2.0, 1.0), (-2.0, 1.5), (-2.0, 2.0), (-2.0, 4.0)):
    M.PRIOR_LOGV = M.BINARIES_LOGV = lim
    M._BUDGET.clear()
    for ds in ("NG15", "IPTA2"):
        rows, _ = M.budget_rows(ds)
        out = [f"{r[1].replace('binaryfog, ', '').replace('binaryfog', 'NANOGrav priors')}: D {r[5]:.2f}, g2 < {r[9]['g2'][1]:.3f}"
               for r in rows if r[1].startswith(("binary", "binaryfog"))]
        print(f"  10^{lim[0]}-10^{lim[1]} GeV {ds}: " + "; ".join(out))

M.PRIOR_LOGV = M.BINARIES_LOGV = (-2.0, 2.0)
from scipy.interpolate import RegularGridInterpolator
D = M._budget_scan(); G, LV = D["G"], D["LOGV"]
bh = RegularGridInterpolator((G, LV), np.log(D["par_betaH_R"]))
for key, arr in (("default", D["NG15"][list(D["names"]).index("nlo")]), ("B1 survey-averaged", D["NG15_binaryfog_gsa"])):
    _, p = M._levels(arr); m, _ = M._moments(p, G, LV)
    print(f"posterior mean {key}: g = {m[0]:.4f}, log10 v = {m[1]:.3f}, (beta/H)_R there {np.exp(bh(m)[0]):.2f}")

print("fits without expansion, NANOGrav, upper ends of v [GeV] (1 sigma, 2 sigma) and max ln L:")
n = M._grid_npz(paths.data("notail_scan.npz"))
for k, name in enumerate(n["names"]):
    _, p = M._levels(n["NG15"][k])
    print(f"  {name:44s} {10 ** M._interval(p, n['LOGV'], 1, 0.68)[1]:6.3g} {10 ** M._interval(p, n['LOGV'], 1, 0.95)[1]:6.3g}"
          f"  {np.nanmax(n['NG15'][k]):.2f}")
