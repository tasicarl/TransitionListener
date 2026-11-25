"""Shift, width change and distance D for rows the budget figure does not show (the lattice-size
shifts, the LO bounce on the NLO-matched theory, Delta S held constant), from plane_scan_u1.npz,
with the measures of make_figures.budget_rows."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import paths  # noqa: E402  (src/, models/ and the folders of paper/reproduce on sys.path)
sys.path.insert(0, os.path.join(paths.PAPER, "figs"))
import numpy as np  # noqa: E402
import make_figures as mf  # noqa: E402
d = dict(np.load(paths.data("plane_scan_u1.npz"), allow_pickle=True))
G, L, names = d["G"], d["LOGV"], list(d["names"])
for ds in ("NG15", "IPTA2"):
    lnl = d[ds]; _, p0 = mf._levels(lnl[names.index("nlo")]); m0, C0 = mf._moments(p0, G, L); Ci = np.linalg.inv(C0)
    iv0 = [mf._interval(p0, G, 0), mf._interval(p0, L, 1)]; w = [iv0[0][1] - iv0[0][0], iv0[1][1] - iv0[1][0]]
    for name in ("lattice shift -", "lattice scale -", "leading order", "Delta S tables", "4D daisy"):
        _, p = mf._levels(lnl[names.index(name)]); iv = [mf._interval(p, G, 0), mf._interval(p, L, 1)]
        s = [100 * abs(0.5 * (iv[a][0] + iv[a][1]) - 0.5 * (iv0[a][0] + iv0[a][1])) / w[a] for a in (0, 1)]
        ch = [100 * ((iv[a][1] - iv[a][0]) - w[a]) / w[a] for a in (0, 1)]
        m, _ = mf._moments(p, G, L); dm = m - m0
        print(f"{ds:5s} {name:18s} shift g {s[0]:5.1f}% (width {ch[0]:+.0f}%), log10 v {s[1]:5.1f}% (width {ch[1]:+.0f}%), "
              f"D {np.sqrt(dm @ Ci @ dm):.2f}, g {iv[0][0]:.3f}-{iv[0][1]:.3f}, centre shift in g {0.5*(iv[0][0]+iv[0][1]) - 0.5*(iv0[0][0]+iv0[0][1]):+.4f}")
