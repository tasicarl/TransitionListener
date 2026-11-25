"""Ranges over the main (largest) connected component of the 68% regions, without the single-cell
islands at (beta/H)_Gamma ~ 1.6 near the trapping boundary."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import paths  # noqa: E402  (src/, models/ and the folders of paper/reproduce on sys.path)
sys.path.insert(0, os.path.join(paths.PAPER, "figs"))
import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402
import transition_table as TT  # noqa: E402
import make_figures as mf  # noqa: E402
import plane_scan as ps  # noqa: E402
d = dict(np.load(paths.data("plane_scan_u1.npz"), allow_pickle=True)); G, L, names = d["G"], d["LOGV"], list(d["names"])
Gm, Lm = np.meshgrid(G, L, indexing="ij")
Q = {k: np.array([[float(TT.get(k, a, b)) for b in L] for a in G]) for k in ("betaH_R", "beta_Gamma_H", "betaH_S", "H_p_over_HV")}
REH = np.array([[ps.reheating(a, b) for b in L] for a in G])
for ds in ("NG15", "IPTA2"):
    lv, p = mf._levels(d[ds][names.index("nlo")])
    for lvl, nm in ((1, "68"), (0, "95")):
        r = p >= lv[lvl]; lab, n = ndimage.label(r)
        main = lab == max(range(1, n + 1), key=lambda i: p[lab == i].sum())
        m = main
        print(f"{ds} {nm}% main: g {Gm[m].min():.3f}-{Gm[m].max():.3f}, v {10**Lm[m].min():.2f}-{10**Lm[m].max():.2f} GeV, "
              f"(b/H)_R {np.nanmin(Q['betaH_R'][m]):.3f}-{np.nanmax(Q['betaH_R'][m]):.3f}, (b/H)_G {np.nanmin(Q['beta_Gamma_H'][m]):.2f}-{np.nanmax(Q['beta_Gamma_H'][m]):.2f}, "
              f"G/R {np.nanmin((Q['beta_Gamma_H']/Q['betaH_R'])[m]):.3f}-{np.nanmax((Q['beta_Gamma_H']/Q['betaH_R'])[m]):.3f}, "
              f"S-G {np.nanmin((Q['betaH_S']-Q['beta_Gamma_H'])[m]):.2f}-{np.nanmax((Q['betaH_S']-Q['beta_Gamma_H'])[m]):.2f}, "
              f"H/HV {np.nanmin(Q['H_p_over_HV'][m]):.3f}-{np.nanmax(Q['H_p_over_HV'][m]):.3f}, "
              f"T_reh {1e3*np.nanmin(REH[...,0][m]):.1f}-{1e3*np.nanmax(REH[...,0][m]):.1f} MeV, g_eff {np.nanmin(REH[...,1][m]):.2f}-{np.nanmax(REH[...,1][m]):.2f}, "
              f"m_A' {np.nanmin((Gm*10**Lm)[m]):.3f}-{np.nanmax((Gm*10**Lm)[m]):.3f} GeV, mass of islands {1 - p[m].sum()/p[r].sum():.4f} of the region")


# main-component values that text_numbers.py takes over all 68% cells
par = d
def main_of(arr):
    lv, p = mf._levels(arr); r = p >= lv[1]; lab, n = ndimage.label(r)
    return lab == max(range(1, n + 1), key=lambda i: p[lab == i].sum()), p
for ds in ("NG15", "IPTA2"):
    m, p = main_of(d[ds][names.index("nlo")])
    sh = d["par_T_p"] * (1 + d["par_alpha"]) ** 0.25 / d["par_T_reh"]
    ratio = TT.reheating_tl(Gm, Lm)[0] / TT.t_reh(Gm, Lm)
    print(f"{ds} main 68%: shortcut overestimate {np.nanmin(sh[m]):.3f}-{np.nanmax(sh[m]):.3f}; T_reh TL/SM-only {np.nanmin(ratio[m]):.3f}-{np.nanmax(ratio[m]):.3f}; "
          f"posterior mass at g < 0.44 {p[Gm < 0.44].sum() / p.sum():.3f}; area with a^3 P_f growing {np.mean((d['par_dlnV_dt_over_H'][m] > 0)):.3f}, mass {p[m & (d['par_dlnV_dt_over_H'] > 0)].sum() / p[m].sum():.3f}")
for name in ("sound waves", "expansion, bulk flow", "no expansion, bubbles R^-3"):
    m, p = main_of(d["NG15"][names.index(name)])
    print(f"NG15 {name} main 68%: g {Gm[m].min():.3f}-{Gm[m].max():.3f}, v {10**Lm[m].min():.2f}-{10**Lm[m].max():.2f}, (b/H)_R {np.nanmin(Q['betaH_R'][m]):.3f}-{np.nanmax(Q['betaH_R'][m]):.3f}")
