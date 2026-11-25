"""Numbers quoted in main.tex that plane_numbers.py does not print, from the current scan
(plane_scan_u1.npz), the maximised best fits (best_fits.json), TransitionListener's reheating
(tl_nodes_table.npz) and the transition table, all in paper/data. Prints them; reproduce/jobs/11_numbers.sh keeps
the output as output/text_numbers.txt.

Run with PYTHONPATH=<repo>/src in the environment of plane_scan.py."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import paths  # noqa: E402  (src/, models/ and the folders of paper/reproduce on sys.path)
sys.path.insert(0, os.path.join(paths.PAPER, "figs"))
import plane_scan as ps  # noqa: E402
import transition_table as TT  # noqa: E402
import make_figures as mf  # noqa: E402

d = dict(np.load(paths.data("plane_scan_u1.npz"), allow_pickle=True))
G, LOGV, names = d["G"], d["LOGV"], [str(n) for n in d["names"]]
k = names.index("nlo")
bf = json.load(open(paths.data("best_fits.json")))
f = np.logspace(-10, -6, 4000)


def point(ds):
    g, lv = bf["nlo"][ds]["at"]
    _, bh, al, bv = ps.central(g, lv)
    reh = ps.reheating(g, lv)
    treh, greh, hreh = reh
    spec = ps.make_spectrum(ps.COSMO_VARIANTS[0][1], reh, bh, al, bv)(f)
    tp = float(TT.get("T_p", g, lv))
    v = 10 ** lv
    short = tp * (1 + al) ** 0.25
    print(f"{ds} best fit (maximised): g = {g:.4f}, v = {v:.3f} GeV, lnL = {bf['nlo'][ds]['max']:.3f} "
          f"(grid {bf['nlo'][ds]['grid']:.3f})")
    print(f"   (beta/H)_RH = {bh:.2f}, (beta/H)_S3 = {float(TT.get('betaH_S', g, lv)):.2f}, "
          f"(beta/H)_Gamma = {float(TT.get('beta_Gamma_H', g, lv)):.2f}, T_p = {tp*1e6:.2f} keV, alpha = {al:.2e}")
    print(f"   T_reh (TL) = {treh*1e3:.1f} MeV, g_reh = {greh:.2f}, h_reh = {hreh:.2f}; SM-only balance "
          f"{float(TT.t_reh(g, lv))*1e3:.1f} MeV (TL/SM = {treh/float(TT.t_reh(g, lv)):.3f}); "
          f"shortcut T_p(1+alpha)^1/4 = {short*1e3:.1f} MeV (x{short/treh:.2f})")
    print(f"   m_A'/T_reh = {0.0 + g*v/treh:.1f}, m_phi/T_reh = {np.sqrt(3/(8*np.pi**2))*g**2*v/treh:.2f}")
    fp = f[np.argmax(spec)]
    print(f"   spectrum peak f = {fp*1e9:.1f} nHz, h^2 Omega_peak = {spec.max():.2e}")
    # horizon frequency f_H of eq. (20) of 2511.15687 with TL's redshift
    H_p0 = 1.65e-5 * (treh / 100) * (greh / 100) ** 0.5 * (100 / hreh) ** (1 / 3)
    # with the expansion speed, as in the bracket of eq. (lv)
    print(f"   f_H = {H_p0/(2*np.pi)*(1-0.6*bv**-0.67)*1e9:.2f} nHz")
    V = dict(ps.COSMO_VARIANTS)
    for name in ("no expansion, bubbles R^-3", "sound waves", "LISA update 2019"):
        s2 = ps.make_spectrum(V[name], reh, bh, al, bv)(f)
        print(f"   {name}: peak x{s2.max()/spec.max():.2f}, frequency x{f[np.argmax(s2)]/fp:.2f}")
    return g, lv


gN, lN = point("NG15")
gI, lI = point("IPTA2")
lnl = d["NG15"][k]
m = np.nanmax(lnl)
for g0 in (0.44, 0.43, 0.42, 0.415):
    i = int(np.argmin(np.abs(G - g0)))
    print(f"NANOGrav profile over v at g = {G[i]:.3f}: Delta lnL = {np.nanmax(lnl[i]) - m:+.2f}")
# regions: local criterion, (beta/H)_RH and T_reh ranges, shortcut factor over the NANOGrav 68% region
for ds in ("NG15", "IPTA2"):
    lv_, p = mf._levels(d[ds][k])
    reg = p >= lv_[1]
    par = {q: d["par_" + q] for q in ("dlnV_dt_over_H", "betaH_R", "T_reh", "T_p", "alpha", "beta_Gamma_H")}
    grows = reg & (par["dlnV_dt_over_H"] > 0)
    print(f"{ds} 68%: area fraction with a^3 P_f growing at T_p {grows.sum()/reg.sum():.3f}, posterior mass "
          f"{p[grows].sum()/p[reg].sum():.3f}; (beta/H)_Gamma min {np.nanmin(par['beta_Gamma_H'][reg]):.2f}")
    sh = par["T_p"] * (1 + par["alpha"]) ** 0.25 / par["T_reh"]
    print(f"   shortcut overestimate of T_reh: {np.nanmin(sh[reg]):.2f}-{np.nanmax(sh[reg]):.2f}")
    Gm, Lm = np.meshgrid(G, LOGV, indexing="ij")
    ratio = TT.reheating_tl(Gm, Lm)[0] / TT.t_reh(Gm, Lm)
    print(f"   T_reh TL / SM-only in the region: {np.nanmin(ratio[reg]):.3f}-{np.nanmax(ratio[reg]):.3f}")
print(f"cross: IPTA best fit in NANOGrav Delta lnL = "
      f"{ps.lnL('NG15', ps.make_spectrum(ps.COSMO_VARIANTS[0][1], ps.reheating(gI, lI), *ps.central(gI, lI)[1:])) - bf['nlo']['NG15']['max']:.2f}; "
      f"NANOGrav best fit in IPTA {ps.lnL('IPTA2', ps.make_spectrum(ps.COSMO_VARIANTS[0][1], ps.reheating(gN, lN), *ps.central(gN, lN)[1:])) - bf['nlo']['IPTA2']['max']:.2f}")
print(f"max Delta N_eff^GW on the grid {np.nanmax(d['dneff_gw']):.4f}; in the NANOGrav 95% region "
      f"{np.nanmax(np.where(mf._levels(d['NG15'][k])[1] >= mf._levels(d['NG15'][k])[0][0], d['dneff_gw'], np.nan)):.4f}; "
      f"min T_reh on the grid {np.nanmin(d['par_T_reh'])*1e3:.1f} MeV")
for name, v in bf.items():
    print(f"best fit {name:40s}", "  ".join(f"{ds} {r['max']:+.3f}" for ds, r in v.items() if r))
