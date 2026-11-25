"""Print the numbers of sections 4.2, 4.3, 5 and appendix D from plane_scan_u1.npz: best
fits with their transition parameters, 68% intervals, the budget of every switch (1D shifts,
the 2D shift of the posterior mean in units of the default posterior's standard deviations
along its principal axes, the area ratio, and Delta ln L_max), the tension between the two
data sets, and the N_eff of the background. Run after plane_scan.py, in an environment with
matplotlib (make_figures imports it)."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import paths  # noqa: E402
sys.path.insert(0, os.path.join(paths.PAPER, "figs"))
import make_figures as mf  # noqa: E402

d = dict(np.load(paths.data(os.environ.get("SCAN_NPZ", "plane_scan_u1.npz")), allow_pickle=True))
G, LOGV, names = d["G"], d["LOGV"], list(d["names"])
k = names.index("nlo")
par = {key[4:]: d[key] for key in d if key.startswith("par_")}


def region(a):
    lv, p = mf._levels(a)
    return p >= lv[1], p


best = {}
for ds in ("NG15", "IPTA2"):
    for key, lab in ((ds, "no foreground"), (ds + "_bhb", "with the (A, gamma) foreground")):
        a = d[key][k]
        i, j = np.unravel_index(np.nanargmax(a), a.shape)
        in68, p = region(a)
        g_lo, g_hi = mf._interval(p, G, 0)
        v_lo, v_hi = mf._interval(p, LOGV, 1)
        bh = par["betaH_R"]
        print(f"{ds} ({lab}): best g = {G[i]:.3f}, v = {10**LOGV[j]:.2f} GeV, (beta/H)_R = {bh[i, j]:.1f}, "
              f"beta_Gamma/H = {par['beta_Gamma_H'][i, j]:.1f}, (beta/H)_S = {par['betaH_S'][i, j]:.1f}, "
              f"T_reh = {par['T_reh'][i, j]*1e3:.0f} MeV, T_p = {par['T_p'][i, j]*1e6:.2f} keV, "
              f"completion d ln(a^3 P_f)/dt/H = {par['dlnV_dt_over_H'][i, j]:+.2f}, lnL = {a[i, j]:.2f}")
        print(f"   68%: g {g_lo:.3f}-{g_hi:.3f}, v {10**v_lo:.2f}-{10**v_hi:.2f} GeV (log10 {v_lo:+.2f} to {v_hi:+.2f}), "
              f"width g {g_hi-g_lo:.3f}, log10 v {v_hi-v_lo:.2f}; (beta/H)_R in the 68% region "
              f"{np.nanmin(bh[in68]):.1f}-{np.nanmax(bh[in68]):.1f}, T_reh {np.nanmin(par['T_reh'][in68])*1e3:.0f}-"
              f"{np.nanmax(par['T_reh'][in68])*1e3:.0f} MeV; posterior mass with T_p outside the Delta S table "
              f"{np.sum(p[in68 & (par['in_table'] < 0.5)]) / np.sum(p[in68]):.2f}")
        best[key] = (i, j)
    for extra in ("_bins5", "_bins30", "_v1", "_hd", "_dmgp", "_binaryfog"):
        if ds + extra in d:
            a = d[ds + extra]; a = a if a.ndim == 2 else a[k]
            if not np.isfinite(a).any():
                continue
            i, j = np.unravel_index(np.nanargmax(a), a.shape)
            _, p = region(a)
            print(f"   {extra[1:]:9s}: best g = {G[i]:.3f}, v = {10**LOGV[j]:.2f}, lnL = {a[i, j]:.2f}; 68% g "
                  f"{mf._interval(p, G, 0)[0]:.3f}-{mf._interval(p, G, 0)[1]:.3f}, v "
                  f"{10**mf._interval(p, LOGV, 1)[0]:.2f}-{10**mf._interval(p, LOGV, 1)[1]:.2f}")

# completion: T_f exists (default) against the local criterion of Ellis et al. at T_p
jb = best["NG15"][1]
dl = par["dlnV_dt_over_H"][:, jb]
ic = np.where(np.diff(np.sign(np.nan_to_num(dl, nan=1.0))) != 0)[0]
print(f"\nlocal completion line at log10 v = {LOGV[jb]:+.2f}: g = {G[ic[0]]:.3f}-{G[ic[0] + 1]:.3f}" if len(ic) else "")
perc = np.isfinite(par["T_p"]); fin = np.isfinite(par["T_f"])
print(f"cells that percolate: {perc.sum()}, of which reach T_f: {(perc & fin).sum()}, fail the local criterion: {(perc & (par['dlnV_dt_over_H'] > 0)).sum()}")
a_loc = d["NG15"][names.index("local completion criterion")]
print(f"NANOGrav lnL max with the local criterion imposed: {np.nanmax(a_loc):.2f} (default {np.nanmax(d['NG15'][k]):.2f})")

# tension: each data set's best fit in the other
for a, b in (("NG15", "IPTA2"), ("IPTA2", "NG15")):
    i, j = best[a]
    print(f"{a} best fit in {b}: Delta lnL = {d[b][k][i, j] - np.nanmax(d[b][k]):.2f}")
tot = d["NG15"][k] + d["IPTA2"][k]
i, j = np.unravel_index(np.nanargmax(tot), tot.shape)
print(f"joint best fit (not used): g = {G[i]:.3f}, v = {10**LOGV[j]:.2f}; Delta lnL NG15 {d['NG15'][k][i, j] - np.nanmax(d['NG15'][k]):.2f}, "
      f"IPTA {d['IPTA2'][k][i, j] - np.nanmax(d['IPTA2'][k]):.2f}")
print(f"max Delta N_eff^GW on the grid: {np.nanmax(d['dneff_gw']):.3g}; min T_reh on the grid {np.nanmin(par['T_reh'])*1e3:.1f} MeV")

for ds in ("NG15", "IPTA2"):
    rows, info = mf.budget_rows(ds)
    print(f"\nbudget {ds}: width g {info['width_g']:.3f}, log10 v {info['width_logv']:.2f}; grid step "
          f"{100*info['dg']/info['width_g']:.0f}% / {100*info['dlogv']/info['width_logv']:.0f}% of the widths")
    print("   sym  name                                  shift, width change g  shift, width change v    D2   area  dlnLmax  edge")
    lnl0 = np.nanmax(d[ds][k])
    for r in rows:
        name = r[1]
        if name in names:
            dl_ = np.nanmax(d[ds][names.index(name)]) - lnl0
        else:
            key = {"binary foreground": "_bhb", "binaryfog": "_binaryfog", "HD instead of CURN": "_hd",
                   "DMGP instead of DMX": "_dmgp", "2024 release": "_v1", "bins: 5": "_bins5"}.get(name, "_bins30")
            # only the power-law foreground is fitted to the same data; the other rows are not comparable
            arr = d.get(ds + key) if name == "binary foreground" else None
            dl_ = np.nanmax(arr if arr.ndim == 2 else arr[k]) - lnl0 if arr is not None else np.nan
        print(f"   {mf.SYMBOL[name]:4s} {name:40s} {r[2][0]:5.0f}{r[2][1]:+5.0f}%        {r[3][0]:5.0f}{r[3][1]:+5.0f}%          "
              f"{r[5]:5.2f} {r[6]:5.2f}  {dl_:+6.2f}  {'>=' if r[7] else ''}")

# the 68% region of every variant: intervals in g, log10 v, (beta/H)_RH and T_reh, and the fit
print("\n68% regions of the variants (NANOGrav, then IPTA):")
for ds in ("NG15", "IPTA2"):
    for kk, name in enumerate(names):
        a = d[ds][kk]
        if not np.any(np.isfinite(a)):
            continue
        in68, p = region(a)
        g_lo, g_hi = mf._interval(p, G, 0); v_lo, v_hi = mf._interval(p, LOGV, 1)
        bh = par["betaH_R"]
        print(f"   {ds:5s} {name:38s} g {g_lo:.3f}-{g_hi:.3f}  log10 v {v_lo:+.2f} to {v_hi:+.2f}  "
              f"(beta/H)_RH {np.nanmin(bh[in68]):5.1f}-{np.nanmax(bh[in68]):5.1f}  lnL_max {np.nanmax(a):7.2f}")
