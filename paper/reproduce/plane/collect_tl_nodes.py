"""Collect the TransitionListener runs at the nodes of the transition table (tl_nodes.py, read from
<work>/<TL_NODES_DIR>/g*.json) into paper/data/tl_nodes_table.npz: T_reh (Treh_SM_GeV), the energy and entropy degrees of freedom right after
reheating (g_eff_tot_reh, h_eff_tot_reh), and TL's T_p, (beta/H)_RH and T_f for the comparison
with the direct integration, on the (g, log10 v) nodes of transition_table.npz.

The reheating temperature is needed wherever a transition completes, also at nodes where TL did
not finish or did not run (no percolation in the table, or an error). There it is filled from the
ratio T_reh^TL/T_reh^SM (T_reh^SM, the Standard-Model-only energy balance rho_R(T_reh) = Delta V, is in
the table at every node), taken from a smooth fit of that ratio in (ln (beta/H)_RH, log10 v) over the
finished nodes; the degrees of freedom are the SM table at the filled T_reh plus the dark-sector
excess of the nearest finished node. `filled` marks these nodes.

The TL runs sit on a grid that is a subset of that of the transition table (the nodes of tl_nodes.txt,
without most of the half steps in g). The runs are collected on their own grid first; a node of the
table grid then takes
  - the value of the run grid where it is one of its nodes;
  - the PCHIP interpolation of the run grid (in g and log10 v, of the logarithms) where the four
    nodes of the run grid around it all finished (`interpolated`);
  - otherwise the fit of the ratio at its own (beta/H)_RH (`filled`), so that in the slow tail,
    where no TL run finished, T_reh follows the speed of the finer table node by node.
The efficiency kappa = kappa_phi + kappa_sw of every finished run goes, on the run grid, into
paper/data/kappa_nodes_table.npz.
"""
import glob
import json
import os

import numpy as np
from scipy.interpolate import RegularGridInterpolator

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data")
WORK = os.path.abspath(os.environ.get("PAPER_WORK", os.path.join(HERE, "..", "work")))
T = np.load(os.path.join(DATA, "transition_table.npz"))
G, LV = T["G"], np.round(T["LOGV"], 4)
treh_sm = T["T_reh_SM"]
k_nlo = [str(v) for v in T["VARIANTS"]].index("nlo")
bh = T["betaH_R"][:, k_nlo, :]
keys = dict(T_reh_TL="Treh_SM_GeV", g_reh_TL="g_eff_tot_reh", h_reh_TL="h_eff_tot_reh",
            T_p_TL="Tperc_SM_GeV", betaH_R_TL="betaH_RH", T_f_TL="Tf_SM_GeV", betaH_S_TL="betaH_S3")
# TL_NODES_DIR selects the folder of runs under the work folder (default tl_nodes). Records whose error says that
# there is no line-scan table are left out, so that their couplings are interpolated between the run couplings like
# any node off the run grid.
NODES_DIR = os.environ.get("TL_NODES_DIR", "tl_nodes")
runs = [json.load(open(f)) for f in glob.glob(os.path.join(WORK, NODES_DIR, "g*.json"))]
runs = [r for r in runs if "no line-scan table" not in (r.get("error") or "")]

# the grid of the runs, and where its nodes sit in the transition table
GC = np.unique(np.round([r["g"] for r in runs], 6))
LC = np.unique(np.round([r["logv"] for r in runs], 4))
IC = np.array([int(np.argmin(np.abs(G - g))) for g in GC])
JC = np.array([int(np.argmin(np.abs(LV - l))) for l in LC])
assert np.allclose(G[IC], GC) and np.allclose(LV[JC], LC), "a run lies off the transition table grid"

out = {k: np.full((len(GC), len(LC)), np.nan) for k in keys}
err = np.zeros((len(GC), len(LC)), dtype=bool)
seen = np.zeros_like(err)
for r in runs:
    i, j = int(np.argmin(np.abs(GC - r["g"]))), int(np.argmin(np.abs(LC - r["logv"])))
    seen[i, j] = True
    err[i, j] = bool(r.get("error"))
    for k, src in keys.items():
        x = r["observables"].get(src)
        if x is not None and np.isfinite(x) and x > 0:
            out[k][i, j] = x
ok = np.isfinite(out["T_reh_TL"]) & np.isfinite(out["g_reh_TL"]) & np.isfinite(out["h_reh_TL"])
treh_sm_c, bh_c = treh_sm[np.ix_(IC, JC)], bh[np.ix_(IC, JC)]
ratio = np.where(ok, out["T_reh_TL"] / treh_sm_c, np.nan)
# The ratio to the instantaneous balance measures how much of the released energy redshifts before
# percolation, which depends on how long the conversion takes, i.e. on (beta/H)_RH, and weakly on v
# through the degrees of freedom. Nodes where TL did not finish are filled from a cubic fit of
# ln(ratio) in (ln (beta/H)_RH, log10 v) over the finished nodes with (beta/H)_RH < 60; nodes without
# a transition take the nearest finished node of the same coupling.
Lm = np.broadcast_to(LC[None, :], bh_c.shape)
sel = ok & np.isfinite(bh_c) & (bh_c < 60)


def design(x, y):
    return np.stack([x**i * y**j for i in range(4) for j in range(4) if i + j <= 3], axis=-1)


coef, *_ = np.linalg.lstsq(design(np.log(bh_c[sel]), Lm[sel]), np.log(ratio[sel]), rcond=None)
fit = lambda b, l: np.exp(design(np.log(b), l) @ coef)
resid = np.log(ratio[sel]) - np.log(fit(bh_c[sel], Lm[sel]))
print(f"fit of the ratio: rms residual {resid.std():.4f}, max {np.abs(resid).max():.4f} (in ln)")
# the excess of TL's degrees of freedom over the SM table at T_reh (the dark Higgs, mostly)
tab = np.genfromtxt(os.path.join(HERE, "..", "..", "..", "src", "transitionlistener", "tab_data", "geff_SM.dat"))
g_sm = lambda T_: np.interp(np.log(T_), np.log(tab[:, 0]), tab[:, 1])
h_sm = lambda T_: np.interp(np.log(T_), np.log(tab[:, 0]), tab[:, 3])


def fill(ratio, dg, dh, have, treh_sm, bh, lv):
    """T_reh, g_reh, h_reh where `have` is False: the fit at the node's own speed, the nearest
    node of the same coupling (or the nearest coupling) where there is no transition; the
    dark-sector excess of the degrees of freedom from the nearest node that has it."""
    ratio, dg, dh = ratio.copy(), dg.copy(), dh.copy()
    use_fit = ~have & np.isfinite(bh)
    ratio[use_fit] = fit(bh[use_fit], np.broadcast_to(lv[None, :], bh.shape)[use_fit])
    for arr_src in (ratio, dg, dh):
        src = arr_src.copy()
        for i in range(src.shape[0]):
            good = np.flatnonzero(np.isfinite(src[i]))
            row = i
            if len(good) == 0:
                rows = np.flatnonzero(np.isfinite(src).any(axis=1))
                row = rows[np.argmin(np.abs(rows - i))]
                good = np.flatnonzero(np.isfinite(src[row]))
            for j in np.flatnonzero(~np.isfinite(src[i])):
                arr_src[i, j] = src[row, good[np.argmin(np.abs(good - j))]]
    t = ratio * treh_sm
    return t, g_sm(t) + dg, h_sm(t) + dh


# 1. the run grid
dg_c = np.where(ok, out["g_reh_TL"] - g_sm(np.where(ok, out["T_reh_TL"], 1.0)), np.nan)
dh_c = np.where(ok, out["h_reh_TL"] - h_sm(np.where(ok, out["T_reh_TL"], 1.0)), np.nan)
t_c, gr_c, hr_c = fill(ratio, dg_c, dh_c, ok, treh_sm_c, bh_c, LC)
for k, v in (("T_reh_TL", t_c), ("g_reh_TL", gr_c), ("h_reh_TL", hr_c)):
    out[k] = np.where(ok, out[k], v)

# 2. the grid of the transition table
shape = (len(G), len(LV))
fine = {k: np.full(shape, np.nan) for k in keys}
have = np.zeros(shape, dtype=bool)          # a run that finished, or interpolated between four
interp = np.zeros(shape, dtype=bool)
on_run = np.zeros(shape, dtype=bool)
for k in keys:
    fine[k][np.ix_(IC, JC)] = out[k]
on_run[np.ix_(IC, JC)] = True
have[np.ix_(IC, JC)] = ok
pchip = {k: RegularGridInterpolator((GC, LC), np.log(out[k]), method="pchip") for k in ("T_reh_TL", "g_reh_TL", "h_reh_TL")}
for i, g in enumerate(G):
    a = np.searchsorted(GC, g) - 1
    for j, lv in enumerate(LV):
        if on_run[i, j] or not (0 <= a < len(GC) - 1):
            continue
        b = np.searchsorted(LC, lv) - 1
        if not (0 <= b < len(LC) - 1) or not ok[a:a + 2, b:b + 2].all():
            continue
        for k in pchip:
            fine[k][i, j] = float(np.exp(pchip[k]([[g, lv]])[0]))
        interp[i, j] = have[i, j] = True
ratio_f = np.where(have, fine["T_reh_TL"] / treh_sm, np.nan)
dg_f = np.where(have, fine["g_reh_TL"] - g_sm(np.where(have, fine["T_reh_TL"], 1.0)), np.nan)
dh_f = np.where(have, fine["h_reh_TL"] - h_sm(np.where(have, fine["T_reh_TL"], 1.0)), np.nan)
t_f, gr_f, hr_f = fill(ratio_f, dg_f, dh_f, have, treh_sm, bh, LV)
filled = ~have
# the nodes of the run grid keep their values (also where they were filled on the run grid)
filled[np.ix_(IC, JC)] = ~ok
for k, v in (("T_reh_TL", t_f), ("g_reh_TL", gr_f), ("h_reh_TL", hr_f)):
    fine[k] = np.where(have | on_run, fine[k], v)
err_f = np.zeros(shape, dtype=bool); err_f[np.ix_(IC, JC)] = err
seen_f = np.zeros(shape, dtype=bool); seen_f[np.ix_(IC, JC)] = seen
np.savez(os.path.join(DATA, "tl_nodes_table.npz"), G=G, LOGV=T["LOGV"], filled=filled, interpolated=interp,
         error=err_f, run=seen_f, **fine)
print(f"run grid {len(GC)} x {len(LC)}: nodes run {seen.sum()} of {seen.size}, with T_reh {ok.sum()}, "
      f"errors {err.sum()}, filled {(~ok).sum()}")
print(f"table grid {len(G)} x {len(LV)}: interpolated {interp.sum()}, filled {filled.sum()} (of which on the run grid {(~ok).sum()})")
# The efficiency at the nodes: kappa = kappa_phi + kappa_sw of TL's run; transition_table.kappa_tl interpolates it.
kp = {k: np.full((len(GC), len(LC)), np.nan) for k in ("kappa_phi", "kappa_sw")}
for r_ in runs:
    if r_.get("error"):
        continue
    i, j = int(np.argmin(np.abs(GC - r_["g"]))), int(np.argmin(np.abs(LC - r_["logv"])))
    for k_ in kp:
        kp[k_][i, j] = r_["observables"].get(k_, np.nan)
np.savez(os.path.join(DATA, "kappa_nodes_table.npz"), G=GC, LOGV=LC, kappa=kp["kappa_phi"] + kp["kappa_sw"], **kp)
r = ratio[ok]
print(f"T_reh TL / SM-only balance: median {np.median(r):.3f}, range {r.min():.3f} to {r.max():.3f}")
