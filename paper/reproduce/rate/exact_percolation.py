"""Exact percolation of the fermion-free conformal U(1) at any (g, v) and for any variant of
the rate, by direct integration of the false-vacuum fraction on the self-consistent mean
background (mean_background.average_background). Independent of TL's
percolation solvers; used to validate them and, through build_exact_table.py, to compute the
transition parameters on the (g, v) plane.

Rate: Gamma(T) = T^4 (S/2 pi)^{3/2} e^{-S},  S = S3_LO(T) + Delta S(T) (+ variant shifts).
RateModel solves S3_LO on the temperature grid of rate_history_u1.py (cached in
<work>/lo_actions) and takes Delta S from the table_mu1.json of the line-scan point
(rate_history_u1.point_dir); build_exact_table.py replaces both. Both are
functions of T/mu0 at fixed g, so a different v only rescales T and shifts
ln(Gamma/H_V^4) by -4 ln(v/v_line) (H_V ~ v^2 / M_Pl).

Time: e-folds tau with a ~ 1/(T g_s(T)^{1/3}) (TL's convention; 'entropy') or a ~ 1/T
('dark', a dark sector that has decoupled from the SM and keeps its own temperature).

Output per point: T_p (P_true = f_perc), T_nuc (one bubble per Hubble volume), T_1/e
(P_false = 1/e), (beta/H)_S = T dS/dT, beta_Gamma = d ln Gamma/d tau at T_p (prefactor
included), (beta/H)_R = (8 pi/f_perc)^{1/3}/(R_* H) with R_* = n_b^{-1/3} at T_p, the
completion measure d ln(a^3 P_f)/dtau at T_p, and alpha(T_p).

Run as a script, it compares the line points with TransitionListener's line scan
(paper/data/linescan_u1_mu0_1GeV.json) and writes <work>/validation_line.json.
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import paths  # noqa: E402  (src/, models/ and the folders of paper/reproduce on sys.path)
ROOT = paths.REPO
from transitionlistener.eft_rate import abelian_higgs_3d  # noqa: E402
import mean_background as C  # noqa: E402
import rate_history_u1 as R  # noqa: E402

F_PERC = 0.28957
MPL = R.MPL


class RateModel:
    """The rate of one coupling as a function of T at the line-scan v; variants shift S."""

    def __init__(self, g, n_T=140, lo_matching=False):
        self.g = g
        pdir = R.point_dir(g)
        summ = json.load(open(os.path.join(pdir, "summary.json")))
        self.v_line = float(summ["v_GeV"])
        self.anchor = float(summ["anchor_GeV"])
        self.dV_line = R.vacuum_energy(g, self.v_line)
        Ts = np.geomspace(self.anchor * 0.05, self.anchor * 8.0, n_T)      # the grid of rate_history_u1.py
        cache = paths.work("lo_actions", f"lo_u1_g{g:.3f}.json" if not lo_matching else f"S3_g{g:.3f}_LOm.json")
        done = json.load(open(cache)) if os.path.exists(cache) else {}
        S3 = []
        for T in Ts:
            key = f"{T:.8e}"
            if key not in done:
                done[key] = self._S3(T, lo_matching)
            S3.append(done[key])
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        json.dump(done, open(cache, "w"))
        self.Ts = Ts
        self.S3 = np.array(S3, float)
        tabp = os.path.join(pdir, "table_mu1.json")
        self.corr = None
        if os.path.exists(tabp):
            t = json.load(open(tabp))["table"]
            self.corr = abelian_higgs_3d.NLOCorrection(t["T_GeV"], t["delta_S"], S3_LO=t.get("S3_LO"))
            self.table_T = (min(t["T_GeV"]), max(t["T_GeV"]))

    def _S3(self, T, lo_matching):
        mu0, lam0 = R.MU0, 0.0
        p = R.dr.matched_soft_parameters(self.g, lam0, mu0, T, mu4_over_T=R.MU4,
                                         matching_order="LO" if lo_matching else "NLO", mu3=self.g * T)
        try:
            return float(abelian_higgs_3d.solve_LO_bounce(p["m3sq"], p["lam3"], p["g3sq"], m_debye_sq=p["mD2"])[0])
        except (ValueError, RuntimeError):
            return float("nan")

    def exponent(self, T_line, variant):
        """S at line-scan temperatures T_line for a variant: dict(nlo=True/False, shift=const,
        dS_of_T=callable(T_line, dS) for T-dependent shifts)."""
        S3 = np.interp(np.log(T_line), np.log(self.Ts), self.S3, left=np.nan, right=np.nan)
        dS = np.zeros_like(S3)
        if variant.get("nlo", True) and self.corr is not None:
            dS = np.array([float(self.corr(T, S3_LO=s)) if np.isfinite(s) else 0.0 for T, s in zip(T_line, S3)])
        S = S3 + dS + variant.get("shift", 0.0)
        if "dS_of_T" in variant:
            S = S + variant["dS_of_T"](T_line, dS)
        return S


def percolate(rm, v=None, variant=None, time="entropy", n_bg=16000, f_perc=F_PERC, clock="lna", radiation=True):
    """Exact percolation of rate model rm at vacuum expectation value v (default: line).

    clock = "lna" (default) evaluates the rate at ln a of the self-consistent background,
    T ~ 1/(a g_s^1/3), which stays right once the released energy redshifts and H falls below
    H_V. clock = "t" evaluates it at cosmic time t (in 1/H_V) identified with the e-folds tau of
    the temperature (the deltaPT convention; exact only while H = H_V). radiation = True (default) adds the plasma's radiation rho_rad(T(a)) with its
    changing degrees of freedom to the background; False is pure vacuum energy (alpha -> inf)."""
    variant = variant or {}
    v = rm.v_line if v is None else v
    s = v / rm.v_line
    dV = rm.dV_line * s**4
    HV = math.sqrt(dV / 3.0) / MPL
    # temperatures in the frame of this v; the exponent is that of T/s on the line
    Tg = rm.Ts * s
    S = rm.exponent(rm.Ts, variant)
    ok = np.isfinite(S) & (S > 0)
    lnG = 4 * np.log(Tg) + 1.5 * np.log(np.where(ok, S, 1.0) / (2 * math.pi)) - S - 4 * math.log(HV)
    lnG = np.where(ok, lnG, -1e3)
    # e-folds: a ~ 1/(T g_s^{1/3}) or 1/T, tau = 0 at the highest T
    gs = R.g_s(Tg) if time == "entropy" else np.ones_like(Tg)
    tau = np.log(Tg[-1] * (R.g_s(Tg[-1]) if time == "entropy" else 1.0) ** (1 / 3) / (Tg * gs ** (1 / 3)))
    order = np.argsort(tau)
    tau, lnG_t, T_t, S_t = tau[order], lnG[order], Tg[order], S[order]

    hi_slope = (lnG_t[-1] - lnG_t[-2]) / (tau[-1] - tau[-2])

    def lnGf(t):
        t = np.asarray(t, float)
        return np.where(t > tau[-1], lnG_t[-1] + hi_slope * (t - tau[-1]), np.interp(t, tau, lnG_t, left=-1e3))
    # start where the rate is negligible, end well after completion
    i0 = np.argmax(lnG_t > -60) if np.any(lnG_t > -60) else 0
    t_lo = tau[max(i0 - 1, 0)]
    rho_lo = None
    plasma = None
    if radiation and clock == "lna":
        # the plasma's radiation at the temperature of each scale factor, with its changing
        # degrees of freedom; the released energy is carried separately and redshifts
        plasma = lambda lna: R.rho_rad(float(np.exp(np.interp(lna, tau, np.log(T_t))))) / dV
    elif radiation:
        rho_lo = float(R.rho_rad(float(np.exp(np.interp(t_lo, tau, np.log(T_t)))))) / dV
    bg = C.average_background(lnGf, t_lo, tau[-1], n=n_bg, alpha=1e30, clock=clock, rho_R_lo=rho_lo,
                              rho_plasma=plasma)
    t, a, H, eta, F = bg["t"], bg["a"], bg["H"], bg["eta"], bg["F"]
    # the e-folds of the temperature at each background time: t itself, or ln a on the background
    x = t if clock == "t" else np.log(a)
    P_t = 1 - F
    if P_t[-1] < f_perc:
        # 'rising': ln(Gamma/H^4) still grows at the lowest temperature of the window, so
        # the transition may percolate below it (a limit of the window, not a result)
        return dict(ok=False, reason="no percolation in the window", g=rm.g, v=v,
                    rising=bool(hi_slope > 0), lnG_max=float(np.max(lnG_t)), P_t_end=float(P_t[-1]))
    ip = np.searchsorted(P_t, f_perc)
    w = (f_perc - P_t[ip - 1]) / (P_t[ip] - P_t[ip - 1])
    tp = t[ip - 1] + w * (t[ip] - t[ip - 1])
    xp = float(np.interp(tp, t, x))
    Tp = float(np.exp(np.interp(xp, tau, np.log(T_t))))
    # T_f: completion as TransitionListener defines it, P_true = 0.99 at a finite temperature.
    # If the window ends first, 'f_rising' says whether the rate still grows there (then the
    # transition may complete below the window) or has turned over (the true-vacuum fraction
    # saturates: the false vacuum survives in patches)
    P_FINAL = 0.99
    if P_t[-1] >= P_FINAL:
        i_f = np.searchsorted(P_t, P_FINAL)
        w_f = (P_FINAL - P_t[i_f - 1]) / (P_t[i_f] - P_t[i_f - 1])
        t_f = t[i_f - 1] + w_f * (t[i_f] - t[i_f - 1])
        T_f = float(np.exp(np.interp(float(np.interp(t_f, t, x)), tau, np.log(T_t))))
        f_rising = False
    else:
        T_f = math.nan
        f_rising = bool(hi_slope > 0)
    # T_1/e (false-vacuum fraction 1/e)
    ie = np.searchsorted(P_t, 1 - math.exp(-1))
    t_e = t[min(ie, len(t) - 1)]
    T_e = float(np.exp(np.interp(float(np.interp(t_e, t, x)), tau, np.log(T_t)))) if ie < len(t) else math.nan
    # bubble number density per comoving volume up to t (nucleation in the false vacuum)
    dt = t[1] - t[0]
    # (cumulative trapezoid: a left sum is first order and at beta/H ~ 200 puts (beta/H)_RH
    # 1.6 % high on 16000 steps)
    rate = np.exp(lnGf(x)) * a**3 * F
    nb = np.concatenate([[0.0], np.cumsum(0.5 * (rate[1:] + rate[:-1]) * dt)])
    nb_p = float(np.interp(tp, t, nb))
    a_p = float(np.interp(tp, t, a)); H_p = float(np.interp(tp, t, H))
    Rstar_H = (nb_p ** (-1 / 3)) * a_p * H_p           # physical R_* times H at T_p
    betaH_R = (8 * math.pi / f_perc) ** (1 / 3) / Rstar_H
    # T_nuc: one bubble per Hubble volume, N = (4 pi/3) nb / (a H)^3 = 1 (comoving Hubble volume)
    Nh = 4 * math.pi / 3 * nb / (a * H) ** 3
    inuc = np.searchsorted(Nh, 1.0)
    T_nuc = float(np.exp(np.interp(x[min(inuc, len(t) - 1)], tau, np.log(T_t)))) if inuc < len(t) else math.nan
    # slopes at T_p
    asc = np.argsort(T_t)                                          # np.interp needs increasing abscissae
    lnT = np.log(T_t[asc]); S_asc = S_t[asc]
    fin = np.isfinite(S_asc)
    dS_dlnT = np.gradient(S_asc[fin], lnT[fin])
    betaH_S = float(np.interp(math.log(Tp), lnT[fin], dS_dlnT))    # T dS/dT (S = S3/T + ...)
    beta_G = float(np.interp(tp, 0.5 * (t[1:] + t[:-1]), np.diff(lnGf(x)) / np.diff(t)))   # per unit t (H_V = 1)
    beta_G_H = beta_G / H_p
    # completion: d ln(a^3 P_f)/dt at t_p over H
    lnV = np.log(np.clip(a**3 * F, 1e-300, None))
    dlnV = float(np.interp(tp, 0.5 * (t[1:] + t[:-1]), np.diff(lnV) / np.diff(t))) / H_p
    alpha_p = dV / R.rho_rad(Tp)
    return dict(ok=True, g=rm.g, v=v, T_p=Tp, T_nuc=T_nuc, T_1e=T_e, T_f=T_f, P_t_end=float(P_t[-1]),
                f_rising=f_rising, betaH_S=betaH_S, beta_Gamma_H=beta_G_H,
                betaH_R=betaH_R, Rstar_H=Rstar_H, H_p_over_HV=H_p, H_V_GeV=HV, dlnV_dt_over_H=dlnV, alpha=alpha_p, S_p=float(np.interp(math.log(Tp), lnT[fin], S_asc[fin])),
                in_table=(rm.table_T[0] <= Tp / s <= rm.table_T[1]) if rm.corr is not None else False,
                tau_grid=tau, lnG_grid=lnG_t)


if __name__ == "__main__":
    import time
    ls = json.load(open(paths.data("linescan_u1_mu0_1GeV.json")))
    out = {}
    t0 = time.time()
    for g in (0.44, 0.45, 0.46, 0.47, 0.475, 0.48, 0.49, 0.5, 0.52, 0.55, 0.6):
        rm = RateModel(g)
        r = percolate(rm)
        L = ls[f"{g:g}"]
        out[f"{g:g}"] = r
        print(f"g={g:.3f}: T_p {r['T_p']:.4e} (TL {L['Tperc']:.4e}, {100*(r['T_p']/L['Tperc']-1):+.1f}%) | "
              f"(b/H)_S {r['betaH_S']:.2f} (TL {L['betaH']:.2f}) | (b/H)_R {r['betaH_R']:.2f} (TL {L['betaH_RH']:.2f}) | "
              f"beta_G/H {r['beta_Gamma_H']:.2f} | dlnV/dt/H {r['dlnV_dt_over_H']:+.2f} | T_nuc/T_p {r['T_nuc']/r['T_p']:.2f} "
              f"| in table {r['in_table']} | {time.time()-t0:.0f} s", flush=True)
    json.dump(out, open(paths.work("validation_line.json"), "w"), indent=1)
