"""The nucleation history of the fermion-free conformal U(1) of the paper, as
ln(Gamma/H_V^4) against e-folds tau, and the helpers that the other scripts of rate/
share: the line-scan point of a coupling (point_dir, in paths.LINESCAN_POINTS), the
vacuum energy, the radiation and entropy degrees of freedom, and the constants.

Model and scheme as in the paper's line scan (linescan/scan_point.py --model u1):
TL_conformal_u1_no_fermions_3d with input_scheme = zero_quartic (lambda(mu0) = 0 at
mu0 = g v e^{-1/6} = 1 GeV), mu3 = g T, NLO matching at mu4 = 7.0553 T, and

    Gamma(T) = T^4 (S/2 pi)^{3/2} e^{-S},   S = S3_LO(T) + Delta S(T),

with Delta S the point's NLO table table_mu1.json (Langer prefactor included).
The vacuum energy is the model's zero-temperature potential difference; the
radiation is the SM table plus the 4 dark degrees of freedom. During supercooling
a ~ 1/(T g_s^{1/3}), so tau(T) = ln[T0 g_s(T0)^{1/3} / (T g_s(T)^{1/3})] with
tau = 0 where Gamma = H_V^4.

Usage: python rate_history_u1.py --g 0.5 [--variants nlo,lo]
Writes <work>/rate_tables/rate_u1_g<g>_<variant>.dat and <work>/history_u1_g<g>.json; the LO
actions are cached in <work>/lo_actions/lo_u1_g<g>.json.
"""
import argparse
import glob
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
import dr_conformal_u1_no_fermions_matching as dr  # noqa: E402

MPL = 2.435e18
MU4 = 7.0553
MU0 = 1.0
DARK_DOF = 4.0
SCAN = paths.LINESCAN_POINTS

_tab = np.genfromtxt(os.path.join(ROOT, "src", "transitionlistener", "tab_data", "geff_SM.dat"))
_lx = np.log(_tab[:, 0])


def g_rho(T):
    return np.interp(np.log(T), _lx, _tab[:, 1]) + DARK_DOF


def g_s(T):
    return np.interp(np.log(T), _lx, _tab[:, 3]) + DARK_DOF


def rho_rad(T):
    return math.pi**2 / 30.0 * g_rho(T) * T**4


def point_dir(g):
    """The line-scan point of g. Only a coupling with three decimals has one: g = 0.4001 must not read the point of
    g = 0.400, whose name has the same three decimals."""
    if abs(round(g, 3) - g) > 1e-9:
        raise FileNotFoundError(f"no line-scan point for g = {g}")
    hits = glob.glob(os.path.join(SCAN, f"g{g:.3f}_u1_v*"))
    if not hits:
        raise FileNotFoundError(f"no line-scan point for g = {g}")
    return hits[0]


def vacuum_energy(g, v):
    """Zero-temperature potential difference [GeV^4] of the TL model."""
    import TL_conformal_u1_no_fermions_3d as m3d
    m = m3d.specific_potential({"g": g, "v_GeV": v}, verbose=False)
    cf = m.conversionFactor
    phis = np.linspace(1e-6, 1.5 * v / cf, 3001)[:, None]
    Vs = np.asarray(m.Vtot(phis, 1e-6, include_radiation=False)).ravel()
    V0 = float(np.asarray(m.Vtot(np.array([[1e-9]]), 1e-6, include_radiation=False)).ravel()[0])
    return float((V0 - np.nanmin(Vs)) * cf**4)


def S3_LO(g, T, mult=1.0):
    mu0, lam0 = MU0, 0.0
    # mu3 = g T with the input coupling g = g(mu0), as in the Delta S tables
    p = dr.matched_soft_parameters(g, lam0, mu0, T, mu4_over_T=MU4 * mult, matching_order="NLO", mu3=g * T)
    try:
        S3 = abelian_higgs_3d.solve_LO_bounce(p["m3sq"], p["lam3"], p["g3sq"], m_debye_sq=p["mD2"])[0]
        return float(S3)
    except (ValueError, RuntimeError):
        return float("nan")


def history(g, variants, n_T=140, t_lo=-6.0, t_hi=6.0, n_t=2401):
    pdir = point_dir(g)
    summ = json.load(open(os.path.join(pdir, "summary.json")))
    v = float(summ["v_GeV"])
    dV = vacuum_energy(g, v)
    HV = math.sqrt(dV / 3.0) / MPL
    anchor = float(summ["anchor_GeV"])
    Ts = np.geomspace(anchor * 0.05, anchor * 8.0, n_T)
    cache = paths.work("lo_actions", f"lo_u1_g{g:.3f}.json")
    done = json.load(open(cache)) if os.path.exists(cache) else {}
    S3 = []
    for T in Ts:
        key = f"{T:.8e}"
        if key not in done:
            done[key] = S3_LO(g, T)
        S3.append(done[key])
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    json.dump(done, open(cache, "w"))
    S3 = np.array(S3, float)
    tabj = json.load(open(os.path.join(pdir, "table_mu1.json")))["table"]
    corr = abelian_higgs_3d.NLOCorrection(tabj["T_GeV"], tabj["delta_S"], S3_LO=tabj.get("S3_LO"))
    out = dict(g=g, v_GeV=v, DeltaV_GeV4=dV, H_V_GeV=HV, anchor_GeV=anchor, variants={},
               TL=json.load(open(os.path.join(pdir, "nlo3d_mu1.json")))["observables"])
    os.makedirs(paths.work("rate_tables"), exist_ok=True)
    for variant in variants:
        dS = np.array([float(corr(T, S3_LO=s)) if (variant == "nlo" and np.isfinite(s)) else 0.0
                       for T, s in zip(Ts, S3)])
        S = S3 + dS
        lnG = 4 * np.log(Ts) + 1.5 * np.log(np.where(S > 0, S, np.nan) / (2 * math.pi)) - S - 4 * math.log(HV)
        ok = np.isfinite(lnG)
        order = np.argsort(Ts)[::-1]
        T0 = None
        for a, b in zip(order[:-1], order[1:]):
            if ok[a] and ok[b] and lnG[a] < 0 <= lnG[b]:
                w = -lnG[a] / (lnG[b] - lnG[a])
                T0 = float(np.exp(np.log(Ts[a]) + w * (np.log(Ts[b]) - np.log(Ts[a]))))
                break
        if T0 is None:
            out["variants"][variant] = dict(note="Gamma never reaches H_V^4")
            continue
        t_of_T = np.log(T0 * g_s(T0) ** (1 / 3) / (Ts * g_s(Ts) ** (1 / 3)))
        srt = np.argsort(t_of_T[ok])
        tt, ll = t_of_T[ok][srt], lnG[ok][srt]
        tgrid = np.linspace(t_lo, t_hi, n_t)
        lgrid = np.interp(tgrid, tt, ll)
        lo_s = (ll[1] - ll[0]) / (tt[1] - tt[0]); hi_s = (ll[-1] - ll[-2]) / (tt[-1] - tt[-2])
        lgrid = np.where(tgrid < tt[0], ll[0] + lo_s * (tgrid - tt[0]), lgrid)
        lgrid = np.where(tgrid > tt[-1], ll[-1] + hi_s * (tgrid - tt[-1]), lgrid)
        path = paths.work("rate_tables", f"rate_u1_g{g:.3f}_{variant}.dat")
        np.savetxt(path, np.column_stack([tgrid, lgrid]),
                   header=f"g {g} v_GeV {v} variant {variant} T0_GeV {T0:.10e} H_V_GeV {HV:.10e} "
                          f"alpha_T0 {dV / rho_rad(T0):.6e}\ncolumns: tau (e-folds, 0 at Gamma = H_V^4), ln(Gamma/H_V^4)")
        beta = float(np.interp(0.0, 0.5 * (tgrid[1:] + tgrid[:-1]), np.diff(lgrid) / np.diff(tgrid)))
        out["variants"][variant] = dict(T0_GeV=T0, alpha_T0=dV / rho_rad(T0), beta_H_at_T0=beta, table=path,
                                        t_range_computed=[float(tt[0]), float(tt[-1])])
        print(f"g={g:.3f} {variant}: T0={T0:.4e} GeV beta/H(t=0)={beta:.2f} computed t in [{tt[0]:.2f},{tt[-1]:.2f}]",
              flush=True)
    json.dump(out, open(paths.work(f"history_u1_g{g:.3f}.json"), "w"), indent=1)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--g", type=float, required=True)
    ap.add_argument("--variants", default="nlo,lo")
    a = ap.parse_args()
    history(a.g, a.variants.split(","))
