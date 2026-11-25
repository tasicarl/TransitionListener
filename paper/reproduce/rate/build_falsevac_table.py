"""The exact transition table at the central rate, with the false-vacuum fraction integrated on
the expansion rate of the FALSE vacuum instead of the self-consistent mean background.

Rate: variant 'nlo' of build_exact_table.py (S3 at NLO matching plus Delta S at mu4 x 1 plus
Delta S_dim6), built by its class Model; grids G and LOGV and the keys KEYS are that file's.

Background 'fvbg': in units of H_V = sqrt(Delta V/3)/M_Pl,
    H^2 = 1 + rho_plasma(T(a))/Delta V,
i.e. the expansion of a region still in the false vacuum: the vacuum energy Delta V is not diluted
by the conversion, and the radiation released by the converted regions is left out (the expansion
rate used in the completion criterion of Ellis, Lewicki, No, arXiv:1809.08242). The mean background
of the paper (mean_background.average_background) has instead
H^2 = P_f + rho_released/Delta V + rho_plasma/Delta V.
Time-temperature relation (as on the mean background, clock 'lna'): the plasma temperature at the
scale factor of this background, a ~ 1/(T g_s(T)^{1/3}); the rate is Gamma(T(a)) and
P_f = exp(-I), I = (4 pi/3) int dt' Gamma(t') a(t')^3 (eta(t) - eta(t'))^3, on the same background.
Every definition downstream is exact_percolation.percolate and build_exact_table.run_g unchanged:
T_p at P_t = f_perc = 0.28957; n_b, R_* = n_b^{-1/3}; (beta/H)_R = (8 pi/f_perc)^{1/3}/(R_* H(T_p))
with H of THIS background; T_f at P_t = 0.99; dlnV_dt_over_H = d ln(a^3 P_f)/dt / H at T_p;
T_nuc, T_1e, beta_Gamma_H, betaH_S, alpha, the window flags.

Integration window. percolate integrates the background in cosmic time t (units 1/H_V) from t_lo
to t_hi = tau_end, the e-folds of the lowest temperature of the rate table. On the mean background
H < H_V after percolation and ln a lags behind t, so the paper's window ends at ln a < tau_end. Here
H >= H_V, and the integration runs with the same step dt until ln a reaches tau_end (the point at
which ln a >= tau_end is not recorded, so the rate is never extrapolated past its table).

Implementation: mean_background.average_background is replaced, during the call of
build_exact_table.run_g, by `background` below, a copy of it whose only changes are marked
CHANGED; run_g runs with its variant list set to ['nlo'] (the output names it 'fvbg').
--background mean selects the copy's original behaviour (the reproduction check of
exact_table.npz).

Output (default <work>/falsevac_table.npz, or <work>/meanbg_check_table.npz with --background mean):
the layout of exact_table.npz, arrays (n_G, 1, n_LOGV) per key in KEYS, plus G, LOGV,
VARIANTS = ['fvbg'] (['nlo'] with --background mean), T_reh_SM and T_reh_SM4 (n_G, n_LOGV;
independent of the background). merge_tables.py adds the variant to the transition table.

Run: PYTHONPATH=<root>/src python build_falsevac_table.py [--background fvbg|mean] [--out PATH]
     [--g 0.45,0.5] [--logv 0.25,0.45] [--procs N]
"""
import argparse
import contextlib
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import paths  # noqa: E402  (src/, models/ and the folders of paper/reproduce on sys.path)
import exact_percolation as EP  # noqa: E402,F401
import build_exact_table as BET  # noqa: E402
import mean_background as C  # noqa: E402

VARIANT_IN = "nlo"           # the rate of build_exact_table.py
NAMES = {"fvbg": "fvbg", "mean": "nlo"}  # variant name written to the output, per background
_ORIG_BACKGROUND = C.average_background


def make_background(kind):
    """The background function for kind = 'fvbg' (false vacuum) or 'mean' (the original)."""
    if kind not in ("fvbg", "mean"):
        raise ValueError(kind)
    fv = kind == "fvbg"

    def background(lnG, t_lo, t_hi, n=6000, alpha=1e6, clock="t", rho_R_lo=None, rho_plasma=None):
        # a copy of mean_background.average_background; the only changes
        # are marked CHANGED. With fv = False it is the original line by line.
        if clock not in ("t", "lna"):
            raise ValueError(f"clock must be 't' or 'lna', not {clock!r}")
        if fv and (clock != "lna" or rho_plasma is None):                       # CHANGED
            raise ValueError("the false-vacuum background needs clock='lna' and rho_plasma")
        t = np.linspace(t_lo, t_hi, n); dt = t[1] - t[0]
        n_used = n
        if fv:                                                                   # CHANGED
            # the same step, continued past t_hi until ln a reaches t_hi (= tau_end) below
            n = 2 * n
            t = t_lo + dt * np.arange(n)
        a = np.empty(n); H = np.empty(n); eta = np.empty(n); F = np.empty(n)
        cn = np.arcsinh(np.sqrt(alpha)) / 2.0
        ai, etai, rhoR = np.exp(t_lo), 1.0 - np.exp(-t_lo), 1.0 / np.sinh(2.0 * (cn + t_lo)) ** 2
        if rho_R_lo is not None:
            rhoR = float(rho_R_lo)
        if rho_plasma is not None:
            if clock != "lna":
                raise ValueError("rho_plasma needs clock='lna'")
            rhoR = 0.0
        M0 = M1 = M2 = M3 = 0.0
        Fi = 1.0
        for i in range(n):
            if fv and np.log(ai) >= t_hi:                                        # CHANGED
                n_used = i                       # end of the rate table reached in ln a
                break
            a[i], eta[i], F[i] = ai, etai, Fi
            w = np.exp(lnG(t[i] if clock == "t" else np.log(ai))) * ai**3 * dt
            M0 += w
            plasma = 0.0 if rho_plasma is None else float(rho_plasma(np.log(ai)))
            if fv:                                                               # CHANGED
                Hi = np.sqrt(1.0 + plasma)       # false vacuum: Delta V undiluted, no released radiation
            else:
                Hi = np.sqrt(max(Fi + rhoR + plasma, 1e-300))
            H[i] = Hi
            ai += dt * Hi * ai
            d = dt / ai
            etai += d
            M3, M2, M1 = (M3 + 3.0 * d * M2 + 3.0 * d * d * M1 + d**3 * M0,
                          M2 + 2.0 * d * M1 + d * d * M0, M1 + d * M0)
            Nt = 4 * np.pi / 3 * M3
            Fn = np.exp(-max(Nt, 0.0))
            rhoR += -4.0 * Hi * rhoR * dt - (Fn - Fi)
            Fi = Fn
        else:
            if fv:                                                               # CHANGED
                raise RuntimeError("ln a did not reach the end of the rate table in 2 n steps")
        sl = slice(0, n_used)                                                    # CHANGED (fv only)
        t, a, H, eta, F = t[sl], a[sl], H[sl], eta[sl], F[sl]
        return dict(t=t, a=a, H=H, eta=eta, F=F, k_max=float(np.max(a * H)))

    return background


@contextlib.contextmanager
def patched(kind, logv=None):
    """build_exact_table.run_g on background `kind`, central rate only, at the nodes `logv`."""
    saved = (C.average_background, BET.VARIANTS, BET.LOGV)
    C.average_background = make_background(kind)
    BET.VARIANTS = [VARIANT_IN]
    if logv is not None:
        BET.LOGV = np.asarray(logv, float)
    try:
        yield
    finally:
        C.average_background, BET.VARIANTS, BET.LOGV = saved


def run_g(args):
    """(g, out, treh) of build_exact_table.run_g, with out[key] shaped (1, n_logv)."""
    g, kind, logv = args
    with patched(kind, logv):
        return BET.run_g(g)


def build(G, logv, kind, procs):
    with Pool(min(procs, len(G))) as pool:
        res = pool.map(run_g, [(g, kind, logv) for g in G])
    res.sort(key=lambda x: x[0])
    # as the __main__ block of build_exact_table.py, with the variant renamed
    tab = {key: np.array([r[1][key] for r in res]) for key in BET.KEYS}      # (g, 1, logv)
    tab.update(G=np.array(G), LOGV=np.asarray(logv, float), VARIANTS=np.array([NAMES[kind]]),
               T_reh_SM=np.array([r[2][0] for r in res]), T_reh_SM4=np.array([r[2][1] for r in res]))
    return tab


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--background", choices=("fvbg", "mean"), default="fvbg")
    ap.add_argument("--out", default=None)
    ap.add_argument("--g", default=None, help="comma-separated couplings (default: the grid of build_exact_table)")
    ap.add_argument("--logv", default=None, help="comma-separated log10(v/GeV) (default: the grid)")
    ap.add_argument("--procs", type=int, default=len(BET.G))
    o = ap.parse_args()
    G = [float(x) for x in o.g.split(",")] if o.g else list(BET.G)
    logv = np.array([float(x) for x in o.logv.split(",")]) if o.logv else np.array(BET.LOGV)
    out = o.out or paths.work("falsevac_table.npz" if o.background == "fvbg" else "meanbg_check_table.npz")
    t0 = time.time()
    tab = build(G, logv, o.background, o.procs)
    np.savez(out, **tab)
    for i, g in enumerate(G):
        print(f"g={g:.3f} [{o.background}] (b/H)_R vs log10 v: "
              + " ".join(f"{x:.2f}" for x in tab["betaH_R"][i, 0]), flush=True)
    print(f"wrote {out}, {time.time() - t0:.0f} s", flush=True)
