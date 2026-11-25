"""The mean expansion of the universe during a transition: the false-vacuum fraction on the background it creates.

average_background integrates, in units H_V = 1, the vacuum energy rho_V = F(t) of the false-vacuum fraction F, the
radiation the transition releases, d rho_R/dt = -4 H rho_R - dF/dt, and H^2 = rho_V + rho_R (plus the radiation of the
plasma when given), with F = exp(-I) from the bubbles nucleated on that same background. rate/exact_percolation.py
integrates the percolation of the paper on it.
"""
import numpy as np


def average_background(lnG, t_lo, t_hi, n=6000, alpha=1e6, clock="t", rho_R_lo=None, rho_plasma=None):
    """The mean expansion during and after the transition, in units H_V = 1 (as deltaPT's
    averageevolution): rho_V = F(t), d rho_R/dt = -4 H rho_R - dF/dt, H^2 = rho_V + rho_R,
    F = exp(-I) with I = int dt' Gamma a'^3 (4 pi/3) (eta - eta')^3 on the same background.
    alpha = rho_V / rho_R at t = 0; rho_R_lo, if given, is rho_R / rho_V at t_lo instead.
    clock = "t": lnG is a function of cosmic time (deltaPT's convention, a rate given in time).
    clock = "lna": lnG is a function of ln a on this background, normalised to ln a = t_lo at
    the start; this is the physical argument for a rate given as a function of temperature,
    T ~ 1/a, and differs from "t" once H departs from H_V.
    rho_plasma, if given (clock = "lna" only), is the radiation of the plasma rho/rho_V as a
    function of ln a, e.g. rho_rad(T(a)) with its changing degrees of freedom; rho_R then
    carries only the released energy, which starts at zero.
    Returns dict(t, a, H, eta, F, k_max = max aH).
    The cost is O(n): the moments M_k = sum_j w_j (eta - eta_j)^k, k = 0..3, of the bubbles
    nucleated so far are advanced by the binomial rule each step, with only positive terms.
    (Expanding the cube into sums of powers of eta instead cancels catastrophically once the
    window spans more than about ten e-folds, where eta saturates and the radii of recent bubbles
    are 1e-6 of it; slow transitions then percolate at random.)"""
    if clock not in ("t", "lna"):
        raise ValueError(f"clock must be 't' or 'lna', not {clock!r}")
    t = np.linspace(t_lo, t_hi, n); dt = t[1] - t[0]
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
        a[i], eta[i], F[i] = ai, etai, Fi
        w = np.exp(lnG(t[i] if clock == "t" else np.log(ai))) * ai**3 * dt
        M0 += w                                        # new bubbles, radius zero
        plasma = 0.0 if rho_plasma is None else float(rho_plasma(np.log(ai)))   # everywhere, inside bubbles too
        Hi = np.sqrt(max(Fi + rhoR + plasma, 1e-300)); H[i] = Hi
        ai += dt * Hi * ai
        d = dt / ai
        etai += d
        M3, M2, M1 = (M3 + 3.0 * d * M2 + 3.0 * d * d * M1 + d**3 * M0,
                      M2 + 2.0 * d * M1 + d * d * M0, M1 + d * M0)
        Nt = 4 * np.pi / 3 * M3
        Fn = np.exp(-max(Nt, 0.0))
        rhoR += -4.0 * Hi * rhoR * dt - (Fn - Fi)
        Fi = Fn
    return dict(t=t, a=a, H=H, eta=eta, F=F, k_max=float(np.max(a * H)))
