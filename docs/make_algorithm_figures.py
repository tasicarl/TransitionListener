#!/usr/bin/env python3
"""Build the figures used by the algorithm pages.

Run from the repository root with the source tree on the path::

    PYTHONPATH=$PWD/src python docs/make_algorithm_figures.py

Each figure is written to ``docs/source/_static/`` and is reproducible from this script alone,
so that a change in the code shows up in the figure rather than being quietly contradicted by
it. The models are the ones shipped with the package.

The figures follow the house style: sans-serif at 9 pt, ticks inward on all four sides, no
grid, frameless legends, and the DESY palette from ``transitionlistener.colors``.
"""
from __future__ import annotations

import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt

from transitionlistener.colors import (DESYcyan, DESYorange, DESYdunkelblau,
                                       DESYaubergine, DESYrot)

OUT = pathlib.Path(__file__).resolve().parent / "source" / "_static"
OUT.mkdir(parents=True, exist_ok=True)

#: Width of the documentation's text column, in inches.
WIDTH = 6.1


def use_house_style() -> None:
    mpl.rcParams.update({
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9, "legend.fontsize": 7,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.minor.visible": True, "ytick.minor.visible": True,
        "axes.grid": False, "axes.linewidth": 0.8,
        "font.family": "sans-serif", "mathtext.fontset": "dejavusans",
        "axes.unicode_minus": True,
    })


def figure_shooting() -> None:
    """What the shooting does: an undershoot, an overshoot and the solution."""
    from transitionlistener.tunneling1D import SingleFieldInstanton

    def V(phi):
        return 0.25 * phi ** 4 - 0.4 * phi ** 3 + 0.1 * phi ** 2

    def dV(phi):
        return phi * (phi - 0.2) * (phi - 1.0)

    inst = SingleFieldInstanton(1.0, 0.0, V, dV)
    solved = inst.findProfile()
    delta_phi = inst.phi_metaMin - inst.phi_absMin
    x_solved = -np.log(abs(solved.Phi[0] - inst.phi_absMin) / abs(delta_phi))
    rmax = solved.R[-1]

    def trial(x):
        """Integrate one trial until it over- or undershoots, and save that much of it.

        A trial cannot be integrated to a fixed outer radius: an overshoot runs past the false
        vacuum and the field then diverges. `integrateProfile` is what finds where the trial
        ends, and only that interval is saved.
        """
        delta_phi0 = np.exp(-x) * delta_phi
        r0, phi0, dphi0 = inst.initialConditions(delta_phi0, 1e-4, 1e-3 * delta_phi)
        y0 = np.array([phi0, dphi0])
        epsfrac = np.array([1.0, 1.0]) * 1e-8
        epsabs = np.abs(np.array([delta_phi, delta_phi / inst.rscale]) * 1e-8)
        rf, _, ctype = inst.integrateProfile(
            r0, y0, 1e-4, epsfrac, epsabs, 1e-12, 10 * rmax)
        R = np.linspace(r0, rf, 400)
        prof = inst.integrateAndSaveProfile(R, y0, 1e-4, epsfrac, epsabs, 1e-12)
        return prof.R, prof.Phi, ctype

    use_house_style()
    fig, ax = plt.subplots(figsize=(WIDTH, 0.58 * WIDTH))

    # phi(0) = phi_true + exp(-x) (phi_false - phi_true), so a larger x starts nearer the
    # true vacuum, with more energy, and overshoots.
    r_o, p_o, c_o = trial(x_solved * 1.30)
    r_u, p_u, c_u = trial(x_solved * 0.78)
    assert c_u == "undershoot" and c_o == "overshoot", (c_u, c_o)

    ax.plot(r_u, p_u, color=DESYorange, lw=1.4,
            label="undershoot: turns back before the false vacuum")
    ax.plot(r_o, p_o, color=DESYaubergine, lw=1.4,
            label="overshoot: passes the false vacuum")
    ax.plot(solved.R, solved.Phi, color=DESYcyan, lw=2.0, label="solution")
    ax.axhline(inst.phi_metaMin, color=DESYdunkelblau, lw=0.8, ls="--")

    ax.set_xlabel(r"radius $r$")
    ax.set_ylabel(r"field $\phi$")
    ax.set_xlim(0, rmax)
    ax.set_ylim(-0.12, 1.06)
    ax.text(0.04 * rmax, -0.075, "false vacuum", color=DESYdunkelblau, fontsize=7,
            ha="left", va="center")
    ax.legend(frameon=False, loc="upper right")
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.145, top=0.975)
    fig.savefig(OUT / "algorithm_shooting.png", dpi=200)
    plt.close(fig)
    print("wrote", OUT / "algorithm_shooting.png")


def figure_rate_and_support() -> None:
    """Why the support is placed on the rise of the rate."""
    T = np.linspace(0.20, 1.00, 2000)
    logG = -40.0 * (T - 0.46) ** 2 / 0.02 + 6.0
    P = 1.0 / (1.0 + np.exp((T - 0.455) / 0.006))

    uniform = np.linspace(0.20, 1.00, 12)
    adaptive = np.concatenate([np.array([0.20, 0.30, 0.38]),
                               np.linspace(0.43, 0.49, 7),
                               np.array([0.60, 0.80, 1.00])])

    use_house_style()
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(WIDTH, 0.80 * WIDTH), sharex=True)
    a1.plot(T, logG, color=DESYcyan, lw=1.6)
    a1.plot(uniform, np.interp(uniform, T, logG), "v", color=DESYorange, ms=5, ls="none")
    a1.plot(adaptive, np.interp(adaptive, T, logG), "o", color=DESYdunkelblau, ms=3.6, ls="none")
    a1.set_ylabel(r"$\log_{10}\,\Gamma/H^4$")
    a1.set_ylim(-32, 14)
    a1.set_xlim(T[0], T[-1])

    a2.plot(T, P, color=DESYcyan, lw=1.6)
    a2.plot(uniform, np.interp(uniform, T, P), "v", color=DESYorange, ms=5, ls="none",
            label="support spread over the range")
    a2.plot(adaptive, np.interp(adaptive, T, P), "o", color=DESYdunkelblau, ms=3.6, ls="none",
            label="support placed on the rise")
    a2.axhline(0.28957, color=DESYrot, lw=0.8, ls="--")
    a2.set_ylabel(r"converted fraction $P$")
    a2.set_xlabel(r"temperature $T$ (arbitrary units)")
    a2.set_ylim(-0.06, 1.30)
    a2.set_xlim(T[0], T[-1])
    a2.text(0.995, 0.255, r"$f_{\mathrm{perc}}$", color=DESYrot, fontsize=7,
            ha="right", va="top", transform=a2.get_yaxis_transform())
    a2.legend(frameon=False, loc="upper left")

    fig.subplots_adjust(left=0.125, right=0.985, bottom=0.11, top=0.985, hspace=0.07)
    fig.savefig(OUT / "algorithm_rate_and_support.png", dpi=200)
    plt.close(fig)
    print("wrote", OUT / "algorithm_rate_and_support.png")


def _check(fig, name: str) -> None:
    """Run the house text checker where it is importable, and say what it found."""
    try:
        import sys
        sys.path.insert(0, "/lhome/ific/t/tasicarl/AccurateAstro-SMBHB-PT-PTA")
        from smbhb.paper_style import check_text
    except Exception:
        return
    try:
        bad = check_text(fig)
    except Exception as exc:
        print("   check_text could not run on %s: %s" % (name, exc)); return
    print("   %s: %s" % (name, "no text overlaps" if not bad else bad))


def figure_images() -> None:
    """Why a traced phase has to be put on one image of the potential's symmetry."""
    use_house_style()
    fig, ax = plt.subplots(figsize=(WIDTH, 0.52 * WIDTH))

    # The broken minimum of the same potential as the figure of V against temperature, traced
    # in temperature. It is a FIRST-order branch: the field value at the critical temperature
    # is nonzero, the two phases coexist above it, and the branch ends at the spinodal
    # temperature where the barrier disappears, rather than going continuously to zero. A
    # branch that vanished at T_c would be a second-order transition, where there is no bounce
    # to compute and so nothing for this figure to be about.
    D, E, lam = 0.50, 0.30, 0.60
    T2c = E ** 2 / (D * lam)
    T_sp = np.sqrt(9.0 * E ** 2 / (8.0 * lam * D) / T2c)

    T = np.linspace(0.0, T_sp, 500)
    disc = np.clip(9.0 * E ** 2 - 8.0 * lam * D * T ** 2 * T2c, 0.0, None)
    broken = (3.0 * E + np.sqrt(disc)) / (2.0 * lam)

    ax.plot(T, broken, color=DESYcyan, lw=1.8)
    ax.plot(T, -broken, color=DESYcyan, lw=1.8, ls=(0, (4, 3)))
    ax.plot([T_sp], [broken[-1]], "o", color=DESYcyan, ms=4)
    ax.plot([T_sp], [-broken[-1]], "o", color=DESYcyan, ms=4, mfc="none")
    ax.axhline(0.0, color=DESYdunkelblau, lw=1.4)
    ax.axvline(1.0, color="0.75", lw=0.6, ls=":", zorder=0)

    ax.annotate("", xy=(0.55, 1.12), xytext=(0.55, 0.0),
                arrowprops=dict(arrowstyle="->", color=DESYorange, lw=1.6))
    ax.annotate("", xy=(0.55, -1.12), xytext=(0.55, 0.0),
                arrowprops=dict(arrowstyle="->", color=DESYaubergine, lw=1.6,
                                linestyle=(0, (3, 2))))

    ax.text(0.04, 1.40, "broken phase", color=DESYcyan, fontsize=7, va="center")
    ax.text(0.04, -1.40, "its mirror image", color=DESYcyan, fontsize=7, va="center")
    ax.text(0.04, 0.16, "symmetric phase", color=DESYdunkelblau, fontsize=7, va="center")
    ax.text(0.62, 0.70, "the channel", color=DESYorange, fontsize=7, va="center")
    ax.text(0.62, -0.70, "not a channel", color=DESYaubergine, fontsize=7, va="center")
    ax.text(0.985, 1.60, r"$T_{\rm c}$", color="0.45", fontsize=7, ha="right", va="top")

    ax.set_xlabel(r"temperature $T/T_{\rm c}$")
    ax.set_ylabel(r"field $\phi$")
    ax.set_xlim(0.0, 1.18)
    ax.set_ylim(-1.75, 1.75)
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.165, top=0.975)
    _check(fig, "algorithm_images")
    fig.savefig(OUT / "algorithm_images.png", dpi=200)
    plt.close(fig)
    print("wrote", OUT / "algorithm_images.png")


def figure_plasma_dof() -> None:
    """The counted degrees of freedom of the Standard Model bath against temperature."""
    from transitionlistener import thermodynamics as td

    # The dispatchers, not the two fits directly: each fit is valid on one side of the seam
    # at 0.12 GeV only, and below it the high-temperature one returns nonsense.
    T = np.logspace(-6.0, 4.0, 2000)
    g_rho = td.e_geffSM_fit(T, 1.0)
    g_s = td.s_geffSM_fit(T, 1.0)

    use_house_style()
    fig, ax = plt.subplots(figsize=(WIDTH, 0.52 * WIDTH))
    ax.plot(T, g_rho, color=DESYcyan, lw=1.6, label=r"$g_{\rm eff}$, energy density")
    ax.plot(T, g_s, color=DESYorange, lw=1.6, ls=(0, (4, 2)),
            label=r"$h_{\rm eff}$, entropy density")
    ax.set_xscale("log")
    ax.set_xlabel(r"temperature $T$ (GeV)")
    ax.set_ylabel("degrees of freedom")
    ax.axvline(0.12, color="0.75", lw=0.6, ls=":", zorder=0)
    ax.text(0.105, 96.0, "seam between the two fits", color="0.45", fontsize=7,
            ha="right", va="center")
    # The two counts differ by 0.48 at the coldest temperatures, which is 0.4 % of this
    # panel's range, so the inset is the only place it can be seen.
    inset = ax.inset_axes([0.085, 0.40, 0.26, 0.34])
    low = T < 3.0e-3
    inset.plot(T[low], g_rho[low], color=DESYcyan, lw=1.2)
    inset.plot(T[low], g_s[low], color=DESYorange, lw=1.2, ls=(0, (3, 1.6)))
    inset.set_xscale("log")
    inset.set_xlim(1e-6, 3e-3)
    inset.set_ylim(2.8, 11.5)
    inset.tick_params(labelsize=6, direction="in", top=True, right=True)
    inset.set_xticks([1e-6, 1e-4])
    inset.minorticks_off()
    ax.set_xlim(0.6 * T[0], 1.6 * T[-1])
    ax.set_ylim(0.0, 125.0)
    ax.legend(frameon=False, loc="lower right")
    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.165, top=0.975)
    _check(fig, "algorithm_plasma_dof")
    fig.savefig(OUT / "algorithm_plasma_dof.png", dpi=200)
    plt.close(fig)
    print("wrote", OUT / "algorithm_plasma_dof.png")


def figure_potential_vs_temperature() -> None:
    """How a barrier appears in the potential as the universe cools."""
    use_house_style()
    fig, ax = plt.subplots(figsize=(WIDTH, 0.55 * WIDTH))

    # A quartic with a cubic term and a thermal mass, which is the shape the one-loop thermal
    # corrections produce: at high temperature the only minimum is the symmetric one, a second
    # minimum appears, the two become degenerate, and below that the broken one is the deeper.
    # lam = 2E puts the degenerate minimum at phi = 1, which fixes the scale of the picture.
    phi = np.linspace(-0.30, 1.35, 800)
    D, E, lam = 0.50, 0.30, 0.60
    T2c = E ** 2 / (D * lam)
    shades = [1.12, 1.00, 0.92, 0.86]
    colors = [DESYorange, DESYaubergine, DESYcyan, DESYdunkelblau]
    labels = [r"$T = 1.12\,T_{\rm c}$", r"$T = T_{\rm c}$", r"$T = 0.92\,T_{\rm c}$",
              r"$T = 0.86\,T_{\rm c}$"]
    for frac, color, label in zip(shades, colors, labels):
        T2 = frac ** 2 * T2c
        V = D * T2 * phi ** 2 - E * phi ** 3 + 0.25 * lam * phi ** 4
        ax.plot(phi, V - V[np.argmin(abs(phi))], color=color, lw=1.5, label=label)

    ax.axhline(0.0, color="0.75", lw=0.6, zorder=0)
    ax.set_xlabel(r"field $\phi$")
    ax.set_ylabel(r"$V(\phi, T) - V(0, T)$")
    ax.set_xlim(phi[0], phi[-1])
    ax.set_ylim(-0.058, 0.030)
    ax.legend(frameon=False, loc="upper left", ncol=2)
    fig.subplots_adjust(left=0.125, right=0.985, bottom=0.155, top=0.975)
    _check(fig, "algorithm_potential_vs_temperature")
    fig.savefig(OUT / "algorithm_potential_vs_temperature.png", dpi=200)
    plt.close(fig)
    print("wrote", OUT / "algorithm_potential_vs_temperature.png")


def figure_fluid_profiles() -> None:
    """The fluid velocity around the wall, from the module's own flow equations."""
    from scipy.integrate import solve_ivp

    from transitionlistener.hydrodynamics import Hydrodynamics

    cs2 = 1.0 / 3.0
    cs = np.sqrt(cs2)

    def flow(v_wall: float, v_at_wall: float):
        """Integrate `Hydrodynamics.dfdv` outward from the wall until the profile ends.

        The integration runs in the fluid velocity, from its value at the wall down to zero,
        and returns the self-similar coordinate it reaches. It ends at the shock front for a
        deflagration and where the velocity vanishes, at the sound speed, for a detonation.
        """
        def deriv(v, X, c):
            return Hydrodynamics.dfdv(None, v, X, c)

        def at_shock(v, X, c):
            xi, _ = X
            return xi * (xi - v) / (1.0 - xi * v) - c

        at_shock.terminal = True
        sol = solve_ivp(deriv, (v_at_wall, 1e-20), [v_wall, 1.0], events=at_shock,
                        args=(cs2,), rtol=1e-10, atol=1e-10, dense_output=True)
        v = np.linspace(sol.t[0], sol.t[-1], 400)
        return sol.sol(v)[0], v

    use_house_style()
    fig, ax = plt.subplots(figsize=(WIDTH, 0.52 * WIDTH))

    # A deflagration: the fluid inside the bubble is at rest, the wall pushes the plasma ahead
    # of it, and the disturbance ends at a shock front where the velocity drops to zero.
    vw_d, v0_d = 0.45, 0.32
    xi_d, v_d = flow(vw_d, v0_d)
    ax.plot([0.0, vw_d], [0.0, 0.0], color=DESYcyan, lw=1.6)
    ax.plot([vw_d, vw_d], [0.0, v0_d], color=DESYcyan, lw=1.6)
    ax.plot(xi_d, v_d, color=DESYcyan, lw=1.6, label=r"deflagration, $v_w = 0.45$")
    ax.plot([xi_d[-1], xi_d[-1]], [v_d[-1], 0.0], color=DESYcyan, lw=1.6)
    ax.plot([xi_d[-1], 1.0], [0.0, 0.0], color=DESYcyan, lw=1.6)

    # A detonation: the wall outruns any signal, so the plasma ahead of it is undisturbed, and
    # behind it a rarefaction carries the velocity back to zero at the speed of sound.
    vw_t, v0_t = 0.80, 0.22
    xi_t, v_t = flow(vw_t, v0_t)
    ax.plot([0.0, xi_t[-1]], [0.0, 0.0], color=DESYorange, lw=1.6, ls=(0, (4, 2)))
    ax.plot(xi_t, v_t, color=DESYorange, lw=1.6, ls=(0, (4, 2)),
            label=r"detonation, $v_w = 0.80$")
    ax.plot([vw_t, vw_t], [v0_t, 0.0], color=DESYorange, lw=1.6, ls=(0, (4, 2)))
    ax.plot([vw_t, 1.0], [0.0, 0.0], color=DESYorange, lw=1.6, ls=(0, (4, 2)))

    ax.axvline(cs, color=DESYdunkelblau, lw=0.8, ls=":")
    ax.text(cs - 0.015, 0.370, r"$c_s$", color=DESYdunkelblau, fontsize=7,
            ha="right", va="center")
    ax.annotate("shock front", xy=(xi_d[-1] - 0.004, 0.045), xytext=(0.300, 0.070),
                color=DESYcyan, fontsize=7, ha="center", va="center",
                arrowprops=dict(arrowstyle="->", color=DESYcyan, lw=0.8))
    ax.text(vw_d - 0.012, 0.325, "wall", color=DESYcyan, fontsize=7, ha="right", va="bottom")
    ax.text(vw_t - 0.012, 0.224, "wall", color=DESYorange, fontsize=7, ha="right",
            va="bottom")
    ax.set_xlabel(r"self-similar coordinate $\xi = r/t$")
    ax.set_ylabel(r"fluid velocity $v(\xi)$")
    ax.set_xlim(-0.015, 1.0)
    ax.set_ylim(-0.014, 0.40)
    ax.legend(frameon=False, loc="upper left")
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.165, top=0.975)
    _check(fig, "algorithm_fluid_profiles")
    fig.savefig(OUT / "algorithm_fluid_profiles.png", dpi=200)
    plt.close(fig)
    print("wrote", OUT / "algorithm_fluid_profiles.png")


def figure_spectrum_sources() -> None:
    """The three sources and their sum, from the code's own templates."""
    from transitionlistener.gwfopt import FOPTspectrum

    from transitionlistener.hydrodynamics import kappa_sw

    # The sound-wave efficiency is the package's own fit at this strength and wall velocity,
    # not a number chosen for the picture: passing 0.35 where the fit gives 0.549 draws the
    # amplitude a factor two low. The turbulent fraction is the usual tenth of it. The
    # collision efficiency is set by hand, because for a wall the plasma holds back it is
    # zero and the template would have nothing to draw.
    alpha, v_wall = 0.2, 0.65
    cs = 1.0 / np.sqrt(3.0)
    k_sw = kappa_sw(alpha, v_wall, cs)
    params = dict(alpha=alpha, RH=0.05, Treh_SM_GeV=100.0, g_eff_tot_reh=100.0,
                  h_eff_tot_reh=100.0, kappa_phi=0.02, kappa_sw=k_sw,
                  kappa_turb=0.1 * k_sw, v_wall=v_wall, c_s=cs)
    spec = FOPTspectrum(params)

    f = np.logspace(-5.0, 0.0, 600)
    parts = [("Sound waves", "sound waves", DESYcyan, "-"),
             ("Turbulence", "turbulence", DESYorange, (0, (4, 2))),
             ("Wall collisions", "bubble collisions", DESYaubergine, (0, (1, 1.6)))]

    use_house_style()
    fig, ax = plt.subplots(figsize=(WIDTH, 0.55 * WIDTH))
    total = np.array([spec.h2Omega_0_sum(x, "all") for x in f])
    ax.plot(f, total, color=DESYdunkelblau, lw=2.0, label="sum")
    for key, label, color, ls in parts:
        y = np.array([spec.h2Omega_0_sum(x, key) for x in f])
        ax.plot(f, y, color=color, lw=1.4, ls=ls, label=label)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"frequency $f$ (Hz)")
    ax.set_ylabel(r"$h^2\Omega_{\rm GW}$")
    ax.set_xlim(0.6 * f[0], 1.8 * f[-1])
    ax.set_ylim(3e-19, 3e-9)
    ax.legend(frameon=False, loc="upper left", ncol=1)
    fig.subplots_adjust(left=0.115, right=0.985, bottom=0.155, top=0.975)
    _check(fig, "algorithm_spectrum_sources")
    fig.savefig(OUT / "algorithm_spectrum_sources.png", dpi=200)
    plt.close(fig)
    print("wrote", OUT / "algorithm_spectrum_sources.png")


def main() -> int:
    figure_shooting()
    figure_rate_and_support()
    figure_images()
    figure_potential_vs_temperature()
    figure_plasma_dof()
    figure_fluid_profiles()
    figure_spectrum_sources()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
