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
                                       DESYaubergine, DESYrot, DESYmagenta)

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
    from scipy.integrate import solve_ivp

    from transitionlistener.tunneling1D import SingleFieldInstanton

    def V(phi):
        return 0.25 * phi ** 4 - 0.4 * phi ** 3 + 0.1 * phi ** 2

    def dV(phi):
        return phi * (phi - 0.2) * (phi - 1.0)

    inst = SingleFieldInstanton(1.0, 0.0, V, dV)
    # Solved at a tight tolerance, which is what carries the bounce out to a radius where it
    # has genuinely reached the false vacuum rather than stopping in the middle of the panel.
    # The bounce is a separatrix, so continuing it by hand past where the solver stops only
    # shows the drift of whatever integrator is doing the continuing; the tolerance is the
    # real control. At the default tolerance it ends at r = 24 with phi = 7e-5, here at
    # r = 42 with phi = 1e-23.
    solved = inst.findProfile(rmax=1e5, phitol=1e-10, xtol=1e-12)
    delta_phi = inst.phi_metaMin - inst.phi_absMin
    x_solved = -np.log(abs(solved.Phi[0] - inst.phi_absMin) / abs(delta_phi))
    rmax = solved.R[-1]

    def trial(x, r_end):
        """One trial, integrated past the radius at which its verdict is already settled.

        `integrateProfile` stops as soon as the trial has over- or undershot, which is all the
        shooting needs. Carried further, the same equation shows what each failure does: an
        overshoot crosses the false vacuum and runs away, and an undershoot turns back and
        oscillates about the top of the barrier, damped by the friction term.
        """
        delta_phi0 = np.exp(-x) * delta_phi
        r0, phi0, dphi0 = inst.initialConditions(delta_phi0, 1e-4, 1e-3 * delta_phi)
        epsfrac = np.array([1.0, 1.0]) * 1e-8
        epsabs = np.abs(np.array([delta_phi, delta_phi / inst.rscale]) * 1e-8)
        r_verdict, _, ctype = inst.integrateProfile(
            r0, np.array([phi0, dphi0]), 1e-4, epsfrac, epsabs, 1e-12, 10 * rmax)

        def rhs(r, y):
            return [y[1], dV(y[0]) - 2.0 * y[1] / r]

        sol = solve_ivp(rhs, (r0, r_end), [phi0, dphi0], rtol=1e-10, atol=1e-12,
                        dense_output=True, max_step=0.05)
        R = np.linspace(r0, sol.t[-1], 2000)
        return R, sol.sol(R)[0], r_verdict, float(sol.sol(r_verdict)[0]), ctype

    use_house_style()
    fig, ax = plt.subplots(figsize=(WIDTH, 0.62 * WIDTH))

    # phi(0) = phi_true + exp(-x) (phi_false - phi_true), so a larger x starts nearer the
    # true vacuum, with more energy, and overshoots.
    r_o, p_o, rv_o, pv_o, c_o = trial(x_solved * 1.30, 14.2)
    r_u, p_u, rv_u, pv_u, c_u = trial(x_solved * 0.78, 44.0)
    assert c_u == "undershoot" and c_o == "overshoot", (c_u, c_o)

    ax.plot(r_u, p_u, color=DESYorange, lw=1.4,
            label="too far from the true vacuum: turns back")
    ax.plot(r_o, p_o, color=DESYaubergine, lw=1.4,
            label="too near the true vacuum: runs past")
    assert abs(solved.Phi[-1] - inst.phi_metaMin) < 1e-6 * abs(delta_phi), solved.Phi[-1]
    ax.plot(solved.R, solved.Phi, color=DESYcyan, lw=2.2, label="the bounce")
    ax.plot([rv_u], [pv_u], "o", color=DESYorange, ms=4)
    ax.plot([rv_o], [pv_o], "o", color=DESYaubergine, ms=4)

    ax.axhline(inst.phi_metaMin, color=DESYdunkelblau, lw=1.0, ls="--")
    ax.axhline(0.2, color="0.75", lw=0.6, ls=":", zorder=0)

    ax.set_xlabel(r"radius $r$")
    ax.set_ylabel(r"field $\phi$")
    ax.set_xlim(0, 44.0)
    ax.set_ylim(-0.30, 0.98)
    ax.text(43.2, 0.028, "false vacuum", color=DESYdunkelblau, fontsize=7,
            ha="right", va="bottom")
    ax.text(43.2, 0.235, "top of the barrier", color="0.45", fontsize=7,
            ha="right", va="bottom")

    ax.annotate("the verdict is settled here,\nand the shooting stops",
                xy=(rv_o, pv_o - 0.012), xytext=(24.0, -0.150), color="0.35", fontsize=7,
                ha="center", va="center",
                arrowprops=dict(arrowstyle="->", color="0.45", lw=0.8))
    ax.annotate("", xy=(rv_u, pv_u - 0.012), xytext=(24.0, -0.115),
                arrowprops=dict(arrowstyle="->", color="0.45", lw=0.8))
    ax.legend(frameon=False, loc="upper right")
    fig.subplots_adjust(left=0.100, right=0.985, bottom=0.135, top=0.975)
    _check(fig, "algorithm_shooting")
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
    a2.set_ylabel(r"true-vacuum fraction $P_\mathrm{t}$")
    a2.set_xlabel(r"temperature $T$ (arbitrary units)")
    a2.set_ylim(-0.06, 1.30)
    a2.set_xlim(T[0], T[-1])
    a2.text(0.995, 0.255, r"$f_\mathrm{perc}$", color=DESYrot, fontsize=7,
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
    ax.axhline(0.0, color=DESYdunkelblau, lw=1.8)
    ax.axvline(1.0, color="0.75", lw=0.6, ls=":", zorder=0)

    # The arrows run from the symmetric phase to each branch and touch both, so that what they
    # connect is not left to the reader.
    T_arrow = 0.55
    phi_arrow = float(np.interp(T_arrow, T, broken))
    ax.annotate("", xy=(T_arrow, phi_arrow), xytext=(T_arrow, 0.0),
                arrowprops=dict(arrowstyle="->", color=DESYorange, lw=1.6,
                                shrinkA=0, shrinkB=0))
    ax.annotate("", xy=(T_arrow, -phi_arrow), xytext=(T_arrow, 0.0),
                arrowprops=dict(arrowstyle="->", color=DESYaubergine, lw=1.6,
                                linestyle=(0, (3, 2)), shrinkA=0, shrinkB=0))

    # Every label is placed in empty space with room around it, so that nothing crosses the
    # box a reader's eye draws around the words.
    ax.text(0.10, 1.88, "the broken phase, as traced", color=DESYcyan, fontsize=7,
            ha="left", va="center")
    ax.text(0.10, -1.88, "its mirror image, the same physical state", color=DESYcyan,
            fontsize=7, ha="left", va="center")
    ax.text(0.10, 0.30, "the symmetric phase", color=DESYdunkelblau, fontsize=7,
            ha="left", va="center")
    ax.text(T_arrow + 0.045, 0.80, "the bubble tunnels\nalong this path", color=DESYorange,
            fontsize=7, ha="left", va="center")
    ax.text(T_arrow + 0.045, -0.80, "a path to the other image\nis not a tunnelling channel",
            color=DESYaubergine, fontsize=7, ha="left", va="center")
    ax.text(0.975, -2.22, r"$T_\mathrm{c}$", color="0.45", fontsize=7, ha="right",
            va="center")
    ax.annotate("the barrier is gone here", xy=(T_sp + 0.008, broken[-1]),
                xytext=(0.62, 2.22), color=DESYcyan, fontsize=7, ha="center", va="center",
                arrowprops=dict(arrowstyle="->", color=DESYcyan, lw=0.8))

    ax.set_xlabel(r"temperature $T/T_{\rm c}$")
    ax.set_ylabel(r"field $\phi$")
    ax.set_xlim(0.0, 1.30)
    ax.set_ylim(-2.55, 2.55)
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.165, top=0.975)
    _check(fig, "algorithm_images")
    fig.savefig(OUT / "algorithm_images.png", dpi=200)
    plt.close(fig)
    print("wrote", OUT / "algorithm_images.png")


def _two_higgs_doublet_model():
    """The 2HDM at the benchmark of ``examples/bp_lisa.yaml``, built from the shipped model."""
    import importlib.util
    import sys

    path = pathlib.Path(__file__).resolve().parent.parent / "models" / "TL_2HDM.py"
    spec = importlib.util.spec_from_file_location("_alg_fig_2hdm", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_alg_fig_2hdm"] = mod
    spec.loader.exec_module(mod)
    params = dict(lambda1=0.28894, lambda2=0.26237, lambda3=5.80958300958301,
                  lambda4=-2.17494, lambda5=-2.24170, m12_sq_GeV2=1573.171,
                  tan_beta=21.3949, yukawa_type=1, v_GeV=246.22)
    return mod.R2HDM(params, verbose=False)


def figure_plasma_dof() -> None:
    """The tabulated bath, and the whole plasma of a model in each of its two phases."""
    from scipy.optimize import brentq

    from transitionlistener import thermodynamics as td
    from transitionlistener.bubbledynamics import g_eff_DS

    # The dispatchers, not the two fits directly: each fit is valid on one side of the seam at
    # 0.12 GeV only, and below it the high-temperature one returns nonsense.
    T = np.logspace(-3.0, 4.0, 1400)
    g_rho = td.e_geffSM_fit(T, 1.0)
    g_s = td.s_geffSM_fit(T, 1.0)

    pot = _two_higgs_doublet_model()

    class _FixedPhase:
        """A phase that sits at one field value, which is all ``g_eff_DS`` reads."""

        def __init__(self, X):
            self.X = np.asarray(X, dtype=float)

        def valAt(self, T):
            return self.X

    # The whole plasma, not the potential's fields alone: the tabulated bath, less the fields
    # of the potential that are Standard Model ones and would otherwise be counted twice, plus
    # the potential's own fields at the field value of the phase. This is what the expansion
    # rate is built from, and it is on the same scale as the panel above.
    origin = np.zeros(pot.Ndim)
    symmetric = _FixedPhase(origin)
    T2 = np.logspace(-3.0, 4.0, 220)
    X = np.asarray(pot.X0, dtype=float)
    tot_sym, tot_bro, stable = [], [], []
    for t in T2:
        t = float(t)
        bath = float(td.e_geffSM_fit(t, 1.0))
        overlap = float(np.squeeze(td.sm_fields_in_potential_geff(pot, t, "e")))
        X = np.asarray(pot.findMinimum(X, t), dtype=float)
        tot_bro.append(bath - overlap + g_eff_DS(t, pot, _FixedPhase(X)))
        # Whether the origin is a minimum, and whether that verdict means anything. `d2V` is a
        # finite difference of the full potential, which at high temperature is dominated by
        # the field-independent radiation and is differenced away: the result is then quantised
        # to the spacing of doubles at |V| divided by the step squared, and its sign is an
        # accident of the last bits of T. Above about 1 TeV here the eigenvalue is at that
        # floor, so the verdict is not drawn rather than drawn wrong.
        hess = np.atleast_2d(np.squeeze(pot.d2V(origin, t)))
        eig_min = float(np.linalg.eigvalsh(hess).min())
        floor = float(np.spacing(abs(float(np.squeeze(pot.Vtot(origin, t)))))) / pot.x_eps ** 2
        stable.append(eig_min > 0.0 and abs(eig_min) > 30.0 * floor)
        tot_sym.append(bath - overlap + g_eff_DS(t, pot, symmetric))
    tot_sym = np.array(tot_sym)
    tot_bro = np.array(tot_bro)

    # The symmetric phase is drawn on the one contiguous stretch that carries the transition,
    # near 100 GeV. At this benchmark the origin also stops being a minimum in a band near
    # 30 GeV and is one again below it; that lower piece is a different branch, and the
    # universe never occupies it.
    stable = np.array(stable)
    keep = np.zeros_like(stable)
    i0 = int(np.argmin(abs(T2 - 100.0)))
    if stable[i0]:
        lo = i0
        while lo > 0 and stable[lo - 1]:
            lo -= 1
        hi = i0
        while hi < len(stable) - 1 and stable[hi + 1]:
            hi += 1
        keep[lo:hi + 1] = True
    tot_sym[~keep] = np.nan

    # The temperature at which each mode's contribution has fallen to half of its relativistic
    # value, solved from the same integral the counting uses rather than taken as m = T.
    # Modes whose masses are within a few per cent of each other leave together and share one
    # label, because two arrows a pixel apart say nothing a reader can use.
    groups = [(r"W,\,Z", [(80.4, "b"), (91.2, "b")]),
              ("h", [(125.0, "b")]),
              ("t", [(172.5, "f")]),
              ("H", [(184.4, "b")]),
              (r"A,\,H^\pm", [(411.9, "b"), (409.4, "b")])]
    half = []
    for label, modes in groups:
        roots = []
        for mass, kind in modes:
            rel = td.e_geff(0.0, 1.0, 1.0, kind)
            roots.append(brentq(lambda t: td.e_geff(mass, t, 1.0, kind) - 0.5 * rel,
                                mass / 50.0, mass * 50.0))
        half.append((label, float(np.mean(roots))))

    use_house_style()
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(WIDTH, 0.98 * WIDTH), sharex=True)

    a1.plot(T, g_rho, color=DESYcyan, lw=1.6, label=r"$g_{\rm eff}$, energy density")
    a1.plot(T, g_s, color=DESYorange, lw=1.6, ls=(0, (4, 2)),
            label=r"$h_{\rm eff}$, entropy density")
    a1.axvline(0.12, color="0.75", lw=0.6, ls=":", zorder=0)
    a1.text(0.105, 36.0, "seam between\nthe two fits", color="0.45", fontsize=6.5,
            ha="right", va="center")
    a1.set_ylabel("degrees of freedom")
    a1.set_ylim(0.0, 125.0)
    a1.legend(frameon=False, loc="upper left")
    a1.set_title("the tabulated Standard Model bath on its own", fontsize=8, pad=4)

    a2.plot(T2, tot_bro, color=DESYdunkelblau, lw=1.6, label="broken phase")
    a2.plot(T2, tot_sym, color=DESYmagenta, lw=1.6, ls=(0, (4, 2)),
            label="symmetric phase")
    # Staggered in height, so that five labels fit into the decade the masses occupy.
    for i, (label, t_half) in enumerate(half):
        y_curve = float(np.interp(t_half, T2, tot_bro))
        y_text = 148.0 if i % 2 == 0 else 136.0
        a2.annotate("", xy=(t_half, y_curve + 1.5), xytext=(t_half, y_text - 3.0),
                    arrowprops=dict(arrowstyle="->", color="0.60", lw=0.7,
                                    shrinkA=0, shrinkB=0))
        a2.text(t_half, y_text, "$%s$" % label, color="0.40", fontsize=6.5,
                ha="center", va="center")
    a2.text(0.985, 0.955, "each mode has lost half\nof its contribution here",
            transform=a2.transAxes, color="0.40", fontsize=6.5, ha="right", va="top")
    a2.set_xscale("log")
    a2.set_xlabel(r"temperature $T$ (GeV)")
    a2.set_ylabel("degrees of freedom")
    a2.set_xlim(0.7 * T[0], 1.4 * T[-1])
    a2.set_ylim(0.0, 162.0)
    a2.legend(frameon=False, loc="lower right")
    a2.set_title("the whole plasma of the 2HDM, in each of its two phases", fontsize=8, pad=4)

    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.090, top=0.950, hspace=0.33)
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
    # Hot to cold: red, orange, cyan, dark blue.
    colors = [DESYrot, DESYorange, DESYcyan, DESYdunkelblau]
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
    ax.set_ylim(-0.058, 0.052)
    ax.legend(frameon=False, loc="upper left", ncol=2)
    fig.subplots_adjust(left=0.125, right=0.985, bottom=0.155, top=0.975)
    _check(fig, "algorithm_potential_vs_temperature")
    fig.savefig(OUT / "algorithm_potential_vs_temperature.png", dpi=200)
    plt.close(fig)
    print("wrote", OUT / "algorithm_potential_vs_temperature.png")


def figure_fluid_profiles() -> None:
    """The fluid velocity around the wall, from the module's own flow equations."""
    from scipy.integrate import solve_ivp
    from scipy.optimize import brentq

    from transitionlistener.hydrodynamics import Hydrodynamics

    cs2 = 1.0 / 3.0
    cs = np.sqrt(cs2)
    # One strength for both curves, so that the two pictures are the same transition seen at
    # two wall velocities and not two different transitions. At 0.3 the Jouguet velocity is
    # 0.858, so a wall at 0.40 is a clean subsonic deflagration and one at 0.90 a clean
    # detonation, with neither of them the hybrid that lives between the two. It is also
    # strong enough that the shock sits four per cent outside the sound speed rather than on
    # top of it, which a weaker transition would not show.
    alpha = 0.3
    v_jouguet = float(Hydrodynamics.find_vJ(None, alpha, cs2))

    def velocity_at_wall(v_wall, branch):
        """The fluid velocity just at the wall, in the plasma frame.

        Taken from the module's own junction condition rather than chosen to make the picture
        look right. For a deflagration the plasma behind the wall is at rest, so the wall-frame
        velocity behind it is the wall velocity itself; for a detonation the plasma ahead is at
        rest, so it is the wall-frame velocity in front that equals the wall velocity.
        """
        if branch < 0:
            v_minus = v_wall
            v_plus = float(Hydrodynamics.get_vp(None, v_minus, alpha, cs2, branch=-1))
        else:
            v_plus = v_wall
            # Bracketed from the sound speed up: the detonation branch is single valued
            # there, and below it the same expression runs off to one.
            v_minus = float(brentq(
                lambda vm: Hydrodynamics.get_vp(None, vm, alpha, cs2, branch=+1) - v_plus,
                cs, 0.9999))
        other = v_plus if branch < 0 else v_minus
        return abs((v_wall - other) / (1.0 - v_wall * other))

    def flow(v_wall, v_at_wall):
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
                        args=(cs2,), rtol=1e-10, atol=1e-10)
        return sol.y[0], sol.t

    use_house_style()
    fig, ax = plt.subplots(figsize=(WIDTH, 0.56 * WIDTH))

    # A deflagration: the wall is slower than sound, the plasma inside the bubble is at rest,
    # the wall pushes the plasma ahead of it, and the disturbance ends at a shock front.
    vw_d = 0.40
    v0_d = velocity_at_wall(vw_d, -1)
    xi_d, v_d = flow(vw_d, v0_d)
    ax.plot([0.0, vw_d], [0.0, 0.0], color=DESYcyan, lw=1.6)
    ax.plot([vw_d, vw_d], [0.0, v0_d], color=DESYcyan, lw=1.6)
    ax.plot(xi_d, v_d, color=DESYcyan, lw=1.6,
            label=r"deflagration, $v_\mathrm{w} = %.2f < c_\mathrm{s}$" % vw_d)
    ax.plot([xi_d[-1], xi_d[-1]], [v_d[-1], 0.0], color=DESYcyan, lw=1.6)
    ax.plot([xi_d[-1], 1.0], [0.0, 0.0], color=DESYcyan, lw=1.6)

    # A detonation: the wall outruns any signal, so the plasma ahead is undisturbed, and a
    # rarefaction behind it carries the velocity back to zero at the speed of sound.
    vw_t = 0.90
    v0_t = velocity_at_wall(vw_t, +1)
    xi_t, v_t = flow(vw_t, v0_t)
    ax.plot([0.0, xi_t[-1]], [0.0, 0.0], color=DESYmagenta, lw=1.6, ls=(0, (4, 2)))
    ax.plot(xi_t, v_t, color=DESYmagenta, lw=1.6, ls=(0, (4, 2)),
            label=r"detonation, $v_\mathrm{w} = %.2f > v_\mathrm{J}$" % vw_t)
    ax.plot([vw_t, vw_t], [v0_t, 0.0], color=DESYmagenta, lw=1.6, ls=(0, (4, 2)))
    ax.plot([vw_t, 1.0], [0.0, 0.0], color=DESYmagenta, lw=1.6, ls=(0, (4, 2)))

    ax.axvline(cs, color=DESYdunkelblau, lw=0.8, ls=":")
    ax.axvline(v_jouguet, color="0.70", lw=0.8, ls=":")
    top = 1.18 * max(v0_d, v0_t)
    ax.text(cs - 0.016, 0.96 * top, r"$c_\mathrm{s}$", color=DESYdunkelblau, fontsize=7,
            ha="right", va="center")
    ax.text(v_jouguet + 0.016, 0.96 * top, r"$v_\mathrm{J}$", color="0.45", fontsize=7,
            ha="left", va="center")
    ax.text(vw_d - 0.020, 1.04 * v0_d, "bubble wall", color=DESYcyan, fontsize=7,
            ha="right", va="bottom")
    ax.text(vw_t, 1.035 * v0_t, "bubble wall", color=DESYmagenta, fontsize=7,
            ha="center", va="bottom")
    ax.annotate("shock front", xy=(xi_d[-1] + 0.004, 0.22 * v_d[-1]),
                xytext=(0.655, 0.30 * top), color=DESYcyan, fontsize=7,
                ha="center", va="center",
                arrowprops=dict(arrowstyle="->", color=DESYcyan, lw=0.8))
    ax.annotate("rarefaction wave", xy=(0.74, float(np.interp(0.74, xi_t[::-1], v_t[::-1]))),
                xytext=(0.645, 0.76 * top), color=DESYmagenta, fontsize=7,
                ha="center", va="center",
                arrowprops=dict(arrowstyle="->", color=DESYmagenta, lw=0.8))
    ax.text(0.17, 0.11 * top, "bubble interior,\nplasma at rest", color="0.40", fontsize=7,
            ha="center", va="center")

    ax.set_xlabel(r"self-similar coordinate $\xi = r/t$")
    ax.set_ylabel(r"fluid velocity $v(\xi)$")
    ax.set_xlim(-0.015, 1.0)
    ax.set_ylim(-0.035 * top, 1.09 * top)
    ax.legend(frameon=False, loc="upper left")
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.155, top=0.975)
    _check(fig, "algorithm_fluid_profiles")
    fig.savefig(OUT / "algorithm_fluid_profiles.png", dpi=200)
    plt.close(fig)
    print("wrote", OUT / "algorithm_fluid_profiles.png")
    print("   alpha = %.3f, v_J = %.4f, v at the wall: %.4f and %.4f, shock at %.4f"
          % (alpha, v_jouguet, v0_d, v0_t, xi_d[-1]))


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
    fig, ax = plt.subplots(figsize=(WIDTH, 0.62 * WIDTH))

    # The shipped LISA power-law-integrated sensitivity, at the threshold and observation time
    # the package stores with it. A spectrum above this curve is detectable.
    from transitionlistener.observability import GW_Sensitivity_Data

    lisa = GW_Sensitivity_Data().det["LISA"]
    f_pli = np.asarray(lisa["f_PLI"], dtype=float)
    pli = np.asarray(lisa["PLI"], dtype=float)
    band = (f_pli >= f[0]) & (f_pli <= f[-1]) & (pli < 1.0)
    # A shallow shadow under the curve rather than the whole region below it, which would
    # cover most of the panel and put the two weaker sources inside grey. What rises above the
    # curve is detectable, and the sound-wave peak here does.
    ax.fill_between(f_pli[band], pli[band] / 40.0, pli[band], color="0.88", lw=0, zorder=0)
    ax.plot(f_pli[band], pli[band], color="0.50", lw=1.1, zorder=1)
    ax.text(3.2e-3, 3.0e-14, "LISA, power-law integrated", color="0.35", fontsize=7,
            ha="center", va="center")

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
    ax.legend(frameon=False, loc="lower left", ncol=1)
    ax.text(0.985, 0.355,
            "\n".join([r"$\alpha = 0.2$,  $R_* = 0.05$,  $T_\mathrm{reh} = 100$ GeV",
                       r"$v_\mathrm{w} = 0.65$,  $c_\mathrm{s} = 1/\sqrt{3}$",
                       r"$\kappa_\mathrm{sw} = %.3f$ (from the fit),  "
                       r"$\kappa_\mathrm{turb} = %.3f$" % (k_sw, 0.1 * k_sw),
                       r"$\kappa_\phi = 0.02$ (set by hand)"]),
            transform=ax.transAxes, fontsize=6.5, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="0.75",
                      linewidth=0.6))
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
