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


def main() -> int:
    figure_shooting()
    figure_rate_and_support()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
