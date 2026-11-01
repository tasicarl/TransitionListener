#!/usr/bin/env python3
"""Build the figures used by the algorithm pages.

Run from the repository root with the source tree on the path::

    PYTHONPATH=$PWD/src python docs/make_algorithm_figures.py

Each figure is written to ``docs/source/_static/`` and is reproducible from this script alone,
so that a change in the code shows up in the figure rather than being quietly contradicted by
it. The models used are the ones shipped with the package.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = pathlib.Path(__file__).resolve().parent / "source" / "_static"
OUT.mkdir(parents=True, exist_ok=True)

INK = "#0b2545"
ACCENT = "#1f9bd1"
WARM = "#e08a1e"


def figure_bounce_profile() -> None:
    """The bounce profile, with the radius inside which a trial counts as arrived."""
    from transitionlistener.tunneling1D import SingleFieldInstanton

    def V(phi):
        return 0.25 * phi ** 4 - 0.4 * phi ** 3 + 0.1 * phi ** 2

    def dV(phi):
        return phi * (phi - 0.2) * (phi - 1.0)

    inst = SingleFieldInstanton(1.0, 0.0, V, dV)
    prof = inst.findProfile()
    r, phi = prof.R, prof.Phi

    # the acceptance radius the integration uses, as the page describes it
    phitol = 1e-10
    excursion = abs(phi[0] - inst.phi_metaMin)
    epsabs = phitol * excursion
    band = 3 * epsabs

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(r, phi, color=ACCENT, lw=2.2, label=r"$\phi(r)$")
    ax.axhline(inst.phi_metaMin, color=INK, lw=1.0, ls="--",
               label=r"false vacuum $\phi_{\rm false}$")
    ax.axhline(inst.phi_absMin, color=INK, lw=1.0, ls=":",
               label=r"true vacuum $\phi_{\rm true}$")

    # the acceptance band is far too narrow to see at this tolerance, so it is annotated
    ax.annotate(
        "a trial counts as arrived inside\n"
        r"$3\,\epsilon_{\rm abs} = 3\,\phi_{\rm tol}\,|\phi(0)-\phi_{\rm false}|$"
        "\n(too narrow to draw here)",
        xy=(r[-1], inst.phi_metaMin), xytext=(0.52 * r[-1], 0.33),
        color=INK, fontsize=9,
        arrowprops=dict(arrowstyle="->", color=INK, lw=1.0))

    ax.annotate(r"$\phi(0)$, the shooting unknown",
                xy=(r[0], phi[0]), xytext=(0.12 * r[-1], 0.80),
                color=WARM, fontsize=9,
                arrowprops=dict(arrowstyle="->", color=WARM, lw=1.0))

    ax.set_xlabel(r"radius $r$")
    ax.set_ylabel(r"field $\phi$")
    ax.set_xlim(0, r[-1])
    ax.set_ylim(-0.05, 1.30)          # headroom so the legend clears the true-vacuum line
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "algorithm_bounce_profile.png", dpi=160)
    plt.close(fig)
    print("wrote", OUT / "algorithm_bounce_profile.png",
          "(acceptance band %.3e in field units)" % band)


def figure_rate_and_support() -> None:
    """Why the support has to resolve the rise of the rate, not the temperature range."""
    # A schematic built from the shape the integrand actually has: log(Gamma/H^4) rises
    # steeply, peaks, and dies. No model is run; the point is the shape and the sampling.
    T = np.linspace(0.2, 1.0, 2000)
    logG = -40.0 * (T - 0.46) ** 2 / 0.02 + 6.0          # a sharp rise and fall
    P = 1.0 / (1.0 + np.exp((T - 0.455) / 0.006))        # the converted fraction

    uniform = np.linspace(0.2, 1.0, 12)
    adaptive = np.concatenate([
        np.array([0.20, 0.30, 0.38]),
        np.linspace(0.43, 0.49, 7),
        np.array([0.60, 0.80, 1.00])])

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.4, 5.2), sharex=True,
                                 gridspec_kw=dict(hspace=0.12))
    a1.plot(T, logG, color=ACCENT, lw=2.0)
    a1.set_ylabel(r"$\log_{10}(\Gamma/H^4)$")
    a1.set_ylim(-34, 16)
    a1.plot(uniform, np.interp(uniform, T, logG), "v", color=WARM, ms=7,
            label="uniform support")
    a1.plot(adaptive, np.interp(adaptive, T, logG), "o", color=INK, ms=5,
            label="support placed on the rise")
    a1.legend(frameon=False, fontsize=9, loc="lower left", ncol=2,
              bbox_to_anchor=(0.0, -0.02))

    a2.plot(T, P, color=ACCENT, lw=2.0)
    a2.plot(uniform, np.interp(uniform, T, P), "v", color=WARM, ms=7)
    a2.plot(adaptive, np.interp(adaptive, T, P), "o", color=INK, ms=5)
    a2.axhline(0.28957, color=INK, lw=1.0, ls="--")
    a2.text(0.985, 0.33, r"$f_{\rm perc}$", color=INK, fontsize=9, ha="right")
    a2.set_ylabel(r"converted fraction $P$")
    a2.set_xlabel("temperature (arbitrary units, cooling to the left)")
    fig.tight_layout()
    fig.savefig(OUT / "algorithm_rate_and_support.png", dpi=160)
    plt.close(fig)
    print("wrote", OUT / "algorithm_rate_and_support.png")


def main() -> int:
    figure_bounce_profile()
    figure_rate_and_support()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
