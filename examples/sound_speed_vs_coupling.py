#!/usr/bin/env python3
"""Broken-phase sound speed against the gauge coupling of the conformal dark U(1).

Run it to reproduce the measurements quoted in the changelog entry for the daisy term and
the sound speed.

Two evaluations of the same quantity, c_s^2 = (dV/dT)/(T d2V/dT2) at fixed field value on
the traced broken phase:

  "as released"  the derivatives of the whole effective potential;
  "stabilised"   the derivatives of its temperature-dependent part alone, with the daisy
                 term evaluated without cancellation and its thermal masses taken
                 analytically.

They are the same quantity analytically, since the temperature-independent part of the
potential drops out of both derivatives.

Percolation is not computed. Each curve is a fixed ratio of the temperature to the scale
v_stable, so reading down the curves is reading into the supercooled regime.

Usage:
    python plot_cs_vs_g.py [--g-min 0.45] [--g-max 0.75] [--n 61] [--out cs_vs_g.pdf]
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from transitionlistener.helper_functions import load_potential   # noqa: E402
from transitionlistener.phases import Phases                     # noqa: E402

MODEL = REPO / "models/TL_conformal_dark_u1.py"
RATIOS = (1.0e-1, 1.0e-2, 1.0e-3, 1.0e-5, 1.0e-7, 1.0e-9)


def broken_phase(phases, T):
    best, best_norm = None, 0.0
    for p in phases.values():
        if not (p.Tmin <= T <= p.Tmax):
            continue
        n = float(np.linalg.norm(np.atleast_1d(np.squeeze(p.valAt(T)))))
        if n > best_norm:
            best, best_norm = p, n
    return best


def cs_values(pot, phases, T):
    """(c_s as released, c_s stabilised) at temperature T, or (nan, nan)."""
    phase = broken_phase(phases, T)
    if phase is None:
        return float("nan"), float("nan")
    X = np.atleast_1d(np.squeeze(phase.valAt(T)))

    def V_released(t):
        return float(np.squeeze(pot.Vtot(X, t, include_decoupled=False)))

    def V_stable(t):
        ta = np.asarray([t], dtype=float)
        b0 = pot.boson_massSq(X, ta * 0.0)
        bT = pot.boson_massSq(X, ta)
        fer = pot.fermion_massSq(X)
        y = pot.V1T(b0, fer, ta) + pot.Vdaisy(b0, bT, ta, Pi=pot.debye_massSq(X, ta))
        return float(np.squeeze(y + pot.constantTerms(ta, include_decoupled=False)))

    out = []
    for f in (V_released, V_stable):
        try:
            h = T * 1.0e-3
            d1 = (f(T + h) - f(T - h)) / (2.0 * h)
            d2 = (f(T + h) - 2.0 * f(T) + f(T - h)) / h ** 2
            cs_sq = d1 / (T * d2) if d2 != 0.0 else float("nan")
            out.append(np.sqrt(cs_sq) if np.isfinite(cs_sq) and cs_sq > 0 else float("nan"))
        except Exception:
            out.append(float("nan"))
    return out[0], out[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g-min", type=float, default=0.45)
    ap.add_argument("--g-max", type=float, default=0.75)
    ap.add_argument("--n", type=int, default=61)
    ap.add_argument("--out", default="cs_vs_g.pdf")
    a = ap.parse_args()

    gs = np.linspace(a.g_min, a.g_max, a.n)
    released = np.full((len(RATIOS), len(gs)), np.nan)
    stabilised = np.full((len(RATIOS), len(gs)), np.nan)

    for j, g in enumerate(gs):
        with contextlib.redirect_stdout(io.StringIO()):
            pot = load_potential(str(MODEL), "specific_potential")(
                {"g": float(g), "y": 0.01, "v_GeV": 0.1}, verbose=False)
            phases = Phases(pot, False).phases
        if not phases:
            continue
        for i, ratio in enumerate(RATIOS):
            T = ratio * pot.v_stable
            released[i, j], stabilised[i, j] = cs_values(pot, phases, T)
        print(f"  g = {g:.4f} done", flush=True)

    np.savez(Path(a.out).with_suffix(".npz"), g=gs, ratios=np.array(RATIOS),
             released=released, stabilised=stabilised)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), sharey=True)
    colours = plt.cm.viridis(np.linspace(0.08, 0.92, len(RATIOS)))
    for ax, data, title in ((axes[0], released, "as released"),
                            (axes[1], stabilised, "stabilised")):
        ax.axhline(1.0, color="0.25", lw=1.0, ls=(0, (6, 3)))
        ax.axhline(1.0 / np.sqrt(2.0), color="0.55", lw=0.9, ls=(0, (2, 2)))
        ax.axhline(1.0 / np.sqrt(3.0), color="0.55", lw=0.9, ls=(0, (1, 2)))
        for i, ratio in enumerate(RATIOS):
            exp = int(round(np.log10(ratio)))
            ax.plot(gs, data[i], color=colours[i], lw=1.5,
                    label=rf"$T/v = 10^{{{exp}}}$")
        ax.set_xlabel(r"gauge coupling $g$")
        ax.set_title(title, fontsize=11)
        ax.set_xlim(gs[0], gs[-1])
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].set_ylabel(r"broken-phase sound speed $c_s$  [units of $c$]")
    axes[0].set_ylim(0.0, 1.35)
    # the reference lines are named in the margin, never written across the curves
    axes[1].text(1.015, 1.0 / np.sqrt(3.0), r"$1/\sqrt{3}$", transform=axes[1].get_yaxis_transform(),
                 va="center", ha="left", fontsize=9, color="0.35")
    axes[1].text(1.015, 1.0 / np.sqrt(2.0), r"$1/\sqrt{2}$", transform=axes[1].get_yaxis_transform(),
                 va="center", ha="left", fontsize=9, color="0.35")
    axes[1].text(1.015, 1.0, r"$c$", transform=axes[1].get_yaxis_transform(),
                 va="center", ha="left", fontsize=9, color="0.25")
    axes[0].legend(fontsize=8, ncol=2, loc="lower right", framealpha=0.95)
    fig.suptitle("Conformal dark U(1), $v = 0.1$ GeV, $y = 0.01$: sound speed of the broken phase",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 0.965, 0.96))
    fig.savefig(a.out)
    print(f"\nwrote {a.out} and {Path(a.out).with_suffix('.npz')}")


if __name__ == "__main__":
    main()
