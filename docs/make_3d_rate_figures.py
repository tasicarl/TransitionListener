#!/usr/bin/env python3
"""Build the figures of the implementation note on the three-dimensional rate.

Run from the repository root with the source tree on the path::

    PYTHONPATH=$PWD/src python docs/make_3d_rate_figures.py [bubble] [gelfand_yaglom] [terms]

``bubble`` and ``gelfand_yaglom`` compute the critical bubble of the conformal U(1) without fermions at g = 0.475,
at the anchor temperature of its line-scan point (``paper/data/linescan_points``), and its fluctuation channels with
TransitionListener's own functions; the gauge-Goldstone mixing takes a few minutes. ``terms`` reads the terms of
the exponent at three couplings from ``paper/reproduce/numbers/output/determinant_reference_values.json``
(``paper/reproduce/numbers/determinant_reference_values.py``). The figures are written to ``docs/source/_static/``
in the house style of ``make_algorithm_figures.py``.
"""
from __future__ import annotations

import glob
import json
import math
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "docs"))
sys.path.insert(0, str(ROOT / "models"))
from make_algorithm_figures import OUT, WIDTH, use_house_style  # noqa: E402
from transitionlistener.eft_rate import abelian_higgs_3d  # noqa: E402
from transitionlistener.eft_rate import abelian_determinants as fl  # noqa: E402
from transitionlistener.eft_rate import nlo_tables as nt  # noqa: E402
from transitionlistener.colors import DESYcyan, DESYorange, DESYdunkelblau, DESYmagenta, DESYgruen  # noqa: E402

G = 0.475
GREY = "0.6"
C_CH = {"radial": DESYdunkelblau, "goldstone": DESYcyan, "gauge": DESYorange, "temporal": DESYgruen,
        "mixing": DESYmagenta}
LAB = {"radial": r"radial mode, $W_h$", "goldstone": r"Goldstone mode, $W_\chi$",
       "gauge": r"spatial gauge field, $W_A$", "temporal": r"$A_0$, $W_0$"}


def bubble_data() -> dict:
    """The bubble, the soft potential and the masses squared of the channels at g = 0.475, in units of T."""
    point = glob.glob(str(ROOT / "paper" / "data" / "linescan_points" / f"g{G:.3f}_u1_v*"))[0]
    tab = json.load(open(pathlib.Path(point) / "table_mu1.json"))
    T = [r for r in tab["rows"].values() if abs(r["mult"] - 1.0) < 1e-9][0]["T_GeV"]
    v = json.load(open(pathlib.Path(point) / "summary.json"))["v_GeV"]
    pot = nt.load_model("TL_conformal_u1_no_fermions_3d")({"g": G, "v_GeV": v}, verbose=False)
    p, S3, r, phi3, _ = nt._bounce(pot, T)
    r_u, phi_u, m3sq, lam3, g3sq, mD2 = nt._in_units(p, r, phi3, T)
    h3 = nt._h3_in_units(pot, p, T)
    kw = dict(m_debye_sq=mD2, n_temporal=abelian_higgs_3d.N_TEMPORAL_ABELIAN, h3=h3)
    nT = abelian_higgs_3d.N_TRANSVERSE_ABELIAN
    V = lambda q: abelian_higgs_3d.V3_LO(np.asarray(q, float), m3sq, lam3, g3sq, nT, **kw)      # noqa: E731
    dV = lambda q: abelian_higgs_3d.dV3_LO(np.asarray(q, float), m3sq, lam3, g3sq, nT, **kw)    # noqa: E731
    d2V = lambda q: abelian_higgs_3d.d2V3_LO(np.asarray(q, float), m3sq, lam3, g3sq, nT, **kw)  # noqa: E731
    lvl = abelian_higgs_3d.temporal_coupling(g3sq, h3)
    r_u, phi_u = np.asarray(r_u, float), np.asarray(phi_u, float)
    safe = np.where(phi_u > 0, phi_u, 1.0)
    W = dict(radial=d2V(phi_u),
             goldstone=np.where(phi_u > 0, dV(safe) / safe, d2V(0.0 * phi_u)) + g3sq * phi_u ** 2,
             gauge=g3sq * phi_u ** 2,
             temporal=mD2 + 0.5 * lvl * phi_u ** 2)
    W_fv = dict(radial=float(d2V(0.0)), goldstone=float(d2V(0.0)), gauge=0.0, temporal=mD2)
    phi_grid = np.linspace(0.0, 1.6 * phi_u[0], 400)
    i_half = int(np.argmin(np.abs(phi_u - 0.5 * phi_u[0])))
    return dict(T=T, v=v, S3=float(S3), r=r_u, phi=phi_u, W=W, W_fv=W_fv, phi_grid=phi_grid, V_grid=V(phi_grid),
                radius_half=float(r_u[i_half]), m3sq=m3sq, lam3=lam3, g3sq=g3sq, mD2=mD2, h3=h3)


def gelfand_yaglom_data(D: dict) -> dict:
    """psi_l/psi_l^FV along r in the channel of A_0, and the summands of the partial-wave sums."""
    r_u, rmax = D["r"], float(D["r"][-1])

    def dW_of(name):
        f = interp1d(r_u, D["W"][name] - D["W_fv"][name], kind="cubic", bounds_error=False, fill_value=0.0)
        return lambda x: float(f(x))

    def ratio_along_r(name, l, rs):
        m = math.sqrt(D["W_fv"][name])
        dWf = dW_of(name)

        def U(x):
            return 2.0 * (l + 1.0) / x if m <= 0 else 1.0 / x + 2.0 * m * fl._iv_log_deriv(l + 0.5, m * x)
        s = solve_ivp(lambda x, y: [y[1], -y[1] ** 2 - U(x) * y[1] + dWf(x)], (1e-4, rs[-1]), [0.0, 0.0],
                      t_eval=rs, rtol=1e-10, atol=1e-12)
        return s.y[0]

    rs = np.linspace(1e-3, rmax, 600)
    paths = {l: ratio_along_r("temporal", l, rs) for l in (0, 1, 2, 5)}
    ells = np.arange(0, 41)
    lnT = {n: np.array([fl.log_gy_single(dW_of(n), int(l), rmax, math.sqrt(D["W_fv"][n])) for l in ells])
           for n in ("goldstone", "gauge", "temporal")}
    mix = abelian_higgs_3d.mixing_log_det(D["r"], D["phi"], D["m3sq"], D["lam3"], D["g3sq"], m_debye_sq=D["mD2"],
                               l_max=abelian_higgs_3d.MIXING_L_MAX, h3=D["h3"])
    J = np.arange(1, abelian_higgs_3d.MIXING_L_MAX + 1)
    mix_summand = (2 * J + 1) * (np.array(mix["log_t_on"]) - np.array(mix["log_t_off"]))
    return dict(rs=rs, paths=paths, ells=ells, lnT=lnT, J=J, mix_summand=mix_summand)


def fig_bubble(D: dict) -> None:
    fig, axs = plt.subplots(1, 3, figsize=(WIDTH, 2.4), layout="constrained")
    ax = axs[0]
    x, y = D["phi_grid"], D["V_grid"]
    z = x <= 2.4
    ax.plot(x[z], 1e3 * y[z], color=DESYdunkelblau, lw=1.2)
    ax.set_xlim(0, 2.4)
    ax.axhline(0, color=GREY, lw=0.4)
    ax.set_xlabel(r"$\phi_3/\sqrt{T}$")
    ax.set_ylabel(r"$10^3\,V_3/T^3$")
    ax.set_title("soft potential near the barrier", fontsize=8)
    r, keep = D["r"], D["r"] < 6 * D["radius_half"]
    ax = axs[1]
    ax.plot(r[keep], D["phi"][keep], color="0.2", lw=1.2)
    ax.set_xlim(r[keep][0], r[keep][-1])
    ax.set_ylim(0, None)
    ax.axvline(D["radius_half"], color=GREY, lw=0.6, ls="--")
    ax.text(D["radius_half"] * 1.08, 0.55 * D["phi"][0], r"$R_{\rm c}$", fontsize=8)
    ax.set_xlabel(r"$rT$")
    ax.set_ylabel(r"$\phi_3/\sqrt{T}$")
    ax.set_title("critical bubble", fontsize=8)
    ax = axs[2]
    for k, ls in zip(("radial", "goldstone", "gauge", "temporal"), ("-", "--", ":", "-.")):
        ax.plot(r[keep], D["W"][k][keep], color=C_CH[k], lw=1.2, ls=ls, label=LAB[k])
    ax.set_xlim(r[keep][0], r[keep][-1])
    ax.axhline(0, color=GREY, lw=0.4)
    ax.set_xlabel(r"$rT$")
    ax.set_ylabel(r"$W/T^2$")
    ax.set_title("masses squared", fontsize=8)
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=4, fontsize=7, handlelength=1.6, frameon=False)
    fig.savefig(OUT / "algorithm_3d_bubble.png", dpi=200)
    plt.close(fig)


def fig_gelfand_yaglom(D: dict, GY: dict) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(WIDTH, 2.7), layout="constrained")
    ax = axs[0]
    rs = GY["rs"]
    k = rs <= 6 * D["radius_half"]
    for l, ls in zip((0, 1, 2, 5), ("-", "--", "-.", ":")):
        ax.plot(rs[k], np.exp(GY["paths"][l][k]), color=C_CH["temporal"], ls=ls, lw=1.1, label=rf"$\ell = {l}$")
    ax.set_yscale("log")
    ax.set_xlim(rs[k][0], rs[k][-1])
    ax.axvline(D["radius_half"], color=GREY, lw=0.5, ls="--")
    ax.text(D["radius_half"] * 1.08, 1.5, r"$R_{\rm c}$", fontsize=8)
    ax.set_xlabel(r"$rT$")
    ax.set_ylabel(r"$\psi_\ell(r)/\psi_\ell^{\rm FV}(r)$")
    ax.set_title(r"channel of $A_0$: the ratio tends to $\mathcal{T}_\ell$", fontsize=8)
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(0.0, -0.22), ncol=4, handlelength=1.6,
              columnspacing=0.8, frameon=False)
    ax = axs[1]
    ells = GY["ells"]
    for n in ("goldstone", "gauge", "temporal"):
        ax.plot(ells[1:], (2 * ells[1:] + 1) * GY["lnT"][n][1:], "o", ms=2.2, color=C_CH[n],
                label=LAB[n].split(",")[0])
    ax.plot(GY["J"], GY["mix_summand"], "s", ms=2.2, color=C_CH["mixing"], label=r"mixing, $(2J+1)\ln\mathcal{T}_J$")
    ax.axhline(0, color=GREY, lw=0.4)
    ax.set_xscale("log")
    ax.set_xlim(0.85, 80)
    ax.set_xlabel(r"$\ell$ or $J$")
    ax.set_ylabel(r"$(2\ell+1)\ln\mathcal{T}_\ell$")
    ax.set_title("summands of the partial-wave sums", fontsize=8)
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(0.0, -0.22), ncol=2, handlelength=1.0,
              columnspacing=0.8, frameon=False)
    fig.savefig(OUT / "algorithm_3d_gelfand_yaglom.png", dpi=200)
    plt.close(fig)


def fig_terms() -> None:
    ref = json.load(open(ROOT / "paper" / "reproduce" / "numbers" / "output" / "determinant_reference_values.json"))
    keys = [("radial", r"$\delta_h$"), ("goldstone", r"$\delta_\chi$"), ("gauge", r"$\delta_1$"),
            ("temporal", r"$\delta_0$"), ("half_mixing", r"$\frac{1}{2}\Delta_{\rm mix}$"),
            ("j0_correction", r"$S^{J=0}$"), ("minus_C", r"$-C$"), ("nlo_potential", r"$\int V_3^{\rm NLO}$"),
            ("S6", r"$S_6$"), ("prefactors", "prefactors")]
    fig, ax = plt.subplots(figsize=(0.85 * WIDTH, 2.6), layout="constrained")
    x = np.arange(len(keys) + 1, dtype=float)
    x[-1] += 0.6                                          # the sum stands apart
    for i, (e, col) in enumerate(zip(ref, (DESYdunkelblau, DESYcyan, DESYorange))):
        vals = dict(e, minus_C=-e["double_counting_C"])
        vals["prefactors"] = -e["log_A_dyn_over_T"] + e["three_halves_log_S_tot"]
        ys = [vals[k] for k, _ in keys]
        ax.bar(x[:-1] + (i - 1) * 0.26, ys, width=0.26, color=col, label=rf"$g = {e['g']}$")
        ax.bar(x[-1] + (i - 1) * 0.26, e["delta_S"], width=0.26, color=col, edgecolor="k", lw=0.8, hatch="////")
        # the bars add up to Delta S, eq. (deltaS): E - S3/T is the sum of the terms from delta_h to S6
        assert abs(sum(ys) - e["delta_S"]) < 1e-3, (e["g"], sum(ys), e["delta_S"])
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_ylim(-300, 600)
    ax.axhline(0, color="0.2", lw=0.6)
    ax.axvline(x[-1] - 0.8, color="0.2", lw=0.6, ls="--")
    ax.text(x[-1], 150, "sum", ha="center", fontsize=7)
    ax.set_xticks(x, [lab for _, lab in keys] + [r"$\Delta S$"], fontsize=7)
    ax.tick_params(axis="x", which="minor", bottom=False, top=False)
    ax.set_ylabel("contribution to the exponent")
    ax.legend(loc="upper left", fontsize=7, ncol=3, frameon=False)
    fig.savefig(OUT / "algorithm_3d_terms.png", dpi=240)
    plt.close(fig)


if __name__ == "__main__":
    use_house_style()
    which = sys.argv[1:] or ["bubble", "gelfand_yaglom", "terms"]
    if "bubble" in which or "gelfand_yaglom" in which:
        D = bubble_data()
        print(f"g = {G}: T = {D['T']:.4e} GeV, S3/T = {D['S3']:.4f}, phi(0) = {D['phi'][0]:.3f} sqrt(T), "
              f"R_c T = {D['radius_half']:.3f}", flush=True)
        if "bubble" in which:
            fig_bubble(D)
        if "gelfand_yaglom" in which:
            fig_gelfand_yaglom(D, gelfand_yaglom_data(D))
    if "terms" in which:
        fig_terms()
