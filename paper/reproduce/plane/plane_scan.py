"""The (g, v) plane: PTA likelihoods for every variant of the rate and of the
cosmological modelling, on the same grid, so that the width of the preferred
region can be decomposed into what produced it.

Inputs (in paper/data)
  transition_table.npz / transition_table.py
      T_p, the three measures of beta/H, alpha, T_nuc, the completion measure and
      T_reh at every (g, log10 v) for every rate variant, from the exact
      integration of the false-vacuum fraction on the self-consistent mean
      background (rate/build_exact_table.py). At fixed g the exponent is a
      function of T/v; the Planck mass is the only other scale, so that
      ln(Gamma/H^4) at fixed T/v shifts by -4 ln v and beta/H at fixed g falls by
      17-25% per decade of v. Scaling T_p ~ v at fixed beta/H would drop that
      dependence.
  linescan_u1_mu0_1GeV.json
      TransitionListener's line scan (v = e^{1/6} mu0/g); its couplings and v are kept as G_SCAN and LOGV_LINE.
  daisy_table_plane.npz (daisy/prepare_daisy_table.py)
      the 4D daisy treatment (N5) on the plane from its own action table (daisy/build_daisy_table.py).
  daisy_rg_table_plane.npz (daisy/prepare_daisy_table.py rg)
      the same with the couplings run to mu = pi T (N4; models/TL_conformal_u1_no_fermions_rg.py).
      (beta/H)_RH of exponential rates against their beta/H_V, for the expansion factors.
  TransitionListener (gwfopt)     every spectrum: the Lewicki-Vaskonen fits with and without
                                  the cosmic expansion (2511.15687, 2208.11697) and the LISA
                                  sound-wave templates, each a configuration of FOPTspectrum.
  transition_table T_reh_TL       the reheating temperature and the degrees of freedom right after
                                  reheating from TransitionListener at every node (tl_nodes.py).
  binaryfog                       the free-spectrum likelihoods (ceffyl only for NG15_v1, PTArcade's 2024 product).

Run with PYTHONPATH=<repo>/src in an environment with PTArcade, ceffyl and binaryfog:
  python plane_scan.py [--linescan JSON] [--out NPZ] [--workers N]
                                  writes paper/data/plane_scan_u1.npz by default
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import paths  # noqa: E402  (src/, models/ and the folders of paper/reproduce on sys.path)
ROOT = paths.REPO
import transition_table as TT  # noqa: E402

# ---------------------------------------------------------------- plasma
# The reheating temperature and the energy and entropy degrees of freedom right after reheating
# are TransitionListener's (transitionObservables: the broken-phase temperature followed through
# the conversion; thermodynamics.reheating_geff: the Standard Model bath plus the dark photon and
# the dark Higgs at their masses), tabulated at the nodes by tl_nodes.py. The shortcut
# T_reh = T_p (1 + alpha)^{1/4} is the alternative; its degrees of freedom are those of the
# Standard Model table TransitionListener ships (tab_data/geff_SM.dat).
_tab = np.genfromtxt(os.path.join(paths.SRC, "transitionlistener", "tab_data", "geff_SM.dat"))
_lx = np.log(_tab[:, 0])


def g_rho(T):
    return np.interp(np.log(T), _lx, _tab[:, 1])


def g_s(T):
    return np.interp(np.log(T), _lx, _tab[:, 3])


class _Reheating(tuple):
    """(T_reh, g_eff, h_eff), with the efficiency kappa of the same point as an attribute."""


def reheating(g, logv):
    """(T_reh in GeV, g_eff, h_eff) right after reheating, from TransitionListener."""
    r = _Reheating(float(x) for x in TT.reheating_tl(g, logv))
    r.kappa = float(TT.kappa_tl(g, logv))
    return r


# ---------------------------------------------------------------- command line and line scan
import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--linescan", default=paths.data("linescan_u1_mu0_1GeV.json"))
_ap.add_argument("--out", default=paths.data("plane_scan_u1.npz"))
_ap.add_argument("--workers", type=int, default=40)
# as a script: the command line; imported by another script: the arguments in PLANE_SCAN_ARGS
ARGS = _ap.parse_args() if __name__ == "__main__" else _ap.parse_args(os.environ.get("PLANE_SCAN_ARGS", "").split())
rows = json.load(open(ARGS.linescan))
G_SCAN = np.array(sorted(float(g) for g in rows))
LOGV_LINE = np.log10(np.array([rows[str(g)]["v_GeV"] for g in G_SCAN]))


# the 4D daisy treatment (N5): its action tabulated once per coupling and the false-vacuum fraction
# integrated on the plane like the NLO table (daisy/build_daisy_table.py, prepared by
# daisy/prepare_daisy_table.py). DAISY_TABLE=tl takes the action of TransitionListener's default path
# solver, which fails below T/v ~ 2.7e-4 g; DAISY_TABLE=direct its one-dimensional instanton solver
# with tolerances relative to v (biased at T/v < 1e-6); the default (bounce) the same solver with
# tolerances relative to the bounce, as for the 3D actions.
_DAISY_FILE = {"bounce": "daisy_table_plane.npz", "direct": "daisy_table_plane_direct.npz",
               "tl": "daisy_table_plane_tl.npz"}[os.environ.get("DAISY_TABLE", "bounce")]
DT = TT.load_instance(paths.data(_DAISY_FILE), "daisy_table")
# N4: the 4D daisy rate with the couplings run to mu = pi T, on the plane the same way
DTR = TT.load_instance(paths.data("daisy_rg_table_plane.npz"), "daisy_rg_table")


def _daisy(g, logv, table=None):
    """((beta/H)_R, alpha, completes, (beta/H)_Gamma) of the 4D daisy rate at (g, log10 v), NaN where it
    does not percolate inside the table (DT, or DTR for the running couplings)."""
    t = DT if table is None else table
    bh = float(t.get("betaH_R", g, logv, "daisy"))
    alpha = float(t.get("alpha", g, logv, "daisy"))
    return bh, alpha, bool(np.isfinite(t.get("T_f", g, logv, "daisy"))), float(t.get("beta_Gamma_H", g, logv, "daisy"))


# ---------------------------------------------------------------- variants
# Each variant of the transition is (T_reh, beta/H, alpha, (beta/H)_Gamma) at (g, log10 v); the
# spectrum then follows from the spectral template. The central beta/H is the
# one inferred from the mean bubble separation, (beta/H)_R = (8 pi/f_perc)^{1/3}/(R_* H);
# (beta/H)_Gamma = H^-1 d ln Gamma/dt at T_p (H ~ H_V) is the speed of the template's expansion factors.
RATE_VARIANTS = {            # name in the paper -> variant of the table
    "nlo": "nlo", "mu4 x0.5": "mu05", "mu4 x2": "mu2", "prefactor x10": "pref+",
    "full leading order": "fulllo", "mu3 x0.5": "mu3x05", "mu3 x2": "mu3x2",
    "prefactor /10": "pref-", "higher order +": "hord+", "higher order -": "hord-",
    # not uncertainties: the reach of the data for an error of the size of the lattice test, in the
    # direction the lattice found (a smaller nucleation rate, i.e. a larger exponent)
    "lattice shift -": "latc-", "lattice scale -": "latp-",
    # checks of Delta S (appendix B): without Delta S, and Delta S from the tables of the line scan instead of TL's grid
    "leading order": "lorate", "Delta S tables": "dStab",
}
TRANSITION_COSMO = ("percolation at 1/e", "beta/H from the action slope",
                    "dark sector with its own temperature", "local completion criterion",
                    "false-vacuum background", "template within its simulated range",
                    "expansion factors at beta/H_V")
BETA_MIN = 4.0      # the slowest transition simulated for the default template (2511.15687): marked in the
                    # figures; the template is extrapolated below it, not cut


def _completes(g, logv, var="nlo"):
    """Completion as TransitionListener requires it: the true-vacuum fraction reaches 0.99 at a
    finite temperature T_f. This follows the false-vacuum fraction to the end of the transition."""
    return bool(np.isfinite(TT.get("T_f", g, logv, var)))


def _completes_locally(g, logv, var="nlo"):
    """The local criterion of Ellis et al. (arXiv:1809.08242): the physical false-vacuum volume
    a^3 P_f decreases at T_p. An estimate from one instant, used for the alternative row only."""
    return float(TT.get("dlnV_dt_over_H", g, logv, var)) < 0.0


def tp_variants(g, logv):
    """{name: (T_reh, beta/H, alpha, (beta/H)_Gamma)} at one (g, log10 v). By default a point needs a
    transition that percolates and completes (a final temperature T_f with a true-vacuum
    fraction of 0.99 exists); where it does not, beta/H is NaN. Below (beta/H)_RH = 4, the slowest
    simulated transition, the template is extrapolated (no cut). A point whose (beta/H)_Gamma is not
    positive has no spectrum (make_spectrum returns None). "local completion criterion" in addition
    demands that a^3 P_f shrinks at T_p."""
    treh = reheating(g, logv)[0]
    out = {}

    def put(name, bh, alpha, bhv, var="nlo", complete=True, local=False):
        ok = (np.isfinite(bh) and bh > 0 and (not complete or _completes(g, logv, var))
              and (not local or _completes_locally(g, logv, var)))
        out[name] = (treh, float(bh) if ok else np.nan, float(alpha), float(bhv))
    V = lambda var: TT.get("beta_Gamma_H", g, logv, var)
    for name, var in RATE_VARIANTS.items():
        put(name, TT.get("betaH_R", g, logv, var), TT.get("alpha", g, logv, var), V(var), var)
    bh4, al4, done4, bv4 = _daisy(g, logv)
    out["4D daisy"] = (treh, bh4 if (np.isfinite(bh4) and bh4 > 0 and done4) else np.nan, al4, bv4)
    if DTR is not None:
        bh6, al6, done6, bv6 = _daisy(g, logv, DTR)
        out["4D daisy RG"] = (treh, bh6 if (np.isfinite(bh6) and bh6 > 0 and done6) else np.nan, al6, bv6)
    a0 = float(TT.get("alpha", g, logv))
    tp = float(TT.get("T_p", g, logv))
    put("percolation at 1/e", TT.get("betaH_R", g, logv, "perc1e"), TT.get("alpha", g, logv, "perc1e"), V("perc1e"))
    # the literature's single speed: the action slope in every factor of the template, the
    # expansion factors included
    bs = TT.get("betaH_S", g, logv)
    put("beta/H from the action slope", bs, a0, bs)
    put("dark sector with its own temperature", TT.get("betaH_R", g, logv, "dark"), TT.get("alpha", g, logv, "dark"),
        V("dark"), "dark")
    put("local completion criterion", TT.get("betaH_R", g, logv), a0, V("nlo"), local=True)
    # the percolation integral on the expansion rate of the false vacuum instead of the mean one
    # (rate/build_falsevac_table.py, variant fvbg of the transition table)
    put("false-vacuum background", TT.get("betaH_R", g, logv, "fvbg"), TT.get("alpha", g, logv, "fvbg"),
        V("fvbg"), "fvbg")
    # the template's own normalisation of its expansion factors: beta/H_V = (beta/H)_Gamma H_p/H_V
    # (2511.15687 write Gamma = H_V^4 e^{beta t}); used (a) as the speed in the expansion factors and
    # (b) to restrict the template to its simulated range, beta/H_V >= 4
    bhV = float(V("nlo")) * float(TT.get("H_p_over_HV", g, logv))
    bR = TT.get("betaH_R", g, logv)
    put("template within its simulated range", bR if bhV >= BETA_MIN else np.nan, a0, V("nlo"))
    put("expansion factors at beta/H_V", bR, a0, bhV)
    return out


def central(g, logv):
    """(T_reh, beta/H, alpha, (beta/H)_Gamma) of the central prescription."""
    return tp_variants(g, logv)["nlo"]


LV = "LewickiVaskonen22"
COSMO_VARIANTS = [
    # name, options of TransitionListener's FOPTspectrum (gwConf) and of the inputs
    ("expansion, dissipative", dict(source=LV, expansion=True, shape="dissipative")),
    ("sound waves", dict(source="LISA-pipeline24", ups="min", expansion=False)),
    ("sound waves, guo", dict(source="LISA-pipeline24", ups="guo", expansion=False)),
    ("LISA update 2019", dict(source="LISA-update2019", ups="min", expansion=False)),
    ("expansion, bulk flow", dict(source=LV, expansion=True, shape="non_dissipative")),
    ("no expansion, bubbles R^-3", dict(source=LV, expansion=False, flat="bubbles_R3")),
    ("no expansion, bubbles R^-2", dict(source=LV, expansion=False, flat="bubbles_R2")),
    ("no expansion, envelope", dict(source=LV, expansion=False, flat="bubbles_envelope")),
    ("no expansion, fluid R^-3 v=1", dict(source=LV, expansion=False, flat="fluid_R3_v1")),
    ("no expansion, fluid R^-3 v=cs", dict(source=LV, expansion=False, flat="fluid_R3_vcs")),
    ("no expansion, fluid R^-2 v=1", dict(source=LV, expansion=False, flat="fluid_R2_v1")),
    ("no expansion, fluid R^-2 v=cs", dict(source=LV, expansion=False, flat="fluid_R2_vcs")),
    ("T_reh = T_p (1+alpha)^1/4", dict(source=LV, expansion=True, shape="dissipative", treh_simple=True)),
    # the speed of the expansion factors of 2511.15687, which were fitted to exponential rates: by
    # default (beta/H)_Gamma = H^-1 d ln Gamma/dt at T_p; alternatively the speed from the bubble
    # separation, as every other factor (the convention of ref. [P-I])
    ("expansion factors at the separation speed", dict(source=LV, expansion=True, shape="dissipative", exp_speed="R")),
]


def cosmo_inputs(c, g, logv, alpha):
    """(T_reh, g_eff, h_eff) of a spectrum-level variant: TransitionListener's reheating, or the
    shortcut T_p (1 + alpha)^{1/4} with the Standard Model degrees of freedom at that temperature."""
    if c[1].get("treh_simple"):
        treh = float(TT.get("T_p", g, logv)) * (1 + alpha) ** 0.25
        r = _Reheating((treh, float(g_rho(treh)), float(g_s(treh))))
        r.kappa = reheating(g, logv).kappa
        return r
    if c[1].get("treh_inst"):
        # the instantaneous balance rho_rad(T) = Delta V with the degrees of freedom at T, those of the
        # dark sector taken from TransitionListener's reheating (the dark Higgs, mostly)
        t_tl, g_tl, h_tl = reheating(g, logv)
        dg, dh = g_tl - float(g_rho(t_tl)), h_tl - float(g_s(t_tl))
        t_sm = float(TT.get("T_reh_SM", g, logv))
        treh = t_sm
        for _ in range(3):
            treh = t_sm * (float(g_rho(treh)) / (float(g_rho(treh)) + dg)) ** 0.25
        r = _Reheating((treh, float(g_rho(treh)) + dg, float(g_s(treh)) + dh))
        r.kappa = reheating(g, logv).kappa
        return r
    return reheating(g, logv)


# ---------------------------------------------------------------- spectra: TransitionListener
from transitionlistener.config import Configuration  # noqa: E402
from transitionlistener.gwfopt import FOPTspectrum  # noqa: E402


class _QuickSpectrum(FOPTspectrum):
    """FOPTspectrum without the precomputed grids and the peak search of its
    constructor, which dominate the cost of a grid scan; the spectrum itself is
    the parent's."""

    def __init__(self, params, config):
        self.gwparams_dict = params
        self.astroparams_dict = {"A": 0.0, "gamma": 0.0}
        self.alphaDS = params.get("alpha_DS", np.nan)
        self._read_modelling_choices(config)
        self.alpha, self.RH, self.Treh_SM_GeV = params["alpha"], params["RH"], params["Treh_SM_GeV"]
        self.g_eff_tot_reh, self.h_eff_tot_reh = params["g_eff_tot_reh"], params["h_eff_tot_reh"]
        self.kappa_phi, self.kappa_sw, self.kappa_turb = params["kappa_phi"], params["kappa_sw"], params["kappa_turb"]
        self.v_wall, self.c_s, self.D = params["v_wall"], params["c_s"], params["D"]
        self.alpha_hyd, self.g0, self.h0 = params["alpha_hyd"], 2, 3.9309363
        self.betaH = (8 * np.pi / self.f_perc) ** (1 / 3) / self.RH * max(self.v_wall, self.c_s)
        self.betaH_Gamma = params["betaH_Gamma"]
        self.verbose = False
        self.sources = list(self.SOURCES[self.source_model])


def make_spectrum(opts, reh, betaH, alpha, betaH_X):
    """h^2 Omega_GW(f) today from TransitionListener's FOPTspectrum for the source options
    `opts` (see COSMO_VARIANTS), the reheating (T_reh in GeV, g_eff, h_eff) and the transition
    (beta/H, alpha). TransitionListener takes the bubble separation at percolation,
    R_* H = (8 pi/f_perc)^{1/3}/(beta/H) for walls at the speed of light, and derives beta/H
    from it; the Lewicki-Vaskonen fits are written in beta/H, the LISA templates in the
    convention H_* R_* = (8 pi)^{1/3}/(beta/H) of their simulations.
    Lewicki-Vaskonen: the fits with expansion take the computed efficiency kappa of `reh` (TL's
    kappa_phi + kappa_sw at the nodes); the fits without expansion are defined with all of the released
    energy in one kind of shell (kappa = 1), scalar-field or fluid. LISA templates: sound waves only, with the
    Espinosa et al. efficiency at this alpha, no collisions or turbulence, as published: without the
    expansion factors of 2511.15687 and without a causal continuation."""
    T_reh, g_reh, h_reh = reh
    # the speed of the expansion factors (2511.15687), betaH_X: (beta/H)_Gamma of the rate by default,
    # or the separation speed itself ("R")
    if opts.get("exp_speed") == "R":
        betaH_X = betaH
    uses_expansion = opts.get("expansion", True) or opts["source"] != LV
    if uses_expansion and not (np.isfinite(betaH_X) and betaH_X > 0):
        return None
    if not uses_expansion:
        betaH_X = np.nan                     # not read without the expansion factors
    conf = Configuration()
    gwc = conf.gwConf
    gwc.source_model = opts["source"]
    f_perc = conf.percolationConf.f_perc
    RH = (8 * np.pi / f_perc) ** (1 / 3) / betaH
    if opts["source"] == LV:
        gwc.expansion_suppression = bool(opts["expansion"])
        gwc.causality_tail = opts.get("tail", "smooth")
        if opts["expansion"]:
            gwc.lv22_expanding_shape = opts["shape"]
            k_phi, k_sw = reh.kappa, 0.0     # the computed efficiency in the single shape of the fit
        elif opts["flat"].startswith("fluid"):
            gwc.lv22_shape_fluid = opts["flat"]
            k_phi, k_sw = 0.0, 1.0
        else:
            gwc.lv22_shape_field = opts["flat"]
            k_phi, k_sw = 1.0, 0.0
    else:
        gwc.sw_lifetime_factor = opts["ups"]
        # the LISA templates as published: without the expansion factors of 2511.15687 and without a causal
        # continuation; a variant may switch the factors on with expansion=True
        gwc.expansion_suppression = bool(opts.get("expansion", True))
        k_phi, k_sw = 0.0, alpha / (0.73 + 0.083 * np.sqrt(alpha) + alpha)
    gwc.expansion_speed = "betaH_Gamma"
    params = dict(alpha=alpha, RH=RH, Treh_SM_GeV=T_reh, g_eff_tot_reh=g_reh, h_eff_tot_reh=h_reh,
                  kappa_phi=k_phi, kappa_sw=k_sw, kappa_turb=0.0, v_wall=1.0, c_s=1 / np.sqrt(3), D=1.0,
                  alpha_hyd=alpha, betaH_Gamma=betaH_X)
    spec = _QuickSpectrum(params, conf)
    return lambda f, **kw: spec.h2Omega_0_sum(np.asarray(f, dtype=float))


# ---------------------------------------------------------------- likelihoods
from ptarcade import signal_builder  # noqa: E402
from transitionlistener.observability import PTAConfig, PTAModel  # noqa: E402


# ---------------------------------------------------------------- binaryfog: other free spectra and the population foreground
import binaryfog as bf  # noqa: E402  (pip install binaryfog==1.0.0)
from binaryfog import priors as bf_priors  # noqa: E402

BF_DATA = {
    "cp_v1": bf.FreeSpectrum.load("ng15_cp_v1", nbins=14),          # the 2024 product PTArcade ships
    "cp_v2": bf.FreeSpectrum.load("ng15_cp_v2", nbins=14),          # corrected 2026 release: the default
    "cp_v2_5": bf.FreeSpectrum.load("ng15_cp_v2", nbins=5),
    "cp_v2_30": bf.FreeSpectrum.load("ng15_cp_v2", nbins=30),
    "hd_v2": bf.FreeSpectrum.load("ng15_hd_v2", nbins=14),          # Hellings-Downs correlated, corrected
    "hd_v1": bf.FreeSpectrum.load("ng15_hd_v1", nbins=14),          # the same in 2024, reference for DMGP
    "hd_dmgp_v1": bf.FreeSpectrum.load("ng15_hd_dmgp_v1", nbins=14),  # HD with the DM Gaussian-process noise model
    "ipta2": bf.FreeSpectrum.load("ipta2_cp_v1", nbins=13),          # IPTA DR2, the free spectrum PTArcade reads
    "ipta2_5": bf.FreeSpectrum.load("ipta2_cp_v1", nbins=5),
    "ipta2_30": bf.FreeSpectrum.load("ipta2_cp_v1", nbins=30),
}
BF_LIKE = {k: bf.Likelihood(v) for k, v in BF_DATA.items()}
BF_FOG = bf.Likelihood(BF_DATA["cp_v2"], bf.Background())
# the number of prior draws that seed the profile (64 by default; BF_NDRAW=256 is the convergence check)
BF_THETA = bf_priors.start_points(np.random.default_rng(20260925), int(os.environ.get("BF_NDRAW", "64")), "nanograv")
BF_LNPRIOR = np.array([bf_priors.ln_prior(th, "nanograv") for th in BF_THETA])


def bf_lnL(like, spec):
    """ln L of the new-physics spectrum alone on a binaryfog free spectrum."""
    return float(like.ln_like(spec(like.data.freqs)))


BF_LAST = {"theta": None}


def bf_lnL_fog(spec):
    """ln L with the holodeck population background of binaryfog, profiled over
    its six astrophysical parameters with the NANOGrav priors as a penalty (the
    same profile as for the power-law foreground). The maximum is found with
    Nelder-Mead from the better of the previous point's optimum and the best of
    64 prior draws; a maximum over random draws alone does not converge in six
    dimensions (256 draws move the region edge by a fifth of its width)."""
    from scipy.optimize import minimize
    pt = spec(BF_FOG.data.freqs)

    def neg(th):
        lp = bf_priors.ln_prior(th, "nanograv")
        if not np.isfinite(lp):
            return 1e30
        return -(BF_FOG.ln_like(pt, th) + lp)
    vals = np.array([BF_FOG.ln_like(pt, th) for th in BF_THETA]) + BF_LNPRIOR
    starts = [BF_THETA[int(np.argmax(vals))]]
    if BF_LAST["theta"] is not None:
        starts.append(BF_LAST["theta"])
    best = None
    for x0 in starts:
        r = minimize(neg, x0, method="Nelder-Mead",
                     options={"maxfev": 600, "xatol": 1e-3, "fatol": 1e-3})
        if best is None or r.fun < best.fun:
            best = r
    BF_LAST["theta"] = best.x
    return float(-best.fun)

STATE = {"spec": None}
H100_HZ = 100.0 * 1e3 / 3.0857e22          # 100 km/s/Mpc in Hz
FYR_HZ = 1.0 / (365.25 * 86400.0)


def _spec(f, **kw):
    """The transition spectrum plus, when A and gamma are passed, the power-law
    binary background in TransitionListener's convention (gwfopt.h2Omega_astro_noise)."""
    out = STATE["spec"](f)
    if "A" in kw:
        f = np.asarray(f, dtype=float)
        out = out + 2 * np.pi**2 / (3 * H100_HZ**2) * kw["A"]**2 * (f / FYR_HZ) ** (5 - kw["gamma"]) * FYR_HZ**2
    return out


Q_GRID = np.linspace(-2.5, 2.5, 7)


def prior_grid(mu, cov):
    """Nodes (log10 A, gamma) and the log prior (up to a constant) on a grid in
    the eigenbasis of the prior covariance."""
    ev, evec = np.linalg.eigh(np.array(cov, dtype=float))
    nodes, lnw = [], []
    for a in Q_GRID:
        for b in Q_GRID:
            nodes.append(np.array(mu) + evec @ (np.sqrt(ev) * np.array([a, b])))
            lnw.append(-0.5 * (a**2 + b**2))
    return np.array(nodes), np.array(lnw)


DATASETS = {"NG15": 14, "IPTA2": 13}
PTAS, BHB_NODES = {}, {}
for name, nb in DATASETS.items():
    PTAS[name] = signal_builder.ceffyl_builder({"config": PTAConfig(nb, name), "model": PTAModel(_spec)})
    # the population-synthesis prior on (log10 A, gamma): the foreground is
    # PROFILED, i.e. the likelihood is maximised over (log10 A, gamma) with the
    # prior as a penalty, on a grid over +-2.5 standard deviations of the prior
    BHB_NODES[name] = prior_grid(*signal_builder.bhb_priors[name])


def _powerlaw(f, log10A, gamma):
    f = np.asarray(f, dtype=float)
    return 2 * np.pi**2 / (3 * H100_HZ**2) * (10.0**log10A)**2 * (f / FYR_HZ) ** (5 - gamma) * FYR_HZ**2


def lnL(name, spec, smbhb=False, optimise=False):
    """Both data sets through binaryfog: NANOGrav the corrected 2026 free spectrum, IPTA DR2 the
    free spectrum of PTArcade (ipta2_cp_v1). With smbhb the (A, gamma) power law is profiled under the
    PTArcade population prior: on the 7 x 7 grid of prior_grid, and with optimise
    continued by Nelder-Mead from the best node (bhb_profile_check.py: the grid
    alone falls short by 0.05 in the median and 0.8 at most near the region edges)."""
    STATE["spec"] = spec
    like = BF_LIKE["cp_v2" if name == "NG15" else "ipta2"]; f = like.data.freqs; s = spec(f)
    if not smbhb:
        return float(like.ln_like(s))
    nodes, lnw = BHB_NODES[name]
    vals = np.array([like.ln_like(s + _powerlaw(f, *x)) for x in nodes]) + lnw
    post = lambda x: float(like.ln_like(s + _powerlaw(f, *x))) + _bhb_lnprior(name, x)
    best = float(np.max(vals))
    if optimise:
        from scipy.optimize import minimize
        r = minimize(lambda x: -post(x), nodes[int(np.argmax(vals))], method="Nelder-Mead",
                     options={"xatol": 1e-4, "fatol": 1e-4, "maxfev": 800})
        best = max(best, -float(r.fun))
    return best


def _bhb_lnprior(name, x):
    """ln of the population prior on (log10 A, gamma), up to the constant that prior_grid drops."""
    mu, cov = signal_builder.bhb_priors[name]
    d = np.asarray(x, dtype=float) - np.array(mu)
    return -0.5 * float(d @ np.linalg.solve(np.array(cov, dtype=float), d))


# ---------------------------------------------------------------- the scan
# GRID_SHIFT (a fraction of a cell, 0 by default) offsets both axes: the resolution check of the budget
_SHIFT = float(os.environ.get("GRID_SHIFT", "0"))
G = np.round(np.arange(0.39, 0.7501, 0.0025) + _SHIFT * 0.0025, 6)
# v from 3.2 MeV to 10 TeV, so that the posterior of every alternative lies inside the scanned range (a range of 0.01
# to 32 GeV cuts G1, G4 and G5; the ridge of the flat-space R^-2 fits along the edge of vacuum trapping fades between
# 2.5 and 5 TeV)
LOGV = np.round(np.arange(-2.5, 4.0001, 0.025) + _SHIFT * 0.025, 5)     # log10(v / GeV)
if os.environ.get("PLANE_SCAN_QUICK"):                  # a three-by-three grid, to test the machinery
    G, LOGV = G[[10, 14, 20]], LOGV[::25][:3]
V = 10.0 ** LOGV

VARIANT_LIST = []          # (group, name, transition key, cosmo)
for _pk in list(RATE_VARIANTS) + ["4D daisy"] + (["4D daisy RG"] if DTR is not None else []) + list(TRANSITION_COSMO):
    VARIANT_LIST.append(("cosmology" if _pk in TRANSITION_COSMO else "rate", _pk, _pk, COSMO_VARIANTS[0]))
for _c in COSMO_VARIANTS[1:]:
    VARIANT_LIST.append(("cosmology", _c[0], "nlo", _c))
# PLANE_SCAN_ONLY: a comma-separated list of variant names; only these rows are computed and written (to be inserted
# into a full scan, which keeps every other row as it was)
if os.environ.get("PLANE_SCAN_ONLY"):
    _only = [s.strip() for s in os.environ["PLANE_SCAN_ONLY"].split(",")]
    VARIANT_LIST = [v for v in VARIANT_LIST if v[1] in _only]
    assert [v[1] for v in VARIANT_LIST] == _only, ("unknown variant", _only)
EXTRA_KEYS = ("NG15_bins5", "NG15_bins30", "IPTA2_bins30", "IPTA2_bins5", "NG15_v1", "NG15_hd", "NG15_hd_v1", "NG15_dmgp",
              "NG15_binaryfog", "dneff_gw")
PAR_KEYS = ("T_reh", "T_p", "T_nuc", "betaH_R", "betaH_S", "beta_Gamma_H", "alpha", "dlnV_dt_over_H", "in_table",
            "window_limited", "T_f")
F_NEFF = np.logspace(-11, -5, 600)


def row(ig):
    """All likelihoods at one coupling G[ig] (one worker)."""
    g = G[ig]
    out = {ds + sfx: np.full((len(VARIANT_LIST), len(V)), np.nan) for ds in DATASETS for sfx in ("", "_bhb")}
    extra = {k: np.full(len(V), np.nan) for k in EXTRA_KEYS}
    par = {k: np.full(len(V), np.nan) for k in PAR_KEYS}
    for iv, lv in enumerate(LOGV):
        tpv = tp_variants(g, lv)
        par["T_reh"][iv] = tpv["nlo"][0]
        for key in PAR_KEYS[1:]:
            par[key][iv] = float(TT.get(key, g, lv))
        for k, (grp, name, pk, c) in enumerate(VARIANT_LIST):
            _, bh, alpha, bhv = tpv[pk]
            if not (np.isfinite(bh) and np.isfinite(alpha)):
                continue                           # the variant has no transition here
            reh = cosmo_inputs(c, g, lv, alpha)
            if not np.isfinite(reh[0]):
                continue
            spec = make_spectrum(c[1], reh, bh, alpha, bhv)
            if spec is None:
                continue
            for ds in DATASETS:
                out[ds][k, iv] = lnL(ds, spec)
                # the foreground row of the budget (B1) is the central one: profiled to convergence
                out[ds + "_bhb"][k, iv] = lnL(ds, spec, smbhb=True, optimise=(name == "nlo"))
            if name == "nlo":
                STATE["spec"] = spec
                extra["IPTA2_bins30"][iv] = bf_lnL(BF_LIKE["ipta2_30"], spec)
                extra["IPTA2_bins5"][iv] = bf_lnL(BF_LIKE["ipta2_5"], spec)
                extra["NG15_bins5"][iv] = bf_lnL(BF_LIKE["cp_v2_5"], spec)
                extra["NG15_bins30"][iv] = bf_lnL(BF_LIKE["cp_v2_30"], spec)
                extra["NG15_v1"][iv] = float(PTAS["NG15"].ln_likelihood(np.array([])))
                extra["NG15_hd"][iv] = bf_lnL(BF_LIKE["hd_v2"], spec)
                extra["NG15_hd_v1"][iv] = bf_lnL(BF_LIKE["hd_v1"], spec)
                extra["NG15_dmgp"][iv] = bf_lnL(BF_LIKE["hd_dmgp_v1"], spec)
                extra["NG15_binaryfog"][iv] = bf_lnL_fog(spec)
                # the contribution of the background itself to N_eff: 1.8e5 int h^2 Omega dln f
                extra["dneff_gw"][iv] = 1.8e5 * np.trapz(spec(F_NEFF), np.log(F_NEFF))
    return ig, out, extra, par


if __name__ == "__main__":
    from multiprocessing import get_context
    t0 = time.time()
    names = [v[1] for v in VARIANT_LIST]
    out = {ds + sfx: np.full((len(VARIANT_LIST), len(G), len(V)), np.nan) for ds in DATASETS for sfx in ("", "_bhb")}
    extra = {k: np.full((len(G), len(V)), np.nan) for k in EXTRA_KEYS}
    par = {k: np.full((len(G), len(V)), np.nan) for k in PAR_KEYS}
    with get_context("fork").Pool(min(ARGS.workers, len(G))) as pool:
        for n, (ig, o, e, p) in enumerate(pool.imap_unordered(row, range(len(G)))):
            for key in out:
                out[key][:, ig, :] = o[key]
            for key in extra:
                extra[key][ig] = e[key]
            for key in par:
                par[key][ig] = p[key]
            if n % 10 == 0:
                print(f"{n + 1}/{len(G)} couplings done, {time.time() - t0:.0f} s", flush=True)
    np.savez(ARGS.out, G=G, LOGV=LOGV, names=np.array(names), groups=np.array([v[0] for v in VARIANT_LIST]),
             **{"par_" + k: v for k, v in par.items()}, **out, **extra)
    print("wrote", ARGS.out, "in", round(time.time() - t0), "s")
