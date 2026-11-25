"""Conformal U(1) without fermions: the four-dimensional daisy-resummed potential with running couplings.

The MS-bar one-loop potential in Landau gauge, with the couplings of the input scale mu0 run at one loop to the
temperature-dependent scale mu = mu_over_piT * pi T (arXiv:2511.02910, whose four-dimensional reference scale is
mu = pi T):

    V = lambda phi^4/4 + sum_i n_i m_i^4/(64 pi^2) [ln(m_i^2/mu^2) - c_i] + T^4/(2 pi^2) sum_i n_i J_B(m_i^2/T^2)
        - T/(12 pi) sum_i n_i [(m_i^2 + Pi_i)^{3/2} - (m_i^2)^{3/2}],

c_i = 3/2 for the dark Higgs and the Goldstone mode, 5/6 for the three polarisations of the dark photon, with
m_phi^2 = 3 lambda phi^2, m_chi^2 = lambda phi^2, m_A'^2 = g^2 phi^2, Pi_phi = Pi_chi = (lambda/3 + g^2/4) T^2 and
Pi_A'L = g^2 T^2/3, and g = g(mu), lambda = lambda(mu). The input is that of the three-dimensional rate: g at mu0
and lambda(mu0) = 0, with mu0 = e^{-1/6} g v. The field is the one renormalised at mu; the anomalous dimension
does not enter a bounce solved in that field.

The scale follows the temperature only, so the potential is meant for the barrier, at field values of a few T. At
the broken minimum, phi = v >> T, the logarithms ln(g v/pi T) are large. The phases, the vacuum energy and the
thermodynamics are therefore those of TL_conformal_u1_no_fermions (``Vtot``), and only the bounce is solved on the
potential above (``V_rg``), as in ``TL_conformal_u1_fermions_rg``.
"""
import math

import numpy as np

import TL_conformal_u1_no_fermions as base4d
import dr_conformal_u1_no_fermions_matching as matching
from TL_conformal_u1_fermions_rg import RunningCouplingV
from transitionlistener.eft_rate.running_coupling_bounce import RunningCouplingBounce
from transitionlistener.finiteT import Jb_spline


class specific_potential(RunningCouplingV, RunningCouplingBounce, base4d.specific_potential):
    """Conformal U(1), dark photon and dark Higgs: running couplings at mu = mu_over_piT * pi T."""

    model_parameters = dict(base4d.specific_potential.model_parameters)
    model_parameters["mu_over_piT"] = {"default": 1.0, "plotname": r"$\mu/\pi T$", "min": 0, "max": np.inf}

    def init(self, inputparam_dict, verbose=True):
        """The fixed-coupling model of the parent, plus the input scale and the scale factor."""
        super().init({k: v for k, v in inputparam_dict.items() if k != "mu_over_piT"}, verbose)
        self.mp["mu_over_piT"] = {"value": float(inputparam_dict.get("mu_over_piT", 1.0))}
        self.mu_over_piT = self.mp["mu_over_piT"]["value"]
        self.mu0 = math.exp(-1.0 / 6.0) * self.g * self.v      # internal units, lambda(mu0) = 0
        self._couplings = {}

    def couplings(self, T):
        """(g^2, lambda, mu) at mu = mu_over_piT * pi T, T in internal units."""
        T = float(T)
        if T not in self._couplings:
            mu = self.mu_over_piT * math.pi * T
            g, lam, _ = matching.rg_run(self.g, 0.0, self.mu0, mu)
            self._couplings[T] = (g * g, lam, mu)
        return self._couplings[T]

    def _V_rg(self, phi, T):
        g2, lam, mu = self.couplings(T)
        p2 = phi * phi
        modes = ((1.0, 3.0 * lam * p2, 1.5, (lam / 3.0 + g2 / 4.0) * T * T),
                 (1.0, lam * p2, 1.5, (lam / 3.0 + g2 / 4.0) * T * T),
                 (2.0, g2 * p2, 5.0 / 6.0, 0.0),
                 (1.0, g2 * p2, 5.0 / 6.0, g2 * T * T / 3.0))
        y = lam / 4.0 * p2 * p2
        for n, m2, c, Pi in modes:
            m2c = m2.astype(complex)
            with np.errstate(divide="ignore", invalid="ignore"):
                cw = np.where(m2 != 0.0, m2c * m2c * (np.log(m2c / mu ** 2) - c), 0.0)
            y = y + n * np.real(cw) / (64.0 * math.pi ** 2)
            if T > 0.0:
                y = y + n * T ** 4 / (2.0 * math.pi ** 2) * Jb_spline(m2 / (T * T))
                y = y - n * T / (12.0 * math.pi) * np.real((m2c + Pi) ** 1.5 - m2c ** 1.5)
        return y
