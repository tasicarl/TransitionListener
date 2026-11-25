"""Conformal U(1) without fermions: the four-dimensional daisy-resummed potential with fixed couplings, renormalised at
the field value phi = mu instead of phi = v.

Every field-dependent mass squared is proportional to phi^2, so the parent's logarithms ln(m_i^2(phi)/m_i^2(v)) are all
ln(phi^2/v^2): the Coleman-Weinberg potential renormalised at phi = v. This class takes ln(phi^2/mu^2) - 25/6 with
mu = mu_over_mv * v and fixes the quartic again by the condition that the zero-temperature minimum lies at phi = v:

    16 pi^2 lambda = k (10 lambda^2 + 3 g^4),    k = 11/3 + 2 ln(mu/v),

which is the parent's 110 lambda^2 - 48 pi^2 lambda + 33 g^4 = 0 at mu = v. The couplings do not run.
"""
import math

import numpy as np

import TL_conformal_u1_no_fermions as base4d


class specific_potential(base4d.specific_potential):
    """Conformal U(1), dark photon and dark Higgs: fixed couplings, logarithms at mu_over_mv times m_i(v)."""

    model_parameters = dict(base4d.specific_potential.model_parameters)
    model_parameters["mu_over_mv"] = {"default": 1.0, "plotname": r"$\mu/m(v)$", "min": 0, "max": np.inf}

    def init(self, inputparam_dict, verbose=True):
        """The parent at (g, v), with the logarithms and the quartic of the scale factor c."""
        self.mu_over_mv = float(inputparam_dict.get("mu_over_mv", 1.0))
        super().init({k: v for k, v in inputparam_dict.items() if k != "mu_over_mv"}, verbose)
        self.mp["mu_over_mv"] = {"value": self.mu_over_mv}
        self.renormScaleSq = self.mu_over_mv ** 2 * self.v * self.v
        self.derived_parameters = {"lambda": self.l}

    def calcSelfcouping(self):
        """The root of 10 k lambda^2 - 16 pi^2 lambda + 3 k g^4 = 0 that vanishes with k = 11/3 + 2 ln c."""
        k = 11.0 / 3.0 + 2.0 * math.log(self.mu_over_mv)
        disc = 256.0 * math.pi ** 4 - 120.0 * k * k * self.g ** 4
        if disc < 0:
            raise ValueError(f"no radiative symmetry breaking at mu/m(v) = {self.mu_over_mv}")
        # the root continuous through k = 0; negative for mu/m(v) < e^{-11/6}, where the logarithm stabilises the vacuum
        return 3.0 * k * self.g ** 4 / (8.0 * math.pi ** 2 + 0.5 * math.sqrt(disc))
