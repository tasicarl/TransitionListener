"""
Particle physics and cosmological quantities used across TransitionListener.

The numerical values listed here follow the PDG conventions and are expressed
in natural units whenever possible.  Additional model-specific inputs (SM
fermion masses, gauge boson pole masses, Wolfenstein CKM parameters, ...)
are collected in dedicated containers so that individual models can reuse them.

Part of TransitionListener v2.0
Documentation: https://tasillo.de/TransitionListener/

Authors:
    Jonas Matuszak <jonas.matuszak@kit.edu>
    Carlo Tasillo <carlo.tasillo@ific.uv.es>
"""

from __future__ import annotations

import scipy.constants as sc

yr = sc.Julian_year  #: One Julian year in seconds (:math:`365.25` days).
Mpl_GeV = 1.220910e19  #: Planck mass in GeV.
Omega_gamma_h2 = 2.473e-5  #: Photon energy density today in units of :math:`h^2`.
T0_SM_GeV = 2.35253655e-13  #: CMB temperature :math:`T_0 = 2.725\,\\mathrm{K}` in GeV.
GeV_Hz = 1.0 / 6.582119e-25  #: Conversion between GeV and Hz.
H100_Hz = 100 / 3.0857e19  #: Hubble constant :math:`H_0 = 100\,\\mathrm{km/s/Mpc}` in Hz.
Lambda = 2.533e-47  #: Dark-energy density :math:`\\rho_{\\Lambda}` in GeV^4.
fyr_Hz = 1 / yr  #: Frequency corresponding to one inverse year in Hz.

# ---------------------------------------------------------------------------
# Standard Model inputs
# ---------------------------------------------------------------------------

SM_FERMION_MASSES_GEV = {
    "m_e": 0.510998928e-3,
    "m_mu": 0.1056583715,
    "m_tau": 1.77682,
    "m_u": 0.1,
    "m_c": 1.51,
    "m_t": 172.5,
    "m_d": 0.1,
    "m_s": 0.1,
    "m_b": 4.92,
}
"""Pole masses of the Standard Model fermions in GeV.

These values serve as the central inputs
for all Yukawa couplings appearing in TransitionListener models.  They are
provided as a plain dictionary so that models can access them directly or feed
them into symbolic pipelines.
"""

SM_GAUGE_MASSES_GEV = {
    "mW": 80.385,
    "mZ": 91.1876,
}
"""Pole masses of the electroweak gauge bosons in GeV."""

CW_CONSTANTS = {
    "scalar": 3.0 / 2.0,
    "fermion": 3.0 / 2.0,
    "gauge": 5.0 / 6.0,
}
"""Coleman–Weinberg subtraction constants for scalars, fermions, and vectors."""
