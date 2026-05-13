"""
This module contains the Higgs potential for a conformal dark
U(1) model with a dark photon, a dark Higgs and a dark fermion.

Part of TransitionListener v2.0
Documentation: https://tasillo.de/TransitionListener/

Authors:
    Jonas Matuszak <jonas.matuszak@kit.edu>
    Carlo Tasillo <carlo.tasillo@ific.uv.es>
"""

import numpy as np
from transitionlistener import generic_potential
from transitionlistener.errors import InitPotentialError
from transitionlistener import config
from transitionlistener.particles import MassSpectrum, Scalar, GaugeBoson, Fermion, Goldstone


class specific_potential(generic_potential.generic_potential):
    """Conformal dark U(1) model potential class."""

    model_parameters = {
        "g": {
            "default": 1,
            "plotname": r"$g$",
            "min": 0,
            "max": 4 * np.pi},
        "y": {
            "default": 0.1,
            "plotname": r"$y$",
            "min": 0,
            "max": 4 * np.pi},
        "v_GeV": {
            "default": 1,
            "plotname": r"$v$ [GeV]",
            "min": 0,
            "max": np.inf}
    }

    def init(self, inputparam_dict, verbose=True):
        """Initialise the conformal dark U(1) model from input parameters."""
        # Set the values of the model parameters and check if they are in range
        self.mp = self.set_modelparams(inputparam_dict)
        g, y, v_GeV = [self.mp[name]["value"] for name in self.mp.keys()]

        self.Ndim = 1  # Number of scalars

        # Fermion parameters:
        self.Q_l = 1/2
        self.Q_r = -1/2
        self.g = g
        self.y = y
        self.v_GeV = v_GeV
        self.verbose = verbose

        # Parameters
        self.v_GeV = v_GeV
        self.daisy = "ArnoldEspinosa"
        self.v_stable = 1000
        self.Z2_cutoff = -5
        self.X0 = np.array([self.v_stable])
        self.conversionFactor = self.computeConversionFactor(self.v_stable, v_GeV)

        self.l = self.calcSelfcouping()
        self.v = self.v_stable
        self.renormScaleSq = self.v * self.v
        self.mass_spectrum = self._build_mass_spectrum()
        self.derived_parameters = {"lambda": self.l}

    def calcSelfcouping(self):
        """Return the quartic coupling consistent with radiative symmetry breaking."""
        l = 1/55*(12*np.pi**2 - np.sqrt((-1815*self.g**4 + 288*np.pi**4 + 605*self.y**4)/2))
        if l < 0:
            raise InitPotentialError(
                "The Higgs quartic coupling is negative, because the Yukawa " +
                "coupling was chosen too large! The Higgs potential is not bounded from below.")
        return l

    def setConfigParameters(self) -> None:
        """Change the default config parameters for tracing, hydrodynamics, and GW
        computation.

        ``algorithm_mode`` (adaptive_step_size / fixed_step_size) is left at
        the ``PercolationConf`` default and can be set per-run from the YAML.
        """
        self.config.tracingConf.gen_mirror_phases = False
        self.config.tracingConf.tracing_field_accuracy = 1e-3  # for Hessian

    def V0(self, X):
        """
        This method defines the tree-level potential.
        """
        X = np.asanyarray(X)
        v = X[..., 0]
        r = self.l * (v**4.) / 4.
        return r

    def V1(self, bosons, fermions) -> np.ndarray:
        """One-loop corrections to the zero-temperature potential."""
        m2, n, _, _ = bosons

        la2 = np.array([3 * self.l, self.l, self.g**2, self.g**2], dtype=float)
        y = np.sum(
            n * m2 * m2 * (np.log(np.abs(m2 / (la2 * self.renormScaleSq)) + 1e-100) - 25.0 / 6.0),
            axis=-1,
        )

        m2f, nf = fermions
        y2 = np.array([self.y**2 / 2.0, self.y**2 / 2.0], dtype=float)
        y -= np.sum(
            nf * m2f * m2f * (np.log(np.abs(m2f / (y2 * self.renormScaleSq)) + 1e-100) - 25.0 / 6.0),
            axis=-1,
        )

        return y / (64 * np.pi * np.pi)

    def _build_mass_spectrum(self) -> MassSpectrum:

        phi_particle = Scalar(
            name="m_phi",
            latex_name=r"$m_\phi$",
            dof=1,
            is_SM=False,
        )
        goldstone = Goldstone(
            name="m_varphi",
            latex_name=r"$m_\varphi$",
            dof=1,
            is_SM=False,
        )
        dark_photon_T = GaugeBoson(
            name="m_A_T",
            latex_name=r"$m_{A^\prime_T}$",
            dof=2,
            gauge_coupling=self.g,
            is_SM=False,
        )
        dark_photon_L = GaugeBoson(
            name="m_A_L",
            latex_name=r"$m_{A^\prime_L}$",
            dof=1,
            gauge_coupling=self.g,
            is_SM=False,
        )

        def boson_mass_function(X, T):
            X = np.asarray(X)
            T2 = T * T

            scalar_prefactor = self.l / 3.0 + self.g**2 / 4.0 + self.y**2 / 12.0
            longitudinal_prefactor = (1.0 / 3.0 + (self.Q_l**2 + self.Q_r**2) / 6.0) * self.g**2

            phi = X[..., 0]
            phi_sq = phi * phi
            even = 3.0 * self.l * phi_sq
            odd = self.l * phi_sq
            mDP2 = self.g**2 * phi_sq
            mDP2L = mDP2

            Nbosons = 4
            M2 = np.empty(phi.shape + (Nbosons,))
            M2[..., 0] = even + scalar_prefactor * T2
            M2[..., 1] = odd + scalar_prefactor * T2
            M2[..., 2] = mDP2
            M2[..., 3] = mDP2L + longitudinal_prefactor * T2
            return M2


        def fermion_mass_function(X):

            X = np.asarray(X)
            phi = X[..., 0]
            msq = 0.5 * self.y**2 * phi**2

            Nfermions = 2
            M2 = np.empty(phi.shape +  (Nfermions,))
            M2[..., 0] = msq
            M2[..., 1] = msq

            return M2

        fermion_L = Fermion(name="m_psi1", latex_name=r"$m_{\psi_1}$", dof=2, is_SM=False)
        fermion_R = Fermion(name="m_psi2", latex_name=r"$m_{\psi_2}$", dof=2, is_SM=False)

        GaugeBosons = [dark_photon_T,
                       dark_photon_L]
        Scalars = [phi_particle, goldstone]
        Fermions = [fermion_L, fermion_R]

        spectrum = MassSpectrum(Scalars, GaugeBosons, Fermions,
                                boson_massSq_fn=boson_mass_function,
                                fermion_massSq_fn=fermion_mass_function)

        return spectrum

    def forbidPhaseCrit(self, X):
        """
        forbidPhaseCrit is used since there is a Z2 symmetry in the theory and
        we don't want to double-count all of the phases.
        """
        return (np.array([X])[..., 0] < self.Z2_cutoff).any()
