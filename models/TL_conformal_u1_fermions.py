"""
This module contains the Higgs potential for a conformal dark
U(1) model with a dark photon, a dark Higgs and two dark fermions.
``TL_conformal_u1_no_fermions`` is its special case without the fermions.

Field content (arXiv:2502.19478 eq. 2.1): a complex dark Higgs of U(1)
charge +1, the U(1) gauge boson, and TWO left-handed Weyl fermions
chi_1, chi_2 carrying OPPOSITE charges Q = -1/2 and +1/2, coupled to the
dark Higgs by Majorana-type Yukawa terms (y/2) Phi chibar^c chi. The
opposite charges are what makes the model anomaly free; a single Dirac
fermion with chiral charges (+1/2, -1/2) would give sum q^3 = 1/4 and
sum q = 1 and would be anomalous. Each Weyl fermion carries 2 degrees of
freedom and mass^2 = y^2 phi^2 / 2.

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
    """Conformal U(1) with fermions model potential class."""

    #: U(1) charges of the left-handed Weyl fermions, two of opposite charge
    #: (see the module docstring). A subclass without fermions sets ``()``;
    #: the spectrum, the one-loop potential and the thermal masses follow.
    fermion_charges = (-1/2, +1/2)

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
        """Initialise the conformal U(1) with fermions model from input parameters."""
        # Set the values of the model parameters and check if they are in range
        self.mp = self.set_modelparams(inputparam_dict)
        g = self.mp["g"]["value"]
        y = self.mp["y"]["value"] if "y" in self.mp else 0.0
        v_GeV = self.mp["v_GeV"]["value"]

        self.Ndim = 1  # Number of scalars

        # Fermion parameters: charges of the two left-handed Weyl fermions.
        # NOT the left/right components of one Dirac fermion -- see the module
        # docstring. Only the sum of squares enters below.
        self.Q_chi = tuple(self.fermion_charges)
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
        self.config.gwConf.bw_collisions = "NLO"  # the walls of a gauge theory can run away at large alpha
        self.config.tracingConf.tracing_field_accuracy = 1e-3  # for Hessian
        # The bounce action of this nearly flat, radiatively generated potential is
        # sensitive to the spline of V along the bounce path. With the default 1000
        # samples the interpolation error biases Tperc, Treh and alpha towards strong
        # supercooling, scatters them along smooth parameter lines, and makes some
        # strongly supercooled points fail. 10000 samples agree with the unsplined
        # potential, at about twice the cost per point.
        self.config.tracingConf.tunneling_params["V_spline_samples"] = 10000

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

        if not self.Q_chi or self.y == 0.0:
            # massless fermions add nothing; the expression below would be 0 * log(0/0) = nan
            return y / (64 * np.pi * np.pi)
        m2f, nf = fermions
        y2 = np.array([self.y**2 / 2.0, self.y**2 / 2.0], dtype=float)
        y -= np.sum(
            nf * m2f * m2f * (np.log(np.abs(m2f / (y2 * self.renormScaleSq)) + 1e-100) - 25.0 / 6.0),
            axis=-1,
        )

        return y / (64 * np.pi * np.pi)

    def debye_massSq(self, X, T):
        """The thermal part of the boson masses squared, exactly rather than by subtraction.

        These are the same ``T^2`` terms that ``boson_mass_function`` adds to the tree-level
        masses: the scalar prefactor for the two scalar modes, nothing for the transverse
        gauge boson, and the longitudinal prefactor for the longitudinal one. Handing them to
        the daisy term directly avoids taking the difference of two masses squared that agree
        to within one part in ``1e9`` at low temperature.
        """
        import numpy as _np
        X = _np.asarray(X)
        T2 = _np.asarray(T, dtype=float) ** 2
        scalar_prefactor = self.l / 3.0 + self.g**2 / 4.0 + self.y**2 / 12.0
        # the same Debye mass as boson_mass_function, for the charges of this model (none in
        # the fermion-free TL_conformal_u1_no_fermions)
        longitudinal_prefactor = (1.0 / 3.0 + sum(q**2 for q in self.Q_chi) / 6.0) * self.g**2
        shape = _np.broadcast(X[..., 0], T2).shape
        Pi = _np.zeros(shape + (4,), dtype=float)
        Pi[..., 0] = scalar_prefactor * T2
        Pi[..., 1] = scalar_prefactor * T2
        Pi[..., 2] = 0.0
        Pi[..., 3] = longitudinal_prefactor * T2
        return Pi

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

            # DRalgo, PrintScalarMass["LO"]:
            #   msq3d = msq + T^2 ( lambda/3 + g^2/4 + |y|^2/12 )
            scalar_prefactor = self.l / 3.0 + self.g**2 / 4.0 + self.y**2 / 12.0
            # m_D^2 = g^2 T^2 [ (1/3) sum_complexScalar Q^2 + (1/6) sum_Weyl Q^2 ]
            #       = (1/3 + 1/12) g^2 T^2 = (5/12) g^2 T^2
            # DRalgo, PrintDebyeMass["LO"].
            sum_Q_sq = sum(q**2 for q in self.Q_chi)
            longitudinal_prefactor = (1.0 / 3.0 + sum_Q_sq / 6.0) * self.g**2

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

        # Two left-handed Weyl fermions (2 dof each), degenerate. The output
        # columns m_psi1/m_psi2 are chi_1 and chi_2 of the module docstring.
        GaugeBosons = [dark_photon_T,
                       dark_photon_L]
        Scalars = [phi_particle, goldstone]
        if not self.Q_chi:
            return MassSpectrum(Scalars, GaugeBosons, [],
                                boson_massSq_fn=boson_mass_function)
        chi_1 = Fermion(name="m_psi1", latex_name=r"$m_{\chi_1}$", dof=2, is_SM=False)
        chi_2 = Fermion(name="m_psi2", latex_name=r"$m_{\chi_2}$", dof=2, is_SM=False)
        Fermions = [chi_1, chi_2]

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
