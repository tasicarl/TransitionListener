"""
This module contains the Higgs potential for a dark
U(1) model with a dark photon and a dark Higgs.

Part of TransitionListener v2.0
Documentation: https://tasillo.de/TransitionListener/

Authors:
    Jonas Matuszak <jonas.matuszak@kit.edu>
    Carlo Tasillo <carlo.tasillo@ific.uv.es>
"""

import numpy as np
from transitionlistener import generic_potential
from transitionlistener.constants import *
from transitionlistener import config
from transitionlistener.particles import MassSpectrum, Scalar, Goldstone, GaugeBoson

class specific_potential(generic_potential.generic_potential):
    """Dark U(1) model variant with corrected Debye masses."""

    model_parameters = {
        "g_tilde": {
            "default": 1,
            "plotname": r"$g\cdot \lambda^{-1/4}$",
            "min": 0,
            "max": 4 * np.pi},
        "l": {
            "default": 0.1,
            "plotname": r"$\lambda$",
            "min": 0,
            "max": 4 * np.pi},
        "v_GeV": {
            "default": 1,
            "plotname": r"$v$ [GeV]",
            "min": 0,
            "max": np.inf}
    }

    def init(self, inp, verbose=True):
        """Initialise the model parameters and thermal masses.""" 
        self.mp = self.set_modelparams(inp)
        g_tilde, l, v_GeV = [self.mp[name]["value"] for name in self.mp.keys()]
        g = g_tilde * np.power(l, 1/4)
        self.retrieved_input = np.array([g, l, v_GeV])
        self.Ndim = 1
        self.v_GeV = v_GeV
        self.verbose = verbose
        
        self.internal_scale = 1000
        self.v = self.internal_scale
        self.X0 = [self.v]
        self.daisy = "ArnoldEspinosa"
        self.Z2_cutoff = -5
        self.conversionFactor = self.computeConversionFactor(self.internal_scale, v_GeV)
        
        self.g = g
        self.l = l
        self.mu2 = self.l * self.v**2.
        self.mu = np.sqrt(self.mu2)
        self.mu_GeV = self.mu * self.conversionFactor
        self.v2 = self.v**2.
        self.mDP = self.g * self.v
        self.mDP_GeV = self.mDP * self.conversionFactor
        self.mDP2 = self.g**2. * self.v**2
        self.mDH = np.sqrt(-self.mu2 + 3.*self.l*self.v**2.)
        self.mDH_GeV = self.mDH * self.conversionFactor
        self.renormScaleSq = self.v2

        self.mass_spectrum = self._build_mass_spectrum()

        # Calculate counter mass and counter coupling
        self.x_eps = 0.1
        dV1 = self.dV1atvev()
        d2V1 = self.d2V1atvev()
        self.x_eps = 0.1
        self.dmu2 = 3./2. * dV1 / self.v - 1./2. * d2V1
        self.dmu2_GeV2 = self.dmu2 * self.conversionFactor**2.
        self.dl = 1./2. * dV1 / self.v**3. - 1./2. * d2V1 / self.v**2.

        self.derived_param_names = [name for name in config.all_observables.keys()]
        self.derived_param_plot_names = [config.all_observables[name] for name in self.derived_param_names]

        if self.verbose:
            input_dict = dict(vev_GeV=self.v_GeV, quartic_coupling=self.l,
                              gauge_coupling=self.g, conversion_factor=self.conversionFactor)
            derived_dict = dict(mDP_GeV=self.mDP_GeV, mDH_GeV=self.mDH_GeV,
                                mu2_GeV2=self.mu_GeV**2,
                                dmu2_GeV2=self.dmu2_GeV2, dl=self.dl,
                                renormScaleSq=self.renormScaleSq)
            self.makePrettyDictionaryPrint(derived_dict)

    def setConfigParameters(self) -> None:
        """Change the default config parameters for tracing, hydrodynamics, and GW
        computation.
        """

        # Adjust to allow for weak phase transitions as well!
        self.config.tracingConf.approx_strength_threshold = 1e-7
        self.config.gwConf.weak_threshold = 1e-7
        self.config.percolationConf.n_action = 30

    def forbidPhaseCrit(self, X):
        """
        forbidPhaseCrit is used since there is a Z2 symmetry in the theory and
        we don't want to double-count all of the phases.
        """
        return (np.array([X])[..., 0] < -5.).any()

    def V0(self, X):
        """
        This method defines the tree-level potential.
        """
        X = np.asanyarray(X)
        phi = X[...,0]
        r = - self.mu2*(phi**2.)/2. + self.l*(phi**4.)/4.
        return r
    
    def Vct(self, X):
        """
        The counterterm lagrangian is the same as the tree level lagrangian but
        with masses and couplings replaced by counter term values (i.e. here
        mu2 -> dmu2 and l -> dl). Assume potential of the form
        V = - 1/2 mu**2 h**2 + lambda/4 h**4 where h is the investigated scalar field
        """
        X = np.array(X)
        phi = X[...,0]
        r = - self.dmu2*(phi**2.)/2. + self.dl*(phi**4.)/4.
        return r


    def _build_mass_spectrum(self) -> MassSpectrum:
        """Construct the mass spectrum for the corrected Debye-mass variant."""

        scalar_prefactor = self.l / 3.0 + self.g * self.g / 4.0
        longitudinal_prefactor = (1.0 / 3.0) * self.g * self.g

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
            phi = X[..., 0]
            phi_sq = phi * phi
            even = -self.mu2 + 3.0 * self.l * phi_sq
            odd = -self.mu2 + self.l * phi_sq
            transverse = self.g * self.g * phi_sq
            longitudinal = transverse

            Nbosons = 4
            M2 = np.empty(phi.shape + (Nbosons,))
            M2[..., 0] = even + scalar_prefactor * T*T
            M2[..., 1] = odd + scalar_prefactor * T*T
            M2[..., 2] = transverse
            M2[..., 3] = longitudinal + longitudinal_prefactor * T * T
            
            return M2


        scalars = [phi_particle, goldstone]
        gauge_bosons = [dark_photon_T, dark_photon_L]
        spectrum = MassSpectrum(scalars=scalars, gaugeBosons=gauge_bosons,
                                boson_massSq_fn=boson_mass_function)

        return spectrum

    def approxZeroTMin(self):
        """Return approximate zero-temperature minima for the corrected model."""
        # There are generically two minima at zero temperature in this model,
        # and we want to include both of them.
        v = self.v2**.5
        return [np.array([v])]

   
