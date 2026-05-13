"""
This module contains the Higgs potential for a dark sector model with two singlet scalars
which can mix with each other, but not with the SM Higgs. The purpose of this model is to
study possible phase transition in a two-dimensional field space.

Part of TransitionListener v2.0
Documentation: https://tasillo.de/TransitionListener/

Authors:
    Jonas Matuszak <jonas.matuszak@kit.edu>
    Carlo Tasillo <carlo.tasillo@ific.uv.es>
"""

import numpy as np
from scipy import optimize
from transitionlistener import generic_potential
from transitionlistener import config
from transitionlistener import helper_functions
from transitionlistener import constants as cn
from transitionlistener.particles import MassSpectrum, Scalar, Fermion
from transitionlistener.errors import InitPotentialError



class DarkFlipFlop(generic_potential.generic_potential):
    """Dark flip-flop model with two interacting singlets."""

    model_parameters = {
        "lambda0": {
            "default": 0.1,
            "plotname": r"$\lambda_0$",
            "min": 0,
            "max": np.inf},
        "gamma": {
            "default": 1,
            "plotname": r"$\gamma$",
            "min": 0,
            "max": np.inf},
        "lambda1": {
            "default": 0.1,
            "plotname": r"$\lambda_1$",
            "min": 0,
            "max": np.inf},
        "lambda12": {
            "default": 0.1,
            "plotname": r"$\lambda_{12}$",
            "min": 0,
            "max": np.inf},
        "v_GeV": {
            "default": 0.1,
            "plotname": r"$v$ [GeV]",
            "min": 0,
            "max": np.inf},
        "y": {
            "default": 1,
            "plotname": r"$y$",
            "min": 0,
            "max": np.inf},
    }

    def init(self, inputparam_dict, verbose=True):
        """Initialise the flip-flop scalar potential with user-supplied parameters."""
        self.mp = self.set_modelparams(inputparam_dict)
        lambda0, gamma, lambda1, lambda12, v_GeV, y = [self.mp[name]["value"] for name in self.mp.keys()]
        self.retrieved_input = np.array([lambda0, gamma, lambda1, lambda12, v_GeV, y])
        self.Ndim = 2

        self.lambda0 = lambda0
        self.gamma = gamma
        self.lambda1 = lambda1
        self.lambda12 = lambda12
        self.v_GeV = v_GeV
        self.y = y
        self.verbose = verbose

        self.v_stable = 1000
        self.v = self.v_stable
        self.daisy = "ArnoldEspinosa"
        self.Z2_cutoff = -5
        self.conversionFactor = self.computeConversionFactor(self.v_stable, v_GeV)
        self.xi_nuc = 1

        self.v2_GeV2 = v_GeV**2

        self.v2 = self.v2_GeV2 / self.conversionFactor**2
        self.renormScaleSq = self.v2

        self.mass_spectrum = self._build_mass_spectrum()

        self.dmu1 = 0.0
        self.dmu2 = 0.0
        self.dlambda1 = 0.0
        self.dlambda2 = 0.0
        self.dlambda12 = 0.0
        self.calcCounterTerms()

        # The next line sets the field values of the absolute minimum.
        def fmin(x, t):
            return optimize.fmin(self.Vtot, x, args=(t,), xtol=self.x_eps, disp=False)

        # X0_1 = fmin([0, self.v], self.Tmin)
        # X0_2 = fmin([self.v, 0], self.Tmin)
        # Its better to use the analytical minima as a starting point here
        self.Tmin = cn.T0_SM_GeV / self.conversionFactor
        X0_1 = fmin([0, self.v/gamma], self.Tmin)
        X0_2 = fmin([self.v*np.sqrt(1 + lambda1/lambda0), 0], self.Tmin)
        V0_1 = self.Vtot(X0_1, self.Tmin)
        V0_2 = self.Vtot(X0_2, self.Tmin)
        self.X0 = X0_1 if V0_1 < V0_2 else X0_2

        bosons_Tmin = self.boson_massSq(self.X0, self.Tmin)
        self.m1sq_Tmin = float(bosons_Tmin[0][..., 0])
        self.m2sq_Tmin = float(bosons_Tmin[0][..., 1])
        self.m1_Tmin_GeV = np.sqrt(self.m1sq_Tmin) * self.conversionFactor
        self.m2_Tmin_GeV = np.sqrt(self.m2sq_Tmin) * self.conversionFactor
        self.derived_param_names = [name for name in config.all_observables.keys()]
        self.derived_param_plot_names = [config.all_observables[name] for name in self.derived_param_names]
        
        # if self.m1_Tmin_GeV < 0.2 or self.m2_Tmin_GeV < 0.2:
        #     # This is an ad-hoc cut to avoid very light scalars that are ruled out by
        #     # BBN DNeff constraints. Above 200 MeV they can decay to muons and are
        #     # hence safe.
        #     msg = ("The scalar masses are below 200 MeV "
        #            "at T=0: m1 = {:.3f} GeV, m2 = {:.3f} GeV.".format(
        #         self.m1_Tmin_GeV, self.m2_Tmin_GeV))
        #     raise InitPotentialError(msg)

        if self.verbose:
            input_dict = {
                "lambda0": self.lambda0,
                "gamma": self.gamma,
                "lambda1": self.lambda1,
                "lambda12": self.lambda12,
                "v / GeV": self.v_GeV
            }
            derived_dict = {
                "v / i.u.": self.v,
                "m1 / GeV": self.m1_Tmin_GeV,
                "m2 / GeV": self.m2_Tmin_GeV,
            }
            self.makePrettyDictionaryPrint(derived_dict)

    def setConfigParameters(self) -> None:
        """Change the default config parameters for tracing, hydrodynamics, and GW
        computation.
        """
        self.config.tracingConf.gen_mirror_phases = False
        self.config.tracingConf.tracing_field_accuracy = 1e-3
        # self.config.gwConf.weak_threshold = 5e-2  # This could be a bit low if alpha_nuc << alpha_perc
        self.config.gwConf.weak_threshold = 1e-4
        # self.config.tracingConf.internal_scale = 1000
        # self.config.tracingConf.Tmax_factor = 2.5
        # ...

    def calcCounterTerms(self, eps=0.1) -> None:
        """Compute the counter terms of the MS bar scheme."""
        v1 = self.v * np.sqrt(1 + self.lambda1/self.lambda0)

        def V_coleman_weinberg(X):
            bosons = self.boson_massSq(X, 0.0)
            fermions = self.fermion_massSq(X)
            return self.V1(bosons, fermions)

        # Take the derivative with larger \Delta\phi, this improves stability!
        dV1 = helper_functions.gradientFunction(
            # V_coleman_weinberg, eps=self.x_eps, Ndim=self.Ndim, order=self.deriv_order)
            V_coleman_weinberg, eps=eps, Ndim=self.Ndim, order=self.deriv_order)

        d2V1 = helper_functions.hessianFunction(
                # V_coleman_weinberg, eps=self.x_eps, Ndim=self.Ndim, order=self.deriv_order)
                V_coleman_weinberg, eps=eps, Ndim=self.Ndim, order=self.deriv_order)

        dVdx1 = dV1([v1, 0])[0]
        d2Vdx1 = d2V1([v1, 0])[0, 0]
        d2Vdx2 = d2V1([v1, 0])[1, 1]
        # d2Vdx2 = d2V1([0, v2])[1,1]

        self.dmu1 = 3/(2*v1) * dVdx1 - 0.5 * d2Vdx1
        self.dmu2 = d2Vdx2
        self.dlambda1 = 0.5/v1**3 * dVdx1 - 0.5/v1**2 * d2Vdx1
        self.dlambda2 = 0.0
        self.dlambda12 = 0.0

    def forbidPhaseCrit(self, X):
        """Return True when a candidate minimum should be excluded by symmetry."""
        Z2_1 = (np.array([X])[..., 0] < self.Z2_cutoff).any()
        Z2_2 = (np.array([X])[..., 1] < self.Z2_cutoff).any()
        return Z2_1 or Z2_2

    def V0(self, X):
        """Evaluate the tree-level potential for the scalar doublet."""
        X = np.asanyarray(X)
        phi1 = X[..., 0]
        phi2 = X[..., 1]
        r = .25*self.lambda0*(phi1**2 + self.gamma**2 * phi2**2 - self.v2)**2
        r += -.5 * self.lambda1 * self.v2 * phi1**2
        r += .5 * self.lambda12 * phi1**2 * phi2**2
        return r

    def Vct(self, X):
        r"""The counter terms in the MS bar scheme.

        .. math::
            `V_{ct} =  - \frac{\delta\mu_1^2}{2} \phi_1^{2} - \frac{\delta\mu_2^2}{2} \phi_{2}^{2} +
                       \frac{\delta\lambda_{1}}{4}\phi_{1}^{4}  +
                       \frac{\delta\lambda_{2}}{4}\phi_{2}^4 + \frac{\delta\lambda_{12}}{2}\phi_{1}^2\phi_2^2`
        """
        X = np.asarray(X)
        phi1 = X[..., 0]
        phi2 = X[..., 1]

        Vct = - 0.5*self.dmu1*phi1**2 - 0.5*self.dmu2*phi2**2
        Vct += 0.25*self.dlambda1*phi1**4 + 0.25*self.dlambda2*phi2**2
        Vct += 0.5*self.dlambda12 * phi1**2 * phi2**2
        return Vct

    def _build_mass_spectrum(self) -> MassSpectrum:

        scalar1 = Scalar(name="m_h1", latex_name=r"$m_{h_1}$", dof=1, is_SM=False)
        scalar2 = Scalar(name="m_h2", latex_name=r"$m_{h_2}$", dof=1, is_SM=False)

        def boson_mass_function(X, T):
            X = np.asarray(X)
            phi1 = X[..., 0]
            phi2 = X[..., 1]
            T_arr = np.asarray(T, dtype=float)
            T2 = T_arr * T_arr
            if T2.ndim == 0:
                T2 = T2 + np.zeros_like(phi1)

            a = 2 * self.lambda0 * phi1**2
            a += self.lambda0 * (phi1**2 + self.gamma**2 * phi2**2 - self.v2)
            a += -self.lambda1 * self.v2 + self.lambda12 * phi2**2

            b = 2 * self.gamma**4 * self.lambda0 * phi2**2
            b += self.lambda0 * self.gamma**2 * (phi1**2 + self.gamma**2 * phi2**2 - self.v2)
            b += self.lambda12 * phi1**2

            coeff1 = (self.y**2 + 2 * ((3 + self.gamma**2) * self.lambda0 + self.lambda12)) / 24.0
            coeff2 = (self.gamma**2 * self.lambda0 + 3 * self.gamma**4 * self.lambda0 + self.lambda12) / 12.0
            a += coeff1 * T2
            b += coeff2 * T2

            c = 2 * self.gamma**2 * self.lambda0 * phi1 * phi2 + 2 * self.lambda12 * phi1 * phi2

            A = 0.5 * (a + b)
            B = np.sqrt(0.25 * (a - b) ** 2 + c ** 2)
            m1sq = A + B
            m2sq = A - B

            return np.stack((m1sq, m2sq), axis=-1)

        fermion = Fermion(name="m_psi", latex_name=r"$m_{\psi}$", dof=2, is_SM=False)

        def fermion_mass_function(X):
            X = np.asarray(X)
            phi1 = X[..., 0]
            m2psi = 0.5 * self.y**2 * phi1**2
            return np.stack((m2psi,), axis=-1)

        spectrum = MassSpectrum([scalar1, scalar2], fermions=[fermion],
                                boson_massSq_fn=boson_mass_function,
                                fermion_massSq_fn=fermion_mass_function)

        return spectrum

    def approxZeroTMin(self):
        """Return approximate zero-temperature minima for the flip-flop model."""
        # There are generically two minima at zero temperature in this model,
        # and we want to include both of them.
        v = self.v2**.5
        v1 = v/self.gamma
        v2 = v*np.sqrt(1 + self.lambda1/self.lambda0)
        return [np.array([0, v1]), np.array([v2, 0])]
