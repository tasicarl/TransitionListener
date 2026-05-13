"""Template to implement the effective potential of a new model.

Part of TransitionListener v2.0
Documentation: https://tasillo.de/TransitionListener/

Authors:
    Jonas Matuszak <jonas.matuszak@kit.edu>
    Carlo Tasillo <carlo.tasillo@ific.uv.es>
"""

import numpy as np
from transitionlistener.generic_potential import generic_potential
import transitionlistener.constants as c
from transitionlistener import config
from transitionlistener.particles import MassSpectrum, Scalar, Goldstone, GaugeBoson, Fermion



# Rename the class to match your implemented model
class TemplatePotenital(generic_potential):
    """Template potential"""

    # Specify the 
    model_parameters = {
        "m1_GeV": {
            "default": 125.20,
            "plotname": r"$m_1$ [GeV]",
            "min": 0,
            "max": np.inf},
        "g": {
            "default": 0.5,
            "plotname": r"$g$",
            "min": 0,
            "max": 4*np.pi
        },
        "l": {
            "default": 1e-3,
            "plotname": r"$\lambda$",
            "min": 0,
            "max": 4*np.pi
        },
        "y": {
            "default": 0.1,
            "plotname": r"$y$",
            "min": 0,
            "max": np.sqrt(2)*4*np.pi,
        },
        "v_GeV": {
            "default": 246.22,
            "plotname": r"$v$ [GeV]",
            "min": 0,
            "max": np.inf},
    }

    def init(self, input_dict, verbose=True):
        """Perform the initialisation of the potential"""
        self.verbose = verbose

        # Specify the number of scalar field dimension:
        self.Ndim = 1

        # read input and set (default) values
        self.mp = self.set_modelparams(input_dict)
        self.m1_GeV = self.mp["m1_GeV"]["value"]
        self.v_GeV = self.mp["v_GeV"]["value"]
        self.y = self.mp["y"]["value"]
        self.l = self.mp["l"]["value"]
        self.g = self.mp["g"]["value"]

        # Use the vev in internal units, normalised to 1000 by default
        self.v = self.v_stable 
        self.conversionFactor = self.computeConversionFactor(self.v_stable, self.v_GeV)
        self.daisy = "ArnoldEspinosa"  # or: "Parwani" or "off"

        # Set any parameters derived from the input
        self.mu2 = -2 * self.v**2
        self.mu2_GeV = - 2 * self.v_GeV**2

        # Set the T = 0 minima, this has to have the shape (1, Ndim)
        self.X0 = np.array([self.v])

        # Build particle spectrum definition used throughout the potential
        self.mass_spectrum = self.build_mass_spectrum()

        # Set the renormalization scale
        self.renormScaleSq = self.v**2  # In internal units
        self.renormScaleSq_GeV2 = self.renormScaleSq * self.conversionFactor**2.  # In GeV
        self.renormScale_GeV = np.sqrt(self.renormScaleSq) * self.conversionFactor

        # Impose the renormalization conditions on the potential.
        dV1 = self.dV1atvev()
        d2V1 = self.d2V1atvev()
        self.dmu2 = 3./2. * dV1 / self.v - 1./2. * d2V1
        self.dmu2_GeV2 = self.dmu2 * self.conversionFactor**2.
        self.dl = 1./2. * dV1 / self.v**3. - 1./2. * d2V1 / self.v**2.

        # Get the zero temperature masses in the global vev
        bosonMassSq_vev, _, _, _ = self.boson_massSq(self.X0, 0.0)

        # Set the background degrees of freedom:
        # The defaults includes the SM degrees of freedom, i.e. the sector undergoing the
        # phase transition is coupled to the SM during the phase transition.
        # self.kin_coupled_e_geff = e_geffSM  # This can be overwritten
        # self.kin_coupled_p_geff = p_geffSM  # This can be overwritten

        # This specifies additional radiation that is not coupled to the sector undergoing
        # a phase transition. If e.g. a dark sector is completely decoupled from the SM,
        # put the SM geff here. 
        # self.kin_decoupled_e_geff = lambda T, cf: 0.0 * T  # If the SM is decoupled, put e_geffSM here
        # self.kin_decoupled_p_geff = lambda T, cf: 0.0 * T  # If the SM is decoupled, put p_geffSM here

        # Include the input and derived parameters in the class.
        self.derived_param_names = [name for name in config.all_observables.keys()]
        self.derived_param_plot_names = [config.all_observables[name] for name in self.derived_param_names]

        if self.verbose:
            input_dict = dict(
                vev_GeV=self.v_GeV,
                vev_iu=self.v,
                quartic_coupling=self.l,
                gauge_coupling=self.g,
                y=self.y,
                conversion_factor_GeV_per_iu=self.conversionFactor
                )
            derived_dict = dict(
                mu2_GeV2=self.mu2_GeV2,
                dmu2_GeV2=self.dmu2_GeV2,
                dl=self.dl,
                renormScale_GeV=self.renormScale_GeV,
                renormScaleSq=self.renormScaleSq)
            self.makePrettyDictionaryPrint(input_dict, derived_dict)

    def setConfigParameters(self) -> None:
        """Change the default config parameters for tracing, hydrodynamics, and GW
        computation.
        """
        self.config.tracingConf.gen_mirror_phases = False
        self.config.tracingConf.tracing_field_accuracy = 1e-3  # for Hessian
        # self.config.tracingConf.internal_scale = 1000
        # self.config.tracingConf.Tmax_factor = 2.5
        # ...

            
    def forbidPhaseCrit(self, X):
        """
        forbidPhaseCrit is used since there is a Z2 symmetry in the theory and
        we don't want to double-count all of the phases.
        """
        return (np.array([X])[..., 0] < self.Z2_cutoff).any()

    def V0(self, X: np.ndarray) -> np.ndarray:
        """ This method defines the tree-level potential."""
        X = np.asanyarray(X)

        phi1 = X[..., 0]
        # For Ndims > 1, the other field directions are exctracted as
        # phi2 = X[..., 1]
        # phi3 = X[..., 2]

        # Compute the contributions to the tree level potential with the
        # model parameters
        V = - self.mu2*(phi1**2.)/2. + self.l*(phi1**4.)/4.

        # Potentially other field directions contribute, e.g.:
        # V += - self.mu2_2 * phi2**2/2 + self.l_2 * phi2**4/4.

        return V

    def Vct(self, X: np.ndarray) -> np.ndarray:
        """ The counterterm lagrangian is the same as the tree level lagrangian but
        with masses and couplings replaced by counter term values (i.e. here
        mu2 -> dmu2 and l -> dl).
        """
        X = np.array(X)
        phi1 = X[..., 0]
        # For Ndims > 1, the other field directions are exctracted as
        # phi2 = X[..., 1]
        # phi3 = X[..., 2]

        # Compute the contribution from the counter terms
        V = - self.dmu2*(phi1**2.)/2. + self.dl*(phi1**4.)/4.
        # Potentially other field directions contribute, e.g.:
        # V += - self.dmu2_2 * phi2**2/2 + self.dl_2 * phi2**4/4.

        return V

    def build_mass_spectrum(self) -> MassSpectrum:
        """This function constructs the mass spectrum by specifying
        the particle content and the mass functions."""

        # First the scalar particles have to be specified.
        scalar_particles = [
            Scalar(name="m_phi", latex_name=r"$m_\phi$", dof=1, is_SM=False),
            Goldstone(name="m_varphi", latex_name=r"$m_\varphi$", dof=1, is_SM=False),
        ]

        # For the gauge bosons, split the longitudinal and transversal
        #  components due to the different thermal masses
        gauge_particles = [
            GaugeBoson(name="m_ZT", latex_name=r"$m_{Z^\prime, T}$", dof=2, gauge_coupling=self.g, is_SM=False),
            GaugeBoson(name="m_ZL", latex_name=r"$m_{Z^\prime, L}$", dof=1, gauge_coupling=self.g, is_SM=False),
        ]

        def boson_mass_function(X, T):
            """This returns the squared masses of the bosonic particles."""
            X = np.asarray(X)
            T = np.asarray(T)

            phi = X[..., 0]
            phi_sq = phi*phi

            Pi_phi = (self.l/3.0 + self.g**2 / 4.0 + self.y**2/12) * T*T # Hard thermal masses scalar
            Pi_Z = (self.g**2/3.0) * T*T  # Hard thermal masses gauge boson

            mPhi_sq = - self.mu2 + 3.0 * self.l * phi_sq
            mVarphi_sq = -self.mu2 + self.l * phi_sq
            mZT_sq = self.g**2 * phi_sq
            mZL_sq = self.g**2 * phi_sq

            # Four entries
            Nbosons = 4
            M2 = np.empty(phi.shape + (Nbosons,))
            # This has to have the same order as the
            # scalar_particles
            M2[..., 0] = mPhi_sq + Pi_phi
            M2[..., 1] = mVarphi_sq + Pi_phi
            # and the same order as the gauge_particles
            M2[..., 2] = mZT_sq
            M2[..., 3] = mZL_sq + Pi_Z

            # Add additional bosonic particles:
            # M2[..., 4] = ...
            # M2[..., 5] = ...

            return M2

        # specify the fermionic particles:
        fermion_particles = [
            Fermion(name="m_chi", latex_name=r"$m_chi$", dof=4, is_SM=False),
        ]

        def fermion_mass_function(X):
            """Compute the squared masses of the fermionic particles."""
            X = np.asarray(X)
            phi = X[..., 0]
            mChi = self.y**2/2.0 * phi**2
            return np.stack((mChi,), axis=-1)

        # Construct the complete mass spectrum contributing to the potential
        spectrum = MassSpectrum(scalars=scalar_particles,
                                gaugeBosons=gauge_particles,
                                fermions=fermion_particles,
                                boson_massSq_fn=boson_mass_function,
                                fermion_massSq_fn=fermion_mass_function)

        return spectrum

    def approxZeroTMin(self):
        """Return approximate zero-temperature minima."""
        return self.X0
