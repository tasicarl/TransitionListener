import math
import numpy as np
import sys
from TransitionListener import generic_potential
from TransitionListener import geff
import matplotlib.pyplot as plt

class dp_eft(generic_potential.generic_potential):
    """
    A child of :func:`generic_potential.generic_potential` which implements the 
    particle physics model. Here, a dark photon-dark higgs model is implemented.
    The dark sector is completely decoupled and only contributes by changing the
    effective number of relativistic degrees of freedom g_eff. The examined
    scalar field is the real part of the U(1)_Dark Higgs field. 

    The following attributes are set during initialiation:

    Attributes
    ----------
    self.Ndim : int
        The number of dynamic field dimensions in this model; is 1.
    self.l : float
        The quartic coupling of the Higgs potential
    self.mu2 : float
        The quared mass parameterin the Higgs potential
    self.mu : float
        The mass parameter in potential
    self.v : float
        The tree level vev of the potential. The renormalization scheme was
        chosen such that this vev is conserved when radiative corrections are
        being included.
    self.v2 : float
        The squared tree level vev of the potential.
    self.g : float
        The dark photon gauge coupling. This parameter scales up the potential barrier
        by giving mass to the dark photon.
    self.mDP: float
        The dark photon mass after the phase transition
    self.mDP2 : float
        The squared dark photon mass after the phase transition
    self.renormScaleSq : float
        The squared renormalization scale that is going to be used in the calculation
        of radiative corrections. The vev is chosen as the renormalization scale to
        conserve its value when changing the temperature.
    self.conversionFactor: float
        The conversionFactor is the factor between GeV and the chosen energy scale. In
        this case it is chose to be 1e-6 since we want to calculate on a keV-scale.
    self.Tmax : float
        The maximum temperature in TransitionListener's analysis. The value should in any
        case be higher than the highest temperature at which the inflection point (from
        which the concurring minimum stems) occurs.
    self.verbose : Boolean
        This switch can be used to (de)activate text output during calculation.
    self.daisyResum: int
        Used to change between the different resummation method for daisy graphs:
        daisyResum = 0: no resummation; daisyResum = 1: all modes resummed;
        daisyResum = 2: only 0 modes resummed. The case 1 is the so-called the Pawani
        method while the case 2 is the Arnold-Espinoza method.
    self.xi_nuc : float
        The ratio between the temperatures of the Standard Model sector and a potential
        Dark Sector T_DS / T_SM at the bubble nucleation. For a discussion of its application see arXiv:
        1811.11175v2, for the dynamics of the ratio in the evolution of the universe
        see 1203.5803.
    """

    def init(self, l=0.01, g=0.7, v=1000., xi_nuc=1, Gamma_GeV=1, conversionFactor=1e-6, daisyType=2, verbose=True, output_folder="", g_DS_until_nuc=4, is_DS_dof_const_before_PT=True, dilution_plots=False):
        self.Ndim = 1
        self.x_eps = .001
        self.T_eps = .001
        self.deriv_order = 4

        self.l = l
        self.g = g
        self.v = v
        self.xi_nuc = xi_nuc
        self.Gamma_GeV = Gamma_GeV
        self.conversionFactor = conversionFactor
        self.daisyResum = daisyType
        self.verbose = verbose
        self.output_folder = output_folder
        self.g_DS_until_nuc = g_DS_until_nuc
        self.is_DS_dof_const_before_PT = is_DS_dof_const_before_PT
        self.dilution_plots = dilution_plots
        
        self.v_GeV = self.v * self.conversionFactor
        self.mu2 = self.l * self.v**2.
        self.mu = np.sqrt(self.mu2)
        self.mu_GeV = self.mu * self.conversionFactor
        self.v2 = self.v**2.
        self.mDP = self.g * self.v
        self.mDP_GeV = self.mDP * self.conversionFactor
        self.mDP2 = self.g**2. * self.v**2
        self.mDH = np.sqrt(-self.mu2 + 3.*self.l*v**2.)
        self.mDH_GeV = self.mDH * self.conversionFactor
        self.renormScaleSq = self.v2
        self.Tmax = 2.5 * self.v

        # Calculate counter mass and counter coupling
        dV1 = self.dV1atvev()
        d2V1 = self.d2V1atvev()
        vev = self.v
        self.dmu2 = 3./2. * dV1 / vev - 1./2. * d2V1
        self.dmu2_GeV2 = self.dmu2 * self.conversionFactor**2.
        self.dl = 1./2. * dV1 / vev**3. - 1./2. * d2V1 / vev**2.

        self.g_eff_instance = geff.effective_dof()
        self.bosons0 = self.boson_massSq(np.array([self.v]), 0.)
        self.fermions = self.fermion_massSq(np.array([self.v]))

        if self.verbose:
            from scipy import optimize
            print("\n")
            print("INPUT PARAMETERS")
            names = ["vev/GeV", "quartic coupling", "gauge coupling", "T ratio at nucleation", "Gamma/GeV", "conversionFactor"]
            values = [self.v * self.conversionFactor, self.l, self.g, self.xi_nuc, self.Gamma_GeV, self.conversionFactor]
            for n, v in zip(names, values):
                print("{:<30} {:<25}".format(n, v))
            print("\nDERIVED PARAMETERS")
            names = ["tree-level mass mu/GeV", "dark photon mass mDP/GeV", "dark Higgs mass m_DH/GeV", "counterterm mass dmu2/GeV^2", "counterterm coupling dl"]
            values = [self.mu_GeV, self.mDP_GeV,self.mDH_GeV, self.dmu2_GeV2, self.dl]
            for n, v in zip(names, values):
                print("{:<30} {:<25}".format(n, v))
            print("\n")

    def forbidPhaseCrit(self, X):
        """
        forbidPhaseCrit is used since there is a Z2 symmetry in the theory and
        we don't want to double-count all of the phases.
        """
        return (np.array([X])[...,0] < -5.).any()
        
                        
    def V0(self, X):
        """
        This method defines the tree-level potential.
        """
        X = np.asanyarray(X)
        v = X[...,0]
        r = - self.mu2*(v**2.)/2. + self.l*(v**4.)/4.
        return r

    def Vct(self, X):
        """
        The counterterm lagranian is the same as the tree level lagrangian but
        with masses and couplings replaced by counter term values (i.e. here
        mu2 -> dmu2 and l -> dl). Assume potential of the form
        V = - 1/2 mu**2 h**2 + lambda/4 h**4 where h is the investigated scalar field
        """
        X = np.array(X)
        v = X[...,0]
        r = - self.dmu2*(v**2.)/2. + self.dl*(v**4.)/4.
        return r
        
    def boson_massSq(self, X, T):
        """
        This moethod defines the squared boson mass spectrum of this theory. It is
        returned with the respective fields' dofs and renormalization constants.
        If the daisies are resummed
        """
        X = np.array(X)
        v = X[...,0]
        g = self.g
        mDP2 = np.array([(g**2.)*(v**2.)])

        if self.daisyResum == 1 or 2:
            # Temperature corrections for scalar (i.e. Higgs)
            # "Even" and "Odd" denote the CP-symmetry factor
            cS = self.l / 3. + g**2 / 4.
            cDP = 2. / 3. * g**2.

            MSqEven = np.array([-self.mu2 + cS*T**2. + 3.*self.l*v**2.])
            MSqOdd = np.array([-self.mu2 + cS*T**2. + self.l*v**2.])

            # Temperature corrections for longitudinal gauge bosons (i.e. dark photon)
            mDP2L = mDP2 + cDP*T*T
        else:
            MSqEven = np.array([-self.mu2 + 3.*self.l*v**2.])
            MSqOdd = np.array([-self.mu2 + self.l*v**2.])
            mDP2L = mDP2 

        M = np.concatenate((MSqEven, MSqOdd, mDP2, mDP2L))
        M = np.rollaxis(M, 0, len(M.shape))
        dof = np.array([1, 1, 2, 1])
        # c_i = 3/2 for fermions and scalars, 5/6 for gauge bosons
        c = np.array([3./2., 3./2., 5./6., 5./6.])
        is_physical = np.array([1, 0, 1, 1])
        return M, dof, c, is_physical

    def fermion_massSq(self, X):
        """
        Since fermions are not included in this theory, their mass spectrum and
        dofs are set to zero.
        """
        M = 0
        dof = 0
        return M, dof

    def Vtot(self, X, T, include_radiation=True):
        """
        The total finite temperature effective potential is calculated by adding
        up the tree level potential, the one-loop-zero-T correction, the respective
        counter terms, and (depending on the daisy resummation scheme) the one-loop-
        temperature-dependent corrections.
        
        Parameters
        ----------
        X : array_like
            Field value(s). 
            Either a single point (with length `Ndim`), or an array of points.
        T : float or array_like
            The temperature. The shapes of `X` and `T`
            should be such that ``X.shape[:-1]`` and ``T.shape`` are
            broadcastable (that is, ``X[...,0]*T`` is a valid operation).
        include_radiation : bool, optional
            If False, this will drop all field-independent radiation
            terms from the effective potential. Useful for calculating
            differences or derivatives.
        """
        T = np.asanyarray(T, dtype=float)
        X = np.asanyarray(X, dtype=float)

        # Without finite-T self energy
        bosons0 = self.boson_massSq(X,0.*T)
        m20, nb, c, _ = bosons0
        # With finite-T self energy
        bosonsT = self.boson_massSq(X,T)
        m2T, nbT, cT, _ = bosonsT
        fermions = self.fermion_massSq(X)
        y = self.V0(X)
        y += self.V1(bosons0, fermions)
        y += self.Vct(X)

        # Parwani (1992) prescription. All modes resummed.
        if self.daisyResum == 1:
            y += self.V1T(bosonsT, fermions, T, include_radiation)
        # Carrington (1992), Arnold and Espinosa (1992) prescription. Zero modes only.
        elif self.daisyResum == 2:
            # Absolute values are a hack. Potential trustworthy only away from where m2T, m20 < 0.  
            Vdaisy = np.real(-(T/(12.*np.pi))*np.sum ( nb*(pow(m2T+0j,1.5) - pow(m20+0j,1.5)), axis=-1))
            y += self.V1T(bosons0, fermions, T, include_radiation) + Vdaisy
        # No daisy resummation
        elif self.daisyResum == 0:
            y += self.V1T(bosons0, fermions, T, include_radiation)

        return y
      
    def approxZeroTMin(self):
        # There are generically two minima at zero temperature in this model, 
        # and we want to include both of them.
        v = self.v2**.5
        return [np.array([v])]
