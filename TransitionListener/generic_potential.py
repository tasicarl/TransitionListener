"""
The primary task of the generic_potential module is to define the
:class:`generic_potential` class, from which realistic scalar field models can
straightforwardly be constructed. The most important part of any such model is,
appropiately, the potential function and its gradient. This module is not
necessary to define a potential, but it does make the process somewhat simpler
by automatically calculating one-loop effects from a model-specific mass
spectrum, constructing numerical derivative functions, providing a
simplified interface to the :mod:`.transitionFinder` module, and providing
several methods for plotting the potential and its phases.
"""
import numpy as np
from scipy import optimize

from .finiteT import Jb_spline as Jb
from .finiteT import Jf_spline as Jf
from . import transitionFinder
from . import helper_functions
from . import observability
from . import dilution

import math
import numpy as np
import sys

import matplotlib.pyplot as plt


class generic_potential(object):
    """
    An abstract class from which one can easily create finite-temperature
    effective potentials.

    This class acts as the skeleton around which different scalar field models
    can be formed. At a bare minimum, subclasses must implement :func:`init`,
    :func:`V0`, and :func:`boson_massSq`. Subclasses will also likely implement
    :func:`fermion_massSq` and :func:`approxZeroTMin`. Once the tree-level
    potential and particle spectrum are defined, the one-loop zero-temperature
    potential (using MS-bar renormalization) and finite-temperature potential
    can be used without any further modification.

    If one wishes to rewrite the effective potential from scratch (that is,
    using a different method to calculate one-loop and finite-temperature
    corrections), this class and its various helper functions can still be used.
    In that case, one would need to override :func:`Vtot` (used by most of the
    helper functions) and :func:`V1T_from_X` (which should only return the
    temperature-dependent part of Vtot; used in temperature derivative
    calculations), and possibly override :func:`V0` (used by
    :func:`massSqMatrix` and for plotting at tree level).

    The `__init__` function performs initialization specific for this abstract
    class. Subclasses should either override this initialization *but make sure
    to call the parent implementation*, or, more simply, override the
    :func:`init` method. In the base implementation, the former calls the latter
    and the latter does nothing. At a bare minimum, subclasses must set the
    `Ndim` parameter to the number of dynamic field dimensions in the model.

    One of the main jobs of this class is to provide an easy interface for
    calculating the phase structure and phase transitions. These are given by
    the methods :func:`getPhases`, :func:`calcTcTrans`, and
    :func:`findAllTransitions`.

    The following attributes can (and should!) be set during initialiation:

    Attributes
    ----------
    Ndim : int
        The number of dynamic field dimensions in the model. This *must* be
        overridden by subclasses during initialization.
    x_eps : float
        The epsilon to use in brute-force evalutations of the gradient and
        for the second derivatives. May be overridden by subclasses;
        defaults to 0.001.
    T_eps : float
        The epsilon to use in brute-force evalutations of the temperature
        derivative. May be overridden by subclasses; defaults to 0.001.
    deriv_order : int
        Sets the order to which finite difference derivatives are calculated.
        Must be 2 or 4. May be overridden by subclasses; defaults to 4.
    renormScaleSq : float
        The square of the renormalization scale to use in the MS-bar one-loop
        zero-temp potential. May be overridden by subclasses;
        defaults to 1000.0**2.
    Tmax : float
        The maximum temperature to which minima should be followed. No
        transitions are calculated above this temperature. This is also used
        as the overall temperature scale in :func:`getPhases`.
        May be overridden by subclasses; defaults to 1000.0.
    num_boson_dof : int or None
        Total number of bosonic degrees of freedom, including radiation.
        This is used to add a field-independent but temperature-dependent
        contribution to the effective potential. It does not affect the relative
        pressure or energy density between phases, so it does not affect the
        critical or nucleation temperatures. If None, the total number of
        degrees of freedom will be taken directly from :meth:`boson_massSq`.
    num_fermion_dof : int or None
        Total number of fermionic degrees of freedom, including radiation.
        If None, the total number of degrees of freedom will be taken
        directly from :meth:`fermion_massSq`.
    """
    def __init__(self, *args, **dargs):
        self.num_boson_dof = self.num_fermion_dof = None
        self.phases = self.transitions = None  # These get set by getPhases
        self.TcTrans = None  # Set by calcTcTrans()
        self.TnTrans = None  # Set by calcFullTrans()
        self.GWTrans = None  # Set by calcGWTrans()

        self.init(*args, **dargs)

        if self.Ndim <= 0:
            raise ValueError("The number of dimensions in the potential must "
                             "be at least 1.")

    def init(self, *args, **dargs):
        """
        Subclasses should override this method (not __init__) to do all model
        initialization. At a bare minimum, subclasses need to specify the number
        of dimensions in the potential with ``self.Ndim``.
        """
        pass

    # EFFECTIVE POTENTIAL CALCULATIONS -----------------------

    def V0(self, X):
        """
        The tree-level potential. Should be overridden by subclasses.

        The input X can either be a single point (with length `Ndim`), or an
        arbitrarily shaped array of points (with a last axis again having shape
        `Ndim`). Subclass implementations should be able to handle either case.
        If the input is a single point, the output should be scalar. If the
        input is an array of points, the output should be an array with the same
        shape (except for the last axis with shape `Ndim`).
        """
        return X[...,0]*0

    def boson_massSq(self, X, T):
        """
        Calculate the boson particle spectrum. Should be overridden by
        subclasses.

        Parameters
        ----------
        X : array_like
            Field value(s).
            Either a single point (with length `Ndim`), or an array of points.
        T : float or array_like
            The temperature at which to calculate the boson masses. Can be used
            for including thermal mass corrrections. The shapes of `X` and `T`
            should be such that ``X.shape[:-1]`` and ``T.shape`` are
            broadcastable (that is, ``X[...,0]*T`` is a valid operation).

        Returns
        -------
        massSq : array_like
            A list of the boson particle masses at each input point `X`. The
            shape should be such that
            ``massSq.shape == (X[...,0]*T).shape + (Nbosons,)``.
            That is, the particle index is the *last* index in the output array
            if the input array(s) are multidimensional.
        degrees_of_freedom : float or array_like
            The number of degrees of freedom for each particle. If an array
            (i.e., different particles have different d.o.f.), it should have
            length `Ndim`.
        c : float or array_like
            A constant used in the one-loop zero-temperature effective
            potential. If an array, it should have length `Ndim`. Generally
            `c = 1/2` for gauge boson transverse modes, and `c = 3/2` for all
            other bosons.
        """
        # The following is an example placeholder which has the correct output
        # shape. Since dof is zero, it does not contribute to the potential.
        #Nboson = 2
        #phi1 = X[...,0]
        #phi2 = X[...,1] # Comment out so that the placeholder doesn't
                         # raise an exception for Ndim < 2.
        #m1 = .5 * phi1**2 + .2 * T**2  # First boson mass
        #m2 = .6 * phi1**2  # Second boson mass, no thermal mass correction
        #massSq = np.empty(m1.shape + (Nboson,))  # Important to make sure that
            # the shape comes from m1 and not m2, since the addition of the
            # thermal mass correction could change the output shape (if, for
            # example, T is an array and X is a single point).
        #massSq[...,0] = m1
        #massSq[...,1] = m2
        #dof = np.array([0.,0.])
        #sc = np.array([0.5, 1.5])
        return massSq, dof, c

    def fermion_massSq(self, X):
        """
        Calculate the fermion particle spectrum. Should be overridden by
        subclasses.

        Parameters
        ----------
        X : array_like
            Field value(s).
            Either a single point (with length `Ndim`), or an array of points.

        Returns
        -------
        massSq : array_like
            A list of the fermion particle masses at each input point `X`. The
            shape should be such that  ``massSq.shape == (X[...,0]).shape``.
            That is, the particle index is the *last* index in the output array
            if the input array(s) are multidimensional.
        degrees_of_freedom : float or array_like
            The number of degrees of freedom for each particle. If an array
            (i.e., different particles have different d.o.f.), it should have
            length `Ndim`.

        Notes
        -----
        Unlike :func:`boson_massSq`, no constant `c` is needed since it is
        assumed to be `c = 3/2` for all fermions. Also, no thermal mass
        corrections are needed.
        """
        # The following is an example placeholder which has the correct output
        # shape. Since dof is zero, it does not contribute to the potential.
        #Nfermions = 2
        #phi1 = X[...,0]
        #phi2 = X[...,1] # Comment out so that the placeholder doesn't
                         # raise an exception for Ndim < 2.
        #m1 = .5 * phi1**2  # First fermion mass
        #m2 = .6 * phi1**2  # Second fermion mass
        #massSq = np.empty(m1.shape + (Nfermions,))
        #massSq[...,0] = m1
        #massSq[...,1] = m2
        #dof = np.array([0.,0.])
        massSq = 0
        dof = 0
        return massSq, dof

    def V1(self, bosons, fermions):
        """
        The one-loop corrections to the zero-temperature potential
        using MS-bar renormalization.

        This is generally not called directly, but is instead used by
        :func:`Vtot`.
        """
        # This does not need to be overridden.
        m2, n, c, _ = bosons
        y = np.sum(n*m2*m2 * (np.log(np.abs(m2/self.renormScaleSq) + 1e-100) - c), axis=-1)

        m2, n = fermions
        c = 3./2.
        y -= np.sum(n*m2*m2 * (np.log(np.abs(m2/self.renormScaleSq) + 1e-100) - c), axis=-1)

        return y/(64*np.pi*np.pi)

    def Vct(self):
        """
        The one-loop counterterm potential for MS-bar renormalization.
        This should be definitely overwritten in the model file.
        """
        r = 0

        return r

    def V1T(self, bosons, fermions, T, include_radiation=True):
        """
        The one-loop finite-temperature potential.

        This is generally not called directly, but is instead used by
        :func:`Vtot`.

        Note
        ----
        The `Jf` and `Jb` functions used here are
        aliases for :func:`finiteT.Jf_spline` and :func:`finiteT.Jb_spline`,
        each of which accept mass over temperature *squared* as inputs
        (this allows for negative mass-squared values, which I take to be the
        real part of the defining integrals.

        .. todo::
            Implement new versions of Jf and Jb that return zero when m=0, only
            adding in the field-independent piece later if
            ``include_radiation == True``. This should reduce floating point
            errors when taking derivatives at very high temperature, where
            the field-independent contribution is much larger than the
            field-dependent contribution.
        """
        # This does not need to be overridden.
        T2 = (T*T)[..., np.newaxis] + 1e-100
             # the 1e-100 is to avoid divide by zero errors
        T4 = T*T*T*T
        m2, nb, c, _ = bosons
        y = np.sum(nb*Jb(m2/T2), axis=-1)
        m2, nf = fermions
        y += np.sum(nf*Jf(m2/T2), axis=-1)
        if include_radiation:
            if self.num_boson_dof is not None:
                nb = self.num_boson_dof - np.sum(nb)
                y -= nb * np.pi**4 / 45.
            if self.num_fermion_dof is not None:
                nf = self.num_fermion_dof - np.sum(nf)
                y -= nf * 7*np.pi**4 / 360.
        return y*T4/(2*np.pi*np.pi)

    def V1T_from_X(self, X, T, include_radiation=True):
        """
        Calculates the mass matrix and resulting one-loop finite-T potential.

        Useful when calculate temperature derivatives, when the zero-temperature
        contributions don't matter.
        """
        T = np.asanyarray(T, dtype=float)
        X = np.asanyarray(X, dtype=float)
        bosons = self.boson_massSq(X,T)
        fermions = self.fermion_massSq(X)
        y = self.V1T(bosons, fermions, T, include_radiation)
        return y

    def Vtot(self, X, T, include_radiation=True):
        """
        The total finite temperature effective potential.

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
        bosons = self.boson_massSq(X,T)
        fermions = self.fermion_massSq(X)
        y = self.V0(X)
        y += self.V1(bosons, fermions)
        y += self.V1T(bosons, fermions, T, include_radiation)
        return y

    def DVtot(self, X, T):
        """
        The finite temperature effective potential, but offset
        such that V(0, T) = 0.
        """
        #X0 = np.zeros(self.Ndim)
        return self.Vtot(X,T,False) - self.Vtot(X*0,T,False)

    def gradV(self, X, T):
        """
        Find the gradient of the full effective potential.

        This uses :func:`helper_functions.gradientFunction` to calculate the
        gradient using finite differences, with differences
        given by `self.x_eps`. Note that `self.x_eps` is only used directly
        the first time this function is called, so subsequently changing it
        will not have an effect.
        """
        try:
            f = self._gradV
        except:
            # Create the gradient function
            self._gradV = helper_functions.gradientFunction(
                self.Vtot, self.x_eps, self.Ndim, self.deriv_order)
            f = self._gradV
        # Need to add extra axes to T since extra axes get added to X in
        # the helper function.
        T = np.asanyarray(T)[...,np.newaxis,np.newaxis]
        return f(X,T,False)

    def dV1atvev(self):
        """
        Find the first derivative of the one loop zero temperature potential,
        evaluated at the vev. Used in the function Vct to calculate the
        counter term lagrangian

        This uses :func:`helper_functions.gradientFunction` to calculate the
        gradient using finite differences, with differences
        given by `self.x_eps`. Note that `self.x_eps` is only used directly
        the first time this function is called, so subsequently changing it
        will not have an effect.
        """

        try:
            res = self._dV1atvev

        except:
            # vev means here an arbitrary point in field space
            def V_coleman_weinberg(X):
                bosonsvev0 = self.boson_massSq(X,0.)
                fermionsvev = self.fermion_massSq(X)
                return self.V1(bosonsvev0, fermionsvev)

            dV1 = helper_functions.gradientFunction(V_coleman_weinberg, eps = self.x_eps, Ndim = self.Ndim, order = self.deriv_order)

            # Use np.squeeze to shrink array sizes from (1,1) and (1,) to a 0d
            # array... otherways problems when evaluating Vtot at more than one
            # point in field space at the same time in the code.
            
            self._dV1atvev = np.squeeze(dV1(np.array([self.v])))
            res = self._dV1atvev
        return res

    def dgradV_dT(self, X, T):
        """
        Find the derivative of the gradient with respect to temperature.

        This is useful when trying to follow the minima of the potential as they
        move with temperature.
        """
        T_eps = self.T_eps
        try:
            gradVT = self._gradVT
        except:
            # Create the gradient function
            self._gradVT = helper_functions.gradientFunction(
                self.V1T_from_X, self.x_eps, self.Ndim, self.deriv_order)
            gradVT = self._gradVT
        # Need to add extra axes to T since extra axes get added to X in
        # the helper function.
        T = np.asanyarray(T)[...,np.newaxis,np.newaxis]
        assert (self.deriv_order == 2 or self.deriv_order == 4)
        if self.deriv_order == 2:
            y = gradVT(X,T+T_eps,False) - gradVT(X,T-T_eps,False)
            y *= 1./(2*T_eps)
        else:
            y = gradVT(X,T-2*T_eps,False)
            y -= 8*gradVT(X,T-T_eps,False)
            y += 8*gradVT(X,T+T_eps,False)
            y -= gradVT(X,T+2*T_eps,False)
            y *= 1./(12*T_eps)
        return y

    def massSqMatrix(self, X):
        """
        Calculate the tree-level mass matrix of the scalar field.

        This uses :func:`helper_functions.hessianFunction` to calculate the
        matrix using finite differences, with differences
        given by `self.x_eps`. Note that `self.x_eps` is only used directly
        the first time this function is called, so subsequently changing it
        will not have an effect.

        The resulting matrix will have rank `Ndim`. This function may be useful
        for subclasses in finding the boson particle spectrum.
        """
        try:
            f = self._massSqMatrix
        except:
            # Create the gradient function
            self._massSqMatrix = helper_functions.hessianFunction(
                self.V0, self.x_eps, self.Ndim, self.deriv_order)
            f = self._massSqMatrix
        return f(X)

    def d2V(self, X, T):
        """
        Calculates the Hessian (second derivative) matrix for the
        finite-temperature effective potential.

        This uses :func:`helper_functions.hessianFunction` to calculate the
        matrix using finite differences, with differences
        given by `self.x_eps`. Note that `self.x_eps` is only used directly
        the first time this function is called, so subsequently changing it
        will not have an effect.
        """
        try:
            f = self._d2V
        except:
            # Create the gradient function
            self._d2V = helper_functions.hessianFunction(
                self.Vtot, self.x_eps, self.Ndim, self.deriv_order)
            f = self._d2V
        # Need to add extra axes to T since extra axes get added to X in
        # the helper function.
        T = np.asanyarray(T)[...,np.newaxis]
        return f(X,T, False)

    def d2V1atvev(self):
        """
        Find the second derivative of the one loop zero temperature potential,
        evaluated at the vev. Used in the function Vct to calculate the
        counter term lagrangian

        This uses :func:`helper_functions.gradientFunction` to calculate the
        gradient using finite differences, with differences
        given by `self.x_eps`. Note that `self.x_eps` is only used directly
        the first time this function is called, so subsequently changing it
        will not have an effect.
        """

        try:
            res = self._d2V1atvev
        except:
            # vev means here an arbitrary point in field space
            def V_coleman_weinberg(X):
                bosonsvev0 = self.boson_massSq(X,0.)
                fermionsvev = self.fermion_massSq(X)
                return self.V1(bosonsvev0, fermionsvev)

            d2V1 = helper_functions.hessianFunction(V_coleman_weinberg, eps = self.x_eps, Ndim = self.Ndim, order = self.deriv_order)

            # Use np.squeeze to shrink array sizes from (1,1) and (1,) to a 0d
            # array... otherways problems when evaluating Vtot at more than one
            # point in field space at the same time in the code.
            self._d2V1atvev = np.squeeze(d2V1(np.array([self.v])))
            res = self._d2V1atvev
        return res


    def energyDensity(self,X,T,include_radiation=True): #actually not an energy density but rather a heat density, that is p - T del p / del T, such that Delta energyDensity is latent heat.
        T_eps = self.T_eps
        if self.deriv_order == 2:
            dVdT = self.V1T_from_X(X,T+T_eps, include_radiation)
            dVdT -= self.V1T_from_X(X,T-T_eps, include_radiation)
            dVdT *= 1./(2*T_eps)
        else:
            dVdT = self.V1T_from_X(X,T-2*T_eps, include_radiation)
            dVdT -= 8*self.V1T_from_X(X,T-T_eps, include_radiation)
            dVdT += 8*self.V1T_from_X(X,T+T_eps, include_radiation)
            dVdT -= self.V1T_from_X(X,T+2*T_eps, include_radiation)
            dVdT *= 1./(12*T_eps)
        V = self.Vtot(X,T, include_radiation)
        return V - T*dVdT

    # MINIMIZATION AND TRANSITION ANALYSIS --------------------------------

    def approxZeroTMin(self):
        """
        Returns approximate values of the zero-temperature minima.

        This should be overridden by subclasses, although it is not strictly
        necessary if there is only one minimum at tree level. The precise values
        of the minima will later be found using :func:`scipy.optimize.fmin`.

        Returns
        -------
        minima : list
            A list of points of the approximate minima.
        """
        # This should be overridden.
        return [np.ones(self.Ndim)*self.renormScaleSq**.5]

    def findMinimum(self, X=None, T=0.0):
        """
        Convenience function for finding the nearest minimum to `X` at
        temperature `T`.
        """
        if X is None:
            X = self.approxZeroTMin()[0]
        return optimize.fmin(self.Vtot, X, args=(T,), disp=0)

    def findT0(self):
        """
        Find the temperature at which the high-T minimum disappears. That is,
        find lowest temperature at which Hessian matrix evaluated at the origin
        has non-negative eigenvalues.

        Notes
        -----
        In prior versions of CosmoTransitions, `T0` was used to set the scale
        in :func:`getPhases`. This became problematic when `T0` was zero, so in
        this version `self.Tmax` is used as the scale. This function is now not
        called directly by anything in the core CosmoTransitions package, but
        is left as a convenience for subclasses.
        """
        X = self.findMinimum(np.zeros(self.Ndim), self.Tmax)
        f = lambda T: min(np.linalg.eigvalsh(self.d2V(X,T)))
        if f(0.0) > 0:
            # barrier at T = 0
            T0 = 0.0
        else:
            T0 = optimize.brentq(f, 0.0, self.Tmax)
        return T0

    def forbidPhaseCrit(self, X):
        """
        Returns True if a phase at point `X` should be discarded,
        False otherwise.

        The default implementation returns False. Can be overridden by
        subclasses to ignore phases. This is useful if, for example, there is a
        Z2 symmetry in the potential and you don't want to double-count all of
        the phases.

        Notes
        -----
        In previous versions of CosmoTransitions, `forbidPhaseCrit` was set to
        None in `__init__`, and then if a subclass needed to forbid some region
        it could set ``self.forbidPhaseCrit = lambda x: ...``. Having this
        instead be a proper method makes for cleaner code.

        The name "forbidPhaseCrit" is supposed to be short for "critera for
        forbidding a phase". Apologies for the name mangling; I'm not sure why
        I originally decided to leave off the "eria" in "criteria", but I should
        leave it as is for easier backwards compatability.
        """
        return False

    def getPhases(self,tracingArgs={}):
        """
        Find different phases as functions of temperature

        Parameters
        ----------
        tracingArgs : dict
            Parameters to pass to :func:`transitionFinder.traceMultiMin`.

        Returns
        -------
        dict
            Each item in the returned dictionary is an instance of
            :class:`transitionFinder.Phase`, and each phase is
            identified by a unique key. This value is also stored in
            `self.phases`.
        """
        tstop = self.Tmax
        points = []
        for x0 in self.approxZeroTMin():
            points.append([x0,0.0])
        tracingArgs_ = dict(forbidCrit=self.forbidPhaseCrit)
        tracingArgs_.update(tracingArgs)
        phases = transitionFinder.traceMultiMin(
            self.Vtot, self.dgradV_dT, self.d2V, points,
            tLow=0.0, tHigh=tstop, deltaX_target=100*self.x_eps, verbose = self.verbose, **tracingArgs_)
        self.phases = phases
        transitionFinder.removeRedundantPhases(
            self.Vtot, phases, self.x_eps*1e-2, self.x_eps*10, verbose = self.verbose)
        return self.phases

    def calcTcTrans(self, startHigh=False):
        """
        Runs :func:`transitionFinder.findCriticalTemperatures`, storing the
        result in `self.TcTrans`.

        In addition to the values output by
        :func:`transitionFinder.findCriticalTemperatures`, this function adds
        the following entries to each transition dictionary:

        - *Delta_rho* : Energy difference between the two phases. Positive
          values mean the high-T phase has more energy.

        Returns
        -------
        self.TcTrans
        """
        if self.phases is None:
            self.getPhases()
        self.TcTrans = transitionFinder.findCriticalTemperatures(
            self.phases, self.Vtot, startHigh)
        for trans in self.TcTrans:
            T = trans['Tcrit']
            xlow = trans['low_vev']
            xhigh = trans['high_vev']
            trans['Delta_rho'] = self.energyDensity(xhigh,T,False) - self.energyDensity(xlow,T,False)
        return self.TcTrans

    def findAllTransitions(self, tunnelFromPhase_args={}):
        """
        Find all phase transitions up to `self.Tmax`, storing the transitions
        in `self.TnTrans`.

        In addition to the values output by
        :func:`transitionFinder.tunnelFromPhase`, this function adds
        the following entries to each transition dictionary:

        - *Delta_rho* : Energy difference between the two phases. Positive
          values mean the high-T phase has more energy.
        - *Delta_p* : Pressure difference between the two phases. Should always
          be positive.
        - *crit_trans* : The transition at the critical temperature, or None
          if no critical temperature can be found.
        - *dS_dT* : Derivative of the Euclidean action with respect to
          temperature. NOT IMPLEMENTED YET.

        Parameters
        ----------
        tunnelFromPhase_args : dict
            Parameters to pass to :func:`transitionFinder.tunnelFromPhase`.

        Returns
        -------
        self.TnTrans
        """
        if self.phases is None:
            self.getPhases()
        # Add in the critical temperature
        if self.TcTrans is None:
            self.calcTcTrans()
        self.TnTrans = transitionFinder.findAllTransitions(
            self.phases, self.DVtot, self.gradV, self.TcTrans[0]['Tcrit'], tunnelFromPhase_args) # changed Vtot to DVtot, included Tmax = Tc2
        transitionFinder.addCritTempsForFullTransitions(
            self.phases, self.TcTrans, self.TnTrans)
        # Add in Delta_rho, Delta_p
        for trans in self.TnTrans:
            T = trans['Tnuc']
            xlow = trans['low_vev']
            xhigh = trans['high_vev']
            trans['Delta_rho'] = self.energyDensity(xhigh,T,False) \
                - self.energyDensity(xlow,T,False)
            trans['Delta_p'] = self.Vtot(xhigh,T,False) \
                - self.Vtot(xlow,T,False)
        return self.TnTrans

    ### Calculation of alpha and beta/H ###
    def calc_g_eff_s_SM(self, T_SM_GeV):
        return self.g_eff_instance.calc_g_eff_s_SM(T_SM_GeV, 1)

    def calc_g_eff_rho_SM(self, T_SM_GeV):
        return self.g_eff_instance.calc_g_eff_rho_SM(T_SM_GeV, 1)

    def calc_g_eff_s_DS(self, T_DS_GeV, Tnuc_DS_GeV):
        if T_DS_GeV*(1+1e-10) >= Tnuc_DS_GeV and self.is_DS_dof_const_before_PT:
            return self.g_DS_until_nuc
        else:
            return self.g_eff_instance.calc_g_eff_s_DS(self.bosons0, self.fermions, T_DS_GeV / self.conversionFactor)

    def calc_g_eff_rho_DS(self, T_DS_GeV, Tnuc_DS_GeV):
        if T_DS_GeV*(1+1e-10) >= Tnuc_DS_GeV and self.is_DS_dof_const_before_PT:
            return self.g_DS_until_nuc
        else:
            return self.g_eff_instance.calc_g_eff_rho_DS(self.bosons0, self.fermions, T_DS_GeV / self.conversionFactor)

    def calc_g_eff_s_tot(self, T_SM_GeV, Tnuc_DS_GeV, xi_T):
        return self.calc_g_eff_s_SM(T_SM_GeV) + xi_T**3 * self.calc_g_eff_s_DS(T_SM_GeV * xi_T, Tnuc_DS_GeV)

    def calc_g_eff_rho_tot(self, T_SM_GeV, Tnuc_DS_GeV, xi_T):
        return self.calc_g_eff_rho_SM(T_SM_GeV) + xi_T**4 * self.calc_g_eff_rho_DS(T_SM_GeV * xi_T, Tnuc_DS_GeV)

    def xi_at_SM_temperature(self, trans, T_SM_GeV):
        # The nucleation temperature and the respective dofs in the SM + DS have to be known.
        xi_nuc = self.xi_nuc
        conversionFactor = self.conversionFactor
        Tnuc_SM_GeV = trans["Tnuc_SM_GeV"]
        g_eff_s_DS_nuc = trans['g_eff_s_DS_nuc']
        g_eff_s_SM_nuc = trans['g_eff_s_SM_nuc']
        g_eff_s_SM_T = self.calc_g_eff_s_SM(T_SM_GeV)

        def xi_T_criterion(xi):
            g_eff_s_DS_T = self.calc_g_eff_s_DS(T_SM_GeV * xi, Tnuc_SM_GeV * xi)
            return np.abs(xi - xi_nuc * (g_eff_s_SM_T / g_eff_s_SM_nuc)**(1/3) * (g_eff_s_DS_nuc / g_eff_s_DS_T)**(1/3))
        
        from scipy.optimize import minimize
        res_xi_T = minimize(xi_T_criterion, xi_nuc, method='Nelder-Mead', tol=1e-6)

        return res_xi_T.x

    def xi_at_DS_temperature(self, trans, T_DS_GeV):
        # The nucleation temperature and the respective dofs in the SM + DS have to be known.
        xi_nuc = self.xi_nuc
        conversionFactor = self.conversionFactor
        Tnuc_DS_GeV = trans["Tnuc_DS_GeV"]
        g_eff_s_DS_nuc = trans['g_eff_s_DS_nuc']
        g_eff_s_SM_nuc = trans['g_eff_s_SM_nuc']
        g_eff_s_DS_T = self.calc_g_eff_s_DS(T_DS_GeV, Tnuc_DS_GeV)

        def xi_T_criterion(xi):
            g_eff_s_SM_T = self.calc_g_eff_s_SM(T_DS_GeV / xi)
            return np.abs(xi - xi_nuc * (g_eff_s_SM_T / g_eff_s_SM_nuc)**(1/3) * (g_eff_s_DS_nuc / g_eff_s_DS_T)**(1/3))
        
        from scipy.optimize import minimize
        res_xi_T = minimize(xi_T_criterion, xi_nuc, method='Nelder-Mead', tol=1e-6)

        return res_xi_T.x

    def calcGWTrans(self, startHigh=False, tunnelFromPhase_args={}):
        '''
        Find all first order phase transitions and its GW spectum parameters
        up to the temperature `self.Tmax`, storing the transitions and the
        parameters in `self.GWTrans`.
        
        The function sorts out all non-first order transitions since they won't
        produce any GW signals. In addition to the values output by 
        :func:`transitionFinder.tunnelFromPhaseGW`, this function adds the
        following entries to each transition dictionary of a FOPT:

        - *g_eff_tot*: The number of effective degrees of freedom within the
            SM and its extension at the nucleation temperature. Since the
            nucleation temperature would have to been known when calculating
            this qantity, the critical temperature of the phase transition
            is used initially, instead. This is ok, since, there's only a
            logarithmic dependece on g_eff_tot when calculating Tnuc. As soon
            as Tnuc as been found, g_eff_tot is calculated anew for the
            calculation of alpha.
        - *alpha*: The GW spectrum parameter alpha, describing the strength
            of the FOPT; determined using the ratio of the latent heat of the
            transition and the radiation energy densitiy.
        - *betaH*: The GW spectrum parameter beta/ H, describing the inverse
            time scale of the FOPT; determined using the formula
            Tnuc * (dS/dT)|_(T=Tnuc).
        
        Parameters
        ----------
        tunnelFromPhase_args : dict
            Parameters to pass to :func:`transitionFinder.tunnelFromPhase`.

        Returns
        -------
        self.GWTrans
        '''
        if self.phases is None:
            self.getPhases()

        if self.TcTrans is None:
            self.TcTrans = transitionFinder.findCriticalTemperatures(self.phases, self.Vtot, startHigh)

        if self.TcTrans == []:
            if self.verbose:
                print("Didn't find a transition in TcTrans. Maybe the critical phase was too supercooled.")
            errordict = dict(error = 9)
            return errordict

        trans_cnt = 0
        for transC in self.TcTrans:
            trans_cnt += 1
            # in case of unexpected numbering of transitions in self.TcTrans to manually jump to phase 2:
            #if trans_cnt==1:
            #    continue

            # Check if self.GWTrans was already defined. Then, return the first transition in self.TcTrans together with its GW params
            if self.GWTrans is not None:
                if self.verbose:
                    print("More than one FOPT has been registered.")
                return self.GWTrans

            # Check if transC is a wrongfully detected a FOPT
            if transC['low_vev'] - transC['high_vev'] == 0:
                # Check if this false FOPT is the last transition in self.TcTrans
                if trans_cnt == len(self.TcTrans):
                    if self.verbose:
                        print("The new phase splitted unexpectedly in two minima. To avoid the calculation of the transitions from one minimum into the other, stop here.")
                    errordict = dict(error = 5)
                    return errordict
                # The false FOPT is not the last one in self.TcTrans, go to the next one
                else:
                    continue

            # For a FOPT....
            if transC['trantype'] == 1:
                if self.verbose:
                    print("Found first order phase transition at Tcrit = ", transC['Tcrit'], ". Now, find Tnuc.")
                # ... calculate the GW params requiring the method findAllTransitions (i.e. beta/H and the DS nucleation temperature)
                self.GWTrans = transitionFinder.findAllTransitionsGW(self.phases, self.DVtot, self.gradV, transC, verbose = self.verbose, tunnelFromPhase_args = {"Ttol": 1e-8, "nuclCriterion": self.nucleationCriterion})

                # In GWTrans it is looked more carefully if the PT is actually a FOPT that can happen
                for transN in self.GWTrans:
                    # If the FOPT cannot succeed (i.e. because the nucleation criterion is never fulfilled), return the dictionary with the "error" as given by the transitionFinder
                    if 'error' in transN:
                        return self.GWTrans
                    # The wanted transition is found, no error occured when calculating the DS nucleation temperature and beta/H. Now calculate other GW params.
                    transN['Tcrit'] = transC['Tcrit']
                    transN['trantype'] = transC['trantype']
                    self.calc_GW_params(transN)
            # if in fact a crossover, return error
            else:
                if self.verbose:
                    print("Found a second-order transition in TcTrans.")
                errordict = dict(error = 4)
                return errordict
        return self.GWTrans

    def nucleationCriterion(self, S, T_DS):
        '''
        For a definition, see arxiv:1811.11175.
        Returns 0 at T = Tnuc, S(T = Tnuc) and > 0 at T > Tnuc and
        < 0 at T < Tnuc. The goal of :func:`transitionFinder.tunnelFromPhase`
        is to find Tnuc using this criterion.
        '''

        T_DS += 1e-100
        T_SM = T_DS / self.xi_nuc
        T_SM_GeV = T_SM * self.conversionFactor
        xi_T = T_DS / T_SM

        # Calculate g_eff(T)
        g_eff_rho_tot_T = self.calc_g_eff_rho_tot(T_SM_GeV, 0, xi_T) # set 0 since then the DS dof = 4 always.
        #g_eff_rho_tot_T = 100
        crit = S / T_DS - 146.
        crit += 2. * np.log(g_eff_rho_tot_T / 100.)
        crit += 4. * np.log(T_DS / 100. * self.conversionFactor)
        return crit

    def calc_GW_params(self, trans):
        if False:
            instanton = trans['instanton']
            plt.figure()
            plt.plot(instanton.profile1D.R, instanton.profile1D.Phi)
            np.savetxt("instanton_R", instanton.profile1D.R)
            np.savetxt("instanton_Phi", instanton.profile1D.Phi)
            plt.xlabel("radius")
            plt.ylabel(R"$\phi-\phi_{min}$ (along the path)")
            plt.title("Tunneling profile")
            plt.show()


        xhigh = trans['high_vev']
        xlow = trans['low_vev']

        # Set nucleation temperatures in hidden sector and SM bath in chosen vev unit and GeV
        Tnuc_DS = trans['Tnuc']
        Tnuc_SM = Tnuc_DS / self.xi_nuc
        Tnuc_SM_GeV = Tnuc_SM * self.conversionFactor
        Tnuc_DS_GeV = Tnuc_DS * self.conversionFactor
        trans['Tnuc_DS'] = Tnuc_DS
        trans['Tnuc_DS_GeV'] = Tnuc_DS_GeV
        trans['Tnuc_SM'] = Tnuc_SM
        trans['Tnuc_SM_GeV'] = Tnuc_SM_GeV

        # Calculate effective degrees of freedom at Tnuc
        g_eff_s_SM_nuc = self.calc_g_eff_s_SM(Tnuc_SM_GeV)
        trans['g_eff_s_SM_nuc'] = g_eff_s_SM_nuc

        g_eff_rho_SM_nuc = self.calc_g_eff_rho_SM(Tnuc_SM_GeV)
        trans['g_eff_rho_SM_nuc'] = g_eff_rho_SM_nuc        

        g_eff_s_DS_nuc = self.calc_g_eff_s_DS(Tnuc_DS_GeV, Tnuc_DS_GeV)
        trans['g_eff_s_DS_nuc'] = g_eff_s_DS_nuc

        g_eff_rho_DS_nuc = self.calc_g_eff_rho_DS(Tnuc_DS_GeV, Tnuc_DS_GeV)
        trans['g_eff_rho_DS_nuc'] = g_eff_rho_DS_nuc

        g_eff_s_tot_nuc = self.calc_g_eff_s_tot(Tnuc_SM_GeV, Tnuc_DS_GeV, self.xi_nuc)
        trans['g_eff_s_tot_nuc'] = g_eff_s_tot_nuc

        g_eff_rho_tot_nuc = self.calc_g_eff_rho_tot(Tnuc_SM_GeV, Tnuc_DS_GeV, self.xi_nuc)
        trans['g_eff_rho_tot_nuc'] = g_eff_rho_tot_nuc


        # Save the critical temperature in the SM and the DS in chosen vev unit and GeV
        trans['Tcrit_DS'] = trans['Tcrit']
        trans['Tcrit_DS_GeV'] = trans['Tcrit'] * self.conversionFactor
        self.xi_crit = self.xi_at_DS_temperature(trans, trans['Tcrit_DS_GeV'])
        trans['Tcrit_SM'] = trans['Tcrit_DS'] / self.xi_crit
        trans['Tcrit_SM_GeV'] = trans['Tcrit_DS_GeV'] / self.xi_crit

        # Latent heat(lambda, g, vev, Gamma, xi nuc)
        latentHeat = self.energyDensity(xhigh, Tnuc_DS, False) - self.energyDensity(xlow, Tnuc_DS, False) # is positive
        trans['Delta_rho'] = latentHeat
        trans['Delta_rho_GeV4'] = latentHeat * self.conversionFactor**4.

        # Pressure between phases
        Delta_p = self.Vtot(xhigh, Tnuc_DS, False) - self.Vtot(xlow, Tnuc_DS, False)
        trans['Delta_p'] = Delta_p # is positive
        trans['Delta_p_GeV4'] = Delta_p * self.conversionFactor**4.

        # Trace of energy-momentum tensor
        Delta_enmomtensortrace = (latentHeat + 3 * Delta_p) / 4 # is positive
        trans['Delta_enmomtensortrace'] = Delta_enmomtensortrace # is positive
        trans['Delta_enmomtensortrace_GeV4'] = Delta_enmomtensortrace * self.conversionFactor**4.

        # alpha (note that independent of self.conversionFactor)
        rho_rad_nuc = np.pi**2. / 30. * g_eff_rho_tot_nuc * Tnuc_SM**4.
        rho_rad_DS_nuc = np.pi**2. / 30. * g_eff_rho_DS_nuc * (self.xi_nuc * Tnuc_SM)**4.
        alpha_latentheat = latentHeat / rho_rad_nuc             # definition used initially in 1512.06239
        alpha_trace = Delta_enmomtensortrace / rho_rad_nuc     # qunatity advertised in 1910.13125v2 that also works for weaker transitions
        alpha_latentheatDSnorm = latentHeat / rho_rad_DS_nuc          # definition used initially in 1512.06239
        alpha_traceDSnorm = Delta_enmomtensortrace / rho_rad_DS_nuc     # qunatity advertised in 1910.13125v2 that also works for weaker transitions
        trans['alpha'] = alpha_trace
        trans['alpha_trace'] = alpha_trace
        trans['alpha_latentheat'] = alpha_latentheat
        trans['alpha_DSnorm'] = alpha_traceDSnorm

        # alpha_inf (note that independent of self.conversionFactor)
        m2_bos_0, _, _, _ = self.boson_massSq(np.array([self.v]), 0.)
        m2_bos, dof_bos, _, is_physical = self.boson_massSq(np.array([self.v]), Tnuc_DS)
        m2_fer_0, _ = self.fermion_massSq(np.array([self.v]))
        m2_fer, dof_fer = self.fermion_massSq(np.array([self.v]))

        delta_m2_bos = np.zeros(len(m2_bos))
        for i in range(len(m2_bos)):
            delta_m2_bos[i] = m2_bos[i] - m2_bos_0[i] if m2_bos[i] - m2_bos_0[i] > 0 else 0
        m2factor = np.sum(dof_bos*is_physical*delta_m2_bos, axis=-1) / 24.
        
        if m2_fer > 0:
            delta_m2_fer = np.zeros(len(m2_fer))
            for i in range(len([m2_fer])):
                delta_m2_fer[i] = m2_fer[i] - m2_fer_0[i] if m2_fer[i] - m2_fer_0[i] > 0 else 0
            m2factor += np.sum(dof_fer*delta_m2_fer, axis=-1) / 48.
        
        alpha_inf = Tnuc_DS**2 / rho_rad_DS_nuc * m2factor # note normlaization on DS rho since only DS contributes to fluid dynamics for DS phase transition
        trans['alpha_inf'] = alpha_inf

        # Dilution factor calculation
        T_DS_chem_dec_GeV = 1 * self.mDP_GeV # assuming that the DP is the next-to-lightest particle
        xi_chem_dec = np.squeeze(self.xi_at_DS_temperature(trans, T_DS_chem_dec_GeV))
        T_SM_chem_dec_GeV = T_DS_chem_dec_GeV / xi_chem_dec

        if T_SM_chem_dec_GeV > Tnuc_SM_GeV:
            T_SM_chem_dec_GeV = Tnuc_SM_GeV
            xi_chem_dec = self.xi_nuc
            trans['warning'] = 1 # set T_cd to T_nuc if nuc happens after DP gets non-rel after nucleation. Set warning.
        else:
            trans['warning'] = 0 # set warning to 0 as default.

        trans['xi_chem_dec'] = xi_chem_dec
        trans['T_SM_chem_dec_GeV'] = T_SM_chem_dec_GeV

        g_eff_s_DS_chem_dec = self.calc_g_eff_s_DS(T_DS_chem_dec_GeV, Tnuc_DS_GeV)
        if g_eff_s_DS_chem_dec < 0.9:
            trans['warning'] += 2

        m_med_GeV = self.mDH_GeV # assuming that the DH is the lightest particle 
        Gamma_GeV = self.Gamma_GeV # assuming that the lightest particle is the only mediator: else, interpret as effective Gamma.
        f_DM_chem_dec = 0 # Assume no relevant DM relic energy density at this point in time
        g_med = 1 # Assuming DH as mediator
        alpha32 = 9 / (2**(1/3) * np.pi) * self.l
        if self.verbose:
            print("\nStart dilution factor calculation.")
        try:
            dil = dilution.dilution_factor_calculation(T_SM_chem_dec_GeV, m_med_GeV, Gamma_GeV, xi_chem_dec, f_DM_chem_dec, g_med, alpha32, verbose=self.verbose, foldername=self.output_folder, plot_results=self.dilution_plots)
            D_SM = dil.DSM
            D = D_SM * g_eff_s_SM_nuc / g_eff_s_tot_nuc
            if D >= 1:
                trans['D_SM'] = D_SM
                trans['D'] = D
                trans['Tf_SM_GeV'] = dil.finalT
                trans['can_warn'] = dil.can_warn
            else:
                if self.verbose:
                    print("Dilution factor D < 1 detected. Set to 1 manually.")
                trans['D_SM'] = g_eff_s_tot_nuc / g_eff_s_SM_nuc
                trans['D'] = 1
                trans['Tf_SM_GeV'] = dil.finalT
                trans['warning'] += 4 # if D < 1 was detected and had to be set manually to 1
                trans['can_warn'] = dil.can_warn
            del dil
        except:
            if self.verbose:
                print("Dilution factor calculation aborted.")
            trans['D_SM'] = g_eff_s_tot_nuc / g_eff_s_SM_nuc
            trans['D'] = 1
            trans['Tf_SM_GeV'] = T_SM_chem_dec_GeV
            trans['warning'] += 8 # works as a warning that the dilution factor calculation failed
            trans['can_warn'] = False 

        # Save information about the DH and DP becoming non-relativistic
        mDH_GeV = self.mDH_GeV
        mDP_GeV = self.mDP_GeV
        xi_DH = self.xi_at_DS_temperature(trans, mDP_GeV)
        xi_DP = self.xi_at_DS_temperature(trans, mDH_GeV)
        T_SM_DH_GeV = mDH_GeV / xi_DH
        T_SM_DP_GeV = mDP_GeV / xi_DP
        g_eff_rho_tot_DH = self.calc_g_eff_rho_tot(T_SM_DH_GeV, Tnuc_DS_GeV, xi_DH)
        g_eff_rho_tot_DP = self.calc_g_eff_rho_tot(T_SM_DP_GeV, Tnuc_DS_GeV, xi_DP)
        trans['xi_DH'] = xi_DH
        trans['xi_DP'] = xi_DP
        trans['g_eff_rho_tot_DH'] = g_eff_rho_tot_DH
        trans['g_eff_rho_tot_DP'] = g_eff_rho_tot_DP
        

    ##### Fancy output methods
    def plot_all_contributions(self, x1, x2, T=0, subtract=False, n=500, **plotParams):
        import matplotlib.pyplot as plot
        if self.Ndim == 1:
            x = np.linspace(x1,x2,n)
            X = x[:,np.newaxis]
        else:
            dX = np.array(x2)-np.array(x1)
            X = dX*np.linspace(0,1,n)[:,np.newaxis] + x1
            x = np.linspace(0,1,n)*np.sum(dX**2)**.5

        T = np.asanyarray(T, dtype=float)

        bosons0 = self.boson_massSq(X,0.*T)
        m20, nb, c, _ = bosons0

        bosonsX0 = self.boson_massSq(X*0.,0.*T) 
        m2X0, nb, c, _ = bosonsX0

        bosonsT = self.boson_massSq(X,T)
        m2T, nbT, cT, _ = bosonsT

        bosonsTX0 = self.boson_massSq(X*0,T)
        m2TX0, nbT, cT, _ = bosonsTX0

        fermions = self.fermion_massSq(X)
        fermionsX0 = self.fermion_massSq(X*0)

        include_radiation = True

        Vtree = self.V0(X) - self.V0(X*0) if subtract else self.V0(X)
        
        V1 = self.V1(bosons0, fermions) - self.V1(bosonsX0, fermions) if subtract else self.V1(bosons0, fermions)
        Vct = self.Vct(X) - self.Vct(X*0) if subtract else self.Vct(X)
        V1T = self.V1T(bosons0, fermions, T, include_radiation) - self.V1T(bosonsX0, fermions, T, include_radiation) if subtract else self.V1T(bosons0, fermions, T, include_radiation)
        Vdaisy = np.real(-(T/(12.*np.pi))*np.sum ( nb*(pow(m2T+0j,1.5) - pow(m20+0j,1.5)), axis=-1) + (T/(12.*np.pi))*np.sum ( nb*(pow(m2TX0+0j,1.5) - pow(m2X0+0j,1.5)), axis=-1)) if subtract else np.real(-(T/(12.*np.pi))*np.sum ( nb*(pow(m2T+0j,1.5) - pow(m20+0j,1.5)), axis=-1))

        Vtot = Vtree + V1 + Vct + V1T + Vdaisy 
        plt.plot(x, Vtree, label = "Vtree", **plotParams)
        plt.plot(x, V1, label = "V1", **plotParams)
        plt.plot(x, Vct, label = "Vct", **plotParams)
        plt.plot(x, V1T, label = "V1T", **plotParams)
        plt.plot(x, Vdaisy, label = "Vdaisy", **plotParams)
        plt.plot(x, Vtot, label = "Vtot", **plotParams)

        plt.xlabel(R"$\phi$")
        plt.ylabel(R"$V(\phi)$")
        plt.legend()
        plt.grid()
        plt.show()

    def fancy_T_plot(self, T):
        '''
        Plots the potential with and without tree-level corrections at a
        specific temperature T
        '''
        print("\n### Plot potential at T ###")
        self.plot1d(0, 1500, 0, treelevel = True, label = "Tree level potential")
        self.plot1d(0, 1500, 0, treelevel = False, label = "T = 0")
        self.plot1d(0, 1500, T, treelevel = False, label = "T = " + str(T))
        plt.grid()
        plt.legend()
        plt.show()
        print("(showed plot)")

    def fancy_T0_plot(self):
        '''
        Plots the potential with and without tree-level corrections at zero temperature
        '''
        print("\n### Plot potential at T = 0 ###")
        self.plot1d(0, 1500, T = 0, treelevel = True, label = "Tree level potential")
        self.plot1d(0, 1500, T = 0, treelevel = False, label = "incl. rad. corr.")
        plt.grid()
        plt.legend()
        plt.show()
        print("(showed plot)")

    def find_minimum_without_CT(self):
        '''
        Calculate the minimum of the potential at zero temperature without the help
        of cosmoTransitions but using a scipy method
        '''
        print("\n### Find minimum of potential using scipy.optimize.minimize ###")
        print(optimize.minimize(lambda x: self.Vtot(x,T=0),[200],method='Nelder-Mead'))

    def find_minimum_with_CT(self):
        '''
        Find the minimum of the zero temperature potential using cosmoTransitions.
        '''
        print("\n### Find mimimum using CT function ###")
        print(np.squeeze(self.findMinimum())) 

    def find_ms_without_CT(self):
        '''
        In the case of a Higgs field, calculate its ass using the potentials zero T vev
        '''
        print("\n### Find scalar field S mass via value of Hessian at minimum for T = 0 ###")
        print(np.squeeze(np.sqrt(self.d2V(self.findMinimum(),T=0))))

    def find_phases(self):
        '''
        Finds the phases of the theory and prints them in a fancy manner.
        '''
        print("\n### Find phases of the theory ###")
        print(self.getPhases())

    def fancy_output_calcGWTrans(self):
        '''
        Lists all the relevant parameters of the phase transition calculated in
        calcGWTrans in a fancy manner.
        '''
        if self.calcGWTrans is None:
            self.calcGWTrans = self.calcGWTrans()
        transition = self.GWTrans

        if transition:
            print("\n### Found the following phase transition parameters ###\n")
            t = transition[0]
            s = self.conversionFactor
            self.print_params_vals = (self.l, self.g, self.xi_nuc, t['low_vev'] * s, t['high_vev'] * s, t['Delta_rho_GeV4'], t['Tcrit_SM_GeV'], t['Tnuc_SM_GeV'], t['trantype'], t['low_phase'], t['high_phase'], t['alpha'], t['alpha_inf'], t['betaH'])
            self.print_params_names = ("lambda", "g", "xi_nuc", "low phase vev / GeV", "high phase vev / GeV", "Delta rho / GeV4", "Tcrit_SM / GeV", "Tnuc_SM / GeV", "Order of PT", "low phase key", "high phase key", "alpha", "alpha_inf", "beta/H")
            self.print_params = dict(zip(self.print_params_names, self.print_params_vals))
            for n, v in self.print_params.items():
                print ("{:<22} {:e}".format(n, np.squeeze(v)))
        else:
            print("No phase transition found.")

    def plot_potential_around_Tcrit(self):
        '''
        Plots the potential around the critical temperature
        '''
        print("\n### Plot potential before, while and after Tcrit")
        if self.TcTrans is None:
            self.TcTrans = transitionFinder.findCriticalTemperatures(self.phases, self.Vtot, startHigh)
        Tc = self.TcTrans[0]['Tcrit']
        self.plot1d(0, 1000, T = Tc * 1.01, label = "T > Tc")
        self.plot1d(0, 1000, T = Tc, label = "T = Tc")
        self.plot1d(0, 1000, T = Tc * 0.99, label = "T < Tc")
        plt.grid()
        plt.legend()
        #plt.savefig("plots/Potential_around_transition.pdf")
        plt.show()
        print("(showed plots)\n")


    def fancy_phases_plot(self):
        '''
        Plots the phases of the theory
        '''
        print("\n### Plot phases ###")
        self.plotPhasesPhi()
        #plt.savefig("plots/Phases_T.pdf")
        plt.grid()
        plt.show()
        print("(showed plot)")

    def prettyPrintTnTrans(self):
        if self.TnTrans is None:
            raise RuntimeError("self.TnTrans has not been set. "
                "Try running self.findAllTransitions() first.")
        if len(self.TnTrans) == 0:
            print("No transitions for this potential.\n")
        for trans in self.TnTrans:
            trantype = trans['trantype']
            if trantype == 1:
                trantype = 'First'
            elif trantype == 2:
                trantype = 'Second'
            print("%s-order transition at Tnuc = %0.4g" %
                  (trantype, trans['Tnuc']))
            print("High-T phase:\n  key = %s; vev = %s" %
                  (trans['high_phase'], trans['high_vev']))
            print("Low-T phase:\n  key = %s; vev = %s" %
                  (trans['low_phase'], trans['low_vev']))
            print("Pressure difference = %0.4g = (%0.4g)^4" %
                  (trans['Delta_p'], trans['Delta_p']**.25))
            print("Energy difference = %0.4g = (%0.4g)^4" %
                  (trans['Delta_rho'], trans['Delta_rho']**.25))
            print("Action = %0.4g" % trans['action'])
            print("Action / Tnuc = %0.6g" % (trans['action']/trans['Tnuc']))
            print("")

    # PLOTTING ---------------------------------

    def plot2d(self, box, T=0, treelevel=False, offset=0,
               xaxis=0, yaxis=1, n=50, clevs=200, cfrac=.8, **contourParams):
        """
        Makes a countour plot of the potential.

        Parameters
        ----------
        box : tuple
            The bounding box for the plot, (xlow, xhigh, ylow, yhigh).
        T : float, optional
            The temperature
        offset : array_like
            A constant to add to all coordinates. Especially
            helpful if Ndim > 2.
        xaxis, yaxis : int, optional
            The integers of the axes that we want to plot.
        n : int
            Number of points evaluated in each direction.
        clevs : int
            Number of contour levels to draw.
        cfrac : float
            The lowest contour is always at ``min(V)``, while the highest is
            at ``min(V) + cfrac*(max(V)-min(V))``. If ``cfrac < 1``, only part
            of the plot will be covered. Useful when the minima are more
            important to resolve than the maximum.
        contourParams :
            Any extra parameters to be passed to :func:`plt.contour`.

        Note
        ----
        .. todo::
            Add an example plot.
            Make documentation for the other plotting functions.
        """
        import matplotlib.pyplot as plt
        xmin,xmax,ymin,ymax = box
        X = np.linspace(xmin, xmax, n).reshape(n,1)*np.ones((1,n))
        Y = np.linspace(ymin, ymax, n).reshape(1,n)*np.ones((n,1))
        XY = np.zeros((n,n,self.Ndim))
        XY[...,xaxis], XY[...,yaxis] = X,Y
        XY += offset
        Z = self.V0(XY) if treelevel else self.Vtot(XY,T)
        minZ, maxZ = min(Z.ravel()), max(Z.ravel())
        N = np.linspace(minZ, minZ+(maxZ-minZ)*cfrac, clevs)
        plt.contour(X,Y,Z, N, **contourParams)
        plt.axis(box)
        plt.show()

    def plot1d(self, x1, x2, T=0, treelevel=False, subtract=True, n=500, **plotParams):
        import matplotlib.pyplot as plt
        if self.Ndim == 1:
            x = np.linspace(x1,x2,n)
            X = x[:,np.newaxis]
        else:
            dX = np.array(x2)-np.array(x1)
            X = dX*np.linspace(0,1,n)[:,np.newaxis] + x1
            x = np.linspace(0,1,n)*np.sum(dX**2)**.5
        if treelevel:
            y = self.V0(X) - self.V0(X*0) if subtract else self.V0(X)
        else:
            y = self.DVtot(X,T) if subtract else self.Vtot(X, T)

        if True:
            from time import time
            timestamp = str(time())
            np.savetxt(timestamp + "pot_at_T=_" + str(T) + ".txt", (x,y))
        plt.plot(x,y, **plotParams)
        plt.xlabel(R"$\phi$")
        plt.ylabel(R"$V(\phi)$")
        plt.show()

    def plotPhasesV(self, useDV=True, **plotArgs):
        import matplotlib.pyplot as plt
        if self.phases is None:
            self.getPhases()
        for key, p in list(self.phases.items()):
            V = self.DVtot(p.X,p.T) if useDV else self.Vtot(p.X,p.T)
            plt.plot(p.T,V,**plotArgs)
        plt.xlabel(R"$T$")
        if useDV:
            plt.ylabel(R"$V[\phi_{min}(T), T] - V(0, T)$")
        else:
            plt.ylabel(R"$V[\phi_{min}(T), T]$")
        plt.show()

    def plotPhasesPhi(self, **plotArgs):
        import matplotlib.pyplot as plt
        if self.phases is None:
            self.getPhases()
        for key, p in list(self.phases.items()):
            phi_mag = np.sum(p.X**2, -1)**.5
            plt.plot(p.T, phi_mag, **plotArgs)

        phi0 = self.phases[0].X
        phi1 = self.phases[1].X
        T0 = self.phases[0].T
        T1 = self.phases[1].T
        np.savetxt("PlotPhases_phi1.txt", phi1)
        np.savetxt("PlotPhases_phi0.txt", phi0)
        np.savetxt("PlotPhases_T1.txt", T1)
        np.savetxt("PlotPhases_T0.txt", T0)
        plt.xlabel(R"$T$")
        plt.ylabel(R"$\phi(T)$")
        plt.show()

# END GENERIC_POTENTIAL CLASS ------------------


# FUNCTIONS ON LISTS OF MODEL INSTANCES ---------------

def funcOnModels(f, models):
    """
    If you have a big array of models, this function allows you
    to extract big arrays of model outputs. For example, suppose
    that you have a 2x5x20 nested list of models and you want to
    find the last critical temperature of each model. Then use

    >>> Tcrit = funcOnModels(lambda A: A.TcTrans[-1]['Tcrit'], models)

    Tcrit will be a numpy array with shape (2,5,20).
    """
    M = []
    for a in models:
        if isinstance(a,list) or isinstance(a,tuple):
            M.append(funcOnModels(f, a))
        else:
            try:
                M.append(f(a))
            except:
                M.append(np.nan)
    return np.array(M)


def _linkTransitions(models, critTrans=True):
    """
    This function will take a list of models that represent the
    variation of some continuous model parameter, and output several
    lists of phase transitions such that all of the transitions
    in a single list roughly correspond to each other.

    NOT UPDATED FOR COSMOTRANSITIONS v2.0.
    """
    allTrans = []
    for model in models:
        allTrans.append(model.TcTrans if critTrans else model.TnTrans)
    # allTrans is now a list of lists of transitions.
    # We want to rearrange each sublist so that it matches the previous sublist.
    for j in range(len(allTrans)-1):
        trans1, trans2 = allTrans[j], allTrans[j+1]
        if trans1 is None: trans1 = []
        if trans2 is None: trans2 = []
        # First, clear the transiction dictionaries of link information
        for t in trans1+trans2:
            if t is not None:
                t['link'] = None
                t['diff'] = np.inf
        for i1 in range(len(trans1)):
            t1 = trans1[i1]  # t1 and t2 are individual transition dictionaries
            if t1 is None: continue
            for i2 in range(len(trans2)):
                t2 = trans2[i2]
                if t2 is None: continue
                # See if t1 and t2 are each other's closest match
                diff = np.sum((t1['low vev']-t2['low vev'])**2)**.5 \
                    + np.sum((t1['high vev']-t2['high vev'])**2)**.5
                if diff < t1['diff'] and diff < t2['diff']:
                    t1['diff'] = t2['diff'] = diff
                    t1['link'], t2['link'] = i2, i1
        for i2 in range(len(trans2)):
            t2 = trans2[i2]
            if (t2 is not None and t2['link'] is not None and
                    trans1[t2['link']]['link'] != i2):
                t2['link'] = None  # doesn't link back.
        # Now each transition in tran2 is linked to its closest match in tran1,
        # or None if it has no match
        newTrans = [None]*len(trans1)
        for t2 in trans2:
            if t2 is None:
                continue
            elif t2['link'] is None:
                # This transition doesn't match up with anything.
                newTrans.append(t2)
            else:
                newTrans[t2['link']] = t2
        allTrans[j+1] = newTrans
    # Almost done. Just need to clean up the transitions and make sure that
    # the allTrans list is rectangular.
    for trans in allTrans:
        for t in trans:
            if t is not None:
                del t['link']
                del t['diff']
    n = len(allTrans[-1])
    for trans in allTrans:
        while len(trans) < n:
            trans.append(None)
    # Finally, transpose allTrans:
    allTrans2 = []
    for i in range(len(allTrans[0])):
        allTrans2.append([])
        for j in range(len(allTrans)):
            allTrans2[-1].append(allTrans[j][i])
    return allTrans2
