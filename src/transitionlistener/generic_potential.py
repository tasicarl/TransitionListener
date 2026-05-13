"""The primary task of the generic_potential module is to define the
:class:`generic_potential` class, from which realistic scalar field models can
straightforwardly be constructed. The most important part of any such model is,
appropriately, the potential function and its gradient. This present module is not
necessary to define a potential, but it does make the process somewhat simpler
by automatically calculating one-loop effects from a model-specific mass
spectrum, constructing numerical derivative functions, providing a
simplified interface to the :mod:`.transitionFinder` module, and providing
several methods for plotting the potential and its phases.


The generic_potential class has to provide the following information to the
rest of the code:

1. Vtot, dVdx, dVdT, d2Vdx (Potential)
2. energyDensity (the total energy density at a given temperature)
3. configuration
4. Tmax, Tmin (range of validitiy of the potential)
5. conversionFactor (conversion between internal units and GeV)
6. X0 (zero temperature global minimum)
7. Mass spectrum of the theory.

Part of TransitionListener v2.0
Documentation: https://tasillo.de/TransitionListener/

Authors:
    Jonas Matuszak <jonas.matuszak@kit.edu>
    Carlo Tasillo <carlo.tasillo@ific.uv.es>
"""

import numpy as np
from scipy import optimize
from rich.columns import Columns as RichColumns
import rich


from transitionlistener.finiteT import Jb_spline as Jb
from transitionlistener.finiteT import Jf_spline as Jf
from transitionlistener import helper_functions
from transitionlistener.thermodynamics import e_geffSM, p_geffSM, set_sm_temperature_cap

from transitionlistener.bubbledynamics import approxNucleationCriterion

from transitionlistener import errors
from transitionlistener import runtime_options
from transitionlistener.config import Configuration
from transitionlistener import constants as cn
from transitionlistener.counterterms.cw import coleman_weinberg_derivatives as _cw_derivatives

from . import console
from transitionlistener.particles import (
    MassSpectrum,
    SpectrumSnapshot,
)


class generic_potential():
    """
    An abstract class from which one can easily create finite-temperature
    effective potentials.

    This class acts as the skeleton around which different scalar field models
    can be formed. At a bare minimum, subclasses must implement :func:`init`,
    :func:`V0`, and they must provide a
    :class:`~transitionlistener.particles.MassSpectrum`
    instance via ``self.mass_spectrum``. Subclasses will also likely implement
    :func:`approxZeroTMin`. Once the tree-level
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
        defaults to 1000.0**2 in internal units.
    Tmax : float
        The maximum temperature to which minima should be followed. No
        transitions are calculated above this temperature. This is also used
        as the overall temperature scale in :func:`getPhases`.
        May be overridden by subclasses; defaults to 1000.0 in internal units.
    """

    def __init__(self, *args, **dargs) -> None:
        """Prepare configuration objects and delegate to :meth:`init` for setup."""

        self.config = Configuration()  # This contains all the tracing settings, etc

        self.kin_coupled_e_geff = e_geffSM  # This can be overwritten
        self.kin_coupled_p_geff = p_geffSM  # This can be overwritten

        self.kin_decoupled_e_geff = lambda T, cf: 0.0 * T  # If the SM is decoupled, put e_geffSM here
        self.kin_decoupled_p_geff = lambda T, cf: 0.0 * T  # If the SM is decoupled, put p_geffSM here

        # The parameters below need to be specified in subclass init()
        self.verbose = False
        self.v_GeV = None
        self.X0 = None
        self.renormScaleSq = None
        self.conversionFactor = None
        self.daisy = None
        self.mass_spectrum: MassSpectrum | None = None
        self.derived_parameters = {}

        self.setConfigParameters()
        _pre_override_fRC = self.config.tracingConf.tunneling_params[
            "deformation_deform_params"
        ]["fRatioConv"]
        args, dargs = runtime_options.apply_input_overrides(self, args, dargs)
        self.v_stable = self.config.tracingConf.internal_scale
        self.x_eps = self.config.tracingConf.tracing_field_accuracy
        self.T_eps = self.config.tracingConf.tracing_temp_accuracy
        self.deriv_order = self.config.tracingConf.tracing_derivative_order

        self.init(*args, **dargs)

        # For multi-field potentials the path-deformation convergence target
        # needs to be tighter than the 1d default; otherwise the bounce action
        # exhibits 0.1-1 dex jitter that downstream percolation cannot resolve.
        # Only override when the user has not already set fRatioConv themselves.
        if (
            getattr(self, "Ndim", 1) >= 2
            and self.config.tracingConf.tunneling_params["deformation_deform_params"][
                "fRatioConv"
            ]
            == _pre_override_fRC
        ):
            self.config.tracingConf.tunneling_params["deformation_deform_params"][
                "fRatioConv"
            ] = 1e-2
        self._ensure_mass_spectrum()
        self._sync_mass_labels()
        self.Tmin = cn.T0_SM_GeV / self.conversionFactor
        self.Tmax = self.config.tracingConf.Tmax_factor * self.v_stable
        self.checkInitialisation()
        self.generateInvGroupElements()
        self._update_sm_temperature_cap()

        if self.verbose:
            self.makePrettyDictionaryPrint(self.derived_parameters)

    def init(self, *args, **dargs) -> None:
        """
        Subclasses should override this method (not __init__) to do all
        initialization. At a bare minimum, subclasses need to specify the number
        of dimensions in the potential with ``self.Ndim``.
        """
        # dummy dimensions
        self.Ndim = 1

    def setConfigParameters(self) -> None:
        """Change the default config parameters for tracing, hydrodynamics, and GW
        computation.

        This method should be overwritten by the subclass.
        """
        # self.config.tracingConf.internal_scale = 1000
        # self.config.tracingConf.Tmax_factor = 2.5
        # ...

    def set_tracing_params(self) -> None:
        """Backward-compatible helper to resync cached tracing attributes.

        Older model implementations still call this after tweaking the config.
        """
        self.v_stable = self.config.tracingConf.internal_scale
        self.x_eps = self.config.tracingConf.tracing_field_accuracy
        self.T_eps = self.config.tracingConf.tracing_temp_accuracy
        self.deriv_order = self.config.tracingConf.tracing_derivative_order

    def set_modelparams(self, inputparam_dict: dict) -> dict:
        """
        Set the model parameters from the inputparam_dict dictionary. Print warnings
        if some parameters are not set and default values are used instead. Also
        check if the parameters are in the allowed ranges.
        """
        mp = {name: params.copy() for name, params in self.model_parameters.items()}

        for key in inputparam_dict.keys():
            if key in mp.keys():
                mp[key]["value"] = float(inputparam_dict[key])
                self.model_parameters[key]["value"] = float(inputparam_dict[key])
            else:
                raise errors.InitPotentialError(f"Unknown model parameter: {key}")

        for key in mp.keys():
            if "value" not in mp[key].keys():
                mp[key]["value"] = mp[key]["default"]
                console.print(
                    f"[yellow]Warning: No value given for model parameter {key}. "
                    f"Using default value {mp[key]['default']}.[/yellow]"
                )
        for key in mp.keys():
            try:
                value = float(mp[key]["value"])
            except (ValueError, TypeError):
                value = eval(mp[key]["value"])

            min_val = mp[key].get("min", -np.inf)
            max_val = mp[key].get("max", np.inf)
            if not (min_val <= value <= max_val):
                raise errors.InitPotentialError(
                    f"Model parameter {key}={value} is out of range "
                    f"[{min_val}, {max_val}]."
                )
        self._update_sm_temperature_cap()
        return mp

    def computeConversionFactor(self, v_stable: float, v_GeV: float) -> float:
        """compute the conversionFactor to go from internal units
        to GeV. """
        conversionFactor = v_GeV / self.v_stable
        return conversionFactor

    def coleman_weinberg_from_curvatures(
        self,
        tensors: dict[str, np.ndarray],
        vev: np.ndarray,
        *,
        scale: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""Evaluate Coleman–Weinberg derivatives from curvature tensors.

        Parameters
        ----------
        tensors:
            Dictionary containing the Higgs, gauge, and Yukawa curvature tensors
            in the gauge basis.  The expected keys are ``H2``, ``H3``, ``H4``,
            ``Gauge``, ``Quark``, and ``Lepton``.
        vev:
            Vacuum expectation value vector expressed in the same field basis
            as the tensors (typically the eight-dimensional Higgs basis).
        scale:
            Renormalisation scale :math:`\mu` entering the Coleman–Weinberg
            logarithms.

        Returns
        -------
        (gradient, hessian): ``np.ndarray``
            Coleman–Weinberg first and second derivatives in physical units.
        """

        grad_phys, hess_phys = _cw_derivatives(
            tensors["H2"],
            tensors["H3"],
            tensors["H4"],
            tensors["Gauge"],
            tensors["Quark"],
            tensors["Lepton"],
            vev,
            scale=scale,
        )
        return grad_phys, hess_phys

    def _ensure_mass_spectrum(self) -> None:
        """Verify that the subclass provided a MassSpectrum instance."""
        if not isinstance(self.mass_spectrum, MassSpectrum):
            raise errors.InitPotentialError(
                "Model initialisation must assign `self.mass_spectrum` to a "
                "transitionlistener.particles.MassSpectrum instance."
            )

    def _sync_mass_labels(self) -> None:
        """Derive mass labels directly from the particle definitions."""
        if not isinstance(self.mass_spectrum, MassSpectrum):
            self.boson_mass_labels = {"latex": [], "text": []}
            self.fermion_mass_labels = {"latex": [], "text": []}
            return
        self.boson_mass_labels = {
            "latex": self.mass_spectrum.boson_labels("latex"),
            "text": self.mass_spectrum.boson_labels("text"),
        }
        self.fermion_mass_labels = {
            "latex": self.mass_spectrum.fermion_labels("latex"),
            "text": self.mass_spectrum.fermion_labels("text"),
        }

    def _update_sm_temperature_cap(self) -> None:
        """Cache the SM temperature cap using the spectrum at the zero-T minimum."""
        if (
            not isinstance(self.mass_spectrum, MassSpectrum)
            or self.conversionFactor is None
            or self.X0 is None
        ):
            return
        try:
            bosons = self.boson_massSq(np.asarray(self.X0), 0.0)
            fermions = self.fermion_massSq(np.asarray(self.X0))
        except Exception as e:
            msg = ("Could not evaluate mass spectrum at zero-T minimum "
                   "to set SM temperature cap: ") + str(e)
            console.print(f"[yellow]Warning: {msg}[/yellow]")
            return
        set_sm_temperature_cap(
            self.conversionFactor,
            bosons, self.mass_spectrum.is_SM_bosons,
            fermions, self.mass_spectrum.is_SM_fermions,
            verbose=self.verbose,
        )

    def get_mass_spectrum(self, X: np.ndarray, T: float | np.ndarray) -> SpectrumSnapshot:
        """Evaluate bosonic and fermionic spectra at (X, T)."""
        return self.mass_spectrum.evaluate(X, T)


    def boson_massSq(self, X, T):
        r"""Return the bosonic mass spectrum :math:`m_i^2(\phi, T)` of the model.

        Subclasses typically implement the actual mass matrices through
        ``self.mass_spectrum``. The returned tuple is forwarded to the one-loop
        and thermal-potential routines, which use the bosonic contribution

        .. math::
           V_1^{B}(\phi, T=0)
           = \sum_i \frac{n_i\,m_i^4(\phi,0)}{64\pi^2}
             \left[\log\!\left(\frac{m_i^2(\phi,0)}{\mu^2}\right)-c_i\right].

        Parameters
        ----------
        X:
            Field configuration :math:`\phi`.
        T:
            Temperature in internal units.
        """
        return self.mass_spectrum.bosons_massSq(X, T)

    def fermion_massSq(self, X):
        r"""Return the fermionic mass spectrum :math:`m_f^2(\phi)` of the model.

        The tuple is used in the fermionic one-loop correction

        .. math::
           V_1^{F}(\phi)
           = -\sum_f \frac{n_f\,m_f^4(\phi)}{64\pi^2}
             \left[\log\!\left(\frac{m_f^2(\phi)}{\mu^2}\right)-\tfrac{3}{2}\right].

        Parameters
        ----------
        X:
            Field configuration :math:`\phi`.
        """
        return self.mass_spectrum.fermion_massSq(X)

    # Model initialisation checks -----------------------------------------------------
    def checkInitialisation(self) -> None:
        """Check if all the necessary paramters in the subclass initialisation are set.
        """
        if self.v_GeV is None:
            raise errors.InitPotentialError(f"Need to specify v_GeV parameter!")
        if self.daisy not in ["ArnoldEspinosa", "Parwani", "off"]:
            raise errors.InitPotentialError(
                f"Unknown daisy resummation method: {self.daisy}. Please choose 'ArnoldEspinosa', 'Parwani' or 'off'.")
        if type(self.X0) == type(None):
            raise errors.InitPotentialError(f"Global minimum X0 at zero temperature not specified!")
        if self.renormScaleSq is None:
            raise errors.InitPotentialError(f"Renormalisation scale renormScaleSq not specified!")
        if not isinstance(self.mass_spectrum, MassSpectrum):
            raise errors.InitPotentialError("Mass spectrum not configured. Set `self.mass_spectrum` during init().")

        # Always do tachyon test!
        self.tachyonTest(self.X0)

    # EFFECTIVE POTENTIAL CALCULATIONS -----------------------
    def tachyonTest(self, X0: np.ndarray) -> None:
        """
        Check if the boson mass matrix has any tachyonic modes at the
        zero-temperature minimum. This is used to check if the potential is
        stable at zero temperature.

        Parameters
        ----------
        X0 : array_like
            The zero-temperature minimum of the potential. This should be a
            single point (with length `Ndim`).

        Raises
        ------
        TachyonError
            If any of the boson masses are tachyonic.
        """
        bosons0 = self.boson_massSq(X0, 0.0)
        m20, _, _, is_physical = bosons0

        # Check if any of the non-goldstone-boson scalars are tachyonic (m20 < 0)
        is_tachyonic = np.logical_and(m20 < -1e-5, is_physical == True)

        if is_tachyonic.any():
            msg = "Tachyonic boson mass at zero temperature minimum, Mass matrix: " + str(m20/self.v_stable)
            raise errors.TachyonError(msg)


    def V0(self, X: np.ndarray) -> np.ndarray:
        """Tree level potential. This has to be overwritten by the
        subclass.

        Parameters
        ----------
        X : np.ndarray
            Field values.
        Returns
        -------
        np.ndarray
            Potential values corresponding to the field values.
        """
        return X * 0.0

    def V1(self, bosons, fermions) -> np.ndarray:
        """
        The one-loop corrections to the zero-temperature potential
        using MS-bar renormalization.

        This is generally not called directly, but is instead used by
        :func:`Vtot`.
        """
        m2, n, c, _ = bosons
        y = np.sum(n * m2 * m2 * (np.log(np.abs(m2 / self.renormScaleSq) + 1e-100) - c), axis=-1)

        m2, n = fermions
        c = 3. / 2.
        y -= np.sum(n * m2 * m2 * (np.log(np.abs(m2 / self.renormScaleSq) + 1e-100) - c), axis=-1)

        return y / (64 * np.pi * np.pi)

    def V1phys(self, bosons, fermions) -> np.ndarray:
        """
        The one-loop corrections to the zero-temperature potential
        using MS-bar renormalization, without the goldstone bosons.
        """
        m2, n, c, phys = bosons
        y = np.sum(n * m2 * m2 * phys * (np.log(np.abs(m2 / self.renormScaleSq) + 1e-100) - c), axis=-1)

        m2 = fermions.masses_sq
        if m2.size != 0:
            n = fermions.dof
            c = 3. / 2.
            y -= np.sum(n * m2 * m2 * (np.log(np.abs(m2 / self.renormScaleSq) + 1e-100) - c), axis=-1)

        return y / (64 * np.pi * np.pi)

    def V1_from_X(self, X : np.ndarray) -> np.ndarray:
        """
        The one-loop corrections to the zero-temperature potential
        using MS-bar renormalization.

        This is generally not called directly, but is instead used by
        :func:`Vtot`.
        """
        X = np.asanyarray(X, dtype=float)
        bosons = self.boson_massSq(X, 0.0)
        fermions = self.fermion_massSq(X)
        return self.V1(bosons, fermions)

    def Vct(self, X : np.ndarray) -> np.ndarray:
        """
        The one-loop counterterm potential for MS-bar renormalization.
        This should be overwritten.
        """
        r = 0
        return r

    def check_renorm_conditions(
        self,
        X: np.ndarray | None = None,
        *,
        eps: float | None = None,
        order: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Return derivatives of V1 + Vct at the given field point.

        This provides a quick diagnostic for OS-like counterterm conditions
        that enforce vanishing first and second derivatives of V1 + Vct at the
        reference vacuum.
        """
        if X is None:
            X = np.asarray(self.X0, dtype=float)
        else:
            X = np.asarray(X, dtype=float)

        eps = self.x_eps if eps is None else eps
        order = self.deriv_order if order is None else order

        def V1_plus_ct(Xp):
            bosons = self.boson_massSq(Xp, 0.0)
            fermions = self.fermion_massSq(Xp)
            return self.V1(bosons, fermions) + self.Vct(Xp)

        grad_fn = helper_functions.gradientFunction(
            V1_plus_ct, eps=eps, Ndim=self.Ndim, order=order
        )
        hess_fn = helper_functions.hessianFunction(
            V1_plus_ct, eps=eps, Ndim=self.Ndim, order=order
        )

        grad = np.squeeze(grad_fn(X))
        hess = np.squeeze(hess_fn(X))
        grad_arr = np.atleast_1d(grad)
        hess_arr = np.atleast_2d(hess)
        offdiag = hess_arr - np.diag(np.diag(hess_arr))

        return {
            "grad": grad_arr,
            "hess": hess_arr,
            "max_abs_grad": float(np.max(np.abs(grad_arr))) if grad_arr.size else 0.0,
            "max_abs_hess": float(np.max(np.abs(hess_arr))) if hess_arr.size else 0.0,
            "max_abs_offdiag": float(np.max(np.abs(offdiag))) if offdiag.size else 0.0,
        }

    def V1T(self, bosons, fermions,  T: float | np.ndarray) -> np.ndarray:
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

        """
        # This does not need to be overridden.
        T2 = (T * T)[..., np.newaxis] + 1e-100
        # the 1e-100 is to avoid divide by zero errors
        T4 = T * T * T * T
        msq, dof, _, _ = bosons
        y = np.sum(dof * Jb(msq / T2), axis=-1)
        msq, dof = fermions
        y += np.sum(dof * Jf(msq / T2), axis=-1)
        return y * T4 / (2 * np.pi * np.pi)

    def V1T_from_X(self, X: np.ndarray, T: float | np.ndarray) -> np.ndarray:
        """
        Calculates the mass matrix and resulting one-loop finite-T potential.

        Useful when calculate temperature derivatives, when the zero-temperature
        contributions don't matter.
        """
        T = np.asanyarray(T, dtype=float)
        X = np.asanyarray(X, dtype=float)
        fermions = self.fermion_massSq(X)
        if self.daisy == "off":
            bosons0 = self.boson_massSq(X, 0)
            return self.V1T(bosons0, fermions, T)
        elif self.daisy == "Parwani":
            # Parwani (1992) prescription. All modes resummed.
            bosonsT = self.boson_massSq(X, T)
            return self.V1T(bosonsT, fermions, T)
        elif self.daisy == "ArnoldEspinosa":
            # Carrington (1992), Arnold and Espinosa (1992) prescription. Zero modes only.
            bosons0 = self.boson_massSq(X, 0)
            return self.V1T(bosons0, fermions, T)

    def constantTerms(self, T: float | np.ndarray, include_decoupled=False) -> np.ndarray:
        r"""Add the field-independent, but temperature-dependent terms to the
        effective potential, i.e. radiation. This allows the calculation
        of the energy density as

        .. math::
            \rho = V - \partial_T V

        This should be computed from the particle definitions in the
        potential.

        Parameters
        ----------
        X : array_like
            The field values
        T : float or array_like
            The temperatures
        include_decoupled : bool, optional
            If False, don't include the contribution of the specified decoupled 
            radiation bath specified in `self.kin_decoupled_p_geff`.

        Returns
        -------
        array_like
            The field-independent but temperature-dependent contribution to
            the effective potential.
        """
        T = np.asanyarray(T, dtype=float)
        geff = self.kin_coupled_p_geff(T, self.conversionFactor)

        # in the landau gauge we have -1 massless DOF from each gauge boson
        # (rather from the ghost fields)
        geff -= self.mass_spectrum.number_gauge_bosons
        
        if include_decoupled:
            geff += self.kin_decoupled_p_geff(T, self.conversionFactor)
        return -np.pi**2/90 * geff * T**4

    def Vtot(self, X: np.ndarray, T: float | np.ndarray, include_radiation=True,
             include_decoupled=True) -> np.ndarray:
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
        include_decoupled : bool, optional
            If True, include radiation terms of the specified decoupled
            radiation bath, if false, don't include them. This is important,
            for e.g. the calculation of alpha for a decoupled SM and DS.

        Returns
        ----------
        Vtot : array_like
            Total effective potential.
        """

        T = np.asanyarray(T)
        X = np.asanyarray(X)

        bosons0 = self.boson_massSq(X, T*0.0)
        bosonsT = self.boson_massSq(X, T)
        fermions = self.fermion_massSq(X)

        y = self.V0(X)  # Tree-level 
        y += self.V1(bosons0, fermions)  # One-loop zero-T correction
        y += self.Vct(X)  # Counter-term potential

        if self.daisy == "off":
            y += self.V1T(bosons0, fermions, T)
        elif self.daisy == "Parwani":
            # Parwani (1992) prescription. All modes resummed.
            y += self.V1T(bosonsT, fermions, T)
        elif self.daisy == "ArnoldEspinosa":
            # Carrington (1992), Arnold and Espinosa (1992) prescription. Zero modes only.
            Vdaisy = self.Vdaisy(bosons0, bosonsT, T)
            y += self.V1T(bosons0, fermions, T) + Vdaisy

        # Add field-independent terms, so that the energy density can be computed
        # correctly
        if include_radiation:
            y += self.constantTerms(T, include_decoupled=include_decoupled)

        return y

    def Vdaisy(self, bosons0, bosonsT , T: float | np.ndarray
               ) -> float | np.ndarray:
        """
        Calculate the daisy resummation term for the Arnold-Espinosa
        prescription.

        Args:
            bosons0 (tuple): Boson mass spectrum at zero temperature.
            bosonsT (tuple): Boson mass spectrum at finite temperature.
            T (float): Temperature.

        Returns:
            float: Daisy resummation potential correction to V1T.
        """
        m20, nb, _, _ = bosons0
        m2T, _, _, _ = bosonsT

        y = np.real(-(T/(12.*np.pi))*np.sum(nb * (pow(m2T+0j, 1.5) - pow(m20+0j, 1.5)), axis=-1))
        return y

    def Vdaisy_from_X(self, X,  T: float | np.ndarray,
               realDaisy=False) -> float | np.ndarray:
        """
        Calculate the daisy resummation term for the Arnold-Espinosa
        prescription.

        Args:
            bosons0 (tuple): Boson mass spectrum at zero temperature.
            bosonsT (tuple): Boson mass spectrum at finite temperature.
            T (float): Temperature.
            realDaisy (bool): If True, enforce real-valued calculations.

        Returns:
            float: Daisy resummation potential correction to V1T.
        """
        T = np.asanyarray(T, dtype=float)
        X = np.asanyarray(X, dtype=float)
        bosons0 = self.boson_massSq(X, 0)
        bosons = self.boson_massSq(X, T)
        return self.Vdaisy(bosons0, bosons, T)


    def DVtot(self, X: np.ndarray, T: float | np.ndarray) -> np.ndarray:
        """
        The finite temperature effective potential, but offset
        such that V(0, T) = 0.

        Parameters
        ----------
        X : array_like
            Field value(s).
            Either a single point (with length `Ndim`), or an array of points.
        T : float or array_like
            The temperature. The shapes of `X` and `T`
            should be such that ``X.shape[:-1]`` and ``T.shape`` are
            broadcastable (that is, ``X[...,0]*T`` is a valid operation).

        Returns
        -------
        np.ndarray
            The effective potential at the given field values and temperatures,
            offset such that V(0, T) = 0.
        """
        X = np.array(X)
        return self.Vtot(X, T, False) - self.Vtot(0 * X, T, False)

    def gradV(self, X: np.ndarray, T: float | np.ndarray) -> np.ndarray:
        """
        Find the gradient of the full effective potential.

        This uses :func:`helper_functions.gradientFunction` to calculate the
        gradient using finite differences, with differences
        given by `self.x_eps`. Note that `self.x_eps` is only used directly
        the first time this function is called, so subsequently changing it
        will not have an effect.

        Parameters
        ----------
        X : array_like
            Field value(s).
            Either a single point (with length `Ndim`), or an array of points.
        T : float or array_like
            The temperature. The shapes of `X` and `T`
            should be such that ``X.shape[:-1]`` and ``T.shape`` are
            broadcastable (that is, ``X[...,0]*T`` is a valid operation).

        Returns
        -------
        np.ndarray
            The gradient of the effective potential at the given field values
            and temperatures.
        """
        try:
            f = self._gradV
        except BaseException:
            # Create the gradient function
            self._gradV = helper_functions.gradientFunction(
                self.Vtot, self.x_eps, self.Ndim, self.deriv_order)
            f = self._gradV
        # Need to add extra axes to T since extra axes get added to X in
        # the helper function.
        T = np.asanyarray(T)[..., np.newaxis, np.newaxis]
        return f(X, T, False)

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
        
        Returns
        -------
        np.ndarray
            The gradient of the one-loop zero-temperature potential at the vev.
            
        .. todo:: 
            Currently, this function assumes that the vev is a point in
            one-dimensional field space. Generalize to arbitrary number of
            dimensions.
        """

        try:
            res = self._dV1atvev

        except BaseException:
            # vev means here an arbitrary point in field space
            def V_coleman_weinberg(X):
                bosonsvev0 = self.boson_massSq(X, 0.0)
                fermionsvev = self.fermion_massSq(X)
                return self.V1(bosonsvev0, fermionsvev)

            dV1 = helper_functions.gradientFunction(
                V_coleman_weinberg, eps=self.x_eps, Ndim=self.Ndim, order=self.deriv_order)

            # Use np.squeeze to shrink array sizes from (1,1) and (1,) to a 0d
            # array... otherways problems when evaluating Vtot at more than one
            # point in field space at the same time in the code.

            self._dV1atvev = np.squeeze(dV1(np.array([self.v])))
            res = self._dV1atvev
        return res

    def dV1physatvev(self):
        """
        Find the first derivative of the one loop zero temperature potential without goldstone bosons,
        evaluated at the vev. Used in the function Vct to calculate the
        counter term lagrangian

        This uses :func:`helper_functions.gradientFunction` to calculate the
        gradient using finite differences, with differences
        given by `self.x_eps`. Note that `self.x_eps` is only used directly
        the first time this function is called, so subsequently changing it
        will not have an effect.
        
        Returns
        -------
        np.ndarray
            The gradient of the one-loop zero-temperature potential without
            goldstone bosons at the vev.

        .. todo::
            Currently, this function assumes that the vev is a point in
            one-dimensional field space. Generalize to arbitrary number of
            dimensions.
        """

        try:
            res = self._dV1physatvev
        except BaseException:
            # vev means here an arbitrary point in field space
            def V_coleman_weinberg(X):
                bosonsvev0 = self.boson_massSq(X, 0.0)
                fermionsvev = self.fermion_massSq(X)
                return self.V1phys(bosonsvev0, fermionsvev)
            dV1 = helper_functions.gradientFunction(
                V_coleman_weinberg, eps = self.x_eps, Ndim = self.Ndim, order = self.deriv_order)
           
            # Use np.squeeze to shrink array sizes from (1,1) and (1,) to a 0d
            # array... otherways problems when evaluating Vtot at more than one
            # point in field space at the same time in the code.

            self._dV1physatvev = np.squeeze(dV1(np.array([self.v])))
            res = self._dV1physatvev
        return res

    def dgradV_dT(self, X, T):
        """
        Find the derivative of the gradient with respect to temperature.

        This is useful when trying to follow the minima of the potential as they
        move with temperature.
        
        This uses :func:`helper_functions.gradientFunction` to calculate the
        gradient using finite differences, with differences
        given by `self.x_eps`. Note that `self.x_eps` is only used directly
        the first time this function is called, so subsequently changing it
        will not have an effect.
        
        Parameters
        ----------
        X : array_like
            Field value(s).
            Either a single point (with length `Ndim`), or an array of points.
        T : float or array_like
            The temperature. The shapes of `X` and `T`
            should be such that ``X.shape[:-1]`` and ``T.shape`` are
            broadcastable (that is, ``X[...,0]*T`` is a valid operation).
        
        Returns
        -------
        np.ndarray
            The derivative of the gradient of the effective potential with
            respect to temperature at the given field values and temperatures.
        """
        T_eps = self.T_eps
        try:
            gradVT = self._gradVT
        except BaseException:
            # Create the gradient function
            self._gradVT = helper_functions.gradientFunction(
                self.V1T_from_X, self.x_eps, self.Ndim, self.deriv_order)
            gradVT = self._gradVT
        # Need to add extra axes to T since extra axes get added to X in
        # the helper function.
        T = np.asanyarray(T)[..., np.newaxis, np.newaxis]
        assert (self.deriv_order == 2 or self.deriv_order == 4)
        if self.deriv_order == 2:
            y = gradVT(X, T + T_eps) - gradVT(X, T - T_eps)
            y *= 1. / (2 * T_eps)
        else:
            y = gradVT(X, T - 2 * T_eps)
            y -= 8 * gradVT(X, T - T_eps)
            y += 8 * gradVT(X, T + T_eps)
            y -= gradVT(X, T + 2 * T_eps)
            y *= 1. / (12 * T_eps)
        return y

    def dVdT(self, X : np.ndarray, T: float, dT: float, include_radiation=True,
             include_decoupled=True):
        """Find the derivative of the potential with respect to the temperature.
        
        Parameters
        ----------
        X : np.ndarray
            Field value(s).
            Either a single point (with length `Ndim`), or an array of points.
        T : float
            The temperature.
        dT : float
            The change in temperature.
        include_radiation : bool
            Whether to include radiation contributions.
        include_decoupled : bool
            Whether to include decoupled-radiation contributions.

        Returns
        -------
        np.ndarray
            The derivative of the potential with respect to temperature at the given
            field values and temperatures.
        """
        dV = (
            self.Vtot(X, T - 2*dT, include_radiation=include_radiation,
                      include_decoupled=include_decoupled)
            - 8*self.Vtot(X, T - dT, include_radiation=include_radiation,
                          include_decoupled=include_decoupled)
            + 8*self.Vtot(X, T + dT, include_radiation=include_radiation,
                          include_decoupled=include_decoupled)
            - self.Vtot(X, T + 2*dT, include_radiation=include_radiation,
                        include_decoupled=include_decoupled)
        )
        dV = dV / (12 * dT)
        return dV

    def d2VdT2(self, X : np.ndarray, T: float, dT: float | None = None,
               include_radiation=True, include_decoupled=True) -> np.ndarray:
        """Find the second derivative of the potential with respect to the temperature.
        
        Parameters
        ----------
        X : np.ndarray
            Field value(s).
            Either a single point (with length `Ndim`), or an array of points.
        T : float
            The temperature.
        dT : float, optional
            Temperature step for the finite-difference stencil.  Defaults to
            the tracing temperature accuracy.
        include_radiation : bool
            Whether to include radiation contributions.
        include_decoupled : bool
            Whether to include decoupled-radiation contributions.
        
        Returns
        -------
        np.ndarray
            The second derivative of the potential with respect to temperature at the given
            field values and temperatures.
        """
        if dT is None:
            dT = self.T_eps
        ddV = (
            -self.Vtot(X, T - 2*dT, include_radiation=include_radiation,
                       include_decoupled=include_decoupled)
            + 16*self.Vtot(X, T - dT, include_radiation=include_radiation,
                           include_decoupled=include_decoupled)
            - 30*self.Vtot(X, T, include_radiation=include_radiation,
                           include_decoupled=include_decoupled)
            + 16*self.Vtot(X, T + dT, include_radiation=include_radiation,
                           include_decoupled=include_decoupled)
            - self.Vtot(X, T + 2*dT, include_radiation=include_radiation,
                        include_decoupled=include_decoupled)
        ) / (12*dT**2)
        return ddV

    def massSqMatrix(self, X):
        """
        Calculate the tree-level mass square matrix of the scalar field.

        This uses :func:`helper_functions.hessianFunction` to calculate the
        matrix using finite differences, with differences
        given by `self.x_eps`. Note that `self.x_eps` is only used directly
        the first time this function is called, so subsequently changing it
        will not have an effect.

        The resulting matrix will have rank `Ndim`. This function may be useful
        for subclasses in finding the boson particle spectrum.
        
        Parameters
        ----------
        X : array_like
            Field value(s).
            Either a single point (with length `Ndim`), or an array of points.
            
        Returns
        -------
        np.ndarray
            The tree-level mass square matrix at the given field values.
        """
        try:
            f = self._massSqMatrix
        except BaseException:
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
        
        Parameters
        ----------
        X : array_like
            Field value(s).
            Either a single point (with length `Ndim`), or an array of points.
        T : float or array_like
            The temperature. The shapes of `X` and `T`
            should be such that ``X.shape[:-1]`` and ``T.shape`` are
            broadcastable (that is, ``X[...,0]*T`` is a valid operation).
        
        Returns
        -------
        np.ndarray
            The Hessian matrix of the effective potential at the given field
            values and temperatures.
        """
        try:
            f = self._d2V
        except BaseException:
            # Create the gradient function
            self._d2V = helper_functions.hessianFunction(
                self.Vtot, self.x_eps, self.Ndim, self.deriv_order)
            f = self._d2V
        # Need to add extra axes to T since extra axes get added to X in
        # the helper function.
        T = np.asanyarray(T)[..., np.newaxis]
        return f(X, T, False)

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
        
        Returns
        -------
        np.ndarray
            The Hessian of the one-loop zero-temperature potential at the vev.
            
        .. todo::
            Currently, this function assumes that the vev is a point in
            one-dimensional field space. Generalize to arbitrary number of
            dimensions.
        """

        try:
            res = self._d2V1atvev
        except BaseException:
            # vev means here an arbitrary point in field space
            def V_coleman_weinberg(X):
                bosonsvev0 = self.boson_massSq(X, 0.0)
                fermionsvev = self.fermion_massSq(X)
                return self.V1(bosonsvev0, fermionsvev)

            d2V1 = helper_functions.hessianFunction(
                V_coleman_weinberg, eps=self.x_eps, Ndim=self.Ndim, order=self.deriv_order)

            # Use np.squeeze to shrink array sizes from (1,1) and (1,) to a 0d
            # array... otherways problems when evaluating Vtot at more than one
            # point in field space at the same time in the code.
            self._d2V1atvev = np.squeeze(d2V1(np.array([self.v])))
            res = self._d2V1atvev
        return res

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
        return [np.ones(self.Ndim) * self.renormScaleSq**.5]

    def findMinimum(self, X : np.ndarray=None, T : float=0.0) -> np.ndarray:
        """
        Convenience function for finding the nearest minimum to `X` at
        temperature `T`.
        
        Parameters
        ----------
        X : np.ndarray, optional
            Starting point for the minimization. If None, use the
            approximate zero-temperature minimum.
        T : float, optional
            Temperature at which to find the minimum. Default is 0.0.
            
        Returns
        -------
        np.ndarray
            The location of the minimum found.
        """
        if X is None:
            X = self.approxZeroTMin()[0]
        return optimize.fmin(self.Vtot, X, args=(T,), disp=0)

    def generateInvGroupElements(self) -> None:
        r"""Generate the group elements under which the potential
        is invariant. The group is

        .. math::
            G = \Pi_i^N \mathbb{Z}_2

        representing the discrete symmetries of the potential with scalar field
        dimension N.

        .. todo:: test, refactor and clean up

        Parameters
        ----------

        Returns
        -------
        None."""
        # First generate the elements of the nplet of C2
        groupElements = [np.identity(self.Ndim)]
        for i in range(self.Ndim):
            oldGroupElements = groupElements.copy()
            for g in oldGroupElements:
                gnew = g.copy()
                gnew[i, i] = -1
                groupElements.append(gnew)

        # Now check under which of these the potential is invariant
        # For this we take a random point at positive field values,
        # apply the transformations \phi -> g \phi with g \element G and
        # check if Vtree(\phi) = Vtree(g \phi)

        # For this we check 5 points
        Npoints = 5
        transformedThreshold = 1e-5  # relative difference between the pot values
        randPoints = (0.01 + np.random.rand(Npoints, self.Ndim)*0.49) * self.v
        randT = self.Tmin + (np.random.rand(Npoints) * (self.Tmax - self.Tmin))

        invGroupElements = []
        for i in range(len(groupElements)):
            g = groupElements.pop()
            V = self.Vtot(randPoints, randT) + 1e-20
            Vbar = self.Vtot(randPoints @ g, randT) + 1e-20
            if (np.abs((V - Vbar)/V) < transformedThreshold).all():
                invGroupElements.append(g)

        self.invGroupElements = invGroupElements

    def applySymmetries(self, X: np.ndarray) -> np.ndarray:
        r"""Check if the point is in the right quadrant of field space
        and transform it if necessary.

        This is done in the fashion of BSMPT with the measure

        ..math::
            M(g \phi) = \sum_i 2^i \theta((g \phi)_i)

        This measure is maximised for one unique 'quadrant' in fieldspace.

        Parameters
        ----------
        X : np.ndarray
            Point in field space.

        Returns
        -------
        np.ndarray
            The transformed point."""
        maxMeasure = 2**self.Ndim - 1

        measure = -1  # Store the largest measure
        Xmax = X.copy()
        for g in self.invGroupElements:
            newMeasure = 0
            XX = X @ g
            for i in range(self.Ndim):
                newMeasure += 2**i * np.heaviside(XX[i], 1)
            if newMeasure > measure:
                measure = newMeasure
                Xmax = XX.copy()
            if measure == maxMeasure:
                break
        return Xmax

    def nucleationCriterion(self, S: float, T: float, high_phase, low_phase) -> float:
        r"""Approximate nucleation criterion

        .. math::
            \Gamma(T_\mathrm{nuc})/H(T_\mathrm{nuc})^4 = 1

        Return 0 if nuclation is fulfilled

        Parameters
        ----------
        S : float
            Action at temperature T
        T : float
            Temperature.
        high_phase : PhaseInfo
            High temperature phase, start of tunneling
        low_phase : PhaseInfo
            Low temperature phase

        Returns
        -------
        float
            Criterion.
        """
        return approxNucleationCriterion(T, S, self, high_phase, low_phase)

    def radiationEnergyDensity(self, X: np.ndarray, T: float, include_decoupled=True) -> float:
        r"""Return the energy density in radiation that is not
        field dependent.

        Parameters
        ----------
        X : np.ndarray
            The scalar field values
        T : float
            Temperature
        include_decoupled : bool, optional
            If true, include the enery density of the decoupled radiation bath

        Returns
        -------
        float :
            The energy density in pure radiation at temperature T."""

        # Radiation energy density of particles that interact with the
        # scalars from the potential
        T2 = T * T
        bosons0 = self.boson_massSq(X, 0.0)
        fermions = self.fermion_massSq(X)
        m2b, nb, _, _ = bosons0
        m2f, nf = fermions
        # This below here is just T dP/dT - P written in terms of the thermal functions
        VTpart = np.sum(-3 * nb * T2 * T2 * Jb(m2b / T2) + 2 * m2b * nb * T2 * Jb(m2b / T2, n=1), axis=-1)
        VTpart += np.sum(-3 * nf * T2 * T2 * Jf(m2f / T2) + 2 * m2f * nf * T2 * Jf(m2f / T2, n=1), axis=-1)

        eRadPotential = VTpart / (2 * np.pi**2)

        # Radiation energy density of particles that are neglected in the
        # effective potential
        geff = self.kin_coupled_e_geff(T, self.conversionFactor)

        # remove unphysical contributions from the gauge bosons (effect of ghost fields)
        # in the landau gauge
        geff -= self.mass_spectrum.number_gauge_bosons

        # Possible additional energy density from decoupled sectors
        if include_decoupled:
            geff += self.kin_decoupled_e_geff(T, self.conversionFactor)

        eRadAdditional = np.pi**2 / 30 * geff * T**4

        return eRadPotential + eRadAdditional

    def energyDensityDaisy(self, X: np.ndarray, T: float | np.ndarray):
        r"""Return the daisy-resummation contribution to the energy density.

        Subclasses with a non-trivial daisy thermal mass can override this.
        The default is zero and preserves the previous behaviour.
        """
        return np.asarray(X)[..., 0] * 0.0

    def energyDensity(self, X: np.ndarray, T: float | np.ndarray,
                      include_decoupled: bool = True) -> float:
        r"""Calculate the total energy density.
        It is important that Vtot includes the field independent
        terms (i.e. radiation terms which are prop to T^4).

        .. math::
            e = V - T \partial_T V

        Parameters
        ----------
        pot : generic_potential
            Effective potential
        X : np.ndarray
            The scalar field values
        T : float|np.ndarray
            The temperature at which to compute the energy density
        include_decoupled : bool, optional
            If true, include the energy density of the decoupled radiation bath
            as well.

        Returns
        -------
        float|np.ndarray :
            The energy density at `T`."""

        V0 = self.V0(X) + self.Vct(X) + self.V1_from_X(X)
        DV0 = V0 - (self.V0(self.X0) + self.Vct(self.X0) + self.V1_from_X(self.X0))
        # Numerical stability:
        if DV0/np.abs(V0) < 1e-10:
            DV0 = 0

        etot = DV0 + self.energyDensityDaisy(X, T)
        if T != 0.0:
            etot += self.radiationEnergyDensity(X, T, include_decoupled)
        return etot

    def makePrettyDictionaryPrint(self, derived_dict: dict={}) -> None:
        """Print the potential information in a nice way.

        Parameters
        ----------
        input_dict : dict
            Input parameters of the potential
        derived_dict
            Parameters derived from the input
        Returns
        -------
        None."""
        input_table = rich.table.Table(title="Input parameters", title_justify="left", box=rich.box.ROUNDED)
        input_table.add_column("Parameter", style="cyan", no_wrap=True)
        input_table.add_column("Value", style="orange1")
        for n, v in self.mp.items():
            value = v["value"]
            input_table.add_row(n, f"{value:.10e}")

        tables = [input_table]

        if derived_dict != {}:
            derived_table = rich.table.Table(title="Derived parameters", title_justify="left", box=rich.box.ROUNDED)
            derived_table.add_column("Parameter", style="cyan", no_wrap=True)
            derived_table.add_column("Value", style="orange1")
            for n, v in derived_dict.items():
                derived_table.add_row(n, f"{v:.10e}")

            tables.append(derived_table)

        mass_entries = self.get_zero_temperature_mass_spectrum()
        
        if mass_entries:
            mass_table = rich.table.Table(
                title="Zero-temperature mass spectrum", title_justify="left", box=rich.box.ROUNDED
            )
            mass_table.add_column("Particle", style="cyan", no_wrap=True)
            mass_table.add_column("Type", style="magenta")
            mass_table.add_column("Mass [GeV]", style="orange1")
            for entry in mass_entries:
                text_label = entry.get("text", "")
                mass_value = entry.get("mass_GeV", np.nan)
                mass_table.add_row(
                    text_label,
                    entry.get("kind", ""),
                    f"{mass_value:.10e}" if np.isfinite(mass_value) else "-",
                )
            tables.append(mass_table)

            console.print(RichColumns(tables))


    def get_mass_spectrum_T0(self):
        """description

        Parameters
        ----------

        Returns
        -------

        """
        spectrum = self.get_mass_spectrum(self.X0, 0.0)
        counts = {"boson": 0, "fermion": 0}
        res = []
        for kind, sector in (("boson", spectrum.bosons), ("fermion", spectrum.fermions)):
            masses_sq = np.real_if_close(np.asarray(sector.masses_sq))
            masses_sq = masses_sq.reshape(-1)
            if masses_sq.size == 0:
                continue

            latex_labels = sector.latex_labels
            text_labels = sector.text_labels

            for idx, mass_sq in enumerate(masses_sq):
                with np.errstate(invalid="ignore"):
                    internal_mass = np.sqrt(mass_sq) if mass_sq >= 0 else np.nan
                mass_GeV = float(internal_mass * self.conversionFactor) if np.isfinite(internal_mass) else np.nan
                log_mass = float(np.log10(mass_GeV)) if np.isfinite(mass_GeV) and mass_GeV > 0 else np.nan

                kind_index = counts[kind]
                counts[kind] += 1

                entry: dict[str, object] = {
                    "kind": kind,
                    "index": kind_index,
                    "mass_GeV": mass_GeV,
                    "log10_mass": log_mass,
                    "latex": latex_labels[idx] if idx < len(latex_labels) else latex_labels[-1] if latex_labels else "",
                    "text": text_labels[idx] if idx < len(text_labels) else text_labels[-1] if text_labels else "",
                }
                res.append(entry)

        return res


    def get_zero_temperature_mass_spectrum(self) -> list[dict[str, object]]:
        """Return the T=0 mass spectrum in GeV for bosons and fermions."""

        conversion_factor = getattr(self, "conversionFactor", 1.0)
        try:
            conversion_factor = float(conversion_factor)
        except (TypeError, ValueError):  # pragma: no cover - defensive guard
            conversion_factor = 1.0

        X0 = getattr(self, "X0", None)
        if X0 is None and hasattr(self, "approxZeroTMin"):
            try:
                minima = self.approxZeroTMin()
            except Exception:  # pragma: no cover - defensive safeguard
                minima = None
            if minima:
                X0 = minima[0]

        if X0 is None:
            return []

        try:
            snapshot = self.get_mass_spectrum(np.asarray(X0), 0.0)
        except Exception:  # pragma: no cover - keep spectrum extraction resilient
            return []

        results: list[dict[str, object]] = []
        counts = {"boson": 0, "fermion": 0}

        for kind, sector in (("boson", snapshot.bosons), ("fermion", snapshot.fermions)):
            masses_sq = np.real_if_close(np.asarray(sector.masses_sq))
            masses_sq = masses_sq.reshape(-1)
            if masses_sq.size == 0:
                continue

            latex_labels = sector.latex_labels
            text_labels = sector.text_labels

            for idx, mass_sq in enumerate(masses_sq):
                with np.errstate(invalid="ignore"):
                    internal_mass = np.sqrt(mass_sq) if mass_sq >= 0 else np.nan
                mass_GeV = float(internal_mass * conversion_factor) if np.isfinite(internal_mass) else np.nan
                log_mass = float(np.log10(mass_GeV)) if np.isfinite(mass_GeV) and mass_GeV > 0 else np.nan

                kind_index = counts[kind]
                counts[kind] += 1

                entry: dict[str, object] = {
                    "kind": kind,
                    "index": kind_index,
                    "mass_GeV": mass_GeV,
                    "log10_mass": log_mass,
                    "latex": latex_labels[idx] if idx < len(latex_labels) else latex_labels[-1] if latex_labels else "",
                    "text": text_labels[idx] if idx < len(text_labels) else text_labels[-1] if text_labels else "",
                }
                results.append(entry)

        return results
