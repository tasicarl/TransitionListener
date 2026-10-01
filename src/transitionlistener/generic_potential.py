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
from transitionlistener.thermodynamics import (e_geffSM, p_geffSM, sm_bath,
                                               sm_fields_in_potential_geff,
                                               sm_fields_in_potential_geff_uncached)

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
    broadcast_fields,
)


def _daisy_mass_cubed_difference(m20, m2T, Pi=None):
    r"""Return ``(m_T^2)^{3/2} - (m_0^2)^{3/2}`` without losing significant digits.

    The daisy term of the Arnold-Espinosa resummation is proportional to this difference.
    Written out directly it subtracts two nearly equal numbers whenever the thermal
    correction is small against the mass itself, which is the whole low-temperature regime:
    on the conformal dark U(1) of ``examples/example_point.yaml`` at an internal temperature
    of 0.02, the two cubes are about ``1e9`` in internal units while their difference is
    about ``0.3``, so ten of the sixteen digits are gone before anything else happens. The
    potential itself tolerates that; its second temperature derivative does not, and the
    sound speed divides by exactly that second derivative. Measured there, the second
    derivative of the daisy term used to change by 60% when the derivative step was changed
    by a factor ten, and now changes by 2e-11.

    Two things are needed. The difference is written as
    ``(m_0^2)^{3/2} [(1 + u)^{3/2} - 1]`` with ``u = Pi/m_0^2``, and the bracket is evaluated
    as ``expm1(1.5 log1p(u))``, which has no cancellation. And that evaluation is kept in
    real arithmetic: numpy's complex ``log1p`` and ``expm1`` lose accuracy for small
    arguments, by a relative ``8e-8`` at ``u = 1e-9`` against ``1e-16`` for the real ones, so
    the ``+0j`` that exists only to admit tachyonic modes must not be applied to the modes
    that are fine. Negative ``m_0^2``, and a thermal correction that is not small, keep the
    direct complex form, where there is nothing to cancel.
    """
    m20 = np.asarray(m20)
    m2T = np.asarray(m2T)
    shape = np.broadcast(m20, m2T).shape
    m20b, m2Tb = np.broadcast_arrays(m20, m2T)
    m20r = np.real(m20b)
    with np.errstate(divide="ignore", invalid="ignore"):
        positive = np.asarray(m20r > 0.0)
        u = np.zeros(shape, dtype=float)
        # `Pi` given analytically is exact; taken as a difference it keeps only the digits
        # by which the two spectra differ, and underflows to zero at low temperature.
        thermal = (np.real(np.broadcast_to(Pi, shape)) if Pi is not None
                   else np.real(m2Tb) - m20r)
        np.divide(thermal, m20r, out=u, where=positive)
        # Only where the correction is genuinely small is the rearrangement worth it; the
        # threshold is where the direct form still has about half its digits.
        small = positive & (np.abs(u) < 1.0e-2)
    # `Vtot` is called hundreds of thousands of times in a scan and the spectra are a handful
    # of modes, so the masked, two-branch general path costs more in numpy overhead than the
    # arithmetic in it. The two cases that actually occur are all modes small, which is every
    # supercooled evaluation, and none small, which is the high-temperature one; both are
    # taken whole, in one array operation, and the mixed case keeps the general path.
    all_small = bool(small.all())
    if all_small:
        return (m20r ** 1.5 * np.expm1(1.5 * np.log1p(u))).astype(complex, copy=False)
    hot = (m20r + np.real(np.broadcast_to(Pi, shape))) if Pi is not None else m2Tb
    if not small.any():
        return pow(hot + 0j, 1.5) - pow(m20b + 0j, 1.5)
    out = np.empty(shape, dtype=complex)
    base = m20r[small] ** 1.5
    out[small] = base * np.expm1(1.5 * np.log1p(u[small]))
    rest = ~small
    out[rest] = pow(hot[rest] + 0j, 1.5) - pow(m20b[rest] + 0j, 1.5)
    return out


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

    #: Finite-difference step, in internal field units, for the derivatives of
    #: the zero-temperature one-loop potential at the vev that fix the
    #: counterterms (``dV1atvev``, ``d2V1atvev``, ``dV1physatvev``). It is kept
    #: separate from the phase-tracing accuracy ``x_eps`` on purpose: tightening a
    #: tracing tolerance must not change the potential. For the abelian dark
    #: Higgs, with |V_CW| of order 1e9 at the vev, steps below 1e-3 are dominated
    #: by round-off (1e-4 shifts the second derivative by up to 1.4 %). Above it
    #: the second derivative drifts with the logarithm of the step, by 0.05 % to
    #: 0.5 % per decade: the tree-level Goldstone mass vanishes at the vacuum, so
    #: its Coleman-Weinberg term is infrared divergent there and the step acts as
    #: the regulator. The physical cure is the resummation of the Goldstone
    #: contributions, m_G^2 -> m_G^2 + (1/phi) dV_1,phys/dphi (arXiv:1406.2652,
    #: arXiv:1406.2355). 1e-3 is the step these counterterms have always been
    #: computed with in the default setup. Models may override it.
    counterterm_derivative_step: float = 1.0e-3

    def __init__(self, *args, **dargs) -> None:
        """Prepare configuration objects and delegate to :meth:`init` for setup."""

        self.config = Configuration()  # This contains all the tracing settings, etc

        self.kin_coupled_e_geff = e_geffSM  # This can be overwritten
        self.kin_coupled_p_geff = p_geffSM  # This can be overwritten

        self.kin_decoupled_e_geff = lambda T, cf: 0.0 * T  # If the SM is decoupled, put e_geffSM here
        self.kin_decoupled_p_geff = lambda T, cf: 0.0 * T  # If the SM is decoupled, put p_geffSM here
        # Bath that holds the Standard Model, "coupled" or "decoupled". None takes the bath
        # whose e_geff is e_geffSM, and the coupled one if neither is.
        self.SM_bath = None

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
        self._check_radiation_baths()

        if self.verbose:
            self.makePrettyDictionaryPrint(self.derived_parameters)

    def _check_radiation_baths(self) -> None:
        """Validate ``SM_bath``, and the Standard Model's place among the radiation baths.

        A potential with Standard Model fields (flagged ``is_SM``, e.g. the 2HDM) contains the
        Standard Model in its transitioning sector, so the Standard Model cannot be decoupled
        from it. Warns if both baths hold the Standard Model table.
        """
        spectrum = getattr(self, "mass_spectrum", None)
        has_sm_fields = spectrum is not None and bool(
            np.any(spectrum.is_SM_bosons) or np.any(spectrum.is_SM_fermions)
        )
        if sm_bath(self) == "decoupled" and has_sm_fields:
            raise ValueError(
                "The potential contains Standard Model fields (flagged is_SM), so the Standard "
                "Model belongs to the transitioning sector and cannot be put into the decoupled "
                "bath. The decoupled bath can hold other radiation, e.g. of a dark sector."
            )
        if has_sm_fields and (self.kin_coupled_e_geff is not e_geffSM
                              or self.kin_coupled_p_geff is not p_geffSM):
            # The Standard Model fields of the potential are subtracted from the coupled bath,
            # so that bath has to contain them. Check it, in energy and in pressure, rather
            # than only saying so: a bath that is smaller than the subtraction would give a
            # negative radiation energy density or pressure.
            shortfall = None
            try:
                for T_GeV in (1.0, 10.0, 100.0, 1000.0):
                    T = T_GeV / self.conversionFactor
                    for kind, bath in (("energy", self.kin_coupled_e_geff(T, self.conversionFactor)),
                                       ("pressure", self.kin_coupled_p_geff(T, self.conversionFactor))):
                        needed = float(sm_fields_in_potential_geff_uncached(self, T, kind[0]))
                        missing = needed - float(np.squeeze(bath))
                        if missing > 0.0 and (shortfall is None or missing > shortfall[3]):
                            shortfall = (T_GeV, kind, float(np.squeeze(bath)), missing, needed)
            except Exception as exc:
                print(
                    "Warning: the potential contains Standard Model fields (flagged is_SM), "
                    "which are subtracted from the coupled radiation bath so that they are "
                    "counted once, but the bath could not be evaluated to check that it "
                    f"contains them: {exc}"
                )
            else:
                if shortfall is not None:
                    T_GeV, kind, bath_value, _, needed = shortfall
                    print(
                        "Warning: the potential contains Standard Model fields (flagged "
                        "is_SM), which are subtracted from the coupled radiation bath so that "
                        "they are counted once, but the coupled bath is smaller than what is "
                        f"subtracted: at {T_GeV:g} GeV it provides {bath_value:.4g} {kind} "
                        f"degrees of freedom against the {needed:.4g} of those fields, so the "
                        "radiation of the coupled bath comes out negative. Put the Standard "
                        "Model into the coupled bath, or do not flag those fields is_SM."
                    )
        if self.kin_coupled_e_geff is e_geffSM and self.kin_decoupled_e_geff is e_geffSM:
            print(
                "Warning: both the coupled and the decoupled radiation bath hold the Standard "
                "Model table e_geffSM, so the Standard Model is counted twice. To decouple it, "
                "also set kin_coupled_e_geff and kin_coupled_p_geff to zero."
            )

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
        # The Standard Model fields of the potential are tabulated per model and keyed on what
        # they are built from; changing the parameters invalidates that table, as it used to
        # reset the temperature cap of the Standard Model tables.
        self._sm_fields_geff_splines = {}
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
        # Standard Model fields of the potential are counted there, with their field- and
        # temperature-dependent masses, so remove them from the tabulated bath.
        geff = geff - sm_fields_in_potential_geff(self, T, "p")

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
        X = broadcast_fields(X, T)

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
            Vdaisy = self.Vdaisy(bosons0, bosonsT, T, Pi=self.debye_massSq(X, T))
            y += self.V1T(bosons0, fermions, T) + Vdaisy

        # Add field-independent terms, so that the energy density can be computed
        # correctly
        if include_radiation:
            y += self.constantTerms(T, include_decoupled=include_decoupled)

        return y

    def V_thermal(self, X: np.ndarray, T: float | np.ndarray,
                  include_decoupled: bool = False) -> float | np.ndarray:
        """The temperature-dependent part of the effective potential, on its own.

        ``Vtot`` is ``V0 + V1 + Vct`` plus this. The first three do not depend on the
        temperature, so they drop out of every temperature derivative analytically, but not
        numerically: forming the sum first and differencing afterwards subtracts the vacuum
        energy from itself, and under supercooling the thermal part is ten or more orders of
        magnitude smaller. On the conformal dark U(1) of ``examples/example_point.yaml`` the
        thermal part is ``7e-4`` of the vacuum energy at the percolation temperature and
        ``1e-13`` of it two decades below, where a second temperature derivative of the whole
        potential has no significant digits left.
        """
        Ta = np.asarray(T, dtype=float)
        # `V1T_from_X` is the hook this class already documents for models that rewrite the
        # effective potential: "should only return the temperature-dependent part of Vtot".
        # Where a model has overridden it, that is its thermal part by definition and
        # rebuilding one from the mass spectrum would compute the sound speed from different
        # thermodynamics than the rest of the run. The base implementation is not used, because
        # it omits the daisy term, which is the whole subject of this branch.
        if type(self).V1T_from_X is not generic_potential.V1T_from_X:
            return (self.V1T_from_X(X, Ta)
                    + self.constantTerms(Ta, include_decoupled=include_decoupled))
        bosons0 = self.boson_massSq(X, Ta * 0.0)
        bosonsT = self.boson_massSq(X, Ta)
        fermions = self.fermion_massSq(X)
        if self.daisy == "off":
            y = self.V1T(bosons0, fermions, Ta)
        elif self.daisy == "Parwani":
            y = self.V1T(bosonsT, fermions, Ta)
        elif self.daisy == "ArnoldEspinosa":
            y = self.V1T(bosons0, fermions, Ta) + self.Vdaisy(
                bosons0, bosonsT, Ta, Pi=self.debye_massSq(X, Ta))
        else:
            raise errors.PotentialError(
                f"Unknown daisy resummation scheme {self.daisy!r}.")
        return y + self.constantTerms(Ta, include_decoupled=include_decoupled)

    def dV_thermal_dT(self, X: np.ndarray, T: float, dT: float,
                      include_decoupled: bool = False):
        """First temperature derivative of :func:`V_thermal`, by central difference.

        ``include_decoupled`` selects the same sector the caller's other thermodynamics uses;
        a derivative taken on one plasma and combined with a quantity taken on another is not
        a thermodynamic identity. The sound speed excludes the decoupled bath, which is the
        default; the wall velocity follows ``GWConf.coupled_hydrodynamics``.
        """
        return ((self.V_thermal(X, T + dT, include_decoupled=include_decoupled)
                 - self.V_thermal(X, T - dT, include_decoupled=include_decoupled))
                / (2.0 * dT))

    def d2V_thermal_dT2(self, X: np.ndarray, T: float, dT: float,
                        include_decoupled: bool = False):
        """Second temperature derivative of :func:`V_thermal`, by central difference."""
        return ((self.V_thermal(X, T + dT, include_decoupled=include_decoupled)
                 - 2.0 * self.V_thermal(X, T, include_decoupled=include_decoupled)
                 + self.V_thermal(X, T - dT, include_decoupled=include_decoupled))
                / (dT * dT))

    def daisy_outside_validity(self, X: np.ndarray, T: float) -> tuple[bool, float]:
        """Whether the daisy resummation is being used where it does not apply.

        The Arnold-Espinosa term resums the zero Matsubara mode of the bosons, which is
        justified where that mode is infrared-enhanced, meaning ``m << T``. Where instead
        ``m >> T`` a mode should be Boltzmann suppressed, and the one-loop thermal integral
        is: ``V1T`` goes to exactly zero. The daisy term does not. With a thermal correction
        ``Pi`` small against ``m^2`` it tends to ``-(T^3/8 pi) sum n_i c_i m_i``, a power law
        with no ``exp(-m/T)`` in it, so it survives where the physics says it should not and
        eventually exceeds the radiation that is still relativistic.

        Measured on the conformal dark U(1) of ``examples/example_point.yaml``: the daisy term
        is 7% of the field-independent radiation at ``T/v = 1e-1``, equal to it at about
        ``1e-2``, thirteen times it at ``1e-3`` and ten million times it at ``1e-9``, while
        the lightest mode runs from ``m/T = 1.3`` to ``1.3e8``. Where it dominates, the free
        energy goes as ``T^3`` rather than ``T^4`` and the sound speed of the plasma tends to
        ``1/sqrt(2)`` instead of the ``1/sqrt(3)`` of the radiation that is actually there.
        A sound speed above ``1/sqrt(3)`` in a late, cold, heavily supercooled phase is
        therefore this, and not a property of the plasma.

        This is reported and not corrected. Which resummation a model uses is the user's
        choice, and a prescription that is Boltzmann suppressed at low temperature is a change
        to the thermodynamics rather than a guard; it is left to a change of its own.

        Returns ``(outside, ratio)`` with the ratio of the daisy term to the field-independent
        radiation. ``(False, nan)`` where it cannot be decided, and where the model is not using
        this resummation at all: with ``daisy`` set to ``"Parwani"`` or ``"off"`` there is no
        Arnold-Espinosa term in the potential, so building one here and reporting its size would
        describe a prescription the run does not use.
        """
        if getattr(self, "daisy", None) != "ArnoldEspinosa":
            return False, float("nan")
        try:
            Ta = np.asarray([float(T)], dtype=float)
            X = np.asarray(X)
            bosons0 = self.boson_massSq(X, Ta * 0.0)
            bosonsT = self.boson_massSq(X, Ta)
            daisy = float(np.squeeze(self.Vdaisy(bosons0, bosonsT, Ta,
                                                 Pi=self.debye_massSq(X, Ta))))
            bath = float(np.squeeze(self.constantTerms(Ta, include_decoupled=False)))
            # Modes that contribute to the daisy term: degrees of freedom, a mass squared
            # that is not negative, and a Debye mass. All three are needed. A massless mode
            # with a Debye mass is kept, because it is the lightest there is and dropping it
            # both misreads the lightest mass and, in a symmetric phase where every
            # zero-temperature mass vanishes, leaves nothing to take a minimum over, so the
            # ratio is discarded along with it. A mode with no Debye mass is dropped however
            # many degrees of freedom it has, because it adds exactly nothing to the term
            # being judged: the transverse gauge bosons are massless and thermally uncorrected
            # in every model here, and counting them pinned the lightest mass at zero and made
            # the flag unable to fire at all.
            #
            # Whether a mode has a Debye mass is asked at the model's own scale rather than at
            # `T`. It is a property of the mode, and at the temperatures this diagnostic
            # exists for the difference of the two spectra has underflowed, which would make
            # every mode look uncorrected.
            m2 = np.ravel(np.asarray(bosons0[0], dtype=float))
            dof = np.ravel(np.broadcast_to(np.asarray(bosons0[1], dtype=float), m2.shape))
            reference = float(getattr(self, "v_stable", 0.0) or 0.0) or float(T)
            Tref = np.asarray([reference], dtype=float)
            thermal_part = self.debye_massSq(X, Tref)
            if thermal_part is None:
                thermal_part = (np.asarray(self.boson_massSq(X, Tref)[0], dtype=float)
                                - np.asarray(self.boson_massSq(X, Tref * 0.0)[0], dtype=float))
            thermal_part = np.ravel(np.asarray(thermal_part, dtype=float))
            if thermal_part.shape != m2.shape:
                return False, float("nan")
            contributing = (np.isfinite(m2) & (m2 >= 0.0) & (dof != 0.0)
                            & np.isfinite(thermal_part) & (thermal_part != 0.0))
            m2 = m2[contributing]
            if m2.size == 0 or not np.isfinite(daisy) or not np.isfinite(bath) or bath == 0.0:
                return False, float("nan")
            ratio = abs(daisy) / abs(bath)
            lightest_over_T = float(np.sqrt(m2.min())) / float(T)
            # Both conditions: at high temperature the daisy term is legitimate and also
            # goes as T^4, so a large ratio alone is not the signature.
            return bool(ratio > 1.0 and lightest_over_T > 1.0), float(ratio)
        except errors.Timeout:
            # A diagnostic must not cancel the abort it is being run inside of.
            raise
        except Exception:
            return False, float("nan")

    def debye_massSq(self, X: np.ndarray, T: float | np.ndarray):
        """The thermal part of the boson masses squared, if the model can give it directly.

        Returns an array shaped like the boson mass spectrum, or ``None``. ``None`` is not the
        default: the base class measures the thermal part, as described below, and returns
        ``None`` only where that measurement fails its own check. A model that states its Debye
        masses in closed form should override this; where neither the override nor the
        measurement is available, the daisy term falls back to the difference of the spectra at
        ``T`` and at zero, which is what it used before.

        That difference is badly conditioned, and the daisy term is the one place it matters.
        On the conformal dark U(1) of ``examples/example_point.yaml``, at an internal
        temperature of 0.02, the masses squared are about ``5e4`` while their thermal part is
        about ``5e-5``, so the subtraction keeps seven of sixteen digits; by an internal
        temperature of ``1e-7`` it keeps none and the thermal part underflows to exactly zero,
        which silently removes the daisy term altogether. A model that builds its masses as
        ``m^2(X) + Pi(T)`` already holds ``Pi`` and can hand it over exactly, and should
        override this with the closed form.

        The default measures it instead of declining. Where the thermal part is
        ``Pi = c(X) T^2``, which is what the Arnold-Espinosa Debye masses are, the coefficient
        can be read off at a reference temperature high enough for the subtraction to keep its
        digits and then evaluated at any temperature. The reference is the model's own scale
        ``v_stable``, where ``Pi`` is of order the masses themselves.

        That the thermal part goes as ``T^2`` is assumed by nothing here: it is checked, at a
        second reference temperature and at two field points, and the default declines
        whenever the check fails.

        The check is not a formality, and neither is doing it at two field points. Of the
        eight models shipped in ``models/``, four pass and four fail, and the failures are all
        the same physics: where the bosonic masses are eigenvalues of a mass matrix whose
        *entries* go as ``T^2``, the eigenvalues do not. Measured on ``models/TL_2HDM.py``,
        the two longitudinal gauge modes move their apparent coefficient by 4.6% and 11%
        between ``T`` and ``T/2``. ``models/TL_dark_flipflop.py`` fails only where both of its
        fields are on: along ``X0``, where the second vanishes and the matrix is diagonal, it
        looks quadratic to ``7e-18``, and at a generic point it is off by ``5e-3``. A check at
        one field point would have accepted it and handed the daisy term a ``Pi`` that is
        wrong by half a per cent.

        For those models there is no ``Pi`` to hand over: the daisy term needs the two spectra
        themselves and the subtraction is the only route, so declining is the correct answer
        rather than a missing feature.

        Returns an array shaped like the boson mass spectrum, or ``None``.
        """
        coefficient = self._debye_coefficient(X)
        if coefficient is None:
            return None
        T2 = np.asarray(T, dtype=float) ** 2
        if np.ndim(T2):
            return coefficient * T2[..., np.newaxis]
        return coefficient * float(T2)

    # Outcome of the check in `_debye_coefficient`, kept because `Vtot` calls it on every
    # evaluation and the answer cannot change for a given model: `None` not yet asked,
    # `"unavailable"` the thermal part does not go as `T^2`, `"field_dependent"` it does but
    # the coefficient moves with the field, or the coefficient array itself where it does not.
    _debye_state = None

    def _debye_coefficient(self, X: np.ndarray):
        """``c`` with ``Pi = c T^2``, measured and verified, or ``None``.

        Separate from :func:`debye_massSq` so the verification is paid once rather than on
        every call, and decided at two field points fixed by the model rather than at whatever
        field value happens to arrive first. ``Vtot`` is called with random field points while
        the model is being built, and a verdict read off those would differ from run to run.

        Three outcomes. A model whose thermal part is not quadratic never measures again. One
        whose coefficient is the same at both reference points is taken to be field
        independent and never measures again either. The rest measure on every call, two extra
        spectrum evaluations. None of the models shipped here takes that third path: the four
        that pass the check all have a field-independent coefficient, and the four whose
        coefficient does move with the field turn out not to be quadratic either, which is the
        same mixing in both cases. It is kept for models that are.
        """
        state = self._debye_state
        # The array test comes first: comparing a cached coefficient array against a string
        # is an elementwise comparison, and `if` on its result raises.
        if isinstance(state, np.ndarray):
            return state
        if state == "unavailable":
            return None

        scale = float(getattr(self, "v_stable", 0.0) or 0.0)
        if not np.isfinite(scale) or scale <= 0.0:
            self._debye_state = "unavailable"
            return None

        def measure(Xa, T):
            zero = np.asarray(self.boson_massSq(Xa, np.asarray(0.0))[0], dtype=float)
            hot = np.asarray(self.boson_massSq(Xa, np.asarray(T))[0], dtype=float)
            return (hot - zero) / T ** 2

        if state != "field_dependent":
            try:
                reference = np.atleast_1d(
                    np.asarray(self.X0, dtype=float)).ravel()[:self.Ndim]
                elsewhere = 0.37 * reference + 0.11 * scale
                here, there = measure(reference, scale), measure(elsewhere, scale)
                # The quadratic law is checked at both points, so a model that is quadratic
                # only where it happens to have been sampled does not pass.
                check_here = measure(reference, 0.5 * scale)
                check_there = measure(elsewhere, 0.5 * scale)
            except errors.Timeout:
                raise
            except Exception:
                self._debye_state = "unavailable"
                return None
            parts = (here, there, check_here, check_there)
            if not all(np.all(np.isfinite(v)) for v in parts):
                self._debye_state = "unavailable"
                return None
            # Relative to the largest coefficient, so a mode with no Debye mass, of which
            # every gauge theory here has some, does not set the scale or divide by zero.
            magnitude = float(np.max(np.abs(here))) if here.size else 0.0
            if magnitude <= 0.0:
                self._debye_state = "unavailable"
                return None
            tolerance = 1.0e-9 * magnitude
            if (np.max(np.abs(check_here - here)) > tolerance
                    or np.max(np.abs(check_there - there)) > tolerance):
                self._debye_state = "unavailable"
                return None
            if here.shape == there.shape and np.max(np.abs(there - here)) <= tolerance:
                self._debye_state = here
                return here
            self._debye_state = "field_dependent"

        # The two fixed probes established the quadratic law at two field points, which is
        # what decides whether the coefficient can be cached. It is not a licence to synthesise
        # `Pi = c(X) T^2` at a third field point without looking: a model that is quadratic
        # where it was probed can fail to be elsewhere, and the daisy term would then be handed
        # a thermal part the model does not have. The law is checked again here, at the field
        # value actually asked for.
        try:
            coefficient = measure(np.asarray(X), scale)
            verification = measure(np.asarray(X), 0.5 * scale)
        except errors.Timeout:
            raise
        except Exception:
            return None
        if not (np.all(np.isfinite(coefficient)) and np.all(np.isfinite(verification))):
            return None
        magnitude = float(np.max(np.abs(coefficient))) if coefficient.size else 0.0
        if magnitude <= 0.0:
            return None
        if np.max(np.abs(verification - coefficient)) > 1.0e-9 * magnitude:
            return None
        return coefficient

    def Vdaisy(self, bosons0, bosonsT , T: float | np.ndarray, Pi=None
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

        y = np.real(-(T/(12.*np.pi))
                    * np.sum(nb * _daisy_mass_cubed_difference(m20, m2T, Pi), axis=-1))
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
        # Same `Pi` as `Vtot` passes, or this helper and the potential it is a piece of
        # disagree wherever the model provides analytic Debye masses.
        return self.Vdaisy(bosons0, bosons, T, Pi=self.debye_massSq(X, T))


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
        given by `self.counterterm_derivative_step`, which is independent of
        the phase-tracing accuracy `self.x_eps`. The result is cached the first
        time this function is called, so subsequently changing the step will
        not have an effect.
        
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
                V_coleman_weinberg, eps=self.counterterm_derivative_step, Ndim=self.Ndim,
                order=self.deriv_order)

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
        given by `self.counterterm_derivative_step`, which is independent of
        the phase-tracing accuracy `self.x_eps`. The result is cached the first
        time this function is called, so subsequently changing the step will
        not have an effect.
        
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
                V_coleman_weinberg, eps=self.counterterm_derivative_step, Ndim=self.Ndim,
                order=self.deriv_order)
           
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
        # hessianFunction allocates its output from X.shape
        X = broadcast_fields(X, T)
        T = np.asanyarray(T)[..., np.newaxis]
        return f(X, T, False)

    def d2V1atvev(self):
        """
        Find the second derivative of the one loop zero temperature potential,
        evaluated at the vev. Used in the function Vct to calculate the
        counter term lagrangian

        This uses :func:`helper_functions.gradientFunction` to calculate the
        gradient using finite differences, with differences
        given by `self.counterterm_derivative_step`, which is independent of
        the phase-tracing accuracy `self.x_eps`. The result is cached the first
        time this function is called, so subsequently changing the step will
        not have an effect.
        
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
                V_coleman_weinberg, eps=self.counterterm_derivative_step, Ndim=self.Ndim,
                order=self.deriv_order)

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

    def radiationEnergyDensity(self, X: np.ndarray, T: float | np.ndarray,
                               include_decoupled=True) -> float | np.ndarray:
        r"""Return the energy density in radiation that is not
        field dependent.

        Parameters
        ----------
        X : np.ndarray
            The scalar field values
        T : float or np.ndarray
            Temperature. ``X.shape[:-1]`` and ``T.shape`` must be broadcastable.
        include_decoupled : bool, optional
            If true, include the enery density of the decoupled radiation bath

        Returns
        -------
        float or np.ndarray :
            The energy density in pure radiation at temperature T, with shape
            ``np.broadcast_shapes(X.shape[:-1], np.shape(T))``."""

        # Radiation energy density of particles that interact with the
        # scalars from the potential
        T2 = T * T
        if np.ndim(T) > 0:
            # one axis for the particle species, as in Vtot
            T2 = np.asanyarray(T2, dtype=float)[..., np.newaxis]
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
        # Standard Model fields of the potential are counted there, with their field- and
        # temperature-dependent masses, so remove them from the tabulated bath.
        geff = geff - sm_fields_in_potential_geff(self, T, "e")

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
                      include_decoupled: bool = True) -> float | np.ndarray:
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

        if np.ndim(T) > 0 or np.ndim(X) > 1:
            T = np.asanyarray(T, dtype=float)
            X = broadcast_fields(X, T)
            V0 = self.V0(X) + self.Vct(X) + self.V1_from_X(X)
            DV0 = V0 - (self.V0(self.X0) + self.Vct(self.X0) + self.V1_from_X(self.X0))
            # Numerical stability:
            with np.errstate(divide="ignore", invalid="ignore"):
                DV0 = np.where(DV0 / np.abs(V0) < 1e-10, 0.0, DV0)
            etot = DV0 + self.energyDensityDaisy(X, T)
            hot = T != 0.0
            radiation = self.radiationEnergyDensity(X, np.where(hot, T, 1.0), include_decoupled)
            return etot + np.where(hot, radiation, 0.0)

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
