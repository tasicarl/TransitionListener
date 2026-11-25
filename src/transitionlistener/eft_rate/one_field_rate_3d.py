"""The nucleation rate of a model with one scalar field from its three-dimensional effective theory.

Mix ``ThreeDimensionalRate`` into a model and that model takes its rate from the 3D theory: its defaults set
``rate_mode: "model"``, and, where the model names a shipped NLO grid, ``include_determinants`` and ``include_dim6``.
With ``rate_mode: "S3overT_4D"`` the model takes the rate of its four-dimensional parent instead, from the bounce of
its four-dimensional effective potential (daisy-resummed where the parent resums). That rate is not the leading order of
the expansion in the 3D theory; it is a different approximation. The class is general in the model; the bounce table
it reads (``abelian_lo_bounce_table``) holds the potential of one scalar with an abelian gauge field::

    class specific_potential(ThreeDimensionalRate, my_4d_model.specific_potential):
        nlo_grid_name = "my_model"                     # tab_data/nlo_grids/<name>.npz, for include_determinants

        def field_dependent_mass_over_field(self):
            return self.g                              # gauge boson mass per unit field

        def soft_parameters(self, T_GeV):
            return my_matching.matched_parameters(...)

What a model supplies:

``soft_parameters(T_GeV)``
    The matched 3D parameters as a dict with at least ``m3sq``, ``lam3``, ``g3sq`` and ``mD2``, in GeV units, and with
    ``nucleationConf.temporal_coupling = "h3"`` (the default) ``lambda_vl``, the A0^2 phi^2 coupling h3 of the temporal
    mass m_D^2 + h3 phi^2 / 2. It raises RuntimeError where the model cannot run its couplings from the input scale to
    the matching scale, for example at a Landau pole of one of them.
``field_dependent_mass_over_field()``
    The mass of the heaviest field-dependent mode per unit field. It says where the symmetric phase is.

For the NLO rate, in addition: ``nlo_grid_name``, ``nlo_grid_point()``, ``input_scale_GeV()``,
``vacuum_energy_GeV4()``, ``two_loop_potential(p)`` and ``two_loop_kinetic(p)``; see ``nlo_tables``.

What this class does at each temperature:

1. asks the model for the matched parameters (``soft_parameters``);
2. takes S_3/T of the bounce of the potential V3, eq. (A.1) of the paper, from the table of ``abelian_lo_bounce_table``;
3. if ``include_determinants`` is set, adds the NLO correction Delta S, eq. (A.4), interpolated from the shipped
   grid (``nlo_tables.correction_for_model``) or from the table in ``nlo_correction``; with ``include_dim6`` the
   grid includes S_6;
4. hands S_3/T + Delta S to the pipeline as the action, so that the rate ``T^4 (S/2 pi T)^{3/2} exp(-S/T)`` is the
   NLO rate ``A_dyn T^3 exp(-E)`` of eq. (A.2). The physics of the terms is described in ``abelian_higgs_3d``.

The constants of this class:

``ACTION_CAP``, ``ACTION_SOFT_FROM``
    A transition needs Gamma ~ H^4, so S_3/T = 4 ln(T/H) up to logarithms. Since H is at least of order T^2/M_Pl, this
    is at most about 4 ln(M_Pl/T): 80 at T = 1e9 GeV, 250 at 1e-9 GeV and 290 at 1e-12 GeV. It reaches
    ``ACTION_SOFT_FROM`` = 600 only at T ~ M_Pl e^-150 ~ 1e-46 GeV. Above ``ACTION_SOFT_FROM`` the action rolls over
    smoothly to ``ACTION_CAP``, and where there is no barrier, or no 3D theory, the action is ``ACTION_CAP``. A constant
    large action alone would leave a step where the barrier appears, which the percolation solver cannot integrate
    through; an infinite one gives nan in ln Gamma. The percolation temperature therefore does not depend on either
    number.
``ORIGIN_MASS_OVER_T``
    A field whose mass is this far below T is treated as sitting at the origin.
``ORIGIN_STEP_OVER_T``
    There, the derivatives of the potential take their step on the scale of T. The fixed field steps of the tracer are
    set by the scale of the broken phase; in strong supercooling they exceed T and sample V beyond the barrier, and the
    field of the symmetric phase would otherwise grow as T^-2 in the tracer's predictor.

Documentation: https://tasillo.de/TransitionListener/

Authors:
    Carlo Tasillo <carlo.tasillo@ific.uv.es>
"""
import numpy as np

from transitionlistener.eft_rate import abelian_lo_bounce_table

from transitionlistener.eft_rate import abelian_higgs_3d

ACTION_CAP = 1.2e3
ACTION_SOFT_FROM = 6.0e2
ORIGIN_MASS_OVER_T = 0.1
ORIGIN_STEP_OVER_T = 1e-2


#: The diagnostics of the 3D theory at the milestone temperature, with their values where the run has no 3D rate.
VALIDITY_KEYS = {"WARNING:eft3d_outside_validity": False, "DIAG:eft3d_expansion_3d": float("nan"),
                 "DIAG:eft3d_hard": float("nan"), "DIAG:eft3d_dim6": float("nan")}


def validity_diagnostics(pot, T: float) -> dict:
    """``VALIDITY_KEYS`` at ``T`` for a model with the 3D rate: the warning is set where the monitor's verdict is not
    "ok" or a check could not be evaluated (``validity`` never counts a missing check as passing)."""
    out = dict(VALIDITY_KEYS)
    report = pot.eft_validity_report(T) if hasattr(pot, "eft_validity_report") else None
    if report is None:
        return out
    out["WARNING:eft3d_outside_validity"] = not report.trusted
    for key in ("expansion_3d", "hard", "dim6"):
        if key in report.ratios:
            out[f"DIAG:eft3d_{key}"] = float(report.ratios[key])
    return out


def saturate_action(s3_over_T: float) -> float:
    """S_3/T below ``ACTION_SOFT_FROM``; above it a tanh that approaches ``ACTION_CAP`` with continuous derivatives."""
    if s3_over_T <= ACTION_SOFT_FROM:
        return float(s3_over_T)
    span = ACTION_CAP - ACTION_SOFT_FROM
    return float(ACTION_SOFT_FROM + span * np.tanh((s3_over_T - ACTION_SOFT_FROM) / span))


class ThreeDimensionalRate:
    """See the module docstring."""

    #: Tabulated next-to-leading-order correction to S_3/T for one parameter point, or None. With None and
    #: ``include_determinants`` set, the model takes the grid shipped with the package (``nlo_grid_name``),
    #: interpolated in the couplings, which raises outside its range.
    nlo_correction = None

    #: Name of the NLO grid shipped for the model, tab_data/nlo_grids/<name>.npz, or None.
    nlo_grid_name = None

    #: The derivatives this class replaces at the origin only.
    _origin_entries = ("gradV", "d2V", "dgradV_dT")

    def setConfigParameters(self) -> None:
        """The settings of the 4D model, then the 3D rate, with its NLO correction where a grid is shipped."""
        super().setConfigParameters()
        conf = self.config.nucleationConf
        conf.rate_mode = "model"
        conf.include_determinants = conf.include_dim6 = self.nlo_grid_name is not None

    def field_dependent_mass_over_field(self) -> float:
        """The mass of the heaviest field-dependent mode per unit field; supplied by the model."""
        raise NotImplementedError("a 3D model supplies the mass of its heaviest field-dependent mode per unit field")

    def soft_parameters(self, T_GeV: float) -> dict:
        """The matched 3D parameters at ``T_GeV`` (see the module docstring); supplied by the model."""
        raise NotImplementedError("a 3D model supplies its matched parameters")

    def matched_3d_parameters(self, T_GeV: float):
        """``(m3sq, lam3, g3sq, mD2)`` in GeV units."""
        p = self.soft_parameters(T_GeV)
        return p["m3sq"], p["lam3"], p["g3sq"], p["mD2"]

    def _at_origin(self, X, T):
        """Where (field, temperature) is the symmetric phase in a 3D run; all False otherwise."""
        phi = np.asanyarray(X)[..., 0]
        T = np.asanyarray(T)
        if self.config.nucleationConf.rate_mode != "model":
            return np.zeros(np.broadcast(phi, T).shape, dtype=bool)
        return (T > 0.0) & (self.field_dependent_mass_over_field() * np.abs(phi) < ORIGIN_MASS_OVER_T * T)

    def gradV(self, X, T):
        """As the base class, with the difference step on the scale of T at the origin."""
        result = np.array(super().gradV(X, T), dtype=float)
        origin = self._at_origin(X, T)
        if origin.any():
            phi, step = np.asanyarray(X)[..., 0, np.newaxis], ORIGIN_STEP_OVER_T * np.asanyarray(T)
            step = step[..., np.newaxis] if step.ndim else step
            difference = (self.Vtot(phi + step, T, False) - self.Vtot(phi - step, T, False))
            result[origin, 0] = (difference / (2.0 * np.squeeze(step)))[origin]
        return result

    def d2V(self, X, T):
        """As the base class, with the difference step on the scale of T at the origin."""
        result = np.array(super().d2V(X, T), dtype=float)
        origin = self._at_origin(X, T)
        if origin.any():
            phi, step = np.asanyarray(X)[..., 0, np.newaxis], ORIGIN_STEP_OVER_T * np.asanyarray(T)
            step = step[..., np.newaxis] if step.ndim else step
            difference = (self.Vtot(phi + step, T, False) + self.Vtot(phi - step, T, False)
                          - 2.0 * self.Vtot(phi, T, False))
            result[origin, 0, 0] = (difference / np.squeeze(step) ** 2)[origin]
        return result

    def dgradV_dT(self, X, T):
        """As the base class, and zero at the origin: the phase does not move with T there."""
        result = np.array(super().dgradV_dT(X, T), dtype=float)
        result[self._at_origin(X, T), 0] = 0.0
        return result

    def _origin_stencil(self):
        """What the compiled minimiser needs of the derivatives at the origin; None outside a 3D run."""
        if self.config.nucleationConf.rate_mode != "model":
            return None
        return float(self.field_dependent_mass_over_field()), ORIGIN_MASS_OVER_T, ORIGIN_STEP_OVER_T

    def bounceFreeAction(self):
        """With the 3D rate the action comes from the 3D theory, so the 4D bounce is not solved."""
        conf = self.config.nucleationConf
        if conf.rate_mode != "model" or getattr(conf, "keep_4d_action", False):
            return None
        return lambda T: self.actionNuc(T, float("nan"))

    def lo_action_over_T(self, T_GeV: float):
        """The leading-order S_3/T, or None where the 3D theory has no bounce or cannot be matched.

        From the universal table of ``abelian_lo_bounce_table``. The matching fails where the model cannot run its
        couplings from the input scale down to the matching scale, at a Landau pole of one of them or outside the
        perturbative regime. The percolation grid can reach such temperatures far below any transition.
        """
        try:
            p = self.soft_parameters(T_GeV)
        except RuntimeError:
            return None
        return abelian_lo_bounce_table.action_over_T(p["m3sq"], p["lam3"], p["g3sq"], p["mD2"], h3=self.temporal_coupling(p))

    def temporal_coupling(self, p):
        """h3 of the temporal mass in GeV for the matched parameters ``p``, or None for its leading-order value 2 g3^2."""
        from transitionlistener.eft_rate import nlo_tables
        return nlo_tables.temporal_coupling(self, p)

    def bounce_profile(self, T: float):
        """``(R, phi)`` of the leading-order 3D bounce at ``T``, with R in internal units, or None."""
        if self.config.nucleationConf.rate_mode != "model":
            return None
        T_GeV = float(T) * self.conversionFactor
        try:
            p = self.soft_parameters(T_GeV)
            _, r, phi3, _ = abelian_higgs_3d.solve_LO_bounce(p["m3sq"], p["lam3"], p["g3sq"], m_debye_sq=p["mD2"],
                                                  h3=self.temporal_coupling(p))
        except (RuntimeError, ValueError):
            return None
        return np.asarray(r) * self.conversionFactor, np.asarray(phi3)

    def eft_validity_report(self, T: float):
        """The validity monitor (:mod:`validity`) of the 3D theory at ``T`` (internal units), with the thresholds of
        the order the run uses; None outside a 3D run or where the theory has no bounce."""
        if self.config.nucleationConf.rate_mode != "model":
            return None
        from transitionlistener.eft_rate import validity
        T_GeV = float(T) * self.conversionFactor
        try:
            p = self.soft_parameters(T_GeV)
            h3 = self.temporal_coupling(p)
            S3_over_T, r, phi3, _ = abelian_higgs_3d.solve_LO_bounce(p["m3sq"], p["lam3"], p["g3sq"], m_debye_sq=p["mD2"],
                                                          wall_power=12, h3=h3)
        except (RuntimeError, ValueError):
            return None
        ratios = abelian_higgs_3d.validity_ratios(T_GeV, p["m3sq"], p["lam3"], p["g3sq"], p["mD2"], r, phi3,
                                       fermion_masses_over_field=p.get("fermion_masses_over_field", ()),
                                       c6=p.get("c6"), S3_over_T=S3_over_T, h3=h3)
        order = "NLO" if self.config.nucleationConf.include_determinants else "LO"
        return validity.evaluate(ratios, context={"T_GeV": T_GeV}, order=order)

    def actionNuc(self, T: float, action: float, instanton=None) -> float:
        """``T`` times S_3/T + Delta S (eq. A.4; Delta S only with ``include_determinants``), saturated by
        :func:`saturate_action`, in place of the 4D action; the pipeline divides by ``T``."""
        if self.config.nucleationConf.rate_mode != "model":
            return action
        T_GeV = float(T) * self.conversionFactor
        if T_GeV <= 0.0:
            return action
        S3_over_T = self.lo_action_over_T(T_GeV)
        if S3_over_T is None:
            return float(T) * ACTION_CAP
        if self.config.nucleationConf.include_determinants:
            if self.nlo_correction is None:
                from transitionlistener.eft_rate import nlo_tables
                self.nlo_correction = nlo_tables.correction_for_model(self)
            S3_over_T = S3_over_T + float(self.nlo_correction(T_GeV, S3_LO=S3_over_T))
        return float(T) * saturate_action(S3_over_T)

    def logGammaNuc(self, T: float, S: float, instanton=None) -> float:
        """ln Gamma of the action :meth:`actionNuc` supplied, the rate the percolation integrals use."""
        if self.config.nucleationConf.rate_mode != "model":
            return super().logGammaNuc(T, S, instanton=instanton)
        from transitionlistener import bubbledynamics as _bd
        return float(np.atleast_1d(_bd.logGamma(T, S))[0])
