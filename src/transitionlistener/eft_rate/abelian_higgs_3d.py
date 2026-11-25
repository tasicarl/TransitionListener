r"""Bubble nucleation rate of an Abelian Higgs-like model in its three-dimensional EFT, at NLO.

The physics specification is appendix A of the paper ("Bubble nucleation rate in the three-dimensional EFT"),
eqs. (A.1)-(A.4). At one temperature the computation runs as follows; the functions named are those of this module.

1. **The potential**, eq. (A.1): :func:`V3_LO`,

   .. math:: V_3 = \tfrac12 m_3^2\phi_3^2 + \tfrac14\lambda_3\phi_3^4
                   - \frac{1}{12\pi}\big(2 m_A^3 + m_{A_0}^3 - m_D^3\big),\qquad
             m_A^2 = g_3^2\phi_3^2,\quad m_{A_0}^2 = m_D^2 + \tfrac12 h_3\phi_3^2 ,

   with the spatial (``n_transverse = 2``) and temporal (``n_temporal = 1``) gauge fields integrated out at one loop.
   ``h3`` is the paper's :math:`\lambda_{\rm VL}`; ``None`` stands for its leading-order value :math:`2 g_3^2`.
2. **The bounce** of :math:`V_3`: :func:`solve_LO_bounce` returns :math:`S_3/T` and the profile
   :math:`\phi_3(r)`. Every NLO term below is evaluated on this profile, in Feynman-'t Hooft gauge
   (:math:`\xi = 1`).
3. **The exponent** :math:`E` of :math:`\Gamma = A_{\rm dyn} T^3 e^{-E}`, eq. (A.2):

   ==================================  ==========================================================================
   term of eq. (A.2)                   computed by
   ==================================  ==========================================================================
   :math:`S_3/T`                       :func:`solve_LO_bounce`
   :math:`\delta_h`                    radial determinant, zero modes removed (:func:`nlo_assembly`)
   :math:`\delta_\chi`                 Goldstone determinant at :math:`\xi = 1` (:func:`nlo_assembly`)
   :math:`\delta_1`                    one gauge channel :math:`-\nabla^2 + g_3^2\phi_3^2` (:func:`nlo_assembly`)
   :math:`\delta_0`                    temporal gauge field (:func:`nlo_assembly`)
   :math:`\tfrac12\Delta_{\rm mix}`    Goldstone-gauge mixing, :func:`mixing_log_det` (passed in)
   :math:`S^{J=0}`                     zero mode of the global :math:`U(1)` rotation, :func:`j0_sector`
   :math:`C`                           :func:`double_counting_subtraction`
   :math:`\int V_3^{\rm NLO}`          :func:`soft_nlo_action` (``potential``)
   :math:`S_6`                         :func:`dim6_action_shift` + :func:`dim6_kinetic_action_shift`
   ==================================  ==========================================================================

   :func:`nlo_assembly` returns :math:`E` without :math:`S_6` as ``total`` and the dynamical prefactor
   :math:`\ln(A_{\rm dyn}/T)`, :math:`A_{\rm dyn} = \sqrt{|\lambda_-|}/2\pi`. The single channels are
   Gelfand-Yaglom determinants from BubbleDet; the mixing and the :math:`J = 0` block, which BubbleDet does not
   treat, come from the coupled radial solver of :mod:`transitionlistener.eft_rate.abelian_determinants`.
4. **The correction to the action**, eq. (A.4): :func:`action_seam_correction` solves
   :math:`\Delta S = E - S_3/T - \ln(A_{\rm dyn}/T) + \tfrac32\ln(S_{\rm tot}/2\pi)`,
   :math:`S_{\rm tot} = S_3/T + \Delta S`, by fixed-point iteration, so that TransitionListener's rate formula
   :math:`T^4 (S_{\rm tot}/2\pi)^{3/2} e^{-S_{\rm tot}}` reproduces the NLO rate.
5. **Tabulation.** The determinants take minutes per temperature, so :math:`\Delta S` is tabulated offline by
   :mod:`transitionlistener.eft_rate.nlo_tables` on a grid in the couplings and :math:`S_3/T`, and interpolated at run time
   (``nlo_tables.GridCorrection``; :class:`NLOCorrection` for a table in T at one parameter point).

Units. 3D fields have mass dimension 1/2; :math:`g_3^2` and :math:`\lambda_3` are masses, :math:`m_3^2` and
:math:`m_D^2` masses squared, :math:`h_3` a mass. The 3D action is dimensionless and *is* :math:`S_3/T`: it must not
be divided by T again. The same potential in the 4D normalisation,
:math:`V_4(\phi, T) = T\,V_3(\phi/\sqrt T)`, has an action of dimension one. Every function accepts any
energy unit; the radial channel, the only dimensionful one, is converted to units of T by :func:`nlo_assembly`.

The rest of the module serves the validity ratios of the reduction (:func:`validity_ratios`, read by
:mod:`transitionlistener.eft_rate.validity`) and the check that the bounce does not depend on its wall
(:func:`bounce_is_wall_limited`).

Documentation: https://tasillo.de/TransitionListener/

Authors:
    Carlo Tasillo <carlo.tasillo@ific.uv.es>
"""
from __future__ import annotations

import numpy as np

#: transverse polarisations of a gauge boson in three spatial dimensions
N_TRANSVERSE_ABELIAN = 2
#: temporal gauge modes of an Abelian gauge field
N_TEMPORAL_ABELIAN = 1

#: Relative tolerance of BubbleDet's Gelfand-Yaglom equations (BubbleDet's default is 1e-6).
BUBBLEDET_GY_TOL = 1e-9
#: Points of the uniform radial grid on which the bounce is handed to BubbleDet. A uniform grid makes the
#: determinants a smooth function of the bounce; the shooting's own grid is not uniform.
BUBBLEDET_POINTS = 4000
#: Fraction of the bubble over which BubbleDet fits the asymptotic amplitude of the bounce, which the radial
#: channel's zero modes need (BubbleDet's default is 7e-3).
BUBBLEDET_TAIL = 5e-4

#: Where a shot profile hands over to its analytic tail ``A exp(-m r)/r``, and where that tail ends, as fractions
#: of phi(0). The match sits where the shot is still accurate and already in its asymptotic form.
TAIL_MATCH_FRACTION = 1e-5
TAIL_END_FRACTION = 1e-8

#: Field values below this are replaced by it when evaluating Z3^NLO, which diverges as 1/phi3. A guard against a
#: profile that reaches exactly zero; the integrand vanishes as phi3 in the tail, so the result does not depend on it.
Z3_FIELD_FLOOR = 1e-12

#: Last partial wave summed for the Goldstone-gauge mixing; the remainder is fitted (:func:`mixing_log_det`).
MIXING_L_MAX = 70
#: The smallest l_max at which the remainder of the mixing sum is taken from the large-l expansion of the summand.
SERIES_TAIL_MIN_L_MAX = 20
#: The powers of 1/(l + 1/2) in that expansion: the mixing enters the WKB expansion of each wave at second order.
SERIES_TAIL_POWERS = (2.0, 3.0, 4.0, 5.0)


# ----------------------------------------------------------------------
# 1. the potential V3, eq. (A.1)
# ----------------------------------------------------------------------
def temporal_coupling(g3sq: float, h3=None) -> float:
    """The coupling h3 (the paper's lambda_VL) of the temporal mass m_D^2 + h3 phi3^2 / 2; None means 2 g3^2."""
    return 2.0 * g3sq if h3 is None else float(h3)


def _gauge_cubic(phi3, g3sq, m_debye_sq, n_transverse, n_temporal, h3=None):
    r"""The gauge-loop term of eq. (A.1),
    :math:`-[n_T m_A^3 + n_0 (m_{A_0}^3 - m_D^3)]/12\pi`, zero at zero field."""
    phi3 = np.asarray(phi3, dtype=float)
    m_gauge_sq = g3sq * phi3**2
    out = -n_transverse * m_gauge_sq**1.5 / (12.0 * np.pi)
    if n_temporal:
        m0_sq = m_debye_sq + 0.5 * temporal_coupling(g3sq, h3) * phi3**2
        out = out - n_temporal * (m0_sq**1.5 - m_debye_sq**1.5) / (12.0 * np.pi)
    return out


def _dgauge_cubic(phi3, g3sq, m_debye_sq, n_transverse, n_temporal, h3=None):
    """Derivative of :func:`_gauge_cubic` with respect to phi3."""
    phi3 = np.asarray(phi3, dtype=float)
    out = -n_transverse * g3sq**1.5 * phi3 * np.abs(phi3) / (4.0 * np.pi)
    if n_temporal:
        h = temporal_coupling(g3sq, h3)
        out = out - (n_temporal * h * phi3
                     * np.sqrt(m_debye_sq + 0.5 * h * phi3**2) / (8.0 * np.pi))
    return out


def V3_LO(phi3, m3sq: float, lam3: float, g3sq: float,
          n_transverse: int = N_TRANSVERSE_ABELIAN, *, m_debye_sq: float,
          n_temporal: int = N_TEMPORAL_ABELIAN, h3: float | None = None):
    r"""The soft potential :math:`V_3(\phi_3)` of eq. (A.1), mass dimension 3.

    ``phi3`` is the 3D field (dimension 1/2); ``m3sq``, ``lam3``, ``g3sq``, ``m_debye_sq`` and ``h3`` the matched
    parameters (see the module docstring); ``n_transverse`` and ``n_temporal`` count the gauge modes in the loop.
    """
    phi3 = np.asarray(phi3, dtype=float)
    return (0.5 * m3sq * phi3**2
            + 0.25 * lam3 * phi3**4
            + _gauge_cubic(phi3, g3sq, m_debye_sq, n_transverse, n_temporal, h3))


def dV3_LO(phi3, m3sq: float, lam3: float, g3sq: float,
           n_transverse: int = N_TRANSVERSE_ABELIAN, *, m_debye_sq: float,
           n_temporal: int = N_TEMPORAL_ABELIAN, h3: float | None = None):
    """dV3/dphi3 of :func:`V3_LO`."""
    phi3 = np.asarray(phi3, dtype=float)
    return (m3sq * phi3 + lam3 * phi3**3
            + _dgauge_cubic(phi3, g3sq, m_debye_sq, n_transverse, n_temporal, h3))


def d2V3_LO(phi3, m3sq: float, lam3: float, g3sq: float,
            n_transverse: int = N_TRANSVERSE_ABELIAN, *, m_debye_sq: float,
            n_temporal: int = N_TEMPORAL_ABELIAN, h3: float | None = None):
    """d^2V3/dphi3^2 of :func:`V3_LO`, the mass squared of the radial fluctuation."""
    phi3 = np.asarray(phi3, dtype=float)
    out = (m3sq + 3.0 * lam3 * phi3**2
           - n_transverse * g3sq**1.5 * np.abs(phi3) / (2.0 * np.pi))
    if n_temporal:
        h = temporal_coupling(g3sq, h3)
        root = np.sqrt(m_debye_sq + 0.5 * h * phi3**2)
        out = out - (n_temporal * h * (m_debye_sq + h * phi3**2)
                     / (8.0 * np.pi * root))
    return out


def _lo_callables(m3sq, lam3, g3sq, n_transverse=N_TRANSVERSE_ABELIAN, *, m_debye_sq,
                  n_temporal=N_TEMPORAL_ABELIAN, h3=None):
    """``(V, dV, d2V)`` of :func:`V3_LO` at fixed parameters, as functions of phi3 alone."""
    kw = dict(m_debye_sq=m_debye_sq, n_temporal=n_temporal, h3=h3)
    return (lambda p: V3_LO(p, m3sq, lam3, g3sq, n_transverse, **kw),
            lambda p: dV3_LO(p, m3sq, lam3, g3sq, n_transverse, **kw),
            lambda p: d2V3_LO(p, m3sq, lam3, g3sq, n_transverse, **kw))


def is_bounded_below(lam3: float) -> bool:
    """Whether :func:`V3_LO` is bounded below at large field, i.e. ``lambda3 > 0``.

    In a classically conformal model the quartic runs negative at low temperatures, so where the barrier exists the
    potential is often unbounded; the bounce solver then needs the wall of :func:`solve_bounce_1d`.
    """
    return lam3 > 0.0


def escape_point_LO(m3sq: float, lam3: float, g3sq: float,
                    n_transverse: int = N_TRANSVERSE_ABELIAN, *, m_debye_sq: float, n_temporal: int = N_TEMPORAL_ABELIAN, h3: float | None = None) -> float:
    """The field beyond the barrier where :func:`V3_LO` returns to ``V3(0) = 0``, in 3D units; nan without a barrier.

    The positive root of the transverse-only quadratic ``lambda3 phi3^2/4 - n_T g3^3 phi3/(12 pi) + m3sq/2`` sets
    the scale of the search in :func:`escape_point`.
    """
    a = 0.25 * lam3
    b = -n_transverse * g3sq**1.5 / (12.0 * np.pi)
    c = 0.5 * m3sq
    disc = b * b - 4.0 * a * c
    if a == 0.0 or disc < 0.0:
        scale = np.sqrt(abs(m3sq) / g3sq) if g3sq > 0.0 else 1.0
    else:
        roots = [(-b - np.sqrt(disc)) / (2.0 * a), (-b + np.sqrt(disc)) / (2.0 * a)]
        positive = [x for x in roots if x > 0.0]
        scale = min(positive) if positive else np.sqrt(abs(m3sq) / g3sq)

    f = lambda p: float(V3_LO(p, m3sq, lam3, g3sq, n_transverse,
                              m_debye_sq=m_debye_sq, n_temporal=n_temporal, h3=h3))
    return escape_point(f, scale, bounded=is_bounded_below(lam3))


def escape_point(V, scale: float, *, bounded: bool) -> float:
    """The field beyond the barrier where a potential with ``V(0) = 0`` at its false vacuum returns to zero.

    ``scale`` is a rough size of the barrier that starts the search; ``bounded`` says whether ``V`` turns back up at
    large field. For a bounded potential the broken minimum is located first and the root bracketed against it,
    since the region where V < 0 is then a window that a coarse outward scan can step over. Returns nan if there is
    no barrier or the broken phase is not deeper than the origin.
    """
    from scipy.optimize import brentq, minimize_scalar

    f = lambda p: float(V(p))
    lo = 1e-8 * scale
    if f(lo) <= 0.0:
        # V(lo) can be roundoff when the mass term is many orders below the constant Debye term it is summed with;
        # a tachyonic false vacuum is negative at the larger field as well
        lo = 1e-5 * scale
        if f(lo) <= 0.0:
            return float("nan")

    if bounded:
        hi = max(4.0 * scale, 1.0)
        for _ in range(200):                    # bracket the minimum first
            r = minimize_scalar(f, bounds=(lo, hi), method="bounded")
            if float(r.x) < 0.9 * hi:
                break
            hi *= 2.0
        phi_min = float(r.x)
        if f(phi_min) >= 0.0:
            # the bracket stopped at the shallow dip next to the origin, the broken minimum lying far beyond the
            # starting scale: search a dense logarithmic grid before declaring no barrier
            grid = np.geomspace(lo, 1e4 * max(scale, 1e-300), 4000)
            vals = np.array([f(p) for p in grid])
            k = int(np.argmin(vals))
            if vals[k] >= 0.0:
                return float("nan")             # broken phase is not deeper
            phi_min = float(grid[k])
        return float(brentq(f, lo, phi_min, xtol=1e-15, rtol=1e-14))

    # unbounded below: the potential runs away, so a geometric outward scan finds the crossing
    hi = scale
    for _ in range(200):
        if f(hi) < 0.0:
            return float(brentq(f, lo, hi, xtol=1e-15, rtol=1e-14))
        hi *= 1.3
    return float("nan")


# ----------------------------------------------------------------------
# 2. the bounce of V3: S3/T and phi3(r)
# ----------------------------------------------------------------------
def solve_LO_bounce(m3sq: float, lam3: float, g3sq: float,
                    n_transverse: int = N_TRANSVERSE_ABELIAN,
                    wall_multiple: float = 6.0, wall_power: int = 24,
                    xtol: float = 1e-11, phitol: float = 1e-11,
                    *, m_debye_sq: float, n_temporal: int = N_TEMPORAL_ABELIAN, h3: float | None = None):
    """The bounce of :func:`V3_LO`: ``(S3_over_T, r, phi3, info)``.

    ``S3_over_T`` is the dimensionless 3D action, which already is S_3/T (the term S_3/T of eq. A.2); ``r`` the
    radii and ``phi3`` the field on them, in the units of the inputs; ``info`` holds ``escape_point``, ``phi_min``
    (the field of the minimum shot towards), ``phi_centre``, ``dphi3`` (d phi3/dr on ``r``), ``bounded`` and
    ``wall`` (its position, or None). An unbounded potential gets the wall of
    :func:`solve_bounce_1d` at ``wall_multiple`` times the escape point (:func:`bounce_is_wall_limited` checks that
    it does not matter). Raises ValueError where there is no barrier.
    """
    escape = escape_point_LO(m3sq, lam3, g3sq, n_transverse, m_debye_sq=m_debye_sq, n_temporal=n_temporal, h3=h3)
    if not np.isfinite(escape):
        raise ValueError(
            "no barrier at these 3D parameters, so there is no bounce to solve; "
            f"m3sq={m3sq:.6g}, lam3={lam3:.6g}, g3sq={g3sq:.6g}"
        )
    V, dV, _ = _lo_callables(m3sq, lam3, g3sq, n_transverse, m_debye_sq=m_debye_sq, n_temporal=n_temporal, h3=h3)
    return solve_bounce_1d(
        V, dV, escape, bounded=is_bounded_below(lam3), wall_multiple=wall_multiple,
        wall_power=wall_power, xtol=xtol, phitol=phitol,
        context=f"m3sq={m3sq:.6g}, lam3={lam3:.6g}, g3sq={g3sq:.6g}",
        _lo_params=(m3sq, lam3, g3sq, m_debye_sq, n_transverse, n_temporal, temporal_coupling(g3sq, h3)))


def _false_vacuum_mass_sq(phi3, dV):
    """V'/phi at a field 1e-6 of the bounce's centre value: the mass squared of the false vacuum along the bounce."""
    probe = 1e-6 * float(phi3[0])
    return float(dV(probe)) / probe


def _with_decaying_tail(r, phi3, dphi3, dV):
    """Continue a bounce by its exact tail ``A exp(-m r)/r`` from where it falls below ``TAIL_MATCH_FRACTION`` of
    phi(0) down to ``TAIL_END_FRACTION``.

    The shooting ends a profile at a radius that varies from one solve to the next; the massless gauge channel and the
    J = 0 block depend on the radial domain, so the domain is fixed here by the tail. Returns the input unchanged
    where the profile never reaches the match or the false vacuum has no positive mass along the bounce.
    """
    phi0 = float(phi3[0])
    m_sq = _false_vacuum_mass_sq(phi3, dV)
    below = np.flatnonzero(phi3 < TAIL_MATCH_FRACTION * phi0)
    if not m_sq > 0.0 or below.size == 0 or below[0] < 2:
        return r, phi3, dphi3
    m = float(np.sqrt(m_sq))
    k = int(below[0]) - 1
    step = float(r[k] - r[k - 1])
    log_A = float(np.log(phi3[k] * r[k]) + m * r[k])
    # the tail ends where A exp(-m r)/r = TAIL_END_FRACTION phi0: m r + ln r = ln(A / phi_end), by fixed point
    log_end = log_A - float(np.log(TAIL_END_FRACTION * phi0))
    r_end = float(r[k])
    for _ in range(50):
        r_end = (log_end - np.log(r_end)) / m
    r_tail = np.linspace(r[k], r_end, int(np.ceil((r_end - r[k]) / step)) + 1)[1:]
    phi_tail = np.exp(log_A - m * r_tail) / r_tail
    return (np.append(r[:k + 1], r_tail), np.append(phi3[:k + 1], phi_tail),
            np.append(dphi3[:k + 1], -phi_tail * (m + 1.0 / r_tail)))


def solve_bounce_1d(V, dV, escape: float, *, bounded: bool,
                    wall_multiple: float = 6.0, wall_power: int = 24,
                    xtol: float = 1e-11, phitol: float = 1e-11, context: str = "",
                    _lo_params=None):
    """The bounce of any single-field 3D potential with its false vacuum at the origin: ``(S3_over_T, r, phi3, info)``.

    ``V`` and ``dV`` are functions of the 3D field; ``escape`` is where ``V`` returns to zero beyond the barrier
    (:func:`escape_point`). An unbounded potential (``bounded`` False) gets the wall
    ``|V(wall)| (phi/wall)**wall_power`` at ``wall = wall_multiple * escape``. The
    shooting solver of :mod:`transitionlistener.tunneling1D` needs the true minimum at coordinate zero, so the field is
    reflected before it is handed over and the profile reflected back; the profile is then continued by its
    analytic tail (:func:`_with_decaying_tail`). ``_lo_params`` passes the parameters of :func:`V3_LO` to the
    solver's closed-form derivative. The solver's failures at the edge of the barrier region, where the action
    diverges, are raised as ValueError, like "no barrier".
    """
    from scipy.optimize import minimize_scalar
    from transitionlistener import errors as _errors
    from transitionlistener.tunneling1D import ClosedFormInstanton, SingleFieldInstanton, _ACTION_BRACKET_TOL

    if bounded:
        V_used, dV_used = V, dV
        search_hi = 400.0 * escape
        wall = None
        scale = 0.0
    else:
        wall = wall_multiple * escape
        scale = abs(float(V(wall)))
        if scale == 0.0:
            scale = 1.0
        V_used = lambda p: V(p) + scale * (np.abs(p) / wall) ** wall_power
        dV_used = lambda p: dV(p) + (scale * wall_power / wall
                                     * (np.abs(p) / wall) ** (wall_power - 1) * np.sign(p))
        search_hi = 1.6 * wall

    res = minimize_scalar(lambda p: float(V_used(p)),
                          bounds=(0.3 * escape, search_hi), method="bounded")
    phi_min = float(res.x)
    if not (phi_min > escape and float(V_used(phi_min)) < 0.0):
        raise ValueError(
            "no minimum below the false vacuum was found beyond the escape point"
        )

    try:
        args = (0.0, phi_min,
                lambda x: V_used(phi_min - np.asarray(x)),
                lambda x: -dV_used(phi_min - np.asarray(x)))
        if _lo_params is None:
            inst = SingleFieldInstanton(*args)
        else:
            m3sq, lam3, g3sq, m_debye_sq, n_transverse, n_temporal, h3 = _lo_params
            sqrt_g3sq = np.sqrt(g3sq)
            inst = ClosedFormInstanton(*args, closed=[
                phi_min, m3sq, lam3, n_transverse * g3sq * sqrt_g3sq / (4.0 * np.pi), n_temporal * h3 / (8.0 * np.pi),
                m_debye_sq, 0.5 * h3, 0.0 if bounded else scale * wall_power / wall, wall if wall is not None else 1.0,
                wall_power - 1.0])
        profile = inst.findProfile(xtol=xtol, phitol=phitol)
    except _errors.PotentialError as err:
        kind = getattr(err, "error_type", None)     # the tag is an attribute, not in err.args
        if kind not in ("no barrier", "stable, not metastable"):
            raise
        raise ValueError(
            "no barrier at these 3D parameters according to the shooting "
            f"solver ({kind}), although the analytic check found one -- this "
            "is the marginal edge where the barrier disappears; " + context
        ) from err
    except _errors.TooSmalldVError as err:
        raise ValueError(
            "the 3D potential is too flat for the shooting solver -- the "
            "extreme thin-wall limit at the edge of the barrier region, where "
            "the action diverges; " + context
        ) from err
    # a bisection that never found the other side of the bounce leaves an infinite spread of the bracketing actions
    if not (getattr(inst, "bracket_action_spread", 0.0) <= _ACTION_BRACKET_TOL):
        raise ValueError(
            "the 3D potential is too flat for the shooting solver to bracket the bounce -- the "
            "extreme thin-wall limit at the edge of the barrier region, where the action "
            "diverges; " + context
        )
    S3_over_T = float(inst.findAction(profile))

    r = np.asarray(profile.R, dtype=float)
    phi3 = phi_min - np.asarray(profile.Phi, dtype=float)   # reflect back
    dphi3 = -np.asarray(profile.dPhi, dtype=float)          # d phi3/dr, reflected back like phi3
    r, phi3, dphi3 = _with_decaying_tail(r, phi3, dphi3, dV)
    info = {
        "escape_point": escape,
        "phi_min": phi_min,
        "phi_centre": float(phi3[0]),
        "dphi3": dphi3,
        "bounded": bounded,
        "wall": wall,
    }
    return S3_over_T, r, phi3, info


def bounce_is_wall_limited(m3sq: float, lam3: float, g3sq: float,
                           n_transverse: int = N_TRANSVERSE_ABELIAN,
                           multiples=(4.0, 8.0), rtol: float = 0.05,
                           *, m_debye_sq: float,
                           n_temporal: int = N_TEMPORAL_ABELIAN, h3: float | None = None) -> bool:
    """Whether the action of :func:`solve_LO_bounce` changes by more than ``rtol`` between the wall distances
    ``multiples``, i.e. is set by the regulator rather than by the EFT. False for a bounded potential (no wall)."""
    if is_bounded_below(lam3):
        return False
    actions = []
    for m in multiples:
        try:
            actions.append(solve_LO_bounce(m3sq, lam3, g3sq, n_transverse,
                                           m_debye_sq=m_debye_sq, n_temporal=n_temporal, h3=h3,
                                           wall_multiple=m)[0])
        except Exception:
            return True
    lo, hi = min(actions), max(actions)
    return not (hi - lo) <= rtol * abs(hi)


# ----------------------------------------------------------------------
# 3. the terms of E, eq. (A.2)
# ----------------------------------------------------------------------
def _bubbledet():
    """``(BubbleConfig, ParticleConfig, BubbleDeterminant)`` of BubbleDet, an optional dependency needed only to
    compute determinants (building an NLO table), not to run on a shipped grid."""
    try:
        from BubbleDet import BubbleConfig, ParticleConfig, BubbleDeterminant
    except ImportError as exc:
        raise ImportError("the fluctuation determinants of the 3D rate need BubbleDet: "
                          "pip install 'transitionlistener[nlo]'") from exc
    return BubbleConfig, ParticleConfig, BubbleDeterminant


def _bubbledet_profile(r, phi3):
    """The bounce on a uniform grid of ``BUBBLEDET_POINTS`` radii, ``(R, Phi, dPhi)``, from a cubic spline with
    phi'(0) = 0. Used for BubbleDet only; every other integral of this module uses the solver's own grid."""
    r = np.asarray(r, dtype=float)
    from scipy.interpolate import CubicSpline
    spline = CubicSpline(r, np.asarray(phi3, dtype=float), bc_type=((1, 0.0), "not-a-knot"))
    R = np.linspace(r[0], r[-1], BUBBLEDET_POINTS)
    return R, spline(R), spline(R, 1)


def _bubble_config(r, phi3, V, dV, d2V):
    """BubbleDet's description of the bounce ``(r, phi3)`` of the potential ``V`` with derivatives ``dV``, ``d2V``."""
    BubbleConfig = _bubbledet()[0]
    R, Phi, dPhi = _bubbledet_profile(r, phi3)
    return BubbleConfig(V=V, phi_metaMin=0.0, R=R, Phi=Phi, dim=3, dV=dV, d2V=d2V, dPhi=dPhi)


def _determinant(bubble, W, *, spin=0, zero_modes="None"):
    """``(value, error)`` of BubbleDet's one-loop action of one field with mass squared ``W(phi3)`` on ``bubble``:
    (1/2) ln of its determinant relative to the false vacuum, renormalised, zero modes removed as ``zero_modes``."""
    _, ParticleConfig, BubbleDeterminant = _bubbledet()
    particle = ParticleConfig(W_Phi=W, spin=spin, dof_internal=1, zero_modes=zero_modes)
    value, error = BubbleDeterminant(bubble, [particle]).findSingleDeterminant(
        particle, gy_tol=BUBBLEDET_GY_TOL, tail=BUBBLEDET_TAIL)
    return float(value), float(error)


def _V_over_phi(dV, d2V):
    """V'(phi3)/phi3 as a function of phi3, with its limit V''(0) at phi3 = 0: the Goldstone mass squared."""
    def W(p):
        q = np.asarray(p, dtype=float)
        safe = np.where(np.abs(q) > 0.0, q, 1.0)
        return np.where(np.abs(q) > 0.0, dV(safe) / safe, d2V(np.zeros_like(q)))
    return W


def double_counting_subtraction(r, phi3, g3sq: float,
                                n_transverse: int = N_TRANSVERSE_ABELIAN,
                                *, m_debye_sq: float, n_temporal: int = N_TEMPORAL_ABELIAN, h3: float | None = None) -> float:
    r"""The subtraction :math:`C` of eq. (A.2), dimensionless: the gauge-loop term of :math:`V_3` integrated over the
    bounce,

    .. math:: C = -\frac{4\pi}{12\pi}\int_0^\infty dr\,r^2
              \big[n_T m_A^3 + n_0 (m_{A_0}^3 - m_D^3)\big] .

    These loops are in :math:`S_3/T` and again, in full, in the gauge determinants :math:`\delta_1, \delta_0`;
    :math:`E` contains :math:`-C` to count them once. It must carry the same terms as :func:`V3_LO`.
    ``r``, ``phi3``: the bounce; the other arguments as in :func:`V3_LO`.
    """
    r = np.asarray(r, dtype=float)
    phi3 = np.asarray(phi3, dtype=float)
    m_gauge_sq = g3sq * phi3**2
    density = n_transverse * m_gauge_sq**1.5
    if n_temporal:
        m0_sq = m_debye_sq + 0.5 * temporal_coupling(g3sq, h3) * phi3**2
        density = density + n_temporal * (m0_sq**1.5 - m_debye_sq**1.5)
    integrand = 4.0 * np.pi * r**2 * density
    return float(-np.trapz(integrand, r) / (12.0 * np.pi))


def soft_nlo_action(r, phi3, *, V3_nlo=None, Z3_nlo=None) -> dict:
    r"""The two-loop terms of the soft EFT on the bounce, in the derivative expansion, dimensionless.

    ``potential`` is :math:`4\pi\int dr\,r^2\,V_3^{\rm NLO}(\phi_3)`, the last term of eq. (A.2) but :math:`S_6`.
    ``kinetic`` is :math:`4\pi\int dr\,r^2\,\tfrac12 Z_3^{\rm NLO}(\phi_3)(d\phi_3/dr)^2`, the derivative expansion of
    the gauge determinants; it is *not* part of :math:`E`, which contains those determinants in full, and is
    returned for comparison only. ``total`` is their sum.

    ``V3_nlo`` and ``Z3_nlo`` are functions of :math:`\phi_3` in the units of ``r`` and ``phi3``, supplied by the
    model's matching (``two_loop_potential``, ``two_loop_kinetic``). An omitted one gives zero, and ``has_potential``,
    ``has_kinetic`` say which were given.
    """
    r = np.asarray(r, dtype=float)
    phi3 = np.asarray(phi3, dtype=float)

    potential = 0.0
    if V3_nlo is not None:
        potential = 4.0 * np.pi * np.trapz(r**2 * np.asarray(V3_nlo(phi3)), r)

    kinetic = 0.0
    if Z3_nlo is not None:
        dphi = np.gradient(phi3, r)
        # Z3 ~ 1/phi3 while dphi3/dr ~ phi3 in the tail, so the integrand vanishes there, but a finite-difference
        # slope at a profile that has reached zero does not: points below 1e-9 of the peak field, and the last point
        # before them (whose slope is a difference across the cut), count as the false vacuum and contribute zero
        nonzero = np.abs(phi3) > 1e-9 * np.max(np.abs(phi3))
        nonzero[:-1] &= nonzero[1:]
        z = np.zeros_like(phi3)
        z[nonzero] = np.asarray(Z3_nlo(np.maximum(np.abs(phi3[nonzero]), Z3_FIELD_FLOOR)))
        kinetic = 4.0 * np.pi * np.trapz(r**2 * 0.5 * z * dphi**2, r)

    return {
        "potential": float(potential),
        "kinetic": float(kinetic),
        "total": float(potential + kinetic),
        "has_potential": V3_nlo is not None,
        "has_kinetic": Z3_nlo is not None,
    }


def dim6_action_shift(r, phi3, c6: float) -> float:
    r"""The sextic part of :math:`S_6` in eq. (A.2): :math:`4\pi\int dr\,r^2\,\tfrac18 c_6\phi_3^6`, dimensionless.

    ``c6`` is the dimensionless coefficient of :math:`c_6 (\Phi^\dagger\Phi)^3`, :math:`\Phi = \phi_3/\sqrt2`, from
    the model's matching (``soft_parameters()["c6"]``); ``r`` and ``phi3`` the bounce. First order in the operator.
    """
    from scipy.integrate import simpson

    r = np.asarray(r, dtype=float)
    phi3 = np.asarray(phi3, dtype=float)
    return float(4.0 * np.pi * simpson(r**2 * (c6 / 8.0) * phi3**6, x=r))


def dim6_kinetic_action_shift(r, phi3, dphi3, c_D2phi4: float) -> float:
    r"""The derivative part of :math:`S_6` in eq. (A.2): :math:`4\pi\int dr\,r^2\,\tfrac14 c_D\phi_3^2(d\phi_3/dr)^2`,
    dimensionless.

    ``c_D2phi4`` is the coefficient :math:`c_D` of :math:`(\Phi^\dagger\Phi)(D_i\Phi)^\dagger(D_i\Phi)` in inverse
    energy units, from the model's matching (``soft_parameters()["c_D2phi4"]``, 1/GeV). ``r``, ``phi3`` and
    ``dphi3`` (from the ``info`` of :func:`solve_LO_bounce`) are the bounce in the same units. With the sextic term
    the only dimension-six operator that does not vanish on a bounce with zero gauge field.
    """
    from scipy.integrate import simpson

    r = np.asarray(r, dtype=float)
    phi3 = np.asarray(phi3, dtype=float)
    dphi3 = np.asarray(dphi3, dtype=float)
    return float(4.0 * np.pi * simpson(r**2 * (c_D2phi4 / 4.0) * phi3**2 * dphi3**2, x=r))


def _mixing_callables(r, phi3, m3sq, lam3, g3sq, *, m_debye_sq, n_temporal, h3=None, dV=None):
    """The mass functions of the coupled Goldstone-photon block along a bounce, as functions of r, for
    ``abelian_determinants.log_gy_mixed``: ``(WX, mXp, dWG, zero, r_max, m_G)`` with the photon mass squared g3^2 phi^2, the
    mixing g3 phi', the Goldstone mass squared V'/phi minus its false-vacuum value m3sq, a zero mixing, the end of the
    domain and the Goldstone's false-vacuum mass. ``dV`` replaces dV3/dphi3 for another potential.
    """
    from scipy.interpolate import CubicSpline

    r = np.asarray(r, dtype=float)
    phi3 = np.asarray(phi3, dtype=float)
    phi = CubicSpline(r, phi3)
    dphi = phi.derivative(1)
    g3 = np.sqrt(g3sq)
    WX = lambda rr: g3sq * float(phi(rr)) ** 2
    mXp = lambda rr: g3 * float(dphi(rr))
    own_dV = dV is None
    if own_dV:
        dV = lambda p: dV3_LO(p, m3sq, lam3, g3sq, m_debye_sq=m_debye_sq,
                              n_temporal=n_temporal, h3=h3)
    dWG = lambda rr: float(dV(float(phi(rr)))) / float(phi(rr)) - m3sq
    zero = lambda rr: 0.0
    # the same functions on an array of radii, which log_gy_mixed uses where present; a caller's dV may be scalar only
    WX.array = lambda rr: g3sq * phi(rr) ** 2
    mXp.array = lambda rr: g3 * dphi(rr)
    if own_dV:
        dWG.array = lambda rr: dV(phi(rr)) / phi(rr) - m3sq
    zero.array = lambda rr: np.zeros_like(rr)
    return WX, mXp, dWG, zero, float(r[-1]), np.sqrt(m3sq)


def _mixing_l0(callables):
    """The J = 0 term of the mixing sum from the output of :func:`_mixing_callables`."""
    from transitionlistener.eft_rate import abelian_determinants as fl

    WX, mXp, dWG, zero, rmax, mG = callables
    return float(fl.log_gy_mixed(WX, mXp, dWG, 0, rmax, mG)
                 - fl.log_gy_mixed(WX, zero, dWG, 0, rmax, mG))


def mixing_l0_term(r, phi3, m3sq: float, lam3: float, g3sq: float, *,
                   m_debye_sq: float, n_temporal: int = N_TEMPORAL_ABELIAN, h3: float | None = None,
                   dV=None) -> float:
    """The J = 0 term of :math:`\\Delta_{\\rm mix}` (weight 1), as a ln det difference with the mixing on and off.

    At J = 0 the Goldstone (orbital 0) mixes with the single J = 0 harmonic of the photon, its orbital-1 radial mode.
    """
    return _mixing_l0(_mixing_callables(r, phi3, m3sq, lam3, g3sq, m_debye_sq=m_debye_sq, n_temporal=n_temporal,
                                        h3=h3, dV=dV))


def _mixing_series_tail(ells, summand):
    r"""The remainder :math:`\sum_{l > l_{\rm max}} d_l` of the mixing sum from the large-l expansion of its summand.

    :math:`d_l = (2l+1)(\ln T_l^{\rm on} - \ln T_l^{\rm off})` falls as :math:`\nu^{-2}`, :math:`\nu = l + 1/2`, with
    corrections in further powers of :math:`1/\nu`. The coefficients of ``SERIES_TAIL_POWERS`` are fitted on the
    upper half of the computed waves and each power summed exactly, :math:`\sum_{l > L}\nu^{-p} = \zeta(p, L + 3/2)`.
    """
    from scipy.special import zeta
    ells = np.asarray(ells, dtype=float)
    summand = np.asarray(summand, dtype=float)
    l_max = int(ells[-1])
    upper = ells >= l_max // 2
    nu = ells[upper] + 0.5
    basis = np.array([nu ** -p for p in SERIES_TAIL_POWERS]).T
    coef = np.linalg.lstsq(basis, summand[upper], rcond=None)[0]
    return float(sum(c * zeta(p, l_max + 1.5) for c, p in zip(coef, SERIES_TAIL_POWERS)))


def mixing_log_det(r, phi3, m3sq: float, lam3: float, g3sq: float, *,
                   m_debye_sq: float, l_max: int = MIXING_L_MAX,
                   n_temporal: int = N_TEMPORAL_ABELIAN, h3: float | None = None, dV=None) -> dict:
    r""":math:`\Delta_{\rm mix}` of eq. (A.2): the change of :math:`\ln\det` of the Goldstone-gauge block when the
    mixing :math:`g_3\phi_3'` is switched on, summed over all :math:`J \ge 0`, dimensionless; it enters :math:`E`
    halved.

    Every partial wave from 1 to ``l_max`` is computed with the coupled solver of
    :mod:`transitionlistener.eft_rate.abelian_determinants`, plus the J = 0 block (:func:`mixing_l0_term`); the remainder beyond
    ``l_max`` comes from the large-l expansion of the summand (:func:`_mixing_series_tail`). Two other estimates of the
    remainder are returned beside it: a power-law fit of the summand and Richardson extrapolation of the partial sums
    in 1/L.

    Returns ``value`` (the expansion's estimate; below ``SERIES_TAIL_MIN_L_MAX`` the Richardson one), ``series``,
    ``power_law``, ``power_law_exponent``, ``richardson``, ``partial_sum``, ``l0_term``, ``spread`` (the range of all
    estimates, the uncertainty of ``value``), ``l_max``, ``r_max`` and the per-wave logarithms ``log_t_on``,
    ``log_t_off``. Inputs as in :func:`V3_LO`; ``dV`` replaces dV3/dphi3 for another potential.
    """
    from transitionlistener.eft_rate import abelian_determinants as fl

    WX, mXp, dWG, zero, rmax, mG = _mixing_callables(
        r, phi3, m3sq, lam3, g3sq, m_debye_sq=m_debye_sq, n_temporal=n_temporal, h3=h3, dV=dV)
    l0 = mixing_l0_term(r, phi3, m3sq, lam3, g3sq, m_debye_sq=m_debye_sq, n_temporal=n_temporal, h3=h3, dV=dV)

    ells = np.arange(1.0, l_max + 1.0)
    on, off = [], []
    for l in ells:
        off.append(fl.log_gy_mixed(WX, zero, dWG, int(l), rmax, mG))
        on.append(fl.log_gy_mixed(WX, mXp, dWG, int(l), rmax, mG))
    on, off = np.array(on), np.array(off)

    partial = l0 + np.cumsum((2.0 * ells + 1.0) * (on - off))
    fit = fl.log_det_difference_from_logs(on, off, ells,
                                          tail_from=max(1, l_max // 3),
                                          max_tail_fraction=1.0)
    richardson = {}
    for lo, hi in ((l_max // 2, l_max), (2 * l_max // 3, l_max)):
        SL, SH = partial[lo - 1], partial[hi - 1]
        c = (SH - SL) / (1.0 / lo - 1.0 / hi)
        richardson[f"L{lo}_{hi}"] = float(SH + c / hi)
    series = (_mixing_series_tail(ells, (2.0 * ells + 1.0) * (on - off)) + float(partial[-1])
              if l_max >= SERIES_TAIL_MIN_L_MAX else np.nan)
    estimates = [float(fit["total"]) + l0] + list(richardson.values()) + ([series] if np.isfinite(series) else [])
    return {
        "value": series if np.isfinite(series) else richardson[f"L{2 * l_max // 3}_{l_max}"],
        "series": float(series),
        "l0_term": float(l0),
        "power_law": float(fit["total"]) + l0,
        "power_law_exponent": float(fit["exponent"]),
        "richardson": richardson,
        "partial_sum": float(partial[-1]),
        "spread": float(max(estimates) - min(estimates)),
        "l_max": int(l_max),
        "r_max": rmax,
        "log_t_on": on.tolist(),
        "log_t_off": off.tolist(),
    }


def j0_sector(r, phi3, m3sq: float, lam3: float, g3sq: float, *,
              m_debye_sq: float, n_temporal: int = N_TEMPORAL_ABELIAN, h3: float | None = None,
              dV=None, tail_window=(1e-6, 1e-5)) -> dict:
    r"""The term :math:`S^{J=0}` of eq. (A.2), dimensionless: the :math:`J = 0` Goldstone-gauge sector with the zero
    mode of the global :math:`U(1)` rotation of the bounce replaced by an integral over its orientation.

    At :math:`\xi = 1` the :math:`J = 0` block (photon at orbital 1, Goldstone at orbital 0) has a normalisable zero
    mode that the ghosts do not cancel, so its Gelfand-Yaglom determinant depends on the radial domain. With the zero
    mode treated as a collective coordinate (arXiv:1704.03492 sec. 5.2, arXiv:2104.11804 sec. 5) the sector is gauge
    independent and equals

    .. math:: S_1^{J=0} = -\ln 2\pi - \tfrac12\ln\big(2 A\,\phi(0)\big),\qquad \phi(r) \to A\,e^{-m r}/r ,

    with :math:`m^2 = \lim V'/\phi` the false-vacuum mass. ``A`` is the median of :math:`\phi\,r\,e^{mr}` over the radii
    where :math:`\phi` lies between the fractions ``tail_window`` of :math:`\phi(0)`.

    Returns ``treated`` (:math:`S_1^{J=0}`); ``naive``, the same sector without the treatment,
    :math:`\tfrac12[\ln T_{\rm mix}^{J=0} + \ln T_\gamma^{\ell=1} + \ln T_\chi^{\ell=0}] - \ln T_{\rm gh}^{\ell=0}`
    (raw Gelfand-Yaglom logarithms, which the single channels and :func:`mixing_log_det` contain); ``correction =
    treated - naive``, which :func:`nlo_assembly` adds to :math:`E`; the tail amplitude and its relative spread over
    the window; the false-vacuum mass; and the four logarithms. Inputs as in :func:`mixing_log_det`.
    """
    from scipy.interpolate import CubicSpline
    from transitionlistener.eft_rate import abelian_determinants as fl

    r = np.asarray(r, dtype=float)
    phi3 = np.asarray(phi3, dtype=float)
    if dV is None:
        dV = lambda p: dV3_LO(p, m3sq, lam3, g3sq, m_debye_sq=m_debye_sq, n_temporal=n_temporal, h3=h3)
    phi0 = float(phi3[0])
    m_fv_sq = _false_vacuum_mass_sq(phi3, dV)
    if not m_fv_sq > 0.0:
        raise ValueError("the false vacuum is not a minimum along the bounce direction")
    m_fv = float(np.sqrt(m_fv_sq))

    lo, hi = tail_window
    win = (phi3 > lo * phi0) & (phi3 < hi * phi0) & (r > 0.0)
    if np.count_nonzero(win) < 5:
        raise ValueError(
            f"the bounce profile does not resolve its tail between {lo:g} and {hi:g} of "
            "phi(0); the J = 0 sector needs the tail amplitude")
    A_r = phi3[win] * r[win] * np.exp(m_fv * r[win])
    A = float(np.median(A_r))
    treated = float(-np.log(2.0 * np.pi) - 0.5 * np.log(2.0 * A * phi0))

    spl = CubicSpline(r, phi3)
    rmax = float(r[-1])
    WX = lambda rr: g3sq * float(spl(rr)) ** 2

    def WG_minus_fv(rr):
        q = float(spl(rr))
        v_over_phi = float(dV(q)) / q if abs(q) > 1e-300 else m_fv_sq
        return v_over_phi + WX(rr) - m_fv_sq

    l0 = mixing_l0_term(r, phi3, m3sq, lam3, g3sq, m_debye_sq=m_debye_sq,
                        n_temporal=n_temporal, h3=h3, dV=dV)
    ln_photon_l1 = fl.log_gy_single(WX, 1, rmax, 0.0)
    ln_goldstone_l0 = fl.log_gy_single(WG_minus_fv, 0, rmax, m_fv)
    ln_ghost_l0 = fl.log_gy_single(WX, 0, rmax, 0.0)
    naive = float(0.5 * (l0 + ln_photon_l1 + ln_goldstone_l0) - ln_ghost_l0)
    return {
        "treated": treated,
        "naive": naive,
        "correction": treated - naive,
        "tail_amplitude": A,
        "tail_amplitude_spread": float((A_r.max() - A_r.min()) / A),
        "false_vacuum_mass": m_fv,
        "mixing_l0": float(l0),
        "ln_T_photon_l1": float(ln_photon_l1),
        "ln_T_goldstone_xi1_l0": float(ln_goldstone_l0),
        "ln_T_ghost_l0": float(ln_ghost_l0),
    }


def nlo_assembly(r, phi3, m3sq: float, lam3: float, g3sq: float,
                 m_debye_sq: float, S3_over_T: float, log_det_mixing: float,
                 n_transverse: int = N_TRANSVERSE_ABELIAN,
                 n_temporal: int = N_TEMPORAL_ABELIAN,
                 verbose: bool = False, *, T: float, h3: float | None = None,
                 V3_nlo=None, Z3_nlo=None, potential=None,
                 temporal_mass_sq=None, subtraction=None) -> dict:
    r"""The exponent :math:`E` of eq. (A.2) on the bounce ``(r, phi3)`` of :func:`solve_LO_bounce`, but :math:`S_6`.

    .. math:: E = S_3/T + \delta_h + \delta_\chi + \delta_1 + \delta_0 + \tfrac12\Delta_{\rm mix} + S^{J=0} - C
                  + 4\pi\int dr\,r^2\,V_3^{\rm NLO}

    in Feynman-'t Hooft gauge, so that :math:`\ln(\Gamma/V) = \ln(A_{\rm dyn}/T) + 4\ln T - E`. The channels are
    BubbleDet's one-loop actions (already :math:`\tfrac12\ln\det`, relative to the false vacuum) of the fields with
    mass squared

    * :math:`\delta_h`, ``radial``: :math:`V''`, with the three translational zero modes removed;
    * :math:`\delta_\chi`, ``goldstone_xi1``: :math:`V'/\phi + g_3^2\phi^2` (no zero mode at :math:`\xi = 1`);
    * :math:`\delta_1`, ``gauge_single_channel``: :math:`g_3^2\phi^2`. At :math:`\xi = 1` and without the mixing the
      three spatial photon components are three such channels and the two ghosts remove two;
    * :math:`\delta_0`, ``temporal``: :math:`m_D^2 + \tfrac12 h_3\phi^2` (zero for ``n_temporal = 0``).

    ``log_det_mixing`` is :math:`\Delta_{\rm mix}` (``mixing_log_det()["value"]``), passed in because it is
    expensive; it must include the :math:`J = 0` block. :math:`S^{J=0}` is ``j0_sector()["correction"]``, :math:`C`
    is :func:`double_counting_subtraction` and the last term is ``soft_nlo_action()["potential"]`` of ``V3_nlo``.

    Units: ``T`` is the temperature in the units of the 3D parameters. All channels are unit free but the radial
    one, whose zero-mode removal leaves mass dimension three; it is returned in units of T
    (``radial = radial_input_units + 3 ln T``) and contains the zero-mode Jacobian :math:`(S_3/2\pi T)^{3/2}`.
    ``log_A_dyn_over_T`` is :math:`\ln(A_{\rm dyn}/T)`, :math:`A_{\rm dyn} = \sqrt{|\lambda_-|}/2\pi` from the
    negative eigenvalue :math:`\lambda_-` of the radial operator. :func:`action_seam_correction` turns ``total`` and
    ``log_A_dyn_over_T`` into :math:`\Delta S` of eq. (A.4).

    Optional inputs:

    * ``V3_nlo``: :math:`V_3^{\rm NLO}(\phi_3)`. Without it the result has ``total_without_NLO_potential`` and no
      ``total``.
    * ``Z3_nlo``: reported as ``nlo_kinetic`` for comparison only; it is the derivative expansion of the gauge
      determinants already in :math:`E` (see :func:`soft_nlo_action`).
    * ``potential = (V, dV, d2V)``, ``temporal_mass_sq`` (a function of :math:`\phi_3`) and ``subtraction`` replace
      :func:`V3_LO` and its derivatives, the temporal mass squared and :math:`C`, for a rate built on another
      leading-order potential; pass the same ``dV`` to :func:`mixing_log_det`.
    * ``verbose``: print each channel and its time.

    Returns the channels named above, ``S3_LO``, ``zero_mode_jacobian_log``, ``negative_eigenvalue``,
    ``mixing_log_det``, ``j0_correction``, ``j0_sector``, ``one_loop`` (the sum of the determinant terms),
    ``double_counting_C``, ``total_without_NLO_potential``, BubbleDet's ``errors``, and with ``V3_nlo``
    ``nlo_potential``, ``nlo_kinetic`` and ``total``.
    """
    import time as _time

    _, ParticleConfig, BubbleDeterminant = _bubbledet()

    r = np.asarray(r, dtype=float)
    phi3 = np.asarray(phi3, dtype=float)

    if potential is None:
        V_lo, dV_lo, d2V_lo = _lo_callables(m3sq, lam3, g3sq, n_transverse, m_debye_sq=m_debye_sq,
                                            n_temporal=n_temporal, h3=h3)
    else:
        V_lo, dV_lo, d2V_lo = potential
    W_radial = lambda p: d2V_lo(np.asarray(p, dtype=float))
    W_gauge = lambda p: g3sq * np.asarray(p, dtype=float) ** 2
    V_over_phi = _V_over_phi(dV_lo, d2V_lo)
    W_goldstone_xi1 = lambda p: V_over_phi(p) + W_gauge(p)
    if temporal_mass_sq is None:
        W_temporal = lambda p: m_debye_sq + 0.5 * temporal_coupling(g3sq, h3) * np.asarray(p, dtype=float) ** 2
    else:
        W_temporal = lambda p: temporal_mass_sq(np.asarray(p, dtype=float))

    bubble = _bubble_config(r, phi3, V_lo, dV_lo, W_radial)

    def channel(W, zero_modes="None", name=""):
        t0 = _time.time()
        value, error = _determinant(bubble, W, zero_modes=zero_modes)
        if verbose:
            print(f"      {name:24s} {value:14.4f}  ({_time.time() - t0:6.1f} s)", flush=True)
        return value, error

    D_h, e_h = channel(W_radial, zero_modes="Higgs", name="radial")
    eig_neg, _ = BubbleDeterminant(
        bubble, [ParticleConfig(W_Phi=W_radial, spin=0, dof_internal=1,
                                zero_modes="Higgs")]).findNegativeEigenvalue()
    D_chi, e_chi = channel(W_goldstone_xi1, name="goldstone (xi=1)")
    D_1, e_1 = channel(W_gauge, name="gauge, 1 channel")
    D_0, e_0 = (channel(W_temporal, name="temporal A0") if n_temporal
                else (0.0, 0.0))

    C = (double_counting_subtraction(r, phi3, g3sq, n_transverse, m_debye_sq=m_debye_sq, n_temporal=n_temporal, h3=h3)
         if subtraction is None else float(subtraction))

    if not T > 0.0:
        raise ValueError("T must be positive, in the units of the 3D parameters")
    D_h_input_units = D_h
    D_h = D_h_input_units + 3.0 * np.log(T)

    j0 = j0_sector(r, phi3, m3sq, lam3, g3sq, m_debye_sq=m_debye_sq, n_temporal=n_temporal, h3=h3, dV=dV_lo)
    one_loop = (D_h + D_chi + D_1 + 0.5 * float(log_det_mixing) + D_0 + j0["correction"])
    without_nlo_potential = float(S3_over_T) + one_loop - C

    out = {
        "S3_LO": float(S3_over_T),
        "radial": D_h,
        "radial_input_units": D_h_input_units,
        "zero_mode_jacobian_log": float(1.5 * np.log(S3_over_T / (2.0 * np.pi))),
        "negative_eigenvalue": float(eig_neg),
        "log_A_dyn_over_T": float(np.log(np.sqrt(abs(eig_neg)) / (2.0 * np.pi)) - np.log(T)),
        "goldstone_xi1": D_chi,
        "gauge_single_channel": D_1,
        "mixing_log_det": float(log_det_mixing),
        "j0_correction": float(j0["correction"]),
        "j0_sector": j0,
        "temporal": D_0,
        "one_loop": float(one_loop),
        "double_counting_C": float(C),
        "total_without_NLO_potential": float(without_nlo_potential),
        "errors": {"radial": e_h, "goldstone": e_chi, "gauge": e_1,
                   "temporal": e_0},
    }

    if V3_nlo is None:
        if Z3_nlo is not None:
            out["nlo_kinetic"] = soft_nlo_action(r, phi3, Z3_nlo=Z3_nlo)["kinetic"]
        return out

    nlo = soft_nlo_action(r, phi3, V3_nlo=V3_nlo, Z3_nlo=Z3_nlo)
    out.update({
        "nlo_potential": nlo["potential"],
        "nlo_kinetic": nlo["kinetic"],
        "total": float(without_nlo_potential + nlo["potential"]),
    })
    return out


# ----------------------------------------------------------------------
# 4. Delta S, eq. (A.4)
# ----------------------------------------------------------------------
def action_seam_correction(S3_LO: float, exponent_minus_S3: float,
                           log_A_dyn_over_T: float = 0.0,
                           tol: float = 1e-12, max_iter: int = 100) -> float:
    r""":math:`\Delta S` of eq. (A.4), the correction to :math:`S_3/T` with which TransitionListener's rate
    :math:`\ln(\Gamma/V) = 4\ln T + \tfrac32\ln(S/2\pi) - S` equals the NLO rate
    :math:`\ln(A_{\rm dyn}/T) + 4\ln T - E`:

    .. math:: \Delta S = (E - S_3/T) - \ln\frac{A_{\rm dyn}}{T} + \tfrac32\ln\frac{S_3/T + \Delta S}{2\pi} .

    ``S3_LO`` is :math:`S_3/T`, ``exponent_minus_S3`` is :math:`E - S_3/T` and ``log_A_dyn_over_T`` is
    :math:`\ln(A_{\rm dyn}/T)` (zero is the estimate :math:`A_{\rm dyn} = T`), all dimensionless. Solved by
    fixed-point iteration, which contracts by :math:`\tfrac32/(S_3/T + \Delta S)`. The zero-mode Jacobian in
    :math:`E` is thereby not counted twice.
    """
    y = float(exponent_minus_S3) - float(log_A_dyn_over_T)
    s3 = float(S3_LO)
    dS = y
    for _ in range(max_iter):
        if s3 + dS <= 0.0:
            raise ValueError("the NLO action is not positive; no rate to match")
        new = y + 1.5 * np.log((s3 + dS) / (2.0 * np.pi))
        if abs(new - dS) < tol * max(1.0, abs(new)):
            return float(new)
        dS = new
    raise RuntimeError("action_seam_correction did not converge")


# ----------------------------------------------------------------------
# 5. Delta S tabulated in T at one parameter point
# ----------------------------------------------------------------------
class NLOCorrection:
    r""":math:`\Delta S(T)` at one parameter point, interpolated from a table in T.

    Inside the table: a C^1 cubic Hermite spline in ln T with PCHIP slopes, which keeps monotone data monotone.
    Outside: the ratio :math:`\Delta S / S_3` is held at its edge value and multiplied by the live leading-order action,
    with the end slopes of the spline matched to that continuation so that the result is C^1 across the edges.
    :meth:`extrapolating` says whether a query left the table. ``T_GeV``, ``delta_S`` and ``S3_LO`` are the
    temperatures, :math:`\Delta S` and :math:`S_3/T` of the table, at least four distinct positive temperatures.
    """

    def __init__(self, T_GeV, delta_S, S3_LO):
        from scipy.interpolate import CubicHermiteSpline, PchipInterpolator

        T = np.asarray(T_GeV, dtype=float)
        d = np.asarray(delta_S, dtype=float)
        if T.ndim != 1 or T.shape != d.shape:
            raise ValueError("T_GeV and delta_S must be 1D and the same length")
        if np.any(T <= 0.0):
            raise ValueError("temperatures must be positive")
        if T.size < 4:
            raise ValueError(f"a table needs at least 4 points, got {T.size}")
        order = np.argsort(T)
        self.T_GeV = T[order]
        self.delta_S = d[order]
        self._x = np.log(self.T_GeV)
        if np.any(np.diff(self._x) <= 0.0):
            raise ValueError("temperatures must be distinct")
        self.T_range = (float(self.T_GeV[0]), float(self.T_GeV[-1]))

        slopes = PchipInterpolator(self._x, self.delta_S).derivative()(self._x)
        s3 = np.asarray(S3_LO, dtype=float)
        if s3.shape != self.delta_S.shape or np.any(s3 <= 0.0):
            raise ValueError("S3_LO must be positive and match delta_S")
        s3 = s3[order]
        self.S3_LO = s3
        self.r_lo = float(self.delta_S[0] / s3[0])
        self.r_hi = float(self.delta_S[-1] / s3[-1])
        slopes[0] = self.r_lo * (s3[1] - s3[0]) / (self._x[1] - self._x[0])
        slopes[-1] = self.r_hi * (s3[-1] - s3[-2]) / (self._x[-1] - self._x[-2])
        self._spline = CubicHermiteSpline(self._x, self.delta_S, slopes)

    def __call__(self, T_GeV, S3_LO):
        """Delta S at one temperature or an array of them; ``S3_LO`` (same shape) is the live leading-order action."""
        T = np.asarray(T_GeV, dtype=float)
        x = np.log(T)
        out = np.array(self._spline(np.clip(x, self._x[0], self._x[-1])), dtype=float)
        lo, hi = x < self._x[0], x > self._x[-1]
        if np.any(lo) or np.any(hi):
            live = np.broadcast_to(np.asarray(S3_LO, dtype=float), T.shape)
            out = np.where(lo, self.r_lo * live, out)
            out = np.where(hi, self.r_hi * live, out)
        return float(out) if out.ndim == 0 else out

    def extrapolating(self, T_GeV) -> bool:
        """True if any temperature asked for lies outside the tabulated range."""
        T = np.asarray(T_GeV, dtype=float)
        return bool(np.any(T < self.T_range[0]) or np.any(T > self.T_range[1]))

    def __repr__(self):
        return (f"NLOCorrection({self.T_GeV.size} points, "
                f"T in [{self.T_range[0]:.4g}, {self.T_range[1]:.4g}] GeV, "
                f"monotone interpolation, ratio to S_3 held outside)")


# ----------------------------------------------------------------------
# validity of the reduction
# ----------------------------------------------------------------------
def validity_ratios(T: float, m3sq: float, lam3: float, g3sq: float,
                    m_debye_sq: float, r=None, phi3=None, *,
                    fermion_masses_over_field=None,
                    c6: float | None = None, S3_over_T: float | None = None,
                    n_transverse: int = N_TRANSVERSE_ABELIAN,
                    n_temporal: int = N_TEMPORAL_ABELIAN, h3: float | None = None) -> dict:
    """The ratios that :func:`transitionlistener.eft_rate.validity.evaluate` turns into a verdict; small means the
    reduction is justified. ``T`` and the parameters in one energy unit; ``r``, ``phi3`` the bounce.

    ``hard``
        The heaviest kept boson (gauge g3 phi3, radial, Goldstone) over 2 pi T, maximised along the bounce; without
        a bounce, at the escape point.
    ``hard_fermion``
        The heaviest fermion over pi T at the largest field of the bounce, from ``fermion_masses_over_field`` (mass
        per unit 4D field, phi = phi3 sqrt T). None if not supplied; an empty sequence gives zero.
    ``debye``
        The Debye mass over 2 pi T.
    ``soft_separation``
        The 3D mass sqrt(m3sq) over the Debye mass.
    ``expansion_3d``
        g3^2 / (pi sqrt(m3sq)), the loop-expansion parameter x_3 of the EFT.
    ``dim6``
        |S_6| / S_3 with the sextic part of S_6 (:func:`dim6_action_shift`); None without the bounce, ``c6`` and
        ``S3_over_T``.
    """
    if T <= 0.0:
        raise ValueError("temperature must be positive")

    if r is not None and phi3 is not None:
        field = np.abs(np.asarray(phi3, dtype=float))
    else:
        escape = escape_point_LO(m3sq, lam3, g3sq, n_transverse,
                                 m_debye_sq=m_debye_sq, n_temporal=n_temporal, h3=h3)
        field = np.array([escape if np.isfinite(escape) else 0.0])

    m_gauge = np.sqrt(g3sq) * field
    m_radial = np.sqrt(np.abs(m3sq + 3.0 * lam3 * field**2))
    m_goldstone = np.sqrt(np.abs(m3sq + lam3 * field**2))
    heaviest_kept = float(np.max(np.maximum(m_gauge,
                                            np.maximum(m_radial, m_goldstone))))

    m_debye = float(np.sqrt(max(m_debye_sq, 0.0)))
    m_light = float(np.sqrt(max(m3sq, 0.0)))

    if fermion_masses_over_field is None:
        hard_fermion = None
    else:
        coefficients = [abs(float(c)) for c in fermion_masses_over_field]
        phi4_max = float(np.max(field)) * np.sqrt(T)
        hard_fermion = (max(coefficients) * phi4_max / (np.pi * T)
                        if coefficients else 0.0)

    ratios: dict[str, float | None] = {
        "hard": heaviest_kept / (2.0 * np.pi * T),
        "hard_fermion": hard_fermion,
        "debye": m_debye / (2.0 * np.pi * T),
        "soft_separation": (m_light / m_debye) if m_debye > 0.0 else float("inf"),
        "expansion_3d": (g3sq / (np.pi * m_light)) if m_light > 0.0 else float("inf"),
    }

    if (c6 is not None and S3_over_T is not None
            and r is not None and phi3 is not None):
        from transitionlistener.eft_rate.validity import dim6_ratio
        ratios["dim6"] = dim6_ratio(dim6_action_shift(r, phi3, c6), S3_over_T)
    else:
        ratios["dim6"] = None

    return ratios


# ----------------------------------------------------------------------
# checks and diagnostics, not part of the rate
# ----------------------------------------------------------------------

