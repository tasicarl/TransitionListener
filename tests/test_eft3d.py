"""Tests for the three-dimensional leading-order effective potential (WP5a)."""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "models")
)

from transitionlistener.eft_rate import abelian_higgs_3d
import dr_conformal_u1_fermions_matching as dr

G0, Y0 = 0.9, 0.3
LAM0, V = 0.04600619792147168, 1.0
TC = 0.2695



def _V4_from_V3(phi, T, m3sq, lam3, g3sq, **kw):
    """The potential of V3_LO in 4D normalisation, T V3(phi / sqrt T): its bounce action over T is the 3D action."""
    return T * abelian_higgs_3d.V3_LO(np.asarray(phi, dtype=float) / np.sqrt(T), m3sq, lam3, g3sq, **kw)

def _matched(T_over_Tc, run=True):
    T = T_over_Tc * TC
    if run:
        g, y, lam, _ = dr.rg_run(G0, Y0, LAM0, V, 7.0553 * T)
    else:
        g, y, lam = G0, Y0, LAM0
    return (T,
            dr.scalar_thermal_mass_sq_LO(g, y, lam, T),
            dr.lambda3_LO(lam, T),
            dr.g3_sq_LO(g, T),
            dr.debye_mass_sq_LO(g, T))


# The temporal gauge mode belongs in the leading-order potential -- both
# arXiv:2312.12413 eq. (32) and arXiv:2503.13597 eq. (4.2) put it there -- so
# every call needs the Debye mass. Passing it is not optional: the functions
# take it keyword-only with no default, so omitting it raises rather than
# silently giving the transverse-only potential.


# --------------------------------------------------------------- shape
def test_potential_vanishes_at_the_origin():
    _, m3sq, l3, g3sq, mD = _matched(0.5)
    assert abelian_higgs_3d.V3_LO(0.0, m3sq, l3, g3sq, m_debye_sq=mD) == 0.0


def test_cubic_coefficient_is_the_transverse_gauge_loop():
    """With the temporal mode switched off, the barrier is -(n_T/12pi)(g3 phi3)^3."""
    m3sq, l3, g3sq, p = 0.0, 0.0, 4.0, 1.7        # mass and quartic off
    expected = -2.0 * (math.sqrt(g3sq) * p) ** 3 / (12.0 * math.pi)
    assert abelian_higgs_3d.V3_LO(p, m3sq, l3, g3sq, m_debye_sq=1.0,
                       n_temporal=0) == pytest.approx(expected, rel=1e-12)


def test_the_temporal_mode_adds_its_own_cubic_with_a_debye_counterterm():
    """arXiv:2503.13597 eq. (4.2): the temporal mode is in the LO potential too.

    Its term is -(n_0/12pi)[(mD^2 + g3^2 phi^2)^{3/2} - (mD^2)^{3/2}], the
    counterterm being what makes the potential vanish at zero field. Where these
    bounces live it is about half the transverse term again.
    """
    m3sq, l3, g3sq, p, mDsq = 0.0, 0.0, 4.0, 1.7, 0.7
    transverse = -2.0 * (math.sqrt(g3sq) * p) ** 3 / (12.0 * math.pi)
    temporal = -((mDsq + g3sq * p**2) ** 1.5 - mDsq**1.5) / (12.0 * math.pi)
    assert abelian_higgs_3d.V3_LO(p, m3sq, l3, g3sq, m_debye_sq=mDsq) == pytest.approx(
        transverse + temporal, rel=1e-12)


def test_the_debye_counterterm_keeps_the_origin_at_zero():
    """Without it the potential would not vanish at zero field."""
    assert abelian_higgs_3d.V3_LO(0.0, 0.3, 0.2, 4.0, m_debye_sq=0.7) == 0.0


def test_where_the_bounce_lives_the_temporal_term_is_not_a_small_correction():
    """It halves the action, so it must not be tiny.

    At the escape point of the running benchmark, g3^2 phi^2 sits well above
    mD^2, so the temporal mode is no heavier than the transverse ones and its
    cubic is comparable rather than suppressed.
    """
    _T, m3sq, l3, g3sq, mD = _matched(0.05)
    esc = abelian_higgs_3d.escape_point_LO(m3sq, l3, g3sq, m_debye_sq=mD)
    assert g3sq * esc**2 > 5.0 * mD          # the heavy-A0 expansion does not apply
    transverse = -2.0 * (g3sq * esc**2) ** 1.5 / (12.0 * math.pi)
    temporal = -((mD + g3sq * esc**2) ** 1.5 - mD**1.5) / (12.0 * math.pi)
    assert abs(temporal) > 0.3 * abs(transverse)


def test_the_barrier_is_even_in_the_field():
    _, m3sq, l3, g3sq, mD = _matched(0.5)
    p = 0.3
    assert abelian_higgs_3d.V3_LO(p, m3sq, l3, g3sq, m_debye_sq=mD) == pytest.approx(
        abelian_higgs_3d.V3_LO(-p, m3sq, l3, g3sq, m_debye_sq=mD), rel=1e-12)


# ------------------------------------------------ the running is essential


def test_the_quartic_runs_negative_under_supercooling():
    """Dimensional transmutation: lambda3 < 0 well below the breaking scale."""
    assert _matched(0.5)[2] > 0.0
    assert _matched(0.02)[2] < 0.0


# ----------------------------------------- unboundedness is reported, not hidden
def test_negative_quartic_is_reported_as_unbounded():
    _, m3sq, l3, g3sq, mD = _matched(0.02)
    assert l3 < 0.0
    assert not abelian_higgs_3d.is_bounded_below(l3), (
        "a negative 3D quartic makes V3 unbounded below; a bounce solver given "
        "an unrestricted field range will run away"
    )


def test_the_potential_really_does_run_away_when_unbounded():
    """Guard the claim the docstring makes, rather than asserting it in prose."""
    _, m3sq, l3, g3sq, mD = _matched(0.02)
    assert abelian_higgs_3d.V3_LO(50.0, m3sq, l3, g3sq, m_debye_sq=mD) < 0.0
    assert abelian_higgs_3d.V3_LO(500.0, m3sq, l3, g3sq, m_debye_sq=mD) < abelian_higgs_3d.V3_LO(50.0, m3sq, l3, g3sq, m_debye_sq=mD)


# ----------------------------------------------- escape point and validity
def test_escape_point_is_where_the_potential_returns_to_zero():
    _, m3sq, l3, g3sq, mD = _matched(0.02)
    esc = abelian_higgs_3d.escape_point_LO(m3sq, l3, g3sq, m_debye_sq=mD)
    assert abelian_higgs_3d.V3_LO(esc, m3sq, l3, g3sq, m_debye_sq=mD) == pytest.approx(0.0, abs=1e-14)


def test_escape_point_reproduces_the_4d_measurement():
    """The four-dimensional treatment gives phi_escape/T = 1.5-2.7.

    An independent check on the matching and the conventions: the 3D escape
    point, converted to phi/T, must land in that band.
    """
    T, m3sq, l3, g3sq, mD = _matched(0.05)
    escape = abelian_higgs_3d.escape_point_LO(m3sq, l3, g3sq, m_debye_sq=mD) / math.sqrt(T)
    # With the temporal mode in the potential the escape point is 1.60, against
    # 2.37 without it.
    assert 1.5 <= escape <= 2.7


def test_escape_point_falls_with_supercooling():
    """Also observed in the 4D treatment."""
    ratios = []
    for f in (0.05, 0.01, 1e-3):
        T, m3sq, l3, g3sq, mD = _matched(f)
        ratios.append(abelian_higgs_3d.escape_point_LO(m3sq, l3, g3sq, m_debye_sq=mD) / math.sqrt(T))
    assert ratios[0] > ratios[1] > ratios[2]


# --------------------------------- the artificial wall, and when it dominates


# ------------------------------- the Coleman-Weinberg term in the 3D quartic
def test_one_loop_quartic_adds_the_coleman_weinberg_term():
    """At the central scale the one-loop matching adds +g^4/(8 pi^2).

    arXiv:2312.12413 below eq. (4.28): "the Coleman-Weinberg term is captured
    by the L_b term in lambda_3". For a conformal model that term is the
    radiative structure itself, not a correction.
    """
    T = 0.05 * TC
    mu = 7.0553 * T                       # 4 pi exp(-gamma_E) T, so Lb = 0
    g, y, lam, _ = dr.rg_run(G0, Y0, LAM0, V, mu)
    tree = dr.lambda3_LO(lam, T) / T
    full = dr.lambda3_NLO(g, y, lam, T, mu) / T
    assert full - tree == pytest.approx(g**4 / (8 * math.pi**2), rel=0.05)


def test_dropping_the_cw_term_wrongly_unbinds_the_potential():
    """The regression this guards: lambda3_LO alone flips the sign of the quartic."""
    T = 0.06 * TC
    mu = 7.0553 * T
    g, y, lam, _ = dr.rg_run(G0, Y0, LAM0, V, mu)
    assert not abelian_higgs_3d.is_bounded_below(dr.lambda3_LO(lam, T))      # tree: unbounded
    assert abelian_higgs_3d.is_bounded_below(dr.lambda3_NLO(g, y, lam, T, mu))  # one loop: bounded


# ------------------------------------------------------------ units
def test_the_3d_action_is_already_S3_over_T():
    """Solving V3 directly gives S3/T; solving T*V3(phi/sqrt T) gives S3.

    Guards a mistake that is easy to make and costs orders of magnitude: the
    three-dimensional action is dimensionless, so dividing it by T a second
    time yields S3/T^2. The two routes must agree once each is interpreted
    correctly.
    """
    from transitionlistener.tunneling1D import SingleFieldInstanton
    from scipy.optimize import minimize_scalar

    T, m3sq, l3, g3sq, mD = _matched(0.09)          # inside the bounded window
    if not abelian_higgs_3d.is_bounded_below(l3):
        pytest.skip("benchmark not in the bounded window at this temperature")

    st = math.sqrt(T)
    r = minimize_scalar(lambda p: float(abelian_higgs_3d.V3_LO(p, m3sq, l3, g3sq, m_debye_sq=mD)),
                        bounds=(1e-7, 300.0), method="bounded")
    pmin = float(r.x)

    # route 1: solve the 3D potential; the action is S3/T as it stands
    i3 = SingleFieldInstanton(
        0.0, pmin,
        lambda x: abelian_higgs_3d.V3_LO(pmin - np.asarray(x), m3sq, l3, g3sq, m_debye_sq=mD),
        lambda x: -abelian_higgs_3d.dV3_LO(pmin - np.asarray(x), m3sq, l3, g3sq, m_debye_sq=mD))
    S_over_T_direct = i3.findAction(i3.findProfile(xtol=1e-11, phitol=1e-11))

    # route 2: solve the 4D-normalised potential; that action has mass
    # dimension one, so here dividing by T IS right
    pmin4 = pmin * st
    i4 = SingleFieldInstanton(
        0.0, pmin4,
        lambda x: _V4_from_V3(pmin4 - np.asarray(x), T, m3sq, l3, g3sq, m_debye_sq=mD),
        lambda x: -st * abelian_higgs_3d.dV3_LO((pmin4 - np.asarray(x)) / st, m3sq, l3, g3sq,
                                     m_debye_sq=mD))
    S_over_T_via_4d = i4.findAction(i4.findProfile(xtol=1e-11, phitol=1e-11)) / T

    assert S_over_T_direct == pytest.approx(S_over_T_via_4d, rel=1e-3)


# --------------------------------------------- double-counting subtraction
def test_subtraction_is_negative_and_scales_with_the_gauge_coupling():
    """C removes the gauge cubic, so it is negative and grows as g3^3."""
    r = np.linspace(1e-6, 20.0, 4000)
    phi3 = 0.3 * np.exp(-r)
    c1 = abelian_higgs_3d.double_counting_subtraction(r, phi3, 0.1, m_debye_sq=0.05,
                                           n_temporal=0)
    c2 = abelian_higgs_3d.double_counting_subtraction(r, phi3, 0.4, m_debye_sq=0.05,
                                           n_temporal=0)
    assert c1 < 0.0
    assert c2 / c1 == pytest.approx((0.4 / 0.1) ** 1.5, rel=1e-6)
    # the temporal piece only makes it more negative, and breaks the pure
    # g3^3 scaling because its mass carries the Debye term as well
    with_temporal = abelian_higgs_3d.double_counting_subtraction(r, phi3, 0.1,
                                                      m_debye_sq=0.05)
    assert with_temporal < c1


def test_subtraction_vanishes_for_a_vanishing_profile():
    r = np.linspace(1e-6, 10.0, 500)
    # the Debye counterterm is what makes this vanish rather than integrate
    # (mD^2)^{3/2} over the volume
    assert abelian_higgs_3d.double_counting_subtraction(
        r, np.zeros_like(r), 0.2, m_debye_sq=0.3) == 0.0


def test_subtraction_matches_a_hand_integral_for_a_top_hat():
    """C = -(n_T/12pi) * (g3 phi)^3 * volume for a constant profile."""
    R, phi, g3sq, nT = 2.0, 0.5, 0.09, 2
    r = np.linspace(0.0, R, 200000)
    phi3 = np.full_like(r, phi)
    expected = -nT / (12 * math.pi) * (math.sqrt(g3sq) * phi) ** 3 * (4 / 3) * math.pi * R**3
    assert abelian_higgs_3d.double_counting_subtraction(
        r, phi3, g3sq, nT, m_debye_sq=0.4, n_temporal=0) == pytest.approx(
            expected, rel=1e-4)

    # and with the temporal mode, against the same integral done by hand
    mDsq = 0.4
    vol = (4 / 3) * math.pi * R**3
    expected_full = expected - ((mDsq + g3sq * phi**2) ** 1.5
                                - mDsq**1.5) * vol / (12 * math.pi)
    assert abelian_higgs_3d.double_counting_subtraction(
        r, phi3, g3sq, nT, m_debye_sq=mDsq) == pytest.approx(expected_full, rel=1e-4)


# ------------------------------------- mass functions used by the determinant
def test_determinant_mass_functions_match_derivatives_of_V3():
    """W_radial must be V'' and W_goldstone V'/phi, cubic term included.

    An earlier version had V'' a factor of two too small and dropped the cubic
    from V'/phi entirely, which silently changed two of the four determinant
    channels. Checked here against finite differences of V3_LO itself.
    """
    _T, m3sq, l3, g3sq, mD = _matched(0.09)
    h = 1e-6
    for p in (0.05, 0.15, 0.4):
        V = lambda x: float(abelian_higgs_3d.V3_LO(x, m3sq, l3, g3sq, m_debye_sq=mD))
        dV = lambda x: float(abelian_higgs_3d.dV3_LO(x, m3sq, l3, g3sq, m_debye_sq=mD))
        # V'' against a second difference of V, and against a first difference
        # of V' -- the second is the sharper of the two
        assert float(abelian_higgs_3d.d2V3_LO(p, m3sq, l3, g3sq, m_debye_sq=mD)) == (
            pytest.approx((V(p + h) - 2 * V(p) + V(p - h)) / h**2, rel=1e-4))
        assert float(abelian_higgs_3d.d2V3_LO(p, m3sq, l3, g3sq, m_debye_sq=mD)) == (
            pytest.approx((dV(p + h) - dV(p - h)) / (2 * h), rel=1e-8))
        # V' against a first difference of V
        assert dV(p) == pytest.approx((V(p + h) - V(p - h)) / (2 * h), rel=1e-8)


def test_the_bounce_reproduces_the_exact_phi2_phi4_action():
    """With the gauge coupling off, V3 = m^2 phi^2/2 + lam phi^4/4 with lam < 0,
    whose bounce action is exactly 18.89725 m/|lam| (shooting to 1e-12 in the
    Derrick identity). Checks the solver, the reflection, and that the stabilising
    wall does not leak into the action: at power 12 it adds +0.063%.
    """
    exact = 18.89725
    for m2, lam in ((1.0, -1.0), (0.01, -0.02), (4.0, -0.5)):
        S = abelian_higgs_3d.solve_LO_bounce(m2, lam, 0.0, m_debye_sq=1.0)[0]
        assert S * abs(lam) / math.sqrt(m2) == pytest.approx(exact, rel=2e-6)
    old = abelian_higgs_3d.solve_LO_bounce(1.0, -1.0, 0.0, m_debye_sq=1.0, wall_power=12)[0]
    assert old / exact - 1.0 == pytest.approx(6.3e-4, rel=0.05)


def test_escape_point_survives_roundoff_at_its_first_probe():
    """Conformal U(1), g = 0.8, cold side (T/v = 1.4e-6): the mass term at 1e-8 of the scale is 6e-17 of the
    Debye term, so V there has the wrong sign by roundoff. The neighbouring temperatures have a bounce."""
    m3sq, lam3, g3sq, mD2 = (float.fromhex(x) for x in (
        "0x1.c652e440eca36p-43", "-0x1.07cc040901bcbp-21", "0x1.5b2e741c20e5ep-20", "0x1.fed9f6ac48c66p-41"))
    a = 0.25 * lam3
    b = -2.0 * g3sq**1.5 / (12.0 * np.pi)
    scale = min(r for r in np.roots([a, b, 0.5 * m3sq]) if r > 0.0)
    assert abelian_higgs_3d.V3_LO(1e-8 * scale, m3sq, lam3, g3sq, m_debye_sq=mD2) <= 0.0   # the case this test is about

    escape = abelian_higgs_3d.escape_point_LO(m3sq, lam3, g3sq, m_debye_sq=mD2)
    assert np.isfinite(escape)
    assert abs(abelian_higgs_3d.V3_LO(escape, m3sq, lam3, g3sq, m_debye_sq=mD2)) < 1e-10 * abs(0.5 * m3sq * escape**2)
    assert 3.0 < abelian_higgs_3d.solve_LO_bounce(m3sq, lam3, g3sq, m_debye_sq=mD2)[0] < 10.0


def test_escape_point_still_refuses_a_tachyonic_false_vacuum():
    m3sq, lam3, g3sq, mD2 = -1e-12, -4.9e-7, 1.3e-6, 1e-12
    assert np.isnan(abelian_higgs_3d.escape_point_LO(m3sq, lam3, g3sq, m_debye_sq=mD2))
