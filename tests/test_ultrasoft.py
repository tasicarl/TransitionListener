"""Soft -> ultrasoft matching and the two-loop ultrasoft potential.

Everything here comes from DRalgo run 2. None of it is taken on trust: each
relation is checked against something computed a different way.
"""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
import dr_conformal_u1_fermions_matching as dr  # noqa: E402

# A representative ultrasoft parameter set. 3D mass dimensions:
# [phi3] = 1/2, [lam3] = [g3^2] = 1, [m3^2] = 2, [V3] = 3.
G3SQ, LAM3, M3SQ, MU3US = 1.2675, 0.135, 1.27, 1.2675


# ----------------------------------------------------------------------
# the matching relations
# ----------------------------------------------------------------------
def test_the_ultrasoft_shifts_are_the_A0_cubic_expanded():
    """DRalgo's LO ultrasoft relations, rederived from the A_0 cubic.

    Integrating out A_0 at this order is the high-temperature expansion of
    -(1/12 pi)(m_D^2 + g3^2 phi3^2)^{3/2}. Its phi3^2 and phi3^4 coefficients
    give the mass and quartic shifts directly, with no reference to DRalgo.
    They agree only if lambda_VL = 2 g^2 T, which is what fixes its
    normalisation for us.
    """
    dral_m, expa_m, dral_l, expa_l = dr._check_lambda_VL_normalisation(0.65, 3.0)
    assert dral_m == pytest.approx(expa_m, rel=1e-12)
    assert dral_l == pytest.approx(expa_l, rel=1e-12)
    assert dral_m < 0.0 and dral_l < 0.0  # both shifts lower the barrier


def test_lambda_VL_is_two_g_squared_T_up_to_a_one_loop_correction():
    g, y, lam, T, mu = 0.65, 0.50, 0.045, 3.0, 2.0 * math.pi * 3.0
    value = dr.lambda_VL(g, y, lam, T, mu)
    leading = 2.0 * g**2 * T
    assert value == pytest.approx(leading, rel=0.05)


def test_integrating_out_A0_needs_a_real_Debye_mass():
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError):
            dr.lambda3_US(0.1, 0.5, bad)
        with pytest.raises(ValueError):
            dr.scalar_mass_sq_US(1.0, 0.5, bad)


def test_the_NLO_ultrasoft_mass_is_a_correction_not_a_replacement():
    """The NLO relation adds to the LO one and is parametrically smaller."""
    lo = dr.scalar_mass_sq_US(M3SQ, 0.85, 1.6)
    nlo = dr.scalar_mass_sq_US(M3SQ, 0.85, 1.6, order="NLO",
                               lambda_vll=0.09, mu3=1.27)
    assert nlo != lo
    assert abs(nlo - lo) < 0.1 * abs(lo)
    with pytest.raises(ValueError):
        dr.scalar_mass_sq_US(M3SQ, 0.85, 1.6, order="NLO")


# ----------------------------------------------------------------------
# the two-loop potential
# ----------------------------------------------------------------------
def _V2(phi, **kw):
    return dr.V3_US_two_loop(phi, M3SQ, LAM3, G3SQ, MU3US, **kw)


def test_the_scale_dependence_cancels_the_three_dimensional_running():
    """The sharpest check on the transcription, and it is parameter-free.

    In three dimensions the theory is super-renormalisable: g3^2 and lam3 do not
    run, and DRalgo's only 3D beta function is

        d msq3dUS/d ln mu3US = (g3^4 - 2 g3^2 lam3 + 2 lam3^2)/(4 pi^2).

    The tree potential contributes m3^2 phi3^2/2, so that running has to be
    cancelled by the explicit mu3US of the two-loop potential:

        d V2/d ln mu3US = -(phi3^2/2) d msq3dUS/d ln mu3US.

    A single misplaced coefficient anywhere in the fifty-odd terms breaks this.
    """
    beta_m3sq = (G3SQ**2 - 2.0 * G3SQ * LAM3 + 2.0 * LAM3**2) / (4.0 * math.pi**2)
    d = 1e-5
    for phi in (0.4, 1.0, 1.7, 3.0):
        hi = dr.V3_US_two_loop(phi, M3SQ, LAM3, G3SQ, MU3US * math.exp(d))
        lo = dr.V3_US_two_loop(phi, M3SQ, LAM3, G3SQ, MU3US / math.exp(d))
        numerical = (hi - lo) / (2.0 * d)
        expected = -0.5 * phi**2 * beta_m3sq
        assert numerical == pytest.approx(expected, rel=1e-6), f"at phi3={phi}"


def test_the_potential_has_mass_dimension_three():
    """Rescale every mass by s: V3 must scale as s^3.

    In 3D that means m3^2 -> s^2 m3^2, lam3 -> s lam3, g3^2 -> s g3^2,
    mu3US -> s mu3US and phi3 -> sqrt(s) phi3. Catches any wrong power of a
    mass in the transcription, including inside a logarithm.
    """
    phi, s = 1.7, 2.3
    plain = _V2(phi)
    scaled = dr.V3_US_two_loop(math.sqrt(s) * phi, s**2 * M3SQ, s * LAM3,
                               s * G3SQ, s * MU3US)
    assert scaled == pytest.approx(s**3 * plain, rel=1e-12)


def test_the_origin_is_subtracted_exactly():
    assert _V2(0.0) == 0.0
    assert _V2(0.0, subtract_origin=False) != 0.0


def test_the_small_field_limit_is_finite_and_approached_linearly():
    """Three groups carry 1/(4 phi3^2) with numerators that vanish as phi3^2.

    Their limits are finite but not zero -- group 1's is zero, groups 2 and 3
    are not -- and getting that wrong shifts the whole potential by a constant,
    which the subtraction at the origin would then fail to remove. Along a
    bounce the small-field region is the entire outer solution, so this is not
    a corner case.

    The approach to the limit is linear in phi3, not quadratic, because the
    transverse gauge mass sqrt(g3^2 phi3^2) is odd in phi3.
    """
    ratios = [_V2(phi) / phi**2 for phi in (1e-4, 1e-5, 1e-6, 1e-7)]
    assert all(math.isfinite(r) for r in ratios)
    steps = [abs(ratios[i + 1] - ratios[i]) for i in range(len(ratios) - 1)]
    for a, b in zip(steps, steps[1:]):
        assert b == pytest.approx(a / 10.0, rel=0.05)


def test_the_subtraction_constant_matches_the_numerical_limit():
    """V2(0) is coded from a sympy limit; check it against brute force.

    If the analytic limit were wrong by a constant, the subtracted potential
    would tend to that constant instead of to zero as phi3 -> 0.
    """
    assert _V2(1e-8) == pytest.approx(0.0, abs=1e-18)


def test_the_potential_is_infrared_divergent_for_a_massless_theory():
    """m3^2 = 0 at the origin leaves every logarithm with a vanishing argument."""
    with pytest.raises(ValueError, match="infrared divergent"):
        dr.V3_US_two_loop(0.0, 0.0, LAM3, G3SQ, MU3US)


def test_a_tachyonic_mass_raises_rather_than_going_complex():
    with pytest.raises(ValueError, match="strictly positive"):
        dr.V3_US_two_loop(0.1, -5.0, LAM3, G3SQ, MU3US, subtract_origin=False)


def test_arrays_and_scalars_both_work():
    grid = [0.5, 1.0, 1.5]
    out = _V2(grid)
    assert isinstance(out, list) and len(out) == 3
    assert out[1] == pytest.approx(_V2(1.0), rel=1e-14)


# ----------------------------------------------------------------------
# DRalgo's one-loop ultrasoft potential against our own leading-order cubic
# ----------------------------------------------------------------------
def test_DRalgo_confirms_our_transverse_gauge_cubic_and_the_absence_of_A0():
    """DRalgo run 2, "### EffectivePotentialUS NLO", is

        -(g3^2 phi3^2)^{3/2}/(6 pi)
        - (m3^2 +   lam3 phi3^2)^{3/2}/(12 pi)
        - (m3^2 + 3 lam3 phi3^2)^{3/2}/(12 pi)

    Two things follow, neither of them assumed anywhere else in this package.

    Its gauge term is exactly our `_gauge_cubic` with two transverse
    polarisations and no temporal mode, so DRalgo independently fixes the
    multiplicity we took from arXiv:2503.13597 eq. (4.2), independently of the
    check against BubbleDet's derivative expansion, which is a different
    statement.

    And there is no A_0 term at all, which is the definition of the ultrasoft
    theory and the reason this potential must not be combined with our
    leading-order one: ours keeps A_0 and its cubic is worth about half the
    transverse one again at these benchmarks.
    """
    from transitionlistener.eft_rate import abelian_higgs_3d

    phi = 1.7
    dralgo_gauge = -(G3SQ * phi**2)**1.5 / (6.0 * math.pi)
    ours = abelian_higgs_3d._gauge_cubic(phi, G3SQ, m_debye_sq=1.0,
                              n_transverse=2, n_temporal=0)
    assert float(ours) == pytest.approx(dralgo_gauge, rel=1e-13)

    # and switching A_0 back on is a large change, not a rounding one
    with_A0 = abelian_higgs_3d._gauge_cubic(phi, G3SQ, m_debye_sq=1.0,
                                 n_transverse=2, n_temporal=1)
    assert abs(float(with_A0) - dralgo_gauge) > 0.1 * abs(dralgo_gauge)


def test_the_two_loop_term_absorbs_the_scale_dependence_of_the_leading_order_action():
    """The strongest check available, because it exercises the whole chain.

    A next-to-leading-order term earns its place by making the answer less
    sensitive to the renormalisation scale, and here that is measurable. In
    three dimensions the theory is super-renormalisable: g3^2 and lam3 do not
    run at all and m3^2 is the only scale-dependent parameter, so varying mu3
    means running m3^2 with its beta function at the same time. Holding it
    fixed would measure the uncancelled half of a cancellation.

    Across mu3 varied by a factor of two either way, at sub-GeV benchmark A:
    the leading-order action moves by about eleven per cent, the total by about
    one. This exercises the bounce solver and the radial integral as well as the
    algebra, so unlike the analytic check above it would catch an error in how
    the potential is assembled rather than only in its coefficients.
    """
    from transitionlistener.eft_rate import abelian_higgs_3d

    K, G0, Y0, LAM0, V, TC = 7.0553, 0.677, 0.224, 0.0146238, 0.173, 0.034772
    T = 0.95 * TC
    mu4 = K * T
    g, y, lam, _ = dr.rg_run(G0, Y0, LAM0, V, mu4)
    lam3 = dr.lambda3_NLO(g, y, lam, T, mu4)
    g3sq = dr.g3_sq_LO(g, T)
    mD2 = dr.debye_mass_sq_LO(g, T)
    lvl = dr.lambda_VL(g, y, lam, T, mu4)
    lam3u = dr.lambda3_US(lam3, lvl, mD2)
    m3sq0 = dr.scalar_mass_sq_US(
        dr.scalar_thermal_mass_sq_LO(g, y, lam, T), lvl, mD2)

    mu0 = g3sq
    beta = (g3sq**2 - 2.0 * g3sq * lam3u + 2.0 * lam3u**2) / (4.0 * math.pi**2)

    lo, total = [], []
    for factor in (0.5, 1.0, 2.0):
        mu3 = factor * mu0
        m3sq = m3sq0 + beta * math.log(mu3 / mu0)
        S, r, phi3, _info = abelian_higgs_3d.solve_LO_bounce(
            m3sq, lam3u, g3sq, m_debye_sq=mD2, n_temporal=0)
        V2 = dr.V3_US_two_loop(list(phi3), m3sq, lam3u, g3sq, mu3)
        integral = 4.0 * math.pi * _radial_integral(r, phi3, V2)
        lo.append(S)
        total.append(S + integral)

    spread_lo = (max(lo) - min(lo)) / (sum(lo) / len(lo))
    spread_total = (max(total) - min(total)) / (sum(total) / len(total))
    assert spread_lo > 0.08                    # the LO action really does move
    assert spread_total < 0.02                 # and the total nearly does not
    assert spread_total < spread_lo / 5.0      # by an order of magnitude


def _radial_integral(r, phi3, values):
    import numpy as np
    return float(np.trapz(np.asarray(r)**2 * np.asarray(values), np.asarray(r)))


# ----------------------------------------------------------------------
# choosing between the two theories, per parameter point
# ----------------------------------------------------------------------

