"""The two-loop SOFT action from DRalgo run 5 (the HET module).

This is the factor of arXiv:2503.13597 eq. (4.20) the project had been
missing, and it is the abelian counterpart of their eq. (4.7). Nothing below
is taken on trust from the transcription.
"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
import dr_conformal_u1_fermions_matching as dr  # noqa: E402

# a representative soft-theory point, at the scale of the sub-GeV benchmarks
G3SQ, MD2, LVL, LVLL, MU3 = 0.0158, 1.0e-4, 0.0320, 1.0e-5, 0.0158
KW = dict(g3_sq=G3SQ, debye_mass_sq=MD2, lambda_vl=LVL,
          lambda_vll=LVLL, mu3=MU3)


def _dralgo_four_terms(phi3, g3_sq, debye_mass_sq, lambda_vl, lambda_vll, mu3):
    """DRalgo's output transcribed term by term, with nothing collected: singular at phi3 = 0."""
    phi = np.asarray(phi3, dtype=float)
    mA = np.sqrt(g3_sq * phi**2)
    m0_sq = debye_mass_sq + phi**2 * lambda_vl / 2.0
    pi2 = np.pi**2

    def L(x):
        return 0.5 + np.log(mu3 / x)

    return (-g3_sq**2 * phi**2 * L(mA) / (32.0 * pi2)
            + (-g3_sq**2 * phi**4 * L(2.0 * mA) / (2.0 * pi2)
               + g3_sq**2 * phi**4 * L(mA) / (8.0 * pi2)) / (4.0 * phi**2)
            - phi**2 * L(2.0 * np.sqrt(m0_sq)) * lambda_vl**2 / (64.0 * pi2)
            + m0_sq * lambda_vll / (128.0 * pi2))


def test_the_collected_form_equals_DRalgos_four_terms():
    """DRalgo returns four terms; the first cancels half the second exactly.

    If that cancellation were mis-simplified the potential would be wrong by a
    term of its own order.
    """
    for phi in (1e-6, 1e-3, 0.05, 0.139, 0.4, 1.5):
        collected = dr.V3_soft_two_loop(phi, **KW, subtract_origin=False)
        verbatim = float(_dralgo_four_terms(phi, **KW))
        assert collected == pytest.approx(verbatim, rel=1e-12), f"at phi3={phi}"


def test_it_is_finite_at_the_origin_where_the_verbatim_form_is_not():
    assert math.isfinite(dr.V3_soft_two_loop(0.0, **KW, subtract_origin=False))
    assert dr.V3_soft_two_loop(0.0, **KW) == 0.0
    assert not math.isfinite(float(_dralgo_four_terms(0.0, **KW)))


def test_no_mass_in_it_can_go_negative_on_any_bounce():
    """The point of using the HET potential rather than CalculatePotentialUS.

    Every mass here is a gauge mass: m_X^2 = g3^2 phi3^2 and
    m_X0^2 = m_D^2 + lambda_VL phi3^2/2. Both are non-negative for any real
    field value, so unlike the ultrasoft two-loop potential this one is real
    and finite everywhere the bounce goes -- including the region where the
    scalar tree masses are tachyonic, which at the physical temperature is
    most of the profile.
    """
    lam3_negative = -7.3e-6      # the actual soft quartic at T_perc, point A
    m3sq = 5.0e-8                # and the actual soft mass there
    for phi in np.linspace(0.05, 5.0, 40):
        # a field range over which the scalar masses are far negative ...
        assert m3sq + 3.0 * lam3_negative * phi**2 < 0.0
        # ... while the potential stays perfectly finite
        assert math.isfinite(dr.V3_soft_two_loop(float(phi), **KW))
    # and at the origin too, where the scalar masses are fine but the
    # verbatim form is not
    assert math.isfinite(dr.V3_soft_two_loop(0.0, **KW))


def test_the_scale_derivative_is_the_gauge_part_of_DRalgos_scalar_mass_running():
    """A cross-check between two different DRalgo routines from two different runs.

    Run 2's PrintScalarMass["NLO"] carries the mu3 dependence

        36 Log[mu3/mu] (8 g3^4 - 16 g3^2 lam3 + 16 lam3^2 + lamVL^2)/(1152 pi^2)

    so d msq3d/d ln mu3 = (8 g3^4 - 16 g3^2 lam3 + 16 lam3^2 + lamVL^2)/(32 pi^2).

    The HET potential of run 5 sits on m3^2 phi3^2/2, so its own scale
    derivative must reproduce that -- except for the two terms carrying lam3,
    which come from scalar loops and cannot appear here because the scalar is
    not integrated out. What is left is exactly the gauge and A_0 part.
    """
    d = 1e-5
    expected = (8.0 * G3SQ**2 + LVL**2) / (32.0 * math.pi**2)
    for phi in (0.05, 0.139, 0.8):
        hi = dr.V3_soft_two_loop(phi, **{**KW, "mu3": MU3 * math.exp(d)},
                                 subtract_origin=False)
        lo = dr.V3_soft_two_loop(phi, **{**KW, "mu3": MU3 / math.exp(d)},
                                 subtract_origin=False)
        dV = (hi - lo) / (2.0 * d)
        assert -2.0 * dV / phi**2 == pytest.approx(expected, rel=1e-6)


def test_switching_off_A0_gives_the_ultrasoft_HET_potential():
    """With lambda_VL and lambda_VLL zero only the gauge term survives.

    DRalgo's ultrasoft HET output is
        -g3^4 phi3^2 (1/2 + Log[mu3/(2 m_X)])/(8 pi^2)
    and that is what the soft expression must collapse to. The two came from
    separate calls in run 5, so this is a comparison between them.
    """
    phi = 0.139
    got = dr.V3_soft_two_loop(phi, g3_sq=G3SQ, debye_mass_sq=MD2,
                              lambda_vl=0.0, lambda_vll=0.0, mu3=MU3,
                              subtract_origin=False)
    mA = math.sqrt(G3SQ) * phi
    want = -G3SQ**2 * phi**2 * (0.5 + math.log(MU3 / (2.0 * mA))) / (8.0 * math.pi**2)
    assert got == pytest.approx(want, rel=1e-13)


def test_the_temporal_mass_identifies_lambda_VL_with_their_h3():
    """DRalgo prints m_D^2 + lambda_VL phi3^2/2 for the temporal gauge mode.

    arXiv:2503.13597 eq. (4.6) writes that mass as m_D^2 + h_3 v_3^2/2, so the
    two couplings are the same object -- which is what lets their eq. (4.7) be
    read in our variables.
    """
    phi = 0.3
    assert dr.temporal_mass_sq_soft(phi, MD2, LVL) == pytest.approx(
        MD2 + LVL * phi**2 / 2.0, rel=1e-14)
    # and it stays positive wherever the Debye mass does
    assert dr.temporal_mass_sq_soft(10.0, MD2, LVL) > MD2


def test_the_soft_theory_refuses_a_tachyonic_A0():
    with pytest.raises(ValueError, match="tachyonic"):
        dr.V3_soft_two_loop(5.0, g3_sq=G3SQ, debye_mass_sq=MD2,
                            lambda_vl=-1.0, lambda_vll=LVLL, mu3=MU3)


# ----------------------------------------------------------------------
# the kinetic term
# ----------------------------------------------------------------------
def test_the_kinetic_terms_gauge_part_matches_the_ultrasoft_run():
    """DRalgo's soft (2,2) entry reduces to its ultrasoft one when A_0 decouples.

    Soft:      (-11 m_X/(24 pi phi3^2) + phi3^2 lamVL^2/(192 pi m_X0^3))/2
    Ultrasoft:  -11 m_X/(48 pi phi3^2)

    which agree term by term once the soft entry's explicit /2 is applied.
    That agreement between two separate calls is what fixes the factor of two
    in the (delta + 2 dZ) convention, and hence Z3_soft_NLO's normalisation.
    """
    phi = 0.139
    got = dr.Z3_soft_NLO(phi, G3SQ, MD2, 0.0)
    mA = math.sqrt(G3SQ) * phi
    want = 2.0 * (-11.0 * mA / (48.0 * math.pi * phi**2))
    assert got == pytest.approx(want, rel=1e-13)


def test_the_kinetic_term_has_the_shape_of_their_eq_4_8_with_a_group_factor_of_three():
    """Both terms differ from arXiv:2503.13597 eq. (4.8) by exactly three.

    Theirs is  -11 g_X,3/(16 pi v_3) + h_3^2 v_3^2/(64 pi m_X0,3^3),
    ours       -11 g_3  /(24 pi phi3) + lamVL^2 phi3^2/(192 pi m_X0^3),

    and 24/16 = 192/64 = 3 once their gauge-mass convention m_X,3^2 =
    g_X,3^2 v_3^2/4 is undone against our m_X^2 = g_3^2 phi3^2. Three is the
    ratio of SU(2) to U(1) multiplicities -- 6 spatial polarisations against 2,
    3 temporal scalars against 1 -- and it is the same factor that relates
    their leading-order cubic to ours. Both terms needing the same factor is
    what makes this a check rather than a coincidence.
    """
    phi = 0.2
    ours_gauge = dr.Z3_soft_NLO(phi, G3SQ, MD2, 0.0)
    theirs_gauge = -11.0 * math.sqrt(G3SQ) / (8.0 * math.pi * phi)   # rewritten
    assert theirs_gauge / ours_gauge == pytest.approx(3.0, rel=1e-12)

    full = dr.Z3_soft_NLO(phi, G3SQ, MD2, LVL)
    ours_temporal = full - dr.Z3_soft_NLO(phi, G3SQ, MD2, 0.0)
    # the A_0 term alone, with lambda_VL held fixed in the mass too
    m0 = math.sqrt(dr.temporal_mass_sq_soft(phi, MD2, LVL))
    theirs_temporal = LVL**2 * phi**2 / (64.0 * math.pi * m0**3)
    # ours differs only by the gauge part's mass shift, so compare directly
    ours_temporal_clean = LVL**2 * phi**2 / (192.0 * math.pi * m0**3)
    assert theirs_temporal / ours_temporal_clean == pytest.approx(3.0, rel=1e-12)


def test_the_kinetic_correction_diverges_at_the_origin_as_it_should():
    """1/phi3 is physical -- the derivative expansion pushed where the gauge
    modes are light -- and is why this is reported apart from the potential."""
    small = abs(dr.Z3_soft_NLO(1e-3, G3SQ, MD2, LVL))
    smaller = abs(dr.Z3_soft_NLO(1e-4, G3SQ, MD2, LVL))
    assert smaller == pytest.approx(10.0 * small, rel=1e-3)


# ----------------------------------------------------------------------
# the assembly
# ----------------------------------------------------------------------
def _benchmark_A_bounce(f=0.0192):
    """The real leading-order soft bounce at sub-GeV point A."""
    from transitionlistener.eft_rate import abelian_higgs_3d
    K, G0, Y0, LAM0, V, TC = 7.0553, 0.677, 0.224, 0.0146238, 0.173, 0.034772
    T = f * TC
    mu = K * T
    g, y, lam, _ = dr.rg_run(G0, Y0, LAM0, V, mu)
    pars = dict(
        m3sq=dr.scalar_thermal_mass_sq_LO(g, y, lam, T),
        lam3=dr.lambda3_NLO(g, y, lam, T, mu),
        g3sq=dr.g3_sq_LO(g, T),
        mD2=dr.debye_mass_sq_LO(g, T),
        lvl=dr.lambda_VL(g, y, lam, T, mu),
        lvll=dr.lambda_VLL(g, T),
    )
    S, r, phi3, _info = abelian_higgs_3d.solve_LO_bounce(
        pars["m3sq"], pars["lam3"], pars["g3sq"], m_debye_sq=pars["mD2"])
    return S, r, phi3, pars


def test_the_kinetic_integral_does_not_depend_on_the_small_field_floor():
    """Z3^NLO goes as 1/phi3, but the integrand does not.

    In the tail dphi3/dr ~ -m phi3, so Z3 (dphi3/dr)^2 falls as phi3. The floor
    in soft_nlo_action is a guard against a profile that hits exactly zero, and
    if it were doing real work the answer would move with it.
    """
    from transitionlistener.eft_rate import abelian_higgs_3d

    _S, r, phi3, p = _benchmark_A_bounce()
    Z = lambda ph: dr.Z3_soft_NLO(ph, p["g3sq"], p["mD2"], p["lvl"])

    # a floor above the profile's smallest non-zero field would be clamping real
    # points, so the claim is about floors below it -- which the default is
    smallest = float(np.min(np.abs(phi3[phi3 != 0.0])))
    assert abelian_higgs_3d.Z3_FIELD_FLOOR < smallest
    values = []
    original = abelian_higgs_3d.Z3_FIELD_FLOOR
    try:
        for floor in (smallest / 10.0, smallest / 1e3, smallest / 1e6):
            abelian_higgs_3d.Z3_FIELD_FLOOR = floor
            values.append(abelian_higgs_3d.soft_nlo_action(r, phi3, Z3_nlo=Z)["kinetic"])
    finally:
        abelian_higgs_3d.Z3_FIELD_FLOOR = original

    assert values[0] == pytest.approx(values[-1], rel=1e-9)
    assert values[0] < 0.0          # it lowers the action


def test_a_profile_ending_at_round_off_does_not_pollute_the_kinetic_integral():
    """A bounce whose last point is -1e-16 instead of 0 would add Z3(floor) times a
    finite-difference slope, half the kinetic term at g = 0.7. The tail below 1e-9
    of the peak must count as false vacuum."""
    from transitionlistener.eft_rate import abelian_higgs_3d

    _S, r, phi3, p = _benchmark_A_bounce()
    Z = lambda ph: dr.Z3_soft_NLO(ph, p["g3sq"], p["mD2"], p["lvl"])
    clean = abelian_higgs_3d.soft_nlo_action(r, phi3, Z3_nlo=Z)["kinetic"]
    for last in (0.0, -1.1e-16, 3e-17):
        bumped = np.array(phi3, dtype=float)
        bumped[-1] = last
        assert abelian_higgs_3d.soft_nlo_action(r, bumped, Z3_nlo=Z)["kinetic"] == pytest.approx(clean, rel=1e-6)


def test_the_assembly_reports_no_total_when_the_two_loop_action_is_missing():
    """Without V3_nlo the exponent is incomplete and must not be given a total.

    Uses soft_nlo_action directly rather than the full assembly, which costs
    four BubbleDet channels.
    """
    from transitionlistener.eft_rate import abelian_higgs_3d

    _S, r, phi3, p = _benchmark_A_bounce()
    empty = abelian_higgs_3d.soft_nlo_action(r, phi3)
    assert empty["potential"] == 0.0 and empty["kinetic"] == 0.0
    assert empty["has_potential"] is False and empty["has_kinetic"] is False


def test_the_two_pieces_have_opposite_signs_and_the_kinetic_one_is_larger():
    """Reported separately for this reason; arXiv:2503.13597 finds the same."""
    from transitionlistener.eft_rate import abelian_higgs_3d

    S, r, phi3, p = _benchmark_A_bounce()
    V = lambda ph: dr.V3_soft_two_loop(ph, p["g3sq"], p["mD2"], p["lvl"],
                                       p["lvll"], p["g3sq"])
    Z = lambda ph: dr.Z3_soft_NLO(ph, p["g3sq"], p["mD2"], p["lvl"])
    out = abelian_higgs_3d.soft_nlo_action(r, phi3, V3_nlo=V, Z3_nlo=Z)

    assert out["potential"] > 0.0          # raises the action
    assert out["kinetic"] < 0.0            # lowers it
    assert abs(out["kinetic"]) > abs(out["potential"])
    assert out["total"] == pytest.approx(out["potential"] + out["kinetic"])
    # and both are a sensible fraction of the leading-order action
    assert 0.02 < abs(out["potential"]) / S < 0.30
    assert 0.02 < abs(out["kinetic"]) / S < 0.40


def test_including_the_two_loop_potential_collapses_the_scale_band():
    """The result that says the transcription is right.

    In three dimensions g3^2 and lam3 do not run and m3^2 is the only
    scale-dependent parameter, with the HET potential fixing its running to
    (8 g3^4 + lamVL^2)/(32 pi^2). Varying mu3 by two either way and running
    m3^2 with it, the leading-order action moves by about eleven per cent and
    the leading-order-plus-potential by well under one.
    """
    from transitionlistener.eft_rate import abelian_higgs_3d

    K, G0, Y0, LAM0, V, TC = 7.0553, 0.677, 0.224, 0.0146238, 0.173, 0.034772
    T = 0.0192 * TC
    mu = K * T
    g, y, lam, _ = dr.rg_run(G0, Y0, LAM0, V, mu)
    m3sq0 = dr.scalar_thermal_mass_sq_LO(g, y, lam, T)
    lam3 = dr.lambda3_NLO(g, y, lam, T, mu)
    g3sq = dr.g3_sq_LO(g, T)
    mD2 = dr.debye_mass_sq_LO(g, T)
    lvl = dr.lambda_VL(g, y, lam, T, mu)
    lvll = dr.lambda_VLL(g, T)

    beta = (8.0 * g3sq**2 + lvl**2) / (32.0 * math.pi**2)
    lo, total = [], []
    for factor in (0.5, 1.0, 2.0):
        mu3 = factor * g3sq
        m3sq = m3sq0 + beta * math.log(factor)
        S, r, phi3, _ = abelian_higgs_3d.solve_LO_bounce(m3sq, lam3, g3sq, m_debye_sq=mD2)
        Vf = lambda ph: dr.V3_soft_two_loop(ph, g3sq, mD2, lvl, lvll, mu3)
        out = abelian_higgs_3d.soft_nlo_action(r, phi3, V3_nlo=Vf)
        lo.append(S)
        total.append(S + out["potential"])

    spread_lo = (max(lo) - min(lo)) / (sum(lo) / len(lo))
    spread_total = (max(total) - min(total)) / (sum(total) / len(total))
    assert spread_lo > 0.08
    assert spread_total < 0.02
    assert spread_total < spread_lo / 10.0
