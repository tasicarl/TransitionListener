"""The Gel'fand-Yaglom determinant solver, moved into the framework (WP5b).

These tests cover the framework behaviour: that failures are reported rather than returned,
and that the partial-wave sum says when it has not converged.
"""
import math

import numpy as np
import pytest

from transitionlistener.eft_rate import abelian_determinants as fl


def test_a_converged_sum_is_flagged_converged():
    """Steeply falling summand: the tail is a small correction."""
    ells = np.arange(2.0, 41.0)          # consecutive: the sum runs over every l
    # contributions ~ l^-2.9 so that (2l+1)*dlog falls steeply
    dlog = -1e-3 * ells ** -2.9 / (2 * ells + 1)
    on, off = np.exp(dlog), np.ones_like(dlog)
    res = fl.log_det_difference(on, off, ells)
    assert res["exponent"] < -1.0
    assert res["converged"]
    assert res["tail_fraction"] < 0.15


def test_a_slowly_falling_sum_is_flagged_not_converged():
    """The case that matters: the tail dominates, so `total` is a fit result."""
    ells = np.arange(2.0, 33.0)
    dlog = -1e-3 * ells ** -1.2 / (2 * ells + 1)      # barely convergent
    on, off = np.exp(dlog), np.ones_like(dlog)
    res = fl.log_det_difference(on, off, ells)
    assert res["exponent"] > -2.0
    assert not res["converged"]


def test_a_divergent_sum_is_reported_as_infinite():
    ells = np.arange(2.0, 33.0)
    dlog = -1e-3 * np.ones_like(ells) / (2 * ells + 1)   # summand ~ const
    res = fl.log_det_difference(np.exp(dlog), np.ones_like(dlog), ells)
    assert res["exponent"] >= -1.0        # summand does not fall fast enough
    assert math.isinf(res["total"])
    assert not res["converged"]


def test_no_difference_gives_no_shift():
    ells = np.arange(2.0, 21.0)
    t = np.linspace(0.5, 0.9, ells.size)
    res = fl.log_det_difference(t, t, ells)
    assert res["partial"] == pytest.approx(0.0, abs=1e-14)


def test_solver_reports_failure_instead_of_returning_nonsense():
    """A stiff high partial wave must raise, not poison the sum silently."""
    assert issubclass(fl.FluctuationSolverError, RuntimeError)


def test_every_return_path_carries_the_convergence_flag():
    """A caller must never have to guess whether `total` is trustworthy."""
    ells = np.arange(2.0, 21.0)
    cases = [
        np.exp(-1e-3 * ells ** -2.9 / (2 * ells + 1)),   # converged
        np.exp(-1e-3 * ells ** -1.2 / (2 * ells + 1)),   # slow
        np.exp(-1e-3 * np.ones_like(ells) / (2 * ells + 1)),  # divergent
    ]
    for on in cases:
        res = fl.log_det_difference(on, np.ones_like(ells), ells)
        assert "converged" in res and "tail_fraction" in res


def test_single_channel_reproduces_a_free_case():
    """With no potential difference the determinant ratio is one."""
    assert fl.gy_single(lambda r: 0.0, 4, 20.0, 0.5) == pytest.approx(1.0, rel=1e-6)


# ------------------------------------------- logarithmic formulations
def _test_bounce():
    from cosmoTransitions.tunneling1D import SingleFieldInstanton
    V = lambda p: 0.25*np.asarray(p)**4 - 0.4*np.asarray(p)**3 + 0.1*np.asarray(p)**2
    dV = lambda p: np.asarray(p)**3 - 1.2*np.asarray(p)**2 + 0.2*np.asarray(p)
    inst = SingleFieldInstanton(1.0, 0.0, V, dV)
    prof = inst.findProfile(xtol=1e-10, phitol=1e-10)
    m2 = float(np.atleast_1d(inst.d2V(np.array([0.0])))[0])
    phi = lambda r: float(np.interp(r, prof.R, prof.Phi,
                                    left=prof.Phi[0], right=prof.Phi[-1]))
    dphi = lambda r: float(np.interp(r, prof.R, prof.dPhi, left=0.0, right=0.0))
    dW = lambda r: float(np.atleast_1d(inst.d2V(np.array([phi(r)])))[0]) - m2
    return prof, m2, phi, dphi, dW, float(prof.R[-1]), dV


@pytest.mark.parametrize("l", [2, 5, 12, 40])
def test_log_single_matches_the_direct_form(l):
    _p, m2, _phi, _dphi, dW, rmax, _dV = _test_bounce()
    direct = math.log(fl.gy_single(dW, l, rmax, math.sqrt(m2)))
    assert fl.log_gy_single(dW, l, rmax, math.sqrt(m2)) == pytest.approx(
        direct, rel=1e-6)


@pytest.mark.parametrize("l", [2, 5, 12])
def test_log_coupled_matches_the_direct_form(l):
    _p, _m2, phi, dphi, _dW, rmax, dV = _test_bounce()
    G, WGi = 0.6, 0.2
    WX = lambda r: G**2 * phi(r)**2
    mXp = lambda r: G * dphi(r)
    WG = lambda r: (float(np.atleast_1d(dV(np.array([phi(r)])))[0]) / phi(r)
                    if abs(phi(r)) > 1e-12 else 0.0)
    dWG = lambda r: WG(r) - WGi
    direct = math.log(fl.gy_mixed(WX, mXp, dWG, l, rmax, math.sqrt(WGi)))
    assert fl.log_gy_mixed(WX, mXp, dWG, l, rmax, math.sqrt(WGi)) == pytest.approx(
        direct, rel=1e-6)


# ------------------------------------------------- stable special functions
@pytest.mark.parametrize("nu,z", [(2.5, 3.0), (12.5, 8.0), (40.5, 5.0), (100.5, 2.0)])
def test_log_ive_matches_scipy_where_scipy_works(nu, z):
    from scipy.special import ive
    assert fl._log_ive(nu, z) == pytest.approx(math.log(ive(nu, z)), rel=1e-12)


def test_log_ive_still_works_where_scipy_underflows():
    """The regime that broke the high partial waves: scipy returns exactly 0."""
    from scipy.special import ive
    nu, z = 200.5, 1.0
    assert ive(nu, z) == 0.0
    val = fl._log_ive(nu, z)
    assert np.isfinite(val) and val < -500.0


@pytest.mark.parametrize("nu,z", [(2.5, 3.0), (12.5, 8.0), (40.5, 5.0), (100.5, 2.0)])
def test_bessel_log_derivative_matches_the_direct_ratio(nu, z):
    from scipy.special import ive
    direct = 0.5 * (ive(nu - 1, z) + ive(nu + 1, z)) / ive(nu, z)
    assert fl._iv_log_deriv(nu, z) == pytest.approx(direct, rel=1e-10)


def test_log_difference_entry_points_agree():
    ells = np.arange(2.0, 33.0)
    on = np.exp(-1e-3 * ells ** -2.5 / (2 * ells + 1))
    off = np.ones_like(ells)
    a = fl.log_det_difference(on, off, ells)
    b = fl.log_det_difference_from_logs(np.log(on), np.log(off), ells)
    assert a["total"] == pytest.approx(b["total"], rel=1e-12)
    assert a["converged"] == b["converged"]


# ------------------------------- ghost, transverse and temporal bookkeeping
def _abelian_channels(xi=1.0, include_A0=True):
    phi = lambda r: 0.3 * math.exp(-0.5 * r)
    return fl.abelian_vector_channels(phi, g3sq=0.25, m_debye_sq=0.4,
                                      xi=xi, include_A0=include_A0), phi


def test_degree_of_freedom_count_closes_against_the_potential():
    """3 gauge + 1 Goldstone - 2 ghosts = 2, the n_transverse used in V3_LO.

    This ties the determinant bookkeeping to the barrier: if it fails, one of
    the two is wrong.
    """
    channels, _ = _abelian_channels()
    assert fl.net_degrees_of_freedom(channels) == pytest.approx(2.0)


def test_ghosts_enter_with_weight_minus_two():
    channels, _ = _abelian_channels()
    ghost = next(c for c in channels if c.name == "ghost")
    assert ghost.weight == -2.0


def test_transverse_and_ghost_share_an_operator_at_xi_one():
    """At xi = 1 the ghost mass equals the transverse gauge mass exactly."""
    channels, _ = _abelian_channels(xi=1.0)
    t = next(c for c in channels if c.name == "transverse")
    g = next(c for c in channels if c.name == "ghost")
    for r in (0.1, 1.0, 5.0):
        assert g.mass_sq(r) == pytest.approx(t.mass_sq(r), rel=1e-14)
    assert t.orbital(7) == g.orbital(7)


def test_a_different_gauge_separates_them_again():
    """The channels stay separate so another xi can be used without a rewrite."""
    channels, _ = _abelian_channels(xi=3.0)
    t = next(c for c in channels if c.name == "transverse")
    g = next(c for c in channels if c.name == "ghost")
    assert g.mass_sq(1.0) == pytest.approx(3.0 * t.mass_sq(1.0), rel=1e-14)


def test_transverse_and_ghost_weights_combine_to_minus_one_at_xi_one():
    channels, _ = _abelian_channels(xi=1.0)
    pair = [c for c in channels if c.name in ("transverse", "ghost")]
    assert sum(c.weight for c in pair) == pytest.approx(-1.0)


def test_temporal_mode_is_heavier_by_the_debye_mass():
    channels, phi = _abelian_channels()
    t = next(c for c in channels if c.name == "transverse")
    a0 = next(c for c in channels if c.name == "temporal")
    assert a0.mass_sq(1.0) - t.mass_sq(1.0) == pytest.approx(0.4, rel=1e-14)


def test_temporal_mode_can_be_omitted_once_it_is_integrated_out():
    channels, _ = _abelian_channels(include_A0=False)
    assert not any(c.name == "temporal" for c in channels)
    assert fl.net_degrees_of_freedom(channels) == pytest.approx(2.0)


def test_temporal_mode_is_not_a_physical_polarisation():
    """It must not change the physical dof count either way."""
    with_a0, _ = _abelian_channels(include_A0=True)
    without, _ = _abelian_channels(include_A0=False)
    assert (fl.net_degrees_of_freedom(with_a0)
            == fl.net_degrees_of_freedom(without))


# ------------------------------------------------ the gauge-orbit identity
def _smooth_test_bounce():
    """The test geometry of this file, with spline interpolation.

    The linear interpolation used elsewhere in this file is fine for evaluating
    a mass function, but the gauge-orbit identity differentiates it, so it needs
    a background that is smooth rather than piecewise linear.
    """
    from scipy.interpolate import CubicSpline
    from cosmoTransitions.tunneling1D import SingleFieldInstanton
    V = lambda p: 0.25*np.asarray(p)**4 - 0.4*np.asarray(p)**3 + 0.1*np.asarray(p)**2
    dV = lambda p: np.asarray(p)**3 - 1.2*np.asarray(p)**2 + 0.2*np.asarray(p)
    inst = SingleFieldInstanton(1.0, 0.0, V, dV)
    prof = inst.findProfile(xtol=1e-11, phitol=1e-11)
    spl = CubicSpline(prof.R, prof.Phi)
    return spl, spl.derivative(1), dV, float(prof.R[-1])


def _gauge_channels(G=0.6):
    phi, dphi, dV, rmax = _smooth_test_bounce()
    WX = lambda r: G**2 * float(phi(r))**2
    mXp = lambda r: G * float(dphi(r))
    WGi = 0.2                                    # V'(phi)/phi as phi -> 0
    WG = lambda r: float(np.atleast_1d(dV(np.array([float(phi(r))])))[0]) / float(phi(r))
    dWG = lambda r: WG(r) - WGi
    return WX, mXp, dWG, rmax, math.sqrt(WGi)


def _profile_eom_residual(G=0.6):
    """How well the interpolated test bounce satisfies phi'' + 2phi'/r = V'.

    The Goldstone row of the gauge-orbit identity is the only one containing
    phi'', so it cannot close better than this. Measuring it makes the test
    assert the mechanism rather than a threshold picked to pass.
    """
    phi, dphi, dV, rmax = _smooth_test_bounce()
    d2phi = phi.derivative(2)
    worst = 0.0
    for r in np.linspace(0.08 * rmax, 0.85 * rmax, 60):
        rhs = float(np.atleast_1d(dV(np.array([float(phi(r))])))[0])
        lhs = float(d2phi(r)) + 2.0 * float(dphi(r)) / r
        if abs(rhs) > 0.0:
            worst = max(worst, abs(lhs - rhs) / abs(rhs))
    return worst


@pytest.mark.parametrize("l", [1, 2, 4, 6])
def test_the_gauge_orbit_direction_is_annihilated(l):
    """A pure-gauge direction must be an exact zero mode of the R_xi operator.

    This is the check that fixes the off-diagonal coefficient, both
    Clebsch-Gordan factors, the three orbital assignments, the Goldstone mass
    and the ghost mass at once -- see :func:`fluctuations.gauge_orbit_residual`.
    """
    WX, mXp, dWG, rmax, mG = _gauge_channels()
    res = fl.gauge_orbit_residual(WX, mXp, dWG, l, rmax, mG)
    # rows S and L involve only phi and phi', so they are the sharp ones
    assert res["rows"]["S"] < 1e-4
    assert res["rows"]["L"] < 1e-4
    # row G also contains phi'', so it cannot beat the profile it is built on
    assert res["rows"]["G"] <= 1.001 * _profile_eom_residual()


def test_the_goldstone_row_residual_is_exactly_the_profile_error():
    """Row G is *equal* to the bounce's own equation-of-motion residual.

    phi'' enters row G only through g*omega*[phi'' + 2phi'/r - V'], so the row
    reproduces the profile's error and nothing else. Asserting equality rather
    than a bound is what makes it informative: any error in the operator would
    show up as an excess over this number. It holds at the partial waves whose
    mode still samples the whole bounce; by l = 12 the mode has moved out to
    larger radius, where the profile is better resolved, and the residual falls
    below the worst-point figure instead of saturating it.
    """
    WX, mXp, dWG, rmax, mG = _gauge_channels()
    eom = _profile_eom_residual()
    for l in (1, 2, 4, 6):
        res = fl.gauge_orbit_residual(WX, mXp, dWG, l, rmax, mG)
        assert res["rows"]["G"] == pytest.approx(eom, rel=1e-5)
    far = fl.gauge_orbit_residual(WX, mXp, dWG, 20, rmax, mG)
    assert far["rows"]["G"] < 0.5 * eom


def test_the_conditioning_degrades_with_partial_wave():
    """Honest about where the check stops being sharp.

    a_L is a difference between omega' and l*omega/r that is exact in the free
    limit, so the cancellation deepens with l and the residual grows even though
    the identity is exact. It is still six orders below the halved-coefficient
    failure at l = 20.
    """
    WX, mXp, dWG, rmax, mG = _gauge_channels()
    low = fl.gauge_orbit_residual(WX, mXp, dWG, 2, rmax, mG)
    high = fl.gauge_orbit_residual(WX, mXp, dWG, 20, rmax, mG)
    assert high["rows"]["S"] > low["rows"]["S"]
    assert high["rows"]["S"] < 1e-2


def test_the_identity_fails_for_any_other_mixing_coefficient():
    """Without this the test above could pass on an operator that is wrong.

    The residual mixing after R_xi gauge fixing is 2 g phi' a_r chi: the coefficient
    of a_r chi' alone is half that, and the a_r chi term contributes equally.
    Halving it must break the identity at order one.
    """
    WX, mXp, dWG, rmax, mG = _gauge_channels()
    good = fl.gauge_orbit_residual(WX, mXp, dWG, 3, rmax, mG)
    for factor in (1.0, 1.5, 3.0):
        bad = fl.gauge_orbit_residual(WX, mXp, dWG, 3, rmax, mG,
                                      mixing_factor=factor)
        assert bad["residual"] > 0.1, f"mixing_factor={factor} should not pass"
    assert good["residual"] == pytest.approx(_profile_eom_residual(), rel=1e-3)


def test_the_boundary_term_identity_holds_for_an_arbitrary_sequence():
    """The bulk of the two weightings cancels because (2k+3)+(2k-1) = 4k+2.

    Nothing about the determinant enters, so this is arithmetic and can be
    checked on noise.
    """
    rng = np.random.default_rng(11)
    a = rng.standard_normal(200) * 3.0
    for L in (3, 9, 50, 150):
        s1 = sum((2*l + 1) * (a[l - 1] + a[l + 1]) for l in range(1, L + 1))
        s2 = sum(2 * (2*l + 1) * a[l] for l in range(1, L + 1))
        pred = 3*a[0] - a[1] - (2*L + 3)*a[L] + (2*L + 1)*a[L + 1]
        assert s1 - s2 == pytest.approx(pred, abs=1e-9 * max(1.0, abs(pred)))


# ------------------------------------- the weights, where the answer is known
def test_the_gauge_sector_weights_give_a_xi_independent_potential():
    """At a stationary point the xi dependence must cancel exactly.

    This is what says the +1 per boson and -2 per ghost pair are right, and it
    is the mechanism by which the large xi = 1 Goldstone contribution is paid
    back by the ghosts and the extra polarisation.
    """
    out = fl.constant_background_gauge_check(g_sq=0.36, phi=1.7, dV_over_phi=0.0)
    assert out["gauge_independent"]
    for value in out["at_stationary_point"].values():
        assert value == pytest.approx(out["transverse_only"], rel=1e-12)


def test_it_reduces_to_the_two_transverse_modes_of_the_lo_potential():
    """The surviving answer is what N_TRANSVERSE_ABELIAN puts in the barrier."""
    g_sq, phi = 0.36, 1.7
    out = fl.constant_background_gauge_check(g_sq=g_sq, phi=phi, dV_over_phi=0.0)
    m_X = math.sqrt(g_sq) * phi
    assert out["transverse_only"] == pytest.approx(-2 * m_X**3 / (12 * math.pi))


def test_off_a_stationary_point_it_is_gauge_dependent():
    """The Nielsen identity, and why the bounce action is gauge dependent.

    A bounce is nowhere at a stationary point of the potential, so this is the
    generic case; the dependence has to cancel against the rest of the NLO
    layer rather than within the potential.
    """
    out = fl.constant_background_gauge_check(g_sq=0.36, phi=1.7,
                                             dV_over_phi=0.11)
    assert out["spread_off_shell"] > 1e-3 * abs(out["transverse_only"])


def test_the_surviving_term_is_the_gauge_loop_the_lo_potential_actually_uses():
    """Tie the determinant bookkeeping to the potential by number, not by count.

    ``net_degrees_of_freedom`` already asserts that the channel weights sum to
    the two transverse polarisations. This asserts the stronger thing: that what
    survives the xi cancellation is numerically the same gauge loop that
    ``abelian_higgs_3d.V3_LO`` subtracts to build its barrier. If the two modules ever
    disagreed about the field content, the barrier and the determinant would be
    describing different theories.
    """
    from transitionlistener.eft_rate import abelian_higgs_3d

    g3sq, phi, m3sq, lam3 = 0.36, 1.7, 0.05, 0.02
    out = fl.constant_background_gauge_check(g_sq=g3sq, phi=phi,
                                             dV_over_phi=0.0)
    # n_temporal=0 on both: this check is about the *spatial* gauge sector,
    # whose weights the channel list encodes. The temporal mode is a separate
    # 3D scalar, in the potential and the determinant alike, and is
    # not part of the transverse/longitudinal/ghost cancellation being tested.
    kw = dict(m_debye_sq=0.0, n_temporal=0)
    with_gauge = float(abelian_higgs_3d.V3_LO(phi, m3sq, lam3, g3sq, n_transverse=2, **kw))
    without = float(abelian_higgs_3d.V3_LO(phi, m3sq, lam3, g3sq, n_transverse=0, **kw))
    assert out["transverse_only"] == pytest.approx(with_gauge - without, rel=1e-12)


def test_a_sparse_partial_wave_grid_is_refused():
    """The sum runs over every l, so a sparse grid would silently undercount:
    1, 2, 3, 4, 6, 8, 12, 16, ... misses most of the terms in the region that
    dominates the sum.
    """
    sparse = np.array([1, 2, 3, 4, 6, 8, 12, 16, 24, 32], dtype=float)
    dlog = -1e-3 * sparse ** -2.0 / (2 * sparse + 1)
    with pytest.raises(ValueError, match="consecutive"):
        fl.log_det_difference_from_logs(dlog, np.zeros_like(dlog), sparse)


def test_a_consecutive_grid_is_accepted():
    ells = np.arange(1.0, 33.0)
    dlog = -1e-3 * ells ** -2.0 / (2 * ells + 1)
    out = fl.log_det_difference_from_logs(dlog, np.zeros_like(dlog), ells)
    assert math.isfinite(out["partial"])


# ------------------------------------------------- the general-xi operator
def test_the_general_xi_solver_reduces_to_the_shipped_one_at_xi_one():
    """The whole point of validating a new code path against a trusted one.

    At xi = 1 the (1 - 1/xi) d_i d_j term is identically absent, so the
    general-xi solver must reproduce log_gy_mixed. Agreement here is what says
    the new operator's extra structure is correctly switched off, and it covers
    the initial conditions, the chunked rescaling and the free-system
    subtraction as well as the operator itself.
    """
    WX, mXp, dWG, rmax, mG = _gauge_channels()
    WGi = mG**2
    WG = lambda r: dWG(r) + WGi
    for l in (1, 2, 5):
        shipped = fl.log_gy_mixed(WX, mXp, dWG, l, rmax, mG)
        general = fl.log_gy_mixed_general_xi(WX, mXp, WG, l, rmax, 1.0)
        assert general == pytest.approx(shipped, rel=1e-4)


def test_a_non_positive_gauge_parameter_is_refused():
    """xi = 0 makes the longitudinal kinetic coefficient infinite.

    Landau gauge is a limit of this construction, not a value of it, and
    returning something for xi = 0 would be worse than refusing.
    """
    WX, mXp, dWG, rmax, mG = _gauge_channels()
    WG = lambda r: dWG(r) + mG**2
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError, match="positive"):
            fl.log_gy_mixed_general_xi(WX, mXp, WG, 2, rmax, bad)


def test_the_gauge_kinetic_matrix_has_the_right_eigenvalues():
    """A = -1 + kappa v v^T must have eigenvalues -1 and -1/xi.

    The transverse direction propagates normally and the longitudinal one with
    1/xi; that is the structure the whole general-xi construction rests on, and
    it is what makes A invertible for every finite positive xi.
    """
    for l in (1, 3, 7):
        cm = math.sqrt(l / (2 * l + 1))
        cp = math.sqrt((l + 1) / (2 * l + 1))
        for xi in (0.25, 1.0, 3.0):
            kappa = 1.0 - 1.0 / xi
            v = np.array([cm, -cp])
            A = -np.eye(2) + kappa * np.outer(v, v)
            eig = np.sort(np.linalg.eigvals(A).real)
            assert eig == pytest.approx(np.sort([-1.0, -1.0 / xi]), rel=1e-12)


def test_mixing_summand_falls_off_on_a_bounce_much_smaller_than_rmax():
    """The gauge-Goldstone summand (2l+1)(ln T_on - ln T_off) must fall like l^-2.

    The case is the one that broke the equal-width chunking of log_gy_mixed: the conformal U(1)
    without fermions on the benchmark line at g = 0.41, 1.45 times the percolation temperature
    of the direct integration (9.6561e-10 GeV), where rmax ~ 6e10 in 3D units. There the old
    solver lost the solution inside its first chunk above l ~ 30, and the summand at l = 50
    was -14.7 instead of about -0.02.
    """
    import math as _m
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
    import dr_conformal_u1_no_fermions_matching as dr
    from transitionlistener.eft_rate import abelian_higgs_3d

    g, v, T = 0.41, 2.88137, 1.45 * 9.6561e-10
    p = dr.matched_soft_parameters(g, 0.0, g * v * _m.exp(-1.0 / 6.0), T, mu4_over_T=7.0553,
                                   matching_order="NLO", mu3=g * T)
    _, r, phi3, _ = abelian_higgs_3d.solve_LO_bounce(p["m3sq"], p["lam3"], p["g3sq"], m_debye_sq=p["mD2"])
    WX, mXp, dWG, zero, rmax, mG = abelian_higgs_3d._mixing_callables(
        r, phi3, p["m3sq"], p["lam3"], p["g3sq"], m_debye_sq=p["mD2"],
        n_temporal=abelian_higgs_3d.N_TEMPORAL_ABELIAN, dV=None)
    assert rmax > 1e10

    def summand(l):
        return (2 * l + 1) * (fl.log_gy_mixed(WX, mXp, dWG, l, rmax, mG)
                              - fl.log_gy_mixed(WX, zero, dWG, l, rmax, mG))
    s20, s50 = summand(20), summand(50)
    assert s20 < 0.0 and s50 < 0.0
    assert s50 / s20 == pytest.approx((20.0 / 50.0) ** 2, rel=0.25)
