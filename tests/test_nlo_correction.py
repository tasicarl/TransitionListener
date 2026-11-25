"""The tabulated next-to-leading-order correction, and how the model uses it."""
import math
import os
import sys

import numpy as np
import pytest

from transitionlistener.eft_rate import one_field_rate_3d  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
from transitionlistener.eft_rate import abelian_higgs_3d  # noqa: E402


def _table(n=7, curvature=0.5):
    T = np.geomspace(1e-4, 3e-3, n)
    ref = 7e-4
    return T, 30.0 + 8.0 * np.log(T / ref) + curvature * np.log(T / ref) ** 2


#: The leading-order action of the tables that do not need a particular one.
S3 = 100.0


def test_the_table_is_reproduced_exactly():
    T, dS = _table()
    c = abelian_higgs_3d.NLOCorrection(T, dS, np.full_like(T, S3))
    for t, want in zip(T, dS):
        assert float(c(t, S3)) == pytest.approx(want, rel=1e-12)


def test_a_steep_monotone_table_stays_monotone():
    """A large-g table rises by a factor of ~80 over the last interval as the
    barrier edge is approached; the shape-preserving interpolant must not oscillate."""
    T = np.array([2.246e-3, 3.089e-3, 4.212e-3, 5.616e-3, 8.424e-3])
    dS = np.array([9.68, 11.28, 19.08, 47.32, 770.25])       # g = 1.0, measured
    S3 = np.array([68.61, 85.72, 113.34, 163.86, 526.54])
    c = abelian_higgs_3d.NLOCorrection(T, dS, S3_LO=S3)
    probe = np.geomspace(T[0], T[-1], 400)
    assert np.all(np.diff(c(probe, S3_LO=100.0)) >= -1e-9)


def test_the_continuation_outside_is_bounded_and_keeps_its_sign():
    """Outside, the correction is r_edge times the live action."""
    T = np.array([1.861e-3, 2.558e-3, 3.489e-3, 4.652e-3, 6.978e-3])
    dS = np.array([10.94, 12.57, 17.68, 31.75, 150.65])      # g = 0.9, measured
    S3 = np.array([83.62, 100.83, 126.14, 165.39, 310.88])
    c = abelian_higgs_3d.NLOCorrection(T, dS, S3_LO=S3)
    for Tq, live in ((1e-11, 20.0), (1e-6, 40.0), (1e-4, 60.0)):
        assert c(Tq, S3_LO=live) == pytest.approx(dS[0] / S3[0] * live)
    assert c(5e-2, S3_LO=1000.0) == pytest.approx(dS[-1] / S3[-1] * 1000.0)


def test_it_is_smooth_across_the_table_edges():
    """C^1 at both edges, checked numerically with a live action consistent
    with the table. The percolation solver treats a kink as jitter."""
    T = np.array([1.861e-3, 2.558e-3, 3.489e-3, 4.652e-3, 6.978e-3])
    dS = np.array([10.94, 12.57, 17.68, 31.75, 150.65])
    S3 = np.array([83.62, 100.83, 126.14, 165.39, 310.88])
    c = abelian_higgs_3d.NLOCorrection(T, dS, S3_LO=S3)
    x = np.log(T)
    h = 1e-6
    for k, (a, b) in ((0, (0, 1)), (-1, (-2, -1))):
        slope_s3 = (S3[b] - S3[a]) / (x[b] - x[a])
        live = lambda xx: S3[k] + slope_s3 * (xx - x[k])
        f = lambda xx: c(np.exp(xx), S3_LO=live(xx))
        assert abs(f(x[k] + h) - f(x[k] - h)) < 1e-3
        inner = (f(x[k] + 2 * h) - f(x[k] + h)) / h
        outer = (f(x[k] - h) - f(x[k] - 2 * h)) / h
        assert inner == pytest.approx(outer, rel=1e-4)


def test_it_says_when_it_is_extrapolating():
    T, dS = _table()
    c = abelian_higgs_3d.NLOCorrection(T, dS, np.full_like(T, S3))
    assert not c.extrapolating(T[0])
    assert not c.extrapolating(T[-1])
    assert c.extrapolating(0.5 * T[0])
    assert c.extrapolating(np.array([T[2], 2.0 * T[-1]]))


def test_it_refuses_a_table_too_small():
    with pytest.raises(ValueError, match="at least 4 points"):
        abelian_higgs_3d.NLOCorrection([1e-4, 2e-4, 3e-4], [1.0, 2.0, 3.0], [S3] * 3)


def test_it_refuses_nonsense_input():
    with pytest.raises(ValueError, match="positive"):
        abelian_higgs_3d.NLOCorrection([0.0, 1e-4, 2e-4, 3e-4], [1.0, 2.0, 3.0, 4.0], [S3] * 4)
    with pytest.raises(ValueError, match="same length"):
        abelian_higgs_3d.NLOCorrection([1e-4, 2e-4, 3e-4, 4e-4], [1.0, 2.0, 3.0], [S3] * 4)
    with pytest.raises(ValueError, match="S3_LO"):
        abelian_higgs_3d.NLOCorrection([1e-4, 2e-4, 3e-4, 4e-4], [1.0, 2.0, 3.0, 4.0], [S3, S3, S3, 0.0])


def test_unsorted_input_is_sorted_not_rejected():
    T, dS = _table()
    order = np.array([3, 0, 5, 1, 6, 2, 4])
    c = abelian_higgs_3d.NLOCorrection(T[order], dS[order], (S3 * T / T[0])[order])
    assert np.all(np.diff(c.T_GeV) > 0)
    assert c.S3_LO == pytest.approx(S3 * T / T[0])
    for t, want in zip(T, dS):
        assert float(c(t, S3)) == pytest.approx(want, rel=1e-12)


# ----------------------------------------------------------------------
# how the model uses it
# ----------------------------------------------------------------------
def test_the_model_refuses_to_run_with_determinants_but_no_table():
    """Silently returning the leading-order action would be indistinguishable
    from the correction being zero, which is the one thing it must not be."""
    import TL_conformal_u1_fermions_3d as m3d

    pot = m3d.specific_potential({"g": 0.677, "y": 0.224, "v_GeV": 0.173},
                                 verbose=False)
    pot.config.nucleationConf.rate_mode = "model"
    pot.config.nucleationConf.include_determinants = True
    assert pot.nlo_correction is None
    # with a grid the model takes it (tests/test_nlo_tables.py); without one it stops
    pot.config.nucleationConf.nlo_grid_file = "/nonexistent/grid.npz"
    with pytest.raises(ValueError, match="no NLO grid at"):
        pot.actionNuc(0.02, 1.0)


def test_attaching_a_table_shifts_the_action_by_exactly_its_value():
    import TL_conformal_u1_fermions_3d as m3d

    pot = m3d.specific_potential({"g": 0.677, "y": 0.224, "v_GeV": 0.173},
                                 verbose=False)
    pot.config.nucleationConf.rate_mode = "model"
    pot.config.nucleationConf.include_determinants = False

    T_internal = 0.02
    T_GeV = T_internal * pot.conversionFactor
    lo = pot.actionNuc(T_internal, 1.0)

    T, dS = _table()
    # a table spanning the temperature actually being queried
    T = np.geomspace(0.3 * T_GeV, 3.0 * T_GeV, 7)
    dS = np.full_like(T, 11.0)
    pot.nlo_correction = abelian_higgs_3d.NLOCorrection(T, dS, np.full_like(T, S3))
    pot.config.nucleationConf.include_determinants = True
    nlo = pot.actionNuc(T_internal, 1.0)

    # actionNuc returns T * S3/T, so the shift is T * dS
    assert (nlo - lo) == pytest.approx(T_internal * 11.0, rel=1e-6)


def test_the_default_is_unchanged_by_all_of_this():
    """include_determinants stays False, so nothing above is on by accident."""
    from transitionlistener.config import Configuration

    assert Configuration().nucleationConf.include_determinants is False


# ----------------------------------------------------------------------
# the barrier edge: solve_LO_bounce's "no barrier" contract
# ----------------------------------------------------------------------
def _edge_parameters(T_over_Tc):
    """Matched parameters for point A at mu4 = 2 x central, LO matching --
    the configuration whose band run died at the barrier edge."""
    import dr_conformal_u1_fermions_matching as dr
    T = T_over_Tc * 0.034772
    mu = 2.0 * 7.0553 * T
    g, y, lam, _ = dr.rg_run(0.677, 0.224, 0.0146238, 0.173, mu)
    return (dr.scalar_thermal_mass_sq_LO(g, y, lam, T),
            dr.lambda3_NLO(g, y, lam, T, mu),
            dr.g3_sq_LO(g, T), dr.debye_mass_sq_LO(g, T))


def test_the_marginal_edge_is_reported_as_no_barrier_not_as_a_crash():
    """At T/Tc = 0.26881 the analytic check finds a barrier and the shooting
    solver finds the potential too flat. Measured, not constructed: this point
    raised TooSmalldVError before the fix, which the model did not catch."""
    m3sq, lam3, g3sq, mD2 = _edge_parameters(0.26881)
    with pytest.raises(ValueError, match="too flat|no barrier"):
        abelian_higgs_3d.solve_LO_bounce(m3sq, lam3, g3sq, m_debye_sq=mD2)


def test_the_neighbouring_points_are_unaffected():
    """Just inside the edge the bounce still solves, with the huge action of
    the thin-wall limit; just outside, the analytic check already says no."""
    m3sq, lam3, g3sq, mD2 = _edge_parameters(0.2685)
    S = abelian_higgs_3d.solve_LO_bounce(m3sq, lam3, g3sq, m_debye_sq=mD2)[0]
    assert S > 1e5
    m3sq, lam3, g3sq, mD2 = _edge_parameters(0.2870)
    with pytest.raises(ValueError, match="no barrier"):
        abelian_higgs_3d.solve_LO_bounce(m3sq, lam3, g3sq, m_debye_sq=mD2)


@pytest.mark.parametrize("exc_args, translated", [
    (("top outside the interval", "no barrier"), True),
    (("V(meta) <= V(abs)", "stable, not metastable"), True),
    (("something else entirely", "some other kind"), False),
])
def test_only_the_no_barrier_family_is_translated(monkeypatch, exc_args, translated):
    """The shooting solver's "no barrier" and "stable" verdicts become
    ValueError; any other PotentialError must still propagate, or a real bug
    would be silently read as a vanishing rate."""
    from transitionlistener import errors, tunneling1D

    class Refusing:
        def __init__(self, *a, **k):
            raise errors.PotentialError(*exc_args)

    monkeypatch.setattr(tunneling1D, "ClosedFormInstanton", Refusing)
    m3sq, lam3, g3sq, mD2 = _edge_parameters(0.0192)
    if translated:
        with pytest.raises(ValueError):
            abelian_higgs_3d.solve_LO_bounce(m3sq, lam3, g3sq, m_debye_sq=mD2)
    else:
        with pytest.raises(errors.PotentialError):
            abelian_higgs_3d.solve_LO_bounce(m3sq, lam3, g3sq, m_debye_sq=mD2)


def test_the_model_reports_the_capped_action_at_the_edge():
    """End to end through actionNuc: the edge now gives T * ACTION_CAP, the
    same continuation as beyond the barrier, instead of aborting the run."""
    import TL_conformal_u1_fermions_3d as m3d
    pot = m3d.specific_potential({"g": 0.677, "y": 0.224, "v_GeV": 0.173},
                                 verbose=False)
    pot.config.nucleationConf.rate_mode = "model"
    pot.config.nucleationConf.mu4_over_T = 2.0 * 7.0553
    # the leading-order matching of the edge has no matched h3: the temporal mass at its leading-order value
    pot.config.nucleationConf.temporal_coupling = "2g3sq"
    pot.soft_parameters = lambda T_GeV: dict(zip(("m3sq", "lam3", "g3sq", "mD2"), _edge_parameters(0.26881)))
    T_internal = 0.26881 * 0.034772 / pot.conversionFactor
    assert pot.actionNuc(T_internal, 1.0) == pytest.approx(
        T_internal * one_field_rate_3d.ACTION_CAP)


def test_a_temperature_the_couplings_cannot_be_run_to_gets_the_capped_action():
    """The adaptive percolation scan's step-2 grid can reach T ~ 1e-13 GeV. At
    g = 0.9 running the couplings from v down to mu ~ 1e-12 GeV hits a pole in
    the quartic; that used to escape actionNuc and kill the run."""
    import TL_conformal_u1_fermions_3d as m3d
    pot = m3d.specific_potential({"g": 0.9, "y": 0.01, "v_GeV": 0.173}, verbose=False)
    pot.config.nucleationConf.rate_mode = "model"
    T_internal = 2.3525e-13 / pot.conversionFactor
    assert pot.actionNuc(T_internal, 1.0) == pytest.approx(T_internal * one_field_rate_3d.ACTION_CAP)
