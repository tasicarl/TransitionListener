"""The nucleation temperature of the integrated bubble count, at both ends of the rate history.

Every test here drives ``bubbledynamics.estimate_spline_tnuc_from_rate_history``, which solves
``N(T) = 4*pi/3 * int_T^Tinit Gamma/H^4 * (1-P) dT/T = 1`` on the saved rate history, on a rate
whose closed form is known. Temperatures carry the units of the history they are given in, so
they are written here without one, and ``k`` is a slope per unit ``ln T``.
"""
import math

import numpy as np
import pytest

from transitionlistener import bubbledynamics as bd

PREFACTOR = 4.0 * math.pi / 3.0


def _flat_history(size):
    """A rate history with H = 1, S = 1 and no true vacuum, so that N depends on the rate alone."""
    return np.ones(size), np.ones(size), np.zeros(size)


@pytest.mark.parametrize("cold_log_rate", [10.0, 60.0, 300.0])
def test_the_crossing_does_not_depend_on_how_large_the_integral_grows_below_it(monkeypatch, cold_log_rate):
    """ln(Gamma/H^4) = c - k ln(T) has N(T) = (4 pi/3) exp(c) T^-k / k, one at the temperature asked for.

    The three values of c put the bubble count at the cold end of the history at 2.3e3, 1.2e25
    and 2.0e129, while the crossing of N = 1 stays inside the history. A count accumulated from the
    cold end and differenced loses the crossing to the rounding of the total.
    """
    k = 40.0
    T = np.geomspace(1e4, 1.0, 1500)
    monkeypatch.setattr(bd, "logGamma", lambda t, s: cold_log_rate - k * np.log(t))
    expected = math.exp((cold_log_rate + math.log(PREFACTOR / k)) / k)
    found = bd.estimate_spline_tnuc_from_rate_history(T, *_flat_history(T.size))
    assert found["warning"] is None
    assert found["Tnuc"] == pytest.approx(expected, rel=1e-5)


def test_each_cell_of_the_dense_grid_is_integrated_as_the_exponential_it_is(monkeypatch):
    """A cell wide enough in ln(Gamma/H^4) that the trapezoid of an exponential is visibly wrong.

    50 support temperatures over 650 e-folds of rate give a dense grid of 1000 cells and a step
    of 0.6507 in ln(Gamma/H^4) per cell. The trapezoid of an exponential over a cell that wide is
    1.035032 times the exact integral, so it reaches N = 1 at a temperature 4.880e-04 higher in
    relative terms, 49 times the tolerance asserted here.
    """
    k = 650.0 / math.log(1e4)
    c = 350.0
    T = np.geomspace(1e4, 1.0, 50)
    monkeypatch.setattr(bd, "logGamma", lambda t, s: c - k * np.log(t))
    expected = math.exp((c + math.log(PREFACTOR / k)) / k)
    found = bd.estimate_spline_tnuc_from_rate_history(T, *_flat_history(T.size))
    assert found["warning"] is None
    assert found["Tnuc"] == pytest.approx(expected, rel=1e-5)


def test_a_cell_that_rises_by_more_than_the_exponent_can_hold_is_still_counted(monkeypatch):
    """A cell whose log rate rises by far more than 700 must not be counted as exp(700).

    Twelve support temperatures over 4000 e-folds of rate give a step of about 4 in the log
    integrand per dense cell at the dense grid, and the cells below the crossing rise by
    thousands in total. Clipping the exponent of a cell at 700 undercounts every cell below
    that rise by the factor it was clipped by, which moves the temperature at which the
    count reaches one. The closed form fixes where that is: N(T) = (4 pi/3) exp(c) T^-k / k.
    """
    k = 4000.0 / math.log(1e4)
    c = 2000.0
    T = np.geomspace(1e4, 1.0, 12)
    monkeypatch.setattr(bd, "logGamma", lambda t, s: c - k * np.log(t))
    expected = math.exp((c + math.log(PREFACTOR / k)) / k)
    found = bd.estimate_spline_tnuc_from_rate_history(T, *_flat_history(T.size))
    assert found["warning"] is None
    assert found["Tnuc"] == pytest.approx(expected, rel=1e-5)


def test_a_cell_beyond_the_range_of_a_double_refuses_rather_than_answers():
    """Where one cell exceeds what a double can hold, the routine says so and returns nothing.

    The cell integral is written in the logarithm, so the clipping that used to undercount a
    steep cell is gone, but the cell is still exponentiated once at the end and a cell whose
    log integrand spans more than about 709 overflows. Measured: at a slope of 1e+04 per
    unit ln(T), which is 92 per dense cell, the crossing is found exactly; at 1e+05, which
    is 920 per dense cell, the count collapses to 1.9e-24 and the routine reports
    `spline_tnuc_not_reached` instead of a temperature. The steepest cell of the conformal
    benchmark in the documentation spans 57.6, so this bound is twelve times beyond anything
    the shipped models reach, and what matters is that it refuses rather than answers.
    """
    PREFACTOR_LOCAL = PREFACTOR
    T = np.geomspace(1e4, 1.0, 12)
    bd_rate = bd.logGamma
    try:
        for slope, reachable in ((1.0e4, True), (1.0e5, False)):
            offset = math.log(slope) + slope * 4.6
            bd.logGamma = lambda t, s, c=offset, k=slope: c - k * np.log(t)
            found = bd.estimate_spline_tnuc_from_rate_history(
                T, np.ones(T.size), np.ones(T.size), np.zeros(T.size))
            if reachable:
                expected = math.exp((offset + math.log(PREFACTOR_LOCAL / slope)) / slope)
                assert found["warning"] is None
                assert found["Tnuc"] == pytest.approx(expected, rel=1e-9)
            else:
                assert found["Tnuc"] is None
                assert found["warning"] == "spline_tnuc_not_reached"
    finally:
        bd.logGamma = bd_rate


def test_a_true_vacuum_fraction_that_rounds_to_one_does_not_discard_its_cell():
    """`log1p(-1)` is minus infinity, and the cell beside it still carries a contribution.

    The percolation integral stops at a true-vacuum probability of 1 - exp(-50), which is
    exactly 1 in double precision. The logarithm of the false vacuum left over is then minus
    infinity at that node, so the cell between it and its neighbour has no exponential model
    and came out as a non-number, which the zeroing of non-finite cells dropped. That cell
    is finite and not zero, and it is integrated in ordinary space instead.

    The fixture puts the probability jump at the cold end, where the integrand is largest,
    which is the placement that makes the dropped cell worth the most. Even there it is
    1.6e-06 of the count: this is a correctness fix of a small quantity, not a large
    correction. The value below is a numpy computation on an analytic rate, with no bounce
    solver in it, and is identical on the two processor models of this machine.
    """
    k = 40.0
    c = 100.0
    T = np.geomspace(1e4, 1.0, 1500)
    bd_rate = bd.logGamma
    try:
        bd.logGamma = lambda t, s: c - k * np.log(t)
        probability = np.where(T < 1.5, 1.0, 0.0)
        assert np.any(probability == 1.0), "the fixture does not reach a probability of one"
        found = bd.estimate_spline_tnuc_from_rate_history(
            T, np.ones(T.size), np.ones(T.size), probability)
    finally:
        bd.logGamma = bd_rate
    assert found["warning"] is None
    assert math.isfinite(found["max_N"])
    # Dropping the cell gives 2.8753756862e+35, lower by 1.6e-06 in relative terms.
    assert found["max_N"] == pytest.approx(2.875380281729e35, rel=1e-7)


@pytest.mark.parametrize("hot_end_slope", [-1.0e-3, -1.0e-6, -1.0e-9, 0.0])
def test_a_flat_rate_at_the_hot_end_counts_no_bubbles_the_history_does_not_support(monkeypatch, hot_end_slope):
    """The count above the hottest temperature of the history must not grow as that slope goes to zero.

    The history has ln(Gamma/H^4) = -20 above ln(T) = 2.3, with the given slope there, and rising
    by 40 per unit ln(T) below. The crossing of N = 1 sits at T = 5.7178, from the steep arm
    alone. Integrating the rate above the hottest node to infinite temperature gives
    (4 pi/3) exp(-20)/(-slope), which reaches 8.6 for a slope of -1e-9 and reports the hottest
    temperature of the history as the nucleation temperature instead.

    A slope of exactly zero is in the list because it is the series limit of the same
    expression and not an absent contribution. Excluding it from the branch dropped the whole
    rectangle, `exp(ln f) * u_above`, which is the size of the terms being kept, so the count
    above the hottest node read as nothing at all.
    """
    knee, cold_slope, plateau = 2.3, 40.0, -20.0
    T = np.geomspace(1e2, 1.0, 400)

    def log_rate(temperatures, action):
        u = np.log(np.atleast_1d(temperatures))
        return np.where(u > knee,
                        plateau + hot_end_slope * (u - knee),
                        plateau - cold_slope * (u - knee))

    monkeypatch.setattr(bd, "logGamma", log_rate)
    expected = math.exp(knee - (-plateau + math.log(cold_slope / PREFACTOR)) / cold_slope)
    found = bd.estimate_spline_tnuc_from_rate_history(T, *_flat_history(T.size))
    assert found["warning"] is None
    assert found["Tnuc"] == pytest.approx(expected, rel=1e-6)
    assert found["Tnuc"] < T.max()


@pytest.mark.parametrize("hot_end_slope", [-1.0e-9, 0.0])
def test_a_count_already_above_one_at_the_hottest_temperature_is_refused(monkeypatch, hot_end_slope):
    """Where the crossing lies above every support point, the history cannot say where it is.

    With ln(Gamma/H^4) = -1 over the whole history, the bubbles counted above the hottest
    temperature already number 7.096, so N = 1 has no bracket inside the history. Reporting the
    hottest temperature of the history as the nucleation temperature would be an answer the rate
    history does not contain.

    The slope of exactly zero is the case that fails when the extrapolation above the hottest
    node excludes a zero slope instead of taking its series limit: the term is then 0 rather
    than 7.096, the count no longer exceeds one at the hot end, and a temperature is reported
    from a bracket the history does not have.
    """
    T = np.geomspace(1e2, 1.0, 400)
    monkeypatch.setattr(bd, "logGamma", lambda t, s: -1.0 + hot_end_slope * np.log(t))
    found = bd.estimate_spline_tnuc_from_rate_history(T, *_flat_history(T.size))
    assert found["Tnuc"] is None
    assert found["warning"] == "spline_tnuc_failed"
    assert found["max_N"] > 1.0
    assert np.isfinite(found["max_N"])
