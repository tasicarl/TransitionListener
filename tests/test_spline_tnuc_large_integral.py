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
