"""The nucleation temperature of the rate history, where the number of bubbles runs over many decades."""
import math

import numpy as np
import pytest

from transitionlistener import bubbledynamics as bd


@pytest.mark.parametrize("cold_log_rate", [10.0, 60.0, 300.0])
def test_the_crossing_does_not_depend_on_how_large_the_integral_grows_below_it(monkeypatch, cold_log_rate):
    """ln(Gamma/H^4) = c - k u has N(u) = (4 pi/3) exp(c - k u)/k, one at the temperature asked for."""
    k = 40.0
    T = np.geomspace(1e4, 1.0, 1500)
    monkeypatch.setattr(bd, "logGamma", lambda t, s: cold_log_rate - k * np.log(t))
    pref = 4.0 * math.pi / 3.0
    expected = math.exp((cold_log_rate + math.log(pref / k)) / k)
    found = bd.estimate_spline_tnuc_from_rate_history(T, np.ones_like(T), np.ones_like(T), np.zeros_like(T))
    assert found["warning"] is None
    assert found["Tnuc"] == pytest.approx(expected, rel=1e-5)
