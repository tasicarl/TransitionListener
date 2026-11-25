"""The RG flow refuses a scale that is not a number instead of integrating for ever."""
import math
import signal

import pytest

from transitionlistener.eft_rate import rg


def _rhs(t, u):
    return [0.01 * u[0] ** 3]


@pytest.mark.parametrize("target", [math.nan, math.inf, -math.inf])
def test_a_non_finite_scale_raises_at_once(target):
    def stop(*_):
        raise TimeoutError("rg.flow did not return within 5 s")
    old = signal.signal(signal.SIGALRM, stop)
    signal.alarm(5)
    try:
        with pytest.raises(ValueError, match="finite scales"):
            rg.flow(("test_nan", target), _rhs, 0.0, [0.5], target, 1e-8, 1e-12)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def test_a_finite_scale_still_flows():
    u = rg.flow(("test_finite",), _rhs, 0.0, [0.5], -2.0, 1e-8, 1e-12)
    assert u is not None and math.isfinite(float(u[0]))
