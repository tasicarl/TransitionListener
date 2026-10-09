"""Support must not be manufactured where the nucleation rate has collapsed.

`_build_rate_step_grid` refines any interval across which `log10(Gamma/H^4)` changes by
more than `max_log10_rate_step = 12`. That is right on the hot side, where the rate *rises*
steeply and the percolation integral has to resolve the rise. It has no counterpart on the
fall: as the cold tail is walked, `log10(Gamma/H^4)` plunges without bound, so every
neighbouring pair exceeds any threshold and the refiner keeps inserting support into a
region where nothing can nucleate.

Measured at lambda3 = 5.8205024605: the solver walked to **T = 1.4e-5 GeV** with
`log10(Gamma/H^4) = -1.4e8`, and the rate-jitter diagnostic - one quadratic in `ln T`
fitted across the resulting seven decades of temperature - reported a residual of
**1.6e10 orders of magnitude** and advised increasing the bounce solver precision for a
sample at fourteen microelectronvolts. The data were not non-smooth; the fit had been asked
to describe a range no quadratic can.

The floor is relative to each profile's own rate peak, never absolute. Contributing samples
reach `log10(Gamma/H^4) = -447` in absolute terms, and a strongly supercooled model has its
peak at very low temperature by construction, so an absolute floor would truncate exactly
the models this code exists to study.

Part of TransitionListener v2
Documentation: https://tasillo.de/TransitionListener/
"""

from __future__ import annotations

import types

import numpy as np

from transitionlistener import bubbledynamics as bd
from transitionlistener import percolation_adaptive_gridbuilders as gridbuilders


def settings(**overrides):
    values = dict(max_log10_rate_step=12.0, n_action_increment=5, n_action_max=60,
                  f_start=1.0e-3, f_final=0.99, f_perc=0.28957, maxit=10,
                  large_delta_p_refine_threshold=0.1, large_delta_p_success_threshold=0.2)
    values.update(overrides)
    return types.SimpleNamespace(**values)


def live_rise():
    """A hot-side profile whose rate climbs steeply: the case the refinement exists for."""
    T = np.array([80.0, 70.0, 60.0, 50.0, 45.0])
    # log10(Gamma/H^4) rising from -60 to 0 in big steps
    target = np.array([-60.0, -40.0, -22.0, -8.0, 0.0])
    H = np.full(T.size, 2.5e-15)
    S = _actions_for(T, H, target)
    return T, S, H


def dead_collapse():
    """A cold tail whose rate has collapsed far below the peak, still stepping hugely."""
    T = np.array([40.0, 1.0, 1.0e-2, 1.0e-4, 1.4e-5])
    target = np.array([0.0, -4.0e3, -4.0e5, -6.0e7, -1.4e8])
    H = np.full(T.size, 1.0e-15)
    S = _actions_for(T, H, target)
    return T, S, H


def _actions_for(T, H, target_log10_rate):
    """Invert log10(Gamma/H^4) for S_3, so the fixtures use the code's own rate relation."""
    S = np.zeros_like(T)
    for i, (t, h, want) in enumerate(zip(T, H, target_log10_rate)):
        lo, hi = 1.0, 1.0e12
        for _ in range(200):                      # bisection on a monotone relation
            mid = 0.5 * (lo + hi)
            got = gridbuilders._log10_gamma_h4_array(
                np.array([t]), np.array([mid * t]), np.array([h]))[0]
            if not np.isfinite(got) or got > want:
                lo = mid
            else:
                hi = mid
        S[i] = 0.5 * (lo + hi) * t
    return S


class TestDeadRateFloor:

    def test_the_floor_sits_below_the_peak_not_at_an_absolute_value(self):
        """A supercooled model whose whole rate is tiny must keep its own scale."""
        tiny = np.array([-400.0, -402.0, -405.0])
        floor = gridbuilders._dead_rate_floor(tiny)
        assert floor < tiny.min(), (
            "the floor was placed above a profile whose entire rate is far below zero, "
            "which would declare a strongly supercooled model dead everywhere")
        assert np.isclose(floor, tiny.max() - gridbuilders._DEAD_RATE_DECADES_BELOW_PEAK)

    def test_an_empty_or_non_finite_rate_gives_no_floor(self):
        """Nothing is excluded when there is nothing to measure a peak against."""
        assert gridbuilders._dead_rate_floor(np.array([])) == -np.inf
        assert gridbuilders._dead_rate_floor(np.array([np.nan, np.inf])) == -np.inf

    def test_a_live_rise_is_still_refined(self):
        """The behaviour the refinement exists for must be untouched."""
        T, S, H = live_rise()
        grid = gridbuilders._build_rate_step_grid(T, S, H, settings(), max_new_points=10)
        assert grid is not None and np.asarray(grid).size > 0, (
            "the floor suppressed refinement of a steeply rising live rate, which is the "
            "unresolved-rise case the criterion is for")

    def test_a_collapsed_rate_is_not_refined(self):
        """The fix: a dead tail is not an unresolved rise, however large its steps."""
        T, S, H = dead_collapse()
        rate = gridbuilders._log10_gamma_h4_array(T, S, H)
        assert np.nanmax(np.abs(np.diff(rate))) > 12.0, (
            "the fixture does not exceed max_log10_rate_step, so it cannot exercise the bug")
        grid = gridbuilders._build_rate_step_grid(T, S, H, settings(), max_new_points=10)
        assert grid is None, (
            f"refinement was requested inside a collapsed tail (rates {rate}); this is the "
            "runaway that walked lambda3 = 5.8205024605 to T = 1.4e-5 GeV")

    def test_the_jitter_fit_ignores_the_dead_tail(self):
        """The symptom: a quadratic in ln T cannot span a dead tail, and must not try."""
        T = np.array([45.0, 42.0, 40.0, 38.0, 36.0, 34.0, 1.0e-4, 1.4e-5])
        H = np.full(T.size, 2.0e-15)
        target = np.array([2.0, 1.0, 0.0, -1.0, -2.0, -3.5, -6.0e7, -1.4e8])
        S = _actions_for(T, H, target)
        P = np.array([0.02, 0.05, 0.10, 0.18, 0.30, 0.45, 0.60, 0.70])
        diag = bd._action_rate_jitter_diagnostic(T, H, S, P)
        residual = diag.get("max_residual_oom")
        assert residual is None or not np.isfinite(residual) or residual < 10.0, (
            f"the diagnostic reported a residual of {residual} orders of magnitude, which "
            "measures the quadratic's inability to span seven decades of temperature "
            "rather than any instability in the bounce data")
