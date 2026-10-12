"""A single RH sample cannot certify its own convergence.

`_dynamiczoomwindow_accuracy_reached` reports a profile as having met its targets when
`RH_uncertainty <= acc_rh`, with `acc_rh = 1e-2`, i.e. a claim that RH is good to 1%. The
uncertainty was reported as **0.0** whenever there was no previous RH to compare against, so
the first estimate declared itself converged to zero per cent and the 1% target was never
tested.

What this does *not* mean is that acceptance was short-circuited. Both exits of
`_try_dynamiczoomwindow_completed_refinement` return `False`, which the caller reads as
"accept", so the targets never prevented acceptance in the first place: they are advisory,
and the profile is accepted either way because by that point refinement has nothing left to
try. What was wrong is that an advisory target reported itself as met when it had not been
measured.

Measured consequence: across 37 couplings that two accepted profiles both resolve, RH differs
by a median of 0.134% but a 90th percentile of 27.6% and a worst case of 95%, while Tperc
agrees to better than 1% throughout. Both profiles claimed to meet acc_rh.

Tperc and Tf never had this hole: each falls back to its own bracket width when there is no
previous value. RH has no bracket measure, so with nothing to compare against the honest
answer is "unknown", which must not satisfy the target.

Part of TransitionListener v2
Documentation: https://tasillo.de/TransitionListener/
"""

from __future__ import annotations

import math
import types

import numpy as np

from transitionlistener import percolation_adaptivestepsize as adaptive


def settings(**over):
    values = dict(acc_tperc=1e-2, acc_tfinal=1e-2, acc_rh=1e-2)
    values.update(over)
    return types.SimpleNamespace(**values)


def estimate(rh_uncertainty):
    return {"Tperc": 40.0, "Tf": 35.0, "RH": 0.1,
            "Tperc_uncertainty": 1e-4, "Tf_uncertainty": 1e-4,
            "RH_uncertainty": rh_uncertainty}


def test_a_converged_rh_still_passes():
    """Control: a genuinely small RH uncertainty must still satisfy the target."""
    assert adaptive._dynamiczoomwindow_accuracy_reached(estimate(1e-3), settings())


def test_an_rh_uncertainty_above_the_target_fails():
    """Control: the gate works when the uncertainty is known and too large."""
    assert not adaptive._dynamiczoomwindow_accuracy_reached(estimate(0.3), settings())


def test_an_unknown_rh_uncertainty_does_not_satisfy_the_target():
    """The fix: 'nothing to compare against' must not read as 'perfectly converged'."""
    assert not adaptive._dynamiczoomwindow_accuracy_reached(estimate(math.inf), settings()), (
        "an unknown RH uncertainty satisfied acc_rh, so a profile can be accepted on a "
        "single RH sample that was never checked against anything")


def test_the_first_estimate_reports_an_unknown_uncertainty():
    """The source of it: with no previous RH, the estimate must not claim 0.0."""
    T = np.array([45.0, 42.0, 40.0, 38.0, 36.0])
    P = np.array([0.02, 0.10, 0.29, 0.60, 0.95])
    S = (150.0 + 0.5 * (T - 36.0)) * T
    H = np.full(T.size, 2.5e-15)
    out = adaptive._estimate_dynamiczoomwindow_accuracy(
        T, H, P, S,
        settings(f_perc=0.28957, f_final=0.99, vw=1.0, integral_method="ode",
                 time_temperature_mode="bag", entropy_definition=None),
        None, previous=None)
    unc = out.get("RH_uncertainty")
    assert unc is None or not np.isfinite(unc) or unc > 0.0, (
        f"the first estimate reported RH_uncertainty = {unc}, i.e. it certified a quantity "
        "it had no second sample for")
