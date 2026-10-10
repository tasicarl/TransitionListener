"""One bad bounce solve must cost one support point, not the whole point.

The rate-jitter gate rejects a profile when a single action sample sits more than
`jitter_GH4_threshold` orders of magnitude off the local smooth fit. The rescue recomputes
that sample with the deformation tightened. When the recomputation returns the same wrong
number -- which is what happens when the relaxation has converged in a different valley
rather than converged poorly -- the old behaviour was to give up and reject, discarding a
profile whose other hundred-odd samples were fine.

Measured on the 2HDM edge line: at lambda3 = 5.8123128723 the sample at T = 37.2595 GeV came
back 2.75 orders of magnitude low, and came back identically low after the tightened retry,
after a retry with fRatioConv divided by five, and after doubling the spline basis to
nb = 20. Three independent precision levers left it unchanged.

So the sample is discarded instead. That is safe only because it is checked: a non-finite
action is dropped from the percolation integral and from the support bank, and the gap it
leaves is judged by the ordinary Delta P and hot-head acceptance gates, which reject the
profile if the gap is too wide to interpolate across. The result is a measured accuracy
statement rather than a hard failure.

Part of TransitionListener v2
Documentation: https://tasillo.de/TransitionListener/
"""

from __future__ import annotations

import types

import numpy as np

from transitionlistener import bubbledynamics as bd


class _State:
    """The slice of the percolation state the rescue touches."""

    def __init__(self, T, H, S, P):
        self.TSYM = np.asarray(T, dtype=float)
        self.Hr = np.asarray(H, dtype=float)
        self.Sr = np.asarray(S, dtype=float)
        self.Pr = np.asarray(P, dtype=float)


class _Pot:
    def __init__(self):
        class Tracing:
            tunneling_params = {"deformation_deform_params": {"converge_0": 5.0,
                                                              "fRatioConv": 2e-2}}

        class Config:
            tracingConf = Tracing()

        self.config = Config()


def _profile(bad_factor=0.90):
    """A smooth rate curve with one sample whose action is far too low."""
    T = np.linspace(45.0, 34.0, 14)
    H = np.full(T.size, 2.5e-15)
    S = (150.0 + 0.55 * (T - 34.0)) * T          # smooth in S_3/T
    P = np.linspace(0.02, 0.85, T.size)
    bad = T.size // 2
    S[bad] *= bad_factor                          # one sample, orders of magnitude off in rate
    return T, H, S, P, bad


def settings(**over):
    values = dict(jitter_GH4_threshold=1.0, n_jitter_save=3)
    values.update(over)
    return types.SimpleNamespace(**values)


def _run_rescue(recompute_returns):
    """Drive the rescue with a solver whose retry returns a chosen value."""
    T, H, S, P, bad = _profile()
    state = _State(T, H, S, P)
    pot = _Pot()
    outdict = {float(t): {"action": float(a), "trantype": 1} for t, a in zip(T, S)}

    def fake_calc(pot_, temperature, sp, ep, od, *a, **k):
        value = recompute_returns(float(temperature))
        od[float(temperature)] = {"action": value, "trantype": 1}
        return value

    original = bd.calcAction
    bd.calcAction = fake_calc
    try:
        diag = bd._action_rate_jitter_diagnostic(state.TSYM, state.Hr, state.Sr, state.Pr)
        out = bd._try_action_jitter_tunneltight_rescue(
            state, settings(), diag, 1.0, pot, None, None, outdict)
    finally:
        bd.calcAction = original
    return state, outdict, out, T, bad


def test_the_fixture_really_is_rejected_first():
    """Premise: without a rescue this profile fails the gate."""
    T, H, S, P, _bad = _profile()
    diag = bd._action_rate_jitter_diagnostic(T, H, S, P)
    assert np.isfinite(diag["max_residual_oom"]) and diag["max_residual_oom"] > 1.0, (
        "the fixture does not exceed the jitter threshold, so it cannot exercise anything")


def test_a_repaired_sample_is_kept():
    """Control: when the retry fixes the action, nothing is discarded."""
    T0, _H, S0, _P, bad = _profile()
    good = {float(T0[bad]): float((150.0 + 0.55 * (T0[bad] - 34.0)) * T0[bad])}
    state, _outdict, diag, _T, bad_i = _run_rescue(
        lambda t: good.get(t, float((150.0 + 0.55 * (t - 34.0)) * t)))
    assert diag.get("rescue_discarded_temperatures") == [], (
        f"a sample was discarded even though the retry repaired it: {diag}")
    assert np.isfinite(state.Sr[bad_i]), "the repaired sample was voided anyway"


def test_an_irreparable_sample_is_discarded_and_recorded():
    """The fix: a retry that reproduces the wrong number costs that sample, not the point."""
    T0, _H, S0, _P, bad = _profile()
    stubborn = float(S0[bad])
    state, outdict, diag, T, bad_i = _run_rescue(lambda t: stubborn)
    assert diag.get("rescue_discarded_temperatures"), (
        f"the irreparable sample was kept, so the gate still sees it: {diag}")
    assert not np.isfinite(state.Sr[bad_i]), (
        "the sample was recorded as discarded but left finite in the profile, so the "
        "percolation integral would still use it")
    reasons = [e.get("reason") for e in outdict.get("_unstable_action_entries", [])]
    assert "action_rate_jitter_discarded_after_failed_retry" in reasons, (
        f"the discard was not recorded for the run report: {reasons}")


def test_the_discard_clears_the_diagnostic():
    """After discarding, the reported residual must reflect the profile that remains."""
    T0, _H, S0, _P, bad = _profile()
    stubborn = float(S0[bad])
    _state, _outdict, diag, _T, _b = _run_rescue(lambda t: stubborn)
    residual = diag.get("max_residual_oom")
    assert residual is None or not np.isfinite(residual) or residual <= 1.0, (
        f"the residual still exceeds the threshold after the discard ({residual}), so the "
        "point would be rejected anyway and nothing was gained")


def test_the_discard_reaches_the_diagnostics_snapshot():
    """A dropped support point must leave a record a consumer can read.

    `rescue_success` is measured on the profile that remains, so it is True after a
    discard: voiding the sample is what removes the residual it was flagged for. The
    observables therefore report "rescue attempted, not failed" for a profile that lost a
    support point, and `_unstable_action_entries` in the action cache is not something any
    consumer reads. The snapshot has to carry it.
    """
    T0, _H, S0, _P, bad = _profile()
    stubborn = float(S0[bad])
    _state, _outdict, diag, _T, _b = _run_rescue(lambda t: stubborn)
    assert diag.get("rescue_discarded_temperatures"), "premise: nothing was discarded"
    assert diag.get("rescue_success") is True, (
        "premise: the discard is what clears the residual, so success reads True here")

    T = np.array([90.0, 80.0, 70.0])
    state = _State(T, np.full(3, 1e-16), np.array([150.0, 140.0, 130.0]),
                   np.array([0.01, 0.3, 0.9]))
    state.support_bank = T.copy()
    state.action_temperatures = T.copy()
    state.rebuild_count = 0
    grid = bd.PercolationGrid(TSYM=T, Tstart=90.0, TpercApprox=80.0, tmin=70.0, tmax=90.0,
                              TBROmin=70.0, TBROmax=90.0)
    metadata = bd.PercolationDiagnostics.from_state(state, grid, diag, {})
    assert metadata.action_jitter_rescue_discarded_temperatures == [
        float(value) for value in diag["rescue_discarded_temperatures"]], (
        "the discarded temperatures did not reach the diagnostics snapshot, so a support "
        "point dropped from the profile leaves no record outside the solver")
