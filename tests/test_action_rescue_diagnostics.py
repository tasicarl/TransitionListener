"""The action rescue, and saying so.

Where the two fits of the action slope disagree, the samples that enter them are recomputed with
the path deformation tightened and the fit is repeated. That costs time and moves the answer, so
the run reports that it happened, whether it helped, how many actions were recomputed, and how
long the point took.
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))

from transitionlistener import bubbledynamics as bd  # noqa: E402
from transitionlistener import config as tl_config  # noqa: E402
from transitionlistener.interface import single_point as sp  # noqa: E402


class _Phase:
    """A phase that exists over a temperature range and nothing else."""

    def __init__(self, tmin=1.0, tmax=100.0):
        self.Tmin, self.Tmax = tmin, tmax


class _Interp:
    """Stands in for the action spline over the support bank."""

    def __init__(self, x, y):
        self.x = np.asarray(x, dtype=float)
        self._y = np.asarray(y, dtype=float)

    def __call__(self, t):
        return np.interp(np.asarray(t, dtype=float), self.x, self._y)


class _Conf:
    betaH_S3_fit_points = 11
    betaH_S3_fit_check_points = 7
    betaH_S3_fit_rel_tol = 0.03
    betaH_S3_fit_min_per_side = 2
    betaH_S3_fit_max_rel_span = 0.2
    betaH_S3_fallback_rel_step = 2e-3
    betaH_S3_fit_rescue = True


class _Pot:
    """Just enough potential for calc_betaH_S3: a config and a tunneling-parameter dict."""

    def __init__(self):
        class Tracing:
            tunneling_params = {"deformation_deform_params": {"converge_0": 5.0,
                                                              "fRatioConv": 2e-2}}

        class Config:
            percolationConf = _Conf()
            tracingConf = Tracing()

        self.config = Config()


T_EVAL = 50.0
# a smooth action, and the same one with one sample displaced enough to split the two fits
T_SAMPLES = T_EVAL * (1.0 + np.linspace(-0.05, 0.05, 15))
SMOOTH = (120.0 + 1.6 * (T_SAMPLES - T_EVAL)) * T_SAMPLES


def _spoiled():
    """One sample displaced by 0.8 %, which splits the eleven- and seven-point fits by 4 %."""
    values = SMOOTH.copy()
    values[len(values) // 2 + 1] *= 1.008
    return values


def _call(pot, samples, monkeypatched_actions=None, **diag_out):
    diagnostics: dict = {}
    betaH = bd.calc_betaH_S3(T_EVAL, _Interp(T_SAMPLES, samples), {}, pot,
                             _Phase(), _Phase(), False, diagnostics=diagnostics)
    diag_out.update(diagnostics)
    return betaH, diagnostics


def test_a_smooth_action_needs_no_rescue():
    betaH, diag = _call(_Pot(), SMOOTH)
    assert diag.get("fit_unstable") is False
    assert "fit_rescue_attempts" not in diag
    assert np.isfinite(betaH)


def test_a_split_fit_triggers_the_rescue(monkeypatch):
    """The recomputed actions come back smooth, so the refit agrees and the value moves."""
    calls = {"n": 0, "tight": 0}

    def fake_action(pot, T, phase_sym, phase_bro, outdict):
        calls["n"] += 1
        deform = pot.config.tracingConf.tunneling_params["deformation_deform_params"]
        if deform["converge_0"] == 1.0 and deform["fRatioConv"] == 5e-3:
            calls["tight"] += 1
        return float(np.interp(T, T_SAMPLES, SMOOTH))

    monkeypatch.setattr(bd, "calcAction", fake_action)
    pot = _Pot()
    betaH, diag = _call(pot, _spoiled())
    assert diag["fit_rescue_attempts"] == 11
    assert diag["fit_rescue_success"] is True
    assert diag["fit_unstable"] is False
    assert calls["tight"] == 11, calls
    # every recomputation was made with the deformation tightened, and the setting is restored
    assert pot.config.tracingConf.tunneling_params["deformation_deform_params"]["converge_0"] == 5.0
    assert diag["betaH_before_rescue"] != pytest.approx(betaH)


def test_a_rescue_that_does_not_help_keeps_the_flag(monkeypatch):
    """Recomputing returns the same split samples, so the refit still disagrees."""
    spoiled = _spoiled()
    monkeypatch.setattr(bd, "calcAction",
                        lambda pot, T, a, b, o: float(np.interp(T, T_SAMPLES, spoiled)))
    betaH, diag = _call(_Pot(), spoiled)
    assert diag["fit_rescue_attempts"] == 11
    assert diag["fit_rescue_success"] is False
    assert diag["fit_unstable"] is True
    assert np.isfinite(betaH)


def test_a_failing_recomputation_keeps_the_original_value(monkeypatch):
    def raising(*args, **kwargs):
        raise RuntimeError("the solver gave up")

    monkeypatch.setattr(bd, "calcAction", raising)
    betaH, diag = _call(_Pot(), _spoiled())
    assert diag["fit_rescue_success"] is False
    assert "fit_rescue_error" in diag
    assert betaH == pytest.approx(diag["betaH_fit"])


def test_the_rescue_can_be_switched_off(monkeypatch):
    monkeypatch.setattr(bd, "calcAction",
                        lambda *a, **k: pytest.fail("no action may be recomputed"))
    pot = _Pot()
    pot.config.percolationConf.betaH_S3_fit_rescue = False
    betaH, diag = _call(pot, _spoiled())
    assert diag["fit_unstable"] is True
    assert "fit_rescue_attempts" not in diag


class _WriterConf:
    derived_params = {name: name for name in
                      ("betaH_S3", "WARNING:action_rescue_attempted",
                       "WARNING:action_rescue_failed", "DIAG:action_rescue_attempts",
                       "DIAG:runtime_s")}


def _write(attempted, failed, attempts, runtime=12.5):
    out = tempfile.mkdtemp() + "/"
    sp._write_transition_outputs(out, _WriterConf(), {
        "strongestTransitionObservables": {
            "betaH_S3": 43.3,
            "WARNING:action_rescue_attempted": attempted,
            "WARNING:action_rescue_failed": failed,
            "DIAG:action_rescue_attempts": attempts,
            "DIAG:runtime_s": runtime,
        },
        "error": np.nan,
    })
    return open(out + "1_All_params.txt", encoding="utf-8").read()


@pytest.mark.parametrize("attempted,failed,attempts", [(False, False, 0), (True, False, 11),
                                                       (True, True, 11)])
def test_the_two_flags_cannot_contradict_each_other(attempted, failed, attempts):
    """`failed` may only be true where a rescue was attempted."""
    text = _write(attempted, failed, attempts)
    def flag(name):
        line = next(l for l in text.splitlines() if name in l)
        return line.split()[-1] == "True"
    assert flag("action_rescue_attempted") is attempted
    assert flag("action_rescue_failed") is failed
    assert not (flag("action_rescue_failed") and not flag("action_rescue_attempted"))


def test_the_diagnostics_are_written_apart_from_the_observables():
    """Two runs of a point differ in the wall clock, and that may not disturb the physics block."""
    text = _write(True, False, 11)
    body, _, rest = text.partition("Diagnostics:")
    assert "runtime_s" not in body and "action_rescue_attempts" not in body
    assert "betaH_S3" in body
    assert "runtime_s" in rest and "action_rescue_attempts" in rest
    assert rest.index("runtime_s") < rest.index("Warnings:")


def test_the_new_keys_are_registered_as_observables():
    """A key the writer never hears about is silently dropped."""
    for name in ("WARNING:action_rescue_attempted", "WARNING:action_rescue_failed",
                 "DIAG:action_rescue_attempts", "DIAG:runtime_s"):
        assert name in tl_config.all_observables, name


def test_the_runtime_is_recorded_where_a_scan_reads_it():
    """A grid scan takes its columns from the observables of the strongest transition alone,
    so a key at the top level of the result reaches the single-point writer and nothing else."""
    from transitionlistener.interface.pipeline import record_runtime

    result = {"strongestTransitionObservables": {"betaH_S3": 1.0}}
    record_runtime(result, 12.5)
    assert result["strongestTransitionObservables"]["DIAG:runtime_s"] == 12.5
    record_runtime({"error": 1.0}, 12.5)          # a result without observables must not raise
    record_runtime(None, 12.5)


def test_the_retry_never_loosens_a_stricter_setting():
    """Tightening is the point; a run already asked for something stricter keeps it."""
    pot = _Pot()
    deform = pot.config.tracingConf.tunneling_params["deformation_deform_params"]
    deform["converge_0"], deform["fRatioConv"] = 0.5, 1e-3
    tightened = bd._tight_tunneling_params(pot)["deformation_deform_params"]
    assert tightened["converge_0"] == 0.5
    assert tightened["fRatioConv"] == 1e-3

    deform["converge_0"], deform["fRatioConv"] = 5.0, 2e-2
    tightened = bd._tight_tunneling_params(pot)["deformation_deform_params"]
    assert tightened["converge_0"] == 1.0
    assert tightened["fRatioConv"] == 5e-3


def test_the_rescue_counter_starts_at_zero():
    """An absent key and a zero are different things to whatever reads the column."""
    from transitionlistener import transitionObservables as adaptive
    from transitionlistener import transitionObservables_fixedstep as fixed

    for module in (adaptive, fixed):
        source = open(module.__file__, encoding="utf-8").read()
        assert '"DIAG:action_rescue_attempts": 0,' in source, module.__name__


def test_a_cached_loose_action_is_not_reused_by_the_retry(monkeypatch):
    """`calcAction` hands back whatever is in the cache, and the percolation temperature is
    always in it, so a retry that did not clear it would refit on the value it is replacing."""
    seen = []

    def fake_action(pot, T, phase_sym, phase_bro, outdict):
        seen.append(float(T))
        value = float(np.interp(T, T_SAMPLES, SMOOTH))
        outdict[float(T)] = {"action": value}
        return value

    monkeypatch.setattr(bd, "calcAction", fake_action)
    # a stale entry at the evaluation point, as percolation leaves behind
    outdict = {T_EVAL: {"action": 1.0e9}}
    diagnostics: dict = {}
    bd.calc_betaH_S3(T_EVAL, _Interp(T_SAMPLES, _spoiled()), outdict, _Pot(),
                     _Phase(), _Phase(), False, diagnostics=diagnostics)
    assert diagnostics["fit_rescue_attempts"] > 0
    assert any(abs(t - T_EVAL) < 1e-9 for t in seen), "the stale entry was reused"
    assert outdict[T_EVAL]["action"] != 1.0e9
    replaced = outdict.get("_unstable_action_entries", [])
    assert any(entry.get("reason") == "betaH_S3_fit_rescue" for entry in replaced)


def test_both_retries_tighten_through_the_same_helper():
    """The rate-jitter retry set the two values outright, so it could loosen a stricter run."""
    import inspect

    source = inspect.getsource(bd._try_action_jitter_tunneltight_rescue)
    assert "_tight_tunneling_params(pot)" in source
    assert 'deform["converge_0"] = 1.0' not in source
    assert 'deform["fRatioConv"] = 5.0e-3' not in source


def test_a_retry_whose_answer_is_not_used_puts_the_cache_back(monkeypatch):
    """The value reported then came from the samples as they were, and the cache must agree."""
    def raising(pot, T, phase_sym, phase_bro, outdict):
        raise RuntimeError("the solver gave up")

    monkeypatch.setattr(bd, "calcAction", raising)
    original = {T_EVAL: {"action": 1234.0}}
    outdict = dict(original)
    betaH = bd.calc_betaH_S3(T_EVAL, _Interp(T_SAMPLES, _spoiled()), outdict, _Pot(),
                             _Phase(), _Phase(), False, diagnostics={})
    assert outdict[T_EVAL] == original[T_EVAL], "the cache still misses what the retry removed"
    assert outdict.get("_unstable_action_entries", []) == []
    assert np.isfinite(betaH)


def test_the_stencil_stays_inside_the_traced_range():
    """The outermost sample sits at `half` steps from T, so the cap has to scale with it."""
    seen = []

    def record(pot, T, phase_sym, phase_bro, outdict):
        seen.append(float(T))
        return float(np.interp(T, T_SAMPLES, SMOOTH))

    # Thirteen fit points put the outermost sample six steps out, where a cap that does not
    # know how many there are lets it past the edge. The support has to be at least that large,
    # or the sparse-support fallback runs instead and no retry happens at all.
    tmin, tmax = 49.0, 51.0
    pot = _Pot()
    pot.config.percolationConf.betaH_S3_fit_points = 13
    pot.config.percolationConf.betaH_S3_fallback_rel_step = 0.05
    diagnostics: dict = {}
    monkey = pytest.MonkeyPatch()
    monkey.setattr(bd, "calcAction", record)
    try:
        bd.calc_betaH_S3(T_EVAL, _Interp(T_SAMPLES, _spoiled()), {}, pot,
                         _Phase(tmin, tmax), _Phase(tmin, tmax), False, diagnostics=diagnostics)
    finally:
        monkey.undo()
    assert diagnostics.get("fit_rescue_attempts"), "the retry did not run, so nothing was tested"
    assert min(seen) > tmin, f"{min(seen)} is below the traced range"
    assert max(seen) < tmax, f"{max(seen)} is above the traced range"


def test_the_cache_is_restored_when_the_solver_fails_part_way_through(monkeypatch):
    """The failure comes after some entries have already been removed, which is the case the
    caller has to be able to undo: the helper never returns, so it cannot hand the list back."""
    # The stencil runs from its lowest temperature upward, so the failure has to come after the
    # entries below have been taken; failing earlier removes nothing and tests nothing.
    calls = {"n": 0}

    def fail_after_seven(pot, T, phase_sym, phase_bro, outdict):
        calls["n"] += 1
        if calls["n"] > 7:
            raise RuntimeError("the solver gave up")
        value = float(np.interp(T, T_SAMPLES, SMOOTH))
        outdict[float(T)] = {"action": value}
        return value

    monkeypatch.setattr(bd, "calcAction", fail_after_seven)
    # three entries the retry reaches within its first seven steps
    step = _Conf.betaH_S3_fallback_rel_step * T_EVAL
    original = {T_EVAL + k * step: {"action": 1000.0 + k} for k in (-1, 0, 1)}
    outdict = {k: dict(v) for k, v in original.items()}
    diagnostics: dict = {}
    betaH = bd.calc_betaH_S3(T_EVAL, _Interp(T_SAMPLES, _spoiled()), outdict, _Pot(),
                             _Phase(), _Phase(), False, diagnostics=diagnostics)
    assert diagnostics["fit_rescue_success"] is False
    assert calls["n"] > 7, "the solver did not fail part of the way through"
    assert diagnostics.get("fit_rescue_error"), "the retry did not go through the failure path"
    for key, payload in original.items():
        assert outdict.get(key) == payload, f"{key} was not put back"
    assert outdict.get("_unstable_action_entries", []) == []
    assert np.isfinite(betaH)


def test_a_failed_run_reports_its_runtime():
    """Most points that fail go through the error branch, which returned before recording it."""
    import inspect
    from transitionlistener.interface import pipeline

    source = inspect.getsource(pipeline.run_TL)
    before, _, after = source.partition("if error is not None:")
    assert "record_runtime" in after.split("return result")[0], \
        "the error branch returns before the runtime is recorded"
