"""The speed of the expansion factors of arXiv:2511.15687, and the observable betaH_Gamma.

The fits with the cosmic expansion were made for an exponential nucleation rate
Gamma ~ exp(beta t) and are functions of its beta/H_V, with H_V the Hubble rate of the
free-energy difference of the two phases alone. ``gwConf.expansion_speed`` selects the speed
passed to the three expansion factors -- the amplitude bracket of eq. (23), the
peak-frequency bracket of eq. (22) and the horizon frequency of eq. (21) -- while every
other factor of the spectrum keeps the speed from the bubble separation. ``betaH_Gamma`` is
H^-1 d ln Gamma/dt at T_perc, with H the Hubble rate of the percolation history, from the
slope of the nucleation rate itself. It equals the fits' beta/H_V up to H(T_perc)/H_V, which
is close to one in a vacuum-dominated transition; no such factor is applied.

Part of TransitionListener v2.0
Documentation: https://tasillo.de/TransitionListener/
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest
from scipy import interpolate

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from transitionlistener import bubbledynamics as bd  # noqa: E402
from transitionlistener import config as tlconfig  # noqa: E402
from transitionlistener import errors  # noqa: E402
from transitionlistener import runtime_options  # noqa: E402
from transitionlistener.config import Configuration, GWConf  # noqa: E402
from transitionlistener.gwfopt import FOPTspectrum  # noqa: E402

F_PERC = 0.28957
BETAH_R = 10.0
BETAH_G = 7.0          # a rate that slows down: smaller than the separation speed
# Field shells only, strongly supercooled, reheating at 150 MeV
PARAMS = dict(
    alpha=1e10, RH=(8 * np.pi / F_PERC) ** (1 / 3) / BETAH_R, Treh_SM_GeV=0.15,
    g_eff_tot_reh=20.0, h_eff_tot_reh=20.0, kappa_phi=1.0, kappa_sw=0.0, kappa_turb=0.0,
    v_wall=1.0, c_s=1 / np.sqrt(3), D=1.0, alpha_hyd=1e10,
)


def _spectrum(params=PARAMS, **gw_choices):
    conf = Configuration()
    for key, value in gw_choices.items():
        setattr(conf.gwConf, key, value)
    return FOPTspectrum(params, config=conf)


def _amplitude_bracket(b):
    return 1 + 0.8 * (1 - np.exp(np.sqrt(1 / b)))           # eq. (23) over 0.06


def _frequency_bracket(b):
    return 1 + 1.8 * b ** (-1.2)                              # eq. (22) over 0.7 beta/H


LV_EXPANDING = dict(source_model="LewickiVaskonen22", expansion_suppression=True, causality_tail="off")


def test_default_speed_is_the_separation_speed_and_ignores_betaH_Gamma():
    assert GWConf.expansion_speed == "betaH_RH"
    plain = _spectrum(**LV_EXPANDING)
    with_g = _spectrum(dict(PARAMS, betaH_Gamma=BETAH_G), **LV_EXPANDING)
    f = np.logspace(-10, -6, 200)
    np.testing.assert_array_equal(plain.h2Omega_0_sum(f), with_g.h2Omega_0_sum(f))


def test_betaH_Gamma_enters_only_the_two_brackets_of_the_expanding_fit():
    """Peak value Rh2 * 0.06 [eq. 23 bracket at betaH_Gamma] * K^2 / (beta/H_RH)^2 at the peak
    frequency 0.7 (beta/H_RH) [eq. 22 bracket at betaH_Gamma] in units of a_p H_p / 2 pi."""
    ref = _spectrum(**LV_EXPANDING)
    new = _spectrum(dict(PARAMS, betaH_Gamma=BETAH_G), expansion_speed="betaH_Gamma", **LV_EXPANDING)
    assert ref.betaH == pytest.approx(BETAH_R) and new.betaH == pytest.approx(BETAH_R)
    f = np.logspace(np.log10(5e-9), np.log10(1e-7), 40001)
    for spec, b_exp in ((ref, BETAH_R), (new, BETAH_G)):
        y = spec.h2Omega_0_sum(f)
        i = int(np.argmax(y))
        assert 0 < i < f.size - 1
        f_peak, y_peak = f[i], y[i]
        # the ratio of the two spectra pins the brackets; (H/beta)^2 and 0.7 beta/H use beta_RH in both
        if spec is ref:
            f_ref, y_ref = f_peak, y_peak
    ratio_A = _amplitude_bracket(BETAH_G) / _amplitude_bracket(BETAH_R)
    ratio_f = _frequency_bracket(BETAH_G) / _frequency_bracket(BETAH_R)
    assert y_peak / y_ref == pytest.approx(ratio_A, rel=1e-6)
    assert f_peak / f_ref == pytest.approx(ratio_f, rel=2e-4)


def test_betaH_Gamma_sets_the_horizon_frequency_and_the_lisa_ratios():
    ref = _spectrum(**LV_EXPANDING)
    new = _spectrum(dict(PARAMS, betaH_Gamma=BETAH_G), expansion_speed="betaH_Gamma", **LV_EXPANDING)
    rA0, rf0, fH0 = ref._expansion_factors(1.0)
    rA1, rf1, fH1 = new._expansion_factors(1.0)
    assert rA1 == pytest.approx(_amplitude_bracket(BETAH_G))
    assert rf1 == pytest.approx(_frequency_bracket(BETAH_G))
    assert 2 * np.pi * fH1 == pytest.approx(1.0 - 0.6 * BETAH_G ** (-0.67))
    assert 2 * np.pi * fH0 == pytest.approx(1.0 - 0.6 * BETAH_R ** (-0.67))
    # the LISA pipeline template takes the same ratios with the expansion on
    lisa = dict(PARAMS, kappa_phi=0.0, kappa_sw=0.9, betaH_Gamma=BETAH_G)
    a = _spectrum(lisa, expansion_suppression=True, causality_tail="off")
    b = _spectrum(lisa, expansion_suppression=True, causality_tail="off", expansion_speed="betaH_Gamma")
    f = np.logspace(-10, -6, 400)
    assert np.max(b.h2Omega_0_sum(f)) / np.max(a.h2Omega_0_sum(f)) == pytest.approx(
        _amplitude_bracket(BETAH_G) / _amplitude_bracket(BETAH_R), rel=1e-2)


@pytest.mark.parametrize("value", [None, np.nan, 0.0, -3.0, "fast"])
def test_missing_or_unphysical_betaH_Gamma_is_refused(value):
    params = dict(PARAMS) if value is None else dict(PARAMS, betaH_Gamma=value)
    with pytest.raises(errors.GWParamsError):
        _spectrum(params, expansion_speed="betaH_Gamma", **LV_EXPANDING)
    # without the expansion the speed is never read, so nothing is refused
    _spectrum(params, expansion_speed="betaH_Gamma", source_model="LewickiVaskonen22")


def test_invalid_speed_is_refused_and_overrides_reach_the_config():
    with pytest.raises(errors.GWParamsError):
        _spectrum(expansion_speed="betaH_S3")
    conf = Configuration()
    runtime_options.apply_gw_overrides(conf.gwConf, {"gw_expansion_speed": "betaH_Gamma"})
    assert conf.gwConf.expansion_speed == "betaH_Gamma"
    with pytest.raises(ValueError):
        runtime_options.apply_gw_overrides(Configuration().gwConf, {"gw_expansion_speed": "fast"})
    assert Configuration().gwConf.expansion_speed == "betaH_RH"


def test_modelling_choices_report_the_speed_with_the_expansion():
    assert "expansion_speed" not in _spectrum(dict(PARAMS, kappa_phi=0.0, kappa_sw=0.9)).modelling_choices()
    report = _spectrum(dict(PARAMS, betaH_Gamma=BETAH_G), expansion_speed="betaH_Gamma",
                       **LV_EXPANDING).modelling_choices()
    assert report["expansion_speed"] == "betaH_Gamma"


def test_betaH_Gamma_is_a_registered_observable():
    from transitionlistener.interface.samplers import get_empty_result
    assert "betaH_Gamma" in tlconfig.all_observables
    assert "betaH_Gamma" in get_empty_result()


# ---------------------------------------------------------------- the slope of the rate
T_PERC = 50.0


def _action(T):
    """S3(T) with S3/T = 140 - 3 (T - T_PERC) + 0.2 (T - T_PERC)**2."""
    x = np.asarray(T, dtype=float) - T_PERC
    return T * (140.0 - 3.0 * x + 0.2 * x * x)


def _setup(T_samples, log_gamma):
    """Support samples with cached actions and a potential whose rate is ``log_gamma(T, S)``."""
    T_samples = np.sort(np.asarray(T_samples, dtype=float))
    outdict = {float(t): {"action": float(_action(t))} for t in T_samples}
    Sint = interpolate.interp1d(T_samples, _action(T_samples), kind="linear")
    pot = types.SimpleNamespace(
        config=types.SimpleNamespace(percolationConf=types.SimpleNamespace()),
        logGammaNuc=lambda T, S, instanton=None: float(log_gamma(T, S)),
    )
    phase = types.SimpleNamespace(Tmin=1.0, Tmax=200.0)
    return Sint, outdict, pot, phase


def _default_log_gamma(T, S):
    return float(np.atleast_1d(bd.logGamma(T, S))[0])


def _expected_slope(extra_dlnT=0.0):
    """-d ln Gamma/d ln T at T_PERC for Gamma = T^4 (x/2 pi)^{3/2} e^{-x}, x = S3/T, minus the
    log-derivative of an extra factor e^{extra}."""
    x = 140.0
    dx_dlnT = T_PERC * (-3.0)
    return -(4.0 + 1.5 * dx_dlnT / x - dx_dlnT + extra_dlnT)


def test_rate_slope_of_the_default_rate():
    T = T_PERC * (1.0 + np.linspace(-0.004, 0.004, 11))
    Sint, outdict, pot, phase = _setup(T, _default_log_gamma)
    slope = bd.calc_rate_log_slope(T_PERC, Sint, outdict, pot, phase, phase)
    assert slope == pytest.approx(_expected_slope(), rel=1e-6)


def test_rate_slope_follows_the_model_rate_not_the_action():
    """In model mode the rate carries corrections beyond the action (the determinant); the
    slope must be the slope of that rate."""
    k = 5.0      # ln Gamma gets + k (T - T_PERC)/T_PERC, d/dlnT = k at T_PERC
    model = lambda T, S: _default_log_gamma(T, S) + k * (T - T_PERC) / T_PERC
    T = T_PERC * (1.0 + np.linspace(-0.004, 0.004, 11))
    Sint, outdict, pot, phase = _setup(T, model)
    slope = bd.calc_rate_log_slope(T_PERC, Sint, outdict, pot, phase, phase)
    assert slope == pytest.approx(_expected_slope(extra_dlnT=k), rel=1e-6)
    assert abs(slope - _expected_slope()) > 1.0


def test_rate_slope_falls_back_to_fresh_actions_on_a_sparse_support(monkeypatch):
    T = np.array([30.0, 40.0, 60.0, 70.0, 80.0])          # far from T_PERC, fewer than the fit wants
    Sint, outdict, pot, phase = _setup(T, _default_log_gamma)

    def fresh_action(pot_, t, start, end, out, *a, **kw):
        out.setdefault(float(t), {"action": float(_action(t))})
        return out[float(t)]["action"]
    monkeypatch.setattr(bd, "calcAction", fresh_action)
    diag = {}
    slope = bd.calc_rate_log_slope(T_PERC, Sint, outdict, pot, phase, phase, diagnostics=diag)
    assert diag["fallback"] is True
    assert slope == pytest.approx(_expected_slope(), rel=1e-4)


def test_rate_slope_outside_the_phase_overlap_is_nan():
    T = T_PERC * (1.0 + np.linspace(-0.004, 0.004, 11))
    Sint, outdict, pot, _ = _setup(T, _default_log_gamma)
    narrow = types.SimpleNamespace(Tmin=60.0, Tmax=200.0)
    assert np.isnan(bd.calc_rate_log_slope(T_PERC, Sint, outdict, pot, narrow, narrow))


# ---------------------------------------------------------------- the observable
def test_betaH_Gamma_observable_is_time_factor_times_slope_without_H_over_H_V(monkeypatch):
    """betaH_Gamma = (3 c_s^2) (-d ln Gamma/d ln T): no factor H(Tperc)/H_V. The history's
    Hubble rate is set to 0.9 H_V, so a reintroduced H/H_V (or H_V/H) factor shows."""
    from transitionlistener import transitionObservables as to
    dV = 2.7e-3                                   # internal units; H_V from HubbleParameter
    CF = 1.0
    H_V = float(bd.HubbleParameter(dV, CF))
    H_perc = 0.9 * H_V
    pot = types.SimpleNamespace(
        conversionFactor=CF,
        Vtot=lambda phi, T, include_decoupled=True: dV if phi == "sym" else 0.0,
    )
    phase_sym = types.SimpleNamespace(valAt=lambda T: "sym")
    phase_bro = types.SimpleNamespace(valAt=lambda T: "bro")
    ctx = types.SimpleNamespace(derived_params={}, pot=pot, outdict={}, phase_symmetric=phase_sym,
                                phase_broken=phase_bro, verbose=False)
    perc = types.SimpleNamespace(Tperc=T_PERC, Sint=object(), Hint=lambda T: H_perc)
    monkeypatch.setattr(to, "calc_rate_log_slope", lambda *a, **kw: 8.0)
    obs = object.__new__(to.TransitionObservables)
    assert obs._betaH_Gamma(ctx, perc, 0.97) == pytest.approx(0.97 * 8.0, rel=1e-12)
    ctx.derived_params["WARNING:no_perc_splines"] = True
    assert np.isnan(obs._betaH_Gamma(ctx, perc, 0.97))


# ---------------------------------------------------------------- the samples of the fit
class _Interp:
    """The action spline over the support bank: its nodes are ``x``."""

    def __init__(self, x, y):
        self.x = np.asarray(x, dtype=float)
        self._y = np.asarray(y, dtype=float)

    def __call__(self, t):
        return np.interp(np.asarray(t, dtype=float), self.x, self._y)


def _rising_s_over_T(T):
    """S3/T = 140 + 3 (T - T_PERC) + 0.2 (T - T_PERC)**2, rising towards T_c."""
    x = np.asarray(T, dtype=float) - T_PERC
    return 140.0 + 3.0 * x + 0.2 * x * x


def _rising_slope():
    """-d ln Gamma/d ln T at T_PERC of the default rate of the action above."""
    x, dx_dlnT = 140.0, 3.0 * T_PERC
    return -(4.0 + 1.5 * dx_dlnT / x - dx_dlnT)


SUPPORT = T_PERC * (1.0 + np.linspace(-0.01, 0.01, 15))
FACTOR, DV, H_RATIO = 0.97, 2.7e-3, 0.9   # H_RATIO: H(Tperc)/H_V of the history, not in betaH_Gamma


def _populate(monkeypatch, spoil=1.0):
    """Run ``_populate_beta_related`` for betaH_S3 and betaH_Gamma on a synthetic support.

    One support sample is multiplied by ``spoil``; actions computed afresh (the fit rescue
    recomputes its stencil) are the exact ones.
    """
    from transitionlistener import transitionObservables as to
    actions = SUPPORT * _rising_s_over_T(SUPPORT)
    actions[len(actions) // 2 + 1] *= spoil
    outdict = {float(t): {"action": float(s)} for t, s in zip(SUPPORT, actions)}

    def fresh_action(pot_, t, start, end, out, *a, **kw):
        out.setdefault(float(t), {"action": float(t * _rising_s_over_T(t))})
        return out[float(t)]["action"]
    monkeypatch.setattr(bd, "calcAction", fresh_action)
    monkeypatch.setattr(to.TransitionObservables, "_beta_time_temperature_factor",
                        lambda self, ctx, T: FACTOR)
    H_V = float(bd.HubbleParameter(DV, 1.0))
    tracing = types.SimpleNamespace(tunneling_params={
        "deformation_deform_params": {"converge_0": 5.0, "fRatioConv": 2e-2}})
    pot = types.SimpleNamespace(
        config=types.SimpleNamespace(percolationConf=types.SimpleNamespace(betaH_S3_fit_rescue=True),
                                     tracingConf=tracing),
        logGammaNuc=lambda T, S, instanton=None: _default_log_gamma(T, S),
        conversionFactor=1.0,
        Vtot=lambda phi, T, include_decoupled=True: DV if phi == "sym" else 0.0,
    )
    phase_sym = types.SimpleNamespace(Tmin=1.0, Tmax=200.0, valAt=lambda T: "sym")
    phase_bro = types.SimpleNamespace(Tmin=1.0, Tmax=200.0, valAt=lambda T: "bro")
    ctx = types.SimpleNamespace(derived_params={}, verbose=False, pot=pot, outdict=outdict,
                                derived_param_names=["betaH_S3", "betaH_Gamma"],
                                phase_symmetric=phase_sym, phase_broken=phase_bro)
    perc = types.SimpleNamespace(Tperc=T_PERC, Sint=_Interp(SUPPORT, actions),
                                 Hint=lambda T: H_RATIO * H_V)
    obs = object.__new__(to.TransitionObservables)
    obs._populate_beta_related(ctx, perc, 1.0, 200.0)
    return ctx.derived_params, perc, ctx


def test_betaH_Gamma_is_fitted_on_the_samples_of_betaH_S3_and_follows_its_rescue(monkeypatch):
    """One support sample displaced by 0.8 % splits the two fits of betaH_S3, so its fit
    rescue refits on a stencil of fresh actions. betaH_Gamma has to rest on the same actions:
    on the displaced support its slope is about 5 % off."""
    derived, perc, ctx = _populate(monkeypatch, spoil=1.008)
    assert derived.get("WARNING:action_rescue_attempted"), "the rescue did not run"
    assert not derived.get("WARNING:action_rescue_failed")
    assert derived["betaH_S3"] == pytest.approx(FACTOR * 3.0 * T_PERC, rel=1e-6)
    expected = FACTOR * _rising_slope()          # no H/H_V, although the history has H = 0.9 H_V
    assert derived["betaH_Gamma"] == pytest.approx(expected, rel=1e-5)
    # the displaced support on its own gives a different answer, so the above can fail
    own = bd.calc_rate_log_slope(T_PERC, perc.Sint, ctx.outdict, ctx.pot,
                                 ctx.phase_symmetric, ctx.phase_broken)
    assert abs(FACTOR * own / expected - 1.0) > 1e-2


def test_without_a_rescue_the_shared_samples_are_its_own_selection(monkeypatch):
    """Passing the samples of betaH_S3 changes nothing where no rescue ran."""
    from transitionlistener import transitionObservables as to
    derived, perc, ctx = _populate(monkeypatch)
    assert not derived.get("WARNING:action_rescue_attempted")
    # bit for bit what the observable is without the samples handed over
    alone = object.__new__(to.TransitionObservables)._betaH_Gamma(ctx, perc, FACTOR)
    assert derived["betaH_Gamma"] == alone
    assert derived["betaH_Gamma"] == pytest.approx(FACTOR * _rising_slope(), rel=1e-5)


def test_requesting_betaH_Gamma_alone_reaches_the_block_that_computes_it(monkeypatch):
    from transitionlistener import transitionObservables as to
    calls = []
    monkeypatch.setattr(to.TransitionObservables, "_beta_time_temperature_factor",
                        lambda self, ctx, T: FACTOR)
    monkeypatch.setattr(to.TransitionObservables, "_populate_beta_related",
                        lambda self, ctx, perc, tmin, tmax: calls.append(ctx.derived_param_names))
    ctx = types.SimpleNamespace(derived_params={}, pot=None, verbose=False, outdict={},
                                phase_symmetric=None, phase_broken=None,
                                derived_param_names=["betaH_Gamma"])
    perc = types.SimpleNamespace(Tperc=T_PERC)
    object.__new__(to.TransitionObservables)._compute_beta_and_separation(ctx, perc, 1.0, 200.0)
    assert calls == [["betaH_Gamma"]]


# ---------------------------------------------------------------- the run's own timeout
def test_a_timeout_while_fitting_the_rate_is_not_swallowed(monkeypatch):
    """The sparse-support fallback computes fresh actions, where the run's alarm can go off."""
    from transitionlistener import transitionObservables as to
    T = np.array([30.0, 40.0, 60.0, 70.0, 80.0])
    Sint, outdict, pot, phase = _setup(T, _default_log_gamma)

    def alarm(*a, **kw):
        raise errors.Timeout()
    monkeypatch.setattr(bd, "calcAction", alarm)
    with pytest.raises(errors.Timeout):
        bd.calc_rate_log_slope(T_PERC, Sint, outdict, pot, phase, phase)
    ctx = types.SimpleNamespace(derived_params={}, pot=pot, outdict=outdict, phase_symmetric=phase,
                                phase_broken=phase, verbose=False)
    perc = types.SimpleNamespace(Tperc=T_PERC, Sint=Sint, Hint=lambda T: 1.0)
    with pytest.raises(errors.Timeout):
        object.__new__(to.TransitionObservables)._betaH_Gamma(ctx, perc, 1.0)


# ---------------------------------------------------------------- the fixed step size solver
def _pot_for(algorithm_mode, expansion, speed):
    conf = Configuration()
    conf.percolationConf.algorithm_mode = algorithm_mode
    conf.gwConf.expansion_suppression = expansion
    conf.gwConf.expansion_speed = speed
    return types.SimpleNamespace(config=conf)


def _construct(monkeypatch, pot, computed):
    """Build the observables class a run would pick, recording instead of computing."""
    from transitionlistener.interface.pipeline import _select_observables_class
    cls = _select_observables_class(pot)
    monkeypatch.setattr(cls, "computeTransitionObservables", lambda self: computed.append(1) or {})
    monkeypatch.setattr(cls, "check_T0_global_min", lambda self: None)
    cls(pot, [], [], False)
    return cls


def test_the_fixed_step_solver_refuses_betaH_Gamma_before_the_percolation(monkeypatch):
    from transitionlistener import transitionObservables_fixedstep as tof
    from transitionlistener.interface.pipeline import _select_observables_class
    pot = _pot_for("fixed_step_size", True, "betaH_Gamma")
    assert _select_observables_class(pot) is tof.TransitionObservables
    computed = []
    with pytest.raises(errors.GWParamsError, match="fixed_step_size"):
        _construct(monkeypatch, pot, computed)
    assert computed == [], "the percolation ran before the refusal"


@pytest.mark.parametrize("mode, expansion, speed", [
    ("fixed_step_size", False, "betaH_Gamma"),    # the speed is never read without the expansion
    ("fixed_step_size", True, "betaH_RH"),
    ("adaptive_step_size", True, "betaH_Gamma"),
])
def test_only_that_combination_is_refused(monkeypatch, mode, expansion, speed):
    computed = []
    _construct(monkeypatch, _pot_for(mode, expansion, speed), computed)
    assert computed == [1]


def test_the_fixed_step_context_leaves_betaH_Gamma_nan_without_a_warning():
    from transitionlistener import transitionObservables_fixedstep as tof
    obs = object.__new__(tof.TransitionObservables)
    conf = Configuration()
    obs.pot, obs.phases = types.SimpleNamespace(config=conf), {"hot": None, "cold": None}
    obs.GWconfig, obs.PercolationConf = conf.gwConf, conf.percolationConf
    obs.derived_param_names, obs.verbose = ["betaH_Gamma"], False
    tr = types.SimpleNamespace(Tnuc=30.0, full_tunneling_info={}, high_phase="hot", low_phase="cold")
    derived = tof.TransitionObservables._initialize_transition_context(obs, tr).derived_params
    assert "betaH_Gamma" in derived and np.isnan(derived["betaH_Gamma"])
