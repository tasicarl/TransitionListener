"""The 3D effective-theory rate mode, end to end (WP5a).

Checks that the opt-in mode is genuinely opt-in, that the subclass changes the
rate and nothing else, and that the rate assembly is what it claims to be.
"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "models")
)

from transitionlistener.eft_rate import one_field_rate_3d  # noqa: E402
from transitionlistener import bubbledynamics as bd
from transitionlistener.eft_rate import abelian_higgs_3d

# sub-GeV paper best-fit point A (arXiv:2502.19478 table 3)
GA, YA, VA = 0.677, 0.224, 0.173
TC_A = 0.034772          # GeV, computed from the 4D potential


def _pot3d():
    import TL_conformal_u1_fermions_3d as mod
    return mod.specific_potential({"g": GA, "y": YA, "v_GeV": VA}, verbose=False)


def _pot4d():
    import TL_conformal_u1_fermions as mod
    return mod.specific_potential({"g": GA, "y": YA, "v_GeV": VA}, verbose=False)


def _T_internal(pot, over_Tc):
    return over_Tc * TC_A / pot.conversionFactor


# ------------------------------------------------- the 4D rate on request
def test_the_3d_rate_with_its_nlo_correction_is_the_default_of_the_3d_model():
    conf = _pot3d().config.nucleationConf
    assert (conf.rate_mode, conf.include_determinants, conf.include_dim6) == ("model", True, True)
    assert (conf.input_scheme, conf.mu3_choice) == ("zero_quartic", "gT")


def test_a_4d_model_keeps_the_4d_defaults():
    conf = _pot4d().config.nucleationConf
    assert (conf.rate_mode, conf.include_determinants, conf.include_dim6) == ("S3overT_4D", False, False)
    assert (conf.input_scheme, conf.mu3_choice) == ("CW", "g3sq")


def test_the_4d_mode_is_bit_for_bit_the_4d_rate():
    pot = _pot3d()
    pot.config.nucleationConf.rate_mode = "S3overT_4D"
    T = _T_internal(pot, 0.0656)
    for S in (100.0, 500.0, 2000.0):
        assert pot.logGammaNuc(T, S) == float(np.atleast_1d(bd.logGamma(T, S))[0])


def test_the_subclass_changes_the_rate_and_nothing_else():
    """The potential, spectrum and derived couplings must be untouched."""
    p3, p4 = _pot3d(), _pot4d()
    assert p3.l == p4.l
    assert p3.conversionFactor == p4.conversionFactor
    X = np.array([[0.4 * p4.v_stable]])
    T = _T_internal(p4, 0.2)
    assert float(p3.Vtot(X, T)[0]) == float(p4.Vtot(X, T)[0])
    # boson_massSq returns (masses, dof, c, is_boson): compare piecewise
    for a, b in zip(p3.boson_massSq(X, T), p4.boson_massSq(X, T)):
        assert np.array_equal(np.asarray(a), np.asarray(b))
    assert np.array_equal(np.asarray(p3.fermion_massSq(X)[0]),
                          np.asarray(p4.fermion_massSq(X)[0]))


def test_model_mode_gives_a_different_and_finite_rate():
    """Now via the action seam, which is what reaches the pipeline."""
    pot = _pot3d()
    pot.config.nucleationConf.rate_mode = "model"
    T = _T_internal(pot, 0.0656)
    S_model = pot.actionNuc(T, 500.0)
    assert np.isfinite(S_model)
    val = float(np.atleast_1d(bd.logGamma(T, S_model))[0])
    assert np.isfinite(val)
    assert val != float(np.atleast_1d(bd.logGamma(T, 500.0))[0])


def test_model_mode_ignores_the_4d_action_it_is_handed():
    """It supplies its own action, so the 4D one must not leak in."""
    pot = _pot3d()
    pot.config.nucleationConf.rate_mode = "model"
    T = _T_internal(pot, 0.0656)
    assert pot.actionNuc(T, 100.0) == pot.actionNuc(T, 9999.0)


def test_matched_parameters_are_finite_and_the_gauge_coupling_positive():
    pot = _pot3d()
    m3sq, lam3, g3sq, mD2 = pot.matched_3d_parameters(0.0656 * TC_A)
    assert all(np.isfinite(v) for v in (m3sq, lam3, g3sq, mD2))
    assert g3sq > 0.0 and m3sq > 0.0
    # the Debye mass is part of the matching, because the temporal mode is
    # in the leading-order potential
    assert mD2 > 0.0


# ------------------------------------------------------ the assembly


# ------------------------------------------------- the bounce solver
def _matched_at(over_Tc):
    import dr_conformal_u1_fermions_matching as dr
    T = over_Tc * TC_A
    mu = 7.0553 * T
    g, y, lam, _ = dr.rg_run(GA, YA, 0.0146238, VA, mu)
    return (T, dr.scalar_thermal_mass_sq_LO(g, y, lam, T),
            dr.lambda3_NLO(g, y, lam, T, mu), dr.g3_sq_LO(g, T),
            dr.debye_mass_sq_LO(g, T))


def test_bounce_action_is_positive_and_falls_with_supercooling():
    hot, cold = _matched_at(0.0656), _matched_at(0.03)
    a = abelian_higgs_3d.solve_LO_bounce(*hot[1:4], m_debye_sq=hot[4])[0]
    b = abelian_higgs_3d.solve_LO_bounce(*cold[1:4], m_debye_sq=cold[4])[0]
    assert a > 0 and b > 0 and b < a


def test_bounce_raises_when_there_is_no_barrier():
    T, m3sq, lam3, g3sq, mD2 = _matched_at(0.0656)
    with pytest.raises(ValueError, match="no barrier"):
        abelian_higgs_3d.solve_LO_bounce(m3sq, 10.0, g3sq, m_debye_sq=mD2)  # huge quartic, no barrier


def test_the_action_does_not_depend_on_the_artificial_wall():
    """The acceptance condition for using a regulator at all."""
    _, m3sq, lam3, g3sq, mD2 = _matched_at(0.0656)
    assert not abelian_higgs_3d.is_bounded_below(lam3)          # a wall is in use here
    assert not abelian_higgs_3d.bounce_is_wall_limited(m3sq, lam3, g3sq, m_debye_sq=mD2)


def test_subtraction_equals_the_cubic_part_of_the_action():
    """C must be exactly the gauge cubic already inside S3, or the NLO
    assembly double-subtracts."""
    _, m3sq, lam3, g3sq, mD2 = _matched_at(0.0656)
    _S, r, phi3, _info = abelian_higgs_3d.solve_LO_bounce(m3sq, lam3, g3sq,
                                               m_debye_sq=mD2)
    C = abelian_higgs_3d.double_counting_subtraction(r, phi3, g3sq, m_debye_sq=mD2)
    # both gauge cubics of V3_LO, transverse and temporal, the latter with its
    # Debye counterterm: C and the potential have to carry the same terms or
    # the subtraction stops cancelling what it is meant to cancel
    m_gauge_sq = g3sq * phi3**2
    cubic = -(2 * m_gauge_sq**1.5
              + ((mD2 + m_gauge_sq) ** 1.5 - mD2**1.5)) / (12 * math.pi)
    assert C == pytest.approx(float(np.trapz(4 * math.pi * r**2 * cubic, r)), rel=1e-10)

    # and it really is what V3_LO subtracts: the same integrand, read off the
    # potential rather than written out again
    from_potential = (abelian_higgs_3d.V3_LO(phi3, m3sq, lam3, g3sq, m_debye_sq=mD2)
                      - abelian_higgs_3d.V3_LO(phi3, m3sq, lam3, g3sq, m_debye_sq=mD2,
                                    n_transverse=0, n_temporal=0))
    assert C == pytest.approx(
        float(np.trapz(4 * math.pi * r**2 * from_potential, r)), rel=1e-10)


def test_model_mode_supplies_its_own_action():
    """The seam that actually reaches the pipeline.

    Overriding the rate alone left every reported observable identical to the
    4D ones, because the percolation integrands call Gamma(T, S) directly and
    never see the model. Supplying the action instead reaches all of them, and
    is exact at leading order because the historical rate expression is the
    same function of S/T that the 3D rate is of S_3/T.
    """
    pot = _pot3d()
    T_GeV = 0.0656 * TC_A
    T = T_GeV / pot.conversionFactor
    pot.config.nucleationConf.rate_mode = "S3overT_4D"
    assert pot.actionNuc(T, 123.0) == 123.0          # default is inert
    pot.config.nucleationConf.rate_mode = "model"
    supplied = pot.actionNuc(T, 123.0)
    assert supplied != 123.0 and math.isfinite(supplied)
    assert supplied > 0.0


def test_the_action_seam_reproduces_the_3d_rate_exactly():
    """bd.logGamma(T, S) at the supplied action IS the 3D LO rate.

    This is the identity the whole fix rests on, so it is asserted rather than
    argued: the two expressions differ only in whether the action arrives
    dimensionful or as S_3/T.
    """
    from transitionlistener import bubbledynamics as bd
    pot = _pot3d()
    pot.config.nucleationConf.rate_mode = "model"
    T_GeV = 0.0656 * TC_A
    T = T_GeV / pot.conversionFactor
    S = pot.actionNuc(T, 123.0)
    via_action = float(np.atleast_1d(bd.logGamma(T, S))[0])
    # the 3D LO rate, ln(Gamma/V) = 4 ln T + (3/2) ln(S_3/2 pi T) - S_3/T
    direct = 4.0 * np.log(T) + 1.5 * np.log(S / T / (2.0 * np.pi)) - S / T
    assert via_action == pytest.approx(direct, rel=1e-12)


def test_no_barrier_means_no_nucleation_not_an_exception():
    """At high temperature there is no barrier, and the pipeline scans through it.

    For a radiative transition the barrier only opens once the running has
    driven the 3D quartic negative, deep in the supercooled regime. The
    pipeline starts above that, so `logGammaNuc` meets the no-barrier case on
    every run and must report a vanishing rate rather than raise. Running the
    3D mode through the real pipeline for the first time is what exposed this;
    before that the mode had only ever been called at temperatures where a
    barrier exists.
    """
    pot = _pot3d()
    pot.config.nucleationConf.rate_mode = "model"
    # above the window: with the MS-bar input quartic the 3D barrier opens
    # between 1.2 and 1.3 Tc (the 4D value); reading the Coleman-Weinberg
    # quartic as MS-bar, it would open only below ~0.25 Tc
    T_hot_GeV = 2.0 * TC_A
    _m3sq, lam3, _g3sq, _mD2 = pot.matched_3d_parameters(T_hot_GeV)
    assert lam3 > 0.0, "expected no barrier here; the test has drifted"
    T = T_hot_GeV / pot.conversionFactor
    S = pot.actionNuc(T, 123.0)
    # large and *finite*: an infinite action becomes nan in bd.logGamma, and
    # the percolation ODE starts inside this no-barrier region, above the
    # temperature where the 3D barrier opens
    import TL_conformal_u1_fermions_3d as mod3d
    assert math.isfinite(S)
    assert S / T == pytest.approx(one_field_rate_3d.ACTION_CAP)
    lg = float(np.atleast_1d(bd.logGamma(T, S))[0])
    assert math.isfinite(lg) and lg < -900
    # and an infinite action would NOT do: it becomes nan downstream
    assert np.isnan(float(np.atleast_1d(bd.logGamma(T, float("inf")))[0]))


def test_the_rate_is_finite_where_a_barrier_does_exist():
    """Otherwise the test above could pass on a mode that never works."""
    pot = _pot3d()
    pot.config.nucleationConf.rate_mode = "model"
    T = 0.0656 * TC_A / pot.conversionFactor
    S = pot.actionNuc(T, 123.0)
    assert math.isfinite(S)
    assert math.isfinite(pot.logGammaNuc(T, S))


# ------------------------------------------- the configuration-file switch
def test_the_nucleation_block_selects_the_rate_mode():
    """rate_mode is reachable from a configuration file, not only from Python.

    This was in the original brief and was missing: the setting lived in the
    Configuration object while the YAML loader had no section for it, so the
    only way to select the 3D mode was to wrap the factory by hand.
    """
    import tempfile
    import textwrap

    from transitionlistener.interface.config_loader import ScanConfig
    from transitionlistener.interface import state

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    body = textwrap.dedent(f"""
        Modelfile:
          {root}/models/TL_conformal_u1_fermions_3d.py
        Potential:
          specific_potential
        Scan:
          SinglePoint
        Parameters:
          g: 0.677
          y: 0.224
          v_GeV: 0.173
        Nucleation:
          rate_mode: model
        timeout: -1
        format: txt
        output_path: /tmp/tl_nucleation_block_test
        description: test
        plot_description: test
    """)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        fh.write(body)
        path = fh.name
    try:
        conf = ScanConfig(path)
        assert conf.nucleation == {"rate_mode": "model"}
        pot = state.potential({"g": 0.677, "y": 0.224, "v_GeV": 0.173},
                              verbose=False)
        assert pot.config.nucleationConf.rate_mode == "model"
    finally:
        os.unlink(path)


def test_a_misspelt_nucleation_key_is_refused():
    """Silently ignoring it would leave the run quietly in the default mode.

    Which is precisely the failure that let the 3D mode look wired for as long
    as it did, so this one is worth refusing loudly.
    """
    import tempfile
    import textwrap

    from transitionlistener.interface.config_loader import ScanConfig

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    body = textwrap.dedent(f"""
        Modelfile:
          {root}/models/TL_conformal_u1_fermions_3d.py
        Potential:
          specific_potential
        Scan:
          SinglePoint
        Parameters:
          g: 0.677
          y: 0.224
          v_GeV: 0.173
        Nucleation:
          rate_mod: model
        timeout: -1
        format: txt
        output_path: /tmp/tl_nucleation_block_test
        description: test
        plot_description: test
    """)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        fh.write(body)
        path = fh.name
    try:
        with pytest.raises(Exception, match="Unknown key"):
            ScanConfig(path)
    finally:
        os.unlink(path)


def test_the_two_action_branches_meet_continuously_at_the_window_edge():
    """No step where the barrier disappears -- that is what stalls the ODE.

    Above the temperature where the 3D barrier closes there is no bounce, and
    just below it the true action is already far above the cap. Capping both
    sides makes them agree there, so the action the percolation solver
    integrates has no jump. Reporting a constant sentinel instead produced a
    step of order 1e5 across one grid step, which it could not step through.
    """
    import TL_conformal_u1_fermions_3d as mod3d

    pot = _pot3d()
    pot.config.nucleationConf.rate_mode = "model"
    fs = np.linspace(1.40, 0.90, 61)          # straddles the window edge
    vals = []
    for f in fs:
        T = f * TC_A / pot.conversionFactor
        vals.append(pot.actionNuc(T, 123.0) / T)
    vals = np.array(vals)
    assert np.all(np.isfinite(vals))
    assert np.all(vals <= one_field_rate_3d.ACTION_CAP * (1 + 1e-9))
    # the capped plateau really is reached from both sides
    assert vals[0] == pytest.approx(one_field_rate_3d.ACTION_CAP)
    assert vals[-1] < one_field_rate_3d.ACTION_CAP
    # and there is no jump anywhere along the way
    assert np.max(np.abs(np.diff(vals))) < 0.25 * one_field_rate_3d.ACTION_CAP


def test_the_cap_does_not_affect_the_physical_region():
    """Doubling the cap must not move any action the pipeline can act on.

    Nucleation needs S_3/T of order 100-250; the cap sits far above that, so it
    is a numerical regularisation with no physical content. If this fails, the
    cap is low enough to be doing physics.
    """
    import TL_conformal_u1_fermions_3d as mod3d

    pot = _pot3d()
    pot.config.nucleationConf.rate_mode = "model"
    fs = np.linspace(0.12, 0.01, 20)
    before = [pot.actionNuc(f * TC_A / pot.conversionFactor, 1.0) for f in fs]
    old_cap = one_field_rate_3d.ACTION_CAP
    try:
        one_field_rate_3d.ACTION_CAP = 2.0 * old_cap
        after = [pot.actionNuc(f * TC_A / pot.conversionFactor, 1.0) for f in fs]
    finally:
        one_field_rate_3d.ACTION_CAP = old_cap
    for a, b in zip(before, after):
        assert a == pytest.approx(b, rel=1e-12)


@pytest.mark.parametrize("modname", ["bubbledynamics", "bubbledynamics_fixedstep"])
def test_both_action_paths_apply_the_seam(modname, monkeypatch):
    """bubbledynamics and its fixed-step twin each have their own calcAction.

    Patching only one would leave the fixed-step path silently running 4D
    physics while the configuration said otherwise.

    This checks *behaviour*, not source text. An earlier version of this test
    asserted that "actionNuc" appeared in the function's source, which passed
    while the observable behaviour was in doubt -- source inspection cannot
    tell whether a function is ever called, nor whether its result survives.
    The bounce solver is stubbed out so the wiring is tested without the cost
    of a real bounce.
    """
    import importlib
    import types

    mod = importlib.import_module(f"transitionlistener.{modname}")

    def fake_bounce(T, V, gradV, outdict, tdict, images, verbose, cf, **kw):
        outdict[T] = {"action": 111.0}
        return outdict

    monkeypatch.setattr(mod, "bounceActionOverImages", fake_bounce)
    monkeypatch.setattr(mod, "findLocalMinimum", lambda *a, **k: np.zeros(1),
                        raising=False)
    if modname == "bubbledynamics":
        import transitionlistener.phases as phases
        monkeypatch.setattr(phases, "findLocalMinimum", lambda *a, **k: np.zeros(1))

    class StubPot:
        conversionFactor = 1.0
        Vtot = gradV = d2V = staticmethod(lambda *a, **k: 0.0)
        config = types.SimpleNamespace(
            tracingConf=types.SimpleNamespace(tunneling_params={}))
        def actionNuc(self, T, action, instanton=None):
            return 2.0 * action                 # unmistakable marker

    class StubPhase:
        def valAt(self, T):
            return np.zeros(1)

    out = {}
    got = mod.calcAction(StubPot(), 5.0, StubPhase(), StubPhase(), out)
    assert got == pytest.approx(222.0), (
        f"{modname}.calcAction did not apply actionNuc")
    assert out[5.0]["action"] == pytest.approx(222.0)
    assert out[5.0]["action_4d"] == pytest.approx(111.0), (
        f"{modname}.calcAction did not keep the solver's own action")


def test_the_saturation_is_exact_where_it_matters_and_smooth_where_it_is_not():
    """A hard min() has a kink; the adaptive solver reports kinks as jitter.

    Nucleation needs S_3/T of order 100-250, so the saturation must not touch
    that region at all, and must be differentiable where it engages. (It turned
    out not to be what stalls the adaptive ODE on point A -- that fails with or
    without the kink -- but an artificial derivative discontinuity in the action
    is worth not having regardless.)
    """
    import TL_conformal_u1_fermions_3d as mod3d

    for s in (100.0, 150.0, 300.0, 500.0, one_field_rate_3d.ACTION_SOFT_FROM - 1.0):
        assert one_field_rate_3d.saturate_action(s) == s, "the physical region must be untouched"

    # bounded above, and monotone (tanh reaches 1.0 in floating point, so the
    # cap is attained rather than approached -- which is fine, the rate is zero
    # either way)
    big = [one_field_rate_3d.saturate_action(s) for s in (700.0, 1e3, 1e4, 1e6)]
    assert all(b <= one_field_rate_3d.ACTION_CAP for b in big)
    assert big == sorted(big)
    assert big[0] < one_field_rate_3d.ACTION_CAP            # not saturated immediately

    # differentiable across the join
    h, s0 = 1e-4, one_field_rate_3d.ACTION_SOFT_FROM
    left = (one_field_rate_3d.saturate_action(s0) - one_field_rate_3d.saturate_action(s0 - h)) / h
    right = (one_field_rate_3d.saturate_action(s0 + h) - one_field_rate_3d.saturate_action(s0)) / h
    assert left == pytest.approx(right, rel=1e-3)
    assert left == pytest.approx(1.0, rel=1e-3)


def test_the_rate_seam_and_the_action_seam_agree_with_determinants():
    """With the next-to-leading-order correction on, logGammaNuc must be the rate of the
    action actionNuc supplies (which carries the correction), not the leading-order rate."""
    pot = _pot3d()
    conf = pot.config.nucleationConf
    conf.rate_mode, conf.include_determinants = "model", False
    T = 0.0656 * TC_A / pot.conversionFactor
    S_lo = pot.actionNuc(T, 123.0)
    conf.include_determinants = True
    pot.nlo_correction = abelian_higgs_3d.NLOCorrection([1e-4, 1e-3, 1e-2, 1.0], [2.5, 2.5, 2.5, 2.5], [100.0] * 4)
    S = pot.actionNuc(T, 123.0)
    assert S / T == pytest.approx(S_lo / T + 2.5, abs=1e-9)
    via_action = float(np.atleast_1d(bd.logGamma(T, S))[0])
    assert pot.logGammaNuc(T, S) == pytest.approx(via_action, rel=1e-12)
    assert pot.logGammaNuc(T, S) < float(np.atleast_1d(bd.logGamma(T, S_lo))[0]) - 2.0
