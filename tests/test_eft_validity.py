"""Tests for the 3D-EFT validity monitor and the generated DRalgo matching.

Two jobs. First, pin the monitor's logic, in particular that a check which
cannot be evaluated is never counted as a pass. Second, pin the DRalgo
expressions in models/dr_conformal_u1_fermions_matching.py against values verified by
hand, so that a regenerated matching with different expressions fails here.
"""
import math
import os
import sys

import pytest

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "models")
)

from transitionlistener.eft_rate import validity as ev
from transitionlistener.config import Configuration

import dr_conformal_u1_fermions_matching as dr

# a benchmark point of the conformal U(1) with fermions
G, Y = 0.9, 0.3
LAM = 0.04600619792147168


# ----------------------------------------------------------------- monitor
def test_all_checks_pass_gives_ok_and_trusted():
    r = ev.evaluate({"hard": 0.3, "debye": 0.13, "dim6": 0.008})
    assert r.verdict == "ok"
    assert r.trusted


def test_warning_band_is_reported_but_not_fatal():
    r = ev.evaluate({"debye": 0.7})
    assert r.verdict == "warn"
    assert not r.trusted          # a warning is not a trusted result
    assert not r.missing          # ... but nothing is missing either
    assert any("debye" in m for m in r.messages)


def test_failure_beats_warning():
    r = ev.evaluate({"debye": 0.7, "dim6": 0.9})
    assert r.verdict == "fail"


def test_the_hard_scale_tests_are_pass_fail_with_no_warning_band():
    """They ask whether the reduction is defined, so one edge, at one."""
    for name in ("hard", "hard_fermion"):
        assert ev.evaluate({name: 0.99}).verdict == "ok"
        assert ev.evaluate({name: 1.01}).verdict == "fail"


def test_missing_check_is_not_a_pass():
    """The point of the monitor: silence must not read as success."""
    r = ev.evaluate({"hard": 0.2, "dim6": None})
    assert r.verdict == "ok"          # nothing evaluated has failed ...
    assert r.missing == ["dim6"]
    assert not r.trusted              # ... but the result is still not trusted


def test_soft_separation_counts_only_for_the_ultrasoft_theory():
    """The soft theory keeps A_0, so whether A_0 could be integrated out
    does not bear on its result; it is recorded, not warned about."""
    soft = ev.evaluate({"soft_separation": 0.9}, branch="soft")
    assert soft.verdict == "ok" and soft.ratios["soft_separation"] == 0.9
    assert ev.evaluate({"soft_separation": 0.9}, branch="ultrasoft").verdict == "warn"
    assert ev.evaluate({"soft_separation": 1.2}, branch="ultrasoft").verdict == "fail"
    assert ev.evaluate({"soft_separation": 1.2}).verdict == "ok"      # default: soft


def test_expansion_3d_is_pass_fail_at_one_when_an_nlo_error_bar_is_quoted():
    """At NLO its size is the error bar; at LO there is none, so the band stays."""
    assert ev.evaluate({"expansion_3d": 0.6}, order="NLO").verdict == "ok"
    assert ev.evaluate({"expansion_3d": 1.1}, order="NLO").verdict == "fail"
    assert ev.evaluate({"expansion_3d": 0.6}, order="LO").verdict == "warn"
    assert ev.evaluate({"expansion_3d": 0.6}).verdict == "warn"       # default: LO


def test_thresholds_for_leaves_the_strict_table_untouched():
    t = ev.thresholds_for("soft", "NLO")
    assert t["soft_separation"][0] == math.inf and t["expansion_3d"][:2] == (1.0, 1.0)
    assert ev.DEFAULT_THRESHOLDS["soft_separation"][:2] == (0.5, 1.0)
    assert ev.DEFAULT_THRESHOLDS["expansion_3d"][:2] == (0.3, 1.0)
    assert ev.ULTRASOFT_SEPARATION_OK == 0.5 and ev.ULTRASOFT_SEPARATION_BAD == 1.0
    with pytest.raises(ValueError):
        ev.thresholds_for("hard", "NLO")
    with pytest.raises(ValueError):
        ev.thresholds_for("soft", "NNLO")


def test_worst_reads_the_thresholds_the_verdict_used():
    r = ev.evaluate({"soft_separation": 0.9, "debye": 0.2}, branch="soft")
    assert r.worst()[0] == "debye"


def test_unknown_check_name_raises():
    with pytest.raises(KeyError):
        ev.evaluate({"not_a_check": 1.0})


def test_worst_offender_is_relative_to_its_own_threshold():
    # dim6 warns at 0.05, hard at 1, so 0.04 in dim6 (0.8 of its band) is worse
    # than 0.2 in hard (0.2 of its band)
    r = ev.evaluate({"hard": 0.2, "dim6": 0.04})
    assert r.worst()[0] == "dim6"


def test_report_flattens_for_a_scan_row():
    r = ev.evaluate({"hard": 0.3}, context={"T": 0.135})
    row = r.as_dict()
    assert row["eft_hard"] == 0.3
    assert row["eft_verdict"] == "ok"
    assert row["eft_T"] == 0.135


# ------------------------------------ the dim-6 check and the error bars
def test_dim6_ratio_is_the_shift_over_the_action_in_magnitude():
    assert ev.dim6_ratio(3.0, 150.0) == pytest.approx(0.02)
    assert ev.dim6_ratio(-3.0, 150.0) == pytest.approx(0.02)
    assert math.isinf(ev.dim6_ratio(3.0, 0.0))


def test_dim6_action_shift_of_a_uniform_ball():
    """4 pi int r^2 (c6/8) phi^6 for phi = const on [0, R] is (pi/6) c6 phi^6 R^3."""
    import numpy as np
    from transitionlistener.eft_rate import abelian_higgs_3d
    r = np.linspace(0.0, 2.0, 4001)
    phi3 = np.full_like(r, 1.5)
    c6 = 3e-4
    assert abelian_higgs_3d.dim6_action_shift(r, phi3, c6) == pytest.approx(
        math.pi / 6.0 * c6 * 1.5**6 * 2.0**3, rel=1e-9)


def test_dim6_kinetic_action_shift_of_a_linear_ramp():
    """4 pi int r^2 (c/4) phi^2 phi'^2 for phi = a r on [0, R] is pi c a^4 R^5 / 5."""
    import numpy as np
    from transitionlistener.eft_rate import abelian_higgs_3d
    r = np.linspace(0.0, 2.0, 4001)
    a, c = 0.7, 2e-3
    assert abelian_higgs_3d.dim6_kinetic_action_shift(r, a * r, np.full_like(r, a), c) == pytest.approx(
        math.pi * c * a**4 * 2.0**5 / 5.0, rel=1e-9)


def test_bounce_returns_its_own_field_derivative():
    """info["dphi3"] is d phi3/dr of the returned profile (sign included), not the solver's reflected coordinate."""
    import numpy as np
    from transitionlistener.eft_rate import abelian_higgs_3d
    import dr_conformal_u1_no_fermions_matching as dr0
    T = 2.1e-4                                   # g = 0.5 on the line mu0 = 1 GeV, near percolation
    p = dr0.matched_soft_parameters(0.5, 0.0, 1.0, T, mu3=0.5 * T)
    _, r, phi3, info = abelian_higgs_3d.solve_LO_bounce(p["m3sq"], p["lam3"], p["g3sq"], m_debye_sq=p["mD2"])
    grad = np.gradient(phi3, r)
    assert np.max(np.abs(info["dphi3"] - grad)) < 1e-2 * np.max(np.abs(grad))


def test_dim6_coefficients_match_arxiv_2610_12284():
    """App. A of 2610.12284: alpha[phi^6] and alpha[D^2 phi^4] of the fermion-free model, in its own normalisation."""
    z = dr.ZETA3
    for g, lam, T in ((0.45, 0.0, 1e-6), (0.7, 0.01, 0.05), (0.9, -0.02, 3.0)):
        assert dr.c6_bosonic(g, lam) == pytest.approx(
            z / (480 * math.pi**4) * (15 * g**6 - 18 * g**4 * lam + 75 * g**2 * lam**2 + 100 * lam**3), rel=1e-12)
        assert dr.cD2phi4_bosonic(g, lam, T) == pytest.approx(
            z / (960 * math.pi**4 * T) * (51 * g**4 + 200 * g**2 * lam - 20 * lam**2), rel=1e-12)


def test_fermion_free_matching_returns_the_kinetic_coefficient_at_the_matching_scale():
    import dr_conformal_u1_no_fermions_matching as dr0
    p = dr0.matched_soft_parameters(0.5, 0.0, 1.0, 2.1e-4)
    assert p["c_D2phi4"] == pytest.approx(dr.cD2phi4_bosonic(p["g4"], p["lam4"], 2.1e-4), rel=1e-14)
    assert p["c_D2phi4"] > 0                      # raises the action for |lambda| << g^2 (2610.12284)


def test_nlo_error_bar_is_the_correction_times_the_3d_loop_parameter():
    u = ev.ExponentUncertainty.from_ratios(delta_S=-12.0, expansion_3d=0.4, delta_S6=0.5)
    assert u.nnlo == pytest.approx(4.8)
    assert u.dim6 == pytest.approx(0.5)
    on_tp = u.on_tperc(beta_over_H=80.0)
    assert on_tp["nnlo"] == pytest.approx(0.06)
    assert on_tp["dim6"] == pytest.approx(0.00625)
    with pytest.raises(ValueError):
        u.on_tperc(beta_over_H=0.0)


def test_the_25_percent_flag_is_kept_as_a_marker():
    assert not ev.nlo_is_large(delta_S=40.0, S3_over_T=170.0)
    assert ev.nlo_is_large(delta_S=-45.0, S3_over_T=170.0)


# ------------------------------------------------ generated DRalgo values
def test_debye_coefficient_is_five_twelfths():
    assert dr.debye_mass_sq_LO(G, 1.0) == pytest.approx(5.0 / 12.0 * G**2)


def test_scalar_thermal_mass_matches_the_4d_model_file():
    """DRalgo's LO scalar mass must equal the model file's scalar_prefactor.

    models/TL_conformal_u1_fermions.py uses  l/3 + g^2/4 + y^2/12.
    """
    expected = LAM / 3.0 + G**2 / 4.0 + Y**2 / 12.0
    assert dr.scalar_thermal_mass_sq_LO(G, Y, LAM, 1.0) == pytest.approx(expected)


def test_beta_g2_matches_the_field_content():
    """beta(g^2) = 2 b g^4/(16 pi^2) with b = 2/3 for this charge assignment."""
    b = 2.0 / 3.0 * (0.25 + 0.25) + 1.0 / 3.0 * 1.0
    assert dr.beta_g2(G, Y, LAM) == pytest.approx(2 * b * G**4 / (16 * math.pi**2))


def test_c6_bosonic_at_the_benchmark():
    assert dr.c6_bosonic(G, LAM) == pytest.approx(1.95e-4, rel=5e-3)


def test_fermionic_sextic_fraction_is_negligible_at_the_benchmark():
    assert dr.fermionic_sextic_fraction(G, Y) < 1e-3


def test_fermionic_sextic_fraction_is_order_one_near_the_lambda_edge():
    """At (g, y) = (0.5, 0.65) the fermion term is ~0.8 of the gauge term."""
    assert dr.fermionic_sextic_fraction(0.5, 0.65) == pytest.approx(0.80, rel=0.05)


def test_the_ultrasoft_relations_are_now_available_end_to_end():
    """DRalgo run 2 supplied lambda_VL, so this path no longer raises.

    Composed the way a caller would: hard matching, then A_0 integrated out.
    The detailed checks live in tests/test_ultrasoft.py; this one only pins
    that the chain runs and moves both parameters in the direction the A_0
    cubic demands, downwards.
    """
    T, mu = 3.0, 2.0 * math.pi * 3.0
    lam3 = dr.lambda3_NLO(G, Y, LAM, T, mu)
    m3sq = dr.scalar_thermal_mass_sq_LO(G, Y, LAM, T)
    mD2 = dr.debye_mass_sq_LO(G, T)
    lam_vl = dr.lambda_VL(G, Y, LAM, T, mu)

    assert dr.lambda3_US(lam3, lam_vl, mD2) < lam3
    assert dr.scalar_mass_sq_US(m3sq, lam_vl, mD2) < m3sq
    assert dr.g3_sq_US(dr.g3_sq_LO(G, T)) == dr.g3_sq_LO(G, T)


# ------------------------------------------------------------- the switch
def test_the_default_rate_is_the_4d_one():
    conf = Configuration()
    assert conf.nucleationConf.rate_mode == "S3overT_4D"


# ------------------------------------------------------- RG running (WP2)
def test_rg_running_is_reversible():
    """Running down and back up must return the starting point."""
    down = dr.rg_run(G, Y, LAM, 1.0, 0.1)
    back = dr.rg_run(down[0], down[1], down[2], 0.1, 1.0)
    assert back[0] == pytest.approx(G, rel=1e-8)
    assert back[1] == pytest.approx(Y, rel=1e-8)
    assert back[2] == pytest.approx(LAM, rel=1e-8)


def test_rg_running_is_a_no_op_at_equal_scales():
    assert dr.rg_run(G, Y, LAM, 0.5, 0.5) == (G, Y, LAM, 0.0)


def test_gauge_coupling_decreases_towards_the_infrared():
    """beta(g^2) > 0, so g is smaller at lower scales (no asymptotic freedom)."""
    g_lo = dr.rg_run(G, Y, LAM, 1.0, 0.01)[0]
    assert g_lo < G


def test_small_running_matches_the_analytic_first_order_step():
    """Over a short interval the numerical run must reproduce beta * dln(mu)."""
    mu0, mu1 = 1.0, 1.0005
    g1 = dr.rg_run(G, Y, LAM, mu0, mu1)[0]
    dlnmu = math.log(mu1 / mu0)
    expected = G + dr.beta_g2(G, Y, LAM) / (2.0 * G) * dlnmu
    assert g1 == pytest.approx(expected, rel=1e-6)


def test_running_rejects_non_positive_scales():
    with pytest.raises(ValueError):
        dr.rg_run(G, Y, LAM, 0.0, 1.0)


def test_running_raises_rather_than_returning_a_blown_up_coupling():
    """A Landau-pole run must fail loudly, not hand back a large number."""
    with pytest.raises(RuntimeError):
        dr.rg_run(3.0, Y, LAM, 1.0, 1e30)


# --------------------------------------------- wiring the ratios (WP5a)
def _setup(over_Tc=0.0656):
    """Sub-GeV point A at its published percolation temperature."""
    import numpy as np
    from transitionlistener.eft_rate import abelian_higgs_3d
    Tc = 0.034772
    T = over_Tc * Tc
    mu = 7.0553 * T
    g, y, lam, _ = dr.rg_run(0.677, 0.224, 0.0146238, 0.173, mu)
    m3sq = dr.scalar_thermal_mass_sq_LO(g, y, lam, T)
    l3 = dr.lambda3_NLO(g, y, lam, T, mu)
    g3sq = dr.g3_sq_LO(g, T)
    mD2 = dr.debye_mass_sq_LO(g, T)
    S3, r, phi3, info = abelian_higgs_3d.solve_LO_bounce(m3sq, l3, g3sq,
                                              m_debye_sq=mD2)
    return abelian_higgs_3d, T, m3sq, l3, g3sq, mD2, r, phi3, g, lam


def test_ratios_are_all_produced_and_finite():
    abelian_higgs_3d, T, m3sq, l3, g3sq, mD2, r, phi3, g, lam = _setup()
    S3 = abelian_higgs_3d.solve_LO_bounce(m3sq, l3, g3sq, m_debye_sq=mD2)[0]
    ratios = abelian_higgs_3d.validity_ratios(T, m3sq, l3, g3sq, mD2, r=r, phi3=phi3,
                                   fermion_masses_over_field=(0.15, 0.15),
                                   c6=dr.c6_bosonic(g, lam), S3_over_T=S3)
    assert set(ratios) == set(ev.DEFAULT_THRESHOLDS)
    for name in ratios:
        assert math.isfinite(ratios[name])
    assert ratios["dim6"] == pytest.approx(
        abelian_higgs_3d.dim6_action_shift(r, phi3, dr.c6_bosonic(g, lam)) / S3, rel=1e-12)


def test_dim6_and_fermions_are_missing_when_their_inputs_are_not_supplied():
    """They must be reported missing, not silently passing."""
    abelian_higgs_3d, T, m3sq, l3, g3sq, mD2, r, phi3, _g, _lam = _setup()
    ratios = abelian_higgs_3d.validity_ratios(T, m3sq, l3, g3sq, mD2, r=r, phi3=phi3)
    assert ratios["dim6"] is None and ratios["hard_fermion"] is None
    report = ev.evaluate(ratios)
    assert sorted(report.missing) == ["dim6", "hard_fermion"]
    assert not report.trusted
    # a model without fermions says so with an empty sequence, and passes
    free = abelian_higgs_3d.validity_ratios(T, m3sq, l3, g3sq, mD2, r=r, phi3=phi3,
                                 fermion_masses_over_field=())
    assert free["hard_fermion"] == 0.0


def test_bosons_are_tested_against_2piT_and_fermions_against_piT():
    import numpy as np
    abelian_higgs_3d, T, m3sq, l3, g3sq, mD2, r, phi3, _g, _lam = _setup()
    c = 0.15
    ratios = abelian_higgs_3d.validity_ratios(T, m3sq, l3, g3sq, mD2, r=r, phi3=phi3,
                                   fermion_masses_over_field=(c,))
    f = float(np.max(np.abs(phi3)))
    photon = math.sqrt(g3sq) * f
    assert ratios["hard"] >= photon / (2 * math.pi * T) * (1 - 1e-12)
    assert ratios["hard_fermion"] == pytest.approx(c * f * math.sqrt(T) / (math.pi * T), rel=1e-12)


def test_hard_is_a_maximum_along_the_bounce_not_a_single_point():
    """Evaluating at the escape point alone understates it substantially."""
    import numpy as np
    abelian_higgs_3d, T, m3sq, l3, g3sq, mD2, r, phi3, _g, _lam = _setup()
    along = abelian_higgs_3d.validity_ratios(T, m3sq, l3, g3sq, mD2, r=r, phi3=phi3)["hard"]
    esc = abelian_higgs_3d.escape_point_LO(m3sq, l3, g3sq, m_debye_sq=mD2)
    at_escape = abelian_higgs_3d.validity_ratios(
        T, m3sq, l3, g3sq, mD2,
        r=np.array([0.0]), phi3=np.array([esc]))["hard"]
    assert along > 2.0 * at_escape


def test_the_benchmark_passes_the_species_test_it_failed_against_piT():
    """Point A at its published T_perc.

    Its photon at the bubble centre is 0.68 of 2 pi T -- inside the convergence
    radius -- and its fermions 0.32 of pi T; against a single pi T the photon
    would read 1.37 and the point would fail.
    """
    import math as _m
    abelian_higgs_3d, T, m3sq, l3, g3sq, mD2, r, phi3, _g, _lam = _setup()
    mu = 7.0553 * T
    _g4, y, _l, _ = dr.rg_run(0.677, 0.224, 0.0146238, 0.173, mu)
    ratios = abelian_higgs_3d.validity_ratios(T, m3sq, l3, g3sq, mD2, r=r, phi3=phi3,
                                   fermion_masses_over_field=(abs(y) / _m.sqrt(2),) * 2)
    assert ratios["hard"] == pytest.approx(0.683, rel=5e-3)
    assert ratios["hard_fermion"] == pytest.approx(0.324, rel=5e-3)
    assert ev.evaluate({k: ratios[k] for k in ("hard", "hard_fermion")}).verdict == "ok"
    assert 2.0 * ratios["hard"] > 1.0          # the old pi T reading
