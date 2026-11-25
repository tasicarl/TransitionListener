"""The next-to-leading-order hard -> soft matching, and why it is needed.

The leading-order thermal masses carry no 4D matching scale at all, so nothing
in a leading-order prediction cancels it. Pairing them with the one-loop
quartic -- which this package did at first -- mixes orders, and the
renormalisation-scale band that results measures the inconsistency rather than
the truncation.
"""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
import dr_conformal_u1_fermions_matching as dr  # noqa: E402

K, G0, Y0, LAM0, V, TC = 7.0553, 0.677, 0.224, 0.0146238, 0.173, 0.034772
T = 0.0192 * TC
BAND = (0.5, 1.0, 2.0)


def _at(mult):
    mu = mult * K * T
    g, y, lam, _ = dr.rg_run(G0, Y0, LAM0, V, mu)
    return g, y, lam, mu


def test_the_debye_log_collection_matches_DRalgos_raw_form():
    """DRalgo writes the logarithms out rather than as Lb, and they collect.

        -40 gE g^2T^2 + 20 g^2T^2 Log[4 pi^2 T] - 20 g^2T^2 Log[mu]
            - 20 g^2T^2 Log[mu/T]
      = -40 g^2 T^2 Log[mu e^{gamma_E}/(2 pi T)]

    Transcribing the collected form is what the code does, so the collection
    itself is what needs checking.
    """
    for mult in BAND:
        g, y, lam, mu = _at(mult)
        raw = (-40.0 * dr.EULER_GAMMA * g**2 * T**2
               + 20.0 * g**2 * T**2 * math.log(4.0 * math.pi**2 * T)
               - 20.0 * g**2 * T**2 * math.log(mu)
               - 20.0 * g**2 * T**2 * math.log(mu / T))
        collected = -40.0 * g**2 * T**2 * math.log(
            mu * math.exp(dr.EULER_GAMMA) / (2.0 * math.pi * T))
        assert collected == pytest.approx(raw, rel=1e-12), f"at mu x{mult}"


def test_both_NLO_masses_are_corrections_not_replacements():
    """Each carries the LO relation inside it and adds a 1/(16 pi^2) piece."""
    for mult in BAND:
        g, y, lam, mu = _at(mult)
        g3sq = dr.g3_sq_LO(g, T)
        m_lo = dr.scalar_thermal_mass_sq_LO(g, y, lam, T)
        m_nlo = dr.scalar_thermal_mass_sq_NLO(g, y, lam, T, mu, mu3=g3sq)
        d_lo = dr.debye_mass_sq_LO(g, T)
        d_nlo = dr.debye_mass_sq_NLO(g, y, lam, T, mu)
        assert m_nlo != m_lo and d_nlo != d_lo
        assert abs(m_nlo - m_lo) < 0.3 * abs(m_lo)
        assert abs(d_nlo - d_lo) < 0.05 * abs(d_lo)


def test_the_NLO_masses_narrow_the_scale_band_which_is_the_whole_point():
    """The leading-order scalar mass has no mubar in it, so its band is pure
    coupling running with nothing to cancel it. The NLO relation supplies the
    Lb, Lf and Log[mu3/mubar] terms that do cancel it."""
    lo, nlo = [], []
    for mult in BAND:
        g, y, lam, mu = _at(mult)
        lo.append(dr.scalar_thermal_mass_sq_LO(g, y, lam, T))
        nlo.append(dr.scalar_thermal_mass_sq_NLO(g, y, lam, T, mu,
                                                 mu3=dr.g3_sq_LO(g, T)))
    spread_lo = (max(lo) - min(lo)) / lo[1]
    spread_nlo = (max(nlo) - min(nlo)) / nlo[1]
    assert spread_lo > 0.02
    assert spread_nlo < 0.005
    assert spread_nlo < spread_lo / 5.0


def test_the_debye_mass_band_narrows_too():
    lo, nlo = [], []
    for mult in BAND:
        g, y, lam, mu = _at(mult)
        lo.append(dr.debye_mass_sq_LO(g, T))
        nlo.append(dr.debye_mass_sq_NLO(g, y, lam, T, mu))
    spread_lo = (max(lo) - min(lo)) / lo[1]
    spread_nlo = (max(nlo) - min(nlo)) / nlo[1]
    assert spread_nlo < spread_lo


def test_the_action_band_narrows_and_the_action_moves():
    """The consequence, on the real bounce at benchmark A.

    Both halves matter: the band narrows by a factor of about three, and the
    action itself falls by about 13%, so this is not a refinement that can be
    folded into an uncertainty -- it moves the central value.
    """
    from transitionlistener.eft_rate import abelian_higgs_3d

    lo, nlo = [], []
    for mult in BAND:
        g, y, lam, mu = _at(mult)
        lam3 = dr.lambda3_NLO(g, y, lam, T, mu)
        g3sq = dr.g3_sq_LO(g, T)
        lo.append(abelian_higgs_3d.solve_LO_bounce(
            dr.scalar_thermal_mass_sq_LO(g, y, lam, T), lam3, g3sq,
            m_debye_sq=dr.debye_mass_sq_LO(g, T))[0])
        nlo.append(abelian_higgs_3d.solve_LO_bounce(
            dr.scalar_thermal_mass_sq_NLO(g, y, lam, T, mu, mu3=g3sq), lam3,
            g3sq, m_debye_sq=dr.debye_mass_sq_NLO(g, y, lam, T, mu))[0])
    spread_lo = (max(lo) - min(lo)) / lo[1]
    spread_nlo = (max(nlo) - min(nlo)) / nlo[1]
    assert spread_lo == pytest.approx(0.035, abs=0.01)
    assert spread_nlo == pytest.approx(0.013, abs=0.008)
    assert spread_nlo < spread_lo / 2.0
    assert nlo[1] / lo[1] == pytest.approx(0.87, abs=0.03)


# ------------------------------------------------------------- the switch
def test_the_model_honours_matching_order_and_defaults_to_NLO():
    import TL_conformal_u1_fermions_3d as m3d
    from transitionlistener.config import Configuration

    assert Configuration().nucleationConf.matching_order == "NLO"

    pot = m3d.specific_potential({"g": G0, "y": Y0, "v_GeV": V}, verbose=False)
    pot.config.nucleationConf.matching_order = "LO"
    lo = pot.matched_3d_parameters(T)
    pot.config.nucleationConf.matching_order = "NLO"
    nlo = pot.matched_3d_parameters(T)

    assert lo[0] != nlo[0]          # scalar mass moves
    assert lo[3] != nlo[3]          # Debye mass moves
    assert lo[1] == nlo[1]          # quartic is already NLO in both
    # g3^2 follows the order: g^2 T at LO, times 1 - g^2 (L_b + L_f)/(48 pi^2) at NLO, with L_f = L_b + 4 ln 2
    mu = pot.config.nucleationConf.mu4_over_T * T
    L_b = 2.0 * math.log(mu * math.exp(0.5772156649015329) / (4.0 * math.pi * T))
    g_sq = lo[2] / T
    assert nlo[2] / lo[2] == pytest.approx(1.0 - g_sq * (2.0 * L_b + 4.0 * math.log(2.0)) / (48.0 * math.pi**2), rel=1e-12)

    pot.config.nucleationConf.matching_order = "bogus"
    with pytest.raises(ValueError, match="must be 'LO' or 'NLO'"):
        pot.matched_3d_parameters(T)


def test_the_offline_tables_and_the_runtime_path_use_the_same_matching():
    """The model and the analysis scripts both call matched_soft_parameters.

    Asserted with array equality, not approx: if they ever diverge, an NLO
    table built offline would be attached to an action computed with different
    3D parameters, and nothing downstream would notice.
    """
    import TL_conformal_u1_fermions_3d as m3d

    pot = m3d.specific_potential({"g": G0, "y": Y0, "v_GeV": V}, verbose=False)
    pot.config.nucleationConf.input_scheme, pot.config.nucleationConf.mu3_choice = "CW", "g3sq"
    for order in ("LO", "NLO"):
        for mult in BAND:
            pot.config.nucleationConf.matching_order = order
            pot.config.nucleationConf.mu4_over_T = mult * K
            runtime = pot.matched_3d_parameters(T)
            offline = dr.matched_soft_parameters(G0, Y0, pot.l, V, T,
                                                 mu4_over_T=mult * K,
                                                 matching_order=order)
            assert runtime == (offline["m3sq"], offline["lam3"],
                               offline["g3sq"], offline["mD2"])
