"""The NLO exponent must not depend on the units the 3D parameters are given in.

Every fluctuation channel is a ratio of determinants except the radial one,
whose three translational zero modes are removed: what is left has mass
dimension three. In GeV units that channel would put -3 ln(T/GeV) into the exponent,
and the zero-mode Jacobian it contains would be supplied a second time by
TransitionListener's own prefactor.
"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
import dr_conformal_u1_fermions_matching as dr  # noqa: E402

pytest.importorskip("BubbleDet")

G, Y, V, T = 0.7, 0.01, 0.173, 5e-3


def _assembled(scale):
    """The same physics with every mass multiplied by ``scale`` (1e3: MeV)."""
    import TL_conformal_u1_fermions as m4d
    from transitionlistener.eft_rate import abelian_higgs_3d

    lam = m4d.specific_potential({"g": G, "y": Y, "v_GeV": V}, verbose=False).l
    p = dr.matched_soft_parameters(G, Y, lam, V, T)
    s = scale
    m3, l3, g3, mD = p["m3sq"] * s * s, p["lam3"] * s, p["g3sq"] * s, p["mD2"] * s * s
    lvl, lvll, mu3 = p["lambda_vl"] * s, p["lambda_vll"] * s, p["mu3"] * s
    S, r, phi3, _ = abelian_higgs_3d.solve_LO_bounce(m3, l3, g3, m_debye_sq=mD)
    return abelian_higgs_3d.nlo_assembly(
        r, phi3, m3, l3, g3, mD, S, 0.0, T=T * s,
        V3_nlo=lambda q: dr.V3_soft_two_loop(q, g3, mD, lvl, lvll, mu3))


def test_the_exponent_is_the_same_in_GeV_and_in_MeV():
    gev, mev = _assembled(1.0), _assembled(1e3)
    # the raw radial channel does carry the units ...
    assert mev["radial_input_units"] - gev["radial_input_units"] == pytest.approx(
        -3.0 * math.log(1e3), abs=1e-3)
    # ... and nothing reported downstream does
    for key in ("radial", "goldstone_xi1", "gauge_single_channel", "temporal",
                "double_counting_C", "nlo_potential", "total"):
        assert mev[key] == pytest.approx(gev[key], abs=1e-3), key


def test_the_seam_correction_makes_TLs_formula_the_NLO_rate():
    """4 ln T + 1.5 ln(S/2 pi) - S with S = S3 + dS equals 4 ln T - E."""
    from transitionlistener.eft_rate import abelian_higgs_3d

    for S3, E in ((151.06, 194.33), (95.2, 110.0), (300.0, 280.0)):
        dS = abelian_higgs_3d.action_seam_correction(S3, E - S3)
        S = S3 + dS
        assert 1.5 * math.log(S / (2 * math.pi)) - S == pytest.approx(-E, abs=1e-9)
    # and the Jacobian is not counted twice: dS exceeds E - S3 by one Jacobian
    dS = abelian_higgs_3d.action_seam_correction(151.06, 43.27)
    assert dS - 43.27 == pytest.approx(1.5 * math.log((151.06 + dS) / (2 * math.pi)), rel=1e-12)


def test_the_dynamical_prefactor_is_unit_free_and_of_the_expected_size():
    """ln(A_dyn/T) with A_dyn = sqrt|lambda_-|/2 pi: the same in GeV and MeV, and
    a few units negative, since |lambda_-| is of order the scalar mass squared."""
    gev, mev = _assembled(1.0), _assembled(1e3)
    assert mev["negative_eigenvalue"] == pytest.approx(1e6 * gev["negative_eigenvalue"], rel=1e-4)
    assert mev["log_A_dyn_over_T"] == pytest.approx(gev["log_A_dyn_over_T"], abs=1e-4)
    assert gev["negative_eigenvalue"] < 0.0
    assert -8.0 < gev["log_A_dyn_over_T"] < 0.0


def test_the_seam_correction_includes_the_dynamical_prefactor():
    """4 ln T + 1.5 ln(S/2 pi) - S equals ln(A_dyn/T) + 4 ln T - E."""
    from transitionlistener.eft_rate import abelian_higgs_3d

    S3, E, la = 151.06, 194.33, -3.2
    dS = abelian_higgs_3d.action_seam_correction(S3, E - S3, log_A_dyn_over_T=la)
    S = S3 + dS
    assert 1.5 * math.log(S / (2 * math.pi)) - S == pytest.approx(la - E, abs=1e-9)


def test_the_domain_of_a_3d_bounce_does_not_depend_on_the_last_shooting_trial():
    """The trial ending the bisection is an undershoot or an overshoot at random and stops anywhere
    from 15.5 to 19 inverse masses. The massless gauge channel depends on where, so the profile
    continues by its exact tail to a fixed fraction of phi(0) and falls monotonically all the way."""
    from transitionlistener.eft_rate import abelian_higgs_3d
    import TL_conformal_u1_fermions as m4d

    lam = m4d.specific_potential({"g": G, "y": Y, "v_GeV": V}, verbose=False).l
    p = dr.matched_soft_parameters(G, Y, lam, V, T)
    ends = []
    # three bisections ending on different trials, and the same physics in other units
    for s, xtol in ((1.0, 1e-11), (1.0, 3e-11), (1e3, 1e-11)):
        m3, l3, g3, mD = p["m3sq"] * s * s, p["lam3"] * s, p["g3sq"] * s, p["mD2"] * s * s
        _, r, phi3, info = abelian_higgs_3d.solve_LO_bounce(m3, l3, g3, m_debye_sq=mD, xtol=xtol)
        assert phi3[-1] / phi3[0] == pytest.approx(abelian_higgs_3d.TAIL_END_FRACTION, rel=1e-9)
        assert np.all(np.diff(phi3) < 0.0)
        assert np.all(info["dphi3"][1:] < 0.0)
        ends.append(r[-1] * math.sqrt(m3))
    assert max(ends) - min(ends) < 1e-3 * ends[0]
