"""Matching for the U(1) + charge-one scalar model without fermions (DRalgo run 6).

Checked against the appendix of arXiv:2511.02910, which printed DRalgo output
for the same model; against scale independence; and against the fermion
model, from which it may differ only where the fermions' gauge charge enters.
"""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
import dr_conformal_u1_no_fermions_matching as nf  # noqa: E402
import dr_conformal_u1_fermions_matching as dark  # noqa: E402

GE, LOGA = 0.5772156649015329, 0.2487544770337842
T = 3.0e-3            # GeV
G, LAM = 0.7, 2.0e-3


def test_the_gauge_coupling_runs_as_published():
    """1/alpha(mu) = 1/alpha0 - ln(mu/mu0)/(6 pi), arXiv:2511.02910 eq. (B.3);
    half the fermion model's slope."""
    for mu in (1e-3, 0.3, 10.0):
        g_mu = nf.rg_run(G, 0.0, 1.0, mu, rtol=1e-12, atol=1e-15)[0]
        alpha0 = G**2 / (4 * math.pi)
        expected = 1.0 / (1.0 / alpha0 - math.log(mu) / (6 * math.pi))
        assert g_mu**2 / (4 * math.pi) == pytest.approx(expected, rel=1e-8)
    assert nf.beta_g2(G) == pytest.approx(0.5 * dark.beta_g2(G, 0.0, LAM))


def test_couplings_and_masses_agree_with_the_2511_appendix():
    """Eqs. (A.2), (A.5)/(A.6) and (A.7) of arXiv:2511.02910, typed from the
    paper. Their L has +m^2|Phi|^2 - m_soft^2|Phi_soft|^2, so msq = -m^2 and
    msq3d = -m_soft^2. At mu3 = mubar the logarithm of (A.6) vanishes, so its
    tree-level couplings inside it do not matter."""
    for mult in (0.5, 1.0, 2.0):
        mu = mult * 2 * math.pi * T
        lb = math.log(mu**2 * math.exp(2 * GE) / (16 * math.pi**2 * T**2))
        assert nf.Lb(mu, T) == pytest.approx(lb, rel=1e-12, abs=1e-14)
        pi2 = math.pi**2
        assert nf.g3_sq_NLO(G, T, mu) == pytest.approx(G**2 * T - G**4 * lb * T / (48 * pi2), rel=1e-12)
        assert nf.lambda3_NLO(G, LAM, T, mu) == pytest.approx(
            LAM * T + T / (16 * pi2) * (G**4 * (2 - 3 * lb) + 6 * G**2 * LAM * lb - 10 * LAM**2 * lb),
            rel=1e-12)
        assert nf.lambda_VLL(G, T) == pytest.approx(G**4 * T / pi2, rel=1e-12)
        assert nf.lambda_VL(G, LAM, T, mu) == pytest.approx(
            2 * G**2 * T + G**2 * T / (24 * pi2) * (24 * LAM - G**2 * (lb - 4)), rel=1e-12)
        cplus = 0.5 * (GE - lb - 12 * LOGA)
        c3 = cplus                                  # mu3 = mubar
        m_soft_sq = (-T**2 * (3 * G**2 + 4 * LAM) / 12
                     + lb * (13 * G**4 - 24 * G**2 * LAM + 40 * LAM**2) * T**2 / (192 * pi2)
                     + c3 * (3 * G**4 - 4 * G**2 * LAM + 4 * LAM**2) * T**2 / (8 * pi2)
                     + (G**4 - 3 * G**2 * LAM) * T**2 / (72 * pi2))
        assert nf.scalar_thermal_mass_sq_NLO(G, LAM, T, mu, mu3=mu) == pytest.approx(-m_soft_sq, rel=1e-12)
        debye = G**2 * T**2 / 3 + G**2 / (144 * pi2) * (12 * LAM * T**2 + G**2 * (7 - lb) * T**2)
        assert nf.debye_mass_sq_NLO(G, LAM, T, mu) == pytest.approx(debye, rel=1e-12)


def _running(mu):
    g, lam, _ = nf.rg_run(G, LAM, 1.0, mu)
    return g, lam


@pytest.mark.parametrize("name,bound", [("g3", 1e-3), ("lam3", 1e-2)])
def test_the_one_loop_couplings_do_not_depend_on_the_matching_scale(name, bound):
    """Across a factor of four in mubar the NLO coupling moves by a small
    fraction of what its explicit logarithm alone moves it by (measured:
    5e-4 for g3^2, 5e-3 for lam3)."""
    def value(mu, run=True):
        g, lam = _running(mu) if run else (G, LAM)
        return {"g3": nf.g3_sq_NLO(g, T, mu), "lam3": nf.lambda3_NLO(g, lam, T, mu)}[name]
    lo, hi = math.pi * T, 4 * math.pi * T
    explicit = abs(value(hi, run=False) - value(lo, run=False))
    actual = abs(value(hi) - value(lo))
    assert actual < bound * explicit


def test_lambda_VL_is_scale_independent_up_to_its_lam_term():
    """lambda_VL keeps a residual: the running of lam (beta ~ 3 g^4/8pi^2) inside
    its g^2 lam T/pi^2 term, which only a two-loop matching would cancel. What
    is left must be that term and nothing else."""
    lo, hi = math.pi * T, 4 * math.pi * T
    (g1, l1), (g2, l2) = _running(lo), _running(hi)
    residual = nf.lambda_VL(g2, l2, T, hi) - nf.lambda_VL(g1, l1, T, lo)
    lam_term = g2**2 * T * l2 / math.pi**2 - g1**2 * T * l1 / math.pi**2
    explicit = abs(nf.lambda_VL(G, LAM, T, hi) - nf.lambda_VL(G, LAM, T, lo))
    assert residual == pytest.approx(lam_term, rel=0.1)
    assert abs(residual - lam_term) < 0.02 * explicit


def test_the_nlo_scalar_mass_cancels_the_matching_scale():
    lo, hi = math.pi * T, 4 * math.pi * T
    def nlo(mu):
        g, lam = _running(mu)
        return nf.scalar_thermal_mass_sq_NLO(g, lam, T, mu, mu3=g**2 * T)
    def lo_(mu):
        g, lam = _running(mu)
        return nf.scalar_thermal_mass_sq_LO(g, lam, T)
    assert abs(nlo(hi) - nlo(lo)) < 0.1 * abs(lo_(hi) - lo_(lo))


def test_the_fermions_enter_only_through_their_gauge_charge():
    """Against the fermion model at y = 0: the Debye mass 1/3 against 5/12,
    lam3 identical (its fermion terms carry y), and the scalar mass differs by
    a lam-independent g^4 T^2 term once the mu3 logarithm is switched off."""
    mu = 2 * math.pi * T
    assert nf.debye_mass_sq_LO(G, T) / dark.debye_mass_sq_LO(G, T) == pytest.approx(0.8)
    assert nf.lambda3_NLO(G, LAM, T, mu) == pytest.approx(dark.lambda3_NLO(G, 0.0, LAM, T, mu), rel=1e-12)
    diffs = [dark.scalar_thermal_mass_sq_NLO(G, 0.0, lam, T, mu, mu3=mu)
             - nf.scalar_thermal_mass_sq_NLO(G, lam, T, mu, mu3=mu) for lam in (0.0, 0.01, 0.05)]
    assert diffs[1] == pytest.approx(diffs[0], rel=1e-9, abs=1e-18)
    assert diffs[2] == pytest.approx(diffs[0], rel=1e-9, abs=1e-18)
    assert abs(diffs[0]) > 1e-4 * G**4 * T**2


def test_the_2511_input_condition_is_our_input_scheme():
    """lam(mu0) = 0 puts the minimum at g v = e^{1/6} mu0. Our Coleman-Weinberg
    conversion at mu = g v (gauge loops only) gives g^4/16pi^2 there, and
    running it down to mu0 must return lam = 0 up to higher orders."""
    for g in (0.5, 0.7, 0.9):
        mu0 = 1.0
        v = nf.vev_from_zero_quartic(g, mu0)
        assert g * v == pytest.approx(math.exp(1 / 6) * mu0)
        lam_gv = dark.msbar_quartic_from_cw(g, 0.0, 11 * g**4 / (16 * math.pi**2), v, g * v,
                                            scalar_loops=False)
        assert lam_gv == pytest.approx(g**4 / (16 * math.pi**2), rel=1e-12)
        lam_mu0 = nf.rg_run(g, lam_gv, g * v, mu0)[1]
        assert abs(lam_mu0) < 0.03 * g**4 / (16 * math.pi**2)


def test_matched_parameters_carry_the_same_keys_as_the_fermion_model():
    a = nf.matched_soft_parameters(G, 0.0, 1.0, T)
    b = dark.matched_soft_parameters(G, 0.01, 0.01, 1.0, T)
    assert set(b) <= set(a)
    assert a["mD2"] == pytest.approx(nf.debye_mass_sq_NLO(a["g4"], a["lam4"], T, a["mu4"]))
